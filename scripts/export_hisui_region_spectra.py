#!/usr/bin/env python3
"""Export a HISUI L1G region as a streamed ``y,x,wave_*`` CSV.

The output schema matches the CSV files consumed by this repository.  Image
coordinates are relative to the exported region by default; the JSON sidecar
always records the source-image bounds and georeferencing needed to recover
absolute or projected coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from hisui_l1g_io import HISUIL1GReader, VNIR_BAND_STOP


DEFAULT_TILE_ROWS_PER_BLOCK = 4


def spectra_column_names(wavelengths_nm: Sequence[float]) -> list[str]:
    """Return CSV column names, rejecting collisions after 0.01 nm rounding."""

    names = [f"wave_{float(wavelength):.2f}nm" for wavelength in wavelengths_nm]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Selected wavelengths produce duplicate CSV column names after "
            f"rounding to 0.01 nm: {', '.join(duplicates)}"
        )
    return names


def select_band_indices(
    wavelengths_nm: Sequence[float],
    wave_min_nm: float | None,
    wave_max_nm: float | None,
) -> np.ndarray:
    """Select bands inclusively while retaining HISUI sensor-band order."""

    if (
        wave_min_nm is not None
        and wave_max_nm is not None
        and float(wave_min_nm) > float(wave_max_nm)
    ):
        raise ValueError("--wave-min cannot be greater than --wave-max")
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    selected = np.ones(wavelengths.shape, dtype=bool)
    if wave_min_nm is not None:
        selected &= wavelengths >= float(wave_min_nm)
    if wave_max_nm is not None:
        selected &= wavelengths <= float(wave_max_nm)
    indices = np.flatnonzero(selected).astype(np.int64)
    if indices.size == 0:
        raise ValueError(
            f"No HISUI bands fall within {wave_min_nm!r} to {wave_max_nm!r} nm"
        )
    return indices


def bounds_from_center(
    center_y: int,
    center_x: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Return exact-sized, half-open bounds centered on an integer pixel."""

    if height <= 0 or width <= 0:
        raise ValueError("--height and --width must be positive")
    y0 = int(center_y) - int(height) // 2
    x0 = int(center_x) - int(width) // 2
    return y0, y0 + int(height), x0, x0 + int(width)


def validate_bounds(
    bounds: Sequence[int], image_shape: Sequence[int]
) -> tuple[int, int, int, int]:
    """Validate half-open source-image bounds without silently clamping them."""

    if len(bounds) != 4:
        raise ValueError("Bounds must contain Y0 Y1 X0 X1")
    y0, y1, x0, x1 = (int(value) for value in bounds)
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if y0 >= y1 or x0 >= x1:
        raise ValueError(f"Empty or reversed bounds: {(y0, y1, x0, x1)}")
    if y0 < 0 or x0 < 0 or y1 > image_height or x1 > image_width:
        raise ValueError(
            f"Bounds {(y0, y1, x0, x1)} fall outside image shape "
            f"{(image_height, image_width)}"
        )
    return y0, y1, x0, x1


def _iter_row_block_bounds(
    y0: int,
    y1: int,
    tile_height: int,
    tile_rows_per_block: int,
) -> Iterator[tuple[int, int]]:
    """Yield blocks ending on tile boundaries to avoid decoding a tile twice."""

    if tile_rows_per_block < 1:
        raise ValueError("tile_rows_per_block must be positive")
    block_span = int(tile_height) * int(tile_rows_per_block)
    cursor = int(y0)
    while cursor < y1:
        next_boundary = ((cursor // block_span) + 1) * block_span
        block_y1 = min(int(y1), next_boundary)
        yield cursor, block_y1
        cursor = block_y1


def sidecar_path_for(output_csv: str | Path) -> Path:
    """Return the JSON sidecar path for an output CSV."""

    return Path(output_csv).with_suffix(".json")


def _temporary_path(directory: Path, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".hisui-export-", suffix=suffix, dir=directory, delete=False
    )
    path = Path(handle.name)
    handle.close()
    return path


def _calibration_sidecar(reader: HISUIL1GReader) -> dict[str, object]:
    metadata = reader.metadata
    required = (
        "RadianceMultiVNIR",
        "RadianceAddVNIR",
        "RadianceMultiSWIR",
        "RadianceAddSWIR",
    )
    missing = [key for key in required if not isinstance(metadata.get(key), (int, float))]
    if missing:
        raise ValueError("Missing numeric radiance metadata: " + ", ".join(missing))
    return {
        "formula": "radiance = DN * multiplier + additive",
        "vnir_band_indices_zero_based": [0, VNIR_BAND_STOP - 1],
        "swir_band_indices_zero_based": [VNIR_BAND_STOP, reader.band_count - 1],
        "radiance_multi_vnir": float(metadata["RadianceMultiVNIR"]),
        "radiance_add_vnir": float(metadata["RadianceAddVNIR"]),
        "radiance_multi_swir": float(metadata["RadianceMultiSWIR"]),
        "radiance_add_swir": float(metadata["RadianceAddSWIR"]),
        "radiance_unit": metadata.get("RadianceUnit"),
    }


def _write_sidecar(
    path: Path,
    *,
    reader: HISUIL1GReader,
    output_csv: Path,
    bounds: tuple[int, int, int, int],
    absolute_coordinates: bool,
    band_indices: np.ndarray,
    column_names: Sequence[str],
    drop_invalid: bool,
    requested_pixel_count: int,
    emitted_pixel_count: int,
    invalid_pixel_count: int,
    invalid_sample_count: int,
    invalid_sample_counts_by_dn: dict[str, int],
    tile_rows_per_block: int,
) -> None:
    y0, y1, x0, x1 = bounds
    georef = reader.georeference
    product = reader.product
    band_records = []
    for band_index, column_name in zip(band_indices.tolist(), column_names):
        band_records.append(
            {
                "source_index_zero_based": int(band_index),
                "band_number_one_based": int(reader.bands.band_number[band_index]),
                "column_name": column_name,
                "center_wavelength_nm": float(
                    reader.bands.center_wavelength_nm[band_index]
                ),
                "fwhm_nm": float(reader.bands.fwhm_nm[band_index]),
                "detector": "VNIR" if band_index < VNIR_BAND_STOP else "SWIR",
            }
        )

    sidecar = {
        "schema_version": 1,
        "product": {
            "product_id": product.product_id,
            "metadata_product_id": reader.metadata.get("ProductID"),
            "product_version": reader.metadata.get("ProductVersion"),
            "scene_center_time": reader.metadata.get("SceneCenterTime"),
        },
        "source": {
            "image_path": str(product.image_path.resolve()),
            "metadata_path": str(product.metadata_path.resolve()),
            "band_csv_path": str(product.band_csv_path.resolve()),
            "image_size_bytes": int(product.image_path.stat().st_size),
            "image_shape_y_x_band": [reader.height, reader.width, reader.band_count],
            "image_dtype": str(reader.dtype),
        },
        "export": {
            "output_csv": str(output_csv.resolve()),
            "coordinate_mode": "absolute" if absolute_coordinates else "relative",
            "bounds_half_open": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
            "relative_origin": {"y": y0, "x": x0},
            "region_shape_y_x": [y1 - y0, x1 - x0],
            "tile_rows_per_block": int(tile_rows_per_block),
        },
        "bands": band_records,
        "calibration": _calibration_sidecar(reader),
        "georeference": {
            "crs": georef.crs,
            "epsg": georef.epsg,
            "affine_gdal_order": list(georef.geotransform),
            "pixel_scale": list(georef.pixel_scale),
            "tiepoint": list(georef.tiepoint),
            "raster_type": georef.raster_type,
            "coordinate_reference": "pixel upper-left corner",
            "region_upper_left_map_xy": list(georef.map_xy(y0, x0)),
            "region_lower_right_map_xy": list(georef.map_xy(y1, x1)),
        },
        "invalid": {
            "drop_invalid": bool(drop_invalid),
            "invalid_dn_values": [int(value) for value in reader.invalid_dn_values],
            "requested_pixel_count": int(requested_pixel_count),
            "emitted_pixel_count": int(emitted_pixel_count),
            "dropped_pixel_count": int(requested_pixel_count - emitted_pixel_count),
            "invalid_pixel_count": int(invalid_pixel_count),
            "invalid_sample_count": int(invalid_sample_count),
            "invalid_sample_counts_by_dn": invalid_sample_counts_by_dn,
        },
    }
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(sidecar, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def export_hisui_region_spectra(
    input_path: str | Path,
    output_csv: str | Path,
    *,
    bounds: Sequence[int],
    absolute_coordinates: bool = False,
    wave_min_nm: float | None = None,
    wave_max_nm: float | None = None,
    drop_invalid: bool = True,
    overwrite: bool = False,
    tile_rows_per_block: int = DEFAULT_TILE_ROWS_PER_BLOCK,
) -> dict[str, object]:
    """Export one HISUI product region and return its summary dictionary."""

    output_path = Path(output_csv)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("--output-csv must have a .csv extension")
    sidecar_path = sidecar_path_for(output_path)
    existing = [path for path in (output_path, sidecar_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_temporary = _temporary_path(output_path.parent, ".csv.tmp")
    json_temporary = _temporary_path(output_path.parent, ".json.tmp")
    try:
        with HISUIL1GReader(input_path) as reader:
            y0, y1, x0, x1 = validate_bounds(bounds, reader.shape)
            band_indices = select_band_indices(
                reader.bands.center_wavelength_nm, wave_min_nm, wave_max_nm
            )
            selected_wavelengths = reader.bands.center_wavelength_nm[band_indices]
            column_names = spectra_column_names(selected_wavelengths)

            requested_pixel_count = (y1 - y0) * (x1 - x0)
            emitted_pixel_count = 0
            invalid_pixel_count = 0
            invalid_sample_count = 0
            counts_by_dn = {str(value): 0 for value in reader.invalid_dn_values}

            with csv_temporary.open(
                "w", encoding="utf-8", newline="", buffering=1024 * 1024
            ) as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["y", "x", *column_names])
                for block_y0, block_y1 in _iter_row_block_bounds(
                    y0,
                    y1,
                    reader.tile_height,
                    tile_rows_per_block,
                ):
                    raw = reader.read_region(
                        block_y0, block_y1, x0, x1, band_indices
                    )
                    invalid_samples = np.isin(raw, reader.invalid_dn_values)
                    invalid_pixels = np.any(invalid_samples, axis=2)
                    invalid_pixel_count += int(np.count_nonzero(invalid_pixels))
                    invalid_sample_count += int(np.count_nonzero(invalid_samples))
                    for value in reader.invalid_dn_values:
                        counts_by_dn[str(value)] += int(np.count_nonzero(raw == value))

                    radiance = reader.dn_to_radiance(raw, band_indices)
                    block_height, block_width, _ = radiance.shape
                    flat_radiance = radiance.reshape(-1, band_indices.size)
                    flat_invalid = invalid_pixels.reshape(-1)
                    for flat_index, spectrum in enumerate(flat_radiance):
                        if drop_invalid and bool(flat_invalid[flat_index]):
                            continue
                        local_y, local_x = divmod(flat_index, block_width)
                        source_y = block_y0 + local_y
                        source_x = x0 + local_x
                        output_y = source_y if absolute_coordinates else source_y - y0
                        output_x = source_x if absolute_coordinates else source_x - x0
                        writer.writerow([output_y, output_x, *spectrum.tolist()])
                        emitted_pixel_count += 1

            _write_sidecar(
                json_temporary,
                reader=reader,
                output_csv=output_path,
                bounds=(y0, y1, x0, x1),
                absolute_coordinates=absolute_coordinates,
                band_indices=band_indices,
                column_names=column_names,
                drop_invalid=drop_invalid,
                requested_pixel_count=requested_pixel_count,
                emitted_pixel_count=emitted_pixel_count,
                invalid_pixel_count=invalid_pixel_count,
                invalid_sample_count=invalid_sample_count,
                invalid_sample_counts_by_dn=counts_by_dn,
                tile_rows_per_block=tile_rows_per_block,
            )

        os.replace(csv_temporary, output_path)
        os.replace(json_temporary, sidecar_path)
    finally:
        csv_temporary.unlink(missing_ok=True)
        json_temporary.unlink(missing_ok=True)

    return {
        "output_csv": str(output_path),
        "sidecar_json": str(sidecar_path),
        "requested_pixels": requested_pixel_count,
        "emitted_pixels": emitted_pixel_count,
        "invalid_pixels": invalid_pixel_count,
        "selected_bands": int(band_indices.size),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a rectangular HISUI L1G region to y,x,wave_* CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="L1G product directory or principal file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Destination .csv path.")
    parser.add_argument(
        "--bounds",
        type=int,
        nargs=4,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Half-open source-image bounds.",
    )
    parser.add_argument("--center-y", type=int, help="ROI center row.")
    parser.add_argument("--center-x", type=int, help="ROI center column.")
    parser.add_argument("--height", type=int, help="Exact ROI height in pixels.")
    parser.add_argument("--width", type=int, help="Exact ROI width in pixels.")
    parser.add_argument(
        "--absolute-coordinates",
        action="store_true",
        help="Write source-image y/x instead of region-relative coordinates.",
    )
    parser.add_argument("--wave-min", type=float, help="Inclusive minimum wavelength in nm.")
    parser.add_argument("--wave-max", type=float, help="Inclusive maximum wavelength in nm.")
    invalid_group = parser.add_mutually_exclusive_group()
    invalid_group.add_argument(
        "--drop-invalid",
        dest="drop_invalid",
        action="store_true",
        default=True,
        help="Drop pixels with any fill/bad/saturated selected-band DN (default).",
    )
    invalid_group.add_argument(
        "--keep-invalid",
        dest="drop_invalid",
        action="store_false",
        help="Keep invalid pixels and write their invalid samples as NaN.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV and/or JSON sidecar.",
    )
    return parser


def _bounds_from_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[int, int, int, int]:
    center_values = (args.center_y, args.center_x, args.height, args.width)
    any_center = any(value is not None for value in center_values)
    all_center = all(value is not None for value in center_values)
    if args.bounds is not None and any_center:
        parser.error(
            "--bounds is mutually exclusive with --center-y, --center-x, --height, and --width"
        )
    if args.bounds is None and not all_center:
        parser.error(
            "provide either --bounds Y0 Y1 X0 X1 or all of --center-y, "
            "--center-x, --height, and --width"
        )
    if args.bounds is not None:
        return tuple(int(value) for value in args.bounds)
    return bounds_from_center(args.center_y, args.center_x, args.height, args.width)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bounds = _bounds_from_arguments(parser, args)
    try:
        summary = export_hisui_region_spectra(
            args.input,
            args.output_csv,
            bounds=bounds,
            absolute_coordinates=args.absolute_coordinates,
            wave_min_nm=args.wave_min,
            wave_max_nm=args.wave_max,
            drop_invalid=args.drop_invalid,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, IndexError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

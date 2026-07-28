"""Memory-bounded access to HISUI L1G GeoTIFF products.

The delivered HISUI hyperspectral image is a single-page, tiled GeoTIFF with
contiguous samples (``YXS``).  Calling :func:`tifffile.imread` on one of these
files materializes roughly 1.2 GB.  :class:`HISUIL1GReader` instead reads and
decodes only the TIFF tiles intersecting the requested region.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import tifffile


VNIR_BAND_STOP = 58
EXPECTED_HISUI_BANDS = 185


@dataclass(frozen=True)
class HISUIL1GProduct:
    """The three files required to read one HISUI L1G product."""

    product_id: str
    directory: Path
    image_path: Path
    metadata_path: Path
    band_csv_path: Path


@dataclass(frozen=True)
class BandMetadata:
    """Spectral coordinates from ``*_B.csv`` (band numbers are one-based)."""

    band_number: np.ndarray
    center_wavelength_nm: np.ndarray
    fwhm_nm: np.ndarray

    def __len__(self) -> int:
        return int(self.band_number.size)


@dataclass(frozen=True)
class GeoReference:
    """North-up GeoTIFF georeferencing information.

    ``geotransform`` follows GDAL order: origin x, pixel width, row rotation,
    origin y, column rotation, and pixel height (negative for north-up data).
    """

    geotransform: tuple[float, float, float, float, float, float]
    crs: str | None
    epsg: int | None
    pixel_scale: tuple[float, float, float]
    tiepoint: tuple[float, float, float, float, float, float]
    raster_type: int | None

    def map_xy(
        self, row: float, column: float, *, pixel_center: bool = False
    ) -> tuple[float, float]:
        """Transform a zero-based image row/column to projected coordinates.

        ``geotransform`` is normalized to the GDAL pixel-corner convention,
        including for GeoTIFF files delivered as RasterPixelIsPoint.  Therefore
        the centre of every pixel is consistently offset by half a pixel.
        """

        offset = 0.5 if pixel_center else 0.0
        col = float(column) + offset
        line = float(row) + offset
        x0, dx, rx, y0, ry, dy = self.geotransform
        return x0 + col * dx + line * rx, y0 + col * ry + line * dy


def _coerce_metadata_value(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    if re.fullmatch(r"[+-]?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(
        r"[+-]?(?:(?:\d+\.\d*)|(?:\d*\.\d+)|(?:\d+))(?:[Ee][+-]?\d+)?",
        value,
    ):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_metadata(path: str | Path) -> dict[str, Any]:
    """Parse the HISUI TXT metadata's ``key = value`` records.

    Section headings and blank lines are ignored.  The split is made only at
    the first equals sign, so quoted values may themselves contain ``=``.
    Quoted strings are unquoted and plain numeric values are converted to
    ``int`` or ``float``.  Other values, including timestamps and ``N/A``, are
    retained as strings.
    """

    metadata: dict[str, Any] = {}
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, raw_value = stripped.partition("=")
            key = key.strip()
            if not key:
                raise ValueError(f"Empty metadata key at {metadata_path}:{line_number}")
            if key in metadata:
                raise ValueError(
                    f"Duplicate metadata key {key!r} at "
                    f"{metadata_path}:{line_number}"
                )
            metadata[key] = _coerce_metadata_value(raw_value)
    return metadata


def read_band_metadata(path: str | Path) -> BandMetadata:
    """Read band number, center wavelength, and FWHM from ``*_B.csv``."""

    csv_path = Path(path)
    band_number: list[int] = []
    wavelength: list[float] = []
    fwhm: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "BandNo",
            "CenterWavelengthNanometer",
            "FullWidthAtHalfMaximumNanometer",
        }
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                band_number.append(int(row["BandNo"].strip()))
                wavelength.append(float(row["CenterWavelengthNanometer"].strip()))
                fwhm.append(float(row["FullWidthAtHalfMaximumNanometer"].strip()))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid spectral metadata at {csv_path}:{row_number}"
                ) from exc

    if not band_number:
        raise ValueError(f"No bands found in {csv_path}")
    numbers = np.asarray(band_number, dtype=np.int64)
    expected = np.arange(1, numbers.size + 1, dtype=np.int64)
    if not np.array_equal(numbers, expected):
        raise ValueError(f"BandNo must be consecutive and one-based in {csv_path}")
    return BandMetadata(
        band_number=numbers,
        center_wavelength_nm=np.asarray(wavelength, dtype=np.float64),
        fwhm_nm=np.asarray(fwhm, dtype=np.float64),
    )


def _paths_for_product_id(directory: Path, product_id: str) -> HISUIL1GProduct:
    product = HISUIL1GProduct(
        product_id=product_id,
        directory=directory,
        image_path=directory / f"{product_id}.tif",
        metadata_path=directory / f"{product_id}.txt",
        band_csv_path=directory / f"{product_id}_B.csv",
    )
    missing = [
        str(path)
        for path in (product.image_path, product.metadata_path, product.band_csv_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing HISUI L1G companion file(s): " + ", ".join(missing))
    return product


def discover_l1g_product(path: str | Path) -> HISUIL1GProduct:
    """Resolve a product directory or any of its three principal files."""

    candidate = Path(path)
    if candidate.is_file():
        if candidate.name.endswith("_B.csv"):
            product_id = candidate.name[: -len("_B.csv")]
        elif candidate.suffix.lower() in {".tif", ".txt"}:
            product_id = candidate.stem
        else:
            raise ValueError(f"Not a HISUI L1G principal file: {candidate}")
        return _paths_for_product_id(candidate.parent, product_id)

    if not candidate.is_dir():
        raise FileNotFoundError(f"HISUI path does not exist: {candidate}")
    products: list[HISUIL1GProduct] = []
    for metadata_path in sorted(candidate.glob("*.txt")):
        try:
            products.append(_paths_for_product_id(candidate, metadata_path.stem))
        except FileNotFoundError:
            continue
    if not products:
        raise FileNotFoundError(f"No complete HISUI L1G product found in {candidate}")
    if len(products) != 1:
        ids = ", ".join(product.product_id for product in products)
        raise ValueError(f"Multiple HISUI L1G products found in {candidate}: {ids}")
    return products[0]


def discover_l1g_products(root: str | Path, *, recursive: bool = True) -> list[HISUIL1GProduct]:
    """Discover all complete HISUI L1G products below ``root``."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"HISUI root does not exist: {root_path}")
    metadata_files = root_path.rglob("*.txt") if recursive else root_path.glob("*.txt")
    products: list[HISUIL1GProduct] = []
    for metadata_path in sorted(metadata_files):
        try:
            products.append(_paths_for_product_id(metadata_path.parent, metadata_path.stem))
        except FileNotFoundError:
            continue
    return products


def _geokey_values(directory: Sequence[int]) -> dict[int, int | tuple[int, int, int]]:
    if len(directory) < 4:
        raise ValueError("Malformed GeoKeyDirectoryTag")
    key_count = int(directory[3])
    if len(directory) < 4 + 4 * key_count:
        raise ValueError("Truncated GeoKeyDirectoryTag")
    values: dict[int, int | tuple[int, int, int]] = {}
    for index in range(key_count):
        key, location, count, value = (
            int(item) for item in directory[4 + 4 * index : 8 + 4 * index]
        )
        values[key] = value if location == 0 and count == 1 else (location, count, value)
    return values


def georeference_from_page(page: tifffile.TiffPage) -> GeoReference:
    """Read a north-up affine transform and EPSG code from a TIFF page."""

    scale_tag = page.tags.get(33550)  # ModelPixelScaleTag
    tiepoint_tag = page.tags.get(33922)  # ModelTiepointTag
    if scale_tag is None or tiepoint_tag is None:
        raise ValueError("GeoTIFF is missing ModelPixelScaleTag or ModelTiepointTag")
    scale_values = tuple(float(value) for value in scale_tag.value)
    tie_values = tuple(float(value) for value in tiepoint_tag.value)
    if len(scale_values) != 3 or len(tie_values) < 6:
        raise ValueError("Unsupported GeoTIFF pixel-scale or tiepoint tag")
    scale = (scale_values[0], scale_values[1], scale_values[2])
    tiepoint = (
        tie_values[0],
        tie_values[1],
        tie_values[2],
        tie_values[3],
        tie_values[4],
        tie_values[5],
    )
    epsg: int | None = None
    raster_type: int | None = None
    directory_tag = page.tags.get(34735)  # GeoKeyDirectoryTag
    if directory_tag is not None:
        keys = _geokey_values(directory_tag.value)
        raster_value = keys.get(1025)  # GTRasterTypeGeoKey
        if isinstance(raster_value, int):
            raster_type = raster_value
        projected = keys.get(3072)  # ProjectedCSTypeGeoKey
        geographic = keys.get(2048)  # GeographicTypeGeoKey
        if isinstance(projected, int) and projected not in {0, 32767}:
            epsg = projected
        elif isinstance(geographic, int) and geographic not in {0, 32767}:
            epsg = geographic

    raster_i, raster_j, _, map_x, map_y, _ = tiepoint
    origin_x = map_x - raster_i * scale[0]
    origin_y = map_y + raster_j * scale[1]
    if raster_type == 2:  # RasterPixelIsPoint: tiepoint is the sample centre.
        origin_x -= 0.5 * scale[0]
        origin_y += 0.5 * scale[1]

    return GeoReference(
        geotransform=(origin_x, scale[0], 0.0, origin_y, 0.0, -scale[1]),
        crs=f"EPSG:{epsg}" if epsg is not None else None,
        epsg=epsg,
        pixel_scale=scale,
        tiepoint=tiepoint,
        raster_type=raster_type,
    )


def read_georeference(path: str | Path) -> GeoReference:
    """Open only the TIFF header and return its georeferencing metadata."""

    with tifffile.TiffFile(str(path)) as tif:
        if len(tif.pages) != 1:
            raise ValueError("Expected a single-page HISUI L1G TIFF")
        return georeference_from_page(tif.pages[0])


def read_tiled_any_mask(
    path: str | Path,
    bands: Sequence[int],
    *,
    bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Read selected bands from a tiled YXS TIFF and OR them into a 2-D mask.

    The helper is intended for the band-resolved HISUI ``QA_DM`` and ``QA_IM``
    products.  It decodes only tiles intersecting ``bounds`` and never
    materializes the full 185-band QA cube.
    """

    selected_bands = np.asarray(list(bands), dtype=np.int64)
    if selected_bands.ndim != 1 or selected_bands.size == 0:
        raise ValueError("At least one QA band is required")
    with tifffile.TiffFile(str(path)) as tif:
        if len(tif.pages) != 1:
            raise ValueError("Expected a single-page QA TIFF")
        page = tif.pages[0]
        if not page.is_tiled or int(page.planarconfig) != 1:
            raise ValueError("Expected a contiguous tiled QA TIFF")
        if tif.series[0].axes != "YXS":
            raise ValueError(f"Expected QA TIFF axes YXS, found {tif.series[0].axes}")
        height = int(page.imagelength)
        width = int(page.imagewidth)
        band_count = int(page.samplesperpixel)
        if np.any((selected_bands < 0) | (selected_bands >= band_count)):
            raise IndexError(f"QA band index outside 0:{band_count}")
        if bounds is None:
            y0, y1, x0, x1 = 0, height, 0, width
        else:
            y0, y1, x0, x1 = (int(value) for value in bounds)
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise IndexError(f"QA region {(y0, y1, x0, x1)} outside {(height, width)}")

        tile_height = int(page.tilelength)
        tile_width = int(page.tilewidth)
        tile_columns = math.ceil(width / tile_width)
        output = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for tile_row in range(y0 // tile_height, (y1 - 1) // tile_height + 1):
            for tile_column in range(x0 // tile_width, (x1 - 1) // tile_width + 1):
                segment_index = tile_row * tile_columns + tile_column
                offset = int(page.dataoffsets[segment_index])
                byte_count = int(page.databytecounts[segment_index])
                tif.filehandle.seek(offset)
                decoded, index, _ = page.decode(tif.filehandle.read(byte_count), segment_index)
                if decoded is None:
                    continue
                tile = np.asarray(decoded)
                while tile.ndim > 3 and tile.shape[0] == 1:
                    tile = tile[0]
                if tile.ndim != 3 or tile.shape[-1] != band_count:
                    raise ValueError(f"Unexpected decoded QA tile shape: {tile.shape}")
                tile_y, tile_x = int(index[2]), int(index[3])
                source_y0 = max(y0, tile_y)
                source_y1 = min(y1, tile_y + tile.shape[0], height)
                source_x0 = max(x0, tile_x)
                source_x1 = min(x1, tile_x + tile.shape[1], width)
                if source_y0 >= source_y1 or source_x0 >= source_x1:
                    continue
                affected = np.any(
                    tile[
                        source_y0 - tile_y : source_y1 - tile_y,
                        source_x0 - tile_x : source_x1 - tile_x,
                        :,
                    ][..., selected_bands],
                    axis=-1,
                )
                output[
                    source_y0 - y0 : source_y1 - y0,
                    source_x0 - x0 : source_x1 - x0,
                ] = affected
        return output


class HISUIL1GReader:
    """Tile-wise reader for one HISUI L1G hyperspectral image."""

    def __init__(
        self,
        product: HISUIL1GProduct | str | Path,
        *,
        extra_invalid_dn_values: Iterable[int] = (),
        mask_metadata_quality_dns: bool = True,
    ) -> None:
        self.product = (
            product if isinstance(product, HISUIL1GProduct) else discover_l1g_product(product)
        )
        self.metadata: Mapping[str, Any] = parse_metadata(self.product.metadata_path)
        self.bands = read_band_metadata(self.product.band_csv_path)
        self._tif = tifffile.TiffFile(str(self.product.image_path))
        try:
            self._initialize_page()
        except Exception:
            self._tif.close()
            raise

        invalid = {0}
        invalid.update(int(value) for value in extra_invalid_dn_values)
        if mask_metadata_quality_dns:
            for key in ("BadPixelDN", "SaturatedPixelDN"):
                value = self.metadata.get(key)
                if isinstance(value, (int, float)) and float(value).is_integer():
                    invalid.add(int(value))
        self.invalid_dn_values = tuple(sorted(invalid))

    def _initialize_page(self) -> None:
        if len(self._tif.pages) != 1:
            raise ValueError("Expected a single-page HISUI L1G TIFF")
        self._page = self._tif.pages[0]
        if not self._page.is_tiled:
            raise ValueError("Expected a tiled HISUI L1G TIFF")
        if int(self._page.compression) != 1:
            raise ValueError("Only uncompressed HISUI L1G TIFFs are supported")
        if int(self._page.planarconfig) != 1:
            raise ValueError("Expected contiguous samples (PLANARCONFIG.CONTIG)")
        axes = self._tif.series[0].axes
        if axes != "YXS":
            raise ValueError(f"Expected TIFF axes YXS, found {axes}")

        self.height = int(self._page.imagelength)
        self.width = int(self._page.imagewidth)
        self.band_count = int(self._page.samplesperpixel)
        self.shape = (self.height, self.width, self.band_count)
        self.dtype = self._page.dtype
        self.tile_height = int(self._page.tilelength)
        self.tile_width = int(self._page.tilewidth)
        self.tile_rows = math.ceil(self.height / self.tile_height)
        self.tile_columns = math.ceil(self.width / self.tile_width)
        expected_segments = self.tile_rows * self.tile_columns
        if len(self._page.dataoffsets) != expected_segments:
            raise ValueError(
                "Unsupported TIFF segment layout: expected one contiguous segment per tile"
            )
        if len(self.bands) != self.band_count:
            raise ValueError(
                f"Band CSV has {len(self.bands)} rows but TIFF has "
                f"{self.band_count} samples"
            )
        self.georeference = georeference_from_page(self._page)

    def close(self) -> None:
        self._tif.close()

    def __enter__(self) -> "HISUIL1GReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _normalize_bands(self, bands: int | slice | Sequence[int] | None) -> np.ndarray:
        if bands is None:
            indices = np.arange(self.band_count, dtype=np.int64)
        elif isinstance(bands, slice):
            indices = np.arange(self.band_count, dtype=np.int64)[bands]
        elif np.isscalar(bands):
            indices = np.asarray([int(bands)], dtype=np.int64)
        else:
            indices = np.asarray(list(bands), dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("At least one band index is required")
        indices = indices.copy()
        indices[indices < 0] += self.band_count
        if np.any((indices < 0) | (indices >= self.band_count)):
            raise IndexError(f"Band index outside 0:{self.band_count}")
        return indices

    def _decode_tile(self, tile_row: int, tile_column: int) -> tuple[np.ndarray, int, int]:
        segment_index = tile_row * self.tile_columns + tile_column
        offset = int(self._page.dataoffsets[segment_index])
        byte_count = int(self._page.databytecounts[segment_index])
        filehandle = self._tif.filehandle
        filehandle.seek(offset)
        encoded = filehandle.read(byte_count)
        decoded, index, _ = self._page.decode(encoded, segment_index)
        tile_y = int(index[2])
        tile_x = int(index[3])
        if decoded is None:
            tile = np.zeros(
                (self.tile_height, self.tile_width, self.band_count), dtype=self.dtype
            )
        else:
            tile = np.asarray(decoded)
            while tile.ndim > 3 and tile.shape[0] == 1:
                tile = tile[0]
            if tile.ndim == 2 and self.band_count == 1:
                tile = tile[..., np.newaxis]
            if tile.ndim != 3 or tile.shape[-1] != self.band_count:
                raise ValueError(f"Unexpected decoded tile shape: {tile.shape}")
        return tile, tile_y, tile_x

    def read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        bands: int | slice | Sequence[int] | None = None,
        *,
        as_radiance: bool = False,
    ) -> np.ndarray:
        """Read ``[y0:y1, x0:x1, bands]`` without decoding other tiles.

        A three-dimensional array is always returned, including for a single
        requested band.  Raw reads retain the TIFF integer dtype.  Radiance
        reads return ``float64`` and replace fill/bad/saturated DNs with NaN.
        """

        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        if not (0 <= y0 < y1 <= self.height and 0 <= x0 < x1 <= self.width):
            raise IndexError(
                f"Region {(y0, y1, x0, x1)} outside image "
                f"shape {(self.height, self.width)}"
            )
        band_indices = self._normalize_bands(bands)
        output = np.empty((y1 - y0, x1 - x0, band_indices.size), dtype=self.dtype)

        first_tile_row = y0 // self.tile_height
        last_tile_row = (y1 - 1) // self.tile_height
        first_tile_column = x0 // self.tile_width
        last_tile_column = (x1 - 1) // self.tile_width
        for tile_row in range(first_tile_row, last_tile_row + 1):
            for tile_column in range(first_tile_column, last_tile_column + 1):
                tile, tile_y, tile_x = self._decode_tile(tile_row, tile_column)
                source_y0 = max(y0, tile_y)
                source_y1 = min(y1, tile_y + tile.shape[0], self.height)
                source_x0 = max(x0, tile_x)
                source_x1 = min(x1, tile_x + tile.shape[1], self.width)
                if source_y0 >= source_y1 or source_x0 >= source_x1:
                    continue
                output[
                    source_y0 - y0 : source_y1 - y0,
                    source_x0 - x0 : source_x1 - x0,
                    :,
                ] = tile[
                    source_y0 - tile_y : source_y1 - tile_y,
                    source_x0 - tile_x : source_x1 - tile_x,
                    :,
                ][..., band_indices]

        if as_radiance:
            return self.dn_to_radiance(output, band_indices)
        return output

    def _radiance_coefficients(self, band_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        required = (
            "RadianceMultiVNIR",
            "RadianceAddVNIR",
            "RadianceMultiSWIR",
            "RadianceAddSWIR",
        )
        missing = [key for key in required if not isinstance(self.metadata.get(key), (int, float))]
        if missing:
            raise ValueError("Missing numeric radiance metadata: " + ", ".join(missing))
        vnir = band_indices < VNIR_BAND_STOP
        multiplier = np.where(
            vnir,
            float(self.metadata["RadianceMultiVNIR"]),
            float(self.metadata["RadianceMultiSWIR"]),
        )
        additive = np.where(
            vnir,
            float(self.metadata["RadianceAddVNIR"]),
            float(self.metadata["RadianceAddSWIR"]),
        )
        return multiplier, additive

    def dn_to_radiance(self, dn: np.ndarray, bands: int | slice | Sequence[int]) -> np.ndarray:
        """Convert a DN cube to radiance using the L1G TXT calibration fields."""

        band_indices = self._normalize_bands(bands)
        values = np.asarray(dn)
        if values.ndim != 3 or values.shape[-1] != band_indices.size:
            raise ValueError(
                "DN array must be YXS and its last dimension must match bands"
            )
        multiplier, additive = self._radiance_coefficients(band_indices)
        invalid = np.isin(values, self.invalid_dn_values)
        radiance = values.astype(np.float64) * multiplier + additive
        radiance[invalid] = np.nan
        return radiance

    def iter_row_blocks(
        self,
        bands: int | slice | Sequence[int] | None = None,
        *,
        x0: int = 0,
        x1: int | None = None,
        y0: int = 0,
        y1: int | None = None,
        tile_rows_per_block: int = 1,
        as_radiance: bool = False,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield tile-aligned row blocks as ``(block_y0, block_y1, array)``.

        ``y0`` must be tile-aligned.  ``y1`` must be tile-aligned unless it is
        the final image row, allowing the last partial TIFF tile to be emitted.
        """

        x1 = self.width if x1 is None else int(x1)
        y1 = self.height if y1 is None else int(y1)
        y0 = int(y0)
        x0 = int(x0)
        if tile_rows_per_block < 1:
            raise ValueError("tile_rows_per_block must be positive")
        if y0 % self.tile_height != 0:
            raise ValueError("y0 must be aligned to a TIFF tile-row boundary")
        if y1 != self.height and y1 % self.tile_height != 0:
            raise ValueError("y1 must be tile-aligned or equal to image height")
        block_height = int(tile_rows_per_block) * self.tile_height
        for block_y0 in range(y0, y1, block_height):
            block_y1 = min(block_y0 + block_height, y1)
            yield (
                block_y0,
                block_y1,
                self.read_region(
                    block_y0,
                    block_y1,
                    x0,
                    x1,
                    bands,
                    as_radiance=as_radiance,
                ),
            )

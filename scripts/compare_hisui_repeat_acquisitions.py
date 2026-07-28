#!/usr/bin/env python3
"""Compare cloud-aware methane score maps from two HISUI acquisitions.

The input score maps must have been written by ``screen_hisui_l1g_scenes.py``
with ``--save-score-maps``.  The two maps are cropped to their exact common
projected grid using the normalized GDAL pixel-corner georeferencing returned
by :mod:`hisui_l1g_io`.  High-score overlap is evaluated only where both
acquisitions are clear and have finite scores.

This is a repeatability and artifact diagnostic.  A high score at the same
projected pixel on different dates can indicate a stationary surface feature,
geometric residual, or detector artifact; it is not confirmation of a methane
plume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np

from hisui_l1g_io import (
    GeoReference,
    discover_l1g_product,
    parse_metadata,
    read_georeference,
)


ANALYSIS_VERSION = "2026-07-28-v2"

# Parameters that can change the stored clear mask or dual-local-z score map.
# Reporting-only component/site thresholds are intentionally excluded.
MAP_CONFIG_KEYS = (
    "weak_min_nm",
    "weak_max_nm",
    "strong_min_nm",
    "strong_max_nm",
    "uas_alpha_min",
    "uas_alpha_max",
    "continuum_degree",
    "sample_size",
    "block_rows",
    "nuisance_folds",
    "spatial_block_size",
    "local_tile_size",
    "local_tile_halo",
    "covariance_shrinkage",
    "covariance_ridge",
    "amplitude_cap",
    "cloud_dilation_pixels",
    "cloud_profile",
    "cloud_thresholds",
    "local_z_sigma",
    "enable_directional_destriping",
    "thin_stripe_slope",
    "thin_stripe_bin_width",
    "broad_stripe_slope",
    "broad_stripe_bin_width",
)


@dataclass(frozen=True)
class GridAlignment:
    """Integer-pixel overlap between two co-registered projected grids."""

    first_bounds: tuple[int, int, int, int]
    second_bounds: tuple[int, int, int, int]
    shape: tuple[int, int]
    second_origin_offset_in_first_col_row: tuple[int, int]
    projected_edge_corners_m: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    maximum_pixel_center_residual_m: float


@dataclass(frozen=True)
class ScoreMaps:
    path: Path
    summary_path: Path
    product_id: str
    acquisition_utc: object
    quality_class: object
    analysis_config: dict[str, object]
    target_provenance: dict[str, object]
    valid: np.ndarray
    dual_local_z: np.ndarray


def _linear_transform(georef: GeoReference) -> np.ndarray:
    _, dx, row_rotation, _, column_rotation, dy = georef.geotransform
    return np.asarray(
        [[dx, row_rotation], [column_rotation, dy]], dtype=np.float64
    )


def _validate_shape(shape: Sequence[int], label: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{label} shape must contain exactly two dimensions")
    height, width = (int(value) for value in shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} shape must be positive")
    return height, width


def common_aligned_overlap(
    first_georef: GeoReference,
    first_shape: Sequence[int],
    second_georef: GeoReference,
    second_shape: Sequence[int],
    *,
    grid_tolerance_pixels: float = 1e-6,
) -> GridAlignment | None:
    """Return the exact common projected pixel grid, or ``None`` if disjoint.

    Both geotransforms are already normalized to the GDAL pixel-corner
    convention by :func:`hisui_l1g_io.read_georeference`.  Integer edge offsets
    therefore imply coincident pixel centres as well, including when the source
    GeoTIFFs declare ``RasterPixelIsPoint``.
    """

    first_height, first_width = _validate_shape(first_shape, "first")
    second_height, second_width = _validate_shape(second_shape, "second")
    if first_georef.epsg is None or second_georef.epsg is None:
        raise ValueError("Both products must provide a projected EPSG code")
    if first_georef.epsg != second_georef.epsg:
        raise ValueError("The two products use different projected CRSs")
    if grid_tolerance_pixels <= 0:
        raise ValueError("grid tolerance must be positive")

    first_linear = _linear_transform(first_georef)
    second_linear = _linear_transform(second_georef)
    if not np.allclose(first_linear, second_linear, rtol=0.0, atol=1e-9):
        raise ValueError("The products do not share the same pixel scale/rotation")
    if abs(float(np.linalg.det(first_linear))) <= 1e-12:
        raise ValueError("The projected pixel transform is singular")

    first_origin = np.asarray(
        [first_georef.geotransform[0], first_georef.geotransform[3]], dtype=float
    )
    second_origin = np.asarray(
        [second_georef.geotransform[0], second_georef.geotransform[3]], dtype=float
    )
    offset_col_row = np.linalg.solve(first_linear, second_origin - first_origin)
    rounded_col_row = np.rint(offset_col_row).astype(np.int64)
    if not np.allclose(
        offset_col_row,
        rounded_col_row,
        rtol=0.0,
        atol=grid_tolerance_pixels,
    ):
        raise ValueError(
            "The projected grids are not aligned at pixel boundaries; "
            "resampling would be required"
        )
    second_col_offset, second_row_offset = (
        int(rounded_col_row[0]),
        int(rounded_col_row[1]),
    )

    first_y0 = max(0, second_row_offset)
    first_y1 = min(first_height, second_row_offset + second_height)
    first_x0 = max(0, second_col_offset)
    first_x1 = min(first_width, second_col_offset + second_width)
    if first_y0 >= first_y1 or first_x0 >= first_x1:
        return None

    second_y0 = first_y0 - second_row_offset
    second_y1 = first_y1 - second_row_offset
    second_x0 = first_x0 - second_col_offset
    second_x1 = first_x1 - second_col_offset
    first_bounds = (first_y0, first_y1, first_x0, first_x1)
    second_bounds = (second_y0, second_y1, second_x0, second_x1)
    shape = (first_y1 - first_y0, first_x1 - first_x0)
    if shape != (second_y1 - second_y0, second_x1 - second_x0):
        raise RuntimeError("Aligned overlap shapes differ")

    edge_indices = (
        (first_y0, first_x0),
        (first_y0, first_x1),
        (first_y1, first_x1),
        (first_y1, first_x0),
    )
    projected_corners = tuple(
        first_georef.map_xy(row, column, pixel_center=False)
        for row, column in edge_indices
    )
    center_pairs = (
        ((first_y0, first_x0), (second_y0, second_x0)),
        ((first_y1 - 1, first_x1 - 1), (second_y1 - 1, second_x1 - 1)),
    )
    residuals = []
    for first_index, second_index in center_pairs:
        first_xy = np.asarray(
            first_georef.map_xy(*first_index, pixel_center=True), dtype=float
        )
        second_xy = np.asarray(
            second_georef.map_xy(*second_index, pixel_center=True), dtype=float
        )
        residuals.append(float(np.linalg.norm(first_xy - second_xy)))
    maximum_residual = max(residuals)
    pixel_scale_m = max(float(np.linalg.norm(first_linear[:, 0])), 1.0)
    if maximum_residual > grid_tolerance_pixels * pixel_scale_m:
        raise ValueError("Aligned pixel centres differ in projected coordinates")

    return GridAlignment(
        first_bounds=first_bounds,
        second_bounds=second_bounds,
        shape=shape,
        second_origin_offset_in_first_col_row=(
            second_col_offset,
            second_row_offset,
        ),
        projected_edge_corners_m=projected_corners,
        maximum_pixel_center_residual_m=maximum_residual,
    )


def resolve_score_maps_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "score_maps.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"Score-map archive not found: {candidate}")
    return candidate


def load_score_maps(path: str | Path) -> ScoreMaps:
    archive_path = resolve_score_maps_path(path)
    summary_path = archive_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "A screen_hisui_l1g_scenes.py summary.json is required beside "
            f"the score archive: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid source summary JSON: {summary_path}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Source summary must be a JSON object: {summary_path}")
    product_id = summary.get("product_id")
    analysis_config = summary.get("analysis_config")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError(f"Source summary lacks a product_id: {summary_path}")
    if not isinstance(analysis_config, dict):
        raise ValueError(f"Source summary lacks analysis_config: {summary_path}")
    with np.load(archive_path, allow_pickle=False) as stored:
        required = {"valid", "dual_local_z"}
        missing = sorted(required - set(stored.files))
        if missing:
            raise ValueError(
                f"Missing arrays in {archive_path}: {', '.join(missing)}"
            )
        raw_valid = np.asarray(stored["valid"])
        dual_local_z = np.asarray(stored["dual_local_z"], dtype=np.float64)
    if raw_valid.dtype == np.bool_:
        valid = raw_valid.copy()
    elif np.issubdtype(raw_valid.dtype, np.number):
        if not np.all(np.isfinite(raw_valid)) or not np.all(
            (raw_valid == 0) | (raw_valid == 1)
        ):
            raise ValueError("valid must be Boolean or contain only finite 0/1 values")
        valid = raw_valid.astype(bool)
    else:
        raise ValueError("valid must be Boolean or a finite binary numeric array")
    if valid.ndim != 2 or dual_local_z.ndim != 2:
        raise ValueError("valid and dual_local_z must both be two-dimensional")
    if valid.shape != dual_local_z.shape:
        raise ValueError("valid and dual_local_z shapes differ")
    if valid.size == 0 or min(valid.shape) <= 0:
        raise ValueError("valid and dual_local_z arrays must not be empty")
    target_provenance = build_target_provenance(
        summary, summary_path=summary_path
    )
    return ScoreMaps(
        path=archive_path,
        summary_path=summary_path,
        product_id=product_id,
        acquisition_utc=summary.get("acquisition_utc"),
        quality_class=summary.get("quality_class"),
        analysis_config=analysis_config,
        target_provenance=target_provenance,
        valid=valid,
        dual_local_z=dual_local_z,
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float_list(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result


def build_target_provenance(
    summary: dict[str, object], *, summary_path: Path
) -> dict[str, object]:
    """Hash the source LUT and all stored spectral inputs that define the target."""

    model = summary.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Source summary lacks model provenance: {summary_path}")
    raw_lut_path = model.get("modtran_csv")
    if not isinstance(raw_lut_path, str) or not raw_lut_path.strip():
        raise ValueError(f"Source summary lacks model.modtran_csv: {summary_path}")
    lut_path = Path(raw_lut_path).expanduser()
    if not lut_path.is_absolute():
        lut_path = summary_path.parent / lut_path
    lut_path = lut_path.resolve()
    if not lut_path.is_file():
        raise FileNotFoundError(
            f"Source MODTRAN LUT is unavailable for provenance verification: {lut_path}"
        )
    wavelengths = _finite_float_list(
        summary.get("selected_wavelengths_nm"),
        label=f"{summary_path} selected_wavelengths_nm",
    )
    fwhm = _finite_float_list(
        summary.get("selected_fwhm_nm"),
        label=f"{summary_path} selected_fwhm_nm",
    )
    if len(wavelengths) != len(fwhm):
        raise ValueError(
            f"Selected wavelength and FWHM lengths differ in {summary_path}"
        )
    if any(value <= 0 for value in fwhm):
        raise ValueError(f"Selected FWHM values must be positive in {summary_path}")
    analysis_config = summary.get("analysis_config")
    assert isinstance(analysis_config, dict)
    lut_sha256 = _sha256_file(lut_path)
    spectral_grid = {
        "selected_wavelengths_nm": wavelengths,
        "selected_fwhm_nm": fwhm,
    }
    target_inputs = {
        "modtran_sha256": lut_sha256,
        **spectral_grid,
        "uas_alpha_min": analysis_config.get("uas_alpha_min"),
        "uas_alpha_max": analysis_config.get("uas_alpha_max"),
        "continuum_degree": analysis_config.get("continuum_degree"),
    }
    return {
        "modtran_path": str(lut_path),
        "modtran_size_bytes": int(lut_path.stat().st_size),
        "modtran_sha256": lut_sha256,
        "spectral_grid_sha256": _canonical_hash(spectral_grid),
        "target_inputs_sha256": _canonical_hash(target_inputs),
    }


def map_analysis_signature(config: dict[str, object]) -> dict[str, object]:
    """Return the map-affecting configuration and its stable SHA-256 digest."""

    missing = [key for key in MAP_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(
            "Source analysis_config is missing map parameters: " + ", ".join(missing)
        )
    selected = {key: config[key] for key in MAP_CONFIG_KEYS}
    return {"parameters": selected, "sha256": _canonical_hash(selected)}


def validate_pair_provenance(
    first_scores: ScoreMaps,
    second_scores: ScoreMaps,
    *,
    first_product_id: str,
    second_product_id: str,
    first_acquisition_utc: object,
    second_acquisition_utc: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate product identity, dates, and map-affecting analysis settings."""

    for label, scores, expected_id, expected_date in (
        (
            "First",
            first_scores,
            first_product_id,
            first_acquisition_utc,
        ),
        (
            "Second",
            second_scores,
            second_product_id,
            second_acquisition_utc,
        ),
    ):
        if scores.product_id != expected_id:
            raise ValueError(
                f"{label} score-map product_id {scores.product_id!r} does not "
                f"match product {expected_id!r}"
            )
        if (
            scores.acquisition_utc is not None
            and expected_date is not None
            and str(scores.acquisition_utc) != str(expected_date)
        ):
            raise ValueError(
                f"{label} score-map acquisition date does not match product metadata"
            )
    first_signature = map_analysis_signature(first_scores.analysis_config)
    second_signature = map_analysis_signature(second_scores.analysis_config)
    if first_signature["sha256"] != second_signature["sha256"]:
        differing = [
            key
            for key in MAP_CONFIG_KEYS
            if first_signature["parameters"][key]
            != second_signature["parameters"][key]
        ]
        raise ValueError(
            "The score maps use different map-affecting analysis settings: "
            + ", ".join(differing)
        )
    first_target = first_scores.target_provenance
    second_target = second_scores.target_provenance
    if first_target.get("target_inputs_sha256") != second_target.get(
        "target_inputs_sha256"
    ):
        differing = [
            key
            for key in (
                "modtran_sha256",
                "spectral_grid_sha256",
                "target_inputs_sha256",
            )
            if first_target.get(key) != second_target.get(key)
        ]
        raise ValueError(
            "The score maps use different MODTRAN/target provenance: "
            + ", ".join(differing)
        )
    target_validation = {
        "status": "matched_current_modtran_file_and_stored_spectral_target_inputs",
        "first": first_target,
        "second": second_target,
        "matched_target_inputs_sha256": first_target.get("target_inputs_sha256"),
        "limitation": (
            "The source scene summaries do not contain a generation-time LUT hash. "
            "This check hashes the currently referenced MODTRAN file and combines it "
            "with the spectral target inputs stored in each summary; it cannot prove "
            "that the LUT file had identical contents when each map was generated."
        ),
    }
    return first_signature, second_signature, target_validation


def _bounds_slice(bounds: tuple[int, int, int, int]) -> tuple[slice, slice]:
    y0, y1, x0, x1 = bounds
    return slice(y0, y1), slice(x0, x1)


def _nullable_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or second.size < 2:
        return None
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def repeat_metrics(
    first_dual_z: np.ndarray,
    second_dual_z: np.ndarray,
    first_clear: np.ndarray,
    second_clear: np.ndarray,
    *,
    threshold: float = 3.0,
) -> dict[str, object]:
    """Compute cloud-aware paired-map and fixed-location tail metrics."""

    arrays = [
        np.asarray(first_dual_z),
        np.asarray(second_dual_z),
        np.asarray(first_clear),
        np.asarray(second_clear),
    ]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("All aligned score and clear-mask arrays must be 2-D")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Aligned score and clear-mask shapes differ")
    if arrays[0].size == 0 or min(arrays[0].shape) <= 0:
        raise ValueError("Aligned score and clear-mask arrays must not be empty")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    first_clear_bool = arrays[2].astype(bool, copy=False)
    second_clear_bool = arrays[3].astype(bool, copy=False)
    common_clear = first_clear_bool & second_clear_bool
    clear_union = first_clear_bool | second_clear_bool
    finite_pair = np.isfinite(arrays[0]) & np.isfinite(arrays[1])
    paired = common_clear & finite_pair
    first_values = np.asarray(arrays[0][paired], dtype=np.float64)
    second_values = np.asarray(arrays[1][paired], dtype=np.float64)
    first_tail = paired & (arrays[0] >= threshold)
    second_tail = paired & (arrays[1] >= threshold)
    tail_overlap = first_tail & second_tail
    tail_union = first_tail | second_tail

    paired_count = int(paired.sum())
    if paired_count:
        bias = float(np.mean(second_values - first_values))
        mae = float(np.mean(np.abs(second_values - first_values)))
    else:
        bias = None
        mae = None
    clear_union_count = int(clear_union.sum())
    tail_union_count = int(tail_union.sum())
    return {
        "overlap_grid_pixels": int(arrays[0].size),
        "clear_masks": {
            "first_clear_pixels": int(first_clear_bool.sum()),
            "second_clear_pixels": int(second_clear_bool.sum()),
            "common_clear_pixels": int(common_clear.sum()),
            "common_clear_fraction_of_overlap": float(common_clear.mean()),
            "first_only_clear_pixels": int(
                (first_clear_bool & ~second_clear_bool).sum()
            ),
            "second_only_clear_pixels": int(
                (second_clear_bool & ~first_clear_bool).sum()
            ),
            "clear_union_pixels": clear_union_count,
            "clear_jaccard": (
                float(common_clear.sum() / clear_union_count)
                if clear_union_count
                else None
            ),
        },
        "paired_dual_local_z": {
            "finite_common_clear_pixels": paired_count,
            "pearson_r": _nullable_correlation(first_values, second_values),
            "second_minus_first_mean_bias": bias,
            "mean_absolute_difference": mae,
        },
        "fixed_location_high_tail": {
            "threshold_dual_local_z": float(threshold),
            "denominator": "finite pixels clear in both acquisitions",
            "first_pixels": int(first_tail.sum()),
            "second_pixels": int(second_tail.sum()),
            "overlap_pixels": int(tail_overlap.sum()),
            "union_pixels": tail_union_count,
            "jaccard": (
                float(tail_overlap.sum() / tail_union_count)
                if tail_union_count
                else None
            ),
            "first_only_pixels": int((first_tail & ~second_tail).sum()),
            "second_only_pixels": int((second_tail & ~first_tail).sum()),
        },
    }


def comparison_status(metrics: dict[str, object]) -> str:
    """Classify whether an aligned comparison has usable paired scores."""

    paired_count = int(metrics["paired_dual_local_z"]["finite_common_clear_pixels"])
    if paired_count == 0:
        return "no_finite_common_clear_scores"
    if int(metrics["fixed_location_high_tail"]["union_pixels"]) == 0:
        return "ok_no_high_tail"
    return "ok"


def _shared_score_limits(
    first: np.ndarray,
    second: np.ndarray,
    first_clear: np.ndarray,
    second_clear: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    values = np.concatenate(
        [
            first[first_clear & np.isfinite(first)],
            second[second_clear & np.isfinite(second)],
        ]
    )
    if not values.size:
        return -3.0, max(6.0, threshold + 1.0)
    lower = min(-3.0, float(np.quantile(values, 0.005)))
    upper = max(threshold + 1.0, 6.0, float(np.quantile(values, 0.995)))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def save_repeat_comparison_figure(
    path: str | Path,
    first_dual_z: np.ndarray,
    second_dual_z: np.ndarray,
    first_clear: np.ndarray,
    second_clear: np.ndarray,
    *,
    first_label: str,
    second_label: str,
    threshold: float,
    epsg: int,
    metrics: dict[str, object],
) -> None:
    """Save aligned score maps and a categorical fixed-location tail map."""

    first_clear = np.asarray(first_clear, dtype=bool)
    second_clear = np.asarray(second_clear, dtype=bool)
    first_dual_z = np.asarray(first_dual_z, dtype=float)
    second_dual_z = np.asarray(second_dual_z, dtype=float)
    common_clear = (
        first_clear
        & second_clear
        & np.isfinite(first_dual_z)
        & np.isfinite(second_dual_z)
    )
    first_tail = common_clear & (first_dual_z >= threshold)
    second_tail = common_clear & (second_dual_z >= threshold)
    category = np.full(common_clear.shape, -1, dtype=np.int8)
    category[common_clear] = 0
    category[first_tail & ~second_tail] = 1
    category[second_tail & ~first_tail] = 2
    category[first_tail & second_tail] = 3

    lower, upper = _shared_score_limits(
        first_dual_z,
        second_dual_z,
        first_clear,
        second_clear,
        threshold,
    )
    score_cmap = plt.get_cmap("coolwarm").copy()
    score_cmap.set_bad("#b8b8b8")
    score_norm = Normalize(vmin=lower, vmax=upper)
    tail_colors = ["#b8b8b8", "#f7f7f7", "#377eb8", "#ff8c42", "#7b3294"]
    tail_cmap = ListedColormap(tail_colors)
    tail_norm = BoundaryNorm(np.arange(-1.5, 4.5, 1.0), tail_cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    first_shown = np.ma.masked_where(
        ~first_clear | ~np.isfinite(first_dual_z), first_dual_z
    )
    second_shown = np.ma.masked_where(
        ~second_clear | ~np.isfinite(second_dual_z), second_dual_z
    )
    first_image = axes[0].imshow(first_shown, cmap=score_cmap, norm=score_norm)
    axes[1].imshow(second_shown, cmap=score_cmap, norm=score_norm)
    axes[0].set_title(f"First acquisition\n{first_label}")
    axes[1].set_title(f"Second acquisition\n{second_label}")
    tail = metrics["fixed_location_high_tail"]
    paired = metrics["paired_dual_local_z"]
    axes[2].imshow(category, cmap=tail_cmap, norm=tail_norm, interpolation="nearest")
    if int(paired["finite_common_clear_pixels"]) == 0:
        tail_title = (
            f"Fixed-location tail (z >= {threshold:g})\n"
            "No finite common-clear score pairs"
        )
    elif int(tail["union_pixels"]) == 0:
        tail_title = f"Fixed-location tail (z >= {threshold:g})\nNo high-tail pixels"
    else:
        jaccard = tail["jaccard"]
        tail_title = (
            f"Fixed-location tail (z >= {threshold:g})\n"
            f"overlap={tail['overlap_pixels']}, Jaccard={jaccard:.3f}"
        )
    axes[2].set_title(tail_title)
    for axis in axes:
        axis.set_xlabel("Common-grid column")
        axis.set_ylabel("Common-grid row")
    fig.colorbar(
        first_image,
        ax=axes[:2],
        shrink=0.82,
        label="dual-band local z (shared scale)",
    )
    axes[2].legend(
        handles=[
            Patch(facecolor=tail_colors[0], label="not clear in both"),
            Patch(facecolor=tail_colors[1], edgecolor="#777777", label="neither"),
            Patch(facecolor=tail_colors[2], label="first only"),
            Patch(facecolor=tail_colors[3], label="second only"),
            Patch(facecolor=tail_colors[4], label="both maps"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    correlation = paired["pearson_r"]
    correlation_text = "undefined" if correlation is None else f"{correlation:.3f}"
    fig.suptitle(
        f"Cloud-aware aligned-scene comparison (EPSG:{epsg}); "
        f"finite common-clear score pairs n={paired['finite_common_clear_pixels']:,}, "
        f"r={correlation_text}\n"
        "Same-pixel persistence is a surface/artifact diagnostic, not plume confirmation.",
        fontsize=12,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _alignment_json(alignment: GridAlignment, epsg: int) -> dict[str, object]:
    return {
        "epsg": int(epsg),
        "pixel_coordinate_convention": (
            "GeoTIFF transforms are normalized to GDAL pixel-corner coordinates; "
            "reported matching locations use a +0.5 row/column pixel-centre offset."
        ),
        "first_bounds_y0_y1_x0_x1": list(alignment.first_bounds),
        "second_bounds_y0_y1_x0_x1": list(alignment.second_bounds),
        "common_shape_y_x": list(alignment.shape),
        "second_origin_offset_in_first_col_row": list(
            alignment.second_origin_offset_in_first_col_row
        ),
        "projected_edge_corners_ul_ur_lr_ll_m": [
            list(point) for point in alignment.projected_edge_corners_m
        ],
        "maximum_checked_pixel_center_residual_m": (
            alignment.maximum_pixel_center_residual_m
        ),
        "limitation": (
            "This verifies GeoTIFF affine-grid coincidence, not sub-pixel surface "
            "co-registration between acquisition dates. Image-based registration "
            "or a spatial-tolerance sensitivity analysis is needed when geolocation "
            "residuals are material."
        ),
    }


def _georeference_json(georef: GeoReference) -> dict[str, object]:
    return {
        "epsg": georef.epsg,
        "crs": georef.crs,
        "normalized_gdal_geotransform": list(georef.geotransform),
        "source_raster_type": georef.raster_type,
        "pixel_scale": list(georef.pixel_scale),
    }


def _prepare_output_paths(output_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create the output directory and invalidate both artifacts from older runs."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "repeat_acquisition_summary.json"
    figure_path = destination / "repeat_acquisition_comparison.png"
    summary_path.unlink(missing_ok=True)
    figure_path.unlink(missing_ok=True)
    return destination, summary_path, figure_path


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir, summary_path, figure_path = _prepare_output_paths(args.output_dir)
    if not np.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    first_product = discover_l1g_product(args.first_product)
    second_product = discover_l1g_product(args.second_product)
    first_scores = load_score_maps(args.first_scene_output)
    second_scores = load_score_maps(args.second_scene_output)
    first_georef = read_georeference(first_product.image_path)
    second_georef = read_georeference(second_product.image_path)
    if first_scores.valid.shape != tuple(
        int(value) for value in _tiff_shape(first_product.image_path)
    ):
        raise ValueError("First score-map shape does not match its HISUI product")
    if second_scores.valid.shape != tuple(
        int(value) for value in _tiff_shape(second_product.image_path)
    ):
        raise ValueError("Second score-map shape does not match its HISUI product")

    first_metadata = parse_metadata(first_product.metadata_path)
    second_metadata = parse_metadata(second_product.metadata_path)
    first_signature, second_signature, target_validation = validate_pair_provenance(
        first_scores,
        second_scores,
        first_product_id=first_product.product_id,
        second_product_id=second_product.product_id,
        first_acquisition_utc=first_metadata.get("SceneCenterTime"),
        second_acquisition_utc=second_metadata.get("SceneCenterTime"),
    )
    input_summary = {
        "analysis_version": ANALYSIS_VERSION,
        "first_product_id": first_product.product_id,
        "second_product_id": second_product.product_id,
        "first_product": str(first_product.directory),
        "second_product": str(second_product.directory),
        "first_score_maps": str(first_scores.path),
        "second_score_maps": str(second_scores.path),
        "first_acquisition_utc": first_metadata.get("SceneCenterTime"),
        "second_acquisition_utc": second_metadata.get("SceneCenterTime"),
        "first_source_summary": str(first_scores.summary_path),
        "second_source_summary": str(second_scores.summary_path),
        "first_source_quality_class": first_scores.quality_class,
        "second_source_quality_class": second_scores.quality_class,
        "first_product_georeference": _georeference_json(first_georef),
        "second_product_georeference": _georeference_json(second_georef),
        "score_archive_georeference_validation": (
            "The archive is linked to the product by adjacent summary product_id "
            "and exact Y/X shape. The archive format itself does not embed an affine "
            "transform; alignment uses the linked product GeoTIFF headers."
        ),
        "source_analysis_config_validation": {
            "status": "matched_map_affecting_parameters",
            "signature_kind": (
                "SHA-256 of map-affecting analysis_config fields from each "
                "adjacent summary.json"
            ),
            "first_sha256": first_signature["sha256"],
            "second_sha256": second_signature["sha256"],
            "validated_parameters": first_signature["parameters"],
        },
        "source_target_provenance_validation": target_validation,
    }
    alignment = common_aligned_overlap(
        first_georef,
        first_scores.valid.shape,
        second_georef,
        second_scores.valid.shape,
        grid_tolerance_pixels=args.grid_tolerance_pixels,
    )
    interpretation = (
        "This is a descriptive comparison of independently fitted scene score "
        "maps. Coincident high tails at fixed projected pixels may reflect a "
        "stationary surface/background feature, geometric residual, or detector "
        "artifact. They do not confirm a methane plume; cloud-free spectral and "
        "spatial plume evidence is still required. Affine-grid agreement also "
        "does not prove sub-pixel surface co-registration between dates."
    )
    if alignment is None:
        summary: dict[str, object] = {
            "status": "no_projected_overlap",
            "inputs": input_summary,
            "parameters": {
                "threshold_dual_local_z": float(args.threshold),
                "grid_tolerance_pixels": float(args.grid_tolerance_pixels),
            },
            "alignment": None,
            "metrics": None,
            "outputs": {"figure": None, "summary": str(summary_path)},
            "interpretation": interpretation,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    first_slice = _bounds_slice(alignment.first_bounds)
    second_slice = _bounds_slice(alignment.second_bounds)
    first_dual = first_scores.dual_local_z[first_slice]
    second_dual = second_scores.dual_local_z[second_slice]
    first_clear = first_scores.valid[first_slice]
    second_clear = second_scores.valid[second_slice]
    metrics = repeat_metrics(
        first_dual,
        second_dual,
        first_clear,
        second_clear,
        threshold=args.threshold,
    )
    status = comparison_status(metrics)
    first_tile = first_product.product_id.removeprefix("HSHL1G_").split("_")[0]
    second_tile = second_product.product_id.removeprefix("HSHL1G_").split("_")[0]
    first_label = (
        f"{first_tile} | "
        f"{str(first_metadata.get('SceneCenterTime', 'unknown'))[:19]}"
    )
    second_label = (
        f"{second_tile} | "
        f"{str(second_metadata.get('SceneCenterTime', 'unknown'))[:19]}"
    )
    save_repeat_comparison_figure(
        figure_path,
        first_dual,
        second_dual,
        first_clear,
        second_clear,
        first_label=first_label,
        second_label=second_label,
        threshold=args.threshold,
        epsg=int(first_georef.epsg),
        metrics=metrics,
    )
    summary = {
        "status": status,
        "inputs": input_summary,
        "parameters": {
            "threshold_dual_local_z": float(args.threshold),
            "grid_tolerance_pixels": float(args.grid_tolerance_pixels),
        },
        "alignment": _alignment_json(alignment, int(first_georef.epsg)),
        "metrics": metrics,
        "outputs": {"figure": str(figure_path), "summary": str(summary_path)},
        "interpretation": interpretation,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _tiff_shape(path: str | Path) -> tuple[int, int]:
    """Read the image Y/X dimensions without decoding any TIFF tiles."""

    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        if len(tif.pages) != 1:
            raise ValueError("Expected a single-page HISUI L1G TIFF")
        page = tif.pages[0]
        return int(page.imagelength), int(page.imagewidth)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-product", type=Path, required=True)
    parser.add_argument("--second-product", type=Path, required=True)
    parser.add_argument(
        "--first-scene-output",
        "--first-score-maps",
        dest="first_scene_output",
        type=Path,
        required=True,
        help="Scene output directory or its score_maps.npz",
    )
    parser.add_argument(
        "--second-scene-output",
        "--second-score-maps",
        dest="second_scene_output",
        type=Path,
        required=True,
        help="Scene output directory or its score_maps.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--grid-tolerance-pixels", type=float, default=1e-6)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

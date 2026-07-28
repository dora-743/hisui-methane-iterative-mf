#!/usr/bin/env python3
"""Compare two HISUI reprocessings on one common projected ROI.

The products are aligned by GeoTIFF projected coordinates, never by bare pixel
index.  One pooled background model and one MODTRAN-derived target are applied
to both arrays, isolating radiometric/geometric processing sensitivity from
differences caused by fitting two separate matched filters.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hisui_l1g_io import HISUIL1GReader, discover_l1g_product, parse_metadata
from physics_aware_crossfit_mf import fit_detector_model, score_log_radiance
from screen_hisui_l1g_scenes import (
    CLOUD_PROFILES,
    _radiance_coefficients,
    _template_for_scene,
    cloud_proxy_from_quality_dn,
    detector_qa_mask,
    dilate_cloud_mask,
    invalid_dn_values,
    make_band_plan,
    projected_to_pixel,
    saturated_dn_values,
)


ANALYSIS_VERSION = "2026-07-28-v2"
ACQUISITION_TIME_TOLERANCE_SECONDS = 1.0


def _product_acquisition_token(product_id: str) -> str:
    """Return the 14-digit acquisition token embedded in a HISUI product ID."""

    parts = str(product_id).split("_")
    if len(parts) < 4 or len(parts[-2]) != 14 or not parts[-2].isdigit():
        raise ValueError(
            f"Cannot identify the acquisition timestamp in product ID {product_id!r}"
        )
    return parts[-2]


def _source_scene_id(product_id: str) -> str:
    """Return the processing-independent scene ID from a HISUI product ID."""

    parts = str(product_id).split("_")
    if len(parts) < 4 or not parts[-1]:
        raise ValueError(f"Cannot identify the source scene in product ID {product_id!r}")
    _product_acquisition_token(product_id)
    return "_".join(parts[:-1])


def _metadata_scene_time(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} metadata lacks SceneCenterTime")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} SceneCenterTime is not valid ISO-8601: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} SceneCenterTime must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_same_acquisition(
    old_product,
    new_product,
    old_metadata: Mapping[str, object],
    new_metadata: Mapping[str, object],
    *,
    tolerance_seconds: float = ACQUISITION_TIME_TOLERANCE_SECONDS,
) -> dict[str, object]:
    """Reject a pair that cannot be proven to be two processings of one scene."""

    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Acquisition-time tolerance must be finite and non-negative")
    for label, product, metadata in (
        ("Old", old_product, old_metadata),
        ("New", new_product, new_metadata),
    ):
        metadata_product_id = metadata.get("ProductID")
        if metadata_product_id != product.product_id:
            raise ValueError(
                f"{label} metadata ProductID {metadata_product_id!r} does not match "
                f"the discovered product {product.product_id!r}"
            )

    old_token = _product_acquisition_token(old_product.product_id)
    new_token = _product_acquisition_token(new_product.product_id)
    if old_token != new_token:
        raise ValueError(
            "The two HISUI products have different acquisition tokens: "
            f"{old_token} != {new_token}"
        )
    old_scene_id = _source_scene_id(old_product.product_id)
    new_scene_id = _source_scene_id(new_product.product_id)
    if old_scene_id != new_scene_id:
        raise ValueError(
            "The two HISUI products do not identify the same source scene: "
            f"{old_scene_id!r} != {new_scene_id!r}"
        )
    old_time = _metadata_scene_time(
        old_metadata.get("SceneCenterTime"), label="Old product"
    )
    new_time = _metadata_scene_time(
        new_metadata.get("SceneCenterTime"), label="New product"
    )
    delta_seconds = abs((new_time - old_time).total_seconds())
    if delta_seconds > tolerance_seconds:
        raise ValueError(
            "The two HISUI products have different SceneCenterTime values "
            f"(difference {delta_seconds:.6f} s exceeds {tolerance_seconds:.6f} s)"
        )
    return {
        "status": "same_acquisition_verified",
        "processing_independent_source_scene_id": old_scene_id,
        "product_id_acquisition_token": old_token,
        "old_scene_center_time": old_metadata["SceneCenterTime"],
        "new_scene_center_time": new_metadata["SceneCenterTime"],
        "absolute_scene_center_time_difference_seconds": delta_seconds,
        "allowed_time_difference_seconds": float(tolerance_seconds),
        "old_metadata_product_id_match": True,
        "new_metadata_product_id_match": True,
    }


def common_projected_bounds(
    old_reader: HISUIL1GReader,
    new_reader: HISUIL1GReader,
    *,
    easting_m: float,
    northing_m: float,
    half_size_pixels: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[float, float, float, float]]:
    if old_reader.georeference.epsg != new_reader.georeference.epsg:
        raise ValueError("The two products use different projected CRSs")
    old_linear = np.asarray(old_reader.georeference.geotransform)[[1, 2, 4, 5]]
    new_linear = np.asarray(new_reader.georeference.geotransform)[[1, 2, 4, 5]]
    if not np.allclose(old_linear, new_linear, rtol=0.0, atol=1e-9):
        raise ValueError("The products do not share the same pixel scale/rotation")
    old_row, old_column = projected_to_pixel(
        old_reader.georeference, easting_m, northing_m
    )
    center_row = int(round(old_row))
    center_column = int(round(old_column))
    old_bounds = (
        center_row - half_size_pixels,
        center_row + half_size_pixels,
        center_column - half_size_pixels,
        center_column + half_size_pixels,
    )
    if not (
        0 <= old_bounds[0] < old_bounds[1] <= old_reader.height
        and 0 <= old_bounds[2] < old_bounds[3] <= old_reader.width
    ):
        raise ValueError("Requested projected ROI falls outside the old product")
    upper_left = old_reader.georeference.map_xy(old_bounds[0], old_bounds[2])
    lower_right = old_reader.georeference.map_xy(old_bounds[1], old_bounds[3])

    def edge_index(reader: HISUIL1GReader, x: float, y: float) -> tuple[int, int]:
        x0, dx, rx, y0, ry, dy = reader.georeference.geotransform
        column, row = np.linalg.solve(
            np.asarray([[dx, rx], [ry, dy]], dtype=float),
            np.asarray([x - x0, y - y0], dtype=float),
        )
        rounded_row, rounded_column = int(round(row)), int(round(column))
        if not np.allclose(
            [row, column], [rounded_row, rounded_column], rtol=0.0, atol=1e-6
        ):
            raise ValueError("The product grids are not aligned at a pixel boundary")
        return rounded_row, rounded_column

    new_y0, new_x0 = edge_index(new_reader, *upper_left)
    new_y1, new_x1 = edge_index(new_reader, *lower_right)
    new_bounds = (new_y0, new_y1, new_x0, new_x1)
    if not (
        0 <= new_y0 < new_y1 <= new_reader.height
        and 0 <= new_x0 < new_x1 <= new_reader.width
    ):
        raise ValueError("Common projected ROI falls outside the new product")
    if (old_bounds[1] - old_bounds[0], old_bounds[3] - old_bounds[2]) != (
        new_y1 - new_y0,
        new_x1 - new_x0,
    ):
        raise ValueError("The products do not share the same projected pixel grid")
    for row, column in (
        (old_bounds[0], old_bounds[2]),
        (old_bounds[0], old_bounds[3]),
        (old_bounds[1], old_bounds[2]),
        (old_bounds[1], old_bounds[3]),
    ):
        x, y = old_reader.georeference.map_xy(row, column)
        new_row, new_column = edge_index(new_reader, x, y)
        old_x, old_y = old_reader.georeference.map_xy(row, column)
        new_x, new_y = new_reader.georeference.map_xy(new_row, new_column)
        if not np.allclose([old_x, old_y], [new_x, new_y], rtol=0.0, atol=1e-6):
            raise ValueError("Aligned ROI corners differ in projected coordinates")
    projected = (upper_left[0], upper_left[1], lower_right[0], lower_right[1])
    return old_bounds, new_bounds, projected


def prepare_roi(
    reader: HISUIL1GReader,
    bounds: tuple[int, int, int, int],
    plan,
    *,
    cloud_dilation_pixels: int,
    cloud_profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    y0, y1, x0, x1 = bounds
    dn = reader.read_region(y0, y1, x0, x1, plan.read_bands)
    metadata = dict(reader.metadata)
    invalid_values = invalid_dn_values(metadata, include_saturation=False)
    saturated_values = saturated_dn_values(metadata)
    quality_dn = dn[..., plan.quality_positions]
    quality_qa, quality_qa_stats = detector_qa_mask(
        reader.product,
        metadata,
        plan.quality_bands,
        bounds=bounds,
        include_interpolated=True,
        stats_prefix="quality_qa",
    )
    if quality_qa.shape != quality_dn.shape[:-1]:
        raise ValueError(
            f"Quality QA shape {quality_qa.shape} differs from ROI "
            f"{quality_dn.shape[:-1]}"
        )
    quality_valid, cloud_raw, stats = cloud_proxy_from_quality_dn(
        quality_dn,
        plan.reflectance_multiplier,
        plan.reflectance_additive,
        invalid_values,
        saturated_values=saturated_values,
        qa_affected=quality_qa,
        profile=cloud_profile,
    )
    cloud = dilate_cloud_mask(cloud_raw, cloud_dilation_pixels)
    detection_dn = dn[..., plan.detection_positions]
    detection_valid = ~np.any(
        np.isin(
            detection_dn,
            invalid_dn_values(metadata, include_saturation=True),
        ),
        axis=-1,
    )
    detector_qa, detector_qa_stats = detector_qa_mask(
        reader.product,
        metadata,
        plan.detection_bands,
        bounds=bounds,
        include_interpolated=True,
    )
    if detector_qa.shape != detection_valid.shape:
        raise ValueError(
            f"Detector QA shape {detector_qa.shape} differs from ROI "
            f"{detection_valid.shape}"
        )
    detection_valid &= ~detector_qa
    multiplier, additive = _radiance_coefficients(metadata, plan.detection_bands)
    radiance = detection_dn.astype(np.float64) * multiplier + additive
    positive = np.all(np.isfinite(radiance) & (radiance > 0), axis=-1)
    clear = quality_valid & detection_valid & positive & ~cloud
    radiance[~clear] = np.nan
    reflectance = quality_dn.astype(np.float32)
    reflectance *= plan.reflectance_multiplier.astype(np.float32)
    reflectance += plan.reflectance_additive.astype(np.float32)
    r665, r865, _r1388, r1650, _r2200 = np.moveaxis(reflectance, -1, 0)
    false_color = np.stack([r1650, r865, r665], axis=-1)
    false_color[~quality_valid] = np.nan
    stats["clear_fraction_in_roi"] = float(clear.mean())
    stats["cloud_dilated_fraction_in_roi"] = float(cloud.mean())
    stats.update(quality_qa_stats)
    stats.update(detector_qa_stats)
    stats["qa_complete"] = bool(
        quality_qa_stats["quality_qa_complete"]
        and detector_qa_stats["detector_qa_complete"]
    )
    if not stats["qa_complete"]:
        raise ValueError(
            f"Delivered quality/detector QA is incomplete for {reader.product.product_id}"
        )
    return radiance, clear, cloud, false_color, stats


def shared_false_color_stretch(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    pooled = np.concatenate(
        [
            first[np.all(np.isfinite(first), axis=-1)],
            second[np.all(np.isfinite(second), axis=-1)],
        ],
        axis=0,
    )
    if len(pooled) == 0:
        return np.zeros_like(first), np.zeros_like(second), [[0.0] * 3, [1.0] * 3]
    lower = np.quantile(pooled, 0.01, axis=0)
    upper = np.quantile(pooled, 0.99, axis=0)

    def stretch(values: np.ndarray) -> np.ndarray:
        shown = np.clip((values - lower) / np.maximum(upper - lower, 1e-6), 0, 1)
        shown[~np.all(np.isfinite(values), axis=-1)] = 0
        return shown

    return stretch(first), stretch(second), [lower.tolist(), upper.tolist()]


def _sample_rows(values: np.ndarray, mask: np.ndarray, count: int, seed: int) -> np.ndarray:
    spectra = values[mask]
    if len(spectra) <= count:
        return spectra
    rng = np.random.default_rng(seed)
    return spectra[rng.choice(len(spectra), size=count, replace=False)]


def score_roi(
    radiance: np.ndarray,
    valid: np.ndarray,
    model,
    transform: np.ndarray,
) -> dict[str, np.ndarray]:
    maps = {
        key: np.full(valid.shape, np.nan, dtype=np.float32)
        for key in score_log_radiance(
            np.log(radiance[valid][:1]) @ transform.T, model
        )
    }
    scores = score_log_radiance(np.log(radiance[valid]) @ transform.T, model)
    for key, values in scores.items():
        maps[key][valid] = values.astype(np.float32)
    return maps


def finite_correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    use = mask & np.isfinite(first) & np.isfinite(second)
    if use.sum() < 2:
        return float("nan")
    return float(np.corrcoef(first[use], second[use])[0, 1])


def paired_diagnostics(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | None]:
    use = mask & np.isfinite(first) & np.isfinite(second)
    old = np.asarray(first[use], dtype=float)
    new = np.asarray(second[use], dtype=float)
    if len(old) < 2:
        return {
            "pixels": int(len(old)),
            "pearson_r": None,
            "new_minus_old_mean_bias": None,
            "mean_absolute_difference": None,
            "new_on_old_slope": None,
            "new_on_old_intercept": None,
        }
    if np.std(old) <= 1e-12 or np.std(new) <= 1e-12:
        correlation = None
        slope = None
        intercept = None
    else:
        correlation = float(np.corrcoef(old, new)[0, 1])
        fitted_slope, fitted_intercept = np.polyfit(old, new, 1)
        slope = float(fitted_slope)
        intercept = float(fitted_intercept)
    return {
        "pixels": int(len(old)),
        "pearson_r": correlation,
        "new_minus_old_mean_bias": float(np.mean(new - old)),
        "mean_absolute_difference": float(np.mean(np.abs(new - old))),
        "new_on_old_slope": slope,
        "new_on_old_intercept": intercept,
    }


def window_summary(
    maps: dict[str, np.ndarray],
    clear: np.ndarray,
    center: tuple[float, float],
    radius: int,
) -> dict[str, float | int | None]:
    cy, cx = (int(round(center[0])), int(round(center[1])))
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, clear.shape[0])
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, clear.shape[1])
    valid = clear[y0:y1, x0:x1]
    output: dict[str, float | int | None] = {
        "clear_pixels": int(valid.sum()),
        "clear_fraction": float(valid.mean()),
    }
    for key in ("weak_z", "strong_z", "dual_min_z", "combined_z", "log_e_average"):
        values = maps[key][y0:y1, x0:x1][valid]
        output[f"{key}_mean"] = float(np.mean(values)) if len(values) else None
        output[f"{key}_max"] = float(np.max(values)) if len(values) else None
    return output


def save_comparison(
    path: Path,
    old_product,
    new_product,
    old_false_color,
    new_false_color,
    old_maps,
    new_maps,
    common_clear,
    old_cloud,
    new_cloud,
    center,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(
        "Common-grid L1G reprocessing comparison | white: cloud/invalid; "
        "green +: fixed site",
        fontsize=13,
    )
    for axis, image, title in (
        (axes[0, 0], old_false_color, "Old aligned SWIR/NIR false color"),
        (axes[0, 1], new_false_color, "New aligned SWIR/NIR false color"),
    ):
        axis.imshow(image)
        axis.plot(center[1], center[0], marker="+", color="lime", markersize=13, markeredgewidth=2)
        axis.set_title(title)
        axis.set_axis_off()
    old_dual = old_maps["dual_min_z"]
    new_dual = new_maps["dual_min_z"]
    difference = new_dual - old_dual
    panels = [
        (axes[0, 2], old_dual, "Old pooled-model dual min z", "coolwarm", -3, 6),
        (axes[0, 3], new_dual, "New pooled-model dual min z", "coolwarm", -3, 6),
        (axes[1, 0], difference, "New - old dual min z", "coolwarm", -2, 2),
        (axes[1, 1], old_cloud.astype(float), "Old cloud proxy", "Reds", 0, 1),
        (axes[1, 2], new_cloud.astype(float), "New cloud proxy", "Reds", 0, 1),
    ]
    for axis, image, title, cmap, lower, upper in panels:
        shown = axis.imshow(image, cmap=cmap, vmin=lower, vmax=upper)
        axis.plot(center[1], center[0], marker="+", color="lime", markersize=13, markeredgewidth=2)
        axis.set_title(title)
        axis.set_axis_off()
        fig.colorbar(shown, ax=axis, shrink=0.7)
    use = common_clear & np.isfinite(old_dual) & np.isfinite(new_dual)
    if use.sum() > 30_000:
        indices = np.linspace(0, use.sum() - 1, 30_000, dtype=int)
        old_values = old_dual[use][indices]
        new_values = new_dual[use][indices]
    else:
        old_values, new_values = old_dual[use], new_dual[use]
    if len(old_values):
        axes[1, 3].hexbin(old_values, new_values, gridsize=55, mincnt=1, cmap="viridis")
        limits = np.nanpercentile(np.concatenate([old_values, new_values]), [0.5, 99.5])
        if np.isclose(limits[0], limits[1]):
            limits = limits + np.asarray([-0.5, 0.5])
        axes[1, 3].plot(limits, limits, color="black", linewidth=1)
        axes[1, 3].set_xlim(limits)
        axes[1, 3].set_ylim(limits)
    else:
        axes[1, 3].text(0.5, 0.5, "No common clear pixels", ha="center", va="center")
    axes[1, 3].set_xlabel("Old dual min z")
    axes[1, 3].set_ylabel("New dual min z")
    axes[1, 3].set_title("Common clear pixels")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_size < 2:
        raise ValueError("sample size must be at least two")
    if args.site_radius_pixels < 0:
        raise ValueError("site radius pixels must be non-negative")
    if args.minimum_common_clear_pixels < 2:
        raise ValueError("minimum common clear pixels must be at least two")
    if not 0.0 <= args.maximum_site_cloud_fraction <= 1.0:
        raise ValueError("maximum site cloud fraction must lie in [0, 1]")
    old_product = discover_l1g_product(args.old_product)
    new_product = discover_l1g_product(args.new_product)
    old_metadata = parse_metadata(old_product.metadata_path)
    new_metadata = parse_metadata(new_product.metadata_path)
    acquisition_validation = validate_same_acquisition(
        old_product, new_product, old_metadata, new_metadata
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_plan = make_band_plan(
        old_product,
        weak_window_nm=(args.weak_min_nm, args.weak_max_nm),
        strong_window_nm=(args.strong_min_nm, args.strong_max_nm),
    )
    new_plan = make_band_plan(
        new_product,
        weak_window_nm=(args.weak_min_nm, args.weak_max_nm),
        strong_window_nm=(args.strong_min_nm, args.strong_max_nm),
    )
    np.testing.assert_allclose(old_plan.wavelengths_nm, new_plan.wavelengths_nm)
    np.testing.assert_allclose(old_plan.fwhm_nm, new_plan.fwhm_nm)

    with HISUIL1GReader(old_product) as old_reader, HISUIL1GReader(new_product) as new_reader:
        old_bounds, new_bounds, projected_bounds = common_projected_bounds(
            old_reader,
            new_reader,
            easting_m=args.site_easting,
            northing_m=args.site_northing,
            half_size_pixels=args.half_size_pixels,
        )
        old_radiance, old_clear, old_cloud, old_false_color, old_quality = prepare_roi(
            old_reader,
            old_bounds,
            old_plan,
            cloud_dilation_pixels=args.cloud_dilation_pixels,
            cloud_profile=args.cloud_profile,
        )
        new_radiance, new_clear, new_cloud, new_false_color, new_quality = prepare_roi(
            new_reader,
            new_bounds,
            new_plan,
            cloud_dilation_pixels=args.cloud_dilation_pixels,
            cloud_profile=args.cloud_profile,
        )
        old_georef = old_reader.georeference
        new_georef = new_reader.georeference

    old_false_color, new_false_color, false_color_limits = shared_false_color_stretch(
        old_false_color, new_false_color
    )
    common_clear = old_clear & new_clear
    if common_clear.sum() < args.minimum_common_clear_pixels:
        raise ValueError("Too few common clear pixels for a stable comparison")
    transform, target, weak_features, strong_features, alpha_grid, _ = _template_for_scene(
        old_plan,
        args.modtran_csv,
        uas_alpha_min=args.uas_alpha_min,
        uas_alpha_max=args.uas_alpha_max,
        continuum_degree=args.continuum_degree,
    )
    old_sample = _sample_rows(old_radiance, common_clear, args.sample_size // 2, args.seed)
    new_sample = _sample_rows(new_radiance, common_clear, args.sample_size // 2, args.seed + 1)
    pooled = np.vstack([old_sample, new_sample])
    model = fit_detector_model(
        np.log(pooled) @ transform.T,
        target,
        weak_features,
        strong_features,
        shrinkage=args.covariance_shrinkage,
        ridge=args.covariance_ridge,
        amplitude_cap=(float(alpha_grid[-1]) if args.amplitude_cap is None else args.amplitude_cap),
    )
    old_maps = score_roi(old_radiance, common_clear, model, transform)
    new_maps = score_roi(new_radiance, common_clear, model, transform)

    old_site_full = projected_to_pixel(old_georef, args.site_easting, args.site_northing)
    old_center = (old_site_full[0] - old_bounds[0], old_site_full[1] - old_bounds[2])
    new_site_full = projected_to_pixel(new_georef, args.site_easting, args.site_northing)
    new_center = (new_site_full[0] - new_bounds[0], new_site_full[1] - new_bounds[2])
    if not np.allclose(old_center, new_center, atol=1e-6):
        raise ValueError("Projected site does not align inside the common ROI")

    score_diagnostics = {
        key: paired_diagnostics(old_maps[key], new_maps[key], common_clear)
        for key in ("weak_z", "strong_z", "dual_min_z", "combined_z", "log_e_average")
    }
    radiance_rows: list[dict[str, float]] = []
    for band, wavelength in enumerate(old_plan.wavelengths_nm):
        old_values = old_radiance[..., band][common_clear]
        new_values = new_radiance[..., band][common_clear]
        denominator = np.maximum(np.abs(old_values) + np.abs(new_values), 1e-9)
        radiance_rows.append(
            {
                "wavelength_nm": float(wavelength),
                "pearson_r": float(np.corrcoef(old_values, new_values)[0, 1]),
                "median_absolute_difference": float(np.median(np.abs(new_values - old_values))),
                "median_symmetric_relative_difference": float(
                    np.median(2.0 * np.abs(new_values - old_values) / denominator)
                ),
            }
        )
    pd.DataFrame(radiance_rows).to_csv(
        output_dir / "bandwise_radiance_stability.csv", index=False
    )
    old_positive = common_clear & (old_maps["weak_z"] >= 3) & (old_maps["strong_z"] >= 3)
    new_positive = common_clear & (new_maps["weak_z"] >= 3) & (new_maps["strong_z"] >= 3)
    union = old_positive | new_positive
    overlap = old_positive & new_positive
    clear_union = old_clear | new_clear
    clear_overlap = old_clear & new_clear
    site_y, site_x = int(round(old_center[0])), int(round(old_center[1]))
    radius = args.site_radius_pixels
    site_slice = (
        slice(max(site_y - radius, 0), min(site_y + radius + 1, common_clear.shape[0])),
        slice(max(site_x - radius, 0), min(site_x + radius + 1, common_clear.shape[1])),
    )
    old_site_cloud_fraction = float(old_cloud[site_slice].mean())
    new_site_cloud_fraction = float(new_cloud[site_slice].mean())
    site_common_clear_fraction = float(common_clear[site_slice].mean())
    if (
        max(old_site_cloud_fraction, new_site_cloud_fraction)
        >= args.maximum_site_cloud_fraction
    ):
        site_status = "excluded_site_cloud"
    elif site_common_clear_fraction <= 0.0:
        site_status = "excluded_no_common_clear_pixels"
    elif site_common_clear_fraction >= 0.90:
        site_status = "site_clear"
    else:
        site_status = "site_partly_observable"
    summary = {
        "inputs": {
            "analysis_version": ANALYSIS_VERSION,
            "old_product": str(old_product.directory),
            "new_product": str(new_product.directory),
            "old_product_id": old_product.product_id,
            "new_product_id": new_product.product_id,
            "same_acquisition_validation": acquisition_validation,
            "modtran_csv": str(args.modtran_csv),
            "modtran_scale_note": "A global x100 scale cancels in the log-radiance UAS.",
            "cloud_profile": args.cloud_profile,
        },
        "parameters": {
            "half_size_pixels": args.half_size_pixels,
            "site_radius_pixels": args.site_radius_pixels,
            "cloud_dilation_pixels": args.cloud_dilation_pixels,
            "maximum_site_cloud_fraction": args.maximum_site_cloud_fraction,
            "minimum_common_clear_pixels": args.minimum_common_clear_pixels,
            "weak_window_nm": [args.weak_min_nm, args.weak_max_nm],
            "strong_window_nm": [args.strong_min_nm, args.strong_max_nm],
            "uas_alpha_range": [args.uas_alpha_min, args.uas_alpha_max],
            "continuum_degree": args.continuum_degree,
            "requested_sample_size": args.sample_size,
            "actual_old_sample_size": int(len(old_sample)),
            "actual_new_sample_size": int(len(new_sample)),
            "seed": args.seed,
            "covariance_shrinkage": args.covariance_shrinkage,
            "covariance_ridge": args.covariance_ridge,
            "amplitude_cap": float(model.amplitude_cap),
            "false_color_shared_quantiles_1_99": false_color_limits,
        },
        "alignment": {
            "epsg": old_georef.epsg,
            "old_pixel_bounds_y0_y1_x0_x1": list(old_bounds),
            "new_pixel_bounds_y0_y1_x0_x1": list(new_bounds),
            "projected_bounds_left_top_right_bottom_m": list(projected_bounds),
            "shape": list(common_clear.shape),
            "site_easting_northing_m": [args.site_easting, args.site_northing],
            "site_roi_y_x": list(old_center),
        },
        "quality": {
            "old": old_quality,
            "new": new_quality,
            "clear_mask_old_pixels": int(old_clear.sum()),
            "clear_mask_new_pixels": int(new_clear.sum()),
            "common_clear_pixels": int(common_clear.sum()),
            "common_clear_fraction": float(common_clear.mean()),
            "clear_mask_overlap_pixels": int(clear_overlap.sum()),
            "clear_mask_union_pixels": int(clear_union.sum()),
            "clear_mask_jaccard": float(
                clear_overlap.sum() / max(int(clear_union.sum()), 1)
            ),
            "old_only_clear_pixels": int((old_clear & ~new_clear).sum()),
            "new_only_clear_pixels": int((new_clear & ~old_clear).sum()),
        },
        "score_stability_on_common_clear_pixels": score_diagnostics,
        "dual_positive_tail_at_z3": {
            "old_pixels": int(old_positive.sum()),
            "new_pixels": int(new_positive.sum()),
            "overlap_pixels": int(overlap.sum()),
            "union_pixels": int(union.sum()),
            "jaccard": (
                float(overlap.sum() / union.sum()) if union.any() else None
            ),
        },
        "known_site_window": {
            "radius_pixels": args.site_radius_pixels,
            "status": site_status,
            "maximum_allowed_cloud_fraction": args.maximum_site_cloud_fraction,
            "old_cloud_fraction": old_site_cloud_fraction,
            "new_cloud_fraction": new_site_cloud_fraction,
            "common_clear_fraction": site_common_clear_fraction,
            "old": window_summary(
                old_maps, common_clear, old_center, args.site_radius_pixels
            ),
            "new": window_summary(
                new_maps, common_clear, new_center, args.site_radius_pixels
            ),
        },
        "interpretation": (
            "A descriptive pooled-model processing-stability diagnostic on one acquisition. "
            "High correlation supports robustness to L1G reprocessing; disagreement "
            "does not by itself identify which processing is more accurate. Scores are "
            "plug-in diagnostics because the pooled background pixels are also scored."
        ),
    }
    map_archive: dict[str, np.ndarray] = {
        "old_clear": old_clear,
        "new_clear": new_clear,
        "common_clear": common_clear,
        "old_cloud": old_cloud,
        "new_cloud": new_cloud,
        "feature_transform": transform,
        "model_mean": model.mean,
        "model_covariance": model.covariance,
        "model_target": model.target,
        "model_weak_indices": model.weak_indices,
        "model_strong_indices": model.strong_indices,
        "selected_wavelengths_nm": old_plan.wavelengths_nm,
        "selected_fwhm_nm": old_plan.fwhm_nm,
        "uas_alpha_grid": alpha_grid,
    }
    for key, values in old_maps.items():
        map_archive[f"old_{key}"] = values
    for key, values in new_maps.items():
        map_archive[f"new_{key}"] = values
    np.savez_compressed(output_dir / "aligned_pair_score_maps.npz", **map_archive)
    save_comparison(
        output_dir / "reprocessing_comparison.png",
        old_product,
        new_product,
        old_false_color,
        new_false_color,
        old_maps,
        new_maps,
        common_clear,
        old_cloud,
        new_cloud,
        old_center,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-product", type=Path, required=True)
    parser.add_argument("--new-product", type=Path, required=True)
    parser.add_argument("--modtran-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-easting", type=float, required=True)
    parser.add_argument("--site-northing", type=float, required=True)
    parser.add_argument("--half-size-pixels", type=int, default=100)
    parser.add_argument("--site-radius-pixels", type=int, default=5)
    parser.add_argument("--cloud-dilation-pixels", type=int, default=2)
    parser.add_argument(
        "--cloud-profile", choices=sorted(CLOUD_PROFILES), default="desert_balanced"
    )
    parser.add_argument("--minimum-common-clear-pixels", type=int, default=1000)
    parser.add_argument("--maximum-site-cloud-fraction", type=float, default=0.50)
    parser.add_argument("--weak-min-nm", type=float, default=1580.0)
    parser.add_argument("--weak-max-nm", type=float, default=1750.0)
    parser.add_argument("--strong-min-nm", type=float, default=2200.0)
    parser.add_argument("--strong-max-nm", type=float, default=2390.0)
    parser.add_argument("--uas-alpha-min", type=float, default=0.0)
    parser.add_argument("--uas-alpha-max", type=float, default=0.5)
    parser.add_argument("--continuum-degree", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=40_000)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.05)
    parser.add_argument("--covariance-ridge", type=float, default=1e-6)
    parser.add_argument("--amplitude-cap", type=float)
    parser.add_argument("--seed", type=int, default=743)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.modtran_csv = args.modtran_csv.expanduser().resolve()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

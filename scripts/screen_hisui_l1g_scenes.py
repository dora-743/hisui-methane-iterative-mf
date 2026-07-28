#!/usr/bin/env python3
"""Cloud-aware, image-producing methane screening of HISUI L1G products.

The script applies one MODTRAN CH4 concentration sweep to multiple L1G scenes.
It first estimates a conservative cloud proxy from TOA reflectance, excludes
cloud-dominated scenes from methane ranking, then fits the existing dual-window
cross-fitted matched filter on clear pixels.  The 2.3 um window is the primary
detector and the 1.65 um window is retained as a consistency check.

This is a candidate-screening workflow, not an emission-rate retrieval.  The
MODTRAN concentration parameter is only equivalent to the supplied simulation
geometry, and the empirical local z maps are not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile

from hisui_l1g_io import (
    GeoReference,
    HISUIL1GReader,
    HISUIL1GProduct,
    discover_l1g_products,
    parse_metadata,
    read_band_metadata,
    read_georeference,
    read_tiled_any_mask,
)
from iterative_mf_1600nm import (
    compute_uas_log_slope,
    gaussian_srf_resample,
    load_ch4_lut,
)
from physics_aware_crossfit_mf import (
    HashReservoir,
    NuisanceModelSet,
    SceneScan,
    build_nuisance_models,
    make_continuum_transform,
    score_log_radiance,
    spatial_fold_ids,
)


QUALITY_WAVELENGTHS_NM = np.asarray([665.0, 865.0, 1388.0, 1650.0, 2200.0])
CLOUD_PROFILES: dict[str, dict[str, float]] = {
    # Sensitive to thin/cirrus cloud while rejecting spectrally red bright soil.
    "cirrus_sensitive": {
        "cirrus_1388": 0.035,
        "cirrus_865": 0.18,
        "bright_665": 0.28,
        "bright_865": 0.30,
        "bright_1650": 0.18,
        "bright_neutrality": 0.12,
    },
    # Three browse-checked sensitivity levels for bright desert scenes.
    "desert_sensitive": {
        "cirrus_1388": 0.08,
        "cirrus_865": 0.20,
        "bright_665": 0.40,
        "bright_865": 0.40,
        "bright_1650": 0.25,
        "bright_neutrality": 0.15,
    },
    "desert_balanced": {
        "cirrus_1388": 0.10,
        "cirrus_865": 0.22,
        "bright_665": 0.45,
        "bright_865": 0.45,
        "bright_1650": 0.28,
        "bright_neutrality": 0.12,
    },
    "desert_core": {
        "cirrus_1388": 0.12,
        "cirrus_865": 0.25,
        "bright_665": 0.50,
        "bright_865": 0.50,
        "bright_1650": 0.30,
        "bright_neutrality": 0.10,
    },
}
SCORE_KEYS = (
    "combined_z",
    "combined_alpha",
    "weak_z",
    "weak_alpha",
    "strong_z",
    "strong_alpha",
    "dual_min_z",
    "log_e_strong_to_weak",
    "log_e_weak_to_strong",
    "log_e_average",
    "log_e_negative_control",
)
OFFICIALLY_SCORED_QUALITY_CLASSES = frozenset({"usable", "partial"})
QUALITY_CLASS_COLORS = {
    "usable": "#4c9f70",
    "partial": "#d69e2e",
    "excluded_cloud": "#c44e52",
    "qa_incomplete": "#777777",
}
SCENE_OUTPUT_ARTIFACT_NAMES = frozenset(
    {
        "candidate_components.csv",
        "known_site_metrics.csv",
        "score_maps.npz",
        "overview.png",
        "summary.json",
    }
)


@dataclass(frozen=True)
class BandPlan:
    detection_bands: np.ndarray
    quality_bands: np.ndarray
    read_bands: np.ndarray
    detection_positions: np.ndarray
    quality_positions: np.ndarray
    wavelengths_nm: np.ndarray
    fwhm_nm: np.ndarray
    weak_indices: np.ndarray
    strong_indices: np.ndarray
    reflectance_multiplier: np.ndarray
    reflectance_additive: np.ndarray


@dataclass(frozen=True)
class KnownSite:
    site_id: str
    name: str
    epsg: int
    easting_m: float
    northing_m: float


@dataclass
class SceneResult:
    summary: dict[str, object]
    dual_thumbnail: np.ndarray | None
    cloud_thumbnail: np.ndarray
    site_rows: list[dict[str, object]]
    candidate_rows: list[dict[str, object]]


def _read_band_table(product: HISUIL1GProduct) -> pd.DataFrame:
    table = pd.read_csv(product.band_csv_path, skipinitialspace=True)
    table.columns = [str(column).strip() for column in table.columns]
    required = {
        "BandNo",
        "CenterWavelengthNanometer",
        "FullWidthAtHalfMaximumNanometer",
        "ReflectanceMulti",
        "ReflectanceAdd",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            f"{product.band_csv_path} lacks required columns: {', '.join(missing)}"
        )
    return table


def make_band_plan(
    product: HISUIL1GProduct,
    *,
    weak_window_nm: tuple[float, float],
    strong_window_nm: tuple[float, float],
) -> BandPlan:
    table = _read_band_table(product)
    wavelengths = table["CenterWavelengthNanometer"].to_numpy(dtype=float)
    fwhm = table["FullWidthAtHalfMaximumNanometer"].to_numpy(dtype=float)
    weak_mask = (wavelengths >= weak_window_nm[0]) & (
        wavelengths <= weak_window_nm[1]
    )
    strong_mask = (wavelengths >= strong_window_nm[0]) & (
        wavelengths <= strong_window_nm[1]
    )
    detection = np.flatnonzero(weak_mask | strong_mask)
    if weak_mask.sum() < 3 or strong_mask.sum() < 3:
        raise ValueError("Each methane window must contain at least three HISUI bands")
    quality = np.asarray(
        [int(np.argmin(np.abs(wavelengths - value))) for value in QUALITY_WAVELENGTHS_NM],
        dtype=int,
    )
    read_bands = np.unique(np.concatenate([detection, quality]))
    position = {int(band): index for index, band in enumerate(read_bands)}
    detection_positions = np.asarray([position[int(band)] for band in detection])
    quality_positions = np.asarray([position[int(band)] for band in quality])
    selected_wavelengths = wavelengths[detection]
    return BandPlan(
        detection_bands=detection,
        quality_bands=quality,
        read_bands=read_bands,
        detection_positions=detection_positions,
        quality_positions=quality_positions,
        wavelengths_nm=selected_wavelengths,
        fwhm_nm=fwhm[detection],
        weak_indices=np.flatnonzero(
            (selected_wavelengths >= weak_window_nm[0])
            & (selected_wavelengths <= weak_window_nm[1])
        ),
        strong_indices=np.flatnonzero(
            (selected_wavelengths >= strong_window_nm[0])
            & (selected_wavelengths <= strong_window_nm[1])
        ),
        reflectance_multiplier=table["ReflectanceMulti"].to_numpy(dtype=float)[quality],
        reflectance_additive=table["ReflectanceAdd"].to_numpy(dtype=float)[quality],
    )


def invalid_dn_values(
    metadata: dict[str, object], *, include_saturation: bool = True
) -> np.ndarray:
    values = {0}
    keys = ["BadPixelDN"]
    if include_saturation:
        keys.append("SaturatedPixelDN")
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and float(value).is_integer():
            values.add(int(value))
    return np.asarray(sorted(values), dtype=np.uint16)


def saturated_dn_values(metadata: dict[str, object]) -> np.ndarray:
    value = metadata.get("SaturatedPixelDN")
    if isinstance(value, (int, float)) and float(value).is_integer():
        return np.asarray([int(value)], dtype=np.uint16)
    return np.asarray([], dtype=np.uint16)


def cloud_proxy_from_quality_dn(
    quality_dn: np.ndarray,
    multiplier: np.ndarray,
    additive: np.ndarray,
    invalid_values: np.ndarray,
    *,
    saturated_values: np.ndarray | Sequence[int] = (),
    qa_affected: np.ndarray | None = None,
    profile: str = "cirrus_sensitive",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return valid pixels and a conservative cloud/cirrus proxy.

    This is intentionally called a proxy: HISUI L1G does not provide a
    validated cloud mask.  Bright VIS/NIR/SWIR clouds and elevated 1.388 um
    cirrus are combined using fixed TOA-reflectance thresholds.
    """

    if profile not in CLOUD_PROFILES:
        raise ValueError(f"Unknown cloud profile: {profile}")
    thresholds = CLOUD_PROFILES[profile]
    quality_dn = np.asarray(quality_dn)
    invalid = np.any(np.isin(quality_dn, invalid_values), axis=-1)
    if qa_affected is None:
        qa_affected_array = np.zeros(quality_dn.shape[:-1], dtype=bool)
    else:
        qa_affected_array = np.asarray(qa_affected, dtype=bool)
        if qa_affected_array.shape != quality_dn.shape[:-1]:
            raise ValueError(
                "Quality QA mask shape "
                f"{qa_affected_array.shape} differs from DN shape {quality_dn.shape[:-1]}"
            )
        invalid |= qa_affected_array
    saturated_values = np.asarray(saturated_values)
    saturated = (
        np.any(np.isin(quality_dn, saturated_values), axis=-1)
        if saturated_values.size
        else np.zeros(quality_dn.shape[:-1], dtype=bool)
    )
    reflectance = quality_dn.astype(np.float32)
    reflectance *= multiplier.astype(np.float32)
    reflectance += additive.astype(np.float32)
    r665, r865, r1388, r1650, _r2200 = np.moveaxis(reflectance, -1, 0)
    visible_nir_mean = np.maximum((r665 + r865) / 2.0, 1e-6)
    neutrality = np.abs(r665 - r865) / visible_nir_mean
    bright = (
        (r665 > thresholds["bright_665"])
        & (r865 > thresholds["bright_865"])
        & (r1650 > thresholds["bright_1650"])
        & (neutrality < thresholds["bright_neutrality"])
    )
    cirrus = (r1388 > thresholds["cirrus_1388"]) & (
        r865 > thresholds["cirrus_865"]
    )
    cloud = (bright | cirrus | saturated) & ~invalid
    valid = ~invalid
    denominator = max(int(valid.sum()), 1)
    stats = {
        "cloud_proxy_fraction": float(cloud.sum() / denominator),
        "bright_cloud_fraction": float((bright & valid).sum() / denominator),
        "cirrus_proxy_fraction": float((cirrus & valid).sum() / denominator),
        "saturated_quality_fraction": float(
            (saturated & valid).sum() / denominator
        ),
        "quality_valid_pixels": int(valid.sum()),
        "quality_qa_affected_pixels": int(qa_affected_array.sum()),
        "cloud_profile": profile,
        "cloud_thresholds": dict(thresholds),
    }
    return valid, cloud, stats


def detector_qa_mask(
    product: HISUIL1GProduct,
    metadata: dict[str, object],
    bands: Sequence[int],
    *,
    bounds: tuple[int, int, int, int] | None = None,
    include_interpolated: bool = True,
    stats_prefix: str = "detector_qa",
) -> tuple[np.ndarray, dict[str, object]]:
    """Return pixels affected in any selected band by delivered detector QA."""

    if bounds is None:
        with tifffile.TiffFile(str(product.image_path)) as tif:
            shape = (int(tif.pages[0].imagelength), int(tif.pages[0].imagewidth))
    else:
        y0, y1, x0, x1 = bounds
        shape = (y1 - y0, x1 - x0)
    combined = np.zeros(shape, dtype=bool)
    rows: dict[str, object] = {f"{stats_prefix}_missing": []}
    fields = [("dead_corrected", "QADeadPixelMapFileName")]
    if include_interpolated:
        fields.append(("bad_interpolated", "QAInterpolatedPixelMapFileName"))
    for label, metadata_key in fields:
        filename = metadata.get(metadata_key)
        path = product.directory / str(filename) if isinstance(filename, str) else None
        if path is None or not path.is_file():
            cast_missing = rows[f"{stats_prefix}_missing"]
            assert isinstance(cast_missing, list)
            cast_missing.append(metadata_key)
            rows[f"{stats_prefix}_{label}_pixels"] = None
            continue
        affected = read_tiled_any_mask(path, bands, bounds=bounds)
        rows[f"{stats_prefix}_{label}_pixels"] = int(affected.sum())
        combined |= affected
    rows[f"{stats_prefix}_masked_pixels"] = int(combined.sum())
    rows[f"{stats_prefix}_interpolated_included"] = bool(include_interpolated)
    rows[f"{stats_prefix}_complete"] = not bool(rows[f"{stats_prefix}_missing"])
    return combined, rows


def dilate_cloud_mask(cloud_mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels < 0:
        raise ValueError("cloud dilation pixels must be non-negative")
    cloud = np.asarray(cloud_mask, dtype=bool)
    if pixels == 0:
        return cloud.copy()
    return ndimage.binary_dilation(cloud, iterations=int(pixels))


def _progressive_tile_strides(initial_stride: int) -> tuple[int, ...]:
    """Return decreasing tile strides that always terminate at full coverage."""

    if initial_stride <= 0:
        raise ValueError("preflight tile stride must be positive")
    strides = [int(initial_stride)]
    while strides[-1] > 1:
        next_stride = max(1, strides[-1] // 2)
        if next_stride == strides[-1]:
            break
        strides.append(next_stride)
    if strides[-1] != 1:
        strides.append(1)
    return tuple(strides)


def sparse_cloud_preflight(
    product: HISUIL1GProduct,
    plan: BandPlan,
    *,
    tile_stride: int,
    cloud_profile: str,
    confirmation_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Estimate scene cloud cover from regularly spaced native TIFF tiles."""

    if tile_stride <= 0:
        raise ValueError("preflight tile stride must be positive")
    metadata = parse_metadata(product.metadata_path)
    invalid_values = invalid_dn_values(metadata, include_saturation=False)
    saturated_values = saturated_dn_values(metadata)
    quality_qa, quality_qa_stats = detector_qa_mask(
        product,
        metadata,
        plan.quality_bands,
        include_interpolated=True,
        stats_prefix="quality_qa",
    )
    with HISUIL1GReader(product) as reader:
        if quality_qa.shape != (reader.height, reader.width):
            raise ValueError(
                f"Quality QA shape {quality_qa.shape} differs from image "
                f"{(reader.height, reader.width)}"
            )
        cloud_grid = np.full((reader.tile_rows, reader.tile_columns), np.nan, dtype=float)
        valid_grid = np.full_like(cloud_grid, np.nan)
        sampled_tiles = np.zeros_like(cloud_grid, dtype=bool)
        cloud_pixels = 0
        valid_pixels = 0
        sampled_pixels = 0
        evaluated_strides: list[int] = []
        level_cloud_fractions: list[float | None] = []
        level_new_tile_counts: list[int] = []
        for current_stride in _progressive_tile_strides(tile_stride):
            new_tile_count = 0
            for tile_row in range(0, reader.tile_rows, current_stride):
                y0 = tile_row * reader.tile_height
                y1 = min(y0 + reader.tile_height, reader.height)
                for tile_column in range(0, reader.tile_columns, current_stride):
                    if sampled_tiles[tile_row, tile_column]:
                        continue
                    x0 = tile_column * reader.tile_width
                    x1 = min(x0 + reader.tile_width, reader.width)
                    dn = reader.read_region(y0, y1, x0, x1, plan.quality_bands)
                    valid, cloud, _ = cloud_proxy_from_quality_dn(
                        dn,
                        plan.reflectance_multiplier,
                        plan.reflectance_additive,
                        invalid_values,
                        saturated_values=saturated_values,
                        qa_affected=quality_qa[y0:y1, x0:x1],
                        profile=cloud_profile,
                    )
                    sampled_tiles[tile_row, tile_column] = True
                    new_tile_count += 1
                    sampled_pixels += int(valid.size)
                    valid_count = int(valid.sum())
                    cloud_count = int(cloud.sum())
                    valid_pixels += valid_count
                    cloud_pixels += cloud_count
                    valid_grid[tile_row, tile_column] = valid_count / max(valid.size, 1)
                    cloud_grid[tile_row, tile_column] = cloud_count / max(valid_count, 1)
            evaluated_strides.append(int(current_stride))
            level_new_tile_counts.append(new_tile_count)
            cumulative_fraction = (
                float(cloud_pixels / valid_pixels) if valid_pixels else None
            )
            level_cloud_fractions.append(cumulative_fraction)
            if confirmation_threshold is None:
                break
            if cumulative_fraction is not None and cumulative_fraction <= confirmation_threshold:
                # A below-threshold sample cannot cause permanent early
                # exclusion. The full-resolution quality mask is computed next.
                break
            # If the sampled estimate remains above threshold (or contains no
            # valid pixels), continue until stride 1 guarantees every tile was
            # inspected before an early cloud exclusion is allowed.

        sampled = np.isfinite(cloud_grid)
        if not sampled.any() or valid_pixels == 0:
            raise ValueError("Sparse cloud preflight found no valid footprint pixels")
        _, nearest = ndimage.distance_transform_edt(~sampled, return_indices=True)
        cloud_tiles = cloud_grid[tuple(nearest)]
        valid_tiles = valid_grid[tuple(nearest)]
        cloud_map = np.repeat(
            np.repeat(cloud_tiles >= 0.25, reader.tile_height, axis=0),
            reader.tile_width,
            axis=1,
        )[: reader.height, : reader.width]
        valid_map = np.repeat(
            np.repeat(valid_tiles >= 0.05, reader.tile_height, axis=0),
            reader.tile_width,
            axis=1,
        )[: reader.height, : reader.width]
    full_coverage = bool(sampled_tiles.all())
    phase_fractions = [
        float(value) for value in level_cloud_fractions if value is not None
    ]
    return valid_map, cloud_map, {
        "quality_estimation": "sparse_native_tile_preflight",
        "preflight_tile_stride": int(tile_stride),
        "preflight_sampled_pixels": int(sampled_pixels),
        "preflight_valid_pixels": int(valid_pixels),
        "cloud_proxy_fraction": float(cloud_pixels / valid_pixels),
        "cloud_profile": cloud_profile,
        "preflight_evaluated_strides": evaluated_strides,
        "preflight_level_new_tile_counts": level_new_tile_counts,
        "preflight_level_cumulative_cloud_fractions": level_cloud_fractions,
        "preflight_full_coverage": full_coverage,
        # Retained for compatibility with summaries written by the preceding
        # stratified-phase implementation.
        "preflight_phase_offsets": [[0, 0]],
        "preflight_phase_cloud_fractions": phase_fractions,
        "preflight_phase_min_cloud_fraction": float(min(phase_fractions)),
        "preflight_phase_max_cloud_fraction": float(max(phase_fractions)),
        "qa_complete": bool(quality_qa_stats["quality_qa_complete"]),
        **quality_qa_stats,
    }


def quality_class(cloud_fraction: float, *, maximum_for_scoring: float) -> str:
    if not 0.0 <= maximum_for_scoring <= 1.0:
        raise ValueError("maximum cloud fraction must lie in [0, 1]")
    if cloud_fraction > maximum_for_scoring:
        return "excluded_cloud"
    if cloud_fraction <= 0.10:
        return "usable"
    return "partial"


def scene_quality_class(
    cloud_fraction: float,
    *,
    maximum_for_scoring: float,
    qa_complete: bool,
) -> str:
    """Classify a scene, giving missing delivered QA precedence over cloud."""

    if not qa_complete:
        return "qa_incomplete"
    return quality_class(
        cloud_fraction, maximum_for_scoring=maximum_for_scoring
    )


def is_officially_scored(classification: str) -> bool:
    """Return whether a scene may contribute to official candidate outputs."""

    return classification in OFFICIALLY_SCORED_QUALITY_CLASSES


def _radiance_coefficients(
    metadata: dict[str, object], band_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vnir = np.asarray(band_indices) < 58
    multiplier = np.where(
        vnir,
        float(metadata["RadianceMultiVNIR"]),
        float(metadata["RadianceMultiSWIR"]),
    ).astype(np.float32)
    additive = np.where(
        vnir,
        float(metadata["RadianceAddVNIR"]),
        float(metadata["RadianceAddSWIR"]),
    ).astype(np.float32)
    return multiplier, additive


def prepare_scene_arrays(
    product: HISUIL1GProduct,
    plan: BandPlan,
    *,
    cloud_dilation_pixels: int,
    cloud_profile: str,
    tiff_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    metadata = parse_metadata(product.metadata_path)
    cube = tifffile.imread(str(product.image_path), maxworkers=tiff_workers)
    if cube.ndim != 3 or cube.shape[-1] != 185:
        raise ValueError(f"Expected YXS 185-band cube, found {cube.shape}")
    selected_dn = np.take(cube, plan.read_bands, axis=2)
    del cube
    gc.collect()

    invalid_values = invalid_dn_values(metadata, include_saturation=False)
    saturated_values = saturated_dn_values(metadata)
    quality_dn = selected_dn[..., plan.quality_positions]
    quality_qa, quality_qa_stats = detector_qa_mask(
        product,
        metadata,
        plan.quality_bands,
        include_interpolated=True,
        stats_prefix="quality_qa",
    )
    if quality_qa.shape != quality_dn.shape[:-1]:
        raise ValueError(
            f"Quality QA shape {quality_qa.shape} differs from image "
            f"{quality_dn.shape[:-1]}"
        )
    quality_valid, cloud_raw, quality_stats = cloud_proxy_from_quality_dn(
        quality_dn,
        plan.reflectance_multiplier,
        plan.reflectance_additive,
        invalid_values,
        saturated_values=saturated_values,
        qa_affected=quality_qa,
        profile=cloud_profile,
    )
    cloud = dilate_cloud_mask(cloud_raw, cloud_dilation_pixels)

    detection_dn = selected_dn[..., plan.detection_positions]
    all_invalid_values = invalid_dn_values(metadata, include_saturation=True)
    detection_valid = ~np.any(
        np.isin(detection_dn, all_invalid_values), axis=-1
    )
    detector_qa, detector_qa_stats = detector_qa_mask(
        product,
        metadata,
        plan.detection_bands,
        include_interpolated=True,
    )
    if detector_qa.shape != detection_valid.shape:
        raise ValueError(
            f"Detector QA shape {detector_qa.shape} differs from image {detection_valid.shape}"
        )
    detection_valid &= ~detector_qa
    multiplier, additive = _radiance_coefficients(metadata, plan.detection_bands)
    radiance = detection_dn.astype(np.float32)
    radiance *= multiplier
    radiance += additive
    positive = np.all(np.isfinite(radiance) & (radiance > 0.0), axis=-1)
    clear = quality_valid & detection_valid & positive & ~cloud
    radiance[~clear] = np.nan
    del selected_dn, detection_dn, quality_dn
    gc.collect()

    quality_stats.update(
        {
            "cloud_dilated_fraction": float(
                (cloud & quality_valid).sum() / max(int(quality_valid.sum()), 1)
            ),
            "analysis_clear_pixels": int(clear.sum()),
            "analysis_clear_fraction_of_quality_valid": float(
                clear.sum() / max(int(quality_valid.sum()), 1)
            ),
            "qa_complete": bool(
                quality_qa_stats["quality_qa_complete"]
                and detector_qa_stats["detector_qa_complete"]
            ),
            **quality_qa_stats,
            **detector_qa_stats,
        }
    )
    return radiance, clear, cloud, quality_stats


def make_scene_scan(
    radiance: np.ndarray,
    valid_map: np.ndarray,
    *,
    sample_size: int,
    block_rows: int,
) -> SceneScan:
    reservoir = HashReservoir(sample_size)
    height, width = valid_map.shape
    for y0 in range(0, height, block_rows):
        y1 = min(y0 + block_rows, height)
        selected = valid_map[y0:y1]
        if not selected.any():
            continue
        yy, xx = np.nonzero(selected)
        values = radiance[y0:y1][selected]
        reservoir.update(values, yy + y0, xx)
    if reservoir.values.size == 0:
        raise ValueError("No clear positive-radiance pixels remain after quality screening")
    return SceneScan(
        sample_radiance=reservoir.values,
        sample_y=reservoir.y,
        sample_x=reservoir.x,
        valid_pixels=int(valid_map.sum()),
        total_rows=int(height * width),
        y_min=0,
        y_max=height - 1,
        x_min=0,
        x_max=width - 1,
    )


def score_radiance_array(
    radiance: np.ndarray,
    valid_map: np.ndarray,
    models: NuisanceModelSet,
    feature_transform: np.ndarray,
    *,
    block_rows: int,
    spatial_block_size: int,
) -> dict[str, np.ndarray]:
    height, width = valid_map.shape
    maps = {
        key: np.full((height, width), np.nan, dtype=np.float32) for key in SCORE_KEYS
    }
    for y0 in range(0, height, block_rows):
        y1 = min(y0 + block_rows, height)
        selected = valid_map[y0:y1]
        if not selected.any():
            continue
        yy, xx = np.nonzero(selected)
        yy = yy + y0
        log_values = np.log(radiance[y0:y1][selected]) @ feature_transform.T
        folds = spatial_fold_ids(
            yy, xx, n_folds=models.n_folds, block_size=spatial_block_size
        )
        if models.tile_size > 0:
            y_tiles = np.floor_divide(yy, models.tile_size)
            x_tiles = np.floor_divide(xx, models.tile_size)
        else:
            y_tiles = np.zeros(len(yy), dtype=int)
            x_tiles = np.zeros(len(xx), dtype=int)
        groups = np.unique(np.column_stack([y_tiles, x_tiles, folds]), axis=0)
        for y_tile, x_tile, fold in groups:
            use = (y_tiles == y_tile) & (x_tiles == x_tile) & (folds == fold)
            model = models.get(int(y_tile), int(x_tile), int(fold))
            scores = score_log_radiance(log_values[use], model)
            for key in SCORE_KEYS:
                maps[key][yy[use], xx[use]] = scores[key].astype(np.float32)
    return maps


def weighted_local_z(
    image: np.ndarray,
    valid_map: np.ndarray,
    *,
    sigma_pixels: float,
) -> np.ndarray:
    valid = np.asarray(valid_map, dtype=bool) & np.isfinite(image)
    weights = ndimage.gaussian_filter(
        valid.astype(np.float32), sigma=sigma_pixels, mode="nearest"
    )
    values = np.where(valid, image, 0.0).astype(np.float32)
    mean = ndimage.gaussian_filter(values, sigma=sigma_pixels, mode="nearest")
    mean = np.divide(mean, weights, out=np.zeros_like(mean), where=weights > 1e-6)
    second = ndimage.gaussian_filter(
        values * values, sigma=sigma_pixels, mode="nearest"
    )
    second = np.divide(
        second, weights, out=np.zeros_like(second), where=weights > 1e-6
    )
    variance = np.maximum(second - mean * mean, 1e-4)
    output = (np.asarray(image, dtype=np.float32) - mean) / np.sqrt(variance)
    output[~valid] = np.nan
    return output


def directional_profile_destripe(
    image: np.ndarray,
    valid_map: np.ndarray,
    *,
    profiles: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Subtract additive means along fixed sensor-line directions.

    The two default directions come from the prior QA_DM analysis of this
    Permian product family (thin trace slope about 0.978 and broad cross-track
    slope about 1.267).  Values are winsorized before profile estimation so a
    compact candidate cannot define the line correction.  This is a scalable
    full-scene analogue of the earlier fixed-slope median/DWT workflow, not a
    replacement for product-specific QA validation.
    """

    corrected = np.asarray(image, dtype=np.float32).copy()
    valid = np.asarray(valid_map, dtype=bool) & np.isfinite(corrected)
    height, width = corrected.shape
    row_coordinate = np.arange(height, dtype=np.float32)[:, np.newaxis]
    column_coordinate = np.arange(width, dtype=np.float32)[np.newaxis, :]
    diagnostics: list[dict[str, float]] = []
    for slope, bin_width in profiles:
        if bin_width <= 0:
            raise ValueError("Directional profile bin width must be positive")
        coordinate = row_coordinate - float(slope) * column_coordinate
        line_ids = np.rint(coordinate / float(bin_width)).astype(np.int32)
        minimum = int(line_ids[valid].min())
        ids = line_ids[valid].astype(np.int64) - minimum
        line_count = int(ids.max()) + 1
        values = corrected[valid].astype(float)
        lower, upper = np.quantile(values, [0.02, 0.98])
        winsorized = np.clip(values, lower, upper)
        counts = np.bincount(ids, minlength=line_count).astype(float)
        sums = np.bincount(ids, weights=winsorized, minlength=line_count)
        profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        global_mean = float(np.sum(sums) / max(float(np.sum(counts)), 1.0))
        offsets = profile - global_mean
        before = float(np.sqrt(np.average(offsets[counts > 0] ** 2, weights=counts[counts > 0])))
        corrected_values = values - offsets[ids]
        corrected[valid] = corrected_values.astype(np.float32)

        new_sums = np.bincount(ids, weights=np.clip(corrected_values, lower, upper), minlength=line_count)
        new_profile = np.divide(
            new_sums, counts, out=np.zeros_like(new_sums), where=counts > 0
        )
        new_global = float(np.sum(new_sums) / max(float(np.sum(counts)), 1.0))
        after_offsets = new_profile - new_global
        after = float(
            np.sqrt(
                np.average(
                    after_offsets[counts > 0] ** 2, weights=counts[counts > 0]
                )
            )
        )
        diagnostics.append(
            {
                "slope_row_per_column": float(slope),
                "line_bin_width_pixels": float(bin_width),
                "winsorized_profile_rms_before": before,
                "winsorized_profile_rms_after": after,
            }
        )
    corrected[~valid] = np.nan
    return corrected, diagnostics


def _component_rows(
    dual_z: np.ndarray,
    weak_z: np.ndarray,
    strong_z: np.ndarray,
    valid_map: np.ndarray,
    cloud_mask: np.ndarray,
    georef: GeoReference,
    *,
    threshold: float,
    minimum_pixels: int,
    sign: str,
) -> list[dict[str, object]]:
    candidate = valid_map & np.isfinite(dual_z) & (dual_z >= threshold)
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=int))
    objects = ndimage.find_objects(labels, max_label=count)
    cloud_distance = (
        ndimage.distance_transform_edt(~cloud_mask) if cloud_mask.any() else None
    )
    rows: list[dict[str, object]] = []
    for label_id, section in enumerate(objects, start=1):
        if section is None:
            continue
        selected = labels[section] == label_id
        size = int(selected.sum())
        if size < minimum_pixels:
            continue
        local_values = dual_z[section]
        local_peak_flat = int(np.nanargmax(np.where(selected, local_values, np.nan)))
        local_peak = np.unravel_index(local_peak_flat, local_values.shape)
        peak_y = int(section[0].start + local_peak[0])
        peak_x = int(section[1].start + local_peak[1])
        local_y, local_x = np.nonzero(selected)
        yy = local_y + int(section[0].start)
        xx = local_x + int(section[1].start)
        height = int(section[0].stop - section[0].start)
        width = int(section[1].stop - section[1].start)
        coordinates = np.column_stack([yy, xx]).astype(float)
        if len(coordinates) >= 3:
            eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
            elongation = float(
                np.sqrt(max(float(eigenvalues[-1]), 0.0) / max(float(eigenvalues[0]), 1e-6))
            )
        else:
            elongation = 1.0
        easting, northing = georef.map_xy(peak_y, peak_x, pixel_center=True)
        rows.append(
            {
                "tail": sign,
                "component_id": int(label_id),
                "pixel_count": size,
                "peak_y": peak_y,
                "peak_x": peak_x,
                "centroid_y": float(yy.mean()),
                "centroid_x": float(xx.mean()),
                "bbox_height": height,
                "bbox_width": width,
                "dual_local_z_peak": float(dual_z[peak_y, peak_x]),
                "dual_local_z_mean": float(np.mean(local_values[selected])),
                "weak_local_z_at_peak": float(weak_z[peak_y, peak_x]),
                "strong_local_z_at_peak": float(strong_z[peak_y, peak_x]),
                "distance_to_cloud_pixels": (
                    float(cloud_distance[peak_y, peak_x])
                    if cloud_distance is not None
                    else None
                ),
                "easting_m": float(easting),
                "northing_m": float(northing),
                "epsg": georef.epsg,
                "component_elongation": elongation,
                "scene_spanning_line_flag": bool(size >= 100 and elongation >= 10.0),
            }
        )
    return sorted(rows, key=lambda row: float(row["dual_local_z_peak"]), reverse=True)


def load_known_sites(path: Path | None) -> list[KnownSite]:
    if path is None:
        return []
    table = pd.read_csv(path)
    required = {"site_id", "name", "epsg", "easting_m", "northing_m"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Known-site CSV lacks: {', '.join(missing)}")
    return [
        KnownSite(
            site_id=str(row.site_id),
            name=str(row.name),
            epsg=int(row.epsg),
            easting_m=float(row.easting_m),
            northing_m=float(row.northing_m),
        )
        for row in table.itertuples(index=False)
    ]


def projected_to_pixel(
    georef: GeoReference, easting: float, northing: float
) -> tuple[float, float]:
    x0, dx, rx, y0, ry, dy = georef.geotransform
    matrix = np.asarray([[dx, rx], [ry, dy]], dtype=float)
    column_line = np.linalg.solve(
        matrix, np.asarray([easting - x0, northing - y0], dtype=float)
    )
    return float(column_line[1] - 0.5), float(column_line[0] - 0.5)


def site_metrics(
    sites: Sequence[KnownSite],
    georef: GeoReference,
    shape: tuple[int, int],
    maps: dict[str, np.ndarray],
    weak_local: np.ndarray,
    strong_local: np.ndarray,
    dual_local: np.ndarray,
    valid_map: np.ndarray,
    *,
    radius_pixels: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if georef.epsg is None:
        return rows
    for site in sites:
        if site.epsg != georef.epsg:
            continue
        row, column = projected_to_pixel(georef, site.easting_m, site.northing_m)
        if not (-0.5 <= row < shape[0] - 0.5 and -0.5 <= column < shape[1] - 0.5):
            continue
        cy, cx = int(round(row)), int(round(column))
        y0, y1 = max(cy - radius_pixels, 0), min(cy + radius_pixels + 1, shape[0])
        x0, x1 = max(cx - radius_pixels, 0), min(cx + radius_pixels + 1, shape[1])
        window_valid = valid_map[y0:y1, x0:x1]
        row_out: dict[str, object] = {
            "site_id": site.site_id,
            "site_name": site.name,
            "pixel_y": row,
            "pixel_x": column,
            "window_radius_pixels": radius_pixels,
            "window_clear_fraction": float(window_valid.mean()),
            "window_clear_pixels": int(window_valid.sum()),
        }
        for name, image in (
            ("weak_z", maps["weak_z"]),
            ("strong_z", maps["strong_z"]),
            ("dual_min_z", maps["dual_min_z"]),
            ("weak_local_z", weak_local),
            ("strong_local_z", strong_local),
            ("dual_local_z", dual_local),
            ("log_e_average", maps["log_e_average"]),
        ):
            values = image[y0:y1, x0:x1][window_valid]
            row_out[f"{name}_mean"] = float(np.mean(values)) if len(values) else None
            row_out[f"{name}_max"] = float(np.max(values)) if len(values) else None
        rows.append(row_out)
    return rows


def _browse1_path(product: HISUIL1GProduct) -> Path | None:
    candidate = product.directory / f"{product.product_id}_1.jpg"
    return candidate if candidate.is_file() else None


def save_scene_overview(
    path: Path,
    product: HISUIL1GProduct,
    cloud_mask: np.ndarray,
    valid_map: np.ndarray,
    weak_local: np.ndarray | None,
    strong_local: np.ndarray | None,
    dual_local: np.ndarray | None,
    candidates: Sequence[dict[str, object]],
    site_rows: Sequence[dict[str, object]],
    *,
    threshold: float,
    classification: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    browse = _browse1_path(product)
    if browse is not None:
        axes[0, 0].imshow(mpimg.imread(browse))
    else:
        axes[0, 0].text(0.5, 0.5, "No browse image", ha="center", va="center")
    axes[0, 0].set_title("HISUI browse image")
    axes[0, 0].set_axis_off()

    shown_cloud = np.zeros((*cloud_mask.shape, 3), dtype=np.float32)
    shown_cloud[..., 0] = cloud_mask
    shown_cloud[..., 1] = valid_map
    axes[0, 1].imshow(shown_cloud)
    axes[0, 1].set_title("Cloud proxy (red); clear analysis pixels (green)")
    axes[0, 1].set_axis_off()

    if dual_local is None:
        skipped_reason = (
            "delivered QA incomplete"
            if classification == "qa_incomplete"
            else "cloud-dominated"
        )
        for axis in (axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]):
            axis.text(
                0.5,
                0.5,
                f"MF skipped: {skipped_reason}",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
    else:
        panels = [
            (axes[0, 2], weak_local, "1.65 um destriped local MF z"),
            (axes[1, 0], strong_local, "2.3 um destriped local MF z"),
            (axes[1, 1], dual_local, "Dual-band destriped min local z"),
        ]
        for axis, image, title in panels:
            displayed = axis.imshow(image, cmap="coolwarm", vmin=-3.0, vmax=6.0)
            axis.set_title(title)
            axis.set_axis_off()
            fig.colorbar(displayed, ax=axis, shrink=0.72)
        axes[1, 1].contour(
            np.isfinite(dual_local) & (dual_local >= threshold),
            levels=[0.5],
            colors="yellow",
            linewidths=0.5,
        )
        top = list(candidates)[:20]
        if top:
            axes[1, 1].scatter(
                [float(row["peak_x"]) for row in top],
                [float(row["peak_y"]) for row in top],
                s=18,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
            )
        reverse = np.minimum(-weak_local, -strong_local)
        shown = axes[1, 2].imshow(reverse, cmap="coolwarm", vmin=-3.0, vmax=6.0)
        axes[1, 2].set_title("Reverse-sign dual-tail control")
        axes[1, 2].set_axis_off()
        fig.colorbar(shown, ax=axes[1, 2], shrink=0.72)
        for site in site_rows:
            for axis in (axes[0, 2], axes[1, 0], axes[1, 1]):
                axis.plot(
                    float(site["pixel_x"]),
                    float(site["pixel_y"]),
                    marker="+",
                    color="lime",
                    markersize=12,
                    markeredgewidth=1.5,
                )
    fig.suptitle(f"{product.product_id} | quality: {classification}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _template_for_scene(
    plan: BandPlan,
    modtran_path: Path,
    *,
    uas_alpha_min: float,
    uas_alpha_max: float,
    continuum_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mod_wave, alpha_grid, mod_radiance = load_ch4_lut(modtran_path)
    resampled = gaussian_srf_resample(
        mod_wave, mod_radiance, plan.wavelengths_nm, plan.fwhm_nm
    )
    uas = compute_uas_log_slope(
        alpha_grid,
        resampled,
        alpha_min=uas_alpha_min,
        alpha_max=uas_alpha_max,
    )
    raw_target = -uas
    transform, weak_features, strong_features = make_continuum_transform(
        plan.wavelengths_nm,
        plan.weak_indices,
        plan.strong_indices,
        degree=continuum_degree,
    )
    return (
        transform,
        transform @ raw_target,
        weak_features,
        strong_features,
        alpha_grid,
        resampled,
    )


def analysis_config(args: argparse.Namespace) -> dict[str, object]:
    keys = (
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
        "maximum_cloud_fraction",
        "cloud_dilation_pixels",
        "cloud_profile",
        "preflight_tile_stride",
        "disable_cloud_preflight",
        "local_z_sigma",
        "enable_directional_destriping",
        "thin_stripe_slope",
        "thin_stripe_bin_width",
        "broad_stripe_slope",
        "broad_stripe_bin_width",
        "cluster_threshold",
        "minimum_cluster_pixels",
        "site_radius_pixels",
        "tiff_workers",
        "save_score_maps",
    )
    config = {key: getattr(args, key) for key in keys}
    config["cloud_thresholds"] = dict(CLOUD_PROFILES[args.cloud_profile])
    return config


def _clear_known_artifacts(directory: Path, names: Sequence[str]) -> None:
    for name in names:
        (directory / name).unlink(missing_ok=True)


def _find_stale_scene_directories(
    output_dir: Path, current_product_ids: Sequence[str]
) -> list[Path]:
    """Return prior per-scene outputs that are not members of this batch.

    The directories are deliberately left in place.  A HISUI-looking directory
    is included even when a previous run stopped before writing an artifact;
    arbitrary auxiliary directories are included only when they contain a
    known per-scene artifact.
    """

    current = set(current_product_ids)
    stale: list[Path] = []
    for child in output_dir.iterdir():
        if not child.is_dir() or child.name in current:
            continue
        looks_like_product = child.name.startswith("HSHL1G_")
        contains_scene_artifact = any(
            (child / artifact).exists() for artifact in SCENE_OUTPUT_ARTIFACT_NAMES
        )
        if looks_like_product or contains_scene_artifact:
            stale.append(child.resolve())
    return sorted(stale, key=lambda path: path.name)


def _write_run_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_run_manifest(
    *,
    product_root: Path,
    output_dir: Path,
    current_product_ids: Sequence[str],
    stale_scene_directories: Sequence[Path],
    started_utc: str,
) -> dict[str, object]:
    stale_warning = None
    if stale_scene_directories:
        stale_warning = (
            "The listed stale scene directories were produced by earlier batches "
            "and are not members of this run. They were preserved and must not be "
            "used as current-run results."
        )
    return {
        "schema_version": 1,
        "status": "running",
        "started_utc": started_utc,
        "finished_utc": None,
        "product_root": str(product_root),
        "output_dir": str(output_dir),
        "current_product_ids": list(current_product_ids),
        "current_scene_directories": [
            str(output_dir / product_id) for product_id in current_product_ids
        ],
        "completed_product_ids": [],
        "stale_scene_directories": [
            str(path) for path in stale_scene_directories
        ],
        "stale_scene_warning": stale_warning,
    }


def analyze_product(
    product: HISUIL1GProduct,
    args: argparse.Namespace,
    known_sites: Sequence[KnownSite],
) -> SceneResult:
    started = time.perf_counter()
    scene_dir = args.output_dir / product.product_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    _clear_known_artifacts(
        scene_dir,
        (
            "candidate_components.csv",
            "known_site_metrics.csv",
            "score_maps.npz",
            "overview.png",
            "summary.json",
        ),
    )
    metadata = parse_metadata(product.metadata_path)
    georef = read_georeference(product.image_path)
    plan = make_band_plan(
        product,
        weak_window_nm=(args.weak_min_nm, args.weak_max_nm),
        strong_window_nm=(args.strong_min_nm, args.strong_max_nm),
    )
    if not args.disable_cloud_preflight:
        preflight_valid, preflight_cloud, preflight = sparse_cloud_preflight(
            product,
            plan,
            tile_stride=args.preflight_tile_stride,
            cloud_profile=args.cloud_profile,
            confirmation_threshold=args.maximum_cloud_fraction,
        )
        preflight_class = scene_quality_class(
            float(preflight["cloud_proxy_fraction"]),
            maximum_for_scoring=args.maximum_cloud_fraction,
            qa_complete=bool(preflight["qa_complete"]),
        )
        preflight_exclusion_supported = (
            bool(preflight["preflight_full_coverage"])
            and float(preflight["cloud_proxy_fraction"])
            > args.maximum_cloud_fraction
        )
        if preflight_class == "qa_incomplete" or (
            preflight_class == "excluded_cloud" and preflight_exclusion_supported
        ):
            preflight_clear = preflight_valid & ~preflight_cloud
            base_summary: dict[str, object] = {
                "product_id": product.product_id,
                "product_path": str(product.directory),
                "acquisition_utc": metadata.get("SceneCenterTime"),
                "processing_utc": metadata.get("ProcessingDate"),
                "product_version": metadata.get("ProductVersion"),
                "radiometric_parameter_file": metadata.get(
                    "RadiometricParameterFileName"
                ),
                "geometric_parameter_file": metadata.get(
                    "GeometricParameterFileName"
                ),
                "shape": [int(preflight_valid.shape[0]), int(preflight_valid.shape[1]), 185],
                "epsg": georef.epsg,
                "quality_class": preflight_class,
                "official_scoring_eligible": False,
                "analysis_config": analysis_config(args),
                **preflight,
                "interpretation": (
                    "MF was excluded because delivered detector QA is incomplete; "
                    "the scene cannot enter official candidate ranking."
                    if preflight_class == "qa_incomplete"
                    else "MF was excluded only after progressive sampling reached "
                    "stride 1 (all native tiles) and the exact quality-band cloud "
                    "fraction exceeded the threshold."
                ),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            save_scene_overview(
                scene_dir / "overview.png",
                product,
                preflight_cloud,
                preflight_clear,
                None,
                None,
                None,
                [],
                [],
                threshold=args.cluster_threshold,
                classification=preflight_class,
            )
            (scene_dir / "summary.json").write_text(
                json.dumps(base_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return SceneResult(
                summary=base_summary,
                dual_thumbnail=None,
                cloud_thumbnail=preflight_cloud[
                    :: args.thumbnail_stride, :: args.thumbnail_stride
                ],
                site_rows=[],
                candidate_rows=[],
            )
    radiance, clear, cloud, quality = prepare_scene_arrays(
        product,
        plan,
        cloud_dilation_pixels=args.cloud_dilation_pixels,
        cloud_profile=args.cloud_profile,
        tiff_workers=args.tiff_workers,
    )
    classification = scene_quality_class(
        quality["cloud_proxy_fraction"],
        maximum_for_scoring=args.maximum_cloud_fraction,
        qa_complete=bool(quality["qa_complete"]),
    )
    base_summary: dict[str, object] = {
        "product_id": product.product_id,
        "product_path": str(product.directory),
        "acquisition_utc": metadata.get("SceneCenterTime"),
        "processing_utc": metadata.get("ProcessingDate"),
        "product_version": metadata.get("ProductVersion"),
        "radiometric_parameter_file": metadata.get("RadiometricParameterFileName"),
        "geometric_parameter_file": metadata.get("GeometricParameterFileName"),
        "shape": [int(clear.shape[0]), int(clear.shape[1]), 185],
        "epsg": georef.epsg,
        "quality_class": classification,
        "official_scoring_eligible": is_officially_scored(classification),
        "analysis_config": analysis_config(args),
        **quality,
    }
    if not args.disable_cloud_preflight:
        base_summary["preflight_cloud_proxy_fraction"] = float(
            preflight["cloud_proxy_fraction"]
        )
        base_summary["quality_estimation"] = "full_scene_after_sparse_preflight"
    if not is_officially_scored(classification):
        save_scene_overview(
            scene_dir / "overview.png",
            product,
            cloud,
            clear,
            None,
            None,
            None,
            [],
            [],
            threshold=args.cluster_threshold,
            classification=classification,
        )
        base_summary["elapsed_seconds"] = float(time.perf_counter() - started)
        (scene_dir / "summary.json").write_text(
            json.dumps(base_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return SceneResult(
            summary=base_summary,
            dual_thumbnail=None,
            cloud_thumbnail=cloud[:: args.thumbnail_stride, :: args.thumbnail_stride],
            site_rows=[],
            candidate_rows=[],
        )

    (
        feature_transform,
        target,
        weak_features,
        strong_features,
        alpha_grid,
        _resampled,
    ) = _template_for_scene(
        plan,
        args.modtran_csv,
        uas_alpha_min=args.uas_alpha_min,
        uas_alpha_max=args.uas_alpha_max,
        continuum_degree=args.continuum_degree,
    )
    scan = make_scene_scan(
        radiance,
        clear,
        sample_size=args.sample_size,
        block_rows=args.block_rows,
    )
    models = build_nuisance_models(
        scan,
        feature_transform,
        target,
        weak_features,
        strong_features,
        n_folds=args.nuisance_folds,
        block_size=args.spatial_block_size,
        shrinkage=args.covariance_shrinkage,
        ridge=args.covariance_ridge,
        amplitude_cap=(
            float(alpha_grid[-1])
            if args.amplitude_cap is None
            else float(args.amplitude_cap)
        ),
        local_tile_size=args.local_tile_size,
        local_tile_halo=args.local_tile_halo,
    )
    maps = score_radiance_array(
        radiance,
        clear,
        models,
        feature_transform,
        block_rows=args.block_rows,
        spatial_block_size=args.spatial_block_size,
    )
    stripe_profiles: tuple[tuple[float, float], ...] = ()
    if args.enable_directional_destriping:
        stripe_profiles = (
            (args.thin_stripe_slope, args.thin_stripe_bin_width),
            (args.broad_stripe_slope, args.broad_stripe_bin_width),
        )
    weak_destriped, weak_stripe_diagnostics = directional_profile_destripe(
        maps["weak_z"], clear, profiles=stripe_profiles
    )
    strong_destriped, strong_stripe_diagnostics = directional_profile_destripe(
        maps["strong_z"], clear, profiles=stripe_profiles
    )
    weak_local = weighted_local_z(
        weak_destriped, clear, sigma_pixels=args.local_z_sigma
    )
    strong_local = weighted_local_z(
        strong_destriped, clear, sigma_pixels=args.local_z_sigma
    )
    dual_local = np.minimum(weak_local, strong_local)
    positive_components = _component_rows(
        dual_local,
        weak_local,
        strong_local,
        clear,
        cloud,
        georef,
        threshold=args.cluster_threshold,
        minimum_pixels=args.minimum_cluster_pixels,
        sign="positive_ch4",
    )
    reverse_components = _component_rows(
        np.minimum(-weak_local, -strong_local),
        -weak_local,
        -strong_local,
        clear,
        cloud,
        georef,
        threshold=args.cluster_threshold,
        minimum_pixels=args.minimum_cluster_pixels,
        sign="reverse_sign_control",
    )
    sites = site_metrics(
        known_sites,
        georef,
        clear.shape,
        maps,
        weak_local,
        strong_local,
        dual_local,
        clear,
        radius_pixels=args.site_radius_pixels,
    )
    for row in positive_components:
        row["product_id"] = product.product_id
        row["acquisition_utc"] = metadata.get("SceneCenterTime")
    for row in reverse_components:
        row["product_id"] = product.product_id
        row["acquisition_utc"] = metadata.get("SceneCenterTime")
    for row in sites:
        row["product_id"] = product.product_id
        row["acquisition_utc"] = metadata.get("SceneCenterTime")
        row["quality_class"] = classification

    component_rows = positive_components + reverse_components
    if component_rows:
        pd.DataFrame(component_rows).to_csv(
            scene_dir / "candidate_components.csv", index=False
        )
    if sites:
        pd.DataFrame(sites).to_csv(scene_dir / "known_site_metrics.csv", index=False)
    if args.save_score_maps:
        np.savez_compressed(
            scene_dir / "score_maps.npz",
            valid=clear,
            cloud_proxy=cloud,
            weak_local_z=weak_local,
            strong_local_z=strong_local,
            dual_local_z=dual_local,
            weak_z_directionally_destriped=weak_destriped,
            strong_z_directionally_destriped=strong_destriped,
            **maps,
        )
    save_scene_overview(
        scene_dir / "overview.png",
        product,
        cloud,
        clear,
        weak_local,
        strong_local,
        dual_local,
        positive_components,
        sites,
        threshold=args.cluster_threshold,
        classification=classification,
    )

    finite_dual = dual_local[clear]
    base_summary.update(
        {
            "sample_pixels": int(len(scan.sample_radiance)),
            "selected_wavelengths_nm": [float(value) for value in plan.wavelengths_nm],
            "selected_fwhm_nm": [float(value) for value in plan.fwhm_nm],
            "positive_component_count": len(positive_components),
            "reverse_control_component_count": len(reverse_components),
            "largest_positive_component_pixels": max(
                (int(row["pixel_count"]) for row in positive_components), default=0
            ),
            "largest_reverse_component_pixels": max(
                (int(row["pixel_count"]) for row in reverse_components), default=0
            ),
            "dual_local_z_quantiles_50_90_95_99_99p9_max": [
                float(value)
                for value in np.quantile(
                    finite_dual, [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
                )
            ],
            "model": {
                "modtran_csv": str(args.modtran_csv),
                "modtran_global_scale_for_uas": 1.0,
                "modtran_times_100_note": (
                    "A global x100 factor cancels exactly in the log-radiance UAS; "
                    "it is not applied to exported HISUI radiance."
                ),
                "continuum_degree": args.continuum_degree,
                "nuisance_folds": args.nuisance_folds,
                "spatial_block_size": args.spatial_block_size,
                "local_tile_size": args.local_tile_size,
                "local_tile_halo": args.local_tile_halo,
                "local_z_sigma_pixels": args.local_z_sigma,
                "directional_destriping": {
                    "enabled": args.enable_directional_destriping,
                    "status": (
                        "Fixed directions inherited from prior QA_DM trace analysis; "
                        "candidate pixels are winsorized during profile estimation."
                    ),
                    "weak_band_profiles": weak_stripe_diagnostics,
                    "strong_band_profiles": strong_stripe_diagnostics,
                },
                "cluster_threshold": args.cluster_threshold,
                "minimum_cluster_pixels": args.minimum_cluster_pixels,
            },
            "interpretation": (
                "Candidate screen only. Positive clusters require both methane windows; "
                "reverse-sign clusters diagnose tail asymmetry. Neither is a calibrated "
                "emission probability or an emission-rate retrieval."
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    (scene_dir / "summary.json").write_text(
        json.dumps(base_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del radiance, maps, weak_destriped, strong_destriped, weak_local, strong_local
    gc.collect()
    return SceneResult(
        summary=base_summary,
        dual_thumbnail=dual_local[:: args.thumbnail_stride, :: args.thumbnail_stride],
        cloud_thumbnail=cloud[:: args.thumbnail_stride, :: args.thumbnail_stride],
        site_rows=sites,
        candidate_rows=positive_components + reverse_components,
    )


def save_batch_gallery(path: Path, results: Sequence[SceneResult]) -> None:
    columns = 4
    rows = math.ceil(len(results) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).ravel()
    for axis, result in zip(axes_flat, results):
        summary = result.summary
        if result.dual_thumbnail is None:
            axis.imshow(result.cloud_thumbnail, cmap="Reds", vmin=0, vmax=1)
            label = f"quality screen; MF excluded ({summary['quality_class']})"
        else:
            axis.imshow(result.dual_thumbnail, cmap="coolwarm", vmin=-3, vmax=6)
            label = "dual-band local z"
        product_id = str(summary["product_id"])
        parts = product_id.split("_")
        short = f"{parts[1]} {str(summary.get('acquisition_utc', ''))[:10]}"
        axis.set_title(
            f"{short}\n{summary['quality_class']}; cloud={float(summary['cloud_proxy_fraction']):.2f}\n{label}",
            fontsize=9,
            color=QUALITY_CLASS_COLORS.get(str(summary["quality_class"]), "black"),
        )
        axis.set_axis_off()
    for axis in axes_flat[len(results) :]:
        axis.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cloud_fraction_plot(
    path: Path, summaries: pd.DataFrame, *, maximum_cloud_fraction: float
) -> None:
    shown = summaries.sort_values("cloud_proxy_fraction")
    labels = [
        f"{value.split('_')[1]} {str(date)[:10]}"
        for value, date in zip(shown.product_id, shown.acquisition_utc)
    ]
    colors = [QUALITY_CLASS_COLORS[value] for value in shown.quality_class]
    fig, axis = plt.subplots(figsize=(11, max(5, 0.38 * len(shown))), constrained_layout=True)
    y = np.arange(len(shown))
    axis.barh(y, shown.cloud_proxy_fraction, color=colors)
    axis.axvline(0.10, color="black", linestyle="--", linewidth=0.8, label="usable/partial")
    axis.axvline(
        maximum_cloud_fraction,
        color="black",
        linestyle=":",
        linewidth=0.8,
        label="scoring exclusion",
    )
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Cloud proxy fraction among valid footprint pixels")
    axis.legend(loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_batch(args: argparse.Namespace) -> dict[str, object]:
    args.product_root = args.product_root.expanduser().resolve()
    args.modtran_csv = args.modtran_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    # Overwrite any earlier successful manifest before discovery.  Otherwise a
    # malformed site CSV or empty product root could leave an old `complete`
    # manifest beside newly cleared/partial outputs and make the failed rerun
    # look successful.
    startup_manifest = _make_run_manifest(
        product_root=args.product_root,
        output_dir=args.output_dir,
        current_product_ids=[],
        stale_scene_directories=[],
        started_utc=started_utc,
    )
    startup_manifest["status"] = "initializing"
    _write_run_manifest(args.output_dir, startup_manifest)
    try:
        _clear_known_artifacts(
            args.output_dir,
            (
                "scene_quality_summary.csv",
                "all_candidate_components.csv",
                "known_site_repeat_metrics.csv",
                "scene_methane_gallery.png",
                "scene_cloud_fraction.png",
                "batch_summary.json",
            ),
        )
        sites = load_known_sites(args.known_site_csv)
        products = discover_l1g_products(args.product_root)
        if not products:
            raise FileNotFoundError(f"No HISUI products below {args.product_root}")
    except BaseException as error:
        startup_manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_run_manifest(args.output_dir, startup_manifest)
        raise
    product_ids = [product.product_id for product in products]
    stale_scene_directories = _find_stale_scene_directories(
        args.output_dir, product_ids
    )
    if stale_scene_directories:
        print(
            "WARNING: preserving stale scene output directories that are not part "
            f"of this run: {', '.join(path.name for path in stale_scene_directories)}",
            flush=True,
        )
    run_manifest = _make_run_manifest(
        product_root=args.product_root,
        output_dir=args.output_dir,
        current_product_ids=product_ids,
        stale_scene_directories=stale_scene_directories,
        started_utc=started_utc,
    )
    _write_run_manifest(args.output_dir, run_manifest)
    results: list[SceneResult] = []
    try:
        for index, product in enumerate(products, start=1):
            print(f"[{index}/{len(products)}] {product.product_id}", flush=True)
            result = analyze_product(product, args, sites)
            results.append(result)
            run_manifest["completed_product_ids"] = [
                item.summary["product_id"] for item in results
            ]
            _write_run_manifest(args.output_dir, run_manifest)
            print(
                f"  {result.summary['quality_class']} cloud="
                f"{float(result.summary['cloud_proxy_fraction']):.3f} "
                f"components={result.summary.get('positive_component_count', 0)}/"
                f"{result.summary.get('reverse_control_component_count', 0)}",
                flush=True,
            )
    except BaseException as error:
        run_manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_run_manifest(args.output_dir, run_manifest)
        raise

    summaries = pd.DataFrame([result.summary for result in results])
    scalar_columns = [
        column
        for column in summaries.columns
        if not summaries[column].map(lambda value: isinstance(value, (dict, list))).any()
    ]
    summaries[scalar_columns].to_csv(args.output_dir / "scene_quality_summary.csv", index=False)
    candidate_rows = [
        row
        for result in results
        if is_officially_scored(str(result.summary["quality_class"]))
        for row in result.candidate_rows
    ]
    site_rows = [
        row
        for result in results
        if is_officially_scored(str(result.summary["quality_class"]))
        for row in result.site_rows
    ]
    candidate_output: str | None = None
    if candidate_rows:
        pd.DataFrame(candidate_rows).to_csv(
            args.output_dir / "all_candidate_components.csv", index=False
        )
        candidate_output = str(args.output_dir / "all_candidate_components.csv")
    site_output: str | None = None
    if site_rows:
        pd.DataFrame(site_rows).to_csv(args.output_dir / "known_site_repeat_metrics.csv", index=False)
        site_output = str(args.output_dir / "known_site_repeat_metrics.csv")
    save_batch_gallery(args.output_dir / "scene_methane_gallery.png", results)
    save_cloud_fraction_plot(
        args.output_dir / "scene_cloud_fraction.png",
        summaries,
        maximum_cloud_fraction=args.maximum_cloud_fraction,
    )
    overall = {
        "inputs": {
            "product_root": str(args.product_root),
            "modtran_csv": str(args.modtran_csv),
            "known_site_csv": str(args.known_site_csv) if args.known_site_csv else None,
        },
        "scene_count": len(results),
        "current_product_ids": product_ids,
        "stale_scene_directories": [
            str(path) for path in stale_scene_directories
        ],
        "stale_scene_warning": run_manifest["stale_scene_warning"],
        "quality_class_counts": {
            str(key): int(value)
            for key, value in summaries.quality_class.value_counts().items()
        },
        "scored_scene_count": int(
            summaries.quality_class.map(is_officially_scored).sum()
        ),
        "excluded_scene_count": int(
            (~summaries.quality_class.map(is_officially_scored)).sum()
        ),
        "cloud_excluded_scene_count": int(
            (summaries.quality_class == "excluded_cloud").sum()
        ),
        "qa_incomplete_scene_count": int(
            (summaries.quality_class == "qa_incomplete").sum()
        ),
        "analysis_config": analysis_config(args),
        "method_status": (
            "Cloud-aware multi-scene candidate screen. Cloud proxy thresholds are "
            "fixed but not a validated HISUI cloud product. Scenes with incomplete "
            "delivered QA are excluded from official scoring. Dual-window clusters "
            "and reverse-sign controls are descriptive, not formal detections."
        ),
        "outputs": {
            "run_manifest": str(args.output_dir / "run_manifest.json"),
            "scene_quality_summary": str(args.output_dir / "scene_quality_summary.csv"),
            "scene_gallery": str(args.output_dir / "scene_methane_gallery.png"),
            "cloud_fraction": str(args.output_dir / "scene_cloud_fraction.png"),
            "candidate_components": candidate_output,
            "known_site_repeat_metrics": site_output,
        },
    }
    (args.output_dir / "batch_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_manifest.update(
        {
            "status": "complete",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "batch_summary": str(args.output_dir / "batch_summary.json"),
        }
    )
    _write_run_manifest(args.output_dir, run_manifest)
    return overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-root", type=Path, required=True)
    parser.add_argument("--modtran-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--known-site-csv", type=Path)
    parser.add_argument("--weak-min-nm", type=float, default=1580.0)
    parser.add_argument("--weak-max-nm", type=float, default=1750.0)
    parser.add_argument("--strong-min-nm", type=float, default=2200.0)
    parser.add_argument("--strong-max-nm", type=float, default=2390.0)
    parser.add_argument("--uas-alpha-min", type=float, default=0.0)
    parser.add_argument("--uas-alpha-max", type=float, default=0.5)
    parser.add_argument("--continuum-degree", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=80_000)
    parser.add_argument("--block-rows", type=int, default=512)
    parser.add_argument("--nuisance-folds", type=int, default=4)
    parser.add_argument("--spatial-block-size", type=int, default=20)
    parser.add_argument("--local-tile-size", type=int, default=0)
    parser.add_argument("--local-tile-halo", type=int, default=1)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.05)
    parser.add_argument("--covariance-ridge", type=float, default=1e-6)
    parser.add_argument("--amplitude-cap", type=float)
    parser.add_argument("--maximum-cloud-fraction", type=float, default=0.50)
    parser.add_argument("--cloud-dilation-pixels", type=int, default=2)
    parser.add_argument(
        "--cloud-profile", choices=sorted(CLOUD_PROFILES), default="cirrus_sensitive"
    )
    parser.add_argument("--preflight-tile-stride", type=int, default=8)
    parser.add_argument("--disable-cloud-preflight", action="store_true")
    parser.add_argument("--local-z-sigma", type=float, default=20.0)
    parser.add_argument("--thin-stripe-slope", type=float, default=0.978)
    parser.add_argument("--thin-stripe-bin-width", type=float, default=2.0)
    parser.add_argument("--broad-stripe-slope", type=float, default=1.267)
    parser.add_argument("--broad-stripe-bin-width", type=float, default=18.0)
    stripe_group = parser.add_mutually_exclusive_group()
    stripe_group.add_argument(
        "--enable-directional-destriping",
        action="store_true",
        help="Apply fixed directions previously estimated from Permian QA traces.",
    )
    stripe_group.add_argument(
        "--disable-directional-destriping",
        action="store_false",
        dest="enable_directional_destriping",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(enable_directional_destriping=False)
    parser.add_argument("--cluster-threshold", type=float, default=3.0)
    parser.add_argument("--minimum-cluster-pixels", type=int, default=3)
    parser.add_argument("--site-radius-pixels", type=int, default=5)
    parser.add_argument("--thumbnail-stride", type=int, default=4)
    parser.add_argument(
        "--tiff-workers",
        type=int,
        default=1,
        help="Use one worker for sequential reads; many workers can thrash external disks.",
    )
    parser.add_argument("--save-score-maps", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

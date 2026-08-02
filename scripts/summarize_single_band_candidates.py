#!/usr/bin/env python
"""Summarize cloud-clear HISUI methane-like regions from either SWIR window.

The input is a completed ``screen_hisui_l1g_scenes.py --save-score-maps``
batch.  Only scenes explicitly classified as ``usable`` are read.  Products
whose scene-centre times fall in the same UTC minute are placed on their common
projected L1G grid so that overlapping tiles are counted once.

This is an exploratory review catalogue, not a methane detection probability or
an emission-rate retrieval.  Positive and exactly sign-reversed masks are
processed symmetrically so the relaxed one-window screen can be audited.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import tifffile

from hisui_l1g_io import (
    GeoReference,
    HISUIL1GProduct,
    discover_l1g_product,
    parse_metadata,
    read_georeference,
)
from plot_hisui_candidate_crop import browse_crop, crop_bounds


ANALYSIS_VERSION = "2026-08-02-v2"
SUPPORT_ORDER = (
    "1600_only",
    "2200_only",
    "both_noncoincident",
    "contains_coincident_dual_pixel",
)
OUTPUT_NAMES = (
    "single_band_candidate_regions.csv",
    "single_band_review_shortlist.csv",
    "single_band_reverse_control_regions.csv",
    "single_band_product_counts.csv",
    "single_band_summary.json",
    "single_band_candidate_gallery.png",
)
REGION_FIELDS = (
    "global_rank",
    "strip_rank",
    "tail",
    "acquisition_utc_minute",
    "strip_product_ids",
    "contributing_product_ids",
    "spectral_support",
    "contains_strict_dual_component",
    "review_shortlist",
    "conservative_single_window",
    "shape_class",
    "region_id",
    "pixel_count",
    "n_1600_threshold_pixels",
    "n_2200_threshold_pixels",
    "n_coincident_dual_pixels",
    "n_strict_dual_pixels",
    "n_1600_core_pixels_z5",
    "n_2200_core_pixels_z5",
    "primary_peak_band",
    "primary_peak_z",
    "peak_1600_z",
    "peak_2200_z",
    "peak_dual_z",
    "counterpart_median_z_on_primary_extent",
    "peak_union_y",
    "peak_union_x",
    "peak_source_product_id",
    "peak_source_y",
    "peak_source_x",
    "centroid_union_y",
    "centroid_union_x",
    "bbox_height",
    "bbox_width",
    "minor_axis_sd_pixels",
    "major_axis_sd_pixels",
    "elongation",
    "major_axis_angle_deg_from_east",
    "shape_stripe_direction_flag",
    "scene_spanning_line_flag",
    "minimum_distance_to_cloud_pixels",
    "minimum_distance_to_invalid_pixels",
    "easting_m",
    "northing_m",
    "centroid_easting_m",
    "centroid_northing_m",
    "epsg",
)
PRODUCT_COUNT_FIELDS = (
    "product_id",
    "acquisition_utc",
    "acquisition_utc_minute",
    "cloud_proxy_fraction",
    "cloud_dilated_fraction",
    "analysis_valid_pixels",
    "minimum_component_pixels",
    "positive_1600_tail_pixels",
    "reverse_1600_tail_pixels",
    "positive_1600_retained_component_count",
    "reverse_1600_retained_component_count",
    "positive_2200_tail_pixels",
    "reverse_2200_tail_pixels",
    "positive_2200_retained_component_count",
    "reverse_2200_retained_component_count",
    "positive_strict_dual_retained_component_count",
    "reverse_strict_dual_retained_component_count",
)


@dataclass
class SceneSource:
    product: HISUIL1GProduct
    product_id: str
    acquisition_utc: str
    acquisition_minute: str
    cloud_fraction: float
    cloud_dilated_fraction: float
    georef: GeoReference
    valid: np.ndarray
    cloud: np.ndarray
    weak: np.ndarray
    strong: np.ndarray
    offset: tuple[int, int] = (0, 0)


@dataclass
class StripGeometry:
    acquisition_minute: str
    georef: GeoReference
    shape: tuple[int, int]
    sources: list[SceneSource]
    valid: np.ndarray
    cloud: np.ndarray
    cloud_distance: np.ndarray
    invalid_distance: np.ndarray


@dataclass
class TailMosaic:
    band1600: np.ndarray
    band2200: np.ndarray
    dual: np.ndarray
    mask1600: np.ndarray
    mask2200: np.ndarray
    coincident: np.ndarray
    strict_dual: np.ndarray


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.dtype != np.bool_ and not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=False)


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not np.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return number


def _parse_acquisition(value: object, *, product_id: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{product_id}: acquisition_utc is missing")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{product_id}: invalid acquisition_utc {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{product_id}: acquisition_utc must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    minute = parsed.replace(second=0, microsecond=0)
    return minute.isoformat().replace("+00:00", "Z"), parsed


def _component_count(mask: np.ndarray, minimum_pixels: int) -> tuple[int, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    keep = sizes >= minimum_pixels
    return int(np.count_nonzero(keep)), int(np.sum(sizes[keep]))


def _kept_component_mask(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def _validated_batch(
    batch_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    manifest = _read_json(batch_dir / "run_manifest.json", label="run manifest")
    if manifest.get("status") != "complete":
        raise ValueError("Batch run_manifest.json is not complete")
    product_ids = manifest.get("current_product_ids")
    if (
        not isinstance(product_ids, list)
        or not product_ids
        or any(not isinstance(value, str) or not value for value in product_ids)
        or len(set(product_ids)) != len(product_ids)
    ):
        raise ValueError("run_manifest current_product_ids is invalid")
    if manifest.get("completed_product_ids") != product_ids:
        raise ValueError("run_manifest completed_product_ids do not match current_product_ids")
    expected_scene_directories = [str((batch_dir / value).resolve()) for value in product_ids]
    actual_scene_directories = manifest.get("current_scene_directories")
    if not isinstance(actual_scene_directories, list) or [
        str(Path(value).expanduser().resolve()) for value in actual_scene_directories
    ] != expected_scene_directories:
        raise ValueError("run_manifest current_scene_directories are invalid")

    batch_summary = _read_json(batch_dir / "batch_summary.json", label="batch summary")
    if batch_summary.get("current_product_ids") != product_ids:
        raise ValueError("batch_summary current_product_ids do not match run_manifest")
    if int(batch_summary.get("scene_count", -1)) != len(product_ids):
        raise ValueError("batch_summary scene_count does not match current_product_ids")
    config = batch_summary.get("analysis_config")
    if not isinstance(config, dict):
        raise ValueError("batch_summary analysis_config is missing")
    if config.get("save_score_maps") is not True:
        raise ValueError("batch_summary was not run with save_score_maps enabled")
    expected_windows = {
        "weak_min_nm": 1580.0,
        "weak_max_nm": 1750.0,
        "strong_min_nm": 2200.0,
        "strong_max_nm": 2390.0,
    }
    if any(float(config.get(key, math.nan)) != value for key, value in expected_windows.items()):
        raise ValueError("batch_summary has unexpected methane-window definitions")
    _finite_positive(config.get("local_z_sigma", math.nan), "batch local_z_sigma")
    return manifest, batch_summary, product_ids


def _load_sources(
    batch_dir: Path,
) -> tuple[list[SceneSource], list[dict[str, Any]], dict[str, Any]]:
    manifest, batch_summary, product_ids = _validated_batch(batch_dir)
    batch_config = batch_summary["analysis_config"]

    sources: list[SceneSource] = []
    skipped: list[dict[str, Any]] = []
    observed_quality_counts: dict[str, int] = {}
    known_quality_classes = {"usable", "partial", "excluded_cloud", "qa_incomplete"}
    for product_id in product_ids:
        scene_dir = batch_dir / product_id
        summary = _read_json(scene_dir / "summary.json", label=f"{product_id} summary")
        if summary.get("product_id") != product_id:
            raise ValueError(f"{product_id}: summary product_id mismatch")
        if summary.get("analysis_config") != batch_config:
            raise ValueError(f"{product_id}: summary analysis_config differs from batch")
        quality = summary.get("quality_class")
        if not isinstance(quality, str) or quality not in known_quality_classes:
            raise ValueError(f"{product_id}: unknown quality_class {quality!r}")
        observed_quality_counts[quality] = observed_quality_counts.get(quality, 0) + 1
        expected_eligible = quality in {"usable", "partial"}
        if summary.get("official_scoring_eligible") is not expected_eligible:
            raise ValueError(f"{product_id}: official_scoring_eligible contradicts quality_class")
        if quality != "usable":
            skipped.append({"product_id": product_id, "quality_class": quality})
            continue
        if summary.get("official_scoring_eligible") is not True:
            raise ValueError(f"{product_id}: usable scene is not scoring eligible")
        minute, summary_acquisition = _parse_acquisition(
            summary.get("acquisition_utc"), product_id=product_id
        )
        product_path = summary.get("product_path")
        if not isinstance(product_path, str) or not product_path:
            raise ValueError(f"{product_id}: product_path is missing")
        product = discover_l1g_product(product_path)
        if product.product_id != product_id:
            raise ValueError(f"{product_id}: discovered product ID mismatch")
        metadata = parse_metadata(product.metadata_path)
        _metadata_minute, metadata_acquisition = _parse_acquisition(
            metadata.get("SceneCenterTime"), product_id=product_id
        )
        if metadata_acquisition != summary_acquisition:
            raise ValueError(f"{product_id}: summary acquisition differs from L1G metadata")
        georef = read_georeference(product.image_path)
        if georef.epsg is None:
            raise ValueError(f"{product_id}: source GeoTIFF lacks projected EPSG")

        score_path = scene_dir / "score_maps.npz"
        if not score_path.is_file():
            raise FileNotFoundError(f"{product_id}: missing score_maps.npz")
        with np.load(score_path, allow_pickle=False) as archive:
            required = {"valid", "cloud_proxy", "weak_local_z", "strong_local_z"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{product_id}: score maps lack {', '.join(missing)}")
            valid = _binary_mask(archive["valid"], f"{product_id} valid")
            cloud = _binary_mask(archive["cloud_proxy"], f"{product_id} cloud_proxy")
            weak = np.asarray(archive["weak_local_z"], dtype=np.float32)
            strong = np.asarray(archive["strong_local_z"], dtype=np.float32)
        if cloud.shape != valid.shape:
            raise ValueError(f"{product_id}: cloud_proxy shape mismatch")
        if np.any(valid & cloud):
            raise ValueError(f"{product_id}: valid and cloud masks overlap")
        for name, array in (("weak_local_z", weak), ("strong_local_z", strong)):
            if array.shape != valid.shape:
                raise ValueError(f"{product_id}: {name} shape mismatch")
            if not np.all(np.isfinite(array[valid])):
                raise ValueError(f"{product_id}: {name} is non-finite on valid pixels")
        shape = summary.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) < 2
            or tuple(int(value) for value in shape[:2]) != valid.shape
        ):
            raise ValueError(f"{product_id}: summary shape mismatch")
        with tifffile.TiffFile(str(product.image_path)) as tif:
            raster_shape = (int(tif.pages[0].imagelength), int(tif.pages[0].imagewidth))
        if raster_shape != valid.shape:
            raise ValueError(f"{product_id}: score-map shape differs from source GeoTIFF")
        sources.append(
            SceneSource(
                product=product,
                product_id=product_id,
                acquisition_utc=str(summary["acquisition_utc"]),
                acquisition_minute=minute,
                cloud_fraction=float(summary["cloud_proxy_fraction"]),
                cloud_dilated_fraction=float(summary["cloud_dilated_fraction"]),
                georef=georef,
                valid=valid,
                cloud=cloud,
                weak=weak,
                strong=strong,
            )
        )
    if not sources:
        raise ValueError("No quality_class=usable scenes were found")
    batch_quality_counts = batch_summary.get("quality_class_counts")
    if not isinstance(batch_quality_counts, dict):
        raise ValueError("batch_summary quality_class_counts is missing")
    try:
        normalized_batch_counts = {
            str(key): int(value) for key, value in batch_quality_counts.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_summary quality_class_counts is invalid") from exc
    if normalized_batch_counts != observed_quality_counts:
        raise ValueError("batch_summary quality_class_counts do not match scene summaries")
    return sources, skipped, {
        "manifest_started_utc": manifest.get("started_utc"),
        "manifest_finished_utc": manifest.get("finished_utc"),
        "batch_summary": str((batch_dir / "batch_summary.json").resolve()),
        "batch_analysis_config": batch_config,
    }


def _strip_geometry(acquisition_minute: str, sources: Sequence[SceneSource]) -> StripGeometry:
    if not sources:
        raise ValueError("A strip requires at least one source")
    for source in sources:
        if source.cloud.shape != source.valid.shape:
            raise ValueError(f"{source.product_id}: cloud_proxy shape mismatch")
        if source.weak.shape != source.valid.shape or source.strong.shape != source.valid.shape:
            raise ValueError(f"{source.product_id}: score-map shape mismatch")
    reference = sources[0].georef
    transform = np.asarray(reference.geotransform, dtype=float)
    linear = np.asarray([[transform[1], transform[2]], [transform[4], transform[5]]])
    if not np.all(np.isfinite(linear)) or abs(float(np.linalg.det(linear))) < 1e-12:
        raise ValueError(f"{acquisition_minute}: singular projected pixel transform")
    raw_offsets: list[tuple[int, int]] = []
    for source in sources:
        if source.georef.epsg != reference.epsg:
            raise ValueError(f"{acquisition_minute}: products have different EPSG codes")
        source_transform = np.asarray(source.georef.geotransform, dtype=float)
        source_linear = np.asarray(
            [[source_transform[1], source_transform[2]], [source_transform[4], source_transform[5]]]
        )
        if not np.allclose(source_linear, linear, rtol=1e-12, atol=1e-9):
            raise ValueError(f"{acquisition_minute}: products have different pixel transforms")
        delta = np.asarray(
            [source_transform[0] - transform[0], source_transform[3] - transform[3]]
        )
        column_offset, row_offset = np.linalg.solve(linear, delta)
        rounded = (int(round(float(row_offset))), int(round(float(column_offset))))
        if not np.allclose(
            [row_offset, column_offset], rounded, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"{acquisition_minute}: non-integer L1G grid offset")
        raw_offsets.append(rounded)
    min_row = min(value[0] for value in raw_offsets)
    min_col = min(value[1] for value in raw_offsets)
    max_row = max(
        offset[0] + source.valid.shape[0]
        for offset, source in zip(raw_offsets, sources)
    )
    max_col = max(
        offset[1] + source.valid.shape[1]
        for offset, source in zip(raw_offsets, sources)
    )
    shape = (max_row - min_row, max_col - min_col)
    union_origin = np.asarray([transform[0], transform[3]]) + linear @ np.asarray(
        [min_col, min_row], dtype=float
    )
    union_georef = GeoReference(
        geotransform=(
            float(union_origin[0]),
            float(transform[1]),
            float(transform[2]),
            float(union_origin[1]),
            float(transform[4]),
            float(transform[5]),
        ),
        crs=reference.crs,
        epsg=reference.epsg,
        pixel_scale=reference.pixel_scale,
        tiepoint=reference.tiepoint,
        raster_type=reference.raster_type,
    )
    valid = np.zeros(shape, dtype=bool)
    cloud = np.zeros(shape, dtype=bool)
    for raw, source in zip(raw_offsets, sources):
        source.offset = (raw[0] - min_row, raw[1] - min_col)
        row, column = source.offset
        height, width = source.valid.shape
        target = np.s_[row : row + height, column : column + width]
        valid[target] |= source.valid
        cloud[target] |= source.cloud
    if cloud.any():
        cloud_distance = ndimage.distance_transform_edt(~cloud).astype(np.float32)
    else:
        cloud_distance = np.full(shape, np.inf, dtype=np.float32)
    invalid_distance = ndimage.distance_transform_edt(
        np.pad(valid, 1, constant_values=False)
    )[1:-1, 1:-1].astype(np.float32)
    return StripGeometry(
        acquisition_minute=acquisition_minute,
        georef=union_georef,
        shape=shape,
        sources=list(sources),
        valid=valid,
        cloud=cloud,
        cloud_distance=cloud_distance,
        invalid_distance=invalid_distance,
    )


def _tail_mosaic(
    geometry: StripGeometry,
    *,
    threshold: float,
    minimum_pixels: int,
    sign: int,
) -> TailMosaic:
    shape = geometry.shape
    weak_best = np.full(shape, -np.inf, dtype=np.float32)
    strong_best = np.full(shape, -np.inf, dtype=np.float32)
    dual_best = np.full(shape, -np.inf, dtype=np.float32)
    weak_mask = np.zeros(shape, dtype=bool)
    strong_mask = np.zeros(shape, dtype=bool)
    coincident = np.zeros(shape, dtype=bool)
    for source in geometry.sources:
        row, column = source.offset
        height, width = source.valid.shape
        target = np.s_[row : row + height, column : column + width]
        weak = source.weak if sign > 0 else -source.weak
        strong = source.strong if sign > 0 else -source.strong
        valid = source.valid & np.isfinite(weak) & np.isfinite(strong)
        target_weak = weak_best[target]
        target_strong = strong_best[target]
        target_dual = dual_best[target]
        np.maximum(target_weak, np.where(valid, weak, -np.inf), out=target_weak)
        np.maximum(target_strong, np.where(valid, strong, -np.inf), out=target_strong)
        np.maximum(
            target_dual,
            np.where(valid, np.minimum(weak, strong), -np.inf),
            out=target_dual,
        )
        weak_mask[target] |= valid & (weak >= threshold)
        strong_mask[target] |= valid & (strong >= threshold)
        coincident[target] |= valid & (weak >= threshold) & (strong >= threshold)
    strict_dual = _kept_component_mask(coincident, minimum_pixels)
    return TailMosaic(
        band1600=weak_best,
        band2200=strong_best,
        dual=dual_best,
        mask1600=weak_mask,
        mask2200=strong_mask,
        coincident=coincident,
        strict_dual=strict_dual,
    )


def _same_source_cross_band_conflict_mask(
    geometry: StripGeometry, *, threshold: float
) -> np.ndarray:
    conflict = np.zeros(geometry.shape, dtype=bool)
    for source in geometry.sources:
        row, column = source.offset
        height, width = source.valid.shape
        target = np.s_[row : row + height, column : column + width]
        source_conflict = source.valid & (
            ((source.weak >= threshold) & (source.strong <= -threshold))
            | ((source.strong >= threshold) & (source.weak <= -threshold))
        )
        conflict[target] |= source_conflict
    return conflict


def _angle_difference_degrees(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _support_name(n1600: int, n2200: int, coincident: int) -> str:
    if coincident > 0:
        return "contains_coincident_dual_pixel"
    if n1600 > 0 and n2200 > 0:
        return "both_noncoincident"
    if n1600 > 0:
        return "1600_only"
    return "2200_only"


def _peak_source(
    geometry: StripGeometry,
    *,
    union_y: int,
    union_x: int,
    band: str,
    sign: int,
) -> tuple[str, int, int]:
    best: tuple[float, str, int, int] | None = None
    for source in geometry.sources:
        local_y = union_y - source.offset[0]
        local_x = union_x - source.offset[1]
        if not (0 <= local_y < source.valid.shape[0] and 0 <= local_x < source.valid.shape[1]):
            continue
        if not source.valid[local_y, local_x]:
            continue
        value = source.weak[local_y, local_x] if band == "1600_nm" else source.strong[local_y, local_x]
        score = float(value) * sign
        candidate = (score, source.product_id, int(local_y), int(local_x))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("No source product supports the selected union-grid peak")
    return best[1], best[2], best[3]


def _contributing_product_ids(
    geometry: StripGeometry,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    threshold: float,
    sign: int,
) -> list[str]:
    contributing: list[str] = []
    for source in geometry.sources:
        local_y = yy - source.offset[0]
        local_x = xx - source.offset[1]
        inside = (
            (local_y >= 0)
            & (local_y < source.valid.shape[0])
            & (local_x >= 0)
            & (local_x < source.valid.shape[1])
        )
        if not np.any(inside):
            continue
        source_y = local_y[inside]
        source_x = local_x[inside]
        hits = source.valid[source_y, source_x] & (
            (sign * source.weak[source_y, source_x] >= threshold)
            | (sign * source.strong[source_y, source_x] >= threshold)
        )
        if np.any(hits):
            contributing.append(source.product_id)
    if not contributing:
        raise ValueError("No source product contributes to a retained union-grid region")
    return contributing


def _extract_regions(
    geometry: StripGeometry,
    mosaic: TailMosaic,
    *,
    threshold: float,
    minimum_pixels: int,
    tail: str,
    sign: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    candidate = geometry.valid & (mosaic.mask1600 | mosaic.mask2200)
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels, max_label=count)
    rows: list[dict[str, Any]] = []
    selected_labels = np.zeros(count + 1, dtype=bool)
    stripe_angles = [math.degrees(math.atan(value)) % 180.0 for value in (0.978, 1.267)]
    for label_id, section in enumerate(objects, start=1):
        if section is None:
            continue
        selected = labels[section] == label_id
        yy_local, xx_local = np.nonzero(selected)
        yy = yy_local + int(section[0].start)
        xx = xx_local + int(section[1].start)
        weak_hits = mosaic.mask1600[yy, xx]
        strong_hits = mosaic.mask2200[yy, xx]
        coincident_hits = mosaic.coincident[yy, xx]
        strict_hits = mosaic.strict_dual[yy, xx]
        n1600 = int(np.count_nonzero(weak_hits))
        n2200 = int(np.count_nonzero(strong_hits))
        if max(n1600, n2200) < minimum_pixels:
            continue
        selected_labels[label_id] = True
        size = int(selected.sum())
        weak_values = mosaic.band1600[yy, xx]
        strong_values = mosaic.band2200[yy, xx]
        peak_values = np.maximum(weak_values, strong_values)
        peak_index = int(np.nanargmax(peak_values))
        peak_y, peak_x = int(yy[peak_index]), int(xx[peak_index])
        primary_band = (
            "1600_nm" if weak_values[peak_index] >= strong_values[peak_index] else "2200_2390_nm"
        )
        primary_hits = weak_hits if primary_band == "1600_nm" else strong_hits
        counterpart_values = strong_values if primary_band == "1600_nm" else weak_values
        counterpart_median = float(np.median(counterpart_values[primary_hits]))
        coordinates = np.column_stack([yy, xx]).astype(float)
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        minor = float(math.sqrt(float(eigenvalues[0])))
        major = float(math.sqrt(float(eigenvalues[1])))
        elongation = major / max(minor, 1e-6)
        major_vector = eigenvectors[:, 1]
        angle = float(math.degrees(math.atan2(major_vector[0], major_vector[1])) % 180.0)
        height = int(section[0].stop - section[0].start)
        width = int(section[1].stop - section[1].start)
        stripe_flag = bool(
            max(height, width) >= 10
            and minor < 1.0
            and any(_angle_difference_degrees(angle, value) <= 5.0 for value in stripe_angles)
        )
        line_flag = bool(size >= 100 and elongation >= 10.0)
        cloud_distance = float(np.min(geometry.cloud_distance[yy, xx]))
        invalid_distance = float(np.min(geometry.invalid_distance[yy, xx]))
        support = _support_name(n1600, n2200, int(np.count_nonzero(coincident_hits)))
        core1600 = int(np.count_nonzero(weak_values[weak_hits] >= 5.0))
        core2200 = int(np.count_nonzero(strong_values[strong_hits] >= 5.0))
        primary_count = n1600 if primary_band == "1600_nm" else n2200
        primary_core = core1600 if primary_band == "1600_nm" else core2200
        review = bool(
            max(n1600, n2200) >= 5
            and float(peak_values[peak_index]) >= 5.0
            and elongation <= 10.0
            and cloud_distance >= 5.0
            and not line_flag
        )
        conservative = bool(
            support in {"1600_only", "2200_only"}
            and primary_count >= 8
            and primary_core >= 2
            and float(peak_values[peak_index]) >= 6.0
            and height >= 3
            and width >= 3
            and minor >= 0.6
            and elongation <= 8.0
            and counterpart_median >= 0.0
            and cloud_distance >= 5.0
            and invalid_distance >= 2.0
            and not stripe_flag
            and not line_flag
        )
        if elongation < 1.2:
            shape_class = "compact_hotspot"
        elif elongation <= 8.0:
            shape_class = "moderately_elongated"
        else:
            shape_class = "linear_artifact_suspect"
        easting, northing = geometry.georef.map_xy(peak_y, peak_x, pixel_center=True)
        centroid_y, centroid_x = float(np.mean(yy)), float(np.mean(xx))
        centroid_easting, centroid_northing = geometry.georef.map_xy(
            centroid_y, centroid_x, pixel_center=True
        )
        source_id, source_y, source_x = _peak_source(
            geometry,
            union_y=peak_y,
            union_x=peak_x,
            band=primary_band,
            sign=sign,
        )
        contributing_ids = _contributing_product_ids(
            geometry, yy, xx, threshold=threshold, sign=sign
        )
        rows.append(
            {
                "global_rank": None,
                "strip_rank": None,
                "tail": tail,
                "acquisition_utc_minute": geometry.acquisition_minute,
                "strip_product_ids": ";".join(
                    source.product_id for source in geometry.sources
                ),
                "contributing_product_ids": ";".join(contributing_ids),
                "spectral_support": support,
                "contains_strict_dual_component": bool(np.any(strict_hits)),
                "review_shortlist": review,
                "conservative_single_window": conservative,
                "shape_class": shape_class,
                "region_id": int(label_id),
                "pixel_count": size,
                "n_1600_threshold_pixels": n1600,
                "n_2200_threshold_pixels": n2200,
                "n_coincident_dual_pixels": int(np.count_nonzero(coincident_hits)),
                "n_strict_dual_pixels": int(np.count_nonzero(strict_hits)),
                "n_1600_core_pixels_z5": core1600,
                "n_2200_core_pixels_z5": core2200,
                "primary_peak_band": primary_band,
                "primary_peak_z": float(peak_values[peak_index]),
                "peak_1600_z": float(np.max(weak_values)),
                "peak_2200_z": float(np.max(strong_values)),
                "peak_dual_z": float(np.max(mosaic.dual[yy, xx])),
                "counterpart_median_z_on_primary_extent": counterpart_median,
                "peak_union_y": peak_y,
                "peak_union_x": peak_x,
                "peak_source_product_id": source_id,
                "peak_source_y": source_y,
                "peak_source_x": source_x,
                "centroid_union_y": centroid_y,
                "centroid_union_x": centroid_x,
                "bbox_height": height,
                "bbox_width": width,
                "minor_axis_sd_pixels": minor,
                "major_axis_sd_pixels": major,
                "elongation": elongation,
                "major_axis_angle_deg_from_east": angle,
                "shape_stripe_direction_flag": stripe_flag,
                "scene_spanning_line_flag": line_flag,
                "minimum_distance_to_cloud_pixels": cloud_distance,
                "minimum_distance_to_invalid_pixels": invalid_distance,
                "easting_m": float(easting),
                "northing_m": float(northing),
                "centroid_easting_m": float(centroid_easting),
                "centroid_northing_m": float(centroid_northing),
                "epsg": int(geometry.georef.epsg),
            }
        )
    kept_labels = selected_labels[labels]
    rows.sort(
        key=lambda row: (
            bool(row["contains_strict_dual_component"]),
            bool(row["conservative_single_window"]),
            float(row["primary_peak_z"]),
            int(row["pixel_count"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["strip_rank"] = rank
    return rows, kept_labels


def _product_counts(
    source: SceneSource, *, threshold: float, minimum_pixels: int
) -> dict[str, Any]:
    valid = source.valid & np.isfinite(source.weak) & np.isfinite(source.strong)
    positive_weak = valid & (source.weak >= threshold)
    reverse_weak = valid & (-source.weak >= threshold)
    positive_strong = valid & (source.strong >= threshold)
    reverse_strong = valid & (-source.strong >= threshold)
    positive_dual = positive_weak & positive_strong
    reverse_dual = reverse_weak & reverse_strong
    return {
        "product_id": source.product_id,
        "acquisition_utc": source.acquisition_utc,
        "acquisition_utc_minute": source.acquisition_minute,
        "cloud_proxy_fraction": source.cloud_fraction,
        "cloud_dilated_fraction": source.cloud_dilated_fraction,
        "analysis_valid_pixels": int(np.count_nonzero(valid)),
        "minimum_component_pixels": minimum_pixels,
        "positive_1600_tail_pixels": int(np.count_nonzero(positive_weak)),
        "reverse_1600_tail_pixels": int(np.count_nonzero(reverse_weak)),
        "positive_1600_retained_component_count": _component_count(positive_weak, minimum_pixels)[0],
        "reverse_1600_retained_component_count": _component_count(reverse_weak, minimum_pixels)[0],
        "positive_2200_tail_pixels": int(np.count_nonzero(positive_strong)),
        "reverse_2200_tail_pixels": int(np.count_nonzero(reverse_strong)),
        "positive_2200_retained_component_count": _component_count(positive_strong, minimum_pixels)[0],
        "reverse_2200_retained_component_count": _component_count(reverse_strong, minimum_pixels)[0],
        "positive_strict_dual_retained_component_count": _component_count(positive_dual, minimum_pixels)[0],
        "reverse_strict_dual_retained_component_count": _component_count(reverse_dual, minimum_pixels)[0],
    }


def _load_known_site(path: Path, site_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Known-site CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("site_id") == site_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one known-site row for {site_id!r}")
    row = matches[0]
    try:
        epsg = int(str(row["epsg"]).strip())
        easting = float(str(row["easting_m"]).strip())
        northing = float(str(row["northing_m"]).strip())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Known-site row {site_id!r} has invalid projected coordinates") from exc
    if not np.isfinite([easting, northing]).all():
        raise ValueError(f"Known-site row {site_id!r} has non-finite coordinates")
    return {
        "site_id": site_id,
        "name": row.get("name"),
        "epsg": epsg,
        "easting_m": easting,
        "northing_m": northing,
        "source_kind": row.get("source_kind"),
        "source_url": row.get("source_url"),
        "note": row.get("note"),
    }


def _map_to_pixel_center(
    georef: GeoReference, easting: float, northing: float
) -> tuple[float, float]:
    x0, dx, rx, y0, ry, dy = georef.geotransform
    linear = np.asarray([[dx, rx], [ry, dy]], dtype=float)
    column_corner, row_corner = np.linalg.solve(
        linear, np.asarray([easting - x0, northing - y0], dtype=float)
    )
    return float(row_corner - 0.5), float(column_corner - 0.5)


def _nearest_mask_pixel_distance_m(
    georef: GeoReference,
    mask: np.ndarray,
    *,
    easting: float,
    northing: float,
) -> float | None:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    x0, dx, rx, y0, ry, dy = georef.geotransform
    columns = xx.astype(float) + 0.5
    rows = yy.astype(float) + 0.5
    eastings = x0 + columns * dx + rows * rx
    northings = y0 + columns * ry + rows * dy
    distances = np.hypot(eastings - easting, northings - northing)
    return float(np.min(distances))


def _known_site_audit(
    geometry: StripGeometry,
    mosaic: TailMosaic,
    site: Mapping[str, Any],
    *,
    radius_pixels: int,
    minimum_pixels: int,
) -> dict[str, Any] | None:
    if geometry.georef.epsg != int(site["epsg"]):
        return None
    easting = float(site["easting_m"])
    northing = float(site["northing_m"])
    row_float, column_float = _map_to_pixel_center(geometry.georef, easting, northing)
    center_y, center_x = int(round(row_float)), int(round(column_float))
    if not (0 <= center_y < geometry.shape[0] and 0 <= center_x < geometry.shape[1]):
        return None
    y0 = max(0, center_y - radius_pixels)
    y1 = min(geometry.shape[0], center_y + radius_pixels + 1)
    x0 = max(0, center_x - radius_pixels)
    x1 = min(geometry.shape[1], center_x + radius_pixels + 1)
    window = np.s_[y0:y1, x0:x1]
    window_valid = geometry.valid[window]

    def valid_peak(values: np.ndarray) -> float | None:
        selected = values[window][window_valid]
        if selected.size == 0:
            return None
        return float(np.max(selected))

    retained1600 = _kept_component_mask(mosaic.mask1600, minimum_pixels)
    retained2200 = _kept_component_mask(mosaic.mask2200, minimum_pixels)
    nearest_center_easting, nearest_center_northing = geometry.georef.map_xy(
        center_y, center_x, pixel_center=True
    )
    return {
        "acquisition_utc_minute": geometry.acquisition_minute,
        "strip_product_ids": [source.product_id for source in geometry.sources],
        "site_id": site["site_id"],
        "site_name": site.get("name"),
        "site_source_kind": site.get("source_kind"),
        "site_source_url": site.get("source_url"),
        "site_note": site.get("note"),
        "epsg": int(site["epsg"]),
        "site_easting_m": easting,
        "site_northing_m": northing,
        "mask_definition": (
            "site-centered square window on the strip-union grid; "
            "not the historical fixed R2 extent"
        ),
        "radius_pixels": radius_pixels,
        "window_shape": [y1 - y0, x1 - x0],
        "window_total_pixels": int((y1 - y0) * (x1 - x0)),
        "window_analysis_valid_pixels": int(np.count_nonzero(window_valid)),
        "site_union_row_float": row_float,
        "site_union_column_float": column_float,
        "nearest_pixel_center_row": center_y,
        "nearest_pixel_center_column": center_x,
        "site_to_nearest_pixel_center_m": float(
            math.hypot(nearest_center_easting - easting, nearest_center_northing - northing)
        ),
        "positive_1600_threshold_pixels_in_window": int(
            np.count_nonzero(mosaic.mask1600[window])
        ),
        "positive_1600_peak_z_in_window": valid_peak(mosaic.band1600),
        "positive_2200_threshold_pixels_in_window": int(
            np.count_nonzero(mosaic.mask2200[window])
        ),
        "positive_2200_peak_z_in_window": valid_peak(mosaic.band2200),
        "positive_dual_peak_z_in_window": valid_peak(mosaic.dual),
        "nearest_positive_1600_retained_component_pixel_m": _nearest_mask_pixel_distance_m(
            geometry.georef,
            retained1600,
            easting=easting,
            northing=northing,
        ),
        "nearest_positive_2200_retained_component_pixel_m": _nearest_mask_pixel_distance_m(
            geometry.georef,
            retained2200,
            easting=easting,
            northing=northing,
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _gallery_selection(
    rows: Sequence[Mapping[str, Any]], *, per_support: int
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    strips = sorted({str(row["acquisition_utc_minute"]) for row in rows})
    for strip in strips:
        members = [row for row in rows if row["acquisition_utc_minute"] == strip]
        for support in SUPPORT_ORDER:
            matching = [
                row
                for row in members
                if row["spectral_support"] == support and row["review_shortlist"]
            ]
            matching.sort(
                key=lambda row: (
                    bool(row["conservative_single_window"]),
                    float(row["primary_peak_z"]),
                    int(row["pixel_count"]),
                ),
                reverse=True,
            )
            selected.extend(matching[:per_support])
    selected_keys = {
        (row["acquisition_utc_minute"], row["region_id"], row["tail"])
        for row in selected
    }
    for row in rows:
        key = (row["acquisition_utc_minute"], row["region_id"], row["tail"])
        if row["conservative_single_window"] and key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    return selected


def _save_gallery(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[SceneSource],
    *,
    threshold: float,
    minimum_pixels: int,
    crop_half_size: int,
) -> None:
    selected = list(rows)
    if not selected:
        fig, axis = plt.subplots(figsize=(9, 3), constrained_layout=True)
        axis.text(0.5, 0.5, "No review-shortlist regions", ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    by_id = {source.product_id: source for source in sources}
    grouped: dict[str, list[SceneSource]] = {}
    for source in sources:
        grouped.setdefault(source.acquisition_minute, []).append(source)
    fig, axes = plt.subplots(
        len(selected), 4, figsize=(14, max(3.0 * len(selected), 6.0))
    )
    axes = np.asarray(axes).reshape(len(selected), 4)
    current_minute: str | None = None
    current_geometry: StripGeometry | None = None
    current_mosaic: TailMosaic | None = None
    for row_index, (row, row_axes) in enumerate(zip(selected, axes), start=1):
        row_minute = str(row["acquisition_utc_minute"])
        if row_minute != current_minute:
            current_geometry = _strip_geometry(
                row_minute,
                sorted(grouped[row_minute], key=lambda value: value.product_id),
            )
            current_mosaic = _tail_mosaic(
                current_geometry,
                threshold=threshold,
                minimum_pixels=minimum_pixels,
                sign=1,
            )
            current_minute = row_minute
        assert current_geometry is not None and current_mosaic is not None
        source = by_id[str(row["peak_source_product_id"])]
        center_y, center_x = int(row["peak_source_y"]), int(row["peak_source_x"])
        browse_bounds = crop_bounds(source.valid.shape, center_y, center_x, crop_half_size)
        browse_path = source.product.directory / f"{source.product_id}_1.jpg"
        if browse_path.is_file():
            browse, _metadata = browse_crop(
                mpimg.imread(browse_path), source.valid.shape, browse_bounds
            )
            row_axes[0].imshow(browse)
        else:
            row_axes[0].text(0.5, 0.5, "No browse", ha="center", va="center")
        union_center_y = int(row["peak_union_y"])
        union_center_x = int(row["peak_union_x"])
        bounds = crop_bounds(
            current_geometry.shape, union_center_y, union_center_x, crop_half_size
        )
        y0, y1, x0, x1 = bounds
        union_valid = current_geometry.valid[y0:y1, x0:x1]
        weak = np.ma.masked_where(
            ~union_valid, current_mosaic.band1600[y0:y1, x0:x1]
        )
        strong = np.ma.masked_where(
            ~union_valid, current_mosaic.band2200[y0:y1, x0:x1]
        )
        row_axes[1].imshow(weak, cmap="coolwarm", vmin=-3, vmax=8)
        row_axes[2].imshow(strong, cmap="coolwarm", vmin=-3, vmax=8)
        overlay = np.zeros((*weak.shape, 3), dtype=np.float32)
        weak_hit = current_mosaic.mask1600[y0:y1, x0:x1]
        strong_hit = current_mosaic.mask2200[y0:y1, x0:x1]
        same_product_coincident = current_mosaic.coincident[y0:y1, x0:x1]
        overlay[..., 0] = weak_hit
        overlay[..., 2] = strong_hit
        overlay[..., 1] = same_product_coincident
        row_axes[3].imshow(overlay)
        browse_y0, _browse_y1, browse_x0, _browse_x1 = browse_bounds
        row_axes[0].plot(
            center_x - browse_x0,
            center_y - browse_y0,
            marker="+",
            color="yellow",
            ms=10,
            mew=2,
        )
        local_y, local_x = union_center_y - y0, union_center_x - x0
        for axis in row_axes[1:]:
            axis.plot(local_x, local_y, marker="+", color="yellow", ms=10, mew=2)
        for axis in row_axes:
            axis.set_axis_off()
        conservative_label = " [conservative]" if row["conservative_single_window"] else ""
        row_axes[0].set_title(
            f"{row_index}. {str(row['acquisition_utc_minute'])[:10]}  "
            f"{row['spectral_support']}{conservative_label}\n"
            f"peak={float(row['primary_peak_z']):.2f}, area={int(row['pixel_count'])} px, "
            f"E={float(row['easting_m']):.0f}, N={float(row['northing_m']):.0f}",
            fontsize=8,
        )
        row_axes[1].set_title("1600 nm-window strip-union local z", fontsize=8)
        row_axes[2].set_title("2200–2390 nm-window strip-union local z", fontsize=8)
        row_axes[3].set_title(
            "strip union: red=1600, blue=2200, pale=same-product dual",
            fontsize=8,
        )
    fig.suptitle(
        "Usable-scene single-window methane-like review regions\n"
        "Balanced by acquisition strip and spectral support; not confirmed plumes",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(
        top=0.82 if len(selected) == 1 else 0.97,
        bottom=0.02,
        left=0.02,
        right=0.99,
        hspace=0.42,
        wspace=0.12,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _rank_globally(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda row: (
            bool(row["contains_strict_dual_component"]),
            bool(row["conservative_single_window"]),
            float(row["primary_peak_z"]),
            int(row["pixel_count"]),
            str(row["acquisition_utc_minute"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["global_rank"] = rank


def _support_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {support: sum(row["spectral_support"] == support for row in rows) for support in SUPPORT_ORDER}


def summarize(
    batch_dir: Path,
    output_dir: Path,
    *,
    threshold: float = 3.0,
    minimum_pixels: int = 3,
    gallery_per_support: int = 1,
    crop_half_size: int = 40,
    known_site_csv: Path | None = None,
    known_site_id: str = "keystone_general",
    site_radius_pixels: int = 10,
) -> dict[str, Any]:
    batch_dir = batch_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    for path in output_paths.values():
        path.unlink(missing_ok=True)
    try:
        threshold = _finite_positive(threshold, "threshold")
        if minimum_pixels < 2:
            raise ValueError("minimum_pixels must be at least 2")
        if gallery_per_support < 1:
            raise ValueError("gallery_per_support must be positive")
        if crop_half_size < 1:
            raise ValueError("crop_half_size must be positive")
        if site_radius_pixels < 0:
            raise ValueError("site_radius_pixels must be non-negative")
        sources, skipped, batch_provenance = _load_sources(batch_dir)
        site = None
        if known_site_csv is not None:
            known_site_csv = known_site_csv.expanduser().resolve()
            site = _load_known_site(known_site_csv, known_site_id)
        product_rows = [
            _product_counts(source, threshold=threshold, minimum_pixels=minimum_pixels)
            for source in sources
        ]
        grouped: dict[str, list[SceneSource]] = {}
        for source in sources:
            grouped.setdefault(source.acquisition_minute, []).append(source)
        positive_rows: list[dict[str, Any]] = []
        reverse_rows: list[dict[str, Any]] = []
        strip_summaries: list[dict[str, Any]] = []
        known_site_audits: list[dict[str, Any]] = []
        for minute, members in sorted(grouped.items()):
            geometry = _strip_geometry(minute, sorted(members, key=lambda value: value.product_id))
            same_source_cross_band_conflict = _same_source_cross_band_conflict_mask(
                geometry, threshold=threshold
            )
            product_valid = sum(int(np.count_nonzero(source.valid)) for source in members)
            union_valid = int(np.count_nonzero(geometry.valid))
            positive_mosaic = _tail_mosaic(
                geometry, threshold=threshold, minimum_pixels=minimum_pixels, sign=1
            )
            strip_positive, _positive_labels = _extract_regions(
                geometry,
                positive_mosaic,
                threshold=threshold,
                minimum_pixels=minimum_pixels,
                tail="positive_methane_like",
                sign=1,
            )
            positive_band_counts = {
                "1600_tail_pixels": int(np.count_nonzero(positive_mosaic.mask1600)),
                "2200_tail_pixels": int(np.count_nonzero(positive_mosaic.mask2200)),
                "coincident_dual_tail_pixels": int(np.count_nonzero(positive_mosaic.coincident)),
                "cross_product_only_dual_tail_pixels": int(
                    np.count_nonzero(
                        positive_mosaic.mask1600
                        & positive_mosaic.mask2200
                        & ~positive_mosaic.coincident
                    )
                ),
                "strict_dual_retained_pixels": int(np.count_nonzero(positive_mosaic.strict_dual)),
                "1600_retained_component_count": _component_count(positive_mosaic.mask1600, minimum_pixels)[0],
                "2200_retained_component_count": _component_count(positive_mosaic.mask2200, minimum_pixels)[0],
                "strict_dual_retained_component_count": _component_count(positive_mosaic.coincident, minimum_pixels)[0],
            }
            if site is not None:
                site_audit = _known_site_audit(
                    geometry,
                    positive_mosaic,
                    site,
                    radius_pixels=site_radius_pixels,
                    minimum_pixels=minimum_pixels,
                )
                if site_audit is not None:
                    known_site_audits.append(site_audit)
            positive_any_mask = positive_mosaic.mask1600 | positive_mosaic.mask2200
            del positive_mosaic, _positive_labels
            gc.collect()
            reverse_mosaic = _tail_mosaic(
                geometry, threshold=threshold, minimum_pixels=minimum_pixels, sign=-1
            )
            strip_reverse, _reverse_labels = _extract_regions(
                geometry,
                reverse_mosaic,
                threshold=threshold,
                minimum_pixels=minimum_pixels,
                tail="reverse_sign_control",
                sign=-1,
            )
            reverse_band_counts = {
                "1600_tail_pixels": int(np.count_nonzero(reverse_mosaic.mask1600)),
                "2200_tail_pixels": int(np.count_nonzero(reverse_mosaic.mask2200)),
                "coincident_dual_tail_pixels": int(np.count_nonzero(reverse_mosaic.coincident)),
                "cross_product_only_dual_tail_pixels": int(
                    np.count_nonzero(
                        reverse_mosaic.mask1600
                        & reverse_mosaic.mask2200
                        & ~reverse_mosaic.coincident
                    )
                ),
                "strict_dual_retained_pixels": int(np.count_nonzero(reverse_mosaic.strict_dual)),
                "1600_retained_component_count": _component_count(reverse_mosaic.mask1600, minimum_pixels)[0],
                "2200_retained_component_count": _component_count(reverse_mosaic.mask2200, minimum_pixels)[0],
                "strict_dual_retained_component_count": _component_count(reverse_mosaic.coincident, minimum_pixels)[0],
            }
            positive_reverse_conflicts = int(
                np.count_nonzero(
                    positive_any_mask
                    & (reverse_mosaic.mask1600 | reverse_mosaic.mask2200)
                )
            )
            same_source_conflicts = int(
                np.count_nonzero(
                    positive_any_mask
                    & (reverse_mosaic.mask1600 | reverse_mosaic.mask2200)
                    & same_source_cross_band_conflict
                )
            )
            cross_product_only_conflicts = positive_reverse_conflicts - same_source_conflicts
            del reverse_mosaic, _reverse_labels
            gc.collect()
            positive_rows.extend(strip_positive)
            reverse_rows.extend(strip_reverse)
            strip_summaries.append(
                {
                    "acquisition_utc_minute": minute,
                    "product_ids": [source.product_id for source in members],
                    "product_count": len(members),
                    "product_analysis_valid_pixels": product_valid,
                    "analysis_valid_pixels": union_valid,
                    "duplicate_product_pixels": product_valid - union_valid,
                    "positive_reverse_union_conflict_pixels": positive_reverse_conflicts,
                    "same_source_cross_band_conflict_pixels": same_source_conflicts,
                    "cross_product_only_conflict_pixels": cross_product_only_conflicts,
                    "positive_region_count": len(strip_positive),
                    "reverse_region_count": len(strip_reverse),
                    "positive_support_counts": _support_counts(strip_positive),
                    "reverse_support_counts": _support_counts(strip_reverse),
                    "positive_review_shortlist_count": sum(bool(row["review_shortlist"]) for row in strip_positive),
                    "reverse_review_shortlist_count": sum(bool(row["review_shortlist"]) for row in strip_reverse),
                    "positive_conservative_single_window_count": sum(bool(row["conservative_single_window"]) for row in strip_positive),
                    "reverse_conservative_single_window_count": sum(bool(row["conservative_single_window"]) for row in strip_reverse),
                    "positive_band_counts": positive_band_counts,
                    "reverse_band_counts": reverse_band_counts,
                }
            )
            del geometry
            gc.collect()
        _rank_globally(positive_rows)
        _rank_globally(reverse_rows)
        _write_csv(output_paths["single_band_candidate_regions.csv"], positive_rows, REGION_FIELDS)
        _write_csv(
            output_paths["single_band_review_shortlist.csv"],
            [row for row in positive_rows if row["review_shortlist"]],
            REGION_FIELDS,
        )
        _write_csv(output_paths["single_band_reverse_control_regions.csv"], reverse_rows, REGION_FIELDS)
        _write_csv(output_paths["single_band_product_counts.csv"], product_rows, PRODUCT_COUNT_FIELDS)
        gallery_rows = _gallery_selection(positive_rows, per_support=gallery_per_support)
        _save_gallery(
            output_paths["single_band_candidate_gallery.png"],
            gallery_rows,
            sources,
            threshold=threshold,
            minimum_pixels=minimum_pixels,
            crop_half_size=crop_half_size,
        )
        product_valid = sum(int(row["analysis_valid_pixels"]) for row in product_rows)
        union_valid = sum(int(row["analysis_valid_pixels"]) for row in strip_summaries)
        script_path = Path(__file__).resolve()
        summary: dict[str, Any] = {
            "analysis_version": ANALYSIS_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "provenance": {
                "script_path": str(script_path),
                "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
                "batch_directory": str(batch_dir),
                "run_manifest": str(batch_dir / "run_manifest.json"),
                **batch_provenance,
                "selection": "quality_class == usable only",
                "usable_product_ids": [source.product_id for source in sources],
                "skipped_products": skipped,
                "known_site_csv": str(known_site_csv) if known_site_csv is not None else None,
                "normalized_arguments": {
                    "batch_dir": str(batch_dir),
                    "output_dir": str(output_dir),
                    "threshold": threshold,
                    "minimum_pixels": minimum_pixels,
                    "gallery_per_support": gallery_per_support,
                    "crop_half_size": crop_half_size,
                    "known_site_csv": (
                        str(known_site_csv) if known_site_csv is not None else None
                    ),
                    "known_site_id": known_site_id,
                    "site_radius_pixels": site_radius_pixels,
                },
            },
            "parameters": {
                "weak_window_nm": [1580.0, 1750.0],
                "strong_window_nm": [2200.0, 2390.0],
                "local_z_threshold": threshold,
                "component_connectivity": 8,
                "minimum_band_threshold_pixels": minimum_pixels,
                "coincident_dual_pixel_semantics": (
                    "both windows cross threshold within the same source product; "
                    "band-wise strip-OR coincidences supplied by different products "
                    "are counted separately"
                ),
                "local_z_gaussian_sigma_pixels": float(
                    batch_provenance["batch_analysis_config"]["local_z_sigma"]
                ),
                "strip_grouping": "SceneCenterTime floored to UTC minute; heuristic, not official orbit/strip ID",
                "within_strip_overlap_reducer": (
                    "logical OR for threshold masks; maximum sign-adjusted local-z "
                    "across products used for per-pixel region metrics, ranking, "
                    "known-site audit, and gallery; potentially optimistic"
                ),
                "review_shortlist": "at least 5 threshold pixels in one band, peak>=5, elongation<=10, whole-region cloud distance>=5 px, not scene-spanning line",
                "conservative_single_window": "pure one-window support; extent>=8, >=2 core pixels at z>=5, peak>=6, bbox>=3x3, minor-axis SD>=0.6, elongation<=8, counterpart median>=0, cloud distance>=5 px, invalid-boundary distance>=2 px, no fixed-direction thin-stripe or scene-spanning-line flag",
            },
            "aggregate": {
                "usable_product_count": len(sources),
                "acquisition_strip_count": len(strip_summaries),
                "product_analysis_valid_pixels": product_valid,
                "analysis_valid_pixels_after_same_strip_union": union_valid,
                "duplicate_product_pixels": product_valid - union_valid,
                "positive_reverse_union_conflict_pixels": sum(
                    int(row["positive_reverse_union_conflict_pixels"])
                    for row in strip_summaries
                ),
                "same_source_cross_band_conflict_pixels": sum(
                    int(row["same_source_cross_band_conflict_pixels"])
                    for row in strip_summaries
                ),
                "cross_product_only_conflict_pixels": sum(
                    int(row["cross_product_only_conflict_pixels"])
                    for row in strip_summaries
                ),
                "positive_region_count": len(positive_rows),
                "reverse_region_count": len(reverse_rows),
                "positive_support_counts": _support_counts(positive_rows),
                "reverse_support_counts": _support_counts(reverse_rows),
                "positive_review_shortlist_count": sum(bool(row["review_shortlist"]) for row in positive_rows),
                "reverse_review_shortlist_count": sum(bool(row["review_shortlist"]) for row in reverse_rows),
                "positive_conservative_single_window_count": sum(bool(row["conservative_single_window"]) for row in positive_rows),
                "reverse_conservative_single_window_count": sum(bool(row["conservative_single_window"]) for row in reverse_rows),
                "positive_strict_dual_region_count": sum(bool(row["contains_strict_dual_component"]) for row in positive_rows),
                "reverse_strict_dual_region_count": sum(bool(row["contains_strict_dual_component"]) for row in reverse_rows),
                "positive_1600_tail_pixels": sum(int(row["positive_band_counts"]["1600_tail_pixels"]) for row in strip_summaries),
                "reverse_1600_tail_pixels": sum(int(row["reverse_band_counts"]["1600_tail_pixels"]) for row in strip_summaries),
                "positive_2200_tail_pixels": sum(int(row["positive_band_counts"]["2200_tail_pixels"]) for row in strip_summaries),
                "reverse_2200_tail_pixels": sum(int(row["reverse_band_counts"]["2200_tail_pixels"]) for row in strip_summaries),
                "positive_coincident_dual_tail_pixels": sum(int(row["positive_band_counts"]["coincident_dual_tail_pixels"]) for row in strip_summaries),
                "reverse_coincident_dual_tail_pixels": sum(int(row["reverse_band_counts"]["coincident_dual_tail_pixels"]) for row in strip_summaries),
                "positive_cross_product_only_dual_tail_pixels": sum(int(row["positive_band_counts"]["cross_product_only_dual_tail_pixels"]) for row in strip_summaries),
                "reverse_cross_product_only_dual_tail_pixels": sum(int(row["reverse_band_counts"]["cross_product_only_dual_tail_pixels"]) for row in strip_summaries),
                "positive_strict_dual_retained_pixels": sum(int(row["positive_band_counts"]["strict_dual_retained_pixels"]) for row in strip_summaries),
                "reverse_strict_dual_retained_pixels": sum(int(row["reverse_band_counts"]["strict_dual_retained_pixels"]) for row in strip_summaries),
                "positive_1600_retained_component_count": sum(int(row["positive_band_counts"]["1600_retained_component_count"]) for row in strip_summaries),
                "reverse_1600_retained_component_count": sum(int(row["reverse_band_counts"]["1600_retained_component_count"]) for row in strip_summaries),
                "positive_2200_retained_component_count": sum(int(row["positive_band_counts"]["2200_retained_component_count"]) for row in strip_summaries),
                "reverse_2200_retained_component_count": sum(int(row["reverse_band_counts"]["2200_retained_component_count"]) for row in strip_summaries),
                "positive_strict_dual_retained_component_count": sum(int(row["positive_band_counts"]["strict_dual_retained_component_count"]) for row in strip_summaries),
                "reverse_strict_dual_retained_component_count": sum(int(row["reverse_band_counts"]["strict_dual_retained_component_count"]) for row in strip_summaries),
            },
            "per_strip": strip_summaries,
            "known_site_audits": known_site_audits,
            "gallery_selection": [
                {
                    "global_rank": row["global_rank"],
                    "acquisition_utc_minute": row["acquisition_utc_minute"],
                    "spectral_support": row["spectral_support"],
                    "primary_peak_z": row["primary_peak_z"],
                    "easting_m": row["easting_m"],
                    "northing_m": row["northing_m"],
                }
                for row in gallery_rows
            ],
            "interpretation": (
                "Single-window positive regions and their exact sign-reversed controls "
                "are exploratory review objects. They are not FDR-controlled methane "
                "detections, confirmed plumes, source attributions, or emission rates."
            ),
            "outputs": {name: str(path) for name, path in output_paths.items()},
        }
        output_paths["single_band_summary.json"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return summary
    except Exception:
        for path in output_paths.values():
            path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--minimum-pixels", type=int, default=3)
    parser.add_argument("--gallery-per-support", type=int, default=1)
    parser.add_argument("--crop-half-size", type=int, default=40)
    parser.add_argument("--known-site-csv", type=Path)
    parser.add_argument("--known-site-id", default="keystone_general")
    parser.add_argument("--site-radius-pixels", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize(
        args.batch_dir,
        args.output_dir,
        threshold=args.threshold,
        minimum_pixels=args.minimum_pixels,
        gallery_per_support=args.gallery_per_support,
        crop_half_size=args.crop_half_size,
        known_site_csv=args.known_site_csv,
        known_site_id=args.known_site_id,
        site_radius_pixels=args.site_radius_pixels,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

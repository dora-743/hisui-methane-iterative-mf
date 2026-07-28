#!/usr/bin/env python3
"""Summarize positive and reverse tails from a completed HISUI scene screen.

Only product IDs listed in ``run_manifest.json/current_product_ids`` are read.
This is important when an output directory also contains scene directories
preserved from older runs.  Scored scenes require ``score_maps.npz`` written
with ``--save-score-maps``; excluded-cloud and QA-incomplete scenes remain in
the output with null score fields.

The reported tail rates and connected-component counts are descriptive
diagnostics.  They are not false-discovery-rate-controlled detections or
emission-rate retrievals.

With ``--spatial-union``, scored products are additionally grouped by their
scene-centre UTC minute.  Their masks are aligned from the source L1G GeoTIFF
georeferencing and OR-combined so an overlapping ground pixel is counted once
per heuristic acquisition strip.  The minute grouping is documented as a
reproducible sensitivity analysis, not an authoritative orbit/strip ID.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile
from scipy import ndimage

from hisui_l1g_io import discover_l1g_product, read_georeference


ANALYSIS_VERSION = "2026-07-28-v2"
COMPONENT_MINIMUM_PIXELS = 3
SCORED_QUALITY_CLASSES = frozenset({"usable", "partial"})
EXCLUDED_QUALITY_CLASSES = frozenset({"excluded_cloud", "qa_incomplete"})
DESCRIPTIVE_CAVEAT = (
    "Positive and symmetric reverse tails, their rates, and connected-component "
    "counts are descriptive diagnostics. They are not FDR-controlled methane "
    "detections, plume confirmations, or emission-rate retrievals."
)
STRIP_GROUPING_HEURISTIC = (
    "Scored products whose scene-centre acquisition timestamps fall in the "
    "same UTC calendar minute are treated as one acquisition strip. The minute "
    "is floored, not rounded. This is a reproducible heuristic, not an "
    "authoritative HISUI orbit or strip identifier."
)
SPATIAL_UNION_CAVEAT = (
    "Within each heuristic acquisition strip, analysis-valid, positive-tail, "
    "and reverse-tail masks are independently OR-combined on the common "
    "projected L1G pixel grid. Thus a ground pixel is counted at most once per "
    "strip and tail membership means at least one overlapping product met the "
    "threshold. Different acquisition strips remain independent observations."
)

SCORE_FIELD_NAMES = (
    "analysis_valid_pixels",
    "positive_tail_pixels",
    "reverse_tail_pixels",
    "positive_tail_rate_per_million_analysis_valid",
    "reverse_tail_rate_per_million_analysis_valid",
    "tail_pixel_difference_positive_minus_reverse",
    "tail_balance_index",
    "positive_to_reverse_tail_ratio",
    "positive_components_ge3",
    "positive_component_pixels_ge3",
    "reverse_components_ge3",
    "reverse_component_pixels_ge3",
    "component_count_difference_positive_minus_reverse",
)

CSV_FIELD_NAMES = (
    "product_id",
    "acquisition_utc",
    "quality_class",
    "official_scoring_eligible",
    "status",
    "threshold_dual_local_z",
    "component_source_minimum_pixels",
    *SCORE_FIELD_NAMES,
)

STRIP_CSV_FIELD_NAMES = (
    "acquisition_utc_minute",
    "product_count",
    "product_ids",
    "first_acquisition_utc",
    "last_acquisition_utc",
    "time_span_seconds",
    "epsg",
    "union_grid_height",
    "union_grid_width",
    "product_analysis_valid_pixels",
    "analysis_valid_pixels",
    "duplicate_product_pixels",
    "duplicate_fraction_of_product_analysis_valid",
    "ground_pixels_observed_by_multiple_products",
    "maximum_valid_product_multiplicity",
    "product_positive_tail_pixels",
    "positive_tail_pixels",
    "duplicate_positive_product_pixels",
    "product_reverse_tail_pixels",
    "reverse_tail_pixels",
    "duplicate_reverse_product_pixels",
    "positive_tail_rate_per_million_analysis_valid",
    "reverse_tail_rate_per_million_analysis_valid",
    "tail_pixel_difference_positive_minus_reverse",
    "tail_balance_index",
    "positive_to_reverse_tail_ratio",
    "positive_components_ge3",
    "positive_component_pixels_ge3",
    "reverse_components_ge3",
    "reverse_component_pixels_ge3",
    "component_count_difference_positive_minus_reverse",
    "positive_reverse_union_conflict_pixels",
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _product_ids(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label} must contain non-empty strings")
        if item in {".", ".."} or "/" in item or "\\" in item:
            raise ValueError(f"Unsafe product_id in {label}: {item!r}")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate product IDs")
    return result


def _as_finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _as_nonnegative_int(value: object, *, label: str) -> int:
    number = _as_finite_float(value, label=label)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return int(number)


def _same_threshold(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def _parse_acquisition_minute(
    value: object, *, product_id: str
) -> tuple[str, datetime]:
    """Return a UTC-minute strip key and the full timezone-aware timestamp."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{product_id}: acquisition_utc must be a non-empty ISO timestamp "
            "for spatial-union analysis"
        )
    timestamp = value.strip()
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"{product_id}: acquisition_utc is not a valid ISO timestamp: "
            f"{timestamp!r}"
        ) from error
    if parsed.tzinfo is None:
        raise ValueError(
            f"{product_id}: acquisition_utc must include a UTC offset"
        )
    parsed_utc = parsed.astimezone(timezone.utc)
    minute = parsed_utc.replace(second=0, microsecond=0)
    return minute.strftime("%Y-%m-%dT%H:%MZ"), parsed_utc


def _scene_threshold(
    summary: Mapping[str, Any],
    *,
    product_id: str,
    batch_threshold: float,
) -> tuple[float, int]:
    config = summary.get("analysis_config")
    if not isinstance(config, dict):
        raise ValueError(f"{product_id}: summary lacks analysis_config")
    threshold = _as_finite_float(
        config.get("cluster_threshold"),
        label=f"{product_id} analysis_config.cluster_threshold",
    )
    if threshold < 0:
        raise ValueError(f"{product_id}: cluster threshold must be non-negative")
    if not _same_threshold(threshold, batch_threshold):
        raise ValueError(
            f"{product_id}: scene cluster threshold {threshold} does not match "
            f"batch threshold {batch_threshold}"
        )
    source_minimum = _as_nonnegative_int(
        config.get("minimum_cluster_pixels"),
        label=f"{product_id} analysis_config.minimum_cluster_pixels",
    )
    model = summary.get("model")
    if model is not None:
        if not isinstance(model, dict):
            raise ValueError(f"{product_id}: model must be an object")
        model_threshold = _as_finite_float(
            model.get("cluster_threshold"),
            label=f"{product_id} model.cluster_threshold",
        )
        if not _same_threshold(threshold, model_threshold):
            raise ValueError(
                f"{product_id}: analysis_config and model cluster thresholds differ"
            )
        model_minimum = _as_nonnegative_int(
            model.get("minimum_cluster_pixels"),
            label=f"{product_id} model.minimum_cluster_pixels",
        )
        if source_minimum != model_minimum:
            raise ValueError(
                f"{product_id}: analysis_config and model component minima differ"
            )
    return threshold, source_minimum


def _validate_summary_provenance(
    summary: Mapping[str, Any], *, product_id: str
) -> None:
    if summary.get("product_id") != product_id:
        raise ValueError(
            f"{product_id}: summary product_id is {summary.get('product_id')!r}"
        )
    product_path = summary.get("product_path")
    if not isinstance(product_path, str) or Path(product_path).name != product_id:
        raise ValueError(
            f"{product_id}: summary product_path does not identify this product"
        )


def _tail_metrics(
    score_path: Path, *, threshold: float, product_id: str
) -> dict[str, int | float | None]:
    analysis_valid, positive, reverse = _load_tail_masks(
        score_path, threshold=threshold, product_id=product_id
    )
    denominator = int(np.count_nonzero(analysis_valid))
    positive_count = int(np.count_nonzero(positive))
    reverse_count = int(np.count_nonzero(reverse))
    total_tail = positive_count + reverse_count
    return {
        "analysis_valid_pixels": denominator,
        "positive_tail_pixels": positive_count,
        "reverse_tail_pixels": reverse_count,
        "positive_tail_rate_per_million_analysis_valid": (
            positive_count * 1_000_000.0 / denominator if denominator else None
        ),
        "reverse_tail_rate_per_million_analysis_valid": (
            reverse_count * 1_000_000.0 / denominator if denominator else None
        ),
        "tail_pixel_difference_positive_minus_reverse": (
            positive_count - reverse_count
        ),
        "tail_balance_index": (
            (positive_count - reverse_count) / total_tail if total_tail else None
        ),
        "positive_to_reverse_tail_ratio": (
            positive_count / reverse_count if reverse_count else None
        ),
    }


def _load_tail_masks(
    score_path: Path, *, threshold: float, product_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the analysis-valid and symmetric threshold masks for one product."""

    if not score_path.is_file():
        raise FileNotFoundError(
            f"{product_id}: missing {score_path.name}; rerun the screen with "
            "--save-score-maps"
        )
    with np.load(score_path, allow_pickle=False) as archive:
        required = {"valid", "weak_local_z", "strong_local_z"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"{product_id}: {score_path.name} lacks {', '.join(missing)}"
            )
        valid = np.asarray(archive["valid"], dtype=bool)
        weak = np.asarray(archive["weak_local_z"], dtype=float)
        strong = np.asarray(archive["strong_local_z"], dtype=float)
    if valid.ndim != 2 or weak.shape != valid.shape or strong.shape != valid.shape:
        raise ValueError(
            f"{product_id}: valid, weak_local_z, and strong_local_z must be "
            "same-shape 2-D arrays"
        )
    analysis_valid = valid & np.isfinite(weak) & np.isfinite(strong)
    positive = analysis_valid & (weak >= threshold) & (strong >= threshold)
    reverse = analysis_valid & (-weak >= threshold) & (-strong >= threshold)
    return analysis_valid, positive, reverse


def _component_metrics(
    csv_path: Path,
    *,
    product_id: str,
    summary: Mapping[str, Any],
    source_minimum_pixels: int,
) -> dict[str, int]:
    if source_minimum_pixels > COMPONENT_MINIMUM_PIXELS:
        raise ValueError(
            f"{product_id}: candidate_components.csv was generated with minimum "
            f"size {source_minimum_pixels}, so complete >=3-pixel counts cannot be "
            "reconstructed"
        )
    records: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"product_id", "tail", "pixel_count"}
            if reader.fieldnames is None:
                raise ValueError(f"{product_id}: empty candidate component CSV")
            missing = sorted(required - set(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"{product_id}: candidate component CSV lacks "
                    f"{', '.join(missing)}"
                )
            records = list(reader)

    counts = Counter(record.get("tail") for record in records)
    allowed_tails = {"positive_ch4", "reverse_sign_control"}
    unknown = sorted(str(tail) for tail in counts if tail not in allowed_tails)
    if unknown:
        raise ValueError(
            f"{product_id}: candidate component CSV has unknown tails: "
            f"{', '.join(unknown)}"
        )
    for record in records:
        if record.get("product_id") != product_id:
            raise ValueError(
                f"{product_id}: candidate component row has product_id "
                f"{record.get('product_id')!r}"
            )

    expected_positive = _as_nonnegative_int(
        summary.get("positive_component_count"),
        label=f"{product_id} positive_component_count",
    )
    expected_reverse = _as_nonnegative_int(
        summary.get("reverse_control_component_count"),
        label=f"{product_id} reverse_control_component_count",
    )
    if counts["positive_ch4"] != expected_positive:
        raise ValueError(
            f"{product_id}: positive component CSV count "
            f"{counts['positive_ch4']} != summary count {expected_positive}"
        )
    if counts["reverse_sign_control"] != expected_reverse:
        raise ValueError(
            f"{product_id}: reverse component CSV count "
            f"{counts['reverse_sign_control']} != summary count {expected_reverse}"
        )

    selected: dict[str, list[int]] = {
        "positive_ch4": [],
        "reverse_sign_control": [],
    }
    for index, record in enumerate(records, start=2):
        size = _as_nonnegative_int(
            record.get("pixel_count"),
            label=f"{product_id} candidate CSV row {index} pixel_count",
        )
        if size < source_minimum_pixels:
            raise ValueError(
                f"{product_id}: candidate CSV contains a component smaller than "
                "the configured source minimum"
            )
        if size >= COMPONENT_MINIMUM_PIXELS:
            selected[str(record["tail"])].append(size)
    positive_sizes = selected["positive_ch4"]
    reverse_sizes = selected["reverse_sign_control"]
    return {
        "positive_components_ge3": len(positive_sizes),
        "positive_component_pixels_ge3": sum(positive_sizes),
        "reverse_components_ge3": len(reverse_sizes),
        "reverse_component_pixels_ge3": sum(reverse_sizes),
        "component_count_difference_positive_minus_reverse": (
            len(positive_sizes) - len(reverse_sizes)
        ),
    }


def _not_scored_row(
    summary: Mapping[str, Any],
    *,
    product_id: str,
    threshold: float,
    source_minimum_pixels: int,
) -> dict[str, Any]:
    quality_class = str(summary.get("quality_class"))
    row: dict[str, Any] = {
        "product_id": product_id,
        "acquisition_utc": summary.get("acquisition_utc"),
        "quality_class": quality_class,
        "official_scoring_eligible": False,
        "status": f"not_scored_{quality_class}",
        "threshold_dual_local_z": threshold,
        "component_source_minimum_pixels": source_minimum_pixels,
    }
    row.update({field: None for field in SCORE_FIELD_NAMES})
    return row


def _validate_batch(
    batch_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str], float]:
    manifest_path = batch_dir / "run_manifest.json"
    manifest = _read_json(manifest_path, label="run manifest")
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Run manifest status must be 'complete', got {manifest.get('status')!r}"
        )
    current_ids = _product_ids(
        manifest.get("current_product_ids"),
        label="run_manifest.current_product_ids",
    )
    completed_ids = _product_ids(
        manifest.get("completed_product_ids"),
        label="run_manifest.completed_product_ids",
    )
    if completed_ids != current_ids:
        raise ValueError(
            "run_manifest.completed_product_ids must exactly match "
            "current_product_ids for a completed summary"
        )

    batch_summary = _read_json(
        batch_dir / "batch_summary.json", label="batch summary"
    )
    batch_ids = _product_ids(
        batch_summary.get("current_product_ids"),
        label="batch_summary.current_product_ids",
    )
    if batch_ids != current_ids:
        raise ValueError(
            "batch_summary.current_product_ids does not match the run manifest"
        )
    scene_count = _as_nonnegative_int(
        batch_summary.get("scene_count"), label="batch_summary.scene_count"
    )
    if scene_count != len(current_ids):
        raise ValueError("batch_summary.scene_count does not match current_product_ids")
    config = batch_summary.get("analysis_config")
    if not isinstance(config, dict):
        raise ValueError("batch_summary lacks analysis_config")
    batch_threshold = _as_finite_float(
        config.get("cluster_threshold"),
        label="batch_summary.analysis_config.cluster_threshold",
    )
    if batch_threshold < 0:
        raise ValueError("Batch cluster threshold must be non-negative")
    return manifest, batch_summary, current_ids, batch_threshold


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["status"] == "scored"]
    valid = sum(int(row["analysis_valid_pixels"]) for row in scored)
    positive = sum(int(row["positive_tail_pixels"]) for row in scored)
    reverse = sum(int(row["reverse_tail_pixels"]) for row in scored)
    total_tail = positive + reverse
    quality_counts = Counter(str(row["quality_class"]) for row in rows)
    thresholds = sorted(
        {float(row["threshold_dual_local_z"]) for row in scored}
    )
    return {
        "current_scene_count": len(rows),
        "scored_scene_count": len(scored),
        "excluded_scene_count": len(rows) - len(scored),
        "qa_incomplete_scene_count": quality_counts["qa_incomplete"],
        "cloud_excluded_scene_count": quality_counts["excluded_cloud"],
        "quality_class_counts": dict(sorted(quality_counts.items())),
        "threshold_values_dual_local_z": thresholds,
        "analysis_valid_pixels": valid,
        "positive_tail_pixels": positive,
        "reverse_tail_pixels": reverse,
        "positive_tail_rate_per_million_analysis_valid": (
            positive * 1_000_000.0 / valid if valid else None
        ),
        "reverse_tail_rate_per_million_analysis_valid": (
            reverse * 1_000_000.0 / valid if valid else None
        ),
        "tail_pixel_difference_positive_minus_reverse": positive - reverse,
        "tail_balance_index": (
            (positive - reverse) / total_tail if total_tail else None
        ),
        "positive_to_reverse_tail_ratio": positive / reverse if reverse else None,
        "positive_components_ge3": sum(
            int(row["positive_components_ge3"]) for row in scored
        ),
        "positive_component_pixels_ge3": sum(
            int(row["positive_component_pixels_ge3"]) for row in scored
        ),
        "reverse_components_ge3": sum(
            int(row["reverse_components_ge3"]) for row in scored
        ),
        "reverse_component_pixels_ge3": sum(
            int(row["reverse_component_pixels_ge3"]) for row in scored
        ),
        "component_count_difference_positive_minus_reverse": sum(
            int(row["component_count_difference_positive_minus_reverse"])
            for row in scored
        ),
    }


def _connected_components_ge3(mask: np.ndarray) -> tuple[int, int]:
    """Count 8-connected components and pixels after a three-pixel filter."""

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    selected = sizes >= COMPONENT_MINIMUM_PIXELS
    return int(np.count_nonzero(selected)), int(np.sum(sizes[selected]))


def _strip_union_metrics(
    acquisition_minute: str,
    sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Merge one acquisition-minute group on its projected pixel grid."""

    if not sources:
        raise ValueError("A spatial-union strip must contain at least one product")
    reference = sources[0]["georeference"]
    if reference.epsg is None:
        raise ValueError(
            f"{acquisition_minute}: source GeoTIFF lacks a projected EPSG code"
        )
    reference_transform = np.asarray(reference.geotransform, dtype=float)
    linear = np.asarray(
        [
            [reference_transform[1], reference_transform[2]],
            [reference_transform[4], reference_transform[5]],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(linear)) or abs(float(np.linalg.det(linear))) < 1e-12:
        raise ValueError(f"{acquisition_minute}: singular projected pixel transform")

    raw_offsets: list[tuple[int, int]] = []
    for source in sources:
        georef = source["georeference"]
        if georef.epsg != reference.epsg:
            raise ValueError(
                f"{acquisition_minute}: products do not share one projected EPSG code"
            )
        transform = np.asarray(georef.geotransform, dtype=float)
        source_linear = np.asarray(
            [[transform[1], transform[2]], [transform[4], transform[5]]],
            dtype=float,
        )
        if not np.allclose(source_linear, linear, rtol=1e-12, atol=1e-9):
            raise ValueError(
                f"{acquisition_minute}: products do not share one pixel transform"
            )
        delta = np.asarray(
            [
                transform[0] - reference_transform[0],
                transform[3] - reference_transform[3],
            ],
            dtype=float,
        )
        column_offset, row_offset = np.linalg.solve(linear, delta)
        rounded_column = int(round(float(column_offset)))
        rounded_row = int(round(float(row_offset)))
        if not np.allclose(
            [column_offset, row_offset],
            [rounded_column, rounded_row],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{acquisition_minute}: product {source['product_id']} is not "
                "aligned to the common integer L1G pixel grid"
            )
        raw_offsets.append((rounded_row, rounded_column))

    minimum_row = min(row for row, _column in raw_offsets)
    minimum_column = min(column for _row, column in raw_offsets)
    maximum_row = max(
        row + int(source["analysis_valid"].shape[0])
        for (row, _column), source in zip(raw_offsets, sources)
    )
    maximum_column = max(
        column + int(source["analysis_valid"].shape[1])
        for (_row, column), source in zip(raw_offsets, sources)
    )
    union_shape = (
        maximum_row - minimum_row,
        maximum_column - minimum_column,
    )
    if union_shape[0] <= 0 or union_shape[1] <= 0:
        raise ValueError(f"{acquisition_minute}: invalid spatial-union grid")

    analysis_valid = np.zeros(union_shape, dtype=bool)
    positive = np.zeros(union_shape, dtype=bool)
    reverse = np.zeros(union_shape, dtype=bool)
    observation_dtype = np.uint16 if len(sources) > 255 else np.uint8
    valid_observations = np.zeros(union_shape, dtype=observation_dtype)
    product_valid = 0
    product_positive = 0
    product_reverse = 0
    product_provenance: list[dict[str, Any]] = []
    for (raw_row, raw_column), source in zip(raw_offsets, sources):
        row_offset = raw_row - minimum_row
        column_offset = raw_column - minimum_column
        source_valid = source["analysis_valid"]
        source_positive = source["positive"]
        source_reverse = source["reverse"]
        height, width = source_valid.shape
        target = np.s_[
            row_offset : row_offset + height,
            column_offset : column_offset + width,
        ]
        analysis_valid[target] |= source_valid
        positive[target] |= source_positive
        reverse[target] |= source_reverse
        valid_observations[target] += source_valid
        product_valid += int(np.count_nonzero(source_valid))
        product_positive += int(np.count_nonzero(source_positive))
        product_reverse += int(np.count_nonzero(source_reverse))
        product_provenance.append(
            {
                "product_id": source["product_id"],
                "product_path": source["product_path"],
                "image_path": source["image_path"],
                "score_maps": source["score_maps"],
                "acquisition_utc": source["acquisition_utc"],
                "score_shape": [int(height), int(width)],
                "normalized_gdal_geotransform": list(
                    source["georeference"].geotransform
                ),
                "epsg": source["georeference"].epsg,
                "source_raster_type": source["georeference"].raster_type,
                "pixel_scale": list(source["georeference"].pixel_scale),
                "union_row_offset": row_offset,
                "union_column_offset": column_offset,
            }
        )

    union_valid = int(np.count_nonzero(analysis_valid))
    union_positive = int(np.count_nonzero(positive))
    union_reverse = int(np.count_nonzero(reverse))
    total_tail = union_positive + union_reverse
    positive_components, positive_component_pixels = _connected_components_ge3(
        positive
    )
    reverse_components, reverse_component_pixels = _connected_components_ge3(reverse)
    timestamps = sorted(source["parsed_acquisition_utc"] for source in sources)
    union_origin = np.asarray(
        [reference_transform[0], reference_transform[3]], dtype=float
    ) + linear @ np.asarray([minimum_column, minimum_row], dtype=float)
    union_geotransform = (
        float(union_origin[0]),
        float(reference_transform[1]),
        float(reference_transform[2]),
        float(union_origin[1]),
        float(reference_transform[4]),
        float(reference_transform[5]),
    )
    duplicate_valid = product_valid - union_valid
    return {
        "acquisition_utc_minute": acquisition_minute,
        "product_count": len(sources),
        "product_ids": [str(source["product_id"]) for source in sources],
        "first_acquisition_utc": timestamps[0].isoformat().replace("+00:00", "Z"),
        "last_acquisition_utc": timestamps[-1].isoformat().replace("+00:00", "Z"),
        "time_span_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
        "epsg": int(reference.epsg),
        "union_grid_height": int(union_shape[0]),
        "union_grid_width": int(union_shape[1]),
        "union_normalized_gdal_geotransform": list(union_geotransform),
        "product_analysis_valid_pixels": product_valid,
        "analysis_valid_pixels": union_valid,
        "duplicate_product_pixels": duplicate_valid,
        "duplicate_fraction_of_product_analysis_valid": (
            duplicate_valid / product_valid if product_valid else None
        ),
        "ground_pixels_observed_by_multiple_products": int(
            np.count_nonzero(valid_observations >= 2)
        ),
        "maximum_valid_product_multiplicity": int(valid_observations.max()),
        "product_positive_tail_pixels": product_positive,
        "positive_tail_pixels": union_positive,
        "duplicate_positive_product_pixels": product_positive - union_positive,
        "product_reverse_tail_pixels": product_reverse,
        "reverse_tail_pixels": union_reverse,
        "duplicate_reverse_product_pixels": product_reverse - union_reverse,
        "positive_tail_rate_per_million_analysis_valid": (
            union_positive * 1_000_000.0 / union_valid if union_valid else None
        ),
        "reverse_tail_rate_per_million_analysis_valid": (
            union_reverse * 1_000_000.0 / union_valid if union_valid else None
        ),
        "tail_pixel_difference_positive_minus_reverse": (
            union_positive - union_reverse
        ),
        "tail_balance_index": (
            (union_positive - union_reverse) / total_tail if total_tail else None
        ),
        "positive_to_reverse_tail_ratio": (
            union_positive / union_reverse if union_reverse else None
        ),
        "positive_components_ge3": positive_components,
        "positive_component_pixels_ge3": positive_component_pixels,
        "reverse_components_ge3": reverse_components,
        "reverse_component_pixels_ge3": reverse_component_pixels,
        "component_count_difference_positive_minus_reverse": (
            positive_components - reverse_components
        ),
        "positive_reverse_union_conflict_pixels": int(
            np.count_nonzero(positive & reverse)
        ),
        "source_products": product_provenance,
    }


def _aggregate_strip_unions(strips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    product_valid = sum(int(strip["product_analysis_valid_pixels"]) for strip in strips)
    union_valid = sum(int(strip["analysis_valid_pixels"]) for strip in strips)
    product_positive = sum(
        int(strip["product_positive_tail_pixels"]) for strip in strips
    )
    positive = sum(int(strip["positive_tail_pixels"]) for strip in strips)
    product_reverse = sum(int(strip["product_reverse_tail_pixels"]) for strip in strips)
    reverse = sum(int(strip["reverse_tail_pixels"]) for strip in strips)
    total_tail = positive + reverse
    duplicate_valid = product_valid - union_valid
    return {
        "acquisition_strip_count": len(strips),
        "multi_product_strip_count": sum(
            int(strip["product_count"] > 1) for strip in strips
        ),
        "scored_product_count": sum(int(strip["product_count"]) for strip in strips),
        "product_analysis_valid_pixels": product_valid,
        "analysis_valid_pixels": union_valid,
        "duplicate_product_pixels": duplicate_valid,
        "duplicate_fraction_of_product_analysis_valid": (
            duplicate_valid / product_valid if product_valid else None
        ),
        "ground_pixels_observed_by_multiple_products": sum(
            int(strip["ground_pixels_observed_by_multiple_products"])
            for strip in strips
        ),
        "maximum_valid_product_multiplicity": max(
            (int(strip["maximum_valid_product_multiplicity"]) for strip in strips),
            default=0,
        ),
        "product_positive_tail_pixels": product_positive,
        "positive_tail_pixels": positive,
        "duplicate_positive_product_pixels": product_positive - positive,
        "product_reverse_tail_pixels": product_reverse,
        "reverse_tail_pixels": reverse,
        "duplicate_reverse_product_pixels": product_reverse - reverse,
        "positive_tail_rate_per_million_analysis_valid": (
            positive * 1_000_000.0 / union_valid if union_valid else None
        ),
        "reverse_tail_rate_per_million_analysis_valid": (
            reverse * 1_000_000.0 / union_valid if union_valid else None
        ),
        "tail_pixel_difference_positive_minus_reverse": positive - reverse,
        "tail_balance_index": (
            (positive - reverse) / total_tail if total_tail else None
        ),
        "positive_to_reverse_tail_ratio": positive / reverse if reverse else None,
        "positive_components_ge3": sum(
            int(strip["positive_components_ge3"]) for strip in strips
        ),
        "positive_component_pixels_ge3": sum(
            int(strip["positive_component_pixels_ge3"]) for strip in strips
        ),
        "reverse_components_ge3": sum(
            int(strip["reverse_components_ge3"]) for strip in strips
        ),
        "reverse_component_pixels_ge3": sum(
            int(strip["reverse_component_pixels_ge3"]) for strip in strips
        ),
        "component_count_difference_positive_minus_reverse": sum(
            int(strip["component_count_difference_positive_minus_reverse"])
            for strip in strips
        ),
        "positive_reverse_union_conflict_pixels": sum(
            int(strip["positive_reverse_union_conflict_pixels"]) for strip in strips
        ),
    }


def _spatial_union_summary(
    batch_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    strip_csv_path: Path,
) -> dict[str, Any]:
    grouped_rows: dict[str, list[tuple[Mapping[str, Any], datetime]]] = {}
    for row in rows:
        if row["status"] != "scored":
            continue
        product_id = str(row["product_id"])
        minute, parsed = _parse_acquisition_minute(
            row.get("acquisition_utc"), product_id=product_id
        )
        grouped_rows.setdefault(minute, []).append((row, parsed))

    strips: list[dict[str, Any]] = []
    for minute, members in sorted(grouped_rows.items()):
        sources: list[dict[str, Any]] = []
        for row, parsed in members:
            product_id = str(row["product_id"])
            scene_dir = batch_dir / product_id
            summary = _read_json(
                scene_dir / "summary.json", label=f"{product_id} scene summary"
            )
            _validate_summary_provenance(summary, product_id=product_id)
            product = discover_l1g_product(str(summary["product_path"]))
            if product.product_id != product_id:
                raise ValueError(
                    f"{product_id}: discovered source product ID is "
                    f"{product.product_id!r}"
                )
            score_path = scene_dir / "score_maps.npz"
            analysis_valid, positive, reverse = _load_tail_masks(
                score_path,
                threshold=float(row["threshold_dual_local_z"]),
                product_id=product_id,
            )
            with tifffile.TiffFile(str(product.image_path)) as tif:
                if len(tif.pages) != 1:
                    raise ValueError(
                        f"{product_id}: source image must be a single-page GeoTIFF"
                    )
                raster_shape = tuple(int(value) for value in tif.pages[0].shape[:2])
            if analysis_valid.shape != raster_shape:
                raise ValueError(
                    f"{product_id}: score-map shape {analysis_valid.shape} does "
                    f"not match source GeoTIFF raster shape {raster_shape}"
                )
            if int(np.count_nonzero(analysis_valid)) != int(
                row["analysis_valid_pixels"]
            ):
                raise ValueError(
                    f"{product_id}: spatial-union valid count does not match "
                    "the per-scene summary"
                )
            sources.append(
                {
                    "product_id": product_id,
                    "product_path": str(product.directory.resolve()),
                    "image_path": str(product.image_path.resolve()),
                    "score_maps": str(score_path.resolve()),
                    "acquisition_utc": row["acquisition_utc"],
                    "parsed_acquisition_utc": parsed,
                    "georeference": read_georeference(product.image_path),
                    "analysis_valid": analysis_valid,
                    "positive": positive,
                    "reverse": reverse,
                }
            )
        strips.append(_strip_union_metrics(minute, sources))

    strip_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with strip_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRIP_CSV_FIELD_NAMES)
        writer.writeheader()
        for strip in strips:
            csv_row = {field: strip.get(field) for field in STRIP_CSV_FIELD_NAMES}
            csv_row["product_ids"] = ";".join(strip["product_ids"])
            writer.writerow(csv_row)

    return {
        "status": "complete",
        "grouping_heuristic": STRIP_GROUPING_HEURISTIC,
        "mask_union_semantics": SPATIAL_UNION_CAVEAT,
        "parameters": {
            "group_key": "UTC scene-centre timestamp floored to calendar minute",
            "grid_source": (
                "normalized GDAL corner geotransform and projected EPSG read "
                "from each source L1G GeoTIFF"
            ),
            "raster_type_normalization": (
                "RasterPixelIsPoint tiepoints are converted to the GDAL "
                "pixel-corner convention by hisui_l1g_io.read_georeference"
            ),
            "grid_alignment_tolerance_pixels": 1e-6,
            "mask_reducer_within_strip": "logical OR",
            "component_connectivity": 8,
            "reported_component_minimum_pixels": COMPONENT_MINIMUM_PIXELS,
            "different_strips_deduplicated": False,
        },
        "per_strip": strips,
        "aggregate": _aggregate_strip_unions(strips),
        "outputs": {"csv": str(strip_csv_path)},
    }


def summarize_batch(
    batch_dir: str | Path,
    *,
    csv_output: str | Path | None = None,
    json_output: str | Path | None = None,
    spatial_union: bool = False,
    strip_csv_output: str | Path | None = None,
) -> dict[str, Any]:
    """Create per-scene and optional acquisition-strip union summaries."""

    batch_dir = Path(batch_dir).expanduser().resolve()
    if strip_csv_output is not None and not spatial_union:
        raise ValueError("strip_csv_output requires spatial_union=True")
    csv_path = (
        Path(csv_output).expanduser().resolve()
        if csv_output is not None
        else batch_dir / "tail_balance_by_scene.csv"
    )
    json_path = (
        Path(json_output).expanduser().resolve()
        if json_output is not None
        else batch_dir / "tail_balance_summary.json"
    )
    strip_csv_path = (
        Path(strip_csv_output).expanduser().resolve()
        if strip_csv_output is not None
        else batch_dir / "tail_balance_by_acquisition_strip.csv"
    )
    active_output_paths = [csv_path, json_path]
    if spatial_union:
        active_output_paths.append(strip_csv_path)
    if len(active_output_paths) != len(set(active_output_paths)):
        raise ValueError("CSV and JSON output paths must all differ")
    for output_path in active_output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    # Invalidate every artifact that could be mistaken for the current run
    # before reading inputs. A non-union rerun also removes the default strip
    # CSV from a previous union run.
    stale_output_paths = set(active_output_paths)
    if not spatial_union:
        stale_output_paths.add(batch_dir / "tail_balance_by_acquisition_strip.csv")
    for output_path in stale_output_paths:
        output_path.unlink(missing_ok=True)

    manifest, _batch_summary, product_ids, batch_threshold = _validate_batch(
        batch_dir
    )
    rows: list[dict[str, Any]] = []
    for product_id in product_ids:
        # Construct paths only from the manifest's direct-child product IDs.  Do
        # not glob the batch directory or follow stale_scene_directories.
        scene_dir = batch_dir / product_id
        summary = _read_json(
            scene_dir / "summary.json", label=f"{product_id} scene summary"
        )
        _validate_summary_provenance(summary, product_id=product_id)
        threshold, source_minimum = _scene_threshold(
            summary,
            product_id=product_id,
            batch_threshold=batch_threshold,
        )
        quality_class = str(summary.get("quality_class"))
        eligible = summary.get("official_scoring_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(
                f"{product_id}: official_scoring_eligible must be boolean"
            )
        if quality_class in SCORED_QUALITY_CLASSES and not eligible:
            raise ValueError(
                f"{product_id}: scored quality class is marked ineligible"
            )
        if quality_class in EXCLUDED_QUALITY_CLASSES and eligible:
            raise ValueError(
                f"{product_id}: excluded quality class is marked eligible"
            )
        if not eligible:
            rows.append(
                _not_scored_row(
                    summary,
                    product_id=product_id,
                    threshold=threshold,
                    source_minimum_pixels=source_minimum,
                )
            )
            continue

        row: dict[str, Any] = {
            "product_id": product_id,
            "acquisition_utc": summary.get("acquisition_utc"),
            "quality_class": quality_class,
            "official_scoring_eligible": True,
            "status": "scored",
            "threshold_dual_local_z": threshold,
            "component_source_minimum_pixels": source_minimum,
        }
        row.update(
            _tail_metrics(
                scene_dir / "score_maps.npz",
                threshold=threshold,
                product_id=product_id,
            )
        )
        row.update(
            _component_metrics(
                scene_dir / "candidate_components.csv",
                product_id=product_id,
                summary=summary,
                source_minimum_pixels=source_minimum,
            )
        )
        rows.append(row)

    result: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "provenance": {
            "batch_directory": str(batch_dir),
            "run_manifest": str(batch_dir / "run_manifest.json"),
            "batch_summary": str(batch_dir / "batch_summary.json"),
            "manifest_status": manifest.get("status"),
            "manifest_started_utc": manifest.get("started_utc"),
            "manifest_finished_utc": manifest.get("finished_utc"),
            "current_product_ids": product_ids,
            "stale_scene_directories_read": [],
        },
        "parameters": {
            "tail_definition": (
                "positive=min(weak_local_z,strong_local_z)>=threshold; "
                "reverse=min(-weak_local_z,-strong_local_z)>=threshold"
            ),
            "analysis_valid_definition": (
                "score_maps.valid and finite weak_local_z and strong_local_z"
            ),
            "batch_threshold_dual_local_z": batch_threshold,
            "reported_component_minimum_pixels": COMPONENT_MINIMUM_PIXELS,
            "rate_denominator": "analysis-valid pixels",
            "rate_scale": 1_000_000,
        },
        "per_scene": rows,
        "aggregate": _aggregate(rows),
        "descriptive_not_fdr_caveat": DESCRIPTIVE_CAVEAT,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
    }
    try:
        if spatial_union:
            result["spatial_union_by_acquisition_strip"] = _spatial_union_summary(
                batch_dir,
                rows,
                strip_csv_path=strip_csv_path,
            )
            result["outputs"]["strip_csv"] = str(strip_csv_path)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELD_NAMES)
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except BaseException:
        for output_path in active_output_paths:
            output_path.unlink(missing_ok=True)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Completed screen_hisui_l1g_scenes.py output directory",
    )
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--spatial-union",
        action="store_true",
        help=(
            "Also deduplicate projected pixels within acquisition-minute strips "
            "using source L1G GeoTIFF georeferencing"
        ),
    )
    parser.add_argument(
        "--strip-csv-output",
        type=Path,
        help=(
            "Optional acquisition-strip CSV path (requires --spatial-union; "
            "default: BATCH/tail_balance_by_acquisition_strip.csv)"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = summarize_batch(
        args.batch_dir,
        csv_output=args.csv_output,
        json_output=args.json_output,
        spatial_union=args.spatial_union,
        strip_csv_output=args.strip_csv_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

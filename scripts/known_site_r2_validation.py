#!/usr/bin/env python3
"""Targeted R2 validation for a fixed gas-plant proxy rectangle.

The facility rectangle is kept fixed during the present aggregation.  Its
aggregate dual-band evidence is compared with identically sized translated
rectangles.  This avoids reusing only the highest-scoring pixels inside R2, but
it is still a post-hoc reanalysis informed by prior work on the same scene.

The translated-box tail fractions remain empirical diagnostics: they require
local spatial exchangeability, and the historical and current maps are
different analyses of the same acquisition rather than independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage

from spatial_region_crossvalidation import load_rgb, render_rgb


@dataclass(frozen=True)
class Box:
    row: int
    column: int
    height: int
    width: int

    @property
    def row_max(self) -> int:
        return self.row + self.height - 1

    @property
    def column_max(self) -> int:
        return self.column + self.width - 1

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.row + 0.5 * (self.height - 1),
            self.column + 0.5 * (self.width - 1),
        )

    @property
    def section(self) -> tuple[slice, slice]:
        return (
            slice(self.row, self.row + self.height),
            slice(self.column, self.column + self.width),
        )


def rectangles_intersect(first: Box, second: Box) -> bool:
    return not (
        first.row_max < second.row
        or first.row > second.row_max
        or first.column_max < second.column
        or first.column > second.column_max
    )


def translated_boxes(
    shape: tuple[int, int],
    site: Box,
    *,
    exclusion_buffer: int,
    minimum_radius: float,
    maximum_radius: float | None,
    stride: int,
) -> list[Box]:
    if stride <= 0:
        raise ValueError("stride must be positive.")
    expanded = Box(
        max(site.row - exclusion_buffer, 0),
        max(site.column - exclusion_buffer, 0),
        min(site.row_max + exclusion_buffer, shape[0] - 1)
        - max(site.row - exclusion_buffer, 0)
        + 1,
        min(site.column_max + exclusion_buffer, shape[1] - 1)
        - max(site.column - exclusion_buffer, 0)
        + 1,
    )
    site_y, site_x = site.center
    output: list[Box] = []
    for row in range(0, shape[0] - site.height + 1, stride):
        for column in range(0, shape[1] - site.width + 1, stride):
            candidate = Box(row, column, site.height, site.width)
            if rectangles_intersect(candidate, expanded):
                continue
            candidate_y, candidate_x = candidate.center
            radius = math.hypot(candidate_y - site_y, candidate_x - site_x)
            if radius < minimum_radius:
                continue
            if maximum_radius is not None and radius > maximum_radius:
                continue
            output.append(candidate)
    return output


def box_statistic(
    values: np.ndarray,
    box: Box,
    *,
    top_fraction: float | None = None,
) -> float:
    sample = np.asarray(values[box.section], dtype=float).ravel()
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        return float("nan")
    if top_fraction is None:
        return float(np.mean(sample))
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    count = max(1, int(math.ceil(top_fraction * len(sample))))
    return float(np.mean(np.partition(sample, -count)[-count:]))


def upper_tail_fraction_plus_one(
    site_score: float, controls: np.ndarray
) -> tuple[float, int, int]:
    controls = np.asarray(controls, dtype=float)
    controls = controls[np.isfinite(controls)]
    exceedances = int(np.sum(controls >= site_score))
    return float((1 + exceedances) / (1 + len(controls))), exceedances, len(controls)


def rgb_features(rgb: np.ndarray, box: Box) -> np.ndarray:
    patch = np.asarray(rgb[box.section], dtype=float)
    return np.concatenate(
        [np.nanmean(patch, axis=(0, 1)), np.nanstd(patch, axis=(0, 1))]
    )


def select_rgb_matched_controls(
    rgb: np.ndarray,
    site: Box,
    candidates: list[Box],
    *,
    count: int,
) -> tuple[list[Box], np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("count must be positive.")
    candidate_features = np.asarray([rgb_features(rgb, box) for box in candidates])
    site_features = rgb_features(rgb, site)
    center = np.nanmedian(candidate_features, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(candidate_features - center), axis=0)
    fallback = np.nanstd(candidate_features, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    distances = np.sqrt(
        np.sum(((candidate_features - site_features) / scale) ** 2, axis=1)
    )
    selected = np.argsort(distances)[: min(count, len(candidates))]
    return [candidates[index] for index in selected], distances[selected], site_features


def largest_joint_component(
    first: np.ndarray,
    second: np.ndarray,
    box: Box,
    *,
    threshold: float,
) -> tuple[int, int]:
    mask = (first[box.section] >= threshold) & (second[box.section] >= threshold)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    return int(mask.sum()), int(sizes[1:].max(initial=0))


def component_with_most_overlap(
    mask: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return the 8-connected component with greatest overlap with reference."""

    mask = np.asarray(mask, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if mask.shape != reference.shape:
        raise ValueError("mask and reference must share a shape.")
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        raise ValueError("mask contains no connected component.")
    overlaps = np.bincount(
        labels[reference].ravel(), minlength=count + 1
    ).astype(int)
    overlaps[0] = 0
    selected_label = int(np.argmax(overlaps))
    if overlaps[selected_label] == 0:
        raise ValueError("No connected component overlaps the reference mask.")
    return labels == selected_label


def mask_geometry(
    mask: np.ndarray,
    *,
    y_origin: int = 0,
    x_origin: int = 0,
) -> dict[str, int]:
    rows, columns = np.where(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        raise ValueError("Cannot describe an empty mask.")
    return {
        "pixels": int(len(rows)),
        "y_min": int(rows.min() + y_origin),
        "y_max": int(rows.max() + y_origin),
        "x_min": int(columns.min() + x_origin),
        "x_max": int(columns.max() + x_origin),
    }


def mask_overlap(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Masks must share a shape.")
    intersection = int(np.sum(first & second))
    union = int(np.sum(first | second))
    return {
        "intersection_pixels": intersection,
        "first_pixels": int(first.sum()),
        "second_pixels": int(second.sum()),
        "fraction_of_first": float(intersection / first.sum()) if first.any() else 0.0,
        "fraction_of_second": float(intersection / second.sum()) if second.any() else 0.0,
        "jaccard": float(intersection / union) if union else 0.0,
    }


def summarize_values_on_mask(
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    sample = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        raise ValueError("Mask contains no finite values.")
    return {
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "p90": float(np.quantile(sample, 0.90)),
        "maximum": float(np.max(sample)),
    }


def peak_on_mask(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    y_origin: int = 0,
    x_origin: int = 0,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    usable = mask & np.isfinite(values)
    if not usable.any():
        raise ValueError("Mask contains no finite values.")
    flat_index = int(np.nanargmax(np.where(usable, values, np.nan)))
    row, column = np.unravel_index(flat_index, values.shape)
    value = float(values[row, column])
    background = values[np.isfinite(values)]
    return {
        "value": value,
        "y": int(row + y_origin),
        "x": int(column + x_origin),
        "whole_map_percentile": float(100.0 * np.mean(background <= value)),
    }


def symmetric_wind_cones(
    shape: tuple[int, int],
    *,
    source_y: float,
    source_x: float,
    wind_from_degrees: float,
    inner_radius: float,
    outer_radius: float,
    half_angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct equal downwind/upwind cones in image coordinates.

    Image x increases east and image y increases south. Meteorological wind
    direction is the bearing from which the wind comes, clockwise from north.
    """

    if not 0.0 <= half_angle_degrees < 90.0:
        raise ValueError("half_angle_degrees must be in [0, 90).")
    if not 0.0 <= inner_radius < outer_radius:
        raise ValueError("Require 0 <= inner_radius < outer_radius.")
    transport_bearing = (wind_from_degrees + 180.0) % 360.0
    angle = math.radians(transport_bearing)
    direction_x = math.sin(angle)
    direction_y = -math.cos(angle)
    rows, columns = np.indices(shape, dtype=float)
    delta_x = columns - source_x
    delta_y = rows - source_y
    along = delta_x * direction_x + delta_y * direction_y
    cross = delta_x * (-direction_y) + delta_y * direction_x
    radius = np.hypot(delta_x, delta_y)
    tangent = math.tan(math.radians(half_angle_degrees))
    downwind = (
        (radius >= inner_radius)
        & (radius <= outer_radius)
        & (along >= 0.0)
        & (np.abs(cross) <= along * tangent)
    )
    upwind_distance = -along
    upwind = (
        (radius >= inner_radius)
        & (radius <= outer_radius)
        & (upwind_distance >= 0.0)
        & (np.abs(cross) <= upwind_distance * tangent)
    )
    return downwind, upwind, transport_bearing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-csv", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--site-y-min", type=int, default=96)
    parser.add_argument("--site-y-max", type=int, default=106)
    parser.add_argument("--site-x-min", type=int, default=95)
    parser.add_argument("--site-x-max", type=int, default=111)
    parser.add_argument("--exclusion-buffer", type=int, default=15)
    parser.add_argument("--minimum-radius", type=float, default=25.0)
    parser.add_argument("--maximum-radius", type=float, default=90.0)
    parser.add_argument("--control-stride", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--rgb-matched-count", type=int, default=200)
    parser.add_argument("--joint-z-threshold", type=float, default=2.0)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--external-facility-y", type=float, default=109.777673)
    parser.add_argument("--external-facility-x", type=float, default=107.045823)
    parser.add_argument("--facility-latitude", type=float, default=31.945833)
    parser.add_argument("--facility-longitude", type=float, default=-103.0425)
    parser.add_argument("--pixel-size-metres", type=float, default=20.0)
    parser.add_argument("--wind-from-degrees", type=float, default=239.2)
    parser.add_argument("--wind-speed-metres-per-second", type=float, default=2.79)
    parser.add_argument("--wind-inner-radius", type=float, default=2.0)
    parser.add_argument("--wind-outer-radius", type=float, default=15.0)
    parser.add_argument("--wind-cone-half-angle", type=float, default=60.0)
    parser.add_argument(
        "--scene-id",
        default="HSHL1G_N320W1032_20221030160051_20231127193053",
    )
    parser.add_argument(
        "--observation-time-utc", default="2022-10-30T16:00:51.479728Z"
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis_dir = args.analysis_dir.resolve()
    historical_dir = args.historical_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else analysis_dir / "known_site_r2_validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stored = np.load(analysis_dir / "score_maps.npz")
    valid = stored["valid"].astype(bool)
    y_origin = int(stored["y_min"])
    x_origin = int(stored["x_min"])
    shape = valid.shape
    site = Box(
        args.site_y_min - y_origin,
        args.site_x_min - x_origin,
        args.site_y_max - args.site_y_min + 1,
        args.site_x_max - args.site_x_min + 1,
    )
    if (
        site.row < 0
        or site.column < 0
        or site.row_max >= shape[0]
        or site.column_max >= shape[1]
    ):
        raise ValueError("The requested site rectangle falls outside the score map.")
    if not np.all(valid[site.section]):
        raise ValueError("The requested site rectangle contains invalid score pixels.")
    site_mask = np.zeros(shape, dtype=bool)
    site_mask[site.section] = True
    facility_local_y = args.external_facility_y - y_origin
    facility_local_x = args.external_facility_x - x_origin
    if not (
        0.0 <= facility_local_y <= shape[0] - 1
        and 0.0 <= facility_local_x <= shape[1] - 1
    ):
        raise ValueError("The external facility coordinate falls outside the score map.")
    if args.pixel_size_metres <= 0.0:
        raise ValueError("pixel-size-metres must be positive.")

    historical_1600 = np.load(
        historical_dir / "dual_band_methane_detection" / "z_1600.npy"
    ).astype(float)
    historical_2200 = np.load(
        historical_dir / "dual_band_methane_detection" / "z_2200.npy"
    ).astype(float)
    historical_joint = np.load(
        historical_dir / "dual_band_methane_detection" / "joint_z.npy"
    ).astype(float)
    historical_alpha_1600 = np.load(
        historical_dir / "qa_median_then_ct_dwt" / "alpha_corrected.npy"
    ).astype(float)
    historical_alpha_2200 = np.load(
        historical_dir
        / "2200nm_residual_thin_refined"
        / "recommended_alpha_corrected_refined.npy"
    ).astype(float)
    historical_core_all = np.load(
        historical_dir / "dual_band_methane_detection" / "methane_core_mask.npy"
    ).astype(bool)
    historical_extent_all = np.load(
        historical_dir / "dual_band_methane_detection" / "methane_extent_mask.npy"
    ).astype(bool)
    historical_consensus_all = np.load(
        historical_dir / "cross_band_consensus_connected_mask.npy"
    ).astype(bool)
    for array in (
        historical_1600,
        historical_2200,
        historical_joint,
        historical_alpha_1600,
        historical_alpha_2200,
        historical_core_all,
        historical_extent_all,
        historical_consensus_all,
    ):
        if array.shape != shape:
            raise ValueError("Historical maps and current score map must share a shape.")

    historical_r2_extent = component_with_most_overlap(
        historical_extent_all, site_mask
    )
    historical_r2_core = historical_core_all & historical_r2_extent
    # R2's consensus pixels form several disjoint cores inside one connected
    # extent, so retain their union inside the selected R2 extent.
    historical_r2_consensus = historical_consensus_all & historical_r2_extent
    region_maps = np.load(analysis_dir / "spatial_region_validation" / "region_maps.npz")
    current_weak_selected_r2 = component_with_most_overlap(
        region_maps["weak_selected_labels"].astype(int) > 0,
        historical_r2_extent,
    )

    maps = {
        "historical_dual_min_z": np.minimum(historical_1600, historical_2200),
        "historical_joint_z": historical_joint,
        "current_dual_min_z": np.minimum(
            stored["weak_z"].astype(float), stored["strong_z"].astype(float)
        ),
        "current_bidirectional_log_e": stored["log_e_average"].astype(float),
        "current_reverse_ch4_log_e": stored["log_e_negative_control"].astype(float),
    }

    local_controls = translated_boxes(
        shape,
        site,
        exclusion_buffer=args.exclusion_buffer,
        minimum_radius=args.minimum_radius,
        maximum_radius=args.maximum_radius,
        stride=args.control_stride,
    )
    global_controls = translated_boxes(
        shape,
        site,
        exclusion_buffer=args.exclusion_buffer,
        minimum_radius=0.0,
        maximum_radius=None,
        stride=args.control_stride,
    )
    if len(local_controls) < 100:
        raise ValueError("Too few local translated controls for an empirical comparison.")

    rgb, rgb_wavelengths = load_rgb(
        args.scene_csv.resolve(),
        shape,
        y_origin,
        x_origin,
        chunksize=args.chunksize,
    )
    rendered_rgb = render_rgb(rgb)
    matched_controls, matched_distances, site_rgb_features = select_rgb_matched_controls(
        rgb,
        site,
        global_controls,
        count=args.rgb_matched_count,
    )

    rows: list[dict[str, float | int | str]] = []
    control_columns: dict[str, np.ndarray] = {}
    for metric_name, values in maps.items():
        for statistic_name, top_fraction in (
            ("whole_box_mean", None),
            (f"top_{100 * args.top_fraction:g}_percent_mean", args.top_fraction),
        ):
            site_score = box_statistic(values, site, top_fraction=top_fraction)
            local_scores = np.asarray(
                [
                    box_statistic(values, box, top_fraction=top_fraction)
                    for box in local_controls
                ],
                dtype=float,
            )
            matched_scores = np.asarray(
                [
                    box_statistic(values, box, top_fraction=top_fraction)
                    for box in matched_controls
                ],
                dtype=float,
            )
            local_tail, local_exceedances, local_count = upper_tail_fraction_plus_one(
                site_score, local_scores
            )
            matched_tail, matched_exceedances, matched_count = upper_tail_fraction_plus_one(
                site_score, matched_scores
            )
            control_columns[f"{metric_name}__{statistic_name}"] = local_scores
            rows.append(
                {
                    "metric": metric_name,
                    "statistic": statistic_name,
                    "site_score": site_score,
                    "local_control_count": local_count,
                    "local_exceedances": local_exceedances,
                    "local_positional_tail_fraction": local_tail,
                    "local_control_median": float(np.nanmedian(local_scores)),
                    "local_control_95th": float(np.nanquantile(local_scores, 0.95)),
                    "local_control_99th": float(np.nanquantile(local_scores, 0.99)),
                    "local_control_max": float(np.nanmax(local_scores)),
                    "rgb_matched_control_count": matched_count,
                    "rgb_matched_exceedances": matched_exceedances,
                    "rgb_matched_positional_tail_fraction": matched_tail,
                    "rgb_matched_control_median": float(np.nanmedian(matched_scores)),
                    "rgb_matched_control_max": float(np.nanmax(matched_scores)),
                }
            )
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(output_dir / "site_metric_summary.csv", index=False)

    control_table = pd.DataFrame(
        {
            "y_min": [box.row + y_origin for box in local_controls],
            "x_min": [box.column + x_origin for box in local_controls],
            **control_columns,
        }
    )
    control_table.to_csv(output_dir / "local_translated_controls.csv", index=False)
    matched_table = pd.DataFrame(
        {
            "y_min": [box.row + y_origin for box in matched_controls],
            "x_min": [box.column + x_origin for box in matched_controls],
            "rgb_robust_distance": matched_distances,
        }
    )
    matched_table.to_csv(output_dir / "rgb_matched_controls.csv", index=False)

    component_rows: list[dict[str, float | int | str]] = []
    for method_name, first, second in (
        ("historical", historical_1600, historical_2200),
        (
            "current",
            stored["weak_z"].astype(float),
            stored["strong_z"].astype(float),
        ),
    ):
        site_pixels, site_largest = largest_joint_component(
            first, second, site, threshold=args.joint_z_threshold
        )
        control_values = np.asarray(
            [
                largest_joint_component(
                    first, second, box, threshold=args.joint_z_threshold
                )
                for box in local_controls
            ],
            dtype=int,
        )
        for endpoint, site_value, column in (
            ("joint_pixels", site_pixels, 0),
            ("largest_joint_component", site_largest, 1),
        ):
            tail_fraction, exceedances, count = upper_tail_fraction_plus_one(
                float(site_value), control_values[:, column].astype(float)
            )
            component_rows.append(
                {
                    "method": method_name,
                    "endpoint": endpoint,
                    "threshold_each_band_z": args.joint_z_threshold,
                    "site_value": site_value,
                    "control_count": count,
                    "control_exceedances": exceedances,
                    "positional_tail_fraction": tail_fraction,
                    "control_95th": float(np.quantile(control_values[:, column], 0.95)),
                    "control_max": int(np.max(control_values[:, column])),
                }
            )
    component_table = pd.DataFrame(component_rows)
    component_table.to_csv(output_dir / "site_spatial_coherence.csv", index=False)

    alpha_summary: dict[str, dict[str, float]] = {}
    alpha_maps = {
        "historical_1600_alpha_arbitrary_scale": historical_alpha_1600,
        "historical_2200_alpha_arbitrary_scale": historical_alpha_2200,
        "current_weak_alpha_modtran_equivalent_ppm": stored["weak_alpha"].astype(float),
        "current_strong_alpha_modtran_equivalent_ppm": stored["strong_alpha"].astype(float),
        "current_combined_alpha_modtran_equivalent_ppm": stored["combined_alpha"].astype(float),
    }
    for name, values in alpha_maps.items():
        alpha_summary[name] = summarize_values_on_mask(values, site_mask)

    fixed_mask_current_scores: dict[str, dict[str, object]] = {}
    current_maps_for_audit = {
        "weak_alpha_modtran_equivalent_ppm": stored["weak_alpha"].astype(float),
        "strong_alpha_modtran_equivalent_ppm": stored["strong_alpha"].astype(float),
        "combined_alpha_modtran_equivalent_ppm": stored["combined_alpha"].astype(float),
        "bidirectional_log_e": stored["log_e_average"].astype(float),
        "weak_z": stored["weak_z"].astype(float),
        "strong_z": stored["strong_z"].astype(float),
    }
    historical_maps_for_audit = {
        "alpha_1600_arbitrary_scale": historical_alpha_1600,
        "alpha_2200_arbitrary_scale": historical_alpha_2200,
        "z_1600": historical_1600,
        "z_2200": historical_2200,
        "joint_z": historical_joint,
    }
    fixed_masks = {
        "historical_r2_core": historical_r2_core,
        "historical_r2_extent": historical_r2_extent,
        "historical_r2_consensus": historical_r2_consensus,
        "current_weak_selected_r2": current_weak_selected_r2,
    }
    for mask_name, mask in fixed_masks.items():
        fixed_mask_current_scores[mask_name] = {
            "geometry": mask_geometry(mask, y_origin=y_origin, x_origin=x_origin),
            "historical_scores": {
                map_name: summarize_values_on_mask(values, mask)
                for map_name, values in historical_maps_for_audit.items()
            },
            "current_scores": {
                map_name: summarize_values_on_mask(values, mask)
                for map_name, values in current_maps_for_audit.items()
            },
        }
    fixed_mask_current_scores["historical_r2_core"]["current_peaks"] = {
        map_name: peak_on_mask(
            values,
            historical_r2_core,
            y_origin=y_origin,
            x_origin=x_origin,
        )
        for map_name, values in current_maps_for_audit.items()
    }

    fixed_mask_rows: list[dict[str, float | int | str]] = []
    for mask_name, mask_result in fixed_mask_current_scores.items():
        geometry = mask_result["geometry"]
        for analysis_name in ("historical_scores", "current_scores"):
            for metric_name, statistics in mask_result[analysis_name].items():
                fixed_mask_rows.append(
                    {
                        "mask": mask_name,
                        "mask_pixels": geometry["pixels"],
                        "analysis": analysis_name.removesuffix("_scores"),
                        "metric": metric_name,
                        **statistics,
                    }
                )
    pd.DataFrame(fixed_mask_rows).to_csv(
        output_dir / "fixed_mask_score_summary.csv", index=False
    )

    mask_reconciliation = {
        "historical_r2_core": mask_geometry(
            historical_r2_core, y_origin=y_origin, x_origin=x_origin
        ),
        "historical_r2_extent": mask_geometry(
            historical_r2_extent, y_origin=y_origin, x_origin=x_origin
        ),
        "historical_r2_consensus": mask_geometry(
            historical_r2_consensus, y_origin=y_origin, x_origin=x_origin
        ),
        "current_weak_selected_r2": mask_geometry(
            current_weak_selected_r2, y_origin=y_origin, x_origin=x_origin
        ),
        "current_vs_historical_core": mask_overlap(
            current_weak_selected_r2, historical_r2_core
        ),
        "current_vs_historical_extent": mask_overlap(
            current_weak_selected_r2, historical_r2_extent
        ),
        "current_vs_historical_consensus": mask_overlap(
            current_weak_selected_r2, historical_r2_consensus
        ),
        "interpretation": (
            "The historical R2 masks and the current weak-selected R2 are not the same "
            "pixel set. Statistics must name the mask explicitly."
        ),
    }

    downwind_mask, upwind_mask, transport_bearing = symmetric_wind_cones(
        shape,
        source_y=facility_local_y,
        source_x=facility_local_x,
        wind_from_degrees=args.wind_from_degrees,
        inner_radius=args.wind_inner_radius,
        outer_radius=args.wind_outer_radius,
        half_angle_degrees=args.wind_cone_half_angle,
    )
    wind_maps = {
        "historical_joint_z": historical_joint,
        "current_bidirectional_log_e": stored["log_e_average"].astype(float),
        "current_weak_alpha_modtran_equivalent_ppm": stored["weak_alpha"].astype(float),
        "current_strong_alpha_modtran_equivalent_ppm": stored["strong_alpha"].astype(float),
        "current_reverse_ch4_log_e": stored["log_e_negative_control"].astype(float),
    }
    wind_rows: list[dict[str, float | int | str]] = []
    for map_name, values in wind_maps.items():
        finite = np.isfinite(values)
        downwind_values = values[downwind_mask & valid & finite]
        upwind_values = values[upwind_mask & valid & finite]
        if len(downwind_values) == 0 or len(upwind_values) == 0:
            raise ValueError(
                f"Wind cone contains no finite values for metric {map_name!r}."
            )
        downwind_top_count = max(
            1, int(math.ceil(args.top_fraction * len(downwind_values)))
        )
        upwind_top_count = max(
            1, int(math.ceil(args.top_fraction * len(upwind_values)))
        )
        downwind_top = np.partition(
            downwind_values, -downwind_top_count
        )[-downwind_top_count:]
        upwind_top = np.partition(upwind_values, -upwind_top_count)[
            -upwind_top_count:
        ]
        wind_rows.append(
            {
                "metric": map_name,
                "downwind_pixels": int(len(downwind_values)),
                "upwind_pixels": int(len(upwind_values)),
                "downwind_mean": float(np.mean(downwind_values)),
                "upwind_mean": float(np.mean(upwind_values)),
                "downwind_minus_upwind_mean": float(
                    np.mean(downwind_values) - np.mean(upwind_values)
                ),
                f"downwind_top_{100 * args.top_fraction:g}_percent_mean": float(
                    np.mean(downwind_top)
                ),
                f"upwind_top_{100 * args.top_fraction:g}_percent_mean": float(
                    np.mean(upwind_top)
                ),
                "downwind_maximum": float(np.max(downwind_values)),
                "upwind_maximum": float(np.max(upwind_values)),
            }
        )
    wind_table = pd.DataFrame(wind_rows)
    wind_table.to_csv(output_dir / "wind_sector_screen.csv", index=False)

    historical_extent_labels, historical_region_count = ndimage.label(
        historical_extent_all, structure=np.ones((3, 3), dtype=int)
    )
    historical_region_catalog = pd.read_csv(
        historical_dir / "dual_band_methane_detection" / "methane_regions.csv"
    )
    region_comparison_rows: list[dict[str, float | int | str]] = []
    region_comparison_maps = {
        "historical_joint_z": historical_joint,
        "current_bidirectional_log_e": stored["log_e_average"].astype(float),
        "current_reverse_ch4_log_e": stored["log_e_negative_control"].astype(float),
        "current_weak_alpha_modtran_equivalent_ppm": stored["weak_alpha"].astype(float),
        "current_strong_alpha_modtran_equivalent_ppm": stored["strong_alpha"].astype(float),
        "current_dual_min_z": np.minimum(
            stored["weak_z"].astype(float), stored["strong_z"].astype(float)
        ),
    }
    for region_label in range(1, historical_region_count + 1):
        region_extent = historical_extent_labels == region_label
        region_core = historical_core_all & region_extent
        extent_geometry = mask_geometry(
            region_extent, y_origin=y_origin, x_origin=x_origin
        )
        catalog_match = historical_region_catalog[
            (historical_region_catalog["row_min"] == extent_geometry["y_min"])
            & (historical_region_catalog["row_max"] == extent_geometry["y_max"])
            & (historical_region_catalog["col_min"] == extent_geometry["x_min"])
            & (historical_region_catalog["col_max"] == extent_geometry["x_max"])
        ]
        if len(catalog_match) != 1:
            raise ValueError(
                "Could not uniquely match a historical extent component to methane_regions.csv."
            )
        historical_region_id = int(catalog_match.iloc[0]["region_id"])
        for mask_name, region_mask in (
            ("extent", region_extent),
            ("core", region_core),
        ):
            geometry = mask_geometry(
                region_mask, y_origin=y_origin, x_origin=x_origin
            )
            for map_name, values in region_comparison_maps.items():
                statistics = summarize_values_on_mask(values, region_mask)
                region_comparison_rows.append(
                    {
                        "historical_region": f"R{historical_region_id}",
                        "mask": mask_name,
                        **geometry,
                        "metric": map_name,
                        **statistics,
                    }
                )
    pd.DataFrame(region_comparison_rows).to_csv(
        output_dir / "historical_region_comparison.csv", index=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rendered_rgb)
    axes[0, 0].set_title(f"Gas-plant R2 (RGB: {rgb_wavelengths} nm)")
    axes[0, 0].add_patch(
        Rectangle(
            (site.column - 0.5, site.row - 0.5),
            site.width,
            site.height,
            fill=False,
            color="cyan",
            linewidth=2.0,
        )
    )
    axes[0, 0].text(
        site.column + site.width + 2,
        site.row,
        "R2 facility box",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
    )

    panels = (
        (axes[0, 1], maps["historical_joint_z"], "Historical destriped joint z", "viridis"),
        (axes[0, 2], maps["current_bidirectional_log_e"], "Current bidirectional cross-fit log e", "magma"),
        (axes[1, 0], maps["current_dual_min_z"], "Current min(weak z, strong z)", "viridis"),
        (axes[1, 1], maps["current_reverse_ch4_log_e"], "Reverse-CH4 log e control", "magma"),
    )
    for axis, values, title, cmap in panels:
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, [0.01, 0.995])
        shown = axis.imshow(values, cmap=cmap, vmin=low, vmax=max(high, low + 1e-9))
        axis.add_patch(
            Rectangle(
                (site.column - 0.5, site.row - 0.5),
                site.width,
                site.height,
                fill=False,
                color="cyan",
                linewidth=1.8,
            )
        )
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.78)

    primary_row = summary_table[
        (summary_table["metric"] == "current_bidirectional_log_e")
        & (summary_table["statistic"] == "whole_box_mean")
    ].iloc[0]
    primary_controls = control_columns[
        "current_bidirectional_log_e__whole_box_mean"
    ]
    axes[1, 2].hist(primary_controls, bins=60, color="0.45", alpha=0.85)
    axes[1, 2].axvline(
        primary_row.site_score,
        color="crimson",
        linewidth=2.2,
        label=f"R2 = {primary_row.site_score:.3f}",
    )
    axes[1, 2].set_title("Local translated-box null: mean log e")
    axes[1, 2].set_xlabel("Whole-box mean bidirectional log e")
    axes[1, 2].set_ylabel("Translated boxes")
    axes[1, 2].legend(frameon=False)
    for axis in axes.ravel()[:5]:
        axis.set_axis_off()
    fig.savefig(output_dir / "known_site_r2_evidence.png", dpi=180)
    plt.close(fig)

    zoom_margin = 18
    zoom_y_min = max(0, site.row - zoom_margin)
    zoom_y_max = min(shape[0] - 1, site.row_max + zoom_margin)
    zoom_x_min = max(0, site.column - zoom_margin)
    zoom_x_max = min(shape[1] - 1, site.column_max + zoom_margin)
    zoom_section = (
        slice(zoom_y_min, zoom_y_max + 1),
        slice(zoom_x_min, zoom_x_max + 1),
    )
    extent = (
        zoom_x_min + x_origin - 0.5,
        zoom_x_max + x_origin + 0.5,
        zoom_y_max + y_origin + 0.5,
        zoom_y_min + y_origin - 0.5,
    )
    zoom_fig, zoom_axes = plt.subplots(
        2, 3, figsize=(14, 9), constrained_layout=True, sharex=True, sharey=True
    )
    zoom_axes[0, 0].imshow(rendered_rgb[zoom_section], extent=extent)
    zoom_axes[0, 0].set_title("RGB and reconciled R2 masks")
    zoom_axes[0, 0].add_patch(
        Rectangle(
            (site.column + x_origin - 0.5, site.row + y_origin - 0.5),
            site.width,
            site.height,
            fill=False,
            color="cyan",
            linewidth=2.0,
            label="Fixed facility box",
        )
    )
    zoom_x = np.arange(zoom_x_min, zoom_x_max + 1) + x_origin
    zoom_y = np.arange(zoom_y_min, zoom_y_max + 1) + y_origin
    zoom_axes[0, 0].contour(
        zoom_x,
        zoom_y,
        historical_r2_extent[zoom_section].astype(float),
        levels=[0.5],
        colors=["red"],
        linewidths=1.8,
    )
    zoom_axes[0, 0].contour(
        zoom_x,
        zoom_y,
        current_weak_selected_r2[zoom_section].astype(float),
        levels=[0.5],
        colors=["yellow"],
        linewidths=1.8,
    )
    core_rows, core_columns = np.where(historical_r2_core)
    zoom_axes[0, 0].scatter(
        core_columns + x_origin,
        core_rows + y_origin,
        marker="s",
        s=19,
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
        label="Historical core",
    )
    facility_x = args.external_facility_x
    facility_y = args.external_facility_y
    zoom_axes[0, 0].scatter(
        [facility_x],
        [facility_y],
        marker="+",
        s=110,
        color="lime",
        linewidths=2.2,
        label="TCEQ general-location marker",
        zorder=8,
    )
    arrow_length = 12.0
    arrow_angle = math.radians(transport_bearing)
    arrow_dx = arrow_length * math.sin(arrow_angle)
    arrow_dy = -arrow_length * math.cos(arrow_angle)
    zoom_axes[0, 0].annotate(
        "",
        xy=(facility_x + arrow_dx, facility_y + arrow_dy),
        xytext=(facility_x, facility_y),
        arrowprops={"arrowstyle": "-|>", "color": "deepskyblue", "lw": 2.2},
        zorder=7,
    )
    zoom_axes[0, 0].plot(
        [],
        [],
        color="deepskyblue",
        label=f"MERRA-2 downwind {transport_bearing:.1f} deg",
    )
    zoom_axes[0, 0].plot([], [], color="red", label="Historical extent")
    zoom_axes[0, 0].plot([], [], color="yellow", label="Current weak selection")
    zoom_axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.75)

    zoom_panels = (
        (historical_joint, "Historical joint z", "viridis"),
        (
            stored["weak_alpha"].astype(float),
            "Current weak-band alpha (MODTRAN-eq ppm)",
            "magma",
        ),
        (
            stored["strong_alpha"].astype(float),
            "Current strong-band alpha (MODTRAN-eq ppm)",
            "magma",
        ),
        (stored["log_e_average"].astype(float), "Bidirectional cross-fit log e", "magma"),
        (stored["log_e_negative_control"].astype(float), "Reverse-CH4 control log e", "magma"),
    )
    for axis, (values, title, cmap) in zip(zoom_axes.ravel()[1:], zoom_panels):
        finite = values[zoom_section]
        finite = finite[np.isfinite(finite)]
        low, high = np.quantile(finite, [0.01, 0.995])
        shown = axis.imshow(
            values[zoom_section],
            extent=extent,
            cmap=cmap,
            vmin=low,
            vmax=max(high, low + 1e-9),
        )
        axis.add_patch(
            Rectangle(
                (site.column + x_origin - 0.5, site.row + y_origin - 0.5),
                site.width,
                site.height,
                fill=False,
                color="cyan",
                linewidth=1.4,
            )
        )
        axis.scatter(
            [facility_x],
            [facility_y],
            marker="+",
            s=65,
            color="lime",
            linewidths=1.7,
            zorder=8,
        )
        axis.contour(
            zoom_x,
            zoom_y,
            historical_r2_extent[zoom_section].astype(float),
            levels=[0.5],
            colors=["white"],
            linewidths=0.9,
        )
        axis.set_title(title)
        zoom_fig.colorbar(shown, ax=axis, shrink=0.78)
    for axis in zoom_axes[-1, :]:
        axis.set_xlabel("ROI x (pixel)")
    for axis in zoom_axes[:, 0]:
        axis.set_ylabel("ROI y (pixel)")
    zoom_fig.savefig(output_dir / "known_site_r2_zoom.png", dpi=190)
    plt.close(zoom_fig)

    summary: dict[str, object] = {
        "run_context": {
            "scene_id": args.scene_id,
            "observation_time_utc": args.observation_time_utc,
            "scene_csv": str(args.scene_csv.resolve()),
            "analysis_dir": str(analysis_dir),
            "historical_dir": str(historical_dir),
            "score_map_shape": [int(shape[0]), int(shape[1])],
            "score_map_origin_yx": [y_origin, x_origin],
            "axis_assumption": (
                "Image x increases east and image y increases south, as verified for this "
                "HISUI UTM GeoTIFF. External context defaults are acquisition-specific and "
                "must be overridden for another scene."
            ),
        },
        "site_definition": {
            "interpretation": (
                "Fixed rectangle equal to the historical R2 extent bounding box. It uses all "
                "pixels in the box rather than only high-score pixels, but its location was "
                "selected in prior analysis of this acquisition."
            ),
            "y_min": args.site_y_min,
            "y_max": args.site_y_max,
            "x_min": args.site_x_min,
            "x_max": args.site_x_max,
            "pixels": int(site.height * site.width),
            "independence_caveat": (
                "The coordinates were known from facility context and prior analysis of the "
                "same acquisition. This is targeted reanalysis, not independent-scene validation."
            ),
        },
        "controls": {
            "local_translated_boxes": len(local_controls),
            "minimum_center_radius_pixels": args.minimum_radius,
            "maximum_center_radius_pixels": args.maximum_radius,
            "site_exclusion_buffer_pixels": args.exclusion_buffer,
            "stride_pixels": args.control_stride,
            "rgb_matched_boxes": len(matched_controls),
            "nearest_rgb_robust_distance": float(np.min(matched_distances)),
            "median_selected_rgb_robust_distance": float(np.median(matched_distances)),
            "site_rgb_mean_and_std": site_rgb_features.tolist(),
            "rgb_balance_caveat": (
                "Large nearest-neighbour distance means the bright facility has no truly "
                "comparable RGB control in this ROI."
            ),
            "translation_caveat": (
                "Translated boxes overlap strongly and local terrain need not be exchangeable. "
                "The reported value is a positional tail fraction, not an inferential p-value "
                "of methane or an independent-sample test."
            ),
        },
        "alpha_site_summary": alpha_summary,
        "mask_reconciliation": mask_reconciliation,
        "fixed_mask_current_scores": fixed_mask_current_scores,
        "external_facility_context": {
            "name": "Keystone Gas Plant",
            "source_type": "TCEQ permit general-location marker",
            "latitude": args.facility_latitude,
            "longitude": args.facility_longitude,
            "roi_y": args.external_facility_y,
            "roi_x": args.external_facility_x,
            "distance_from_fixed_box_center_pixels": float(
                math.hypot(
                    args.external_facility_y - (site.center[0] + y_origin),
                    args.external_facility_x - (site.center[1] + x_origin),
                )
            ),
            "distance_from_fixed_box_center_metres": float(
                args.pixel_size_metres
                * math.hypot(
                    args.external_facility_y - (site.center[0] + y_origin),
                    args.external_facility_x - (site.center[1] + x_origin),
                )
            ),
            "caveat": (
                "The TCEQ marker is explicitly a general location, and nominal HISUI "
                "absolute geolocation error can be below 200 m. It is not an equipment-level "
                "source coordinate."
            ),
        },
        "wind_screen": {
            "source": "NASA POWER MERRA-2 hourly 10 m wind",
            "time_utc": "2022-10-30T16:00:00Z",
            "wind_from_degrees": args.wind_from_degrees,
            "wind_speed_metres_per_second": args.wind_speed_metres_per_second,
            "transport_bearing_degrees": transport_bearing,
            "inner_radius_pixels": args.wind_inner_radius,
            "outer_radius_pixels": args.wind_outer_radius,
            "cone_half_angle_degrees": args.wind_cone_half_angle,
            "interpretation": (
                "Exploratory morphology screen only. MERRA-2 is coarse reanalysis and the "
                "TCEQ point is not an exact stack or leak coordinate. Consistent downwind "
                "enhancement across metrics would support a plume; mixed directions do not."
            ),
        },
        "primary_endpoint": {
            "metric": "current_bidirectional_log_e",
            "statistic": "whole_box_mean",
            "site_score": float(primary_row.site_score),
            "local_positional_tail_fraction": float(
                primary_row.local_positional_tail_fraction
            ),
            "local_exceedances": int(primary_row.local_exceedances),
            "local_control_count": int(primary_row.local_control_count),
            "interpretation": (
                "A fixed-box post-hoc diagnostic ranked among local translations; not an "
                "inferential p-value or a calibrated emission probability."
            ),
        },
        "outputs": {
            "metric_table": str(output_dir / "site_metric_summary.csv"),
            "coherence_table": str(output_dir / "site_spatial_coherence.csv"),
            "local_controls": str(output_dir / "local_translated_controls.csv"),
            "rgb_matched_controls": str(output_dir / "rgb_matched_controls.csv"),
            "fixed_mask_scores": str(output_dir / "fixed_mask_score_summary.csv"),
            "wind_sector_screen": str(output_dir / "wind_sector_screen.csv"),
            "historical_region_comparison": str(
                output_dir / "historical_region_comparison.csv"
            ),
            "figure": str(output_dir / "known_site_r2_evidence.png"),
            "zoom_figure": str(output_dir / "known_site_r2_zoom.png"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Estimate one broad-stripe direction independently for each HISUI scene.

The implementation follows the broad-band score described on pages 15--16 of
the July 3 research slides.  For every candidate angle, pixels are grouped by
``b = y - tan(theta) * x``.  A median offset is estimated for each broad line
group and temporarily subtracted.  The score is the fractional reduction in
robust standard deviation.  A rolling-median angular trend is removed before
selecting a local peak.

The geometry is shared by the weak and strong methane windows.  Therefore the
two band-specific score curves are averaged and exactly one slope is selected
per scene; slopes are never optimized separately for the two bands.  A frozen,
sign-symmetric candidate mask can be excluded from both fitting and scoring so
that a compact methane candidate cannot determine the DWT direction.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from directional_destriping import robust_std


@dataclass(frozen=True)
class SceneSlopeSearchConfig:
    """Fixed, auditable settings for the scene-level broad-slope search."""

    angle_min_deg: float = 10.0
    angle_max_deg: float = 80.0
    angle_step_deg: float = 0.1
    search_negative_slopes: bool = True
    line_bin_width_pixels: float = 18.0
    minimum_pixels_per_line: int = 80
    sample_step: int = 4
    trend_window_deg: float = 8.0
    local_window_deg: float = 1.0
    edge_exclusion_deg: float = 1.0
    minimum_peak_robust_z: float = 5.0
    excluded_angles_deg: tuple[float, ...] = (
        math.degrees(math.atan(0.9773460526106752)),
    )
    excluded_half_width_deg: float = 1.5

    def validate(self) -> None:
        if not (0.0 < self.angle_min_deg < self.angle_max_deg < 90.0):
            raise ValueError("slope-search angles must satisfy 0 < min < max < 90")
        if self.angle_step_deg <= 0:
            raise ValueError("slope-search angle step must be positive")
        if self.line_bin_width_pixels <= 0:
            raise ValueError("slope-search line bin width must be positive")
        if self.minimum_pixels_per_line < 2:
            raise ValueError("minimum pixels per line must be at least two")
        if self.sample_step < 1:
            raise ValueError("slope-search sample step must be positive")
        if min(
            self.trend_window_deg,
            self.local_window_deg,
            self.edge_exclusion_deg,
        ) <= 0:
            raise ValueError("slope-search angular windows must be positive")
        if self.minimum_peak_robust_z <= 0:
            raise ValueError("minimum peak robust z must be positive")
        if self.excluded_half_width_deg < 0:
            raise ValueError("excluded angle half width must be non-negative")
        if any(not np.isfinite(value) for value in self.excluded_angles_deg):
            raise ValueError("excluded angles must be finite")

    def angles(self) -> np.ndarray:
        self.validate()
        values = np.arange(
            self.angle_min_deg,
            self.angle_max_deg + 0.5 * self.angle_step_deg,
            self.angle_step_deg,
            dtype=float,
        )
        positive = values[(values > 0.0) & (values < 90.0)]
        if not self.search_negative_slopes:
            return positive
        return np.concatenate((-positive[::-1], positive))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["excluded_angles_deg"] = list(self.excluded_angles_deg)
        return value


def _rolling_median(values: np.ndarray, window_points: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("rolling median input must be one-dimensional")
    window = max(3, int(window_points))
    if window % 2 == 0:
        window += 1
    half = window // 2
    output = np.empty_like(array)
    for index in range(array.size):
        selected = array[max(0, index - half) : min(array.size, index + half + 1)]
        finite = selected[np.isfinite(selected)]
        output[index] = float(np.median(finite)) if finite.size else np.nan
    return output


def _local_peak_mask(
    values: np.ndarray,
    *,
    local_half_points: int,
    edge_points: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.zeros(array.shape, dtype=bool)
    half = max(1, int(local_half_points))
    edge = max(half, int(edge_points))
    for index in range(edge, max(edge, array.size - edge)):
        if not np.isfinite(array[index]):
            continue
        lo = max(0, index - half)
        hi = min(array.size, index + half + 1)
        left = array[lo:index]
        right = array[index + 1 : hi]
        left_max = float(np.nanmax(left)) if np.any(np.isfinite(left)) else -np.inf
        right_max = float(np.nanmax(right)) if np.any(np.isfinite(right)) else -np.inf
        result[index] = array[index] > left_max and array[index] >= right_max
    return result


def _annotate_angular_curve(
    angles: np.ndarray,
    scores: np.ndarray,
    config: SceneSlopeSearchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detrend and find peaks independently on negative/positive branches."""
    angle_array = np.asarray(angles, dtype=float)
    score_array = np.asarray(scores, dtype=float)
    trend = np.full(score_array.shape, np.nan, dtype=float)
    local = np.zeros(score_array.shape, dtype=bool)
    step = float(config.angle_step_deg)
    trend_points = max(3, int(round(config.trend_window_deg / step)))
    local_points = max(1, int(round(config.local_window_deg / step)))
    edge_points = max(1, int(round(config.edge_exclusion_deg / step)))
    signs = (-1, 1) if config.search_negative_slopes else (1,)
    for sign in signs:
        selected = angle_array < 0.0 if sign < 0 else angle_array > 0.0
        indices = np.flatnonzero(selected)
        if indices.size == 0:
            continue
        branch_scores = score_array[indices]
        branch_trend = _rolling_median(branch_scores, trend_points)
        branch_prominence = branch_scores - branch_trend
        trend[indices] = branch_trend
        local[indices] = _local_peak_mask(
            branch_prominence,
            local_half_points=local_points,
            edge_points=edge_points,
        )
    return trend, score_array - trend, local


def _score_band_at_angle(
    sampled: np.ndarray,
    score_mask: np.ndarray,
    estimate_mask: np.ndarray,
    row_coordinates: np.ndarray,
    column_coordinates: np.ndarray,
    *,
    slope: float,
    line_bin_width_pixels: float,
    minimum_sampled_pixels_per_line: int,
    before_rstd: float,
) -> dict[str, float | int | None]:
    line_ids = np.rint(
        (row_coordinates - float(slope) * column_coordinates)
        / float(line_bin_width_pixels)
    ).astype(np.int32)
    valid_ids = line_ids[score_mask]
    if valid_ids.size == 0:
        return {
            "absolute_gain": None,
            "fractional_gain": None,
            "after_rstd": None,
            "line_count": 0,
            "estimated_line_count": 0,
        }
    minimum_id = int(valid_ids.min())
    maximum_id = int(valid_ids.max())
    line_count = maximum_id - minimum_id + 1
    normalized = line_ids - minimum_id
    labels = np.where(estimate_mask, normalized + 1, 0).astype(np.int32)
    indices = np.arange(1, line_count + 1, dtype=np.int32)
    counts = np.bincount(labels.ravel(), minlength=line_count + 1)[1:]
    medians = np.asarray(
        ndimage.median(sampled, labels=labels, index=indices), dtype=float
    )
    good = (
        (counts >= int(minimum_sampled_pixels_per_line))
        & np.isfinite(medians)
    )
    if not np.any(good):
        return {
            "absolute_gain": None,
            "fractional_gain": None,
            "after_rstd": None,
            "line_count": int(line_count),
            "estimated_line_count": 0,
        }
    center = float(np.median(medians[good]))
    offsets = np.zeros(line_count, dtype=float)
    offsets[good] = medians[good] - center
    selected_ids = normalized[score_mask]
    if np.any((selected_ids < 0) | (selected_ids >= line_count)):
        raise RuntimeError("valid slope-search line IDs fall outside the lookup")
    corrected = sampled[score_mask] - offsets[selected_ids]
    after_rstd = robust_std(corrected)
    if not np.isfinite(after_rstd) or not np.isfinite(before_rstd):
        absolute_gain = None
        fractional_gain = None
    else:
        absolute_gain = float(before_rstd - after_rstd)
        fractional_gain = (
            float(absolute_gain / before_rstd) if before_rstd > 0 else None
        )
    return {
        "absolute_gain": absolute_gain,
        "fractional_gain": fractional_gain,
        "after_rstd": float(after_rstd) if np.isfinite(after_rstd) else None,
        "line_count": int(line_count),
        "estimated_line_count": int(np.count_nonzero(good)),
    }


def _band_curve(
    image: np.ndarray,
    valid_mask: np.ndarray,
    protected_mask: np.ndarray,
    angles: np.ndarray,
    config: SceneSlopeSearchConfig,
) -> tuple[list[dict[str, float | int | None]], float, int]:
    step = int(config.sample_step)
    sampled = np.asarray(image, dtype=float)[::step, ::step]
    valid = np.asarray(valid_mask, dtype=bool)[::step, ::step]
    protected = np.asarray(protected_mask, dtype=bool)[::step, ::step]
    finite_valid = valid & np.isfinite(sampled)
    score_mask = finite_valid & ~protected
    estimate_mask = score_mask.copy()
    if not np.any(score_mask):
        raise ValueError("no unprotected valid pixels are available for slope search")
    before_rstd = robust_std(sampled[score_mask])
    if not np.isfinite(before_rstd) or before_rstd <= 0:
        raise ValueError("slope-search image has no finite robust dispersion")
    rows = np.arange(0, image.shape[0], step, dtype=float)[:, None]
    columns = np.arange(0, image.shape[1], step, dtype=float)[None, :]
    minimum_sampled = max(
        2, int(math.ceil(config.minimum_pixels_per_line / step))
    )
    curve: list[dict[str, float | int | None]] = []
    for angle in angles:
        slope = math.tan(math.radians(float(angle)))
        curve.append(
            _score_band_at_angle(
                sampled,
                score_mask,
                estimate_mask,
                rows,
                columns,
                slope=slope,
                line_bin_width_pixels=config.line_bin_width_pixels,
                minimum_sampled_pixels_per_line=minimum_sampled,
                before_rstd=before_rstd,
            )
        )
    return curve, float(before_rstd), int(np.count_nonzero(score_mask))


def _finite_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _axial_angle_difference_degrees(first: float, second: float) -> float:
    """Return the acute difference between two unoriented line axes."""
    return float(abs((float(first) - float(second) + 90.0) % 180.0 - 90.0))


def _best_peak_angle(
    angles: np.ndarray,
    scores: np.ndarray,
    allowed: np.ndarray,
    config: SceneSlopeSearchConfig,
) -> float | None:
    _trend, prominence, local = _annotate_angular_curve(
        angles, scores, config
    )
    eligible = allowed & local & np.isfinite(prominence)
    if not np.any(eligible):
        eligible = allowed & np.isfinite(prominence)
    if not np.any(eligible):
        return None
    indices = np.flatnonzero(eligible)
    return float(angles[indices[np.argmax(prominence[indices])]])


def estimate_scene_broad_slope(
    band_images: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
    protected_mask: np.ndarray,
    *,
    config: SceneSlopeSearchConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one joint broad slope and an auditable angle-score table.

    ``band_images`` should normally contain the uncorrected weak and strong MF
    score images.  The selected angle is shared across all supplied bands.
    """
    cfg = config or SceneSlopeSearchConfig()
    cfg.validate()
    if not band_images:
        raise ValueError("at least one band image is required")
    valid = np.asarray(valid_mask, dtype=bool)
    protected = np.asarray(protected_mask, dtype=bool)
    if valid.ndim != 2 or protected.shape != valid.shape:
        raise ValueError("valid and protected masks must share a 2-D shape")
    ordered_bands = list(band_images)
    arrays: dict[str, np.ndarray] = {}
    for name in ordered_bands:
        array = np.asarray(band_images[name], dtype=float)
        if array.shape != valid.shape:
            raise ValueError(f"{name}: slope-search image shape mismatch")
        if not np.all(np.isfinite(array[valid])):
            raise ValueError(f"{name}: non-finite slope-search score on valid pixels")
        arrays[name] = array

    angles = cfg.angles()
    band_curves: dict[str, list[dict[str, float | int | None]]] = {}
    band_before: dict[str, float] = {}
    band_sample_counts: dict[str, int] = {}
    for name in ordered_bands:
        curve, before_rstd, sample_count = _band_curve(
            arrays[name], valid, protected, angles, cfg
        )
        band_curves[name] = curve
        band_before[name] = before_rstd
        band_sample_counts[name] = sample_count

    rows: list[dict[str, Any]] = []
    joint_scores = np.full(angles.shape, np.nan, dtype=float)
    for index, angle in enumerate(angles):
        row: dict[str, Any] = {
            "angle_deg": float(angle),
            "slope_row_per_column": float(math.tan(math.radians(float(angle)))),
        }
        fractions: list[float | None] = []
        absolutes: list[float | None] = []
        for name in ordered_bands:
            band_row = band_curves[name][index]
            for key, value in band_row.items():
                row[f"{name}_{key}"] = value
            fractions.append(band_row["fractional_gain"])
            absolutes.append(band_row["absolute_gain"])
        row["joint_fractional_gain"] = _finite_mean(fractions)
        row["joint_absolute_gain"] = _finite_mean(absolutes)
        if row["joint_fractional_gain"] is not None:
            joint_scores[index] = float(row["joint_fractional_gain"])
        rows.append(row)

    trend, prominence, local = _annotate_angular_curve(
        angles, joint_scores, cfg
    )
    allowed = np.ones(angles.shape, dtype=bool)
    for excluded in cfg.excluded_angles_deg:
        allowed &= np.abs(angles - float(excluded)) > cfg.excluded_half_width_deg
    for index, row in enumerate(rows):
        row["joint_score_trend"] = (
            float(trend[index]) if np.isfinite(trend[index]) else None
        )
        row["joint_peak_prominence"] = (
            float(prominence[index]) if np.isfinite(prominence[index]) else None
        )
        row["is_local_peak"] = int(local[index])
        row["excluded_as_thin_direction"] = int(not allowed[index])

    eligible = allowed & local & np.isfinite(prominence) & (prominence > 0.0)
    selection_mode = "positive_local_peak"
    if not np.any(eligible):
        eligible = allowed & np.isfinite(prominence)
        selection_mode = "unsupported_maximum_prominence"
    if not np.any(eligible):
        raise ValueError("no finite broad-slope candidate is available")
    eligible_indices = np.flatnonzero(eligible)
    selected_index = int(
        eligible_indices[np.argmax(prominence[eligible_indices])]
    )
    rows[selected_index]["selected"] = 1
    for index, row in enumerate(rows):
        row.setdefault("selected", 0)

    peak_indices = np.flatnonzero(
        allowed & local & np.isfinite(prominence) & (prominence > 0.0)
    )
    other_peak_values = sorted(
        [float(prominence[index]) for index in peak_indices if index != selected_index],
        reverse=True,
    )
    per_band_best: dict[str, float | None] = {}
    for name in ordered_bands:
        band_scores = np.asarray(
            [
                np.nan
                if row["fractional_gain"] is None
                else float(row["fractional_gain"])
                for row in band_curves[name]
            ],
            dtype=float,
        )
        per_band_best[name] = _best_peak_angle(angles, band_scores, allowed, cfg)
    finite_band_angles = [
        float(value) for value in per_band_best.values() if value is not None
    ]
    selected_angle = float(angles[selected_index])
    selected_slope = float(math.tan(math.radians(selected_angle)))
    selected_band_gains = {
        name: band_curves[name][selected_index]["fractional_gain"]
        for name in ordered_bands
    }
    selected_band_line_counts = {
        name: band_curves[name][selected_index]["estimated_line_count"]
        for name in ordered_bands
    }
    prominence_values = prominence[allowed & np.isfinite(prominence)]
    prominence_scale = robust_std(prominence_values)
    selected_prominence = float(prominence[selected_index])
    peak_robust_z = (
        float(selected_prominence / prominence_scale)
        if np.isfinite(prominence_scale) and prominence_scale > 0.0
        else None
    )
    distance_to_search_edge = min(
        abs(abs(selected_angle) - cfg.angle_min_deg),
        abs(cfg.angle_max_deg - abs(selected_angle)),
    )
    selected_at_search_edge = bool(
        distance_to_search_edge < cfg.edge_exclusion_deg - 0.5 * cfg.angle_step_deg
    )
    support_failures: list[str] = []
    if selection_mode != "positive_local_peak":
        support_failures.append("no_positive_local_peak")
    selected_joint_gain = rows[selected_index]["joint_fractional_gain"]
    if selected_joint_gain is None or float(selected_joint_gain) <= 0.0:
        support_failures.append("nonpositive_joint_gain")
    if any(value is None or float(value) <= 0.0 for value in selected_band_gains.values()):
        support_failures.append("nonpositive_band_gain")
    if any(int(value or 0) < 1 for value in selected_band_line_counts.values()):
        support_failures.append("no_supported_lines")
    if selected_at_search_edge:
        support_failures.append("search_edge")
    if peak_robust_z is None or peak_robust_z < cfg.minimum_peak_robust_z:
        support_failures.append("peak_below_robust_z_gate")
    slope_status = "supported" if not support_failures else "unsupported"
    for index, row in enumerate(rows):
        row["slope_supported"] = int(
            index == selected_index and slope_status == "supported"
        )
    pairwise_band_differences = [
        _axial_angle_difference_degrees(first, second)
        for index, first in enumerate(finite_band_angles)
        for second in finite_band_angles[index + 1 :]
    ]
    diagnostics: dict[str, Any] = {
        "definition": (
            "scene-shared weak/strong mean fractional RStd reduction after "
            "temporary b=y-a*x line-median subtraction; rolling-median "
            "angular trend removed before local-peak selection"
        ),
        "selection_mode": selection_mode,
        "slope_status": slope_status,
        "slope_support_failures": support_failures,
        "selected_angle_deg": selected_angle,
        "selected_slope_row_per_column": selected_slope,
        "selected_joint_fractional_gain": rows[selected_index][
            "joint_fractional_gain"
        ],
        "selected_joint_absolute_gain": rows[selected_index][
            "joint_absolute_gain"
        ],
        "selected_peak_prominence": selected_prominence,
        "selected_peak_prominence_robust_scale": (
            float(prominence_scale) if np.isfinite(prominence_scale) else None
        ),
        "selected_peak_robust_z": peak_robust_z,
        "second_peak_prominence": other_peak_values[0] if other_peak_values else None,
        "selected_at_search_edge": selected_at_search_edge,
        "distance_to_search_edge_deg": float(distance_to_search_edge),
        "selected_band_fractional_gain": selected_band_gains,
        "selected_band_estimated_line_count": selected_band_line_counts,
        "per_band_best_angle_deg": per_band_best,
        "per_band_angle_spread_deg": (
            float(max(pairwise_band_differences))
            if pairwise_band_differences
            else None
        ),
        "per_band_before_rstd": band_before,
        "per_band_sample_count": band_sample_counts,
        "protected_sample_pixels_excluded": int(
            np.count_nonzero(valid[:: cfg.sample_step, :: cfg.sample_step]
                             & protected[:: cfg.sample_step, :: cfg.sample_step])
        ),
        "config": cfg.to_dict(),
    }
    return diagnostics, rows


def write_slope_search_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("slope-search CSV requires at least one row")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

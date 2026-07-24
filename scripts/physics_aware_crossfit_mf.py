#!/usr/bin/env python3
"""Pilot physics-aware, wavelength-cross-fitted methane detection for HISUI.

This script turns a MODTRAN concentration sweep into a HISUI-resolution unit
absorption spectrum (UAS), runs conventional matched filters, and constructs a
selection-safe held-out-band likelihood ratio.  The two physical band groups
are used in both directions:

* 2200--2390 nm estimates a non-negative methane amplitude and 1580--1750 nm
  validates the predicted held-out response.
* 1580--1750 nm estimates the amplitude and 2200--2390 nm validates it.

The two likelihood ratios are averaged as e-values.  With a known Gaussian
background model each direction has conditional null expectation one, even
though its amplitude is selected from the other band group.  In real scenes we
estimate nuisance parameters on held-out spatial blocks, so the reported
e-values are a plug-in pilot rather than a finite-sample guarantee.

Both small and multi-gigabyte ``y,x,wave_*`` CSV files are handled in chunks.
Raw imagery and generated outputs remain outside Git through the repository's
ignore rules.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata

from iterative_mf_1600nm import (
    compute_uas_log_slope,
    gaussian_srf_resample,
    get_wave_columns,
    load_ch4_lut,
)


DEFAULT_WEAK_WINDOW_NM = (1580.0, 1750.0)
DEFAULT_STRONG_WINDOW_NM = (2200.0, 2390.0)


@dataclass(frozen=True)
class DetectorModel:
    """Gaussian plug-in background model and a fixed log-radiance target."""

    mean: np.ndarray
    covariance: np.ndarray
    target: np.ndarray
    weak_indices: np.ndarray
    strong_indices: np.ndarray
    amplitude_cap: float


@dataclass(frozen=True)
class SceneScan:
    """First-pass metadata and a deterministic spatially distributed sample."""

    sample_radiance: np.ndarray
    sample_y: np.ndarray
    sample_x: np.ndarray
    valid_pixels: int
    total_rows: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int


@dataclass(frozen=True)
class NuisanceModelSet:
    """Global fallbacks and optional spatially local background models."""

    global_models: tuple[DetectorModel, ...]
    local_models: dict[tuple[int, int, int], DetectorModel]
    tile_size: int
    tile_halo: int

    @property
    def n_folds(self) -> int:
        return len(self.global_models)

    def get(self, y_tile: int, x_tile: int, fold: int) -> DetectorModel:
        return self.local_models.get(
            (int(y_tile), int(x_tile), int(fold)), self.global_models[int(fold)]
        )


def make_continuum_transform(
    wavelengths: np.ndarray,
    weak_indices: np.ndarray,
    strong_indices: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal feature transform that removes bandwise continua.

    A polynomial is fitted independently in each physical window.  The returned
    rows span the orthogonal complement of those polynomial nuisance terms, so
    the transformed covariance is full-dimensional rather than a singular
    residual covariance in the original band coordinates.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    groups = [np.asarray(weak_indices, dtype=int), np.asarray(strong_indices, dtype=int)]
    if degree < -1:
        raise ValueError("continuum degree must be -1 (disabled) or non-negative.")
    if degree == -1:
        transform = np.eye(len(wavelengths))
        return transform, groups[0].copy(), groups[1].copy()

    blocks: list[np.ndarray] = []
    feature_groups: list[np.ndarray] = []
    feature_start = 0
    for indices in groups:
        if len(indices) <= degree + 1:
            raise ValueError("Continuum polynomial leaves no target-sensitive features.")
        wave = wavelengths[indices]
        normalized_wave = (wave - wave.mean()) / max(float(wave.std()), 1e-12)
        design = np.column_stack(
            [normalized_wave**power for power in range(degree + 1)]
        )
        u, singular_values, _ = np.linalg.svd(design, full_matrices=True)
        tolerance = singular_values[0] * max(design.shape) * np.finfo(float).eps
        rank = int(np.sum(singular_values > tolerance))
        residual_basis = u[:, rank:]
        block = np.zeros((residual_basis.shape[1], len(wavelengths)), dtype=float)
        block[:, indices] = residual_basis.T
        blocks.append(block)
        feature_groups.append(
            np.arange(feature_start, feature_start + residual_basis.shape[1], dtype=int)
        )
        feature_start += residual_basis.shape[1]
    return np.vstack(blocks), feature_groups[0], feature_groups[1]


def regularized_covariance(
    values: np.ndarray,
    *,
    shrinkage: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a mean and positive-definite shrinkage covariance."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < values.shape[1] + 2:
        raise ValueError("Need a 2-D sample with more rows than spectral bands.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between zero and one.")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")

    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    covariance = covariance + ridge * scale * np.eye(covariance.shape[0])
    return mean, covariance


def fit_detector_model(
    log_radiance: np.ndarray,
    target: np.ndarray,
    weak_indices: np.ndarray,
    strong_indices: np.ndarray,
    *,
    shrinkage: float,
    ridge: float,
    amplitude_cap: float,
) -> DetectorModel:
    """Fit the nuisance background model used by all detector scores."""
    mean, covariance = regularized_covariance(
        log_radiance, shrinkage=shrinkage, ridge=ridge
    )
    return DetectorModel(
        mean=mean,
        covariance=covariance,
        target=np.asarray(target, dtype=float),
        weak_indices=np.asarray(weak_indices, dtype=int),
        strong_indices=np.asarray(strong_indices, dtype=int),
        amplitude_cap=float(amplitude_cap),
    )


def matched_filter_statistics(
    centered: np.ndarray,
    covariance: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-null-variance MF z-scores and target amplitudes."""
    inverse = np.linalg.pinv(covariance, rcond=1e-10)
    weight = inverse @ target
    information = float(target @ weight)
    if not np.isfinite(information) or information <= 1e-14:
        raise ValueError("Target has negligible energy in the background metric.")
    numerator = centered @ weight
    return numerator / math.sqrt(information), numerator / information


def _one_way_log_e(
    centered: np.ndarray,
    model: DetectorModel,
    selection_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> np.ndarray:
    """Compute a held-out likelihood-ratio e-value in log space."""
    covariance = model.covariance
    target = model.target
    sigma_aa = covariance[np.ix_(selection_indices, selection_indices)]
    sigma_ba = covariance[np.ix_(validation_indices, selection_indices)]
    sigma_ab = covariance[np.ix_(selection_indices, validation_indices)]
    sigma_bb = covariance[np.ix_(validation_indices, validation_indices)]
    inverse_aa = np.linalg.pinv(sigma_aa, rcond=1e-10)
    regression = sigma_ba @ inverse_aa
    conditional_covariance = sigma_bb - regression @ sigma_ab
    scale = float(np.trace(conditional_covariance) / len(validation_indices))
    conditional_covariance = conditional_covariance + max(scale, 1e-12) * 1e-10 * np.eye(
        len(validation_indices)
    )
    inverse_conditional = np.linalg.pinv(conditional_covariance, rcond=1e-10)

    target_a = target[selection_indices]
    target_b = target[validation_indices]
    centered_a = centered[:, selection_indices]
    centered_b = centered[:, validation_indices]

    selection_weight = inverse_aa @ target_a
    selection_information = float(target_a @ selection_weight)
    if selection_information <= 1e-14:
        raise ValueError("Selection-band target has negligible background-metric energy.")
    amplitude = centered_a @ selection_weight / selection_information
    amplitude = np.clip(amplitude, 0.0, model.amplitude_cap)

    innovation = centered_b - centered_a @ regression.T
    target_innovation = target_b - regression @ target_a
    validation_weight = inverse_conditional @ target_innovation
    validation_information = float(target_innovation @ validation_weight)
    evidence = innovation @ validation_weight
    return amplitude * evidence - 0.5 * amplitude**2 * validation_information


def score_log_radiance(
    log_radiance: np.ndarray,
    model: DetectorModel,
) -> dict[str, np.ndarray]:
    """Score spectra with MF baselines and bidirectional band cross-fitting."""
    centered = np.asarray(log_radiance, dtype=float) - model.mean
    combined_z, combined_alpha = matched_filter_statistics(
        centered, model.covariance, model.target
    )

    weak = model.weak_indices
    strong = model.strong_indices
    weak_z, weak_alpha = matched_filter_statistics(
        centered[:, weak],
        model.covariance[np.ix_(weak, weak)],
        model.target[weak],
    )
    strong_z, strong_alpha = matched_filter_statistics(
        centered[:, strong],
        model.covariance[np.ix_(strong, strong)],
        model.target[strong],
    )

    log_e_strong_to_weak = _one_way_log_e(centered, model, strong, weak)
    log_e_weak_to_strong = _one_way_log_e(centered, model, weak, strong)
    log_e_average = (
        np.logaddexp(log_e_strong_to_weak, log_e_weak_to_strong) - math.log(2.0)
    )
    negative_control_model = DetectorModel(
        mean=model.mean,
        covariance=model.covariance,
        target=-model.target,
        weak_indices=model.weak_indices,
        strong_indices=model.strong_indices,
        amplitude_cap=model.amplitude_cap,
    )
    negative_strong_to_weak = _one_way_log_e(
        centered, negative_control_model, strong, weak
    )
    negative_weak_to_strong = _one_way_log_e(
        centered, negative_control_model, weak, strong
    )
    log_e_negative_control = (
        np.logaddexp(negative_strong_to_weak, negative_weak_to_strong)
        - math.log(2.0)
    )
    return {
        "combined_z": combined_z,
        "combined_alpha": combined_alpha,
        "weak_z": weak_z,
        "weak_alpha": weak_alpha,
        "strong_z": strong_z,
        "strong_alpha": strong_alpha,
        "dual_min_z": np.minimum(weak_z, strong_z),
        "log_e_strong_to_weak": log_e_strong_to_weak,
        "log_e_weak_to_strong": log_e_weak_to_strong,
        "log_e_average": log_e_average,
        "log_e_negative_control": log_e_negative_control,
    }


def e_bh_mask(log_e_values: np.ndarray, fdr: float) -> tuple[np.ndarray, float, int]:
    """Apply e-BH without exponentiating potentially very large e-values."""
    values = np.asarray(log_e_values, dtype=float)
    valid = np.isfinite(values)
    if not 0.0 < fdr < 1.0:
        raise ValueError("fdr must be between zero and one.")
    output = np.zeros(values.shape, dtype=bool)
    finite_values = values[valid]
    m = len(finite_values)
    if m == 0:
        return output, float("inf"), 0
    ordered = np.sort(finite_values)[::-1]
    ranks = np.arange(1, m + 1, dtype=float)
    qualifies = ordered >= np.log(m / (fdr * ranks))
    if not np.any(qualifies):
        return output, float("inf"), 0
    selected_rank = int(np.flatnonzero(qualifies)[-1] + 1)
    threshold = float(ordered[selected_rank - 1])
    output[valid] = finite_values >= threshold
    return output, threshold, int(output.sum())


def signed_null_calibration(
    positive_log_e: np.ndarray,
    reverse_log_e: np.ndarray,
    *,
    log_e_cap: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Clip and normalize target/reverse scores as a sign-paired diagnostic.

    The pooled empirical mean of the returned e-values is exactly one.  This is
    useful for detecting severe plug-in tail inflation, but it is only a valid
    null calibration when target and reverse signs are exchangeable; the output
    is therefore deliberately reported as a diagnostic rather than a theorem.
    """
    positive = np.asarray(positive_log_e, dtype=float)
    reverse = np.asarray(reverse_log_e, dtype=float)
    if positive.shape != reverse.shape:
        raise ValueError("positive and reverse log-e arrays must have the same shape.")
    if not np.isfinite(log_e_cap) or log_e_cap <= 0.0:
        raise ValueError("log_e_cap must be a positive finite number.")
    valid = np.isfinite(positive) & np.isfinite(reverse)
    calibrated_positive = np.full(positive.shape, np.nan, dtype=float)
    calibrated_reverse = np.full(reverse.shape, np.nan, dtype=float)
    if not np.any(valid):
        return calibrated_positive, calibrated_reverse, float("nan")
    clipped_positive = np.minimum(positive[valid], log_e_cap)
    clipped_reverse = np.minimum(reverse[valid], log_e_cap)
    pooled = np.concatenate([clipped_positive, clipped_reverse])
    log_normalizer = float(logsumexp(pooled) - math.log(len(pooled)))
    calibrated_positive[valid] = clipped_positive - log_normalizer
    calibrated_reverse[valid] = clipped_reverse - log_normalizer
    return calibrated_positive, calibrated_reverse, log_normalizer


def spatial_fold_ids(
    y: np.ndarray,
    x: np.ndarray,
    *,
    n_folds: int,
    block_size: int,
) -> np.ndarray:
    """Assign nearby pixels to the same nuisance-estimation fold."""
    if n_folds <= 1:
        return np.zeros(len(y), dtype=int)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    y_block = np.floor_divide(np.asarray(y, dtype=np.int64), block_size)
    x_block = np.floor_divide(np.asarray(x, dtype=np.int64), block_size)
    return np.mod(y_block + 3 * x_block, n_folds).astype(int)


def _coordinate_hash(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Deterministic splitmix64 hash for spatial reservoir sampling."""
    value = (
        np.asarray(y, dtype=np.uint64) * np.uint64(0x9E3779B185EBCA87)
        + np.asarray(x, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        + np.uint64(0x165667B19E3779F9)
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


class HashReservoir:
    """Keep the globally smallest coordinate hashes using bounded memory."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("sample_size must be positive.")
        self.capacity = int(capacity)
        self.hashes = np.empty(0, dtype=np.uint64)
        self.values = np.empty((0, 0), dtype=float)
        self.y = np.empty(0, dtype=np.int64)
        self.x = np.empty(0, dtype=np.int64)

    def update(self, values: np.ndarray, y: np.ndarray, x: np.ndarray) -> None:
        if len(values) == 0:
            return
        hashes = _coordinate_hash(y, x)
        if self.values.size == 0:
            combined_values = np.asarray(values, dtype=float)
        else:
            combined_values = np.vstack([self.values, values])
        combined_hashes = np.concatenate([self.hashes, hashes])
        combined_y = np.concatenate([self.y, np.asarray(y, dtype=np.int64)])
        combined_x = np.concatenate([self.x, np.asarray(x, dtype=np.int64)])
        if len(combined_hashes) > self.capacity:
            keep = np.argpartition(combined_hashes, self.capacity - 1)[: self.capacity]
            combined_hashes = combined_hashes[keep]
            combined_values = combined_values[keep]
            combined_y = combined_y[keep]
            combined_x = combined_x[keep]
        self.hashes = combined_hashes
        self.values = combined_values
        self.y = combined_y
        self.x = combined_x


def iter_scene_chunks(
    path: Path,
    wave_columns: Sequence[str],
    *,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """Read coordinates and selected spectra while preserving column order."""
    usecols = ["y", "x", *wave_columns]
    yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize)


def scan_scene(
    path: Path,
    wave_columns: Sequence[str],
    *,
    chunksize: int,
    sample_size: int,
) -> SceneScan:
    """First pass: validate the grid and collect a bounded background sample."""
    reservoir = HashReservoir(sample_size)
    total_rows = 0
    valid_pixels = 0
    y_min = x_min = np.iinfo(np.int64).max
    y_max = x_max = np.iinfo(np.int64).min

    for chunk in iter_scene_chunks(path, wave_columns, chunksize=chunksize):
        y = chunk["y"].to_numpy(dtype=np.int64)
        x = chunk["x"].to_numpy(dtype=np.int64)
        values = chunk[list(wave_columns)].to_numpy(dtype=float)
        total_rows += len(chunk)
        if len(chunk):
            y_min = min(y_min, int(y.min()))
            y_max = max(y_max, int(y.max()))
            x_min = min(x_min, int(x.min()))
            x_max = max(x_max, int(x.max()))
        valid = np.all(np.isfinite(values), axis=1) & np.all(values > 0.0, axis=1)
        valid_pixels += int(valid.sum())
        reservoir.update(values[valid], y[valid], x[valid])

    if total_rows == 0 or valid_pixels == 0:
        raise ValueError("The scene contains no positive, finite spectra in both windows.")
    return SceneScan(
        sample_radiance=reservoir.values,
        sample_y=reservoir.y,
        sample_x=reservoir.x,
        valid_pixels=valid_pixels,
        total_rows=total_rows,
        y_min=int(y_min),
        y_max=int(y_max),
        x_min=int(x_min),
        x_max=int(x_max),
    )


def build_nuisance_models(
    scan: SceneScan,
    feature_transform: np.ndarray,
    target: np.ndarray,
    weak_indices: np.ndarray,
    strong_indices: np.ndarray,
    *,
    n_folds: int,
    block_size: int,
    shrinkage: float,
    ridge: float,
    amplitude_cap: float,
    local_tile_size: int,
    local_tile_halo: int,
) -> NuisanceModelSet:
    """Fit held-out-fold global models and optional local neighborhood models."""
    log_sample = np.log(scan.sample_radiance) @ feature_transform.T
    folds = spatial_fold_ids(
        scan.sample_y, scan.sample_x, n_folds=n_folds, block_size=block_size
    )
    models: list[DetectorModel] = []
    for fold in range(max(n_folds, 1)):
        training = np.ones(len(log_sample), dtype=bool) if n_folds <= 1 else folds != fold
        if int(training.sum()) < len(target) + 20:
            raise ValueError(
                f"Nuisance fold {fold} has only {int(training.sum())} training pixels; "
                "increase --sample-size or reduce --nuisance-folds."
            )
        models.append(
            fit_detector_model(
                log_sample[training],
                target,
                weak_indices,
                strong_indices,
                shrinkage=shrinkage,
                ridge=ridge,
                amplitude_cap=amplitude_cap,
            )
        )
    if local_tile_size < 0:
        raise ValueError("local_tile_size must be zero (disabled) or positive.")
    if local_tile_halo < 0:
        raise ValueError("local_tile_halo must be non-negative.")
    local_models: dict[tuple[int, int, int], DetectorModel] = {}
    if local_tile_size > 0:
        y_tiles = np.floor_divide(scan.sample_y, local_tile_size)
        x_tiles = np.floor_divide(scan.sample_x, local_tile_size)
        occupied_tiles = np.unique(np.column_stack([y_tiles, x_tiles]), axis=0)
        minimum_training = len(target) + 20
        for y_tile, x_tile in occupied_tiles:
            local = (
                (y_tiles >= y_tile - local_tile_halo)
                & (y_tiles <= y_tile + local_tile_halo)
                & (x_tiles >= x_tile - local_tile_halo)
                & (x_tiles <= x_tile + local_tile_halo)
            )
            for fold in range(max(n_folds, 1)):
                training = local if n_folds <= 1 else local & (folds != fold)
                if int(training.sum()) < minimum_training:
                    continue
                local_models[(int(y_tile), int(x_tile), int(fold))] = fit_detector_model(
                    log_sample[training],
                    target,
                    weak_indices,
                    strong_indices,
                    shrinkage=shrinkage,
                    ridge=ridge,
                    amplitude_cap=amplitude_cap,
                )
    return NuisanceModelSet(
        global_models=tuple(models),
        local_models=local_models,
        tile_size=int(local_tile_size),
        tile_halo=int(local_tile_halo),
    )


def score_scene(
    path: Path,
    wave_columns: Sequence[str],
    scan: SceneScan,
    models: NuisanceModelSet,
    feature_transform: np.ndarray,
    *,
    chunksize: int,
    block_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Second pass: score every valid pixel into rectangular map arrays."""
    height = scan.y_max - scan.y_min + 1
    width = scan.x_max - scan.x_min + 1
    if height * width > 100_000_000:
        raise ValueError(
            f"Coordinate bounding box {height}x{width} is unexpectedly large."
        )
    keys = [
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
    ]
    maps = {key: np.full((height, width), np.nan, dtype=np.float32) for key in keys}
    valid_map = np.zeros((height, width), dtype=bool)

    for chunk in iter_scene_chunks(path, wave_columns, chunksize=chunksize):
        y = chunk["y"].to_numpy(dtype=np.int64)
        x = chunk["x"].to_numpy(dtype=np.int64)
        values = chunk[list(wave_columns)].to_numpy(dtype=float)
        valid = np.all(np.isfinite(values), axis=1) & np.all(values > 0.0, axis=1)
        if not np.any(valid):
            continue
        yv = y[valid]
        xv = x[valid]
        log_values = np.log(values[valid]) @ feature_transform.T
        folds = spatial_fold_ids(
            yv, xv, n_folds=models.n_folds, block_size=block_size
        )
        row = yv - scan.y_min
        column = xv - scan.x_min
        if np.any(valid_map[row, column]):
            raise ValueError("The scene contains duplicate valid y,x coordinates.")
        valid_map[row, column] = True
        if models.tile_size > 0:
            y_tiles = np.floor_divide(yv, models.tile_size)
            x_tiles = np.floor_divide(xv, models.tile_size)
        else:
            y_tiles = np.zeros(len(yv), dtype=int)
            x_tiles = np.zeros(len(xv), dtype=int)
        groups = np.unique(np.column_stack([y_tiles, x_tiles, folds]), axis=0)
        for y_tile, x_tile, fold in groups:
            selected = (folds == fold) & (y_tiles == y_tile) & (x_tiles == x_tile)
            model = models.get(int(y_tile), int(x_tile), int(fold))
            scores = score_log_radiance(log_values[selected], model)
            rr = row[selected]
            cc = column[selected]
            for key in keys:
                maps[key][rr, cc] = scores[key].astype(np.float32)
    return maps, valid_map


def template_family_diagnostics(
    alpha_grid: np.ndarray,
    resampled_radiance: np.ndarray,
    unit_target: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int | list[float]]]:
    """Quantify whether concentration changes create shape or only amplitude."""
    log_shift = np.log(resampled_radiance / resampled_radiance[[0]])
    rows: list[dict[str, float]] = []
    normalized: list[np.ndarray] = []
    previous: np.ndarray | None = None
    target_norm = float(np.linalg.norm(unit_target))
    for alpha, shift in zip(alpha_grid, log_shift):
        norm = float(np.linalg.norm(shift))
        if norm > 0.0:
            current = shift / norm
            normalized.append(current)
            corr_target = float(current @ (unit_target / target_norm))
            corr_previous = float(current @ previous) if previous is not None else float("nan")
            previous = current
        else:
            corr_target = float("nan")
            corr_previous = float("nan")
        rows.append(
            {
                "concentration_parameter": float(alpha),
                "log_shift_l2": norm,
                "cosine_with_unit_target": corr_target,
                "cosine_with_previous_nonzero_template": corr_previous,
            }
        )
    normalized_array = np.vstack(normalized)
    singular_values = np.linalg.svd(normalized_array, compute_uv=False)
    explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    summary: dict[str, float | int | list[float]] = {
        "rank_99_percent": int(np.searchsorted(explained, 0.99) + 1),
        "rank_999_percent": int(np.searchsorted(explained, 0.999) + 1),
        "first_singular_values": [float(value) for value in singular_values[:10]],
        "first_component_energy_fraction": float(explained[0]),
        "minimum_adjacent_nonzero_cosine": float(
            np.nanmin(pd.DataFrame(rows)["cosine_with_previous_nonzero_template"])
        ),
    }
    return pd.DataFrame(rows), summary


def _roc_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    combined = np.concatenate([negative, positive])
    ranks = rankdata(combined, method="average")
    n0 = len(negative)
    n1 = len(positive)
    rank_sum_positive = float(ranks[n0:].sum())
    return (rank_sum_positive - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def _average_precision(negative: np.ndarray, positive: np.ndarray) -> float:
    scores = np.concatenate([negative, positive])
    labels = np.concatenate(
        [np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)]
    )
    order = np.argsort(scores, kind="mergesort")[::-1]
    labels = labels[order]
    true_positives = np.cumsum(labels)
    precision = true_positives / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * labels) / max(labels.sum(), 1))


def interpolate_log_shift(
    concentration: float,
    alpha_grid: np.ndarray,
    resampled_radiance: np.ndarray,
) -> np.ndarray:
    """Interpolate the physical log-radiance shift at one LUT concentration."""
    if not alpha_grid[0] <= concentration <= alpha_grid[-1]:
        raise ValueError(
            f"Injection concentration {concentration} is outside the LUT range "
            f"{alpha_grid[0]}--{alpha_grid[-1]}."
        )
    log_radiance = np.log(resampled_radiance)
    selected = np.asarray(
        [np.interp(concentration, alpha_grid, log_radiance[:, band]) for band in range(log_radiance.shape[1])]
    )
    return selected - log_radiance[0]


def injection_benchmark(
    sample_radiance: np.ndarray,
    model: DetectorModel,
    alpha_grid: np.ndarray,
    resampled_radiance: np.ndarray,
    feature_transform: np.ndarray,
    concentrations: Sequence[float],
    *,
    max_pixels: int,
    seed: int,
) -> pd.DataFrame:
    """Inject exact MODTRAN ratios into real spectra and measure detection power."""
    rng = np.random.default_rng(seed)
    if len(sample_radiance) > max_pixels:
        selected = rng.choice(len(sample_radiance), size=max_pixels, replace=False)
        sample_radiance = sample_radiance[selected]
    original = np.log(sample_radiance) @ feature_transform.T
    negative_scores = score_log_radiance(original, model)
    method_names = ["strong_z", "combined_z", "weak_z", "dual_min_z", "log_e_average"]
    rows: list[dict[str, float | str]] = []
    for concentration in concentrations:
        shift = feature_transform @ interpolate_log_shift(
            concentration, alpha_grid, resampled_radiance
        )
        positive_scores = score_log_radiance(original + shift, model)
        for method in method_names:
            negative = np.asarray(negative_scores[method], dtype=float)
            positive = np.asarray(positive_scores[method], dtype=float)
            row: dict[str, float | str] = {
                "concentration_parameter": float(concentration),
                "method": method,
                "roc_auc": float(_roc_auc(negative, positive)),
                "average_precision_balanced": float(_average_precision(negative, positive)),
            }
            for fpr in (0.01, 0.001):
                threshold = float(np.quantile(negative, 1.0 - fpr, method="higher"))
                label = "1pct" if fpr == 0.01 else "0p1pct"
                row[f"threshold_at_{label}_fpr"] = threshold
                row[f"empirical_fpr_{label}"] = float(np.mean(negative >= threshold))
                row[f"tpr_at_{label}_fpr"] = float(np.mean(positive >= threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def null_diagnostics(log_e_values: np.ndarray) -> dict[str, float | int | None | list[float]]:
    """Summarize empirical behavior expected from a null e-value sample."""
    values = np.asarray(log_e_values, dtype=float)
    values = values[np.isfinite(values)]
    log_mean = float(logsumexp(values) - math.log(len(values)))
    return {
        "n": int(len(values)),
        "log_mean_e": log_mean,
        "mean_e": float(math.exp(log_mean)) if log_mean < 700 else None,
        "log_e_quantiles_50_90_95_99_99p9_max": [
            float(value)
            for value in np.quantile(values, [0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
        ],
        "fraction_e_ge_10": float(np.mean(values >= math.log(10.0))),
        "fraction_e_ge_20": float(np.mean(values >= math.log(20.0))),
        "fraction_e_ge_100": float(np.mean(values >= math.log(100.0))),
    }


def candidate_table(
    maps: dict[str, np.ndarray],
    valid_map: np.ndarray,
    discoveries: np.ndarray,
    scan: SceneScan,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Return the strongest candidate pixels without exporting raw spectra."""
    row, column = np.nonzero(valid_map)
    log_e = maps["log_e_average"][valid_map].astype(float)
    combined_z = maps["combined_z"][valid_map].astype(float)
    n_each = min(top_n, len(row))
    top_e = np.argpartition(log_e, -n_each)[-n_each:]
    top_mf = np.argpartition(combined_z, -n_each)[-n_each:]
    selected = np.unique(np.concatenate([top_e, top_mf]))
    output = pd.DataFrame(
        {
            "y": row[selected] + scan.y_min,
            "x": column[selected] + scan.x_min,
            "is_e_bh_discovery": discoveries[row[selected], column[selected]],
        }
    )
    for key, image in maps.items():
        output[key] = image[row[selected], column[selected]].astype(float)
    denominator = np.maximum(
        np.abs(output["weak_alpha"]) + np.abs(output["strong_alpha"]), 1e-12
    )
    output["relative_alpha_disagreement"] = (
        np.abs(output["weak_alpha"] - output["strong_alpha"]) / denominator
    )
    return output.sort_values(
        ["is_e_bh_discovery", "log_e_average", "combined_z"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_overview(
    path: Path,
    maps: dict[str, np.ndarray],
    discoveries: np.ndarray,
) -> None:
    """Save compact diagnostic maps for visual artifact inspection."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    panels = [
        ("strong_z", "Strong-band MF z", "viridis"),
        ("weak_z", "Weak-band MF z", "viridis"),
        ("combined_z", "Combined-band MF z", "viridis"),
        ("dual_min_z", "min(weak z, strong z)", "viridis"),
        ("log_e_average", "Bidirectional cross-fit log e", "magma"),
    ]
    for axis, (key, title, cmap) in zip(axes.ravel()[:5], panels):
        image = maps[key].astype(float)
        finite = image[np.isfinite(image)]
        lower, upper = np.quantile(finite, [0.01, 0.995])
        shown = axis.imshow(image, cmap=cmap, vmin=float(lower), vmax=float(upper))
        if discoveries.any():
            axis.contour(discoveries, levels=[0.5], colors="cyan", linewidths=0.7)
        axis.set_title(title)
        axis.set_axis_off()
        fig.colorbar(shown, ax=axis, shrink=0.75)
    negative = maps["log_e_negative_control"].astype(float)
    finite_negative = negative[np.isfinite(negative)]
    lower, upper = np.quantile(finite_negative, [0.01, 0.995])
    shown = axes.ravel()[5].imshow(
        negative, cmap="magma", vmin=float(lower), vmax=float(upper)
    )
    if discoveries.any():
        axes.ravel()[5].contour(
            discoveries, levels=[0.5], colors="cyan", linewidths=0.7
        )
    axes.ravel()[5].set_title("Reverse-CH4 negative-control log e")
    axes.ravel()[5].set_axis_off()
    fig.colorbar(shown, ax=axes.ravel()[5], shrink=0.75)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_concentrations(value: str) -> list[float]:
    output = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise argparse.ArgumentTypeError("Provide at least one concentration.")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-csv", type=Path, required=True)
    parser.add_argument("--modtran-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weak-min-nm", type=float, default=DEFAULT_WEAK_WINDOW_NM[0])
    parser.add_argument("--weak-max-nm", type=float, default=DEFAULT_WEAK_WINDOW_NM[1])
    parser.add_argument("--strong-min-nm", type=float, default=DEFAULT_STRONG_WINDOW_NM[0])
    parser.add_argument("--strong-max-nm", type=float, default=DEFAULT_STRONG_WINDOW_NM[1])
    parser.add_argument("--fwhm-nm", type=float, default=12.5)
    parser.add_argument("--uas-alpha-min", type=float, default=0.0)
    parser.add_argument("--uas-alpha-max", type=float, default=0.5)
    parser.add_argument(
        "--continuum-degree",
        type=int,
        default=1,
        help="Polynomial degree removed independently in each band group; -1 disables.",
    )
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--spatial-block-size", type=int, default=10)
    parser.add_argument(
        "--local-tile-size",
        type=int,
        default=0,
        help=(
            "Fit a separate background model for each spatial tile using nearby "
            "sample tiles; zero keeps one scene-wide model per fold."
        ),
    )
    parser.add_argument(
        "--local-tile-halo",
        type=int,
        default=1,
        help="Number of neighboring sample tiles used around each local scoring tile.",
    )
    parser.add_argument("--covariance-shrinkage", type=float, default=0.05)
    parser.add_argument("--covariance-ridge", type=float, default=1e-6)
    parser.add_argument("--amplitude-cap", type=float, default=None)
    parser.add_argument("--fdr", type=float, default=0.10)
    parser.add_argument(
        "--signed-null-log-e-cap",
        type=float,
        default=20.0,
        help=(
            "Clip level for the target/reverse sign-paired tail diagnostic; "
            "this calibration is heuristic unless sign exchangeability holds."
        ),
    )
    parser.add_argument(
        "--injection-concentrations",
        type=parse_concentrations,
        default=parse_concentrations("0.1,0.2,0.5,1,2,5"),
    )
    parser.add_argument("--injection-sample-size", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=743)
    parser.add_argument("--top-candidates", type=int, default=250)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    scene_path = args.scene_csv.expanduser().resolve()
    modtran_path = args.modtran_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(scene_path, nrows=0)
    if not {"y", "x"}.issubset(header.columns):
        raise ValueError("Scene CSV must contain y and x columns.")
    all_wave_columns, all_wavelengths = get_wave_columns(header.columns)
    weak_mask_all = (all_wavelengths >= args.weak_min_nm) & (
        all_wavelengths <= args.weak_max_nm
    )
    strong_mask_all = (all_wavelengths >= args.strong_min_nm) & (
        all_wavelengths <= args.strong_max_nm
    )
    selected_mask = weak_mask_all | strong_mask_all
    if weak_mask_all.sum() < 3 or strong_mask_all.sum() < 3:
        raise ValueError("Each physical band window must contain at least three HISUI bands.")
    wave_columns = [column for column, keep in zip(all_wave_columns, selected_mask) if keep]
    wavelengths = all_wavelengths[selected_mask]
    weak_indices = np.flatnonzero(
        (wavelengths >= args.weak_min_nm) & (wavelengths <= args.weak_max_nm)
    )
    strong_indices = np.flatnonzero(
        (wavelengths >= args.strong_min_nm) & (wavelengths <= args.strong_max_nm)
    )

    mod_wave, alpha_grid, mod_radiance = load_ch4_lut(modtran_path)
    resampled = gaussian_srf_resample(
        mod_wave, mod_radiance, wavelengths, args.fwhm_nm
    )
    uas = compute_uas_log_slope(
        alpha_grid,
        resampled,
        alpha_min=args.uas_alpha_min,
        alpha_max=args.uas_alpha_max,
    )
    raw_target = -uas
    feature_transform, weak_feature_indices, strong_feature_indices = make_continuum_transform(
        wavelengths,
        weak_indices,
        strong_indices,
        degree=args.continuum_degree,
    )
    target = feature_transform @ raw_target
    amplitude_cap = (
        float(alpha_grid[-1]) if args.amplitude_cap is None else float(args.amplitude_cap)
    )

    scan = scan_scene(
        scene_path,
        wave_columns,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
    )
    models = build_nuisance_models(
        scan,
        feature_transform,
        target,
        weak_feature_indices,
        strong_feature_indices,
        n_folds=args.nuisance_folds,
        block_size=args.spatial_block_size,
        shrinkage=args.covariance_shrinkage,
        ridge=args.covariance_ridge,
        amplitude_cap=amplitude_cap,
        local_tile_size=args.local_tile_size,
        local_tile_halo=args.local_tile_halo,
    )
    maps, valid_map = score_scene(
        scene_path,
        wave_columns,
        scan,
        models,
        feature_transform,
        chunksize=args.chunksize,
        block_size=args.spatial_block_size,
    )
    discoveries, log_e_threshold, discovery_count = e_bh_mask(
        maps["log_e_average"], args.fdr
    )
    discoveries &= valid_map
    negative_discoveries, negative_log_e_threshold, negative_discovery_count = e_bh_mask(
        maps["log_e_negative_control"], args.fdr
    )
    negative_discoveries &= valid_map
    calibrated_log_e, calibrated_reverse_log_e, signed_log_normalizer = (
        signed_null_calibration(
            maps["log_e_average"],
            maps["log_e_negative_control"],
            log_e_cap=args.signed_null_log_e_cap,
        )
    )
    calibrated_discoveries, calibrated_threshold, calibrated_count = e_bh_mask(
        calibrated_log_e, args.fdr
    )
    calibrated_discoveries &= valid_map
    calibrated_reverse_discoveries, _, calibrated_reverse_count = e_bh_mask(
        calibrated_reverse_log_e, args.fdr
    )
    calibrated_reverse_discoveries &= valid_map

    family_table, family_summary = template_family_diagnostics(
        alpha_grid, resampled, raw_target
    )
    family_table.to_csv(output_dir / "template_family_diagnostics.csv", index=False)
    target_table = pd.DataFrame(
        {
            "wavelength_nm": wavelengths,
            "band_group": np.where(
                np.isin(np.arange(len(wavelengths)), weak_indices), "weak", "strong"
            ),
            "uas_per_concentration": uas,
            "log_radiance_target_per_concentration": raw_target,
            "modtran_baseline_radiance_raw": resampled[0],
            "modtran_baseline_radiance_times_100": resampled[0] * 100.0,
            "scene_sample_median_radiance": np.median(scan.sample_radiance, axis=0),
        }
    )
    target_table["scene_to_modtran_times_100_ratio"] = (
        target_table["scene_sample_median_radiance"]
        / target_table["modtran_baseline_radiance_times_100"]
    )
    target_table.to_csv(output_dir / "hisui_target_spectrum.csv", index=False)

    global_model = fit_detector_model(
        np.log(scan.sample_radiance) @ feature_transform.T,
        target,
        weak_feature_indices,
        strong_feature_indices,
        shrinkage=args.covariance_shrinkage,
        ridge=args.covariance_ridge,
        amplitude_cap=amplitude_cap,
    )
    benchmark = injection_benchmark(
        scan.sample_radiance,
        global_model,
        alpha_grid,
        resampled,
        feature_transform,
        args.injection_concentrations,
        max_pixels=args.injection_sample_size,
        seed=args.seed,
    )
    benchmark.to_csv(output_dir / "injection_benchmark.csv", index=False)

    candidates = candidate_table(
        maps,
        valid_map,
        discoveries,
        scan,
        top_n=args.top_candidates,
    )
    candidates.to_csv(output_dir / "candidate_pixels.csv", index=False)
    np.savez_compressed(
        output_dir / "score_maps.npz",
        valid=valid_map,
        e_bh_discovery=discoveries,
        reverse_ch4_e_bh_discovery=negative_discoveries,
        signed_calibrated_log_e=calibrated_log_e.astype(np.float32),
        signed_calibrated_e_bh_discovery=calibrated_discoveries,
        y_min=np.asarray(scan.y_min),
        x_min=np.asarray(scan.x_min),
        **maps,
    )
    save_overview(output_dir / "overview.png", maps, discoveries)

    window_scale = {}
    scene_median = np.median(scan.sample_radiance, axis=0)
    for name, indices in (("weak", weak_indices), ("strong", strong_indices)):
        ratio = scene_median[indices] / (resampled[0, indices] * 100.0)
        window_scale[name] = {
            "n_bands": int(len(indices)),
            "median_scene_to_modtran_times_100_ratio": float(np.median(ratio)),
            "ratio_min": float(np.min(ratio)),
            "ratio_max": float(np.max(ratio)),
        }

    scene_e_summary = null_diagnostics(maps["log_e_average"][valid_map])
    negative_control_summary = null_diagnostics(
        maps["log_e_negative_control"][valid_map]
    )
    summary: dict[str, object] = {
        "inputs": {
            "scene_csv": str(scene_path),
            "modtran_csv": str(modtran_path),
            "modtran_radiance_scale_used_for_uas": 1.0,
            "note_on_scale": (
                "Multiplying every MODTRAN radiance by 100 cancels exactly in the "
                "log-radiance slope/UAS. The factor is used only in the absolute-scale diagnostic."
            ),
        },
        "scene": {
            "total_rows": scan.total_rows,
            "valid_pixels": scan.valid_pixels,
            "sample_pixels": int(len(scan.sample_radiance)),
            "shape": [scan.y_max - scan.y_min + 1, scan.x_max - scan.x_min + 1],
            "y_range": [scan.y_min, scan.y_max],
            "x_range": [scan.x_min, scan.x_max],
        },
        "bands": {
            "weak_window_nm": [args.weak_min_nm, args.weak_max_nm],
            "strong_window_nm": [args.strong_min_nm, args.strong_max_nm],
            "weak_band_count": int(len(weak_indices)),
            "strong_band_count": int(len(strong_indices)),
            "weak_feature_count_after_continuum_removal": int(
                len(weak_feature_indices)
            ),
            "strong_feature_count_after_continuum_removal": int(
                len(strong_feature_indices)
            ),
            "selected_wavelengths_nm": [float(value) for value in wavelengths],
        },
        "model": {
            "fwhm_nm": args.fwhm_nm,
            "uas_alpha_fit_range": [args.uas_alpha_min, args.uas_alpha_max],
            "continuum_degree_removed_per_band_group": args.continuum_degree,
            "amplitude_cap": amplitude_cap,
            "covariance_shrinkage": args.covariance_shrinkage,
            "covariance_ridge": args.covariance_ridge,
            "nuisance_folds": args.nuisance_folds,
            "spatial_block_size": args.spatial_block_size,
            "local_tile_size": args.local_tile_size,
            "local_tile_halo": args.local_tile_halo,
            "local_model_count": len(models.local_models),
            "e_value_status": (
                "Gaussian plug-in pilot: exact conditional validity requires externally fixed "
                "mean/covariance; spatial nuisance folds reduce reuse but do not remove spatial dependence."
            ),
        },
        "template_family": family_summary,
        "absolute_radiance_scale_diagnostic": window_scale,
        "e_bh": {
            "target_fdr": args.fdr,
            "discovery_count": discovery_count,
            "log_e_threshold": log_e_threshold if np.isfinite(log_e_threshold) else None,
        },
        "scene_e_diagnostics": scene_e_summary,
        "signed_null_calibration_diagnostic": {
            "status": (
                "Heuristic only: clipping and pooled target/reverse normalization is "
                "an e-value calibration only under sign exchangeability."
            ),
            "log_e_cap": args.signed_null_log_e_cap,
            "pooled_log_mean_before_normalization": signed_log_normalizer,
            "positive_e_bh_discovery_count_at_target_fdr": calibrated_count,
            "positive_e_bh_log_e_threshold": (
                calibrated_threshold if np.isfinite(calibrated_threshold) else None
            ),
            "reverse_e_bh_discovery_count_at_target_fdr": calibrated_reverse_count,
        },
        "reverse_ch4_negative_control": {
            **negative_control_summary,
            "e_bh_discovery_count_at_target_fdr": negative_discovery_count,
            "e_bh_log_e_threshold": (
                negative_log_e_threshold
                if np.isfinite(negative_log_e_threshold)
                else None
            ),
        },
        "outputs": {
            "candidate_pixels": str(output_dir / "candidate_pixels.csv"),
            "injection_benchmark": str(output_dir / "injection_benchmark.csv"),
            "template_family": str(output_dir / "template_family_diagnostics.csv"),
            "target_spectrum": str(output_dir / "hisui_target_spectrum.csv"),
            "score_maps": str(output_dir / "score_maps.npz"),
            "overview": str(output_dir / "overview.png"),
        },
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

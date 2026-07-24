#!/usr/bin/env python3
"""Benchmark image-level methane plume recovery with exact MODTRAN ppm shifts.

An anisotropic downwind plume field is injected into real HISUI background
pixels.  Every pixel's spectral perturbation is obtained by interpolating the
MODTRAN log-radiance LUT at its known ppm enhancement, rather than by merely
adding a fixed matched-filter template.  Background models are fitted only on
the original scene and are kept fixed while injected images are scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from iterative_mf_1600nm import (
    compute_uas_log_slope,
    gaussian_srf_resample,
    get_wave_columns,
    load_ch4_lut,
)
from physics_aware_crossfit_mf import (
    DEFAULT_STRONG_WINDOW_NM,
    DEFAULT_WEAK_WINDOW_NM,
    NuisanceModelSet,
    build_nuisance_models,
    make_continuum_transform,
    scan_scene,
    score_log_radiance,
    spatial_fold_ids,
)


METHODS = ("strong_z", "combined_z", "weak_z", "dual_min_z", "log_e_average")


def parse_floats(value: str) -> list[float]:
    output = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated value.")
    return output


def score_features(
    features: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    models: NuisanceModelSet,
    *,
    block_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    folds = spatial_fold_ids(y, x, n_folds=models.n_folds, block_size=block_size)
    if models.tile_size > 0:
        y_tiles = np.floor_divide(y, models.tile_size)
        x_tiles = np.floor_divide(x, models.tile_size)
    else:
        y_tiles = np.zeros(len(y), dtype=int)
        x_tiles = np.zeros(len(x), dtype=int)
    groups = np.unique(np.column_stack([y_tiles, x_tiles, folds]), axis=0)
    for y_tile, x_tile, fold in groups:
        selected = (y_tiles == y_tile) & (x_tiles == x_tile) & (folds == fold)
        scores = score_log_radiance(
            features[selected], models.get(int(y_tile), int(x_tile), int(fold))
        )
        if not output:
            output = {
                key: np.full(len(features), np.nan, dtype=float) for key in scores
            }
        for key, values in scores.items():
            output[key][selected] = values
    return output


def plume_field(
    y: np.ndarray,
    x: np.ndarray,
    *,
    source_y: float,
    source_x: float,
    angle_radians: float,
    peak_ppm: float,
    length_scale: float,
    initial_width: float,
    spreading: float,
) -> np.ndarray:
    dy = np.asarray(y, dtype=float) - source_y
    dx = np.asarray(x, dtype=float) - source_x
    parallel = dx * math.cos(angle_radians) + dy * math.sin(angle_radians)
    perpendicular = -dx * math.sin(angle_radians) + dy * math.cos(angle_radians)
    width = initial_width + spreading * np.maximum(parallel, 0.0)
    field = peak_ppm * np.exp(-np.maximum(parallel, 0.0) / length_scale)
    field *= np.exp(-0.5 * (perpendicular / width) ** 2)
    field[parallel < 0.0] = 0.0
    return field


def draw_random_plume(
    y: np.ndarray,
    x: np.ndarray,
    valid: np.ndarray,
    peak_ppm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    y_min, y_max = int(y.min()), int(y.max())
    x_min, x_max = int(x.min()), int(x.max())
    y_margin = max(int(0.15 * (y_max - y_min + 1)), 8)
    x_margin = max(int(0.15 * (x_max - x_min + 1)), 8)
    for _ in range(1_000):
        source_y = float(rng.integers(y_min + y_margin, y_max - y_margin + 1))
        source_x = float(rng.integers(x_min + x_margin, x_max - x_margin + 1))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        length_scale = float(rng.uniform(15.0, 32.0))
        initial_width = float(rng.uniform(1.5, 3.5))
        spreading = float(rng.uniform(0.06, 0.14))
        field = plume_field(
            y,
            x,
            source_y=source_y,
            source_x=source_x,
            angle_radians=angle,
            peak_ppm=peak_ppm,
            length_scale=length_scale,
            initial_width=initial_width,
            spreading=spreading,
        )
        support = valid & (field >= 0.10 * peak_ppm)
        if int(support.sum()) >= 30 and np.all(valid[field >= 0.50 * peak_ppm]):
            return field, {
                "source_y": source_y,
                "source_x": source_x,
                "angle_degrees": float(np.degrees(angle)),
                "length_scale_pixels": length_scale,
                "initial_width_pixels": initial_width,
                "spreading": spreading,
            }
    raise RuntimeError("Could not place a sufficiently large valid plume.")


def exact_log_shift(
    ppm: np.ndarray,
    alpha_grid: np.ndarray,
    resampled_radiance: np.ndarray,
) -> np.ndarray:
    ppm = np.asarray(ppm, dtype=float)
    if ppm.min() < alpha_grid[0] or ppm.max() > alpha_grid[-1]:
        raise ValueError("Injected ppm values fall outside the MODTRAN LUT range.")
    log_radiance = np.log(resampled_radiance)
    output = np.empty((len(ppm), log_radiance.shape[1]), dtype=float)
    for band in range(log_radiance.shape[1]):
        output[:, band] = np.interp(ppm, alpha_grid, log_radiance[:, band])
        output[:, band] -= log_radiance[0, band]
    return output


def connected_overlap_detected(
    detections: np.ndarray,
    support: np.ndarray,
    shape: tuple[int, int],
    row: np.ndarray,
    column: np.ndarray,
    *,
    minimum_overlap: int,
) -> tuple[bool, int]:
    detection_map = np.zeros(shape, dtype=bool)
    support_map = np.zeros(shape, dtype=bool)
    detection_map[row, column] = detections
    support_map[row, column] = support
    labels, count = ndimage.label(detection_map, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return False, 0
    overlaps = np.bincount(labels[support_map], minlength=count + 1)
    largest = int(overlaps[1:].max(initial=0))
    return largest >= minimum_overlap, largest


def render_rgb(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.ones(rgb.shape, dtype=float)
    for band in range(3):
        values = rgb[:, :, band]
        finite = valid & np.isfinite(values)
        low, high = np.quantile(values[finite], [0.02, 0.98])
        output[:, :, band] = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    output[~valid] = 1.0
    return output


def to_map(
    values: np.ndarray,
    shape: tuple[int, int],
    row: np.ndarray,
    column: np.ndarray,
) -> np.ndarray:
    output = np.full(shape, np.nan, dtype=float)
    output[row, column] = values
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
    parser.add_argument("--continuum-degree", type=int, default=1)
    parser.add_argument("--uas-alpha-min", type=float, default=0.0)
    parser.add_argument("--uas-alpha-max", type=float, default=0.5)
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--spatial-block-size", type=int, default=10)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.05)
    parser.add_argument("--covariance-ridge", type=float, default=1e-6)
    parser.add_argument("--peaks-ppm", type=parse_floats, default=parse_floats("0.1,0.2,0.5,1"))
    parser.add_argument("--trials-per-peak", type=int, default=10)
    parser.add_argument("--representative-peak-ppm", type=float, default=0.5)
    parser.add_argument("--support-fraction", type=float, default=0.10)
    parser.add_argument("--fpr", type=float, default=0.001)
    parser.add_argument("--minimum-overlap-pixels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=743)
    parser.add_argument("--chunksize", type=int, default=50_000)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    scene_path = args.scene_csv.resolve()
    modtran_path = args.modtran_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(scene_path, nrows=0)
    all_wave_columns, all_wavelengths = get_wave_columns(header.columns)
    selected_mask = (
        ((all_wavelengths >= args.weak_min_nm) & (all_wavelengths <= args.weak_max_nm))
        | ((all_wavelengths >= args.strong_min_nm) & (all_wavelengths <= args.strong_max_nm))
    )
    wave_columns = [column for column, keep in zip(all_wave_columns, selected_mask) if keep]
    wavelengths = all_wavelengths[selected_mask]
    weak_indices = np.flatnonzero(
        (wavelengths >= args.weak_min_nm) & (wavelengths <= args.weak_max_nm)
    )
    strong_indices = np.flatnonzero(
        (wavelengths >= args.strong_min_nm) & (wavelengths <= args.strong_max_nm)
    )
    rgb_targets = np.asarray([655.0, 555.0, 485.0])
    rgb_indices = [int(np.argmin(np.abs(all_wavelengths - value))) for value in rgb_targets]
    rgb_columns = [all_wave_columns[index] for index in rgb_indices]
    rgb_wavelengths = [float(all_wavelengths[index]) for index in rgb_indices]

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
    transform, weak_features, strong_features = make_continuum_transform(
        wavelengths, weak_indices, strong_indices, degree=args.continuum_degree
    )
    target = transform @ raw_target
    scan = scan_scene(
        scene_path,
        wave_columns,
        chunksize=args.chunksize,
        sample_size=10_000_000,
    )
    models = build_nuisance_models(
        scan,
        transform,
        target,
        weak_features,
        strong_features,
        n_folds=args.nuisance_folds,
        block_size=args.spatial_block_size,
        shrinkage=args.covariance_shrinkage,
        ridge=args.covariance_ridge,
        amplitude_cap=float(alpha_grid[-1]),
        local_tile_size=0,
        local_tile_halo=1,
    )

    usecols = ["y", "x", *dict.fromkeys([*wave_columns, *rgb_columns])]
    frame = pd.read_csv(scene_path, usecols=usecols)
    y = frame["y"].to_numpy(int)
    x = frame["x"].to_numpy(int)
    radiance = frame[wave_columns].to_numpy(float)
    valid = np.all(np.isfinite(radiance), axis=1) & np.all(radiance > 0.0, axis=1)
    y = y[valid]
    x = x[valid]
    radiance = radiance[valid]
    if len(np.unique(np.column_stack([y, x]), axis=0)) != len(y):
        raise ValueError("Duplicate valid y,x coordinates are not supported.")
    original_features = np.log(radiance) @ transform.T
    original_scores = score_features(
        original_features,
        y,
        x,
        models,
        block_size=args.spatial_block_size,
    )
    thresholds = {
        method: float(
            np.quantile(original_scores[method], 1.0 - args.fpr, method="higher")
        )
        for method in METHODS
    }
    y_origin, x_origin = int(y.min()), int(x.min())
    shape = (int(y.max() - y_origin + 1), int(x.max() - x_origin + 1))
    row = y - y_origin
    column = x - x_origin
    valid_vector = np.ones(len(y), dtype=bool)
    valid_map = np.zeros(shape, dtype=bool)
    valid_map[row, column] = True
    rgb_map = np.full((*shape, 3), np.nan, dtype=float)
    rgb_map[row, column] = frame.loc[valid, rgb_columns].to_numpy(float)
    rendered_rgb = render_rgb(rgb_map, valid_map)

    rng = np.random.default_rng(args.seed)
    trial_rows: list[dict[str, float | int | str | bool]] = []
    recovery_rows: list[dict[str, float | int]] = []
    representative: dict[str, object] | None = None
    trial_id = 0
    for peak_ppm in args.peaks_ppm:
        for replicate in range(args.trials_per_peak):
            trial_id += 1
            ppm, geometry = draw_random_plume(
                y, x, valid_vector, peak_ppm, rng
            )
            support = ppm >= args.support_fraction * peak_ppm
            log_shift = exact_log_shift(ppm, alpha_grid, resampled)
            injected_features = original_features + log_shift @ transform.T
            injected_scores = score_features(
                injected_features,
                y,
                x,
                models,
                block_size=args.spatial_block_size,
            )
            delta_alpha = (
                injected_scores["combined_alpha"] - original_scores["combined_alpha"]
            )
            true_support = ppm[support]
            retrieved_support = delta_alpha[support]
            denominator = float(true_support @ true_support)
            slope = float(true_support @ retrieved_support / max(denominator, 1e-12))
            correlation = float(
                np.corrcoef(true_support, retrieved_support)[0, 1]
                if len(true_support) > 1
                else np.nan
            )
            recovery_rows.append(
                {
                    "trial_id": trial_id,
                    "peak_ppm": peak_ppm,
                    "support_pixels": int(support.sum()),
                    "combined_alpha_delta_slope": slope,
                    "combined_alpha_delta_correlation": correlation,
                    "combined_alpha_delta_rmse_ppm": float(
                        np.sqrt(np.mean((retrieved_support - true_support) ** 2))
                    ),
                    **geometry,
                }
            )
            for method in METHODS:
                detections = injected_scores[method] >= thresholds[method]
                detected, largest_overlap = connected_overlap_detected(
                    detections,
                    support,
                    shape,
                    row,
                    column,
                    minimum_overlap=args.minimum_overlap_pixels,
                )
                trial_rows.append(
                    {
                        "trial_id": trial_id,
                        "peak_ppm": peak_ppm,
                        "replicate": replicate,
                        "method": method,
                        "threshold_at_background_fpr": thresholds[method],
                        "background_fpr": float(
                            np.mean(original_scores[method] >= thresholds[method])
                        ),
                        "support_pixels": int(support.sum()),
                        "pixel_tpr_in_support": float(np.mean(detections[support])),
                        "plume_detected_minimum_connected_overlap": bool(detected),
                        "largest_connected_overlap_pixels": largest_overlap,
                        "mean_score_increase_in_support": float(
                            np.mean(
                                injected_scores[method][support]
                                - original_scores[method][support]
                            )
                        ),
                    }
                )
            if (
                representative is None
                and math.isclose(peak_ppm, args.representative_peak_ppm)
            ):
                representative = {
                    "ppm": ppm,
                    "support": support,
                    "scores": injected_scores,
                    "delta_alpha": delta_alpha,
                    "geometry": geometry,
                }

    trials = pd.DataFrame(trial_rows)
    recovery = pd.DataFrame(recovery_rows)
    summary_table = (
        trials.groupby(["peak_ppm", "method"], as_index=False)
        .agg(
            trial_count=("plume_detected_minimum_connected_overlap", "size"),
            detected_trials=("plume_detected_minimum_connected_overlap", "sum"),
            plume_detection_probability=(
                "plume_detected_minimum_connected_overlap",
                "mean",
            ),
            mean_pixel_tpr=("pixel_tpr_in_support", "mean"),
            median_pixel_tpr=("pixel_tpr_in_support", "median"),
            mean_largest_connected_overlap=("largest_connected_overlap_pixels", "mean"),
            mean_score_increase=("mean_score_increase_in_support", "mean"),
        )
        .sort_values(["peak_ppm", "method"])
    )
    # Wilson intervals make the uncertainty from the finite number of plume
    # placements visible instead of presenting each detection rate as exact.
    z_95 = 1.959963984540054
    trial_count = summary_table["trial_count"].to_numpy(float)
    probability = summary_table["plume_detection_probability"].to_numpy(float)
    denominator = 1.0 + z_95**2 / trial_count
    center = (probability + z_95**2 / (2.0 * trial_count)) / denominator
    half_width = (
        z_95
        * np.sqrt(
            probability * (1.0 - probability) / trial_count
            + z_95**2 / (4.0 * trial_count**2)
        )
        / denominator
    )
    summary_table["wilson_95_low"] = np.maximum(center - half_width, 0.0)
    summary_table["wilson_95_high"] = np.minimum(center + half_width, 1.0)
    trials.to_csv(output_dir / "spatial_injection_trials.csv", index=False)
    recovery.to_csv(output_dir / "ppm_recovery_trials.csv", index=False)
    summary_table.to_csv(output_dir / "spatial_injection_summary.csv", index=False)

    method_labels = {
        "combined_z": "Combined MF",
        "dual_min_z": "Dual-band minimum",
        "log_e_average": "Bidirectional cross-fit",
        "strong_z": "Strong-band MF",
        "weak_z": "Weak-band MF",
    }
    fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for method, label in method_labels.items():
        selected = summary_table[summary_table["method"] == method].sort_values(
            "peak_ppm"
        )
        x = selected["peak_ppm"].to_numpy(float)
        y = selected["plume_detection_probability"].to_numpy(float)
        lower = selected["wilson_95_low"].to_numpy(float)
        upper = selected["wilson_95_high"].to_numpy(float)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_xticks(args.peaks_ppm, [f"{value:g}" for value in args.peaks_ppm])
    axis.set_ylim(-0.02, 1.04)
    axis.set_xlabel("Injected plume peak enhancement [ppm]")
    axis.set_ylabel("Plume detection probability")
    axis.set_title(
        f"Spatial plume recovery at background FPR={args.fpr:g}\n"
        f"{args.trials_per_peak} placements per concentration; bars are Wilson 95% intervals"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    fig.savefig(output_dir / "spatial_plume_detection_probability.png", dpi=180)
    plt.close(fig)

    if representative is None:
        raise ValueError("representative_peak_ppm must be included in peaks_ppm.")
    ppm_map = to_map(representative["ppm"], shape, row, column)
    delta_alpha_map = to_map(representative["delta_alpha"], shape, row, column)
    combined_z_map = to_map(representative["scores"]["combined_z"], shape, row, column)
    log_e_map = to_map(representative["scores"]["log_e_average"], shape, row, column)
    combined_detection = combined_z_map >= thresholds["combined_z"]
    support_map = np.zeros(shape, dtype=bool)
    support_map[row, column] = representative["support"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rendered_rgb)
    axes[0, 0].set_title(f"Real HISUI background RGB {rgb_wavelengths} nm")
    shown = axes[0, 1].imshow(ppm_map, cmap="magma", vmin=0.0, vmax=args.representative_peak_ppm)
    axes[0, 1].set_title("Injected MODTRAN plume [ppm]")
    fig.colorbar(shown, ax=axes[0, 1], shrink=0.78)
    finite_delta = delta_alpha_map[np.isfinite(delta_alpha_map)]
    limit = max(
        float(np.quantile(np.abs(finite_delta), 0.995)),
        1.10 * args.representative_peak_ppm,
    )
    shown = axes[0, 2].imshow(delta_alpha_map, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0, 2].set_title("Retrieved combined-alpha increase [ppm]")
    fig.colorbar(shown, ax=axes[0, 2], shrink=0.78)

    finite_combined = combined_z_map[np.isfinite(combined_z_map)]
    shown = axes[1, 0].imshow(
        combined_z_map,
        cmap="viridis",
        vmin=float(np.quantile(finite_combined, 0.01)),
        vmax=float(np.quantile(finite_combined, 0.995)),
    )
    axes[1, 0].contour(support_map, [0.5], colors="cyan", linewidths=0.8)
    axes[1, 0].set_title("Injected combined MF z; cyan=true support")
    fig.colorbar(shown, ax=axes[1, 0], shrink=0.78)
    finite_e = log_e_map[np.isfinite(log_e_map)]
    shown = axes[1, 1].imshow(
        log_e_map,
        cmap="plasma",
        vmin=float(np.quantile(finite_e, 0.01)),
        vmax=float(np.quantile(finite_e, 0.995)),
    )
    axes[1, 1].contour(support_map, [0.5], colors="cyan", linewidths=0.8)
    axes[1, 1].set_title("Injected bidirectional cross-fit log e")
    fig.colorbar(shown, ax=axes[1, 1], shrink=0.78)
    axes[1, 2].imshow(rendered_rgb)
    axes[1, 2].contour(support_map, [0.5], colors="cyan", linewidths=1.0)
    if combined_detection.any():
        axes[1, 2].contour(combined_detection, [0.5], colors="yellow", linewidths=0.8)
    axes[1, 2].set_title("cyan=true plume; yellow=combined-MF detections")
    for axis in axes.ravel():
        axis.set_axis_off()
    fig.savefig(output_dir / "representative_spatial_plume_recovery.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "representative_spatial_plume_maps.npz",
        valid=valid_map,
        injected_ppm=ppm_map.astype(np.float32),
        true_support=support_map,
        retrieved_delta_alpha_ppm=delta_alpha_map.astype(np.float32),
        injected_combined_z=combined_z_map.astype(np.float32),
        injected_log_e=log_e_map.astype(np.float32),
        combined_detection=combined_detection,
    )
    summary: dict[str, object] = {
        "inputs": {
            "scene_csv": str(scene_path),
            "modtran_csv": str(modtran_path),
            "ppm_interpretation": (
                "MODTRAN LUT column headers are ppm; injection and retrieved alpha "
                "are reported as ppm enhancement relative to alpha=0."
            ),
        },
        "experiment": {
            "peaks_ppm": args.peaks_ppm,
            "trials_per_peak": args.trials_per_peak,
            "support_fraction_of_peak": args.support_fraction,
            "background_fpr": args.fpr,
            "minimum_connected_overlap_pixels": args.minimum_overlap_pixels,
            "exact_modtran_log_ratio_injection": True,
            "background_model_refitted_after_injection": False,
        },
        "thresholds": thresholds,
        "ppm_recovery_by_peak": recovery.groupby("peak_ppm").agg(
            mean_slope=("combined_alpha_delta_slope", "mean"),
            mean_correlation=("combined_alpha_delta_correlation", "mean"),
            mean_rmse_ppm=("combined_alpha_delta_rmse_ppm", "mean"),
        ).reset_index().to_dict(orient="records"),
        "outputs": {
            "trial_table": str(output_dir / "spatial_injection_trials.csv"),
            "summary_table": str(output_dir / "spatial_injection_summary.csv"),
            "ppm_recovery_table": str(output_dir / "ppm_recovery_trials.csv"),
            "detection_probability_figure": str(
                output_dir / "spatial_plume_detection_probability.png"
            ),
            "representative_figure": str(output_dir / "representative_spatial_plume_recovery.png"),
            "representative_maps": str(output_dir / "representative_spatial_plume_maps.npz"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

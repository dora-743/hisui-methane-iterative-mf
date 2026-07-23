#!/usr/bin/env python3
"""Compare in-loop stripe removal strategies for 1600 nm Iterative MF.

Three runs share the same input spectra, methane target, covariance settings,
and robust threshold:

* ``baseline``: no stripe removal.
* ``pdf_dwt_then_median``: broad DWT (slope 1.257, levels 3--5), then
  narrow-line median, matching the supplied PDF flow.
* ``qa_median_then_ct_dwt``: narrow primary and orthogonal medians, then CT
  metadata-guided DWT (slope 1.266720, levels 2--5), matching the newer
  QA-guided 2200 nm recommendation.

The narrow-line slope is re-estimated from QA_DM in the selected 1600 nm
bands (0.977346 row/column). The corrected alpha map, rather than the raw MF map, is thresholded during
every iteration.  This prevents a stripe from being excluded as a plume and
then amplified by the next background-covariance estimate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

import iterative_mf_1600nm as imf
from directional_destriping import (
    directional_profile,
    fixed_slope_line_ids,
    median_fixed_slope_destripe,
    robust_std,
    wavelet_horizontal_destripe,
)


# Mean PCA slope of QA_DM traces in HISUI bands 117, 118, 121, and 122
# (1625.435, 1637.925, 1675.395, and 1687.885 nm). QA_IM has no marked
# pixels in any of the nine selected 1600 nm bands.
THIN_SLOPE = 0.9773460526106752
THIN_ORTHOGONAL_SLOPE = 1.0 / THIN_SLOPE
DETECTED_BROAD_SLOPE = 1.257172298918948
CT_METADATA_MIRROR_SLOPE = 1.2667203308533619

METHODS = ("baseline", "pdf_dwt_then_median", "qa_median_then_ct_dwt")

DEFAULT_ROI = Path("data/all_roi_spectra200x200.csv")
DEFAULT_LUT = Path("data/ch4_lut.csv")
DEFAULT_OUTPUT = Path("outputs/iterative_mf_1600nm_destriped")
DEFAULT_REFERENCE_2200_MASK = Path(
    "data/2200nm/reference_plume_mask.npy"
)
DEFAULT_REFERENCE_2200_ALPHA = Path(
    "outputs/iterative_mf_1600nm_destriped/2200nm_residual_thin_refined/"
    "recommended_alpha_corrected_refined.npy"
)


@dataclass(frozen=True)
class PreparedInput:
    roi: imf.RoiCube
    cube: np.ndarray
    wavelengths_nm: np.ndarray
    uas: np.ndarray
    valid_mask: np.ndarray


def prepare_input(
    roi_csv: Path,
    lut_csv: Path,
    *,
    wl_min_nm: float,
    wl_max_nm: float,
    fwhm_nm: float,
    alpha_min: float,
    alpha_max: float,
    nodata_values: Sequence[float],
    require_positive: bool,
) -> PreparedInput:
    roi = imf.load_roi_cube(roi_csv)
    lut_wavelengths, alpha_grid, lut_spectra = imf.load_ch4_lut(lut_csv)
    lut_at_sensor = imf.gaussian_srf_resample(
        lut_wavelengths, lut_spectra, roi.wavelengths_nm, fwhm_nm
    )
    uas_all = imf.compute_uas_log_slope(alpha_grid, lut_at_sensor, alpha_min, alpha_max)
    cube, wavelengths_nm, uas = imf.select_wavelength_window(
        roi.cube, roi.wavelengths_nm, uas_all, wl_min_nm, wl_max_nm
    )
    valid_mask = imf.make_valid_mask(cube, nodata_values, require_positive)
    return PreparedInput(roi, cube, wavelengths_nm, uas, valid_mask)


def conservative_protection_mask(
    raw_alpha: np.ndarray,
    valid_mask: np.ndarray,
    previous_plume: np.ndarray | None,
    protect_nsigma: float,
    line_protection_cap: int,
) -> tuple[np.ndarray, int]:
    threshold, median, scale = imf.robust_threshold(raw_alpha[valid_mask], protect_nsigma)
    protected = valid_mask & (raw_alpha > threshold)
    if previous_plume is not None:
        # Keep a previous candidate protected only while it is still above the
        # current raw median. This avoids permanently protecting a transient stripe.
        protected |= previous_plume & valid_mask & (raw_alpha > median + scale)

    # QA_IM showed that the strongest thin feature is a sensor/geometry line.
    # Do not protect an entire high-alpha line from the stripe estimator. A
    # compact plume spans several line IDs and has far fewer pixels per ID.
    ids = fixed_slope_line_ids(
        raw_alpha.shape,
        slope=THIN_SLOPE,
        line_bin_width=2.0,
        direction_key="y_minus_x",
    )
    line_like = np.zeros_like(protected)
    protected_ids, counts = np.unique(ids[protected], return_counts=True)
    for line_id in protected_ids[counts >= line_protection_cap]:
        line_like |= protected & (ids == line_id)
    protected &= ~line_like
    return protected, int(line_like.sum())


def correct_alpha(
    raw_alpha: np.ndarray,
    valid_mask: np.ndarray,
    protected_mask: np.ndarray,
    *,
    method: str,
    iteration: int,
    canvas_size: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    if method == "baseline":
        return raw_alpha.copy(), {"total": np.zeros_like(raw_alpha)}, [], []

    dwt_rows: list[dict[str, float | int | str]] = []
    median_rows: list[dict[str, float | int | str]] = []
    stripe_maps: dict[str, np.ndarray] = {}

    if method == "pdf_dwt_then_median":
        broad_corrected, broad_stripe, rows = wavelet_horizontal_destripe(
            raw_alpha,
            protected_mask,
            slope=DETECTED_BROAD_SLOPE,
            levels_to_filter=(3, 4, 5),
            threshold_scale=0.75,
            diff_fraction=0.25,
            canvas_size=canvas_size,
            operation_name=f"{method}:iter{iteration:02d}:broad_dwt",
        )
        dwt_rows.extend(rows)
        corrected, thin_stripe, rows = median_fixed_slope_destripe(
            broad_corrected,
            slope=THIN_SLOPE,
            line_bin_width=2.0,
            min_pixels_per_line=5,
            direction_key="y_minus_x",
            valid_mask=valid_mask,
            exclude_mask=ndimage.binary_dilation(protected_mask, iterations=1),
            operation_name=f"{method}:iter{iteration:02d}:thin_median",
        )
        median_rows.extend(rows)
        stripe_maps = {"broad_dwt": broad_stripe, "thin_median": thin_stripe}

    elif method == "qa_median_then_ct_dwt":
        primary_corrected, primary_stripe, rows = median_fixed_slope_destripe(
            raw_alpha,
            slope=THIN_SLOPE,
            line_bin_width=2.0,
            min_pixels_per_line=5,
            direction_key="y_minus_x",
            valid_mask=valid_mask,
            exclude_mask=ndimage.binary_dilation(protected_mask, iterations=1),
            operation_name=f"{method}:iter{iteration:02d}:thin_primary_median",
        )
        median_rows.extend(rows)
        thin_corrected, orthogonal_stripe, rows = median_fixed_slope_destripe(
            primary_corrected,
            slope=THIN_ORTHOGONAL_SLOPE,
            line_bin_width=2.0,
            min_pixels_per_line=5,
            direction_key="y_plus_x",
            valid_mask=valid_mask,
            exclude_mask=ndimage.binary_dilation(protected_mask, iterations=1),
            operation_name=f"{method}:iter{iteration:02d}:thin_orthogonal_median",
        )
        median_rows.extend(rows)
        corrected, ct_stripe, rows = wavelet_horizontal_destripe(
            thin_corrected,
            protected_mask,
            slope=CT_METADATA_MIRROR_SLOPE,
            levels_to_filter=(2, 3, 4, 5),
            threshold_scale=1.05,
            diff_fraction=0.25,
            canvas_size=canvas_size,
            operation_name=f"{method}:iter{iteration:02d}:ct_dwt",
        )
        dwt_rows.extend(rows)
        stripe_maps = {
            "thin_primary_median": primary_stripe,
            "thin_orthogonal_median": orthogonal_stripe,
            "ct_dwt": ct_stripe,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    stripe_maps["total"] = raw_alpha - corrected
    return corrected, stripe_maps, dwt_rows, median_rows


def run_iterative_method(
    prepared: PreparedInput,
    *,
    method: str,
    n_iter: int,
    nsigma: float,
    protect_nsigma: float,
    regularization: float,
    rcond: float,
    canvas_size: int,
    line_protection_cap: int,
    convergence_pixels: int,
    verbose: bool,
) -> dict[str, object]:
    cube = prepared.cube
    uas = prepared.uas
    valid_mask = prepared.valid_mask
    height, width, n_bands = cube.shape
    min_background_pixels = max(n_bands + 5, 30)
    background_mask = valid_mask.copy()
    previous_plume: np.ndarray | None = None
    valid_spectra = cube[valid_mask]

    raw_history: list[np.ndarray] = []
    corrected_history: list[np.ndarray] = []
    plume_history: list[np.ndarray] = []
    protection_history: list[np.ndarray] = []
    background_history: list[np.ndarray] = []
    mean_history: list[np.ndarray] = []
    covariance_history: list[np.ndarray] = []
    stripe_history: list[dict[str, np.ndarray]] = []
    iteration_rows: list[dict[str, float | int | str]] = []
    all_dwt_rows: list[dict[str, float | int | str]] = []
    all_median_rows: list[dict[str, float | int | str]] = []
    did_converge = False
    convergence_mode = "maximum_iterations"
    use_two_cycle_average = False

    for iteration in range(1, n_iter + 1):
        background_history.append(background_mask.copy())
        mean, covariance = imf.estimate_background(
            cube, background_mask, regularization, min_background_pixels
        )
        target = -mean * uas
        inverse_covariance = np.linalg.pinv(covariance, rcond=rcond)
        weighted_target = inverse_covariance @ target
        denominator = float(target @ weighted_target)
        if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
            raise ValueError(f"{method}: MF denominator is too small at iteration {iteration}.")

        raw_alpha = np.full((height, width), np.nan, dtype=float)
        raw_alpha[valid_mask] = (valid_spectra - mean) @ weighted_target / denominator
        protected, line_like_unprotected = conservative_protection_mask(
            raw_alpha,
            valid_mask,
            previous_plume,
            protect_nsigma,
            line_protection_cap,
        )
        corrected, stripe_maps, dwt_rows, median_rows = correct_alpha(
            raw_alpha,
            valid_mask,
            protected,
            method=method,
            iteration=iteration,
            canvas_size=canvas_size,
        )
        threshold, median, scale = imf.robust_threshold(corrected[valid_mask], nsigma)
        plume = valid_mask & (corrected > threshold)
        next_background = valid_mask & ~plume
        changed_pixels = (
            int(np.sum(plume ^ previous_plume)) if previous_plume is not None else -1
        )

        raw_history.append(raw_alpha)
        corrected_history.append(corrected)
        plume_history.append(plume)
        protection_history.append(protected)
        mean_history.append(mean)
        covariance_history.append(covariance)
        stripe_history.append(stripe_maps)
        all_dwt_rows.extend(dwt_rows)
        all_median_rows.extend(median_rows)
        iteration_rows.append(
            {
                "method": method,
                "iteration": iteration,
                "background_pixels": int(background_mask.sum()),
                "protected_pixels": int(protected.sum()),
                "line_like_pixels_unprotected": line_like_unprotected,
                "threshold": threshold,
                "median": median,
                "robust_std": scale,
                "plume_pixels": int(plume.sum()),
                "candidate_mask_changed_pixels": changed_pixels,
                "stripe_robust_std": robust_std(stripe_maps["total"], valid_mask),
            }
        )
        if verbose:
            print(
                f"{method:26s} iter {iteration:02d}: background={background_mask.sum():5d}, "
                f"protected={protected.sum():4d}, threshold={threshold:.6g}, "
                f"rstd={scale:.6g}, plume={plume.sum():4d}, changed={changed_pixels:4d}"
            )

        background_mask = next_background
        if previous_plume is not None and changed_pixels <= convergence_pixels:
            did_converge = True
            convergence_mode = "fixed_point_tolerance"
            if verbose:
                print(
                    f"{method}: converged at iteration {iteration} "
                    f"(changed pixels={changed_pixels})."
                )
            break
        if (
            len(plume_history) >= 3
            and np.array_equal(plume_history[-1], plume_history[-3])
        ):
            did_converge = True
            convergence_mode = "two_cycle_average"
            use_two_cycle_average = True
            if verbose:
                print(
                    f"{method}: detected a two-cycle at iteration {iteration}; "
                    "using the mean of the last two corrected alpha maps."
                )
            break
        previous_plume = plume.copy()

    if use_two_cycle_average:
        final_raw = 0.5 * (raw_history[-1] + raw_history[-2])
        final_corrected = 0.5 * (corrected_history[-1] + corrected_history[-2])
        final_threshold, _, _ = imf.robust_threshold(final_corrected[valid_mask], nsigma)
        final_plume = valid_mask & (final_corrected > final_threshold)
        final_background = valid_mask & ~final_plume
        final_protected = protection_history[-1] | protection_history[-2]
        stripe_keys = set(stripe_history[-1]) | set(stripe_history[-2])
        final_stripes = {
            key: 0.5 * (stripe_history[-1][key] + stripe_history[-2][key])
            for key in stripe_keys
        }
    else:
        final_raw = raw_history[-1]
        final_corrected = corrected_history[-1]
        final_plume = plume_history[-1]
        final_background = background_mask
        final_protected = protection_history[-1]
        final_stripes = stripe_history[-1]

    return {
        "method": method,
        "alpha_raw": final_raw,
        "alpha_corrected": final_corrected,
        "plume_mask": final_plume,
        "background_mask": final_background,
        "protected_mask": final_protected,
        "stripe_maps": final_stripes,
        "raw_history": raw_history,
        "corrected_history": corrected_history,
        "plume_history": plume_history,
        "background_history": background_history,
        "mean_history": mean_history,
        "covariance_history": covariance_history,
        "iteration_table": pd.DataFrame(iteration_rows),
        "dwt_table": pd.DataFrame(all_dwt_rows),
        "median_table": pd.DataFrame(all_median_rows),
        "converged": did_converge,
        "convergence_mode": convergence_mode,
    }


def connected_component_metrics(mask: np.ndarray) -> dict[str, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "component_count": int(count),
        "largest_component_pixels": int(sizes.max()) if sizes.size else 0,
        "singleton_components": int(np.sum(sizes == 1)),
    }


def method_metrics(
    result: dict[str, object],
    valid_mask: np.ndarray,
    reference_2200_mask: np.ndarray | None,
) -> dict[str, float | int | str | bool]:
    method = str(result["method"])
    alpha = np.asarray(result["alpha_corrected"], dtype=float)
    plume = np.asarray(result["plume_mask"], dtype=bool)
    background = valid_mask & ~plume

    def profile_scale(slope: float, width: float) -> float:
        _, profile = directional_profile(
            alpha, slope=slope, bin_width=width, protected_mask=plume
        )
        return robust_std(profile)

    row: dict[str, float | int | str | bool] = {
        "method": method,
        "iterations_completed": len(result["plume_history"]),
        "converged": bool(result["converged"]),
        "convergence_mode": str(result["convergence_mode"]),
        "candidate_pixels": int(plume.sum()),
        "robust_std_background": robust_std(alpha, background),
        "p95_abs_background": float(np.percentile(np.abs(alpha[background]), 95)),
        "thin_profile_rstd_slope_0p977": profile_scale(THIN_SLOPE, 2.0),
        "orthogonal_profile_rstd_slope_minus_1p023": profile_scale(-THIN_ORTHOGONAL_SLOPE, 2.0),
        "broad_profile_rstd_slope_1p257": profile_scale(DETECTED_BROAD_SLOPE, 18.0),
        "ct_profile_rstd_slope_1p267": profile_scale(CT_METADATA_MIRROR_SLOPE, 18.0),
    }
    row.update(connected_component_metrics(plume))
    if reference_2200_mask is not None:
        overlap = int(np.sum(plume & reference_2200_mask))
        union = int(np.sum(plume | reference_2200_mask))
        row.update(
            {
                "overlap_with_2200_pixels": overlap,
                "fraction_of_1600_candidates_in_2200": overlap / max(int(plume.sum()), 1),
                "fraction_of_2200_candidates_recovered": overlap
                / max(int(reference_2200_mask.sum()), 1),
                "jaccard_with_2200": overlap / max(union, 1),
            }
        )
    return row


def save_method_outputs(
    output_dir: Path,
    prepared: PreparedInput,
    result: dict[str, object],
) -> None:
    method = str(result["method"])
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "alpha_raw": result["alpha_raw"],
        "alpha_corrected": result["alpha_corrected"],
        "plume_mask": result["plume_mask"],
        "background_mask": result["background_mask"],
        "protected_mask": result["protected_mask"],
    }
    for name, array in arrays.items():
        np.save(method_dir / f"{name}.npy", np.asarray(array))
    for name, stripe in dict(result["stripe_maps"]).items():
        np.save(method_dir / f"stripe_{name}.npy", np.asarray(stripe))

    iteration_table = result["iteration_table"]
    dwt_table = result["dwt_table"]
    median_table = result["median_table"]
    assert isinstance(iteration_table, pd.DataFrame)
    assert isinstance(dwt_table, pd.DataFrame)
    assert isinstance(median_table, pd.DataFrame)
    iteration_table.to_csv(method_dir / "iteration_history.csv", index=False)
    if not dwt_table.empty:
        dwt_table.to_csv(method_dir / "dwt_thresholds.csv", index=False)
    if not median_table.empty:
        median_table.to_csv(method_dir / "median_line_offsets.csv", index=False)

    rows, columns = np.indices(prepared.valid_mask.shape)
    alpha_raw = np.asarray(result["alpha_raw"])
    alpha_corrected = np.asarray(result["alpha_corrected"])
    plume = np.asarray(result["plume_mask"], dtype=bool)
    pixel_table = pd.DataFrame(
        {
            "row": rows.ravel(),
            "col": columns.ravel(),
            "y": prepared.roi.ys[rows.ravel()],
            "x": prepared.roi.xs[columns.ravel()],
            "alpha_raw": alpha_raw.ravel(),
            "alpha_corrected": alpha_corrected.ravel(),
            "stripe_removed": (alpha_raw - alpha_corrected).ravel(),
            "is_candidate": plume.ravel(),
            "is_valid": prepared.valid_mask.ravel(),
        }
    ).sort_values("alpha_corrected", ascending=False, na_position="last")
    pixel_table.to_csv(method_dir / "pixel_results.csv", index=False)
    pixel_table[pixel_table["is_candidate"]].to_csv(
        method_dir / "candidate_pixels.csv", index=False
    )


def save_comparison_figure(
    path: Path,
    results: dict[str, dict[str, object]],
    reference_2200_mask: np.ndarray | None,
) -> None:
    corrected_images = [np.asarray(results[name]["alpha_corrected"]) for name in METHODS]
    values = np.concatenate([image[np.isfinite(image)] for image in corrected_images])
    vmin, vmax = np.percentile(values, [1, 99.5])
    stripe_values = np.concatenate(
        [
            np.asarray(results[name]["stripe_maps"]["total"])[
                np.isfinite(np.asarray(results[name]["stripe_maps"]["total"]))
            ]
            for name in METHODS[1:]
        ]
    )
    stripe_limit = max(float(np.percentile(np.abs(stripe_values), 99)), 1e-12)

    fig, axes = plt.subplots(4, len(METHODS), figsize=(13, 14), constrained_layout=True)
    for column, name in enumerate(METHODS):
        result = results[name]
        raw = np.asarray(result["alpha_raw"])
        corrected = np.asarray(result["alpha_corrected"])
        stripe = np.asarray(result["stripe_maps"]["total"])
        plume = np.asarray(result["plume_mask"], dtype=bool)
        for row, (image, title, cmap, lo, hi) in enumerate(
            [
                (raw, "raw MF alpha", "viridis", vmin, vmax),
                (corrected, "corrected alpha", "viridis", vmin, vmax),
                (stripe, "removed stripe", "coolwarm", -stripe_limit, stripe_limit),
                (plume.astype(float), "1600 nm candidates", "gray_r", 0, 1),
            ]
        ):
            axis = axes[row, column]
            shown = axis.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
            if reference_2200_mask is not None and row in (1, 3):
                axis.contour(reference_2200_mask, levels=[0.5], colors="magenta", linewidths=0.55)
            axis.set_title(f"{name}\n{title}" if row == 0 else title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == len(METHODS) - 1 and row < 3:
                fig.colorbar(shown, ax=axis, shrink=0.72)
    fig.suptitle("1600 nm Iterative MF: in-loop directional stripe removal\nmagenta = 2200 nm reference candidates")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_profile_figure(
    path: Path,
    results: dict[str, dict[str, object]],
) -> None:
    specs = [
        ("1600 nm QA_DM thin slope 0.977", THIN_SLOPE, 2.0),
        ("orthogonal slope -1.023", -THIN_ORTHOGONAL_SLOPE, 2.0),
        ("detected broad slope 1.257", DETECTED_BROAD_SLOPE, 18.0),
        ("CT metadata slope 1.267", CT_METADATA_MIRROR_SLOPE, 18.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (title, slope, width) in zip(axes.ravel(), specs):
        for name in METHODS:
            result = results[name]
            x_values, profile = directional_profile(
                np.asarray(result["alpha_corrected"]),
                slope=slope,
                bin_width=width,
                protected_mask=np.asarray(result["plume_mask"]),
            )
            axis.plot(x_values, profile, linewidth=1.05, label=name)
        axis.set_title(title)
        axis.set_xlabel("line coordinate")
        axis.set_ylabel("median high-pass MF")
        axis.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_reference_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    mask = np.load(path).astype(bool)
    if mask.shape != shape:
        raise ValueError(f"2200 nm reference mask shape {mask.shape} != 1600 nm shape {shape}.")
    return mask


def load_reference_alpha(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    alpha = np.load(path).astype(float)
    if alpha.shape != shape:
        raise ValueError(f"2200 nm reference alpha shape {alpha.shape} != 1600 nm shape {shape}.")
    return alpha


def save_cross_band_consensus(
    output_dir: Path,
    prepared: PreparedInput,
    alpha_1600: np.ndarray,
    mask_1600: np.ndarray,
    mask_2200: np.ndarray | None,
    alpha_2200: np.ndarray | None,
    *,
    min_component_pixels: int = 2,
) -> dict[str, int] | None:
    if mask_2200 is None:
        return None
    consensus = np.asarray(mask_1600, dtype=bool) & mask_2200
    labels, component_count = ndimage.label(
        consensus, structure=np.ones((3, 3), dtype=int)
    )
    sizes = np.bincount(labels.ravel())
    high_confidence = np.zeros_like(consensus)
    cluster_rows: list[dict[str, float | int]] = []
    for component_id in range(1, component_count + 1):
        selected = labels == component_id
        size = int(selected.sum())
        rows, columns = np.nonzero(selected)
        if size >= min_component_pixels:
            high_confidence |= selected
        alpha_values = alpha_1600[selected]
        peak_index = int(np.nanargmax(alpha_values))
        peak_row = int(rows[peak_index])
        peak_col = int(columns[peak_index])
        row: dict[str, float | int] = {
            "component_id": component_id,
            "pixels": size,
            "passes_min_component_pixels": int(size >= min_component_pixels),
            "row_min": int(rows.min()),
            "row_max": int(rows.max()),
            "col_min": int(columns.min()),
            "col_max": int(columns.max()),
            "centroid_row": float(rows.mean()),
            "centroid_col": float(columns.mean()),
            "peak_row": peak_row,
            "peak_col": peak_col,
            "peak_y": float(prepared.roi.ys[peak_row]),
            "peak_x": float(prepared.roi.xs[peak_col]),
            "alpha_1600_mean": float(np.nanmean(alpha_values)),
            "alpha_1600_max": float(np.nanmax(alpha_values)),
        }
        if alpha_2200 is not None:
            row["alpha_2200_mean"] = float(np.nanmean(alpha_2200[selected]))
            row["alpha_2200_max"] = float(np.nanmax(alpha_2200[selected]))
        cluster_rows.append(row)
    cluster_rows.sort(key=lambda row: int(row["pixels"]), reverse=True)
    pd.DataFrame(cluster_rows).to_csv(
        output_dir / "cross_band_consensus_clusters.csv", index=False
    )

    # Join components separated by a one-pixel gap into practical candidate
    # regions while retaining only the original consensus pixels for statistics.
    region_canvas = ndimage.binary_dilation(
        high_confidence, structure=np.ones((3, 3), dtype=bool), iterations=1
    )
    region_labels, region_count = ndimage.label(
        region_canvas, structure=np.ones((3, 3), dtype=int)
    )
    region_rows: list[dict[str, float | int]] = []
    used_regions = 0
    for region_id in range(1, region_count + 1):
        selected = high_confidence & (region_labels == region_id)
        if not selected.any():
            continue
        used_regions += 1
        rows_region, columns_region = np.nonzero(selected)
        values_1600 = alpha_1600[selected]
        peak_index = int(np.nanargmax(values_1600))
        peak_row = int(rows_region[peak_index])
        peak_col = int(columns_region[peak_index])
        row_region: dict[str, float | int] = {
            "region_id": used_regions,
            "consensus_pixels": int(selected.sum()),
            "row_min": int(rows_region.min()),
            "row_max": int(rows_region.max()),
            "col_min": int(columns_region.min()),
            "col_max": int(columns_region.max()),
            "centroid_row": float(rows_region.mean()),
            "centroid_col": float(columns_region.mean()),
            "peak_row": peak_row,
            "peak_col": peak_col,
            "peak_y": float(prepared.roi.ys[peak_row]),
            "peak_x": float(prepared.roi.xs[peak_col]),
            "alpha_1600_mean": float(np.nanmean(values_1600)),
            "alpha_1600_max": float(np.nanmax(values_1600)),
        }
        if alpha_2200 is not None:
            row_region["alpha_2200_mean"] = float(np.nanmean(alpha_2200[selected]))
            row_region["alpha_2200_max"] = float(np.nanmax(alpha_2200[selected]))
        region_rows.append(row_region)
    region_table = pd.DataFrame(region_rows)
    if not region_table.empty:
        region_table = region_table.sort_values("consensus_pixels", ascending=False)
    region_table.to_csv(output_dir / "cross_band_consensus_regions.csv", index=False)

    rows, columns = np.nonzero(consensus)
    pixel_data: dict[str, np.ndarray] = {
        "component_id": labels[consensus],
        "row": rows,
        "col": columns,
        "y": prepared.roi.ys[rows],
        "x": prepared.roi.xs[columns],
        "alpha_1600": alpha_1600[consensus],
        "is_high_confidence_component": high_confidence[consensus],
    }
    if alpha_2200 is not None:
        pixel_data["alpha_2200"] = alpha_2200[consensus]
    pd.DataFrame(pixel_data).sort_values("alpha_1600", ascending=False).to_csv(
        output_dir / "cross_band_consensus_pixels.csv", index=False
    )
    np.save(output_dir / "cross_band_consensus_mask.npy", consensus)
    np.save(output_dir / "cross_band_consensus_connected_mask.npy", high_confidence)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    finite_1600 = alpha_1600[np.isfinite(alpha_1600)]
    lo1600, hi1600 = np.percentile(finite_1600, [1, 99.5])
    shown = axes[0].imshow(alpha_1600, cmap="viridis", vmin=lo1600, vmax=hi1600)
    axes[0].contour(high_confidence, levels=[0.5], colors="red", linewidths=0.8)
    axes[0].set_title("1600 nm corrected alpha")
    fig.colorbar(shown, ax=axes[0], shrink=0.75)
    if alpha_2200 is not None:
        finite_2200 = alpha_2200[np.isfinite(alpha_2200)]
        lo2200, hi2200 = np.percentile(finite_2200, [1, 99.5])
        shown = axes[1].imshow(alpha_2200, cmap="viridis", vmin=lo2200, vmax=hi2200)
        axes[1].contour(high_confidence, levels=[0.5], colors="red", linewidths=0.8)
        fig.colorbar(shown, ax=axes[1], shrink=0.75)
    else:
        axes[1].imshow(mask_2200, cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("2200 nm corrected alpha")
    axes[2].imshow(high_confidence, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title(f"cross-band components >= {min_component_pixels} px")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("1600/2200 nm cross-band consensus")
    fig.savefig(output_dir / "cross_band_consensus.png", dpi=180)
    plt.close(fig)
    return {
        "consensus_pixels": int(consensus.sum()),
        "consensus_components": int(component_count),
        "connected_consensus_pixels": int(high_confidence.sum()),
        "connected_consensus_components": int(
            np.sum(sizes[1:] >= min_component_pixels)
        ),
        "one_pixel_gap_merged_regions": used_regions,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_input(
        args.roi_csv,
        args.modtran_csv,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        fwhm_nm=args.fwhm,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        nodata_values=args.nodata,
        require_positive=args.require_positive,
    )
    if prepared.valid_mask.sum() < max(prepared.cube.shape[2] + 5, 30):
        raise ValueError("Too few valid pixels after masking.")
    reference_mask = load_reference_mask(args.reference_2200_mask, prepared.valid_mask.shape)
    reference_alpha = load_reference_alpha(
        args.reference_2200_alpha, prepared.valid_mask.shape
    )

    print(
        f"Input cube={prepared.roi.cube.shape}; selected={len(prepared.wavelengths_nm)} bands "
        f"({prepared.wavelengths_nm[0]:.2f}--{prepared.wavelengths_nm[-1]:.2f} nm); "
        f"valid={prepared.valid_mask.sum()}/{prepared.valid_mask.size}"
    )
    if reference_mask is not None:
        print(f"2200 nm reference candidates: {reference_mask.sum()}")

    results: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, float | int | str | bool]] = []
    for method in METHODS:
        result = run_iterative_method(
            prepared,
            method=method,
            n_iter=args.n_iter,
            nsigma=args.nsigma,
            protect_nsigma=args.protect_nsigma,
            regularization=args.regularization,
            rcond=args.rcond,
            canvas_size=args.canvas_size,
            line_protection_cap=args.line_protection_cap,
            convergence_pixels=args.convergence_pixels,
            verbose=not args.quiet,
        )
        results[method] = result
        save_method_outputs(output_dir, prepared, result)
        metric_rows.append(method_metrics(result, prepared.valid_mask, reference_mask))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "method_comparison.csv", index=False)
    save_comparison_figure(output_dir / "method_comparison.png", results, reference_mask)
    save_profile_figure(output_dir / "directional_profiles.png", results)
    consensus_summary = save_cross_band_consensus(
        output_dir,
        prepared,
        np.asarray(results["qa_median_then_ct_dwt"]["alpha_corrected"]),
        np.asarray(results["qa_median_then_ct_dwt"]["plume_mask"], dtype=bool),
        reference_mask,
        reference_alpha,
    )

    target = pd.DataFrame(
        {
            "wavelength_nm": prepared.wavelengths_nm,
            "uas": prepared.uas,
            "final_background_mean_qa": np.asarray(
                results["qa_median_then_ct_dwt"]["mean_history"][-1]
            ),
        }
    )
    target.to_csv(output_dir / "selected_1600nm_target.csv", index=False)
    np.save(output_dir / "valid_mask.npy", prepared.valid_mask)

    summary = {
        "roi_csv": str(args.roi_csv.resolve()),
        "modtran_csv": str(args.modtran_csv.resolve()),
        "output_dir": str(output_dir),
        "wavelength_window_nm": [args.wl_min, args.wl_max],
        "selected_wavelengths_nm": prepared.wavelengths_nm.tolist(),
        "selected_band_count": len(prepared.wavelengths_nm),
        "valid_pixels": int(prepared.valid_mask.sum()),
        "iterations_requested": args.n_iter,
        "nsigma": args.nsigma,
        "protect_nsigma": args.protect_nsigma,
        "line_protection_cap": args.line_protection_cap,
        "convergence_pixels": args.convergence_pixels,
        "stripe_parameters": {
            "thin_slope": THIN_SLOPE,
            "thin_orthogonal_slope": -THIN_ORTHOGONAL_SLOPE,
            "detected_broad_slope": DETECTED_BROAD_SLOPE,
            "ct_metadata_mirror_slope": CT_METADATA_MIRROR_SLOPE,
            "pdf_dwt_levels": [3, 4, 5],
            "qa_ct_dwt_levels": [2, 3, 4, 5],
        },
        "reference_2200_mask": str(args.reference_2200_mask.resolve())
        if reference_mask is not None
        else None,
        "reference_2200_alpha": str(args.reference_2200_alpha.resolve())
        if reference_alpha is not None
        else None,
        "cross_band_consensus": consensus_summary,
        "metrics": metrics.to_dict(orient="records"),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print("\nMethod comparison:")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")
    return results


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare in-loop stripe removal for 1600 nm Iterative MF."
    )
    parser.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--modtran-csv", type=Path, default=DEFAULT_LUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-2200-mask", type=Path, default=DEFAULT_REFERENCE_2200_MASK)
    parser.add_argument(
        "--reference-2200-alpha", type=Path, default=DEFAULT_REFERENCE_2200_ALPHA
    )
    parser.add_argument("--wl-min", type=float, default=1580.0)
    parser.add_argument("--wl-max", type=float, default=1700.0)
    parser.add_argument("--fwhm", type=float, default=12.5)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--nsigma", type=float, default=3.0)
    parser.add_argument("--protect-nsigma", type=float, default=4.0)
    parser.add_argument(
        "--line-protection-cap",
        type=int,
        default=50,
        help="Do not protect a high-alpha thin-line bin containing at least this many pixels.",
    )
    parser.add_argument(
        "--convergence-pixels",
        type=int,
        default=2,
        help="Stop when at most this many candidate-mask pixels change.",
    )
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--rcond", type=float, default=1e-8)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--nodata", type=parse_float_list, default=(0.0, -9999.0))
    parser.add_argument(
        "--require-positive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, object]]:
    return run_pipeline(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

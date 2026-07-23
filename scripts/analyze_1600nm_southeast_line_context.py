#!/usr/bin/env python3
"""Diagnose the southeast, upper-right-running line in the 1600 nm MF map.

The original 200 x 200 ROI was cut from the full HISUI image around
``center=(1066, 1463)``. This script extracts a wider 400 x 400 context from
``all_map_spectra.csv`` without loading the 2.65 GB CSV in full, creates RGB
images, reruns the QA-guided 1600 nm MF on the wider context, and compares the
line spectrum with nearby parallel controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

import iterative_mf_1600nm as imf
from iterative_mf_1600nm_destriped import (
    PreparedInput,
    run_iterative_method,
)


DEFAULT_MAP_CSV = Path("data/all_map_spectra.csv")
DEFAULT_ROI_CSV = Path("data/all_roi_spectra200x200.csv")
DEFAULT_TARGET_CSV = Path(
    "outputs/iterative_mf_1600nm_destriped/selected_1600nm_target.csv"
)
DEFAULT_ROI_ALPHA = Path(
    "outputs/iterative_mf_1600nm_destriped/qa_median_then_ct_dwt/alpha_corrected.npy"
)
DEFAULT_ROI_MASK = Path(
    "outputs/iterative_mf_1600nm_destriped/qa_median_then_ct_dwt/plume_mask.npy"
)
DEFAULT_OUTPUT = Path(
    "outputs/iterative_mf_1600nm_destriped/southeast_line_context"
)

FULL_HEIGHT = 1894
FULL_WIDTH = 1761
ROI_CENTER_Y = 1066
ROI_CENTER_X = 1463
ROI_HALF_SIZE = 100

RGB_COLUMNS = ("wave_685.00nm", "wave_585.00nm", "wave_485.00nm")


def parse_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return next(csv.reader(stream))


def find_pixel_line(
    mapped: mmap.mmap,
    target_linear_index: int,
    *,
    full_width: int,
) -> tuple[int, int]:
    """Return byte [start, end) for a row-major CSV pixel using binary search."""
    lower = mapped.find(b"\n") + 1
    upper = len(mapped)
    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(100):
        if upper - lower < 2:
            break
        midpoint = (lower + upper) // 2
        start = lower if midpoint <= lower else mapped.find(b"\n", midpoint, upper)
        if start < 0:
            upper = midpoint
            continue
        start += 1
        if start >= upper:
            upper = midpoint
            continue
        end = mapped.find(b"\n", start)
        if end < 0:
            end = len(mapped)
        first_comma = mapped.find(b",", start, end)
        second_comma = mapped.find(b",", first_comma + 1, end)
        linear_index = (
            int(mapped[start:first_comma]) * full_width
            + int(mapped[first_comma + 1 : second_comma])
        )
        if linear_index == target_linear_index:
            return start, end
        state = (lower, upper, start, linear_index)
        if state in seen:
            break
        seen.add(state)
        if linear_index < target_linear_index:
            lower = end + 1
        else:
            upper = start

    # Variable-length zero/nonzero rows can make the byte-position bisection
    # settle just before the target. Finish with a short sequential scan.
    start = lower
    if start > 0 and mapped[start - 1 : start] != b"\n":
        newline = mapped.find(b"\n", start, upper)
        if newline >= 0:
            start = newline + 1
    for _ in range(10_000):
        if start < 0 or start >= len(mapped):
            break
        end = mapped.find(b"\n", start)
        if end < 0:
            end = len(mapped)
        first_comma = mapped.find(b",", start, end)
        second_comma = mapped.find(b",", first_comma + 1, end)
        linear_index = (
            int(mapped[start:first_comma]) * full_width
            + int(mapped[first_comma + 1 : second_comma])
        )
        if linear_index == target_linear_index:
            return start, end
        if linear_index > target_linear_index:
            break
        start = end + 1
    raise RuntimeError(f"Could not locate pixel index {target_linear_index} in map CSV.")


def extract_context_window(
    path: Path,
    *,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    selected_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    header = parse_header(path)
    index_by_name = {name: index for index, name in enumerate(header)}
    missing = [name for name in selected_columns if name not in index_by_name]
    if missing:
        raise ValueError(f"Columns missing from map CSV: {missing}")
    selected_indices = [index_by_name[name] for name in selected_columns]
    height = y_max - y_min
    width = x_max - x_min
    output = np.empty((height, width, len(selected_columns)), dtype=np.float32)

    with path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for local_y, global_y in enumerate(range(y_min, y_max)):
                start, end = find_pixel_line(
                    mapped,
                    global_y * FULL_WIDTH + x_min,
                    full_width=FULL_WIDTH,
                )
                position = start
                for local_x, expected_x in enumerate(range(x_min, x_max)):
                    if local_x > 0:
                        position = end + 1
                        end = mapped.find(b"\n", position)
                        if end < 0:
                            end = len(mapped)
                    parts = mapped[position:end].split(b",")
                    row_value = int(parts[0])
                    column_value = int(parts[1])
                    if row_value != global_y or column_value != expected_x:
                        raise RuntimeError(
                            f"Unexpected CSV coordinate {(row_value, column_value)}; "
                            f"expected {(global_y, expected_x)}."
                        )
                    output[local_y, local_x] = [
                        float(parts[index]) for index in selected_indices
                    ]
        finally:
            mapped.close()
    return output, list(selected_columns)


def percentile_rgb(radiance_rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(radiance_rgb, dtype=float)
    for channel in range(3):
        values = radiance_rgb[:, :, channel][valid_mask]
        lower, upper = np.percentile(values, [2, 98])
        output[:, :, channel] = np.clip(
            (radiance_rgb[:, :, channel] - lower) / max(upper - lower, 1e-12),
            0,
            1,
        )
    output = np.power(output, 0.85)
    output[~valid_mask] = 0
    return output


def load_roi_rgb(roi_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(roi_csv, usecols=["y", "x", *RGB_COLUMNS])
    height = int(frame["y"].max()) + 1
    width = int(frame["x"].max()) + 1
    radiance = np.full((height, width, 3), np.nan, dtype=float)
    radiance[
        frame["y"].to_numpy(dtype=int), frame["x"].to_numpy(dtype=int)
    ] = frame[list(RGB_COLUMNS)].to_numpy(dtype=float)
    valid = np.all(np.isfinite(radiance) & (radiance > 0), axis=2)
    return radiance, percentile_rgb(radiance, valid)


def fit_southeast_line(
    alpha: np.ndarray,
    plume_mask: np.ndarray,
) -> dict[str, float | int]:
    rows, columns = np.indices(alpha.shape)
    region = (rows >= 125) & (columns >= 40) & ~plume_mask
    residual = alpha - ndimage.gaussian_filter(alpha, sigma=8, mode="nearest")
    best: tuple[float, float, int, np.ndarray] | None = None
    for slope in np.linspace(-0.7, 0.0, 281):
        line_ids = np.rint((rows - slope * columns) / 2.0).astype(int)
        for line_id in np.unique(line_ids[region]):
            selected = region & (line_ids == line_id)
            if selected.sum() < 60:
                continue
            selected_columns = columns[selected]
            if selected_columns.max() - selected_columns.min() < 80:
                continue
            score = float(np.median(residual[selected]))
            if best is None or score > best[0]:
                best = (score, float(slope), int(line_id), selected)
    if best is None:
        raise RuntimeError("Could not fit the southeast line.")
    score, initial_slope, line_id, selected = best
    intercept = float(np.median(rows[selected] - initial_slope * columns[selected]))
    near_line = (
        region
        & (
            np.abs(rows - (initial_slope * columns + intercept))
            / math.sqrt(initial_slope**2 + 1)
            <= 2.0
        )
        & (residual > np.percentile(residual[region], 75))
    )
    if near_line.sum() >= 20:
        refined_slope, refined_intercept = np.polyfit(
            columns[near_line], rows[near_line], 1
        )
    else:
        refined_slope, refined_intercept = initial_slope, intercept
    return {
        "initial_slope": initial_slope,
        "initial_intercept": intercept,
        "refined_slope": float(refined_slope),
        "refined_intercept": float(refined_intercept),
        "angle_deg": float(np.degrees(np.arctan(refined_slope))),
        "scan_score": score,
        "fit_pixels": int(near_line.sum()),
    }


def line_and_control_masks(
    shape: tuple[int, int],
    *,
    slope: float,
    intercept: float,
    line_half_width: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(shape)
    signed_distance = (
        rows - (slope * columns + intercept)
    ) / math.sqrt(slope**2 + 1)
    line = np.abs(signed_distance) <= line_half_width
    controls = (np.abs(signed_distance) >= 7.0) & (np.abs(signed_distance) <= 13.0)
    line &= (rows >= 125) & (columns >= 40)
    controls &= (rows >= 125) & (columns >= 40)
    return line, controls


def save_rgb(path: Path, rgb: np.ndarray, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.imshow(rgb)
    axis.set_title(title)
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_context_mf(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    uas: np.ndarray,
    global_y: np.ndarray,
    global_x: np.ndarray,
) -> dict[str, object]:
    valid = np.all(np.isfinite(cube) & (cube > 0), axis=2)
    roi = imf.RoiCube(
        cube=cube,
        wavelengths_nm=wavelengths_nm,
        ys=global_y,
        xs=global_x,
    )
    prepared = PreparedInput(roi, cube, wavelengths_nm, uas, valid)
    return run_iterative_method(
        prepared,
        method="qa_median_then_ct_dwt",
        n_iter=20,
        nsigma=3.0,
        protect_nsigma=4.0,
        regularization=1e-6,
        rcond=1e-8,
        canvas_size=512,
        line_protection_cap=50,
        convergence_pixels=2,
        verbose=False,
    )


def extract_roi_spectral_comparison(
    roi_csv: Path,
    line_mask: np.ndarray,
    control_mask: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(roi_csv)
    wave_columns, wavelengths = imf.get_wave_columns(frame.columns)
    y_values = frame["y"].to_numpy(dtype=int)
    x_values = frame["x"].to_numpy(dtype=int)
    is_line = line_mask[y_values, x_values]
    is_control = control_mask[y_values, x_values]
    line_mean = frame.loc[is_line, wave_columns].to_numpy(dtype=float).mean(axis=0)
    control_mean = frame.loc[is_control, wave_columns].to_numpy(dtype=float).mean(axis=0)
    table = pd.DataFrame(
        {
            "wavelength_nm": wavelengths,
            "line_mean_radiance": line_mean,
            "control_mean_radiance": control_mean,
            "line_minus_control": line_mean - control_mean,
            "line_to_control_ratio": line_mean / control_mean,
        }
    )
    table.to_csv(output_dir / "southeast_line_vs_control_spectrum.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(wavelengths, line_mean, label="line", linewidth=1.2)
    axes[0].plot(wavelengths, control_mean, label="parallel controls", linewidth=1.2)
    axes[0].set_ylabel("mean radiance")
    axes[0].set_title("Southeast line and nearby-control spectra")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(wavelengths, line_mean / control_mean, color="tab:red")
    axes[1].axhline(1, color="black", linewidth=0.8)
    axes[1].axvspan(1580, 1700, color="tab:blue", alpha=0.12, label="1600 nm MF window")
    axes[1].set_xlabel("wavelength [nm]")
    axes[1].set_ylabel("line / control")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "southeast_line_vs_control_spectrum.png", dpi=180)
    plt.close(fig)
    return table


def save_diagnostic_figure(
    path: Path,
    roi_rgb: np.ndarray,
    roi_alpha: np.ndarray,
    context_rgb: np.ndarray,
    context_alpha: np.ndarray,
    line_fit: dict[str, float | int],
    *,
    context_margin: int,
) -> None:
    slope = float(line_fit["refined_slope"])
    intercept = float(line_fit["refined_intercept"])
    roi_x = np.array([40, 199])
    roi_y = slope * roi_x + intercept
    context_x = np.array([0, context_rgb.shape[1] - 1])
    context_intercept = intercept + context_margin - slope * context_margin
    context_y = slope * context_x + context_intercept
    roi_box_x = [context_margin, context_margin + 200, context_margin + 200, context_margin, context_margin]
    roi_box_y = [context_margin, context_margin, context_margin + 200, context_margin + 200, context_margin]

    alpha_values = np.concatenate(
        [roi_alpha[np.isfinite(roi_alpha)], context_alpha[np.isfinite(context_alpha)]]
    )
    lower, upper = np.percentile(alpha_values, [1, 99.5])
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    axes[0, 0].imshow(roi_rgb)
    axes[0, 0].plot(roi_x, roi_y, color="red", linewidth=1.2)
    axes[0, 0].set_title("200 x 200 RGB + fitted line")
    shown = axes[0, 1].imshow(roi_alpha, cmap="viridis", vmin=lower, vmax=upper)
    axes[0, 1].plot(roi_x, roi_y, color="red", linewidth=1.2)
    axes[0, 1].set_title("200 x 200 corrected 1600 nm alpha")
    fig.colorbar(shown, ax=axes[0, 1], shrink=0.75)
    axes[1, 0].imshow(context_rgb)
    axes[1, 0].plot(context_x, context_y, color="red", linewidth=1.0)
    axes[1, 0].plot(roi_box_x, roi_box_y, color="white", linewidth=0.9)
    axes[1, 0].set_title("400 x 400 RGB context; white = original ROI")
    shown = axes[1, 1].imshow(context_alpha, cmap="viridis", vmin=lower, vmax=upper)
    axes[1, 1].plot(context_x, context_y, color="red", linewidth=1.0)
    axes[1, 1].plot(roi_box_x, roi_box_y, color="white", linewidth=0.9)
    axes[1, 1].set_title("400 x 400 corrected 1600 nm alpha")
    fig.colorbar(shown, ax=axes[1, 1], shrink=0.75)
    for axis in axes.ravel():
        axis.set_xlabel("column")
        axis.set_ylabel("row")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_y_min = ROI_CENTER_Y - ROI_HALF_SIZE
    roi_x_min = ROI_CENTER_X - ROI_HALF_SIZE
    context_y_min = roi_y_min - args.context_margin
    context_y_max = roi_y_min + 200 + args.context_margin
    context_x_min = roi_x_min - args.context_margin
    context_x_max = roi_x_min + 200 + args.context_margin

    target = pd.read_csv(args.target_csv)
    wavelengths = target["wavelength_nm"].to_numpy(dtype=float)
    uas = target["uas"].to_numpy(dtype=float)
    wave_columns = [f"wave_{value:.2f}nm" for value in wavelengths]
    selected_columns = [*wave_columns, *RGB_COLUMNS]
    context_values, column_names = extract_context_window(
        args.map_csv,
        y_min=context_y_min,
        y_max=context_y_max,
        x_min=context_x_min,
        x_max=context_x_max,
        selected_columns=selected_columns,
    )
    column_index = {name: index for index, name in enumerate(column_names)}
    context_cube = context_values[:, :, [column_index[name] for name in wave_columns]].astype(float)
    context_rgb_radiance = context_values[
        :, :, [column_index[name] for name in RGB_COLUMNS]
    ].astype(float)
    context_valid = np.all(context_rgb_radiance > 0, axis=2)
    context_rgb = percentile_rgb(context_rgb_radiance, context_valid)

    roi_rgb_radiance, roi_rgb = load_roi_rgb(args.roi_csv)
    roi_alpha = np.load(args.roi_alpha).astype(float)
    roi_plume = np.load(args.roi_mask).astype(bool)
    line_fit = fit_southeast_line(roi_alpha, roi_plume)
    line_mask, control_mask = line_and_control_masks(
        roi_alpha.shape,
        slope=float(line_fit["refined_slope"]),
        intercept=float(line_fit["refined_intercept"]),
    )
    spectrum_table = extract_roi_spectral_comparison(
        args.roi_csv, line_mask, control_mask, output_dir
    )

    context_result = run_context_mf(
        context_cube,
        wavelengths,
        uas,
        np.arange(context_y_min, context_y_max),
        np.arange(context_x_min, context_x_max),
    )
    context_alpha = np.asarray(context_result["alpha_corrected"], dtype=float)
    context_plume = np.asarray(context_result["plume_mask"], dtype=bool)

    margin = args.context_margin
    extracted_roi_cube = context_cube[margin : margin + 200, margin : margin + 200]
    roi_frame = pd.read_csv(args.roi_csv, usecols=wave_columns)
    roi_cube_reference = roi_frame[wave_columns].to_numpy(dtype=float).reshape(200, 200, -1)
    max_roi_difference = float(np.max(np.abs(extracted_roi_cube - roi_cube_reference)))

    np.save(output_dir / "roi_200x200_rgb.npy", roi_rgb)
    np.save(output_dir / "context_400x400_rgb.npy", context_rgb)
    np.save(
        output_dir / "context_400x400_rgb_radiance.npy",
        context_rgb_radiance.astype(np.float32),
    )
    np.save(output_dir / "context_400x400_alpha_corrected.npy", context_alpha)
    np.save(output_dir / "context_400x400_plume_mask.npy", context_plume)
    np.savez_compressed(
        output_dir / "context_400x400_selected_radiance.npz",
        cube=context_cube.astype(np.float32),
        wavelengths_nm=wavelengths,
        y=np.arange(context_y_min, context_y_max),
        x=np.arange(context_x_min, context_x_max),
    )
    save_rgb(output_dir / "roi_200x200_rgb.png", roi_rgb, "HISUI RGB: original 200 x 200 ROI")
    save_rgb(output_dir / "context_400x400_rgb.png", context_rgb, "HISUI RGB: 400 x 400 context")
    save_diagnostic_figure(
        output_dir / "southeast_line_context_diagnostics.png",
        roi_rgb,
        roi_alpha,
        context_rgb,
        context_alpha,
        line_fit,
        context_margin=margin,
    )

    ratio = spectrum_table["line_to_control_ratio"].to_numpy(dtype=float)
    visible = spectrum_table["wavelength_nm"].between(450, 700).to_numpy()
    mf_window = spectrum_table["wavelength_nm"].between(1580, 1700).to_numpy()
    summary: dict[str, object] = {
        "map_csv": str(args.map_csv.resolve()),
        "roi_full_coordinates": {
            "y_min": roi_y_min,
            "y_max_inclusive": roi_y_min + 199,
            "x_min": roi_x_min,
            "x_max_inclusive": roi_x_min + 199,
        },
        "context_full_coordinates": {
            "y_min": context_y_min,
            "y_max_inclusive": context_y_max - 1,
            "x_min": context_x_min,
            "x_max_inclusive": context_x_max - 1,
        },
        "roi_map_csv_max_abs_difference": max_roi_difference,
        "line_fit_roi_coordinates": line_fit,
        "known_artifact_slopes": {
            "qa_thin": 0.9773460526106752,
            "qa_thin_orthogonal": -1.023179044238028,
            "ct_metadata_mirror": 1.2667203308533619,
        },
        "line_pixels_for_spectrum": int(line_mask.sum()),
        "control_pixels_for_spectrum": int(control_mask.sum()),
        "mean_line_control_ratio_visible_450_700nm": float(np.mean(ratio[visible])),
        "mean_line_control_ratio_1580_1700nm": float(np.mean(ratio[mf_window])),
        "context_iterations": len(context_result["plume_history"]),
        "context_convergence_mode": context_result["convergence_mode"],
    }
    with (output_dir / "southeast_line_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-csv", type=Path, default=DEFAULT_MAP_CSV)
    parser.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI_CSV)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--roi-alpha", type=Path, default=DEFAULT_ROI_ALPHA)
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_ROI_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-margin", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

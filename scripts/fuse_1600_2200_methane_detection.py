#!/usr/bin/env python3
"""Fuse corrected 1600 and 2200 nm MF maps for methane-plume detection.

Alpha magnitudes are not added directly. Each map is robustly standardized,
then a high-confidence core requires independent positive evidence in both
bands. A lower threshold grows the core into a plume-extent mask. Negative-tail
and spatial-shift controls quantify how often comparable overlap occurs by
chance in these maps.
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


DEFAULT_ALPHA_1600 = Path(
    "outputs/iterative_mf_1600nm_destriped/qa_median_then_ct_dwt/alpha_corrected.npy"
)
DEFAULT_ALPHA_2200 = Path(
    "outputs/iterative_mf_1600nm_destriped/2200nm_residual_thin_refined/"
    "recommended_alpha_corrected_refined.npy"
)
DEFAULT_RGB = Path(
    "outputs/iterative_mf_1600nm_destriped/southeast_line_context/roi_200x200_rgb.npy"
)
DEFAULT_FALSE_LINE_SUMMARY = Path(
    "outputs/iterative_mf_1600nm_destriped/southeast_line_context/"
    "southeast_line_summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/iterative_mf_1600nm_destriped/dual_band_methane_detection"
)


def robust_zscore(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = image[np.isfinite(image)]
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    return (image - median) / scale, median, scale


def keep_connected_components(
    mask: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())
    keep_ids = np.flatnonzero(sizes >= min_pixels)
    keep_ids = keep_ids[keep_ids > 0]
    return np.isin(labels, keep_ids), labels, sizes


def threshold_control_row(
    z1600: np.ndarray,
    z2200: np.ndarray,
    threshold: float,
    sign: int,
    min_component_pixels: int,
) -> dict[str, float | int | str]:
    raw = (sign * z1600 >= threshold) & (sign * z2200 >= threshold)
    connected, labels, sizes = keep_connected_components(raw, min_component_pixels)
    component_ids = np.unique(labels[connected])
    component_ids = component_ids[component_ids > 0]
    return {
        "threshold": threshold,
        "tail": "positive" if sign > 0 else "negative",
        "raw_pixels": int(raw.sum()),
        "connected_pixels": int(connected.sum()),
        "connected_components": int(len(component_ids)),
        "largest_component_pixels": int(
            max((sizes[index] for index in component_ids), default=0)
        ),
    }


def spatial_shift_null(
    z1600: np.ndarray,
    z2200: np.ndarray,
    *,
    background_correlation: float,
    core_threshold: float,
    joint_threshold: float,
    min_component_pixels: int,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int]] = []
    height, width = z1600.shape
    denominator = math.sqrt(2.0 + 2.0 * background_correlation)
    for iteration in range(1, iterations + 1):
        while True:
            shift_y = int(rng.integers(-height // 2, height // 2 + 1))
            shift_x = int(rng.integers(-width // 2, width // 2 + 1))
            if abs(shift_y) > 10 or abs(shift_x) > 10:
                break
        shifted = np.roll(z2200, shift=(shift_y, shift_x), axis=(0, 1))
        shifted_joint = (z1600 + shifted) / denominator
        raw = (
            (z1600 >= core_threshold)
            & (shifted >= core_threshold)
            & (shifted_joint >= joint_threshold)
        )
        connected, labels, sizes = keep_connected_components(
            raw, min_component_pixels
        )
        component_ids = np.unique(labels[connected])
        component_ids = component_ids[component_ids > 0]
        rows.append(
            {
                "iteration": iteration,
                "shift_y": shift_y,
                "shift_x": shift_x,
                "raw_pixels": int(raw.sum()),
                "connected_pixels": int(connected.sum()),
                "connected_components": int(len(component_ids)),
                "largest_component_pixels": int(
                    max((sizes[index] for index in component_ids), default=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_extent(
    core: np.ndarray,
    z1600: np.ndarray,
    z2200: np.ndarray,
    joint_z: np.ndarray,
    *,
    extent_threshold: float,
    extent_joint_threshold: float,
) -> np.ndarray:
    support = (
        (z1600 >= extent_threshold)
        & (z2200 >= extent_threshold)
        & (joint_z >= extent_joint_threshold)
    )
    support_labels, _ = ndimage.label(
        support, structure=np.ones((3, 3), dtype=int)
    )
    selected_ids = np.unique(support_labels[core])
    selected_ids = selected_ids[selected_ids > 0]
    return np.isin(support_labels, selected_ids)


def build_region_table(
    extent: np.ndarray,
    core: np.ndarray,
    alpha1600: np.ndarray,
    alpha2200: np.ndarray,
    z1600: np.ndarray,
    z2200: np.ndarray,
    joint_z: np.ndarray,
    agreement_z: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    labels, count = ndimage.label(
        extent, structure=np.ones((3, 3), dtype=int)
    )
    rows_out: list[dict[str, float | int | str]] = []
    for region_id in range(1, count + 1):
        selected = labels == region_id
        core_selected = selected & core
        if not core_selected.any():
            continue
        rows, columns = np.nonzero(selected)
        weights = np.maximum(joint_z[selected], 0)
        peak_flat = int(np.argmax(joint_z[selected]))
        peak_row = int(rows[peak_flat])
        peak_col = int(columns[peak_flat])
        correlation = (
            float(np.corrcoef(z1600[selected], z2200[selected])[0, 1])
            if selected.sum() >= 3
            else float("nan")
        )
        confidence = (
            "high"
            if core_selected.sum() >= 5
            and float(np.mean(agreement_z[core_selected])) >= 3.5
            else "moderate"
        )
        rows_out.append(
            {
                "region_id": region_id,
                "methane_confidence": confidence,
                "extent_pixels": int(selected.sum()),
                "core_pixels": int(core_selected.sum()),
                "row_min": int(rows.min()),
                "row_max": int(rows.max()),
                "col_min": int(columns.min()),
                "col_max": int(columns.max()),
                "full_row_min": int(rows.min() + row_offset),
                "full_row_max": int(rows.max() + row_offset),
                "full_col_min": int(columns.min() + col_offset),
                "full_col_max": int(columns.max() + col_offset),
                "weighted_centroid_row": float(np.average(rows, weights=weights)),
                "weighted_centroid_col": float(np.average(columns, weights=weights)),
                "full_weighted_centroid_row": float(
                    np.average(rows, weights=weights) + row_offset
                ),
                "full_weighted_centroid_col": float(
                    np.average(columns, weights=weights) + col_offset
                ),
                "peak_row": peak_row,
                "peak_col": peak_col,
                "full_peak_row": peak_row + row_offset,
                "full_peak_col": peak_col + col_offset,
                "alpha_1600_mean": float(np.mean(alpha1600[selected])),
                "alpha_1600_max": float(np.max(alpha1600[selected])),
                "alpha_2200_mean": float(np.mean(alpha2200[selected])),
                "alpha_2200_max": float(np.max(alpha2200[selected])),
                "z_1600_core_mean": float(np.mean(z1600[core_selected])),
                "z_2200_core_mean": float(np.mean(z2200[core_selected])),
                "agreement_z_core_mean": float(
                    np.mean(agreement_z[core_selected])
                ),
                "joint_z_core_mean": float(np.mean(joint_z[core_selected])),
                "joint_z_max": float(np.max(joint_z[selected])),
                "cross_band_z_correlation_extent": correlation,
            }
        )
    table = pd.DataFrame(rows_out).sort_values(
        ["methane_confidence", "core_pixels"], ascending=[True, False]
    )
    return table, labels


def save_pixel_table(
    path: Path,
    extent: np.ndarray,
    core: np.ndarray,
    labels: np.ndarray,
    alpha1600: np.ndarray,
    alpha2200: np.ndarray,
    z1600: np.ndarray,
    z2200: np.ndarray,
    joint_z: np.ndarray,
    agreement_z: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> None:
    rows, columns = np.nonzero(extent)
    table = pd.DataFrame(
        {
            "region_id": labels[extent],
            "row": rows,
            "col": columns,
            "full_row": rows + row_offset,
            "full_col": columns + col_offset,
            "is_high_confidence_core": core[extent],
            "alpha_1600": alpha1600[extent],
            "alpha_2200": alpha2200[extent],
            "z_1600": z1600[extent],
            "z_2200": z2200[extent],
            "agreement_z": agreement_z[extent],
            "joint_z": joint_z[extent],
        }
    ).sort_values(["region_id", "joint_z"], ascending=[True, False])
    table.to_csv(path, index=False)


def save_figures(
    output_dir: Path,
    alpha1600: np.ndarray,
    alpha2200: np.ndarray,
    z1600: np.ndarray,
    z2200: np.ndarray,
    joint_z: np.ndarray,
    core: np.ndarray,
    extent: np.ndarray,
    region_table: pd.DataFrame,
    rgb: np.ndarray | None,
) -> None:
    z_limit = max(float(np.percentile(z1600, 99.7)), float(np.percentile(z2200, 99.7)), 4)
    classification = np.zeros_like(z1600)
    classification[extent] = 1
    classification[core] = 2
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    panels = [
        (z1600, "1600 nm robust z", "viridis", -2, z_limit),
        (z2200, "2200 nm robust z", "viridis", -2, z_limit),
        (joint_z, "correlation-corrected joint z", "viridis", -2, 8),
    ]
    for axis, (image, title, cmap, lower, upper) in zip(axes[0], panels):
        shown = axis.imshow(image, cmap=cmap, vmin=lower, vmax=upper)
        axis.contour(extent, levels=[0.5], colors="cyan", linewidths=0.8)
        axis.contour(core, levels=[0.5], colors="red", linewidths=0.8)
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.72)

    shown = axes[1, 0].imshow(classification, cmap="viridis", vmin=0, vmax=2)
    axes[1, 0].set_title("classification: cyan extent / red core")
    axes[1, 0].contour(extent, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1, 0].contour(core, levels=[0.5], colors="red", linewidths=0.8)
    fig.colorbar(shown, ax=axes[1, 0], shrink=0.72)
    if rgb is not None:
        axes[1, 1].imshow(rgb)
        axes[1, 1].contour(extent, levels=[0.5], colors="cyan", linewidths=0.9)
        axes[1, 1].contour(core, levels=[0.5], colors="red", linewidths=0.9)
        axes[1, 1].set_title("RGB with dual-band detections")
    else:
        axes[1, 1].axis("off")

    all_points = np.ones(z1600.shape, dtype=bool)
    sample = np.flatnonzero(all_points.ravel())[::8]
    axes[1, 2].scatter(
        z1600.ravel()[sample],
        z2200.ravel()[sample],
        s=3,
        alpha=0.15,
        color="gray",
        label="all pixels (1/8 sample)",
    )
    axes[1, 2].scatter(
        z1600[extent], z2200[extent], s=12, color="cyan", label="extent"
    )
    axes[1, 2].scatter(
        z1600[core], z2200[core], s=14, color="red", label="core"
    )
    axes[1, 2].axvline(3, color="black", linestyle="--", linewidth=0.8)
    axes[1, 2].axhline(3, color="black", linestyle="--", linewidth=0.8)
    axes[1, 2].set_xlim(-4, 9)
    axes[1, 2].set_ylim(-4, 10)
    axes[1, 2].set_xlabel("1600 nm z")
    axes[1, 2].set_ylabel("2200 nm z")
    axes[1, 2].set_title("cross-band agreement")
    axes[1, 2].legend(fontsize=8, frameon=False)

    for axis in axes.ravel()[:5]:
        axis.set_xticks([])
        axis.set_yticks([])
    for _, region in region_table.iterrows():
        row = float(region["weighted_centroid_row"])
        col = float(region["weighted_centroid_col"])
        for axis in axes.ravel()[:5]:
            if axis.axison:
                axis.text(
                    col,
                    row,
                    f"R{int(region['region_id'])}",
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                )
    fig.suptitle("Dual-band methane-plume detection: 1600 + 2200 nm")
    fig.savefig(output_dir / "dual_band_detection_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, image, title in [
        (axes[0], alpha1600, "1600 nm corrected alpha"),
        (axes[1], alpha2200, "2200 nm corrected alpha"),
    ]:
        lower, upper = np.percentile(image[np.isfinite(image)], [1, 99.5])
        shown = axis.imshow(image, cmap="viridis", vmin=lower, vmax=upper)
        axis.contour(extent, levels=[0.5], colors="cyan", linewidths=0.8)
        axis.contour(core, levels=[0.5], colors="red", linewidths=0.8)
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.72)
    if rgb is not None:
        axes[2].imshow(rgb)
        axes[2].contour(extent, levels=[0.5], colors="cyan", linewidths=0.9)
        axes[2].contour(core, levels=[0.5], colors="red", linewidths=0.9)
    axes[2].set_title("RGB; cyan=extent, red=core")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(output_dir / "dual_band_detection_result.png", dpi=180)
    plt.close(fig)


def load_false_line_corridor(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    fit = summary["line_fit_roi_coordinates"]
    slope = float(fit["refined_slope"])
    intercept = float(fit["refined_intercept"])
    rows, columns = np.indices(shape)
    distance = np.abs(rows - (slope * columns + intercept)) / math.sqrt(
        slope**2 + 1
    )
    return distance <= 2.5


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha1600 = np.load(args.alpha_1600).astype(float)
    alpha2200 = np.load(args.alpha_2200).astype(float)
    if alpha1600.shape != alpha2200.shape:
        raise ValueError("1600 and 2200 nm alpha maps must have the same shape.")
    valid = np.isfinite(alpha1600) & np.isfinite(alpha2200)

    z1600, median1600, scale1600 = robust_zscore(alpha1600)
    z2200, median2200, scale2200 = robust_zscore(alpha2200)
    background = valid & (np.abs(z1600) < 3) & (np.abs(z2200) < 3)
    background_correlation = float(
        np.corrcoef(z1600[background], z2200[background])[0, 1]
    )
    joint_z = (z1600 + z2200) / math.sqrt(
        2.0 + 2.0 * background_correlation
    )
    agreement_z = np.minimum(z1600, z2200)

    raw_core = (
        valid
        & (z1600 >= args.core_threshold)
        & (z2200 >= args.core_threshold)
        & (joint_z >= args.joint_threshold)
    )
    core, raw_labels, raw_sizes = keep_connected_components(
        raw_core, args.min_core_component_pixels
    )
    rejected_isolated = raw_core & ~core
    extent = build_extent(
        core,
        z1600,
        z2200,
        joint_z,
        extent_threshold=args.extent_threshold,
        extent_joint_threshold=args.extent_joint_threshold,
    )
    region_table, region_labels = build_region_table(
        extent,
        core,
        alpha1600,
        alpha2200,
        z1600,
        z2200,
        joint_z,
        agreement_z,
        args.roi_row_offset,
        args.roi_col_offset,
    )

    sensitivity_rows: list[dict[str, float | int | str]] = []
    for threshold in (2.5, 3.0, 3.5, 4.0):
        sensitivity_rows.append(
            threshold_control_row(
                z1600,
                z2200,
                threshold,
                1,
                args.min_core_component_pixels,
            )
        )
        sensitivity_rows.append(
            threshold_control_row(
                z1600,
                z2200,
                threshold,
                -1,
                args.min_core_component_pixels,
            )
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "threshold_sensitivity_and_negative_control.csv", index=False
    )

    null_table = spatial_shift_null(
        z1600,
        z2200,
        background_correlation=background_correlation,
        core_threshold=args.core_threshold,
        joint_threshold=args.joint_threshold,
        min_component_pixels=args.min_core_component_pixels,
        iterations=args.null_iterations,
        seed=args.random_seed,
    )
    null_table.to_csv(output_dir / "spatial_shift_null.csv", index=False)
    empirical_p = float(
        (1 + np.sum(null_table["connected_pixels"] >= int(core.sum())))
        / (len(null_table) + 1)
    )

    false_line = load_false_line_corridor(
        args.false_line_summary, alpha1600.shape
    )
    false_line_counts = None
    if false_line is not None:
        false_line_counts = {
            "raw_core_pixels_on_false_surface_line": int(
                np.sum(raw_core & false_line)
            ),
            "connected_core_pixels_on_false_surface_line": int(
                np.sum(core & false_line)
            ),
            "extent_pixels_on_false_surface_line": int(
                np.sum(extent & false_line)
            ),
        }

    np.save(output_dir / "z_1600.npy", z1600)
    np.save(output_dir / "z_2200.npy", z2200)
    np.save(output_dir / "joint_z.npy", joint_z)
    np.save(output_dir / "agreement_z.npy", agreement_z)
    np.save(output_dir / "methane_core_mask.npy", core)
    np.save(output_dir / "methane_extent_mask.npy", extent)
    np.save(output_dir / "rejected_isolated_dual_band_pixels.npy", rejected_isolated)
    region_table.to_csv(output_dir / "methane_regions.csv", index=False)
    save_pixel_table(
        output_dir / "methane_pixels.csv",
        extent,
        core,
        region_labels,
        alpha1600,
        alpha2200,
        z1600,
        z2200,
        joint_z,
        agreement_z,
        args.roi_row_offset,
        args.roi_col_offset,
    )
    rejected_rows, rejected_columns = np.nonzero(rejected_isolated)
    pd.DataFrame(
        {
            "row": rejected_rows,
            "col": rejected_columns,
            "z_1600": z1600[rejected_isolated],
            "z_2200": z2200[rejected_isolated],
            "joint_z": joint_z[rejected_isolated],
        }
    ).to_csv(output_dir / "rejected_isolated_dual_band_pixels.csv", index=False)

    rgb = None
    if args.rgb is not None and args.rgb.exists():
        rgb = np.load(args.rgb).astype(float)
        if rgb.shape[:2] != alpha1600.shape:
            raise ValueError("RGB and alpha-map spatial shapes do not match.")
    save_figures(
        output_dir,
        alpha1600,
        alpha2200,
        z1600,
        z2200,
        joint_z,
        core,
        extent,
        region_table,
        rgb,
    )

    negative_at_core = next(
        row
        for row in sensitivity_rows
        if row["tail"] == "negative" and row["threshold"] == 3.0
    )
    summary: dict[str, object] = {
        "alpha_1600": str(args.alpha_1600.resolve()),
        "alpha_2200": str(args.alpha_2200.resolve()),
        "robust_normalization": {
            "median_1600": median1600,
            "scale_1600": scale1600,
            "median_2200": median2200,
            "scale_2200": scale2200,
            "background_z_correlation": background_correlation,
        },
        "thresholds": {
            "core_each_band_z": args.core_threshold,
            "core_joint_z": args.joint_threshold,
            "minimum_core_component_pixels": args.min_core_component_pixels,
            "extent_each_band_z": args.extent_threshold,
            "extent_joint_z": args.extent_joint_threshold,
        },
        "coordinates": {
            "roi_shape": list(alpha1600.shape),
            "full_map_row_offset": args.roi_row_offset,
            "full_map_col_offset": args.roi_col_offset,
            "full_map_roi_inclusive": {
                "row_min": args.roi_row_offset,
                "row_max": args.roi_row_offset + alpha1600.shape[0] - 1,
                "col_min": args.roi_col_offset,
                "col_max": args.roi_col_offset + alpha1600.shape[1] - 1,
            },
        },
        "result": {
            "raw_dual_band_core_pixels": int(raw_core.sum()),
            "rejected_isolated_pixels": int(rejected_isolated.sum()),
            "connected_core_pixels": int(core.sum()),
            "plume_extent_pixels": int(extent.sum()),
            "methane_regions": int(len(region_table)),
            "high_confidence_regions": int(
                np.sum(region_table["methane_confidence"] == "high")
            ),
        },
        "negative_tail_control_at_3sigma": negative_at_core,
        "spatial_shift_null": {
            "iterations": args.null_iterations,
            "actual_connected_core_pixels": int(core.sum()),
            "null_median_connected_pixels": float(
                null_table["connected_pixels"].median()
            ),
            "null_99th_percentile_connected_pixels": float(
                null_table["connected_pixels"].quantile(0.99)
            ),
            "null_max_connected_pixels": int(
                null_table["connected_pixels"].max()
            ),
            "empirical_p_connected_pixels": empirical_p,
        },
        "known_false_surface_line_check": false_line_counts,
        "interpretation": (
            "High-confidence methane-compatible regions require independent "
            "positive MF evidence in both bands and spatial coherence. This is "
            "not a substitute for external plume/wind validation."
        ),
    }
    with (output_dir / "dual_band_detection_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(region_table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-1600", type=Path, default=DEFAULT_ALPHA_1600)
    parser.add_argument("--alpha-2200", type=Path, default=DEFAULT_ALPHA_2200)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument(
        "--false-line-summary", type=Path, default=DEFAULT_FALSE_LINE_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--core-threshold", type=float, default=3.0)
    parser.add_argument("--joint-threshold", type=float, default=4.0)
    parser.add_argument("--min-core-component-pixels", type=int, default=2)
    parser.add_argument("--extent-threshold", type=float, default=2.0)
    parser.add_argument("--extent-joint-threshold", type=float, default=2.7)
    parser.add_argument("--roi-row-offset", type=int, default=966)
    parser.add_argument("--roi-col-offset", type=int, default=1363)
    parser.add_argument("--null-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=743)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create image-based CH4 candidate regions and validate them in the other band.

Candidate masks are selected from only one methane absorption window after
masked Gaussian smoothing and tile-wise robust standardization.  The other
window is used only for validation.  Local random spatial shifts of the fixed
mask provide an empirical null distribution that preserves the validator
image's spatial texture better than an independent-pixel Gaussian model.

The reported shift p-values are a research diagnostic, not an exact theorem:
their interpretation requires approximate local translation exchangeability.
Positive and reverse-CH4 directions are processed identically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from iterative_mf_1600nm import get_wave_columns


DIRECTIONS = (
    ("weak_to_strong", "weak_z", "strong_z", "weak_alpha", "strong_alpha"),
    ("strong_to_weak", "strong_z", "weak_z", "strong_alpha", "weak_alpha"),
)


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    return median, max(scale, 1e-6)


def masked_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    numerator = ndimage.gaussian_filter(np.where(valid, values, 0.0), sigma=sigma)
    denominator = ndimage.gaussian_filter(valid.astype(float), sigma=sigma)
    output = numerator / np.maximum(denominator, 1e-12)
    output[denominator < 0.5] = np.nan
    return output


def tile_robust_z(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    tile_size: int,
) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    height, width = values.shape
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            section = np.s_[
                y0 : min(y0 + tile_size, height),
                x0 : min(x0 + tile_size, width),
            ]
            keep = valid[section] & np.isfinite(values[section])
            if int(keep.sum()) < 50:
                continue
            median, scale = robust_scale(values[section][keep])
            tile_output = output[section]
            tile_output[keep] = (values[section][keep] - median) / scale
    return output


def grow_regions(
    selection_z: np.ndarray,
    *,
    core_threshold: float,
    extent_threshold: float,
    minimum_pixels: int,
    maximum_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    core = np.nan_to_num(selection_z, nan=-np.inf) >= core_threshold
    extent = np.nan_to_num(selection_z, nan=-np.inf) >= extent_threshold
    grown = ndimage.binary_propagation(core, mask=extent)
    labels, count = ndimage.label(grown, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep_ids = np.flatnonzero(
        (sizes >= minimum_pixels) & (sizes <= maximum_pixels)
    )
    keep_ids = keep_ids[keep_ids > 0]
    if len(keep_ids) == 0:
        return np.zeros_like(labels), np.empty(0, dtype=int)
    cleaned = np.where(np.isin(labels, keep_ids), labels, 0)
    return cleaned, keep_ids


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def local_shift_distribution(
    validator: np.ndarray,
    edge_valid: np.ndarray,
    row: np.ndarray,
    column: np.ndarray,
    weights: np.ndarray,
    *,
    count: int,
    radius: int,
    minimum_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = validator.shape
    output: list[np.ndarray] = []
    accepted = 0
    maximum_attempts = max(count * 50, 1_000)
    attempts = 0
    weight_sum = max(float(np.sum(weights)), 1e-12)
    while accepted < count and attempts < maximum_attempts:
        draw_count = min(max(4 * (count - accepted), 256), 4_096)
        attempts += draw_count
        dy = rng.integers(-radius, radius + 1, size=draw_count)
        dx = rng.integers(-radius, radius + 1, size=draw_count)
        separated = (np.abs(dy) >= minimum_shift) | (np.abs(dx) >= minimum_shift)
        inside = (
            (row.min() + dy >= 0)
            & (row.max() + dy < height)
            & (column.min() + dx >= 0)
            & (column.max() + dx < width)
        )
        keep = separated & inside
        if not np.any(keep):
            continue
        dy = dy[keep]
        dx = dx[keep]
        shifted_row = row[None, :] + dy[:, None]
        shifted_column = column[None, :] + dx[:, None]
        valid_shift = np.all(edge_valid[shifted_row, shifted_column], axis=1)
        if not np.any(valid_shift):
            continue
        shifted_values = validator[
            shifted_row[valid_shift], shifted_column[valid_shift]
        ]
        finite_shift = np.all(np.isfinite(shifted_values), axis=1)
        if not np.any(finite_shift):
            continue
        scores = shifted_values[finite_shift] @ weights / weight_sum
        remaining = count - accepted
        scores = scores[:remaining]
        output.append(np.asarray(scores, dtype=float))
        accepted += len(scores)
    if not output:
        return np.empty(0, dtype=float)
    return np.concatenate(output)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return adjusted
    values = p_values[valid]
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = q_ranked
    adjusted[valid] = restored
    return adjusted


def load_rgb(
    scene_csv: Path,
    shape: tuple[int, int],
    y_origin: int,
    x_origin: int,
    *,
    chunksize: int,
) -> tuple[np.ndarray, list[float]]:
    header = pd.read_csv(scene_csv, nrows=0)
    wave_columns, wavelengths = get_wave_columns(header.columns)
    targets = np.asarray([655.0, 555.0, 485.0])
    indices = [int(np.argmin(np.abs(wavelengths - target))) for target in targets]
    columns = [wave_columns[index] for index in indices]
    rgb_wavelengths = [float(wavelengths[index]) for index in indices]
    rgb = np.full((*shape, 3), np.nan, dtype=np.float32)
    for chunk in pd.read_csv(
        scene_csv, usecols=["y", "x", *columns], chunksize=chunksize
    ):
        row = chunk["y"].to_numpy(int) - y_origin
        column = chunk["x"].to_numpy(int) - x_origin
        inside = (
            (row >= 0)
            & (row < shape[0])
            & (column >= 0)
            & (column < shape[1])
        )
        rgb[row[inside], column[inside]] = chunk.loc[inside, columns].to_numpy(
            dtype=np.float32
        )
    return rgb, rgb_wavelengths


def render_rgb(rgb: np.ndarray) -> np.ndarray:
    output = np.ones(rgb.shape, dtype=float)
    finite_pixel = np.all(np.isfinite(rgb), axis=2)
    for band in range(3):
        values = rgb[:, :, band]
        finite = values[finite_pixel]
        low, high = np.quantile(finite, [0.02, 0.98])
        output[:, :, band] = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    output[~finite_pixel] = 1.0
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-csv", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--core-threshold", type=float, default=3.5)
    parser.add_argument("--extent-threshold", type=float, default=2.0)
    parser.add_argument("--minimum-pixels", type=int, default=5)
    parser.add_argument("--maximum-pixels", type=int, default=500)
    parser.add_argument(
        "--maximum-regions-per-direction",
        type=int,
        default=None,
        help=(
            "Keep only this many strongest selector-only regions for each "
            "direction and sign before validation."
        ),
    )
    parser.add_argument("--edge-exclusion", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--shift-count", type=int, default=4_999)
    parser.add_argument("--shift-radius", type=int, default=None)
    parser.add_argument("--minimum-shift", type=int, default=10)
    parser.add_argument("--fdr", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=743)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis_dir = args.analysis_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else analysis_dir / "spatial_region_validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "analysis_summary.json").open(encoding="utf-8") as stream:
        analysis_summary = json.load(stream)
    stored = np.load(analysis_dir / "score_maps.npz")
    valid = stored["valid"].astype(bool)
    y_origin = int(stored["y_min"])
    x_origin = int(stored["x_min"])
    tile_size = args.tile_size
    if tile_size is None:
        tile_size = int(analysis_summary["model"].get("local_tile_size", 0))
        if tile_size <= 0:
            tile_size = int(max(valid.shape))
    shift_radius = args.shift_radius or tile_size
    distance_to_edge = ndimage.distance_transform_edt(valid)
    edge_valid = valid & (distance_to_edge >= args.edge_exclusion)
    rng = np.random.default_rng(args.seed)

    selection_maps: dict[tuple[str, int], np.ndarray] = {}
    label_maps: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict[str, float | int | str | bool]] = []
    for direction, selector_key, validator_key, selector_alpha_key, validator_alpha_key in DIRECTIONS:
        for sign, sign_name in ((1, "positive_ch4"), (-1, "reverse_ch4")):
            selector = sign * stored[selector_key].astype(float)
            validator = sign * stored[validator_key].astype(float)
            smoothed = masked_gaussian(selector, valid, args.sigma)
            selection_z = tile_robust_z(
                smoothed, edge_valid, tile_size=tile_size
            )
            labels, component_ids = grow_regions(
                selection_z,
                core_threshold=args.core_threshold,
                extent_threshold=args.extent_threshold,
                minimum_pixels=args.minimum_pixels,
                maximum_pixels=args.maximum_pixels,
            )
            if (
                args.maximum_regions_per_direction is not None
                and len(component_ids) > args.maximum_regions_per_direction
            ):
                peaks = np.asarray(
                    ndimage.maximum(
                        np.nan_to_num(selection_z, nan=-np.inf),
                        labels=labels,
                        index=component_ids,
                    ),
                    dtype=float,
                )
                strongest = np.argsort(peaks)[
                    -args.maximum_regions_per_direction :
                ]
                component_ids = component_ids[strongest]
                labels = np.where(np.isin(labels, component_ids), labels, 0)
            selection_maps[(direction, sign)] = selection_z
            label_maps[(direction, sign)] = labels
            object_slices = ndimage.find_objects(labels)
            for component_id in component_ids:
                section = object_slices[int(component_id) - 1]
                if section is None:
                    continue
                local_row, local_column = np.nonzero(
                    labels[section] == component_id
                )
                row = local_row + int(section[0].start)
                column = local_column + int(section[1].start)
                weights = np.maximum(
                    selection_z[row, column] - args.extent_threshold, 0.05
                )
                observed = weighted_mean(validator[row, column], weights)
                null = local_shift_distribution(
                    validator,
                    edge_valid,
                    row,
                    column,
                    weights,
                    count=args.shift_count,
                    radius=shift_radius,
                    minimum_shift=args.minimum_shift,
                    rng=rng,
                )
                if len(null) < min(100, args.shift_count):
                    p_value = float("nan")
                    validation_z = float("nan")
                    null_median = float("nan")
                    null_scale = float("nan")
                else:
                    p_value = float((1 + np.sum(null >= observed)) / (1 + len(null)))
                    null_median, null_scale = robust_scale(null)
                    validation_z = float((observed - null_median) / null_scale)
                selector_ppm = sign * stored[selector_alpha_key][row, column].astype(float)
                validator_ppm = sign * stored[validator_alpha_key][row, column].astype(float)
                combined_ppm = sign * stored["combined_alpha"][row, column].astype(float)
                selector_ppm_mean = weighted_mean(selector_ppm, weights)
                validator_ppm_mean = weighted_mean(validator_ppm, weights)
                denominator = max(
                    abs(selector_ppm_mean) + abs(validator_ppm_mean), 1e-12
                )
                rows.append(
                    {
                        "direction": direction,
                        "sign": sign_name,
                        "component_id": int(component_id),
                        "pixels": int(len(row)),
                        "y_min": int(row.min() + y_origin),
                        "y_max": int(row.max() + y_origin),
                        "x_min": int(column.min() + x_origin),
                        "x_max": int(column.max() + x_origin),
                        "selection_peak_robust_z": float(
                            np.nanmax(selection_z[row, column])
                        ),
                        "validation_weighted_score": observed,
                        "validation_local_shift_z": validation_z,
                        "local_shift_p_value": p_value,
                        "local_shift_count_used": int(len(null)),
                        "local_shift_null_median": null_median,
                        "local_shift_null_robust_scale": null_scale,
                        "selector_ppm_mean": selector_ppm_mean,
                        "validator_ppm_mean": validator_ppm_mean,
                        "combined_ppm_mean": weighted_mean(combined_ppm, weights),
                        "relative_ppm_disagreement": float(
                            abs(selector_ppm_mean - validator_ppm_mean) / denominator
                        ),
                        "minimum_distance_to_scene_edge_pixels": float(
                            np.min(distance_to_edge[row, column])
                        ),
                        "max_pixel_log_e": float(
                            np.nanmax(
                                stored[
                                    "log_e_average"
                                    if sign > 0
                                    else "log_e_negative_control"
                                ][row, column]
                            )
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    if len(table):
        table["bh_q_value"] = np.nan
        table["accepted_at_fdr"] = False
        for sign_name in ("positive_ch4", "reverse_ch4"):
            selected = table["sign"] == sign_name
            q_values = benjamini_hochberg(
                table.loc[selected, "local_shift_p_value"].to_numpy(float)
            )
            table.loc[selected, "bh_q_value"] = q_values
            table.loc[selected, "accepted_at_fdr"] = q_values <= args.fdr
        table = table.sort_values(
            ["sign", "local_shift_p_value", "validation_local_shift_z"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    table.to_csv(output_dir / "region_validation.csv", index=False)

    positive_p_map = np.full(valid.shape, np.nan, dtype=float)
    reverse_p_map = np.full(valid.shape, np.nan, dtype=float)
    if len(table):
        for record in table.itertuples(index=False):
            sign = 1 if record.sign == "positive_ch4" else -1
            labels = label_maps[(record.direction, sign)]
            destination = positive_p_map if sign > 0 else reverse_p_map
            score = -math.log10(max(float(record.local_shift_p_value), 1e-12))
            row0 = int(record.y_min - y_origin)
            row1 = int(record.y_max - y_origin + 1)
            column0 = int(record.x_min - x_origin)
            column1 = int(record.x_max - x_origin + 1)
            section = np.s_[row0:row1, column0:column1]
            region = labels[section] == record.component_id
            local_destination = destination[section]
            local_destination[region] = np.fmax(local_destination[region], score)

    weak_selection = selection_maps[("weak_to_strong", 1)]
    strong_selection = selection_maps[("strong_to_weak", 1)]
    conservative_ppm = np.maximum(
        np.minimum(
            stored["weak_alpha"].astype(float),
            stored["strong_alpha"].astype(float),
        ),
        0.0,
    )
    conservative_ppm[~valid] = np.nan
    rgb, rgb_wavelengths = load_rgb(
        args.scene_csv.resolve(),
        valid.shape,
        y_origin,
        x_origin,
        chunksize=args.chunksize,
    )
    rendered_rgb = render_rgb(rgb)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rendered_rgb)
    axes[0, 0].set_title(f"RGB context {rgb_wavelengths} nm")

    weak_mask = label_maps[("weak_to_strong", 1)] > 0
    strong_mask = label_maps[("strong_to_weak", 1)] > 0
    if weak_mask.any():
        axes[0, 0].contour(weak_mask, [0.5], colors="cyan", linewidths=0.8)
    if strong_mask.any():
        axes[0, 0].contour(strong_mask, [0.5], colors="yellow", linewidths=0.8)
    axes[0, 0].text(
        0.01,
        0.01,
        "cyan: weak-selected; yellow: strong-selected",
        transform=axes[0, 0].transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
    )

    panels = [
        (axes[0, 1], weak_selection, "Weak-band spatial selection robust z", "viridis"),
        (axes[0, 2], strong_selection, "Strong-band spatial selection robust z", "viridis"),
        (axes[1, 0], conservative_ppm, "Concordant MODTRAN-equivalent ppm", "magma"),
        (axes[1, 1], positive_p_map, "Positive regions: -log10(local-shift p)", "plasma"),
        (axes[1, 2], reverse_p_map, "Reverse-CH4 regions: -log10(local-shift p)", "plasma"),
    ]
    for axis, image, title, cmap in panels:
        finite = image[np.isfinite(image)]
        if len(finite):
            low = 0.0 if "ppm" in title or "log10" in title else float(np.quantile(finite, 0.02))
            high = float(np.quantile(finite, 0.995))
            shown = axis.imshow(image, cmap=cmap, vmin=low, vmax=max(high, low + 1e-6))
            fig.colorbar(shown, ax=axis, shrink=0.78)
        else:
            axis.imshow(np.zeros(valid.shape), cmap=cmap)
        axis.set_title(title)
    for axis in axes.ravel():
        axis.set_axis_off()
    fig.savefig(output_dir / "observed_spatial_candidate_map.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "region_maps.npz",
        valid=valid,
        weak_selection_z=weak_selection.astype(np.float32),
        strong_selection_z=strong_selection.astype(np.float32),
        conservative_ppm=conservative_ppm.astype(np.float32),
        positive_minus_log10_p=positive_p_map.astype(np.float32),
        reverse_minus_log10_p=reverse_p_map.astype(np.float32),
        weak_selected_labels=label_maps[("weak_to_strong", 1)].astype(np.int32),
        strong_selected_labels=label_maps[("strong_to_weak", 1)].astype(np.int32),
    )
    positive_count = int(np.sum(table["sign"] == "positive_ch4")) if len(table) else 0
    reverse_count = int(np.sum(table["sign"] == "reverse_ch4")) if len(table) else 0
    positive_accepted = int(
        np.sum((table["sign"] == "positive_ch4") & table["accepted_at_fdr"])
    ) if len(table) else 0
    reverse_accepted = int(
        np.sum((table["sign"] == "reverse_ch4") & table["accepted_at_fdr"])
    ) if len(table) else 0
    summary: dict[str, object] = {
        "inputs": {
            "scene_csv": str(args.scene_csv.resolve()),
            "analysis_dir": str(analysis_dir),
            "ppm_interpretation": (
                "MODTRAN LUT column headers are treated as ppm; alpha maps are "
                "MODTRAN-equivalent ppm enhancements relative to the fitted background."
            ),
        },
        "selection": {
            "sigma_pixels": args.sigma,
            "core_threshold_robust_z": args.core_threshold,
            "extent_threshold_robust_z": args.extent_threshold,
            "minimum_pixels": args.minimum_pixels,
            "maximum_pixels": args.maximum_pixels,
            "maximum_regions_per_direction": args.maximum_regions_per_direction,
            "tile_size": tile_size,
            "edge_exclusion_pixels": args.edge_exclusion,
        },
        "validation": {
            "method": "local spatial shifts of the fixed region in the other absorption-band map",
            "shift_count_requested": args.shift_count,
            "shift_radius_pixels": shift_radius,
            "minimum_shift_pixels": args.minimum_shift,
            "fdr": args.fdr,
            "validity_note": (
                "Empirical diagnostic requiring approximate local translation exchangeability; "
                "not an exact p-value guarantee in a heterogeneous scene."
            ),
        },
        "results": {
            "positive_region_hypotheses": positive_count,
            "reverse_region_hypotheses": reverse_count,
            "positive_regions_bh_accepted": positive_accepted,
            "reverse_regions_bh_accepted": reverse_accepted,
            "minimum_positive_p": (
                float(table.loc[table["sign"] == "positive_ch4", "local_shift_p_value"].min())
                if positive_count
                else None
            ),
            "minimum_reverse_p": (
                float(table.loc[table["sign"] == "reverse_ch4", "local_shift_p_value"].min())
                if reverse_count
                else None
            ),
        },
        "outputs": {
            "region_table": str(output_dir / "region_validation.csv"),
            "map_figure": str(output_dir / "observed_spatial_candidate_map.png"),
            "map_arrays": str(output_dir / "region_maps.npz"),
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

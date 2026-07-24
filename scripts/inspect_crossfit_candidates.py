#!/usr/bin/env python3
"""Inspect cross-fit discovery components with RGB context and local spectra."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from iterative_mf_1600nm import get_wave_columns
from physics_aware_crossfit_mf import make_continuum_transform


@dataclass
class Crop:
    component_id: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    rgb: np.ndarray
    spectra: np.ndarray


def robust_rgb(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=float)
    for channel in range(3):
        band = values[:, :, channel]
        finite = band[np.isfinite(band)]
        low, high = np.quantile(finite, [0.02, 0.98])
        output[:, :, channel] = np.clip((band - low) / max(high - low, 1e-12), 0, 1)
    output[~np.all(np.isfinite(values), axis=2)] = 1.0
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-csv", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--margin", type=int, default=35)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else analysis_dir / "candidate_inspection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stored = np.load(analysis_dir / "score_maps.npz")
    valid = stored["valid"].astype(bool)
    discoveries = stored["e_bh_discovery"].astype(bool)
    y_origin = int(stored["y_min"])
    x_origin = int(stored["x_min"])
    labels, count = ndimage.label(discoveries, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())
    component_ids = np.arange(1, count + 1)
    component_ids = component_ids[np.argsort(sizes[1:])[::-1]]
    component_ids = component_ids[: args.max_components]

    target = pd.read_csv(analysis_dir / "hisui_target_spectrum.csv")
    target_waves = target["wavelength_nm"].to_numpy(float)
    target_vector = target["log_radiance_target_per_concentration"].to_numpy(float)
    weak_indices = np.flatnonzero(target["band_group"].to_numpy() == "weak")
    strong_indices = np.flatnonzero(target["band_group"].to_numpy() == "strong")
    transform, _, _ = make_continuum_transform(
        target_waves, weak_indices, strong_indices, degree=1
    )
    header = pd.read_csv(args.scene_csv, nrows=0)
    wave_columns, waves = get_wave_columns(header.columns)
    selected_columns = [wave_columns[int(np.argmin(np.abs(waves - value)))] for value in target_waves]
    rgb_targets = np.asarray([655.0, 555.0, 485.0])
    rgb_columns = [wave_columns[int(np.argmin(np.abs(waves - value)))] for value in rgb_targets]
    rgb_waves = [float(waves[int(np.argmin(np.abs(waves - value)))]) for value in rgb_targets]

    crops: list[Crop] = []
    rows_out: list[dict[str, float | int]] = []
    height, width = valid.shape
    for component_id in component_ids:
        rr, cc = np.nonzero(labels == component_id)
        r0 = max(int(rr.min()) - args.margin, 0)
        r1 = min(int(rr.max()) + args.margin + 1, height)
        c0 = max(int(cc.min()) - args.margin, 0)
        c1 = min(int(cc.max()) + args.margin + 1, width)
        crops.append(
            Crop(
                component_id=int(component_id),
                row_min=r0,
                row_max=r1,
                col_min=c0,
                col_max=c1,
                rgb=np.full((r1 - r0, c1 - c0, 3), np.nan, dtype=float),
                spectra=np.full(
                    (r1 - r0, c1 - c0, len(selected_columns)), np.nan, dtype=float
                ),
            )
        )
        region = labels == component_id
        rows_out.append(
            {
                "component_id": int(component_id),
                "pixels": int(region.sum()),
                "y_min": int(rr.min() + y_origin),
                "y_max": int(rr.max() + y_origin),
                "x_min": int(cc.min() + x_origin),
                "x_max": int(cc.max() + x_origin),
                "max_log_e": float(np.nanmax(stored["log_e_average"][region])),
                "max_combined_z": float(np.nanmax(stored["combined_z"][region])),
                "mean_weak_z": float(np.nanmean(stored["weak_z"][region])),
                "mean_strong_z": float(np.nanmean(stored["strong_z"][region])),
                "max_reverse_ch4_log_e": float(
                    np.nanmax(stored["log_e_negative_control"][region])
                ),
            }
        )

    component_columns = [
        "component_id",
        "pixels",
        "y_min",
        "y_max",
        "x_min",
        "x_max",
        "max_log_e",
        "max_combined_z",
        "mean_weak_z",
        "mean_strong_z",
        "max_reverse_ch4_log_e",
    ]
    if not crops:
        pd.DataFrame(columns=component_columns).to_csv(
            output_dir / "components.csv", index=False
        )
        print("No e-BH discovery components to inspect.")
        return

    usecols = ["y", "x", *dict.fromkeys([*rgb_columns, *selected_columns])]
    for chunk in pd.read_csv(args.scene_csv, usecols=usecols, chunksize=args.chunksize):
        y = chunk["y"].to_numpy(int)
        x = chunk["x"].to_numpy(int)
        for crop in crops:
            selected = (
                (y >= crop.row_min + y_origin)
                & (y < crop.row_max + y_origin)
                & (x >= crop.col_min + x_origin)
                & (x < crop.col_max + x_origin)
            )
            if not np.any(selected):
                continue
            rr = y[selected] - y_origin - crop.row_min
            cc = x[selected] - x_origin - crop.col_min
            crop.rgb[rr, cc] = chunk.loc[selected, rgb_columns].to_numpy(float)
            crop.spectra[rr, cc] = chunk.loc[selected, selected_columns].to_numpy(float)

    component_table = pd.DataFrame(rows_out, columns=component_columns).sort_values(
        ["max_log_e", "pixels"], ascending=False
    )
    component_table.to_csv(output_dir / "components.csv", index=False)

    for crop in crops:
        region_full = labels == crop.component_id
        region = region_full[crop.row_min : crop.row_max, crop.col_min : crop.col_max]
        valid_crop = valid[crop.row_min : crop.row_max, crop.col_min : crop.col_max]
        dilated = ndimage.binary_dilation(region, iterations=8)
        local_background = valid_crop & ~dilated & np.all(crop.spectra > 0, axis=2)
        region_valid = region & np.all(crop.spectra > 0, axis=2)
        candidate_log = np.mean(np.log(crop.spectra[region_valid]), axis=0)
        background_log = np.median(np.log(crop.spectra[local_background]), axis=0)
        residual = candidate_log - background_log
        residual_feature = transform @ residual
        target_feature = transform @ target_vector
        amplitude = max(
            float(residual_feature @ target_feature / (target_feature @ target_feature)),
            0.0,
        )
        residual_continuum_removed = transform.T @ residual_feature
        fitted = transform.T @ (amplitude * target_feature)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        axes[0].imshow(robust_rgb(crop.rgb))
        axes[0].contour(region, levels=[0.5], colors="cyan", linewidths=1.1)
        axes[0].set_title(f"RGB {rgb_waves} nm")
        log_e_crop = stored["log_e_average"][
            crop.row_min : crop.row_max, crop.col_min : crop.col_max
        ].astype(float)
        finite = log_e_crop[np.isfinite(log_e_crop)]
        shown = axes[1].imshow(
            log_e_crop,
            cmap="magma",
            vmin=float(np.quantile(finite, 0.05)),
            vmax=float(np.quantile(finite, 0.995)),
        )
        axes[1].contour(region, levels=[0.5], colors="cyan", linewidths=1.1)
        axes[1].set_title("cross-fit log e")
        fig.colorbar(shown, ax=axes[1], shrink=0.75)
        axes[2].plot(
            target_waves,
            residual_continuum_removed,
            "o-",
            label="component-local residual after linear-continuum removal",
        )
        axes[2].plot(target_waves, fitted, "--", label=f"fitted CH4 target, alpha={amplitude:.2f}")
        axes[2].axvspan(1750, 2200, color="white", alpha=0.9)
        axes[2].axhline(0, color="black", linewidth=0.6)
        axes[2].set_xlabel("Wavelength [nm]")
        axes[2].set_ylabel("log-radiance difference")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.25)
        for axis in axes[:2]:
            axis.set_axis_off()
        row = component_table.loc[component_table.component_id == crop.component_id].iloc[0]
        fig.suptitle(
            f"Component {crop.component_id}: y={row.y_min:.0f}..{row.y_max:.0f}, "
            f"x={row.x_min:.0f}..{row.x_max:.0f}, pixels={row.pixels:.0f}"
        )
        fig.savefig(output_dir / f"component_{crop.component_id:02d}.png", dpi=180)
        plt.close(fig)

    print(component_table.to_string(index=False))


if __name__ == "__main__":
    main()

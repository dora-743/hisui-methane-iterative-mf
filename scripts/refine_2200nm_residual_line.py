#!/usr/bin/env python3
"""Remove the remaining lower-left thin line from the corrected 2200 nm MF map.

The current QA-guided 2200 nm result uses a two-pixel line bin before CT-DWT.
At the lower-left image edge, a one-pixel core remains because the original
high-alpha line was included in the protection mask and because the line
amplitude is diluted in the wider bin. This script adds a final, protected,
one-pixel-bin median cleanup after CT-DWT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from directional_destriping import (
    directional_profile,
    median_fixed_slope_destripe,
    robust_std,
)


QA_DM_1600_THIN_SLOPE = 0.9773460526106752
DETECTED_BROAD_SLOPE = 1.257172298918948
CT_METADATA_MIRROR_SLOPE = 1.2667203308533619

DEFAULT_QA_OUTPUT = Path("data/2200nm/qa_guided_outputs")
DEFAULT_MF_OUTPUT = Path("data/2200nm/mf_outputs")
DEFAULT_OUTPUT = Path(
    "outputs/iterative_mf_1600nm_destriped/2200nm_residual_thin_refined"
)


def local_line_score(
    image: np.ndarray,
    protected_mask: np.ndarray,
    *,
    slope_min: float = 0.94,
    slope_max: float = 1.01,
) -> float:
    """Maximum positive line-median residual in the lower-left diagnostic crop."""
    rows, columns = np.indices(image.shape)
    crop = (rows >= 105) & (columns <= 115) & ~protected_mask
    residual = image - ndimage.gaussian_filter(image, sigma=8, mode="nearest")
    maxima: list[float] = []
    for slope in np.linspace(slope_min, slope_max, 141):
        line_ids = np.rint(rows - slope * columns).astype(int)
        line_values: list[float] = []
        for line_id in np.unique(line_ids[crop]):
            selected = crop & (line_ids == line_id)
            if selected.sum() >= 20:
                line_values.append(float(np.median(residual[selected])))
        if line_values:
            maxima.append(max(line_values))
    return max(maxima) if maxima else float("nan")


def collect_metrics(
    name: str,
    image: np.ndarray,
    protected_mask: np.ndarray,
) -> dict[str, float | str]:
    rows, columns = np.indices(image.shape)
    lower_left = (rows >= 105) & (columns <= 115) & ~protected_mask

    def profile_scale(slope: float, width: float) -> float:
        _, profile = directional_profile(
            image,
            slope=slope,
            bin_width=width,
            protected_mask=protected_mask,
        )
        return robust_std(profile)

    return {
        "method": name,
        "robust_std_non_plume": robust_std(image, ~protected_mask),
        "robust_std_lower_left": robust_std(image, lower_left),
        "lower_left_max_positive_line_residual": local_line_score(
            image, protected_mask
        ),
        "thin_profile_rstd_bin_1px": profile_scale(QA_DM_1600_THIN_SLOPE, 1.0),
        "thin_profile_rstd_bin_2px": profile_scale(QA_DM_1600_THIN_SLOPE, 2.0),
        "broad_profile_rstd": profile_scale(DETECTED_BROAD_SLOPE, 18.0),
        "ct_profile_rstd": profile_scale(CT_METADATA_MIRROR_SLOPE, 18.0),
        "plume_mean": float(np.mean(image[protected_mask])),
        "plume_p95": float(np.percentile(image[protected_mask], 95)),
        "plume_max": float(np.max(image[protected_mask])),
    }


def save_comparison(
    path: Path,
    raw: np.ndarray,
    current: np.ndarray,
    refined: np.ndarray,
    stripe: np.ndarray,
    protected_mask: np.ndarray,
) -> None:
    values = raw[np.isfinite(raw)]
    value_min, value_max = np.percentile(values, [1, 99.5])
    stripe_limit = max(float(np.percentile(np.abs(stripe), 99.5)), 1e-12)
    crop = np.s_[100:200, 0:120]
    panels = [
        (raw, "raw 2200 nm MF", "viridis", value_min, value_max),
        (current, "current QA-guided", "viridis", value_min, value_max),
        (refined, "refined: final 1 px median", "viridis", value_min, value_max),
        (current[crop], "current lower-left crop", "viridis", value_min, value_max),
        (refined[crop], "refined lower-left crop", "viridis", value_min, value_max),
        (stripe[crop], "additional removed thin line", "coolwarm", -stripe_limit, stripe_limit),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for index, (axis, (image, title, cmap, lower, upper)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        shown = axis.imshow(image, cmap=cmap, vmin=lower, vmax=upper)
        if index < 3:
            axis.contour(protected_mask, levels=[0.5], colors="magenta", linewidths=0.45)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if index in (2, 5):
            fig.colorbar(shown, ax=axis, shrink=0.75)
    fig.suptitle("2200 nm residual lower-left thin-line cleanup")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_profile_plot(
    path: Path,
    current: np.ndarray,
    refined: np.ndarray,
    protected_mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, image in [("current", current), ("refined", refined)]:
        for axis, width in zip(axes, (1.0, 2.0)):
            x_values, profile = directional_profile(
                image,
                slope=QA_DM_1600_THIN_SLOPE,
                bin_width=width,
                protected_mask=protected_mask,
            )
            axis.plot(x_values, profile, label=name, linewidth=1.1)
            axis.set_title(f"QA thin direction, bin={width:g} px")
            axis.set_xlabel("y - 0.977346 x")
            axis.set_ylabel("median high-pass MF")
            axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> np.ndarray:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.qa_output_dir / "raw_mf.npy").astype(float)
    current = np.load(
        args.qa_output_dir / "recommended_alpha_corrected.npy"
    ).astype(float)
    protected = np.load(
        args.mf_output_dir
        / "median_thin_eachiter_then_broad_median_plume_mask.npy"
    ).astype(bool)
    if raw.shape != current.shape or raw.shape != protected.shape:
        raise ValueError("Raw, corrected, and protected-mask shapes do not match.")

    refined, stripe, line_rows = median_fixed_slope_destripe(
        current,
        slope=args.thin_slope,
        line_bin_width=args.line_bin_width,
        min_pixels_per_line=5,
        direction_key="y_minus_x",
        valid_mask=np.isfinite(current),
        exclude_mask=ndimage.binary_dilation(protected, iterations=1),
        operation_name="2200nm_post_ct_dwt_residual_thin_median",
    )
    metrics = pd.DataFrame(
        [
            collect_metrics("current_qa_guided", current, protected),
            collect_metrics("refined_final_1px_median", refined, protected),
        ]
    )
    metrics.to_csv(output_dir / "refinement_metrics.csv", index=False)
    pd.DataFrame(line_rows).to_csv(
        output_dir / "residual_thin_line_offsets.csv", index=False
    )
    np.save(output_dir / "recommended_alpha_corrected_refined.npy", refined)
    np.save(output_dir / "residual_thin_stripe.npy", stripe)
    np.save(output_dir / "protected_plume_mask.npy", protected)
    save_comparison(
        output_dir / "refinement_comparison.png",
        raw,
        current,
        refined,
        stripe,
        protected,
    )
    save_profile_plot(
        output_dir / "refinement_thin_profiles.png", current, refined, protected
    )

    current_row = metrics.iloc[0]
    refined_row = metrics.iloc[1]
    summary = {
        "input_current_alpha": str(
            (args.qa_output_dir / "recommended_alpha_corrected.npy").resolve()
        ),
        "output_refined_alpha": str(
            (output_dir / "recommended_alpha_corrected_refined.npy").resolve()
        ),
        "thin_slope": args.thin_slope,
        "line_bin_width": args.line_bin_width,
        "lower_left_line_residual_reduction_fraction": 1.0
        - float(refined_row["lower_left_max_positive_line_residual"])
        / float(current_row["lower_left_max_positive_line_residual"]),
        "non_plume_robust_std_reduction_fraction": 1.0
        - float(refined_row["robust_std_non_plume"])
        / float(current_row["robust_std_non_plume"]),
        "plume_p95_change_fraction": float(refined_row["plume_p95"])
        / float(current_row["plume_p95"])
        - 1.0,
        "plume_max_change_fraction": float(refined_row["plume_max"])
        / float(current_row["plume_max"])
        - 1.0,
    }
    with (output_dir / "refinement_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(metrics.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return refined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-output-dir", type=Path, default=DEFAULT_QA_OUTPUT)
    parser.add_argument("--mf-output-dir", type=Path, default=DEFAULT_MF_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thin-slope", type=float, default=QA_DM_1600_THIN_SLOPE)
    parser.add_argument("--line-bin-width", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> np.ndarray:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the fixed Keystone R2 region between ROI and full-scene runs.

This is a sensitivity audit, not a new detector.  It puts the score maps from
the existing 200 x 200 analysis and the corresponding crop of the full-scene
analysis on the same pixel grid, then evaluates the historical R2 masks without
selecting a new location from either result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


METRICS = ("weak_z", "strong_z", "dual_min_z", "log_e_average")
METRIC_LABELS = {
    "weak_z": "1.65 um MF z",
    "strong_z": "2.3 um MF z",
    "dual_min_z": "dual-band min z",
    "log_e_average": "bidirectional log e",
}


def _load_score_maps(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = [key for key in (*METRICS, "valid") if key not in archive]
        if missing:
            raise KeyError(f"{path} is missing arrays: {', '.join(missing)}")
        return {key: np.asarray(archive[key]) for key in (*METRICS, "valid")}


def _load_strict_screen_maps(path: Path) -> dict[str, np.ndarray]:
    """Load final-screen local scores under the historical context key names."""

    archive_path = path / "score_maps.npz" if path.is_dir() else path
    key_map = {
        "weak_z": "weak_local_z",
        "strong_z": "strong_local_z",
        "dual_min_z": "dual_local_z",
        "log_e_average": "log_e_average",
        "valid": "valid",
    }
    with np.load(archive_path) as archive:
        missing = [source for source in key_map.values() if source not in archive]
        if missing:
            raise KeyError(
                f"{archive_path} is missing strict-screen arrays: {', '.join(missing)}"
            )
        return {
            target: np.asarray(archive[source])
            for target, source in key_map.items()
        }


def _exclusive_bounds(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("bounds must be y0,y1,x0,x1")
    y0, y1, x0, x1 = values
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise argparse.ArgumentTypeError("bounds must be non-empty and non-negative")
    return values


def summarize_fixed_regions(
    roi_maps: dict[str, np.ndarray],
    full_crop: dict[str, np.ndarray],
    regions: dict[str, tuple[int, int, int, int]],
    strict_crop: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return preregistered-region summaries and whole-crop agreement metrics."""

    rows: list[dict[str, object]] = []
    contexts = [("roi_run", roi_maps), ("full_scene_run", full_crop)]
    if strict_crop is not None:
        contexts.append(("qa_strict_final_screen", strict_crop))
    reference_shape = np.asarray(roi_maps["valid"]).shape
    if len(reference_shape) != 2:
        raise ValueError("ROI valid map must be two-dimensional")
    for context, maps in contexts:
        for metric in ("valid", *METRICS):
            if metric not in maps:
                raise KeyError(f"{context} is missing map: {metric}")
            if np.asarray(maps[metric]).shape != reference_shape:
                raise ValueError(
                    f"{context} {metric} shape does not match ROI grid "
                    f"{reference_shape}"
                )
    for region_name, (y0, y1, x0, x1) in regions.items():
        height, width = reference_shape
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
            raise ValueError(f"{region_name} bounds must be non-empty and non-negative")
        if y1 > height or x1 > width:
            raise ValueError(
                f"{region_name} bounds {(y0, y1, x0, x1)} fall outside "
                f"ROI shape {reference_shape}"
            )
        for context, maps in contexts:
            valid = maps["valid"][y0:y1, x0:x1].astype(bool)
            for metric in METRICS:
                values = maps[metric][y0:y1, x0:x1]
                use = valid & np.isfinite(values)
                selected = values[use]
                rows.append(
                    {
                        "region": region_name,
                        "context": context,
                        "metric": metric,
                        "valid_pixels": int(use.sum()),
                        "mean": float(np.mean(selected)) if len(selected) else np.nan,
                        "median": float(np.median(selected)) if len(selected) else np.nan,
                        "maximum": float(np.max(selected)) if len(selected) else np.nan,
                    }
                )

    agreement_rows: list[dict[str, object]] = []
    common_valid = roi_maps["valid"].astype(bool) & full_crop["valid"].astype(bool)
    for metric in METRICS:
        first = roi_maps[metric]
        second = full_crop[metric]
        use = common_valid & np.isfinite(first) & np.isfinite(second)
        if use.sum() >= 2:
            correlation = float(np.corrcoef(first[use], second[use])[0, 1])
            median_abs = float(np.median(np.abs(second[use] - first[use])))
        else:
            correlation = np.nan
            median_abs = np.nan
        agreement_rows.append(
            {
                "metric": metric,
                "common_valid_pixels": int(use.sum()),
                "pearson_r": correlation,
                "median_absolute_difference": median_abs,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(agreement_rows)


def _draw_fixed_annotations(
    axis: plt.Axes,
    extent: tuple[int, int, int, int],
    site_y: float,
    site_x: float,
) -> None:
    y0, y1, x0, x1 = extent
    axis.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="cyan",
            linewidth=1.4,
        )
    )
    axis.plot(site_x, site_y, marker="+", color="lime", markersize=10, markeredgewidth=1.8)


def save_comparison_figure(
    path: Path,
    roi_maps: dict[str, np.ndarray],
    full_crop: dict[str, np.ndarray],
    *,
    extent: tuple[int, int, int, int],
    site_y: float,
    site_x: float,
    strict_crop: dict[str, np.ndarray] | None = None,
) -> None:
    contexts = [
        ("200 x 200 ROI model", roi_maps),
        ("Full-scene model, same crop", full_crop),
    ]
    if strict_crop is not None:
        contexts.append(("QA-strict final screen, same crop", strict_crop))
    fig, axes = plt.subplots(
        len(contexts),
        4,
        figsize=(17, 4.2 * len(contexts)),
        constrained_layout=True,
        squeeze=False,
    )
    for column, metric in enumerate(METRICS):
        finite = np.concatenate(
            [maps[metric][np.isfinite(maps[metric])] for _, maps in contexts]
        )
        if metric == "log_e_average":
            lower, upper = np.quantile(finite, [0.01, 0.999])
        else:
            lower, upper = -3.0, 6.0
        for row, (context, maps) in enumerate(contexts):
            image = np.where(maps["valid"], maps[metric], np.nan)
            shown = axes[row, column].imshow(
                image, cmap="coolwarm", vmin=float(lower), vmax=float(upper)
            )
            _draw_fixed_annotations(axes[row, column], extent, site_y, site_x)
            axes[row, column].set_title(f"{context}\n{METRIC_LABELS[metric]}")
            axes[row, column].set_axis_off()
            fig.colorbar(shown, ax=axes[row, column], shrink=0.68)
    fig.suptitle(
        "Keystone R2 sensitivity to analysis context\n"
        "cyan: historical R2 extent; green +: TCEQ general facility location",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    roi_dir = args.roi_analysis_dir.expanduser().resolve()
    full_dir = args.full_analysis_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_maps = _load_score_maps(roi_dir / "score_maps.npz")
    full_maps = _load_score_maps(full_dir / "score_maps.npz")
    if roi_maps["valid"].ndim != 2:
        raise ValueError("ROI valid map must be two-dimensional")
    height, width = roi_maps["valid"].shape
    y0, x0 = args.full_y0, args.full_x0
    if y0 < 0 or x0 < 0 or y0 + height > full_maps["valid"].shape[0] or x0 + width > full_maps["valid"].shape[1]:
        raise ValueError("ROI crop falls outside the full-scene score maps")
    full_crop = {
        key: values[y0 : y0 + height, x0 : x0 + width]
        for key, values in full_maps.items()
    }
    strict_crop = None
    strict_input = None
    if args.strict_scene_output is not None:
        strict_input = args.strict_scene_output.expanduser().resolve()
        strict_maps = _load_strict_screen_maps(strict_input)
        if strict_maps["valid"].shape != full_maps["valid"].shape:
            raise ValueError(
                "Strict-screen score maps do not match the historical full-scene grid"
            )
        strict_crop = {
            key: values[y0 : y0 + height, x0 : x0 + width]
            for key, values in strict_maps.items()
        }
    regions = {
        "historical_r2_core": args.r2_core,
        "historical_r2_extent": args.r2_extent,
    }
    region_summary, agreement = summarize_fixed_regions(
        roi_maps, full_crop, regions, strict_crop
    )
    region_summary.to_csv(output_dir / "fixed_r2_context_summary.csv", index=False)
    agreement.to_csv(output_dir / "roi_full_context_agreement.csv", index=False)
    save_comparison_figure(
        output_dir / "r2_analysis_context_comparison.png",
        roi_maps,
        full_crop,
        extent=args.r2_extent,
        site_y=args.site_y,
        site_x=args.site_x,
        strict_crop=strict_crop,
    )

    roi_settings = json.loads((roi_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    full_settings = json.loads((full_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    summary = {
        "inputs": {
            "roi_analysis_dir": str(roi_dir),
            "full_analysis_dir": str(full_dir),
            "strict_scene_output": str(strict_input) if strict_input else None,
            "full_scene_crop_y0_x0_height_width": [y0, x0, height, width],
        },
        "fixed_regions_exclusive_y0_y1_x0_x1": {
            key: list(value) for key, value in regions.items()
        },
        "tceq_general_location_roi_y_x": [args.site_y, args.site_x],
        "roi_model": roi_settings.get("model", {}),
        "full_scene_model": full_settings.get("model", {}),
        "interpretation": (
            "Sensitivity audit on a location fixed by the historical R2 result. "
            "The runs differ in nuisance-model spatial context and some fold settings, "
            "so score changes quantify pipeline-context dependence but cannot be "
            "attributed to a single setting. When provided, the QA-strict final screen "
            "also differs by detector masking, directional-profile correction, and "
            "local standardization."
        ),
        "outputs": {
            "fixed_region_summary": str(output_dir / "fixed_r2_context_summary.csv"),
            "agreement": str(output_dir / "roi_full_context_agreement.csv"),
            "figure": str(output_dir / "r2_analysis_context_comparison.png"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-analysis-dir", type=Path, required=True)
    parser.add_argument("--full-analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--strict-scene-output",
        type=Path,
        help="Final screen scene directory or score_maps.npz on the same full-scene grid",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-y0", type=int, default=966)
    parser.add_argument("--full-x0", type=int, default=1363)
    parser.add_argument("--r2-core", type=_exclusive_bounds, default=(97, 102, 96, 109))
    parser.add_argument("--r2-extent", type=_exclusive_bounds, default=(96, 107, 95, 112))
    parser.add_argument("--site-y", type=float, default=109.78)
    parser.add_argument("--site-x", type=float, default=107.05)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

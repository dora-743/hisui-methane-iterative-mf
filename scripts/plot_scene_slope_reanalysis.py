#!/usr/bin/env python3
"""Plot the audit figures for the per-scene broad-stripe DWT reanalysis."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "joint": "#007C91",
    "weak": "#E07A3F",
    "strong": "#6554C0",
    "raw": "#A7B0B7",
    "selected": "#C83737",
    "positive": "#168AAD",
    "reverse": "#9B5DE5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-batch", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def scene_parts(product_id: str) -> tuple[str, str]:
    match = re.search(r"_(N\d+W\d+)_(\d{8})\d{6}_", product_id)
    if not match:
        return product_id, "unknown"
    tile, date = match.groups()
    return tile, f"{date[:4]}-{date[4:6]}-{date[6:]}"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_selected_slopes(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    reference_angle_deg: float,
    thin_angle_deg: float,
    thin_exclusion_half_width_deg: float,
) -> None:
    entries = []
    for row in rows:
        tile, date = scene_parts(row["product_id"])
        entries.append(
            {
                "date": date,
                "tile": tile,
                "label": f"{date}  {tile}",
                "joint": float(row["selected_angle_deg"]),
                "weak": float(row["weak_best_angle_deg"]),
                "strong": float(row["strong_best_angle_deg"]),
                "prominence": float(row["selected_peak_prominence"]),
                "second": optional_float(row.get("second_peak_prominence")),
                "robust_z": optional_float(row.get("selected_peak_robust_z")),
                "status": row.get("slope_status", "unrated"),
            }
        )
    entries.sort(key=lambda item: (item["date"], item["tile"]))

    y = np.arange(len(entries))
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    ax.axvspan(
        thin_angle_deg - thin_exclusion_half_width_deg,
        thin_angle_deg + thin_exclusion_half_width_deg,
        color="#F2C14E",
        alpha=0.16,
        label="Thin-stripe exclusion",
        zorder=0,
    )
    ax.axvline(reference_angle_deg, color="#555555", linestyle="--", linewidth=1.1,
               label=f"Earlier reference scene ({reference_angle_deg:.1f}°)")
    ax.scatter([item["weak"] for item in entries], y - 0.16, s=48, marker="^",
               color=COLORS["weak"], label="1600 nm individual optimum", zorder=3)
    ax.scatter([item["strong"] for item in entries], y + 0.16, s=48, marker="s",
               color=COLORS["strong"], label="2200 nm individual optimum", zorder=3)
    ax.scatter([item["joint"] for item in entries], y, s=75, marker="o",
               color=COLORS["joint"], edgecolor="white", linewidth=0.8,
               label="Selected joint scene angle", zorder=4)
    for yi, item in zip(y, entries):
        offset = 1.8 if item["joint"] >= 0 else -1.8
        align = "left" if offset > 0 else "right"
        ratio = item["prominence"] / item["second"] if item["second"] > 0 else np.nan
        ratio_label = f"{ratio:.2f}" if np.isfinite(ratio) else "n/a"
        robust_z_label = (
            f"{item['robust_z']:.1f}" if np.isfinite(item["robust_z"]) else "n/a"
        )
        status_label = "" if item["status"] == "supported" else f"  [{item['status']}]"
        ax.text(item["joint"] + offset, yi, f'{item["joint"]:+.1f}°  Zq={robust_z_label}  P1/P2={ratio_label}{status_label}',
                va="center", ha=align, fontsize=8.4, color="#263238")

    ax.axvline(0, color="#263238", linewidth=0.8)
    ax.set_xlim(-85, 85)
    ax.set_xticks(np.arange(-80, 81, 20))
    ax.set_yticks(y, [item["label"] for item in entries])
    ax.invert_yaxis()
    ax.set_xlabel("Stripe angle, θ = arctan(slope) [degrees]")
    ax.set_title("Broad-stripe direction is estimated independently for every HISUI scene")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=3, frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_score_curves(
    search_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    output_path: Path,
    *,
    thin_angle_deg: float,
    thin_exclusion_half_width_deg: float,
    trend_window_deg: float,
) -> None:
    selected = {row["product_id"]: float(row["selected_angle_deg"]) for row in selected_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in search_rows:
        grouped[row["product_id"]].append(row)

    products = sorted(grouped, key=lambda pid: (scene_parts(pid)[1], scene_parts(pid)[0]))
    columns = 2
    panel_rows = max(1, int(np.ceil(len(products) / columns)))
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(12.3, max(4.2, 3.15 * panel_rows)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, product_id in zip(axes_flat, products):
        rows = sorted(grouped[product_id], key=lambda row: float(row["angle_deg"]))
        angle = np.array([float(row["angle_deg"]) for row in rows])
        score = np.array([optional_float(row.get("joint_fractional_gain")) for row in rows])
        trend = np.array([optional_float(row.get("joint_score_trend")) for row in rows])
        tile, date = scene_parts(product_id)

        for branch in (angle < 0, angle > 0):
            ax.plot(angle[branch], score[branch] * 100, color=COLORS["raw"], linewidth=0.75,
                    alpha=0.72)
            ax.plot(angle[branch], trend[branch] * 100, color=COLORS["joint"], linewidth=1.8)
        chosen = selected[product_id]
        chosen_index = int(np.argmin(np.abs(angle - chosen)))
        ax.axvline(chosen, color=COLORS["selected"], linestyle="--", linewidth=1.1)
        ax.scatter([chosen], [score[chosen_index] * 100], color=COLORS["selected"], s=28, zorder=4)
        ax.axvspan(
            thin_angle_deg - thin_exclusion_half_width_deg,
            thin_angle_deg + thin_exclusion_half_width_deg,
            color="#F2C14E",
            alpha=0.14,
        )
        ax.text(0.02, 0.93, f"selected {chosen:+.1f}°", transform=ax.transAxes,
                ha="left", va="top", color=COLORS["selected"], fontsize=8.5)
        ax.set_title(f"{date}  {tile}", loc="left")
        ax.set_ylabel("RStd reduction [%]")
        ax.set_xlim(-82, 82)
        ax.axhline(0, color="#263238", linewidth=0.65)

    for ax in axes_flat[len(products):]:
        ax.axis("off")
    for ax in axes_flat:
        if ax.axison:
            ax.set_xlabel("Signed candidate angle [degrees]")
    fig.suptitle(
        "Per-scene broad-stripe angle search\n"
        f"gray: raw joint score; teal: {trend_window_deg:g}° rolling-median trend; red: selected peak",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=210)
    plt.close(fig)


def metric_pair(summary: dict[str, Any], metric: str) -> tuple[int, int]:
    aggregate = summary["aggregate"]
    fields = {
        "All retained regions": ("positive_region_count", "reverse_region_count"),
        "Review shortlist": ("positive_review_shortlist_count", "reverse_review_shortlist_count"),
        "Conservative single-band": (
            "positive_conservative_single_window_count",
            "reverse_conservative_single_window_count",
        ),
        "Strict dual-band": ("positive_strict_dual_region_count", "reverse_strict_dual_region_count"),
    }
    positive_key, reverse_key = fields[metric]
    return int(aggregate[positive_key]), int(aggregate[reverse_key])


def plot_candidate_balance(summaries: list[dict[str, Any]], output_path: Path) -> None:
    methods = ["Profile only", "Fixed 51.5° DWT", "Per-scene slope DWT"]
    metrics = ["All retained regions", "Review shortlist", "Conservative single-band", "Strict dual-band"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.34
    for ax, metric in zip(axes.ravel(), metrics):
        pairs = [metric_pair(summary, metric) for summary in summaries]
        positive = np.array([pair[0] for pair in pairs])
        reverse = np.array([pair[1] for pair in pairs])
        pos_bars = ax.bar(x - width / 2, positive, width, color=COLORS["positive"], label="Positive")
        rev_bars = ax.bar(x + width / 2, reverse, width, color=COLORS["reverse"], label="Reverse control")
        ax.set_title(metric, loc="left")
        ax.set_xticks(x, methods, rotation=12, ha="right")
        ax.set_ylabel("Region count")
        for xi, pos, rev in zip(x, positive, reverse):
            ratio = pos / rev if rev else np.inf
            ymax = max(pos, rev)
            ax.text(xi, ymax * 1.045 if ymax else 0.2, f"P/R={ratio:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="#263238")
        ax.bar_label(pos_bars, padding=2, fontsize=8)
        ax.bar_label(rev_bars, padding=2, fontsize=8)
        ax.margins(y=0.18)

    axes[0, 0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.suptitle("Positive vs reverse-control candidate balance", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = read_csv(args.adaptive_batch / "posthoc_pdf_dwt_scene_slopes.csv")
    search_rows = read_csv(args.adaptive_batch / "posthoc_pdf_dwt_slope_search.csv")
    batch_summary = read_json(args.adaptive_batch / "batch_summary.json")
    analysis_config = batch_summary["analysis_config"]
    search_config = analysis_config["posthoc_broad_slope_search"]
    reference_angle = float(
        analysis_config["posthoc_reference_broad_rotation_degrees"]
    )
    thin_angle = float(np.degrees(np.arctan(analysis_config["posthoc_thin_slope"])))
    thin_exclusion_half_width = float(search_config["excluded_half_width_deg"])
    summaries = [read_json(path) for path in (args.profile_summary, args.fixed_summary, args.adaptive_summary)]

    plot_selected_slopes(
        selected_rows,
        args.output_dir / "scene_slope_estimates.png",
        reference_angle_deg=reference_angle,
        thin_angle_deg=thin_angle,
        thin_exclusion_half_width_deg=thin_exclusion_half_width,
    )
    plot_score_curves(
        search_rows,
        selected_rows,
        args.output_dir / "scene_slope_score_curves.png",
        thin_angle_deg=thin_angle,
        thin_exclusion_half_width_deg=thin_exclusion_half_width,
        trend_window_deg=float(search_config["trend_window_deg"]),
    )
    plot_candidate_balance(summaries, args.output_dir / "scene_adaptive_candidate_balance.png")


if __name__ == "__main__":
    main()

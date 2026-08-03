#!/usr/bin/env python3
"""Compare profile-only and PDF-DWT single-band review results.

The script keeps the detection rule fixed and produces two diagnostics:

* positive methane-like versus exactly sign-reversed candidate counts, and
* a site-centred, same-colour-scale comparison of the 1600 and 2200 nm maps.

It is a sensitivity comparison, not a methane attribution or an emission-rate
retrieval.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from summarize_single_band_candidates import (
    _load_known_site,
    _load_sources,
    _map_to_pixel_center,
    _strip_geometry,
    _tail_mosaic,
)


ANALYSIS_VERSION = "2026-08-03-pdf-dwt-compare-v2"
COMPARISON_METRICS = (
    ("all_or_regions", "All OR regions", "positive_region_count", "reverse_region_count"),
    (
        "review_shortlist",
        "Review shortlist",
        "positive_review_shortlist_count",
        "reverse_review_shortlist_count",
    ),
    (
        "conservative_single_window",
        "Conservative single-window",
        "positive_conservative_single_window_count",
        "reverse_conservative_single_window_count",
    ),
    (
        "strict_dual_regions",
        "Strict-dual regions",
        "positive_strict_dual_region_count",
        "reverse_strict_dual_region_count",
    ),
)
OUTPUT_NAMES = (
    "pdf_dwt_aggregate_comparison.csv",
    "pdf_dwt_reanalysis_comparison.json",
    "pdf_dwt_candidate_balance.png",
    "pdf_dwt_site_window_comparison.png",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_aggregate_rows(
    profile_aggregate: Mapping[str, Any],
    pdf_aggregate: Mapping[str, Any],
    *,
    pdf_method_label: str = "PDF-DWT then median",
) -> list[dict[str, Any]]:
    """Return matched positive/reverse rows for the two correction methods."""
    rows: list[dict[str, Any]] = []
    for method_key, method_label, aggregate in (
        ("profile_only", "Profile-only", profile_aggregate),
        ("pdf_dwt_then_median", pdf_method_label, pdf_aggregate),
    ):
        for metric_key, metric_label, positive_key, reverse_key in COMPARISON_METRICS:
            positive = int(aggregate[positive_key])
            reverse = int(aggregate[reverse_key])
            rows.append(
                {
                    "method": method_key,
                    "method_label": method_label,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "positive_count": positive,
                    "reverse_count": reverse,
                    "positive_to_reverse_ratio": (
                        float(positive / reverse) if reverse else None
                    ),
                }
            )
    return rows


def _profile_diagnostics(posthoc: Mapping[str, Any]) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    requested_rows: list[Mapping[str, Any]] = []
    supported_scene_count = 0
    for scene in posthoc.get("scene_diagnostics", []):
        if scene.get("broad_slope_estimation", {}).get("slope_status") in (
            None,
            "supported",
        ):
            supported_scene_count += 1
        for band in ("weak", "strong"):
            comparison = scene[band]["comparison"]
            source_broad = float(comparison["source_broad_profile_rstd"])
            pdf_broad = float(comparison["pdf_broad_profile_rstd"])
            source_thin = float(comparison["source_thin_profile_rstd"])
            pdf_thin = float(comparison["pdf_thin_profile_rstd"])
            broad_change = (
                pdf_broad / source_broad - 1.0
                if np.isfinite([source_broad, pdf_broad]).all()
                and source_broad > 0
                else None
            )
            thin_change = (
                pdf_thin / source_thin - 1.0
                if np.isfinite([source_thin, pdf_thin]).all() and source_thin > 0
                else None
            )
            correlation = comparison.get("valid_local_correlation")
            changes.append(
                {
                    "product_id": scene["product_id"],
                    "band": band,
                    "broad_relative_change": broad_change,
                    "thin_relative_change": thin_change,
                    "valid_local_correlation": (
                        float(correlation)
                        if correlation is not None
                        and np.isfinite(float(correlation))
                        else None
                    ),
                }
            )
            requested_rows.extend(
                row for row in scene[band]["dwt"] if bool(row["requested"])
            )
    broad_changes = np.asarray(
        [
            row["broad_relative_change"]
            for row in changes
            if row["broad_relative_change"] is not None
        ],
        dtype=float,
    )
    thin_changes = np.asarray(
        [
            row["thin_relative_change"]
            for row in changes
            if row["thin_relative_change"] is not None
        ],
        dtype=float,
    )
    if broad_changes.size == 0 or thin_changes.size == 0:
        raise ValueError("posthoc summary lacks finite profile rows")
    coefficient_counts = [
        int(row["estimate_coefficient_count"]) for row in requested_rows
    ]
    return {
        "scene_band_count": len(changes),
        "supported_scene_count": supported_scene_count,
        "broad_profile_improved_count": int(np.count_nonzero(broad_changes < 0)),
        "broad_profile_worsened_count": int(np.count_nonzero(broad_changes > 0)),
        "broad_profile_median_relative_change": float(np.median(broad_changes)),
        "thin_profile_improved_count": int(np.count_nonzero(thin_changes < 0)),
        "thin_profile_worsened_count": int(np.count_nonzero(thin_changes > 0)),
        "thin_profile_median_relative_change": float(np.median(thin_changes)),
        "requested_dwt_operation_count": len(requested_rows),
        "filtered_dwt_operation_count": int(
            sum(bool(row["filtered"]) for row in requested_rows)
        ),
        "minimum_estimation_coefficient_count": (
            int(min(coefficient_counts)) if coefficient_counts else None
        ),
        "maximum_estimation_coefficient_count": (
            int(max(coefficient_counts)) if coefficient_counts else None
        ),
        "minimum_estimation_support_fraction": sorted(
            {
                float(row["minimum_estimation_support_fraction"])
                for row in requested_rows
            }
        ),
        "rows": changes,
    }


def _select_site_audit(
    summary: Mapping[str, Any], acquisition_minute: str, site_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summary.get("known_site_audits", [])
        if row.get("acquisition_utc_minute") == acquisition_minute
        and row.get("site_id") == site_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one known-site audit for {site_id} at {acquisition_minute}"
        )
    return dict(matches[0])


def _load_site_crop(
    batch_dir: Path,
    *,
    acquisition_minute: str,
    site: Mapping[str, Any],
    threshold: float,
    minimum_pixels: int,
    site_radius_pixels: int,
    crop_half_size: int,
) -> dict[str, Any]:
    sources, _skipped, _provenance = _load_sources(batch_dir)
    members = [
        source
        for source in sources
        if source.acquisition_minute == acquisition_minute
    ]
    if not members:
        raise ValueError(f"no usable products for {acquisition_minute}")
    del sources
    gc.collect()
    geometry = _strip_geometry(
        acquisition_minute, sorted(members, key=lambda value: value.product_id)
    )
    if geometry.georef.epsg != int(site["epsg"]):
        raise ValueError("site and strip use different EPSG codes")
    mosaic = _tail_mosaic(
        geometry, threshold=threshold, minimum_pixels=minimum_pixels, sign=1
    )
    site_easting = float(site["easting_m"])
    site_northing = float(site["northing_m"])
    row_float, column_float = _map_to_pixel_center(
        geometry.georef, site_easting, site_northing
    )
    center_y, center_x = int(round(row_float)), int(round(column_float))
    y0 = max(0, center_y - crop_half_size)
    y1 = min(geometry.shape[0], center_y + crop_half_size + 1)
    x0 = max(0, center_x - crop_half_size)
    x1 = min(geometry.shape[1], center_x + crop_half_size + 1)
    valid = geometry.valid[y0:y1, x0:x1]
    weak = np.where(valid, mosaic.band1600[y0:y1, x0:x1], np.nan)
    strong = np.where(valid, mosaic.band2200[y0:y1, x0:x1], np.nan)

    transform = geometry.georef.geotransform
    if not math.isclose(transform[2], 0.0, abs_tol=1.0e-9) or not math.isclose(
        transform[4], 0.0, abs_tol=1.0e-9
    ):
        raise ValueError("site comparison currently requires a north-up L1G grid")
    x_centers = (
        transform[0]
        + (np.arange(x0, x1, dtype=float) + 0.5) * transform[1]
        - site_easting
    ) / 1000.0
    y_centers = (
        transform[3]
        + (np.arange(y0, y1, dtype=float) + 0.5) * transform[5]
        - site_northing
    ) / 1000.0
    extent = (
        (transform[0] + x0 * transform[1] - site_easting) / 1000.0,
        (transform[0] + x1 * transform[1] - site_easting) / 1000.0,
        (transform[3] + y1 * transform[5] - site_northing) / 1000.0,
        (transform[3] + y0 * transform[5] - site_northing) / 1000.0,
    )
    window_radius = site_radius_pixels
    wx0 = center_x - window_radius
    wx1 = center_x + window_radius + 1
    wy0 = center_y - window_radius
    wy1 = center_y + window_radius + 1
    audit_rectangle = (
        (transform[0] + wx0 * transform[1] - site_easting) / 1000.0,
        (transform[3] + wy1 * transform[5] - site_northing) / 1000.0,
        (wx1 - wx0) * abs(transform[1]) / 1000.0,
        (wy1 - wy0) * abs(transform[5]) / 1000.0,
    )
    result = {
        "weak": weak.astype(np.float32),
        "strong": strong.astype(np.float32),
        "valid": valid,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "extent": extent,
        "audit_rectangle": audit_rectangle,
        "epsg": geometry.georef.epsg,
        "product_ids": [source.product_id for source in geometry.sources],
    }
    del members, geometry, mosaic
    gc.collect()
    return result


def _plot_balance(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    lookup = {(row["method"], row["metric"]): row for row in rows}
    method_labels = [
        str(lookup[(method, COMPARISON_METRICS[0][0])]["method_label"])
        for method in ("profile_only", "pdf_dwt_then_median")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for axis, (metric, label, _positive, _reverse) in zip(
        axes.ravel(), COMPARISON_METRICS
    ):
        x = np.arange(2, dtype=float)
        positive = [
            int(lookup[(method, metric)]["positive_count"])
            for method in ("profile_only", "pdf_dwt_then_median")
        ]
        reverse = [
            int(lookup[(method, metric)]["reverse_count"])
            for method in ("profile_only", "pdf_dwt_then_median")
        ]
        width = 0.34
        positive_bars = axis.bar(
            x - width / 2,
            positive,
            width,
            color="#247BA0",
            label="Positive methane-like",
        )
        reverse_bars = axis.bar(
            x + width / 2,
            reverse,
            width,
            color="#8C8C8C",
            label="Sign-reversed control",
        )
        axis.bar_label(positive_bars, padding=2, fontsize=8)
        axis.bar_label(reverse_bars, padding=2, fontsize=8)
        axis.set_title(label)
        axis.set_xticks(x, method_labels)
        axis.set_ylabel("Region count")
        axis.grid(axis="y", alpha=0.25)
        maximum = max(positive + reverse)
        axis.set_ylim(0, maximum * 1.2 + 0.5)
        ratios = [p / r if r else math.nan for p, r in zip(positive, reverse)]
        axis.text(
            0.02,
            0.96,
            f"positive/reverse: {ratios[0]:.3f} → {ratios[1]:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84, hspace=0.34, wspace=0.17)
    fig.suptitle(
        "Candidate balance under matched z ≥ 3 and ≥ 3-pixel rules", y=0.98
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_site_comparison(
    profile: Mapping[str, Any],
    pdf: Mapping[str, Any],
    output_path: Path,
    *,
    pdf_label: str = "PDF-DWT",
) -> None:
    if profile["epsg"] != pdf["epsg"]:
        raise ValueError("profile and PDF-DWT crops use different EPSG codes")
    if not np.allclose(profile["extent"], pdf["extent"], atol=1.0e-9):
        raise ValueError("profile and PDF-DWT crops are not on the same grid")
    arrays = (
        profile["weak"],
        pdf["weak"],
        pdf["weak"] - profile["weak"],
        profile["strong"],
        pdf["strong"],
        pdf["strong"] - profile["strong"],
    )
    titles = (
        "Profile-only: 1600 nm",
        f"{pdf_label}: 1600 nm",
        "Change: 1600 nm",
        "Profile-only: 2200–2390 nm",
        f"{pdf_label}: 2200–2390 nm",
        "Change: 2200–2390 nm",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.6), constrained_layout=True)
    score_image = None
    difference_image = None
    for index, (axis, values, title) in enumerate(zip(axes.ravel(), arrays, titles)):
        is_difference = index in (2, 5)
        image = axis.imshow(
            values,
            extent=profile["extent"],
            origin="upper",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1.5 if is_difference else -5.0,
            vmax=1.5 if is_difference else 5.0,
        )
        if is_difference:
            difference_image = image
        else:
            score_image = image
            finite = np.isfinite(values)
            if finite.any() and np.nanmin(values) <= 3.0 <= np.nanmax(values):
                axis.contour(
                    profile["x_centers"],
                    profile["y_centers"],
                    values,
                    levels=[3.0],
                    colors=["#F2C14E"],
                    linewidths=1.2,
                )
        rectangle = profile["audit_rectangle"]
        axis.add_patch(
            Rectangle(
                (rectangle[0], rectangle[1]),
                rectangle[2],
                rectangle[3],
                fill=False,
                edgecolor="black",
                linewidth=1.1,
                linestyle="--",
            )
        )
        axis.plot(0.0, 0.0, marker="+", color="black", markersize=10, mew=1.8)
        axis.set_title(title)
        axis.set_xlabel("Easting offset from site (km)")
        axis.set_ylabel("Northing offset from site (km)")
        axis.set_aspect("equal")
    if score_image is None or difference_image is None:
        raise RuntimeError("site comparison did not create both colour scales")
    fig.colorbar(score_image, ax=axes[:, :2], shrink=0.86, label="Local z")
    fig.colorbar(
        difference_image, ax=axes[:, 2], shrink=0.86, label="Local-z difference"
    )
    fig.suptitle(
        f"Keystone site-centred audit: profile vs {pdf_label}; "
        "dashed = 21×21 pixels, yellow = z 3 contour"
    )
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def compare(
    profile_summary_path: Path,
    pdf_summary_path: Path,
    profile_batch: Path,
    pdf_batch: Path,
    posthoc_summary_path: Path,
    output_dir: Path,
    *,
    known_site_csv: Path,
    known_site_id: str,
    acquisition_minute: str,
    crop_half_size: int = 40,
) -> dict[str, Any]:
    inputs = [
        profile_summary_path,
        pdf_summary_path,
        profile_batch,
        pdf_batch,
        posthoc_summary_path,
        known_site_csv,
    ]
    resolved = [path.expanduser().resolve() for path in inputs]
    (
        profile_summary_path,
        pdf_summary_path,
        profile_batch,
        pdf_batch,
        posthoc_summary_path,
        known_site_csv,
    ) = resolved
    output_dir = output_dir.expanduser().resolve()

    profile_summary = _read_json(profile_summary_path)
    pdf_summary = _read_json(pdf_summary_path)
    posthoc = _read_json(posthoc_summary_path)
    profile_batch_summary = _read_json(profile_batch / "batch_summary.json")
    pdf_batch_summary = _read_json(pdf_batch / "batch_summary.json")
    profile_manifest_path = profile_batch / "run_manifest.json"
    pdf_manifest_path = pdf_batch / "run_manifest.json"
    profile_manifest = _read_json(profile_manifest_path)
    pdf_manifest = _read_json(pdf_manifest_path)
    if Path(profile_summary["provenance"]["batch_directory"]).resolve() != profile_batch:
        raise ValueError("profile summary was not generated from --profile-batch")
    if Path(pdf_summary["provenance"]["batch_directory"]).resolve() != pdf_batch:
        raise ValueError("PDF summary was not generated from --pdf-batch")
    if posthoc_summary_path != pdf_batch / "posthoc_pdf_dwt_summary.json":
        raise ValueError("--posthoc-summary is not the canonical PDF-batch summary")
    if Path(posthoc["source_batch"]).resolve() != profile_batch:
        raise ValueError("posthoc source batch differs from --profile-batch")
    if Path(pdf_batch_summary["source_batch"]).resolve() != profile_batch:
        raise ValueError("PDF batch summary points to a different source batch")
    if posthoc["analysis_version"] != pdf_batch_summary["analysis_config"].get(
        "posthoc_analysis_version"
    ):
        raise ValueError("posthoc and PDF batch analysis versions differ")
    if posthoc["source_manifest_sha256"] != _sha256(profile_manifest_path):
        raise ValueError("posthoc source-manifest hash differs from profile batch")
    if profile_manifest.get("current_product_ids") != pdf_manifest.get(
        "current_product_ids"
    ):
        raise ValueError("profile and PDF batches contain different product IDs")
    if profile_summary["provenance"]["usable_product_ids"] != pdf_summary[
        "provenance"
    ]["usable_product_ids"]:
        raise ValueError("profile and PDF summaries contain different usable products")
    if (
        profile_batch_summary.get("analysis_config", {}).get("save_score_maps")
        is not True
    ):
        raise ValueError("profile batch does not preserve score maps")
    for key in (
        "usable_product_count",
        "acquisition_strip_count",
        "analysis_valid_pixels_after_same_strip_union",
    ):
        if profile_summary["aggregate"][key] != pdf_summary["aggregate"][key]:
            raise ValueError(f"profile and PDF summaries differ on {key}")
    profile_parameters = dict(profile_summary["parameters"])
    pdf_parameters = dict(pdf_summary["parameters"])
    profile_conservative_rule = profile_parameters.pop(
        "conservative_single_window", None
    )
    pdf_conservative_rule = pdf_parameters.pop("conservative_single_window", None)
    if profile_parameters != pdf_parameters:
        raise ValueError(
            "profile and PDF summaries use different core candidate rules"
        )
    threshold = float(profile_summary["parameters"]["local_z_threshold"])
    minimum_pixels = int(
        profile_summary["parameters"]["minimum_band_threshold_pixels"]
    )
    if not np.isfinite(threshold) or threshold <= 0 or minimum_pixels < 2:
        raise ValueError("candidate threshold or component size is invalid")
    profile_site_audit = _select_site_audit(
        profile_summary, acquisition_minute, known_site_id
    )
    pdf_site_audit = _select_site_audit(
        pdf_summary, acquisition_minute, known_site_id
    )
    if profile_site_audit["radius_pixels"] != pdf_site_audit["radius_pixels"]:
        raise ValueError("profile and PDF summaries use different site-window radii")
    site_radius_pixels = int(profile_site_audit["radius_pixels"])
    if site_radius_pixels < 0:
        raise ValueError("site-window radius must be non-negative")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    paths = {name: output_dir / name for name in OUTPUT_NAMES}

    scene_adaptive = (
        pdf_batch_summary["analysis_config"].get("posthoc_broad_slope_mode")
        == "per_scene_shared_weak_strong"
    )
    pdf_method_label = (
        "Per-scene slope DWT then median"
        if scene_adaptive
        else "PDF-DWT then median"
    )
    pdf_figure_label = "Scene-slope DWT" if scene_adaptive else "PDF-DWT"
    rows = build_aggregate_rows(
        profile_summary["aggregate"],
        pdf_summary["aggregate"],
        pdf_method_label=pdf_method_label,
    )
    with paths["pdf_dwt_aggregate_comparison.csv"].open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_balance(rows, paths["pdf_dwt_candidate_balance.png"])

    site = _load_known_site(known_site_csv, known_site_id)
    profile_crop = _load_site_crop(
        profile_batch,
        acquisition_minute=acquisition_minute,
        site=site,
        threshold=threshold,
        minimum_pixels=minimum_pixels,
        site_radius_pixels=site_radius_pixels,
        crop_half_size=crop_half_size,
    )
    pdf_crop = _load_site_crop(
        pdf_batch,
        acquisition_minute=acquisition_minute,
        site=site,
        threshold=threshold,
        minimum_pixels=minimum_pixels,
        site_radius_pixels=site_radius_pixels,
        crop_half_size=crop_half_size,
    )
    _plot_site_comparison(
        profile_crop,
        pdf_crop,
        paths["pdf_dwt_site_window_comparison.png"],
        pdf_label=pdf_figure_label,
    )
    del profile_crop, pdf_crop
    gc.collect()

    result = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "generated_utc": _utc_now(),
        "provenance": {
            "profile_summary": str(profile_summary_path),
            "profile_summary_sha256": _sha256(profile_summary_path),
            "pdf_summary": str(pdf_summary_path),
            "pdf_summary_sha256": _sha256(pdf_summary_path),
            "profile_batch": str(profile_batch),
            "pdf_batch": str(pdf_batch),
            "posthoc_summary": str(posthoc_summary_path),
            "posthoc_summary_sha256": _sha256(posthoc_summary_path),
            "known_site_csv": str(known_site_csv),
            "known_site_csv_sha256": _sha256(known_site_csv),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "fixed_candidate_rule": {
            "local_z_threshold": threshold,
            "minimum_component_pixels": minimum_pixels,
            "positive_and_sign_reversed_processed_symmetrically": True,
        },
        "candidate_rule_comparison": {
            "core_parameters_identical": True,
            "profile_conservative_single_window": profile_conservative_rule,
            "pdf_conservative_single_window": pdf_conservative_rule,
            "conservative_rule_difference": (
                "stripe-direction exclusion follows each correction branch; "
                "all other conservative thresholds are identical"
            ),
        },
        "pdf_method_label": pdf_method_label,
        "aggregate_comparison": rows,
        "directional_profile_diagnostics": _profile_diagnostics(posthoc),
        "known_site_comparison": {
            "profile_only": profile_site_audit,
            "pdf_dwt_then_median": pdf_site_audit,
        },
        "interpretation": (
            "The PDF-DWT branch is a sensitivity analysis. It is not adopted as "
            "the primary correction unless positive/reverse balance and physical "
            "MODTRAN injection recovery both improve without plume attenuation."
        ),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["pdf_dwt_reanalysis_comparison.json"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--pdf-summary", type=Path, required=True)
    parser.add_argument("--profile-batch", type=Path, required=True)
    parser.add_argument("--pdf-batch", type=Path, required=True)
    parser.add_argument("--posthoc-summary", type=Path, required=True)
    parser.add_argument("--known-site-csv", type=Path, required=True)
    parser.add_argument("--known-site-id", default="keystone_general")
    parser.add_argument(
        "--acquisition-minute", default="2022-10-30T16:00:00Z"
    )
    parser.add_argument("--crop-half-size", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    result = compare(
        args.profile_summary,
        args.pdf_summary,
        args.profile_batch,
        args.pdf_batch,
        args.posthoc_summary,
        args.output_dir,
        known_site_csv=args.known_site_csv,
        known_site_id=args.known_site_id,
        acquisition_minute=args.acquisition_minute,
        crop_half_size=args.crop_half_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a provenance-checked crop around one HISUI methane candidate.

The input scene output must have been created by
``screen_hisui_l1g_scenes.py --save-score-maps``.  A candidate can be selected
either by an explicit zero-based pixel centre or by its one-based rank within
the positive or reverse-control tail in ``candidate_components.csv``.

The browse JPEG is used directly when it has the score-map dimensions.  When
the dimensions differ, it is sampled onto the score grid under the explicit
assumption that it covers the complete raster with an axis-aligned
proportional mapping.  The JSON sidecar records this mapping; the browse image
is not treated as independently georeferenced evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tifffile

from hisui_l1g_io import HISUIL1GProduct, discover_l1g_product, parse_metadata


ANALYSIS_VERSION = "2026-07-28-v1"
REQUIRED_SCORE_ARRAYS = (
    "valid",
    "cloud_proxy",
    "weak_local_z",
    "strong_local_z",
    "dual_local_z",
)
TAIL_CHOICES = ("positive_ch4", "reverse_sign_control")
INTERPRETATION = (
    "Screening visualization only. Morphology, source association, wind "
    "direction and speed, and independent clear-sky or repeat observations "
    "are required before attributing the feature to methane."
)


@dataclass(frozen=True)
class SceneCropSource:
    product: HISUIL1GProduct
    score_path: Path
    summary_path: Path
    summary: dict[str, object]
    valid: np.ndarray
    cloud: np.ndarray
    weak: np.ndarray
    strong: np.ndarray
    dual: np.ndarray
    metadata: dict[str, object]
    provenance: dict[str, object]


@dataclass(frozen=True)
class CropSelection:
    center_y: int
    center_x: int
    mode: str
    tail: str | None
    candidate_rank: int | None
    candidate: dict[str, object] | None


def resolve_score_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "score_maps.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"Score-map archive not found: {candidate}")
    return candidate


def _binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array.copy()
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be Boolean or finite binary numeric")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be Boolean or finite binary numeric")
    return array.astype(bool)


def _normalized_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    return os.path.normcase(os.path.normpath(str(resolved)))


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_scene_crop_source(
    product_path: str | Path, scene_output: str | Path
) -> SceneCropSource:
    """Load maps and reject a product/output provenance mismatch."""

    product = discover_l1g_product(product_path)
    score_path = resolve_score_path(scene_output)
    summary_path = score_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "A screen_hisui_l1g_scenes.py summary.json is required beside "
            f"the score archive: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid source summary JSON: {summary_path}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Source summary must be a JSON object: {summary_path}")

    summary_product_id = summary.get("product_id")
    if summary_product_id != product.product_id:
        raise ValueError(
            f"Score-map product_id {summary_product_id!r} does not match "
            f"product {product.product_id!r}"
        )
    summary_product_path = summary.get("product_path")
    if not isinstance(summary_product_path, str) or not summary_product_path:
        raise ValueError(f"Source summary lacks product_path: {summary_path}")
    if _normalized_path(summary_product_path) != _normalized_path(product.directory):
        raise ValueError(
            "Score-map product_path does not match the supplied HISUI product "
            f"directory: {summary_product_path!r} != {str(product.directory)!r}"
        )
    analysis_config = summary.get("analysis_config")
    if not isinstance(analysis_config, dict):
        raise ValueError(f"Source summary lacks analysis_config: {summary_path}")

    with np.load(score_path, allow_pickle=False) as stored:
        missing = sorted(set(REQUIRED_SCORE_ARRAYS) - set(stored.files))
        if missing:
            raise ValueError(
                f"Missing arrays in {score_path}: {', '.join(missing)}"
            )
        valid = _binary_mask(stored["valid"], "valid")
        cloud = _binary_mask(stored["cloud_proxy"], "cloud_proxy")
        weak = np.asarray(stored["weak_local_z"], dtype=np.float64)
        strong = np.asarray(stored["strong_local_z"], dtype=np.float64)
        dual = np.asarray(stored["dual_local_z"], dtype=np.float64)

    arrays = {
        "valid": valid,
        "cloud_proxy": cloud,
        "weak": weak,
        "strong": strong,
        "dual": dual,
    }
    for name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if array.shape != valid.shape:
            raise ValueError(f"{name} shape {array.shape} differs from {valid.shape}")
    if valid.size == 0:
        raise ValueError("Score maps must not be empty")
    if np.any(valid & cloud):
        raise ValueError("valid and cloud_proxy overlap; clear/cloud provenance is invalid")

    jointly_finite = valid & np.isfinite(weak) & np.isfinite(strong) & np.isfinite(dual)
    if np.any(jointly_finite) and not np.allclose(
        dual[jointly_finite],
        np.minimum(weak, strong)[jointly_finite],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("dual_local_z is not the minimum of weak_local_z and strong_local_z")

    raw_shape = summary.get("shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) < 2
        or tuple(int(value) for value in raw_shape[:2]) != valid.shape
    ):
        raise ValueError("Source summary shape does not match score-map shape")
    with tifffile.TiffFile(str(product.image_path)) as tif:
        page = tif.pages[0]
        raster_shape = (int(page.imagelength), int(page.imagewidth))
    if raster_shape != valid.shape:
        raise ValueError(
            f"Product raster shape {raster_shape} does not match score maps {valid.shape}"
        )

    metadata = parse_metadata(product.metadata_path)
    summary_acquisition = summary.get("acquisition_utc")
    metadata_acquisition = metadata.get("SceneCenterTime")
    acquisition_checked = (
        summary_acquisition is not None and metadata_acquisition is not None
    )
    if acquisition_checked and str(summary_acquisition) != str(metadata_acquisition):
        raise ValueError(
            "Score-map acquisition_utc does not match product SceneCenterTime"
        )

    provenance = {
        "product_id_match": True,
        "product_path_match": True,
        "summary_shape_match": True,
        "raster_shape_match": True,
        "acquisition_time_checked": acquisition_checked,
        "acquisition_time_match": True if acquisition_checked else None,
        "analysis_config_sha256": _canonical_hash(analysis_config),
    }
    return SceneCropSource(
        product=product,
        score_path=score_path,
        summary_path=summary_path,
        summary=summary,
        valid=valid,
        cloud=cloud,
        weak=weak,
        strong=strong,
        dual=dual,
        metadata=metadata,
        provenance=provenance,
    )


def _candidate_rows(
    path: Path, product_id: str, acquisition_utc: object
) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate table not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "tail",
            "peak_y",
            "peak_x",
            "dual_local_z_peak",
            "product_id",
            "acquisition_utc",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Candidate table lacks: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    for row in rows:
        row_product_id = row.get("product_id")
        if row_product_id != product_id:
            raise ValueError(
                f"Candidate table product_id {row_product_id!r} does not match "
                f"product {product_id!r}"
            )
        if acquisition_utc is not None and str(row.get("acquisition_utc")) != str(
            acquisition_utc
        ):
            raise ValueError(
                "Candidate table acquisition_utc does not match the source summary"
            )
    return rows


def _coerce_candidate_row(row: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    integer_fields = {
        "component_id",
        "pixel_count",
        "peak_y",
        "peak_x",
        "bbox_height",
        "bbox_width",
        "epsg",
    }
    boolean_fields = {"scene_spanning_line_flag"}
    for key, raw in row.items():
        if raw is None or raw == "":
            output[key] = None
        elif key in integer_fields:
            output[key] = int(float(str(raw)))
        elif key in boolean_fields:
            output[key] = str(raw).strip().lower() in {"1", "true", "yes"}
        elif key in {"tail", "product_id", "acquisition_utc"}:
            output[key] = str(raw)
        else:
            try:
                numeric = float(str(raw))
                output[key] = numeric if np.isfinite(numeric) else None
            except ValueError:
                output[key] = str(raw)
    return output


def resolve_selection(
    source: SceneCropSource,
    *,
    center_y: int | None,
    center_x: int | None,
    candidate_rank: int | None,
    tail: str,
) -> CropSelection:
    has_y = center_y is not None
    has_x = center_x is not None
    if has_y != has_x:
        raise ValueError("--center-y and --center-x must be supplied together")
    if has_y == (candidate_rank is not None):
        raise ValueError(
            "Choose exactly one selection mode: centre y/x or candidate rank"
        )

    selected_candidate: dict[str, object] | None = None
    if candidate_rank is not None:
        if candidate_rank < 1:
            raise ValueError("candidate rank is one-based and must be positive")
        if tail not in TAIL_CHOICES:
            raise ValueError(f"Unknown candidate tail: {tail}")
        rows = _candidate_rows(
            source.score_path.parent / "candidate_components.csv",
            source.product.product_id,
            source.summary.get("acquisition_utc"),
        )
        matching = [
            _coerce_candidate_row(row) for row in rows if row.get("tail") == tail
        ]
        matching.sort(
            key=lambda row: float(row["dual_local_z_peak"]), reverse=True
        )
        if candidate_rank > len(matching):
            raise ValueError(
                f"Candidate rank {candidate_rank} exceeds {len(matching)} "
                f"rows in tail {tail!r}"
            )
        selected_candidate = matching[candidate_rank - 1]
        center_y = int(selected_candidate["peak_y"])
        center_x = int(selected_candidate["peak_x"])
        mode = "candidate_rank"
    else:
        assert center_y is not None and center_x is not None
        center_y, center_x = int(center_y), int(center_x)
        mode = "explicit_pixel_center"

    height, width = source.valid.shape
    if not (0 <= center_y < height and 0 <= center_x < width):
        raise ValueError(
            f"Centre {(center_y, center_x)} lies outside score-map shape "
            f"{source.valid.shape}"
        )
    if selected_candidate is not None:
        expected_peak = float(selected_candidate["dual_local_z_peak"])
        if tail == "positive_ch4":
            stored_peak = source.dual[center_y, center_x]
        else:
            stored_peak = min(
                -source.weak[center_y, center_x],
                -source.strong[center_y, center_x],
            )
        if (
            not np.isfinite(expected_peak)
            or not np.isfinite(stored_peak)
            or not np.isclose(
                expected_peak, stored_peak, rtol=1e-6, atol=1e-6
            )
        ):
            raise ValueError(
                "Candidate-table peak score does not match score_maps.npz at "
                "the recorded peak pixel"
            )
    return CropSelection(
        center_y=center_y,
        center_x=center_x,
        mode=mode,
        tail=tail if candidate_rank is not None else None,
        candidate_rank=candidate_rank,
        candidate=selected_candidate,
    )


def crop_bounds(
    shape: tuple[int, int], center_y: int, center_x: int, half_size: int
) -> tuple[int, int, int, int]:
    if half_size < 1:
        raise ValueError("half-size must be positive")
    height, width = shape
    return (
        max(center_y - half_size, 0),
        min(center_y + half_size + 1, height),
        max(center_x - half_size, 0),
        min(center_x + half_size + 1, width),
    )


def browse_crop(
    browse: np.ndarray,
    score_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Map a full-scene browse image to score pixels using pixel centres."""

    image = np.asarray(browse)
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Browse image must have non-empty YX or YXC dimensions")
    browse_shape = (int(image.shape[0]), int(image.shape[1]))
    score_height, score_width = score_shape
    y0, y1, x0, x1 = bounds
    if browse_shape == score_shape:
        cropped = image[y0:y1, x0:x1].copy()
        mode = "native_pixel_grid"
        assumption = "Browse and score dimensions match; no resampling was used."
    else:
        source_y = np.floor(
            (np.arange(y0, y1, dtype=float) + 0.5)
            * browse_shape[0]
            / score_height
        ).astype(int)
        source_x = np.floor(
            (np.arange(x0, x1, dtype=float) + 0.5)
            * browse_shape[1]
            / score_width
        ).astype(int)
        source_y = np.clip(source_y, 0, browse_shape[0] - 1)
        source_x = np.clip(source_x, 0, browse_shape[1] - 1)
        if image.ndim == 2:
            cropped = image[source_y[:, None], source_x[None, :]]
        else:
            cropped = image[source_y[:, None], source_x[None, :], :]
        mode = "axis_aligned_proportional_nearest_resample"
        assumption = (
            "Browse is assumed to cover the complete score raster with an "
            "axis-aligned proportional mapping; it is not independently "
            "georeferenced. Nearest browse pixels were sampled at score-pixel "
            "centres."
        )
    return cropped, {
        "status": "available",
        "mode": mode,
        "browse_shape_yx": list(browse_shape),
        "score_shape_yx": list(score_shape),
        "browse_pixels_per_score_pixel_y": browse_shape[0] / score_height,
        "browse_pixels_per_score_pixel_x": browse_shape[1] / score_width,
        "assumption": assumption,
    }


def _finite_score_stats(
    image: np.ndarray, clear: np.ndarray, center: tuple[int, int], threshold: float
) -> dict[str, object]:
    usable = clear & np.isfinite(image)
    values = image[usable]
    center_y, center_x = center
    center_value = image[center_y, center_x]
    return {
        "finite_clear_pixels": int(values.size),
        "center_value": (
            float(center_value)
            if clear[center_y, center_x] and np.isfinite(center_value)
            else None
        ),
        "mean": float(np.mean(values)) if values.size else None,
        "maximum": float(np.max(values)) if values.size else None,
        "quantiles_50_90_95_99": (
            [float(value) for value in np.quantile(values, [0.5, 0.9, 0.95, 0.99])]
            if values.size
            else [None, None, None, None]
        ),
        "pixels_at_or_above_candidate_threshold": int(np.sum(values >= threshold)),
    }


def _mark_center(axis: plt.Axes, center_y: int, center_x: int) -> None:
    axis.plot(
        center_x,
        center_y,
        marker="x",
        color="black",
        markersize=10,
        markeredgewidth=3.0,
        linestyle="none",
        zorder=10,
    )
    axis.plot(
        center_x,
        center_y,
        marker="+",
        color="#ffff33",
        markersize=12,
        markeredgewidth=2.0,
        linestyle="none",
        zorder=11,
    )


def save_candidate_figure(
    path: Path,
    source: SceneCropSource,
    selection: CropSelection,
    bounds: tuple[int, int, int, int],
    browse_image: np.ndarray | None,
    browse_mapping: Mapping[str, object],
    *,
    score_min: float,
    score_max: float,
) -> None:
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_min >= score_max:
        raise ValueError("score-min must be finite and less than score-max")
    y0, y1, x0, x1 = bounds
    extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
    valid = source.valid[y0:y1, x0:x1]
    cloud = source.cloud[y0:y1, x0:x1]
    weak = source.weak[y0:y1, x0:x1]
    strong = source.strong[y0:y1, x0:x1]
    dual = source.dual[y0:y1, x0:x1]
    reverse = np.minimum(-weak, -strong)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if browse_image is None:
        axes[0, 0].set_facecolor("#d9d9d9")
        axes[0, 0].text(
            0.5,
            0.5,
            "Browse JPG unavailable",
            transform=axes[0, 0].transAxes,
            ha="center",
            va="center",
        )
        browse_title = "Browse context (missing)"
    else:
        axes[0, 0].imshow(
            browse_image,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap="gray" if browse_image.ndim == 2 else None,
        )
        mapping_mode = str(browse_mapping["mode"])
        browse_title = (
            "Browse context (native grid)"
            if mapping_mode == "native_pixel_grid"
            else "Browse context (proportional resampling)"
        )
    axes[0, 0].set_title(browse_title)

    context = np.zeros(valid.shape, dtype=np.uint8)
    context[valid] = 1
    context[cloud] = 2
    axes[0, 1].imshow(
        context,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#d9d9d9", "#65a765", "#7b3294"]),
        vmin=0,
        vmax=2,
    )
    axes[0, 1].set_title("Mask: green clear; purple cloud; gray invalid")

    score_panels = (
        (axes[0, 2], weak, "Weak-window local z"),
        (axes[1, 0], strong, "Strong-window local z"),
        (axes[1, 1], dual, "Dual-window minimum local z"),
        (axes[1, 2], reverse, "Reverse-sign dual-tail control"),
    )
    score_cmap = plt.get_cmap("coolwarm").copy()
    score_cmap.set_bad("#d9d9d9")
    cloud_cmap = ListedColormap(["#7b3294"])
    score_artist = None
    for axis, image, title in score_panels:
        shown = np.ma.masked_where(~valid | ~np.isfinite(image), image)
        score_artist = axis.imshow(
            shown,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap=score_cmap,
            vmin=score_min,
            vmax=score_max,
        )
        cloud_overlay = np.ma.masked_where(~cloud, np.ones(cloud.shape))
        axis.imshow(
            cloud_overlay,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap=cloud_cmap,
            vmin=0,
            vmax=1,
        )
        axis.set_title(title)

    for axis in axes.flat:
        axis.set_xlim(x0 - 0.5, x1 - 0.5)
        axis.set_ylim(y1 - 0.5, y0 - 0.5)
        axis.set_aspect("equal")
        axis.set_xlabel("HISUI pixel x")
        axis.set_ylabel("HISUI pixel y")
        _mark_center(axis, selection.center_y, selection.center_x)

    assert score_artist is not None
    fig.colorbar(
        score_artist,
        ax=[panel[0] for panel in score_panels],
        shrink=0.75,
        label="Local matched-filter z score",
    )
    selection_label = (
        f"{selection.tail} rank {selection.candidate_rank}"
        if selection.mode == "candidate_rank"
        else "explicit centre"
    )
    fig.suptitle(
        f"{source.product.product_id} | {selection_label} | "
        f"centre (y={selection.center_y}, x={selection.center_x})"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_candidate_crop(
    *,
    product_path: str | Path,
    scene_output: str | Path,
    output_dir: str | Path,
    center_y: int | None = None,
    center_x: int | None = None,
    candidate_rank: int | None = None,
    tail: str = "positive_ch4",
    half_size: int = 40,
    score_min: float = -3.0,
    score_max: float = 6.0,
) -> dict[str, object]:
    source = load_scene_crop_source(product_path, scene_output)
    selection = resolve_selection(
        source,
        center_y=center_y,
        center_x=center_x,
        candidate_rank=candidate_rank,
        tail=tail,
    )
    bounds = crop_bounds(
        source.valid.shape, selection.center_y, selection.center_x, half_size
    )
    y0, y1, x0, x1 = bounds

    browse_path = source.product.directory / f"{source.product.product_id}_1.jpg"
    if browse_path.is_file():
        browse_image, browse_mapping = browse_crop(
            mpimg.imread(browse_path), source.valid.shape, bounds
        )
        browse_mapping = {"path": str(browse_path.resolve()), **browse_mapping}
    else:
        browse_image = None
        browse_mapping = {
            "status": "missing",
            "path": None,
            "mode": None,
            "browse_shape_yx": None,
            "score_shape_yx": list(source.valid.shape),
            "browse_pixels_per_score_pixel_y": None,
            "browse_pixels_per_score_pixel_x": None,
            "assumption": None,
        }

    valid_crop = source.valid[y0:y1, x0:x1]
    cloud_crop = source.cloud[y0:y1, x0:x1]
    weak_crop = source.weak[y0:y1, x0:x1]
    strong_crop = source.strong[y0:y1, x0:x1]
    dual_crop = source.dual[y0:y1, x0:x1]
    reverse_crop = np.minimum(-weak_crop, -strong_crop)
    local_center = (selection.center_y - y0, selection.center_x - x0)
    threshold = float(source.summary["analysis_config"].get("cluster_threshold", 3.0))
    if not np.isfinite(threshold):
        raise ValueError("analysis_config cluster_threshold must be finite")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if selection.mode == "candidate_rank":
        stem = (
            f"candidate_crop_{selection.tail}_rank{selection.candidate_rank}_"
            f"y{selection.center_y}_x{selection.center_x}"
        )
    else:
        stem = f"candidate_crop_y{selection.center_y}_x{selection.center_x}"
    figure_path = output_path / f"{stem}.png"
    json_path = output_path / f"{stem}.json"

    result: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "product_id": source.product.product_id,
        "source": {
            "product_path": str(source.product.directory.resolve()),
            "image_path": str(source.product.image_path.resolve()),
            "score_maps": str(source.score_path),
            "summary": str(source.summary_path),
            "acquisition_utc": source.summary.get("acquisition_utc"),
            "processing_utc": source.summary.get("processing_utc"),
            "quality_class": source.summary.get("quality_class"),
            "analysis_config": source.summary["analysis_config"],
        },
        "provenance_verification": source.provenance,
        "selection": {
            "mode": selection.mode,
            "center_y": selection.center_y,
            "center_x": selection.center_x,
            "tail": selection.tail,
            "candidate_rank": selection.candidate_rank,
            "candidate_row": selection.candidate,
        },
        "crop": {
            "bounds_y0_y1_x0_x1_exclusive": [y0, y1, x0, x1],
            "shape_yx": [y1 - y0, x1 - x0],
            "requested_half_size_pixels": half_size,
            "truncated_at_scene_edge": (
                y0 != selection.center_y - half_size
                or y1 != selection.center_y + half_size + 1
                or x0 != selection.center_x - half_size
                or x1 != selection.center_x + half_size + 1
            ),
        },
        "browse_mapping": browse_mapping,
        "mask_statistics": {
            "crop_pixels": int(valid_crop.size),
            "clear_analysis_pixels": int(valid_crop.sum()),
            "cloud_proxy_pixels": int(cloud_crop.sum()),
            "other_invalid_pixels": int((~valid_crop & ~cloud_crop).sum()),
            "clear_fraction": float(valid_crop.mean()),
            "cloud_proxy_fraction": float(cloud_crop.mean()),
            "center_is_clear": bool(valid_crop[local_center]),
            "center_is_cloud_proxy": bool(cloud_crop[local_center]),
        },
        "candidate_threshold_local_z": threshold,
        "score_display_range": [float(score_min), float(score_max)],
        "score_statistics": {
            "weak_local_z": _finite_score_stats(
                weak_crop, valid_crop, local_center, threshold
            ),
            "strong_local_z": _finite_score_stats(
                strong_crop, valid_crop, local_center, threshold
            ),
            "dual_local_z": _finite_score_stats(
                dual_crop, valid_crop, local_center, threshold
            ),
            "reverse_sign_control": _finite_score_stats(
                reverse_crop, valid_crop, local_center, threshold
            ),
        },
        "interpretation": INTERPRETATION,
        "outputs": {
            "figure": str(figure_path),
            "json": str(json_path),
        },
    }
    save_candidate_figure(
        figure_path,
        source,
        selection,
        bounds,
        browse_image,
        browse_mapping,
        score_min=score_min,
        score_max=score_max,
    )
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-output",
        type=Path,
        required=True,
        help="Scene output directory or its score_maps.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-y", type=int)
    parser.add_argument("--center-x", type=int)
    parser.add_argument(
        "--candidate-rank",
        type=int,
        help="One-based rank after sorting the selected tail by peak score",
    )
    parser.add_argument("--tail", choices=TAIL_CHOICES, default="positive_ch4")
    parser.add_argument("--half-size", type=int, default=40)
    parser.add_argument("--score-min", type=float, default=-3.0)
    parser.add_argument("--score-max", type=float, default=6.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = make_candidate_crop(
        product_path=args.product_dir,
        scene_output=args.scene_output,
        output_dir=args.output_dir,
        center_y=args.center_y,
        center_x=args.center_x,
        candidate_rank=args.candidate_rank,
        tail=args.tail,
        half_size=args.half_size,
        score_min=args.score_min,
        score_max=args.score_max,
    )
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

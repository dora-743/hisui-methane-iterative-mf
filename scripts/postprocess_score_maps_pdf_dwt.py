#!/usr/bin/env python3
"""Create a PDF-method DWT-derived batch from saved full-scene MF maps.

The source batch is never modified.  For each ``quality_class=usable`` product,
the script starts from the saved, pre-destriping ``weak_z`` and ``strong_z``
maps and applies the workflow documented in the July 3 slides:

1. estimate one broad-stripe slope for that scene from the shared weak/strong
   RStd-reduction score, then apply support-aware Haar DWT (levels 3--5), and
2. fixed-slope line-median subtraction at the thin QA trace slope
   0.9773460526.

The candidate-protection mask is frozen before either band is corrected and is
the sign-symmetric union of both bands.  Invalid pixels, protected pixels, and
reflect padding never contribute to DWT threshold estimation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy
from scipy import ndimage

from directional_destriping import (
    directional_profile,
    fixed_slope_line_ids,
    robust_std,
    support_aware_wavelet_horizontal_destripe,
)
from screen_hisui_l1g_scenes import weighted_local_z
from scene_stripe_slope import (
    SceneSlopeSearchConfig,
    estimate_scene_broad_slope,
    write_slope_search_csv,
)


ANALYSIS_VERSION = "2026-08-03-pdf-dwt-scene-slope-v5"
REFERENCE_BROAD_SLOPE = 1.257172298918948
REFERENCE_BROAD_ROTATION_DEGREES = math.degrees(
    math.atan(REFERENCE_BROAD_SLOPE)
)
THIN_SLOPE = 0.9773460526106752
THIN_BIN_WIDTH = 2.0
DWT_LEVELS = (3, 4, 5)
DWT_MAX_LEVEL = 6
DWT_THRESHOLD_SCALE = 0.75
DWT_DIFF_FRACTION = 0.25
DWT_MINIMUM_SUPPORT = 0.95
DWT_MINIMUM_ESTIMATION_SUPPORT = 1.0
DWT_MINIMUM_COEFFICIENTS = 300
PROTECTION_CORE_Z = 4.0
PROTECTION_EXTENT_Z = 3.0
PROTECTION_MINIMUM_PIXELS = 3
PROTECTION_DILATION_PIXELS = 2
THIN_MINIMUM_LINE_PIXELS = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


@contextmanager
def _exclusive_output_lock(output_batch: Path):
    """Prevent concurrent writers from deriving the same long-running batch."""
    lock_path = output_batch.with_name(f".{output_batch.name}.lock")
    try:
        descriptor = os.open(
            str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"output batch is already being created: {output_batch}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "output_batch": str(output_batch),
                        "pid": os.getpid(),
                        "started_utc": _utc_now(),
                    },
                    ensure_ascii=False,
                )
            )
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def safe_rotation_canvas(
    shape: tuple[int, int], *, angle_degrees: float, divisor: int = 64
) -> int:
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must contain two positive dimensions")
    if divisor < 1:
        raise ValueError("divisor must be positive")
    height, width = shape
    theta = math.radians(angle_degrees)
    rotated_height = abs(height * math.cos(theta)) + abs(width * math.sin(theta))
    rotated_width = abs(width * math.cos(theta)) + abs(height * math.sin(theta))
    required = max(rotated_height, rotated_width)
    return int(math.ceil(required / divisor) * divisor)


def _kept_component_mask(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(
        np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def symmetric_protection_mask(
    weak_raw: np.ndarray,
    strong_raw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    local_sigma_pixels: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.asarray(valid_mask, dtype=bool)
    raw_locals: list[np.ndarray] = []
    initial = np.zeros_like(valid)
    per_band: dict[str, dict[str, int]] = {}
    for name, raw in (("weak", weak_raw), ("strong", strong_raw)):
        local = weighted_local_z(raw, valid, sigma_pixels=local_sigma_pixels)
        raw_locals.append(local)
        core = valid & (np.abs(local) >= PROTECTION_CORE_Z)
        extent = _kept_component_mask(
            valid & (np.abs(local) >= PROTECTION_EXTENT_Z),
            PROTECTION_MINIMUM_PIXELS,
        )
        initial |= core | extent
        per_band[name] = {
            "absolute_z4_pixels": int(np.count_nonzero(core)),
            "retained_absolute_z3_pixels": int(np.count_nonzero(extent)),
        }
    protected = ndimage.binary_dilation(
        initial, iterations=PROTECTION_DILATION_PIXELS
    ) & valid
    diagnostics: dict[str, Any] = {
        "definition": (
            "shared weak/strong union of |raw local-z|>=4 and retained "
            "3-pixel |raw local-z|>=3 components; dilated by 2 pixels"
        ),
        "initial_pixels": int(np.count_nonzero(initial)),
        "dilated_pixels": int(np.count_nonzero(protected)),
        "valid_fraction": float(np.count_nonzero(protected) / np.count_nonzero(valid)),
        "per_band": per_band,
    }
    del raw_locals
    return protected, diagnostics


def fast_fixed_slope_median_destripe(
    image: np.ndarray,
    valid_mask: np.ndarray,
    exclude_mask: np.ndarray,
    *,
    slope: float = THIN_SLOPE,
    line_bin_width: float = THIN_BIN_WIDTH,
    minimum_pixels_per_line: int = THIN_MINIMUM_LINE_PIXELS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    array = np.asarray(image, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(array)
    exclude = np.asarray(exclude_mask, dtype=bool)
    if array.ndim != 2 or valid.shape != array.shape or exclude.shape != array.shape:
        raise ValueError("image, valid_mask, and exclude_mask must share a 2-D shape")
    estimate = valid & ~exclude
    if not estimate.any():
        raise ValueError("no unprotected valid pixels are available for thin median")
    ids = fixed_slope_line_ids(
        array.shape,
        slope=slope,
        line_bin_width=line_bin_width,
        direction_key="y_minus_x",
    )
    minimum_id = int(ids[valid].min())
    maximum_id = int(ids[valid].max())
    line_count = maximum_id - minimum_id + 1
    normalized = ids - minimum_id
    estimate_labels = np.where(estimate, normalized + 1, 0).astype(np.int32)
    indices = np.arange(1, line_count + 1, dtype=np.int32)
    counts = np.bincount(estimate_labels.ravel(), minlength=line_count + 1)[1:]
    medians = np.asarray(
        ndimage.median(array, labels=estimate_labels, index=indices), dtype=float
    )
    good = (counts >= minimum_pixels_per_line) & np.isfinite(medians)
    if not np.any(good):
        raise ValueError("no thin line has enough unprotected valid pixels")
    global_median = float(np.median(array[estimate]))
    offsets = np.zeros(line_count, dtype=float)
    offsets[good] = medians[good] - global_median
    missing = ~good
    if np.any(missing):
        line_positions = np.arange(line_count, dtype=float)
        offsets[missing] = np.interp(
            line_positions[missing], line_positions[good], offsets[good]
        )
    stripe = np.full(array.shape, np.nan, dtype=float)
    stripe[valid] = offsets[normalized[valid]]
    corrected = array - stripe
    corrected[~valid] = np.nan
    diagnostics = {
        "slope": float(slope),
        "line_bin_width_pixels": float(line_bin_width),
        "minimum_pixels_per_line": int(minimum_pixels_per_line),
        "line_count": int(line_count),
        "directly_estimated_line_count": int(np.count_nonzero(good)),
        "interpolated_line_count": int(np.count_nonzero(missing)),
        "global_median": global_median,
        "offset_robust_std": robust_std(offsets),
        "maximum_absolute_offset": float(np.max(np.abs(offsets))),
        "protected_pixels_excluded": int(np.count_nonzero(valid & exclude)),
    }
    return corrected, stripe, diagnostics


def _profile_rstd(
    image: np.ndarray,
    *,
    slope: float,
    bin_width: float,
    protected_mask: np.ndarray,
) -> float:
    _coordinate, profile = directional_profile(
        image,
        slope=slope,
        bin_width=bin_width,
        protected_mask=protected_mask,
    )
    return robust_std(profile)


def _tail_metrics(local: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    return {
        "positive_z3_pixels": int(np.count_nonzero(valid & (local >= 3.0))),
        "reverse_z3_pixels": int(np.count_nonzero(valid & (local <= -3.0))),
        "positive_z5_pixels": int(np.count_nonzero(valid & (local >= 5.0))),
        "reverse_z5_pixels": int(np.count_nonzero(valid & (local <= -5.0))),
    }


def _process_band(
    raw: np.ndarray,
    old_local: np.ndarray,
    old_corrected: np.ndarray | None,
    valid: np.ndarray,
    protected: np.ndarray,
    *,
    local_sigma_pixels: float,
    canvas_size: int,
    band_name: str,
    minimum_threshold_coefficients: int,
    broad_slope: float,
    broad_rotation_degrees: float,
    apply_broad_dwt: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if apply_broad_dwt:
        dwt_corrected, dwt_stripe, dwt_rows = (
            support_aware_wavelet_horizontal_destripe(
                raw,
                valid,
                protected,
                slope=broad_slope,
                rotation_angle_deg=broad_rotation_degrees,
                levels_to_filter=DWT_LEVELS,
                max_level=DWT_MAX_LEVEL,
                threshold_scale=DWT_THRESHOLD_SCALE,
                diff_fraction=DWT_DIFF_FRACTION,
                canvas_size=canvas_size,
                minimum_support_fraction=DWT_MINIMUM_SUPPORT,
                minimum_estimation_support_fraction=DWT_MINIMUM_ESTIMATION_SUPPORT,
                minimum_threshold_coefficients=minimum_threshold_coefficients,
                operation_name=f"{band_name}:pdf_broad_dwt",
            )
        )
    else:
        dwt_corrected = np.asarray(raw, dtype=float).copy()
        dwt_stripe = np.zeros(raw.shape, dtype=float)
        dwt_rows = [
            {
                "operation": f"{band_name}:pdf_broad_dwt",
                "slope": float(broad_slope),
                "rotation_angle_deg": float(broad_rotation_degrees),
                "level": level,
                "requested": 0,
                "filtered": 0,
                "status": "skipped_unsupported_scene_slope",
                "raw_threshold": 0.0,
                "applied_threshold": 0.0,
                "estimate_coefficient_count": 0,
                "apply_coefficient_count": 0,
                "total_coefficient_count": 0,
                "minimum_application_support_fraction": DWT_MINIMUM_SUPPORT,
                "minimum_estimation_support_fraction": (
                    DWT_MINIMUM_ESTIMATION_SUPPORT
                ),
                "horizontal_robust_std_supported": None,
                "vertical_robust_std_supported": None,
                "horizontal_energy_supported": None,
                "vertical_energy_supported": None,
            }
            for level in range(1, DWT_MAX_LEVEL + 1)
        ]
    thin_exclusion = ndimage.binary_dilation(protected, iterations=1) & valid
    corrected, thin_stripe, thin_diagnostics = fast_fixed_slope_median_destripe(
        dwt_corrected,
        valid,
        thin_exclusion,
    )
    local = weighted_local_z(corrected, valid, sigma_pixels=local_sigma_pixels)
    comparison: dict[str, Any] = {
        "source_profile_local_tail": _tail_metrics(old_local, valid),
        "pdf_dwt_local_tail": _tail_metrics(local, valid),
        "valid_local_correlation": _finite_or_none(
            np.corrcoef(old_local[valid], local[valid])[0, 1]
        ),
        "protected_local_change_rms": float(
            np.sqrt(np.mean((local[protected] - old_local[protected]) ** 2))
        )
        if np.any(protected)
        else None,
        "protected_absolute_local_change_q99": float(
            np.quantile(np.abs(local[protected] - old_local[protected]), 0.99)
        )
        if np.any(protected)
        else None,
        "pdf_broad_profile_rstd": _profile_rstd(
            corrected,
            slope=broad_slope,
            bin_width=18.0,
            protected_mask=protected,
        ),
        "pdf_thin_profile_rstd": _profile_rstd(
            corrected,
            slope=THIN_SLOPE,
            bin_width=THIN_BIN_WIDTH,
            protected_mask=protected,
        ),
    }
    if old_corrected is not None:
        comparison.update(
            {
                "source_broad_profile_rstd": _profile_rstd(
                    old_corrected,
                    slope=broad_slope,
                    bin_width=18.0,
                    protected_mask=protected,
                ),
                "source_thin_profile_rstd": _profile_rstd(
                    old_corrected,
                    slope=THIN_SLOPE,
                    bin_width=THIN_BIN_WIDTH,
                    protected_mask=protected,
                ),
            }
        )
    diagnostics = {
        "scene_broad_slope": float(broad_slope),
        "scene_broad_rotation_degrees": float(broad_rotation_degrees),
        "broad_dwt_applied": bool(apply_broad_dwt),
        "dwt": dwt_rows,
        "thin_median": thin_diagnostics,
        "comparison": comparison,
        "dwt_stripe_robust_std": robust_std(dwt_stripe, valid),
        "thin_stripe_robust_std": robust_std(thin_stripe, valid),
        "total_removed_robust_std": robust_std(raw - corrected, valid),
    }
    return corrected.astype(np.float32), local.astype(np.float32), diagnostics


def _validated_source_batch(
    source_batch: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    manifest = _read_json(source_batch / "run_manifest.json", label="run manifest")
    summary = _read_json(source_batch / "batch_summary.json", label="batch summary")
    if manifest.get("status") != "complete":
        raise ValueError("source run_manifest status must be complete")
    product_ids = manifest.get("current_product_ids")
    if (
        not isinstance(product_ids, list)
        or not product_ids
        or any(not isinstance(value, str) or not value for value in product_ids)
        or len(set(product_ids)) != len(product_ids)
    ):
        raise ValueError("source current_product_ids are invalid")
    if any(Path(value).name != value or value in {".", ".."} for value in product_ids):
        raise ValueError("source product IDs must be plain directory names")
    if manifest.get("completed_product_ids") != product_ids:
        raise ValueError("source completed_product_ids differ from current_product_ids")
    if summary.get("current_product_ids") != product_ids:
        raise ValueError("source batch_summary product IDs differ from manifest")
    if int(summary.get("scene_count", -1)) != len(product_ids):
        raise ValueError("source batch_summary scene_count is inconsistent")
    config = summary.get("analysis_config")
    if not isinstance(config, dict) or config.get("save_score_maps") is not True:
        raise ValueError("source batch lacks saved score maps")
    return manifest, summary, product_ids


def derive_batch(
    source_batch: Path,
    output_batch: Path,
    *,
    minimum_threshold_coefficients: int = DWT_MINIMUM_COEFFICIENTS,
    slope_search_config: SceneSlopeSearchConfig | None = None,
) -> dict[str, Any]:
    source_batch = source_batch.expanduser().resolve()
    output_batch = output_batch.expanduser().resolve()
    if source_batch == output_batch or source_batch in output_batch.parents:
        raise ValueError("output batch must not be located inside the source batch")
    if minimum_threshold_coefficients < 1:
        raise ValueError("minimum_threshold_coefficients must be positive")
    search_config = slope_search_config or SceneSlopeSearchConfig()
    search_config.validate()
    output_batch.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output_batch):
        if output_batch.exists():
            raise FileExistsError(f"output batch already exists: {output_batch}")
        return _derive_batch_locked(
            source_batch,
            output_batch,
            minimum_threshold_coefficients=minimum_threshold_coefficients,
            slope_search_config=search_config,
        )


def _derive_batch_locked(
    source_batch: Path,
    output_batch: Path,
    *,
    minimum_threshold_coefficients: int,
    slope_search_config: SceneSlopeSearchConfig,
) -> dict[str, Any]:
    temporary = output_batch.with_name(
        f".{output_batch.name}.tmp-{uuid.uuid4().hex}"
    )
    started = _utc_now()
    temporary.mkdir()
    try:
        source_manifest, source_summary, product_ids = _validated_source_batch(
            source_batch
        )
        source_config = source_summary["analysis_config"]
        local_sigma = float(source_config.get("local_z_sigma", math.nan))
        if not np.isfinite(local_sigma) or local_sigma <= 0:
            raise ValueError("source local_z_sigma must be finite and positive")
        derived_config = dict(source_config)
        derived_config.update(
            {
                "posthoc_score_correction": "pdf_broad_dwt_then_thin_median",
                "posthoc_analysis_version": ANALYSIS_VERSION,
                "posthoc_input_score_keys": ["weak_z", "strong_z"],
                "posthoc_broad_slope_mode": "per_scene_shared_weak_strong",
                "posthoc_broad_slope_search": slope_search_config.to_dict(),
                "posthoc_reference_broad_slope": REFERENCE_BROAD_SLOPE,
                "posthoc_reference_broad_rotation_degrees": (
                    REFERENCE_BROAD_ROTATION_DEGREES
                ),
                "posthoc_dwt_levels": list(DWT_LEVELS),
                "posthoc_dwt_threshold_scale": DWT_THRESHOLD_SCALE,
                "posthoc_dwt_diff_fraction": DWT_DIFF_FRACTION,
                "posthoc_dwt_minimum_support_fraction": DWT_MINIMUM_SUPPORT,
                "posthoc_dwt_minimum_estimation_support_fraction": (
                    DWT_MINIMUM_ESTIMATION_SUPPORT
                ),
                "posthoc_dwt_minimum_threshold_coefficients": (
                    minimum_threshold_coefficients
                ),
                "posthoc_thin_slope": THIN_SLOPE,
                "posthoc_thin_bin_width": THIN_BIN_WIDTH,
                "posthoc_symmetric_protection": True,
            }
        )
        script_path = Path(__file__).resolve()
        slope_script_path = Path(__file__).with_name("scene_stripe_slope.py")
        provenance = {
            "analysis_version": ANALYSIS_VERSION,
            "source_batch": str(source_batch),
            "source_manifest_sha256": _sha256(source_batch / "run_manifest.json"),
            "source_batch_summary_sha256": _sha256(
                source_batch / "batch_summary.json"
            ),
            "script_path": str(script_path),
            "script_sha256": _sha256(script_path),
            "scene_stripe_slope_path": str(slope_script_path),
            "scene_stripe_slope_sha256": _sha256(slope_script_path),
            "directional_destriping_sha256": _sha256(
                Path(__file__).with_name("directional_destriping.py")
            ),
            "screen_hisui_l1g_scenes_sha256": _sha256(
                Path(__file__).with_name("screen_hisui_l1g_scenes.py")
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "minimum_threshold_coefficients": minimum_threshold_coefficients,
            "slope_search_config": slope_search_config.to_dict(),
            "started_utc": started,
        }
        scene_diagnostics: list[dict[str, Any]] = []
        all_slope_search_rows: list[dict[str, Any]] = []
        observed_quality: dict[str, int] = {}
        for product_id in product_ids:
            source_scene = source_batch / product_id
            scene_summary = _read_json(
                source_scene / "summary.json", label=f"{product_id} summary"
            )
            if scene_summary.get("product_id") != product_id:
                raise ValueError(f"{product_id}: source summary product mismatch")
            quality = str(scene_summary.get("quality_class"))
            observed_quality[quality] = observed_quality.get(quality, 0) + 1
            derived_scene = temporary / product_id
            derived_scene.mkdir()
            derived_scene_summary = dict(scene_summary)
            derived_scene_summary["analysis_config"] = derived_config
            derived_scene_summary["source_scene_directory"] = str(source_scene.resolve())
            derived_scene_summary["source_summary_sha256"] = _sha256(
                source_scene / "summary.json"
            )
            if quality == "usable":
                score_path = source_scene / "score_maps.npz"
                if not score_path.is_file():
                    raise FileNotFoundError(f"{product_id}: source score maps missing")
                with np.load(score_path, allow_pickle=False) as archive:
                    required = {
                        "valid",
                        "cloud_proxy",
                        "weak_z",
                        "strong_z",
                        "weak_local_z",
                        "strong_local_z",
                    }
                    missing = sorted(required - set(archive.files))
                    if missing:
                        raise ValueError(
                            f"{product_id}: source score maps lack {', '.join(missing)}"
                        )
                    valid = np.asarray(archive["valid"], dtype=bool)
                    cloud = np.asarray(archive["cloud_proxy"], dtype=bool)
                    weak_raw = np.asarray(archive["weak_z"], dtype=np.float32)
                    strong_raw = np.asarray(archive["strong_z"], dtype=np.float32)
                    weak_old_local = np.asarray(
                        archive["weak_local_z"], dtype=np.float32
                    )
                    strong_old_local = np.asarray(
                        archive["strong_local_z"], dtype=np.float32
                    )
                    weak_old_corrected = (
                        np.asarray(
                            archive["weak_z_directionally_destriped"],
                            dtype=np.float32,
                        )
                        if "weak_z_directionally_destriped" in archive.files
                        else None
                    )
                    strong_old_corrected = (
                        np.asarray(
                            archive["strong_z_directionally_destriped"],
                            dtype=np.float32,
                        )
                        if "strong_z_directionally_destriped" in archive.files
                        else None
                    )
                arrays = (
                    weak_raw,
                    strong_raw,
                    weak_old_local,
                    strong_old_local,
                )
                if any(array.shape != valid.shape for array in arrays):
                    raise ValueError(f"{product_id}: source score-map shape mismatch")
                if cloud.shape != valid.shape or np.any(cloud & valid):
                    raise ValueError(f"{product_id}: invalid cloud/valid masks")
                if any(not np.all(np.isfinite(array[valid])) for array in arrays):
                    raise ValueError(f"{product_id}: non-finite source scores on valid")
                for name, array in (
                    ("weak_z_directionally_destriped", weak_old_corrected),
                    ("strong_z_directionally_destriped", strong_old_corrected),
                ):
                    if array is not None and array.shape != valid.shape:
                        raise ValueError(f"{product_id}: {name} shape mismatch")
                    if array is not None and not np.all(np.isfinite(array[valid])):
                        raise ValueError(
                            f"{product_id}: non-finite {name} on valid pixels"
                        )
                protected, protection_diagnostics = symmetric_protection_mask(
                    weak_raw,
                    strong_raw,
                    valid,
                    local_sigma_pixels=local_sigma,
                )
                slope_diagnostics, slope_rows = estimate_scene_broad_slope(
                    {
                        "weak": weak_raw,
                        "strong": strong_raw,
                    },
                    valid,
                    protected,
                    config=slope_search_config,
                )
                scene_slope_rows = [
                    {"product_id": product_id, **row} for row in slope_rows
                ]
                write_slope_search_csv(
                    scene_slope_rows,
                    derived_scene / "broad_slope_search.csv",
                )
                all_slope_search_rows.extend(scene_slope_rows)
                broad_slope = float(
                    slope_diagnostics["selected_slope_row_per_column"]
                )
                broad_rotation_degrees = float(
                    slope_diagnostics["selected_angle_deg"]
                )
                apply_broad_dwt = (
                    slope_diagnostics.get("slope_status") == "supported"
                )
                canvas = safe_rotation_canvas(
                    valid.shape,
                    angle_degrees=broad_rotation_degrees,
                    divisor=2**DWT_MAX_LEVEL,
                )
                scene_started = time.perf_counter()
                weak_corrected, weak_local, weak_diagnostics = _process_band(
                    weak_raw,
                    weak_old_local,
                    weak_old_corrected,
                    valid,
                    protected,
                    local_sigma_pixels=local_sigma,
                    canvas_size=canvas,
                    band_name="weak_1580_1750_nm",
                    minimum_threshold_coefficients=minimum_threshold_coefficients,
                    broad_slope=broad_slope,
                    broad_rotation_degrees=broad_rotation_degrees,
                    apply_broad_dwt=apply_broad_dwt,
                )
                strong_corrected, strong_local, strong_diagnostics = _process_band(
                    strong_raw,
                    strong_old_local,
                    strong_old_corrected,
                    valid,
                    protected,
                    local_sigma_pixels=local_sigma,
                    canvas_size=canvas,
                    band_name="strong_2200_2390_nm",
                    minimum_threshold_coefficients=minimum_threshold_coefficients,
                    broad_slope=broad_slope,
                    broad_rotation_degrees=broad_rotation_degrees,
                    apply_broad_dwt=apply_broad_dwt,
                )
                dual_local = np.minimum(weak_local, strong_local).astype(np.float32)
                output_score_path = derived_scene / "score_maps.npz"
                np.savez_compressed(
                    output_score_path,
                    valid=valid,
                    cloud_proxy=cloud,
                    weak_local_z=weak_local,
                    strong_local_z=strong_local,
                    dual_local_z=dual_local,
                    weak_z_pdf_dwt_then_median=weak_corrected,
                    strong_z_pdf_dwt_then_median=strong_corrected,
                )
                diagnostic = {
                    "product_id": product_id,
                    "quality_class": quality,
                    "shape": list(valid.shape),
                    "analysis_valid_pixels": int(np.count_nonzero(valid)),
                    "safe_canvas_size": canvas,
                    "broad_slope_estimation": slope_diagnostics,
                    "broad_slope_search": str(
                        (
                            output_batch
                            / product_id
                            / "broad_slope_search.csv"
                        ).resolve()
                    ),
                    "broad_slope_search_sha256": _sha256(
                        derived_scene / "broad_slope_search.csv"
                    ),
                    "source_score_maps": str(score_path.resolve()),
                    "source_score_maps_sha256": _sha256(score_path),
                    "output_score_maps": str(
                        (output_batch / product_id / "score_maps.npz").resolve()
                    ),
                    "output_score_maps_sha256": _sha256(output_score_path),
                    "protection": protection_diagnostics,
                    "weak": weak_diagnostics,
                    "strong": strong_diagnostics,
                    "elapsed_seconds": float(time.perf_counter() - scene_started),
                }
                diagnostic_path = derived_scene / "pdf_dwt_diagnostics.json"
                _write_json(diagnostic_path, diagnostic)
                diagnostic["diagnostics_file"] = str(
                    (output_batch / product_id / diagnostic_path.name).resolve()
                )
                diagnostic["diagnostics_file_sha256"] = _sha256(diagnostic_path)
                derived_scene_summary["posthoc_score_correction"] = diagnostic
                scene_diagnostics.append(diagnostic)
                del (
                    weak_raw,
                    strong_raw,
                    weak_old_local,
                    strong_old_local,
                    weak_old_corrected,
                    strong_old_corrected,
                    weak_corrected,
                    strong_corrected,
                    weak_local,
                    strong_local,
                    dual_local,
                    protected,
                    slope_rows,
                    scene_slope_rows,
                )
                gc.collect()
            else:
                derived_scene_summary["posthoc_score_correction"] = {
                    "status": "not_processed",
                    "reason": f"quality_class={quality}",
                }
            _write_json(derived_scene / "summary.json", derived_scene_summary)

        expected_quality = {
            str(key): int(value)
            for key, value in source_summary.get("quality_class_counts", {}).items()
        }
        if observed_quality != expected_quality:
            raise ValueError("observed quality counts differ from source batch summary")
        finished = _utc_now()
        provenance["finished_utc"] = finished
        provenance["usable_scene_count"] = len(scene_diagnostics)
        provenance["scene_diagnostics"] = scene_diagnostics
        if all_slope_search_rows:
            write_slope_search_csv(
                all_slope_search_rows,
                temporary / "posthoc_pdf_dwt_slope_search.csv",
            )
        write_scene_metrics_csv(
            provenance, temporary / "posthoc_pdf_dwt_scene_metrics.csv"
        )
        write_scene_slopes_csv(
            provenance, temporary / "posthoc_pdf_dwt_scene_slopes.csv"
        )
        audit_names = (
            "posthoc_pdf_dwt_slope_search.csv",
            "posthoc_pdf_dwt_scene_metrics.csv",
            "posthoc_pdf_dwt_scene_slopes.csv",
        )
        provenance["audit_output_sha256"] = {
            name: _sha256(temporary / name)
            for name in audit_names
            if (temporary / name).is_file()
        }
        _write_json(temporary / "posthoc_pdf_dwt_summary.json", provenance)

        derived_batch_summary = dict(source_summary)
        derived_batch_summary["analysis_config"] = derived_config
        derived_batch_summary["source_batch"] = str(source_batch)
        derived_batch_summary["posthoc_score_correction"] = {
            "analysis_version": ANALYSIS_VERSION,
            "method": (
                "confidence-gated per-scene shared weak/strong broad-slope "
                "search, PDF DWT levels 3-5 when supported, then thin median "
                "slope 0.977346"
            ),
            "provenance": str(
                (output_batch / "posthoc_pdf_dwt_summary.json").resolve()
            ),
        }
        derived_batch_summary["outputs"] = {
            "run_manifest": str((output_batch / "run_manifest.json").resolve()),
            "posthoc_pdf_dwt_summary": str(
                (output_batch / "posthoc_pdf_dwt_summary.json").resolve()
            ),
            "posthoc_pdf_dwt_scene_metrics": str(
                (output_batch / "posthoc_pdf_dwt_scene_metrics.csv").resolve()
            ),
            "posthoc_pdf_dwt_scene_slopes": str(
                (output_batch / "posthoc_pdf_dwt_scene_slopes.csv").resolve()
            ),
            "posthoc_pdf_dwt_slope_search": str(
                (output_batch / "posthoc_pdf_dwt_slope_search.csv").resolve()
            ),
        }
        _write_json(temporary / "batch_summary.json", derived_batch_summary)
        derived_manifest = dict(source_manifest)
        derived_manifest.update(
            {
                "status": "complete",
                "started_utc": started,
                "finished_utc": finished,
                "output_dir": str(output_batch),
                "current_scene_directories": [
                    str((output_batch / product_id).resolve())
                    for product_id in product_ids
                ],
                "completed_product_ids": product_ids,
                "stale_scene_directories": [],
                "stale_scene_warning": None,
                "batch_summary": str(
                    (output_batch / "batch_summary.json").resolve()
                ),
                "source_batch": str(source_batch),
                "posthoc_analysis_version": ANALYSIS_VERSION,
            }
        )
        _write_json(temporary / "run_manifest.json", derived_manifest)
        temporary.replace(output_batch)
        return provenance
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def write_scene_slopes_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "product_id",
        "selected_angle_deg",
        "selected_slope_row_per_column",
        "selection_mode",
        "slope_status",
        "slope_support_failures",
        "selected_joint_fractional_gain",
        "selected_joint_absolute_gain",
        "selected_peak_prominence",
        "selected_peak_robust_z",
        "second_peak_prominence",
        "selected_at_search_edge",
        "weak_best_angle_deg",
        "strong_best_angle_deg",
        "per_band_angle_spread_deg",
        "protected_sample_pixels_excluded",
        "safe_canvas_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in summary.get("scene_diagnostics", []):
            slope = scene["broad_slope_estimation"]
            per_band = slope.get("per_band_best_angle_deg", {})
            writer.writerow(
                {
                    "product_id": scene["product_id"],
                    "selected_angle_deg": slope["selected_angle_deg"],
                    "selected_slope_row_per_column": slope[
                        "selected_slope_row_per_column"
                    ],
                    "selection_mode": slope["selection_mode"],
                    "slope_status": slope["slope_status"],
                    "slope_support_failures": ";".join(
                        slope.get("slope_support_failures", [])
                    ),
                    "selected_joint_fractional_gain": slope[
                        "selected_joint_fractional_gain"
                    ],
                    "selected_joint_absolute_gain": slope[
                        "selected_joint_absolute_gain"
                    ],
                    "selected_peak_prominence": slope[
                        "selected_peak_prominence"
                    ],
                    "selected_peak_robust_z": slope[
                        "selected_peak_robust_z"
                    ],
                    "second_peak_prominence": slope["second_peak_prominence"],
                    "selected_at_search_edge": slope["selected_at_search_edge"],
                    "weak_best_angle_deg": per_band.get("weak"),
                    "strong_best_angle_deg": per_band.get("strong"),
                    "per_band_angle_spread_deg": slope[
                        "per_band_angle_spread_deg"
                    ],
                    "protected_sample_pixels_excluded": slope[
                        "protected_sample_pixels_excluded"
                    ],
                    "safe_canvas_size": scene["safe_canvas_size"],
                }
            )


def write_scene_metrics_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "product_id",
        "band",
        "scene_broad_angle_deg",
        "scene_broad_slope_row_per_column",
        "source_positive_z3_pixels",
        "pdf_positive_z3_pixels",
        "source_reverse_z3_pixels",
        "pdf_reverse_z3_pixels",
        "source_broad_profile_rstd",
        "pdf_broad_profile_rstd",
        "source_thin_profile_rstd",
        "pdf_thin_profile_rstd",
        "valid_local_correlation",
        "protected_local_change_rms",
        "protected_absolute_local_change_q99",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in summary.get("scene_diagnostics", []):
            for band in ("weak", "strong"):
                comparison = scene[band]["comparison"]
                source_tail = comparison["source_profile_local_tail"]
                pdf_tail = comparison["pdf_dwt_local_tail"]
                writer.writerow(
                    {
                        "product_id": scene["product_id"],
                        "band": band,
                        "scene_broad_angle_deg": scene[
                            "broad_slope_estimation"
                        ]["selected_angle_deg"],
                        "scene_broad_slope_row_per_column": scene[
                            "broad_slope_estimation"
                        ]["selected_slope_row_per_column"],
                        "source_positive_z3_pixels": source_tail["positive_z3_pixels"],
                        "pdf_positive_z3_pixels": pdf_tail["positive_z3_pixels"],
                        "source_reverse_z3_pixels": source_tail["reverse_z3_pixels"],
                        "pdf_reverse_z3_pixels": pdf_tail["reverse_z3_pixels"],
                        "source_broad_profile_rstd": comparison.get(
                            "source_broad_profile_rstd"
                        ),
                        "pdf_broad_profile_rstd": comparison[
                            "pdf_broad_profile_rstd"
                        ],
                        "source_thin_profile_rstd": comparison.get(
                            "source_thin_profile_rstd"
                        ),
                        "pdf_thin_profile_rstd": comparison[
                            "pdf_thin_profile_rstd"
                        ],
                        "valid_local_correlation": comparison[
                            "valid_local_correlation"
                        ],
                        "protected_local_change_rms": comparison[
                            "protected_local_change_rms"
                        ],
                        "protected_absolute_local_change_q99": comparison[
                            "protected_absolute_local_change_q99"
                        ],
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a support-aware PDF-DWT derivative of a saved HISUI batch."
    )
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output-batch", type=Path, required=True)
    parser.add_argument(
        "--minimum-threshold-coefficients",
        type=int,
        default=DWT_MINIMUM_COEFFICIENTS,
        help=(
            "Minimum fully valid, non-protected coefficients required at each "
            "DWT level (default: 300; use 500 for the level-5-off sensitivity)."
        ),
    )
    defaults = SceneSlopeSearchConfig()
    parser.add_argument(
        "--slope-angle-min-deg", type=float, default=defaults.angle_min_deg
    )
    parser.add_argument(
        "--slope-angle-max-deg", type=float, default=defaults.angle_max_deg
    )
    parser.add_argument(
        "--slope-angle-step-deg", type=float, default=defaults.angle_step_deg
    )
    parser.add_argument(
        "--slope-search-positive-only",
        action="store_true",
        help="Disable the default negative-slope branch (not used for this reanalysis).",
    )
    parser.add_argument(
        "--slope-line-bin-width",
        type=float,
        default=defaults.line_bin_width_pixels,
    )
    parser.add_argument(
        "--slope-minimum-line-pixels",
        type=int,
        default=defaults.minimum_pixels_per_line,
    )
    parser.add_argument(
        "--slope-sample-step", type=int, default=defaults.sample_step
    )
    parser.add_argument(
        "--slope-trend-window-deg",
        type=float,
        default=defaults.trend_window_deg,
    )
    parser.add_argument(
        "--slope-local-window-deg",
        type=float,
        default=defaults.local_window_deg,
    )
    parser.add_argument(
        "--slope-edge-exclusion-deg",
        type=float,
        default=defaults.edge_exclusion_deg,
    )
    parser.add_argument(
        "--slope-minimum-peak-robust-z",
        type=float,
        default=defaults.minimum_peak_robust_z,
        help=(
            "Provisional no-stripe safety gate: selected prominence divided "
            "by its angular-curve robust scale (default: 5)."
        ),
    )
    parser.add_argument(
        "--slope-thin-exclusion-half-width-deg",
        type=float,
        default=defaults.excluded_half_width_deg,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    defaults = SceneSlopeSearchConfig()
    slope_search_config = SceneSlopeSearchConfig(
        angle_min_deg=args.slope_angle_min_deg,
        angle_max_deg=args.slope_angle_max_deg,
        angle_step_deg=args.slope_angle_step_deg,
        search_negative_slopes=not args.slope_search_positive_only,
        line_bin_width_pixels=args.slope_line_bin_width,
        minimum_pixels_per_line=args.slope_minimum_line_pixels,
        sample_step=args.slope_sample_step,
        trend_window_deg=args.slope_trend_window_deg,
        local_window_deg=args.slope_local_window_deg,
        edge_exclusion_deg=args.slope_edge_exclusion_deg,
        minimum_peak_robust_z=args.slope_minimum_peak_robust_z,
        excluded_angles_deg=defaults.excluded_angles_deg,
        excluded_half_width_deg=args.slope_thin_exclusion_half_width_deg,
    )
    summary = derive_batch(
        args.source_batch,
        args.output_batch,
        minimum_threshold_coefficients=args.minimum_threshold_coefficients,
        slope_search_config=slope_search_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()

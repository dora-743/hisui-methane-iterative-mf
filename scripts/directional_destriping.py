"""Directional stripe-removal helpers for MF score images.

The implementation follows the workflow documented in the supplied PDF and
the QA-guided 2200 nm experiment: rotate the stripe direction to horizontal,
soft-threshold selected Haar horizontal-detail levels, rotate back, and use a
fixed-slope line median for the remaining narrow stripes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def robust_std(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    median = np.median(arr)
    value = float(1.4826 * np.median(np.abs(arr - median)))
    return value if value > 0 else float(np.std(arr))


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def nearest_fill(image: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    invalid = np.asarray(invalid_mask, dtype=bool) | ~np.isfinite(image)
    if not invalid.any():
        return image.copy()
    valid = ~invalid
    if not valid.any():
        raise ValueError("Cannot fill an image with no finite unmasked pixels.")
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = image.copy()
    filled[invalid] = image[tuple(indices[:, invalid])]
    return filled


def central_reflect_pad(
    image: np.ndarray, canvas_size: int
) -> tuple[np.ndarray, tuple[slice, slice]]:
    height, width = image.shape
    if canvas_size < max(height, width):
        raise ValueError("canvas_size must be at least the largest image dimension.")
    y0 = (canvas_size - height) // 2
    x0 = (canvas_size - width) // 2
    y1 = canvas_size - height - y0
    x1 = canvas_size - width - x0
    padded = np.pad(image, ((y0, y1), (x0, x1)), mode="reflect")
    return padded, (slice(y0, y0 + height), slice(x0, x0 + width))


def haar_decompose2d(
    image: np.ndarray, levels: int
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    current = np.asarray(image, dtype=float)
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(levels):
        if current.shape[0] % 2 or current.shape[1] % 2:
            raise ValueError("Haar canvas dimensions must be divisible by 2**levels.")
        a = current[0::2, 0::2]
        b = current[0::2, 1::2]
        c = current[1::2, 0::2]
        d = current[1::2, 1::2]
        approximation = (a + b + c + d) / 2.0
        horizontal = (a + b - c - d) / 2.0
        vertical = (a - b + c - d) / 2.0
        diagonal = (a - b - c + d) / 2.0
        details.append((horizontal, vertical, diagonal))
        current = approximation
    return current, details


def haar_reconstruct2d(
    approximation: np.ndarray,
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    current = approximation
    for horizontal, vertical, diagonal in reversed(details):
        a = (current + horizontal + vertical + diagonal) / 2.0
        b = (current + horizontal - vertical - diagonal) / 2.0
        c = (current - horizontal + vertical - diagonal) / 2.0
        d = (current - horizontal - vertical + diagonal) / 2.0
        output = np.empty((current.shape[0] * 2, current.shape[1] * 2), dtype=float)
        output[0::2, 0::2] = a
        output[0::2, 1::2] = b
        output[1::2, 0::2] = c
        output[1::2, 1::2] = d
        current = output
    return current


def histogram_difference_threshold(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    bins: int = 128,
    diff_fraction: float = 0.25,
) -> float:
    target_abs = np.abs(target[np.isfinite(target)]).ravel()
    reference_abs = np.abs(reference[np.isfinite(reference)]).ravel()
    if target_abs.size == 0 or reference_abs.size == 0:
        return 0.0
    upper = float(np.percentile(np.concatenate([target_abs, reference_abs]), 99.5))
    if not np.isfinite(upper) or upper <= 0:
        return 0.0
    hist_target, edges = np.histogram(target_abs, bins=bins, range=(0, upper), density=True)
    hist_reference, _ = np.histogram(reference_abs, bins=bins, range=(0, upper), density=True)
    difference = hist_target - hist_reference
    maximum = float(np.max(difference))
    if not np.isfinite(maximum) or maximum <= 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    selected = np.flatnonzero(difference >= diff_fraction * maximum)
    if selected.size == 0:
        return 0.0
    return min(float(centers[selected[-1]]), float(np.percentile(target_abs, 90)))


def masked_histogram_difference_threshold(
    target: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
    *,
    bins: int = 128,
    diff_fraction: float = 0.25,
) -> float:
    """Histogram-difference threshold using only fully supported coefficients."""
    mask = np.asarray(support, dtype=bool)
    if mask.shape != np.asarray(target).shape or mask.shape != np.asarray(reference).shape:
        raise ValueError("support must match both coefficient arrays")
    target_abs = np.abs(np.asarray(target, dtype=float)[mask])
    reference_abs = np.abs(np.asarray(reference, dtype=float)[mask])
    target_abs = target_abs[np.isfinite(target_abs)]
    reference_abs = reference_abs[np.isfinite(reference_abs)]
    if target_abs.size == 0 or reference_abs.size == 0:
        return 0.0
    upper = float(np.percentile(np.concatenate([target_abs, reference_abs]), 99.5))
    if not np.isfinite(upper) or upper <= 0:
        return 0.0
    hist_target, edges = np.histogram(target_abs, bins=bins, range=(0, upper), density=True)
    hist_reference, _ = np.histogram(
        reference_abs, bins=bins, range=(0, upper), density=True
    )
    difference = hist_target - hist_reference
    maximum = float(np.max(difference))
    if not np.isfinite(maximum) or maximum <= 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    selected = np.flatnonzero(difference >= diff_fraction * maximum)
    if selected.size == 0:
        return 0.0
    return min(float(centers[selected[-1]]), float(np.percentile(target_abs, 90)))


def soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0 or not np.isfinite(threshold):
        return coefficients.copy()
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0.0)


def choose_rotation_angle(
    image: np.ndarray, slope: float, canvas_size: int
) -> tuple[float, dict[str, float]]:
    padded, _ = central_reflect_pad(image, canvas_size)
    theta = math.degrees(math.atan(slope))
    scores: dict[str, float] = {}
    best_angle = theta
    best_score = -np.inf
    for angle in (theta, -theta):
        rotated = ndimage.rotate(padded, angle=angle, reshape=False, order=1, mode="reflect")
        centered = rotated - np.median(rotated)
        row_profile = np.median(centered, axis=1)
        column_profile = np.median(centered, axis=0)
        score = robust_std(row_profile) / (robust_std(column_profile) + 1e-12)
        scores[f"{angle:.6f}"] = score
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle, scores


def wavelet_horizontal_destripe(
    image: np.ndarray,
    protected_mask: np.ndarray,
    *,
    slope: float,
    levels_to_filter: tuple[int, ...],
    max_level: int = 6,
    threshold_scale: float = 0.75,
    diff_fraction: float = 0.25,
    canvas_size: int = 512,
    operation_name: str = "directional_dwt",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    """Remove broad stripes using rotated directional Haar detail coefficients."""
    image = np.asarray(image, dtype=float)
    protected = ndimage.binary_dilation(np.asarray(protected_mask, dtype=bool), iterations=2)
    filled = nearest_fill(image, protected)
    padded, crop = central_reflect_pad(filled, canvas_size)
    angle, angle_scores = choose_rotation_angle(filled, slope, canvas_size)
    rotated = ndimage.rotate(padded, angle=angle, reshape=False, order=1, mode="reflect")
    approximation, details = haar_decompose2d(rotated, max_level)

    new_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rows: list[dict[str, float | int | str]] = []
    theta = math.degrees(math.atan(slope))
    for level, (horizontal, vertical, diagonal) in enumerate(details, start=1):
        if level in levels_to_filter:
            raw_threshold = histogram_difference_threshold(
                horizontal, vertical, diff_fraction=diff_fraction
            )
            applied_threshold = threshold_scale * raw_threshold
            filtered_horizontal = soft_threshold(horizontal, applied_threshold)
        else:
            raw_threshold = 0.0
            applied_threshold = 0.0
            filtered_horizontal = horizontal
        rows.append(
            {
                "operation": operation_name,
                "slope": slope,
                "chosen_rotation_deg": angle,
                "angle_score_positive": angle_scores.get(f"{theta:.6f}", float("nan")),
                "angle_score_negative": angle_scores.get(f"{-theta:.6f}", float("nan")),
                "level": level,
                "filtered": int(level in levels_to_filter),
                "raw_threshold": raw_threshold,
                "applied_threshold": applied_threshold,
                "horizontal_robust_std": robust_std(horizontal),
                "vertical_robust_std": robust_std(vertical),
            }
        )
        new_details.append((filtered_horizontal, vertical, diagonal))

    reconstructed = haar_reconstruct2d(approximation, new_details)
    stripe_rotated = rotated - reconstructed
    stripe_padded = ndimage.rotate(
        stripe_rotated, angle=-angle, reshape=False, order=1, mode="reflect"
    )
    stripe = stripe_padded[crop]
    corrected = image - stripe
    corrected[~np.isfinite(image)] = np.nan
    stripe[~np.isfinite(image)] = np.nan
    return corrected, stripe, rows


def _central_constant_pad(
    image: np.ndarray, canvas_size: int, *, fill_value: float = 0.0
) -> tuple[np.ndarray, tuple[slice, slice]]:
    height, width = image.shape
    if canvas_size < max(height, width):
        raise ValueError("canvas_size must be at least the largest image dimension")
    y0 = (canvas_size - height) // 2
    x0 = (canvas_size - width) // 2
    y1 = canvas_size - height - y0
    x1 = canvas_size - width - x0
    padded = np.pad(
        image,
        ((y0, y1), (x0, x1)),
        mode="constant",
        constant_values=fill_value,
    )
    return padded, (slice(y0, y0 + height), slice(x0, x0 + width))


def _haar_support_levels(support: np.ndarray, levels: int) -> list[np.ndarray]:
    current = np.asarray(support, dtype=float)
    outputs: list[np.ndarray] = []
    for _ in range(levels):
        if current.shape[0] % 2 or current.shape[1] % 2:
            raise ValueError("support canvas dimensions must be divisible by 2**levels")
        current = (
            current[0::2, 0::2]
            + current[0::2, 1::2]
            + current[1::2, 0::2]
            + current[1::2, 1::2]
        ) / 4.0
        outputs.append(current)
    return outputs


def support_aware_wavelet_horizontal_destripe(
    image: np.ndarray,
    valid_mask: np.ndarray,
    protected_mask: np.ndarray,
    *,
    slope: float,
    rotation_angle_deg: float,
    levels_to_filter: tuple[int, ...],
    max_level: int = 6,
    threshold_scale: float = 0.75,
    diff_fraction: float = 0.25,
    canvas_size: int,
    minimum_support_fraction: float = 0.95,
    minimum_estimation_support_fraction: float = 1.0,
    minimum_threshold_coefficients: int = 500,
    operation_name: str = "support_aware_directional_dwt",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str | None]]]:
    """Remove broad stripes without using invalid, protected, or pad coefficients.

    The image is filled only to make rotation and the Haar transform numerical.
    A separately rotated support mask is propagated through every Haar level.
    Thresholds are estimated from coefficients whose rotated coverage average
    meets ``minimum_estimation_support_fraction`` (normally 1.0) for valid,
    non-protected source pixels.  Thresholding is applied where the analogous
    valid-coverage average meets ``minimum_support_fraction`` (normally 0.95).
    Reflect padding therefore cannot set the coefficient threshold.
    """
    array = np.asarray(image, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    protected = np.asarray(protected_mask, dtype=bool)
    if array.ndim != 2 or valid.shape != array.shape or protected.shape != array.shape:
        raise ValueError("image, valid_mask, and protected_mask must share a 2-D shape")
    if not (0.0 < minimum_support_fraction <= 1.0):
        raise ValueError("minimum_support_fraction must be in (0, 1]")
    if not (0.0 < minimum_estimation_support_fraction <= 1.0):
        raise ValueError("minimum_estimation_support_fraction must be in (0, 1]")
    if minimum_threshold_coefficients < 1:
        raise ValueError("minimum_threshold_coefficients must be positive")
    if canvas_size % (2**max_level):
        raise ValueError("canvas_size must be divisible by 2**max_level")
    finite_valid = valid & np.isfinite(array)
    if not finite_valid.any():
        raise ValueError("no finite valid pixels are available")
    protected = protected & finite_valid
    filled = nearest_fill(array, (~finite_valid) | protected)
    padded, crop = central_reflect_pad(filled, canvas_size)

    valid_canvas, support_crop = _central_constant_pad(
        finite_valid.astype(float), canvas_size
    )
    threshold_canvas, _ = _central_constant_pad(
        (finite_valid & ~protected).astype(float), canvas_size
    )
    if support_crop != crop:
        raise RuntimeError("image and support padding disagree")

    rotated = ndimage.rotate(
        padded,
        angle=rotation_angle_deg,
        reshape=False,
        order=1,
        mode="reflect",
    )
    rotated_valid = np.clip(
        ndimage.rotate(
            valid_canvas,
            angle=rotation_angle_deg,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ),
        0.0,
        1.0,
    )
    rotated_threshold = np.clip(
        ndimage.rotate(
            threshold_canvas,
            angle=rotation_angle_deg,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ),
        0.0,
        1.0,
    )
    approximation, details = haar_decompose2d(rotated, max_level)
    valid_levels = _haar_support_levels(rotated_valid, max_level)
    threshold_levels = _haar_support_levels(rotated_threshold, max_level)

    new_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    diagnostics: list[dict[str, float | int | str | None]] = []
    for level, ((horizontal, vertical, diagonal), valid_fraction, threshold_fraction) in enumerate(
        zip(details, valid_levels, threshold_levels), start=1
    ):
        apply_support = valid_fraction >= minimum_support_fraction
        estimate_support = apply_support & (
            threshold_fraction
            >= minimum_estimation_support_fraction - 1.0e-12
        )
        eligible = int(np.count_nonzero(estimate_support))
        requested = level in levels_to_filter
        filtered = requested and eligible >= minimum_threshold_coefficients
        if filtered:
            raw_threshold = masked_histogram_difference_threshold(
                horizontal,
                vertical,
                estimate_support,
                diff_fraction=diff_fraction,
            )
            applied_threshold = threshold_scale * raw_threshold
            filtered_horizontal = horizontal.copy()
            filtered_horizontal[apply_support] = soft_threshold(
                horizontal[apply_support], applied_threshold
            )
            status = "filtered" if applied_threshold > 0 else "zero_threshold"
        else:
            raw_threshold = 0.0
            applied_threshold = 0.0
            filtered_horizontal = horizontal
            status = "insufficient_support" if requested else "not_requested"
        diagnostics.append(
            {
                "operation": operation_name,
                "slope": float(slope),
                "rotation_angle_deg": float(rotation_angle_deg),
                "level": level,
                "requested": int(requested),
                "filtered": int(filtered),
                "status": status,
                "raw_threshold": float(raw_threshold),
                "applied_threshold": float(applied_threshold),
                "estimate_coefficient_count": eligible,
                "apply_coefficient_count": int(np.count_nonzero(apply_support)),
                "total_coefficient_count": int(horizontal.size),
                "minimum_application_support_fraction": float(
                    minimum_support_fraction
                ),
                "minimum_estimation_support_fraction": float(
                    minimum_estimation_support_fraction
                ),
                "horizontal_robust_std_supported": _finite_or_none(
                    robust_std(horizontal, estimate_support)
                ),
                "vertical_robust_std_supported": _finite_or_none(
                    robust_std(vertical, estimate_support)
                ),
                "horizontal_energy_supported": float(
                    np.sum(horizontal[estimate_support] ** 2)
                ),
                "vertical_energy_supported": float(
                    np.sum(vertical[estimate_support] ** 2)
                ),
            }
        )
        new_details.append((filtered_horizontal, vertical, diagonal))

    reconstructed = haar_reconstruct2d(approximation, new_details)
    stripe_rotated = rotated - reconstructed
    stripe_padded = ndimage.rotate(
        stripe_rotated,
        angle=-rotation_angle_deg,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )
    stripe = stripe_padded[crop]
    corrected = array - stripe
    corrected[~finite_valid] = np.nan
    stripe[~finite_valid] = np.nan
    return corrected, stripe, diagnostics


def fixed_slope_line_ids(
    shape: tuple[int, int],
    *,
    slope: float,
    line_bin_width: float,
    direction_key: str,
) -> np.ndarray:
    rows, columns = np.indices(shape)
    if direction_key == "y_minus_x":
        coordinate = rows - slope * columns
    elif direction_key == "y_plus_x":
        coordinate = rows + slope * columns
    else:
        raise ValueError("direction_key must be 'y_minus_x' or 'y_plus_x'.")
    return np.rint(coordinate / line_bin_width).astype(np.int32)


def median_fixed_slope_destripe(
    image: np.ndarray,
    *,
    slope: float,
    line_bin_width: float = 2.0,
    min_pixels_per_line: int = 5,
    direction_key: str = "y_minus_x",
    valid_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    operation_name: str = "directional_median",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    """Subtract a robust offset estimated independently for fixed-slope lines."""
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    valid = finite if valid_mask is None else finite & np.asarray(valid_mask, dtype=bool)
    estimate = valid.copy()
    if exclude_mask is not None:
        estimate &= ~np.asarray(exclude_mask, dtype=bool)
    if not estimate.any():
        estimate = valid.copy()
    ids = fixed_slope_line_ids(
        image.shape,
        slope=slope,
        line_bin_width=line_bin_width,
        direction_key=direction_key,
    )
    global_median = float(np.median(image[estimate]))
    stripe = np.zeros_like(image)
    rows: list[dict[str, float | int | str]] = []
    for line_id in range(int(ids[valid].min()), int(ids[valid].max()) + 1):
        selected = (ids == line_id) & estimate
        fallback = False
        if selected.sum() < min_pixels_per_line:
            selected = (ids == line_id) & valid
            fallback = True
        count = int(selected.sum())
        if count >= min_pixels_per_line:
            line_median = float(np.median(image[selected]))
            offset = line_median - global_median
        else:
            line_median = float("nan")
            offset = 0.0
        stripe[ids == line_id] = offset
        rows.append(
            {
                "operation": operation_name,
                "direction_key": direction_key,
                "slope": slope,
                "signed_slope": slope if direction_key == "y_minus_x" else -slope,
                "line_bin_width": line_bin_width,
                "line_id": line_id,
                "n_pixels_used": count,
                "used_fallback": int(fallback),
                "line_median": line_median,
                "global_median": global_median,
                "stripe_offset_subtracted": offset,
            }
        )
    stripe[~valid] = np.nan
    corrected = image - stripe
    corrected[~finite] = np.nan
    return corrected, stripe, rows


def directional_profile(
    image: np.ndarray,
    *,
    slope: float,
    bin_width: float,
    protected_mask: np.ndarray | None = None,
    residual_sigma: float = 12.0,
    min_count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Median high-pass residual as a function of y - slope*x."""
    image = np.asarray(image, dtype=float)
    filled = nearest_fill(image, ~np.isfinite(image))
    residual = image - ndimage.gaussian_filter(filled, sigma=residual_sigma, mode="nearest")
    rows, columns = np.indices(image.shape)
    coordinate = rows - slope * columns
    valid = np.isfinite(residual)
    if protected_mask is not None:
        valid &= ~np.asarray(protected_mask, dtype=bool)
    if not valid.any():
        return np.array([]), np.array([])
    origin = float(coordinate[valid].min())
    bin_ids = np.floor((coordinate[valid] - origin) / bin_width).astype(int)
    x_values: list[float] = []
    y_values: list[float] = []
    for bin_id in range(int(bin_ids.max()) + 1):
        selected = bin_ids == bin_id
        if selected.sum() < min_count:
            continue
        x_values.append(float(np.median(coordinate[valid][selected])))
        y_values.append(float(np.median(residual[valid][selected])))
    return np.asarray(x_values), np.asarray(y_values)

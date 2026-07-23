#!/usr/bin/env python3
"""Iterative Matched Filter for the methane band near 1.6 micrometres.

The input formats are compatible with the files used by dora-743/MF:

* ROI CSV: ``y``, ``x``, and ``wave_XXXXnm`` columns.
* CH4 LUT CSV: one wavelength column followed by numeric enhancement columns.

The default MF window is 1580--1700 nm.  The methane unit absorption spectrum
(UAS) is estimated from the log-slope of the LUT spectra, after Gaussian sensor
spectral-response resampling.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_WL_MIN_NM = 1580.0
DEFAULT_WL_MAX_NM = 1700.0
DEFAULT_FWHM_NM = 12.5
WAVE_COLUMN_RE = re.compile(r"wave_([0-9]+(?:\.[0-9]+)?)nm", re.IGNORECASE)


@dataclass(frozen=True)
class RoiCube:
    """ROI spectra represented as a rectangular hyperspectral cube."""

    cube: np.ndarray
    wavelengths_nm: np.ndarray
    ys: np.ndarray
    xs: np.ndarray


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def get_wave_columns(columns: Sequence[object]) -> tuple[list[str], np.ndarray]:
    """Return wavelength-column names sorted by their wavelength in nm."""
    pairs: list[tuple[float, str]] = []
    for column in columns:
        name = str(column)
        match = WAVE_COLUMN_RE.fullmatch(name)
        if match:
            pairs.append((float(match.group(1)), name))

    if not pairs:
        raise ValueError("No columns matching 'wave_XXXXnm' were found in the ROI CSV.")

    pairs.sort(key=lambda item: item[0])
    wavelengths = np.asarray([item[0] for item in pairs], dtype=float)
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError("The ROI CSV contains duplicate center wavelengths.")
    return [item[1] for item in pairs], wavelengths


def load_roi_cube(path: str | Path) -> RoiCube:
    """Load a ``y,x,wave_*`` table and place its spectra into ``(H,W,B)``."""
    path = _as_path(path)
    df = pd.read_csv(path)
    if not {"y", "x"}.issubset(df.columns):
        raise ValueError("The ROI CSV must contain 'y' and 'x' columns.")
    if df.duplicated(["y", "x"]).any():
        duplicate = df.loc[df.duplicated(["y", "x"], keep=False), ["y", "x"]].iloc[0]
        raise ValueError(
            f"Duplicate pixel coordinate in ROI CSV: y={duplicate['y']}, x={duplicate['x']}"
        )

    wave_columns, wavelengths_nm = get_wave_columns(df.columns)
    spectra = df[wave_columns].to_numpy(dtype=float)
    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())

    y_index = np.searchsorted(ys, df["y"].to_numpy())
    x_index = np.searchsorted(xs, df["x"].to_numpy())
    cube = np.full((len(ys), len(xs), len(wave_columns)), np.nan, dtype=float)
    cube[y_index, x_index, :] = spectra
    return RoiCube(cube=cube, wavelengths_nm=wavelengths_nm, ys=ys, xs=xs)


def load_ch4_lut(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CH4 LUT as ``wavelength, alpha_0, alpha_1, ...``.

    The wavelength header may be ``wavelength``, ``wavelength_nm``, ``wave``,
    or ``waveln``.  If all wavelength values are below 20 they are interpreted
    as micrometres and converted to nanometres.
    """
    path = _as_path(path)
    df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise ValueError("The CH4 LUT must contain wavelength and at least two alpha columns.")

    aliases = {"wavelength", "wavelength_nm", "wave", "waveln", "lambda", "lambda_nm"}
    wavelength_column = next(
        (column for column in df.columns if str(column).strip().lower() in aliases),
        df.columns[0],
    )
    alpha_columns = [column for column in df.columns if column != wavelength_column]
    try:
        alpha_grid = np.asarray([float(column) for column in alpha_columns], dtype=float)
    except ValueError as exc:
        raise ValueError("All non-wavelength LUT column names must be numeric alpha values.") from exc

    wavelengths_nm = pd.to_numeric(df[wavelength_column], errors="raise").to_numpy(dtype=float)
    spectra = df[alpha_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float).T
    if not np.all(np.isfinite(wavelengths_nm)) or not np.all(np.isfinite(spectra)):
        raise ValueError("The CH4 LUT contains NaN or infinite values.")
    if np.nanmax(wavelengths_nm) <= 20.0:
        wavelengths_nm = wavelengths_nm * 1000.0

    wavelength_order = np.argsort(wavelengths_nm)
    alpha_order = np.argsort(alpha_grid)
    wavelengths_nm = wavelengths_nm[wavelength_order]
    alpha_grid = alpha_grid[alpha_order]
    spectra = spectra[alpha_order][:, wavelength_order]

    if np.any(np.diff(wavelengths_nm) <= 0):
        raise ValueError("CH4 LUT wavelengths must be unique.")
    if np.any(np.diff(alpha_grid) <= 0):
        raise ValueError("CH4 LUT alpha values must be unique.")
    if np.any(spectra <= 0):
        raise ValueError("CH4 LUT radiances must be positive for the log-slope calculation.")
    return wavelengths_nm, alpha_grid, spectra


def gaussian_srf_resample(
    source_wavelengths_nm: np.ndarray,
    source_spectra: np.ndarray,
    sensor_wavelengths_nm: np.ndarray,
    fwhm_nm: float | np.ndarray,
) -> np.ndarray:
    """Resample spectra using a Gaussian spectral response for each sensor band."""
    source_wavelengths_nm = np.asarray(source_wavelengths_nm, dtype=float)
    source_spectra = np.asarray(source_spectra, dtype=float)
    sensor_wavelengths_nm = np.asarray(sensor_wavelengths_nm, dtype=float)

    if source_spectra.ndim != 2 or source_spectra.shape[1] != len(source_wavelengths_nm):
        raise ValueError("source_spectra must have shape (n_spectra, n_source_wavelengths).")
    if sensor_wavelengths_nm.min() < source_wavelengths_nm.min() or sensor_wavelengths_nm.max() > source_wavelengths_nm.max():
        raise ValueError(
            "The CH4 LUT does not cover all ROI wavelengths: "
            f"LUT={source_wavelengths_nm.min():.2f}--{source_wavelengths_nm.max():.2f} nm, "
            f"ROI={sensor_wavelengths_nm.min():.2f}--{sensor_wavelengths_nm.max():.2f} nm."
        )

    if np.isscalar(fwhm_nm):
        fwhm = np.full(len(sensor_wavelengths_nm), float(fwhm_nm), dtype=float)
    else:
        fwhm = np.asarray(fwhm_nm, dtype=float)
        if fwhm.shape != sensor_wavelengths_nm.shape:
            raise ValueError("fwhm_nm must be scalar or match sensor_wavelengths_nm.")
    if np.any(fwhm <= 0):
        raise ValueError("Every FWHM value must be positive.")

    output = np.empty((source_spectra.shape[0], len(sensor_wavelengths_nm)), dtype=float)
    for band, (center, band_fwhm) in enumerate(zip(sensor_wavelengths_nm, fwhm)):
        sigma = band_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        use = np.abs(source_wavelengths_nm - center) <= 4.0 * sigma
        if np.count_nonzero(use) < 2:
            output[:, band] = np.asarray(
                [np.interp(center, source_wavelengths_nm, spectrum) for spectrum in source_spectra]
            )
            continue
        weights = np.exp(-0.5 * ((source_wavelengths_nm[use] - center) / sigma) ** 2)
        weights /= weights.sum()
        output[:, band] = source_spectra[:, use] @ weights
    return output


def compute_uas_log_slope(
    alpha_grid: np.ndarray,
    spectra_grid: np.ndarray,
    alpha_min: float | None = 0.0,
    alpha_max: float | None = 0.5,
) -> np.ndarray:
    """Estimate UAS as the negative log-radiance slope against LUT alpha."""
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    spectra_grid = np.asarray(spectra_grid, dtype=float)
    use = np.ones(len(alpha_grid), dtype=bool)
    if alpha_min is not None:
        use &= alpha_grid >= alpha_min
    if alpha_max is not None:
        use &= alpha_grid <= alpha_max
    if np.count_nonzero(use) < 2:
        raise ValueError("At least two LUT alpha levels are required in the UAS fitting range.")

    alpha = alpha_grid[use]
    log_spectra = np.log(spectra_grid[use])
    design = np.column_stack([np.ones(len(alpha)), alpha])
    coefficients, _, _, _ = np.linalg.lstsq(design, log_spectra, rcond=None)
    return -coefficients[1]


def select_wavelength_window(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    uas: np.ndarray,
    wl_min_nm: float,
    wl_max_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the 1.6 micrometre methane window from cube and UAS."""
    if wl_min_nm >= wl_max_nm:
        raise ValueError("wl_min_nm must be smaller than wl_max_nm.")
    mask = (wavelengths_nm >= wl_min_nm) & (wavelengths_nm <= wl_max_nm)
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            f"Only {np.count_nonzero(mask)} sensor bands fall in {wl_min_nm}--{wl_max_nm} nm; "
            "at least three are required."
        )
    return cube[:, :, mask], wavelengths_nm[mask], uas[mask]


def make_valid_mask(
    cube: np.ndarray,
    nodata_values: Sequence[float] = (0.0, -9999.0),
    require_positive: bool = True,
) -> np.ndarray:
    """Require a complete, finite selected-window spectrum at each valid pixel."""
    valid = np.all(np.isfinite(cube), axis=2)
    for value in nodata_values:
        valid &= ~np.any(np.isclose(cube, value, rtol=0.0, atol=0.0), axis=2)
    if require_positive:
        valid &= np.all(cube > 0, axis=2)
    return valid


def estimate_background(
    cube: np.ndarray,
    background_mask: np.ndarray,
    regularization: float,
    min_background_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate mean and a scale-aware ridge-regularized covariance matrix."""
    spectra = cube[background_mask]
    if len(spectra) < min_background_pixels:
        raise ValueError(
            f"Too few background pixels: {len(spectra)}; required >= {min_background_pixels}."
        )
    mean = spectra.mean(axis=0)
    covariance = np.atleast_2d(np.cov(spectra - mean, rowvar=False))
    variance_scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        variance_scale = 1.0
    covariance += regularization * variance_scale * np.eye(covariance.shape[0])
    return mean, covariance


def robust_threshold(values: np.ndarray, nsigma: float) -> tuple[float, float, float]:
    """Return median + nsigma * 1.4826 * MAD."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite MF values are available for thresholding.")
    median = float(np.median(values))
    robust_std = float(1.4826 * np.median(np.abs(values - median)))
    if robust_std == 0.0:
        robust_std = float(np.std(values))
    return median + nsigma * robust_std, median, robust_std


def iterative_matched_filter(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: np.ndarray,
    n_iter: int = 5,
    nsigma: float = 3.0,
    regularization: float = 1e-6,
    rcond: float = 1e-8,
    min_background_pixels: int | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Run Iterative MF, excluding robust high-score candidates each iteration."""
    if cube.ndim != 3:
        raise ValueError("cube must have shape (height, width, bands).")
    height, width, n_bands = cube.shape
    uas = np.asarray(uas, dtype=float).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(uas) != n_bands:
        raise ValueError("UAS length does not match the number of cube bands.")
    if valid_mask.shape != (height, width):
        raise ValueError("valid_mask shape does not match the cube.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least one.")
    if nsigma <= 0 or regularization < 0 or rcond <= 0:
        raise ValueError("nsigma and rcond must be positive; regularization must be non-negative.")
    if not np.all(np.isfinite(uas)) or np.linalg.norm(uas) <= np.finfo(float).eps:
        raise ValueError("The selected UAS is empty, non-finite, or numerically zero.")

    if min_background_pixels is None:
        min_background_pixels = max(n_bands + 5, 30)
    background_mask = valid_mask.copy()
    previous_plume: np.ndarray | None = None

    alpha_history: list[np.ndarray] = []
    plume_history: list[np.ndarray] = []
    background_history: list[np.ndarray] = []
    mean_history: list[np.ndarray] = []
    covariance_history: list[np.ndarray] = []
    iteration_rows: list[dict[str, float | int]] = []

    valid_spectra = cube[valid_mask]
    for iteration in range(1, n_iter + 1):
        background_history.append(background_mask.copy())
        mean, covariance = estimate_background(
            cube, background_mask, regularization, min_background_pixels
        )

        # Linearized methane target: delta L = -alpha * mean(L) * UAS.
        target = -mean * uas
        inverse_covariance = np.linalg.pinv(covariance, rcond=rcond)
        weighted_target = inverse_covariance @ target
        denominator = float(target @ weighted_target)
        if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
            raise ValueError("MF denominator is too small; inspect UAS and covariance conditioning.")

        alpha_map = np.full((height, width), np.nan, dtype=float)
        alpha_map[valid_mask] = (valid_spectra - mean) @ weighted_target / denominator
        threshold, median, robust_std = robust_threshold(alpha_map[valid_mask], nsigma)
        plume_mask = valid_mask & (alpha_map > threshold)
        next_background = valid_mask & ~plume_mask

        alpha_history.append(alpha_map.copy())
        plume_history.append(plume_mask.copy())
        mean_history.append(mean.copy())
        covariance_history.append(covariance.copy())
        iteration_rows.append(
            {
                "iteration": iteration,
                "background_pixels": int(background_mask.sum()),
                "threshold": float(threshold),
                "median": float(median),
                "robust_std": float(robust_std),
                "plume_pixels": int(plume_mask.sum()),
            }
        )

        if verbose:
            print(
                f"iter {iteration:02d}: background={int(background_mask.sum())}, "
                f"threshold={threshold:.6g}, median={median:.6g}, "
                f"robust_std={robust_std:.6g}, plume={int(plume_mask.sum())}"
            )
        background_mask = next_background
        if previous_plume is not None and np.array_equal(plume_mask, previous_plume):
            if verbose:
                print(f"Converged at iteration {iteration}.")
            break
        previous_plume = plume_mask.copy()

    return {
        "alpha_map": alpha_history[-1],
        "plume_mask": plume_history[-1],
        "background_mask": background_mask,
        "alpha_history": alpha_history,
        "plume_history": plume_history,
        "background_history": background_history,
        "mean_history": mean_history,
        "covariance_history": covariance_history,
        "iteration_table": pd.DataFrame(iteration_rows),
    }


def _save_diagnostic_figure(
    output_path: Path,
    result: dict[str, object],
    wavelengths_nm: np.ndarray,
    uas: np.ndarray,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    alpha_map = np.asarray(result["alpha_map"])
    plume_mask = np.asarray(result["plume_mask"])
    iteration_table = result["iteration_table"]
    if not isinstance(iteration_table, pd.DataFrame):
        raise TypeError("result['iteration_table'] must be a DataFrame.")

    valid_values = alpha_map[np.isfinite(alpha_map)]
    vmin, vmax = np.percentile(valid_values, [2, 98])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    image = axes[0, 0].imshow(alpha_map, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Iterative MF alpha (1580–1700 nm)")
    axes[0, 0].set_xlabel("column")
    axes[0, 0].set_ylabel("row")
    fig.colorbar(image, ax=axes[0, 0], label="alpha")

    axes[0, 1].imshow(plume_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Plume candidate mask")
    axes[0, 1].set_xlabel("column")
    axes[0, 1].set_ylabel("row")

    axes[1, 0].plot(wavelengths_nm, uas, marker="o")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Selected CH4 unit absorption spectrum")
    axes[1, 0].set_xlabel("Wavelength [nm]")
    axes[1, 0].set_ylabel("UAS per LUT alpha unit")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        iteration_table["iteration"], iteration_table["threshold"], marker="o"
    )
    axes[1, 1].set_title("Robust threshold history")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Threshold")
    axes[1, 1].grid(alpha=0.3)

    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def save_outputs(
    output_dir: str | Path,
    roi: RoiCube,
    selected_wavelengths_nm: np.ndarray,
    selected_uas: np.ndarray,
    valid_mask: np.ndarray,
    result: dict[str, object],
    metadata: dict[str, object],
    show_plot: bool = False,
) -> None:
    """Save maps, per-pixel tables, spectral target, history, and a QA figure."""
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_map = np.asarray(result["alpha_map"])
    plume_mask = np.asarray(result["plume_mask"], dtype=bool)
    background_mask = np.asarray(result["background_mask"], dtype=bool)
    row_index, column_index = np.indices(alpha_map.shape)

    pixel_table = pd.DataFrame(
        {
            "row": row_index.ravel(),
            "col": column_index.ravel(),
            "y": roi.ys[row_index.ravel()],
            "x": roi.xs[column_index.ravel()],
            "alpha": alpha_map.ravel(),
            "is_plume": plume_mask.ravel(),
            "is_background": background_mask.ravel(),
            "is_valid": valid_mask.ravel(),
        }
    ).sort_values("alpha", ascending=False, na_position="last")
    pixel_table.to_csv(output_dir / "imf_1600nm_pixel_results.csv", index=False)
    pixel_table[pixel_table["is_plume"]].to_csv(
        output_dir / "imf_1600nm_plume_pixels.csv", index=False
    )

    final_mean = np.asarray(result["mean_history"][-1])
    pd.DataFrame(
        {
            "wavelength_nm": selected_wavelengths_nm,
            "uas": selected_uas,
            "final_background_mean": final_mean,
            "final_mf_target": -final_mean * selected_uas,
        }
    ).to_csv(output_dir / "imf_1600nm_selected_target.csv", index=False)

    iteration_table = result["iteration_table"]
    if not isinstance(iteration_table, pd.DataFrame):
        raise TypeError("result['iteration_table'] must be a DataFrame.")
    iteration_table.to_csv(output_dir / "imf_1600nm_iteration_history.csv", index=False)

    np.save(output_dir / "imf_1600nm_alpha_map.npy", alpha_map)
    np.save(output_dir / "imf_1600nm_plume_mask.npy", plume_mask)
    np.save(output_dir / "imf_1600nm_background_mask.npy", background_mask)
    np.save(output_dir / "imf_1600nm_valid_mask.npy", valid_mask)
    with (output_dir / "imf_1600nm_run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    _save_diagnostic_figure(
        output_dir / "imf_1600nm_diagnostics.png",
        result,
        selected_wavelengths_nm,
        selected_uas,
        show=show_plot,
    )


def run_pipeline(
    roi_csv: str | Path,
    modtran_csv: str | Path,
    output_dir: str | Path,
    wl_min_nm: float = DEFAULT_WL_MIN_NM,
    wl_max_nm: float = DEFAULT_WL_MAX_NM,
    fwhm_nm: float = DEFAULT_FWHM_NM,
    alpha_min: float = 0.0,
    alpha_max: float = 0.5,
    n_iter: int = 5,
    nsigma: float = 3.0,
    regularization: float = 1e-6,
    rcond: float = 1e-8,
    nodata_values: Sequence[float] = (0.0, -9999.0),
    require_positive: bool = True,
    show_plot: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Run the complete 1600 nm Iterative MF workflow."""
    roi_path = _as_path(roi_csv)
    lut_path = _as_path(modtran_csv)
    output_path = _as_path(output_dir)

    roi = load_roi_cube(roi_path)
    lut_wavelengths, alpha_grid, lut_spectra = load_ch4_lut(lut_path)

    # Resampling all ROI bands first preserves the same processing order as the
    # referenced implementation; only then is the MF wavelength window selected.
    lut_at_sensor = gaussian_srf_resample(
        lut_wavelengths, lut_spectra, roi.wavelengths_nm, fwhm_nm
    )
    uas_all = compute_uas_log_slope(alpha_grid, lut_at_sensor, alpha_min, alpha_max)
    cube_selected, wavelengths_selected, uas_selected = select_wavelength_window(
        roi.cube, roi.wavelengths_nm, uas_all, wl_min_nm, wl_max_nm
    )
    valid_mask = make_valid_mask(cube_selected, nodata_values, require_positive)
    if valid_mask.sum() < max(cube_selected.shape[2] + 5, 30):
        raise ValueError(
            f"Only {int(valid_mask.sum())} valid pixels remain after masking; select a larger ROI."
        )

    if verbose:
        print(f"ROI cube: {roi.cube.shape}")
        print(
            f"Selected {len(wavelengths_selected)} bands: "
            f"{wavelengths_selected[0]:.2f}--{wavelengths_selected[-1]:.2f} nm"
        )
        print(f"Valid pixels: {int(valid_mask.sum())}/{valid_mask.size}")
        print(f"Selected UAS range: {uas_selected.min():.6g}--{uas_selected.max():.6g}")

    result = iterative_matched_filter(
        cube_selected,
        uas_selected,
        valid_mask,
        n_iter=n_iter,
        nsigma=nsigma,
        regularization=regularization,
        rcond=rcond,
        verbose=verbose,
    )
    iteration_table = result["iteration_table"]
    if not isinstance(iteration_table, pd.DataFrame):
        raise TypeError("result['iteration_table'] must be a DataFrame.")

    metadata: dict[str, object] = {
        "roi_csv": str(roi_path.resolve()),
        "modtran_csv": str(lut_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "wavelength_window_nm": [float(wl_min_nm), float(wl_max_nm)],
        "selected_sensor_wavelengths_nm": wavelengths_selected.tolist(),
        "selected_band_count": int(len(wavelengths_selected)),
        "sensor_fwhm_nm": float(fwhm_nm),
        "uas_alpha_fit_range": [float(alpha_min), float(alpha_max)],
        "iterations_requested": int(n_iter),
        "iterations_completed": int(len(iteration_table)),
        "nsigma": float(nsigma),
        "covariance_regularization": float(regularization),
        "pinv_rcond": float(rcond),
        "valid_pixels": int(valid_mask.sum()),
        "plume_candidate_pixels": int(np.asarray(result["plume_mask"]).sum()),
        "final_threshold": float(iteration_table.iloc[-1]["threshold"]),
        "alpha_unit_note": "Alpha inherits the enhancement unit used by the numeric LUT column headers.",
    }
    save_outputs(
        output_path,
        roi,
        wavelengths_selected,
        uas_selected,
        valid_mask,
        result,
        metadata,
        show_plot=show_plot,
    )
    if verbose:
        print(f"Saved outputs to: {output_path.resolve()}")
    return {
        "roi": roi,
        "wavelengths_selected": wavelengths_selected,
        "uas_selected": uas_selected,
        "valid_mask": valid_mask,
        "result": result,
        "metadata": metadata,
    }


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run methane Iterative MF with the 1580--1700 nm wavelength window."
    )
    parser.add_argument("--roi-csv", type=Path, required=True, help="y,x,wave_* ROI CSV.")
    parser.add_argument(
        "--modtran-csv", type=Path, required=True, help="CH4 LUT/MODTRAN spectra CSV."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/iterative_mf_1600nm")
    )
    parser.add_argument("--wl-min", type=float, default=DEFAULT_WL_MIN_NM)
    parser.add_argument("--wl-max", type=float, default=DEFAULT_WL_MAX_NM)
    parser.add_argument("--fwhm", type=float, default=DEFAULT_FWHM_NM)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--nsigma", type=float, default=3.0)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--rcond", type=float, default=1e-8)
    parser.add_argument(
        "--nodata", type=_parse_float_list, default=(0.0, -9999.0), help="Comma-separated values."
    )
    parser.add_argument(
        "--require-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require positive radiance in every selected band.",
    )
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    return run_pipeline(
        roi_csv=args.roi_csv,
        modtran_csv=args.modtran_csv,
        output_dir=args.output_dir,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        fwhm_nm=args.fwhm,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_iter=args.n_iter,
        nsigma=args.nsigma,
        regularization=args.regularization,
        rcond=args.rcond,
        nodata_values=args.nodata,
        require_positive=args.require_positive,
        show_plot=args.show_plot,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

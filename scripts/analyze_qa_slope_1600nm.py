#!/usr/bin/env python3
"""Estimate 1600 nm sensor-line slopes from HISUI QA_IM and QA_DM."""

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
import tifffile


DEFAULT_QA_DIR = Path("data/qa")
DEFAULT_OUTPUT_DIR = Path("outputs/qa_1600nm_slope")


def find_one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern!r} in {directory}; found {len(matches)}."
        )
    return matches[0]


def fit_trace(mask: np.ndarray) -> dict[str, float | int]:
    rows, columns = np.nonzero(mask)
    output: dict[str, float | int] = {"count": int(len(columns))}
    if len(columns) < 2:
        output.update(
            {
                "pca_slope_row_per_col_abs": float("nan"),
                "pca_angle_deg": float("nan"),
                "binned_median_fit_slope": float("nan"),
                "binned_median_fit_r2": float("nan"),
                "row_min": -1,
                "row_max": -1,
                "col_min": -1,
                "col_max": -1,
            }
        )
        return output

    centered = np.column_stack([columns - columns.mean(), rows - rows.mean()])
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    vector_col, vector_row = vectors[0]
    pca_slope = abs(float(vector_row / vector_col))

    bin_ids = columns // 10
    binned_columns: list[float] = []
    binned_rows: list[float] = []
    for bin_id in np.unique(bin_ids):
        selected = bin_ids == bin_id
        if selected.sum() >= 2:
            binned_columns.append(float(np.median(columns[selected])))
            binned_rows.append(float(np.median(rows[selected])))
    if len(binned_columns) >= 2:
        coefficients = np.polyfit(binned_columns, binned_rows, 1)
        predicted = np.polyval(coefficients, binned_columns)
        residual_sum = float(np.sum((np.asarray(binned_rows) - predicted) ** 2))
        total_sum = float(
            np.sum((np.asarray(binned_rows) - np.mean(binned_rows)) ** 2)
        )
        median_slope = abs(float(coefficients[0]))
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    else:
        median_slope = float("nan")
        r_squared = float("nan")

    output.update(
        {
            "pca_slope_row_per_col_abs": pca_slope,
            "pca_angle_deg": float(np.degrees(np.arctan(pca_slope))),
            "binned_median_fit_slope": median_slope,
            "binned_median_fit_r2": r_squared,
            "row_min": int(rows.min()),
            "row_max": int(rows.max()),
            "col_min": int(columns.min()),
            "col_max": int(columns.max()),
        }
    )
    return output


def analyze_qa_map(
    path: Path,
    selected_indices: np.ndarray,
    band_table: pd.DataFrame,
    qa_name: str,
) -> tuple[list[dict[str, float | int | str]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    # The TIFF stores all 185 samples in one page. Reading once is considerably
    # faster than decoding the compressed page separately for every band.
    qa_cube = tifffile.imread(path)
    rows_out: list[dict[str, float | int | str]] = []
    trace_coordinates: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in selected_indices:
        mask = np.asarray(qa_cube[:, :, index], dtype=bool)
        fit = fit_trace(mask)
        rows, columns = np.nonzero(mask)
        if len(columns):
            trace_coordinates[int(index)] = (rows, columns)
        row: dict[str, float | int | str] = {
            "qa_map": qa_name,
            "band": int(band_table.iloc[index]["BandNo"]),
            "wavelength_nm": float(
                band_table.iloc[index]["CenterWavelengthNanometer"]
            ),
        }
        row.update(fit)
        rows_out.append(row)
    del qa_cube
    return rows_out, trace_coordinates


def save_trace_plot(
    path: Path,
    trace_coordinates: dict[int, tuple[np.ndarray, np.ndarray]],
    band_table: pd.DataFrame,
) -> None:
    if not trace_coordinates:
        return
    fig, axes = plt.subplots(
        1, len(trace_coordinates), figsize=(3.5 * len(trace_coordinates), 4), constrained_layout=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, (index, (rows, columns)) in zip(axes_flat, trace_coordinates.items()):
        axis.scatter(columns, rows, s=0.7, alpha=0.45, rasterized=True)
        centered = np.column_stack([columns - columns.mean(), rows - rows.mean()])
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        slope = float(vectors[0, 1] / vectors[0, 0])
        intercept = float(rows.mean() - slope * columns.mean())
        x_line = np.array([columns.min(), columns.max()])
        axis.plot(x_line, slope * x_line + intercept, color="red", linewidth=1)
        band = int(band_table.iloc[index]["BandNo"])
        wavelength = float(band_table.iloc[index]["CenterWavelengthNanometer"])
        axis.set_title(f"QA_DM band {band}\n{wavelength:.3f} nm")
        axis.set_xlabel("full-scene column")
        axis.set_ylabel("full-scene row")
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    fig.suptitle("1600 nm QA_DM line traces and PCA fits")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> pd.DataFrame:
    qa_dir = args.qa_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    band_table = pd.read_csv(find_one(qa_dir, "*_B.csv"))
    wavelengths = band_table["CenterWavelengthNanometer"].to_numpy(dtype=float)
    selected = np.flatnonzero((wavelengths >= args.wl_min) & (wavelengths <= args.wl_max))
    if selected.size == 0:
        raise ValueError("No bands fall in the requested wavelength interval.")

    im_rows, _ = analyze_qa_map(
        find_one(qa_dir, "*_QA_IM.tif"), selected, band_table, "QA_IM"
    )
    dm_rows, dm_coordinates = analyze_qa_map(
        find_one(qa_dir, "*_QA_DM.tif"), selected, band_table, "QA_DM"
    )
    table = pd.DataFrame(im_rows + dm_rows)
    table.to_csv(output_dir / "qa_1600nm_band_slopes.csv", index=False)
    save_trace_plot(output_dir / "qa_dm_1600nm_line_traces.png", dm_coordinates, band_table)

    usable = table[
        (table["qa_map"] == "QA_DM")
        & np.isfinite(table["pca_slope_row_per_col_abs"])
    ]
    summary = {
        "qa_directory": str(qa_dir),
        "wavelength_window_nm": [args.wl_min, args.wl_max],
        "selected_bands": table[table["qa_map"] == "QA_IM"]["band"].tolist(),
        "selected_wavelengths_nm": table[table["qa_map"] == "QA_IM"][
            "wavelength_nm"
        ].tolist(),
        "qa_im_trace_bands": int(
            np.sum((table["qa_map"] == "QA_IM") & (table["count"] > 0))
        ),
        "qa_dm_trace_bands": int(len(usable)),
        "qa_dm_pca_slope_mean": float(usable["pca_slope_row_per_col_abs"].mean()),
        "qa_dm_pca_slope_sample_std": float(
            usable["pca_slope_row_per_col_abs"].std(ddof=1)
        ),
        "qa_dm_binned_median_slope_mean": float(
            usable["binned_median_fit_slope"].mean()
        ),
        "qa_dm_angle_deg_from_mean_pca_slope": float(
            np.degrees(np.arctan(usable["pca_slope_row_per_col_abs"].mean()))
        ),
    }
    with (output_dir / "qa_1600nm_slope_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wl-min", type=float, default=1580.0)
    parser.add_argument("--wl-max", type=float, default=1700.0)
    return parser


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()

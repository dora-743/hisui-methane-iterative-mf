from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_hisui_repeat_acquisitions import (  # noqa: E402
    MAP_CONFIG_KEYS,
    ScoreMaps,
    _prepare_output_paths,
    common_aligned_overlap,
    comparison_status,
    load_score_maps,
    repeat_metrics,
    run,
    save_repeat_comparison_figure,
    validate_pair_provenance,
)
from hisui_l1g_io import GeoReference  # noqa: E402


def _georef(origin_x: float, origin_y: float) -> GeoReference:
    return GeoReference(
        geotransform=(origin_x, 20.0, 0.0, origin_y, 0.0, -20.0),
        crs="EPSG:32613",
        epsg=32613,
        pixel_scale=(20.0, 20.0, 0.0),
        tiepoint=(0.0,) * 6,
        raster_type=2,
    )


def _analysis_config() -> dict[str, object]:
    config = {key: index for index, key in enumerate(MAP_CONFIG_KEYS)}
    config["cloud_thresholds"] = {"cirrus_1388": 0.1, "bright_665": 0.4}
    return config


def _target_provenance(tag: str = "same") -> dict[str, object]:
    return {
        "modtran_path": f"{tag}.csv",
        "modtran_size_bytes": 10,
        "modtran_sha256": f"lut-{tag}",
        "spectral_grid_sha256": "grid-same",
        "target_inputs_sha256": f"target-{tag}",
    }


class CompareHISUIRepeatAcquisitionsTests(unittest.TestCase):
    def test_nonfinite_threshold_invalidates_outputs_from_an_earlier_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = output / "repeat_acquisition_summary.json"
            figure = output / "repeat_acquisition_comparison.png"
            summary.write_text("old", encoding="utf-8")
            figure.write_bytes(b"old")

            with self.assertRaisesRegex(ValueError, "threshold must be finite"):
                run(argparse.Namespace(output_dir=output, threshold=np.nan))

            self.assertFalse(summary.exists())
            self.assertFalse(figure.exists())

    def test_shifted_grids_align_matching_pixel_centres(self) -> None:
        first = _georef(100.0, 200.0)
        second = _georef(120.0, 180.0)
        alignment = common_aligned_overlap(first, (4, 5), second, (4, 5))
        self.assertIsNotNone(alignment)
        assert alignment is not None
        self.assertEqual(alignment.first_bounds, (1, 4, 1, 5))
        self.assertEqual(alignment.second_bounds, (0, 3, 0, 4))
        self.assertEqual(alignment.shape, (3, 4))
        self.assertEqual(
            first.map_xy(1, 1, pixel_center=True),
            second.map_xy(0, 0, pixel_center=True),
        )
        self.assertAlmostEqual(alignment.maximum_pixel_center_residual_m, 0.0)

    def test_disjoint_grids_return_no_overlap(self) -> None:
        first = _georef(100.0, 200.0)
        second = _georef(1000.0, 200.0)
        self.assertIsNone(common_aligned_overlap(first, (4, 5), second, (4, 5)))

    def test_half_pixel_shift_is_rejected_instead_of_silently_rounded(self) -> None:
        first = _georef(100.0, 200.0)
        second = _georef(110.0, 200.0)
        with self.assertRaisesRegex(ValueError, "pixel boundaries"):
            common_aligned_overlap(first, (4, 5), second, (4, 5))

    def test_cloud_aware_fixed_location_tail_metrics(self) -> None:
        first = np.asarray([[4.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        second = np.asarray([[4.5, 0.0, 4.0], [0.0, 0.0, 0.0]])
        first_clear = np.ones((2, 3), dtype=bool)
        second_clear = np.ones((2, 3), dtype=bool)
        second_clear[1, 2] = False
        metrics = repeat_metrics(
            first, second, first_clear, second_clear, threshold=3.0
        )
        tails = metrics["fixed_location_high_tail"]
        self.assertEqual(metrics["clear_masks"]["common_clear_pixels"], 5)
        self.assertEqual(tails["first_pixels"], 2)
        self.assertEqual(tails["second_pixels"], 2)
        self.assertEqual(tails["overlap_pixels"], 1)
        self.assertEqual(tails["union_pixels"], 3)
        self.assertAlmostEqual(tails["jaccard"], 1.0 / 3.0)

    def test_no_tail_and_constant_maps_emit_json_safe_nulls(self) -> None:
        first = np.ones((2, 2), dtype=float)
        second = np.ones((2, 2), dtype=float) * 2.0
        clear = np.ones((2, 2), dtype=bool)
        metrics = repeat_metrics(first, second, clear, clear, threshold=3.0)
        self.assertIsNone(metrics["paired_dual_local_z"]["pearson_r"])
        self.assertEqual(metrics["fixed_location_high_tail"]["union_pixels"], 0)
        self.assertIsNone(metrics["fixed_location_high_tail"]["jaccard"])

    def test_no_common_clear_pixels_are_handled(self) -> None:
        values = np.arange(4, dtype=float).reshape(2, 2)
        metrics = repeat_metrics(
            values,
            values,
            np.zeros((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
        )
        self.assertEqual(metrics["clear_masks"]["common_clear_pixels"], 0)
        self.assertIsNone(metrics["paired_dual_local_z"]["pearson_r"])
        self.assertIsNone(metrics["fixed_location_high_tail"]["jaccard"])
        self.assertEqual(comparison_status(metrics), "no_finite_common_clear_scores")

    def test_overlapping_clear_masks_with_nan_scores_have_distinct_status(self) -> None:
        values = np.full((2, 2), np.nan)
        clear = np.ones((2, 2), dtype=bool)
        metrics = repeat_metrics(values, values, clear, clear)
        self.assertEqual(metrics["clear_masks"]["common_clear_pixels"], 4)
        self.assertEqual(
            metrics["paired_dual_local_z"]["finite_common_clear_pixels"], 0
        )
        self.assertEqual(comparison_status(metrics), "no_finite_common_clear_scores")

    def test_empty_aligned_arrays_are_rejected(self) -> None:
        empty = np.empty((0, 3), dtype=float)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            repeat_metrics(empty, empty, empty.astype(bool), empty.astype(bool))

    def test_score_archive_can_be_loaded_from_scene_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            lut = scene / "lut.csv"
            lut.write_text("wavelength,1,2\n1600,1,0.9\n", encoding="utf-8")
            np.savez_compressed(
                scene / "score_maps.npz",
                valid=np.ones((2, 3), dtype=bool),
                dual_local_z=np.arange(6, dtype=float).reshape(2, 3),
            )
            (scene / "summary.json").write_text(
                json.dumps(
                    {
                        "product_id": "HSHL1G_test",
                        "acquisition_utc": "2022-01-01T00:00:00Z",
                        "analysis_config": _analysis_config(),
                        "model": {"modtran_csv": str(lut)},
                        "selected_wavelengths_nm": [1600.0, 1610.0],
                        "selected_fwhm_nm": [10.0, 10.0],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_score_maps(scene)
            self.assertEqual(loaded.valid.shape, (2, 3))
            self.assertEqual(loaded.path, (scene / "score_maps.npz").resolve())
            self.assertEqual(loaded.product_id, "HSHL1G_test")
            self.assertEqual(len(loaded.target_provenance["modtran_sha256"]), 64)

    def test_malformed_clear_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            np.savez_compressed(
                scene / "score_maps.npz",
                valid=np.asarray([[1.0, np.nan]]),
                dual_local_z=np.ones((1, 2)),
            )
            (scene / "summary.json").write_text(
                json.dumps(
                    {
                        "product_id": "HSHL1G_test",
                        "analysis_config": _analysis_config(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite 0/1"):
                load_score_maps(scene)

    def test_product_id_provenance_mismatch_is_rejected(self) -> None:
        arrays = np.ones((2, 2))
        first = ScoreMaps(
            path=Path("first.npz"),
            summary_path=Path("first-summary.json"),
            product_id="wrong",
            acquisition_utc="2022-01-01T00:00:00Z",
            quality_class="usable",
            analysis_config=_analysis_config(),
            target_provenance=_target_provenance(),
            valid=arrays.astype(bool),
            dual_local_z=arrays,
        )
        second = ScoreMaps(
            path=Path("second.npz"),
            summary_path=Path("second-summary.json"),
            product_id="second",
            acquisition_utc="2023-01-01T00:00:00Z",
            quality_class="usable",
            analysis_config=_analysis_config(),
            target_provenance=_target_provenance(),
            valid=arrays.astype(bool),
            dual_local_z=arrays,
        )
        with self.assertRaisesRegex(ValueError, "does not match product"):
            validate_pair_provenance(
                first,
                second,
                first_product_id="first",
                second_product_id="second",
                first_acquisition_utc="2022-01-01T00:00:00Z",
                second_acquisition_utc="2023-01-01T00:00:00Z",
            )

    def test_map_affecting_configuration_mismatch_is_rejected(self) -> None:
        arrays = np.ones((2, 2))
        first_config = _analysis_config()
        second_config = _analysis_config()
        second_config["cloud_profile"] = "different"
        common = {
            "quality_class": "usable",
            "target_provenance": _target_provenance(),
            "valid": arrays.astype(bool),
            "dual_local_z": arrays,
        }
        first = ScoreMaps(
            path=Path("first.npz"),
            summary_path=Path("first-summary.json"),
            product_id="first",
            acquisition_utc="2022-01-01T00:00:00Z",
            analysis_config=first_config,
            **common,
        )
        second = ScoreMaps(
            path=Path("second.npz"),
            summary_path=Path("second-summary.json"),
            product_id="second",
            acquisition_utc="2023-01-01T00:00:00Z",
            analysis_config=second_config,
            **common,
        )
        with self.assertRaisesRegex(ValueError, "cloud_profile"):
            validate_pair_provenance(
                first,
                second,
                first_product_id="first",
                second_product_id="second",
                first_acquisition_utc="2022-01-01T00:00:00Z",
                second_acquisition_utc="2023-01-01T00:00:00Z",
            )

    def test_stored_cloud_threshold_mismatch_is_rejected(self) -> None:
        arrays = np.ones((2, 2))
        first_config = _analysis_config()
        second_config = _analysis_config()
        second_config["cloud_thresholds"] = {
            **second_config["cloud_thresholds"],
            "cirrus_1388": 0.2,
        }
        common = {
            "quality_class": "usable",
            "target_provenance": _target_provenance(),
            "valid": arrays.astype(bool),
            "dual_local_z": arrays,
        }
        first = ScoreMaps(
            path=Path("first.npz"),
            summary_path=Path("first-summary.json"),
            product_id="first",
            acquisition_utc="2022-01-01T00:00:00Z",
            analysis_config=first_config,
            **common,
        )
        second = ScoreMaps(
            path=Path("second.npz"),
            summary_path=Path("second-summary.json"),
            product_id="second",
            acquisition_utc="2023-01-01T00:00:00Z",
            analysis_config=second_config,
            **common,
        )
        with self.assertRaisesRegex(ValueError, "cloud_thresholds"):
            validate_pair_provenance(
                first,
                second,
                first_product_id="first",
                second_product_id="second",
                first_acquisition_utc="2022-01-01T00:00:00Z",
                second_acquisition_utc="2023-01-01T00:00:00Z",
            )

    def test_modtran_target_provenance_mismatch_is_rejected(self) -> None:
        arrays = np.ones((2, 2))
        common = {
            "quality_class": "usable",
            "analysis_config": _analysis_config(),
            "valid": arrays.astype(bool),
            "dual_local_z": arrays,
        }
        first = ScoreMaps(
            path=Path("first.npz"),
            summary_path=Path("first-summary.json"),
            product_id="first",
            acquisition_utc="2022-01-01T00:00:00Z",
            target_provenance=_target_provenance("first"),
            **common,
        )
        second = ScoreMaps(
            path=Path("second.npz"),
            summary_path=Path("second-summary.json"),
            product_id="second",
            acquisition_utc="2023-01-01T00:00:00Z",
            target_provenance=_target_provenance("second"),
            **common,
        )
        with self.assertRaisesRegex(ValueError, "MODTRAN/target provenance"):
            validate_pair_provenance(
                first,
                second,
                first_product_id="first",
                second_product_id="second",
                first_acquisition_utc="2022-01-01T00:00:00Z",
                second_acquisition_utc="2023-01-01T00:00:00Z",
            )

    def test_output_preparation_removes_both_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = output / "repeat_acquisition_summary.json"
            figure = output / "repeat_acquisition_comparison.png"
            summary.write_text("old", encoding="utf-8")
            figure.write_bytes(b"old")
            destination, returned_summary, returned_figure = _prepare_output_paths(output)
            self.assertEqual(destination, output.resolve())
            self.assertEqual(returned_summary, summary.resolve())
            self.assertEqual(returned_figure, figure.resolve())
            self.assertFalse(summary.exists())
            self.assertFalse(figure.exists())

    def test_synthetic_no_tail_figure_is_created(self) -> None:
        first = np.ones((4, 5), dtype=float)
        second = np.ones((4, 5), dtype=float) * 2.0
        clear = np.ones((4, 5), dtype=bool)
        metrics = repeat_metrics(first, second, clear, clear, threshold=3.0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.png"
            save_repeat_comparison_figure(
                output,
                first,
                second,
                clear,
                clear,
                first_label="2021-01-01",
                second_label="2022-01-01",
                threshold=3.0,
                epsg=32613,
                metrics=metrics,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

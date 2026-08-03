from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from directional_destriping import (  # noqa: E402
    masked_histogram_difference_threshold,
    support_aware_wavelet_horizontal_destripe,
)
from postprocess_score_maps_pdf_dwt import (  # noqa: E402
    _process_band,
    derive_batch,
    fast_fixed_slope_median_destripe,
    safe_rotation_canvas,
    symmetric_protection_mask,
)
from scene_stripe_slope import (  # noqa: E402
    SceneSlopeSearchConfig,
    estimate_scene_broad_slope,
)
from compare_pdf_dwt_reanalysis import build_aggregate_rows  # noqa: E402


PDF_BROAD_SLOPE = 1.257172298918948
PDF_ROTATION_DEGREES = math.degrees(math.atan(PDF_BROAD_SLOPE))


class PDFDWTReanalysisTests(unittest.TestCase):
    @staticmethod
    def _synthetic_slope_config() -> SceneSlopeSearchConfig:
        return SceneSlopeSearchConfig(
            angle_min_deg=35.0,
            angle_max_deg=60.0,
            angle_step_deg=0.5,
            search_negative_slopes=True,
            line_bin_width_pixels=12.0,
            minimum_pixels_per_line=12,
            sample_step=1,
            trend_window_deg=6.0,
            local_window_deg=1.0,
            edge_exclusion_deg=1.0,
            excluded_angles_deg=(44.3,),
            excluded_half_width_deg=1.0,
        )

    def test_scene_slope_estimator_recovers_signed_shared_geometry(self) -> None:
        height, width = 128, 120
        rows, columns = np.indices((height, width))
        true_angle = -40.5
        true_slope = math.tan(math.radians(true_angle))
        coordinate = rows - true_slope * columns
        stripe = np.sin(2.0 * np.pi * coordinate / 24.0)
        stripe += 0.4 * np.sin(2.0 * np.pi * coordinate / 48.0)
        plume = 7.0 * np.exp(
            -((rows - 62.0) ** 2 + (columns - 58.0) ** 2) / (2.0 * 3.0**2)
        )
        valid = np.ones((height, width), dtype=bool)
        valid[:5, :] = False
        valid[:, :4] = False
        protected = plume > 0.5
        diagnostics, curve = estimate_scene_broad_slope(
            {
                "weak": stripe + plume,
                "strong": 0.55 * stripe + 0.8 * plume,
            },
            valid,
            protected,
            config=self._synthetic_slope_config(),
        )
        self.assertAlmostEqual(
            diagnostics["selected_angle_deg"], true_angle, delta=3.0
        )
        self.assertEqual(diagnostics["slope_status"], "supported")
        self.assertEqual(len([row for row in curve if row["selected"]]), 1)
        self.assertGreater(diagnostics["selected_joint_fractional_gain"], 0.2)

    def test_scene_slope_estimator_rejects_noise_peak(self) -> None:
        generator = np.random.default_rng(12345)
        shape = (128, 120)
        diagnostics, curve = estimate_scene_broad_slope(
            {
                "weak": generator.normal(size=shape),
                "strong": generator.normal(size=shape),
            },
            np.ones(shape, dtype=bool),
            np.zeros(shape, dtype=bool),
            config=self._synthetic_slope_config(),
        )
        self.assertEqual(diagnostics["slope_status"], "unsupported")
        self.assertIn(
            "peak_below_robust_z_gate", diagnostics["slope_support_failures"]
        )
        self.assertEqual(sum(row["selected"] for row in curve), 1)
        self.assertEqual(sum(row["slope_supported"] for row in curve), 0)

    def test_unsupported_scene_slope_skips_broad_dwt(self) -> None:
        rows, columns = np.indices((48, 44))
        raw = np.sin(2.0 * np.pi * (rows + columns) / 17.0)
        valid = np.ones(raw.shape, dtype=bool)
        corrected, local, diagnostics = _process_band(
            raw,
            raw,
            None,
            valid,
            np.zeros(raw.shape, dtype=bool),
            local_sigma_pixels=3.0,
            canvas_size=64,
            band_name="synthetic",
            minimum_threshold_coefficients=1,
            broad_slope=1.0,
            broad_rotation_degrees=45.0,
            apply_broad_dwt=False,
        )
        self.assertFalse(diagnostics["broad_dwt_applied"])
        self.assertTrue(
            all(
                row["status"] == "skipped_unsupported_scene_slope"
                and not row["requested"]
                for row in diagnostics["dwt"]
            )
        )
        self.assertEqual(corrected.shape, raw.shape)
        self.assertTrue(np.all(np.isfinite(local[valid])))

    def test_scene_slope_estimator_is_sign_scale_and_band_order_invariant(self) -> None:
        rows, columns = np.indices((112, 104))
        angle = 52.0
        slope = math.tan(math.radians(angle))
        coordinate = rows - slope * columns
        weak = np.sin(2.0 * np.pi * coordinate / 24.0)
        strong = 0.6 * np.sin(2.0 * np.pi * coordinate / 48.0)
        valid = np.ones(weak.shape, dtype=bool)
        protected = np.zeros(weak.shape, dtype=bool)
        config = self._synthetic_slope_config()
        first, first_curve = estimate_scene_broad_slope(
            {"weak": weak, "strong": strong},
            valid,
            protected,
            config=config,
        )
        transformed, transformed_curve = estimate_scene_broad_slope(
            {"strong": -7.0 * strong, "weak": -3.0 * weak},
            valid,
            protected,
            config=config,
        )
        self.assertEqual(first["selected_angle_deg"], transformed["selected_angle_deg"])
        np.testing.assert_allclose(
            [row["joint_fractional_gain"] for row in first_curve],
            [row["joint_fractional_gain"] for row in transformed_curve],
            atol=1e-12,
        )

    def test_aggregate_comparison_keeps_positive_and_reverse_paired(self) -> None:
        profile = {
            "positive_region_count": 10,
            "reverse_region_count": 20,
            "positive_review_shortlist_count": 4,
            "reverse_review_shortlist_count": 5,
            "positive_conservative_single_window_count": 2,
            "reverse_conservative_single_window_count": 1,
            "positive_strict_dual_region_count": 3,
            "reverse_strict_dual_region_count": 2,
        }
        pdf = dict(profile)
        pdf["positive_region_count"] = 12
        rows = build_aggregate_rows(profile, pdf)
        self.assertEqual(len(rows), 8)
        first = rows[0]
        self.assertEqual(first["method"], "profile_only")
        self.assertEqual(first["positive_count"], 10)
        self.assertEqual(first["reverse_count"], 20)
        self.assertEqual(first["positive_to_reverse_ratio"], 0.5)
        self.assertEqual(rows[4]["positive_count"], 12)

    def test_full_scene_safe_canvas_is_2624(self) -> None:
        self.assertEqual(
            safe_rotation_canvas(
                (1895, 1761), angle_degrees=PDF_ROTATION_DEGREES, divisor=64
            ),
            2624,
        )

    def test_masked_threshold_ignores_unsupported_extreme(self) -> None:
        horizontal = np.linspace(0.2, 1.0, 101).reshape(1, -1)
        vertical = np.linspace(0.05, 0.2, 101).reshape(1, -1)
        support = np.ones_like(horizontal, dtype=bool)
        support[0, -1] = False
        baseline = masked_histogram_difference_threshold(
            horizontal, vertical, support, bins=32
        )
        horizontal[0, -1] = 1.0e9
        vertical[0, -1] = -1.0e9
        mutated = masked_histogram_difference_threshold(
            horizontal, vertical, support, bins=32
        )
        self.assertGreater(baseline, 0.0)
        self.assertEqual(mutated, baseline)

    def test_supported_dwt_is_sign_equivariant_and_reduces_broad_stripe(self) -> None:
        height, width = 160, 144
        rows, columns = np.indices((height, width))
        coordinate = rows - PDF_BROAD_SLOPE * columns
        stripe = 0.8 * np.sin(2.0 * np.pi * coordinate / 32.0)
        stripe += 0.35 * np.sin(2.0 * np.pi * coordinate / 64.0)
        plume = 6.0 * np.exp(
            -((rows - 80.0) ** 2 + (columns - 72.0) ** 2) / (2.0 * 3.0**2)
        )
        image = stripe + plume
        valid = np.ones(image.shape, dtype=bool)
        valid[:12, :] = False
        valid[-8:, :] = False
        valid[:, :8] = False
        protected = plume > 0.6

        corrected, _stripe_map, diagnostics = (
            support_aware_wavelet_horizontal_destripe(
                image,
                valid,
                protected,
                slope=PDF_BROAD_SLOPE,
                rotation_angle_deg=PDF_ROTATION_DEGREES,
                levels_to_filter=(3, 4, 5),
                max_level=6,
                threshold_scale=0.75,
                diff_fraction=0.25,
                canvas_size=256,
                minimum_threshold_coefficients=5,
            )
        )
        negative, _negative_stripe, _negative_diagnostics = (
            support_aware_wavelet_horizontal_destripe(
                -image,
                valid,
                protected,
                slope=PDF_BROAD_SLOPE,
                rotation_angle_deg=PDF_ROTATION_DEGREES,
                levels_to_filter=(3, 4, 5),
                max_level=6,
                threshold_scale=0.75,
                diff_fraction=0.25,
                canvas_size=256,
                minimum_threshold_coefficients=5,
            )
        )

        background = valid & ~protected
        before_mse = float(np.mean((image[background] - plume[background]) ** 2))
        after_mse = float(np.mean((corrected[background] - plume[background]) ** 2))
        self.assertLess(after_mse, 0.6 * before_mse)
        self.assertLess(abs(float(corrected[80, 72]) - 6.0), 1.0)
        np.testing.assert_allclose(negative[valid], -corrected[valid], atol=1e-10)
        self.assertTrue(all(diagnostics[level - 1]["filtered"] for level in (3, 4, 5)))
        self.assertTrue(np.all(np.isnan(corrected[~valid])))

    def test_threshold_estimation_requires_fully_supported_coefficients(self) -> None:
        image = np.arange(64 * 64, dtype=float).reshape(64, 64)
        valid = np.ones_like(image, dtype=bool)
        protected = np.zeros_like(valid)
        protected[7, 7] = True
        _corrected, _stripe, diagnostics = (
            support_aware_wavelet_horizontal_destripe(
                image,
                valid,
                protected,
                slope=0.0,
                rotation_angle_deg=0.0,
                levels_to_filter=(3,),
                max_level=3,
                canvas_size=64,
                minimum_threshold_coefficients=1,
            )
        )
        level3 = diagnostics[2]
        self.assertEqual(level3["total_coefficient_count"], 64)
        self.assertEqual(level3["estimate_coefficient_count"], 63)
        self.assertEqual(level3["minimum_estimation_support_fraction"], 1.0)

    def test_insufficient_supported_coefficients_leave_image_unchanged(self) -> None:
        image = np.arange(64 * 64, dtype=float).reshape(64, 64)
        valid = np.zeros_like(image, dtype=bool)
        valid[24:40, 24:40] = True
        protected = np.zeros_like(valid)
        corrected, stripe, diagnostics = support_aware_wavelet_horizontal_destripe(
            image,
            valid,
            protected,
            slope=PDF_BROAD_SLOPE,
            rotation_angle_deg=PDF_ROTATION_DEGREES,
            levels_to_filter=(3, 4, 5),
            max_level=6,
            canvas_size=128,
            minimum_threshold_coefficients=500,
        )
        np.testing.assert_allclose(corrected[valid], image[valid], atol=1e-10)
        np.testing.assert_allclose(stripe[valid], 0.0, atol=1e-10)
        self.assertTrue(
            all(
                row["status"] == "insufficient_support"
                for row in diagnostics
                if row["requested"]
            )
        )

    def test_thin_median_and_protection_are_sign_symmetric(self) -> None:
        rows, columns = np.indices((96, 88))
        thin_ids = np.rint((rows - 0.9773460526106752 * columns) / 2.0)
        thin_stripe = 0.15 * ((thin_ids % 7) - 3)
        weak = thin_stripe.astype(float)
        strong = (-0.6 * thin_stripe).astype(float)
        weak[48, 44] += 8.0
        strong[48, 44] += 5.0
        valid = np.ones(weak.shape, dtype=bool)
        protected, _ = symmetric_protection_mask(
            weak, strong, valid, local_sigma_pixels=8.0
        )
        negative_protected, _ = symmetric_protection_mask(
            -weak, -strong, valid, local_sigma_pixels=8.0
        )
        np.testing.assert_array_equal(negative_protected, protected)
        corrected, _stripe, _diagnostics = fast_fixed_slope_median_destripe(
            weak, valid, protected
        )
        negative, _negative_stripe, _negative_diagnostics = (
            fast_fixed_slope_median_destripe(-weak, valid, protected)
        )
        np.testing.assert_allclose(negative, -corrected, atol=1e-12)
        self.assertLess(float(np.nanstd(corrected[~protected])), 0.05)

    def test_derived_batch_is_separate_complete_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "derived"
            source.mkdir()
            product_id = "TEST_PRODUCT"
            scene = source / product_id
            scene.mkdir()
            config = {
                "save_score_maps": True,
                "weak_min_nm": 1580.0,
                "weak_max_nm": 1750.0,
                "strong_min_nm": 2200.0,
                "strong_max_nm": 2390.0,
                "local_z_sigma": 8.0,
            }
            (source / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "current_product_ids": [product_id],
                        "completed_product_ids": [product_id],
                    }
                ),
                encoding="utf-8",
            )
            (source / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "scene_count": 1,
                        "current_product_ids": [product_id],
                        "quality_class_counts": {"usable": 1},
                        "analysis_config": config,
                    }
                ),
                encoding="utf-8",
            )
            (scene / "summary.json").write_text(
                json.dumps(
                    {
                        "product_id": product_id,
                        "quality_class": "usable",
                        "analysis_config": config,
                        "shape": [96, 88],
                    }
                ),
                encoding="utf-8",
            )
            rows, columns = np.indices((96, 88))
            coordinate = rows - PDF_BROAD_SLOPE * columns
            weak = np.sin(2.0 * np.pi * coordinate / 32.0).astype(np.float32)
            strong = (0.8 * weak).astype(np.float32)
            valid = np.ones(weak.shape, dtype=bool)
            cloud = np.zeros(weak.shape, dtype=bool)
            np.savez_compressed(
                scene / "score_maps.npz",
                valid=valid,
                cloud_proxy=cloud,
                weak_z=weak,
                strong_z=strong,
                weak_local_z=weak,
                strong_local_z=strong,
                weak_z_directionally_destriped=weak,
                strong_z_directionally_destriped=strong,
            )
            source_hash = (scene / "score_maps.npz").read_bytes()
            result = derive_batch(
                source,
                output,
                minimum_threshold_coefficients=1,
                slope_search_config=SceneSlopeSearchConfig(
                    angle_min_deg=45.0,
                    angle_max_deg=60.0,
                    angle_step_deg=1.0,
                    search_negative_slopes=False,
                    line_bin_width_pixels=12.0,
                    minimum_pixels_per_line=8,
                    sample_step=1,
                    trend_window_deg=6.0,
                    local_window_deg=1.0,
                    edge_exclusion_deg=1.0,
                    minimum_peak_robust_z=0.5,
                    excluded_angles_deg=(),
                    excluded_half_width_deg=0.0,
                ),
            )
            self.assertEqual(result["usable_scene_count"], 1)
            self.assertTrue((output / "run_manifest.json").is_file())
            self.assertTrue((output / "posthoc_pdf_dwt_scene_metrics.csv").is_file())
            self.assertTrue((output / "posthoc_pdf_dwt_scene_slopes.csv").is_file())
            self.assertTrue((output / "posthoc_pdf_dwt_slope_search.csv").is_file())
            self.assertFalse(
                (output.parent / f".{output.name}.lock").exists()
            )
            self.assertTrue((output / product_id / "score_maps.npz").is_file())
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                (scene / "score_maps.npz").read_bytes(), source_hash
            )
            batch_summary = json.loads(
                (output / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                batch_summary["analysis_config"][
                    "posthoc_dwt_minimum_threshold_coefficients"
                ],
                1,
            )
            self.assertEqual(
                batch_summary["analysis_config"]["posthoc_broad_slope_mode"],
                "per_scene_shared_weak_strong",
            )
            posthoc = json.loads(
                (output / "posthoc_pdf_dwt_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("scene_stripe_slope_sha256", posthoc)
            self.assertIn("audit_output_sha256", posthoc)
            scene_diagnostic = posthoc["scene_diagnostics"][0]
            self.assertEqual(
                scene_diagnostic["weak"]["scene_broad_rotation_degrees"],
                scene_diagnostic["strong"]["scene_broad_rotation_degrees"],
            )
            self.assertEqual(
                scene_diagnostic["weak"]["broad_dwt_applied"],
                scene_diagnostic["strong"]["broad_dwt_applied"],
            )
            self.assertTrue(scene_diagnostic["weak"]["broad_dwt_applied"])
            with np.load(output / product_id / "score_maps.npz") as archive:
                self.assertEqual(archive["weak_local_z"].shape, weak.shape)
                self.assertTrue(np.all(np.isfinite(archive["weak_local_z"])))
            with self.assertRaises(FileExistsError):
                derive_batch(source, output)

    def test_output_inside_source_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                derive_batch(source, source / "derived")

    def test_nonpositive_threshold_coefficient_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(ValueError):
                derive_batch(
                    root / "source",
                    root / "output",
                    minimum_threshold_coefficients=0,
                )


if __name__ == "__main__":
    unittest.main()

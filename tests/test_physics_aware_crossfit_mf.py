from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from iterative_mf_1600nm import compute_uas_log_slope  # noqa: E402
from physics_aware_crossfit_mf import (  # noqa: E402
    DetectorModel,
    e_bh_mask,
    make_continuum_transform,
    score_log_radiance,
    signed_null_calibration,
    spatial_fold_ids,
)


class PhysicsAwareCrossfitTests(unittest.TestCase):
    def test_uas_is_invariant_to_global_radiance_scale(self) -> None:
        alpha = np.asarray([0.0, 0.1, 0.2, 0.3])
        baseline = np.asarray([2.0, 4.0, 8.0])
        absorption = np.asarray([0.05, 0.1, 0.2])
        radiance = baseline[None, :] * np.exp(-alpha[:, None] * absorption)

        uas = compute_uas_log_slope(alpha, radiance, 0.0, 0.3)
        uas_times_100 = compute_uas_log_slope(alpha, radiance * 100.0, 0.0, 0.3)

        np.testing.assert_allclose(uas, absorption, atol=1e-12)
        np.testing.assert_allclose(uas_times_100, uas, atol=1e-12)

    def test_crossfit_e_value_has_null_mean_near_one_with_known_model(self) -> None:
        rng = np.random.default_rng(743)
        covariance = np.asarray(
            [
                [1.0, 0.25, 0.15, 0.05],
                [0.25, 1.2, 0.10, 0.20],
                [0.15, 0.10, 0.9, 0.30],
                [0.05, 0.20, 0.30, 1.1],
            ]
        )
        target = np.asarray([-0.2, -0.1, -0.35, -0.25])
        model = DetectorModel(
            mean=np.zeros(4),
            covariance=covariance,
            target=target,
            weak_indices=np.asarray([0, 1]),
            strong_indices=np.asarray([2, 3]),
            amplitude_cap=2.0,
        )
        null = rng.multivariate_normal(np.zeros(4), covariance, size=200_000)

        log_e = score_log_radiance(null, model)["log_e_average"]
        log_e_negative = score_log_radiance(null, model)["log_e_negative_control"]
        mean_e = float(np.exp(np.logaddexp.reduce(log_e) - np.log(len(log_e))))
        mean_e_negative = float(
            np.exp(np.logaddexp.reduce(log_e_negative) - np.log(len(log_e_negative)))
        )

        self.assertGreater(mean_e, 0.97)
        self.assertLess(mean_e, 1.03)
        self.assertGreater(mean_e_negative, 0.97)
        self.assertLess(mean_e_negative, 1.03)

    def test_crossfit_score_increases_for_consistent_signal(self) -> None:
        rng = np.random.default_rng(2)
        covariance = np.eye(4)
        target = np.asarray([-0.3, -0.2, -0.5, -0.4])
        model = DetectorModel(
            mean=np.zeros(4),
            covariance=covariance,
            target=target,
            weak_indices=np.asarray([0, 1]),
            strong_indices=np.asarray([2, 3]),
            amplitude_cap=5.0,
        )
        background = rng.normal(size=(10_000, 4))
        null_score = score_log_radiance(background, model)["log_e_average"]
        signal_score = score_log_radiance(background + 3.0 * target, model)[
            "log_e_average"
        ]

        self.assertGreater(np.median(signal_score), np.median(null_score) + 0.5)

    def test_e_bh_selects_expected_large_e_values(self) -> None:
        log_e = np.log(np.asarray([100.0, 60.0, 1.0, 0.5]))
        selected, threshold, count = e_bh_mask(log_e, fdr=0.1)

        self.assertEqual(count, 2)
        np.testing.assert_array_equal(selected, [True, True, False, False])
        self.assertAlmostEqual(threshold, np.log(60.0))

    def test_signed_null_calibration_normalizes_pooled_e_mean(self) -> None:
        positive = np.asarray([0.0, 2.0, 100.0, np.nan])
        reverse = np.asarray([0.5, 1.0, 3.0, np.nan])

        calibrated_positive, calibrated_reverse, normalizer = signed_null_calibration(
            positive, reverse, log_e_cap=10.0
        )
        pooled_e = np.exp(
            np.concatenate(
                [
                    calibrated_positive[np.isfinite(calibrated_positive)],
                    calibrated_reverse[np.isfinite(calibrated_reverse)],
                ]
            )
        )

        self.assertAlmostEqual(float(pooled_e.mean()), 1.0)
        self.assertGreater(normalizer, 0.0)
        self.assertTrue(np.isnan(calibrated_positive[-1]))

    def test_spatial_blocks_are_not_split(self) -> None:
        y = np.asarray([0, 9, 10, 19])
        x = np.asarray([0, 9, 0, 9])
        folds = spatial_fold_ids(y, x, n_folds=5, block_size=10)

        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[2], folds[3])
        self.assertNotEqual(folds[0], folds[2])

    def test_continuum_transform_removes_constant_and_linear_terms(self) -> None:
        wavelengths = np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0])
        weak = np.arange(4)
        strong = np.arange(4, 8)
        transform, weak_features, strong_features = make_continuum_transform(
            wavelengths, weak, strong, degree=1
        )
        continuum = np.concatenate(
            [2.0 + 0.3 * wavelengths[weak], -1.0 + 0.8 * wavelengths[strong]]
        )

        np.testing.assert_allclose(transform @ continuum, 0.0, atol=1e-12)
        self.assertEqual(len(weak_features), 2)
        self.assertEqual(len(strong_features), 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spatial_plume_injection_benchmark import exact_log_shift, plume_field  # noqa: E402
from spatial_region_crossvalidation import (  # noqa: E402
    benjamini_hochberg,
    grow_regions,
)


class SpatialPlumeImagingTests(unittest.TestCase):
    def test_plume_is_zero_upwind_and_peaks_at_source(self) -> None:
        y, x = np.mgrid[-2:3, -2:5]
        field = plume_field(
            y.ravel(),
            x.ravel(),
            source_y=0.0,
            source_x=0.0,
            angle_radians=0.0,
            peak_ppm=1.0,
            length_scale=10.0,
            initial_width=1.0,
            spreading=0.1,
        ).reshape(y.shape)

        self.assertTrue(np.all(field[:, x[0] < 0] == 0.0))
        self.assertAlmostEqual(float(field[2, 2]), 1.0)
        self.assertLess(float(field[2, 3]), 1.0)

    def test_exact_log_shift_interpolates_modtran_ppm(self) -> None:
        alpha = np.asarray([0.0, 1.0, 2.0])
        radiance = np.exp(
            -alpha[:, None] * np.asarray([0.2, 0.5])[None, :]
        )

        shift = exact_log_shift(np.asarray([0.0, 0.5, 2.0]), alpha, radiance)

        np.testing.assert_allclose(shift[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(shift[1], [-0.1, -0.25], atol=1e-12)
        np.testing.assert_allclose(shift[2], [-0.4, -1.0], atol=1e-12)

    def test_grow_regions_keeps_only_requested_sizes(self) -> None:
        score = np.zeros((12, 12), dtype=float)
        score[2:5, 2:5] = 2.5
        score[3, 3] = 4.0
        score[8, 8] = 4.0

        labels, component_ids = grow_regions(
            score,
            core_threshold=3.5,
            extent_threshold=2.0,
            minimum_pixels=5,
            maximum_pixels=20,
        )

        self.assertEqual(len(component_ids), 1)
        self.assertEqual(int(np.sum(labels > 0)), 9)

    def test_bh_adjustment_is_monotone_in_rank(self) -> None:
        p_values = np.asarray([0.001, 0.02, 0.04, 0.8])
        adjusted = benjamini_hochberg(p_values)

        np.testing.assert_allclose(adjusted, [0.004, 0.04, 0.053333333333, 0.8])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fuse_1600_2200_methane_detection import (  # noqa: E402
    build_extent,
    keep_connected_components,
    robust_zscore,
)


class DualBandFusionTests(unittest.TestCase):
    def test_robust_zscore_resists_positive_outlier(self) -> None:
        image = np.random.default_rng(5).normal(0.0, 1.0, size=(9, 9))
        image[4, 4] = 100.0

        z, median, scale = robust_zscore(image)

        self.assertLess(abs(median), 0.5)
        self.assertGreater(scale, 0.0)
        self.assertGreater(z[4, 4], 10.0)

    def test_singleton_is_rejected_but_connected_core_is_kept(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = True
        mask[4, 4] = True
        mask[5, 5] = True

        connected, _, _ = keep_connected_components(mask, min_pixels=2)

        self.assertFalse(connected[1, 1])
        self.assertTrue(connected[4, 4])
        self.assertTrue(connected[5, 5])
        self.assertEqual(int(connected.sum()), 2)

    def test_extent_only_grows_support_touching_the_core(self) -> None:
        core = np.zeros((10, 10), dtype=bool)
        core[2, 2] = True
        z1600 = np.zeros((10, 10), dtype=float)
        z2200 = np.zeros((10, 10), dtype=float)
        joint = np.zeros((10, 10), dtype=float)

        z1600[1:4, 1:4] = 2.5
        z2200[1:4, 1:4] = 2.5
        joint[1:4, 1:4] = 3.0
        z1600[7:9, 7:9] = 3.0
        z2200[7:9, 7:9] = 3.0
        joint[7:9, 7:9] = 4.0

        extent = build_extent(
            core,
            z1600,
            z2200,
            joint,
            extent_threshold=2.0,
            extent_joint_threshold=2.7,
        )

        self.assertEqual(int(extent.sum()), 9)
        self.assertTrue(extent[2, 2])
        self.assertFalse(extent[7, 7])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from known_site_r2_validation import (  # noqa: E402
    Box,
    box_statistic,
    component_with_most_overlap,
    largest_joint_component,
    mask_geometry,
    mask_overlap,
    peak_on_mask,
    rectangles_intersect,
    select_rgb_matched_controls,
    symmetric_wind_cones,
    translated_boxes,
    upper_tail_fraction_plus_one,
)


class KnownSiteR2ValidationTests(unittest.TestCase):
    def test_box_statistics_use_full_box_or_fixed_top_fraction(self) -> None:
        values = np.arange(36, dtype=float).reshape(6, 6)
        box = Box(1, 2, 2, 3)

        self.assertAlmostEqual(box_statistic(values, box), np.mean([8, 9, 10, 14, 15, 16]))
        self.assertAlmostEqual(
            box_statistic(values, box, top_fraction=0.5), np.mean([14, 15, 16])
        )

    def test_translated_controls_do_not_intersect_expanded_site(self) -> None:
        site = Box(8, 8, 3, 3)
        controls = translated_boxes(
            (20, 20),
            site,
            exclusion_buffer=2,
            minimum_radius=4.0,
            maximum_radius=12.0,
            stride=1,
        )
        expanded = Box(6, 6, 7, 7)

        self.assertGreater(len(controls), 0)
        self.assertTrue(all(not rectangles_intersect(box, expanded) for box in controls))

    def test_upper_tail_fraction_uses_plus_one_correction(self) -> None:
        tail_fraction, exceedances, count = upper_tail_fraction_plus_one(
            4.0, np.asarray([1.0, 2.0, 4.0, 5.0])
        )

        self.assertEqual(exceedances, 2)
        self.assertEqual(count, 4)
        self.assertAlmostEqual(tail_fraction, 3.0 / 5.0)

    def test_rgb_matching_selects_identical_patch_first(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=float)
        rgb[0:2, 0:2] = [1.0, 2.0, 3.0]
        rgb[4:6, 4:6] = [1.0, 2.0, 3.0]
        rgb[6:8, 6:8] = [8.0, 8.0, 8.0]
        site = Box(0, 0, 2, 2)
        candidates = [Box(6, 6, 2, 2), Box(4, 4, 2, 2)]

        selected, distances, _ = select_rgb_matched_controls(
            rgb, site, candidates, count=1
        )

        self.assertEqual(selected[0], candidates[1])
        self.assertAlmostEqual(float(distances[0]), 0.0)

    def test_largest_joint_component_requires_both_bands(self) -> None:
        first = np.zeros((8, 8), dtype=float)
        second = np.zeros((8, 8), dtype=float)
        first[2:5, 2:5] = 3.0
        second[2:4, 2:4] = 3.0

        pixels, largest = largest_joint_component(
            first, second, Box(1, 1, 5, 5), threshold=2.0
        )

        self.assertEqual(pixels, 4)
        self.assertEqual(largest, 4)

    def test_component_selection_and_overlap_are_explicit(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[0:2, 0:2] = True
        mask[5:8, 4:7] = True
        reference = np.zeros_like(mask)
        reference[6:8, 5:8] = True

        selected = component_with_most_overlap(mask, reference)
        overlap = mask_overlap(selected, reference)

        self.assertEqual(int(selected.sum()), 9)
        self.assertEqual(overlap["intersection_pixels"], 4)
        self.assertAlmostEqual(overlap["jaccard"], 4.0 / 11.0)
        self.assertEqual(
            mask_geometry(selected, y_origin=10, x_origin=20),
            {"pixels": 9, "y_min": 15, "y_max": 17, "x_min": 24, "x_max": 26},
        )

    def test_peak_reports_global_percentile(self) -> None:
        values = np.arange(16, dtype=float).reshape(4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True

        peak = peak_on_mask(values, mask, y_origin=100, x_origin=200)

        self.assertEqual((peak["y"], peak["x"]), (102, 202))
        self.assertEqual(peak["value"], 10.0)
        self.assertAlmostEqual(peak["whole_map_percentile"], 68.75)

    def test_wind_cones_follow_image_coordinates(self) -> None:
        downwind, upwind, bearing = symmetric_wind_cones(
            (21, 21),
            source_y=10.0,
            source_x=10.0,
            wind_from_degrees=270.0,
            inner_radius=1.0,
            outer_radius=6.0,
            half_angle_degrees=30.0,
        )

        self.assertAlmostEqual(bearing, 90.0)
        self.assertTrue(downwind[10, 15])
        self.assertFalse(downwind[10, 5])
        self.assertTrue(upwind[10, 5])
        self.assertEqual(int(downwind.sum()), int(upwind.sum()))

    def test_wind_cone_outer_radius_is_euclidean(self) -> None:
        downwind, _, _ = symmetric_wind_cones(
            (41, 41),
            source_y=20.0,
            source_x=20.0,
            wind_from_degrees=270.0,
            inner_radius=1.0,
            outer_radius=10.0,
            half_angle_degrees=60.0,
        )

        rows, columns = np.where(downwind)
        radii = np.hypot(rows - 20.0, columns - 20.0)
        self.assertLessEqual(float(radii.max()), 10.0)
        self.assertFalse(downwind[28, 28])


if __name__ == "__main__":
    unittest.main()

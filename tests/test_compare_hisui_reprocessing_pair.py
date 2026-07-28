from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_hisui_reprocessing_pair import (  # noqa: E402
    common_projected_bounds,
    paired_diagnostics,
    prepare_roi,
    shared_false_color_stretch,
    validate_same_acquisition,
)
from hisui_l1g_io import GeoReference, HISUIL1GProduct  # noqa: E402


def _reader(origin_x: float, *, pixel_width: float = 20.0):
    georef = GeoReference(
        geotransform=(origin_x, pixel_width, 0.0, 4470810.0, 0.0, -20.0),
        crs="EPSG:32640",
        epsg=32640,
        pixel_scale=(pixel_width, 20.0, 0.0),
        tiepoint=(0.0,) * 6,
        raster_type=2,
    )
    return SimpleNamespace(georeference=georef, height=20, width=20)


def _product(product_id: str) -> HISUIL1GProduct:
    directory = Path(product_id)
    return HISUIL1GProduct(
        product_id=product_id,
        directory=directory,
        image_path=directory / f"{product_id}.tif",
        metadata_path=directory / f"{product_id}.txt",
        band_csv_path=directory / f"{product_id}_B.csv",
    )


class CompareHISUIReprocessingPairTests(unittest.TestCase):
    def test_same_acquisition_allows_small_reprocessing_timestamp_roundoff(self) -> None:
        old_id = "HSHL1G_N402E0583_20210803092156_20220830001542"
        new_id = "HSHL1G_N402E0583_20210803092156_20240106123056"
        result = validate_same_acquisition(
            _product(old_id),
            _product(new_id),
            {
                "ProductID": old_id,
                "SceneCenterTime": "2021-08-03T09:21:55.517446Z",
            },
            {
                "ProductID": new_id,
                "SceneCenterTime": "2021-08-03T09:21:55.517295Z",
            },
        )
        self.assertEqual(result["status"], "same_acquisition_verified")
        self.assertAlmostEqual(
            result["absolute_scene_center_time_difference_seconds"], 0.000151
        )

    def test_different_acquisition_token_is_rejected(self) -> None:
        old_id = "HSHL1G_N402E0583_20210803092156_20220830001542"
        new_id = "HSHL1G_N402E0583_20210804092156_20240106123056"
        with self.assertRaisesRegex(ValueError, "different acquisition tokens"):
            validate_same_acquisition(
                _product(old_id),
                _product(new_id),
                {
                    "ProductID": old_id,
                    "SceneCenterTime": "2021-08-03T09:21:55.5Z",
                },
                {
                    "ProductID": new_id,
                    "SceneCenterTime": "2021-08-04T09:21:55.5Z",
                },
            )

    def test_different_scene_tile_is_rejected_even_with_same_timestamp(self) -> None:
        old_id = "HSHL1G_N402E0583_20210803092156_20220830001542"
        new_id = "HSHL1G_N403E0583_20210803092156_20240106123056"
        with self.assertRaisesRegex(ValueError, "same source scene"):
            validate_same_acquisition(
                _product(old_id),
                _product(new_id),
                {
                    "ProductID": old_id,
                    "SceneCenterTime": "2021-08-03T09:21:55.517446Z",
                },
                {
                    "ProductID": new_id,
                    "SceneCenterTime": "2021-08-03T09:21:55.517295Z",
                },
            )

    def test_prepare_roi_masks_quality_band_qa_before_cloud_screening(self) -> None:
        product_id = "HSHL1G_N402E0583_20210803092156_20220830001542"
        product = _product(product_id)
        metadata = {
            "RadianceMultiVNIR": 1.0,
            "RadianceAddVNIR": 0.0,
            "RadianceMultiSWIR": 1.0,
            "RadianceAddSWIR": 0.0,
        }
        dn = np.full((2, 2, 5), 100, dtype=np.uint16)
        reader = SimpleNamespace(
            product=product,
            metadata=metadata,
            read_region=lambda *_args: dn.copy(),
        )
        plan = SimpleNamespace(
            read_bands=np.arange(5),
            quality_positions=np.arange(5),
            quality_bands=np.arange(5),
            detection_positions=np.arange(5),
            detection_bands=np.arange(58, 63),
            reflectance_multiplier=np.full(5, 0.001),
            reflectance_additive=np.zeros(5),
        )
        quality_mask = np.zeros((2, 2), dtype=bool)
        quality_mask[0, 0] = True
        detector_mask = np.zeros((2, 2), dtype=bool)
        quality_stats = {
            "quality_qa_missing": [],
            "quality_qa_dead_corrected_pixels": 1,
            "quality_qa_bad_interpolated_pixels": 0,
            "quality_qa_masked_pixels": 1,
            "quality_qa_interpolated_included": True,
            "quality_qa_complete": True,
        }
        detector_stats = {
            "detector_qa_missing": [],
            "detector_qa_dead_corrected_pixels": 0,
            "detector_qa_bad_interpolated_pixels": 0,
            "detector_qa_masked_pixels": 0,
            "detector_qa_interpolated_included": True,
            "detector_qa_complete": True,
        }
        with patch(
            "compare_hisui_reprocessing_pair.detector_qa_mask",
            side_effect=[
                (quality_mask, quality_stats),
                (detector_mask, detector_stats),
            ],
        ) as mocked_qa:
            _radiance, clear, _cloud, _browse, stats = prepare_roi(
                reader,
                (0, 2, 0, 2),
                plan,
                cloud_dilation_pixels=0,
                cloud_profile="cirrus_sensitive",
            )
        self.assertFalse(clear[0, 0])
        self.assertEqual(int(clear.sum()), 3)
        self.assertEqual(stats["quality_qa_affected_pixels"], 1)
        self.assertEqual(stats["quality_qa_masked_pixels"], 1)
        self.assertTrue(stats["qa_complete"])
        np.testing.assert_array_equal(mocked_qa.call_args_list[0].args[2], plan.quality_bands)

    def test_point_grids_with_one_column_origin_shift_align(self) -> None:
        old = _reader(592550.0)
        new = _reader(592530.0)
        site_x, site_y = old.georeference.map_xy(5, 7, pixel_center=True)
        old_bounds, new_bounds, _ = common_projected_bounds(
            old,
            new,
            easting_m=site_x,
            northing_m=site_y,
            half_size_pixels=2,
        )
        self.assertEqual(old_bounds, (3, 7, 5, 9))
        self.assertEqual(new_bounds, (3, 7, 6, 10))
        for old_row, old_column, new_row, new_column in (
            (3, 5, 3, 6),
            (7, 9, 7, 10),
        ):
            self.assertEqual(
                old.georeference.map_xy(old_row, old_column),
                new.georeference.map_xy(new_row, new_column),
            )

    def test_scale_mismatch_is_rejected_before_rounding(self) -> None:
        old = _reader(592550.0)
        new = _reader(592530.0, pixel_width=20.001)
        site_x, site_y = old.georeference.map_xy(5, 7, pixel_center=True)
        with self.assertRaisesRegex(ValueError, "pixel scale"):
            common_projected_bounds(
                old,
                new,
                easting_m=site_x,
                northing_m=site_y,
                half_size_pixels=2,
            )

    def test_paired_diagnostics_report_bias_mae_and_regression(self) -> None:
        old = np.arange(6, dtype=float).reshape(2, 3)
        new = 2.0 * old + 1.0
        result = paired_diagnostics(old, new, np.ones_like(old, dtype=bool))
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["new_on_old_slope"], 2.0)
        self.assertAlmostEqual(result["new_on_old_intercept"], 1.0)
        self.assertAlmostEqual(result["new_minus_old_mean_bias"], 3.5)

    def test_constant_pairs_do_not_emit_nonfinite_regression(self) -> None:
        old = np.ones((3, 3), dtype=float)
        new = np.ones((3, 3), dtype=float) * 2
        result = paired_diagnostics(old, new, np.ones_like(old, dtype=bool))
        self.assertIsNone(result["pearson_r"])
        self.assertIsNone(result["new_on_old_slope"])
        self.assertIsNone(result["new_on_old_intercept"])

    def test_false_color_uses_one_shared_stretch(self) -> None:
        old = np.zeros((2, 2, 3), dtype=float)
        new = np.ones((2, 2, 3), dtype=float)
        shown_old, shown_new, limits = shared_false_color_stretch(old, new)
        self.assertTrue(np.all(shown_old < shown_new))
        self.assertEqual(len(limits), 2)
        self.assertEqual(len(limits[0]), 3)


if __name__ == "__main__":
    unittest.main()

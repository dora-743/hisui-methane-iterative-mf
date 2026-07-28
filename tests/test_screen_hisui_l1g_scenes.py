from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hisui_l1g_io import GeoReference, HISUIL1GProduct  # noqa: E402
from screen_hisui_l1g_scenes import (  # noqa: E402
    BandPlan,
    _component_rows,
    _find_stale_scene_directories,
    _make_run_manifest,
    _progressive_tile_strides,
    _write_run_manifest,
    build_parser,
    cloud_proxy_from_quality_dn,
    dilate_cloud_mask,
    directional_profile_destripe,
    is_officially_scored,
    projected_to_pixel,
    quality_class,
    run_batch,
    scene_quality_class,
    sparse_cloud_preflight,
    weighted_local_z,
)


class HISUIMultisceneScreenTests(unittest.TestCase):
    def test_startup_failure_replaces_old_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            product_root = root / "empty_products"
            output_dir = root / "outputs"
            product_root.mkdir()
            output_dir.mkdir()
            (output_dir / "run_manifest.json").write_text(
                json.dumps({"status": "complete", "old": True}), encoding="utf-8"
            )
            args = build_parser().parse_args(
                [
                    "--product-root",
                    str(product_root),
                    "--modtran-csv",
                    str(root / "unused.csv"),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with self.assertRaisesRegex(FileNotFoundError, "No HISUI products"):
                run_batch(args)

            manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure_type"], "FileNotFoundError")
            self.assertNotIn("old", manifest)

    def test_stale_scene_directories_are_reported_without_being_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            current = output_dir / "HSHL1G_current"
            stale_complete = output_dir / "HSHL1G_stale_complete"
            stale_interrupted = output_dir / "HSHL1G_stale_interrupted"
            auxiliary = output_dir / "cache"
            artifact_only = output_dir / "legacy_scene_name"
            for directory in (
                current,
                stale_complete,
                stale_interrupted,
                auxiliary,
                artifact_only,
            ):
                directory.mkdir()
            (stale_complete / "summary.json").write_text("{}", encoding="utf-8")
            (artifact_only / "overview.png").write_bytes(b"old")

            stale = _find_stale_scene_directories(
                output_dir, ["HSHL1G_current"]
            )

            self.assertEqual(
                [path.name for path in stale],
                [
                    "HSHL1G_stale_complete",
                    "HSHL1G_stale_interrupted",
                    "legacy_scene_name",
                ],
            )
            self.assertTrue(stale_complete.exists())
            self.assertTrue(stale_interrupted.exists())
            self.assertTrue(artifact_only.exists())
            self.assertNotIn(auxiliary.resolve(), stale)

    def test_run_manifest_explicitly_separates_current_and_stale_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory).resolve()
            manifest = _make_run_manifest(
                product_root=output_dir / "products",
                output_dir=output_dir,
                current_product_ids=["HSHL1G_current"],
                stale_scene_directories=[output_dir / "HSHL1G_old"],
                started_utc="2026-07-28T00:00:00+00:00",
            )

            _write_run_manifest(output_dir, manifest)
            loaded = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(loaded["current_product_ids"], ["HSHL1G_current"])
            self.assertEqual(
                Path(loaded["stale_scene_directories"][0]).name,
                "HSHL1G_old",
            )
            self.assertEqual(loaded["status"], "running")
            self.assertIn("not members of this run", loaded["stale_scene_warning"])
            self.assertEqual(
                Path(loaded["current_scene_directories"][0]),
                output_dir / "HSHL1G_current",
            )

    def test_cloud_proxy_distinguishes_bright_and_cirrus_pixels(self) -> None:
        # Unit reflectance coefficients make DN/1000 equal reflectance.
        dn = np.asarray(
            [
                [[100, 100, 5, 100, 100], [400, 400, 5, 300, 100]],
                [[100, 250, 50, 100, 100], [0, 400, 100, 300, 100]],
            ],
            dtype=np.uint16,
        )
        valid, cloud, stats = cloud_proxy_from_quality_dn(
            dn,
            np.full(5, 0.001),
            np.zeros(5),
            np.asarray([0, 1, 65535], dtype=np.uint16),
        )
        np.testing.assert_array_equal(valid, [[True, True], [True, False]])
        np.testing.assert_array_equal(cloud, [[False, True], [True, False]])
        self.assertAlmostEqual(stats["cloud_proxy_fraction"], 2 / 3)

    def test_quality_classes_use_fixed_boundaries(self) -> None:
        self.assertEqual(quality_class(0.10, maximum_for_scoring=0.5), "usable")
        self.assertEqual(quality_class(0.11, maximum_for_scoring=0.5), "partial")
        self.assertEqual(
            quality_class(0.51, maximum_for_scoring=0.5), "excluded_cloud"
        )
        self.assertEqual(
            quality_class(0.08, maximum_for_scoring=0.05), "excluded_cloud"
        )

    def test_saturated_quality_pixel_is_counted_as_cloud(self) -> None:
        dn = np.full((1, 1, 5), 100, dtype=np.uint16)
        dn[0, 0, 0] = 65535
        valid, cloud, stats = cloud_proxy_from_quality_dn(
            dn,
            np.full(5, 0.001),
            np.zeros(5),
            np.asarray([0, 1], dtype=np.uint16),
            saturated_values=np.asarray([65535], dtype=np.uint16),
        )
        self.assertTrue(valid[0, 0])
        self.assertTrue(cloud[0, 0])
        self.assertEqual(stats["saturated_quality_fraction"], 1.0)

    def test_quality_qa_affected_pixel_is_invalid_not_cloud(self) -> None:
        dn = np.full((1, 2, 5), 400, dtype=np.uint16)
        qa_affected = np.asarray([[True, False]])
        valid, cloud, stats = cloud_proxy_from_quality_dn(
            dn,
            np.full(5, 0.001),
            np.zeros(5),
            np.asarray([0, 1], dtype=np.uint16),
            qa_affected=qa_affected,
        )
        np.testing.assert_array_equal(valid, [[False, True]])
        np.testing.assert_array_equal(cloud, [[False, True]])
        self.assertEqual(stats["quality_qa_affected_pixels"], 1)
        self.assertEqual(stats["quality_valid_pixels"], 1)

    def test_qa_incomplete_is_not_officially_scored(self) -> None:
        self.assertEqual(
            scene_quality_class(
                0.01, maximum_for_scoring=0.5, qa_complete=False
            ),
            "qa_incomplete",
        )
        self.assertFalse(is_officially_scored("qa_incomplete"))
        self.assertFalse(is_officially_scored("excluded_cloud"))
        self.assertTrue(is_officially_scored("usable"))
        self.assertTrue(is_officially_scored("partial"))

    def test_progressive_strides_reach_full_tile_coverage(self) -> None:
        self.assertEqual(_progressive_tile_strides(8), (8, 4, 2, 1))
        self.assertEqual(_progressive_tile_strides(1), (1,))
        self.assertEqual(_progressive_tile_strides(7), (7, 3, 1))
        with self.assertRaises(ValueError):
            _progressive_tile_strides(0)

    def test_cloud_preflight_densifies_before_exclusion_and_applies_qa(self) -> None:
        cloud_tiles = np.zeros((8, 8), dtype=bool)
        cloud_tiles[::2, ::2] = True
        quality_qa = np.zeros((8, 8), dtype=bool)
        quality_qa[0, 1] = True

        class FakeReader:
            height = 8
            width = 8
            tile_rows = 8
            tile_columns = 8
            tile_height = 1
            tile_width = 1

            def __init__(self) -> None:
                self.read_tiles: list[tuple[int, int]] = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read_region(self, y0, y1, x0, x1, _bands):
                self.read_tiles.append((y0, x0))
                value = 400 if cloud_tiles[y0, x0] else 100
                return np.full((y1 - y0, x1 - x0, 5), value, dtype=np.uint16)

        fake_reader = FakeReader()
        product = HISUIL1GProduct(
            product_id="test",
            directory=Path("."),
            image_path=Path("image.tif"),
            metadata_path=Path("metadata.txt"),
            band_csv_path=Path("bands.csv"),
        )
        indices = np.arange(5, dtype=int)
        plan = BandPlan(
            detection_bands=indices,
            quality_bands=indices,
            read_bands=indices,
            detection_positions=indices,
            quality_positions=indices,
            wavelengths_nm=np.arange(5, dtype=float),
            fwhm_nm=np.ones(5),
            weak_indices=np.arange(3),
            strong_indices=np.arange(2, 5),
            reflectance_multiplier=np.full(5, 0.001),
            reflectance_additive=np.zeros(5),
        )
        qa_stats = {
            "quality_qa_missing": [],
            "quality_qa_dead_corrected_pixels": 1,
            "quality_qa_bad_interpolated_pixels": 0,
            "quality_qa_masked_pixels": 1,
            "quality_qa_interpolated_included": True,
            "quality_qa_complete": True,
        }
        with (
            patch("screen_hisui_l1g_scenes.parse_metadata", return_value={}),
            patch("screen_hisui_l1g_scenes.HISUIL1GReader", return_value=fake_reader),
            patch(
                "screen_hisui_l1g_scenes.detector_qa_mask",
                return_value=(quality_qa, qa_stats),
            ),
        ):
            valid, _cloud, stats = sparse_cloud_preflight(
                product,
                plan,
                tile_stride=8,
                cloud_profile="cirrus_sensitive",
                confirmation_threshold=0.20,
            )

        self.assertEqual(stats["preflight_evaluated_strides"], [8, 4, 2, 1])
        self.assertTrue(stats["preflight_full_coverage"])
        self.assertEqual(len(fake_reader.read_tiles), 64)
        self.assertFalse(valid[0, 1])
        self.assertAlmostEqual(stats["cloud_proxy_fraction"], 16 / 63)

    def test_zero_cloud_dilation_is_a_no_op(self) -> None:
        cloud = np.zeros((5, 5), dtype=bool)
        cloud[2, 2] = True
        np.testing.assert_array_equal(dilate_cloud_mask(cloud, 0), cloud)
        self.assertGreater(dilate_cloud_mask(cloud, 1).sum(), cloud.sum())
        with self.assertRaises(ValueError):
            dilate_cloud_mask(cloud, -1)

    def test_weighted_local_z_preserves_local_positive_anomaly(self) -> None:
        image = np.zeros((31, 31), dtype=float)
        image[15, 15] = 4.0
        valid = np.ones(image.shape, dtype=bool)
        output = weighted_local_z(image, valid, sigma_pixels=4.0)
        self.assertGreater(output[15, 15], 5.0)
        self.assertLess(abs(float(np.nanmedian(output))), 0.1)

    def test_directional_profile_removes_line_offsets_without_erasing_spot(self) -> None:
        rows, columns = np.indices((80, 70))
        line_ids = np.rint((rows - 0.8 * columns) / 2.0).astype(int)
        image = ((line_ids % 5) - 2).astype(float) * 0.4
        image[40, 35] += 6.0
        valid = np.ones(image.shape, dtype=bool)
        corrected, diagnostics = directional_profile_destripe(
            image, valid, profiles=[(0.8, 2.0)]
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertLess(
            diagnostics[0]["winsorized_profile_rms_after"],
            0.15 * diagnostics[0]["winsorized_profile_rms_before"],
        )
        self.assertGreater(corrected[40, 35], 4.0)

    def test_projected_pixel_and_components_use_pixel_centres(self) -> None:
        georef = GeoReference(
            geotransform=(500000.0, 20.0, 0.0, 4000000.0, 0.0, -20.0),
            crs="EPSG:32613",
            epsg=32613,
            pixel_scale=(20.0, 20.0, 0.0),
            tiepoint=(0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0),
            raster_type=1,
        )
        row, column = projected_to_pixel(georef, 500070.0, 3999950.0)
        self.assertEqual((row, column), (2.0, 3.0))

        dual = np.zeros((10, 10), dtype=float)
        weak = np.zeros_like(dual)
        strong = np.zeros_like(dual)
        dual[2:4, 3:5] = [[3.1, 3.2], [3.3, 4.0]]
        weak[2:4, 3:5] = 4.1
        strong[2:4, 3:5] = 3.5
        rows = _component_rows(
            dual,
            weak,
            strong,
            np.ones_like(dual, dtype=bool),
            np.zeros_like(dual, dtype=bool),
            georef,
            threshold=3.0,
            minimum_pixels=3,
            sign="positive_ch4",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pixel_count"], 4)
        self.assertEqual((rows[0]["peak_y"], rows[0]["peak_x"]), (3, 4))
        self.assertEqual((rows[0]["easting_m"], rows[0]["northing_m"]), (500090.0, 3999930.0))
        self.assertIsNone(rows[0]["distance_to_cloud_pixels"])

    def test_pixel_is_point_affine_maps_site_to_exact_sample(self) -> None:
        georef = GeoReference(
            geotransform=(499990.0, 20.0, 0.0, 4000010.0, 0.0, -20.0),
            crs="EPSG:32613",
            epsg=32613,
            pixel_scale=(20.0, 20.0, 0.0),
            tiepoint=(0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0),
            raster_type=2,
        )
        row, column = projected_to_pixel(georef, 500060.0, 3999960.0)
        self.assertEqual((row, column), (2.0, 3.0))
        self.assertEqual(
            georef.map_xy(2, 3, pixel_center=True),
            (500060.0, 3999960.0),
        )

    def test_diagonal_scene_spanning_component_is_flagged(self) -> None:
        georef = GeoReference(
            geotransform=(0.0, 20.0, 0.0, 0.0, 0.0, -20.0),
            crs=None,
            epsg=None,
            pixel_scale=(20.0, 20.0, 0.0),
            tiepoint=(0.0,) * 6,
            raster_type=1,
        )
        dual = np.zeros((150, 150), dtype=float)
        index = np.arange(120)
        dual[index, index] = 4.0
        rows = _component_rows(
            dual,
            dual,
            dual,
            np.ones_like(dual, dtype=bool),
            np.zeros_like(dual, dtype=bool),
            georef,
            threshold=3.0,
            minimum_pixels=3,
            sign="positive_ch4",
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["scene_spanning_line_flag"])
        self.assertGreater(rows[0]["component_elongation"], 10.0)


if __name__ == "__main__":
    unittest.main()

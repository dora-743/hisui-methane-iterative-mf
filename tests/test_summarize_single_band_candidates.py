from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hisui_l1g_io import GeoReference, HISUIL1GProduct  # noqa: E402
from summarize_single_band_candidates import (  # noqa: E402
    OUTPUT_NAMES,
    SceneSource,
    StripGeometry,
    TailMosaic,
    _extract_regions,
    _same_source_cross_band_conflict_mask,
    _strip_geometry,
    _tail_mosaic,
    summarize,
)


class SummarizeSingleBandCandidatesTests(unittest.TestCase):
    def _georef(self, *, origin_x: float = 100.0, origin_y: float = 500.0) -> GeoReference:
        return GeoReference(
            geotransform=(origin_x, 20.0, 0.0, origin_y, 0.0, -20.0),
            crs="EPSG:32613",
            epsg=32613,
            pixel_scale=(20.0, 20.0, 0.0),
            tiepoint=(0.0, 0.0, 0.0, origin_x + 10.0, origin_y - 10.0, 0.0),
            raster_type=2,
        )

    def _source(
        self,
        product_id: str,
        weak: np.ndarray,
        strong: np.ndarray,
        *,
        georef: GeoReference | None = None,
    ) -> SceneSource:
        shape = weak.shape
        product = HISUIL1GProduct(
            product_id=product_id,
            directory=Path(product_id),
            image_path=Path(product_id) / f"{product_id}.tif",
            metadata_path=Path(product_id) / f"{product_id}.txt",
            band_csv_path=Path(product_id) / f"{product_id}_B.csv",
        )
        return SceneSource(
            product=product,
            product_id=product_id,
            acquisition_utc="2022-01-01T00:00:05Z",
            acquisition_minute="2022-01-01T00:00:00Z",
            cloud_fraction=0.0,
            cloud_dilated_fraction=0.0,
            georef=georef or self._georef(),
            valid=np.ones(shape, dtype=bool),
            cloud=np.zeros(shape, dtype=bool),
            weak=np.asarray(weak, dtype=np.float32),
            strong=np.asarray(strong, dtype=np.float32),
        )

    def _geometry_for_source(self, source: SceneSource) -> StripGeometry:
        source.offset = (0, 0)
        return StripGeometry(
            acquisition_minute=source.acquisition_minute,
            georef=source.georef,
            shape=source.valid.shape,
            sources=[source],
            valid=source.valid.copy(),
            cloud=source.cloud.copy(),
            cloud_distance=np.full(source.valid.shape, np.inf, dtype=np.float32),
            invalid_distance=np.full(source.valid.shape, 10.0, dtype=np.float32),
        )

    def test_support_classes_and_eight_connectivity(self) -> None:
        shape = (24, 24)
        weak = np.zeros(shape, dtype=np.float32)
        strong = np.zeros(shape, dtype=np.float32)
        weak_mask = np.zeros(shape, dtype=bool)
        strong_mask = np.zeros(shape, dtype=bool)
        coincident = np.zeros(shape, dtype=bool)

        weak_only = [(2, 2), (2, 3), (3, 2)]
        strong_only = [(2, 15), (2, 16), (3, 15)]
        weak_near = [(9, 2), (9, 3), (10, 2)]
        strong_near = [(10, 3), (11, 3), (11, 4)]
        both = [(17, 17), (17, 18), (18, 17)]
        for y, x in weak_only + weak_near + both:
            weak[y, x] = 4.0
            weak_mask[y, x] = True
        for y, x in strong_only + strong_near + both:
            strong[y, x] = 4.0
            strong_mask[y, x] = True
        for y, x in both:
            coincident[y, x] = True

        source = self._source("HSHL1G_test", weak, strong)
        geometry = self._geometry_for_source(source)
        mosaic = TailMosaic(
            band1600=weak,
            band2200=strong,
            dual=np.minimum(weak, strong),
            mask1600=weak_mask,
            mask2200=strong_mask,
            coincident=coincident,
            strict_dual=coincident.copy(),
        )
        rows, selected = _extract_regions(
            geometry,
            mosaic,
            threshold=3.0,
            minimum_pixels=3,
            tail="positive_methane_like",
            sign=1,
        )
        self.assertEqual(
            {row["spectral_support"] for row in rows},
            {
                "1600_only",
                "2200_only",
                "both_noncoincident",
                "contains_coincident_dual_pixel",
            },
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(int(selected.sum()), 15)
        dual_row = next(
            row
            for row in rows
            if row["spectral_support"] == "contains_coincident_dual_pixel"
        )
        self.assertTrue(dual_row["contains_strict_dual_component"])
        self.assertEqual(dual_row["n_coincident_dual_pixels"], 3)
        self.assertEqual(dual_row["strip_product_ids"], "HSHL1G_test")
        self.assertEqual(dual_row["contributing_product_ids"], "HSHL1G_test")

    def test_shape_flag_uses_only_supported_scene_broad_angle(self) -> None:
        shape = (64, 64)
        weak = np.zeros(shape, dtype=np.float32)
        strong = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        columns = np.arange(6, 56)
        rows = 10 + np.rint(np.tan(np.deg2rad(20.0)) * (columns - 6)).astype(int)
        weak[rows, columns] = 4.0
        mask[rows, columns] = True

        source = self._source("HSHL1G_scene_angle", weak, strong)
        source.stripe_angles_deg = (44.3436, 20.0)
        source.broad_slope_status = "supported"
        mosaic = TailMosaic(
            band1600=weak,
            band2200=strong,
            dual=np.minimum(weak, strong),
            mask1600=mask,
            mask2200=np.zeros(shape, dtype=bool),
            coincident=np.zeros(shape, dtype=bool),
            strict_dual=np.zeros(shape, dtype=bool),
        )
        supported_rows, _selected = _extract_regions(
            self._geometry_for_source(source),
            mosaic,
            threshold=3.0,
            minimum_pixels=3,
            tail="positive_methane_like",
            sign=1,
        )
        self.assertEqual(len(supported_rows), 1)
        self.assertTrue(supported_rows[0]["shape_stripe_direction_flag"])
        self.assertEqual(
            supported_rows[0]["nearest_stripe_direction_kind"], "broad_scene"
        )
        self.assertEqual(
            supported_rows[0]["nearest_stripe_slope_status"], "supported"
        )

        source.stripe_angles_deg = (44.3436,)
        source.broad_slope_status = "unsupported"
        unsupported_rows, _selected = _extract_regions(
            self._geometry_for_source(source),
            mosaic,
            threshold=3.0,
            minimum_pixels=3,
            tail="positive_methane_like",
            sign=1,
        )
        self.assertFalse(unsupported_rows[0]["shape_stripe_direction_flag"])
        self.assertEqual(
            unsupported_rows[0]["nearest_stripe_direction_kind"], "thin_fixed"
        )

    def test_positive_and_reverse_are_exactly_sign_symmetric(self) -> None:
        weak = np.zeros((12, 12), dtype=np.float32)
        strong = np.zeros_like(weak)
        weak[3:6, 3:6] = 6.0
        strong[3:6, 3:6] = 4.0
        positive_source = self._source("HSHL1G_positive", weak, strong)
        negative_source = self._source("HSHL1G_negative", -weak, -strong)
        positive_geometry = self._geometry_for_source(positive_source)
        negative_geometry = self._geometry_for_source(negative_source)
        positive = _tail_mosaic(
            positive_geometry, threshold=3.0, minimum_pixels=3, sign=1
        )
        reverse = _tail_mosaic(
            negative_geometry, threshold=3.0, minimum_pixels=3, sign=-1
        )
        for name in (
            "band1600",
            "band2200",
            "dual",
            "mask1600",
            "mask2200",
            "coincident",
            "strict_dual",
        ):
            np.testing.assert_array_equal(getattr(positive, name), getattr(reverse, name))

    def test_cross_tail_conflict_distinguishes_same_source_from_cross_product(self) -> None:
        weak = np.zeros((4, 4), dtype=np.float32)
        strong = np.zeros_like(weak)
        weak[1, 1] = 4.0
        strong[1, 1] = -4.0
        source = self._source("HSHL1G_same", weak, strong)
        geometry = self._geometry_for_source(source)
        same_source = _same_source_cross_band_conflict_mask(geometry, threshold=3.0)
        self.assertEqual(int(same_source.sum()), 1)

        positive_source = self._source("HSHL1G_positive", weak, np.zeros_like(weak))
        negative_source = self._source("HSHL1G_negative", -weak, np.zeros_like(weak))
        geometry = _strip_geometry(
            positive_source.acquisition_minute, [positive_source, negative_source]
        )
        positive = _tail_mosaic(
            geometry, threshold=3.0, minimum_pixels=3, sign=1
        )
        reverse = _tail_mosaic(
            geometry, threshold=3.0, minimum_pixels=3, sign=-1
        )
        total_conflict = (positive.mask1600 | positive.mask2200) & (
            reverse.mask1600 | reverse.mask2200
        )
        same_source = _same_source_cross_band_conflict_mask(geometry, threshold=3.0)
        self.assertEqual(int(total_conflict.sum()), 1)
        self.assertEqual(int((total_conflict & same_source).sum()), 0)

        strong_only = np.zeros_like(weak)
        strong_only[1, 1] = 4.0
        weak_product = self._source("HSHL1G_weak", weak, np.zeros_like(weak))
        strong_product = self._source(
            "HSHL1G_strong", np.zeros_like(weak), strong_only
        )
        geometry = _strip_geometry(
            weak_product.acquisition_minute, [weak_product, strong_product]
        )
        positive = _tail_mosaic(
            geometry, threshold=3.0, minimum_pixels=3, sign=1
        )
        self.assertEqual(
            int((positive.mask1600 & positive.mask2200).sum()), 1
        )
        self.assertEqual(int(positive.coincident.sum()), 0)

    def test_strip_geometry_deduplicates_integer_aligned_overlap(self) -> None:
        weak = np.zeros((4, 5), dtype=np.float32)
        first = self._source("HSHL1G_a", weak, weak, georef=self._georef(origin_x=100.0))
        second = self._source("HSHL1G_b", weak, weak, georef=self._georef(origin_x=140.0))
        geometry = _strip_geometry(first.acquisition_minute, [first, second])
        self.assertEqual(geometry.shape, (4, 7))
        self.assertEqual(int(geometry.valid.sum()), 28)
        self.assertEqual(first.offset, (0, 0))
        self.assertEqual(second.offset, (0, 2))

    def test_strip_geometry_rejects_noninteger_offset(self) -> None:
        weak = np.zeros((4, 5), dtype=np.float32)
        first = self._source("HSHL1G_a", weak, weak, georef=self._georef(origin_x=100.0))
        second = self._source("HSHL1G_b", weak, weak, georef=self._georef(origin_x=141.0))
        with self.assertRaisesRegex(ValueError, "non-integer"):
            _strip_geometry(first.acquisition_minute, [first, second])

    def test_strip_geometry_rejects_broadcastable_cloud_shape(self) -> None:
        weak = np.zeros((4, 5), dtype=np.float32)
        source = self._source("HSHL1G_a", weak, weak)
        source.cloud = np.zeros((1, 5), dtype=bool)
        with self.assertRaisesRegex(ValueError, "cloud_proxy shape mismatch"):
            _strip_geometry(source.acquisition_minute, [source])

    def test_strip_geometry_computes_distances_on_union_grid(self) -> None:
        weak = np.zeros((5, 5), dtype=np.float32)
        first = self._source("HSHL1G_a", weak, weak, georef=self._georef(origin_x=100.0))
        second = self._source("HSHL1G_b", weak, weak, georef=self._georef(origin_x=200.0))
        first.cloud[2, 4] = True
        first.valid[2, 4] = False
        geometry = _strip_geometry(first.acquisition_minute, [first, second])
        self.assertEqual(geometry.shape, (5, 10))
        self.assertEqual(float(geometry.cloud_distance[2, 5]), 1.0)

        overlap = self._source(
            "HSHL1G_overlap", weak, weak, georef=self._georef(origin_x=140.0)
        )
        first.valid[2, 2] = False
        overlap.valid[2, 0] = True
        geometry = _strip_geometry(first.acquisition_minute, [first, overlap])
        self.assertTrue(geometry.valid[2, 2])
        self.assertGreater(float(geometry.invalid_distance[2, 2]), 0.0)

    def _write_product(
        self,
        root: Path,
        product_id: str,
        *,
        origin_x: float,
    ) -> Path:
        product_dir = root / product_id
        product_dir.mkdir()
        geokeys = (
            1,
            1,
            0,
            3,
            1024,
            0,
            1,
            1,
            1025,
            0,
            1,
            2,
            3072,
            0,
            1,
            32613,
        )
        tifffile.imwrite(
            product_dir / f"{product_id}.tif",
            np.zeros((12, 12), dtype=np.uint16),
            metadata=None,
            extratags=[
                (33550, "d", 3, (20.0, 20.0, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, origin_x, 500.0, 0.0), False),
                (34735, "H", len(geokeys), geokeys, False),
            ],
        )
        (product_dir / f"{product_id}.txt").write_text(
            f'ProductID = "{product_id}"\n'
            'SceneCenterTime = "2022-01-01T00:00:05Z"\n',
            encoding="utf-8",
        )
        (product_dir / f"{product_id}_B.csv").write_text(
            "BandNo,CenterWavelengthNanometer,FullWidthAtHalfMaximumNanometer\n",
            encoding="utf-8",
        )
        return product_dir

    def _write_scene(
        self,
        batch: Path,
        product_dir: Path,
        *,
        quality_class: str,
        save_maps: bool,
    ) -> None:
        product_id = product_dir.name
        scene = batch / product_id
        scene.mkdir()
        analysis_config = {
            "weak_min_nm": 1580.0,
            "weak_max_nm": 1750.0,
            "strong_min_nm": 2200.0,
            "strong_max_nm": 2390.0,
            "local_z_sigma": 20.0,
            "save_score_maps": True,
        }
        summary = {
            "product_id": product_id,
            "product_path": str(product_dir),
            "acquisition_utc": "2022-01-01T00:00:05Z",
            "shape": [12, 12, 185],
            "quality_class": quality_class,
            "official_scoring_eligible": quality_class in {"usable", "partial"},
            "cloud_proxy_fraction": 0.0,
            "cloud_dilated_fraction": 0.0,
            "analysis_config": analysis_config,
        }
        (scene / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        if save_maps:
            weak = np.zeros((12, 12), dtype=np.float32)
            strong = np.zeros_like(weak)
            weak[4:7, 4:7] = 7.0
            np.savez_compressed(
                scene / "score_maps.npz",
                valid=np.ones((12, 12), dtype=bool),
                cloud_proxy=np.zeros((12, 12), dtype=bool),
                weak_local_z=weak,
                strong_local_z=strong,
            )

    def test_end_to_end_uses_manifest_and_only_usable_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            products = root / "products"
            products.mkdir()
            usable = self._write_product(products, "HSHL1G_usable", origin_x=100.0)
            partial = self._write_product(products, "HSHL1G_partial", origin_x=400.0)
            batch = root / "batch"
            batch.mkdir()
            self._write_scene(batch, usable, quality_class="usable", save_maps=True)
            self._write_scene(batch, partial, quality_class="partial", save_maps=False)
            stale = batch / "HSHL1G_stale"
            stale.mkdir()
            (stale / "summary.json").write_text("not json", encoding="utf-8")
            (batch / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "current_product_ids": [usable.name, partial.name],
                        "completed_product_ids": [usable.name, partial.name],
                        "current_scene_directories": [
                            str((batch / usable.name).resolve()),
                            str((batch / partial.name).resolve()),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (batch / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "scene_count": 2,
                        "current_product_ids": [usable.name, partial.name],
                        "quality_class_counts": {"usable": 1, "partial": 1},
                        "analysis_config": {
                            "weak_min_nm": 1580.0,
                            "weak_max_nm": 1750.0,
                            "strong_min_nm": 2200.0,
                            "strong_max_nm": 2390.0,
                            "local_z_sigma": 20.0,
                            "save_score_maps": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            site_csv = root / "known_sites.csv"
            site_csv.write_text(
                "site_id,name,epsg,easting_m,northing_m,source_kind,source_url,note\n"
                "test_site,Test site,32613,210,390,test,https://example.test,proxy\n",
                encoding="utf-8",
            )
            output = root / "output"
            summary = summarize(
                batch,
                output,
                crop_half_size=3,
                known_site_csv=site_csv,
                known_site_id="test_site",
                site_radius_pixels=2,
            )
            self.assertEqual(summary["aggregate"]["usable_product_count"], 1)
            self.assertEqual(summary["aggregate"]["acquisition_strip_count"], 1)
            self.assertEqual(
                summary["provenance"]["skipped_products"],
                [{"product_id": partial.name, "quality_class": "partial"}],
            )
            self.assertEqual(summary["aggregate"]["positive_region_count"], 1)
            self.assertEqual(len(summary["known_site_audits"]), 1)
            self.assertEqual(
                summary["known_site_audits"][0][
                    "positive_1600_threshold_pixels_in_window"
                ],
                9,
            )
            with (output / "single_band_candidate_regions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["spectral_support"], "1600_only")
            self.assertEqual(rows[0]["conservative_single_window"], "True")
            for name in OUTPUT_NAMES:
                self.assertTrue((output / name).is_file(), name)

    def test_minimum_pixels_below_two_invalidates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(ValueError, "at least 2"):
                summarize(Path(temporary) / "missing", output, minimum_pixels=1)
            for name in OUTPUT_NAMES:
                self.assertFalse((output / name).exists(), name)

    def test_failed_run_invalidates_outputs_from_an_earlier_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            batch.mkdir()
            (batch / "run_manifest.json").write_text(
                json.dumps({"status": "running", "current_product_ids": ["x"]}),
                encoding="utf-8",
            )
            output = root / "output"
            output.mkdir()
            for name in OUTPUT_NAMES:
                (output / name).write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not complete"):
                summarize(batch, output)
            for name in OUTPUT_NAMES:
                self.assertFalse((output / name).exists(), name)

    def test_complete_manifest_requires_all_current_products_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            batch.mkdir()
            (batch / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "current_product_ids": ["HSHL1G_x"],
                        "completed_product_ids": [],
                        "current_scene_directories": [
                            str((batch / "HSHL1G_x").resolve())
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "completed_product_ids"):
                summarize(batch, root / "output")

    def test_nonfinite_threshold_invalidates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            for name in OUTPUT_NAMES:
                (output / name).write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "threshold"):
                summarize(root / "missing", output, threshold=float("nan"))
            for name in OUTPUT_NAMES:
                self.assertFalse((output / name).exists(), name)


if __name__ == "__main__":
    unittest.main()

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

from summarize_multiscene_tail_balance import summarize_batch  # noqa: E402


class SummarizeMultisceneTailBalanceTests(unittest.TestCase):
    def _write_georeferenced_product(
        self,
        product_dir: Path,
        product_id: str,
        *,
        origin_x: float,
        origin_y: float = 200.0,
    ) -> None:
        product_dir.mkdir(parents=True)
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
            1,
            3072,
            0,
            1,
            32613,
        )
        tifffile.imwrite(
            product_dir / f"{product_id}.tif",
            np.zeros((3, 4), dtype=np.uint16),
            metadata=None,
            extratags=[
                (33550, "d", 3, (20.0, 20.0, 0.0), False),
                (
                    33922,
                    "d",
                    6,
                    (0.0, 0.0, 0.0, origin_x, origin_y, 0.0),
                    False,
                ),
                (34735, "H", len(geokeys), geokeys, False),
            ],
        )
        (product_dir / f"{product_id}.txt").write_text(
            f'ProductID = "{product_id}"\n', encoding="utf-8"
        )
        (product_dir / f"{product_id}_B.csv").write_text(
            "BandNo,CenterWavelengthNanometer,FullWidthAtHalfMaximumNanometer\n",
            encoding="utf-8",
        )

    def _write_spatial_union_batch(
        self, root: Path, *, second_origin_x: float = 140.0
    ) -> tuple[Path, list[str]]:
        batch = root / "union_batch"
        batch.mkdir()
        product_ids = ["HSHL1G_union_a", "HSHL1G_union_b"]
        timestamps = [
            "2022-01-01T00:00:05Z",
            "2022-01-01T00:00:55Z",
        ]
        (batch / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "started_utc": "2026-07-28T00:00:00+00:00",
                    "finished_utc": "2026-07-28T01:00:00+00:00",
                    "current_product_ids": product_ids,
                    "completed_product_ids": product_ids,
                }
            ),
            encoding="utf-8",
        )
        (batch / "batch_summary.json").write_text(
            json.dumps(
                {
                    "scene_count": 2,
                    "current_product_ids": product_ids,
                    "analysis_config": {"cluster_threshold": 3.0},
                }
            ),
            encoding="utf-8",
        )
        for index, (product_id, timestamp) in enumerate(
            zip(product_ids, timestamps)
        ):
            product_dir = root / "products" / product_id
            self._write_georeferenced_product(
                product_dir,
                product_id,
                origin_x=100.0 if index == 0 else second_origin_x,
            )
            scene_dir = batch / product_id
            scene_dir.mkdir()
            summary = {
                "product_id": product_id,
                "product_path": str(product_dir),
                "acquisition_utc": timestamp,
                "quality_class": "usable",
                "official_scoring_eligible": True,
                "positive_component_count": 1,
                "reverse_control_component_count": 1,
                "analysis_config": {
                    "cluster_threshold": 3.0,
                    "minimum_cluster_pixels": 3,
                },
                "model": {
                    "cluster_threshold": 3.0,
                    "minimum_cluster_pixels": 3,
                },
            }
            (scene_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            valid = np.ones((3, 4), dtype=bool)
            weak = np.zeros((3, 4), dtype=float)
            strong = np.zeros((3, 4), dtype=float)
            if index == 0:
                weak[0, 1:4] = strong[0, 1:4] = 4.0
            else:
                weak[0, 0:3] = strong[0, 0:3] = 4.0
            weak[2, 0:3] = strong[2, 0:3] = -4.0
            np.savez_compressed(
                scene_dir / "score_maps.npz",
                valid=valid,
                weak_local_z=weak,
                strong_local_z=strong,
            )
            with (scene_dir / "candidate_components.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["product_id", "tail", "pixel_count"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "product_id": product_id,
                            "tail": "positive_ch4",
                            "pixel_count": 3,
                        },
                        {
                            "product_id": product_id,
                            "tail": "reverse_sign_control",
                            "pixel_count": 3,
                        },
                    ]
                )
        return batch, product_ids

    def _write_batch(self, root: Path) -> tuple[Path, str, str, str]:
        batch = root / "batch"
        batch.mkdir()
        scored = "HSHL1G_scored"
        excluded = "HSHL1G_cloud"
        qa_incomplete = "HSHL1G_qa"
        product_ids = [scored, excluded, qa_incomplete]
        manifest = {
            "status": "complete",
            "started_utc": "2026-07-28T00:00:00+00:00",
            "finished_utc": "2026-07-28T01:00:00+00:00",
            "current_product_ids": product_ids,
            "completed_product_ids": product_ids,
            "stale_scene_directories": [str(batch / "HSHL1G_stale")],
        }
        (batch / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        batch_summary = {
            "scene_count": 3,
            "current_product_ids": product_ids,
            "analysis_config": {"cluster_threshold": 3.0},
        }
        (batch / "batch_summary.json").write_text(
            json.dumps(batch_summary), encoding="utf-8"
        )

        common = {
            "analysis_config": {
                "cluster_threshold": 3.0,
                "minimum_cluster_pixels": 2,
            }
        }
        scored_dir = batch / scored
        scored_dir.mkdir()
        scored_summary = {
            **common,
            "product_id": scored,
            "product_path": str(root / "products" / scored),
            "acquisition_utc": "2022-01-01T00:00:00Z",
            "quality_class": "usable",
            "official_scoring_eligible": True,
            "positive_component_count": 2,
            "reverse_control_component_count": 1,
            "model": {"cluster_threshold": 3.0, "minimum_cluster_pixels": 2},
        }
        (scored_dir / "summary.json").write_text(
            json.dumps(scored_summary), encoding="utf-8"
        )
        valid = np.asarray(
            [[True, True, True, True], [True, True, True, False], [True, True, True, True]]
        )
        weak = np.asarray(
            [[4, 4, -4, -4], [4, 2, -5, 99], [np.nan, 4, 10, -4]],
            dtype=float,
        )
        strong = np.asarray(
            [[5, 4, -5, -3], [3.1, 4, -4, 99], [1, np.nan, 10, -4]],
            dtype=float,
        )
        np.savez_compressed(
            scored_dir / "score_maps.npz",
            valid=valid,
            weak_local_z=weak,
            strong_local_z=strong,
        )
        with (scored_dir / "candidate_components.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["product_id", "tail", "pixel_count"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"product_id": scored, "tail": "positive_ch4", "pixel_count": 3},
                    {"product_id": scored, "tail": "positive_ch4", "pixel_count": 2},
                    {
                        "product_id": scored,
                        "tail": "reverse_sign_control",
                        "pixel_count": 4,
                    },
                ]
            )

        for product_id, quality in (
            (excluded, "excluded_cloud"),
            (qa_incomplete, "qa_incomplete"),
        ):
            scene_dir = batch / product_id
            scene_dir.mkdir()
            summary = {
                **common,
                "product_id": product_id,
                "product_path": str(root / "products" / product_id),
                "acquisition_utc": None,
                "quality_class": quality,
                "official_scoring_eligible": False,
            }
            (scene_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

        stale_dir = batch / "HSHL1G_stale"
        stale_dir.mkdir()
        np.savez_compressed(
            stale_dir / "score_maps.npz",
            valid=np.ones((100, 100), dtype=bool),
            weak_local_z=np.full((100, 100), 99.0),
            strong_local_z=np.full((100, 100), 99.0),
        )
        return batch, scored, excluded, qa_incomplete

    def test_counts_rates_components_and_explicit_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, scored, excluded, qa_incomplete = self._write_batch(root)
            result = summarize_batch(batch)

            self.assertEqual(
                [row["product_id"] for row in result["per_scene"]],
                [scored, excluded, qa_incomplete],
            )
            scored_row = result["per_scene"][0]
            self.assertEqual(scored_row["analysis_valid_pixels"], 9)
            self.assertEqual(scored_row["positive_tail_pixels"], 4)
            self.assertEqual(scored_row["reverse_tail_pixels"], 4)
            self.assertAlmostEqual(
                scored_row["positive_tail_rate_per_million_analysis_valid"],
                4e6 / 9,
            )
            self.assertEqual(scored_row["positive_components_ge3"], 1)
            self.assertEqual(scored_row["positive_component_pixels_ge3"], 3)
            self.assertEqual(scored_row["reverse_components_ge3"], 1)
            self.assertEqual(scored_row["reverse_component_pixels_ge3"], 4)
            self.assertEqual(result["aggregate"]["analysis_valid_pixels"], 9)
            self.assertEqual(result["aggregate"]["positive_tail_pixels"], 4)
            self.assertEqual(result["aggregate"]["reverse_tail_pixels"], 4)
            self.assertEqual(result["aggregate"]["qa_incomplete_scene_count"], 1)
            self.assertEqual(result["provenance"]["stale_scene_directories_read"], [])
            for row in result["per_scene"][1:]:
                self.assertTrue(row["status"].startswith("not_scored_"))
                self.assertIsNone(row["analysis_valid_pixels"])
                self.assertIsNone(row["positive_components_ge3"])

            self.assertTrue((batch / "tail_balance_by_scene.csv").is_file())
            written = json.loads(
                (batch / "tail_balance_summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("not FDR-controlled", written["descriptive_not_fdr_caveat"])

    def test_custom_output_paths_are_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, *_ = self._write_batch(root)
            csv_output = root / "elsewhere" / "balance.csv"
            json_output = root / "elsewhere" / "balance.json"
            result = summarize_batch(
                batch, csv_output=csv_output, json_output=json_output
            )
            self.assertTrue(csv_output.is_file())
            self.assertTrue(json_output.is_file())
            self.assertEqual(result["outputs"]["csv"], str(csv_output.resolve()))

    def test_manifest_must_be_complete_and_match_current_products(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, *_ = self._write_batch(root)
            manifest_path = batch / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status must be 'complete'"):
                summarize_batch(batch)

            manifest["status"] = "complete"
            manifest["completed_product_ids"] = manifest[
                "completed_product_ids"
            ][:-1]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must exactly match"):
                summarize_batch(batch)

    def test_threshold_and_product_provenance_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, scored, *_ = self._write_batch(root)
            summary_path = batch / scored / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["model"]["cluster_threshold"] = 3.5
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cluster thresholds differ"):
                summarize_batch(batch)

            summary["model"]["cluster_threshold"] = 3.0
            summary["product_id"] = "HSHL1G_wrong"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary product_id"):
                summarize_batch(batch)

    def test_candidate_component_product_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, scored, *_ = self._write_batch(root)
            component_path = batch / scored / "candidate_components.csv"
            text = component_path.read_text(encoding="utf-8")
            component_path.write_text(
                text.replace(scored, "HSHL1G_wrong", 1), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "candidate component row"):
                summarize_batch(batch)

    def test_scored_scene_requires_saved_score_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, scored, *_ = self._write_batch(root)
            (batch / scored / "score_maps.npz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "--save-score-maps"):
                summarize_batch(batch)

    def test_spatial_union_deduplicates_projected_pixels_within_utc_minute(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, product_ids = self._write_spatial_union_batch(root)
            result = summarize_batch(batch, spatial_union=True)

            spatial = result["spatial_union_by_acquisition_strip"]
            aggregate = spatial["aggregate"]
            self.assertEqual(aggregate["acquisition_strip_count"], 1)
            self.assertEqual(aggregate["multi_product_strip_count"], 1)
            self.assertEqual(aggregate["product_analysis_valid_pixels"], 24)
            self.assertEqual(aggregate["analysis_valid_pixels"], 18)
            self.assertEqual(aggregate["duplicate_product_pixels"], 6)
            self.assertAlmostEqual(
                aggregate["duplicate_fraction_of_product_analysis_valid"], 0.25
            )
            self.assertEqual(aggregate["positive_tail_pixels"], 4)
            self.assertEqual(aggregate["duplicate_positive_product_pixels"], 2)
            self.assertEqual(aggregate["reverse_tail_pixels"], 5)
            self.assertEqual(aggregate["duplicate_reverse_product_pixels"], 1)
            self.assertEqual(aggregate["positive_components_ge3"], 1)
            self.assertEqual(aggregate["positive_component_pixels_ge3"], 4)
            self.assertEqual(aggregate["reverse_components_ge3"], 1)
            self.assertEqual(aggregate["reverse_component_pixels_ge3"], 5)

            strip = spatial["per_strip"][0]
            self.assertEqual(strip["acquisition_utc_minute"], "2022-01-01T00:00Z")
            self.assertEqual(strip["time_span_seconds"], 50.0)
            self.assertEqual(strip["product_ids"], product_ids)
            self.assertEqual(strip["union_grid_height"], 3)
            self.assertEqual(strip["union_grid_width"], 6)
            self.assertEqual(
                strip["union_normalized_gdal_geotransform"],
                [100.0, 20.0, 0.0, 200.0, 0.0, -20.0],
            )
            self.assertEqual(
                [source["union_column_offset"] for source in strip["source_products"]],
                [0, 2],
            )
            self.assertEqual(strip["source_products"][0]["epsg"], 32613)
            self.assertEqual(strip["source_products"][0]["source_raster_type"], 1)
            self.assertIn("not an authoritative", spatial["grouping_heuristic"])
            strip_csv = batch / "tail_balance_by_acquisition_strip.csv"
            self.assertTrue(strip_csv.is_file())
            with strip_csv.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["product_ids"], ";".join(product_ids))

    def test_spatial_union_rejects_noninteger_grid_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, _product_ids = self._write_spatial_union_batch(
                root, second_origin_x=130.0
            )
            expected_outputs = (
                batch / "tail_balance_by_scene.csv",
                batch / "tail_balance_summary.json",
                batch / "tail_balance_by_acquisition_strip.csv",
            )
            for output in expected_outputs:
                output.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not aligned"):
                summarize_batch(batch, spatial_union=True)
            self.assertFalse(any(output.exists() for output in expected_outputs))

    def test_strip_csv_output_requires_spatial_union(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch, *_ = self._write_batch(root)
            with self.assertRaisesRegex(ValueError, "requires spatial_union"):
                summarize_batch(
                    batch,
                    strip_csv_output=root / "strip.csv",
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import tifffile


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_hisui_candidate_crop import (  # noqa: E402
    browse_crop,
    load_scene_crop_source,
    make_candidate_crop,
    resolve_selection,
)


PRODUCT_ID = "HSHL1G_N320W1032_test"
ACQUISITION = "2022-10-30T17:00:00Z"


def _write_product(root: Path, shape: tuple[int, int] = (8, 12)) -> Path:
    product = root / PRODUCT_ID
    product.mkdir()
    tifffile.imwrite(product / f"{PRODUCT_ID}.tif", np.zeros(shape, dtype=np.uint16))
    (product / f"{PRODUCT_ID}.txt").write_text(
        f'SceneCenterTime = "{ACQUISITION}"\n', encoding="utf-8"
    )
    (product / f"{PRODUCT_ID}_B.csv").write_text(
        "BandNo,CenterWavelengthNanometer,FullWidthAtHalfMaximumNanometer\n",
        encoding="utf-8",
    )
    return product


def _write_scene(
    root: Path,
    product: Path,
    shape: tuple[int, int] = (8, 12),
    *,
    summary_product_id: str = PRODUCT_ID,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    scene = root / "scene"
    scene.mkdir()
    yy, xx = np.indices(shape)
    weak = (0.3 * yy + 0.1 * xx).astype(float)
    strong = (0.2 * yy + 0.15 * xx + 0.5).astype(float)
    weak[3, 7], strong[3, 7] = 4.4, 4.0
    weak[5, 8], strong[5, 8] = 6.0, 5.5
    weak[2, 4], strong[2, 4] = -6.0, -5.8
    dual = np.minimum(weak, strong)
    valid = np.ones(shape, dtype=bool)
    cloud = np.zeros(shape, dtype=bool)
    valid[0, 0] = False
    cloud[0, 0] = True
    valid[1, 1] = False
    weak[~valid] = np.nan
    strong[~valid] = np.nan
    dual[~valid] = np.nan
    np.savez_compressed(
        scene / "score_maps.npz",
        valid=valid,
        cloud_proxy=cloud,
        weak_local_z=weak,
        strong_local_z=strong,
        dual_local_z=dual,
    )
    (scene / "summary.json").write_text(
        json.dumps(
            {
                "product_id": summary_product_id,
                "product_path": str(product.resolve()),
                "acquisition_utc": ACQUISITION,
                "processing_utc": "2022-11-01T00:00:00Z",
                "quality_class": "usable",
                "shape": [shape[0], shape[1], 185],
                "analysis_config": {
                    "cluster_threshold": 3.0,
                    "cloud_profile": "cirrus_sensitive",
                },
            }
        ),
        encoding="utf-8",
    )
    return scene, weak, strong, dual


def _write_candidates(scene: Path) -> None:
    rows = [
        {
            "tail": "positive_ch4",
            "component_id": 2,
            "pixel_count": 4,
            "peak_y": 3,
            "peak_x": 7,
            "dual_local_z_peak": 4.0,
            "scene_spanning_line_flag": False,
            "product_id": PRODUCT_ID,
            "acquisition_utc": ACQUISITION,
        },
        {
            "tail": "positive_ch4",
            "component_id": 1,
            "pixel_count": 5,
            "peak_y": 5,
            "peak_x": 8,
            "dual_local_z_peak": 5.5,
            "scene_spanning_line_flag": True,
            "product_id": PRODUCT_ID,
            "acquisition_utc": ACQUISITION,
        },
        {
            "tail": "reverse_sign_control",
            "component_id": 3,
            "pixel_count": 3,
            "peak_y": 2,
            "peak_x": 4,
            "dual_local_z_peak": 5.8,
            "scene_spanning_line_flag": False,
            "product_id": PRODUCT_ID,
            "acquisition_utc": ACQUISITION,
        },
    ]
    with (scene / "candidate_components.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PlotHISUICandidateCropTests(unittest.TestCase):
    def test_browse_crop_uses_native_grid_without_resampling(self) -> None:
        image = np.arange(8 * 12).reshape(8, 12)
        crop, mapping = browse_crop(image, (8, 12), (2, 6, 3, 9))
        np.testing.assert_array_equal(crop, image[2:6, 3:9])
        self.assertEqual(mapping["mode"], "native_pixel_grid")

    def test_browse_crop_records_proportional_resampling(self) -> None:
        image = np.arange(4 * 6).reshape(4, 6)
        crop, mapping = browse_crop(image, (8, 12), (0, 4, 0, 4))
        self.assertEqual(crop.shape, (4, 4))
        np.testing.assert_array_equal(crop[:, 0], np.asarray([0, 0, 6, 6]))
        self.assertEqual(
            mapping["mode"], "axis_aligned_proportional_nearest_resample"
        )
        self.assertEqual(mapping["browse_pixels_per_score_pixel_y"], 0.5)
        self.assertIn("not independently georeferenced", mapping["assumption"])

    def test_candidate_rank_is_sorted_within_requested_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = _write_product(root)
            scene, _, _, _ = _write_scene(root, product)
            _write_candidates(scene)
            source = load_scene_crop_source(product, scene)
            selected = resolve_selection(
                source,
                center_y=None,
                center_x=None,
                candidate_rank=1,
                tail="positive_ch4",
            )
            self.assertEqual((selected.center_y, selected.center_x), (5, 8))
            self.assertEqual(selected.candidate["component_id"], 1)
            self.assertTrue(selected.candidate["scene_spanning_line_flag"])

    def test_explicit_center_requires_both_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = _write_product(root)
            scene, _, _, _ = _write_scene(root, product)
            source = load_scene_crop_source(product, scene)
            with self.assertRaisesRegex(ValueError, "supplied together"):
                resolve_selection(
                    source,
                    center_y=3,
                    center_x=None,
                    candidate_rank=None,
                    tail="positive_ch4",
                )

    def test_product_id_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = _write_product(root)
            scene, _, _, _ = _write_scene(
                root, product, summary_product_id="HSHL1G_wrong"
            )
            with self.assertRaisesRegex(ValueError, "does not match product"):
                load_scene_crop_source(product, scene)

    def test_end_to_end_scaled_browse_writes_png_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = _write_product(root)
            scene, _, _, _ = _write_scene(root, product)
            _write_candidates(scene)
            browse = np.zeros((4, 6, 3), dtype=float)
            browse[..., 0] = np.linspace(0.0, 1.0, 6)
            browse[..., 1] = np.linspace(0.0, 1.0, 4)[:, None]
            mpimg.imsave(product / f"{PRODUCT_ID}_1.jpg", browse)

            result = make_candidate_crop(
                product_path=product,
                scene_output=scene,
                output_dir=root / "output",
                candidate_rank=1,
                tail="positive_ch4",
                half_size=2,
            )
            figure = Path(result["outputs"]["figure"])
            sidecar = Path(result["outputs"]["json"])
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)
            self.assertTrue(sidecar.is_file())
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["selection"]["candidate_row"]["dual_local_z_peak"], 5.5
            )
            self.assertEqual(
                loaded["browse_mapping"]["mode"],
                "axis_aligned_proportional_nearest_resample",
            )
            self.assertEqual(
                loaded["crop"]["bounds_y0_y1_x0_x1_exclusive"], [3, 8, 6, 11]
            )
            self.assertTrue(
                loaded["provenance_verification"]["acquisition_time_match"]
            )
            self.assertIn("wind", loaded["interpretation"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hisui_l1g_io import (  # noqa: E402
    HISUIL1GReader,
    discover_l1g_product,
    discover_l1g_products,
    parse_metadata,
    read_band_metadata,
    read_georeference,
    read_tiled_any_mask,
)


class HISUIL1GIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.product_id = "HSHL1G_N320W1032_20221030160051_TEST"
        self.product_dir = self.root / self.product_id
        self.product_dir.mkdir()
        self.image_path = self.product_dir / f"{self.product_id}.tif"
        self.metadata_path = self.product_dir / f"{self.product_id}.txt"
        self.band_path = self.product_dir / f"{self.product_id}_B.csv"

        height, width, bands = 35, 29, 60
        self.data = (
            np.arange(height * width * bands, dtype=np.uint32).reshape(height, width, bands)
            % 50_000
            + 2
        ).astype(np.uint16)
        self.data[2, 3, 0] = 0
        self.data[2, 3, 57] = 1
        self.data[2, 3, 58] = 65535

        geokeys = (
            1,
            1,
            0,
            5,
            1024,
            0,
            1,
            1,
            1025,
            0,
            1,
            1,
            2054,
            0,
            1,
            9102,
            3072,
            0,
            1,
            32613,
            3076,
            0,
            1,
            9001,
        )
        tifffile.imwrite(
            self.image_path,
            self.data,
            tile=(16, 16),
            planarconfig="contig",
            photometric="minisblack",
            compression=None,
            metadata=None,
            extratags=[
                (33550, "d", 3, (20.0, 20.0, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0), False),
                (34735, "H", len(geokeys), geokeys, False),
            ],
        )
        self.metadata_path.write_text(
            "\n".join(
                [
                    "################ Product information ################",
                    f'ProductID = "{self.product_id}"',
                    'Comment = "a=b is retained"',
                    "NumberOfBands = 60",
                    "RadianceMultiVNIR = 1.000000e-02",
                    "RadianceAddVNIR = -10.000000",
                    "RadianceMultiSWIR = 3.200000e-03",
                    "RadianceAddSWIR = -3.200000",
                    "BadPixelDN = 1",
                    "SaturatedPixelDN = 65535",
                    "SceneCenterTime = 2022-10-30T16:00:51.479728Z",
                ]
            ),
            encoding="utf-8",
        )
        with self.band_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "BandNo",
                    "CenterWavelengthNanometer",
                    "FullWidthAtHalfMaximumNanometer",
                ]
            )
            for band in range(bands):
                writer.writerow([band + 1, 400.0 + 10.0 * band, 9.5 + 0.01 * band])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_discovery_metadata_and_band_csv(self) -> None:
        product = discover_l1g_product(self.product_dir)
        self.assertEqual(product.product_id, self.product_id)
        self.assertEqual(discover_l1g_product(self.image_path), product)
        self.assertEqual(discover_l1g_product(self.band_path), product)
        self.assertEqual(discover_l1g_products(self.root), [product])

        metadata = parse_metadata(product.metadata_path)
        self.assertEqual(metadata["ProductID"], self.product_id)
        self.assertEqual(metadata["Comment"], "a=b is retained")
        self.assertEqual(metadata["NumberOfBands"], 60)
        self.assertAlmostEqual(metadata["RadianceMultiSWIR"], 0.0032)
        self.assertEqual(metadata["SceneCenterTime"], "2022-10-30T16:00:51.479728Z")

        bands = read_band_metadata(product.band_csv_path)
        self.assertEqual(len(bands), 60)
        np.testing.assert_array_equal(bands.band_number[[0, -1]], [1, 60])
        np.testing.assert_allclose(bands.center_wavelength_nm[[0, 59]], [400.0, 990.0])
        np.testing.assert_allclose(bands.fwhm_nm[[0, 59]], [9.5, 10.09])

    def test_geotransform_and_crs(self) -> None:
        georef = read_georeference(self.image_path)
        self.assertEqual(georef.crs, "EPSG:32613")
        self.assertEqual(georef.epsg, 32613)
        self.assertEqual(
            georef.geotransform,
            (500000.0, 20.0, 0.0, 4000000.0, 0.0, -20.0),
        )
        self.assertEqual(georef.map_xy(2, 3), (500060.0, 3999960.0))
        self.assertEqual(georef.map_xy(0, 0, pixel_center=True), (500010.0, 3999990.0))

    def test_raster_pixel_is_point_is_normalized_to_corner_affine(self) -> None:
        point_path = self.root / "point.tif"
        geokeys = (
            1, 1, 0, 3,
            1024, 0, 1, 1,
            1025, 0, 1, 2,
            3072, 0, 1, 32613,
        )
        tifffile.imwrite(
            point_path,
            np.zeros((16, 16), dtype=np.uint16),
            metadata=None,
            extratags=[
                (33550, "d", 3, (20.0, 20.0, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0), False),
                (34735, "H", len(geokeys), geokeys, False),
            ],
        )
        georef = read_georeference(point_path)
        self.assertEqual(georef.raster_type, 2)
        self.assertEqual(
            georef.geotransform,
            (499990.0, 20.0, 0.0, 4000010.0, 0.0, -20.0),
        )
        self.assertEqual(georef.map_xy(0, 0, pixel_center=True), (500000.0, 4000000.0))

    def test_tiled_any_mask_reads_only_requested_region_and_bands(self) -> None:
        qa_path = self.root / "qa_bool.tif"
        qa = np.zeros((35, 29, 8), dtype=np.uint8)
        qa[7, 9, 3] = True
        qa[8, 10, 5] = True
        qa[9, 11, 1] = True
        tifffile.imwrite(
            qa_path,
            qa,
            tile=(16, 16),
            planarconfig="contig",
            photometric="minisblack",
            compression=None,
            metadata=None,
        )
        actual = read_tiled_any_mask(
            qa_path, [3, 5], bounds=(5, 12, 6, 14)
        )
        expected = np.any(qa[5:12, 6:14][..., [3, 5]], axis=-1)
        np.testing.assert_array_equal(actual, expected)

    def test_read_region_decodes_only_tiles_and_preserves_band_order(self) -> None:
        selected = [59, 0, 58, 5]
        with HISUIL1GReader(self.product_dir) as reader:
            self.assertEqual(reader.shape, self.data.shape)
            self.assertEqual((reader.tile_height, reader.tile_width), (16, 16))
            with patch.object(
                tifffile.TiffPage,
                "asarray",
                side_effect=AssertionError("full-page read is forbidden"),
            ):
                actual = reader.read_region(5, 25, 7, 24, selected)
        expected = self.data[5:25, 7:24, :][..., selected]
        np.testing.assert_array_equal(actual, expected)

    def test_dn_to_radiance_uses_vnir_swir_split_and_masks_quality_dns(self) -> None:
        selected = [0, 57, 58, 59]
        with HISUIL1GReader(self.product_dir) as reader:
            radiance = reader.read_region(2, 4, 3, 5, selected, as_radiance=True)

        raw = self.data[2:4, 3:5, :][..., selected]
        expected = raw.astype(float)
        expected[..., :2] = expected[..., :2] * 0.01 - 10.0
        expected[..., 2:] = expected[..., 2:] * 0.0032 - 3.2
        expected[np.isin(raw, [0, 1, 65535])] = np.nan
        np.testing.assert_allclose(radiance, expected, equal_nan=True)
        self.assertTrue(np.isnan(radiance[0, 0, :3]).all())
        self.assertTrue(np.isfinite(radiance[0, 0, 3]))

    def test_iter_row_blocks_uses_tile_aligned_boundaries(self) -> None:
        selected = [0, 58]
        with HISUIL1GReader(self.product_dir) as reader:
            blocks = list(reader.iter_row_blocks(selected, tile_rows_per_block=1))
            with self.assertRaisesRegex(ValueError, "tile-row boundary"):
                list(reader.iter_row_blocks(selected, y0=1))

        self.assertEqual([(y0, y1) for y0, y1, _ in blocks], [(0, 16), (16, 32), (32, 35)])
        combined = np.concatenate([block for _, _, block in blocks], axis=0)
        np.testing.assert_array_equal(combined, self.data[..., selected])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_hisui_region_spectra import (  # noqa: E402
    bounds_from_center,
    export_hisui_region_spectra,
    main,
    sidecar_path_for,
)


class ExportHISUIRegionSpectraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.product_id = "HSHL1G_N320W1032_20221030160051_EXPORT_TEST"
        self.product_dir = self.root / self.product_id
        self.product_dir.mkdir()
        self.image_path = self.product_dir / f"{self.product_id}.tif"
        self.metadata_path = self.product_dir / f"{self.product_id}.txt"
        self.band_path = self.product_dir / f"{self.product_id}_B.csv"

        height, width, bands = 20, 18, 60
        self.data = (
            np.arange(height * width * bands, dtype=np.uint32)
            .reshape(height, width, bands)
            .__mod__(40_000)
            + 2
        ).astype(np.uint16)
        self.data[4, 5, 0] = 0
        self.data[6, 7, 58] = 1
        self.data[8, 9, 59] = 65535

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
                    f'ProductID = "{self.product_id}"',
                    'ProductVersion = "3.1.0"',
                    "SceneCenterTime = 2022-10-30T16:00:51.479728Z",
                    "RadianceMultiVNIR = 1.000000e-02",
                    "RadianceAddVNIR = -10.000000",
                    "RadianceMultiSWIR = 3.200000e-03",
                    "RadianceAddSWIR = -3.200000",
                    'RadianceUnit = "W/m2/micron/sr"',
                    "BadPixelDN = 1",
                    "SaturatedPixelDN = 65535",
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

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        return rows[0], rows[1:]

    def test_streamed_relative_export_drops_invalid_and_writes_provenance(self) -> None:
        output = self.root / "relative.csv"
        bounds = (2, 10, 3, 12)
        with patch.object(
            tifffile.TiffPage,
            "asarray",
            side_effect=AssertionError("full-page read is forbidden"),
        ):
            summary = export_hisui_region_spectra(
                self.product_dir,
                output,
                bounds=bounds,
                tile_rows_per_block=1,
            )

        header, rows = self._read_csv(output)
        self.assertEqual(header[:2], ["y", "x"])
        self.assertEqual(header[2], "wave_400.00nm")
        self.assertEqual(header[-1], "wave_990.00nm")
        self.assertEqual(len(header), 62)
        self.assertEqual(len(rows), 8 * 9 - 3)
        self.assertEqual(rows[0][:2], ["0", "0"])
        self.assertEqual(rows[-1][:2], ["7", "8"])

        source = self.data[2, 3]
        self.assertAlmostEqual(float(rows[0][2]), float(source[0]) * 0.01 - 10.0)
        self.assertAlmostEqual(float(rows[0][-1]), float(source[59]) * 0.0032 - 3.2)
        self.assertEqual(summary["requested_pixels"], 72)
        self.assertEqual(summary["emitted_pixels"], 69)
        self.assertEqual(summary["invalid_pixels"], 3)

        sidecar = json.loads(sidecar_path_for(output).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["product"]["product_id"], self.product_id)
        self.assertEqual(sidecar["product"]["product_version"], "3.1.0")
        self.assertEqual(sidecar["export"]["coordinate_mode"], "relative")
        self.assertEqual(
            sidecar["export"]["bounds_half_open"],
            {"y0": 2, "y1": 10, "x0": 3, "x1": 12},
        )
        self.assertEqual(sidecar["export"]["relative_origin"], {"y": 2, "x": 3})
        self.assertEqual(len(sidecar["bands"]), 60)
        self.assertAlmostEqual(sidecar["bands"][58]["fwhm_nm"], 10.08)
        self.assertEqual(sidecar["calibration"]["radiance_unit"], "W/m2/micron/sr")
        self.assertEqual(sidecar["georeference"]["epsg"], 32613)
        self.assertEqual(
            sidecar["georeference"]["affine_gdal_order"],
            [500000.0, 20.0, 0.0, 4000000.0, 0.0, -20.0],
        )
        self.assertEqual(sidecar["invalid"]["invalid_pixel_count"], 3)
        self.assertEqual(sidecar["invalid"]["invalid_sample_count"], 3)
        self.assertEqual(
            sidecar["invalid"]["invalid_sample_counts_by_dn"],
            {"0": 1, "1": 1, "65535": 1},
        )

    def test_center_absolute_export_filters_wavelengths_and_keeps_nan(self) -> None:
        output = self.root / "absolute.csv"
        bounds = bounds_from_center(7, 8, 8, 8)
        self.assertEqual(bounds, (3, 11, 4, 12))
        export_hisui_region_spectra(
            self.image_path,
            output,
            bounds=bounds,
            absolute_coordinates=True,
            wave_min_nm=980.0,
            wave_max_nm=990.0,
            drop_invalid=False,
        )

        header, rows = self._read_csv(output)
        self.assertEqual(header, ["y", "x", "wave_980.00nm", "wave_990.00nm"])
        self.assertEqual(len(rows), 64)
        self.assertEqual(rows[0][:2], ["3", "4"])
        self.assertEqual(rows[-1][:2], ["10", "11"])
        by_coordinate = {(int(row[0]), int(row[1])): row for row in rows}
        self.assertEqual(by_coordinate[(6, 7)][2].lower(), "nan")
        self.assertEqual(by_coordinate[(8, 9)][3].lower(), "nan")

        sidecar = json.loads(sidecar_path_for(output).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["export"]["coordinate_mode"], "absolute")
        self.assertEqual(sidecar["invalid"]["requested_pixel_count"], 64)
        self.assertEqual(sidecar["invalid"]["emitted_pixel_count"], 64)
        self.assertEqual(sidecar["invalid"]["dropped_pixel_count"], 0)
        self.assertEqual(sidecar["invalid"]["invalid_pixel_count"], 2)

    def test_duplicate_rounded_wavelength_names_fail_without_outputs(self) -> None:
        with self.band_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        rows[1][1] = "400.001"
        rows[2][1] = "400.004"
        with self.band_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

        output = self.root / "duplicate.csv"
        with self.assertRaisesRegex(ValueError, "duplicate CSV column names"):
            export_hisui_region_spectra(
                self.product_dir,
                output,
                bounds=(0, 4, 0, 4),
                wave_min_nm=400.0,
                wave_max_nm=401.0,
            )
        self.assertFalse(output.exists())
        self.assertFalse(sidecar_path_for(output).exists())

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        output = self.root / "overwrite.csv"
        export_hisui_region_spectra(
            self.product_dir, output, bounds=(0, 2, 0, 2)
        )
        original = output.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            export_hisui_region_spectra(
                self.product_dir, output, bounds=(0, 3, 0, 3)
            )
        self.assertEqual(output.read_bytes(), original)

        summary = export_hisui_region_spectra(
            self.product_dir,
            output,
            bounds=(0, 3, 0, 3),
            overwrite=True,
        )
        self.assertEqual(summary["emitted_pixels"], 9)

    def test_cli_rejects_incomplete_or_conflicting_roi_modes(self) -> None:
        common = [
            "--input",
            str(self.product_dir),
            "--output-csv",
            str(self.root / "cli.csv"),
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as incomplete:
            main([*common, "--center-y", "5"])
        self.assertEqual(incomplete.exception.code, 2)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as conflicting:
            main(
                [
                    *common,
                    "--bounds",
                    "0",
                    "2",
                    "0",
                    "2",
                    "--center-y",
                    "1",
                    "--center-x",
                    "1",
                    "--height",
                    "2",
                    "--width",
                    "2",
                ]
            )
        self.assertEqual(conflicting.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

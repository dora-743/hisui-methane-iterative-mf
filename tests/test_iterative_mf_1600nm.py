from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from iterative_mf_1600nm import (
    iterative_matched_filter,
    load_ch4_lut,
    make_valid_mask,
)


class IterativeMF1600Tests(unittest.TestCase):
    def test_lut_accepts_waveln_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lut.csv"
            pd.DataFrame(
                {
                    "waveln": [1.59, 1.61, 1.65, 1.69],
                    "0": [10.0, 10.0, 10.0, 10.0],
                    "0.5": [9.9, 9.8, 9.4, 9.8],
                }
            ).to_csv(path, index=False)
            wavelengths, alpha, spectra = load_ch4_lut(path)

        np.testing.assert_allclose(wavelengths, [1590.0, 1610.0, 1650.0, 1690.0])
        np.testing.assert_allclose(alpha, [0.0, 0.5])
        self.assertEqual(spectra.shape, (2, 4))

    def test_injected_methane_region_has_high_alpha(self) -> None:
        rng = np.random.default_rng(7)
        height, width, bands = 24, 24, 7
        background_mean = np.linspace(90.0, 110.0, bands)
        uas = np.asarray([0.001, 0.003, 0.008, 0.016, 0.010, 0.004, 0.001])
        cube = background_mean + rng.normal(0.0, 0.15, size=(height, width, bands))

        injected = np.zeros((height, width), dtype=bool)
        injected[9:14, 10:15] = True
        cube[injected] += 35.0 * (-background_mean * uas)

        valid = make_valid_mask(cube)
        result = iterative_matched_filter(
            cube, uas, valid, n_iter=6, nsigma=3.0, verbose=False
        )
        alpha_map = np.asarray(result["alpha_map"])
        plume = np.asarray(result["plume_mask"])

        self.assertGreater(float(np.mean(alpha_map[injected])), float(np.mean(alpha_map[~injected])) + 20)
        self.assertGreaterEqual(int(np.count_nonzero(plume & injected)), 20)


if __name__ == "__main__":
    unittest.main()

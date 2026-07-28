import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_r2_analysis_contexts import summarize_fixed_regions


class CompareR2AnalysisContextsTests(unittest.TestCase):
    def test_fixed_region_summary_and_agreement(self):
        base = np.arange(16, dtype=float).reshape(4, 4)
        valid = np.ones((4, 4), dtype=bool)
        roi = {"valid": valid}
        full = {"valid": valid.copy()}
        for metric in ("weak_z", "strong_z", "dual_min_z", "log_e_average"):
            roi[metric] = base.copy()
            full[metric] = base + 1.0

        summary, agreement = summarize_fixed_regions(
            roi, full, {"fixed": (1, 3, 1, 3)}
        )

        self.assertEqual(len(summary), 8)
        weak_roi = summary[
            (summary.region == "fixed")
            & (summary.context == "roi_run")
            & (summary.metric == "weak_z")
        ].iloc[0]
        self.assertEqual(weak_roi.valid_pixels, 4)
        self.assertAlmostEqual(weak_roi["mean"], 7.5)
        self.assertTrue(np.allclose(agreement.pearson_r, 1.0))
        self.assertTrue(np.allclose(agreement.median_absolute_difference, 1.0))

    def test_optional_strict_screen_is_summarized_without_changing_agreement(self):
        valid = np.ones((3, 3), dtype=bool)
        contexts = []
        for offset in (0.0, 1.0, 2.0):
            maps = {"valid": valid.copy()}
            for metric in ("weak_z", "strong_z", "dual_min_z", "log_e_average"):
                maps[metric] = np.arange(9, dtype=float).reshape(3, 3) + offset
            contexts.append(maps)

        summary, agreement = summarize_fixed_regions(
            contexts[0],
            contexts[1],
            {"fixed": (0, 2, 0, 2)},
            contexts[2],
        )

        self.assertEqual(len(summary), 12)
        strict = summary[
            (summary.context == "qa_strict_final_screen")
            & (summary.metric == "dual_min_z")
        ].iloc[0]
        self.assertAlmostEqual(strict["mean"], 4.0)
        self.assertTrue(np.allclose(agreement.pearson_r, 1.0))

    def test_region_outside_roi_is_rejected_instead_of_silently_clipped(self):
        valid = np.ones((3, 4), dtype=bool)
        maps = {"valid": valid}
        for metric in ("weak_z", "strong_z", "dual_min_z", "log_e_average"):
            maps[metric] = np.zeros_like(valid, dtype=float)

        with self.assertRaisesRegex(ValueError, "outside ROI shape"):
            summarize_fixed_regions(
                maps,
                maps,
                {"bad": (1, 4, 1, 3)},
            )


if __name__ == "__main__":
    unittest.main()

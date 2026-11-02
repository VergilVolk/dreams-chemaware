#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_gnps2_mega_frozen_scoring_axis import _panel
from score_gnps2_mega_frozen_full23 import top3_positions


class FrozenMegaScoringContracts(unittest.TestCase):
    def test_panel_keeps_global_rows_and_one_positive(self):
        rows = []
        # Three identities, two spectra each, same domain and mass window.
        for identity, formula, base in (("A"*14, "C1H2", 0), ("B"*14, "C1H2", 2), ("C"*14, "C2H4", 4)):
            for offset in range(2):
                rows.append({
                    "global_row": base + offset, "source_id": "s", "source_row": base + offset,
                    "record_token": f"r{base+offset}", "polarity": "positive", "adduct": "[M+H]+",
                    "ik14": identity, "formula": formula, "precursor_mz": 100.0 + offset * 1e-6,
                })
        arrays, report = _panel(pd.DataFrame(rows), ppm=10.0, seed=20261102)
        self.assertGreater(report["N_Q"], 0)
        self.assertTrue(np.all(np.add.reduceat(arrays["molecule_label"], arrays["query_ptr"][:-1]) == 1))
        self.assertTrue(set(arrays["query_row"]).issubset(set(range(6))))
        self.assertTrue(set(arrays["candidate_row"]).issubset(set(range(6))))

    def test_top3_reference_positions_are_score_ordered(self):
        scores = np.asarray([0.1, 0.8, 0.4, 0.2], dtype=np.float32)
        molecule_ptr = np.asarray([0, 3, 4])
        query_ptr = np.asarray([0, 2])
        query_rows = np.asarray([9])
        candidate_rows = np.asarray([1, 2, 3, 4])
        mapping = np.arange(10)
        query, references, mask = top3_positions(
            scores, molecule_ptr, query_ptr, query_rows, candidate_rows, mapping,
        )
        np.testing.assert_array_equal(query, [9, 9])
        np.testing.assert_array_equal(references[0], [2, 3, 1])
        np.testing.assert_array_equal(mask[0], [True, True, True])
        self.assertEqual(int(references[1, 0]), 4)
        self.assertEqual(int(mask[1].sum()), 1)


if __name__ == "__main__":
    unittest.main()

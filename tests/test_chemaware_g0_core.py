import json
import unittest
from pathlib import Path

import numpy as np

from tasks.chemaware_g0_core import (
    compile_rules,
    match_compiled_rules,
    nan_group_max,
    packed_jaccard,
    packed_mask,
)


ROOT = Path(__file__).resolve().parent.parent


class TestChemAwareG0Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core = json.loads(
            (ROOT / "dreams/models/chem_aware/chem_rules_data.json").read_text(encoding="utf-8")
        )["rules"]
        massbank = json.loads(
            (ROOT / "dreams/models/chem_aware/chem_rules_massbank.json").read_text(encoding="utf-8")
        )["rules"]
        cls.rules = core + massbank
        cls.compiled = compile_rules(cls.rules)

    def test_fragment_loss_uses_precursor_not_arbitrary_peak_pair(self):
        index = next(i for i, rule in enumerate(self.rules) if rule["name"] == "NL:H2O")
        peaks = np.zeros(100, dtype=np.float32)
        # 50 and 68.0106 create an H2O-sized peak-pair delta, but neither is
        # an H2O loss from precursor 100.
        peaks[:2] = [50.0, 68.0106]
        observed = match_compiled_rules(peaks, 100.0, self.compiled, parent_mass=98.9922)
        self.assertEqual(int(observed[index]), 0)
        peaks[2] = 81.9894
        observed = match_compiled_rules(peaks, 100.0, self.compiled, parent_mass=98.9922)
        self.assertEqual(int(observed[index]), 1)

    def test_massbank_offset_uses_parent_mass(self):
        index = next(
            i for i, rule in enumerate(self.rules)
            if rule["match_type"] == "mass_diff"
            and rule.get("source") == "MassBank record-derived"
        )
        target = float(self.rules[index]["value"])
        peaks = np.asarray([50.0, 75.0, 0.0, 0.0])
        observed = match_compiled_rules(
            peaks, 200.0, self.compiled, parent_mass=200.0 - target,
        )
        self.assertEqual(int(observed[index]), 1)

    def test_packed_jaccard_and_empty_union(self):
        values = np.asarray([
            [1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        packed = np.packbits(values, axis=1, bitorder="little")
        got = packed_jaccard(packed[[0, 2]], packed[[1, 2]])
        self.assertAlmostEqual(float(got[0]), 1.0 / 4.0)
        self.assertTrue(np.isnan(got[1]))
        mask = packed_mask(np.asarray([0, 1]), values.shape[1])
        masked = packed_jaccard(packed[[0]], packed[[1]], mask)
        self.assertAlmostEqual(float(masked[0]), 1.0 / 2.0)

    def test_nan_group_max_preserves_missing_groups(self):
        got = nan_group_max(
            np.asarray([np.nan, np.nan, 0.2, 0.7, np.nan]),
            np.asarray([0, 2, 4, 5]),
        )
        self.assertTrue(np.isnan(got[0]))
        self.assertAlmostEqual(float(got[1]), 0.7)
        self.assertTrue(np.isnan(got[2]))


if __name__ == "__main__":
    unittest.main()

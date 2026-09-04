from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tasks"))


def _load_pilot():
    spec = importlib.util.spec_from_file_location(
        "pilot_bioaware_metdna3_ms1_features",
        ROOT / "tasks/pilot_bioaware_metdna3_ms1_features.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT = _load_pilot()


def test_external_filename_prefix_does_not_change_named_fields() -> None:
    for filename, prefix in (
        ("BV2cell_neg_70_300_1.mzML", "BV2cell"),
        ("brain_pos_590_1200_2.mzML", "brain"),
        ("NIST_plasma_pos_290_600_1.mzML", "NIST_plasma"),
    ):
        match = PILOT.FILE_RE.match(filename)
        assert match is not None
        assert match.group("prefix") == prefix
        assert match.group("polarity") in {"pos", "neg"}
        assert match.group("window") in {"70_300", "70_1200", "290_600", "590_1200"}
        assert match.group("replicate") in {"1", "2"}

import pandas as pd

from tasks.audit_bioaware_metdna3_ms1_premapping import (
    MassCandidateIndex,
    enumerate_candidates,
)
from tasks.extract_metdna2_emrn_mass_adduct_index import (
    charge_magnitude,
    molecular_multiplier,
)


def test_adduct_stoichiometry_parser() -> None:
    assert molecular_multiplier("[M+H]+") == 1
    assert molecular_multiplier("[2M+H]+") == 2
    assert molecular_multiplier("[3M-H]-") == 3
    assert charge_magnitude("[M+H]+") == 1
    assert charge_magnitude("[M-2H]2-") == 2


def test_candidate_enumeration_is_mass_and_step_only() -> None:
    masses = pd.DataFrame({
        "inchikey1": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
        "monoisotopic_mass": [100.0, 100.0],
        "min_reaction_step": [0, 1],
    })
    adducts = pd.DataFrame({
        "polarity": ["positive"],
        "adduct": ["[M+H]+"],
        "delta_mz": [1.0],
        "nmol": [1],
        "charge": [1],
        "default_annotation": [True],
    })
    assert enumerate_candidates(101.0, "positive", 0, masses, adducts, 5.0, None) == {
        "AAAAAAAAAAAAAA"
    }
    assert enumerate_candidates(101.0, "positive", 1, masses, adducts, 5.0, None) == {
        "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"
    }
    index = MassCandidateIndex(masses, adducts)
    assert index.query(101.0, "positive", 0, 5.0, None) == {"AAAAAAAAAAAAAA"}
    assert index.query(101.0, "positive", 1, 5.0, "[M+H]+") == {
        "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"
    }

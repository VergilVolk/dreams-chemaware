from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCK = load("lock_bioaware_metdna3_external_source", "tasks/lock_bioaware_metdna3_external_source.py")
BUILD = load("build_bioaware_metdna3_development_manifest", "tasks/build_bioaware_metdna3_development_manifest.py")


def test_panel_classifier_is_specific_to_targeted_ms2() -> None:
    path = (
        "ccms_peak/MetDNA3 Exploris 480 datasets/NIST_urine_hilic/"
        "urine_neg_70_300_1.mzML"
    )
    assert LOCK.classify_panel(path) == ("NIST_urine", "hilic", "negative")
    assert LOCK.classify_panel(path.replace("_neg_", "_pos_")) == (
        "NIST_urine",
        "hilic",
        "positive",
    )
    assert LOCK.classify_panel(
        "ccms_peak/MetDNA3 Exploris 480 datasets/NIST_urine_hilic/urine_01.mzML"
    ) is None


def test_rotating_split_is_exactly_three_seed_roles() -> None:
    for base in range(10):
        roles = [BUILD.fold_membership(base, fold) for fold in range(10)]
        assert roles.count("seed") == 3
        assert roles.count("heldout") == 7

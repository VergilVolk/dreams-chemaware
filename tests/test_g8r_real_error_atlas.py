import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_g8r_real_error_atlas", ROOT / "tasks/build_g8r_real_error_atlas.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_strict_candidate_ranks_count_ties_against_candidate():
    ranks = MODULE.strict_candidate_ranks(np.asarray([0.8, 0.8, 0.2]))
    assert ranks.tolist() == [2, 2, 3]


def test_transition_name_four_cases():
    assert MODULE.transition_name(True, True) == "protected_correct"
    assert MODULE.transition_name(False, True) == "corrected"
    assert MODULE.transition_name(True, False) == "introduced"
    assert MODULE.transition_name(False, False) == "persistent_wrong"


def test_collect_ik14_nested_manifest():
    found = set()
    MODULE.collect_ik14({"queries": [{"ik14": "ABCDEFGHIJKLMNZZ"}]}, found)
    assert found == {"ABCDEFGHIJKLMN"}


def test_p3_loader_excludes_query_identities_not_reference_candidates(tmp_path):
    body = {
        "queries": [{
            "ik14": "QUERYIDENTITY01",
            "candidates": [{"ik14": "LIBRARYCANDID1"}],
        }]
    }
    (tmp_path / "p3_main_real_pristine_manifest.json").write_text(
        json.dumps(body), encoding="utf-8"
    )
    assert MODULE.load_p3_identities(tmp_path) == {"QUERYIDENTITY0"}


def test_rule_jaccard_empty_and_overlap():
    assert np.isnan(MODULE.rule_jaccard(np.zeros(3, bool), np.zeros(3, bool)))
    value = MODULE.rule_jaccard(
        np.asarray([1, 1, 0], bool), np.asarray([0, 1, 1], bool)
    )
    assert np.isclose(value, 1 / 3)

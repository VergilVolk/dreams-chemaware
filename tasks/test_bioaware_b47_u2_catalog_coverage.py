#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b47_u2_catalog_coverage import (  # noqa: E402
    edge_degree, normalize_ik14, route_decision,
)


def test_degree_is_undirected_and_deduplicated() -> None:
    edges = pd.DataFrame({
        "a": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"],
        "b": ["BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC"],
    })
    degree = edge_degree(edges, "a", "b")
    assert degree.to_dict() == {
        "AAAAAAAAAAAAAA": 2, "BBBBBBBBBBBBBB": 1, "CCCCCCCCCCCCCC": 1,
    }


def test_identity_normalization_rejects_malformed_values() -> None:
    value = normalize_ik14(pd.Series(["abcdefghijklmn-extra", "bad", None]))
    assert value.tolist() == ["ABCDEFGHIJKLMN", "", ""]


def test_route_never_promotes_emrn_to_exact_event() -> None:
    assert route_decision(127, 210, 230)["pass_to_kegg_currency_curation"] is True
    route = route_decision(127, 190, 230)
    assert route["code"] == "ONLY_EMRN_EXPANSION_CLOSES_DENOMINATOR"
    assert route["pass_to_kegg_currency_curation"] is False


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print("[test_bioaware_b47_u2_catalog_coverage] PASS", flush=True)

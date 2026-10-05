"""CPU contracts for source-local ranking, controls, conflicts and consensus."""
from __future__ import annotations

import math

from build_chemaware_multisource_native_triplets import pair_evidence


def body(score: float, formula: str, scope: str, a: float = math.nan, b: float = math.nan):
    return {
        "score": score, "formula": formula, "scope": scope,
        "control_scores": (a, b), "control_names": ("control_a_score", "control_b_score"),
        "controls": math.isfinite(a) and math.isfinite(b),
    }


def main() -> None:
    families = {
        "iceberg": {"scope": "all_candidates", "confidence_tier": "B_curated"},
        "tree": {"scope": "cross_formula", "confidence_tier": "A_tree"},
        "csi": {"scope": "within_formula", "confidence_tier": "A_csi"},
    }
    candidates = {
        0: {
            "iceberg": body(0.8, "A", "all_candidates", 0.4, 0.5),
            "tree": body(9.0, "A", "cross_formula"),
            "csi": body(5.0, "A", "within_formula"),
        },
        1: {
            "iceberg": body(0.2, "B", "all_candidates", 0.3, 0.45),
            "tree": body(6.0, "B", "cross_formula"),
        },
        2: {
            "iceberg": body(0.1, "A", "all_candidates", 0.0, 0.2),
            "tree": body(9.0, "A", "cross_formula"),
            "csi": body(3.0, "A", "within_formula"),
        },
    }
    cross_support, cross_oppose, cross_nonblocking = pair_evidence(candidates, families, 0, 1)
    assert [name for name, _ in cross_support] == ["iceberg", "tree"]
    assert not cross_oppose and not cross_nonblocking
    within_support, within_oppose, within_nonblocking = pair_evidence(candidates, families, 0, 2)
    assert [name for name, _ in within_support] == ["csi", "iceberg"]
    assert not within_oppose and not within_nonblocking
    candidates[1]["tree"]["score"] = 10.0
    support, oppose, nonblocking = pair_evidence(candidates, families, 0, 1)
    assert oppose == ["tree"] and support
    assert not nonblocking
    candidates[1]["tree"]["score"] = 6.0
    candidates[0]["iceberg"]["control_scores"] = (1.0, 0.5)
    support, _, _ = pair_evidence(candidates, families, 0, 1)
    assert [name for name, _ in support] == ["tree"]

    # Pair-local supervision must not be vetoed by an unrelated third
    # candidate.  The chemical source selects the concrete (true, false)
    # boundary; it is not required to solve global candidate classification.
    candidates[2]["iceberg"]["score"] = 2.0
    candidates[0]["iceberg"]["control_scores"] = (0.4, 0.5)
    support, oppose, nonblocking = pair_evidence(candidates, families, 0, 1)
    assert [name for name, _ in support] == ["iceberg", "tree"]
    assert not oppose and not nonblocking

    # Lower-confidence opposition is audited but cannot veto stronger support.
    families["iceberg"]["confidence_tier"] = "C_longtail"
    candidates[1]["iceberg"]["score"] = 3.0
    support, oppose, nonblocking = pair_evidence(candidates, families, 0, 1)
    assert [name for name, _ in support] == ["tree"]
    assert not oppose and nonblocking == ["iceberg"]

    # A C-tier family cannot become formal relation evidence when both rows do
    # not carry their matched-null observations.
    candidates[0]["iceberg"] = body(0.8, "A", "all_candidates")
    candidates[1]["iceberg"] = body(0.2, "B", "all_candidates")
    candidates[0]["tree"]["score"] = candidates[1]["tree"]["score"]
    support, _, _ = pair_evidence(candidates, families, 0, 1)
    assert support == []
    print("PASS: ChemAware multisource native-triplet contracts", flush=True)


if __name__ == "__main__":
    main()

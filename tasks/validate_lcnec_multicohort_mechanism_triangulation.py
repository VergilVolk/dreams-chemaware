"""Fail-closed validation for the LCNEC multi-cohort triangulation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(OUT / "candidate_triangulation.csv")
    axes = pd.read_csv(OUT / "mechanism_triangulation.csv")
    assert report["formal"] is True
    assert report["counts"]["priority_hypotheses"] == 4
    assert report["counts"]["passing_proteins"] == 13
    assert report["decisions"]["new_exact_metabolite_claims"] == 0
    assert len(candidates) == 4 and candidates["priority_name"].nunique() == 4
    assert not candidates["exact_identity_allowed"].astype(bool).any()
    assert candidates["local_pairs"].eq(34).all()
    assert candidates["local_direction_stable"].astype(bool).all()
    assert len(axes) == 4
    adpr = candidates.set_index("priority_name").loc["adenosine_diphosphoribose_family"]
    assert adpr["independent_support_class"] == "direct_pathway_context"
    quin = candidates.set_index("priority_name").loc["quinolinate"]
    assert quin["validation_rank"] == 1
    for filename in ["multicohort_triangulation.png", "multicohort_triangulation.pdf", "README.md"]:
        path = OUT / filename
        assert path.is_file() and path.stat().st_size > 100
    print("[validate_lcnec_multicohort_mechanism_triangulation] PASS candidates=4 axes=4")


if __name__ == "__main__":
    main()

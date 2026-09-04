"""Fail-closed validation for the MTBLS13729 annotation/biology benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/annotation_biology_benchmark_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(OUT / "annotation_rate_comparison.csv")
    mapped = pd.read_csv(OUT / "author_rplc_to_current_targets.csv")
    modules = pd.read_csv(OUT / "algorithm_to_biology_module_ledger.csv")
    claims = pd.read_csv(OUT / "frozen_biology_claim_ledger.csv")
    contribution = pd.read_csv(OUT / "candidate_method_contribution_matrix.csv")
    biology_modules = pd.read_csv(OUT / "biology_module_method_summary.csv")
    contribution_report = json.loads((OUT / "method_contribution_report.json").read_text(encoding="utf-8"))
    survival = pd.read_csv(OUT / "initial_annotation_priority_survival.csv")
    survival_report = json.loads(
        (OUT / "initial_annotation_priority_survival_report.json").read_text(encoding="utf-8")
    )

    assert report["formal"] is True
    native = report["source_paper_native"]
    assert native["detected_total"] == 9766
    assert native["ms2_total"] == 6054
    assert native["annotated_total"] == 345
    shared = report["shared_rplc_target_universe"]
    assert shared["targets"] == 16953
    systems = shared["systems"]
    assert systems["official_dreams"]["count"] == 3417
    assert systems["experimental_e6"]["count"] == 3426
    assert systems["frozen_p2b"]["count"] == 3588
    assert systems["threeway_consensus"]["count"] == 2162
    assert systems["threeway_union"]["count"] == 3599
    assert systems["threeway_consensus"]["count"] <= min(
        systems[key]["count"] for key in ("official_dreams", "experimental_e6", "frozen_p2b")
    )
    assert systems["threeway_union"]["count"] >= max(
        systems[key]["count"] for key in ("official_dreams", "experimental_e6", "frozen_p2b")
    )
    assert mapped.groupby(["panel", "feature_id"]).ngroups == 141
    assert set(comparison.panel) == {"neg_rp", "pos_rp", "combined_rplc"}
    assert len(modules) == 8 and modules.feature_id.nunique() == 8
    assert (claims.status == "PASS_DISCOVERY").sum() == 2
    assert (claims.status == "MISSING").sum() == 3
    assert contribution_report["formal"] is True
    assert len(contribution) == contribution.feature_id.nunique() == 18
    assert contribution_report["source_identity_rows"] == 13
    assert contribution_report["source_identity_positive_anchors"] == 12
    assert contribution_report["source_identity_downgraded_controls"] == 1
    assert contribution_report["official_dreams_assignments_present"] == 6
    assert contribution_report["official_dreams_identity_agreements"] == 5
    assert contribution_report["official_dreams_positive_usable_agreements"] == 4
    assert contribution_report["classical_library_orthogonal_candidates"] == 3
    assert contribution_report["bioaware_family_anchors"] == 2
    assert contribution_report["source_table_absent_family_candidates"] == 5
    assert contribution_report["new_exact_metabolite_claims"] == 0
    assert len(biology_modules) == contribution.module.nunique() == 8
    assert not contribution.new_exact_metabolite_claim_permitted.astype(bool).any()
    assert survival_report["formal"] is True
    assert len(survival) == survival.feature_id.nunique() == 6
    assert survival_report["retained_positive_biology_nodes"] == 3
    assert survival_report["filtered_or_downgraded"] == 3
    assert survival_report["new_exact_metabolite_claims"] == 0
    assert set(survival_report["retained_feature_ids"]) == {73, 398, 732}
    assert set(survival_report["filtered_feature_ids"]) == {41, 79, 486}
    print("[validate_mtbls13729_annotation_biology_benchmark] PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Freeze the annotation panel used by the MTBLS13729 biology analysis.

The panel is selected without phenotype or abundance information.  It separates
candidate assignment from annotation evidence and keeps P2b-only additions out
of the primary biological claim set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


E6 = "e6_fixed_v2_sw2"
LEVEL2A = {"Level 2a-supported", "Level 2a-single/ambiguous"}
FORBIDDEN_COLUMN_TOKENS = ("phenotype", "histology", "tumor", "normal", "qvalue", "log2fc", "pvalue")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(frame: pd.DataFrame, anchor_feature_id: int | None) -> pd.DataFrame:
    result = frame.copy()
    e6_tier = result[f"{E6}_annotation_evidence_tier"].fillna("unassigned")
    e6_present = result[f"{E6}_ik14"].fillna("").astype(str).str.len().eq(14)
    consensus = result["threeway_consensus"].fillna(False).astype(bool)
    p2b_only_strong = (
        result["official_vs_p2b"].eq("right_only")
        & (result["p2b_n_support_samples"].fillna(0) >= 2)
        & (result["p2b_agreement_fraction"].fillna(0) >= 0.6)
        & (result["p2b_maximum_dreams_similarity"].fillna(-1) >= 0.8)
    )
    result["analysis_tier"] = np.select(
        [
            consensus & e6_tier.eq("Level 2a-supported"),
            e6_present & e6_tier.eq("Level 2a-supported"),
            consensus & e6_tier.eq("Level 2a-single/ambiguous"),
            e6_present & e6_tier.eq("Level 2a-single/ambiguous"),
            p2b_only_strong,
        ],
        [
            "A_threeway_consensus_level2a_supported",
            "B_e6_level2a_supported",
            "C_threeway_consensus_level2a_ambiguous",
            "D_e6_level2a_ambiguous_exploratory",
            "E_p2b_only_orthogonal_validation",
        ],
        default="excluded",
    )
    result["selected_for_targeted_requantification"] = result["analysis_tier"].ne("excluded")
    result["primary_identity_claim_eligible"] = result["analysis_tier"].eq(
        "A_threeway_consensus_level2a_supported"
    )
    result["secondary_identity_hypothesis"] = result["analysis_tier"].isin(
        ["B_e6_level2a_supported", "C_threeway_consensus_level2a_ambiguous"]
    )
    result["exploratory_only"] = result["analysis_tier"].isin(
        ["D_e6_level2a_ambiguous_exploratory", "E_p2b_only_orthogonal_validation"]
    )
    result["predeclared_c20_4_anchor"] = False
    if anchor_feature_id is not None:
        result.loc[result["feature_id"].eq(anchor_feature_id), "predeclared_c20_4_anchor"] = True
        result.loc[result["feature_id"].eq(anchor_feature_id), "selected_for_targeted_requantification"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("data/mtbls13729/threeway_application_audit_v1"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/frozen_biology_panel_v1"))
    args = parser.parse_args()

    audit_report_path = args.audit_dir / "report.json"
    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_threeway_application_audit_complete":
        raise RuntimeError("three-way application audit is incomplete")
    anchor = report.get("predeclared_c20_4_anchor") or {}
    anchor_feature_id = int(anchor["resolved_feature_id"]) if anchor.get("resolved_feature_id") is not None else None

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite frozen biology panel: {out}")
    out.mkdir(parents=True, exist_ok=True)
    panel_report: dict[str, object] = {}
    for panel in ("neg_rp", "pos_rp"):
        audited_path = args.audit_dir / f"{panel}__audited_features.csv.gz"
        targets_path = args.consensus_dir / f"{panel}__requantification_targets.csv.gz"
        samples_path = args.consensus_dir / f"{panel}__samples.csv"
        for path in (audited_path, targets_path, samples_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        audited = pd.read_csv(audited_path)
        suspicious = [column for column in audited.columns if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)]
        if suspicious:
            raise RuntimeError(f"phenotype/outcome-like columns forbidden in frozen panel input: {suspicious}")
        current_anchor = anchor_feature_id if panel == "pos_rp" else None
        classified = classify(audited, current_anchor)
        selected = classified.loc[classified["selected_for_targeted_requantification"]].copy()
        if selected["feature_id"].duplicated().any():
            raise RuntimeError(f"{panel}: duplicate selected feature IDs")

        targets = pd.read_csv(targets_path)
        if targets["feature_id"].duplicated().any():
            raise RuntimeError(f"{panel}: duplicate consensus target feature IDs")
        selected = selected.merge(targets, on="feature_id", how="left", validate="one_to_one", suffixes=("", "_target"))
        if selected[["mz", "rt_sec"]].isna().any().any():
            missing = selected.loc[selected["mz"].isna() | selected["rt_sec"].isna(), "feature_id"].tolist()
            raise RuntimeError(f"{panel}: selected features missing target coordinates: {missing[:10]}")

        # The requantifier consumes only feature_id, mz and rt_sec, but keeping
        # the evidence columns makes every quantified row auditable.
        selected.to_csv(out / f"{panel}__frozen_annotations.csv.gz", index=False, compression="gzip")
        selected.to_csv(out / f"{panel}__requantification_targets.csv.gz", index=False, compression="gzip")
        shutil.copy2(samples_path, out / f"{panel}__samples.csv")

        tier_counts = selected["analysis_tier"].value_counts().sort_index().to_dict()
        expected_supported = int(report["panels"][panel]["systems"][E6]["level2a_supported"])
        observed_supported = int(
            selected[f"{E6}_annotation_evidence_tier"].eq("Level 2a-supported").sum()
        )
        if observed_supported != expected_supported:
            raise RuntimeError(
                f"{panel}: E6 supported count mismatch, expected {expected_supported}, observed {observed_supported}"
            )
        panel_report[panel] = {
            "selected_targets": int(len(selected)),
            "tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
            "primary_identity_claim_eligible": int(selected["primary_identity_claim_eligible"].sum()),
            "e6_level2a_supported": observed_supported,
            "e6_level2a_any": int(selected[f"{E6}_annotation_evidence_tier"].isin(LEVEL2A).sum()),
            "p2b_only_validation": int(selected["analysis_tier"].eq("E_p2b_only_orthogonal_validation").sum()),
            "anchor_present": bool(selected["predeclared_c20_4_anchor"].any()),
            "targets_sha256": sha256(out / f"{panel}__requantification_targets.csv.gz"),
        }

    if not panel_report["pos_rp"]["anchor_present"]:
        raise RuntimeError("predeclared C20:4 anchor is absent from the frozen positive-mode panel")
    frozen_report = {
        "status": "mtbls13729_frozen_biology_panel_complete",
        "formal": True,
        "selection_is_phenotype_blind": True,
        "panels": panel_report,
        "contracts": {
            "candidate_assignment_is_not_annotation": True,
            "primary_identity_claim": "three-way consensus and E6 Level 2a-supported only",
            "secondary_identity_hypothesis": "E6-supported or three-way ambiguous; manual fragment review required",
            "p2b_only": "orthogonal validation queue only; never a primary identity claim",
            "biological_statistics": "computed only after frozen-panel targeted EIC requantification",
            "flux_claims": "forbidden from static tissue abundance",
        },
        "provenance": {
            "threeway_audit_sha256": sha256(audit_report_path),
            "builder_sha256": sha256(Path(__file__)),
        },
    }
    (out / "report.json").write_text(json.dumps(frozen_report, indent=2), encoding="utf-8")
    print(json.dumps(frozen_report, indent=2), flush=True)


if __name__ == "__main__":
    main()

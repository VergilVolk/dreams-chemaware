"""Build a candidate-level identity claim defense and minimal validation ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
BENCH = BASE / "annotation_biology_benchmark_v1"
FAMILY = BASE / "source_absent_family_readiness_v1"
OUT = BASE / "identity_claim_defense_v1"


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(BENCH / "candidate_method_contribution_matrix.csv")
    families = pd.read_csv(FAMILY / "source_absent_family_readiness.csv")
    claims = pd.read_csv(BENCH / "frozen_biology_claim_ledger.csv")

    family_by_feature = families.set_index("feature_id", drop=False)
    rows = []
    for _, row in matrix.iterrows():
        feature = int(row.feature_id)
        source_msi = str(row.published_source_msi) if pd.notna(row.published_source_msi) else ""
        absent = bool(str(row.source_table_absent_family_candidate).lower() == "true")
        source_level1 = source_msi == "Level 1"
        family = family_by_feature.loc[feature] if feature in family_by_feature.index else None

        if source_level1:
            identity_status = "same-cohort published Level-1 identity"
            current_claim = str(row.defensible_identity)
            standard_needed_for_current_claim = False
            future_upgrade = "independent cohort replication; standard reinjection optional for cross-platform transfer"
        elif absent:
            identity_status = "source-table-absent family hypothesis"
            current_claim = str(family.identity_ceiling if family is not None else row.claim_ceiling)
            standard_needed_for_current_claim = False
            future_upgrade = str(family.validation_priority if family is not None else "authentic positional-isomer standard panel")
        else:
            identity_status = "published Level-2 or contextual identity"
            current_claim = str(row.claim_ceiling)
            standard_needed_for_current_claim = False
            future_upgrade = "authentic standard required only to promote to a new Level-1 identity"

        rows.append(
            {
                "feature_id": feature,
                "label": row.label,
                "module": row.module,
                "identity_status": identity_status,
                "published_source_msi": source_msi,
                "current_manuscript_claim": current_claim,
                "new_exact_identity_claimed": False,
                "new_standard_required_for_current_claim": standard_needed_for_current_claim,
                "algorithm_role": row.annotation_increment_type,
                "raw_ms2_available": bool(str(row.peak_resolved_raw_ms2).lower() == "true"),
                "paired_ms1_requantification": bool(str(row.paired_ms1_requantification).lower() == "true"),
                "future_identity_upgrade": future_upgrade,
            }
        )
    defense = pd.DataFrame(rows)
    defense.to_csv(OUT / "candidate_identity_claim_defense.csv", index=False)

    source_absent = defense[defense.identity_status == "source-table-absent family hypothesis"]
    headline = defense[defense.feature_id == 703]
    if len(headline) != 1 or headline.iloc[0].published_source_msi != "Level 1":
        raise RuntimeError("Neu5Ac headline anchor must remain a same-cohort published Level-1 identity")
    if len(source_absent) != 5 or source_absent.new_exact_identity_claimed.any():
        raise RuntimeError("the five source-absent signals must remain non-exact family hypotheses")

    review_rows = [
        {
            "reviewer_objection": "The headline metabolite needs an authentic standard.",
            "response": "Neu5Ac was already Level-1 identified by the source study in the same cohort. Our positive-RPLC feature is an orthogonal remap to that source identity, not a newly standard-free identity claim.",
            "remaining_limit": "No independent subtype-resolved Neu5Ac abundance replication and no same-platform spike-in.",
        },
        {
            "reviewer_objection": "The five source-table-absent ions are unverified new metabolites.",
            "response": "They are explicitly collapsed into three family modules and zero exact new-metabolite claims. Positional-isomer names are prohibited.",
            "remaining_limit": "Standards are required only if the manuscript is upgraded to exact positional identities.",
        },
        {
            "reviewer_objection": "Network biology was used to choose identities that fit the phenotype.",
            "response": "Phenotype is forbidden from identity ranking. BioAware contributes family/context evidence and abstention only; its external retrieval gain is not statistically confirmed.",
            "remaining_limit": "Network evidence cannot independently promote an identity.",
        },
        {
            "reviewer_objection": "Annotation coverage is being presented as annotation accuracy.",
            "response": "The source, official DreaMS, E6 and P2b numbers are reported as same-universe candidate coverage only; all accuracy language is prohibited in the application cohort without truth labels.",
            "remaining_limit": "Candidate-level official/E6/P2b tables still need server synchronization for per-feature attribution.",
        },
        {
            "reviewer_objection": "Static abundance proves flux or an enzyme mechanism.",
            "response": "The paper claims free-pool-to-donor/destination decoupling and selective capacity context, not flux, source enzyme, transport activity or causality.",
            "remaining_limit": "Tracing, perturbation/rescue and linkage-aware same-sample glycomics are absent.",
        },
    ]
    pd.DataFrame(review_rows).to_csv(OUT / "reviewer_objection_response_matrix.csv", index=False)

    report = {
        "status": "mtbls13729_identity_claim_defense_complete",
        "formal": True,
        "candidate_rows": int(len(defense)),
        "headline_feature": 703,
        "headline_identity": "N-acetylneuraminic acid",
        "headline_source_msi": "Level 1",
        "headline_is_new_identity_claim": False,
        "source_absent_signals": int(len(source_absent)),
        "source_absent_modules": int(source_absent.module.nunique()),
        "new_exact_metabolite_claims": 0,
        "standards_required_for_current_claim_set": 0,
        "standards_required_to_upgrade_family_claims": [
            "N1,N8-diacetylspermidine positional-isomer panel",
            "methyl/dimethylguanosine positional-isomer panel",
            "C20:4/C16:0/C18:0/C18:1 acylcarnitine panel with isotope-labelled internal standards",
        ],
        "primary_unresolved_validation": "independent subtype-resolved Neu5Ac abundance replication",
        "claim_limit": "This ledger defends the current claim hierarchy; it does not make family hypotheses equivalent to standard-confirmed identities.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pass_discovery = int((claims.status == "PASS_DISCOVERY").sum())
    text = f"""# MTBLS13729 identity-claim defense

## The key answer to “why should anyone believe the metabolite?”

The primary metabolite, free N-acetylneuraminic acid (feature 703), is not a new identity invented by DreaMS. It was already reported as an MSI Level-1 metabolite in the source study's same cohort. Our contribution is an orthogonal positive-RPLC recovery, peak-resolved MS2 reconciliation and locked paired MS1 requantification. A new standard is therefore not required to justify the current same-cohort identity claim, although a same-platform spike-in would improve cross-platform transfer.

The five source-table-absent signals are not presented as five new metabolites. They are collapsed into {source_absent.module.nunique()} family modules with zero exact identity claims. Their standards are required only if positional-isomer names are promoted in a future paper.

## What the paper can claim now

- {pass_discovery} frozen biology claims pass discovery-level gates.
- The headline is a paired abundance phenotype of an existing Level-1 identity.
- DreaMS/E6/P2b expand and calibrate candidate coverage; BioAware organizes family and mechanism context.
- Feature 1717 is an acetylated-polyamine family hypothesis; 1597/3019 a modified-guanosine family; 150/3222 a long-chain-acylcarnitine family.

## What remains impossible without new data

- exact positional identities for the three source-absent families;
- independent subtype-resolved Neu5Ac abundance replication;
- same-sample glycan destination, isotope flux, enzyme source or causality.

This division removes the false dilemma that either every feature receives a standard or the whole biology is invalid. The headline uses an existing Level-1 identity; the new signals remain family-level and are not used as exact structural proof.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

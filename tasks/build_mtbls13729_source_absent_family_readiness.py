"""Build an evidence-gated readiness audit for source-table-absent MTBLS13729 families.

This is deliberately not a numerical discovery score.  Each candidate is
evaluated against explicit, independently reported evidence dimensions, and
chemically related ions are collapsed into biological modules before a
manuscript role is assigned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
BENCH = BASE / "annotation_biology_benchmark_v1"
OUT = BASE / "source_absent_family_readiness_v1"

SCORECARD = BASE / "candidate_claim_scorecard_v3/candidate_claim_scorecard_v3.csv"
LEDGER = BASE / "candidate_evidence_ledger_v1/candidate_evidence_ledger.csv"
FRAGMENTATION = BASE / "fragmentation_standard_consistency_v1/fragmentation_standard_consistency.csv"
RECURRENCE = BASE / "candidate_ce_recurrence_v4/diagnostic_recurrence_summary.csv"
METHOD = BENCH / "candidate_method_contribution_matrix.csv"
ABUNDANCE_AUDIT = BASE / "candidate_abundance_protocol_audit_v1/report.json"

FEATURES = [150, 1597, 1717, 3019, 3222]

MANUAL = {
    150: {
        "chemistry_role": "putative palmitoylcarnitine library anchor",
        "orthogonal_support": "official DreaMS library consensus only; no independent standard or diagnostic-transition audit",
        "external_support": "none specific to exact identity",
        "identity_ceiling": "putative library identity; no same-method authentic standard",
        "manuscript_role": "supporting long-chain-acylcarnitine module anchor",
        "validation_priority": "P2 palmitoylcarnitine standard plus isotope-labelled internal standard",
    },
    1597: {
        "chemistry_role": "methylguanosine positional-isomer family member",
        "orthogonal_support": "ribose-loss/aglycone transition plus BioAware ion-family consolidation",
        "external_support": "independent modified-guanosine context is heterogeneous and therefore not identity replication",
        "identity_ceiling": "m1G/m2G/m7G and related positional isomers unresolved",
        "manuscript_role": "modified-guanosine module contributor; never a standalone exact-metabolite claim",
        "validation_priority": "P1 methylguanosine positional-isomer standard panel",
    },
    1717: {
        "chemistry_role": "N1,N8-diacetylspermidine-like acetylated-polyamine family",
        "orthogonal_support": "diagnostic 230.2->100.0 transition and same-source HILIC cross-chromatography concordance",
        "external_support": "independent epithelial polyamine-acetylation/catabolism program; exact metabolite not replicated",
        "identity_ceiling": "strong family candidate; exact positional identity unresolved",
        "manuscript_role": "highest-priority secondary novel family hypothesis",
        "validation_priority": "P0 N1,N8-diacetylspermidine and positional-isomer standards",
    },
    3019: {
        "chemistry_role": "dimethylguanosine positional-isomer family member",
        "orthogonal_support": "ribose-loss/aglycone transition plus BioAware ion-family consolidation",
        "external_support": "independent modified-guanosine context is heterogeneous and therefore not identity replication",
        "identity_ceiling": "1,7-/N2,N2- and other dimethylguanosine positional isomers unresolved",
        "manuscript_role": "modified-guanosine module anchor; never a standalone exact-metabolite claim",
        "validation_priority": "P1 dimethylguanosine positional-isomer standard panel",
    },
    3222: {
        "chemistry_role": "long-chain/C20:4-acylcarnitine-like class member",
        "orthogonal_support": "recurrent carnitine-class fragments in raw MS2",
        "external_support": "independent long-chain-lipid context; exact acylcarnitine not replicated",
        "identity_ceiling": "long-chain acylcarnitine class and nominal composition only",
        "manuscript_role": "supporting long-chain-acylcarnitine class anchor",
        "validation_priority": "P1 C20:4/C16:0/C18:0/C18:1 acylcarnitine standard panel",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_feature(frame: pd.DataFrame, feature_id: int, source: str) -> pd.Series | None:
    rows = frame.loc[frame.feature_id.eq(feature_id)]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise RuntimeError(f"{source}: feature {feature_id} has {len(rows)} rows")
    return rows.iloc[0]


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    scorecard = pd.read_csv(SCORECARD)
    ledger = pd.read_csv(LEDGER)
    fragmentation = pd.read_csv(FRAGMENTATION)
    recurrence = pd.read_csv(RECURRENCE)
    method = pd.read_csv(METHOD)
    abundance_audit = json.loads(ABUNDANCE_AUDIT.read_text(encoding="utf-8"))
    if abundance_audit.get("formal") is not True:
        raise RuntimeError("candidate abundance protocol audit is not formal")
    for frame in (scorecard, ledger, fragmentation, recurrence, method):
        frame["feature_id"] = pd.to_numeric(frame.feature_id, errors="raise").astype(int)

    rows: list[dict] = []
    for feature_id in FEATURES:
        score = unique_feature(scorecard, feature_id, "scorecard")
        contribution = unique_feature(method, feature_id, "method matrix")
        if score is None or contribution is None:
            raise RuntimeError(f"feature {feature_id} absent from required scorecard/method matrix")
        evidence = unique_feature(ledger, feature_id, "candidate ledger")
        frag = unique_feature(fragmentation, feature_id, "fragmentation audit")
        diagnostic_rows = recurrence.loc[recurrence.feature_id.eq(feature_id)]
        best_diagnostic = None
        if not diagnostic_rows.empty:
            best_diagnostic = diagnostic_rows.sort_values(
                ["support_fraction", "present_spectra"], ascending=False
            ).iloc[0]

        primary_pairs = int(evidence.rmu_n) if evidence is not None else int(float(score.pairs))
        primary_positive = (
            int(round(float(evidence.rmu_positive_fraction) * primary_pairs))
            if evidence is not None
            else int(float(score.positive_pairs))
        )
        primary_log2fc = float(evidence.rmu_mean_log2fc) if evidence is not None else float(score.mean_log2fc)
        primary_sign_p = finite(evidence.rmu_exact_signflip_p) if evidence is not None else None
        ms2_spectra = int(evidence.peak_resolved_ms2_spectra) if evidence is not None else int(float(score.peak_resolved_ms2_spectra))
        ms2_samples = int(evidence.peak_resolved_samples) if evidence is not None else None
        diagnostic_fraction = finite(best_diagnostic.support_fraction) if best_diagnostic is not None else None
        diagnostic_spectra = int(best_diagnostic.present_spectra) if best_diagnostic is not None else None
        diagnostic_label = str(best_diagnostic.diagnostic_label) if best_diagnostic is not None else "not audited"

        abundance_gate = primary_pairs >= 8 and primary_positive / primary_pairs >= 0.80
        ms2_recurrence_gate = ms2_spectra >= 30 and (ms2_samples is None or ms2_samples >= 20)
        diagnostic_gate = diagnostic_fraction is not None and diagnostic_fraction >= 0.70
        full_space_fdr_gate = bool(score.full_untargeted_exact_fdr10)
        source_absent = bool(contribution.source_table_absent_family_candidate)
        exact_claim_permitted = bool(contribution.new_exact_metabolite_claim_permitted)
        if exact_claim_permitted:
            raise RuntimeError(f"feature {feature_id}: source-absent exact identity was unexpectedly permitted")

        alternate_pairs = int(float(score.pairs))
        alternate_log2fc = float(score.mean_log2fc)
        abundance_protocol_discordant = (
            alternate_pairs != primary_pairs or abs(alternate_log2fc - primary_log2fc) > 0.5
        )

        abundance_protocol_resolved = feature_id in abundance_audit["material_protocol_difference_features"]
        if feature_id == 1717:
            placement = "SECONDARY_FAMILY_HYPOTHESIS"
        elif feature_id in (1597, 3019):
            placement = "MODULE_CONTRIBUTOR_ONLY"
        elif feature_id in (150, 3222):
            placement = "SUPPORTING_CLASS_ANCHOR"
        else:  # pragma: no cover
            raise AssertionError(feature_id)

        rows.append(
            {
                "feature_id": feature_id,
                "label": score.label,
                "module": score.module,
                "chemistry_role": MANUAL[feature_id]["chemistry_role"],
                "source_table_absent": source_absent,
                "primary_abundance_source": "candidate_evidence_ledger_v1" if evidence is not None else "candidate_claim_scorecard_v3",
                "primary_rmu_pairs": primary_pairs,
                "primary_positive_pairs": primary_positive,
                "primary_positive_fraction": primary_positive / primary_pairs,
                "primary_mean_log2fc": primary_log2fc,
                "primary_exact_sign_p": primary_sign_p,
                "alternate_panel_pairs": alternate_pairs,
                "alternate_panel_mean_log2fc": alternate_log2fc,
                "abundance_protocol_discordant": abundance_protocol_discordant,
                "abundance_protocol_discrepancy_resolved": abundance_protocol_resolved,
                "peak_resolved_ms2_spectra": ms2_spectra,
                "peak_resolved_ms2_samples": ms2_samples,
                "diagnostic_label": diagnostic_label,
                "diagnostic_support_spectra": diagnostic_spectra,
                "diagnostic_support_fraction": diagnostic_fraction,
                "collision_energy_levels": int(best_diagnostic.collision_energies) if best_diagnostic is not None else None,
                "official_dreams_assignment": bool(contribution.official_dreams_assignment_present),
                "bioaware_family_consolidation": bool(contribution.bioaware_ion_family_consolidation),
                "orthogonal_support": MANUAL[feature_id]["orthogonal_support"],
                "external_support": MANUAL[feature_id]["external_support"],
                "abundance_gate": abundance_gate,
                "ms2_recurrence_gate": ms2_recurrence_gate,
                "diagnostic_transition_gate": diagnostic_gate,
                "candidate_panel_primary_fdr10": bool(score.candidate_panel_primary_fdr10),
                "full_untargeted_exact_fdr10": full_space_fdr_gate,
                "exact_metabolite_claim_permitted": exact_claim_permitted,
                "identity_ceiling": MANUAL[feature_id]["identity_ceiling"],
                "manuscript_placement": placement,
                "manuscript_role": MANUAL[feature_id]["manuscript_role"],
                "validation_priority": MANUAL[feature_id]["validation_priority"],
            }
        )

    readiness = pd.DataFrame(rows)
    readiness.to_csv(OUT / "source_absent_family_readiness.csv", index=False)

    by_id = readiness.set_index("feature_id")
    modules = pd.DataFrame(
        [
            {
                "module": "acetylated_polyamine_mta_turnover",
                "features": "1717",
                "paired_direction": "9/9 positive in formal candidate ledger",
                "identity_evidence": "73/73 diagnostic products; same-source HILIC concordance",
                "independent_context": "epithelial polyamine-acetylation/catabolism program",
                "module_decision": "highest-priority secondary family hypothesis",
                "hard_blocker": "no same-method authentic standard; complete-detection effect is accepted after protocol reconciliation",
            },
            {
                "module": "purine_modified_nucleoside_pool",
                "features": "1597;3019",
                "paired_direction": "18/18 combined positive pair observations",
                "identity_evidence": "aglycone/ribose-loss recurrence in both ions; BioAware family consolidation",
                "independent_context": "external modified-guanosine effect is molecular-context dependent",
                "module_decision": "main-text secondary module, not two independent exact metabolites",
                "hard_blocker": "positional isomers unresolved and no independent mucinous metabolomics replication",
            },
            {
                "module": "long_chain_acylcarnitine_accumulation",
                "features": "150;3222",
                "paired_direction": "17/20 combined positive pair observations",
                "identity_evidence": "DreaMS palmitoylcarnitine consensus plus recurrent acylcarnitine-class fragments",
                "independent_context": "long-chain lipid context only",
                "module_decision": "supporting metabolic-state module; not lead mechanism",
                "hard_blocker": "C20:4 chain/double-bond identity unresolved and no exact external replication",
            },
        ]
    )
    modules.to_csv(OUT / "module_readiness.csv", index=False)

    report = {
        "status": "mtbls13729_source_absent_family_readiness_complete",
        "formal": False,
        "candidates": int(len(readiness)),
        "modules": int(len(modules)),
        "exact_metabolite_claims": int(readiness.exact_metabolite_claim_permitted.sum()),
        "abundance_gate_pass": int(readiness.abundance_gate.sum()),
        "ms2_recurrence_gate_pass": int(readiness.ms2_recurrence_gate.sum()),
        "diagnostic_transition_gate_pass": int(readiness.diagnostic_transition_gate.sum()),
        "full_untargeted_exact_fdr10_pass": int(readiness.full_untargeted_exact_fdr10.sum()),
        "abundance_protocol_discordant_features": readiness.loc[
            readiness.abundance_protocol_discordant, "feature_id"
        ].astype(int).tolist(),
        "unresolved_abundance_protocol_discordance": readiness.loc[
            readiness.abundance_protocol_discordant & ~readiness.abundance_protocol_discrepancy_resolved,
            "feature_id",
        ].astype(int).tolist(),
        "primary_secondary_hypothesis": {
            "feature_id": 1717,
            "identity": MANUAL[1717]["chemistry_role"],
            "claim": "acetylated-polyamine family abundance hypothesis",
            "not_claimed": "exact N1,N8-diacetylspermidine identity, flux, enzyme activity or mucinous causality",
        },
        "module_decisions": modules[["module", "module_decision", "hard_blocker"]].to_dict(orient="records"),
        "decision_contract": {
            "abundance_gate": "at least 8 paired Rmu observations and at least 80% positive",
            "ms2_gate": "at least 30 peak-resolved spectra and, when available, at least 20 samples",
            "diagnostic_gate": "audited diagnostic transition present in at least 70% of peak-resolved spectra",
            "identity_gate": "no source-table-absent row may become an exact metabolite without same-method standard evidence",
            "module_gate": "chemically related ions are collapsed before manuscript placement",
        },
        "provenance": {
            "scorecard_sha256": sha256(SCORECARD),
            "candidate_ledger_sha256": sha256(LEDGER),
            "fragmentation_sha256": sha256(FRAGMENTATION),
            "recurrence_sha256": sha256(RECURRENCE),
            "method_matrix_sha256": sha256(METHOD),
            "abundance_audit_sha256": sha256(ABUNDANCE_AUDIT),
            "readiness_sha256": sha256(OUT / "source_absent_family_readiness.csv"),
            "module_readiness_sha256": sha256(OUT / "module_readiness.csv"),
        },
        "claim_limit": (
            "This is an evidence-readiness synthesis over selected candidate families, not an untargeted discovery-rate "
            "estimate. No source-table-absent candidate is an exact metabolite claim; no abundance result establishes "
            "flux, enzyme activity, cellular source or causality."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    narrative = f"""# MTBLS13729 source-table-absent family readiness v1

## Decision

The five rows are not five new metabolites. They collapse into three biological modules and **zero exact new-metabolite claims**.

1. **Acetylated polyamine (feature 1717):** the strongest secondary novel-family hypothesis. The formal candidate ledger contains 9/9 positive Rmu pairs (+3.009 log2), 73 peak-resolved spectra across 45 samples, a 73/73 diagnostic product, same-source HILIC concordance, and independent pathway context. The previous +4.817 estimate was traced to treating an explicitly undetected P28 normal peak as zero; it is rejected. The complete-detection estimate is now frozen, while exact positional identity remains unresolved.
2. **Modified guanosines (features 1597 and 3019):** one module, not two independent discoveries. Both are 9/9 positive and have recurrent aglycone/ribose-loss evidence. BioAware contributes family consolidation only. External evidence is context dependent, so this supports a secondary modified-nucleoside-pool hypothesis, not exact positional identities or a pan-CRC mechanism.
3. **Long-chain acylcarnitines (features 150 and 3222):** a supporting metabolic-state module. Feature 150 supplies a strong DreaMS palmitoylcarnitine library anchor; feature 3222 supplies recurrent class fragments. The C20:4-like chain identity remains unresolved and the module is not an independent flux result.

## Prespecified gates

- Abundance: >=8 Rmu pairs and >=80% positive.
- Raw MS2 recurrence: >=30 peak-resolved spectra and, when available, >=20 samples.
- Diagnostic transition: support in >=70% of audited spectra.
- Exact identity: impossible without same-method standard evidence.
- Biology: related ions are collapsed into modules before interpretation.

## Immediate validation order

1. P0: N1,N8-diacetylspermidine positional-isomer standard panel and effect-size protocol reconciliation.
2. P1: methyl- and dimethylguanosine positional-isomer panel, interpreted jointly.
3. P1/P2: C20:4/C16:0/C18:0/C18:1 acylcarnitines with isotope-labelled internal standards.

## Claim boundary

{report['claim_limit']}
"""
    (OUT / "REPORT.md").write_text(narrative, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

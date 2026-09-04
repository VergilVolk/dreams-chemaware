"""Build a protocol-safe manuscript evidence package for MTBLS13729."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
BENCH = BASE / "annotation_biology_benchmark_v1"
OUT = BASE / "manuscript_evidence_package_v3"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark = load_json(BENCH / "report.json")
    method = load_json(BENCH / "method_contribution_report.json")
    survival = load_json(BENCH / "initial_annotation_priority_survival_report.json")
    family = load_json(BASE / "source_absent_family_readiness_v1/report.json")
    abundance_protocol = load_json(BASE / "candidate_abundance_protocol_audit_v1/report.json")
    independence = load_json(BASE / "secondary_module_independence_v1/report.json")
    external_context = load_json(BASE / "external_biology_context_v2/report.json")
    biology = load_json(BASE / "biology_package_a_release_v1/report.json")
    mechanism = load_json(BASE / "mechanism_paper_completion_audit_v10_final/report.json")
    claims = pd.read_csv(BASE / "biology_package_a_release_v1/biology_claim_ledger_v1.csv")

    if not all(item.get("formal") is True for item in (benchmark, method, survival, biology)):
        raise RuntimeError("all primary package inputs must be frozen formal artifacts")
    if mechanism.get("biology_package_A_ready") is not True:
        raise RuntimeError("biology package A is not ready")

    shared = benchmark["shared_rplc_target_universe"]
    systems = shared["systems"]
    result_rows = [
        {
            "result_id": "R1",
            "result": "Denominator-safe annotation expansion",
            "primary_number": "345/9766 (3.53%) source-native; 3417/16953 (20.16%) official DreaMS shared-target coverage",
            "algorithmic_contribution": "official DreaMS expands auditable candidate coverage on a frozen RPLC target universe",
            "evidence_grade": "candidate coverage; not accuracy",
            "forbidden_claim": "do not divide 20.16% by 3.53% and call it a six-fold accuracy gain",
            "source_artifact": "annotation_biology_benchmark_v1/report.json",
        },
        {
            "result_id": "R2",
            "result": "E6 stabilizes evidence while P2b expands leads",
            "primary_number": "Level2a 254->276 for E6; candidate count 3417->3588 for P2b",
            "algorithmic_contribution": "E6 changes shared embedding; P2b is a separate post-embedding candidate expert",
            "evidence_grade": "same-protocol aggregate application evidence",
            "forbidden_claim": "do not call P2b coverage high-confidence identity or attribute candidate-level names without synced three-way tables",
            "source_artifact": "annotation_biology_benchmark_v1/report.json",
        },
        {
            "result_id": "R3",
            "result": "Evidence calibration rejects attractive false or unstable stories",
            "primary_number": "3/6 initial priorities retained as known context; 3/6 filtered; 0 new exact metabolites",
            "algorithmic_contribution": "source overlap, targeted EIC and conflict checks constrain DreaMS votes",
            "evidence_grade": "formal selected-screen survival audit",
            "forbidden_claim": "do not call selected-screen survival an annotation-accuracy estimate",
            "source_artifact": "annotation_biology_benchmark_v1/initial_annotation_priority_survival_report.json",
        },
        {
            "result_id": "R4",
            "result": "Frozen candidate panel separates known anchors from family hypotheses",
            "primary_number": "9 remaps + 3 orthogonal Level-1 recoveries + 5 source-table-absent families + 1 downgrade",
            "algorithmic_contribution": "DreaMS/raw-MS2/classical libraries/BioAware contribute different evidence layers",
            "evidence_grade": "candidate-level provenance, not structural truth",
            "forbidden_claim": "source-table absence or BioAware family consolidation does not establish a new exact metabolite",
            "source_artifact": "annotation_biology_benchmark_v1/method_contribution_report.json",
        },
        {
            "result_id": "R5",
            "result": "Free Neu5Ac pool expands relative to measured donor/precursor pools",
            "primary_number": "10/10 Rmu pairs; free +2.249 log2; free-minus-CMP +1.693 and free-minus-UDP +1.922, Holm p=0.0273",
            "algorithmic_contribution": "same-cohort Level-1 identity was orthogonally recovered and linked to locked paired MS1 quantification",
            "evidence_grade": "PASS_DISCOVERY",
            "forbidden_claim": "no flux, source enzyme, glycan destination or independent mucinous abundance replication",
            "source_artifact": "biology_package_a_release_v1/biology_claim_ledger_v1.csv",
        },
        {
            "result_id": "R6",
            "result": "External evidence supports selective secretory and Golgi transport capacity, not a single release mechanism",
            "primary_number": "AGR2 and SLC35A1 q=0.00678 in frozen raw-UMI context; host NEU1/NEU3 release unsupported",
            "algorithmic_contribution": "BioAware-style biological context is used for mechanism discrimination, not identity promotion",
            "evidence_grade": "PASS_CONTEXT plus negative results",
            "forbidden_claim": "RNA capacity is not protein activity, transport flux, biochemical source or causality",
            "source_artifact": "mechanism_paper_completion_audit_v10_final/report.json",
        },
        {
            "result_id": "R7",
            "result": "Five source-table-absent signals collapse to three family-level modules",
            "primary_number": "5 candidates; 3 modules; 0 exact new-metabolite claims",
            "algorithmic_contribution": "DreaMS, raw peak-resolved MS2 and BioAware contribute separate library, fragmentation and family-consolidation evidence",
            "evidence_grade": "selected family hypotheses with explicit identity ceilings",
            "forbidden_claim": "do not present five ions as five new metabolites or BioAware consolidation as identity proof",
            "source_artifact": "source_absent_family_readiness_v1/report.json",
        },
        {
            "result_id": "R8",
            "result": "Secondary modules are patient-level nonredundant but remain post-selection",
            "primary_number": "Neu5Ac rho: polyamine +0.42, modified guanosine -0.10, acylcarnitine +0.12; max module-technical |rho| 0.317",
            "algorithmic_contribution": "complete-detection abundance reconciliation prevents missing-as-zero inflation before module comparison",
            "evidence_grade": "formal post-selection sensitivity analysis",
            "forbidden_claim": "lack of correlation is not statistical independence, external replication or mechanism",
            "source_artifact": "secondary_module_independence_v1/report.json",
        },
        {
            "result_id": "R9",
            "result": "External literature strengthens context while narrowing novelty",
            "primary_number": "9 primary-literature records across Neu5Ac, modified guanosine and acetylated polyamine axes",
            "algorithmic_contribution": "algorithm-enabled local recovery is separated from previously reported disease associations",
            "evidence_grade": "external biological context; no local identity promotion",
            "forbidden_claim": "do not call known CRC-associated families newly discovered metabolites or treat literature as local structural validation",
            "source_artifact": "external_biology_context_v2/report.json",
        },
    ]
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "manuscript_result_evidence_table.csv", index=False)

    figures = pd.DataFrame(
        [
            ["Figure 1", "annotation protocol and evidence ladder", "annotation benchmark + candidate survival", "candidate coverage, evidence tiers, filters"],
            ["Figure 2", "paired abundance programs", "18-candidate ledger + paired effects", "Neu5Ac primary; other modules parallel/contextual"],
            ["Figure 3", "identity recovery and counterexamples", "proline/glutamate/Neu5Ac plus kynurenine/phenyllactate filters", "positive and negative evidence side by side"],
            ["Figure 4", "free-pool to donor/destination decoupling", "paired Neu5Ac/CMP/UDP + transcript/glycan context", "dashed evidence links only"],
            ["Extended Data", "algorithm attribution", "official/E6/P2b aggregate + BioAware family roles", "no candidate-level E6/P2b attribution until tables are synced"],
            ["Extended Data", "source-absent family evidence gates", "five candidates collapsed to three modules", "zero exact new-metabolite claims"],
            ["Extended Data", "patient-level module nonredundancy", "module correlations + phenotype-blind drift controls", "post-selection; not proof of independence"],
        ],
        columns=["figure", "message", "data", "claim_boundary"],
    )
    figures.to_csv(OUT / "figure_manifest.csv", index=False)

    native = benchmark["source_paper_native"]
    text = f"""# MTBLS13729 manuscript evidence package v3

## Position

**{biology['position']}**

The current publishable phenomenon is **{biology['primary_phenomenon']}**. The project does not claim a newly standard-confirmed metabolite, flux, enzyme causality or therapeutic mechanism.

## Result 1 — Candidate coverage expanded, but accuracy was not measured in the application cohort

The source paper annotated {native['annotated_total']:,} of {native['detected_total']:,} detected features ({100*native['annotated_total']/native['detected_total']:.2f}%), or {100*native['annotated_total']/native['ms2_total']:.2f}% when restricted to MS2-bearing features. On the frozen {shared['targets']:,}-target RPLC universe, official DreaMS assigned {systems['official_dreams']['count']:,} candidates ({100*systems['official_dreams']['count']/shared['targets']:.2f}%), E6 {systems['experimental_e6']['count']:,}, and P2b {systems['frozen_p2b']['count']:,}. These are candidate-coverage numbers, not structure-accuracy estimates.

E6 increased Level2a-supported features from 254 to 276 while adding only nine assignments. P2b increased assignments by 171 but reduced Level2a-supported features from 254 to 230. Therefore E6 is presented as an experimental evidence stabilizer and P2b as a lead-expansion module.

## Result 2 — Evidence calibration prevented annotation inflation

Of six early DreaMS annotation priorities, {survival['retained_positive_biology_nodes']} survived as known-identity context nodes and {survival['filtered_or_downgraded']} were filtered or downgraded. L-kynurenine was already present in the source identity table and lost robust significance after targeted EIC; malic acid was a source Level-1 remap with non-robust abundance; the 3-phenyllactic-acid vote conflicted with a source-table coordinate assignment. No new exact metabolite claim survived this screen.

## Result 3 — The frozen biology panel contains evidence classes, not 18 new metabolites

The 18-candidate ledger contains 9 source remaps, 3 same-cohort orthogonal Level-1 recoveries, 5 source-table-absent family hypotheses and 1 deliberate downgrade. Twelve source identities are usable positive anchors. Four candidates have usable DreaMS identity agreement, three have orthogonal classical-library support, and two modified-guanosine candidates are consolidated by BioAware at the ion-family level. Exact positional identity remains unresolved for every source-table-absent family.

## Result 4 — The primary biology is free-Neu5Ac pool-to-donor/destination decoupling

Free Neu5Ac increased in all 10 Rmu-versus-matched-normal pairs (mean +2.249 log2). Its increase exceeded measured CMP-Neu5Ac and UDP-GlcNAc changes by +1.693 and +1.922 log2, respectively (Holm p=0.0273 for both contrasts). Independent patient-level raw-UMI context supports selective epithelial secretory-folding and Golgi donor-transport capacity through AGR2 and SLC35A1, while host NEU1/NEU3 release and uniform pathway activation are not supported.

The defensible conclusion is a discovery-level abundance phenotype with donor/carrier/linkage decoupling. Static abundance does not establish flux; the source identity was already Level 1 in the same cohort; and no independent subtype-resolved Neu5Ac metabolomics replication is available.

## Result 5 — Source-table-absent signals form three secondary modules, not five new metabolites

The five selected source-table-absent signals collapse to three modules. Feature 1717 is the highest-priority acetylated-polyamine family hypothesis: 9/9 complete-detection Rmu pairs (+3.009 log2), 73 peak-resolved spectra across 45 samples, a 73/73 diagnostic product, same-source HILIC concordance and independent pathway context. Features 1597 and 3019 are jointly interpreted as a modified-guanosine module, not as two exact positional-isomer identifications. Features 150 and 3222 form a supporting long-chain-acylcarnitine module, with palmitoylcarnitine library consensus and recurrent class fragments serving different evidence roles.

All five remain below an exact new-metabolite claim. Feature 1717's former +4.817 estimate was traced to a missing-as-zero artifact in P28; the accepted complete-detection estimate is +3.009. Patient ordering correlations with Neu5Ac are +0.42, -0.10 and +0.12 for the polyamine, modified-guanosine and acylcarnitine modules, respectively, and the maximum absolute correlation with phenotype-blind pair-normalization factors is 0.317. Thus the modules are nonredundant descriptive axes, not a single global abundance shift; small n and post-selection preclude independence or causal claims.

## Result 6 — External evidence strengthens context and narrows, rather than inflates, novelty

Large prospective and targeted CRC studies independently implicate 1-methylguanosine and N2,N2-dimethylguanosine, while another targeted serum cohort reports methylguanosines in the opposite direction. This makes the local modified-guanosine family module biologically credible but explicitly tissue-, molecular- and isomer-context dependent. N1,N8-diacetylspermidine is already a reported urinary CRC-associated analyte, so feature 1717 is not claimed as the discovery of a previously unknown cancer metabolite; its value is paired-tissue recovery with convergent spectral evidence. Independent transcriptomic and glycan studies link mucinous CRC to sialylome and sialic-acid O-acetylation remodeling, strengthening the free-pool-to-glycan-destination decoupling hypothesis without replicating free Neu5Ac abundance.

## Required negative statements

- Full 13,155-target exact FDR10: zero hits.
- BioAware v1: zero corrections and one introduced error; network evidence does not raise identity confidence.
- P2b near-core regression prevents its use as an unqualified global upgrade.
- The initial L-kynurenine and 3-phenyllactic-acid stories do not survive biology-grade evidence calibration.
- Independent same-method Neu5Ac standard/spike-in, same-sample linkage-aware glycan destination, tracing, perturbation and rescue remain absent.

## Hard unresolved implementation gap

The local workspace still lacks `threeway_application_v1/*__threeway_features.csv.gz`. Aggregate official/E6/P2b comparison is frozen, but candidate-level attribution for feature 703 and other anchors is not locally verifiable. This gap must remain explicit until the two tables are synchronized and the frozen candidate-level audit is rerun.
"""
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")

    report = {
        "status": "mtbls13729_manuscript_evidence_package_v3_complete",
        "formal": True,
        "result_claims": int(len(results)),
        "figures": int(len(figures)),
        "biology_package_A_ready": True,
        "new_exact_metabolite_claims": 0,
        "hard_missing_items": [
            "candidate-level official/E6/P2b three-way feature tables",
            "independent subtype-resolved Neu5Ac metabolomics replication",
            "same-method Neu5Ac standard/spike-in and linkage-aware glycan readout",
            "N1,N8-diacetylspermidine positional-isomer standard panel",
            "methyl/dimethylguanosine and acylcarnitine standard panels",
        ],
        "claim_limit": biology["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

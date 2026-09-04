"""Build an auditable completion ledger for the MTBLS13729 biology paper.

The ledger separates observations already supported by frozen project artifacts
from evidence that would be required for identity, subtype, pathway, flux, or
causal claims.  It is deliberately not a post-hoc discovery score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v1"

SCORECARD = ROOT / "data/mtbls13729/candidate_claim_scorecard_v3/report.json"
SUBTYPE = ROOT / "data/mtbls13729/module_subtype_interactions_v1/report.json"
FULL_SPACE = ROOT / "data/mtbls13729/full_requantifiable_space_audit_v1/report.json"
BACKGROUND = ROOT / "data/mtbls13729/module_matched_background_sensitivity_v2/report.json"
COORDINATION = ROOT / "data/mtbls13729/module_coordination_v2/report.json"
MECHANISM = ROOT / "data/mtbls13729/mechanism_evidence_matrix_v2/report.json"
BIOAWARE = ROOT / "data/mtbls13729/bioaware_v1_eval/report.json"
TCGA = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1/report.json"
SPATIAL = ROOT / "data/external/GSE236697/spatial_metabolic_axes_v1/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    scorecard = load(SCORECARD)
    subtype = load(SUBTYPE)
    full_space = load(FULL_SPACE)
    background = load(BACKGROUND)
    coordination = load(COORDINATION)
    mechanism = load(MECHANISM)
    bioaware = load(BIOAWARE)
    tcga = load(TCGA)
    spatial = load(SPATIAL)

    neu5ac = [
        row
        for row in subtype["module_results"]
        if row["module"] == "neu5ac" and row["normalization"] in {"log_raw", "pqn_prev60"}
    ]
    if len(neu5ac) != 2:
        raise RuntimeError(f"expected two Neu5Ac normalization rows, found {len(neu5ac)}")
    if scorecard["full_untargeted_exact_fdr10_candidates"] != 0:
        raise RuntimeError("completion audit assumes the frozen full-space FDR10 result is zero")
    if full_space["panels"]["pos_rp"]["requantifiable_targets"] != 13155:
        raise RuntimeError("unexpected positive-RP full-space denominator")
    if bioaware["real_network"]["n_queries"] != 21:
        raise RuntimeError("unexpected BioAware v1 evaluation denominator")

    rows = [
        {
            "gate_id": "G01",
            "domain": "study_design",
            "status": "PASS",
            "evidence": "Paired Rmu-RN is primary; (Rmu-RN)-(Rtu-RN) is a separate subtype endpoint.",
            "claim_enabled": "Paired abundance and subtype-sensitivity can be reported separately.",
            "claim_forbidden": "Do not call a primary Rmu effect mucinous-specific without the interaction.",
            "next_action": "Freeze these endpoints in all tables, figures, and manuscript text.",
            "priority": "P0",
        },
        {
            "gate_id": "G02",
            "domain": "raw_data_quantification",
            "status": "PASS_WITH_LIMITATION",
            "evidence": "Targeted EIC and full-space discovery matrices exist; positive RP has 59 public samples and no pooled QC/blank.",
            "claim_enabled": "Raw-data paired abundance can be audited across normalization schemes.",
            "claim_forbidden": "Do not claim classical QC-CV drift correction or blank subtraction.",
            "next_action": "Retain pair-aware normalization sensitivity and disclose missing sample/QC limitations.",
            "priority": "P0",
        },
        {
            "gate_id": "G03",
            "domain": "ms2_identity",
            "status": "PARTIAL",
            "evidence": "Source Level-1 identities were orthogonally recovered; new 1717/1597/3019/3222 families remain unresolved or low-coverage.",
            "claim_enabled": "Neu5Ac and source identities can anchor pathway-level abundance; unresolved ions remain family-level.",
            "claim_forbidden": "Do not assign positional isomers or C20:4 acylcarnitine identity without standards.",
            "next_action": "Manual MS2/adduct review, then a minimal Neu5Ac and positional-isomer/acylcarnitine standard panel.",
            "priority": "P0",
        },
        {
            "gate_id": "G04",
            "domain": "multiplicity",
            "status": "FAIL_FULL_SPACE",
            "evidence": "0 candidates pass exact FDR10 in the complete 13,155 positive-RP target space; 132 pass only a nominal exact gate.",
            "claim_enabled": "Candidate-panel and module results may be labelled hypothesis-driven/same-cohort.",
            "claim_forbidden": "Do not describe any candidate as full-space untargeted FDR-confirmed.",
            "next_action": "Seek an independent cohort or pre-register a narrowly targeted validation panel; do not re-tune this cohort.",
            "priority": "P0",
        },
        {
            "gate_id": "G05",
            "domain": "subtype_biology",
            "status": "PASS_DISCOVERY",
            "evidence": (
                f"Neu5Ac is the only five-module subtype signal across raw/PQN: interaction q values "
                f"{neu5ac[0]['rmu_vs_rtu_bh_q_five_modules']:.4g} and "
                f"{neu5ac[1]['rmu_vs_rtu_bh_q_five_modules']:.4g}."
            ),
            "claim_enabled": "Mucinous-relative Neu5Ac/mucin-glycan remodeling is the primary discovery axis.",
            "claim_forbidden": "Do not call this global hypersialylation, linkage-specific glycosylation, or flux.",
            "next_action": "Validate Neu5Ac abundance independently and add a linkage/glycan-destination readout if feasible.",
            "priority": "P0",
        },
        {
            "gate_id": "G06",
            "domain": "matched_background",
            "status": "PASS_DESCRIPTIVE",
            "evidence": "Five frozen modules exceed acquisition-matched background in three outcome-blind sensitivity specifications.",
            "claim_enabled": "The module effects are not explained solely by m/z, RT, prevalence, support, or ion-family size.",
            "claim_forbidden": "Do not treat same-cohort post-selection tail areas as independent confirmation.",
            "next_action": "Keep this as robustness evidence and replicate the module panel externally.",
            "priority": "P1",
        },
        {
            "gate_id": "G07",
            "domain": "external_transcriptomics",
            "status": "PASS_CONTEXT",
            "evidence": "TCGA has 42 mucinous and 329 conventional tumours; sialic synthesis/transport and secretory-mucin programs are mucinous-enriched after broad-lineage adjustment.",
            "claim_enabled": "Independent transcriptomic context supports a mucinous sialic/mucin program.",
            "claim_forbidden": "Transcript levels do not replicate feature 703 abundance or glycan linkage.",
            "next_action": "Use TCGA as orthogonal context, not metabolite replication.",
            "priority": "P1",
        },
        {
            "gate_id": "G08",
            "domain": "spatial_or_cellular_localization",
            "status": "PARTIAL",
            "evidence": "One public spatial tumour/normal case provides compartment context; external mucinous glycoproteomics/glycomics support glycoform heterogeneity.",
            "claim_enabled": "A compartment-aware mechanistic model can be proposed.",
            "claim_forbidden": "One case cannot establish population-level cellular origin or subtype replication.",
            "next_action": "Prioritize a multi-patient mucinous spatial/glycomic dataset if one becomes accessible.",
            "priority": "P1",
        },
        {
            "gate_id": "G09",
            "domain": "pathway_coordination",
            "status": "NEGATIVE_RESULT",
            "evidence": "Patient-level module coordination does not support one shared upstream regulator; Neu5Ac is not coordinated with amino-acid or purine modules.",
            "claim_enabled": "Parallel abundance programs are a data-supported model.",
            "claim_forbidden": "Do not connect all modules into a single causal metabolic chain.",
            "next_action": "Write Neu5Ac as the subtype axis and retain other modules as parallel tumour programs.",
            "priority": "P0",
        },
        {
            "gate_id": "G10",
            "domain": "bioaware_network_expert",
            "status": "FAIL_V1",
            "evidence": "Phenotype-blind Rhea one-hop BioAware v1 had 21 queries, corrected 0 and introduced 1.",
            "claim_enabled": "The negative result justifies abstention and larger benchmark design.",
            "claim_forbidden": "Do not claim network-aware annotation improvement from v1.",
            "next_action": "Keep BioAware out of the biology claim until a larger external identity benchmark passes.",
            "priority": "P2",
        },
        {
            "gate_id": "G11",
            "domain": "independent_metabolomics_replication",
            "status": "FAIL_MISSING",
            "evidence": "No independent patient-level mucinous-vs-conventional Neu5Ac abundance dataset has been validated.",
            "claim_enabled": "Current result remains discovery-level.",
            "claim_forbidden": "Do not call TCGA or glycomic composition an independent metabolite replication.",
            "next_action": "Search/reprocess one public CRC cohort with histology labels, or obtain a narrowly targeted external assay.",
            "priority": "P0",
        },
        {
            "gate_id": "G12",
            "domain": "causal_mechanism",
            "status": "FAIL_MISSING",
            "evidence": "No isotope tracing, enzyme perturbation, rescue, organoid, or in-vivo intervention is available.",
            "claim_enabled": "Static abundance and orthogonal context can motivate competing mechanisms.",
            "claim_forbidden": "Do not claim flux, enzyme activity, pathway dependency, or therapeutic causality.",
            "next_action": "If wet work remains unavailable, explicitly position the paper as computational annotation plus biological discovery, not causal metabolism.",
            "priority": "P1",
        },
    ]
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "mechanism_paper_completion_audit_v1.csv", index=False)

    counts = ledger.status.value_counts().to_dict()
    report = {
        "status": "mtbls13729_mechanism_paper_completion_audit_v1_complete",
        "formal": False,
        "gates": int(len(ledger)),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "primary_publishable_phenomenon": "mucinous-relative Neu5Ac/mucin-glycan remodeling",
        "current_paper_position": (
            "Viable as an algorithm-plus-biological-application paper with a discovery-level subtype axis; "
            "not yet a causal mechanism paper or a full-space untargeted discovery claim."
        ),
        "minimum_publication_closure": [
            "freeze the Neu5Ac primary/subtype endpoint and preserve the full-space FDR limitation",
            "complete manual/standard-based identity validation for the primary and highest-novelty ions",
            "obtain one independent abundance replication or clearly limit the paper to discovery plus orthogonal context",
            "keep BioAware v1 and all flux/enzyme claims out unless their own validation gates pass",
        ],
        "negative_results_with_value": [
            "full-space exact FDR10 has zero hits",
            "BioAware v1 does not improve annotation",
            "module coordination rejects a single unified upstream chain",
        ],
        "provenance": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [SCORECARD, SUBTYPE, FULL_SPACE, BACKGROUND, COORDINATION, MECHANISM, BIOAWARE, TCGA, SPATIAL]
        },
        "claim_limit": (
            "This ledger audits evidence completeness. PASS_DISCOVERY and PASS_CONTEXT do not imply "
            "independent metabolite replication, MSI Level 1 confirmation in this assay, flux, or causality."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        "# MTBLS13729 mechanism-paper completion audit v1",
        "",
        f"**Primary publishable phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        f"**Current position:** {report['current_paper_position']}",
        "",
        "| Gate | Domain | Status | Current evidence | Shortest next action |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['gate_id']} | {row['domain']} | {row['status']} | "
            f"{row['evidence']} | {row['next_action']} |"
        )
    markdown.extend(
        [
            "",
            "## Claim boundary",
            "",
            report["claim_limit"],
            "",
            "## Minimum closure",
            "",
            *[f"- {item}" for item in report["minimum_publication_closure"]],
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

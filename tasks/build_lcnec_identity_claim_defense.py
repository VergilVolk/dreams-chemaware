"""Build a reviewer-facing identity and mechanism claim boundary for LCNEC.

This script does not promote any annotation.  It translates the frozen four-priority
evidence ledger into explicit claim tiers, unresolved alternatives, and the minimum
orthogonal evidence needed to upgrade each claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIORITY = ROOT / "data/validation/lcnec_hsst3n_manuscript_supplement/table_s5_4_priority_evidence_ledger.csv"
READINESS = ROOT / "data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json"
BIOAWARE = ROOT / "data/validation/lcnec_hsst3n_bioaware_context/bioaware_context_report.json"
BENCHMARK = ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1/report.json"
OUT = ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1"


POLICY = {
    "adenosine_diphosphate_family": {
        "claim_level": "connectivity-family hypothesis",
        "exact_identity_allowed": False,
        "primary_role": "abundance-family evidence only",
        "mechanism_axis": "adenylate/nucleotide pool remodeling",
        "main_ambiguity": "multiple full InChIKeys and a small spectral margin; ADP is also a high-degree currency hub",
        "standard_upgrade": "ADP authentic standard with matched chromatography and MS/MS; orthogonal AMP/ATP panel if energy-charge language is desired",
        "wet_priority": "low",
        "reason": "The current family signal is strong, but exact ADP and pathway-specific interpretations are not necessary for the primary paper claim.",
    },
    "adenosine_diphosphoribose_family": {
        "claim_level": "connectivity-family hypothesis",
        "exact_identity_allowed": False,
        "primary_role": "specific non-hub BioAware context anchor",
        "mechanism_axis": "NAD turnover / ADP-ribose handling",
        "main_ambiguity": "multiple full InChIKeys and no authentic retention-time match",
        "standard_upgrade": "ADP-ribose authentic standard with matched chromatography and MS/MS; isomer-aware comparison if available",
        "wet_priority": "medium",
        "reason": "It is biochemically informative and non-hub, but the present evidence supports a family rather than an exact stereochemical identity.",
    },
    "ascorbate": {
        "claim_level": "MSI Level-2 compound hypothesis",
        "exact_identity_allowed": False,
        "primary_role": "specific non-hub BioAware context anchor",
        "mechanism_axis": "antioxidant/redox-associated abundance remodeling",
        "main_ambiguity": "library-spectrum and formula support lack same-method authentic retention time",
        "standard_upgrade": "ascorbic acid authentic standard, co-injection or matched-method retention time, and MS/MS agreement",
        "wet_priority": "high",
        "reason": "Largest paired effect and strong fragment agreement make this the highest-yield single standard purchase, while oxidation-state and flux claims remain forbidden.",
    },
    "quinolinate": {
        "claim_level": "MSI Level-2 compound hypothesis",
        "exact_identity_allowed": False,
        "primary_role": "specific low-degree BioAware context anchor",
        "mechanism_axis": "tryptophan-to-de-novo-NAD pathway context",
        "main_ambiguity": "library-spectrum and formula support lack same-method authentic retention time",
        "standard_upgrade": "quinolinic acid authentic standard, co-injection or matched-method retention time, and MS/MS agreement",
        "wet_priority": "highest",
        "reason": "Low network degree and a focused biochemical reaction make this the most mechanistically discriminating standard target.",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(PRIORITY.open(encoding="utf-8-sig", newline="")))
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    bioaware = json.loads(BIOAWARE.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))

    if len(rows) != 4:
        raise RuntimeError(f"expected four frozen priority rows, found {len(rows)}")
    if {row["priority_name"] for row in rows} != set(POLICY):
        raise RuntimeError("priority ledger no longer matches the frozen claim policy")
    if not readiness["readiness"]["ready_for_algorithm_enabled_level2_biology_manuscript"]:
        raise RuntimeError("upstream manuscript readiness gate is not passing")

    ledger = []
    for row in rows:
        policy = POLICY[row["priority_name"]]
        ledger.append(
            {
                "priority_name": row["priority_name"],
                "spectral_hypothesis": row["spectral_hypothesis"].replace("&#39;", "'"),
                "ik14": row["ik14"],
                "formula": row["formula"],
                "target_mz": float(row["target_mz"]),
                "target_rt_sec": float(row["target_rt_sec"]),
                "formula_mass_error_ppm": float(row["formula_mass_error_ppm"]),
                "dreams_score": float(row["dreams_score"]),
                "dreams_margin": row["dreams_margin"],
                "reference_spectra": int(row["reference_spectra"]),
                "full_inchikey_count": int(row["full_inchikey_count"]),
                "matched_fragments": int(row["matched_fragments"]),
                "sqrt_cosine": float(row["sqrt_cosine"]),
                "entropy_similarity": float(row["entropy_similarity"]),
                "mean_per_mg_log2fc": float(row["mean_per_mg_log2fc"]),
                "concordant_pairs": int(row["concordant_pairs"]),
                "pairs": int(row["pairs"]),
                "leave_one_pair_out_sign_stable": parse_bool(row["leave_one_pair_out_sign_stable"]),
                "is_currency": parse_bool(row["is_currency"]),
                "bioaware_specific_anchor": parse_bool(row["bioaware_specific_anchor"]),
                **policy,
                "standard_required_for_current_claim": False,
                "standard_required_for_level1_upgrade": True,
            }
        )

    report = {
        "status": "lcnec_hsst3n_identity_claim_defense_complete",
        "formal": True,
        "priority_hypotheses": len(ledger),
        "new_exact_metabolite_claims": 0,
        "standards_required_for_current_claim_set": 0,
        "standards_required_to_upgrade_all_four_to_level1": 4,
        "highest_value_optional_standards": ["quinolinic acid", "ascorbic acid"],
        "primary_claim": readiness["allowed_primary_claim"],
        "identity_rule": "All four rows remain author-unreported Level-2/connectivity-family hypotheses. Formula, fragment, paired-abundance and network-context evidence may converge, but none substitutes for same-method authentic retention time.",
        "novelty_rule": "Author-unreported means absent from the source study's reported HSST3n table; it does not mean a chemically novel metabolite or a first report in cancer.",
        "network_rule": bioaware["decision"],
        "biology_rule": "Static paired abundance supports a remodeled metabolite pool. It does not establish flux, enzyme activity, dependence, adaptation, or therapeutic vulnerability.",
        "denominator_rule": benchmark["claim_limit"],
        "rows": ledger,
        "reviewer_objections": [
            {
                "objection": "How can these be real metabolites without standards?",
                "response": "The paper does not claim new Level-1 identities. It reports four explicitly labeled Level-2/connectivity-family hypotheses supported by exact formula, direct fragments, library spectra, paired abundance and leave-one-pair-out stability. Standards are required only to upgrade identity level.",
            },
            {
                "objection": "Did pathway knowledge choose the metabolite identity?",
                "response": "No. Spectral hypotheses were frozen before BioAware context. BioAware only retains non-hub pathway context and abstains on ADP; phenotype and pathway labels never promote an identity.",
            },
            {
                "objection": "Are these metabolites novel?",
                "response": "No chemical novelty claim is made. The novelty is analytical recovery from source-table-absent LCNEC MS/MS and the resulting coordinated abundance hypotheses.",
            },
            {
                "objection": "Does increased quinolinate or ADP-ribose prove NAD flux?",
                "response": "No. These static abundance changes motivate NAD-related pathway hypotheses; flux and enzyme activity require isotope tracing or functional perturbation.",
            },
        ],
        "provenance": {
            "priority_ledger": {"path": str(PRIORITY.relative_to(ROOT)), "sha256": sha256(PRIORITY)},
            "readiness": {"path": str(READINESS.relative_to(ROOT)), "sha256": sha256(READINESS)},
            "bioaware": {"path": str(BIOAWARE.relative_to(ROOT)), "sha256": sha256(BIOAWARE)},
            "annotation_benchmark": {"path": str(BENCHMARK.relative_to(ROOT)), "sha256": sha256(BENCHMARK)},
        },
        "claim_limit": "No authentic-standard, exact-stereoisomer, flux, enzyme-activity, causal-dependency, biomarker-performance or independent-metabolite-replication claim.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    columns = list(ledger[0])
    with (OUT / "priority_identity_claim_ledger.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ledger)

    md = [
        "# LCNEC priority identity-claim defense",
        "",
        "## Defensible primary statement",
        "",
        report["primary_claim"],
        "",
        "## Candidate-specific claim boundary",
        "",
        "| Candidate | Current claim | Mean paired log2FC | Concordance | BioAware role | Optional standard priority |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in ledger:
        md.append(
            f"| {item['priority_name']} | {item['claim_level']} | {item['mean_per_mg_log2fc']:.3f} | "
            f"{item['concordant_pairs']}/{item['pairs']} | {item['primary_role']} | {item['wet_priority']} |"
        )
    md += [
        "",
        "## Non-negotiable wording",
        "",
        "- `author-unreported` means absent from the source HSST3n table; it is not a chemical-novelty claim.",
        "- No row is an authentic-standard-confirmed Level-1 identity.",
        "- BioAware supplies context and hub abstention; it does not choose or validate identities.",
        "- Paired abundance is not metabolic flux or enzyme activity.",
        "- If only two standards can be purchased, quinolinic acid is the most mechanistically discriminating and ascorbic acid has the largest paired effect.",
        "",
        "## Reviewer response core",
        "",
    ]
    for objection in report["reviewer_objections"]:
        md += [f"**Objection:** {objection['objection']}", "", objection["response"], ""]
    (OUT / "REVIEWER_DEFENSE.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

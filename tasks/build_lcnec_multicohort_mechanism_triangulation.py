"""Build a conservative, paper-ready LCNEC multi-cohort evidence triangulation.

This is an evidence synthesis, not a new hypothesis test.  It combines only
frozen outputs and keeps metabolite identity, abundance, protein context and
flux as separate claim dimensions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1"
PRIORITY = ROOT / "data/validation/lcnec_hsst3n_manuscript_supplement/table_s5_4_priority_evidence_ledger.csv"
IDENTITY = ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1/priority_identity_claim_ledger.csv"
MECHANISM = ROOT / "data/validation/lcnec_hsst3n_mechanism_coherence_v1/axis_evidence_ledger.csv"
PROTEINS = ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/protein_results.csv"
PROTEIN_REPORT = ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/report.json"
SAME_UNIVERSE = ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_effect(value: float) -> str:
    return f"{value:+.2f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    priority = pd.read_csv(PRIORITY)
    identity = pd.read_csv(IDENTITY)
    mechanism = pd.read_csv(MECHANISM)
    proteins = pd.read_csv(PROTEINS)
    protein_report = json.loads(PROTEIN_REPORT.read_text(encoding="utf-8"))
    same_universe = json.loads(SAME_UNIVERSE.read_text(encoding="utf-8"))

    expected_priority = {
        "adenosine_diphosphate_family",
        "adenosine_diphosphoribose_family",
        "ascorbate",
        "quinolinate",
    }
    if set(priority["priority_name"]) != expected_priority:
        raise RuntimeError("frozen four-priority ledger changed")
    merged = priority.merge(
        identity[[
            "priority_name", "claim_level", "exact_identity_allowed", "primary_role",
            "mechanism_axis", "main_ambiguity", "standard_upgrade", "wet_priority",
        ]],
        on="priority_name",
        validate="one_to_one",
        suffixes=("", "_identity"),
    )
    if merged["exact_identity_allowed"].astype(bool).any():
        raise RuntimeError("no priority identity may be upgraded by this synthesis")

    passed = proteins.loc[proteins["primary_protein_gate"].astype(bool)].copy()
    protein_by_gene = proteins.set_index("gene")
    mechanism_counts = mechanism.groupby("axis").agg(
        local_metabolites=("family_id", "size"),
        source_reproduced=("evidence_class", lambda values: int(sum(str(v).startswith("R_") for v in values))),
        priority_hypotheses=("evidence_class", lambda values: int(sum(str(v).startswith("N_") for v in values))),
    )

    mappings = {
        "adenosine_diphosphate_family": {
            "display": "ADP connectivity family",
            "local_axis": "phosphorylated_nucleotide_and_sugar_accumulation",
            "protein_genes": [],
            "protein_context": "No frozen protein bridge specific to ADP",
            "independent_class": "none_specific",
            "integrated_interpretation": "Family-level sentinel of phosphorylated nucleotide-pool accumulation",
            "paper_role": "supporting abundance-family result",
            "validation_rank": 4,
        },
        "adenosine_diphosphoribose_family": {
            "display": "ADP-ribose connectivity family",
            "local_axis": "tryptophan_quinolinate_nad_context",
            "protein_genes": ["PARP1", "PARP2"],
            "protein_context": "PARP1 and PARP2 increased in independent pure LCNEC",
            "independent_class": "direct_pathway_context",
            "integrated_interpretation": "ADP-ribose/PARP turnover hypothesis; strongest coherent cross-omics context",
            "paper_role": "primary mechanism-triangulation hypothesis",
            "validation_rank": 2,
        },
        "ascorbate": {
            "display": "Ascorbate (MSI Level 2)",
            "local_axis": "antioxidant_pool_remodeling",
            "protein_genes": ["GSR", "TXNRD1", "G6PD", "TKT", "TALDO1"],
            "protein_context": "GSR/G6PD/TKT/TALDO1 down; TXNRD1 up; transporters not measured",
            "independent_class": "mixed_system_context",
            "integrated_interpretation": "Antioxidant-pool remodeling with mixed compensatory protein context",
            "paper_role": "high-effect analytical hypothesis",
            "validation_rank": 2,
        },
        "quinolinate": {
            "display": "Quinolinate (MSI Level 2)",
            "local_axis": "tryptophan_quinolinate_nad_context",
            "protein_genes": ["IDO1", "KYNU", "HAAO", "QPRT", "NMNAT3", "NADSYN1"],
            "protein_context": "QPRT and upstream enzymes down; NMNAT3 up",
            "independent_class": "bottleneck_consistent_with_redistribution",
            "integrated_interpretation": "Quinolinate-utilization bottleneck / NAD redistribution hypothesis; no flux claim",
            "paper_role": "highest chemical-confirmation priority",
            "validation_rank": 1,
        },
    }

    rows: list[dict[str, object]] = []
    for item in merged.to_dict(orient="records"):
        cfg = mappings[item["priority_name"]]
        local_axis = mechanism.loc[mechanism["axis"] == cfg["local_axis"]]
        if local_axis.empty:
            raise RuntimeError(f"missing local mechanism axis: {cfg['local_axis']}")
        genes = cfg["protein_genes"]
        gene_rows = protein_by_gene.loc[genes] if genes else pd.DataFrame()
        passing_genes = [] if not genes else gene_rows.loc[gene_rows["primary_protein_gate"].astype(bool)].index.tolist()
        directions = [] if not passing_genes else [
            f"{gene} {'up' if float(protein_by_gene.loc[gene, 'mean_effect']) > 0 else 'down'}"
            for gene in passing_genes
        ]
        rows.append({
            "priority_name": item["priority_name"],
            "display": cfg["display"],
            "spectral_hypothesis": item["spectral_hypothesis"],
            "claim_level": item["claim_level"],
            "exact_identity_allowed": False,
            "local_pairs": int(item["pairs"]),
            "local_mean_log2fc": float(item["mean_per_mg_log2fc"]),
            "local_effect_q": float(item["effect_q"]),
            "local_concordant_pairs": int(item["concordant_pairs"]),
            "local_direction_stable": bool(item["leave_one_pair_out_sign_stable"]),
            "dreams_score": float(item["dreams_score"]),
            "matched_fragments": int(item["matched_fragments"]),
            "formula_mass_error_ppm": float(item["formula_mass_error_ppm"]),
            "bioaware_specific_anchor": bool(item["bioaware_specific_anchor"]),
            "hub_abstention": bool(item["hub_abstention"]),
            "local_axis": cfg["local_axis"],
            "axis_local_metabolites": int(mechanism_counts.loc[cfg["local_axis"], "local_metabolites"]),
            "axis_source_reproduced": int(mechanism_counts.loc[cfg["local_axis"], "source_reproduced"]),
            "independent_protein_context": cfg["protein_context"],
            "independent_passing_proteins": ";".join(directions),
            "independent_support_class": cfg["independent_class"],
            "integrated_interpretation": cfg["integrated_interpretation"],
            "paper_role": cfg["paper_role"],
            "standard_upgrade": item["standard_upgrade"],
            "validation_rank": cfg["validation_rank"],
            "identity_boundary": "Level 1 requires same-method authentic RT and MS/MS",
            "flux_boundary": "static abundance and protein context do not establish flux or enzyme activity",
        })
    candidates = pd.DataFrame(rows).sort_values(["validation_rank", "priority_name"])
    candidates.to_csv(OUT / "candidate_triangulation.csv", index=False)

    axis_rows = [
        {
            "integrated_axis": "phosphorylated nucleotide-pool accumulation",
            "local_metabolite_evidence": "5/5 fixed-direction members; ADP family is author-unreported",
            "source_reproduced_members": 3,
            "author_unreported_priorities": 1,
            "independent_protein_evidence": "none specific in frozen panel",
            "triangulation_status": "strong local axis; no independent protein bridge",
            "allowed_claim": "measured phosphorylated nucleotide/sugar pools accumulate",
        },
        {
            "integrated_axis": "ADP-ribose / PARP turnover",
            "local_metabolite_evidence": "ADP-ribose family +1.56 log2, 31/34 concordant",
            "source_reproduced_members": 0,
            "author_unreported_priorities": 1,
            "independent_protein_evidence": "PARP1 +1.32 and PARP2 +0.87 log2; both fixed-gate pass",
            "triangulation_status": "strongest coherent independent context",
            "allowed_claim": "ADP-ribose/PARP turnover is prioritized for mechanistic follow-up",
        },
        {
            "integrated_axis": "quinolinate / de novo NAD redistribution",
            "local_metabolite_evidence": "tryptophan, quinolinate and ADP-ribose all increased",
            "source_reproduced_members": 1,
            "author_unreported_priorities": 2,
            "independent_protein_evidence": "IDO1/KYNU/HAAO/QPRT/NADSYN1 down; NMNAT3 up",
            "triangulation_status": "multi-level support with directional conflict/redistribution",
            "allowed_claim": "consistent with a utilization bottleneck or redistribution; not flux",
        },
        {
            "integrated_axis": "antioxidant-pool remodeling",
            "local_metabolite_evidence": "ascorbate/GSH/GSSG up; ophthalmate down",
            "source_reproduced_members": 3,
            "author_unreported_priorities": 1,
            "independent_protein_evidence": "GSR/G6PD/TKT/TALDO1 down; TXNRD1 up",
            "triangulation_status": "strong local axis with mixed compensatory protein context",
            "allowed_claim": "antioxidant pools are remodeled; no uniform PPP or redox-flux claim",
        },
    ]
    axes = pd.DataFrame(axis_rows)
    axes.to_csv(OUT / "mechanism_triangulation.csv", index=False)

    # Categorical evidence matrix: 2 direct/strong, 1 contextual, 0 absent,
    # -1 explicit limitation.  The values only control display colors; cell text
    # contains the auditable evidence.
    ordered = candidates.set_index("priority_name").loc[[
        "adenosine_diphosphoribose_family", "quinolinate", "ascorbate",
        "adenosine_diphosphate_family",
    ]]
    values = np.array([
        [2, 2, 1, 2, 2, -1],
        [2, 2, 1, 2, 2, -1],
        [2, 2, 2, 2, 1, -1],
        [2, 1, 2, 0, 0, -1],
    ], dtype=float)
    texts = []
    for name, row in ordered.iterrows():
        if name == "adenosine_diphosphoribose_family":
            protein_text = "PARP1/2 up"
        elif name == "quinolinate":
            protein_text = "QPRT down\nNMNAT3 up"
        elif name == "ascorbate":
            protein_text = "mixed redox\nproteins"
        else:
            protein_text = "no specific\nbridge"
        texts.append([
            f"{format_effect(row.local_mean_log2fc)} log2\n{row.local_concordant_pairs}/34",
            f"{int(row.matched_fragments)} fragments\n{row.formula_mass_error_ppm:.2f} ppm",
            f"{int(row.axis_source_reproduced)} source\naxis members",
            "specific" if bool(row.bioaware_specific_anchor) else ("hub abstain" if bool(row.hub_abstention) else "none"),
            protein_text,
            "not Level 1",
        ])
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    cmap = ListedColormap(["#E8C4C4", "#F0F0F0", "#CFE4F2", "#4F91BD"])
    shown = values + 1
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    columns = ["Paired abundance", "Spectral/formula", "Source-axis\nreproduction",
               "BioAware", "Independent proteins", "Identity boundary"]
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(ordered)), labels=ordered["display"].tolist())
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] == 2 else "#222222")
    ax.set_title("LCNEC multi-cohort evidence triangulation: support and explicit claim boundaries",
                 fontsize=14, fontweight="bold", pad=18)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.5, 0.015,
        "Independent proteins provide pathway context only. No row is an independently replicated or Level-1 metabolite; "
        "static abundance does not establish flux.",
        ha="center", fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"multicohort_triangulation.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "lcnec_multicohort_mechanism_triangulation_complete",
        "formal": True,
        "design": "frozen-output evidence synthesis; no new feature or protein selection",
        "counts": {
            "local_paired_patients": 34,
            "same_universe_families": 263,
            "source_table_overlap": 42,
            "official_dreams_candidates": 158,
            "full_evidence_retained": 66,
            "priority_hypotheses": 4,
            "independent_cohort_patients": 107,
            "independent_quantified_pairs": 103,
            "independent_pure_lcnec_pairs": 80,
            "fixed_proteins": 22,
            "passing_proteins": int(passed.shape[0]),
        },
        "decisions": {
            "strongest_cross_omics_mechanism_context": "ADP-ribose / PARP turnover",
            "highest_chemical_confirmation_priority": "quinolinate",
            "largest_local_effect_priority": "ascorbate",
            "family_level_sentinel_only": "ADP connectivity family",
            "new_exact_metabolite_claims": 0,
        },
        "boundaries": {
            "source_annotation_rate": "not reconstructable from the published source supplement",
            "same_universe_counts": "coverage/evidence yield, not annotation accuracy",
            "independent_proteins": "pathway context, not metabolite replication",
            "identity": "all four remain Level-2 or connectivity-family hypotheses",
            "flux": "not established",
        },
        "protein_report_status": protein_report["status"],
        "same_universe_status": same_universe["status"],
        "provenance": {
            path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in [PRIORITY, IDENTITY, MECHANISM, PROTEINS, PROTEIN_REPORT, SAME_UNIVERSE]
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    readme = "# LCNEC multi-cohort mechanism triangulation\n\n"
    readme += "This directory integrates frozen metabolite, source-atlas, BioAware and independent-protein evidence. "
    readme += "It does not reselect candidates or upgrade metabolite identity.\n\n"
    readme += "## Main result\n\n"
    readme += "- ADP-ribose/PARP is the cleanest cross-omics mechanism context.\n"
    readme += "- Quinolinate is the highest-priority standard target and supports a bottleneck/redistribution hypothesis.\n"
    readme += "- Ascorbate has the largest local effect but only mixed independent redox-protein context.\n"
    readme += "- ADP remains a family-level nucleotide-pool sentinel because it is a hub and exact identity is unresolved.\n\n"
    readme += "No new exact metabolite, flux, enzyme-activity or causal claim is made.\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

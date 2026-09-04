"""Extend the frozen mechanism ledger with proline/glutamate and Neu5Ac evidence.

The v1 ledger is retained verbatim.  New rows deliberately separate local
abundance, same-cohort orthogonal identity recovery, general CRC context,
mucinous-relative context, and counterevidence.  No downstream algorithm score
is used as biological truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/mechanism_evidence_matrix_v2"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row(**values: object) -> dict[str, object]:
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_path = ROOT / "data/mtbls13729/mechanism_evidence_matrix_v1/mechanism_evidence_matrix.csv"
    ledger_path = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
    tcga_path = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_axes_v1/summary.json"
    gse_path = ROOT / "data/external/GSE236696/proline_sialic_by_lineage_v1/summary.json"
    gse_null_path = ROOT / "data/external/GSE236696/proline_genomewide_matched_null_v1/summary.json"
    spatial_path = ROOT / "data/external/GSE236697/spatial_proline_sialic_v1/report.json"
    proteomics_path = ROOT / "data/external/mucinous_crc_proteomics_2021/proline_sialic_reanalysis_v1/axis_summary.csv"

    required = [v1_path, ledger_path, tcga_path, gse_path, gse_null_path, spatial_path, proteomics_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    base = pd.read_csv(v1_path)
    ledger = pd.read_csv(ledger_path).set_index("feature_id")
    tcga = load_json(tcga_path)
    gse = load_json(gse_path)
    gse_null = load_json(gse_null_path)
    spatial = load_json(spatial_path)
    proteomics = pd.read_csv(proteomics_path).set_index("axis")

    additions: list[dict[str, object]] = []
    for feature, axis, identity in [
        (345, "proline_p5c_matrix", "proline"),
        (374, "proline_p5c_matrix", "glutamic acid"),
        (703, "neu5ac_mucin_glycan", "N-acetylneuraminic acid (Neu5Ac)"),
    ]:
        item = ledger.loc[feature]
        additions.append(
            row(
                axis=axis,
                dataset="MTBLS13729",
                evidence_layer="paired tissue abundance plus same-cohort orthogonal identity recovery",
                cohort_or_unit="10 Rmu tumour-normal pairs; source and recovery panels measured the same cohort",
                contrast="Rmu vs matched RN",
                observation=(
                    f"feature {feature} {identity}: mean log2FC {float(item.mean_log2fc):+.3f}; "
                    f"{int(item.positive_pairs)}/{int(item.pairs)} positive; within-tissue rho "
                    f"{float(item.crosspanel_within_tissue_spearman):.3f}; paired-delta rho "
                    f"{float(item.crosspanel_paired_spearman):.3f}; source rank {int(item.crosspanel_source_rank)}"
                ),
                statistical_support=(
                    f"patient bootstrap 95% CI [{float(item.abundance_bootstrap_ci_low):+.3f}, "
                    f"{float(item.abundance_bootstrap_ci_high):+.3f}]; classical median cosine "
                    f"{float(item.classical_median_cosine):.3f}; {int(item.classical_strong_support_samples)} strong-support samples"
                ),
                independent_of_discovery="no",
                subtype_relevance="direct Rmu discovery, but not an independent subtype-resolved cohort",
                identity_resolution=(
                    f"source-table {item.published_source_msi}; orthogonal-panel recovery with raw MS2 and "
                    "same-sample abundance coupling; not a new same-method standard injection"
                ),
                causal_strength="descriptive human abundance with orthogonal technical identity support",
                claim_allowed=f"increased recoverable {identity} pool in the Rmu discovery subgroup",
                claim_forbidden="new metabolite discovery, independent replication, flux, enzyme activity, or causality",
            )
        )

    paired = {item["axis"]: item for item in tcga["paired_tumor_normal"]["results"]}
    subtype = {item["axis"]: item for item in tcga["results"]}
    for tcga_axis, axis in [
        ("proline_synthesis", "proline_p5c_matrix"),
        ("glutamate_supply", "proline_p5c_matrix"),
        ("sialic_acid_synthesis_transport", "neu5ac_mucin_glycan"),
        ("mucin_sialylation", "neu5ac_mucin_glycan"),
        ("secretory_mucin_program", "neu5ac_mucin_glycan"),
        ("collagen_proline_context", "proline_p5c_matrix"),
    ]:
        p = paired[tcga_axis]
        s = subtype[tcga_axis]
        additions.append(
            row(
                axis=axis,
                dataset="TCGA COADREAD",
                evidence_layer="bulk RNA general-CRC and histology-relative context",
                cohort_or_unit="32 paired tumour-normal patients; 42 mucinous and 329 conventional primary tumours",
                contrast=f"{tcga_axis}: paired tumour-normal and adjusted mucinous-conventional",
                observation=(
                    f"paired delta {float(p['mean_tumor_minus_normal']):+.3f} "
                    f"({int(p['tumor_higher_pairs'])}/32 tumour-higher); adjusted mucinous beta "
                    f"{float(s['adjusted_beta']):+.3f}"
                ),
                statistical_support=(
                    f"paired BH q={float(p['wilcoxon_bh_q_all_axes']):.4g}; "
                    f"subtype BH q={float(s['adjusted_hc3_bh_q_all_axes']):.4g}"
                ),
                independent_of_discovery="yes",
                subtype_relevance=(
                    "general CRC and mucinous-relative effects are distinct endpoints; TCGA paired data contain "
                    "too few mucinous pairs to estimate a mucinous paired effect"
                ),
                identity_resolution="bulk transcript pathway axis, not metabolite or glycan identity",
                causal_strength="large-cohort orthogonal context",
                claim_allowed="general CRC or mucinous-relative pathway context exactly as specified by the contrast",
                claim_forbidden="metabolite replication, cell origin, glycan linkage, flux, or causal enzyme assignment",
            )
        )

    epithelial = {
        item["axis"]: item
        for item in gse["axis_results"]
        if item["lineage"] == "epithelial"
    }
    p = epithelial["proline_synthesis"]
    null = gse_null["results"]["proline_synthesis"]
    additions.append(
        row(
            axis="proline_p5c_matrix",
            dataset="GSE236696",
            evidence_layer="paired marker-gated epithelial pseudobulk with matched random-axis audit",
            cohort_or_unit="6 mucinous CRC tumour-normal patient pairs",
            contrast="epithelial tumour vs matched normal",
            observation=f"proline-synthesis delta {float(p['mean_paired_delta']):+.3f}; {int(p['tumor_higher_pairs'])}/6 positive",
            statistical_support=(
                f"two-sided exact p={float(p['exact_sign_flip_p']):.4g}; genome-wide matched-null magnitude "
                f"p={float(null['directional_empirical_p_mean']):.3f}, concordance p={float(null['directional_empirical_p_concordance']):.3f}"
            ),
            independent_of_discovery="yes",
            subtype_relevance="mucinous-labelled cohort, but broad marker gating is not malignant-cell CNV annotation",
            identity_resolution="transcript pathway score",
            causal_strength="directional but non-specific orthogonal context",
            claim_allowed="PYCR/proline direction is compatible in a small epithelial sensitivity analysis",
            claim_forbidden="significant or specific epithelial replication, metabolite abundance, flux, or causality",
        )
    )

    prot = proteomics.loc["proline_synthesis"]
    additions.append(
        row(
            axis="proline_p5c_matrix",
            dataset="independent pooled mucinous CRC proteomics",
            evidence_layer="pooled protein pathway context",
            cohort_or_unit="29-patient source study pooled into tissue groups; no patient-level matrix",
            contrast="left/right mucinous tumour vs normal",
            observation=(
                f"proline-synthesis proteins positive "
                f"{int(round(float(prot.LMC_vs_NC__positive_fraction) * int(prot.LMC_vs_NC__n)))}/"
                f"{int(prot.LMC_vs_NC__n)} "
                f"in left mucinous (median {float(prot.LMC_vs_NC__median_log2):+.3f}); right mucinous median "
                f"{float(prot.RMC_vs_NC__median_log2):+.3f}"
            ),
            statistical_support="descriptive pooled direction only; no patient-level p-value is available",
            independent_of_discovery="yes",
            subtype_relevance="mucinous-labelled orthogonal protein context",
            identity_resolution="protein pathway axis",
            causal_strength="descriptive orthogonal support",
            claim_allowed="proline/P5C enzyme abundance is directionally compatible at pooled group level",
            claim_forbidden="patient-level replication, proline flux, tumour-cell origin, or enzyme causality",
        )
    )

    spatial_rows = {item["axis"]: item for item in spatial["tumour_normal_descriptive"]}
    for spatial_axis, axis in [
        ("proline_synthesis", "proline_p5c_matrix"),
        ("sialic_acid_synthesis_transport", "neu5ac_mucin_glycan"),
        ("mucin_sialylation", "neu5ac_mucin_glycan"),
        ("secretory_mucin_program", "neu5ac_mucin_glycan"),
        ("collagen_proline_context", "proline_p5c_matrix"),
    ]:
        item = spatial_rows[spatial_axis]
        additions.append(
            row(
                axis=axis,
                dataset="GSE236697 spatial transcriptomics",
                evidence_layer="single paired-case spatial context",
                cohort_or_unit="one mucinous CRC case; 3,481 tumour and 1,725 normal spots",
                contrast=f"{spatial_axis}: tumour vs normal spot distributions",
                observation=(
                    f"tumour-minus-normal median score {float(item['tumour_minus_normal_median']):+.3f}; "
                    f"descriptive Cliff's delta {float(item['cliffs_delta_spot_distribution_descriptive']):+.3f}"
                ),
                statistical_support="no population p-value; spots are not biological replicates",
                independent_of_discovery="yes",
                subtype_relevance="mucinous spatial localization in one case only",
                identity_resolution="spatial transcript score",
                causal_strength="descriptive localization/counterevidence",
                claim_allowed="secretory-mucin and matrix compartment context in this single case",
                claim_forbidden="population replication, metabolite abundance, glycan structure, flux, or causality",
            )
        )

    addition_frame = pd.DataFrame(additions)
    key = ["axis", "dataset", "evidence_layer", "contrast", "observation"]
    if addition_frame.duplicated(key).any():
        raise RuntimeError("duplicate newly added evidence rows in v2 matrix")
    frame = pd.concat([base, addition_frame], ignore_index=True)
    frame.insert(0, "evidence_id", [f"MEV2-{i:03d}" for i in range(len(frame))])
    frame.to_csv(OUT / "mechanism_evidence_matrix_v2.csv", index=False)

    summary = {
        "status": "mtbls13729_mechanism_evidence_matrix_v2_complete",
        "formal": False,
        "rows": int(len(frame)),
        "v1_rows_preserved": int(len(base)),
        "v2_rows_added": int(len(additions)),
        "axes": sorted(frame.axis.unique().tolist()),
        "datasets": sorted(frame.dataset.unique().tolist()),
        "central_model": (
            "The discovery subgroup contains parallel abundance programs rather than one proven causal chain: "
            "a general CRC proline/P5C-matrix program; a mucinous-relative but internally heterogeneous "
            "Neu5Ac/mucin-glycan program; and previously defined modified-guanosine, acetylated-polyamine, "
            "purine and long-chain acylcarnitine programs."
        ),
        "strongest_new_identity_result": (
            "Features 345, 374 and 703 are same-cohort orthogonal recoveries of source-table Level-1 proline, "
            "glutamate and Neu5Ac, respectively; they are not new metabolites or independent replication."
        ),
        "adversarial_verdict": (
            "GSE236696 matched random axes do not establish a specific epithelial proline program; one-case "
            "spatial data do not show tumour-wide proline or sialic-axis elevation; pooled proteomics lacks "
            "patient-level inference; TCGA separates general CRC from mucinous-relative effects."
        ),
        "causal_verdict": (
            "Static abundance plus transcript/protein context cannot establish flux, glycan linkage, enzyme "
            "activity, cellular source or causality."
        ),
        "provenance": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in required
        },
    }
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = [
        "axis", "dataset", "evidence_layer", "observation", "statistical_support",
        "subtype_relevance", "claim_allowed", "claim_forbidden",
    ]
    markdown = [
        "# MTBLS13729 mechanism evidence matrix v2",
        "",
        "> Abundance, identity, subtype context, cellular/spatial context and causality are separate evidence layers.",
        "",
        frame[columns].to_markdown(index=False),
        "",
        "## Frozen synthesis",
        "",
        f"- Central model: {summary['central_model']}",
        f"- New identity result: {summary['strongest_new_identity_result']}",
        f"- Adversarial verdict: {summary['adversarial_verdict']}",
        f"- Causal verdict: {summary['causal_verdict']}",
        "",
    ]
    (OUT / "mechanism_evidence_matrix_v2.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

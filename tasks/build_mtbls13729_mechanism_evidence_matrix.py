"""Build an auditable cross-cohort mechanism evidence and claim-boundary matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/mechanism_evidence_matrix_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add(rows: list[dict], **kwargs: object) -> None:
    rows.append(kwargs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local_path = ROOT / "data/mtbls13729/biology_closure_analysis_v1/candidate_identity_and_abundance.csv"
    st_path = ROOT / "data/mtbls13729/external_st001087_axis_validation_v1/external_axis_metabolite_results.csv"
    tcga_path = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/summary.json"
    gse_path = ROOT / "data/external/GSE236696/epithelial_axis_adversarial_audit_v1/score_method_results.csv"
    poly_path = ROOT / "data/external/GSE236696/polyamine_gate_specificity_v1/summary.json"
    prot_path = ROOT / "data/external/mucinous_crc_proteomics_2021/axis_reanalysis_v1/axis_summary.csv"
    oep_path = ROOT / "data/external/OEP00006137_support/modified_guanosine_reanalysis/report.json"
    risk_gse_path = ROOT / "data/external/GSE281917/mucinous_axis_composition_audit_v1/report.json"
    risk_tcga_path = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/mucinous_risk_axis_replication_v1/report.json"

    local = pd.read_csv(local_path).set_index("feature_id")
    st = pd.read_csv(st_path).set_index("metabolite")
    tcga = load_json(tcga_path)
    gse = pd.read_csv(gse_path)
    poly = load_json(poly_path)
    prot = pd.read_csv(prot_path).set_index("axis")
    oep = load_json(oep_path)
    risk_gse = load_json(risk_gse_path)
    risk_tcga = load_json(risk_tcga_path)
    rows: list[dict] = []

    # Discovery cohort: separate abundance from identity and from interaction specificity.
    for feature, axis, label, resolution in [
        (1597, "modified_guanosine", "methylguanosine isomer family", "42 peak-resolved MS2; m/z 166.0725 in 71.4%, consistent with several methylguanosine standards; positional isomer unresolved"),
        (3019, "modified_guanosine", "dimethylguanosine isomer family", "32 peak-resolved MS2; m/z 180.0886 in 100%, consistent with several dimethylguanosine standards; positional isomer unresolved"),
        (4966, "purine_turnover", "C7H9N5O purine-like family", "formula family only; database identities discordant"),
        (1717, "polyamine_acetylation", "diacetylspermidine-like", "73 peak-resolved MS2; m/z 100.0759 in 100%, consistent with the published authentic-standard 230.2-to-100.0 transition; no same-method standard RT/full-spectrum match"),
        (3222, "long_chain_acylcarnitine", "C20:4 acylcarnitine-like", "30 peak-resolved consensus MS2 with recurrent m/z 85.0281/60.0808 acylcarnitine-class fragments; chain position/isomer and standard RT unresolved"),
    ]:
        item = local.loc[feature]
        add(
            rows,
            axis=axis,
            dataset="MTBLS13729",
            evidence_layer="paired tissue abundance",
            cohort_or_unit="10 Rmu tumour-normal pairs (feature-dependent complete pairs shown)",
            contrast="Rmu vs matched RN",
            observation=f"feature {feature} {label}: mean log2FC {float(item.rmu_mean_log2fc):+.3f}; {int(item.rmu_n)} complete pairs; positive fraction {float(item.rmu_positive_fraction):.2f}",
            statistical_support=f"exact sign-flip p={float(item.rmu_exact_signflip_p):.5g}; leave-one-patient-out direction stable={bool(item.rmu_loo_direction_stable)}",
            independent_of_discovery="no",
            subtype_relevance="direct Rmu discovery; interaction is a separate endpoint and not globally FDR-confirmed",
            identity_resolution=resolution,
            causal_strength="descriptive human abundance",
            claim_allowed="Rmu-associated abundance feature/family and candidate pathway axis",
            claim_forbidden="flux, enzyme activity, exact positional isomer, or established mucinous specificity",
        )

    # Independent metabolomics: supportive and adversarial directions are retained together.
    for metabolite, axis in [
        ("N2,N2-Dimethylguanosine", "modified_guanosine"),
        ("N1,N12-Diacetylspermine", "polyamine_acetylation"),
    ]:
        item = st.loc[metabolite]
        add(
            rows,
            axis=axis,
            dataset="ST001087",
            evidence_layer="independent paired tissue metabolomics",
            cohort_or_unit="17 CRC tumour-normal pairs; sparse FindByFormula annotations",
            contrast="tumour vs matched normal",
            observation=f"{metabolite}: mean log2FC {float(item.all_pairs_mean_log2fc):+.3f}; tumour-only {int(item.tumor_only_detected_pairs)}, normal-only {int(item.normal_only_detected_pairs)}",
            statistical_support=f"all-pair sign p={float(item.all_pairs_sign_p):.5g}; detection McNemar p={float(item.paired_detection_mcnemar_p):.5g}",
            independent_of_discovery="yes",
            subtype_relevance="generic CRC; no mucinous label",
            identity_resolution="formula/adduct-level external annotation, not positional-isomer Level 1",
            causal_strength="descriptive external abundance",
            claim_allowed="independent family-level direction/context",
            claim_forbidden="exact-isomer replication or mucinous specificity",
        )

    oep_met = oep["metabolites"]["N2,N2-Dimethylguanosine"]
    for subtype in ["MSI", "MSS"]:
        item = oep_met[subtype]
        add(
            rows,
            axis="modified_guanosine",
            dataset="OEP00006137",
            evidence_layer="independent paired Level-1 tissue metabolomics",
            cohort_or_unit=f"CRC {subtype} paired tumour-normal subset",
            contrast="tumour vs matched normal",
            observation=f"N2,N2-dimethylguanosine mean log2FC {float(item['mean_log2fc']):+.3f}",
            statistical_support=f"positive pairs {int(item['positive_pairs'])}/{int(item['n_pairs'])}; exact sign p={float(item['exact_sign_p']):.5g}",
            independent_of_discovery="yes",
            subtype_relevance="molecular subtype, not mucinous histology",
            identity_resolution="Level 1 in source matrix; raw frozen-coordinate re-extraction independently reproduced direction",
            causal_strength="adversarial external abundance",
            claim_allowed="metabolite direction is cohort/subtype dependent",
            claim_forbidden="universal CRC increase or direct replication of local positional isomer",
        )

    # Orthogonal expression/proteomics context.
    gse_primary = gse[gse.score_method == "difference_of_gene_medians"].set_index("axis")
    for axis_key, axis in [
        ("modified_nucleoside_processing", "modified_guanosine"),
        ("purine_synthesis_salvage", "purine_turnover"),
        ("carnitine_long_chain_fao", "long_chain_acylcarnitine"),
    ]:
        item = gse_primary.loc[axis_key]
        add(
            rows,
            axis=axis,
            dataset="GSE236696",
            evidence_layer="paired epithelial single-cell pseudobulk transcript context",
            cohort_or_unit="6 mucinous CRC tumour-normal patient pairs",
            contrast="epithelial tumour vs matched normal",
            observation=f"{axis_key} axis delta {float(item['mean']):+.3f}; {int(item.positive_pairs)}/6 positive",
            statistical_support=f"two-sided exact p={float(item.two_sided_exact_p):.5g}; composition and matched-null sensitivity reported separately",
            independent_of_discovery="yes",
            subtype_relevance="mucinous-labelled orthogonal cohort, but transcript score is not metabolite replication",
            identity_resolution="pathway gene set, not metabolite identity",
            causal_strength="orthogonal mechanism-supporting context",
            claim_allowed="directional epithelial pathway context",
            claim_forbidden="metabolite abundance, flux, or writer-enzyme causality",
        )

    poly_item = poly["results"]["broad_frozen"]["axes"]["polyamine_acetylation_catabolism"]
    add(
        rows,
        axis="polyamine_acetylation",
        dataset="GSE236696",
        evidence_layer="paired epithelial single-cell pseudobulk transcript context",
        cohort_or_unit="6 mucinous CRC tumour-normal patient pairs",
        contrast="epithelial tumour vs matched normal",
        observation=f"polyamine acetylation/catabolism delta {float(poly_item['mean_tumour_minus_normal']):+.3f}; {int(poly_item['positive_patients'])}/6 positive",
        statistical_support=f"exact sign-flip p={float(poly_item['exact_sign_flip_p']):.5g}; matched-null mean-effect p={float(poly_item['matched_null']['directional_empirical_p_mean']):.5g}; alternative epithelial gates are sensitivity analyses",
        independent_of_discovery="yes",
        subtype_relevance="mucinous-labelled orthogonal cohort",
        identity_resolution="pathway gene set, not N1,N8-diacetylspermidine identity",
        causal_strength="orthogonal mechanism-supporting context",
        claim_allowed="polyamine acetylation/catabolism program is directionally compatible",
        claim_forbidden="SAT1 causality, secretion, immune recruitment causality, or metabolite flux",
    )

    tcga_pair = {x["axis"]: x for x in tcga["paired_tumor_normal"]["results"]}
    tcga_sub = {x["axis"]: x for x in tcga["results"]}
    for axis_key, axis in [
        ("modified_nucleoside_processing", "modified_guanosine"),
        ("purine_synthesis_salvage", "purine_turnover"),
        ("carnitine_long_chain_fao", "long_chain_acylcarnitine"),
    ]:
        pair = tcga_pair[axis_key]
        sub = tcga_sub[axis_key]
        add(
            rows,
            axis=axis,
            dataset="TCGA COADREAD",
            evidence_layer="bulk RNA pathway context",
            cohort_or_unit="32 paired tumour-normal; 42 mucinous and 329 conventional tumours",
            contrast="paired tumour-normal and covariate-adjusted mucinous-conventional",
            observation=f"tumour-normal axis {float(pair['mean_tumor_minus_normal']):+.3f}; mucinous-conventional beta {float(sub['adjusted_beta']):+.3f}",
            statistical_support=f"paired BH q={float(pair['wilcoxon_bh_q_5axes']):.4g}; subtype HC3 p={float(sub['adjusted_hc3_p']):.4g}",
            independent_of_discovery="yes",
            subtype_relevance="directly tests subtype; does not support enhanced modified-nucleoside or FAO program in mucinous tumours",
            identity_resolution="pathway gene set",
            causal_strength="large-cohort orthogonal context and adversarial subtype test",
            claim_allowed="general CRC program where paired evidence supports it; explicit non-replication of mucinous specificity",
            claim_forbidden="mucinous-specific metabolite mechanism or flux",
        )

    for axis_key, axis in [
        ("modified_nucleoside_processing", "modified_guanosine"),
        ("purine_synthesis_salvage", "purine_turnover"),
        ("carnitine_long_chain_fao", "long_chain_acylcarnitine"),
    ]:
        item = prot.loc[axis_key]
        add(
            rows,
            axis=axis,
            dataset="independent pooled mucinous CRC proteomics",
            evidence_layer="pooled protein pathway context",
            cohort_or_unit="pooled LMC/LNMC/RMC/RNMC/normal groups; not patient-level",
            contrast="LMC-normal and LMC-LNMC",
            observation=f"LMC-normal median log2 {float(item['LMC_vs_NC__median_log2']):+.3f}; LMC-LNMC {float(item['LMC_vs_LNMC__median_log2']):+.3f}",
            statistical_support=f"genes detected {int(item.genes_detected)}/{int(item.genes_requested)}; descriptive pooled direction only",
            independent_of_discovery="yes",
            subtype_relevance="mucinous-vs-nonmucinous direction available but pooled, without patient-level inference",
            identity_resolution="protein pathway axis",
            causal_strength="descriptive orthogonal context",
            claim_allowed="directional pathway consistency only",
            claim_forbidden="statistical patient replication, metabolite identity, or flux",
        )

    # Risk-context associations are a different endpoint from tumour-normal
    # abundance and subtype contrasts.  They are retained as such rather than
    # being counted as metabolite replication.
    axis_map = {
        "modified_nucleoside_processing": "modified_guanosine",
        "purine_synthesis_salvage": "purine_turnover",
        "carnitine_long_chain_fao": "long_chain_acylcarnitine",
        "polyamine_acetylation_catabolism": "polyamine_acetylation",
    }
    for item in risk_gse["axis_associations"]:
        add(
            rows,
            axis=axis_map[item["axis"]],
            dataset="GSE281917 MuC23",
            evidence_layer="within-mucinous bulk transcript risk context with composition sensitivity",
            cohort_or_unit="140 mucinous CRC tumours",
            contrast="association with frozen MuC23 risk score",
            observation=(
                f"{item['axis']}: clinical-adjusted rho {float(item['clinical_adjusted_rho']):+.3f}; "
                f"clinical+lineage rho {float(item['clinical_and_lineage_adjusted_rho']):+.3f}"
            ),
            statistical_support=(
                f"composition-adjusted 95% CI [{float(item['clinical_and_lineage_adjusted_ci_low']):+.3f}, "
                f"{float(item['clinical_and_lineage_adjusted_ci_high']):+.3f}]; "
                f"BH q={float(item['clinical_and_lineage_adjusted_q']):.4g}"
            ),
            independent_of_discovery="yes",
            subtype_relevance="within-mucinous risk association; not a mucinous-vs-conventional contrast",
            identity_resolution="bulk transcript pathway score",
            causal_strength="risk-associated context with post-hoc broad-lineage sensitivity",
            claim_allowed="bulk risk-state alignment and explicit composition sensitivity",
            claim_forbidden="metabolite replication, independent prognosis, cell-autonomous metabolism, flux, or causality",
        )

    tcga_risk_rows = [risk_tcga["primary_result"], *risk_tcga["secondary_results"]]
    for item in tcga_risk_rows:
        add(
            rows,
            axis=axis_map[item["axis"]],
            dataset="TCGA COADREAD MuC23 targeted replication",
            evidence_layer="within-mucinous bulk transcript risk-context replication",
            cohort_or_unit="42 mucinous primary tumours; TCGA previously used for related axis analyses",
            contrast="association with frozen MuC23 risk score",
            observation=(
                f"{item['axis']}: clinical-adjusted rho {float(item['clinical_adjusted_rho']):+.3f}; "
                f"clinical+lineage rho {float(item['clinical_and_lineage_adjusted_rho']):+.3f}"
            ),
            statistical_support=(
                f"composition-adjusted 95% CI [{float(item['clinical_and_lineage_adjusted_ci_low']):+.3f}, "
                f"{float(item['clinical_and_lineage_adjusted_ci_high']):+.3f}]; "
                f"p={float(item['clinical_and_lineage_adjusted_p']):.4g}"
            ),
            independent_of_discovery="yes",
            subtype_relevance="within-mucinous risk association; targeted direction check, not a new blinded cohort",
            identity_resolution="bulk transcript pathway score",
            causal_strength="small-cohort targeted risk-context sensitivity",
            claim_allowed="directional reproducibility or non-replication of a bulk risk-state association",
            claim_forbidden="metabolite abundance, independent prognosis, cell-autonomous metabolism, flux, or causality",
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "mechanism_evidence_matrix.csv", index=False)
    summary = {
        "status": "mtbls13729_mechanism_evidence_matrix_complete",
        "rows": len(frame),
        "axes": sorted(frame.axis.unique().tolist()),
        "datasets": sorted(frame.dataset.unique().tolist()),
        "independent_rows": int((frame.independent_of_discovery == "yes").sum()),
        "central_model": "Rmu exhibits high-amplitude pools in modified-guanosine/purine, acetylated-polyamine, and long-chain acylcarnitine axes within broader CRC programs. Acylcarnitine accumulation supports carnitine-shuttle imbalance, while increased entry, incomplete oxidation, impaired downstream utilization and tissue composition remain competing explanations; it is not proof of flux direction.",
        "subtype_verdict": "Mucinous specificity is not externally established. TCGA does not show stronger modified-nucleoside or FAO expression axes in mucinous tumours, and independent metabolite cohorts are heterogeneous.",
        "causal_verdict": "No isotope tracing, authentic-standard positional-isomer confirmation, or enzyme perturbation is available; therefore the result is mechanism-supporting but not causal.",
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in [local_path, st_path, tcga_path, gse_path, poly_path, prot_path, oep_path, risk_gse_path, risk_tcga_path]},
    }
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["axis", "dataset", "evidence_layer", "observation", "subtype_relevance", "claim_allowed", "claim_forbidden"]
    markdown = [
        "# MTBLS13729 cross-cohort mechanism evidence matrix",
        "",
        "> This is an auditable evidence ledger. Abundance, orthogonal pathway context, subtype replication, structural identity and causality are deliberately separated.",
        "",
        frame[columns].to_markdown(index=False),
        "",
        "## Frozen synthesis",
        "",
        f"- Central model: {summary['central_model']}",
        f"- Subtype verdict: {summary['subtype_verdict']}",
        f"- Causal verdict: {summary['causal_verdict']}",
        "",
    ]
    (OUT / "mechanism_evidence_matrix.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

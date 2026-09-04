"""Consolidate BioAware algorithm evidence and its safe MTBLS13729 biology role."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/validation"
MTBLS = ROOT / "data/mtbls13729"
OUT = MTBLS / "bioaware_algorithm_biology_bridge_v1"

INPUTS = {
    "internal_v3": VAL / "bioaware_metdna3_internal_rplc_frozen_v3_result_v1/report.json",
    "internal_audit": VAL / "bioaware_metdna3_internal_rplc_frozen_v3_audit_v1/report.json",
    "external_v4": VAL / "bioaware_v4_external_7panel_summary_local_v2_20260901/report.json",
    "external_v6": VAL / "bioaware_v6_external_5panel_summary_local_v2_20260901/report.json",
    "failure": VAL / "bioaware_metdna3_failure_decomposition_v1/report.json",
    "headroom": VAL / "bioaware_10pp_headroom_v1/report.json",
    "candidate_headroom": VAL / "bioaware_candidate_specific_headroom_v1/report.json",
    "reproducibility": VAL / "metabolic_network_framework_reproducibility_local_20260901.json",
    "mtbls_v1": MTBLS / "bioaware_v1_eval/report.json",
    "family": MTBLS / "source_absent_family_readiness_v1/report.json",
    "manuscript": MTBLS / "manuscript_evidence_package_v3/report.json",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load_json(path) for name, path in INPUTS.items()}

    internal = data["internal_v3"]
    v4 = data["external_v4"]
    v6 = data["external_v6"]
    failure = data["failure"]
    reproducibility = data["reproducibility"]
    v1 = data["mtbls_v1"]

    required_formal = (internal, v4, v6, failure, reproducibility, v1)
    if not all(item.get("formal") is True for item in required_formal):
        raise RuntimeError("every primary BioAware audit input must be formal")
    if internal.get("contracts", {}).get("fit_performed") is not False:
        raise RuntimeError("internal V3 must be a frozen zero-refit evaluation")
    if v6["gates"].get("formula_cluster_ci_positive") is not False:
        raise RuntimeError("V6 external result unexpectedly became statistically positive")
    if reproducibility["metdna3"].get("exact_reproduction_available") is not False:
        raise RuntimeError("MetDNA3 exact reproduction boundary changed")

    benchmark_rows = [
        {
            "evaluation": "MTBLS13729 early one-hop Rhea pilot",
            "artifact_version": "BioAware v1",
            "queries": v1["real_network"]["n_queries"],
            "baseline_recall1": v1["real_network"]["baseline_recall1"],
            "method_recall1": v1["real_network"]["bioaware_recall1"],
            "delta_recall1": v1["real_network"]["delta_recall1"],
            "corrected": v1["real_network"]["corrected"],
            "introduced": v1["real_network"]["introduced"],
            "ci_low": v1["real_network"]["cluster_bootstrap"]["ci_low"],
            "ci_high": v1["real_network"]["cluster_bootstrap"]["ci_high"],
            "interpretation": "failed pilot; demonstrates that one-hop reaction adjacency is not identity evidence",
        },
        {
            "evaluation": "internal RPLC frozen zero-refit",
            "artifact_version": "BioAware V3",
            "queries": internal["pooled"]["queries"],
            "baseline_recall1": internal["pooled"]["baseline_recall1"],
            "method_recall1": internal["pooled"]["router_recall1"],
            "delta_recall1": internal["pooled"]["delta_recall1"],
            "corrected": internal["pooled"]["corrected"],
            "introduced": internal["pooled"]["introduced"],
            "ci_low": internal["formula_cluster_bootstrap"]["ci_low"],
            "ci_high": internal["formula_cluster_bootstrap"]["ci_high"],
            "interpretation": "positive small-panel signal; confidence interval crosses zero",
        },
        {
            "evaluation": "independent seven-panel external",
            "artifact_version": "BioAware V4",
            "queries": v4["queries"],
            "baseline_recall1": v4["baseline_recall1"],
            "method_recall1": v4["recall1"],
            "delta_recall1": v4["delta_recall1"],
            "corrected": v4["corrected"],
            "introduced": v4["introduced"],
            "ci_low": v4["formula_cluster_bootstrap"]["ci_low"],
            "ci_high": v4["formula_cluster_bootstrap"]["ci_high"],
            "interpretation": "negative external result; prevents selective reporting of later versions",
        },
        {
            "evaluation": "untouched five-panel external",
            "artifact_version": "BioAware V6",
            "queries": v6["queries"],
            "baseline_recall1": v6["baseline_recall1"],
            "method_recall1": v6["recall1"],
            "delta_recall1": v6["delta_recall1"],
            "corrected": v6["corrected"],
            "introduced": v6["introduced"],
            "ci_low": v6["formula_cluster_bootstrap"]["ci_low"],
            "ci_high": v6["formula_cluster_bootstrap"]["ci_high"],
            "interpretation": "conservative positive direction without statistical confirmation",
        },
    ]
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(OUT / "bioaware_benchmark_ledger.csv", index=False)

    bottlenecks = failure["primary_bottlenecks"]
    bottleneck_rows = [
        {
            "bottleneck": key,
            "queries": value["queries"],
            "formulas": value["formulas"],
            "fraction_of_official_errors": value["queries"] / failure["official_dreams"]["errors"],
        }
        for key, value in bottlenecks.items()
    ]
    pd.DataFrame(bottleneck_rows).sort_values("queries", ascending=False).to_csv(
        OUT / "bioaware_failure_decomposition.csv", index=False
    )

    application_rows = [
        {
            "application_object": "exact candidate identity",
            "allowed_role": "abstain unless spectral, RT and orthogonal identity evidence agree",
            "current_evidence": "external retrieval gain is not statistically confirmed",
            "claim_ceiling": "no exact identity promotion from network proximity",
        },
        {
            "application_object": "modified-guanosine signals 1597 and 3019",
            "allowed_role": "ion-family consolidation and pathway-context generation",
            "current_evidence": "two family anchors in the frozen biology panel",
            "claim_ceiling": "modified-guanosine family; positional isomer unresolved",
        },
        {
            "application_object": "feature 1717 acetylated-polyamine signal",
            "allowed_role": "contextualize a spectrally supported family hypothesis",
            "current_evidence": "abundance and raw-MS2 are primary; network context is secondary",
            "claim_ceiling": "acetylated-polyamine family; no N1,N8 positional identity",
        },
        {
            "application_object": "Neu5Ac mechanism module",
            "allowed_role": "rank competing biological interpretations after Level-1 identity is fixed",
            "current_evidence": "AGR2/SLC35A1 context supports capacity, not flux",
            "claim_ceiling": "mechanism discrimination only; no source enzyme or causal flux",
        },
    ]
    pd.DataFrame(application_rows).to_csv(OUT / "mtbls13729_bioaware_role_ledger.csv", index=False)

    comparison_rows = [
        {
            "criterion": "frozen held-out identity benchmark",
            "metdna3_style_expectation": "hidden Level-1 standards evaluated on matched candidate protocols",
            "ours": "internal V3 plus version-specific external panels",
            "status": "partial",
        },
        {
            "criterion": "statistically confirmed external gain",
            "metdna3_style_expectation": "positive held-out gain with uncertainty and matched baselines",
            "ours": f"V6 {pct(v6['delta_recall1'])}, CI [{pct(v6['formula_cluster_bootstrap']['ci_low'])}, {pct(v6['formula_cluster_bootstrap']['ci_high'])}]",
            "status": "not met",
        },
        {
            "criterion": "complete public reference implementation",
            "metdna3_style_expectation": "code plus network assets and runnable workdir",
            "ours": "MetDNA3 core code present; official MRN assets/workdir absent; MetDNA2 fallback complete",
            "status": "not met for MetDNA3",
        },
        {
            "criterion": "new exact metabolite validation",
            "metdna3_style_expectation": "recurrence plus orthogonal prediction followed by synthesized/authentic standard RT+MS2",
            "ours": "three source-absent family modules; zero exact new-metabolite claims",
            "status": "not met",
        },
        {
            "criterion": "biology application without identity inflation",
            "metdna3_style_expectation": "network propagates evidence but does not replace structural validation",
            "ours": "BioAware restricted to family/context evidence and explicit abstention",
            "status": "met",
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(OUT / "frontier_method_gap_matrix.csv", index=False)

    report = {
        "status": "bioaware_algorithm_biology_bridge_audit_complete",
        "formal": True,
        "algorithm_verdict": {
            "internal_v3_delta_recall1": internal["pooled"]["delta_recall1"],
            "internal_v3_corrected_introduced": [internal["pooled"]["corrected"], internal["pooled"]["introduced"]],
            "external_v4_delta_recall1": v4["delta_recall1"],
            "external_v4_corrected_introduced": [v4["corrected"], v4["introduced"]],
            "external_v6_delta_recall1": v6["delta_recall1"],
            "external_v6_corrected_introduced": [v6["corrected"], v6["introduced"]],
            "external_v6_formula_ci": [
                v6["formula_cluster_bootstrap"]["ci_low"],
                v6["formula_cluster_bootstrap"]["ci_high"],
            ],
            "statistically_confirmed_external_gain": False,
            "sota_claim_allowed": False,
        },
        "primary_failure_bottleneck": {
            "name": "truth_absent_from_Rhea",
            "queries": bottlenecks["A_truth_absent_from_rhea"]["queries"],
            "official_errors": failure["official_dreams"]["errors"],
            "fraction": bottlenecks["A_truth_absent_from_rhea"]["queries"] / failure["official_dreams"]["errors"],
        },
        "headroom_boundary": {
            "current_evidence_actual_union": data["headroom"]["actual_union"]["unique_errors_corrected_by_at_least_one_current_rule"],
            "candidate_specific_optimistic_union": data["candidate_headroom"]["combined_actual_unique_headroom"],
            "deployable_gain": False,
            "reason": "candidate-specific union uses truth after frozen evidence and unsafe overrides",
        },
        "mtbls13729_role": {
            "exact_identity_promotions": 0,
            "family_context_modules": 2,
            "mechanism_context_only": True,
            "safe_use": "family consolidation, candidate abstention and mechanism-hypothesis discrimination",
        },
        "reproducibility": {
            "metdna3_exact": False,
            "metdna2_fallback_ready": reproducibility["kgmn_metdna2"]["reproducible_baseline_ready"],
        },
        "decision": "retain BioAware as a conservative context/family expert; do not market it as a statistically confirmed annotation upgrade",
        "claim_limit": "BioAware may organize orthogonal evidence and biological context. It cannot promote a network neighbor to an exact metabolite identity, establish SOTA, or prove pathway flux or causality.",
        "provenance": {name: sha256(path) for name, path in INPUTS.items()},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# BioAware algorithm-to-biology bridge audit

## Bottom line

BioAware is **not yet a statistically confirmed annotation-performance upgrade**. The frozen internal RPLC panel improved by {pct(internal['pooled']['delta_recall1'])} ({internal['pooled']['corrected']} corrected, {internal['pooled']['introduced']} introduced), but its formula-cluster interval crossed zero. An independent V4 seven-panel evaluation was negative ({pct(v4['delta_recall1'])}; {v4['corrected']}/{v4['introduced']}), while the later frozen V6 five-panel evaluation was slightly positive ({pct(v6['delta_recall1'])}; {v6['corrected']}/{v6['introduced']}) and also non-significant. These version-specific results must be reported together.

## What the failure decomposition says

The largest limitation is not a router threshold: {bottlenecks['A_truth_absent_from_rhea']['queries']} of {failure['official_dreams']['errors']} development errors have the truth absent from Rhea. Another {bottlenecks['F_raw_ms2_edge_favors_wrong_or_ties']['queries']} errors have raw network MS2 evidence that favors the wrong candidate or ties. Therefore repeatedly tuning one-hop propagation cannot solve the dominant error space.

## What BioAware contributes to MTBLS13729

BioAware remains useful in a narrower, scientifically defensible role:

- consolidate features 1597 and 3019 as a modified-guanosine **family** module;
- contextualize feature 1717 after raw-MS2 and paired-abundance evidence, without naming its positional isomer;
- discriminate competing Neu5Ac biological interpretations after the same-cohort Level-1 identity is already fixed;
- abstain when network, spectral and RT evidence conflict.

It contributes zero exact identity promotions in the frozen biology panel. Phenotype is forbidden from identity ranking.

## Frontier comparison

MetDNA3-style studies validate propagation by hiding known Level-1 standards and ultimately confirm selected new structures with authentic or synthesized standards. We currently meet the evidence-calibration and abstention discipline, but we do not have statistically confirmed external retrieval gain, exact public MetDNA3 reproduction assets, or a newly standard-confirmed metabolite. Public MetDNA2/KGMN assets are complete and form the reproducible network baseline.

## Manuscript-safe sentence

“A phenotype-blind BioAware expert provided conservative ion-family and pathway-context evidence. Across five untouched external panels it changed only six of 542 queries and yielded a non-significant +0.37 percentage-point Recall@1 difference; accordingly, network evidence was used to consolidate families and reject conflicts, not to promote exact metabolite identities.”

## Hard prohibition

Do not call BioAware SOTA, do not combine internal V3 and external V4/V6 as if they were one frozen model, and do not present optimistic candidate-specific oracle headroom as deployable accuracy.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

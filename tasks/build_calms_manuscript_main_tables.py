#!/usr/bin/env python3
"""Build the two frozen CALMS manuscript tables without recomputation.

This script only reformats already generated JSON/CSV/TSV evidence.  It does
not load candidate scores, fit a model, resample a bootstrap, or select a
threshold.  By default, the diagnostic reports are read from the committed
CALMS evidence ZIP and the common-domain method reports are read from the
committed GNPS ladder deliverable.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile


PANELS = ("identity_disjoint", "formula_disjoint")
CALMS = "full23_v132"
SELECTIVE_CALMS = "calms_v132"
WSE = "weighted_spectral_entropy"
DREAMS = "official_dreams"
MS2DS = "ms2deepscore_2x_public"

EXPECTED_METHODS = {
    "weighted_spectral_entropy",
    "official_dreams",
    "noise_v1",
    "cosine_greedy",
    "modified_cosine",
    "p2b_sqrt_cosine",
    "p2b_unweighted_entropy",
    "neutral_loss_sqrt_cosine",
    "p2b_official_frozen",
    "p2b_noise_v1_frozen",
    "ms2deepscore_2x_public",
    "spec2vec_gnps_public",
    "entropy_raw_public",
    "denoising_search_public",
    "spec2vec_2026_retrained",
    "full23_v132",
    "specembedding_official",
    "msbert_official",
}

COMMON_EXPECTED = {
    "identity_disjoint": {"queries": 10_983, "formula_clusters": 4_199},
    "formula_disjoint": {"queries": 5_253, "formula_clusters": 2_207},
}
NATIVE_EXPECTED = {
    "identity_disjoint": {"queries": 10_995, "formula_clusters": 4_204},
    "formula_disjoint": {"queries": 5_261, "formula_clusters": 2_210},
}

TABLE1_FIELDS = (
    "panel",
    "axis",
    "n_queries",
    "n_formula_clusters",
    "common_domain_retention",
    "calms_recall1_pct",
    "calms_recall3_pct",
    "calms_recall5_pct",
    "calms_mrr",
    "calms_micro_auroc",
    "calms_micro_ap",
    "delta_r1_vs_dreams_pp",
    "dreams_ci_low_pp",
    "dreams_ci_high_pp",
    "error_reduction_vs_dreams_pct",
    "delta_r1_vs_wse_pp",
    "wse_ci_low_pp",
    "wse_ci_high_pp",
    "delta_r1_vs_ms2deepscore2x_pp",
    "ms2ds_ci_low_pp",
    "ms2ds_ci_high_pp",
)

TABLE2_FIELDS = (
    "panel",
    "axis",
    "n_queries",
    "n_formula_clusters",
    "calms_recall1_pct",
    "wse_recall1_pct",
    "delta_vs_wse_pp",
    "descriptive_ci_low_pp",
    "descriptive_ci_high_pp",
    "corrected",
    "introduced",
    "near_n",
    "near_delta_vs_wse_pp",
    "near_ci_low_pp",
    "near_ci_high_pp",
    "selective_recall1_at_90pct_coverage_pct",
    "selective_delta_vs_wse_at_90pct_pp",
    "selective_delta_ci_low_pp",
    "selective_delta_ci_high_pp",
    "calms_aurc",
    "wse_aurc",
    "aurc_relative_reduction_pct",
    "calms_margin_correctness_auroc",
    "wse_margin_correctness_auroc",
)

TABLE3_FIELDS = (
    "row_type",
    "dataset",
    "panel",
    "axis",
    "reference_model",
    "comparator_model",
    "comparison_label",
    "n_queries",
    "n_formula_clusters",
    "reference_recall1_pct",
    "comparator_recall1_pct",
    "delta_recall1_pp",
    "ci_low_pp",
    "ci_high_pp",
    "ci_excludes_zero",
    "corrected",
    "introduced",
    "residual_scale",
    "calibration_timing",
)

PARENT_VARIANTS = (
    "full_23_additive",
    "official_dreams_wse_additive",
    "without_chemaware_representation_group",
    "without_chemaware_v2",
    "without_p2b_msg",
)
PARENT_COMPARISONS = {
    "official_dreams_wse_additive": "full parent minus DreaMS+WSE learned Additive baseline",
    "without_chemaware_representation_group": "full parent minus no representation group",
    "without_chemaware_v2": "full parent minus no ChemAware V2",
    "without_p2b_msg": "full parent minus no P2b+MSG",
}
PARENT_DATASETS = {
    "enveda": {
        "status": "ENVEDA_UNIFIED_ADDITIVE_STRICT_REFIT_ABLATION_COMPLETE",
        "dataset": "Enveda-180 cross-condition benchmark primary panels",
        "evaluation_role": "fusion_contribution_development_consumed",
        "panels": {
            "identity_disjoint": {"queries": 25_000, "formula_clusters": 8_454},
            "formula_disjoint": {"queries": 9_435, "formula_clusters": 4_426},
        },
        "comparison_key": "full_additive_vs",
    },
    "gnps": {
        "status": "ENVEDA_ADDITIVE_ABLATION_FIXED_GNPS_DEVELOPMENT_EVALUATION_COMPLETE",
        "dataset": "GNPS Gold/Silver 10 ppm primary panels",
        "evaluation_role": "development_consumed_fixed_contribution_comparison",
        "panels": NATIVE_EXPECTED,
        "comparison_key": "fixed_full_additive_vs",
    },
}


class ContractError(RuntimeError):
    """Raised when a frozen input no longer matches the manuscript contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not numeric: {value!r}") from exc


def _int(value: Any, label: str) -> int:
    number = _float(value, label)
    integer = int(number)
    _require(number == integer, f"{label} is not an integer: {value!r}")
    return integer


def _pct(value: Any, label: str) -> float:
    return 100.0 * _float(value, label)


def _index_unique(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str], label: str
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    indexed: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        _require(key not in indexed, f"duplicate {label} row: {key}")
        indexed[key] = row
    return indexed


def validate_common_report(
    report: Mapping[str, Any], panel: str
) -> Mapping[str, Any]:
    expected = COMMON_EXPECTED[panel]
    _require(report.get("status") == "complete", f"{panel}: report not complete")
    _require(report.get("evaluation_role") == "development", f"{panel}: role drift")
    _require(report.get("truth_status") == "consumed", f"{panel}: truth status drift")
    _require(report.get("allow_final_claim") is False, f"{panel}: final-claim flag drift")
    _require(report.get("closed_set_identity_contract") is True, f"{panel}: not closed set")
    _require(report.get("open_set_claim") is False, f"{panel}: open-set claim drift")
    _require(set(report.get("methods", {})) == EXPECTED_METHODS, f"{panel}: method set drift")
    panel_meta = report.get("panel", {})
    _require(_int(panel_meta.get("N_Q"), f"{panel}.N_Q") == expected["queries"], f"{panel}: query denominator drift")
    _require(_int(panel_meta.get("N_F"), f"{panel}.N_F") == expected["formula_clusters"], f"{panel}: formula-cluster denominator drift")
    calms = report["methods"].get(CALMS)
    _require(isinstance(calms, Mapping), f"{panel}: missing CALMS method")
    _require(
        _int(calms.get("retrieval", {}).get("queries"), f"{panel}.CALMS queries")
        == expected["queries"],
        f"{panel}: CALMS method denominator drift",
    )
    return calms


def build_table1(
    common_reports: Mapping[str, Mapping[str, Any]],
    paired_summary: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(set(common_reports) == set(PANELS), "common-domain panel set drift")
    _require(paired_summary.get("status") == "CALMS_PAIRED_SUPERIORITY_SUMMARY_COMPLETE", "paired summary status drift")
    _require(paired_summary.get("internal_method_id") == CALMS, "paired CALMS id drift")
    _require(_int(paired_summary.get("methods_total_per_panel"), "methods_total_per_panel") == 18, "paired method count drift")
    _require(_int(paired_summary.get("competitors_per_panel"), "competitors_per_panel") == 17, "paired competitor count drift")
    summary_reports = paired_summary.get("reports", {})
    _require(set(summary_reports) == set(PANELS), "paired summary panel set drift")
    paired = _index_unique(paired_rows, ("panel", "method_id"), "paired")
    _require(len(paired) == 34, "paired table must contain exactly 17 competitors x 2 panels")

    output: list[dict[str, Any]] = []
    for panel in PANELS:
        expected = COMMON_EXPECTED[panel]
        calms = validate_common_report(common_reports[panel], panel)
        meta = summary_reports[panel]
        _require(_int(meta.get("queries"), f"{panel}.paired queries") == expected["queries"], f"{panel}: paired query denominator drift")
        _require(_int(meta.get("formula_clusters"), f"{panel}.paired clusters") == expected["formula_clusters"], f"{panel}: paired cluster denominator drift")
        full_panel_queries = _int(meta.get("full_panel_queries"), f"{panel}.full panel queries")
        _require(full_panel_queries == NATIVE_EXPECTED[panel]["queries"], f"{panel}: full-panel denominator drift")
        retention = _float(meta.get("common_domain_retention"), f"{panel}.common-domain retention")
        _require(abs(retention - expected["queries"] / full_panel_queries) < 1e-15, f"{panel}: common-domain retention drift")
        _require(meta.get("reference_method") == CALMS, f"{panel}: paired reference drift")
        _require(meta.get("evaluation_role") == "development", f"{panel}: paired role drift")
        _require(meta.get("truth_status") == "consumed", f"{panel}: paired truth status drift")

        retrieval = calms["retrieval"]
        micro = calms["micro_candidate"]
        comparisons: dict[str, Mapping[str, Any]] = {}
        for method in (DREAMS, WSE, MS2DS):
            key = (panel, method)
            _require(key in paired, f"{panel}: missing paired comparator {method}")
            row = paired[key]
            _require(_int(row["queries"], f"{panel}.{method}.queries") == expected["queries"], f"{panel}/{method}: denominator drift")
            _require(abs(_float(row["calms_recall_at_1"], "paired CALMS R1") - _float(retrieval["recall@1"], "report CALMS R1")) < 1e-12, f"{panel}/{method}: CALMS R1 alignment drift")
            comparisons[method] = row

        dreams_recall = _float(comparisons[DREAMS]["competitor_recall_at_1"], "DreaMS recall@1")
        _require(0.0 <= dreams_recall < 1.0, f"{panel}: invalid DreaMS recall@1")
        calms_recall = _float(retrieval["recall@1"], "CALMS recall@1")
        error_reduction_vs_dreams = 100.0 * (calms_recall - dreams_recall) / (1.0 - dreams_recall)

        output.append(
            {
                "panel": panel,
                "axis": "18_method_common_domain",
                "n_queries": expected["queries"],
                "n_formula_clusters": expected["formula_clusters"],
                "common_domain_retention": retention,
                "calms_recall1_pct": _pct(retrieval["recall@1"], "CALMS recall@1"),
                "calms_recall3_pct": _pct(retrieval["recall@3"], "CALMS recall@3"),
                "calms_recall5_pct": _pct(retrieval["recall@5"], "CALMS recall@5"),
                "calms_mrr": _float(retrieval["mrr"], "CALMS MRR"),
                "calms_micro_auroc": _float(micro["auroc"], "CALMS micro AUROC"),
                "calms_micro_ap": _float(micro["average_precision"], "CALMS micro AP"),
                "delta_r1_vs_dreams_pp": _float(comparisons[DREAMS]["calms_advantage_pp"], "DreaMS delta"),
                "dreams_ci_low_pp": _float(comparisons[DREAMS]["ci_low_pp"], "DreaMS CI low"),
                "dreams_ci_high_pp": _float(comparisons[DREAMS]["ci_high_pp"], "DreaMS CI high"),
                "error_reduction_vs_dreams_pct": error_reduction_vs_dreams,
                "delta_r1_vs_wse_pp": _float(comparisons[WSE]["calms_advantage_pp"], "WSE delta"),
                "wse_ci_low_pp": _float(comparisons[WSE]["ci_low_pp"], "WSE CI low"),
                "wse_ci_high_pp": _float(comparisons[WSE]["ci_high_pp"], "WSE CI high"),
                "delta_r1_vs_ms2deepscore2x_pp": _float(comparisons[MS2DS]["calms_advantage_pp"], "MS2DS delta"),
                "ms2ds_ci_low_pp": _float(comparisons[MS2DS]["ci_low_pp"], "MS2DS CI low"),
                "ms2ds_ci_high_pp": _float(comparisons[MS2DS]["ci_high_pp"], "MS2DS CI high"),
            }
        )
    return output


def build_table2(
    correction_report: Mapping[str, Any],
    strata_rows: Sequence[Mapping[str, Any]],
    selective_report: Mapping[str, Any],
    global_rows: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(correction_report.get("status") == "CALMS_CORRECTION_LANDSCAPE_COMPLETE", "correction report status drift")
    _require(correction_report.get("truth_status") == "consumed", "correction truth status drift")
    _require(correction_report.get("allow_model_selection") is False, "correction selection boundary drift")
    _require(correction_report.get("allow_routing") is False, "correction routing boundary drift")
    _require(selective_report.get("status") == "CALMS_CLOSED_SET_SELECTIVE_RANKING_COMPLETE", "selective report status drift")
    boundary = selective_report.get("closed_set_boundary", {})
    _require(boundary.get("true_candidate_present_for_every_query") is True, "selective true-candidate contract drift")
    _require(boundary.get("open_set_or_no_match_evaluated") is False, "selective open-set boundary drift")

    strata = _index_unique(strata_rows, ("panel", "baseline", "stratum_dimension", "stratum"), "stratum")
    global_index = _index_unique(global_rows, ("panel", "method"), "global metric")
    operating = _index_unique(operating_rows, ("panel", "method", "target_coverage"), "operating point")
    deltas = _index_unique(delta_rows, ("panel", "candidate", "comparator", "target_coverage"), "functional delta")

    output: list[dict[str, Any]] = []
    for panel in PANELS:
        expected = NATIVE_EXPECTED[panel]
        correction_panel = correction_report.get("panels", {}).get(panel, {})
        selective_panel = selective_report.get("dataset", {}).get("panels", {}).get(panel, {})
        for source, label in ((correction_panel, "correction"), (selective_panel, "selective")):
            _require(_int(source.get("queries"), f"{panel}.{label}.queries") == expected["queries"], f"{panel}: {label} query denominator drift")
        _require(_int(selective_panel.get("formula_clusters"), f"{panel}.selective clusters") == expected["formula_clusters"], f"{panel}: selective cluster denominator drift")
        _require(set(selective_panel.get("methods", [])) == {SELECTIVE_CALMS, WSE}, f"{panel}: selective method set drift")

        global_correction = correction_panel.get("baselines", {}).get(WSE, {}).get("global")
        _require(isinstance(global_correction, Mapping), f"{panel}: missing WSE correction record")
        _require(_int(global_correction.get("queries"), f"{panel}.correction global queries") == expected["queries"], f"{panel}: correction global denominator drift")
        _require(_int(global_correction.get("formula_clusters"), f"{panel}.correction clusters") == expected["formula_clusters"], f"{panel}: correction cluster denominator drift")

        near_key = (panel, WSE, "near", "near")
        _require(near_key in strata, f"{panel}: missing near/WSE stratum")
        near = strata[near_key]
        calms_global = global_index.get((panel, SELECTIVE_CALMS))
        wse_global = global_index.get((panel, WSE))
        _require(calms_global is not None and wse_global is not None, f"{panel}: missing global selective methods")
        for method, row in ((SELECTIVE_CALMS, calms_global), (WSE, wse_global)):
            _require(_int(row["queries"], f"{panel}.{method}.global queries") == expected["queries"], f"{panel}/{method}: selective denominator drift")

        coverage = "0.9"
        op = operating.get((panel, SELECTIVE_CALMS, coverage))
        delta = deltas.get((panel, SELECTIVE_CALMS, WSE, coverage))
        _require(op is not None, f"{panel}: missing CALMS 90% operating point")
        _require(delta is not None, f"{panel}: missing CALMS-WSE 90% delta")
        _require(abs(_float(op["target_coverage"], "target coverage") - 0.9) < 1e-12, f"{panel}: coverage drift")

        calms_aurc = _float(calms_global["aurc"], f"{panel}.CALMS AURC")
        wse_aurc = _float(wse_global["aurc"], f"{panel}.WSE AURC")
        _require(wse_aurc > 0.0, f"{panel}: WSE AURC must be positive")
        output.append(
            {
                "panel": panel,
                "axis": "native_full_panel",
                "n_queries": expected["queries"],
                "n_formula_clusters": expected["formula_clusters"],
                "calms_recall1_pct": _pct(global_correction["calms_recall1"], "CALMS R1"),
                "wse_recall1_pct": _pct(global_correction["baseline_recall1"], "WSE R1"),
                "delta_vs_wse_pp": _float(global_correction["delta_recall1_pp"], "WSE delta"),
                "descriptive_ci_low_pp": _float(global_correction["formula_ci_low_pp"], "global CI low"),
                "descriptive_ci_high_pp": _float(global_correction["formula_ci_high_pp"], "global CI high"),
                "corrected": _int(global_correction["corrected"], "corrected"),
                "introduced": _int(global_correction["introduced"], "introduced"),
                "near_n": _int(near["queries"], "near n"),
                "near_delta_vs_wse_pp": _float(near["delta_recall1_pp"], "near delta"),
                "near_ci_low_pp": _float(near["formula_ci_low_pp"], "near CI low"),
                "near_ci_high_pp": _float(near["formula_ci_high_pp"], "near CI high"),
                "selective_recall1_at_90pct_coverage_pct": _pct(op["selective_recall_at_1"], "selective R1"),
                "selective_delta_vs_wse_at_90pct_pp": _float(delta["candidate_minus_comparator_pp"], "selective delta"),
                "selective_delta_ci_low_pp": _float(delta["formula_cluster_ci_low_pp"], "selective CI low"),
                "selective_delta_ci_high_pp": _float(delta["formula_cluster_ci_high_pp"], "selective CI high"),
                "calms_aurc": calms_aurc,
                "wse_aurc": wse_aurc,
                "aurc_relative_reduction_pct": 100.0 * (wse_aurc - calms_aurc) / wse_aurc,
                "calms_margin_correctness_auroc": _float(calms_global["correctness_auroc_from_margin"], "CALMS margin AUROC"),
                "wse_margin_correctness_auroc": _float(wse_global["correctness_auroc_from_margin"], "WSE margin AUROC"),
            }
        )
    return output


def _validate_parent_report(
    dataset_key: str, report: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected = PARENT_DATASETS[dataset_key]
    _require(report.get("status") == expected["status"], f"{dataset_key}: parent report status drift")
    _require(report.get("dataset") == expected["dataset"], f"{dataset_key}: dataset label drift")
    _require(report.get("evaluation_role") == expected["evaluation_role"], f"{dataset_key}: evaluation role drift")
    _require(report.get("allow_final_external_claim") is False, f"{dataset_key}: final-claim flag drift")
    _require(set(report.get("variants", {})) == set(PARENT_VARIANTS), f"{dataset_key}: parent variant set drift")
    _require(set(report.get("panels", {})) == set(PANELS), f"{dataset_key}: panel set drift")
    if dataset_key == "enveda":
        _require(report.get("truth_status") == "consumed", "enveda: truth status drift")
        _require(report.get("reference_model") == "full_23_additive", "enveda: reference model drift")
        _require(tuple(report.get("variant_order", ())) == PARENT_VARIANTS, "enveda: variant order drift")
        contract = report.get("scientific_contract", {})
        _require(contract.get("dependent_reliability_summaries_recomputed_after_removal") is True, "enveda: dependent features not recomputed")
        _require(contract.get("query_constant_context_excluded_for_every_variant") is True, "enveda: query-constant context contract drift")
        _require(contract.get("same_formula_folds_optimizer_budget_and_seed_schedule") is True, "enveda: matched refit contract drift")
        _require(contract.get("full_reference_retrained") is False, "enveda: full parent reference drift")
        _require(set(contract.get("strict_refits", ())) == set(PARENT_COMPARISONS), "enveda: strict-refit set drift")
        _require(contract.get("no_hyperparameter_or_variant_scan") is True, "enveda: scan boundary drift")
    else:
        _require(report.get("parameter_fitting_on_gnps") is False, "gnps: parameter fitting boundary drift")
        _require(report.get("model_selection_in_this_evaluation") is False, "gnps: selection boundary drift")
        _require(report.get("gnps_previously_consumed_for_project_decisions") is True, "gnps: consumed-data boundary drift")
        contract = report.get("scientific_contract", {})
        _require(contract.get("fixed_enveda_models_only") is True, "gnps: models were not fixed from Enveda")
        _require(contract.get("gnps_parameter_fitting") is False, "gnps: fitting contract drift")
        _require(contract.get("gnps_model_selection_in_this_evaluation") is False, "gnps: model-selection contract drift")
        _require(contract.get("dependent_reliability_summaries_recomputed_after_removal") is True, "gnps: dependent features not recomputed")
    return expected


def build_table3(
    parent_enveda: Mapping[str, Any],
    parent_gnps: Mapping[str, Any],
    final_increment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a deliberately two-layer ablation table.

    Parent strict-refit rows and final embedding-increment rows answer different
    questions.  The function never sums them and never labels the parent
    strict-refit matrix as a final-v1.3.2 ablation.
    """

    output: list[dict[str, Any]] = []
    for dataset_key, report in (("enveda", parent_enveda), ("gnps", parent_gnps)):
        expected = _validate_parent_report(dataset_key, report)
        comparison_key = str(expected["comparison_key"])
        for panel in PANELS:
            panel_expected = expected["panels"][panel]
            panel_body = report["panels"][panel]
            n_queries = _int(panel_body.get("queries"), f"{dataset_key}.{panel}.queries")
            _require(n_queries == panel_expected["queries"], f"{dataset_key}/{panel}: query denominator drift")
            metrics = panel_body.get("metrics", {})
            _require(set(PARENT_VARIANTS).issubset(metrics), f"{dataset_key}/{panel}: parent metric set drift")
            parent_metric = metrics["full_23_additive"]
            _require(_int(parent_metric.get("queries"), f"{dataset_key}.{panel}.parent queries") == n_queries, f"{dataset_key}/{panel}: parent metric denominator drift")
            comparisons = panel_body.get(comparison_key, {})
            for comparator, interpretation in PARENT_COMPARISONS.items():
                _require(comparator in comparisons, f"{dataset_key}/{panel}: missing parent comparison {comparator}")
                comparator_metric = metrics[comparator]
                _require(_int(comparator_metric.get("queries"), f"{dataset_key}.{panel}.{comparator} queries") == n_queries, f"{dataset_key}/{panel}/{comparator}: denominator drift")
                comparison = comparisons[comparator]
                ci = comparison.get("recall1_formula_cluster_ci", {})
                n_clusters = _int(ci.get("formula_clusters"), f"{dataset_key}.{panel}.{comparator} clusters")
                _require(n_clusters == panel_expected["formula_clusters"], f"{dataset_key}/{panel}/{comparator}: cluster denominator drift")
                _require(_int(ci.get("bootstrap_resamples"), "parent bootstrap resamples") == 5000, f"{dataset_key}/{panel}/{comparator}: bootstrap count drift")
                delta_pp = _float(ci.get("delta_pp"), "parent delta")
                computed_delta = 100.0 * (
                    _float(parent_metric.get("recall@1"), "parent R1")
                    - _float(comparator_metric.get("recall@1"), "comparator R1")
                )
                _require(abs(delta_pp - computed_delta) < 1e-10, f"{dataset_key}/{panel}/{comparator}: delta alignment drift")
                ci_low = _float(ci.get("ci_low_pp"), "parent CI low")
                ci_high = _float(ci.get("ci_high_pp"), "parent CI high")
                output.append(
                    {
                        "row_type": "parent_strict_refit",
                        "dataset": dataset_key,
                        "panel": panel,
                        "axis": "strict_refit_native_panel",
                        "reference_model": "full_23_additive_parent_v1",
                        "comparator_model": comparator,
                        "comparison_label": interpretation,
                        "n_queries": n_queries,
                        "n_formula_clusters": n_clusters,
                        "reference_recall1_pct": _pct(parent_metric["recall@1"], "parent R1"),
                        "comparator_recall1_pct": _pct(comparator_metric["recall@1"], "comparator R1"),
                        "delta_recall1_pp": delta_pp,
                        "ci_low_pp": ci_low,
                        "ci_high_pp": ci_high,
                        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                        "corrected": _int(comparison.get("corrected"), "parent corrected"),
                        "introduced": _int(comparison.get("introduced"), "parent introduced"),
                        "residual_scale": "",
                        "calibration_timing": (
                            "Enveda formula-OOF strict-refit meta-development"
                            if dataset_key == "enveda"
                            else "fixed Enveda-fitted parent transferred to consumed GNPS development"
                        ),
                    }
                )

    _require(final_increment.get("status") == "FULL23_V132_RESIDUAL_SCALE_DEVELOPMENT_COMPLETE", "final increment status drift")
    method = final_increment.get("method", {})
    _require(method.get("formula") == "Full23_v1 + 0.9 * (Full23_v1.3.1 - Full23_v1)", "final formula drift")
    residual_scale = _float(method.get("residual_scale"), "residual scale")
    _require(residual_scale == 0.9, "final residual scale drift")
    _require(method.get("network_refit") is False, "final increment unexpectedly refit network")
    _require(method.get("upstream_checkpoints_changed") is False, "final increment changed upstream checkpoints")
    dataset = final_increment.get("dataset", {})
    _require(dataset.get("name") == "GNPS Gold/Silver", "final increment dataset drift")
    _require(dataset.get("role") == "consumed development and post-result amplitude calibration", "final increment timing drift")
    _require(set(dataset.get("queries", {})) == set(PANELS), "final increment panel set drift")
    disclosure = str(final_increment.get("selection_disclosure", ""))
    _require("after inspection" in disclosure and "not external confirmation" in disclosure, "final increment post-result disclosure drift")

    final_metrics = final_increment.get("metrics", {})
    comparisons = final_increment.get("comparisons", {})
    _require(set(final_metrics) == set(PANELS), "final increment metric panel drift")
    _require(set(comparisons) == set(PANELS), "final increment comparison panel drift")
    for panel in PANELS:
        n_queries = NATIVE_EXPECTED[panel]["queries"]
        n_clusters = NATIVE_EXPECTED[panel]["formula_clusters"]
        _require(_int(dataset["queries"][panel], f"final.{panel}.queries") == n_queries, f"final/{panel}: query denominator drift")
        metrics = final_metrics[panel]
        _require(set(metrics) == {"full23_v1", "full23_v131", "full23_v132"}, f"final/{panel}: method set drift")
        parent_metric = metrics["full23_v1"]
        final_metric = metrics["full23_v132"]
        for name, body in metrics.items():
            _require(_int(body.get("queries"), f"final.{panel}.{name}.queries") == n_queries, f"final/{panel}/{name}: denominator drift")
        comparison_set = comparisons[panel]
        _require(set(comparison_set) == {"full23_v132_vs_full23_v131", "full23_v132_vs_full23_v1"}, f"final/{panel}: comparison set drift")
        comparison = comparison_set["full23_v132_vs_full23_v1"]
        ci = comparison.get("recall1_formula_cluster_ci", {})
        _require(_int(ci.get("formula_clusters"), "final clusters") == n_clusters, f"final/{panel}: cluster denominator drift")
        _require(_int(ci.get("bootstrap_resamples"), "final bootstrap resamples") == 5000, f"final/{panel}: bootstrap count drift")
        delta_pp = _float(ci.get("delta_pp"), "final delta")
        computed_delta = 100.0 * (
            _float(final_metric.get("recall@1"), "final R1")
            - _float(parent_metric.get("recall@1"), "parent R1")
        )
        _require(abs(delta_pp - computed_delta) < 1e-10, f"final/{panel}: delta alignment drift")
        ci_low = _float(ci.get("ci_low_pp"), "final CI low")
        ci_high = _float(ci.get("ci_high_pp"), "final CI high")
        _require(ci_low <= 0.0 <= ci_high, f"final/{panel}: expected development CI to cross zero")
        output.append(
            {
                "row_type": "final_embedding_increment",
                "dataset": "gnps",
                "panel": panel,
                "axis": "native_full_panel",
                "reference_model": "full23_v132_final",
                "comparator_model": "full23_v1_parent",
                "comparison_label": "final 0.9-scaled embedding correction minus Full23 v1 parent",
                "n_queries": n_queries,
                "n_formula_clusters": n_clusters,
                "reference_recall1_pct": _pct(final_metric["recall@1"], "final R1"),
                "comparator_recall1_pct": _pct(parent_metric["recall@1"], "parent R1"),
                "delta_recall1_pp": delta_pp,
                "ci_low_pp": ci_low,
                "ci_high_pp": ci_high,
                "ci_excludes_zero": False,
                "corrected": _int(comparison.get("corrected"), "final corrected"),
                "introduced": _int(comparison.get("introduced"), "final introduced"),
                "residual_scale": residual_scale,
                "calibration_timing": "0.9 selected post-result after inspecting consumed GNPS development predictions",
            }
        )

    _require(len(output) == 18, "two-layer ablation table must contain 16 parent rows and 2 final rows")
    return output


def statistical_notes() -> str:
    return """# Statistical notes for CALMS manuscript main tables

## Frozen method and evidence role

- CALMS v1.0 is the frozen released method (internal identifier `full23_v132`,
  internal version Full23 v1.3.2). No model was fitted and no bootstrap was
  rerun while producing these tables.
- GNPS Gold/Silver is **consumed development evidence**. It was used during
  CALMS development, including residual-amplitude selection. These tables are
  not unopened external confirmation and do not by themselves establish a
  field-wide SOTA claim.

## Two axes that must not be mixed

- Table 1 uses the strict 18-method common-availability axis: 10,983
  identity-disjoint queries (4,199 formula clusters) and 5,253
  formula-disjoint queries (2,207 formula clusters). This retains
  10,983/10,995 and 5,253/5,261 queries from the corresponding full panels.
- Table 2 uses each native full panel: 10,995 identity-disjoint queries (4,204
  formula clusters) and 5,261 formula-disjoint queries (2,210 formula
  clusters). Values can differ slightly from Table 1 because the axes differ.
- The two panels are not independent: 5,237 of 5,261 formula-panel query rows
  also occur in the identity panel. Query counts, corrected/introduced counts,
  and apparent discoveries must never be added across panels.

## Intervals and multiplicity

- Table 1 Recall@1 intervals use formula-cluster bootstrap resampling and are
  familywise-adjusted over 20 endpoints **within one CALMS-versus-one-
  comparator analysis**. They are not simultaneous over all 17 competitors,
  over both panels, or over prior CALMS development and scale selection.
- Table 2 global and near-subset intervals are descriptive 95% formula-cluster
  bootstrap intervals (10,000 resamples, seed 20261105) with no multiplicity
  adjustment. `corrected` and `introduced` are oriented from WSE to CALMS.
- The 90%-coverage interval is a paired functional bootstrap delta. Each
  method selects its own highest-confidence prefix at matched coverage under
  the same formula-cluster draw; the accepted query sets need not be identical.
- AURC relative reduction is descriptive. No bootstrap interval was computed
  for the AURC difference.

## Closed-set and mechanism boundaries

- Every query has its true candidate in the candidate block. The selective
  analysis is closed-set and does not evaluate no-match detection, open-set
  false acceptance, FDR control, calibrated match probability, or a deployable
  abstention threshold.
- Confidence is the frozen Top1-Top2 candidate-score gap. Margin correctness
  AUROC measures confidence ordering; it is distinct from the candidate-level
  micro AUROC in Table 1.
- The `near` subset uses truth/structure metadata. It is a post-freeze
  diagnostic subset, not an observable deployment router.
- All percentages in the CSVs are percentage values (for example, 88.4), and
  all `*_pp` fields are percentage-point differences.
- `common_domain_retention` is a fraction of the native full-panel query count,
  not a percentage.

## Table 3 is deliberately two-layered

- `parent_strict_refit` rows test the Full23 v1 **candidate-evidence parent**.
  Each reduced parent was strictly refitted with the same low-capacity Enveda
  formula-OOF protocol and dependent reliability summaries recomputed after
  feature removal. Full-minus-reduced differences estimate grouped evidence
  contributions under that fixed protocol; they do not isolate a single
  module's causal effect.
- The four parent comparisons are: full parent minus the DreaMS+WSE learned
  Additive baseline, minus the representation-group ablation, minus the
  ChemAware V2 ablation, and minus the P2b+MSG ablation. Enveda rows are
  consumed meta-development results; GNPS rows are transfer of the fixed
  Enveda-fitted models to already consumed development data.
- `final_embedding_increment` rows answer a different question: frozen final
  Full23 v1.3.2 minus its Full23 v1 parent on GNPS. The correction is
  `Full23_v1 + 0.9 * (Full23_v1.3.1 - Full23_v1)`. The 0.9 scale was selected
  **post-result after inspection of consumed GNPS development predictions**.
  Both formula-cluster intervals cross zero.
- Parent and final-increment rows must not be added. Table 3 is not a complete
  strict-refit ablation of final CALMS v1.0/v1.3.2, and it cannot assign the
  final model's total gain to individual modules.
"""


class ZipEvidence:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.zip = ZipFile(path)

    def close(self) -> None:
        self.zip.close()

    def _name(self, suffix: str) -> str:
        matches = [name for name in self.zip.namelist() if name.endswith(suffix)]
        _require(len(matches) == 1, f"evidence ZIP expected one {suffix!r}, found {len(matches)}")
        return matches[0]

    def json(self, suffix: str) -> Mapping[str, Any]:
        return json.loads(self.zip.read(self._name(suffix)).decode("utf-8"))

    def rows(self, suffix: str, delimiter: str) -> list[dict[str, str]]:
        text = self.zip.read(self._name(suffix)).decode("utf-8")
        return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


class DirectoryEvidence:
    """Read evidence from an unpacked compact bundle by unique suffix."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def close(self) -> None:
        return None

    def _path(self, suffix: str) -> Path:
        normalized = suffix.replace("\\", "/")
        matches = [
            path
            for path in self.root.rglob(Path(normalized).name)
            if path.is_file()
            and path.relative_to(self.root).as_posix().endswith(normalized)
        ]
        _require(
            len(matches) == 1,
            f"evidence directory expected one {suffix!r}, found {len(matches)}",
        )
        return matches[0]

    def json(self, suffix: str) -> Mapping[str, Any]:
        return json.loads(self._path(suffix).read_text(encoding="utf-8"))

    def rows(self, suffix: str, delimiter: str) -> list[dict[str, str]]:
        with self._path(suffix).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-bundle",
        type=Path,
        default=Path("deliverables/CALMS_v1.0_evidence_bundle_20261105.zip"),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help=(
            "Read compact diagnostic evidence from an unpacked evidence-bundle "
            "root instead of --evidence-bundle."
        ),
    )
    parser.add_argument(
        "--common-identity-report",
        type=Path,
        default=Path("deliverables/mega_sota_ladder/gnps_run18_msbert_common_20261102/identity/report.json"),
    )
    parser.add_argument(
        "--common-formula-report",
        type=Path,
        default=Path("deliverables/mega_sota_ladder/gnps_run18_msbert_common_20261102/formula/report.json"),
    )
    parser.add_argument(
        "--parent-enveda-report",
        type=Path,
        default=Path("deploy/unified_additive_full23_v1/evidence/strict_refit_enveda_report.json"),
    )
    parser.add_argument(
        "--parent-gnps-report",
        type=Path,
        default=Path("deploy/unified_additive_full23_v1/evidence/strict_refit_gnps_report.json"),
    )
    parser.add_argument(
        "--final-increment-report",
        type=Path,
        default=Path("deploy/unified_additive_full23_v132/evidence/gnps_development_residual_scale_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deliverables/calms_v1_manuscript_tables_20261105"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.common_identity_report,
        args.common_formula_report,
        args.parent_enveda_report,
        args.parent_gnps_report,
        args.final_increment_report,
    ]
    if args.evidence_root is None:
        required.append(args.evidence_bundle)
    elif not args.evidence_root.is_dir():
        raise NotADirectoryError(args.evidence_root)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    common_reports = {
        "identity_disjoint": json.loads(args.common_identity_report.read_text(encoding="utf-8")),
        "formula_disjoint": json.loads(args.common_formula_report.read_text(encoding="utf-8")),
    }
    evidence = (
        DirectoryEvidence(args.evidence_root)
        if args.evidence_root is not None
        else ZipEvidence(args.evidence_bundle)
    )
    try:
        table1 = build_table1(
            common_reports,
            evidence.json("calms_gnps_paired_superiority_v3_20261105/report.json"),
            evidence.rows("calms_gnps_paired_superiority_v3_20261105/calms_paired_superiority.tsv", "\t"),
        )
        table2 = build_table2(
            evidence.json("calms_correction_landscape_20261105/report.json"),
            evidence.rows("calms_correction_landscape_20261105/strata_summary.tsv", "\t"),
            evidence.json("calms_v132_closed_set_selective_ranking_20261105/report.json"),
            evidence.rows("calms_v132_closed_set_selective_ranking_20261105/global_metrics.csv", ","),
            evidence.rows("calms_v132_closed_set_selective_ranking_20261105/selective_operating_points.csv", ","),
            evidence.rows("calms_v132_closed_set_selective_ranking_20261105/paired_functional_deltas.csv", ","),
        )
    finally:
        evidence.close()

    table3 = build_table3(
        json.loads(args.parent_enveda_report.read_text(encoding="utf-8")),
        json.loads(args.parent_gnps_report.read_text(encoding="utf-8")),
        json.loads(args.final_increment_report.read_text(encoding="utf-8")),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "table1_primary_results.csv", TABLE1_FIELDS, table1)
    _write_csv(args.output / "table2_mechanism_selective.csv", TABLE2_FIELDS, table2)
    _write_csv(args.output / "table3_two_layer_ablation.csv", TABLE3_FIELDS, table3)
    (args.output / "STATISTICAL_NOTES.md").write_text(statistical_notes(), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "CALMS_MANUSCRIPT_MAIN_TABLES_COMPLETE",
                "output": str(args.output),
                "table1_rows": len(table1),
                "table2_rows": len(table2),
                "table3_rows": len(table3),
                "model_recomputed": False,
                "bootstrap_rerun": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

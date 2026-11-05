import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from build_calms_manuscript_main_tables import (
    CALMS,
    COMMON_EXPECTED,
    DREAMS,
    EXPECTED_METHODS,
    MS2DS,
    NATIVE_EXPECTED,
    PANELS,
    PARENT_COMPARISONS,
    PARENT_DATASETS,
    PARENT_VARIANTS,
    TABLE1_FIELDS,
    TABLE2_FIELDS,
    TABLE3_FIELDS,
    SELECTIVE_CALMS,
    WSE,
    ContractError,
    DirectoryEvidence,
    build_table1,
    build_table2,
    build_table3,
)


def test_directory_evidence_reads_unique_suffixes(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "validation" / "example"
    nested.mkdir(parents=True)
    (nested / "report.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    with (nested / "rows.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"name": "x", "value": "1"})

    evidence = DirectoryEvidence(tmp_path)
    assert evidence.json("example/report.json") == {"status": "ok"}
    assert evidence.rows("example/rows.tsv", "\t") == [{"name": "x", "value": "1"}]

    duplicate = tmp_path / "duplicate" / "example"
    duplicate.mkdir(parents=True)
    (duplicate / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="expected one"):
        evidence.json("example/report.json")


def common_report(panel: str) -> dict:
    expected = COMMON_EXPECTED[panel]
    methods = {name: {"retrieval": {"queries": expected["queries"]}} for name in EXPECTED_METHODS}
    methods[CALMS] = {
        "retrieval": {
            "queries": expected["queries"],
            "recall@1": 0.88,
            "recall@3": 0.96,
            "recall@5": 0.98,
            "mrr": 0.92,
        },
        "micro_candidate": {"auroc": 0.95, "average_precision": 0.84},
    }
    return {
        "status": "complete",
        "evaluation_role": "development",
        "truth_status": "consumed",
        "allow_final_claim": False,
        "closed_set_identity_contract": True,
        "open_set_claim": False,
        "panel": {"N_Q": expected["queries"], "N_F": expected["formula_clusters"]},
        "methods": methods,
    }


def paired_inputs() -> tuple[dict, list[dict]]:
    summary = {
        "status": "CALMS_PAIRED_SUPERIORITY_SUMMARY_COMPLETE",
        "internal_method_id": CALMS,
        "methods_total_per_panel": 18,
        "competitors_per_panel": 17,
        "reports": {},
    }
    rows = []
    for panel in PANELS:
        expected = COMMON_EXPECTED[panel]
        summary["reports"][panel] = {
            "queries": expected["queries"],
            "formula_clusters": expected["formula_clusters"],
            "full_panel_queries": NATIVE_EXPECTED[panel]["queries"],
            "common_domain_retention": expected["queries"] / NATIVE_EXPECTED[panel]["queries"],
            "reference_method": CALMS,
            "evaluation_role": "development",
            "truth_status": "consumed",
        }
        for method in sorted(EXPECTED_METHODS - {CALMS}):
            rows.append(
                {
                    "panel": panel,
                    "method_id": method,
                    "queries": str(expected["queries"]),
                    "calms_recall_at_1": "0.88",
                    "competitor_recall_at_1": "0.85",
                    "calms_advantage_pp": "1.25",
                    "ci_low_pp": "0.50",
                    "ci_high_pp": "2.00",
                }
            )
    return summary, rows


def test_build_table1_enforces_common_axis_and_emits_exact_schema() -> None:
    summary, paired = paired_inputs()
    reports = {panel: common_report(panel) for panel in PANELS}
    rows = build_table1(reports, summary, paired)
    assert len(rows) == 2
    assert tuple(rows[0]) == TABLE1_FIELDS
    assert rows[0]["axis"] == "18_method_common_domain"
    assert rows[0]["calms_recall1_pct"] == pytest.approx(88.0)
    assert rows[0]["delta_r1_vs_dreams_pp"] == pytest.approx(1.25)
    assert rows[0]["delta_r1_vs_wse_pp"] == pytest.approx(1.25)
    assert rows[0]["delta_r1_vs_ms2deepscore2x_pp"] == pytest.approx(1.25)
    assert rows[0]["error_reduction_vs_dreams_pct"] == pytest.approx(20.0)
    broken = {panel: common_report(panel) for panel in PANELS}
    broken["identity_disjoint"]["panel"]["N_Q"] -= 1
    with pytest.raises(ContractError, match="query denominator drift"):
        build_table1(broken, summary, paired)

    bad = deepcopy(reports)
    bad["identity_disjoint"]["panel"]["N_Q"] += 1
    with pytest.raises(ContractError, match="query denominator drift"):
        build_table1(bad, summary, paired)


def correction_and_selective_inputs() -> tuple[dict, list[dict], dict, list[dict], list[dict], list[dict]]:
    correction = {
        "status": "CALMS_CORRECTION_LANDSCAPE_COMPLETE",
        "truth_status": "consumed",
        "allow_model_selection": False,
        "allow_routing": False,
        "panels": {},
    }
    selective = {
        "status": "CALMS_CLOSED_SET_SELECTIVE_RANKING_COMPLETE",
        "closed_set_boundary": {
            "true_candidate_present_for_every_query": True,
            "open_set_or_no_match_evaluated": False,
        },
        "dataset": {"panels": {}},
    }
    strata = []
    globals_ = []
    operating = []
    deltas = []
    for panel in PANELS:
        expected = NATIVE_EXPECTED[panel]
        correction["panels"][panel] = {
            "queries": expected["queries"],
            "baselines": {
                WSE: {
                    "global": {
                        "queries": expected["queries"],
                        "formula_clusters": expected["formula_clusters"],
                        "calms_recall1": 0.89,
                        "baseline_recall1": 0.88,
                        "delta_recall1_pp": 1.0,
                        "formula_ci_low_pp": 0.4,
                        "formula_ci_high_pp": 1.6,
                        "corrected": 20,
                        "introduced": 10,
                    }
                }
            },
        }
        selective["dataset"]["panels"][panel] = {
            "queries": expected["queries"],
            "formula_clusters": expected["formula_clusters"],
            "methods": [SELECTIVE_CALMS, WSE],
        }
        strata.append(
            {
                "panel": panel,
                "baseline": WSE,
                "stratum_dimension": "near",
                "stratum": "near",
                "queries": "100",
                "delta_recall1_pp": "2.0",
                "formula_ci_low_pp": "0.8",
                "formula_ci_high_pp": "3.2",
            }
        )
        for method, aurc, auc in ((SELECTIVE_CALMS, "0.018", "0.91"), (WSE, "0.020", "0.92")):
            globals_.append(
                {
                    "panel": panel,
                    "method": method,
                    "queries": str(expected["queries"]),
                    "aurc": aurc,
                    "correctness_auroc_from_margin": auc,
                }
            )
        operating.append(
            {
                "panel": panel,
                "method": SELECTIVE_CALMS,
                "target_coverage": "0.9",
                "selective_recall_at_1": "0.93",
            }
        )
        deltas.append(
            {
                "panel": panel,
                "candidate": SELECTIVE_CALMS,
                "comparator": WSE,
                "target_coverage": "0.9",
                "candidate_minus_comparator_pp": "0.7",
                "formula_cluster_ci_low_pp": "0.2",
                "formula_cluster_ci_high_pp": "1.1",
            }
        )
    return correction, strata, selective, globals_, operating, deltas


def test_build_table2_keeps_native_axis_and_closed_set_boundary() -> None:
    inputs = correction_and_selective_inputs()
    rows = build_table2(*inputs)
    assert len(rows) == 2
    assert tuple(rows[0]) == TABLE2_FIELDS
    assert rows[0]["axis"] == "native_full_panel"
    assert rows[0]["selective_recall1_at_90pct_coverage_pct"] == pytest.approx(93.0)
    assert rows[0]["aurc_relative_reduction_pct"] == pytest.approx(10.0)
    assert rows[0]["corrected"] == 20
    assert rows[0]["introduced"] == 10

    bad = list(inputs)
    bad[2] = deepcopy(bad[2])
    bad[2]["closed_set_boundary"]["open_set_or_no_match_evaluated"] = True
    with pytest.raises(ContractError, match="open-set boundary drift"):
        build_table2(*bad)


def parent_report(dataset_key: str) -> dict:
    expected = PARENT_DATASETS[dataset_key]
    report = {
        "status": expected["status"],
        "dataset": expected["dataset"],
        "evaluation_role": expected["evaluation_role"],
        "allow_final_external_claim": False,
        "variants": {name: {} for name in PARENT_VARIANTS},
        "panels": {},
    }
    if dataset_key == "enveda":
        report.update(
            {
                "truth_status": "consumed",
                "reference_model": "full_23_additive",
                "variant_order": list(PARENT_VARIANTS),
                "scientific_contract": {
                    "dependent_reliability_summaries_recomputed_after_removal": True,
                    "query_constant_context_excluded_for_every_variant": True,
                    "same_formula_folds_optimizer_budget_and_seed_schedule": True,
                    "full_reference_retrained": False,
                    "strict_refits": list(PARENT_COMPARISONS),
                    "no_hyperparameter_or_variant_scan": True,
                },
            }
        )
    else:
        report.update(
            {
                "parameter_fitting_on_gnps": False,
                "model_selection_in_this_evaluation": False,
                "gnps_previously_consumed_for_project_decisions": True,
                "scientific_contract": {
                    "dependent_reliability_summaries_recomputed_after_removal": True,
                    "fixed_enveda_models_only": True,
                    "gnps_parameter_fitting": False,
                    "gnps_model_selection_in_this_evaluation": False,
                },
            }
        )
    for panel in PANELS:
        panel_expected = expected["panels"][panel]
        metrics = {
            name: {"queries": panel_expected["queries"], "recall@1": 0.89}
            for name in PARENT_VARIANTS
        }
        metrics["full_23_additive"]["recall@1"] = 0.90
        comparisons = {
            comparator: {
                "corrected": 20,
                "introduced": 10,
                "recall1_formula_cluster_ci": {
                    "delta_pp": 1.0,
                    "ci_low_pp": 0.2,
                    "ci_high_pp": 1.8,
                    "formula_clusters": panel_expected["formula_clusters"],
                    "bootstrap_resamples": 5000,
                },
            }
            for comparator in PARENT_COMPARISONS
        }
        report["panels"][panel] = {
            "queries": panel_expected["queries"],
            "metrics": metrics,
            expected["comparison_key"]: comparisons,
        }
    return report


def final_increment_report() -> dict:
    report = {
        "status": "FULL23_V132_RESIDUAL_SCALE_DEVELOPMENT_COMPLETE",
        "method": {
            "formula": "Full23_v1 + 0.9 * (Full23_v1.3.1 - Full23_v1)",
            "residual_scale": 0.9,
            "network_refit": False,
            "upstream_checkpoints_changed": False,
        },
        "dataset": {
            "name": "GNPS Gold/Silver",
            "role": "consumed development and post-result amplitude calibration",
            "queries": {panel: NATIVE_EXPECTED[panel]["queries"] for panel in PANELS},
        },
        "selection_disclosure": "The 0.9 scale was selected after inspection; this is not external confirmation.",
        "metrics": {},
        "comparisons": {},
    }
    for panel in PANELS:
        n = NATIVE_EXPECTED[panel]["queries"]
        clusters = NATIVE_EXPECTED[panel]["formula_clusters"]
        report["metrics"][panel] = {
            "full23_v1": {"queries": n, "recall@1": 0.880},
            "full23_v131": {"queries": n, "recall@1": 0.8805},
            "full23_v132": {"queries": n, "recall@1": 0.881},
        }
        report["comparisons"][panel] = {
            "full23_v132_vs_full23_v131": {},
            "full23_v132_vs_full23_v1": {
                "corrected": 9,
                "introduced": 4,
                "recall1_formula_cluster_ci": {
                    "delta_pp": 0.1,
                    "ci_low_pp": -0.1,
                    "ci_high_pp": 0.3,
                    "formula_clusters": clusters,
                    "bootstrap_resamples": 5000,
                },
            },
        }
    return report


def test_build_table3_keeps_parent_and_final_layers_separate() -> None:
    enveda = parent_report("enveda")
    gnps = parent_report("gnps")
    final = final_increment_report()
    rows = build_table3(enveda, gnps, final)
    assert len(rows) == 18
    assert tuple(rows[0]) == TABLE3_FIELDS
    assert sum(row["row_type"] == "parent_strict_refit" for row in rows) == 16
    final_rows = [row for row in rows if row["row_type"] == "final_embedding_increment"]
    assert len(final_rows) == 2
    assert all(row["ci_excludes_zero"] is False for row in final_rows)
    assert all(row["residual_scale"] == pytest.approx(0.9) for row in final_rows)
    broken = final_increment_report()
    broken["method"]["residual_scale"] = 1.0
    with pytest.raises(ContractError, match="residual scale drift"):
        build_table3(enveda, gnps, broken)

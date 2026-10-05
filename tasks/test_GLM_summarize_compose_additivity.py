"""Contract tests for the GLM composition additivity summarizer.

Synthetic role-2/role-3/GNPS reports exercise every pre-registered verdict
branch plus the checkpoint-distinctness integrity gate.  No torch, no data.
"""
from __future__ import annotations

import GLM_summarize_compose_additivity as summarize


def row(name: str, recall: float, delta: float = 0.0, ci_low: float = 0.0,
        corrected: int = 0, introduced: int = 0, checkpoint: str | None = None) -> dict:
    return {
        "name": name,
        "checkpoint": checkpoint or f"/ckpt/{name}.pt",
        "metrics": {"recall1": recall, "recall3": recall, "micro_auc": 0.9,
                    "macro_auc": 0.9},
        "paired_vs_base": {
            "delta_recall1": delta,
            "delta_mrr": delta,
            "corrected_at_1": corrected,
            "introduced_at_1": introduced,
            "formula_cluster_bootstrap_delta_recall1_ci95": [ci_low, 0.5],
        },
    }


def report(rows: list) -> dict:
    return {"formula_role": 2, "formula_role_contract_passed": True, "results": rows}


def base_rows(**overrides) -> dict:
    values = {
        "official": 0.8962,
        "phasea_2pp": 0.9175,
        "noise_stage1": 0.9010,
        "compose_balanced": 0.9225,
        "compose_advantage_max": 0.9205,
        "compose_control": 0.9180,
    }
    values.update(overrides)
    return values


def build(**overrides) -> dict:
    values = base_rows(**overrides)
    base = values["phasea_2pp"]
    risks = {
        "compose_balanced": (45, 10),
        "compose_advantage_max": (40, 14),
        "compose_control": (25, 24),
    }
    rows = [
        row("official", values["official"], checkpoint="/ckpt/official.pt"),
        row("phasea_2pp", values["phasea_2pp"], checkpoint="/ckpt/phasea.pt"),
        row("noise_stage1", values["noise_stage1"], checkpoint="/ckpt/noise.pt"),
    ]
    for name, (corrected, introduced) in risks.items():
        delta = values[name] - base
        rows.append(row(name, values[name], delta=delta,
                        ci_low=delta if delta > 0 else -abs(delta) / 3.0,
                        corrected=corrected, introduced=introduced))
    return report(rows)


def test_super_additive_verdict() -> None:
    result = summarize.verdicts(build(), None, None, [])
    assert result["status"] == "GLM_COMPOSE_SUPER_ADDITIVE", result["status"]
    accounting = result["accounting"]
    # composed +2.63pp > chem +2.13pp + noise +0.48pp = +2.61pp
    assert accounting["super_additive"] is True
    assert result["gates"]["compose_beats_matched_continuation_control"] is True
    assert result["gates"]["arm_paired_deltas_match_absolute_recall1"] is True


def test_additive_only_verdict() -> None:
    result = summarize.verdicts(build(compose_balanced=0.9210), None, None, [])
    assert result["status"] == "GLM_COMPOSE_ADDITIVE_ONLY", result["status"]
    assert result["accounting"]["super_additive"] is False


def test_unsafe_control_verdict_when_control_matches() -> None:
    result = summarize.verdicts(
        build(compose_balanced=0.9200, compose_control=0.9200), None, None, [])
    assert result["status"] == "GLM_COMPOSE_SIGNAL_UNSAFE_CONTROL", result["status"]
    assert result["gates"]["compose_beats_matched_continuation_control"] is False


def test_incumbent_retained_when_no_gain() -> None:
    result = summarize.verdicts(
        build(compose_balanced=0.9160, compose_advantage_max=0.9150,
              compose_control=0.9155), None, None, [])
    assert result["status"] == "GLM_COMPOSE_INCUMBENT_RETAINED", result["status"]


def test_axis_effect_requires_positive_paired_ci() -> None:
    axis_rows = [
        row("compose_balanced", 0.9225, delta=0.0020, ci_low=0.0004,
            corrected=30, introduced=12),
        row("compose_advantage_max", 0.9205, delta=0.0, ci_low=0.0),
    ]
    result = summarize.verdicts(build(), report(axis_rows), None, [])
    assert result["axis_hypothesis"]["axis_effect_established"] is True
    weak = [row("compose_balanced", 0.9225, delta=0.0005, ci_low=-0.0009),
            row("compose_advantage_max", 0.9205, delta=0.0, ci_low=0.0)]
    result_weak = summarize.verdicts(build(), report(weak), None, [])
    assert result_weak["axis_hypothesis"]["axis_effect_established"] is False


def test_role3_regression_blocks_the_win() -> None:
    role3 = report([row("compose_balanced", 0.9000, checkpoint="/ckpt/cb3.pt"),
                    row("phasea_2pp", 0.9050, checkpoint="/ckpt/p3.pt")])
    result = summarize.verdicts(build(), None, role3, [])
    assert result["gates"]["role3_non_regression"] is False
    assert result["status"] != "GLM_COMPOSE_SUPER_ADDITIVE"


def test_gnps_regression_blocks_the_win() -> None:
    gnps = [{"candidate_label": "compose_balanced",
             "identity_disjoint": {"paired_delta_recall1": {"mean": -0.002,
                                                            "ci_low": -0.005}},
             "formula_disjoint": {"paired_delta_recall1": {"mean": 0.001,
                                                           "ci_low": -0.001}}}]
    result = summarize.verdicts(build(), None, None, gnps)
    assert result["gates"]["gnps_dual_panel_non_regression"] is False
    assert result["status"] != "GLM_COMPOSE_SUPER_ADDITIVE"


def test_duplicate_checkpoint_is_detected() -> None:
    data = build()
    for entry in data["results"]:
        if entry["name"] == "compose_control":
            entry["checkpoint"] = "/ckpt/compose_balanced.pt"
    data["results"][3]["checkpoint"] = "/ckpt/compose_balanced.pt"
    result = summarize.verdicts(data, None, None, [])
    assert result["gates"]["role2_rows_are_distinct_checkpoints"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("GLM composition additivity contracts passed")

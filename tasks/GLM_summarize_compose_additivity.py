"""GLM composition additivity summarizer (pure JSON logic, no torch).

Reads the role-2 / role-3 / GNPS reports produced by the orchestrating job and
applies the pre-registered verdict rules of GLM_INTEGRATION_PREREGISTRATION:
1+1 accounting on ONE frozen universe (the role-2 selection panel), the causal
content contrast against the matched continuation control, the axis hypothesis
contrast against the un-stratified advantage-max arm, protection/non-regression
guards, and the honest data-integrity check that the compared rows really come
from distinct checkpoints.

Usage:
    python GLM_summarize_compose_additivity.py \
        --role2 <role2_checkpoint_evaluation.json> \
        --role2-axis <role2_axis_comparison.json> \
        --role3 <role3_evaluation.json> \
        --gnps <dir with *_vs_* report.json> ... \
        --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_NAME = "phasea_2pp"
OFFICIAL_NAME = "official"
NOISE_NAME = "noise_stage1"
ARMS = {
    "compose_balanced": "axis-balanced curriculum on the chemistry champion",
    "compose_advantage_max": "un-stratified advantage-max curriculum (axis control)",
    "compose_control": "matched continuation control (same relations, control spectra)",
}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_by_name(report: dict) -> dict:
    return {str(row["name"]): row for row in report["results"]}


def recall1(row: dict) -> float:
    return float(row["metrics"]["recall1"])


def paired(row: dict) -> dict:
    value = row.get("paired_vs_base")
    if value is None:
        raise KeyError(f"row {row.get('name')!r} has no paired_vs_base block")
    return value


def ci_low(row: dict) -> float:
    return float(paired(row)["formula_cluster_bootstrap_delta_recall1_ci95"][0])


def risk_utility(row: dict) -> int:
    block = paired(row)
    return int(block["corrected_at_1"]) - 2 * int(block["introduced_at_1"])


def gain_vs_official(rows: dict, name: str) -> float:
    return recall1(rows[name]) - recall1(rows[OFFICIAL_NAME])


def additivity(rows: dict) -> dict:
    chem = gain_vs_official(rows, BASE_NAME)
    noise = gain_vs_official(rows, NOISE_NAME)
    composed = gain_vs_official(rows, "compose_balanced")
    control = gain_vs_official(rows, "compose_control")
    return {
        "delta_chem_pp": chem * 100.0,
        "delta_noise_pp": noise * 100.0,
        "delta_compose_pp": composed * 100.0,
        "delta_control_pp": control * 100.0,
        "sum_of_parts_pp": (chem + noise) * 100.0,
        "additivity_residual_pp": (composed - chem - noise) * 100.0,
        "super_additive": bool(composed > chem + noise),
    }


def distinct_checkpoints(rows: dict, names: list) -> bool:
    seen = {}
    for name in names:
        checkpoint = str(rows[name]["checkpoint"])
        if checkpoint in seen:
            return False
        seen[checkpoint] = name
    return True


def paired_delta_matches_absolute(rows: dict, names: list) -> bool:
    """The paired delta of a row must equal its absolute recall@1 difference
    against the paired reference; otherwise the report mixes stale rows."""
    base = recall1(rows[BASE_NAME])
    for name in names:
        delta = float(paired(rows[name])["delta_recall1"])
        if abs(delta - (recall1(rows[name]) - base)) > 1e-9:
            return False
    return True


def verdicts(role2: dict, role2_axis: dict | None, role3: dict | None,
             gnps: list[dict]) -> dict:
    rows = rows_by_name(role2)
    missing = [n for n in (OFFICIAL_NAME, BASE_NAME, NOISE_NAME, *ARMS) if n not in rows]
    if missing:
        raise KeyError(f"role-2 report lacks required rows: {missing}")

    accounting = additivity(rows)
    balanced = rows["compose_balanced"]
    control = rows["compose_control"]
    axis_arm = rows["compose_advantage_max"]

    compose_beats_base = (float(paired(balanced)["delta_recall1"]) > 0.0
                          and ci_low(balanced) > 0.0)
    causal_content = (recall1(balanced) - recall1(control) > 0.0
                      and int(paired(balanced)["corrected_at_1"])
                      > int(paired(balanced)["introduced_at_1"]))
    axis_effect = None
    if role2_axis is not None:
        axis_rows = rows_by_name(role2_axis)
        if "compose_balanced" in axis_rows and "compose_advantage_max" in axis_rows:
            delta = float(paired(axis_rows["compose_balanced"])["delta_recall1"])
            axis_effect = bool(delta > 0.0
                               and ci_low(axis_rows["compose_balanced"]) > 0.0)

    role3_ok = None
    if role3 is not None:
        role3_rows = rows_by_name(role3)
        if "compose_balanced" in role3_rows and BASE_NAME in role3_rows:
            role3_ok = recall1(role3_rows["compose_balanced"]) >= recall1(
                role3_rows[BASE_NAME]) - 1e-12

    gnps_ok = None
    gnps_detail = []
    for report in gnps:
        candidate = str(report.get("candidate_label", report.get("candidate", "")))
        for panel in ("identity_disjoint", "formula_disjoint"):
            block = report.get(panel)
            if not isinstance(block, dict):
                continue
            delta = block.get("paired_delta_recall1")
            lo = None
            if isinstance(delta, dict):
                lo = delta.get("ci_low")
                delta = delta.get("mean", delta.get("delta"))
            if delta is None:
                continue
            gnps_detail.append({"candidate": candidate, "panel": panel,
                                "delta_recall1": float(delta),
                                "ci_low": None if lo is None else float(lo)})
    if gnps_detail:
        gnps_ok = all(entry["delta_recall1"] >= -1e-12 for entry in gnps_detail)

    gates = {
        "role2_rows_are_distinct_checkpoints": distinct_checkpoints(
            rows, [OFFICIAL_NAME, BASE_NAME, NOISE_NAME, *ARMS]),
        "arm_paired_deltas_match_absolute_recall1": paired_delta_matches_absolute(
            rows, list(ARMS)),
        "compose_beats_chemistry_champion_with_positive_formula_ci": compose_beats_base,
        "compose_beats_matched_continuation_control": causal_content,
        "compose_risk_net_positive": risk_utility(balanced) > 0,
        "role3_non_regression": role3_ok,
        "gnps_dual_panel_non_regression": gnps_ok,
    }
    required = ("role2_rows_are_distinct_checkpoints",
                "arm_paired_deltas_match_absolute_recall1",
                "compose_beats_chemistry_champion_with_positive_formula_ci",
                "compose_beats_matched_continuation_control",
                "compose_risk_net_positive")
    compose_wins = all(gates[key] for key in required) and (
        role3_ok is not False) and (gnps_ok is not False)

    if compose_wins and accounting["super_additive"]:
        status = "GLM_COMPOSE_SUPER_ADDITIVE"
    elif compose_wins:
        status = "GLM_COMPOSE_ADDITIVE_ONLY"
    elif compose_beats_base:
        status = "GLM_COMPOSE_SIGNAL_UNSAFE_CONTROL"
    else:
        status = "GLM_COMPOSE_INCUMBENT_RETAINED"

    return {
        "status": status,
        "accounting": accounting,
        "arm_recall1": {name: recall1(rows[name]) for name in
                        (OFFICIAL_NAME, BASE_NAME, NOISE_NAME, *ARMS)},
        "arm_paired_vs_base": {name: paired(rows[name]) for name in ARMS},
        "axis_hypothesis": {
            "contrast": "compose_balanced - compose_advantage_max",
            "paired_delta_recall1_pp": (
                None if axis_effect is None else
                float(paired(rows_by_name(role2_axis)["compose_balanced"])[
                    "delta_recall1"]) * 100.0),
            "axis_effect_established": axis_effect,
        },
        "gnps_detail": gnps_detail,
        "gates": gates,
        "reading": {
            "super_additive": "composed gain exceeds chemistry + noise on one panel",
            "additive_only": "composed gain matches the sum of parts, no synergy",
            "signal_unsafe_control": "beats the champion but not the continuation "
                                     "control, so the gain is common continuation",
            "incumbent_retained": "no composition gain; keep the champion",
        }[status.replace("GLM_COMPOSE_", "").lower()],
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role2", type=Path, required=True)
    parser.add_argument("--role2-axis", type=Path, default=None)
    parser.add_argument("--role3", type=Path, default=None)
    parser.add_argument("--gnps", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    gnps_reports = []
    for path in args.gnps:
        candidate_paths = ([path] if path.is_file()
                           else sorted(path.glob("report.json")))
        for candidate in candidate_paths:
            gnps_reports.append(load_json(candidate))
    report = verdicts(
        load_json(args.role2),
        None if args.role2_axis is None else load_json(args.role2_axis),
        None if args.role3 is None else load_json(args.role3),
        gnps_reports,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

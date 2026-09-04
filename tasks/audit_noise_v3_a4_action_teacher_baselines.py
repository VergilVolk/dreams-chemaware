"""Compare the nonlinear A4 teacher with simple action heuristics at equal coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_noise_v3_a4_nonlinear_action_teacher import (
    build_variant_table, cluster_bootstrap, select_query_actions,
)


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a4-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan",
    )
    parser.add_argument(
        "--teacher-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_action_teacher",
    )
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def materialize_at_coverage(
    selected: pd.DataFrame, coverage: float, risk_penalty: float, resamples: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    finite = np.isfinite(selected["predicted_utility"].to_numpy(float))
    target = int(np.ceil(coverage * len(selected)))
    candidates = np.flatnonzero(finite)
    order = candidates[np.lexsort((
        selected.iloc[candidates]["scan_position"].to_numpy(np.int64),
        -selected.iloc[candidates]["predicted_utility"].to_numpy(float),
    ))]
    applied = np.zeros(len(selected), dtype=bool)
    applied[order[:min(target, len(order))]] = True
    corrected = applied & selected["corrected_if_applied"].to_numpy(bool)
    introduced = applied & selected["introduced_if_applied"].to_numpy(bool)
    contribution = corrected.astype(float) - risk_penalty * introduced.astype(float)
    summary = {
        "requested_coverage": coverage,
        "interventions": int(applied.sum()),
        "actual_coverage": float(applied.mean()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "risk_weighted_net": float(contribution.sum()),
        "formula_cluster_net_per_query": cluster_bootstrap(
            selected, contribution, resamples, 20260825,
        ),
    }
    return summary, contribution, applied


def main() -> None:
    args = parse_args()
    decision_path = args.teacher_dir / "decision.json"
    score_path = args.teacher_dir / "oof_action_scores.npz"
    selected_path = args.teacher_dir / "oof_selected_actions.csv.gz"
    for path in (decision_path, score_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("formal"):
        raise RuntimeError("baseline audit requires the formal teacher")
    table = build_variant_table(args.a4_dir, 0)
    learned_selected = pd.read_csv(selected_path)
    with np.load(score_path) as body:
        utility = np.asarray(body["utility"], dtype=float)
        if len(utility) != len(table.x):
            raise RuntimeError("OOF utility and exact variants do not align")
    reconstructed = select_query_actions(table, utility)
    compare = learned_selected.merge(
        reconstructed[["scan_position", "action_index", "token", "attenuation"]],
        on="scan_position", suffixes=("_saved", "_rebuilt"), validate="one_to_one",
    )
    for column in ("action_index", "token", "attenuation"):
        if not np.allclose(compare[f"{column}_saved"], compare[f"{column}_rebuilt"]):
            raise RuntimeError(f"saved nonlinear selection does not reproduce: {column}")

    gain_index = table.feature_names.index("dose_specific_predicted_gain")
    gain = table.x[:, gain_index].astype(float)
    strategy_scores = {
        "nonlinear_teacher": utility,
        "gradient_only": gain,
        "confounder_only_gradient": np.where(table.role == 1, gain, -np.inf),
        "shared_only_gradient": np.where(table.role == 2, gain, -np.inf),
        "unmatched_only_gradient": np.where(table.role == 3, gain, -np.inf),
    }
    output = {}
    materialized = {}
    for name, score in strategy_scores.items():
        selected = select_query_actions(table, score)
        if name != "nonlinear_teacher":
            # Queries without the requested role get a literal no-action score.
            valid_by_query = pd.DataFrame({
                "position": table.query_position, "valid": np.isfinite(score),
            }).groupby("position")["valid"].any()
            selected["predicted_utility"] = np.where(
                selected["scan_position"].map(valid_by_query).fillna(False),
                selected["predicted_utility"], -np.inf,
            )
        rows = []
        for coverage in (0.05, 0.10, 0.20, 0.40):
            summary, contribution, applied = materialize_at_coverage(
                selected, coverage, args.risk_penalty, args.bootstrap_resamples,
            )
            rows.append(summary)
            materialized[(name, coverage)] = (selected, contribution, applied)
        output[name] = rows
        print(f"[baseline audit] {name}", flush=True)

    paired_policy_comparisons = {}
    for baseline_name in (
        "gradient_only", "confounder_only_gradient",
        "shared_only_gradient", "unmatched_only_gradient",
    ):
        comparisons = []
        for coverage in (0.05, 0.10, 0.20, 0.40):
            teacher_selected, teacher_contribution, teacher_applied = materialized[
                ("nonlinear_teacher", coverage)
            ]
            baseline_selected, baseline_contribution, baseline_applied = materialized[
                (baseline_name, coverage)
            ]
            if not np.array_equal(
                teacher_selected["scan_position"].to_numpy(),
                baseline_selected["scan_position"].to_numpy(),
            ):
                raise RuntimeError("policy rows are not query aligned")
            delta = teacher_contribution - baseline_contribution
            comparisons.append({
                "coverage": coverage,
                "teacher_minus_baseline_risk_weighted_net": float(delta.sum()),
                "formula_cluster_teacher_minus_baseline_net_per_query": cluster_bootstrap(
                    teacher_selected, delta, args.bootstrap_resamples, 20260826,
                ),
                "action_overlap": int(np.sum(teacher_applied & baseline_applied)),
                "teacher_only_actions": int(np.sum(teacher_applied & ~baseline_applied)),
                "baseline_only_actions": int(np.sum(~teacher_applied & baseline_applied)),
            })
        paired_policy_comparisons[f"nonlinear_teacher_vs_{baseline_name}"] = comparisons
    result = {
        "status": "noise_v3_a4_action_teacher_baseline_audit_complete",
        "queries": int(len(learned_selected)),
        "risk_penalty": args.risk_penalty,
        "strategies": output,
        "paired_policy_comparisons": paired_policy_comparisons,
        "interpretation": (
            "All strategies use the same exact A4 outcomes and global query coverage. "
            "This is formula-OOF only for the nonlinear teacher; heuristic baselines train nothing."
        ),
    }
    path = args.teacher_dir / "baseline_comparison.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

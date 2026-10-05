"""Paired formula-cluster summary of candidate-distribution teacher arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("spectrum_only", "correct", "candidate_swapped", "peak_permuted")


def paired_bootstrap(delta: np.ndarray, formula: np.ndarray, seed: int, draws: int) -> dict:
    unique = np.unique(formula.astype(str))
    macro = np.asarray([np.mean(delta[formula.astype(str) == value]) for value in unique])
    rng = np.random.default_rng(seed)
    estimates = np.asarray([
        np.mean(macro[rng.integers(0, len(macro), len(macro))]) for _ in range(draws)
    ])
    return {
        "formula_macro_advantage": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)), "draws": int(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    reports = {}
    arrays = {}
    for arm in ARMS:
        report_path = args.root / arm / "report.json"
        query_path = args.root / arm / "inner_per_query.npz"
        if not report_path.is_file() or not query_path.is_file():
            raise FileNotFoundError(f"incomplete candidate-distribution arm: {arm}")
        reports[arm] = json.loads(report_path.read_text(encoding="utf-8"))
        arrays[arm] = np.load(query_path, allow_pickle=True)
        if reports[arm].get("preflight", {}).get("arm") != arm:
            raise RuntimeError(f"arm metadata mismatch: {arm}")
    reference = arrays["spectrum_only"]
    for arm in ARMS[1:]:
        if (not np.array_equal(reference["query_index"], arrays[arm]["query_index"])
                or not np.array_equal(reference["initial_rank"], arrays[arm]["initial_rank"])):
            raise RuntimeError(f"held membership/baseline differs for arm {arm}")
        left = reports["spectrum_only"]["optimization"]
        right = reports[arm]["optimization"]
        excluded = {"arm", "output"}
        if any(left[key] != right[key] for key in left if key not in excluded):
            raise RuntimeError(f"optimizer/schedule arguments differ for arm {arm}")
    final_correct = {
        arm: arrays[arm]["final_rank"] == 1 for arm in ARMS
    }
    comparisons = {}
    for index, control in enumerate(("spectrum_only", "candidate_swapped", "peak_permuted")):
        delta = final_correct["correct"].astype(float) - final_correct[control].astype(float)
        comparisons[f"correct_minus_{control}"] = {
            "recall1_advantage": float(np.mean(delta)),
            "correct_wins": int(np.sum(delta > 0)),
            "control_wins": int(np.sum(delta < 0)),
            **paired_bootstrap(delta, reference["formula"], args.seed + index * 101, args.bootstrap_draws),
        }
    absolute = reports["correct"]["final_inner"]
    gates = {
        "correct_absolute_positive": absolute["delta_recall1"] > 0,
        "correct_risk_positive": absolute["corrected"] > absolute["introduced"],
        "correct_beats_spectrum_only": comparisons["correct_minus_spectrum_only"]["formula_cluster_bootstrap_95ci"][0] > 0,
        "correct_beats_candidate_swapped": comparisons["correct_minus_candidate_swapped"]["formula_cluster_bootstrap_95ci"][0] > 0,
        "correct_beats_peak_permuted": comparisons["correct_minus_peak_permuted"]["formula_cluster_bootstrap_95ci"][0] > 0,
    }
    report = {
        "status": "DEVELOPMENT_PASS" if all(gates.values()) else "DEVELOPMENT_FAIL",
        "scope": "development chemical-attribution gate; outer remains untouched",
        "arms": {arm: reports[arm]["final_inner"] for arm in ARMS},
        "paired_comparisons": comparisons, "gates": gates,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

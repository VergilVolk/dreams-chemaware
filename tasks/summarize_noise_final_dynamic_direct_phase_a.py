"""Paired formula-cluster decision for the four-arm dynamic-direct Phase A."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
from noise_final_core import sha256_file

ARMS = ("clean_continuation", "matched_random", "static_target", "dynamic_np")


def cluster_ci(effect: np.ndarray, formula: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    frame = pd.DataFrame({"effect": effect.astype(float), "formula": formula.astype(str)})
    group = frame.groupby("formula", sort=True)["effect"].agg(["sum", "count"])
    sums, counts = group["sum"].to_numpy(float), group["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        chosen = rng.integers(0, len(sums), len(sums))
        draws[index] = sums[chosen].sum() / counts[chosen].sum()
    return {"mean": float(np.mean(effect)), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975))}


def signflip_p(effect: np.ndarray, formula: np.ndarray, repeats: int, seed: int) -> float:
    group = pd.DataFrame({"e": effect.astype(float), "f": formula.astype(str)}).groupby("f")["e"].sum()
    values = group.to_numpy(float)
    observed = float(values.sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repeats):
        exceed += int(float(np.sum(values * rng.choice((-1.0, 1.0), len(values)))) >= observed)
    return float((exceed + 1) / (repeats + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite Phase-A summary: {args.output_dir}")
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for arm in ARMS:
        directory = args.root / arm
        report_path, query_path = directory / "report.json", directory / "held_per_query.csv.gz"
        if not report_path.is_file() or not query_path.is_file():
            raise FileNotFoundError(f"incomplete Phase-A arm: {directory}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "noise_final_dynamic_direct_phase_a_arm_complete"
                or report.get("formal") is not True or report.get("arm") != arm):
            raise RuntimeError(f"invalid Phase-A arm report: {arm}")
        reports[arm] = report
        frames[arm] = pd.read_csv(query_path).sort_values("query_index", kind="stable").reset_index(drop=True)
    identity = ["query_index", "query_row", "query_formula", "has_near", "official_rank", "initial_rank"]
    reference = frames[ARMS[0]]
    for arm in ARMS[1:]:
        if not reference[identity].equals(frames[arm][identity]):
            raise RuntimeError(f"held ledger differs across Phase-A arms: {arm}")
    dynamic = frames["dynamic_np"]["final_rank"].to_numpy(np.int16)
    formula = reference["query_formula"].astype(str).to_numpy()
    near = reference["has_near"].astype(bool).to_numpy()
    comparisons: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for offset, other in enumerate(("matched_random", "static_target", "clean_continuation")):
        baseline = frames[other]["final_rank"].to_numpy(np.int16)
        effect = (dynamic == 1).astype(np.int8) - (baseline == 1).astype(np.int8)
        name = f"dynamic_np_vs_{other}"
        comparisons[name] = {
            "delta_recall1": float(np.mean(effect)),
            "corrected": int(np.sum((baseline != 1) & (dynamic == 1))),
            "introduced": int(np.sum((baseline == 1) & (dynamic != 1))),
            "risk_net_lambda2": int(np.sum((baseline != 1) & (dynamic == 1)) - 2 * np.sum((baseline == 1) & (dynamic != 1))),
            "near_delta_recall1": float(np.mean(effect[near])),
            "formula_cluster_ci": cluster_ci(effect, formula, args.bootstrap_resamples, args.seed + offset),
        }
        if other in {"matched_random", "static_target"}:
            raw_p[name] = signflip_p(effect, formula, args.bootstrap_resamples, args.seed + 100 + offset)
    ordered = sorted(raw_p, key=raw_p.get)
    holm: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for position, name in enumerate(ordered):
        adjusted = min(1.0, raw_p[name] * (len(ordered) - position))
        running = max(running, adjusted)
        holm[name] = {"raw_one_sided_p": raw_p[name], "holm_adjusted_p": running,
                      "significant_0_05": running < 0.05}
    d = reports["dynamic_np"]["held"]
    gates = {
        "dynamic_beats_matched_random_holm": holm["dynamic_np_vs_matched_random"]["significant_0_05"],
        "dynamic_beats_matched_random_formula_ci_positive": (
            comparisons["dynamic_np_vs_matched_random"]["formula_cluster_ci"]["ci_low"] > 0
        ),
        "dynamic_beats_static_target_holm": holm["dynamic_np_vs_static_target"]["significant_0_05"],
        "dynamic_beats_static_target_formula_ci_positive": (
            comparisons["dynamic_np_vs_static_target"]["formula_cluster_ci"]["ci_low"] > 0
        ),
        "dynamic_beats_clean_point": comparisons["dynamic_np_vs_clean_continuation"]["delta_recall1"] > 0,
        "dynamic_corrected_gt_introduced_vs_initial": d["corrected_vs_initial"] > d["introduced_vs_initial"],
        "dynamic_risk_net_positive_vs_initial": d["risk_net_vs_initial"] > 0,
        "dynamic_near_nonnegative_vs_initial": d["near_delta_vs_initial"] >= 0,
        "dynamic_mrr_nonnegative_vs_initial": d["mrr_delta_vs_initial"] >= 0,
        "dynamic_preservation_ge_0_995": d["preservation_vs_initial_mean"] >= 0.995,
        "dynamic_not_systematically_clipped": max(
            float(row["clip_fraction"]) for row in reports["dynamic_np"].get("history", [])
        ) < 0.95,
    }
    report = {
        "status": "noise_final_dynamic_direct_phase_a_summary_complete", "formal": True,
        "arms": {arm: reports[arm]["held"] for arm in ARMS},
        "paired_comparisons": comparisons, "multiplicity": holm, "gates": gates,
        "pass_to_second_seed": bool(all(gates.values())),
        "decision": "advance to a second seed" if all(gates.values()) else "stop and diagnose the failed paired Phase-A gate",
        "contracts": {
            "paired_same_held_queries": True, "formula_cluster_unit": True,
            "matched_control_semantics": {
                "N": "matched random peak path",
                "P_intensity": "wrong-reference direction control",
                "P_transfer": "wrong-reference direction control",
            },
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {arm: sha256_file(args.root / arm / "report.json") for arm in ARMS},
        "claim_limit": "One development formula fold; not multifold or sealed P3 performance.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dd_phase_a_summary_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

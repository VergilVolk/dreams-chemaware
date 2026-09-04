"""Paired formula-cluster audit of E7 recurrent peak-transfer augmentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import formula_bootstrap_delta  # noqa: E402


EXPECTED = {
    0.0: "e7_fixed_control",
    0.025: "e7_recur_w0025",
    0.05: "e7_recur_w005",
    0.10: "e7_recur_w010",
    0.20: "e7_recur_w020",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e7_recurrent_transfer",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e7_recurrent_transfer_summary.json",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def paired(reference: pd.DataFrame, candidate: pd.DataFrame, n: int, seed: int) -> dict:
    columns = ("query_index", "query_formula", "baseline_rank")
    for column in columns:
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise RuntimeError(f"E7 paired query mismatch: {column}")
    result = formula_bootstrap_delta(
        reference["final_rank"].to_numpy(np.int16),
        candidate["final_rank"].to_numpy(np.int16),
        candidate["query_formula"].astype(str).to_numpy(), n, seed,
    )
    old = reference["final_rank"].to_numpy(int) == 1
    new = candidate["final_rank"].to_numpy(int) == 1
    result.update({
        "corrected_vs_control": int(np.sum(~old & new)),
        "introduced_vs_control": int(np.sum(old & ~new)),
        "net_vs_control": int(np.sum(new) - np.sum(old)),
    })
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E7 summary: {args.output}")
    records: dict[float, tuple[Path, dict, pd.DataFrame]] = {}
    for path in sorted(args.root.rglob("decision.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        configuration = body.get("configuration", {})
        if int(configuration.get("seed", -1)) != args.seed:
            continue
        if int(configuration.get("outer_fold", -1)) != 0:
            continue
        weight = float(configuration.get("guided_noise_weight", -1))
        if weight not in EXPECTED:
            continue
        if str(configuration.get("run_suffix", "")) != EXPECTED[weight]:
            continue
        expected_policy = "none" if weight == 0 else "transfer"
        if configuration.get("guided_noise_policy") != expected_policy:
            raise RuntimeError(f"E7 guided policy/weight mismatch at {path}")
        if weight in records:
            raise RuntimeError(f"duplicate E7 weight: {weight}")
        frame = pd.read_csv(path.parent / "held_per_query.csv.gz")
        frame = frame.sort_values("query_index").reset_index(drop=True)
        records[weight] = path, body, frame
    if set(records) != set(EXPECTED):
        raise RuntimeError(f"expected E7 weights {sorted(EXPECTED)}, observed {sorted(records)}")

    control = records[0.0][2]
    arms = {}
    comparisons = {}
    for offset, weight in enumerate(sorted(records)):
        path, body, frame = records[weight]
        held = body["held_clean"]
        arms[f"weight={weight:g}"] = {
            "decision_sha256": sha256_file(path),
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "delta_mrr_vs_official": float(held["delta_mrr"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "risk_net_vs_official": int(held["risk_net"]),
            "preservation_mean": float(held["preservation_mean"]),
            "official_gate_pass": bool(body["pass_to_multifold"]),
        }
        if weight > 0:
            comparisons[f"weight={weight:g}_vs_control"] = paired(
                control, frame, args.bootstrap, args.seed + offset + 1,
            )

    eligible = []
    for weight in sorted(EXPECTED):
        if weight == 0:
            continue
        arm = arms[f"weight={weight:g}"]
        comparison = comparisons[f"weight={weight:g}_vs_control"]
        if (
            arm["official_gate_pass"]
            and comparison["ci_low"] > 0
            and comparison["corrected_vs_control"] > comparison["introduced_vs_control"]
        ):
            eligible.append(weight)
    # Select only among pre-registered weights after reporting all arms.  This
    # is a development-fold choice and must be frozen before multifold testing.
    selected = max(
        eligible,
        key=lambda weight: (
            comparisons[f"weight={weight:g}_vs_control"]["mean"], -weight,
        ),
    ) if eligible else None
    report = {
        "status": "noise_final_e7_recurrent_transfer_summary_complete",
        "formal": True,
        "seed": args.seed,
        "outer_formula_fold": 0,
        "arms": arms,
        "paired_vs_fixed_control": comparisons,
        "eligible_weights": eligible,
        "selected_weight_for_multifold": selected,
        "pass_to_multifold": selected is not None,
        "decision": (
            "freeze selected recurrent-transfer weight and run unseen formula folds"
            if selected is not None
            else "recurrent_union_mix has no significant increment over fixed direct noise"
        ),
        "claim_limit": (
            "One consumed development formula fold. Every arm trains one shared clean-spectrum "
            "DreaMS encoder; no downstream expert, P2b or P3 is used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

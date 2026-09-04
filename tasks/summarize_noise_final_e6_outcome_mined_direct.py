"""Strict paired audit of direct fixed versus train-fold outcome-mined noise."""
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
    "fixed|v=2|sw=2": "e6_fixed_v2_sw2",
    "fixed|v=4|sw=2": "e6_fixed_v4_sw2",
    "fixed|v=4|sw=4": "e6_fixed_v4_sw4",
    "outcome_mined|v=2|sw=2": "e6_mined_v2_sw2",
    "outcome_mined|v=4|sw=2": "e6_mined_v4_sw2",
    "outcome_mined|v=4|sw=4": "e6_mined_v4_sw4",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e6_outcome_mined_direct",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e6_outcome_mined_direct_summary.json",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def key(configuration: dict) -> str:
    return (
        f"{configuration.get('action_selection', 'fixed')}"
        f"|v={int(configuration.get('views_per_identity', -1))}"
        f"|sw={float(configuration.get('safety_stream_weight', 1.0)):g}"
    )


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, n: int, seed: int) -> dict:
    for column in ("query_index", "query_formula", "baseline_rank"):
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise RuntimeError(f"paired E6 query mismatch: {column}")
    result = formula_bootstrap_delta(
        reference["final_rank"].to_numpy(np.int16),
        candidate["final_rank"].to_numpy(np.int16),
        candidate["query_formula"].astype(str).to_numpy(), n, seed,
    )
    old = reference["final_rank"].to_numpy(int) == 1
    new = candidate["final_rank"].to_numpy(int) == 1
    result.update({
        "corrected_vs_reference": int(np.sum(~old & new)),
        "introduced_vs_reference": int(np.sum(old & ~new)),
        "net_vs_reference": int(np.sum(new) - np.sum(old)),
    })
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E6 summary: {args.output}")
    records = {}
    for path in sorted(args.root.rglob("decision.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        configuration = body.get("configuration", {})
        if int(configuration.get("seed", -1)) != args.seed:
            continue
        if int(configuration.get("outer_fold", -1)) != 0:
            continue
        arm = key(configuration)
        if arm not in EXPECTED or str(configuration.get("run_suffix", "")) != EXPECTED[arm]:
            continue
        if arm in records:
            raise RuntimeError(f"duplicate E6 arm: {arm}")
        per_query = path.parent / "held_per_query.csv.gz"
        if not per_query.is_file():
            raise FileNotFoundError(per_query)
        records[arm] = (
            path, body,
            pd.read_csv(per_query).sort_values("query_index").reset_index(drop=True),
        )
    if set(records) != set(EXPECTED):
        raise RuntimeError(f"expected E6 arms {sorted(EXPECTED)}, observed {sorted(records)}")

    fixed_v2_sw2 = records["fixed|v=2|sw=2"][2]
    fixed_sw2 = records["fixed|v=4|sw=2"][2]
    fixed_sw4 = records["fixed|v=4|sw=4"][2]
    mined_v2_sw2 = records["outcome_mined|v=2|sw=2"][2]
    mined_v4_sw2 = records["outcome_mined|v=4|sw=2"][2]
    mined_v4_sw4 = records["outcome_mined|v=4|sw=4"][2]
    comparisons = {
        "mined_v2_sw2_vs_fixed_v2_sw2": compare(
            fixed_v2_sw2, mined_v2_sw2, args.bootstrap, args.seed + 1,
        ),
        "mined_v4_sw2_vs_fixed_v4_sw2": compare(
            fixed_sw2, mined_v4_sw2, args.bootstrap, args.seed + 2,
        ),
        "mined_v4_sw4_vs_fixed_v4_sw4": compare(
            fixed_sw4, mined_v4_sw4, args.bootstrap, args.seed + 3,
        ),
        "fixed_views4_vs_views2_at_sw2": compare(
            fixed_v2_sw2, fixed_sw2, args.bootstrap, args.seed + 4,
        ),
        "mined_views4_vs_views2_at_sw2": compare(
            mined_v2_sw2, mined_v4_sw2, args.bootstrap, args.seed + 5,
        ),
        "fixed_sw4_vs_sw2_at_views4": compare(
            fixed_sw2, fixed_sw4, args.bootstrap, args.seed + 6,
        ),
        "mined_sw4_vs_sw2_at_views4": compare(
            mined_v4_sw2, mined_v4_sw4, args.bootstrap, args.seed + 7,
        ),
    }
    arms = {}
    for arm, (path, body, _) in records.items():
        held = body["held_clean"]
        arms[arm] = {
            "decision_sha256": sha256_file(path),
            "train_action_rows": int(body["data"]["train_action_rows"]),
            "train_action_identities": int(body["data"]["train_action_identities"]),
            "train_action_formulas": int(body["data"]["train_action_formulas"]),
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "risk_net_vs_official": int(held["risk_net"]),
            "preservation_mean": float(held["preservation_mean"]),
            "official_gate_pass": bool(body["pass_to_multifold"]),
        }
    primary = comparisons["mined_v4_sw2_vs_fixed_v4_sw2"]
    selected = "outcome_mined|v=4|sw=2"
    pass_to_multifold = bool(
        arms[selected]["official_gate_pass"]
        and primary["ci_low"] > 0
        and primary["corrected_vs_reference"] > primary["introduced_vs_reference"]
    )
    report = {
        "status": "noise_final_e6_outcome_mined_direct_summary_complete",
        "formal": True,
        "outer_formula_fold": 0,
        "seed": args.seed,
        "arms": arms,
        "paired_comparisons": comparisons,
        "primary_arm": selected,
        "pass_to_multifold": pass_to_multifold,
        "decision": (
            "advance direct outcome-mined raw-spectrum augmentation to multifold"
            if pass_to_multifold
            else "do not advance this direct outcome-mined recipe"
        ),
        "claim_limit": (
            "One held formula fold. Outcome mining is training-only and per-query separable; "
            "held action outcomes, P2b and P3 are not used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

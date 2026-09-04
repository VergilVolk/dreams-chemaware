"""Paired formula-held audit of the four E5 guided-noise pilot arms."""
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/validation/g8r_noise_final_e5_guided_shared")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_noise_final_e5_guided_pilot_summary.json")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E5 pilot summary: {args.output}")
    decisions = sorted(args.root.rglob("decision.json"))
    records: dict[str, tuple[Path, dict, pd.DataFrame]] = {}
    for path in decisions:
        body = json.loads(path.read_text(encoding="utf-8"))
        configuration = body.get("configuration", {})
        if int(configuration.get("seed", -1)) != args.seed or int(configuration.get("outer_fold", -1)) != 0:
            continue
        policy = str(configuration.get("guided_noise_policy", "none"))
        per_query_path = path.parent / "held_per_query.csv.gz"
        if policy in records:
            raise RuntimeError(f"duplicate E5 pilot policy {policy}")
        if not per_query_path.is_file():
            raise FileNotFoundError(per_query_path)
        records[policy] = (path, body, pd.read_csv(per_query_path).sort_values("query_index"))
    expected = {"none", "transfer", "intensity", "both"}
    if set(records) != expected:
        raise RuntimeError(f"expected E5 policies {sorted(expected)}, observed {sorted(records)}")
    base_frame = records["none"][2].reset_index(drop=True)
    summaries = {}
    for offset, policy in enumerate(("none", "transfer", "intensity", "both")):
        path, body, frame = records[policy]
        frame = frame.reset_index(drop=True)
        for column in ("query_index", "query_formula", "baseline_rank"):
            if not np.array_equal(frame[column].to_numpy(), base_frame[column].to_numpy()):
                raise RuntimeError(f"paired held-query mismatch for {policy}: {column}")
        held = body["held_clean"]
        comparison = None
        if policy != "none":
            comparison = formula_bootstrap_delta(
                base_frame["final_rank"].to_numpy(np.int16),
                frame["final_rank"].to_numpy(np.int16),
                frame["query_formula"].astype(str).to_numpy(),
                args.bootstrap, args.seed + offset,
            )
            n_correct = base_frame["final_rank"].to_numpy(int) == 1
            p_correct = frame["final_rank"].to_numpy(int) == 1
            comparison.update({
                "corrected_vs_n_only": int(np.sum(~n_correct & p_correct)),
                "introduced_vs_n_only": int(np.sum(n_correct & ~p_correct)),
            })
        summaries[policy] = {
            "decision_path": str(path),
            "decision_sha256": sha256_file(path),
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "risk_net_vs_official": int(held["risk_net"]),
            "formula_ci_vs_official": held["formula_cluster_delta_recall1"],
            "preservation_mean": float(held["preservation_mean"]),
            "pass_to_multifold": bool(body["pass_to_multifold"]),
            "incremental_vs_n_only": comparison,
        }
    eligible = [
        policy for policy in ("transfer", "intensity", "both")
        if summaries[policy]["pass_to_multifold"]
        and summaries[policy]["incremental_vs_n_only"]["ci_low"] > 0
        and summaries[policy]["incremental_vs_n_only"]["corrected_vs_n_only"]
        > summaries[policy]["incremental_vs_n_only"]["introduced_vs_n_only"]
    ]
    selected = max(
        eligible,
        key=lambda policy: (
            summaries[policy]["risk_net_vs_official"],
            summaries[policy]["delta_recall1_vs_official"],
        ),
    ) if eligible else None
    report = {
        "status": "noise_final_e5_guided_pilot_summary_complete",
        "formal": True,
        "paired_formula_fold": 0,
        "seed": args.seed,
        "policies": summaries,
        "eligible_for_multifold": eligible,
        "selected_policy": selected,
        "pass_to_multifold": selected is not None,
        "decision_rule": (
            "A guided arm must pass all official-baseline gates, have a positive formula-cluster "
            "CI versus the paired N-only run, and correct more N-only errors than it introduces."
        ),
        "claim_limit": "One held-formula-fold pilot; not a multi-fold or sealed-P3 result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

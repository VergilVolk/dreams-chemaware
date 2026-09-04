"""Paired held-formula audit of the preregistered E5-B dose/safety arms."""
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
    "none|w=0|s=1": "e5b_n_only",
    "intensity|w=0.1|s=1": "e5b_intensity_w010_s1",
    "intensity|w=0.25|s=1": "e5b_intensity_w025_s1",
    "intensity|w=0.5|s=1": "e5b_intensity_w050_s1",
    "intensity|w=0.25|s=2": "e5b_intensity_w025_s2",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e5b_guided_dose",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e5b_guided_dose_summary.json",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def key(configuration: dict) -> str:
    return (
        f"{configuration.get('guided_noise_policy', 'none')}"
        f"|w={float(configuration.get('guided_noise_weight', 0.0)):g}"
        f"|s={float(configuration.get('safety_ratio', 1.0)):g}"
    )


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E5-B summary: {args.output}")
    records: dict[str, tuple[Path, dict, pd.DataFrame]] = {}
    for path in sorted(args.root.rglob("decision.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        configuration = body.get("configuration", {})
        if int(configuration.get("seed", -1)) != args.seed:
            continue
        if int(configuration.get("outer_fold", -1)) != 0:
            continue
        arm = key(configuration)
        suffix = str(configuration.get("run_suffix", ""))
        if arm not in EXPECTED or suffix != EXPECTED[arm]:
            continue
        per_query_path = path.parent / "held_per_query.csv.gz"
        if not per_query_path.is_file():
            raise FileNotFoundError(per_query_path)
        if arm in records:
            raise RuntimeError(f"duplicate E5-B arm: {arm}")
        records[arm] = (
            path,
            body,
            pd.read_csv(per_query_path).sort_values("query_index").reset_index(drop=True),
        )
    if set(records) != set(EXPECTED):
        raise RuntimeError(
            f"expected E5-B arms {sorted(EXPECTED)}, observed {sorted(records)}"
        )

    control_key = "none|w=0|s=1"
    control = records[control_key][2]
    summaries: dict[str, dict] = {}
    for offset, arm in enumerate(EXPECTED):
        path, body, frame = records[arm]
        for column in ("query_index", "query_formula", "baseline_rank"):
            if not np.array_equal(frame[column].to_numpy(), control[column].to_numpy()):
                raise RuntimeError(f"paired held-query mismatch for {arm}: {column}")
        held = body["held_clean"]
        comparison = None
        if arm != control_key:
            comparison = formula_bootstrap_delta(
                control["final_rank"].to_numpy(np.int16),
                frame["final_rank"].to_numpy(np.int16),
                frame["query_formula"].astype(str).to_numpy(),
                args.bootstrap,
                args.seed + offset,
            )
            control_correct = control["final_rank"].to_numpy(int) == 1
            candidate_correct = frame["final_rank"].to_numpy(int) == 1
            comparison.update({
                "corrected_vs_n_only": int(np.sum(~control_correct & candidate_correct)),
                "introduced_vs_n_only": int(np.sum(control_correct & ~candidate_correct)),
            })
        summaries[arm] = {
            "run_suffix": EXPECTED[arm],
            "decision_path": str(path),
            "decision_sha256": sha256_file(path),
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "delta_mrr_vs_official": float(held["delta_mrr"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "risk_net_vs_official": int(held["risk_net"]),
            "formula_ci_vs_official": held["formula_cluster_delta_recall1"],
            "preservation_mean": float(held["preservation_mean"]),
            "pass_to_multifold": bool(body["pass_to_multifold"]),
            "incremental_vs_n_only": comparison,
            "last_epoch": body["history"][-1],
        }

    eligible = []
    for arm in EXPECTED:
        if arm == control_key:
            continue
        result = summaries[arm]
        incremental = result["incremental_vs_n_only"]
        if (
            result["pass_to_multifold"]
            and incremental["mean"] > 0
            and incremental["corrected_vs_n_only"] > incremental["introduced_vs_n_only"]
        ):
            eligible.append(arm)
    selected = max(
        eligible,
        key=lambda arm: (
            summaries[arm]["risk_net_vs_official"],
            summaries[arm]["delta_recall1_vs_official"],
            summaries[arm]["preservation_mean"],
        ),
    ) if eligible else None
    report = {
        "status": "noise_final_e5b_guided_dose_summary_complete",
        "formal": True,
        "paired_formula_fold": 0,
        "seed": args.seed,
        "control": control_key,
        "arms": summaries,
        "eligible_for_multifold": eligible,
        "selected_arm": selected,
        "pass_to_multifold": selected is not None,
        "decision_rule": (
            "A guided dose must pass all official-baseline gates, have positive paired point "
            "gain over N-only, and correct more N-only errors than it introduces. Formula-CI "
            "versus N-only is reported but is not required in this single-fold dose screen."
        ),
        "claim_limit": "One held-formula-fold dose/safety screen; not sealed-P3 performance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Strict paired formula-cluster audit of the five E5-C safety-weight arms."""
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
    "none|gw=0|sw=1": "e5c_n_only_sw1",
    "none|gw=0|sw=2": "e5c_n_only_sw2",
    "intensity|gw=0.1|sw=2": "e5c_iw010_sw2",
    "intensity|gw=0.1|sw=4": "e5c_iw010_sw4",
    "intensity|gw=0.05|sw=2": "e5c_iw005_sw2",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e5c_safety_weight",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e5c_safety_weight_summary.json",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def arm_key(configuration: dict) -> str:
    return (
        f"{configuration.get('guided_noise_policy', 'none')}"
        f"|gw={float(configuration.get('guided_noise_weight', 0.0)):g}"
        f"|sw={float(configuration.get('safety_stream_weight', 1.0)):g}"
    )


def paired(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> dict:
    for column in ("query_index", "query_formula", "baseline_rank"):
        if not np.array_equal(candidate[column].to_numpy(), reference[column].to_numpy()):
            raise RuntimeError(f"paired held-query mismatch: {column}")
    result = formula_bootstrap_delta(
        reference["final_rank"].to_numpy(np.int16),
        candidate["final_rank"].to_numpy(np.int16),
        candidate["query_formula"].astype(str).to_numpy(),
        bootstrap,
        seed,
    )
    reference_correct = reference["final_rank"].to_numpy(int) == 1
    candidate_correct = candidate["final_rank"].to_numpy(int) == 1
    result.update({
        "corrected_vs_reference": int(np.sum(~reference_correct & candidate_correct)),
        "introduced_vs_reference": int(np.sum(reference_correct & ~candidate_correct)),
        "net_vs_reference": int(np.sum(candidate_correct) - np.sum(reference_correct)),
    })
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E5-C summary: {args.output}")
    records: dict[str, tuple[Path, dict, pd.DataFrame]] = {}
    for path in sorted(args.root.rglob("decision.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        configuration = body.get("configuration", {})
        if int(configuration.get("seed", -1)) != args.seed:
            continue
        if int(configuration.get("outer_fold", -1)) != 0:
            continue
        arm = arm_key(configuration)
        if arm not in EXPECTED:
            continue
        if str(configuration.get("run_suffix", "")) != EXPECTED[arm]:
            continue
        if arm in records:
            raise RuntimeError(f"duplicate E5-C arm: {arm}")
        per_query = path.parent / "held_per_query.csv.gz"
        if not per_query.is_file():
            raise FileNotFoundError(per_query)
        records[arm] = (
            path,
            body,
            pd.read_csv(per_query).sort_values("query_index").reset_index(drop=True),
        )
    if set(records) != set(EXPECTED):
        raise RuntimeError(
            f"expected E5-C arms {sorted(EXPECTED)}, observed {sorted(records)}"
        )

    sw1 = records["none|gw=0|sw=1"][2]
    sw2 = records["none|gw=0|sw=2"][2]
    comparisons = {
        "none_sw2_vs_none_sw1": paired(sw1, sw2, args.bootstrap, args.seed + 1),
        "intensity_gw010_sw2_vs_none_sw2": paired(
            sw2, records["intensity|gw=0.1|sw=2"][2], args.bootstrap, args.seed + 2,
        ),
        "intensity_gw005_sw2_vs_none_sw2": paired(
            sw2, records["intensity|gw=0.05|sw=2"][2], args.bootstrap, args.seed + 3,
        ),
        # There is no sw=4 N-only control.  This comparison is reported only
        # against the original N-only model and cannot isolate safety weight.
        "intensity_gw010_sw4_vs_none_sw1_nonfactorial": paired(
            sw1, records["intensity|gw=0.1|sw=4"][2], args.bootstrap, args.seed + 4,
        ),
    }
    arms = {}
    for arm in EXPECTED:
        path, body, _ = records[arm]
        held = body["held_clean"]
        arms[arm] = {
            "decision_sha256": sha256_file(path),
            "delta_recall1_vs_official": float(held["delta_recall1"]),
            "delta_near_recall1_vs_official": float(held["delta_near_recall1"]),
            "corrected_vs_official": int(held["corrected"]),
            "introduced_vs_official": int(held["introduced"]),
            "risk_net_vs_official": int(held["risk_net"]),
            "preservation_mean": float(held["preservation_mean"]),
            "official_gate_pass": bool(body["pass_to_multifold"]),
        }

    key_comparison = comparisons["intensity_gw010_sw2_vs_none_sw2"]
    safe_guided_incremental_pass = bool(
        arms["intensity|gw=0.1|sw=2"]["preservation_mean"] >= 0.995
        and key_comparison["ci_low"] > 0
        and key_comparison["corrected_vs_reference"]
        > key_comparison["introduced_vs_reference"]
    )
    report = {
        "status": "noise_final_e5c_safety_weight_summary_complete",
        "formal": True,
        "seed": args.seed,
        "outer_formula_fold": 0,
        "arms": arms,
        "paired_comparisons": comparisons,
        "safe_guided_incremental_pass": safe_guided_incremental_pass,
        "decision": (
            "advance fixed intensity to multifold"
            if safe_guided_incremental_pass
            else "stop fixed-intensity dose scanning; retain matrix for training-only per-query action mining"
        ),
        "claim_limit": (
            "One held formula fold. The sw4 arm lacks a factorial N-only/sw4 control and is not used "
            "to attribute a guided effect."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

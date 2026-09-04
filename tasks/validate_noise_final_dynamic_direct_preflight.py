"""Independent validator for a completed dynamic-direct preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [args.output_dir / "report.json", args.output_dir / "formula_split.csv.gz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    split = pd.read_csv(required[1])
    gates = {
        "status": report.get("status") == "noise_final_dynamic_direct_preflight_complete",
        "formal": report.get("formal") is True,
        "queries": report.get("graph", {}).get("queries") == 23876,
        "N_cells": report.get("N", {}).get("mature_cells") == 9,
        "four_arms": report.get("phase_a_arms") == [
            "clean_continuation", "matched_random", "static_target", "dynamic_np",
        ],
        "model_not_loaded": report.get("contracts", {}).get("model_loaded") is False,
        "P2b_forbidden": report.get("contracts", {}).get("P2b") == "forbidden",
        "P3_not_consumed": report.get("contracts", {}).get("P3_consumed") is False,
        "full_graph": report.get("contracts", {}).get("full_candidate_graph_preserved") is True,
        "held_outcomes_quarantined": report.get("contracts", {}).get("outer_held_outcome_used_for_training") is False,
        "pass": report.get("pass_to_gpu_replay") is True,
        "split_rows": len(split) == 23876,
        "split_values": set(split["split"].astype(str)) == {"train", "held"},
        "formula_disjoint": not (
            set(split.loc[split["split"].eq("train"), "query_formula"].astype(str))
            & set(split.loc[split["split"].eq("held"), "query_formula"].astype(str))
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"dynamic-direct preflight validation failed: {gates}")
    print("[validate_noise_final_dynamic_direct_preflight] PASS")


if __name__ == "__main__":
    main()

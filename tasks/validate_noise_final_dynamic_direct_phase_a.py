"""Fail-closed structural validator for one Phase-A arm or its summary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.summary:
        gates = {
            "status": report.get("status") == "noise_final_dynamic_direct_phase_a_summary_complete",
            "formal": report.get("formal") is True,
            "arms": set(report.get("arms", {})) == {"clean_continuation", "matched_random", "static_target", "dynamic_np"},
            "comparisons": len(report.get("paired_comparisons", {})) == 3,
            "boundary": report.get("contracts", {}).get("P2b") == "forbidden" and report.get("contracts", {}).get("P3_consumed") is False,
        }
    else:
        query_path, checkpoint = args.output_dir / "held_per_query.csv.gz", args.output_dir / "final_shared_encoder.pt"
        if not query_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Phase-A arm output is incomplete")
        frame = pd.read_csv(query_path)
        gates = {
            "status": report.get("status") == "noise_final_dynamic_direct_phase_a_arm_complete",
            "formal": report.get("formal") is True,
            "held_rows": len(frame) == int(report.get("held", {}).get("n_queries", -1)),
            "ranks": {"official_rank", "initial_rank", "final_rank"} <= set(frame.columns),
            "shared": report.get("contracts", {}).get("shared_query_reference_encoder") is True,
            "full_list": report.get("contracts", {}).get("full_candidate_molecule_list_training") is True,
            "P2b": report.get("contracts", {}).get("P2b") == "forbidden",
        }
    if not all(gates.values()):
        raise RuntimeError(f"Phase-A validation failed: {gates}")
    print(f"[validate_noise_final_dynamic_direct_phase_a] PASS summary={args.summary}")


if __name__ == "__main__":
    main()

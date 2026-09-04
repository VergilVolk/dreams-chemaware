from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    actions = pd.read_csv(args.output_dir / "corrective_teacher_actions.csv.gz")
    safety = pd.read_csv(args.output_dir / "robustness_teacher_actions.csv.gz")
    ledger = pd.read_csv(args.output_dir / "query_ledger.csv.gz")
    if report.get("status") != "noise_final_r1_privileged_teacher_complete" or not report.get("formal"):
        raise RuntimeError("R1 report is not formal")
    if report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("P2b entered the R1 teacher")
    if len(ledger) != 23876 or int(ledger["baseline_rank"].gt(1).sum()) != 1805:
        raise RuntimeError("R1 clean ledger drifted")
    if len(actions) != 882 or actions["query_index"].duplicated().any():
        raise RuntimeError("R1 corrective union drifted")
    if actions["baseline_rank"].le(1).any() or actions["teacher_rank"].ne(1).any():
        raise RuntimeError("R1 includes a false corrective teacher view")
    if set(actions["teacher_source"].astype(str)) != {"s3a_dynamic", "a4_exact"}:
        raise RuntimeError("R1 lost one historical action source")
    forbidden_safety = {"baseline_rank", "target_rank", "corrected", "introduced"}
    if forbidden_safety & set(safety.columns):
        raise RuntimeError("outcome columns leaked into the robustness table")
    if report["historical_full_union_recoverable"] != 920:
        raise RuntimeError("historical oracle reference drifted")
    print(
        "[validate_noise_final_r1_privileged_teacher] PASS "
        f"corrective={len(actions):,} robustness={len(safety):,}"
    )


if __name__ == "__main__":
    main()

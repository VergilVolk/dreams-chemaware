"""Fail-closed validator for the E9 action-staleness audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report_path, table_path = args.output_dir / "report.json", args.output_dir / "per_action.csv.gz"
    if not report_path.is_file() or not table_path.is_file():
        raise FileNotFoundError("E9 report/per-action artifact is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e9_action_staleness_complete":
        raise RuntimeError("unexpected E9 status")
    if not report.get("formal") and not args.allow_smoke:
        raise RuntimeError("formal E9 validation received a smoke artifact")
    expected = {"shared_clean_embedding_checkpoint": True, "current_student_used_for_action_mining": True,
                "outcome_labels_used_for_action_selection": False,
                "candidate_information_training_development_only": True,
                "P2b": "forbidden", "P3_consumed": False}
    if any(report.get("contracts", {}).get(key) != value for key, value in expected.items()):
        raise RuntimeError("E9 contract drift")
    table = pd.read_csv(table_path)
    required = {"query_index", "query_formula", "selector", "step", "frozen_path", "online_path",
                "clean_rank", "frozen_rank", "online_rank"}
    if required - set(table.columns):
        raise RuntimeError(f"E9 table is missing {sorted(required - set(table.columns))}")
    if len(table) != int(report["held_action_rows"]):
        raise RuntimeError("E9 row count drift")
    if int(report.get("mature_e8_rank_reproduction_mismatches", -1)) != 0:
        raise RuntimeError("E9 did not reproduce mature E8 ranks")
    if set(table["selector"].astype(str)) != {"candidate_gradient", "role_confounder"}:
        raise RuntimeError("E9 selector coverage drift")
    if table[["clean_rank", "frozen_rank", "online_rank"]].lt(1).any().any():
        raise RuntimeError("E9 contains invalid ranks")
    print(f"[validate_noise_final_e9_action_staleness] PASS rows={len(table):,} online={report['pass_to_online_remining_training']}", flush=True)


if __name__ == "__main__":
    main()

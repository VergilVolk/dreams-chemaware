"""Fail-closed validation of a formal E15 multi-action ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "report": args.output_dir / "report.json",
        "all": args.output_dir / "all_action_ledger.csv.gz",
        "corrective": args.output_dir / "corrective_actions.csv.gz",
        "harmful": args.output_dir / "harmful_actions.csv.gz",
        "no_op": args.output_dir / "no_op.csv.gz",
        "support": args.output_dir / "conditional_support.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e15_multi_action_ledger_complete"
        or not report.get("formal")
        or not report.get("pass_to_loss_and_sampler_smoke")
        or not all(report.get("gates", {}).values())
    ):
        raise RuntimeError("E15 ledger report did not pass every gate")
    corrective = pd.read_csv(required["corrective"], low_memory=False)
    harmful = pd.read_csv(required["harmful"], low_memory=False)
    no_op = pd.read_csv(required["no_op"], low_memory=False)
    all_actions = pd.read_csv(required["all"], low_memory=False)
    required_columns = {
        "source", "query_index", "query_row", "query_ik14", "query_formula",
        "formula_fold", "action_family", "action_id", "action_payload",
        "supervision_kind", "conditional_key", "replicated_formula_folds",
        "conditional_identities", "margin_delta",
    }
    for name, frame in (("corrective", corrective), ("harmful", harmful), ("all", all_actions)):
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise RuntimeError(f"E15 {name} ledger missing columns: {sorted(missing_columns)}")
        if frame.duplicated(["source", "query_index", "action_id"]).any():
            raise RuntimeError(f"E15 {name} ledger has duplicate actions")
        if frame["formula_fold"].astype(int).eq(int(report["outer_formula_fold"])).any():
            raise RuntimeError(f"E15 {name} ledger contains held formula fold")
    if not corrective["supervision_kind"].eq("corrective").all():
        raise RuntimeError("non-corrective row entered corrective ledger")
    if not harmful["supervision_kind"].eq("harmful").all():
        raise RuntimeError("non-harmful row entered harmful ledger")
    if corrective.groupby("query_index").size().max() > int(report["maximum_corrective_actions_per_query"]):
        raise RuntimeError("corrective exposure cap disagrees with report")
    if harmful.groupby("query_index").size().max() > int(report["maximum_harmful_actions_per_query"]):
        raise RuntimeError("harmful exposure cap disagrees with report")
    if no_op["query_index"].duplicated().any() or not no_op["supervision_kind"].eq("no_op").all():
        raise RuntimeError("no-op ledger is not exactly one row per query")
    if int(report["queries_with_multiple_corrective_actions"]) <= 0:
        raise RuntimeError("E15 did not preserve any multi-action query")
    if report.get("contracts", {}).get("P2b") != "forbidden" or report.get("contracts", {}).get("P3_consumed"):
        raise RuntimeError("E15 violated P2b/P3 isolation")
    print(
        "[validate_noise_final_e15_multi_action_ledger] PASS "
        f"actions={len(all_actions):,} corrective={len(corrective):,} "
        f"harmful={len(harmful):,} no_op={len(no_op):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

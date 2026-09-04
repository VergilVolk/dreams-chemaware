"""Fail-closed validator for a completed E15-M2 shared-encoder capacity run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import torch_load_compat  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "report": args.output_dir / "report.json",
        "checkpoint": args.output_dir / "shared_encoder.pt",
        "corrective": args.output_dir / "selected_corrective_actions.csv.gz",
        "risk_routes": args.output_dir / "selected_risk_actions.csv.gz",
        "risk_queries": args.output_dir / "selected_risk_queries.csv.gz",
        "per_query": args.output_dir / "per_query.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e15_m2_shared_encoder_overfit_complete"
        or not report.get("formal") or not report.get("pass_to_identity_holdout")
        or not all(report.get("gates", {}).values())
    ):
        raise RuntimeError("E15-M2 report is not a passing formal capacity result")
    corrective = pd.read_csv(required["corrective"], low_memory=False)
    risk_routes = pd.read_csv(required["risk_routes"], low_memory=False)
    risk = pd.read_csv(required["risk_queries"], low_memory=False)
    per_query = pd.read_csv(required["per_query"], low_memory=False)
    if corrective.empty or risk.empty or risk_routes.empty or per_query.empty:
        raise RuntimeError("E15-M2 ledgers are empty")
    if corrective.duplicated(["source", "query_index", "action_id"]).any():
        raise RuntimeError("E15-M2 corrective action was duplicated")
    if risk_routes.duplicated(["source", "query_index", "action_id"]).any():
        raise RuntimeError("E15-M2 risk action was duplicated")
    if risk.duplicated(["source", "query_index"]).any():
        raise RuntimeError("E15-M2 repeated one clean risk query")
    if corrective.groupby(["source", "query_index"]).ngroups != 32:
        raise RuntimeError("E15-M2 did not preserve the 32 source-query capacity panel")
    action_counts = corrective.groupby(["source", "query_index"]).size()
    if int(action_counts.max()) != int(report["maximum_corrective_actions_per_query"]):
        raise RuntimeError("E15-M2 maximum action exposure does not reproduce")
    observed_multi = (
        action_counts.ge(2).groupby(level=0).sum().astype(int).to_dict()
    )
    expected_multi = {
        str(key): int(value)
        for key, value in report["multiaction_corrective_queries_by_source"].items()
    }
    if observed_multi != expected_multi:
        raise RuntimeError("E15-M2 multi-action source counts do not reproduce")
    if sum(observed_multi.values()) < int(report["minimum_multiaction_queries_total"]):
        raise RuntimeError("E15-M2 retained too few genuine multi-action queries globally")
    calibration = report.get("gradient_calibration", {}).get("records", [])
    if sum(int(row["microbatches"]) for row in calibration) < 32:
        raise RuntimeError("E15-M2 gradient calibration used fewer than 32 microbatches")
    if sum(int(row["actions"]) for row in calibration) < 128:
        raise RuntimeError("E15-M2 gradient calibration used fewer than 128 actions")
    for epoch in report.get("history", []):
        if int(epoch["corrective_actions"]) != len(corrective) or int(epoch["risk_actions"]) != len(risk):
            raise RuntimeError("E15-M2 epoch exposure drifted")
        if int(epoch["corrective_query_steps"]) != 32 or int(epoch["risk_query_steps"]) != 32:
            raise RuntimeError("E15-M2 did not use exactly one optimizer step per source/query")
    package = torch_load_compat(required["checkpoint"], map_location="cpu")
    if (
        package.get("status") != "noise_final_e15_m2_shared_dreams_encoder"
        or not package.get("inference_clean_only") or package.get("P2b_used")
        or int(package.get("outer_fold", -1)) != int(report["outer_formula_fold"])
    ):
        raise RuntimeError("E15-M2 checkpoint contract failed")
    state = package.get("model_state", {})
    if not state:
        raise RuntimeError("E15-M2 checkpoint has no model state")
    nonfinite = [name for name, value in state.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
    if nonfinite:
        raise RuntimeError(f"E15-M2 checkpoint contains non-finite tensors: {nonfinite[:10]}")
    if int(report["introduced"]) != int(np.sum(
        per_query["risk_panel"].astype(bool)
        & per_query["initial_rank"].astype(int).eq(1)
        & per_query["final_rank"].astype(int).ne(1)
    )):
        raise RuntimeError("E15-M2 introduced count does not reproduce per-query ledger")
    print(
        f"[validate_noise_final_e15_m2_overfit] PASS corrected={report['corrected']} "
        f"introduced={report['introduced']}", flush=True,
    )


if __name__ == "__main__":
    main()

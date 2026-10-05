"""Build the strict N-only action manifest for minimal E4 paired transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from noise_final_core import sha256_file
from noise_final_e4_pmt_core import (
    MATURE_N_CELLS, restrict_corrective_query_scope, route_replayed_actions,
)
from train_noise_final_r2_shared_encoder import parse_controls, parse_path


def arguments() -> argparse.Namespace:
    validation = ROOT / "data/validation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-dir", type=Path, default=validation / "g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--advantage-threshold", type=float, default=0.01)
    parser.add_argument(
        "--corrective-query-scope", choices=("all", "errors"), default="all",
        help="Optionally restrict positive target exposure to current-geometry errors.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def join_r0_to_published_n_ledger(
    r0: pd.DataFrame, ledger: pd.DataFrame, outer_fold: int,
) -> pd.DataFrame:
    """Reproduce normalized ledger rows without assuming duplicated R0 fields."""
    r0 = r0.copy()
    r0["action_id"] = r0.apply(
        lambda row: (
            f"N|{int(row.query_index)}|{row.selector}|"
            f"{float(row.attenuation):.2f}|{int(row.step)}"
        ), axis=1,
    )
    if r0["action_id"].duplicated().any() or ledger["action_id"].duplicated().any():
        raise RuntimeError("R0 or dynamic N ledger contains duplicate stable action ids")
    frame = r0.merge(
        ledger[["action_id", "cell_id", "control_payload"]],
        on="action_id", how="inner", validate="one_to_one",
    )
    expected_r0 = r0.loc[r0["formula_fold"].astype(int).ne(outer_fold)]
    if len(frame) != len(ledger) or len(frame) != len(expected_r0):
        raise RuntimeError("R0 and dynamic N ledger membership do not reproduce exactly")
    expected_cell = frame.apply(
        lambda row: (
            f"{row.selector}|a={float(row.attenuation):.2f}|step={int(row.step)}"
        ), axis=1,
    )
    if not expected_cell.equals(frame["cell_id"].astype(str)):
        raise RuntimeError("dynamic ledger cell_id does not reproduce canonical R0 fields")
    return frame


def contract_mismatches(
    observed: dict[str, object], expected: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Compare contracts by declared value, including required False values."""
    return {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items() if observed.get(key) != value
    }


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be 0..4")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite PMT manifest: {args.output_dir}")
    paths = {
        "r0_report": args.r0_dir / "report.json",
        "r0_actions": args.r0_dir / "training_actions.csv.gz",
        "ledger_report": args.ledger_dir / "report.json",
        "ledger_actions": args.ledger_dir / "training_actions.csv.gz",
        "replay_report": args.replay_dir / "report.json",
        "outcomes": args.replay_dir / "current_geometry_outcomes.csv.gz",
        "initial_checkpoint": args.initial_checkpoint,
    }
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(missing)
    reports = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items() if key.endswith("report")
    }
    if (
        reports["r0_report"].get("formal") is not True
        or reports["ledger_report"].get("formal") is not True
        or reports["replay_report"].get("formal") is not True
        or int(reports["ledger_report"].get("outer_formula_fold", -1)) != args.outer_fold
        or reports["replay_report"].get("optimizer_steps") != 0
        or reports["replay_report"].get("provenance", {}).get("clean_checkpoint_sha256")
        != sha256_file(args.initial_checkpoint)
    ):
        raise RuntimeError("PMT source artifacts are not aligned formal no-update artifacts")
    r0 = pd.read_csv(paths["r0_actions"], low_memory=False)
    ledger = pd.read_csv(paths["ledger_actions"], low_memory=False)
    ledger = ledger.loc[ledger["source"].astype(str).eq("N")].copy()
    outcomes = pd.read_csv(paths["outcomes"], low_memory=False)
    outcomes = outcomes.loc[outcomes["source"].astype(str).eq("N")].copy()
    source_keys = ["query_index", "selector", "attenuation", "step"]
    required_r0 = set(source_keys) | {
        "query_row", "query_ik14", "query_formula", "formula_fold",
        "target_path", "matched_control_paths", "hard_negative_row",
    }
    if missing := required_r0 - set(r0.columns):
        raise RuntimeError(f"R0 lacks PMT columns: {sorted(missing)}")
    # The published dynamic ledger intentionally normalizes N actions and does
    # not repeat selector/attenuation/step. Reconstruct its stable action_id
    # from the canonical R0 fields and join on that identifier.
    required_ledger = {"action_id", "source", "cell_id", "control_payload"}
    if missing := required_ledger - set(ledger.columns):
        raise RuntimeError(f"dynamic ledger lacks N PMT columns: {sorted(missing)}")
    outcome_columns = {
        "action_id", "clean_rank", "target_rank", "control_rank",
        "paired_advantage", "corrected", "introduced",
    }
    if missing := outcome_columns - set(outcomes.columns):
        raise RuntimeError(f"replay lacks PMT outcomes: {sorted(missing)}")
    frame = join_r0_to_published_n_ledger(r0, ledger, args.outer_fold)
    frame = frame.merge(
        outcomes[list(outcome_columns)], on="action_id", how="inner", validate="one_to_one",
    )
    if len(frame) != len(ledger):
        raise RuntimeError("N replay join is incomplete")
    frame["control_path"] = frame["control_payload"].astype(str)
    for row in frame[["target_path", "matched_control_paths", "control_path"]].itertuples(index=False):
        target = parse_path(row.target_path)
        controls = parse_controls(row.matched_control_paths)
        selected = parse_path(row.control_path)
        if selected not in controls or selected == target:
            raise RuntimeError("selected PMT control does not reproduce a frozen R0 control")
    routed = restrict_corrective_query_scope(
        route_replayed_actions(frame, args.advantage_threshold),
        args.corrective_query_scope,
    )
    routing_corrective = routed.loc[routed["route"].eq("corrective")].copy()
    corrective = routing_corrective.copy()
    if corrective.empty or corrective["teacher_advantage"].le(0).any():
        raise RuntimeError("PMT has no strict-positive corrective actions")
    if set(corrective["cell_id"].astype(str)) != MATURE_N_CELLS:
        raise RuntimeError(
            "scoped PMT corrective actions no longer retain all nine mature N cells"
        )
    if set(corrective["formula_fold"].astype(int)) == {args.outer_fold}:
        raise RuntimeError("PMT corrective manifest contains only held fold")
    route_counts = routed["route"].value_counts().astype(int).to_dict()
    report = {
        "status": "noise_final_e4_pmt_manifest_complete",
        "formal": True,
        "outer_formula_fold": args.outer_fold,
        "all_n_actions": int(len(routed)),
        "routing_corrective_actions": int(len(routing_corrective)),
        "corrective_actions": int(len(corrective)),
        "corrective_queries": int(corrective["query_index"].nunique()),
        "corrective_identities": int(corrective["query_ik14"].astype(str).nunique()),
        "corrective_formulas": int(corrective["query_formula"].astype(str).nunique()),
        "route_counts": route_counts,
        "cells": int(routed["cell_id"].nunique()),
        "corrective_query_scope": args.corrective_query_scope,
        "positive_harmful_overlap_policy": "harmful_precedence",
        "contracts": {
            "all_mature_n_actions_routed": True,
            "nine_mature_n_cells_only": True,
            "noncorrective_target_weight_exact_zero": bool(
                routed.loc[~routed["route"].eq("corrective"), "corrective_weight"].eq(0).all()
            ),
            "outer_fold_absent": bool(routed["formula_fold"].astype(int).ne(args.outer_fold).all()),
            "current_geometry_full_candidate_replay": True,
            "selected_control_reproduces_frozen_R0_path": True,
            "M2_predictions_used": False,
            "P_actions_used": False,
            "corrective_target_exposure_errors_only": bool(
                args.corrective_query_scope == "errors"
            ),
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {key: sha256_file(path) for key, path in paths.items()},
        "claim_limit": "Outer-train action routing only; not a trained embedding result.",
    }
    expected_contracts = {
        "all_mature_n_actions_routed": True,
        "nine_mature_n_cells_only": True,
        "noncorrective_target_weight_exact_zero": True,
        "outer_fold_absent": True,
        "current_geometry_full_candidate_replay": True,
        "selected_control_reproduces_frozen_R0_path": True,
        "M2_predictions_used": False,
        "P_actions_used": False,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    contract_failures = contract_mismatches(report["contracts"], expected_contracts)
    if contract_failures:
        raise RuntimeError(f"PMT manifest contract mismatches: {contract_failures}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e4_pmt_manifest_", dir=args.output_dir.parent))
    try:
        routed.to_csv(staging / "all_routed_actions.csv.gz", index=False, compression="gzip")
        corrective.to_csv(staging / "corrective_actions.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

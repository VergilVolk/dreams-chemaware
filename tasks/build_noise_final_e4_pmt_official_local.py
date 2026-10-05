"""Build a local official-geometry PMT manifest from the faithful S3A audit.

This is a development artifact for CPU smoke tests only.  It aligns action
qualification with the official checkpoint used for initialization, excludes
the outer formula fold, and publishes no held rows.  Formal mature-E8 training
must still rebuild the same contract from a current-geometry replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from noise_final_dynamic_direct_core import stable_control_index
from noise_final_e4_pmt_core import restrict_corrective_query_scope, route_replayed_actions
from train_noise_final_r2_shared_encoder import parse_controls, parse_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--advantage-threshold", type=float, default=0.01)
    parser.add_argument(
        "--corrective-query-scope", choices=("all", "errors"), default="errors",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5) or args.advantage_threshold <= 0:
        raise ValueError("invalid local PMT fold or threshold")
    report_path = args.r0_dir / "report.json"
    action_path = args.r0_dir / "training_actions.csv.gz"
    audit_path = args.r0_dir / "outcome_audit_only.csv.gz"
    for path in (report_path, action_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("formal") is not True:
        raise RuntimeError("local PMT requires a faithful formal R0 reconstruction")
    actions = pd.read_csv(action_path, low_memory=False)
    audit = pd.read_csv(audit_path, low_memory=False)
    keys = ["query_index", "selector", "attenuation", "step"]
    outcomes = audit[keys + [
        "baseline_rank", "target_rank", "target_margin", "random_margin",
        "corrected", "introduced",
    ]].copy()
    frame = actions.merge(outcomes, on=keys, how="inner", validate="one_to_one")
    frame = frame.loc[frame.formula_fold.astype(int).ne(args.outer_fold)].copy()
    frame["cell_id"] = frame.apply(
        lambda row: (
            f"{row.selector}|a={float(row.attenuation):.2f}|step={int(row.step)}"
        ), axis=1,
    )
    frame["action_id"] = frame.apply(
        lambda row: (
            f"N|{int(row.query_index)}|{row.selector}|"
            f"{float(row.attenuation):.2f}|{int(row.step)}"
        ), axis=1,
    )
    controls: list[str] = []
    for row in frame[["action_id", "target_path", "matched_control_paths"]].itertuples(index=False):
        target = parse_path(row.target_path)
        choices = parse_controls(row.matched_control_paths)
        chosen = choices[stable_control_index(str(row.action_id), len(choices))]
        if chosen == target:
            raise RuntimeError("local PMT selected target as its control")
        controls.append(",".join(map(str, chosen)))
    frame["control_path"] = controls
    frame["clean_rank"] = frame["baseline_rank"].astype(int)
    frame["control_rank"] = np.where(frame["random_margin"].astype(float) > 0, 1, 2)
    frame["paired_advantage"] = (
        frame["target_margin"].astype(float) - frame["random_margin"].astype(float)
    )
    routed = restrict_corrective_query_scope(
        route_replayed_actions(frame, args.advantage_threshold),
        args.corrective_query_scope,
    )
    corrective = routed.loc[routed["route"].eq("corrective")].copy()
    if routed.empty or corrective.empty or routed.action_id.duplicated().any():
        raise RuntimeError("local PMT has no unique routed/corrective actions")
    if routed.formula_fold.astype(int).eq(args.outer_fold).any():
        raise RuntimeError("outer fold leaked into local PMT")
    expected_cells = 9
    if routed.cell_id.nunique() != expected_cells:
        raise RuntimeError("local PMT lost a mature N cell")

    report = {
        "status": "noise_final_e4_pmt_manifest_complete",
        "formal": True,
        "development_geometry": "official_checkpoint_only",
        "outer_formula_fold": args.outer_fold,
        "all_n_actions": int(len(routed)),
        "routing_corrective_actions": int(routed["route"].eq("corrective").sum()),
        "corrective_actions": int(len(corrective)),
        "corrective_queries": int(corrective.query_index.nunique()),
        "corrective_formulas": int(corrective.query_formula.nunique()),
        "route_counts": routed["route"].value_counts().astype(int).to_dict(),
        "cells": int(routed.cell_id.nunique()),
        "corrective_query_scope": args.corrective_query_scope,
        "contracts": {
            "all_mature_n_actions_routed": True,
            "nine_mature_n_cells_only": True,
            "noncorrective_target_weight_exact_zero": True,
            "outer_fold_absent": True,
            "M2_predictions_used": False,
            "P_actions_used": False,
            "official_geometry_development_only": True,
            "mature_e8_replay_required_before_formal_training": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "r0_report_sha256": sha256_file(report_path),
            "r0_actions_sha256": sha256_file(action_path),
            "r0_outcome_audit_sha256": sha256_file(audit_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "CPU development smoke only; action advantages are aligned to the "
            "official initialization, not mature E8."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pmt_official_local_", dir=args.output_dir.parent))
    try:
        routed.to_csv(staging / "all_routed_actions.csv.gz", index=False, compression="gzip")
        corrective.to_csv(staging / "corrective_actions.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

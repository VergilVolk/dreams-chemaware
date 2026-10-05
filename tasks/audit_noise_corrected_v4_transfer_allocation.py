"""Audit a dose-neutral, rank-preserving replacement for the v3 hard cap.

The audit uses detached action/control/clean margins from the frozen local v3
ledger.  Allocation is performed separately inside each
query/source/family cell, so it cannot change that cell's cumulative transfer
dose or the source/family equalization contract.  This is a scalar screening
proxy for a future v4 arm; it does not train or evaluate an encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from noise_corrected_action_expansion_v4 import mass_matched_monotone_transfer_delta
from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-actions", type=Path,
        default=ROOT / (
            "data/validation/noise_corrected_routed_npa4_ledger_e8_dev_v5_20260907/"
            "training_actions.csv.gz"
        ),
    )
    parser.add_argument("--hard-cap", type=float, default=0.10)
    parser.add_argument("--maximum-cap-factor", type=float, default=2.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/noise_corrected_v4_transfer_allocation_local_20260907",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    actions = pd.read_csv(args.training_actions)
    required = {
        "action_id", "query_index", "source", "family", "supervision_kind",
        "action_margin", "clean_margin", "control_margin",
    }
    missing = required - set(actions)
    if missing:
        raise RuntimeError(f"training ledger misses columns: {sorted(missing)}")
    frame = actions[actions["supervision_kind"] == "corrective"].copy()
    if frame.empty or frame["action_id"].duplicated().any():
        raise RuntimeError("corrective action ledger is empty or duplicated")
    frame["delta_uncapped"] = np.minimum(
        np.maximum(frame["action_margin"] - frame["clean_margin"], 0),
        np.maximum(frame["action_margin"] - frame["control_margin"], 0),
    )
    frame["v3_hard_delta"] = np.minimum(frame["delta_uncapped"], args.hard_cap)
    frame["v4_mass_matched_delta"] = 0.0
    group_columns = ["query_index", "source", "family"]
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        values = torch.as_tensor(group["delta_uncapped"].to_numpy(), dtype=torch.float64)
        transformed = mass_matched_monotone_transfer_delta(
            values, values > 0, hard_cap=args.hard_cap,
            maximum_cap_factor=args.maximum_cap_factor,
        )
        frame.loc[group.index, "v4_mass_matched_delta"] = transformed.numpy()

    group_mass = frame.groupby(group_columns, dropna=False).agg(
        v3_mass=("v3_hard_delta", "sum"),
        v4_mass=("v4_mass_matched_delta", "sum"),
    ).reset_index()
    group_mass["absolute_error"] = np.abs(group_mass["v3_mass"] - group_mass["v4_mass"])
    source_mass = frame.groupby("source").agg(
        rows=("action_id", "size"),
        v3_mass=("v3_hard_delta", "sum"),
        v4_mass=("v4_mass_matched_delta", "sum"),
        v3_mean=("v3_hard_delta", "mean"),
        v4_mean=("v4_mass_matched_delta", "mean"),
    ).reset_index()
    source_mass["absolute_mass_error"] = np.abs(source_mass["v3_mass"] - source_mass["v4_mass"])

    ordering_violations = 0
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        local = group.sort_values("delta_uncapped", kind="mergesort")
        positive = local[local["delta_uncapped"] > 0]["v4_mass_matched_delta"].to_numpy()
        ordering_violations += int(np.sum(np.diff(positive) < -1e-12))
    old_nonzero = frame.loc[frame["v3_hard_delta"] > 0, "v3_hard_delta"].round(8)
    new_nonzero = frame.loc[
        frame["v4_mass_matched_delta"] > 0, "v4_mass_matched_delta"
    ].round(8)
    report = {
        "status": "noise_corrected_v4_transfer_allocation_audit_complete",
        "formal": False,
        "shared_encoder_trained": False,
        "corrective_actions": int(len(frame)),
        "corrective_queries": int(frame["query_index"].nunique()),
        "query_source_family_cells": int(len(group_mass)),
        "v3_hard_cap": args.hard_cap,
        "v3_rows_at_hard_cap": int(np.sum(frame["v3_hard_delta"] >= args.hard_cap - 1e-12)),
        "v3_fraction_at_hard_cap": float(np.mean(
            frame["v3_hard_delta"] >= args.hard_cap - 1e-12
        )),
        "v3_distinct_positive_targets_rounded_1e8": int(old_nonzero.nunique()),
        "v4_distinct_positive_targets_rounded_1e8": int(new_nonzero.nunique()),
        "v4_distinct_target_increase_fraction": float(
            new_nonzero.nunique() / max(old_nonzero.nunique(), 1) - 1
        ),
        "v4_rows_above_old_cap": int(np.sum(
            frame["v4_mass_matched_delta"] > args.hard_cap + 1e-12
        )),
        "v4_maximum_delta": float(frame["v4_mass_matched_delta"].max()),
        "maximum_allowed_delta": args.maximum_cap_factor * args.hard_cap,
        "query_source_family_mass_max_absolute_error": float(group_mass["absolute_error"].max()),
        "source_mass_max_absolute_error": float(source_mass["absolute_mass_error"].max()),
        "within_cell_ordering_violations": ordering_violations,
        "source_summary": source_mass.to_dict("records"),
        "gates": {
            "exact_cell_mass_preservation": bool(group_mass["absolute_error"].max() <= 1e-10),
            "exact_source_mass_preservation": bool(source_mass["absolute_mass_error"].max() <= 1e-10),
            "within_cell_order_preserved": ordering_violations == 0,
            "maximum_edge_cap_respected": bool(
                frame["v4_mass_matched_delta"].max()
                <= args.maximum_cap_factor * args.hard_cap + 1e-12
            ),
            "positive_target_resolution_increased_ge_20pct": bool(
                new_nonzero.nunique() >= 1.20 * old_nonzero.nunique()
            ),
        },
        "contracts": {
            "allocation_scope": "query_source_family",
            "total_optimizer_dose_increased": False,
            "learning_rate_or_epoch_scan": False,
            "running_v3_implementation_modified": False,
            "detached_action_margins_only": True,
        },
        "provenance": {
            "training_actions_sha256": sha256_file(args.training_actions),
            "action_core_sha256": sha256_file(
                ROOT / "tasks/noise_corrected_action_expansion_v4.py"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Development-ledger scalar allocation audit. Actual per-candidate-edge "
            "gradient and held shared-encoder improvement remain untested."
        ),
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"v4 transfer allocation audit failed: {report['gates']}")
    staging = Path(tempfile.mkdtemp(prefix="noise_v4_transfer_", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "per_action.csv.gz", index=False)
        group_mass.to_csv(staging / "per_query_source_family_mass.csv.gz", index=False)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

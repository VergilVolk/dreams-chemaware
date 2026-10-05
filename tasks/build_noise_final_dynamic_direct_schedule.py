"""Materialize one arm-invariant, bounded Phase-A action schedule.

The schedule fixes query/action membership and order once.  Phase-A arms may
change only the view payload and the registered loss weight; they cannot draw
different examples or recycle an action within an epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_dynamic_direct_core import PHASE_A_ARMS, stratified_action_schedule


def json_default(value: object) -> object:
    """Convert NumPy scalar report values without weakening JSON validation."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_exposure(base: np.ndarray, target_mean: float, maximum: float) -> np.ndarray:
    """Monotonically rescale positive query utilities to a fixed mean dose."""
    base = np.asarray(base, dtype=np.float64)
    if not len(base) or np.any(base <= 0) or not 0 < target_mean <= maximum < 1:
        raise ValueError("invalid conditional exposure calibration inputs")
    low, high = 0.0, 1.0
    while float(np.mean(np.minimum(base * high, maximum))) < target_mean:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("could not calibrate action exposure")
    for _ in range(80):
        middle = (low + high) / 2.0
        if float(np.mean(np.minimum(base * middle, maximum))) < target_mean:
            low = middle
        else:
            high = middle
    result = np.minimum(base * high, maximum)
    if abs(float(result.mean()) - target_mean) > 1e-8:
        raise RuntimeError("action exposure calibration missed its target")
    return result


def assign_epoch_weights(
    part: pd.DataFrame, full_query_mass: pd.Series, target_mean: float, maximum: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Match conditional and static total dose while changing allocation only."""
    part = part.copy()
    queries = part["query_index"].drop_duplicates().to_numpy(np.int64)
    missing = set(map(int, queries)) - set(map(int, full_query_mass.index))
    if missing:
        raise RuntimeError(f"scheduled queries miss full-ledger exposure: {len(missing)}")
    base = full_query_mass.loc[queries].to_numpy(np.float64)
    desired = scaled_exposure(base, target_mean, maximum)
    desired_by_query = dict(zip(map(int, queries), map(float, desired)))
    for column in (
        "dynamic_weight", "static_weight", "dynamic_selected_no_op_weight",
        "static_selected_no_op_weight",
    ):
        part[column] = 0.0
    for query, block in part.groupby("query_index", sort=False):
        index = block.index
        target_mass = desired_by_query[int(query)]
        raw = block["m2_dynamic_weight"].to_numpy(np.float64)
        if not np.any(raw > 0):
            raise RuntimeError(f"query {query} has no positive conditional action utility")
        part.loc[index, "dynamic_weight"] = target_mass * raw / float(raw.sum())
        families = block["family"].astype(str)
        family_names = sorted(families.unique())
        for family in family_names:
            family_index = index[families.eq(family)]
            part.loc[family_index, "static_weight"] = (
                target_mass / len(family_names) / len(family_index)
            )
        no_op = 1.0 - target_mass
        part.loc[index, "dynamic_selected_no_op_weight"] = no_op
        part.loc[index, "static_selected_no_op_weight"] = no_op
    dynamic_mass = part.groupby("query_index")["dynamic_weight"].sum().sort_index()
    static_mass = part.groupby("query_index")["static_weight"].sum().sort_index()
    report = {
        "epoch": int(part["epoch"].iloc[0]), "actions": int(len(part)),
        "queries": int(part["query_index"].nunique()),
        "cells": int(part["cell_id"].nunique()),
        "mean_action_mass": float(dynamic_mass.mean()),
        "maximum_action_mass": float(dynamic_mass.max()),
        "minimum_no_op_mass": float(1.0 - dynamic_mass.max()),
        "static_dynamic_mass_max_abs_difference": float(
            np.max(np.abs(dynamic_mass.to_numpy() - static_mass.to_numpy()))
        ),
    }
    return part, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--actions-per-identity-family", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--action-dose-curriculum", type=float, nargs="+",
        default=[0.35, 0.45, 0.55, 0.55],
    )
    parser.add_argument("--maximum-action-dose", type=float, default=0.65)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite Phase-A schedule: {args.output_dir}")
    report_path = args.ledger_dir / "report.json"
    actions_path = args.ledger_dir / "training_actions.csv.gz"
    if not report_path.is_file() or not actions_path.is_file():
        raise FileNotFoundError("completed dynamic-direct ledger is required")
    ledger_report = json.loads(report_path.read_text(encoding="utf-8"))
    accepted = {
        "noise_final_dynamic_direct_action_ledger_complete": "pass_to_gpu_replay",
        "noise_final_dynamic_direct_m2_current_crossfit_complete": "pass_to_schedule",
    }
    status = str(ledger_report.get("status"))
    if (status not in accepted or ledger_report.get("formal") is not True
            or ledger_report.get(accepted.get(status, "")) is not True):
        raise RuntimeError("dynamic-direct ledger has not passed")
    actions = pd.read_csv(actions_path, low_memory=False)
    required = {
        "action_id", "query_index", "identity", "formula", "family",
        "target_payload", "control_payload", "dynamic_weight", "static_weight",
        "dynamic_no_op_weight", "static_no_op_weight",
    }
    if missing := required - set(actions.columns):
        raise RuntimeError(f"schedule input misses columns: {sorted(missing)}")

    if (args.epochs < 1 or len(args.action_dose_curriculum) != args.epochs
            or not 0 < args.maximum_action_dose < 1
            or any(not 0 < value <= args.maximum_action_dose for value in args.action_dose_curriculum)):
        raise ValueError("invalid epoch action-dose curriculum")
    schedule = stratified_action_schedule(
        actions, args.seed, args.epochs, args.actions_per_identity_family,
    )
    schedule = schedule.rename(columns={"dynamic_weight": "m2_dynamic_weight"})
    full_query_mass = actions.groupby("query_index", sort=False)["dynamic_weight"].sum()

    epoch_reports: list[dict[str, object]] = []
    weighted_parts: list[pd.DataFrame] = []
    for epoch, part in schedule.groupby("epoch", sort=True):
        part, epoch_report = assign_epoch_weights(
            part, full_query_mass,
            float(args.action_dose_curriculum[int(epoch) - 1]),
            args.maximum_action_dose,
        )
        epoch_reports.append(epoch_report)
        weighted_parts.append(part)
    schedule = pd.concat(weighted_parts, ignore_index=True)
    schedule.insert(0, "schedule_index", np.arange(len(schedule), dtype=np.int64))
    if schedule.duplicated(["epoch", "action_id"]).any():
        raise RuntimeError("schedule recycled an action within an epoch")
    membership = [
        f"{epoch}|{action}" for epoch, action in
        schedule[["epoch", "action_id"]].itertuples(index=False, name=None)
    ]
    arm_manifest = pd.DataFrame({
        "arm": list(PHASE_A_ARMS),
        "membership_sha256": [
            hashlib.sha256("\n".join(membership).encode("utf-8")).hexdigest()
        ] * len(PHASE_A_ARMS),
        "scheduled_actions": [len(schedule)] * len(PHASE_A_ARMS),
        "payload": ["clean", "control_payload", "target_payload", "target_payload"],
        "weight": ["zero", "dynamic_weight", "static_weight", "dynamic_weight"],
    })
    per_identity_family = schedule.groupby(["epoch", "identity", "family"]).size()
    dynamic_mass = schedule.groupby(["epoch", "query_index"])["dynamic_weight"].sum()
    static_mass = schedule.groupby(["epoch", "query_index"])["static_weight"].sum()
    risk_summary = []
    for family, block in schedule.groupby("family", sort=True):
        weight = block["dynamic_weight"].to_numpy(np.float64)
        risk = block["risk"].to_numpy(np.float64)
        risk_summary.append({
            "family": str(family), "scheduled_rows": int(len(block)),
            "dynamic_weight_sum": float(weight.sum()),
            "risk_weighted_mass": float(np.sum(weight * risk)),
            "risk_weighted_mean": float(np.sum(weight * risk) / max(weight.sum(), 1e-12)),
        })
    gates = {
        "all_four_arms_present": set(arm_manifest["arm"]) == set(PHASE_A_ARMS),
        "arm_membership_and_order_identical": arm_manifest["membership_sha256"].nunique() == 1,
        "without_replacement_within_epoch": not schedule.duplicated(["epoch", "action_id"]).any(),
        "identity_family_cap": int(per_identity_family.max()) <= args.actions_per_identity_family,
        "all_30_cells_retained": schedule["cell_id"].nunique() == 30,
        "all_30_cells_each_epoch": bool(schedule.groupby("epoch")["cell_id"].nunique().eq(30).all()),
        "explicit_dynamic_no_op": schedule["dynamic_selected_no_op_weight"].gt(0).all(),
        "explicit_static_no_op": schedule["static_selected_no_op_weight"].gt(0).all(),
        "selected_dynamic_mass_exact": bool(np.allclose(
            dynamic_mass.to_numpy()
            + schedule.groupby(["epoch", "query_index"])["dynamic_selected_no_op_weight"].first().to_numpy(),
            1.0, atol=1e-6,
        )),
        "selected_static_mass_exact": bool(np.allclose(
            static_mass.to_numpy()
            + schedule.groupby(["epoch", "query_index"])["static_selected_no_op_weight"].first().to_numpy(),
            1.0, atol=1e-6,
        )),
        "static_matches_dynamic_query_dose": bool(np.allclose(
            dynamic_mass.sort_index().to_numpy(), static_mass.sort_index().to_numpy(), atol=1e-7,
        )),
        "epoch_curriculum_exact": bool(all(
            abs(float(row["mean_action_mass"]) - float(args.action_dose_curriculum[int(row["epoch"]) - 1])) < 1e-6
            for row in epoch_reports
        )),
        "outer_fold_consistent": schedule["formula_fold"].nunique() == 4,
    }
    report = {
        "status": "noise_final_dynamic_direct_phase_a_schedule_complete",
        "formal": True,
        "outer_formula_fold": int(ledger_report["outer_formula_fold"]),
        "scheduled_actions": int(len(schedule)),
        "queries": int(schedule["query_index"].nunique()),
        "identities": int(schedule["identity"].nunique()),
        "formulas": int(schedule["formula"].nunique()),
        "cells": int(schedule["cell_id"].nunique()),
        "family_counts": schedule["family"].value_counts().to_dict(),
        "maximum_identity_family_exposure": int(per_identity_family.max()),
        "epochs": int(args.epochs), "epoch_exposure": epoch_reports,
        "risk_exposure_by_family": risk_summary,
        "gates": gates,
        "contracts": {
            "one_schedule_for_all_arms": True,
            "no_within_epoch_recycling": True, "epoch_cell_cycling": True,
            "matched_random_uses_dynamic_weight": True,
            "clean_arm_uses_same_membership_with_clean_payload": True,
            "static_matches_conditional_query_dose": True,
            "conditional_policy_frozen_across_training": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            "ledger_report": sha256_file(report_path),
            "ledger_actions": sha256_file(actions_path),
            "script": sha256_file(Path(__file__)),
        },
        "pass_to_gpu_replay": bool(all(gates.values())),
    }
    if not report["pass_to_gpu_replay"]:
        raise RuntimeError(f"Phase-A schedule gates failed: {gates}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dynamic_direct_schedule_", dir=args.output_dir.parent))
    try:
        schedule.to_csv(staging / "epoch_schedule.csv.gz", index=False, compression="gzip")
        arm_manifest.to_csv(staging / "arm_manifest.csv", index=False)
        serialized_report = json.dumps(report, indent=2, default=json_default)
        (staging / "report.json").write_text(serialized_report, encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(serialized_report, flush=True)


if __name__ == "__main__":
    main()

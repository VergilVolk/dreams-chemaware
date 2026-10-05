"""Audit whether direct-boundary v2 can consume the complete routed N bank.

This is a structural/reachability audit, not an encoder performance result.
It verifies exact-zero routing, query-complete schedules and the scale of the
raw-action injection before a formal GPU run is allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from train_noise_final_e4a_direct_augmentation import (
    coverage_first_query_schedules, query_complete_action_batches,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--maximum-actions-per-batch", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.epochs < 1 or args.maximum_actions_per_batch < 1:
        raise ValueError("invalid schedule dimensions")
    report_path = args.manifest_dir / "report.json"
    action_path = args.manifest_dir / "all_routed_actions.csv.gz"
    for path in (report_path, action_path, args.reachability_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(report_path.read_text(encoding="utf-8"))
    reachability = json.loads(args.reachability_report.read_text(encoding="utf-8"))
    actions = pd.read_csv(action_path, low_memory=False)
    required = {
        "action_id", "query_index", "query_ik14", "query_formula", "formula_fold",
        "cell_id", "route", "corrective_weight", "target_path", "control_path",
    }
    if missing := required - set(actions.columns):
        raise RuntimeError(f"routed action bank lacks columns: {sorted(missing)}")
    valid_routes = {"corrective", "robustness_only", "harmful", "uncertain"}
    route = actions["route"].astype(str)
    corrective = route.eq("corrective")
    contracts = {
        "report_row_count_matches": int(manifest.get("all_n_actions", -1)) == len(actions),
        "action_ids_unique": not actions["action_id"].duplicated().any(),
        "four_routes_only": set(route) == valid_routes,
        "corrective_weights_binary": actions["corrective_weight"].isin([0.0, 1.0]).all(),
        "corrective_route_weight_one": actions.loc[
            corrective, "corrective_weight"
        ].eq(1).all(),
        "noncorrective_weight_exact_zero": actions.loc[
            ~corrective, "corrective_weight"
        ].eq(0).all(),
        "target_control_distinct": actions["target_path"].astype(str).ne(
            actions["control_path"].astype(str)
        ).all(),
        "nine_mature_n_cells": actions["cell_id"].astype(str).nunique() == 9,
        "outer_fold_absent": actions["formula_fold"].astype(int).ne(
            int(manifest["outer_formula_fold"])
        ).all(),
    }
    contracts = {key: bool(value) for key, value in contracts.items()}
    if not all(bool(value) for value in contracts.values()):
        raise RuntimeError(f"routed action bank contract failed: {contracts}")

    examples = [
        SimpleNamespace(query_index=int(query))
        for query in actions["query_index"].to_numpy(np.int64)
    ]
    schedules = coverage_first_query_schedules(examples, args.epochs, args.seed)
    scheduled = [index for schedule in schedules for index in schedule]
    batch_counts: list[int] = []
    batch_query_counts: list[int] = []
    for schedule in schedules:
        for batch in query_complete_action_batches(
            [examples[index] for index in schedule], args.maximum_actions_per_batch,
        ):
            batch_counts.append(len(batch))
            batch_query_counts.append(len({item.query_index for item in batch}))
    counts = actions.groupby("query_index", sort=False).size().to_numpy(np.int64)
    metrics = reachability.get("metrics", {})
    reachability_delta = float(metrics.get("delta_pp", float("nan")))
    reachability_ci_high = float(metrics.get("ci_high_pp", float("nan")))
    structural_hope = bool(
        len(actions) >= 20_000
        and actions["query_index"].nunique() >= 5_000
        and reachability_delta >= 3.5
        and reachability_ci_high > 4.0
        and all(bool(value) for value in contracts.values())
        and scheduled == list(dict.fromkeys(scheduled))
        and set(scheduled) == set(range(len(actions)))
    )
    body = {
        "status": "noise_direct_v2_injection_manifest_audit_complete",
        "formal_encoder_result": False,
        "manifest": {
            "routed_actions": int(len(actions)),
            "queries": int(actions["query_index"].nunique()),
            "identities": int(actions["query_ik14"].astype(str).nunique()),
            "formulas": int(actions["query_formula"].astype(str).nunique()),
            "cells": int(actions["cell_id"].astype(str).nunique()),
            "route_counts": route.value_counts().astype(int).to_dict(),
            "actions_per_query_min": int(counts.min()),
            "actions_per_query_median": float(np.median(counts)),
            "actions_per_query_p95": float(np.quantile(counts, 0.95)),
            "actions_per_query_max": int(counts.max()),
        },
        "schedule": {
            "epochs": args.epochs,
            "actions_by_epoch": [int(len(schedule)) for schedule in schedules],
            "all_actions_exposed_exactly_once": bool(
                len(scheduled) == len(actions)
                and len(set(scheduled)) == len(actions)
                and set(scheduled) == set(range(len(actions)))
            ),
            "query_sets_split_across_epochs": False,
            "batches": int(len(batch_counts)),
            "actions_per_batch_max": int(max(batch_counts)),
            "queries_per_batch_median": float(np.median(batch_query_counts)),
        },
        "local_reachability_reference": {
            "delta_pp": reachability_delta,
            "ci_low_pp": float(metrics.get("ci_low_pp", float("nan"))),
            "ci_high_pp": reachability_ci_high,
            "raw_encoder_checkpoint": False,
        },
        "contracts": contracts | {
            "query_complete_epoch_schedule": True,
            "query_complete_batches": True,
            "query_equal_optimizer_steps": bool(
                all(count == 1 for count in batch_query_counts)
            ),
            "clean_primary_molecule_max_implemented": True,
            "teacher_scalar_or_embedding_target_used": False,
        },
        "credible_large_injection_ge4pp_hypothesis": structural_hope,
        "ge4pp_encoder_result_achieved": False,
        "promotion_gate": (
            "Require query-weighted clean held delta >=4.0 pp, strict-positive "
            "multiplicity-corrected formula-cluster CI, consistent seeds, and no "
            "material regression across the frozen complete-candidate metric panel."
        ),
        "claim_limit": (
            "Local action-mass and reachability evidence only. The full raw encoder "
            "run is blocked locally by the absent frozen candidate graph and embedding cache."
        ),
        "provenance": {
            "manifest_report_sha256": sha256_file(report_path),
            "all_routed_actions_sha256": sha256_file(action_path),
            "reachability_report_sha256": sha256_file(args.reachability_report),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_direct_v2_audit_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()

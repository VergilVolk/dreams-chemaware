"""Repair the frozen Stage-3 triplet bank without rebuilding its science.

The original Stage-3 bank already contains the useful scientific objects:
same-identity easy/medium/hard positives, the current closest different-
identity negative, the targeted action spectrum, and its registered matched
control.  The zero-update audit showed that row multiplicity and action rows
without targeted positive evidence dominated the continuation.  This module
therefore performs only two operations on those frozen artifacts:

1. retain at most one action relation per query, selected only when the
   targeted action has both a larger same-identity similarity and a larger
   triplet margin than its matched control; and
2. rebuild the native event registry with exactly one clean event for every
   retained action query plus action-free protection events.

No spectrum, positive, negative, action recipe, loss, or held outcome is
recomputed or changed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_noise_dreams_native_official_replay import load_npz
from build_noise_dreams_native_residual_stage2 import (
    assert_exact_checkpoint_reconstruction,
    sha256_file,
)
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from evaluate_noise_dreams_native import encode_rows
from train_e1_identity import load_base_model


STAGE3_REPAIR_BUILDER_VERSION = "noise_native_stage3_calibrated_query_dose_v1"
STAGE3_REPAIR_STATUS = "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_REPAIR_COMPLETE"
CALIBRATED_GAIN_FLOOR = 1e-4
HDF5 = 0
ACTION = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-triplets", type=Path, required=True)
    parser.add_argument("--zero-update-audit", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-action-events", type=int, default=1500)
    parser.add_argument("--protection-events", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def fixed_unicode(values: object) -> np.ndarray:
    body = [str(value) for value in values]
    width = max((len(value) for value in body), default=1)
    return np.asarray(body, dtype=f"<U{width}")


def select_calibrated_events(geometry: pd.DataFrame) -> pd.DataFrame:
    """Choose one frozen relation per query using pre-optimizer evidence only."""
    required = {
        "query_index", "query_formula", "query_ik14", "source", "family",
        "difficulty_tier", "action_id", "stage1_action_index",
        "action_positive_row", "action_negative_row",
        "targeted_positive_similarity", "control_positive_similarity",
        "targeted_margin", "control_margin", "targeted_hinge_active",
        "clean_same_boundary_margin",
    }
    if missing := required - set(geometry.columns):
        raise RuntimeError(f"Stage-3 repair geometry lacks {sorted(missing)}")
    body = geometry.copy()
    body["stage3_action_index"] = np.arange(len(body), dtype=np.int64)
    body["targeted_positive_gain"] = (
        body["targeted_positive_similarity"].astype(float)
        - body["control_positive_similarity"].astype(float)
    )
    body["targeted_margin_advantage"] = (
        body["targeted_margin"].astype(float)
        - body["control_margin"].astype(float)
    )
    body["targeted_clean_margin_advantage"] = (
        body["targeted_margin"].astype(float)
        - body["clean_same_boundary_margin"].astype(float)
    )
    body["minimum_registered_advantage"] = body[[
        "targeted_positive_gain", "targeted_margin_advantage",
        "targeted_clean_margin_advantage",
    ]].min(axis=1)
    active = body["targeted_hinge_active"]
    active_mask = (
        active.astype(bool)
        if pd.api.types.is_bool_dtype(active.dtype)
        else active.astype(str).str.lower().eq("true")
    )
    body = body[
        active_mask
        & (body["targeted_positive_gain"] > CALIBRATED_GAIN_FLOOR)
        & (body["targeted_margin_advantage"] > CALIBRATED_GAIN_FLOOR)
        & (body["targeted_clean_margin_advantage"] > CALIBRATED_GAIN_FLOOR)
    ].copy()
    if body.empty:
        raise RuntimeError("Stage-3 repair found no targeted-specific relation")

    # Semantic aliases must not become repeated evidence.  The most specific
    # member is retained before applying the one-query/one-opportunity rule.
    body = body.sort_values(
        [
            "minimum_registered_advantage", "targeted_margin_advantage",
            "targeted_positive_gain", "targeted_margin", "action_id",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    )
    body = body.drop_duplicates(
        ["query_index", "action_positive_row", "action_negative_row"],
        keep="first",
    )
    selected = body.drop_duplicates("query_index", keep="first").copy()
    selected = selected.sort_values(
        ["query_index", "difficulty_tier", "action_id"], kind="stable",
    ).reset_index(drop=True)
    return selected


def compact_native_pool(
    pool: dict[str, np.ndarray], selected: pd.DataFrame, protection_events: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Subset events and compact registry indices without changing memberships."""
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    queries = np.asarray(pool["event_query"], dtype=np.int64)
    actions = np.asarray(pool["event_action_index"], dtype=np.int64)
    old_action_positions = {
        int(actions[position]): int(position)
        for position in np.flatnonzero(kinds == 2)
    }
    requested_old_actions = list(map(int, selected["stage3_action_index"]))
    if len(set(requested_old_actions)) != len(requested_old_actions):
        raise RuntimeError("Stage-3 repair selected an action twice")
    if not set(requested_old_actions) <= set(old_action_positions):
        raise RuntimeError("Stage-3 repair selected an unknown action event")

    selected_queries = set(map(int, selected["query_index"]))
    original_action_queries = set(map(int, queries[kinds == 2]))
    first_clean_by_query: dict[int, int] = {}
    for position in np.flatnonzero(kinds == 0):
        first_clean_by_query.setdefault(int(queries[position]), int(position))
    if not selected_queries <= set(first_clean_by_query):
        raise RuntimeError("Stage-3 repair lacks a clean event for an action query")

    protection_queries = sorted(
        query for query in first_clean_by_query if query not in original_action_queries
    )[:protection_events]
    if len(protection_queries) != protection_events:
        raise RuntimeError("Stage-3 repair lacks action-free protection events")

    action_positions = [old_action_positions[action] for action in requested_old_actions]
    clean_positions = [first_clean_by_query[query] for query in sorted(selected_queries)]
    protection_positions = [first_clean_by_query[query] for query in protection_queries]
    keep_positions = action_positions + clean_positions + protection_positions

    referenced_registry: list[int] = []
    seen_registry: set[int] = set()
    for position in keep_positions:
        p0, p1 = map(int, pool["positive_ptr"][position:position + 2])
        n0, n1 = map(int, pool["negative_ptr"][position:position + 2])
        references = [
            int(pool["anchor_idx"][position]),
            *map(int, pool["positive_idx"][p0:p1]),
            *map(int, pool["negative_idx"][n0:n1]),
        ]
        for registry_index in references:
            if registry_index not in seen_registry:
                seen_registry.add(registry_index)
                referenced_registry.append(registry_index)
    registry_remap = {
        old: new for new, old in enumerate(referenced_registry)
    }
    action_remap = {
        old: new for new, old in enumerate(requested_old_actions)
    }
    registry_kind = np.asarray(pool["registry_kind"], dtype=np.int8)[
        referenced_registry
    ].copy()
    registry_source = np.asarray(pool["registry_source_index"], dtype=np.int64)[
        referenced_registry
    ].copy()
    for position in np.flatnonzero(registry_kind == ACTION):
        old_action = int(registry_source[position])
        if old_action not in action_remap:
            raise RuntimeError("unused action spectrum survived registry compaction")
        registry_source[position] = action_remap[old_action]

    anchors: list[int] = []
    positive: list[int] = []
    negative: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    event_kind: list[int] = []
    event_query: list[int] = []
    event_action: list[int] = []
    event_formula: list[str] = []
    for position in keep_positions:
        p0, p1 = map(int, pool["positive_ptr"][position:position + 2])
        n0, n1 = map(int, pool["negative_ptr"][position:position + 2])
        anchors.append(registry_remap[int(pool["anchor_idx"][position])])
        positive.extend(
            registry_remap[int(value)] for value in pool["positive_idx"][p0:p1]
        )
        negative.extend(
            registry_remap[int(value)] for value in pool["negative_idx"][n0:n1]
        )
        positive_ptr.append(len(positive))
        negative_ptr.append(len(negative))
        kind = int(kinds[position])
        old_action = int(actions[position])
        event_kind.append(kind)
        event_query.append(int(queries[position]))
        event_action.append(action_remap[old_action] if kind == 2 else -1)
        event_formula.append(str(pool["event_formula"][position]))

    compact = {
        "registry_kind": registry_kind,
        "registry_source_index": registry_source,
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positive, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negative, dtype=np.int64),
        "event_kind": np.asarray(event_kind, dtype=np.int8),
        "event_query": np.asarray(event_query, dtype=np.int64),
        "event_action_index": np.asarray(event_action, dtype=np.int64),
        "event_formula": fixed_unicode(event_formula),
    }
    return compact, {
        "action_events": len(action_positions),
        "action_queries": len(selected_queries),
        "clean_events_for_action_queries": len(clean_positions),
        "action_free_protection_events": len(protection_positions),
        "total_events": len(keep_positions),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.minimum_action_events < 1 or args.protection_events < 4
        or args.batch_size < 1
    ):
        raise RuntimeError("invalid Stage-3 repair scale")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-3 repair clean-boundary audit requires an allocated GPU")
    stage3 = args.stage3_triplets
    audit = args.zero_update_audit
    required = (
        stage3 / "report.json", stage3 / "train_pool.npz",
        stage3 / "validation_pool.npz", stage3 / "action_spectra.npz",
        stage3 / "selected_actions.csv.gz", audit / "report.json",
        audit / "action_geometry.csv.gz",
        args.data, args.warm_start_checkpoint, args.architecture_checkpoint,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    stage3_report = json.loads((stage3 / "report.json").read_text(encoding="utf-8"))
    audit_report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    if stage3_report.get("status") != "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_COMPLETE":
        raise RuntimeError("repair input is not the frozen Stage-3 bank")
    if (
        audit_report.get("status") != "NOISE_DREAMS_NATIVE_STAGE3_ZERO_UPDATE_AUDIT_COMPLETE"
        or audit_report.get("provenance", {}).get("triplet_report_sha256")
        != sha256_file(stage3 / "report.json")
        or audit_report.get("provenance", {}).get("selected_actions_sha256")
        != sha256_file(stage3 / "selected_actions.csv.gz")
        or audit_report.get("artifacts", {}).get("action_geometry_sha256")
        != sha256_file(audit / "action_geometry.csv.gz")
        or audit_report.get("provenance", {}).get("warm_start_checkpoint_sha256")
        != sha256_file(args.warm_start_checkpoint)
    ):
        raise RuntimeError("zero-update audit is not bound to the frozen Stage-3 bank")

    original = pd.read_csv(stage3 / "selected_actions.csv.gz", low_memory=False)
    geometry = pd.read_csv(audit / "action_geometry.csv.gz", low_memory=False)
    if len(original) != len(geometry) or not np.array_equal(
        original["action_id"].astype(str), geometry["action_id"].astype(str)
    ):
        raise RuntimeError("Stage-3 geometry rows are not aligned with actions")

    # Restore the successful Stage-2 selection invariant on the exact frozen
    # Stage-3 positive/negative boundary: targeted must also beat the clean
    # query on that same relation. Only these measured rows are encoded.
    measured_rows = np.unique(np.concatenate([
        geometry["query_row"].to_numpy(np.int64),
        geometry["action_positive_row"].to_numpy(np.int64),
        geometry["action_negative_row"].to_numpy(np.int64),
    ]))
    model, initialization_kind = load_base_model(
        args.warm_start_checkpoint, args.architecture_checkpoint,
        torch.device("cuda"), 100,
    )
    assert_exact_checkpoint_reconstruction(model, args.warm_start_checkpoint)
    preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    measured_embeddings = encode_rows(
        model, measured_rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label="noise-stage3-repair-clean-boundaries",
    )
    del model
    row_position = {int(row): index for index, row in enumerate(measured_rows)}
    clean = measured_embeddings[[
        row_position[int(row)] for row in geometry["query_row"]
    ]]
    positive = measured_embeddings[[
        row_position[int(row)] for row in geometry["action_positive_row"]
    ]]
    negative = measured_embeddings[[
        row_position[int(row)] for row in geometry["action_negative_row"]
    ]]
    geometry["clean_same_boundary_margin"] = (
        np.sum(clean * positive, axis=1) - np.sum(clean * negative, axis=1)
    )
    del measured_embeddings, clean, positive, negative
    selected = select_calibrated_events(geometry)
    if len(selected) < args.minimum_action_events:
        raise RuntimeError(
            f"calibrated Stage-3 relations {len(selected)} < {args.minimum_action_events}"
        )

    pool, dose = compact_native_pool(
        load_npz(stage3 / "train_pool.npz"), selected, args.protection_events,
    )
    bank = load_npz(stage3 / "action_spectra.npz")
    old_indices = selected["stage3_action_index"].to_numpy(np.int64)
    targeted = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)[old_indices]
    control = np.asarray(bank["control_action_spectra"], dtype=np.float32)[old_indices]
    if targeted.shape != control.shape or len(targeted) != len(selected):
        raise RuntimeError("Stage-3 repair action tensors are not aligned")

    tiers = Counter(selected["difficulty_tier"].astype(str))
    sources = set(selected["source"].astype(str))
    original_sources = set(original["source"].astype(str))
    action_mask = np.asarray(pool["event_kind"], dtype=np.int8) == 2
    clean_mask = np.asarray(pool["event_kind"], dtype=np.int8) == 0
    action_queries = np.asarray(pool["event_query"], dtype=np.int64)[action_mask]
    clean_queries = set(map(int, np.asarray(pool["event_query"], dtype=np.int64)[clean_mask]))
    gates = {
        "frozen_stage3_relations_reused_without_remining": True,
        "no_held_outcome_used_for_selection": True,
        "minimum_action_scale": len(selected) >= args.minimum_action_events,
        "exactly_one_action_event_per_query": (
            len(action_queries) == len(set(map(int, action_queries)))
        ),
        "every_action_query_has_one_clean_event": set(map(int, action_queries)) <= clean_queries,
        "all_selected_targeted_positive_similarities_exceed_control": bool(
            (selected["targeted_positive_gain"] > CALIBRATED_GAIN_FLOOR).all()
        ),
        "all_selected_targeted_margins_exceed_control": bool(
            (selected["targeted_margin_advantage"] > CALIBRATED_GAIN_FLOOR).all()
        ),
        "all_selected_targeted_margins_exceed_clean_same_boundary": bool(
            (
                selected["targeted_clean_margin_advantage"]
                > CALIBRATED_GAIN_FLOOR
            ).all()
        ),
        "no_duplicate_query_positive_negative_boundary": not bool(
            selected.duplicated(
                ["query_index", "action_positive_row", "action_negative_row"]
            ).any()
        ),
        "easy_medium_and_hard_advantages_retained_at_scale": bool(
            set(tiers) == {"easy", "medium", "hard"}
            and min(tiers.values()) >= max(50, int(0.02 * len(selected)))
        ),
        "registered_source_closure_retained": sources == original_sources,
        "matched_target_control_tensors_retained": bool(
            np.all(np.any(targeted != control, axis=(1, 2)))
        ),
        "action_free_protection_events_retained": (
            dose["action_free_protection_events"] == args.protection_events
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-3 calibrated repair gates failed: {gates}")

    report = {
        "status": STAGE3_REPAIR_STATUS,
        "builder_version": STAGE3_REPAIR_BUILDER_VERSION,
        "initialization_kind": initialization_kind,
        "repair_scope": (
            "Frozen Stage-3 relations; targeted-over-control positive and margin "
            "evidence; one action opportunity and one clean event per query."
        ),
        "original_action_events": int(len(original)),
        "selected_action_events": int(len(selected)),
        "selected_action_queries": int(selected["query_index"].nunique()),
        "tier_counts": dict(sorted(tiers.items())),
        "source_counts": dict(sorted(Counter(selected["source"].astype(str)).items())),
        "dose": dose,
        "selection_geometry": {
            "registered_numerical_gain_floor": CALIBRATED_GAIN_FLOOR,
            "positive_gain_minimum": float(selected["targeted_positive_gain"].min()),
            "positive_gain_median": float(selected["targeted_positive_gain"].median()),
            "margin_advantage_minimum": float(
                selected["targeted_margin_advantage"].min()
            ),
            "margin_advantage_median": float(
                selected["targeted_margin_advantage"].median()
            ),
            "clean_margin_advantage_minimum": float(
                selected["targeted_clean_margin_advantage"].min()
            ),
            "clean_margin_advantage_median": float(
                selected["targeted_clean_margin_advantage"].median()
            ),
        },
        "gates": gates,
        "provenance": {
            "stage1_checkpoint_sha256": stage3_report["provenance"][
                "stage1_checkpoint_sha256"
            ],
            "frozen_stage3_report_sha256": sha256_file(stage3 / "report.json"),
            "zero_update_report_sha256": sha256_file(audit / "report.json"),
            "zero_update_geometry_sha256": sha256_file(
                audit / "action_geometry.csv.gz"
            ),
            "warm_start_checkpoint_sha256": sha256_file(
                args.warm_start_checkpoint
            ),
        },
        "outer_performance_claimed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_stage3_repair_", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        shutil.copy2(stage3 / "validation_pool.npz", staging / "validation_pool.npz")
        np.savez_compressed(
            staging / "action_spectra.npz",
            targeted_action_spectra=targeted,
            control_action_spectra=control,
            native_action_view_representable=np.ones(len(selected), dtype=bool),
            stage1_action_index=selected["stage1_action_index"].to_numpy(np.int64),
            frozen_stage3_action_index=old_indices,
            difficulty_tier=fixed_unicode(selected["difficulty_tier"]),
        )
        selected.to_csv(
            staging / "selected_actions.csv.gz", index=False, compression="gzip",
        )
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "selected_actions_sha256": sha256_file(
                staging / "selected_actions.csv.gz"
            ),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

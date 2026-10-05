"""Build a query-balanced native-DreaMS hard-negative continuation.

Stage-1 is the only warm start.  Registered Noise actions never become model
inputs in this curriculum: they are used only to choose *which measured,
different-identity candidate molecule* is the difficult negative for a clean
query.  The anchor and positive memberships remain measured spectra from the
frozen Stage-1 pool.  Multiple actions for one query are collapsed before
training, so action multiplicity cannot become optimizer dose.

Targeted and matched-control arms contain the same queries, anchors, positive
pools, event counts and negative-pool cardinalities.  They differ only in the
negative molecule selected by their respective action views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_noise_dreams_native_official_replay import encode_actions, load_npz
from build_noise_dreams_native_residual_stage2 import (
    assert_exact_checkpoint_reconstruction,
    sha256_file,
)
from build_noise_dreams_native_triplets import Registry, fixed_unicode
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from evaluate_noise_dreams_native import encode_rows
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model


STAGE5_BUILDER_VERSION = "noise_native_action_mined_negative_residual_v1"
STAGE5_STATUS = "NOISE_DREAMS_NATIVE_NEGATIVE_RESIDUAL_STAGE5_COMPLETE"
STAGE5_MAXIMUM_OPTIMIZER_STEPS = 4000


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-molecules-per-query", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _event_members(pool: dict[str, np.ndarray], pointer: str, values: str, event: int) -> np.ndarray:
    left, right = map(int, pool[pointer][event:event + 2])
    return np.asarray(pool[values][left:right], dtype=np.int64)


def _registry_sources(
    pool: dict[str, np.ndarray], registry_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(registry_positions, dtype=np.int64)
    return (
        np.asarray(pool["registry_kind"], dtype=np.int8)[positions],
        np.asarray(pool["registry_source_index"], dtype=np.int64)[positions],
    )


def _clean_events_by_query(pool: dict[str, np.ndarray]) -> dict[int, int]:
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    actions = np.asarray(pool["event_action_index"], dtype=np.int64)
    queries = np.asarray(pool["event_query"], dtype=np.int64)
    output: dict[int, int] = {}
    for event in np.flatnonzero((kinds == 0) & (actions == -1)):
        query = int(queries[event])
        if query in output:
            raise RuntimeError(f"Stage-1 has multiple clean base events for query {query}")
        output[query] = int(event)
    if not output:
        raise RuntimeError("Stage-1 clean base is empty")
    return output


def _negative_groups(graph: CandidateGraph, query: int) -> list[np.ndarray]:
    _, rows, pointers, _ = graph.query_block(query)
    return [
        np.asarray(rows[int(left):int(right)], dtype=np.int64)
        for left, right in zip(pointers[1:-1], pointers[2:])
    ]


def aggregate_action_row_scores(
    action_embeddings: np.ndarray,
    action_sources: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Collapse action multiplicity within source, then average sources."""
    action_embeddings = np.asarray(action_embeddings, dtype=np.float32)
    candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
    action_sources = np.asarray(action_sources).astype(str)
    if (
        action_embeddings.ndim != 2
        or candidate_embeddings.ndim != 2
        or action_embeddings.shape[1] != candidate_embeddings.shape[1]
        or len(action_embeddings) != len(action_sources)
        or not len(action_embeddings)
    ):
        raise ValueError("action-score inputs are not aligned")
    per_action = action_embeddings @ candidate_embeddings.T
    per_source = []
    for source in sorted(set(action_sources)):
        per_source.append(np.max(per_action[action_sources == source], axis=0))
    return np.mean(np.stack(per_source, axis=0), axis=0).astype(np.float32)


def select_hinge_active_negative_rows(
    groups: list[np.ndarray],
    row_position: dict[int, int],
    measured_embeddings: np.ndarray,
    clean_embedding: np.ndarray,
    positive_rows: np.ndarray,
    action_row_scores: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    margin: float,
    maximum_molecules: int,
) -> tuple[list[int], list[int], dict[str, float]]:
    """Select action-ranked molecules whose hardest measured row is hinge-active."""
    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    action_row_scores = np.asarray(action_row_scores, dtype=np.float32)
    if action_row_scores.shape != candidate_rows.shape:
        raise ValueError("candidate action scores are not row aligned")
    action_score = {int(row): float(score) for row, score in zip(candidate_rows, action_row_scores)}
    positive_similarity = np.asarray([
        float(measured_embeddings[row_position[int(row)]] @ clean_embedding)
        for row in positive_rows
    ])
    # Requiring activity against the easiest positive means every dynamic
    # positive draw remains capable of producing a native hinge gradient.
    threshold = float(np.max(positive_similarity) - margin)
    ranked: list[tuple[float, float, int, int]] = []
    for molecule, rows in enumerate(groups):
        available = [int(row) for row in rows if int(row) in row_position]
        if not available:
            continue
        clean_scores = np.asarray([
            float(measured_embeddings[row_position[row]] @ clean_embedding)
            for row in available
        ])
        representative = int(available[int(np.argmax(clean_scores))])
        representative_clean = float(np.max(clean_scores))
        molecule_action = max(action_score[row] for row in available)
        if representative_clean > threshold:
            ranked.append((molecule_action, representative_clean, molecule, representative))
    if not ranked:
        # This is not a semantic fallback: the query is omitted symmetrically
        # from both arms by the caller when either arm has no active molecule.
        return [], [], {
            "positive_similarity_max": float(np.max(positive_similarity)),
            "hinge_threshold": threshold,
        }
    ranked.sort(key=lambda body: (-body[0], -body[1], body[2], body[3]))
    chosen = ranked[:maximum_molecules]
    return (
        [int(body[3]) for body in chosen],
        [int(body[2]) for body in chosen],
        {
            "positive_similarity_max": float(np.max(positive_similarity)),
            "hinge_threshold": threshold,
            "negative_similarity_min": float(min(body[1] for body in chosen)),
            "negative_similarity_max": float(max(body[1] for body in chosen)),
        },
    )


def build_arm_pool(
    stage1_pool: dict[str, np.ndarray],
    clean_events: dict[int, int],
    replacements: dict[int, list[int]],
    action_unit_by_query: dict[int, int],
) -> dict[str, np.ndarray]:
    """Copy the Stage-1 clean base, replacing at most one event per query."""
    registry = Registry()
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    event_kind: list[int] = []
    event_query: list[int] = []
    event_action: list[int] = []
    event_formula: list[str] = []

    for query in sorted(clean_events):
        event = clean_events[query]
        anchor_position = int(stage1_pool["anchor_idx"][event])
        anchor_kind, anchor_source = _registry_sources(
            stage1_pool, np.asarray([anchor_position]),
        )
        if int(anchor_kind[0]) not in (Registry.HDF5, Registry.HDF5_CLONE):
            raise RuntimeError("Stage-1 clean anchor is not a measured spectrum")
        anchor = registry.add(Registry.HDF5, int(anchor_source[0]))

        positive_positions = _event_members(
            stage1_pool, "positive_ptr", "positive_idx", event,
        )
        positive_kinds, positive_sources = _registry_sources(stage1_pool, positive_positions)
        if np.any((positive_kinds != Registry.HDF5) & (positive_kinds != Registry.HDF5_CLONE)):
            raise RuntimeError("Stage-1 clean positive pool contains an action spectrum")
        positive = [registry.add(Registry.HDF5, int(row)) for row in positive_sources]

        if query in replacements:
            negative_sources = np.asarray(replacements[query], dtype=np.int64)
            kind = 1  # measured clean boundary, tracked as one semantic action unit
            action = int(action_unit_by_query[query])
        else:
            negative_positions = _event_members(
                stage1_pool, "negative_ptr", "negative_idx", event,
            )
            negative_kinds, negative_sources = _registry_sources(stage1_pool, negative_positions)
            if np.any((negative_kinds != Registry.HDF5) & (negative_kinds != Registry.HDF5_CLONE)):
                raise RuntimeError("Stage-1 clean negative pool contains an action spectrum")
            kind = 0
            action = -1
        negative = [registry.add(Registry.HDF5, int(row)) for row in negative_sources]
        if anchor in positive or anchor in negative or set(positive) & set(negative):
            raise RuntimeError("Stage-5 triplet roles overlap")
        anchors.append(anchor)
        positives.extend(positive)
        negatives.extend(negative)
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        event_kind.append(kind)
        event_query.append(query)
        event_action.append(action)
        event_formula.append(str(stage1_pool["event_formula"][event]))

    registry_kind, registry_source_index = registry.arrays()
    return {
        "registry_kind": registry_kind,
        "registry_source_index": registry_source_index,
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "event_kind": np.asarray(event_kind, dtype=np.int8),
        "event_query": np.asarray(event_query, dtype=np.int64),
        "event_action_index": np.asarray(event_action, dtype=np.int64),
        "event_formula": fixed_unicode(event_formula),
    }


def _pool_semantics(pool: dict[str, np.ndarray]) -> list[tuple]:
    result = []
    for event in range(len(pool["anchor_idx"])):
        anchor = int(pool["anchor_idx"][event])
        p = _event_members(pool, "positive_ptr", "positive_idx", event)
        n = _event_members(pool, "negative_ptr", "negative_idx", event)
        result.append((
            int(pool["event_query"][event]),
            int(pool["registry_source_index"][anchor]),
            tuple(map(int, pool["registry_source_index"][p])),
            tuple(map(int, pool["registry_source_index"][n])),
        ))
    return result


def _write_arm(
    directory: Path,
    pool: dict[str, np.ndarray],
    validation_pool: Path,
    selected: pd.DataFrame,
    report: dict[str, object],
    action_count: int,
) -> None:
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "train_pool.npz", **pool)
    shutil.copy2(validation_pool, directory / "validation_pool.npz")
    # Deliberately inert.  Semantic units are represented by measured clean
    # triplets; no action tensor is referenced by either registry.
    empty_views = np.zeros((action_count, 1, 2), dtype=np.float32)
    np.savez_compressed(
        directory / "action_spectra.npz",
        targeted_action_spectra=empty_views,
        control_action_spectra=empty_views.copy(),
        native_action_view_representable=np.zeros(action_count, dtype=bool),
        stage1_action_index=np.full(action_count, -1, dtype=np.int64),
    )
    selected.to_csv(directory / "selected_actions.csv.gz", index=False, compression="gzip")
    report = dict(report)
    report["output_artifacts"] = {
        "train_pool_sha256": sha256_file(directory / "train_pool.npz"),
        "validation_pool_sha256": sha256_file(directory / "validation_pool.npz"),
        "action_spectra_sha256": sha256_file(directory / "action_spectra.npz"),
        "selected_actions_sha256": sha256_file(directory / "selected_actions.csv.gz"),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-5 construction requires an allocated GPU")
    if args.negative_molecules_per_query < 1 or args.margin != 0.1:
        raise ValueError("Stage-5 negative-pool or native margin setting drifted")

    stage1_triplets = args.stage1_run / "triplets"
    checkpoint = args.stage1_run / "checkpoint/targeted_final.ckpt"
    summary_path = args.stage1_run / "summary/report.json"
    evaluation_path = args.stage1_run / "evaluation/targeted/report.json"
    required = (
        checkpoint, summary_path, evaluation_path,
        stage1_triplets / "train_pool.npz",
        stage1_triplets / "validation_pool.npz",
        stage1_triplets / "action_spectra.npz",
        stage1_triplets / "selected_actions.csv.gz",
        stage1_triplets / "report.json", args.graph, args.embedding_cache,
        args.data, args.architecture_checkpoint,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        float(summary.get("targeted_vs_control_recall1_pp", 0.0)) <= 0.0
        or float(summary.get("targeted_vs_control_formula_cluster_ci", {}).get("ci_low_pp", 0.0)) <= 0.0
    ):
        raise RuntimeError("Stage-1 warm start is not the registered positive champion")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("provenance", {}).get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Stage-1 evaluated checkpoint differs from warm start")

    stage1_pool = load_npz(stage1_triplets / "train_pool.npz")
    clean_events = _clean_events_by_query(stage1_pool)
    actions = pd.read_csv(stage1_triplets / "selected_actions.csv.gz", low_memory=False)
    bank = load_npz(stage1_triplets / "action_spectra.npz")
    targeted_bank = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_bank = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if len(actions) != len(targeted_bank) or capable.shape != (len(actions),):
        raise RuntimeError("Stage-1 action ledger is not aligned")
    action_queries = sorted(set(map(int, actions["query_index"])))
    if set(action_queries) - set(clean_events):
        raise RuntimeError("an action query is absent from the Stage-1 clean base")

    graph = CandidateGraph(args.graph)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        measured_rows = np.asarray(cache["rows"], dtype=np.int64)
    row_position = {int(row): index for index, row in enumerate(measured_rows)}
    model, initialization_kind = load_base_model(
        checkpoint, args.architecture_checkpoint, torch.device("cuda"), 100,
    )
    assert_exact_checkpoint_reconstruction(model, checkpoint)
    preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    measured_embeddings = encode_rows(
        model, measured_rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label="noise-stage5-measured",
    )
    capable_indices = np.flatnonzero(capable)
    targeted_embeddings = encode_actions(
        model, targeted_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage5-targeted-actions",
        preprocessor=preprocessor,
    )
    control_embeddings = encode_actions(
        model, control_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage5-control-actions",
        preprocessor=preprocessor,
    )
    del model
    targeted_by_action = {
        int(action): targeted_embeddings[index]
        for index, action in enumerate(capable_indices)
    }
    control_by_action = {
        int(action): control_embeddings[index]
        for index, action in enumerate(capable_indices)
    }
    actions_by_query: dict[int, list[int]] = defaultdict(list)
    for action, query in enumerate(actions["query_index"].to_numpy(np.int64)):
        actions_by_query[int(query)].append(int(action))

    target_replacements: dict[int, list[int]] = {}
    control_replacements: dict[int, list[int]] = {}
    records: list[dict[str, object]] = []
    for query in action_queries:
        action_indices = [a for a in actions_by_query[query] if a in targeted_by_action]
        if not action_indices:
            continue
        event = clean_events[query]
        positive_positions = _event_members(stage1_pool, "positive_ptr", "positive_idx", event)
        positive_kinds, positive_rows = _registry_sources(stage1_pool, positive_positions)
        if np.any((positive_kinds != Registry.HDF5) & (positive_kinds != Registry.HDF5_CLONE)):
            raise RuntimeError("Stage-1 positive membership is not measured")
        groups = _negative_groups(graph, query)
        candidate_rows = np.concatenate(groups)
        if any(int(row) not in row_position for row in candidate_rows):
            raise RuntimeError("Stage-5 candidate row is absent from measured embeddings")
        candidate_embeddings = measured_embeddings[[row_position[int(row)] for row in candidate_rows]]
        sources = actions.loc[action_indices, "source"].astype(str).to_numpy()
        targeted_scores = aggregate_action_row_scores(
            np.stack([targeted_by_action[a] for a in action_indices]),
            sources, candidate_embeddings,
        )
        control_scores = aggregate_action_row_scores(
            np.stack([control_by_action[a] for a in action_indices]),
            sources, candidate_embeddings,
        )
        query_row = int(graph.query_row[query])
        clean_embedding = measured_embeddings[row_position[query_row]]
        target_rows, target_molecules, target_audit = select_hinge_active_negative_rows(
            groups, row_position, measured_embeddings, clean_embedding,
            positive_rows, targeted_scores, candidate_rows,
            margin=args.margin, maximum_molecules=args.negative_molecules_per_query,
        )
        control_rows, control_molecules, control_audit = select_hinge_active_negative_rows(
            groups, row_position, measured_embeddings, clean_embedding,
            positive_rows, control_scores, candidate_rows,
            margin=args.margin, maximum_molecules=args.negative_molecules_per_query,
        )
        common_size = min(len(target_rows), len(control_rows))
        if common_size == 0:
            continue
        target_rows = target_rows[:common_size]
        control_rows = control_rows[:common_size]
        target_molecules = target_molecules[:common_size]
        control_molecules = control_molecules[:common_size]
        target_replacements[query] = target_rows
        control_replacements[query] = control_rows
        source_counts = actions.loc[action_indices, "source"].astype(str).value_counts()
        records.append({
            "query_index": query,
            "query_row": query_row,
            "query_formula": str(graph.query_formula[query]),
            "query_ik14": str(graph.query_ik14[query]),
            "registered_action_rows": len(actions_by_query[query]),
            "representable_action_rows": len(action_indices),
            "source_families": json.dumps(sorted(source_counts.index.tolist())),
            "targeted_negative_rows": json.dumps(target_rows),
            "control_negative_rows": json.dumps(control_rows),
            "targeted_negative_molecules": json.dumps(target_molecules),
            "control_negative_molecules": json.dumps(control_molecules),
            "negative_pool_size": common_size,
            "negative_membership_switched": target_rows != control_rows,
            "negative_molecule_switched": target_molecules != control_molecules,
            "targeted_hinge_threshold": target_audit["hinge_threshold"],
            "control_hinge_threshold": control_audit["hinge_threshold"],
            "targeted_negative_similarity_min": target_audit["negative_similarity_min"],
            "control_negative_similarity_min": control_audit["negative_similarity_min"],
        })
    selected = pd.DataFrame(records).sort_values("query_index", kind="stable").reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("Stage-5 found no paired hinge-active action query")
    selected_queries = list(map(int, selected["query_index"]))
    action_unit_by_query = {query: index for index, query in enumerate(selected_queries)}
    targeted_pool = build_arm_pool(
        stage1_pool, clean_events, target_replacements, action_unit_by_query,
    )
    control_pool = build_arm_pool(
        stage1_pool, clean_events, control_replacements, action_unit_by_query,
    )
    targeted_semantics = _pool_semantics(targeted_pool)
    control_semantics = _pool_semantics(control_pool)
    if len(targeted_semantics) != len(control_semantics):
        raise RuntimeError("Stage-5 matched arms have different event counts")
    only_negative_diff = all(
        left[:3] == right[:3]
        for left, right in zip(targeted_semantics, control_semantics, strict=True)
    ) and all(
        np.array_equal(targeted_pool[key], control_pool[key])
        for key in (
            "event_kind", "event_query", "event_action_index", "event_formula",
            "positive_ptr", "negative_ptr",
        )
    )
    negative_switches = int(selected["negative_membership_switched"].sum())
    formulas = set(map(str, targeted_pool["event_formula"]))
    validation_pool = load_npz(stage1_triplets / "validation_pool.npz")
    validation_formulas = set(map(str, validation_pool["event_formula"]))
    steps = (len(targeted_pool["anchor_idx"]) + 3) // 4
    gates = {
        "stage1_champion_tensor_reconstruction_is_exact": True,
        "stage1_evaluated_checkpoint_is_the_warm_start": True,
        "stage1_clean_base_is_preserved_one_event_per_query": (
            len(targeted_pool["anchor_idx"]) == len(clean_events)
            and len(np.unique(targeted_pool["event_query"])) == len(clean_events)
        ),
        "all_registered_actions_were_considered_before_query_eligibility": (
            sum(len(actions_by_query[query]) for query in action_queries) == len(actions)
        ),
        "all_seven_registered_action_sources_were_considered": set(
            actions["source"].astype(str)
        ) == {
            "N_mature", "P_guided_original", "E10B", "E11", "E12B",
            "A4_exact", "V4_gradient_path",
        },
        "all_selected_queries_use_all_their_representable_actions": all(
            int(row.representable_action_rows) == int(np.sum(capable[actions_by_query[int(row.query_index)]]))
            for row in selected.itertuples(index=False)
        ),
        "action_multiplicity_does_not_change_query_dose": (
            len(np.unique(targeted_pool["event_query"])) == len(targeted_pool["event_query"])
        ),
        "targeted_and_control_share_anchor_positive_and_count": only_negative_diff,
        "targeted_and_control_have_a_real_negative_membership_switch": negative_switches > 0,
        "all_selected_negatives_are_native_hinge_active": bool(
            (selected["targeted_negative_similarity_min"] > selected["targeted_hinge_threshold"]).all()
            and (selected["control_negative_similarity_min"] > selected["control_hinge_threshold"]).all()
        ),
        "action_spectra_never_enter_either_model_input_registry": bool(
            np.all(targeted_pool["registry_kind"] != Registry.ACTION)
            and np.all(control_pool["registry_kind"] != Registry.ACTION)
        ),
        "optimizer_step_budget_is_bounded": steps <= STAGE5_MAXIMUM_OPTIMIZER_STEPS,
        "train_validation_formulas_are_disjoint": not bool(formulas & validation_formulas),
        "validation_pool_is_bitwise_reused_from_stage1": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-5 negative-residual triplet gates failed: {gates}")

    common_report = {
        "status": STAGE5_STATUS,
        "builder_version": STAGE5_BUILDER_VERSION,
        "initialization_kind": initialization_kind,
        "scientific_contract": (
            "Continue the evaluated Stage-1 champion with the unmodified native "
            "DreaMS triplet runtime. Keep one measured clean-anchor event per "
            "Stage-1 clean query; use all representable actions only to mine "
            "hinge-active different-identity negative molecules; collapse action "
            "multiplicity by query and source; matched arms differ only in negative membership."
        ),
        "stage1_clean_base_events": len(clean_events),
        "paired_action_queries": len(selected),
        "registered_action_rows": len(actions),
        "registered_action_rows_on_selected_queries": int(selected["registered_action_rows"].sum()),
        "representable_action_rows_consumed": int(selected["representable_action_rows"].sum()),
        "negative_membership_switch_queries": negative_switches,
        "negative_membership_switch_fraction": float(negative_switches / len(selected)),
        "source_families": sorted(set(actions["source"].astype(str))),
        "training_budget": {
            "events": len(targeted_pool["anchor_idx"]),
            "optimizer_steps": steps,
            "batch_size": 4,
            "epochs": 1,
        },
        "model_input_contract": {
            "anchor": "measured clean query spectrum",
            "positive": "dynamic measured same-identity Stage-1 pool",
            "negative": "dynamic measured different-identity action-mined pool",
            "action_tensor_enters_encoder": False,
        },
        "gates": gates,
        "provenance": {
            "stage1_checkpoint_sha256": sha256_file(checkpoint),
            "stage1_summary_sha256": sha256_file(summary_path),
            "stage1_evaluation_sha256": sha256_file(evaluation_path),
            "stage1_triplet_report_sha256": sha256_file(stage1_triplets / "report.json"),
            "candidate_graph_sha256": sha256_file(args.graph),
            "embedding_registry_sha256": sha256_file(args.embedding_cache),
        },
        "outer_performance_claimed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_stage5_", dir=args.output.parent))
    try:
        for arm, pool in (("targeted", targeted_pool), ("control", control_pool)):
            report = dict(common_report)
            report["arm"] = arm
            _write_arm(
                staging / arm, pool, stage1_triplets / "validation_pool.npz",
                selected, report, len(selected),
            )
        selected.to_csv(staging / "selected_queries.csv.gz", index=False, compression="gzip")
        root_report = dict(common_report)
        root_report["arm_artifacts"] = {
            arm: {
                "report_sha256": sha256_file(staging / arm / "report.json"),
                "train_pool_sha256": sha256_file(staging / arm / "train_pool.npz"),
            }
            for arm in ("targeted", "control")
        }
        root_report["selected_queries_sha256"] = sha256_file(staging / "selected_queries.csv.gz")
        (staging / "report.json").write_text(json.dumps(root_report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(root_report, indent=2), flush=True)


if __name__ == "__main__":
    main()

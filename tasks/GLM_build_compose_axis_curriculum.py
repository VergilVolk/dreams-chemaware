"""GLM composition builder: chemistry-champion-current Noise residual curriculum.

Purpose (integration experiment, pre-registered 2026-09-30):
    Put the *Noise* action curriculum on top of the *ChemAware* champion
    (Phase-A) instead of on top of the Noise Stage-1 champion.  Nothing in the
    native DreaMS model, loss, optimizer, preprocessor or checkpoint format
    changes; only which spectra occupy the anchor / positive / negative roles,
    and which one relation per query is selected.

Two selection modes share one frozen geometry and one dose:
    --axis-balance advantage-max
        exact Stage-2 recipe: per query keep the relation with the largest
        min(control advantage, clean advantage).  This is the un-stratified
        control for the axis hypothesis.
    --axis-balance balanced
        classify every action query into a positive-deficit (PD) or
        negative-excess (NE) axis from its *own* current clean boundary versus
        the processed population medians, then interleave the two axis-ranked
        lists so both axes are represented as far as availability allows.

Both modes keep exactly one relation per query, one clean-boundary event when
active, and the same protection sentinels.  The matched control arm is the
registered same-query control action spectrum of the *same* relation, so the
targeted-minus-control contrast isolates action content.

All helpers are imported read-only from existing project modules; this file
never modifies pre-existing code.
"""
from __future__ import annotations

import argparse
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
    graph_boundary,
    make_pool,
    sha256_file,
)
from build_noise_dreams_native_triplets import Registry, stable_fold
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from evaluate_noise_dreams_native import encode_rows
from GLM_compose_axis_core import AXIS_NE, AXIS_PD, classify_axis, order_queries
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model


COMPOSE_BUILDER_VERSION = "glm_compose_axis_curriculum_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", type=Path, required=True,
                        help="Noise Stage-1 run providing the action ledger+bank")
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True,
                        help="ChemAware champion (Phase-A) used as the warm start")
    parser.add_argument("--expected-warm-start-sha256", type=str, required=True,
                        help="Fail-closed pin of the warm-start checkpoint")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--minimum-action-advantage", type=float, default=5e-6)
    parser.add_argument("--protection-sentinels", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--axis-balance", choices=("balanced", "advantage-max"),
                        default="balanced")
    parser.add_argument("--minimum-axis-queries", type=int, default=1,
                        help="Reported axis floor; the SBATCH enforces the "
                             "pre-registered value and stops before training.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if (args.margin <= 0 or args.minimum_action_advantage <= 0
            or args.protection_sentinels < 4 or args.batch_size < 1):
        raise ValueError("invalid composition selection settings")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("composition construction requires an allocated GPU")

    stage1_triplets = args.stage1_run / "triplets"
    required = (
        args.warm_start_checkpoint, args.graph, args.embedding_cache,
        args.data, args.architecture_checkpoint,
        stage1_triplets / "train_pool.npz",
        stage1_triplets / "validation_pool.npz",
        stage1_triplets / "action_spectra.npz",
        stage1_triplets / "selected_actions.csv.gz",
        stage1_triplets / "report.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    warm_sha = sha256_file(args.warm_start_checkpoint)
    if warm_sha != args.expected_warm_start_sha256:
        raise RuntimeError(
            "warm-start checkpoint does not match the pinned hash: "
            f"{warm_sha} != {args.expected_warm_start_sha256}"
        )

    graph = CandidateGraph(args.graph)
    # The embedding cache provides the frozen row registry only; geometry comes
    # from the pinned warm start.  Accept either a frozen .npz carrying a "rows"
    # array or a raw .npy row vector file.
    if args.embedding_cache.suffix == ".npz":
        with np.load(args.embedding_cache, allow_pickle=False) as cache:
            rows = np.asarray(cache["rows"], dtype=np.int64)
    else:
        rows = np.asarray(np.load(args.embedding_cache, allow_pickle=False),
                          dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise RuntimeError("embedding row registry is empty or malformed")
    row_position = {int(row): index for index, row in enumerate(rows)}
    actions = pd.read_csv(stage1_triplets / "selected_actions.csv.gz", low_memory=False)
    bank = load_npz(stage1_triplets / "action_spectra.npz")
    targeted_bank = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_bank = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if len(actions) != len(targeted_bank) or capable.shape != (len(actions),):
        raise RuntimeError("Stage-1 action artifacts are not aligned")

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
    embeddings = encode_rows(
        model, rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label="glm-compose-measured-rows",
    )
    target_embedding = encode_actions(
        model, targeted_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="glm-compose-targeted-actions",
        preprocessor=preprocessor,
    )
    control_embedding = encode_actions(
        model, control_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="glm-compose-control-actions",
        preprocessor=preprocessor,
    )
    del model
    capable_indices = np.flatnonzero(capable)
    target_by_action = {int(action): target_embedding[index]
                        for index, action in enumerate(capable_indices)}
    control_by_action = {int(action): control_embedding[index]
                         for index, action in enumerate(capable_indices)}
    by_query: dict[int, list[int]] = defaultdict(list)
    for action in capable_indices:
        by_query[int(actions.at[int(action), "query_index"])].append(int(action))

    # ---- per-query qualified candidates under the frozen warm-start geometry
    per_query: dict[int, dict[str, object]] = {}
    for query in sorted(by_query):
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("outer-held formula reached composition construction")
        clean_vector = embeddings[row_position[int(graph.query_row[query])]]
        clean = graph_boundary(graph, query, row_position, embeddings, clean_vector)
        pair_slice, candidate_rows, _, _ = graph.query_block(query)
        del pair_slice
        pair_embeddings = embeddings[[row_position[int(row)] for row in candidate_rows]]
        candidates = []
        for action in by_query[query]:
            targeted_vector = target_by_action[action]
            targeted_scores = pair_embeddings @ targeted_vector
            _, _, pointers, _ = graph.query_block(query)
            p_end = int(pointers[1])
            p_local = int(np.argmax(targeted_scores[:p_end]))
            n_local = p_end + int(np.argmax(targeted_scores[p_end:]))
            p_embedding = pair_embeddings[p_local]
            n_embedding = pair_embeddings[n_local]
            target_margin = float(targeted_vector @ p_embedding
                                  - targeted_vector @ n_embedding)
            control_vector = control_by_action[action]
            control_margin = float(control_vector @ p_embedding
                                   - control_vector @ n_embedding)
            clean_same_margin = float(clean_vector @ p_embedding
                                      - clean_vector @ n_embedding)
            control_advantage = target_margin - control_margin
            clean_advantage = target_margin - clean_same_margin
            if (target_margin < args.margin
                    and control_advantage >= args.minimum_action_advantage
                    and clean_advantage >= args.minimum_action_advantage):
                candidates.append({
                    "action": int(action),
                    "action_id": str(actions.at[action, "action_id"]),
                    "rank_key": (min(control_advantage, clean_advantage),
                                 control_advantage + clean_advantage,
                                 str(actions.at[action, "action_id"])),
                    "action_positive_row": int(candidate_rows[p_local]),
                    "action_negative_row": int(candidate_rows[n_local]),
                    "target_margin": target_margin,
                    "control_margin": control_margin,
                    "clean_same_margin": clean_same_margin,
                    "control_advantage": control_advantage,
                    "clean_advantage": clean_advantage,
                })
        if not candidates:
            continue
        candidates.sort(key=lambda body: body["rank_key"], reverse=True)
        per_query[query] = {
            "formula": formula,
            "candidates": candidates,
            "clean": clean,
            "clean_positive_score": float(clean["positive_score"]),
            "clean_negative_score": float(clean["negative_score"]),
        }
    if not per_query:
        raise RuntimeError("composition geometry has no qualified residual relation")

    # ---- axis classification on the processed population (train-only, no RNG)
    positive_scores = np.asarray(
        [body["clean_positive_score"] for body in per_query.values()], dtype=np.float64)
    negative_scores = np.asarray(
        [body["clean_negative_score"] for body in per_query.values()], dtype=np.float64)
    median_positive = float(np.median(positive_scores))
    median_negative = float(np.median(negative_scores))
    for query, body in per_query.items():
        body["axis"] = classify_axis(
            median_positive - float(body["clean_positive_score"]),
            float(body["clean_negative_score"]) - median_negative,
        )
        body["axis_deficit"] = median_positive - float(body["clean_positive_score"])
        body["axis_excess"] = float(body["clean_negative_score"]) - median_negative

    # ---- selection
    ordered_queries: list[int] = order_queries(args.axis_balance, per_query)

    selected_rows: list[dict[str, object]] = []
    for query in sorted(ordered_queries):
        body = per_query[query]
        best = body["candidates"][0]
        action = int(best["action"])
        source = actions.loc[action].to_dict()
        clean = body["clean"]
        source.update({
            "stage1_action_index": action,
            "action_positive_row": int(best["action_positive_row"]),
            "action_negative_row": int(best["action_negative_row"]),
            "stage1_target_margin": float(best["target_margin"]),
            "stage1_control_margin": float(best["control_margin"]),
            "stage1_clean_same_boundary_margin": float(best["clean_same_margin"]),
            "target_minus_control_margin": float(best["control_advantage"]),
            "target_minus_clean_margin": float(best["clean_advantage"]),
            "clean_positive_row": int(clean["positive_row"]),
            "clean_negative_row": int(clean["negative_row"]),
            "stage1_clean_current_margin": float(clean["margin"]),
            "clean_boundary_active": bool(float(clean["margin"]) < args.margin),
            "compose_axis": str(body["axis"]),
            "compose_axis_deficit": float(body["axis_deficit"]),
            "compose_axis_excess": float(body["axis_excess"]),
            "compose_clean_positive_score": float(body["clean_positive_score"]),
            "compose_clean_negative_score": float(body["clean_negative_score"]),
            "compose_mode": args.axis_balance,
            "compose_qualified_relations": int(len(body["candidates"])),
        })
        selected_rows.append(source)
    if not selected_rows:
        raise RuntimeError("composition selection produced no relation")
    selected = pd.DataFrame(selected_rows).sort_values(
        ["query_index", "action_id"], kind="stable").reset_index(drop=True)
    if selected["query_index"].duplicated().any():
        raise RuntimeError("composition did not produce one unique relation per query")

    # ---- protection sentinels (action-free, currently correct, hinge-active)
    action_queries = set(map(int, selected["query_index"]))
    sentinel_rows: list[dict[str, object]] = []
    train_pool = load_npz(stage1_triplets / "train_pool.npz")
    eligible_queries = np.unique(np.asarray(train_pool["event_query"], dtype=np.int64))
    for query in eligible_queries:
        query = int(query)
        if query in action_queries:
            continue
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("outer-held formula reached protection selection")
        clean_vector = embeddings[row_position[int(graph.query_row[query])]]
        clean = graph_boundary(graph, query, row_position, embeddings, clean_vector)
        margin = float(clean["margin"])
        if 0 < margin < args.margin:
            sentinel_rows.append({
                "query_index": query,
                "query_row": int(graph.query_row[query]),
                "query_formula": formula,
                "clean_positive_row": int(clean["positive_row"]),
                "clean_negative_row": int(clean["negative_row"]),
                "stage1_clean_current_margin": margin,
            })
    if len(sentinel_rows) < 4:
        raise RuntimeError("composition lacks action-free protection queries")
    sentinels = pd.DataFrame(sentinel_rows).sort_values(
        ["stage1_clean_current_margin", "query_index"], kind="stable"
    ).head(args.protection_sentinels).reset_index(drop=True)

    pool = make_pool(selected, sentinels)
    validation_pool = load_npz(stage1_triplets / "validation_pool.npz")
    selected_targeted = targeted_bank[selected["stage1_action_index"].to_numpy(np.int64)]
    selected_control = control_bank[selected["stage1_action_index"].to_numpy(np.int64)]
    action_events = np.asarray(pool["event_action_index"], dtype=np.int64)
    action_events = action_events[action_events >= 0]
    event_queries, event_query_counts = np.unique(
        np.asarray(pool["event_query"], dtype=np.int64), return_counts=True)
    axis_counts = dict(sorted(
        selected["compose_axis"].astype(str).value_counts().items()))

    gates = {
        "warm_start_matches_pinned_sha256": True,
        "warm_start_tensor_reconstruction_is_exact": True,
        "outer_held_formulas_are_absent": bool(all(
            stable_fold(str(value), 5, args.formula_fold_seed) != args.outer_fold
            for value in pool["event_formula"])),
        "one_residual_relation_per_query": bool(
            selected["query_index"].nunique() == len(selected)),
        "all_residual_relations_are_native_hinge_active": bool(
            np.all(selected["stage1_target_margin"] < args.margin)),
        "all_residual_relations_beat_same_query_control": bool(
            np.all(selected["target_minus_control_margin"] > 0)),
        "all_residual_relations_beat_clean_same_boundary": bool(
            np.all(selected["target_minus_clean_margin"] > 0)),
        "each_residual_relation_is_exposed_once": bool(
            np.array_equal(action_events, np.arange(len(selected), dtype=np.int64))),
        "each_query_contributes_at_most_action_plus_clean": bool(
            len(event_queries) and int(event_query_counts.max()) <= 2),
        "all_protection_sentinels_are_correct_and_active": bool(
            np.all((sentinels["stage1_clean_current_margin"] > 0)
                   & (sentinels["stage1_clean_current_margin"] < args.margin))),
        "targeted_and_control_payloads_are_distinct": bool(
            np.all(np.any(selected_targeted != selected_control, axis=(1, 2)))),
        "validation_pool_is_action_free": bool(
            np.all(np.asarray(validation_pool["event_kind"], dtype=np.int8) == 0)
            and np.all(np.asarray(validation_pool["event_action_index"],
                                  dtype=np.int64) == -1)
            and not np.any(np.asarray(validation_pool["registry_kind"],
                                      dtype=np.int8) == Registry.ACTION)),
        "train_and_validation_formulas_are_disjoint": not bool(
            set(map(str, pool["event_formula"]))
            & set(map(str, validation_pool["event_formula"]))),
        "both_axes_present_when_balanced": bool(
            args.axis_balance != "balanced"
            or (axis_counts.get(AXIS_PD, 0) > 0 and axis_counts.get(AXIS_NE, 0) > 0)),
        "axis_floor_declared_before_training": bool(
            min(axis_counts.values()) >= args.minimum_axis_queries),
    }
    if not all(gates.values()):
        raise RuntimeError(f"GLM composition gates failed: {gates}")

    report = {
        "status": "GLM_COMPOSE_AXIS_CURRICULUM_BUILD_COMPLETE",
        "builder_version": COMPOSE_BUILDER_VERSION,
        "initialization_kind": initialization_kind,
        "compose_mode": args.axis_balance,
        "scientific_contract": (
            "Noise action curriculum on top of the ChemAware champion: one "
            "frozen relation per query selected in the champion's own current "
            "geometry, matched control = the same relation's registered "
            "control action, native DreaMS loss/model/optimizer unchanged."
        ),
        "axis_definitions": {
            "positive_deficit": (
                "median(clean positive score) - query clean positive score >= "
                "query clean negative score - median(clean negative score)"),
            "negative_excess": "the complementary assignment",
            "population_medians": {
                "clean_positive_score": median_positive,
                "clean_negative_score": median_negative,
            },
            "note": "detection heuristic fixed before training; label used for "
                    "reporting and (in balanced mode) for selection interleaving",
        },
        "selection_thresholds": {
            "native_triplet_margin": args.margin,
            "minimum_action_advantage": args.minimum_action_advantage,
            "protection_sentinels": args.protection_sentinels,
            "minimum_axis_queries": args.minimum_axis_queries,
        },
        "selected_action_queries": int(len(selected)),
        "axis_selected_queries": axis_counts,
        "axis_qualified_queries": {
            AXIS_PD: int(sum(1 for b in per_query.values() if b["axis"] == AXIS_PD)),
            AXIS_NE: int(sum(1 for b in per_query.values() if b["axis"] == AXIS_NE)),
        },
        "direct_clean_boundary_events": int(selected["clean_boundary_active"].sum()),
        "protection_sentinels": int(len(sentinels)),
        "train_events": int(len(pool["anchor_idx"])),
        "sources": dict(sorted(selected["source"].astype(str).value_counts().items())),
        "margin_summary": {
            key: {"minimum": float(selected[key].min()),
                  "median": float(selected[key].median()),
                  "maximum": float(selected[key].max())}
            for key in ("stage1_target_margin", "target_minus_control_margin",
                        "target_minus_clean_margin", "stage1_clean_current_margin")
        },
        "axis_margin_summary": {
            axis: {
                "queries": int((selected["compose_axis"] == axis).sum()),
                "target_margin_median": float(
                    selected.loc[selected["compose_axis"] == axis,
                                 "stage1_target_margin"].median()),
                "clean_positive_score_median": float(
                    selected.loc[selected["compose_axis"] == axis,
                                 "compose_clean_positive_score"].median()),
                "clean_negative_score_median": float(
                    selected.loc[selected["compose_axis"] == axis,
                                 "compose_clean_negative_score"].median()),
            }
            for axis in (AXIS_PD, AXIS_NE)
            if (selected["compose_axis"] == axis).any()
        },
        "gates": gates,
        "provenance": {
            "warm_start_checkpoint": str(args.warm_start_checkpoint),
            "warm_start_checkpoint_sha256": warm_sha,
            "stage1_action_ledger_sha256": sha256_file(
                stage1_triplets / "selected_actions.csv.gz"),
            "stage1_action_bank_sha256": sha256_file(
                stage1_triplets / "action_spectra.npz"),
            "stage1_triplet_report_sha256": sha256_file(
                stage1_triplets / "report.json"),
            "candidate_graph_sha256": sha256_file(args.graph),
            "embedding_registry_sha256": sha256_file(args.embedding_cache),
        },
        "outer_performance_claimed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="glm_compose_", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        shutil.copy2(stage1_triplets / "validation_pool.npz",
                     staging / "validation_pool.npz")
        np.savez_compressed(
            staging / "action_spectra.npz",
            targeted_action_spectra=selected_targeted,
            control_action_spectra=selected_control,
            native_action_view_representable=np.ones(len(selected), dtype=bool),
            stage1_action_index=selected["stage1_action_index"].to_numpy(np.int64),
        )
        selected.to_csv(staging / "selected_actions.csv.gz", index=False,
                        compression="gzip")
        sentinels.to_csv(staging / "protection_sentinels.csv.gz", index=False,
                         compression="gzip")
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "selected_actions_sha256": sha256_file(
                staging / "selected_actions.csv.gz"),
            "protection_sentinels_sha256": sha256_file(
                staging / "protection_sentinels.csv.gz"),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

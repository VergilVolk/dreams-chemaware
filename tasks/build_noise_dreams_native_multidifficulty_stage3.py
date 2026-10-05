"""Build a large easy/medium/hard Noise curriculum under the Stage-1 champion.

Difficulty is semantic, not merely hinge activity.  Easy actions remain close
to the clean query and use its easiest independent same-identity reference;
medium actions use a median same-identity reference; hard actions are strongly
noise-shifted and use the least-similar valid same-identity reference.  Every
action uses the currently most-similar different-identity candidate spectrum
as its hard negative.  Outer-held formulas and outcomes are never inspected.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_noise_dreams_native_official_replay import encode_actions, load_npz
from build_noise_dreams_native_residual_stage2 import (
    assert_exact_checkpoint_reconstruction,
    sha256_file,
)
from build_noise_dreams_native_triplets import Registry, fixed_unicode, stable_fold
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from evaluate_noise_dreams_native import encode_rows
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model


STAGE3_BUILDER_VERSION = "noise_native_multidifficulty_hard_positive_v2"
REGISTERED_SOURCES = {
    "A4_exact",
    "E10B",
    "E11",
    "E12B",
    "N_mature",
    "P_guided_original",
    "V4_gradient_path",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--maximum-action-events-per-query", type=int, default=4)
    parser.add_argument("--minimum-action-events", type=int, default=10000)
    parser.add_argument("--maximum-clean-negatives", type=int, default=16)
    parser.add_argument("--protection-sentinels", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def candidate_rows(
    graph: CandidateGraph,
    query: int,
    query_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, rows, pointers, _ = graph.query_block(query)
    positive_end = int(pointers[1])
    positive = np.asarray([
        int(row) for row in rows[:positive_end] if int(row) != query_row
    ], dtype=np.int64)
    negative = np.asarray(rows[positive_end:], dtype=np.int64)
    if not len(positive) or not len(negative):
        raise RuntimeError(f"query {query} lacks an independent positive or negative")
    return positive, negative


def choose_query_events(
    records: list[dict[str, object]], maximum: int, margin: float = 0.1,
) -> list[dict]:
    """Choose distinct action views spanning easy, medium and hard positives."""
    chosen: list[dict] = []
    used: set[int] = set()

    def materialize(body: dict[str, object], tier: str) -> dict[str, object]:
        base = {
            key: value for key, value in body.items()
            if key not in {"easy", "medium", "hard"}
        }
        return {**base, "difficulty_tier": tier, **body[tier]}

    easy = [body for body in records if float(body["easy_margin"]) < margin]
    if easy:
        body = max(
            easy,
            key=lambda item: (
                float(item["action_clean_similarity"]),
                float(item["easy_margin"]),
                str(item["action_id"]),
            ),
        )
        chosen.append(materialize(body, "easy"))
        used.add(int(body["stage1_action_index"]))

    medium = [
        body for body in records
        if int(body["stage1_action_index"]) not in used
        and float(body["medium_margin"]) < margin
    ]
    if medium and len(chosen) < maximum:
        similarities = np.asarray(
            [float(body["action_clean_similarity"]) for body in medium]
        )
        median = float(np.median(similarities))
        body = min(
            medium,
            key=lambda item: (
                abs(float(item["action_clean_similarity"]) - median),
                abs(float(item["medium_margin"])),
                str(item["action_id"]),
            ),
        )
        chosen.append(materialize(body, "medium"))
        used.add(int(body["stage1_action_index"]))

    hard = sorted(
        (
            body for body in records
            if int(body["stage1_action_index"]) not in used
            and float(body["hard_margin"]) < margin
        ),
        key=lambda item: (
            float(item["action_clean_similarity"]),
            float(item["hard_margin"]),
            str(item["action_id"]),
        ),
    )
    for body in hard:
        if len(chosen) >= maximum:
            break
        chosen.append(materialize(body, "hard"))
        used.add(int(body["stage1_action_index"]))
    return chosen


def make_pool(
    selected: pd.DataFrame,
    clean_by_query: dict[int, dict[str, object]],
    sentinels: list[dict[str, object]],
) -> dict[str, np.ndarray]:
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

    def append(
        anchor: int,
        positive_rows: list[int],
        negative_rows: list[int],
        *,
        kind: int,
        query: int,
        action: int,
        formula: str,
    ) -> None:
        p = list(map(int, positive_rows))
        n = list(map(int, negative_rows))
        if not p or not n or anchor in p or anchor in n or set(p) & set(n):
            raise RuntimeError("Stage-3 triplet roles are empty or overlapping")
        anchors.append(anchor)
        positives.extend(p)
        negatives.extend(n)
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        event_kind.append(kind)
        event_query.append(query)
        event_action.append(action)
        event_formula.append(formula)

    for event, row in selected.reset_index(drop=True).iterrows():
        append(
            registry.add(Registry.ACTION, event),
            [registry.add(Registry.HDF5, int(row.action_positive_row))],
            [registry.add(Registry.HDF5, int(row.action_negative_row))],
            kind=2,
            query=int(row.query_index),
            action=event,
            formula=str(row.query_formula),
        )
    for query in sorted(clean_by_query):
        body = clean_by_query[query]
        append(
            registry.add(Registry.HDF5, int(body["query_row"])),
            [
                registry.add(Registry.HDF5, int(row))
                for row in body["positive_rows"]
            ],
            [
                registry.add(Registry.HDF5, int(row))
                for row in body["negative_rows"]
            ],
            kind=0,
            query=query,
            action=-1,
            formula=str(body["query_formula"]),
        )
    for body in sentinels:
        append(
            registry.add(Registry.HDF5, int(body["query_row"])),
            [
                registry.add(Registry.HDF5, int(row))
                for row in body["positive_rows"]
            ],
            [
                registry.add(Registry.HDF5, int(row))
                for row in body["negative_rows"]
            ],
            kind=0,
            query=int(body["query_index"]),
            action=-1,
            formula=str(body["query_formula"]),
        )
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


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-3 triplet construction requires an allocated GPU")
    if (
        args.margin != 0.1
        or args.maximum_action_events_per_query != 4
        or args.minimum_action_events < 10000
        or args.protection_sentinels < 4
    ):
        raise RuntimeError("Stage-3 registered scale/difficulty settings drifted")

    triplets = args.stage1_run / "triplets"
    checkpoint = args.stage1_run / "checkpoint/targeted_final.ckpt"
    evaluation = args.stage1_run / "evaluation/targeted/report.json"
    required = (
        checkpoint,
        evaluation,
        triplets / "report.json",
        triplets / "train_pool.npz",
        triplets / "validation_pool.npz",
        triplets / "action_spectra.npz",
        triplets / "selected_actions.csv.gz",
        args.graph,
        args.embedding_cache,
        args.data,
        args.architecture_checkpoint,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluated = json.loads(evaluation.read_text(encoding="utf-8"))
    if evaluated.get("provenance", {}).get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Stage-3 warm start differs from evaluated Stage-1 champion")

    graph = CandidateGraph(args.graph)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        rows = np.asarray(cache["rows"], dtype=np.int64)
    row_position = {int(row): index for index, row in enumerate(rows)}
    actions = pd.read_csv(triplets / "selected_actions.csv.gz", low_memory=False)
    bank = load_npz(triplets / "action_spectra.npz")
    targeted_bank = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_bank = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if len(actions) != len(targeted_bank) or capable.shape != (len(actions),):
        raise RuntimeError("Stage-1 action artifacts are not aligned")

    model, initialization_kind = load_base_model(
        checkpoint, args.architecture_checkpoint, torch.device("cuda"), 100,
    )
    assert_exact_checkpoint_reconstruction(model, checkpoint)
    preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    embeddings = encode_rows(
        model, rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label="noise-stage3-measured",
    )
    capable_indices = np.flatnonzero(capable)
    action_embeddings = encode_actions(
        model, targeted_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage3-targeted-actions",
        preprocessor=preprocessor,
    )
    del model
    action_embedding = {
        int(action): action_embeddings[index]
        for index, action in enumerate(capable_indices)
    }
    by_query: dict[int, list[int]] = defaultdict(list)
    for action in capable_indices:
        by_query[int(actions.at[int(action), "query_index"])].append(int(action))

    selected_rows: list[dict[str, object]] = []
    clean_by_query: dict[int, dict[str, object]] = {}
    for query in sorted(by_query):
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("outer-held formula reached Stage-3 construction")
        query_row = int(graph.query_row[query])
        clean_vector = embeddings[row_position[query_row]]
        positive_rows, negative_rows = candidate_rows(graph, query, query_row)
        positive_embeddings = embeddings[[row_position[int(row)] for row in positive_rows]]
        negative_embeddings = embeddings[[row_position[int(row)] for row in negative_rows]]
        clean_negative_scores = negative_embeddings @ clean_vector
        clean_negative_order = np.argsort(-clean_negative_scores, kind="stable")
        clean_by_query[query] = {
            "query_row": query_row,
            "query_formula": formula,
            "positive_rows": list(map(int, positive_rows)),
            "negative_rows": list(map(
                int,
                negative_rows[clean_negative_order[:args.maximum_clean_negatives]],
            )),
        }

        records: list[dict[str, object]] = []
        for action in by_query[query]:
            vector = action_embedding[action]
            positive_scores = positive_embeddings @ vector
            negative_scores = negative_embeddings @ vector
            positive_order = np.argsort(positive_scores, kind="stable")
            negative_index = int(np.argmax(negative_scores))
            hard_index = int(positive_order[0])
            medium_index = int(positive_order[len(positive_order) // 2])
            easy_index = int(positive_order[-1])
            negative_score = float(negative_scores[negative_index])

            def relation(index: int) -> dict[str, object]:
                positive_score = float(positive_scores[index])
                return {
                    "action_positive_row": int(positive_rows[index]),
                    "action_negative_row": int(negative_rows[negative_index]),
                    "positive_similarity": positive_score,
                    "negative_similarity": negative_score,
                    "triplet_margin": positive_score - negative_score,
                    "positive_pool_size": int(len(positive_rows)),
                    "positive_identity_verified": True,
                    "negative_identity_verified": True,
                }

            easy = relation(easy_index)
            medium = relation(medium_index)
            hard = relation(hard_index)
            records.append({
                "stage1_action_index": action,
                "action_id": str(actions.at[action, "action_id"]),
                "query_index": query,
                "query_row": query_row,
                "query_formula": formula,
                "query_ik14": str(graph.query_ik14[query]),
                "source": str(actions.at[action, "source"]),
                "family": str(actions.at[action, "family"]),
                "action_clean_similarity": float(vector @ clean_vector),
                "easy_margin": float(easy["triplet_margin"]),
                "medium_margin": float(medium["triplet_margin"]),
                "hard_margin": float(hard["triplet_margin"]),
                "easy": easy,
                "medium": medium,
                "hard": hard,
            })
        selected_rows.extend(choose_query_events(
            records, args.maximum_action_events_per_query, args.margin,
        ))

    if not selected_rows:
        raise RuntimeError("Stage-3 produced no active multi-difficulty actions")
    selected = pd.DataFrame(selected_rows).sort_values(
        ["query_index", "difficulty_tier", "action_id"], kind="stable"
    ).reset_index(drop=True)
    if selected["stage1_action_index"].duplicated().any():
        raise RuntimeError("Stage-3 reused one action view as multiple dose")

    action_queries = set(map(int, selected["query_index"]))
    stage1_pool = load_npz(triplets / "train_pool.npz")
    sentinel_candidates: list[dict[str, object]] = []
    for query in np.unique(np.asarray(stage1_pool["event_query"], dtype=np.int64)):
        query = int(query)
        if query in action_queries:
            continue
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("outer-held formula reached Stage-3 protection")
        query_row = int(graph.query_row[query])
        positive_rows, negative_rows = candidate_rows(graph, query, query_row)
        vector = embeddings[row_position[query_row]]
        p_score = embeddings[[row_position[int(row)] for row in positive_rows]] @ vector
        n_score = embeddings[[row_position[int(row)] for row in negative_rows]] @ vector
        margin = float(np.max(p_score) - np.max(n_score))
        if 0 < margin < args.margin:
            order = np.argsort(-n_score, kind="stable")[:args.maximum_clean_negatives]
            sentinel_candidates.append({
                "query_index": query,
                "query_row": query_row,
                "query_formula": formula,
                "positive_rows": list(map(int, positive_rows)),
                "negative_rows": list(map(int, negative_rows[order])),
                "clean_margin": margin,
            })
    sentinels = sorted(
        sentinel_candidates,
        key=lambda body: (float(body["clean_margin"]), int(body["query_index"])),
    )[:args.protection_sentinels]
    if len(sentinels) < 4:
        raise RuntimeError("Stage-3 lacks action-free final-batch fillers")

    pool = make_pool(selected, clean_by_query, sentinels)
    selected_indices = selected["stage1_action_index"].to_numpy(np.int64)
    selected_targeted = targeted_bank[selected_indices]
    selected_control = control_bank[selected_indices]
    validation_pool = load_npz(triplets / "validation_pool.npz")
    action_mask = np.asarray(pool["event_kind"], dtype=np.int8) == 2
    action_counts = pd.Series(
        np.asarray(pool["event_query"], dtype=np.int64)[action_mask]
    ).value_counts()
    tier_counts = Counter(selected["difficulty_tier"].astype(str))
    tier_frames = {
        tier: selected[selected["difficulty_tier"] == tier]
        for tier in ("easy", "medium", "hard")
    }
    gates = {
        "stage1_checkpoint_tensor_reconstruction_is_exact": True,
        "outer_held_formulas_are_absent": bool(all(
            stable_fold(str(value), 5, args.formula_fold_seed) != args.outer_fold
            for value in pool["event_formula"]
        )),
        "at_least_ten_thousand_action_triplets": len(selected) >= args.minimum_action_events,
        "all_three_difficulty_tiers_present_at_scale": bool(
            tier_counts["easy"] >= 1000
            and tier_counts["medium"] >= 1000
            and tier_counts["hard"] >= 4000
            and tier_counts["hard"] / len(selected) >= 0.35
        ),
        "hard_tier_is_empirically_harder_than_easy_tier": bool(
            tier_counts["easy"]
            and tier_counts["hard"]
            and tier_frames["hard"]["positive_similarity"].median()
            < tier_frames["easy"]["positive_similarity"].median()
            and tier_frames["hard"]["action_clean_similarity"].median()
            < tier_frames["easy"]["action_clean_similarity"].median()
        ),
        "maximum_four_action_events_per_query": bool(
            len(action_counts) and int(action_counts.max()) <= 4
        ),
        "every_action_view_is_unique": bool(
            selected["stage1_action_index"].nunique() == len(selected)
        ),
        "every_action_triplet_is_native_hinge_active": bool(
            np.all(selected["triplet_margin"].to_numpy(float) < args.margin)
        ),
        "every_positive_is_same_identity": bool(
            selected["positive_identity_verified"].astype(bool).all()
        ),
        "every_negative_is_different_identity": bool(
            selected["negative_identity_verified"].astype(bool).all()
            and np.all(selected["action_positive_row"] != selected["action_negative_row"])
        ),
        "registered_source_closure": set(selected["source"].astype(str))
        == REGISTERED_SOURCES,
        "targeted_and_control_payloads_are_distinct": bool(
            np.all(np.any(selected_targeted != selected_control, axis=(1, 2)))
        ),
        "clean_events_use_multi_member_positive_pools": bool(
            np.any(np.diff(pool["positive_ptr"])[~action_mask] > 1)
        ),
        "validation_pool_is_action_free": bool(
            np.all(np.asarray(validation_pool["event_kind"], dtype=np.int8) == 0)
            and np.all(
                np.asarray(validation_pool["event_action_index"], dtype=np.int64) == -1
            )
        ),
        "train_and_validation_formulas_are_disjoint": not bool(
            set(map(str, pool["event_formula"]))
            & set(map(str, validation_pool["event_formula"]))
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-3 multi-difficulty gates failed: {gates}")

    report = {
        "status": "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_COMPLETE",
        "builder_version": STAGE3_BUILDER_VERSION,
        "initialization_kind": initialization_kind,
        "scientific_contract": (
            "Noise severity stratified action anchors; easy/median/hard valid "
            "same-identity positives; current closest different-identity hard "
            "negative; native DreaMS triplet runtime only."
        ),
        "action_triplets": int(len(selected)),
        "action_queries": int(selected["query_index"].nunique()),
        "tier_counts": dict(sorted(tier_counts.items())),
        "clean_dynamic_events": int(len(clean_by_query)),
        "protection_events": int(len(sentinels)),
        "total_events": int(len(pool["anchor_idx"])),
        "source_counts": dict(sorted(Counter(selected["source"].astype(str)).items())),
        "difficulty_summary": {
            tier: {
                "events": int(len(frame)),
                "action_clean_similarity_median": float(
                    frame["action_clean_similarity"].median()
                ),
                "positive_similarity_median": float(frame["positive_similarity"].median()),
                "negative_similarity_median": float(frame["negative_similarity"].median()),
                "triplet_margin_median": float(frame["triplet_margin"].median()),
            }
            for tier, frame in selected.groupby("difficulty_tier", sort=True)
        },
        "gates": gates,
        "provenance": {
            "stage1_checkpoint_sha256": sha256_file(checkpoint),
            "stage1_triplet_report_sha256": sha256_file(triplets / "report.json"),
            "candidate_graph_sha256": sha256_file(args.graph),
            "embedding_registry_sha256": sha256_file(args.embedding_cache),
        },
        "outer_performance_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_stage3_", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        shutil.copy2(triplets / "validation_pool.npz", staging / "validation_pool.npz")
        np.savez_compressed(
            staging / "action_spectra.npz",
            targeted_action_spectra=selected_targeted,
            control_action_spectra=selected_control,
            native_action_view_representable=np.ones(len(selected), dtype=bool),
            stage1_action_index=selected_indices,
            difficulty_tier=fixed_unicode(selected["difficulty_tier"]),
        )
        selected.to_csv(staging / "selected_actions.csv.gz", index=False, compression="gzip")
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "selected_actions_sha256": sha256_file(staging / "selected_actions.csv.gz"),
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

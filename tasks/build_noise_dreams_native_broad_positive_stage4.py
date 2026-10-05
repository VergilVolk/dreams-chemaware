"""Build the corrected action-positive residual continuation from Noise Stage-1.

Stage-1 already consumed every registered action with one exact positive and
negative boundary.  Replaying those same triplets from the Stage-1 checkpoint
is not preservation: it is a second dose of already learned supervision.  The
failed v3 builder also put every new ``broad positive`` in an action-free clean
triplet, so no new Noise relation was added at all.

This repair keeps the native DreaMS trainer unchanged and changes membership
only.  Every representable Stage-1 action becomes the anchor of a *new*
same-identity relation:

    targeted/control action view -> original clean query plus additional
                                      measured same-identity spectra
                                   -> easy/medium/hard different-identity rows.

The exact Stage-1 positive is forbidden from the residual positive pool.  The
exact Stage-1 negative is excluded whenever another negative row exists.  The
few action-free events are only formula-disjoint batch fillers; the former
7,839 generic hardest-negative clean triplets are removed.
"""
from __future__ import annotations

import argparse
import hashlib
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


STAGE4_BUILDER_VERSION = "noise_native_identity_grounded_residual_v6"
STAGE4_STATUS = "NOISE_DREAMS_NATIVE_BROAD_POSITIVE_STAGE4_COMPLETE"
STAGE4_TOTAL_EVENTS = 33575  # maximum, not a target that may be padded with noise
STAGE4_OPTIMIZER_STEPS = 8394
STAGE4_MINIMUM_MARGIN = -0.1


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
    parser.add_argument("--minimum-margin", type=float, default=-0.1)
    parser.add_argument("--total-events", type=int, default=STAGE4_TOTAL_EVENTS)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def stable_digest_int(*values: object) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def preferred_tier(query: int, formula: str, ik14: str) -> str:
    # 25% easy, 50% medium, 25% semi-hard without depending on table order.
    return ("easy", "medium", "medium", "hard")[
        stable_digest_int("stage4-tier", query, formula, ik14) % 4
    ]


def candidate_rows(
    graph: CandidateGraph,
    query: int,
    query_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, rows, pointers, _ = graph.query_block(query)
    positive_end = int(pointers[1])
    positive = np.asarray(
        [int(row) for row in rows[:positive_end] if int(row) != query_row],
        dtype=np.int64,
    )
    negative = np.asarray(rows[positive_end:], dtype=np.int64)
    if not len(positive) or not len(negative):
        raise ValueError("query lacks an independent positive or negative")
    return positive, negative


def negative_molecule_rows(
    graph: CandidateGraph,
    query: int,
) -> list[np.ndarray]:
    """Return the different-identity candidate rows grouped by molecule."""
    _, rows, pointers, _ = graph.query_block(query)
    return [
        np.asarray(rows[int(left):int(right)], dtype=np.int64)
        for left, right in zip(pointers[1:-1], pointers[2:])
    ]


def three_level_rows(rows: np.ndarray, scores: np.ndarray) -> list[int]:
    """Select deterministic easy/medium/hard representatives without duplicates."""
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if rows.shape != scores.shape or not len(rows) or not np.all(np.isfinite(scores)):
        raise ValueError("difficulty rows and scores are not aligned")
    indices = tier_indices(scores)
    selected: list[int] = []
    for tier in ("easy", "medium", "hard"):
        row = int(rows[indices[tier]])
        if row not in selected:
            selected.append(row)
    return selected


def residual_negative_rows(
    groups: list[np.ndarray],
    row_position: dict[int, int],
    embeddings: np.ndarray,
    clean_vector: np.ndarray,
    exact_negative: int,
) -> tuple[list[int], bool]:
    """Choose three difficulty levels across molecules, avoiding the old row."""
    representatives: list[int] = []
    representative_scores: list[float] = []
    fallback: list[tuple[float, int]] = []
    for group in groups:
        available = [int(row) for row in group if int(row) in row_position]
        if not available:
            continue
        scores = np.asarray([
            float(embeddings[row_position[row]] @ clean_vector) for row in available
        ])
        order = np.argsort(-scores, kind="stable")
        fallback.extend((float(scores[index]), available[int(index)]) for index in order)
        nonold = [int(index) for index in order if available[int(index)] != exact_negative]
        if nonold:
            index = nonold[0]
            representatives.append(available[index])
            representative_scores.append(float(scores[index]))
    if representatives:
        return three_level_rows(
            np.asarray(representatives), np.asarray(representative_scores),
        ), False
    if not fallback:
        raise RuntimeError("residual action has no negative spectrum")
    fallback.sort(key=lambda body: (-body[0], body[1]))
    # A one-spectrum negative molecule can make the old row unavoidable.  The
    # positive relation is still new, and this exception is counted explicitly.
    return [int(fallback[0][1])], int(fallback[0][1]) == int(exact_negative)


def robust_dynamic_subpool(
    valid: np.ndarray,
    quality: np.ndarray,
    *,
    required_positive_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Largest all-valid subpool, optionally retaining one mandatory positive."""
    valid = np.asarray(valid, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)
    if valid.shape != quality.shape or valid.ndim != 2:
        raise ValueError("dynamic-pool validity arrays are not aligned")
    if required_positive_index is not None and not (
        0 <= int(required_positive_index) < valid.shape[0]
    ):
        raise ValueError("required positive index is outside the dynamic pool")
    best: tuple[tuple[float, ...], np.ndarray, np.ndarray] | None = None
    for positive_mask in range(1, 1 << valid.shape[0]):
        if (
            required_positive_index is not None
            and not (positive_mask & (1 << int(required_positive_index)))
        ):
            continue
        positive = np.asarray([
            index for index in range(valid.shape[0])
            if positive_mask & (1 << index)
        ], dtype=np.int64)
        for negative_mask in range(1, 1 << valid.shape[1]):
            negative = np.asarray([
                index for index in range(valid.shape[1])
                if negative_mask & (1 << index)
            ], dtype=np.int64)
            block = valid[np.ix_(positive, negative)]
            if not bool(np.all(block)):
                continue
            block_quality = quality[np.ix_(positive, negative)]
            key = (
                float(len(positive) * len(negative)),
                float(min(len(positive), len(negative))),
                float(len(positive) + len(negative)),
                float(np.min(block_quality)),
                float(np.mean(block_quality)),
            )
            if best is None or key > best[0]:
                best = (key, positive, negative)
    return None if best is None else (best[1], best[2])


def bounded_hinge_active_pairs(
    target_margin_grid: np.ndarray,
    *,
    minimum_margin: float,
    maximum_margin: float,
) -> np.ndarray:
    """Select native hinge-active difficulty without judging truth by the model."""
    margins = np.asarray(target_margin_grid, dtype=np.float64)
    if margins.ndim != 2 or not np.all(np.isfinite(margins)):
        raise ValueError("target margin grid is malformed")
    return (margins >= float(minimum_margin)) & (
        margins < float(maximum_margin)
    )


def tier_indices(scores: np.ndarray) -> dict[str, int]:
    """Return easy/median/semi-hard indices; never select the extreme minimum."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("positive scores are malformed")
    order = np.argsort(values, kind="stable")
    # The 25th percentile is deliberately semi-hard.  With one or two
    # positives it degrades to the available measured relation.
    hard_position = int(round(0.25 * (len(order) - 1)))
    return {
        "easy": int(order[-1]),
        "medium": int(order[len(order) // 2]),
        "hard": int(order[hard_position]),
    }


def choose_measured_relation(
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    tier: str,
    *,
    minimum_margin: float,
    maximum_margin: float,
) -> dict[str, object] | None:
    """Choose one active, bounded measured triplet without hardest-tail mining."""
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    negative_rows = np.asarray(negative_rows, dtype=np.int64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if (
        positive_rows.shape != positive_scores.shape
        or negative_rows.shape != negative_scores.shape
        or tier not in {"easy", "medium", "hard"}
        or not np.all(np.isfinite(positive_scores))
        or not np.all(np.isfinite(negative_scores))
    ):
        raise ValueError("measured relation arrays are not aligned")
    negative_index = int(np.argmax(negative_scores))
    negative_score = float(negative_scores[negative_index])
    indices = tier_indices(positive_scores)
    fallback = {
        "easy": ("easy", "medium", "hard"),
        "medium": ("medium", "hard", "easy"),
        "hard": ("hard", "medium", "easy"),
    }[tier]
    for selected_tier in fallback:
        positive_index = indices[selected_tier]
        positive_score = float(positive_scores[positive_index])
        margin = positive_score - negative_score
        if minimum_margin <= margin < maximum_margin:
            return {
                "positive_row": int(positive_rows[positive_index]),
                "negative_row": int(negative_rows[negative_index]),
                "positive_similarity": positive_score,
                "negative_similarity": negative_score,
                "triplet_margin": margin,
                "difficulty_tier": selected_tier,
                "requested_difficulty_tier": tier,
                "positive_pool_size": int(len(positive_rows)),
                "negative_pool_size": int(len(negative_rows)),
            }
    return None


def formula_round_robin(
    records: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    """Select deterministic formula-balanced query units without replacement."""
    if count < 0:
        raise ValueError("formula-balanced selection count is negative")
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record["query_formula"])].append(record)
    for formula in groups:
        groups[formula].sort(
            key=lambda body: stable_digest_int(
                "stage4-query", formula, body["query_ik14"], body["query_index"]
            )
        )
    formulas = sorted(
        groups,
        key=lambda formula: stable_digest_int("stage4-formula", formula),
    )
    selected: list[dict[str, object]] = []
    offset = 0
    while len(selected) < count:
        progressed = False
        for formula in formulas:
            group = groups[formula]
            if offset < len(group):
                selected.append(group[offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        offset += 1
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} formula-balanced triplets available for {count} slots"
        )
    return selected


def formula_coverage_first(
    records: list[dict[str, object]],
    count: int,
    existing_formulas: set[str],
) -> list[dict[str, object]]:
    """Maximize new formula coverage, then fill without repeating a query."""
    new_formula_records = [
        record for record in records
        if str(record["query_formula"]) not in existing_formulas
    ]
    new_formula_count = min(
        count, len({str(record["query_formula"]) for record in new_formula_records}),
    )
    selected = formula_round_robin(new_formula_records, new_formula_count)
    selected_queries = {int(record["query_index"]) for record in selected}
    remaining = count - len(selected)
    if remaining:
        candidates = [
            record for record in records
            if int(record["query_index"]) not in selected_queries
        ]
        new_query_candidates = [
            record for record in candidates
            if bool(record["broadens_action_query_coverage"])
        ]
        if len(new_query_candidates) >= remaining:
            candidates = new_query_candidates
        selected.extend(formula_round_robin(candidates, remaining))
    if len(selected) != count or len({int(r["query_index"]) for r in selected}) != count:
        raise RuntimeError("formula-coverage-first selection lost a measured query")
    return selected


def make_pool(selected: pd.DataFrame) -> tuple[dict[str, np.ndarray], int]:
    registry = Registry()
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    event_kind: list[int] = []
    event_query: list[int] = []
    event_action: list[int] = []
    event_formula: list[str] = []
    compact_action = 0
    for row in selected.itertuples(index=False):
        if bool(row.is_action_event):
            if bool(row.action_anchor_representable):
                anchor = registry.add(Registry.ACTION, compact_action)
                kind = 2
            else:
                # Exact Stage-1 policy for an all-zero action view: retain the
                # boundary once, but keep the tensor out of the model input.
                anchor = registry.add(Registry.HDF5, int(row.query_row))
                kind = 1
            action = compact_action
            compact_action += 1
        else:
            anchor = registry.add(Registry.HDF5, int(row.query_row))
            action = -1
            kind = 0
        positive = [
            registry.add(Registry.HDF5, int(value))
            for value in row.positive_rows
        ]
        negative = [
            registry.add(Registry.HDF5, int(value))
            for value in row.negative_rows
        ]
        if anchor in positive or anchor in negative or set(positive) & set(negative):
            raise RuntimeError("Stage-4 triplet roles overlap")
        anchors.append(anchor)
        positives.extend(positive)
        negatives.extend(negative)
        event_kind.append(kind)
        event_query.append(int(row.query_index))
        event_action.append(action)
        event_formula.append(str(row.query_formula))
    registry_kind, registry_source_index = registry.arrays()
    size = len(selected)
    return {
        "registry_kind": registry_kind,
        "registry_source_index": registry_source_index,
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum([len(value) for value in selected["positive_rows"]], dtype=np.int64),
        )),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum([len(value) for value in selected["negative_rows"]], dtype=np.int64),
        )),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "event_kind": np.asarray(event_kind, dtype=np.int8),
        "event_query": np.asarray(event_query, dtype=np.int64),
        "event_action_index": np.asarray(event_action, dtype=np.int64),
        "event_formula": fixed_unicode(event_formula),
    }, compact_action


def audit_stage1_action_retention(
    action_rows: pd.DataFrame,
    actions: pd.DataFrame,
    capable: np.ndarray,
) -> dict[str, bool]:
    """Independently prove that Stage-4 did not thin or mutate Stage-1 actions."""
    capable = np.asarray(capable, dtype=bool)
    if len(action_rows) != len(actions) or capable.shape != (len(actions),):
        return {
            "every_stage1_effective_action_is_retained_exactly_once": False,
            "every_stage1_action_keeps_its_exact_positive_negative_boundary": False,
            "action_representability_policy_matches_stage1": False,
        }
    ordered = action_rows.sort_values("stage1_action_index", kind="stable")
    return {
        "every_stage1_effective_action_is_retained_exactly_once": bool(
            np.array_equal(
                ordered["stage1_action_index"].to_numpy(np.int64),
                np.arange(len(actions), dtype=np.int64),
            )
        ),
        "every_stage1_action_keeps_its_exact_positive_negative_boundary": bool(
            np.array_equal(
                ordered["positive_row"].to_numpy(np.int64),
                actions["action_positive_row"].to_numpy(np.int64),
            )
            and np.array_equal(
                ordered["negative_row"].to_numpy(np.int64),
                actions["action_hard_negative_row"].to_numpy(np.int64),
            )
        ),
        "action_representability_policy_matches_stage1": bool(
            np.array_equal(
                ordered["action_anchor_representable"].to_numpy(bool), capable,
            )
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-4 triplet construction requires an allocated GPU")
    if (
        args.margin != 0.1
        or args.minimum_margin != STAGE4_MINIMUM_MARGIN
        or args.total_events != STAGE4_TOTAL_EVENTS
        or args.batch_size != 64
    ):
        raise RuntimeError("Stage-4 registered triplet settings drifted")

    stage1_triplets = args.stage1_run / "triplets"
    checkpoint = args.stage1_run / "checkpoint/targeted_final.ckpt"
    evaluation = args.stage1_run / "evaluation/targeted/report.json"
    required = (
        checkpoint,
        evaluation,
        stage1_triplets / "report.json",
        stage1_triplets / "validation_pool.npz",
        stage1_triplets / "action_spectra.npz",
        stage1_triplets / "selected_actions.csv.gz",
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
        raise RuntimeError("Stage-4 warm start differs from evaluated Stage-1 champion")

    graph = CandidateGraph(args.graph)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        rows = np.asarray(cache["rows"], dtype=np.int64)
    row_position = {int(row): index for index, row in enumerate(rows)}
    actions = pd.read_csv(
        stage1_triplets / "selected_actions.csv.gz", low_memory=False,
    )
    bank = load_npz(stage1_triplets / "action_spectra.npz")
    targeted_bank = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_bank = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if (
        len(actions) != len(targeted_bank)
        or targeted_bank.shape != control_bank.shape
        or capable.shape != (len(actions),)
    ):
        raise RuntimeError("Stage-1 action artifacts are not aligned")
    validation_pool = load_npz(stage1_triplets / "validation_pool.npz")
    validation_formulas = set(map(str, validation_pool["event_formula"]))

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
        label="noise-stage4-measured",
    )
    capable_indices = np.flatnonzero(capable)
    targeted_embeddings = encode_actions(
        model, targeted_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage4-targeted-actions",
        preprocessor=preprocessor,
    )
    control_embeddings = encode_actions(
        model, control_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage4-control-actions",
        preprocessor=preprocessor,
    )
    del model
    target_by_action = {
        int(action): targeted_embeddings[index]
        for index, action in enumerate(capable_indices)
    }
    control_by_action = {
        int(action): control_embeddings[index]
        for index, action in enumerate(capable_indices)
    }
    # Stage-1 already trained every exact action/positive/negative boundary.
    # Build a genuinely residual relation for every representable action: the
    # action remains the anchor, but the old exact positive is forbidden and a
    # multi-difficulty negative pool replaces the old singleton boundary.
    action_records: list[dict[str, object]] = []
    for action in range(len(actions)):
        if not bool(capable[action]):
            # The all-zero view cannot enter the native preprocessor.  Stage-1
            # already consumed its measured clean fallback; do not dose it twice.
            continue
        query = int(actions.at[action, "query_index"])
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("Stage-1 action ledger contains an outer-held formula")
        if formula in validation_formulas:
            raise RuntimeError("Stage-1 action ledger overlaps its validation formulas")
        query_row = int(graph.query_row[query])
        exact_positive = int(actions.at[action, "action_positive_row"])
        exact_negative = int(actions.at[action, "action_hard_negative_row"])
        if any(row not in row_position for row in (query_row, exact_positive, exact_negative)):
            raise RuntimeError("Stage-1 action boundary is absent from the frozen cache")
        clean_vector = embeddings[row_position[query_row]]
        positive_candidates, _ = candidate_rows(graph, query, query_row)
        # The original clean query is a valid same-identity target for the
        # noised action anchor and was not the Stage-1 independent positive.
        residual_positive_candidates = np.asarray(
            [query_row] + [
                int(row) for row in positive_candidates
                if int(row) != exact_positive and int(row) != query_row
            ],
            dtype=np.int64,
        )
        if any(int(row) not in row_position for row in residual_positive_candidates):
            raise RuntimeError("residual positive is absent from the frozen cache")
        # The original clean query is the defining hard-positive target and is
        # never delegated to a difficulty sampler.  Additional independent
        # measurements supply easy/medium/semi-hard diversity around it.
        additional_positive_candidates = residual_positive_candidates[
            residual_positive_candidates != query_row
        ]
        positive_rows = [query_row]
        if len(additional_positive_candidates):
            additional_positive_scores = embeddings[
                [row_position[int(row)] for row in additional_positive_candidates]
            ] @ clean_vector
            positive_rows.extend(three_level_rows(
                additional_positive_candidates, additional_positive_scores,
            ))
        clean_query_positive_index = 0
        negative_rows, reused_exact_negative = residual_negative_rows(
            negative_molecule_rows(graph, query), row_position, embeddings,
            clean_vector, exact_negative,
        )
        target_vector = target_by_action[action]
        control_vector = control_by_action[action]
        positive_vectors = embeddings[[row_position[row] for row in positive_rows]]
        negative_vectors = embeddings[[row_position[row] for row in negative_rows]]
        target_positive_values = positive_vectors @ target_vector
        control_positive_values = positive_vectors @ control_vector
        target_negative_values = negative_vectors @ target_vector
        control_negative_values = negative_vectors @ control_vector
        clean_positive_values = positive_vectors @ clean_vector
        clean_negative_values = negative_vectors @ clean_vector
        target_positive = float(np.mean(target_positive_values))
        target_negative = float(np.mean(negative_vectors @ target_vector))
        target_margin = target_positive - target_negative
        control_margin = float(
            np.mean(control_positive_values) - np.mean(control_negative_values)
        )
        clean_margin = float(
            np.mean(clean_positive_values) - np.mean(clean_negative_values)
        )
        target_margin_grid = (
            target_positive_values[:, None] - target_negative_values[None, :]
        )
        control_margin_grid = (
            control_positive_values[:, None] - control_negative_values[None, :]
        )
        clean_margin_grid = (
            clean_positive_values[:, None] - clean_negative_values[None, :]
        )
        positive_advantage_grid = np.broadcast_to(
            (target_positive_values - control_positive_values)[:, None],
            target_margin_grid.shape,
        )
        control_advantage_grid = target_margin_grid - control_margin_grid
        clean_advantage_grid = target_margin_grid - clean_margin_grid
        # Same-identity truth comes from the frozen candidate ledger, not from
        # whether the current embedding already knows the relation.  Requiring
        # targeted > control here selected only already-solved positives and
        # erased the hard-positive learning problem.  The matched control is a
        # causal training/evaluation arm; it is diagnostic before optimization,
        # never an eligibility label.  Eligibility is therefore limited to the
        # registered identity and a bounded, native hinge-active difficulty.
        valid_dynamic_pairs = bounded_hinge_active_pairs(
            target_margin_grid,
            minimum_margin=args.minimum_margin,
            maximum_margin=args.margin,
        )
        robust_pool = robust_dynamic_subpool(
            valid_dynamic_pairs,
            -np.abs(target_margin_grid),
            required_positive_index=clean_query_positive_index,
        )
        if robust_pool is None:
            continue
        positive_keep, negative_keep = robust_pool
        positive_rows = [positive_rows[int(index)] for index in positive_keep]
        negative_rows = [negative_rows[int(index)] for index in negative_keep]
        positive_vectors = positive_vectors[positive_keep]
        negative_vectors = negative_vectors[negative_keep]
        target_positive_values = target_positive_values[positive_keep]
        control_positive_values = control_positive_values[positive_keep]
        clean_positive_values = clean_positive_values[positive_keep]
        target_negative_values = target_negative_values[negative_keep]
        control_negative_values = control_negative_values[negative_keep]
        clean_negative_values = clean_negative_values[negative_keep]
        target_positive = float(np.mean(target_positive_values))
        target_negative = float(np.mean(target_negative_values))
        target_margin = target_positive - target_negative
        control_margin = float(
            np.mean(control_positive_values) - np.mean(control_negative_values)
        )
        clean_margin = float(
            np.mean(clean_positive_values) - np.mean(clean_negative_values)
        )
        target_margin_grid = (
            target_positive_values[:, None] - target_negative_values[None, :]
        )
        control_margin_grid = (
            control_positive_values[:, None] - control_negative_values[None, :]
        )
        clean_margin_grid = (
            clean_positive_values[:, None] - clean_negative_values[None, :]
        )
        minimum_positive_advantage = float(np.min(
            target_positive_values - control_positive_values
        ))
        minimum_control_margin_advantage = float(np.min(
            target_margin_grid - control_margin_grid
        ))
        additional_positive_mask = np.asarray(
            [int(row) != query_row for row in positive_rows], dtype=bool,
        )
        minimum_additional_clean_margin_advantage = (
            float(np.min(
                (target_margin_grid - clean_margin_grid)[additional_positive_mask]
            ))
            if np.any(additional_positive_mask) else float("nan")
        )
        action_records.append({
            "query_index": query,
            "query_row": query_row,
            "query_formula": formula,
            "query_ik14": str(graph.query_ik14[query]),
            "positive_rows": tuple(positive_rows),
            "negative_rows": tuple(negative_rows),
            "stage1_exact_positive_row": exact_positive,
            "stage1_exact_negative_row": exact_negative,
            "exact_negative_unavoidable": bool(reused_exact_negative),
            "positive_similarity": target_positive,
            "negative_similarity": target_negative,
            "triplet_margin": target_margin,
            "difficulty_tier": "dynamic_easy_medium_hard",
            "requested_difficulty_tier": "dynamic_easy_medium_hard",
            "positive_pool_size": len(positive_rows),
            "negative_pool_size": len(negative_rows),
            "is_action_event": True,
            "action_anchor_representable": True,
            "stage1_action_index": action,
            "action_id": str(actions.at[action, "action_id"]),
            "source": str(actions.at[action, "source"]),
            "family": str(actions.at[action, "family"]),
            "target_control_margin_advantage": target_margin - control_margin,
            "target_clean_margin_advantage": target_margin - clean_margin,
            "minimum_target_control_positive_advantage": minimum_positive_advantage,
            "minimum_target_control_margin_advantage": minimum_control_margin_advantage,
            "additional_positive_count": int(np.sum(additional_positive_mask)),
            "minimum_additional_target_clean_margin_advantage": (
                minimum_additional_clean_margin_advantage
            ),
            "minimum_target_margin": float(np.min(target_margin_grid)),
            "maximum_target_margin": float(np.max(target_margin_grid)),
            "relation_combinations": int(
                len(positive_rows) * len(negative_rows)
            ),
            "boundary_distance": float(abs(target_margin)),
            "control_margin": control_margin,
            "clean_same_boundary_margin": clean_margin,
        })
    all_action_records = pd.DataFrame(action_records)
    if all_action_records.empty:
        raise RuntimeError("Stage-4 found no representable residual action")
    # The failed Stage-3 run proved that action-row multiplicity must not become
    # optimizer dose. Exclude the destructive extreme margin tail and retain
    # exactly one identity-grounded, high-coverage relation per query.
    eligible = all_action_records.copy()
    eligible = eligible.sort_values(
        ["relation_combinations", "positive_pool_size", "negative_pool_size",
         "boundary_distance", "action_id"],
        ascending=[False, False, False, True, True], kind="stable",
    )
    action_records = eligible.drop_duplicates("query_index", keep="first").copy()
    action_records = action_records.sort_values(
        ["query_index", "action_id"], kind="stable",
    ).to_dict("records")
    if len(action_records) < 1500:
        raise RuntimeError(
            f"Stage-4 identity-grounded action queries {len(action_records)} < 1500"
        )

    # Every selected action query receives exactly one ordinary clean-boundary
    # event.  This is the preservation half of the empirically positive Stage-3
    # repair and prevents hard residuals from defining the whole local geometry.
    clean_preservation: list[dict[str, object]] = []
    for record in action_records:
        clean_preservation.append({
            **record,
            "positive_rows": (int(record["stage1_exact_positive_row"]),),
            "negative_rows": (int(record["stage1_exact_negative_row"]),),
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "triplet_margin": float(record["clean_same_boundary_margin"]),
            "difficulty_tier": "clean_preservation",
            "requested_difficulty_tier": "clean_preservation",
            "positive_pool_size": 1,
            "negative_pool_size": 1,
            "is_action_event": False,
            "action_anchor_representable": False,
            "stage1_action_index": -1,
            "action_id": "",
            "source": "measured_same_identity",
            "family": "clean_boundary_preservation",
        })

    # Add 256 formula-disjoint action-free protection queries, matching the
    # safe Stage-2/Stage-3 repair rather than the old four-row padding-only path.
    clean_fillers: list[dict[str, object]] = []
    action_queries = {int(record["query_index"]) for record in action_records}
    for query in range(graph.n_queries):
        if len(clean_fillers) == 256:
            break
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            continue
        if formula in validation_formulas:
            continue
        query_row = int(graph.query_row[query])
        if query in action_queries or query_row not in row_position:
            continue
        try:
            positive_rows, negative_rows = candidate_rows(graph, query, query_row)
        except ValueError:
            continue
        if any(int(row) not in row_position for row in np.concatenate((positive_rows, negative_rows))):
            continue
        clean_vector = embeddings[row_position[query_row]]
        selected_positive_rows = three_level_rows(
            positive_rows,
            embeddings[[row_position[int(row)] for row in positive_rows]] @ clean_vector,
        )
        selected_negative_rows, _ = residual_negative_rows(
            negative_molecule_rows(graph, query), row_position, embeddings,
            clean_vector, -1,
        )
        positive_vectors = embeddings[[row_position[row] for row in selected_positive_rows]]
        negative_vectors = embeddings[[row_position[row] for row in selected_negative_rows]]
        margin = float(np.mean(positive_vectors @ clean_vector) - np.mean(
            negative_vectors @ clean_vector
        ))
        clean_fillers.append({
                "positive_rows": tuple(selected_positive_rows),
                "negative_rows": tuple(selected_negative_rows),
                "stage1_exact_positive_row": -1,
                "stage1_exact_negative_row": -1,
                "exact_negative_unavoidable": False,
                "positive_similarity": float(np.mean(positive_vectors @ clean_vector)),
                "negative_similarity": float(np.mean(negative_vectors @ clean_vector)),
                "triplet_margin": margin,
                "difficulty_tier": "unmined_clean_filler",
                "requested_difficulty_tier": "unmined_clean_filler",
                "positive_pool_size": len(selected_positive_rows),
                "negative_pool_size": len(selected_negative_rows),
                "is_action_event": False,
                "action_anchor_representable": False,
                "stage1_action_index": -1,
                "action_id": "",
                "source": "measured_same_identity",
                "family": "measured_semihard_positive",
                "target_control_margin_advantage": 0.0,
                "target_clean_margin_advantage": 0.0,
                "control_margin": margin,
                "clean_same_boundary_margin": margin,
                "query_index": query,
                "query_row": query_row,
                "query_formula": formula,
                "query_ik14": str(graph.query_ik14[query]),
            })
    if len(clean_fillers) != 256:
        raise RuntimeError("Stage-4 residual ledger lacks 256 clean protection events")

    selected_records = action_records + clean_preservation + clean_fillers
    selected = pd.DataFrame(selected_records).sort_values(
        ["query_index", "is_action_event", "stage1_action_index"], kind="stable",
    ).reset_index(drop=True)
    pool, action_count = make_pool(selected)
    if action_count != int(selected["is_action_event"].sum()):
        raise RuntimeError("Stage-4 compact action registry is inconsistent")
    action_rows = selected.loc[selected["is_action_event"]].copy().reset_index(drop=True)
    selected_action_indices = action_rows["stage1_action_index"].to_numpy(np.int64)
    selected_targeted = targeted_bank[selected_action_indices]
    selected_control = control_bank[selected_action_indices]
    measured_rows = selected.loc[~selected["is_action_event"]]
    action_sources = Counter(action_rows["source"].astype(str))
    representable_actions = action_rows["action_anchor_representable"].to_numpy(bool)
    target_control_advantage = action_rows.loc[
        representable_actions, "target_control_margin_advantage"
    ].to_numpy(float)
    positive_pool_sizes = action_rows["positive_pool_size"].to_numpy(np.int64)
    negative_pool_sizes = action_rows["negative_pool_size"].to_numpy(np.int64)
    optimizer_steps = (len(selected) + (-len(selected)) % 4) // 4
    gates = {
        "stage1_champion_tensor_reconstruction_is_exact": True,
        "stage1_evaluated_checkpoint_is_the_warm_start": True,
        "residual_events_do_not_exceed_stage1_budget": len(selected) <= STAGE4_TOTAL_EVENTS,
        "residual_steps_do_not_exceed_stage1_budget": optimizer_steps <= STAGE4_OPTIMIZER_STEPS,
        "all_seven_registered_sources_entered_residual_selection": set(
            all_action_records["source"].astype(str)
        ) == {"N_mature", "P_guided_original", "E10B", "E11", "E12B",
              "A4_exact", "V4_gradient_path"},
        "selected_actions_retain_all_seven_registered_sources": set(
            action_rows["source"].astype(str)
        ) == {"N_mature", "P_guided_original", "E10B", "E11", "E12B",
              "A4_exact", "V4_gradient_path"},
        "exactly_one_action_event_per_query": (
            len(action_rows) == action_rows["query_index"].nunique()
        ),
        "every_action_query_has_one_clean_preservation_event": (
            len(clean_preservation) == len(action_rows)
            and {int(row["query_index"]) for row in clean_preservation}
            == set(map(int, action_rows["query_index"]))
        ),
        "hard_positive_membership_is_identity_grounded": True,
        "pretraining_target_control_advantage_not_used_for_eligibility": True,
        "all_dynamic_pairs_are_bounded_and_hinge_active": bool(
            (action_rows["minimum_target_margin"] >= args.minimum_margin).all()
            and (action_rows["maximum_target_margin"] < args.margin).all()
        ),
        "stage1_exact_positive_is_absent_from_every_residual_pool": all(
            int(row.stage1_exact_positive_row) not in set(map(int, row.positive_rows))
            for row in action_rows.itertuples(index=False)
        ),
        "every_residual_positive_pool_contains_original_clean_query": all(
            int(row.query_row) in set(map(int, row.positive_rows))
            for row in action_rows.itertuples(index=False)
        ),
        "at_least_fifteen_hundred_independent_action_queries": len(action_rows) >= 1500,
        "multiple_positive_levels_exist": bool(np.any(positive_pool_sizes > 1)),
        "multiple_negative_levels_exist": bool(np.any(negative_pool_sizes > 1)),
        "clean_stream_is_only_paired_preservation_plus_256_protection": (
            len(measured_rows) == len(action_rows) + 256
        ),
        "targeted_and_control_action_payloads_are_distinct": bool(
            len(representable_actions)
            and np.all(np.any(
                selected_targeted != selected_control,
                axis=(1, 2),
            ))
        ),
        "outer_held_formulas_are_absent": bool(all(
            stable_fold(str(value), 5, args.formula_fold_seed) != args.outer_fold
            for value in selected["query_formula"]
        )),
        "train_validation_formulas_are_disjoint": not bool(
            set(selected["query_formula"].astype(str))
            & set(map(str, validation_pool["event_formula"]))
        ),
        "validation_pool_is_action_free": bool(
            np.all(np.asarray(validation_pool["event_kind"], dtype=np.int8) == 0)
            and np.all(
                np.asarray(validation_pool["event_action_index"], dtype=np.int64) == -1
            )
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-4 broad-positive triplet gates failed: {gates}")

    report = {
        "status": STAGE4_STATUS,
        "builder_version": STAGE4_BUILDER_VERSION,
        "initialization_kind": initialization_kind,
        "scientific_contract": (
            "Continue the evaluated Stage-1 targeted champion; change only native "
            "triplet membership; never replay the Stage-1 exact positive boundary; "
            "consider every seven-source action, retain one identity-grounded "
            "multi-positive/multi-negative residual per query, pair it with one clean "
            "preservation event, and retain 256 action-free protection events."
        ),
        "selected_events": int(len(selected)),
        "selected_queries": int(selected["query_index"].nunique()),
        "selected_formulas": int(selected["query_formula"].nunique()),
        "selected_identities": int(selected["query_ik14"].nunique()),
        "action_events": int(len(action_rows)),
        "eligible_action_rows_before_query_collapse": int(len(eligible)),
        "all_representable_action_rows_considered": int(len(all_action_records)),
        "representable_action_events": int(np.sum(representable_actions)),
        "stage1_unrepresentable_actions_already_in_warm_start": int(np.sum(~capable)),
        "clean_boundary_action_fallback_events": 0,
        "measured_only_events": int(len(selected) - len(action_rows)),
        "clean_preservation_events": int(len(clean_preservation)),
        "action_free_protection_events": int(len(clean_fillers)),
        "action_positive_pool_size": {
            "minimum": int(positive_pool_sizes.min()),
            "median": float(np.median(positive_pool_sizes)),
            "maximum": int(positive_pool_sizes.max()),
        },
        "action_negative_pool_size": {
            "minimum": int(negative_pool_sizes.min()),
            "median": float(np.median(negative_pool_sizes)),
            "maximum": int(negative_pool_sizes.max()),
        },
        "unavoidable_exact_negative_reuses": int(
            action_rows["exact_negative_unavoidable"].astype(bool).sum()
        ),
        "action_sources": dict(sorted(action_sources.items())),
        "residual_action_signal": {
            "target_minus_control_margin_mean": float(
                np.mean(target_control_advantage)
            ),
            "target_wins_fraction": float(
                np.mean(target_control_advantage > 0.0)
            ),
            "selection_used_held_outcomes": False,
            "selection_used_pretraining_target_control_advantage": False,
            "pretraining_target_control_is_diagnostic_only": True,
            "old_exact_positive_boundaries_replayed": False,
            "all_representable_stage1_actions_were_considered": True,
        },
        "triplet_geometry": {
            "margin_minimum": float(selected["triplet_margin"].min()),
            "margin_median": float(selected["triplet_margin"].median()),
            "margin_maximum": float(selected["triplet_margin"].max()),
            "positive_similarity_median": float(
                selected["positive_similarity"].median()
            ),
            "negative_similarity_median": float(
                selected["negative_similarity"].median()
            ),
        },
        "frozen_training_budget": {
            "base_events": int(len(selected)),
            "final_batch_padding_events": (-len(selected)) % 4,
            "optimizer_steps": int(optimizer_steps),
            "stage1_maximum_events": STAGE4_TOTAL_EVENTS,
            "stage1_maximum_optimizer_steps": STAGE4_OPTIMIZER_STEPS,
            "does_not_pad_to_stage1_with_generic_triplets": True,
        },
        "gates": gates,
        "provenance": {
            "stage1_checkpoint_sha256": sha256_file(checkpoint),
            "stage1_evaluation_sha256": sha256_file(evaluation),
            "stage1_triplet_report_sha256": sha256_file(
                stage1_triplets / "report.json"
            ),
            "candidate_graph_sha256": sha256_file(args.graph),
            "embedding_registry_sha256": sha256_file(args.embedding_cache),
        },
        "outer_performance_claimed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_stage4_", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        shutil.copy2(
            stage1_triplets / "validation_pool.npz",
            staging / "validation_pool.npz",
        )
        np.savez_compressed(
            staging / "action_spectra.npz",
            targeted_action_spectra=selected_targeted,
            control_action_spectra=selected_control,
            native_action_view_representable=representable_actions,
            stage1_action_index=selected_action_indices,
        )
        action_rows.to_csv(
            staging / "selected_actions.csv.gz", index=False, compression="gzip",
        )
        selected_for_csv = selected.copy()
        for column in ("positive_rows", "negative_rows"):
            selected_for_csv[column] = selected_for_csv[column].map(
                lambda values: json.dumps(list(map(int, values)))
            )
        selected_for_csv.to_csv(
            staging / "selected_triplets.csv.gz", index=False, compression="gzip",
        )
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "selected_actions_sha256": sha256_file(
                staging / "selected_actions.csv.gz"
            ),
            "selected_triplets_sha256": sha256_file(
                staging / "selected_triplets.csv.gz"
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

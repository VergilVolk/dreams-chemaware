"""F1: transfer the support-disjoint C1 positive-evidence teacher into one
shared query/reference embedding encoder.

Only the zero-initialized peak adapter is trainable.  Query, held-out positive,
hard negative and the complete evaluation library all pass through that same
adapter.  C1 prototypes are stop-gradient training teachers and are unavailable
at inference.  P2b and every downstream reranker score are forbidden.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, ZeroInitPeakAdapter, json_dump, load_embedding_cache,
    seed_everything, sha256_file, strict_rank,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--d0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--f0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f0_protocol")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_parm")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-bound", type=float, default=0.10)
    parser.add_argument("--teacher-alpha", type=float, default=0.25)
    parser.add_argument("--objective", choices=("legacy_rank", "teacher_margin"), default="legacy_rank")
    parser.add_argument("--distill-negatives", type=int, default=4)
    parser.add_argument("--teacher-policy", choices=("all_improving", "corrected_only"), default="all_improving")
    parser.add_argument("--safety-ratio", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.25)
    parser.add_argument("--lambda-safety", type=float, default=4.0)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--full-r1-floor", type=float, default=-5e-4)
    parser.add_argument("--full-near-floor", type=float, default=-1e-3)
    parser.add_argument("--full-risk-floor", type=float, default=-5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-examples", type=int, default=0, help="smoke only")
    parser.add_argument("--max-eval-queries", type=int, default=0, help="smoke only")
    return parser.parse_args()


class TokenStore:
    def __init__(self, directory: Path, embedding_cache: Path):
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("status") != "noise_final_f1_full_token_cache_complete":
            raise RuntimeError("invalid F1 full-token cache")
        self.rows = np.load(directory / "rows.npy")
        self.tokens = np.load(directory / "tokens_f16.npy", mmap_mode="r")
        self.mz = np.load(directory / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(directory / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(directory / "valid.npy", mmap_mode="r")
        embedding_rows, embeddings, embedding_index = load_embedding_cache(embedding_cache)
        self.embedding_by_row = embedding_index
        self.embeddings = embeddings
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        if set(map(int, self.rows)) - set(map(int, embedding_rows)):
            raise RuntimeError("token cache contains a row absent from official embedding cache")
        self.dimension = int(embeddings.shape[1])

    def official(self, rows: np.ndarray) -> np.ndarray:
        return self.embeddings[np.asarray([self.embedding_by_row[int(row)] for row in rows], dtype=np.int64)]

    def adapt(self, adapter, rows: np.ndarray, device: torch.device):
        rows = np.asarray(rows, dtype=np.int64)
        positions = np.asarray([self.position[int(row)] for row in rows], dtype=np.int64)
        official = torch.from_numpy(self.official(rows)).to(device=device, dtype=torch.float32)
        tokens = torch.from_numpy(np.asarray(self.tokens[positions])).to(device=device, dtype=torch.float32)
        mz = torch.from_numpy(np.asarray(self.mz[positions])).to(device=device, dtype=torch.float32)
        intensity = torch.from_numpy(np.asarray(self.intensity[positions])).to(device=device, dtype=torch.float32)
        valid = torch.from_numpy(np.asarray(self.valid[positions])).to(device=device)
        return official, adapter(official, tokens, mz, intensity, valid)[0]


def normalized_mean(values: np.ndarray) -> np.ndarray:
    value = np.mean(values.astype(np.float64), axis=0)
    return (value / np.clip(np.linalg.norm(value), 1e-12, None)).astype(np.float32)


def hard_negative_rows(graph: CandidateGraph, allowed_molecule_mask: np.ndarray) -> np.ndarray:
    output = np.full(graph.n_queries, -1, dtype=np.int64)
    score_column = graph.dreams_column
    for query in range(graph.n_queries):
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        best_score = -np.inf
        best_row = -1
        for molecule in range(molecule_left + 1, molecule_right):
            if not allowed_molecule_mask[molecule]:
                continue
            left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
            local = int(np.argmax(graph.features[left:right, score_column]))
            score = float(graph.features[left + local, score_column])
            if score > best_score:
                best_score = score
                best_row = int(graph.pair_candidate_row[left + local])
        if best_row < 0:
            raise RuntimeError(f"query {query} has no allowed hard negative")
        output[query] = best_row
    return output


def top_negative_rows(graph: CandidateGraph, allowed_molecule_mask: np.ndarray,
                      count: int) -> np.ndarray:
    """Return the strongest official negative reference rows per query.

    Selection is by candidate molecule first (maximum official DreaMS score),
    so a molecule with many replicate spectra cannot occupy every top-k slot.
    """
    if count < 1:
        raise ValueError("distill-negatives must be positive")
    output = np.full((graph.n_queries, count), -1, dtype=np.int64)
    score_column = graph.dreams_column
    for query in range(graph.n_queries):
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        candidates: list[tuple[float, int]] = []
        for molecule in range(molecule_left + 1, molecule_right):
            if not allowed_molecule_mask[molecule]:
                continue
            left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
            local = int(np.argmax(graph.features[left:right, score_column]))
            pair = left + local
            candidates.append((float(graph.features[pair, score_column]), int(graph.pair_candidate_row[pair])))
        if not candidates:
            raise RuntimeError(f"query {query} has no allowed negative")
        candidates.sort(key=lambda value: (-value[0], value[1]))
        selected = [row for _, row in candidates[:count]]
        selected.extend([selected[-1]] * (count - len(selected)))
        output[query] = selected
    return output


def build_examples(c1: pd.DataFrame, graph: CandidateGraph, store: TokenStore,
                   negative_rows_by_query: np.ndarray, formula_fold: np.ndarray,
                   alpha: float) -> dict[str, np.ndarray]:
    required = {
        "query_index", "query_row", "evaluation_positive_row", "teacher_rows",
        "baseline_rank", "baseline_margin", "teacher_rank", "teacher_margin", "corrected",
        "introduced", "has_near", "query_ik14",
    }
    if not required.issubset(c1.columns):
        raise RuntimeError(f"C1 schema missing {sorted(required - set(c1.columns))}")
    n = len(c1)
    query_index = c1["query_index"].to_numpy(np.int64)
    query_row = c1["query_row"].to_numpy(np.int64)
    positive_row = c1["evaluation_positive_row"].to_numpy(np.int64)
    negative_rows = negative_rows_by_query[query_index]
    teacher = np.empty((n, store.dimension), dtype=np.float16)
    target_margin_delta = np.empty((n, negative_rows.shape[1]), dtype=np.float32)
    for index, (qrow, encoded_rows) in enumerate(zip(query_row, c1["teacher_rows"].astype(str))):
        rows = np.asarray([int(value) for value in encoded_rows.split(";") if value], dtype=np.int64)
        prototype = normalized_mean(store.official(rows))
        clean = store.official(np.asarray([qrow]))[0]
        mixed = (1.0 - alpha) * clean.astype(np.float64) + alpha * prototype.astype(np.float64)
        target = (mixed / np.clip(np.linalg.norm(mixed), 1e-12, None)).astype(np.float32)
        teacher[index] = target.astype(np.float16)
        positive = store.official(np.asarray([positive_row[index]], dtype=np.int64))[0].astype(np.float32)
        negative = store.official(negative_rows[index]).astype(np.float32)
        clean_margin = float(clean @ positive) - negative @ clean.astype(np.float32)
        target_margin = float(target @ positive) - negative @ target
        target_margin_delta[index] = np.clip(target_margin - clean_margin, 0.0, 0.5).astype(np.float32)
    gain = c1["teacher_margin"].to_numpy(np.float32) - c1["baseline_margin"].to_numpy(np.float32)
    eligible = (gain > 0) & (
        c1["teacher_rank"].to_numpy(np.int32) <= c1["baseline_rank"].to_numpy(np.int32)
    )
    eligible |= c1["corrected"].to_numpy(bool)
    teacher_weight = np.where(eligible, np.clip(gain / 0.10, 0.25, 2.0), 0.0).astype(np.float32)
    identity = c1["query_ik14"].astype(str).to_numpy()
    unique_identity, identity_inverse, identity_count = np.unique(
        identity, return_inverse=True, return_counts=True,
    )
    del unique_identity
    example_identity_weight = (1.0 / identity_count[identity_inverse]).astype(np.float32)
    example_identity_weight /= example_identity_weight.mean()
    return {
        "query_index": query_index, "query_row": query_row,
        "positive_row": positive_row, "negative_row": negative_rows[:, 0],
        "negative_rows": negative_rows,
        "target_margin_delta": target_margin_delta,
        "teacher": teacher, "teacher_weight": teacher_weight,
        "fold": formula_fold[query_index], "identity_weight": example_identity_weight,
        "baseline_rank": c1["baseline_rank"].to_numpy(np.int16),
        "has_near": c1["has_near"].to_numpy(bool),
        "identity": identity,
        "corrected_by_teacher": c1["corrected"].to_numpy(bool),
        "introduced_by_teacher": c1["introduced"].to_numpy(bool),
    }


@torch.no_grad()
def encode_all(adapter, store: TokenStore, device: torch.device, batch_size: int) -> np.ndarray:
    adapter.eval()
    output = np.empty((len(store.rows), store.dimension), dtype=np.float32)
    for left in range(0, len(store.rows), batch_size):
        right = min(left + batch_size, len(store.rows))
        _, adapted = store.adapt(adapter, store.rows[left:right], device)
        output[left:right] = adapted.cpu().numpy()
    return output


def evaluate_full(encoded: np.ndarray, store: TokenStore, graph: CandidateGraph,
                  query_subset: np.ndarray, symmetric_baseline: np.ndarray) -> dict:
    row_position = store.position
    query_pos = np.asarray([row_position[int(row)] for row in graph.query_row], dtype=np.int64)
    candidate_pos = np.asarray([row_position[int(row)] for row in graph.pair_candidate_row], dtype=np.int64)
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    pair_score = np.einsum("ij,ij->i", encoded[query_pos[pair_query]], encoded[candidate_pos])
    molecule_score = np.maximum.reduceat(pair_score, graph.molecule_ptr[:-1])
    rank = np.asarray([
        strict_rank(molecule_score[int(graph.query_ptr[q]):int(graph.query_ptr[q + 1])])
        for q in query_subset
    ], dtype=np.int16)
    old = symmetric_baseline[query_subset]
    base_correct, new_correct = old == 1, rank == 1
    near = graph.query_has_near[query_subset]
    official = store.official(store.rows)
    preservation = np.einsum("ij,ij->i", official, encoded)
    return {
        "query": query_subset, "old_rank": old, "new_rank": rank,
        "summary": {
            "n_queries": int(len(rank)),
            "baseline_recall1": float(np.mean(base_correct)),
            "recall1": float(np.mean(new_correct)),
            "delta_recall1": float(np.mean(new_correct) - np.mean(base_correct)),
            "baseline_mrr": float(np.mean(1.0 / old)),
            "mrr": float(np.mean(1.0 / rank)),
            "delta_mrr": float(np.mean(1.0 / rank) - np.mean(1.0 / old)),
            "corrected": int(np.sum(~base_correct & new_correct)),
            "introduced": int(np.sum(base_correct & ~new_correct)),
            "near_n": int(np.sum(near)),
            "baseline_near_recall1": float(np.mean(base_correct[near])),
            "near_recall1": float(np.mean(new_correct[near])),
            "delta_near_recall1": float(np.mean(new_correct[near]) - np.mean(base_correct[near])),
            "preservation_mean": float(np.mean(preservation)),
            "preservation_min": float(np.min(preservation)),
        },
    }


def evaluate_challenge(encoded: np.ndarray, store: TokenStore, graph: CandidateGraph,
                       examples: dict[str, np.ndarray], example_subset: np.ndarray,
                       baseline_rank: np.ndarray | None = None) -> dict:
    """Evaluate the exact C1 held-out-positive protocol used by the teacher.

    All same-identity teacher/reference spectra are hidden.  Only the declared
    evaluation-positive row competes against every negative molecule.
    """
    row_position = store.position
    query_position = np.asarray([row_position[int(row)] for row in graph.query_row], dtype=np.int64)
    candidate_position = np.asarray([row_position[int(row)] for row in graph.pair_candidate_row], dtype=np.int64)
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    pair_score = np.einsum("ij,ij->i", encoded[query_position[pair_query]], encoded[candidate_position])
    molecule_score = np.maximum.reduceat(pair_score, graph.molecule_ptr[:-1])
    rank = np.empty(len(example_subset), dtype=np.int16)
    for output_index, example_index in enumerate(example_subset):
        query = int(examples["query_index"][example_index])
        positive_row = int(examples["positive_row"][example_index])
        positive_score = float(
            encoded[query_position[query]] @ encoded[row_position[positive_row]]
        )
        left, right = map(int, graph.query_ptr[query:query + 2])
        negatives = molecule_score[left + 1:right]
        rank[output_index] = 1 + int(np.sum(negatives >= positive_score))
    old = (
        examples["baseline_rank"][example_subset]
        if baseline_rank is None else np.asarray(baseline_rank[example_subset], dtype=np.int16)
    )
    near = examples["has_near"][example_subset]
    weight = examples["identity_weight"][example_subset].astype(np.float64)
    weight /= weight.sum()
    base_correct, new_correct = old == 1, rank == 1
    corrected = ~base_correct & new_correct
    introduced = base_correct & ~new_correct
    near_weight = weight * near
    near_weight /= np.clip(near_weight.sum(), 1e-12, None)
    return {
        "example": example_subset, "old_rank": old, "new_rank": rank,
        "summary": {
            "n_examples": int(len(rank)),
            "n_query_identities": int(len(np.unique(
                graph.query_ik14[examples["query_index"][example_subset]]
            ))),
            "baseline_recall1": float(np.mean(base_correct)),
            "recall1": float(np.mean(new_correct)),
            "delta_recall1": float(np.mean(new_correct) - np.mean(base_correct)),
            "identity_equal_baseline_recall1": float(np.sum(weight * base_correct)),
            "identity_equal_recall1": float(np.sum(weight * new_correct)),
            "identity_equal_delta_recall1": float(np.sum(weight * (new_correct.astype(float) - base_correct.astype(float)))),
            "corrected": int(np.sum(corrected)), "introduced": int(np.sum(introduced)),
            "identity_equal_corrected_fraction": float(np.sum(weight * corrected)),
            "identity_equal_introduced_fraction": float(np.sum(weight * introduced)),
            "risk_weighted_net_per_example": float(np.mean(corrected.astype(float) - 2.0 * introduced.astype(float))),
            "identity_equal_risk_net": float(np.sum(weight * (corrected.astype(float) - 2.0 * introduced.astype(float)))),
            "near_n": int(np.sum(near)),
            "identity_equal_near_delta_recall1": float(np.sum(
                near_weight * (new_correct.astype(float) - base_correct.astype(float))
            )),
        },
    }


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("F1 requires CUDA")
    seed_everything(args.seed)
    device = torch.device(args.device)
    required = [
        args.graph, args.d0_dir / "manifest.npz", args.f0_dir / "decision.json",
        args.f0_dir / "symmetric_zero_rank.npy", args.f0_dir / "allowed_molecule_mask.npy",
        args.c1_dir / "crossfit_examples.csv.gz", args.c1_dir / "decision.json",
        args.token_dir / "report.json", args.embedding_cache,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    f0 = json.loads((args.f0_dir / "decision.json").read_text(encoding="utf-8"))
    if not f0.get("pass") or f0["training_contract"].get("p2b", "").lower().find("forbidden") < 0:
        raise RuntimeError("F1 requires passing P2b-free symmetric F0")
    c1_decision = json.loads((args.c1_dir / "decision.json").read_text(encoding="utf-8"))
    if not c1_decision.get("pass_to_candidate_aware_student"):
        raise RuntimeError("C1 teacher did not pass")

    graph = CandidateGraph(args.graph)
    with np.load(args.d0_dir / "manifest.npz") as body:
        formula_fold = np.asarray(body["formula_fold"], dtype=np.int8)
    symmetric_baseline = np.load(args.f0_dir / "symmetric_zero_rank.npy")
    allowed_molecule_mask = np.load(args.f0_dir / "allowed_molecule_mask.npy")
    store = TokenStore(args.token_dir, args.embedding_cache)
    if store.dimension != 1024 or store.tokens.shape[2] != store.dimension:
        raise RuntimeError("F1 requires full 1024-dimensional official peak tokens")
    negative_rows_by_query = top_negative_rows(
        graph, allowed_molecule_mask, args.distill_negatives,
    )
    c1 = pd.read_csv(args.c1_dir / "crossfit_examples.csv.gz")
    examples = build_examples(
        c1, graph, store, negative_rows_by_query, formula_fold, args.teacher_alpha,
    )
    if args.teacher_policy == "corrected_only":
        examples["teacher_weight"] = np.where(
            examples["corrected_by_teacher"], examples["teacher_weight"], 0.0,
        ).astype(np.float32)
    inner_fold = (args.outer_fold + 1) % 5
    train = np.flatnonzero((examples["fold"] != args.outer_fold) & (examples["fold"] != inner_fold))
    inner_query = np.flatnonzero(formula_fold == inner_fold)
    outer_query = np.flatnonzero(formula_fold == args.outer_fold)
    if args.max_train_examples:
        train = train[:args.max_train_examples]
    if args.max_eval_queries:
        inner_query = inner_query[:args.max_eval_queries]
        outer_query = outer_query[:args.max_eval_queries]
    inner_examples = np.flatnonzero(np.isin(examples["query_index"], inner_query))
    outer_examples = np.flatnonzero(np.isin(examples["query_index"], outer_query))
    if not len(train) or not len(inner_query) or not len(outer_query) or not len(inner_examples) or not len(outer_examples):
        raise RuntimeError("empty F1 split")
    rescue_train = train[examples["teacher_weight"][train] > 0]
    safety_train = train[
        (examples["baseline_rank"][train] == 1)
        & ~examples["corrected_by_teacher"][train]
    ]
    if not len(rescue_train):
        raise RuntimeError("F1 split has no teacher rescue examples")
    train_audit_examples = rescue_train[:min(len(rescue_train), 5000)]

    adapter = ZeroInitPeakAdapter(store.dimension, args.hidden_dim, args.delta_bound).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    initial_full = evaluate_full(encoded, store, graph, inner_query, symmetric_baseline)
    if initial_full["summary"]["corrected"] or initial_full["summary"]["introduced"]:
        raise RuntimeError("zero-init F1 does not reproduce symmetric F0")
    c1_reconciliation = evaluate_challenge(encoded, store, graph, examples, inner_examples)
    challenge_zero_rank = examples["baseline_rank"].copy()
    challenge_zero_rank[inner_examples] = c1_reconciliation["new_rank"]
    challenge_zero_rank[outer_examples] = evaluate_challenge(
        encoded, store, graph, examples, outer_examples,
    )["new_rank"]
    challenge_zero_rank[train_audit_examples] = evaluate_challenge(
        encoded, store, graph, examples, train_audit_examples,
    )["new_rank"]
    initial_challenge = evaluate_challenge(
        encoded, store, graph, examples, inner_examples, challenge_zero_rank,
    )
    initial_train_challenge = evaluate_challenge(
        encoded, store, graph, examples, train_audit_examples, challenge_zero_rank,
    )
    c1_rank_mismatches = int(np.sum(c1_reconciliation["old_rank"] != c1_reconciliation["new_rank"]))
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch, best_utility = 0, 0.0
    history = [{
        "epoch": 0,
        "inner_full_graph": initial_full["summary"],
        "inner_c1_challenge": initial_challenge["summary"],
        "train_rescue_challenge": initial_train_challenge["summary"],
        "selection_utility": 0.0,
        "eligible": True,
    }]
    rng = np.random.default_rng(args.seed)
    print(
        f"[F1 fold={args.outer_fold} seed={args.seed}] epoch 0 "
        f"full={initial_full['summary']} challenge={initial_challenge['summary']} "
        f"c1_rank_mismatches={c1_rank_mismatches}", flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        if args.teacher_policy == "corrected_only":
            safety_target = min(len(safety_train), int(round(args.safety_ratio * len(rescue_train))))
            mandatory = safety_train[examples["introduced_by_teacher"][safety_train]]
            mandatory = mandatory[:safety_target]
            remaining = np.setdiff1d(safety_train, mandatory, assume_unique=False)
            extra_count = max(0, safety_target - len(mandatory))
            extra = rng.choice(remaining, size=extra_count, replace=False) if extra_count else np.empty(0, dtype=np.int64)
            epoch_pool = np.concatenate((rescue_train, mandatory, extra))
        else:
            epoch_pool = train
        pool_identity = examples["identity"][epoch_pool]
        _, pool_inverse, pool_count = np.unique(pool_identity, return_inverse=True, return_counts=True)
        pool_weight = (1.0 / pool_count[pool_inverse]).astype(np.float32)
        pool_weight /= pool_weight.mean()
        epoch_weight = np.zeros(len(examples["query_index"]), dtype=np.float32)
        epoch_weight[epoch_pool] = pool_weight
        order = rng.permutation(epoch_pool)
        component_sum = {name: 0.0 for name in ("loss", "teacher", "rank", "safety", "preserve")}
        batches = 0
        started = time.time()
        for left in range(0, len(order), args.batch_size):
            index = order[left:left + args.batch_size]
            qrow = examples["query_row"][index]
            prow = examples["positive_row"][index]
            nrow = examples["negative_rows"][index]
            negative_count = nrow.shape[1]
            joined = np.concatenate((qrow, prow, nrow.reshape(-1)))
            unique, inverse = np.unique(joined, return_inverse=True)
            official_unique, adapted_unique = store.adapt(adapter, unique, device)
            count = len(index)
            q = adapted_unique[inverse[:count]]
            p = adapted_unique[inverse[count:2 * count]]
            n = adapted_unique[inverse[2 * count:]].reshape(count, negative_count, -1)
            q0 = official_unique[inverse[:count]]
            p0 = official_unique[inverse[count:2 * count]]
            n0 = official_unique[inverse[2 * count:]].reshape(count, negative_count, -1)
            teacher = torch.from_numpy(examples["teacher"][index].astype(np.float32)).to(device)
            teacher_weight = torch.from_numpy(examples["teacher_weight"][index]).to(device)
            sample_weight = torch.from_numpy(epoch_weight[index]).to(device)
            sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-8)
            spos = torch.sum(q * p, dim=1)
            sneg = torch.sum(q[:, None, :] * n, dim=2)
            old_pos = torch.sum(q0 * p0, dim=1)
            old_neg = torch.sum(q0[:, None, :] * n0, dim=2)
            teacher_loss = (1.0 - torch.sum(q * teacher, dim=1)) * teacher_weight
            old_margin = old_pos[:, None] - old_neg
            new_margin = spos[:, None] - sneg
            if args.objective == "legacy_rank":
                rank_loss = F.softplus(
                    (args.rank_margin + sneg[:, 0] - spos) / args.temperature
                )
                safety = (
                    F.relu(old_margin[:, 0] - new_margin[:, 0])
                    * (old_margin[:, 0] > 0).float()
                )
            else:
                target_delta = torch.from_numpy(
                    examples["target_margin_delta"][index].astype(np.float32)
                ).to(device)
                target_mask = (target_delta > 1e-6).float()
                element = F.smooth_l1_loss(
                    new_margin - old_margin, target_delta,
                    reduction="none", beta=0.05,
                )
                rank_loss = (
                    torch.sum(element * target_mask, dim=1)
                    / target_mask.sum(dim=1).clamp_min(1.0)
                ) * teacher_weight
                safety_mask = (old_margin > 0).float()
                safety = (
                    torch.sum(F.relu(old_margin - new_margin) * safety_mask, dim=1)
                    / safety_mask.sum(dim=1).clamp_min(1.0)
                )
            preserve = (
                (1.0 - torch.sum(q * q0, dim=1))
                + (1.0 - torch.sum(p * p0, dim=1))
                + torch.mean(1.0 - torch.sum(n * n0, dim=2), dim=1)
            ) / 3.0
            per_example = (
                args.lambda_teacher * teacher_loss + args.lambda_rank * rank_loss
                + args.lambda_safety * safety + args.lambda_preserve * preserve
            )
            loss = torch.mean(per_example * sample_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite F1 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            for name, value in (
                ("loss", loss), ("teacher", teacher_loss.mean()), ("rank", rank_loss.mean()),
                ("safety", safety.mean()), ("preserve", preserve.mean()),
            ):
                component_sum[name] += float(value.detach())
            batches += 1
        encoded = encode_all(adapter, store, device, args.eval_batch_size)
        inner_full = evaluate_full(encoded, store, graph, inner_query, symmetric_baseline)
        inner_challenge = evaluate_challenge(
            encoded, store, graph, examples, inner_examples, challenge_zero_rank,
        )
        train_challenge = evaluate_challenge(
            encoded, store, graph, examples, train_audit_examples, challenge_zero_rank,
        )
        full_summary = inner_full["summary"]
        challenge_summary = inner_challenge["summary"]
        utility = (
            challenge_summary["identity_equal_risk_net"]
            + 0.25 * challenge_summary["identity_equal_near_delta_recall1"]
        )
        full_risk = (
            full_summary["corrected"] - 2.0 * full_summary["introduced"]
        ) / full_summary["n_queries"]
        eligible = (
            full_summary["preservation_mean"] >= args.minimum_preservation
            and full_summary["delta_recall1"] >= args.full_r1_floor
            and full_summary["delta_near_recall1"] >= args.full_near_floor
            and full_risk >= args.full_risk_floor
        )
        if eligible and utility > best_utility + 1e-12:
            best_state = copy.deepcopy(adapter.state_dict())
            best_epoch, best_utility = epoch, float(utility)
        record = {
            "epoch": epoch,
            "train": {name: value / max(batches, 1) for name, value in component_sum.items()},
            "inner_full_graph": full_summary,
            "inner_c1_challenge": challenge_summary,
            "train_rescue_challenge": train_challenge["summary"],
            "epoch_training_examples": int(len(epoch_pool)),
            "full_risk_net_per_query": float(full_risk),
            "selection_utility": float(utility), "eligible": bool(eligible),
            "seconds": time.time() - started,
        }
        history.append(record)
        print(f"[F1 fold={args.outer_fold} seed={args.seed}] {record}", flush=True)

    adapter.load_state_dict(best_state)
    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    outer_full = evaluate_full(encoded, store, graph, outer_query, symmetric_baseline)
    outer_challenge = evaluate_challenge(
        encoded, store, graph, examples, outer_examples, challenge_zero_rank,
    )
    output = args.output_root / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "status": "noise_final_f1_parm_shared_encoder",
        "adapter_state": {key: value.cpu() for key, value in best_state.items()},
        "adapter_config": {"embedding_dim": store.dimension, "hidden_dim": args.hidden_dim,
                           "delta_bound": args.delta_bound},
        "seed": args.seed, "outer_fold": args.outer_fold, "inner_fold": inner_fold,
        "best_epoch": best_epoch, "best_inner_utility": best_utility,
        "objective": args.objective, "distill_negatives": args.distill_negatives,
        "teacher_policy": args.teacher_policy, "safety_ratio": args.safety_ratio,
        "P2b_used": False, "query_reference_encoder_shared": True,
        "f0_decision_sha256": sha256_file(args.f0_dir / "decision.json"),
        "token_report_sha256": sha256_file(args.token_dir / "report.json"),
    }, output / "adapter.pt")
    np.savez_compressed(
        output / "outer_full_predictions.npz", query=outer_full["query"],
        old_rank=outer_full["old_rank"], new_rank=outer_full["new_rank"],
    )
    np.savez_compressed(
        output / "outer_challenge_predictions.npz", example=outer_challenge["example"],
        old_rank=outer_challenge["old_rank"], new_rank=outer_challenge["new_rank"],
    )
    decision = {
        "status": "noise_final_f1_parm_fold_complete", "seed": args.seed,
        "outer_fold": args.outer_fold, "inner_fold": inner_fold,
        "train_examples": int(len(train)), "teacher_eligible_examples": int(np.sum(examples["teacher_weight"][train] > 0)),
        "rescue_train_examples": int(len(rescue_train)),
        "safety_train_examples": int(len(safety_train)),
        "best_epoch": best_epoch, "best_inner_utility": best_utility,
        "objective": args.objective, "distill_negatives": args.distill_negatives,
        "teacher_policy": args.teacher_policy, "safety_ratio": args.safety_ratio,
        "distillation_target": {
            "positive_pair_fraction": float(np.mean(examples["target_margin_delta"][train] > 1e-6)),
            "mean_positive_margin_delta": float(np.mean(
                examples["target_margin_delta"][train][examples["target_margin_delta"][train] > 1e-6]
            )) if np.any(examples["target_margin_delta"][train] > 1e-6) else 0.0,
            "maximum_margin_delta": float(np.max(examples["target_margin_delta"][train])),
        },
        "outer_full_graph": outer_full["summary"],
        "outer_c1_challenge": outer_challenge["summary"],
        "c1_zero_init_rank_mismatches_inner": c1_rank_mismatches,
        "selection_protocol": "C1 held-out-positive challenge; full natural graph is a safety gate",
        "history": history,
        "query_reference_encoder_shared": True, "P2b_used": False,
        "claim_limit": "formula-outer-fold P-arm student result; no P3, no N-arm, no downstream reranker",
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

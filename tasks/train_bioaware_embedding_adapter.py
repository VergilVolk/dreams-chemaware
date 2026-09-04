#!/usr/bin/env python
"""Train one candidate-independent BioAware shared embedding adapter.

The primary objective is spectrum retrieval.  Rhea neighbours enter only as a
typed auxiliary relation task; they are never treated as positive spectra.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, ZeroInitPeakAdapter, json_dump, seed_everything, sha256_file  # noqa: E402
from train_noise_final_f1_parm import TokenStore, encode_all, evaluate_full, top_negative_rows  # noqa: E402


RELATION_CLASS = {
    "same_identity": 0,
    "reaction": 1,
    "near_isomer": 2,
    "other": 3,
}


class RelationHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, 4),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((torch.abs(left - right), left * right), dim=-1))


def relation_family(value: str) -> str:
    if value == "same_identity":
        return "same_identity"
    if value == "near_isomer":
        return "near_isomer"
    if value.startswith("reaction_"):
        return "reaction"
    return "other"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/validation/bioaware_embedding_relation_manifest")
    parser.add_argument("--preflight", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter_preflight.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--d0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--f0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f0_protocol")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter")
    parser.add_argument("--oof-decision", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter_oof.json")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-bound", type=float, default=0.06)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-relation", type=float, default=0.10)
    parser.add_argument("--relation-gradient-ratio-cap", type=float, default=0.25)
    parser.add_argument("--lambda-safety", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=2.0)
    parser.add_argument("--safety-slack", type=float, default=0.01)
    parser.add_argument("--relation-warmup-epochs", type=int, default=1)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-eval-queries", type=int, default=0, help="smoke only")
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic execution only")
    return parser.parse_args()


def sample_rows(rng: np.random.Generator, mapping: dict[str, np.ndarray], identities: list[str], count=1) -> np.ndarray:
    output = []
    for identity in identities:
        values = mapping[identity]
        output.append(int(values[rng.integers(len(values))]))
    return np.asarray(output, dtype=np.int64)


def sample_distinct_same_identity_rows(
    rng: np.random.Generator, mapping: dict[str, np.ndarray], identities: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    left = sample_rows(rng, mapping, identities)
    right = sample_rows(rng, mapping, identities)
    for index, identity in enumerate(identities):
        values = mapping[identity]
        if right[index] == left[index]:
            position = int(np.flatnonzero(values == left[index])[0])
            right[index] = values[(position + 1) % len(values)]
    return left, right


def adapter_gradient_norm(loss: torch.Tensor, adapter: nn.Module) -> float:
    gradients = torch.autograd.grad(
        loss, tuple(adapter.parameters()), retain_graph=True, allow_unused=True,
    )
    squared = sum(
        torch.sum(value.detach() ** 2)
        for value in gradients if value is not None
    )
    return float(torch.sqrt(squared.clamp_min(0.0)))


def relation_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Small dependency-free multiclass diagnostic for the frozen relation head."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if len(labels) == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None,
                "reaction_precision": None, "reaction_recall": None}
    f1_values = []
    for label in range(len(RELATION_CLASS)):
        true_positive = int(np.sum((labels == label) & (predictions == label)))
        false_positive = int(np.sum((labels != label) & (predictions == label)))
        false_negative = int(np.sum((labels == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    reaction = RELATION_CLASS["reaction"]
    reaction_tp = int(np.sum((labels == reaction) & (predictions == reaction)))
    reaction_fp = int(np.sum((labels != reaction) & (predictions == reaction)))
    reaction_fn = int(np.sum((labels == reaction) & (predictions != reaction)))
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "reaction_precision": float(reaction_tp / max(reaction_tp + reaction_fp, 1)),
        "reaction_recall": float(reaction_tp / max(reaction_tp + reaction_fn, 1)),
        "class_counts": {
            family: int(np.sum(labels == label))
            for family, label in RELATION_CLASS.items()
        },
    }


@torch.no_grad()
def evaluate_relation_readout(
    adapter: nn.Module,
    relation_head: nn.Module,
    store: TokenStore,
    row_by_identity: dict[str, np.ndarray],
    table: pd.DataFrame,
    device: torch.device,
) -> dict:
    """Evaluate a train-only relation head without using the result for selection.

    One deterministic spectrum is used per identity.  The same frozen head is
    evaluated on official and adapted embeddings, so the delta measures whether
    the adapter made the held-out typed relation more readable rather than
    whether the auxiliary head merely memorised the training pairs.
    """
    if table.empty:
        empty = relation_metrics(np.empty(0, np.int64), np.empty(0, np.int64))
        return {"official": empty, "adapted": empty, "adapted_minus_official_accuracy": None}
    left_rows = np.asarray(
        [int(np.min(row_by_identity[str(value)])) for value in table.identity_a],
        dtype=np.int64,
    )
    right_rows = np.asarray(
        [int(np.min(row_by_identity[str(value)])) for value in table.identity_b],
        dtype=np.int64,
    )
    labels = table.relation_type.astype(str).map(
        lambda value: RELATION_CLASS[relation_family(value)]
    ).to_numpy(np.int64)
    relation_head.eval(); adapter.eval()
    official_left = torch.from_numpy(store.official(left_rows)).to(device)
    official_right = torch.from_numpy(store.official(right_rows)).to(device)
    _, adapted_left = store.adapt(adapter, left_rows, device)
    _, adapted_right = store.adapt(adapter, right_rows, device)
    official_predictions = relation_head(official_left, official_right).argmax(dim=1).cpu().numpy()
    adapted_predictions = relation_head(adapted_left, adapted_right).argmax(dim=1).cpu().numpy()
    official = relation_metrics(labels, official_predictions)
    adapted = relation_metrics(labels, adapted_predictions)
    return {
        "official": official,
        "adapted": adapted,
        "adapted_minus_official_accuracy": float(adapted["accuracy"] - official["accuracy"]),
        "selection_use": "diagnostic_only",
    }


def best_positive_rows(graph: CandidateGraph) -> np.ndarray:
    output = np.empty(graph.n_queries, dtype=np.int64)
    column = graph.dreams_column
    for query in range(graph.n_queries):
        molecule = int(graph.query_ptr[query])
        left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
        output[query] = int(graph.pair_candidate_row[left + np.argmax(graph.features[left:right, column])])
    return output


def main() -> None:
    args = arguments()
    if args.outer_fold not in {-1, 0, 1, 2, 3, 4}:
        raise ValueError("outer-fold must be -1 (frozen final refit) or 0..4")
    final_refit = args.outer_fold == -1
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BioAware embedding adapter requires CUDA")
    required = [
        args.manifest_dir / "report.json", args.manifest_dir / "rows.csv.gz",
        args.manifest_dir / "identity_pairs.csv.gz", args.preflight, args.graph,
        args.d0_dir / "manifest.npz", args.f0_dir / "symmetric_zero_rank.npy",
        args.token_dir / "report.json", args.embedding_cache,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if not preflight.get("formal") or not all(preflight.get("gates", {}).values()):
        raise RuntimeError("BioAware shared-embedding preflight did not pass")
    manifest_report = json.loads((args.manifest_dir / "report.json").read_text(encoding="utf-8"))
    if not manifest_report.get("formal") or manifest_report["contracts"].get("reaction_neighbour_is_positive"):
        raise RuntimeError("invalid BioAware relation manifest contract")
    oof_decision = None
    if final_refit:
        if not args.oof_decision.exists():
            raise FileNotFoundError(f"final refit requires passing frozen OOF decision: {args.oof_decision}")
        oof_decision = json.loads(args.oof_decision.read_text(encoding="utf-8"))
        if oof_decision.get("status") != "bioaware_embedding_adapter_formula_oof_complete" or not oof_decision.get("gates", {}).get("pass"):
            raise RuntimeError("final refit is forbidden until formula-OOF gates pass")
        expected = oof_decision.get("frozen_hyperparameters")
        observed = {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "steps_per_epoch": args.steps_per_epoch, "lr": args.lr,
            "weight_decay": args.weight_decay, "hidden_dim": args.hidden_dim,
            "delta_bound": args.delta_bound, "rank_margin": args.rank_margin,
            "temperature": args.temperature, "lambda_relation": args.lambda_relation,
            "relation_gradient_ratio_cap": args.relation_gradient_ratio_cap,
            "lambda_safety": args.lambda_safety, "lambda_preserve": args.lambda_preserve,
            "safety_slack": args.safety_slack,
            "relation_warmup_epochs": args.relation_warmup_epochs,
        }
        if expected != observed:
            raise RuntimeError(f"final refit hyperparameters differ from OOF recipe: expected={expected}, observed={observed}")
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed + max(args.outer_fold, 0))
    device = torch.device(args.device)
    graph = CandidateGraph(args.graph)
    store = TokenStore(args.token_dir, args.embedding_cache)
    rows = pd.read_csv(args.manifest_dir / "rows.csv.gz")
    rows = rows[rows["row"].astype(int).isin(set(map(int, store.rows)))].copy()
    row_by_identity = {
        str(identity): group["row"].to_numpy(np.int64)
        for identity, group in rows.groupby("ik14", sort=True)
        if len(group) >= 2
    }
    pairs = pd.read_csv(args.manifest_dir / "identity_pairs.csv.gz")
    pair_mask = (
        pairs.identity_a.astype(str).isin(row_by_identity)
        & pairs.identity_b.astype(str).isin(row_by_identity)
    )
    if not final_refit:
        pair_mask &= (pairs.formula_fold_a != args.outer_fold) & (pairs.formula_fold_b != args.outer_fold)
    pairs_all = pairs[pairs.identity_a.astype(str).isin(row_by_identity)
                      & pairs.identity_b.astype(str).isin(row_by_identity)].copy()
    pairs = pairs[pair_mask].copy()
    pairs["family"] = pairs.relation_type.astype(str).map(relation_family)
    by_family = {key: value.reset_index(drop=True) for key, value in pairs.groupby("family")}
    if set(by_family) != set(RELATION_CLASS):
        raise RuntimeError(f"training relation families incomplete: {sorted(by_family)}")
    near_by_identity: dict[str, list[str]] = defaultdict(list)
    fallback_by_identity: dict[str, list[str]] = defaultdict(list)
    for row in pairs.itertuples(index=False):
        if row.identity_a == row.identity_b:
            continue
        target = near_by_identity if row.family == "near_isomer" else fallback_by_identity
        target[str(row.identity_a)].append(str(row.identity_b))
        target[str(row.identity_b)].append(str(row.identity_a))
    rank_identities = sorted(
        identity for identity in row_by_identity
        if near_by_identity.get(identity) or fallback_by_identity.get(identity)
    )
    minimum_rank_identities = 2 if args.smoke else 500
    if len(rank_identities) < minimum_rank_identities:
        raise RuntimeError(f"too few rank identities after fold isolation: {len(rank_identities)}")

    with np.load(args.d0_dir / "manifest.npz") as body:
        formula_fold = np.asarray(body["formula_fold"], dtype=np.int8)
        baseline_rank = np.asarray(body["baseline_rank"], dtype=np.int16)
    symmetric_baseline = np.load(args.f0_dir / "symmetric_zero_rank.npy")
    if len(formula_fold) != graph.n_queries or not np.array_equal(symmetric_baseline, baseline_rank):
        raise RuntimeError("baseline or formula-fold mismatch")
    train_correct_queries = np.flatnonzero(
        (baseline_rank == 1) if final_refit else ((formula_fold != args.outer_fold) & (baseline_rank == 1))
    )
    eval_queries = np.arange(graph.n_queries) if final_refit else np.flatnonzero(formula_fold == args.outer_fold)
    if args.max_eval_queries:
        eval_queries = eval_queries[: args.max_eval_queries]
    positive_rows = best_positive_rows(graph)
    all_allowed = np.ones(len(graph.molecule_label), dtype=bool)
    negative_rows = top_negative_rows(graph, all_allowed, 1)[:, 0]
    official_query = store.official(graph.query_row)
    official_positive = store.official(positive_rows)
    official_negative = store.official(negative_rows)
    official_margin = np.sum(official_query * official_positive, axis=1) - np.sum(official_query * official_negative, axis=1)

    adapter = ZeroInitPeakAdapter(store.dimension, args.hidden_dim, args.delta_bound).to(device)
    relation_head = RelationHead(store.dimension, args.hidden_dim).to(device)
    parameters = list(adapter.parameters()) + list(relation_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        adapter.train(); relation_head.train()
        totals = defaultdict(float)
        relation_multiplier = 1.0
        for _ in range(args.steps_per_epoch):
            chosen = rng.choice(rank_identities, size=args.batch_size, replace=len(rank_identities) < args.batch_size)
            chosen = list(map(str, chosen))
            anchor, positive = sample_distinct_same_identity_rows(
                rng, row_by_identity, chosen,
            )
            negative_identity = []
            for identity in chosen:
                values = near_by_identity.get(identity) or fallback_by_identity[identity]
                negative_identity.append(values[int(rng.integers(len(values)))])
            negative = sample_rows(rng, row_by_identity, negative_identity)
            _, za = store.adapt(adapter, anchor, device)
            _, zp = store.adapt(adapter, positive, device)
            _, zn = store.adapt(adapter, negative, device)
            s_pos = torch.sum(za * zp, dim=1)
            s_neg = torch.sum(za * zn, dim=1)
            rank_loss = F.softplus((args.rank_margin + s_neg - s_pos) / args.temperature).mean()

            per_family = max(1, args.batch_size // 4)
            relation_blocks = []
            relation_labels = []
            for family, label in RELATION_CLASS.items():
                table = by_family[family]
                positions = rng.integers(len(table), size=per_family)
                relation_blocks.append(table.iloc[positions])
                relation_labels.extend([label] * per_family)
            relation_table = pd.concat(relation_blocks, ignore_index=True)
            left_identity = relation_table.identity_a.astype(str).tolist()
            right_identity = relation_table.identity_b.astype(str).tolist()
            left_rows = sample_rows(rng, row_by_identity, left_identity)
            right_rows = sample_rows(rng, row_by_identity, right_identity)
            for index, (left_id, right_id) in enumerate(zip(left_identity, right_identity)):
                if left_id == right_id and left_rows[index] == right_rows[index]:
                    values = row_by_identity[left_id]
                    position = int(np.flatnonzero(values == left_rows[index])[0])
                    right_rows[index] = values[(position + 1) % len(values)]
            if epoch < args.relation_warmup_epochs:
                # Warm up the auxiliary decoder on frozen official embeddings;
                # no random relation gradient is allowed to reach the adapter.
                zl = torch.from_numpy(store.official(left_rows)).to(device)
                zr = torch.from_numpy(store.official(right_rows)).to(device)
            else:
                _, zl = store.adapt(adapter, left_rows, device)
                _, zr = store.adapt(adapter, right_rows, device)
            relation_loss = F.cross_entropy(
                relation_head(zl, zr), torch.tensor(relation_labels, device=device)
            )
            safety_query = rng.choice(
                train_correct_queries, size=args.batch_size,
                replace=len(train_correct_queries) < args.batch_size,
            )
            _, zq = store.adapt(adapter, graph.query_row[safety_query], device)
            _, zps = store.adapt(adapter, positive_rows[safety_query], device)
            _, zns = store.adapt(adapter, negative_rows[safety_query], device)
            new_margin = torch.sum(zq * zps, dim=1) - torch.sum(zq * zns, dim=1)
            floor = torch.from_numpy(
                (official_margin[safety_query] - args.safety_slack).astype(np.float32)
            ).to(device)
            safety_loss = F.relu(floor - new_margin).mean()
            official_batch = torch.from_numpy(store.official(np.concatenate((anchor, positive, negative)))).to(device)
            adapted_batch = torch.cat((za, zp, zn), dim=0)
            preserve_loss = torch.clamp(
                1.0 - torch.sum(official_batch * adapted_batch, dim=1), min=0.0,
            ).mean()
            if totals.get("steps", 0.0) == 0.0:
                gradient_rank = adapter_gradient_norm(rank_loss, adapter)
                gradient_relation = (
                    adapter_gradient_norm(relation_loss, adapter)
                    if epoch >= args.relation_warmup_epochs else 0.0
                )
                gradient_safety = adapter_gradient_norm(safety_loss, adapter)
                gradient_preserve = adapter_gradient_norm(preserve_loss, adapter)
                if gradient_relation > 0.0 and args.lambda_relation > 0.0:
                    relation_multiplier = min(
                        1.0,
                        args.relation_gradient_ratio_cap * gradient_rank
                        / (args.lambda_relation * gradient_relation + 1e-12),
                    )
                totals["grad_rank"] = gradient_rank
                totals["grad_relation"] = gradient_relation
                totals["grad_safety"] = gradient_safety
                totals["grad_preserve"] = gradient_preserve
                totals["relation_multiplier"] = relation_multiplier
            loss = (
                rank_loss + args.lambda_relation * relation_multiplier * relation_loss
                + args.lambda_safety * safety_loss
                + args.lambda_preserve * preserve_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for key, value in {
                "loss": loss, "rank": rank_loss, "relation": relation_loss,
                "safety": safety_loss, "preserve": preserve_loss,
                "margin": (s_pos - s_neg).mean(),
            }.items():
                totals[key] += float(value.detach())
            totals["steps"] += 1.0
        history.append({
            key: (
                value if key.startswith("grad_") or key == "relation_multiplier"
                else value / args.steps_per_epoch
            )
            for key, value in totals.items() if key != "steps"
        })
        print(f"[epoch {epoch + 1}/{args.epochs}] {history[-1]}", flush=True)

    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    evaluation = evaluate_full(encoded, store, graph, eval_queries, symmetric_baseline)
    summary = evaluation["summary"]
    if final_refit:
        relation_readout = {"status": "not_applicable_to_final_refit"}
    else:
        strict_relation_table = pairs_all[
            (pairs_all.formula_fold_a == args.outer_fold)
            & (pairs_all.formula_fold_b == args.outer_fold)
        ].copy()
        relation_readout = evaluate_relation_readout(
            adapter, relation_head, store, row_by_identity,
            strict_relation_table, device,
        )
        relation_readout["protocol"] = "both endpoint formulas held out from adapter and relation-head training"
    output_dir = (
        args.output_root / "final" / f"seed_{args.seed}"
        if final_refit else args.output_root / f"fold_{args.outer_fold}" / f"seed_{args.seed}"
    )
    if output_dir.exists():
        raise RuntimeError(f"fail-closed: output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    torch.save({
        "adapter": adapter.state_dict(), "relation_head": relation_head.state_dict(),
        "configuration": vars(args), "embedding_dim": store.dimension,
    }, output_dir / "final.pt")
    np.savez_compressed(
        output_dir / "heldout_predictions.npz", query=evaluation["query"],
        old_rank=evaluation["old_rank"], new_rank=evaluation["new_rank"],
    )
    report = {
        "status": (
            "bioaware_embedding_adapter_final_refit_complete"
            if final_refit else "bioaware_embedding_adapter_fold_complete"
        ),
        "formal": not args.smoke,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "training": {
            "rank_identities": len(rank_identities),
            "relation_pairs": int(len(pairs)),
            "history": history,
            "frozen_hyperparameters": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "steps_per_epoch": args.steps_per_epoch, "lr": args.lr,
                "weight_decay": args.weight_decay, "hidden_dim": args.hidden_dim,
                "delta_bound": args.delta_bound, "rank_margin": args.rank_margin,
                "temperature": args.temperature, "lambda_relation": args.lambda_relation,
                "relation_gradient_ratio_cap": args.relation_gradient_ratio_cap,
                "lambda_safety": args.lambda_safety, "lambda_preserve": args.lambda_preserve,
                "safety_slack": args.safety_slack,
                "relation_warmup_epochs": args.relation_warmup_epochs,
            },
        },
        "heldout": summary,
        "heldout_relation_readout": relation_readout,
        "gates": {
            "recall1_nonnegative": summary["delta_recall1"] >= 0,
            "near_nonnegative": summary.get("delta_near_recall1", 0.0) >= 0,
            "corrected_ge_introduced": summary["corrected"] >= summary["introduced"],
            "preservation_ok": summary["preservation_mean"] >= args.minimum_preservation,
        },
        "contracts": {
            "reaction_neighbour_is_positive": False,
            "relation_head_deployed": False,
            "query_reference_encoder_shared": True,
            "inference_candidate_independent": True,
            "P2b": "forbidden",
            "P3": "not opened",
            "external_transfer": "not opened",
            "final_refit": final_refit,
        },
        "provenance": {
            "preflight_sha256": sha256_file(args.preflight),
            "manifest_report_sha256": sha256_file(args.manifest_dir / "report.json"),
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "oof_decision_sha256": sha256_file(args.oof_decision) if final_refit else None,
        },
        "claim_limit": (
            "Frozen all-development refit after a passing formula-OOF decision; training-graph metrics are diagnostic only."
            if final_refit else
            "One formula-held-out fold; five-fold aggregation is required before any embedding claim."
        ),
    }
    report["gates"]["pass"] = all(report["gates"].values())
    json_dump(output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

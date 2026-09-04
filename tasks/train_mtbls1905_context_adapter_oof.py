"""Formula-held-out development replay for the BioAware context adapter.

This is deliberately a mechanism audit on a consumed 36-query benchmark.  It
tests whether the clean context tensor can be learned without hand-weighting;
it cannot establish external performance or SOTA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware_context_adapter import BiologicalContextAdapter


def formula_fold(value: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % folds


def strict_rank(scores: np.ndarray, positive: int) -> int:
    mask = np.ones(len(scores), dtype=bool)
    mask[positive] = False
    return 1 + int(np.sum(scores[mask] >= scores[positive]))


def bootstrap(transitions: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = transitions.groupby("truth_formula", sort=False).delta.mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        rng.choice(grouped, size=len(grouped), replace=True).mean() for _ in range(repeats)
    ])
    return {
        "mean": float(transitions.delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "formula_clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


class Dataset:
    def __init__(self, path: Path, device: torch.device):
        values = np.load(path)
        self.query_ids = values["query_ids"].astype(str)
        self.formulas = values["truth_formulas"].astype(str)
        self.query = torch.from_numpy(values["query_embeddings"]).to(device)
        self.offsets = values["offsets"].astype(np.int64)
        self.positive = values["positive_indices"].astype(np.int64)
        self.candidate_ids = values["candidate_ids"].astype(str)
        self.candidate = torch.from_numpy(values["candidate_embeddings"]).to(device)
        self.seeds = torch.from_numpy(values["seed_embeddings"]).to(device)
        self.relations = torch.from_numpy(values["relation_types"]).to(device)
        self.features = torch.from_numpy(values["edge_features"]).to(device)
        self.masks = torch.from_numpy(values["edge_masks"]).to(device)

    def candidate_slice(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


def query_loss(
    model: BiologicalContextAdapter, data: Dataset, index: int, temperature: float,
    preserve_weight: float, gate_weight: float, safety_weight: float,
) -> tuple[torch.Tensor, dict]:
    section = data.candidate_slice(index)
    universal = data.candidate[section]
    adapted, delta, gates = model(
        universal, data.seeds[section], data.relations[section], data.features[section], data.masks[section]
    )
    scores = adapted @ data.query[index]
    target = torch.tensor([int(data.positive[index])], device=scores.device)
    rank_loss = F.cross_entropy(scores[None, :] / temperature, target)
    preservation = torch.clamp(1.0 - torch.sum(adapted * universal, dim=-1), min=0).mean()
    sparsity = gates.mean()
    baseline_scores = universal @ data.query[index]
    baseline_correct = strict_rank(baseline_scores.detach().cpu().numpy(), int(data.positive[index])) == 1
    weight = safety_weight if baseline_correct else 1.0
    total = weight * rank_loss + preserve_weight * preservation + gate_weight * sparsity
    return total, {
        "rank": float(rank_loss.detach()), "preserve": float(preservation.detach()),
        "gate": float(sparsity.detach()), "delta_norm": float(delta.norm(dim=1).mean().detach()),
    }


@torch.no_grad()
def score_query(model: BiologicalContextAdapter, data: Dataset, index: int) -> tuple[np.ndarray, float, float]:
    section = data.candidate_slice(index)
    universal = data.candidate[section]
    adapted, _, gates = model(
        universal, data.seeds[section], data.relations[section], data.features[section], data.masks[section]
    )
    scores = (adapted @ data.query[index]).cpu().numpy()
    preservation = float(torch.sum(adapted * universal, dim=-1).mean().cpu())
    return scores, preservation, float(gates.mean().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/validation/mtbls1905_context_adapter_dataset_20260830/dataset.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/mtbls1905_context_adapter_oof_20260830"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260830, 20260831, 20260832])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--relation-dim", type=int, default=8)
    parser.add_argument("--delta-bound", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--preserve-weight", type=float, default=8.0)
    parser.add_argument("--gate-weight", type=float, default=0.0)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = Dataset(args.dataset, device)
    fold_ids = np.asarray([formula_fold(value, args.folds) for value in data.formulas])
    predictions: dict[int, list[np.ndarray]] = {index: [] for index in range(len(data.query_ids))}
    preservations: list[float] = []
    gate_values: list[float] = []
    fold_reports = []
    for fold in range(args.folds):
        train_indices = np.flatnonzero(fold_ids != fold)
        heldout_indices = np.flatnonzero(fold_ids == fold)
        if len(train_indices) == 0 or len(heldout_indices) == 0:
            raise RuntimeError(f"empty train/heldout fold {fold}")
        for random_seed in args.seeds:
            torch.manual_seed(random_seed + fold * 1000)
            np.random.seed(random_seed + fold * 1000)
            model = BiologicalContextAdapter(
                embedding_dim=data.query.shape[1], relation_types=5,
                hidden_dim=args.hidden_dim, relation_dim=args.relation_dim,
                delta_bound=args.delta_bound,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            final_loss = 0.0
            active_train = [
                int(index) for index in train_indices
                if bool(data.masks[data.candidate_slice(int(index))].any())
            ]
            flat_indices = np.concatenate([
                np.arange(data.offsets[index], data.offsets[index + 1], dtype=np.int64)
                for index in active_train
            ])
            flat_tensor = torch.from_numpy(flat_indices).to(device)
            query_for_candidate = torch.cat([
                torch.full(
                    (int(data.offsets[index + 1] - data.offsets[index]),), index,
                    dtype=torch.long, device=device,
                )
                for index in active_train
            ])
            for _ in range(args.epochs):
                optimizer.zero_grad(set_to_none=True)
                universal = data.candidate[flat_tensor]
                adapted, _, gates = model(
                    universal, data.seeds[flat_tensor], data.relations[flat_tensor],
                    data.features[flat_tensor], data.masks[flat_tensor],
                )
                scores = torch.sum(adapted * data.query[query_for_candidate], dim=1)
                losses = []
                cursor = 0
                for index in active_train:
                    count = int(data.offsets[index + 1] - data.offsets[index])
                    local_scores = scores[cursor:cursor + count]
                    target = torch.tensor([int(data.positive[index])], device=device)
                    local_loss = F.cross_entropy(local_scores[None, :] / args.temperature, target)
                    baseline = (universal[cursor:cursor + count] @ data.query[index]).detach().cpu().numpy()
                    if strict_rank(baseline, int(data.positive[index])) == 1:
                        local_loss = args.safety_weight * local_loss
                    losses.append(local_loss)
                    cursor += count
                preservation = torch.clamp(1.0 - torch.sum(adapted * universal, dim=-1), min=0).mean()
                loss = torch.stack(losses).mean() + args.preserve_weight * preservation + args.gate_weight * gates.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                final_loss = float(loss.detach())
            model.eval()
            for index in heldout_indices:
                scores, preservation, gate = score_query(model, data, int(index))
                predictions[int(index)].append(scores)
                preservations.append(preservation)
                gate_values.append(gate)
            fold_reports.append({
                "fold": fold, "seed": random_seed, "train_queries": int(len(train_indices)),
                "heldout_queries": int(len(heldout_indices)), "active_training_queries": len(active_train),
                "gradient_steps": int(args.epochs),
                "final_training_loss": final_loss,
            })

    transition_rows = []
    for index in range(len(data.query_ids)):
        if len(predictions[index]) != len(args.seeds):
            raise RuntimeError(f"query {index}: incomplete seed ensemble")
        section = data.candidate_slice(index)
        baseline_scores = (data.candidate[section] @ data.query[index]).cpu().numpy()
        adapted_scores = np.mean(np.stack(predictions[index], axis=0), axis=0)
        positive = int(data.positive[index])
        baseline_rank = strict_rank(baseline_scores, positive)
        adapted_rank = strict_rank(adapted_scores, positive)
        baseline_correct = baseline_rank == 1
        adapted_correct = adapted_rank == 1
        transition_rows.append({
            "query_id": data.query_ids[index], "truth_formula": data.formulas[index],
            "baseline_rank": baseline_rank, "adapted_rank": adapted_rank,
            "baseline_correct": baseline_correct, "adapted_correct": adapted_correct,
            "corrected": (not baseline_correct) and adapted_correct,
            "introduced": baseline_correct and (not adapted_correct),
            "delta": float(adapted_correct) - float(baseline_correct),
            "context_candidates": int(data.masks[section].any(dim=1).sum().cpu()),
        })
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(args.output_dir / "oof_transitions.csv.gz", index=False)
    report = {
        "status": "mtbls1905_context_adapter_oof_complete",
        "formal": False,
        "protocol": f"formula-held-out {args.folds}-fold, fixed hyperparameters, {len(args.seeds)}-seed score ensemble, consumed development benchmark",
        "queries": int(len(transitions)),
        "formulas": int(transitions.truth_formula.nunique()),
        "baseline_recall1": float(transitions.baseline_correct.mean()),
        "adapted_recall1": float(transitions.adapted_correct.mean()),
        "delta_recall1": float(transitions.delta.mean()),
        "corrected": int(transitions.corrected.sum()),
        "introduced": int(transitions.introduced.sum()),
        "queries_with_context_candidate": int((transitions.context_candidates > 0).sum()),
        "mean_preservation": float(np.mean(preservations)),
        "mean_gate": float(np.mean(gate_values)),
        "formula_cluster_bootstrap": bootstrap(transitions, args.bootstrap_resamples, args.seeds[0]),
        "fold_runs": fold_reports,
        "configuration": {
            "folds": args.folds, "seeds": args.seeds, "epochs": args.epochs,
            "hidden_dim": args.hidden_dim, "relation_dim": args.relation_dim,
            "delta_bound": args.delta_bound, "learning_rate": args.learning_rate,
            "temperature": args.temperature, "preserve_weight": args.preserve_weight,
            "gate_weight": args.gate_weight, "safety_weight": args.safety_weight,
        },
        "contracts": {
            "candidate_specific_context": True,
            "query_embedding_unmodified": True,
            "no_context_exact_fallback": True,
            "formula_heldout": True,
            "consumed_development_only": True,
        },
        "claim_limit": "Small consumed mechanism audit; it cannot establish external performance or SOTA.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

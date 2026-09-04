"""Formula-held-out OOF training of candidate-specific BioAware context embeddings."""
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


def formula_bootstrap(transitions: pd.DataFrame, repeats: int, seed: int) -> dict:
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
        self.identities = values["query_identities"].astype(str)
        self.rotation_folds = values["rotation_folds"].astype(np.int16)
        self.query = torch.from_numpy(values["query_embeddings"]).to(device)
        self.offsets = values["offsets"].astype(np.int64)
        self.positive = values["positive_indices"].astype(np.int64)
        self.candidate_ids = values["candidate_ids"].astype(str)
        self.candidate = torch.from_numpy(values["candidate_embeddings"]).to(device)
        self.seed_prototypes = torch.from_numpy(values["seed_prototypes"]).to(device)
        self.seed_indices = torch.from_numpy(values["seed_indices"]).to(device)
        self.relations = torch.from_numpy(values["relation_types"]).to(device)
        self.features = torch.from_numpy(values["edge_features"]).to(device)
        self.masks = torch.from_numpy(values["edge_masks"]).to(device)

    def section(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))

    def seeds(self, flat: torch.Tensor) -> torch.Tensor:
        return self.seed_prototypes[self.seed_indices[flat]]


def batched_loss(
    model: BiologicalContextAdapter, data: Dataset, indices: np.ndarray, args: argparse.Namespace,
) -> torch.Tensor:
    flat_numpy = np.concatenate([
        np.arange(data.offsets[index], data.offsets[index + 1], dtype=np.int64)
        for index in indices
    ])
    flat = torch.from_numpy(flat_numpy).to(data.query.device)
    query_for_candidate = torch.cat([
        torch.full(
            (int(data.offsets[index + 1] - data.offsets[index]),), int(index),
            dtype=torch.long, device=data.query.device,
        ) for index in indices
    ])
    universal = data.candidate[flat]
    adapted, _, gates = model(
        universal, data.seeds(flat), data.relations[flat], data.features[flat], data.masks[flat],
    )
    scores = torch.sum(adapted * data.query[query_for_candidate], dim=1)
    losses = []
    cursor = 0
    for index in indices:
        count = int(data.offsets[index + 1] - data.offsets[index])
        local_scores = scores[cursor:cursor + count]
        target = torch.tensor([int(data.positive[index])], device=scores.device)
        loss = F.cross_entropy(local_scores[None, :] / args.temperature, target)
        baseline = (universal[cursor:cursor + count] @ data.query[index]).detach().cpu().numpy()
        if strict_rank(baseline, int(data.positive[index])) == 1:
            loss = args.safety_weight * loss
        losses.append(loss)
        cursor += count
    preservation = torch.clamp(1.0 - torch.sum(adapted * universal, dim=-1), min=0).mean()
    return torch.stack(losses).mean() + args.preserve_weight * preservation + args.gate_weight * gates.mean()


@torch.no_grad()
def score_instance(model: BiologicalContextAdapter, data: Dataset, index: int) -> tuple[np.ndarray, float, float]:
    section = data.section(index)
    flat = torch.arange(section.start, section.stop, device=data.query.device)
    universal = data.candidate[section]
    adapted, _, gates = model(
        universal, data.seeds(flat), data.relations[section], data.features[section], data.masks[section],
    )
    return (
        (adapted @ data.query[index]).cpu().numpy(),
        float(torch.sum(adapted * universal, dim=-1).mean().cpu()),
        float(gates.mean().cpu()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/validation/bioaware_metdna3_context_adapter_dataset_v1/dataset.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_context_adapter_oof_v1"))
    parser.add_argument("--formula-folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260830, 20260831, 20260832])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--instance-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--relation-dim", type=int, default=16)
    parser.add_argument("--delta-bound", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--preserve-weight", type=float, default=8.0)
    parser.add_argument("--gate-weight", type=float, default=0.0)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = Dataset(args.dataset, device)
    formula_folds = np.asarray([formula_fold(value, args.formula_folds) for value in data.formulas])
    predictions: dict[int, list[np.ndarray]] = {i: [] for i in range(len(data.query_ids))}
    preservation_values: list[float] = []
    gate_values: list[float] = []
    fold_reports = []
    for fold in range(args.formula_folds):
        train = np.flatnonzero(formula_folds != fold)
        heldout = np.flatnonzero(formula_folds == fold)
        active_train = np.asarray([
            index for index in train if bool(data.masks[data.section(int(index))].any())
        ], dtype=np.int64)
        if len(active_train) == 0 or len(heldout) == 0:
            raise RuntimeError(f"fold {fold}: empty active train or heldout")
        for random_seed in args.seeds:
            torch.manual_seed(random_seed + fold * 1000)
            np.random.seed(random_seed + fold * 1000)
            model = BiologicalContextAdapter(
                embedding_dim=data.query.shape[1], relation_types=5,
                hidden_dim=args.hidden_dim, relation_dim=args.relation_dim,
                delta_bound=args.delta_bound,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            rng = np.random.default_rng(random_seed + fold * 1000)
            final_loss = 0.0
            steps = 0
            for _ in range(args.epochs):
                order = rng.permutation(active_train)
                for start in range(0, len(order), args.instance_batch_size):
                    batch = order[start:start + args.instance_batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = batched_loss(model, data, batch, args)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    final_loss = float(loss.detach())
                    steps += 1
            model.eval()
            for index in heldout:
                scores, preservation, gate = score_instance(model, data, int(index))
                predictions[int(index)].append(scores)
                preservation_values.append(preservation)
                gate_values.append(gate)
            fold_reports.append({
                "formula_fold": fold, "seed": random_seed,
                "train_instances": int(len(train)), "active_train_instances": int(len(active_train)),
                "heldout_instances": int(len(heldout)), "gradient_steps": steps,
                "final_training_loss": final_loss,
            })

    # Average across preregistered model seeds only.  A seed rotation represents
    # a distinct observable biological context, not another model prediction.
    # Averaging scores across rotations before ranking would create a context
    # ensemble that cannot exist for one deployed sample and can cancel real
    # context-specific effects.
    per_instance_scores = {}
    for index, values in predictions.items():
        if len(values) != len(args.seeds):
            raise RuntimeError(f"instance {index}: incomplete seed ensemble")
        per_instance_scores[index] = np.mean(np.stack(values, axis=0), axis=0)
    by_query: dict[str, list[int]] = {}
    for index, query_id in enumerate(data.query_ids):
        by_query.setdefault(str(query_id), []).append(index)
    instance_transitions = []
    for query_id, indices in by_query.items():
        first = indices[0]
        candidate_ids = data.candidate_ids[data.section(first)]
        for index in indices[1:]:
            if not np.array_equal(candidate_ids, data.candidate_ids[data.section(index)]):
                raise RuntimeError(f"{query_id}: candidate order changes across rotations")
        for index in indices:
            section = data.section(index)
            baseline = (data.candidate[section] @ data.query[index]).cpu().numpy()
            adapted = per_instance_scores[index]
            positive = int(data.positive[index])
            baseline_rank = strict_rank(baseline, positive)
            adapted_rank = strict_rank(adapted, positive)
            baseline_correct = baseline_rank == 1
            adapted_correct = adapted_rank == 1
            negative = np.ones(len(baseline), dtype=bool)
            negative[positive] = False
            baseline_margin = float(baseline[positive] - np.max(baseline[negative]))
            adapted_margin = float(adapted[positive] - np.max(adapted[negative]))
            has_context = bool(data.masks[section].any())
            instance_transitions.append({
                "query_id": query_id, "rotation_fold": int(data.rotation_folds[index]),
                "truth_formula": str(data.formulas[index]),
                "truth_identity": str(data.identities[index]),
                "baseline_rank": baseline_rank, "adapted_rank": adapted_rank,
                "baseline_correct": baseline_correct, "adapted_correct": adapted_correct,
                "corrected": (not baseline_correct) and adapted_correct,
                "introduced": baseline_correct and (not adapted_correct),
                "delta": float(adapted_correct) - float(baseline_correct),
                "has_context": has_context,
                "context_candidates": int(data.masks[section].any(dim=1).sum().cpu()),
                "baseline_margin": baseline_margin,
                "adapted_margin": adapted_margin,
                "margin_delta": adapted_margin - baseline_margin,
                "max_abs_score_delta": float(np.max(np.abs(adapted - baseline))),
            })
    transitions = pd.DataFrame(instance_transitions)
    transitions.to_csv(args.output_dir / "rotation_oof_transitions.csv.gz", index=False)
    bootstrap = formula_bootstrap(transitions, args.bootstrap_resamples, args.seeds[0])
    active = transitions.loc[transitions.has_context].copy()
    active_bootstrap = formula_bootstrap(active, args.bootstrap_resamples, args.seeds[0] + 1)

    # Retain the historical cross-rotation score average as a sensitivity
    # analysis only.  It is not the primary deployment protocol.
    averaged_rows = []
    for query_id, indices in by_query.items():
        first = indices[0]
        section = data.section(first)
        baseline = (data.candidate[section] @ data.query[first]).cpu().numpy()
        adapted = np.mean(np.stack([per_instance_scores[index] for index in indices], axis=0), axis=0)
        positive = int(data.positive[first])
        baseline_correct = strict_rank(baseline, positive) == 1
        adapted_correct = strict_rank(adapted, positive) == 1
        averaged_rows.append({
            "query_id": query_id, "truth_formula": str(data.formulas[first]),
            "truth_identity": str(data.identities[first]),
            "baseline_correct": baseline_correct, "adapted_correct": adapted_correct,
            "corrected": (not baseline_correct) and adapted_correct,
            "introduced": baseline_correct and (not adapted_correct),
            "delta": float(adapted_correct) - float(baseline_correct),
        })
    averaged = pd.DataFrame(averaged_rows)
    averaged.to_csv(args.output_dir / "cross_rotation_average_sensitivity.csv.gz", index=False)
    report = {
        "status": "bioaware_metdna3_context_adapter_oof_complete",
        "formal": True,
        "protocol": "consumed HILIC development; formula-held-out OOF; identity-isolated context rotations evaluated separately; fixed 3-seed model ensemble",
        "rotation_instances": int(len(transitions)),
        "queries": int(transitions.query_id.nunique()),
        "identities": int(transitions.truth_identity.nunique()),
        "formulas": int(transitions.truth_formula.nunique()),
        "baseline_recall1": float(transitions.baseline_correct.mean()),
        "context_recall1": float(transitions.adapted_correct.mean()),
        "delta_recall1": float(transitions.delta.mean()),
        "corrected": int(transitions.corrected.sum()),
        "introduced": int(transitions.introduced.sum()),
        "instances_with_context": int(transitions.has_context.sum()),
        "queries_with_context": int(transitions.loc[transitions.has_context, "query_id"].nunique()),
        "mean_preservation": float(np.mean(preservation_values)),
        "mean_gate": float(np.mean(gate_values)),
        "mean_abs_score_delta_with_context": float(active.max_abs_score_delta.mean()),
        "mean_margin_delta_with_context": float(active.margin_delta.mean()),
        "formula_cluster_bootstrap": bootstrap,
        "context_active_subgroup": {
            "rotation_instances": int(len(active)),
            "baseline_recall1": float(active.baseline_correct.mean()),
            "context_recall1": float(active.adapted_correct.mean()),
            "delta_recall1": float(active.delta.mean()),
            "corrected": int(active.corrected.sum()),
            "introduced": int(active.introduced.sum()),
            "formula_cluster_bootstrap": active_bootstrap,
        },
        "cross_rotation_average_sensitivity": {
            "queries": int(len(averaged)),
            "delta_recall1": float(averaged.delta.mean()),
            "corrected": int(averaged.corrected.sum()),
            "introduced": int(averaged.introduced.sum()),
        },
        "configuration": vars(args) | {"dataset": str(args.dataset), "output_dir": str(args.output_dir)},
        "fold_runs": fold_reports,
        "gates": {
            "recall_formula_ci_positive": bootstrap["ci_low"] > 0,
            "corrected_gt_introduced": int(transitions.corrected.sum()) > int(transitions.introduced.sum()),
            "preservation_ge_0_995": float(np.mean(preservation_values)) >= 0.995,
            "pass_to_locked_rp_internal_validation": (
                bootstrap["ci_low"] > 0
                and int(transitions.corrected.sum()) > int(transitions.introduced.sum())
                and float(np.mean(preservation_values)) >= 0.995
            ),
        },
        "contracts": {
            "P2b": "forbidden", "phenotype_blind": True,
            "heldout_truth_absent_from_context_seeds": True,
            "candidate_specific_context": True,
            "no_context_exact_fallback": True,
            "context_rotations_are_not_score_averaged_in_primary_endpoint": True,
            "internal_validation_or_external_test_opened": False,
        },
        "claim_limit": "Consumed HILIC development result; even a pass only unlocks frozen RPLC internal validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

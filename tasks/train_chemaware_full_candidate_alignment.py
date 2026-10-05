"""Large-scale candidate-alignment adapter for the shared DreaMS embedding.

This is a deployment-compatible adaptation of MSAlign: frozen DreaMS and
ChemBERTa features are aligned with candidate-based InfoNCE, while a parallel
spectrum-to-spectrum listwise loss keeps query and reference retrieval in the
same 1024-dimensional space.  Molecule structures and candidates are strictly
training-time inputs.  The deployable artifact is only one shared spectrum
adapter applied identically to query and reference DreaMS embeddings.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from chemaware_boundary_pmt_core import (  # noqa: E402
    active_margin_transfer_loss,
    inherited_clean_margin_target,
)
from noise_final_core import sha256_file, strict_rank  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--molecule-dir", type=Path, default=ROOT / "data/validation/chemaware_chemberta_teacher_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_full_candidate_alignment_v1")
    parser.add_argument("--arm", choices=("correct", "candidate_swapped", "spectrum_only"), default="correct")
    parser.add_argument("--bpmt-manifest-dir", type=Path)
    parser.add_argument(
        "--bpmt-arm",
        choices=("none", "clean_duplicate", "matched_formula_deranged", "alpha025", "alpha050"),
        default="none",
    )
    parser.add_argument("--lambda-bpmt", type=float, default=1.0)
    parser.add_argument("--bpmt-batch-actions", type=int, default=8)
    parser.add_argument("--bpmt-temperature", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="If positive, repeat identity-balanced epochs until this optimizer-step budget is reached.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--molecule-only-warmup-steps", type=int, default=0,
        help="First align the molecule projector to frozen official spectrum geometry.",
    )
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--batch-queries", type=int, default=128)
    parser.add_argument("--references-per-molecule", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--molecule-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-spectrum", type=float, default=1.0)
    parser.add_argument("--lambda-molecule", type=float, default=1.0)
    parser.add_argument("--lambda-inbatch-molecule", type=float, default=1.0)
    parser.add_argument("--lambda-inbatch-spectrum", type=float, default=0.25)
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--lambda-preserve", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-identities", type=int, default=0)
    parser.add_argument(
        "--error-identity-fraction", type=float, default=0.0,
        help="Fraction of each identity-balanced epoch drawn from train identities with official errors.",
    )
    parser.add_argument(
        "--clean-safety-selection", choices=("random", "boundary_mixture"), default="random",
        help=("How to draw baseline-correct safety identities in the error curriculum. "
              "boundary_mixture uses a deterministic half low-margin, half random mixture."),
    )
    parser.add_argument(
        "--training-mass", choices=("identity", "formula_identity"), default="identity",
        help="Equalize optimizer mass by identity alone or by formula then identity.",
    )
    parser.add_argument(
        "--train-diagnostic-identities", type=int, default=256,
        help="Training identities evaluated only as a capacity diagnostic; never used for selection.",
    )
    parser.add_argument("--max-eval-identities", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete-cache", action="store_true",
        help="Non-formal smoke on the atomically flushed prefix of a running token cache.",
    )
    parser.add_argument(
        "--incomplete-cache-rows", type=int, default=0,
        help="Freeze an incomplete-cache smoke to this many completed rows for matched arms.",
    )
    return parser.parse_args()


class ResidualSpectrumProjector(nn.Module):
    """Identity-initialized shared adapter, kept local to avoid optional imports."""

    def __init__(self, dimension: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        if dimension <= 0 or hidden_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid global embedding adapter configuration")
        self.norm = nn.LayerNorm(dimension)
        self.fc1 = nn.Linear(dimension, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dimension)
        self.dropout = float(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise RuntimeError("global embedding adapter expects a matrix")
        delta = self.fc2(
            F.dropout(F.gelu(self.fc1(self.norm(value))), self.dropout, self.training)
        )
        return F.normalize(value + delta, dim=-1)


class MoleculeProjector(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden), nn.Linear(hidden, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(value), dim=-1)


def identity_balanced_queries(query: np.ndarray, identities: np.ndarray,
                              rng: np.random.Generator, limit: int = 0) -> np.ndarray:
    by_identity: dict[str, list[int]] = {}
    for value in np.asarray(query, dtype=np.int64):
        by_identity.setdefault(str(identities[value]), []).append(int(value))
    keys = np.asarray(sorted(by_identity), dtype=object)
    rng.shuffle(keys)
    if limit:
        keys = keys[:limit]
    selected = [by_identity[str(key)][int(rng.integers(len(by_identity[str(key)])))] for key in keys]
    return np.asarray(selected, dtype=np.int64)


def error_curriculum_queries(query: np.ndarray, identities: np.ndarray,
                             official_error: np.ndarray, official_margin: np.ndarray,
                             rng: np.random.Generator, limit: int, error_fraction: float,
                             clean_safety_selection: str = "random") -> np.ndarray:
    """Identity-equal sampling with train-only official-error enrichment."""
    if not 0.0 <= error_fraction <= 1.0:
        raise ValueError("error-identity-fraction must be between zero and one")
    by_identity: dict[str, list[int]] = {}
    for value in np.asarray(query, dtype=np.int64):
        by_identity.setdefault(str(identities[value]), []).append(int(value))
    error_keys = np.asarray([
        key for key, values in by_identity.items()
        if np.any(official_error[np.asarray(values, dtype=np.int64)])
    ], dtype=object)
    clean_keys = np.asarray([
        key for key, values in by_identity.items()
        if not np.any(official_error[np.asarray(values, dtype=np.int64)])
    ], dtype=object)
    rng.shuffle(error_keys)
    target = min(limit or len(by_identity), len(by_identity))
    n_error = min(len(error_keys), int(round(target * error_fraction)))
    n_clean = min(len(clean_keys), target - n_error)
    if n_error + n_clean < target:
        n_error = min(len(error_keys), target - n_clean)
    if clean_safety_selection == "random":
        rng.shuffle(clean_keys)
        selected_clean = clean_keys[:n_clean]
    elif clean_safety_selection == "boundary_mixture":
        # A pure hard-boundary curriculum over-concentrates on ambiguous chemistry.
        # Keep half of the clean identities representative and spend the other half
        # on the baseline-correct identities closest to a top-1 decision boundary.
        boundary_count = n_clean // 2
        clean_margin = {
            str(key): float(np.nanmin(official_margin[np.asarray(by_identity[str(key)], dtype=np.int64)]))
            for key in clean_keys
        }
        boundary = np.asarray(sorted(
            map(str, clean_keys), key=lambda key: (clean_margin[key], key),
        )[:boundary_count], dtype=object)
        boundary_set = set(map(str, boundary))
        remaining = np.asarray([
            key for key in clean_keys if str(key) not in boundary_set
        ], dtype=object)
        rng.shuffle(remaining)
        selected_clean = np.concatenate((boundary, remaining[:n_clean - boundary_count]))
    else:
        raise ValueError(f"unsupported clean safety selection: {clean_safety_selection}")
    selected_keys = np.concatenate((error_keys[:n_error], selected_clean))
    rng.shuffle(selected_keys)
    selected = []
    error_key_set = set(map(str, error_keys[:n_error]))
    for key in selected_keys:
        values = np.asarray(by_identity[str(key)], dtype=np.int64)
        if str(key) in error_key_set:
            values = values[official_error[values]]
        selected.append(int(values[int(rng.integers(len(values)))]))
    return np.asarray(selected, dtype=np.int64)


def official_outcomes(body: dict[str, np.ndarray], queries: np.ndarray,
                      official: np.ndarray, row_position: dict[int, int],
                      molecule_allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    error = np.zeros(len(body["query_row"]), dtype=bool)
    margin = np.full(len(body["query_row"]), np.nan, dtype=np.float32)
    for query in np.asarray(queries, dtype=np.int64):
        qpos = row_position[int(body["query_row"][query])]
        left, right = map(int, body["query_ptr"][query:query + 2])
        scores = []
        for molecule in range(left, right):
            if molecule_allowed is not None and not molecule_allowed[molecule]:
                continue
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            positions = [row_position[int(row)]
                         for row in body["pair_candidate_row"][rleft:rright]
                         if int(row) in row_position]
            if positions:
                scores.append(float(np.max(official[positions] @ official[qpos])))
        if len(scores) < 2:
            raise RuntimeError("training outcome query has fewer than two cached candidates")
        rank = strict_rank(np.asarray(scores))
        error[int(query)] = rank != 1
        margin[int(query)] = scores[0] - max(scores[1:])
    return error, margin


def formula_identity_epoch_weights(
    queries: np.ndarray, formulas: np.ndarray, mode: str,
) -> np.ndarray:
    """Return mean-one query weights with equal total mass per selected formula."""
    if mode == "identity":
        return np.ones(len(queries), dtype=np.float32)
    if mode != "formula_identity":
        raise ValueError(f"unsupported training mass: {mode}")
    selected_formula = np.asarray(formulas)[np.asarray(queries, dtype=np.int64)].astype(str)
    _, inverse, counts = np.unique(selected_formula, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def formula_bootstrap(delta: np.ndarray, formula: np.ndarray, seed: int, draws: int) -> dict:
    unique = np.unique(formula)
    macro = np.asarray([np.mean(delta[formula == value]) for value in unique])
    rng = np.random.default_rng(seed)
    estimate = np.asarray([
        np.mean(macro[rng.integers(0, len(macro), len(macro))]) for _ in range(draws)
    ])
    return {
        "formula_macro_delta": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimate, 0.025)), float(np.quantile(estimate, 0.975)),
        ],
        "formula_clusters": int(len(unique)), "draws": int(draws),
    }


def sample_training_batch(body: dict[str, np.ndarray], queries: np.ndarray,
                          row_position: dict[int, int], molecule_position: np.ndarray | None,
                          references_per_molecule: int, rng: np.random.Generator,
                          molecule_allowed: np.ndarray | None = None) -> dict:
    query_cache = np.asarray([row_position[int(body["query_row"][q])] for q in queries], dtype=np.int64)
    reference_cache: list[int] = []
    molecule_teacher: list[int] = []
    candidate_ptr = [0]
    reference_ptr = [0]
    for query in queries:
        left, right = map(int, body["query_ptr"][query:query + 2])
        for molecule in range(left, right):
            if molecule_allowed is not None and not molecule_allowed[molecule]:
                continue
            molecule_teacher.append(
                int(molecule_position[molecule]) if molecule_position is not None else 0
            )
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            available = np.asarray([
                row for row in body["pair_candidate_row"][rleft:rright]
                if int(row) in row_position
            ], dtype=np.int64)
            if not len(available):
                raise RuntimeError("eligible query has a candidate molecule without a cached reference")
            take = min(references_per_molecule, len(available))
            chosen = rng.choice(available, size=take, replace=False)
            reference_cache.extend(row_position[int(row)] for row in chosen)
            reference_ptr.append(len(reference_cache))
        candidate_ptr.append(len(molecule_teacher))
        if candidate_ptr[-1] == candidate_ptr[-2]:
            raise RuntimeError("formula-disjoint filtering removed every candidate")
        if candidate_ptr[-1] - candidate_ptr[-2] < 2:
            raise RuntimeError("formula-disjoint training query has no legal negative candidate")
        first_molecule = next(
            molecule for molecule in range(left, right)
            if molecule_allowed is None or molecule_allowed[molecule]
        )
        if int(body["molecule_label"][first_molecule]) != 1:
            raise RuntimeError("formula-disjoint filtering removed the true candidate")
    reference_cache_array = np.asarray(reference_cache, dtype=np.int64)
    unique_reference, reference_edge = np.unique(reference_cache_array, return_inverse=True)
    return {
        "query_cache": query_cache,
        "reference_cache": unique_reference,
        "reference_edge": reference_edge.astype(np.int64),
        "molecule_teacher": np.asarray(molecule_teacher, dtype=np.int64),
        "candidate_ptr": np.asarray(candidate_ptr, dtype=np.int64),
        "reference_ptr": np.asarray(reference_ptr, dtype=np.int64),
    }


def listwise_losses(query_z: torch.Tensor, reference_z: torch.Tensor,
                    molecule_z: torch.Tensor | None, candidate_ptr: np.ndarray,
                    reference_ptr: np.ndarray, reference_edge: np.ndarray,
                    temperature: float, official_query: torch.Tensor | None = None,
                    official_reference: torch.Tensor | None = None,
                    margin_floor_slack: float = 0.005,
                    query_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    spectrum_losses, molecule_losses, margin_floors = [], [], []
    reference_edge_tensor = torch.as_tensor(reference_edge, device=reference_z.device)
    for index, (left, right) in enumerate(zip(candidate_ptr[:-1], candidate_ptr[1:])):
        spectrum_score = []
        for molecule in range(int(left), int(right)):
            rleft, rright = map(int, reference_ptr[molecule:molecule + 2])
            spectrum_score.append(
                (reference_z[reference_edge_tensor[rleft:rright]] @ query_z[index]).max()
            )
        spectrum_score = torch.stack(spectrum_score)
        spectrum_losses.append(-F.log_softmax(spectrum_score / temperature, dim=0)[0])
        if official_query is not None and official_reference is not None:
            baseline_scores = []
            for molecule in range(int(left), int(right)):
                rleft, rright = map(int, reference_ptr[molecule:molecule + 2])
                baseline_scores.append(
                    (official_reference[reference_edge_tensor[rleft:rright]]
                     @ official_query[index]).max()
                )
            baseline_scores = torch.stack(baseline_scores)
            current_margin = spectrum_score[0] - torch.max(spectrum_score[1:])
            baseline_margin = baseline_scores[0] - torch.max(baseline_scores[1:])
            margin_floors.append(
                F.relu(baseline_margin.detach() - margin_floor_slack - current_margin)
            )
        if molecule_z is not None:
            molecule_score = molecule_z[int(left):int(right)] @ query_z[index]
            molecule_losses.append(-F.log_softmax(molecule_score / temperature, dim=0)[0])
    # Candidate lists provide mass/formula-matched hard negatives.  Their first
    # molecule is the labelled true identity.  Other queries' true molecules
    # and references add cheap batch negatives and enlarge the contrastive set.
    positive_reference = torch.stack([
        reference_z[int(reference_edge[int(reference_ptr[int(left)])])]
        for left in candidate_ptr[:-1]
    ])
    labels = torch.arange(len(query_z), device=query_z.device)
    weight = (
        torch.ones(len(query_z), device=query_z.device, dtype=query_z.dtype)
        if query_weight is None else query_weight.to(device=query_z.device, dtype=query_z.dtype)
    )
    if weight.shape != (len(query_z),) or not torch.all(torch.isfinite(weight)) or torch.any(weight <= 0):
        raise ValueError("query weights must be one finite positive value per query")
    # Caller-provided weights are normalized once over the complete epoch.
    # Do not renormalize per batch: that would destroy exact formula mass.
    if molecule_z is None:
        molecule_loss = query_z.sum() * 0.0
        inbatch_molecule = query_z.sum() * 0.0
    else:
        positive_molecule = torch.stack([
            molecule_z[int(left)] for left in candidate_ptr[:-1]
        ])
        molecule_loss = torch.mean(torch.stack(molecule_losses) * weight)
        inbatch_molecule = torch.mean(F.cross_entropy(
            query_z @ positive_molecule.T / temperature, labels, reduction="none",
        ) * weight)
    inbatch_spectrum = torch.mean(F.cross_entropy(
        query_z @ positive_reference.T / temperature, labels, reduction="none",
    ) * weight)
    margin_floor = (
        torch.mean(torch.stack(margin_floors) * weight)
        if margin_floors else query_z.sum() * 0.0
    )
    return (torch.mean(torch.stack(spectrum_losses) * weight), molecule_loss,
            inbatch_spectrum, inbatch_molecule, margin_floor)


def learning_rate_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup followed by cosine decay; ``step`` is one-based."""
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def project_numpy(model: nn.Module, values: np.ndarray, device: torch.device,
                  batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    for left in range(0, len(values), batch_size):
        # memmap slices are read-only; make ownership explicit before Torch.
        tensor = torch.from_numpy(
            np.array(values[left:left + batch_size], dtype=np.float32, copy=True)
        ).to(device)
        output.append(model(tensor).cpu().numpy())
    return np.concatenate(output)


def evaluate(body: dict[str, np.ndarray], queries: np.ndarray, official: np.ndarray,
             adapted: np.ndarray, row_position: dict[int, int]) -> dict:
    old_rank, new_rank, old_margin, new_margin, candidate_count = [], [], [], [], []
    for query in queries:
        qpos = row_position[int(body["query_row"][query])]
        left, right = map(int, body["query_ptr"][query:query + 2])
        old_scores, new_scores = [], []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            pos = [row_position[int(row)] for row in body["pair_candidate_row"][rleft:rright]
                   if int(row) in row_position]
            if not pos:
                raise RuntimeError("evaluation query has a candidate molecule without a cached reference")
            old_scores.append(float(np.max(official[pos] @ official[qpos])))
            new_scores.append(float(np.max(adapted[pos] @ adapted[qpos])))
        old_rank.append(strict_rank(np.asarray(old_scores)))
        new_rank.append(strict_rank(np.asarray(new_scores)))
        old_margin.append(old_scores[0] - max(old_scores[1:]))
        new_margin.append(new_scores[0] - max(new_scores[1:]))
        candidate_count.append(len(old_scores))
    old_rank = np.asarray(old_rank); new_rank = np.asarray(new_rank)
    old_margin = np.asarray(old_margin); new_margin = np.asarray(new_margin)
    candidate_count = np.asarray(candidate_count)
    old_ok, new_ok = old_rank == 1, new_rank == 1
    denominator = np.maximum(candidate_count - 1, 1)
    old_auc_each = (candidate_count - old_rank) / denominator
    new_auc_each = (candidate_count - new_rank) / denominator
    summary = {
        "queries": int(len(queries)),
        "baseline_recall1": float(np.mean(old_ok)), "recall1": float(np.mean(new_ok)),
        "delta_recall1": float(np.mean(new_ok) - np.mean(old_ok)),
        "baseline_mrr": float(np.mean(1 / old_rank)), "mrr": float(np.mean(1 / new_rank)),
        "delta_mrr": float(np.mean(1 / new_rank) - np.mean(1 / old_rank)),
        "baseline_macro_auc": float(np.mean(old_auc_each)),
        "macro_auc": float(np.mean(new_auc_each)),
        "delta_macro_auc": float(np.mean(new_auc_each - old_auc_each)),
        "baseline_micro_auc": float(np.sum(candidate_count - old_rank) / np.sum(denominator)),
        "micro_auc": float(np.sum(candidate_count - new_rank) / np.sum(denominator)),
        "delta_micro_auc": float(np.sum(old_rank - new_rank) / np.sum(denominator)),
        "corrected": int(np.sum(~old_ok & new_ok)), "introduced": int(np.sum(old_ok & ~new_ok)),
        "delta_mean_margin": float(np.mean(new_margin - old_margin)),
    }
    for k in (5, 10, 20, 50):
        summary[f"baseline_recall{k}"] = float(np.mean(old_rank <= k))
        summary[f"recall{k}"] = float(np.mean(new_rank <= k))
        summary[f"delta_recall{k}"] = float(
            np.mean(new_rank <= k) - np.mean(old_rank <= k)
        )
    return {
        "old_rank": old_rank, "new_rank": new_rank,
        "candidate_count": candidate_count,
        "summary": summary,
    }


def bpmt_loss_for_positions(
    model: nn.Module,
    official: np.ndarray,
    manifest: dict[str, np.ndarray],
    positions: np.ndarray,
    action_advantage: np.ndarray,
    alpha: float,
    formula_weight: np.ndarray,
    row_position: dict[int, int],
    device: torch.device,
    temperature: float,
    advantage_cap: float,
) -> torch.Tensor:
    """Deliver a privileged action advantage only to the clean query path."""
    selected = np.asarray(positions, dtype=np.int64)
    query_position = np.asarray(
        [row_position[int(row)] for row in manifest["query_row"][selected]],
        dtype=np.int64,
    )
    positive_position = np.asarray(
        [row_position[int(row)] for row in manifest["positive_reference_row"][selected]],
        dtype=np.int64,
    )
    negative_position = np.asarray(
        [row_position[int(row)] for row in manifest["negative_reference_row"][selected]],
        dtype=np.int64,
    )
    query_x = torch.from_numpy(np.array(official[query_position], copy=True)).to(device)
    positive_x = torch.from_numpy(np.array(official[positive_position], copy=True)).to(device)
    negative_x = torch.from_numpy(np.array(official[negative_position], copy=True)).to(device)
    query = model(query_x)
    # References use the shared adapter at deployment, but the privileged PMT
    # term cannot move shared high-multiplicity candidates directly.
    positive = model(positive_x).detach()
    negative = model(negative_x).detach()
    current_margin = torch.sum(query * positive, dim=1) - torch.sum(query * negative, dim=1)
    initial = torch.as_tensor(
        manifest["official_margin"][selected], dtype=query.dtype, device=device
    )
    advantage = torch.as_tensor(
        action_advantage[selected], dtype=query.dtype, device=device
    )
    target = inherited_clean_margin_target(initial, advantage, alpha, advantage_cap)
    weight = torch.as_tensor(
        formula_weight[selected], dtype=query.dtype, device=device
    )
    return active_margin_transfer_loss(current_margin, target, weight, temperature)


def evaluate_crossmodal(body: dict[str, np.ndarray], queries: np.ndarray,
                        spectrum: np.ndarray, molecule: np.ndarray,
                        row_position: dict[int, int],
                        molecule_position: np.ndarray) -> dict:
    ranks, margins = [], []
    for query in queries:
        qpos = row_position[int(body["query_row"][query])]
        left, right = map(int, body["query_ptr"][query:query + 2])
        scores = molecule[molecule_position[left:right]] @ spectrum[qpos]
        ranks.append(strict_rank(scores))
        margins.append(float(scores[0] - np.max(scores[1:])))
    ranks = np.asarray(ranks)
    return {
        "queries": int(len(queries)), "recall1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1 / ranks)), "mean_margin": float(np.mean(margins)),
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.inner_fold == args.outer_fold:
        raise ValueError("inner and outer folds must differ")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    bpmt_enabled = args.bpmt_arm != "none"
    if bpmt_enabled and (
        args.arm != "spectrum_only"
        or args.bpmt_manifest_dir is None
        or args.lambda_bpmt <= 0
        or args.bpmt_batch_actions < 1
        or args.bpmt_temperature <= 0
    ):
        raise ValueError("B-PMT requires spectrum_only and valid paired-margin settings")
    if not bpmt_enabled and args.bpmt_manifest_dir is not None:
        raise ValueError("a B-PMT manifest was supplied to a disabled B-PMT arm")
    token_ledger = args.token_dir / ("progress.json" if args.allow_incomplete_cache else "report.json")
    required = [args.manifest, token_ledger, args.token_dir / "rows.npy",
                args.token_dir / "official_embeddings_f32.npy"]
    if bpmt_enabled:
        required.extend((
            args.bpmt_manifest_dir / "report.json",
            args.bpmt_manifest_dir / "transfer_manifest.npz",
        ))
    if args.arm != "spectrum_only":
        required.extend((args.molecule_dir / "report.json", args.molecule_dir / "identities.npy",
                         args.molecule_dir / "embeddings_f16.npy"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    token_report = json.loads(token_ledger.read_text(encoding="utf-8"))
    molecule_report = (
        json.loads((args.molecule_dir / "report.json").read_text(encoding="utf-8"))
        if args.arm != "spectrum_only" else None
    )
    if not args.allow_incomplete_cache and token_report.get("status") != "chemaware_corrected_manifest_token_cache_complete":
        raise RuntimeError("full token cache is not complete")
    if args.allow_incomplete_cache and token_report.get("status") != "in_progress":
        raise RuntimeError("incomplete-cache smoke requires an in-progress ledger")
    if molecule_report is not None and molecule_report.get("status") != "chemaware_chemberta_teacher_complete":
        raise RuntimeError("ChemBERTa teacher is not complete")
    with np.load(args.manifest) as source:
        body = {key: source[key] for key in source.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    if args.allow_incomplete_cache:
        completed = int(token_report["completed_rows"])
        if args.incomplete_cache_rows:
            if args.incomplete_cache_rows > completed:
                raise ValueError("requested incomplete-cache-rows exceed atomically completed rows")
            completed = args.incomplete_cache_rows
        rows = rows[:completed]
        official = official[:completed]
    row_position = {int(row): index for index, row in enumerate(rows)}
    if not args.allow_incomplete_cache and set(map(int, np.unique(np.r_[body["query_row"], body["pair_candidate_row"]]))) - set(row_position):
        raise RuntimeError("token cache misses manifest-reachable rows")
    bpmt_report = None
    bpmt_manifest = None
    bpmt_advantage = None
    bpmt_alpha = 0.0
    bpmt_formula_weight = None
    bpmt_advantage_cap = None
    if bpmt_enabled:
        bpmt_report_path = args.bpmt_manifest_dir / "report.json"
        bpmt_manifest_path = args.bpmt_manifest_dir / "transfer_manifest.npz"
        bpmt_report = json.loads(bpmt_report_path.read_text(encoding="utf-8"))
        if (
            bpmt_report.get("status") != "CHEMAWARE_BOUNDARY_PMT_MANIFEST_ADMITTED"
            or bpmt_report.get("training_authorized") is not True
            or bpmt_report.get("provenance", {}).get("transfer_manifest_sha256")
            != sha256_file(bpmt_manifest_path)
        ):
            raise RuntimeError("B-PMT manifest is not admitted or its provenance changed")
        bpmt_advantage_cap = float(bpmt_report.get("thresholds", {}).get("advantage_cap", -1))
        if bpmt_advantage_cap <= 0:
            raise RuntimeError("B-PMT report has no valid frozen advantage cap")
        with np.load(bpmt_manifest_path) as source:
            bpmt_manifest = {key: source[key] for key in source.files}
        required_keys = {
            "selected_query", "query_row", "positive_reference_row",
            "negative_reference_row", "formula", "formula_fold", "official_margin",
            "strict_action_advantage", "corrective_weight",
            "matched_formula_deranged_advantage",
            "matched_formula_deranged_source_index",
            "matched_formula_deranged_source_formula",
        }
        if missing_keys := required_keys - set(bpmt_manifest):
            raise RuntimeError(f"B-PMT manifest misses arrays: {sorted(missing_keys)}")
        count = len(bpmt_manifest["selected_query"])
        if not count or any(len(bpmt_manifest[key]) != count for key in required_keys):
            raise RuntimeError("B-PMT manifest arrays are empty or misaligned")
        selected_query = bpmt_manifest["selected_query"].astype(np.int64)
        if (
            np.any(selected_query < 0)
            or np.any(selected_query >= len(body["query_row"]))
            or not np.array_equal(body["query_row"][selected_query], bpmt_manifest["query_row"])
            or not np.array_equal(body["query_formula"][selected_query].astype(str), bpmt_manifest["formula"].astype(str))
        ):
            raise RuntimeError("B-PMT queries do not align to the full candidate manifest")
        needed_rows = np.unique(np.concatenate((
            bpmt_manifest["query_row"], bpmt_manifest["positive_reference_row"],
            bpmt_manifest["negative_reference_row"],
        ))).astype(np.int64)
        if set(map(int, needed_rows)) - set(row_position):
            raise RuntimeError("B-PMT references are absent from the official cache")
        query_pos = np.asarray([row_position[int(row)] for row in bpmt_manifest["query_row"]])
        positive_pos = np.asarray([row_position[int(row)] for row in bpmt_manifest["positive_reference_row"]])
        negative_pos = np.asarray([row_position[int(row)] for row in bpmt_manifest["negative_reference_row"]])
        replayed_margin = np.sum(official[query_pos] * official[positive_pos], axis=1) - np.sum(
            official[query_pos] * official[negative_pos], axis=1
        )
        if not np.allclose(replayed_margin, bpmt_manifest["official_margin"], atol=2e-5):
            raise RuntimeError("B-PMT clean boundary does not replay in official geometry")
        bpmt_alpha = {
            "clean_duplicate": 0.0,
            "matched_formula_deranged": 0.5,
            "alpha025": 0.25,
            "alpha050": 0.5,
        }[args.bpmt_arm]
        bpmt_advantage = (
            bpmt_manifest["matched_formula_deranged_advantage"].astype(np.float32)
            if args.bpmt_arm == "matched_formula_deranged"
            else bpmt_manifest["strict_action_advantage"].astype(np.float32)
        )
        if len(bpmt_advantage) != count or np.any(bpmt_advantage < 0):
            raise RuntimeError("B-PMT action advantages are invalid")
        formula_name = bpmt_manifest["formula"].astype(str)
        deranged_source = bpmt_manifest[
            "matched_formula_deranged_source_index"
        ].astype(np.int64)
        if (
            np.any(deranged_source < 0)
            or np.any(deranged_source >= count)
            or not np.array_equal(
                bpmt_manifest["matched_formula_deranged_source_formula"].astype(str),
                formula_name[deranged_source],
            )
            or np.any(formula_name[deranged_source] == formula_name)
            or not np.allclose(
                bpmt_manifest["matched_formula_deranged_advantage"],
                bpmt_manifest["strict_action_advantage"][deranged_source],
            )
        ):
            raise RuntimeError("B-PMT matched control lost formula-disjoint provenance")
        _, formula_inverse, formula_count = np.unique(
            formula_name, return_inverse=True, return_counts=True
        )
        bpmt_formula_weight = (
            bpmt_manifest["corrective_weight"].astype(np.float64)
            / formula_count[formula_inverse]
        )
        bpmt_formula_weight *= len(bpmt_formula_weight) / np.sum(bpmt_formula_weight)
        bpmt_formula_weight = bpmt_formula_weight.astype(np.float32)
    mol_embedding = None
    molecule_position = None
    if args.arm != "spectrum_only":
        mol_identity = np.load(args.molecule_dir / "identities.npy").astype(str)
        mol_embedding = np.load(args.molecule_dir / "embeddings_f16.npy", mmap_mode="r")
        mol_lookup = {value: index for index, value in enumerate(mol_identity)}
        try:
            molecule_position = np.asarray([mol_lookup[value] for value in body["molecule_ik14"].astype(str)], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"molecule teacher misses manifest identity: {error}") from error

    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    molecule_fold = stable_formula_folds(body["molecule_formula"], args.folds, args.fold_seed)
    training_molecule_allowed = ((molecule_fold != args.inner_fold)
                                 & (molecule_fold != args.outer_fold))
    eligible = np.ones(len(body["query_row"]), dtype=bool)
    if args.allow_incomplete_cache:
        available = np.zeros(int(max(np.max(body["pair_candidate_row"]), np.max(body["query_row"]))) + 1, dtype=bool)
        available[rows] = True
        pair_available = available[body["pair_candidate_row"]]
        molecule_available = np.add.reduceat(pair_available.astype(np.int32), body["molecule_ptr"][:-1]) > 0
        query_candidates_available = np.minimum.reduceat(
            molecule_available.astype(np.int8), body["query_ptr"][:-1]
        ).astype(bool)
        eligible = available[body["query_row"]] & query_candidates_available
    allowed_candidate_count = np.add.reduceat(
        training_molecule_allowed.astype(np.int32), body["query_ptr"][:-1]
    )
    train_pool = np.flatnonzero(
        eligible & (fold != args.inner_fold) & (fold != args.outer_fold)
        & (allowed_candidate_count >= 2)
    )
    inner_pool = np.flatnonzero(eligible & (fold == args.inner_fold))
    outer_pool = np.flatnonzero(eligible & (fold == args.outer_fold))
    if bpmt_enabled and (
        not np.array_equal(
            fold[bpmt_manifest["selected_query"].astype(np.int64)],
            bpmt_manifest["formula_fold"].astype(np.int16),
        )
        or np.any(np.isin(bpmt_manifest["formula_fold"], (args.inner_fold, args.outer_fold)))
    ):
        raise RuntimeError("B-PMT formula folds conflict with train/evaluation roles")
    if not len(train_pool) or not len(inner_pool):
        raise RuntimeError("cache prefix has insufficient formula-split train/inner queries")
    train_official_error = np.zeros(len(body["query_row"]), dtype=bool)
    train_official_margin = np.full(len(body["query_row"]), np.nan, dtype=np.float32)
    if args.error_identity_fraction > 0:
        train_official_error, train_official_margin = official_outcomes(
            body, train_pool, official, row_position, training_molecule_allowed,
        )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_eval_identities,
    )
    train_diagnostic = identity_balanced_queries(
        train_pool, body["query_ik14"], np.random.default_rng(args.seed + 23),
        args.train_diagnostic_identities,
    ) if args.train_diagnostic_identities else np.empty(0, dtype=np.int64)
    preflight = {
        "status": (
            "chemaware_full_candidate_alignment_incomplete_cache_smoke"
            if args.allow_incomplete_cache else
            "chemaware_full_candidate_alignment_preflight_passed"
        ),
        "manifest_queries": int(len(body["query_row"])), "reachable_spectra": int(len(rows)),
        "candidate_molecules": int(len(body["molecule_label"])),
        "candidate_spectrum_edges": int(len(body["pair_candidate_row"])),
        "train_query_pool": int(len(train_pool)), "inner_eval_identities": int(len(inner)),
        "train_queries_removed_without_legal_negative": int(np.sum(
            eligible & (fold != args.inner_fold) & (fold != args.outer_fold)
            & (allowed_candidate_count < 2)
        )),
        "train_official_errors": int(np.sum(train_official_error[train_pool])),
        "error_identity_fraction": float(args.error_identity_fraction),
        "clean_safety_selection": args.clean_safety_selection,
        "train_diagnostic_identities": int(len(train_diagnostic)),
        "outer_queries_untouched": int(len(outer_pool)),
        "bpmt": (
            None
            if not bpmt_enabled
            else {
                "arm": args.bpmt_arm,
                "alpha": bpmt_alpha,
                "actions": int(len(bpmt_manifest["selected_query"])),
                "formulas": int(len(np.unique(bpmt_manifest["formula"].astype(str)))),
                "all_actions_covered_once_before_recycling": True,
                "control_gradient_subtraction": False,
                "noncorrective_weight": 0.0,
            }
        ),
        "contracts": {
            "formula_disjoint": True, "identity_equal_epoch_sampling": True,
            "formula_identity_equal_training_mass": args.training_mass == "formula_identity",
            "held_formula_candidates_excluded_from_training_gradients": True,
            "same_spectrum_adapter_query_reference": True,
            "molecule_teacher_training_only": (
                True if molecule_report is not None else None
            ),
            "molecule_teacher_used": molecule_report is not None,
            "outer_fold_evaluated": False,
            "candidate_input_at_deployment": False,
            "bpmt_controls_are_separate_arms": True if bpmt_enabled else None,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(token_ledger),
            "molecule_report_sha256": (
                sha256_file(args.molecule_dir / "report.json")
                if molecule_report is not None else None
            ),
            "bpmt_report_sha256": (
                sha256_file(args.bpmt_manifest_dir / "report.json")
                if bpmt_enabled else None
            ),
            "bpmt_manifest_sha256": (
                sha256_file(args.bpmt_manifest_dir / "transfer_manifest.npz")
                if bpmt_enabled else None
            ),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2)); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    spectrum_model = ResidualSpectrumProjector(official.shape[1], args.hidden_dim, args.dropout).to(device)
    molecule_model = (
        MoleculeProjector(mol_embedding.shape[1], args.hidden_dim, official.shape[1], args.dropout).to(device)
        if mol_embedding is not None else None
    )
    parameter_groups = [
            {"params": list(spectrum_model.parameters()), "lr": args.learning_rate,
             "initial_lr": args.learning_rate, "role": "spectrum"},
    ]
    if molecule_model is not None:
        parameter_groups.append(
            {"params": list(molecule_model.parameters()), "lr": args.molecule_learning_rate,
             "initial_lr": args.molecule_learning_rate, "role": "molecule"},
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 101)
    history = []
    initial_inner = evaluate(body, inner, official, official, row_position)
    initial_train_diagnostic = (
        evaluate(body, train_diagnostic, official, official, row_position)
        if len(train_diagnostic) else None
    )
    initial_inner_crossmodal = None
    if molecule_model is not None:
        initial_molecule_projection = project_numpy(
            molecule_model, mol_embedding, device, args.eval_batch_size,
        )
        initial_inner_crossmodal = evaluate_crossmodal(
            body, inner, official, initial_molecule_projection, row_position, molecule_position,
        )
    best_state = copy.deepcopy(spectrum_model.state_dict())
    best_step = 0
    best_score = (
        initial_inner["summary"]["recall1"], initial_inner["summary"]["mrr"],
        initial_inner["summary"]["delta_mean_margin"], 1.0,
    )
    identities_per_epoch = min(
        len(np.unique(body["query_ik14"][train_pool])),
        args.max_train_identities or len(train_pool),
    )
    batches_per_epoch = math.ceil(identities_per_epoch / args.batch_queries)
    bpmt_actions_per_batch = (
        max(
            args.bpmt_batch_actions,
            math.ceil(len(bpmt_manifest["selected_query"]) / batches_per_epoch),
        )
        if bpmt_enabled
        else 0
    )
    total_steps = args.max_steps or (args.epochs * batches_per_epoch)
    if total_steps <= 0:
        raise ValueError("training requires a positive optimizer-step budget")
    if args.molecule_only_warmup_steps >= total_steps:
        raise ValueError("molecule-only-warmup-steps must be smaller than total optimizer steps")
    joint_steps = total_steps - args.molecule_only_warmup_steps
    if args.warmup_steps >= joint_steps:
        raise ValueError("warmup-steps must be smaller than joint-training steps")
    if args.arm == "spectrum_only" and args.molecule_only_warmup_steps:
        raise ValueError("spectrum_only arm cannot use molecule-only warmup")
    global_step = 0
    epoch = 0
    inner_eval = initial_inner
    adapted = np.asarray(official)
    while global_step < total_steps:
        epoch += 1
        spectrum_model.train()
        if molecule_model is not None:
            molecule_model.train()
        epoch_query = (
            error_curriculum_queries(
                train_pool, body["query_ik14"], train_official_error,
                train_official_margin, rng, args.max_train_identities,
                args.error_identity_fraction, args.clean_safety_selection,
            )
            if args.error_identity_fraction > 0 else
            identity_balanced_queries(
                train_pool, body["query_ik14"], rng, args.max_train_identities,
            )
        )
        rng.shuffle(epoch_query)
        epoch_weight = formula_identity_epoch_weights(
            epoch_query, body["query_formula"], args.training_mass,
        )
        epoch_weight_by_query = {
            int(query): float(weight) for query, weight in zip(epoch_query, epoch_weight)
        }
        bpmt_order = (
            rng.permutation(len(bpmt_manifest["selected_query"]))
            if bpmt_enabled
            else np.empty(0, dtype=np.int64)
        )
        bpmt_cursor = 0
        totals = {"loss": 0.0, "spectrum": 0.0, "molecule": 0.0,
                  "inbatch_spectrum": 0.0, "inbatch_molecule": 0.0, "preserve": 0.0,
                  "margin_floor": 0.0, "bpmt": 0.0,
                  "grad_norm": 0.0, "clip_fraction": 0.0}
        batches = 0
        for left in range(0, len(epoch_query), args.batch_queries):
            if global_step >= total_steps:
                break
            queries = epoch_query[left:left + args.batch_queries]
            batch = sample_training_batch(
                body, queries, row_position, molecule_position,
                args.references_per_molecule, rng, training_molecule_allowed,
            )
            query_x = torch.from_numpy(np.asarray(official[batch["query_cache"]], dtype=np.float32)).to(device)
            reference_x = torch.from_numpy(np.asarray(official[batch["reference_cache"]], dtype=np.float32)).to(device)
            molecule_x = (
                torch.from_numpy(np.asarray(mol_embedding[batch["molecule_teacher"]], dtype=np.float32)).to(device)
                if mol_embedding is not None else None
            )
            optimizer.zero_grad(set_to_none=True)
            molecule_warmup = global_step < args.molecule_only_warmup_steps
            if molecule_warmup:
                query_z = F.normalize(query_x, dim=-1)
                reference_z = F.normalize(reference_x, dim=-1)
            else:
                query_z = spectrum_model(query_x)
                reference_z = spectrum_model(reference_x)
            molecule_z = molecule_model(molecule_x) if molecule_model is not None else None
            if args.arm == "candidate_swapped":
                assert molecule_z is not None
                swapped = molecule_z.clone()
                for a, b in zip(batch["candidate_ptr"][:-1], batch["candidate_ptr"][1:]):
                    swapped[int(a):int(b)] = torch.roll(molecule_z[int(a):int(b)], 1, 0)
                molecule_z = swapped
            (spectrum_loss, molecule_loss, inbatch_spectrum,
             inbatch_molecule, margin_floor) = listwise_losses(
                query_z, reference_z, molecule_z, batch["candidate_ptr"],
                batch["reference_ptr"], batch["reference_edge"], args.temperature,
                query_x, reference_x, args.margin_floor_slack,
                torch.as_tensor(
                    [epoch_weight_by_query[int(query)] for query in queries],
                    device=device, dtype=query_z.dtype,
                ),
            )
            preserve = torch.cat((1 - torch.sum(query_z * query_x, dim=1),
                                  1 - torch.sum(reference_z * reference_x, dim=1))).mean()
            bpmt_loss = query_z.sum() * 0.0
            if bpmt_enabled:
                right = min(
                    len(bpmt_order), bpmt_cursor + bpmt_actions_per_batch
                )
                bpmt_positions = bpmt_order[bpmt_cursor:right]
                bpmt_cursor = right
                if not len(bpmt_positions):
                    bpmt_order = rng.permutation(len(bpmt_manifest["selected_query"]))
                    bpmt_cursor = min(bpmt_actions_per_batch, len(bpmt_order))
                    bpmt_positions = bpmt_order[:bpmt_cursor]
                bpmt_loss = bpmt_loss_for_positions(
                    spectrum_model,
                    official,
                    bpmt_manifest,
                    bpmt_positions,
                    bpmt_advantage,
                    bpmt_alpha,
                    bpmt_formula_weight,
                    row_position,
                    device,
                    args.bpmt_temperature,
                    bpmt_advantage_cap,
                )
            molecule_weight = 0.0 if args.arm == "spectrum_only" else args.lambda_molecule
            inbatch_molecule_weight = (
                0.0 if args.arm == "spectrum_only" else args.lambda_inbatch_molecule
            )
            if molecule_warmup:
                loss = (molecule_weight * molecule_loss
                        + inbatch_molecule_weight * inbatch_molecule)
            else:
                loss = (args.lambda_spectrum * spectrum_loss + molecule_weight * molecule_loss
                        + args.lambda_inbatch_spectrum * inbatch_spectrum
                        + inbatch_molecule_weight * inbatch_molecule
                        + args.lambda_margin_floor * margin_floor
                        + args.lambda_preserve * preserve
                        + (args.lambda_bpmt * bpmt_loss if bpmt_enabled else 0.0))
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                list(spectrum_model.parameters()) + (
                    list(molecule_model.parameters()) if molecule_model is not None else []
                ), args.grad_clip,
            ))
            global_step += 1
            if molecule_warmup:
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = args.molecule_learning_rate
            else:
                joint_step = global_step - args.molecule_only_warmup_steps
                scale = learning_rate_scale(joint_step, joint_steps, args.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * scale
            optimizer.step()
            for key, value in (("loss", loss), ("spectrum", spectrum_loss),
                               ("molecule", molecule_loss),
                               ("inbatch_spectrum", inbatch_spectrum),
                               ("inbatch_molecule", inbatch_molecule),
                               ("margin_floor", margin_floor),
                               ("bpmt", bpmt_loss),
                               ("preserve", preserve)):
                totals[key] += float(value.detach())
            totals["grad_norm"] += grad_norm; totals["clip_fraction"] += float(grad_norm > args.grad_clip)
            batches += 1
        should_evaluate = (
            global_step >= total_steps
            or (args.eval_every_steps > 0 and global_step % args.eval_every_steps < batches)
        )
        if should_evaluate:
            adapted = project_numpy(spectrum_model, official, device, args.eval_batch_size)
            inner_eval = evaluate(body, inner, official, adapted, row_position)
            preservation_mean = float(np.mean(np.sum(adapted * official, axis=1)))
            inner_crossmodal = None
            if molecule_model is not None:
                molecule_projection = project_numpy(
                    molecule_model, mol_embedding, device, args.eval_batch_size,
                )
                inner_crossmodal = evaluate_crossmodal(
                    body, inner, adapted, molecule_projection, row_position, molecule_position,
                )
            train_eval = (
                evaluate(body, train_diagnostic, official, adapted, row_position)
                if len(train_diagnostic) else None
            )
            entry = {
                "epoch": epoch, "global_step": global_step,
                "spectrum_learning_rate": optimizer.param_groups[0]["lr"],
                "molecule_learning_rate": (
                    optimizer.param_groups[1]["lr"] if molecule_model is not None else None
                ),
                "train": {key: value / max(1, batches) for key, value in totals.items()},
                "inner": inner_eval["summary"],
                "inner_crossmodal": inner_crossmodal,
                "preservation_mean": preservation_mean,
                "train_diagnostic": None if train_eval is None else train_eval["summary"],
            }
            history.append(entry)
            score = (
                entry["inner"]["recall1"], entry["inner"]["mrr"],
                entry["inner"]["delta_mean_margin"], preservation_mean,
            )
            if preservation_mean >= 0.995 and score > best_score:
                best_score = score
                best_step = global_step
                best_state = copy.deepcopy(spectrum_model.state_dict())
            print(f"arm={args.arm} step={global_step}/{total_steps} delta={entry['inner']['delta_recall1']:+.4f} corrected={entry['inner']['corrected']} introduced={entry['inner']['introduced']} preserve={preservation_mean:.6f} best_step={best_step}", flush=True)
    spectrum_model.load_state_dict(best_state, strict=True)
    adapted = project_numpy(spectrum_model, official, device, args.eval_batch_size)
    final = evaluate(body, inner, official, adapted, row_position)
    selected_train_diagnostic = (
        evaluate(body, train_diagnostic, official, adapted, row_position)["summary"]
        if len(train_diagnostic) else None
    )
    delta = (final["new_rank"] == 1).astype(float) - (final["old_rank"] == 1).astype(float)
    ci = formula_bootstrap(delta, body["query_formula"][inner], args.seed + 701, args.bootstrap_draws)
    preservation = np.sum(adapted * official, axis=1)
    gates = {
        "inner_recall_positive": final["summary"]["delta_recall1"] > 0,
        "formula_ci_positive": ci["formula_cluster_bootstrap_95ci"][0] > 0,
        "corrected_exceeds_introduced": final["summary"]["corrected"] > final["summary"]["introduced"],
        "mean_preservation": float(np.mean(preservation)) >= 0.995,
    }
    torch.save({
        "status": "chemaware_full_candidate_aligned_embedding",
        "format": "chemaware_global_embedding_adapter_v1",
        "adapter_state": spectrum_model.state_dict(),
        "adapter_config": {
            "dimension": int(official.shape[1]), "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "arm": args.arm, "seed": args.seed, "outer_fold": args.outer_fold,
        "bpmt_arm": args.bpmt_arm,
        "P2b_used": False, "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "training_only_molecule_projector_used": args.arm != "spectrum_only",
        "molecule_projector_state": None,
        "chemical_supervision": args.arm != "spectrum_only",
        "chemical_boundary_supervision": bpmt_enabled,
        "teacher_control": args.arm,
        "formal": not args.allow_incomplete_cache,
        "validation_pass": (not args.allow_incomplete_cache) and all(gates.values()),
        "provenance": token_report.get("provenance", {}) | {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(token_ledger),
            "molecule_report_sha256": (
                sha256_file(args.molecule_dir / "report.json")
                if molecule_report is not None else None
            ),
            "official_checkpoint_sha256": token_report.get(
                "official_checkpoint_sha256",
                token_report.get("provenance", {}).get("official_checkpoint_sha256"),
            ),
        },
        "deployment_input": "official normalized DreaMS embedding",
        "deployment_output": "adapted normalized 1024-dimensional shared embedding",
        "molecule_projector_discarded": molecule_model is not None,
    }, args.output / "shared_spectrum_adapter.pt")
    report = {
        "status": (
            "SMOKE_ONLY" if args.allow_incomplete_cache
            else ("PASS" if all(gates.values()) else "FAIL")
        ),
        "preflight": preflight, "arm": args.arm,
        "bpmt_arm": args.bpmt_arm,
        "optimization": vars(args) | {"manifest": str(args.manifest), "token_dir": str(args.token_dir),
                                       "molecule_dir": str(args.molecule_dir), "output": str(args.output)},
        "initial_inner": initial_inner["summary"],
        "final_inner": final["summary"], "formula_bootstrap": ci,
        "initial_inner_crossmodal": initial_inner_crossmodal,
        "final_inner_crossmodal": (
            None if not history else history[-1]["inner_crossmodal"]
        ),
        "initial_train_diagnostic": (
            None if initial_train_diagnostic is None else initial_train_diagnostic["summary"]
        ),
        "final_train_diagnostic": (
            selected_train_diagnostic
        ),
        "optimizer_steps": int(global_step), "effective_total_steps": int(total_steps),
        "selected_step": int(best_step), "selection_score": list(map(float, best_score)),
        "preservation": {"mean": float(np.mean(preservation)), "min": float(np.min(preservation)),
                         "q01": float(np.quantile(preservation, 0.01))},
        "gates": gates, "history": history,
        "scope": {"outer_fold_evaluated": False, "development_only": True,
                  "candidate_or_structure_at_deployment": False,
                  "same_adapter_for_query_and_reference": True,
                  "bpmt_control_gradient_subtraction": False,
                  "bpmt_noncorrective_weight": 0.0 if bpmt_enabled else None},
        "runtime_seconds": time.time() - started,
    }
    np.savez_compressed(
        args.output / "inner_per_query.npz",
        query=inner,
        formula=body["query_formula"][inner].astype(str),
        old_rank=final["old_rank"].astype(np.int16),
        new_rank=final["new_rank"].astype(np.int16),
        candidate_count=final["candidate_count"].astype(np.int32),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "final_inner": report["final_inner"],
                      "formula_bootstrap": ci, "preservation": report["preservation"]}, indent=2))


if __name__ == "__main__":
    main()

"""Direct, non-distillation reachability test for action-routed noise training.

The exact A4 action scan is used only to route training examples: an error is
marked action-correctable when at least one legal peak intervention changes its
molecular Top-1 to the truth.  No action embedding, teacher margin target or
teacher residual is regressed.  The optimization label is always the true
molecular identity under the real 10-ppm candidate set.

A zero-initialized residual metric is applied to both clean queries and
candidate references.  This makes the test a cheap local proxy for a shared
encoder update and, unlike the historical E4 loss, directly asks the clean
query to beat its candidate molecules.  Formula-OOF and deliberately weaker
in-sample-capacity modes are both supported and labelled separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from audit_noise_peak_gate_candidate_injection import (
    build_candidate_rows,
    formula_ci,
    stable_fold,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a4-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_a4_exact_peak_scan"),
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("data/validation/chemaware_corrected_manifest_tokens_v1"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/noise_direct_shared_metric_reachability_20260905"),
    )
    parser.add_argument("--evaluation-mode", choices=("formula_oof", "in_sample_capacity"), default="formula_oof")
    parser.add_argument("--route", choices=("clean_uniform", "action_correctable"), default="action_correctable")
    parser.add_argument(
        "--map-side", choices=("shared", "query_only"), default="shared",
        help="query_only is a diagnostic for cancellation caused by moving the reference bank.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--adapter-type", choices=("lowrank", "diagonal"), default="lowrank")
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--residual-strength", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--correctable-weight", type=float, default=4.0)
    parser.add_argument("--other-error-weight", type=float, default=0.75)
    parser.add_argument("--correct-control-weight", type=float, default=0.50)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--safety-slack", type=float, default=0.005)
    parser.add_argument("--preserve-weight", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


class SharedResidualMetric(torch.nn.Module):
    """Identity-initialized low-rank map shared by queries and references."""

    def __init__(self, dimension: int, hidden: int, strength: float):
        super().__init__()
        self.strength = float(strength)
        self.norm = torch.nn.LayerNorm(dimension)
        self.down = torch.nn.Linear(dimension, hidden)
        self.up = torch.nn.Linear(hidden, dimension)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(value))))
        # A preservation penalty measured only on training supports does not
        # constrain extrapolation to an unseen formula.  Bound every residual
        # structurally so ``residual_strength`` is a real per-spectrum trust
        # region rather than a multiplier on an unbounded MLP output.
        residual = residual / torch.linalg.vector_norm(
            residual, dim=-1, keepdim=True
        ).clamp_min(1.0)
        return F.normalize(value + self.strength * residual, dim=-1)


class SharedDiagonalMetric(torch.nn.Module):
    """Bounded global feature reweighting with identity initialization."""

    def __init__(self, dimension: int, strength: float):
        super().__init__()
        self.strength = float(strength)
        self.log_scale = torch.nn.Parameter(torch.zeros(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.strength * torch.tanh(self.log_scale))
        return F.normalize(value * scale, dim=-1)


def attach_action_route(a4_dir: Path, query: pd.DataFrame) -> pd.DataFrame:
    import h5py

    with h5py.File(a4_dir / "exact_peak_scan.h5", "r") as handle:
        action_query = handle["action_query"][:].astype(np.int64)
        doses = json.loads(handle.attrs["attenuations_json"])
        ranks = handle["result_rank"][:].reshape(-1, len(doses))
        margins = handle["result_margin"][:].reshape(-1, len(doses))
    by_scan = {
        int(value): np.flatnonzero(action_query == int(value))
        for value in query.scan_position
    }
    best_rank = np.empty(len(query), dtype=np.int16)
    best_margin = np.empty(len(query), dtype=np.float32)
    for output, item in enumerate(query.itertuples(index=False)):
        action = by_scan[int(item.scan_position)]
        local_rank = ranks[action].reshape(-1)
        local_margin = margins[action].reshape(-1)
        # Lexicographic retrieval endpoint: minimize rank, then maximize margin.
        best = int(np.lexsort((-local_margin, local_rank))[0])
        best_rank[output] = local_rank[best]
        best_margin[output] = local_margin[best]
    result = query.copy()
    is_error = result.baseline_rank.to_numpy(np.int64) > 1
    result["action_best_rank"] = best_rank
    result["action_best_margin"] = best_margin
    result["action_correctable"] = is_error & (best_rank == 1)
    result["action_positive_advantage"] = np.maximum(
        0.0, best_margin - result.baseline_margin.to_numpy(np.float32)
    )
    return result


def route_weights(query: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if args.route == "clean_uniform":
        return np.ones(len(query), dtype=np.float32)
    is_error = query.baseline_rank.to_numpy(np.int64) > 1
    correctable = query.action_correctable.to_numpy(bool)
    return np.where(
        correctable,
        args.correctable_weight,
        np.where(is_error, args.other_error_weight, args.correct_control_weight),
    ).astype(np.float32)


def transform_numpy(
    model: SharedResidualMetric, values: np.ndarray, device: torch.device, batch: int,
) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(values), batch):
            right = min(left + batch, len(values))
            block = torch.as_tensor(
                np.array(values[left:right], dtype=np.float32, copy=True), device=device
            )
            output[left:right] = model(block).cpu().numpy()
    return output


def train_fold(
    train: np.ndarray,
    base_embeddings: np.ndarray,
    query_positions: np.ndarray,
    positive_positions: np.ndarray,
    negative_positions: np.ndarray,
    query: pd.DataFrame,
    weights: np.ndarray,
    args: argparse.Namespace,
    fold: int,
) -> tuple[SharedResidualMetric, dict[str, float]]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    device = torch.device(args.device)
    if args.adapter_type == "lowrank":
        model = SharedResidualMetric(
            base_embeddings.shape[1], args.hidden_dim, args.residual_strength,
        ).to(device)
    else:
        model = SharedDiagonalMetric(
            base_embeddings.shape[1], args.residual_strength,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + fold)
    baseline_correct = query.baseline_rank.to_numpy(np.int64) == 1
    last = None
    for epoch in range(args.epochs):
        order = rng.permutation(train)
        total = np.zeros(6, dtype=np.float64)
        for left in range(0, len(order), args.batch_size):
            index = order[left:left + args.batch_size]
            q0 = torch.as_tensor(np.asarray(base_embeddings[query_positions[index]], dtype=np.float32), device=device)
            p0 = torch.as_tensor(np.asarray(base_embeddings[positive_positions[index]], dtype=np.float32), device=device)
            n0 = torch.as_tensor(np.asarray(base_embeddings[negative_positions[index]], dtype=np.float32), device=device)
            q = model(q0)
            if args.map_side == "shared":
                p = model(p0)
                n = model(n0.reshape(-1, n0.shape[-1])).reshape_as(n0)
            else:
                p, n = p0, n0
            positive_score = torch.sum(q * p, dim=1)
            negative_score = torch.einsum("bd,bkd->bk", q, n)
            logits = torch.cat((positive_score[:, None], negative_score), dim=1) / args.temperature
            per_query = F.cross_entropy(logits, torch.zeros(len(index), dtype=torch.long, device=device), reduction="none")
            sample_weight = torch.as_tensor(weights[index], device=device)
            rank_loss = torch.sum(sample_weight * per_query) / sample_weight.sum().clamp_min(1e-8)

            initial_positive = torch.sum(q0 * p0, dim=1)
            initial_negative = torch.einsum("bd,bkd->bk", q0, n0)
            initial_margin = initial_positive - torch.max(initial_negative, dim=1).values
            current_margin = positive_score - torch.max(negative_score, dim=1).values
            safe = torch.as_tensor(baseline_correct[index], device=device)
            if bool(safe.any()):
                safety = F.relu(initial_margin[safe].detach() - args.safety_slack - current_margin[safe]).mean()
            else:
                safety = rank_loss * 0.0
            unique = torch.cat((q, p, n.reshape(-1, n.shape[-1])), dim=0)
            unique0 = torch.cat((q0, p0, n0.reshape(-1, n0.shape[-1])), dim=0)
            preserve = (1.0 - torch.sum(unique * unique0, dim=1)).mean()
            loss = rank_loss + args.safety_weight * safety + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += np.asarray([
                float(loss.detach()), float(rank_loss.detach()), float(safety.detach()),
                float(preserve.detach()), float(grad_norm), len(index),
            ]) * len(index)
        last = total[:5] / total[5]
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 10 == 0:
            print(
                f"[direct-shared route={args.route} fold={fold} epoch={epoch + 1}] "
                f"loss={last[0]:.5f} rank={last[1]:.5f} safety={last[2]:.5f} "
                f"preserve={last[3]:.5f} grad={last[4]:.3f}", flush=True,
            )
    assert last is not None
    return model, {
        "final_loss": float(last[0]),
        "final_rank_loss": float(last[1]),
        "final_safety_loss": float(last[2]),
        "final_preserve_loss": float(last[3]),
        "final_gradient_norm": float(last[4]),
    }


def evaluate_fold(
    test: np.ndarray,
    query: pd.DataFrame,
    query_vectors: np.ndarray,
    candidate_embeddings: np.ndarray,
    base_embeddings: np.ndarray,
    query_positions: np.ndarray,
    candidates: list[np.ndarray],
    candidate_identities: list[np.ndarray],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for local in test:
        item = query.iloc[int(local)]
        query_vector = query_vectors[local]
        local_candidates = candidates[local]
        identities = candidate_identities[local]
        scores = np.asarray(candidate_embeddings[local_candidates], dtype=np.float32) @ query_vector
        molecules = np.unique(identities)
        molecule_scores = np.asarray([np.max(scores[identities == value]) for value in molecules])
        positive = molecule_scores[molecules == str(item.query_ik14)][0]
        wrong = molecule_scores[molecules != str(item.query_ik14)]
        rank = 1 + int(np.sum(wrong >= positive))
        baseline_rank = int(item.baseline_rank)
        records.append({
            "query_index": int(item.query_index),
            "query_formula": str(item.query_formula),
            "formula_fold": int(item.formula_fold),
            "baseline_rank": baseline_rank,
            "final_rank": rank,
            "action_correctable": bool(item.action_correctable),
            "action_positive_advantage": float(item.action_positive_advantage),
            "corrected": bool(baseline_rank > 1 and rank == 1),
            "introduced": bool(baseline_rank == 1 and rank > 1),
            "embedding_cosine": float(
                query_vectors[local] @ np.asarray(
                    base_embeddings[query_positions[local]], dtype=np.float32
                )
            ),
        })
    return records


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cache_rows = np.load(args.cache_dir / "rows.npy")
    embeddings = np.load(args.cache_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    query = pd.read_csv(args.a4_dir / "scan_queries.csv.gz")
    query = query.loc[query.query_row.isin(cache_position)].reset_index(drop=True)
    query, positive, negatives, candidates, candidate_identities = build_candidate_rows(
        query, cache_rows, embeddings, args.data, args.top_negatives,
    )
    exact = query.rebuilt_rank.eq(query.baseline_rank) & query.baseline_margin_error.le(1e-5)
    query = query.loc[exact].reset_index(drop=True)
    positive = positive[exact]
    negatives = negatives[exact]
    candidates = [value for value, keep in zip(candidates, exact) if keep]
    candidate_identities = [value for value, keep in zip(candidate_identities, exact) if keep]
    query = attach_action_route(args.a4_dir, query)
    query["formula_fold"] = query.query_formula.astype(str).map(
        lambda value: stable_fold(value, args.folds, args.formula_fold_seed)
    ).astype(np.int8)
    query_positions = np.asarray([cache_position[int(row)] for row in query.query_row], dtype=np.int64)
    weights = route_weights(query, args)

    records: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    if args.evaluation_mode == "formula_oof":
        fold_specs = [
            (
                fold,
                np.flatnonzero(query.formula_fold.to_numpy() != fold),
                np.flatnonzero(query.formula_fold.to_numpy() == fold),
            )
            for fold in range(args.folds)
        ]
    else:
        all_indices = np.arange(len(query), dtype=np.int64)
        fold_specs = [(0, all_indices, all_indices)]
    for fold, train, test in fold_specs:
        if args.evaluation_mode == "formula_oof" and (
            set(query.iloc[train].query_formula) & set(query.iloc[test].query_formula)
        ):
            raise RuntimeError(f"formula leakage in fold {fold}")
        model, log = train_fold(
            train, embeddings, query_positions, positive, negatives, query, weights, args, fold,
        )
        if args.map_side == "shared":
            transformed = transform_numpy(model, embeddings, torch.device(args.device), 512)
            query_vectors = transformed[query_positions]
            candidate_embeddings = transformed
        else:
            query_vectors = transform_numpy(
                model, np.asarray(embeddings[query_positions], dtype=np.float32),
                torch.device(args.device), 512,
            )
            candidate_embeddings = embeddings
        records.extend(evaluate_fold(
            test, query, query_vectors, candidate_embeddings, embeddings, query_positions,
            candidates, candidate_identities,
        ))
        logs.append({
            "fold": fold if args.evaluation_mode == "formula_oof" else "in_sample_capacity",
            "train_queries": int(len(train)), "held_queries": int(len(test)), **log,
        })
    result = pd.DataFrame(records).sort_values("query_index").reset_index(drop=True)
    metrics = formula_ci(result, args.bootstrap_resamples, args.seed)
    metrics.update({
        "queries": int(len(result)),
        "formulas": int(result.query_formula.nunique()),
        "baseline_errors": int(result.baseline_rank.gt(1).sum()),
        "action_correctable_errors": int(result.action_correctable.sum()),
        "corrected": int(result.corrected.sum()),
        "introduced": int(result.introduced.sum()),
        "preservation_mean": float(result.embedding_cosine.mean()),
        "preservation_p01": float(result.embedding_cosine.quantile(0.01)),
    })
    report = {
        "status": "noise_direct_shared_metric_reachability_complete",
        "metrics": metrics,
        "folds": logs,
        "pass_4pp_local_reachability": bool(metrics["delta_pp"] >= 4.0),
        "pass_positive_formula_ci": bool(metrics["ci_low_pp"] > 0),
        "contracts": {
            "formula_oof": bool(args.evaluation_mode == "formula_oof"),
            "in_sample_capacity_only": bool(args.evaluation_mode == "in_sample_capacity"),
            "shared_query_and_candidate_map": bool(args.map_side == "shared"),
            "query_only_cancellation_diagnostic": bool(args.map_side == "query_only"),
            "teacher_embedding_target_used": False,
            "teacher_margin_target_used": False,
            "action_outcome_used_only_for_training_route": bool(args.route == "action_correctable"),
            "identity_is_only_ranking_label": True,
            "held_formula_action_outcomes_enter_training": bool(args.evaluation_mode == "in_sample_capacity"),
            "baseline_graph_reproduced_before_training": True,
        },
        "claim_limit": (
            "In-sample action-routed capacity diagnostic; not generalization or an encoder checkpoint."
            if args.evaluation_mode == "in_sample_capacity" else
            "Formula-OOF shared-metric proxy on the baseline-exact A4 cohort; not yet an encoder checkpoint or full official graph result."
        ),
        "configuration": vars(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output_dir / "query_results.csv.gz", index=False, compression="gzip")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

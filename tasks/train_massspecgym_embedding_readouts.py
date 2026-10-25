#!/usr/bin/env python
"""Train three frozen-encoder candidate readouts on MassSpecGym.

The encoders were trained only on formula folds 1--4.  Heads therefore train
on the same 65,286-query side and are selected once on the complete 18,333-
query outer fold 0.  Calling predictions for folds 1--4 OOF would be false:
their embeddings were already trained with those formulas.  GNPS and
application responses are not accepted by this entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from massspecgym_embedding_readout_core import (
    ARMS, ARM_PAIR, ARM_PROJECTED, ARM_SCALAR, CONTROL_ARMS,
    FULL_INTEGRATION_ARMS, REQUIRED_ENCODERS, FrozenEmbeddingReadout,
    ReadoutSettings, build_independent_scalar_features, hard_competitor_matrix, padded_candidate_matrix,
    fit_scalar_normalization, parse_named_paths, percentile_by_query, query_gap_column, rank_metrics,
    select_full_integration_arm, strict_ranks,
)
from noise_final_core import CandidateGraph, sha256_file, stable_fold


FOLDS = 5
HELD_FOLD = 0
FOLD_SEED = 20260825
TRAIN_SEED = 20261025


def load_complete_embeddings(paths: list[Path], rows: int) -> np.ndarray:
    """Assemble deterministic encoder shards without importing training code."""
    output: np.ndarray | None = None
    seen = np.zeros(rows, dtype=bool)
    dimension: int | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as body:
            shard_rows = np.asarray(body["rows"], dtype=np.int64)
            shard = np.asarray(body["embeddings"], dtype=np.float32)
        if (shard.ndim != 2 or len(shard_rows) != len(shard)
                or np.any((shard_rows < 0) | (shard_rows >= rows))
                or np.any(seen[shard_rows])):
            raise RuntimeError(f"invalid or overlapping embedding shard: {path}")
        if output is None:
            dimension = int(shard.shape[1])
            output = np.empty((rows, dimension), dtype=np.float32)
        if shard.shape[1] != dimension:
            raise RuntimeError("embedding shard dimensions disagree")
        norms = np.linalg.norm(shard, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise RuntimeError(f"non-finite embedding shard: {path}")
        output[shard_rows] = shard / norms[:, None]
        seen[shard_rows] = True
    if output is None or not np.all(seen):
        raise RuntimeError(f"embedding shards cover {int(seen.sum())}/{rows} rows")
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--chem-evidence", type=Path, required=True)
    parser.add_argument("--embedding", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--noise-name", default="noise_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dimension", type=int, default=16)
    parser.add_argument("--maximum-negatives", type=int, default=6)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_evidence(path: Path, graph: CandidateGraph) -> dict[str, np.ndarray]:
    target = path / "evidence.npz" if path.is_dir() else path
    with np.load(target, allow_pickle=False) as body:
        evidence = {name: np.asarray(body[name]) for name in body.files}
    if not np.array_equal(evidence["query_ptr"], graph.query_ptr):
        raise RuntimeError("scalar evidence query axis differs from corrected graph")
    if not np.array_equal(evidence["molecule_ptr"], graph.molecule_ptr):
        raise RuntimeError("scalar evidence molecule axis differs from corrected graph")
    if not np.array_equal(evidence["molecule_label"], graph.molecule_label):
        raise RuntimeError("scalar evidence labels differ from corrected graph")
    return evidence


def load_chemical_evidence(path: Path, evidence_path: Path, graph: CandidateGraph) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        chemical = {name: np.asarray(body[name]) for name in body.files}
    required = (
        "chem_valid", "candidate_utility_correct", "candidate_utility_zero",
        "candidate_utility_reversed", "candidate_utility_rotated", "evidence_sha256",
    )
    missing = [name for name in required if name not in chemical]
    if missing:
        raise RuntimeError(f"ChemAware V2 evidence misses arrays: {missing}")
    if str(chemical["evidence_sha256"].item()) != sha256_file(evidence_path):
        raise RuntimeError("ChemAware V2 evidence is not aligned to scalar evidence")
    if any(np.asarray(chemical[name]).shape != (len(graph.molecule_label),)
           for name in required[:-1]):
        raise RuntimeError("ChemAware V2 candidate axis differs from corrected graph")
    return chemical


def edge_cosines(
    graph: CandidateGraph, embedding: np.ndarray, device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    query_edge_ptr = graph.molecule_ptr[graph.query_ptr]
    edge_query = np.repeat(np.arange(graph.n_queries), np.diff(query_edge_ptr))
    query_rows = graph.query_row[edge_query]
    reference_rows = graph.pair_candidate_row
    # Reference selection is part of the ranking contract; compute it in
    # float32 so half-precision ties cannot change the molecule maximum.
    bank = torch.as_tensor(embedding, dtype=torch.float32, device=device)
    output = np.empty(len(reference_rows), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(output), chunk):
            right = min(left + chunk, len(output))
            q = bank[torch.as_tensor(query_rows[left:right], device=device)]
            r = bank[torch.as_tensor(reference_rows[left:right], device=device)]
            output[left:right] = torch.sum(q.float() * r.float(), 1).cpu().numpy()
    del bank
    torch.cuda.empty_cache()
    if not np.all(np.isfinite(output)):
        raise RuntimeError("embedding cosine construction produced non-finite values")
    return output


def molecule_max_and_reference(
    edge_scores: np.ndarray, graph: CandidateGraph,
) -> tuple[np.ndarray, np.ndarray]:
    molecule_scores = np.empty(len(graph.molecule_label), dtype=np.float32)
    reference_rows = np.empty(len(graph.molecule_label), dtype=np.int64)
    for molecule, (left, right) in enumerate(zip(graph.molecule_ptr[:-1], graph.molecule_ptr[1:])):
        left, right = int(left), int(right)
        local = int(np.argmax(edge_scores[left:right]))
        molecule_scores[molecule] = edge_scores[left + local]
        reference_rows[molecule] = graph.pair_candidate_row[left + local]
    return molecule_scores, reference_rows


def make_batches(query_ids: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = np.asarray(query_ids, dtype=np.int64).copy()
    rng.shuffle(order)
    for left in range(0, len(order), batch_size):
        yield order[left:left + batch_size]


def flat_batch(
    queries: np.ndarray, candidate_matrix: np.ndarray, candidate_mask: np.ndarray,
    molecule_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    chosen = candidate_matrix[queries]
    mask = candidate_mask[queries]
    flat = chosen[mask]
    return flat, mask, chosen.shape


def model_scores(
    model: FrozenEmbeddingReadout,
    queries: np.ndarray,
    candidate_matrix: np.ndarray,
    candidate_mask: np.ndarray,
    scalar: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    molecule_query: np.ndarray,
    graph: CandidateGraph,
    reference_rows: np.ndarray,
    banks: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, mask_np, shape = flat_batch(queries, candidate_matrix, candidate_mask, molecule_query)
    scalar_tensor = torch.as_tensor((scalar[flat] - mean) / scale, device=device)
    query_rows = torch.as_tensor(graph.query_row[molecule_query[flat]], device=device)
    references = torch.as_tensor(reference_rows[flat], device=device)
    values = model(scalar_tensor, query_rows, references, banks)
    padded = torch.full(shape, -1e9, dtype=values.dtype, device=device)
    mask = torch.as_tensor(mask_np, device=device)
    padded[mask] = values
    return padded, torch.as_tensor(flat, device=device)


def fit_one(
    arm: str,
    settings: ReadoutSettings,
    train_queries: np.ndarray,
    train_matrix: np.ndarray,
    train_mask: np.ndarray,
    scalar: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    molecule_query: np.ndarray,
    graph: CandidateGraph,
    reference_rows: np.ndarray,
    banks: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> FrozenEmbeddingReadout:
    seed_all(seed)
    model = FrozenEmbeddingReadout(arm, settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(args.epochs):
        losses: list[float] = []
        for queries in make_batches(train_queries, args.batch_queries, rng):
            optimizer.zero_grad(set_to_none=True)
            scores, _ = model_scores(
                model, queries, train_matrix, train_mask, scalar, mean, scale,
                molecule_query, graph, reference_rows, banks, device,
            )
            target = torch.zeros(len(queries), dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(scores, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"arm": arm, "epoch": epoch + 1,
                          "mean_loss": float(np.mean(losses))}), flush=True)
    return model


def predict_queries(
    model: FrozenEmbeddingReadout,
    query_ids: np.ndarray,
    candidate_matrix: np.ndarray,
    candidate_mask: np.ndarray,
    scalar: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    molecule_query: np.ndarray,
    graph: CandidateGraph,
    reference_rows: np.ndarray,
    banks: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.full(len(graph.molecule_label), np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(query_ids), batch_size):
            queries = query_ids[left:left + batch_size]
            scores, flat = model_scores(
                model, queries, candidate_matrix, candidate_mask, scalar, mean,
                scale, molecule_query, graph, reference_rows, banks, device,
            )
            mask = torch.as_tensor(candidate_mask[queries], device=device)
            output[flat.cpu().numpy()] = scores[mask].float().cpu().numpy()
    return output


def cluster_ci(
    values: np.ndarray, formulas: np.ndarray, repeats: int, seed: int,
    hypotheses: int = 24,
) -> dict[str, float | int]:
    unique, inverse = np.unique(formulas.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        take = rng.integers(0, len(unique), len(unique))
        boot[index] = sums[take].sum() / counts[take].sum()
    tail = 0.05 / (2 * hypotheses)
    return {"delta": float(np.mean(values)), "delta_pp": float(100 * np.mean(values)),
            "ci_low_pp": float(100 * np.quantile(boot, tail)),
            "ci_high_pp": float(100 * np.quantile(boot, 1 - tail)),
            "formula_clusters": int(len(unique)),
            "familywise_hypotheses": hypotheses}


def comparison(
    candidate: np.ndarray, baseline: np.ndarray, query_ptr: np.ndarray,
    formulas: np.ndarray, repeats: int, seed: int,
) -> dict:
    c_rank, b_rank = strict_ranks(candidate, query_ptr), strict_ranks(baseline, query_ptr)
    c_ok, b_ok = c_rank == 1, b_rank == 1
    delta = c_ok.astype(float) - b_ok.astype(float)
    return {
        "corrected": int(np.sum(c_ok & ~b_ok)),
        "introduced": int(np.sum(~c_ok & b_ok)),
        "risk_net_lambda2": int(np.sum(c_ok & ~b_ok) - 2 * np.sum(~c_ok & b_ok)),
        "paired_formula_ci": cluster_ci(delta, formulas, repeats, seed),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("the complete embedding readout requires one allocated GPU")
    if args.epochs != 3 or args.maximum_negatives != 6:
        raise RuntimeError("registered one-shot training settings drifted")
    named = parse_named_paths(args.embedding)
    if args.noise_name not in named:
        raise RuntimeError(f"noise encoder {args.noise_name!r} is absent")
    if set(named) != set(REQUIRED_ENCODERS):
        missing = sorted(set(REQUIRED_ENCODERS) - set(named))
        unexpected = sorted(set(named) - set(REQUIRED_ENCODERS))
        raise RuntimeError(
            "registered unified ranker requires the exact four encoders; "
            f"missing={missing}, unexpected={unexpected}"
        )
    # Stable order is part of the saved-model contract, independent of CLI order.
    named = {name: named[name] for name in REQUIRED_ENCODERS}
    graph = CandidateGraph(args.graph)
    evidence = load_evidence(args.evidence, graph)
    evidence_path = args.evidence / "evidence.npz" if args.evidence.is_dir() else args.evidence
    chemical = load_chemical_evidence(args.chem_evidence, evidence_path, graph)
    with h5py.File(args.data, "r") as handle:
        n_rows = len(handle["spectrum"])
    device = torch.device("cuda:0")

    embeddings: dict[str, np.ndarray] = {}
    encoder_scores: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    embedding_paths = {name: [Path(path) for path in paths] for name, paths in named.items()}
    for name, paths in embedding_paths.items():
        embedding = load_complete_embeddings(paths, n_rows).astype(np.float32)
        norms = np.linalg.norm(embedding, axis=1)
        if np.max(np.abs(norms - 1.0)) > 2e-3:
            raise RuntimeError(f"{name} embeddings are not unit normalized")
        embeddings[name] = embedding
        if name == args.noise_name:
            edge = np.asarray(evidence["v1_cosine"], dtype=np.float32)
        else:
            edge = edge_cosines(graph, embedding, device)
        score, reference = molecule_max_and_reference(edge, graph)
        encoder_scores[name], references[name] = score, reference
        print(f"prepared {name}: {len(score):,} molecule scores", flush=True)

    scalar, feature_names = build_independent_scalar_features(
        evidence, graph, encoder_scores, args.noise_name, chemical,
    )
    encoder_names = tuple(named)
    reference_rows = np.stack([references[name] for name in encoder_names], axis=1)
    embedding_dimension = next(iter(embeddings.values())).shape[1]
    if any(value.shape[1] != embedding_dimension for value in embeddings.values()):
        raise RuntimeError("encoder embedding dimensions differ")
    banks = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in embeddings.items()
    }
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    full_matrix, full_mask = padded_candidate_matrix(graph.query_ptr)
    score_families = [
        np.maximum.reduceat(np.asarray(evidence[name], dtype=np.float64), graph.molecule_ptr[:-1])
        for name in ("weighted_entropy", "p2b_fused")
    ] + [encoder_scores[name] for name in encoder_names]
    train_matrix, train_mask = hard_competitor_matrix(
        graph.query_ptr, score_families, args.maximum_negatives,
    )
    query_folds = np.asarray([
        stable_fold(str(formula), FOLDS, FOLD_SEED) for formula in graph.query_formula
    ], dtype=np.int8)
    settings = ReadoutSettings(
        scalar_features=scalar.shape[1], encoder_names=encoder_names,
        embedding_dimension=embedding_dimension,
        projection_dimension=args.projection_dimension,
    )
    held_scores = {arm: np.full(len(graph.molecule_label), np.nan, dtype=np.float32) for arm in ARMS}
    fold_reports: list[dict] = []
    parameter_counts: dict[str, int] = {}
    train_queries = np.flatnonzero(query_folds != HELD_FOLD)
    held_queries = np.flatnonzero(query_folds == HELD_FOLD)
    train_molecules = np.flatnonzero(query_folds[molecule_query] != HELD_FOLD)
    held_molecule_mask = query_folds[molecule_query] == HELD_FOLD
    held_ptr = np.concatenate(([0], np.cumsum(np.diff(graph.query_ptr)[held_queries])))
    mean, scale = fit_scalar_normalization(scalar, feature_names, train_molecules)
    for arm_index, arm in enumerate(ARMS):
        model = fit_one(
            arm, settings, train_queries, train_matrix, train_mask, scalar,
            mean, scale, molecule_query, graph, reference_rows, banks, device,
            args, TRAIN_SEED + arm_index,
        )
        parameter_counts[arm] = model.trainable_parameters
        predicted = predict_queries(
            model, held_queries, full_matrix, full_mask, scalar, mean, scale,
            molecule_query, graph, reference_rows, banks, device,
            args.batch_queries,
        )
        held_scores[arm][held_molecule_mask] = predicted[held_molecule_mask]
        ranks = strict_ranks(predicted[held_molecule_mask], held_ptr)
        fold_reports.append({"fold": HELD_FOLD, "arm": arm,
                             "held_queries": int(len(held_queries)),
                             "metrics": rank_metrics(ranks, graph.query_has_near[held_queries])})
        print(json.dumps(fold_reports[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    if any(not np.all(np.isfinite(scores[held_molecule_mask])) for scores in held_scores.values()):
        raise RuntimeError("held-fold score coverage is incomplete")

    references_scores = {
        "weighted_spectral_entropy": score_families[0],
        "p2b_noise_v1_frozen": score_families[1],
        **encoder_scores,
    }
    metrics = {arm: rank_metrics(strict_ranks(scores[held_molecule_mask], held_ptr),
                                 graph.query_has_near[held_queries])
               for arm, scores in held_scores.items()}
    metrics.update({name: rank_metrics(strict_ranks(scores[held_molecule_mask], held_ptr),
                                       graph.query_has_near[held_queries])
                    for name, scores in references_scores.items()})
    comparisons = {
        f"{arm}_vs_{baseline}": comparison(
            held_scores[arm][held_molecule_mask], base_scores[held_molecule_mask],
            held_ptr, graph.query_formula[held_queries], args.bootstrap_resamples,
            TRAIN_SEED + arm_index * 20 + baseline_index,
        )
        for arm_index, arm in enumerate(ARMS)
        for baseline_index, (baseline, base_scores) in enumerate(references_scores.items())
    }
    for left_index, left in enumerate(ARMS):
        for right_index, right in enumerate(ARMS):
            if left_index <= right_index:
                continue
            comparisons[f"{left}_vs_{right}"] = comparison(
                held_scores[left][held_molecule_mask],
                held_scores[right][held_molecule_mask], held_ptr,
                graph.query_formula[held_queries], args.bootstrap_resamples,
                TRAIN_SEED + 400 + 10 * left_index + right_index,
            )
    registered_encoder_pairs = (
        ("noise_chemaware_fullmsg", "noise_v1"),
        ("chemaware_phasea", "chemaware_stage1"),
    )
    for pair_index, (left, right) in enumerate(registered_encoder_pairs):
        if left in encoder_scores and right in encoder_scores:
            comparisons[f"raw_{left}_vs_{right}"] = comparison(
                encoder_scores[left][held_molecule_mask],
                encoder_scores[right][held_molecule_mask], held_ptr,
                graph.query_formula[held_queries], args.bootstrap_resamples,
                TRAIN_SEED + 500 + pair_index,
            )
    for arm in ARMS:
        comparisons[f"{arm}_candidate_auc"] = {
            "micro_auroc": float(roc_auc_score(graph.molecule_label[held_molecule_mask],
                                                held_scores[arm][held_molecule_mask])),
            "micro_auprc": float(average_precision_score(graph.molecule_label[held_molecule_mask],
                                                          held_scores[arm][held_molecule_mask])),
        }
    winner = select_full_integration_arm(metrics)
    scalar_control = ARM_SCALAR
    full_vs_scalar_key = f"{winner}_vs_{scalar_control}"
    full_vs_scalar = comparisons[full_vs_scalar_key]
    full_delta = float(metrics[winner]["recall1"] - metrics[scalar_control]["recall1"])
    full_ci_low = float(full_vs_scalar["paired_formula_ci"]["ci_low_pp"])
    if full_delta > 0 and full_ci_low > 0:
        representation_verdict = "FULL_EMBEDDING_INCREMENT_SUPPORTED_ON_HELD_DEVELOPMENT"
    elif full_delta > 0:
        representation_verdict = "FULL_EMBEDDING_POINT_GAIN_NOT_CONFIRMED"
    else:
        representation_verdict = "FULL_EMBEDDING_INCREMENT_NOT_SUPPORTED"

    # One final refit of the selected readout; no GNPS or application input is read.
    all_molecules = np.arange(len(scalar))
    mean, scale = fit_scalar_normalization(scalar, feature_names, all_molecules)
    final_model = fit_one(
        winner, settings, np.arange(graph.n_queries), train_matrix, train_mask,
        scalar, mean, scale, molecule_query, graph, reference_rows, banks, device,
        args, TRAIN_SEED + 999,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez_compressed(
            staging / "held_candidate_scores.npz",
            **{name: values[held_molecule_mask] for name, values in held_scores.items()},
            query_index=held_queries, query_ptr=held_ptr,
            molecule_label=graph.molecule_label[held_molecule_mask],
            molecule_ik14=graph.molecule_ik14[held_molecule_mask],
        )
        torch.save({
            "arm": winner, "state_dict": final_model.state_dict(),
            "selection_class": "full_embedding_integration",
            "non_promotable_control": scalar_control,
            "representation_increment_verdict": representation_verdict,
            "settings": settings.__dict__, "scalar_mean": mean,
            "scalar_scale": scale, "scalar_feature_names": feature_names,
            "encoder_names": encoder_names,
        }, staging / "selected_readout.pt")
        report = {
            "status": "MASSSPECGYM_UNIFIED_CANDIDATE_RANKER_COMPLETE",
            "dataset": {"name": "MassSpecGym corrected 10ppm candidate graph",
                        "role": "development", "total_queries": graph.n_queries,
                        "train_queries": int(len(train_queries)),
                        "held_queries": int(len(held_queries)), "held_fold": HELD_FOLD,
                        "candidate_molecules": int(len(graph.molecule_label)),
                        "reference_edges": int(len(graph.pair_candidate_row)),
                        "split": "stable formula folds; encoder and head train folds 1-4, selection fold 0"},
            "selection": winner,
            "selection_class": "full_embedding_integration",
            "non_promotable_control": scalar_control,
            "representation_increment_verdict": representation_verdict,
            "selected_full_integration_vs_scalar_control": full_vs_scalar,
            "algorithm": (
                "single candidate-level ranker over WSE, Noise V1, frozen P2b, "
                "ChemAware V2 correct-vs-null evidence, and independent Noise/"
                "Stage-1/Phase-A/FullMSG embedding geometry"
            ),
            "arms": list(ARMS),
            "full_integration_arms": list(FULL_INTEGRATION_ARMS),
            "control_arms": list(CONTROL_ARMS),
            "encoder_names": list(encoder_names),
            "chemical_evidence": {
                "module": "ChemAware V2 continuous correct-vs-null residual",
                "available_candidates": int(np.asarray(chemical["chem_valid"]).sum()),
                "missing_semantics": "unavailable candidates contribute availability=0 and exact zero residual",
                "normalization": "training-only RMS without centering; missing zero remains zero",
            },
            "parameter_counts": parameter_counts,
            "training": {"epochs": args.epochs, "batch_queries": args.batch_queries,
                         "learning_rate": args.learning_rate,
                         "weight_decay": args.weight_decay,
                         "maximum_family_hard_negatives": args.maximum_negatives,
                         "selection_rule": (
                             "highest held-fold-0 Recall@1 then MRR among projected_cosine "
                             "and embedding_pair only; scalar_evidence is a non-promotable "
                             "capacity-matched attribution control"
                         )},
            "metrics": metrics, "comparisons": comparisons,
            "fold_reports": fold_reports,
            "gnps_access": "none", "application_response_access": "none",
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "data_sha256": sha256_file(args.data),
                "evidence_sha256": sha256_file(evidence_path),
                "chem_evidence_sha256": sha256_file(args.chem_evidence),
                "embedding_shards": {name: [sha256_file(path) for path in paths]
                                     for name, paths in embedding_paths.items()},
            },
            "claim_limit": (
                "MassSpecGym development selection. Folds 1-4 are not called OOF "
                "because the frozen encoders trained on them; GNPS remains fixed evaluation."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "selection": winner,
                      "representation_increment_verdict": representation_verdict,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

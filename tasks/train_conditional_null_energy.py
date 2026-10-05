#!/usr/bin/env python
"""Train formula-isolated conditional-null candidate energy models.

The model is intentionally small.  Its methodological content is the matched
counterfactual construction and the worst-null ranking objective, not network
capacity.  Four preregistered arms share one recipe: joint, joint without the
explicit interaction, spectral-only and chemistry-only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from conditional_null_energy_core import (
    FEATURE_NAMES,
    AdditiveBoundedResidualEnergy,
    AnchoredEvidenceEnergy,
    build_candidate_features,
    conditional_null_loss,
    deterministic_keyed_derangements,
    percentile_by_query,
    strict_top1,
)
from noise_final_core import sha256_file, stable_fold


FOLD_COUNT = 5
FOLD_SEED = 20261004
ARMS = {
    "joint": tuple(range(len(FEATURE_NAMES))),
    "no_interaction": tuple(range(len(FEATURE_NAMES))),
    "spectral_only": tuple(range(len(FEATURE_NAMES))),
    "chem_only": tuple(range(len(FEATURE_NAMES))),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--chem-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--residual-bound", type=float, default=0.25)
    parser.add_argument("--null-margin", type=float, default=0.02)
    parser.add_argument("--null-weight", type=float, default=0.5)
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def molecule_max(values: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(
        np.asarray(values, dtype=np.float64), np.asarray(molecule_ptr, dtype=np.int64)[:-1],
    )


def query_candidate_indices(query_ids: np.ndarray, query_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_ids = np.asarray(query_ids, dtype=np.int64)
    blocks = [
        np.arange(int(query_ptr[q]), int(query_ptr[q + 1]), dtype=np.int64)
        for q in query_ids
    ]
    indices = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
    lengths = np.asarray([len(block) for block in blocks], dtype=np.int64)
    packed_ptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    return indices, packed_ptr


def fit_scaler(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(features[indices], dtype=np.float64).mean(axis=0)
    scale = np.asarray(features[indices], dtype=np.float64).std(axis=0)
    # Chemical zero is a semantic anchor and must remain exactly zero after
    # preprocessing.  Use RMS scaling without centering for those columns.
    mean[3:] = 0.0
    scale[3:] = np.sqrt(np.mean(np.asarray(features[indices, 3:], dtype=np.float64) ** 2, axis=0))
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transform(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - mean) / scale).astype(np.float32)


def train_one(
    *,
    actual: np.ndarray,
    nulls: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    query_ptr: np.ndarray,
    query_ids: np.ndarray,
    feature_columns: tuple[int, ...],
    args: argparse.Namespace,
    seed_offset: int,
    arm: str,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, list[dict[str, float]]]:
    indices, packed_ptr = query_candidate_indices(query_ids, query_ptr)
    mean, scale = fit_scaler(actual[:, feature_columns], indices)
    actual_selected = transform(actual[:, feature_columns], mean, scale)
    null_selected = np.stack([
        transform(nulls[arm][:, feature_columns], mean, scale)
        for arm in range(nulls.shape[0])
    ])
    device = torch.device(args.device)
    torch.manual_seed(args.seed + seed_offset)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + seed_offset)
    if arm == "no_interaction":
        model = AdditiveBoundedResidualEnergy(
            len(feature_columns), spectral_feature_count=3,
            hidden=args.hidden, residual_bound=args.residual_bound,
        ).to(device)
    else:
        model = AnchoredEvidenceEnergy(
            len(feature_columns), mode=arm, spectral_feature_count=3,
            hidden=args.hidden, residual_bound=args.residual_bound,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    base_tensor = torch.as_tensor(base[indices], dtype=torch.float32, device=device)
    actual_tensor = torch.as_tensor(actual_selected[indices], dtype=torch.float32, device=device)
    null_tensor = torch.as_tensor(null_selected[:, indices], dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(labels[indices], dtype=torch.int8, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        breakdown = conditional_null_loss(
            model=model, base_score=base_tensor, actual_features=actual_tensor,
            null_features=null_tensor, labels=label_tensor, query_ptr=packed_ptr,
            null_margin=args.null_margin, null_weight=args.null_weight,
            trust_weight=args.trust_weight,
        )
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        row = {
            "epoch": epoch + 1,
            "total": float(breakdown.total.detach().cpu()),
            "listwise": float(breakdown.listwise.detach().cpu()),
            "worst_null": float(breakdown.worst_null.detach().cpu()),
            "trust": float(breakdown.trust.detach().cpu()),
        }
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(row), flush=True)
    return model, mean, scale, history


def predict(
    model: torch.nn.Module,
    base: np.ndarray,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    columns: tuple[int, ...],
    indices: np.ndarray,
    device: str,
) -> np.ndarray:
    selected = transform(features[:, columns], mean, scale)
    with torch.no_grad():
        score = model(
            torch.as_tensor(base[indices], dtype=torch.float32, device=device),
            torch.as_tensor(selected[indices], dtype=torch.float32, device=device),
        )
    return score.cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.hidden < 1 or args.learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evidence_file = args.evidence / "evidence.npz"
    report = json.loads((args.evidence / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "NOISE_MSG_PAIR_EVIDENCE_COMPLETE":
        raise RuntimeError("pair evidence is not frozen")
    with np.load(evidence_file, allow_pickle=False) as body:
        evidence = {name: np.asarray(body[name]) for name in body.files}
    with np.load(args.chem_evidence, allow_pickle=False) as body:
        chem = {name: np.asarray(body[name]) for name in body.files}
    if str(chem["evidence_sha256"].item()) != sha256_file(evidence_file):
        raise RuntimeError("ChemAware candidate utilities are row-misaligned")

    query_ptr = np.asarray(evidence["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(evidence["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(evidence["molecule_label"], dtype=np.int8)
    formulas = np.asarray(evidence["query_formula"], dtype=str)
    molecule_count = int(query_ptr[-1])
    required_chem = (
        "candidate_utility_correct", "candidate_utility_zero",
        "candidate_utility_reversed", "candidate_utility_rotated",
        "candidate_key", "candidate_reference_count",
    )
    if any(np.asarray(chem[name]).shape != (molecule_count,) for name in required_chem):
        raise RuntimeError("ChemAware candidate utility shape drifted")

    v1 = molecule_max(evidence["v1_cosine"], molecule_ptr)
    p2b = molecule_max(evidence["p2b_fused"], molecule_ptr)
    neutral = molecule_max(evidence["neutral_loss_sqrt_cosine"], molecule_ptr)
    base = percentile_by_query(v1, query_ptr).astype(np.float32)
    chem_correct = np.asarray(chem["candidate_utility_correct"], dtype=np.float64)
    chem_null = np.stack([
        np.asarray(chem["candidate_utility_zero"], dtype=np.float64),
        np.asarray(chem["candidate_utility_reversed"], dtype=np.float64),
        np.asarray(chem["candidate_utility_rotated"], dtype=np.float64),
    ])
    candidate_key = np.asarray(chem["candidate_key"], dtype=str)
    query_key = np.asarray(chem["query_key"], dtype=str)
    if candidate_key.shape != (molecule_count,) or query_key.shape != (len(formulas),):
        raise RuntimeError("candidate/query key shape drifted")
    if np.asarray(chem["query_candidate_count"]).shape != (len(formulas),):
        raise RuntimeError("query candidate-count shape drifted")
    if np.asarray(chem["query_has_near"]).shape != (len(formulas),):
        raise RuntimeError("near-structure query shape drifted")
    actual = build_candidate_features(
        spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
        chem_primary=chem_correct, chem_references=chem_null,
        query_ptr=query_ptr,
    )
    null_features = []
    spectral_derangements = deterministic_keyed_derangements(
        p2b, candidate_key, query_key, query_ptr,
    )
    for spectral_null in spectral_derangements:
        null_features.append(build_candidate_features(
            spectral_primary=spectral_null, v1_score=v1, neutral_loss=neutral,
            chem_primary=chem_correct, chem_references=chem_null,
            query_ptr=query_ptr,
        ))
    for arm in range(3):
        null_features.append(build_candidate_features(
            spectral_primary=p2b, v1_score=v1, neutral_loss=neutral,
            chem_primary=chem_null[arm], chem_references=chem_null,
            query_ptr=query_ptr,
        ))
    for arm in range(3):
        null_features.append(build_candidate_features(
            spectral_primary=spectral_derangements[arm], v1_score=v1, neutral_loss=neutral,
            chem_primary=chem_null[arm], chem_references=chem_null,
            query_ptr=query_ptr,
        ))
    nulls = np.stack(null_features).astype(np.float32)
    columns = ARMS[args.arm]
    folds = np.asarray([stable_fold(value, FOLD_COUNT, FOLD_SEED) for value in formulas], dtype=np.int8)
    oof_score = np.full(molecule_count, np.nan, dtype=np.float32)
    fold_reports = []
    histories = {}
    for fold in range(FOLD_COUNT):
        train_queries = np.flatnonzero(folds != fold)
        held_queries = np.flatnonzero(folds == fold)
        model, mean, scale, history = train_one(
            actual=actual, nulls=nulls, base=base, labels=labels,
            query_ptr=query_ptr, query_ids=train_queries, feature_columns=columns,
            args=args, seed_offset=fold,
            arm=args.arm,
        )
        held_indices, held_ptr = query_candidate_indices(held_queries, query_ptr)
        oof_score[held_indices] = predict(
            model, base, actual, mean, scale, columns, held_indices, args.device,
        )
        held_correct = strict_top1(oof_score[held_indices], labels[held_indices], held_ptr)
        base_correct = strict_top1(base[held_indices], labels[held_indices], held_ptr)
        fold_reports.append({
            "fold": fold,
            "held_queries": int(len(held_queries)),
            "baseline_top1": float(base_correct.mean()),
            "candidate_energy_top1": float(held_correct.mean()),
            "delta_pp": 100.0 * float(held_correct.mean() - base_correct.mean()),
            "corrected": int(np.sum((base_correct == 0) & (held_correct == 1))),
            "introduced": int(np.sum((base_correct == 1) & (held_correct == 0))),
        })
        histories[str(fold)] = history
        print(json.dumps(fold_reports[-1]), flush=True)
    if not np.all(np.isfinite(oof_score)):
        raise RuntimeError("formula-OOF candidate score ledger is incomplete")

    baseline_correct = strict_top1(base, labels, query_ptr)
    oof_correct = strict_top1(oof_score, labels, query_ptr)
    all_queries = np.arange(len(formulas), dtype=np.int64)
    final_model, final_mean, final_scale, final_history = train_one(
        actual=actual, nulls=nulls, base=base, labels=labels,
        query_ptr=query_ptr, query_ids=all_queries, feature_columns=columns,
        args=args, seed_offset=100,
        arm=args.arm,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".conditional_null_{args.arm}.", dir=args.output.parent))
    try:
        torch.save({
            "state_dict": final_model.cpu().state_dict(),
            "feature_names": [FEATURE_NAMES[index] for index in columns],
            "feature_columns": list(columns),
            "mean": final_mean,
            "scale": final_scale,
            "hidden": args.hidden,
            "residual_bound": args.residual_bound,
            "model_kind": "additive_two_head" if args.arm == "no_interaction" else "anchored_evidence",
            "mode": args.arm,
            "spectral_feature_count": 3,
        }, staging / "model.pt")
        np.savez_compressed(
            staging / "oof_scores.npz",
            candidate_scores=oof_score,
            baseline_scores=base,
            labels=labels,
            query_ptr=query_ptr,
            query_formula=formulas,
            folds=folds,
            candidate_reference_count=np.asarray(chem["candidate_reference_count"], dtype=np.int32),
            query_candidate_count=np.asarray(chem["query_candidate_count"], dtype=np.int32),
            query_has_near=np.asarray(chem["query_has_near"], dtype=bool),
        )
        output_report = {
            "status": "CONDITIONAL_NULL_CANDIDATE_ENERGY_FROZEN",
            "arm": args.arm,
            "method": "single candidate energy; formula-OOF listwise conditional-null NCE",
            "model_kind": "additive_two_head" if args.arm == "no_interaction" else "anchored_evidence",
            "features": [FEATURE_NAMES[index] for index in columns],
            "null_arms": [
                "spectral_keyed_derangement_1", "spectral_keyed_derangement_2", "spectral_keyed_derangement_3",
                "chem_zero", "chem_reversed", "chem_candidate_rotated",
                "joint_null_1", "joint_null_2", "joint_null_3",
            ],
            "recipe": {
                "fold_count": FOLD_COUNT, "fold_seed": FOLD_SEED,
                "epochs": args.epochs, "hidden": args.hidden,
                "learning_rate": args.learning_rate,
                "residual_bound": args.residual_bound,
                "null_margin": args.null_margin, "null_weight": args.null_weight,
                "trust_weight": args.trust_weight, "seed": args.seed,
            },
            "formula_oof": {
                "queries": int(len(formulas)),
                "baseline_top1": float(baseline_correct.mean()),
                "candidate_energy_top1": float(oof_correct.mean()),
                "delta_pp": 100.0 * float(oof_correct.mean() - baseline_correct.mean()),
                "corrected": int(np.sum((baseline_correct == 0) & (oof_correct == 1))),
                "introduced": int(np.sum((baseline_correct == 1) & (oof_correct == 0))),
                "risk_net_lambda_2": int(np.sum((baseline_correct == 0) & (oof_correct == 1)))
                - 2 * int(np.sum((baseline_correct == 1) & (oof_correct == 0))),
                "folds": fold_reports,
            },
            "training_history": histories,
            "final_history": final_history,
            "provenance": {
                "evidence_sha256": sha256_file(evidence_file),
                "chem_evidence_sha256": sha256_file(args.chem_evidence),
            },
            "claim_limit": (
                "Second-stage formula-OOF development fit on a panel already consumed by upstream "
                "components. It may select one frozen method for a genuinely unconsumed external "
                "evaluation, but it cannot support an independent-generalization claim."
            ),
        }
        (staging / "report.json").write_text(json.dumps(output_report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(output_report["formula_oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()

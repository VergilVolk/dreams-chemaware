#!/usr/bin/env python
"""Two-arm WSE-anchored conditional residual development on consumed GNPS.

Arm A learns a bounded spectral residual from raw WSE, Noise V1, and one
P2b-on-Noise composite.  Arm B adds a conditional chemical residual defined
as f(spectral, chemical contrast) - f(spectral, zero contrast).  The chemical
contrast contains only continuous correct-minus-null V2 differences; the
thresholded V2 action and nuisance utility are never model inputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from conditional_null_energy_core import AnchoredEvidenceEnergy, conditional_null_loss
from grand_unified_evidence_core import rank_logit_evidence, strict_ranks
from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import (
    candidate_indices,
    clustered_ci,
    formula_fold,
    metric_summary,
    prepare,
    select_modules,
)


WSE = "weighted_spectral_entropy"
NOISE = "noise_v1"
P2B = "p2b_noise_v1_frozen"
OLD_MODULES = (NOISE, WSE, P2B, "chemaware_v2_action")
V2_ACTION = "chemaware_v2_action"
SPECTRAL_FEATURE_NAMES = (
    "wse_raw", "wse_gap_to_query_top", "wse_query_top1_top2_margin",
    "noise_v1_raw", "noise_v1_gap_to_query_top", "noise_v1_query_top1_top2_margin",
    "p2b_noise_composite_raw", "p2b_noise_composite_gap_to_query_top",
    "p2b_noise_composite_query_top1_top2_margin",
)
CHEMICAL_FEATURE_NAMES = (
    "v2_primary_minus_null_mean",
    "v2_primary_minus_null_min",
    "v2_primary_minus_null_max",
    "v2_primary_minus_null_std",
)
ARM_MODES = {
    "A_spectral_residual": "spectral_only",
    "B_joint_residual": "joint",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def query_score_features(values: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.empty((len(values), 3), dtype=np.float32)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        block = values[left:right]
        if len(block) < 2 or not np.all(np.isfinite(block)):
            raise ValueError("each query needs at least two finite candidate scores")
        ordered = np.sort(block)
        top = float(ordered[-1])
        margin = float(ordered[-1] - ordered[-2])
        output[left:right, 0] = block
        output[left:right, 1] = block - top
        output[left:right, 2] = margin
    return output


def summarize_primary_against_nulls(
    primary: np.ndarray,
    references: np.ndarray,
    availability: np.ndarray,
) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    availability = np.asarray(availability, dtype=bool)
    if references.ndim != 2 or references.shape[1] != len(primary):
        raise ValueError("chemical reference matrix must be [null, candidate]")
    differences = primary[None, :] - references
    features = np.column_stack((
        np.mean(differences, axis=0),
        np.min(differences, axis=0),
        np.max(differences, axis=0),
        np.std(differences, axis=0),
    )).astype(np.float32)
    features[~availability] = 0.0
    if not np.all(np.isfinite(features)):
        raise ValueError("non-finite chemical contrast")
    return features


def build_chemical_features(chem: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    availability = np.asarray(chem["availability"], dtype=bool)
    correct = np.asarray(chem["utility_correct"], dtype=np.float32)
    zero = np.asarray(chem["utility_zero"], dtype=np.float32)
    reversed_ = np.asarray(chem["utility_reversed"], dtype=np.float32)
    rotated = np.asarray(chem["utility_candidate_mismatch"], dtype=np.float32)
    if any(value.shape != availability.shape for value in (correct, zero, reversed_, rotated)):
        raise ValueError("chemical utility axes differ")
    actual = summarize_primary_against_nulls(
        correct, np.stack((zero, reversed_, rotated)), availability,
    )
    semantic_zero = np.zeros_like(actual)
    reversed_control = summarize_primary_against_nulls(
        reversed_, np.stack((zero, correct, rotated)), availability,
    )
    rotated_control = summarize_primary_against_nulls(
        rotated, np.stack((zero, reversed_, correct)), availability,
    )
    nulls = np.stack((semantic_zero, reversed_control, rotated_control))
    if np.any(nulls[0] != 0):
        raise RuntimeError("semantic chemical zero drifted")
    return actual, nulls, availability


def load_chem(path: Path, bundle: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        chem = {name: np.asarray(source[name]) for name in source.files}
    if str(np.asarray(chem["schema"]).item()) != "gnps_chemaware_continuous_null_evidence_v1":
        raise RuntimeError(f"unsupported chemical evidence: {path}")
    checks = {
        "query_ptr": np.array_equal(chem["query_ptr"], bundle["query_ptr"]),
        "query_ids": np.array_equal(chem["query_ids"].astype("U"), bundle["query_ids"].astype("U")),
        "candidate_ids": np.array_equal(
            chem["candidate_ids"].astype("U"), bundle["candidate_ids"].astype("U")
        ),
        "formula": np.array_equal(chem["query_formula"].astype("U"), bundle["group_ids"].astype("U")),
        "near": np.array_equal(chem["near_query"].astype(bool), bundle["near_query"].astype(bool)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"chemical evidence candidate axis drifted: {failed}")
    return chem


def build_panel(bundle: dict, chem_path: Path) -> dict:
    names = tuple(map(str, bundle["module_names"]))
    positions = {name: names.index(name) for name in (WSE, NOISE, P2B)}
    availability = np.asarray(bundle["availability"], dtype=np.float32)
    for name, position in positions.items():
        if not np.all(availability[position] == 1):
            raise RuntimeError(f"{name} must be complete; missingness cannot be fabricated")
    scores = np.asarray(bundle["scores"], dtype=np.float32)
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    spectral = np.column_stack(tuple(
        query_score_features(scores[positions[name]], ptr)
        for name in (WSE, NOISE, P2B)
    )).astype(np.float32)
    chem = load_chem(chem_path, bundle)
    chemical, chemical_nulls, chemical_available = build_chemical_features(chem)
    actual = np.column_stack((spectral, chemical)).astype(np.float32)
    nulls = np.stack([
        np.column_stack((spectral, chemical_nulls[index])).astype(np.float32)
        for index in range(len(chemical_nulls))
    ])
    return {
        "base": scores[positions[WSE]].copy(),
        # Keep the frozen component axes for descriptive diagnostics only.  They
        # are never fed back into the fitted A/B models or used for selection.
        "raw_module_scores": {
            name: scores[names.index(name)].copy()
            for name in (WSE, NOISE, P2B, V2_ACTION) if name in names
        },
        "raw_module_available": {
            name: availability[names.index(name)].astype(bool, copy=True)
            for name in (WSE, NOISE, P2B, V2_ACTION) if name in names
        },
        "features": actual,
        "null_features": nulls,
        "chemical_available": chemical_available,
        "query_ptr": ptr,
        "labels": np.asarray(bundle["labels"], dtype=np.int8),
        "formulas": np.asarray(bundle["group_ids"]).astype("U"),
        "near": np.asarray(bundle["near_query"], dtype=bool),
        "query_ids": np.asarray(bundle["query_ids"]).astype("U"),
        "candidate_ids": np.asarray(bundle["candidate_ids"]).astype("U"),
    }


def fit_scaler(
    features: np.ndarray,
    train_indices: np.ndarray,
    chemical_available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spectral_count = len(SPECTRAL_FEATURE_NAMES)
    train = np.asarray(features[train_indices], dtype=np.float64)
    mean = np.zeros(features.shape[1], dtype=np.float64)
    scale = np.ones(features.shape[1], dtype=np.float64)
    mean[:spectral_count] = np.mean(train[:, :spectral_count], axis=0)
    scale[:spectral_count] = np.std(train[:, :spectral_count], axis=0)
    chem_rows = train[np.asarray(chemical_available[train_indices], dtype=bool), spectral_count:]
    if len(chem_rows):
        scale[spectral_count:] = np.sqrt(np.mean(np.square(chem_rows), axis=0))
    scale[scale < 1e-7] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transform_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = ((np.asarray(features, dtype=np.float32) - mean) / scale).astype(np.float32)
    if np.any(mean[len(SPECTRAL_FEATURE_NAMES):] != 0):
        raise RuntimeError("chemical semantic zero was centered away from zero")
    return transformed


def training_residual_bound(base: np.ndarray, ptr: np.ndarray, queries: np.ndarray) -> tuple[float, dict]:
    gaps = []
    for query in np.asarray(queries, dtype=np.int64):
        block = np.sort(np.asarray(base[ptr[query]:ptr[query + 1]], dtype=np.float64))
        gap = float(block[-1] - block[-2])
        if np.isfinite(gap) and gap > 0:
            gaps.append(gap)
    if not gaps:
        raise RuntimeError("training queries have no positive WSE competition margins")
    gaps = np.asarray(gaps, dtype=np.float64)
    bound = float(np.quantile(gaps, 0.75))
    if not np.isfinite(bound) or bound <= 0:
        raise RuntimeError("training-derived residual bound is invalid")
    return bound, {
        "rule": "training-fold Q75 of positive raw-WSE top1-top2 margins",
        "positive_margin_queries": int(len(gaps)),
        "q25": float(np.quantile(gaps, 0.25)),
        "median": float(np.median(gaps)),
        "q75_residual_bound": bound,
        "q90": float(np.quantile(gaps, 0.90)),
        "architecturally_crossable_fraction": float(np.mean(gaps <= bound)),
    }


def fit_arm(
    panel: dict,
    train_queries: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[AnchoredEvidenceEnergy, np.ndarray, np.ndarray, float, list[dict], dict]:
    index, packed_ptr = candidate_indices(panel["query_ptr"], train_queries)
    mean, scale = fit_scaler(panel["features"], index, panel["chemical_available"])
    actual = transform_features(panel["features"], mean, scale)
    nulls = np.stack([
        transform_features(block, mean, scale) for block in panel["null_features"]
    ])
    if np.any(nulls[0, :, len(SPECTRAL_FEATURE_NAMES):] != 0):
        raise RuntimeError("standardized chemical zero is not exact")
    residual_bound, capacity = training_residual_bound(
        panel["base"], panel["query_ptr"], train_queries,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = AnchoredEvidenceEnergy(
        panel["features"].shape[1], mode=mode,
        spectral_feature_count=len(SPECTRAL_FEATURE_NAMES),
        hidden=args.hidden, residual_bound=residual_bound,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    base = torch.as_tensor(panel["base"][index], dtype=torch.float32, device=device)
    actual_tensor = torch.as_tensor(actual[index], dtype=torch.float32, device=device)
    null_tensor = torch.as_tensor(nulls[:, index], dtype=torch.float32, device=device)
    labels = torch.as_tensor(panel["labels"][index], dtype=torch.int8, device=device)
    history = []
    null_margin = 0.08 * residual_bound  # preserves the established 0.02/0.25 ratio
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_null_loss(
            model=model, base_score=base, actual_features=actual_tensor,
            null_features=null_tensor, labels=labels, query_ptr=packed_ptr,
            null_margin=null_margin, null_weight=args.null_weight,
            trust_weight=args.trust_weight,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append({
            "epoch": epoch + 1,
            "total": float(loss.total.detach().cpu()),
            "listwise": float(loss.listwise.detach().cpu()),
            "worst_null": float(loss.worst_null.detach().cpu()),
            "trust": float(loss.trust.detach().cpu()),
        })
    capacity["null_margin"] = null_margin
    return model, mean, scale, residual_bound, history, capacity


@torch.inference_mode()
def predict_arm(
    model: AnchoredEvidenceEnergy,
    panel: dict,
    queries: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    index, _ = candidate_indices(panel["query_ptr"], queries)
    actual = transform_features(panel["features"], mean, scale)
    nulls = np.stack([
        transform_features(block, mean, scale) for block in panel["null_features"]
    ])
    base = torch.as_tensor(panel["base"][index], dtype=torch.float32, device=device)
    score = model(
        base,
        torch.as_tensor(actual[index], dtype=torch.float32, device=device),
    ).cpu().numpy().astype(np.float32)
    replay = np.stack([
        model(
            base,
            torch.as_tensor(block[index], dtype=torch.float32, device=device),
        ).cpu().numpy().astype(np.float32)
        for block in nulls
    ])
    return score, replay


def reconstruct_old_scores(bundle: dict, report: dict, panel_name: str) -> tuple[np.ndarray, str]:
    matching = [
        (name, arm) for name, arm in report["arms"].items()
        if tuple(arm.get("modules", ())) == OLD_MODULES
    ]
    if len(matching) != 1:
        raise RuntimeError(f"cannot uniquely locate old four-module arm: {[x[0] for x in matching]}")
    arm_name, arm = matching[0]
    selected = prepare(select_modules(bundle, OLD_MODULES))
    base_index = OLD_MODULES.index(WSE)
    output = np.full(len(selected["labels"]), np.nan, dtype=np.float32)
    seed = int(report["optimization"]["seed"])
    folds = int(report["formula_crossfit_folds"])
    formula_folds = np.asarray([formula_fold(value, folds, seed) for value in selected["formulas"]])
    for fold_report in arm["folds"]:
        fold = int(fold_report["fold"])
        queries = np.flatnonzero(formula_folds == fold)
        index, _ = candidate_indices(selected["query_ptr"], queries)
        base = selected["evidence"][base_index, index]
        score = base.astype(np.float64, copy=True)
        coefficients = fold_report["residual_coefficients"]
        for module_index, module in enumerate(OLD_MODULES):
            alpha = float(coefficients[module])
            score += alpha * (
                selected["evidence"][module_index, index] - base
            ) * selected["availability"][module_index, index]
        output[index] = score.astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("old four-module OOF reconstruction is incomplete")
    expected = arm["panels"][panel_name]["candidate"]
    observed, _ = metric_summary(output, selected)
    for metric in ("recall@1", "mrr", "macro_query_auroc", "micro_candidate_auroc"):
        if abs(float(observed[metric]) - float(expected[metric])) > 2e-6:
            raise RuntimeError(f"old OOF reconstruction mismatch for {panel_name}/{metric}")
    return output, arm_name


def evaluation_scores(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    calibrated, _ = rank_logit_evidence(
        np.asarray(scores, dtype=np.float32)[None, :], ptr,
        np.ones((1, len(scores)), dtype=np.float32),
    )
    return calibrated[0]


def metric_input(panel: dict) -> dict:
    return {
        "query_ptr": panel["query_ptr"], "labels": panel["labels"],
        "formulas": panel["formulas"], "near": panel["near"],
    }


def paired_comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    panel: dict,
    seed: int,
    resamples: int,
) -> dict:
    ptr, labels = panel["query_ptr"], panel["labels"]
    candidate_rank = strict_ranks(candidate, ptr, labels)
    baseline_rank = strict_ranks(baseline, ptr, labels)
    corrected = (candidate_rank == 1) & (baseline_rank != 1)
    introduced = (candidate_rank != 1) & (baseline_rank == 1)
    delta = (candidate_rank == 1).astype(float) - (baseline_rank == 1).astype(float)
    near = panel["near"]
    result = {
        "delta_recall@1": float(np.mean(delta)),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "risk_net_corrected_minus_2introduced": int(corrected.sum() - 2 * introduced.sum()),
        "formula_cluster_bootstrap_95ci": clustered_ci(
            delta, panel["formulas"], seed, resamples,
        ),
    }
    if np.any(near):
        result["near"] = {
            "queries": int(np.sum(near)),
            "delta_recall@1": float(np.mean(delta[near])),
            "corrected": int(corrected[near].sum()),
            "introduced": int(introduced[near].sum()),
            "risk_net_corrected_minus_2introduced": int(
                corrected[near].sum() - 2 * introduced[near].sum()
            ),
            "formula_cluster_bootstrap_95ci": clustered_ci(
                delta[near], panel["formulas"][near], seed + 10000, resamples,
            ),
        }
    return result


def arm_beats(comparisons: dict, panels: tuple[str, ...]) -> bool:
    """Primary retrieval-selection rule.

    ``corrected - 2 * introduced`` is retained as a separately reported,
    application-specific conservative utility.  It is not a standard ranking
    endpoint and must not silently veto a model that has a strictly positive
    paired Recall@1 effect, positive formula-cluster interval, and no near-set
    Recall@1 regression on both development panels.
    """
    return all(
        comparisons[panel]["delta_recall@1"] > 0
        and comparisons[panel]["formula_cluster_bootstrap_95ci"][0] > 0
        and comparisons[panel].get("near", {}).get("delta_recall@1", 0.0) >= 0
        for panel in panels
    )


def lambda2_risk_beats(comparisons: dict, panels: tuple[str, ...]) -> bool:
    """Legacy conservative-use utility, reported independently of selection."""
    return all(
        comparisons[panel]["risk_net_corrected_minus_2introduced"] > 0
        and comparisons[panel].get("near", {}).get(
            "risk_net_corrected_minus_2introduced", 0,
        ) >= 0
        for panel in panels
    )


def metric_profile_noninferior(
    report_panels: dict,
    candidate: str,
    baseline: str,
    panels: tuple[str, ...],
) -> bool:
    """Require the principal continuous ranking metrics not to regress."""
    keys = ("mrr", "micro_candidate_auroc", "micro_candidate_auprc")
    return all(
        all(
            report_panels[panel]["methods"][candidate][key]
            >= report_panels[panel]["methods"][baseline][key]
            for key in keys
        )
        and report_panels[panel]["methods"][candidate]["near"]["recall@1"]
        >= report_panels[panel]["methods"][baseline]["near"]["recall@1"]
        for panel in panels
    )


def unique_top_candidate(scores: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    """Deployment-visible unique winners; -1 represents a top tie.

    This deliberately does not look at labels.  It is used only to form the
    observable spectral-agreement stratum, never as a router.
    """
    scores = np.asarray(scores, dtype=np.float64)
    result = np.full(len(query_ptr) - 1, -1, dtype=np.int64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        block = scores[left:right]
        winners = np.flatnonzero(block == np.max(block))
        if len(winners) == 1:
            result[query] = left + int(winners[0])
    return result


def query_true_gap(base: np.ndarray, query_ptr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Raw-WSE deficit of the true candidate relative to the WSE leader."""
    gap = np.empty(len(query_ptr) - 1, dtype=np.float64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        block_labels = labels[left:right]
        if int(block_labels.sum()) != 1:
            raise ValueError("every query needs one positive candidate")
        true_score = float(base[left:right][block_labels.astype(bool)][0])
        gap[query] = float(np.max(base[left:right]) - true_score)
    return gap


def query_complete_availability(availability: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    availability = np.asarray(availability, dtype=bool)
    return np.asarray([
        bool(np.all(availability[int(left):int(right)]))
        for left, right in zip(query_ptr[:-1], query_ptr[1:])
    ])


def quantile_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "q90": None, "max": None}
    return {
        "n": int(len(values)), "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)), "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)), "q90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def rank_metrics_for_queries(ranks: np.ndarray, query_mask: np.ndarray) -> dict:
    selected = np.asarray(ranks, dtype=np.int64)[np.asarray(query_mask, dtype=bool)]
    if not len(selected):
        return {"queries": 0, "recall@1": None, "recall@3": None, "recall@5": None,
                "recall@10": None, "recall@20": None, "mrr": None}
    return {
        "queries": int(len(selected)),
        **{f"recall@{k}": float(np.mean(selected <= k)) for k in (1, 3, 5, 10, 20)},
        "mrr": float(np.mean(1.0 / selected)),
    }


def residual_diagnostics(
    panel: dict,
    scores: dict[str, np.ndarray],
    b_bound_by_query: np.ndarray,
) -> dict:
    """Describe capacity and exchanges without changing the fitted system.

    For B, each candidate residual lies in [-rho, rho].  Therefore a true
    candidate whose raw-WSE deficit is >= 2*rho cannot become a *strict* Top-1
    candidate under B.  The equality case is included because strict ties are
    not counted as correct throughout this benchmark.
    """
    ptr, labels = panel["query_ptr"], panel["labels"]
    ranks = {
        name: strict_ranks(value, ptr, labels)
        for name, value in scores.items()
        if name in ("wse", "old_four_module", "A", "B")
    }
    wse_error = ranks["wse"] != 1
    b_corrected = (ranks["B"] == 1) & wse_error
    b_introduced = (ranks["B"] != 1) & (ranks["wse"] == 1)
    true_gap = query_true_gap(panel["base"], ptr, labels)
    b_bound_by_query = np.asarray(b_bound_by_query, dtype=np.float64)
    if b_bound_by_query.shape != true_gap.shape or np.any(~np.isfinite(b_bound_by_query)):
        raise ValueError("B residual-bound axis is incomplete")
    unreachable = true_gap >= (2.0 * b_bound_by_query - 1e-12)

    direct_corrections = {}
    direct_any = np.zeros(len(wse_error), dtype=bool)
    for name in (NOISE, P2B, V2_ACTION):
        if name not in panel["raw_module_scores"]:
            direct_corrections[name] = {"status": "NOT_PRESENT", "complete_queries": 0, "corrected_wse_errors": 0}
            continue
        complete = query_complete_availability(panel["raw_module_available"][name], ptr)
        module_ranks = strict_ranks(panel["raw_module_scores"][name], ptr, labels)
        correction = wse_error & complete & (module_ranks == 1)
        direct_any |= correction
        direct_corrections[name] = {
            "status": "COMPLETE_QUERY_ONLY",
            "complete_queries": int(complete.sum()),
            "corrected_wse_errors": int(correction.sum()),
            "wse_errors_with_complete_module": int((wse_error & complete).sum()),
        }
    old_correction = wse_error & (ranks["old_four_module"] == 1)

    wse_leader = unique_top_candidate(panel["raw_module_scores"][WSE], ptr)
    noise_leader = unique_top_candidate(panel["raw_module_scores"][NOISE], ptr)
    p2b_leader = unique_top_candidate(panel["raw_module_scores"][P2B], ptr)
    spectral_agreement = (
        (wse_leader >= 0) & (wse_leader == noise_leader) & (wse_leader == p2b_leader)
    )
    chemistry_available = np.asarray([
        bool(np.any(panel["chemical_available"][int(left):int(right)]))
        for left, right in zip(ptr[:-1], ptr[1:])
    ])
    chemistry_complete = query_complete_availability(panel["chemical_available"], ptr)
    strata = {
        "spectral_agreement_unique_top1": spectral_agreement,
        "spectral_conflict_or_top_tie": ~spectral_agreement,
        "chemistry_unavailable_all_candidates": ~chemistry_available,
        "chemistry_partially_available": chemistry_available & ~chemistry_complete,
        "chemistry_available_all_candidates": chemistry_complete,
    }
    stratum_report = {}
    for name, mask in strata.items():
        stratum_report[name] = {
            "observable_pretruth_only": True,
            "methods": {method: rank_metrics_for_queries(rank, mask) for method, rank in ranks.items()},
            "B_vs_wse": {
                "corrected": int(np.sum(b_corrected & mask)),
                "introduced": int(np.sum(b_introduced & mask)),
                "delta_recall@1": float(np.mean(
                    (ranks["B"] == 1)[mask].astype(float)
                    - (ranks["wse"] == 1)[mask].astype(float)
                )) if np.any(mask) else None,
            },
        }
    return {
        "status": "DESCRIPTIVE_NO_SELECTION_OR_RETRAINING",
        "residual_capacity": {
            "theorem": "With B residuals bounded in [-rho,+rho], a raw-WSE true-candidate deficit >= 2*rho cannot yield strict Top-1.",
            "wse_errors": int(wse_error.sum()),
            "wse_errors_structurally_unreachable_by_B_bound": int(np.sum(wse_error & unreachable)),
            "wse_errors_structurally_crossable_by_B_bound": int(np.sum(wse_error & ~unreachable)),
            "B_corrected": int(b_corrected.sum()),
            "B_corrected_above_or_at_bound_should_be_zero": int(np.sum(b_corrected & unreachable)),
            "B_introduced": int(b_introduced.sum()),
            "B_residual_bound_raw_wse_units": quantile_summary(b_bound_by_query),
            "raw_wse_true_candidate_gap": {
                "all_wse_errors": quantile_summary(true_gap[wse_error]),
                "B_corrected": quantile_summary(true_gap[b_corrected]),
                "B_introduced": quantile_summary(true_gap[b_introduced]),
            },
        },
        "existing_component_correctability": {
            "definition": "WSE errors made strict Top-1 by a frozen component only on queries where that component is complete; descriptive, not a fusion rule.",
            "per_component": direct_corrections,
            "corrected_by_any_noise_p2b_or_v2_action": int(direct_any.sum()),
            "of_any_component_correctable_unreachable_by_B_bound": int(np.sum(direct_any & unreachable)),
            "old_four_module_corrected_wse_errors": int(old_correction.sum()),
        },
        "observable_strata": stratum_report,
    }


def write_query_ledger(path: Path, panels: dict, scores: dict) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "panel", "query_index", "query_id", "formula", "near",
            "wse_rank", "old_four_module_rank", "A_rank", "B_rank",
            "A_corrected_vs_wse", "A_introduced_vs_wse",
            "B_corrected_vs_wse", "B_introduced_vs_wse",
            "B_corrected_vs_A", "B_introduced_vs_A",
        ))
        for panel_name, panel in panels.items():
            ranks = {
                name: strict_ranks(value, panel["query_ptr"], panel["labels"])
                for name, value in scores[panel_name].items()
                if name in ("wse", "old_four_module", "A", "B")
            }
            for query in range(len(panel["query_ids"])):
                writer.writerow((
                    panel_name, query, panel["query_ids"][query], panel["formulas"][query],
                    int(panel["near"][query]), ranks["wse"][query],
                    ranks["old_four_module"][query], ranks["A"][query], ranks["B"][query],
                    int(ranks["A"][query] == 1 and ranks["wse"][query] != 1),
                    int(ranks["A"][query] != 1 and ranks["wse"][query] == 1),
                    int(ranks["B"][query] == 1 and ranks["wse"][query] != 1),
                    int(ranks["B"][query] != 1 and ranks["wse"][query] == 1),
                    int(ranks["B"][query] == 1 and ranks["A"][query] != 1),
                    int(ranks["B"][query] != 1 and ranks["A"][query] == 1),
                ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--identity-chem", type=Path, required=True)
    parser.add_argument("--formula-chem", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--null-weight", type=float, default=0.5)
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.hidden < 1 or args.learning_rate <= 0:
        raise ValueError("invalid fixed training recipe")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    bundles = {
        "identity": load_bundle(args.identity_bundle),
        "formula": load_bundle(args.formula_bundle),
    }
    panels = {
        "identity": build_panel(bundles["identity"], args.identity_chem),
        "formula": build_panel(bundles["formula"], args.formula_chem),
    }
    old_report = json.loads(args.old_report.read_text(encoding="utf-8"))
    old_scores = {}
    old_arm_names = set()
    for name in panels:
        old_scores[name], old_name = reconstruct_old_scores(bundles[name], old_report, name)
        old_arm_names.add(old_name)
    if len(old_arm_names) != 1:
        raise RuntimeError("old comparator arm differs between panels")

    outputs = {
        panel: {
            "wse": body["base"].copy(),
            "old_four_module": old_scores[panel],
            "A": np.full(len(body["base"]), np.nan, dtype=np.float32),
            "B": np.full(len(body["base"]), np.nan, dtype=np.float32),
            "B_zero_replay": np.full(len(body["base"]), np.nan, dtype=np.float32),
            "B_reversed_replay": np.full(len(body["base"]), np.nan, dtype=np.float32),
            "B_candidate_mismatch_replay": np.full(len(body["base"]), np.nan, dtype=np.float32),
        }
        for panel, body in panels.items()
    }
    # Bound values are assigned only from the model trained without the held
    # formula fold.  They are recorded for a post-fit capacity diagnosis, not
    # revisited to select a larger bound.
    b_bound_by_query = {
        panel: np.full(len(body["query_ids"]), np.nan, dtype=np.float64)
        for panel, body in panels.items()
    }
    identity_folds = np.asarray([
        formula_fold(value, args.folds, args.seed) for value in panels["identity"]["formulas"]
    ])
    target_folds = {
        name: np.asarray([formula_fold(value, args.folds, args.seed) for value in body["formulas"]])
        for name, body in panels.items()
    }
    fold_reports = []
    for fold in range(args.folds):
        train_queries = np.flatnonzero(identity_folds != fold)
        fold_report = {"fold": fold, "train_queries": int(len(train_queries)), "arms": {}}
        for arm_offset, (arm_name, mode) in enumerate(ARM_MODES.items()):
            model, mean, scale, bound, history, capacity = fit_arm(
                panels["identity"], train_queries, mode, args,
                args.seed + fold,  # identical spectral initialization in A and B
            )
            arm_record = {
                "mode": mode,
                "residual_bound_raw_wse_units": bound,
                "scaler_fit_queries": int(len(train_queries)),
                "scaler_chemical_mean_exact_zero": bool(
                    np.all(mean[len(SPECTRAL_FEATURE_NAMES):] == 0)
                ),
                "capacity": capacity,
                "final_loss": history[-1],
                "prediction_queries": {},
            }
            for panel_name, panel in panels.items():
                queries = np.flatnonzero(target_folds[panel_name] == fold)
                index, _ = candidate_indices(panel["query_ptr"], queries)
                score, replay = predict_arm(model, panel, queries, mean, scale, args.device)
                outputs[panel_name]["A" if arm_name.startswith("A_") else "B"][index] = score
                if arm_name.startswith("B_"):
                    outputs[panel_name]["B_zero_replay"][index] = replay[0]
                    outputs[panel_name]["B_reversed_replay"][index] = replay[1]
                    outputs[panel_name]["B_candidate_mismatch_replay"][index] = replay[2]
                    b_bound_by_query[panel_name][queries] = bound
                arm_record["prediction_queries"][panel_name] = int(len(queries))
            fold_report["arms"][arm_name] = arm_record
        fold_reports.append(fold_report)

    if any(
        not np.all(np.isfinite(values))
        for methods in outputs.values() for values in methods.values()
    ):
        raise RuntimeError("cross-fitted prediction ledger is incomplete")
    if any(np.any(~np.isfinite(values)) for values in b_bound_by_query.values()):
        raise RuntimeError("cross-fitted B residual-bound ledger is incomplete")

    report = {
        "status": "WSE_CONDITIONAL_RESIDUAL_DEVELOPMENT_COMPLETE",
        "dataset": "GNPS Gold/Silver 10-ppm identity/formula panels",
        "dataset_role": "development_consumed",
        "enveda_accessed": False,
        "method": (
            "raw-WSE anchor + bounded Noise/P2b spectral residual + optional "
            "conditional continuous correct-minus-null ChemAware residual"
        ),
        "arms": {
            "A": "WSE + Noise V1/P2b-on-Noise composite spectral residual",
            "B": "A + f(spectral, chemical contrast)-f(spectral, zero contrast)",
        },
        "spectral_features": list(SPECTRAL_FEATURE_NAMES),
        "chemical_features": list(CHEMICAL_FEATURE_NAMES),
        "p2b_family_input_count": 1,
        "thresholded_chemaware_action_input": False,
        "formula_crossfit_folds": args.folds,
        "formula_panel_labels_used_for_fitting": False,
        "same_formula_always_same_held_fold": True,
        "optimization": {
            "epochs": args.epochs, "hidden": args.hidden,
            "learning_rate": args.learning_rate, "null_weight": args.null_weight,
            "trust_weight": args.trust_weight, "seed": args.seed,
            "hyperparameter_scan": False,
        },
        "old_comparator": {
            "source_report": str(args.old_report),
            "arm": next(iter(old_arm_names)),
            "reconstruction": "exact replay of saved fold residual coefficients; no retraining",
        },
        "metric_score_transform": "within-query rank-logit for AUROC/AUPRC comparability only",
        "folds": fold_reports,
        "panels": {},
        "stratification": {
            "near": "reported",
            "source": "unavailable in frozen candidate bundle",
            "measurement_condition": "unavailable in frozen candidate bundle",
        },
    }
    comparison_index = 0
    for panel_name, panel in panels.items():
        metrics = {}
        for method, score in outputs[panel_name].items():
            calibrated = evaluation_scores(score, panel["query_ptr"])
            metrics[method], _ = metric_summary(calibrated, metric_input(panel))
        comparisons = {}
        for candidate, baseline in (
            ("A", "wse"), ("A", "old_four_module"),
            ("B", "wse"), ("B", "old_four_module"), ("B", "A"),
            ("B", "B_zero_replay"), ("B", "B_reversed_replay"),
            ("B", "B_candidate_mismatch_replay"),
        ):
            comparisons[f"{candidate}_vs_{baseline}"] = paired_comparison(
                outputs[panel_name][candidate], outputs[panel_name][baseline], panel,
                args.seed + 1000 + comparison_index, args.bootstrap_resamples,
            )
            comparison_index += 1
        report["panels"][panel_name] = {
            "queries": int(len(panel["query_ids"])),
            "candidates": int(len(panel["candidate_ids"])),
            "chemical_available_candidates": int(panel["chemical_available"].sum()),
            "methods": metrics,
            "comparisons": comparisons,
            "null_replay_interpretation": (
                "input-dependence falsification only; not a standalone chemical causal claim"
            ),
            "diagnostics": residual_diagnostics(
                panel, outputs[panel_name], b_bound_by_query[panel_name],
            ),
        }

    panel_names = tuple(panels)
    comparison = lambda key: {
        panel: report["panels"][panel]["comparisons"][key] for panel in panel_names
    }
    b_comparison_keys = ("B_vs_wse", "B_vs_old_four_module", "B_vs_A")
    b_metric_baselines = ("wse", "old_four_module", "A")
    b_selected = (
        all(arm_beats(comparison(key), panel_names) for key in b_comparison_keys)
        and all(
            metric_profile_noninferior(report["panels"], "B", baseline, panel_names)
            for baseline in b_metric_baselines
        )
    )
    a_comparison_keys = ("A_vs_wse", "A_vs_old_four_module")
    a_metric_baselines = ("wse", "old_four_module")
    a_qualified = (
        all(arm_beats(comparison(key), panel_names) for key in a_comparison_keys)
        and all(
            metric_profile_noninferior(report["panels"], "A", baseline, panel_names)
            for baseline in a_metric_baselines
        )
    )
    b_independent = (
        arm_beats(comparison("B_vs_A"), panel_names)
        and metric_profile_noninferior(report["panels"], "B", "A", panel_names)
    )
    report["selection_policy"] = {
        "status": (
            "post-result specification correction on consumed GNPS development data; "
            "not a preregistered confirmation"
        ),
        "primary": (
            "paired Recall@1 positive with formula-cluster CI lower bound > 0, "
            "near Recall@1 nonnegative, and MRR/micro-AUROC/micro-AUPRC noninferior "
            "on both consumed GNPS development panels"
        ),
        "legacy_lambda2_utility": (
            "corrected - 2*introduced is reported as a separate conservative-use "
            "utility and is not a primary ranking-selection veto"
        ),
        "legacy_lambda2_pass": {
            key: lambda2_risk_beats(comparison(key), panel_names)
            for key in b_comparison_keys
        },
    }
    if b_selected:
        decision = "SELECT_B"
        reason = (
            "B stably beats WSE, the old four-module model, and A on primary ranking "
            "endpoints without near regression; legacy lambda-2 utility remains separately reported"
        )
    elif a_qualified and not b_independent:
        decision = "SELECT_A"
        reason = "A has stable incremental value; B adds no independent stable gain"
    else:
        decision = "STOP_CONDITIONAL_RESIDUAL_ROUTE"
        reason = "neither permitted construction meets the corrected primary development-selection rule"
    report["decision"] = decision
    report["decision_reason"] = reason
    report["new_external_validation"] = (
        "required before a generalization claim; Enveda-180 is consumed and cannot select or retest this method"
    )

    args.output.mkdir(parents=True)
    if decision in ("SELECT_A", "SELECT_B"):
        selected_name = "A_spectral_residual" if decision == "SELECT_A" else "B_joint_residual"
        selected_mode = ARM_MODES[selected_name]
        all_identity_queries = np.arange(
            len(panels["identity"]["query_ptr"]) - 1, dtype=np.int64,
        )
        final_model, final_mean, final_scale, final_bound, final_history, final_capacity = fit_arm(
            panels["identity"], all_identity_queries, selected_mode, args,
            args.seed + 1000,
        )
        model_path = args.output / "selected_model.pt"
        torch.save({
            "state_dict": final_model.cpu().state_dict(),
            "model_class": "AnchoredEvidenceEnergy",
            "selected_arm": selected_name,
            "mode": selected_mode,
            "spectral_feature_names": SPECTRAL_FEATURE_NAMES,
            "chemical_feature_names": CHEMICAL_FEATURE_NAMES,
            "spectral_feature_count": len(SPECTRAL_FEATURE_NAMES),
            "hidden": args.hidden,
            "residual_bound_raw_wse_units": final_bound,
            "scaler_mean": final_mean,
            "scaler_scale": final_scale,
            "training_seed": args.seed + 1000,
            "training_queries": len(all_identity_queries),
            "enveda_accessed": False,
        }, model_path)
        report["selected_model"] = {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "fit_scope": "all GNPS identity-development queries after formula-OOF selection",
            "residual_bound_raw_wse_units": final_bound,
            "capacity": final_capacity,
            "final_loss": final_history[-1],
        }
    else:
        report["selected_model"] = None
    prediction_path = args.output / "candidate_predictions.npz"
    arrays = {}
    for panel_name, panel in panels.items():
        for method, values in outputs[panel_name].items():
            arrays[f"{panel_name}__{method}"] = values
        arrays[f"{panel_name}__query_ptr"] = panel["query_ptr"]
        arrays[f"{panel_name}__labels"] = panel["labels"]
        arrays[f"{panel_name}__query_ids"] = panel["query_ids"]
        arrays[f"{panel_name}__candidate_ids"] = panel["candidate_ids"]
        arrays[f"{panel_name}__formula"] = panel["formulas"]
        arrays[f"{panel_name}__near"] = panel["near"]
        arrays[f"{panel_name}__chemical_available"] = panel["chemical_available"]
        arrays[f"{panel_name}__B_residual_bound_by_query"] = b_bound_by_query[panel_name]
    np.savez_compressed(prediction_path, **arrays)
    ledger_path = args.output / "query_predictions_and_exchange.csv.gz"
    write_query_ledger(ledger_path, panels, outputs)
    table_path = args.output / "results_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "panel", "method", "queries", "recall@1", "recall@3", "recall@5",
            "recall@10", "recall@20", "mrr",
            "macro_query_auroc", "macro_query_auprc",
            "micro_candidate_auroc", "micro_candidate_auprc", "near_recall@1",
        ))
        for panel_name in panels:
            for method in ("wse", "old_four_module", "A", "B"):
                metrics = report["panels"][panel_name]["methods"][method]
                writer.writerow((
                    panel_name, method, metrics["queries"], metrics["recall@1"],
                    metrics["recall@3"], metrics["recall@5"], metrics["recall@10"],
                    metrics["recall@20"],
                    metrics["mrr"], metrics["macro_query_auroc"],
                    metrics["macro_query_auprc"], metrics["micro_candidate_auroc"],
                    metrics["micro_candidate_auprc"], metrics["near"].get("recall@1", ""),
                ))
    report["artifacts"] = {
        "candidate_predictions": str(prediction_path),
        "candidate_predictions_sha256": sha256(prediction_path),
        "query_exchange": str(ledger_path),
        "query_exchange_sha256": sha256(ledger_path),
        "results_table": str(table_path),
        "results_table_sha256": sha256(table_path),
    }
    report["input_sha256"] = {
        "identity_bundle": sha256(args.identity_bundle),
        "formula_bundle": sha256(args.formula_bundle),
        "identity_chem": sha256(args.identity_chem),
        "formula_chem": sha256(args.formula_chem),
        "old_report": sha256(args.old_report),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "decision": decision,
        "decision_reason": reason, "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

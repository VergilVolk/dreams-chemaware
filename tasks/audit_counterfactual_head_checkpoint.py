"""Audit a head-only counterfactual DreaMS checkpoint without re-encoding spectra.

The audit uses frozen official-backbone hidden states, so it is exact for the
``head`` training stage.  It reports paired error transitions, formula-clustered
bootstrap intervals, counterfactual intervention behavior, parameter drift, and
full-candidate retrieval transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from e1_checkpoint_io import official_head_state, torch_load_compat


ROOT = Path(__file__).resolve().parents[1]


def encode(hidden: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    with torch.inference_mode():
        result = F.normalize(
            F.linear(
                torch.from_numpy(hidden).float(),
                state["weight"].float(),
                state["bias"].float(),
            ),
            dim=-1,
        )
    return result.numpy()


def paired_metrics(
    split: pd.DataFrame,
    embedding: np.ndarray,
    row_to_local: dict[int, int],
    mask: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    clean_index = np.asarray([row_to_local[int(row)] for row in split["query_hdf5_row"]], dtype=int)
    positive_index = np.asarray([row_to_local[int(row)] for row in split["identity_hdf5_row"]], dtype=int)
    negative_index = np.asarray([row_to_local[int(row)] for row in split["confounder_hdf5_row"]], dtype=int)
    clean = embedding[clean_index]
    positive = embedding[positive_index]
    negative = embedding[negative_index]
    clean_margin = np.sum(clean * positive, axis=1) - np.sum(clean * negative, axis=1)
    frame = split[["formula", "transition"]].copy()
    frame["margin"] = clean_margin
    frame["correct"] = clean_margin > 0
    frame = frame.loc[mask].reset_index(drop=True)
    summary = {
        "n": int(len(frame)),
        "formulas": int(frame["formula"].nunique()),
        "pairwise_accuracy": float(frame["correct"].mean()),
        "mean_margin": float(frame["margin"].mean()),
        "median_margin": float(frame["margin"].median()),
        "margin_quantiles": [float(x) for x in frame["margin"].quantile([0.05, 0.25, 0.75, 0.95])],
    }
    return summary, frame


def cluster_bootstrap_delta(
    frame: pd.DataFrame,
    column: str,
    iterations: int,
    seed: int,
) -> list[float]:
    groups = {
        formula: group[f"{column}_model"].to_numpy(float) - group[f"{column}_official"].to_numpy(float)
        for formula, group in frame.groupby("formula")
    }
    formulas = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        draws[i] = np.concatenate([groups[formula] for formula in sampled]).mean()
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def transitions(official: pd.Series, model: pd.Series) -> dict[str, int]:
    a, b = official.astype(bool).to_numpy(), model.astype(bool).to_numpy()
    return {
        "official_wrong_model_correct": int(np.sum(~a & b)),
        "official_correct_model_wrong": int(np.sum(a & ~b)),
        "both_correct": int(np.sum(a & b)),
        "both_wrong": int(np.sum(~a & ~b)),
        "net_corrections": int(np.sum(~a & b) - np.sum(a & ~b)),
    }


def threshold_audit(frame: pd.DataFrame, thresholds: tuple[float, ...]) -> list[dict]:
    values = []
    official_margin = frame["margin_official"].to_numpy(float)
    model_margin = frame["margin_model"].to_numpy(float)
    for threshold in thresholds:
        official = official_margin > threshold
        model = model_margin > threshold
        values.append({
            "threshold": threshold,
            "official_accuracy": float(official.mean()),
            "model_accuracy": float(model.mean()),
            "delta": float(model.mean() - official.mean()),
            "official_near_ties": int(np.sum(np.abs(official_margin) <= threshold)),
            "model_near_ties": int(np.sum(np.abs(model_margin) <= threshold)),
            **transitions(pd.Series(official), pd.Series(model)),
        })
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/e1/counterfactual_head_cache/hidden.npy")
    parser.add_argument("--cache-index", type=Path, default=ROOT / "data/e1/counterfactual_head_cache/index.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv")
    parser.add_argument("--all-cache", type=Path, default=ROOT / "data/e1/counterfactual_head_cache/all_discovery_hidden.npy")
    parser.add_argument("--official-embeddings", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/official_embeddings.npy")
    parser.add_argument("--retrieval-dir", type=Path, default=ROOT / "data/validation/counterfactual_formal_cpu_head_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/counterfactual_formal_cpu_head_audit")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = torch_load_compat(args.base_checkpoint, map_location="cpu")
    trained_head = trained["head_state_dict"]
    base_head = official_head_state(base)

    base_backbone = base["backbone_state_dict"]
    trained_backbone = trained["backbone_state_dict"]
    backbone_exact = set(base_backbone) == set(trained_backbone) and all(
        torch.equal(base_backbone[key], trained_backbone[key]) for key in base_backbone
    )
    weight_delta = torch.linalg.vector_norm(trained_head["weight"] - base_head["weight"])
    weight_norm = torch.linalg.vector_norm(base_head["weight"])
    bias_delta = torch.linalg.vector_norm(trained_head["bias"] - base_head["bias"])
    bias_norm = torch.linalg.vector_norm(base_head["bias"])

    split = pd.read_csv(args.split)
    validation = split["pilot_split"].eq("validation").to_numpy()
    manifest = pd.read_csv(ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv")
    row_to_local = pd.Series(manifest.index.to_numpy(), index=manifest["hdf5_row"].astype(int)).to_dict()
    all_hidden = np.load(args.all_cache)
    all_model = encode(all_hidden, trained_head)
    all_official = np.load(args.official_embeddings).astype(np.float32)
    official_summary, official_frame = paired_metrics(split, all_official, row_to_local, validation)
    model_summary, model_frame = paired_metrics(split, all_model, row_to_local, validation)
    paired = official_frame.add_suffix("_official").join(model_frame.add_suffix("_model"))
    paired["formula"] = paired["formula_official"]
    paired["transition"] = paired["transition_official"]
    pair_transitions = transitions(paired["correct_official"], paired["correct_model"])
    pair_delta = float(model_summary["pairwise_accuracy"] - official_summary["pairwise_accuracy"])
    pair_ci = cluster_bootstrap_delta(paired, "correct", args.bootstrap, 20260814)
    margin_delta = float(model_summary["mean_margin"] - official_summary["mean_margin"])
    margin_ci = cluster_bootstrap_delta(paired, "margin", args.bootstrap, 20260815)

    subgroup = []
    for label, group in paired.groupby("transition", sort=True):
        subgroup.append({
            "transition": label,
            "n": int(len(group)),
            "official_accuracy": float(group["correct_official"].mean()),
            "model_accuracy": float(group["correct_model"].mean()),
            "delta": float(group["correct_model"].mean() - group["correct_official"].mean()),
            **transitions(group["correct_official"], group["correct_model"]),
        })

    cosine = np.sum(all_model * all_official, axis=1)

    official_query = pd.read_csv(args.retrieval_dir / "official_queries.csv")
    model_query = pd.read_csv(args.retrieval_dir / "formal_head_queries.csv")
    retrieval = official_query.merge(
        model_query,
        on=["query_index", "ik14", "formula"],
        suffixes=("_official", "_model"),
        validate="one_to_one",
    )
    retrieval_transitions = transitions(retrieval["top1_official"], retrieval["top1_model"])

    history = trained.get("history", [])
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(trained.get("epoch", -1)),
        "checkpoint_stage": trained.get("stage"),
        "history_epochs_in_checkpoint": len(history),
        "parameter_audit": {
            "backbone_bitwise_equal_to_official": bool(backbone_exact),
            "head_weight_relative_l2_delta": float(weight_delta / weight_norm),
            "head_bias_relative_l2_delta": float(bias_delta / bias_norm),
            "all_discovery_embedding_cosine_quantiles": {
                "min": float(cosine.min()),
                "p01": float(np.quantile(cosine, 0.01)),
                "p05": float(np.quantile(cosine, 0.05)),
                "median": float(np.median(cosine)),
                "mean": float(cosine.mean()),
            },
        },
        "validation_pairwise": {
            "official": official_summary,
            "model": model_summary,
            "accuracy_delta": pair_delta,
            "accuracy_delta_formula_bootstrap_ci95": pair_ci,
            "mean_margin_delta": margin_delta,
            "mean_margin_delta_formula_bootstrap_ci95": margin_ci,
            "paired_transitions": pair_transitions,
            "tie_threshold_sensitivity": threshold_audit(paired, (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)),
            "by_discovery_transition": subgroup,
            "counterfactual_metrics_from_training_evaluator": trained.get("val_metrics", {}),
        },
        "full_candidate_retrieval": {
            "n_queries": int(len(retrieval)),
            "top1_transitions": retrieval_transitions,
            "top1_tie_threshold_sensitivity": threshold_audit(retrieval, (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)),
        },
        "claim_limit": "single seed; internal formula-isolated validation; head-only; confirmation/test untouched",
    }
    paired.to_csv(args.output_dir / "paired_validation_audit.csv", index=False)
    retrieval.to_csv(args.output_dir / "retrieval_transition_audit.csv", index=False)
    (args.output_dir / "checkpoint_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

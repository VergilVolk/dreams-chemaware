#!/usr/bin/env python
"""Audit deployable typed-context signal before reranker/adapter training.

This is a consumed-development diagnostic.  Truth is used only to measure
within-candidate-set separation; no feature is written with outcome labels for
later locked evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RELATION_NAMES = {
    0: "complete_direction_supported",
    1: "complete_direction_unknown",
    2: "incomplete_direction_supported",
    3: "incomplete_direction_unknown",
    4: "direction_conflicted",
}


def strict_rank(values: np.ndarray, positive: int) -> int:
    mask = np.ones(len(values), dtype=bool)
    mask[positive] = False
    return 1 + int(np.sum(values[mask] >= values[positive]))


def summarize_delta(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].to_numpy(float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/validation/bioaware_metdna3_context_adapter_dataset_local_20260830/dataset.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_typed_context_signal_audit_20260830"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    values = np.load(args.dataset)
    offsets = values["offsets"].astype(np.int64)
    positives = values["positive_indices"].astype(np.int64)
    query = values["query_embeddings"].astype(np.float32)
    candidate = values["candidate_embeddings"].astype(np.float32)
    seed = values["seed_prototypes"].astype(np.float32)
    seed_indices = values["seed_indices"].astype(np.int64)
    relation = values["relation_types"].astype(np.int64)
    features = values["edge_features"].astype(np.float32)
    masks = values["edge_masks"].astype(bool)
    rows = []
    candidate_rows = []
    for index in range(len(query)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        positive = int(positives[index])
        local_candidate = candidate[start:stop]
        baseline = local_candidate @ query[index]
        local_metrics = []
        for local, flat in enumerate(range(start, stop)):
            mask = masks[flat]
            edge_features = features[flat, mask]
            edge_relations = relation[flat, mask]
            edge_seeds = seed[seed_indices[flat, mask]]
            if len(edge_features):
                confidence = edge_features[:, 0]
                experimental = edge_features[:, 1]
                completeness = edge_features[:, 2]
                conflict = edge_features[:, 3]
                reliability = confidence * experimental * completeness * (1.0 - conflict)
                seed_cosine = edge_seeds @ local_candidate[local]
                weighted_cosine = reliability * seed_cosine
            else:
                reliability = np.empty(0, dtype=np.float32)
                weighted_cosine = np.empty(0, dtype=np.float32)
            metrics = {
                "has_context": float(len(edge_features) > 0),
                "edge_count": float(len(edge_features)),
                "reliability_sum": float(reliability.sum()),
                "reliability_max": float(reliability.max(initial=0)),
                "seed_cosine_weighted_sum": float(weighted_cosine.sum()),
                "seed_cosine_weighted_max": float(weighted_cosine.max(initial=0)),
            }
            for code, name in RELATION_NAMES.items():
                metrics[f"type_{name}"] = float(reliability[edge_relations == code].sum())
            local_metrics.append(metrics)
            candidate_rows.append({
                "query_id": str(values["query_ids"][index]),
                "rotation_fold": int(values["rotation_folds"][index]),
                "truth_formula": str(values["truth_formulas"][index]),
                "is_positive": local == positive,
                "baseline_score": float(baseline[local]),
                **metrics,
            })
        negative = np.ones(len(local_candidate), dtype=bool)
        negative[positive] = False
        hardest = int(np.flatnonzero(negative)[np.argmax(baseline[negative])])
        row = {
            "query_id": str(values["query_ids"][index]),
            "rotation_fold": int(values["rotation_folds"][index]),
            "truth_formula": str(values["truth_formulas"][index]),
            "baseline_correct": strict_rank(baseline, positive) == 1,
        }
        for key in local_metrics[positive]:
            row[f"delta_{key}"] = local_metrics[positive][key] - local_metrics[hardest][key]
        rows.append(row)
    pairs = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "truth_minus_hardest_negative.csv.gz", index=False)
    candidates.to_csv(args.output_dir / "candidate_signal_rows.csv.gz", index=False)
    metric_columns = [column.removeprefix("delta_") for column in pairs if column.startswith("delta_")]
    all_summary = {metric: summarize_delta(pairs, f"delta_{metric}") for metric in metric_columns}
    error_pairs = pairs.loc[~pairs.baseline_correct]
    error_summary = {metric: summarize_delta(error_pairs, f"delta_{metric}") for metric in metric_columns}
    # A feature-wise oracle is strictly headroom: it uses truth to choose a
    # feature and therefore is never reported as a model result.
    evidence_columns = [
        "reliability_sum", "reliability_max", "seed_cosine_weighted_sum",
        "seed_cosine_weighted_max", *[f"type_{name}" for name in RELATION_NAMES.values()],
    ]
    oracle_rescuable = np.zeros(len(error_pairs), dtype=bool)
    for metric in evidence_columns:
        oracle_rescuable |= error_pairs[f"delta_{metric}"].to_numpy(float) > 0
    report = {
        "status": "bioaware_metdna3_typed_context_signal_audit_complete",
        "formal": False,
        "protocol": "consumed HILIC development; identity-isolated context rotations; truth used for diagnostic only",
        "rotation_instances": int(len(pairs)),
        "queries": int(pairs.query_id.nunique()),
        "formulas": int(pairs.truth_formula.nunique()),
        "baseline_errors": int((~pairs.baseline_correct).sum()),
        "candidate_rows": int(len(candidates)),
        "positive_candidate_context_fraction": float(candidates.loc[candidates.is_positive, "has_context"].mean()),
        "negative_candidate_context_fraction": float(candidates.loc[~candidates.is_positive, "has_context"].mean()),
        "truth_minus_hardest_negative_all": all_summary,
        "truth_minus_hardest_negative_baseline_errors": error_summary,
        "per_feature_oracle_rescuable_errors": int(oracle_rescuable.sum()),
        "per_feature_oracle_rescuable_fraction": float(oracle_rescuable.mean()),
        "contracts": {
            "P2b": "forbidden",
            "phenotype_blind_inputs": True,
            "truth_used_only_for_consumed_diagnostic": True,
            "locked_internal_or_external_outcomes_opened": False,
        },
        "claim_limit": "Signal/headroom audit only; it is neither a deployable reranker nor an embedding result.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate official DreaMS on external ring-stratified 10-ppm retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def per_anchor(values: np.ndarray, units: list[dict], candidate_key: str) -> pd.DataFrame:
    rows = []
    for unit in units:
        if not unit["is_query_anchor"]:
            continue
        pair_id = int(unit["pair_id"])
        negatives = [int(value) for value in unit[candidate_key]]
        if not negatives:
            continue
        for view in (0, 1):
            query = values[pair_id, view]
            positive = float(query @ values[pair_id, 1 - view])
            matrix = np.einsum("nvd,d->nv", values[negatives], query)
            molecule_scores = matrix.max(axis=1)
            best_position = int(np.argmax(molecule_scores))
            best_pair = negatives[best_position]
            rows.append({
                "pair_id": pair_id, "ik14": unit["ik14"], "query_view": view,
                "ring_class": unit["ring_class"], "formula": unit["formula"],
                "candidate_protocol": candidate_key,
                "n_negative_molecules": len(negatives),
                "positive_similarity": positive,
                "best_negative_similarity": float(molecule_scores[best_position]),
                "margin": positive - float(molecule_scores[best_position]),
                "top1_correct": bool(positive > molecule_scores.max()),
                "pairwise_accuracy": float(np.mean(
                    (positive > molecule_scores).astype(float)
                    + 0.5 * (positive == molecule_scores).astype(float)
                )),
                "best_negative_pair_id": best_pair,
                "best_negative_ik14": units[best_pair]["ik14"],
                "best_negative_formula": units[best_pair]["formula"],
                "best_negative_ring_class": units[best_pair]["ring_class"],
            })
    return pd.DataFrame(rows)


def bootstrap(frame: pd.DataFrame, iterations: int, seed: int) -> dict:
    per_molecule = frame.groupby("ik14", sort=False).agg(
        top1=("top1_correct", "mean"),
        pairwise=("pairwise_accuracy", "mean"),
        margin=("margin", "mean"),
    )
    values = per_molecule[["top1", "pairwise", "margin"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    draws = values[indices].mean(axis=1)
    return {
        name + "_ci95": np.quantile(draws[:, position], [0.025, 0.975]).tolist()
        for position, name in enumerate(("top1", "pairwise", "margin"))
    }


def summarize(frame: pd.DataFrame, bootstrap_n: int, seed: int) -> dict:
    # Pooled AUC requires every negative score; query-level table stores only
    # the best negative, so report a directly auditable positive-vs-hardest-
    # negative AUC instead of reconstructing discarded scores.
    hard_scores = np.concatenate((frame["positive_similarity"], frame["best_negative_similarity"]))
    hard_labels = np.concatenate((np.ones(len(frame)), np.zeros(len(frame))))
    result = {
        "query_molecules": int(frame["ik14"].nunique()),
        "query_views": len(frame),
        "negative_links": int(frame["n_negative_molecules"].sum()),
        "hard_negative_roc_auc": float(roc_auc_score(hard_labels, hard_scores)),
        "query_macro_pairwise_accuracy": float(frame["pairwise_accuracy"].mean()),
        "top1_accuracy": float(frame["top1_correct"].mean()),
        "margin_mean": float(frame["margin"].mean()),
        "margin_median": float(frame["margin"].median()),
    }
    result.update(bootstrap(frame, bootstrap_n, seed))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/validation/external_ring_balanced_embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_ring_balanced_e0"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    report = {"status": "external_ring_balanced_e0", "splits": {}}
    for split in ("discovery", "confirmation"):
        units = json.loads((args.pilot_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        values = np.load(args.embedding_dir / f"{split}_official.npy").astype(np.float64)
        values /= np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)
        report["splits"][split] = {}
        for protocol in ("negative_pair_ids", "same_formula_negative_pair_ids"):
            frame = per_anchor(values, units, protocol)
            frame["split"] = split
            all_frames.append(frame)
            result = {"overall": summarize(frame, args.bootstrap, args.seed)}
            ring_seeds = {"acyclic": 11, "single_ring": 22, "multi_ring": 33}
            for ring_class in ("acyclic", "single_ring", "multi_ring"):
                subset = frame.loc[frame["ring_class"] == ring_class]
                result[ring_class] = summarize(subset, args.bootstrap, args.seed + ring_seeds[ring_class])
            report["splits"][split][protocol] = result
    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(args.output_dir / "query_results.csv", index=False)
    failures = full.loc[~full["top1_correct"]].copy()
    failures.to_csv(args.output_dir / "failures.csv", index=False)
    report["protocols"] = {
        "negative_pair_ids": "All different-molecule, same-inferred-adduct candidates within 10 ppm.",
        "same_formula_negative_pair_ids": "Subset restricted to candidates with the same molecular formula.",
        "hard_negative_roc_auc": "ROC AUC over positive scores and the single highest-scoring negative per query view.",
    }
    report["claim_limit"] = (
        "External annotated01 pilot after excluding MassSpecGym molecules. annotated01 provenance is absent, "
        "so overlap with the original DreaMS self-supervised pretraining corpus cannot be ruled out."
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

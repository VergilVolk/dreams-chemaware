"""Build query-level success and failure manifests for strict 10 ppm retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import discover_condition_invariant_subspace as factor_model
import evaluate_mass_dense_factor_retrieval as retrieval


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_OUTPUT = ROOT / "data/validation/mass_dense_failure_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-factors", type=int, default=16)
    parser.add_argument("--ridge-fraction", type=float, default=0.1)
    return parser.parse_args()


def cosine_scores(values: np.ndarray, pair_id: int, query_view: int, negatives):
    normalized = retrieval.normalize(values)
    query = normalized[pair_id, query_view]
    positive = float(query @ normalized[pair_id, 1 - query_view])
    negative_matrix = np.einsum("nvd,d->nv", normalized[negatives], query)
    molecule_scores = np.max(negative_matrix, axis=1)
    best_index = int(np.argmax(molecule_scores))
    best_view = int(np.argmax(negative_matrix[best_index]))
    return positive, float(molecule_scores[best_index]), int(negatives[best_index]), best_view


def build_rows(
    directory: Path,
    raw: np.ndarray,
    official: np.ndarray,
    official_factor: np.ndarray,
) -> pd.DataFrame:
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    rows = []
    for pair_id, pair in enumerate(pairs):
        negatives = [int(value) for value in pair["negative_pair_ids"]]
        for query_view in (0, 1):
            raw_pos, raw_neg, raw_best, raw_best_view = cosine_scores(
                raw, pair_id, query_view, negatives
            )
            off_pos, off_neg, off_best, off_best_view = cosine_scores(
                official, pair_id, query_view, negatives
            )
            fac_pos, fac_neg, fac_best, fac_best_view = cosine_scores(
                official_factor, pair_id, query_view, negatives
            )
            raw_ok = raw_pos > raw_neg
            official_ok = off_pos > off_neg
            factor_ok = fac_pos > fac_neg
            if not raw_ok and official_ok:
                group = "raw_wrong_official_correct"
            elif not raw_ok and not official_ok:
                group = "both_wrong"
            elif raw_ok and not official_ok:
                group = "raw_correct_official_wrong"
            elif official_ok and not factor_ok:
                group = "official_correct_factor_wrong"
            else:
                group = "both_correct"
            official_best_pair = pairs[off_best]
            rows.append({
                "pair_id": pair_id,
                "original_unit_id": pair["original_unit_id"],
                "ik14": pair["ik14"],
                "query_view": query_view,
                "query_row": pair["rows"][query_view],
                "positive_row": pair["rows"][1 - query_view],
                "adduct": pair["adduct"][query_view],
                "query_precursor_mz": pair["precursor_mz"][query_view],
                "n_negative_molecules": len(negatives),
                "group": group,
                "raw_correct": raw_ok,
                "official_correct": official_ok,
                "factor_correct": factor_ok,
                "raw_positive_similarity": raw_pos,
                "raw_best_negative_similarity": raw_neg,
                "raw_margin": raw_pos - raw_neg,
                "official_positive_similarity": off_pos,
                "official_best_negative_similarity": off_neg,
                "official_margin": off_pos - off_neg,
                "official_margin_gain": (off_pos - off_neg) - (raw_pos - raw_neg),
                "factor_positive_similarity": fac_pos,
                "factor_best_negative_similarity": fac_neg,
                "factor_margin": fac_pos - fac_neg,
                "official_best_negative_pair_id": off_best,
                "official_best_negative_ik14": official_best_pair["ik14"],
                "official_best_negative_row": official_best_pair["rows"][off_best_view],
                "official_best_negative_precursor_mz": official_best_pair["precursor_mz"][off_best_view],
                "official_best_negative_ppm": abs(
                    official_best_pair["precursor_mz"][off_best_view]
                    - pair["precursor_mz"][query_view]
                ) / pair["precursor_mz"][query_view] * 1e6,
                "raw_best_negative_pair_id": raw_best,
                "raw_best_negative_view": raw_best_view,
                "factor_best_negative_pair_id": fac_best,
                "factor_best_negative_view": fac_best_view,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery_official = retrieval.load_activations(
        args.discovery, "official", args.layer
    )
    factor_fit = factor_model.fit_factorization(
        discovery_official,
        args.pca_dim,
        args.n_factors,
        args.ridge_fraction,
    )
    summaries = {}
    for split, directory in (
        ("discovery", args.discovery),
        ("confirmation", args.confirmation),
    ):
        raw = retrieval.load_activations(directory, "raw", args.layer)
        official = retrieval.load_activations(directory, "official", args.layer)
        official_factor = (official - factor_fit.center) @ factor_fit.directions
        frame = build_rows(directory, raw, official, official_factor)
        frame.to_csv(args.output_dir / f"{split}_queries.csv", index=False)
        counts = frame["group"].value_counts().to_dict()
        summaries[split] = {
            "n_queries": len(frame),
            "group_counts": {key: int(value) for key, value in counts.items()},
            "group_fractions": {
                key: float(value / len(frame)) for key, value in counts.items()
            },
            "official_margin_gain_mean": float(frame["official_margin_gain"].mean()),
            "official_margin_gain_median": float(frame["official_margin_gain"].median()),
        }
    report = {
        "status": "mass_dense_failure_manifest",
        "config": {
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "layer": args.layer,
            "pca_dim": args.pca_dim,
            "n_factors": args.n_factors,
        },
        "group_definition": {
            "raw_wrong_official_correct": "Official fine-tuning repairs a raw SSL error.",
            "both_wrong": "Neither full embedding ranks the identity positive first.",
            "raw_correct_official_wrong": "Official fine-tuning introduces an error.",
            "official_correct_factor_wrong": "Full official space succeeds but invariant projection fails.",
            "both_correct": "Raw and official full embeddings both succeed; invariant projection also succeeds.",
        },
        "splits": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()

"""Evaluate full and condition-invariant DreaMS spaces on strict 10 ppm retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import discover_condition_invariant_subspace as factors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_OUTPUT = ROOT / "data/validation/mass_dense_factor_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-factors", type=int, default=16)
    parser.add_argument("--ridge-fraction", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_activations(directory: Path, kind: str, layer: int) -> np.ndarray:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    layers = report["config"]["layers"]
    layer_index = layers.index(layer)
    values = np.load(directory / f"{kind}_precursor.npy")[:, layer_index]
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    if len(values) != 2 * len(pairs):
        raise ValueError("Expected exactly two spectra per pair")
    return values.reshape(len(pairs), 2, values.shape[-1]).astype(np.float64)


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)


def query_records(values: np.ndarray, pair_manifest: list[dict]) -> list[dict]:
    values = normalize(values)
    records = []
    for pair_id, pair in enumerate(pair_manifest):
        negatives = [int(value) for value in pair["negative_pair_ids"]]
        if not negatives:
            continue
        negative_values = values[negatives]
        for query_view in (0, 1):
            query = values[pair_id, query_view]
            positive = float(query @ values[pair_id, 1 - query_view])
            negative_scores = np.max(
                np.einsum("nvd,d->nv", negative_values, query), axis=1
            )
            records.append({
                "positive": positive,
                "negatives": negative_scores,
            })
    return records


def summarize_records(records: list[dict], bootstrap: int, seed: int) -> dict:
    positives = np.asarray([record["positive"] for record in records])
    negatives = np.concatenate([record["negatives"] for record in records])
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    scores = np.concatenate([positives, negatives])
    per_query_pairwise = np.asarray([
        np.mean(
            (record["positive"] > record["negatives"]).astype(float)
            + 0.5 * (record["positive"] == record["negatives"]).astype(float)
        )
        for record in records
    ])
    per_query_top1 = np.asarray([
        float(record["positive"] > np.max(record["negatives"]))
        for record in records
    ])
    rng = np.random.RandomState(seed)
    boot_pairwise, boot_top1 = [], []
    for _ in range(bootstrap):
        sample = rng.randint(0, len(records), size=len(records))
        boot_pairwise.append(float(np.mean(per_query_pairwise[sample])))
        boot_top1.append(float(np.mean(per_query_top1[sample])))
    return {
        "n_queries": len(records),
        "n_positive_scores": len(positives),
        "n_negative_scores": len(negatives),
        "pooled_roc_auc": float(roc_auc_score(labels, scores)),
        "query_macro_pairwise_accuracy": float(np.mean(per_query_pairwise)),
        "query_macro_pairwise_accuracy_ci95": np.quantile(
            boot_pairwise, [0.025, 0.975]
        ).tolist(),
        "top1_accuracy": float(np.mean(per_query_top1)),
        "top1_accuracy_ci95": np.quantile(
            boot_top1, [0.025, 0.975]
        ).tolist(),
        "positive_similarity_median": float(np.median(positives)),
        "negative_similarity_median": float(np.median(negatives)),
    }


def evaluate_space(
    values: np.ndarray,
    pair_manifest: list[dict],
    bootstrap: int,
    seed: int,
) -> dict:
    return summarize_records(
        query_records(values, pair_manifest), bootstrap=bootstrap, seed=seed
    )


def analyze_kind(args: argparse.Namespace, kind: str, pair_manifest: list[dict]) -> dict:
    discovery = load_activations(args.discovery, kind, args.layer)
    confirmation = load_activations(args.confirmation, kind, args.layer)
    discovery_fit = factors.fit_factorization(
        discovery, args.pca_dim, args.n_factors, args.ridge_fraction
    )
    confirmation_fit = factors.fit_factorization(
        confirmation, args.pca_dim, args.n_factors, args.ridge_fraction
    )
    projected = (confirmation - discovery_fit.center) @ discovery_fit.directions
    return {
        "full_embedding": evaluate_space(
            confirmation, pair_manifest, args.bootstrap, args.seed
        ),
        "condition_invariant_subspace": evaluate_space(
            projected, pair_manifest, args.bootstrap, args.seed
        ),
        "external_factor_invariance": factors.external_metrics(
            confirmation, discovery_fit
        ),
        "independent_direction_replication": factors.direction_replication(
            discovery_fit, confirmation_fit
        ),
    }


def main() -> None:
    args = parse_args()
    discovery_pairs = json.loads(
        (args.discovery / "pairs.json").read_text(encoding="utf-8")
    )
    confirmation_pairs = json.loads(
        (args.confirmation / "pairs.json").read_text(encoding="utf-8")
    )
    discovery_ik = {pair["ik14"] for pair in discovery_pairs}
    confirmation_ik = {pair["ik14"] for pair in confirmation_pairs}
    overlap = discovery_ik & confirmation_ik
    if overlap:
        raise RuntimeError(f"Molecule leakage detected: {len(overlap)} IK14 values")
    result = {
        "status": "strict_10ppm_factor_retrieval",
        "protocol": (
            "For each spectrum query, the other spectrum of the same molecule is the "
            "positive. Negatives are different molecules with the same adduct and "
            "precursor m/z within 10 ppm; each negative molecule is represented by its "
            "maximum similarity across two condition views."
        ),
        "config": {
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "layer": args.layer,
            "pca_dim": args.pca_dim,
            "n_factors": args.n_factors,
            "ridge_fraction": args.ridge_fraction,
            "bootstrap": args.bootstrap,
        },
        "audit": {
            "discovery_molecules": len(discovery_pairs),
            "confirmation_molecules": len(confirmation_pairs),
            "molecule_overlap": len(overlap),
            "all_confirmation_queries_have_negatives": all(
                pair["negative_pair_ids"] for pair in confirmation_pairs
            ),
            "maximum_nearest_negative_ppm": max(
                pair["nearest_negative_ppm"] for pair in confirmation_pairs
            ),
        },
        "raw_ssl": analyze_kind(args, "raw", confirmation_pairs),
        "official_finetuned": analyze_kind(args, "official", confirmation_pairs),
        "interpretation_limit": (
            "A factor subspace that improves strict retrieval is still not a named "
            "chemical mechanism. Individual axes must replicate and pass peak-level "
            "perturbation before chemical annotation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for kind in ("raw_ssl", "official_finetuned"):
        full = result[kind]["full_embedding"]
        subspace = result[kind]["condition_invariant_subspace"]
        replication = result[kind]["independent_direction_replication"]
        print(
            f"{kind}: full AUC={full['pooled_roc_auc']:.3f}, "
            f"Top1={full['top1_accuracy']:.3f}; factor AUC={subspace['pooled_roc_auc']:.3f}, "
            f"Top1={subspace['top1_accuracy']:.3f}; stable axes "
            f">=0.7={replication['directions_ge_0_7']}/{args.n_factors}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

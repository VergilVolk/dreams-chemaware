"""Audit a label-free local-spectral consensus on frozen observability cohorts.

Five raw MS/MS similarities each nominate a molecule.  A route is activated
only when at least three views nominate the same molecule and that molecule
differs from the official DreaMS winner.  The routing rule contains no identity
label, fitted threshold, or DreaMS correctness flag.  Discovery and
formula-disjoint confirmation are both reported; no model is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
VIEWS = (
    "entropy_similarity",
    "sqrt_cosine",
    "linear_cosine",
    "top10_match_fraction",
    "intensity_coverage_min",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir", type=Path,
        default=ROOT / "data/validation/large_observability_residual_audit",
    )
    parser.add_argument(
        "--discovery-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
    )
    parser.add_argument(
        "--confirmation-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_label_free_spectral_consensus_v1/report.json",
    )
    return parser.parse_args()


def predicted_identity(frame: pd.DataFrame, manifest: pd.DataFrame, prefix: str) -> np.ndarray:
    correct = frame[f"{prefix}_top1_correct"].astype(bool).to_numpy()
    negative_index = frame[f"{prefix}_best_negative_index"].to_numpy(np.int64)
    negative_identity = manifest.iloc[negative_index]["ik14"].astype(str).to_numpy()
    return np.where(correct, frame["ik14"].astype(str).to_numpy(), negative_identity)


def formula_bootstrap(frame: pd.DataFrame, iterations: int, seed: int) -> list[float]:
    by_formula = frame.groupby("formula", sort=False).agg(
        n=("query_index", "size"),
        delta=("delta_correct", "sum"),
    )
    n = by_formula["n"].to_numpy(np.float64)
    delta = by_formula["delta"].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 500):
        stop = min(start + 500, iterations)
        draw = rng.integers(0, len(n), size=(stop - start, len(n)))
        values[start:stop] = delta[draw].sum(axis=1) / n[draw].sum(axis=1)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def audit_split(
    name: str,
    query_path: Path,
    manifest_path: Path,
    bootstrap: int,
    seed: int,
) -> dict:
    frame = pd.read_csv(query_path)
    manifest = pd.read_csv(manifest_path)
    if set(frame["split"].astype(str)) != {name}:
        raise RuntimeError(f"query audit split mismatch: {query_path}")
    truth = frame["ik14"].astype(str).to_numpy()
    dreams = predicted_identity(frame, manifest, "dreams")
    view_predictions = np.stack([
        predicted_identity(frame, manifest, view) for view in VIEWS
    ], axis=1)
    consensus = np.empty(len(frame), dtype=object)
    votes = np.zeros(len(frame), dtype=np.int64)
    for index, row in enumerate(view_predictions):
        winner, count = Counter(map(str, row)).most_common(1)[0]
        consensus[index] = winner
        votes[index] = count
    has_majority = votes >= 3
    route = has_majority & (consensus != dreams)
    routed = dreams.copy()
    routed[route] = consensus[route]
    base_correct = dreams == truth
    routed_correct = routed == truth
    corrected = (~base_correct) & routed_correct
    introduced = base_correct & (~routed_correct)
    changed_wrong_to_wrong = route & (~base_correct) & (~routed_correct)
    frame = frame.copy()
    frame["delta_correct"] = routed_correct.astype(np.int8) - base_correct.astype(np.int8)
    view_accuracy = {
        view: float(np.mean(view_predictions[:, index] == truth))
        for index, view in enumerate(VIEWS)
    }
    return {
        "queries": len(frame),
        "identities": int(frame["ik14"].nunique()),
        "formulas": int(frame["formula"].nunique()),
        "official_dreams_recall1": float(np.mean(base_correct)),
        "individual_view_recall1": view_accuracy,
        "majority_available": int(np.sum(has_majority)),
        "majority_available_fraction": float(np.mean(has_majority)),
        "route_activated": int(np.sum(route)),
        "route_activated_fraction": float(np.mean(route)),
        "consensus_accuracy_when_routed": (
            float(np.mean(consensus[route] == truth[route])) if np.any(route) else None
        ),
        "dreams_accuracy_when_routed": (
            float(np.mean(base_correct[route])) if np.any(route) else None
        ),
        "routed_recall1": float(np.mean(routed_correct)),
        "routed_minus_dreams_pp": float(100.0 * np.mean(routed_correct.astype(float) - base_correct.astype(float))),
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "wrong_to_different_wrong": int(np.sum(changed_wrong_to_wrong)),
        "risk_utility_corrected_minus_2x_introduced": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "formula_cluster_bootstrap_delta_ci95": formula_bootstrap(frame, bootstrap, seed),
        "vote_count_histogram": {
            str(value): int(np.sum(votes == value)) for value in np.unique(votes)
        },
        "provenance": {
            "query_audit": str(query_path),
            "query_audit_sha256": sha256(query_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    paths = {
        "discovery": (
            args.audit_dir / "discovery_query_audit.csv", args.discovery_manifest,
        ),
        "confirmation": (
            args.audit_dir / "confirmation_query_audit.csv", args.confirmation_manifest,
        ),
    }
    missing = [str(path) for pair in paths.values() for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    result = {
        "status": "chemaware_label_free_spectral_consensus_audited",
        "training_was_run": False,
        "routing_rule": (
            "route from DreaMS only when >=3/5 raw spectral views nominate the same "
            "molecule and that molecule differs from the DreaMS winner"
        ),
        "routing_rule_uses_identity_labels": False,
        "views": list(VIEWS),
        "splits": {
            name: audit_split(name, query, manifest, args.bootstrap, args.seed + index)
            for index, (name, (query, manifest)) in enumerate(paths.items())
        },
        "claim_limit": (
            "Internal score-independent multi-replicate same-formula cohort; train/val were "
            "pooled before formula isolation, the final test split remains untouched, and this "
            "is neither a P3 result nor evidence that the consensus can be compressed into DreaMS."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Probe whether DreaMS repair/failure cases occupy reproducible linear directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import evaluate_mass_dense_factor_retrieval as retrieval


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_FAILURE = ROOT / "data/validation/mass_dense_failure_audit"
DEFAULT_OUTPUT = ROOT / "data/validation/finetuning_gain_probe.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--failure-dir", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--regularization-c", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def feature_matrices(directory: Path, layer: int) -> dict[str, np.ndarray]:
    raw = retrieval.load_activations(directory, "raw", layer)
    official = retrieval.load_activations(directory, "official", layer)
    raw = retrieval.normalize(raw).reshape(-1, raw.shape[-1])
    official = retrieval.normalize(official).reshape(-1, official.shape[-1])
    return {
        "raw_embedding": raw,
        "official_embedding": official,
        "finetuning_delta": official - raw,
    }


def labels(frame: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "official_failure":
        mask = np.ones(len(frame), dtype=bool)
        y = (~frame["official_correct"].astype(bool)).to_numpy(dtype=int)
    elif task == "raw_error_repaired":
        mask = (~frame["raw_correct"].astype(bool)).to_numpy()
        y = frame.loc[mask, "official_correct"].astype(bool).to_numpy(dtype=int)
    else:
        raise ValueError(task)
    return mask, y


def fit_probe(x: np.ndarray, y: np.ndarray, c_value: float):
    scaler = StandardScaler()
    transformed = scaler.fit_transform(x)
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=0,
    )
    model.fit(transformed, y)
    direction = model.coef_[0] / scaler.scale_.clip(1e-12)
    direction /= np.linalg.norm(direction).clip(1e-12)
    return scaler, model, direction


def cluster_bootstrap_auc(
    labels_array: np.ndarray,
    scores: np.ndarray,
    pair_ids: np.ndarray,
    runs: int,
    seed: int,
) -> list[float]:
    rng = np.random.RandomState(seed)
    unique = np.unique(pair_ids)
    values = []
    for _ in range(runs):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(pair_ids == value) for value in sampled])
        y = labels_array[indices]
        if len(np.unique(y)) < 2:
            continue
        values.append(float(roc_auc_score(y, scores[indices])))
    return np.quantile(values, [0.025, 0.975]).tolist()


def analyze_task(
    task: str,
    feature_name: str,
    discovery_x: np.ndarray,
    confirmation_x: np.ndarray,
    discovery_frame: pd.DataFrame,
    confirmation_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    discovery_mask, discovery_y = labels(discovery_frame, task)
    confirmation_mask, confirmation_y = labels(confirmation_frame, task)
    train_x = discovery_x[discovery_mask]
    test_x = confirmation_x[confirmation_mask]
    scaler, model, discovery_direction = fit_probe(
        train_x, discovery_y, args.regularization_c
    )
    probabilities = model.predict_proba(scaler.transform(test_x))[:, 1]
    _, _, confirmation_direction = fit_probe(
        test_x, confirmation_y, args.regularization_c
    )
    pair_ids = confirmation_frame.loc[confirmation_mask, "pair_id"].to_numpy()
    prevalence = float(np.mean(confirmation_y))
    return {
        "task": task,
        "feature": feature_name,
        "n_discovery": int(len(discovery_y)),
        "n_confirmation": int(len(confirmation_y)),
        "confirmation_prevalence": prevalence,
        "confirmation_roc_auc": float(roc_auc_score(confirmation_y, probabilities)),
        "confirmation_roc_auc_cluster_bootstrap_ci95": cluster_bootstrap_auc(
            confirmation_y,
            probabilities,
            pair_ids,
            args.bootstrap,
            args.seed,
        ),
        "confirmation_auprc": float(
            average_precision_score(confirmation_y, probabilities)
        ),
        "random_auprc_reference": prevalence,
        "independent_direction_absolute_cosine": float(
            abs(discovery_direction @ confirmation_direction)
        ),
        "direction": discovery_direction.tolist(),
    }


def main() -> None:
    args = parse_args()
    discovery_features = feature_matrices(args.discovery, args.layer)
    confirmation_features = feature_matrices(args.confirmation, args.layer)
    discovery_frame = pd.read_csv(args.failure_dir / "discovery_queries.csv")
    confirmation_frame = pd.read_csv(args.failure_dir / "confirmation_queries.csv")
    results = []
    for task in ("official_failure", "raw_error_repaired"):
        for feature_name in discovery_features:
            results.append(analyze_task(
                task,
                feature_name,
                discovery_features[feature_name],
                confirmation_features[feature_name],
                discovery_frame,
                confirmation_frame,
                args,
            ))
    report = {
        "status": "finetuning_gain_linear_probe",
        "config": {
            "layer": args.layer,
            "regularization_c": args.regularization_c,
            "bootstrap": args.bootstrap,
        },
        "results": results,
        "gate": (
            "A direction is eligible for peak-level follow-up only if it predicts the "
            "molecule-disjoint confirmation set and its independently fitted coefficient "
            "direction is reproducible. Prediction alone does not establish chemistry."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in results:
        print(
            f"{row['task']} / {row['feature']}: AUC={row['confirmation_roc_auc']:.3f} "
            f"CI={row['confirmation_roc_auc_cluster_bootstrap_ci95']}; "
            f"AUPRC={row['confirmation_auprc']:.3f}; "
            f"direction |cos|={row['independent_direction_absolute_cosine']:.3f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Freeze the RAW reranker v1 model artifact (no re-training at eval time).

Trains once from the train cache, then saves the exact model parameters plus
config + provenance hashes to a JSON artifact.  The eval loads ONLY this
artifact and reconstructs the score as (x - mean)/scale @ coef, never calling
fit_ranker().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_g8r_raw_reranker import fit_ranker, RAW_FEATURES  # noqa: E402

DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_TRAIN_SCRIPT = ROOT / "tasks/train_g8r_raw_reranker.py"
DEFAULT_OUT = ROOT / "data/validation/g8r_raw_reranker_v1_artifact.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hard-k", type=int, default=5)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--gate-threshold", type=float, default=0.24098341166973114)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    a = parse_args()
    cache = np.load(a.cache, allow_pickle=True)
    tr = pd.DataFrame({k: cache[k] for k in cache.files})
    features = ["dreams_similarity"] + RAW_FEATURES
    scaler, model = fit_ranker(tr, features, a.hard_k, a.C)

    artifact = {
        "format": "raw_reranker_v1_artifact",
        "feature_names": features,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "model_coef": [float(x) for x in model.coef_[0]],
        "model_intercept": float(model.intercept_[0]) if model.intercept_ is not None else 0.0,
        "C": a.C,
        "hard_k": a.hard_k,
        "gate_threshold": a.gate_threshold,
        "gate_require_disagreement": False,
        "train_cache_sha256": sha256(a.cache),
        "train_script_sha256": sha256(a.train_script),
        "sklearn_version": sklearn.__version__,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(artifact, ensure_ascii=False, indent=2)
    a.output.write_text(raw, encoding="utf-8")
    artifact_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    a.output.with_suffix(".sha256").write_text(artifact_sha + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in artifact.items()
                      if k not in ("scaler_mean", "scaler_scale", "model_coef")}, indent=2))
    print(f"artifact_sha256 = {artifact_sha}")
    print(f"Saved artifact: {a.output} + {a.output.with_suffix('.sha256')}")


if __name__ == "__main__":
    main()

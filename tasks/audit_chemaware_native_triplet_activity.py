"""Audit expected native DreaMS hinge activity of a frozen triplet pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_chemaware_action_hard_native_triplets import load_npz
from build_chemaware_residual_native_triplets import cache_arrays, positions


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    pool = load_npz(args.pool)
    cache_rows, embedding, embedding_path = cache_arrays(args.embedding_cache)
    activation = []
    mean_hinge = []
    molecule_max_violation = []
    for event, anchor_row in enumerate(np.asarray(pool["anchor_idx"], dtype=np.int64)):
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        positive_rows = np.asarray(pool["positive_idx"][p0:p1], dtype=np.int64)
        negative_rows = np.asarray(pool["negative_idx"][n0:n1], dtype=np.int64)
        if not len(positive_rows) or not len(negative_rows):
            raise RuntimeError("native triplet event has an empty reference side")
        query_z = np.asarray(embedding[positions(cache_rows, np.asarray([anchor_row]))[0]])
        positive = np.asarray(embedding[positions(cache_rows, positive_rows)]) @ query_z
        negative = np.asarray(embedding[positions(cache_rows, negative_rows)]) @ query_z
        hinge = np.maximum(float(args.margin) + negative[:, None] - positive[None, :], 0.0)
        activation.append(float(np.mean(hinge > 0.0)))
        mean_hinge.append(float(np.mean(hinge)))
        molecule_max_violation.append(
            bool(float(args.margin) + float(np.max(negative)) - float(np.max(positive)) > 0.0)
        )
    activation_array = np.asarray(activation, dtype=np.float64)
    hinge_array = np.asarray(mean_hinge, dtype=np.float64)
    report = {
        "status": "CHEMAWARE_NATIVE_TRIPLET_ACTIVITY_AUDIT_COMPLETE",
        "pool": str(args.pool.resolve()),
        "embedding_cache": str(args.embedding_cache.resolve()),
        "embedding_cache_file": str(embedding_path.resolve()),
        "margin": float(args.margin),
        "events": int(len(activation_array)),
        "active_events": int(np.sum(activation_array > 0.0)),
        "zero_activation_events": int(np.sum(activation_array == 0.0)),
        "mean_activation_probability": float(np.mean(activation_array)),
        "median_activation_probability": float(np.median(activation_array)),
        "mean_native_hinge": float(np.mean(hinge_array)),
        "molecule_max_margin_violations": int(np.sum(molecule_max_violation)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

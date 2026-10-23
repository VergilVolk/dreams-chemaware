#!/usr/bin/env python
"""Encode only MassSpecGym training rows reachable from a native anchor pool.

The resulting cache is used to mine ChemAware hard negatives under a frozen
Noise-V1 geometry.  Held-query rows are not encoded or inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluate_chemaware_v2_direct_triplet import encode_rows, load_model
from noise_final_core import sha256_file


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def required_rows(manifest: dict[str, np.ndarray], anchors: np.ndarray) -> tuple[np.ndarray, int]:
    selected = np.flatnonzero(np.isin(manifest["query_row"], anchors))
    if not len(selected):
        raise RuntimeError("anchor pool has no query in the candidate manifest")
    blocks = [np.asarray(manifest["query_row"][selected], dtype=np.int64)]
    for query in selected:
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        molecule_left = int(manifest["molecule_ptr"][left])
        molecule_right = int(manifest["molecule_ptr"][right])
        blocks.append(np.asarray(
            manifest["pair_candidate_row"][molecule_left:molecule_right],
            dtype=np.int64,
        ))
    return np.unique(np.concatenate(blocks)), int(len(selected))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-pool", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("training-manifest encoding requires one allocated GPU")
    for path in (
        args.checkpoint, args.anchor_pool, args.manifest, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in (
            "query_row", "query_ptr", "molecule_ptr", "pair_candidate_row",
        )}
    with np.load(args.anchor_pool, allow_pickle=False) as loaded:
        anchors = np.asarray(loaded["anchor_idx"], dtype=np.int64)
    if len(anchors) == 0 or len(np.unique(anchors)) != len(anchors):
        raise RuntimeError("training anchor pool is empty or repeats anchors")
    rows, queries = required_rows(manifest, anchors)
    model, kind = load_model(
        args.checkpoint, args.official_checkpoint,
        args.architecture_checkpoint, torch.device("cuda"), 100,
    )
    embeddings = encode_rows(
        model, rows, args.data, torch.device("cuda"), args.batch_size, 100,
    ).astype(np.float32, copy=False)
    del model
    torch.cuda.empty_cache()
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError("training-manifest embeddings are not finite unit vectors")
    report = {
        "status": "CHEMAWARE_TRAINING_MANIFEST_EMBEDDINGS_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_kind": kind,
        "anchor_pool": str(args.anchor_pool.resolve()),
        "anchor_pool_sha256": sha256_file(args.anchor_pool),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "training_queries": queries,
        "training_anchors": int(len(anchors)),
        "rows": int(len(rows)),
        "embedding_dimension": int(embeddings.shape[1]),
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(embeddings),
        "candidate_scores_computed": False,
        "truth_labels_accessed": False,
        "scope_train_anchors_only": True,
        "held_query_rows_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_train_geometry_", dir=args.output.parent))
    try:
        np.save(temporary / "rows.npy", rows, allow_pickle=False)
        np.save(temporary / "embeddings_f32.npy", embeddings, allow_pickle=False)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

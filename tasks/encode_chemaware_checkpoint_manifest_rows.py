"""Encode every spectrum row reachable from the corrected ChemAware graph.

The cache is checkpoint-specific and contains no candidate score or label.  It
is used only to re-mine identity-valid triplets after a short DreaMS
continuation, without repeatedly running the 116M-parameter encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_chemaware_counterfactual_rule_kernel import sha256  # noqa: E402
from evaluate_chemaware_v2_direct_triplet import encode_rows, load_model  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def evidence_queries(directory: Path) -> tuple[np.ndarray, dict[str, int]]:
    blocks = []
    counts = {}
    for role, file_name in (
        ("train", "train_triplet_evidence.npz"),
        ("selection", "selection_triplet_evidence.npz"),
        ("confirmation", "confirmation_triplet_evidence.npz"),
    ):
        with np.load(directory / file_name, allow_pickle=False) as loaded:
            query = np.asarray(loaded["query"], dtype=np.int64)
        if len(np.unique(query)) != len(query):
            raise RuntimeError(f"{role} evidence repeats query indices")
        blocks.append(query)
        counts[role] = int(len(query))
    queries = np.concatenate(blocks)
    if len(np.unique(queries)) != len(queries):
        raise RuntimeError("formula-role evidence repeats queries across roles")
    return queries, counts


def required_rows(manifest: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    blocks = [np.asarray(manifest["query_row"], dtype=np.int64)[queries]]
    for query in queries:
        mleft, mright = map(int, manifest["query_ptr"][int(query):int(query) + 2])
        pleft = int(manifest["molecule_ptr"][mleft])
        pright = int(manifest["molecule_ptr"][mright])
        blocks.append(np.asarray(manifest["pair_candidate_row"][pleft:pright], dtype=np.int64))
    rows = np.unique(np.concatenate(blocks))
    if not len(rows) or np.any(np.diff(rows) <= 0):
        raise RuntimeError("manifest row registry is empty or not strictly increasing")
    return rows


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal ChemAware checkpoint encoding requires one CUDA GPU")
    for path in (
        args.checkpoint, args.manifest, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {
            key: np.asarray(loaded[key])
            for key in ("query_row", "query_ptr", "molecule_ptr", "pair_candidate_row")
        }
    queries, role_query_counts = evidence_queries(args.evidence_dir)
    rows = required_rows(manifest, queries)
    device = torch.device(args.device)
    model, kind = load_model(
        args.checkpoint, args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    encoded = encode_rows(
        model, rows, args.data, device, args.batch_size, args.n_highest_peaks,
    ).astype(np.float32, copy=False)
    del model
    torch.cuda.empty_cache()
    if encoded.shape[0] != len(rows) or encoded.ndim != 2:
        raise RuntimeError("checkpoint embedding cache has an invalid shape")
    norms = np.linalg.norm(encoded, axis=1)
    report = {
        "status": "CHEMAWARE_CHECKPOINT_MANIFEST_EMBEDDINGS_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_kind": kind,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "evidence_directory": str(args.evidence_dir.resolve()),
        "formula_role_query_counts": role_query_counts,
        "queries": int(len(queries)),
        "rows": int(len(rows)),
        "embedding_dimension": int(encoded.shape[1]),
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(encoded),
        "candidate_scores_computed": False,
        "row_scope": "only queries and candidate references reachable from roles 0-3 evidence",
        "formula_role_4_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_checkpoint_rows_", dir=args.output.parent))
    try:
        np.save(temporary / "rows.npy", rows, allow_pickle=False)
        np.save(temporary / "embeddings_f32.npy", encoded, allow_pickle=False)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

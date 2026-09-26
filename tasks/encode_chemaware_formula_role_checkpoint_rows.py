"""Encode every candidate-graph row for selected formula roles.

This cache is used only to mine native DreaMS triplets under an already trained
shared embedding.  It contains spectrum embeddings and row indices, never
candidate labels or candidate scores.  Formal residual mining is restricted to
training formula roles 0 and 1.
"""
from __future__ import annotations

import argparse
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
from chemaware_numpy_sampling import stable_formula_folds  # noqa: E402
from encode_chemaware_checkpoint_manifest_rows import (  # noqa: E402
    array_sha256,
    required_rows,
)
from evaluate_chemaware_v2_direct_triplet import encode_rows, load_model  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--formula-role", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    roles = tuple(dict.fromkeys(map(int, args.formula_role)))
    if set(roles) != {0, 1}:
        raise ValueError("formal ChemAware residual cache is restricted to roles 0 and 1")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal ChemAware role cache requires one CUDA GPU")
    for path in (
        args.checkpoint, args.manifest, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    folds = stable_formula_folds(manifest["query_formula"], args.folds, args.fold_seed)
    queries = np.flatnonzero(np.isin(folds, np.asarray(roles, dtype=folds.dtype)))
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
    norms = np.linalg.norm(encoded, axis=1)
    if encoded.ndim != 2 or len(encoded) != len(rows):
        raise RuntimeError("formula-role checkpoint cache has an invalid shape")
    if np.max(np.abs(norms - 1.0)) > 2e-4:
        raise RuntimeError("formula-role checkpoint cache is not normalized")
    report = {
        "status": "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_kind": kind,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "formula_roles": list(roles),
        "queries": int(len(queries)),
        "rows": int(len(rows)),
        "embedding_dimension": int(encoded.shape[1]),
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(encoded),
        "candidate_scores_stored": False,
        "candidate_labels_stored": False,
        "outer_roles_2_3_4_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_role_cache_", dir=args.output.parent))
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

"""E4-M0: freeze action targets for shared-embedding noise fine-tuning.

The selected E3 actions are *training-only views*.  At inference the adapter
receives an untouched spectrum and is applied identically to query and library
spectra.  This cache deliberately contains no P2b/reranker feature.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from build_g8r_real_error_atlas import load_p3_identities  # noqa: E402
from noise_final_core import load_embedding_cache, sha256_file  # noqa: E402
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


# Two nested doses per mechanism retain a safe and a strong view without
# counting every nested action as independent evidence.
REPRESENTATIVE_CELLS = {
    "E2-000": "candidate_gradient",
    "E2-003": "candidate_gradient",
    "E2-004": "role_confounder",
    "E2-009": "role_confounder",
    "E2-012": "acquisition_positive_gradient",
    "E2-015": "acquisition_positive_gradient",
    "E2-024": "acquisition_positive_gradient",
    "E2-027": "acquisition_positive_gradient",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e3_gradient_compatibility")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--p3-dir", type=Path, default=ROOT / "data/validation/g8r_p3_test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4_target_cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-actions", type=int, default=0, help="smoke only")
    return parser.parse_args()


def parse_tokens(value: object) -> np.ndarray:
    return np.asarray([int(token) for token in str(value).split(",") if token], dtype=np.int64)


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E4-M0: {args.output_dir}")
    report_path = args.e3_dir / "report.json"
    action_path = args.e3_dir / "action_gradient_summary.csv.gz"
    for path in (report_path, action_path, args.data, args.embedding_cache,
                 args.official_checkpoint, args.architecture_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    e3 = json.loads(report_path.read_text(encoding="utf-8"))
    if not e3.get("formal") or e3.get("status") != "noise_final_e3_gradient_compatibility_complete":
        raise RuntimeError("E4-M0 requires formal E3")
    if e3.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("E3 provenance does not forbid P2b")

    actions = pd.read_csv(action_path)
    actions = actions.loc[actions["cell_id"].astype(str).isin(REPRESENTATIVE_CELLS)].copy()
    actions["e4_family"] = actions["cell_id"].astype(str).map(REPRESENTATIVE_CELLS)
    if args.max_actions:
        actions = actions.head(args.max_actions).copy()
    actions = actions.sort_values(["e4_family", "query_ik14", "cell_id", "query_index"], kind="stable").reset_index(drop=True)
    if actions.empty:
        raise RuntimeError("no E4 representative actions")
    if set(actions["query_ik14"].astype(str)) & load_p3_identities(args.p3_dir):
        raise RuntimeError("P3 identity leakage in E4 targets")
    if set(actions["cell_id"].astype(str)) != set(REPRESENTATIVE_CELLS) and not args.max_actions:
        raise RuntimeError("formal E4 target cache is missing a representative cell")

    _, clean_embeddings, embedding_index = load_embedding_cache(args.embedding_cache)
    query_rows = actions["query_row"].astype(np.int64).to_numpy()
    if any(int(row) not in embedding_index for row in query_rows):
        raise RuntimeError("E4 query absent from official embedding cache")
    clean = clean_embeddings[np.asarray([embedding_index[int(row)] for row in query_rows], dtype=np.int64)]

    unique_rows = np.unique(query_rows)
    tensor_by_row: dict[int, torch.Tensor] = {}
    with h5py.File(args.data, "r") as handle:
        for position, row in enumerate(unique_rows, start=1):
            tensor_by_row[int(row)] = preprocess_spectrum(
                np.asarray(handle["spectrum"][int(row)]),
                float(handle["precursor_mz"][int(row)]), args.n_highest_peaks,
            )
            if position % 1000 == 0 or position == len(unique_rows):
                print(f"[E4-M0 spectra] {position:,}/{len(unique_rows):,}", flush=True)

    variants = []
    for row in actions.itertuples(index=False):
        attenuation = float(row.dose) if str(row.operator) == "attenuate" else 1.0
        variants.append(attenuate_sequence(
            tensor_by_row[int(row.query_row)], parse_tokens(row.target_tokens), attenuation,
        ))
    device = torch.device(args.device)
    model, checkpoint_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if checkpoint_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("E4-M0 requires official fine-tuned DreaMS")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    target = np.empty_like(clean, dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(variants), args.batch_size):
            right = min(left + args.batch_size, len(variants))
            target[left:right] = model(torch.stack(variants[left:right]).to(device)).float().cpu().numpy()
            if right % 5000 < args.batch_size or right == len(variants):
                print(f"[E4-M0 encode] {right:,}/{len(variants):,}", flush=True)
    del model, variants, tensor_by_row
    gc.collect()
    target /= np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-12, None)
    tangent = target - np.sum(target * clean, axis=1, keepdims=True) * clean
    magnitude = np.linalg.norm(tangent, axis=1)
    if not np.all(np.isfinite(target)) or np.any(magnitude <= 1e-8):
        raise RuntimeError("invalid E4 target vectors")

    # Equal total weight per identity within each family; family balancing is
    # applied by the E4 trainer, not baked into duplicated rows.
    key = actions["e4_family"].astype(str) + "|" + actions["query_ik14"].astype(str)
    count = key.groupby(key).transform("size").to_numpy(np.float32)
    identity_weight = 1.0 / count
    actions["identity_weight_within_family"] = identity_weight
    actions["target_tangent_magnitude_recomputed"] = magnitude

    temporary = Path(tempfile.mkdtemp(prefix="noise_e4_target_", dir=args.output_dir.parent))
    try:
        actions.to_csv(temporary / "actions.csv.gz", index=False, compression="gzip")
        np.save(temporary / "target_embedding_f16.npy", target.astype(np.float16))
        summary = actions.groupby("e4_family", as_index=False).agg(
            actions=("query_index", "size"), identities=("query_ik14", "nunique"),
            formulas=("query_formula", "nunique"), cells=("cell_id", "nunique"),
            mean_tangent=("target_tangent_magnitude_recomputed", "mean"),
        )
        summary.to_csv(temporary / "family_summary.csv", index=False)
        report = {
            "status": "noise_final_e4_target_cache_complete",
            "formal": args.max_actions == 0,
            "actions": int(len(actions)),
            "identities": int(actions["query_ik14"].nunique()),
            "formulas": int(actions["query_formula"].nunique()),
            "families": summary.to_dict("records"),
            "representative_cells": REPRESENTATIVE_CELLS,
            "contracts": {
                "targets_are_training_only": True,
                "inference_uses_clean_spectrum_only": True,
                "query_reference_encoder_shared": True,
                "P2b": "forbidden",
                "P3_identity_overlap": 0,
                "outcome_columns_used_as_sample_weights": False,
            },
            "provenance": {
                "e3_report_sha256": sha256_file(report_path),
                "e3_actions_sha256": sha256_file(action_path),
                "hdf5_sha256": sha256_file(args.data),
                "embedding_cache_sha256": sha256_file(args.embedding_cache),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": "E4-M0 freezes action targets; it is not a trained embedding result.",
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

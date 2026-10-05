"""Consume one frozen ChemAware release candidate on outer fold exactly once."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import formula_bootstrap, identity_balanced_queries  # noqa: E402
from train_chemaware_full_candidate_direct import evaluate, evaluation_rows  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError("outer seal has already been consumed or output exists")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "CHEMAWARE_RELEASE_CANDIDATE_FROZEN" or freeze.get("outer_evaluated") is not False:
        raise RuntimeError("outer evaluation requires an unused frozen release candidate")
    checkpoint_path = Path(freeze["candidate"]["checkpoint_path"])
    if sha256_file(checkpoint_path) != freeze["candidate"]["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal outer evaluation requires CUDA")

    # Claim the single-use output before reading any outer outcome.  A crash
    # leaves a consumed seal that must be investigated, never silently retried.
    args.output.mkdir(parents=True, exist_ok=False)
    consumed = {
        "status": "CHEMAWARE_OUTER_SEAL_CONSUMED",
        "freeze_sha256": sha256_file(args.freeze),
        "consumed_utc": datetime.now(timezone.utc).isoformat(),
        "retry_allowed": False,
    }
    (args.output / "SEAL_CONSUMED.json").write_text(json.dumps(consumed, indent=2), encoding="utf-8")

    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    protocol = freeze["outer_protocol"]
    outer_pool = np.flatnonzero(fold == int(protocol["fold"]))
    outer = identity_balanced_queries(
        outer_pool, body["query_ik14"], np.random.default_rng(int(protocol["selection_seed"])), 0,
    )
    eval_rows = evaluation_rows(body, outer)
    device = torch.device(args.device)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    if (checkpoint.get("status") != "chemaware_full_candidate_direct_arm_checkpoint"
            or checkpoint.get("causal_chemistry_pass") is not False
            or checkpoint.get("release_eligible") is not False):
        raise RuntimeError("checkpoint is not a quarantined single-arm artifact")
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    store = SpectrumStore(args.data, eval_rows, args.n_highest_peaks)
    encoded = encode_rows(model, store, eval_rows, device, args.eval_batch_size, False, "outer-once")
    adapted = np.array(official, copy=True)
    positions = np.asarray([row_position[int(row)] for row in eval_rows], dtype=np.int64)
    adapted[positions] = encoded
    result = evaluate(body, outer, official, adapted, row_position)
    preservation = float(np.mean(np.sum(encoded * official[positions], axis=1)))
    delta = (result["new_rank"] == 1).astype(float) - (result["old_rank"] == 1).astype(float)
    ci = formula_bootstrap(delta, body["query_formula"][outer], int(protocol["selection_seed"]) + 901,
                           int(protocol["bootstrap_draws"]))
    gates = {
        "recall_point_positive": result["summary"]["delta_recall1"] > 0,
        "formula_ci_strict_positive": ci["formula_cluster_bootstrap_95ci"][0] > 0,
        "corrected_exceeds_introduced": result["summary"]["corrected"] > result["summary"]["introduced"],
        "mrr_nonnegative": result["summary"]["delta_mrr"] >= 0,
        "preservation": preservation >= 0.995,
    }
    passed = bool(all(gates.values()))
    report = {
        "status": "CHEMAWARE_OUTER_RELEASE_PASS" if passed else "CHEMAWARE_OUTER_RELEASE_FAIL",
        "release_eligible": passed, "outer_evaluated": True,
        "freeze_sha256": sha256_file(args.freeze), "checkpoint_sha256": sha256_file(checkpoint_path),
        "initialization": initialization, "outer_queries": int(len(outer)),
        "outer_formulas": int(len(np.unique(body["query_formula"][outer]))),
        "summary": result["summary"], "formula_bootstrap": ci,
        "preservation": preservation, "gates": gates,
        "claim_limit": "One frozen checkpoint, one outer evaluation. A failure ends this release candidate.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "outer_per_query.npz", query=outer,
                        formula=body["query_formula"][outer], old_rank=result["old_rank"], new_rank=result["new_rank"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

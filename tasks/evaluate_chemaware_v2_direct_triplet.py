"""Evaluate direct-triplet checkpoints on the complete ChemAware role-3 graph.

This is a shared-embedding evaluation: every query and reference spectrum is
encoded independently, candidates are aggregated by max spectrum similarity,
and no chemical rule or candidate feature is available at inference time.
Formula role 4 is intentionally unreachable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import checkpoint_kind, torch_load_compat  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from chemaware_v2_triplet_eval_core import (  # noqa: E402
    evaluate_graph, paired_summary, summarize,
)


DEFAULT_MANIFEST = (
    ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
)
DEFAULT_POLICY = (
    ROOT / "data/validation/chemaware_multinull_deployment_safe_full_20260919"
    / "inner_policy.npz"
)
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_ARCHITECTURE = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Named checkpoint as NAME=PATH; repeat for multiple arms.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--formula-role", type=int, default=3)
    parser.add_argument(
        "--paired-reference", default=None,
        help="Optional checkpoint name for an additional paired comparison.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise ValueError("checkpoint name is empty")
    path = Path(raw_path)
    return name.strip(), path if path.is_absolute() else ROOT / path


def load_model(
    checkpoint: Path, official: Path, architecture: Path,
    device: torch.device, n_highest_peaks: int,
):
    package = torch_load_compat(checkpoint, map_location="cpu")
    kind = checkpoint_kind(package)
    if kind == "e1_identity":
        model, _ = load_base_model(
            official, architecture, device, n_highest_peaks,
        )
        model.backbone.load_state_dict(package["backbone_state_dict"], strict=True)
        model.head.load_state_dict(package["head_state_dict"], strict=True)
    elif kind in {"official_embedding", "official_embedding_slim"}:
        model, _ = load_base_model(
            checkpoint, architecture, device, n_highest_peaks,
        )
    else:
        raise ValueError(f"unsupported direct-triplet evaluation checkpoint: {kind}")
    model.eval()
    return model, kind


def required_rows(manifest: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    rows = [np.asarray(manifest["query_row"])[queries]]
    for query in queries:
        molecule_left, molecule_right = map(
            int, manifest["query_ptr"][int(query):int(query) + 2],
        )
        pair_left = int(manifest["molecule_ptr"][molecule_left])
        pair_right = int(manifest["molecule_ptr"][molecule_right])
        rows.append(np.asarray(manifest["pair_candidate_row"][pair_left:pair_right]))
    return np.unique(np.concatenate(rows).astype(np.int64))


@torch.no_grad()
def encode_rows(
    model, rows: np.ndarray, data: Path, device: torch.device,
    batch_size: int, n_highest_peaks: int,
) -> np.ndarray:
    output = []
    with h5py.File(data, "r") as handle:
        for left in range(0, len(rows), batch_size):
            batch_rows = rows[left:left + batch_size]
            raw = np.asarray(handle["spectrum"][batch_rows])
            precursor = np.asarray(handle["precursor_mz"][batch_rows], dtype=np.float64)
            spectra = torch.stack([
                preprocess_spectrum(spectrum, float(mz), n_highest_peaks)
                for spectrum, mz in zip(raw, precursor, strict=True)
            ]).to(device)
            output.append(model(spectra).cpu().numpy())
    encoded = np.concatenate(output).astype(np.float32, copy=False)
    norms = np.linalg.norm(encoded, axis=1)
    if not np.all(np.isfinite(encoded)) or np.max(np.abs(norms - 1.0)) > 2e-4:
        raise RuntimeError("evaluation embeddings are non-finite or not normalized")
    return encoded


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.policy, allow_pickle=False) as loaded:
        policy = {key: np.asarray(loaded[key]) for key in loaded.files}
    queries = np.asarray(policy["query"], dtype=np.int64)
    formulas = np.asarray(policy["formula"]).astype(str)
    expected_baseline = np.asarray(policy["baseline_rank"], dtype=np.int32)
    rows = required_rows(manifest, queries)

    reports = []
    ranks_by_name: dict[str, np.ndarray] = {}
    baseline_ranks = None
    baseline_embeddings = None
    for index, raw_checkpoint in enumerate(args.checkpoint):
        name, path = parse_checkpoint(raw_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Evaluating {name}: {path}", flush=True)
        model, kind = load_model(
            path, args.official_checkpoint, args.architecture_checkpoint,
            device, args.n_highest_peaks,
        )
        encoded = encode_rows(
            model, rows, args.data, device, args.batch_size, args.n_highest_peaks,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ranks, positive, negative, auc = evaluate_graph(encoded, rows, manifest, queries)
        metrics = summarize(ranks, positive, negative, auc)
        row = {"name": name, "checkpoint": str(path.resolve()), "kind": kind, "metrics": metrics}
        if index == 0:
            mismatch = int(np.sum(ranks != expected_baseline))
            row["frozen_ledger_rank_mismatches"] = mismatch
            if mismatch:
                raise RuntimeError(
                    f"official retrieval replay mismatch: {mismatch}; no trained comparison is valid"
                )
            baseline_ranks = ranks.copy()
            baseline_embeddings = encoded.copy()
        else:
            if baseline_ranks is None or baseline_embeddings is None:
                raise AssertionError("official baseline must be evaluated first")
            row["paired_vs_official"] = paired_summary(
                baseline_ranks, ranks, formulas, args.bootstrap_draws, args.seed + index,
            )
            row["mean_cosine_to_official_embedding"] = float(np.mean(np.sum(
                encoded * baseline_embeddings, axis=1,
            )))
        reports.append(row)
        ranks_by_name[name] = ranks.copy()
        print(json.dumps(row, indent=2), flush=True)

    if args.paired_reference is not None:
        if args.paired_reference not in ranks_by_name:
            raise ValueError(
                f"paired reference checkpoint is absent: {args.paired_reference}"
            )
        reference_rank = ranks_by_name[args.paired_reference]
        field = f"paired_vs_{args.paired_reference}"
        for index, row in enumerate(reports):
            if row["name"] == args.paired_reference:
                continue
            row[field] = paired_summary(
                reference_rank, ranks_by_name[row["name"]], formulas,
                args.bootstrap_draws, args.seed + 10_000 + index,
            )

    report = {
        "status": "CHEMAWARE_V2_DIRECT_TRIPLET_SHARED_EMBEDDING_EVALUATION_COMPLETE",
        "shared_embedding_result": True,
        "candidate_features_used_at_inference": False,
        "chemical_rules_used_at_inference": False,
        "formula_role": int(args.formula_role),
        "outer_role_4_accessed": False,
        "queries": int(len(queries)),
        "unique_spectrum_rows_encoded": int(len(rows)),
        "results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()

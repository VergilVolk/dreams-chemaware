"""Evaluate direct-triplet checkpoints on the complete ChemAware role-3 graph.

This is a shared-embedding evaluation: every query and reference spectrum is
encoded independently, candidates are aggregated by max spectrum similarity,
and no chemical rule or candidate feature is available at inference time.
Formula role 4 is intentionally unreachable.
"""
from __future__ import annotations

import argparse
import hashlib
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
    evaluate_graph, numerical_rank_replay_audit, paired_summary, summarize,
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
DEFAULT_ROLE_CONTRACT_DIR = (
    ROOT / "data/validation/chemaware_high_coverage_native/run_2340524/evidence"
)


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
        "--role-contract-dir", type=Path, default=DEFAULT_ROLE_CONTRACT_DIR,
        help=(
            "Frozen evidence directory used to prove that --policy is exactly "
            "the requested formula-role panel."
        ),
    )
    parser.add_argument(
        "--paired-reference", default=None,
        help="Optional checkpoint name for an additional paired comparison.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-tie-tolerance", type=float, default=5e-7)
    parser.add_argument("--maximum-replay-boundary-fraction", type=float, default=0.005)
    parser.add_argument("--maximum-replay-boundary-exclusions", type=int, default=3)
    parser.add_argument(
        "--expected-replay-exclusion-sha256",
        help="Optional frozen hash of the exact excluded manifest-query array.",
    )
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
    output = np.unique(np.concatenate(rows).astype(np.int64))
    if not len(output) or np.any(np.diff(output) <= 0):
        raise RuntimeError("manifest row registry is empty or not strictly increasing")
    return output


def validate_formula_role_policy(
    policy: dict[str, np.ndarray], formula_role: int, contract_dir: Path,
) -> Path:
    """Fail unless the supplied policy is the exact frozen panel for its role."""
    file_name = {
        2: "selection_triplet_evidence.npz",
        3: "confirmation_triplet_evidence.npz",
    }.get(int(formula_role))
    if file_name is None:
        raise ValueError("ChemAware direct-triplet evaluation supports formula roles 2 and 3")
    canonical_path = contract_dir / file_name
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)
    with np.load(canonical_path, allow_pickle=False) as loaded:
        canonical = {key: np.asarray(loaded[key]) for key in loaded.files}
    for key in ("query", "formula", "identity", "baseline_rank"):
        if key not in policy or key not in canonical:
            raise RuntimeError(f"formula-role contract lacks required field: {key}")
        observed = np.asarray(policy[key])
        expected = np.asarray(canonical[key])
        if observed.dtype.kind in "OUS" or expected.dtype.kind in "OUS":
            equal = np.array_equal(observed.astype(str), expected.astype(str))
        else:
            equal = np.array_equal(observed, expected)
        if not equal:
            raise RuntimeError(
                f"policy does not match frozen formula role {formula_role}: {key}"
            )
    return canonical_path.resolve()


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
    canonical_role_policy = validate_formula_role_policy(
        policy, args.formula_role, args.role_contract_dir,
    )
    queries = np.asarray(policy["query"], dtype=np.int64)
    formulas = np.asarray(policy["formula"]).astype(str)
    expected_baseline = np.asarray(policy["baseline_rank"], dtype=np.int32)
    rows = required_rows(manifest, queries)

    reports = []
    ranks_by_name: dict[str, np.ndarray] = {}
    baseline_ranks = None
    baseline_embeddings = None
    replay_stable = None
    replay_audit: list[dict[str, object]] = []
    for index, raw_checkpoint in enumerate(args.checkpoint):
        name, path = parse_checkpoint(raw_checkpoint)
        if index == 0 and name != "official":
            raise ValueError("the first --checkpoint must be named official")
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
        evaluated_queries = queries if replay_stable is None else queries[replay_stable]
        evaluated_formulas = formulas if replay_stable is None else formulas[replay_stable]
        ranks, positive, negative, auc = evaluate_graph(
            encoded, rows, manifest, evaluated_queries,
        )
        if index == 0:
            mismatch = int(np.sum(ranks != expected_baseline))
            replay_stable, replay_audit = numerical_rank_replay_audit(
                encoded, rows, manifest, queries, expected_baseline, ranks,
                tie_tolerance=args.replay_tie_tolerance,
                maximum_fraction=args.maximum_replay_boundary_fraction,
                maximum_count=args.maximum_replay_boundary_exclusions,
            )
            if mismatch:
                evaluated_queries = queries[replay_stable]
                evaluated_formulas = formulas[replay_stable]
                ranks, positive, negative, auc = evaluate_graph(
                    encoded, rows, manifest, evaluated_queries,
                )
        metrics = summarize(ranks, positive, negative, auc)
        row = {
            "name": name, "checkpoint": str(path.resolve()), "kind": kind,
            "metrics": metrics,
            "recall1_boundary_ties_at_tolerance": int(np.sum(
                np.abs(np.asarray(negative) - np.asarray(positive))
                <= args.replay_tie_tolerance
            )),
        }
        if index == 0:
            row["frozen_ledger_rank_mismatches"] = mismatch
            row["numerical_boundary_exclusions"] = int(np.sum(~replay_stable))
            row["numerical_boundary_audit"] = replay_audit
            baseline_ranks = ranks.copy()
            baseline_embeddings = encoded.copy()
        else:
            if baseline_ranks is None or baseline_embeddings is None:
                raise AssertionError("official baseline must be evaluated first")
            row["paired_vs_official"] = paired_summary(
                baseline_ranks, ranks, evaluated_formulas,
                args.bootstrap_draws, args.seed + index,
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
        if replay_stable is None:
            raise AssertionError("official replay audit was not executed")
        paired_formulas = formulas[replay_stable]
        for index, row in enumerate(reports):
            if row["name"] == args.paired_reference:
                continue
            row[field] = paired_summary(
                reference_rank, ranks_by_name[row["name"]], paired_formulas,
                args.bootstrap_draws, args.seed + 10_000 + index,
            )

    if replay_stable is None:
        raise AssertionError("official replay audit was not executed")
    excluded_queries = np.asarray(queries[~replay_stable], dtype=np.int64)
    exclusion_digest = hashlib.sha256()
    exclusion_digest.update(excluded_queries.tobytes(order="C"))
    exclusion_sha256 = exclusion_digest.hexdigest()
    if (
        args.expected_replay_exclusion_sha256 is not None
        and exclusion_sha256 != args.expected_replay_exclusion_sha256
    ):
        raise RuntimeError(
            "official replay exclusion set drifted: "
            f"expected={args.expected_replay_exclusion_sha256} observed={exclusion_sha256}"
        )
    report = {
        "status": "CHEMAWARE_V2_DIRECT_TRIPLET_SHARED_EMBEDDING_EVALUATION_COMPLETE",
        "shared_embedding_result": True,
        "candidate_features_used_at_inference": False,
        "chemical_rules_used_at_inference": False,
        "formula_role": int(args.formula_role),
        "formula_role_contract_passed": True,
        "canonical_role_policy": str(canonical_role_policy),
        "outer_role_4_accessed": False,
        "queries_before_numerical_boundary_exclusion": int(len(queries)),
        "queries": int(np.sum(replay_stable)) if replay_stable is not None else 0,
        "numerical_boundary_exclusions": int(np.sum(~replay_stable)) if replay_stable is not None else 0,
        "numerical_boundary_excluded_manifest_queries": excluded_queries.tolist(),
        "numerical_boundary_exclusion_sha256": exclusion_sha256,
        "maximum_replay_boundary_exclusions": args.maximum_replay_boundary_exclusions,
        "numerical_boundary_policy": (
            "exclude only frozen-rank mismatches whose expected rank lies inside "
            "the explicit float32 score-tie interval; fail closed otherwise"
        ),
        "unique_spectrum_rows_encoded": int(len(rows)),
        "results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()

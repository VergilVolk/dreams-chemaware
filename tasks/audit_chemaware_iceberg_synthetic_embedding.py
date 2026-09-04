"""Test ICEBERG synthetic spectra as high-bandwidth DreaMS teacher vectors.

The structure-conditioned spectra are encoded by the frozen official DreaMS
model.  No model is trained.  Two complementary directions are evaluated:

1. experimental query -> one synthetic spectrum per candidate structure;
2. positive synthetic spectrum -> the original experimental reference library.

The second direction is the necessary gate for using a positive synthetic
embedding as a training-only target for a spectrum-only shared encoder.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

import pilot_multilevel_factor_activations as multi  # noqa: E402
from e1_checkpoint_io import checkpoint_kind, official_head_state  # noqa: E402
from noise_final_core import CandidateGraph, strict_rank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz"),
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/tokens"),
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_teacher_headroom_inner_v1"),
    )
    parser.add_argument(
        "--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5")
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=Path("dreams/models/pretrained/ssl_model_server.pt"),
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=Path("data/e1/official_embedding_slim.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_synthetic_embedding_inner_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_permute(predicted: np.ndarray, seed: int) -> np.ndarray:
    output = predicted.copy()
    for index in range(len(output)):
        nonzero = np.flatnonzero(output[index] > 0)
        if len(nonzero) < 2:
            continue
        rng = np.random.default_rng(seed + 1_000_003 * (index + 1))
        permutation = rng.permutation(len(nonzero))
        if np.array_equal(permutation, np.arange(len(nonzero))):
            permutation = np.roll(permutation, 1)
        output[index, nonzero] = output[index, nonzero][permutation]
    return output


def synthetic_tensor(
    binned: np.ndarray,
    precursor_mz: float,
    n_highest: int,
    max_mz: float,
) -> torch.Tensor:
    masses = np.linspace(0.0, 1500.0, 15_000, dtype=np.float32)
    valid = np.flatnonzero(
        (binned > 0) & (masses > 0) & (masses <= max_mz)
    )
    if not len(valid):
        raise RuntimeError("ICEBERG produced no DreaMS-valid peaks")
    if len(valid) > n_highest:
        keep = np.argsort(binned[valid], kind="stable")[-n_highest:]
        valid = valid[keep]
    valid = np.sort(valid)
    peaks = np.stack((masses[valid], binned[valid]), axis=1).astype(np.float32)
    if len(peaks) < n_highest:
        peaks = np.pad(peaks, ((0, n_highest - len(peaks)), (0, 0)))
    maximum = float(peaks[:, 1].max())
    if maximum > 0:
        peaks[:, 1] /= maximum
    precursor = np.asarray([[precursor_mz, 1.1]], dtype=np.float32)
    return torch.from_numpy(np.vstack((precursor, peaks)))


@torch.no_grad()
def encode_synthetic(
    predictions: np.ndarray,
    precursor: np.ndarray,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    batch_size: int,
    n_highest: int,
    max_mz: float,
) -> np.ndarray:
    dtype = next(backbone.parameters()).dtype
    output = np.empty((len(predictions), int(backbone.d_model)), dtype=np.float32)
    for left in range(0, len(predictions), batch_size):
        right = min(left + batch_size, len(predictions))
        spectra = torch.stack(
            [
                synthetic_tensor(predictions[index], precursor[index], n_highest, max_mz)
                for index in range(left, right)
            ]
        ).to(dtype=dtype)
        contextual = backbone(spectra, None)
        output[left:right] = F.normalize(head(contextual[:, 0].float()), dim=-1).numpy()
        print(f"encoded synthetic spectra {right}/{len(predictions)}", flush=True)
    return output


def ranks_margins_from_flat_scores(
    scores: np.ndarray, query_ptr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ranks = []
    margins = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        values = scores[int(left) : int(right)]
        ranks.append(strict_rank(values))
        margins.append(float(values[0] - np.max(values[1:])))
    return np.asarray(ranks, dtype=np.int32), np.asarray(margins, dtype=np.float32)


def summarize(ranks: np.ndarray, margins: np.ndarray) -> dict:
    return {
        "queries": int(len(ranks)),
        "hit1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mean_positive_margin": float(np.mean(margins)),
        "median_positive_margin": float(np.median(margins)),
    }


def bootstrap(
    difference: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    unique = np.unique(formulas)
    values = np.asarray(
        [np.mean(difference[formulas == value]) for value in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        estimates[draw] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "formula_macro_advantage": float(np.mean(values)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def experimental_reference_ranks(
    synthetic_query: np.ndarray,
    selected_queries: np.ndarray,
    graph: CandidateGraph,
    row_position: dict[int, int],
    official: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ranks = []
    margins = []
    for embedding, query in zip(synthetic_query, selected_queries):
        _, candidate_rows, local_ptr, _ = graph.query_block(int(query))
        candidate = official[[row_position[int(row)] for row in candidate_rows]]
        pair_scores = candidate @ embedding
        molecule_scores = np.maximum.reduceat(pair_scores, local_ptr[:-1])
        ranks.append(strict_rank(molecule_scores))
        margins.append(float(molecule_scores[0] - np.max(molecule_scores[1:])))
    return np.asarray(ranks, dtype=np.int32), np.asarray(margins, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.n_highest_peaks, args.bootstrap_draws) < 1:
        raise ValueError("batch, peak and bootstrap arguments must be positive")
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    graph = CandidateGraph(args.graph)
    teacher_report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if teacher_report.get("status") != "PASS":
        raise RuntimeError("teacher directory did not pass its headroom gate")
    if teacher_report["inputs"].get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("teacher/graph provenance mismatch")
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    query_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    predictions = np.load(args.teacher_dir / "iceberg_predictions_f16.npy").astype(np.float32)
    if query_ptr.shape != (len(selected) + 1,) or query_ptr[-1] != len(predictions):
        raise RuntimeError("teacher arrays are not aligned")
    token_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy").astype(np.float32)
    row_position = {int(row): index for index, row in enumerate(token_rows)}
    experimental_query = official[[row_position[int(graph.query_row[q])] for q in selected]]
    with h5py.File(args.hdf5, "r") as h5:
        query_precursor = np.asarray(
            [float(h5["precursor_mz"][int(graph.query_row[q])]) for q in selected],
            dtype=np.float32,
        )
    flat_precursor = np.repeat(query_precursor, np.diff(query_ptr))

    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_checkpoint = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    if checkpoint_kind(official_checkpoint) not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("unsupported official checkpoint")
    backbone = multi.reconstruct_backbone(
        raw, multi.official_backbone_state(official_checkpoint), torch.device("cpu")
    )
    backbone.eval()
    head = torch.nn.Linear(int(backbone.d_model), int(backbone.d_model), bias=True)
    head.load_state_dict(official_head_state(official_checkpoint), strict=True)
    head.eval()
    for parameter in list(backbone.parameters()) + list(head.parameters()):
        parameter.requires_grad_(False)
    max_mz = float(raw["args"]["max_mz"])
    correct_embedding = encode_synthetic(
        predictions,
        flat_precursor,
        backbone,
        head,
        args.batch_size,
        args.n_highest_peaks,
        max_mz,
    )
    permuted_predictions = peak_permute(
        predictions, int(teacher_report["protocol"]["seed"]) + 41
    )
    permuted_embedding = encode_synthetic(
        permuted_predictions,
        flat_precursor,
        backbone,
        head,
        args.batch_size,
        args.n_highest_peaks,
        max_mz,
    )

    correct_scores = np.empty(len(predictions), dtype=np.float32)
    swapped_scores = np.empty(len(predictions), dtype=np.float32)
    permuted_scores = np.empty(len(predictions), dtype=np.float32)
    positive_correct = []
    positive_swapped = []
    positive_permuted = []
    for index, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        query_embedding = experimental_query[index]
        correct_scores[left:right] = correct_embedding[left:right] @ query_embedding
        swapped = np.roll(correct_embedding[left:right], 1, axis=0)
        swapped_scores[left:right] = swapped @ query_embedding
        permuted_scores[left:right] = permuted_embedding[left:right] @ query_embedding
        positive_correct.append(correct_embedding[left])
        positive_swapped.append(swapped[0])
        positive_permuted.append(permuted_embedding[left])

    direct_correct_rank, direct_correct_margin = ranks_margins_from_flat_scores(
        correct_scores, query_ptr
    )
    direct_swapped_rank, direct_swapped_margin = ranks_margins_from_flat_scores(
        swapped_scores, query_ptr
    )
    direct_permuted_rank, direct_permuted_margin = ranks_margins_from_flat_scores(
        permuted_scores, query_ptr
    )
    target_correct_rank, target_correct_margin = experimental_reference_ranks(
        np.asarray(positive_correct), selected, graph, row_position, official
    )
    target_swapped_rank, target_swapped_margin = experimental_reference_ranks(
        np.asarray(positive_swapped), selected, graph, row_position, official
    )
    target_permuted_rank, target_permuted_margin = experimental_reference_ranks(
        np.asarray(positive_permuted), selected, graph, row_position, official
    )
    formulas = graph.query_formula[selected]
    target_hit_swap = bootstrap(
        (target_correct_rank == 1).astype(float) - (target_swapped_rank == 1).astype(float),
        formulas,
        20260904,
        args.bootstrap_draws,
    )
    target_hit_perm = bootstrap(
        (target_correct_rank == 1).astype(float) - (target_permuted_rank == 1).astype(float),
        formulas,
        20260905,
        args.bootstrap_draws,
    )
    target_margin_swap = bootstrap(
        target_correct_margin - target_swapped_margin,
        formulas,
        20260906,
        args.bootstrap_draws,
    )
    target_margin_perm = bootstrap(
        target_correct_margin - target_permuted_margin,
        formulas,
        20260907,
        args.bootstrap_draws,
    )
    passed = bool(
        target_hit_swap["formula_cluster_bootstrap_95ci"][0] > 0
        and target_hit_perm["formula_cluster_bootstrap_95ci"][0] > 0
        and target_margin_swap["formula_cluster_bootstrap_95ci"][0] > 0
        and target_margin_perm["formula_cluster_bootstrap_95ci"][0] > 0
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "Positive synthetic DreaMS embeddings are viable high-bandwidth distillation targets."
            if passed
            else "Synthetic DreaMS target bridge failed; do not train embedding regression on it."
        ),
        "scope": {
            "no_training": True,
            "official_dreams_frozen": True,
            "teacher_structure_training_only": True,
            "massspecgym_overlap_warning": True,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(
                args.teacher_dir / "iceberg_predictions_f16.npy"
            ),
            "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
        "protocol": {
            "queries": int(len(selected)),
            "candidate_synthetic_spectra": int(len(predictions)),
            "n_highest_peaks": args.n_highest_peaks,
            "max_mz": max_mz,
            "synthetic_preprocessing": (
                "ICEBERG 0-1500 binned prediction restricted to official DreaMS max_mz, "
                "top intensity peaks retained, m/z sorted, max-intensity normalized"
            ),
            "target_gate": (
                "positive-synthetic to experimental-reference correct arm must beat both controls "
                "on formula-cluster-bootstrap Hit@1 and margin lower bounds"
            ),
        },
        "metrics": {
            "experimental_query_to_synthetic_candidates": {
                "correct": summarize(direct_correct_rank, direct_correct_margin),
                "candidate_swapped": summarize(direct_swapped_rank, direct_swapped_margin),
                "peak_permuted": summarize(direct_permuted_rank, direct_permuted_margin),
            },
            "positive_synthetic_to_experimental_references": {
                "correct": summarize(target_correct_rank, target_correct_margin),
                "candidate_swapped": summarize(target_swapped_rank, target_swapped_margin),
                "peak_permuted": summarize(target_permuted_rank, target_permuted_margin),
                "paired_formula_bootstrap": {
                    "hit1_correct_minus_candidate_swapped": target_hit_swap,
                    "hit1_correct_minus_peak_permuted": target_hit_perm,
                    "margin_correct_minus_candidate_swapped": target_margin_swap,
                    "margin_correct_minus_peak_permuted": target_margin_perm,
                },
            },
        },
    }
    np.save(args.output / "correct_synthetic_embeddings_f32.npy", correct_embedding)
    np.save(args.output / "peak_permuted_synthetic_embeddings_f32.npy", permuted_embedding)
    np.savez_compressed(
        args.output / "scores_and_ranks.npz",
        direct_correct_score=correct_scores,
        direct_correct_rank=direct_correct_rank,
        direct_correct_margin=direct_correct_margin,
        direct_swapped_score=swapped_scores,
        direct_swapped_rank=direct_swapped_rank,
        direct_swapped_margin=direct_swapped_margin,
        direct_permuted_score=permuted_scores,
        direct_permuted_rank=direct_permuted_rank,
        direct_permuted_margin=direct_permuted_margin,
        target_correct_rank=target_correct_rank,
        target_correct_margin=target_correct_margin,
        target_swapped_rank=target_swapped_rank,
        target_swapped_margin=target_swapped_margin,
        target_permuted_rank=target_permuted_rank,
        target_permuted_margin=target_permuted_margin,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False), flush=True)
    print(f"decision={report['status']} report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

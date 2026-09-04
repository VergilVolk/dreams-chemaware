"""Full-graph scan of real-positive-guided, peak-level noise actions.

This is a training-action discovery experiment, not a trained model result.
For every P3-disjoint query we build a consensus from up to three real spectra
of the same molecule.  We then modify only peaks already present in the query:

* matched_intensity_transport: move matched-peak intensity toward the real
  positive consensus;
* prevalence_attenuation: softly attenuate peaks absent from positive repeats;
* consensus_projection: combine the two operations.

The same operations are repeated toward the current hardest wrong molecule as
a direction-matched negative control.  Every fixed cell is evaluated on all
23,876 strict-10ppm query graphs, so corrected and introduced errors are exact.
Outcome-aware unions are explicitly labelled headroom and are never a model
claim or a deployable policy.
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
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, sha256_file, strict_rank,
)
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


FAMILIES = (
    "matched_intensity_transport",
    "prevalence_attenuation",
    "consensus_projection",
)
REFERENCE_KINDS = ("positive", "hardest_wrong_control")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--uncovered-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_uncovered_errors")
    parser.add_argument("--fivepoint", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_fivepoint_headroom.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_positive_guided_matrix")
    parser.add_argument("--doses", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--positive-references", type=int, default=3)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-queries", type=int, default=0, help="smoke only; formal is 0")
    return parser.parse_args()


def top_rows(scores: np.ndarray, rows: np.ndarray, ptr: np.ndarray, molecule: int, count: int) -> np.ndarray:
    left, right = map(int, ptr[molecule:molecule + 2])
    if right <= left:
        raise RuntimeError("candidate molecule has no reference spectrum")
    order = np.argsort(-np.asarray(scores[left:right]), kind="mergesort")[:count] + left
    return np.asarray(rows[order], dtype=np.int64)


def baseline_detail(graph: CandidateGraph, query: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    pair_slice, rows, ptr, molecule_left = graph.query_block(query)
    pair_scores = graph.features[pair_slice, graph.dreams_column]
    molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
    rank = strict_rank(molecule_scores)
    wrong = int(np.argmax(molecule_scores[1:])) + 1
    margin = float(molecule_scores[0] - molecule_scores[wrong])
    return pair_scores, rows, ptr, molecule_left, rank, margin


def reference_profile(query: torch.Tensor, references: list[torch.Tensor], tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-query-token reference prevalence and median matched intensity."""
    q = query.detach().cpu().numpy()
    prevalence = np.zeros(len(q), dtype=np.float32)
    target = q[:, 1].astype(np.float32, copy=True)
    valid_q = (q[:, 0] > 0) & (q[:, 1] > 0)
    valid_q[0] = False
    for token in np.flatnonzero(valid_q):
        hits: list[float] = []
        for reference in references:
            ref = reference.detach().cpu().numpy()
            valid = (ref[1:, 0] > 0) & (ref[1:, 1] > 0)
            mz = ref[1:, 0][valid]
            intensity = ref[1:, 1][valid]
            if not len(mz):
                continue
            distance = np.abs(mz - q[token, 0])
            nearest = int(np.argmin(distance))
            if float(distance[nearest]) <= tolerance:
                hits.append(float(intensity[nearest]))
        prevalence[token] = len(hits) / max(len(references), 1)
        if hits:
            target[token] = float(np.median(hits))
    return prevalence, target


def apply_action(
    clean: torch.Tensor, prevalence: np.ndarray, target: np.ndarray,
    family: str, dose: float,
) -> torch.Tensor:
    if family not in FAMILIES or not 0 < dose <= 1:
        raise ValueError("invalid positive-guided action")
    output = clean.clone()
    values = output.numpy()
    valid = (values[:, 0] > 0) & (values[:, 1] > 0)
    valid[0] = False
    q = values[:, 1].copy()
    if family == "matched_intensity_transport":
        desired = q.copy()
        matched = valid & (prevalence > 0)
        desired[matched] = target[matched]
    elif family == "prevalence_attenuation":
        desired = q * prevalence
    else:
        desired = prevalence * target
    values[valid, 1] = (1.0 - dose) * q[valid] + dose * desired[valid]
    maximum = float(np.max(values[1:, 1]))
    if maximum > 0:
        values[1:, 1] /= maximum
    values[0, 1] = float(clean[0, 1])
    return output


def molecule_rank_margin(scores: np.ndarray, ptr: np.ndarray) -> tuple[int, float]:
    molecule_scores = np.maximum.reduceat(np.asarray(scores, dtype=float), ptr[:-1])
    rank = strict_rank(molecule_scores)
    return rank, float(molecule_scores[0] - np.max(molecule_scores[1:]))


def cluster_ci(values: np.ndarray, clusters: np.ndarray, repeats: int, seed: int) -> list[float]:
    frame = pd.DataFrame({"value": np.asarray(values, float), "cluster": np.asarray(clusters, str)})
    grouped = frame.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        chosen = rng.integers(0, len(sums), len(sums))
        draws[index] = sums[chosen].sum() / counts[chosen].sum()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def create_dataset(handle: h5py.File, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    chunks = (min(max(len(array), 1), 262_144),) + array.shape[1:]
    handle.create_dataset(name, data=array, compression="gzip", compression_opts=4, shuffle=True, chunks=chunks)


def main() -> None:
    args = arguments()
    formal = args.max_queries == 0
    required = [
        args.graph, args.embeddings, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.uncovered_dir / "uncovered_errors.csv.gz",
        args.uncovered_dir / "report.json", args.fivepoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite positive-guided matrix: {args.output_dir}")
    doses = np.asarray(args.doses, dtype=np.float32)
    if len(doses) != len(np.unique(doses)) or np.any((doses <= 0) | (doses > 1)):
        raise ValueError("doses must be unique and in (0, 1]")
    if args.positive_references < 1 or args.batch_size < 1:
        raise ValueError("positive-references and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph = CandidateGraph(args.graph)
    n_queries = graph.n_queries if formal else min(args.max_queries, graph.n_queries)
    if formal and n_queries != 23876:
        raise RuntimeError(f"formal matrix expects 23,876 queries, observed {n_queries}")
    embedding_rows, embeddings, embedding_index = load_embedding_cache(args.embeddings)
    needed = set(map(int, graph.query_row[:n_queries]))
    for query in range(n_queries):
        _, rows, _, _ = graph.query_block(query)
        needed.update(map(int, rows))
    missing_embedding = needed - set(embedding_index)
    if missing_embedding:
        raise RuntimeError(f"embedding cache misses {len(missing_embedding)} graph rows")

    baseline_rank = np.empty(n_queries, dtype=np.int16)
    baseline_margin = np.empty(n_queries, dtype=np.float32)
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    candidate_indices: list[np.ndarray] = []
    candidate_ptr: list[np.ndarray] = []
    for query in range(n_queries):
        scores, rows, ptr, _, rank, margin = baseline_detail(graph, query)
        molecule_scores = np.maximum.reduceat(scores, ptr[:-1])
        wrong = int(np.argmax(molecule_scores[1:])) + 1
        baseline_rank[query] = rank
        baseline_margin[query] = margin
        positive_rows.append(top_rows(scores, rows, ptr, 0, args.positive_references))
        negative_rows.append(top_rows(scores, rows, ptr, wrong, args.positive_references))
        candidate_indices.append(np.asarray([embedding_index[int(row)] for row in rows], dtype=np.int64))
        candidate_ptr.append(np.asarray(ptr, dtype=np.int64))

    tensor_rows = set(map(int, graph.query_row[:n_queries]))
    for rows in positive_rows + negative_rows:
        tensor_rows.update(map(int, rows))
    tensors: dict[int, torch.Tensor] = {}
    with h5py.File(args.data, "r") as handle:
        for position, row in enumerate(sorted(tensor_rows), start=1):
            tensors[row] = preprocess_spectrum(
                np.asarray(handle["spectrum"][row]), float(handle["precursor_mz"][row]),
                args.n_highest_peaks,
            )
            if position % 2500 == 0 or position == len(tensor_rows):
                print(f"[positive-guided spectra] {position:,}/{len(tensor_rows):,}", flush=True)

    query_tensors = [tensors[int(row)] for row in graph.query_row[:n_queries]]
    profiles = np.empty((n_queries, 2, 2, args.n_highest_peaks + 1), dtype=np.float32)
    for query in range(n_queries):
        for kind, rows in enumerate((positive_rows[query], negative_rows[query])):
            prevalence, target = reference_profile(
                query_tensors[query], [tensors[int(row)] for row in rows], args.fragment_tolerance,
            )
            profiles[query, kind, 0] = prevalence
            profiles[query, kind, 1] = target
        if (query + 1) % 2500 == 0 or query + 1 == n_queries:
            print(f"[positive-guided profiles] {query + 1:,}/{n_queries:,}", flush=True)

    model, checkpoint_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if checkpoint_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("positive-guided scan requires official DreaMS weights")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    # Fresh-forward reproduction is mandatory because all effects below use
    # this executor while graph baselines were cached earlier.
    clean_vectors = np.empty((n_queries, embeddings.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, n_queries, args.batch_size):
            right = min(left + args.batch_size, n_queries)
            clean_vectors[left:right] = model(torch.stack(query_tensors[left:right]).to(device)).float().cpu().numpy()
    cached_query = np.stack([embeddings[embedding_index[int(row)]] for row in graph.query_row[:n_queries]])
    preservation = np.sum(clean_vectors * cached_query, axis=1)
    fresh_rank = np.empty(n_queries, dtype=np.int16)
    fresh_margin = np.empty(n_queries, dtype=np.float32)
    for query in range(n_queries):
        scores = clean_vectors[query] @ embeddings[candidate_indices[query]].T
        fresh_rank[query], fresh_margin[query] = molecule_rank_margin(scores, candidate_ptr[query])
    mismatch = int(np.sum(fresh_rank != baseline_rank))
    if formal and (float(np.quantile(preservation, 0.01)) < 0.999 or mismatch > 3):
        raise RuntimeError(
            f"fresh official reproduction failed: p01={np.quantile(preservation, 0.01):.8f}, "
            f"rank mismatches={mismatch}"
        )

    cells = [(family, float(dose), kind) for family in FAMILIES for dose in doses for kind in range(2)]
    n_cells = len(cells)
    result_rank = np.empty((n_queries, n_cells), dtype=np.int16)
    result_margin = np.empty((n_queries, n_cells), dtype=np.float32)
    total = n_queries * n_cells
    with torch.inference_mode():
        for left in range(0, total, args.batch_size):
            right = min(left + args.batch_size, total)
            linear = np.arange(left, right, dtype=np.int64)
            queries = linear // n_cells
            local_cells = linear % n_cells
            batch = []
            for query, cell in zip(queries, local_cells):
                family, dose, kind = cells[int(cell)]
                batch.append(apply_action(
                    query_tensors[int(query)], profiles[int(query), kind, 0],
                    profiles[int(query), kind, 1], family, dose,
                ))
            vectors = model(torch.stack(batch).to(device)).float().cpu().numpy()
            for query in np.unique(queries):
                offsets = np.flatnonzero(queries == query)
                scores = vectors[offsets] @ embeddings[candidate_indices[int(query)]].T
                for offset, pair_scores in zip(offsets, scores):
                    rank, margin = molecule_rank_margin(pair_scores, candidate_ptr[int(query)])
                    result_rank[int(query), int(local_cells[offset])] = rank
                    result_margin[int(query), int(local_cells[offset])] = margin
            if right % 25000 < args.batch_size or right == total:
                print(f"[positive-guided exact] {right:,}/{total:,}", flush=True)

    del model
    gc.collect()
    formulas = graph.query_formula[:n_queries]
    near = graph.query_has_near[:n_queries]
    # Numerical tie-boundary mismatches between the archived graph and this
    # fresh executor are reported but excluded from every action effect.  They
    # must never be counted as either a correction or an introduced error.
    stable = fresh_rank == baseline_rank
    baseline_correct = fresh_rank == 1
    summaries: list[dict] = []
    passing_cells: list[str] = []
    for family_index, family in enumerate(FAMILIES):
        for dose_index, dose in enumerate(doses):
            target_cell = cells.index((family, float(dose), 0))
            control_cell = cells.index((family, float(dose), 1))
            target_correct = result_rank[:, target_cell] == 1
            control_correct = result_rank[:, control_cell] == 1
            delta = target_correct.astype(float) - baseline_correct.astype(float)
            specificity = target_correct.astype(float) - control_correct.astype(float)
            corrected = int(np.sum(stable & ~baseline_correct & target_correct))
            introduced = int(np.sum(stable & baseline_correct & ~target_correct))
            near_eval = near & stable
            near_delta = float(np.mean(delta[near_eval])) if np.any(near_eval) else float("nan")
            ci = cluster_ci(
                delta[stable], formulas[stable], args.bootstrap,
                args.seed + family_index * 100 + dose_index,
            )
            specificity_ci = cluster_ci(
                specificity[stable], formulas[stable], args.bootstrap,
                args.seed + 10000 + family_index * 100 + dose_index,
            )
            cell_id = f"{family}|dose={float(dose):.2f}"
            passed = bool(
                ci[0] > 0 and corrected > introduced and corrected - 2 * introduced > 0
                and near_delta >= 0 and specificity_ci[0] > 0
            )
            if passed:
                passing_cells.append(cell_id)
            summaries.append({
                "cell_id": cell_id,
                "family": family,
                "dose": float(dose),
                "queries": int(np.sum(stable)),
                "excluded_reproduction_mismatches": int(np.sum(~stable)),
                "baseline_recall1": float(np.mean(baseline_correct[stable])),
                "recall1": float(np.mean(target_correct[stable])),
                "delta_recall1": float(np.mean(delta[stable])),
                "corrected": corrected,
                "introduced": introduced,
                "net": corrected - introduced,
                "risk_net_lambda2": corrected - 2 * introduced,
                "near_delta_recall1": near_delta,
                "mean_margin_delta": float(np.mean(result_margin[stable, target_cell] - fresh_margin[stable])),
                "formula_cluster_delta_ci_low": ci[0],
                "formula_cluster_delta_ci_high": ci[1],
                "positive_minus_wrong_control_top1": float(np.mean(specificity[stable])),
                "specificity_formula_ci_low": specificity_ci[0],
                "specificity_formula_ci_high": specificity_ci[1],
                "pass_to_training_policy_screen": passed,
            })

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["risk_net_lambda2", "delta_recall1"], ascending=False, kind="stable",
    )
    positive_cell_indices = [index for index, (_, _, kind) in enumerate(cells) if kind == 0]
    official_error = baseline_rank != 1
    oracle_recoverable = official_error & np.any(result_rank[:, positive_cell_indices] == 1, axis=1)
    oracle_recoverable &= stable
    uncovered_frame = pd.read_csv(args.uncovered_dir / "uncovered_errors.csv.gz")
    uncovered_queries = set(map(int, uncovered_frame["query_index"]))
    if formal and len(uncovered_queries) != 883:
        raise RuntimeError(f"expected 883 uncovered errors, observed {len(uncovered_queries)}")
    uncovered_mask = np.asarray([query in uncovered_queries for query in range(n_queries)], dtype=bool)
    newly_recoverable = oracle_recoverable & uncovered_mask
    new_queries = np.flatnonzero(newly_recoverable)
    fivepoint = json.loads(args.fivepoint.read_text(encoding="utf-8"))
    pn_union = int(fivepoint["p_n_union_recoverable_queries"])
    total_union = pn_union + int(np.sum(newly_recoverable))
    required = int(fivepoint["required_net_corrections"])
    action_manifest = pd.DataFrame({
        "query_index": np.arange(n_queries, dtype=np.int64),
        "query_row": graph.query_row[:n_queries],
        "query_ik14": graph.query_ik14[:n_queries],
        "query_formula": graph.query_formula[:n_queries],
        "positive_reference_rows": [";".join(map(str, rows)) for rows in positive_rows],
        "hardest_wrong_reference_rows": [";".join(map(str, rows)) for rows in negative_rows],
        "baseline_rank": baseline_rank,
        "baseline_margin": baseline_margin,
        "positive_guided_oracle_recoverable": oracle_recoverable,
        "new_beyond_pn": newly_recoverable,
    })
    report = {
        "status": "noise_final_positive_guided_matrix_complete",
        "formal": formal,
        "queries": n_queries,
        "official_errors": int(np.sum(official_error)),
        "cells": len(FAMILIES) * len(doses),
        "direction_controls": len(FAMILIES) * len(doses),
        "encoded_variants": total,
        "passing_cells": passing_cells,
        "best_fixed_cell": json.loads(summary_frame.iloc[0].to_json()),
        "headroom_only": {
            "positive_guided_oracle_recoverable_errors": int(np.sum(oracle_recoverable)),
            "new_unique_errors_beyond_frozen_pn": int(np.sum(newly_recoverable)),
            "p_n_union_before": pn_union,
            "expanded_union": total_union,
            "required_for_five_points": required,
            "remaining_to_five_points": max(required - total_union, 0),
            "reaches_five_points": bool(total_union >= required),
            "reaches_recommended_350_buffer": bool(np.sum(newly_recoverable) >= 350),
        },
        "fresh_official_reproduction": {
            "preservation_mean": float(np.mean(preservation)),
            "preservation_p01": float(np.quantile(preservation, 0.01)),
            "rank_mismatches": mismatch,
        },
        "contracts": {
            "modifies_only_existing_query_peaks": True,
            "positive_references_are_real_same_identity_spectra": True,
            "hardest_wrong_reference_is_direction_control": True,
            "fixed_cells_evaluated_on_full_graph": True,
            "outcome_used_only_for_oracle_headroom_union": True,
            "P2b_forbidden": True,
            "shared_embedding_training_result": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embeddings),
            "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "uncovered_report_sha256": sha256_file(args.uncovered_dir / "report.json"),
            "fivepoint_sha256": sha256_file(args.fivepoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Fixed cells are frozen-encoder action outcomes. The union is an outcome-aware "
            "supervision-space upper bound. Neither is a trained shared-embedding gain."
        ),
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".positive_guided_", dir=args.output_dir.parent))
    try:
        summary_frame.to_csv(staging / "cell_summary.csv", index=False)
        action_manifest.to_csv(staging / "action_manifest.csv.gz", index=False, compression="gzip")
        action_manifest.loc[new_queries].to_csv(
            staging / "newly_recoverable_errors.csv.gz", index=False, compression="gzip",
        )
        with h5py.File(staging / "matrix_results.h5", "w") as handle:
            handle.attrs["families_json"] = json.dumps(FAMILIES)
            handle.attrs["doses_json"] = json.dumps([float(value) for value in doses])
            handle.attrs["reference_kinds_json"] = json.dumps(REFERENCE_KINDS)
            create_dataset(handle, "baseline_rank", baseline_rank)
            create_dataset(handle, "baseline_margin", baseline_margin)
            create_dataset(handle, "result_rank", result_rank)
            create_dataset(handle, "result_margin", result_margin)
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

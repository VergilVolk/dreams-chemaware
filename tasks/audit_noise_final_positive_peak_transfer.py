"""Full-graph audit of recurrent real-positive peak transfer actions.

The preceding positive-guided matrix can only change intensities of peaks that
already exist in the query.  This stage addresses the remaining positive-
deficit errors by transferring only fragment peaks repeatedly observed in up to
three real same-identity references and absent from the query.  No theoretical
mass and no chemical-rule-imputed peak is inserted.

Every action is evaluated on the complete P3-disjoint graph and repeated with
the hardest wrong molecule as a direction-matched control.  Outcome-aware
unions remain headroom only.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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

from audit_noise_final_positive_guided_matrix import (  # noqa: E402
    baseline_detail, cluster_ci, create_dataset, molecule_rank_margin, reference_profile, top_rows,
)
from noise_final_core import CandidateGraph, json_dump, load_embedding_cache, sha256_file  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


FAMILIES = ("recurrent_peak_graft", "balanced_peak_exchange", "recurrent_union_mix")
REFERENCE_KINDS = ("positive", "hardest_wrong_control")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--uncovered-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_uncovered_errors")
    parser.add_argument("--previous-matrix", type=Path, default=ROOT / "data/validation/g8r_noise_final_positive_guided_matrix")
    parser.add_argument("--fivepoint", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_fivepoint_headroom.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_positive_peak_transfer")
    parser.add_argument("--doses", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--positive-references", type=int, default=3)
    parser.add_argument("--minimum-reference-prevalence", type=float, default=0.67)
    parser.add_argument("--maximum-transferred-peaks", type=int, default=5)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-queries", type=int, default=0)
    return parser.parse_args()


def recurrent_missing_peaks(
    query: torch.Tensor, references: list[torch.Tensor], tolerance: float,
    minimum_prevalence: float, maximum: int,
) -> np.ndarray:
    """Return [m/z, median intensity, prevalence] for recurrent missing peaks."""
    if not references:
        return np.empty((0, 3), dtype=np.float32)
    per_reference: list[np.ndarray] = []
    pooled: list[tuple[float, float]] = []
    for reference in references:
        values = reference.detach().cpu().numpy()[1:]
        valid = values[(values[:, 0] > 0) & (values[:, 1] > 0)]
        per_reference.append(valid)
        pooled.extend((float(mz), float(intensity)) for mz, intensity in valid)
    if not pooled:
        return np.empty((0, 3), dtype=np.float32)
    pooled.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, float]]] = []
    for peak in pooled:
        if not clusters or peak[0] - float(np.median([item[0] for item in clusters[-1]])) > tolerance:
            clusters.append([peak])
        else:
            clusters[-1].append(peak)
    query_values = query.detach().cpu().numpy()[1:]
    query_mz = query_values[:, 0][(query_values[:, 0] > 0) & (query_values[:, 1] > 0)]
    required_support = max(1, int(math.ceil(minimum_prevalence * len(references) - 1e-12)))
    output: list[tuple[float, float, float]] = []
    for cluster in clusters:
        center = float(np.median([item[0] for item in cluster]))
        intensities: list[float] = []
        support = 0
        for values in per_reference:
            if not len(values):
                continue
            distance = np.abs(values[:, 0] - center)
            nearest = int(np.argmin(distance))
            if float(distance[nearest]) <= tolerance:
                support += 1
                intensities.append(float(values[nearest, 1]))
        if support < required_support:
            continue
        if len(query_mz) and float(np.min(np.abs(query_mz - center))) <= tolerance:
            continue
        output.append((center, float(np.median(intensities)), support / len(references)))
    output.sort(key=lambda item: (-item[2], -item[1], item[0]))
    return np.asarray(output[:maximum], dtype=np.float32).reshape(-1, 3)


def apply_transfer(
    clean: torch.Tensor, missing: np.ndarray, query_prevalence: np.ndarray,
    family: str, dose: float,
) -> tuple[torch.Tensor, int]:
    if family not in FAMILIES or not 0 < dose <= 1:
        raise ValueError("invalid transfer action")
    output = clean.clone()
    precursor = output[0].clone()
    fragments = output[1:].detach().cpu().numpy().copy()
    valid = (fragments[:, 0] > 0) & (fragments[:, 1] > 0)
    real = fragments[valid].copy()
    additions = np.asarray(missing, dtype=np.float32)
    if not len(additions):
        return output, 0
    inserted = np.column_stack((additions[:, 0], dose * additions[:, 1])).astype(np.float32)
    retained_insertions = 0
    if family == "recurrent_peak_graft":
        capacity = len(fragments) - len(real)
        inserted = inserted[:max(capacity, 0)]
        merged = np.vstack((real, inserted)) if len(inserted) else real
        retained_insertions = len(inserted)
    elif family == "balanced_peak_exchange":
        count = min(len(inserted), len(real))
        if count == 0:
            return output, 0
        real_token_indices = np.flatnonzero(valid) + 1
        prevalence = query_prevalence[real_token_indices]
        remove_order = np.lexsort((real[:, 0], real[:, 1], prevalence))[:count]
        keep = np.ones(len(real), dtype=bool)
        keep[remove_order] = False
        inserted = inserted[:count]
        merged = np.vstack((real[keep], inserted))
        retained_insertions = len(inserted)
    else:
        merged = np.vstack((real, inserted))
        retained_insertions = len(inserted)
        if len(merged) > len(fragments):
            keep = np.argsort(-merged[:, 1], kind="stable")[:len(fragments)]
            retained_insertions = int(np.sum(keep >= len(real)))
            merged = merged[keep]
    merged = merged[np.argsort(merged[:, 0], kind="stable")]
    if len(merged) > len(fragments):
        raise RuntimeError("transfer action overflowed fragment tensor")
    fragments[:] = 0
    fragments[:len(merged)] = merged
    maximum = float(fragments[:, 1].max())
    if maximum > 0:
        fragments[:, 1] /= maximum
    output[1:] = torch.from_numpy(fragments)
    output[0] = precursor
    return output, int(retained_insertions)


def strict_bool(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{name} is not a strict boolean column")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def main() -> None:
    args = arguments()
    formal = args.max_queries == 0
    previous_manifest = args.previous_matrix / "action_manifest.csv.gz"
    previous_report = args.previous_matrix / "report.json"
    required = [
        args.graph, args.embeddings, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.uncovered_dir / "uncovered_errors.csv.gz",
        previous_manifest, previous_report, args.fivepoint,
    ]
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(missing_files)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite peak-transfer matrix: {args.output_dir}")
    doses = np.asarray(args.doses, dtype=np.float32)
    if len(doses) != len(np.unique(doses)) or np.any((doses <= 0) | (doses > 1)):
        raise ValueError("doses must be unique and in (0, 1]")
    if not 0 < args.minimum_reference_prevalence <= 1:
        raise ValueError("minimum-reference-prevalence must be in (0, 1]")
    if args.maximum_transferred_peaks < 1 or args.positive_references < 1:
        raise ValueError("reference and transfer counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph = CandidateGraph(args.graph)
    n_queries = graph.n_queries if formal else min(args.max_queries, graph.n_queries)
    if formal and n_queries != 23876:
        raise RuntimeError(f"formal transfer expects 23,876 queries, observed {n_queries}")
    old_report = json.loads(previous_report.read_text(encoding="utf-8"))
    if old_report.get("status") != "noise_final_positive_guided_matrix_complete":
        raise RuntimeError("previous positive-guided report is malformed")
    if old_report.get("provenance", {}).get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("previous positive-guided matrix used a different candidate graph")
    _, embeddings, embedding_index = load_embedding_cache(args.embeddings)
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    candidate_indices: list[np.ndarray] = []
    candidate_ptr: list[np.ndarray] = []
    baseline_rank = np.empty(n_queries, dtype=np.int16)
    for query in range(n_queries):
        scores, rows, ptr, _, rank, _ = baseline_detail(graph, query)
        molecule_scores = np.maximum.reduceat(scores, ptr[:-1])
        wrong = int(np.argmax(molecule_scores[1:])) + 1
        baseline_rank[query] = rank
        positive_rows.append(top_rows(scores, rows, ptr, 0, args.positive_references))
        negative_rows.append(top_rows(scores, rows, ptr, wrong, args.positive_references))
        candidate_indices.append(np.asarray([embedding_index[int(row)] for row in rows], dtype=np.int64))
        candidate_ptr.append(np.asarray(ptr, dtype=np.int64))
    if formal and int(np.sum(baseline_rank != 1)) != 1805:
        raise RuntimeError(
            f"formal transfer expects 1,805 official errors, observed {np.sum(baseline_rank != 1)}"
        )

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
                print(f"[peak-transfer spectra] {position:,}/{len(tensor_rows):,}", flush=True)
    query_tensors = [tensors[int(row)] for row in graph.query_row[:n_queries]]
    missing_peaks: list[list[np.ndarray]] = [[], []]
    query_prevalence = np.empty((n_queries, 2, args.n_highest_peaks + 1), dtype=np.float32)
    for query in range(n_queries):
        for kind, rows in enumerate((positive_rows[query], negative_rows[query])):
            references = [tensors[int(row)] for row in rows]
            prevalence, _ = reference_profile(query_tensors[query], references, args.fragment_tolerance)
            query_prevalence[query, kind] = prevalence
            missing_peaks[kind].append(recurrent_missing_peaks(
                query_tensors[query], references, args.fragment_tolerance,
                args.minimum_reference_prevalence, args.maximum_transferred_peaks,
            ))
        if (query + 1) % 2500 == 0 or query + 1 == n_queries:
            print(f"[peak-transfer profiles] {query + 1:,}/{n_queries:,}", flush=True)

    model, checkpoint_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if checkpoint_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("peak-transfer scan requires official DreaMS weights")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    clean_vectors = np.empty((n_queries, embeddings.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, n_queries, args.batch_size):
            right = min(left + args.batch_size, n_queries)
            clean_vectors[left:right] = model(torch.stack(query_tensors[left:right]).to(device)).float().cpu().numpy()
    cached_query = np.stack([embeddings[embedding_index[int(row)]] for row in graph.query_row[:n_queries]])
    preservation = np.sum(clean_vectors * cached_query, axis=1)
    fresh_rank = np.empty(n_queries, dtype=np.int16)
    for query in range(n_queries):
        scores = clean_vectors[query] @ embeddings[candidate_indices[query]].T
        fresh_rank[query], _ = molecule_rank_margin(scores, candidate_ptr[query])
    stable = fresh_rank == baseline_rank
    mismatch = int(np.sum(~stable))
    if formal and (float(np.quantile(preservation, 0.01)) < 0.999 or mismatch > 3):
        raise RuntimeError("fresh official reproduction failed")

    cells = [(family, float(dose), kind) for family in FAMILIES for dose in doses for kind in range(2)]
    n_cells = len(cells)
    result_rank = np.empty((n_queries, n_cells), dtype=np.int16)
    transferred_count = np.empty((n_queries, n_cells), dtype=np.int8)
    total = n_queries * n_cells
    with torch.inference_mode():
        for left in range(0, total, args.batch_size):
            right = min(left + args.batch_size, total)
            linear = np.arange(left, right, dtype=np.int64)
            queries = linear // n_cells
            local_cells = linear % n_cells
            batch: list[torch.Tensor] = []
            counts: list[int] = []
            for query, cell in zip(queries, local_cells):
                family, dose, kind = cells[int(cell)]
                variant, count = apply_transfer(
                    query_tensors[int(query)], missing_peaks[kind][int(query)],
                    query_prevalence[int(query), kind], family, dose,
                )
                batch.append(variant)
                counts.append(count)
            vectors = model(torch.stack(batch).to(device)).float().cpu().numpy()
            for query in np.unique(queries):
                offsets = np.flatnonzero(queries == query)
                scores = vectors[offsets] @ embeddings[candidate_indices[int(query)]].T
                for offset, pair_scores in zip(offsets, scores):
                    rank, _ = molecule_rank_margin(pair_scores, candidate_ptr[int(query)])
                    cell = int(local_cells[offset])
                    result_rank[int(query), cell] = rank
                    transferred_count[int(query), cell] = counts[int(offset)]
            if right % 25000 < args.batch_size or right == total:
                print(f"[peak-transfer exact] {right:,}/{total:,}", flush=True)
    del model
    gc.collect()

    formulas = graph.query_formula[:n_queries]
    near = graph.query_has_near[:n_queries]
    baseline_correct = fresh_rank == 1
    summaries: list[dict] = []
    passing: list[str] = []
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
            near_delta = float(np.mean(delta[near_eval]))
            ci = cluster_ci(delta[stable], formulas[stable], args.bootstrap, args.seed + family_index * 100 + dose_index)
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
                passing.append(cell_id)
            summaries.append({
                "cell_id": cell_id,
                "family": family,
                "dose": float(dose),
                "queries": int(np.sum(stable)),
                "eligible_query_fraction": float(np.mean(transferred_count[stable, target_cell] > 0)),
                "mean_transferred_peaks_when_eligible": float(
                    np.mean(transferred_count[stable, target_cell][transferred_count[stable, target_cell] > 0])
                ) if np.any(transferred_count[stable, target_cell] > 0) else 0.0,
                "delta_recall1": float(np.mean(delta[stable])),
                "corrected": corrected,
                "introduced": introduced,
                "net": corrected - introduced,
                "risk_net_lambda2": corrected - 2 * introduced,
                "near_delta_recall1": near_delta,
                "formula_cluster_delta_ci_low": ci[0],
                "formula_cluster_delta_ci_high": ci[1],
                "positive_minus_wrong_control_top1": float(np.mean(specificity[stable])),
                "specificity_formula_ci_low": specificity_ci[0],
                "specificity_formula_ci_high": specificity_ci[1],
                "pass_to_training_policy_screen": passed,
            })
    summary = pd.DataFrame(summaries).sort_values(
        ["risk_net_lambda2", "delta_recall1"], ascending=False, kind="stable",
    )

    previous = pd.read_csv(previous_manifest)
    if len(previous) != graph.n_queries or previous["query_index"].duplicated().any():
        raise RuntimeError("previous positive-guided manifest is not one-to-one with graph")
    previous = previous.sort_values("query_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(previous["query_index"].to_numpy(np.int64), np.arange(graph.n_queries)):
        raise RuntimeError("previous positive-guided manifest query order mismatch")
    uncovered = pd.read_csv(args.uncovered_dir / "uncovered_errors.csv.gz")
    uncovered_set = set(map(int, uncovered["query_index"]))
    pn_uncovered = np.asarray([query in uncovered_set for query in range(n_queries)], dtype=bool)
    previous_oracle = strict_bool(
        previous["positive_guided_oracle_recoverable"], "previous oracle recoverable",
    )[:n_queries]
    residual_before = (baseline_rank != 1) & pn_uncovered & ~previous_oracle
    positive_cells = [index for index, (_, _, kind) in enumerate(cells) if kind == 0]
    transfer_oracle = (baseline_rank != 1) & stable & np.any(result_rank[:, positive_cells] == 1, axis=1)
    new_unique = residual_before & transfer_oracle
    expanded_before = int(old_report["headroom_only"]["expanded_union"])
    expanded_after = expanded_before + int(np.sum(new_unique))
    fivepoint = json.loads(args.fivepoint.read_text(encoding="utf-8"))
    required_for_five = int(fivepoint["required_net_corrections"])
    manifest = pd.DataFrame({
        "query_index": np.arange(n_queries, dtype=np.int64),
        "query_row": graph.query_row[:n_queries],
        "query_ik14": graph.query_ik14[:n_queries],
        "query_formula": graph.query_formula[:n_queries],
        "baseline_rank": baseline_rank,
        "positive_missing_peak_count": [len(values) for values in missing_peaks[0]],
        "wrong_control_missing_peak_count": [len(values) for values in missing_peaks[1]],
        "transfer_oracle_recoverable": transfer_oracle,
        "new_beyond_pn_and_intensity_matrix": new_unique,
    })
    report = {
        "status": "noise_final_positive_peak_transfer_complete",
        "formal": formal,
        "queries": n_queries,
        "official_errors": int(np.sum(baseline_rank != 1)),
        "cells": len(FAMILIES) * len(doses),
        "direction_controls": len(FAMILIES) * len(doses),
        "encoded_variants": total,
        "passing_cells": passing,
        "best_fixed_cell": json.loads(summary.iloc[0].to_json()),
        "headroom_only": {
            "residual_errors_before_transfer": int(np.sum(residual_before)),
            "transfer_oracle_recoverable_errors": int(np.sum(transfer_oracle)),
            "new_unique_beyond_pn_and_intensity_matrix": int(np.sum(new_unique)),
            "expanded_union_before": expanded_before,
            "expanded_union_after": expanded_after,
            "required_for_five_points": required_for_five,
            "remaining_to_five_points": max(required_for_five - expanded_after, 0),
            "reaches_five_points": bool(expanded_after >= required_for_five),
            "positive_arm_new_unique_total": int(old_report["headroom_only"]["new_unique_errors_beyond_frozen_pn"] + np.sum(new_unique)),
            "reaches_positive_arm_350_buffer": bool(
                old_report["headroom_only"]["new_unique_errors_beyond_frozen_pn"] + np.sum(new_unique) >= 350
            ),
        },
        "fresh_official_reproduction": {
            "preservation_mean": float(np.mean(preservation)),
            "preservation_p01": float(np.quantile(preservation, 0.01)),
            "rank_mismatches": mismatch,
        },
        "contracts": {
            "transferred_peaks_observed_in_real_references": True,
            "minimum_reference_prevalence": args.minimum_reference_prevalence,
            "maximum_transferred_peaks": args.maximum_transferred_peaks,
            "wrong_candidate_is_direction_control": True,
            "full_graph_exact_safety": True,
            "outcome_used_only_for_headroom_union": True,
            "P2b_forbidden": True,
            "shared_embedding_training_result": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embeddings),
            "hdf5_sha256": sha256_file(args.data),
            "previous_manifest_sha256": sha256_file(previous_manifest),
            "previous_report_sha256": sha256_file(previous_report),
            "uncovered_report_sha256": sha256_file(args.uncovered_dir / "report.json"),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Fixed cells are frozen-encoder action outcomes. Cross-action unions are "
            "outcome-aware headroom, not shared-embedding performance."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".peak_transfer_", dir=args.output_dir.parent))
    try:
        summary.to_csv(staging / "cell_summary.csv", index=False)
        manifest.to_csv(staging / "action_manifest.csv.gz", index=False, compression="gzip")
        manifest.loc[manifest["new_beyond_pn_and_intensity_matrix"]].to_csv(
            staging / "newly_recoverable_errors.csv.gz", index=False, compression="gzip",
        )
        with h5py.File(staging / "matrix_results.h5", "w") as handle:
            handle.attrs["families_json"] = json.dumps(FAMILIES)
            handle.attrs["doses_json"] = json.dumps([float(value) for value in doses])
            handle.attrs["reference_kinds_json"] = json.dumps(REFERENCE_KINDS)
            create_dataset(handle, "baseline_rank", baseline_rank)
            create_dataset(handle, "result_rank", result_rank)
            create_dataset(handle, "transferred_count", transferred_count)
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

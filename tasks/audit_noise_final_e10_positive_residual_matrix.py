"""E10: positive-guided residual action matrix on the mature E8 embedding.

This is the required action-space expansion after E9-B showed that mature
candidate-gradient/role-confounder actions cannot reach five total points.
On the already-consumed held formula fold, real same-identity reference spectra
define two preregistered peak-noise families: intensity consensus projection
and recurrent missing-peak transfer. The current hardest wrong molecule supplies
a direction-matched control. All references and candidates are re-encoded by
the same frozen mature E8 shared encoder.

Fixed-cell results are outcome-free action audits. The no-op-aware union is an
explicit outcome-aware capacity upper bound and is never a model result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_e9_action_staleness import load_student, rank_margin  # noqa: E402
from audit_noise_final_positive_guided_matrix import apply_action, reference_profile  # noqa: E402
from audit_noise_final_positive_peak_transfer import (  # noqa: E402
    apply_transfer,
    recurrent_missing_peaks,
)
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore,
    encode_rows,
    formula_bootstrap_mean,
    forward_embeddings,
)


CORE_INTENSITY_CELLS = tuple(("consensus_projection", dose, 0.0) for dose in (0.25, 0.50, 0.75, 1.00))
CORE_TRANSFER_CELLS = tuple(("recurrent_union_mix", dose, 0.0) for dose in (0.10, 0.25, 0.50))
EXPANSION_CELLS = (
    ("matched_intensity_transport", 0.50, 0.0),
    ("matched_intensity_transport", 1.00, 0.0),
    ("recurrent_peak_graft", 0.10, 0.0),
    ("recurrent_peak_graft", 0.25, 0.0),
    ("recurrent_peak_graft", 0.50, 0.0),
    ("balanced_peak_exchange", 0.10, 0.0),
    ("balanced_peak_exchange", 0.25, 0.0),
    ("balanced_peak_exchange", 0.50, 0.0),
    ("transport_then_union", 0.50, 0.50),
    ("transport_then_union", 1.00, 0.50),
    ("consensus_then_union", 0.50, 0.50),
    ("consensus_then_union", 0.75, 0.50),
)
REFERENCE_KINDS = ("positive", "hardest_wrong_control")


def positive_cells(cell_set: str = "core") -> tuple[tuple[str, float, float], ...]:
    core = CORE_INTENSITY_CELLS + CORE_TRANSFER_CELLS
    if cell_set == "core":
        return core
    if cell_set == "expanded":
        return core + EXPANSION_CELLS
    raise ValueError(f"unknown E10 cell set: {cell_set}")


def action_cells(cell_set: str = "core") -> tuple[tuple[str, float, float, int], ...]:
    return tuple((family, float(dose), float(aux), kind)
                 for family, dose, aux in positive_cells(cell_set) for kind in range(2))


def cell_id(family: str, dose: float, auxiliary_dose: float) -> str:
    if auxiliary_dose:
        return f"{family}|intensity={dose:.2f}|transfer={auxiliary_dose:.2f}"
    return f"{family}|dose={dose:.2f}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--e9-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e9_action_staleness")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e10_positive_residual_matrix")
    parser.add_argument("--cell-set", choices=("core", "expanded"), default="core")
    parser.add_argument("--positive-references", type=int, default=3)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-reference-prevalence", type=float, default=0.67)
    parser.add_argument("--maximum-transferred-peaks", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def top_rows(scores: np.ndarray, rows: np.ndarray, ptr: np.ndarray,
             molecule: int, count: int) -> np.ndarray:
    left, right = map(int, ptr[molecule:molecule + 2])
    order = np.argsort(-np.asarray(scores[left:right]), kind="stable")[:count] + left
    return np.asarray(rows[order], dtype=np.int64)


def cell_variant(clean: torch.Tensor, profile: tuple[np.ndarray, np.ndarray],
                 missing: np.ndarray, family: str, dose: float,
                 auxiliary_dose: float = 0.0) -> torch.Tensor:
    prevalence, target = profile
    if family in {"consensus_projection", "matched_intensity_transport"}:
        return apply_action(clean, prevalence, target, family, dose)
    if family in {"recurrent_union_mix", "recurrent_peak_graft", "balanced_peak_exchange"}:
        variant, _ = apply_transfer(clean, missing, prevalence, family, dose)
        return variant
    if family in {"transport_then_union", "consensus_then_union"}:
        first_family = (
            "matched_intensity_transport" if family == "transport_then_union"
            else "consensus_projection"
        )
        intensity_variant = apply_action(clean, prevalence, target, first_family, dose)
        variant, _ = apply_transfer(
            intensity_variant, missing, prevalence, "recurrent_union_mix", auxiliary_dose,
        )
        return variant
    raise RuntimeError(f"unregistered E10 family: {family}")


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E10 output: {args.output_dir}")
    source_per_query = args.student_checkpoint.parent / "held_per_query.csv.gz"
    e9_report_path, e9_table_path = args.e9_dir / "report.json", args.e9_dir / "per_action.csv.gz"
    required = [args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
                args.student_checkpoint, source_per_query, e9_report_path, e9_table_path]
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(missing_files)
    if not 0 < args.minimum_reference_prevalence <= 1 or args.maximum_transferred_peaks < 1:
        raise ValueError("invalid recurrent transfer parameters")
    if args.positive_references < 1 or args.batch_size < 1:
        raise ValueError("positive reference and batch counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E10 requires CUDA")

    e9 = json.loads(e9_report_path.read_text(encoding="utf-8"))
    if e9.get("status") != "noise_final_e9_action_staleness_complete" or not e9.get("formal"):
        raise RuntimeError("E10 requires formal E9")
    if e9.get("pass_to_online_remining_training"):
        raise RuntimeError("E10 branch assumes E9 rejected online re-mining")
    source = pd.read_csv(source_per_query)
    needed_source = {"query_index", "query_row", "query_ik14", "query_formula", "baseline_rank", "final_rank"}
    if needed_source - set(source.columns) or source["query_index"].duplicated().any():
        raise RuntimeError("mature E8 held-query table is malformed")
    source = source.sort_values("query_index", kind="stable").reset_index(drop=True)
    if len(source) != 5923:
        raise RuntimeError(f"E10 expects 5,923 held queries, observed {len(source)}")

    graph = CandidateGraph(args.graph)
    queries = source["query_index"].to_numpy(np.int64)
    if not np.array_equal(graph.query_row[queries], source["query_row"].to_numpy(np.int64)):
        raise RuntimeError("E10 source query rows drifted from graph")
    needed_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row]))
    store = SpectrumStore(args.data, needed_rows, 100)
    model = load_student(args, device)
    embeddings = encode_rows(model, store, store.rows, device, args.batch_size, args.amp, "E10-student")
    embedding_index = {int(row): index for index, row in enumerate(store.rows)}

    clean_rank = np.empty(len(queries), dtype=np.int16)
    clean_margin = np.empty(len(queries), dtype=np.float32)
    positive_rows: list[np.ndarray] = []
    wrong_rows: list[np.ndarray] = []
    for local, query in enumerate(queries):
        _, rows, ptr, _ = graph.query_block(int(query))
        qvector = embeddings[embedding_index[int(graph.query_row[int(query)])]]
        candidate = embeddings[[embedding_index[int(row)] for row in rows]]
        pair_scores = candidate @ qvector
        molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
        wrong = int(np.argmax(molecule_scores[1:])) + 1
        clean_rank[local] = 1 + int(np.sum(molecule_scores[1:] >= molecule_scores[0]))
        clean_margin[local] = float(molecule_scores[0] - molecule_scores[wrong])
        positive_rows.append(top_rows(pair_scores, rows, ptr, 0, args.positive_references))
        wrong_rows.append(top_rows(pair_scores, rows, ptr, wrong, args.positive_references))
    rank_mismatches = int(np.sum(clean_rank != source["final_rank"].to_numpy(np.int16)))
    if rank_mismatches:
        raise RuntimeError(f"E10 failed to reproduce {rank_mismatches} mature E8 ranks")

    profiles: list[list[tuple[np.ndarray, np.ndarray]]] = []
    missing_peaks: list[list[np.ndarray]] = []
    for local, query in enumerate(queries):
        clean = store.one(int(graph.query_row[int(query)]))
        local_profiles, local_missing = [], []
        for rows in (positive_rows[local], wrong_rows[local]):
            references = [store.one(int(row)) for row in rows]
            profile = reference_profile(clean, references, args.fragment_tolerance)
            local_profiles.append(profile)
            local_missing.append(recurrent_missing_peaks(
                clean, references, args.fragment_tolerance,
                args.minimum_reference_prevalence, args.maximum_transferred_peaks,
            ))
        profiles.append(local_profiles)
        missing_peaks.append(local_missing)
        if (local + 1) % 1000 == 0 or local + 1 == len(queries):
            print(f"[E10 profiles] {local + 1:,}/{len(queries):,}", flush=True)

    positive_cell_definitions = positive_cells(args.cell_set)
    cells = action_cells(args.cell_set)
    result_rank = np.empty((len(queries), len(cells)), dtype=np.int16)
    result_margin = np.empty((len(queries), len(cells)), dtype=np.float32)
    total = len(queries) * len(cells)
    with torch.inference_mode():
        for left in range(0, total, args.batch_size):
            right = min(left + args.batch_size, total)
            linear = np.arange(left, right, dtype=np.int64)
            local_queries = linear // len(cells)
            local_cells = linear % len(cells)
            variants = []
            for local_query, local_cell in zip(local_queries, local_cells):
                family, dose, auxiliary_dose, kind = cells[int(local_cell)]
                query = int(queries[int(local_query)])
                variants.append(cell_variant(
                    store.one(int(graph.query_row[query])),
                    profiles[int(local_query)][int(kind)],
                    missing_peaks[int(local_query)][int(kind)], family, dose, auxiliary_dose,
                ))
            vectors = forward_embeddings(model, torch.stack(variants).to(device), args.amp).float().cpu().numpy()
            for vector, local_query, local_cell in zip(vectors, local_queries, local_cells):
                query = int(queries[int(local_query)])
                rank, margin, _ = rank_margin(graph, query, vector, embeddings, embedding_index)
                result_rank[int(local_query), int(local_cell)] = rank
                result_margin[int(local_query), int(local_cell)] = margin
            if right % 25000 < args.batch_size or right == total:
                print(f"[E10 actions] {right:,}/{total:,}", flush=True)

    formulas = source["query_formula"].astype(str).to_numpy()
    clean_correct = clean_rank == 1
    cell_records = []
    positive_cell_indices = []
    for family_index, (family, dose, auxiliary_dose) in enumerate(positive_cell_definitions):
        target_cell = cells.index((family, float(dose), float(auxiliary_dose), 0))
        control_cell = cells.index((family, float(dose), float(auxiliary_dose), 1))
        positive_cell_indices.append(target_cell)
        target_correct = result_rank[:, target_cell] == 1
        control_correct = result_rank[:, control_cell] == 1
        delta = target_correct.astype(float) - clean_correct.astype(float)
        specificity = target_correct.astype(float) - control_correct.astype(float)
        corrected = int(np.sum(~clean_correct & target_correct))
        introduced = int(np.sum(clean_correct & ~target_correct))
        ci = formula_bootstrap_mean(delta, formulas, args.bootstrap_resamples, args.seed + family_index)
        specificity_ci = formula_bootstrap_mean(
            specificity, formulas, args.bootstrap_resamples, args.seed + 1000 + family_index,
        )
        cell_records.append({
            "cell_id": cell_id(family, dose, auxiliary_dose), "family": family,
            "dose": dose, "auxiliary_dose": auxiliary_dose,
            "delta_recall1_vs_mature_e8": float(np.mean(delta)), "corrected": corrected,
            "introduced": introduced, "risk_net_lambda2": corrected - 2 * introduced,
            "mean_margin_delta": float(np.mean(result_margin[:, target_cell] - clean_margin)),
            "formula_ci_low": ci["ci_low"], "formula_ci_high": ci["ci_high"],
            "positive_minus_wrong_control_top1": float(np.mean(specificity)),
            "specificity_formula_ci_low": specificity_ci["ci_low"],
            "specificity_formula_ci_high": specificity_ci["ci_high"],
            "fixed_cell_pass": bool(ci["ci_low"] > 0 and corrected > 2 * introduced and specificity_ci["ci_low"] > 0),
        })
    cell_summary = pd.DataFrame(cell_records).sort_values(
        ["risk_net_lambda2", "delta_recall1_vs_mature_e8"], ascending=False, kind="stable",
    )

    # Full-task no-op-aware union: mature clean, every positive action, and
    # every frozen mature N-arm action from E9. Outcomes choose the maximum
    # margin, so this is a strict upper bound only.
    best_margin = clean_margin.astype(np.float64, copy=True)
    best_rank = clean_rank.astype(np.int16, copy=True)
    best_source = np.asarray(["no_op"] * len(queries), dtype=object)
    for cell in positive_cell_indices:
        improve = result_margin[:, cell] > best_margin
        best_margin[improve] = result_margin[improve, cell]
        best_rank[improve] = result_rank[improve, cell]
        family, dose, auxiliary_dose, _ = cells[cell]
        best_source[improve] = f"P:{cell_id(family, dose, auxiliary_dose)}"
    query_position = {int(query): index for index, query in enumerate(queries)}
    e9_table = pd.read_csv(e9_table_path)
    for row in e9_table.itertuples(index=False):
        local = query_position.get(int(row.query_index))
        if local is None:
            continue
        margin = float(row.frozen_margin)
        if margin > best_margin[local]:
            best_margin[local] = margin
            best_rank[local] = int(row.frozen_rank)
            best_source[local] = f"N:{row.selector}|{int(row.step)}"
    official_correct = source["baseline_rank"].to_numpy(int) == 1
    oracle_correct = best_rank == 1
    incremental = oracle_correct.astype(float) - clean_correct.astype(float)
    total_effect = oracle_correct.astype(float) - official_correct.astype(float)
    incremental_ci = formula_bootstrap_mean(incremental, formulas, args.bootstrap_resamples, args.seed + 9000)
    total_ci = formula_bootstrap_mean(total_effect, formulas, args.bootstrap_resamples, args.seed + 9001)
    total_delta = float(np.mean(total_effect))
    fixed_passing = cell_summary.loc[cell_summary["fixed_cell_pass"], "cell_id"].astype(str).tolist()
    report = {
        "status": (
            "noise_final_e10_positive_residual_matrix_complete" if args.cell_set == "core"
            else "noise_final_e10b_positive_action_expansion_complete"
        ), "formal": True, "cell_set": args.cell_set,
        "held_queries": int(len(queries)), "held_formulas": int(source["query_formula"].nunique()),
        "mature_e8_rank_reproduction_mismatches": rank_mismatches,
        "cells": len(positive_cell_definitions),
        "direction_controls": len(positive_cell_definitions),
        "passing_fixed_cells": fixed_passing,
        "best_fixed_cell": cell_summary.iloc[0].to_dict(),
        "no_op_aware_union_headroom": {
            "mature_e8_accuracy": float(np.mean(clean_correct)),
            "oracle_accuracy": float(np.mean(oracle_correct)),
            "incremental_delta_over_mature_e8": float(np.mean(incremental)),
            "incremental_corrected": int(np.sum(~clean_correct & oracle_correct)),
            "incremental_introduced": int(np.sum(clean_correct & ~oracle_correct)),
            "incremental_formula_ci": incremental_ci,
            "total_delta_over_official": total_delta,
            "total_formula_ci": total_ci,
            "selected_source_counts": {str(k): int(v) for k, v in pd.Series(best_source).value_counts().items()},
            "reaches_five_total_points": bool(total_delta >= 0.05),
        },
        "gates": {
            "at_least_one_fixed_p_cell_passes": bool(fixed_passing),
            "expanded_union_reaches_five_points": bool(total_delta >= 0.05),
        },
        "pass_to_conditional_noise_training": bool(fixed_passing and total_delta >= 0.05),
        "decision": (
            "freeze passing P/N action families and train a training-only conditional no-op curriculum"
            if fixed_passing and total_delta >= 0.05
            else (
                "fixed P-arm safety passes, but the expanded union remains below five-point capacity; expand actions before selector training"
                if fixed_passing
                else "no fixed P-arm cell passes the safety/specificity gate; do not train a selector"
            )
        ),
        "contracts": {"real_same_identity_positive_references": True,
                      "wrong_candidate_direction_control": True,
                      "one_shared_mature_embedding_geometry": True,
                      "outcome_used_only_for_union_headroom": True,
                      "P2b": "forbidden", "P3_consumed": False},
        "provenance": {"graph_sha256": sha256_file(args.graph),
                       "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
                       "source_per_query_sha256": sha256_file(source_per_query),
                       "e9_per_action_sha256": sha256_file(e9_table_path),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Held-development action scan and outcome-aware capacity bound; not a newly trained encoder or deployable gain.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cell_summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
    pd.DataFrame({"query_index": queries, "query_formula": formulas, "clean_rank": clean_rank,
                  "oracle_rank": best_rank, "oracle_source": best_source}).to_csv(
        args.output_dir / "oracle_per_query.csv.gz", index=False, compression="gzip",
    )
    np.savez_compressed(args.output_dir / "matrix.npz", queries=queries,
                        cells=np.asarray(cells, dtype=object), result_rank=result_rank,
                        result_margin=result_margin, clean_rank=clean_rank, clean_margin=clean_margin)
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

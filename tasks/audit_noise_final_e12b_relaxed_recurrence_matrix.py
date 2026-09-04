"""E12-B: full-task matrix for empirically justified relaxed recurrence.

E12-A showed that 50% reference prevalence exposes recurrent missing peaks for
127/133 residual errors, whereas 34% adds essentially no coverage. E12-B fixes
prevalence at 0.50 and evaluates only a preregistered 2x2 design (5/10 peaks,
dose 0.25/0.50) across five reference policies, plus one support-weighted
10-peak dose-0.50 safety variant. The hardest wrong molecule is the matched
direction control. Every result uses the complete 5,923-query held fold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_e11_reference_diversity_matrix import REFERENCE_POLICIES, select_rows  # noqa: E402
from audit_noise_final_e9_action_staleness import load_student, rank_margin  # noqa: E402
from audit_noise_final_positive_guided_matrix import reference_profile  # noqa: E402
from audit_noise_final_positive_peak_transfer import apply_transfer, recurrent_missing_peaks  # noqa: E402
from calibrate_noise_final_e1_empirical import clean_instrument, decode  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, formula_bootstrap_mean, forward_embeddings,
)


POLICIES = ("top3",) + REFERENCE_POLICIES
STANDARD_RECIPES = tuple((maximum, dose, False) for maximum in (5, 10) for dose in (0.25, 0.50))
RECIPES = STANDARD_RECIPES + ((10, 0.50, True),)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--e9-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e9_action_staleness")
    parser.add_argument("--e10b-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e10b_positive_action_expansion")
    parser.add_argument("--e11-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e11_reference_diversity")
    parser.add_argument("--e12a-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e12a_residual_reachability")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e12b_relaxed_recurrence")
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def cell_name(policy: str, maximum: int, dose: float, weighted: bool) -> str:
    mode = "support_weighted" if weighted else "standard"
    return f"{policy}|{mode}|max={maximum}|dose={dose:.2f}"


def relaxed_variant(clean: torch.Tensor, missing: np.ndarray, prevalence: np.ndarray,
                    maximum: int, dose: float, weighted: bool) -> torch.Tensor:
    selected = np.asarray(missing[:maximum], dtype=np.float32).copy()
    if weighted and len(selected):
        selected[:, 1] *= selected[:, 2]
    variant, _ = apply_transfer(
        clean, selected, prevalence, "recurrent_union_mix", dose,
    )
    return variant


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E12-B output: {args.output_dir}")
    source_path = args.student_checkpoint.parent / "held_per_query.csv.gz"
    e9_report_path, e9_table_path = args.e9_dir / "report.json", args.e9_dir / "per_action.csv.gz"
    e10b_report_path, e10b_matrix_path = args.e10b_dir / "report.json", args.e10b_dir / "matrix.npz"
    e11_report_path, e11_matrix_path = args.e11_dir / "report.json", args.e11_dir / "matrix.npz"
    e11_oracle_path = args.e11_dir / "oracle_per_query.csv.gz"
    e12a_report_path = args.e12a_dir / "report.json"
    required = [args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
                args.student_checkpoint, source_path, e9_report_path, e9_table_path,
                e10b_report_path, e10b_matrix_path, e11_report_path, e11_matrix_path,
                e11_oracle_path, e12a_report_path]
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(missing_files)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E12-B requires CUDA")
    e9 = json.loads(e9_report_path.read_text(encoding="utf-8"))
    e10b = json.loads(e10b_report_path.read_text(encoding="utf-8"))
    e11 = json.loads(e11_report_path.read_text(encoding="utf-8"))
    e12a = json.loads(e12a_report_path.read_text(encoding="utf-8"))
    if e9.get("status") != "noise_final_e9_action_staleness_complete" or not e9.get("formal"):
        raise RuntimeError("E12-B requires formal E9")
    if e10b.get("status") != "noise_final_e10b_positive_action_expansion_complete" or not e10b.get("formal"):
        raise RuntimeError("E12-B requires formal E10-B")
    if e11.get("status") != "noise_final_e11_reference_diversity_complete" or not e11.get("formal"):
        raise RuntimeError("E12-B requires formal E11")
    if not e12a.get("pass_to_e12b_relaxed_recurrence_matrix"):
        raise RuntimeError("E12-A did not authorize E12-B")
    checkpoint_hash = sha256_file(args.student_checkpoint)
    if any(report.get("provenance", {}).get("student_checkpoint_sha256") != checkpoint_hash
           for report in (e10b, e11, e12a)):
        raise RuntimeError("E10-B/E11/E12-A checkpoint provenance differs")

    source = pd.read_csv(source_path).sort_values("query_index", kind="stable").reset_index(drop=True)
    if len(source) != 5923 or source["query_index"].duplicated().any():
        raise RuntimeError("E12-B expects the complete 5,923-query held task")
    queries = source["query_index"].to_numpy(np.int64)
    graph = CandidateGraph(args.graph)
    if not np.array_equal(graph.query_row[queries], source["query_row"].to_numpy(np.int64)):
        raise RuntimeError("E12-B source rows drifted from graph")
    needed_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row]))
    store = SpectrumStore(args.data, needed_rows, 100)
    with h5py.File(args.data, "r") as handle:
        inst_values = decode(handle["INSTRUMENT_TYPE"][store.rows])
        ce_values = np.asarray(handle["COLLISION_ENERGY"][store.rows], dtype=float)
    instruments = {int(row): clean_instrument(str(value)) for row, value in zip(store.rows, inst_values)}
    collision_energy = {int(row): float(value) for row, value in zip(store.rows, ce_values)}
    model = load_student(args, device)
    embeddings = encode_rows(model, store, store.rows, device, args.batch_size, args.amp, "E12B-student")
    embedding_index = {int(row): index for index, row in enumerate(store.rows)}

    clean_rank = np.empty(len(queries), dtype=np.int16)
    clean_margin = np.empty(len(queries), dtype=np.float32)
    reference_rows: list[list[list[np.ndarray]]] = []
    for local, query in enumerate(queries):
        _, rows, ptr, _ = graph.query_block(int(query))
        qrow = int(graph.query_row[int(query)])
        qvector = embeddings[embedding_index[qrow]]
        candidate_vectors = embeddings[[embedding_index[int(row)] for row in rows]]
        pair_scores = candidate_vectors @ qvector
        molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
        wrong = int(np.argmax(molecule_scores[1:])) + 1
        clean_rank[local] = 1 + int(np.sum(molecule_scores[1:] >= molecule_scores[0]))
        clean_margin[local] = float(molecule_scores[0] - molecule_scores[wrong])
        query_policies = []
        for policy in POLICIES:
            policy_kinds = []
            for molecule in (0, wrong):
                left, right = map(int, ptr[molecule:molecule + 2])
                local_rows = np.asarray(rows[left:right], dtype=np.int64)
                local_scores = pair_scores[left:right]
                local_vectors = candidate_vectors[left:right]
                if policy == "top3":
                    selected = local_rows[np.argsort(-local_scores, kind="stable")[:3]]
                else:
                    selected = select_rows(
                        local_rows, local_scores, local_vectors, policy,
                        instruments[qrow], collision_energy[qrow], instruments, collision_energy,
                    )
                policy_kinds.append(selected)
            query_policies.append(policy_kinds)
        reference_rows.append(query_policies)
    if int(np.sum(clean_rank != source["final_rank"].to_numpy(np.int16))):
        raise RuntimeError("E12-B failed to reproduce mature E8 ranks")

    profiles: list[list[list[tuple[np.ndarray, np.ndarray]]]] = []
    missing_peaks: list[list[list[np.ndarray]]] = []
    for local, query in enumerate(queries):
        clean = store.one(int(graph.query_row[int(query)]))
        query_profiles, query_missing = [], []
        for policy_rows in reference_rows[local]:
            policy_profiles, policy_missing = [], []
            for rows in policy_rows:
                references = [store.one(int(row)) for row in rows]
                prevalence, target = reference_profile(clean, references, args.fragment_tolerance)
                policy_profiles.append((prevalence, target))
                policy_missing.append(recurrent_missing_peaks(
                    clean, references, args.fragment_tolerance, 0.50, 10,
                ))
            query_profiles.append(policy_profiles)
            query_missing.append(policy_missing)
        profiles.append(query_profiles)
        missing_peaks.append(query_missing)
        if (local + 1) % 1000 == 0 or local + 1 == len(queries):
            print(f"[E12-B profiles] {local + 1:,}/{len(queries):,}", flush=True)

    cells = tuple((policy, maximum, dose, weighted, kind)
                  for policy in POLICIES for maximum, dose, weighted in RECIPES for kind in range(2))
    result_rank = np.empty((len(queries), len(cells)), dtype=np.int16)
    result_margin = np.empty((len(queries), len(cells)), dtype=np.float32)
    total = len(queries) * len(cells)
    with torch.inference_mode():
        for left in range(0, total, args.batch_size):
            right = min(left + args.batch_size, total)
            linear = np.arange(left, right, dtype=np.int64)
            local_queries, local_cells = linear // len(cells), linear % len(cells)
            variants = []
            for local_query, local_cell in zip(local_queries, local_cells):
                policy, maximum, dose, weighted, kind = cells[int(local_cell)]
                policy_index = POLICIES.index(policy)
                query = int(queries[int(local_query)])
                prevalence, _ = profiles[int(local_query)][policy_index][int(kind)]
                variants.append(relaxed_variant(
                    store.one(int(graph.query_row[query])),
                    missing_peaks[int(local_query)][policy_index][int(kind)],
                    prevalence, int(maximum), float(dose), bool(weighted),
                ))
            vectors = forward_embeddings(model, torch.stack(variants).to(device), args.amp).float().cpu().numpy()
            for vector, local_query, local_cell in zip(vectors, local_queries, local_cells):
                rank, margin, _ = rank_margin(
                    graph, int(queries[int(local_query)]), vector, embeddings, embedding_index,
                )
                result_rank[int(local_query), int(local_cell)] = rank
                result_margin[int(local_query), int(local_cell)] = margin
            if right % 25000 < args.batch_size or right == total:
                print(f"[E12-B actions] {right:,}/{total:,}", flush=True)

    formulas = source["query_formula"].astype(str).to_numpy()
    clean_correct = clean_rank == 1
    records, positive_indices = [], []
    positive_definitions = tuple((policy, maximum, dose, weighted)
                                 for policy in POLICIES for maximum, dose, weighted in RECIPES)
    for index, (policy, maximum, dose, weighted) in enumerate(positive_definitions):
        target_cell = cells.index((policy, maximum, dose, weighted, 0))
        control_cell = cells.index((policy, maximum, dose, weighted, 1))
        positive_indices.append(target_cell)
        target_correct = result_rank[:, target_cell] == 1
        control_correct = result_rank[:, control_cell] == 1
        effect = target_correct.astype(float) - clean_correct.astype(float)
        specificity = target_correct.astype(float) - control_correct.astype(float)
        corrected = int(np.sum(~clean_correct & target_correct))
        introduced = int(np.sum(clean_correct & ~target_correct))
        ci = formula_bootstrap_mean(effect, formulas, args.bootstrap_resamples, args.seed + index)
        specificity_ci = formula_bootstrap_mean(
            specificity, formulas, args.bootstrap_resamples, args.seed + 1000 + index,
        )
        records.append({
            "cell_id": cell_name(policy, int(maximum), float(dose), bool(weighted)),
            "reference_policy": policy, "maximum_peaks": int(maximum), "dose": float(dose),
            "support_weighted": bool(weighted),
            "delta_recall1_vs_mature_e8": float(np.mean(effect)), "corrected": corrected,
            "introduced": introduced, "risk_net_lambda2": corrected - 2 * introduced,
            "mean_margin_delta": float(np.mean(result_margin[:, target_cell] - clean_margin)),
            "formula_ci_low": ci["ci_low"], "formula_ci_high": ci["ci_high"],
            "positive_minus_wrong_control_top1": float(np.mean(specificity)),
            "specificity_formula_ci_low": specificity_ci["ci_low"],
            "specificity_formula_ci_high": specificity_ci["ci_high"],
            "fixed_cell_pass": bool(ci["ci_low"] > 0 and corrected > 2 * introduced
                                    and specificity_ci["ci_low"] > 0),
        })
    summary = pd.DataFrame(records).sort_values(
        ["risk_net_lambda2", "delta_recall1_vs_mature_e8"], ascending=False, kind="stable",
    )

    # Reconstruct E11 no-op oracle exactly, then add E12-B actions.
    old10 = np.load(e10b_matrix_path, allow_pickle=True)
    old11 = np.load(e11_matrix_path, allow_pickle=True)
    if (not np.array_equal(old10["queries"], queries) or not np.array_equal(old11["queries"], queries)
            or not np.array_equal(old10["clean_rank"], clean_rank)
            or not np.array_equal(old11["clean_rank"], clean_rank)):
        raise RuntimeError("prior matrix query/clean states drifted")
    best_margin = clean_margin.astype(np.float64, copy=True)
    best_rank = clean_rank.astype(np.int16, copy=True)
    best_source = np.asarray(["no_op"] * len(queries), dtype=object)
    # Preserve the historical operation order exactly: E10-B positive actions,
    # then mature E9 N-arm actions, then E11 diverse-reference actions. Strict
    # `>` keeps the earlier source on exact margin ties.
    for prefix, matrix, kind_column in (("E10B", old10, 3),):
        old_cells = matrix["cells"]
        for old_index in range(len(old_cells)):
            if int(old_cells[old_index, kind_column]) != 0:
                continue
            improve = matrix["result_margin"][:, old_index] > best_margin
            best_margin[improve] = matrix["result_margin"][improve, old_index]
            best_rank[improve] = matrix["result_rank"][improve, old_index]
            best_source[improve] = f"{prefix}:{old_index}"
    query_position = {int(query): index for index, query in enumerate(queries)}
    for row in pd.read_csv(e9_table_path).itertuples(index=False):
        local = query_position.get(int(row.query_index))
        if local is not None and float(row.frozen_margin) > best_margin[local]:
            best_margin[local] = float(row.frozen_margin)
            best_rank[local] = int(row.frozen_rank)
            best_source[local] = f"N:{row.selector}|{int(row.step)}"
    old_cells = old11["cells"]
    for old_index in range(len(old_cells)):
        if int(old_cells[old_index, 4]) != 0:
            continue
        improve = old11["result_margin"][:, old_index] > best_margin
        best_margin[improve] = old11["result_margin"][improve, old_index]
        best_rank[improve] = old11["result_rank"][improve, old_index]
        best_source[improve] = f"E11:{old_index}"
    e11_oracle = pd.read_csv(e11_oracle_path).sort_values("query_index", kind="stable")
    if (not np.array_equal(e11_oracle["query_index"].to_numpy(np.int64), queries)
            or not np.array_equal(e11_oracle["e11_oracle_rank"].to_numpy(np.int16), best_rank)):
        raise RuntimeError("E12-B failed to reproduce the E11 union")
    old_oracle_rank = best_rank.copy()
    for cell in positive_indices:
        improve = result_margin[:, cell] > best_margin
        best_margin[improve] = result_margin[improve, cell]
        best_rank[improve] = result_rank[improve, cell]
        policy, maximum, dose, weighted, _ = cells[cell]
        best_source[improve] = f"E12B:{cell_name(policy, int(maximum), float(dose), bool(weighted))}"

    official_correct = source["baseline_rank"].to_numpy(int) == 1
    old_correct, oracle_correct = old_oracle_rank == 1, best_rank == 1
    incremental = oracle_correct.astype(float) - old_correct.astype(float)
    total_effect = oracle_correct.astype(float) - official_correct.astype(float)
    total_delta = float(np.mean(total_effect))
    passing = summary.loc[summary["fixed_cell_pass"], "cell_id"].astype(str).tolist()
    report = {
        "status": "noise_final_e12b_relaxed_recurrence_complete", "formal": True,
        "held_queries": int(len(queries)), "held_formulas": int(source["query_formula"].nunique()),
        "reference_policies": list(POLICIES), "prevalence": 0.50,
        "cells": len(positive_definitions), "direction_controls": len(positive_definitions),
        "passing_fixed_cells": passing, "best_fixed_cell": summary.iloc[0].to_dict(),
        "e11_union_reproduction_mismatches": int(np.sum(old_oracle_rank != e11_oracle["e11_oracle_rank"].to_numpy(np.int16))),
        "union_headroom": {
            "e11_total_delta_over_official": float(e11["union_headroom"]["total_delta_over_official"]),
            "new_unique_corrected_beyond_e11": int(np.sum(~old_correct & oracle_correct)),
            "newly_lost_vs_e11": int(np.sum(old_correct & ~oracle_correct)),
            "incremental_delta_over_e11": float(np.mean(incremental)),
            "incremental_formula_ci": formula_bootstrap_mean(
                incremental, formulas, args.bootstrap_resamples, args.seed + 9000,
            ),
            "total_delta_over_official": total_delta,
            "total_formula_ci": formula_bootstrap_mean(
                total_effect, formulas, args.bootstrap_resamples, args.seed + 9001,
            ),
            "remaining_oracle_errors": int(np.sum(~oracle_correct)),
            "selected_source_counts": {
                str(key): int(value) for key, value in pd.Series(best_source).value_counts().items()
            },
            "reaches_five_total_points": bool(total_delta >= 0.05),
        },
        "gates": {"at_least_one_relaxed_fixed_cell_passes": bool(passing),
                  "expanded_union_reaches_five_points": bool(total_delta >= 0.05)},
        "pass_to_conditional_noise_training": bool(passing and total_delta >= 0.05),
        "decision": (
            "freeze mature and relaxed P/N actions and begin conditional shared-encoder noise training"
            if passing and total_delta >= 0.05
            else "relaxed recurrence did not supply both fixed safety and five-point capacity"
        ),
        "contracts": {"e12a_authorized": True, "prevalence_fixed_before_outcomes": 0.50,
                      "wrong_candidate_direction_control": True,
                      "complete_held_task_scored": True,
                      "outcome_used_only_for_union_headroom": True,
                      "P2b": "forbidden", "P3_consumed": False},
        "provenance": {"graph_sha256": sha256_file(args.graph),
                       "student_checkpoint_sha256": checkpoint_hash,
                       "e11_report_sha256": sha256_file(e11_report_path),
                       "e11_matrix_sha256": sha256_file(e11_matrix_path),
                       "e12a_report_sha256": sha256_file(e12a_report_path),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Consumed held-fold action capacity audit; not a trained encoder or deployable gain.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
    pd.DataFrame({"query_index": queries, "query_formula": formulas,
                  "mature_e8_rank": clean_rank, "e11_oracle_rank": old_oracle_rank,
                  "e12b_oracle_rank": best_rank, "oracle_source": best_source}).to_csv(
        args.output_dir / "oracle_per_query.csv.gz", index=False, compression="gzip",
    )
    np.savez_compressed(args.output_dir / "matrix.npz", queries=queries,
                        cells=np.asarray(cells, dtype=object), result_rank=result_rank,
                        result_margin=result_margin, clean_rank=clean_rank, clean_margin=clean_margin)
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

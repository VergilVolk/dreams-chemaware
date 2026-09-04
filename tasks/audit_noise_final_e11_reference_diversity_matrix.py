"""E11: reference-diversity expansion of mature positive-guided noise.

E10-B used only the three most similar same-identity references. E11 holds the
four strongest mature action recipes fixed and varies only how real positive
references are selected: farthest-3, embedding max-min-6, acquisition-condition
diverse-6, and embedding max-min-12. The hardest wrong molecule receives the
same selection policy as a direction control. All variants are evaluated by the
same frozen mature E8 shared encoder on the complete consumed formula fold.

The no-op-aware union is an outcome-aware capacity bound, never a model result.
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

from audit_noise_final_e10_positive_residual_matrix import (  # noqa: E402
    cell_variant, cell_id,
)
from audit_noise_final_e9_action_staleness import load_student, rank_margin  # noqa: E402
from audit_noise_final_positive_guided_matrix import reference_profile  # noqa: E402
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks  # noqa: E402
from calibrate_noise_final_e1_empirical import clean_instrument, condition_relation, decode  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, formula_bootstrap_mean, forward_embeddings,
)


REFERENCE_POLICIES = ("farthest3", "maxmin6", "condition6", "maxmin12")
RECIPES = (
    ("recurrent_union_mix", 0.50, 0.0),
    ("balanced_peak_exchange", 0.50, 0.0),
    ("consensus_then_union", 0.75, 0.50),
    ("transport_then_union", 1.00, 0.50),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--e9-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e9_action_staleness")
    parser.add_argument("--e10b-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e10b_positive_action_expansion")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e11_reference_diversity")
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-reference-prevalence", type=float, default=0.67)
    parser.add_argument("--maximum-transferred-peaks", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def maxmin_indices(vectors: np.ndarray, query_scores: np.ndarray, count: int) -> np.ndarray:
    """Deterministic cosine max-min subset, seeded by the closest reference."""
    count = min(int(count), len(vectors))
    if count < 1:
        return np.empty(0, dtype=np.int64)
    selected = [int(np.argmax(query_scores))]
    available = np.ones(len(vectors), dtype=bool)
    available[selected[0]] = False
    while len(selected) < count:
        similarity = vectors @ vectors[np.asarray(selected)].T
        maximum_similarity = np.max(similarity, axis=1)
        maximum_similarity[~available] = np.inf
        chosen = int(np.argmin(maximum_similarity))
        selected.append(chosen)
        available[chosen] = False
    return np.asarray(selected, dtype=np.int64)


def condition_priority(query_inst: str, query_ce: float, ref_inst: str, ref_ce: float) -> int:
    relation = condition_relation(query_inst, query_ce, ref_inst, ref_ce)
    order = {
        "cross_instrument": 0,
        "same_instrument_cross_ce": 1,
        "same_instrument_unknown_ce": 2,
        "unknown_instrument": 3,
        "same_instrument_same_ce": 4,
    }
    return order[relation]


def select_rows(rows: np.ndarray, scores: np.ndarray, vectors: np.ndarray, policy: str,
                query_inst: str, query_ce: float, instruments: dict[int, str],
                collision_energy: dict[int, float]) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if not len(rows):
        raise RuntimeError("reference policy received no candidate spectra")
    if policy == "farthest3":
        chosen = np.argsort(scores, kind="stable")[:3]
    elif policy in {"maxmin6", "maxmin12"}:
        chosen = maxmin_indices(vectors, scores, 6 if policy == "maxmin6" else 12)
    elif policy == "condition6":
        buckets: list[list[int]] = [[] for _ in range(5)]
        for index, row in enumerate(rows):
            priority = condition_priority(
                query_inst, query_ce, instruments[int(row)], collision_energy[int(row)],
            )
            buckets[priority].append(index)
        for bucket in buckets:
            bucket.sort(key=lambda index: (scores[index], int(rows[index])))
        chosen_list: list[int] = []
        while len(chosen_list) < min(6, len(rows)):
            changed = False
            for bucket in buckets:
                if bucket and len(chosen_list) < min(6, len(rows)):
                    chosen_list.append(bucket.pop(0))
                    changed = True
            if not changed:
                break
        chosen = np.asarray(chosen_list, dtype=np.int64)
    else:
        raise ValueError(f"unknown reference policy: {policy}")
    return rows[np.asarray(chosen, dtype=np.int64)]


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E11 output: {args.output_dir}")
    source_path = args.student_checkpoint.parent / "held_per_query.csv.gz"
    e9_report_path, e9_table_path = args.e9_dir / "report.json", args.e9_dir / "per_action.csv.gz"
    e10b_report_path = args.e10b_dir / "report.json"
    e10b_matrix_path = args.e10b_dir / "matrix.npz"
    e10b_oracle_path = args.e10b_dir / "oracle_per_query.csv.gz"
    required = [args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
                args.student_checkpoint, source_path, e9_report_path, e9_table_path,
                e10b_report_path, e10b_matrix_path, e10b_oracle_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E11 requires CUDA")
    e9 = json.loads(e9_report_path.read_text(encoding="utf-8"))
    e10b = json.loads(e10b_report_path.read_text(encoding="utf-8"))
    if e9.get("status") != "noise_final_e9_action_staleness_complete" or not e9.get("formal"):
        raise RuntimeError("E11 requires formal E9")
    if e10b.get("status") != "noise_final_e10b_positive_action_expansion_complete" or not e10b.get("formal"):
        raise RuntimeError("E11 requires formal E10-B")
    if e10b.get("provenance", {}).get("student_checkpoint_sha256") != sha256_file(args.student_checkpoint):
        raise RuntimeError("E10-B and E11 student checkpoints differ")

    source = pd.read_csv(source_path).sort_values("query_index", kind="stable").reset_index(drop=True)
    if len(source) != 5923 or source["query_index"].duplicated().any():
        raise RuntimeError("E11 expects the complete 5,923-query mature E8 held task")
    queries = source["query_index"].to_numpy(np.int64)
    graph = CandidateGraph(args.graph)
    if not np.array_equal(graph.query_row[queries], source["query_row"].to_numpy(np.int64)):
        raise RuntimeError("E11 source rows drifted from the graph")
    needed_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row]))
    store = SpectrumStore(args.data, needed_rows, 100)
    with h5py.File(args.data, "r") as handle:
        inst_values = decode(handle["INSTRUMENT_TYPE"][store.rows])
        ce_values = np.asarray(handle["COLLISION_ENERGY"][store.rows], dtype=float)
    instruments = {int(row): clean_instrument(str(value)) for row, value in zip(store.rows, inst_values)}
    collision_energy = {int(row): float(value) for row, value in zip(store.rows, ce_values)}
    model = load_student(args, device)
    embeddings = encode_rows(model, store, store.rows, device, args.batch_size, args.amp, "E11-student")
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
        query_policy_rows: list[list[np.ndarray]] = []
        for policy in REFERENCE_POLICIES:
            policy_kinds = []
            for molecule in (0, wrong):
                left, right = map(int, ptr[molecule:molecule + 2])
                local_rows = np.asarray(rows[left:right], dtype=np.int64)
                local_vectors = candidate_vectors[left:right]
                selected = select_rows(
                    local_rows, pair_scores[left:right], local_vectors, policy,
                    instruments[qrow], collision_energy[qrow], instruments, collision_energy,
                )
                policy_kinds.append(selected)
            query_policy_rows.append(policy_kinds)
        reference_rows.append(query_policy_rows)
    if int(np.sum(clean_rank != source["final_rank"].to_numpy(np.int16))):
        raise RuntimeError("E11 failed to reproduce mature E8 ranks")

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
                    clean, references, args.fragment_tolerance,
                    args.minimum_reference_prevalence, args.maximum_transferred_peaks,
                ))
            query_profiles.append(policy_profiles)
            query_missing.append(policy_missing)
        profiles.append(query_profiles)
        missing_peaks.append(query_missing)
        if (local + 1) % 1000 == 0 or local + 1 == len(queries):
            print(f"[E11 profiles] {local + 1:,}/{len(queries):,}", flush=True)

    cells = tuple((policy, family, dose, aux, kind)
                  for policy in REFERENCE_POLICIES
                  for family, dose, aux in RECIPES for kind in range(2))
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
                policy, family, dose, aux, kind = cells[int(local_cell)]
                policy_index = REFERENCE_POLICIES.index(policy)
                query = int(queries[int(local_query)])
                variants.append(cell_variant(
                    store.one(int(graph.query_row[query])),
                    profiles[int(local_query)][policy_index][int(kind)],
                    missing_peaks[int(local_query)][policy_index][int(kind)],
                    family, dose, aux,
                ))
            vectors = forward_embeddings(model, torch.stack(variants).to(device), args.amp).float().cpu().numpy()
            for vector, local_query, local_cell in zip(vectors, local_queries, local_cells):
                rank, margin, _ = rank_margin(
                    graph, int(queries[int(local_query)]), vector, embeddings, embedding_index,
                )
                result_rank[int(local_query), int(local_cell)] = rank
                result_margin[int(local_query), int(local_cell)] = margin
            if right % 25000 < args.batch_size or right == total:
                print(f"[E11 actions] {right:,}/{total:,}", flush=True)

    formulas = source["query_formula"].astype(str).to_numpy()
    clean_correct = clean_rank == 1
    records, positive_indices = [], []
    for index, (policy, family, dose, aux) in enumerate(
        (item for policy in REFERENCE_POLICIES for item in ((policy,) + recipe for recipe in RECIPES))
    ):
        target_cell = cells.index((policy, family, dose, aux, 0))
        control_cell = cells.index((policy, family, dose, aux, 1))
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
            "cell_id": f"{policy}|{cell_id(family, dose, aux)}", "reference_policy": policy,
            "family": family, "dose": dose, "auxiliary_dose": aux,
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

    # Reconstruct the exact E10-B union before adding E11 actions.
    old = np.load(e10b_matrix_path, allow_pickle=True)
    if not np.array_equal(old["queries"], queries) or not np.array_equal(old["clean_rank"], clean_rank):
        raise RuntimeError("E10-B matrix query/clean state drifted")
    best_margin = clean_margin.astype(np.float64, copy=True)
    best_rank = clean_rank.astype(np.int16, copy=True)
    best_source = np.asarray(["no_op"] * len(queries), dtype=object)
    old_cells = old["cells"]
    for old_index in range(len(old_cells)):
        if int(old_cells[old_index, 3]) != 0:
            continue
        improve = old["result_margin"][:, old_index] > best_margin
        best_margin[improve] = old["result_margin"][improve, old_index]
        best_rank[improve] = old["result_rank"][improve, old_index]
        best_source[improve] = f"E10B:{old_index}"
    query_position = {int(query): index for index, query in enumerate(queries)}
    e9_table = pd.read_csv(e9_table_path)
    for row in e9_table.itertuples(index=False):
        local = query_position.get(int(row.query_index))
        if local is not None and float(row.frozen_margin) > best_margin[local]:
            best_margin[local] = float(row.frozen_margin)
            best_rank[local] = int(row.frozen_rank)
            best_source[local] = f"N:{row.selector}|{int(row.step)}"
    e10b_oracle = pd.read_csv(e10b_oracle_path).sort_values("query_index", kind="stable")
    if not np.array_equal(e10b_oracle["query_index"].to_numpy(np.int64), queries):
        raise RuntimeError("E10-B oracle query order drifted")
    old_oracle_rank = best_rank.copy()
    if not np.array_equal(old_oracle_rank, e10b_oracle["oracle_rank"].to_numpy(np.int16)):
        raise RuntimeError("E11 failed to reproduce the E10-B union")
    for cell in positive_indices:
        improve = result_margin[:, cell] > best_margin
        best_margin[improve] = result_margin[improve, cell]
        best_rank[improve] = result_rank[improve, cell]
        policy, family, dose, aux, _ = cells[cell]
        best_source[improve] = f"E11:{policy}|{cell_id(family, dose, aux)}"

    official_correct = source["baseline_rank"].to_numpy(int) == 1
    oracle_correct = best_rank == 1
    old_correct = old_oracle_rank == 1
    incremental_e11 = oracle_correct.astype(float) - old_correct.astype(float)
    total_effect = oracle_correct.astype(float) - official_correct.astype(float)
    total_delta = float(np.mean(total_effect))
    passing = summary.loc[summary["fixed_cell_pass"], "cell_id"].astype(str).tolist()
    report = {
        "status": "noise_final_e11_reference_diversity_complete", "formal": True,
        "held_queries": int(len(queries)), "held_formulas": int(source["query_formula"].nunique()),
        "reference_policies": list(REFERENCE_POLICIES), "recipes_per_policy": len(RECIPES),
        "cells": len(REFERENCE_POLICIES) * len(RECIPES),
        "direction_controls": len(REFERENCE_POLICIES) * len(RECIPES),
        "passing_fixed_cells": passing, "best_fixed_cell": summary.iloc[0].to_dict(),
        "e10b_union_reproduction_mismatches": int(np.sum(old_oracle_rank != e10b_oracle["oracle_rank"].to_numpy(np.int16))),
        "union_headroom": {
            "e10b_total_delta_over_official": float(e10b["no_op_aware_union_headroom"]["total_delta_over_official"]),
            "new_unique_corrected_beyond_e10b": int(np.sum(~old_correct & oracle_correct)),
            "newly_lost_vs_e10b": int(np.sum(old_correct & ~oracle_correct)),
            "incremental_delta_over_e10b": float(np.mean(incremental_e11)),
            "incremental_formula_ci": formula_bootstrap_mean(
                incremental_e11, formulas, args.bootstrap_resamples, args.seed + 9000,
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
        "gates": {"at_least_one_diverse_fixed_cell_passes": bool(passing),
                  "expanded_union_reaches_five_points": bool(total_delta >= 0.05)},
        "pass_to_conditional_noise_training": bool(passing and total_delta >= 0.05),
        "decision": (
            "freeze condition-diverse P/N action curriculum and begin shared-encoder transfer"
            if passing and total_delta >= 0.05
            else "reference diversity alone is insufficient; audit remaining residual mechanisms before training"
        ),
        "contracts": {"only_reference_selection_changed": True,
                      "real_same_identity_positive_references": True,
                      "wrong_candidate_direction_control": True,
                      "one_shared_mature_embedding_geometry": True,
                      "outcome_used_only_for_union_headroom": True,
                      "P2b": "forbidden", "P3_consumed": False},
        "provenance": {"graph_sha256": sha256_file(args.graph),
                       "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
                       "e10b_report_sha256": sha256_file(e10b_report_path),
                       "e10b_matrix_sha256": sha256_file(e10b_matrix_path),
                       "e9_per_action_sha256": sha256_file(e9_table_path),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Consumed held-fold action capacity audit; not a trained encoder or deployable gain.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
    pd.DataFrame({"query_index": queries, "query_formula": formulas,
                  "mature_e8_rank": clean_rank, "e10b_oracle_rank": old_oracle_rank,
                  "e11_oracle_rank": best_rank, "oracle_source": best_source}).to_csv(
        args.output_dir / "oracle_per_query.csv.gz", index=False, compression="gzip",
    )
    np.savez_compressed(args.output_dir / "matrix.npz", queries=queries,
                        cells=np.asarray(cells, dtype=object), result_rank=result_rank,
                        result_margin=result_margin, clean_rank=clean_rank, clean_margin=clean_margin)
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

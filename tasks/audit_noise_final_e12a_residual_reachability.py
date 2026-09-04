"""E12-A: diagnose the 133 errors left by the E11 outcome-aware union.

No new perturbation is scored here. For each residual error this audit asks:
1) can a real same-identity reference, with self-match excluded, already induce
   a correct candidate ranking under the mature E8 geometry; and
2) how many recurrent missing peaks become available as reference prevalence is
   relaxed from 0.67 to 0.50/0.34 under top-3 and diverse reference policies.

This is a necessary reachability gate for E12-B, not a model result.
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

from audit_noise_final_e11_reference_diversity_matrix import (  # noqa: E402
    REFERENCE_POLICIES, select_rows,
)
from audit_noise_final_e9_action_staleness import load_student  # noqa: E402
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks  # noqa: E402
from calibrate_noise_final_e1_empirical import clean_instrument, decode  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows  # noqa: E402


POLICIES = ("top3",) + REFERENCE_POLICIES
PREVALENCE = (0.67, 0.50, 0.34)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--e11-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e11_reference_diversity")
    parser.add_argument("--error-signatures", type=Path, default=ROOT / "data/validation/g8r_real_error_analysis/query_error_signatures.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e12a_residual_reachability")
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def positive_teacher_rank(graph: CandidateGraph, query: int, teacher_row: int,
                          embeddings: np.ndarray, embedding_index: dict[int, int]) -> int | None:
    _, rows, ptr, _ = graph.query_block(query)
    vector = embeddings[embedding_index[int(teacher_row)]]
    scores = embeddings[[embedding_index[int(row)] for row in rows]] @ vector
    positive_rows = np.asarray(rows[: int(ptr[1])], dtype=np.int64)
    keep_positive = positive_rows != int(teacher_row)
    if not np.any(keep_positive):
        return None
    molecule_scores = [float(np.max(scores[: int(ptr[1])][keep_positive]))]
    for left, right in zip(ptr[1:-1], ptr[2:]):
        molecule_scores.append(float(np.max(scores[int(left):int(right)])))
    values = np.asarray(molecule_scores, dtype=float)
    return 1 + int(np.sum(values[1:] >= values[0]))


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E12-A output: {args.output_dir}")
    report_path, oracle_path = args.e11_dir / "report.json", args.e11_dir / "oracle_per_query.csv.gz"
    required = [args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
                args.student_checkpoint, report_path, oracle_path, args.error_signatures]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E12-A requires CUDA")
    e11 = json.loads(report_path.read_text(encoding="utf-8"))
    if e11.get("status") != "noise_final_e11_reference_diversity_complete" or not e11.get("formal"):
        raise RuntimeError("E12-A requires formal E11")
    if e11.get("provenance", {}).get("student_checkpoint_sha256") != sha256_file(args.student_checkpoint):
        raise RuntimeError("E11 and E12-A checkpoints differ")
    oracle = pd.read_csv(oracle_path).sort_values("query_index", kind="stable").reset_index(drop=True)
    residual = oracle.loc[oracle["e11_oracle_rank"].astype(int).ne(1)].copy()
    if len(oracle) != 5923 or len(residual) != int(e11["union_headroom"]["remaining_oracle_errors"]):
        raise RuntimeError("E12-A residual count does not reproduce E11")
    # Convert the full-task E11 point estimate back to an exact net-query count;
    # never compare a residual subset proportion to the full-task five-point target.
    current_net = int(round(float(e11["union_headroom"]["total_delta_over_official"]) * len(oracle)))
    required_new = max(0, int(np.ceil(0.05 * len(oracle))) - current_net)

    graph = CandidateGraph(args.graph)
    residual_queries = residual["query_index"].to_numpy(np.int64)
    needed_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row]))
    store = SpectrumStore(args.data, needed_rows, 100)
    with h5py.File(args.data, "r") as handle:
        inst_values = decode(handle["INSTRUMENT_TYPE"][store.rows])
        ce_values = np.asarray(handle["COLLISION_ENERGY"][store.rows], dtype=float)
    instruments = {int(row): clean_instrument(str(value)) for row, value in zip(store.rows, inst_values)}
    collision_energy = {int(row): float(value) for row, value in zip(store.rows, ce_values)}
    model = load_student(args, device)
    embeddings = encode_rows(model, store, store.rows, device, args.batch_size, args.amp, "E12A-student")
    embedding_index = {int(row): index for index, row in enumerate(store.rows)}

    records = []
    for position, query in enumerate(residual_queries, start=1):
        _, rows, ptr, _ = graph.query_block(int(query))
        qrow = int(graph.query_row[int(query)])
        qvector = embeddings[embedding_index[qrow]]
        positive_rows = np.asarray(rows[: int(ptr[1])], dtype=np.int64)
        positive_vectors = embeddings[[embedding_index[int(row)] for row in positive_rows]]
        positive_scores = positive_vectors @ qvector
        item: dict[str, object] = {
            "query_index": int(query), "query_row": qrow,
            "query_formula": str(graph.query_formula[int(query)]),
            "query_ik14": str(graph.query_ik14[int(query)]),
            "query_has_near": bool(graph.query_has_near[int(query)]),
            "positive_reference_count": int(len(positive_rows)),
        }
        teacher_ranks = [
            rank for rank in (
                positive_teacher_rank(graph, int(query), int(row), embeddings, embedding_index)
                for row in positive_rows
            ) if rank is not None
        ]
        item["support_disjoint_teacher_available"] = bool(teacher_ranks)
        item["support_disjoint_teacher_best_rank"] = min(teacher_ranks) if teacher_ranks else -1
        item["support_disjoint_teacher_correct"] = bool(teacher_ranks and min(teacher_ranks) == 1)
        for policy in POLICIES:
            if policy == "top3":
                selected = positive_rows[np.argsort(-positive_scores, kind="stable")[:3]]
            else:
                selected = select_rows(
                    positive_rows, positive_scores, positive_vectors, policy,
                    instruments[qrow], collision_energy[qrow], instruments, collision_energy,
                )
            references = [store.one(int(row)) for row in selected]
            for prevalence in PREVALENCE:
                peaks = recurrent_missing_peaks(
                    store.one(qrow), references, args.fragment_tolerance, prevalence, 100,
                )
                item[f"missing_{policy}_p{prevalence:.2f}"] = int(len(peaks))
        records.append(item)
        if position % 25 == 0 or position == len(residual_queries):
            print(f"[E12-A] {position:,}/{len(residual_queries):,}", flush=True)
    detail = pd.DataFrame(records)

    signatures = pd.read_csv(args.error_signatures)
    if signatures["query_index"].duplicated().any():
        raise RuntimeError("error signature ledger is not one row per query")
    available_columns = [column for column in (
        "query_index", "score_error_family", "positive_deficit", "negative_excess",
        "shared_major_peak_screen", "neutral_loss_convergence_screen",
        "cross_condition_positive_screen", "raw_evidence_can_rescue",
        "rules_favor_positive", "rules_favor_wrong",
    ) if column in signatures.columns]
    detail = detail.merge(signatures[available_columns], on="query_index", how="left", validate="one_to_one")
    missing_columns = [column for column in detail.columns if column.startswith("missing_")]
    reachability = {}
    for prevalence in PREVALENCE:
        columns = [column for column in missing_columns if column.endswith(f"p{prevalence:.2f}")]
        maximum = detail[columns].max(axis=1).to_numpy(int)
        reachability[f"prevalence_{prevalence:.2f}"] = {
            "queries_with_any_missing_peak": int(np.sum(maximum >= 1)),
            "queries_with_at_least_5_missing_peaks": int(np.sum(maximum >= 5)),
            "queries_with_at_least_10_missing_peaks": int(np.sum(maximum >= 10)),
            "median_max_missing_peaks": float(np.median(maximum)),
            "p90_max_missing_peaks": float(np.quantile(maximum, 0.9)),
        }
    teacher_recoverable = int(detail["support_disjoint_teacher_correct"].sum())
    report = {
        "status": "noise_final_e12a_residual_reachability_complete", "formal": True,
        "held_queries": int(len(oracle)), "residual_errors": int(len(detail)),
        "additional_unique_errors_required_for_five_points": int(required_new),
        "near_residual_errors": int(detail["query_has_near"].sum()),
        "score_error_families": {
            str(key): int(value) for key, value in detail.get(
                "score_error_family", pd.Series(dtype=str)
            ).value_counts(dropna=False).items()
        },
        "support_disjoint_positive_teacher": {
            "available": int(detail["support_disjoint_teacher_available"].sum()),
            "correct": teacher_recoverable,
            "fraction_of_residual_correct": teacher_recoverable / len(detail) if len(detail) else 0.0,
        },
        "recurrent_missing_peak_reachability": reachability,
        "gates": {
            "support_disjoint_teacher_covers_required_gap": bool(teacher_recoverable >= required_new),
            "relaxed_recurrence_has_required_eligible_queries": bool(
                reachability["prevalence_0.50"]["queries_with_any_missing_peak"] >= required_new
            ),
        },
        "pass_to_e12b_relaxed_recurrence_matrix": bool(
            teacher_recoverable >= required_new
            and reachability["prevalence_0.50"]["queries_with_any_missing_peak"] >= required_new
        ),
        "decision": "run fixed relaxed-recurrence matrix" if (
            teacher_recoverable >= required_new
            and reachability["prevalence_0.50"]["queries_with_any_missing_peak"] >= required_new
        ) else "relaxed recurrence lacks sufficient reachability; design a different residual mechanism",
        "contracts": {"self_match_excluded_from_positive_teacher": True,
                      "no_new_action_outcomes_scored": True,
                      "P2b": "forbidden", "P3_consumed": False},
        "provenance": {"graph_sha256": sha256_file(args.graph),
                       "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
                       "e11_report_sha256": sha256_file(report_path),
                       "e11_oracle_sha256": sha256_file(oracle_path),
                       "error_signatures_sha256": sha256_file(args.error_signatures),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Residual supervision reachability audit; no action gain or trained embedding result.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail.to_csv(args.output_dir / "residual_detail.csv.gz", index=False, compression="gzip")
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

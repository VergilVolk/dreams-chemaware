"""Expand positive-evidence supervision with support-disjoint identity cross-fitting.

For each P3-disjoint real query with at least two positive reference spectra,
one positive spectrum is held out for ranking and all remaining positive rows
form the identity teacher prototype.  Teacher rows are masked from the positive
candidate score.  This removes the direct prototype/reference overlap present
in a simple identity-centroid upper bound while expanding supervision across
the complete 23,876-query candidate graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from audit_noise_v3_a4_exact_peak_scan import load_embeddings, query_candidate_block, strict_detail
from build_g8r_real_error_atlas import Cache
from diagnose_noise_v3_a4b_positive_evidence import cluster_bootstrap, normalized_mean


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--maximum-teacher-spectra", type=int, default=12)
    parser.add_argument("--maximum-holdouts-per-query", type=int, default=4)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--formula-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def mix(clean: np.ndarray, prototype: np.ndarray, alpha: float) -> np.ndarray:
    value = (1.0 - alpha) * clean.astype(np.float64) + alpha * prototype.astype(np.float64)
    return (value / np.clip(np.linalg.norm(value), 1e-12, None)).astype(np.float32)


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1 or args.maximum_teacher_spectra < 1 or args.maximum_holdouts_per_query < 1:
        raise ValueError("invalid C1 teacher parameters")
    for path in (args.graph, args.embeddings):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    graph = Cache(args.graph)
    score_column = graph.feature_names.index("dreams_similarity")
    embedding_rows, embeddings, embedding_index = load_embeddings(args.embeddings)
    query_limit = min(graph.n_queries, args.max_queries or graph.n_queries)
    rng = np.random.default_rng(args.seed)
    records = []
    skipped_positive_lt2 = 0
    missing_embedding = 0
    for query in range(query_limit):
        official, candidate_rows, ptr, molecule_left = query_candidate_block(graph, query, score_column)
        pos_left, pos_right = map(int, ptr[:2])
        positive_pair_indices = np.arange(pos_left, pos_right, dtype=np.int64)
        usable = np.asarray([
            index for index in positive_pair_indices if int(candidate_rows[index]) in embedding_index
        ], dtype=np.int64)
        if len(usable) < 2:
            skipped_positive_lt2 += 1
            continue
        order = rng.permutation(len(usable))[:args.maximum_holdouts_per_query]
        candidate_positions = np.asarray([
            embedding_index[int(value)] for value in candidate_rows
        ], dtype=np.int64) if all(int(value) in embedding_index for value in candidate_rows) else None
        if candidate_positions is None:
            missing_embedding += 1
            continue
        candidate_embedding = embeddings[candidate_positions]
        query_row = int(graph.query_row[query])
        query_position = embedding_index.get(query_row)
        if query_position is None:
            missing_embedding += 1
            continue
        clean = embeddings[query_position]
        recomputed_clean = candidate_embedding @ clean
        for local_holdout in order:
            eval_pair = int(usable[int(local_holdout)])
            teacher_pairs = [int(value) for value in usable if int(value) != eval_pair]
            teacher_pairs = teacher_pairs[:args.maximum_teacher_spectra]
            teacher_rows = np.asarray(candidate_rows[teacher_pairs], dtype=np.int64)
            teacher_positions = np.asarray([embedding_index[int(value)] for value in teacher_rows])
            prototype = normalized_mean(embeddings[teacher_positions])
            teacher_embedding = mix(clean, prototype, args.alpha)

            allowed = np.ones(len(official), dtype=bool)
            allowed[pos_left:pos_right] = False
            allowed[eval_pair] = True
            baseline_scores = np.asarray(official, dtype=np.float64).copy()
            baseline_scores[~allowed] = -1e6
            baseline = strict_detail(baseline_scores, candidate_rows, ptr)
            teacher_scores = np.asarray(official, dtype=np.float64) + (
                candidate_embedding @ teacher_embedding - recomputed_clean
            )
            teacher_scores[~allowed] = -1e6
            target = strict_detail(teacher_scores, candidate_rows, ptr)

            negative_local = int(baseline["adversarial_molecule_local"])
            neg_left, neg_right = map(int, ptr[negative_local:negative_local + 2])
            negative_rows = np.asarray(candidate_rows[neg_left:neg_right], dtype=np.int64)
            negative_positions = np.asarray([embedding_index[int(value)] for value in negative_rows])
            wrong_prototype = normalized_mean(embeddings[negative_positions[:args.maximum_teacher_spectra]])
            wrong_embedding = mix(clean, wrong_prototype, args.alpha)
            wrong_scores = np.asarray(official, dtype=np.float64) + (
                candidate_embedding @ wrong_embedding - recomputed_clean
            )
            wrong_scores[~allowed] = -1e6
            wrong = strict_detail(wrong_scores, candidate_rows, ptr)
            baseline_correct = int(baseline["rank"]) == 1
            target_correct = int(target["rank"]) == 1
            records.append({
                "query_index": query, "query_row": query_row,
                "query_ik14": str(graph.query_ik14[query]),
                "query_formula": str(graph.query_formula[query]),
                "has_near": bool(graph.query_has_near[query]),
                "evaluation_positive_row": int(candidate_rows[eval_pair]),
                "teacher_rows": ";".join(map(str, teacher_rows.tolist())),
                "teacher_spectra": int(len(teacher_rows)),
                "candidate_molecules": int(len(ptr) - 1),
                "adversarial_mces_grade": int(
                    graph.molecule_mces_grade[molecule_left + negative_local]
                ),
                "baseline_rank": int(baseline["rank"]),
                "baseline_margin": float(baseline["margin"]),
                "teacher_rank": int(target["rank"]),
                "teacher_margin": float(target["margin"]),
                "wrong_identity_rank": int(wrong["rank"]),
                "wrong_identity_margin": float(wrong["margin"]),
                "corrected": bool((not baseline_correct) and target_correct),
                "introduced": bool(baseline_correct and (not target_correct)),
                "teacher_beats_wrong_margin": float(target["margin"] - wrong["margin"]),
            })
        if (query + 1) % 1000 == 0 or query + 1 == query_limit:
            print(f"[C1] {query + 1:,}/{query_limit:,} graph queries; examples={len(records):,}", flush=True)
    frame = pd.DataFrame(records)
    formal = args.max_queries == 0
    if frame.empty:
        raise RuntimeError("C1 generated no support-disjoint examples")
    if formal and query_limit != 23876:
        raise RuntimeError(f"formal C1 expects 23,876 graph queries, got {query_limit}")
    formulas = frame["query_formula"].astype(str).to_numpy()
    unique_formulas = np.unique(formulas)
    formula_to_fold = {
        formula: index % args.formula_folds
        for index, formula in enumerate(rng.permutation(unique_formulas))
    }
    frame["formula_fold"] = [formula_to_fold[value] for value in formulas]
    corrected = frame["corrected"].to_numpy(bool)
    introduced = frame["introduced"].to_numpy(bool)
    contribution = corrected.astype(float) - args.risk_penalty * introduced.astype(float)
    baseline_correct = frame["baseline_rank"].to_numpy(int) == 1
    teacher_correct = frame["teacher_rank"].to_numpy(int) == 1
    teacher_vs_wrong = (
        teacher_correct.astype(float)
        - (frame["wrong_identity_rank"].to_numpy(int) == 1).astype(float)
    )
    near = frame["has_near"].to_numpy(bool)
    summary = {
        "status": "noise_v3_c1_crossfit_teacher_complete", "formal": formal,
        "protocol": "positive evaluation row is disjoint from every identity-prototype teacher row",
        "graph_queries_considered": int(query_limit), "examples": int(len(frame)),
        "query_identities": int(frame["query_ik14"].nunique()),
        "query_formulas": int(frame["query_formula"].nunique()),
        "near_examples": int(near.sum()),
        "skipped_queries_with_lt2_positive_references": int(skipped_positive_lt2),
        "queries_skipped_for_missing_embedding": int(missing_embedding),
        "baseline_accuracy": float(baseline_correct.mean()),
        "teacher_accuracy": float(teacher_correct.mean()),
        "delta_accuracy": float(teacher_correct.mean() - baseline_correct.mean()),
        "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
        "risk_weighted_net": float(contribution.sum()),
        "near_baseline_accuracy": float(baseline_correct[near].mean()),
        "near_teacher_accuracy": float(teacher_correct[near].mean()),
        "near_delta_accuracy": float(teacher_correct[near].mean() - baseline_correct[near].mean()),
        "formula_cluster_risk_net_per_example": cluster_bootstrap(
            frame, contribution, "query_formula", args.bootstrap_resamples, args.seed + 1,
        ),
        "formula_cluster_teacher_vs_wrong_accuracy": cluster_bootstrap(
            frame, teacher_vs_wrong, "query_formula", args.bootstrap_resamples, args.seed + 2,
        ),
        "support_spectra": {
            "median": float(frame["teacher_spectra"].median()),
            "p10": float(frame["teacher_spectra"].quantile(0.1)),
            "p90": float(frame["teacher_spectra"].quantile(0.9)),
        },
        "gates": {
            "examples_ge_10000": bool(len(frame) >= 10000),
            "identities_ge_1000": bool(frame["query_ik14"].nunique() >= 1000),
            "formulas_ge_500": bool(frame["query_formula"].nunique() >= 500),
            "risk_net_formula_ci_positive": bool(
                cluster_bootstrap(frame, contribution, "query_formula", 1000, args.seed + 3)["ci_low"] > 0
            ),
            "teacher_beats_wrong_formula_ci_positive": bool(
                cluster_bootstrap(frame, teacher_vs_wrong, "query_formula", 1000, args.seed + 4)["ci_low"] > 0
            ),
            "near_nonnegative": bool(
                teacher_correct[near].mean() >= baseline_correct[near].mean()
            ),
        },
        "claim_limit": (
            "C1 is a support-disjoint training-space geometry audit. The identity teacher is still "
            "label-supervised and is not deployable retrieval performance."
        ),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "provenance": {"graph_sha256": sha256_file(args.graph), "embeddings_sha256": sha256_file(args.embeddings)},
    }
    summary["pass_to_candidate_aware_student"] = bool(all(summary["gates"].values()))
    staging = Path(tempfile.mkdtemp(prefix="noise_c1_", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "crossfit_examples.csv.gz", index=False, compression="gzip")
        frame.loc[frame["corrected"]].to_csv(
            staging / "crossfit_teacher_rescues.csv.gz", index=False, compression="gzip"
        )
        (staging / "decision.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

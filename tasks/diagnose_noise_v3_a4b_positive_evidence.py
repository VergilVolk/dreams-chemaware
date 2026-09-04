"""A4-B0 diagnostic for real cross-condition positive-evidence recovery.

This is a training-only geometry experiment.  For each P3-disjoint A4 query,
the clean official query embedding is moved by a fixed dose toward a prototype
formed from *real spectra of the same IK14 and adduct*.  The counterfactual is
then ranked against the original strict-10ppm candidate group.  Prototypes
formed from wrong candidate identities in the same group are the matched
negative control.

The experiment tests whether real identity evidence creates sufficiently large
and safe gradient headroom for later adapter distillation.  It is not a deployable
retrieval method and it never adds synthetic peaks to an input spectrum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit_noise_v3_a4_exact_peak_scan import (
    load_embeddings, query_candidate_block, strict_detail,
)
from build_g8r_real_error_atlas import Cache
from train_noise_v3_a4_nonlinear_action_teacher import previous_recoverable


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path,
                        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embedding-cache", type=Path,
                        default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--a4-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan")
    parser.add_argument("--s1c-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_s1c_topk_matrix")
    parser.add_argument("--s2-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_s2_sequential")
    parser.add_argument("--s3a-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_a4b_positive_evidence")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--primary-alpha", type=float, default=0.25)
    parser.add_argument("--maximum-support-spectra", type=int, default=12)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--minimum-new-corrections", type=int, default=80)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-queries", type=int, default=0, help="balanced smoke only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def normalized_mean(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("prototype has zero or non-finite norm")
    return (vector / norm).astype(np.float32)


def mix_embedding(query: np.ndarray, prototype: np.ndarray, alpha: float) -> np.ndarray:
    value = (1.0 - alpha) * query.astype(np.float64) + alpha * prototype.astype(np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("mixed embedding has zero or non-finite norm")
    return (value / norm).astype(np.float32)


def cluster_bootstrap(
    frame: pd.DataFrame, values: np.ndarray, cluster: str, resamples: int, seed: int,
) -> dict[str, float]:
    local = pd.DataFrame({"cluster": frame[cluster].astype(str), "value": values})
    grouped = local.groupby("cluster", sort=False)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    if len(sums) < 2:
        raise RuntimeError(f"bootstrap needs at least two {cluster} clusters")
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draw = rng.integers(0, len(sums), size=len(sums))
        distribution[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(np.sum(sums) / np.sum(counts)),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
    }


def a4_exact_recoverable(a4_dir: Path) -> set[int]:
    queries = pd.read_csv(a4_dir / "scan_queries.csv.gz").sort_values("scan_position")
    with h5py.File(a4_dir / "exact_peak_scan.h5", "r") as handle:
        action_query = handle["action_query"][:].astype(np.int64)
        result_rank = handle["result_rank"][:]
        doses = len(json.loads(handle.attrs["attenuations_json"]))
    recovered_positions = np.unique(action_query[result_rank.reshape(-1, doses).min(axis=1) == 1])
    recovered = queries.set_index("scan_position").loc[recovered_positions]
    return set(map(int, recovered.loc[
        recovered["scan_kind"].eq("official_error"), "query_index"
    ]))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (
        args.graph, args.embedding_cache, args.data,
        args.a4_dir / "scan_queries.csv.gz", args.a4_dir / "exact_peak_scan.h5",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (not args.alphas or any(not 0 < value < 1 for value in args.alphas)
            or args.primary_alpha not in args.alphas):
        raise ValueError("alphas must be unique values in (0,1) and include primary-alpha")
    if len(set(args.alphas)) != len(args.alphas) or args.random_repeats < 1:
        raise ValueError("invalid alpha grid or random repeats")

    graph = Cache(args.graph)
    if "dreams_similarity" not in graph.feature_names:
        raise RuntimeError("candidate graph lacks official DreaMS similarity")
    score_column = graph.feature_names.index("dreams_similarity")
    embedding_rows, embeddings, embedding_index = load_embeddings(args.embedding_cache)

    scan = pd.read_csv(args.a4_dir / "scan_queries.csv.gz").sort_values("scan_position")
    if args.max_queries:
        errors = scan.loc[scan["scan_kind"].eq("official_error")].head((args.max_queries + 1) // 2)
        controls = scan.loc[scan["scan_kind"].eq("safety_control")].head(args.max_queries // 2)
        scan = pd.concat((errors, controls)).sort_values("scan_position").reset_index(drop=True)
    formal = args.max_queries == 0
    if formal and (len(scan) != 4998 or int(scan["scan_kind"].eq("official_error").sum()) != 1805):
        raise RuntimeError("formal B0 expects the complete 4,998-query A4 panel")

    with h5py.File(args.data, "r") as handle:
        ik14 = np.asarray([decode(value)[:14] for value in handle["INCHIKEY"][embedding_rows]], dtype=object)
        adduct = np.asarray([decode(value) for value in handle["adduct"][embedding_rows]], dtype=object)
    group_positions: dict[tuple[str, str], list[int]] = {}
    for position, key in enumerate(zip(ik14, adduct)):
        group_positions.setdefault((str(key[0]), str(key[1])), []).append(position)

    previous = set().union(
        previous_recoverable(args.s1c_dir), previous_recoverable(args.s2_dir),
        previous_recoverable(args.s3a_dir), a4_exact_recoverable(args.a4_dir),
    )
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    skipped_no_support = 0
    skipped_missing_embedding = 0
    baseline_mismatch = 0
    for done, row in enumerate(scan.itertuples(index=False), start=1):
        query = int(row.query_index)
        query_row = int(row.query_row)
        if query >= graph.n_queries or int(graph.query_row[query]) != query_row:
            raise RuntimeError(f"A4/graph query misalignment at {query}")
        query_position = embedding_index.get(query_row)
        if query_position is None:
            skipped_missing_embedding += 1
            continue
        key = (str(row.query_ik14)[:14], str(adduct[query_position]))
        support_positions = [
            value for value in group_positions.get(key, [])
            if int(embedding_rows[value]) != query_row
        ]
        if not support_positions:
            skipped_no_support += 1
            continue
        support_positions = sorted(support_positions, key=lambda value: int(embedding_rows[value]))[
            :args.maximum_support_spectra
        ]
        target_prototype = normalized_mean(embeddings[support_positions])

        base_scores, candidate_rows, molecule_ptr, molecule_left = query_candidate_block(
            graph, query, score_column,
        )
        try:
            candidate_positions = np.asarray(
                [embedding_index[int(value)] for value in candidate_rows], dtype=np.int64,
            )
        except KeyError:
            skipped_missing_embedding += 1
            continue
        candidate_embeddings = embeddings[candidate_positions]
        query_embedding = embeddings[query_position]
        recomputed = candidate_embeddings @ query_embedding
        maximum_similarity_error = float(np.max(np.abs(recomputed - base_scores)))
        if maximum_similarity_error > 2e-4:
            baseline_mismatch += 1
            continue
        # The locked candidate graph is the authoritative official-DreaMS
        # baseline.  Re-encoding/casting the same normalized embeddings can
        # move nearly tied scores by a few ulps and change the project's strict
        # tie-sensitive rank.  Counterfactual scores therefore preserve the
        # exact locked baseline and add only the intervention-induced cosine
        # delta.  At alpha=0 this is exactly base_scores by construction.
        baseline = strict_detail(base_scores, candidate_rows, molecule_ptr)
        if int(baseline["rank"]) != int(row.baseline_rank):
            raise RuntimeError(f"baseline rank mismatch at query {query}")

        negative_molecules = np.arange(1, len(molecule_ptr) - 1, dtype=np.int64)
        if len(negative_molecules) == 0:
            raise RuntimeError("strict candidate group has no negative molecule")
        random_order = rng.permutation(negative_molecules)
        random_prototypes = []
        for repeat in range(args.random_repeats):
            molecule = int(random_order[repeat % len(random_order)])
            left, right = map(int, molecule_ptr[molecule:molecule + 2])
            positions = candidate_positions[left:right][:args.maximum_support_spectra]
            random_prototypes.append(normalized_mean(embeddings[positions]))

        for alpha in args.alphas:
            target_query = mix_embedding(query_embedding, target_prototype, alpha)
            target_scores = base_scores + (candidate_embeddings @ target_query - recomputed)
            target = strict_detail(target_scores, candidate_rows, molecule_ptr)
            random_ranks = []
            random_margins = []
            for prototype in random_prototypes:
                control_query = mix_embedding(query_embedding, prototype, alpha)
                control_scores = base_scores + (candidate_embeddings @ control_query - recomputed)
                detail = strict_detail(control_scores, candidate_rows, molecule_ptr)
                random_ranks.append(int(detail["rank"]))
                random_margins.append(float(detail["margin"]))
            baseline_correct = int(baseline["rank"]) == 1
            target_correct = int(target["rank"]) == 1
            records.append({
                "query_index": query,
                "query_row": query_row,
                "query_ik14": str(row.query_ik14)[:14],
                "query_formula": str(row.query_formula),
                "scan_kind": str(row.scan_kind),
                "positive_deficit": bool(row.positive_deficit),
                "negative_excess": bool(row.negative_excess),
                "has_near": bool(row.has_near),
                "alpha": float(alpha),
                "support_spectra": len(support_positions),
                "baseline_rank": int(baseline["rank"]),
                "baseline_margin": float(baseline["margin"]),
                "target_rank": int(target["rank"]),
                "target_margin": float(target["margin"]),
                "mean_random_accuracy": float(np.mean(np.asarray(random_ranks) == 1)),
                "mean_random_margin": float(np.mean(random_margins)),
                "corrected": bool(not baseline_correct and target_correct),
                "introduced": bool(baseline_correct and not target_correct),
                "new_correction": bool(
                    not baseline_correct and target_correct and query not in previous
                ),
            })
        if done % 250 == 0 or done == len(scan):
            print(f"[A4-B0] {done:,}/{len(scan):,} queries", flush=True)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("A4-B0 produced no aligned query results")
    if baseline_mismatch:
        raise RuntimeError(f"official embedding/cache similarity mismatch: {baseline_mismatch}")

    summaries = []
    for alpha in args.alphas:
        local = frame.loc[frame["alpha"].eq(alpha)].reset_index(drop=True)
        corrected = local["corrected"].to_numpy(bool)
        introduced = local["introduced"].to_numpy(bool)
        contribution = corrected.astype(float) - args.risk_penalty * introduced.astype(float)
        target_correct = local["target_rank"].to_numpy(int) == 1
        baseline_correct = local["baseline_rank"].to_numpy(int) == 1
        target_minus_random = target_correct.astype(float) - local["mean_random_accuracy"].to_numpy(float)
        error = local["scan_kind"].eq("official_error").to_numpy()
        control = local["scan_kind"].eq("safety_control").to_numpy()
        summaries.append({
            "alpha": float(alpha),
            "queries": int(len(local)),
            "errors": int(error.sum()),
            "controls": int(control.sum()),
            "corrected": int(corrected.sum()),
            "introduced": int(introduced.sum()),
            "risk_weighted_net": float(contribution.sum()),
            "new_corrections_beyond_history_and_A4": int(local["new_correction"].sum()),
            "baseline_accuracy": float(baseline_correct.mean()),
            "target_accuracy": float(target_correct.mean()),
            "mean_random_accuracy": float(local["mean_random_accuracy"].mean()),
            "formula_cluster_risk_net_per_query": cluster_bootstrap(
                local, contribution, "query_formula", args.bootstrap_resamples, args.seed,
            ),
            "formula_cluster_target_minus_random_accuracy": cluster_bootstrap(
                local, target_minus_random, "query_formula", args.bootstrap_resamples, args.seed + 1,
            ),
            "error_formula_cluster_target_minus_random_accuracy": cluster_bootstrap(
                local.loc[error].reset_index(drop=True), target_minus_random[error],
                "query_formula", args.bootstrap_resamples, args.seed + 2,
            ),
            "control_formula_cluster_target_minus_random_accuracy": cluster_bootstrap(
                local.loc[control].reset_index(drop=True), target_minus_random[control],
                "query_formula", args.bootstrap_resamples, args.seed + 3,
            ),
            "positive_deficit_corrected": int(
                (corrected & local["positive_deficit"].to_numpy(bool)).sum()
            ),
            "near_corrected": int((corrected & local["has_near"].to_numpy(bool)).sum()),
        })

    primary = next(row for row in summaries if row["alpha"] == args.primary_alpha)
    gates = {
        "new_corrections_ge_minimum": bool(
            primary["new_corrections_beyond_history_and_A4"] >= args.minimum_new_corrections
        ),
        "risk_weighted_formula_ci_positive": bool(
            primary["formula_cluster_risk_net_per_query"]["ci_low"] > 0
        ),
        "error_target_beats_random_formula_ci": bool(
            primary["error_formula_cluster_target_minus_random_accuracy"]["ci_low"] > 0
        ),
        "introduced_no_more_than_half_corrected": bool(
            primary["introduced"] <= 0.5 * max(primary["corrected"], 1)
        ),
    }
    decision = {
        "status": "noise_v3_a4b_positive_evidence_complete",
        "formal": formal,
        "integrity": {
            "source_queries": int(len(scan)),
            "aligned_queries": int(frame["query_index"].nunique()),
            "skipped_no_same_identity_support": int(skipped_no_support),
            "skipped_missing_embedding": int(skipped_missing_embedding),
            "baseline_similarity_mismatch": int(baseline_mismatch),
            "P3_disjoint_graph": True,
        },
        "support_spectra_per_query": {
            "median": float(frame.groupby("query_index")["support_spectra"].first().median()),
            "p10": float(frame.groupby("query_index")["support_spectra"].first().quantile(0.1)),
            "p90": float(frame.groupby("query_index")["support_spectra"].first().quantile(0.9)),
        },
        "dose_results": summaries,
        "primary_alpha": args.primary_alpha,
        "primary_result": primary,
        "gates": gates,
        "pass_to_two_expert_teacher": bool(all(gates.values())),
        "interpretation": (
            "A passing result establishes training-time geometric headroom for distilling real "
            "same-identity evidence into a clean-query adapter. It is not a deployable method: "
            "the true-identity prototype is used only to construct a training target."
        ),
        "claim_limit": (
            "No DreaMS weights are updated here. Correct-identity prototype mixing must not be "
            "reported as retrieval performance or used during sealed evaluation."
        ),
        "parameters": {
            "alphas": args.alphas, "primary_alpha": args.primary_alpha,
            "maximum_support_spectra": args.maximum_support_spectra,
            "random_repeats": args.random_repeats, "risk_penalty": args.risk_penalty,
            "minimum_new_corrections": args.minimum_new_corrections, "seed": args.seed,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "a4_scan_sha256": sha256_file(args.a4_dir / "exact_peak_scan.h5"),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }

    staging = Path(tempfile.mkdtemp(prefix="a4b_positive_", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "paired_results.csv.gz", index=False, compression="gzip")
        (staging / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

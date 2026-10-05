#!/usr/bin/env python
"""B47-U0 truth-blind audit of reference multiplicity and adduct pooling.

This program deliberately does not open annotation truth, phenotype, Rhea, or
any BioAware/P2b output.  It scores the already frozen candidate-reference
edges with the already frozen official DreaMS embeddings, then asks how much
the candidate ordering changes when only the reference aggregation changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402
from bioaware_b47_u0_core import (  # noqa: E402
    ALLOWED_ADDUCTS,
    aggregate_candidate_scores,
    candidate_exposure_summary,
    quantiles,
    spearman_without_scipy,
    top_summary,
)


FORBIDDEN_COLUMN_TOKENS = (
    "truth", "label", "correct", "annotation", "phenotype", "disease", "case_control",
)
EXPECTED_INPUT_SHA256 = {
    "graph_report": "a47ebd3e3cff35ad8a135e72fbf14a7b72013c21d7248f6b84c7d825141bce5d",
    "queries": "c9551bc5d32fcec93405bd80cbe62880e3ac309c3df18e120123fb5fbf4133be",
    "candidate_references": "64f5940669876db01d17a4b6cd36977d043a930ac471e53d7af3c5018bf67c56",
    "query_embeddings": "afbd3a029433ab6b0f981d4d8cc41f6c1592de3ebecbd98c60496499db15f798",
    "reference_embeddings": "634e6db40998768bacee337d48a57583f46665f1f5e650606959a541405d01b2",
    "reference_rows": "32b5909b8a69f4e218edabe3aa70a027c3627e50a2d2c700271a09c80a199c9c",
    "query_index": "8ba4e698e0bc4f072baa4ee84fefc7757dc7b84a38fdb005e87affa8e180a745",
    "official_checkpoint": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    "architecture_checkpoint": "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
}


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def load_and_verify(graph: Path, embeddings: Path) -> tuple[dict, dict]:
    graph_report_path = graph / "report.json"
    embedding_report_path = embeddings / "report.json"
    for path in (
        graph_report_path, graph / "queries.csv.gz", graph / "candidate_references.csv.gz",
        embedding_report_path, embeddings / "query_embeddings.npy",
        embeddings / "reference_embeddings.npy", embeddings / "reference_rows.npy",
        embeddings / "query_index.csv",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
    if graph_report.get("status") != "bioaware_b47_truthblind_candidate_graph_frozen":
        raise RuntimeError("unexpected B47 graph status")
    if embedding_report.get("status") != "bioaware_b47_truthblind_embeddings_complete":
        raise RuntimeError("unexpected B47 embedding status")
    if sha256_file(graph_report_path) != EXPECTED_INPUT_SHA256["graph_report"]:
        raise RuntimeError("B47 frozen graph-report SHA256 changed")
    if not graph_report.get("pass_to_query_embedding_and_seed_construction"):
        raise RuntimeError("B47 graph gate failed")
    if not embedding_report.get("pass_to_truthblind_seed_construction"):
        raise RuntimeError("B47 embedding gate failed")
    expected_graph_contract = {
        "P2b_used": False,
        "algorithm_outputs_opened": False,
        "phenotype_used": False,
        "truth_opened": False,
    }
    if graph_report.get("contracts") != expected_graph_contract:
        raise RuntimeError("B47 graph truth-blind contract changed")
    embedding_contract = embedding_report.get("contracts", {})
    expected_false = (
        "truth_opened", "phenotype_used", "algorithm_outputs_opened",
        "candidate_scores_computed", "seed_selection_performed", "model_fitted", "P2b_used",
    )
    if embedding_contract.get("one_shared_official_encoder") is not True or any(
        embedding_contract.get(key) is not False for key in expected_false
    ):
        raise RuntimeError("B47 embedding truth-blind contract changed")
    if embedding_report.get("provenance", {}).get("graph_report_sha256") != sha256_file(
        graph_report_path
    ):
        raise RuntimeError("B47 graph/embedding provenance mismatch")
    for key, filename in (
        ("queries_sha256", "queries.csv.gz"),
        ("candidate_references_sha256", "candidate_references.csv.gz"),
    ):
        observed = sha256_file(graph / filename)
        if graph_report.get("provenance", {}).get(key) != observed:
            raise RuntimeError(f"B47 graph file hash mismatch: {filename}")
        if embedding_report.get("provenance", {}).get(key) != observed:
            raise RuntimeError(f"B47 embedding input hash mismatch: {filename}")
        if observed != EXPECTED_INPUT_SHA256[key.removesuffix("_sha256")]:
            raise RuntimeError(f"B47 preregistered input SHA256 changed: {filename}")
    for key, filename in (
        ("query_embeddings_sha256", "query_embeddings.npy"),
        ("reference_embeddings_sha256", "reference_embeddings.npy"),
        ("reference_rows_sha256", "reference_rows.npy"),
        ("query_index_sha256", "query_index.csv"),
    ):
        observed = sha256_file(embeddings / filename)
        if embedding_report.get("provenance", {}).get(key) != observed:
            raise RuntimeError(f"B47 embedding output hash mismatch: {filename}")
        if observed != EXPECTED_INPUT_SHA256[key.removesuffix("_sha256")]:
            raise RuntimeError(f"B47 preregistered embedding SHA256 changed: {filename}")
    for key in ("official_checkpoint", "architecture_checkpoint"):
        if embedding_report.get("provenance", {}).get(f"{key}_sha256") != EXPECTED_INPUT_SHA256[key]:
            raise RuntimeError(f"B47 encoder provenance changed: {key}")
    return graph_report, embedding_report


def reject_forbidden_columns(frame: pd.DataFrame, label: str) -> None:
    forbidden = sorted(
        column for column in frame.columns
        if any(token in column.casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise RuntimeError(f"{label} contains forbidden outcome/context columns: {forbidden}")


def score_reference_pairs(
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if query_embedding.dtype != np.float32 or reference_embedding.dtype != np.float32:
        raise RuntimeError("B47 embeddings must be float32")
    output = np.empty(len(query_positions), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(output), batch_size):
            end = min(len(output), start + batch_size)
            # Explicit writable copies avoid undefined behaviour when source arrays
            # are read-only memory maps.  Only the indexed batch is transferred.
            q_batch = np.array(
                query_embedding[query_positions[start:end]], dtype=np.float32,
                copy=True, order="C",
            )
            r_batch = np.array(
                reference_embedding[reference_positions[start:end]], dtype=np.float32,
                copy=True, order="C",
            )
            q = torch.from_numpy(q_batch).to(device=device, non_blocking=True)
            r = torch.from_numpy(r_batch).to(device=device, non_blocking=True)
            output[start:end] = (q * r).sum(dim=1).float().cpu().numpy()
            if end % (batch_size * 4) == 0 or end == len(output):
                print(f"[B47-U0 score] {end:,}/{len(output):,}", flush=True)
    if not np.isfinite(output).all() or output.min() < -1.0001 or output.max() > 1.0001:
        raise RuntimeError("invalid B47 cosine scores")
    return output


def comparison(left: pd.DataFrame, left_prefix: str, right_prefix: str) -> dict[str, object]:
    left_unique = left[f"{left_prefix}_unique_top1"].astype(bool)
    right_unique = left[f"{right_prefix}_unique_top1"].astype(bool)
    common = left_unique & right_unique
    changed = (
        left[f"{left_prefix}_top_candidate_id"].astype(str)
        != left[f"{right_prefix}_top_candidate_id"].astype(str)
    )
    adduct_changed = (
        left[f"{left_prefix}_top_adduct"].astype(str)
        != left[f"{right_prefix}_top_adduct"].astype(str)
    )
    return {
        "queries": int(len(left)),
        "both_unique_top1": int(common.sum()),
        "identity_flips_among_both_unique": int((common & changed).sum()),
        "identity_flip_fraction_among_both_unique": float(
            (common & changed).sum() / common.sum()
        ) if common.any() else None,
        "adduct_flips_among_both_unique": int((common & adduct_changed).sum()),
        "adduct_flip_fraction_among_both_unique": float(
            (common & adduct_changed).sum() / common.sum()
        ) if common.any() else None,
    }


def reference_count_bins(candidates: pd.DataFrame) -> list[dict[str, object]]:
    bins = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, None))
    rows = []
    for low, high in bins:
        mask = candidates["reference_count"].ge(low)
        label = f"{low}+" if high is None else (str(low) if low == high else f"{low}-{high}")
        if high is not None:
            mask &= candidates["reference_count"].le(high)
        group = candidates.loc[mask]
        rows.append({
            "bin": label,
            "candidates": int(len(group)),
            "mean_max_minus_mean": float(group["max_minus_mean"].mean()) if len(group) else None,
            "median_max_minus_mean": float(group["max_minus_mean"].median()) if len(group) else None,
            "p90_max_minus_mean": float(group["max_minus_mean"].quantile(0.9)) if len(group) else None,
        })
    return rows


def branch_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (query_id, adduct), group in candidates.groupby(
        ["query_id", "reference_adduct"], sort=False
    ):
        ordered = group.sort_values(
            ["max_score", "candidate_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        top = float(ordered.loc[0, "max_score"])
        tied = np.isclose(ordered["max_score"].to_numpy(float), top, atol=1e-12, rtol=0)
        rows.append({
            "query_id": str(query_id), "branch_adduct": str(adduct),
            "branch_candidate_count": int(len(ordered)),
            "branch_top_score": top,
            "branch_top_candidate_id": str(ordered.loc[0, "candidate_id"]) if tied.sum() == 1 else "",
            "branch_unique_top1": bool(tied.sum() == 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.score_batch_size <= 0:
        raise ValueError("score batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph, embeddings = args.graph_dir.resolve(), args.embedding_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    graph_report, embedding_report = load_and_verify(graph, embeddings)

    queries = pd.read_csv(graph / "queries.csv.gz", dtype={"query_id": str})
    candidate = pd.read_csv(
        graph / "candidate_references.csv.gz",
        dtype={"query_id": str, "candidate_id": str, "reference_row": np.int64,
               "reference_adduct": str, "candidate_formula": str},
    )
    reject_forbidden_columns(queries, "query manifest")
    reject_forbidden_columns(candidate, "candidate manifest")
    if len(queries) != int(graph_report["queries"]):
        raise RuntimeError("B47 query count mismatch")
    if len(candidate) != int(graph_report["candidate_reference_rows"]):
        raise RuntimeError("B47 candidate-reference count mismatch")
    if not candidate["reference_adduct"].isin(ALLOWED_ADDUCTS).all():
        raise RuntimeError("unexpected B47 reference adduct")
    for column in ("query_id", "candidate_id", "candidate_formula", "reference_adduct"):
        text = candidate[column].astype(str).str.strip()
        if text.eq("").any() or text.str.casefold().isin({"nan", "none"}).any():
            raise RuntimeError(f"B47 candidate manifest has missing {column}")
    if candidate.duplicated(["query_id", "candidate_id", "reference_row"]).any():
        raise RuntimeError("B47 candidate graph contains duplicate candidate-reference edges")

    query_index = pd.read_csv(embeddings / "query_index.csv", dtype={"query_id": str})
    if queries["query_id"].astype(str).tolist() != query_index["query_id"].tolist():
        raise RuntimeError("B47 query/index order mismatch")
    positions = query_index["embedding_position"].to_numpy(np.int64)
    if not np.array_equal(positions, np.arange(len(queries))):
        raise RuntimeError("B47 query positions are not canonical")
    query_position = pd.Series(positions, index=query_index["query_id"])

    reference_rows = np.load(embeddings / "reference_rows.npy", mmap_mode="r", allow_pickle=False)
    query_embedding = np.load(
        embeddings / "query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    reference_embedding = np.load(
        embeddings / "reference_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    if query_embedding.shape != (
        int(embedding_report["queries"]["rows"]), int(embedding_report["queries"]["dimension"])
    ):
        raise RuntimeError("query embedding shape mismatch")
    if reference_embedding.shape != (
        int(embedding_report["references"]["rows"]),
        int(embedding_report["references"]["dimension"]),
    ):
        raise RuntimeError("reference embedding shape mismatch")
    qpos = candidate["query_id"].map(query_position)
    if qpos.isna().any():
        raise RuntimeError("candidate has unknown query ID")
    candidate_rows = candidate["reference_row"].to_numpy(np.int64)
    rpos = np.searchsorted(reference_rows, candidate_rows)
    if np.any(rpos >= len(reference_rows)) or not np.array_equal(
        np.asarray(reference_rows[rpos]), candidate_rows
    ):
        raise RuntimeError("candidate reference row is absent from embedding index")
    if args.preflight_only:
        print(json.dumps({
            "status": "bioaware_b47_u0_preflight_passed",
            "queries": int(len(queries)),
            "candidate_reference_rows": int(len(candidate)),
            "unique_reference_rows": int(len(reference_rows)),
            "truth_opened": False,
            "output_created": False,
        }, indent=2, sort_keys=True), flush=True)
        return
    candidate["spectral_score"] = score_reference_pairs(
        query_embedding, reference_embedding, qpos.to_numpy(np.int64), rpos.astype(np.int64),
        torch.device(args.device), args.score_batch_size,
    )

    aggregates = aggregate_candidate_scores(candidate, subset_sizes=(2, 4, 8))
    expected_candidate_counts = candidate.groupby(["query_id", "candidate_id"], sort=False).ngroups
    if len(aggregates) != expected_candidate_counts:
        raise RuntimeError("candidate aggregation count mismatch")
    max_top = top_summary(aggregates, "max_score", "max")
    mean_top = top_summary(aggregates, "mean_score", "mean")
    count_candidates = aggregates.copy()
    count_candidates["count_score"] = count_candidates["reference_count"].astype(float)
    count_top = top_summary(count_candidates, "count_score", "count")
    per_query = (
        queries[[
            "query_id", "study", "sample", "feature_id", "candidate_identities",
            "candidate_reference_spectra",
        ]]
        .merge(max_top, on="query_id", how="left", validate="one_to_one")
        .merge(mean_top, on="query_id", how="left", validate="one_to_one")
        .merge(count_top, on="query_id", how="left", validate="one_to_one")
    )
    if per_query.filter(regex="_top_score$").isna().any().any():
        raise RuntimeError("one or more queries lost a top-score summary")
    observed_candidate_counts = aggregates.groupby("query_id")["candidate_id"].nunique()
    observed_reference_counts = candidate.groupby("query_id").size()
    if not np.array_equal(
        per_query["query_id"].map(observed_candidate_counts).to_numpy(np.int64),
        per_query["candidate_identities"].to_numpy(np.int64),
    ):
        raise RuntimeError("B47 per-query candidate counts do not replay")
    if not np.array_equal(
        per_query["query_id"].map(observed_reference_counts).to_numpy(np.int64),
        per_query["candidate_reference_spectra"].to_numpy(np.int64),
    ):
        raise RuntimeError("B47 per-query candidate-reference counts do not replay")
    query_candidate_median = aggregates.groupby("query_id")["reference_count"].median()
    per_query["candidate_reference_count_median"] = per_query["query_id"].map(
        query_candidate_median
    )
    per_query["max_winner_reference_count_ratio_to_query_median"] = (
        per_query["max_top_reference_count"] / per_query["candidate_reference_count_median"]
    )
    per_query["max_vs_mean_identity_changed"] = (
        per_query["max_top_candidate_id"].astype(str)
        != per_query["mean_top_candidate_id"].astype(str)
    )
    per_query["max_vs_mean_adduct_changed"] = (
        per_query["max_top_adduct"].astype(str) != per_query["mean_top_adduct"].astype(str)
    )

    branches = branch_summary(aggregates)
    branch_counts = branches.groupby("query_id")["branch_adduct"].nunique()
    both_ids = set(branch_counts[branch_counts.eq(2)].index.astype(str))
    both = branches[branches["query_id"].isin(both_ids)].pivot(
        index="query_id", columns="branch_adduct", values="branch_top_score"
    )
    both = both.dropna(subset=list(ALLOWED_ADDUCTS))
    cross_branch_abs_gap = np.abs(
        both[ALLOWED_ADDUCTS[0]].to_numpy(float) - both[ALLOWED_ADDUCTS[1]].to_numpy(float)
    )
    per_query["has_both_adduct_branches"] = per_query["query_id"].isin(both_ids)

    max_mean = comparison(per_query, "max", "mean")
    max_count = comparison(per_query, "max", "count")
    both_unique = per_query["max_unique_top1"].astype(bool) & per_query["mean_unique_top1"].astype(bool)
    high_count = aggregates.loc[aggregates["reference_count"].ge(9), "max_minus_mean"].to_numpy(float)
    lift_median_high_count = float(np.median(high_count)) if len(high_count) else 0.0
    multiplicity_material = bool(
        (max_mean["identity_flip_fraction_among_both_unique"] or 0.0) >= 0.01
        or lift_median_high_count >= 0.02
    )
    adduct_exposure_fraction = float(len(both) / len(queries))
    close_cross_branch_fraction = float(np.mean(cross_branch_abs_gap <= 0.05)) if len(both) else 0.0
    adduct_competition_material = bool(
        adduct_exposure_fraction >= 0.05 and close_cross_branch_fraction >= 0.10
    )

    aggregates["study"] = aggregates["query_id"].map(
        queries.set_index("query_id")["study"]
    )
    if aggregates["study"].isna().any():
        raise RuntimeError("candidate aggregate lost its source-study label")

    exposure = candidate_exposure_summary(aggregates)

    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "candidate_aggregates": output / "candidate_aggregates.csv.gz",
        "per_query": output / "per_query.csv.gz",
    }
    write_csv(aggregates, paths["candidate_aggregates"])
    write_csv(per_query, paths["per_query"])

    subset_curve = {}
    for k in (1, 2, 4, 8):
        column = "mean_score" if k == 1 else f"expected_max_{k}"
        eligible = aggregates[np.isfinite(aggregates[column].to_numpy(float))]
        subset_curve[str(k)] = {
            "eligible_candidates": int(len(eligible)),
            "eligible_queries": int(eligible["query_id"].nunique()),
            "mean_expected_max": float(eligible[column].mean()),
            "median_expected_max": float(eligible[column].median()),
            "mean_lift_over_expected_single": float(
                (eligible[column] - eligible["mean_score"]).mean()
            ),
        }
    common_support = aggregates[aggregates["reference_count"].ge(8)].copy()
    common_support_curve = {}
    for k in (1, 2, 4, 8):
        column = "mean_score" if k == 1 else f"expected_max_{k}"
        common_support_curve[str(k)] = {
            "candidates": int(len(common_support)),
            "queries": int(common_support["query_id"].nunique()),
            "mean_expected_max": float(common_support[column].mean()),
            "median_expected_max": float(common_support[column].median()),
            "mean_lift_over_expected_single": float(
                (common_support[column] - common_support["mean_score"]).mean()
            ),
        }
    source_sensitivity = {}
    for source, source_query in per_query.groupby("study", sort=True):
        source_both = source_query["has_both_adduct_branches"].astype(bool)
        source_sensitivity[str(source)] = {
            "queries": int(len(source_query)),
            "max_vs_expected_single_top1": comparison(source_query, "max", "mean"),
            "queries_with_both_adduct_branches": int(source_both.sum()),
            "fraction_queries_with_both_adduct_branches": float(source_both.mean()),
            "max_winner_count_ratio_to_query_median": quantiles(
                source_query["max_winner_reference_count_ratio_to_query_median"].to_numpy(float)
            ),
        }
    top_adduct_counts = Counter(
        per_query.loc[per_query["max_unique_top1"].astype(bool), "max_top_adduct"].astype(str)
    )
    engineering_gates = {
        "all_queries_scored": len(per_query) == int(graph_report["queries"]),
        "all_candidate_reference_rows_scored": len(candidate) == int(
            graph_report["candidate_reference_rows"]
        ),
        "every_query_has_at_least_two_candidates": bool(
            aggregates.groupby("query_id")["candidate_id"].nunique().ge(2).all()
        ),
        "per_query_manifest_counts_replayed_exactly": True,
        "candidate_reference_edges_unique": bool(
            not candidate.duplicated(["query_id", "candidate_id", "reference_row"]).any()
        ),
        "one_adduct_branch_per_candidate": bool(
            aggregates.groupby(["query_id", "candidate_id"])["reference_adduct"].nunique().eq(1).all()
        ),
        "all_scores_finite_and_bounded": bool(
            np.isfinite(candidate["spectral_score"]).all()
            and candidate["spectral_score"].between(-1.0001, 1.0001).all()
        ),
        "truth_columns_absent": True,
        "phenotype_columns_absent": True,
        "algorithm_outputs_absent": True,
        "P2b_absent": True,
    }
    report = {
        "status": "bioaware_b47_u0_reference_multiplicity_adduct_audit_complete",
        "formal": True,
        "protocol_version": "B47-U0-v2-candidate-exposure",
        "queries": int(len(per_query)),
        "candidate_query_pairs": int(len(aggregates)),
        "unique_candidate_identities_global": int(aggregates["candidate_id"].nunique()),
        "candidate_reference_rows": int(len(candidate)),
        "candidate_exposure": exposure,
        "reference_multiplicity": {
            "per_candidate": quantiles(aggregates["reference_count"].to_numpy(float)),
            "max_minus_expected_single": quantiles(aggregates["max_minus_mean"].to_numpy(float)),
            "max_minus_expected_single_by_reference_count": reference_count_bins(aggregates),
            "expected_max_without_replacement_curve": subset_curve,
            "same_candidates_n_ge_8_expected_max_curve": common_support_curve,
            "spearman_log_count_vs_max_lift": spearman_without_scipy(
                np.log1p(aggregates["reference_count"]), aggregates["max_minus_mean"]
            ),
            "max_vs_expected_single_top1": max_mean,
            "max_vs_reference_count_only_top1": max_count,
            "max_winner_count_ratio_to_query_median": quantiles(
                per_query["max_winner_reference_count_ratio_to_query_median"].to_numpy(float)
            ),
            "material_by_preregistered_sensitivity_threshold": multiplicity_material,
        },
        "source_replication": source_sensitivity,
        "adduct_pooling": {
            "candidate_identities_by_adduct": {
                key: int(value) for key, value in sorted(Counter(aggregates["reference_adduct"]).items())
            },
            "queries_with_both_adduct_branches": int(len(both)),
            "fraction_queries_with_both_adduct_branches": adduct_exposure_fraction,
            "pooled_unique_top1_by_adduct": {
                key: int(value) for key, value in sorted(top_adduct_counts.items())
            },
            "absolute_best_branch_score_gap": quantiles(cross_branch_abs_gap),
            "fraction_both_branch_queries_with_gap_le_0_05": close_cross_branch_fraction,
            "max_vs_expected_single_adduct_flips_among_both_unique": int(
                (both_unique & per_query["max_vs_mean_adduct_changed"]).sum()
            ),
            "material_by_preregistered_exposure_threshold": adduct_competition_material,
            "interpretation": (
                "The two adduct branches represent different neutral-mass hypotheses. "
                "U0 measures competition exposure and score sensitivity, not which branch is correct."
            ),
        },
        "engineering_gates": {key: bool(value) for key, value in engineering_gates.items()},
        "pass_to_u1_strong_unary_bakeoff": all(bool(value) for value in engineering_gates.values()),
        "contracts": {
            "truth_opened": False,
            "phenotype_used": False,
            "reaction_network_used": False,
            "algorithm_outputs_opened": False,
            "model_fitted": False,
            "aggregation_selected": False,
            "queries_or_candidates_removed_after_scoring": False,
            "P2b_used": False,
        },
        "parameters": {
            "score_batch_size": int(args.score_batch_size),
            "device": str(args.device),
            "reference_subset_sizes": [1, 2, 4, 8],
            "top_ties": "adverse; tied candidates have no unique Top-1",
            "multiplicity_materiality": (
                "max-vs-expected-single identity flip >=1% among both-unique queries OR "
                "median max lift >=0.02 among candidates with >=9 references"
            ),
            "adduct_materiality": (
                ">=5% queries expose both branches AND >=10% of exposed queries have "
                "absolute best-branch score gap <=0.05"
            ),
        },
        "provenance": {
            "graph_report_sha256": sha256_file(graph / "report.json"),
            "embedding_report_sha256": sha256_file(embeddings / "report.json"),
            "queries_sha256": sha256_file(graph / "queries.csv.gz"),
            "candidate_references_sha256": sha256_file(graph / "candidate_references.csv.gz"),
            "query_embeddings_sha256": sha256_file(embeddings / "query_embeddings.npy"),
            "reference_embeddings_sha256": sha256_file(embeddings / "reference_embeddings.npy"),
            "reference_rows_sha256": sha256_file(embeddings / "reference_rows.npy"),
            "candidate_aggregates_sha256": sha256_file(paths["candidate_aggregates"]),
            "per_query_sha256": sha256_file(paths["per_query"]),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Truth-blind nuisance audit only. Rank flips do not identify the better aggregator, "
            "and no annotation accuracy, BioAware gain, reaction signal, or SOTA claim is tested."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()

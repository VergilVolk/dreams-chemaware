#!/usr/bin/env python
"""Evaluate aligned embedding caches on the sealed GNPS 10-ppm benchmark.

The evaluator is model-agnostic.  Each cache is either an ``.npy`` matrix in
manifest-row order or an ``.npz`` containing unique ``rows`` and unit-normalised
``embeddings``.  Candidate and baseline are always scored on the identical
frozen graph.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from noise_corrected_fullgraph_evaluation import (
    full_metrics,
    paired_outcome_table,
    score_embeddings,
)


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--baseline-embeddings", type=Path, required=True)
    parser.add_argument("--candidate-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    return parser.parse_args()


def load_embeddings(path: Path, manifest_rows: int) -> tuple[np.ndarray, np.ndarray]:
    # Dispatch on the serialized object, not the filename suffix.  ``np.savez``
    # can legally write an NPZ archive to a path ending in ``.npy``; one early
    # GLM run did exactly that and suffix dispatch then interpreted the archive
    # member names ("rows", "embeddings") as a numeric matrix.
    body = np.load(path, mmap_mode="r", allow_pickle=False)
    if isinstance(body, np.lib.npyio.NpzFile):
        try:
            if set(body.files) != {"rows", "embeddings"}:
                raise RuntimeError(
                    f"embedding archive has unexpected members {body.files}: {path}"
                )
            embeddings = np.asarray(body["embeddings"], dtype=np.float32)
            rows = np.asarray(body["rows"], dtype=np.int64)
        finally:
            body.close()
    else:
        embeddings = np.asarray(body, dtype=np.float32)
        rows = np.arange(len(embeddings), dtype=np.int64)
    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise RuntimeError(f"malformed embedding cache: {path}")
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= manifest_rows):
        raise RuntimeError(f"embedding rows are duplicated or out of bounds: {path}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms < 1e-8):
        raise RuntimeError(f"embedding cache is non-finite or zero: {path}")
    embeddings = embeddings / norms[:, None]
    return rows, embeddings


def graph_from_panel(path: Path):
    with np.load(path, allow_pickle=False) as body:
        panel = {key: np.asarray(body[key]) for key in body.files}
    return SimpleNamespace(
        n_queries=len(panel["query_row"]),
        query_row=panel["query_row"],
        query_ik14=panel["query_ik14"].astype(str),
        query_formula=panel["query_formula"].astype(str),
        query_has_near=panel["near_query"].astype(bool),
        query_ptr=panel["query_ptr"],
        molecule_ptr=panel["molecule_ptr"],
        molecule_label=panel["molecule_label"].astype(np.int8),
        molecule_ik14=panel["molecule_ik14"].astype(str),
        molecule_formula=panel["molecule_formula"].astype(str),
        pair_candidate_row=panel["candidate_row"],
    )


def rename_pairwise_metric(metrics: dict, panel_name: str) -> dict:
    metrics = dict(metrics)
    value = metrics.pop("massspecgym_10ppm_pooled_pairwise")
    value["dataset"] = f"GNPS Gold/Silver {panel_name.replace('_', '-')}"
    value["edge_semantics"] = "directed query-reference spectrum edges; [M+H]+; strict 10 ppm"
    metrics["gnps_10ppm_pooled_pairwise"] = value
    mh = metrics.pop("massspecgym_mh_10ppm_pooled_pairwise", None)
    if mh is not None:
        mh["dataset"] = f"GNPS Gold/Silver {panel_name.replace('_', '-')}"
        metrics["gnps_mh_10ppm_pooled_pairwise"] = mh
    return metrics


def cluster_ci(
    formulas: np.ndarray,
    values: np.ndarray,
    repeats: int,
    seed: int,
    hypotheses: int,
) -> dict:
    table = pd.DataFrame({"formula": formulas.astype(str), "value": values.astype(float)})
    grouped = table.groupby("formula", sort=True).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), size=len(grouped))
        bootstrap[index] = sums[take].sum() / counts[take].sum()
    tail = 0.05 / (2 * hypotheses)
    return {
        "delta": float(np.mean(values)),
        "delta_pp": float(100 * np.mean(values)),
        "ci_low": float(np.quantile(bootstrap, tail)),
        "ci_high": float(np.quantile(bootstrap, 1 - tail)),
        "ci_low_pp": float(100 * np.quantile(bootstrap, tail)),
        "ci_high_pp": float(100 * np.quantile(bootstrap, 1 - tail)),
        "formula_clusters": int(len(grouped)),
        "bootstrap_resamples": repeats,
        "familywise_hypotheses": hypotheses,
    }


def paired_summary(
    baseline_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    repeats: int,
    seed: int,
    hypotheses: int,
) -> tuple[dict, pd.DataFrame]:
    outcome = paired_outcome_table(baseline_table, candidate_table)
    near = outcome.near.to_numpy(bool)
    baseline_rank = outcome.official_rank.to_numpy(int)
    candidate_rank = outcome.candidate_rank.to_numpy(int)
    deltas = {
        "recall@1": (candidate_rank == 1).astype(float) - (baseline_rank == 1).astype(float),
        "mrr": 1.0 / candidate_rank - 1.0 / baseline_rank,
        "macro_query_auroc": (
            outcome.candidate_macro_query_auc.to_numpy(float)
            - outcome.official_macro_query_auc.to_numpy(float)
        ),
        "macro_query_auprc": (
            outcome.candidate_macro_query_auprc.to_numpy(float)
            - outcome.official_macro_query_auprc.to_numpy(float)
        ),
        "positive_vs_best_negative_margin": (
            outcome.candidate_positive_vs_best_negative_margin.to_numpy(float)
            - outcome.official_positive_vs_best_negative_margin.to_numpy(float)
        ),
        "signed_top1_top2_gap": (
            outcome.candidate_signed_top1_top2_gap.to_numpy(float)
            - outcome.official_signed_top1_top2_gap.to_numpy(float)
        ),
    }
    formulas = outcome.query_formula.to_numpy(str)
    ci = {
        name: cluster_ci(formulas, values, repeats, seed + index, hypotheses)
        for index, (name, values) in enumerate(deltas.items())
    }
    near_ci = {
        name: cluster_ci(formulas[near], values[near], repeats, seed + 100 + index, hypotheses)
        for index, (name, values) in enumerate(deltas.items())
    }
    summary = {
        "queries": len(outcome),
        "corrected": int(outcome.corrected.sum()),
        "introduced": int(outcome.introduced.sum()),
        "risk_net_lambda2": int(outcome.corrected.sum() - 2 * outcome.introduced.sum()),
        "near_queries": int(near.sum()),
        "near_corrected": int(outcome.loc[near, "corrected"].sum()),
        "near_introduced": int(outcome.loc[near, "introduced"].sum()),
        "near_risk_net_lambda2": int(
            outcome.loc[near, "corrected"].sum() - 2 * outcome.loc[near, "introduced"].sum()
        ),
        "formula_cluster_paired_ci": ci,
        "near_formula_cluster_paired_ci": near_ci,
    }
    return summary, outcome


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    baseline_rows, baseline_embeddings = load_embeddings(args.baseline_embeddings, len(manifest))
    candidate_rows, candidate_embeddings = load_embeddings(args.candidate_embeddings, len(manifest))
    if not np.array_equal(baseline_rows, candidate_rows):
        raise RuntimeError("baseline and candidate embedding row registries differ")

    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    report = {
        "status": "gnps_gold_silver_10ppm_embedding_evaluation_complete",
        "exact_nist20_replication": False,
        "panels": {},
    }
    try:
        for panel_index, name in enumerate(("identity_disjoint", "formula_disjoint")):
            graph = graph_from_panel(args.benchmark / f"panel_{name}.npz")
            baseline_scores = score_embeddings(graph, baseline_rows, baseline_embeddings)
            candidate_scores = score_embeddings(graph, candidate_rows, candidate_embeddings)
            adduct = np.full(graph.n_queries, "[M+H]+", dtype="U6")
            baseline_metrics, baseline_table = full_metrics(graph, baseline_scores, adduct)
            candidate_metrics, candidate_table = full_metrics(graph, candidate_scores, adduct)
            paired, outcome = paired_summary(
                baseline_table, candidate_table,
                args.bootstrap_resamples, args.bootstrap_seed + panel_index * 1000,
                hypotheses=24,
            )
            report["panels"][name] = {
                "baseline": rename_pairwise_metric(baseline_metrics, name),
                "candidate": rename_pairwise_metric(candidate_metrics, name),
                "paired": paired,
            }
            outcome.to_csv(staging / f"paired_queries_{name}.csv.gz", index=False, compression="gzip")
        report["embedding_rows"] = int(len(baseline_rows))
        report["embedding_dimension"] = int(baseline_embeddings.shape[1])
        report["claim_limit"] = (
            "GNPS Gold/Silver identity/formula-disjoint transfer benchmark; NIST20-like "
            "10-ppm pair mathematics, not an exact NIST20 paper replication."
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

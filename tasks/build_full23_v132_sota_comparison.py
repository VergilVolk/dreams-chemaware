#!/usr/bin/env python3
"""Build the same-axis Full23 v1.3.2 manuscript comparison.

This script appends Full23 v1.3.2 to the frozen 15-method GNPS ladder.  It
recomputes every Full23 ranking metric directly from its molecule-candidate
ledger and keeps spectrum-pair metrics unavailable: Full23 emits one score per
candidate molecule after reference-spectrum aggregation, so a spectrum-pair
AUROC would be a different endpoint.

GNPS is consumed development data.  The output is a development comparison,
not an unopened external confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PANELS = ("identity_disjoint", "formula_disjoint")
KS = (1, 2, 3, 5, 10, 20)
METHOD = "full23_v132"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run15-ladder",
        type=Path,
        default=Path("deliverables/GLM_gnps_article_ladder/run15/ladder_full.csv"),
    )
    parser.add_argument(
        "--run15-evaluation",
        type=Path,
        default=Path(
            "data/validation/GLM_gnps_article_benchmark_s2v26/run15/evaluation"
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "data/validation/full23_v131_direct_repairs_run_2434102/"
            "gnps_benchmark_view"
        ),
    )
    parser.add_argument(
        "--v132",
        type=Path,
        default=Path("data/validation/full23_v132_residual_scale_20261102"),
    )
    parser.add_argument(
        "--b-report",
        type=Path,
        default=Path("data/validation/wse_unified_B_candidate_run_2388498/report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deliverables/GLM_gnps_article_ladder/run16_full23_v132"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261102)
    return parser.parse_args()


def strict_rank(values: np.ndarray, positive_index: int) -> int:
    """Rank with ties conservatively placed ahead of the positive."""
    positive = float(values[positive_index])
    negative = np.delete(values, positive_index)
    return 1 + int(np.sum(negative >= positive))


def query_table(
    scores: np.ndarray,
    labels: np.ndarray,
    query_ptr: np.ndarray,
    raw: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        values = np.asarray(scores[left:right], dtype=np.float64)
        truth = np.flatnonzero(labels[left:right])
        if len(truth) != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"query {query} violates the closed-set contract")
        positive_index = int(truth[0])
        positive = float(values[positive_index])
        negative = np.delete(values, positive_index)
        rank = strict_rank(values, positive_index)
        order = np.argsort(-values, kind="stable")
        gap = float(values[order[0]] - values[order[1]])
        records.append(
            {
                "query_index": query,
                "query_row": int(raw["query_row"][query]),
                "query_ik14": str(raw["query_ik14"][query]),
                "query_formula": str(raw["query_formula"][query]),
                "near": bool(raw["near_query"][query]),
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "macro_query_auc": (
                    float(np.sum(positive > negative))
                    + 0.5 * float(np.sum(positive == negative))
                )
                / len(negative),
                "macro_query_auprc": 1.0 / rank,
                "positive_vs_best_negative_margin": positive - float(np.max(negative)),
                "top1_top2_gap": gap,
                "signed_top1_top2_gap": gap if rank == 1 else -gap,
            }
        )
    return pd.DataFrame.from_records(records)


def ladder_row(table: pd.DataFrame, scores: np.ndarray, labels: np.ndarray) -> dict:
    rank = table["rank"].to_numpy(np.int64)
    near = table["near"].to_numpy(bool)
    row: dict[str, object] = {
        "panel": "",
        "method": METHOD,
        "queries": int(len(table)),
        "mrr": float(np.mean(1.0 / rank)),
        "mean_rank": float(np.mean(rank)),
        "median_rank": float(np.median(rank)),
        "macro_query_auroc": float(table["macro_query_auc"].mean()),
        "macro_query_auprc_recomputed": float(table["macro_query_auprc"].mean()),
        "mean_positive_vs_best_negative_margin": float(
            table["positive_vs_best_negative_margin"].mean()
        ),
        "mean_top1_top2_gap": float(table["top1_top2_gap"].mean()),
        "mean_signed_top1_top2_gap": float(
            table["signed_top1_top2_gap"].mean()
        ),
        "near_queries": int(np.sum(near)),
        "near_mrr": float(np.mean(1.0 / rank[near])),
        "micro_auroc": float(roc_auc_score(labels, scores)),
        "micro_auprc": float(average_precision_score(labels, scores)),
        "micro_molecules": int(len(labels)),
        # These endpoints require a score for every query/reference spectrum
        # pair. Full23 has already aggregated references to molecules.
        "gnps_10ppm_pooled_pairwise_auroc": np.nan,
        "gnps_10ppm_pooled_pairwise_auprc": np.nan,
        "gnps_mh_10ppm_pooled_pairwise_auroc": np.nan,
        "gnps_mh_10ppm_pooled_pairwise_auprc": np.nan,
    }
    for k in KS:
        row[f"recall@{k}"] = float(np.mean(rank <= k))
        row[f"near_recall@{k}"] = float(np.mean(rank[near] <= k))
    return row


def formula_cluster_ci(
    delta: np.ndarray,
    formulas: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    formulas = np.char.strip(np.asarray(formulas).astype("U"))
    unique, inverse = np.unique(formulas, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(delta, dtype=np.float64))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        bootstrap[index] = float(sums[chosen].sum() / counts[chosen].sum())
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "delta": float(np.mean(delta)),
        "delta_pp": float(100 * np.mean(delta)),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_low_pp": float(100 * low),
        "ci_high_pp": float(100 * high),
        "formula_clusters": int(len(unique)),
        "bootstrap_resamples": int(resamples),
        "seed": int(seed),
    }


def paired_comparison(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    resamples: int,
    seed: int,
) -> dict:
    merged = candidate.merge(
        baseline[["query_row", "query_formula", "rank"]],
        on=["query_row", "query_formula"],
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    if len(merged) != len(candidate):
        raise RuntimeError("paired method query axis drifted")
    candidate_ok = merged["rank_candidate"].to_numpy() == 1
    baseline_ok = merged["rank_baseline"].to_numpy() == 1
    delta = candidate_ok.astype(float) - baseline_ok.astype(float)
    corrected = int(np.sum(candidate_ok & ~baseline_ok))
    introduced = int(np.sum(~candidate_ok & baseline_ok))
    return {
        "queries": int(len(merged)),
        "corrected": corrected,
        "introduced": introduced,
        "net_correct": corrected - introduced,
        "legacy_risk_net_lambda2": corrected - 2 * introduced,
        "recall1_formula_cluster_ci": formula_cluster_ci(
            delta,
            merged["query_formula"].to_numpy(),
            resamples,
            seed,
        ),
        "mrr_delta": float(
            np.mean(
                1.0 / merged["rank_candidate"].to_numpy()
                - 1.0 / merged["rank_baseline"].to_numpy()
            )
        ),
    }


def internal_b_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    body = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for source, panel in (("identity", "identity_disjoint"), ("formula", "formula_disjoint")):
        metrics = body["panels"][source]["methods"]["B"]
        rows.append(
            {
                "panel": panel,
                "method": "B_gnps_nested_oof_internal_reference",
                "queries": metrics["queries"],
                "recall@1": metrics["recall@1"],
                "mrr": metrics["mrr"],
                "near_recall@1": metrics["near"]["recall@1"],
                "macro_query_auroc": metrics["macro_query_auroc"],
                "micro_auroc": metrics["micro_candidate_auroc"],
                "micro_auprc": metrics["micro_candidate_auprc"],
                "comparison_class": (
                    "internal GNPS nested-OOF development reference; not a "
                    "cross-source or external SOTA comparator"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 1000:
        raise ValueError("bootstrap-resamples must be at least 1000")

    old = pd.read_csv(args.run15_ladder)
    expected_columns = list(old.columns)
    new_rows: list[dict] = []
    comparisons: dict[str, dict] = {}
    tables: dict[str, pd.DataFrame] = {}
    v132_report = json.loads((args.v132 / "report.json").read_text(encoding="utf-8"))

    for panel_index, panel in enumerate(PANELS):
        with np.load(args.benchmark / f"panel_{panel}.npz", allow_pickle=False) as body:
            raw = {key: np.asarray(body[key]) for key in body.files}
        with np.load(
            args.v132 / f"candidate_predictions_{panel}.npz", allow_pickle=False
        ) as body:
            query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
            labels = np.asarray(body["labels"], dtype=bool)
            scores = np.asarray(body[METHOD], dtype=np.float64)
        if not (
            np.array_equal(query_ptr, np.asarray(raw["query_ptr"], dtype=np.int64))
            and np.array_equal(labels, np.asarray(raw["molecule_label"], dtype=bool))
            and len(scores) == len(labels)
        ):
            raise RuntimeError(f"{panel} candidate axis drifted")

        table = query_table(scores, labels, query_ptr, raw)
        tables[panel] = table
        row = ladder_row(table, scores, labels)
        row["panel"] = panel

        reported = v132_report["metrics"][panel][METHOD]
        for key in (
            "recall@1", "recall@2", "recall@3", "recall@5", "recall@10",
            "recall@20", "mrr", "mean_rank", "median_rank",
            "macro_query_auroc", "micro_candidate_auroc", "micro_candidate_auprc",
        ):
            ladder_key = {
                "micro_candidate_auroc": "micro_auroc",
                "micro_candidate_auprc": "micro_auprc",
            }.get(key, key)
            if not np.isclose(float(reported[key]), float(row[ladder_key]), atol=1e-12):
                raise RuntimeError(f"{panel}/{key} disagrees with frozen v1.3.2 report")

        official = pd.read_csv(
            args.run15_evaluation / f"queries_{panel}_official_dreams.csv.gz"
        )
        wse = pd.read_csv(
            args.run15_evaluation
            / f"queries_{panel}_weighted_spectral_entropy.csv.gz"
        )
        vs_official = paired_comparison(
            table, official, args.bootstrap_resamples, args.seed + 10 * panel_index
        )
        vs_wse = paired_comparison(
            table, wse, args.bootstrap_resamples, args.seed + 10 * panel_index + 1
        )
        row["corrected_vs_official"] = vs_official["corrected"]
        row["introduced_vs_official"] = vs_official["introduced"]
        row["risk_net_lambda2_vs_official"] = vs_official[
            "legacy_risk_net_lambda2"
        ]
        new_rows.append(row)
        comparisons[panel] = {
            "full23_v132_vs_official_dreams": vs_official,
            "full23_v132_vs_weighted_spectral_entropy": vs_wse,
        }

    appended = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    appended = appended[expected_columns]
    args.output.mkdir(parents=True)
    appended.to_csv(args.output / "ladder_full.csv", index=False)
    for panel, table in tables.items():
        table.to_csv(args.output / f"queries_{panel}_{METHOD}.csv.gz", index=False)

    selected = {
        METHOD,
        "weighted_spectral_entropy",
        "entropy_raw_public",
        "denoising_search_public",
        "noise_v1",
        "p2b_noise_v1_frozen",
        "official_dreams",
        "spec2vec_gnps_public",
        "ms2deepscore_2x_public",
    }
    paper = appended[appended["method"].isin(selected)].copy()
    paper.to_csv(args.output / "paper_core_table.csv", index=False)
    internal_b_reference(args.b_report).to_csv(
        args.output / "internal_B_reference.csv", index=False
    )

    rank_columns = (
        "recall@1", "mrr", "near_recall@1", "macro_query_auroc",
        "micro_auroc", "micro_auprc",
    )
    rank_summary = {}
    for panel in PANELS:
        subset = appended[appended["panel"] == panel]
        rank_summary[panel] = {
            metric: {
                "winner": str(subset.loc[subset[metric].idxmax(), "method"]),
                "value": float(subset[metric].max()),
            }
            for metric in rank_columns
        }

    verification = {
        "status": "FULL23_V132_SAME_AXIS_SOTA_COMPARISON_COMPLETE",
        "dataset": {
            "name": "GNPS Gold/Silver 10-ppm [M+H]+ candidate benchmark",
            "role": "consumed development",
            "truth_status": "consumed",
            "panels": {
                panel: int(len(tables[panel])) for panel in PANELS
            },
        },
        "method_count_in_ladder": int(appended["method"].nunique()),
        "rank_summary": rank_summary,
        "paired_comparisons": comparisons,
        "internal_B_reference": (
            "B is reported separately because its score is nested OOF after fitting on "
            "GNPS labels; it is not a same-status cross-source comparator."
        ),
        "unavailable_endpoint": (
            "Full23 pooled pairwise metrics are NA because the released endpoint is a "
            "molecule-candidate score after multi-reference aggregation."
        ),
        "claim_boundary": (
            "Full23 v1.3.2 is the development leader among the 15 frozen comparator "
            "methods on the reported molecule-ranking endpoints. The comparator set "
            "includes public/classical methods and frozen project baselines. The 0.9 scale was "
            "chosen after inspecting GNPS. This is neither unopened external "
            "confirmation nor a field-wide SOTA claim."
        ),
    }
    (args.output / "verification_report.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": verification["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

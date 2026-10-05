"""Frozen full-graph metrics for corrected MassSpecGym noise experiments.

The candidate graph is the evaluation ledger.  Query/candidate spectrum
cosines are first aggregated by maximum within each candidate molecule.  The
same unaggregated 10-ppm, same-adduct spectrum edges additionally define the
MassSpecGym pooled pairwise AUROC; that metric is not a NIST20 reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from noise_final_core import CandidateGraph, strict_rank


HELD_METRIC_EVIDENCE_SCHEMA = "noise_v6_held_metric_evidence_v1"


@dataclass(frozen=True)
class GraphScores:
    pair: np.ndarray
    molecule: np.ndarray


def expanded_indices(graph: CandidateGraph) -> tuple[np.ndarray, np.ndarray]:
    """Return molecule->query and pair->query indices in graph order."""
    molecule_query = np.repeat(
        np.arange(graph.n_queries, dtype=np.int64), np.diff(graph.query_ptr),
    )
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    if len(molecule_query) != len(graph.molecule_label):
        raise RuntimeError("query_ptr expansion does not align to molecules")
    if len(pair_query) != len(graph.pair_candidate_row):
        raise RuntimeError("molecule_ptr expansion does not align to spectrum edges")
    return molecule_query, pair_query


def _positions(rows: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise RuntimeError("embedding rows must be one-dimensional and unique")
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    at = np.searchsorted(sorted_rows, requested)
    if np.any(at == len(sorted_rows)):
        raise RuntimeError(f"embedding cache misses at least one {label} row")
    if not np.array_equal(sorted_rows[at], requested):
        raise RuntimeError(f"embedding cache misses at least one {label} row")
    return order[at]


def score_embeddings(
    graph: CandidateGraph,
    rows: np.ndarray,
    embeddings: np.ndarray,
    chunk_edges: int = 20_000,
) -> GraphScores:
    """Score all frozen graph edges without materialising giant 3-D tensors."""
    rows = np.asarray(rows, dtype=np.int64)
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise RuntimeError("embedding row/value schema is invalid")
    if chunk_edges < 1:
        raise ValueError("chunk_edges must be positive")
    norms = np.linalg.norm(np.asarray(embeddings[:: max(1, len(embeddings) // 2048)], dtype=np.float32), axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError("embeddings must be finite unit vectors")
    _, pair_query = expanded_indices(graph)
    query_position = _positions(rows, graph.query_row, "query")
    candidate_position = _positions(rows, graph.pair_candidate_row, "candidate")
    pair = np.empty(len(pair_query), dtype=np.float32)
    for left in range(0, len(pair), chunk_edges):
        right = min(left + chunk_edges, len(pair))
        query = np.asarray(
            embeddings[query_position[pair_query[left:right]]], dtype=np.float32,
        )
        candidate = np.asarray(
            embeddings[candidate_position[left:right]], dtype=np.float32,
        )
        pair[left:right] = np.einsum("ij,ij->i", query, candidate, optimize=True)
    if not np.all(np.isfinite(pair)):
        raise RuntimeError("full-graph scoring produced non-finite values")
    molecule = np.maximum.reduceat(pair, graph.molecule_ptr[:-1])
    return GraphScores(pair=pair, molecule=molecule)


def score_embedding_query_subset(
    graph: CandidateGraph,
    rows: np.ndarray,
    embeddings: np.ndarray,
    queries: np.ndarray,
) -> GraphScores:
    """Score complete candidate blocks for a registered query subset.

    Unselected positions are left at zero and must never be consumed.  The
    returned arrays retain full graph shape so ``full_metrics`` can apply its
    frozen masks and compute exactly the same metric family on a bounded
    development held panel.
    """
    rows = np.asarray(rows, dtype=np.int64)
    embeddings = np.asarray(embeddings)
    selected = np.asarray(queries, dtype=np.int64)
    if (
        embeddings.ndim != 2 or len(embeddings) != len(rows)
        or selected.ndim != 1 or not len(selected)
        or len(np.unique(selected)) != len(selected)
        or np.any((selected < 0) | (selected >= graph.n_queries))
    ):
        raise RuntimeError("subset embedding score inputs are malformed")
    index = {int(row): position for position, row in enumerate(rows)}
    if len(index) != len(rows):
        raise RuntimeError("subset embedding rows are duplicated")
    pair = np.zeros(len(graph.pair_candidate_row), dtype=np.float32)
    molecule = np.zeros(len(graph.molecule_label), dtype=np.float32)
    for query_value in selected:
        query = int(query_value)
        molecule_left, molecule_right = map(
            int, graph.query_ptr[query:query + 2],
        )
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        try:
            query_vector = embeddings[index[int(graph.query_row[query])]]
            candidate = embeddings[[
                index[int(row)]
                for row in graph.pair_candidate_row[pair_left:pair_right]
            ]]
        except KeyError as error:
            raise RuntimeError(
                "bounded embedding cache misses a selected candidate row"
            ) from error
        local_pair = np.asarray(candidate @ query_vector, dtype=np.float32)
        if not np.all(np.isfinite(local_pair)):
            raise RuntimeError("subset graph scoring produced non-finite values")
        pair[pair_left:pair_right] = local_pair
        local_ptr = (
            graph.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        )
        molecule[molecule_left:molecule_right] = np.maximum.reduceat(
            local_pair, local_ptr[:-1],
        )
    return GraphScores(pair=pair, molecule=molecule)


def official_scores(graph: CandidateGraph) -> GraphScores:
    pair = np.asarray(graph.features[:, graph.dreams_column], dtype=np.float32)
    return GraphScores(
        pair=pair,
        molecule=np.maximum.reduceat(pair, graph.molecule_ptr[:-1]),
    )


def query_table(
    graph: CandidateGraph,
    scores: GraphScores,
    queries: np.ndarray | None = None,
) -> pd.DataFrame:
    if len(scores.pair) != len(graph.pair_candidate_row):
        raise RuntimeError("pair score count differs from frozen graph")
    if len(scores.molecule) != len(graph.molecule_label):
        raise RuntimeError("molecule score count differs from frozen graph")
    selected = (
        np.arange(graph.n_queries, dtype=np.int64)
        if queries is None else np.asarray(queries, dtype=np.int64)
    )
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise RuntimeError("query selection must be one-dimensional and unique")
    if len(selected) == 0 or np.any((selected < 0) | (selected >= graph.n_queries)):
        raise RuntimeError("query selection is empty or out of range")
    records: list[dict[str, object]] = []
    for query in selected:
        left, right = map(int, graph.query_ptr[int(query):int(query) + 2])
        values = np.asarray(scores.molecule[left:right], dtype=np.float64)
        positive = float(values[0])
        negative = values[1:]
        rank = strict_rank(values)
        order = np.argsort(-values, kind="stable")
        top1_top2_gap = float(values[order[0]] - values[order[1]])
        records.append({
            "query_index": int(query),
            "query_row": int(graph.query_row[query]),
            "query_ik14": str(graph.query_ik14[query]),
            "query_formula": str(graph.query_formula[query]),
            "near": bool(graph.query_has_near[query]),
            "rank": int(rank),
            "reciprocal_rank": float(1.0 / rank),
            "macro_query_auc": float(
                (np.sum(positive > negative) + 0.5 * np.sum(positive == negative))
                / len(negative)
            ),
            # There is exactly one positive molecule per frozen query.  Under
            # the strict tie policy, its per-query AP is reciprocal rank.
            "macro_query_auprc": float(1.0 / rank),
            "positive_vs_best_negative_margin": float(positive - np.max(negative)),
            # Raw confidence is reported but is not monotone in retrieval
            # quality: a confidently wrong top-1 also has a large gap.  The
            # signed form is positive only for a correct top-1 and is the
            # quantity suitable for a promotion direction check.
            "top1_top2_gap": top1_top2_gap,
            "signed_top1_top2_gap": (
                top1_top2_gap if rank == 1 else -top1_top2_gap
            ),
        })
    return pd.DataFrame.from_records(records)


def _metric_summary(table: pd.DataFrame) -> dict[str, float | int]:
    rank = table["rank"].to_numpy(np.int64)
    result: dict[str, float | int] = {
        "queries": int(len(table)),
        "mrr": float(np.mean(1.0 / rank)),
        "mean_rank": float(np.mean(rank)),
        "median_rank": float(np.median(rank)),
        "macro_query_auroc": float(table["macro_query_auc"].mean()),
        "macro_query_auprc": float(table["macro_query_auprc"].mean()),
        "mean_positive_vs_best_negative_margin": float(
            table["positive_vs_best_negative_margin"].mean()
        ),
        "mean_top1_top2_gap": float(table["top1_top2_gap"].mean()),
        "mean_signed_top1_top2_gap": float(
            table["signed_top1_top2_gap"].mean()
        ),
    }
    for cutoff in (1, 2, 3, 5, 10, 20):
        result[f"recall@{cutoff}"] = float(np.mean(rank <= cutoff))
    return result


def held_metric_evidence(
    graph: CandidateGraph,
    score_panels: dict[str, GraphScores],
    query_adduct: np.ndarray | None,
    queries: np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialize the exact ledgers needed to independently replay pooled metrics."""
    selected = np.asarray(queries, dtype=np.int64)
    if (
        selected.ndim != 1 or not len(selected)
        or len(np.unique(selected)) != len(selected)
        or np.any((selected < 0) | (selected >= graph.n_queries))
    ):
        raise RuntimeError("held metric evidence query selection is malformed")
    if not score_panels:
        raise RuntimeError("held metric evidence requires at least one score panel")
    molecule_query, pair_query = expanded_indices(graph)
    molecule_mask = np.isin(molecule_query, selected, assume_unique=False)
    pair_mask = np.isin(pair_query, selected, assume_unique=False)
    molecule_label = np.asarray(graph.molecule_label[molecule_mask], dtype=np.uint8)
    pair_label = np.asarray(
        np.repeat(graph.molecule_label, np.diff(graph.molecule_ptr))[pair_mask],
        dtype=np.uint8,
    )
    if set(map(int, np.unique(molecule_label))) != {0, 1}:
        raise RuntimeError("held metric molecule evidence is not binary")
    if set(map(int, np.unique(pair_label))) != {0, 1}:
        raise RuntimeError("held metric spectrum-pair evidence is not binary")
    if query_adduct is None:
        pair_is_mh = np.zeros(len(pair_label), dtype=bool)
    else:
        query_adduct = np.asarray(query_adduct, dtype=str)
        if query_adduct.shape != (graph.n_queries,):
            raise RuntimeError("query adducts do not align to graph queries")
        pair_is_mh = np.asarray(
            query_adduct[pair_query[pair_mask]] == "[M+H]+", dtype=bool,
        )
    output = {
        "schema_version": np.asarray(HELD_METRIC_EVIDENCE_SCHEMA),
        "query_index": np.ascontiguousarray(selected),
        "molecule_label": np.ascontiguousarray(molecule_label),
        "pair_label": np.ascontiguousarray(pair_label),
        "pair_is_mh": np.ascontiguousarray(pair_is_mh),
    }
    for name, scores in score_panels.items():
        if (
            len(scores.molecule) != len(graph.molecule_label)
            or len(scores.pair) != len(graph.pair_candidate_row)
        ):
            raise RuntimeError(f"held metric score panel has wrong shape: {name}")
        molecule_score = np.asarray(scores.molecule[molecule_mask], dtype=np.float32)
        pair_score = np.asarray(scores.pair[pair_mask], dtype=np.float32)
        if not np.all(np.isfinite(molecule_score)) or not np.all(np.isfinite(pair_score)):
            raise RuntimeError(f"held metric score panel is non-finite: {name}")
        output[f"{name}_molecule_score"] = np.ascontiguousarray(molecule_score)
        output[f"{name}_pair_score"] = np.ascontiguousarray(pair_score)
    return output


def full_metrics(
    graph: CandidateGraph,
    scores: GraphScores,
    query_adduct: np.ndarray | None = None,
    queries: np.ndarray | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute retrieval, candidate and spectrum-edge metrics in one contract."""
    table = query_table(graph, scores, queries)
    selected = table["query_index"].to_numpy(np.int64)
    evidence = held_metric_evidence(
        graph, {"panel": scores}, query_adduct, selected,
    )
    molecule_labels = evidence["molecule_label"]
    molecule_values = evidence["panel_molecule_score"]
    pair_labels = evidence["pair_label"]
    pair_values = evidence["panel_pair_score"]
    report: dict[str, object] = {
        "retrieval": _metric_summary(table),
        "near_subset": _metric_summary(table.loc[table["near"]].reset_index(drop=True)),
        "micro_candidate": {
            "molecules": int(len(molecule_labels)),
            "auroc": float(roc_auc_score(molecule_labels, molecule_values)),
            "auprc": float(average_precision_score(molecule_labels, molecule_values)),
        },
        "massspecgym_10ppm_pooled_pairwise": {
            "exact_nist20_paper_replication": False,
            "edge_semantics": "directed query-candidate spectrum edges; same adduct; strict 10 ppm",
            "spectrum_pairs": int(len(pair_labels)),
            "positive_pairs": int(np.sum(pair_labels == 1)),
            "negative_pairs": int(np.sum(pair_labels == 0)),
            "auroc": float(roc_auc_score(pair_labels, pair_values)),
            "auprc": float(average_precision_score(pair_labels, pair_values)),
        },
    }
    if query_adduct is not None:
        mh_pair_mask = evidence["pair_is_mh"]
        mh_labels = pair_labels[mh_pair_mask]
        mh_values = pair_values[mh_pair_mask]
        report["massspecgym_mh_10ppm_pooled_pairwise"] = {
            "exact_nist20_paper_replication": False,
            "query_adduct": "[M+H]+",
            "spectrum_pairs": int(len(mh_labels)),
            "positive_pairs": int(np.sum(mh_labels == 1)),
            "negative_pairs": int(np.sum(mh_labels == 0)),
            "auroc": float(roc_auc_score(mh_labels, mh_values)),
            "auprc": float(average_precision_score(mh_labels, mh_values)),
        }
    return report, table


def paired_outcome_table(
    official: pd.DataFrame, candidate: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["query_index", "query_row", "query_ik14", "query_formula", "near"]
    if official[keys].to_dict("list") != candidate[keys].to_dict("list"):
        raise RuntimeError("official and candidate query tables are not exactly aligned")
    output = official[keys].copy()
    output["official_rank"] = official["rank"].to_numpy(np.int64)
    output["candidate_rank"] = candidate["rank"].to_numpy(np.int64)
    output["corrected"] = output.official_rank.gt(1) & output.candidate_rank.eq(1)
    output["introduced"] = output.official_rank.eq(1) & output.candidate_rank.gt(1)
    output["risk_net_lambda2"] = (
        output.corrected.astype(np.int8) - 2 * output.introduced.astype(np.int8)
    )
    for name in (
        "reciprocal_rank", "macro_query_auc", "macro_query_auprc",
        "positive_vs_best_negative_margin", "top1_top2_gap",
        "signed_top1_top2_gap",
    ):
        output[f"official_{name}"] = official[name].to_numpy(np.float64)
        output[f"candidate_{name}"] = candidate[name].to_numpy(np.float64)
    return output

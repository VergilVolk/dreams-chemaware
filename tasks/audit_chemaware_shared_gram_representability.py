"""Audit whether a frozen chemical score residual fits shared-embedding geometry.

No DreaMS weights are loaded or updated.  The audit uses cached official
embeddings, the existing full candidate graph, and the frozen ICEBERG strict
corrective ledger.  Every spectrum receives an independent tangent update, so
the result is an upper bound before clean-spectrum observability/generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from chemaware_shared_gram_representability_core import (
    apply_normalized_updates,
    build_edge_operator,
    solve_tangent_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data/validation/chemaware_iceberg_corrective_residual_ledger_v2/ledger.npz",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_shared_gram_representability_v1",
    )
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-2])
    parser.add_argument("--max-active-queries", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(scores: np.ndarray) -> int:
    return 1 + int(np.sum(np.asarray(scores)[1:] >= float(scores[0])))


def retrieval_summary(old_rank: np.ndarray, new_rank: np.ndarray) -> dict[str, object]:
    old_rank = np.asarray(old_rank, dtype=np.int64)
    new_rank = np.asarray(new_rank, dtype=np.int64)
    return {
        "queries": int(len(old_rank)),
        "baseline_recall1": float(np.mean(old_rank == 1)),
        "recall1": float(np.mean(new_rank == 1)),
        "delta_recall1": float(np.mean(new_rank == 1) - np.mean(old_rank == 1)),
        "corrected": int(np.sum((old_rank > 1) & (new_rank == 1))),
        "introduced": int(np.sum((old_rank == 1) & (new_rank > 1))),
        "mean_reciprocal_rank_delta": float(np.mean(1.0 / new_rank - 1.0 / old_rank)),
        "rank_improved": int(np.sum(new_rank < old_rank)),
        "rank_worsened": int(np.sum(new_rank > old_rank)),
    }


def main() -> None:
    args = parse_args()
    if not (0 < args.dose <= 1.0):
        raise ValueError("dose must lie in (0, 1]")
    if any(value <= 0 for value in args.ridge):
        raise ValueError("all ridge values must be positive")
    if args.max_active_queries < 0:
        raise ValueError("max-active-queries must be nonnegative")

    ledger = np.load(args.ledger, allow_pickle=False)
    graph_path = args.graph_dir / "graph.npz"
    graph = np.load(graph_path, allow_pickle=True)
    required_ledger = {
        "query_ptr", "query_row", "query_formula", "active_query", "molecule_ik14",
        "centered_residual", "structure_swapped_centered_residual",
        "peak_permuted_centered_residual",
    }
    required_graph = {
        "query_row", "query_ptr", "molecule_ptr", "molecule_ik14", "pair_candidate_row",
        "features", "molecule_label",
    }
    if not required_ledger.issubset(ledger.files):
        raise RuntimeError(f"ledger arrays missing: {sorted(required_ledger - set(ledger.files))}")
    if not required_graph.issubset(graph.files):
        raise RuntimeError(f"graph arrays missing: {sorted(required_graph - set(graph.files))}")
    if not np.array_equal(ledger["query_row"], graph["query_row"]):
        raise RuntimeError("ledger and ICEBERG graph query rows are not aligned")
    if not np.array_equal(
        np.asarray(ledger["molecule_ik14"]).astype(str),
        np.asarray(graph["molecule_ik14"]).astype(str),
    ):
        raise RuntimeError("ledger and ICEBERG graph candidate identities are not aligned")

    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    cache_embeddings = np.load(
        args.token_dir / "official_embeddings_f32.npy", mmap_mode="r",
    )
    if cache_embeddings.shape != (len(cache_rows), 1024):
        raise RuntimeError("official embedding cache has an unexpected shape")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    active = np.flatnonzero(np.asarray(ledger["active_query"], dtype=bool))
    if args.max_active_queries:
        active = active[: args.max_active_queries]
    if len(active) == 0:
        raise RuntimeError("no active corrective queries were selected")

    edge_left: list[int] = []
    edge_right: list[int] = []
    edge_baseline: list[float] = []
    target = {"correct": [], "structure_swapped": [], "peak_permuted": []}
    query_edge_ptr = [0]
    query_rows: list[int] = []
    query_formulas: list[str] = []
    all_reference_rows: list[list[np.ndarray]] = []
    embedding_by_row: dict[int, np.ndarray] = {}
    max_reference_switch_capacity = 0

    ledger_ptr = np.asarray(ledger["query_ptr"], dtype=np.int64)
    graph_query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    pair_candidate_row = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
    pair_official_score = np.asarray(graph["features"][:, 0], dtype=np.float64)
    graph_query_rows = np.asarray(graph["query_row"], dtype=np.int64)
    residual_names = {
        "correct": "centered_residual",
        "structure_swapped": "structure_swapped_centered_residual",
        "peak_permuted": "peak_permuted_centered_residual",
    }

    for ledger_query in active:
        query_row = int(ledger["query_row"][ledger_query])
        if query_row not in cache_position:
            raise RuntimeError(f"active query row is absent from a required cache: {query_row}")
        ledger_left, ledger_right = ledger_ptr[ledger_query : ledger_query + 2]
        molecule_left, molecule_right = graph_query_ptr[ledger_query : ledger_query + 2]
        if (int(ledger_left), int(ledger_right)) != (int(molecule_left), int(molecule_right)):
            raise RuntimeError(f"ledger and ICEBERG graph pointers drifted at query {ledger_query}")

        query_embedding = np.asarray(cache_embeddings[cache_position[query_row]], dtype=np.float64)
        query_embedding /= np.linalg.norm(query_embedding)
        embedding_by_row[query_row] = query_embedding
        query_rows.append(query_row)
        query_formulas.append(str(ledger["query_formula"][ledger_query]))
        candidate_references: list[np.ndarray] = []

        for local, molecule in enumerate(range(int(molecule_left), int(molecule_right))):
            reference_rows = pair_candidate_row[molecule_ptr[molecule] : molecule_ptr[molecule + 1]]
            positions = np.asarray([cache_position[int(row)] for row in reference_rows], dtype=np.int64)
            reference_embeddings = np.asarray(cache_embeddings[positions], dtype=np.float64)
            reference_embeddings /= np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
            scores = reference_embeddings @ query_embedding
            winner = int(np.argmax(scores))
            winner_row = int(reference_rows[winner])
            embedding_by_row[winner_row] = reference_embeddings[winner]
            edge_left.append(query_row)
            edge_right.append(winner_row)
            edge_baseline.append(float(scores[winner]))
            candidate_references.append(np.asarray(reference_rows, dtype=np.int64))
            if len(reference_rows) > 1:
                max_reference_switch_capacity += 1
            for arm, name in residual_names.items():
                target[arm].append(float(ledger[name][int(ledger_left) + local]) * args.dose)
        all_reference_rows.append(candidate_references)
        query_edge_ptr.append(len(edge_left))

    edge_left_array = np.asarray(edge_left, dtype=np.int64)
    edge_right_array = np.asarray(edge_right, dtype=np.int64)
    baseline = np.asarray(edge_baseline, dtype=np.float64)
    query_edge_ptr_array = np.asarray(query_edge_ptr, dtype=np.int64)
    old_rank = np.asarray(
        [rank(baseline[left:right]) for left, right in zip(query_edge_ptr_array[:-1], query_edge_ptr_array[1:])],
        dtype=np.int64,
    )
    if np.any(old_rank == 1):
        raise RuntimeError("strict corrective ledger contains an official-correct active query")

    operator = build_edge_operator(edge_left_array, edge_right_array, embedding_by_row)
    node_position = {int(row): index for index, row in enumerate(operator.node_rows)}
    cache_embedding_by_row: dict[int, np.ndarray] = {}

    def cached_embedding(row: int) -> np.ndarray:
        if row in cache_embedding_by_row:
            return cache_embedding_by_row[row]
        value = np.asarray(cache_embeddings[cache_position[row]], dtype=np.float32).copy()
        value /= np.linalg.norm(value)
        cache_embedding_by_row[row] = value
        return value

    full_old_rank: list[int] = []
    for query in range(len(graph_query_ptr) - 1):
        left, right = graph_query_ptr[query : query + 2]
        scores = np.asarray(
            [
                np.max(pair_official_score[molecule_ptr[molecule] : molecule_ptr[molecule + 1]])
                for molecule in range(int(left), int(right))
            ],
            dtype=np.float64,
        )
        full_old_rank.append(rank(scores))
    full_old_rank_array = np.asarray(full_old_rank, dtype=np.int64)

    def evaluate_full_graph(updated: dict[int, np.ndarray]) -> tuple[np.ndarray, int]:
        ranks: list[int] = []
        switches = 0
        for query, query_row in enumerate(graph_query_rows):
            query_value = updated.get(int(query_row), cached_embedding(int(query_row)))
            left, right = graph_query_ptr[query : query + 2]
            scores: list[float] = []
            for molecule in range(int(left), int(right)):
                pair_left, pair_right = molecule_ptr[molecule : molecule + 2]
                reference_rows = pair_candidate_row[pair_left:pair_right]
                values = np.asarray(
                    [
                        updated.get(int(row), cached_embedding(int(row))) @ query_value
                        for row in reference_rows
                    ],
                    dtype=np.float64,
                )
                scores.append(float(np.max(values)))
                old_winner = int(reference_rows[int(np.argmax(pair_official_score[pair_left:pair_right]))])
                new_winner = int(reference_rows[int(np.argmax(values))])
                switches += int(old_winner != new_winner)
            ranks.append(rank(np.asarray(scores)))
        return np.asarray(ranks, dtype=np.int64), switches

    arm_reports: dict[str, object] = {}
    for arm, values in target.items():
        target_values = np.asarray(values, dtype=np.float64)
        oracle_rank = np.asarray(
            [
                rank(baseline[left:right] + target_values[left:right])
                for left, right in zip(query_edge_ptr_array[:-1], query_edge_ptr_array[1:])
            ],
            dtype=np.int64,
        )
        ridge_reports = []
        for ridge in args.ridge:
            projection = solve_tangent_projection(operator, target_values, ridge=float(ridge))
            projected = np.asarray(projection["projected"])
            linear_rank = np.asarray(
                [
                    rank(baseline[left:right] + projected[left:right])
                    for left, right in zip(query_edge_ptr_array[:-1], query_edge_ptr_array[1:])
                ],
                dtype=np.int64,
            )

            updated = apply_normalized_updates(operator, projection["updates"])
            finite_scores: list[float] = []
            reference_switches = 0
            for query_index, query_row in enumerate(query_rows):
                query_value = updated.get(query_row, cached_embedding(query_row))
                left, right = query_edge_ptr_array[query_index : query_index + 2]
                for local, reference_rows in enumerate(all_reference_rows[query_index]):
                    scores = np.asarray(
                        [updated.get(int(row), cached_embedding(int(row))) @ query_value for row in reference_rows],
                        dtype=np.float64,
                    )
                    finite_scores.append(float(np.max(scores)))
                    old_winner = edge_right_array[int(left) + local]
                    new_winner = int(reference_rows[int(np.argmax(scores))])
                    reference_switches += int(new_winner != old_winner)
            finite_scores_array = np.asarray(finite_scores, dtype=np.float64)
            finite_rank = np.asarray(
                [
                    rank(finite_scores_array[left:right])
                    for left, right in zip(query_edge_ptr_array[:-1], query_edge_ptr_array[1:])
                ],
                dtype=np.int64,
            )
            updates = np.asarray(projection["updates"])
            updated_matrix = np.stack([updated[int(row)] for row in operator.node_rows])
            preservation = np.einsum("ij,ij->i", operator.node_embeddings, updated_matrix)
            broad_rank, broad_reference_switches = evaluate_full_graph(updated)
            ridge_reports.append(
                {
                    "ridge": float(ridge),
                    "linear_projection": {
                        "explained_energy_fraction": float(projection["explained_energy_fraction"]),
                        "target_projected_cosine": float(projection["target_projected_cosine"]),
                        "rms_edge_error": float(np.sqrt(np.mean((target_values - projected) ** 2))),
                        "retrieval": retrieval_summary(old_rank, linear_rank),
                    },
                    "finite_normalized_shared_embedding": {
                        "retrieval": retrieval_summary(old_rank, finite_rank),
                        "reference_max_switches": int(reference_switches),
                        "mean_preservation_cosine": float(np.mean(preservation)),
                        "minimum_preservation_cosine": float(np.min(preservation)),
                        "q01_preservation_cosine": float(np.quantile(preservation, 0.01)),
                    },
                    "full_2048_graph_side_effects": {
                        "retrieval": retrieval_summary(full_old_rank_array, broad_rank),
                        "reference_max_switches": int(broad_reference_switches),
                    },
                    "free_update": {
                        "mean_l2": float(np.mean(projection["update_norms"])),
                        "q95_l2": float(np.quantile(projection["update_norms"], 0.95)),
                        "maximum_l2": float(np.max(projection["update_norms"])),
                    },
                }
            )
        arm_reports[arm] = {
            "target": {
                "dose": args.dose,
                "mean_absolute_edge_delta": float(np.mean(np.abs(target_values))),
                "maximum_absolute_edge_delta": float(np.max(np.abs(target_values))),
                "oracle_retrieval": retrieval_summary(old_rank, oracle_rank),
            },
            "ridge_grid": ridge_reports,
        }

    unordered = [tuple(sorted((int(i), int(j)))) for i, j in zip(edge_left_array, edge_right_array)]
    reciprocal_duplicates = len(unordered) - len(set(unordered))
    report = {
        "status": "CHEMAWARE_SHARED_GRAM_REPRESENTABILITY_AUDIT_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "claim_limit": (
            "Free per-spectrum tangent updates provide a shared-geometry ceiling only; "
            "they do not establish clean-spectrum observability or encoder generalization."
        ),
        "data": {
            "active_queries": int(len(active)),
            "query_formula_clusters": int(len(set(query_formulas))),
            "candidate_edges": int(len(edge_left_array)),
            "unique_spectrum_nodes": int(len(operator.node_rows)),
            "reciprocal_or_duplicate_unordered_edges": int(reciprocal_duplicates),
            "candidate_molecules_with_multiple_references": int(max_reference_switch_capacity),
            "baseline_recall1": float(np.mean(old_rank == 1)),
        },
        "operator": {
            "definition": "delta K_ij = z_i^T u_j + u_i^T z_j; z_i^T u_i = 0",
            "edge_gram_shape": list(operator.edge_gram.shape),
            "edge_gram_nnz": int(operator.edge_gram.nnz),
            "free_per_spectrum_updates": True,
            "shared_symmetric_score": True,
            "normalization_applied_for_finite_evaluation": True,
            "molecule_max_linearization_uses_official_winning_reference": True,
        },
        "arms": arm_reports,
        "provenance": {
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": sha256(args.ledger),
            "iceberg_graph": str(graph_path.resolve()),
            "iceberg_graph_sha256": sha256(graph_path),
            "cache_rows_sha256": sha256(args.token_dir / "rows.npy"),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "active_queries": report["data"]["active_queries"],
        "candidate_edges": report["data"]["candidate_edges"],
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

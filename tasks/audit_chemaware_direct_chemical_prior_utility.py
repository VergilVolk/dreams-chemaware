"""Audit corrective utility of a frozen direct chemical-prior ledger.

Teacher construction is complete before this script reads identity labels or
official DreaMS scores.  This is therefore an action-utility audit, not rule
discovery.  By default it inspects only formula folds 0--2; folds 3 and 4 stay
unread for later confirmation/reserve use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_direct_action_core import formula_bootstrap  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT
        / "data/validation/chemaware_direct_chemical_prior_full_manifest_ledger_v1"
        / "ledger.npz",
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_direct_chemical_prior_utility_dev_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=(0.005, 0.01, 0.02, 0.05, 0.10)
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def exact_position(rows: np.ndarray) -> dict[int, int]:
    values = np.asarray(rows, dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise RuntimeError("embedding cache rows are not unique")
    return {int(row): index for index, row in enumerate(values)}


def strict_rank(scores: np.ndarray, labels: np.ndarray) -> int:
    positive = np.flatnonzero(np.asarray(labels) == 1)
    if len(positive) != 1:
        raise RuntimeError("every query must have exactly one positive molecule")
    pos = int(positive[0])
    negative = np.flatnonzero(np.asarray(labels) == 0)
    return 1 + int(np.sum(np.asarray(scores)[negative] >= float(scores[pos])))


def metrics(
    old_rank: np.ndarray,
    new_rank: np.ndarray,
    candidate_count: np.ndarray,
    formulas: np.ndarray,
    total_queries: int,
    total_pairwise_denominator: int,
    seed: int,
    draws: int,
) -> dict:
    old_ok, new_ok = old_rank == 1, new_rank == 1
    delta_r1 = new_ok.astype(np.float64) - old_ok.astype(np.float64)
    result = {
        "active_queries": int(len(old_rank)),
        "corrected": int(np.sum(~old_ok & new_ok)),
        "introduced": int(np.sum(old_ok & ~new_ok)),
        "risk_utility_corrected_minus_2_introduced": int(
            np.sum(~old_ok & new_ok) - 2 * np.sum(old_ok & ~new_ok)
        ),
        "global_delta_recall1": float(np.sum(delta_r1) / total_queries),
        "global_delta_mrr": float(np.sum(1 / new_rank - 1 / old_rank) / total_queries),
        "global_delta_macro_auc": float(
            np.sum((old_rank - new_rank) / np.maximum(candidate_count - 1, 1))
            / total_queries
        ),
        "global_delta_micro_auc": float(
            np.sum(old_rank - new_rank) / total_pairwise_denominator
        ),
    }
    for k in (5, 10, 20, 50):
        result[f"global_delta_recall{k}"] = float(
            np.sum((new_rank <= k).astype(np.int8) - (old_rank <= k).astype(np.int8))
            / total_queries
        )
    # This interval describes the action-covered population only.  Global
    # effect sizes above retain every inactive query in the denominator and are
    # therefore reported separately rather than conflated with this interval.
    result["active_formula_bootstrap_delta_recall1"] = formula_bootstrap(
        delta_r1, formulas, seed, draws
    )
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_draws < 10_000:
        raise ValueError("bootstrap draws were weakened below 10,000")
    if not args.folds or any(fold not in (0, 1, 2) for fold in args.folds):
        raise ValueError("development audit may inspect only formula folds 0, 1, and 2")
    alpha_grid = np.asarray(args.alpha_grid, dtype=np.float64)
    if (
        np.any(~np.isfinite(alpha_grid))
        or np.any(alpha_grid <= 0)
        or len(np.unique(alpha_grid)) != len(alpha_grid)
    ):
        raise ValueError("alpha grid must contain unique positive finite doses")

    with np.load(args.graph, allow_pickle=True) as graph:
        query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
        pair_candidate_row = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
        query_row = np.asarray(graph["query_row"], dtype=np.int64)
        query_formula = np.asarray(graph["query_formula"]).astype(str)
        molecule_label = np.asarray(graph["molecule_label"], dtype=np.int8)
    with np.load(args.ledger, allow_pickle=False) as ledger:
        if not np.array_equal(query_ptr, ledger["query_ptr"]):
            raise RuntimeError("ledger query graph is not the requested candidate graph")
        if not np.array_equal(query_row, ledger["query_row"]):
            raise RuntimeError("ledger query rows are not aligned")
        formula_fold = np.asarray(ledger["query_formula_fold"], dtype=np.int16)
        active = np.asarray(ledger["active_query"], dtype=bool)
        correct_prior = np.asarray(ledger["centered_residual"], dtype=np.float32)
        swapped_prior = np.asarray(
            ledger["structure_swapped_centered_residual"], dtype=np.float32
        )
        peak_prior = (
            np.asarray(ledger["peak_permuted_centered_residual"], dtype=np.float32)
            if "peak_permuted_centered_residual" in ledger.files
            else None
        )

    audited_query = np.flatnonzero(active & np.isin(formula_fold, args.folds))
    if not len(audited_query):
        raise RuntimeError("no active query is available in the development folds")
    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    row_position = exact_position(cache_rows)
    official = np.load(
        args.token_dir / "official_embeddings_f32.npy", mmap_mode="r"
    )

    old_rank = []
    candidate_count = []
    formulas = []
    score_blocks = []
    correct_blocks = []
    swapped_blocks = []
    peak_blocks = []
    chemical_advantage = []
    official_margin = []
    correct_boundary_delta = []
    structure_boundary_delta = []
    peak_boundary_delta = []
    for query in audited_query:
        qpos = row_position[int(query_row[query])]
        left, right = map(int, query_ptr[query : query + 2])
        scores = []
        for molecule in range(left, right):
            edge_left, edge_right = map(int, molecule_ptr[molecule : molecule + 2])
            positions = np.asarray(
                [row_position[int(row)] for row in pair_candidate_row[edge_left:edge_right]],
                dtype=np.int64,
            )
            scores.append(float(np.max(official[positions] @ official[qpos])))
        scores = np.asarray(scores, dtype=np.float64)
        labels = molecule_label[left:right]
        positive = int(np.flatnonzero(labels == 1)[0])
        negative = np.flatnonzero(labels == 0)
        target = correct_prior[left:right].astype(np.float64)
        chemical_advantage.append(float(target[positive] - np.max(target[negative])))
        winning_negative = int(negative[np.argmax(scores[negative])])
        official_margin.append(float(scores[positive] - scores[winning_negative]))
        correct_boundary_delta.append(
            float(target[positive] - target[winning_negative])
        )
        structure_target = swapped_prior[left:right].astype(np.float64)
        structure_boundary_delta.append(
            float(structure_target[positive] - structure_target[winning_negative])
        )
        if peak_prior is not None:
            peak_target = peak_prior[left:right].astype(np.float64)
            peak_boundary_delta.append(
                float(peak_target[positive] - peak_target[winning_negative])
            )
        old_rank.append(strict_rank(scores, labels))
        candidate_count.append(len(scores))
        formulas.append(query_formula[query])
        score_blocks.append(scores)
        correct_blocks.append(target)
        swapped_blocks.append(swapped_prior[left:right].astype(np.float64))
        if peak_prior is not None:
            peak_blocks.append(peak_prior[left:right].astype(np.float64))

    old_rank = np.asarray(old_rank, dtype=np.int64)
    candidate_count = np.asarray(candidate_count, dtype=np.int64)
    formulas = np.asarray(formulas)
    chemical_advantage = np.asarray(chemical_advantage, dtype=np.float64)
    official_margin = np.asarray(official_margin, dtype=np.float64)
    correct_boundary_delta = np.asarray(correct_boundary_delta, dtype=np.float64)
    structure_boundary_delta = np.asarray(structure_boundary_delta, dtype=np.float64)
    causal_advantage = correct_boundary_delta - structure_boundary_delta
    if peak_prior is not None:
        peak_boundary_delta = np.asarray(peak_boundary_delta, dtype=np.float64)
        causal_advantage = np.minimum(
            causal_advantage, correct_boundary_delta - peak_boundary_delta
        )
    strict_corrective = (
        (official_margin <= 0)
        & (correct_boundary_delta > 0)
        & (causal_advantage > 0)
    )
    total_mask = np.isin(formula_fold, args.folds)
    total_queries = int(np.sum(total_mask))
    total_pairwise_denominator = int(
        np.sum(np.maximum(np.diff(query_ptr)[total_mask] - 1, 1))
    )

    arms = {"correct": correct_blocks, "structure_swapped": swapped_blocks}
    if peak_prior is not None:
        arms["peak_permuted"] = peak_blocks
    grid = []
    for alpha_index, alpha in enumerate(alpha_grid):
        row = {"alpha": float(alpha), "arms": {}}
        for arm_index, (arm, targets) in enumerate(arms.items()):
            new_rank = np.asarray(
                [
                    strict_rank(score + alpha * target, molecule_label[query_ptr[q] : query_ptr[q + 1]])
                    for q, score, target in zip(audited_query, score_blocks, targets)
                ],
                dtype=np.int64,
            )
            row["arms"][arm] = metrics(
                old_rank,
                new_rank,
                candidate_count,
                formulas,
                total_queries,
                total_pairwise_denominator,
                args.seed + 100 * alpha_index + arm_index,
                args.bootstrap_draws,
            )
        correct = row["arms"]["correct"]
        controls = [value for key, value in row["arms"].items() if key != "correct"]
        lower = correct["active_formula_bootstrap_delta_recall1"][
            "formula_cluster_bootstrap_95ci"
        ][0]
        row["passes_development_gate"] = bool(
            correct["corrected"] >= 3
            and correct["risk_utility_corrected_minus_2_introduced"] > 0
            and correct["introduced"] * 2 <= correct["corrected"]
            and correct["risk_utility_corrected_minus_2_introduced"]
            > max(
                control["risk_utility_corrected_minus_2_introduced"]
                for control in controls
            )
            and lower > 0
        )
        grid.append(row)

    passing = [row for row in grid if row["passes_development_gate"]]
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda row: (
                -row["arms"]["correct"]["risk_utility_corrected_minus_2_introduced"],
                row["alpha"],
            ),
        )[0]["alpha"]
    report = {
        "status": (
            "CHEMAWARE_DIRECT_PRIOR_UTILITY_DEVELOPMENT_PASS"
            if selected is not None
            else "CHEMAWARE_DIRECT_PRIOR_UTILITY_DEVELOPMENT_FAIL"
        ),
        "training_authorized": False,
        "embedding_confirmation_fold3_consumed": bool(3 in args.folds),
        "reserve_fold4_consumed": bool(4 in args.folds),
        "audited_formula_folds": list(args.folds),
        "untouched_formula_folds": sorted(set(range(5)) - set(args.folds)),
        "audited_total_queries": total_queries,
        "audited_active_queries": int(len(audited_query)),
        "chemical_target_direction": {
            "helpful": int(np.sum(chemical_advantage > 0)),
            "neutral": int(np.sum(chemical_advantage == 0)),
            "harmful": int(np.sum(chemical_advantage < 0)),
            "mean_true_vs_hardest_negative_advantage": float(np.mean(chemical_advantage)),
        },
        "strict_corrective_action_route": {
            "active_official_errors": int(np.sum(official_margin <= 0)),
            "errors_with_positive_correct_chemical_boundary_delta": int(
                np.sum((official_margin <= 0) & (correct_boundary_delta > 0))
            ),
            "errors_strictly_positive_vs_every_available_control": int(
                np.sum(strict_corrective)
            ),
            "corrective_formula_clusters": int(
                len(np.unique(formulas[strict_corrective]))
            ),
            "noncorrective_chemical_weight_exact_zero_required": True,
            "causal_advantage_positive_quantiles": (
                {
                    str(q): float(np.quantile(causal_advantage[strict_corrective], q))
                    for q in (0.0, 0.5, 0.9, 1.0)
                }
                if np.any(strict_corrective)
                else {}
            ),
        },
        "development_gate": {
            "minimum_corrected": 3,
            "risk_utility": "corrected - 2*introduced must be positive",
            "introduction_budget": "introduced <= corrected/2",
            "matched_control": "correct risk utility must exceed every available causal control",
            "formula_cluster_ci": "lower 95% bound on covered-query Recall@1 delta > 0",
        },
        "selected_alpha_for_future_confirmation": selected,
        "grid": grid,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

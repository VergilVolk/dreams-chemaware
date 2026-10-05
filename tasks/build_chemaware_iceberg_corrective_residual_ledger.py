"""Convert the passed ICEBERG teacher into strict corrective score residuals.

This recovers the strongest existing ChemAware action source without creating
synthetic spectra or selecting peaks through the DreaMS input Jacobian.
ICEBERG distances become candidate-centred score targets.  An action is active
only when the official DreaMS query is wrong, the correct ICEBERG teacher ranks
the truth first, and the chemical residual across *the current DreaMS error
boundary* (truth versus DreaMS's winning negative) is positive and strictly
larger than both candidate-swapped and peak-permuted controls.  Thus the target
is an existing model error, not a generic ICEBERG ranking relation.  Every
other query is an exact zero chemical action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_iceberg_corrective_residual_ledger_v2",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    return parser.parse_args()


def formula_folds(formulas: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little"
            )
            % 5
            for value in np.asarray(formulas).astype(str)
        ],
        dtype=np.int16,
    )


def centered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - np.mean(values)).astype(np.float32)


def teacher_margin(score: np.ndarray, label: np.ndarray) -> float:
    positive = np.flatnonzero(label == 1)
    negative = np.flatnonzero(label == 0)
    if len(positive) != 1 or not len(negative):
        raise RuntimeError("candidate block lacks one truth and at least one negative")
    return float(score[int(positive[0])] - np.max(score[negative]))


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    teacher_report = json.loads(
        (args.teacher_dir / "report.json").read_text(encoding="utf-8")
    )
    if (
        teacher_report.get("status") != "PASS"
        or teacher_report.get("scope", {}).get("teacher_only") is not True
    ):
        raise RuntimeError("ICEBERG teacher-specificity gate did not pass")
    with np.load(args.graph_dir / "graph.npz", allow_pickle=True) as graph:
        query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
        features = np.asarray(graph["features"][:, 0], dtype=np.float32)
        molecule_label = np.asarray(graph["molecule_label"], dtype=np.int8)
        query_row = np.asarray(graph["query_row"], dtype=np.int64)
        query_formula = np.asarray(graph["query_formula"]).astype(str)
        molecule_ik14 = np.asarray(graph["molecule_ik14"]).astype(str)
        molecule_formula = np.asarray(graph["molecule_formula"]).astype(str)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    teacher_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    with np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True) as source:
        official_rank = np.asarray(source["official_rank"], dtype=np.int64)
        correct_distance = np.asarray(source["correct_score"], dtype=np.float32)
        swapped_distance = np.asarray(source["candidate_swapped_score"], dtype=np.float32)
        peak_distance = np.asarray(source["peak_permuted_score"], dtype=np.float32)
        correct_rank = np.asarray(source["correct_rank"], dtype=np.int64)
    if len(selected) != len(official_rank) or len(teacher_ptr) != len(selected) + 1:
        raise RuntimeError("ICEBERG query ledgers are misaligned")
    if int(teacher_ptr[-1]) != len(correct_distance):
        raise RuntimeError("ICEBERG candidate score ledger is incomplete")

    n_candidates = int(query_ptr[-1])
    correct_residual = np.zeros(n_candidates, dtype=np.float32)
    swapped_residual = np.zeros(n_candidates, dtype=np.float32)
    peak_residual = np.zeros(n_candidates, dtype=np.float32)
    active = np.zeros(len(query_ptr) - 1, dtype=bool)
    correct_margin = np.full(len(selected), np.nan, dtype=np.float32)
    swapped_margin = np.full(len(selected), np.nan, dtype=np.float32)
    peak_margin = np.full(len(selected), np.nan, dtype=np.float32)
    causal_advantage = np.full(len(selected), np.nan, dtype=np.float32)
    graph_official_margin = np.full(len(selected), np.nan, dtype=np.float32)
    correct_boundary_delta = np.full(len(selected), np.nan, dtype=np.float32)
    swapped_boundary_delta = np.full(len(selected), np.nan, dtype=np.float32)
    peak_boundary_delta = np.full(len(selected), np.nan, dtype=np.float32)
    current_winning_negative = np.full(len(selected), -1, dtype=np.int32)

    for position, query in enumerate(selected):
        left, right = map(int, query_ptr[int(query) : int(query) + 2])
        tleft, tright = map(int, teacher_ptr[position : position + 2])
        if right - left != tright - tleft:
            raise RuntimeError(f"candidate count mismatch at selected query {position}")
        label = molecule_label[left:right]
        # ICEBERG values are distances, so lower is better.  Negation yields a
        # score whose larger value has the same candidate preference.
        correct_score = -correct_distance[tleft:tright].astype(np.float64)
        swapped_score = -swapped_distance[tleft:tright].astype(np.float64)
        peak_score = -peak_distance[tleft:tright].astype(np.float64)
        correct_margin[position] = teacher_margin(correct_score, label)
        swapped_margin[position] = teacher_margin(swapped_score, label)
        peak_margin[position] = teacher_margin(peak_score, label)
        official_score = np.asarray(
            [
                np.max(features[int(molecule_ptr[m]) : int(molecule_ptr[m + 1])])
                for m in range(left, right)
            ],
            dtype=np.float64,
        )
        graph_official_margin[position] = teacher_margin(official_score, label)
        positive = int(np.flatnonzero(label == 1)[0])
        negative = np.flatnonzero(label == 0)
        winning_negative = int(negative[np.argmax(official_score[negative])])
        current_winning_negative[position] = winning_negative
        correct_boundary_delta[position] = float(
            correct_score[positive] - correct_score[winning_negative]
        )
        swapped_boundary_delta[position] = float(
            swapped_score[positive] - swapped_score[winning_negative]
        )
        peak_boundary_delta[position] = float(
            peak_score[positive] - peak_score[winning_negative]
        )
        causal_advantage[position] = min(
            correct_boundary_delta[position] - swapped_boundary_delta[position],
            correct_boundary_delta[position] - peak_boundary_delta[position],
        )
        is_corrective = bool(
            graph_official_margin[position] <= 0
            and official_rank[position] > 1
            and correct_rank[position] == 1
            and correct_boundary_delta[position] > 0
            and causal_advantage[position] > 0
        )
        if not is_corrective:
            continue
        active[int(query)] = True
        correct_residual[left:right] = centered(correct_score)
        swapped_residual[left:right] = centered(swapped_score)
        peak_residual[left:right] = centered(peak_score)

    inactive = np.repeat(~active, np.diff(query_ptr))
    if not (
        np.all(correct_residual[inactive] == 0)
        and np.all(swapped_residual[inactive] == 0)
        and np.all(peak_residual[inactive] == 0)
    ):
        raise RuntimeError("noncorrective ICEBERG action is not an exact no-op")
    folds = formula_folds(query_formula, args.fold_seed)
    strict = np.flatnonzero(active[selected])
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "ledger.npz",
        query_ptr=query_ptr,
        query_row=query_row,
        query_formula=query_formula,
        query_formula_fold=folds,
        molecule_ik14=molecule_ik14,
        molecule_formula=molecule_formula,
        active_query=active,
        centered_residual=correct_residual,
        structure_swapped_centered_residual=swapped_residual,
        peak_permuted_centered_residual=peak_residual,
        selected_teacher_query=selected,
        selected_official_rank=official_rank,
        selected_correct_teacher_rank=correct_rank,
        selected_graph_official_margin=graph_official_margin,
        selected_correct_margin=correct_margin,
        selected_structure_swapped_margin=swapped_margin,
        selected_peak_permuted_margin=peak_margin,
        selected_current_winning_negative=current_winning_negative,
        selected_correct_boundary_delta=correct_boundary_delta,
        selected_structure_swapped_boundary_delta=swapped_boundary_delta,
        selected_peak_permuted_boundary_delta=peak_boundary_delta,
        selected_causal_advantage=causal_advantage,
    )
    report = {
        "status": "CHEMAWARE_ICEBERG_STRICT_CORRECTIVE_LEDGER_COMPLETE",
        "formal_training_authorized": False,
        "teacher_action": (
            "candidate-centred negative ICEBERG distance, routed on the current "
            "DreaMS truth-versus-winning-negative error boundary"
        ),
        "teacher_selection_reads": [
            "official error status",
            "identity label for training-only corrective routing",
            "current DreaMS winning negative",
            "correct and matched-control residuals on that fixed boundary",
        ],
        "teacher_selection_does_not_read": [
            "DreaMS input Jacobian",
            "synthetic-action embedding",
            "held-formula evaluation outcome",
        ],
        "teacher_queries": int(len(selected)),
        "official_errors": int(np.sum(official_rank > 1)),
        "correct_teacher_rescues": int(np.sum((official_rank > 1) & (correct_rank == 1))),
        "strict_corrective_actions": int(np.sum(active)),
        "strict_corrective_formula_clusters": int(
            len(np.unique(query_formula[selected[strict]]))
        ),
        "rejected_rescues_not_corrective_on_current_boundary_or_controls": int(
            np.sum((official_rank > 1) & (correct_rank == 1)) - np.sum(active)
        ),
        "route_contract": {
            "official_graph_margin_nonpositive": True,
            "correct_teacher_full_candidate_rank_one": True,
            "correct_current_boundary_delta_positive": True,
            "correct_current_boundary_delta_strictly_exceeds_each_control": True,
            "nonselected_queries_exact_zero": True,
        },
        "causal_advantage_quantiles": {
            str(q): float(np.quantile(causal_advantage[strict], q))
            for q in (0.0, 0.5, 0.9, 1.0)
        },
        "active_by_formula_fold": {
            str(fold): int(np.sum(active & (folds == fold))) for fold in range(5)
        },
        "noncorrective_all_arms_exact_zero": True,
        "matched_membership_across_arms": True,
        "next_gate": (
            "candidate-residual dose audit; shared-embedding training remains blocked "
            "until a correct arm beats both controls under the frozen ledger"
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

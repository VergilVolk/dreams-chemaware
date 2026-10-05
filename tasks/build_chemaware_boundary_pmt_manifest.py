"""Compile a passed B1 action atlas into a fail-closed clean-margin ledger.

This command performs no model training.  It selects current clean retrieval
boundaries whose correct chemical action strictly beats the unchanged input
and all three capacity-matched controls.  The resulting non-negative action
advantage is a target increment for B-PMT; controls receive no optimizer
gradient and a formula-disjoint dose derangement is saved as a separate arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_boundary_pmt_core import (
    active_transfer_weights,
    margin_bin_formula_derangement_indices,
    strict_action_advantage,
)
from chemaware_retrieval_graph import RetrievalGraph
from noise_final_core import sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-margin", type=float, default=0.05)
    parser.add_argument("--advantage-cap", type=float, default=0.05)
    parser.add_argument("--minimum-train-queries", type=int, default=20)
    parser.add_argument("--minimum-train-formulas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def boundary_reference_rows(graph: RetrievalGraph, query: int) -> tuple[int, int]:
    pair_slice, rows, ptr, _ = graph.query_block(query)
    pair_scores = graph.features[pair_slice, graph.dreams_column]
    molecule_score = np.asarray(
        [np.max(pair_scores[left:right]) for left, right in zip(ptr[:-1], ptr[1:])]
    )
    positive_pair = int(np.argmax(pair_scores[ptr[0] : ptr[1]])) + int(ptr[0])
    negative_molecule = 1 + int(np.argmax(molecule_score[1:]))
    negative_pair = (
        int(np.argmax(pair_scores[ptr[negative_molecule] : ptr[negative_molecule + 1]]))
        + int(ptr[negative_molecule])
    )
    return int(rows[positive_pair]), int(rows[negative_pair])


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        not 0 <= args.activation_margin <= 0.1
        or args.advantage_cap <= 0
        or args.minimum_train_queries < 20
        or args.minimum_train_formulas < 10
    ):
        raise ValueError("B-PMT admission thresholds were weakened")
    required = (
        args.action_dir / "report.json",
        args.action_dir / "screen.npz",
        args.graph,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"B-PMT inputs missing: {missing}")
    report = json.loads((args.action_dir / "report.json").read_text(encoding="utf-8"))
    screen_path = args.action_dir / "screen.npz"
    if report.get("provenance", {}).get("screen_sha256") != sha256_file(screen_path):
        raise RuntimeError("B1 report and screen provenance differ")
    with np.load(screen_path) as source:
        screen = {key: source[key] for key in source.files}
    selected_setting = int(screen["selected_setting"][0])
    authorized_source = bool(
        report.get("status") == "CHEMAWARE_BOUNDARY_CONSENSUS_ACTION_CONFIRMATION_PASS"
        and report.get("pass_to_action_transfer_audit") is True
        and report.get("discovery", {}).get("selected_setting_was_qualified") is True
        and selected_setting == int(report["discovery"]["selected_setting_id"])
    )
    args.output.mkdir(parents=True, exist_ok=False)
    if not authorized_source:
        body = {
            "status": "CHEMAWARE_BOUNDARY_PMT_SOURCE_NOT_AUTHORIZED",
            "training_authorized": False,
            "reason": "B1 same-boundary action did not pass discovery and confirmation",
            "provenance": {
                "action_report_sha256": sha256_file(args.action_dir / "report.json"),
                "action_screen_sha256": sha256_file(screen_path),
                "graph_sha256": sha256_file(args.graph),
            },
        }
        (args.output / "report.json").write_text(json.dumps(body, indent=2) + "\n")
        print(json.dumps(body, indent=2))
        return

    selected_query = screen["selected_query"].astype(np.int64)
    formula = screen["formula"].astype(str)
    fold = screen["formula_fold"].astype(np.int16)
    old_margin = screen["old_margin"].astype(np.float32)
    margins = screen["margins"].astype(np.float32)
    action_count = screen["action_count"].astype(np.int16)
    if margins.shape[1] != 4 or margins.shape[2] != len(selected_query):
        raise RuntimeError("B1 four-arm screen is misaligned")
    correct = margins[selected_setting, 0]
    controls = margins[selected_setting, 1:]
    advantage = strict_action_advantage(old_margin, correct, controls)
    weights = active_transfer_weights(
        old_margin,
        advantage,
        action_count[selected_setting],
        activation_margin=args.activation_margin,
        advantage_cap=args.advantage_cap,
    )
    train_scope = np.isin(fold, (0, 1, 2))
    active = np.flatnonzero(train_scope & (weights > 0))
    training_authorized = bool(
        len(active) >= args.minimum_train_queries
        and len(np.unique(formula[active])) >= args.minimum_train_formulas
    )
    control_source = (
        margin_bin_formula_derangement_indices(
            old_margin[active], formula[active], seed=args.seed
        )
        if training_authorized
        else np.empty(0, dtype=np.int64)
    )
    control_dose = (
        advantage[active][control_source].astype(np.float32)
        if training_authorized else np.empty(0, dtype=np.float32)
    )
    graph = RetrievalGraph(args.graph)
    positive_rows, negative_rows = [], []
    for position in active:
        positive, negative = boundary_reference_rows(
            graph, int(selected_query[position])
        )
        positive_rows.append(positive)
        negative_rows.append(negative)
    np.savez_compressed(
        args.output / "transfer_manifest.npz",
        screen_position=active,
        selected_query=selected_query[active],
        query_row=graph.query_row[selected_query[active]],
        positive_reference_row=np.asarray(positive_rows, dtype=np.int64),
        negative_reference_row=np.asarray(negative_rows, dtype=np.int64),
        formula=formula[active],
        formula_fold=fold[active],
        official_margin=old_margin[active],
        strict_action_advantage=advantage[active],
        corrective_weight=weights[active],
        matched_formula_deranged_advantage=control_dose,
        matched_formula_deranged_source_index=control_source,
        matched_formula_deranged_source_formula=formula[active][control_source],
    )
    body = {
        "status": (
            "CHEMAWARE_BOUNDARY_PMT_MANIFEST_ADMITTED"
            if training_authorized
            else "CHEMAWARE_BOUNDARY_PMT_ACTION_COVERAGE_FAIL"
        ),
        "training_authorized": training_authorized,
        "method": {
            "name": "boundary_paired_margin_transfer",
            "strict_advantage": "min(correct-old, correct-candidate_control, correct-peak_control, correct-direction_control) clipped at zero",
            "active_gate": "official margin <= activation margin and action count > 0",
            "clean_target": "m_initial + alpha * clip(strict_advantage, 0, advantage_cap)",
            "alpha_arms": [0.0, 0.25, 0.5],
            "matched_control": "same membership and dose multiset, formula-disjoint hardness-matched derangement",
            "control_gradient_subtraction": False,
            "nonactive_corrective_weight": 0.0,
            "candidate_input_at_inference": False,
        },
        "counts": {
            "source_queries": int(len(selected_query)),
            "strict_positive_active_train_queries": int(len(active)),
            "strict_positive_active_train_formulas": int(len(np.unique(formula[active]))),
            "embedding_evaluation_queries_untouched": int(np.sum(fold == 3)),
            "reserve_queries_untouched": int(np.sum(fold == 4)),
        },
        "thresholds": {
            "activation_margin": args.activation_margin,
            "advantage_cap": args.advantage_cap,
            "minimum_train_queries": args.minimum_train_queries,
            "minimum_train_formulas": args.minimum_train_formulas,
        },
        "provenance": {
            "action_report_sha256": sha256_file(args.action_dir / "report.json"),
            "action_screen_sha256": sha256_file(screen_path),
            "graph_sha256": sha256_file(args.graph),
        },
    }
    body["provenance"]["transfer_manifest_sha256"] = sha256_file(
        args.output / "transfer_manifest.npz"
    )
    (args.output / "report.json").write_text(json.dumps(body, indent=2) + "\n")
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()

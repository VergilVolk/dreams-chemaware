"""Run nested formula-group OOF for the frozen frontier G0 readout audit.

The preregistered primary is official DreaMS plus frozen peak-token local
interaction plus leave-query-out candidate consensus.  RAW multichannel and
raw-landmark profile arms are strong diagnostics.  A larger all-channel arm is
reported as exploratory and cannot by itself authorize G1.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_frontier_g0_core import (  # noqa: E402
    FrozenCandidateGraph,
    clustered_bootstrap,
    formula_folds,
    ranks_from_pair_scores,
    retrieval_summary,
    sha256_file,
    within_query_zscore,
)


RAW_CHANNELS = (
    "sqrt_cosine",
    "entropy_similarity",
    "top10_match_fraction",
    "intensity_coverage_min",
    "matched_peak_fraction_min",
    "neutral_loss_sqrt_cosine",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/validation/noise_frontier_g0_20260914/cache.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/noise_frontier_g0_20260914/report.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260914)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def ranks_for_weights(
    component: dict[str, np.ndarray],
    weights: tuple[float, float, float, float],
    graph: FrozenCandidateGraph,
    controls: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    raw_weight, token_weight, consensus_weight, landmark_weight = weights
    token_name = "token_control" if controls[0] else "token"
    consensus_name = "consensus_control" if controls[1] else "consensus"
    landmark_name = "landmark_control" if controls[2] else "landmark"
    score = np.array(component["official"], copy=True)
    score += raw_weight * component["raw"]
    score += token_weight * component[token_name]
    score += consensus_weight * component[consensus_name]
    score += landmark_weight * component[landmark_name]
    return ranks_from_pair_scores(score, graph.query_ptr, graph.molecule_ptr, graph.molecule_label)


def choose_configuration(
    grid: list[tuple[float, float, float, float]],
    ranks: list[np.ndarray],
    baseline: np.ndarray,
    train: np.ndarray,
    near: np.ndarray,
    identity: np.ndarray,
    formula: np.ndarray,
) -> int:
    def nested_weights(mask: np.ndarray) -> np.ndarray:
        """Identity-equal inside formula, then formula-equal across the split."""

        selected = np.flatnonzero(mask)
        selected_identity = identity[selected].astype(str)
        selected_formula = formula[selected].astype(str)
        identity_name, identity_inverse, query_count = np.unique(
            selected_identity, return_inverse=True, return_counts=True,
        )
        identity_formula = np.empty(len(identity_name), dtype=object)
        for index in range(len(identity_name)):
            formulas = set(selected_formula[identity_inverse == index])
            if len(formulas) != 1:
                raise RuntimeError("one query identity maps to multiple formulas")
            identity_formula[index] = next(iter(formulas))
        formula_name, formula_inverse, identity_count = np.unique(
            identity_formula.astype(str), return_inverse=True, return_counts=True,
        )
        local = 1.0 / query_count[identity_inverse]
        local /= identity_count[formula_inverse[identity_inverse]]
        local /= len(formula_name)
        output = np.zeros(len(mask), dtype=np.float64)
        output[selected] = local
        if not np.isclose(np.sum(output), 1.0):
            raise RuntimeError("hierarchical selection weights do not sum to one")
        return output

    best_index = -1
    best_key = None
    near_mask = train & near
    if not np.any(near_mask):
        raise RuntimeError("outer-training fold contains no near queries")
    train_weight = nested_weights(train)
    near_weight = nested_weights(near_mask)
    for index, (weights, candidate_rank) in enumerate(zip(grid, ranks, strict=True)):
        old_hit = baseline == 1
        new_hit = candidate_rank == 1
        recall_delta = new_hit.astype(float) - old_hit.astype(float)
        mrr_delta = 1.0 / candidate_rank - 1.0 / baseline
        risk = ((~old_hit) & new_hit).astype(float) - 2.0 * (old_hit & (~new_hit)).astype(float)
        balanced_recall = float(recall_delta @ train_weight)
        balanced_near = float(recall_delta @ near_weight)
        balanced_mrr = float(mrr_delta @ train_weight)
        balanced_risk = float(risk @ train_weight)
        safe = balanced_recall >= -1e-12 and balanced_near >= -1e-12 and balanced_mrr >= -1e-12
        key = (
            int(safe), balanced_risk, balanced_recall, balanced_near,
            balanced_mrr, -sum(weights),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    if best_index < 0:
        raise RuntimeError("configuration selection failed")
    return best_index


def evaluate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    formula: np.ndarray,
    identity: np.ndarray,
    near: np.ndarray,
    instrument: np.ndarray,
    ion_mode: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict:
    result = retrieval_summary(baseline, candidate)
    result["near"] = retrieval_summary(baseline, candidate, near)
    delta = (candidate == 1).astype(float) - (baseline == 1).astype(float)
    result["formula_cluster_bootstrap"] = clustered_bootstrap(delta, formula, bootstrap, seed)
    result["identity_cluster_bootstrap"] = clustered_bootstrap(
        delta, identity, bootstrap, seed + 7,
    )
    result["strata"] = {"instrument": {}, "ion_mode": {}}
    for axis, values in (("instrument", instrument), ("ion_mode", ion_mode)):
        for value in sorted(set(map(str, values))):
            mask = values.astype(str) == value
            if np.sum(mask) < 100:
                continue
            result["strata"][axis][value] = retrieval_summary(baseline, candidate, mask)
    return result


def paired_comparison(
    preferred: np.ndarray,
    control: np.ndarray,
    formula: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict:
    delta = (preferred == 1).astype(float) - (control == 1).astype(float)
    output = clustered_bootstrap(delta, formula, bootstrap, seed)
    output.update({
        "preferred_recall1": float(np.mean(preferred == 1)),
        "control_recall1": float(np.mean(control == 1)),
        "delta_recall1": float(np.mean(delta)),
    })
    return output


def main() -> None:
    args = arguments()
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite G0 result: {args.output}")
    if args.folds != 5 or args.bootstrap < 1000:
        raise ValueError("formal G0 requires five formula folds and >=1000 bootstrap draws")

    graph = FrozenCandidateGraph(args.cache)
    with np.load(args.cache, allow_pickle=True) as body:
        instrument = np.asarray(body["query_instrument"], dtype=str)
        ion_mode = np.asarray(body["query_ion_mode"], dtype=str)
    required = [
        "dreams_similarity", *RAW_CHANNELS,
        "g0_token_rule_mass", "g0_token_peak_permuted_control",
        "g0_candidate_consensus", "g0_random_consensus_control",
        "g0_landmark_profile", "g0_random_landmark_control",
    ]
    missing = [name for name in required if name not in graph.feature_names]
    if missing:
        raise RuntimeError(f"G0 cache lacks required channels: {missing}")
    if len(instrument) != graph.n_queries or len(ion_mode) != graph.n_queries:
        raise RuntimeError("G0 query strata are not aligned")

    index = {name: graph.feature_names.index(name) for name in required}
    query_pair_ptr = graph.molecule_ptr[graph.query_ptr]
    raw_direct = np.mean(
        graph.features[:, [index[name] for name in RAW_CHANNELS]], axis=1,
    )
    component = {
        "official": within_query_zscore(graph.features[:, index["dreams_similarity"]], query_pair_ptr),
        "raw": within_query_zscore(raw_direct, query_pair_ptr),
        "token": within_query_zscore(graph.features[:, index["g0_token_rule_mass"]], query_pair_ptr),
        "token_control": within_query_zscore(
            graph.features[:, index["g0_token_peak_permuted_control"]], query_pair_ptr,
        ),
        "consensus": within_query_zscore(
            graph.features[:, index["g0_candidate_consensus"]], query_pair_ptr,
        ),
        "consensus_control": within_query_zscore(
            graph.features[:, index["g0_random_consensus_control"]], query_pair_ptr,
        ),
        "landmark": within_query_zscore(
            graph.features[:, index["g0_landmark_profile"]], query_pair_ptr,
        ),
        "landmark_control": within_query_zscore(
            graph.features[:, index["g0_random_landmark_control"]], query_pair_ptr,
        ),
    }

    baseline = ranks_from_pair_scores(
        graph.features[:, index["dreams_similarity"]],
        graph.query_ptr,
        graph.molecule_ptr,
        graph.molecule_label,
    )
    raw_rank = ranks_from_pair_scores(
        raw_direct, graph.query_ptr, graph.molecule_ptr, graph.molecule_label,
    )
    fold = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    near = np.asarray(graph.query_has_near, dtype=bool)

    one_dimensional = [(0.0, value, 0.0, 0.0) for value in (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)]
    consensus_grid = [(0.0, 0.0, value, 0.0) for value in (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)]
    landmark_grid = [(0.0, 0.0, 0.0, value) for value in (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)]
    primary_grid = [
        (0.0, token, consensus, 0.0)
        for token, consensus in itertools.product((0.0, 0.05, 0.10, 0.20, 0.40, 0.80), repeat=2)
    ]
    exploratory_grid = [
        tuple(map(float, values))
        for values in itertools.product((0.0, 0.10, 0.25, 0.50), repeat=4)
    ]
    grids = {
        "token_late_interaction": one_dimensional,
        "candidate_consensus": consensus_grid,
        "raw_landmark_profile": landmark_grid,
        "token_consensus_primary": primary_grid,
        "frontier_full_exploratory": exploratory_grid,
    }
    precomputed: dict[str, list[np.ndarray]] = {}
    for name, grid in grids.items():
        print(f"[G0 rank grid] {name}: {len(grid)} configurations", flush=True)
        precomputed[name] = [ranks_for_weights(component, weights, graph) for weights in grid]

    oof = {name: np.empty(graph.n_queries, dtype=np.int16) for name in grids}
    selected_by_fold: dict[str, list[dict]] = {name: [] for name in grids}
    control_oof = {
        "token_permuted": np.empty(graph.n_queries, dtype=np.int16),
        "random_consensus": np.empty(graph.n_queries, dtype=np.int16),
        "both_primary_controls": np.empty(graph.n_queries, dtype=np.int16),
        "random_landmark": np.empty(graph.n_queries, dtype=np.int16),
    }
    for outer in range(args.folds):
        train = fold != outer
        held = fold == outer
        for name, grid in grids.items():
            selected_index = choose_configuration(
                grid, precomputed[name], baseline, train, near,
                graph.query_ik14, graph.query_formula,
            )
            oof[name][held] = precomputed[name][selected_index][held]
            selection = retrieval_summary(baseline, precomputed[name][selected_index], train)
            selected_by_fold[name].append({
                "outer_fold": outer,
                "weights": list(grid[selected_index]),
                "selection_queries": int(np.sum(train)),
                "selection": selection,
            })

        primary_weights = tuple(selected_by_fold["token_consensus_primary"][-1]["weights"])
        token_control = ranks_for_weights(component, primary_weights, graph, controls=(True, False, False))
        consensus_control = ranks_for_weights(component, primary_weights, graph, controls=(False, True, False))
        both_control = ranks_for_weights(component, primary_weights, graph, controls=(True, True, False))
        control_oof["token_permuted"][held] = token_control[held]
        control_oof["random_consensus"][held] = consensus_control[held]
        control_oof["both_primary_controls"][held] = both_control[held]

        landmark_weights = tuple(selected_by_fold["raw_landmark_profile"][-1]["weights"])
        landmark_control = ranks_for_weights(
            component, landmark_weights, graph, controls=(False, False, True),
        )
        control_oof["random_landmark"][held] = landmark_control[held]

    reports = {
        "official_global_cosine": retrieval_summary(baseline, baseline),
        "raw_multichannel_strong_baseline": evaluate(
            baseline, raw_rank, graph.query_formula, graph.query_ik14,
            near, instrument, ion_mode,
            args.bootstrap, args.seed + 11,
        ),
    }
    for offset, (name, rank) in enumerate(oof.items()):
        reports[name] = evaluate(
            baseline, rank, graph.query_formula, graph.query_ik14,
            near, instrument, ion_mode,
            args.bootstrap, args.seed + 101 + offset * 31,
        )
    control_reports = {}
    for offset, (name, rank) in enumerate(control_oof.items()):
        control_reports[name] = evaluate(
            baseline, rank, graph.query_formula, graph.query_ik14,
            near, instrument, ion_mode,
            args.bootstrap, args.seed + 401 + offset * 37,
        )

    primary = oof["token_consensus_primary"]
    control_comparisons = {
        name: paired_comparison(
            primary, rank, graph.query_formula, args.bootstrap,
            args.seed + 701 + offset * 41,
        )
        for offset, (name, rank) in enumerate(control_oof.items())
        if name != "random_landmark"
    }
    landmark_comparison = paired_comparison(
        oof["raw_landmark_profile"], control_oof["random_landmark"],
        graph.query_formula, args.bootstrap, args.seed + 901,
    )
    primary_report = reports["token_consensus_primary"]
    major_strata = [
        item
        for axis in primary_report["strata"].values()
        for item in axis.values()
        if item["n_queries"] >= 200
    ]
    control_specific = all(value["ci_low"] > 0 for value in control_comparisons.values())
    primary_gates = {
        "primary_overall_gain_ge_2pp": primary_report["delta_recall1"] >= 0.02,
        "primary_near_gain_ge_2pp": primary_report["near"]["delta_recall1"] >= 0.02,
        "primary_formula_ci_low_positive": primary_report["formula_cluster_bootstrap"]["ci_low"] > 0,
        "primary_corrected_gt_2x_introduced": primary_report["corrected"] > 2 * primary_report["introduced"],
        "major_instrument_and_ion_strata_nonnegative": bool(major_strata) and all(
            item["delta_recall1"] >= 0 for item in major_strata
        ),
        "token_and_consensus_controls_cannot_reproduce": control_specific,
        "all_queries_receive_exactly_one_outer_prediction": bool(
            all(np.all(rank >= 1) for rank in [*oof.values(), *control_oof.values()])
        ),
    }
    diagnostic_gates = {
        "landmark_profile_beats_matched_random": landmark_comparison["ci_low"] > 0,
    }
    gates = {
        **primary_gates,
        **diagnostic_gates,
        "pass_to_g1": bool(all(primary_gates.values())),
    }
    report = {
        "status": "noise_frontier_g0_representation_pass" if gates["pass_to_g1"] else "noise_frontier_g0_representation_fail",
        "formal": True,
        "scope": "frozen representation/readout sufficiency; no DreaMS parameter update",
        "primary_preregistered_arm": "token_consensus_primary",
        "queries": int(graph.n_queries),
        "identities": int(len(set(map(str, graph.query_ik14)))),
        "formulas": int(len(set(map(str, graph.query_formula)))),
        "near_queries": int(np.sum(near)),
        "formula_outer_fold_counts": {str(value): int(np.sum(fold == value)) for value in range(args.folds)},
        "results": reports,
        "negative_controls": control_reports,
        "paired_primary_minus_control": control_comparisons,
        "paired_landmark_minus_random_landmark": landmark_comparison,
        "selected_configuration_by_outer_fold": selected_by_fold,
        "gates": gates,
        "contracts": {
            "candidate_protocol_identical_across_arms": True,
            "formula_group_outer_OOF": True,
            "weights_selected_without_outer_fold": True,
            "pair_fusion_before_molecule_max": True,
            "ties_count_against_positive": True,
            "structure_used_for_posthoc_subset_selection": False,
            "P2b_used": False,
            "DreaMS_parameters_updated": False,
            "exploratory_full_arm_can_authorize_G1": False,
            "landmark_diagnostic_can_veto_primary_G1": False,
        },
        "provenance": {
            "cache_sha256": sha256_file(args.cache),
            "cache_report_sha256": sha256_file(args.cache.with_suffix(".json")),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "A pass shows that frozen global/local/consensus observations contain transferable ranking signal. "
            "It is not a better shared embedding; G1-G4 are still required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output.with_name("oof_ranks.npz"),
        formula_fold=fold,
        baseline_rank=baseline,
        raw_multichannel_rank=raw_rank,
        **{f"{name}_rank": rank for name, rank in oof.items()},
        **{f"control_{name}_rank": rank for name, rank in control_oof.items()},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

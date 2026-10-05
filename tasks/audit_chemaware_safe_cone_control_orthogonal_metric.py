"""Audit a control-orthogonal ChemAware residual inside a hard safety cone.

This is a frozen-embedding mechanism experiment.  ICEBERG score residuals are
training-only privileged information.  A low-rank shared tangent metric is
fitted on formula fold 0, selected on fold 1, refitted on folds 0+1, and checked
on fold 2.  The frozen map is then evaluated on the ordinary full-manifest fold
3 retrieval cohort; fold 4 is never evaluated.

Two changes distinguish this audit from the earlier observable tangent field:

1. the correct teacher target is residualized against candidate-swapped and
   peak-permuted targets before fitting;
2. every positive-versus-negative margin of an already-correct training query
   is a linear inequality, not a mean-squared safety example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from audit_chemaware_observable_tangent_metric import (  # noqa: E402
    evaluate_queries,
    formula_bootstrap,
    retrieval,
)
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from chemaware_observable_tangent_metric_core import (  # noqa: E402
    TangentFieldFit,
    apply_tangent_field,
    edge_design,
    margin_design,
    tangent_basis_from_edges,
    unit_rows,
)
from chemaware_safe_cone_metric_core import (  # noqa: E402
    SafeConeFit,
    control_orthogonal_residual,
    fit_safe_cone_ridge,
)
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    evaluate,
    identity_balanced_queries,
)


ARMS = (
    "control_orthogonal_correct",
    "raw_correct",
    "structure_swapped",
    "peak_permuted",
    "matched_hard_label",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path,
        default=ROOT / "data/validation/chemaware_iceberg_corrective_residual_ledger_v2/ledger.npz",
    )
    parser.add_argument(
        "--graph-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_safe_cone_control_orthogonal_metric_v1",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, nargs="+", default=(0.1, 1.0, 10.0))
    parser.add_argument(
        "--safety-preservation", type=float, nargs="+", default=(0.0, 0.5, 0.9, 1.0),
        help="Required fraction of every already-correct training margin after the linear update.",
    )
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--control-ridge", type=float, default=0.1)
    parser.add_argument("--hard-label-margin", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_fit(
    solve: SafeConeFit,
    basis: np.ndarray,
    training_prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> TangentFieldFit:
    rank = basis.shape[1]
    upper_a, upper_b = np.triu_indices(rank)
    matrix = np.zeros((rank, rank), dtype=np.float64)
    matrix[upper_a, upper_b] = solve.coefficient
    matrix[upper_b, upper_a] = solve.coefficient
    error = target - training_prediction
    return TangentFieldFit(
        basis=basis,
        matrices=matrix[None, :, :],
        column_scale=solve.column_scale,
        training_rms=float(np.sqrt(np.mean(error * error))),
        training_weighted_rms=float(np.sqrt(np.average(error * error, weights=weight))),
        effective_columns=int(np.sum(solve.column_scale > 1e-10)),
    )


def formula_balanced_pair_weights(
    pair_query: np.ndarray,
    selected: np.ndarray,
    formula: np.ndarray,
) -> np.ndarray:
    query = np.asarray(pair_query[selected], dtype=np.int64)
    values = np.asarray(formula).astype(str)[query]
    unique, inverse = np.unique(values, return_inverse=True)
    pair_count = np.bincount(query, minlength=len(formula))
    weight = 1.0 / np.maximum(pair_count[query], 1)
    formula_mass = np.bincount(inverse, weights=weight)
    weight /= formula_mass[inverse]
    weight *= len(weight) / np.sum(weight)
    if len(unique) == 0 or np.any(~np.isfinite(weight)) or np.any(weight <= 0):
        raise RuntimeError("formula-balanced action weights are invalid")
    return weight


def prepare_targets(
    ledger: np.lib.npyio.NpzFile,
    pair_positive: np.ndarray,
    pair_negative: np.ndarray,
    pair_query: np.ndarray,
    train_active_pair: np.ndarray,
    train_weight: np.ndarray,
    baseline_margin: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    residual = {
        "raw_correct": np.asarray(ledger["centered_residual"], dtype=np.float64),
        "structure_swapped": np.asarray(
            ledger["structure_swapped_centered_residual"], dtype=np.float64
        ),
        "peak_permuted": np.asarray(
            ledger["peak_permuted_centered_residual"], dtype=np.float64
        ),
    }
    pair = {
        arm: args.dose * (values[pair_positive] - values[pair_negative])
        for arm, values in residual.items()
    }
    active_index = np.flatnonzero(train_active_pair)
    orthogonal, coefficient, orthogonal_report = control_orthogonal_residual(
        pair["raw_correct"][active_index],
        np.stack((
            pair["structure_swapped"][active_index],
            pair["peak_permuted"][active_index],
        ), axis=1),
        train_weight,
        ridge=args.control_ridge,
    )
    orthogonal_full = (
        pair["raw_correct"]
        - coefficient[0] * pair["structure_swapped"]
        - coefficient[1] * pair["peak_permuted"]
    )
    required = np.maximum(0.0, -baseline_margin + args.hard_label_margin)
    correct_rms = float(np.sqrt(np.average(orthogonal * orthogonal, weights=train_weight)))
    label_rms = float(np.sqrt(np.average(
        required[active_index] * required[active_index], weights=train_weight,
    )))
    label_scale = correct_rms / label_rms if label_rms else 0.0
    pair["control_orthogonal_correct"] = orthogonal_full
    pair["matched_hard_label"] = required * label_scale
    return {arm: pair[arm] for arm in ARMS}, {
        "control_coefficients": coefficient.astype(float).tolist(),
        "control_residualization": orthogonal_report,
        "matched_hard_label_scale": label_scale,
        "matched_action_weighted_rms": correct_rms,
    }


def fit_one(
    pair_design: np.ndarray,
    target: np.ndarray,
    pair_query: np.ndarray,
    active: np.ndarray,
    safe: np.ndarray,
    formula: np.ndarray,
    train_query: np.ndarray,
    baseline_margin: np.ndarray,
    basis: np.ndarray,
    *,
    ridge: float,
    safety_preservation: float,
    max_iterations: int,
) -> tuple[TangentFieldFit, dict[str, object]]:
    action_mask = train_query[pair_query] & active[pair_query]
    safety_mask = train_query[pair_query] & safe[pair_query]
    weight = formula_balanced_pair_weights(pair_query, action_mask, formula)
    lower = -(1.0 - safety_preservation) * baseline_margin[safety_mask]
    solve = fit_safe_cone_ridge(
        pair_design[action_mask], target[action_mask], weight,
        pair_design[safety_mask], lower,
        ridge=ridge, max_iterations=max_iterations,
    )
    prediction = pair_design[action_mask] @ solve.coefficient
    fit = unpack_fit(
        solve, basis, prediction, target[action_mask], weight,
    )
    return fit, {
        "action_pairs": int(np.sum(action_mask)),
        "safety_pairs": int(np.sum(safety_mask)),
        "action_formula_clusters": int(len(np.unique(formula[pair_query[action_mask]]))),
        "control": {
            "converged": solve.converged,
            "iterations": solve.iterations,
            "objective": solve.objective,
            "maximum_constraint_violation": solve.maximum_constraint_violation,
            "active_constraints": solve.active_constraints,
            "unconstrained_violations": solve.unconstrained_violations,
            "minimum_linear_safety_margin_after": float(np.min(
                baseline_margin[safety_mask] + pair_design[safety_mask] @ solve.coefficient
            )),
        },
        "training_rms": fit.training_rms,
        "training_weighted_rms": fit.training_weighted_rms,
    }


def subset_report(
    baseline_rank: np.ndarray,
    ranks: np.ndarray,
    active: np.ndarray,
    safe: np.ndarray,
    selected: np.ndarray,
) -> dict[str, object]:
    return {
        "all": retrieval(baseline_rank[selected], ranks[selected]),
        "strict_corrective": retrieval(
            baseline_rank[selected & active], ranks[selected & active]
        ),
        "baseline_correct_safety": retrieval(
            baseline_rank[selected & safe], ranks[selected & safe]
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rank <= 0 or not 0 < args.dose <= 1 or args.control_ridge < 0:
        raise ValueError("invalid rank, dose, or control ridge")
    if any(value <= 0 for value in args.ridge):
        raise ValueError("ridge values must be positive")
    if any(value < 0 or value > 1 for value in args.safety_preservation):
        raise ValueError("safety preservation must lie in [0, 1]")
    if args.smoke:
        args.rank = min(args.rank, 8)
        args.ridge = [1.0]
        args.safety_preservation = [0.9]
        args.bootstrap_draws = min(args.bootstrap_draws, 200)

    graph_path = args.graph_dir / "graph.npz"
    required = [
        args.ledger, graph_path, args.manifest,
        args.token_dir / "rows.npy", args.token_dir / "official_embeddings_f32.npy",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    ledger = np.load(args.ledger, allow_pickle=False)
    graph = np.load(graph_path, allow_pickle=True)
    if not np.array_equal(ledger["query_row"], graph["query_row"]):
        raise RuntimeError("ledger and graph query rows drifted")

    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    official_cache = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    graph_rows = np.unique(np.concatenate((
        graph["query_row"], graph["pair_candidate_row"],
    ))).astype(np.int64)
    positions = np.asarray([cache_position[int(row)] for row in graph_rows], dtype=np.int64)
    embeddings = unit_rows(np.asarray(official_cache[positions], dtype=np.float64))
    node_position = {int(row): index for index, row in enumerate(graph_rows)}
    query_nodes = np.asarray(
        [node_position[int(row)] for row in graph["query_row"]], dtype=np.int64
    )
    reference_nodes = np.asarray(
        [node_position[int(row)] for row in graph["pair_candidate_row"]], dtype=np.int64
    )
    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(graph["molecule_label"], dtype=bool)
    molecule_query = np.repeat(np.arange(len(query_nodes)), np.diff(query_ptr))

    molecule_best_ref = np.empty(len(labels), dtype=np.int64)
    molecule_score = np.empty(len(labels), dtype=np.float64)
    for molecule, query in enumerate(molecule_query):
        left, right = molecule_ptr[molecule : molecule + 2]
        values = embeddings[reference_nodes[left:right]] @ embeddings[query_nodes[query]]
        winner = int(np.argmax(values))
        molecule_best_ref[molecule] = reference_nodes[int(left) + winner]
        molecule_score[molecule] = float(values[winner])

    baseline_rank = np.empty(len(query_nodes), dtype=np.int64)
    pair_query: list[int] = []
    pair_positive: list[int] = []
    pair_negative: list[int] = []
    for query in range(len(query_nodes)):
        left, right = query_ptr[query : query + 2]
        local_label = labels[left:right]
        positive = int(left) + int(np.flatnonzero(local_label)[0])
        positive_score = molecule_score[positive]
        baseline_rank[query] = 1 + int(np.sum(molecule_score[left:right][~local_label] >= positive_score))
        for negative in np.flatnonzero(~local_label) + int(left):
            pair_query.append(query)
            pair_positive.append(positive)
            pair_negative.append(int(negative))
    pair_query_array = np.asarray(pair_query, dtype=np.int64)
    pair_positive_array = np.asarray(pair_positive, dtype=np.int64)
    pair_negative_array = np.asarray(pair_negative, dtype=np.int64)
    baseline_pair_margin = (
        molecule_score[pair_positive_array] - molecule_score[pair_negative_array]
    )
    active = np.asarray(ledger["active_query"], dtype=bool)
    safe = baseline_rank == 1
    formula = np.asarray(ledger["query_formula"]).astype(str)
    formula_fold = np.asarray(ledger["query_formula_fold"], dtype=np.int64)
    if not np.array_equal(np.unique(formula_fold), [0, 1, 2]):
        raise RuntimeError("expected frozen action folds 0, 1, 2")

    def split_design(train_folds: tuple[int, ...], basis_seed: int):
        train_query = np.isin(formula_fold, train_folds)
        allowed = train_query & (active | safe)
        train_molecules = allowed[molecule_query]
        basis = tangent_basis_from_edges(
            embeddings,
            query_nodes[molecule_query[train_molecules]],
            molecule_best_ref[train_molecules],
            rank=args.rank, seed=basis_seed,
        )
        edge = edge_design(
            embeddings, basis, np.ones((len(embeddings), 1), dtype=np.float64),
            query_nodes[molecule_query], molecule_best_ref,
        )
        return train_query, basis, margin_design(
            edge, pair_positive_array, pair_negative_array,
        )

    # Architecture selection: fold 0 -> fold 1.
    train0, basis0, design0 = split_design((0,), args.seed)
    active_pair0 = train0[pair_query_array] & active[pair_query_array]
    weight0 = formula_balanced_pair_weights(pair_query_array, active_pair0, formula)
    target0, target_report0 = prepare_targets(
        ledger, pair_positive_array, pair_negative_array, pair_query_array,
        active_pair0, weight0, baseline_pair_margin, args,
    )
    fold1 = formula_fold == 1
    grid = []
    for ridge in args.ridge:
        for preservation in args.safety_preservation:
            fit, fit_report = fit_one(
                design0, target0["control_orthogonal_correct"], pair_query_array,
                active, safe, formula, train0, baseline_pair_margin, basis0,
                ridge=float(ridge), safety_preservation=float(preservation),
                max_iterations=args.max_iterations,
            )
            updated, adapter = apply_tangent_field(
                embeddings, np.ones((len(embeddings), 1), dtype=np.float64), fit,
            )
            ranks, _ = evaluate_queries(
                updated, query_nodes, query_ptr, molecule_ptr, labels,
                reference_nodes, np.flatnonzero(fold1),
            )
            all_ranks = baseline_rank.copy(); all_ranks[fold1] = ranks
            held = subset_report(baseline_rank, all_ranks, active, safe, fold1)
            grid.append({
                "ridge": float(ridge),
                "safety_preservation": float(preservation),
                "fit": fit_report,
                "adapter": adapter,
                "fold1": held,
                "_fit": fit,
            })
            print(
                f"selection ridge={ridge:g} safety={preservation:g} "
                f"net={held['all']['corrected_at_1'] - held['all']['introduced_at_1']} "
                f"risk={held['all']['risk_utility_at_1']}", flush=True,
            )
    selected = max(
        grid,
        key=lambda item: (
            int(item["fold1"]["all"]["risk_utility_at_1"]),
            -int(item["fold1"]["all"]["introduced_at_1"]),
            float(item["fold1"]["all"]["delta_mrr"]),
            float(item["safety_preservation"]),
            -float(item["ridge"]),
        ),
    )
    selected_setting = {
        "ridge": selected["ridge"],
        "safety_preservation": selected["safety_preservation"],
    }

    # Matched fold-1 arm comparison at the selected setting.
    fold1_arms = {}
    for arm in ARMS:
        fit, fit_report = fit_one(
            design0, target0[arm], pair_query_array, active, safe, formula,
            train0, baseline_pair_margin, basis0,
            ridge=selected_setting["ridge"],
            safety_preservation=selected_setting["safety_preservation"],
            max_iterations=args.max_iterations,
        )
        updated, adapter = apply_tangent_field(
            embeddings, np.ones((len(embeddings), 1), dtype=np.float64), fit,
        )
        ranks, _ = evaluate_queries(
            updated, query_nodes, query_ptr, molecule_ptr, labels,
            reference_nodes, np.flatnonzero(fold1),
        )
        all_ranks = baseline_rank.copy(); all_ranks[fold1] = ranks
        fold1_arms[arm] = {
            "fit": fit_report, "adapter": adapter,
            "retrieval": subset_report(baseline_rank, all_ranks, active, safe, fold1),
        }

    # Locked confirmation: refit folds 0+1 -> fold 2.
    train01, basis01, design01 = split_design((0, 1), args.seed + 1)
    active_pair01 = train01[pair_query_array] & active[pair_query_array]
    weight01 = formula_balanced_pair_weights(pair_query_array, active_pair01, formula)
    target01, target_report01 = prepare_targets(
        ledger, pair_positive_array, pair_negative_array, pair_query_array,
        active_pair01, weight01, baseline_pair_margin, args,
    )
    fold2 = formula_fold == 2
    confirmation = {}
    final_fits: dict[str, TangentFieldFit] = {}
    for arm in ARMS:
        fit, fit_report = fit_one(
            design01, target01[arm], pair_query_array, active, safe, formula,
            train01, baseline_pair_margin, basis01,
            ridge=selected_setting["ridge"],
            safety_preservation=selected_setting["safety_preservation"],
            max_iterations=args.max_iterations,
        )
        final_fits[arm] = fit
        updated, adapter = apply_tangent_field(
            embeddings, np.ones((len(embeddings), 1), dtype=np.float64), fit,
        )
        ranks, _ = evaluate_queries(
            updated, query_nodes, query_ptr, molecule_ptr, labels,
            reference_nodes, np.flatnonzero(fold2),
        )
        all_ranks = baseline_rank.copy(); all_ranks[fold2] = ranks
        confirmation[arm] = {
            "fit": fit_report, "adapter": adapter,
            "retrieval": subset_report(baseline_rank, all_ranks, active, safe, fold2),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula[fold2], baseline_rank[fold2], all_ranks[fold2],
                draws=args.bootstrap_draws, seed=args.seed + 100 + ARMS.index(arm),
            ),
        }

    # Candidate-free full-manifest evaluation on fold 3.  Fold 4 stays sealed.
    with np.load(args.manifest, allow_pickle=True) as loaded:
        manifest = {key: loaded[key] for key in loaded.files}
    manifest_fold = stable_formula_folds(
        manifest["query_formula"], 5, args.fold_seed,
    )
    inner = identity_balanced_queries(
        np.flatnonzero(manifest_fold == 3), manifest["query_ik14"],
        # Match the existing candidate-distribution and formula-policy ledger.
        np.random.default_rng(args.fold_seed + 19), 0,
    )
    full_official = unit_rows(np.asarray(official_cache, dtype=np.float64))
    full_manifest = {}
    for arm, fit in final_fits.items():
        adapted, adapter = apply_tangent_field(
            full_official, np.ones((len(full_official), 1), dtype=np.float64), fit,
        )
        outcome = evaluate(manifest, inner, full_official, adapted, cache_position)
        full_manifest[arm] = {
            "summary": outcome["summary"],
            "adapter": adapter,
        }

    correct = confirmation["control_orthogonal_correct"]["retrieval"]["all"]
    correct_inner = full_manifest["control_orthogonal_correct"]["summary"]
    gates = {
        "fold2_correct_risk_positive": correct["risk_utility_at_1"] > 0,
        "fold2_correct_beats_structure_swapped": (
            correct["delta_recall1"]
            > confirmation["structure_swapped"]["retrieval"]["all"]["delta_recall1"]
        ),
        "fold2_correct_beats_peak_permuted": (
            correct["delta_recall1"]
            > confirmation["peak_permuted"]["retrieval"]["all"]["delta_recall1"]
        ),
        "fold2_correct_beats_matched_hard_label": (
            correct["delta_recall1"]
            > confirmation["matched_hard_label"]["retrieval"]["all"]["delta_recall1"]
        ),
        "fold3_full_manifest_risk_positive": (
            correct_inner["corrected"] - 2 * correct_inner["introduced"] > 0
        ),
        "fold3_full_manifest_correct_beats_all_controls": all(
            correct_inner["delta_recall1"] > full_manifest[arm]["summary"]["delta_recall1"]
            for arm in ARMS if arm != "control_orthogonal_correct"
        ),
    }
    report = {
        "status": (
            "CHEMAWARE_SAFE_CONE_CONTROL_ORTHOGONAL_PASS"
            if all(gates.values()) else "CHEMAWARE_SAFE_CONE_CONTROL_ORTHOGONAL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "claim_limit": (
            "Frozen-embedding development audit. Formula fold 1 selected the setting, "
            "fold 2 checks the teacher-defined action class, fold 3 checks ordinary retrieval, "
            "and fold 4 remains untouched."
        ),
        "method": {
            "shared_map": "normalize(z + (I-zzT) Q B QT z)",
            "control_orthogonal_target": "correct - projection(correct | swapped, peak-permuted)",
            "objective": "formula-balanced action ridge inside exact linear training-safety cone",
            "safety_constraint": "new positive-minus-each-negative margin >= selected fraction of baseline margin",
            "candidate_or_structure_input_at_deployment": False,
            "all_arms_matched": True,
        },
        "data": {
            "action_graph_queries": int(len(query_nodes)),
            "strict_corrective_queries": int(np.sum(active)),
            "baseline_correct_safety_queries": int(np.sum(safe)),
            "formula_fold_roles": {
                "architecture_train": 0,
                "architecture_selection": 1,
                "locked_action_confirmation": 2,
                "full_manifest_inner_evaluation": 3,
                "outer_untouched": 4,
            },
        },
        "grid": [
            {key: value for key, value in item.items() if key != "_fit"}
            for item in grid
        ],
        "selected_setting": selected_setting,
        "target_fit_fold0": target_report0,
        "fold1_matched_arms": fold1_arms,
        "target_refit_folds01": target_report01,
        "fold2_confirmation": confirmation,
        "fold3_full_manifest": full_manifest,
        "gates": gates,
        "provenance": {
            "ledger_sha256": sha256(args.ledger),
            "graph_sha256": sha256(graph_path),
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(
                args.token_dir / "official_embeddings_f32.npy"
            ),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_safe_cone_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "selected_setting": selected_setting,
        "fold2_correct": confirmation["control_orthogonal_correct"]["retrieval"],
        "fold3_correct": full_manifest["control_orthogonal_correct"]["summary"],
        "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

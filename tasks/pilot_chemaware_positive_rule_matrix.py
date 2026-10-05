"""Formula/domain-isolated pilot of a positive structure-observation rule matrix.

The pilot learns a small bilinear teacher from all existing parent predicates
and formula-specified observation channels.  Training residualizes both sides
inside molecular-formula/acquisition-domain strata, so formula or instrument
composition cannot create the association.  Negative coefficients are set to
zero: only observed peaks and positive structure support can act.

Fold 0 fits candidate regularization strengths, fold 1 selects one, and folds
0+1 refit the frozen matrix evaluated on fold 2.  Folds 3 and 4 are untouched.
This is a teacher-headroom test; it never updates DreaMS.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from chemaware_direct_action_core import formula_bootstrap


ROOT = Path(__file__).resolve().parents[1]
FORMULA_PATTERN = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT
        / "data/validation/chemaware_domain_conditioned_action_rules_local_runtime_assets_20260907_v1"
        / "rules/identity_domain_evidence.npz",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_positive_rule_matrix_pilot_v1",
    )
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=(1.0, 10.0, 100.0))
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def residual_cross_products(
    x: np.ndarray,
    y: np.ndarray,
    formula: np.ndarray,
    domain: np.ndarray,
    folds: np.ndarray,
    selected_folds: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    dict[str, tuple[np.ndarray, np.ndarray, int, int]],
]:
    """Accumulate within-formula/domain sufficient statistics."""
    p, o = x.shape[1], y.shape[1]
    xtx = np.zeros((p, p), dtype=np.float64)
    xty = np.zeros((p, o), dtype=np.float64)
    selected = np.flatnonzero(np.isin(folds, selected_folds))
    groups: dict[tuple[str, str], list[int]] = {}
    for position in selected:
        groups.setdefault((str(formula[position]), str(domain[position])), []).append(
            int(position)
        )
    used_units = 0
    used_strata = 0
    by_domain: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for positions in groups.values():
        if len(positions) < 2:
            continue
        index = np.asarray(positions, dtype=np.int64)
        gx = x[index].astype(np.float64)
        gy = y[index].astype(np.float64)
        gx -= np.mean(gx, axis=0, keepdims=True)
        gy -= np.mean(gy, axis=0, keepdims=True)
        if not np.any(gx) or not np.any(gy):
            continue
        xtx += gx.T @ gx
        xty += gx.T @ gy
        domain_value = str(domain[index[0]])
        if domain_value not in by_domain:
            by_domain[domain_value] = (
                np.zeros((p, p), dtype=np.float64),
                np.zeros((p, o), dtype=np.float64),
                0,
                0,
            )
        domain_xtx, domain_xty, domain_units, domain_strata = by_domain[domain_value]
        domain_xtx += gx.T @ gx
        domain_xty += gx.T @ gy
        by_domain[domain_value] = (
            domain_xtx,
            domain_xty,
            domain_units + len(index),
            domain_strata + 1,
        )
        used_units += len(index)
        used_strata += 1
    return xtx, xty, used_units, used_strata, by_domain


def fit_positive_matrix(xtx: np.ndarray, xty: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    regularized = xtx + float(ridge) * np.eye(len(xtx), dtype=np.float64)
    coefficient = np.linalg.solve(regularized, xty)
    # The action library authorizes positive observed support only.  Negative
    # associations and absent peaks receive an exact zero executable weight.
    return np.maximum(coefficient, 0.0).astype(np.float32)


def fit_hierarchical_domain_matrices(
    xtx: np.ndarray,
    xty: np.ndarray,
    by_domain: dict[str, tuple[np.ndarray, np.ndarray, int, int]],
    ridge: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit domain matrices shrunk toward a shared, formula-residualized map."""
    shared = fit_positive_matrix(xtx, xty, ridge)
    identity = np.eye(len(xtx), dtype=np.float64)
    matrices: dict[str, np.ndarray] = {}
    for domain, (domain_xtx, domain_xty, _units, _strata) in by_domain.items():
        # argmin ||X_d W-Y_d||^2 + ridge ||W-W_shared||^2
        coefficient = np.linalg.solve(
            domain_xtx + float(ridge) * identity,
            domain_xty + float(ridge) * shared,
        )
        matrices[domain] = np.maximum(coefficient, 0.0).astype(np.float32)
    return shared, matrices


def channel_permutation(channels: list[dict], selected: np.ndarray, seed: int) -> np.ndarray:
    """Permute semantics within fragment/loss type while preserving capacity."""
    rng = np.random.default_rng(seed)
    selected_channels = [channels[int(index)] for index in selected]
    kind = np.asarray([item["match_type"] for item in selected_channels])
    permutation = np.arange(len(selected), dtype=np.int64)
    for value in np.unique(kind):
        positions = np.flatnonzero(kind == value)
        if len(positions) > 1:
            permutation[positions] = positions[rng.permutation(len(positions))]
    if np.array_equal(permutation, np.arange(len(selected))):
        raise RuntimeError("peak-channel control did not permute any channel")
    return permutation


def identity_candidates(
    identity_formula: np.ndarray,
    identity_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for formula in np.unique(identity_formula):
        positions = np.flatnonzero((identity_formula == formula) & identity_valid)
        if len(positions) >= 2:
            result[str(formula)] = positions
    return result


def evaluate(
    shared_matrix: np.ndarray,
    domain_matrices: dict[str, np.ndarray],
    x_identity: np.ndarray,
    y_unit: np.ndarray,
    unit_identity: np.ndarray,
    unit_formula: np.ndarray,
    unit_domain: np.ndarray,
    unit_fold: np.ndarray,
    fold: int,
    candidate_by_formula: dict[str, np.ndarray],
    *,
    structure_swap: bool,
    seed: int,
) -> dict:
    ranks = []
    formulas = []
    pairwise = []
    rng = np.random.default_rng(seed)
    for unit in np.flatnonzero(unit_fold == fold):
        formula = str(unit_formula[unit])
        candidates = candidate_by_formula.get(formula)
        if candidates is None:
            continue
        truth = int(unit_identity[unit])
        truth_position = np.flatnonzero(candidates == truth)
        if len(truth_position) != 1:
            continue
        candidate_x = x_identity[candidates]
        if structure_swap:
            # A label-blind cyclic shift preserves the exact formula-specific
            # candidate structure multiset and score capacity.
            offset = int(rng.integers(1, len(candidates)))
            candidate_x = np.roll(candidate_x, offset, axis=0)
        matrix = domain_matrices.get(str(unit_domain[unit]), shared_matrix)
        query_latent = matrix @ y_unit[unit]
        scores = candidate_x @ query_latent
        positive = int(truth_position[0])
        negative = np.delete(scores, positive)
        rank = 1 + int(np.sum(negative >= scores[positive]))
        ranks.append(rank)
        formulas.append(formula)
        # AUC convention: a tied positive/negative pair contributes 0.5.  The
        # strict retrieval rank above still counts every tie against Top-1.
        pairwise.append(
            float(
                np.mean(scores[positive] > negative)
                + 0.5 * np.mean(scores[positive] == negative)
            )
        )
    if not ranks:
        raise RuntimeError(f"fold {fold} has no evaluable rule-matrix queries")
    ranks = np.asarray(ranks, dtype=np.int64)
    pairwise = np.asarray(pairwise, dtype=np.float64)
    return {
        "queries": int(len(ranks)),
        "formulas": int(len(np.unique(formulas))),
        "recall1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1 / ranks)),
        "pairwise_accuracy": float(np.mean(pairwise)),
        "rank": ranks,
        "formula": np.asarray(formulas),
    }


def public_metrics(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"rank", "formula"}}


def contrast(correct: dict, control: dict, seed: int, draws: int) -> dict:
    if not np.array_equal(correct["formula"], control["formula"]):
        raise RuntimeError("correct/control evaluation cohorts differ")
    delta = (correct["rank"] == 1).astype(float) - (control["rank"] == 1).astype(float)
    return formula_bootstrap(delta, correct["formula"], seed, draws)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.draws < 10_000:
        raise ValueError("bootstrap draws were weakened")
    ridge_grid = np.asarray(args.ridge_grid, dtype=np.float64)
    if np.any(~np.isfinite(ridge_grid)) or np.any(ridge_grid <= 0):
        raise ValueError("ridge grid must be positive and finite")

    channels = json.loads(args.observations.read_text(encoding="utf-8"))["channels"]
    selected_channel = np.flatnonzero(
        [
            bool(item.get("observed_species_formulae"))
            and bool(FORMULA_PATTERN.fullmatch(str(item["observed_species_formulae"][0])))
            for item in channels
        ]
    )
    with np.load(args.evidence, allow_pickle=False) as loaded:
        identity_formula = np.asarray(loaded["identity_formula"]).astype(str)
        x_identity = np.asarray(loaded["predicate_presence"], dtype=np.float32)
        unit_identity = np.asarray(loaded["unit_identity"], dtype=np.int64)
        unit_formula = np.asarray(loaded["unit_formula"]).astype(str)
        unit_domain = np.asarray(loaded["unit_domain"]).astype(str)
        unit_fold = np.asarray(loaded["unit_formula_fold"], dtype=np.int16)
        identity_fold = np.asarray(loaded["identity_formula_fold"], dtype=np.int16)
        identity_valid = np.zeros(len(identity_formula), dtype=bool)
        np.logical_or.at(
            identity_valid,
            unit_identity,
            np.asarray(loaded["valid_structure_unit"], dtype=bool),
        )
        y_unit = np.asarray(
            loaded["observation_prevalence"][:, selected_channel], dtype=np.float32
        )
    if not np.array_equal(identity_fold[unit_identity], unit_fold):
        raise RuntimeError("identity and unit formula folds disagree")
    candidates = identity_candidates(identity_formula, identity_valid)
    permutation = channel_permutation(channels, selected_channel, args.seed)

    validation_grid = []
    xtx0, xty0, train_units0, train_strata0, domain0 = residual_cross_products(
        x_identity[unit_identity],
        y_unit,
        unit_formula,
        unit_domain,
        unit_fold,
        (0,),
    )
    for index, ridge in enumerate(ridge_grid):
        correct_shared, correct_matrices = fit_hierarchical_domain_matrices(
            xtx0, xty0, domain0, float(ridge)
        )
        permuted_domain0 = {
            domain: (values[0], values[1][:, permutation], values[2], values[3])
            for domain, values in domain0.items()
        }
        permuted_shared, permuted_matrices = fit_hierarchical_domain_matrices(
            xtx0, xty0[:, permutation], permuted_domain0, float(ridge)
        )
        correct = evaluate(
            correct_shared,
            correct_matrices,
            x_identity,
            y_unit,
            unit_identity,
            unit_formula,
            unit_domain,
            unit_fold,
            1,
            candidates,
            structure_swap=False,
            seed=args.seed + index,
        )
        structure = evaluate(
            correct_shared,
            correct_matrices,
            x_identity,
            y_unit,
            unit_identity,
            unit_formula,
            unit_domain,
            unit_fold,
            1,
            candidates,
            structure_swap=True,
            seed=args.seed + 100 + index,
        )
        peak = evaluate(
            permuted_shared,
            permuted_matrices,
            x_identity,
            y_unit,
            unit_identity,
            unit_formula,
            unit_domain,
            unit_fold,
            1,
            candidates,
            structure_swap=False,
            seed=args.seed + 200 + index,
        )
        validation_grid.append(
            {
                "ridge": float(ridge),
                "correct": public_metrics(correct),
                "structure_swapped": public_metrics(structure),
                "peak_permuted": public_metrics(peak),
                "selection_score": float(
                    min(
                        correct["pairwise_accuracy"] - structure["pairwise_accuracy"],
                        correct["pairwise_accuracy"] - peak["pairwise_accuracy"],
                    )
                ),
            }
        )
    selected = sorted(
        validation_grid, key=lambda row: (-row["selection_score"], row["ridge"])
    )[0]
    selected_ridge = float(selected["ridge"])

    xtx, xty, train_units, train_strata, domain_stats = residual_cross_products(
        x_identity[unit_identity],
        y_unit,
        unit_formula,
        unit_domain,
        unit_fold,
        (0, 1),
    )
    correct_shared, correct_matrices = fit_hierarchical_domain_matrices(
        xtx, xty, domain_stats, selected_ridge
    )
    permuted_domain_stats = {
        domain: (values[0], values[1][:, permutation], values[2], values[3])
        for domain, values in domain_stats.items()
    }
    peak_shared, peak_matrices = fit_hierarchical_domain_matrices(
        xtx, xty[:, permutation], permuted_domain_stats, selected_ridge
    )
    if set(peak_matrices) != set(correct_matrices):
        raise RuntimeError("correct and peak-permuted domain matrices differ in scope")
    correct = evaluate(
        correct_shared,
        correct_matrices,
        x_identity,
        y_unit,
        unit_identity,
        unit_formula,
        unit_domain,
        unit_fold,
        2,
        candidates,
        structure_swap=False,
        seed=args.seed + 1000,
    )
    structure = evaluate(
        correct_shared,
        correct_matrices,
        x_identity,
        y_unit,
        unit_identity,
        unit_formula,
        unit_domain,
        unit_fold,
        2,
        candidates,
        structure_swap=True,
        seed=args.seed + 1100,
    )
    peak = evaluate(
        peak_shared,
        peak_matrices,
        x_identity,
        y_unit,
        unit_identity,
        unit_formula,
        unit_domain,
        unit_fold,
        2,
        candidates,
        structure_swap=False,
        seed=args.seed + 1200,
    )
    structure_contrast = contrast(correct, structure, args.seed + 2000, args.draws)
    peak_contrast = contrast(correct, peak, args.seed + 3000, args.draws)
    pass_gate = bool(
        structure_contrast["formula_cluster_bootstrap_95ci"][0] > 0
        and peak_contrast["formula_cluster_bootstrap_95ci"][0] > 0
        and correct["pairwise_accuracy"] > 0.5
    )
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "rule_matrix.npz",
        shared_matrix=correct_shared,
        domain_names=np.asarray(sorted(correct_matrices)),
        domain_matrices=np.stack(
            [correct_matrices[value] for value in sorted(correct_matrices)]
        ),
        peak_permuted_shared_matrix=peak_shared,
        peak_permuted_domain_matrices=np.stack(
            [peak_matrices[value] for value in sorted(correct_matrices)]
        ),
        selected_observation_channel=selected_channel,
        peak_permutation=permutation,
        ridge=np.asarray(selected_ridge),
    )
    report = {
        "status": (
            "CHEMAWARE_POSITIVE_RULE_MATRIX_CONFIRMATION_PASS"
            if pass_gate
            else "CHEMAWARE_POSITIVE_RULE_MATRIX_CONFIRMATION_FAIL"
        ),
        "formal_training_authorized": False,
        "official_dreams_parameters_updated": False,
        "fold_contract": {
            "ridge_fit": [0],
            "ridge_selection": 1,
            "confirmation": 2,
            "embedding_evaluation_untouched": 3,
            "reserve_untouched": 4,
        },
        "dimensions": {
            "parent_predicates": int(x_identity.shape[1]),
            "formula_specified_observations": int(y_unit.shape[1]),
            "positive_shared_matrix_entries": int(np.sum(correct_shared > 0)),
            "positive_domain_matrix_entries": int(
                sum(np.sum(value > 0) for value in correct_matrices.values())
            ),
            "domain_matrices": len(correct_matrices),
        },
        "fit_support": {
            "selection_fit_units": train_units0,
            "selection_fit_formula_domain_strata": train_strata0,
            "confirmation_fit_units": train_units,
            "confirmation_fit_formula_domain_strata": train_strata,
        },
        "ridge_selection": {
            "selected": selected_ridge,
            "grid": validation_grid,
        },
        "confirmation": {
            "correct": public_metrics(correct),
            "structure_swapped": public_metrics(structure),
            "peak_permuted": public_metrics(peak),
            "correct_minus_structure_swapped_recall1": structure_contrast,
            "correct_minus_peak_permuted_recall1": peak_contrast,
        },
        "pass_to_direct_candidate_ledger": pass_gate,
        "claim_limit": (
            "the matrix is an OOF-validated aggregate teacher; nonzero coefficients "
            "are not individually admitted mechanistic rules"
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

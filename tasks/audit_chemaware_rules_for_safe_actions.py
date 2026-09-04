"""Test whether chemical rules improve counterfactual action selection.

Unlike an absolute-mass rule scorer, this audit places chemistry where it can
causally affect a shared encoder: selecting which exact peak interventions are
safe training teachers.  The outcome labels come from the frozen A4 deletion
scan.  Formula-group OOF models compare identical capacity and budgets:

* base action evidence + zero padding;
* base + true/adversary candidate-differential structural compatibility;
* base + a within-query peak-rotated chemistry control.

No DreaMS parameter is updated, no P3 spectrum is read, and chemistry is never
an inference feature.  A positive result would only authorize constructing a
matched counterfactual-training arm; it would not itself be a model gain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from audit_chemaware_candidate_differential_rules import (
    decode,
    stable_u64,
    structure_masks,
)
from train_noise_v3_a4_nonlinear_action_teacher import (
    VariantTable,
    build_variant_table,
    select_query_actions,
    train_fold_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a4-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan",
    )
    parser.add_argument(
        "--hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[260903])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cuts", type=int, default=1)
    parser.add_argument("--hydrogen-shift", type=int, default=0)
    parser.add_argument("--fragment-backend", choices=("simple_cut", "magma"), default="simple_cut")
    parser.add_argument(
        "--magma-source-root", type=Path,
        default=ROOT / "data/external/ms-pred-src",
    )
    parser.add_argument("--magma-tree-depth", type=int, default=3)
    parser.add_argument("--magma-max-broken-bonds", type=int, default=6)
    parser.add_argument("--structure-ppm", type=float, default=20.0)
    parser.add_argument("--structure-floor-da", type=float, default=0.01)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260903)
    return parser.parse_args()


def safe_metric(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    if len(np.unique(labels)) < 2:
        return {"prevalence": float(labels.mean()), "roc_auc": None, "average_precision": None}
    return {
        "prevalence": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def formula_bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    resamples: int,
    seed: int,
) -> dict:
    keys = ["query_index", "query_formula", "scan_kind"]
    merged = left[keys + ["contribution"]].merge(
        right[keys + ["contribution"]], on=keys, suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    merged["difference"] = merged["contribution_left"] - merged["contribution_right"]
    grouped = merged.groupby("query_formula", sort=True)["difference"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        chosen = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[chosen].sum() / counts[chosen].sum()
    return {
        "mean_per_query": float(merged["difference"].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def chemical_features(
    table: VariantTable,
    a4_dir: Path,
    hdf5_path: Path,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    queries = pd.read_csv(a4_dir / "scan_queries.csv.gz").set_index("scan_position")
    unique_positions = np.unique(table.query_position)
    correct_by_position: dict[int, dict[int, np.ndarray]] = {}
    rotated_by_position: dict[int, dict[int, np.ndarray]] = {}
    swapped_by_position: dict[int, dict[int, np.ndarray]] = {}
    coverage_rows = []
    with h5py.File(hdf5_path, "r") as handle:
        for position in unique_positions:
            q = queries.loc[int(position)]
            indices = np.flatnonzero(table.query_position == position)
            tokens = np.unique(table.token[indices])
            # One row per peak token; dose variants receive the same chemistry.
            mz_by_token = {
                int(token): float(table.x[indices[table.token[indices] == token][0], 2] * 1000.0)
                for token in tokens
            }
            ordered_tokens = np.asarray(sorted(tokens), dtype=np.int64)
            mz = np.asarray([mz_by_token[int(token)] for token in ordered_tokens], dtype=np.float64)
            query_row = int(q["query_row"])
            adversary_row = int(q["baseline_adversarial_pair_row"])
            precursor = float(handle["precursor_mz"][query_row])
            truth_smiles = decode(handle["smiles"][query_row])
            adversary_smiles = decode(handle["smiles"][adversary_row])
            true_direct, true_loss = structure_masks(
                mz, precursor, truth_smiles, args.max_cuts, args.hydrogen_shift,
                args.structure_ppm, args.structure_floor_da,
                args.fragment_backend, str(args.magma_source_root),
                args.magma_tree_depth, args.magma_max_broken_bonds,
            )
            adversary_direct, adversary_loss = structure_masks(
                mz, precursor, adversary_smiles, args.max_cuts, args.hydrogen_shift,
                args.structure_ppm, args.structure_floor_da,
                args.fragment_backend, str(args.magma_source_root),
                args.magma_tree_depth, args.magma_max_broken_bonds,
            )
            true_any, adversary_any = true_direct | true_loss, adversary_direct | adversary_loss
            true_only = true_any & ~adversary_any
            adversary_only = adversary_any & ~true_any
            both = true_any & adversary_any
            neither = ~true_any & ~adversary_any
            role = np.column_stack((
                true_direct, true_loss, adversary_direct, adversary_loss,
                true_only, adversary_only, both, neither,
                true_any.astype(np.float32) - adversary_any.astype(np.float32),
            )).astype(np.float32)
            # Circularly rotate whole peak roles within the same query.  This
            # preserves formula, structures, query-level coverage, feature
            # marginals, and model capacity while breaking peak attribution.
            shift = 1 + stable_u64(int(q["query_index"]), seed=args.seed) % max(1, len(role) - 1)
            rotated = np.roll(role, int(shift), axis=0) if len(role) > 1 else role.copy()
            # Candidate swap keeps the same two structures and all peak-level
            # cardinalities but reverses which molecule is declared correct.
            # This is a stronger chemical-label control than peak rotation.
            swapped = role[:, [2, 3, 0, 1, 5, 4, 6, 7, 8]].copy()
            swapped[:, 8] *= -1.0
            correct_by_position[int(position)] = {
                int(token): role[index] for index, token in enumerate(ordered_tokens)
            }
            rotated_by_position[int(position)] = {
                int(token): rotated[index] for index, token in enumerate(ordered_tokens)
            }
            swapped_by_position[int(position)] = {
                int(token): swapped[index] for index, token in enumerate(ordered_tokens)
            }
            coverage_rows.append({
                "position": int(position), "query_index": int(q["query_index"]),
                "formula": str(q["query_formula"]), "peaks": int(len(role)),
                "true_only": int(true_only.sum()), "adversary_only": int(adversary_only.sum()),
                "both": int(both.sum()), "neither": int(neither.sum()),
            })

    correct = np.vstack([
        correct_by_position[int(position)][int(token)]
        for position, token in zip(table.query_position, table.token)
    ])
    rotated = np.vstack([
        rotated_by_position[int(position)][int(token)]
        for position, token in zip(table.query_position, table.token)
    ])
    swapped = np.vstack([
        swapped_by_position[int(position)][int(token)]
        for position, token in zip(table.query_position, table.token)
    ])
    dose = table.dose[:, None].astype(np.float32)
    # Let chemistry interact with intervention dose and pre-existing spectral
    # peak role.  The rotated arm receives the exact same expansion.
    role_confounder = (table.role == 1)[:, None].astype(np.float32)
    role_shared = (table.role == 2)[:, None].astype(np.float32)
    def expand(values: np.ndarray) -> np.ndarray:
        return np.column_stack((
            values,
            dose * values[:, [4, 5, 6]],
            role_confounder * values[:, [4, 5]],
            role_shared * values[:, [4, 5]],
        )).astype(np.float32)
    names = [
        "chem_true_direct", "chem_true_loss", "chem_adversary_direct", "chem_adversary_loss",
        "chem_true_only", "chem_adversary_only", "chem_both", "chem_neither", "chem_signed",
        "dose_x_chem_true_only", "dose_x_chem_adversary_only", "dose_x_chem_both",
        "confounder_x_chem_true_only", "confounder_x_chem_adversary_only",
        "shared_x_chem_true_only", "shared_x_chem_adversary_only",
    ]
    coverage = pd.DataFrame(coverage_rows)
    summary = {
        "queries": int(len(coverage)),
        "any_true_only_fraction": float(np.mean(coverage["true_only"] > 0)),
        "any_adversary_only_fraction": float(np.mean(coverage["adversary_only"] > 0)),
        "median_true_only_peaks": float(coverage["true_only"].median()),
        "median_adversary_only_peaks": float(coverage["adversary_only"].median()),
    }
    return expand(correct), expand(rotated), expand(swapped), names, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_arm(table: VariantTable, added: np.ndarray, names: list[str], variant_fold: np.ndarray, args) -> dict:
    arm = copy.copy(table)
    arm.x = np.column_stack((table.x, added)).astype(np.float32)
    arm.feature_names = table.feature_names + names
    oof_benefit = np.zeros(len(arm.x), dtype=np.float32)
    oof_harm = np.zeros(len(arm.x), dtype=np.float32)
    oof_delta = np.zeros(len(arm.x), dtype=np.float32)
    for fold in range(args.folds):
        test = np.flatnonzero(variant_fold == fold)
        train = np.flatnonzero(variant_fold != fold)
        b, h, d = np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test))
        for seed in args.seeds:
            local_b, local_h, local_d, _ = train_fold_seed(arm, train, test, args, seed)
            b += local_b / len(args.seeds); h += local_h / len(args.seeds); d += local_d / len(args.seeds)
        oof_benefit[test] = b; oof_harm[test] = h; oof_delta[test] = d
    error, control = arm.scan_kind == "official_error", arm.scan_kind == "safety_control"
    utility = oof_benefit - args.risk_penalty * oof_harm
    selected = select_query_actions(arm, utility)
    selected["contribution_if_applied"] = (
        selected["corrected_if_applied"].astype(float)
        - args.risk_penalty * selected["introduced_if_applied"].astype(float)
    )
    policies = {}
    for coverage in (0.05, 0.10, 0.20, 0.40):
        count = max(1, int(np.ceil(coverage * len(selected))))
        order = np.lexsort((selected["scan_position"], -selected["predicted_utility"]))
        applied = np.zeros(len(selected), dtype=bool); applied[order[:count]] = True
        contribution = applied.astype(float) * selected["contribution_if_applied"].to_numpy(float)
        policies[f"{coverage:.2f}"] = {
            "corrected": int(np.sum(applied & selected["corrected_if_applied"])),
            "introduced": int(np.sum(applied & selected["introduced_if_applied"])),
            "risk_weighted_net": float(np.sum(contribution)),
        }
        if coverage == 0.20:
            selected["contribution"] = contribution
    return {
        "benefit": safe_metric(arm.corrected[error], oof_benefit[error]),
        "harm": safe_metric(arm.introduced[control], oof_harm[control]),
        "margin_change_spearman": float(pd.Series(oof_delta).corr(pd.Series(arm.margin_change), method="spearman")),
        "policies": policies,
        "selected": selected,
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    table = build_variant_table(args.a4_dir, args.max_queries)
    correct, rotated, swapped, names, coverage = chemical_features(
        table, args.a4_dir, args.hdf5, args,
    )
    query_meta = pd.DataFrame({
        "position": table.query_position, "formula": table.formula,
    }).drop_duplicates("position").sort_values("position")
    query_fold = np.full(int(query_meta.position.max()) + 1, -1, dtype=np.int8)
    splitter = GroupKFold(args.folds)
    positions, groups = query_meta.position.to_numpy(np.int64), query_meta.formula.to_numpy(str)
    for fold, (_, test) in enumerate(splitter.split(positions, groups=groups)):
        query_fold[positions[test]] = fold
    variant_fold = query_fold[table.query_position]
    if np.any(variant_fold < 0):
        raise RuntimeError("formula fold assignment failed")

    zeros = np.zeros_like(correct)
    arms = {
        "base_matched_capacity": run_arm(table, zeros, names, variant_fold, args),
        "correct_chemical_rules": run_arm(table, correct, names, variant_fold, args),
        "peak_rotated_chemical_control": run_arm(table, rotated, names, variant_fold, args),
        "candidate_swapped_chemical_control": run_arm(table, swapped, names, variant_fold, args),
    }
    primary_coverage = "0.20"
    correct_minus_base = formula_bootstrap_difference(
        arms["correct_chemical_rules"]["selected"],
        arms["base_matched_capacity"]["selected"],
        args.bootstrap_resamples, args.seed,
    )
    correct_minus_rotated = formula_bootstrap_difference(
        arms["correct_chemical_rules"]["selected"],
        arms["peak_rotated_chemical_control"]["selected"],
        args.bootstrap_resamples, args.seed + 1,
    )
    correct_minus_swapped = formula_bootstrap_difference(
        arms["correct_chemical_rules"]["selected"],
        arms["candidate_swapped_chemical_control"]["selected"],
        args.bootstrap_resamples, args.seed + 2,
    )
    pass_gate = bool(
        correct_minus_base["ci_low"] > 0
        and correct_minus_rotated["ci_low"] > 0
        and correct_minus_swapped["ci_low"] > 0
        and coverage["any_true_only_fraction"] >= 0.10
        and coverage["any_adversary_only_fraction"] >= 0.10
    )
    report = {
        "status": "candidate_differential_rules_for_safe_actions_preflight",
        "formal_training_authorized": False,
        "dreaMS_parameters_updated": False,
        "queries": int(len(np.unique(table.query_position))),
        "variants": int(len(table.x)),
        "formulas": int(len(np.unique(table.formula))),
        "formula_fold_overlap": 0,
        "chemical_rule_engine": {
            "fragment_backend": args.fragment_backend,
            "max_cuts": args.max_cuts if args.fragment_backend == "simple_cut" else None,
            "hydrogen_shift": args.hydrogen_shift if args.fragment_backend == "simple_cut" else None,
            "magma_tree_depth": args.magma_tree_depth if args.fragment_backend == "magma" else None,
            "magma_max_broken_bonds": (
                args.magma_max_broken_bonds if args.fragment_backend == "magma" else None
            ),
            "magma_source_root": (
                str(args.magma_source_root.resolve()) if args.fragment_backend == "magma" else None
            ),
            "magma_fragmentation_sha256": (
                sha256_file(args.magma_source_root / "src/ms_pred/magma/fragmentation.py")
                if args.fragment_backend == "magma" else None
            ),
        },
        "chemical_coverage": coverage,
        "arms": {
            name: {key: value for key, value in result.items() if key != "selected"}
            for name, result in arms.items()
        },
        "paired_primary_policy_difference": {
            "correct_minus_base": correct_minus_base,
            "correct_minus_rotated": correct_minus_rotated,
            "correct_minus_candidate_swapped": correct_minus_swapped,
            "coverage": primary_coverage,
        },
        "pass_to_counterfactual_training_mechanism": pass_gate,
        "gate": (
            "Chemical rules may enter a matched counterfactual-training arm only if correct chemistry "
            "beats zero-padded base, peak-rotated chemistry, and candidate-swapped chemistry in "
            "formula-OOF policy utility, "
            "with positive cluster-bootstrap lower bounds and nontrivial true/adversary-only coverage."
        ),
        "claim_limit": (
            "This tests training-action selection, not a retrieval model. Even a pass requires a "
            "separate clean-input shared-encoder experiment against an identity-only matched arm."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in arms.items():
        result["selected"].to_csv(args.output_dir / f"{name}_selected.csv.gz", index=False, compression="gzip")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

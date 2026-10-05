"""Test the frozen orthogonal ChemAware teacher on a multi-spectrum identity panel.

The strong policy is refit with its frozen fold-0/1 recipe and its dose and
threshold are reselected on fold 2. A target-free sample of identities with at
least two fold-3 spectra then supplies a repeat panel. No fold-3 result changes
the policy and outer fold 4 is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audit_chemaware_candidate_evidence_policy as candidate_policy
import audit_chemaware_orthogonal_rule_residual_policy as orthogonal
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_orthogonal_rule_policy_core import (
    base_and_chemical_feature_indices,
    signed_rule_contrast,
    validate_matched_tables,
)
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("correct", "zero_contrast", "reversed_contrast", "alignment_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--frozen-report", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_rule_residual_policy_v2/report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v1",
    )
    parser.add_argument("--repeat-identities", type=int, default=512)
    parser.add_argument("--spectra-per-identity", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repeat_panel(
    pool: np.ndarray,
    identity: np.ndarray,
    *,
    identities: int,
    spectra_per_identity: int,
    seed: int,
) -> np.ndarray:
    pool_identity = np.asarray(identity)[pool].astype(str)
    unique, counts = np.unique(pool_identity, return_counts=True)
    eligible = unique[counts >= 2]
    rng = np.random.default_rng(seed)
    selected_identity = eligible[rng.permutation(len(eligible))[:identities]]
    selected: list[int] = []
    for value in selected_identity:
        positions = pool[pool_identity == value]
        chosen = positions[rng.permutation(len(positions))[:spectra_per_identity]]
        selected.extend(map(int, chosen))
    return np.asarray(selected, dtype=np.int64)


def pair_indices(identity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    for value in np.unique(identity):
        positions = np.flatnonzero(identity == value)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                left.append(int(positions[i])); right.append(int(positions[j]))
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def agreement(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> float | None:
    if not len(left):
        return None
    return float(np.mean(np.asarray(values)[left] == np.asarray(values)[right]))


def permutation_agreement(
    values: np.ndarray,
    formula: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    observed = agreement(values, left, right)
    if observed is None:
        return {"repeat_pairs": 0, "observed_agreement": None}
    groups = [np.flatnonzero(formula == value) for value in np.unique(formula)]
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(values)
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        for positions in groups:
            permuted[positions] = values[positions][rng.permutation(len(positions))]
        null[draw] = float(np.mean(permuted[left] == permuted[right]))
    return {
        "repeat_pairs": int(len(left)),
        "observed_agreement": observed,
        "formula_permutation_null_quantiles": [
            float(x) for x in np.quantile(null, (0.025, 0.5, 0.975))
        ],
        "excess_over_null_median": float(observed - np.median(null)),
        "empirical_p_null_ge_observed": float(
            (1 + np.sum(null >= observed)) / (1 + len(null))
        ),
    }


def selected_identities(
    table: dict[str, np.ndarray],
    selected: np.ndarray,
    queries: np.ndarray,
    body: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.full(len(queries), "__NO_OP__", dtype="U14")
    for index, slot in enumerate(np.asarray(selected, dtype=np.int64)):
        if slot < 0:
            continue
        local = int(table["proposed_candidate"][index, slot])
        if local < 0:
            raise RuntimeError("active policy selected an invalid candidate slot")
        left = int(body["query_ptr"][int(queries[index])])
        output[index] = str(body["molecule_ik14"][left + local])
    return output


def candidate_identity_matrix(
    table: dict[str, np.ndarray], queries: np.ndarray, body: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.full(table["valid"].shape, "", dtype="U14")
    for index, query in enumerate(np.asarray(queries, dtype=np.int64)):
        left = int(body["query_ptr"][query])
        slots = np.flatnonzero(table["valid"][index])
        local = table["proposed_candidate"][index, slots].astype(np.int64)
        output[index, slots] = body["molecule_ik14"][left + local].astype(str)
    return output


def arm_repeat_report(
    table: dict[str, np.ndarray],
    rank: np.ndarray,
    selected: np.ndarray,
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    formula: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    baseline = np.asarray(table["baseline_rank"], dtype=np.int64)
    rank = np.asarray(rank, dtype=np.int64)
    active = np.asarray(selected) >= 0
    corrected = (baseline > 1) & (rank == 1)
    introduced = (baseline == 1) & (rank > 1)
    state = np.full(len(rank), "neutral", dtype="U10")
    state[corrected] = "corrected"; state[introduced] = "introduced"
    chosen_identity = selected_identities(table, selected, queries, body)
    both_active = active[left] & active[right]
    both_error = (baseline[left] > 1) & (baseline[right] > 1)
    both_correct = (baseline[left] == 1) & (baseline[right] == 1)
    return {
        "active_queries": int(np.sum(active)),
        "corrected_queries": int(np.sum(corrected)),
        "introduced_queries": int(np.sum(introduced)),
        "risk_utility_at_1": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "top1_delta": float(np.mean(rank == 1) - np.mean(baseline == 1)),
        "active_repeat": permutation_agreement(
            active.astype(np.int8), formula, left, right,
            permutations=permutations, seed=seed,
        ),
        "outcome_state_repeat": permutation_agreement(
            state, formula, left, right,
            permutations=permutations, seed=seed + 1,
        ),
        "selected_identity_agreement_when_both_active": agreement(
            chosen_identity, left[both_active], right[both_active],
        ),
        "both_active_pairs": int(np.sum(both_active)),
        "correction_agreement_when_both_baseline_error": agreement(
            corrected.astype(np.int8), left[both_error], right[both_error],
        ),
        "both_baseline_error_pairs": int(np.sum(both_error)),
        "harm_agreement_when_both_baseline_correct": agreement(
            introduced.astype(np.int8), left[both_correct], right[both_correct],
        ),
        "both_baseline_correct_pairs": int(np.sum(both_correct)),
    }, state, chosen_identity


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeat_identities <= 0 or args.spectra_per_identity < 2:
        raise ValueError("repeat panel requires positive identities and at least two spectra each")
    if args.permutations < 100:
        raise ValueError("at least 100 permutations are required")
    frozen = json.loads(args.frozen_report.read_text(encoding="utf-8"))
    frozen_args = dict(frozen["replay_contract"]["arguments"])
    # v2 predates serialization of these parser defaults.  The frozen method
    # and contrast hashes identify the legacy single-null/mean construction.
    frozen_args.setdefault("rule_control_variants", ("rule_response_content_permuted",))
    frozen_args.setdefault("contrast_representation", "mean")
    if args.smoke:
        frozen_args.update(
            train_identities=192, validation_identities=128, max_iter=30,
            min_samples_leaf=20, min_selected_formulas=10,
        )
        args.repeat_identities = min(args.repeat_identities, 32)
        args.spectra_per_identity = min(args.spectra_per_identity, 3)
        args.permutations = min(args.permutations, 200)
    recipe = SimpleNamespace(**frozen_args)
    recipe.manifest = args.manifest; recipe.token_dir = args.token_dir
    recipe.rule_library = args.rule_library

    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], recipe.folds, recipe.fold_seed)
    train_pool = np.flatnonzero((fold == 0) | (fold == 1))
    validation_pool = np.flatnonzero(fold == 2)
    train = identity_balanced_queries(
        train_pool, body["query_ik14"], np.random.default_rng(recipe.sampling_seed + 1),
        recipe.train_identities,
    )
    validation = identity_balanced_queries(
        validation_pool, body["query_ik14"], np.random.default_rng(recipe.sampling_seed + 2),
        recipe.validation_identities,
    )
    panel = repeat_panel(
        np.flatnonzero(fold == 3), body["query_ik14"],
        identities=args.repeat_identities, spectra_per_identity=args.spectra_per_identity,
        seed=args.seed,
    )
    queries = (train, validation, panel)
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    train_formula_fold = fold[train]

    actions = [
        (float(mass), float(rule))
        for mass in recipe.beta for rule in recipe.beta
        if float(mass) > 0.0 or float(rule) > 0.0
    ]
    global_action = actions.index((recipe.global_mass_beta, recipe.global_rule_beta))
    cache = KernelCache(
        recipe, row_position,
        variants=("mass", "rule_response", *tuple(recipe.rule_control_variants)),
    )
    correct_tables: list[dict[str, np.ndarray]] = []
    control_tables: list[dict[str, np.ndarray]] = []
    for name, query in zip(("train", "validation", "repeat_panel"), queries, strict=True):
        scored = score_queries(
            query, body, official, row_position, cache,
            ("mass", "rule_response", *tuple(recipe.rule_control_variants)),
        )
        correct = candidate_policy.build_candidate_table(
            scored, actions, global_action, "rule_response",
        )
        controls = [
            candidate_policy.build_candidate_table(scored, actions, global_action, variant)
            for variant in recipe.rule_control_variants
        ]
        averaged = orthogonal.averaged_control_table(controls)
        validate_matched_tables(correct, averaged)
        correct_tables.append(correct); control_tables.append(averaged)
        print(f"completed {name}: queries={len(query)} rows={int(correct['valid'].sum())}", flush=True)

    base_indices, chemical_indices = base_and_chemical_feature_indices(candidate_policy.FEATURE_NAMES)
    base_features = [table["feature"][..., base_indices] for table in correct_tables]
    contrasts = [
        signed_rule_contrast(correct, control, chemical_indices)
        for correct, control in zip(correct_tables, control_tables, strict=True)
    ]
    channels = {
        target: orthogonal.fit_channel(
            correct_tables[0], formulas[0], train_formula_fold,
            base_features[0], contrasts[0], target, recipe,
            recipe.seed + (100 if target == "benefit" else 200),
        )
        for target in ("benefit", "harmful")
    }
    validation_arms, _ = orthogonal.make_contrast_arms(
        contrasts[1], correct_tables[1], recipe.seed + 301,
    )
    utilities = {
        float(dose): {
            name: orthogonal.utility_for_contrast(
                channels, base_features[1], contrast, correct_tables[1]["valid"],
                float(dose), recipe.risk_penalty,
            )[0]
            for name, contrast in validation_arms.items()
        }
        for dose in recipe.residual_dose
    }
    selected_setting, _ = orthogonal.select_specific_setting(
        correct_tables[1], formulas[1], utilities,
        recipe.min_selected_formulas,
        tuple(name for name in validation_arms if name != "correct"),
    )
    if not args.smoke:
        expected = frozen["selection"]
        if (
            float(selected_setting["dose"]) != float(expected["dose"])
            or float(selected_setting["threshold"]) != float(expected["threshold"])
        ):
            raise RuntimeError("frozen orthogonal teacher selection was not exactly replayed")

    panel_arms, _ = orthogonal.make_contrast_arms(
        contrasts[2], correct_tables[2], recipe.seed + 302,
    )
    evaluated: dict[str, dict[str, np.ndarray]] = {}
    for name, contrast in panel_arms.items():
        utility = orthogonal.utility_for_contrast(
            channels, base_features[2], contrast, correct_tables[2]["valid"],
            float(selected_setting["dose"]), recipe.risk_penalty,
        )[0]
        rank, selected, best = orthogonal.rank_at_threshold(
            correct_tables[2], utility, float(selected_setting["threshold"]),
        )
        evaluated[name] = {
            "rank": rank, "selected": selected, "best": best, "utility": utility,
        }

    identity = body["query_ik14"][panel].astype(str)
    formula = formulas[2]
    pair_left, pair_right = pair_indices(identity)
    arm_reports = {}; state = {}; chosen_identity = {}
    for offset, name in enumerate(ARMS):
        arm_reports[name], state[name], chosen_identity[name] = arm_repeat_report(
            correct_tables[2], evaluated[name]["rank"], evaluated[name]["selected"],
            panel, body, formula, pair_left, pair_right,
            permutations=args.permutations, seed=args.seed + 100 * offset,
        )
    report = {
        "status": "CHEMAWARE_ORTHOGONAL_TEACHER_REPEAT_CONSISTENCY_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": (
            "Frozen folds 0-1 fit and fold-2 selection replay; target-free multi-spectrum "
            "identity sample from used inner fold 3; outer fold 4 untouched."
        ),
        "claim_limit": (
            "Repeat consistency diagnoses whether a candidate teacher can supervise a "
            "spectrum-only student; it is not a new retrieval-performance estimate."
        ),
        "frozen_selection_replay": selected_setting,
        "data": {
            "panel_queries": int(len(panel)),
            "panel_identities": int(len(np.unique(identity))),
            "panel_formulas": int(len(np.unique(formula))),
            "within_identity_pairs": int(len(pair_left)),
            "spectra_per_identity_cap": int(args.spectra_per_identity),
            "outer_fold_untouched": int(np.sum(fold == 4)),
        },
        "arms": arm_reports,
        "specificity": {
            "correct_minus_alignment_permuted_active_excess_over_null": float(
                arm_reports["correct"]["active_repeat"]["excess_over_null_median"]
                - arm_reports["alignment_permuted"]["active_repeat"]["excess_over_null_median"]
            ),
            "correct_minus_alignment_permuted_state_excess_over_null": float(
                arm_reports["correct"]["outcome_state_repeat"]["excess_over_null_median"]
                - arm_reports["alignment_permuted"]["outcome_state_repeat"]["excess_over_null_median"]
            ),
        },
        "interpretation_contract": {
            "same_frozen_model_dose_threshold_all_arms": True,
            "panel_identity_selection_uses_no_outcomes": True,
            "formula_prevalence_preserved_in_null": True,
            "fold3_does_not_change_policy": True,
            "outer_fold_untouched": True,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "frozen_report_sha256": sha256(args.frozen_report),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_repeat_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        candidate_identity = candidate_identity_matrix(correct_tables[2], panel, body)
        np.savez_compressed(
            temporary / "repeat_ledger.npz", query=panel, formula=formula, identity=identity,
            baseline_rank=correct_tables[2]["baseline_rank"],
            candidate_identity=candidate_identity, candidate_valid=correct_tables[2]["valid"],
            proposal_rank=correct_tables[2]["rank"],
            **{f"{name}_rank": evaluated[name]["rank"] for name in ARMS},
            **{f"{name}_selected": evaluated[name]["selected"] for name in ARMS},
            **{f"{name}_best_utility": evaluated[name]["best"] for name in ARMS},
            **{f"{name}_candidate_utility": evaluated[name]["utility"] for name in ARMS},
            **{f"{name}_state": state[name] for name in ARMS},
            **{f"{name}_selected_identity": chosen_identity[name] for name in ARMS},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

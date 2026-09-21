"""Candidate-level ChemAware intervention with reference-set evidence.

Each non-baseline molecule is represented once, rather than once per action
weight.  Features summarize the full reference-spectrum set and the connected
area in mass/rule weight space over which that molecule wins.  Separate models
estimate corrective benefit and destructive harm.  Formula folds 0--1 fit,
fold 2 freezes the abstention threshold, fold 3 is the used inner audit, and
fold 4 remains sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
try:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
except ModuleNotFoundError:  # Feature construction itself is NumPy-only.
    sklearn = None
    HistGradientBoostingClassifier = None

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from chemaware_numpy_sampling import identity_balanced_queries, stable_formula_folds
from chemaware_truthblind_candidate_core import (
    action_topology,
    build_truthblind_candidate_features,
    truthblind_scored_view,
)

action_policy = None
paired_rank_comparison = None


ROOT = Path(__file__).resolve().parents[1]
STAT_NAMES = ("max", "second", "top2_mean", "mean", "median", "std", "q75", "max_gap")
CHANNELS = ("official", "mass", "rule")
DESCRIPTOR_NAMES = (
    ["log_reference_count"]
    + [f"{channel}_{stat}" for channel in CHANNELS for stat in STAT_NAMES]
    + ["corr_official_mass", "corr_official_rule", "corr_mass_rule"]
    + ["argmax_official_mass", "argmax_official_rule", "argmax_mass_rule"]
    + ["mass_at_official_best", "rule_at_official_best", "official_at_mass_best", "official_at_rule_best"]
)
ACTION_NAMES = [
    "action_top_fraction", "action_largest_region_fraction",
    "action_same_neighbor_fraction", "action_margin_max", "action_margin_mean",
    "action_min_total_beta", "action_max_total_beta",
    "action_best_advantage_over_baseline", "global_action_advantage_over_baseline",
    "global_action_selects_candidate",
]
CONTEXT_NAMES = [
    "log_candidate_count", "baseline_official_margin",
    "candidate_official_rank_fraction", "candidate_mass_rank_fraction",
    "candidate_rule_rank_fraction",
]
FEATURE_NAMES = (
    CONTEXT_NAMES
    + [f"candidate_{name}" for name in DESCRIPTOR_NAMES]
    + [f"delta_{name}" for name in DESCRIPTOR_NAMES]
    + ACTION_NAMES
)


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
        "--action-policy", type=Path,
        default=ROOT / "data/validation/chemaware_conservative_action_policy_v3",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_candidate_evidence_policy_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--global-mass-beta", type=float, default=0.1)
    parser.add_argument("--global-rule-beta", type=float, default=0.2)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--min-selected-formulas", type=int, default=50)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    output = np.empty(len(order), dtype=np.float32)
    output[order] = np.arange(len(order), dtype=np.float32)
    return output / max(1, len(order) - 1)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if np.isfinite(result) else 0.0


def statistics(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)
    maximum = float(ordered[-1])
    second = float(ordered[-2]) if len(ordered) > 1 else maximum
    return [
        maximum, second, float(np.mean(ordered[-min(2, len(ordered)):])),
        float(np.mean(values)), float(np.median(values)), float(np.std(values)),
        float(np.quantile(values, 0.75)), maximum - second,
    ]


def reference_descriptor(official: np.ndarray, mass: np.ndarray, rule: np.ndarray) -> np.ndarray:
    values = [np.asarray(x, dtype=np.float64) for x in (official, mass, rule)]
    result: list[float] = [float(np.log1p(len(official)))]
    for channel in values:
        result.extend(statistics(channel))
    result.extend([
        safe_correlation(values[0], values[1]),
        safe_correlation(values[0], values[2]),
        safe_correlation(values[1], values[2]),
        float(np.argmax(values[0]) == np.argmax(values[1])),
        float(np.argmax(values[0]) == np.argmax(values[2])),
        float(np.argmax(values[1]) == np.argmax(values[2])),
        float(values[1][np.argmax(values[0])]),
        float(values[2][np.argmax(values[0])]),
        float(values[0][np.argmax(values[1])]),
        float(values[0][np.argmax(values[2])]),
    ])
    output = np.asarray(result, dtype=np.float32)
    if len(output) != len(DESCRIPTOR_NAMES):
        raise RuntimeError("reference descriptor schema drifted")
    return output


def connected_size(
    top: np.ndarray, candidate: int, neighbors: list[list[int]],
) -> tuple[int, float]:
    remaining = set(np.flatnonzero(top == candidate).tolist())
    largest = 0
    supports: list[float] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            degree = len(neighbors[current])
            supports.append(sum(int(top[x]) == candidate for x in neighbors[current]) / max(1, degree))
            for other in neighbors[current]:
                if other in remaining and int(top[other]) == candidate:
                    remaining.remove(other)
                    stack.append(other)
        largest = max(largest, size)
    return largest, (float(np.mean(supports)) if supports else 0.0)


def build_candidate_table(
    scored: dict[str, np.ndarray], actions: list[tuple[float, float]],
    global_action: int, rule_key: str,
) -> dict[str, np.ndarray]:
    # The training/evaluation ledger contains truth, but the deployment core
    # must never receive it.  Pass an explicit allow-list rather than deleting
    # known truth keys so future metadata cannot silently cross the boundary.
    truthblind_scored = truthblind_scored_view(scored, rule_key)
    truthblind = build_truthblind_candidate_features(
        truthblind_scored, actions, global_action, rule_key,
    )
    query_count, maximum_challengers = truthblind["valid"].shape
    proposal_rank = np.ones((query_count, maximum_challengers), dtype=np.int16)
    for query in range(query_count):
        pointer = np.asarray(scored["reference_ptr"][query], dtype=np.int64)
        labels = np.asarray(scored["labels"][query], dtype=bool)
        official = np.maximum.reduceat(
            np.asarray(scored["global"][query], dtype=np.float32), pointer[:-1],
        )
        if strict_rank(official, labels) != int(scored["old_rank"][query]):
            raise RuntimeError("candidate aggregation does not replay official baseline")
        for slot in np.flatnonzero(truthblind["valid"][query]):
            candidate = int(truthblind["proposed_candidate"][query, slot])
            promoted = np.asarray(official, dtype=np.float64).copy()
            promoted[candidate] = np.nextafter(float(np.max(official)), np.inf)
            proposal_rank[query, slot] = strict_rank(promoted, labels)
    baseline_rank = np.asarray(scored["old_rank"], dtype=np.int16)
    return {
        **truthblind, "rank": proposal_rank,
        "benefit": truthblind["valid"] & (baseline_rank[:, None] > 1) & (proposal_rank == 1),
        "harmful": truthblind["valid"] & (baseline_rank[:, None] == 1) & (proposal_rank > 1),
        "baseline_rank": baseline_rank,
    }


def max_only_indices() -> np.ndarray:
    disallowed = ("second", "top2", "mean", "median", "std", "q75", "max_gap", "corr_", "argmax_", "_at_")
    return np.asarray([
        index for index, name in enumerate(FEATURE_NAMES)
        if not any(token in name for token in disallowed)
    ], dtype=np.int64)


def fit_classifier(
    table: dict[str, np.ndarray], target_name: str, formula: np.ndarray,
    feature_index: np.ndarray, args: argparse.Namespace, *, permute: bool,
) -> tuple[HistGradientBoostingClassifier, float]:
    valid = table["valid"]
    target = table[target_name][valid].astype(bool)
    feature = table["feature"][valid][:, feature_index]
    count = valid.sum(axis=1)
    query_weight = action_policy.formula_query_weight(formula)
    event_weight = np.concatenate([
        np.full(n, query_weight[index] / max(1, n), dtype=np.float64)
        for index, n in enumerate(count)
    ])
    if permute:
        target = target[np.random.default_rng(
            args.seed + (991 if target_name == "benefit" else 997)
        ).permutation(len(target))]
    sample_weight, prevalence = action_policy.balanced_event_weight(target, event_weight)
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=args.learning_rate, max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes, min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization, early_stopping=False,
        random_state=args.seed,
    )
    model.fit(feature, target, sample_weight=sample_weight)
    return model, prevalence


def predict(
    table: dict[str, np.ndarray], feature_index: np.ndarray,
    benefit_model: HistGradientBoostingClassifier, harm_model: HistGradientBoostingClassifier,
    benefit_prevalence: float, harm_prevalence: float, risk_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = table["valid"].shape
    benefit = np.zeros(shape, dtype=np.float64)
    harm = np.ones(shape, dtype=np.float64)
    feature = table["feature"][table["valid"]][:, feature_index]
    benefit[table["valid"]] = action_policy.restore_prior(
        benefit_model.predict_proba(feature)[:, 1], benefit_prevalence,
    )
    harm[table["valid"]] = action_policy.restore_prior(
        harm_model.predict_proba(feature)[:, 1], harm_prevalence,
    )
    utility = benefit - float(risk_penalty) * harm
    utility[~table["valid"]] = -np.inf
    return utility, benefit, harm


def ranks_at_threshold(
    table: dict[str, np.ndarray], utility: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.argmax(utility, axis=1)
    best = utility[np.arange(len(selected)), selected]
    active = best >= threshold
    rank = np.array(table["baseline_rank"], copy=True)
    rank[active] = table["rank"][np.arange(len(rank))[active], selected[active]]
    selected[~active] = -1
    return rank, selected.astype(np.int16), best


def choose_threshold(
    table: dict[str, np.ndarray], formula: np.ndarray, utility: np.ndarray,
    min_selected_formulas: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best = np.max(utility, axis=1)
    finite = best[np.isfinite(best)]
    thresholds = np.unique(np.r_[
        np.quantile(finite, np.linspace(0.0, 1.0, 101)),
        np.nextafter(np.max(finite), np.inf),
    ])
    rows = []
    for threshold in thresholds:
        rank, selected, _ = ranks_at_threshold(table, utility, float(threshold))
        active = selected >= 0
        rows.append({
            "threshold": float(threshold), "selected": int(active.sum()),
            "selected_formulas": int(len(np.unique(formula[active].astype(str)))),
            **retrieval(table["baseline_rank"], rank),
        })
    eligible = [
        row for row in rows
        if row["selected"] == 0 or row["selected_formulas"] >= min_selected_formulas
    ]
    selected = max(
        eligible,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), float(row["threshold"]),
        ),
    )
    return selected, rows


def train_arm(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
    train_formula: np.ndarray, validation_formula: np.ndarray,
    feature_index: np.ndarray, args: argparse.Namespace, *, permute: bool,
) -> dict[str, object]:
    benefit_model, benefit_prevalence = fit_classifier(
        train, "benefit", train_formula, feature_index, args, permute=permute,
    )
    harm_model, harm_prevalence = fit_classifier(
        train, "harmful", train_formula, feature_index, args, permute=permute,
    )
    utility, benefit, harm = predict(
        validation, feature_index, benefit_model, harm_model,
        benefit_prevalence, harm_prevalence, args.risk_penalty,
    )
    selected, grid = choose_threshold(
        validation, validation_formula, utility, args.min_selected_formulas,
    )
    return {
        "benefit_model": benefit_model, "harm_model": harm_model,
        "benefit_prevalence": benefit_prevalence,
        "harm_prevalence": harm_prevalence,
        "validation_utility": utility, "validation_benefit": benefit,
        "validation_harm": harm, "threshold": float(selected["threshold"]),
        "threshold_selection": selected, "threshold_grid": grid,
        "feature_index": feature_index,
    }


def evaluate_arm(
    arm: dict[str, object], table: dict[str, np.ndarray], formula: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    utility, benefit, harm = predict(
        table, arm["feature_index"], arm["benefit_model"], arm["harm_model"],
        arm["benefit_prevalence"], arm["harm_prevalence"], args.risk_penalty,
    )
    rank, selected, best = ranks_at_threshold(table, utility, arm["threshold"])
    return {
        "rank": rank, "selected": selected, "best": best,
        "benefit": benefit, "harm": harm,
        "metric": retrieval(table["baseline_rank"], rank),
        "ci": bootstrap(formula, table["baseline_rank"], rank, args.bootstrap_draws, args.seed + 500),
    }


def main() -> None:
    global action_policy, paired_rank_comparison
    if sklearn is None:
        raise RuntimeError("scikit-learn is required for candidate policy fitting")
    import audit_chemaware_conservative_action_policy as action_policy_module
    from audit_chemaware_counterfactual_rule_kernel_natural import (
        paired_rank_comparison as paired_rank_comparison_function,
    )
    action_policy = action_policy_module
    paired_rank_comparison = paired_rank_comparison_function
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    actions = [
        (float(mass), float(rule))
        for mass in args.beta for rule in args.beta
        if float(mass) > 0.0 or float(rule) > 0.0
    ]
    global_action = actions.index((args.global_mass_beta, args.global_rule_beta))
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    pools = [
        np.flatnonzero((fold == 0) | (fold == 1)),
        np.flatnonzero(fold == 2), np.flatnonzero(fold == 3),
    ]
    queries = [
        identity_balanced_queries(pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1), args.train_identities),
        identity_balanced_queries(pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2), args.validation_identities),
        identity_balanced_queries(pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19), args.max_inner_identities),
    ]
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    kernel_args = SimpleNamespace(**vars(args))
    cache = KernelCache(
        kernel_args, row_position,
        variants=("mass", "rule_response", "rule_response_content_permuted"),
    )
    scored = []
    correct_tables = []
    permuted_tables = []
    for name, query in zip(("train", "validation", "inner"), queries, strict=True):
        score = score_queries(
            query, body, official, row_position, cache,
            ("mass", "rule_response", "rule_response_content_permuted"),
        )
        scored.append(score)
        print(f"completed {name} score geometry rows={len(cache.cache)}", flush=True)
        correct_tables.append(build_candidate_table(score, actions, global_action, "rule_response"))
        permuted_tables.append(build_candidate_table(score, actions, global_action, "rule_response_content_permuted"))
        print(
            f"completed {name} candidate rows={int(correct_tables[-1]['valid'].sum())}",
            flush=True,
        )

    all_features = np.arange(len(FEATURE_NAMES), dtype=np.int64)
    max_features = max_only_indices()
    arms = {
        "correct_full": train_arm(
            correct_tables[0], correct_tables[1], formulas[0], formulas[1],
            all_features, args, permute=False,
        ),
        "correct_max_only": train_arm(
            correct_tables[0], correct_tables[1], formulas[0], formulas[1],
            max_features, args, permute=False,
        ),
        "rule_content_permuted": train_arm(
            permuted_tables[0], permuted_tables[1], formulas[0], formulas[1],
            all_features, args, permute=False,
        ),
        "label_permuted": train_arm(
            correct_tables[0], correct_tables[1], formulas[0], formulas[1],
            all_features, args, permute=True,
        ),
    }
    evaluated = {
        "correct_full": evaluate_arm(arms["correct_full"], correct_tables[2], formulas[2], args),
        "correct_max_only": evaluate_arm(arms["correct_max_only"], correct_tables[2], formulas[2], args),
        "rule_content_permuted": evaluate_arm(arms["rule_content_permuted"], permuted_tables[2], formulas[2], args),
        "label_permuted": evaluate_arm(arms["label_permuted"], correct_tables[2], formulas[2], args),
    }
    primary = evaluated["correct_full"]
    comparisons = {
        "correct_full_minus_correct_max_only": paired_rank_comparison(
            primary["rank"], evaluated["correct_max_only"]["rank"], formulas[2],
            draws=args.bootstrap_draws, seed=args.seed + 601,
        ),
        "correct_full_minus_rule_content_permuted": paired_rank_comparison(
            primary["rank"], evaluated["rule_content_permuted"]["rank"], formulas[2],
            draws=args.bootstrap_draws, seed=args.seed + 602,
        ),
        "correct_full_minus_label_permuted": paired_rank_comparison(
            primary["rank"], evaluated["label_permuted"]["rank"], formulas[2],
            draws=args.bootstrap_draws, seed=args.seed + 603,
        ),
    }
    with np.load(args.action_policy / "inner_policy.npz", allow_pickle=False) as loaded:
        if not np.array_equal(np.asarray(loaded["query"]), queries[2]):
            raise RuntimeError("action policy comparator query set drifted")
        action_rank = np.asarray(loaded["policy_rank"])
    comparisons["correct_full_minus_action_policy_v3"] = paired_rank_comparison(
        primary["rank"], action_rank, formulas[2], draws=args.bootstrap_draws,
        seed=args.seed + 604,
    )
    valid_rank = np.where(correct_tables[2]["valid"], correct_tables[2]["rank"], 32767)
    oracle_rank = np.minimum(correct_tables[2]["baseline_rank"], np.min(valid_rank, axis=1))
    report = {
        "status": "CHEMAWARE_CANDIDATE_EVIDENCE_POLICY_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "folds 0-1 fit; fold 2 threshold; used inner fold 3; fold 4 sealed",
        "claim_limit": (
            "Development candidate intervention, not embedding performance or external confirmation. "
            "The candidate oracle observes truth and is headroom only."
        ),
        "method": {
            "unit": "one non-baseline candidate molecule per query",
            "utility": "P(correct official error)-2*P(destroy official correct)",
            "feature_names": FEATURE_NAMES,
            "max_only_feature_names": [FEATURE_NAMES[index] for index in max_features],
            "features_observable_at_inference": True,
            "candidate_identity_or_formula_feature": False,
            "truth_feature": False,
            "explicit_no_op": True,
            "reference_aggregation": "max, second, top2 mean, mean, median, std, q75 and cross-channel reference agreement",
            "action_topology": "largest same-candidate 4-neighbor region on mass/rule beta grid",
        },
        "data": {
            "train_queries": int(len(queries[0])), "validation_queries": int(len(queries[1])),
            "inner_queries": int(len(queries[2])), "outer_queries_untouched": int(np.sum(fold == 4)),
            "formula_overlap": 0,
            "candidate_rows": [int(table["valid"].sum()) for table in correct_tables],
        },
        "training_events": {
            "benefit": int(correct_tables[0]["benefit"].sum()),
            "harmful": int(correct_tables[0]["harmful"].sum()),
        },
        "validation": {
            name: arm["threshold_selection"] for name, arm in arms.items()
        },
        "held_inner": {
            name: result["metric"] for name, result in evaluated.items()
        },
        "held_inner_absolute_formula_bootstrap_ci95": {
            name: result["ci"] for name, result in evaluated.items()
        },
        "held_inner_candidate_oracle": {
            **retrieval(correct_tables[2]["baseline_rank"], oracle_rank),
            "claim_limit": "best candidate chosen after observing truth; headroom only",
        },
        "paired_inner": comparisons,
        "gates": {
            "absolute_ci_positive": primary["ci"][0] > 0,
            "corrected_exceeds_twice_introduced": (
                primary["metric"]["corrected_at_1"] > 2 * primary["metric"]["introduced_at_1"]
            ),
            "beats_max_only_ci": comparisons["correct_full_minus_correct_max_only"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_rule_permuted_ci": comparisons["correct_full_minus_rule_content_permuted"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_label_permuted_ci": comparisons["correct_full_minus_label_permuted"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_action_policy_v3_ci": comparisons["correct_full_minus_action_policy_v3"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "outer_fold_untouched": True,
        },
        "replay_contract": {
            "arguments": {
                key: (str(value.resolve()) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "environment": {
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            },
            "query_sha256": {
                name: array_sha256(query)
                for name, query in zip(("train", "validation", "inner"), queries, strict=True)
            },
            "candidate_rank_sha256": {
                name: array_sha256(table["rank"])
                for name, table in zip(("train", "validation", "inner"), correct_tables, strict=True)
            },
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
            "action_policy_report_sha256": sha256(args.action_policy / "report.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_candidate_evidence_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz", query=queries[2], formula=formulas[2],
            baseline_rank=correct_tables[2]["baseline_rank"], policy_rank=primary["rank"],
            oracle_rank=oracle_rank, selected_candidate_slot=primary["selected"],
            best_predicted_utility=primary["best"], predicted_benefit=primary["benefit"],
            predicted_harm=primary["harm"], valid_candidate=correct_tables[2]["valid"],
            proposed_candidate=correct_tables[2]["proposed_candidate"],
            proposal_rank=correct_tables[2]["rank"],
        )
        np.savez_compressed(
            temporary / "validation_policy.npz", query=queries[1], formula=formulas[1],
            baseline_rank=correct_tables[1]["baseline_rank"],
            predicted_benefit=arms["correct_full"]["validation_benefit"],
            predicted_harm=arms["correct_full"]["validation_harm"],
            valid_candidate=correct_tables[1]["valid"],
            proposed_candidate=correct_tables[1]["proposed_candidate"],
            proposal_rank=correct_tables[1]["rank"],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "validation": report["validation"],
        "held_inner": report["held_inner"], "oracle": report["held_inner_candidate_oracle"],
        "paired_inner": comparisons, "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

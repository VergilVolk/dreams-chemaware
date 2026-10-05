"""Conservative finite-action ChemAware policy over mass/rule kernel weights.

This is deliberately candidate-conditioned and is not reported as a shared
embedding.  It chooses among already-audited shared-kernel actions, including
an explicit no-op, using only score geometry observable at inference.  Formula
folds 0--1 fit benefit/harm models, fold 2 freezes the abstention threshold,
fold 3 is an already-used inner audit, and fold 4 stays sealed.
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
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from chemaware_iceberg_direct_core import stable_formula_folds
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


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
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_conservative_action_policy_v1",
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


FEATURE_NAMES = [
    "mass_beta", "rule_beta", "log_candidate_count", "baseline_margin",
    "action_margin", "top_candidate_changed", "official_delta_proposal_vs_baseline",
    "mass_delta_proposal_vs_baseline", "rule_delta_proposal_vs_baseline",
    "action_delta_proposal_vs_baseline", "official_top_score", "mass_top_score",
    "rule_top_score", "mass_official_rank_correlation", "rule_official_rank_correlation",
    "mass_rule_rank_correlation", "official_mass_top_agree", "official_rule_top_agree",
    "mass_rule_top_agree", "proposal_reference_count", "baseline_reference_count",
    "mass_action_fraction", "rule_action_fraction",
]


def array_sha256(value: np.ndarray) -> str:
    """Hash an array together with its shape and dtype for replay audits."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def candidate_channels(scored: dict, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
    official_pair = np.asarray(scored["global"][index], dtype=np.float32)
    mass_pair = np.asarray(scored["mass"][index], dtype=np.float32)
    rule_pair = np.asarray(scored["rule_response"][index], dtype=np.float32)
    official = np.maximum.reduceat(official_pair, pointer[:-1])
    mass = np.maximum.reduceat(mass_pair, pointer[:-1])
    rule = np.maximum.reduceat(rule_pair, pointer[:-1])
    return pointer, official, mass, rule


def build_action_table(
    scored: dict,
    actions: list[tuple[float, float]],
) -> dict[str, np.ndarray]:
    query_count = len(scored["query"])
    action_count = len(actions)
    feature = np.empty((query_count, action_count, len(FEATURE_NAMES)), dtype=np.float32)
    rank = np.empty((query_count, action_count), dtype=np.int16)
    proposed = np.empty((query_count, action_count), dtype=np.int16)
    labels = []
    for query in range(query_count):
        pointer, official, mass, rule = candidate_channels(scored, query)
        molecule_label = np.asarray(scored["labels"][query], dtype=bool)
        labels.append(molecule_label)
        official_top = int(np.argmax(official))
        mass_top = int(np.argmax(mass))
        rule_top = int(np.argmax(rule))
        official_order = np.argsort(-official, kind="stable")
        mass_order = np.argsort(-mass, kind="stable")
        rule_order = np.argsort(-rule, kind="stable")
        official_rank_position = np.empty(len(official_order), dtype=np.float32)
        mass_rank_position = np.empty_like(official_rank_position)
        rule_rank_position = np.empty_like(official_rank_position)
        official_rank_position[official_order] = np.arange(len(official_order))
        mass_rank_position[mass_order] = np.arange(len(mass_order))
        rule_rank_position[rule_order] = np.arange(len(rule_order))
        corr_om = safe_correlation(official_rank_position, mass_rank_position)
        corr_or = safe_correlation(official_rank_position, rule_rank_position)
        corr_mr = safe_correlation(mass_rank_position, rule_rank_position)
        baseline_sorted = np.sort(official)
        baseline_margin = float(baseline_sorted[-1] - baseline_sorted[-2]) if len(official) > 1 else 1.0
        official_pair = np.asarray(scored["global"][query], dtype=np.float32)
        mass_pair = np.asarray(scored["mass"][query], dtype=np.float32)
        rule_pair = np.asarray(scored["rule_response"][query], dtype=np.float32)
        for action, (mass_beta, rule_beta) in enumerate(actions):
            combined_pair = official_pair + mass_beta * mass_pair + rule_beta * rule_pair
            combined = np.maximum.reduceat(combined_pair, pointer[:-1])
            top = int(np.argmax(combined))
            proposed[query, action] = top
            rank[query, action] = strict_rank(combined, molecule_label)
            sorted_score = np.sort(combined)
            action_margin = float(sorted_score[-1] - sorted_score[-2]) if len(combined) > 1 else 1.0
            ref_count = np.diff(pointer)
            total_beta = mass_beta + rule_beta
            feature[query, action] = np.asarray([
                mass_beta, rule_beta, np.log1p(len(combined)), baseline_margin,
                action_margin, float(top != official_top),
                float(official[top] - official[official_top]),
                float(mass[top] - mass[official_top]),
                float(rule[top] - rule[official_top]),
                float(combined[top] - combined[official_top]),
                float(official[official_top]), float(mass[mass_top]), float(rule[rule_top]),
                corr_om, corr_or, corr_mr,
                float(official_top == mass_top), float(official_top == rule_top),
                float(mass_top == rule_top), float(ref_count[top]), float(ref_count[official_top]),
                float(mass_beta / total_beta) if total_beta else 0.0,
                float(rule_beta / total_beta) if total_beta else 0.0,
            ], dtype=np.float32)
        if (query + 1) % 512 == 0:
            print(f"action geometry {query + 1}/{query_count}", flush=True)
    baseline_rank = np.asarray(scored["old_rank"], dtype=np.int16)
    benefit = (baseline_rank[:, None] > 1) & (rank == 1)
    harmful = (baseline_rank[:, None] == 1) & (rank > 1)
    return {
        "feature": feature, "rank": rank, "proposed": proposed,
        "benefit": benefit, "harmful": harmful,
    }


def formula_query_weight(formula: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(formula.astype(str), return_inverse=True)
    count = np.bincount(inverse)
    weight = 1.0 / count[inverse]
    return weight * (len(weight) / weight.sum())


def balanced_event_weight(target: np.ndarray, query_weight: np.ndarray) -> tuple[np.ndarray, float]:
    target = target.astype(bool)
    prevalence = float(np.average(target, weights=query_weight))
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise RuntimeError("event classifier requires both classes")
    balance = np.where(target, 0.5 / prevalence, 0.5 / (1.0 - prevalence))
    return query_weight * balance, prevalence


def fit_classifier(
    feature: np.ndarray,
    target: np.ndarray,
    formula: np.ndarray,
    args: argparse.Namespace,
) -> tuple[HistGradientBoostingClassifier, float]:
    query_weight = formula_query_weight(formula)
    action_weight = np.repeat(query_weight, feature.shape[1])
    flat_target = target.reshape(-1)
    sample_weight, prevalence = balanced_event_weight(flat_target, action_weight)
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=args.learning_rate, max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes, min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization, early_stopping=False,
        random_state=args.seed,
    )
    model.fit(feature.reshape(-1, feature.shape[-1]), flat_target, sample_weight=sample_weight)
    return model, prevalence


def restore_prior(probability: np.ndarray, prevalence: float) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    balanced_logit = np.log(probability / (1.0 - probability))
    prior_logit = np.log(prevalence / (1.0 - prevalence))
    return 1.0 / (1.0 + np.exp(-(balanced_logit + prior_logit)))


def predict_utility(
    benefit_model: HistGradientBoostingClassifier,
    harm_model: HistGradientBoostingClassifier,
    table: dict[str, np.ndarray],
    benefit_prevalence: float,
    harm_prevalence: float,
    risk_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature = table["feature"]
    flat = feature.reshape(-1, feature.shape[-1])
    benefit = restore_prior(benefit_model.predict_proba(flat)[:, 1], benefit_prevalence).reshape(feature.shape[:2])
    harm = restore_prior(harm_model.predict_proba(flat)[:, 1], harm_prevalence).reshape(feature.shape[:2])
    return benefit - float(risk_penalty) * harm, benefit, harm


def policy_ranks(
    table: dict[str, np.ndarray],
    baseline_rank: np.ndarray,
    utility: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_action = np.argmax(utility, axis=1)
    best_value = utility[np.arange(len(utility)), best_action]
    active = best_value >= float(threshold)
    rank = np.array(baseline_rank, copy=True)
    rank[active] = table["rank"][np.arange(len(rank))[active], best_action[active]]
    chosen = np.full(len(rank), -1, dtype=np.int16)
    chosen[active] = best_action[active].astype(np.int16)
    return rank, chosen, best_value


def choose_threshold(
    table: dict[str, np.ndarray], baseline_rank: np.ndarray, formula: np.ndarray,
    utility: np.ndarray, min_selected_formulas: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best_value = np.max(utility, axis=1)
    finite = best_value[np.isfinite(best_value)]
    # Keep the explicit no-op candidate JSON-portable.  A finite value strictly
    # above the observed utility maximum has exactly the same abstention effect
    # as +inf without emitting non-standard ``Infinity`` in report.json.
    no_op_threshold = np.nextafter(np.max(finite), np.inf)
    candidates = np.unique(np.r_[
        np.quantile(finite, np.linspace(0.0, 1.0, 101)), no_op_threshold,
    ])
    rows = []
    for threshold in candidates:
        rank, chosen, _ = policy_ranks(table, baseline_rank, utility, float(threshold))
        active = chosen >= 0
        metric = retrieval(baseline_rank, rank)
        rows.append({
            "threshold": float(threshold), "selected": int(active.sum()),
            "selected_formulas": int(len(np.unique(formula[active].astype(str)))), **metric,
        })
    # No-op is the safety baseline and must remain admissible even though it
    # intentionally selects zero formulas.  The coverage floor applies only to
    # non-empty learned policies.
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


def train_policy(
    train: dict[str, np.ndarray], train_formula: np.ndarray,
    validation: dict[str, np.ndarray], validation_rank: np.ndarray,
    validation_formula: np.ndarray, args: argparse.Namespace,
    *, permute_labels: bool,
) -> dict[str, object]:
    benefit = train["benefit"]
    harmful = train["harmful"]
    if permute_labels:
        permutation = np.random.default_rng(args.seed + 991).permutation(len(benefit))
        benefit = benefit[permutation]
        harmful = harmful[permutation]
    benefit_model, benefit_prevalence = fit_classifier(train["feature"], benefit, train_formula, args)
    harm_model, harm_prevalence = fit_classifier(train["feature"], harmful, train_formula, args)
    utility, p_benefit, p_harm = predict_utility(
        benefit_model, harm_model, validation, benefit_prevalence, harm_prevalence,
        args.risk_penalty,
    )
    selected, threshold_rows = choose_threshold(
        validation, validation_rank, validation_formula, utility, args.min_selected_formulas,
    )
    return {
        "benefit_model": benefit_model, "harm_model": harm_model,
        "benefit_prevalence": benefit_prevalence, "harm_prevalence": harm_prevalence,
        "validation_utility": utility, "validation_p_benefit": p_benefit,
        "validation_p_harm": p_harm, "threshold": float(selected["threshold"]),
        "threshold_selection": selected, "threshold_grid": threshold_rows,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    actions = [
        (float(mass_beta), float(rule_beta))
        for mass_beta in args.beta for rule_beta in args.beta
        if float(mass_beta) > 0.0 or float(rule_beta) > 0.0
    ]
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    pools = [np.flatnonzero((fold == 0) | (fold == 1)), np.flatnonzero(fold == 2), np.flatnonzero(fold == 3)]
    train = identity_balanced_queries(
        pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1), args.train_identities,
    )
    validation = identity_balanced_queries(
        pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2), args.validation_identities,
    )
    inner = identity_balanced_queries(
        pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19), args.max_inner_identities,
    )
    formulas = [body["query_formula"][query].astype(str) for query in (train, validation, inner)]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    kernel_args = SimpleNamespace(**vars(args))
    cache = KernelCache(kernel_args, row_position, variants=("mass", "rule_response"))
    scored = []
    tables = []
    for name, query in zip(("train", "validation", "inner"), (train, validation, inner), strict=True):
        item = score_queries(query, body, official, row_position, cache, ("mass", "rule_response"))
        scored.append(item)
        print(f"completed {name} score geometry rows={len(cache.cache)}", flush=True)
        tables.append(build_action_table(item, actions))

    primary = train_policy(
        tables[0], formulas[0], tables[1], scored[1]["old_rank"], formulas[1], args,
        permute_labels=False,
    )
    control = train_policy(
        tables[0], formulas[0], tables[1], scored[1]["old_rank"], formulas[1], args,
        permute_labels=True,
    )
    validation_rank, validation_action, validation_best = policy_ranks(
        tables[1], scored[1]["old_rank"], primary["validation_utility"],
        primary["threshold"],
    )
    primary_utility, inner_p_benefit, inner_p_harm = predict_utility(
        primary["benefit_model"], primary["harm_model"], tables[2],
        primary["benefit_prevalence"], primary["harm_prevalence"], args.risk_penalty,
    )
    control_utility, _, _ = predict_utility(
        control["benefit_model"], control["harm_model"], tables[2],
        control["benefit_prevalence"], control["harm_prevalence"], args.risk_penalty,
    )
    primary_rank, primary_action, primary_best = policy_ranks(
        tables[2], scored[2]["old_rank"], primary_utility, primary["threshold"],
    )
    control_rank, control_action, _ = policy_ranks(
        tables[2], scored[2]["old_rank"], control_utility, control["threshold"],
    )
    baseline_rank = np.asarray(scored[2]["old_rank"], dtype=np.int16)
    primary_metric = retrieval(baseline_rank, primary_rank)
    control_metric = retrieval(baseline_rank, control_rank)
    absolute_ci = bootstrap(
        formulas[2], baseline_rank, primary_rank, args.bootstrap_draws, args.seed + 500,
    )
    versus_control = paired_rank_comparison(
        primary_rank, control_rank, formulas[2], draws=args.bootstrap_draws, seed=args.seed + 600,
    )
    all_rank = tables[2]["rank"]
    oracle_rank = np.minimum(baseline_rank, np.min(all_rank, axis=1))
    selected_mask = primary_action >= 0
    selected_actions = np.bincount(
        primary_action[selected_mask], minlength=len(actions),
    ) if np.any(selected_mask) else np.zeros(len(actions), dtype=np.int64)
    report = {
        "status": "CHEMAWARE_CONSERVATIVE_ACTION_POLICY_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "folds 0-1 fit; fold 2 threshold; used inner fold 3 evaluation; fold 4 sealed",
        "claim_limit": (
            "Development candidate-action policy, not an embedding result and not external confirmation. "
            "The no-op-aware oracle observes truth and is headroom only."
        ),
        "data": {
            "train_queries": int(len(train)), "validation_queries": int(len(validation)),
            "inner_queries": int(len(inner)), "outer_queries_untouched": int(np.sum(fold == 4)),
            "formula_overlap": 0, "cached_spectrum_rows": int(len(cache.cache)),
            "actions_excluding_no_op": int(len(actions)),
        },
        "method": {
            "actions": [{"mass_beta": x[0], "rule_beta": x[1]} for x in actions],
            "feature_names": FEATURE_NAMES,
            "features_observable_at_inference": True,
            "candidate_identity_or_formula_feature": False,
            "truth_feature": False,
            "explicit_no_op": True,
            "risk_penalty": float(args.risk_penalty),
            "model": "separate shallow histogram-gradient-boosted benefit and harm classifiers",
        },
        "replay_contract": {
            "arguments": {
                key: (str(value.resolve()) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "queries_sha256": {
                "train": array_sha256(train),
                "validation": array_sha256(validation),
                "inner": array_sha256(inner),
            },
            "action_rank_sha256": {
                "train": array_sha256(tables[0]["rank"]),
                "validation": array_sha256(tables[1]["rank"]),
                "inner": array_sha256(tables[2]["rank"]),
            },
        },
        "training_events": {
            "benefit": int(tables[0]["benefit"].sum()),
            "harmful": int(tables[0]["harmful"].sum()),
            "action_rows": int(np.prod(tables[0]["rank"].shape)),
            "benefit_weighted_prevalence": float(primary["benefit_prevalence"]),
            "harm_weighted_prevalence": float(primary["harm_prevalence"]),
        },
        "validation": {
            "primary_threshold_selection": primary["threshold_selection"],
            "primary_threshold_grid": primary["threshold_grid"],
            "label_permutation_threshold_selection": control["threshold_selection"],
        },
        "held_inner": {
            "primary": primary_metric, "label_permutation_control": control_metric,
            "no_op_aware_action_oracle": {
                **retrieval(baseline_rank, oracle_rank),
                "claim_limit": "best action chosen per query after observing truth; headroom only",
            },
            "selected_queries": int(selected_mask.sum()),
            "selected_formulas": int(len(np.unique(formulas[2][selected_mask]))),
            "selected_action_counts": [
                {"mass_beta": actions[i][0], "rule_beta": actions[i][1], "count": int(count)}
                for i, count in enumerate(selected_actions) if count
            ],
        },
        "held_inner_absolute_formula_bootstrap_ci95": absolute_ci,
        "paired_inner": {"primary_minus_label_permutation": versus_control},
        "gates": {
            "absolute_ci_positive": absolute_ci[0] > 0,
            "corrected_exceeds_twice_introduced": (
                primary_metric["corrected_at_1"] > 2 * primary_metric["introduced_at_1"]
            ),
            "beats_label_permutation_ci": versus_control[
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "selected_at_least_50_formulas": int(len(np.unique(formulas[2][selected_mask]))) >= 50,
            "outer_fold_untouched": True,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_action_policy_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "validation_policy.npz", query=validation,
            formula=formulas[1], baseline_rank=scored[1]["old_rank"],
            policy_rank=validation_rank, selected_action=validation_action,
            best_predicted_utility=validation_best,
            predicted_benefit=primary["validation_p_benefit"],
            predicted_harm=primary["validation_p_harm"],
            action_rank=tables[1]["rank"],
        )
        np.savez_compressed(
            temporary / "inner_policy.npz", query=inner, formula=formulas[2],
            baseline_rank=baseline_rank, policy_rank=primary_rank, control_rank=control_rank,
            oracle_rank=oracle_rank, selected_action=primary_action,
            best_predicted_utility=primary_best, predicted_benefit=inner_p_benefit,
            predicted_harm=inner_p_harm, action_rank=tables[2]["rank"],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "training_events": report["training_events"],
        "validation_selection": primary["threshold_selection"],
        "primary": primary_metric, "label_permutation_control": control_metric,
        "oracle": report["held_inner"]["no_op_aware_action_oracle"],
        "paired_inner": report["paired_inner"], "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

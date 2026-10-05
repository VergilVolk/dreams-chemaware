"""Audit a topology-persistent ChemAware candidate policy.

The action grid is not treated as 63 independent decisions.  For each query,
adjacent mass/rule weights that select the same candidate are collapsed into a
connected candidate region.  Benefit and harm are learned per region, and an
explicit no-op is retained.  Folds 0--1 fit the models, fold 2 freezes the
abstention threshold, fold 3 is an already-used inner development audit, and
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
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

import audit_chemaware_conservative_action_policy as action_policy
from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
REGION_FEATURE_NAMES = [
    "region_action_fraction",
    "proposal_action_fraction",
    "region_fraction_of_proposal",
    "same_proposal_neighbor_fraction",
    "region_margin_min",
    "region_margin_mean",
    "region_margin_max",
    "region_margin_std",
    "region_mass_beta_min",
    "region_mass_beta_max",
    "region_mass_beta_span",
    "region_rule_beta_min",
    "region_rule_beta_max",
    "region_rule_beta_span",
    "region_total_beta_min",
    "region_total_beta_max",
    "region_total_beta_mean",
    "region_contains_global_action",
    "region_integrated_margin",
]


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
        default=ROOT / "data/validation/chemaware_persistent_candidate_policy_v1",
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


def action_topology(actions: list[tuple[float, float]]) -> list[list[int]]:
    mass_values = sorted({x[0] for x in actions} | {0.0})
    rule_values = sorted({x[1] for x in actions} | {0.0})
    coordinate = {
        (mass_values.index(mass), rule_values.index(rule)): index
        for index, (mass, rule) in enumerate(actions)
    }
    neighbors: list[list[int]] = [[] for _ in actions]
    for (left, right), index in coordinate.items():
        for delta_left, delta_right in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = coordinate.get((left + delta_left, right + delta_right))
            if other is not None:
                neighbors[index].append(other)
    return neighbors


def connected_candidate_regions(
    proposed: np.ndarray,
    changed: np.ndarray,
    neighbors: list[list[int]],
) -> list[np.ndarray]:
    """Return same-candidate 4-neighbor components, excluding official-top regions."""
    unseen = set(np.flatnonzero(changed).tolist())
    regions: list[np.ndarray] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        candidate = int(proposed[start])
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for other in neighbors[current]:
                if other in unseen and int(proposed[other]) == candidate:
                    unseen.remove(other)
                    stack.append(other)
                    component.append(other)
        regions.append(np.asarray(sorted(component), dtype=np.int16))
    return regions


def region_table(
    table: dict[str, np.ndarray], actions: list[tuple[float, float]], global_action: int,
) -> dict[str, np.ndarray]:
    query_count, action_count, base_dimension = table["feature"].shape
    feature = np.zeros(
        (query_count, action_count, base_dimension + len(REGION_FEATURE_NAMES)), dtype=np.float32,
    )
    rank = np.ones((query_count, action_count), dtype=np.int16)
    representative = np.full((query_count, action_count), -1, dtype=np.int16)
    valid = np.zeros((query_count, action_count), dtype=bool)
    region_size = np.zeros((query_count, action_count), dtype=np.int16)
    neighbors = action_topology(actions)
    action_array = np.asarray(actions, dtype=np.float32)
    for query in range(query_count):
        proposed = table["proposed"][query]
        changed = table["feature"][query, :, 5] > 0.5
        regions = connected_candidate_regions(proposed, changed, neighbors)
        proposal_count = {
            int(candidate): int(np.sum(proposed == candidate))
            for candidate in np.unique(proposed)
        }
        for slot, component in enumerate(regions):
            margins = table["feature"][query, component, 4]
            # A deterministic robust representative: largest winning margin,
            # then smallest total chemical weight, then stable action index.
            ordering = sorted(
                map(int, component),
                key=lambda index: (
                    -float(table["feature"][query, index, 4]),
                    float(sum(actions[index])), index,
                ),
            )
            selected = ordering[0]
            candidate = int(proposed[selected])
            local_support = []
            component_set = set(map(int, component))
            for index in component_set:
                degree = len(neighbors[index])
                local_support.append(
                    sum(other in component_set for other in neighbors[index]) / max(1, degree)
                )
            betas = action_array[component]
            totals = betas.sum(axis=1)
            extra = np.asarray([
                len(component) / action_count,
                proposal_count[candidate] / action_count,
                len(component) / proposal_count[candidate],
                float(np.mean(local_support)),
                float(np.min(margins)), float(np.mean(margins)),
                float(np.max(margins)), float(np.std(margins)),
                float(np.min(betas[:, 0])), float(np.max(betas[:, 0])),
                float(np.ptp(betas[:, 0])),
                float(np.min(betas[:, 1])), float(np.max(betas[:, 1])),
                float(np.ptp(betas[:, 1])),
                float(np.min(totals)), float(np.max(totals)), float(np.mean(totals)),
                float(global_action in component_set),
                float(np.mean(margins) * np.sqrt(len(component))),
            ], dtype=np.float32)
            feature[query, slot] = np.concatenate((table["feature"][query, selected], extra))
            rank[query, slot] = table["rank"][query, selected]
            representative[query, slot] = selected
            valid[query, slot] = True
            region_size[query, slot] = len(component)
    baseline_rank = np.asarray(table["baseline_rank"], dtype=np.int16)
    return {
        "feature": feature,
        "rank": rank,
        "representative": representative,
        "valid": valid,
        "region_size": region_size,
        "benefit": valid & (baseline_rank[:, None] > 1) & (rank == 1),
        "harmful": valid & (baseline_rank[:, None] == 1) & (rank > 1),
    }


def fit_classifier(
    table: dict[str, np.ndarray], target_name: str, formula: np.ndarray,
    args: argparse.Namespace, *, permute: bool,
) -> tuple[HistGradientBoostingClassifier, float]:
    valid = table["valid"]
    target = table[target_name][valid].astype(bool)
    feature = table["feature"][valid]
    query_count = valid.sum(axis=1)
    query_weight = action_policy.formula_query_weight(formula)
    event_weight = np.concatenate([
        np.full(count, query_weight[index] / max(1, count), dtype=np.float64)
        for index, count in enumerate(query_count)
    ])
    if permute:
        target = target[np.random.default_rng(args.seed + (991 if target_name == "benefit" else 997)).permutation(len(target))]
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
    table: dict[str, np.ndarray], benefit_model: HistGradientBoostingClassifier,
    harm_model: HistGradientBoostingClassifier, benefit_prevalence: float,
    harm_prevalence: float, risk_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = table["valid"].shape
    benefit = np.zeros(shape, dtype=np.float64)
    harm = np.ones(shape, dtype=np.float64)
    valid_feature = table["feature"][table["valid"]]
    benefit[table["valid"]] = action_policy.restore_prior(
        benefit_model.predict_proba(valid_feature)[:, 1], benefit_prevalence,
    )
    harm[table["valid"]] = action_policy.restore_prior(
        harm_model.predict_proba(valid_feature)[:, 1], harm_prevalence,
    )
    utility = benefit - float(risk_penalty) * harm
    utility[~table["valid"]] = -np.inf
    return utility, benefit, harm


def ranks_at_threshold(
    table: dict[str, np.ndarray], baseline_rank: np.ndarray,
    utility: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_region = np.argmax(utility, axis=1)
    best_value = utility[np.arange(len(best_region)), best_region]
    active = best_value >= threshold
    rank = np.array(baseline_rank, copy=True)
    rank[active] = table["rank"][np.arange(len(rank))[active], best_region[active]]
    chosen = np.full(len(rank), -1, dtype=np.int16)
    chosen[active] = best_region[active].astype(np.int16)
    return rank, chosen, best_value


def choose_threshold(
    table: dict[str, np.ndarray], baseline_rank: np.ndarray, formula: np.ndarray,
    utility: np.ndarray, min_selected_formulas: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best = np.max(utility, axis=1)
    finite = best[np.isfinite(best)]
    candidates = np.unique(np.r_[
        np.quantile(finite, np.linspace(0.0, 1.0, 101)),
        np.nextafter(np.max(finite), np.inf),
    ])
    rows = []
    for threshold in candidates:
        rank, chosen, _ = ranks_at_threshold(table, baseline_rank, utility, float(threshold))
        active = chosen >= 0
        rows.append({
            "threshold": float(threshold),
            "selected": int(active.sum()),
            "selected_formulas": int(len(np.unique(formula[active].astype(str)))),
            **retrieval(baseline_rank, rank),
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


def train_models(
    train: dict[str, np.ndarray], train_formula: np.ndarray,
    validation: dict[str, np.ndarray], validation_rank: np.ndarray,
    validation_formula: np.ndarray, args: argparse.Namespace, *, permute: bool,
) -> dict[str, object]:
    benefit_model, benefit_prevalence = fit_classifier(
        train, "benefit", train_formula, args, permute=permute,
    )
    harm_model, harm_prevalence = fit_classifier(
        train, "harmful", train_formula, args, permute=permute,
    )
    utility, p_benefit, p_harm = predict(
        validation, benefit_model, harm_model, benefit_prevalence,
        harm_prevalence, args.risk_penalty,
    )
    selected, grid = choose_threshold(
        validation, validation_rank, validation_formula, utility,
        args.min_selected_formulas,
    )
    return {
        "benefit_model": benefit_model, "harm_model": harm_model,
        "benefit_prevalence": benefit_prevalence,
        "harm_prevalence": harm_prevalence,
        "validation_utility": utility, "validation_p_benefit": p_benefit,
        "validation_p_harm": p_harm, "threshold": float(selected["threshold"]),
        "threshold_selection": selected, "threshold_grid": grid,
    }


def main() -> None:
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
        identity_balanced_queries(
            pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1),
            args.train_identities,
        ),
        identity_balanced_queries(
            pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2),
            args.validation_identities,
        ),
        identity_balanced_queries(
            pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19),
            args.max_inner_identities,
        ),
    ]
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    kernel_args = SimpleNamespace(**vars(args))
    cache = KernelCache(kernel_args, row_position, variants=("mass", "rule_response"))
    scored = []
    action_tables = []
    region_tables = []
    for name, query in zip(("train", "validation", "inner"), queries, strict=True):
        score = score_queries(query, body, official, row_position, cache, ("mass", "rule_response"))
        scored.append(score)
        print(f"completed {name} score geometry rows={len(cache.cache)}", flush=True)
        action_table = action_policy.build_action_table(score, actions)
        action_table["baseline_rank"] = np.asarray(score["old_rank"], dtype=np.int16)
        action_tables.append(action_table)
        region_tables.append(region_table(action_table, actions, global_action))
        print(
            f"completed {name} regions={int(region_tables[-1]['valid'].sum())}",
            flush=True,
        )

    primary = train_models(
        region_tables[0], formulas[0], region_tables[1], scored[1]["old_rank"],
        formulas[1], args, permute=False,
    )
    control = train_models(
        region_tables[0], formulas[0], region_tables[1], scored[1]["old_rank"],
        formulas[1], args, permute=True,
    )
    primary_utility, inner_benefit, inner_harm = predict(
        region_tables[2], primary["benefit_model"], primary["harm_model"],
        primary["benefit_prevalence"], primary["harm_prevalence"], args.risk_penalty,
    )
    control_utility, _, _ = predict(
        region_tables[2], control["benefit_model"], control["harm_model"],
        control["benefit_prevalence"], control["harm_prevalence"], args.risk_penalty,
    )
    baseline_rank = np.asarray(scored[2]["old_rank"], dtype=np.int16)
    primary_rank, primary_region, primary_best = ranks_at_threshold(
        region_tables[2], baseline_rank, primary_utility, primary["threshold"],
    )
    control_rank, control_region, _ = ranks_at_threshold(
        region_tables[2], baseline_rank, control_utility, control["threshold"],
    )
    primary_metric = retrieval(baseline_rank, primary_rank)
    control_metric = retrieval(baseline_rank, control_rank)
    absolute_ci = bootstrap(
        formulas[2], baseline_rank, primary_rank, args.bootstrap_draws, args.seed + 500,
    )
    versus_control = paired_rank_comparison(
        primary_rank, control_rank, formulas[2], draws=args.bootstrap_draws,
        seed=args.seed + 600,
    )
    with np.load(args.action_policy / "inner_policy.npz", allow_pickle=False) as loaded:
        old_action_query = np.asarray(loaded["query"])
        old_action_rank = np.asarray(loaded["policy_rank"])
    if not np.array_equal(old_action_query, queries[2]):
        raise RuntimeError("action-policy comparator query set drifted")
    versus_action_policy = paired_rank_comparison(
        primary_rank, old_action_rank, formulas[2], draws=args.bootstrap_draws,
        seed=args.seed + 700,
    )
    valid_rank = np.where(region_tables[2]["valid"], region_tables[2]["rank"], 32767)
    oracle_rank = np.minimum(baseline_rank, np.min(valid_rank, axis=1))
    selected = primary_region >= 0
    selected_action = np.full(len(primary_region), -1, dtype=np.int16)
    selected_action[selected] = region_tables[2]["representative"][
        np.arange(len(selected))[selected], primary_region[selected]
    ]
    report = {
        "status": "CHEMAWARE_PERSISTENT_CANDIDATE_POLICY_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "folds 0-1 fit; fold 2 threshold; used inner fold 3; fold 4 sealed",
        "claim_limit": (
            "Development candidate-region policy, not embedding performance or external confirmation. "
            "The oracle observes truth and is headroom only."
        ),
        "method": {
            "unit": "4-neighbor connected region of weights proposing the same non-baseline candidate",
            "feature_names": action_policy.FEATURE_NAMES + REGION_FEATURE_NAMES,
            "features_observable_at_inference": True,
            "truth_feature": False,
            "candidate_identity_or_formula_feature": False,
            "explicit_no_op": True,
            "region_balanced_query_weight": True,
            "representative": "maximum action margin; tie by minimum total beta and action index",
            "risk_penalty": float(args.risk_penalty),
            "global_action": {
                "index": int(global_action), "mass_beta": args.global_mass_beta,
                "rule_beta": args.global_rule_beta,
            },
        },
        "data": {
            "train_queries": int(len(queries[0])),
            "validation_queries": int(len(queries[1])),
            "inner_queries": int(len(queries[2])),
            "outer_queries_untouched": int(np.sum(fold == 4)),
            "formula_overlap": 0,
            "region_events": [int(table["valid"].sum()) for table in region_tables],
            "queries_with_regions": [int(np.sum(table["valid"].any(axis=1))) for table in region_tables],
        },
        "training_events": {
            "benefit": int(region_tables[0]["benefit"].sum()),
            "harmful": int(region_tables[0]["harmful"].sum()),
            "benefit_weighted_prevalence": float(primary["benefit_prevalence"]),
            "harm_weighted_prevalence": float(primary["harm_prevalence"]),
        },
        "validation": {
            "primary_threshold_selection": primary["threshold_selection"],
            "label_permutation_threshold_selection": control["threshold_selection"],
        },
        "held_inner": {
            "primary": primary_metric,
            "label_permutation_control": control_metric,
            "no_op_aware_region_oracle": {
                **retrieval(baseline_rank, oracle_rank),
                "claim_limit": "best candidate region chosen after observing truth; headroom only",
            },
            "selected_queries": int(selected.sum()),
            "selected_formulas": int(len(np.unique(formulas[2][selected]))),
        },
        "held_inner_absolute_formula_bootstrap_ci95": absolute_ci,
        "paired_inner": {
            "primary_minus_label_permutation": versus_control,
            "primary_minus_reproducible_action_policy_v3": versus_action_policy,
        },
        "gates": {
            "absolute_ci_positive": absolute_ci[0] > 0,
            "corrected_exceeds_twice_introduced": (
                primary_metric["corrected_at_1"] > 2 * primary_metric["introduced_at_1"]
            ),
            "beats_label_permutation_ci": versus_control[
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "beats_action_policy_v3_ci": versus_action_policy[
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
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
            "queries_sha256": {
                name: array_sha256(query)
                for name, query in zip(("train", "validation", "inner"), queries, strict=True)
            },
            "region_rank_sha256": {
                name: array_sha256(table["rank"])
                for name, table in zip(("train", "validation", "inner"), region_tables, strict=True)
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
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_persistent_policy_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz", query=queries[2], formula=formulas[2],
            baseline_rank=baseline_rank, policy_rank=primary_rank,
            control_rank=control_rank, oracle_rank=oracle_rank,
            selected_region=primary_region, selected_action=selected_action,
            best_predicted_utility=primary_best, predicted_benefit=inner_benefit,
            predicted_harm=inner_harm, valid_region=region_tables[2]["valid"],
            region_rank=region_tables[2]["rank"],
            representative_action=region_tables[2]["representative"],
            region_size=region_tables[2]["region_size"],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "validation": report["validation"],
        "primary": primary_metric,
        "control": control_metric,
        "paired": report["paired_inner"],
        "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Nested formula-OOF rotation-aware BioAware v4 development."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tasks") not in sys.path:
    sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_rotation_listwise_headroom import (  # noqa: E402
    CONFIGS,
    EPS,
    FEATURES,
    aggregate_predictions,
    fit_ranker,
    top,
)


POLICIES = list(
    itertools.product(
        [0.01, 0.025, 0.05, 0.10, 0.20],
        [0.025, 0.05, 0.10, 0.20, 1.0],
        [1, 2],
    )
)


def assigned_fold(value: str, folds: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()
    return int(digest[:16], 16) % folds


def score_decisions(
    frame: pd.DataFrame, policy: tuple[float, float, int]
) -> pd.DataFrame:
    alpha, max_margin, min_support = policy
    scored = frame.copy()
    scored["fusion"] = scored["spectral_score"] + alpha * scored["model_z_mean"]
    rows: list[dict] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline, base_unique = top(group, "spectral_score")
        proposed, proposed_unique = top(group, "fusion")
        ordered = group.sort_values(
            ["spectral_score", "candidate_id"], ascending=[False, True]
        )
        margin = float(ordered.iloc[0].spectral_score - ordered.iloc[1].spectral_score)
        proposed_row = group[group["candidate_id"].astype(str) == proposed].iloc[0]
        baseline_row = group[group["candidate_id"].astype(str) == baseline].iloc[0]
        advantages = [
            proposed_row.context_support_fraction - baseline_row.context_support_fraction,
            proposed_row.direction_support_fraction - baseline_row.direction_support_fraction,
            proposed_row.known_edge_best_bottleneck - baseline_row.known_edge_best_bottleneck,
            proposed_row.predicted_edge_best_bottleneck - baseline_row.predicted_edge_best_bottleneck,
            proposed_row.decoder_score_scaled - baseline_row.decoder_score_scaled,
            proposed_row.rule_jaccard_idf - baseline_row.rule_jaccard_idf,
            proposed_row.smn_best_bottleneck - baseline_row.smn_best_bottleneck,
            proposed_row.rt_score - baseline_row.rt_score,
        ]
        support = int(sum(value > EPS for value in advantages))
        intervene = bool(
            proposed != baseline and proposed_unique
            and margin <= max_margin + EPS and support >= min_support
        )
        final = proposed if intervene else baseline
        baseline_correct = bool(base_unique and baseline == truth)
        final_correct = final == truth
        rows.append(
            {
                "query_id": str(query_id),
                "truth_candidate_id": truth,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "baseline_candidate_id": baseline,
                "proposed_candidate_id": proposed,
                "final_candidate_id": final,
                "baseline_correct": baseline_correct,
                "final_correct": final_correct,
                "corrected": (not baseline_correct) and final_correct,
                "introduced": baseline_correct and not final_correct,
                "intervene": intervene,
                "spectral_margin": margin,
                "support_count": support,
            }
        )
    return pd.DataFrame(rows)


def recipe_key(decisions: pd.DataFrame) -> tuple[int, int, int, int]:
    corrected = int(decisions.corrected.sum())
    introduced = int(decisions.introduced.sum())
    interventions = int(decisions.intervene.sum())
    return corrected - 2 * introduced, corrected, -introduced, -interventions


def inner_predictions(
    train: pd.DataFrame,
    config: dict,
    outer_fold: int,
    inner_folds: int,
    seed: int,
) -> pd.DataFrame:
    formula = train[["truth_formula"]].drop_duplicates().copy()
    formula["inner_fold"] = formula.truth_formula.map(
        lambda value: assigned_fold(
            str(value), inner_folds, f"bioaware-v4-inner-{seed}-{outer_fold}"
        )
    )
    pieces: list[pd.DataFrame] = []
    for inner_fold in range(inner_folds):
        validation_formula = set(
            formula.loc[formula.inner_fold == inner_fold, "truth_formula"]
        )
        fit = train[~train.truth_formula.isin(validation_formula)]
        validation = train[train.truth_formula.isin(validation_formula)]
        if validation.empty:
            raise RuntimeError("empty inner formula fold")
        model = fit_ranker(
            fit, config, seed + outer_fold * 100 + inner_fold
        )
        pieces.append(aggregate_predictions(model, validation))
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("inner query/candidate duplication")
    if result.query_id.nunique() != train.query_id.nunique():
        raise RuntimeError("inner query coverage mismatch")
    return result


def select_recipe(
    train: pd.DataFrame,
    outer_fold: int,
    inner_folds: int,
    seed: int,
) -> tuple[dict, tuple[float, float, int], dict]:
    options: list[tuple[tuple[int, int, int, int], int, dict, tuple, dict]] = []
    for config_index, config in enumerate(CONFIGS):
        predictions = inner_predictions(
            train, config, outer_fold, inner_folds, seed + 1000 * config_index
        )
        for policy in POLICIES:
            evaluated = score_decisions(predictions, policy)
            key = recipe_key(evaluated)
            options.append(
                (
                    key, -config_index, config, policy,
                    {
                        "risk_weighted_net": key[0],
                        "corrected": key[1],
                        "introduced": -key[2],
                        "interventions": -key[3],
                    },
                )
            )
    best = max(options, key=lambda item: (item[0], item[1]))
    if best[0][0] <= 0 or best[0][1] <= 0:
        return CONFIGS[0], (0.0, 0.0, 99), {
            "selected_no_intervention": True,
            **best[4],
        }
    return best[2], best[3], {"selected_no_intervention": False, **best[4]}


def bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    groups = {key: value for key, value in frame.groupby("truth_formula", sort=True)}
    formulas = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for index in range(repeats):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        sample = pd.concat([groups[str(key)] for key in sampled], ignore_index=True)
        values[index] = sample.final_correct.mean() - sample.baseline_correct.mean()
    return {
        "mean": float(frame.final_correct.mean() - frame.baseline_correct.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": len(formulas),
        "resamples": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger", type=Path,
        default=Path("data/validation/bioaware_typed_candidate_ledger_v1_20260830/candidate_evidence_typed.csv.gz"),
    )
    parser.add_argument(
        "--rotations", type=Path,
        default=Path("data/validation/bioaware_typed_candidate_ledger_v1_20260830/rotation_typed_features.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_rotation_listwise_v4_20260830"),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.ledger)
    ledger["decoder_score_scaled"] = (ledger.decoder_score.clip(-1, 1) + 1) / 2
    rotations = pd.read_csv(args.rotations)
    static_columns = [
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "spectral_score", "decoder_score_scaled", "rule_jaccard_idf",
        "sparse_rule_overlap", "known_edge_best_bottleneck",
        "predicted_edge_best_bottleneck", "smn_best_bottleneck", "rt_score",
    ]
    frame = rotations.merge(
        ledger[static_columns], on=["query_id", "candidate_id"], validate="many_to_one"
    )
    frame["rotation_unit"] = frame.rotation_fold.astype(str) + "|" + frame.query_id.astype(str)
    if frame[FEATURES].isna().any().any():
        raise RuntimeError("feature matrix has missing values")
    formula = frame[["truth_formula"]].drop_duplicates().copy()
    formula["outer_fold"] = formula.truth_formula.map(
        lambda value: assigned_fold(
            str(value), args.outer_folds, f"bioaware-v4-outer-{args.seed}"
        )
    )
    frame = frame.merge(formula, on="truth_formula", validate="many_to_one")

    candidate_parts: list[pd.DataFrame] = []
    transition_parts: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_fold in range(args.outer_folds):
        train = frame[frame.outer_fold != outer_fold]
        test = frame[frame.outer_fold == outer_fold]
        config, policy, inner = select_recipe(
            train, outer_fold, args.inner_folds, args.seed
        )
        model = fit_ranker(train, config, args.seed + 10000 + outer_fold)
        candidates = aggregate_predictions(model, test)
        decisions = score_decisions(candidates, policy)
        candidates["outer_fold"] = outer_fold
        decisions["outer_fold"] = outer_fold
        candidate_parts.append(candidates)
        transition_parts.append(decisions)
        folds.append(
            {
                "outer_fold": outer_fold,
                "queries": int(decisions.query_id.nunique()),
                "formulas": int(decisions.truth_formula.nunique()),
                "model": config,
                "policy": list(policy),
                "inner_selection": inner,
                "corrected": int(decisions.corrected.sum()),
                "introduced": int(decisions.introduced.sum()),
                "interventions": int(decisions.intervene.sum()),
            }
        )
        print(
            f"[v4 outer {outer_fold}] model={config['name']} policy={policy} "
            f"C/I={folds[-1]['corrected']}/{folds[-1]['introduced']}",
            flush=True,
        )
    candidates = pd.concat(candidate_parts, ignore_index=True)
    result = pd.concat(transition_parts, ignore_index=True)
    if result.query_id.duplicated().any() or result.query_id.nunique() != 117:
        raise RuntimeError("outer OOF query coverage mismatch")
    corrected = int(result.corrected.sum())
    introduced = int(result.introduced.sum())
    baseline = float(result.baseline_correct.mean())
    final = float(result.final_correct.mean())
    ci = bootstrap(result, args.bootstrap_resamples, args.seed)
    discordant = corrected + introduced
    pvalue = float(
        binomtest(min(corrected, introduced), discordant, 0.5).pvalue
    ) if discordant else 1.0
    candidate_path = args.output_dir / "candidate_oof_scores.csv.gz"
    transition_path = args.output_dir / "query_oof_transitions.csv.gz"
    candidates.to_csv(candidate_path, index=False, compression="gzip")
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_rotation_listwise_v4_development_complete",
        "formal": True,
        "protocol": "nested formula-OOF rotation-level LambdaRank; query-level aggregation; risk gate",
        "queries": int(len(result)),
        "rotation_units": int(frame.rotation_unit.nunique()),
        "baseline_recall1": baseline,
        "recall1": final,
        "delta_recall1": final - baseline,
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "interventions": int(result.intervene.sum()),
        "formula_cluster_bootstrap": ci,
        "mcnemar_exact_p": pvalue,
        "folds": folds,
        "gates": {
            "risk_weighted_net_positive": corrected - 2 * introduced > 0,
            "formula_cluster_ci_positive": ci["ci_low"] > 0,
            "mcnemar_p_lt_0_05": pvalue < 0.05,
        },
        "contracts": {
            "P2b": "forbidden", "phenotype": "forbidden",
            "model_and_policy_selection": "inner formula OOF only",
            "evaluation": "outer formula OOF only",
            "RP": "not opened",
        },
        "claim_limit": "Consumed development; external validation is required before an SOTA claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

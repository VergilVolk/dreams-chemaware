#!/usr/bin/env python3
"""Cross-fit unresolved response information against held-out known identities.

The exact-InChIKey anchors are deterministically split into two disjoint folds.
For each fold swap, one half supplies a known-only response score and the other
half supplies held-out empirical response truth.  Unresolved--unresolved WSE
edges are disjoint from both known halves.  The primary endpoint asks whether a
fixed, truth-blind combination of train-known and unresolved scores improves
within-target-study contrast ranking over train-known alone after removing the
query- and candidate-axis main effects inside each source pair.

This is an internal development gate.  It does not make the study-level
effects, library identities, or source-pair dyads externally independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_contrast_atlas_known_anchor_unresolved_test import (
    add_response_weight,
    select_known_entities,
    signed_cosine,
)


DEFAULT_SPLIT_SALT = "contrast-atlas-known-crossfit-v1|20261031"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inchikey_fold(inchikey: str, salt: str = DEFAULT_SPLIT_SALT) -> int:
    digest = hashlib.sha256(f"{salt}|{inchikey}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2


def fold_known_scores(effects: pd.DataFrame, selected: pd.DataFrame, salt: str) -> pd.DataFrame:
    known = effects.merge(
        selected[["study_id", "entity_id", "library_inchikey"]],
        on=["study_id", "entity_id"],
        how="inner",
        validate="many_to_one",
    )
    known["known_fold"] = known["library_inchikey"].astype(str).map(
        lambda value: inchikey_fold(value, salt)
    )
    studies = sorted(known["study_id"].astype(str).unique())
    rows: list[dict[str, object]] = []
    for left_index, left_study in enumerate(studies):
        left_all = known.loc[known["study_id"].astype(str).eq(left_study)]
        for right_study in studies[left_index + 1 :]:
            right_all = known.loc[known["study_id"].astype(str).eq(right_study)]
            common = sorted(
                set(left_all["library_inchikey"].astype(str))
                & set(right_all["library_inchikey"].astype(str))
            )
            if not common:
                continue
            left_common = left_all.loc[left_all["library_inchikey"].isin(common)]
            right_common = right_all.loc[right_all["library_inchikey"].isin(common)]
            for left_contrast, left in left_common.groupby("contrast_id", sort=True):
                left = left[["library_inchikey", "response_weight", "known_fold"]]
                for right_contrast, right in right_common.groupby("contrast_id", sort=True):
                    joined = left.merge(
                        right[["library_inchikey", "response_weight"]],
                        on="library_inchikey",
                        how="inner",
                        suffixes=("_left", "_right"),
                        validate="one_to_one",
                    ).dropna(subset=["response_weight_left", "response_weight_right"])
                    row: dict[str, object] = {
                        "left_study_id": str(left_study),
                        "left_contrast_id": str(left_contrast),
                        "right_study_id": str(right_study),
                        "right_contrast_id": str(right_contrast),
                    }
                    for fold in (0, 1):
                        block = joined.loc[joined["known_fold"].eq(fold)].copy()
                        finite = (
                            np.isfinite(pd.to_numeric(
                                block["response_weight_left"], errors="coerce"
                            ))
                            & np.isfinite(pd.to_numeric(
                                block["response_weight_right"], errors="coerce"
                            ))
                        )
                        block = block.loc[finite]
                        row[f"known_fold_{fold}_support"] = int(len(block))
                        row[f"known_fold_{fold}_score"] = signed_cosine(
                            block["response_weight_left"].to_numpy(float),
                            block["response_weight_right"].to_numpy(float),
                            np.ones(len(block), dtype=float),
                        )
                        row[f"known_fold_{fold}_inchikeys"] = ";".join(
                            sorted(block["library_inchikey"].astype(str))
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def fold_swaps(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for truth_fold in (0, 1):
        train_fold = 1 - truth_fold
        block = frame.copy()
        block["truth_fold"] = truth_fold
        block["train_fold"] = train_fold
        block["train_known_support"] = block[f"known_fold_{train_fold}_support"]
        block["heldout_known_support"] = block[f"known_fold_{truth_fold}_support"]
        block["train_known_score"] = block[f"known_fold_{train_fold}_score"]
        block["heldout_known_truth"] = block[f"known_fold_{truth_fold}_score"]
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def double_center(frame: pd.DataFrame, column: str) -> pd.Series:
    pair = ["truth_fold", "left_study_id", "right_study_id"]
    query_axis = pair + ["left_contrast_id"]
    candidate_axis = pair + ["right_contrast_id"]
    pair_mean = frame.groupby(pair, sort=False)[column].transform("mean")
    query_mean = frame.groupby(query_axis, sort=False)[column].transform("mean")
    candidate_mean = frame.groupby(candidate_axis, sort=False)[column].transform("mean")
    return frame[column] - query_mean - candidate_mean + pair_mean


def equal_norm_additive(frame: pd.DataFrame) -> pd.Series:
    """Truth-blind equal-L2 combination within each fold and source pair."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    keys = ["truth_fold", "left_study_id", "right_study_id"]
    for _, index in frame.groupby(keys, sort=False).groups.items():
        idx = np.asarray(list(index), dtype=int)
        known = frame.loc[idx, "train_known_interaction"].to_numpy(float)
        unresolved = frame.loc[idx, "unresolved_interaction"].to_numpy(float)
        known_norm = float(np.sqrt(np.mean(known * known)))
        unresolved_norm = float(np.sqrt(np.mean(unresolved * unresolved)))
        known_unit = known / known_norm if known_norm > 0 else np.zeros_like(known)
        unresolved_unit = (
            unresolved / unresolved_norm if unresolved_norm > 0 else np.zeros_like(unresolved)
        )
        result.loc[idx] = (known_unit + unresolved_unit) / np.sqrt(2.0)
    return result


def pairwise_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    block_keys = [
        "truth_fold", "left_study_id", "left_contrast_id", "right_study_id"
    ]
    predictors = {
        "train_known": "train_known_interaction",
        "unresolved_only": "unresolved_interaction",
        "train_known_plus_unresolved_fixed": "combined_fixed_interaction",
    }
    for key, block in frame.groupby(block_keys, sort=True):
        block = block.sort_values("right_contrast_id", kind="mergesort").reset_index(drop=True)
        if len(block) < 2:
            continue
        truth = block["heldout_known_truth_interaction"].to_numpy(float)
        for first in range(len(block)):
            for second in range(first + 1, len(block)):
                truth_delta = float(truth[first] - truth[second])
                if not np.isfinite(truth_delta) or truth_delta == 0:
                    continue
                row: dict[str, object] = {
                    "truth_fold": int(key[0]),
                    "left_study_id": str(key[1]),
                    "left_contrast_id": str(key[2]),
                    "right_study_id": str(key[3]),
                    "right_contrast_a": str(block.loc[first, "right_contrast_id"]),
                    "right_contrast_b": str(block.loc[second, "right_contrast_id"]),
                    "heldout_truth_delta": truth_delta,
                    "heldout_truth_order": int(np.sign(truth_delta)),
                }
                for label, column in predictors.items():
                    delta = float(block.loc[first, column] - block.loc[second, column])
                    row[f"{label}_delta"] = delta
                    row[f"{label}_evaluable"] = bool(np.isfinite(delta) and delta != 0)
                    row[f"{label}_concordant"] = (
                        bool(np.sign(delta) == np.sign(truth_delta))
                        if np.isfinite(delta) and delta != 0
                        else False
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def endpoint_by_dyad(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    arms = (
        "train_known", "unresolved_only", "train_known_plus_unresolved_fixed"
    )
    for (left, right), block in pairs.groupby(
        ["left_study_id", "right_study_id"], sort=True
    ):
        row: dict[str, object] = {
            "left_study_id": left,
            "right_study_id": right,
            "source_dyad": f"{left}||{right}",
        }
        for arm in arms:
            evaluable = block[f"{arm}_evaluable"].astype(bool)
            values = block.loc[evaluable, f"{arm}_concordant"].astype(float)
            row[f"{arm}_comparisons"] = int(len(values))
            row[f"{arm}_concordance"] = float(values.mean()) if len(values) else np.nan
        row["fixed_combination_minus_train_known"] = (
            row["train_known_plus_unresolved_fixed_concordance"]
            - row["train_known_concordance"]
            if np.isfinite(row["train_known_plus_unresolved_fixed_concordance"])
            and np.isfinite(row["train_known_concordance"])
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_dyad_delta(
    dyads: pd.DataFrame, replicates: int, seed: int
) -> dict[str, object]:
    values = dyads["fixed_combination_minus_train_known"].dropna().to_numpy(float)
    if not len(values):
        return {"dyads": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "dyads": int(len(values)),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "replicates": int(replicates),
    }


def sign_flip_delta(
    dyads: pd.DataFrame, replicates: int, seed: int
) -> dict[str, object]:
    values = dyads["fixed_combination_minus_train_known"].dropna().to_numpy(float)
    if not len(values):
        return {"dyads": 0, "one_sided_p": None}
    observed = float(values.mean())
    if len(values) <= 20:
        configurations = np.arange(1 << len(values), dtype=np.uint64)[:, None]
        bits = (
            configurations
            >> np.arange(len(values), dtype=np.uint64)[None, :]
        ) & np.uint64(1)
        signs = bits.astype(np.float64) * 2.0 - 1.0
        null = (signs * values).mean(axis=1)
        method = "exact_all_sign_configurations"
        one_sided_p = float(np.mean(null >= observed - 1e-15))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(replicates, len(values)))
        null = (signs * values).mean(axis=1)
        method = "monte_carlo_plus_one"
        one_sided_p = float((1 + np.sum(null >= observed)) / (replicates + 1))
    return {
        "dyads": int(len(values)),
        "observed_mean_delta": observed,
        "method": method,
        "one_sided_p": one_sided_p,
        "null_configurations_or_replicates": int(len(null)),
        "requested_monte_carlo_replicates": int(replicates),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_median": float(np.quantile(null, 0.5)),
        "null_q975": float(np.quantile(null, 0.975)),
    }


def leave_out_sensitivity(dyads: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    finite = dyads.dropna(subset=["fixed_combination_minus_train_known"]).copy()
    rows: list[dict[str, object]] = []
    studies = sorted(set(finite["left_study_id"]) | set(finite["right_study_id"]))
    for study in studies:
        retained = finite.loc[
            ~finite["left_study_id"].eq(study) & ~finite["right_study_id"].eq(study)
        ]
        rows.append(
            {
                "left_out_type": "study",
                "left_out_id": study,
                "retained_dyads": int(len(retained)),
                "mean_delta": (
                    float(retained["fixed_combination_minus_train_known"].mean())
                    if len(retained) else np.nan
                ),
            }
        )
    for dyad in finite["source_dyad"]:
        retained = finite.loc[~finite["source_dyad"].eq(dyad)]
        rows.append(
            {
                "left_out_type": "dyad",
                "left_out_id": dyad,
                "retained_dyads": int(len(retained)),
                "mean_delta": float(retained["fixed_combination_minus_train_known"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {}
    for kind, block in frame.groupby("left_out_type", sort=True):
        values = block["mean_delta"].dropna()
        summary[str(kind)] = {
            "omissions": int(len(values)),
            "minimum_mean_delta": float(values.min()) if len(values) else None,
            "maximum_mean_delta": float(values.max()) if len(values) else None,
            "all_positive": bool((values > 0).all()) if len(values) else False,
        }
    return frame, summary


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--library-evidence", required=True, type=Path)
    parser.add_argument("--unresolved-score-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--min-known-per-fold", type=int, default=3)
    parser.add_argument("--min-unresolved-edges", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--sign-flips", type=int, default=99999)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    effects_path = args.collection / "response_entities.tsv.gz"
    contrasts_path = args.collection / "contrast_manifest.tsv"
    effects = add_response_weight(pd.read_csv(effects_path, sep="\t", low_memory=False))
    contrasts = pd.read_csv(contrasts_path, sep="\t")
    library = pd.read_csv(args.library_evidence, sep="\t", low_memory=False)
    selected = select_known_entities(library)
    selected = selected.copy()
    selected["known_fold"] = selected["library_inchikey"].astype(str).map(
        lambda value: inchikey_fold(value, args.split_salt)
    )
    known = fold_known_scores(effects, selected, args.split_salt)
    unresolved = pd.read_csv(args.unresolved_score_ledger, sep="\t", low_memory=False)
    required = (
        "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id",
        "unresolved_mutual_edge_support", "unresolved_signed_cosine", "same_parent_project",
    )
    missing_required = set(required) - set(unresolved.columns)
    if missing_required:
        raise ValueError(f"unresolved score ledger lacks {sorted(missing_required)}")
    keys = ["left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"]
    joined = known.merge(unresolved[list(required)], on=keys, how="inner", validate="one_to_one")
    joined = fold_swaps(joined)
    counts = contrasts.groupby("study_id", sort=False)["contrast_id"].nunique()
    joined["left_study_contrasts"] = joined["left_study_id"].map(counts)
    joined["right_study_contrasts"] = joined["right_study_id"].map(counts)
    # Predictor transformations are frozen on a feature-side universe that
    # does not use held-out truth support or held-out response values.  Truth
    # availability is applied only after these transformations are complete.
    feature_universe = joined.loc[
        ~as_bool(joined["same_parent_project"])
        & joined["train_known_support"].ge(args.min_known_per_fold)
        & joined["unresolved_mutual_edge_support"].ge(args.min_unresolved_edges)
        & joined["left_study_contrasts"].ge(2)
        & joined["right_study_contrasts"].ge(2)
        & np.isfinite(pd.to_numeric(joined["train_known_score"], errors="coerce"))
        & np.isfinite(pd.to_numeric(joined["unresolved_signed_cosine"], errors="coerce"))
    ].copy()
    for column, output in (
        ("train_known_score", "train_known_interaction"),
        ("unresolved_signed_cosine", "unresolved_interaction"),
    ):
        feature_universe[output] = double_center(feature_universe, column)
    feature_universe["combined_fixed_interaction"] = equal_norm_additive(feature_universe)
    eligible = feature_universe.loc[
        feature_universe["heldout_known_support"].ge(args.min_known_per_fold)
        & np.isfinite(
            pd.to_numeric(feature_universe["heldout_known_truth"], errors="coerce")
        )
    ].copy()
    eligible["heldout_known_truth_interaction"] = double_center(
        eligible, "heldout_known_truth"
    )

    pairs = pairwise_ledger(eligible)
    dyads = endpoint_by_dyad(pairs)
    bootstrap = bootstrap_dyad_delta(dyads, args.bootstrap, args.seed)
    sign_flip = sign_flip_delta(dyads, args.sign_flips, args.seed + 1)
    sensitivity, sensitivity_summary = leave_out_sensitivity(dyads)

    def pooled(arm: str) -> dict[str, object]:
        evaluable = pairs[f"{arm}_evaluable"].astype(bool)
        values = pairs.loc[evaluable, f"{arm}_concordant"].astype(float)
        return {
            "pairwise_comparisons": int(len(values)),
            "micro_concordance": float(values.mean()) if len(values) else None,
            "dyad_macro_concordance": float(dyads[f"{arm}_concordance"].mean()),
        }

    decision_pass = bool(
        bootstrap["ci95_low"] is not None
        and bootstrap["ci95_low"] > 0
        and sign_flip["one_sided_p"] is not None
        and sign_flip["one_sided_p"] <= 0.05
        and sensitivity_summary["study"]["all_positive"]
    )

    args.output.mkdir(parents=True)
    score_path = args.output / "crossfit_scores.tsv.gz"
    pair_path = args.output / "pairwise_concordance.tsv.gz"
    dyad_path = args.output / "dyad_endpoints.tsv"
    sensitivity_path = args.output / "leave_out_sensitivity.tsv"
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    eligible.to_csv(score_path, sep="\t", index=False, compression=deterministic_gzip)
    pairs.to_csv(pair_path, sep="\t", index=False, compression=deterministic_gzip)
    dyads.to_csv(dyad_path, sep="\t", index=False)
    sensitivity.to_csv(sensitivity_path, sep="\t", index=False)

    selected_keys = selected[["study_id", "library_inchikey"]].drop_duplicates()
    eligible_study_contrasts = set(
        zip(eligible["left_study_id"], eligible["left_contrast_id"])
    ) | set(zip(eligible["right_study_id"], eligible["right_contrast_id"]))
    eligible_truth_keys_by_fold: dict[int, set[str]] = {0: set(), 1: set()}
    for truth_fold in (0, 1):
        column = f"known_fold_{truth_fold}_inchikeys"
        for raw in eligible.loc[eligible["truth_fold"].eq(truth_fold), column].fillna(""):
            eligible_truth_keys_by_fold[truth_fold].update(
                value for value in str(raw).split(";") if value
            )
    eligible_truth_keys = eligible_truth_keys_by_fold[0] | eligible_truth_keys_by_fold[1]
    report = {
        "schema": "contrast_atlas_crossfit_unresolved_increment_v1",
        "status": (
            "PASS_INTERNAL_CROSSFIT_UNRESOLVED_INCREMENT_GATE"
            if decision_pass else "FAIL_INTERNAL_CROSSFIT_UNRESOLVED_INCREMENT_GATE"
        ),
        "data_role": "development_consumed",
        "question": (
            "Does a truth-blind fixed combination of unresolved and train-known response "
            "scores improve within-target-study ranking of held-out known response axes?"
        ),
        "contract": {
            "split": "SHA256(split_salt + '|' + full_InChIKey), first_64_bits_mod_2",
            "split_salt": args.split_salt,
            "fold_swaps": 2,
            "cross_parent_only": True,
            "min_known_exact_inchikeys_per_fold": args.min_known_per_fold,
            "min_unresolved_reciprocal_edges": args.min_unresolved_edges,
            "both_studies_require_at_least_two_contrasts": True,
            "interaction": (
                "score - query-axis mean - candidate-axis mean + source-pair mean, "
                "separately within each truth fold and source pair"
            ),
            "fixed_combination": (
                "equal-L2-norm additive train-known plus unresolved within each truth-fold "
                "and source pair; no held-out truth is read"
            ),
            "primary_endpoint": (
                "source-dyad macro change in within-target-study pairwise contrast "
                "concordance: fixed combination minus train-known"
            ),
        },
        "denominators": {
            "selected_study_inchikey_records": int(len(selected_keys)),
            "unique_full_inchikeys": int(selected_keys["library_inchikey"].nunique()),
            "unique_full_inchikeys_fold_0": int(
                selected.loc[selected["known_fold"].eq(0), "library_inchikey"].nunique()
            ),
            "unique_full_inchikeys_fold_1": int(
                selected.loc[selected["known_fold"].eq(1), "library_inchikey"].nunique()
            ),
            "study_sources_in_selected_known": int(selected_keys["study_id"].nunique()),
            "contrasts_represented_in_known_unresolved_ledger": int(
                len(set(zip(unresolved["left_study_id"], unresolved["left_contrast_id"]))
                    | set(zip(unresolved["right_study_id"], unresolved["right_contrast_id"])))
            ),
            "feature_side_fold_swapped_rows_before_truth_availability_filter": int(
                len(feature_universe)
            ),
            "fold_swapped_eligible_rows": int(len(eligible)),
            "eligible_rows_truth_fold_0": int(eligible["truth_fold"].eq(0).sum()),
            "eligible_rows_truth_fold_1": int(eligible["truth_fold"].eq(1).sum()),
            "base_contrast_pair_rows": int(len(eligible) // 2),
            "eligible_multi_axis_studies": int(
                len(set(eligible["left_study_id"]) | set(eligible["right_study_id"]))
            ),
            "eligible_study_contrast_units": int(len(eligible_study_contrasts)),
            "eligible_heldout_full_inchikeys": int(len(eligible_truth_keys)),
            "eligible_heldout_full_inchikeys_fold_0": int(
                len(eligible_truth_keys_by_fold[0])
            ),
            "eligible_heldout_full_inchikeys_fold_1": int(
                len(eligible_truth_keys_by_fold[1])
            ),
            "eligible_heldout_full_inchikey_fold_overlap": int(
                len(eligible_truth_keys_by_fold[0] & eligible_truth_keys_by_fold[1])
            ),
            "eligible_source_dyads": int(len(dyads)),
            "pairwise_comparisons": int(len(pairs)),
        },
        "endpoints": {
            "train_known": pooled("train_known"),
            "unresolved_only": pooled("unresolved_only"),
            "train_known_plus_unresolved_fixed": pooled(
                "train_known_plus_unresolved_fixed"
            ),
            "dyad_delta_bootstrap": bootstrap,
            "dyad_delta_sign_flip": sign_flip,
            "leave_out_sensitivity": sensitivity_summary,
        },
        "decision": {
            "gate_passed": decision_pass,
            "required": [
                "dyad bootstrap CI lower bound > 0",
                "dyad sign-flip one-sided p <= 0.05",
                "leave-one-study-out mean delta > 0 for every eligible study",
            ],
            "unresolved_increment": (
                "established_on_internal_crossfit_gate" if decision_pass
                else "not_established_on_internal_crossfit_gate"
            ),
            "algorithm_increment": "not_tested_wse_unresolved_edges_only",
        },
        "claim_boundary": [
            "All studies and response effects are development/consumed.",
            "Known and held-out folds are entity-disjoint, but their beta/SE estimates reuse the same experimental samples and labels.",
            "The common library policy and representative-ion selection use WSE and may favour the WSE baseline.",
            "A GNPS full-InChIKey match is not MSI level-1 identification.",
            "Unresolved entities may include adducts, fragments, or family members of known molecules; unresolved does not mean novel molecule.",
            "Source dyads share studies; dyad bootstrap and exact sign-flip results are development sensitivities, not independent-study confidence intervals.",
            "The fixed combination is truth-blind, but this result is not prospective or external biological validation.",
            "Train-known and unresolved centering/scaling are frozen on the feature-side universe before held-out truth availability is applied.",
            "Official, Noise, and Full23 require independent scoring and reciprocal decisions on the complete model-independent candidate axis before algorithm increment can be claimed.",
        ],
        "input_sha256": {
            str(effects_path): sha256(effects_path),
            str(contrasts_path): sha256(contrasts_path),
            str(args.library_evidence): sha256(args.library_evidence),
            str(args.unresolved_score_ledger): sha256(args.unresolved_score_ledger),
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksums = args.output / "checksums.sha256"
    outputs = [report_path, score_path, pair_path, dyad_path, sensitivity_path]
    checksums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Test reaction-neighbour co-abundance as a BioAware candidate action.

This is a candidate-ranking action audit, not embedding training.  The primary
comparison is a direct-reaction co-abundance model versus a nested model that
contains the same catalog/opportunity variables and abundance/degree-matched
non-neighbour correlations.  Query truth is used only for scoring outcomes and
to verify the technical row pointer used to recover the observed feature's six
sample abundances; it is never a candidate feature.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b2_reaction_transform_action import load_relations  # noqa: E402
from develop_bioaware_b1_multisource_action import (  # noqa: E402
    EXPECTED_SOURCES,
    EXPECTED_UNITS,
    add_derived_features,
    load_negative,
    load_positive,
    paired_cluster_bootstrap,
    paired_transition,
    run_source_loso,
    sha256,
    summarize,
)


SOURCE_PREFIX = {
    "BV2cell": "BV2cell_",
    "Mouse_brain": "brain_",
    "Mouse_liver": "liver_",
    "NIST_plasma": "plasma_",
}

OPPORTUNITY_FEATURES = [
    "spectral_score",
    "log_reference_spectra",
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
]
MATCHED_RANDOM_FEATURES = OPPORTUNITY_FEATURES + [
    "coabundance_available_fraction",
    "coabundance_log_neighbours_mean",
    "coabundance_random_abs_top3_mean",
    "coabundance_random_positive_top3_mean",
    "coabundance_random_negative_top3_mean",
    "coabundance_random_sign_stability_top3_mean",
]
COABUNDANCE_FEATURES = MATCHED_RANDOM_FEATURES + [
    "coabundance_actual_abs_top3_mean",
    "coabundance_actual_positive_top3_mean",
    "coabundance_actual_negative_top3_mean",
    "coabundance_actual_sign_stability_top3_mean",
    "coabundance_abs_excess_top3_mean",
    "coabundance_positive_excess_top3_mean",
    "coabundance_negative_excess_top3_mean",
    "coabundance_multiwitness_fraction",
]


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 4:
        return 0.0
    x = left[valid] - float(np.mean(left[valid]))
    y = right[valid] - float(np.mean(right[valid]))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def stable_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Log-scale Pearson correlation plus leave-one-out sign stability."""
    x = np.log1p(np.clip(np.asarray(left, dtype=float), 0.0, None))
    y = np.log1p(np.clip(np.asarray(right, dtype=float), 0.0, None))
    value = pearson(x, y)
    leave_one_out = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        leave_one_out.append(pearson(x[keep], y[keep]))
    direction = 1.0 if value >= 0 else -1.0
    stability = float(np.mean(np.asarray(leave_one_out) * direction >= 0))
    return {
        "signed": value,
        "absolute": abs(value),
        "positive": max(value, 0.0),
        "negative": max(-value, 0.0),
        "sign_stability": stability,
    }


def profile_statistics(profile: np.ndarray) -> tuple[float, float]:
    values = np.log1p(np.clip(np.asarray(profile, dtype=float), 0.0, None))
    return float(np.mean(values)), float(np.std(values))


def abundance_columns(source: str) -> list[str]:
    prefix = SOURCE_PREFIX[source]
    return [f"{prefix}{index:02d}" for index in range(1, 7)]


def load_unit_observations(
    manifest_unit: Path,
    positive_unit: Path,
    negative_unit: Path,
    source: str,
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, dict]]:
    level1_path = manifest_unit / "external_level1.csv.gz"
    positive_path = positive_unit / "cache/queries.csv.gz"
    negative_path = negative_unit / "queries.csv.gz"
    for path in (level1_path, positive_path, negative_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    level1 = pd.read_csv(level1_path)
    columns = abundance_columns(source)
    missing = set(["ik14", "polarity", "mz", "rt", *columns]) - set(level1.columns)
    if missing:
        raise RuntimeError(f"{manifest_unit.name}: missing Level-1 columns {sorted(missing)}")
    for column in columns:
        level1[column] = pd.to_numeric(level1[column], errors="coerce")
    metadata = pd.concat(
        [pd.read_csv(positive_path), pd.read_csv(negative_path)], ignore_index=True
    )
    if metadata["query_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{manifest_unit.name}: duplicate query IDs")

    query_profiles: dict[str, dict] = {}
    for row in metadata.itertuples(index=False):
        position = int(row.truth_row)
        if position < 0 or position >= len(level1):
            raise RuntimeError(f"{row.query_id}: invalid Level-1 row pointer {position}")
        observed = level1.iloc[position]
        if str(observed.ik14) != str(row.truth_ik14):
            raise RuntimeError(f"{row.query_id}: technical row pointer identity mismatch")
        if str(observed.polarity) != str(row.polarity):
            raise RuntimeError(f"{row.query_id}: technical row pointer polarity mismatch")
        values = observed[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or float(np.std(np.log1p(np.clip(values, 0, None)))) <= 1e-8:
            raise RuntimeError(f"{row.query_id}: unusable six-replicate abundance profile")
        mean, spread = profile_statistics(values)
        query_profiles[str(row.query_id)] = {
            "profile": values,
            "truth_ik14": str(row.truth_ik14),
            "polarity": str(row.polarity),
            "mean": mean,
            "spread": spread,
        }

    # A seed identity can have duplicate/adduct rows.  Choose deterministically
    # by highest mean abundance, then lowest original row number.  No query
    # outcome is involved in this selection.
    seed_profiles: dict[str, dict] = {}
    for position, observed in level1.iterrows():
        values = observed[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        mean, spread = profile_statistics(values)
        if spread <= 1e-8:
            continue
        identity = str(observed.ik14)
        polarity = str(observed.polarity)
        key = f"{identity}|{polarity}"
        record = {
            "identity": identity,
            "polarity": polarity,
            "profile": values,
            "mean": mean,
            "spread": spread,
            "row": int(position),
        }
        incumbent = seed_profiles.get(key)
        if incumbent is None or (mean, -position) > (incumbent["mean"], -incumbent["row"]):
            seed_profiles[key] = record
    return metadata, query_profiles, seed_profiles


def top3_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(sorted(map(float, values), reverse=True)[:3]))


def matched_controls(
    neighbour: str,
    candidate: str,
    polarity: str,
    seeds: set[str],
    seed_profiles: dict[str, dict],
    relation_neighbours: dict[str, set[str]],
    degree: dict[str, int],
    controls: int = 3,
) -> list[str]:
    neighbour_profile = seed_profiles[f"{neighbour}|{polarity}"]
    forbidden = relation_neighbours.get(candidate, set()) | {candidate, neighbour}
    pool = (seeds & {record["identity"] for record in seed_profiles.values() if record["polarity"] == polarity}) - forbidden
    ranked = sorted(
        pool,
        key=lambda identity: (
            abs(math.log1p(degree.get(identity, 0)) - math.log1p(degree.get(neighbour, 0))),
            abs(seed_profiles[f"{identity}|{polarity}"]["mean"] - neighbour_profile["mean"]),
            abs(seed_profiles[f"{identity}|{polarity}"]["spread"] - neighbour_profile["spread"]),
            identity,
        ),
    )
    return ranked[:controls] if len(ranked) >= controls else []


def candidate_rotation_features(
    candidate: str,
    query_profile: np.ndarray,
    polarity: str,
    seeds: set[str],
    seed_profiles: dict[str, dict],
    relation_neighbours: dict[str, set[str]],
    degree: dict[str, int],
) -> dict[str, float]:
    available_seeds = {
        record["identity"]
        for record in seed_profiles.values()
        if record["polarity"] == polarity
    }
    neighbours = sorted(relation_neighbours.get(candidate, set()) & seeds & available_seeds)
    actual: list[dict[str, float]] = []
    control_sets: list[list[dict[str, float]]] = [[], [], []]
    for neighbour in neighbours:
        controls = matched_controls(
            neighbour, candidate, polarity, seeds, seed_profiles,
            relation_neighbours, degree, controls=3,
        )
        if len(controls) != 3:
            continue
        actual.append(stable_correlation(query_profile, seed_profiles[f"{neighbour}|{polarity}"]["profile"]))
        for control_index, control in enumerate(controls):
            control_sets[control_index].append(
                stable_correlation(query_profile, seed_profiles[f"{control}|{polarity}"]["profile"])
            )
    count = len(actual)
    names = ("absolute", "positive", "negative", "sign_stability")
    actual_score = {name: top3_mean([item[name] for item in actual]) for name in names}
    random_score = {
        name: float(np.mean([top3_mean([item[name] for item in group]) for group in control_sets]))
        if count else 0.0
        for name in names
    }
    return {
        "available": float(count > 0),
        "neighbours": float(count),
        **{f"actual_{name}": actual_score[name] for name in names},
        **{f"random_{name}": random_score[name] for name in names},
        "abs_excess": actual_score["absolute"] - random_score["absolute"],
        "positive_excess": actual_score["positive"] - random_score["positive"],
        "negative_excess": actual_score["negative"] - random_score["negative"],
        "multiwitness": float(count >= 2),
    }


def aggregate_rotations(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "coabundance_available_fraction": ("available", "mean"),
        "coabundance_log_neighbours_mean": ("neighbours", lambda x: float(np.log1p(np.mean(x)))),
        "coabundance_multiwitness_fraction": ("multiwitness", "mean"),
        "coabundance_rotations": ("fold", "count"),
    }
    for prefix in ("actual", "random"):
        for name in ("abs", "positive", "negative", "sign_stability"):
            source_name = f"{prefix}_{'absolute' if name == 'abs' else name}"
            aggregations[f"coabundance_{prefix}_{name}_top3_mean"] = (source_name, "mean")
    for name in ("abs", "positive", "negative"):
        aggregations[f"coabundance_{name}_excess_top3_mean"] = (f"{name}_excess", "mean")
    return frame.groupby(["query_id", "candidate_id"], sort=False).agg(**aggregations).reset_index()


def enforce_negative_only(result: pd.DataFrame) -> pd.DataFrame:
    output = result.copy()
    positive = output["polarity"].astype(str).eq("positive")
    output.loc[positive, "gated_correct"] = output.loc[positive, "baseline_correct"]
    output.loc[positive, "corrected"] = False
    output.loc[positive, "introduced"] = False
    output.loc[positive, "intervene"] = False
    return output


def headroom(candidates: pd.DataFrame, column: str) -> dict:
    rows = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        positive = group.loc[group["candidate_id"].astype(str).eq(truth)]
        wrong = group.loc[~group["candidate_id"].astype(str).eq(truth)]
        truth_value = float(positive[column].iloc[0])
        wrong_value = float(wrong[column].max())
        rows.append({
            "query_id": str(query_id),
            "baseline_correct": bool(group["baseline_correct"].iloc[0]),
            "polarity": str(group["polarity"].iloc[0]),
            "truth_gt_wrong": truth_value > wrong_value,
            "truth_positive_wrong_zero": truth_value > 0 and wrong_value == 0,
        })
    frame = pd.DataFrame(rows)
    error = frame.loc[~frame.baseline_correct & frame.polarity.eq("negative")]
    return {
        "negative_baseline_errors": int(len(error)),
        "truth_gt_hardest_wrong": int(error.truth_gt_wrong.sum()),
        "truth_positive_wrong_zero": int(error.truth_positive_wrong_zero.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_v3_v1",
    )
    parser.add_argument(
        "--negative-query-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_units_v2",
    )
    parser.add_argument(
        "--negative-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--negative-path-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_paths_v1",
    )
    parser.add_argument(
        "--negative-edge0-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_edge_step0_v1",
    )
    parser.add_argument(
        "--manifest-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_manifest_v1",
    )
    parser.add_argument(
        "--network-dir", type=Path,
        default=ROOT / "data/reference/metdna2_emrn_network_20260828",
    )
    parser.add_argument(
        "--rhea-pairs", type=Path,
        default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--max-queries-per-unit", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    positive, positive_provenance = load_positive(args.positive_root)
    negative, negative_provenance = load_negative(
        args.negative_candidates, args.negative_path_root, args.negative_edge0_root
    )
    candidates = add_derived_features(pd.concat([positive, negative], ignore_index=True))
    relations, relation_report = load_relations(args.network_dir, args.rhea_pairs)
    relation_neighbours: dict[str, set[str]] = defaultdict(set)
    for left, right in relations:
        relation_neighbours[left].add(right)
    degree = {identity: len(neighbours) for identity, neighbours in relation_neighbours.items()}

    rotation_rows: list[dict] = []
    unit_reports: dict[str, dict] = {}
    level1_provenance: dict[str, str] = {}
    for unit in EXPECTED_UNITS:
        source = unit.split("__", 1)[0]
        manifest_unit = args.manifest_root / unit
        positive_unit = args.positive_root / unit
        negative_unit = args.negative_query_root / unit
        splits_path = manifest_unit / "identity_splits.csv.gz"
        if not splits_path.is_file():
            raise FileNotFoundError(splits_path)
        splits = pd.read_csv(splits_path)
        query_meta, query_profiles, seed_profiles = load_unit_observations(
            manifest_unit, positive_unit, negative_unit, source
        )
        metadata_by_query = query_meta.set_index(query_meta.query_id.astype(str))
        local = candidates.loc[candidates.unit_id.astype(str).eq(unit)].copy()
        if args.max_queries_per_unit > 0:
            keep = sorted(local.query_id.astype(str).unique())[: args.max_queries_per_unit]
            local = local.loc[local.query_id.astype(str).isin(keep)]
        if set(local.query_id.astype(str)) - set(query_profiles):
            raise RuntimeError(f"{unit}: candidate queries missing abundance profiles")

        attempted = 0
        available = 0
        for fold in range(10):
            seeds = set(splits.loc[splits.fold.eq(fold) & splits.role.eq("seed"), "ik14"].astype(str))
            heldout = set(splits.loc[splits.fold.eq(fold) & splits.role.eq("heldout"), "ik14"].astype(str))
            fold_candidates = local.loc[local.truth_candidate_id.astype(str).isin(heldout)]
            for row in fold_candidates.itertuples(index=False):
                query_id = str(row.query_id)
                query_record = query_profiles[query_id]
                if query_record["truth_ik14"] != str(row.truth_candidate_id):
                    raise RuntimeError(f"{query_id}: truth alignment mismatch")
                features = candidate_rotation_features(
                    str(row.candidate_id), query_record["profile"], str(row.polarity),
                    seeds, seed_profiles, relation_neighbours, degree,
                )
                rotation_rows.append({
                    "query_id": query_id,
                    "candidate_id": str(row.candidate_id),
                    "fold": int(fold),
                    **features,
                })
                attempted += 1
                available += int(features["available"] > 0)
        unit_reports[unit] = {
            "query_profiles": int(len(query_profiles)),
            "seed_profiles": int(len(seed_profiles)),
            "candidate_rotations": int(attempted),
            "rotations_with_complete_matched_controls": int(available),
        }
        level1_path = manifest_unit / "external_level1.csv.gz"
        level1_provenance[unit] = sha256(level1_path)
        print(f"[B3] {unit}: rotations={attempted:,} available={available:,}", flush=True)

    rotations = pd.DataFrame(rotation_rows)
    if rotations.empty:
        raise RuntimeError("no co-abundance rotations were constructed")
    aggregated = aggregate_rotations(rotations)
    before = len(candidates)
    candidates = candidates.merge(
        aggregated, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
    )
    if len(candidates) != before:
        raise RuntimeError("co-abundance merge changed candidate count")
    feature_columns = sorted(set(MATCHED_RANDOM_FEATURES + COABUNDANCE_FEATURES) - set(OPPORTUNITY_FEATURES))
    for column in feature_columns:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
    if not np.isfinite(candidates[feature_columns].to_numpy(float)).all():
        raise RuntimeError("non-finite co-abundance features")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rotations_path = args.output_dir / "reaction_coabundance_rotations.csv.gz"
    candidates_path = args.output_dir / "candidate_features.csv.gz"
    rotations.to_csv(rotations_path, index=False, compression="gzip")
    candidates.to_csv(candidates_path, index=False, compression="gzip")

    recipes = {
        "catalog_opportunity": OPPORTUNITY_FEATURES,
        "matched_random_coabundance": MATCHED_RANDOM_FEATURES,
        "reaction_coabundance": COABUNDANCE_FEATURES,
    }
    results: dict[str, pd.DataFrame] = {}
    recipe_reports: dict[str, dict] = {}
    for offset, (name, features) in enumerate(recipes.items()):
        print(f"[B3] source-LOSO {name} features={len(features)}", flush=True)
        result, folds = run_source_loso(candidates, name, features)
        result = enforce_negative_only(result)
        result_path = args.output_dir / f"{name}__transitions.csv.gz"
        result.to_csv(result_path, index=False, compression="gzip")
        results[name] = result
        recipe_reports[name] = {
            "features": features,
            "pooled": summarize(result),
            "by_source": {source: summarize(result.loc[result.source.eq(source)]) for source in EXPECTED_SOURCES},
            "by_polarity": {polarity: summarize(result.loc[result.polarity.eq(polarity)]) for polarity in ("negative", "positive")},
            "folds": folds,
            "transitions_sha256": sha256(result_path),
        }

    full = results["reaction_coabundance"]
    comparator = results["matched_random_coabundance"]
    primary_transition = paired_transition(full, comparator)
    primary_identity = paired_cluster_bootstrap(
        full, comparator, "truth_candidate_id", args.bootstrap_resamples, args.seed + 101
    )
    primary_formula = paired_cluster_bootstrap(
        full, comparator, "truth_formula", args.bootstrap_resamples, args.seed + 102
    )
    source_primary = {}
    for source in EXPECTED_SOURCES:
        left = full.loc[full.source.eq(source)]
        right = comparator.loc[comparator.source.eq(source)]
        source_primary[source] = {
            **paired_transition(left, right),
            "delta": float(left.gated_correct.mean() - right.gated_correct.mean()),
        }
    paired = full[["query_id", "gated_correct", "truth_candidate_id"]].merge(
        comparator[["query_id", "gated_correct"]], on="query_id",
        suffixes=("_full", "_comparator"), validate="one_to_one",
    )
    corrected_identities = int(
        paired.loc[paired.gated_correct_full & ~paired.gated_correct_comparator, "truth_candidate_id"].nunique()
    )
    positive_interventions = int(full.loc[full.polarity.eq("positive"), "intervene"].sum())
    gates = {
        "reaction_specific_increment_ge_3pp": primary_formula["mean"] >= 0.03,
        "identity_ci_low_positive": primary_identity["ci_low"] > 0,
        "formula_ci_low_positive": primary_formula["ci_low"] > 0,
        "corrected_gt_2x_introduced": primary_transition["corrected_vs_right"] > 2 * primary_transition["introduced_vs_right"],
        "corrected_identities_ge_20": corrected_identities >= 20,
        "all_four_sources_nonnegative": all(item["delta"] >= 0 for item in source_primary.values()),
        "positive_interventions_zero": positive_interventions == 0,
    }
    report = {
        "status": "bioaware_b3_reaction_coabundance_action_complete",
        "formal": args.max_queries_per_unit == 0,
        "model_type": "low-dimensional source-LOSO pairwise logistic ranker",
        "candidate_protocol": {
            "queries": int(candidates.query_id.nunique()),
            "candidate_rows": int(len(candidates)),
            "rotation_rows": int(len(rotations)),
            "truth_identities": int(candidates.truth_candidate_id.nunique()),
            "truth_formulas": int(candidates.truth_formula.nunique()),
            "sources": int(candidates.source.nunique()),
            "units": int(candidates.unit_id.nunique()),
            "baseline_recall1": float(candidates[["query_id", "baseline_correct"]].drop_duplicates().baseline_correct.mean()),
        },
        "relation_catalog": relation_report,
        "unit_reports": unit_reports,
        "headroom": {
            column: headroom(candidates, column)
            for column in (
                "coabundance_actual_abs_top3_mean",
                "coabundance_abs_excess_top3_mean",
                "coabundance_positive_excess_top3_mean",
            )
        },
        "recipes": recipe_reports,
        "primary_reaction_coabundance_vs_matched_random": {
            "transition": primary_transition,
            "corrected_identities": corrected_identities,
            "identity_cluster_bootstrap": primary_identity,
            "formula_cluster_bootstrap": primary_formula,
            "by_source": source_primary,
        },
        "gates": gates,
        "pass_to_context_model": bool(all(gates.values())),
        "contracts": {
            "direct_reactions_only": True,
            "seed_identity_held_out_by_rotation": True,
            "matched_controls_per_reaction_neighbour": 3,
            "controls_match_polarity_degree_abundance_mean_and_spread": True,
            "held_source_truth_identity_and_formula_purged": True,
            "query_abundance_is_observed_feature_only": True,
            "candidate_features_truth_blind": True,
            "negative_only_action": True,
            "positive_queries_untouched": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "positive": positive_provenance,
            "negative": negative_provenance,
            "level1": level1_provenance,
            "network_compounds": sha256(args.network_dir / "metdna2_emrn_compounds.csv.gz"),
            "network_edges": sha256(args.network_dir / "metdna2_emrn_edges.csv.gz"),
            "rhea_pairs": sha256(args.rhea_pairs),
            "rotations": sha256(rotations_path),
            "candidate_features": sha256(candidates_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Opened-development sample-context action audit. A pass permits frozen validation of a context reranker/adapter; it is not a universal spectrum-only embedding gain or SOTA evidence.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

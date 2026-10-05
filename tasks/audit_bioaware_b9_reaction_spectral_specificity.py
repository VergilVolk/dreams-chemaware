#!/usr/bin/env python
"""Mine reaction-specific spectral actions beyond catalogue opportunity.

B2 encoded the theoretical precursor-mass shift of a candidate/seed relation.
That quantity is nearly tautological in the strict mass-matched candidate graph
and cannot resolve same-formula isomers.  B9 instead asks whether an observed
reaction neighbour has *more* query-spectrum support than chemically and
topologically matched seed metabolites that are not recorded neighbours.

Action construction is truth blind.  For every candidate and every held-out
seed rotation, the script pairs each observed direct reaction neighbour with
three non-neighbour seed identities matched on polarity, adduct, formula mass,
element counts, peak count and graph degree.  Six fixed action cells are then
tested as nested increments over a comparator that already sees catalogue
opportunity and the corresponding matched-control score.  Truth labels are
opened only after the candidate feature table has been frozen.

This is an opened action-discovery experiment, not a deployable result and not
shared-embedding fine-tuning.  A cell may advance only if it beats its nested
matched-control model under source-LOSO identity/formula purging and a
multiplicity-adjusted cluster bootstrap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b2_reaction_transform_action import (  # noqa: E402
    EXPECTED_UNITS,
    MASS_CONCORDANCE_DA,
    OPPORTUNITY_FEATURES,
    add_derived_features,
    formula_mass,
    load_negative,
    load_positive,
    load_query_tensors,
    load_relations,
    load_seed_spectra,
    sha256,
    spectrum,
)
from develop_bioaware_b1_multisource_action import (  # noqa: E402
    EXPECTED_SOURCES,
    paired_transition,
    run_source_loso,
    summarize,
)


FRAGMENT_TOLERANCE_DA = 0.02
CONTROLS_PER_EDGE = 3

# These six cells are fixed before outcomes are opened.  Each feature is an
# actual-edge score minus a matched-non-neighbour score.
ACTION_CELLS: dict[str, tuple[str, str]] = {
    "truncated_direct_mean": (
        "b9_truncated_direct_control_mean",
        "b9_truncated_direct_excess_mean",
    ),
    "truncated_direct_best": (
        "b9_truncated_direct_control_mean",
        "b9_truncated_direct_excess_best",
    ),
    "neutral_loss_mean": (
        "b9_neutral_loss_control_mean",
        "b9_neutral_loss_excess_mean",
    ),
    "modified_cosine_mean": (
        "b9_modified_cosine_control_mean",
        "b9_modified_cosine_excess_mean",
    ),
    "dual_view_mean": (
        "b9_dual_view_control_mean",
        "b9_dual_view_excess_mean",
    ),
    "kgmn_support_mean": (
        "b9_kgmn_support_control_mean",
        "b9_kgmn_support_excess_mean",
    ),
}

MATCH_FEATURES = [
    "b9_edge_available_fraction",
    "b9_log_matched_edges_mean",
    "b9_control_tier0_fraction",
    "b9_control_tier1_fraction",
]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def formula_counts(formula: str) -> dict[str, int] | None:
    """Parse the same simple molecular-formula grammar used by B2."""
    import re

    text = str(formula).strip()
    if formula_mass(text) is None:
        return None
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if not tokens:
        return None
    return {element: int(count) if count else 1 for element, count in tokens}


def formula_descriptor(formula: str) -> dict[str, Any] | None:
    counts = formula_counts(formula)
    mass = formula_mass(formula)
    if counts is None or mass is None:
        return None
    common = ("C", "H", "N", "O", "P", "S")
    rare = tuple(sorted(element for element, count in counts.items() if count and element not in common))
    vector = np.asarray([counts.get(element, 0) for element in common], dtype=float)
    heavy = float(sum(count for element, count in counts.items() if element != "H"))
    return {"mass": float(mass), "vector": vector, "heavy": heavy, "rare": rare}


def _renormalise(intensity: np.ndarray) -> np.ndarray:
    values = np.asarray(intensity, dtype=float)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 else values


def _greedy_score(
    left_mz: np.ndarray,
    left_intensity: np.ndarray,
    right_mz: np.ndarray,
    right_intensity: np.ndarray,
    shifts: tuple[float, ...],
    tolerance: float,
) -> tuple[float, int]:
    """Maximum-product one-to-one matching with deterministic ties."""
    options: list[tuple[float, float, int, int]] = []
    for shift in shifts:
        for left_index, mass in enumerate(left_mz):
            low = int(np.searchsorted(right_mz, mass - shift - tolerance, side="left"))
            high = int(np.searchsorted(right_mz, mass - shift + tolerance, side="right"))
            for right_index in range(low, high):
                product = float(left_intensity[left_index] * right_intensity[right_index])
                error = abs(float(mass - right_mz[right_index] - shift))
                options.append((-product, error, left_index, right_index))
    options.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    score = 0.0
    matches = 0
    for negative_product, _error, left_index, right_index in options:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        score -= negative_product
        matches += 1
    return float(score), int(matches)


def reaction_spectral_views(
    query_tensor: np.ndarray,
    seed_tensor: np.ndarray,
    tolerance: float = FRAGMENT_TOLERANCE_DA,
) -> dict[str, float]:
    """Compute fixed MetDNA/KGMN-inspired, label-free pair evidence."""
    query_precursor, query_mz, query_intensity = spectrum(query_tensor)
    seed_precursor, seed_mz, seed_intensity = spectrum(seed_tensor)
    if not len(query_mz) or not len(seed_mz):
        return {name: 0.0 for name in (
            "truncated_direct", "neutral_loss", "modified_cosine",
            "dual_view", "kgmn_support", "matched_fragments",
        )}

    # MetDNA3-style truncation: peaks above the smaller precursor are removed
    # from the spectrum with the larger precursor before direct matching.
    cutoff = min(query_precursor, seed_precursor)
    query_keep = query_mz <= cutoff + tolerance
    seed_keep = seed_mz <= cutoff + tolerance
    tq_mz, ts_mz = query_mz[query_keep], seed_mz[seed_keep]
    tq_int = _renormalise(query_intensity[query_keep])
    ts_int = _renormalise(seed_intensity[seed_keep])
    truncated, matched = _greedy_score(
        tq_mz, tq_int, ts_mz, ts_int, (0.0,), tolerance
    )

    query_loss = query_precursor - query_mz
    seed_loss = seed_precursor - seed_mz
    query_loss_keep = query_loss > 0
    seed_loss_keep = seed_loss > 0
    query_loss = query_loss[query_loss_keep]
    seed_loss = seed_loss[seed_loss_keep]
    query_loss_intensity = _renormalise(query_intensity[query_loss_keep])
    seed_loss_intensity = _renormalise(seed_intensity[seed_loss_keep])
    q_order = np.argsort(query_loss, kind="stable")
    s_order = np.argsort(seed_loss, kind="stable")
    neutral, _ = _greedy_score(
        query_loss[q_order], query_loss_intensity[q_order],
        seed_loss[s_order], seed_loss_intensity[s_order], (0.0,), tolerance,
    )
    shift = query_precursor - seed_precursor
    modified, _ = _greedy_score(
        query_mz, query_intensity, seed_mz, seed_intensity,
        (0.0, shift), tolerance,
    )
    # Dual view requires both fragment and neutral-loss agreement.  KGMN's
    # published qualification rule is represented as a continuous support
    # score whose value reaches one at DP >= 0.5 or >= 5 matched fragments.
    dual = float(min(truncated, neutral))
    kgmn = float(max(min(truncated / 0.5, 1.0), min(matched / 5.0, 1.0)))
    return {
        "truncated_direct": float(truncated),
        "neutral_loss": float(neutral),
        "modified_cosine": float(modified),
        "dual_view": dual,
        "kgmn_support": kgmn,
        "matched_fragments": float(matched),
    }


def enrich_seed_records(
    seed_spectra: dict[str, list[dict]], relation_neighbours: dict[str, set[str]]
) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for identity, records in seed_spectra.items():
        enriched: list[dict] = []
        for record in records:
            descriptor = formula_descriptor(str(record["formula"]))
            if descriptor is None:
                continue
            precursor, mz, _intensity = spectrum(record["tensor"])
            enriched.append({
                **record,
                "precursor": float(precursor),
                "peak_count": int(len(mz)),
                "formula_descriptor": descriptor,
                "log_degree": float(np.log1p(len(relation_neighbours.get(identity, set())))),
            })
        if enriched:
            output[str(identity)] = sorted(enriched, key=lambda item: int(item["node"]))
    return output


def identity_profile(records: list[dict]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot profile an empty record list")
    first = records[0]["formula_descriptor"]
    return {
        "mass": float(np.mean([item["formula_descriptor"]["mass"] for item in records])),
        "vector": np.mean([item["formula_descriptor"]["vector"] for item in records], axis=0),
        "heavy": float(np.mean([item["formula_descriptor"]["heavy"] for item in records])),
        "rare": tuple(first["rare"]),
        "precursor": float(np.mean([item["precursor"] for item in records])),
        "peak_count": float(np.mean([item["peak_count"] for item in records])),
        "log_degree": float(np.mean([item["log_degree"] for item in records])),
        "record_count": int(len(records)),
    }


def control_distance(target: dict[str, Any], control: dict[str, Any]) -> tuple[int, float]:
    mass = abs(target["mass"] - control["mass"])
    heavy = abs(target["heavy"] - control["heavy"])
    element = float(np.abs(target["vector"] - control["vector"]).sum())
    degree = abs(target["log_degree"] - control["log_degree"])
    peaks = abs(math.log1p(target["peak_count"]) - math.log1p(control["peak_count"]))
    precursor = abs(target["precursor"] - control["precursor"])
    same_rare = target["rare"] == control["rare"]
    if same_rare and mass <= 2.0 and heavy <= 2.0 and element <= 4.0 and degree <= 0.75:
        tier = 0
    elif same_rare and mass <= 10.0 and heavy <= 4.0 and element <= 8.0 and degree <= 1.5:
        tier = 1
    else:
        tier = 2
    cost = mass / 5.0 + heavy / 2.0 + element / 4.0 + degree + peaks + precursor / 10.0
    if not same_rare:
        cost += 5.0
    return tier, float(cost)


def select_controls(
    target_records: list[dict],
    profiles: dict[str, tuple[list[dict], dict[str, Any]]],
    forbidden: set[str],
    count: int = CONTROLS_PER_EDGE,
) -> tuple[list[tuple[str, list[dict]]], int] | None:
    target = identity_profile(target_records)
    options: list[tuple[int, float, str, list[dict]]] = []
    for identity, (records, profile) in profiles.items():
        if identity in forbidden:
            continue
        tier, cost = control_distance(target, profile)
        if tier <= 1:
            options.append((tier, cost, identity, records))
    options.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(options) < count:
        return None
    selected = options[:count]
    maximum_tier = max(item[0] for item in selected)
    return [(identity, records) for _tier, _cost, identity, records in selected], int(maximum_tier)


def mean_views(
    query_id: str,
    query_tensor: np.ndarray,
    records: list[dict],
    cache: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float]:
    values: list[dict[str, float]] = []
    for record in records:
        key = (query_id, int(record["node"]))
        if key not in cache:
            cache[key] = reaction_spectral_views(query_tensor, record["tensor"])
        values.append(cache[key])
    return {
        name: float(np.mean([item[name] for item in values]))
        for name in values[0]
    }


def actual_seed_records(
    candidate: str,
    neighbour: str,
    candidate_formula: str,
    polarity: str,
    adduct: str,
    relations: dict[tuple[str, str], list[tuple[str, str, str]]],
    seeds: dict[str, list[dict]],
) -> list[dict]:
    candidate_mass = formula_mass(candidate_formula)
    if candidate_mass is None:
        return []
    allowed_seed_masses: list[float] = []
    for edge_candidate_formula, edge_seed_formula, _source in relations.get((candidate, neighbour), []):
        edge_candidate_mass = formula_mass(edge_candidate_formula)
        edge_seed_mass = formula_mass(edge_seed_formula)
        if edge_candidate_mass is None or edge_seed_mass is None:
            continue
        if abs(edge_candidate_mass - candidate_mass) <= MASS_CONCORDANCE_DA:
            allowed_seed_masses.append(float(edge_seed_mass))
    if not allowed_seed_masses:
        return []
    result = []
    for record in seeds.get(neighbour, []):
        if str(record["polarity"]) != polarity or str(record["adduct"]) != adduct:
            continue
        mass = float(record["formula_descriptor"]["mass"])
        if min(abs(mass - target) for target in allowed_seed_masses) <= MASS_CONCORDANCE_DA:
            result.append(record)
    return result


def candidate_specific_features(
    *,
    candidate: str,
    candidate_formula: str,
    query_id: str,
    query_tensor: np.ndarray,
    polarity: str,
    adduct: str,
    seed_identities: set[str],
    seeds: dict[str, list[dict]],
    relations: dict[tuple[str, str], list[tuple[str, str, str]]],
    relation_neighbours: dict[str, set[str]],
    pair_cache: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float]:
    neighbours = sorted(relation_neighbours.get(candidate, set()) & seed_identities & set(seeds))
    profiles: dict[str, tuple[list[dict], dict[str, Any]]] = {}
    for identity in sorted(seed_identities & set(seeds)):
        records = [
            record for record in seeds[identity]
            if str(record["polarity"]) == polarity and str(record["adduct"]) == adduct
        ]
        if records:
            profiles[identity] = (records, identity_profile(records))

    edge_rows: list[dict[str, float]] = []
    all_true_neighbours = set(relation_neighbours.get(candidate, set()))
    for neighbour in neighbours:
        real_records = actual_seed_records(
            candidate, neighbour, candidate_formula, polarity, adduct, relations, seeds
        )
        if not real_records:
            continue
        selected = select_controls(
            real_records,
            profiles,
            forbidden=all_true_neighbours | {candidate, neighbour},
        )
        if selected is None:
            continue
        controls, tier = selected
        real = mean_views(query_id, query_tensor, real_records, pair_cache)
        control_values = [
            mean_views(query_id, query_tensor, records, pair_cache)
            for _identity, records in controls
        ]
        control = {
            name: float(np.mean([item[name] for item in control_values]))
            for name in real
        }
        edge_rows.append({
            "control_tier": float(tier),
            **{f"real_{name}": float(value) for name, value in real.items()},
            **{f"control_{name}": float(value) for name, value in control.items()},
            **{f"excess_{name}": float(real[name] - control[name]) for name in real},
        })

    result: dict[str, float] = {
        "edge_available": float(bool(edge_rows)),
        "matched_edges": float(len(edge_rows)),
        "control_tier0_fraction": float(np.mean([row["control_tier"] == 0 for row in edge_rows])) if edge_rows else 0.0,
        "control_tier1_fraction": float(np.mean([row["control_tier"] == 1 for row in edge_rows])) if edge_rows else 0.0,
    }
    for view in ("truncated_direct", "neutral_loss", "modified_cosine", "dual_view", "kgmn_support"):
        for kind in ("real", "control", "excess"):
            values = [row[f"{kind}_{view}"] for row in edge_rows]
            result[f"{view}_{kind}_mean"] = float(np.mean(values)) if values else 0.0
        excess = [row[f"excess_{view}"] for row in edge_rows]
        result[f"{view}_excess_best"] = float(max(excess)) if excess else 0.0
    return result


def aggregate_rotations(rotations: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "b9_edge_available_fraction": ("edge_available", "mean"),
        "b9_log_matched_edges_mean": ("matched_edges", lambda values: float(np.mean(np.log1p(values)))),
        "b9_control_tier0_fraction": ("control_tier0_fraction", "mean"),
        "b9_control_tier1_fraction": ("control_tier1_fraction", "mean"),
    }
    for view in ("truncated_direct", "neutral_loss", "modified_cosine", "dual_view", "kgmn_support"):
        for kind in ("real", "control", "excess"):
            aggregations[f"b9_{view}_{kind}_mean"] = (f"{view}_{kind}_mean", "mean")
        aggregations[f"b9_{view}_excess_best"] = (f"{view}_excess_best", "mean")
    return (
        rotations.groupby(["query_id", "candidate_id"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )


def cluster_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cluster: str,
    repeats: int,
    seed: int,
    family_size: int,
) -> dict[str, float | int]:
    keys = ["query_id", "truth_candidate_id", "truth_formula", "source", "polarity"]
    paired = left[keys + ["gated_correct"]].merge(
        right[keys + ["gated_correct"]],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired["gated_correct_left"].astype(int)
        - paired["gated_correct_right"].astype(int)
    )
    grouped = paired.groupby(cluster, sort=False)["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    # Bonferroni family-wise interval across every pre-registered action cell.
    tail = 0.05 / (2.0 * family_size)
    return {
        "mean": float(paired["difference"].mean()),
        "ci_low_95": float(np.quantile(draws, 0.025)),
        "ci_high_95": float(np.quantile(draws, 0.975)),
        "ci_low_familywise": float(np.quantile(draws, tail)),
        "ci_high_familywise": float(np.quantile(draws, 1.0 - tail)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
        "family_size": int(family_size),
    }


def evidence_headroom(candidates: pd.DataFrame, column: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        truth = group.loc[group["candidate_id"].astype(str).eq(truth_id)]
        wrong = group.loc[~group["candidate_id"].astype(str).eq(truth_id)]
        truth_value = float(truth[column].iloc[0])
        wrong_value = float(wrong[column].max())
        rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": truth_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_correct": bool(group["baseline_correct"].iloc[0]),
            "truth_minus_hardest_wrong": truth_value - wrong_value,
        })
    frame = pd.DataFrame(rows)
    errors = frame.loc[~frame["baseline_correct"]]
    safe = frame.loc[frame["baseline_correct"]]
    return {
        "column": column,
        "queries": int(len(frame)),
        "baseline_errors": int(len(errors)),
        "recoverable_errors": int((errors["truth_minus_hardest_wrong"] > 0).sum()),
        "correct_queries_at_risk": int((safe["truth_minus_hardest_wrong"] < 0).sum()),
        "recoverable_identity_count": int(errors.loc[
            errors["truth_minus_hardest_wrong"] > 0, "truth_candidate_id"
        ].nunique()),
        "recoverable_formula_count": int(errors.loc[
            errors["truth_minus_hardest_wrong"] > 0, "truth_formula"
        ].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_v3_v1")
    parser.add_argument("--negative-query-root", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_negative_units_v2")
    parser.add_argument("--negative-candidates", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz")
    parser.add_argument("--negative-path-root", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_negative_paths_v1")
    parser.add_argument("--negative-edge0-root", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_negative_edge_step0_v1")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "data/validation/bioaware_metdna3_external_manifest_v1")
    parser.add_argument("--network-dir", type=Path, default=ROOT / "data/reference/metdna2_emrn_network_20260828")
    parser.add_argument("--rhea-pairs", type=Path, default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
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

    rotation_rows: list[dict[str, Any]] = []
    unit_reports: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    pair_cache: dict[tuple[str, int], dict[str, float]] = {}
    selected_candidate_parts: list[pd.DataFrame] = []
    for unit in EXPECTED_UNITS:
        positive_unit = args.positive_root / unit
        negative_unit = args.negative_query_root / unit
        manifest_unit = args.manifest_root / unit
        splits_path = manifest_unit / "identity_splits.csv.gz"
        splits = pd.read_csv(splits_path)
        query_meta, query_tensors = load_query_tensors(positive_unit, negative_unit)
        query_meta["query_id"] = query_meta["query_id"].astype(str)
        query_meta = query_meta.set_index("query_id")
        raw_seeds, seed_report = load_seed_spectra(positive_unit, manifest_unit, args.ppm, args.rt_sec)
        seeds = enrich_seed_records(raw_seeds, relation_neighbours)
        local = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        if args.max_queries_per_unit > 0:
            keep = sorted(local["query_id"].astype(str).unique())[: args.max_queries_per_unit]
            local = local.loc[local["query_id"].astype(str).isin(keep)].copy()
        selected_candidate_parts.append(local)
        if set(local["query_id"].astype(str)) - set(query_meta.index.astype(str)):
            raise RuntimeError(f"{unit}: candidate/query tensor mismatch")
        attempts = matched = 0
        tiers: Counter[int] = Counter()
        for fold in range(10):
            fold_seeds = set(splits.loc[
                splits["fold"].eq(fold) & splits["role"].eq("seed"), "ik14"
            ].astype(str))
            heldout = set(splits.loc[
                splits["fold"].eq(fold) & splits["role"].eq("heldout"), "ik14"
            ].astype(str))
            fold_candidates = local.loc[local["truth_candidate_id"].astype(str).isin(heldout)]
            for row in fold_candidates.itertuples(index=False):
                query_id = str(row.query_id)
                meta = query_meta.loc[query_id]
                features = candidate_specific_features(
                    candidate=str(row.candidate_id),
                    candidate_formula=str(row.truth_formula),
                    query_id=query_id,
                    query_tensor=query_tensors[query_id],
                    polarity=str(meta.polarity),
                    adduct=str(meta.adduct),
                    seed_identities=fold_seeds,
                    seeds=seeds,
                    relations=relations,
                    relation_neighbours=relation_neighbours,
                    pair_cache=pair_cache,
                )
                attempts += 1
                matched += int(features["edge_available"] > 0)
                if features["edge_available"] > 0:
                    tiers[0] += int(features["control_tier0_fraction"] > 0)
                    tiers[1] += int(features["control_tier1_fraction"] > 0)
                rotation_rows.append({
                    "fold": int(fold), "unit_id": unit,
                    "source": str(row.source), "polarity": str(row.polarity),
                    "query_id": query_id, "candidate_id": str(row.candidate_id),
                    "truth_candidate_id": str(row.truth_candidate_id),
                    "truth_formula": str(row.truth_formula), **features,
                })
        unit_reports[unit] = {
            "candidate_rotations": int(attempts),
            "rotations_with_complete_matched_edge_controls": int(matched),
            "fraction_with_complete_matched_edge_controls": float(matched / attempts) if attempts else 0.0,
            "query_count": int(local["query_id"].nunique()),
            "control_tier_presence": {str(key): int(value) for key, value in tiers.items()},
            "seed_spectra": seed_report,
        }
        provenance[f"{unit}/splits"] = sha256(splits_path)
        print(f"[B9] {unit}: rotations={attempts:,} complete={matched:,}", flush=True)

    rotations = pd.DataFrame(rotation_rows)
    local_candidates = pd.concat(selected_candidate_parts, ignore_index=True)
    if rotations.empty:
        raise RuntimeError("no B9 rotations constructed")
    aggregated = aggregate_rotations(rotations)
    merged = local_candidates.merge(
        aggregated, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
    )
    b9_columns = [column for column in merged if column.startswith("b9_")]
    if len(merged) != len(local_candidates) or merged[b9_columns].isna().any().any():
        raise RuntimeError("B9 aggregation lost candidate rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rotations_path = args.output_dir / "reaction_spectral_rotations.csv.gz"
    candidates_path = args.output_dir / "candidate_features.csv.gz"
    rotations.to_csv(rotations_path, index=False, compression="gzip")
    merged.to_csv(candidates_path, index=False, compression="gzip")

    headroom = {
        name: evidence_headroom(merged, action_column)
        for name, (_control_column, action_column) in ACTION_CELLS.items()
    }
    opened = args.max_queries_per_unit == 0
    action_reports: dict[str, Any] = {}
    advancing: list[str] = []
    if opened:
        for offset, (name, (control_column, action_column)) in enumerate(ACTION_CELLS.items()):
            common = OPPORTUNITY_FEATURES + MATCH_FEATURES + [control_column]
            comparator, comparator_folds = run_source_loso(
                merged, f"{name}__matched_control", common
            )
            action, action_folds = run_source_loso(
                merged, f"{name}__reaction_specific", common + [action_column]
            )
            comparator_path = args.output_dir / f"{name}__matched_control.csv.gz"
            action_path = args.output_dir / f"{name}__reaction_specific.csv.gz"
            comparator.to_csv(comparator_path, index=False, compression="gzip")
            action.to_csv(action_path, index=False, compression="gzip")
            transition = paired_transition(action, comparator)
            formula_ci = cluster_bootstrap(
                action, comparator, "truth_formula", args.bootstrap_resamples,
                args.seed + offset * 10 + 1, len(ACTION_CELLS),
            )
            identity_ci = cluster_bootstrap(
                action, comparator, "truth_candidate_id", args.bootstrap_resamples,
                args.seed + offset * 10 + 2, len(ACTION_CELLS),
            )
            by_source = {
                source: {
                    **paired_transition(
                        action.loc[action["source"].eq(source)],
                        comparator.loc[comparator["source"].eq(source)],
                    ),
                    "delta": float(
                        action.loc[action["source"].eq(source), "gated_correct"].mean()
                        - comparator.loc[comparator["source"].eq(source), "gated_correct"].mean()
                    ),
                }
                for source in EXPECTED_SOURCES
            }
            corrected = int(transition["corrected_vs_right"])
            introduced = int(transition["introduced_vs_right"])
            paired = action[["query_id", "truth_candidate_id", "gated_correct"]].merge(
                comparator[["query_id", "gated_correct"]], on="query_id",
                suffixes=("_action", "_control"), validate="one_to_one",
            )
            corrected_identities = int(paired.loc[
                paired["gated_correct_action"] & ~paired["gated_correct_control"],
                "truth_candidate_id",
            ].nunique())
            gates = {
                "increment_ge_3pp": formula_ci["mean"] >= 0.03,
                "formula_familywise_ci_low_positive": formula_ci["ci_low_familywise"] > 0,
                "identity_familywise_ci_low_positive": identity_ci["ci_low_familywise"] > 0,
                "corrected_gt_2x_introduced": corrected > 2 * introduced,
                "corrected_identities_ge_25": corrected_identities >= 25,
                "every_source_nonnegative": all(item["delta"] >= 0 for item in by_source.values()),
            }
            passed = bool(all(gates.values()))
            if passed:
                advancing.append(name)
            action_reports[name] = {
                "control_feature": control_column,
                "specific_action_feature": action_column,
                "matched_control": summarize(comparator),
                "reaction_specific": summarize(action),
                "transition": transition,
                "formula_cluster_bootstrap": formula_ci,
                "identity_cluster_bootstrap": identity_ci,
                "corrected_identities": corrected_identities,
                "by_source": by_source,
                "gates": gates,
                "pass_to_direct_training": passed,
                "control_folds": comparator_folds,
                "action_folds": action_folds,
                "provenance": {
                    "matched_control": sha256(comparator_path),
                    "reaction_specific": sha256(action_path),
                },
            }

    report = {
        "status": "bioaware_b9_reaction_spectral_specificity_complete",
        "formal": bool(opened),
        "queries": int(merged["query_id"].nunique()),
        "candidate_rows": int(len(merged)),
        "rotation_rows": int(len(rotations)),
        "queries_with_any_complete_reaction_control": int(
            merged.groupby("query_id")["b9_edge_available_fraction"].max().gt(0).sum()
        ),
        "candidates_with_any_complete_reaction_control": int(
            merged["b9_edge_available_fraction"].gt(0).sum()
        ),
        "action_cells": list(ACTION_CELLS),
        "headroom": headroom,
        "results": action_reports,
        "advancing_cells": advancing,
        "pass_to_direct_training": bool(advancing),
        "unit_coverage": unit_reports,
        "relation_graph": relation_report,
        "contracts": {
            "truth_blind_action_construction": True,
            "same_unit_polarity_and_adduct_controls": True,
            "controls_are_recorded_seed_metabolites_but_unrecorded_candidate_neighbours": True,
            "control_matching": "formula mass + element counts + heavy atoms + peak count + graph degree",
            "source_loso_truth_identity_and_formula_purge": True,
            "multiplicity": "Bonferroni family-wise cluster-bootstrap intervals across six fixed cells",
            "P2b_used": False,
            "phenotype_used": False,
        },
        "parameters": {
            "fragment_tolerance_da": FRAGMENT_TOLERANCE_DA,
            "controls_per_edge": CONTROLS_PER_EDGE,
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "seed": int(args.seed),
            "max_queries_per_unit": int(args.max_queries_per_unit),
        },
        "provenance": {
            "positive_inputs": positive_provenance,
            "negative_inputs": negative_provenance,
            "unit_inputs": provenance,
            "rotations_sha256": sha256(rotations_path),
            "candidate_features_sha256": sha256(candidates_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "decision_rule": (
            "Only a reaction-specific residual that adds >=3 pp over its matched-control nested model, "
            "has identity- and formula-cluster family-wise CI lower bounds >0, corrected >2*introduced, "
            ">=25 corrected identities and no negative source may enter direct embedding training."
        ),
        "claim_limit": (
            "Opened action discovery only. Passing would establish a reaction-specific candidate-ranking "
            "action beyond matched catalogue opportunity; it would not yet establish shared-embedding gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "queries": report["queries"],
        "coverage_queries": report["queries_with_any_complete_reaction_control"],
        "advancing_cells": advancing,
        "pass_to_direct_training": report["pass_to_direct_training"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Audit a candidate-specific reaction-transform BioAware action.

The historical BioAware path score mostly measured whether a candidate was
represented in a reaction catalogue.  This experiment asks the narrower and
scientifically testable question: when a candidate has a *direct* Rhea/KEGG
neighbour among the identity-held-out seeds, does the candidate-specific
reaction mass shift explain the query/seed fragment relationship better than
generic spectrum similarity and matched wrong-shift controls?

Every candidate feature is constructed without consulting the query truth.
Truth is opened only after all per-candidate features have been frozen.  The
primary endpoint is the source-LOSO transform model minus a nested generic-edge
model, not either model minus DreaMS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_metdna3_recursive_headroom import nearest_feature  # noqa: E402
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


MASS = {
    "H": 1.00782503223,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Br": 78.9183376,
    "I": 126.904468,
}
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

OPPORTUNITY_FEATURES = [
    "spectral_score",
    "log_reference_spectra",
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
]
GENERIC_EDGE_FEATURES = OPPORTUNITY_FEATURES + [
    "rxn_available_fraction",
    "rxn_log_neighbours_mean",
    "rxn_direct_top3_mean",
    "rxn_precursor_hybrid_top3_mean",
]
TRANSFORM_FEATURES = GENERIC_EDGE_FEATURES + [
    "rxn_mass_concordant_fraction",
    "rxn_mass_weight_top3_mean",
    "rxn_theoretical_shift_top3_mean",
    "rxn_theoretical_hybrid_top3_mean",
    "rxn_transform_concordance_top3_mean",
    "rxn_transform_specificity_top3_mean",
]
WRONG_SHIFT_FEATURES = GENERIC_EDGE_FEATURES + [
    "rxn_mass_concordant_fraction",
    "rxn_mass_weight_top3_mean",
    "rxn_wrong_sign_hybrid_top3_mean",
    "rxn_wrong_sign_concordance_top3_mean",
]
REACTION_FEATURES = sorted(set(GENERIC_EDGE_FEATURES + TRANSFORM_FEATURES + WRONG_SHIFT_FEATURES) - set(OPPORTUNITY_FEATURES))

FRAGMENT_TOLERANCE_DA = 0.02
MASS_RESIDUAL_SIGMA_DA = 0.01
MASS_CONCORDANCE_DA = 0.02


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def formula_mass(formula: str) -> float | None:
    """Return monoisotopic formula mass, rejecting partial parses."""
    text = str(formula).strip()
    if not text or text.lower() == "nan":
        return None
    position = 0
    total = 0.0
    for match in FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            return None
        element, count_text = match.groups()
        if element not in MASS:
            return None
        count = int(count_text) if count_text else 1
        if count <= 0:
            return None
        total += MASS[element] * count
        position = match.end()
    return total if position == len(text) and position > 0 else None


def adduct_charge(adduct: str) -> int | None:
    text = str(adduct).strip()
    match = re.search(r"(\d*)([+-])$", text)
    if not match:
        return None
    magnitude = int(match.group(1)) if match.group(1) else 1
    return magnitude if magnitude > 0 else None


def spectrum(tensor: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    array = np.asarray(tensor, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError("expected [tokens,2] DreaMS spectrum tensor")
    precursor = float(array[0, 0])
    fragments = array[1:]
    valid = (
        np.isfinite(fragments[:, 0])
        & np.isfinite(fragments[:, 1])
        & (fragments[:, 0] > 0)
        & (fragments[:, 1] > 0)
        & (fragments[:, 0] < precursor + 0.5)
    )
    mz = fragments[valid, 0]
    intensity = np.sqrt(fragments[valid, 1])
    norm = float(np.linalg.norm(intensity))
    if norm > 0:
        intensity = intensity / norm
    order = np.argsort(mz, kind="stable")
    return precursor, mz[order], intensity[order]


def greedy_shifted_cosine(
    left_mz: np.ndarray,
    left_intensity: np.ndarray,
    right_mz: np.ndarray,
    right_intensity: np.ndarray,
    shifts: tuple[float, ...],
    tolerance: float = FRAGMENT_TOLERANCE_DA,
) -> tuple[float, float]:
    """One-to-one greedy cosine for direct and/or mass-shifted matches."""
    options: list[tuple[float, float, int, int]] = []
    for shift in shifts:
        for left_index, mass in enumerate(left_mz):
            lower = int(np.searchsorted(right_mz, mass - shift - tolerance, side="left"))
            upper = int(np.searchsorted(right_mz, mass - shift + tolerance, side="right"))
            for right_index in range(lower, upper):
                error = abs(float(mass - right_mz[right_index] - shift))
                product = float(left_intensity[left_index] * right_intensity[right_index])
                options.append((-product, error, left_index, right_index))
    options.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    score = 0.0
    matched = 0
    for negative_product, _error, left_index, right_index in options:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        score -= negative_product
        matched += 1
    denominator = min(len(left_mz), len(right_mz))
    fraction = matched / denominator if denominator else 0.0
    return float(score), float(fraction)


def transform_pair_features(
    query_tensor: np.ndarray,
    seed_tensor: np.ndarray,
    candidate_formula: str,
    seed_formula: str,
    charge: int,
) -> dict[str, float] | None:
    candidate_mass = formula_mass(candidate_formula)
    seed_mass = formula_mass(seed_formula)
    if candidate_mass is None or seed_mass is None or charge <= 0:
        return None
    query_precursor, query_mz, query_intensity = spectrum(query_tensor)
    seed_precursor, seed_mz, seed_intensity = spectrum(seed_tensor)
    if not len(query_mz) or not len(seed_mz):
        return None
    theoretical_shift = (candidate_mass - seed_mass) / charge
    observed_shift = query_precursor - seed_precursor
    mass_residual = abs(observed_shift - theoretical_shift)
    mass_weight = math.exp(-0.5 * (mass_residual / MASS_RESIDUAL_SIGMA_DA) ** 2)
    direct, direct_fraction = greedy_shifted_cosine(
        query_mz, query_intensity, seed_mz, seed_intensity, (0.0,)
    )
    theoretical_shift_score, theoretical_shift_fraction = greedy_shifted_cosine(
        query_mz, query_intensity, seed_mz, seed_intensity, (theoretical_shift,)
    )
    theoretical_hybrid, theoretical_hybrid_fraction = greedy_shifted_cosine(
        query_mz, query_intensity, seed_mz, seed_intensity, (0.0, theoretical_shift)
    )
    precursor_hybrid, _ = greedy_shifted_cosine(
        query_mz, query_intensity, seed_mz, seed_intensity, (0.0, observed_shift)
    )
    wrong_sign_hybrid, _ = greedy_shifted_cosine(
        query_mz, query_intensity, seed_mz, seed_intensity, (0.0, -theoretical_shift)
    )
    specificity = theoretical_hybrid - max(direct, wrong_sign_hybrid)
    return {
        "mass_residual": float(mass_residual),
        "mass_weight": float(mass_weight),
        "direct": direct,
        "direct_fraction": direct_fraction,
        "theoretical_shift": theoretical_shift_score,
        "theoretical_shift_fraction": theoretical_shift_fraction,
        "theoretical_hybrid": theoretical_hybrid,
        "theoretical_hybrid_fraction": theoretical_hybrid_fraction,
        "precursor_hybrid": precursor_hybrid,
        "wrong_sign_hybrid": wrong_sign_hybrid,
        "transform_concordance": float(mass_weight * theoretical_hybrid),
        "transform_specificity": float(mass_weight * specificity),
        "wrong_sign_concordance": float(mass_weight * wrong_sign_hybrid),
    }


def add_relation(
    mapping: dict[tuple[str, str], list[tuple[str, str, str]]],
    left: str,
    right: str,
    left_formula: str,
    right_formula: str,
    source: str,
) -> None:
    if left == right:
        return
    if formula_mass(left_formula) is None or formula_mass(right_formula) is None:
        return
    mapping[(left, right)].append((left_formula, right_formula, source))
    mapping[(right, left)].append((right_formula, left_formula, source))


def load_relations(
    network_dir: Path, rhea_pairs_path: Path
) -> tuple[dict[tuple[str, str], list[tuple[str, str, str]]], dict]:
    compounds_path = network_dir / "metdna2_emrn_compounds.csv.gz"
    edges_path = network_dir / "metdna2_emrn_edges.csv.gz"
    for path in (compounds_path, edges_path, rhea_pairs_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    compounds = pd.read_csv(compounds_path).set_index("id")
    edges = pd.read_csv(edges_path)
    edges = edges.loc[pd.to_numeric(edges["minimum_step"], errors="coerce").eq(0)]
    relations: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    kegg_added = 0
    for row in edges.itertuples(index=False):
        if row.source_id not in compounds.index or row.target_id not in compounds.index:
            continue
        left = compounds.loc[row.source_id]
        right = compounds.loc[row.target_id]
        add_relation(
            relations,
            str(row.source_ik14),
            str(row.target_ik14),
            str(left.formula),
            str(right.formula),
            "KEGG",
        )
        kegg_added += 1
    rhea = pd.read_csv(rhea_pairs_path)
    rhea = rhea.loc[rhea["relation_type"].astype(str).eq("reaction_direction_unknown")]
    rhea_added = 0
    for row in rhea.itertuples(index=False):
        add_relation(
            relations,
            str(row.identity_a),
            str(row.identity_b),
            str(row.formula_a),
            str(row.formula_b),
            "Rhea",
        )
        rhea_added += 1
    deduplicated = {
        key: sorted(set(values))
        for key, values in relations.items()
    }
    return deduplicated, {
        "kegg_step0_rows": int(kegg_added),
        "rhea_rows": int(rhea_added),
        "oriented_identity_pairs": int(len(deduplicated)),
        "undirected_identity_pairs": int(len({tuple(sorted(key)) for key in deduplicated})),
    }


def load_query_tensors(
    positive_unit: Path, negative_unit: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    sources = [
        (positive_unit / "cache/queries.csv.gz", positive_unit / "cache/query_tensors.npz"),
        (negative_unit / "queries.csv.gz", negative_unit / "query_tensors.npz"),
    ]
    frames: list[pd.DataFrame] = []
    tensors: dict[str, np.ndarray] = {}
    for meta_path, tensor_path in sources:
        if not meta_path.is_file() or not tensor_path.is_file():
            raise FileNotFoundError(meta_path if not meta_path.is_file() else tensor_path)
        metadata = pd.read_csv(meta_path)
        array = np.load(tensor_path, allow_pickle=False)["query_tensor"]
        if len(metadata) != len(array):
            raise RuntimeError(f"metadata/tensor mismatch: {meta_path}")
        for query_id, tensor in zip(metadata["query_id"].astype(str), array, strict=True):
            if query_id in tensors:
                raise RuntimeError(f"duplicate query tensor: {query_id}")
            tensors[query_id] = tensor
        frames.append(metadata)
    metadata = pd.concat(frames, ignore_index=True)
    if metadata["query_id"].astype(str).duplicated().any():
        raise RuntimeError("query metadata collision between polarities")
    return metadata, tensors


def load_seed_spectra(
    positive_unit: Path,
    manifest_unit: Path,
    ppm: float,
    rt_sec: float,
) -> tuple[dict[str, list[dict]], dict]:
    paths = {
        "nodes": positive_unit / "recursive/stable_ms1_feature_nodes.csv.gz",
        "feature_meta": positive_unit / "feature_ms2/feature_ms2.csv.gz",
        "feature_tensor": positive_unit / "feature_ms2/feature_ms2_tensors.npz",
        "level1": manifest_unit / "external_level1.csv.gz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    nodes = pd.read_csv(paths["nodes"])
    feature_meta = pd.read_csv(paths["feature_meta"])
    feature_array = np.load(paths["feature_tensor"], allow_pickle=False)["feature_ms2_tensor"]
    if len(feature_meta) != len(feature_array):
        raise RuntimeError("feature MS2 metadata/tensor mismatch")
    tensor_by_node = {
        int(row.feature_node): feature_array[position]
        for position, row in enumerate(feature_meta.itertuples(index=False))
    }
    level1 = pd.read_csv(paths["level1"])
    records: dict[str, list[dict]] = defaultdict(list)
    recovered = 0
    for row in level1.itertuples(index=False):
        node = nearest_feature(nodes, str(row.polarity), float(row.mz), float(row.rt), ppm, rt_sec)
        if node is None or int(node) not in tensor_by_node:
            continue
        charge = adduct_charge(str(row.adduct))
        if charge is None or formula_mass(str(row.formula)) is None:
            continue
        records[str(row.ik14)].append(
            {
                "node": int(node),
                "polarity": str(row.polarity),
                "adduct": str(row.adduct),
                "charge": int(charge),
                "formula": str(row.formula),
                "tensor": tensor_by_node[int(node)],
            }
        )
        recovered += 1
    for identity in list(records):
        unique: dict[tuple, dict] = {}
        for record in records[identity]:
            key = (record["node"], record["adduct"], record["formula"])
            unique[key] = record
        records[identity] = list(unique.values())
    return dict(records), {
        "level1_rows": int(len(level1)),
        "recovered_level1_rows": int(recovered),
        "recovered_seed_identities": int(len(records)),
        "paths": {name: sha256(path) for name, path in paths.items()},
    }


def top3_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(map(float, values), reverse=True)[:3]
    return float(np.mean(ordered))


def candidate_rotation_features(
    candidate: str,
    query_tensor: np.ndarray,
    query_adduct: str,
    query_polarity: str,
    seeds: set[str],
    relations: dict[tuple[str, str], list[tuple[str, str, str]]],
    relation_neighbours: dict[str, set[str]],
    seed_spectra: dict[str, list[dict]],
) -> dict:
    neighbours = sorted(
        relation_neighbours.get(candidate, set()) & seeds & set(seed_spectra)
    )
    neighbour_best: list[dict] = []
    source_counts: Counter[str] = Counter()
    same_formula_edges = 0
    for neighbour in neighbours:
        options: list[dict] = []
        for candidate_formula, edge_seed_formula, source in relations[(candidate, neighbour)]:
            edge_seed_mass = formula_mass(edge_seed_formula)
            if edge_seed_mass is None:
                continue
            for seed in seed_spectra[neighbour]:
                if seed["polarity"] != query_polarity or seed["adduct"] != query_adduct:
                    continue
                observed_seed_mass = formula_mass(seed["formula"])
                if observed_seed_mass is None or abs(edge_seed_mass - observed_seed_mass) > MASS_CONCORDANCE_DA:
                    continue
                features = transform_pair_features(
                    query_tensor,
                    seed["tensor"],
                    candidate_formula,
                    edge_seed_formula,
                    int(seed["charge"]),
                )
                if features is None:
                    continue
                features["source"] = source
                features["same_formula_edge"] = float(candidate_formula == edge_seed_formula)
                options.append(features)
        if not options:
            continue
        # This choice is label-free and fixed: retain the edge/spectrum with
        # the highest candidate-specific physical concordance.
        best = max(
            options,
            key=lambda item: (
                item["transform_concordance"],
                item["theoretical_hybrid"],
                -item["mass_residual"],
            ),
        )
        source_counts[str(best["source"])] += 1
        same_formula_edges += int(best["same_formula_edge"] > 0)
        neighbour_best.append(best)
    count = len(neighbour_best)
    result = {
        "eligible_neighbours": count,
        "same_formula_neighbours": same_formula_edges,
        "rhea_neighbours": int(source_counts["Rhea"]),
        "kegg_neighbours": int(source_counts["KEGG"]),
        "available": float(count > 0),
        "log_neighbours": float(np.log1p(count)),
        "mass_concordant": float(any(item["mass_residual"] <= MASS_CONCORDANCE_DA for item in neighbour_best)),
    }
    for key in (
        "mass_weight", "direct", "precursor_hybrid", "theoretical_shift",
        "theoretical_hybrid", "wrong_sign_hybrid", "transform_concordance",
        "transform_specificity", "wrong_sign_concordance",
    ):
        result[key] = top3_mean([item[key] for item in neighbour_best])
    return result


def aggregate_rotations(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "rxn_available_fraction": ("available", "mean"),
        "rxn_log_neighbours_mean": ("log_neighbours", "mean"),
        "rxn_mass_concordant_fraction": ("mass_concordant", "mean"),
        "rxn_mass_weight_top3_mean": ("mass_weight", "mean"),
        "rxn_direct_top3_mean": ("direct", "mean"),
        "rxn_precursor_hybrid_top3_mean": ("precursor_hybrid", "mean"),
        "rxn_theoretical_shift_top3_mean": ("theoretical_shift", "mean"),
        "rxn_theoretical_hybrid_top3_mean": ("theoretical_hybrid", "mean"),
        "rxn_wrong_sign_hybrid_top3_mean": ("wrong_sign_hybrid", "mean"),
        "rxn_transform_concordance_top3_mean": ("transform_concordance", "mean"),
        "rxn_transform_specificity_top3_mean": ("transform_specificity", "mean"),
        "rxn_wrong_sign_concordance_top3_mean": ("wrong_sign_concordance", "mean"),
        "rxn_eligible_neighbours_mean": ("eligible_neighbours", "mean"),
        "rxn_rhea_neighbours_mean": ("rhea_neighbours", "mean"),
        "rxn_kegg_neighbours_mean": ("kegg_neighbours", "mean"),
        "rxn_same_formula_neighbours_mean": ("same_formula_neighbours", "mean"),
        "rxn_rotations": ("fold", "count"),
    }
    return frame.groupby(["query_id", "candidate_id"], sort=False).agg(**columns).reset_index()


def evidence_headroom(candidates: pd.DataFrame, column: str) -> dict:
    rows: list[dict] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        truth = group.loc[group["candidate_id"].astype(str).eq(truth_id)]
        wrong = group.loc[~group["candidate_id"].astype(str).eq(truth_id)]
        truth_score = float(truth[column].iloc[0])
        wrong_score = float(wrong[column].max())
        rows.append(
            {
                "query_id": str(query_id),
                "truth_candidate_id": truth_id,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "source": str(group["source"].iloc[0]),
                "polarity": str(group["polarity"].iloc[0]),
                "baseline_correct": bool(group["baseline_correct"].iloc[0]),
                "delta": truth_score - wrong_score,
                "truth_score": truth_score,
                "wrong_score": wrong_score,
            }
        )
    result = pd.DataFrame(rows)
    error = result.loc[~result["baseline_correct"]]
    safety = result.loc[result["baseline_correct"]]
    return {
        "column": column,
        "queries": int(len(result)),
        "baseline_errors": int(len(error)),
        "error_truth_gt_hardest_wrong": int((error["delta"] > 0).sum()),
        "error_truth_positive_wrong_zero": int(((error["truth_score"] > 0) & (error["wrong_score"] == 0)).sum()),
        "baseline_correct_at_risk": int((safety["delta"] < 0).sum()),
        "coverage_truth_positive": int((result["truth_score"] > 0).sum()),
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
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
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

    rotation_rows: list[dict] = []
    unit_reports: dict[str, dict] = {}
    query_metadata_provenance: dict[str, str] = {}
    for unit in EXPECTED_UNITS:
        positive_unit = args.positive_root / unit
        negative_unit = args.negative_query_root / unit
        manifest_unit = args.manifest_root / unit
        splits_path = manifest_unit / "identity_splits.csv.gz"
        if not splits_path.is_file():
            raise FileNotFoundError(splits_path)
        splits = pd.read_csv(splits_path)
        query_meta, query_tensors = load_query_tensors(positive_unit, negative_unit)
        query_meta["query_id"] = query_meta["query_id"].astype(str)
        query_meta = query_meta.set_index("query_id")
        seed_spectra, seed_report = load_seed_spectra(
            positive_unit, manifest_unit, args.ppm, args.rt_sec
        )
        local = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        if args.max_queries_per_unit > 0:
            keep = sorted(local["query_id"].astype(str).unique())[: args.max_queries_per_unit]
            local = local.loc[local["query_id"].astype(str).isin(keep)]
        missing_queries = set(local["query_id"].astype(str)) - set(query_meta.index.astype(str))
        if missing_queries:
            raise RuntimeError(f"{unit}: missing {len(missing_queries)} query tensors")
        attempted = 0
        available = 0
        for fold in range(10):
            seeds = set(
                splits.loc[ splits["fold"].eq(fold) & splits["role"].eq("seed"), "ik14"].astype(str)
            )
            heldout = set(
                splits.loc[splits["fold"].eq(fold) & splits["role"].eq("heldout"), "ik14"].astype(str)
            )
            fold_candidates = local.loc[local["truth_candidate_id"].astype(str).isin(heldout)]
            for row in fold_candidates.itertuples(index=False):
                query_id = str(row.query_id)
                meta = query_meta.loc[query_id]
                features = candidate_rotation_features(
                    str(row.candidate_id),
                    query_tensors[query_id],
                    str(meta.adduct),
                    str(meta.polarity),
                    seeds,
                    relations,
                    relation_neighbours,
                    seed_spectra,
                )
                attempted += 1
                available += int(features["available"] > 0)
                rotation_rows.append(
                    {
                        "fold": int(fold),
                        "unit_id": unit,
                        "source": str(row.source),
                        "polarity": str(row.polarity),
                        "query_id": query_id,
                        "candidate_id": str(row.candidate_id),
                        "truth_candidate_id": str(row.truth_candidate_id),
                        "truth_formula": str(row.truth_formula),
                        **features,
                    }
                )
        unit_reports[unit] = {
            "candidate_rotations": int(attempted),
            "rotations_with_exact_adduct_direct_relation": int(available),
            "fraction_available": float(available / attempted) if attempted else 0.0,
            "query_count": int(local["query_id"].nunique()),
            "seed_spectra": seed_report,
        }
        query_metadata_provenance[f"{unit}/positive_queries"] = sha256(positive_unit / "cache/queries.csv.gz")
        query_metadata_provenance[f"{unit}/negative_queries"] = sha256(negative_unit / "queries.csv.gz")
        query_metadata_provenance[f"{unit}/splits"] = sha256(splits_path)
        print(
            f"[B2] {unit}: rotations={attempted:,} transform_available={available:,}",
            flush=True,
        )

    rotations = pd.DataFrame(rotation_rows)
    if rotations.empty:
        raise RuntimeError("no reaction-transform rotations were constructed")
    expected_rotations = 0
    local_candidates = candidates
    if args.max_queries_per_unit > 0:
        selected: list[pd.DataFrame] = []
        for unit, group in candidates.groupby("unit_id", sort=False):
            keep = sorted(group["query_id"].astype(str).unique())[: args.max_queries_per_unit]
            selected.append(group.loc[group["query_id"].astype(str).isin(keep)])
        local_candidates = pd.concat(selected, ignore_index=True)
    for unit, group in local_candidates.groupby("unit_id", sort=False):
        split = pd.read_csv(args.manifest_root / str(unit) / "identity_splits.csv.gz")
        heldout_counts = (
            split.loc[split["role"].eq("heldout")]
            .groupby(split.loc[split["role"].eq("heldout"), "ik14"].astype(str))["fold"]
            .nunique()
            .to_dict()
        )
        for identity, count in group["truth_candidate_id"].astype(str).value_counts().items():
            if identity not in heldout_counts:
                raise RuntimeError(f"{unit}: truth identity absent from heldout rotations: {identity}")
            expected_rotations += int(count) * int(heldout_counts[identity])
    if len(rotations) != expected_rotations:
        raise RuntimeError(
            f"rotation cardinality mismatch: expected={expected_rotations} got={len(rotations)}"
        )
    aggregated = aggregate_rotations(rotations)
    merged = local_candidates.merge(
        aggregated, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
    )
    if len(merged) != len(local_candidates) or merged[REACTION_FEATURES].isna().any().any():
        raise RuntimeError("reaction feature aggregation lost candidate rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rotations_path = args.output_dir / "candidate_rotation_features.csv.gz"
    candidate_path = args.output_dir / "candidate_transform_features.csv.gz"
    rotations.to_csv(rotations_path, index=False, compression="gzip")
    merged.to_csv(candidate_path, index=False, compression="gzip")

    headroom = {
        name: evidence_headroom(merged, name)
        for name in (
            "rxn_transform_concordance_top3_mean",
            "rxn_transform_specificity_top3_mean",
            "rxn_theoretical_hybrid_top3_mean",
            "rxn_wrong_sign_concordance_top3_mean",
        )
    }
    opened_development = args.max_queries_per_unit == 0
    recipe_features = {
        "catalog_opportunity": OPPORTUNITY_FEATURES,
        "generic_edge": GENERIC_EDGE_FEATURES,
        "wrong_shift_control": WRONG_SHIFT_FEATURES,
        "reaction_transform": TRANSFORM_FEATURES,
    }
    transitions: dict[str, pd.DataFrame] = {}
    recipe_reports: dict[str, dict] = {}
    if opened_development:
        for offset, (name, features) in enumerate(recipe_features.items()):
            result, folds = run_source_loso(merged, name, features)
            transitions[name] = result
            transition_path = args.output_dir / f"{name}__transitions.csv.gz"
            result.to_csv(transition_path, index=False, compression="gzip")
            recipe_reports[name] = {
                "features": features,
                "pooled": summarize(result),
                "by_source": {
                    source: summarize(result.loc[result["source"].eq(source)])
                    for source in EXPECTED_SOURCES
                },
                "by_polarity": {
                    polarity: summarize(result.loc[result["polarity"].eq(polarity)])
                    for polarity in ("positive", "negative")
                },
                "folds": folds,
                "transitions_sha256": sha256(transition_path),
            }
        transform = transitions["reaction_transform"]
        generic = transitions["generic_edge"]
        wrong = transitions["wrong_shift_control"]
        primary = {
            "transition": paired_transition(transform, generic),
            "identity_cluster_bootstrap": paired_cluster_bootstrap(
                transform, generic, "truth_candidate_id", args.bootstrap_resamples, args.seed + 11
            ),
            "formula_cluster_bootstrap": paired_cluster_bootstrap(
                transform, generic, "truth_formula", args.bootstrap_resamples, args.seed + 12
            ),
            "by_source": {
                source: paired_transition(
                    transform.loc[transform["source"].eq(source)],
                    generic.loc[generic["source"].eq(source)],
                )
                | {
                    "delta": float(
                        transform.loc[transform["source"].eq(source), "gated_correct"].mean()
                        - generic.loc[generic["source"].eq(source), "gated_correct"].mean()
                    )
                }
                for source in EXPECTED_SOURCES
            },
        }
        negative_control = {
            "transition": paired_transition(transform, wrong),
            "formula_cluster_bootstrap": paired_cluster_bootstrap(
                transform, wrong, "truth_formula", args.bootstrap_resamples, args.seed + 13
            ),
        }
        delta = float(primary["formula_cluster_bootstrap"]["mean"])
        source_deltas = [value["delta"] for value in primary["by_source"].values()]
        corrected = int(primary["transition"]["corrected_vs_right"])
        introduced = int(primary["transition"]["introduced_vs_right"])
        gates = {
            "transform_increment_ge_3pp": delta >= 0.03,
            "identity_ci_low_positive": primary["identity_cluster_bootstrap"]["ci_low"] > 0,
            "formula_ci_low_positive": primary["formula_cluster_bootstrap"]["ci_low"] > 0,
            "corrected_gt_2x_introduced": corrected > 2 * introduced,
            "corrected_identities_ge_25": int(
                transform.loc[
                    ~generic["gated_correct"].to_numpy(bool)
                    & transform["gated_correct"].to_numpy(bool),
                    "truth_candidate_id",
                ].nunique()
            ) >= 25,
            "at_least_3_of_4_sources_nonnegative": sum(value >= 0 for value in source_deltas) >= 3,
            "wrong_shift_control_beaten": negative_control["formula_cluster_bootstrap"]["ci_low"] > 0,
        }
        scientific_pass = bool(all(gates.values()))
    else:
        primary = {}
        negative_control = {}
        gates = {}
        scientific_pass = False

    report = {
        "status": "bioaware_b2_reaction_transform_action_complete",
        "formal": bool(opened_development),
        "schema_version": "bioaware_b2_reaction_transform_action_v1",
        "candidate_protocol": {
            "queries": int(merged["query_id"].nunique()),
            "candidate_rows": int(len(merged)),
            "truth_identities": int(merged["truth_candidate_id"].nunique()),
            "truth_formulas": int(merged["truth_formula"].nunique()),
            "sources": int(merged["source"].nunique()),
            "units": int(merged["unit_id"].nunique()),
            "baseline_recall1": float(
                merged[["query_id", "baseline_correct"]].drop_duplicates()["baseline_correct"].mean()
            ),
        },
        "relation_graph": relation_report,
        "unit_coverage": unit_reports,
        "rotation_rows": int(len(rotations)),
        "candidate_pairs_with_transform": int((merged["rxn_available_fraction"] > 0).sum()),
        "queries_with_any_transform": int(
            merged.groupby("query_id")["rxn_available_fraction"].max().gt(0).sum()
        ),
        "truth_candidates_with_transform": int(
            merged.loc[merged["is_positive"] & merged["rxn_available_fraction"].gt(0), "query_id"].nunique()
        ),
        "evidence_headroom": headroom,
        "recipes": recipe_reports,
        "primary_reaction_transform_vs_generic_edge": primary,
        "negative_control_reaction_transform_vs_wrong_shift": negative_control,
        "gates": gates,
        "scientific_pass": scientific_pass,
        "contracts": {
            "direct_reactions_only": True,
            "seed_identity_held_out_by_rotation": True,
            "exact_adduct_and_polarity_required": True,
            "candidate_features_truth_blind": True,
            "held_source_truth_identity_and_formula_purged": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "parameters": {
            "fragment_tolerance_da": FRAGMENT_TOLERANCE_DA,
            "mass_residual_sigma_da": MASS_RESIDUAL_SIGMA_DA,
            "mass_concordance_da": MASS_CONCORDANCE_DA,
            "ppm": args.ppm,
            "rt_sec": args.rt_sec,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "max_queries_per_unit": args.max_queries_per_unit,
        },
        "provenance": {
            "positive": positive_provenance,
            "negative": negative_provenance,
            "query_metadata": query_metadata_provenance,
            "network_edges": sha256(args.network_dir / "metdna2_emrn_edges.csv.gz"),
            "network_compounds": sha256(args.network_dir / "metdna2_emrn_compounds.csv.gz"),
            "rhea_pairs": sha256(args.rhea_pairs),
            "rotation_features": sha256(rotations_path),
            "candidate_features": sha256(candidate_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened-development action audit. A pass establishes a candidate-ranking action "
            "worth frozen external validation; it is not SOTA evidence and does not establish "
            "a shared-embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

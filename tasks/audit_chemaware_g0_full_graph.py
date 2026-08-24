"""Formal full-graph preflight for ChemAware fine-grained DreaMS finetuning.

This is a read-only data/signal audit.  It consumes the complete P3-disjoint
real candidate graph, aligns all 3,486 observed-rule vectors to spectrum pairs,
and answers whether a four-point training target is supported by enough real
errors and local evidence.  It does not train or select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from build_g8r_real_error_atlas import Cache, load_p3_identities  # noqa: E402
from chemaware_g0_core import nan_group_max, packed_jaccard, packed_mask  # noqa: E402
from g8r_p2_rank_fusion_core import (  # noqa: E402
    fuse_one_query,
    fusion_configuration_from_mapping,
    grouped_max,
    normalize_pair_features,
    strict_rank,
)
from g8r_p2_listwise_core import deterministic_formula_fold  # noqa: E402


DEFAULT_GRAPH = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_RULES = ROOT / "data/validation/g8r_chemaware_g0_rule_cache.npz"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_p2b_rank_fusion.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_chemaware_g0_full_audit.json"
DEFAULT_PRIORITY = ROOT / "data/validation/g8r_chemaware_g0_query_priority.csv.gz"
DEFAULT_PAIR_RULES = ROOT / "data/validation/g8r_chemaware_g0_pair_rule_features.npz"
DEFAULT_RULE_LEVEL = ROOT / "data/validation/g8r_chemaware_g0_rule_level.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--rule-cache", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--priority-output", type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument("--pair-rule-output", type=Path, default=DEFAULT_PAIR_RULES)
    parser.add_argument("--rule-level-output", type=Path, default=DEFAULT_RULE_LEVEL)
    parser.add_argument("--target-delta", type=float, default=0.04)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pair-block-size", type=int, default=20000)
    parser.add_argument("--allow-small-smoke", action="store_true",
                        help="Tests only: permit a non-formal graph/rule cache and skip formal gates.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def percentile(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if len(values) == 0:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def strict_result(molecule_scores: np.ndarray) -> tuple[bool, int, float, int]:
    values = np.asarray(molecule_scores, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("candidate group needs a positive and a negative")
    rank, _, margin = strict_rank(values, 0)
    top = int(np.argmax(values))
    return rank == 1, rank, margin, top


def nan_rule_result(molecule_scores: np.ndarray) -> tuple[bool, bool, float]:
    values = np.asarray(molecule_scores, dtype=np.float64)
    if not np.isfinite(values[0]) or not np.any(np.isfinite(values[1:])):
        return False, False, float("nan")
    finite = np.where(np.isfinite(values), values, -np.inf)
    rank, _, margin = strict_rank(finite, 0)
    return True, rank == 1, margin


def identity_equal_mean(values: np.ndarray, identities: np.ndarray) -> float:
    frame = pd.DataFrame({"identity": identities.astype(str), "value": values.astype(float)})
    return float(frame.groupby("identity", sort=False)["value"].mean().mean())


def build_pair_rule_features(
    graph: Cache,
    rule_rows: np.ndarray,
    packed: np.ndarray,
    rule_library: np.ndarray,
    rule_category: np.ndarray,
    rule_semantics: np.ndarray,
    n_rules: int,
    block_size: int,
) -> tuple[np.ndarray, list[str]]:
    row_to_rule = {int(row): index for index, row in enumerate(rule_rows)}
    missing_query = [int(row) for row in graph.query_row if int(row) not in row_to_rule]
    missing_pair = [int(row) for row in np.unique(graph.pair_candidate_row) if int(row) not in row_to_rule]
    if missing_query or missing_pair:
        raise RuntimeError(
            f"rule cache does not cover graph: query={len(missing_query)}, pair={len(missing_pair)}"
        )
    query_pair_ptr = graph.molecule_ptr[graph.query_ptr]
    pair_query = np.empty(len(graph.features), dtype=np.int64)
    for query, (left, right) in enumerate(zip(query_pair_ptr[:-1], query_pair_ptr[1:])):
        pair_query[int(left):int(right)] = query
    query_rule_index = np.asarray([row_to_rule[int(row)] for row in graph.query_row], dtype=np.int64)
    pair_rule_index = np.asarray([
        row_to_rule[int(row)] for row in graph.pair_candidate_row
    ], dtype=np.int64)

    names = [
        "rule_all", "rule_core", "rule_massbank", "rule_cf",
        "rule_fragment_neutral_loss", "rule_precursor_offset",
        "rule_iso", "rule_hr", "rule_nr", "rule_ee",
    ]
    masks = [
        packed_mask(np.arange(n_rules), n_rules),
        packed_mask(np.flatnonzero(rule_library == "core"), n_rules),
        packed_mask(np.flatnonzero(rule_library == "massbank"), n_rules),
    ]
    masks.extend([
        packed_mask(np.flatnonzero(rule_category == "CF"), n_rules),
        packed_mask(np.flatnonzero(rule_semantics == "fragment_neutral_loss"), n_rules),
        packed_mask(np.flatnonzero(rule_semantics == "precursor_exact_mass_offset"), n_rules),
    ])
    for category in ("ISO", "HR", "NR", "EE"):
        masks.append(packed_mask(np.flatnonzero(rule_category == category), n_rules))
    output = np.full((len(graph.features), len(names)), np.nan, dtype=np.float32)
    for left in range(0, len(output), block_size):
        right = min(left + block_size, len(output))
        q = packed[query_rule_index[pair_query[left:right]]]
        c = packed[pair_rule_index[left:right]]
        for column, mask in enumerate(masks):
            output[left:right, column] = packed_jaccard(q, c, mask)
        if right % 100000 == 0 or right == len(output):
            print(f"[rule-pairs] {right:,}/{len(output):,}", flush=True)
    return output, names


def main() -> None:
    args = parse_args()
    if not 0 < args.target_delta < 1 or args.folds < 3 or args.pair_block_size < 100:
        raise ValueError("invalid G0 audit parameters")
    outputs = (args.output, args.priority_output, args.pair_rule_output, args.rule_level_output)
    if not args.overwrite and any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"refusing to overwrite G0 outputs: {existing}")
    for path in (args.graph, args.rule_cache, args.artifact, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    graph = Cache(args.graph)
    formal = not args.allow_small_smoke
    if formal and graph.n_queries < 15000:
        raise RuntimeError(f"formal G0 refuses small graph: {graph.n_queries} queries")
    p3_identities = load_p3_identities(args.p3_dir)
    overlap = set(map(str, graph.query_ik14)) & p3_identities
    if overlap:
        raise RuntimeError(f"G0 graph overlaps sealed P3 by {len(overlap)} identities")

    rule_report_path = args.rule_cache.with_suffix(".json")
    rule_report = json.loads(rule_report_path.read_text(encoding="utf-8"))
    if formal and (
        not rule_report.get("formal")
        or rule_report.get("status") != "chemaware_g0_rule_cache_complete"
    ):
        raise RuntimeError("formal full rule cache is required")
    if rule_report["provenance"]["candidate_graph_sha256"] != sha256_file(args.graph):
        raise RuntimeError("rule cache belongs to a different candidate graph")
    with np.load(args.rule_cache, allow_pickle=True) as body:
        rule_rows = np.asarray(body["hdf5_row"], dtype=np.int64)
        packed = np.asarray(body["packed_rule_hits"], dtype=np.uint8)
        n_rules = int(np.asarray(body["n_rules"]).reshape(-1)[0])
        rule_name = np.asarray(body["rule_name"], dtype=object)
        rule_category = np.asarray(body["rule_category"], dtype=object).astype(str)
        rule_library = np.asarray(body["rule_library"], dtype=object).astype(str)
        rule_source = np.asarray(body["rule_source"], dtype=object).astype(str)
        rule_semantics = np.asarray(body["rule_semantics"], dtype=object).astype(str)
        rule_declared_support = np.asarray(body["rule_declared_support"], dtype=np.int64)
        rule_spectrum_support = np.asarray(body["spectrum_support"], dtype=np.int64)
        rule_identity_support = np.asarray(body["identity_support"], dtype=np.int64)
    if n_rules != 3486 or packed.shape[0] != len(rule_rows):
        raise RuntimeError("unexpected combined rule cache shape")

    pair_rules, pair_rule_names = build_pair_rule_features(
        graph, rule_rows, packed, rule_library, rule_category, rule_semantics,
        n_rules, args.pair_block_size,
    )
    args.pair_rule_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_rule_pairs = args.pair_rule_output.with_name(args.pair_rule_output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_rule_pairs,
        feature_names=np.asarray(pair_rule_names, dtype=object),
        features=pair_rules,
        graph_sha256=np.asarray([sha256_file(args.graph)], dtype=object),
        rule_cache_sha256=np.asarray([sha256_file(args.rule_cache)], dtype=object),
    )
    temporary_rule_pairs.replace(args.pair_rule_output)

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    configuration = fusion_configuration_from_mapping(artifact["configuration"])
    selected_features = list(map(str, artifact["selected_features"]))
    feature_index = {name: index for index, name in enumerate(graph.feature_names)}
    selected_columns = np.asarray([feature_index[name] for name in selected_features], dtype=np.int64)
    query_pair_ptr = graph.molecule_ptr[graph.query_ptr]
    selected_pairs = graph.features[:, selected_columns]
    normalized_pairs = normalize_pair_features(
        selected_pairs, query_pair_ptr, configuration.normalization,
    )

    reachable_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    with h5py.File(args.data, "r") as handle:
        instrument = {int(row): decode(value) for row, value in zip(
            reachable_rows, handle["INSTRUMENT_TYPE"][reachable_rows],
        )}
        collision = {int(row): float(value) for row, value in zip(
            reachable_rows, handle["COLLISION_ENERGY"][reachable_rows],
        )}

    n_queries = graph.n_queries
    candidate_counts = np.diff(graph.query_ptr)
    negative_counts = candidate_counts - 1
    eligible = negative_counts >= 2
    eligible_index = np.flatnonzero(eligible)
    identity_counts = Counter(map(str, graph.query_ik14[eligible]))
    identity_weight = np.asarray([
        (1.0 / identity_counts[str(identity)]) if is_eligible else 0.0
        for identity, is_eligible in zip(graph.query_ik14, eligible)
    ], dtype=np.float64)

    base_correct = np.zeros(n_queries, dtype=bool)
    base_rank = np.zeros(n_queries, dtype=np.int32)
    base_margin = np.full(n_queries, np.nan, dtype=np.float64)
    p2b_correct = np.zeros(n_queries, dtype=bool)
    any_raw_correct = np.zeros(n_queries, dtype=bool)
    raw_correct_count = np.zeros(n_queries, dtype=np.int16)
    any_rule_correct = np.zeros(n_queries, dtype=bool)
    rule_evidence_present = np.zeros(n_queries, dtype=bool)
    best_rule_margin = np.full(n_queries, np.nan, dtype=np.float64)
    positive_spectra = np.zeros(n_queries, dtype=np.int32)
    positive_min_cosine = np.full(n_queries, np.nan, dtype=np.float64)
    positive_median_cosine = np.full(n_queries, np.nan, dtype=np.float64)
    positive_max_cosine = np.full(n_queries, np.nan, dtype=np.float64)
    top_negative_grade = np.full(n_queries, -9, dtype=np.int8)
    cross_instrument_positive = np.zeros(n_queries, dtype=bool)
    cross_ce_positive = np.zeros(n_queries, dtype=bool)
    best_raw_feature = np.full(n_queries, "", dtype=object)
    rule_positive_advantage = np.full(n_queries, np.nan, dtype=np.float64)

    positive_shared_count = np.zeros(n_rules, dtype=np.int64)
    negative_shared_count = np.zeros(n_rules, dtype=np.int64)
    positive_shared_weight = np.zeros(n_rules, dtype=np.float64)
    negative_shared_weight = np.zeros(n_rules, dtype=np.float64)
    error_positive_shared_weight = np.zeros(n_rules, dtype=np.float64)
    error_negative_shared_weight = np.zeros(n_rules, dtype=np.float64)
    row_to_rule = {int(row): index for index, row in enumerate(rule_rows)}

    raw_columns = list(range(1, len(graph.feature_names)))
    for query in range(n_queries):
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        local_ptr = graph.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        baseline_molecule = grouped_max(graph.features[pair_left:pair_right, 0], local_ptr)
        correct, rank, margin, top = strict_result(baseline_molecule)
        base_correct[query] = correct
        base_rank[query] = rank
        base_margin[query] = margin
        top_negative_local = 1 + int(np.argmax(baseline_molecule[1:]))
        top_negative_grade[query] = int(graph.molecule_mces_grade[molecule_left + top_negative_local])

        positive_pair_left = pair_left
        positive_pair_right = pair_left + int(local_ptr[1])
        pos_scores = graph.features[positive_pair_left:positive_pair_right, 0]
        positive_spectra[query] = len(pos_scores)
        positive_min_cosine[query] = float(np.min(pos_scores))
        positive_median_cosine[query] = float(np.median(pos_scores))
        positive_max_cosine[query] = float(np.max(pos_scores))
        query_row = int(graph.query_row[query])
        q_instrument = instrument.get(query_row, "")
        q_ce = collision.get(query_row, float("nan"))
        positive_rows = graph.pair_candidate_row[positive_pair_left:positive_pair_right]
        cross_instrument_positive[query] = any(
            instrument.get(int(row), "") != q_instrument for row in positive_rows
        )
        cross_ce_positive[query] = any(
            np.isfinite(q_ce)
            and np.isfinite(collision.get(int(row), float("nan")))
            and abs(collision[int(row)] - q_ce) >= 10.0
            for row in positive_rows
        )

        raw_winners = []
        for column in raw_columns:
            molecule_scores = grouped_max(graph.features[pair_left:pair_right, column], local_ptr)
            raw_winners.append(strict_result(molecule_scores)[0])
        raw_winners_array = np.asarray(raw_winners, dtype=bool)
        any_raw_correct[query] = bool(np.any(raw_winners_array))
        raw_correct_count[query] = int(np.sum(raw_winners_array))
        if np.any(raw_winners_array):
            best_raw_feature[query] = graph.feature_names[1 + int(np.flatnonzero(raw_winners_array)[0])]

        rule_winners = []
        rule_margins = []
        for column in range(pair_rules.shape[1]):
            molecule_scores = nan_group_max(pair_rules[pair_left:pair_right, column], local_ptr)
            present, rule_correct, rule_margin = nan_rule_result(molecule_scores)
            rule_winners.append(present and rule_correct)
            if present:
                rule_margins.append(rule_margin)
            if column == 0 and present:
                rule_positive_advantage[query] = rule_margin
        rule_evidence_present[query] = bool(rule_margins)
        any_rule_correct[query] = bool(np.any(rule_winners))
        if rule_margins:
            best_rule_margin[query] = float(np.max(rule_margins))

        p2b_molecule, _, _ = fuse_one_query(
            normalized_pairs[pair_left:pair_right],
            selected_pairs[pair_left:pair_right, 0],
            local_ptr,
            np.asarray(configuration.weights, dtype=np.float64),
            (1, 2, 3),
            configuration.min_support,
            configuration.min_advantage,
        )
        p2b_correct[query] = strict_result(p2b_molecule)[0]

        if eligible[query]:
            positive_pair = positive_pair_left + int(np.argmax(pos_scores))
            neg_molecule = molecule_left + top_negative_local
            neg_left = int(graph.molecule_ptr[neg_molecule])
            neg_right = int(graph.molecule_ptr[neg_molecule + 1])
            negative_pair = neg_left + int(np.argmax(graph.features[neg_left:neg_right, 0]))
            q_bits = np.unpackbits(
                packed[row_to_rule[query_row]], bitorder="little", count=n_rules,
            ).astype(bool)
            p_bits = np.unpackbits(
                packed[row_to_rule[int(graph.pair_candidate_row[positive_pair])]],
                bitorder="little", count=n_rules,
            ).astype(bool)
            n_bits = np.unpackbits(
                packed[row_to_rule[int(graph.pair_candidate_row[negative_pair])]],
                bitorder="little", count=n_rules,
            ).astype(bool)
            shared_positive = q_bits & p_bits
            shared_negative = q_bits & n_bits
            weight = identity_weight[query]
            positive_shared_count += shared_positive
            negative_shared_count += shared_negative
            positive_shared_weight += weight * shared_positive
            negative_shared_weight += weight * shared_negative
            if not correct:
                error_positive_shared_weight += weight * shared_positive
                error_negative_shared_weight += weight * shared_negative

        if (query + 1) % 1000 == 0 or query + 1 == n_queries:
            print(f"[audit] {query + 1:,}/{n_queries:,}", flush=True)

    eligible_identity = graph.query_ik14[eligible].astype(str)
    eligible_formula = graph.query_formula[eligible].astype(str)
    eligible_base = base_correct[eligible]
    eligible_error = ~eligible_base
    eligible_raw_rescue = eligible_error & any_raw_correct[eligible]
    eligible_rule_rescue = eligible_error & any_rule_correct[eligible]
    eligible_p2b_rescue = eligible_error & p2b_correct[eligible]
    required_query_corrections = int(math.ceil(args.target_delta * len(eligible_index)))
    n_eligible_identities = len(set(eligible_identity))
    required_identity_weight = args.target_delta * n_eligible_identities
    recoverable_identity_weight = float(np.sum(identity_weight[eligible] * any_raw_correct[eligible] * eligible_error))
    error_identity_weight = float(np.sum(identity_weight[eligible] * eligible_error))

    reachability = {}
    for bound in (0.03, 0.06, 0.10):
        reachable = eligible_error & (base_margin[eligible] > -2.0 * bound)
        raw_reachable = reachable & any_raw_correct[eligible]
        reachability[f"delta_bound_{bound:.2f}"] = {
            "geometrically_reachable_errors": int(np.sum(reachable)),
            "reachable_errors_with_raw_signal": int(np.sum(raw_reachable)),
            "query_delta_headroom": float(np.mean(raw_reachable)),
        }

    fold = np.asarray([
        deterministic_formula_fold(str(value), args.folds) for value in eligible_formula
    ], dtype=np.int8)
    per_fold = []
    for value in range(args.folds):
        mask = fold == value
        per_fold.append({
            "fold": value,
            "queries": int(np.sum(mask)),
            "identities": int(len(set(eligible_identity[mask]))),
            "formulas": int(len(set(eligible_formula[mask]))),
            "near_queries": int(np.sum(graph.query_has_near[eligible][mask])),
            "baseline_errors": int(np.sum(eligible_error[mask])),
            "raw_oracle_recoverable_errors": int(np.sum(eligible_raw_rescue[mask])),
            "rule_oracle_recoverable_errors": int(np.sum(eligible_rule_rescue[mask])),
        })

    transition = np.full(n_queries, "protected_correct", dtype=object)
    transition[(~base_correct) & p2b_correct] = "corrected"
    transition[base_correct & (~p2b_correct)] = "introduced"
    transition[(~base_correct) & (~p2b_correct)] = "persistent_wrong"
    priority_tier = np.full(n_queries, 5, dtype=np.int8)
    priority_tier[base_margin <= 0.05] = 4
    priority_tier[graph.query_has_near] = np.minimum(priority_tier[graph.query_has_near], 3)
    priority_tier[base_correct & (~p2b_correct)] = 2
    priority_tier[~base_correct] = 1
    priority_tier[(~base_correct) & p2b_correct] = 0
    priority = pd.DataFrame({
        "query_index": np.arange(n_queries),
        "query_row": graph.query_row,
        "query_ik14": graph.query_ik14.astype(str),
        "query_formula": graph.query_formula.astype(str),
        "eligible_two_negative_molecules": eligible,
        "priority_tier": priority_tier,
        "dreams_correct": base_correct,
        "dreams_rank": base_rank,
        "dreams_margin": base_margin,
        "p2b_correct": p2b_correct,
        "transition": transition,
        "has_near": graph.query_has_near,
        "top_negative_mces_grade": top_negative_grade,
        "candidate_molecules": candidate_counts,
        "negative_molecules": negative_counts,
        "positive_spectra": positive_spectra,
        "positive_min_cosine": positive_min_cosine,
        "positive_median_cosine": positive_median_cosine,
        "positive_max_cosine": positive_max_cosine,
        "cross_instrument_positive": cross_instrument_positive,
        "cross_ce_ge10_positive": cross_ce_positive,
        "raw_feature_can_rank_positive_first": any_raw_correct,
        "raw_features_ranking_positive_first": raw_correct_count,
        "first_rescuing_raw_feature": best_raw_feature,
        "rule_evidence_present": rule_evidence_present,
        "rule_feature_can_rank_positive_first": any_rule_correct,
        "best_rule_margin": best_rule_margin,
        "all_rule_positive_advantage": rule_positive_advantage,
    })
    priority.sort_values(
        ["eligible_two_negative_molecules", "priority_tier", "dreams_margin", "query_ik14"],
        ascending=[False, True, True, True], inplace=True, kind="mergesort",
    )
    args.priority_output.parent.mkdir(parents=True, exist_ok=True)
    priority.to_csv(args.priority_output, index=False, compression="gzip")

    identity_denominator = float(n_eligible_identities)
    rule_level = pd.DataFrame({
        "rule_index": np.arange(n_rules),
        "rule_name": rule_name.astype(str),
        "category": rule_category,
        "library": rule_library,
        "source": rule_source,
        "semantics": rule_semantics,
        "declared_support": rule_declared_support,
        "spectrum_support": rule_spectrum_support,
        "identity_support": rule_identity_support,
        "shared_with_positive_count": positive_shared_count,
        "shared_with_hardest_negative_count": negative_shared_count,
        "identity_equal_positive_shared_rate": positive_shared_weight / identity_denominator,
        "identity_equal_negative_shared_rate": negative_shared_weight / identity_denominator,
        "identity_equal_shared_rate_delta": (
            positive_shared_weight - negative_shared_weight
        ) / identity_denominator,
        "error_identity_equal_shared_rate_delta": (
            error_positive_shared_weight - error_negative_shared_weight
        ) / identity_denominator,
    })
    rule_level.to_csv(args.rule_level_output, index=False, compression="gzip")

    n_near = int(np.sum(graph.query_has_near[eligible]))
    report = {
        "status": "chemaware_g0_full_graph_passed" if formal else "chemaware_g0_full_graph_smoke",
        "formal": formal,
        "purpose": "training-data and signal preflight; no model fitting and no sealed-test evaluation",
        "full_graph": {
            "queries": n_queries,
            "identities": int(len(set(map(str, graph.query_ik14)))),
            "candidate_molecules": int(len(graph.molecule_label)),
            "spectrum_pairs": int(len(graph.features)),
        },
        "trainable_graph": {
            "queries_with_at_least_two_negative_molecules": int(np.sum(eligible)),
            "identities": n_eligible_identities,
            "formulas": int(len(set(eligible_formula))),
            "near_queries": n_near,
            "candidate_molecules_per_query": percentile(candidate_counts[eligible]),
            "negative_molecules_per_query": percentile(negative_counts[eligible]),
            "positive_spectra_per_query": percentile(positive_spectra[eligible]),
            "cross_instrument_positive_queries": int(np.sum(cross_instrument_positive[eligible])),
            "cross_ce_ge10_positive_queries": int(np.sum(cross_ce_positive[eligible])),
            "queries_per_identity": percentile(np.asarray(list(identity_counts.values()))),
        },
        "official_dreams": {
            "query_recall1": float(np.mean(eligible_base)),
            "identity_equal_recall1": identity_equal_mean(eligible_base, eligible_identity),
            "query_errors": int(np.sum(eligible_error)),
            "identity_equal_error_mass": error_identity_weight,
            "error_margin": percentile(base_margin[eligible][eligible_error]),
        },
        "four_point_target": {
            "target_delta": args.target_delta,
            "required_query_net_corrections": required_query_corrections,
            "available_query_errors": int(np.sum(eligible_error)),
            "raw_oracle_recoverable_query_errors": int(np.sum(eligible_raw_rescue)),
            "rule_oracle_recoverable_query_errors": int(np.sum(eligible_rule_rescue)),
            "p2b_recoverable_query_errors": int(np.sum(eligible_p2b_rescue)),
            "required_identity_equal_correction_mass": required_identity_weight,
            "available_identity_equal_error_mass": error_identity_weight,
            "raw_oracle_identity_equal_recoverable_mass": recoverable_identity_weight,
            "oracle_warning": "Per-query best feature uses the answer and is headroom only, never a model result.",
        },
        "bounded_residual_reachability": reachability,
        "p2b_training_signal": {
            "corrected": int(np.sum((~base_correct[eligible]) & p2b_correct[eligible])),
            "introduced_safety_replay": int(np.sum(base_correct[eligible] & (~p2b_correct[eligible]))),
            "persistent_wrong": int(np.sum((~base_correct[eligible]) & (~p2b_correct[eligible]))),
        },
        "rule_evidence": {
            "rules": n_rules,
            "core_rules": int(np.sum(rule_library == "core")),
            "massbank_rules": int(np.sum(rule_library == "massbank")),
            "queries_with_any_comparable_rule_evidence": int(np.sum(rule_evidence_present[eligible])),
            "official_errors_with_rule_oracle_headroom": int(np.sum(eligible_rule_rescue)),
            "semantics": "observed spectrum mass motifs used for evidence stratification, never identity labels",
        },
        "formula_isolated_folds": per_fold,
        "p3_query_identity_overlap": 0,
        "provenance": {
            "graph": str(args.graph.resolve()),
            "graph_sha256": sha256_file(args.graph),
            "rule_cache": str(args.rule_cache.resolve()),
            "rule_cache_sha256": sha256_file(args.rule_cache),
            "pair_rule_features_sha256": sha256_file(args.pair_rule_output),
            "p2b_artifact_sha256": sha256_file(args.artifact),
            "hdf5_sha256": sha256_file(args.data),
        },
    }
    gates = {
        "trainable_queries_ge_15000": int(np.sum(eligible)) >= 15000,
        "identities_ge_2500": n_eligible_identities >= 2500,
        "near_queries_ge_1500": n_near >= 1500,
        "query_error_pool_covers_four_points": int(np.sum(eligible_error)) >= required_query_corrections,
        "raw_oracle_query_headroom_covers_four_points": int(np.sum(eligible_raw_rescue)) >= required_query_corrections,
        "identity_error_pool_covers_four_points": error_identity_weight >= required_identity_weight,
        "identity_raw_headroom_covers_four_points": recoverable_identity_weight >= required_identity_weight,
        "every_fold_has_100_near": all(row["near_queries"] >= 100 for row in per_fold),
        "every_fold_has_20_errors": all(row["baseline_errors"] >= 20 for row in per_fold),
        "full_rule_library_aligned": n_rules == 3486,
    }
    gates["pass"] = all(gates.values())
    report["gates"] = gates
    if formal:
        report["status"] = (
            "chemaware_g0_full_graph_passed" if gates["pass"] else "chemaware_g0_full_graph_failed"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if formal and not gates["pass"]:
        raise RuntimeError(f"ChemAware G0 failed; see {args.output}")


if __name__ == "__main__":
    main()

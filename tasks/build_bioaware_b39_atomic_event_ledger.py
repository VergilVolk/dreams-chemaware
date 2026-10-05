#!/usr/bin/env python
"""Build the outcome-blind BioAware B39 atomic reaction-event ledger.

B38 demonstrated that a candidate-level direct-path fraction does not improve
the B37 catalogue prior.  B39 therefore preserves the identity of each visible
seed, reaction and candidate, annotates Rhea hyperedge completeness, and marks
legacy B3/B9 evidence at its honest (candidate-context aggregate) resolution.

This stage never fits a model, reads an embedding, or reads ranking outcomes.
It is a semantic and identifiability ledger for the later fixed-action scan.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b12_multicohort_catalog_action import build_universe  # noqa: E402
from audit_bioaware_b36_reaction_specificity_ablation import prepare_candidates  # noqa: E402
from audit_bioaware_b38_m0_explicit_path_coverage import (  # noqa: E402
    EXPECTED_DOMAINS,
    INTERNAL_DOMAINS,
    SeedResolver,
    fold_from_stratum,
    load_direct_edges,
    normalise_reaction_id,
)


FORBIDDEN_LEDGER_COLUMNS = frozenset(
    {
        "truth_candidate_id",
        "truth_formula",
        "is_positive",
        "baseline_candidate_id",
        "baseline_correct",
        "corrected",
        "introduced",
        "final_correct",
        "delta",
        "spectral_score",
        "baseline_gap",
    }
)

SPECTRAL_PROXY_COLUMNS = (
    "edge_available",
    "matched_edges",
    "control_tier0_fraction",
    "control_tier1_fraction",
    "truncated_direct_real_mean",
    "truncated_direct_control_mean",
    "truncated_direct_excess_mean",
    "neutral_loss_real_mean",
    "neutral_loss_control_mean",
    "neutral_loss_excess_mean",
    "modified_cosine_real_mean",
    "modified_cosine_control_mean",
    "modified_cosine_excess_mean",
    "dual_view_real_mean",
    "dual_view_control_mean",
    "dual_view_excess_mean",
    "kgmn_support_real_mean",
    "kgmn_support_control_mean",
    "kgmn_support_excess_mean",
)
COABUNDANCE_PROXY_COLUMNS = (
    "available",
    "neighbours",
    "actual_absolute",
    "actual_positive",
    "actual_negative",
    "random_absolute",
    "random_positive",
    "random_negative",
    "abs_excess",
    "positive_excess",
    "negative_excess",
    "multiwitness",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} misses columns: {sorted(missing)}")


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def source_semantics(source: str) -> dict[str, Any]:
    source = str(source)
    if source in INTERNAL_DOMAINS:
        return {
            "context_class": "S",
            "context_semantics": "synthetic_rotation",
            "context_selection": "held_identity_keyed_rotation",
            "prospective_unknown": False,
        }
    if source == "ST001154_same_formula_10ppm":
        return {
            "context_class": "L",
            "context_semantics": "sample_local_leave_one_seed_out",
            "context_selection": "sample_seed_set_with_evaluation_identity_removed",
            "prospective_unknown": False,
        }
    if source == "KGMN200STD_hidden_seed":
        return {
            "context_class": "H",
            "context_semantics": "hidden_standard_repeat",
            "context_selection": "preregistered_hidden_seed_repeat",
            "prospective_unknown": False,
        }
    raise RuntimeError(f"unsupported source semantics: {source}")


def reaction_participant_lookup(
    participants_path: Path,
    reactions_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    participants = pd.read_csv(participants_path)
    reactions = pd.read_csv(reactions_path)
    require_columns(
        participants,
        ("compound_id", "reaction_id", "side", "stoichiometry", "is_currency"),
        "Rhea participants",
    )
    require_columns(
        reactions,
        ("reaction_id", "direction_semantics", "n_left", "n_right"),
        "Rhea reactions",
    )
    participants = participants.copy()
    participants["reaction_key"] = participants["reaction_id"].map(normalise_reaction_id)
    participants["participant_identity"] = participants["compound_id"].astype(str)
    participants["is_currency_bool"] = parse_bool(participants["is_currency"])
    participants["stoichiometry_numeric"] = pd.to_numeric(
        participants["stoichiometry"], errors="coerce"
    ).fillna(1.0)
    reactions = reactions.copy()
    reactions["reaction_key"] = reactions["reaction_id"].map(normalise_reaction_id)
    reaction_metadata = {
        str(row.reaction_key): {
            "direction_semantics": str(row.direction_semantics),
            "catalogue_n_left": int(row.n_left),
            "catalogue_n_right": int(row.n_right),
        }
        for row in reactions.itertuples(index=False)
        if str(row.reaction_key)
    }

    lookup: dict[str, dict[str, Any]] = {}
    identity_noop = 0
    for reaction, group in participants.groupby("reaction_key", sort=False):
        reaction = str(reaction)
        if not reaction:
            continue
        noncurrency = group.loc[~group["is_currency_bool"]].copy()
        side_maps: dict[str, dict[str, float]] = {}
        for side in ("left", "right"):
            local = noncurrency.loc[noncurrency["side"].astype(str).eq(side)]
            side_maps[side] = {
                str(identity): float(values["stoichiometry_numeric"].sum())
                for identity, values in local.groupby("participant_identity", sort=False)
            }
        noop = side_maps["left"] == side_maps["right"] and bool(side_maps["left"])
        identity_noop += int(noop)
        meta = reaction_metadata.get(reaction, {})
        lookup[reaction] = {
            "direction_semantics": str(
                meta.get("direction_semantics", "reaction_direction_unknown")
            ),
            "catalogue_n_left": int(meta.get("catalogue_n_left", 0)),
            "catalogue_n_right": int(meta.get("catalogue_n_right", 0)),
            "noncurrency_left": side_maps["left"],
            "noncurrency_right": side_maps["right"],
            "noncurrency_left_count": int(len(side_maps["left"])),
            "noncurrency_right_count": int(len(side_maps["right"])),
            "noncurrency_left_stoichiometry": float(sum(side_maps["left"].values())),
            "noncurrency_right_stoichiometry": float(sum(side_maps["right"].values())),
            "identity_noop": bool(noop),
        }
    return lookup, {
        "participant_rows": int(len(participants)),
        "reactions_with_participants": int(len(lookup)),
        "identity_noop_reactions": int(identity_noop),
    }


def hyperedge_features(
    reaction: str,
    seed_identity: str,
    candidate_identity: str,
    seed_side: str,
    candidate_side: str,
    visible_seeds: set[str],
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reaction = normalise_reaction_id(reaction)
    if not reaction or reaction not in lookup:
        return {
            "hyperedge_resolution": "not_available_for_non_rhea_or_unmapped_event",
            "reaction_identity_noop": False,
            "noncurrency_seed_side_count": 0,
            "noncurrency_candidate_side_count": 0,
            "noncurrency_seed_side_stoichiometry": 0.0,
            "noncurrency_candidate_side_stoichiometry": 0.0,
            "required_context_participant_count": 0,
            "observed_context_participant_count": 0,
            "hyperedge_context_completeness": math.nan,
            "hyperedge_context_complete": False,
            "missing_context_signature": "unmapped",
        }
    if seed_side not in {"left", "right"} or candidate_side not in {"left", "right"}:
        return {
            "hyperedge_resolution": "participant_side_unresolved",
            "reaction_identity_noop": bool(lookup[reaction]["identity_noop"]),
            "noncurrency_seed_side_count": 0,
            "noncurrency_candidate_side_count": 0,
            "noncurrency_seed_side_stoichiometry": 0.0,
            "noncurrency_candidate_side_stoichiometry": 0.0,
            "required_context_participant_count": 0,
            "observed_context_participant_count": 0,
            "hyperedge_context_completeness": math.nan,
            "hyperedge_context_complete": False,
            "missing_context_signature": "side_unresolved",
        }

    seed_map = lookup[reaction][f"noncurrency_{seed_side}"]
    candidate_map = lookup[reaction][f"noncurrency_{candidate_side}"]
    required = (set(seed_map) | set(candidate_map)) - {
        str(seed_identity), str(candidate_identity)
    }
    observed_universe = set(visible_seeds) | {str(seed_identity), str(candidate_identity)}
    observed = required & observed_universe
    missing = sorted(required - observed_universe)
    completeness = 1.0 if not required else len(observed) / len(required)
    return {
        "hyperedge_resolution": "rhea_noncurrency_participants_resolved",
        "reaction_identity_noop": bool(lookup[reaction]["identity_noop"]),
        "noncurrency_seed_side_count": int(len(seed_map)),
        "noncurrency_candidate_side_count": int(len(candidate_map)),
        "noncurrency_seed_side_stoichiometry": float(sum(seed_map.values())),
        "noncurrency_candidate_side_stoichiometry": float(sum(candidate_map.values())),
        "required_context_participant_count": int(len(required)),
        "observed_context_participant_count": int(len(observed)),
        "hyperedge_context_completeness": float(completeness),
        "hyperedge_context_complete": bool(completeness == 1.0),
        "missing_context_signature": ";".join(missing) if missing else "complete",
    }


def connected_components(
    edges: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, str], dict[str, int]]:
    adjacency = {
        str(node): set(str(neighbour) for neighbour in neighbours)
        for node, neighbours in edges.items()
    }
    component_by_node: dict[str, str] = {}
    degree = {node: len(neighbours) for node, neighbours in adjacency.items()}
    for start in sorted(adjacency):
        if start in component_by_node:
            continue
        stack = [start]
        members: list[str] = []
        while stack:
            node = stack.pop()
            if node in component_by_node:
                continue
            component_by_node[node] = "pending"
            members.append(node)
            stack.extend(sorted(adjacency.get(node, set()) - set(component_by_node)))
        component_id = min(members)
        for node in members:
            component_by_node[node] = component_id
    return component_by_node, degree


def add_candidate_specificity(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "query_id", "seed_stratum", "seed_identity", "edge_key"]
    support = (
        events.groupby(keys, sort=False)["candidate_id"]
        .nunique()
        .rename("event_supported_candidate_count")
        .reset_index()
    )
    output = events.merge(support, on=keys, how="left", validate="many_to_one")
    output["event_candidate_specific"] = output["event_supported_candidate_count"].eq(1)
    output["event_candidate_specificity"] = (
        1.0 / output["event_supported_candidate_count"].astype(float)
    )
    return output


def load_proxy_lookup(
    path: Path,
    columns: tuple[str, ...],
    label: str,
) -> dict[tuple[str, str, int], dict[str, float]]:
    usecols = ["query_id", "candidate_id", "fold", *columns]
    frame = pd.read_csv(path, usecols=lambda value: value in set(usecols))
    require_columns(frame, usecols, label)
    if frame.duplicated(["query_id", "candidate_id", "fold"]).any():
        raise RuntimeError(f"{label} has duplicate candidate-context rows")
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.query_id), str(row.candidate_id), int(row.fold))
        lookup[key] = {
            column: float(getattr(row, column))
            if np.isfinite(float(getattr(row, column))) else math.nan
            for column in columns
        }
    return lookup


def stable_dependency_id(parts: Iterable[Any]) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def st_leave_one_seed_out_audit(queries_path: Path, seeds_path: Path) -> dict[str, Any]:
    queries = pd.read_csv(
        queries_path, usecols=["query_id", "sample_id", "truth_candidate_id"]
    )
    seeds = pd.read_csv(seeds_path, usecols=["sample_id", "ik14"])
    seed_sets = {
        str(sample): set(group["ik14"].astype(str))
        for sample, group in seeds.groupby("sample_id", sort=False)
    }
    present = [
        str(row.truth_candidate_id) in seed_sets.get(str(row.sample_id), set())
        for row in queries.itertuples(index=False)
    ]
    return {
        "manifest_queries": int(len(queries)),
        "evaluation_identity_already_in_sample_seed_set": int(sum(present)),
        "fraction": float(np.mean(present)) if present else 0.0,
        "interpretation": (
            "sample-local leave-one-seed-out counterfactual recovery; "
            "not prospective unknown annotation"
        ),
    }


def seed_feature_lookup(args: argparse.Namespace) -> dict[tuple[str, str, str], str]:
    output: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    st = pd.read_csv(args.st_manifest_dir / "seed_features.csv.gz")
    require_columns(st, ("sample_id", "seed_id", "ik14"), "ST seed features")
    for row in st.itertuples(index=False):
        output[("ST001154_same_formula_10ppm", f"sample={row.sample_id}", str(row.ik14))].append(
            str(row.seed_id)
        )
    kgmn = pd.read_csv(args.kgmn_seeds)
    require_columns(kgmn, ("feature_name", "ik14"), "KGMN seed features")
    split = pd.read_csv(args.kgmn_manifest_dir / "hidden_seed_splits.csv.gz")
    require_columns(split, ("repeat", "ik14", "role"), "KGMN hidden splits")
    names = {
        str(identity): sorted(set(group["feature_name"].astype(str)))
        for identity, group in kgmn.groupby("ik14", sort=False)
    }
    for row in split.loc[split["role"].eq("seed")].itertuples(index=False):
        output[("KGMN200STD_hidden_seed", f"repeat={int(row.repeat)}", str(row.ik14))].extend(
            names.get(str(row.ik14), [])
        )
    return {key: ";".join(sorted(set(values))) for key, values in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b38-m0-dir", type=Path, required=True)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument(
        "--internal-manifest-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_manifest_v1",
    )
    parser.add_argument(
        "--st-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_manifest_v1",
    )
    parser.add_argument(
        "--kgmn-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2",
    )
    parser.add_argument(
        "--network-edges", type=Path,
        default=ROOT / "data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz",
    )
    parser.add_argument(
        "--rhea-pairs", type=Path,
        default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz",
    )
    parser.add_argument(
        "--rhea-participants", type=Path,
        default=ROOT / "data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz",
    )
    parser.add_argument(
        "--rhea-reactions", type=Path,
        default=ROOT / "data/reference/bioaware_rhea_reactome_direction_20260830/rhea_reactions.csv.gz",
    )
    parser.add_argument(
        "--b3-rotations", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/reaction_coabundance_rotations.csv.gz",
    )
    parser.add_argument(
        "--b9-rotations", type=Path,
        default=ROOT / "data/validation/bioaware_b9_local_fullcheck_20260907_v1/reaction_spectral_rotations.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    b38_paths = {
        "report": args.b38_m0_dir / "report.json",
        "events": args.b38_m0_dir / "explicit_direct_path_events.csv.gz",
        "contexts": args.b38_m0_dir / "candidate_path_coverage.csv.gz",
    }
    required_paths = [
        *b38_paths.values(),
        args.internal_candidates,
        args.st_candidates,
        args.st_queries,
        args.kgmn_candidates,
        args.kgmn_seeds,
        args.network_edges,
        args.rhea_pairs,
        args.rhea_participants,
        args.rhea_reactions,
        args.b3_rotations,
        args.b9_rotations,
        args.st_manifest_dir / "queries.csv.gz",
        args.st_manifest_dir / "seed_features.csv.gz",
        args.kgmn_manifest_dir / "hidden_seed_splits.csv.gz",
    ]
    for path in required_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    b38_report = json.loads(b38_paths["report"].read_text(encoding="utf-8"))
    if b38_report.get("status") != "bioaware_b38_m0_explicit_path_coverage_complete":
        raise RuntimeError("B39 requires a completed B38-M0 result")
    b38_provenance = b38_report.get("provenance", {})
    for key, path in (("explicit_direct_path_events", b38_paths["events"]),
                      ("candidate_path_coverage", b38_paths["contexts"])):
        if b38_provenance.get(key) != sha256(path):
            raise RuntimeError(f"B38 provenance mismatch: {key}")

    events = pd.read_csv(b38_paths["events"])
    contexts = pd.read_csv(b38_paths["contexts"])
    required_event_columns = {
        "source", "unit_id", "polarity", "query_id", "candidate_id",
        "seed_stratum", "seed_identity", "edge_source", "edge_key",
        "reaction_id", "seed_side", "candidate_side", "direction_status",
    }
    require_columns(events, required_event_columns, "B38 event ledger")
    require_columns(
        contexts,
        ("source", "unit_id", "polarity", "query_id", "candidate_id", "seed_stratum",
         "visible_seed_identities", "direct_event_count", "rewire_support_fraction"),
        "B38 candidate contexts",
    )
    if FORBIDDEN_LEDGER_COLUMNS & set(events.columns):
        raise RuntimeError("B38 input event ledger unexpectedly contains ranking outcomes")

    candidates, universe_report = build_universe(args)
    candidates = prepare_candidates(candidates)
    resolver = SeedResolver(args, candidates)
    visible_seed_sets: dict[tuple[str, str], set[str]] = {}
    query_source: dict[str, str] = {}
    for query_id, group in candidates.groupby("query_id", sort=False):
        representative = next(group.itertuples(index=False))
        query_source[str(query_id)] = str(representative.source)
        resolved = resolver.resolve(representative)
        for stratum, seed_set in resolved:
            key = (str(query_id), str(stratum))
            if key in visible_seed_sets and visible_seed_sets[key] != set(seed_set):
                raise RuntimeError(f"inconsistent visible seed context: {key}")
            visible_seed_sets[key] = set(seed_set)
    context_keys = set(zip(contexts["query_id"].astype(str), contexts["seed_stratum"].astype(str)))
    if context_keys != set(visible_seed_sets):
        missing = len(context_keys - set(visible_seed_sets))
        extra = len(set(visible_seed_sets) - context_keys)
        raise RuntimeError(f"B38/seed resolver context mismatch: missing={missing} extra={extra}")

    rhea_lookup, rhea_report = reaction_participant_lookup(
        args.rhea_participants, args.rhea_reactions
    )
    direct_edges, direct_edge_report = load_direct_edges(
        args.network_edges, args.rhea_pairs, args.rhea_participants, args.rhea_reactions
    )
    component_by_node, degree_by_node = connected_components(direct_edges)
    b3_lookup = load_proxy_lookup(
        args.b3_rotations, COABUNDANCE_PROXY_COLUMNS, "B3 co-abundance proxy"
    )
    b9_lookup = load_proxy_lookup(
        args.b9_rotations, SPECTRAL_PROXY_COLUMNS, "B9 spectral proxy"
    )
    feature_lookup = seed_feature_lookup(args)

    enriched_rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        source = str(row.source)
        query_id = str(row.query_id)
        candidate = str(row.candidate_id)
        seed = str(row.seed_identity)
        stratum = str(row.seed_stratum)
        visible = visible_seed_sets[(query_id, stratum)]
        if seed not in visible:
            raise RuntimeError(f"B38 event seed is not visible: {(query_id, stratum, seed)}")
        semantics = source_semantics(source)
        reaction = normalise_reaction_id(row.reaction_id)
        seed_side = "" if pd.isna(row.seed_side) else str(row.seed_side)
        candidate_side = "" if pd.isna(row.candidate_side) else str(row.candidate_side)
        hyperedge = hyperedge_features(
            reaction, seed, candidate, seed_side, candidate_side, visible, rhea_lookup
        )
        fold = fold_from_stratum(stratum)
        b3 = b3_lookup.get((query_id, candidate, int(fold))) if fold is not None else None
        b9 = b9_lookup.get((query_id, candidate, int(fold))) if fold is not None else None
        feature_id = feature_lookup.get((source, stratum, seed), "")
        component = component_by_node.get(candidate, component_by_node.get(seed, ""))
        structure_family = (
            f"{row.edge_source}|{row.direction_status}|"
            f"{hyperedge['noncurrency_seed_side_count']}x"
            f"{hyperedge['noncurrency_candidate_side_count']}"
        )
        item: dict[str, Any] = {
            "source": source,
            "context_class": semantics["context_class"],
            "context_semantics": semantics["context_semantics"],
            "context_selection": semantics["context_selection"],
            "prospective_unknown": bool(semantics["prospective_unknown"]),
            "unit_id": str(row.unit_id),
            "polarity": str(row.polarity),
            "query_id": query_id,
            "candidate_id": candidate,
            "seed_stratum": stratum,
            "seed_identity": seed,
            "seed_feature_id": feature_id,
            "seed_feature_resolution": "feature_resolved" if feature_id else "identity_only",
            "seed_score_available": False,
            "edge_source": str(row.edge_source),
            "edge_label": str(row.edge_label),
            "edge_key": str(row.edge_key),
            "reaction_id": reaction,
            "direction_semantics": str(row.direction_semantics),
            "direction_status": str(row.direction_status),
            "seed_side": seed_side,
            "candidate_side": candidate_side,
            "reaction_component_id": component,
            "reaction_structure_family": structure_family,
            "candidate_catalog_degree": int(degree_by_node.get(candidate, 0)),
            "seed_catalog_degree": int(degree_by_node.get(seed, 0)),
            **hyperedge,
            "legacy_spectral_proxy_available": bool(
                b9 is not None and b9.get("edge_available", 0.0) > 0
            ),
            "legacy_coabundance_proxy_available": bool(
                b3 is not None and b3.get("available", 0.0) > 0
            ),
            "experimental_evidence_resolution": (
                "candidate_context_aggregate_not_event_specific"
                if b3 is not None or b9 is not None
                else "absent"
            ),
            "event_specific_spectral_evidence_available": False,
            "event_specific_coabundance_evidence_available": False,
        }
        for name in SPECTRAL_PROXY_COLUMNS:
            item[f"legacy_spectral_{name}"] = b9.get(name, math.nan) if b9 else math.nan
        for name in COABUNDANCE_PROXY_COLUMNS:
            item[f"legacy_coabundance_{name}"] = b3.get(name, math.nan) if b3 else math.nan
        item["dependency_group_id"] = stable_dependency_id(
            (source, query_id, stratum, seed, row.edge_key, hyperedge["missing_context_signature"])
        )
        enriched_rows.append(item)

    atomic_events = add_candidate_specificity(pd.DataFrame(enriched_rows))
    if atomic_events.duplicated(
        ["query_id", "candidate_id", "seed_stratum", "seed_identity", "edge_source", "edge_key", "reaction_id"]
    ).any():
        raise RuntimeError("B39 atomic ledger contains duplicate events")
    if FORBIDDEN_LEDGER_COLUMNS & set(atomic_events.columns):
        raise RuntimeError("ranking outcome leaked into B39 atomic event ledger")

    contexts = contexts.copy()
    semantics_frame = pd.DataFrame(
        [source_semantics(source) for source in contexts["source"].astype(str)]
    )
    for column in semantics_frame:
        contexts[column] = semantics_frame[column].to_numpy()
    candidate_counts = (
        contexts.groupby(["query_id", "seed_stratum"], sort=False)["candidate_id"]
        .nunique().rename("query_context_candidate_count").reset_index()
    )
    contexts = contexts.merge(
        candidate_counts, on=["query_id", "seed_stratum"], how="left", validate="many_to_one"
    )
    contexts["candidate_catalog_degree"] = contexts["candidate_id"].astype(str).map(
        lambda value: int(degree_by_node.get(value, 0))
    )
    contexts["reaction_component_id"] = contexts["candidate_id"].astype(str).map(
        lambda value: component_by_node.get(value, "")
    )
    contexts["legacy_experimental_evidence_is_event_specific"] = False
    safe_context_columns = [
        "source", "context_class", "context_semantics", "context_selection",
        "prospective_unknown", "unit_id", "polarity", "query_id", "candidate_id",
        "seed_stratum", "visible_seed_identities", "query_context_candidate_count",
        "direct_event_count", "direct_seed_count", "direct_rhea_event_count",
        "direct_kegg_event_count", "direction_supported_event_count", "has_direct_path",
        "spectral_controlled", "coabundance_controlled", "any_positive_excess_screen",
        "path_missingness", "rewire_repeats_with_seed_support", "rewire_support_fraction",
        "candidate_catalog_degree", "reaction_component_id",
        "legacy_experimental_evidence_is_event_specific",
    ]
    candidate_contexts = contexts[safe_context_columns].copy()
    if FORBIDDEN_LEDGER_COLUMNS & set(candidate_contexts.columns):
        raise RuntimeError("ranking outcome leaked into B39 candidate-context ledger")

    seed_contexts = contexts[[
        "source", "context_class", "context_semantics", "context_selection",
        "prospective_unknown", "unit_id", "polarity", "query_id", "seed_stratum",
        "visible_seed_identities",
    ]].drop_duplicates().copy()
    if seed_contexts.duplicated(["query_id", "seed_stratum"]).any():
        raise RuntimeError("B39 seed contexts are not unique")
    seed_contexts["visible_seed_identity_list"] = [
        ";".join(sorted(visible_seed_sets[(str(row.query_id), str(row.seed_stratum))]))
        for row in seed_contexts.itertuples(index=False)
    ]
    seed_contexts["visible_seed_count"] = [
        len(visible_seed_sets[(str(row.query_id), str(row.seed_stratum))])
        for row in seed_contexts.itertuples(index=False)
    ]
    frozen_counts = pd.to_numeric(seed_contexts["visible_seed_identities"], errors="raise").astype(int)
    if not frozen_counts.equals(seed_contexts["visible_seed_count"].astype(int)):
        raise RuntimeError("B39 full visible seed ledger does not reproduce B38 seed counts")
    seed_contexts = seed_contexts.drop(columns=["visible_seed_identities"])
    if FORBIDDEN_LEDGER_COLUMNS & set(seed_contexts.columns):
        raise RuntimeError("ranking outcome leaked into B39 visible-seed ledger")

    by_source: dict[str, Any] = {}
    for source in EXPECTED_DOMAINS:
        local = atomic_events.loc[atomic_events["source"].eq(source)]
        local_context = candidate_contexts.loc[candidate_contexts["source"].eq(source)]
        by_source[source] = {
            **source_semantics(source),
            "queries": int(local_context["query_id"].nunique()),
            "candidate_contexts": int(len(local_context)),
            "events": int(len(local)),
            "rhea_events": int(local["edge_source"].eq("Rhea").sum()),
            "candidate_specific_events": int(local["event_candidate_specific"].sum()),
            "direction_supported_events": int(
                local["direction_status"].isin({"supported", "supported_bidirectional"}).sum()
            ),
            "rhea_hyperedge_resolved_events": int(
                local["hyperedge_resolution"].eq(
                    "rhea_noncurrency_participants_resolved"
                ).sum()
            ),
            "event_specific_spectral_evidence_events": 0,
            "event_specific_coabundance_evidence_events": 0,
        }

    direction_counts = Counter(atomic_events["direction_status"].astype(str))
    source_class_counts = Counter(atomic_events["context_class"].astype(str))
    hyperedge_resolved = atomic_events["hyperedge_resolution"].eq(
        "rhea_noncurrency_participants_resolved"
    )
    structural_gates = {
        "b38_provenance_replayed": True,
        "atomic_events_nonempty": bool(len(atomic_events)),
        "candidate_contexts_replay_b38": bool(len(candidate_contexts) == len(contexts)),
        "ranking_outcome_columns_absent": not bool(
            FORBIDDEN_LEDGER_COLUMNS
            & (set(atomic_events.columns) | set(candidate_contexts.columns))
        ),
        "all_event_seeds_visible": True,
        "all_six_sources_present": set(by_source) == set(EXPECTED_DOMAINS),
        "candidate_specificity_defined_for_every_event": bool(
            atomic_events["event_supported_candidate_count"].notna().all()
        ),
    }
    pass_to_m1 = bool(all(structural_gates.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / "atomic_reaction_events.csv.gz"
    context_path = args.output_dir / "candidate_context_semantics.csv.gz"
    seed_context_path = args.output_dir / "visible_seed_contexts.csv.gz"
    atomic_csv_gzip(event_path, atomic_events)
    atomic_csv_gzip(context_path, candidate_contexts)
    atomic_csv_gzip(seed_context_path, seed_contexts)

    report = {
        "status": "bioaware_b39_m0_atomic_event_ledger_complete",
        "formal": True,
        "model_fitted": False,
        "embedding_values_read": False,
        "ranking_outcomes_read": False,
        "atomic_events": int(len(atomic_events)),
        "candidate_contexts": int(len(candidate_contexts)),
        "visible_seed_contexts": int(len(seed_contexts)),
        "queries": int(candidate_contexts["query_id"].nunique()),
        "candidates": int(candidate_contexts[["query_id", "candidate_id"]].drop_duplicates().shape[0]),
        "sources": by_source,
        "context_class_event_counts": {str(key): int(value) for key, value in source_class_counts.items()},
        "direction_status_event_counts": {str(key): int(value) for key, value in direction_counts.items()},
        "candidate_specificity": {
            "candidate_specific_events": int(atomic_events["event_candidate_specific"].sum()),
            "fraction": float(atomic_events["event_candidate_specific"].mean()),
            "maximum_competing_candidates": int(
                atomic_events["event_supported_candidate_count"].max()
            ),
        },
        "hyperedge": {
            **rhea_report,
            "resolved_event_rows": int(hyperedge_resolved.sum()),
            "complete_event_rows": int(
                (hyperedge_resolved & atomic_events["hyperedge_context_complete"]).sum()
            ),
            "complete_fraction_among_resolved": float(
                atomic_events.loc[hyperedge_resolved, "hyperedge_context_complete"].mean()
            ) if hyperedge_resolved.any() else 0.0,
        },
        "experimental_evidence_resolution": {
            "legacy_spectral_proxy": "candidate-context aggregate; not event-specific",
            "legacy_coabundance_proxy": "candidate-context aggregate; not event-specific",
            "event_specific_spectral_rows": 0,
            "event_specific_coabundance_rows": 0,
            "interpretation": (
                "B3/B9 may be used only as coverage proxies in M1. They cannot qualify "
                "a Rhea event for a corrective action until rebuilt per event."
            ),
        },
        "st_leave_one_seed_out_audit": st_leave_one_seed_out_audit(
            args.st_manifest_dir / "queries.csv.gz",
            args.st_manifest_dir / "seed_features.csv.gz",
        ),
        "prospective_unknown_sources": 0,
        "direct_edge_catalog": direct_edge_report,
        "gates": structural_gates,
        "pass_to_b39_m1": pass_to_m1,
        "next_stage": (
            "B39-M1 source-stratified constructibility and event-specific evidence audit"
            if pass_to_m1
            else "repair B39-M0 structural ledger"
        ),
        "contracts": {
            "truth_used_only_by_preexisting_evaluation_context_construction": True,
            "truth_or_outcome_columns_written": False,
            "candidate_contexts_mixed_across_L_H_S": False,
            "legacy_candidate_level_proxy_promoted_to_event_evidence": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe_report["provenance"],
            "b38_report": sha256(b38_paths["report"]),
            "b38_events": sha256(b38_paths["events"]),
            "b38_contexts": sha256(b38_paths["contexts"]),
            "network_edges": sha256(args.network_edges),
            "rhea_pairs": sha256(args.rhea_pairs),
            "rhea_participants": sha256(args.rhea_participants),
            "rhea_reactions": sha256(args.rhea_reactions),
            "b3_rotations": sha256(args.b3_rotations),
            "b9_rotations": sha256(args.b9_rotations),
            "atomic_events": sha256(event_path),
            "candidate_contexts": sha256(context_path),
            "visible_seed_contexts": sha256(seed_context_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B39-M0 is an outcome-blind semantic ledger. It establishes neither "
            "reaction-specific ranking gain, prospective deployment performance, "
            "biological causality, shared-embedding improvement, nor SOTA."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

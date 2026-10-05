#!/usr/bin/env python
"""Build a truth-blind explicit direct-reaction ledger for BioAware B38.

B37 showed that the opened six-domain gain was primarily a catalogue topology
prior.  This preflight therefore does not fit a model or read an embedding
outcome.  It reconstructs the direct seed--reaction--candidate events that are
available at inference time and measures whether existing spectral,
co-abundance, and degree-preserving-null assets can support a reaction-specific
B38 experiment.

The event ledger deliberately excludes truth labels and transition outcomes.
Truth identity is used only to choose the pre-existing held-out seed rotation
and to assert the frozen candidate-graph integrity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

EXPECTED_QUERY_COUNTS = {
    "BV2cell": 95,
    "Mouse_brain": 131,
    "Mouse_liver": 176,
    "NIST_plasma": 146,
    "ST001154_same_formula_10ppm": 150,
    "KGMN200STD_hidden_seed": 162,
}
EXPECTED_DOMAINS = tuple(EXPECTED_QUERY_COUNTS)
INTERNAL_DOMAINS = frozenset(("BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"))
FORBIDDEN_EVENT_COLUMNS = frozenset(
    (
        "truth_candidate_id",
        "truth_formula",
        "is_positive",
        "baseline_correct",
        "corrected",
        "introduced",
        "final_correct",
        "delta",
    )
)
SPECTRAL_EXCESS_COLUMNS = (
    "truncated_direct_excess_mean",
    "neutral_loss_excess_mean",
    "modified_cosine_excess_mean",
    "dual_view_excess_mean",
    "kgmn_support_excess_mean",
)
COABUNDANCE_EXCESS_COLUMNS = (
    "abs_excess",
    "positive_excess",
    "negative_excess",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} misses columns: {sorted(missing)}")


def normalise_reaction_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def ik14(value: Any) -> str:
    text = str(value).strip()
    return text.split("-", 1)[0] if text and text.lower() != "nan" else ""


def direction_status(
    semantics: str, seed_sides: set[str], candidate_sides: set[str]
) -> str:
    if len(seed_sides) != 1 or len(candidate_sides) != 1:
        return "ambiguous_participant_side"
    seed_side = next(iter(seed_sides))
    candidate_side = next(iter(candidate_sides))
    if seed_side == candidate_side:
        return "same_side_not_transformative"
    if semantics == "reactome_consensus_bidirectional":
        return "supported_bidirectional"
    if semantics == "reactome_consensus_lr":
        return "supported" if (seed_side, candidate_side) == ("left", "right") else "conflicted"
    if semantics == "reactome_consensus_rl":
        return "supported" if (seed_side, candidate_side) == ("right", "left") else "conflicted"
    return "direction_unknown"


def missingness_class(
    direct_events: int,
    spectral_controlled: bool,
    coabundance_controlled: bool,
    any_positive_excess: bool,
) -> str:
    if direct_events == 0:
        return "no_path"
    if not spectral_controlled and not coabundance_controlled:
        return "path_unobservable"
    if any_positive_excess:
        return "complete_specific_screen"
    return "complete_nonspecific_screen"


def parse_prefixed_query(query_id: str, domain: str) -> str:
    prefix = f"{domain}::"
    if not str(query_id).startswith(prefix):
        raise RuntimeError(f"query {query_id!r} does not start with {prefix!r}")
    return str(query_id)[len(prefix) :]


def parse_kgmn_query(query_id: str) -> tuple[str, int]:
    base = parse_prefixed_query(query_id, "KGMN200STD_hidden_seed")
    match = re.fullmatch(r"(.+)::repeat=(\d+)", base)
    if not match:
        raise RuntimeError(f"invalid KGMN repeated query id: {query_id}")
    return match.group(1), int(match.group(2))


def add_oriented_edge(
    mapping: dict[str, dict[str, list[dict[str, Any]]]],
    left: str,
    right: str,
    record: dict[str, Any],
) -> None:
    if not left or not right or left == right:
        return
    forward = dict(record)
    forward["candidate_identity"] = left
    forward["seed_identity"] = right
    reverse = dict(record)
    reverse["candidate_identity"] = right
    reverse["seed_identity"] = left
    mapping[left][right].append(forward)
    mapping[right][left].append(reverse)


def load_direct_edges(
    network_edges: Path,
    rhea_pairs: Path,
    rhea_participants: Path,
    rhea_reactions: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    edges = pd.read_csv(network_edges)
    require_columns(
        edges,
        ("source_id", "target_id", "minimum_step", "edge_source", "edge_label", "ik14_a", "ik14_b"),
        "EMRN edges",
    )
    known = edges.loc[pd.to_numeric(edges["minimum_step"], errors="coerce").eq(0)].copy()

    participants = pd.read_csv(rhea_participants)
    reactions = pd.read_csv(rhea_reactions)
    require_columns(participants, ("reaction_id", "full_inchikey", "side"), "Rhea participants")
    require_columns(reactions, ("reaction_id", "direction_semantics"), "Rhea reactions")
    participants["reaction_key"] = participants["reaction_id"].map(normalise_reaction_id)
    participants["ik14"] = participants["full_inchikey"].map(ik14)
    side_lookup: dict[tuple[str, str], set[str]] = {
        (str(reaction), str(identity)): set(group["side"].astype(str))
        for (reaction, identity), group in participants.groupby(["reaction_key", "ik14"], sort=False)
        if reaction and identity
    }
    semantics = {
        normalise_reaction_id(row.reaction_id): str(row.direction_semantics)
        for row in reactions.itertuples(index=False)
    }

    mapping: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in known.itertuples(index=False):
        left, right = str(row.ik14_a), str(row.ik14_b)
        edge_key = "KEGG:" + "|".join(sorted((str(row.source_id), str(row.target_id))))
        add_oriented_edge(
            mapping,
            left,
            right,
            {
                "edge_source": str(row.edge_source),
                "edge_label": str(row.edge_label),
                "reaction_id": normalise_reaction_id(getattr(row, "reaction_id", "")),
                "edge_key": edge_key,
                "direction_semantics": "direction_unknown",
            },
        )

    pairs = pd.read_csv(rhea_pairs)
    require_columns(
        pairs,
        ("identity_a", "identity_b", "relation_type", "reaction_ids"),
        "Rhea identity pairs",
    )
    rhea = pairs.loc[pairs["relation_type"].astype(str).eq("reaction_direction_unknown")]
    rhea_records = 0
    for row in rhea.itertuples(index=False):
        ids = [normalise_reaction_id(value) for value in str(row.reaction_ids).split(";")]
        ids = sorted({value for value in ids if value})
        for reaction in ids:
            add_oriented_edge(
                mapping,
                str(row.identity_a),
                str(row.identity_b),
                {
                    "edge_source": "Rhea",
                    "edge_label": "known_reaction_pair",
                    "reaction_id": reaction,
                    "edge_key": f"Rhea:{reaction}",
                    "direction_semantics": semantics.get(reaction, "reaction_direction_unknown"),
                },
            )
            rhea_records += 1

    # Add side and direction information after orientation has been expanded.
    deduplicated: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for candidate, neighbours in mapping.items():
        for seed, records in neighbours.items():
            unique: dict[tuple[str, str, str], dict[str, Any]] = {}
            for record in records:
                key = (str(record["edge_source"]), str(record["edge_key"]), str(record["reaction_id"]))
                if key in unique:
                    continue
                reaction = str(record["reaction_id"])
                candidate_sides = side_lookup.get((reaction, candidate), set()) if reaction else set()
                seed_sides = side_lookup.get((reaction, seed), set()) if reaction else set()
                item = dict(record)
                item["candidate_side"] = ";".join(sorted(candidate_sides))
                item["seed_side"] = ";".join(sorted(seed_sides))
                item["direction_status"] = (
                    direction_status(str(item["direction_semantics"]), seed_sides, candidate_sides)
                    if reaction and str(item["edge_source"]) == "Rhea"
                    else "direction_unknown"
                )
                unique[key] = item
            deduplicated[candidate][seed] = list(unique.values())
    return deduplicated, {
        "kegg_step0_rows": int(len(known)),
        "rhea_pair_rows": int(len(rhea)),
        "rhea_reaction_records": int(rhea_records),
        "candidate_nodes": int(len(deduplicated)),
        "undirected_identity_pairs": int(
            len({tuple(sorted((left, right))) for left, values in deduplicated.items() for right in values})
        ),
    }


class SeedResolver:
    def __init__(self, args: argparse.Namespace, candidates: pd.DataFrame):
        self.internal: dict[tuple[str, str], list[tuple[int, set[str]]]] = defaultdict(list)
        units = sorted(
            set(candidates.loc[candidates["source"].isin(INTERNAL_DOMAINS), "unit_id"].astype(str))
        )
        for unit in units:
            path = args.internal_manifest_root / unit / "identity_splits.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            splits = pd.read_csv(path)
            require_columns(splits, ("fold", "role", "ik14"), f"{unit} splits")
            for fold in sorted(pd.to_numeric(splits["fold"], errors="raise").astype(int).unique()):
                held = set(splits.loc[(splits["fold"] == fold) & splits["role"].eq("heldout"), "ik14"].astype(str))
                seeds = set(splits.loc[(splits["fold"] == fold) & splits["role"].eq("seed"), "ik14"].astype(str))
                for identity in held:
                    key = (unit, identity)
                    self.internal[key].append((int(fold), seeds))
        # The frozen internal design evaluates each held-out identity in seven
        # rotations (and uses it as a seed in the remaining three).  Preserve
        # every rotation: selecting only one would be an arbitrary, lossy
        # change of the B3/B9 protocol.
        for key, rotations in self.internal.items():
            folds = [fold for fold, _seeds in rotations]
            if len(folds) != len(set(folds)):
                raise RuntimeError(f"{key}: duplicate internal rotation")

        st_queries = pd.read_csv(args.st_manifest_dir / "queries.csv.gz")
        st_seeds = pd.read_csv(args.st_manifest_dir / "seed_features.csv.gz")
        require_columns(st_queries, ("query_id", "sample_id", "truth_candidate_id"), "ST queries")
        require_columns(st_seeds, ("sample_id", "ik14"), "ST seeds")
        self.st_query_sample = dict(zip(st_queries["query_id"].astype(str), st_queries["sample_id"].astype(str), strict=True))
        self.st_seeds = {
            str(sample): set(group["ik14"].astype(str))
            for sample, group in st_seeds.groupby("sample_id", sort=False)
        }

        kgmn_splits = pd.read_csv(args.kgmn_manifest_dir / "hidden_seed_splits.csv.gz")
        require_columns(kgmn_splits, ("repeat", "ik14", "role"), "KGMN splits")
        self.kgmn_seeds = {
            int(repeat): set(group.loc[group["role"].eq("seed"), "ik14"].astype(str))
            for repeat, group in kgmn_splits.groupby("repeat", sort=False)
        }

    def resolve(self, row: Any) -> list[tuple[str, set[str]]]:
        source = str(row.source)
        truth = str(row.truth_candidate_id)
        if source in INTERNAL_DOMAINS:
            key = (str(row.unit_id), truth)
            if key not in self.internal:
                raise RuntimeError(f"missing internal held-out rotation for {key}")
            return [
                (f"fold={fold}", set(seeds) - {truth})
                for fold, seeds in sorted(self.internal[key])
            ]
        if source == "ST001154_same_formula_10ppm":
            base = parse_prefixed_query(str(row.query_id), source)
            if base not in self.st_query_sample:
                raise RuntimeError(f"ST query absent from manifest: {base}")
            sample = self.st_query_sample[base]
            return [(f"sample={sample}", set(self.st_seeds.get(sample, set())) - {truth})]
        if source == "KGMN200STD_hidden_seed":
            _base, repeat = parse_kgmn_query(str(row.query_id))
            if repeat not in self.kgmn_seeds:
                raise RuntimeError(f"KGMN repeat absent from split: {repeat}")
            return [(f"repeat={repeat}", set(self.kgmn_seeds[repeat]) - {truth})]
        raise RuntimeError(f"unsupported source: {source}")


def load_rotation_lookup(path: Path, label: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    frame = pd.read_csv(path)
    require_columns(frame, ("query_id", "candidate_id", "fold"), label)
    keys = frame[["query_id", "candidate_id", "fold"]].astype({"query_id": str, "candidate_id": str, "fold": int})
    if keys.duplicated().any():
        raise RuntimeError(f"{label}: duplicate query/candidate/fold keys")
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = (str(row["query_id"]), str(row["candidate_id"]), int(row["fold"]))
        output[key] = row
    return output


def fold_from_stratum(stratum: str) -> int | None:
    match = re.fullmatch(r"fold=(\d+)", stratum)
    return int(match.group(1)) if match else None


def adjacency_sets(path: Path) -> dict[str, set[str]]:
    frame = pd.read_csv(path, usecols=["minimum_step", "ik14_a", "ik14_b"])
    frame = frame.loc[pd.to_numeric(frame["minimum_step"], errors="coerce").eq(0)]
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in frame[["ik14_a", "ik14_b"]].itertuples(index=False):
        left, right = str(left), str(right)
        if left and right and left != right:
            graph[left].add(right)
            graph[right].add(left)
    return graph


def source_summary(
    candidate_context: pd.DataFrame,
    query_context: pd.DataFrame,
    query: pd.DataFrame,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in EXPECTED_DOMAINS:
        local_candidates = candidate_context.loc[candidate_context["source"].eq(source)]
        local_contexts = query_context.loc[query_context["source"].eq(source)]
        local_queries = query.loc[query["source"].eq(source)]
        output[source] = {
            "queries": int(len(local_queries)),
            "query_contexts": int(len(local_contexts)),
            "candidate_context_rows": int(len(local_candidates)),
            "queries_with_any_direct_path": int(local_queries["any_direct_path"].sum()),
            "query_contexts_with_candidate_varying_path": int(local_contexts["path_varies_within_context"].sum()),
            "candidate_varying_path_fraction": float(local_contexts["path_varies_within_context"].mean()),
            "path_candidate_rows": int(local_candidates["has_direct_path"].sum()),
            "spectral_controlled_path_rows": int(
                (local_candidates["has_direct_path"] & local_candidates["spectral_controlled"]).sum()
            ),
            "coabundance_controlled_path_rows": int(
                (local_candidates["has_direct_path"] & local_candidates["coabundance_controlled"]).sum()
            ),
            "missingness": {
                str(key): int(value)
                for key, value in local_candidates["path_missingness"].value_counts().to_dict().items()
            },
        }
    return output


def main() -> None:
    # Keep the fast helper/unit checks independent of the older BioAware
    # evaluation stack (which imports scipy).  The heavy dependencies are
    # required only when constructing the formal 860-query ledger.
    from audit_bioaware_b12_multicohort_catalog_action import build_universe
    from audit_bioaware_b36_reaction_specificity_ablation import prepare_candidates

    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--decoy-root", type=Path,
        default=ROOT / "data/reference/bioaware_degree_preserving_decoys_v1",
    )
    parser.add_argument("--decoy-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    required_paths = (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.network_edges,
        args.rhea_pairs, args.rhea_participants, args.rhea_reactions,
        args.b3_rotations, args.b9_rotations,
        args.st_manifest_dir / "queries.csv.gz",
        args.st_manifest_dir / "seed_features.csv.gz",
        args.kgmn_manifest_dir / "hidden_seed_splits.csv.gz",
    )
    for path in required_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.decoy_repeats <= 0:
        raise ValueError("decoy-repeats must be positive")
    decoy_paths = [
        args.decoy_root / f"repeat_{repeat:02d}" / "metdna2_emrn_edges.csv.gz"
        for repeat in range(args.decoy_repeats)
    ]
    for path in decoy_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, universe = build_universe(args)
    candidates = prepare_candidates(candidates)
    evaluated = candidates.loc[candidates["polarity"].eq("negative")].copy()
    query_counts = evaluated.groupby("source")["query_id"].nunique().to_dict()
    if query_counts != EXPECTED_QUERY_COUNTS:
        raise RuntimeError(f"frozen B37 query counts changed: {query_counts}")
    if evaluated["query_id"].nunique() != 860:
        raise RuntimeError("B38-M0 requires exactly 860 B37 queries")
    if evaluated.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate query/candidate rows in frozen B37 graph")
    if not evaluated.groupby("query_id")["is_positive"].sum().eq(1).all():
        raise RuntimeError("B37 graph truth multiplicity changed")

    edges, relation_report = load_direct_edges(
        args.network_edges, args.rhea_pairs, args.rhea_participants, args.rhea_reactions
    )
    resolver = SeedResolver(args, candidates)
    b3 = load_rotation_lookup(args.b3_rotations, "B3 co-abundance rotations")
    b9 = load_rotation_lookup(args.b9_rotations, "B9 spectral rotations")
    decoy_graphs = [adjacency_sets(path) for path in decoy_paths]

    event_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        query_id = str(row.query_id)
        candidate = str(row.candidate_id)
        for stratum, seeds in resolver.resolve(row):
            fold = fold_from_stratum(stratum)
            neighbours = sorted(set(edges.get(candidate, {})) & seeds)
            records: list[dict[str, Any]] = []
            for seed_identity in neighbours:
                for edge in edges[candidate][seed_identity]:
                    record = {
                        "source": str(row.source),
                        "unit_id": str(row.unit_id),
                        "polarity": str(row.polarity),
                        "query_id": query_id,
                        "candidate_id": candidate,
                        "seed_stratum": stratum,
                        "seed_identity": seed_identity,
                        "edge_source": str(edge["edge_source"]),
                        "edge_label": str(edge["edge_label"]),
                        "edge_key": str(edge["edge_key"]),
                        "reaction_id": str(edge["reaction_id"]),
                        "direction_semantics": str(edge["direction_semantics"]),
                        "seed_side": str(edge["seed_side"]),
                        "candidate_side": str(edge["candidate_side"]),
                        "direction_status": str(edge["direction_status"]),
                    }
                    if FORBIDDEN_EVENT_COLUMNS & set(record):
                        raise RuntimeError("outcome column leaked into explicit event ledger")
                    event_rows.append(record)
                    records.append(record)

            b3_record = b3.get((query_id, candidate, int(fold))) if fold is not None else None
            b9_record = b9.get((query_id, candidate, int(fold))) if fold is not None else None
            spectral_controlled = bool(b9_record is not None and float(b9_record.get("edge_available", 0.0)) > 0)
            coabundance_controlled = bool(b3_record is not None and float(b3_record.get("available", 0.0)) > 0)
            excess_values: list[float] = []
            if b9_record is not None:
                excess_values.extend(float(b9_record.get(column, 0.0)) for column in SPECTRAL_EXCESS_COLUMNS)
            if b3_record is not None:
                excess_values.extend(float(b3_record.get(column, 0.0)) for column in COABUNDANCE_EXCESS_COLUMNS)
            any_positive_excess = bool(any(np.isfinite(value) and value > 0 for value in excess_values))
            rewire_support = [
                int(bool(decoy.get(candidate, set()) & seeds)) for decoy in decoy_graphs
            ]
            candidate_rows.append({
                "source": str(row.source),
                "unit_id": str(row.unit_id),
                "polarity": str(row.polarity),
                "query_id": query_id,
                "candidate_id": candidate,
                "seed_stratum": stratum,
                "visible_seed_identities": int(len(seeds)),
                "direct_event_count": int(len(records)),
                "direct_seed_count": int(len({record["seed_identity"] for record in records})),
                "direct_rhea_event_count": int(sum(record["edge_source"] == "Rhea" for record in records)),
                "direct_kegg_event_count": int(sum(record["edge_source"] != "Rhea" for record in records)),
                "direction_supported_event_count": int(sum(record["direction_status"].startswith("supported") for record in records)),
                "has_direct_path": bool(records),
                "spectral_controlled": spectral_controlled,
                "coabundance_controlled": coabundance_controlled,
                "any_positive_excess_screen": any_positive_excess,
                "path_missingness": missingness_class(
                    len(records), spectral_controlled, coabundance_controlled, any_positive_excess
                ),
                "rewire_repeats_with_seed_support": int(sum(rewire_support)),
                "rewire_support_fraction": float(np.mean(rewire_support)),
                **{
                    f"rewire_support_{repeat:02d}": int(value)
                    for repeat, value in enumerate(rewire_support)
                },
            })

    events = pd.DataFrame(event_rows)
    candidate_coverage = pd.DataFrame(candidate_rows)
    if candidate_coverage[["query_id", "candidate_id"]].drop_duplicates().shape[0] != len(candidates):
        raise RuntimeError("candidate contexts do not replay the complete B37 training graph")
    if candidate_coverage.duplicated(["query_id", "candidate_id", "seed_stratum"]).any():
        raise RuntimeError("candidate coverage contains duplicate context rows")
    if FORBIDDEN_EVENT_COLUMNS & set(events.columns):
        raise RuntimeError("event ledger contains forbidden outcome columns")
    evaluation_events = events.loc[events["polarity"].eq("negative")].copy()

    evaluated_coverage = candidate_coverage.loc[
        candidate_coverage["polarity"].eq("negative")
    ].copy()
    context_rows: list[dict[str, Any]] = []
    for (query_id, seed_stratum), group in evaluated_coverage.groupby(
        ["query_id", "seed_stratum"], sort=False
    ):
        path_count = int(group["has_direct_path"].sum())
        context_rows.append({
            "source": str(group["source"].iloc[0]),
            "query_id": str(query_id),
            "seed_stratum": str(seed_stratum),
            "candidate_count": int(len(group)),
            "candidates_with_direct_path": path_count,
            "any_direct_path": bool(path_count),
            "path_varies_within_context": bool(0 < path_count < len(group)),
            "candidates_with_spectral_controls": int(group["spectral_controlled"].sum()),
            "candidates_with_coabundance_controls": int(group["coabundance_controlled"].sum()),
            "candidates_screen_specific": int(group["path_missingness"].eq("complete_specific_screen").sum()),
        })
    query_context_coverage = pd.DataFrame(context_rows)

    query_rows: list[dict[str, Any]] = []
    for query_id, group in query_context_coverage.groupby("query_id", sort=False):
        query_rows.append({
            "source": str(group["source"].iloc[0]),
            "query_id": str(query_id),
            "seed_contexts": int(len(group)),
            "any_direct_path": bool(group["any_direct_path"].any()),
            "path_varies_in_any_context": bool(group["path_varies_within_context"].any()),
            "path_varies_in_every_context": bool(group["path_varies_within_context"].all()),
        })
    query_coverage = pd.DataFrame(query_rows)
    if len(query_coverage) != 860:
        raise RuntimeError("query coverage changed frozen denominator")

    path_candidates = evaluated_coverage.loc[evaluated_coverage["has_direct_path"]]
    source_reports = source_summary(evaluated_coverage, query_context_coverage, query_coverage)
    variation_overall = float(query_context_coverage["path_varies_within_context"].mean())
    spectral_control_fraction = float(path_candidates["spectral_controlled"].mean()) if len(path_candidates) else 0.0
    coabundance_control_fraction = float(path_candidates["coabundance_controlled"].mean()) if len(path_candidates) else 0.0
    any_control_fraction = float(
        (path_candidates["spectral_controlled"] | path_candidates["coabundance_controlled"]).mean()
    ) if len(path_candidates) else 0.0
    primary_gates = {
        "exact_860_query_candidate_graph": True,
        "event_ledger_outcome_columns_absent": not bool(FORBIDDEN_EVENT_COLUMNS & set(events.columns)),
        "queries_with_any_direct_path_ge_400": bool(query_coverage["any_direct_path"].sum() >= 400),
        "candidate_path_variation_ge_30pct_every_source": bool(
            all(item["candidate_varying_path_fraction"] >= 0.30 for item in source_reports.values())
        ),
        "all_degree_preserving_rewires_loaded": len(decoy_graphs) == args.decoy_repeats,
        "direct_path_events_nonzero": bool(len(events)),
    }
    joint_data_layer_gates = {
        "spectral_matched_control_coverage_ge_90pct": bool(spectral_control_fraction >= 0.90),
        "coabundance_matched_control_coverage_ge_90pct": bool(coabundance_control_fraction >= 0.90),
    }
    # Protocol amendment after the outcome-blind coverage audit: the explicit
    # path-vs-topology primary test and the spectral/co-abundance specificity
    # test are distinct estimands.  Missing old B3/B9 caches in external
    # cohorts may block the latter but must not erase an identifiable direct
    # path contrast backed by degree-preserving rewires.
    pass_to_m1_primary = bool(all(primary_gates.values()))
    pass_to_m1_joint = bool(
        pass_to_m1_primary and all(joint_data_layer_gates.values())
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / "explicit_direct_path_events.csv.gz"
    candidate_path = args.output_dir / "candidate_path_coverage.csv.gz"
    query_path = args.output_dir / "query_path_coverage.csv.gz"
    query_context_path = args.output_dir / "query_context_path_coverage.csv.gz"
    events.to_csv(event_path, index=False, compression="gzip")
    candidate_coverage.to_csv(candidate_path, index=False, compression="gzip")
    query_coverage.to_csv(query_path, index=False, compression="gzip")
    query_context_coverage.to_csv(query_context_path, index=False, compression="gzip")

    report = {
        "status": "bioaware_b38_m0_explicit_path_coverage_complete",
        "formal": True,
        "model_fitted": False,
        "embedding_values_read": False,
        "pass_to_b38_m1": pass_to_m1_primary,
        "pass_to_b38_m1_primary": pass_to_m1_primary,
        "pass_to_b38_m1_joint_data_layer": pass_to_m1_joint,
        "frozen_candidate_graph": {
            "queries": int(query_coverage.shape[0]),
            "candidate_rows": int(len(evaluated)),
            "candidate_context_rows": int(evaluated_coverage.shape[0]),
            "query_contexts": int(query_context_coverage.shape[0]),
            "query_counts": {str(key): int(value) for key, value in query_counts.items()},
        },
        "complete_training_candidate_graph": {
            "queries": int(candidates["query_id"].nunique()),
            "candidate_rows": int(len(candidates)),
            "candidate_context_rows": int(len(candidate_coverage)),
            "polarities": {
                str(key): int(value)
                for key, value in candidate_coverage.groupby("polarity").size().to_dict().items()
            },
        },
        "explicit_direct_paths": {
            "events": int(len(evaluation_events)),
            "candidate_rows_with_path": int(len(path_candidates)),
            "queries_with_any_path": int(query_coverage["any_direct_path"].sum()),
            "queries_with_candidate_varying_path": int(query_coverage["path_varies_in_any_context"].sum()),
            "query_contexts_with_candidate_varying_path": int(query_context_coverage["path_varies_within_context"].sum()),
            "candidate_varying_path_fraction_across_query_contexts": variation_overall,
            "all_training_events": int(len(events)),
            "edge_sources": {str(key): int(value) for key, value in events.loc[events["polarity"].eq("negative"), "edge_source"].value_counts().to_dict().items()} if len(events) else {},
            "direction_status": {str(key): int(value) for key, value in events.loc[events["polarity"].eq("negative"), "direction_status"].value_counts().to_dict().items()} if len(events) else {},
            "missingness": {str(key): int(value) for key, value in evaluated_coverage["path_missingness"].value_counts().to_dict().items()},
        },
        "experimental_control_coverage_among_path_candidates": {
            "spectral": spectral_control_fraction,
            "coabundance": coabundance_control_fraction,
            "either": any_control_fraction,
            "both": float(
                (path_candidates["spectral_controlled"] & path_candidates["coabundance_controlled"]).mean()
            ) if len(path_candidates) else 0.0,
        },
        "relation_catalog": relation_report,
        "degree_preserving_rewires": {
            "repeats": int(len(decoy_graphs)),
            "mean_candidate_support_fraction": float(evaluated_coverage["rewire_support_fraction"].mean()),
        },
        "by_source": source_reports,
        "gates": {
            "primary_explicit_path": primary_gates,
            "joint_data_layer_specificity": joint_data_layer_gates,
        },
        "next_decision": (
            "run_primary_explicit_path_incremental_test__joint_data_layer_claim_blocked"
            if pass_to_m1_primary and not pass_to_m1_joint
            else (
                "run_primary_and_joint_data_layer_b38_m1"
                if pass_to_m1_joint
                else "repair_explicit_path_identifiability_before_b38_m1"
            )
        ),
        "contracts": {
            "truth_or_transition_outcomes_used_for_event_construction": False,
            "truth_identity_used_only_for_preexisting_seed_rotation_and_integrity": True,
            "explicit_event_unit": "query-candidate-seed-reaction-direction",
            "missingness_not_zero_imputed": True,
            "coverage_amendment_used_retrieval_outcomes": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe["provenance"],
            "network_edges": sha256(args.network_edges),
            "rhea_pairs": sha256(args.rhea_pairs),
            "rhea_participants": sha256(args.rhea_participants),
            "rhea_reactions": sha256(args.rhea_reactions),
            "b3_rotations": sha256(args.b3_rotations),
            "b9_rotations": sha256(args.b9_rotations),
            "degree_preserving_rewires": [sha256(path) for path in decoy_paths],
            "explicit_direct_path_events": sha256(event_path),
            "candidate_path_coverage": sha256(candidate_path),
            "query_path_coverage": sha256(query_path),
            "query_context_path_coverage": sha256(query_context_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B38-M0 is a truth-blind identifiability and coverage preflight. "
            "It does not establish reaction-specific ranking gain, biological causality, "
            "shared-embedding improvement, or SOTA performance."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

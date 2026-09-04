#!/usr/bin/env python3
"""Freeze paired known-reaction versus formula-matched decoy spectrum triples.

This stage contains no DreaMS outcome.  It fixes the identities, spectra,
acquisition strata, component-disjoint folds and weights used to ask whether a
spectral edge score supports KGMN's recursive reaction edges better than a
hard, same-formula non-edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def polarity(adduct: str) -> str:
    if "+" in adduct and "-" not in adduct:
        return "positive"
    if "-" in adduct and "+" not in adduct:
        return "negative"
    if adduct.endswith("+"):
        return "positive"
    if adduct.endswith("-"):
        return "negative"
    return "unknown"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            if root_left > root_right:
                root_left, root_right = root_right, root_left
            self.parent[root_right] = root_left


def closest_condition_rows(
    rows_by_identity: dict[str, list[int]],
    source: str,
    positive: str,
    decoy: str,
    instrument: np.ndarray,
    adduct: np.ndarray,
    collision_energy: np.ndarray,
    maximum_strata: int,
) -> list[dict[str, object]]:
    def exact_strata(identity: str) -> dict[tuple[str, str], list[int]]:
        result: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in rows_by_identity[identity]:
            result[(str(instrument[row]), str(adduct[row]))].append(row)
        return result

    def coarse_strata(identity: str) -> dict[tuple[str, str], list[int]]:
        result: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in rows_by_identity[identity]:
            result[(str(instrument[row]), polarity(str(adduct[row])))].append(row)
        return result

    exact = [exact_strata(identity) for identity in (source, positive, decoy)]
    common = sorted(set(exact[0]) & set(exact[1]) & set(exact[2]))
    level = "instrument_adduct"
    strata = exact
    if not common:
        strata = [coarse_strata(identity) for identity in (source, positive, decoy)]
        common = sorted(set(strata[0]) & set(strata[1]) & set(strata[2]))
        level = "instrument_polarity"

    chosen: list[dict[str, object]] = []
    for stratum in common[:maximum_strata]:
        best: tuple[tuple[float, int, int, int], tuple[int, int, int]] | None = None
        for source_row in strata[0][stratum]:
            source_ce = collision_energy[source_row]
            for positive_row in strata[1][stratum]:
                positive_ce = collision_energy[positive_row]
                for decoy_row in strata[2][stratum]:
                    decoy_ce = collision_energy[decoy_row]
                    values = np.asarray([source_ce, positive_ce, decoy_ce], dtype=float)
                    finite = np.isfinite(values)
                    missing_penalty = int((~finite).sum())
                    if finite.all():
                        ce_distance = abs(source_ce - positive_ce) + abs(source_ce - decoy_ce)
                    elif not finite.any():
                        ce_distance = 0.0
                    else:
                        ce_distance = 1000.0
                    criterion = (
                        float(ce_distance),
                        missing_penalty,
                        int(source_row),
                        int(positive_row) + int(decoy_row),
                    )
                    candidate = (criterion, (int(source_row), int(positive_row), int(decoy_row)))
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            continue
        source_row, positive_row, decoy_row = best[1]
        chosen.append(
            {
                "source_row": source_row,
                "positive_row": positive_row,
                "decoy_row": decoy_row,
                "match_level": level,
                "instrument": str(instrument[source_row]),
                "source_adduct": str(adduct[source_row]),
                "positive_adduct": str(adduct[positive_row]),
                "decoy_adduct": str(adduct[decoy_row]),
                "polarity": polarity(str(adduct[source_row])),
                "source_ce": float(collision_energy[source_row]),
                "positive_ce": float(collision_energy[positive_row]),
                "decoy_ce": float(collision_energy[decoy_row]),
            }
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"))
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument(
        "--p3-allow",
        type=Path,
        default=Path(
            "data/validation/g8r_p3_allow_recovered_from_audit_v2_20260830/"
            "p3_p2_allowed_training_ik14.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-network-degree", type=int, default=10)
    parser.add_argument("--decoys-per-edge", type=int, default=3)
    parser.add_argument("--maximum-acquisition-strata", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    for path in (args.edges, args.hdf5, args.p3_allow):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    allow_report = json.loads(args.p3_allow.read_text(encoding="utf-8"))
    if allow_report.get("p3_query_overlap") != 0:
        raise RuntimeError("P3 allow-list audit does not report zero query overlap")
    allowed_block = allow_report["real_train_primary"]
    allowed_identities = set(map(str, allowed_block["ik14"]))
    allowed_rows = set(map(int, allowed_block["rows"]))

    edges = pd.read_csv(args.edges)
    required_edge_columns = {"ik14_a", "ik14_b", "minimum_step", "edge_label"}
    if not required_edge_columns.issubset(edges.columns):
        raise RuntimeError(f"network edge table misses {sorted(required_edge_columns - set(edges.columns))}")
    edges = edges.dropna(subset=["ik14_a", "ik14_b"]).copy()
    edges["ik14_a"] = edges["ik14_a"].astype(str)
    edges["ik14_b"] = edges["ik14_b"].astype(str)
    all_network_pairs = {
        tuple(sorted((left, right)))
        for left, right in edges[["ik14_a", "ik14_b"]].itertuples(index=False, name=None)
        if left != right
    }
    known = edges[(edges["minimum_step"] == 0) & (edges["edge_label"] == "known_pair")].copy()
    known["pair"] = known.apply(lambda row: tuple(sorted((row["ik14_a"], row["ik14_b"]))), axis=1)
    known = known.drop_duplicates("pair")

    degree: dict[str, int] = defaultdict(int)
    for left, right in known["pair"]:
        degree[left] += 1
        degree[right] += 1
    known = known[
        known["ik14_a"].isin(allowed_identities)
        & known["ik14_b"].isin(allowed_identities)
        & (known["ik14_a"].map(degree) <= args.maximum_network_degree)
        & (known["ik14_b"].map(degree) <= args.maximum_network_degree)
    ].copy()

    with h5py.File(args.hdf5, "r") as handle:
        identity = decode(handle["INCHIKEY"][:])
        formula = decode(handle["FORMULA"][:])
        fold = decode(handle["fold"][:])
        instrument = decode(handle["INSTRUMENT_TYPE"][:])
        adduct = decode(handle["adduct"][:])
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][:], dtype=float).reshape(-1)

    selected_rows = np.asarray(sorted(allowed_rows), dtype=int)
    if selected_rows.size == 0 or selected_rows.min() < 0 or selected_rows.max() >= len(identity):
        raise RuntimeError("P3 allow-list contains invalid or empty HDF5 rows")
    selected_rows = selected_rows[
        (fold[selected_rows] == "train") & np.isin(identity[selected_rows], list(allowed_identities))
    ]
    rows_by_identity: dict[str, list[int]] = defaultdict(list)
    formula_by_identity: dict[str, str] = {}
    for row in selected_rows:
        rows_by_identity[str(identity[row])].append(int(row))
        formula_by_identity[str(identity[row])] = str(formula[row])

    formula_identities: dict[str, set[str]] = defaultdict(set)
    for ik14, molecular_formula in formula_by_identity.items():
        formula_identities[molecular_formula].add(ik14)

    triples: list[dict[str, object]] = []
    positive_edges_considered = 0
    positive_edges_with_decoys = 0
    for left, right in sorted(known["pair"].tolist()):
        if left not in rows_by_identity or right not in rows_by_identity:
            continue
        positive_edges_considered += 1
        orientations: list[tuple[int, str, str, list[str]]] = []
        for source, target in ((left, right), (right, left)):
            target_formula = formula_by_identity[target]
            possible_decoys = [
                decoy
                for decoy in formula_identities[target_formula]
                if decoy not in {source, target}
                and tuple(sorted((source, decoy))) not in all_network_pairs
            ]
            possible_decoys.sort(
                key=lambda decoy: (
                    abs(math.log2(degree.get(target, 0) + 1) - math.log2(degree.get(decoy, 0) + 1)),
                    stable_key(SEED, source, target, decoy),
                )
            )
            orientations.append((len(possible_decoys), source, target, possible_decoys))
        orientations.sort(key=lambda item: (-item[0], stable_key(item[1], item[2])))
        _, source, target, possible_decoys = orientations[0]
        if not possible_decoys:
            continue
        positive_edges_with_decoys += 1
        edge_id = stable_key("edge", *sorted((source, target)))[:16]
        for decoy in possible_decoys[: args.decoys_per_edge]:
            conditions = closest_condition_rows(
                rows_by_identity,
                source,
                target,
                decoy,
                instrument,
                adduct,
                collision_energy,
                args.maximum_acquisition_strata,
            )
            for stratum_index, condition in enumerate(conditions):
                triples.append(
                    {
                        "edge_id": edge_id,
                        "triple_id": stable_key(edge_id, decoy, stratum_index)[:20],
                        "source_ik14": source,
                        "positive_ik14": target,
                        "decoy_ik14": decoy,
                        "source_formula": formula_by_identity[source],
                        "positive_formula": formula_by_identity[target],
                        "decoy_formula": formula_by_identity[decoy],
                        "source_degree": degree.get(source, 0),
                        "positive_degree": degree.get(target, 0),
                        "decoy_degree": degree.get(decoy, 0),
                        **condition,
                    }
                )
    if not triples:
        raise RuntimeError("no acquisition-matched reaction/decoy triples were constructed")

    table = pd.DataFrame(triples)
    if not (table["positive_formula"] == table["decoy_formula"]).all():
        raise RuntimeError("formula matching contract failed")
    if any(tuple(sorted(pair)) in all_network_pairs for pair in table[["source_ik14", "decoy_ik14"]].itertuples(index=False, name=None)):
        raise RuntimeError("a decoy is connected to its source in the extracted MetDNA2 network")

    union = UnionFind()
    for row in table.itertuples(index=False):
        union.union(row.source_ik14, row.positive_ik14)
        union.union(row.source_ik14, row.decoy_ik14)
    component_members: dict[str, set[str]] = defaultdict(set)
    for identity_value in set(table["source_ik14"]) | set(table["positive_ik14"]) | set(table["decoy_ik14"]):
        component_members[union.find(identity_value)].add(identity_value)
    component_ids = {
        root: stable_key("component", *sorted(members))[:16] for root, members in component_members.items()
    }
    table["component_id"] = table["source_ik14"].map(lambda value: component_ids[union.find(value)])

    component_sizes = table.groupby("component_id").size().sort_values(ascending=False)
    fold_loads = [0] * args.folds
    component_fold: dict[str, int] = {}
    for component_id, size in component_sizes.items():
        target_fold = min(range(args.folds), key=lambda index: (fold_loads[index], index))
        component_fold[component_id] = target_fold
        fold_loads[target_fold] += int(size)
    table["component_fold"] = table["component_id"].map(component_fold).astype(int)
    edge_counts = table.groupby("edge_id")["triple_id"].transform("count")
    table["edge_equal_weight"] = 1.0 / edge_counts

    identities_by_fold = {
        fold_index: set(
            table.loc[table["component_fold"] == fold_index, ["source_ik14", "positive_ik14", "decoy_ik14"]]
            .to_numpy()
            .ravel()
        )
        for fold_index in range(args.folds)
    }
    for left_fold in range(args.folds):
        for right_fold in range(left_fold + 1, args.folds):
            overlap = identities_by_fold[left_fold] & identities_by_fold[right_fold]
            if overlap:
                raise RuntimeError(f"identity leakage between component folds: {left_fold}, {right_fold}")

    output_table = args.output_dir / "paired_reaction_decoy_triples.csv.gz"
    table.sort_values(["component_fold", "component_id", "edge_id", "triple_id"]).to_csv(
        output_table, index=False
    )
    report = {
        "status": "kgmn_dreams_edge_calibration_manifest_frozen",
        "formal": True,
        "outcome_columns_present": False,
        "positive_definition": "MetDNA2 minimum_step=0, edge_label=known_pair",
        "decoy_definition": "same target formula, no extracted MetDNA2 edge to the fixed source",
        "counts": {
            "positive_edges_considered": positive_edges_considered,
            "positive_edges_with_decoys": positive_edges_with_decoys,
            "paired_triples": int(len(table)),
            "unique_edge_ids": int(table["edge_id"].nunique()),
            "identities": int(
                len(set(table["source_ik14"]) | set(table["positive_ik14"]) | set(table["decoy_ik14"]))
            ),
            "formulas": int(
                len(set(table["source_formula"]) | set(table["positive_formula"]) | set(table["decoy_formula"]))
            ),
            "components": int(table["component_id"].nunique()),
            "match_levels": {str(key): int(value) for key, value in table["match_level"].value_counts().items()},
            "fold_triples": {str(index): int((table["component_fold"] == index).sum()) for index in range(args.folds)},
        },
        "parameters": {
            "maximum_network_degree": args.maximum_network_degree,
            "decoys_per_edge": args.decoys_per_edge,
            "maximum_acquisition_strata": args.maximum_acquisition_strata,
            "folds": args.folds,
            "seed": SEED,
        },
        "contracts": {
            "p3_query_identity_overlap": 0,
            "real_train_rows_only": True,
            "positive_decoy_formula_exact": True,
            "source_shared_within_triple": True,
            "component_fold_identity_overlap": 0,
            "currency_hubs_excluded_by_degree": True,
            "edge_equal_weighting": True,
            "dreams_scores_used": False,
            "author_scores_used": False,
        },
        "provenance": {
            "edges_sha256": sha256(args.edges),
            "hdf5_sha256": sha256(args.hdf5),
            "p3_allow_sha256": sha256(args.p3_allow),
            "triples_sha256": sha256(output_table),
            "script_sha256": sha256(Path(__file__)),
        },
        "next_gate": (
            "After the untouched author 200STD baseline is frozen, encode only these rows and compare paired "
            "author/DreaMS margins with component-fold cross-fitting. This proxy cannot replace dynamic 200STD propagation evaluation."
        ),
        "claim_limit": (
            "This is an outcome-free calibration manifest. It neither proves reaction-edge discrimination nor KGMN improvement."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

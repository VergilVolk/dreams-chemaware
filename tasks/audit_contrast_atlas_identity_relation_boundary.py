#!/usr/bin/env python3
"""Audit whether a frozen open spectral graph supports identity transport.

The current open graph was frozen from embedding similarity without precursor-
mass gating.  That is appropriate for chemical-neighbour discovery but not
automatically for combining quantitative effects as the same entity.  This
audit attaches phenotype-blind precursor masses from the source MGF files and
opens strict planar-identity truth only after the graph is fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_precursors(path: Path, study_id: str) -> dict[str, float]:
    """Read one representative precursor per FEATURE_ID, SCANS, or TITLE."""
    result: dict[str, float] = {}
    feature_id = scans = title = None
    precursor = None
    inside = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                inside = True
                feature_id = scans = title = None
                precursor = None
            elif not inside:
                continue
            elif line.startswith("FEATURE_ID="):
                feature_id = line.split("=", 1)[1].strip()
            elif line.startswith("SCANS="):
                scans = line.split("=", 1)[1].strip()
            elif line.startswith("TITLE="):
                title = line.split("=", 1)[1].strip()
            elif line.startswith("PEPMASS="):
                precursor = float(line.split("=", 1)[1].split()[0])
            elif line == "END IONS":
                key = feature_id or scans or title
                if key and precursor is not None:
                    entity_id = f"{study_id}::{key}"
                    if entity_id in result and not np.isclose(result[entity_id], precursor):
                        raise ValueError(f"conflicting precursor masses for {entity_id}")
                    result[entity_id] = precursor
                inside = False
    return result


def attach_mass(frame: pd.DataFrame, precursors: dict[str, float], left: str, right: str) -> pd.DataFrame:
    output = frame.copy()
    output["left_precursor_mz"] = output[left].astype(str).map(precursors)
    output["right_precursor_mz"] = output[right].astype(str).map(precursors)
    output["precursor_delta_da"] = (output.left_precursor_mz - output.right_precursor_mz).abs()
    denominator = (output.left_precursor_mz + output.right_precursor_mz) / 2.0
    output["precursor_delta_ppm"] = output.precursor_delta_da / denominator * 1e6
    for tolerance in (5, 10, 20):
        output[f"mass_compatible_{tolerance}ppm"] = output.precursor_delta_ppm.le(tolerance)
    return output


def transitive_component_audit(edges: pd.DataFrame, identity: pd.DataFrame) -> dict[str, int | float]:
    """Diagnose mutual-edge transitive closure; never use it to create truth."""
    oriented: list[tuple[str, str]] = []
    for row in edges.itertuples(index=False):
        if row.direction == "left_to_right":
            oriented.append((str(row.left_entity_id), str(row.right_entity_id)))
        elif row.direction == "right_to_left":
            oriented.append((str(row.right_entity_id), str(row.left_entity_id)))
        else:
            raise ValueError(f"unknown edge direction {row.direction}")
    directed = set(oriented)
    mutual = {tuple(sorted((left, right))) for left, right in oriented if (right, left) in directed}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in mutual:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[str]] = []
    seen: set[str] = set()
    for seed in adjacency:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    key_by_entity = identity.set_index("entity_id")["planar_inchikey"].to_dict()
    calibratable: list[tuple[int, bool]] = []
    for component in components:
        keys = [key_by_entity.get(entity) for entity in component]
        keys = [str(value) for value in keys if pd.notna(value)]
        if len(keys) >= 2:
            pairs = len(keys) * (len(keys) - 1) // 2
            calibratable.append((pairs, len(set(keys)) == 1))
    pure = sum(is_pure for _, is_pure in calibratable)
    pair_denominator = sum(pairs for pairs, _ in calibratable)
    pairs_in_pure = sum(pairs for pairs, is_pure in calibratable if is_pure)
    return {
        "mutual_unordered_edges": len(mutual),
        "transitive_components": len(components),
        "calibratable_components_with_at_least_two_strict_identities": len(calibratable),
        "identity_pure_calibratable_components": pure,
        "identity_pure_component_fraction": pure / len(calibratable) if calibratable else None,
        "known_member_pairs_in_calibratable_components": pair_denominator,
        "known_member_pairs_belonging_to_pure_components": pairs_in_pure,
        "pure_component_pair_fraction": pairs_in_pure / pair_denominator if pair_denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-contract", required=True, type=Path)
    parser.add_argument("--frozen-edges", required=True, type=Path)
    parser.add_argument("--strict-truth-audit", required=True, type=Path)
    parser.add_argument("--strict-identity-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    contract = json.loads(args.collection_contract.read_text(encoding="utf-8"))
    precursors: dict[str, float] = {}
    mgf_paths: dict[str, Path] = {}
    for source in contract["sources"]:
        study_id = str(source["study_id"])
        mgf_path = Path(source["mgf_path"])
        mgf_paths[study_id] = mgf_path
        source_precursors = read_precursors(mgf_path, study_id)
        overlap = set(precursors) & set(source_precursors)
        if overlap:
            raise ValueError(f"entity identifiers overlap between sources: {sorted(overlap)[:3]}")
        precursors.update(source_precursors)

    edges = pd.read_csv(args.frozen_edges, sep="\t")
    edges = edges.loc[edges.evidence_world.eq("all")].copy()
    edges = attach_mass(edges, precursors, "left_entity_id", "right_entity_id")

    truth = pd.read_csv(args.strict_truth_audit, sep="\t")
    truth = truth.loc[truth.evidence_world.eq("all")].copy()
    truth = attach_mass(truth, precursors, "query_entity_id", "candidate_entity_id")
    truth["edge_and_mass_available"] = truth.edge_available & truth.precursor_delta_ppm.notna()
    identity = pd.read_csv(args.strict_identity_ledger, sep="\t")
    component_audit = transitive_component_audit(edges, identity)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    edge_path = args.output_dir / "frozen_all_world_edges_with_mass.tsv.gz"
    truth_path = args.output_dir / "strict_identity_groups_with_mass.tsv.gz"
    edges.to_csv(edge_path, sep="\t", index=False, compression=compression)
    truth.to_csv(truth_path, sep="\t", index=False, compression=compression)

    edge_mass_available = edges.precursor_delta_ppm.notna()
    edge_denominator = int(edge_mass_available.sum())
    answered = truth.loc[truth.edge_available].copy()
    thresholds: dict[str, object] = {}
    for tolerance in (5, 10, 20):
        eligible = answered.loc[answered[f"mass_compatible_{tolerance}ppm"]]
        correct = int(eligible.top1_exact_identity_correct.sum())
        thresholds[f"{tolerance}ppm"] = {
            "answered_groups": int(len(eligible)),
            "correct_groups": correct,
            "precision_among_answered": correct / len(eligible) if len(eligible) else None,
            "coverage_of_all_truth_groups": len(eligible) / len(truth) if len(truth) else None,
            "coverage_of_previous_answered_groups": len(eligible) / len(answered) if len(answered) else None,
        }

    correct_answered = answered.loc[answered.top1_exact_identity_correct]
    incorrect_answered = answered.loc[~answered.top1_exact_identity_correct]
    report = {
        "schema": "contrast_atlas_identity_relation_boundary_audit_v1",
        "status": "DEVELOPMENT_CONSUMED_IDENTITY_RELATION_BOUNDARY_AUDITED",
        "evidence_role": {
            "dataset_status": "development_consumed",
            "leakage_state": "strict_identity_truth_opened_post_graph_for_diagnostic_calibration",
            "external_confirmation_claim": False,
            "threshold_selection_claim": False,
        },
        "frozen_graph": {
            "all_world_directional_edges": int(len(edges)),
            "edges_with_both_precursor_masses": edge_denominator,
            "precursor_mass_coverage": edge_denominator / len(edges),
            "median_absolute_delta_da": float(edges.loc[edge_mass_available, "precursor_delta_da"].median()),
            "within_5ppm": int(edges.mass_compatible_5ppm.sum()),
            "within_10ppm": int(edges.mass_compatible_10ppm.sum()),
            "within_20ppm": int(edges.mass_compatible_20ppm.sum()),
        },
        "strict_identity_calibration": {
            "truth_query_target_source_groups": int(len(truth)),
            "previous_answered_groups": int(len(answered)),
            "previous_correct_groups": int(answered.top1_exact_identity_correct.sum()),
            "previous_precision_among_answered": float(answered.top1_exact_identity_correct.mean()),
            "correct_edges_within_5ppm": int(correct_answered.mass_compatible_5ppm.sum()),
            "incorrect_edges_within_5ppm": int(incorrect_answered.mass_compatible_5ppm.sum()),
            "incorrect_edges_beyond_5ppm": int((~incorrect_answered.mass_compatible_5ppm).sum()),
            "mass_compatibility_sensitivity": thresholds,
        },
        "transitive_component_diagnostic": component_audit,
        "scientific_decision": {
            "identity_track": (
                "Quantitative effects may be combined only across precursor/adduct-compatible, "
                "identity-calibrated links; a calibrated no-match option is mandatory."
            ),
            "relation_track": (
                "Cross-mass spectral neighbours remain chemical-analogy hypotheses and may not "
                "count as replication of one entity or be meta-analysed as one effect."
            ),
            "transitive_family_closure": "forbidden_for_claim_bearing_identity_evidence",
        },
        "claim_boundary": [
            "The 5/10/20 ppm rows are a posttruth development sensitivity grid, not a selected production threshold.",
            "Planar identity is used only to diagnose the already-frozen graph.",
            "Mass compatibility alone does not establish identity, especially across adducts and isomers.",
            "No unresolved entity, cross-study response, or biological mechanism is validated by this audit.",
        ],
        "provenance": {
            "collection_contract_sha256": sha256(args.collection_contract),
            "frozen_edges_sha256": sha256(args.frozen_edges),
            "strict_truth_audit_sha256": sha256(args.strict_truth_audit),
            "strict_identity_ledger_sha256": sha256(args.strict_identity_ledger),
            "mgf_sha256": {study: sha256(path) for study, path in mgf_paths.items()},
            "edge_output_sha256": sha256(edge_path),
            "truth_output_sha256": sha256(truth_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

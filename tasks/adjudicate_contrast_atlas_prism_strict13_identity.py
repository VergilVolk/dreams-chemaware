#!/usr/bin/env python3
"""Adjudicate identity evidence and analytical redundancy for PRISM strict hits.

The input rows have already passed the three within-fit response filters in the
PRISM/Netherlands development analysis.  This script does not use response
magnitudes to decide chemical identity.  It joins the frozen source-pair MS/MS
and library evidence, then performs a phenotype-blind analytical-family audit
using only acquisition mode, retention time and prespecified positive-ion mass
differences.

The resulting family components are analytical hypotheses, not confirmed
molecular identities.  Only an existing library-policy decision may authorize
merging the two source-study entities as the same identified compound.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable


ION_RELATIONSHIPS_DA = (
    ("water_loss_or_gain", 18.010565),
    ("ammonium_minus_proton", 17.026549),
    ("sodium_minus_proton", 21.981942),
    ("potassium_minus_proton", 37.955882),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-candidates", type=Path, required=True)
    parser.add_argument("--library-annotations", type=Path, required=True)
    parser.add_argument("--left-mgf", type=Path, required=True)
    parser.add_argument("--right-mgf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rt-tolerance-min", type=float, default=0.03)
    parser.add_argument("--mass-delta-tolerance-da", type=float, default=0.01)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_or_blank(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_rt(feature_id: str) -> float:
    try:
        return float(feature_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"feature_id does not end in retention time: {feature_id}") from exc


def raw_entity_id(entity_id: str) -> int:
    try:
        return int(entity_id.rsplit("::", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"entity_id does not end in an integer: {entity_id}") from exc


def read_mgf(path: Path, wanted_ids: set[int]) -> dict[int, list[tuple[float, float]]]:
    spectra: dict[int, list[tuple[float, float]]] = {}
    scan: int | None = None
    peaks: list[tuple[float, float]] = []
    inside = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                inside, scan, peaks = True, None, []
            elif inside and line.startswith("SCANS="):
                scan = int(line.split("=", 1)[1])
            elif inside and line == "END IONS":
                if scan in wanted_ids:
                    spectra[int(scan)] = peaks
                inside = False
            elif inside and line and "=" not in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except ValueError:
                        pass
    if set(spectra) != wanted_ids:
        raise ValueError(f"missing requested spectra in {path}: {wanted_ids - set(spectra)}")
    return spectra


def sqrt_cosine(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    tolerance: float = 0.02,
) -> tuple[float, int]:
    """Phenotype-blind raw-peak diagnostic; not a frozen Atlas model score."""

    def normalise(peaks: list[tuple[float, float]]) -> list[tuple[float, float]]:
        values = sorted((mz, math.sqrt(max(intensity, 0.0))) for mz, intensity in peaks)
        norm = math.sqrt(sum(intensity * intensity for _, intensity in values))
        if norm <= 0:
            raise ValueError("zero-intensity spectrum")
        return [(mz, intensity / norm) for mz, intensity in values]

    left_n, right_n = normalise(left), normalise(right)
    left_index = right_index = matched = 0
    score = 0.0
    while left_index < len(left_n) and right_index < len(right_n):
        delta = left_n[left_index][0] - right_n[right_index][0]
        if abs(delta) <= tolerance:
            score += left_n[left_index][1] * right_n[right_index][1]
            matched += 1
            left_index += 1
            right_index += 1
        elif delta < 0:
            left_index += 1
        else:
            right_index += 1
    return float(score), matched


def identity_evidence_class(row: dict[str, str]) -> str:
    if as_bool(row["identity_merge_allowed_by_library_evidence"]):
        return "exact_known_identity_merge_allowed"
    if row["library_edge_class"] == "mixed_one_named_anchor":
        return "mixed_one_named_anchor_not_mergeable"

    left_key = row.get("left_library_inchikey", "").strip()
    right_key = row.get("right_library_inchikey", "").strip()
    left_class = row.get("left_library_evidence_class", "")
    right_class = row.get("right_library_evidence_class", "")
    left_candidate = left_class == "below_policy_library_candidate" and bool(left_key)
    right_candidate = right_class == "below_policy_library_candidate" and bool(right_key)
    if left_candidate and right_candidate and left_key == right_key:
        return "shared_below_policy_candidate_not_mergeable"
    if left_candidate and right_candidate:
        return "conflicting_below_policy_candidates"
    if left_candidate or right_candidate:
        return "one_sided_below_policy_candidate"
    return "no_qualified_library_candidate"


def compact_candidate(value: str) -> str:
    return value.strip() if value and value.strip() else ""


def build_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        queue: deque[str] = deque([start])
        seen.add(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    return components


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strict_rows = read_tsv(args.strict_candidates)
    annotation_rows = read_tsv(args.library_annotations)
    if not strict_rows:
        raise ValueError("strict-candidates input is empty")

    strict_pair_keys = {
        (row["method_edge_id"], row["left_entity_id"], row["right_entity_id"])
        for row in strict_rows
    }
    identity_world_by_edge: dict[str, str] = {}
    for row in strict_rows:
        method_edge_id = row["method_edge_id"]
        identity_world = row["identity_world"]
        previous = identity_world_by_edge.setdefault(method_edge_id, identity_world)
        if previous != identity_world:
            raise ValueError(f"inconsistent identity world for {method_edge_id}")
    annotations = {
        (row["method_edge_id"], row["left_entity_id"], row["right_entity_id"]): row
        for row in annotation_rows
        if (row["method_edge_id"], row["left_entity_id"], row["right_entity_id"])
        in strict_pair_keys
    }
    if set(annotations) != strict_pair_keys:
        missing = sorted(strict_pair_keys - set(annotations))
        raise ValueError(f"missing library annotations for {missing}")
    if len(annotations) != len(strict_pair_keys):
        raise ValueError("library annotation join is not one-to-one")

    left_ids = {raw_entity_id(key[1]) for key in strict_pair_keys}
    right_ids = {raw_entity_id(key[2]) for key in strict_pair_keys}
    left_spectra = read_mgf(args.left_mgf, left_ids)
    right_spectra = read_mgf(args.right_mgf, right_ids)

    pair_identity_rows: list[dict] = []
    class_counts: dict[str, int] = defaultdict(int)
    for key in sorted(strict_pair_keys):
        row = annotations[key]
        evidence_class = identity_evidence_class(row)
        class_counts[evidence_class] += 1
        left_peaks = left_spectra[raw_entity_id(row["left_entity_id"])]
        right_peaks = right_spectra[raw_entity_id(row["right_entity_id"])]
        diagnostic_cosine, diagnostic_matches = sqrt_cosine(left_peaks, right_peaks)
        pair_identity_rows.append(
            {
                "method_edge_id": row["method_edge_id"],
                "left_entity_id": row["left_entity_id"],
                "right_entity_id": row["right_entity_id"],
                "left_precursor_mz": row["left_precursor_mz"],
                "right_precursor_mz": row["right_precursor_mz"],
                "precursor_ppm_difference": row["precursor_ppm_difference"],
                "cross_study_wse": row["method_evidence_weight"],
                "diagnostic_raw_sqrt_cosine_0p02": diagnostic_cosine,
                "diagnostic_raw_matched_peaks_0p02": diagnostic_matches,
                "left_raw_peak_count": len(left_peaks),
                "right_raw_peak_count": len(right_peaks),
                "left_common_identity_status": row["left_common_identity_status"],
                "right_common_identity_status": row["right_common_identity_status"],
                "left_library_evidence_class": row["left_library_evidence_class"],
                "right_library_evidence_class": row["right_library_evidence_class"],
                "left_library_name": compact_candidate(row.get("left_library_name", "")),
                "right_library_name": compact_candidate(row.get("right_library_name", "")),
                "left_library_inchikey": compact_candidate(row.get("left_library_inchikey", "")),
                "right_library_inchikey": compact_candidate(row.get("right_library_inchikey", "")),
                "left_library_spectrum_id": compact_candidate(row.get("left_library_spectrum_id", "")),
                "right_library_spectrum_id": compact_candidate(row.get("right_library_spectrum_id", "")),
                "left_library_wse": row.get("left_wse", ""),
                "right_library_wse": row.get("right_wse", ""),
                "left_matched_peaks": row.get("left_matched_peaks", ""),
                "right_matched_peaks": row.get("right_matched_peaks", ""),
                "left_library_candidates_10ppm": row.get("left_library_candidates_10ppm", ""),
                "right_library_candidates_10ppm": row.get("right_library_candidates_10ppm", ""),
                "library_edge_class": row["library_edge_class"],
                "identity_merge_allowed": as_bool(row["identity_merge_allowed_by_library_evidence"]),
                "identity_evidence_class": evidence_class,
                "claim_boundary": (
                    "same_source_entity_merge_allowed" if as_bool(
                        row["identity_merge_allowed_by_library_evidence"]
                    ) else "identity_hypothesis_only_do_not_merge_beta_or_name"
                ),
            }
        )

    identity_fields = list(pair_identity_rows[0])
    identity_path = args.output_dir / "strict13_identity_evidence.tsv"
    write_tsv(identity_path, pair_identity_rows, identity_fields)

    # Collapse repeated diagnosis rows into unique feature-mode observations.
    feature_map: dict[tuple[str, str, str], dict] = {}
    for row in strict_rows:
        key = (row["method_edge_id"], row["analysis_id"], row["feature_id"])
        current = feature_map.setdefault(
            key,
            {
                "method_edge_id": row["method_edge_id"],
                "analysis_id": row["analysis_id"],
                "mode": row["mode"],
                "feature_id": row["feature_id"],
                "feature_mz": float(row["feature_mz"]),
                "feature_rt_min": parse_rt(row["feature_id"]),
                "identity_world": row["identity_world"],
                "expected_direction": int(row["expected_direction"]),
                "diagnoses": set(),
            },
        )
        current["diagnoses"].add(row["diagnosis"])
    features = list(feature_map.values())
    for row in features:
        row["diagnoses"] = ",".join(sorted(row["diagnoses"]))

    relation_rows: list[dict] = []
    graph_edges: list[tuple[str, str]] = []
    for index, left in enumerate(features):
        for right in features[index + 1 :]:
            if left["method_edge_id"] == right["method_edge_id"]:
                continue
            if left["mode"] != right["mode"]:
                continue
            rt_delta = abs(left["feature_rt_min"] - right["feature_rt_min"])
            if rt_delta > args.rt_tolerance_min:
                continue
            mass_delta = abs(left["feature_mz"] - right["feature_mz"])
            matches = [
                (name, target, abs(mass_delta - target))
                for name, target in ION_RELATIONSHIPS_DA
                if abs(mass_delta - target) <= args.mass_delta_tolerance_da
            ]
            if not matches:
                continue
            relation, target, error = min(matches, key=lambda item: item[2])
            graph_edges.append((left["method_edge_id"], right["method_edge_id"]))
            relation_rows.append(
                {
                    "left_method_edge_id": left["method_edge_id"],
                    "right_method_edge_id": right["method_edge_id"],
                    "mode": left["mode"],
                    "left_feature_id": left["feature_id"],
                    "right_feature_id": right["feature_id"],
                    "left_mz": left["feature_mz"],
                    "right_mz": right["feature_mz"],
                    "left_rt_min": left["feature_rt_min"],
                    "right_rt_min": right["feature_rt_min"],
                    "rt_delta_min": rt_delta,
                    "observed_mass_delta_da": mass_delta,
                    "candidate_relation": relation,
                    "reference_mass_delta_da": target,
                    "mass_delta_error_da": error,
                    "claim_boundary": "phenotype_blind_analytical_family_hypothesis_not_confirmed_identity",
                }
            )

    relation_fields = [
        "left_method_edge_id",
        "right_method_edge_id",
        "mode",
        "left_feature_id",
        "right_feature_id",
        "left_mz",
        "right_mz",
        "left_rt_min",
        "right_rt_min",
        "rt_delta_min",
        "observed_mass_delta_da",
        "candidate_relation",
        "reference_mass_delta_da",
        "mass_delta_error_da",
        "claim_boundary",
    ]
    relation_path = args.output_dir / "strict13_candidate_ion_relation_edges.tsv"
    write_tsv(relation_path, relation_rows, relation_fields)

    pair_nodes = sorted({row["method_edge_id"] for row in pair_identity_rows})
    components = build_components(pair_nodes, graph_edges)
    component_by_node: dict[str, tuple[str, list[str]]] = {}
    for index, component in enumerate(components, start=1):
        family_id = f"analytical_family_{index:02d}"
        for node in component:
            component_by_node[node] = (family_id, component)

    family_rows: list[dict] = []
    identity_by_edge = {row["method_edge_id"]: row for row in pair_identity_rows}
    features_by_edge: dict[str, list[dict]] = defaultdict(list)
    for row in features:
        features_by_edge[row["method_edge_id"]].append(row)
    for method_edge_id in pair_nodes:
        family_id, component = component_by_node[method_edge_id]
        identity = identity_by_edge[method_edge_id]
        observation_text = ";".join(
            f"{row['mode']}:{row['feature_id']}"
            for row in sorted(
                features_by_edge[method_edge_id],
                key=lambda item: (item["mode"], item["feature_mz"]),
            )
        )
        family_rows.append(
            {
                "analytical_family_id": family_id,
                "family_source_pair_count": len(component),
                "family_method_edge_ids": ",".join(component),
                "method_edge_id": method_edge_id,
                "left_entity_id": identity["left_entity_id"],
                "right_entity_id": identity["right_entity_id"],
                "identity_evidence_class": identity["identity_evidence_class"],
                "identity_merge_allowed": identity["identity_merge_allowed"],
                "st001000_feature_observations": observation_text,
                "family_claim_boundary": (
                    "single_source_pair_hypothesis"
                    if len(component) == 1
                    else "coeluting_mass_delta_family_hypothesis_not_confirmed_single_molecule"
                ),
            }
        )

    family_path = args.output_dir / "strict13_candidate_analytical_families.tsv"
    write_tsv(family_path, family_rows, list(family_rows[0]))

    relations_by_component: dict[frozenset[str], list[dict]] = defaultdict(list)
    for relation in relation_rows:
        for component in components:
            if {
                relation["left_method_edge_id"],
                relation["right_method_edge_id"],
            }.issubset(component):
                relations_by_component[frozenset(component)].append(relation)
                break

    family_summary_rows: list[dict] = []
    family_world_counts: dict[str, int] = defaultdict(int)
    for index, component in enumerate(components, start=1):
        family_id = f"analytical_family_{index:02d}"
        worlds = {identity_world_by_edge[edge] for edge in component}
        if worlds == {"known"}:
            family_world = "known"
        elif worlds == {"unresolved"}:
            family_world = "unresolved"
        else:
            family_world = "mixed_known_unresolved"
        family_world_counts[family_world] += 1
        observations = []
        for edge in component:
            observations.extend(
                f"{row['mode']}:{row['feature_id']}"
                for row in features_by_edge[edge]
            )
        component_relations = relations_by_component[frozenset(component)]
        family_summary_rows.append(
            {
                "analytical_family_id": family_id,
                "family_identity_world": family_world,
                "source_pair_count": len(component),
                "method_edge_ids": ",".join(component),
                "st001000_feature_observations": ";".join(sorted(observations)),
                "candidate_ion_relations": ",".join(
                    sorted({row["candidate_relation"] for row in component_relations})
                ),
                "identity_merge_allowed_source_pairs": sum(
                    as_bool(str(identity_by_edge[edge]["identity_merge_allowed"]))
                    for edge in component
                ),
                "family_claim_boundary": (
                    "single_source_pair_hypothesis"
                    if len(component) == 1
                    else "coeluting_mass_delta_family_hypothesis_not_confirmed_single_molecule"
                ),
            }
        )
    family_summary_path = args.output_dir / "strict13_candidate_analytical_family_summary.tsv"
    write_tsv(family_summary_path, family_summary_rows, list(family_summary_rows[0]))

    feature_rows = []
    for row in sorted(features, key=lambda item: (item["feature_mz"], item["mode"])):
        feature_rows.append({key: row[key] for key in row})
    feature_path = args.output_dir / "strict13_unique_feature_observations.tsv"
    write_tsv(feature_path, feature_rows, list(feature_rows[0]))

    report = {
        "status": "COMPLETE",
        "data_role": "development_consumed_identity_adjudication",
        "scientific_question": (
            "How many of the 13 PRISM-prioritized source entity-pair hypotheses have "
            "policy-qualified identity evidence, and how many remain after a phenotype-blind "
            "coelution/adduct-family redundancy audit?"
        ),
        "strict_response_rows": len(strict_rows),
        "strict_unique_source_entity_pairs": len(pair_nodes),
        "strict_unique_st001000_feature_mode_observations": len(features),
        "identity_evidence_class_counts": dict(sorted(class_counts.items())),
        "identity_merge_allowed_pairs": sum(
            bool(row["identity_merge_allowed"]) for row in pair_identity_rows
        ),
        "identity_hypothesis_only_pairs": sum(
            not bool(row["identity_merge_allowed"]) for row in pair_identity_rows
        ),
        "candidate_ion_relation_edges": len(relation_rows),
        "candidate_analytical_families_after_prespecified_collapse": len(components),
        "candidate_analytical_family_range": {
            "lower_bound_under_prespecified_coelution_mass_delta_collapse": len(components),
            "upper_bound_without_cross_pair_collapse": len(pair_nodes),
        },
        "candidate_analytical_families_by_identity_world": dict(
            sorted(family_world_counts.items())
        ),
        "multi_pair_candidate_analytical_families": sum(
            len(component) > 1 for component in components
        ),
        "family_parameters": {
            "same_acquisition_mode_required": True,
            "rt_tolerance_min": args.rt_tolerance_min,
            "mass_delta_tolerance_da": args.mass_delta_tolerance_da,
            "positive_ion_relationships_da": {
                name: value for name, value in ION_RELATIONSHIPS_DA
            },
        },
        "claim_boundary": (
            "The 13 rows are response-prioritized source entity-pair hypotheses, not 13 "
            "metabolites. Library-policy evidence permits two source-pair identity merges. "
            "Analytical-family components are phenotype-blind redundancy hypotheses and do "
            "not themselves confirm a common neutral molecule."
        ),
        "input_sha256": {
            "strict_candidates": sha256(args.strict_candidates),
            "library_annotations": sha256(args.library_annotations),
            "left_mgf": sha256(args.left_mgf),
            "right_mgf": sha256(args.right_mgf),
        },
        "diagnostic_score_boundary": (
            "diagnostic_raw_sqrt_cosine_0p02 is recomputed from exact MGF peaks with a "
            "greedy 0.02-Da matcher. It is response-blind but is not a frozen Atlas model "
            "score and must not replace the canonical cross-study WSE."
        ),
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            "identity_evidence": identity_path.name,
            "candidate_ion_relation_edges": relation_path.name,
            "candidate_analytical_families": family_path.name,
            "candidate_analytical_family_summary": family_summary_path.name,
            "unique_feature_observations": feature_path.name,
        },
        "unknown_increment_boundary": (
            "Family collapse changes both numerator and denominator. Unknown enrichment must "
            "not be recomputed by collapsing only the 13 prioritized pairs; the full set of "
            "101 retained testable pairs requires the same phenotype-blind family procedure."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_paths = [
        identity_path,
        relation_path,
        family_path,
        family_summary_path,
        feature_path,
        report_path,
    ]
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_paths),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute the PRISM unknown-increment sensitivity after family collapse.

The same phenotype-blind coelution/mass-difference graph is applied to every
retained testable source entity-pair, not only to response-prioritized pairs.
This prevents a numerator-only collapse from creating an artificial known or
unresolved advantage.  The analysis is a family-level sensitivity result; it
does not convert the upstream within-fit BH filters into replicated-findings
FDR control.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from adjudicate_contrast_atlas_prism_strict13_identity import (
    ION_RELATIONSHIPS_DA,
    build_components,
    parse_rt,
    read_tsv,
    sha256,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-candidates", type=Path, required=True)
    parser.add_argument("--strict-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rt-tolerance-min", type=float, default=0.03)
    parser.add_argument("--mass-delta-tolerance-da", type=float, default=0.01)
    return parser.parse_args()


def classify_world(worlds: set[str]) -> str:
    if worlds == {"known"}:
        return "known"
    if worlds == {"unresolved"}:
        return "unresolved"
    return "mixed_known_unresolved"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained = read_tsv(args.retained_candidates)
    strict = read_tsv(args.strict_candidates)
    if not retained or not strict:
        raise ValueError("input table is empty")

    pair_rows: dict[str, dict[str, str]] = {}
    features: list[dict] = []
    feature_keys: set[tuple[str, str]] = set()
    for row in retained:
        method_edge_id = row["method_edge_id"]
        previous = pair_rows.setdefault(method_edge_id, row)
        for column in (
            "left_entity_id",
            "right_entity_id",
            "identity_world",
            "expected_direction",
        ):
            if previous[column] != row[column]:
                raise ValueError(f"inconsistent {column} for {method_edge_id}")
        feature_key = (row["analysis_id"], row["feature_id"])
        if feature_key in feature_keys:
            raise ValueError(f"retained feature was assigned more than once: {feature_key}")
        feature_keys.add(feature_key)
        features.append(
            {
                "method_edge_id": method_edge_id,
                "analysis_id": row["analysis_id"],
                "mode": row["mode"],
                "feature_id": row["feature_id"],
                "feature_mz": float(row["feature_mz"]),
                "feature_rt_min": parse_rt(row["feature_id"]),
            }
        )

    strict_pairs = {row["method_edge_id"] for row in strict}
    if not strict_pairs.issubset(pair_rows):
        raise ValueError("strict pair is absent from retained denominator")

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
                    "rt_delta_min": rt_delta,
                    "observed_mass_delta_da": mass_delta,
                    "candidate_relation": relation,
                    "reference_mass_delta_da": target,
                    "mass_delta_error_da": error,
                    "claim_boundary": "phenotype_blind_family_edge_not_confirmed_identity",
                }
            )

    components = build_components(sorted(pair_rows), graph_edges)
    features_by_edge: dict[str, list[str]] = defaultdict(list)
    for row in features:
        features_by_edge[row["method_edge_id"]].append(
            f"{row['mode']}:{row['feature_id']}"
        )
    relations_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in relation_rows:
        relations_by_pair[row["left_method_edge_id"]].add(row["candidate_relation"])
        relations_by_pair[row["right_method_edge_id"]].add(row["candidate_relation"])

    assignment_rows: list[dict] = []
    summary_rows: list[dict] = []
    for index, component in enumerate(components, start=1):
        family_id = f"retained_family_{index:03d}"
        worlds = {pair_rows[edge]["identity_world"] for edge in component}
        family_world = classify_world(worlds)
        strict_members = sorted(set(component) & strict_pairs)
        observations: list[str] = []
        relations: set[str] = set()
        for edge in component:
            observations.extend(features_by_edge[edge])
            relations.update(relations_by_pair[edge])
            row = pair_rows[edge]
            assignment_rows.append(
                {
                    "family_id": family_id,
                    "family_identity_world": family_world,
                    "family_source_pair_count": len(component),
                    "family_passed_strict_screen": bool(strict_members),
                    "method_edge_id": edge,
                    "pair_identity_world": row["identity_world"],
                    "left_entity_id": row["left_entity_id"],
                    "right_entity_id": row["right_entity_id"],
                    "pair_passed_strict_screen": edge in strict_pairs,
                    "feature_observations": ";".join(sorted(features_by_edge[edge])),
                }
            )
        summary_rows.append(
            {
                "family_id": family_id,
                "family_identity_world": family_world,
                "source_pair_count": len(component),
                "method_edge_ids": ",".join(component),
                "strict_source_pair_count": len(strict_members),
                "strict_method_edge_ids": ",".join(strict_members),
                "family_passed_strict_screen": bool(strict_members),
                "candidate_ion_relations": ",".join(sorted(relations)),
                "feature_observations": ";".join(sorted(observations)),
                "claim_boundary": (
                    "family_level_response_priority_not_confirmed_molecule_or_replicated_findings_fdr"
                ),
            }
        )

    assignment_path = args.output_dir / "retained_pair_family_assignments.tsv"
    family_path = args.output_dir / "retained_family_summary.tsv"
    relation_path = args.output_dir / "retained_candidate_ion_relation_edges.tsv"
    write_tsv(assignment_path, assignment_rows, list(assignment_rows[0]))
    write_tsv(family_path, summary_rows, list(summary_rows[0]))
    relation_fields = [
        "left_method_edge_id",
        "right_method_edge_id",
        "mode",
        "left_feature_id",
        "right_feature_id",
        "rt_delta_min",
        "observed_mass_delta_da",
        "candidate_relation",
        "reference_mass_delta_da",
        "mass_delta_error_da",
        "claim_boundary",
    ]
    write_tsv(relation_path, relation_rows, relation_fields)

    counts: dict[str, dict[str, int | float]] = {}
    for world in ("known", "unresolved", "mixed_known_unresolved"):
        world_rows = [row for row in summary_rows if row["family_identity_world"] == world]
        passed = sum(bool(row["family_passed_strict_screen"]) for row in world_rows)
        counts[world] = {
            "families": len(world_rows),
            "strict_priority_families": passed,
            "strict_priority_rate": passed / len(world_rows) if world_rows else float("nan"),
        }

    unresolved_passed = int(counts["unresolved"]["strict_priority_families"])
    unresolved_total = int(counts["unresolved"]["families"])
    known_passed = int(counts["known"]["strict_priority_families"])
    known_total = int(counts["known"]["families"])
    odds_ratio, p_value = fisher_exact(
        [
            [unresolved_passed, unresolved_total - unresolved_passed],
            [known_passed, known_total - known_passed],
        ],
        alternative="greater",
    )

    report = {
        "status": "COMPLETE",
        "data_role": "development_consumed_family_level_sensitivity",
        "scientific_question": (
            "Does unresolved-versus-known response prioritization change after applying the "
            "same phenotype-blind analytical-family collapse to all retained testable pairs?"
        ),
        "retained_source_entity_pairs": len(pair_rows),
        "retained_feature_mode_observations": len(features),
        "strict_source_entity_pairs": len(strict_pairs),
        "candidate_ion_relation_edges": len(relation_rows),
        "retained_candidate_analytical_families": len(components),
        "multi_pair_candidate_analytical_families": sum(
            len(component) > 1 for component in components
        ),
        "largest_family_source_pair_count": max(len(component) for component in components),
        "strict_priority_candidate_analytical_families": sum(
            bool(row["family_passed_strict_screen"]) for row in summary_rows
        ),
        "family_level_counts_by_identity_world": counts,
        "family_level_unknown_increment_test": {
            "comparison": "unresolved_vs_known_strict_priority_rate_after_common_family_collapse",
            "mixed_world_excluded": True,
            "alternative": "unresolved_greater_than_known",
            "odds_ratio": float(odds_ratio),
            "fisher_exact_p": float(p_value),
            "interpretation": "no_evidence_of_unresolved_family_enrichment",
        },
        "pair_level_reference": {
            "unresolved": {"strict_pairs": 9, "retained_pairs": 82},
            "known": {"strict_pairs": 2, "retained_pairs": 12},
            "mixed_known_unresolved": {"strict_pairs": 2, "retained_pairs": 7},
            "unresolved_vs_known_odds_ratio": 0.6164383561643836,
            "unresolved_vs_known_one_sided_fisher_p": 0.8547707845158647,
        },
        "family_parameters": {
            "same_acquisition_mode_required": True,
            "rt_tolerance_min": args.rt_tolerance_min,
            "mass_delta_tolerance_da": args.mass_delta_tolerance_da,
            "positive_ion_relationships_da": {
                name: value for name, value in ION_RELATIONSHIPS_DA
            },
        },
        "claim_boundary": (
            "Family construction is phenotype-blind and applied to the complete retained "
            "denominator. A family passes when any member pair was already prioritized by the "
            "upstream three-fit screen. This is a redundancy sensitivity analysis, not "
            "family-level replicated-findings FDR control or molecular identity confirmation."
        ),
        "input_sha256": {
            "retained_candidates": sha256(args.retained_candidates),
            "strict_candidates": sha256(args.strict_candidates),
        },
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            "pair_family_assignments": assignment_path.name,
            "family_summary": family_path.name,
            "candidate_ion_relation_edges": relation_path.name,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths = [assignment_path, family_path, relation_path, report_path]
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_paths),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

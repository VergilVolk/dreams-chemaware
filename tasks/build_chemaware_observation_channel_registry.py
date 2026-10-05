"""Rebuild the legacy ChemAware rule list as an auditable observation registry.

The legacy NL/CF entries identify observed masses. They generally do not encode
an executable parent-structure prerequisite, collision-energy domain, or a
validated per-rule primary citation. Therefore they are observation-channel
candidates, not mechanistic chemical rules. This builder retains aliases and
provenance, enforces the positive-ion task domain, merges mass-degenerate
channels, and quarantines unsupported records instead of silently enabling them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "data/validation/chemaware_observation_channel_registry_v2/report.json",
    )
    parser.add_argument("--merge-tolerance-da", type=float, default=0.005)
    return parser.parse_args()


def mode_set(value: object) -> list[str]:
    if value == "pos":
        return ["positive"]
    if value == "neg":
        return ["negative"]
    if value == "pos+neg":
        return ["positive", "negative"]
    return []


def evidence_class(rule: dict) -> str:
    source = str(rule.get("source", ""))
    if source.startswith("CompMS2Miner") and str(rule.get("ref", "")).strip():
        return "curated_package_record_with_citation_text"
    if source.startswith("baseline"):
        return "legacy_unverified"
    if source.startswith("ESI"):
        return "manual_empirical_unverified"
    return "unsupported"


def candidate_for_positive_calibration(rule: dict) -> bool:
    return (
        rule.get("category") in {"NL", "CF"}
        and rule.get("match_type") in {"mass_diff", "peak_mz"}
        and np.isscalar(rule.get("value"))
        and np.isfinite(float(rule["value"]))
        and float(rule["value"]) > 0
        and "positive" in mode_set(rule.get("mode"))
        and evidence_class(rule) == "curated_package_record_with_citation_text"
    )


def group_channels(rules: list[dict], tolerance: float) -> list[dict]:
    channels = []
    for category, match_type in (("NL", "mass_diff"), ("CF", "peak_mz")):
        selected = sorted(
            [rule for rule in rules if candidate_for_positive_calibration(rule)
             and rule["category"] == category and rule["match_type"] == match_type],
            key=lambda rule: (float(rule["value"]), str(rule["name"])),
        )
        groups: list[list[dict]] = []
        for rule in selected:
            if groups and float(rule["value"]) - float(groups[-1][-1]["value"]) < tolerance:
                groups[-1].append(rule)
            else:
                groups.append([rule])
        for group in groups:
            values = [float(rule["value"]) for rule in group]
            channel_id = f"OBS:{category}:{len(channels):04d}"
            channels.append({
                "channel_id": channel_id,
                "category": category,
                "observation": "precursor_minus_fragment_da" if category == "NL" else "fragment_mz",
                "match_type": match_type,
                "value_da": float(np.mean(values)),
                "value_span_da": [float(min(values)), float(max(values))],
                "match_tolerance_da": tolerance,
                "ion_modes": ["positive"],
                "adduct_scope": ["[M+H]+", "[M+Na]+"],
                "aliases": [str(rule["name"]) for rule in group],
                "observed_species_formulae": sorted({
                    str(rule.get("formula", "")).strip() for rule in group
                    if str(rule.get("formula", "")).strip()
                }),
                "citation_text": sorted({str(rule.get("ref", "")).strip() for rule in group}),
                "source_records": [
                    {"name": rule["name"], "source": rule.get("source", ""),
                     "value": float(rule["value"]), "mode": rule.get("mode"),
                     "formula": rule.get("formula", ""), "ref": rule.get("ref", "")}
                    for rule in group
                ],
                "parent_structure_prerequisite": None,
                "collision_energy_domain": None,
                "instrument_domain": None,
                "evidence_class": "curated_package_record_with_citation_text",
                "mechanistic_rule": False,
                "training_policy": "empirical_calibration_required",
                "enabled_for_formal_training": False,
            })
    return channels


def mass_collision_groups(rules: list[dict], tolerance: float) -> list[list[str]]:
    output = []
    for category, match_type in (("NL", "mass_diff"), ("CF", "peak_mz")):
        selected = sorted(
            [rule for rule in rules if rule.get("category") == category
             and rule.get("match_type") == match_type and np.isscalar(rule.get("value"))],
            key=lambda rule: float(rule["value"]),
        )
        group = []
        for rule in selected:
            if group and float(rule["value"]) - float(group[-1]["value"]) >= tolerance:
                if len(group) > 1:
                    output.append([str(item["name"]) for item in group])
                group = []
            group.append(rule)
        if len(group) > 1:
            output.append([str(item["name"]) for item in group])
    return output


def main() -> None:
    args = arguments()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite observation registry")
    if args.merge_tolerance_da <= 0:
        raise ValueError("merge tolerance must be positive")
    legacy = json.loads(args.legacy_library.read_text(encoding="utf-8"))
    rules = legacy.get("rules", [])
    if int(legacy.get("n_rules", -1)) != len(rules):
        raise RuntimeError("legacy rule count is internally inconsistent")
    candidates = [rule for rule in rules if candidate_for_positive_calibration(rule)]
    channels = group_channels(rules, args.merge_tolerance_da)
    legacy_collisions = mass_collision_groups(rules, args.merge_tolerance_da)
    quarantined = []
    for rule in rules:
        reasons = []
        if rule.get("category") not in {"NL", "CF"}:
            reasons.append("unsupported_by_current_spectrum_kernel")
        if rule.get("category") in {"NL", "CF"} and not mode_set(rule.get("mode")):
            reasons.append("ion_mode_unknown")
        if rule.get("category") in {"NL", "CF"} and mode_set(rule.get("mode")) == ["negative"]:
            reasons.append("outside_positive_ion_task")
        if evidence_class(rule) != "curated_package_record_with_citation_text":
            reasons.append(evidence_class(rule))
        if rule.get("category") in {"NL", "CF"} and not str(rule.get("formula", "")).strip():
            reasons.append("observed_species_formula_missing")
        if not candidate_for_positive_calibration(rule):
            quarantined.append({"name": rule.get("name"), "reasons": sorted(set(reasons))})
    alias_collisions = [channel for channel in channels if len(channel["aliases"]) > 1]
    output = {
        "schema": "chemaware_observation_channel_registry_v2",
        "scientific_type": "spectrum_observation_dictionary_not_mechanistic_rule_library",
        "task_domain": {"ion_mode": "positive", "adducts": ["[M+H]+", "[M+Na]+"]},
        "channels": channels,
        "contracts": {
            "mass_degenerate_aliases_merged": True,
            "alias_count_does_not_increase_channel_weight": True,
            "negative_mode_records_excluded": True,
            "unknown_mode_records_excluded": True,
            "parent_structure_prerequisite_required_for_mechanistic_claim": True,
            "mechanistic_rules_ready": False,
            "empirical_calibration_required_before_training": True,
        },
        "provenance": {
            "legacy_library": str(args.legacy_library),
            "legacy_library_sha256": sha256(args.legacy_library),
            "builder_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    report = {
        "status": "CHEMAWARE_OBSERVATION_REGISTRY_SCHEMA_PASS",
        "formal_training_authorized": False,
        "legacy": {
            "declared_rules": int(legacy["n_rules"]),
            "kernel_consumable_nl_cf": sum(rule.get("category") in {"NL", "CF"} for rule in rules),
            "mechanistic_rules_with_executable_parent_prerequisite": 0,
            "mass_degenerate_groups_that_were_separate_kernel_dimensions": len(legacy_collisions),
            "mass_degenerate_group_members": sum(map(len, legacy_collisions)),
            "records_by_evidence_class": dict(Counter(evidence_class(rule) for rule in rules)),
            "records_by_mode": dict(Counter(str(rule.get("mode", "missing")) for rule in rules)),
        },
        "registry": {
            "positive_curated_records_before_mass_merge": len(candidates),
            "canonical_observation_channels": len(channels),
            "neutral_loss_channels": sum(channel["category"] == "NL" for channel in channels),
            "fragment_channels": sum(channel["category"] == "CF" for channel in channels),
            "mass_degenerate_alias_groups": len(alias_collisions),
            "quarantined_records": len(quarantined),
        },
        "quarantine_reason_counts": dict(Counter(
            reason for record in quarantined for reason in record["reasons"]
        )),
        "legacy_mass_degenerate_aliases": legacy_collisions,
        "gates": {
            "uniform_schema": True, "positive_mode_enforced": True,
            "mass_aliases_merged": True, "provenance_retained": True,
            "mechanistic_claim_allowed": False, "formal_training_allowed": False,
        },
        "artifacts": {"registry": str(args.output), "registry_sha256": sha256(args.output)},
        "claim_limit": (
            "The registry is ready for train-formula empirical channel calibration. It is not a "
            "validated mechanistic rule library and cannot authorize model training by itself."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Post-hoc match validated data-driven factors to the existing rule library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--core-rules", type=Path, required=True)
    parser.add_argument("--massbank-rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def load_rules(path: Path, library: str) -> tuple[list[dict], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for index, rule in enumerate(data.get("rules", [])):
        value = rule.get("value")
        if isinstance(value, (int, float)):
            output.append({
                "library": library,
                "index": index,
                "name": rule.get("name"),
                "category": rule.get("category"),
                "match_type": rule.get("match_type"),
                "value": float(value),
                "source": rule.get("source", ""),
                "formula": rule.get("formula"),
                "support_count": rule.get("support_count"),
            })
    return output, len(data.get("rules", []))


def main() -> None:
    args = parse_args()
    localization = json.loads(args.localization.read_text(encoding="utf-8"))
    core_rules, core_total = load_rules(args.core_rules, "core")
    massbank_rules, massbank_total = load_rules(args.massbank_rules, "massbank")
    rules = core_rules + massbank_rules
    validated = []
    for factor in localization["factors"]:
        factor_id = int(factor["factor"])
        for kind, result in factor["confirmation"].items():
            if not result.get("localization_pass_bh"):
                continue
            expected_type = "peak_mz" if kind == "fragment_mz" else "mass_diff"
            mass = float(result["fixed_mass_da"])
            matches = []
            for rule in rules:
                if rule["match_type"] != expected_type:
                    continue
                delta = abs(rule["value"] - mass)
                if delta <= args.tolerance:
                    matches.append({**rule, "absolute_mass_error_da": delta})
            matches.sort(
                key=lambda item: (
                    item["absolute_mass_error_da"],
                    item["library"] != "core",
                    item["name"] or "",
                )
            )
            validated.append({
                "factor": factor_id,
                "spectral_kind": kind,
                "mass_da": mass,
                "peak_localization_q": result[
                    "bh_q_across_confirmation_candidates"
                ],
                "rule_matches": matches,
                "matched_core_rules": sum(item["library"] == "core" for item in matches),
                "matched_massbank_rules": sum(item["library"] == "massbank" for item in matches),
                "rule_library_gap": len(matches) == 0,
            })
    report = {
        "status": "post_hoc_rule_coverage_of_validated_factors",
        "discovery_rule_labels_used": False,
        "tolerance_da": args.tolerance,
        "full_rule_records": core_total + massbank_total,
        "scalar_exact_mass_rules_audited": len(rules),
        "non_scalar_rules_excluded_from_exact_mass_lookup": (
            core_total + massbank_total - len(rules)
        ),
        "validated_factors": validated,
        "summary": {
            "validated_factors": len(validated),
            "matched_by_any_rule": sum(not item["rule_library_gap"] for item in validated),
            "matched_by_core_rule": sum(item["matched_core_rules"] > 0 for item in validated),
            "matched_by_massbank_rule": sum(item["matched_massbank_rules"] > 0 for item in validated),
            "unmatched_rule_gaps": sum(item["rule_library_gap"] for item in validated),
        },
        "claim_limit": (
            "Mass agreement is only a coverage lookup. It does not identify a "
            "unique formula, substructure, or fragmentation mechanism."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Independent fail-closed validator for BioAware B3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = report["candidate_protocol"]
    if report["formal"]:
        expected = {
            "queries": 1426,
            "candidate_rows": 5384,
            "rotation_rows": 37688,
            "truth_identities": 361,
            "truth_formulas": 310,
            "sources": 4,
            "units": 8,
        }
        for key, value in expected.items():
            if int(protocol[key]) != value:
                raise RuntimeError(f"formal {key} changed: {protocol[key]} != {value}")
    if set(report["recipes"]) != {
        "catalog_opportunity", "matched_random_coabundance", "reaction_coabundance"
    }:
        raise RuntimeError("recipe set changed")
    contracts = report["contracts"]
    required = (
        "direct_reactions_only", "seed_identity_held_out_by_rotation",
        "controls_match_polarity_degree_abundance_mean_and_spread",
        "held_source_truth_identity_and_formula_purged",
        "query_abundance_is_observed_feature_only", "candidate_features_truth_blind",
        "negative_only_action", "positive_queries_untouched",
    )
    if not all(contracts[name] for name in required):
        raise RuntimeError("scientific contract failed")
    if contracts["P2b_used"] or contracts["phenotype_used"] or contracts["shared_embedding_changed"]:
        raise RuntimeError("forbidden dependency or claim detected")
    positive = report["recipes"]["reaction_coabundance"]["by_polarity"]["positive"]
    if int(positive["interventions"]) != 0 or int(positive["corrected"]) != 0 or int(positive["introduced"]) != 0:
        raise RuntimeError("positive safety stratum was modified")
    print(
        "[validate_bioaware_b3_reaction_coabundance_action] PASS "
        f"queries={protocol['queries']:,} scientific_pass={report['pass_to_context_model']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

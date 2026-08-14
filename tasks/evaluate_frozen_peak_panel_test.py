"""One-shot evaluation of the frozen chemical peak panel on test formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from attribute_large_failure_peaks import expand_peaks, load_rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-evidence", type=Path, default=Path("data/validation/large_test_failure_peak_localization/test_peak_evidence.csv"))
    parser.add_argument("--frozen-panel", type=Path, default=Path("data/validation/large_failure_peak_evidence_strata/frozen_test_panel.csv"))
    parser.add_argument("--rules", type=Path, default=Path("dreams/models/chem_aware/chem_rules_data.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_test_frozen_peak_panel"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.peak_evidence)
    frozen = pd.read_csv(args.frozen_panel)
    rules = load_rules(args.rules)
    peaks, features = expand_peaks(source, rules, args.tolerance)
    features = features.loc[features["feature_id"].isin(frozen["feature_id"])].copy()
    features.to_csv(args.output_dir / "test_frozen_panel_hits.csv", index=False)

    count_table = peaks.groupby(["query_index", "evidence"]).size().unstack(fill_value=0)
    eligible = count_table.index[(count_table.get("identity", 0) > 0) & (count_table.get("confounder", 0) > 0)]
    hits = features.loc[features["query_index"].isin(eligible)].drop_duplicates(
        ["query_index", "evidence", "feature_id"]
    )
    hit_set = set(zip(hits["query_index"], hits["evidence"], hits["feature_id"]))
    rows = []
    for feature_id in frozen["feature_id"]:
        identity = np.asarray([(query, "identity", feature_id) in hit_set for query in eligible], bool)
        confounder = np.asarray([(query, "confounder", feature_id) in hit_set for query in eligible], bool)
        identity_only = int(np.sum(identity & ~confounder))
        confounder_only = int(np.sum(~identity & confounder))
        discordant = identity_only + confounder_only
        p_value = float(binomtest(min(identity_only, confounder_only), discordant, 0.5).pvalue) if discordant else 1.0
        rows.append({
            "feature_id": feature_id, "eligible_queries": len(eligible),
            "identity_query_fraction": float(identity.mean()),
            "confounder_query_fraction": float(confounder.mean()),
            "identity_minus_confounder": float(identity.mean() - confounder.mean()),
            "identity_only_queries": identity_only, "confounder_only_queries": confounder_only,
            "discordant_queries": discordant, "exact_p": p_value,
            "direction_replicated": bool(identity.mean() < confounder.mean()),
            "nominal_p05": bool(p_value <= 0.05),
        })
    results = frozen[["feature_id", "feature_name", "feature_family"]].merge(
        pd.DataFrame(rows), on="feature_id", validate="one_to_one"
    )
    results.to_csv(args.output_dir / "test_panel_results.csv", index=False)
    report = {
        "status": "one_shot_frozen_peak_panel_test",
        "panel_frozen_before_test": bool(frozen["frozen_before_test"].all()),
        "panel_size": len(frozen), "test_failure_queries": len(source),
        "test_failure_molecules": int(source["ik14"].nunique()),
        "test_failure_formulas": int(source["formula"].nunique()),
        "paired_queries_with_both_peak_types": len(eligible),
        "features_direction_replicated": int(results["direction_replicated"].sum()),
        "features_nominal_p05": int(results["nominal_p05"].sum()),
        "joint_direction_success": bool(results["direction_replicated"].all()),
        "no_test_driven_selection": True,
        "claim_limit": "Frozen numerical rule concepts are hypotheses, not structure-confirmed fragment identities.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

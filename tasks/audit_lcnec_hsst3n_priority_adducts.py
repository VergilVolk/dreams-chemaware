"""Audit obvious coeluting isotope/adduct alternatives for frozen LCNEC priorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPACINGS = {
    "C13_isotope": 1.003355,
    "Na_minus_H_exchange": 21.981943,
    "chloride_vs_deprotonated": 35.976129,
    "formate_vs_deprotonated": 46.005477,
    "acetate_vs_deprotonated": 60.021127,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-ledger", type=Path, default=Path("data/validation/lcnec_hsst3n_dark_robustness_gate/normalization_robustness_ledger.csv"))
    parser.add_argument("--module-ledger", type=Path, default=Path("data/validation/lcnec_hsst3n_dark_robustness_gate/nonredundant_module_membership.csv"))
    parser.add_argument("--priority-ledger", type=Path, default=Path("data/validation/lcnec_hsst3n_priority_structure/priority_structure_ledger.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lcnec_hsst3n_priority_adduct_audit"))
    parser.add_argument("--rt-window-sec", type=float, default=5.0)
    parser.add_argument("--spacing-tolerance-da", type=float, default=0.01)
    args = parser.parse_args()

    features = pd.read_csv(args.feature_ledger)
    modules = pd.read_csv(args.module_ledger)
    priorities = pd.read_csv(args.priority_ledger)
    features = features.loc[features["quality_pass"].astype(bool)].copy()
    module_sizes = modules.groupby("family_id")["module_size"].first().to_dict()

    rows: list[dict[str, object]] = []
    for priority in priorities.itertuples():
        coeluting = features.loc[
            (features["family_id"] != priority.family_id)
            & ((features["rt_sec"] - priority.target_rt_sec).abs() <= args.rt_window_sec)
        ].copy()
        spacing_flags: list[str] = []
        nearest_spacing_error = np.inf
        nearest_spacing_label = ""
        nearest_family = None
        for candidate in coeluting.itertuples():
            gap = abs(float(candidate.mz) - float(priority.target_mz))
            for label, expected in SPACINGS.items():
                error = abs(gap - expected)
                if error < nearest_spacing_error:
                    nearest_spacing_error = error
                    nearest_spacing_label = label
                    nearest_family = int(candidate.family_id)
                if error <= args.spacing_tolerance_da:
                    spacing_flags.append(f"{label}:family_{int(candidate.family_id)}")
        rows.append(
            {
                "family_id": int(priority.family_id),
                "priority_name": priority.priority_name,
                "target_mz": float(priority.target_mz),
                "target_rt_sec": float(priority.target_rt_sec),
                "quality_pass_coeluting_families_5sec": int(len(coeluting)),
                "frozen_module_size": int(module_sizes.get(priority.family_id, -1)),
                "common_spacing_flags": ";".join(sorted(set(spacing_flags))),
                "common_spacing_flag_count": int(len(set(spacing_flags))),
                "nearest_common_spacing_label": nearest_spacing_label,
                "nearest_common_spacing_family": nearest_family,
                "nearest_common_spacing_error_da": None if not np.isfinite(nearest_spacing_error) else float(nearest_spacing_error),
            }
        )

    ledger = pd.DataFrame(rows)
    report = {
        "status": "lcnec_hsst3n_priority_adduct_audit_complete",
        "formal": True,
        "hypotheses": int(len(ledger)),
        "singleton_modules": int((ledger["frozen_module_size"] == 1).sum()),
        "hypotheses_with_common_spacing_flags": int((ledger["common_spacing_flag_count"] > 0).sum()),
        "rows": rows,
        "decision": "A zero flag count excludes obvious alternatives only within the quality-passed family ledger and frozen RT/mass-spacing rules. It is not an adduct assignment or authentic-standard confirmation.",
        "claim_limit": "Does not inspect unpicked raw ions, in-source multimers outside the predefined spacings, ion mobility, or authentic-standard retention time.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output_dir / "priority_adduct_audit.csv", index=False)
    (args.output_dir / "priority_adduct_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

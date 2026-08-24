"""Create a biology-ready candidate table after technical ion-level gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def best_isotope_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["feature_id"])
    plausible = frame[frame.plausible_isotope_support.astype(bool)].copy()
    rows = []
    for feature_id, group in frame.groupby("feature_id"):
        supported = plausible[plausible.feature_id == feature_id]
        low = supported[supported.charge_hypothesis <= 3]
        high = supported[supported.charge_hypothesis >= 4]
        best = supported.sort_values("median_trace_correlation", ascending=False).head(1)
        rows.append(
            {
                "feature_id": int(feature_id),
                "small_charge_isotope_support": bool(len(low)),
                "high_charge_isotope_support": bool(len(high)),
                "best_charge_hypothesis": int(best.charge_hypothesis.iloc[0]) if len(best) else np.nan,
                "best_isotope_trace_correlation": (
                    float(best.median_trace_correlation.iloc[0]) if len(best) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def panel_table(
    root: Path,
    panel: str,
    paired_dir: Path | None = None,
    audit_dir: Path | None = None,
    ion_family_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    paired = paired_dir or (root / "ms1_paired_analysis")
    audit = audit_dir or (root / "discovery_candidate_audit")
    priority = pd.read_csv(paired / f"{panel}__discovery_priority_features.csv")
    quality = pd.read_csv(audit / f"{panel}__candidate_quality_summary.csv")
    isotope = best_isotope_summary(load_optional(audit / f"{panel}__candidate_isotope_summary.csv"))
    frame = priority.merge(quality.drop(columns=["mz", "rt_sec"], errors="ignore"), on="feature_id", how="left")
    frame = frame.merge(isotope, on="feature_id", how="left")

    families_root = ion_family_dir or (root / "ion_families")
    families = load_optional(families_root / f"{panel}__candidate_ion_families.csv")
    if not families.empty:
        family_columns = [
            "feature_id",
            "family_id",
            "family_size",
            "family_mz",
            "family_rt_sec",
            "inferred_charge",
            "dc_charge_adducts",
            "adducts",
        ]
        frame = frame.merge(families[[column for column in family_columns if column in families]], on="feature_id", how="left")
    else:
        frame["family_id"] = frame.feature_id
        frame["family_size"] = 1

    frame.insert(0, "panel", panel)
    reject_reasons: list[str] = []
    for row in frame.itertuples(index=False):
        reasons = []
        if bool(getattr(row, "chromatography_edge_flag", False)):
            reasons.append("chromatography_edge")
        if bool(getattr(row, "rt_instability_flag", False)):
            reasons.append("rt_instability")
        if bool(getattr(row, "global_run_order_flag", False)):
            reasons.append("global_run_order")
        delta_p = getattr(row, "rmu_delta_order_p", np.nan)
        if np.isfinite(delta_p) and delta_p < 0.05:
            reasons.append("paired_delta_run_order")
        if getattr(row, "rmu_n_pairs", 0) < 8:
            reasons.append("fewer_than_8_pairs")
        low_charge = bool(getattr(row, "small_charge_isotope_support", False))
        high_charge = bool(getattr(row, "high_charge_isotope_support", False))
        if high_charge and not low_charge:
            reasons.append("high_charge_only")
        reject_reasons.append(";".join(reasons))
    frame["technical_reject_reasons"] = reject_reasons
    frame["passes_technical_gate"] = frame.technical_reject_reasons.eq("")
    low_support = frame.small_charge_isotope_support.fillna(False)
    high_support = frame.high_charge_isotope_support.fillna(False)
    frame["ion_evidence_class"] = np.select(
        [low_support & high_support, low_support, high_support],
        ["ambiguous_charge_support", "small_charge_isotope_supported", "high_charge_isotope_supported"],
        default="no_isotope_support",
    )

    # One representative per OpenMS ion family.  Detection and interaction
    # evidence are used only to choose among already admissible members.
    frame["family_representative"] = False
    admissible = frame[frame.passes_technical_gate].copy()
    if len(admissible):
        admissible["_interaction_p"] = admissible.max_interaction_p_across_normalizations.fillna(1.0)
        admissible["_detection"] = admissible.n_detected.fillna(0)
        ordered = admissible.sort_values(
            ["family_id", "_interaction_p", "_detection", "min_abs_rmu_log2fc"],
            ascending=[True, True, False, False],
        )
        representatives = ordered.drop_duplicates("family_id").feature_id
        frame.loc[frame.feature_id.isin(representatives), "family_representative"] = True
    frame["send_to_annotation"] = frame.passes_technical_gate & frame.family_representative

    counts = {
        "initial_priority_ions": int(len(frame)),
        "pass_technical_gate": int(frame.passes_technical_gate.sum()),
        "family_representatives_for_annotation": int(frame.send_to_annotation.sum()),
        "high_charge_only_rejected": int(frame.technical_reject_reasons.str.contains("high_charge_only").sum()),
        "run_order_rejected": int(frame.technical_reject_reasons.str.contains("run_order").sum()),
    }
    return frame, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/mtbls13729"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--paired-dir", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--ion-family-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/biology_candidates"))
    args = parser.parse_args()

    outputs = []
    report = {"status": "complete", "panels": {}}
    for panel in args.panels:
        frame, counts = panel_table(
            args.root,
            panel,
            paired_dir=args.paired_dir,
            audit_dir=args.audit_dir,
            ion_family_dir=args.ion_family_dir,
        )
        outputs.append(frame)
        report["panels"][panel] = counts

    combined = pd.concat(outputs, ignore_index=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "all_candidates_with_gates.csv"
    annotation_path = out / "candidates_for_ms2_annotation.csv"
    combined.to_csv(full_path, index=False)
    combined[combined.send_to_annotation].to_csv(annotation_path, index=False)
    report["all_candidates"] = str(full_path)
    report["annotation_candidates"] = str(annotation_path)
    report["interpretation_limit"] = (
        "Technical gating and ion-family selection do not establish metabolite identity or biological causality."
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

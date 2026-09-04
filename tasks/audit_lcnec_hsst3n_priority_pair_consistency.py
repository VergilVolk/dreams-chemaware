"""Patient-pair consistency audit for LCNEC priority structure hypotheses."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel, wilcoxon


PRIORITY_FAMILY = {
    104: "adenosine_diphosphate_family",
    102: "adenosine_diphosphoribose_family",
    109: "ascorbate",
    169: "quinolinate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", type=Path,
        default=Path("data/validation/lcnec_hsst3n_dark_eic_gate/dark_feature_eic_matrix.npz"),
    )
    parser.add_argument(
        "--overview", type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/lcnec_hsst3n_priority_pair_consistency"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.matrix)
    area = cache["area"].astype(float)
    family_ids = cache["family_id"].astype(int)
    sample_ids = cache["sample_id"].astype(str)
    overview = pd.read_csv(args.overview, sep="\t")
    study = overview[overview["NOTE"].eq("Study sample")].copy()
    sample_row = {value: index for index, value in enumerate(sample_ids)}
    pair_map: dict[str, dict[str, int]] = defaultdict(dict)
    amount: dict[int, float] = {}
    for _, record in study.iterrows():
        matrix_row = sample_row[str(record["SAMPLE_ID"])]
        pair_map[str(record["SAMPLE_CODE"])][str(record["GROUP_CODE"])] = matrix_row
        amount[matrix_row] = float(record["AMOUNT"])
    if len(pair_map) != 34 or any(set(value) != {"TU", "NG"} for value in pair_map.values()):
        raise RuntimeError("expected 34 complete tumor/normal pairs")

    rows = []
    patient_rows = []
    for family_id, name in PRIORITY_FAMILY.items():
        positions = np.flatnonzero(family_ids == family_id)
        if len(positions) != 1:
            raise RuntimeError(f"family {family_id} absent or duplicated")
        target = int(positions[0])
        raw_tu = np.asarray([area[pair["TU"], target] for pair in pair_map.values()])
        raw_ng = np.asarray([area[pair["NG"], target] for pair in pair_map.values()])
        mg_tu = np.asarray([area[pair["TU"], target] / amount[pair["TU"]] for pair in pair_map.values()])
        mg_ng = np.asarray([area[pair["NG"], target] / amount[pair["NG"]] for pair in pair_map.values()])
        positive = np.r_[mg_tu[mg_tu > 0], mg_ng[mg_ng > 0]]
        pseudo = float(positive.min() / 2) if len(positive) else 1.0
        delta = np.log2(mg_tu + pseudo) - np.log2(mg_ng + pseudo)
        expected_positive = float(delta.mean()) > 0
        concordant = delta > 0 if expected_positive else delta < 0
        lop_mean = np.asarray([np.delete(delta, index).mean() for index in range(len(delta))])
        try:
            wilcoxon_p = float(wilcoxon(delta).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append({
            "family_id": family_id, "priority_name": name,
            "pairs": len(delta), "mean_per_mg_log2fc": float(delta.mean()),
            "median_per_mg_log2fc": float(np.median(delta)),
            "concordant_pairs": int(concordant.sum()),
            "direction_concordance": float(concordant.mean()),
            "two_sided_sign_test_p": float(binomtest(int(concordant.sum()), len(delta), 0.5).pvalue),
            "paired_t_p": float(ttest_rel(np.log2(mg_tu + pseudo), np.log2(mg_ng + pseudo)).pvalue),
            "wilcoxon_p": wilcoxon_p,
            "leave_one_pair_out_effect_min": float(lop_mean.min()),
            "leave_one_pair_out_effect_max": float(lop_mean.max()),
            "leave_one_pair_out_sign_stable": bool(np.all(lop_mean > 0) if expected_positive else np.all(lop_mean < 0)),
        })
        for patient, value in zip(pair_map, delta, strict=True):
            patient_rows.append({
                "family_id": family_id, "priority_name": name,
                "patient_code": patient, "per_mg_log2fc_tumor_vs_normal": float(value),
            })
    ledger = pd.DataFrame(rows)
    report = {
        "status": "lcnec_hsst3n_priority_pair_consistency_complete",
        "formal": True,
        "pairs": 34,
        "hypotheses": len(ledger),
        "all_leave_one_pair_out_sign_stable": bool(ledger["leave_one_pair_out_sign_stable"].all()),
        "all_direction_concordance_ge_0_65": bool(ledger["direction_concordance"].ge(0.65).all()),
        "all_wilcoxon_p_le_0_05": bool(ledger["wilcoxon_p"].le(0.05).all()),
        "rows": ledger.to_dict("records"),
        "claim_limit": "Paired abundance robustness only; no structure, flux, or causal mechanism claim.",
    }
    report["pass"] = all([
        report["all_leave_one_pair_out_sign_stable"],
        report["all_direction_concordance_ge_0_65"],
        report["all_wilcoxon_p_le_0_05"],
    ])
    ledger.to_csv(args.output_dir / "pair_consistency_ledger.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(args.output_dir / "per_patient_effects.csv", index=False)
    (args.output_dir / "pair_consistency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

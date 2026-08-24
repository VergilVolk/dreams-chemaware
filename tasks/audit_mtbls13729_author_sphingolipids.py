"""Re-audit the sphingolipid evidence reported in the MTBLS13729 MAF files.

This uses the authors' own abundance table and patient-matched tumour/normal
contrasts.  It is a robustness audit of the published exploratory hypothesis,
not a replacement for raw-data re-quantification or authentic standards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAF_DIR = ROOT / "_mtbls13729_meta"
DEFAULT_OUT = ROOT / "data/mtbls13729/biology_candidates/author_sphingolipid_audit.csv"


def paired_log2_deltas(row: pd.Series, ids: range, tumour_label: str) -> np.ndarray:
    values: list[float] = []
    for patient_id in ids:
        tumour = pd.to_numeric(pd.Series([row.get(f"P{patient_id:02d}-{tumour_label}")]), errors="coerce").iloc[0]
        normal = pd.to_numeric(pd.Series([row.get(f"P{patient_id:02d}-RN")]), errors="coerce").iloc[0]
        if pd.notna(tumour) and pd.notna(normal) and tumour > 0 and normal > 0:
            values.append(float(np.log2(tumour / normal)))
    return np.asarray(values, dtype=float)


def safe_wilcoxon(values: np.ndarray) -> float:
    if len(values) < 3 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values without an extra dependency."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--maf-dir", type=Path, default=DEFAULT_MAF_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    rows: list[dict] = []
    pattern = r"sphing|ceramide|cer\(|hexcer|lactosyl|glucosyl|galactosyl"
    for path in sorted(args.maf_dir.glob("*maf.tsv")):
        maf = pd.read_csv(path, sep="\t")
        names = maf["metabolite_identification"].fillna("").astype(str)
        for _, row in maf.loc[names.str.contains(pattern, case=False, regex=True)].iterrows():
            rmu = paired_log2_deltas(row, range(21, 31), "Rmu")
            rtu = paired_log2_deltas(row, range(11, 21), "Rtu")
            rows.append({
                "maf_file": path.name,
                "database_identifier": row.get("database_identifier", ""),
                "metabolite_identification": row["metabolite_identification"],
                "mz": row["mass_to_charge"],
                "rt_min": row["retention_time"],
                "rmu_n_pairs": len(rmu),
                "rmu_mean_log2fc": np.mean(rmu) if len(rmu) else np.nan,
                "rmu_ttest_p": ttest_1samp(rmu, 0).pvalue if len(rmu) >= 3 else np.nan,
                "rmu_wilcoxon_p": safe_wilcoxon(rmu),
                "rtu_n_pairs": len(rtu),
                "rtu_mean_log2fc": np.mean(rtu) if len(rtu) else np.nan,
                "rtu_ttest_p": ttest_1samp(rtu, 0).pvalue if len(rtu) >= 3 else np.nan,
                "rtu_wilcoxon_p": safe_wilcoxon(rtu),
                "interaction_welch_p": (
                    ttest_ind(rmu, rtu, equal_var=False).pvalue if len(rmu) >= 3 and len(rtu) >= 3 else np.nan
                ),
                "interaction_mannwhitney_p": (
                    mannwhitneyu(rmu, rtu, alternative="two-sided").pvalue
                    if len(rmu) >= 3 and len(rtu) >= 3 else np.nan
                ),
            })

    out = pd.DataFrame(rows)
    for p_column, q_column in [
        ("rmu_ttest_p", "rmu_ttest_bh_q"),
        ("rmu_wilcoxon_p", "rmu_wilcoxon_bh_q"),
        ("interaction_welch_p", "interaction_welch_bh_q"),
        ("interaction_mannwhitney_p", "interaction_mannwhitney_bh_q"),
    ]:
        valid = out[p_column].notna()
        out[q_column] = np.nan
        if valid.any():
            out.loc[valid, q_column] = benjamini_hochberg(out.loc[valid, p_column].to_numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    report = {
        "n_author_sphingolipid_entries": int(len(out)),
        "n_rmu_nominal_ttest_p_lt_0_05": int((out.rmu_ttest_p < 0.05).sum()),
        "n_rmu_bh_q_lt_0_05": int((out.rmu_ttest_bh_q < 0.05).sum()),
        "n_interaction_nominal_welch_p_lt_0_05": int((out.interaction_welch_p < 0.05).sum()),
        "n_interaction_bh_q_lt_0_05": int((out.interaction_welch_bh_q < 0.05).sum()),
        "interpretation_limit": (
            "Authors' MAF abundances support an exploratory re-audit only; raw-data re-quantification, "
            "MS2 identity and authentic standards remain required. Static tissue abundances do not prove flux."
        ),
        "output": str(args.out),
    }
    args.out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(out[[
        "metabolite_identification", "rmu_mean_log2fc", "rmu_ttest_p",
        "rmu_ttest_bh_q", "rtu_mean_log2fc", "interaction_welch_p",
        "interaction_welch_bh_q"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Independent paired-tissue support for MTBLS13729 biology axes.

ST001087 contains 17 paired colorectal tumour/adjacent tissues measured in
positive and negative LC-MS.  Its compound annotations are formula/database
assignments (mostly FindByFormula), so this analysis evaluates abundance-axis
replication, not structural confirmation.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/mtbls13729/ST001087_AN001772_Results.txt"
ADDITIONAL = ROOT / "data/mtbls13729/ST001087_AN001772_Results_Additional_info.txt"
FACTORS = ROOT / "data/mtbls13729/external_st001087_factors.json"
OUT = ROOT / "data/mtbls13729/external_st001087_axis_validation_v1"

TARGETS = {
    "modified_guanosine": ["2-Methylguanosine", "N2,N2-Dimethylguanosine"],
    "methyl_donor": ["SAH / S-Adenosyl-L-homocysteine"],
    "acetylated_polyamine": [
        "N1,N8-Diacetylspermidine",
        "N1,N12-Diacetylspermine",
        "N1-Acetylspermidine",
    ],
    "polyamine_pool": ["Spermidine", "Spermine"],
    "tryptophan_kynurenine": ["D-Tryptophan", "Kynurenine"],
}

SAMPLE_RE = re.compile(r"^([np])(\d+)([NT])([LR])$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    positives = int((values > 0).sum())
    nonzero = int((values != 0).sum())
    if len(values):
        try:
            wp = float(wilcoxon(values, zero_method="wilcox").pvalue)
        except ValueError:
            wp = 1.0
    else:
        wp = np.nan
    return {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean_log2fc": float(np.mean(values)) if len(values) else np.nan,
        f"{prefix}_median_log2fc": float(np.median(values)) if len(values) else np.nan,
        f"{prefix}_positive_fraction": float(positives / len(values)) if len(values) else np.nan,
        f"{prefix}_sign_p": float(binomtest(positives, nonzero, 0.5).pvalue) if nonzero else 1.0,
        f"{prefix}_ttest_p": float(ttest_1samp(values, 0).pvalue) if len(values) >= 2 and np.std(values) > 0 else 1.0,
        f"{prefix}_wilcoxon_p": wp,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RESULTS, sep="\t")
    data = data.rename(columns={data.columns[0]: "metabolite"})
    # Repository export contains legacy Windows-1252 symbols in annotations.
    additional = pd.read_csv(ADDITIONAL, sep="\t", encoding="cp1252")
    additional = additional.rename(columns={additional.columns[0]: "metabolite"})
    annotation_meta = additional[["metabolite", "CompoundAlgo", "Annotations"]].drop_duplicates("metabolite")
    data = data.merge(annotation_meta, on="metabolite", how="left", validate="many_to_one")

    sample_columns = [column for column in data.columns if SAMPLE_RE.match(str(column))]
    sample_meta = []
    for column in sample_columns:
        mode, patient, tissue, side = SAMPLE_RE.match(column).groups()
        sample_meta.append(
            {
                "sample": column,
                "mode": mode,
                "patient": patient,
                "tissue": "tumor" if tissue == "T" else "normal",
                "side": side,
            }
        )
    sample_meta = pd.DataFrame(sample_meta)

    rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    requested = {name for names in TARGETS.values() for name in names}
    subset = data.loc[data["metabolite"].isin(requested)].copy()
    if set(subset["metabolite"]) != requested:
        missing = sorted(requested - set(subset["metabolite"]))
        raise RuntimeError(f"ST001087 target rows missing: {missing}")

    for axis, names in TARGETS.items():
        for name in names:
            row = subset.loc[subset["metabolite"].eq(name)].iloc[0]
            mode_symbol = str(row["Ionization mode"]).strip()
            mode = "p" if mode_symbol == "+" else "n"
            meta = sample_meta.loc[sample_meta["mode"].eq(mode)]
            values = pd.to_numeric(row[meta["sample"]], errors="coerce")
            values.index = meta["sample"].to_numpy()
            positives = values[values > 0]
            pseudo = float(positives.min() / 2) if len(positives) else 1.0

            deltas_all, deltas_complete = [], []
            tumor_only = 0
            normal_only = 0
            for patient, group in meta.groupby("patient"):
                normal = group.loc[group["tissue"].eq("normal")]
                tumor = group.loc[group["tissue"].eq("tumor")]
                if len(normal) != 1 or len(tumor) != 1:
                    continue
                normal_sample = str(normal.iloc[0]["sample"])
                tumor_sample = str(tumor.iloc[0]["sample"])
                normal_value = float(values.get(normal_sample, np.nan))
                tumor_value = float(values.get(tumor_sample, np.nan))
                normal_detected = np.isfinite(normal_value) and normal_value > 0
                tumor_detected = np.isfinite(tumor_value) and tumor_value > 0
                delta_all = float(np.log2((tumor_value if tumor_detected else pseudo) + pseudo) - np.log2((normal_value if normal_detected else pseudo) + pseudo))
                deltas_all.append(delta_all)
                if normal_detected and tumor_detected:
                    delta_complete = float(np.log2(tumor_value) - np.log2(normal_value))
                    deltas_complete.append(delta_complete)
                else:
                    delta_complete = np.nan
                tumor_only += int(tumor_detected and not normal_detected)
                normal_only += int(normal_detected and not tumor_detected)
                delta_rows.append(
                    {
                        "axis": axis,
                        "metabolite": name,
                        "patient": patient,
                        "side": str(normal.iloc[0]["side"]),
                        "normal_value": normal_value,
                        "tumor_value": tumor_value,
                        "normal_detected": normal_detected,
                        "tumor_detected": tumor_detected,
                        "pseudocount_log2fc": delta_all,
                        "complete_case_log2fc": delta_complete,
                    }
                )

            record: dict[str, object] = {
                "axis": axis,
                "metabolite": name,
                "ionization_mode": mode_symbol,
                "mz": float(row["m/z"]),
                "rt_min": float(row["retention time"]),
                "compound_algorithm": str(row.get("CompoundAlgo", "")),
                "normal_detection_fraction": float(np.mean([r["normal_detected"] for r in delta_rows if r["metabolite"] == name])),
                "tumor_detection_fraction": float(np.mean([r["tumor_detected"] for r in delta_rows if r["metabolite"] == name])),
                "pseudocount": pseudo,
                "tumor_only_detected_pairs": tumor_only,
                "normal_only_detected_pairs": normal_only,
                "paired_detection_mcnemar_p": float(
                    binomtest(tumor_only, tumor_only + normal_only, 0.5).pvalue
                ) if (tumor_only + normal_only) else 1.0,
            }
            record.update(summarize(np.asarray(deltas_all, dtype=float), "all_pairs"))
            record.update(summarize(np.asarray(deltas_complete, dtype=float), "complete_pairs"))
            record["direction_concordant_all_vs_complete"] = bool(
                len(deltas_complete) >= 4
                and np.sign(np.mean(deltas_all)) == np.sign(np.mean(deltas_complete))
            )
            rows.append(record)

    result = pd.DataFrame(rows)
    deltas = pd.DataFrame(delta_rows)
    result.to_csv(OUT / "external_axis_metabolite_results.csv", index=False)
    deltas.to_csv(OUT / "external_axis_paired_deltas.csv", index=False)
    sample_meta.to_csv(OUT / "external_sample_metadata.csv", index=False)

    axis_rows = []
    for axis, table in result.groupby("axis", sort=False):
        usable = table.loc[table["direction_concordant_all_vs_complete"]]
        axis_rows.append(
            {
                "axis": axis,
                "metabolites": int(len(table)),
                "direction_robust_metabolites": int(len(usable)),
                "positive_metabolites": int(usable["all_pairs_mean_log2fc"].gt(0).sum()),
                "negative_metabolites": int(usable["all_pairs_mean_log2fc"].lt(0).sum()),
                "median_all_pairs_log2fc": float(usable["all_pairs_mean_log2fc"].median()) if len(usable) else np.nan,
            }
        )
    axes = pd.DataFrame(axis_rows)
    axes.to_csv(OUT / "external_axis_summary.csv", index=False)

    report = {
        "status": "st001087_external_biology_axis_validation_complete",
        "formal": False,
        "study": "ST001087 / AN001772",
        "paired_patients_available": int(sample_meta["patient"].nunique()),
        "target_metabolites": int(len(result)),
        "direction_concordant_all_vs_complete": int(result["direction_concordant_all_vs_complete"].sum()),
        "axis_summary": axes.to_dict(orient="records"),
        "claim_limit": (
            "Independent cohort abundance support only. ST001087 annotations are database/formula assignments, "
            "not authentic-standard confirmations; absence or missingness is not quantitative proof of flux."
        ),
        "source": {
            "study_url": "https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001087",
            "project_doi": "10.21228/M8TX0N",
            "results_sha256": sha256(RESULTS),
            "additional_sha256": sha256(ADDITIONAL),
            "factors_sha256": sha256(FACTORS),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result[["axis", "metabolite", "normal_detection_fraction", "tumor_detection_fraction", "tumor_only_detected_pairs", "normal_only_detected_pairs", "paired_detection_mcnemar_p", "all_pairs_mean_log2fc", "all_pairs_sign_p", "complete_pairs_n", "complete_pairs_mean_log2fc", "direction_concordant_all_vs_complete"]].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

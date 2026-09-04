"""Audit modified-guanosine evidence across MTBLS13729, ST001087 and GSE236696.

This script deliberately stops at formula/adduct-family and transcript-program
evidence.  It does not promote formula-only external annotations to MSI level 1,
nor infer flux or a causal writer enzyme from steady-state abundance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/crosscohort_modified_nucleoside_audit_v1"
CANDIDATES = ROOT / "data/mtbls13729/biology_closure_analysis_v1/candidate_identity_and_abundance.csv"
ST_RESULTS = ROOT / "data/mtbls13729/external_st001087_axis_validation_v1/external_axis_metabolite_results.csv"
ST_INFO = ROOT / "data/mtbls13729/ST001087_AN001772_Results_Additional_info.txt"
GSE_GENES = ROOT / "data/external/GSE236696/paired_axis_by_lineage_v3/lineage_gene_paired_results.csv"

PROTON = 1.007276466621
SODIUM = 22.989218
ELECTRON = 0.000548579909
ATOMIC = {
    "C": 12.0,
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
}

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
PEAK_RE = re.compile(r"\(([-+0-9.eE]+),\s*[-+0-9.eE]+\)")
FORMULA_IN_ANNOTATION_RE = re.compile(r"\[\s*((?:[A-Z][a-z]?\s*\d*\s*)+),")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_mass(formula: str) -> float:
    cursor = 0
    total = 0.0
    for match in FORMULA_RE.finditer(formula.replace(" ", "")):
        if match.start() != cursor:
            raise ValueError(f"cannot parse formula {formula!r}")
        element, count = match.groups()
        if element not in ATOMIC:
            raise ValueError(f"unsupported element {element} in {formula}")
        total += ATOMIC[element] * int(count or "1")
        cursor = match.end()
    if cursor != len(formula.replace(" ", "")):
        raise ValueError(f"cannot parse formula {formula!r}")
    return total


def ppm(observed: float, expected: float) -> float:
    return 1e6 * (observed - expected) / expected


def read_st_info() -> dict[str, dict[str, object]]:
    wanted = {"2-Methylguanosine", "N2,N2-Dimethylguanosine"}
    result: dict[str, dict[str, object]] = {}
    # Metabolomics Workbench exports this legacy table with Windows-1252 bytes.
    with ST_INFO.open("r", encoding="cp1252", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row["Metabolite Name"]
            if name not in wanted:
                continue
            annotation = row["Annotations"]
            match = FORMULA_IN_ANNOTATION_RE.search(annotation)
            if not match:
                raise RuntimeError(f"formula not found for ST001087 row {name}")
            formula = match.group(1).replace(" ", "")
            peaks = [float(value) for value in PEAK_RE.findall(row["CompositeSpectrum"])]
            if not peaks:
                raise RuntimeError(f"composite spectrum missing for {name}")
            result[name] = {
                "external_name": name,
                "formula": formula,
                "compound_algorithm": row["CompoundAlgo"],
                "reported_mass_field": float(row["Mass"]),
                "first_composite_peak_mz": peaks[0],
                "rt_min": float(row["Retention Time"]),
            }
    if set(result) != wanted:
        raise RuntimeError(f"missing ST001087 targets: {wanted - set(result)}")
    return result


def build_adduct_audit() -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATES)
    local = candidates[candidates["feature_id"].isin([1597, 3019, 7489, 8481])].copy()
    rows: list[dict[str, object]] = []
    for row in local.itertuples(index=False):
        adduct = "[M+Na]+" if int(row.feature_id) in {7489, 8481} else "[M+H]+"
        neutral = exact_mass(row.neutral_formula)
        expected = neutral + (SODIUM if adduct == "[M+Na]+" else PROTON)
        rows.append(
            {
                "cohort": "MTBLS13729",
                "feature_or_name": str(int(row.feature_id)),
                "reported_label": row.refined_label,
                "formula": row.neutral_formula,
                "annotation_level": "formula/adduct family; positional isomer unresolved",
                "adduct_interpretation": adduct,
                "observed_ion_mz": float(row.mz),
                "theoretical_ion_mz": expected,
                "ion_mass_error_ppm": ppm(float(row.mz), expected),
                "reported_software_mass": math.nan,
                "software_mass_interpretation": "feature m/z",
            }
        )

    st_info = read_st_info()
    st_stats = pd.read_csv(ST_RESULTS).set_index("metabolite")
    for name, info in st_info.items():
        formula = str(info["formula"])
        neutral = exact_mass(formula)
        expected_na = neutral + SODIUM
        observed = float(info["first_composite_peak_mz"])
        stat = st_stats.loc[name]
        rows.append(
            {
                "cohort": "ST001087",
                "feature_or_name": name,
                "reported_label": name,
                "formula": formula,
                "annotation_level": "FindByFormula; no authentic-standard confirmation",
                "adduct_interpretation": "[M+Na]+",
                "observed_ion_mz": observed,
                "theoretical_ion_mz": expected_na,
                "ion_mass_error_ppm": ppm(observed, expected_na),
                "reported_software_mass": float(info["reported_mass_field"]),
                "software_mass_interpretation": "approximately observed ion m/z minus proton",
                "normal_detection_fraction": float(stat.normal_detection_fraction),
                "tumor_detection_fraction": float(stat.tumor_detection_fraction),
                "all_pairs_mean_log2fc": float(stat.all_pairs_mean_log2fc),
                "all_pairs_sign_p": float(stat.all_pairs_sign_p),
                "paired_detection_mcnemar_p": float(stat.paired_detection_mcnemar_p),
            }
        )
    return pd.DataFrame(rows)


def build_gene_audit() -> pd.DataFrame:
    genes = pd.read_csv(GSE_GENES)
    targets = [
        "METTL1", "WDR4", "TRMT1", "TRMT5", "TRMT10C", "THUMPD3",
        "RNMT", "CMTR1", "CMTR2", "TGS1", "NUDT16", "DCP2",
    ]
    out = genes[(genes["lineage"] == "epithelial") & genes["gene"].isin(targets)].copy()
    out["direction_stable"] = (
        (out["tumor_higher_pairs"] >= 5)
        & (out["patient_bootstrap_ci_low"] > 0)
    )
    out["causal_writer_claim_allowed"] = False
    return out.sort_values(["direction_stable", "mean_paired_delta"], ascending=[False, False])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adducts = build_adduct_audit()
    genes = build_gene_audit()
    adducts.to_csv(OUT / "crosscohort_adduct_mass_audit.csv", index=False)
    genes.to_csv(OUT / "epithelial_modified_nucleoside_gene_audit.csv", index=False)

    dm = adducts[(adducts["formula"] == "C12H17N5O5")]
    mg = adducts[(adducts["formula"] == "C11H15N5O5")]
    report = {
        "status": "mtbls13729_crosscohort_modified_nucleoside_audit_complete",
        "formal": False,
        "formula_family_replication": {
            "methylguanosine_formula_seen_in_both_cohorts": set(mg["cohort"]) == {"MTBLS13729", "ST001087"},
            "dimethylguanosine_formula_seen_in_both_cohorts": set(dm["cohort"]) == {"MTBLS13729", "ST001087"},
            "all_reported_adduct_mass_errors_abs_ppm_lt_5": bool((adducts["ion_mass_error_ppm"].abs() < 5).all()),
        },
        "gse236696_epithelial_gene_program": {
            "genes_audited": int(len(genes)),
            "direction_stable_genes": genes.loc[genes["direction_stable"], "gene"].tolist(),
            "mettl1_mean_paired_delta": float(genes.loc[genes["gene"] == "METTL1", "mean_paired_delta"].iloc[0]),
            "mettl1_direction_stable": bool(genes.loc[genes["gene"] == "METTL1", "direction_stable"].iloc[0]),
        },
        "interpretation": (
            "MTBLS13729 and ST001087 independently contain methylated-guanosine formula families "
            "under different adduct conventions. ST001087 N2,N2-dimethylguanosine is formula-only "
            "but directionally increased in paired CRC tissue. GSE236696 supports a broader epithelial "
            "RNA-modification/nucleoside-processing program, not a proven METTL1 causal chain."
        ),
        "claim_limit": (
            "Cross-cohort formula/adduct and transcript-program support only. Positional isomers, MSI "
            "Level 1 identity, metabolic flux, and writer-enzyme causality remain unproven."
        ),
        "provenance": {
            "candidates_sha256": sha256(CANDIDATES),
            "st_results_sha256": sha256(ST_RESULTS),
            "st_additional_info_sha256": sha256(ST_INFO),
            "gse_gene_results_sha256": sha256(GSE_GENES),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the MSV000092887 matched COVID-control response object.

The public MS-DIAL mzTab contains 3,041 HILIC-positive features across 129
assays.  This builder joins the first 78 biological assays to the deposited
clinical workbook and preserves two response channels:

* abundance: mean within-pair log2 difference among pairs detected in both;
* detection: paired discordance log odds among COVID-only/control-only pairs.

Zeros are never replaced by an arbitrary intensity.  Reference-spectrum
evidence is taken only from mzTab SME links.  Features without such a link are
not called MS2-bearing until the raw mzXML linkage is completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mztab(path: Path) -> tuple[dict[int, str], pd.DataFrame, pd.DataFrame]:
    """Return assay names, small-molecule features, and spectral evidence."""
    lines = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    assays: dict[int, str] = {}
    headers: dict[str, list[str]] = {}
    rows: dict[str, list[list[str]]] = {"SML": [], "SMF": [], "SME": []}
    for line in lines:
        fields = line.split("\t")
        if fields[0] == "MTD" and len(fields) >= 3:
            match = re.fullmatch(r"assay\[(\d+)\]", fields[1])
            if match:
                assays[int(match.group(1))] = fields[2]
        elif fields[0] in {"SMH", "SFH", "SEH"}:
            headers[fields[0]] = fields
        elif fields[0] in rows:
            rows[fields[0]].append(fields)
    if set(headers) != {"SMH", "SFH", "SEH"}:
        raise ValueError(f"incomplete mzTab headers: {sorted(headers)}")
    if not assays:
        raise ValueError("no mzTab assays found")

    frames: dict[str, pd.DataFrame] = {}
    for row_type, header_type in (("SML", "SMH"), ("SMF", "SFH"), ("SME", "SEH")):
        header = headers[header_type]
        normalized = [row[: len(header)] + [""] * max(0, len(header) - len(row)) for row in rows[row_type]]
        frames[row_type] = pd.DataFrame(normalized, columns=header)
    return assays, frames["SML"], frames["SMF"], frames["SME"]


def parse_clinical_design(path: Path) -> pd.DataFrame:
    clinical = pd.read_excel(path, sheet_name=0, dtype=object)
    required = {"ID", "MS_ID_metabolomics", "Pairs_MS", "Group "}
    if missing := required - set(clinical):
        raise ValueError(f"clinical workbook lacks {sorted(missing)}")
    design = clinical[["ID", "MS_ID_metabolomics", "Pairs_MS", "Group "]].copy()
    design.columns = ["biological_unit_id", "assay_name", "pair_id", "group"]
    design = design.fillna("").astype(str).apply(lambda column: column.str.strip())
    design["group"] = design["group"].replace({"Covid_19": "covid", "Healthy": "control"})
    if set(design.group) != {"covid", "control"}:
        raise ValueError(f"unexpected groups: {sorted(set(design.group))}")
    if design.biological_unit_id.duplicated().any():
        raise ValueError("biological identifiers must be unique")
    pair_counts = design.groupby(["pair_id", "group"]).size().unstack(fill_value=0)
    if len(pair_counts) != 39 or not (pair_counts == 1).all().all():
        raise ValueError("expected exactly 39 pairs with one COVID and one control assay")
    return design.sort_values(["pair_id", "group"]).reset_index(drop=True)


def apply_frozen_metadata_correction(design: pd.DataFrame) -> pd.DataFrame:
    """Apply the sole deposited-assay correction supported by three fields."""
    corrected = design.copy()
    corrected["assay_name_deposited"] = corrected.assay_name
    corrected["metadata_correction"] = ""
    mask = (
        corrected.biological_unit_id.eq("C43")
        & corrected.assay_name.eq("21FP")
        & corrected.pair_id.eq("24")
        & corrected.group.eq("covid")
    )
    if int(mask.sum()) != 1:
        raise ValueError("frozen C43 21FP-to-24FP correction precondition failed")
    corrected.loc[mask, "assay_name"] = "24FP"
    corrected.loc[mask, "metadata_correction"] = (
        "deposited 21FP corrected to 24FP: pair_id=24, lipidomics=24FL, "
        "21FP already belongs to C31, and mzTab contains 24FP"
    )
    if corrected.assay_name.duplicated().any():
        raise ValueError("assay identifiers remain duplicated after frozen correction")
    return corrected


def paired_effects(
    abundance: np.ndarray,
    covid_indices: list[int],
    control_indices: list[int],
    minimum_both_present_pairs: int = 20,
    minimum_discordant_pairs: int = 6,
) -> pd.DataFrame:
    """Estimate continuous and detection effects without zero imputation."""
    covid = abundance[:, covid_indices]
    control = abundance[:, control_indices]
    covid_positive = np.isfinite(covid) & (covid > 0)
    control_positive = np.isfinite(control) & (control > 0)
    both = covid_positive & control_positive
    covid_only = covid_positive & ~control_positive
    control_only = control_positive & ~covid_positive
    neither = ~covid_positive & ~control_positive

    log_difference = np.full(covid.shape, np.nan, dtype=float)
    log_difference[both] = np.log2(covid[both]) - np.log2(control[both])
    n_both = both.sum(axis=1).astype(int)
    beta = np.divide(
        np.nansum(log_difference, axis=1), n_both,
        out=np.full(len(n_both), np.nan), where=n_both > 0,
    )
    centered = np.where(both, log_difference - beta[:, None], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), n_both - 1,
        out=np.full(len(n_both), np.nan), where=n_both > 1,
    )
    abundance_se = np.sqrt(np.divide(
        variance, n_both, out=np.full(len(n_both), np.nan), where=n_both > 0,
    ))

    n_covid_only = covid_only.sum(axis=1).astype(int)
    n_control_only = control_only.sum(axis=1).astype(int)
    n_discordant = n_covid_only + n_control_only
    detection_beta = np.log((n_covid_only + 0.5) / (n_control_only + 0.5))
    detection_se = np.sqrt(1.0 / (n_covid_only + 0.5) + 1.0 / (n_control_only + 0.5))

    return pd.DataFrame({
        "abundance_beta_log2_covid_vs_control": beta,
        "abundance_se": abundance_se,
        "abundance_n_both_present_pairs": n_both,
        "abundance_qc_pass": (
            (n_both >= minimum_both_present_pairs)
            & np.isfinite(beta) & np.isfinite(abundance_se) & (abundance_se > 0)
        ),
        "detection_beta_log_odds_covid_vs_control": detection_beta,
        "detection_se": detection_se,
        "detection_n_covid_only_pairs": n_covid_only,
        "detection_n_control_only_pairs": n_control_only,
        "detection_n_discordant_pairs": n_discordant,
        "detection_qc_pass": n_discordant >= minimum_discordant_pairs,
        "n_neither_detected_pairs": neither.sum(axis=1).astype(int),
    })


def build(
    mztab_path: Path,
    clinical_path: Path,
    minimum_both_present_pairs: int = 20,
    minimum_discordant_pairs: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assays, sml, smf, sme = read_mztab(mztab_path)
    design = apply_frozen_metadata_correction(parse_clinical_design(clinical_path))
    if len(smf) != 3041 or len(sml) != 3041:
        raise ValueError(f"expected 3,041 SML/SMF rows, got {len(sml)}/{len(smf)}")
    if len(sme) != 175:
        raise ValueError(f"expected 175 reference-spectrum evidence rows, got {len(sme)}")
    if len(assays) != 129:
        raise ValueError(f"expected 129 assays, got {len(assays)}")

    assay_to_number = {name: number for number, name in assays.items()}
    missing_assays = sorted(set(design.assay_name) - set(assay_to_number))
    if missing_assays:
        raise ValueError(f"clinical assays absent from mzTab: {missing_assays[:5]}")

    design = design.copy()
    design["assay_number"] = design.assay_name.map(assay_to_number).astype(int)
    design["mzml_relative_path"] = design.assay_name.map(
        lambda value: f"peak/HILIC Metabolomics -MZXML/{value}.mzXML"
    )

    feature_ids = smf["SMF_ID"].astype(str)
    if feature_ids.duplicated().any():
        raise ValueError("SMF_ID must be unique")
    sml_lookup = sml.set_index(sml["SML_ID"].astype(str))
    missing_sml = sorted(set(feature_ids) - set(sml_lookup.index))
    if missing_sml:
        raise ValueError(f"SMF rows lack SML rows: {missing_sml[:5]}")

    ordered_design = design.sort_values(["pair_id", "group"])
    sample_columns = [f"abundance_assay[{assay_to_number[name]}]" for name in ordered_design.assay_name]
    abundance = smf[sample_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    covid_indices: list[int] = []
    control_indices: list[int] = []
    for _, pair in ordered_design.groupby("pair_id", sort=True):
        control_indices.append(int(np.flatnonzero(ordered_design.index.to_numpy() == pair.loc[pair.group.eq("control")].index[0])[0]))
        covid_indices.append(int(np.flatnonzero(ordered_design.index.to_numpy() == pair.loc[pair.group.eq("covid")].index[0])[0]))

    effects = paired_effects(
        abundance, covid_indices, control_indices,
        minimum_both_present_pairs=minimum_both_present_pairs,
        minimum_discordant_pairs=minimum_discordant_pairs,
    )
    reference_link = smf["SME_ID_REFS"].astype(str).str.strip().ne("null") & smf["SME_ID_REFS"].astype(str).str.strip().ne("")
    entities = pd.DataFrame({
        "study_id": "MSV000092887",
        "contrast_id": "covid_vs_age_sex_matched_control",
        "entity_id": feature_ids,
        "precursor_mz": pd.to_numeric(smf["exp_mass_to_charge"], errors="coerce"),
        "retention_time_seconds": pd.to_numeric(smf["retention_time_in_seconds"], errors="coerce"),
        "adduct": smf["adduct_ion"].astype(str),
        "chemical_name": [str(sml_lookup.loc[value, "chemical_name"]) for value in feature_ids],
        "reference_spectrum_evidence": reference_link.to_numpy(bool),
        "identity_status": np.where(
            reference_link.to_numpy(bool),
            "reference_spectrum_hit",
            "no_reference_spectrum_hit_feature_level_ms2_membership_pending",
        ),
    })
    entities = pd.concat([entities, effects], axis=1)

    report: dict[str, object] = {
        "schema": "contrast_atlas_msv000092887_paired_response_v1",
        "status": "PAIRED_RESPONSE_BUILT_RAW_MS2_LINK_PENDING",
        "evidence_role": {
            "dataset_status": "development_consumed",
            "leakage_state": "outcome_open_public_study",
            "cross_study_discovery_claim": False,
            "external_validation_claim": False,
        },
        "denominators": {
            "features": int(len(entities)),
            "mztab_assays": int(len(assays)),
            "biological_assays": int(len(design)),
            "matched_pairs": int(design.pair_id.nunique()),
            "covid": int(design.group.eq("covid").sum()),
            "control": int(design.group.eq("control").sum()),
            "reference_spectrum_hits": int(reference_link.sum()),
            "without_reference_spectrum_hit": int((~reference_link).sum()),
            "abundance_qc_pass": int(entities.abundance_qc_pass.sum()),
            "detection_qc_pass": int(entities.detection_qc_pass.sum()),
            "either_response_channel_qc_pass": int(
                (entities.abundance_qc_pass | entities.detection_qc_pass).sum()
            ),
        },
        "published_aggregate_ms2_denominators": {
            "features_with_ms2": 1982,
            "features_with_reference_spectrum": 175,
            "implied_ms2_without_reference_spectrum": 1807,
            "feature_level_membership_available_in_mztab": False,
        },
        "response_contract": {
            "abundance": "mean paired log2(COVID/control) among pairs positive in both; no zero imputation",
            "abundance_se": "standard error across observed within-pair log2 differences",
            "abundance_minimum_pairs": int(minimum_both_present_pairs),
            "detection": "Haldane-corrected log odds of COVID-only versus control-only paired detections",
            "detection_se": "sqrt(1/(COVID-only+0.5)+1/(control-only+0.5))",
            "detection_minimum_discordant_pairs": int(minimum_discordant_pairs),
            "channel_collapse": "forbidden_before_cross_source_calibration",
        },
        "metadata_corrections": [{
            "biological_unit_id": "C43",
            "pair_id": "24",
            "field": "MS_ID_metabolomics",
            "deposited": "21FP",
            "corrected": "24FP",
            "independent_support": [
                "same-row MS_ID_lipidomics=24FL",
                "same-row Pairs_MS=24",
                "21FP already maps to C31",
                "mzTab assay inventory contains 24FP",
            ],
        }],
        "provenance": {
            "mztab_sha256": sha256(mztab_path),
            "clinical_workbook_sha256": sha256(clinical_path),
        },
        "claim_boundary": [
            "This is one real study-level response object, not a cross-study discovery.",
            "The 2,866 features without a reference-spectrum SME link are not automatically called MS2-bearing.",
            "Unresolved-MS2 status requires phenotype-blind raw mzXML linkage.",
            "The public outcome is open; this study is development/consumed, never unopened external validation.",
        ],
    }
    return entities, design, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mztab", required=True, type=Path)
    parser.add_argument("--clinical-workbook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-both-present-pairs", type=int, default=20)
    parser.add_argument("--minimum-discordant-pairs", type=int, default=6)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    entities, design, report = build(
        args.mztab, args.clinical_workbook,
        minimum_both_present_pairs=args.minimum_both_present_pairs,
        minimum_discordant_pairs=args.minimum_discordant_pairs,
    )
    args.output_dir.mkdir(parents=True)
    entities.to_csv(
        args.output_dir / "paired_response_object.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    design.to_csv(args.output_dir / "biological_assay_ledger.tsv", sep="\t", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

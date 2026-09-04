#!/usr/bin/env python
"""Fail-closed metadata preflight for the GSE178341 mucinous/NXPE1 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/GSE178341_mucinous_secretory_audit"
META = BASE / "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
CLUSTERS = BASE / "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
OUT = BASE / "metadata_preflight_v1.json"
MATCHES = BASE / "metadata_matches_v1.csv"

PURE = ("Adenocarcinoma", "Adenocarcinoma;Mucinous")
GOBLET_CODES = ("cE02", "cE06", "cE07", "cE08")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_per_patient(frame: pd.DataFrame, column: str) -> dict[str, object]:
    counts = frame.groupby("PID", observed=True)[column].nunique(dropna=False)
    inconsistent = counts[counts > 1].index.astype(str).tolist()
    return {"column": column, "inconsistent_patients": inconsistent, "pass": not inconsistent}


def stage_number(value: object) -> float:
    text = str(value).lower()
    for token, number in (("t4", 4.0), ("t3", 3.0), ("t2", 2.0), ("t1", 1.0)):
        if token in text:
            return number
    return 3.0


def patient_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, subset in frame.groupby("PID", sort=True, observed=True):
        first = subset.sort_values("PatientTypeID").iloc[0]
        processing = subset["PROCESSING_TYPE"].value_counts(normalize=True)
        chemistry = subset["SINGLECELL_TYPE"].value_counts(normalize=True)
        rows.append({
            "PID": str(patient),
            "histology": first["HistologicTypeSimple"],
            "MMRStatus": first["MMRStatus"],
            "site": first["TissueSiteSimple"],
            "team": first["TISSUE_PROCESSING_TEAM"],
            "hospital": first["SOURCE_HOSPITAL"],
            "age": float(first["Age"]),
            "stage": stage_number(first["TumorStage"]),
            "frac_sc3pv3": float(chemistry.get("SC3Pv3", 0.0)),
            "frac_unsorted": float(processing.get("unsorted", 0.0)),
            "frac_cd45pmacs": float(processing.get("CD45pMACS", 0.0)),
            "frac_livemacs": float(processing.get("LiveMACS", 0.0)),
            "frac_mixed": float(processing.get("mixUnsortCD45MACS", 0.0)),
        })
    return pd.DataFrame(rows)


def freeze_matches(profiles: pd.DataFrame) -> pd.DataFrame:
    cases = profiles[profiles["histology"].eq("Adenocarcinoma;Mucinous")].sort_values("PID").reset_index(drop=True)
    controls = profiles[profiles["histology"].eq("Adenocarcinoma")].sort_values("PID").reset_index(drop=True)
    slots = cases.loc[cases.index.repeat(3)].reset_index(drop=True)
    cost = np.full((len(slots), len(controls)), 1e9, dtype=float)
    for i, case in slots.iterrows():
        for j, control in controls.iterrows():
            if control["site"] != case["site"] or control["MMRStatus"] != case["MMRStatus"]:
                continue
            processing_l1 = sum(
                abs(float(case[column]) - float(control[column]))
                for column in ("frac_unsorted", "frac_cd45pmacs", "frac_livemacs", "frac_mixed")
            )
            value = (
                2.0 * abs(float(case["frac_sc3pv3"]) - float(control["frac_sc3pv3"]))
                + processing_l1
                + 1.0 * (case["team"] != control["team"])
                + 0.5 * (case["hospital"] != control["hospital"])
                + abs(float(case["stage"]) - float(control["stage"])) / 4.0
                + abs(float(case["age"]) - float(control["age"])) / 40.0
                + j * 1e-9
            )
            cost[i, j] = value
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(slots) or np.any(cost[rows, columns] >= 1e8):
        raise RuntimeError("unable to freeze three unique exact-site/MMR controls per mucinous patient")
    matched = []
    for row, column in zip(rows, columns):
        case = slots.iloc[row]
        control = controls.iloc[column]
        matched.append({
            "case_PID": case["PID"],
            "control_PID": control["PID"],
            "case_MMRStatus": case["MMRStatus"],
            "case_site": case["site"],
            "match_cost": float(cost[row, column]),
        })
    result = pd.DataFrame(matched).sort_values(["case_PID", "match_cost", "control_PID"]).reset_index(drop=True)
    if result["control_PID"].duplicated().any() or not (result.groupby("case_PID").size() == 3).all():
        raise RuntimeError("invalid frozen match allocation")
    return result


def main() -> None:
    for path in (META, CLUSTERS):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    meta = pd.read_csv(META)
    clusters = pd.read_csv(CLUSTERS)
    if len(meta) != 370115 or len(clusters) != 370115:
        raise RuntimeError(f"unexpected rows: metadata={len(meta)} clusters={len(clusters)}")
    if not meta["cellID"].astype(str).equals(clusters["sampleID"].astype(str)):
        raise RuntimeError("metadata cellID and cluster sampleID are not exactly aligned")
    if meta["cellID"].duplicated().any():
        raise RuntimeError("duplicated cellID")

    frame = pd.concat([meta.reset_index(drop=True), clusters.drop(columns=["sampleID"]).reset_index(drop=True)], axis=1)
    tumour = frame[frame["SPECIMEN_TYPE"].eq("T")].copy()
    pure = tumour[tumour["HistologicTypeSimple"].isin(PURE)].copy()
    profiles = patient_profiles(pure)
    matches = freeze_matches(profiles)
    matches.to_csv(MATCHES, index=False)
    patient_rows = pure.sort_values("PatientTypeID").drop_duplicates("PID")
    mucinous = pure[pure["HistologicTypeSimple"].eq("Adenocarcinoma;Mucinous")]
    adeno = pure[pure["HistologicTypeSimple"].eq("Adenocarcinoma")]
    epithelial = pure[pure["clTopLevel"].eq("Epi")]
    goblet = epithelial[epithelial["cl295v11SubShort"].isin(GOBLET_CODES)]

    biological_covariates = [
        "MMRStatus",
        "TissueSiteSimple",
        "SOURCE_HOSPITAL",
        "TISSUE_PROCESSING_TEAM",
        "TumorStage",
        "HistologicGradeSimple",
        "Sex",
        "Age",
    ]
    consistency = [scalar_per_patient(pure, column) for column in biological_covariates]
    # C130 and C171 legitimately have two tumour blocks. Biological patient-level covariates must agree.
    # PROCESSING_TYPE is a cell-stream attribute and SINGLECELL_TYPE can be mixed for a patient;
    # those are summarized as patient-level proportions instead of being forced into scalars.
    if any(not item["pass"] for item in consistency):
        raise RuntimeError(f"patient-level covariate inconsistency: {consistency}")

    duplicate_tumour_blocks = (
        pure[["PID", "PatientTypeID", "HistologicTypeSimple"]]
        .drop_duplicates()
        .groupby(["PID", "HistologicTypeSimple"], observed=True)["PatientTypeID"]
        .agg(list)
    )
    duplicate_tumour_blocks = {
        patient: blocks
        for (patient, _), blocks in duplicate_tumour_blocks.items()
        if len(blocks) > 1
    }

    muc_patient = mucinous.sort_values("PatientTypeID").drop_duplicates("PID")
    muc_epithelial = epithelial[epithelial["HistologicTypeSimple"].eq("Adenocarcinoma;Mucinous")]
    muc_goblet = goblet[goblet["HistologicTypeSimple"].eq("Adenocarcinoma;Mucinous")]
    per_mucinous = []
    for patient in sorted(muc_patient["PID"].astype(str)):
        row = muc_patient[muc_patient["PID"].astype(str).eq(patient)].iloc[0]
        patient_cells = mucinous[mucinous["PID"].astype(str).eq(patient)]
        processing = patient_cells["PROCESSING_TYPE"].value_counts(normalize=True).sort_index()
        chemistry = patient_cells["SINGLECELL_TYPE"].value_counts(normalize=True).sort_index()
        per_mucinous.append({
            "PID": patient,
            "tumour_blocks": sorted(mucinous[mucinous["PID"].astype(str).eq(patient)]["PatientTypeID"].unique().tolist()),
            "MMRStatus": row["MMRStatus"],
            "TissueSiteSimple": row["TissueSiteSimple"],
            "processing_type_fraction": {str(key): float(value) for key, value in processing.items()},
            "singlecell_type_fraction": {str(key): float(value) for key, value in chemistry.items()},
            "epithelial_cells": int((muc_epithelial["PID"].astype(str) == patient).sum()),
            "goblet_family_cells": int((muc_goblet["PID"].astype(str) == patient).sum()),
        })

    report = {
        "status": "gse178341_nxpe1_mucinous_metadata_preflight_passed",
        "formal": True,
        "cells": len(frame),
        "pure_tumour": {
            "adenocarcinoma_patients": int(adeno["PID"].nunique()),
            "adenocarcinoma_samples": int(adeno["PatientTypeID"].nunique()),
            "mucinous_patients": int(mucinous["PID"].nunique()),
            "mucinous_samples": int(mucinous["PatientTypeID"].nunique()),
            "duplicate_tumour_blocks": duplicate_tumour_blocks,
        },
        "mucinous_covariates": {
            "right_side_patients": int(muc_patient["TissueSiteSimple"].eq("right").sum()),
            "MMRd_patients": int(muc_patient["MMRStatus"].eq("MMRd").sum()),
            "MMRp_patients": int(muc_patient["MMRStatus"].eq("MMRp").sum()),
        },
        "mucinous_patient_support": per_mucinous,
        "pure_tumour_epithelial_cells": int(len(epithelial)),
        "pure_tumour_goblet_family_cells": int(len(goblet)),
        "patient_covariate_consistency": consistency,
        "metadata_blind_matching": {
            "controls_per_case": 3,
            "unique_controls": int(matches["control_PID"].nunique()),
            "exact_fields": ["TissueSiteSimple", "MMRStatus"],
            "distance_fields": [
                "SINGLECELL_TYPE fractions",
                "PROCESSING_TYPE fractions",
                "TISSUE_PROCESSING_TEAM",
                "SOURCE_HOSPITAL",
                "Age",
                "TumorStage",
            ],
            "matches_sha256": sha256(MATCHES),
        },
        "contracts": {
            "biological_unit": "PID",
            "primary_compartment": "clTopLevel=Epi",
            "goblet_sensitivity_codes": list(GOBLET_CODES),
            "pure_histology_only": list(PURE),
            "mixed_mucinous_neuroendocrine_excluded": True,
            "cell_level_inference_forbidden": True,
        },
        "provenance": {"metadata_sha256": sha256(META), "cluster_sha256": sha256(CLUSTERS)},
        "claim_limit": "Metadata-only preflight. It contains no expression result.",
    }
    expected = {
        "adenocarcinoma_patients": 53,
        "adenocarcinoma_samples": 54,
        "mucinous_patients": 6,
        "mucinous_samples": 7,
    }
    for key, value in expected.items():
        if report["pure_tumour"][key] != value:
            raise RuntimeError(f"unexpected {key}: {report['pure_tumour'][key]} != {value}")
    if report["mucinous_covariates"]["right_side_patients"] != 6:
        raise RuntimeError("not all six mucinous patients are right-sided")
    if min(item["epithelial_cells"] for item in per_mucinous) < 30:
        raise RuntimeError("a mucinous patient has fewer than 30 epithelial cells")
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

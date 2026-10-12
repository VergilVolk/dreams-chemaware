#!/usr/bin/env python
"""Phenotype-blind linkage of ST003325 feature tables to pooled-QC DDA scans."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PANELS = {
    "AN005444": {"polarity": "negative", "dda_token": "HILICneg"},
    "AN005445": {"polarity": "positive", "dda_token": "RPpos"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cv_value(element: ET.Element, accession: str) -> tuple[str | None, str | None]:
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] == "cvParam" and child.attrib.get("accession") == accession:
            return child.attrib.get("value"), child.attrib.get("unitAccession")
    return None, None


def read_ms2(path: Path, file_index: int) -> pd.DataFrame:
    rows = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "spectrum":
            continue
        level, _ = cv_value(element, "MS:1000511")
        if level == "2":
            precursor, _ = cv_value(element, "MS:1000744")
            rt, unit = cv_value(element, "MS:1000016")
            if precursor is not None and rt is not None:
                rt_sec = float(rt) * 60.0 if unit == "UO:0000031" else float(rt)
                rows.append({
                    "dda_file": file_index,
                    "scan_id": element.attrib.get("id", ""),
                    "precursor_mz": float(precursor),
                    "rt_sec": rt_sec,
                })
        element.clear()
    return pd.DataFrame(rows)


def technical_columns(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    qc = [name for name in columns if "Pool_QC_" in name and "_DDA_" not in name]
    blank = [name for name in columns if "Blank" in name]
    excluded = [name for name in columns if any(token in name for token in
                ("Pool_QC", "Blank", "Qstd", "Reagent", "reagent"))]
    study = [name for name in columns if name not in excluded]
    return qc, blank, study


def link(features: pd.DataFrame, scans: pd.DataFrame, ppm: float, rt_tolerance: float) -> pd.DataFrame:
    scans = scans.sort_values("precursor_mz").reset_index(drop=True)
    mzs = scans["precursor_mz"].to_numpy(float)
    output = []
    for row in features.itertuples(index=False):
        delta = row.mz * ppm * 1e-6
        left, right = np.searchsorted(mzs, [row.mz - delta, row.mz + delta])
        candidates = scans.iloc[left:right].copy()
        if len(candidates):
            candidates["rt_error_sec"] = np.abs(candidates["rt_sec"] - row.rt_sec)
            candidates = candidates.loc[candidates["rt_error_sec"] <= rt_tolerance]
        if candidates.empty:
            output.append({"candidate_scans": 0, "dda_replicates": 0, "best_scan_id": "",
                           "best_dda_file": -1, "best_ppm_error": np.nan,
                           "best_rt_error_sec": np.nan})
            continue
        candidates["ppm_error"] = np.abs(candidates["precursor_mz"] - row.mz) / row.mz * 1e6
        candidates["distance"] = candidates["ppm_error"] / ppm + candidates["rt_error_sec"] / rt_tolerance
        best = candidates.sort_values(["distance", "dda_file", "scan_id"]).iloc[0]
        output.append({
            "candidate_scans": int(len(candidates)),
            "dda_replicates": int(candidates["dda_file"].nunique()),
            "best_scan_id": str(best["scan_id"]),
            "best_dda_file": int(best["dda_file"]),
            "best_ppm_error": float(best["ppm_error"]),
            "best_rt_error_sec": float(best["rt_error_sec"]),
        })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--dda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=10.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    panel_reports = []
    provenance = {}
    for analysis, specification in PANELS.items():
        table_path = args.tables / f"ST003325_{analysis}_Results.txt"
        dda_paths = sorted(args.dda.rglob(f"*DDA_{specification['dda_token']}*.mzML"))
        if not table_path.is_file() or len(dda_paths) != 3:
            raise FileNotFoundError(f"{analysis}: table or three DDA files missing")
        table = pd.read_csv(table_path, sep="\t")
        parsed = table["id_number"].astype(str).str.extract(r"^([0-9.]+)_([0-9.]+)$")
        if parsed.isna().any().any():
            raise ValueError(f"{analysis}: malformed feature IDs")
        feature = pd.DataFrame({
            "analysis_id": analysis,
            "query_id": analysis + ":" + table["id_number"].astype(str),
            "mz": parsed[0].astype(float),
            "rt_sec": parsed[1].astype(float),
        })
        measurement_columns = [name for name in table.columns if name != "id_number"]
        qc_columns, blank_columns, study_columns = technical_columns(measurement_columns)
        if not qc_columns or not blank_columns or not study_columns:
            raise RuntimeError(f"{analysis}: cannot construct phenotype-blind QC groups")
        qc = table[qc_columns].to_numpy(float)
        blank = table[blank_columns].to_numpy(float)
        study = table[study_columns].to_numpy(float)
        qc_detection = np.mean(qc > 0, axis=1)
        study_detection = np.mean(study > 0, axis=1)
        qc_median = np.median(qc, axis=1)
        blank_median = np.median(blank, axis=1)
        qc_mean = np.mean(qc, axis=1)
        qc_cv = np.divide(np.std(qc, axis=1), qc_mean,
                          out=np.full(len(table), np.inf), where=qc_mean > 0)
        blank_ratio = np.divide(blank_median, qc_median,
                                out=np.full(len(table), np.inf), where=qc_median > 0)
        feature["qc_detection_fraction"] = qc_detection
        feature["study_detection_fraction"] = study_detection
        feature["qc_cv"] = qc_cv
        feature["blank_qc_ratio"] = blank_ratio
        feature["measurement_qualified"] = (
            (qc_detection >= 0.75) & (study_detection >= 0.10)
            & (qc_cv <= 0.30) & (blank_ratio <= 0.20)
        )
        scans = pd.concat([read_ms2(path, index) for index, path in enumerate(dda_paths)],
                          ignore_index=True)
        linked = link(feature[["mz", "rt_sec"]], scans, args.ppm, args.rt_sec)
        feature = pd.concat([feature, linked], axis=1)
        feature["linked_any"] = feature["candidate_scans"] > 0
        feature["linked_replicated"] = feature["dda_replicates"] >= 2
        feature.to_csv(args.output / f"{analysis}_feature_linkage.csv.gz", index=False, compression="gzip")
        qualified = feature.loc[feature["measurement_qualified"]]
        report = {
            "analysis_id": analysis,
            "polarity": specification["polarity"],
            "features": int(len(feature)),
            "injection_columns": len(measurement_columns),
            "phenotype_blind_column_counts": {
                "pooled_qc_non_dda": len(qc_columns), "blank": len(blank_columns),
                "study_without_group_use": len(study_columns),
            },
            "dda_files": len(dda_paths),
            "dda_ms2_scans": int(len(scans)),
            "all_features_linked_any": int(feature["linked_any"].sum()),
            "all_features_linked_any_fraction": float(feature["linked_any"].mean()),
            "all_features_linked_ge2_dda": int(feature["linked_replicated"].sum()),
            "all_features_linked_ge2_dda_fraction": float(feature["linked_replicated"].mean()),
            "measurement_qualified_features": int(len(qualified)),
            "qualified_linked_any": int(qualified["linked_any"].sum()),
            "qualified_linked_any_fraction": float(qualified["linked_any"].mean()) if len(qualified) else 0.0,
            "qualified_linked_ge2_dda": int(qualified["linked_replicated"].sum()),
            "qualified_linked_ge2_dda_fraction": float(qualified["linked_replicated"].mean()) if len(qualified) else 0.0,
        }
        panel_reports.append(report)
        provenance[table_path.name] = sha256(table_path)
        provenance.update({str(path.relative_to(args.dda)): sha256(path) for path in dda_paths})
        print(json.dumps(report), flush=True)

    total_qualified = sum(row["measurement_qualified_features"] for row in panel_reports)
    total_linked = sum(row["qualified_linked_ge2_dda"] for row in panel_reports)
    gates = {
        "both_primary_panels_evaluable": len(panel_reports) == 2,
        "each_panel_has_three_dda_files": all(row["dda_files"] == 3 for row in panel_reports),
        "qualified_replicated_ms2_entities_ge_250": total_linked >= 250,
        "qualified_replicated_ms2_fraction_ge_5pct": total_linked / max(total_qualified, 1) >= 0.05,
    }
    report = {
        "status": "ST003325_FEATURE_SPECTRUM_LINKAGE_COMPLETE",
        "formal": True,
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed",
        "sealed": False,
        "phenotype_or_perturbation_labels_used": False,
        "parameters": {
            "ppm": args.ppm, "rt_sec": args.rt_sec,
            "qc_detection_min": 0.75, "study_detection_min": 0.10,
            "qc_cv_max": 0.30, "blank_qc_ratio_max": 0.20,
            "replicated_link_min_dda_files": 2,
        },
        "panels": panel_reports,
        "combined": {
            "measurement_qualified_features": total_qualified,
            "qualified_linked_ge2_dda": total_linked,
            "qualified_linked_ge2_dda_fraction": total_linked / max(total_qualified, 1),
        },
        "gates": gates,
        "pass_to_structure_hidden_perturbation_development": all(gates.values()),
        "provenance": provenance,
        "claim_limit": (
            "Phenotype-blind feature-to-pooled-QC-MS2 headroom only; m/z/RT linkage is not identity. "
            "No STS or DHEA-S effect, public evidence increment, or biological result."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

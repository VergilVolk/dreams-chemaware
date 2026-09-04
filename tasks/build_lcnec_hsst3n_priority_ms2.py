"""Freeze representative QC MS2 spectra for the strongest LCNEC dark modules."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=30, help="0 selects every robust nonredundant module")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/lcnec_hsst3n_priority_ms2"),
    )
    parser.add_argument(
        "--target-mode",
        choices=("robust_modules", "all_qc_qualified"),
        default="robust_modules",
        help="robust_modules preserves the original biology screen; all_qc_qualified is phenotype-blind",
    )
    return parser.parse_args()


def params(element: ET.Element) -> dict[str, tuple[str | None, str | None]]:
    output = {}
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] == "cvParam" and child.attrib.get("value") not in (None, ""):
            output[child.attrib.get("accession")] = (child.attrib.get("value"), child.attrib.get("unitName"))
    return output


def arrays(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    output = {}
    for bda in element.iter():
        if bda.tag.rsplit("}", 1)[-1] != "binaryDataArray":
            continue
        accessions = {child.attrib.get("accession") for child in bda.iter() if child.tag.rsplit("}", 1)[-1] == "cvParam"}
        kind = "mz" if "MS:1000514" in accessions else "intensity" if "MS:1000515" in accessions else None
        binary = next((child.text for child in bda if child.tag.rsplit("}", 1)[-1] == "binary"), None)
        if kind and binary and "MS:1000523" in accessions and "MS:1000574" in accessions:
            output[kind] = np.frombuffer(zlib.decompress(base64.b64decode(binary)), dtype="<f8")
    return output.get("mz", np.empty(0)), output.get("intensity", np.empty(0))


def main() -> None:
    args = parse_args()
    root = Path("data/validation")
    if args.target_mode == "robust_modules":
        robustness = pd.read_csv(root / "lcnec_hsst3n_dark_robustness_gate/normalization_robustness_ledger.csv")
        modules = pd.read_csv(root / "lcnec_hsst3n_dark_robustness_gate/nonredundant_module_membership.csv")
        merged = modules.merge(robustness, on=["family_id", "mz", "rt_sec"], how="left", validate="one_to_one")
        merged = merged[merged["cross_normalization_robust"].astype(bool)]
        representatives = (
            merged.sort_values(["module_id", "per_mg_drift_pqn_q", "family_id"])
            .groupby("module_id", as_index=False).first()
            .sort_values(["per_mg_drift_pqn_q", "module_id"])
            .reset_index(drop=True)
        )
        available_targets = int(merged["module_id"].nunique())
    else:
        ledger = pd.read_csv(root / "lcnec_hsst3n_qc_headroom_gate/precursor_family_ledger.csv")
        representatives = ledger[ledger["passes_all"].astype(bool)].copy()
        representatives = representatives.rename(columns={"mz_median": "mz", "rt_median_sec": "rt_sec"})
        representatives["module_id"] = -1
        representatives["per_mg_drift_pqn_log2fc"] = np.nan
        representatives["per_mg_drift_pqn_q"] = np.nan
        representatives = representatives.sort_values("family_id").reset_index(drop=True)
        if len(representatives) != 263:
            raise RuntimeError(f"expected 263 QC-qualified families, found {len(representatives)}")
        available_targets = len(representatives)
    if args.top_n:
        representatives = representatives.head(args.top_n).reset_index(drop=True)
    targets = representatives.to_dict("records")

    overview = pd.read_csv(root / "lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt", sep="\t")
    qc = overview[overview["NOTE"].eq("QC sample")]
    zip_path = root / "lcnec_zenodo19005638_preflight/MTB22_P073_HSST3n_mzML_public.zip"
    best: dict[int, dict[str, object]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if info.filename.lower().endswith(".mzml")}
        for _, file_row in qc.iterrows():
            with archive.open(members[file_row["mzML_FILE_NAME"]]) as handle:
                for _event, element in ET.iterparse(handle, events=("end",)):
                    if element.tag.rsplit("}", 1)[-1] != "spectrum":
                        continue
                    body = params(element)
                    if body.get("MS:1000511", (None, None))[0] != "2":
                        element.clear(); continue
                    mz = float(body.get("MS:1000744", ("nan", None))[0])
                    intensity = float(body.get("MS:1000042", ("0", None))[0])
                    rt_value, rt_unit = body.get("MS:1000016", ("nan", None))
                    rt = float(rt_value)
                    if rt_unit and "minute" in rt_unit.lower(): rt *= 60.0
                    for target in targets:
                        ppm = abs(mz - float(target["mz"])) / float(target["mz"]) * 1e6
                        if ppm <= 5.0 and abs(rt - float(target["rt_sec"])) <= 15.0:
                            current = best.get(int(target["family_id"]))
                            if current is None or intensity > float(current["precursor_intensity"]):
                                peak_mz, peak_intensity = arrays(element)
                                best[int(target["family_id"])] = {
                                    "sample_id": file_row["SAMPLE_ID"], "precursor_mz": mz, "rt_sec": rt,
                                    "precursor_intensity": intensity, "peak_mz": peak_mz, "peak_intensity": peak_intensity,
                                }
                    element.clear()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mgf_path = output_dir / "priority_dark_modules.mgf"
    manifest_rows = []
    with mgf_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            family_id = int(target["family_id"])
            spectrum = best.get(family_id)
            if spectrum is None or not len(spectrum["peak_mz"]):
                continue
            peak_mz = np.asarray(spectrum["peak_mz"])
            peak_intensity = np.asarray(spectrum["peak_intensity"])
            keep = (peak_mz > 0) & (peak_mz <= float(target["mz"]) + 2.0) & (peak_intensity > 0)
            peak_mz, peak_intensity = peak_mz[keep], peak_intensity[keep]
            order = np.argsort(peak_intensity)[-100:]
            peak_mz, peak_intensity = peak_mz[order], peak_intensity[order]
            order = np.argsort(peak_mz)
            handle.write("BEGIN IONS\n")
            handle.write(f"NAME=LCNEC_dark_family_{family_id}\n")
            handle.write(f"PEPMASS={float(target['mz']):.8f}\nIONMODE=negative\nMSLEVEL=2\n")
            handle.write(f"SOURCE={spectrum['sample_id']}\n")
            for mass, value in zip(peak_mz[order], peak_intensity[order], strict=True):
                handle.write(f"{mass:.6f} {value:.6f}\n")
            handle.write("END IONS\n\n")
            manifest_rows.append({
                "family_id": family_id, "module_id": int(target["module_id"]), "target_mz": float(target["mz"]),
                "target_rt_sec": float(target["rt_sec"]), "effect_log2fc": float(target["per_mg_drift_pqn_log2fc"]),
                "effect_q": float(target["per_mg_drift_pqn_q"]), "source_qc": spectrum["sample_id"],
                "source_precursor_mz": spectrum["precursor_mz"], "source_rt_sec": spectrum["rt_sec"],
                "source_precursor_intensity": spectrum["precursor_intensity"], "n_peaks": len(peak_mz),
            })
    pd.DataFrame(manifest_rows).to_csv(output_dir / "priority_dark_modules.csv", index=False)
    report = {
        "status": "lcnec_hsst3n_priority_ms2_complete", "formal": True,
        "target_mode": args.target_mode,
        "nonredundant_modules_available": available_targets,
        "modules_requested": len(targets), "modules_with_qc_ms2": len(manifest_rows),
        "pass_to_dreams_mona": len(manifest_rows) >= 20,
        "claim_limit": "Representative QC MS2 spectra only; no identity assignment.",
    }
    (output_dir / "priority_ms2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

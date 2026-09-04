#!/usr/bin/env python
"""Freeze the public NetID mouse-liver positive-mode targeted MS2 cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_PATTERN = re.compile(r"ID\s*=\s*(\d+)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_string(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/external/netid_v1/source/LiChenPU-NetID-9f63202"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_public_positive_ms2_20260901"),
    )
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    inventory_path = args.output_dir / "mouse_liver_ms2_inventory.csv.gz"
    cache_path = args.output_dir / "mouse_liver_ms2_spectra.npz"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "netid_public_positive_ms2_complete":
            raise RuntimeError("invalid existing positive-mode report")
        for name, path in (("inventory", inventory_path), ("cache", cache_path)):
            if sha256(path) != report["artifacts"][name]["sha256"]:
                raise RuntimeError(f"existing positive-mode {name} changed")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {args.output_dir}")

    panel = args.source_root / "Mouse_liver_pos"
    raw_path = panel / "raw_data.csv"
    output_path = panel / "NetID_output.csv"
    raw = pd.read_csv(raw_path)
    output = pd.read_csv(output_path)
    if len(raw) != len(output) or raw["groupId"].duplicated().any():
        raise RuntimeError("positive-mode raw/output row contract failed")
    if output["peak_id"].tolist() != list(range(1, len(output) + 1)):
        raise RuntimeError("positive-mode output row order is not one-based preserved order")
    row_by_group = {
        int(group): int(row0)
        for row0, group in enumerate(pd.to_numeric(raw["groupId"], errors="raise"))
    }

    records: list[dict[str, Any]] = []
    all_mz: list[np.ndarray] = []
    all_intensity: list[np.ndarray] = []
    offsets = [0]
    workbooks = sorted((panel / "MS2_pos_200524").glob("*.xlsx"))
    if not workbooks:
        raise RuntimeError("positive-mode targeted-MS2 workbooks are absent")
    for workbook_index, workbook in enumerate(workbooks, start=1):
        excel = pd.ExcelFile(workbook, engine="openpyxl")
        if "Sheet1" not in excel.sheet_names:
            raise RuntimeError(f"missing target table in {workbook}")
        metadata = pd.read_excel(excel, sheet_name="Sheet1")
        numeric = {name for name in excel.sheet_names if str(name).isdigit()}
        expected = {str(position + 2) for position in range(len(metadata))}
        if numeric != expected:
            raise RuntimeError(f"target/spectrum sheet mismatch in {workbook.name}")
        for position, target in metadata.iterrows():
            match = ID_PATTERN.search(safe_string(target.get("Comment")))
            if match is None:
                raise RuntimeError(f"missing NetID ID in {workbook.name} row {position}")
            group_id = int(match.group(1))
            if group_id not in row_by_group:
                raise RuntimeError(f"unknown NetID groupId={group_id}")
            raw_row0 = row_by_group[group_id]
            raw_row = raw.iloc[raw_row0]
            prediction = output.iloc[raw_row0]
            precursor = float(target["Mass_m_z_"])
            if abs(precursor - float(raw_row["medMz"])) > 5e-4:
                raise RuntimeError(f"precursor mismatch for groupId={group_id}")
            spectrum_raw = pd.read_excel(excel, sheet_name=str(position + 2), header=None)
            if spectrum_raw.shape[1] < 2:
                spectrum = pd.DataFrame(columns=["mz", "intensity"])
            else:
                spectrum = spectrum_raw.iloc[:, :2].copy()
                spectrum.columns = ["mz", "intensity"]
                spectrum = spectrum.apply(pd.to_numeric, errors="coerce").dropna()
                spectrum = spectrum[(spectrum["mz"] > 0) & (spectrum["intensity"] > 0)]
                spectrum = spectrum.sort_values("mz", kind="mergesort")
            mz = spectrum["mz"].to_numpy(dtype=np.float32)
            intensity = spectrum["intensity"].to_numpy(dtype=np.float32)
            all_mz.append(mz)
            all_intensity.append(intensity)
            offsets.append(offsets[-1] + len(mz))
            start, end = float(target["Start_min_"]), float(target["End_min_"])
            raw_rt = float(raw_row["medRt"])
            records.append(
                {
                    "source_file": workbook.name,
                    "source_sheet": str(position + 2),
                    "feature_group_id": group_id,
                    "netid_peak_id": raw_row0 + 1,
                    "precursor_mz": precursor,
                    "raw_rt_min": raw_rt,
                    "window_start_min": start,
                    "window_end_min": end,
                    "rt_in_window": bool(start <= raw_rt <= end),
                    "collision_energy": float(target["x_N_CE"]),
                    "n_fragment_peaks": int(len(mz)),
                    "netid_class": safe_string(prediction["class"]),
                    "netid_formula": safe_string(prediction["formula"]),
                    "netid_annotation": safe_string(prediction["annotation"]),
                }
            )
        if workbook_index % 25 == 0 or workbook_index == len(workbooks):
            print(f"[positive MS2] {workbook_index}/{len(workbooks)} workbooks", flush=True)

    inventory = pd.DataFrame.from_records(records)
    if inventory.empty:
        raise RuntimeError("no positive-mode targeted MS2 records")
    ge3 = inventory["n_fragment_peaks"] >= 3
    packed = {
        "source_file": inventory["source_file"].to_numpy(dtype=str),
        "source_sheet": inventory["source_sheet"].to_numpy(dtype=str),
        "feature_group_id": inventory["feature_group_id"].to_numpy(dtype=np.int64),
        "netid_peak_id": inventory["netid_peak_id"].to_numpy(dtype=np.int64),
        "precursor_mz": inventory["precursor_mz"].to_numpy(dtype=np.float32),
        "raw_rt_min": inventory["raw_rt_min"].to_numpy(dtype=np.float32),
        "collision_energy": inventory["collision_energy"].to_numpy(dtype=np.float32),
        "peak_offsets": np.asarray(offsets, dtype=np.int64),
        "fragment_mz": np.concatenate(all_mz).astype(np.float32, copy=False),
        "fragment_intensity": np.concatenate(all_intensity).astype(np.float32, copy=False),
        "netid_class": inventory["netid_class"].to_numpy(dtype=str),
        "netid_formula": inventory["netid_formula"].to_numpy(dtype=str),
        "netid_annotation": inventory["netid_annotation"].to_numpy(dtype=str),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(inventory_path, inventory)
    atomic_npz(cache_path, **packed)
    summary = {
        "workbooks": len(workbooks),
        "target_requests": len(inventory),
        "nonempty_spectra": int((inventory["n_fragment_peaks"] > 0).sum()),
        "spectra_ge_3_peaks": int(ge3.sum()),
        "unique_features_ge_3_peaks": int(inventory.loc[ge3, "netid_peak_id"].nunique()),
        "missing_placeholder_sheets": int((inventory["n_fragment_peaks"] == 0).sum()),
        "fragment_peaks": int(inventory["n_fragment_peaks"].sum()),
        "rt_window_matches": int(inventory["rt_in_window"].sum()),
    }
    report = {
        "status": "netid_public_positive_ms2_complete",
        "formal": True,
        "panel": "Mouse_liver_pos",
        "summary": summary,
        "gates": {
            "spectra_ge_3_peaks_ge_750": summary["spectra_ge_3_peaks"] >= 750,
            "unique_features_ge_3_peaks_ge_750": summary["unique_features_ge_3_peaks"] >= 750,
        },
        "artifacts": {
            "mouse_liver_ms2_inventory": {
                "relative_path": inventory_path.name,
                "sha256": sha256(inventory_path),
            },
            "mouse_liver_ms2_spectra": {
                "relative_path": cache_path.name,
                "sha256": sha256(cache_path),
            },
        },
        "provenance": {
            "raw_data_sha256": sha256(raw_path),
            "netid_output_sha256": sha256(output_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": "Positive-mode spectrum execution cache; NetID assignments remain predictions, not independent structure truth.",
    }
    report["gates"]["pass_to_component_isolated_ms2_edge_stage"] = all(report["gates"].values())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

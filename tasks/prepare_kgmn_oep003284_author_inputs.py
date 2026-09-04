#!/usr/bin/env python3
"""Materialize exact author MS1 tables plus byte-verified public OEP003284 MS2.

Supplementary Data 3 publishes the complete positive/negative raw peak tables
with the feature identifiers used by the paper's hidden-seed truth.  These
tables are the formal MS1 input.  Public mzXML files are linked (not copied)
after the NODE inventory has verified every MD5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SUPPLEMENT_NAME = "Supplementary data3.xlsx"
SUPPLEMENT_MD5 = "3e936cbbb22863371213ff8825c9f006"
PANELS = {
    "positive": {"sheet": "Raw_peak_table (Pos)", "token": "pos", "rows": 15942},
    "negative": {"sheet": "Raw_peak_table (Neg)", "token": "neg", "rows": 16760},
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def expected_samples() -> list[str]:
    """Published peak-table sample columns (polarity is omitted by the author)."""
    return [f"g{group}_46std_{repeat}" for group in (1, 2, 4) for repeat in range(1, 5)]


def expected_raw_names(token: str) -> list[str]:
    """NODE raw filenames (polarity is explicit)."""
    return [
        f"g{group}_46std_{token}_{repeat}.mzXML"
        for group in (1, 2, 4)
        for repeat in range(1, 5)
    ]


def validate_inventory(inventory_dir: Path, raw_root: Path) -> tuple[dict, pd.DataFrame]:
    report_path = inventory_dir / "report.json"
    files_path = inventory_dir / "node_files.csv"
    if not report_path.is_file() or not files_path.is_file():
        raise FileNotFoundError("NODE inventory is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "kgmn_oep003284_node_inventory_complete":
        raise RuntimeError("NODE inventory status mismatch")
    if report.get("local_validation", {}).get("ready") is not True:
        raise RuntimeError("NODE inventory did not byte-verify the local raw set")
    if report.get("files_csv_sha256") != digest(files_path):
        raise RuntimeError("NODE inventory file-list hash mismatch")
    files = pd.read_csv(files_path)
    required = {"name", "bytes", "md5", "run_no"}
    if not required.issubset(files.columns) or len(files) != 24:
        raise RuntimeError("NODE inventory file table drift")
    for row in files.itertuples(index=False):
        path = raw_root / str(row.name)
        if not path.is_file() or path.stat().st_size != int(row.bytes):
            raise RuntimeError(f"raw file missing or size changed after inventory: {path}")
    return report, files


def validate_peak_table(frame: pd.DataFrame, polarity: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = PANELS[polarity]
    samples = expected_samples()
    if list(map(str, frame.columns[:3])) != ["name", "mz", "rt"]:
        raise RuntimeError(f"{polarity} author peak table must start with name,mz,rt")
    if list(map(str, frame.columns[3:])) != samples:
        raise RuntimeError(f"{polarity} author sample columns drift")
    if len(frame) != meta["rows"]:
        raise RuntimeError(f"{polarity} author peak-table row drift: {len(frame)}")
    names = frame["name"].fillna("").astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise RuntimeError(f"{polarity} author feature names are empty or duplicated")
    numeric = frame[["mz", "rt", *samples]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric[["mz", "rt"]].to_numpy(float)).all():
        raise RuntimeError(f"{polarity} author m/z or RT contains non-finite values")
    if (numeric["mz"] <= 0).any() or (numeric["rt"] < 0).any():
        raise RuntimeError(f"{polarity} author m/z or RT is outside its physical range")
    abundance = numeric[samples].to_numpy(float)
    if np.any(np.isinf(abundance)) or np.any(abundance < 0):
        raise RuntimeError(f"{polarity} author abundance matrix is invalid")
    # MetDNA2 accepts missing abundances. Preserve author missingness exactly;
    # do not fill, normalize, filter, or transform the published table.
    exported = frame.copy()
    exported["name"] = names
    exported[["mz", "rt", *samples]] = numeric
    sample_info = pd.DataFrame(
        {
            "sample.name": samples,
            "group": [value.split("_", 1)[0] for value in samples],
        }
    )
    return exported, sample_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplements-dir", type=Path, default=Path("data/reference/kgmn_zenodo_7089991"))
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/reference/OEP003284"))
    args = parser.parse_args()

    if args.output_root.exists():
        raise RuntimeError(f"refusing to overwrite KGMN author input root: {args.output_root}")
    supplement = args.supplements_dir / SUPPLEMENT_NAME
    if not supplement.is_file() or digest(supplement, "md5") != SUPPLEMENT_MD5:
        raise RuntimeError("Supplementary Data 3 is missing or has the wrong MD5")
    inventory, files = validate_inventory(args.inventory_dir, args.raw_root)

    args.output_root.mkdir(parents=True)
    panel_reports: dict[str, dict[str, object]] = {}
    try:
        for polarity, meta in PANELS.items():
            panel_dir = args.output_root / polarity
            panel_dir.mkdir()
            frame = pd.read_excel(supplement, sheet_name=meta["sheet"])
            peak_table, sample_info = validate_peak_table(frame, polarity)
            data_path = panel_dir / "data.csv"
            sample_path = panel_dir / "sample.info.csv"
            peak_table.to_csv(data_path, index=False, na_rep="NA", float_format="%.15g")
            sample_info.to_csv(sample_path, index=False)

            panel_raw_names = expected_raw_names(meta["token"])
            inventory_names = set(files["name"].astype(str))
            if not set(panel_raw_names).issubset(inventory_names):
                raise RuntimeError(f"{polarity} NODE inventory lacks expected mzXML files")
            for name in panel_raw_names:
                source = (args.raw_root / name).resolve(strict=True)
                destination = panel_dir / name
                os.symlink(source, destination)
            if len(list(panel_dir.glob("*.mzXML"))) != 12:
                raise RuntimeError(f"{polarity} did not stage exactly 12 mzXML links")

            panel_reports[polarity] = {
                "source_sheet": meta["sheet"],
                "features": int(len(peak_table)),
                "samples": int(len(sample_info)),
                "mzxml_files": 12,
                "data_csv_sha256": digest(data_path),
                "sample_info_sha256": digest(sample_path),
                "feature_name_sha256": hashlib.sha256(
                    "\n".join(peak_table["name"].astype(str)).encode("utf-8")
                ).hexdigest(),
            }
    except Exception:
        # Fail closed without leaving an apparently ready partial input tree.
        (args.output_root / "INCOMPLETE").write_text("input preparation failed\n", encoding="utf-8")
        raise

    report = {
        "status": "kgmn_oep003284_author_inputs_complete",
        "formal": True,
        "panels": panel_reports,
        "contracts": {
            "ms1_source": "published Supplementary Data 3 raw peak tables",
            "feature_identifiers_are_author_identifiers": True,
            "ms1_reprocessing_performed": False,
            "mzxml_byte_verified_by_node_inventory": True,
            "phenotype_or_outcome_used": False,
            "hidden_seed_split_used": False,
            "same_inputs_for_all_annotation_arms": True,
        },
        "provenance": {
            "supplementary_data3_sha256": digest(supplement),
            "node_inventory_report_sha256": digest(args.inventory_dir / "report.json"),
            "node_inventory_files_sha256": digest(args.inventory_dir / "node_files.csv"),
            "script_sha256": digest(Path(__file__)),
        },
        "claim_limit": "Exact published MS1 feature table plus public raw MS2 inputs; no annotation result.",
    }
    report_path = args.output_root / "author_input_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

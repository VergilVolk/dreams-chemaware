#!/usr/bin/env python3
"""Freeze the identifier bridge between KGMN truth and author MS1 peak tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SUPPLEMENT_MD5 = "3e936cbbb22863371213ff8825c9f006"
SHEETS = {
    "positive": ("Raw_peak_table (Pos)", 15942),
    "negative": ("Raw_peak_table (Neg)", 16760),
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_peak_name(series: pd.Series, label: str) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise RuntimeError(f"{label} contains empty peak_name")
    return values


def load_raw_tables(path: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for polarity, (sheet, expected_rows) in SHEETS.items():
        frame = pd.read_excel(path, sheet_name=sheet, usecols=["name", "mz", "rt"])
        if len(frame) != expected_rows:
            raise RuntimeError(f"{sheet} row drift: {len(frame)} != {expected_rows}")
        frame = frame.rename(columns={"name": "peak_name", "mz": "raw_mz", "rt": "raw_rt"})
        frame["peak_name"] = clean_peak_name(frame["peak_name"], sheet)
        if frame["peak_name"].duplicated().any():
            raise RuntimeError(f"{sheet} contains duplicate feature identifiers")
        for column in ("raw_mz", "raw_rt"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[["raw_mz", "raw_rt"]].to_numpy()).all():
            raise RuntimeError(f"{sheet} contains non-finite coordinates")
        result[polarity] = frame
    return result


def reconcile(
    truth: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    *,
    label: str,
    mz_tolerance_da: float,
    rt_tolerance_sec: float,
) -> pd.DataFrame:
    required = {"polarity", "peak_name", "mz", "rt"}
    missing = required.difference(truth.columns)
    if missing:
        raise RuntimeError(f"{label} misses columns: {sorted(missing)}")
    pieces: list[pd.DataFrame] = []
    for polarity, raw in raw_tables.items():
        part = truth.loc[truth["polarity"].eq(polarity)].copy()
        part["peak_name"] = clean_peak_name(part["peak_name"], label)
        part["mz"] = pd.to_numeric(part["mz"], errors="raise").astype(float)
        part["rt"] = pd.to_numeric(part["rt"], errors="raise").astype(float)
        part = part.merge(raw, on="peak_name", how="left", validate="one_to_one")
        pieces.append(part)
    merged = pd.concat(pieces, ignore_index=True)
    if len(merged) != len(truth):
        raise RuntimeError(f"{label} reconciliation changed row count")
    if merged[["raw_mz", "raw_rt"]].isna().any().any():
        missing_names = merged.loc[merged["raw_mz"].isna(), "peak_name"].tolist()
        raise RuntimeError(f"{label} features missing from author peak tables: {missing_names[:10]}")
    merged["abs_mz_delta_da"] = (merged["mz"] - merged["raw_mz"]).abs()
    merged["abs_rt_delta_sec"] = (merged["rt"] - merged["raw_rt"]).abs()
    bad = merged.loc[
        merged["abs_mz_delta_da"].gt(mz_tolerance_da)
        | merged["abs_rt_delta_sec"].gt(rt_tolerance_sec)
    ]
    if not bad.empty:
        raise RuntimeError(
            f"{label} author-coordinate mismatch for {len(bad)} rows; "
            f"max dmz={bad['abs_mz_delta_da'].max():.6g}, "
            f"max drt={bad['abs_rt_delta_sec'].max():.6g}"
        )
    return merged.sort_values(["polarity", "peak_name"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supplement", type=Path,
        default=Path("data/reference/kgmn_zenodo_7089991/Supplementary data3.xlsx"),
    )
    parser.add_argument(
        "--contract-dir", type=Path,
        default=Path("data/validation/kgmn_external_validation_contract_20260831"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/kgmn_oep003284_author_identifier_contract_20260831"),
    )
    parser.add_argument("--mz-tolerance-da", type=float, default=1e-4)
    parser.add_argument("--rt-tolerance-sec", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite identifier contract: {args.output_dir}")
    if digest(args.supplement, "md5") != EXPECTED_SUPPLEMENT_MD5:
        raise RuntimeError("Supplementary data3 MD5 mismatch")
    report_path = args.contract_dir / "report.json"
    level1_path = args.contract_dir / "level1_seed_universe.csv.gz"
    products_path = args.contract_dir / "standard_confirmed_products.csv.gz"
    if not all(path.is_file() for path in (report_path, level1_path, products_path)):
        raise FileNotFoundError("external validation contract is incomplete")
    contract = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hashes = contract.get("provenance", {}).get("outputs_sha256", {})
    for path in (level1_path, products_path):
        if expected_hashes.get(path.name) != digest(path):
            raise RuntimeError(f"contract output hash mismatch: {path.name}")

    raw_tables = load_raw_tables(args.supplement)
    level1 = reconcile(
        pd.read_csv(level1_path), raw_tables, label="Level-1 seed universe",
        mz_tolerance_da=args.mz_tolerance_da, rt_tolerance_sec=args.rt_tolerance_sec,
    )
    products = reconcile(
        pd.read_csv(products_path), raw_tables, label="standard-confirmed products",
        mz_tolerance_da=args.mz_tolerance_da, rt_tolerance_sec=args.rt_tolerance_sec,
    )
    if len(level1) != 80 or level1["inchikey1"].nunique() != 42:
        raise RuntimeError("Level-1 identifier universe drift")
    if level1["formula"].nunique() != level1["inchikey1"].nunique():
        raise RuntimeError("Level-1 formula clusters are not one-to-one with identities")
    if len(products) != 20 or products["truth_compound_id"].nunique() != 9:
        raise RuntimeError("standard-confirmed product universe drift")

    args.output_dir.mkdir(parents=True)
    level1_out = args.output_dir / "level1_author_peak_reconciliation.csv.gz"
    products_out = args.output_dir / "standard_product_author_peak_reconciliation.csv.gz"
    level1.to_csv(level1_out, index=False, compression="gzip")
    products.to_csv(products_out, index=False, compression="gzip")
    report = {
        "status": "kgmn_oep003284_author_identifier_contract_frozen",
        "formal": True,
        "level1": {
            "peak_rows": int(len(level1)), "identities": int(level1["inchikey1"].nunique()),
            "formulas": int(level1["formula"].nunique()),
            "maximum_abs_mz_delta_da": float(level1["abs_mz_delta_da"].max()),
            "maximum_abs_rt_delta_sec": float(level1["abs_rt_delta_sec"].max()),
            "all_author_peak_names_found": True,
        },
        "standard_confirmed_products": {
            "peak_rows": int(len(products)),
            "identities": int(products["truth_compound_id"].nunique()),
            "maximum_abs_mz_delta_da": float(products["abs_mz_delta_da"].max()),
            "maximum_abs_rt_delta_sec": float(products["abs_rt_delta_sec"].max()),
            "all_author_peak_names_found": True,
        },
        "thresholds": {"mz_tolerance_da": args.mz_tolerance_da, "rt_tolerance_sec": args.rt_tolerance_sec},
        "contracts": {
            "formal_ms1_input": "exact Supplementary Data 3 raw peak tables",
            "xcms_reconstruction": "sensitivity analysis only",
            "truth_key": "author peak_name plus polarity",
            "formula_cluster_note": "42 Level-1 identities map one-to-one to 42 formulas",
            "outcomes_used": False,
        },
        "provenance": {
            "supplement_md5": EXPECTED_SUPPLEMENT_MD5,
            "supplement_sha256": digest(args.supplement),
            "external_contract_report_sha256": digest(report_path),
            "level1_input_sha256": digest(level1_path),
            "products_input_sha256": digest(products_path),
            "level1_reconciliation_sha256": digest(level1_out),
            "products_reconciliation_sha256": digest(products_out),
            "script_sha256": digest(Path(__file__)),
        },
        "claim_limit": "Identifier bridge only; no prediction or performance claim.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

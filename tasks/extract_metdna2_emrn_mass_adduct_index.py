#!/usr/bin/env python
"""Freeze the MetDNA2 eMRN mass index and official annotation adducts.

This is a source-extraction stage.  It deliberately contains no benchmark
outcomes and makes no annotation claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def molecular_multiplier(adduct: str) -> int:
    match = re.match(r"^\[(\d*)M", adduct)
    if match is None:
        raise ValueError(f"unsupported MetDNA adduct syntax: {adduct}")
    return int(match.group(1) or "1")


def charge_magnitude(adduct: str) -> int:
    match = re.search(r"\](\d*)[+-]$", adduct)
    if match is None:
        raise ValueError(f"unsupported MetDNA adduct charge: {adduct}")
    return int(match.group(1) or "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compound-rda", type=Path,
        default=Path("third_party/MetDNA2/data/cpd_emrn.rda"),
    )
    parser.add_argument(
        "--adduct-rda", type=Path,
        default=Path("third_party/MetDNA2/data/lib_adduct_nl.rda"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/reference/metdna2_emrn_mass_adduct_20260828"),
    )
    args = parser.parse_args()
    for path in (args.compound_rda, args.adduct_rda):
        if not path.exists():
            raise FileNotFoundError(path)
    try:
        import pyreadr
        import rdata
    except ImportError as exc:
        raise RuntimeError("extractor requires pyreadr and rdata") from exc

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    compounds = pyreadr.read_r(str(args.compound_rda)).get("cpd_emrn")
    if compounds is None:
        raise RuntimeError("cpd_emrn object missing")
    columns = [
        "id", "inchikey1", "inchikey", "formula", "monoisotopic_mass",
        "source", "type", "min_reaction_step", "name",
    ]
    missing = sorted(set(columns) - set(compounds.columns))
    if missing:
        raise RuntimeError(f"cpd_emrn schema changed: missing {missing}")
    mass = compounds[columns].copy()
    for column in ["id", "inchikey1", "inchikey", "formula", "source", "type", "name"]:
        mass[column] = mass[column].fillna("").astype(str)
    mass["inchikey1"] = mass["inchikey1"].str.upper()
    mass["monoisotopic_mass"] = pd.to_numeric(mass["monoisotopic_mass"], errors="coerce")
    mass["min_reaction_step"] = pd.to_numeric(mass["min_reaction_step"], errors="coerce")
    mass = mass[
        mass["inchikey1"].str.fullmatch(r"[A-Z]{14}")
        & np.isfinite(mass["monoisotopic_mass"])
        & mass["monoisotopic_mass"].gt(0)
        & mass["min_reaction_step"].between(0, 8)
    ].copy()
    mass["min_reaction_step"] = mass["min_reaction_step"].astype(np.int8)
    mass = mass.sort_values(
        ["inchikey1", "monoisotopic_mass", "min_reaction_step", "id", "source"]
    ).drop_duplicates(
        ["inchikey1", "monoisotopic_mass", "min_reaction_step", "id"]
    ).reset_index(drop=True)

    adduct_object = rdata.read_rda(str(args.adduct_rda)).get("lib_adduct_nl")
    if not isinstance(adduct_object, dict):
        raise RuntimeError("lib_adduct_nl object missing")
    adduct_frames: list[pd.DataFrame] = []
    for polarity in ("positive", "negative"):
        source = adduct_object.get(np.str_(polarity), adduct_object.get(polarity))
        if source is None:
            raise RuntimeError(f"lib_adduct_nl${polarity} missing")
        frame = pd.DataFrame(source).copy()
        required = ["adduct", "delta_mz", "type", "annotation", "credential"]
        if not set(required).issubset(frame.columns):
            raise RuntimeError(f"adduct schema changed for {polarity}")
        frame = frame[required + (["rule_limitation"] if "rule_limitation" in frame else [])]
        frame["polarity"] = polarity
        frame["adduct"] = frame["adduct"].astype(str)
        frame["delta_mz"] = pd.to_numeric(frame["delta_mz"], errors="coerce")
        frame["nmol"] = frame["adduct"].map(molecular_multiplier)
        frame["charge"] = frame["adduct"].map(charge_magnitude)
        frame["default_annotation"] = frame["annotation"].astype(str).eq("Yes")
        if not np.isfinite(frame["delta_mz"]).all():
            raise RuntimeError(f"non-finite adduct mass offset for {polarity}")
        adduct_frames.append(frame)
    adducts = pd.concat(adduct_frames, ignore_index=True)
    if adducts.duplicated(["polarity", "adduct"]).any():
        raise RuntimeError("duplicate polarity/adduct rows in official table")

    mass_path = output / "emrn_compound_mass.csv.gz"
    adduct_path = output / "metdna2_adducts.csv"
    mass.to_csv(mass_path, index=False, compression="gzip")
    adducts.to_csv(adduct_path, index=False)
    report = {
        "status": "metdna2_emrn_mass_adduct_index_complete",
        "formal": True,
        "compound_rows": int(len(mass)),
        "compound_identities": int(mass["inchikey1"].nunique()),
        "compound_formulas": int(mass.loc[mass["formula"].ne(""), "formula"].nunique()),
        "minimum_step_counts": {
            str(int(key)): int(value)
            for key, value in mass.groupby("min_reaction_step").size().items()
        },
        "adduct_rows": int(len(adducts)),
        "default_annotation_adducts": {
            polarity: int(group["default_annotation"].sum())
            for polarity, group in adducts.groupby("polarity")
        },
        "mz_definition": "(monoisotopic_mass * nmol + delta_mz) / abs(charge)",
        "provenance": {
            "compound_rda_sha256": sha256(args.compound_rda),
            "adduct_rda_sha256": sha256(args.adduct_rda),
            "compound_index_sha256": sha256(mass_path),
            "adduct_index_sha256": sha256(adduct_path),
        },
        "contracts": {
            "benchmark_outcomes_used": False,
            "default_adduct_filter": "official annotation == Yes",
            "reaction_steps_preserved": True,
        },
        "claim_limit": "Source index only; no MS1 candidate coverage or annotation performance.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

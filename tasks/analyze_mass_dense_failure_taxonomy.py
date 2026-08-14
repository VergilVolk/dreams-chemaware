"""Chemical and acquisition-condition taxonomy of mass-dense retrieval cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator

import audit_e0_failures as chemistry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/validation/mass_dense_failure_audit"
DEFAULT_HDF5 = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/mass_dense_failure_taxonomy"
DEFAULT_MCES_CACHE = ROOT / "data/validation/e0_failure_audit/mces_cache.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compute-mces", action="store_true")
    parser.add_argument("--reuse-mces-cache", type=Path, default=DEFAULT_MCES_CACHE)
    parser.add_argument("--mces-threshold", type=float, default=10.0)
    parser.add_argument("--mces-time-limit", type=float, default=3.0)
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def morgan_bin(value: float) -> str:
    if value < 0.25:
        return "<0.25"
    if value < 0.5:
        return "0.25-0.49"
    if value < 0.75:
        return "0.50-0.74"
    return ">=0.75"


def peak_count(spectrum: np.ndarray) -> int:
    return int(np.sum(
        np.isfinite(spectrum[0]) & np.isfinite(spectrum[1])
        & (spectrum[0] > 0) & (spectrum[1] > 0)
    ))


def load_row_metadata(hdf5_path: Path, rows: np.ndarray) -> dict[int, dict]:
    unique = np.unique(rows.astype(np.int64))
    with h5py.File(hdf5_path, "r") as handle:
        smiles = handle["smiles"].asstr()[unique]
        instruments = handle["INSTRUMENT_TYPE"].asstr()[unique]
        energies = np.asarray(handle["COLLISION_ENERGY"][unique], dtype=float)
        spectra = np.asarray(handle["spectrum"][unique])
    return {
        int(row): {
            "smiles": str(smiles[i]),
            "instrument": str(instruments[i]),
            "collision_energy": (
                None if not np.isfinite(energies[i]) else float(energies[i])
            ),
            "peak_count": peak_count(spectra[i]),
        }
        for i, row in enumerate(unique)
    }


def add_structure_and_conditions(frame: pd.DataFrame, hdf5_path: Path) -> pd.DataFrame:
    all_rows = np.concatenate([
        frame["query_row"].to_numpy(dtype=np.int64),
        frame["positive_row"].to_numpy(dtype=np.int64),
        frame["official_best_negative_row"].to_numpy(dtype=np.int64),
    ])
    metadata = load_row_metadata(hdf5_path, all_rows)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    molecule_cache = {}

    def molecule(row: int):
        smiles = metadata[int(row)]["smiles"]
        if smiles not in molecule_cache:
            molecule_cache[smiles] = chemistry.molecule_features(smiles, fpgen)
        return molecule_cache[smiles]

    records = []
    for item in frame.to_dict("records"):
        query_meta = metadata[int(item["query_row"])]
        positive_meta = metadata[int(item["positive_row"])]
        negative_meta = metadata[int(item["official_best_negative_row"])]
        query_molecule = molecule(item["query_row"])
        negative_molecule = molecule(item["official_best_negative_row"])
        relation = chemistry.pair_structure_features(query_molecule, negative_molecule)
        query_ce = query_meta["collision_energy"]
        positive_ce = positive_meta["collision_energy"]
        ce_difference = (
            abs(query_ce - positive_ce)
            if query_ce is not None and positive_ce is not None else None
        )
        item.update({
            "query_smiles": query_meta["smiles"],
            "negative_smiles": negative_meta["smiles"],
            "query_formula": query_molecule["formula"],
            "negative_formula": negative_molecule["formula"],
            "same_formula": relation["same_formula"],
            "scaffold_relation": relation["scaffold_relation"],
            "formula_scaffold_group": relation["formula_scaffold_group"],
            "morgan_tanimoto": relation["morgan_tanimoto"],
            "morgan_bin": morgan_bin(relation["morgan_tanimoto"]),
            "query_ring_class": query_molecule["ring_class"],
            "query_heavy_atoms": query_molecule["heavy_atoms"],
            "query_peak_count": query_meta["peak_count"],
            "negative_peak_count": negative_meta["peak_count"],
            "positive_instrument_changed": (
                query_meta["instrument"] != positive_meta["instrument"]
            ),
            "positive_ce_difference": ce_difference,
            "positive_ce_changed_ge_10": (
                ce_difference is not None and ce_difference >= 10
            ),
            "official_failure": not bool(item["official_correct"]),
            "raw_failure": not bool(item["raw_correct"]),
            "raw_error_repaired": item["group"] == "raw_wrong_official_correct",
        })
        records.append(item)
    return pd.DataFrame(records)


def compute_mces(frame: pd.DataFrame, cache_path: Path, threshold: float, time_limit: float):
    from myopic_mces import MCES

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    unique = {}
    for row in frame.itertuples():
        key = "|".join(sorted([row.ik14, row.official_best_negative_ik14]))
        unique.setdefault(key, (row.query_smiles, row.negative_smiles))
    missing = [key for key in unique if key not in cache]
    print(f"MCES unique pairs={len(unique)}; missing={len(missing)}", flush=True)
    for index, key in enumerate(missing, 1):
        left, right = unique[key]
        try:
            _, distance, elapsed, mode = MCES(
                left,
                right,
                threshold=threshold,
                solver_options={"msg": False, "timeLimit": time_limit, "threads": 1},
                catch_errors=True,
            )
            cache[key] = {
                "distance": float(distance),
                "elapsed_seconds": float(elapsed),
                "mode": int(mode),
            }
        except Exception as exc:
            cache[key] = {"distance": -1.0, "error": str(exc), "mode": -1}
        if index % 50 == 0 or index == len(missing):
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            print(f"  MCES {index}/{len(missing)}", flush=True)
    values, bins = [], []
    for row in frame.itertuples():
        key = "|".join(sorted([row.ik14, row.official_best_negative_ik14]))
        distance = cache[key].get("distance", -1.0)
        values.append(distance)
        bins.append(chemistry.mces_bin(distance))
    frame = frame.copy()
    frame["mces"] = values
    frame["mces_bin"] = bins
    return frame


def attach_cached_mces(frame: pd.DataFrame, cache: dict) -> pd.DataFrame:
    values, bins, cached = [], [], []
    for row in frame.itertuples():
        key = "|".join(sorted([row.ik14, row.official_best_negative_ik14]))
        entry = cache.get(key)
        distance = entry.get("distance", -1.0) if entry is not None else -1.0
        values.append(distance)
        bins.append(chemistry.mces_bin(distance))
        cached.append(entry is not None and distance >= 0)
    frame = frame.copy()
    frame["mces"] = values
    frame["mces_bin"] = bins
    frame["mces_cached"] = cached
    return frame


def categorical_enrichment(frame: pd.DataFrame, field: str, outcome: str) -> list[dict]:
    overall = float(frame[outcome].mean())
    output = []
    for category, subset in frame.groupby(field, dropna=False):
        if len(subset) < 10:
            continue
        rate = float(subset[outcome].mean())
        output.append({
            "field": field,
            "category": str(category),
            "outcome": outcome,
            "n_queries": int(len(subset)),
            "event_rate": rate,
            "overall_rate": overall,
            "risk_ratio": rate / overall if overall > 0 else math.nan,
        })
    return sorted(output, key=lambda row: -row["risk_ratio"])


def summarize_split(frame: pd.DataFrame) -> dict:
    official_errors = frame.loc[frame["official_failure"]]
    raw_errors = frame.loc[frame["raw_failure"]]
    return {
        "n_queries": len(frame),
        "official_failure_rate": float(frame["official_failure"].mean()),
        "raw_failure_rate": float(frame["raw_failure"].mean()),
        "raw_error_repair_rate": float(raw_errors["raw_error_repaired"].mean()),
        "official_failure_same_formula_fraction": float(
            (official_errors["same_formula"] == "true").mean()
        ) if len(official_errors) else None,
        "official_failure_same_scaffold_fraction": float(
            (official_errors["scaffold_relation"] == "same_scaffold").mean()
        ) if len(official_errors) else None,
        "official_failure_morgan_median": float(
            official_errors["morgan_tanimoto"].median()
        ) if len(official_errors) else None,
        "official_failure_mces_median": (
            float(official_errors.loc[official_errors["mces"] >= 0, "mces"].median())
            if "mces" in frame and np.any(official_errors["mces"] >= 0) else None
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    enrichment_tables = []
    frames = {}
    reusable_cache = (
        json.loads(args.reuse_mces_cache.read_text(encoding="utf-8"))
        if args.reuse_mces_cache.exists() else {}
    )
    for split in ("discovery", "confirmation"):
        frame = pd.read_csv(args.input_dir / f"{split}_queries.csv")
        frame = add_structure_and_conditions(frame, args.hdf5)
        if args.compute_mces:
            split_cache = args.output_dir / f"{split}_mces_cache.json"
            if not split_cache.exists() and reusable_cache:
                split_cache.write_text(
                    json.dumps(reusable_cache, indent=2), encoding="utf-8"
                )
            frame = compute_mces(
                frame,
                split_cache,
                args.mces_threshold,
                args.mces_time_limit,
            )
        else:
            frame = attach_cached_mces(frame, reusable_cache)
        frame.to_csv(args.output_dir / f"{split}_taxonomy.csv", index=False)
        frames[split] = frame
        reports[split] = summarize_split(frame)
        fields = [
            "same_formula", "scaffold_relation", "morgan_bin", "query_ring_class",
            "positive_instrument_changed", "positive_ce_changed_ge_10", "adduct",
        ]
        if np.any(frame["mces"] >= 0):
            fields.append("mces_bin")
        for outcome in ("official_failure", "raw_error_repaired"):
            eligible = frame if outcome == "official_failure" else frame.loc[frame["raw_failure"]]
            for field in fields:
                for row in categorical_enrichment(eligible, field, outcome):
                    row["split"] = split
                    enrichment_tables.append(row)
    enrichment = pd.DataFrame(enrichment_tables)
    enrichment.to_csv(args.output_dir / "categorical_enrichment.csv", index=False)

    replicated = []
    for keys, subset in enrichment.groupby(["field", "category", "outcome"]):
        if set(subset["split"]) != {"discovery", "confirmation"}:
            continue
        by_split = subset.set_index("split")
        disc = float(by_split.loc["discovery", "risk_ratio"])
        conf = float(by_split.loc["confirmation", "risk_ratio"])
        if (disc - 1) * (conf - 1) > 0:
            replicated.append({
                "field": keys[0],
                "category": keys[1],
                "outcome": keys[2],
                "discovery_risk_ratio": disc,
                "confirmation_risk_ratio": conf,
                "direction": "enriched" if disc > 1 else "depleted",
            })
    replicated_frame = pd.DataFrame(replicated)
    replicated_frame.to_csv(
        args.output_dir / "replicated_enrichment_directions.csv", index=False
    )
    report = {
        "status": "mass_dense_failure_taxonomy",
        "mces_computed": args.compute_mces,
        "reused_mces_cache": str(args.reuse_mces_cache),
        "mces_query_coverage": {
            split: float(np.mean(frame["mces"] >= 0))
            for split, frame in frames.items()
        },
        "splits": reports,
        "replicated_enrichment_directions": len(replicated),
        "interpretation_limit": (
            "Morgan similarity is diagnostic only. MCES bins describe structural "
            "difficulty and are not used as the retrieval label in this audit."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()

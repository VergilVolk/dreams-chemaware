#!/usr/bin/env python3
"""Freeze outcome-free KGMN external-validation panels from author supplements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MD5 = {
    "Supplementary data1.xlsx": "8eadc3821d6e6973cc81cb3596ef414b",
    "Supplementary data2.xlsx": "9a047288772908c6bb0d34573bb3b2f8",
    "Supplementary data3.xlsx": "3e936cbbb22863371213ff8825c9f006",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def read_peak_truth(path: Path) -> pd.DataFrame:
    frames = []
    for polarity, sheet in (
        ("positive", "Manual_curated_table (Positive)"),
        ("negative", "Manual_curated_table (Negative)"),
    ):
        frame = pd.read_excel(path, sheet_name=sheet)
        require_columns(
            frame,
            {"peak_name", "mz", "rt", "isotope", "adduct", "compound_id", "inchikey1", "data_set"},
            sheet,
        )
        frame = frame.copy()
        frame.insert(0, "polarity", polarity)
        frame.insert(1, "source_sheet", sheet)
        frames.append(frame)
    truth = pd.concat(frames, ignore_index=True)
    truth["inchikey1"] = clean_text(truth["inchikey1"])
    if len(truth) != 3451 or truth["inchikey1"].nunique() != 242:
        raise RuntimeError(
            f"manual peak truth drift: rows={len(truth)} identities={truth['inchikey1'].nunique()}"
        )
    if truth[["polarity", "peak_name", "data_set"]].duplicated().any():
        raise RuntimeError("manual peak truth contains duplicate panel peak keys")
    return truth


def read_level1_seed_universe(path: Path) -> pd.DataFrame:
    frames = []
    for polarity, sheet in (
        ("positive", "Knowledge_guided_network (Pos)"),
        ("negative", "Knowledge_guided_network (Neg)"),
    ):
        frame = pd.read_excel(path, sheet_name=sheet)
        require_columns(
            frame,
            {"peak_name", "id_zhulab", "confidence_level", "inchikey", "formula", "adduct"},
            sheet,
        )
        frame = frame.loc[clean_text(frame["confidence_level"]).eq("level1")].copy()
        frame.insert(0, "polarity", polarity)
        frame.insert(1, "source_sheet", sheet)
        frame["inchikey1"] = clean_text(frame["inchikey"]).str.slice(0, 14)
        frames.append(frame)
    seeds = pd.concat(frames, ignore_index=True)
    if seeds["inchikey1"].nunique() != 42:
        raise RuntimeError(f"expected 42 observed Level-1 identities, found {seeds['inchikey1'].nunique()}")
    return seeds


def build_hidden_seed_splits(seeds: pd.DataFrame, repeats: int, fraction: float, seed: int) -> pd.DataFrame:
    presence = (
        seeds.groupby("inchikey1", sort=True)["polarity"]
        .agg(lambda values: "+".join(sorted(set(values))))
        .rename("polarity_presence")
        .reset_index()
    )
    # A fresh random draw per repeat can leave some identities permanently hidden
    # while exposing others many times.  Shuffle each polarity stratum once, then
    # advance a circular window.  This keeps every repeat stratified and makes the
    # number of seed appearances differ by at most one within each stratum.
    rng = np.random.default_rng(seed)
    strata: list[tuple[np.ndarray, int]] = []
    for _, stratum in presence.groupby("polarity_presence", sort=True):
        identities = stratum["inchikey1"].sort_values().to_numpy(copy=True)
        rng.shuffle(identities)
        n_seed = max(1, int(round(len(identities) * fraction)))
        n_seed = min(n_seed, len(identities) - 1) if len(identities) > 1 else 1
        strata.append((identities, n_seed))

    rows = []
    for repeat in range(repeats):
        selected: set[str] = set()
        for identities, n_seed in strata:
            positions = (np.arange(n_seed) + repeat * n_seed) % len(identities)
            selected.update(identities[positions].tolist())
        for record in presence.itertuples(index=False):
            rows.append(
                {
                    "repeat": repeat,
                    "inchikey1": record.inchikey1,
                    "polarity_presence": record.polarity_presence,
                    "role": "seed" if record.inchikey1 in selected else "hidden_validation",
                }
            )
    result = pd.DataFrame(rows)
    if result.groupby("repeat")["inchikey1"].nunique().ne(42).any():
        raise RuntimeError("hidden-seed split lost identities")
    if result.groupby(["repeat", "inchikey1"]).size().ne(1).any():
        raise RuntimeError("hidden-seed split duplicated identities")
    appearances = (
        result.loc[result["role"].eq("seed")]
        .groupby(["polarity_presence", "inchikey1"])
        .size()
    )
    all_appearances = (
        presence.set_index(["polarity_presence", "inchikey1"])
        .assign(seed_appearances=appearances)
        ["seed_appearances"]
        .fillna(0)
        .astype(int)
    )
    spread = all_appearances.groupby(level=0).agg(lambda values: int(values.max() - values.min()))
    if spread.gt(1).any():
        raise RuntimeError(f"unbalanced repeated hidden-seed exposure: {spread.to_dict()}")
    if repeats * fraction >= 1.0 and all_appearances.eq(0).any():
        raise RuntimeError("balanced design unexpectedly left identities never used as seeds")
    if repeats * (1.0 - fraction) >= 1.0 and all_appearances.eq(repeats).any():
        raise RuntimeError("balanced design unexpectedly left identities never hidden")
    return result


def read_confirmed_products(path_data2: Path, path_data3: Path) -> pd.DataFrame:
    universe = pd.concat(
        [
            pd.read_excel(path_data2, sheet_name="Known metabolite (46STD)"),
            pd.read_excel(path_data2, sheet_name="Unknown metabolite (46STD)"),
        ],
        ignore_index=True,
    )
    require_columns(universe, {"id", "formula", "inchikey", "min_reaction_step"}, "46STD universe")
    universe = universe.drop_duplicates("id", keep="first").set_index("id", drop=False)

    rows = []
    for polarity, sheet in (("positive", "Validation_result (Pos)"), ("negative", "Validation_result (Neg)")):
        frame = pd.read_excel(path_data3, sheet_name=sheet)
        require_columns(
            frame,
            {
                "peak_name",
                "id_kegg",
                "validation_standard",
                "validation_spectral_DB",
                "validation_insilico_tool",
            },
            sheet,
        )
        frame = frame.loc[frame["validation_standard"].fillna(False).astype(bool)].copy()
        frame["id_kegg"] = clean_text(frame["id_kegg"])
        frame["candidate_count"] = frame["id_kegg"].str.count(";") + 1
        frame = frame.loc[frame["candidate_count"].eq(1)].copy()
        frame.insert(0, "polarity", polarity)
        frame.insert(1, "source_sheet", sheet)
        for record in frame.to_dict("records"):
            compound_id = record["id_kegg"]
            if compound_id not in universe.index:
                raise RuntimeError(f"standard-confirmed product not found in Data2 universe: {compound_id}")
            compound = universe.loc[compound_id]
            record.update(
                {
                    "truth_compound_id": compound_id,
                    "truth_inchikey1": str(compound["inchikey"])[:14],
                    "truth_formula": compound["formula"],
                    "truth_min_reaction_step": compound["min_reaction_step"],
                }
            )
            rows.append(record)
    products = pd.DataFrame(rows)
    if len(products) != 20 or products["truth_compound_id"].nunique() != 9:
        raise RuntimeError(
            f"confirmed product panel drift: rows={len(products)} identities={products['truth_compound_id'].nunique()}"
        )
    return products


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/reference/kgmn_zenodo_7089991"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/kgmn_external_validation_contract_20260831"),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.repeats < 2 or not 0.0 < args.seed_fraction < 1.0:
        raise ValueError("invalid hidden-seed parameters")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: args.source_dir / name for name in EXPECTED_MD5}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = digest(path, "md5")
        if observed != EXPECTED_MD5[name]:
            raise RuntimeError(f"source MD5 mismatch for {name}: {observed}")

    peak_truth = read_peak_truth(paths["Supplementary data1.xlsx"])
    level1_seeds = read_level1_seed_universe(paths["Supplementary data3.xlsx"])
    splits = build_hidden_seed_splits(level1_seeds, args.repeats, args.seed_fraction, args.seed)
    products = read_confirmed_products(paths["Supplementary data2.xlsx"], paths["Supplementary data3.xlsx"])

    outputs = {
        "manual_peak_truth.csv.gz": peak_truth,
        "level1_seed_universe.csv.gz": level1_seeds,
        "hidden_seed_splits.csv.gz": splits,
        "standard_confirmed_products.csv.gz": products,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False, compression="gzip")

    split_counts = (
        splits.groupby(["repeat", "role"]).size().unstack(fill_value=0).astype(int).to_dict("index")
    )
    seed_appearances = (
        splits.loc[splits["role"].eq("seed")]
        .groupby(["polarity_presence", "inchikey1"])
        .size()
        .rename("seed_appearances")
        .reset_index()
    )
    split_balance = {
        str(polarity): {
            "minimum_seed_appearances": int(group["seed_appearances"].min()),
            "maximum_seed_appearances": int(group["seed_appearances"].max()),
        }
        for polarity, group in seed_appearances.groupby("polarity_presence", sort=True)
    }
    report = {
        "status": "kgmn_external_validation_contract_frozen",
        "formal": True,
        "primary_protocol": {
            "name": "46STD_S9_hidden_seed",
            "level1_identities": int(level1_seeds["inchikey1"].nunique()),
            "level1_peak_rows": int(len(level1_seeds)),
            "repeats": args.repeats,
            "seed_fraction": args.seed_fraction,
            "split_counts": {str(key): value for key, value in split_counts.items()},
            "balanced_seed_appearances_by_polarity": split_balance,
            "every_identity_seeded_and_hidden": True,
            "metrics": [
                "hidden-identity coverage",
                "Top-1",
                "Top-3",
                "corrected and introduced versus author DP",
                "propagation-depth error rate",
            ],
            "threshold_rule": "all edge calibration and gates frozen outside OEP003284",
        },
        "secondary_protocol": {
            "name": "manual_peak_and_ion_form_assignment",
            "peak_rows": int(len(peak_truth)),
            "identities": int(peak_truth["inchikey1"].nunique()),
            "datasets": peak_truth.groupby(["data_set", "polarity"]).size().astype(int).to_dict(),
            "metric": "Top-3 identity plus isotope/adduct/ISF assignment under author peak-credential protocol",
        },
        "exploratory_product_panel": {
            "standard_confirmed_feature_rows": int(len(products)),
            "unique_compound_ids": int(products["truth_compound_id"].nunique()),
            "claim_limit": "too small for a standalone performance claim; mechanism corroboration only",
        },
        "arm_contract": {
            "noop_author": "author DP, exact output reproduction required",
            "official_dreams": "component-isolated calibrated official DreaMS edge probability",
            "author_official_intersection": "pre-registered primary; both edge systems must support propagation",
        },
        "forbidden": [
            "MTBLS13729 phenotype in seeds, thresholds or edge weights",
            "treating network neighbors as identity-positive embedding pairs",
            "using the 9-product exploratory panel to tune the model",
            "calling the consumed 200STD demonstration an independent external validation",
        ],
        "provenance": {
            "zenodo_record": 7089991,
            "source_md5": EXPECTED_MD5,
            "source_sha256": {name: digest(path) for name, path in paths.items()},
            "outputs_sha256": {name: digest(args.output_dir / name) for name in outputs},
        },
        "claim_limit": (
            "Frozen evaluation contract only. Full replay still requires the raw LC-MS files from "
            "OEP003284 and the author KGMN runtime; no performance has been evaluated here."
        ),
    }
    # JSON cannot encode tuple keys from the secondary panel.
    report["secondary_protocol"]["datasets"] = {
        f"{dataset}|{polarity}": int(count)
        for (dataset, polarity), count in peak_truth.groupby(["data_set", "polarity"]).size().items()
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

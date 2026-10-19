#!/usr/bin/env python
"""Build frozen evidence-control worlds for unknown-chemistry adjudication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contrast_atlas_core import (
    FORBIDDEN_EVIDENCE_COLUMNS,
    validate_effects,
    validate_links,
    validate_loadings,
    validate_unit_effects,
)


KNOWN = {"confirmed_known", "library_supported"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_covariates(raw: pd.DataFrame, covariate_columns: list[str], strata_columns: list[str]) -> pd.DataFrame:
    required = {"study_id", "contrast_id", "entity_id", *covariate_columns, *strata_columns}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"measurement covariates missing columns: {sorted(missing)}")
    forbidden = FORBIDDEN_EVIDENCE_COLUMNS & set(raw.columns)
    if forbidden:
        raise RuntimeError(f"measurement covariates contain phenotype/effect columns: {sorted(forbidden)}")
    frame = raw.copy()
    if frame.duplicated(["study_id", "contrast_id", "entity_id"]).any():
        raise RuntimeError("duplicate entity in measurement covariates")
    for column in covariate_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column])).any():
            raise RuntimeError(f"non-finite measurement covariate: {column}")
    return frame


def filter_world(
    effects: pd.DataFrame,
    loadings: pd.DataFrame,
    links: pd.DataFrame,
    eligible_keys: set[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_key = list(zip(effects["study_id"], effects["contrast_id"], effects["entity_id"]))
    selected_effects = effects.loc[[key in eligible_keys for key in effect_key]].copy()
    loading_key = list(zip(loadings["study_id"], loadings["contrast_id"], loadings["entity_id"]))
    selected_loadings = loadings.loc[[key in eligible_keys for key in loading_key]].copy()
    query_key = list(zip(links["query_study_id"], links["query_contrast_id"], links["query_entity_id"]))
    target_key = list(zip(links["target_study_id"], links["target_contrast_id"], links["target_entity_id"]))
    selected_links = links.loc[
        [query in eligible_keys and target in eligible_keys for query, target in zip(query_key, target_key)]
    ].copy()
    return selected_effects, selected_loadings, selected_links


def matched_known_keys(
    effects: pd.DataFrame,
    covariates: pd.DataFrame,
    covariate_columns: list[str],
) -> tuple[set[tuple[str, str, str]], pd.DataFrame]:
    joined = effects[["study_id", "contrast_id", "entity_id", "identity_status"]].merge(
        covariates,
        on=["study_id", "contrast_id", "entity_id"],
        how="left",
        validate="one_to_one",
    )
    if joined[covariate_columns].isna().any().any():
        raise RuntimeError("eligible effect entity lacks measurement covariates")
    selected: set[tuple[str, str, str]] = set()
    ledger = []
    for (study, contrast), group in joined.groupby(["study_id", "contrast_id"], sort=False):
        known = group.loc[group["identity_status"].isin(KNOWN)].copy()
        unresolved = group.loc[group["identity_status"].eq("unresolved_entity")].copy()
        if known.empty or unresolved.empty:
            continue
        pooled = pd.concat([known[covariate_columns], unresolved[covariate_columns]])
        scale = pooled.std(axis=0, ddof=1).replace(0, 1.0).fillna(1.0)
        center = pooled.mean(axis=0)
        known_values = ((known[covariate_columns] - center) / scale).to_numpy(float)
        unresolved_values = ((unresolved[covariate_columns] - center) / scale).to_numpy(float)
        pairs = []
        for unresolved_index in range(len(unresolved)):
            distances = np.sqrt(np.sum((known_values - unresolved_values[unresolved_index]) ** 2, axis=1))
            for known_index, distance in enumerate(distances):
                pairs.append((float(distance), known_index, unresolved_index))
        used_known: set[int] = set()
        used_unresolved: set[int] = set()
        for distance, known_index, unresolved_index in sorted(pairs):
            if known_index in used_known or unresolved_index in used_unresolved:
                continue
            used_known.add(known_index)
            used_unresolved.add(unresolved_index)
            known_row = known.iloc[known_index]
            unresolved_row = unresolved.iloc[unresolved_index]
            selected.add((str(study), str(contrast), str(known_row.entity_id)))
            ledger.append({
                "study_id": study, "contrast_id": contrast,
                "known_entity_id": known_row.entity_id,
                "unresolved_entity_id": unresolved_row.entity_id,
                "standardized_distance": distance,
            })
            if len(used_known) == min(len(known), len(unresolved)):
                break
    return selected, pd.DataFrame(ledger)


def shuffled_unresolved_loadings(
    effects: pd.DataFrame,
    loadings: pd.DataFrame,
    covariates: pd.DataFrame,
    strata_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = effects[["study_id", "contrast_id", "entity_id", "identity_status"]]
    annotated = loadings.merge(identity, on=["study_id", "contrast_id", "entity_id"], how="left", validate="many_to_one")
    annotated = annotated.merge(
        covariates[["study_id", "contrast_id", "entity_id", *strata_columns]],
        on=["study_id", "contrast_id", "entity_id"],
        how="left",
        validate="many_to_one",
    )
    if annotated["identity_status"].isna().any() or annotated[strata_columns].isna().any().any():
        raise RuntimeError("pattern loading lacks identity or shuffle stratum")
    unresolved = annotated["identity_status"].eq("unresolved_entity")
    rng = np.random.default_rng(seed)
    ledger = []
    output = annotated.copy()
    group_columns = ["study_id", "contrast_id", *strata_columns]
    for group_key, group in annotated.loc[unresolved].groupby(group_columns, sort=True, dropna=False):
        entities = sorted(group["entity_id"].unique())
        if len(entities) < 2:
            raise RuntimeError(f"unresolved MS2 shuffle stratum has fewer than two entities: {group_key}")
        shift = int(rng.integers(1, len(entities)))
        targets = entities[shift:] + entities[:shift]
        mapping = dict(zip(entities, targets))
        mask = unresolved.copy()
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        for column, value in zip(group_columns, key_values):
            mask &= output[column].eq(value)
        output.loc[mask, "entity_id"] = output.loc[mask, "entity_id"].map(mapping)
        ledger.extend({
            "study_id": key_values[0], "contrast_id": key_values[1],
            "source_entity_id": source, "target_entity_id": target,
            "stratum": "|".join(str(value) for value in key_values[2:]),
        } for source, target in mapping.items())
    keep = list(loadings.columns)
    shuffled = output[keep].copy()
    if shuffled.duplicated(["study_id", "contrast_id", "entity_id", "pattern_id"]).any():
        raise RuntimeError("shuffled loadings create duplicate entity-pattern rows")
    return shuffled, pd.DataFrame(ledger)


def filter_unit_effects(
    unit_effects: pd.DataFrame,
    eligible_keys: set[tuple[str, str, str]],
) -> pd.DataFrame:
    keys = list(zip(
        unit_effects["study_id"], unit_effects["contrast_id"], unit_effects["entity_id"],
    ))
    return unit_effects.loc[[key in eligible_keys for key in keys]].copy()


def write_world(
    path: Path,
    effects: pd.DataFrame,
    loadings: pd.DataFrame,
    links: pd.DataFrame,
    unit_effects: pd.DataFrame | None = None,
) -> None:
    path.mkdir(parents=True)
    effects.to_csv(path / "contrast_records.csv.gz", index=False, compression="gzip")
    loadings.to_csv(path / "pattern_loadings.csv.gz", index=False, compression="gzip")
    links.to_csv(path / "strict_links.csv.gz", index=False, compression="gzip")
    if unit_effects is not None:
        unit_effects.to_csv(path / "contrast_unit_effects.csv.gz", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast-records", type=Path, required=True)
    parser.add_argument("--pattern-loadings", type=Path, required=True)
    parser.add_argument("--strict-links", type=Path, required=True)
    parser.add_argument("--measurement-covariates", type=Path, required=True)
    parser.add_argument("--contrast-unit-effects", type=Path)
    parser.add_argument("--covariate-columns", required=True)
    parser.add_argument("--shuffle-strata", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    covariate_columns = [item.strip() for item in args.covariate_columns.split(",") if item.strip()]
    strata_columns = [item.strip() for item in args.shuffle_strata.split(",") if item.strip()]
    if not covariate_columns or not strata_columns:
        raise ValueError("covariate columns and shuffle strata are required")

    effects = validate_effects(pd.read_csv(args.contrast_records))
    unit_effects = (
        validate_unit_effects(pd.read_csv(args.contrast_unit_effects), effects)
        if args.contrast_unit_effects is not None else None
    )
    loadings = validate_loadings(pd.read_csv(args.pattern_loadings))
    links = validate_links(pd.read_csv(args.strict_links))
    covariates = validate_covariates(pd.read_csv(args.measurement_covariates), covariate_columns, strata_columns)
    all_keys = set(zip(effects["study_id"].astype(str), effects["contrast_id"].astype(str), effects["entity_id"].astype(str)))
    known_keys = set(
        (str(row.study_id), str(row.contrast_id), str(row.entity_id))
        for row in effects.loc[effects["identity_status"].isin(KNOWN)].itertuples(index=False)
    )
    matched_keys, match_ledger = matched_known_keys(effects, covariates, covariate_columns)
    if not matched_keys:
        raise RuntimeError("no measurement-matched known entities were constructed")
    shuffled_loadings, shuffle_ledger = shuffled_unresolved_loadings(
        effects, loadings, covariates, strata_columns, args.seed,
    )

    worlds = {}
    worlds["all"] = (effects.copy(), loadings.copy(), links.copy(), unit_effects)
    known_tables = filter_world(effects, loadings, links, known_keys)
    matched_tables = filter_world(effects, loadings, links, matched_keys)
    worlds["known_only"] = (*known_tables, filter_unit_effects(unit_effects, known_keys) if unit_effects is not None else None)
    worlds["measurement_matched_known"] = (*matched_tables, filter_unit_effects(unit_effects, matched_keys) if unit_effects is not None else None)
    worlds["unresolved_shuffled_ms2"] = (effects.copy(), shuffled_loadings, links.copy(), unit_effects)
    args.output_dir.mkdir(parents=True)
    for name, tables in worlds.items():
        write_world(args.output_dir / name, *tables)
    match_ledger.to_csv(args.output_dir / "measurement_match_ledger.csv", index=False)
    shuffle_ledger.to_csv(args.output_dir / "unresolved_shuffle_ledger.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_CONTROL_WORLDS_COMPLETE",
        "formal": True,
        "seed": args.seed,
        "covariate_columns": covariate_columns,
        "shuffle_strata": strata_columns,
        "input_sha256": {
            "contrast_records": sha256(args.contrast_records),
            "pattern_loadings": sha256(args.pattern_loadings),
            "strict_links": sha256(args.strict_links),
            "measurement_covariates": sha256(args.measurement_covariates),
        },
        "world_rows": {
            name: {
                "effects": int(len(tables[0])),
                "loadings": int(len(tables[1])),
                "links": int(len(tables[2])),
                "unit_effects": int(len(tables[3])) if tables[3] is not None else 0,
            }
            for name, tables in worlds.items()
        },
        "measurement_matches": int(len(match_ledger)),
        "shuffled_unresolved_entities": int(len(shuffle_ledger)),
        "claim_limit": "Phenotype-blind control construction only; no retrieval increment is established.",
    }
    if args.contrast_unit_effects is not None:
        report["input_sha256"]["contrast_unit_effects"] = sha256(args.contrast_unit_effects)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate a standardized response object from a public FBMN feature table.

The design is explicit and study agnostic.  Technical files are first collapsed
within biological units.  A contrast may use two or more levels through frozen
coefficients that sum to zero, for example co-culture - 0.5*A - 0.5*B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sample_column(column: str) -> str:
    value = str(column).strip()
    for suffix in (" Peak area", " peak area", " Area", " area"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    return value


def normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.full(values.shape, np.nan, dtype=float)
    positive = np.isfinite(values) & (values > 0)
    transformed[positive] = np.log2(values[positive])
    offsets = np.nanmedian(transformed, axis=0)
    if np.any(~np.isfinite(offsets)):
        raise ValueError("at least one design sample has no positive feature intensity")
    return transformed - offsets[None, :], offsets


def collapse_units(
    normalized: np.ndarray, design: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    unit_level_count = design.groupby("biological_unit_id")["level"].nunique()
    crossing = unit_level_count.loc[unit_level_count > 1]
    if len(crossing):
        raise ValueError(f"{len(crossing)} biological units cross contrast levels")
    unit_columns = ["biological_unit_id", "level"]
    if "pair_id" in design:
        pair_count = design.groupby("biological_unit_id")["pair_id"].nunique()
        if (pair_count > 1).any():
            raise ValueError("a biological unit maps to multiple pair IDs")
        unit_columns.append("pair_id")
    units = design[unit_columns].drop_duplicates().sort_values("biological_unit_id")
    columns: list[np.ndarray] = []
    for unit in units.biological_unit_id:
        indices = np.flatnonzero(design.biological_unit_id.to_numpy(str) == str(unit))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            columns.append(np.nanmedian(normalized[:, indices], axis=1))
    return np.column_stack(columns), units.reset_index(drop=True)


def group_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    n = finite.sum(axis=1).astype(int)
    mean = np.divide(
        np.nansum(values, axis=1), n,
        out=np.full(values.shape[0], np.nan), where=n > 0,
    )
    centered = np.where(finite, values - mean[:, None], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), n - 1,
        out=np.full(values.shape[0], np.nan), where=n > 1,
    )
    return n, mean, variance


def estimate(
    unit_values: np.ndarray,
    unit_levels: np.ndarray,
    coefficients: dict[str, float],
    minimum_units: int,
    minimum_prevalence: float,
) -> pd.DataFrame:
    beta = np.zeros(unit_values.shape[0], dtype=float)
    se_squared = np.zeros(unit_values.shape[0], dtype=float)
    eligible = np.ones(unit_values.shape[0], dtype=bool)
    counts: dict[str, np.ndarray] = {}
    for level, coefficient in coefficients.items():
        n, mean, variance = group_stats(unit_values[:, unit_levels == level])
        beta += coefficient * mean
        se_squared += coefficient * coefficient * np.divide(
            variance, n, out=np.full(len(n), np.nan), where=n > 0,
        )
        eligible &= n >= minimum_units
        counts[level] = n
    se = np.sqrt(se_squared)
    prevalence = np.isfinite(unit_values).mean(axis=1)
    eligible &= (
        np.isfinite(beta) & np.isfinite(se) & (se > 0)
        & (prevalence >= minimum_prevalence)
    )
    result = pd.DataFrame({
        "beta": beta, "se": se, "prevalence": prevalence, "qc_pass": eligible,
    })
    for level, n in counts.items():
        result[f"n_measured__{level}"] = n
    return result


def estimate_paired(
    unit_values: np.ndarray,
    units: pd.DataFrame,
    coefficients: dict[str, float],
    minimum_pairs: int,
    minimum_prevalence: float,
) -> pd.DataFrame:
    """Estimate a paired two-level contrast without treating subjects as independent."""
    if "pair_id" not in units:
        raise ValueError("paired analysis requires pair_id in the design")
    if len(coefficients) != 2 or sorted(coefficients.values()) != [-1.0, 1.0]:
        raise ValueError("paired analysis requires exactly one +1 and one -1 level")
    positive = next(level for level, value in coefficients.items() if value == 1.0)
    negative = next(level for level, value in coefficients.items() if value == -1.0)
    if units.groupby(["pair_id", "level"]).size().gt(1).any():
        raise ValueError("paired analysis requires one biological unit per pair and level")
    positive_units = units.loc[units.level.eq(positive), ["biological_unit_id", "pair_id"]]
    negative_units = units.loc[units.level.eq(negative), ["biological_unit_id", "pair_id"]]
    pairs = positive_units.merge(negative_units, on="pair_id", suffixes=("__positive", "__negative"))
    if len(pairs) < minimum_pairs:
        raise ValueError(f"only {len(pairs)} complete design pairs; need {minimum_pairs}")
    index = {str(unit): offset for offset, unit in enumerate(units.biological_unit_id)}
    positive_index = [index[str(unit)] for unit in pairs.biological_unit_id__positive]
    negative_index = [index[str(unit)] for unit in pairs.biological_unit_id__negative]
    differences = unit_values[:, positive_index] - unit_values[:, negative_index]
    finite = np.isfinite(differences)
    n = finite.sum(axis=1).astype(int)
    beta = np.divide(
        np.nansum(differences, axis=1), n,
        out=np.full(unit_values.shape[0], np.nan), where=n > 0,
    )
    centered = np.where(finite, differences - beta[:, None], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), n - 1,
        out=np.full(unit_values.shape[0], np.nan), where=n > 1,
    )
    se = np.sqrt(np.divide(
        variance, n, out=np.full(unit_values.shape[0], np.nan), where=n > 0,
    ))
    prevalence = n / len(pairs)
    eligible = (
        (n >= minimum_pairs) & np.isfinite(beta) & np.isfinite(se) & (se > 0)
        & (prevalence >= minimum_prevalence)
    )
    return pd.DataFrame({
        "beta": beta,
        "se": se,
        "prevalence": prevalence,
        "qc_pass": eligible,
        "n_measured_pairs": n,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantification", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--contrast", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-units-per-level", type=int, default=3)
    parser.add_argument("--minimum-prevalence", type=float, default=0.20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contrast = json.loads(args.contrast.read_text(encoding="utf-8"))
    coefficients = {str(k): float(v) for k, v in contrast["level_coefficients"].items()}
    if len(coefficients) < 2 or not np.isclose(sum(coefficients.values()), 0.0):
        raise ValueError("contrast needs at least two level coefficients summing to zero")
    if not any(value > 0 for value in coefficients.values()) or not any(value < 0 for value in coefficients.values()):
        raise ValueError("contrast must contain positive and negative coefficients")

    quant = pd.read_csv(args.quantification)
    design = pd.read_csv(args.design, sep="\t", dtype=str, keep_default_na=False)
    required_design = {"filename", "biological_unit_id", "level"}
    if missing := required_design - set(design):
        raise ValueError(f"design lacks {sorted(missing)}")
    if design.filename.duplicated().any():
        raise ValueError("design filenames must be unique")
    unknown_levels = sorted(set(design.level) - set(coefficients))
    if unknown_levels:
        raise ValueError(f"design includes levels absent from contrast: {unknown_levels}")
    missing_levels = sorted(set(coefficients) - set(design.level))
    if missing_levels:
        raise ValueError(f"contrast levels absent from design: {missing_levels}")

    quant_lookup: dict[str, str] = {}
    for column in quant.columns:
        canonical = canonical_sample_column(column)
        if canonical in quant_lookup:
            raise ValueError(f"duplicate canonical sample column: {canonical}")
        quant_lookup[canonical] = column
    if missing := sorted(set(design.filename) - set(quant_lookup)):
        raise ValueError(f"design samples absent from quantification table: {missing[:5]}")
    sample_columns = [quant_lookup[value] for value in design.filename]
    values = quant[sample_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    normalized, offsets = normalize(values)
    unit_values, units = collapse_units(normalized, design)
    analysis = str(contrast.get("analysis", "independent_groups"))
    if analysis == "paired_difference":
        effects = estimate_paired(
            unit_values, units, coefficients,
            args.minimum_units_per_level, args.minimum_prevalence,
        )
    elif analysis == "independent_groups":
        effects = estimate(
            unit_values, units.level.to_numpy(str), coefficients,
            args.minimum_units_per_level, args.minimum_prevalence,
        )
    else:
        raise ValueError(f"unsupported analysis: {analysis}")
    feature_id_column = "row ID" if "row ID" in quant else quant.columns[0]
    effects.insert(0, "entity_id", quant[feature_id_column].astype(str))
    effects["identity_status"] = "unpartitioned"
    args.output_dir.mkdir(parents=True)
    effects.to_csv(args.output_dir / "response_object.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame({"filename": design.filename, "log2_median_offset": offsets}).to_csv(
        args.output_dir / "sample_normalization.tsv", sep="\t", index=False
    )
    report = {
        "schema": "public_fbmn_response_object_v1",
        "status": "response_object_estimated_not_cross_study_evidence",
        "object_id": contrast["object_id"],
        "study_id": contrast["study_id"],
        "analysis": analysis,
        "level_coefficients": coefficients,
        "denominators": {
            "quantified_features": int(len(effects)),
            "design_files": int(len(design)),
            "biological_units": int(len(units)),
            "eligible_response_entities": int(effects.qc_pass.sum()),
            "units_per_level": units.groupby("level").size().astype(int).to_dict(),
            "complete_pairs": (
                int(units.groupby("pair_id")["level"].nunique().eq(2).sum())
                if analysis == "paired_difference" else None
            ),
        },
        "measurement_contract": {
            "normalization": "positive-only log2 then sample-median centering before contrast labels are used",
            "technical_replicates": "median within explicit biological_unit_id",
            "missingness": "zero/nonpositive remains missing; no imputation",
            "effect": (
                "mean within-pair difference with standard error across complete pairs"
                if analysis == "paired_difference"
                else "linear contrast of group means with independent-group Welch standard error"
            ),
            "identity_status": "unpartitioned until a frozen library-coverage search is supplied",
        },
        "provenance": {
            "quantification_sha256": sha256(args.quantification),
            "design_sha256": sha256(args.design),
            "contrast_sha256": sha256(args.contrast),
        },
        "claim_boundary": [
            "The design file must encode true biological units; filenames are never silently treated as replicates.",
            "An estimated response object is not a validated biological relation.",
            "Unresolved-only analysis is forbidden until identity/search coverage is completely partitioned.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

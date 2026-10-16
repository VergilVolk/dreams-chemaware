#!/usr/bin/env python3
"""Estimate unit-level effects after a Pan-ReDU abundance panel is frozen.

Normalization is performed across the complete phenotype-blind panel before
the sealed membership table is opened.  Repeated files are collapsed within a
hashed biological unit, so files never become pseudo-replicates.  Contrasts in
which one unit crosses levels fail closed.
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


def normalize_positive_log2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive-only log2 followed by outcome-blind sample median centering."""
    transformed = np.full(values.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(values) & (values > 0)
    transformed[positive] = np.log2(values[positive])
    offsets = np.nanmedian(transformed, axis=0)
    if not np.isfinite(offsets).all():
        bad = np.flatnonzero(~np.isfinite(offsets)).tolist()
        raise RuntimeError(f"samples with no positive anchor measurement: {bad[:10]}")
    transformed -= offsets[None, :]
    return transformed, offsets


def aggregate_units(
    normalized: np.ndarray, sample_ids: np.ndarray, membership: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_index = {str(value): index for index, value in enumerate(sample_ids)}
    if unknown := sorted(set(membership["sample_id"]) - set(sample_index)):
        raise ValueError(f"membership samples absent from frozen matrix: {unknown[:5]}")
    unit_levels = membership.groupby("biological_unit_id")["level"].nunique()
    crossing = unit_levels[unit_levels > 1]
    if len(crossing):
        raise ValueError(f"{len(crossing)} biological units cross contrast levels")
    units = membership[["biological_unit_id", "level"]].drop_duplicates().sort_values("biological_unit_id")
    columns: list[np.ndarray] = []
    for unit in units["biological_unit_id"]:
        ids = membership.loc[membership["biological_unit_id"].eq(unit), "sample_id"].drop_duplicates()
        indices = [sample_index[str(value)] for value in ids]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            columns.append(np.nanmedian(normalized[:, indices], axis=1))
    return np.column_stack(columns), units["biological_unit_id"].to_numpy(str), units["level"].to_numpy(str)


def group_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    n = finite.sum(axis=1).astype(np.int32)
    total = np.nansum(values, axis=1)
    mean = np.divide(total, n, out=np.full(len(values), np.nan), where=n > 0)
    centered = np.where(finite, values - mean[:, None], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), n - 1,
        out=np.full(len(values), np.nan), where=n > 1,
    )
    return n, mean, variance


def estimate_effects(
    unit_values: np.ndarray,
    unit_levels: np.ndarray,
    level_a: str,
    level_b: str,
    minimum_group_measurements: int,
    minimum_prevalence: float,
) -> pd.DataFrame:
    mask_a, mask_b = unit_levels == level_a, unit_levels == level_b
    if mask_a.sum() < minimum_group_measurements or mask_b.sum() < minimum_group_measurements:
        raise ValueError(
            f"too few biological units: {level_a}={mask_a.sum()}, {level_b}={mask_b.sum()}"
        )
    n_a, mean_a, variance_a = group_stats(unit_values[:, mask_a])
    n_b, mean_b, variance_b = group_stats(unit_values[:, mask_b])
    beta = mean_b - mean_a
    mean_variance_a = np.divide(
        variance_a, n_a, out=np.full(len(n_a), np.nan), where=n_a > 0,
    )
    mean_variance_b = np.divide(
        variance_b, n_b, out=np.full(len(n_b), np.nan), where=n_b > 0,
    )
    se = np.sqrt(mean_variance_a + mean_variance_b)
    prevalence = np.isfinite(unit_values).mean(axis=1)
    eligible = (
        (n_a >= minimum_group_measurements)
        & (n_b >= minimum_group_measurements)
        & (prevalence >= minimum_prevalence)
        & np.isfinite(beta)
        & np.isfinite(se)
        & (se > 0)
    )
    return pd.DataFrame({
        "beta_b_minus_a": beta,
        "se": se,
        "n_measured_a": n_a,
        "n_measured_b": n_b,
        "unit_prevalence": prevalence,
        "passes_effect_qc": eligible,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument("--memberships", required=True, type=Path)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-group-measurements", type=int, default=3)
    parser.add_argument("--minimum-prevalence", type=float, default=0.20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    object_report_path = args.panel_object / "report.json"
    object_report = json.loads(object_report_path.read_text(encoding="utf-8"))
    if object_report.get("status") != "development_consumed_abundance_matrix_frozen":
        raise RuntimeError("panel abundance matrix is not frozen")
    with np.load(args.panel_object / "ms1_peak_height.npz", allow_pickle=False) as packed:
        values = np.asarray(packed["values"], dtype=np.float64)
        anchor_ids = np.asarray(packed["anchor_id"], dtype=str)
        sample_ids = np.asarray(packed["sample_id"], dtype=str)
    anchors = pd.read_csv(args.panel_object / "anchor_manifest.tsv", sep="\t")
    samples = pd.read_csv(args.panel_object / "sample_manifest.tsv", sep="\t", dtype=str)
    if values.shape != (len(anchor_ids), len(sample_ids)):
        raise RuntimeError("invalid matrix dimensions")
    if not np.array_equal(anchor_ids, anchors["anchor_id"].to_numpy(str)):
        raise RuntimeError("anchor order mismatch")
    if not np.array_equal(sample_ids, samples["sample_id"].to_numpy(str)):
        raise RuntimeError("sample order mismatch")

    normalized, offsets = normalize_positive_log2(values)
    memberships = pd.read_csv(args.memberships, sep="\t", dtype=str, keep_default_na=False)
    memberships = memberships.loc[memberships["panel_id"].eq(args.panel_id)].copy()
    if memberships.empty:
        raise ValueError(f"no sealed memberships for panel {args.panel_id}")
    args.output_dir.mkdir(parents=True)
    effect_tables: list[pd.DataFrame] = []
    decisions: list[dict[str, object]] = []
    for contrast_id, block in memberships.groupby("contrast_id", sort=True):
        fields = block[["contrast_field", "level_a", "level_b"]].drop_duplicates()
        if len(fields) != 1:
            raise ValueError(f"inconsistent contrast definition: {contrast_id}")
        definition = fields.iloc[0]
        try:
            unit_values, unit_ids, unit_levels = aggregate_units(normalized, sample_ids, block)
            effects = estimate_effects(
                unit_values, unit_levels, str(definition.level_a), str(definition.level_b),
                args.minimum_group_measurements, args.minimum_prevalence,
            )
        except ValueError as error:
            decisions.append({
                "contrast_id": contrast_id,
                "contrast_field": str(definition.contrast_field),
                "level_a": str(definition.level_a),
                "level_b": str(definition.level_b),
                "status": "FAIL_CLOSED_DESIGN",
                "reason": str(error),
            })
            continue
        effects.insert(0, "anchor_id", anchor_ids)
        effects.insert(1, "precursor_mz", anchors["precursor_mz"].to_numpy(float))
        effects.insert(2, "rt_seconds", anchors["rt_seconds"].to_numpy(float))
        effects.insert(0, "contrast_id", contrast_id)
        effects.insert(1, "contrast_field", str(definition.contrast_field))
        effects.insert(2, "level_a", str(definition.level_a))
        effects.insert(3, "level_b", str(definition.level_b))
        effect_tables.append(effects)
        unique, counts = np.unique(unit_levels, return_counts=True)
        decisions.append({
            "contrast_id": contrast_id,
            "contrast_field": str(definition.contrast_field),
            "level_a": str(definition.level_a),
            "level_b": str(definition.level_b),
            "status": "DEVELOPMENT_EFFECT_OBJECT_BUILT",
            "biological_units": dict(zip(unique.tolist(), counts.astype(int).tolist())),
            "eligible_anchors": int(effects["passes_effect_qc"].sum()),
            "total_anchors": len(effects),
        })
    combined = pd.concat(effect_tables, ignore_index=True) if effect_tables else pd.DataFrame()
    combined.to_csv(args.output_dir / "contrast_effects.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame({"sample_id": sample_ids, "positive_log2_median_offset": offsets}).to_csv(
        args.output_dir / "sample_normalization.tsv", sep="\t", index=False
    )
    pd.DataFrame(decisions).to_json(args.output_dir / "contrast_decisions.jsonl", orient="records", lines=True)
    passed = sum(row["status"] == "DEVELOPMENT_EFFECT_OBJECT_BUILT" for row in decisions)
    report = {
        "schema": "pan_redu_panel_contrast_effects_v1",
        "status": "development_effect_objects_not_cross_study_evidence",
        "panel_id": args.panel_id,
        "denominators": {
            "input_anchors": len(anchor_ids),
            "input_files": len(sample_ids),
            "candidate_contrasts": len(decisions),
            "effect_objects_built": passed,
            "contrasts_failed_closed": len(decisions) - passed,
            "effect_rows": len(combined),
            "eligible_effect_rows": int(combined["passes_effect_qc"].sum()) if len(combined) else 0,
        },
        "measurement_contract": {
            "normalization": "positive-only log2 then per-sample median centering across the complete phenotype-blind panel",
            "zero_handling": "zero peak height remains missing; no pseudocount imputation",
            "replicate_handling": "median within hashed biological unit before effect estimation",
            "direction": "level_b minus level_a",
            "minimum_group_measurements_per_anchor": args.minimum_group_measurements,
            "minimum_unit_prevalence": args.minimum_prevalence,
        },
        "leakage_status": "normalization fixed before sealed contrast membership opened; no cross-study truth read",
        "claim_boundary": [
            "These are unadjusted development effects, not evidence that Pan-ReDU ontology labels define comparable biology.",
            "Contrasts with units crossing levels fail closed.",
            "Covariate adjustment and design-native semantic adjudication are required before claim-bearing retrieval.",
        ],
        "input_sha256": {
            "panel_report": sha256(object_report_path),
            "matrix": sha256(args.panel_object / "ms1_peak_height.npz"),
            "anchor_manifest": sha256(args.panel_object / "anchor_manifest.tsv"),
            "sample_manifest": sha256(args.panel_object / "sample_manifest.tsv"),
            "sealed_memberships": sha256(args.memberships),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate design-native 2x2 genotype-by-diet response objects with HC3 SEs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from estimate_pan_redu_panel_contrasts import normalize_positive_log2


CONTRASTS = {
    "diet_reference": np.array([0.0, 0.0, 1.0, 0.0]),
    "diet_modified": np.array([0.0, 0.0, 1.0, 1.0]),
    "genotype_control": np.array([0.0, 1.0, 0.0, 0.0]),
    "genotype_hfd": np.array([0.0, 1.0, 0.0, 1.0]),
    "genotype_by_hfd": np.array([0.0, 0.0, 0.0, 1.0]),
}


def fit_feature_hc3(y: np.ndarray, genotype: np.ndarray, diet: np.ndarray, minimum_cell_n: int) -> dict[str, tuple[float, float]] | None:
    finite = np.isfinite(y)
    g, d, observed = genotype[finite], diet[finite], y[finite]
    cells = np.array([np.sum((g == gv) & (d == dv)) for gv in (0, 1) for dv in (0, 1)])
    if len(observed) < 8 or cells.min() < minimum_cell_n:
        return None
    x = np.column_stack([np.ones(len(observed)), g, d, g * d]).astype(float)
    if np.linalg.matrix_rank(x) != 4:
        return None
    xtx_inv = np.linalg.inv(x.T @ x)
    coefficients = xtx_inv @ x.T @ observed
    residual = observed - x @ coefficients
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(leverage >= 1 - 1e-10):
        return None
    adjusted = residual / (1 - leverage)
    meat = x.T @ (x * (adjusted * adjusted)[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    output = {}
    for name, contrast in CONTRASTS.items():
        estimate = float(contrast @ coefficients)
        variance = float(contrast @ covariance @ contrast)
        if not np.isfinite(variance) or variance <= 0:
            return None
        output[name] = (estimate, float(np.sqrt(variance)))
    return output


def estimate_matrix(values: np.ndarray, genotype: np.ndarray, diet: np.ndarray, minimum_cell_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_index, y in enumerate(values):
        fitted = fit_feature_hc3(y, genotype, diet, minimum_cell_n)
        for contrast_name in CONTRASTS:
            estimate, se = (np.nan, np.nan) if fitted is None else fitted[contrast_name]
            rows.append({
                "feature_index": feature_index, "contrast_type": contrast_name,
                "beta": estimate, "se_hc3": se, "passes_effect_qc": fitted is not None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument("--sealed-design", required=True, type=Path)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-cell-n", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = json.loads((args.panel_object / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "development_consumed_abundance_matrix_frozen":
        raise RuntimeError("panel abundance matrix is not frozen")
    with np.load(args.panel_object / "ms1_peak_height.npz", allow_pickle=False) as packed:
        raw = np.asarray(packed["values"], dtype=float)
        anchor_id = np.asarray(packed["anchor_id"], dtype=str)
        sample_id = np.asarray(packed["sample_id"], dtype=str)
    normalized, offsets = normalize_positive_log2(raw)
    design = pd.read_csv(args.sealed_design, sep="\t", dtype={"genotype_code": int, "diet_code": int})
    design = design.loc[design.panel_id.astype(str).eq(args.panel_id)].copy()
    if design.empty or design.sample_id.duplicated().any():
        raise ValueError("missing or duplicate panel design")
    order = design.set_index("sample_id").reindex(sample_id)
    if order[["genotype_code", "diet_code"]].isna().any().any():
        raise ValueError("frozen matrix and sealed design do not align")
    genotype = order.genotype_code.to_numpy(int)
    diet = order.diet_code.to_numpy(int)
    effects = estimate_matrix(normalized, genotype, diet, args.minimum_cell_n)
    effects.insert(0, "anchor_id", np.repeat(anchor_id, len(CONTRASTS)))
    effects.insert(0, "panel_id", args.panel_id)
    prevalence = pd.Series((raw > 0).mean(axis=1), index=anchor_id)
    effects["unit_prevalence"] = effects["anchor_id"].map(prevalence).astype(float)
    args.output_dir.mkdir(parents=True)
    effects.to_csv(args.output_dir / "factorial_contrast_effects.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame({"sample_id": sample_id, "positive_log2_median_offset": offsets}).to_csv(
        args.output_dir / "sample_normalization.tsv", sep="\t", index=False
    )
    cell_counts = order.groupby(["genotype_code", "diet_code"]).size().to_dict()
    summary = {
        "schema": "factorial_hfd_panel_effects_v1",
        "status": "development_factorial_effect_objects_not_cross_study_evidence",
        "panel_id": args.panel_id,
        "denominators": {
            "biological_units": len(sample_id), "anchors": len(anchor_id),
            "cell_counts": {f"g{g}_d{d}": int(n) for (g, d), n in cell_counts.items()},
            "eligible_anchors_all_five_contrasts": int(
                effects.groupby("anchor_id")["passes_effect_qc"].all().sum()
            ),
        },
        "model": "intercept + genotype + HFD + genotype:HFD with feature-wise HC3 covariance",
        "direction": "modified-minus-reference and HFD-minus-control according to sealed 0/1 codes",
        "leakage_status": "normalization frozen before design opened; no cross-study truth read",
        "claim_boundary": "Effect objects only; cross-study similarity and mechanism remain untested.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

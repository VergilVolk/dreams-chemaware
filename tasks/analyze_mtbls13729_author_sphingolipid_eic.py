#!/usr/bin/env python
"""Recompute the source-paper sphingolipid claims from uniform raw-data EICs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from analyze_mtbls13729_full_space_eic import full_background_pqn, interaction
from audit_mtbls13729_full_requantifiable_space import paired_summary, sample_pairs


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data/mtbls13729/author_sphingolipid_targets_v2"
EIC = ROOT / "data/mtbls13729/author_sphingolipid_eic_v2"
OUT = ROOT / "data/mtbls13729/author_sphingolipid_audit_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    panel_report: dict[str, object] = {}
    for panel in ("neg_rp", "pos_rp", "pos_hilic"):
        target_path = TARGETS / f"{panel}__requantification_targets.csv.gz"
        targets = pd.read_csv(target_path).set_index("feature_id")
        auc = pd.read_csv(EIC / f"{panel}__eic_auc_matrix.csv.gz").set_index("feature_id")
        detected = pd.read_csv(EIC / f"{panel}__eic_detection_matrix.csv.gz").set_index("feature_id")
        detected = detected.apply(lambda x: x.astype(str).str.lower().isin({"true", "1"}))
        if set(targets.index) != set(auc.index):
            raise RuntimeError(f"{panel}: target/EIC mismatch")
        auc = auc.loc[targets.index].where(detected.loc[targets.index] & (auc.loc[targets.index] > 0))
        samples = list(auc.columns)
        positive = auc.stack().to_numpy(float)
        pseudo = float(np.percentile(positive, 1) / 2.0) if len(positive) else 1.0
        raw = np.log2(auc + pseudo)
        matrices = {"log_raw": raw}
        background_path = ROOT / f"data/mtbls13729/ms1_consensus/{panel}__discovery_intensity_matrix.csv.gz"
        if background_path.exists():
            factors = full_background_pqn(panel, samples)
            matrices["global_pqn_prev60"] = raw.sub(factors, axis=1)
        rmu_pairs = sample_pairs(samples, "Rmu", "RN")
        rtu_pairs = sample_pairs(samples, "Rtu", "RN")
        result = targets.copy()
        result["eic_detection_fraction"] = detected.loc[result.index].mean(axis=1)
        for name, matrix in matrices.items():
            primary = paired_summary(matrix, rmu_pairs, "rmu")
            secondary = interaction(matrix, rmu_pairs, rtu_pairs)
            for column in primary.columns:
                result[f"{name}__{column}"] = primary[column]
            for column in secondary.columns:
                result[f"{name}__{column}"] = secondary[column]
        effect_columns = [f"{name}__rmu_mean_log2fc" for name in matrices]
        exact_columns = [f"{name}__rmu_exact_signflip_p" for name in matrices]
        loo_columns = [f"{name}__rmu_loo_direction_stable" for name in matrices]
        result["direction_consistent"] = np.sign(result[effect_columns]).nunique(axis=1, dropna=True).eq(1)
        result["min_abs_rmu_log2fc"] = result[effect_columns].abs().min(axis=1)
        result["max_exact_signflip_p"] = result[exact_columns].max(axis=1)
        result["loo_stable_all"] = result[loo_columns].fillna(False).all(axis=1)
        result["raw_reextraction_gate"] = (
            result["eic_detection_fraction"].ge(0.80)
            & result["direction_consistent"]
            & result["min_abs_rmu_log2fc"].ge(0.50)
            & result["max_exact_signflip_p"].le(0.05)
            & result["loo_stable_all"]
        )
        result = result.reset_index()
        result.insert(0, "panel", panel)
        result.to_csv(OUT / f"{panel}__author_sphingolipid_eic_audit.csv", index=False)
        rows.append(result)
        panel_report[panel] = {
            "targets": int(len(result)),
            "raw_reextraction_gate_pass": int(result.raw_reextraction_gate.sum()),
            "median_detection_fraction": float(result.eic_detection_fraction.median()),
            "normalizations": list(matrices),
        }
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(OUT / "author_sphingolipid_eic_audit.csv", index=False)
    report = {
        "status": "mtbls13729_author_sphingolipid_raw_eic_audit_complete",
        "formal": False,
        "panels": panel_report,
        "primary_endpoint": "paired Rmu versus patient-matched normal abundance",
        "secondary_endpoint": "Rmu-RN versus Rtu-RN interaction",
        "targets_sha256": sha256(TARGETS / "author_sphingolipid_panel.csv"),
        "claim_limit": (
            "This is a same-cohort raw-data re-extraction of source-paper coordinates. "
            "It does not independently confirm metabolite identity, subtype specificity, flux, enzyme activity, or causality."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

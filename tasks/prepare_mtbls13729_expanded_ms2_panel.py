#!/usr/bin/env python
"""Freeze technically retained full-space candidates for raw-MS2 linkage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mtbls13729/full_space_eic_analysis_v1/all_targeted_eic_results.csv"
OUT = ROOT / "data/mtbls13729/expanded_ms2_panel_v1"


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
    frame = pd.read_csv(SOURCE)
    selected = frame[
        frame["technical_retention_gate"].fillna(False)
        & (~frame["target_role"].eq("matched_null_control"))
    ].copy()
    selected["biology_label"] = selected["best_name"].fillna(
        "unannotated_feature_" + selected["feature_id"].astype(int).astype(str)
    )
    keep = [
        "panel", "feature_id", "biology_label", "mz", "rt_sec", "ion_family_id",
        "ion_family_size", "target_role", "best_name", "annotation_evidence_tier",
        "eic_detection_fraction", "eic_raw_rmu_log2fc", "eic_pqn_rmu_log2fc",
        "eic_max_exact_p", "raw_interaction_log2fc", "max_interaction_p",
    ]
    selected = selected[keep].sort_values(["panel", "ion_family_id", "feature_id"])
    path = OUT / "retained_candidates_for_ms2.csv"
    selected.to_csv(path, index=False)
    report = {
        "status": "mtbls13729_expanded_ms2_panel_frozen",
        "candidates": int(len(selected)),
        "ion_families": int(selected.assign(key=selected.panel + ':' + selected.ion_family_id.astype(str)).key.nunique()),
        "panels": selected.groupby("panel").size().astype(int).to_dict(),
        "source_sha256": sha256(SOURCE),
        "panel_sha256": sha256(path),
        "claim_limit": "Technical EIC retention selects features for MS2 acquisition review; it does not identify structures.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

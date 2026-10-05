#!/usr/bin/env python
"""Freeze B40-M1 graph-completion actions before reading ranking outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def cells() -> list[dict[str, Any]]:
    return [
        {"cell_id": "B40-00", "name": "real_diffusion", "score": "real_diffusion_score", "role": "core"},
        {"cell_id": "B40-01", "name": "real_minus_matched_seed_null", "score": "real_minus_null_mean", "role": "core"},
        {"cell_id": "B40-02", "name": "matched_seed_null_0", "score": "null_0_diffusion_score", "role": "negative_control"},
        {"cell_id": "B40-03", "name": "matched_seed_null_1", "score": "null_1_diffusion_score", "role": "negative_control"},
        {"cell_id": "B40-04", "name": "matched_seed_null_2", "score": "null_2_diffusion_score", "role": "negative_control"},
        {"cell_id": "B40-05", "name": "candidate_degree_only", "score": "candidate_rhea_degree", "role": "negative_control"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = cells()
    cells_path = args.output_dir / "cells.json"
    write(cells_path, matrix)
    report = {
        "status": "bioaware_b40_m1_graph_completion_manifest_frozen",
        "formal": True,
        "outcomes_read": False,
        "models_fitted": False,
        "cells": len(matrix),
        "core_cells": 2,
        "negative_control_cells": 4,
        "aggregation": "mean candidate score across every visible context, including structural zeros",
        "identifiable_primary_subgroup": "at least two competing candidates are Rhea-mapped; membership is outcome-blind",
        "starting_action": "frozen B37 catalog_topology final candidate",
        "common_risk_layer": "DreaMS baseline_gap <= frozen B37 topology gate_margin",
        "proposal": "unique candidate with maximum strictly-positive aggregate graph score",
        "decision": {
            "primary_delta_vs_topology": ">=0.03",
            "overall_delta_vs_topology": ">=0",
            "primary_corrected_gt_2x_introduced": True,
            "identity_and_formula_cluster_ci_low_gt_zero": True,
            "each_internal_source_primary_nonnegative": True,
            "minimum_interventions": 30,
            "real_core_delta_gt_each_seed_null": True,
            "negative_controls_never_promoted": True,
        },
        "forbidden": [
            "score weights or thresholds chosen after outcomes",
            "truth-mapped subgroup definition",
            "P2b features",
            "claiming graph completion from network membership alone",
            "passing directly to a clean-spectrum shared encoder",
        ],
        "provenance": {"script": sha256(Path(__file__)), "cells": sha256(cells_path)},
        "claim_limit": "Outcome-blind graph-action preregistration; no ranking or embedding result.",
    }
    write(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Fail-closed validation for A4-B1 formula-OOF residual adapter outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    required = ["decision.json", "oof_queries.csv.gz", "oof_embeddings.npz"]
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing B1 outputs: {missing}")
    decision = json.loads((args.output_dir / "decision.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.output_dir / "oof_queries.csv.gz")
    with np.load(args.output_dir / "oof_embeddings.npz") as archive:
        shapes = {name: archive[name].shape for name in ("clean", "linear", "nonlinear")}
        fold = archive["formula_fold"]
    n = int(decision["integrity"]["queries"])
    if len(frame) != n or any(shape[0] != n for shape in shapes.values()) or len(fold) != n:
        raise RuntimeError("B1 output row alignment failed")
    if decision["integrity"]["formula_fold_overlap"] != 0:
        raise RuntimeError("B1 formula leakage")
    if not args.allow_smoke:
        expected = {"queries": 4998, "rescue_targets": 542, "safety_controls": 3193}
        for key, value in expected.items():
            if int(decision["integrity"][key]) != value:
                raise RuntimeError(f"formal B1 {key}: expected {value}")
    for model in ("linear_residual", "nonlinear_residual"):
        metrics = decision["models"][model]
        for key in ("recall1", "delta_recall1", "near_delta_recall1",
                    "control_preservation_cosine_mean"):
            if not np.isfinite(float(metrics[key])):
                raise RuntimeError(f"non-finite {model}.{key}")
    print(f"[validate_noise_v3_a4b_rescue_adapter] PASS: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

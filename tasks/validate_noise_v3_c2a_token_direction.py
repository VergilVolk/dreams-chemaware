"""Fail-closed validation for C2-A formula-OOF outputs."""
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
    for name in ("decision.json", "training_queries.csv.gz", "oof_query_embeddings.npz"):
        if not (args.output_dir / name).is_file(): raise FileNotFoundError(args.output_dir / name)
    decision = json.loads((args.output_dir / "decision.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.output_dir / "training_queries.csv.gz")
    with np.load(args.output_dir / "oof_query_embeddings.npz") as archive:
        shapes = {name: archive[name].shape for name in ("clean", "global_control", "peak_token_expert")}
    if len(frame) != int(decision["training_queries"]) or any(shape[0] != len(frame) for shape in shapes.values()):
        raise RuntimeError("C2-A query/embedding alignment failed")
    if not args.allow_smoke and (int(decision["formulas"]) < 500 or int(decision["training_queries"]) < 1000):
        raise RuntimeError("formal C2-A chemical space is too small")
    for model in ("global_control", "peak_token_expert"):
        metrics = decision["models"][model]
        if not args.allow_smoke and int(metrics["examples"]) != 80250:
            raise RuntimeError(f"formal {model} silently omitted C1 examples")
        for key in ("accuracy", "delta_accuracy", "near_delta_accuracy", "safety_preservation_mean"):
            if not np.isfinite(float(metrics[key])): raise RuntimeError(f"non-finite {model}.{key}")
    print(f"[validate_noise_v3_c2a_token_direction] PASS: {args.output_dir}", flush=True)


if __name__ == "__main__": main()

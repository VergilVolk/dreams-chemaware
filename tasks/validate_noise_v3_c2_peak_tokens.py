"""Validate the formal C2 contextual peak-token cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = ["report.json", "rows.npy", "projection.npy", "tokens_f16.npy", "mz_f32.npy", "intensity_f32.npy", "valid.npy"]
    for name in names:
        if not (args.output_dir / name).is_file():
            raise FileNotFoundError(args.output_dir / name)
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    rows = np.load(args.output_dir / "rows.npy", mmap_mode="r")
    tokens = np.load(args.output_dir / "tokens_f16.npy", mmap_mode="r")
    valid = np.load(args.output_dir / "valid.npy", mmap_mode="r")
    scope = report.get("row_scope", "queries")
    if scope not in {"queries", "reachable"}:
        raise RuntimeError(f"unknown C2 row scope: {scope}")
    expected = {"queries": 23876, "reachable": 25275}[scope]
    if len(rows) != expected or int(report["spectra"]) != expected:
        raise RuntimeError(f"formal C2 cache does not cover expected scope={scope}: {expected:,} spectra")
    if tokens.shape != (expected, 100, 256) or valid.shape != (expected, 100):
        raise RuntimeError(f"unexpected C2 shapes: tokens={tokens.shape}, valid={valid.shape}")
    if len(np.unique(rows)) != expected or np.any(np.diff(rows) <= 0):
        raise RuntimeError("C2 rows must be unique and strictly increasing")
    sampled = tokens[::997].astype(np.float32)
    sampled_valid = valid[::997]
    norms = np.linalg.norm(sampled[sampled_valid], axis=1)
    if len(norms) == 0 or not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError("C2 projected token normalization failed")
    print(f"[validate_noise_v3_c2_peak_tokens] PASS: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

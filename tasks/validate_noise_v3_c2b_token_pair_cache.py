"""Fail-closed validation for the formal C2-B0 augmented graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_noise_v3_c2b_token_pair_cache import TOKEN_FEATURES
from build_g8r_real_error_atlas import Cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    report_path = args.cache.with_suffix(".json")
    if not args.cache.is_file(): raise FileNotFoundError(args.cache)
    if not report_path.is_file(): raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache = Cache(args.cache)
    if not report.get("formal") or int(report.get("queries", 0)) != 23876:
        raise RuntimeError("C2-B0 is not a full formal graph")
    if int(report.get("pairs", 0)) != len(cache.features):
        raise RuntimeError("C2-B0 pair count mismatch")
    if cache.feature_names[-len(TOKEN_FEATURES):] != TOKEN_FEATURES:
        raise RuntimeError("C2-B0 token feature schema mismatch")
    values = cache.features[:, -len(TOKEN_FEATURES):]
    if not np.isfinite(values).all(): raise RuntimeError("C2-B0 contains non-finite token evidence")
    if not report.get("gates", {}).get("pass"):
        raise RuntimeError("C2-B0 evidence gates failed; refusing downstream training")
    print(f"[validate_noise_v3_c2b_token_pair_cache] PASS: {args.cache}")


if __name__ == "__main__": main()

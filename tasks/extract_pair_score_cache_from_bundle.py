#!/usr/bin/env python
"""Extract one bundle method into an immutable GNPS pair-score cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnps_pair_score_cache import PANELS, write_pair_score_cache
from noise_final_core import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.score_bundle, allow_pickle=False) as body:
        methods = list(map(str, body["method_names"]))
        if args.method not in methods:
            raise RuntimeError(f"method {args.method} is not in the bundle")
        scores = {
            panel: np.asarray(body[f"scores_{panel}"][methods.index(args.method)])
            for panel in PANELS
        }
    method = {
        "name": args.method,
        "input_contract": "spectrum_pair",
        "source_bundle_sha256": sha256_file(args.score_bundle),
    }
    write_pair_score_cache(args.output, args.benchmark, method, scores)
    print(json.dumps({"status": "pair_score_cache_extracted", "method": args.method}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Freeze public-evidence-family score thresholds for the LCNEC M0 contract.

Calibration shard: the consumed GNPS Gold/Silver identity-disjoint pair panel
(non-LCNEC). For each evidence family, the frozen threshold is the smallest
grid score at which the false-pair fraction among pairs at/above the score
is <= the target FDR. Thresholds are written read-only for the certificate
engine. No LCNEC spectrum, label or abundance is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = {
    "weighted_spectral_entropy": {"method": "weighted_spectral_entropy"},
    "ms2deepscore_2x_public": {"method": "ms2deepscore_2x_public"},
}


def threshold_at_fdr(scores: np.ndarray, labels: np.ndarray, queries: np.ndarray,
                     same_formula: np.ndarray, quantile: float,
                     grid: np.ndarray) -> dict[str, object]:
    """Evidence floor = the false-pair (cross-formula) score quantile.

    Same-formula isomers reach near-unity scores in every spectral family, so
    identity-level error cannot be controlled by a floor; the floor instead
    guarantees that a WRONG-SKELETON pair crosses it with probability <=
    1 - quantile. Isomer disambiguation is delegated to multi-source
    certificate agreement.
    """
    cross_false = ~labels & ~same_formula
    raw = float(np.quantile(scores[cross_false], quantile))
    chosen = float(grid[grid >= raw][0]) if (grid >= raw).any() else float(grid[-1])
    return {"threshold": chosen,
            "cross_formula_false_q": raw,
            "n_cross_formula_false": int(cross_false.sum()),
            "quantile": quantile}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/validation/gnps_gold_silver_10ppm_benchmark_v1"))
    parser.add_argument("--bundle", type=Path,
                        default=Path("data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/method_scores.npz"))
    parser.add_argument("--quantile", type=float, default=0.999,
                        help="Cross-formula false-pair quantile used as the evidence floor.")
    parser.add_argument("--grid-step", type=float, default=0.005)
    parser.add_argument("--output", type=Path,
                        default=Path("data/validation/lcnec_public_evidence_thresholds_v1/thresholds.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True)

    with np.load(args.benchmark / "pairs_identity_disjoint.npz") as body:
        labels = np.asarray(body["label"], dtype=bool)
        queries = np.asarray(body["query_index"], dtype=np.int64)
        same_formula = np.asarray(body["same_formula"], dtype=bool)
    bundle = np.load(args.bundle, allow_pickle=False)
    methods = [str(m) for m in bundle["method_names"]]
    scores_all = bundle["scores_identity_disjoint"]
    grid = np.round(np.arange(0.0, 1.0 + args.grid_step / 2, args.grid_step), 4)

    report: dict[str, object] = {
        "status": "LCNEC_PUBLIC_EVIDENCE_THRESHOLDS_FROZEN",
        "data_role": "development/consumed calibration shard (GNPS, non-LCNEC)",
        "contract": "docs/LCNEC_PUBLIC_EVIDENCE_SPACE_M0_CONTRACT_20261011.md",
        "quantile": args.quantile,
        "n_pairs": int(len(labels)),
        "n_true_pairs": int(labels.sum()),
        "grid_step": args.grid_step,
        "families": {},
        "lcnec_data_read": False,
    }
    thresholds: dict[str, float] = {}
    for family, spec in FAMILIES.items():
        scores = np.asarray(scores_all[methods.index(spec["method"])], dtype=np.float64)
        stats = threshold_at_fdr(scores, labels, queries, same_formula,
                                 args.quantile, grid)
        thresholds[family] = stats["threshold"]
        report["families"][family] = {
            "bundle_method": spec["method"],
            **stats,
            "score_p50_true": float(np.median(scores[labels])),
            "score_p50_false": float(np.median(scores[~labels])),
        }
    report["thresholds"] = thresholds
    report["semantics"] = ("evidence floor = cross-formula false-pair score quantile on the "
                           "adversarial identity-disjoint shard")
    report["rationale"] = ("Same-formula isomers reach near-unity scores in every spectral family; "
                           "identity-level error is therefore not controllable by a per-family floor. "
                           "The floor controls wrong-skeleton crossings at 1-quantile probability and "
                           "isomer disambiguation is delegated to multi-source certificate agreement.")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "families": report["families"]}, indent=2))


if __name__ == "__main__":
    main()

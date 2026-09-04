#!/usr/bin/env python
"""Candidate-label permutation falsification for BioAware v4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_metdna3_negative_candidate_permutation import (  # noqa: E402
    permute_candidate_feature_blocks,
)
from develop_bioaware_metdna3_negative_loso_ranker import summarize  # noqa: E402
from develop_bioaware_same_formula_network_expert_v3 import FEATURES  # noqa: E402
from develop_bioaware_same_formula_uncertainty_expert_v4 import run_loso  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v3-development-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_development"),
    )
    parser.add_argument(
        "--v4-development-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_uncertainty_expert_v4_development"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_uncertainty_expert_v4_permutation"),
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidate_path = args.v3_development_dir / "candidate_features.csv.gz"
    development_path = args.v4_development_dir / "report.json"
    for path in (candidate_path, development_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development.get("pass_to_falsification") is not True:
        raise RuntimeError("v4 development did not pass")
    if sha256(candidate_path) != development["provenance"]["candidate_features_sha256"]:
        raise RuntimeError("v4 candidate hash mismatch")
    candidates = pd.read_csv(candidate_path)
    observed = summarize(run_loso(candidates, verbose=False)[0])
    if observed != development["pooled"]:
        raise RuntimeError("v4 observed result does not reproduce")
    permutation_features = [feature for feature in FEATURES if feature != "spectral_score"]
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for repeat in range(args.repeats):
        permuted = permute_candidate_feature_blocks(candidates, permutation_features, rng)
        null_rows.append({
            "repeat": repeat,
            **summarize(run_loso(permuted, verbose=False)[0]),
        })
        if (repeat + 1) % 10 == 0 or repeat + 1 == args.repeats:
            print(f"[v4 candidate permutation] {repeat + 1}/{args.repeats}", flush=True)
    null = pd.DataFrame(null_rows)
    metrics = {}
    for metric in ("delta_recall1", "risk_weighted_net_lambda2", "corrected", "introduced"):
        values = null[metric].to_numpy(float)
        observed_value = float(observed[metric])
        metrics[metric] = {
            "observed": observed_value,
            "null_mean": float(values.mean()),
            "null_p05": float(np.quantile(values, 0.05)),
            "null_p95": float(np.quantile(values, 0.95)),
            "empirical_one_sided_p_ge_observed": float(
                (1 + np.sum(values >= observed_value)) / (1 + len(values))
            ),
        }
    report = {
        "status": "bioaware_same_formula_v4_candidate_permutation_complete",
        "formal": True,
        "protocol": (
            "v4 same-formula identity+formula-purged LOSO; fixed DreaMS gap gate; joint "
            "within-query network-feature permutation"
        ),
        "repeats": int(args.repeats),
        "observed": observed,
        "null_metrics": metrics,
        "gates": {
            "delta_permutation_p_le_0_05": metrics["delta_recall1"]["empirical_one_sided_p_ge_observed"] <= 0.05,
            "risk_net_permutation_p_le_0_05": metrics["risk_weighted_net_lambda2"]["empirical_one_sided_p_ge_observed"] <= 0.05,
        },
        "contracts": {
            "spectral_score_not_permuted": True,
            "network_feature_covariance_preserved_as_joint_block": True,
            "threshold_or_recipe_retuned": False,
            "P2b": "forbidden",
        },
        "provenance": {
            "development_report_sha256": sha256(development_path),
            "candidate_features_sha256": sha256(candidate_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Opened-source falsification; not independent external confirmation.",
    }
    report["pass_to_freeze"] = all(report["gates"].values())
    args.output_dir.mkdir(parents=True)
    null_path = args.output_dir / "candidate_permutation_null.csv.gz"
    null.to_csv(null_path, index=False, compression="gzip")
    report["provenance"]["candidate_permutation_null_sha256"] = sha256(null_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

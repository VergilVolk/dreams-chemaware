#!/usr/bin/env python
"""Within-query network-evidence permutation for BioAware v3 development."""

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
from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    evaluate_fold,
    summarize,
)
from develop_bioaware_same_formula_network_expert_v3 import FEATURES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_identity_formula_purged_run(candidates: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for unit in sorted(candidates["unit_id"].astype(str).unique()):
        test = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        identities = set(test["truth_candidate_id"].astype(str))
        formulas = set(test["truth_formula"].astype(str))
        train = candidates.loc[
            ~candidates["unit_id"].astype(str).eq(unit)
            & ~candidates["truth_candidate_id"].astype(str).isin(identities)
            & ~candidates["truth_formula"].astype(str).isin(formulas)
        ].copy()
        result, _ = evaluate_fold(
            train, test, unit, features=FEATURES, require_raw_step0_edge=True
        )
        outputs.append(result)
    transitions = pd.concat(outputs, ignore_index=True)
    if len(transitions) != candidates["query_id"].nunique():
        raise RuntimeError("permutation run changed query coverage")
    if transitions["query_id"].duplicated().any():
        raise RuntimeError("permutation run duplicated query transitions")
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_development"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_permutation"),
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidate_path = args.development_dir / "candidate_features.csv.gz"
    development_report_path = args.development_dir / "report.json"
    if not candidate_path.is_file() or not development_report_path.is_file():
        raise FileNotFoundError(args.development_dir)
    development_report = json.loads(development_report_path.read_text(encoding="utf-8"))
    if development_report.get("pass_to_falsification") is not True:
        raise RuntimeError("same-formula development did not pass its gates")
    if sha256(candidate_path) != development_report["provenance"]["candidate_features_sha256"]:
        raise RuntimeError("same-formula candidate feature hash mismatch")
    candidates = pd.read_csv(candidate_path)
    permutation_features = [feature for feature in FEATURES if feature != "spectral_score"]
    observed = summarize(unit_identity_formula_purged_run(candidates))
    if observed != development_report["pooled"]:
        raise RuntimeError("observed same-formula result does not reproduce exactly")
    rng = np.random.default_rng(args.seed)
    null_rows: list[dict[str, float | int]] = []
    for repeat in range(args.repeats):
        permuted = permute_candidate_feature_blocks(candidates, permutation_features, rng)
        summary = summarize(unit_identity_formula_purged_run(permuted))
        null_rows.append({"repeat": repeat, **summary})
        if (repeat + 1) % 10 == 0 or repeat + 1 == args.repeats:
            print(f"[same-formula candidate permutation] {repeat + 1}/{args.repeats}", flush=True)
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
        "status": "bioaware_same_formula_v3_candidate_permutation_complete",
        "formal": True,
        "protocol": (
            "same-formula eight-unit identity+formula-purged LOSO; fixed recipe and gates; "
            "joint within-query permutation of all network evidence"
        ),
        "repeats": int(args.repeats),
        "features": FEATURES,
        "permuted_network_features": permutation_features,
        "observed": observed,
        "null_metrics": metrics,
        "gates": {
            "delta_permutation_p_le_0_05": (
                metrics["delta_recall1"]["empirical_one_sided_p_ge_observed"] <= 0.05
            ),
            "risk_net_permutation_p_le_0_05": (
                metrics["risk_weighted_net_lambda2"]["empirical_one_sided_p_ge_observed"] <= 0.05
            ),
        },
        "contracts": {
            "candidate_count_preserved": True,
            "spectral_score_not_permuted": True,
            "network_feature_covariance_preserved_as_joint_block": True,
            "threshold_or_recipe_retuned": False,
            "ST001154_outcomes_read": False,
            "P2b": "forbidden",
        },
        "provenance": {
            "development_report_sha256": sha256(development_report_path),
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

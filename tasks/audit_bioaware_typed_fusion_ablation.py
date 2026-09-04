#!/usr/bin/env python
"""Fixed-weight BioAware typed-context ablation on exposed development data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware import BioAwareConfig, fuse_candidates, top1_transition_table  # noqa: E402


FEATURES = (
    "raw_network_support",
    "dependency_corrected_network_support",
    "candidate_specific_network_support",
    "complete_network_support",
    "complete_candidate_specific_network_support",
    "direction_supported_network_support",
    "complete_direction_supported_network_support",
    "fully_observed_direction_supported_network_support",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_bootstrap(
    transitions: pd.DataFrame,
    formulas: pd.Series,
    *,
    repeats: int,
    seed: int,
) -> dict:
    frame = transitions[["query_id", "baseline_correct", "final_correct"]].merge(
        formulas.rename("truth_formula").reset_index(),
        on="query_id",
        validate="one_to_one",
    )
    frame["delta"] = (
        frame["final_correct"].astype(float)
        - frame["baseline_correct"].astype(float)
    )
    clusters = [group["delta"].to_numpy(float) for _, group in frame.groupby("truth_formula")]
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for index in range(repeats):
        selected = rng.integers(0, len(clusters), len(clusters))
        values = np.concatenate([clusters[position] for position in selected])
        samples[index] = float(values.mean())
    return {
        "mean": float(frame["delta"].mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "clusters": int(len(clusters)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        type=Path,
        default=Path(
            "data/validation/bioaware_reaction_context_broad_calibrated_20260830/"
            "mtbls1905_auto__candidate_context.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_typed_fusion_ablation_20260830"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.context)
    missing = set(FEATURES) - set(frame)
    if missing:
        raise RuntimeError(f"context missing typed features: {sorted(missing)}")
    required = {"query_id", "candidate_id", "spectral_score", "truth_candidate_id", "truth_formula"}
    missing_required = required - set(frame)
    if missing_required:
        raise RuntimeError(f"context missing required columns: {sorted(missing_required)}")
    candidates = frame[list(required)].copy()
    formulas = (
        candidates.groupby("query_id", sort=False)["truth_formula"].first()
    )
    config = BioAwareConfig()
    reports: dict[str, dict] = {}
    transition_frames: list[pd.DataFrame] = []
    for feature_index, feature in enumerate(FEATURES):
        scored_input = candidates.copy()
        scored_input["network_support"] = pd.to_numeric(frame[feature], errors="raise")
        scored_input["network_path_count"] = (
            scored_input["network_support"] > 0
        ).astype(int)
        scored, decisions = fuse_candidates(scored_input, config)
        transitions, result = top1_transition_table(
            scored, truth_col="truth_candidate_id"
        )
        transitions.insert(0, "feature", feature)
        transition_frames.append(transitions)
        corrected = int(result["corrected"])
        introduced = int(result["introduced"])
        discordant = corrected + introduced
        reports[feature] = {
            **result,
            "interventions": int(decisions["bioaware_applied"].sum()),
            "queries_with_nonzero_support": int(
                scored_input.groupby("query_id")["network_support"].max().gt(0).sum()
            ),
            "mcnemar_exact_p": (
                float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            ),
            "formula_cluster_bootstrap": formula_bootstrap(
                transitions,
                formulas,
                repeats=args.bootstrap_resamples,
                seed=args.seed + feature_index,
            ),
        }

    transitions_path = args.output_dir / "transitions.csv.gz"
    report_path = args.output_dir / "report.json"
    pd.concat(transition_frames, ignore_index=True).to_csv(
        transitions_path, index=False
    )
    report = {
        "status": "bioaware_typed_fusion_ablation_complete",
        "formal": False,
        "exposed_development_data": True,
        "feature_selection_performed": False,
        "configuration": {
            "network_weight": config.network_weight,
            "maximum_spectral_margin_for_override": (
                config.maximum_spectral_margin_for_override
            ),
            "minimum_network_advantage": config.minimum_network_advantage,
        },
        "features": reports,
        "decision": (
            "This fixed ablation may identify a schema to preregister, but cannot "
            "establish external gain or tune weights on this exposed cohort."
        ),
        "provenance": {
            "context_sha256": sha256(args.context),
            "transitions_sha256": sha256(transitions_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

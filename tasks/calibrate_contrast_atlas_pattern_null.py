#!/usr/bin/env python
"""Calibrate shared-pattern alignment with a within-target permutation null.

The null preserves each study's effect/SE distribution and pattern coverage but
breaks the mapping between target effects and chemical pattern IDs.  It is a
chemical-alignment null, not a substitute for source/instrument-matched outer
validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contrast_atlas_core import weighted_cosine


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = np.minimum.accumulate((valid.to_numpy() * n / np.arange(1, n + 1))[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def calibrate_pair(
    query: pd.DataFrame,
    target: pd.DataFrame,
    draws: int,
    seed: int,
    min_common: int,
) -> dict[str, object]:
    if query["pattern_id"].duplicated().any() or target["pattern_id"].duplicated().any():
        raise RuntimeError("pattern IDs must be unique within a study contrast and world")
    common_ids = sorted(set(query["pattern_id"]) & set(target["pattern_id"]))
    if len(common_ids) < min_common:
        return {
            "calibration_status": "abstain_insufficient_patterns_for_null",
            "n_common_patterns": len(common_ids),
            "null_draws": draws,
            "observed_score": float("nan"),
            "p_similar": float("nan"),
            "p_opposite": float("nan"),
            "p_two_sided": float("nan"),
        }
    query_indexed = query.set_index("pattern_id")
    target_indexed = target.set_index("pattern_id")
    q_beta = query_indexed.loc[common_ids, "pattern_beta"].to_numpy(float)
    q_se = query_indexed.loc[common_ids, "pattern_se"].to_numpy(float)
    target_ids = target_indexed.index.to_numpy()
    common_positions = np.asarray([int(np.flatnonzero(target_ids == pattern)[0]) for pattern in common_ids])
    target_beta = target_indexed["pattern_beta"].to_numpy(float)
    target_se = target_indexed["pattern_se"].to_numpy(float)
    observed_beta = target_beta[common_positions]
    observed_se = target_se[common_positions]
    observed_weight = 1.0 / np.maximum(q_se ** 2 + observed_se ** 2, 1e-12)
    observed = weighted_cosine(q_beta, observed_beta, observed_weight)
    if not np.isfinite(observed):
        raise RuntimeError("non-finite observed shared-pattern score")

    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=float)
    for index in range(draws):
        permutation = rng.permutation(len(target_ids))
        permuted_beta = target_beta[permutation][common_positions]
        permuted_se = target_se[permutation][common_positions]
        weight = 1.0 / np.maximum(q_se ** 2 + permuted_se ** 2, 1e-12)
        null[index] = weighted_cosine(q_beta, permuted_beta, weight)
    p_similar = float((1 + np.sum(null >= observed)) / (draws + 1))
    p_opposite = float((1 + np.sum(null <= observed)) / (draws + 1))
    return {
        "calibration_status": "calibrated_within_target_pattern_permutation",
        "n_common_patterns": len(common_ids),
        "n_target_patterns": int(len(target)),
        "null_draws": draws,
        "observed_score": observed,
        "null_mean": float(np.mean(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "p_similar": p_similar,
        "p_opposite": p_opposite,
        "p_two_sided": min(1.0, 2 * min(p_similar, p_opposite)),
    }


def calibrate_retrievals(
    retrieval: pd.DataFrame,
    patterns: pd.DataFrame,
    draws: int = 999,
    base_seed: int = 20261013,
    min_common: int = 8,
    fdr_threshold: float = 0.10,
) -> pd.DataFrame:
    required_retrieval = {
        "world", "query_study_id", "query_contrast_id", "target_study_id", "target_contrast_id",
        "shared_pattern_pass", "shared_pattern_score",
    }
    required_patterns = {"world", "study_id", "contrast_id", "pattern_id", "pattern_beta", "pattern_se"}
    if required_retrieval - set(retrieval.columns):
        raise RuntimeError(f"retrieval table missing: {sorted(required_retrieval - set(retrieval.columns))}")
    if required_patterns - set(patterns.columns):
        raise RuntimeError(f"pattern table missing: {sorted(required_patterns - set(patterns.columns))}")
    if draws < 99:
        raise ValueError("at least 99 null draws are required")
    rows = []
    for item in retrieval.itertuples(index=False):
        base = item._asdict()
        if not bool(item.shared_pattern_pass):
            calibrated = {
                "calibration_status": "abstain_shared_pattern_gate_failed",
                "n_common_patterns": int(getattr(item, "n_common_patterns", 0)),
                "null_draws": draws,
                "observed_score": float("nan"),
                "p_similar": float("nan"), "p_opposite": float("nan"), "p_two_sided": float("nan"),
            }
        else:
            query = patterns.loc[
                (patterns["world"] == item.world)
                & (patterns["study_id"] == item.query_study_id)
                & (patterns["contrast_id"] == item.query_contrast_id)
            ]
            target = patterns.loc[
                (patterns["world"] == item.world)
                & (patterns["study_id"] == item.target_study_id)
                & (patterns["contrast_id"] == item.target_contrast_id)
            ]
            calibrated = calibrate_pair(
                query, target, draws,
                stable_seed(base_seed, item.world, item.query_study_id, item.query_contrast_id,
                            item.target_study_id, item.target_contrast_id),
                min_common,
            )
            if np.isfinite(float(calibrated["observed_score"])) and not np.isclose(
                float(calibrated["observed_score"]), float(item.shared_pattern_score), atol=1e-10,
            ):
                raise RuntimeError("recomputed pattern score disagrees with retrieval table")
        rows.append({**base, **calibrated})
    output = pd.DataFrame(rows)
    output["q_similar"] = output.groupby(["world", "query_study_id", "query_contrast_id"], dropna=False)["p_similar"].transform(bh_adjust)
    output["q_opposite"] = output.groupby(["world", "query_study_id", "query_contrast_id"], dropna=False)["p_opposite"].transform(bh_adjust)
    output["calibrated_association"] = "abstain_not_calibrated"
    valid = output["calibration_status"].eq("calibrated_within_target_pattern_permutation")
    output.loc[valid, "calibrated_association"] = "indeterminate_after_pattern_null"
    output.loc[valid & (output["q_similar"] <= fdr_threshold), "calibrated_association"] = "similar_response"
    output.loc[valid & (output["q_opposite"] <= fdr_threshold), "calibrated_association"] = "opposite_response"
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--pattern-responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--min-common", type=int, default=8)
    parser.add_argument("--fdr-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    retrieval = pd.read_csv(args.retrieval_results)
    patterns = pd.read_csv(args.pattern_responses)
    output = calibrate_retrievals(
        retrieval, patterns, args.draws, args.seed, args.min_common, args.fdr_threshold,
    )
    args.output_dir.mkdir(parents=True)
    output.to_csv(args.output_dir / "calibrated_retrieval_results.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_WITHIN_TARGET_PATTERN_NULL_COMPLETE",
        "formal": False,
        "analysis_role": "development calibration component",
        "draws": args.draws,
        "seed": args.seed,
        "min_common_patterns": args.min_common,
        "fdr_threshold": args.fdr_threshold,
        "rows": int(len(output)),
        "calibrated_rows": int(output["calibration_status"].eq("calibrated_within_target_pattern_permutation").sum()),
        "claim_limit": "Within-target chemical-alignment null only; source/instrument-matched outer null and independent validation remain required.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

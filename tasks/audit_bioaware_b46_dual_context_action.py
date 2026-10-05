#!/usr/bin/env python
"""Fixed dual-context BioAware action with coverage-neutral strong controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["query_id", "candidate_id", "fold"]
SIGNATURE = ["independent_member_count", "independent_member_intersection"]
ARMS = {
    "real_dual": ("transform_specificity", "positive_excess"),
    "wrong_sign_dual": ("wrong_sign_concordance", "positive_excess"),
    "random_coabundance_dual": ("transform_specificity", "random_positive"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def percentile_rank(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values.rank(method="average") - 1.0) / (len(values) - 1.0)


def harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = left + right
    return np.divide(
        2.0 * left * right,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def build_scores(
    transform: pd.DataFrame,
    coabundance: pd.DataFrame,
    minimum_rotations: int,
) -> tuple[pd.DataFrame, dict]:
    transform_columns = KEY + [
        "available",
        "transform_specificity",
        "wrong_sign_concordance",
    ]
    coabundance_columns = KEY + [
        "available",
        "positive_excess",
        "random_positive",
    ]
    left = transform[transform_columns].rename(columns={"available": "transform_available"})
    right = coabundance[coabundance_columns].rename(columns={"available": "coabundance_available"})
    if left.duplicated(KEY).any() or right.duplicated(KEY).any():
        raise RuntimeError("B46 rotation keys are not unique")
    merged = left.merge(right, on=KEY, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("B46 transform/coabundance rotation ledgers do not align")
    joint_available = (
        merged["transform_available"].astype(bool)
        & merged["coabundance_available"].astype(bool)
    )
    merged["joint_available"] = joint_available.astype(float)
    for feature in {
        feature for pair in ARMS.values() for feature in pair
    }:
        merged[f"rank__{feature}"] = merged.groupby(
            ["query_id", "fold"], sort=False
        )[feature].transform(percentile_rank)
    for arm, (spectral_feature, abundance_feature) in ARMS.items():
        merged[f"rotation_score__{arm}"] = harmonic(
            merged[f"rank__{spectral_feature}"].to_numpy(float),
            merged[f"rank__{abundance_feature}"].to_numpy(float),
        ) * merged["joint_available"].to_numpy(float)
    aggregation = {
        "joint_rotations": ("joint_available", "sum"),
        "total_rotations": ("fold", "nunique"),
        **{
            f"score__{arm}": (f"rotation_score__{arm}", "mean")
            for arm in ARMS
        },
    }
    scores = merged.groupby(["query_id", "candidate_id"], sort=False).agg(**aggregation).reset_index()
    eligible = scores["joint_rotations"].ge(minimum_rotations)
    for arm in ARMS:
        scores.loc[~eligible, f"score__{arm}"] = 0.0
    return scores, {
        "rotation_rows": int(len(merged)),
        "joint_available_rotation_rows": int(joint_available.sum()),
        "candidate_rows_with_minimum_rotations": int(eligible.sum()),
        "minimum_rotations": int(minimum_rotations),
    }


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    frame = pd.DataFrame({"value": values.astype(float), "cluster": clusters.astype(str)})
    grouped = frame.groupby("cluster", sort=False)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(frame["value"].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def apply_fixed_action(
    candidates: pd.DataFrame,
    arm: str,
    margin: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    score_column = f"score__{arm}"
    for query_id, group in candidates.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline = str(group["baseline_candidate_id"].iloc[0])
        baseline_rows = group.loc[group["candidate_id"].astype(str).eq(baseline)]
        if len(baseline_rows) != 1:
            raise RuntimeError(f"{query_id}: baseline candidate is not unique")
        baseline_row = baseline_rows.iloc[0]
        eligible = group.copy()
        for signature in SIGNATURE:
            eligible = eligible.loc[
                eligible[signature].astype(int).eq(int(baseline_row[signature]))
            ]
        maximum = float(eligible[score_column].max())
        best = eligible.loc[np.isclose(eligible[score_column], maximum, rtol=0.0, atol=1e-12)]
        unique = len(best) == 1
        proposal = str(best["candidate_id"].iloc[0]) if unique else baseline
        evidence_advantage = maximum - float(baseline_row[score_column])
        intervene = bool(
            float(group["baseline_gap"].iloc[0]) <= margin
            and unique
            and proposal != baseline
            and evidence_advantage > 1e-12
        )
        final = proposal if intervene else baseline
        # The frozen protocol counts a DreaMS Top-1 tie against the truth even
        # when the deterministic representative ID happens to equal it.  A
        # no-op must therefore replay the stored baseline outcome, not infer it
        # from string equality.  A genuine intervention creates a unique
        # promoted Top-1 and can be judged by identity equality.
        baseline_correct_values = group["baseline_correct"].astype(bool).unique()
        if len(baseline_correct_values) != 1:
            raise RuntimeError(f"{query_id}: inconsistent frozen baseline outcome")
        baseline_correct = bool(baseline_correct_values[0])
        final_correct = bool(final == truth) if intervene else baseline_correct
        truth_row = group.loc[group["candidate_id"].astype(str).eq(truth)]
        if len(truth_row) != 1:
            raise RuntimeError(f"{query_id}: truth candidate is not unique")
        truth_coverage_neutral = all(
            int(truth_row[signature].iloc[0]) == int(baseline_row[signature])
            for signature in SIGNATURE
        )
        truth_score = float(truth_row[score_column].iloc[0])
        rows.append(
            {
                "query_id": str(query_id),
                "source": str(group["source"].iloc[0]),
                "truth_candidate_id": truth,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "baseline_candidate_id": baseline,
                "proposal_candidate_id": proposal,
                "final_candidate_id": final,
                "baseline_correct": baseline_correct,
                "final_correct": final_correct,
                "intervene": intervene,
                "evidence_advantage": float(evidence_advantage),
                "truth_coverage_neutral": bool(truth_coverage_neutral),
                "truth_score": truth_score,
                "baseline_score": float(baseline_row[score_column]),
                "truth_joint_rotations": int(truth_row["joint_rotations"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    transitions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    baseline = transitions["baseline_correct"].astype(bool).to_numpy()
    final = transitions["final_correct"].astype(bool).to_numpy()
    delta = final.astype(float) - baseline.astype(float)
    corrected = (~baseline) & final
    introduced = baseline & (~final)
    by_source = {}
    for source, group in transitions.groupby("source", sort=True):
        base = group["baseline_correct"].astype(bool).to_numpy()
        outcome = group["final_correct"].astype(bool).to_numpy()
        by_source[str(source)] = {
            "queries": int(len(group)),
            "delta_recall1": float(outcome.mean() - base.mean()),
            "corrected": int(((~base) & outcome).sum()),
            "introduced": int((base & (~outcome)).sum()),
        }
    return {
        "queries": int(len(transitions)),
        "baseline_recall1": float(baseline.mean()),
        "recall1": float(final.mean()),
        "delta_recall1": float(delta.mean()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "risk_net_lambda2": int(corrected.sum() - 2 * introduced.sum()),
        "interventions": int(transitions["intervene"].sum()),
        "intervention_rate": float(transitions["intervene"].mean()),
        "corrected_identities": int(transitions.loc[corrected, "truth_candidate_id"].nunique()),
        "coverage_neutral_baseline_errors": int(
            ((~baseline) & transitions["truth_coverage_neutral"].astype(bool).to_numpy()).sum()
        ),
        "formula_cluster_bootstrap": cluster_bootstrap(
            delta,
            transitions["truth_formula"].astype(str).to_numpy(),
            repeats,
            seed,
        ),
        "identity_cluster_bootstrap": cluster_bootstrap(
            delta,
            transitions["truth_candidate_id"].astype(str).to_numpy(),
            repeats,
            seed + 1,
        ),
        "by_source": by_source,
    }


def compare(
    real: pd.DataFrame,
    control: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    columns = ["query_id", "truth_formula", "truth_candidate_id", "final_correct"]
    joined = real[columns].merge(
        control[["query_id", "final_correct"]],
        on="query_id",
        suffixes=("_real", "_control"),
        validate="one_to_one",
    )
    delta = (
        joined["final_correct_real"].astype(float)
        - joined["final_correct_control"].astype(float)
    ).to_numpy()
    return {
        "mean_delta": float(delta.mean()),
        "formula_cluster_bootstrap": cluster_bootstrap(
            delta, joined["truth_formula"].astype(str).to_numpy(), repeats, seed
        ),
        "identity_cluster_bootstrap": cluster_bootstrap(
            delta, joined["truth_candidate_id"].astype(str).to_numpy(), repeats, seed + 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features",
        type=Path,
        default=ROOT / "data/validation/bioaware_b2_reaction_transform_local_20260906/candidate_transform_features.csv.gz",
    )
    parser.add_argument(
        "--transform-rotations",
        type=Path,
        default=ROOT / "data/validation/bioaware_b2_reaction_transform_local_20260906/candidate_rotation_features.csv.gz",
    )
    parser.add_argument(
        "--coabundance-rotations",
        type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/reaction_coabundance_rotations.csv.gz",
    )
    parser.add_argument(
        "--topology-features",
        type=Path,
        default=ROOT / "data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2/candidate_catalog_features.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-rotations", type=int, default=2)
    parser.add_argument("--baseline-margin", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.minimum_rotations != 2 or args.baseline_margin != 0.05:
        raise ValueError("formal B46 uses two rotations and margin 0.05")
    if args.bootstrap_resamples < 10000:
        raise ValueError("formal B46 requires at least 10,000 bootstrap resamples")
    for path in (
        args.candidate_features,
        args.transform_rotations,
        args.coabundance_rotations,
        args.topology_features,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    base = pd.read_csv(args.candidate_features)
    transform = pd.read_csv(args.transform_rotations)
    coabundance = pd.read_csv(args.coabundance_rotations)
    topology = pd.read_csv(args.topology_features)
    if len(base) != 5384 or base["query_id"].nunique() != 1426:
        raise RuntimeError("B46 frozen four-source candidate universe changed")
    if set(base["source"].astype(str)) != {"BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"}:
        raise RuntimeError("B46 source universe changed")
    scores, score_report = build_scores(transform, coabundance, args.minimum_rotations)
    candidates = base.merge(scores, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    topology = topology[["query_id", "candidate_id", *SIGNATURE]]
    candidates = candidates.merge(topology, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    required = [*(f"score__{arm}" for arm in ARMS), "joint_rotations", *SIGNATURE]
    if candidates[required].isna().any().any():
        raise RuntimeError("B46 score/topology merge is incomplete")

    transitions: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    for index, arm in enumerate(ARMS):
        table = apply_fixed_action(candidates, arm, args.baseline_margin)
        transitions[arm] = table
        summaries[arm] = summarize(
            table, args.bootstrap_resamples, args.seed + index * 100
        )
        atomic_csv(args.output_dir / f"{arm}_transitions.csv.gz", table)
    contrasts = {
        f"real_dual_minus_{control}": compare(
            transitions["real_dual"],
            transitions[control],
            args.bootstrap_resamples,
            args.seed + 1000 + index * 100,
        )
        for index, control in enumerate(("wrong_sign_dual", "random_coabundance_dual"))
    }
    real = summaries["real_dual"]
    gates = {
        "queries_ge_800": real["queries"] >= 800,
        "baseline_errors_ge_150": int((~transitions["real_dual"]["baseline_correct"].astype(bool)).sum()) >= 150,
        "coverage_neutral_errors_ge_100": real["coverage_neutral_baseline_errors"] >= 100,
        "gain_ge_3pp": real["delta_recall1"] >= 0.03,
        "corrected_identities_ge_25": real["corrected_identities"] >= 25,
        "corrected_gt_2x_introduced": real["corrected"] > 2 * real["introduced"],
        "formula_ci_low_positive": real["formula_cluster_bootstrap"]["ci_low"] > 0,
        "identity_ci_low_positive": real["identity_cluster_bootstrap"]["ci_low"] > 0,
        "every_source_nonnegative": all(
            row["delta_recall1"] >= 0 for row in real["by_source"].values()
        ),
        "real_beats_both_controls_formula_ci": all(
            row["formula_cluster_bootstrap"]["ci_low"] > 0
            for row in contrasts.values()
        ),
    }
    report = {
        "status": "bioaware_b46_dual_context_action_complete",
        "formal": True,
        "arms": summaries,
        "contrasts": contrasts,
        "score_support": score_report,
        "gates": gates,
        "pass_to_external_context_benchmark": bool(all(gates.values())),
        "contracts": {
            "B44_read": False,
            "candidate_static_degree_used_in_score": False,
            "coverage_signature_exact_for_action": True,
            "truth_or_outcome_used_to_construct_evidence": False,
            "shared_embedding_changed": False,
            "internal_context_is_prospective_unknown": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256(args.candidate_features),
            "transform_rotations_sha256": sha256(args.transform_rotations),
            "coabundance_rotations_sha256": sha256(args.coabundance_rotations),
            "topology_features_sha256": sha256(args.topology_features),
            "script_sha256": sha256(Path(__file__)),
        },
        "parameters": {
            "minimum_rotations": args.minimum_rotations,
            "baseline_margin": args.baseline_margin,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "claim_limit": (
            "Opened-development held-seed-rotation action screen; not prospective "
            "unknown annotation, external performance, biological causality, shared-"
            "embedding improvement, or SOTA."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "real_dual": real,
        "contrasts": contrasts,
        "gates": gates,
        "pass_to_external_context_benchmark": report["pass_to_external_context_benchmark"],
    }, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

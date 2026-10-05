"""Build and test a reference-anchored peak-operator tensor on MTBLS13729.

This is a falsification-oriented A0, not an annotation or biological discovery
claim.  Public reference spectra act as probes.  For source-paper Level-1
calibration identities, the exact feature identity is hidden from scoring and
used only for an outer IK14-isolated evaluation.

The script reuses the frozen reverse-probe raw scan.  It does not reread mzML,
does not use phenotype labels, and does not treat a public library spectrum as
a same-platform authentic standard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_FEATURES = (
    "sqrt_cosine",
    "linear_cosine",
    "entropy_similarity",
    "intensity_coverage_min",
    "intensity_coverage_mean",
    "matched_peak_fraction_min",
    "top10_match_fraction",
    "neutral_loss_sqrt_cosine",
    "neutral_loss_coverage_min",
    "neutral_loss_coverage_mean",
    "peak_count_ratio",
)
QUERY_COLUMNS = ("panel", "ik14", "sample_id")

FEATURE_SETS = {
    "identity_multireference": (
        "max_sqrt_cosine",
        "mean_sqrt_cosine",
        "max_entropy_similarity",
        "mean_entropy_similarity",
        "reference_support_fraction",
    ),
    "spectral_peak_operator": (
        "max_sqrt_cosine",
        "mean_sqrt_cosine",
        "max_entropy_similarity",
        "mean_entropy_similarity",
        "reference_support_fraction",
        "max_top10_match_fraction",
        "mean_top10_match_fraction",
        "max_intensity_coverage_min",
        "mean_intensity_coverage_min",
        "max_matched_peak_fraction_min",
        "mean_matched_peak_fraction_min",
        "max_neutral_loss_sqrt_cosine",
        "mean_neutral_loss_sqrt_cosine",
    ),
    "quant_detectability_only": (
        "ms1_log_intensity",
        "ms1_within_sample_percentile",
        "global_prevalence",
        "group_prevalence_max",
    ),
    "full_operator_tensor": (
        "max_sqrt_cosine",
        "mean_sqrt_cosine",
        "max_entropy_similarity",
        "mean_entropy_similarity",
        "reference_support_fraction",
        "max_top10_match_fraction",
        "mean_top10_match_fraction",
        "max_intensity_coverage_min",
        "mean_intensity_coverage_min",
        "max_matched_peak_fraction_min",
        "mean_matched_peak_fraction_min",
        "max_neutral_loss_sqrt_cosine",
        "mean_neutral_loss_sqrt_cosine",
        "ms1_log_intensity",
        "ms1_within_sample_percentile",
        "global_prevalence",
        "group_prevalence_max",
    ),
}


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_feature_ids(value: object) -> set[int]:
    if pd.isna(value) or not str(value).strip():
        return set()
    output: set[int] = set()
    for token in str(value).split(";"):
        token = token.strip()
        if token:
            output.add(int(float(token)))
    return output


def stable_fold(identity: str, folds: int) -> int:
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16) % folds


def balanced_identity_folds(candidates: pd.DataFrame, folds: int) -> dict[str, int]:
    """Assign intact identities to folds without using labels or outcomes."""
    counts = (
        candidates[list(QUERY_COLUMNS)]
        .drop_duplicates()
        .groupby("ik14", sort=False)
        .size()
        .to_dict()
    )
    identities = sorted(
        counts,
        key=lambda identity: (
            -counts[identity],
            hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        ),
    )
    fold_query_counts = [0] * folds
    fold_identity_counts = [0] * folds
    assignment: dict[str, int] = {}
    for identity in identities:
        fold = min(
            range(folds),
            key=lambda value: (fold_query_counts[value], fold_identity_counts[value], value),
        )
        assignment[identity] = fold
        fold_query_counts[fold] += int(counts[identity])
        fold_identity_counts[fold] += 1
    return assignment


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_pairwise_logistic(
    candidates: pd.DataFrame,
    feature_columns: tuple[str, ...],
    train_identities: set[str],
    l2: float = 0.1,
    maximum_iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit a deterministic pairwise ranker without sklearn.

    Every positive-minus-negative difference is mirrored, so there is no
    intercept and candidate-independent feature shifts cancel exactly.
    """
    differences: list[np.ndarray] = []
    training = candidates[candidates.ik14.astype(str).isin(train_identities)]
    for _, group in training.groupby(list(QUERY_COLUMNS), sort=False):
        positive = group[group.is_positive]
        negative = group[~group.is_positive]
        if positive.empty or negative.empty:
            continue
        p = positive.loc[:, feature_columns].to_numpy(np.float64)
        n = negative.loc[:, feature_columns].to_numpy(np.float64)
        for p_row in p:
            differences.extend(p_row - n_row for n_row in n)
    if not differences:
        raise RuntimeError("no positive-negative training pairs for the operator ranker")
    d = np.asarray(differences, dtype=np.float64)
    x = np.concatenate([d, -d], axis=0)
    y = np.concatenate([np.ones(len(d)), np.zeros(len(d))])
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    x = x / scale
    weight = np.zeros(x.shape[1], dtype=np.float64)
    converged = False
    for iteration in range(maximum_iterations):
        probability = sigmoid(x @ weight)
        curvature = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = x.T @ (probability - y) / len(y) + l2 * weight
        hessian = (x.T * curvature) @ x / len(y) + l2 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weight -= step
        if float(np.max(np.abs(step))) < 1e-8:
            converged = True
            break
    return weight, scale, {
        "positive_negative_pairs": int(len(d)),
        "iterations": int(iteration + 1),
        "converged": bool(converged),
        "l2": float(l2),
    }


def candidate_scores(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    weight: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x = frame.loc[:, feature_columns].to_numpy(np.float64)
    return (x / scale) @ weight


def exact_mcnemar_p(corrected: int, introduced: int) -> float:
    discordant = corrected + introduced
    if discordant == 0:
        return 1.0
    tail = min(corrected, introduced)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def identity_bootstrap_delta(
    per_query: pd.DataFrame,
    candidate_column: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    by_identity = (
        per_query.assign(delta=per_query[candidate_column].astype(float) - per_query.baseline_correct.astype(float))
        .groupby("ik14", sort=True).delta.mean()
        .to_numpy(np.float64)
    )
    if not len(by_identity):
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, len(by_identity), size=len(by_identity))
        values[index] = float(np.mean(by_identity[sample]))
    return {
        "identity_equal_mean": float(np.mean(by_identity)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "identities": int(len(by_identity)),
        "resamples": int(repeats),
    }


def summarize_method(
    per_query: pd.DataFrame,
    column: str,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    baseline = per_query.baseline_correct.astype(bool)
    candidate = per_query[column].astype(bool)
    corrected = int((~baseline & candidate).sum())
    introduced = int((baseline & ~candidate).sum())
    reachable = per_query.reachable_positive.astype(bool)
    return {
        "recall1_all_candidate_reachable_queries": float(candidate.mean()),
        "baseline_recall1": float(baseline.mean()),
        "delta_recall1": float(candidate.mean() - baseline.mean()),
        "reachable_positive_fraction": float(reachable.mean()),
        "recall1_when_positive_reachable": (
            float(candidate[reachable].mean()) if reachable.any() else math.nan
        ),
        "corrected": corrected,
        "introduced": introduced,
        "net": corrected - introduced,
        "mcnemar_exact_p": exact_mcnemar_p(corrected, introduced),
        "identity_cluster_bootstrap": identity_bootstrap_delta(per_query, column, repeats, seed),
    }


def load_ms1_channels(
    panels: Iterable[str],
    target_root: Path,
) -> tuple[dict[tuple[str, int, str], tuple[float, float]], pd.DataFrame]:
    intensity_lookup: dict[tuple[str, int, str], tuple[float, float]] = {}
    metadata: list[pd.DataFrame] = []
    for panel in panels:
        matrix_path = target_root / f"{panel}__discovery_intensity_matrix.csv.gz"
        target_path = target_root / f"{panel}__requantification_targets.csv.gz"
        if not matrix_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(matrix_path if not matrix_path.is_file() else target_path)
        matrix = pd.read_csv(matrix_path)
        matrix["feature_id"] = matrix.feature_id.astype(int)
        sample_columns = [column for column in matrix.columns if column != "feature_id"]
        for sample in sample_columns:
            values = pd.to_numeric(matrix[sample], errors="coerce").fillna(0.0).to_numpy(np.float64)
            percentile = pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)
            for feature, value, pct in zip(matrix.feature_id, values, percentile):
                intensity_lookup[(panel, int(feature), sample)] = (float(value), float(pct))
        targets = pd.read_csv(target_path)
        targets["panel"] = panel
        metadata.append(targets)
    return intensity_lookup, pd.concat(metadata, ignore_index=True)


def build_candidate_tensor(
    hits: pd.DataFrame,
    identities: pd.DataFrame,
    baseline: pd.DataFrame,
    intensity_lookup: dict[tuple[str, int, str], tuple[float, float]],
    target_metadata: pd.DataFrame,
) -> pd.DataFrame:
    calibration = identities[identities.calibration_panel.astype(bool)].copy()
    expected = calibration.set_index(["panel", "ik14"]).author_level1_feature_ids.to_dict()
    reference_counts = calibration.set_index(["panel", "ik14"]).n_reference_spectra.to_dict()
    query_universe = baseline[
        baseline.calibration_panel.astype(bool) & baseline.linked_feature_id.notna()
    ][list(QUERY_COLUMNS) + ["linked_feature_id", "calibration_feature_match"]].copy()
    query_universe["linked_feature_id"] = query_universe.linked_feature_id.astype(int)
    query_keys = set(map(tuple, query_universe[list(QUERY_COLUMNS)].itertuples(index=False, name=None)))

    selected = hits[hits.calibration_panel.astype(bool) & hits.linked_feature_id.notna()].copy()
    selected["linked_feature_id"] = selected.linked_feature_id.astype(int)
    selected = selected[
        [tuple(row) in query_keys for row in selected[list(QUERY_COLUMNS)].itertuples(index=False, name=None)]
    ]
    aggregations: dict[str, tuple[str, str]] = {}
    for feature in (
        "sqrt_cosine",
        "entropy_similarity",
        "top10_match_fraction",
        "intensity_coverage_min",
        "matched_peak_fraction_min",
        "neutral_loss_sqrt_cosine",
    ):
        aggregations[f"max_{feature}"] = (feature, "max")
        aggregations[f"mean_{feature}"] = (feature, "mean")
    aggregations["supporting_reference_spectra"] = ("reference_spectrum_id", "nunique")
    grouped = (
        selected.groupby(list(QUERY_COLUMNS) + ["linked_feature_id"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    grouped["eligible_reference_spectra"] = grouped.groupby(list(QUERY_COLUMNS), sort=False)[
        "supporting_reference_spectra"
    ].transform("sum")
    grouped["reference_support_fraction"] = [
        support / max(1, int(reference_counts[(panel, ik14)]))
        for panel, ik14, support in zip(grouped.panel, grouped.ik14, grouped.supporting_reference_spectra)
    ]
    grouped["neutral_loss_minus_fragment"] = (
        grouped.max_neutral_loss_sqrt_cosine - grouped.max_sqrt_cosine
    )
    grouped["unmatched_top10_fraction"] = 1.0 - grouped.max_top10_match_fraction
    grouped["is_positive"] = [
        int(feature) in parse_feature_ids(expected[(panel, ik14)])
        for panel, ik14, feature in zip(grouped.panel, grouped.ik14, grouped.linked_feature_id)
    ]
    ms1 = [
        intensity_lookup.get((panel, int(feature), sample), (0.0, 0.0))
        for panel, feature, sample in zip(grouped.panel, grouped.linked_feature_id, grouped.sample_id)
    ]
    grouped["ms1_log_intensity"] = np.log1p([value[0] for value in ms1])
    grouped["ms1_within_sample_percentile"] = [value[1] for value in ms1]

    target = target_metadata.copy()
    target["feature_id"] = target.feature_id.astype(int)
    target["group_prevalence_max"] = target[
        ["prevalence_tumor", "prevalence_normal", "prevalence_mucinous", "prevalence_tubular"]
    ].max(axis=1, skipna=True)
    grouped = grouped.merge(
        target[["panel", "feature_id", "global_prevalence", "group_prevalence_max"]],
        left_on=["panel", "linked_feature_id"],
        right_on=["panel", "feature_id"],
        how="left",
        validate="many_to_one",
    ).drop(columns="feature_id")
    grouped[["global_prevalence", "group_prevalence_max"]] = grouped[
        ["global_prevalence", "group_prevalence_max"]
    ].fillna(0.0)

    grouped["identity_channel"] = 0.5 * (
        grouped.max_sqrt_cosine + grouped.max_entropy_similarity
    )
    grouped["conserved_core_channel"] = (
        grouped.max_top10_match_fraction
        + grouped.max_intensity_coverage_min
        + grouped.max_matched_peak_fraction_min
    ) / 3.0
    grouped["transformation_channel"] = grouped.neutral_loss_minus_fragment
    grouped["neutral_loss_channel"] = grouped.max_neutral_loss_sqrt_cosine
    grouped["quantitative_channel"] = grouped.ms1_within_sample_percentile
    grouped["detectability_channel"] = grouped.global_prevalence

    baseline = query_universe.rename(
        columns={
            "linked_feature_id": "baseline_feature_id",
            "calibration_feature_match": "baseline_correct",
        }
    )
    grouped = grouped.merge(baseline, on=list(QUERY_COLUMNS), how="left", validate="many_to_one")
    if grouped.baseline_correct.isna().any():
        raise RuntimeError("candidate tensor contains rows outside the baseline query universe")
    return grouped


def evaluate_oof(
    candidates: pd.DataFrame,
    folds: int,
    l2: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fold_by_identity = balanced_identity_folds(candidates, folds)
    candidates = candidates.copy()
    candidates["outer_fold"] = candidates.ik14.astype(str).map(fold_by_identity).astype(int)
    model_audit: dict[str, object] = {}
    for model_name, features in FEATURE_SETS.items():
        score_column = f"score_{model_name}"
        candidates[score_column] = np.nan
        fold_audit: list[dict[str, object]] = []
        for fold in range(folds):
            train_ids = {identity for identity, identity_fold in fold_by_identity.items() if identity_fold != fold}
            weight, scale, fit_report = fit_pairwise_logistic(
                candidates, features, train_ids, l2=l2
            )
            test_mask = candidates.outer_fold.eq(fold)
            candidates.loc[test_mask, score_column] = candidate_scores(
                candidates.loc[test_mask], features, weight, scale
            )
            fold_audit.append({
                "fold": fold,
                "test_identities": int(candidates.loc[test_mask, "ik14"].nunique()),
                "weights": {name: float(value) for name, value in zip(features, weight / scale)},
                **fit_report,
            })
        if candidates[score_column].isna().any():
            raise RuntimeError(f"missing OOF score for {model_name}")
        model_audit[model_name] = {"features": list(features), "outer_folds": fold_audit}
    return candidates, model_audit


def per_query_decisions(candidates: pd.DataFrame) -> pd.DataFrame:
    base = (
        candidates.groupby(list(QUERY_COLUMNS), sort=False)
        .agg(
            baseline_correct=("baseline_correct", "first"),
            baseline_feature_id=("baseline_feature_id", "first"),
            reachable_positive=("is_positive", "max"),
            candidate_features=("linked_feature_id", "nunique"),
            outer_fold=("outer_fold", "first"),
        )
        .reset_index()
    )
    base["baseline_correct"] = base.baseline_correct.astype(bool)
    base["reachable_positive"] = base.reachable_positive.astype(bool)
    for model_name in FEATURE_SETS:
        score = f"score_{model_name}"
        winner = (
            candidates.sort_values(
                list(QUERY_COLUMNS) + [score, "linked_feature_id"],
                ascending=[True, True, True, False, True],
                kind="stable",
            )
            .groupby(list(QUERY_COLUMNS), sort=False, as_index=False)
            .head(1)
            [list(QUERY_COLUMNS) + ["linked_feature_id", "is_positive", score]]
            .rename(columns={
                "linked_feature_id": f"{model_name}_feature_id",
                "is_positive": f"{model_name}_correct",
                score: f"{model_name}_winning_score",
            })
        )
        base = base.merge(winner, on=list(QUERY_COLUMNS), how="left", validate="one_to_one")
        base[f"{model_name}_correct"] = base[f"{model_name}_correct"].fillna(False).astype(bool)
    return base


def fold_summaries(per_query: pd.DataFrame, column: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for fold, group in per_query.groupby("outer_fold", sort=True):
        baseline = group.baseline_correct.astype(bool)
        candidate = group[column].astype(bool)
        output.append({
            "fold": int(fold),
            "queries": int(len(group)),
            "identities": int(group.ik14.nunique()),
            "baseline_recall1": float(baseline.mean()),
            "recall1": float(candidate.mean()),
            "delta_recall1": float(candidate.mean() - baseline.mean()),
            "corrected": int((~baseline & candidate).sum()),
            "introduced": int((baseline & ~candidate).sum()),
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_raw_v3",
    )
    parser.add_argument(
        "--ms1-root", type=Path,
        default=ROOT / "data/mtbls13729/ms1_consensus",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/mtbls13729/reference_anchored_peak_operator_a0",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "hits": args.raw_dir / "reference_spectrum_sample_best_hits.csv.gz",
        "baseline": args.raw_dir / "identity_sample_evidence.csv.gz",
        "identities": args.manifest_dir / "reference_identities.csv",
        "manifest_report": args.manifest_dir / "report.json",
        "raw_report": args.raw_dir / "report.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_report = json.loads(paths["manifest_report"].read_text(encoding="utf-8"))
    if not manifest_report.get("pass_to_reverse_scan"):
        raise RuntimeError("frozen reverse-probe manifest did not pass")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    hits = pd.read_csv(paths["hits"])
    baseline = pd.read_csv(paths["baseline"])
    identities = pd.read_csv(paths["identities"])
    for frame_name, frame, columns in (
        ("hits", hits, set(QUERY_COLUMNS) | {"reference_spectrum_id", "linked_feature_id"} | set(RAW_FEATURES)),
        ("baseline", baseline, set(QUERY_COLUMNS) | {"linked_feature_id", "calibration_feature_match"}),
        ("identities", identities, {"panel", "ik14", "calibration_panel", "author_level1_feature_ids", "n_reference_spectra"}),
    ):
        missing = columns - set(frame.columns)
        if missing:
            raise RuntimeError(f"{frame_name} misses columns: {sorted(missing)}")

    panels = sorted(set(hits.panel.astype(str)))
    intensity_lookup, target_metadata = load_ms1_channels(panels, args.ms1_root)
    candidates = build_candidate_tensor(
        hits, identities, baseline, intensity_lookup, target_metadata
    )
    candidates, model_audit = evaluate_oof(candidates, args.folds, args.l2)
    per_query = per_query_decisions(candidates)

    methods = {
        name: summarize_method(
            per_query,
            f"{name}_correct",
            args.bootstrap_resamples,
            args.seed + position,
        )
        for position, name in enumerate(FEATURE_SETS)
    }
    for name in FEATURE_SETS:
        methods[name]["outer_fold_results"] = fold_summaries(
            per_query, f"{name}_correct"
        )
    spectral = methods["spectral_peak_operator"]
    full = methods["full_operator_tensor"]
    report = {
        "status": "reference_anchored_peak_operator_a0_complete",
        "formal": False,
        "purpose": "hidden-Level-1-feature falsification of a multichannel standard-probe tensor",
        "queries": int(len(per_query)),
        "identities": int(per_query.ik14.nunique()),
        "panels": per_query.panel.value_counts().sort_index().astype(int).to_dict(),
        "candidate_rows": int(len(candidates)),
        "queries_with_multiple_candidate_features": int(per_query.candidate_features.gt(1).sum()),
        "baseline": {
            "selection": "frozen maximum sqrt-cosine reverse-probe result",
            "recall1": float(per_query.baseline_correct.mean()),
        },
        "methods": methods,
        "gates": {
            "spectral_operator_identity_ci_positive": bool(
                spectral["identity_cluster_bootstrap"]["ci_low"] > 0
            ),
            "spectral_operator_corrected_gt_introduced": bool(
                spectral["corrected"] > spectral["introduced"]
            ),
            "full_tensor_identity_ci_positive": bool(
                full["identity_cluster_bootstrap"]["ci_low"] > 0
            ),
            "full_tensor_corrected_gt_introduced": bool(
                full["corrected"] > full["introduced"]
            ),
            "quantitation_adds_nonnegative_delta": bool(
                full["delta_recall1"] >= spectral["delta_recall1"]
            ),
        },
        "model_audit": model_audit,
        "contracts": {
            "phenotype_used": False,
            "outer_split": "IK14-isolated deterministic five-fold OOF",
            "identity_truth_used": "training labels inside outer train identities and evaluation only inside outer test identities",
            "public_reference_status": "not a same-platform authentic standard",
            "DDA_miss_is_absence": False,
            "transformation_channel_status": "proxy from fragment-vs-neutral-loss behavior; no structural-site claim",
        },
        "provenance": {name: sha256(path) for name, path in paths.items()},
        "claim_limit": (
            "A0 tests whether factorized evidence improves recovery of already known Level-1 feature coordinates. "
            "It does not validate a new metabolite, transformation structure, phenotype association, or atlas-scale novelty."
        ),
    }
    candidates_path = output / "operator_candidate_tensor.csv.gz"
    query_path = output / "per_query_hidden_truth.csv.gz"
    candidates.to_csv(candidates_path, index=False, compression="gzip")
    per_query.to_csv(query_path, index=False, compression="gzip")
    report["provenance"]["candidate_tensor"] = sha256(candidates_path)
    report["provenance"]["per_query"] = sha256(query_path)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Cross-dataset calibration of deployable BioAware seed reliability."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


FEATURES = ("top_score", "top_margin", "log1p_candidate_count")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float((centre - spread) / denominator)


def calibration_examples(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in ledger.groupby("query_id", sort=False):
        group = group.sort_values(
            ["spectral_score", "candidate_id"], ascending=[False, True]
        ).reset_index(drop=True)
        if len(group) < 2:
            continue
        truths = group["truth_candidate_id"].astype(str).unique()
        formulas = group["truth_formula"].astype(str).unique()
        if len(truths) != 1 or len(formulas) != 1:
            raise RuntimeError(f"query {query_id} truth/formula is not unique")
        top = group.iloc[0]
        second = group.iloc[1]
        rows.append(
            {
                "query_id": str(query_id),
                "truth_formula": formulas[0],
                "top_candidate_id": str(top.candidate_id),
                "top_score": float(top.spectral_score),
                "top_margin": float(top.spectral_score - second.spectral_score),
                "candidate_count": int(len(group)),
                "log1p_candidate_count": float(np.log1p(len(group))),
                "top_correct": int(str(top.candidate_id) == truths[0]),
            }
        )
    return pd.DataFrame(rows)


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, FEATURES].to_numpy(float)


def apply_linear_probability(
    frame: pd.DataFrame,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
) -> np.ndarray:
    standardized = (matrix(frame) - mean) / scale
    logit = standardized @ coefficient + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "data/validation/bioaware_candidate_evidence_ledger_v1/"
            "candidate_evidence.csv.gz"
        ),
    )
    parser.add_argument(
        "--qc-audit",
        type=Path,
        default=Path(
            "data/external/MTBLS1905/bioaware_a0_seed_rebuild_v2_20260830/"
            "auto_seed_audit.csv.gz"
        ),
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path(
            "data/reference/bioaware_rhea_offline_20260827/"
            "rhea_participants.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_seed_calibration_20260830"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-oof-seeds", type=int, default=30)
    parser.add_argument("--minimum-oof-precision", type=float, default=0.98)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.90)
    parser.add_argument("--minimum-per-fold-seeds", type=int, default=3)
    parser.add_argument("--minimum-per-fold-precision", type=float, default=0.80)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_csv(args.ledger)
    examples = calibration_examples(ledger)
    if len(examples) < 100 or examples["truth_formula"].nunique() < 50:
        raise RuntimeError("seed calibration set is too small")
    x = matrix(examples)
    y = examples["top_correct"].to_numpy(int)
    groups = examples["truth_formula"].astype(str).to_numpy()
    oof_probability = np.full(len(examples), np.nan, dtype=float)
    oof_fold = np.full(len(examples), -1, dtype=int)
    splitter = GroupKFold(n_splits=args.folds)
    fold_artifacts: list[dict[str, object]] = []
    for fold, (train, validation) in enumerate(splitter.split(x, y, groups)):
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=20260830)
        model.fit(scaler.transform(x[train]), y[train])
        oof_probability[validation] = model.predict_proba(
            scaler.transform(x[validation])
        )[:, 1]
        oof_fold[validation] = fold
        fold_artifacts.append(
            {
                "fold": fold,
                "scaler_mean": scaler.mean_.copy(),
                "scaler_scale": scaler.scale_.copy(),
                "coefficient": model.coef_[0].copy(),
                "intercept": float(model.intercept_[0]),
            }
        )
    if not np.isfinite(oof_probability).all() or np.any(oof_fold < 0):
        raise RuntimeError("OOF calibration predictions are incomplete")
    examples["oof_probability"] = oof_probability
    examples["oof_fold"] = oof_fold

    threshold_candidates = np.unique(oof_probability)
    eligible_thresholds: list[tuple[float, int, float, float, list[dict]]] = []
    for threshold in threshold_candidates:
        selected = oof_probability >= threshold
        total = int(selected.sum())
        successes = int(y[selected].sum())
        if total < args.minimum_oof_seeds:
            continue
        precision = successes / total
        lower = wilson_lower(successes, total)
        folds: list[dict] = []
        fold_pass = True
        for fold in range(args.folds):
            fold_selected = selected & (oof_fold == fold)
            fold_total = int(fold_selected.sum())
            fold_successes = int(y[fold_selected].sum())
            fold_precision = fold_successes / fold_total if fold_total else 0.0
            folds.append(
                {
                    "fold": fold,
                    "selected": fold_total,
                    "correct": fold_successes,
                    "precision": fold_precision,
                }
            )
            fold_pass &= (
                fold_total >= args.minimum_per_fold_seeds
                and fold_precision >= args.minimum_per_fold_precision
            )
        if (
            precision >= args.minimum_oof_precision
            and lower >= args.minimum_wilson_lower
            and fold_pass
        ):
            eligible_thresholds.append((threshold, total, precision, lower, folds))
    if not eligible_thresholds:
        raise RuntimeError("no OOF seed threshold passed the frozen reliability gate")
    # Lowest passing probability maximizes coverage under the frozen risk gate.
    threshold, selected_total, selected_precision, selected_lower, fold_summary = min(
        eligible_thresholds, key=lambda item: item[0]
    )

    qc = pd.read_csv(args.qc_audit)
    evaluable = (
        pd.to_numeric(qc["top_score"], errors="coerce").notna()
        & pd.to_numeric(qc["top_margin"], errors="coerce").notna()
        & (pd.to_numeric(qc["candidate_count"], errors="coerce") >= 2)
    )
    qc["log1p_candidate_count"] = np.log1p(
        pd.to_numeric(qc["candidate_count"], errors="coerce").fillna(0)
    )
    qc["calibrated_seed_probability"] = np.nan
    # The selection threshold was learned from cross-fitted probabilities.
    # Deployment therefore uses the mean probability from the same five
    # cross-fitted models, rather than silently changing probability scale by
    # fitting a sixth model on all calibration examples.
    deployment_probability = np.column_stack(
        [
            apply_linear_probability(
                qc.loc[evaluable],
                mean=np.asarray(artifact["scaler_mean"], dtype=float),
                scale=np.asarray(artifact["scaler_scale"], dtype=float),
                coefficient=np.asarray(artifact["coefficient"], dtype=float),
                intercept=float(artifact["intercept"]),
            )
            for artifact in fold_artifacts
        ]
    ).mean(axis=1)
    qc.loc[evaluable, "calibrated_seed_probability"] = deployment_probability
    graph = set(
        pd.read_csv(args.participants, usecols=["compound_id"])["compound_id"]
        .dropna()
        .astype(str)
    )
    qc["calibrated_seed_selected"] = (
        evaluable
        & qc["top_candidate_id"].astype(str).isin(graph)
        & (qc["calibrated_seed_probability"] >= threshold)
    )
    selected_qc = qc[qc["calibrated_seed_selected"]].copy()
    seeds = selected_qc[
        ["seed_query_id", "top_candidate_id", "calibrated_seed_probability"]
    ].rename(
        columns={
            "top_candidate_id": "seed_compound_id",
            "calibrated_seed_probability": "seed_score",
        }
    )
    seeds["reference_kind"] = (
        "deployable_cross_dataset_calibrated_dreams_top1"
    )

    examples_path = args.output_dir / "calibration_examples_oof.csv.gz"
    qc_path = args.output_dir / "qc_seed_probabilities.csv.gz"
    seeds_path = args.output_dir / "seeds_auto_calibrated.csv"
    artifact_path = args.output_dir / "calibrator.json"
    report_path = args.output_dir / "report.json"
    examples.to_csv(examples_path, index=False)
    qc.to_csv(qc_path, index=False)
    seeds.to_csv(seeds_path, index=False)
    artifact = {
        "features": list(FEATURES),
        "deployment": "mean_probability_of_formula_group_crossfit_models",
        "fold_models": [
            {
                "fold": int(item["fold"]),
                "scaler_mean": np.asarray(item["scaler_mean"]).tolist(),
                "scaler_scale": np.asarray(item["scaler_scale"]).tolist(),
                "coefficient": np.asarray(item["coefficient"]).tolist(),
                "intercept": float(item["intercept"]),
            }
            for item in fold_artifacts
        ],
        "probability_threshold": float(threshold),
        "training_ledger_sha256": sha256(args.ledger),
        "qc_audit_sha256": sha256(args.qc_audit),
    }
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    report = {
        "status": "bioaware_seed_reliability_calibration_complete",
        "formal": True,
        "calibration_queries": int(len(examples)),
        "calibration_formulas": int(examples["truth_formula"].nunique()),
        "calibration_baseline_accuracy": float(y.mean()),
        "oof_selection": {
            "threshold": float(threshold),
            "selected": int(selected_total),
            "correct": int(round(selected_precision * selected_total)),
            "precision": float(selected_precision),
            "wilson_95ci_lower": float(selected_lower),
            "folds": fold_summary,
        },
        "mtbls1905_qc_application": {
            "audit_rows": int(len(qc)),
            "evaluable_rows": int(evaluable.sum()),
            "selected_graph_seeds": int(len(seeds)),
            "selected_identities": int(seeds["seed_compound_id"].nunique()),
        },
        "contracts": {
            "mtbls1905_truth_labels_used_for_threshold": False,
            "formula_group_oof": True,
            "deployment_probability_matches_oof_model_family": True,
            "phenotype_labels_used": False,
            "deployment_requires_external_seed_precision_revalidation": True,
        },
        "claim_limit": (
            "Cross-dataset seed calibration, not BioAware annotation gain. "
            "MTBLS1905 seed precision remains unlabelled outside its known panel."
        ),
        "provenance": {
            "ledger_sha256": sha256(args.ledger),
            "qc_audit_sha256": sha256(args.qc_audit),
            "participants_sha256": sha256(args.participants),
            "calibrator_sha256": sha256(artifact_path),
            "seeds_sha256": sha256(seeds_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

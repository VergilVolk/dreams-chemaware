#!/usr/bin/env python
"""Nested leave-study-out BioAware rank-consensus evaluation.

This is a cross-study development protocol, not a blind-test protocol.  Each
outer study is scored by weights and an abstention gate learned without any
query or outcome from that study.  The inner gate is itself selected by
leave-one-study-out predictions among the remaining studies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES,
    add_family_features,
    apply_gate,
    fit_family_weights,
    score_queries,
    select_gate,
)


UNITS = (
    "BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic",
    "Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
    "NIST_plasma__hilic", "NIST_plasma__rplc",
)
STUDY = {
    "BV2cell__hilic": "BV2cell", "BV2cell__rplc": "BV2cell",
    "Mouse_brain__hilic": "Mouse_brain", "Mouse_brain__rplc": "Mouse_brain",
    "Mouse_liver__hilic": "Mouse_liver", "Mouse_liver__rplc": "Mouse_liver",
    "NIST_plasma__hilic": "NIST_plasma", "NIST_plasma__rplc": "NIST_plasma",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ledgers(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for unit in UNITS:
        path = root / unit / "ledger" / "candidate_evidence.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
            "spectral_score",
        }
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} lacks {sorted(missing)}")
        frame = frame.copy()
        frame["source_query_id"] = frame.query_id.astype(str)
        frame["query_id"] = unit + "|" + frame.source_query_id
        frame["unit_id"] = unit
        frame["study_id"] = STUDY[unit]
        parts.append(frame)
        hashes[unit] = sha256(path)
    ledger = pd.concat(parts, ignore_index=True)
    if ledger[["query_id", "candidate_id"]].duplicated().any():
        raise RuntimeError("candidate rows overlap after unit namespacing")
    study_per_query = ledger.groupby("query_id").study_id.nunique()
    if not study_per_query.eq(1).all():
        raise RuntimeError("a query spans multiple studies")
    return add_family_features(ledger), hashes


def inner_study_oof(train: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    studies = sorted(train.study_id.unique())
    if len(studies) != 3:
        raise RuntimeError(f"outer training must contain three studies, found {studies}")
    for heldout in studies:
        fit = train[train.study_id != heldout]
        validation = train[train.study_id == heldout]
        if fit.query_id.nunique() < 100 or validation.query_id.nunique() < 50:
            raise RuntimeError(f"insufficient inner study coverage for {heldout}")
        weight = fit_family_weights(
            fit, args.temperature, args.l2, args.maximum_family_weight
        )
        prediction = score_queries(validation, weight)
        prediction["inner_study"] = heldout
        parts.append(prediction)
    result = pd.concat(parts, ignore_index=True)
    if result.query_id.duplicated().any():
        raise RuntimeError("inner study OOF predictions overlap")
    if set(result.query_id) != set(train.query_id.unique()):
        raise RuntimeError("inner study OOF coverage mismatch")
    return result


def query_metrics(frame: pd.DataFrame) -> dict:
    corrected = int(frame.corrected.sum())
    introduced = int(frame.introduced.sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)),
        "formulas": int(frame.truth_formula.nunique()),
        "baseline_recall1": float(frame.baseline_correct.mean()),
        "recall1": float(frame.final_correct.mean()),
        "delta_recall1": float(frame.final_correct.mean() - frame.baseline_correct.mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(frame.intervene.sum()),
        "mcnemar_exact_p": (
            float(binomtest(min(corrected, introduced), discordant, .5).pvalue)
            if discordant else 1.0
        ),
    }


def cluster_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    work = frame.copy()
    # Cluster repeated observations of one chemical formula across all studies
    # together; study-prefixing would understate uncertainty when formulas recur.
    work["cluster"] = work.truth_formula.astype(str)
    groups = {str(key): group for key, group in work.groupby("cluster", sort=True)}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        draw = rng.choice(keys, len(keys), replace=True)
        sample = pd.concat([groups[str(key)] for key in draw], ignore_index=True)
        values[index] = float(sample.final_correct.mean() - sample.baseline_correct.mean())
    return {
        "mean": float(work.final_correct.mean() - work.baseline_correct.mean()),
        "ci_low": float(np.quantile(values, .025)),
        "ci_high": float(np.quantile(values, .975)),
        "clusters": len(keys),
        "resamples": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_v5_leave_study_out_v1"),
    )
    parser.add_argument("--temperature", type=float, default=.10)
    parser.add_argument("--l2", type=float, default=.05)
    parser.add_argument("--maximum-family-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")

    ledger, hashes = load_ledgers(args.root)
    predictions: list[pd.DataFrame] = []
    fold_reports: dict[str, dict] = {}
    studies = sorted(ledger.study_id.unique())
    if len(studies) != 4:
        raise RuntimeError(f"expected four external studies, found {studies}")
    for outer_study in studies:
        train = ledger[ledger.study_id != outer_study].copy()
        test = ledger[ledger.study_id == outer_study].copy()
        inner = inner_study_oof(train, args)
        gate = select_gate(inner)
        weights = fit_family_weights(
            train, args.temperature, args.l2, args.maximum_family_weight
        )
        scored = apply_gate(score_queries(test, weights), gate)
        scored["outer_study"] = outer_study
        scored["study_id"] = outer_study
        training_formulas = set(train.truth_formula.astype(str))
        scored["formula_seen_in_training_studies"] = scored.truth_formula.astype(str).isin(
            training_formulas
        )
        predictions.append(scored)
        fold_reports[outer_study] = {
            **query_metrics(scored),
            "training_queries": int(train.query_id.nunique()),
            "inner_oof_queries": int(len(inner)),
            "inner_oof_gate": {
                "maximum_spectral_margin": float(gate[0]),
                "minimum_fusion_advantage": (
                    float(gate[1]) if np.isfinite(gate[1]) else None
                ),
                "minimum_support_families": int(gate[2]),
            },
            "weights": dict(zip(FAMILY_FEATURES, map(float, weights), strict=True)),
        }

    pooled = pd.concat(predictions, ignore_index=True)
    if pooled.query_id.duplicated().any():
        raise RuntimeError("outer-study OOF query predictions overlap")
    expected = set(ledger.query_id.unique())
    if set(pooled.query_id) != expected:
        raise RuntimeError("outer-study OOF coverage mismatch")
    overall = query_metrics(pooled)
    bootstrap = cluster_bootstrap(pooled, args.bootstrap_resamples, args.seed)
    seen = pooled[pooled.formula_seen_in_training_studies]
    unseen = pooled[~pooled.formula_seen_in_training_studies]
    every_study_nonnegative = all(
        value["delta_recall1"] >= 0 for value in fold_reports.values()
    )
    gates = {
        "all_four_studies_present": len(fold_reports) == 4,
        "study_formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
        "corrected_gt_introduced": overall["corrected"] > overall["introduced"],
        "risk_weighted_net_lambda2_positive": overall["risk_weighted_net_lambda2"] > 0,
        "every_outer_study_nonnegative": every_study_nonnegative,
        "unseen_formula_nonnegative": unseen.empty or query_metrics(unseen)["delta_recall1"] >= 0,
    }
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "query_oof_transitions.csv.gz"
    pooled.to_csv(prediction_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_v5_leave_study_out_complete",
        "formal": True,
        "protocol": (
            "nested leave-study-out: outer study never contributes to weights or gate; "
            "gate selected from leave-study-out predictions inside the remaining studies"
        ),
        "overall": overall,
        "study_formula_cluster_bootstrap": bootstrap,
        "seen_formula": query_metrics(seen) if not seen.empty else None,
        "unseen_formula": query_metrics(unseen) if not unseen.empty else None,
        "outer_studies": fold_reports,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "contracts": {
            "outer_outcomes_used_for_training": False,
            "candidate_protocol_matched": True,
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "claim_type": "cross-study OOF development, not final blind test",
        },
        "provenance": {
            "ledgers": hashes,
            "transitions_sha256": sha256(prediction_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A positive result establishes cross-study transfer under nested OOF. "
            "It is not an untouched external blind result and cannot alone support SOTA."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

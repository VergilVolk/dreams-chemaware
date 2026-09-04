#!/usr/bin/env python
"""Freeze the discovered negative-ion network-only BioAware expert.

The artifact contains plain numerical preprocessing/model parameters and fixed
deployment gates.  It deliberately excludes sklearn pickles and all phenotype
or P2b information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS
    from develop_bioaware_metdna3_negative_loso_ranker import (
        PRIMARY_BASELINE_MARGIN_MAX,
        PRIMARY_C,
        PRIMARY_PROPOSAL_PROBABILITY_MIN,
        RISK_WEIGHT_BASELINE_CORRECT,
        pairwise_training_rows,
    )
except ModuleNotFoundError:  # pragma: no cover
    from tasks.audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS
    from tasks.develop_bioaware_metdna3_negative_loso_ranker import (
        PRIMARY_BASELINE_MARGIN_MAX,
        PRIMARY_C,
        PRIMARY_PROPOSAL_PROBABILITY_MIN,
        RISK_WEIGHT_BASELINE_CORRECT,
        pairwise_training_rows,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--source-loso-report", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_source_loso_v1/report.json"),
    )
    parser.add_argument(
        "--permutation-report", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_candidate_permutation_v1/report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_negative_network_expert_v1"),
    )
    parser.add_argument(
        "--recipe-name", choices=sorted(ABLATIONS),
        default="network_only_same_edge_gate",
    )
    parser.add_argument(
        "--library-integrity-report", type=Path, default=None,
        help="Optional chemical-integrity report required by the v2 [M-H]- artifact.",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    source_report = json.loads(args.source_loso_report.read_text(encoding="utf-8"))
    permutation_report = json.loads(args.permutation_report.read_text(encoding="utf-8"))
    if source_report.get("status") != "bioaware_metdna3_negative_source_loso_complete":
        raise RuntimeError("source-LOSO evidence is missing")
    if permutation_report.get("pass") is not True:
        raise RuntimeError("candidate-permutation falsification did not pass")
    if source_report.get("recipe_name", args.recipe_name) != args.recipe_name:
        raise RuntimeError("source-LOSO report recipe does not match requested frozen recipe")
    if permutation_report.get("recipe_name", args.recipe_name) != args.recipe_name:
        raise RuntimeError("candidate-permutation report recipe does not match requested frozen recipe")
    integrity_report = None
    if args.library_integrity_report is not None:
        integrity_report = json.loads(args.library_integrity_report.read_text(encoding="utf-8"))
        if integrity_report.get("status") != "mona_negative_library_chemical_integrity_complete":
            raise RuntimeError("invalid MONA negative chemical-integrity report")
        if integrity_report.get("declared_adduct_scope") != "[M-H]-":
            raise RuntimeError("chemical-integrity report does not cover [M-H]-")
    candidates = pd.read_csv(args.candidate_features)
    recipe = ABLATIONS[args.recipe_name]
    features = recipe["features"]
    x, y, weights = pairwise_training_rows(candidates, features)
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = LogisticRegression(
        C=PRIMARY_C, fit_intercept=False, solver="lbfgs", max_iter=2000,
        random_state=20260901,
    ).fit(transformed, y, sample_weight=weights)
    sklearn_score = model.decision_function(transformed)
    manual_score = ((x - scaler.mean_) / scaler.scale_) @ model.coef_[0]
    maximum_reconstruction_error = float(np.max(np.abs(sklearn_score - manual_score)))
    if maximum_reconstruction_error > 1e-10:
        raise RuntimeError(f"plain-array score reconstruction failed: {maximum_reconstruction_error}")
    artifact = {
        "status": "bioaware_metdna3_negative_network_expert_frozen",
        "version": 2 if integrity_report is not None else 1,
        "scope": "negative [M-H]- candidate ranking with DreaMS and sample-level MetDNA2/KGMN-style context",
        "recipe_name": args.recipe_name,
        "feature_names": features,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "model_coef": model.coef_[0].astype(float).tolist(),
        "model_intercept": 0.0,
        "configuration": {
            "pairwise_logistic_C": PRIMARY_C,
            "baseline_correct_training_weight": RISK_WEIGHT_BASELINE_CORRECT,
            "maximum_dreams_top1_top2_gap": PRIMARY_BASELINE_MARGIN_MAX,
            "minimum_pairwise_proposal_probability": PRIMARY_PROPOSAL_PROBABILITY_MIN,
            "requires_raw_step0_edge_validation": bool(recipe["require_raw_step0_edge"]),
            "tie_policy": "abstain",
        },
        "training_scope": {
            "queries": int(candidates["query_id"].nunique()),
            "truth_identities": int(candidates["truth_candidate_id"].nunique()),
            "truth_formulas": int(candidates["truth_formula"].nunique()),
            "biological_units": sorted(candidates["unit_id"].astype(str).unique()),
            "pairwise_rows": int(len(x)),
        },
        "contracts": {
            "inference_uses_candidate_identity_and_sample_network_context": True,
            "inference_uses_truth": False,
            "inference_uses_phenotype": False,
            "P2b": "forbidden",
            "shared_embedding_changed": False,
            "unsupported_negative_adducts": "excluded; current evidence is [M-H]- only",
        },
        "provenance": {
            "candidate_features_sha256": sha256(args.candidate_features),
            "source_loso_report_sha256": sha256(args.source_loso_report),
            "candidate_permutation_report_sha256": sha256(args.permutation_report),
            "library_integrity_report_sha256": (
                sha256(args.library_integrity_report)
                if args.library_integrity_report is not None else None
            ),
        },
        "validation_evidence_pointer": {
            "source_identity_formula_purged_delta_recall1": source_report["results"]["source_identity_formula_purged"]["pooled"]["delta_recall1"],
            "source_identity_formula_purged_corrected": source_report["results"]["source_identity_formula_purged"]["pooled"]["corrected"],
            "source_identity_formula_purged_introduced": source_report["results"]["source_identity_formula_purged"]["pooled"]["introduced"],
            "candidate_permutation_empirical_p": permutation_report["null_metrics"]["delta_recall1"]["empirical_one_sided_p_ge_observed"],
            "chemically_approved_m_h_library_rows": (
                integrity_report["approved_m_h_rows"] if integrity_report is not None else None
            ),
        },
        "implementation_validation": {
            "plain_array_reconstruction_max_abs_error": maximum_reconstruction_error,
            "passed": True,
        },
        "claim_limit": "Frozen after opened-cohort discovery. Requires a new independent cohort before confirmatory or SOTA claims.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    report = {
        "status": "bioaware_metdna3_negative_network_expert_freeze_complete",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "score_reconstruction_pass": True,
        "ready_for_new_external_validation": True,
        "not_a_sota_claim": True,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

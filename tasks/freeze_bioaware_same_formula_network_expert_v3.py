#!/usr/bin/env python
"""Freeze the protocol-matched BioAware same-formula v3 expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation.bioaware_negative_expert import FrozenNegativeBioAwareExpert  # noqa: E402
from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    PRIMARY_BASELINE_MARGIN_MAX,
    PRIMARY_C,
    PRIMARY_PROPOSAL_PROBABILITY_MIN,
    RISK_WEIGHT_BASELINE_CORRECT,
    pairwise_training_rows,
)
from develop_bioaware_same_formula_network_expert_v3 import FEATURES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_development"),
    )
    parser.add_argument(
        "--permutation-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_permutation"),
    )
    parser.add_argument(
        "--integrity-report", type=Path,
        default=Path("data/validation/mona_negative_library_chemical_integrity_v1/report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_frozen"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidate_path = args.development_dir / "candidate_features.csv.gz"
    development_path = args.development_dir / "report.json"
    permutation_path = args.permutation_dir / "report.json"
    for path in (candidate_path, development_path, permutation_path, args.integrity_report):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    permutation = json.loads(permutation_path.read_text(encoding="utf-8"))
    integrity = json.loads(args.integrity_report.read_text(encoding="utf-8"))
    if development.get("pass_to_falsification") is not True:
        raise RuntimeError("v3 development gates did not pass")
    if permutation.get("pass_to_freeze") is not True:
        raise RuntimeError("v3 candidate-permutation falsification did not pass")
    if integrity.get("status") != "mona_negative_library_chemical_integrity_complete":
        raise RuntimeError("chemical-integrity report is invalid")
    if sha256(candidate_path) != development["provenance"]["candidate_features_sha256"]:
        raise RuntimeError("v3 candidate feature hash mismatch")
    if development["features"] != FEATURES or permutation["features"] != FEATURES:
        raise RuntimeError("v3 feature recipe changed before freeze")
    candidates = pd.read_csv(candidate_path)
    if "known_mass_candidate_fraction" in FEATURES:
        raise RuntimeError("protocol-mismatched mass recovery shortcut re-entered v3")
    if not candidates["calculated_formula"].astype(str).eq(
        candidates["truth_formula"].astype(str)
    ).all():
        raise RuntimeError("v3 freeze input is not same-formula only")
    x, y, weights = pairwise_training_rows(candidates, FEATURES)
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = LogisticRegression(
        C=PRIMARY_C, fit_intercept=False, solver="lbfgs", max_iter=2000,
        random_state=20260901,
    ).fit(transformed, y, sample_weight=weights)
    manual = ((x - scaler.mean_) / scaler.scale_) @ model.coef_[0]
    sklearn = model.decision_function(transformed)
    maximum_error = float(np.max(np.abs(manual - sklearn)))
    if maximum_error > 1e-10:
        raise RuntimeError(f"plain-array reconstruction failed: {maximum_error}")
    artifact = {
        "status": "bioaware_metdna3_negative_network_expert_frozen",
        "version": 3,
        "scope": (
            "negative exact [M-H]- same-formula candidate ranking with DreaMS and "
            "sample-level MetDNA2/KGMN-style context"
        ),
        "recipe_name": "same_formula_full_without_mass_recovery",
        "feature_names": FEATURES,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "model_coef": model.coef_[0].astype(float).tolist(),
        "model_intercept": 0.0,
        "configuration": {
            "pairwise_logistic_C": PRIMARY_C,
            "baseline_correct_training_weight": RISK_WEIGHT_BASELINE_CORRECT,
            "maximum_dreams_top1_top2_gap": PRIMARY_BASELINE_MARGIN_MAX,
            "minimum_pairwise_proposal_probability": PRIMARY_PROPOSAL_PROBABILITY_MIN,
            "requires_raw_step0_edge_validation": True,
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
            "same_formula_candidates_required": True,
            "known_mass_candidate_fraction_forbidden": True,
            "inference_uses_truth": False,
            "inference_uses_phenotype": False,
            "P2b": "forbidden",
            "shared_embedding_changed": False,
            "unsupported_adducts": "excluded; exact [M-H]- only",
        },
        "provenance": {
            "candidate_features_sha256": sha256(candidate_path),
            "development_report_sha256": sha256(development_path),
            "permutation_report_sha256": sha256(permutation_path),
            "library_integrity_report_sha256": sha256(args.integrity_report),
            "script_sha256": sha256(Path(__file__)),
        },
        "validation_evidence_pointer": {
            "same_formula_identity_formula_purged_delta_recall1": development["pooled"]["delta_recall1"],
            "same_formula_corrected": development["pooled"]["corrected"],
            "same_formula_introduced": development["pooled"]["introduced"],
            "formula_cluster_ci_low": development["formula_cluster_bootstrap"]["ci_low"],
            "candidate_permutation_delta_p": permutation["null_metrics"]["delta_recall1"]["empirical_one_sided_p_ge_observed"],
            "candidate_permutation_risk_net_p": permutation["null_metrics"]["risk_weighted_net_lambda2"]["empirical_one_sided_p_ge_observed"],
            "chemically_approved_m_h_library_rows": integrity["approved_m_h_rows"],
        },
        "implementation_validation": {
            "plain_array_reconstruction_max_abs_error": maximum_error,
            "passed": True,
        },
        "claim_limit": (
            "Frozen after protocol-matched development and permutation falsification. Requires a "
            "new untouched holdout before confirmatory or SOTA claims."
        ),
    }
    args.output_dir.mkdir(parents=True)
    artifact_path = args.output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    loaded = FrozenNegativeBioAwareExpert.load(artifact_path)
    reconstruction = loaded.score(x)
    loader_error = float(np.max(np.abs(reconstruction - sklearn)))
    if loader_error > 1e-10:
        raise RuntimeError(f"runtime loader reconstruction failed: {loader_error}")
    report = {
        "status": "bioaware_same_formula_network_expert_v3_freeze_complete",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "score_reconstruction_max_abs_error": maximum_error,
        "runtime_loader_max_abs_error": loader_error,
        "ready_for_new_holdout": True,
        "not_a_sota_claim": True,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

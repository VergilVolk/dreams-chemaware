"""Fail-closed validation for the reference-anchored peak-operator A0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
QUERY = ["panel", "ik14", "sample_id"]
FORBIDDEN_MODEL_TOKENS = ("tissue", "histology", "patient", "rmu", "tumor", "phenotype", "p2b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path,
        default=ROOT / "data/mtbls13729/reference_anchored_peak_operator_a0_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads((args.result_dir / "report.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.result_dir / "operator_candidate_tensor.csv.gz")
    per_query = pd.read_csv(args.result_dir / "per_query_hidden_truth.csv.gz")
    if report["status"] != "reference_anchored_peak_operator_a0_complete":
        raise RuntimeError("unexpected A0 status")
    if report["formal"]:
        raise RuntimeError("A0 must not be marked formal")
    if len(candidates) != report["candidate_rows"] or len(per_query) != report["queries"]:
        raise RuntimeError("reported row counts do not reproduce")
    if per_query.duplicated(QUERY).any():
        raise RuntimeError("duplicate query decisions")
    if candidates.duplicated(QUERY + ["linked_feature_id"]).any():
        raise RuntimeError("duplicate candidate feature rows")
    if candidates.groupby("ik14").outer_fold.nunique().max() != 1:
        raise RuntimeError("an identity crosses outer folds")
    if int(per_query.ik14.nunique()) != report["identities"]:
        raise RuntimeError("identity count does not reproduce")
    baseline = float(per_query.baseline_correct.astype(bool).mean())
    if not np.isclose(baseline, report["baseline"]["recall1"], atol=1e-12):
        raise RuntimeError("baseline Recall@1 does not reproduce")

    all_features: list[str] = []
    for name, audit in report["model_audit"].items():
        features = [str(value) for value in audit["features"]]
        all_features.extend(features)
        if any(token in feature.casefold() for feature in features for token in FORBIDDEN_MODEL_TOKENS):
            raise RuntimeError(f"forbidden feature in model {name}")
        score_column = f"score_{name}"
        if score_column not in candidates or not np.isfinite(candidates[score_column]).all():
            raise RuntimeError(f"invalid OOF scores for {name}")
        correct_column = f"{name}_correct"
        candidate = per_query[correct_column].astype(bool)
        base = per_query.baseline_correct.astype(bool)
        corrected = int((~base & candidate).sum())
        introduced = int((base & ~candidate).sum())
        result = report["methods"][name]
        if corrected != result["corrected"] or introduced != result["introduced"]:
            raise RuntimeError(f"transition counts do not reproduce for {name}")
        if not np.isclose(float(candidate.mean()), result["recall1_all_candidate_reachable_queries"]):
            raise RuntimeError(f"Recall@1 does not reproduce for {name}")

    if report["gates"]["spectral_operator_identity_ci_positive"]:
        raise RuntimeError("spectral operator was incorrectly promoted despite non-positive CI lower bound")
    if report["gates"]["full_tensor_identity_ci_positive"]:
        raise RuntimeError("full tensor was incorrectly promoted despite non-positive CI lower bound")
    if report["contracts"]["phenotype_used"]:
        raise RuntimeError("phenotype use is forbidden in A0")
    print(
        "[validate_reference_anchored_peak_operator_a0] PASS "
        f"queries={len(per_query):,} identities={per_query.ik14.nunique():,} "
        f"model_features={len(set(all_features)):,}"
    )


if __name__ == "__main__":
    main()

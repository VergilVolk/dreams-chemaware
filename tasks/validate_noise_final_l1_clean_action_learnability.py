"""Fail-closed validator for L1 clean-input formula-crossfit outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_l1_clean_action_learnability")
    parser.add_argument("--l0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_l0_action_learnability_ledger")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = {
        "report": args.output_dir / "report.json",
        "oof": args.output_dir / "action_oof_predictions.csv.gz",
        "query": args.output_dir / "primary_per_query.csv.gz",
        "false_positive": args.output_dir / "primary_false_positive_audit.csv.gz",
        "features": args.output_dir / "clean_query_features.npz",
        "l0_report": args.l0_dir / "report.json",
        "l0_labels": args.l0_dir / "action_labels.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing: raise FileNotFoundError(f"L1 output is incomplete: {missing}")
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_l1_clean_action_learnability_complete" or not report.get("formal"):
        raise RuntimeError("L1 report status/formality mismatch")
    if report.get("provenance", {}).get("l0_report_sha256") != sha256_file(required["l0_report"]):
        raise RuntimeError("L0 report changed after L1")
    if report.get("provenance", {}).get("l0_labels_sha256") != sha256_file(required["l0_labels"]):
        raise RuntimeError("L0 labels changed after L1")
    oof = pd.read_csv(required["oof"], low_memory=False)
    l0 = pd.read_csv(required["l0_labels"], low_memory=False)
    predictions = [column for column in oof if column.endswith(("_pred_gain", "_p_positive", "_p_harmful"))]
    if len(oof) != len(l0) or len(oof) != int(report.get("actions", -1)): raise RuntimeError("L1 lost L0 actions")
    keys = ["action_index", "query_index", "selector", "attenuation", "step"]
    if not oof[keys].equals(l0[keys]): raise RuntimeError("L1 action identity/order differs from L0")
    if len(predictions) != 9 or not np.all(np.isfinite(oof[predictions].to_numpy(float))):
        raise RuntimeError("L1 OOF predictions are incomplete")
    probability = oof[[column for column in predictions if "_p_" in column]].to_numpy(float)
    if np.any((probability < 0) | (probability > 1)): raise RuntimeError("L1 probability outside [0,1]")
    per_query = pd.read_csv(required["query"], low_memory=False)
    if per_query["query_index"].duplicated().any() or len(per_query) != oof["query_index"].nunique():
        raise RuntimeError("L1 primary policy is not one-decision-per-query")
    false_positive = pd.read_csv(required["false_positive"], low_memory=False)
    if len(false_positive) != int(report.get("primary_false_positive_actions", -1)):
        raise RuntimeError("L1 false-positive audit count mismatch")
    with np.load(required["features"]) as body:
        if len(body["query_index"]) != oof["query_index"].nunique(): raise RuntimeError("L1 feature cache misses queries")
        if body["features"].shape[1] >= int(report["clean_query_feature_dimension"]):
            raise RuntimeError("L1 query/action feature dimensions are inconsistent")
        if not np.all(np.isfinite(body["features"])): raise RuntimeError("L1 feature cache contains non-finite values")
        if len(body["feature_names"]) != body["features"].shape[1]:
            raise RuntimeError("L1 feature names do not align with the feature cache")
        forbidden = ("identity", "formula", "rank", "margin", "target", "control", "outcome", "path", "p2b")
        if any(any(token in str(name).lower() for token in forbidden) for name in body["feature_names"]):
            raise RuntimeError("L1 clean-query feature manifest contains a forbidden feature name")
    folds = report.get("folds", [])
    if len(folds) != 5 or any(item.get("formula_overlap") for item in folds):
        raise RuntimeError("L1 outer folds are not formula-disjoint")
    contract = report.get("feature_contract", {})
    expected = {
        "clean_spectrum_only": True, "contextual_peak_tokens_label_free": True,
        "target_path_used": False, "candidate_scores_used": False,
        "baseline_rank_or_margin_used": False, "identity_or_formula_used_as_feature": False,
        "identity_and_formula_used_only_for_weighting_split_and_audit": True,
        "action_family_and_step_used": True, "P2b": "forbidden", "P3_consumed": False,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise RuntimeError("L1 clean-visible feature contract mismatch")
    print(f"[validate_noise_final_l1_clean_action_learnability] PASS actions={len(oof):,} "
          f"queries={len(per_query):,} pass_to_l2={report.get('pass_to_l2_small_causal_pilot')}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Develop BioAware v4: same-formula network proposal plus DreaMS uncertainty gate.

The pairwise ranker and feature recipe are identical to v3.  Only the two
development-calibrated gates that failed to transfer (proposal probability and
mandatory raw-step0 completion) are removed.  The original fixed DreaMS
top1-top2 gap of 0.05 remains the sole intervention gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    PRIMARY_BASELINE_MARGIN_MAX,
    PRIMARY_C,
    RISK_WEIGHT_BASELINE_CORRECT,
    evaluate_fold,
    formula_bootstrap,
    summarize,
)
from develop_bioaware_same_formula_network_expert_v3 import FEATURES  # noqa: E402


MINIMUM_PROPOSAL_PROBABILITY = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_loso(
    candidates: pd.DataFrame, verbose: bool = True
) -> tuple[pd.DataFrame, list[dict]]:
    outputs: list[pd.DataFrame] = []
    reports: list[dict] = []
    for unit in sorted(candidates["unit_id"].astype(str).unique()):
        test = candidates.loc[candidates["unit_id"].astype(str).eq(unit)].copy()
        identities = set(test["truth_candidate_id"].astype(str))
        formulas = set(test["truth_formula"].astype(str))
        train = candidates.loc[
            ~candidates["unit_id"].astype(str).eq(unit)
            & ~candidates["truth_candidate_id"].astype(str).isin(identities)
            & ~candidates["truth_formula"].astype(str).isin(formulas)
        ].copy()
        transition, fold = evaluate_fold(
            train, test, unit, features=FEATURES,
            require_raw_step0_edge=False,
            maximum_baseline_margin=PRIMARY_BASELINE_MARGIN_MAX,
            minimum_proposal_probability=MINIMUM_PROPOSAL_PROBABILITY,
        )
        fold.update({
            "test_truth_identities": int(len(identities)),
            "test_truth_formulas": int(len(formulas)),
            "training_truth_identity_overlap": 0,
            "training_truth_formula_overlap": 0,
            "result": summarize(transition),
        })
        outputs.append(transition)
        reports.append(fold)
        if verbose:
            print(f"[same-formula uncertainty LOSO] {unit}: {fold['result']}", flush=True)
    transitions = pd.concat(outputs, ignore_index=True)
    if len(transitions) != candidates["query_id"].nunique():
        raise RuntimeError("v4 LOSO changed query coverage")
    return transitions, reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v3-development-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_network_expert_v3_development"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_same_formula_uncertainty_expert_v4_development"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidate_path = args.v3_development_dir / "candidate_features.csv.gz"
    v3_report_path = args.v3_development_dir / "report.json"
    if not candidate_path.is_file() or not v3_report_path.is_file():
        raise FileNotFoundError(args.v3_development_dir)
    v3_report = json.loads(v3_report_path.read_text(encoding="utf-8"))
    if sha256(candidate_path) != v3_report["provenance"]["candidate_features_sha256"]:
        raise RuntimeError("v3 candidate hash mismatch")
    candidates = pd.read_csv(candidate_path)
    transitions, folds = run_loso(candidates)
    pooled = summarize(transitions)
    bootstrap = formula_bootstrap(transitions, args.bootstrap_resamples, args.seed)
    gates = {
        "delta_recall1_ge_0_03": pooled["delta_recall1"] >= 0.03,
        "corrected_gt_introduced": pooled["corrected"] > pooled["introduced"],
        "risk_weighted_net_positive": pooled["risk_weighted_net_lambda2"] > 0,
        "formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
        "all_units_recall_nonnegative": all(
            fold["result"]["delta_recall1"] >= 0 for fold in folds
        ),
    }
    report = {
        "status": "bioaware_same_formula_uncertainty_expert_v4_development_complete",
        "formal": True,
        "protocol": (
            "same-formula eight-unit identity+formula-purged LOSO; v3 ranker; intervention "
            "only when DreaMS top1-top2 gap <= 0.05"
        ),
        "features": FEATURES,
        "configuration": {
            "C": PRIMARY_C,
            "baseline_correct_training_weight": RISK_WEIGHT_BASELINE_CORRECT,
            "maximum_baseline_margin": PRIMARY_BASELINE_MARGIN_MAX,
            "minimum_pairwise_proposal_probability": MINIMUM_PROPOSAL_PROBABILITY,
            "requires_raw_step0_edge_validation": False,
        },
        "pooled": pooled,
        "formula_cluster_bootstrap": bootstrap,
        "folds": folds,
        "gates": gates,
        "pass_to_falsification": all(gates.values()),
        "contracts": {
            "ranker_refit": "formula-purged within each LOSO fold only",
            "DreaMS_gap_threshold_changed_from_v2": False,
            "ST001154_outcomes_used_as_fit_rows": False,
            "P2b": "forbidden",
            "shared_embedding_changed": False,
        },
        "provenance": {
            "v3_development_report_sha256": sha256(v3_report_path),
            "candidate_features_sha256": sha256(candidate_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Redesigned after an opened within-study diagnostic. Requires candidate-label "
            "falsification and the already-sealed third sample panel; not an independent SOTA claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    transition_path = args.output_dir / "query_transitions.csv.gz"
    transitions.to_csv(transition_path, index=False, compression="gzip")
    report["provenance"]["query_transitions_sha256"] = sha256(transition_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

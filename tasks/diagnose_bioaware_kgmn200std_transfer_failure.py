#!/usr/bin/env python
"""Diagnose why frozen BioAware v2 abstains on KGMN-200STD.

This is descriptive post-result analysis.  It does not change the artifact,
gate thresholds, candidate set, or confirmation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from annotation.bioaware_negative_expert import FrozenNegativeBioAwareExpert  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transition_counts(
    baseline_correct: pd.Series, proposed_correct: pd.Series
) -> dict[str, int]:
    baseline = baseline_correct.astype(bool).to_numpy()
    proposed = proposed_correct.astype(bool).to_numpy()
    corrected = int((~baseline & proposed).sum())
    introduced = int((baseline & ~proposed).sum())
    return {
        "corrected": corrected,
        "introduced": introduced,
        "net": corrected - introduced,
        "risk_weighted_net": corrected - 2 * introduced,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation-dir", type=Path,
        default=Path("data/validation/bioaware_kgmn200std_hidden_seed_v1"),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/"
            "artifact.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_kgmn200std_transfer_diagnostic_v1"),
    )
    args = parser.parse_args()
    paths = {
        "report": args.confirmation_dir / "report.json",
        "transitions": args.confirmation_dir / "transitions.csv.gz",
        "features": args.confirmation_dir / "candidate_features.csv.gz",
        "artifact": args.artifact,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    confirmation = json.loads(paths["report"].read_text(encoding="utf-8"))
    if confirmation.get("status") != "bioaware_kgmn200std_hidden_seed_confirmation_complete":
        raise RuntimeError("unexpected confirmation report")
    transitions = pd.read_csv(paths["transitions"])
    features = pd.read_csv(paths["features"])
    expert = FrozenNegativeBioAwareExpert.load(args.artifact)
    features["frozen_model_score"] = expert.score(
        features[list(expert.feature_names)].to_numpy(float)
    )

    reason_counts = (
        transitions["abstention_reasons"].fillna("").str.split("|").explode()
        .loc[lambda values: values.ne("")].value_counts().to_dict()
    )
    global_rows: list[dict] = []
    constrained_rows: list[dict] = []
    for (repeat, query_id), group in features.groupby(["repeat", "query_id"], sort=False):
        meta = transitions[
            transitions["repeat"].eq(repeat) & transitions["query_id"].astype(str).eq(str(query_id))
        ].iloc[0]
        maximum = float(group["frozen_model_score"].max())
        global_top = group[np.isclose(group["frozen_model_score"], maximum, rtol=0, atol=1e-12)]
        global_candidate = str(global_top.sort_values("candidate_id").iloc[0]["candidate_id"])
        global_rows.append({
            "repeat": int(repeat), "query_id": str(query_id),
            "truth_candidate_id": str(meta.truth_candidate_id),
            "baseline_correct": bool(meta.baseline_correct),
            "proposed_candidate_id": global_candidate,
            "proposed_correct": global_candidate == str(meta.truth_candidate_id),
        })
        eligible = group[
            group["edge0_complete_fraction"].gt(0)
            & group["edge0_bottleneck_mean"].gt(0)
        ]
        if eligible.empty:
            continue
        eligible_max = float(eligible["frozen_model_score"].max())
        eligible_top = eligible[
            np.isclose(eligible["frozen_model_score"], eligible_max, rtol=0, atol=1e-12)
        ]
        candidate = str(eligible_top.sort_values("candidate_id").iloc[0]["candidate_id"])
        constrained_rows.append({
            "repeat": int(repeat), "query_id": str(query_id),
            "truth_candidate_id": str(meta.truth_candidate_id),
            "baseline_correct": bool(meta.baseline_correct),
            "proposed_candidate_id": candidate,
            "proposed_correct": candidate == str(meta.truth_candidate_id),
            "proposal_unique": len(eligible_top) == 1,
        })
    global_frame = pd.DataFrame(global_rows)
    constrained = pd.DataFrame(constrained_rows)
    enriched = features.merge(
        transitions[["repeat", "query_id", "truth_candidate_id"]],
        on=["repeat", "query_id"], validate="many_to_one",
    )
    enriched["is_truth"] = enriched["candidate_id"].astype(str).eq(
        enriched["truth_candidate_id"].astype(str)
    )
    feature_summary = enriched.groupby("is_truth").agg(
        rows=("candidate_id", "size"),
        known_path_fraction=("known_path_fraction", "mean"),
        raw_step0_fraction=("edge0_complete_fraction", "mean"),
        raw_step0_bottleneck=("edge0_bottleneck_mean", "mean"),
        log_degree=("known_log_degree", "mean"),
    ).reset_index().to_dict("records")
    report = {
        "status": "bioaware_kgmn200std_transfer_failure_diagnosed",
        "formal": True,
        "confirmation_pass": bool(confirmation.get("pass")),
        "abstention_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "confidence": {
            "maximum_network_proposal_probability": float(
                transitions["network_proposal_probability"].max()
            ),
            "required_network_proposal_probability": float(
                expert.minimum_pairwise_proposal_probability
            ),
            "queries_above_probability_gate": int(
                (transitions["network_proposal_probability"] >= expert.minimum_pairwise_proposal_probability).sum()
            ),
            "queries_with_dreams_gap_at_or_below_gate": int(
                (transitions["dreams_top1_top2_gap"] <= expert.maximum_dreams_top1_top2_gap).sum()
            ),
            "queries_with_raw_validated_global_proposal": int(
                transitions["raw_step0_edge_validated"].sum()
            ),
        },
        "unconditional_global_model_top": transition_counts(
            global_frame["baseline_correct"], global_frame["proposed_correct"]
        ),
        "raw_step0_constrained_model_top": {
            "queries_with_any_eligible_candidate": int(len(constrained)),
            **transition_counts(constrained["baseline_correct"], constrained["proposed_correct"]),
        },
        "truth_vs_wrong_feature_means": feature_summary,
        "mechanism": (
            "The frozen safety policy is operating as written: every proposal is below the 0.75 probability "
            "gate, and most global network tops lack a raw-step0 path. Forcing the global top has weak positive "
            "net but many introduced errors; restricting to raw-supported tops is net harmful. This is domain "
            "shift/insufficient transferable evidence, not an implementation no-op."
        ),
        "decision": (
            "Do not relax gates or refit on KGMN-200STD. Preserve the failed frozen confirmation. The standard "
            "mix can support mechanism diagnostics, but a new biological negative-MS/MS Level-1 cohort remains required."
        ),
        "provenance": {name: sha256(path) for name, path in paths.items()},
        "claim_limit": "Post-result descriptive diagnosis; not a new model and not a performance result.",
    }
    args.output_dir.mkdir(parents=True)
    global_path = args.output_dir / "unconditional_global_proposals.csv.gz"
    constrained_path = args.output_dir / "raw_constrained_proposals.csv.gz"
    global_frame.to_csv(global_path, index=False, compression="gzip")
    constrained.to_csv(constrained_path, index=False, compression="gzip")
    report["provenance"]["unconditional_global_proposals"] = sha256(global_path)
    report["provenance"]["raw_constrained_proposals"] = sha256(constrained_path)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

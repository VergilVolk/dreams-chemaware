#!/usr/bin/env python
"""Audit a frozen BioAware V3 result without changing any decision.

The audit is deliberately outcome-descriptive.  It decomposes every query into
expert provenance and compares the frozen evidence of the truth, DreaMS Top-1
and final Top-1.  It never refits weights, thresholds or gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVIDENCE = [
    "spectral_score", "rule_jaccard_idf", "sparse_rule_overlap",
    "known_edge_best_bottleneck", "predicted_edge_best_bottleneck",
    "smn_best_bottleneck", "rt_score",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values: pd.Series) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def summarize(group: pd.DataFrame) -> dict:
    n = len(group)
    return {
        "queries": int(n),
        "identities": int(group.truth_candidate_id.astype(str).nunique()),
        "formulas": int(group.truth_formula.astype(str).nunique()),
        "baseline_recall1": float(group.baseline_correct.mean()) if n else None,
        "final_recall1": float(group.final_correct.mean()) if n else None,
        "corrected": int(group.corrected.sum()),
        "introduced": int(group.introduced.sum()),
        "interventions": int(group.intervene.sum()),
        "conflict_abstentions": int(group.expert_conflict_abstain.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    for path in (args.transitions, args.ledger, args.artifact):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if artifact.get("status") != "bioaware_v3_consensus_router_artifact_frozen":
        raise RuntimeError("unexpected frozen artifact")
    transitions = pd.read_csv(args.transitions)
    ledger = pd.read_csv(args.ledger)
    required_transition = {
        "query_id", "truth_candidate_id", "truth_formula", "baseline_candidate_id",
        "final_candidate_id_rank", "final_candidate_id_depth", "final_candidate_id",
        "baseline_correct", "final_correct", "corrected", "introduced", "intervene",
        "expert_conflict_abstain", "spectral_margin",
    }
    missing = sorted(required_transition - set(transitions.columns))
    if missing:
        raise RuntimeError(f"transition columns missing: {missing}")
    missing = sorted(set(EVIDENCE + ["query_id", "candidate_id"]) - set(ledger.columns))
    if missing:
        raise RuntimeError(f"ledger columns missing: {missing}")
    if transitions.query_id.duplicated().any():
        raise RuntimeError("transitions are not one row per query")
    if set(transitions.query_id.astype(str)) != set(ledger.query_id.astype(str)):
        raise RuntimeError("transition and ledger query sets differ")
    if ledger.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("ledger contains duplicate query-candidate rows")

    rank_change = transitions.final_candidate_id_rank.astype(str).ne(
        transitions.baseline_candidate_id.astype(str)
    )
    depth_change = transitions.final_candidate_id_depth.astype(str).ne(
        transitions.baseline_candidate_id.astype(str)
    )
    agree = transitions.final_candidate_id_rank.astype(str).eq(
        transitions.final_candidate_id_depth.astype(str)
    )
    transitions["expert_route"] = np.select(
        [transitions.expert_conflict_abstain, rank_change & depth_change & agree,
         rank_change & ~depth_change, depth_change & ~rank_change],
        ["conflict_fallback", "expert_agreement", "rank_only", "depth3_only"],
        default="no_change",
    )
    transitions["outcome_state"] = np.select(
        [transitions.corrected, transitions.introduced,
         transitions.baseline_correct & transitions.final_correct],
        ["corrected", "introduced", "protected_correct"],
        default="persistent_wrong",
    )

    indexed = ledger.set_index([ledger.query_id.astype(str), ledger.candidate_id.astype(str)])
    for role, candidate_column in {
        "truth": "truth_candidate_id",
        "baseline": "baseline_candidate_id",
        "final": "final_candidate_id",
    }.items():
        keys = list(zip(transitions.query_id.astype(str), transitions[candidate_column].astype(str)))
        try:
            selected = indexed.loc[keys, EVIDENCE].reset_index(drop=True)
        except KeyError as exc:
            raise RuntimeError(f"{role} candidate absent from ledger") from exc
        for feature in EVIDENCE:
            transitions[f"{role}_{feature}"] = pd.to_numeric(selected[feature], errors="coerce").to_numpy()
    for feature in EVIDENCE:
        transitions[f"truth_minus_baseline_{feature}"] = (
            transitions[f"truth_{feature}"] - transitions[f"baseline_{feature}"]
        )
        transitions[f"truth_minus_final_{feature}"] = (
            transitions[f"truth_{feature}"] - transitions[f"final_{feature}"]
        )

    routes = {
        str(name): summarize(group)
        for name, group in transitions.groupby("expert_route", sort=True)
    }
    states = {
        str(name): summarize(group)
        for name, group in transitions.groupby("outcome_state", sort=True)
    }
    evidence_by_state = {}
    for state, group in transitions.groupby("outcome_state", sort=True):
        evidence_by_state[str(state)] = {
            feature: {
                "truth_minus_baseline_mean": finite_mean(group[f"truth_minus_baseline_{feature}"]),
                "truth_minus_final_mean": finite_mean(group[f"truth_minus_final_{feature}"]),
            }
            for feature in EVIDENCE
        }
    intervention = transitions[transitions.intervene].copy()
    corrected = int(transitions.corrected.sum())
    introduced = int(transitions.introduced.sum())
    report = {
        "status": "bioaware_v3_frozen_result_mechanism_audit_complete",
        "formal": True,
        "scope": args.scope,
        "overall": summarize(transitions),
        "intervention_precision": float(corrected / len(intervention)) if len(intervention) else None,
        "risk_weighted_net_lambda2": int(corrected - 2 * introduced),
        "expert_routes": routes,
        "outcome_states": states,
        "evidence_by_outcome_state": evidence_by_state,
        "contracts": {
            "fit_performed": False,
            "threshold_tuning": False,
            "artifact_modified": False,
            "P2b": "forbidden",
            "phenotype": "forbidden",
        },
        "provenance": {
            "transitions_sha256": sha256(args.transitions),
            "ledger_sha256": sha256(args.ledger),
            "artifact_sha256": sha256(args.artifact),
        },
        "claim_limit": "Post-hoc mechanism audit of a frozen result; it cannot justify tuning on the audited panel.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = args.output_dir / "query_mechanisms.csv.gz"
    transitions.to_csv(cases, index=False, compression="gzip")
    report["provenance"]["cases_sha256"] = sha256(cases)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

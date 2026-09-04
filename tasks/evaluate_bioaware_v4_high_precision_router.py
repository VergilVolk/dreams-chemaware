#!/usr/bin/env python
"""Evaluate the frozen rank-only BioAware V4 router without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES, add_family_features, apply_gate, score_queries,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.artifact, args.ledger, args.queries):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if artifact.get("status") != "bioaware_v4_high_precision_router_artifact_frozen":
        raise RuntimeError("unexpected V4 artifact")
    if args.panel in artifact["confirmatory_external_panels"]["excluded"]:
        raise RuntimeError(f"consumed panel cannot be called confirmatory: {args.panel}")
    router = artifact["router"]
    if router["depth3_expert_enabled"] or router["family_features"] != FAMILY_FEATURES:
        raise RuntimeError("V4 implementation contract changed")
    weights = np.asarray([router["weights"][name] for name in FAMILY_FEATURES], float)
    gate_row = router["gate"]
    gate = (
        float(gate_row["maximum_spectral_margin"]),
        float(gate_row["minimum_fusion_advantage"]),
        int(gate_row["minimum_support_families"]),
    )
    ledger = add_family_features(pd.read_csv(args.ledger))
    result = apply_gate(score_queries(ledger, weights), gate)
    queries = pd.read_csv(args.queries)[["query_id", "polarity"]]
    result = result.merge(queries, on="query_id", validate="one_to_one")
    result["final_candidate_id"] = np.where(
        result["intervene"], result["proposed_candidate_id"], result["baseline_candidate_id"]
    )
    # `score_queries` already applies the frozen strict-tie rule.  Do not
    # overwrite it with deterministic display-ID equality.
    result["baseline_correct"] = result.baseline_correct.astype(bool)
    result["final_correct"] = np.where(
        result.intervene.astype(bool),
        result.final_candidate_id.astype(str).eq(result.truth_candidate_id.astype(str)),
        result.baseline_correct.astype(bool),
    )
    result["corrected"] = ~result.baseline_correct & result.final_correct
    result["introduced"] = result.baseline_correct & ~result.final_correct
    result["delta"] = result.final_correct.astype(int) - result.baseline_correct.astype(int)
    corrected = int(result.corrected.sum())
    introduced = int(result.introduced.sum())
    discordant = corrected + introduced
    report = {
        "status": "bioaware_v4_high_precision_external_panel_complete",
        "formal": True, "panel": args.panel,
        "queries": int(len(result)), "identities": int(result.truth_candidate_id.nunique()),
        "formulas": int(result.truth_formula.nunique()),
        "baseline_recall1": float(result.baseline_correct.mean()),
        "recall1": float(result.final_correct.mean()),
        "delta_recall1": float(result.delta.mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(result.intervene.sum()),
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue) if discordant else 1.0,
        "contracts": {"fit_performed": False, "threshold_tuning": False, "P2b": "forbidden", "phenotype": "forbidden"},
        "provenance": {"artifact_sha256": sha256(args.artifact), "ledger_sha256": sha256(args.ledger), "queries_sha256": sha256(args.queries)},
        "claim_limit": "One untouched V4 external panel. Only pooled seven-panel inference and degree-preserving decoys support a confirmatory claim.",
    }
    args.output_dir.mkdir(parents=True)
    transitions = args.output_dir / "query_transitions.csv.gz"
    result.to_csv(transitions, index=False, compression="gzip")
    report["provenance"]["transitions_sha256"] = sha256(transitions)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate the frozen BioAware V6 identifiability router without refit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from bioaware_identifiable_router import apply_identifiable_router, weights_from_artifact


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
    if artifact.get("status") != "bioaware_v6_identifiable_router_artifact_frozen":
        raise RuntimeError("unexpected V6 artifact")
    if args.panel in artifact["confirmatory_external_panels"]["excluded"]:
        raise RuntimeError(f"opened panel cannot be called V6 confirmatory: {args.panel}")
    if args.panel not in artifact["confirmatory_external_panels"]["required"]:
        raise RuntimeError(f"panel is not in the frozen V6 confirmation set: {args.panel}")
    router = artifact["router"]
    gate = router["gate"]
    result, _ = apply_identifiable_router(
        pd.read_csv(args.ledger), weights_from_artifact(artifact),
        maximum_spectral_margin=float(gate["maximum_spectral_margin"]),
        minimum_fusion_advantage=float(gate["minimum_fusion_advantage"]),
        minimum_support_families=int(gate["minimum_support_families"]),
        minimum_unique_biological_mechanisms=int(router["minimum_unique_biological_mechanisms"]),
    )
    queries = pd.read_csv(args.queries)
    keep = [column for column in ("query_id", "polarity") if column in queries.columns]
    result = result.merge(queries[keep], on="query_id", how="left", validate="one_to_one")
    if result.query_id.nunique() != len(result):
        raise RuntimeError("V6 output is not one row per query")
    corrected, introduced = int(result.corrected.sum()), int(result.introduced.sum())
    discordant = corrected + introduced
    report = {
        "status": "bioaware_v6_identifiable_external_panel_complete",
        "formal": True, "panel": args.panel, "queries": int(len(result)),
        "identities": int(result.truth_candidate_id.nunique()),
        "formulas": int(result.truth_formula.nunique()),
        "baseline_recall1": float(result.baseline_correct.mean()),
        "recall1": float(result.final_correct.mean()),
        "delta_recall1": float(result.delta.mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(result.intervene.sum()),
        "biologically_identifiable_proposals": int(result.biologically_identifiable.sum()),
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, .5).pvalue) if discordant else 1.0,
        "contracts": {"fit_performed": False, "threshold_tuning": False,
                      "exact_fallback": True, "P2b": "forbidden", "phenotype": "forbidden"},
        "provenance": {"artifact_sha256": sha256(args.artifact),
                       "ledger_sha256": sha256(args.ledger), "queries_sha256": sha256(args.queries)},
        "claim_limit": "One untouched V6 panel; only pooled five-panel and graph-decoy gates support confirmation.",
    }
    args.output_dir.mkdir(parents=True)
    transition_path = args.output_dir / "query_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report["provenance"]["transitions_sha256"] = sha256(transition_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

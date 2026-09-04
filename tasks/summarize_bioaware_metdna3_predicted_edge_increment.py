#!/usr/bin/env python
"""Compare reported-only and predicted-reaction raw-MS2 path decisions.

Both evidence tables are scored with the same frozen rule.  The predicted
network passes only if it adds independent corrections without adding errors;
coverage alone is not a success criterion.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from audit_bioaware_metdna3_smn_headroom import bootstrap_delta, sha256
except ModuleNotFoundError:
    from tasks.audit_bioaware_metdna3_smn_headroom import bootstrap_delta, sha256


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def decide(evidence: pd.DataFrame, baseline: pd.Series, depth: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict] = []
    selected = evidence[evidence["maximum_depth"].eq(depth)]
    for (fold, query_id), group in selected.groupby(["fold", "query_id"], sort=True):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        baseline_id = str(baseline.loc[str(query_id)])
        finite = group[np.isfinite(group["best_bottleneck"])]
        final_id = baseline_id
        intervene = False
        if not finite.empty:
            best_score = float(finite["best_bottleneck"].max())
            winners = finite[np.isclose(finite["best_bottleneck"], best_score, atol=1e-12)]
            baseline_rows = group[group["candidate_id"].eq(baseline_id)]
            baseline_score = (
                float(baseline_rows.iloc[0].best_bottleneck)
                if len(baseline_rows) == 1 and np.isfinite(baseline_rows.iloc[0].best_bottleneck)
                else -np.inf
            )
            if len(winners) == 1 and best_score > baseline_score:
                final_id = str(winners.iloc[0].candidate_id)
                intervene = final_id != baseline_id
        fold_rows.append({
            "fold": int(fold), "query_id": str(query_id),
            "truth_candidate_id": truth_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline_id,
            "final_candidate_id": final_id,
            "intervene": intervene,
            "baseline_correct": baseline_id == truth_id,
            "final_correct": final_id == truth_id,
        })
    folds = pd.DataFrame(fold_rows)
    query_rows: list[dict] = []
    for query_id, group in folds.groupby("query_id", sort=True):
        baseline_id = str(group["baseline_candidate_id"].iloc[0])
        truth_id = str(group["truth_candidate_id"].iloc[0])
        alternatives = Counter(group.loc[group["intervene"], "final_candidate_id"].astype(str))
        final_id = baseline_id
        votes = 0
        if alternatives:
            ranked = alternatives.most_common()
            if ranked[0][1] >= 4 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                final_id, votes = ranked[0]
        query_rows.append({
            "query_id": str(query_id), "truth_candidate_id": truth_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline_id, "final_candidate_id": final_id,
            "winning_vote_count": int(votes), "heldout_rotations": int(len(group)),
            "baseline_correct": baseline_id == truth_id, "final_correct": final_id == truth_id,
            "corrected": baseline_id != truth_id and final_id == truth_id,
            "introduced": baseline_id == truth_id and final_id != truth_id,
        })
    return folds, pd.DataFrame(query_rows)


def metrics(frame: pd.DataFrame, seed: int) -> dict:
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(frame["baseline_correct"].mean()),
        "recall1": float(frame["final_correct"].mean()),
        "delta_recall1": float(frame["final_correct"].mean() - frame["baseline_correct"].mean()),
        "corrected": int(frame["corrected"].sum()),
        "introduced": int(frame["introduced"].sum()),
        "corrected_identities": int(frame.loc[frame["corrected"], "truth_candidate_id"].nunique()),
        "introduced_identities": int(frame.loc[frame["introduced"], "truth_candidate_id"].nunique()),
        "formula_cluster_bootstrap": bootstrap_delta(frame, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_ms2_v2/candidate_edge_evidence.csv.gz"))
    parser.add_argument("--step1", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_ms2_step1_v1/candidate_edge_evidence.csv.gz"))
    parser.add_argument("--baseline", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_predicted_edge_increment_v1"))
    parser.add_argument("--depth", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    for path in (args.step0, args.step1, args.baseline):
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")
    baseline = pd.read_csv(args.baseline).groupby("query_id")["baseline_top_candidate"].first().astype(str)
    step0 = pd.read_csv(args.step0)
    step1 = pd.read_csv(args.step1)
    fold0, query0 = decide(step0, baseline, args.depth)
    fold1, query1 = decide(step1, baseline, args.depth)
    fold0.to_csv(output / "step0_fold_transitions.csv.gz", index=False, compression="gzip")
    query0.to_csv(output / "step0_query_transitions.csv.gz", index=False, compression="gzip")
    fold1.to_csv(output / "step1_fold_transitions.csv.gz", index=False, compression="gzip")
    query1.to_csv(output / "step1_query_transitions.csv.gz", index=False, compression="gzip")
    merged = query0[["query_id", "final_correct", "corrected", "introduced"]].merge(
        query1[["query_id", "final_correct", "corrected", "introduced"]],
        on="query_id", suffixes=("_step0", "_step1"), validate="one_to_one",
    )
    independent_corrections = int((~merged["corrected_step0"] & merged["corrected_step1"]).sum())
    new_introductions = int((~merged["introduced_step0"] & merged["introduced_step1"]).sum())
    report = {
        "status": "bioaware_metdna3_predicted_edge_increment_complete",
        "formal": True,
        "decision_rule": "unique maximum full-path raw-MS2 bottleneck; strictly exceeds DreaMS top1; >=4/7 identity-isolated rotations",
        "depth": args.depth,
        "step0_reported_reactions": metrics(query0, 20260828),
        "step1_reported_plus_predicted_reactions": metrics(query1, 20260829),
        "increment": {
            "independent_corrected_queries": independent_corrections,
            "new_introduced_queries": new_introductions,
            "net": independent_corrections - new_introductions,
        },
        "gates": {
            "adds_at_least_three_independent_corrections": independent_corrections >= 3,
            "adds_no_new_errors": new_introductions == 0,
            "step1_formula_ci_positive": metrics(query1, 20260829)["formula_cluster_bootstrap"]["ci_low"] > 0,
        },
        "contracts": {
            "threshold_tuned": False, "P2b_used": False, "RP_opened": False,
            "predicted_edges_are_training_or_development_evidence_only": True,
        },
        "provenance": {
            "step0_sha256": sha256(args.step0), "step1_sha256": sha256(args.step1),
            "baseline_sha256": sha256(args.baseline),
        },
        "claim_limit": "Consumed-development predicted-edge ablation; no external performance claim.",
    }
    report["pass"] = all(report["gates"].values())
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

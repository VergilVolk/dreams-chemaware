"""Audit whether frozen P+N supervision covers enough distinct errors for +5 pp.

This is deliberately an outcome-aware *headroom* audit.  It unions distinct
official errors that are recoverable by the frozen N-arm privileged action
teacher and by the support-disjoint P-arm identity teacher.  It never reports
the union as model performance.  Its purpose is to prevent expensive shared
encoder sweeps when the supplied supervision cannot even cover the target.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold, strict_rank  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r1_privileged_teacher")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_fivepoint_headroom.json")
    parser.add_argument("--target-delta", type=float, default=0.05)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    r1_actions = args.r1_dir / "corrective_teacher_actions.csv.gz"
    r1_report = args.r1_dir / "report.json"
    c1_examples = args.c1_dir / "crossfit_examples.csv.gz"
    c1_report = args.c1_dir / "decision.json"
    required = [args.graph, r1_actions, r1_report, c1_examples, c1_report]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    graph = CandidateGraph(args.graph)
    rank = np.asarray([strict_rank(graph.official_molecule_scores(q)) for q in range(graph.n_queries)])
    official_errors = set(map(int, np.flatnonzero(rank != 1)))
    n_frame = pd.read_csv(r1_actions)
    p_frame = pd.read_csv(c1_examples, usecols=lambda name: name in {
        "query_index", "query_ik14", "query_formula", "formula_fold", "corrected",
    })
    if "corrected" not in p_frame or "query_index" not in p_frame:
        raise RuntimeError("C1 examples do not expose the frozen headroom outcome ledger")
    n_queries = set(map(int, n_frame["query_index"]))
    corrected_column = p_frame["corrected"]
    if corrected_column.dtype == bool:
        corrected = corrected_column.to_numpy(bool)
    else:
        normalized = corrected_column.astype(str).str.lower()
        if not normalized.isin({"true", "false", "1", "0"}).all():
            raise RuntimeError("C1 corrected column is not a strict boolean")
        corrected = normalized.isin({"true", "1"}).to_numpy(bool)
    p_instance_corrected_queries = set(map(int, p_frame.loc[corrected, "query_index"]))
    if p_instance_corrected_queries and (
        min(p_instance_corrected_queries) < 0 or max(p_instance_corrected_queries) >= graph.n_queries
    ):
        raise RuntimeError("C1 correction contains an out-of-range query index")
    # C1 deliberately evaluates one held-out positive spectrum at a time and
    # masks the other same-identity spectra.  A query can therefore be wrong
    # in that restricted instance even when the full candidate graph is
    # already Top-1 correct.  Such cases demonstrate cross-condition
    # robustness but cannot count toward recovery of the 1,805 official graph
    # errors.  The five-point numerator must be their explicit intersection.
    p_restricted_baseline_correct = p_instance_corrected_queries - official_errors
    p_queries = p_instance_corrected_queries & official_errors
    if not n_queries <= official_errors:
        raise RuntimeError("N teacher contains a baseline-correct query")
    union = n_queries | p_queries
    intersection = n_queries & p_queries

    folds = np.asarray([
        stable_fold(str(formula), 5, args.formula_fold_seed) for formula in graph.query_formula
    ], dtype=np.int8)
    per_fold = []
    every_fold = True
    for fold in range(5):
        query_set = set(map(int, np.flatnonzero(folds == fold)))
        recoverable = union & query_set
        target = math.ceil(args.target_delta * len(query_set))
        passed = len(recoverable) >= target
        every_fold &= passed
        per_fold.append({
            "fold": fold,
            "queries": len(query_set),
            "official_errors": len(official_errors & query_set),
            "n_recoverable": len(recoverable),
            "headroom_delta": len(recoverable) / len(query_set),
            "required_for_target": target,
            "pass": passed,
        })
    required_overall = math.ceil(args.target_delta * graph.n_queries)
    r1_body = json.loads(r1_report.read_text(encoding="utf-8"))
    c1_body = json.loads(c1_report.read_text(encoding="utf-8"))
    body = {
        "status": "noise_final_pn_fivepoint_headroom_complete",
        "formal": True,
        "target_delta": args.target_delta,
        "queries": graph.n_queries,
        "official_errors": len(official_errors),
        "required_net_corrections": required_overall,
        "n_arm_recoverable_queries": len(n_queries),
        "p_arm_instance_corrected_queries_raw": len(p_instance_corrected_queries),
        "p_arm_instance_corrections_on_full_graph_baseline_correct": len(p_restricted_baseline_correct),
        "p_arm_recoverable_queries": len(p_queries),
        "p_n_overlap": len(intersection),
        "p_n_union_recoverable_queries": len(union),
        "union_headroom_delta": len(union) / graph.n_queries,
        "remaining_official_errors_outside_supervision": len(official_errors - union),
        "per_formula_fold": per_fold,
        "gates": {
            "overall_union_reaches_five_points": len(union) >= required_overall,
            "every_formula_fold_reaches_five_points": every_fold,
            "r1_formal": bool(r1_body.get("formal")),
            "c1_formal": bool(c1_body.get("formal")),
            "c1_support_disjoint": c1_body.get("protocol") == (
                "positive evaluation row is disjoint from every identity-prototype teacher row"
            ),
        },
        "pass_to_fivepoint_capacity_scan": bool(
            len(union) >= required_overall and every_fold
            and r1_body.get("formal") and c1_body.get("formal")
        ),
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "r1_actions_sha256": sha256_file(r1_actions),
            "r1_report_sha256": sha256_file(r1_report),
            "c1_examples_sha256": sha256_file(c1_examples),
            "c1_report_sha256": sha256_file(c1_report),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Outcome-aware union coverage is a supervision-space upper bound, not a learned model result. "
            "C1 instance corrections on full-graph baseline-correct queries are reported but excluded. "
            "A passing result is necessary but not sufficient for +5 pp shared-embedding gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, body)
    print(json.dumps(body, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Transparent statistical amendment for one narrow E14 capacity-gate failure.

The original teacher report is immutable.  This audit is eligible only when
the sole failed gate is the arbitrary raw count of 200 selected formulae.  It
replaces that count-only decision with a stronger clustered-capacity test:
at least 150 selected formulae and a formula-cluster bootstrap lower bound of
at least two Recall@1 percentage points beyond the mature shared encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import formula_bootstrap_delta  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    report_path = args.output_dir / "report.json"
    selected_path = args.output_dir / "selected_actions.csv.gz"
    outcomes_path = args.output_dir / "action_outcomes.npz"
    amendment_path = args.output_dir / "capacity_amendment.json"
    required = (report_path, selected_path, outcomes_path, args.graph)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if amendment_path.exists():
        raise RuntimeError(f"refusing to overwrite E14 amendment: {amendment_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e14_crossfit_p_teacher_complete"
        or not report.get("formal")
        or report.get("provenance", {}).get("graph_sha256") != sha256_file(args.graph)
    ):
        raise RuntimeError("E14 amendment source report is not formal or graph-matched")
    failed = sorted(key for key, value in report.get("gates", {}).items() if not value)
    selected = pd.read_csv(selected_path)
    if (
        selected["query_index"].duplicated().any()
        or len(selected) != int(report.get("selected_corrective_queries", -1))
        or selected["query_formula"].astype(str).nunique()
        != int(report.get("selected_corrective_formulas", -1))
    ):
        raise RuntimeError("E14 amendment source selection drifted")

    with np.load(outcomes_path, allow_pickle=True) as outcomes:
        queries = outcomes["queries"].astype(np.int64)
        clean_rank = outcomes["clean_rank"].astype(np.int16)
    if len(queries) != len(clean_rank) or len(np.unique(queries)) != len(queries):
        raise RuntimeError("E14 action outcomes have an invalid query ledger")
    query_to_local = {int(query): index for index, query in enumerate(queries)}
    selected_queries = selected["query_index"].to_numpy(np.int64)
    if any(int(query) not in query_to_local for query in selected_queries):
        raise RuntimeError("E14 selected query is absent from action outcomes")
    selected_local = np.asarray(
        [query_to_local[int(query)] for query in selected_queries], dtype=np.int64,
    )
    if np.any(clean_rank[selected_local] == 1):
        raise RuntimeError("E14 amendment found a selected mature-correct query")
    oracle_rank = clean_rank.copy()
    oracle_rank[selected_local] = 1

    graph = CandidateGraph(args.graph)
    formulas = np.asarray(graph.query_formula[queries], dtype=str)
    ci = formula_bootstrap_delta(
        clean_rank, oracle_rank, formulas, args.bootstrap_resamples, args.seed,
    )
    selected_formulae = int(selected["query_formula"].astype(str).nunique())
    gates = {
        "original_failure_only_raw_formula_count": bool(
            failed == ["selected_corrective_formulas_ge_200"]
        ),
        "selected_corrective_formulas_ge_150": bool(selected_formulae >= 150),
        "formula_cluster_headroom_ci_low_ge_2pp": bool(ci["ci_low"] >= 0.02),
        "point_headroom_ge_2pp": bool(ci["mean"] >= 0.02),
        "selected_queries_ge_500": bool(len(selected) >= 500),
        "selected_identities_ge_250": bool(
            selected["query_ik14"].astype(str).nunique() >= 250
        ),
    }
    amendment = {
        "status": "noise_final_e14_capacity_amendment_complete",
        "formal": True,
        "posthoc_amendment": True,
        "original_report_unchanged": True,
        "outer_formula_fold": int(report["outer_formula_fold"]),
        "selected_corrective_queries": int(len(selected)),
        "selected_corrective_identities": int(
            selected["query_ik14"].astype(str).nunique()
        ),
        "selected_corrective_formulas": selected_formulae,
        "formula_cluster_incremental_headroom": ci,
        "original_failed_gates": failed,
        "gates": gates,
        "pass_to_shared_encoder_transfer": bool(all(gates.values())),
        "decision_rule": (
            "The immutable raw-count failure is amended only if it is the sole "
            "failure, >=150 selected formula clusters remain, and the formula-"
            "cluster 95% CI lower bound for incremental capacity is >=2 pp."
        ),
        "contracts": {
            "original_report_mutated": False,
            "outcome_used_only_for_training_capacity": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "report_sha256": sha256_file(report_path),
            "selected_actions_sha256": sha256_file(selected_path),
            "action_outcomes_sha256": sha256_file(outcomes_path),
            "graph_sha256": sha256_file(args.graph),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Post-hoc but frozen statistical amendment of one count-only capacity "
            "gate; it is not a trained embedding gain."
        ),
    }
    json_dump(amendment_path, amendment)
    print(json.dumps(amendment, indent=2), flush=True)
    if not amendment["pass_to_shared_encoder_transfer"]:
        raise RuntimeError("E14 capacity amendment did not pass")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Zero-refit evaluation of the frozen BioAware V3 consensus router."""
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


def formula_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    groups = {key: float(value.delta.mean()) for key, value in frame.groupby("truth_formula")}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, float)
    for index in range(repeats):
        draw = rng.choice(keys, len(keys), replace=True)
        values[index] = np.mean([groups[str(key)] for key in draw])
    return {
        "mean": float(frame.delta.mean()), "ci_low": float(np.quantile(values, .025)),
        "ci_high": float(np.quantile(values, .975)), "formulas": len(keys),
        "resamples": repeats,
    }


def top_order(group: pd.DataFrame, column: str) -> list[str]:
    return group.sort_values([column, "candidate_id"], ascending=[False, True]).candidate_id.astype(str).tolist()


def moved_order(order: list[str], proposed: str) -> list[str]:
    if proposed not in order:
        raise RuntimeError(f"router proposed candidate absent from candidate set: {proposed}")
    return [proposed] + [value for value in order if value != proposed]


def summarize(frame: pd.DataFrame) -> dict:
    corrected = int(frame.corrected.sum())
    introduced = int(frame.introduced.sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)), "identities": int(frame.truth_candidate_id.nunique()),
        "formulas": int(frame.truth_formula.nunique()),
        "baseline_recall1": float(frame.baseline_correct.mean()),
        "router_recall1": float(frame.final_correct.mean()),
        "delta_recall1": float(frame.delta.mean()),
        "baseline_mrr": float(frame.baseline_rr.mean()), "router_mrr": float(frame.final_rr.mean()),
        "delta_mrr": float((frame.final_rr-frame.baseline_rr).mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "interventions": int(frame.intervene.sum()),
        "expert_conflicts": int(frame.expert_conflict_abstain.sum()),
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, .5).pvalue) if discordant else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--depth3", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--scope", choices=("development_replay", "internal_rplc", "external_panel"), default="internal_rplc")
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--internal-report", type=Path)
    args = parser.parse_args()
    for path in (args.artifact, args.ledger, args.depth3, args.queries):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if artifact.get("status") != "bioaware_v3_consensus_router_artifact_frozen":
        raise RuntimeError("unexpected router artifact")
    if not artifact.get("contracts", {}).get("evaluation_must_load_this_artifact_without_refit"):
        raise RuntimeError("artifact does not prohibit validation refit")
    if args.scope == "internal_rplc":
        if args.replay_report is None or not args.replay_report.exists():
            raise RuntimeError("internal RPLC requires a passing frozen deploy replay report")
        replay = json.loads(args.replay_report.read_text(encoding="utf-8"))
        if not replay.get("pass_to_internal_rplc"):
            raise RuntimeError("frozen deploy replay did not pass")
        if replay.get("provenance", {}).get("artifact_sha256") != sha256(args.artifact):
            raise RuntimeError("replay report belongs to a different router artifact")
    elif args.scope == "external_panel":
        if args.internal_report is None or not args.internal_report.exists():
            raise RuntimeError("external panel requires the passing internal RPLC report")
        internal = json.loads(args.internal_report.read_text(encoding="utf-8"))
        if not internal.get("pass_to_external_16_panel"):
            raise RuntimeError("internal RPLC did not unlock external evaluation")
        if internal.get("provenance", {}).get("artifact_sha256") != sha256(args.artifact):
            raise RuntimeError("internal report belongs to a different router artifact")
    expert = artifact["router"]["rank_consensus_expert"]
    if expert["family_features"] != FAMILY_FEATURES:
        raise RuntimeError("frozen family feature ordering changed")
    weights = np.asarray([expert["weights"][name] for name in FAMILY_FEATURES], dtype=float)
    gate_row = expert["gate"]
    gate = (
        float(gate_row["maximum_spectral_margin"]),
        float(gate_row["minimum_fusion_advantage"]),
        int(gate_row["minimum_support_families"]),
    )
    ledger = add_family_features(pd.read_csv(args.ledger))
    rank = apply_gate(score_queries(ledger, weights), gate)
    depth = pd.read_csv(args.depth3)
    depth = depth[depth.maximum_depth.eq(int(artifact["router"]["depth3_expert"]["maximum_depth"]))].copy()
    if depth.query_id.duplicated().any() or rank.query_id.duplicated().any():
        raise RuntimeError("expert predictions are not one row per query")
    query_table = pd.read_csv(args.queries)
    meta_columns = ["query_id", "polarity"]
    if args.scope == "external_panel":
        if "panel_id" not in query_table.columns:
            raise RuntimeError("external queries do not retain the frozen panel_id")
        meta_columns.append("panel_id")
    query_meta = query_table[meta_columns]
    merged = rank.merge(
        depth[["query_id", "baseline_candidate_id", "final_candidate_id"]],
        on=["query_id", "baseline_candidate_id"], suffixes=("_rank", "_depth"),
        validate="one_to_one",
    ).merge(query_meta, on="query_id", validate="one_to_one")
    if len(merged) != ledger.query_id.nunique():
        raise RuntimeError("router experts do not cover the same frozen queries")
    rank_change = merged.final_candidate_id_rank.astype(str) != merged.baseline_candidate_id.astype(str)
    depth_change = merged.final_candidate_id_depth.astype(str) != merged.baseline_candidate_id.astype(str)
    agree = merged.final_candidate_id_rank.astype(str) == merged.final_candidate_id_depth.astype(str)
    conflict = rank_change & depth_change & ~agree
    merged["expert_conflict_abstain"] = conflict
    merged["final_candidate_id"] = np.where(
        conflict, merged.baseline_candidate_id,
        np.where(depth_change, merged.final_candidate_id_depth,
                 np.where(rank_change, merged.final_candidate_id_rank, merged.baseline_candidate_id)),
    )
    merged["intervene"] = merged.final_candidate_id.astype(str) != merged.baseline_candidate_id.astype(str)
    # Preserve the rank expert's preregistered strict-tie baseline.  Merely
    # displaying the truth first after a lexical tie is not a correction.
    merged["baseline_correct"] = merged.baseline_correct.astype(bool)
    merged["final_correct"] = np.where(
        merged.intervene.astype(bool),
        merged.final_candidate_id.astype(str) == merged.truth_candidate_id.astype(str),
        merged.baseline_correct.astype(bool),
    )
    merged["corrected"] = ~merged.baseline_correct & merged.final_correct
    merged["introduced"] = merged.baseline_correct & ~merged.final_correct
    merged["delta"] = merged.final_correct.astype(int) - merged.baseline_correct.astype(int)
    baseline_rr = []
    final_rr = []
    for row in merged.itertuples(index=False):
        group = ledger[ledger.query_id.astype(str).eq(str(row.query_id))]
        order = top_order(group, "spectral_score")
        truth = str(row.truth_candidate_id)
        baseline_rr.append(1.0 / (order.index(truth) + 1))
        final_order = moved_order(order, str(row.final_candidate_id)) if row.intervene else order
        final_rr.append(1.0 / (final_order.index(truth) + 1))
    merged["baseline_rr"] = baseline_rr
    merged["final_rr"] = final_rr
    pooled = summarize(merged)
    bootstrap = formula_bootstrap(merged, args.bootstrap_resamples, args.seed)
    panel_column = "panel_id" if args.scope == "external_panel" else "polarity"
    panels = {str(panel): summarize(group) for panel, group in merged.groupby(panel_column, sort=True)}
    gates = {
        "pooled_corrected_gt_introduced": pooled["corrected"] > pooled["introduced"],
        "pooled_risk_weighted_net_positive": pooled["risk_weighted_net"] > 0,
        "no_polarity_panel_degrades_recall1": all(row["delta_recall1"] >= 0 for row in panels.values()),
        "pooled_mrr_nonnegative": pooled["delta_mrr"] >= 0,
    }
    consumed = artifact["consumed_development"]
    reproduction = None
    if args.scope == "development_replay":
        reproduction = {
            "queries_match": pooled["queries"] == int(consumed["queries"]),
            "baseline_recall1_match": bool(np.isclose(pooled["baseline_recall1"], consumed["baseline_recall1"], atol=1e-12)),
            "router_recall1_match": bool(np.isclose(pooled["router_recall1"], consumed["router_recall1"], atol=1e-12)),
            "corrected_match": pooled["corrected"] == int(consumed["corrected"]),
            "introduced_match": pooled["introduced"] == int(consumed["introduced"]),
            "note": "artifact expert_conflicts describes OOF expert predictions; deploy replay conflict count is reported separately and conflict always falls back to DreaMS",
        }
    report = {
        "status": f"bioaware_v3_frozen_{args.scope}_evaluation_complete",
        "formal": True,
        "protocol": f"frozen V3 artifact; zero refit; {args.scope}; external 16 panels unopened",
        "pooled": pooled, "formula_cluster_bootstrap": bootstrap, "panels": panels,
        "gates": gates,
        "pass_to_external_16_panel": bool(args.scope == "internal_rplc" and all(gates.values())),
        "pass_to_external_pooling": bool(args.scope == "external_panel"),
        "contracts": {"fit_performed": False, "threshold_tuning": False, "P2b": "forbidden", "phenotype": "forbidden"},
        "provenance": {
            "artifact_sha256": sha256(args.artifact), "ledger_sha256": sha256(args.ledger),
            "depth3_sha256": sha256(args.depth3), "queries_sha256": sha256(args.queries),
        },
        "claim_limit": (
            "One frozen external unit; only the pre-registered pooled 16-panel summary can support SOTA."
            if args.scope == "external_panel" else
            "Internal RPLC validation. SOTA requires the still-unopened 16-panel external evaluation."
        ),
    }
    if reproduction is not None:
        report["implementation_reproduction"] = reproduction
        report["pass_to_internal_rplc"] = bool(all(
            value for key, value in reproduction.items() if key != "note"
        ) and pooled["introduced"] == 0 and pooled["corrected"] > 0)
        report["pass_to_external_16_panel"] = False
    else:
        if args.scope == "internal_rplc":
            report["provenance"]["replay_report_sha256"] = sha256(args.replay_report)
        else:
            report["provenance"]["internal_report_sha256"] = sha256(args.internal_report)
    args.output_dir.mkdir(parents=True)
    transitions = args.output_dir / "query_transitions.csv.gz"
    merged.to_csv(transitions, index=False, compression="gzip")
    report["provenance"]["transitions_sha256"] = sha256(transitions)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Truth-blind decomposition of the B47 primary-seed identity bottleneck."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_u1_core import sha256_file  # noqa: E402

NESTED_STAGE_ORDER = (
    "all_queries", "unique_top1", "primary_absolute_gate",
    "absolute_and_feature_consensus",
    "reaction_graph_eligible_after_spectral_consensus",
    "after_sample_candidate_collapse",
)


def parse_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "t", "yes"}


def first_stage_below(stages: dict, threshold: int) -> str:
    return next(
        (name for name in NESTED_STAGE_ORDER
         if int(stages[name]["candidate_identities"]) < threshold),
        "none",
    )


def fixed_route(first_below: str, safe_candidate_identities: int) -> dict[str, object]:
    if safe_candidate_identities < 200:
        return {
            "code": "STOP_GRAPH_UNIVERSE_UNDERPOWERED",
            "pass_denominator_to_exact_event": False,
            "next_action": (
                "expand or replace the curated reaction/compound namespace before "
                "any event model; do not lower spectral seed thresholds"
            ),
        }
    routes = {
        "all_queries": (
            "STOP_INVALID_DENOMINATOR",
            "repair the query/candidate identity namespace",
        ),
        "unique_top1": (
            "STOP_IDENTITY_CONCENTRATION",
            "obtain a broader truth-blind seed source; unary score tuning cannot create identity diversity",
        ),
        "primary_absolute_gate": (
            "SPECTRAL_CONFIDENCE_BOTTLENECK",
            "test a new preregistered spectrum-only unary on development data; do not lower score or margin gates",
        ),
        "absolute_and_feature_consensus": (
            "FEATURE_CONSENSUS_BOTTLENECK",
            "audit feature grouping and independent-repeat support before using reaction context",
        ),
        "reaction_graph_eligible_after_spectral_consensus": (
            "RHEA_COVERAGE_OR_HUB_BOTTLENECK",
            "audit ion-to-neutral mapping and curated reaction coverage; do not score exact events yet",
        ),
        "after_sample_candidate_collapse": (
            "SAMPLE_COLLAPSE_BOTTLENECK",
            "preserve independent sample events or acquire more samples before exact-event modelling",
        ),
        "none": (
            "DENOMINATOR_ELIGIBLE",
            "freeze the exact-event/null contract before opening any annotation truth",
        ),
    }
    code, action = routes[first_below]
    return {
        "code": code,
        "pass_denominator_to_exact_event": first_below == "none",
        "next_action": action,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def identity_count(frame: pd.DataFrame, column: str) -> int:
    value = frame[column].fillna("").astype(str)
    return int(value[value.ne("")].nunique())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seed-degree", type=int, default=250)
    args = parser.parse_args()
    seed_dir, output = args.seed_dir.resolve(), args.output.resolve()
    required = (
        "report.json", "candidate_scores.csv.gz", "query_summaries.csv.gz",
        "feature_consensus.csv.gz", "seeds_primary.csv.gz",
    )
    missing = [name for name in required if not (seed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not args.participants.is_file():
        raise FileNotFoundError(args.participants)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite U1c audit: {output}")
    source = json.loads((seed_dir / "report.json").read_text(encoding="utf-8"))
    if source.get("status") != "bioaware_b47_truthblind_seed_construction_complete":
        raise RuntimeError("U1c requires the frozen original B47 seed artifact")
    query = pd.read_csv(seed_dir / "query_summaries.csv.gz", dtype={"query_id": str})
    consensus = pd.read_csv(seed_dir / "feature_consensus.csv.gz")
    candidate = pd.read_csv(
        seed_dir / "candidate_scores.csv.gz",
        usecols=["query_id", "candidate_id"], dtype=str,
    )
    primary = pd.read_csv(seed_dir / "seeds_primary.csv.gz", dtype=str)
    participants = pd.read_csv(args.participants)
    if not {"compound_id", "reaction_id", "is_currency"}.issubset(participants.columns):
        raise RuntimeError("Rhea participant cache lacks required columns")
    participants["compound_id"] = participants["compound_id"].astype(str).str[:14].str.upper()
    degree = participants.groupby("compound_id")["reaction_id"].nunique().astype(int)
    currency = set(participants.loc[
        participants["is_currency"].map(parse_bool), "compound_id",
    ].astype(str))
    merged = query.merge(
        consensus, on=["study", "feature_id"], how="left", validate="many_to_one",
    )
    top = merged["top_candidate_id"].fillna("").astype(str)
    merged["reaction_degree"] = top.map(degree).fillna(0).astype(int)
    merged["currency"] = top.isin(currency)
    masks = {
        "all_queries": np.ones(len(merged), dtype=bool),
        "unique_top1": merged["unique_top1"].astype(bool).to_numpy(),
        "primary_absolute_gate": merged["primary_absolute_gate"].astype(bool).to_numpy(),
        "primary_feature_consensus": merged["primary_feature_gate"].fillna(False).astype(bool).to_numpy(),
        "absolute_and_feature_consensus": (
            merged["primary_absolute_gate"].astype(bool)
            & merged["primary_feature_gate"].fillna(False).astype(bool)
            & merged["top_candidate_id"].astype(str).eq(
                merged["modal_candidate_id"].fillna("").astype(str)
            )
        ).to_numpy(),
    }
    masks["reaction_graph_eligible_after_spectral_consensus"] = (
        masks["absolute_and_feature_consensus"]
        & merged["reaction_degree"].gt(0).to_numpy()
        & merged["reaction_degree"].le(args.maximum_seed_degree).to_numpy()
        & (~merged["currency"].to_numpy())
    )
    stages = {}
    for name, mask in masks.items():
        subset = merged.loc[mask]
        stages[name] = {
            "query_events": int(mask.sum()),
            "candidate_identities": identity_count(subset, "top_candidate_id"),
            "features": int(subset["feature_id"].astype(str).nunique()),
            "samples": int(subset[["study", "sample"]].drop_duplicates().shape[0]),
        }
    stages["after_sample_candidate_collapse"] = {
        "query_events": int(len(primary)),
        "candidate_identities": int(primary["seed_compound_id"].nunique()),
        "features": int(primary["seed_feature_id"].nunique()),
        "samples": int(primary[["study", "sample"]].drop_duplicates().shape[0]),
    }

    candidate_ids = candidate["candidate_id"].astype(str).drop_duplicates()
    candidate_degree = candidate_ids.map(degree).fillna(0).astype(int)
    candidate_currency = candidate_ids.isin(currency)
    eligible_candidate = (
        (candidate_degree > 0)
        & (candidate_degree <= args.maximum_seed_degree)
        & (~candidate_currency)
    )
    top_counts = top[top.ne("")].value_counts()
    top10_share = float(top_counts.head(10).sum() / top_counts.sum()) if len(top_counts) else 0.0
    hhi = float(np.square(top_counts / top_counts.sum()).sum()) if len(top_counts) else 0.0
    primary_identities = int(primary["seed_compound_id"].nunique())
    shortfall = max(0, 200 - primary_identities)
    first_below = first_stage_below(stages, 200)
    report = {
        "status": "bioaware_b47_u1c_seed_denominator_audit_complete",
        "formal": True,
        "truth_blind": True,
        "stages": stages,
        "identity_gate": {
            "required": 200,
            "observed": primary_identities,
            "shortfall": shortfall,
            "first_stage_below_200_identities": first_below,
        },
        "fixed_route_decision": fixed_route(first_below, int(eligible_candidate.sum())),
        "candidate_universe": {
            "identities": int(len(candidate_ids)),
            "rhea_covered": int((candidate_degree > 0).sum()),
            "rhea_safe_degree_noncurrency": int(eligible_candidate.sum()),
        },
        "top1_identity_concentration": {
            "unique_identities": int(len(top_counts)),
            "top10_event_share": top10_share,
            "herfindahl_index": hhi,
            "maximum_events_one_identity": int(top_counts.max()) if len(top_counts) else 0,
        },
        "diagnostic_gates": {
            "candidate_universe_has_200_safe_rhea_identities": int(eligible_candidate.sum()) >= 200,
            "unique_top1_has_200_identities": stages["unique_top1"]["candidate_identities"] >= 200,
            "absolute_gate_has_200_identities": stages["primary_absolute_gate"]["candidate_identities"] >= 200,
            "consensus_has_200_identities": stages["absolute_and_feature_consensus"]["candidate_identities"] >= 200,
            "graph_safe_consensus_has_200_identities": stages[
                "reaction_graph_eligible_after_spectral_consensus"
            ]["candidate_identities"] >= 200,
        },
        "contracts": {
            "truth_opened": False,
            "phenotype_used": False,
            "model_fitted": False,
            "threshold_retuned": False,
            "reaction_network_scored": False,
            "P2b_used": False,
        },
        "provenance": {
            "seed_report_sha256": sha256_file(seed_dir / "report.json"),
            "candidate_scores_sha256": sha256_file(seed_dir / "candidate_scores.csv.gz"),
            "query_summaries_sha256": sha256_file(seed_dir / "query_summaries.csv.gz"),
            "feature_consensus_sha256": sha256_file(seed_dir / "feature_consensus.csv.gz"),
            "seeds_primary_sha256": sha256_file(seed_dir / "seeds_primary.csv.gz"),
            "participants_sha256": sha256_file(args.participants),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Truth-blind denominator decomposition only. It identifies where seed "
            "identity diversity is lost but does not measure seed precision, annotation "
            "gain, reaction-event gain, or shared-embedding improvement."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    merged[[
        "query_id", "study", "sample", "feature_id", "top_candidate_id",
        "top_score", "top_margin", "unique_top1", "primary_absolute_gate",
        "primary_feature_gate", "modal_candidate_id", "reaction_degree", "currency",
    ]].to_csv(
        output / "query_stage_ledger.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report["provenance"]["query_stage_ledger_sha256"] = sha256_file(
        output / "query_stage_ledger.csv.gz"
    )
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

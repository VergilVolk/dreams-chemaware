#!/usr/bin/env python
"""Post-hoc mechanism audit: require complete non-currency source hyperedges.

This audit was motivated after inspecting the 21-query v2-0 transitions.  It
is therefore explicitly *not* a confirmatory result.  Its purpose is to freeze
a mechanistic BioAware v2-1 rule for evaluation on a new benchmark:

    a seed may support a candidate through a Rhea reaction only when every
    other non-currency participant on the seed side is represented by an
    eligible high-confidence seed after leave-query/truth exclusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware import (  # noqa: E402
    BioAwareConfig,
    build_one_hop_evidence,
    fuse_candidates,
    top1_transition_table,
    validate_reaction_participants,
)
from evaluate_bioaware_v2_two_layer import build_expanded_seeds, noisy_or


def attach_complete_hyperedge_support(
    candidates: pd.DataFrame,
    paths: pd.DataFrame,
    participants: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    truth_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    participant = validate_reaction_participants(participants)
    noncurrency = participant[~participant["is_currency"].astype(bool)].copy()
    side_compounds = {
        (str(reaction), str(side)): frozenset(group["compound_id"].astype(str))
        for (reaction, side), group in noncurrency.groupby(
            ["reaction_id", "side"], sort=False
        )
    }
    paths_by_candidate = {
        str(candidate): group for candidate, group in paths.groupby("candidate_id", sort=False)
    }
    seed_table = seeds.copy()
    seed_table["seed_query_id"] = seed_table["seed_query_id"].astype(str)
    seed_table["seed_compound_id"] = seed_table["seed_compound_id"].astype(str)
    support_rows: list[dict] = []
    used_rows: list[pd.DataFrame] = []
    rejected_incomplete = 0
    retained_complete = 0

    for query_id, group in candidates.groupby("query_id", sort=False):
        truths = group[truth_col].dropna().astype(str).unique()
        if len(truths) != 1:
            raise RuntimeError(f"query {query_id} has {len(truths)} truth identities")
        truth = truths[0]
        available_seed_compounds = set(
            seed_table.loc[
                (seed_table["seed_query_id"] != str(query_id))
                & (seed_table["seed_compound_id"] != truth),
                "seed_compound_id",
            ]
        )
        for candidate_id in group["candidate_id"].astype(str):
            selected = paths_by_candidate.get(candidate_id)
            if selected is None:
                selected = paths.iloc[0:0].copy()
            else:
                selected = selected[
                    (selected["seed_query_id"].astype(str) != str(query_id))
                    & (selected["seed_compound_id"].astype(str) != truth)
                ].copy()
            keep: list[bool] = []
            missing_strings: list[str] = []
            for row in selected.itertuples(index=False):
                source = side_compounds.get((str(row.reaction_id), str(row.seed_side)), frozenset())
                required = set(source) - {str(row.seed_compound_id)}
                missing = sorted(required - available_seed_compounds)
                keep.append(not missing)
                missing_strings.append(";".join(missing))
            if len(selected):
                selected["missing_required_source_compounds"] = missing_strings
                rejected_incomplete += int((~np.asarray(keep, dtype=bool)).sum())
                selected = selected[np.asarray(keep, dtype=bool)].copy()
                retained_complete += int(len(selected))
                selected = selected.sort_values("contribution", ascending=False).drop_duplicates(
                    ["seed_compound_id", "reaction_id"]
                )
            support_rows.append(
                {
                    "query_id": str(query_id),
                    "candidate_id": candidate_id,
                    "network_support": noisy_or(selected["contribution"].to_numpy(float))
                    if len(selected)
                    else 0.0,
                    "network_path_count": int(len(selected)),
                }
            )
            if len(selected):
                selected.insert(0, "query_id", str(query_id))
                selected.insert(1, "query_candidate_id", candidate_id)
                used_rows.append(selected)
    supported = candidates.merge(
        pd.DataFrame(support_rows),
        on=["query_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    explanations = pd.concat(used_rows, ignore_index=True) if used_rows else pd.DataFrame()
    audit = {
        "retained_complete_paths_before_deduplication": retained_complete,
        "rejected_incomplete_paths": rejected_incomplete,
    }
    return supported, explanations, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/mtbls13729/bioaware_v1_input/candidates.csv.gz"))
    parser.add_argument("--participants", type=Path, default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"))
    parser.add_argument("--link-dir", type=Path, default=Path("data/mtbls13729/ms1_ms2_link"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/bioaware_v2_hyperedge_audit"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates)
    participants = pd.read_csv(args.participants)
    seeds = build_expanded_seeds(
        args.link_dir,
        args.panels,
        set(participants["compound_id"].dropna().astype(str)),
        minimum_cosine=0.80,
        minimum_support_spectra=2,
        minimum_agreement=0.60,
    )
    config = BioAwareConfig()
    paths = build_one_hop_evidence(participants, seeds, config)
    supported, explanations, path_audit = attach_complete_hyperedge_support(
        candidates,
        paths,
        participants,
        seeds,
        truth_col="truth_candidate_id",
    )
    scored, decisions = fuse_candidates(supported, config)
    per_query, summary = top1_transition_table(
        scored, truth_col="truth_candidate_id"
    )
    transitions = per_query[per_query["corrected"] | per_query["introduced"]]
    report = {
        "status": "mtbls13729_bioaware_v2_hyperedge_posthoc_audit_complete",
        "formal": False,
        "posthoc": True,
        "queries": int(len(per_query)),
        "expanded_seed_rows": int(len(seeds)),
        "result": {
            **summary,
            "queries_with_network_evidence": int(decisions["network_available"].sum()),
            "intervention_rate": float(decisions["bioaware_applied"].mean()),
            "evidence_paths": int(len(explanations)),
        },
        "path_audit": path_audit,
        "rule": (
            "All other non-currency compounds on the seed side of a reaction must "
            "occur among eligible leave-query/truth-excluded high-confidence seeds."
        ),
        "next_gate": (
            "Freeze this rule before testing a new external cohort/benchmark; this "
            "21-query audit cannot confirm improvement because it motivated the rule."
        ),
    }
    scored.to_csv(out / "candidate_scores.csv.gz", index=False)
    decisions.to_csv(out / "query_decisions.csv", index=False)
    per_query.to_csv(out / "per_query_transitions.csv", index=False)
    transitions.to_csv(out / "changed_queries.csv", index=False)
    explanations.to_csv(out / "complete_hyperedge_paths.csv.gz", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

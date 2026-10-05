#!/usr/bin/env python
"""Post-outcome coverage funnel for the failed B39-M2 atomic actions.

This diagnostic never selects, edits or promotes an action.  It explains how
many candidate events survive each independently named semantic/evidence
condition and where the frozen B37 topology transitions lose atomic support.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd


INTERNAL = {"BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def candidate_coverage(events: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = events.loc[mask, ["query_id", "candidate_id"]].drop_duplicates()
    selected["supported"] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--m2-dir", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, default=Path("data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz"))
    parser.add_argument("--b37-dir", type=Path, default=Path("data/validation/bioaware_b37_local_fullcheck_v2_20260913"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events_path = args.m1_dir / "internal_atomic_events_with_edge_evidence.csv.gz"
    m1_report_path = args.m1_dir / "report.json"
    m2_report_path = args.m2_dir / "report.json"
    transitions_path = args.b37_dir / "catalog_ablation_transitions.csv.gz"
    for path in (events_path, m1_report_path, m2_report_path, args.candidate_features, transitions_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    m1_report = json.loads(m1_report_path.read_text(encoding="utf-8"))
    m2_report = json.loads(m2_report_path.read_text(encoding="utf-8"))
    if m1_report.get("provenance", {}).get("joined_events") != sha256(events_path):
        raise RuntimeError("M1 event provenance mismatch")
    if m2_report.get("status") != "bioaware_b39_m2_internal_fixed_action_complete":
        raise RuntimeError("M2 result is unavailable or incomplete")
    events = pd.read_csv(events_path, dtype={"reaction_id": str}, low_memory=False)
    candidates = pd.read_csv(args.candidate_features, low_memory=False)
    transitions = pd.read_csv(transitions_path, low_memory=False)
    topology = transitions.loc[transitions["arm"].eq("catalog_topology") & transitions["source"].isin(INTERNAL)].copy()
    queries = set(topology["query_id"].astype(str))
    if len(topology) != 548 or len(queries) != 548:
        raise RuntimeError("B37 internal topology panel drift")
    events = events.loc[events["query_id"].astype(str).isin(queries)].copy()
    candidates = candidates.loc[candidates["query_id"].astype(str).isin(queries)].copy()
    candidate_truth = candidates[["query_id", "candidate_id", "is_positive"]].drop_duplicates()
    candidate_truth["is_positive"] = boolean(candidate_truth["is_positive"])

    rhea = events["edge_source"].astype(str).eq("Rhea")
    nonnoop = ~boolean(events["reaction_identity_noop"])
    directed = events["direction_status"].astype(str).isin({"supported", "supported_bidirectional"})
    complete = boolean(events["hyperedge_context_complete"])
    candidate_specific = boolean(events["event_candidate_specific"])
    spectral_positive = boolean(events["spectral_available"]) & events["spectral_excess_dual_view"].fillna(0.0).gt(0)
    coabundance_positive = boolean(events["coabundance_available"]) & events["coabundance_excess_positive"].fillna(0.0).gt(0)
    definitions: list[tuple[str, pd.Series]] = [
        ("F00_any_atomic_catalogue_event", pd.Series(True, index=events.index)),
        ("F01_rhea_event", rhea),
        ("F02_rhea_nonidentity_event", rhea & nonnoop),
        ("F03_direction_supported", rhea & nonnoop & directed),
        ("F04_hyperedge_complete", rhea & nonnoop & complete),
        ("F05_spectral_positive", rhea & nonnoop & spectral_positive),
        ("F06_coabundance_positive", rhea & nonnoop & coabundance_positive),
        ("F07_both_experimental_positive", rhea & nonnoop & spectral_positive & coabundance_positive),
        ("F08_directed_complete", rhea & nonnoop & directed & complete & candidate_specific),
        ("F09_directed_complete_spectral", rhea & nonnoop & directed & complete & candidate_specific & spectral_positive),
        ("F10_directed_complete_coabundance", rhea & nonnoop & directed & complete & candidate_specific & coabundance_positive),
        ("F11_directed_complete_both", rhea & nonnoop & directed & complete & candidate_specific & spectral_positive & coabundance_positive),
    ]
    topology["transition_class"] = np.select(
        [boolean(topology["corrected"]), boolean(topology["introduced"]), boolean(topology["baseline_correct"])],
        ["topology_corrected", "topology_introduced", "protected_correct"],
        default="persistent_wrong",
    )
    funnel: dict[str, Any] = {}
    transition_support: dict[str, Any] = {}
    for name, mask in definitions:
        covered = candidate_coverage(events, mask)
        labelled = covered.merge(candidate_truth, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
        truth_supported = labelled.loc[labelled["is_positive"], ["query_id"]].drop_duplicates()
        wrong_supported = labelled.loc[~labelled["is_positive"], ["query_id"]].drop_duplicates()
        funnel[name] = {
            "event_rows": int(mask.sum()),
            "queries": int(covered["query_id"].nunique()),
            "candidate_pairs": int(len(covered)),
            "queries_with_truth_supported": int(truth_supported["query_id"].nunique()),
            "queries_with_wrong_supported": int(wrong_supported["query_id"].nunique()),
        }
        truth_keys = set(zip(labelled.loc[labelled["is_positive"], "query_id"].astype(str), labelled.loc[labelled["is_positive"], "candidate_id"].astype(str)))
        final_keys = set(zip(covered["query_id"].astype(str), covered["candidate_id"].astype(str)))
        per_class = {}
        for transition_class, local in topology.groupby("transition_class", sort=False):
            truth_hits = sum((str(row.query_id), str(row.truth_candidate_id)) in truth_keys for row in local.itertuples())
            final_hits = sum((str(row.query_id), str(row.final_candidate_id)) in final_keys for row in local.itertuples())
            per_class[str(transition_class)] = {
                "queries": int(len(local)),
                "truth_candidate_supported": int(truth_hits),
                "topology_final_candidate_supported": int(final_hits),
            }
        transition_support[name] = per_class

    report = {
        "status": "bioaware_b39_m2_coverage_funnel_complete",
        "formal": True,
        "post_outcome_diagnostic": True,
        "action_selection_permitted": False,
        "queries": 548,
        "candidate_pairs": int(len(candidate_truth)),
        "atomic_event_rows": int(len(events)),
        "topology_transition_counts": dict(Counter(topology["transition_class"])),
        "funnel": funnel,
        "topology_transition_support": transition_support,
        "diagnosis": {
            "strict_action_bottleneck": "intersection coverage, not statistical power or optimizer capacity",
            "candidate_specificity_axis_identified": False,
            "hyperedge_complete_fraction_among_rhea": float(complete.loc[rhea].mean()),
            "directed_complete_both_query_fraction": float(funnel["F11_directed_complete_both"]["queries"] / 548),
        },
        "next_decision": "Do not tune B39-M2 thresholds. Reconstruct event-specific evidence and atomic nulls for L/H sources, while treating B37 topology as the engineering baseline. A context-representation model is justified only if event evidence survives source transfer or if a separately preregistered broader graph-completion action passes its null.",
        "provenance": {
            "events": sha256(events_path),
            "m1_report": sha256(m1_report_path),
            "m2_report": sha256(m2_report_path),
            "candidate_features": sha256(args.candidate_features),
            "b37_transitions": sha256(transitions_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Post-outcome diagnostic only. Funnel cells cannot be promoted as actions or counted as independent hypothesis tests.",
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

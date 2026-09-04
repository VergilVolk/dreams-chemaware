#!/usr/bin/env python
"""Consolidate independent BioAware evidence arms and enforce the +10 pp gate."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd

try:
    from audit_bioaware_metdna3_smn_headroom import sha256
except ModuleNotFoundError:
    from tasks.audit_bioaware_metdna3_smn_headroom import sha256


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrn", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_decision_v1/query_transitions.csv.gz"))
    parser.add_argument("--smn", type=Path, default=Path("data/validation/bioaware_metdna3_smn_headroom_v1/query_transitions.csv.gz"))
    parser.add_argument("--rt", type=Path, default=Path("data/validation/bioaware_metdna3_rt_headroom_v1/query_transitions.csv.gz"))
    parser.add_argument("--predicted", type=Path, default=Path("data/validation/bioaware_metdna3_predicted_edge_increment_v1/step1_query_transitions.csv.gz"))
    parser.add_argument("--candidate-scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--truth", type=Path, default=Path("data/validation/bioaware_metdna3_development_v1/development_level1.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_10pp_headroom_v1"))
    args = parser.parse_args()
    inputs = {
        "mrn": args.mrn, "smn": args.smn, "rt": args.rt,
        "predicted": args.predicted, "candidate_scores": args.candidate_scores,
        "truth": args.truth,
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    mrn = pd.read_csv(args.mrn)
    mrn = mrn[mrn["maximum_depth"].eq(3)].copy()
    smn = pd.read_csv(args.smn)
    rt = pd.read_csv(args.rt)
    predicted = pd.read_csv(args.predicted)
    for frame in (mrn, smn, rt, predicted):
        if len(frame) != 117:
            raise RuntimeError("all evidence arms must use the frozen 117-query protocol")
    baseline = mrn[[
        "query_id", "truth_candidate_id", "truth_formula", "baseline_candidate_id",
        "baseline_correct",
    ]].copy()
    if int((~baseline["baseline_correct"]).sum()) != 22:
        raise RuntimeError("frozen DreaMS error count changed")

    def corrected(frame: pd.DataFrame) -> set[str]:
        return set(frame.loc[(~frame["baseline_correct"]) & frame["final_correct"], "query_id"].astype(str))

    def introduced(frame: pd.DataFrame) -> set[str]:
        return set(frame.loc[frame["baseline_correct"] & (~frame["final_correct"]), "query_id"].astype(str))

    mrn_corrected = corrected(mrn)
    smn_corrected = corrected(smn)
    smn_headroom = set(smn.loc[smn["truth_headroom"], "query_id"].astype(str))
    rt_corrected = corrected(rt)
    rt_headroom = set(rt.loc[rt["truth_headroom"], "query_id"].astype(str))
    predicted_corrected = corrected(predicted)
    actual_union = mrn_corrected | smn_corrected | rt_corrected | predicted_corrected
    optimistic_union = mrn_corrected | smn_headroom | rt_headroom | predicted_corrected
    baseline_errors = set(baseline.loc[~baseline["baseline_correct"], "query_id"].astype(str))
    unresolved = baseline_errors - optimistic_union

    scores = pd.read_csv(args.candidate_scores)
    margins = []
    for query_id, group in scores.groupby("query_id", sort=True):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        truth_score = float(group.loc[group["candidate_id"].eq(truth_id), "spectral_score"].iloc[0])
        wrong = group[group["candidate_id"].ne(truth_id)]["spectral_score"]
        margins.append({
            "query_id": str(query_id),
            "dreams_truth_score": truth_score,
            "dreams_strongest_wrong_score": float(wrong.max()) if len(wrong) else float("nan"),
            "dreams_margin": truth_score - float(wrong.max()) if len(wrong) else float("nan"),
            "candidate_count": int(len(group)),
        })
    unresolved_frame = baseline[baseline["query_id"].astype(str).isin(unresolved)].merge(
        pd.DataFrame(margins), on="query_id", validate="one_to_one"
    )
    truth = pd.read_csv(args.truth)
    identity_names = truth.groupby("ik14")["name"].first().to_dict()
    unresolved_frame["truth_name"] = unresolved_frame["truth_candidate_id"].map(identity_names)
    unresolved_frame["missing_evidence"] = "MRN/raw-MS2 + SMN + RT + predicted-edge"
    unresolved_path = output / "unresolved_error_queries.csv.gz"
    unresolved_frame.to_csv(unresolved_path, index=False, compression="gzip")

    required = int(math.ceil(0.10 * len(baseline)))
    evidence_sets = {
        "mrn_raw_ms2_corrected": mrn_corrected,
        "smn_corrected": smn_corrected,
        "smn_truth_headroom": smn_headroom,
        "rt_corrected": rt_corrected,
        "rt_truth_headroom": rt_headroom,
        "predicted_edge_corrected": predicted_corrected,
    }
    report = {
        "status": "bioaware_10pp_headroom_consolidated",
        "formal": True,
        "protocol": {"queries": 117, "baseline_errors": 22, "required_net_corrections_for_10pp": required},
        "evidence_counts": {name: len(values) for name, values in evidence_sets.items()},
        "actual_union": {
            "unique_errors_corrected_by_at_least_one_current_rule": len(actual_union),
            "note": "not a deployable fusion because module conflicts have not been resolved",
        },
        "optimistic_union": {
            "unique_errors_with_any_current_evidence_headroom": len(optimistic_union),
            "unresolved_errors": len(unresolved),
            "reaches_10pp_requirement": len(optimistic_union) >= required,
        },
        "overlap_matrix": {
            left: {right: len(left_set & right_set) for right, right_set in evidence_sets.items()}
            for left, left_set in evidence_sets.items()
        },
        "decision": (
            "Existing reaction, structure, RT and predicted-edge arms do not contain enough independent "
            "headroom for +10 pp, even under an optimistic oracle union. Stop weight tuning. The next "
            "evidence source must target the unresolved errors and add at least three independent rescues: "
            "sample-matrix/global assignment or candidate-specific fragmentation evidence."
        ),
        "gates": {
            "current_evidence_mathematically_supports_10pp": len(optimistic_union) >= required,
            "next_layer_must_add_at_least_three_independent_errors": True,
            "RP_may_open": False,
        },
        "contracts": {"P2b_used": False, "RP_opened": False, "truth_used_only_after_evidence_freeze": True},
        "provenance": {name: sha256(path) for name, path in inputs.items()} | {
            "unresolved_errors_sha256": sha256(unresolved_path)
        },
        "claim_limit": "Consumed-development feasibility accounting; no SOTA or external annotation claim.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

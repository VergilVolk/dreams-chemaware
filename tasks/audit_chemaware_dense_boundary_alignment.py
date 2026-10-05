"""Audit whether dense ChemAware corrections retain query-level evidence.

The dense true-support curriculum first establishes chemical support on one
query per molecular identity.  This audit measures how many emitted native
DreaMS correction events use that exact query and how many inherit support
only through molecular identity.  Exact-query alignment is a necessary, but
not sufficient, condition for an event to be an evidence-bound chemical
triplet: an aligned query can still use a false candidate not contrasted by
the original chemical evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_true_support_native_triplets import true_candidate_support


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_ROLES = (31, 32)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument("--minimum-rule-wins", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def boundary_alignment_audit(
    evidence: Mapping[str, np.ndarray],
    triplets: Mapping[str, np.ndarray],
    manifest: Mapping[str, np.ndarray],
    minimum_rule_wins: int = 2,
) -> dict[str, object]:
    required = {"source_query", "negative_candidate", "curriculum_role"}
    if missing := sorted(required - set(triplets)):
        raise RuntimeError(f"triplet pool lacks alignment fields: {missing}")
    support, support_audit = true_candidate_support(
        evidence, manifest, minimum_rule_wins,
    )
    evidence_query_by_identity = {
        str(body["identity"]): int(query) for query, body in support.items()
    }
    if len(evidence_query_by_identity) != len(support):
        raise RuntimeError("chemical support contains repeated identities")

    roles = np.asarray(triplets["curriculum_role"], dtype=np.int64)
    event_mask = np.isin(roles, np.asarray(CORRECTION_ROLES, dtype=np.int64))
    queries = np.asarray(triplets["source_query"], dtype=np.int64)[event_mask]
    if not len(queries):
        raise RuntimeError("triplet pool contains no dense correction events")
    if np.any((queries < 0) | (queries >= len(manifest["query_ik14"]))):
        raise RuntimeError("dense correction query is outside the manifest")
    identities = np.asarray(manifest["query_ik14"])[queries].astype(str)
    negative_candidate = np.asarray(
        triplets["negative_candidate"], dtype=np.int64,
    )[event_mask]
    missing_identity = sorted(set(identities) - set(evidence_query_by_identity))
    if missing_identity:
        raise RuntimeError(
            f"dense correction identities lack source evidence: {len(missing_identity)}"
        )
    aligned = np.asarray([
        int(query) == evidence_query_by_identity[identity]
        for query, identity in zip(queries, identities, strict=True)
    ], dtype=bool)
    evidence_row_by_query = {
        int(query): row
        for row, query in enumerate(np.asarray(evidence["query"], dtype=np.int64))
    }
    if "baseline_candidate" not in evidence:
        raise RuntimeError("source evidence lacks the contrasted baseline candidate")
    same_baseline_candidate = np.asarray([
        (
            int(query) in evidence_row_by_query
            and int(candidate) == int(evidence["baseline_candidate"][evidence_row_by_query[int(query)]])
        )
        for query, candidate in zip(queries, negative_candidate, strict=True)
    ], dtype=bool)
    exact = aligned & same_baseline_candidate
    aligned_queries = np.unique(queries[aligned])
    broadcast_queries = np.unique(queries[~aligned])
    correction_events = int(len(queries))
    report = {
        "status": "CHEMAWARE_DENSE_BOUNDARY_ALIGNMENT_AUDIT_COMPLETE",
        "chemical_support": support_audit,
        "correction_roles": list(CORRECTION_ROLES),
        "correction_events": correction_events,
        "correction_queries": int(len(np.unique(queries))),
        "direct_query_evidence_events": int(np.sum(aligned)),
        "identity_broadcast_events": int(np.sum(~aligned)),
        "direct_query_evidence_fraction": float(np.mean(aligned)),
        "identity_broadcast_fraction": float(np.mean(~aligned)),
        "direct_query_evidence_queries": int(len(aligned_queries)),
        "identity_broadcast_queries": int(len(broadcast_queries)),
        "exact_query_and_contrasted_candidate_events": int(np.sum(exact)),
        "exact_query_and_contrasted_candidate_fraction": float(np.mean(exact)),
        "exact_query_and_contrasted_candidate_queries": int(len(np.unique(queries[exact]))),
        "exact_query_and_contrasted_candidate_formulas": int(len(np.unique(
            np.asarray(manifest["query_formula"])[queries[exact]].astype(str)
        ))),
        "strict_interpretation": (
            "Direct-query alignment is only a necessary upper bound on exact "
            "chemical-boundary alignment because one aligned query may emit "
            "multiple false candidates not contrasted by its source evidence."
        ),
        "gate": {
            "all_corrections_have_direct_query_evidence": bool(np.all(aligned)),
            "all_corrections_match_the_contrasted_candidate": bool(np.all(exact)),
        },
    }
    return report


def main() -> None:
    args = arguments()
    report = boundary_alignment_audit(
        load_npz(args.evidence), load_npz(args.triplets), load_npz(args.manifest),
        args.minimum_rule_wins,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

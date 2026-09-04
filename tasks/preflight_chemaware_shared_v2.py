"""Audit the frozen legacy inputs used by ChemAware shared embedding v2.

The frozen graph was built from the historical ``real_train_primary`` block,
which in turn treated MassSpecGym ``SIMULATION_CHALLENGE`` as spectrum
provenance.  That interpretation is incorrect: the field is benchmark-subset
membership.  Consequently this graph remains reproducible as a restricted
cohort, but is no longer admissible as the formal full-data ChemAware input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402


EXPECTED = {
    "graph_sha256": "5f2340751c7521c5a93114e2b134d5796f157148736ad9162d545b84c11d9f71",
    "hdf5_sha256": "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
    "official_checkpoint_sha256": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    "raw_checkpoint_sha256": "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--g0-rule-report", type=Path, default=ROOT / "data/validation/g8r_chemaware_g0_rule_cache.json")
    parser.add_argument("--g0-audit", type=Path, default=ROOT / "data/validation/g8r_chemaware_g0_full_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_preflight.json")
    parser.add_argument(
        "--legacy-cohort-reproduction-only",
        action="store_true",
        help="Audit the frozen restricted cohort only; never marks it formal or enables new training.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.legacy_cohort_reproduction_only:
        raise RuntimeError(
            "fail-closed: the frozen ChemAware graph was derived from the obsolete "
            "real_train_primary interpretation of SIMULATION_CHALLENGE. Rebuild the "
            "P3-disjoint graph from train_primary_all before any new formal training; "
            "use --legacy-cohort-reproduction-only only to audit historical results."
        )
    required = {
        "graph": args.graph, "hdf5": args.data,
        "raw_checkpoint": args.raw_checkpoint,
        "official_checkpoint": args.official_checkpoint,
        "g0_rule_report": args.g0_rule_report, "g0_audit": args.g0_audit,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    observed = {
        "graph_sha256": sha256_file(args.graph),
        "hdf5_sha256": sha256_file(args.data),
        "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
    }
    mismatch = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in EXPECTED.items() if observed[key] != value
    }
    if mismatch:
        raise RuntimeError(f"frozen ChemAware input hash mismatch: {mismatch}")
    graph = CandidateGraph(args.graph)
    reachable = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    identities = np.unique(graph.molecule_ik14.astype(str))
    with h5py.File(args.data, "r") as handle:
        membership = np.asarray(handle["SIMULATION_CHALLENGE"].asstr()[reachable], dtype=str)
    membership_counts = {
        str(value): int(np.sum(membership == value)) for value in np.unique(membership)
    }
    if graph.n_queries != 23876 or len(reachable) != 25275 or len(identities) != 3472:
        raise RuntimeError(
            "formal graph cardinality mismatch: "
            f"queries={graph.n_queries}, spectra={len(reachable)}, identities={len(identities)}"
        )
    if np.any(graph.query_formula.astype(str) == "") or np.any(graph.molecule_formula.astype(str) == ""):
        raise RuntimeError("formal graph contains empty formulas")
    if not np.all(np.isfinite(graph.features)):
        raise RuntimeError("formal graph contains non-finite features")

    rule = json.loads(args.g0_rule_report.read_text(encoding="utf-8"))
    if (
        rule.get("status") != "chemaware_g0_rule_cache_complete"
        or rule.get("formal") is not True
        or int(rule.get("rules", -1)) != 3486
        or int(rule.get("cached_spectra", -1)) != len(reachable)
        or rule.get("provenance", {}).get("candidate_graph_sha256") != observed["graph_sha256"]
        or rule.get("provenance", {}).get("hdf5_sha256") != observed["hdf5_sha256"]
    ):
        raise RuntimeError("formal G0 rule report is absent, stale, or inconsistent")
    audit = json.loads(args.g0_audit.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "chemaware_g0_full_graph_passed"
        or audit.get("formal") is not True
        or audit.get("gates", {}).get("pass") is not True
        or int(audit.get("p3_query_identity_overlap", -1)) != 0
        or audit.get("provenance", {}).get("graph_sha256") != observed["graph_sha256"]
        or audit.get("provenance", {}).get("hdf5_sha256") != observed["hdf5_sha256"]
    ):
        raise RuntimeError("formal G0 full audit did not pass the frozen graph")
    report = {
        "status": "chemaware_shared_v2_legacy_cohort_audited",
        "formal": False,
        "admissible_for_new_training": False,
        "reason": (
            "graph was built from a cohort selected by an obsolete interpretation of "
            "SIMULATION_CHALLENGE; membership is not spectrum provenance"
        ),
        "graph": {
            "queries": graph.n_queries,
            "reachable_spectra": int(len(reachable)),
            "candidate_identities": int(len(identities)),
            "rules": int(rule["rules"]),
            "p3_query_identity_overlap": 0,
            "simulation_challenge_membership_counts": membership_counts,
        },
        "hashes": observed | {
            "g0_rule_report_sha256": sha256_file(args.g0_rule_report),
            "g0_audit_sha256": sha256_file(args.g0_audit),
        },
        "contracts": {
            "strict_10ppm_same_adduct": True,
            "positive_unique_and_first": True,
            "formula_isolation_required_downstream": True,
            "query_reference_encoder_shared": True,
            "candidate_inputs_at_inference": False,
            "simulation_challenge_used_as_provenance": False,
        },
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError(f"existing preflight differs from fresh audit: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

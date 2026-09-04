#!/usr/bin/env python
"""Recover the P3-disjoint training allow-list from the consumed P3 audit.

The local checkout contains the full consumed per-query audit but not the
sealed directory copied from the server.  The original P3 builder defines the
allow-list as every train-fold identity minus every P3 query identity; this
script applies that exact public contract and records the reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--audit", type=Path, default=Path("data/validation/g8r_p3_transition_audit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_p3_allow_recovered_corrected_v3_20260902"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    audit = pd.read_csv(args.audit)
    if not {"panel", "query_row", "ik14"}.issubset(audit):
        raise RuntimeError("consumed P3 audit schema mismatch")
    p3 = set(audit.ik14.astype(str))
    if len(p3) != 4219 or len(audit) != 7606:
        raise RuntimeError("consumed P3 identity/query count changed")
    with h5py.File(args.hdf5, "r") as handle:
        fold = decode(handle["fold"][:])
        simulation_challenge = decode(handle["SIMULATION_CHALLENGE"][:])
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])])
    train_primary = sorted(set(ik14[fold == "train"]) - p3)
    train_rows_mask = (fold == "train") & np.isin(ik14, train_primary)
    train_rows = np.flatnonzero(train_rows_mask).astype(int).tolist()
    member_rows = np.flatnonzero(train_rows_mask & (simulation_challenge == "True")).astype(int).tolist()
    nonmember_rows = np.flatnonzero(train_rows_mask & (simulation_challenge == "False")).astype(int).tolist()
    if len(train_primary) != 19403 or len(train_rows) != 137830:
        raise RuntimeError(f"corrected allow-list count mismatch: {len(train_primary)}/{len(train_rows)}")
    if set(train_primary) & p3:
        raise RuntimeError("recovered allow-list overlaps P3")
    body = {
        "train_primary_all": {"n": len(train_primary), "ik14": train_primary,
                              "n_rows": len(train_rows), "rows": train_rows},
        "simulation_challenge_members": {"n_rows": len(member_rows), "rows": member_rows},
        "simulation_challenge_nonmembers": {"n_rows": len(nonmember_rows), "rows": nonmember_rows},
        "p3_query_overlap": 0,
        "simulation_challenge_semantics": (
            "Eligibility mask for the spectrum-simulation benchmark, not spectrum provenance"
        ),
        "rule": "P2 loaders must intersect train rows with train_primary_all identities",
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "p3_p2_allowed_training_ik14.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"status": "g8r_p3_allow_recovered_corrected_semantics", "formal": False,
              "p3_queries": len(audit), "p3_identities": len(p3),
              "train_identities": len(train_primary), "train_rows": len(train_rows),
              "simulation_challenge_member_rows": len(member_rows),
              "simulation_challenge_nonmember_rows": len(nonmember_rows), "p3_overlap": 0,
              "field_semantics_corrected": True,
              "byte_identity_to_server_seal_claimed": False,
              "provenance": {"hdf5_sha256": sha256(args.hdf5), "audit_sha256": sha256(args.audit),
                             "allow_sha256": sha256(path)},
              "claim_limit": "Corrected local semantic reconstruction from the consumed P3 audit; it is not the unavailable server-sealed artifact."}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

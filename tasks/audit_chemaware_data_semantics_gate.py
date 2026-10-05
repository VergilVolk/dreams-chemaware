"""Fail-closed audit of the corrected ChemAware data semantics and row set.

The MassSpecGym ``SIMULATION_CHALLENGE`` field is benchmark-subset membership,
not experimental/simulated provenance.  This audit independently reconstructs
the allowed training row set without using that field, then proves that both
membership values survive unchanged into the pool reports and manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument(
        "--allow", type=Path,
        default=(ROOT / "data/validation/g8r_p3_allow_recovered_corrected_v3_20260902/"
                 "p3_p2_allowed_training_ik14.json"),
    )
    parser.add_argument(
        "--allow-report", type=Path,
        default=ROOT / "data/validation/g8r_p3_allow_recovered_corrected_v3_20260902/report.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument("--pools", nargs="+", type=Path, default=[
        ROOT / "data/e1/chemaware_control_train_mh_triplet_pool_10ppm_p3disjoint_v3.npz",
        ROOT / "data/e1/chemaware_control_train_mna_triplet_pool_10ppm_p3disjoint_v3.npz",
    ])
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_data_semantics_gate_v1/report.json",
    )
    return parser.parse_args()


def exact_set(name: str, observed: np.ndarray, expected: np.ndarray) -> None:
    observed = np.unique(np.asarray(observed, dtype=np.int64))
    expected = np.unique(np.asarray(expected, dtype=np.int64))
    if not np.array_equal(observed, expected):
        missing = np.setdiff1d(expected, observed); extra = np.setdiff1d(observed, expected)
        raise RuntimeError(f"{name} row-set mismatch: missing={len(missing)} extra={len(extra)}")


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    required = [args.data, args.allow, args.allow_report, args.manifest, args.manifest.with_suffix(".json")]
    for pool in args.pools:
        required.extend((pool, pool.with_suffix(".json")))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    allow = json.loads(args.allow.read_text(encoding="utf-8"))
    allow_report = json.loads(args.allow_report.read_text(encoding="utf-8"))
    semantics = allow.get("simulation_challenge_semantics", "")
    if "not spectrum provenance" not in semantics:
        raise RuntimeError("allow-list does not declare corrected SIMULATION_CHALLENGE semantics")
    train = allow.get("train_primary_all", {})
    identities = np.asarray(train.get("ik14", []), dtype=str)
    allowed_rows = np.asarray(train.get("rows", []), dtype=np.int64)
    member_rows = np.asarray(allow.get("simulation_challenge_members", {}).get("rows", []), dtype=np.int64)
    nonmember_rows = np.asarray(allow.get("simulation_challenge_nonmembers", {}).get("rows", []), dtype=np.int64)
    if len(identities) != int(train.get("n", -1)) or len(allowed_rows) != int(train.get("n_rows", -1)):
        raise RuntimeError("allow-list declared counts do not match payload")
    if len(np.unique(identities)) != len(identities) or len(np.unique(allowed_rows)) != len(allowed_rows):
        raise RuntimeError("allow-list identities or rows are duplicated")
    if np.intersect1d(member_rows, nonmember_rows).size:
        raise RuntimeError("membership partitions overlap")
    exact_set("membership union", np.r_[member_rows, nonmember_rows], allowed_rows)

    with h5py.File(args.data, "r") as handle:
        required_fields = {"INCHIKEY", "FORMULA", "adduct", "fold", "SIMULATION_CHALLENGE"}
        if missing_fields := required_fields - set(handle.keys()):
            raise RuntimeError(f"HDF5 schema misses {sorted(missing_fields)}")
        total = len(handle["INCHIKEY"])
        if any(len(handle[field]) != total for field in required_fields):
            raise RuntimeError("HDF5 schema fields have unequal row counts")
        ik14 = np.asarray([value[:14] for value in handle["INCHIKEY"].asstr()[:]], dtype=str)
        formula = np.asarray(handle["FORMULA"].asstr()[:], dtype=str)
        adduct = np.asarray(handle["adduct"].asstr()[:], dtype=str)
        fold = np.asarray(handle["fold"].asstr()[:], dtype=str)
        membership = np.asarray(handle["SIMULATION_CHALLENGE"].asstr()[:], dtype=str)
    if set(np.unique(membership)) != {"False", "True"}:
        raise RuntimeError("SIMULATION_CHALLENGE is not a strict False/True membership field")
    if np.any((allowed_rows < 0) | (allowed_rows >= total)):
        raise RuntimeError("allow-list contains out-of-range rows")
    # Critical invariant: reconstruct the row set without reading membership.
    reconstructed = np.flatnonzero(
        (fold == "train") & np.isin(ik14, identities) & np.isin(adduct, ("[M+H]+", "[M+Na]+"))
    )
    exact_set("membership-blind reconstructed training set", allowed_rows, reconstructed)
    exact_set("member partition", member_rows, allowed_rows[membership[allowed_rows] == "True"])
    exact_set("nonmember partition", nonmember_rows, allowed_rows[membership[allowed_rows] == "False"])
    if np.any(formula[allowed_rows] == ""):
        raise RuntimeError("allowed rows contain empty formula")

    pool_reports = []
    covered_rows = []
    for pool in args.pools:
        report = json.loads(pool.with_suffix(".json").read_text(encoding="utf-8"))
        if report.get("simulation_challenge_semantics") != (
            "spectrum-simulation benchmark subset membership; never used as provenance or a filter"
        ):
            raise RuntimeError(f"pool has obsolete membership semantics: {pool}")
        pool_adduct = str(report.get("adduct"))
        expected = allowed_rows[adduct[allowed_rows] == pool_adduct]
        if int(report.get("selected_spectra", -1)) != len(expected):
            raise RuntimeError(f"pool selected_spectra is not membership-blind: {pool}")
        expected_membership = {
            str(value): int(np.sum(membership[expected] == value)) for value in np.unique(membership[expected])
        }
        if report.get("simulation_challenge_membership_counts") != expected_membership:
            raise RuntimeError(f"pool membership counts differ from independent reconstruction: {pool}")
        covered_rows.append(expected)
        pool_reports.append({
            "path": str(pool), "sha256": sha256(pool), "adduct": pool_adduct,
            "selected_spectra": len(expected), "membership_counts": expected_membership,
        })
    exact_set("pool selected-row union", np.concatenate(covered_rows), allowed_rows)

    manifest_report = json.loads(args.manifest.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest_report.get("simulation_challenge_semantics") != (
        "spectrum-simulation benchmark subset membership; not provenance; not filtered"
    ):
        raise RuntimeError("candidate manifest has obsolete membership semantics")
    with np.load(args.manifest) as manifest:
        query_rows = manifest["query_row"].astype(np.int64)
        query_membership = manifest["query_simulation_challenge_membership"].astype(str)
        reachable = np.unique(np.r_[query_rows, manifest["pair_candidate_row"].astype(np.int64)])
    if not np.array_equal(query_membership, membership[query_rows]):
        raise RuntimeError("manifest membership annotations do not match HDF5")
    if not np.all(np.isin(reachable, allowed_rows)):
        raise RuntimeError("manifest reaches rows outside corrected training set")
    if set(np.unique(query_membership)) != {"False", "True"}:
        raise RuntimeError("manifest lost one membership stratum")
    manifest_counts = {str(value): int(np.sum(query_membership == value)) for value in np.unique(query_membership)}
    if manifest_report.get("counts", {}).get("membership") != manifest_counts:
        raise RuntimeError("manifest report membership counts are stale")

    server_seal_verified = bool(allow_report.get("byte_identity_to_server_seal_claimed", False))
    result = {
        "status": "CHEMAWARE_DATA_SEMANTICS_PASS",
        "development_training_admissible": True,
        "release_eligible": server_seal_verified,
        "schema_contract": {
            "simulation_challenge_semantics": "spectrum-simulation benchmark membership; never provenance",
            "membership_used_to_select_rows": False,
            "membership_blind_row_reconstruction_exact": True,
            "required_hdf5_fields_present": True,
            "allowed_rows_train_only": True,
            "allowed_rows_adducts": ["[M+H]+", "[M+Na]+"],
        },
        "counts": {
            "hdf5_rows": total,
            "hdf5_membership": {str(v): int(np.sum(membership == v)) for v in np.unique(membership)},
            "allowed_identities": len(identities), "allowed_rows": len(allowed_rows),
            "allowed_membership": {str(v): int(np.sum(membership[allowed_rows] == v)) for v in np.unique(membership[allowed_rows])},
            "manifest_queries": len(query_rows), "manifest_reachable_rows": len(reachable),
            "manifest_query_membership": manifest_counts,
        },
        "pool_reports": pool_reports,
        "provenance": {
            "hdf5_sha256": sha256(args.data), "allow_sha256": sha256(args.allow),
            "allow_report_sha256": sha256(args.allow_report), "manifest_sha256": sha256(args.manifest),
            "manifest_report_sha256": sha256(args.manifest.with_suffix(".json")),
            "script_sha256": sha256(Path(__file__)),
        },
        "limitations": {
            "server_seal_verified": server_seal_verified,
            "local_allow_report_formal": bool(allow_report.get("formal", False)),
            "claim": "Locally exact semantic and row-set validation; release remains blocked without the server seal.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

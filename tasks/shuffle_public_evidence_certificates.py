#!/usr/bin/env python
"""Create matched nulls that break query-to-public-unit correspondence.

The null keeps every query's assignment/abstention state and certificate score
fixed.  Within frozen query-covariate strata, it permutes the assigned unit
labels among certified queries.  This preserves coverage and confidence while
destroying the proposed public-unit meaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CERTIFICATE_REQUIRED = {"query_id", "assigned_public_unit", "certificate_score"}
FORBIDDEN = ("truth", "label", "phenotype", "disease", "drug", "case", "control", "outcome")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--query-strata", type=Path, required=True,
                        help="CSV: query_id plus frozen technical matching strata only.")
    parser.add_argument("--strata", required=True,
                        help="Comma-separated columns, e.g. polarity,mass_bin,peak_count_bin,instrument.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    certificates = pd.read_csv(args.certificates, dtype={"query_id": str})
    strata = pd.read_csv(args.query_strata, dtype={"query_id": str})
    missing = CERTIFICATE_REQUIRED - set(certificates.columns)
    if missing:
        raise ValueError(f"certificates missing columns: {sorted(missing)}")
    if certificates["query_id"].duplicated().any() or strata["query_id"].duplicated().any():
        raise ValueError("certificate and stratum tables require unique query_id")
    if set(certificates["query_id"]) != set(strata["query_id"]):
        raise ValueError("certificate and stratum query universes differ")
    contaminated = [column for column in strata.columns
                    if any(token in column.lower() for token in FORBIDDEN)]
    if contaminated:
        raise ValueError(f"query strata contain forbidden outcome columns: {contaminated}")
    stratum_columns = [value.strip() for value in args.strata.split(",") if value.strip()]
    if not stratum_columns or set(stratum_columns) - set(strata.columns):
        raise ValueError("all requested strata columns must exist")

    merged = certificates.merge(strata[["query_id", *stratum_columns]], on="query_id",
                                how="left", validate="one_to_one")
    if merged[stratum_columns].isna().any().any():
        raise ValueError("matching strata cannot be missing")
    assigned = merged["assigned_public_unit"].ne("__NO_MATCH__")
    rng = np.random.default_rng(args.seed)
    outputs: list[pd.DataFrame] = []
    unchanged_by_repeat: list[int] = []
    singleton_strata = 0
    grouping = stratum_columns[0] if len(stratum_columns) == 1 else stratum_columns
    groups = list(merged.loc[assigned].groupby(grouping, sort=True, dropna=False).groups.values())
    singleton_strata = int(sum(len(indices) < 2 for indices in groups))
    for repeat in range(args.repeats):
        frame = certificates.copy()
        frame["null_repeat"] = repeat
        unchanged = 0
        for indices in groups:
            locations = np.asarray(list(indices), dtype=int)
            original = merged.loc[locations, "assigned_public_unit"].to_numpy(copy=True)
            permuted = rng.permutation(original) if len(locations) >= 2 else original
            frame.loc[locations, "assigned_public_unit"] = permuted
            unchanged += int(np.sum(original == permuted))
        outputs.append(frame)
        unchanged_by_repeat.append(unchanged)

    result = pd.concat(outputs, ignore_index=True)
    args.output.mkdir(parents=True)
    result.to_csv(args.output / "shuffled_certificates.csv.gz", index=False, compression="gzip")
    result.loc[result["null_repeat"] == 0].drop(columns="null_repeat").to_csv(
        args.output / "primary_null_certificates.csv", index=False)
    report = {
        "status": "PUBLIC_EVIDENCE_MATCHED_CERTIFICATE_NULL_COMPLETE",
        "formal": False,
        "queries": int(len(certificates)),
        "assigned_queries": int(assigned.sum()),
        "repeats": args.repeats,
        "seed": args.seed,
        "strata": stratum_columns,
        "singleton_assigned_strata": singleton_strata,
        "mean_unchanged_assigned_queries": float(np.mean(unchanged_by_repeat)),
        "contracts": {
            "assignment_coverage_preserved_per_query": True,
            "certificate_score_preserved_per_query": True,
            "dustbin_preserved_per_query": True,
            "unit_frequency_preserved_within_stratum": True,
            "outcome_blind": True,
        },
        "provenance": {
            "certificates_sha256": sha256(args.certificates),
            "query_strata_sha256": sha256(args.query_strata),
        },
        "claim_limit": "Matched null certificates only; not public evidence or biological inference.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build label-blind, abstaining certificates from public spectral evidence.

The input is a long table of query-to-public-unit evidence.  Biological labels,
structure truth and candidate truth are forbidden.  A unit is returned only
when it is supported by multiple evidence families and independent public
sources; otherwise the query is assigned to an explicit dustbin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"query_id", "public_spectrum_id", "public_unit_id", "public_source_id",
            "public_instrument", "evidence_family", "score"}
FORBIDDEN = ("truth", "label", "phenotype", "disease", "drug", "case", "control", "smiles", "inchikey")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True,
                   help="Frozen JSON mapping evidence_family to minimum score.")
    p.add_argument("--query-universe", type=Path,
                   help="Optional CSV with every query_id, including queries with no public edge.")
    p.add_argument("--excluded-public-sources", type=Path,
                   help=("Optional CSV of forbidden public_source_id and/or public_raw_hash. "
                         "Formal applications must supply this leakage-exclusion ledger."))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-families", type=int, default=2)
    p.add_argument("--minimum-sources", type=int, default=2)
    p.add_argument("--maximum-source-dominance", type=float, default=0.8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    edges = pd.read_csv(a.edges)
    missing = REQUIRED - set(edges.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    contaminated = [c for c in edges.columns if any(token in c.lower() for token in FORBIDDEN)]
    if contaminated:
        raise ValueError(f"outcome/identity columns forbidden: {contaminated}")
    if a.query_universe is not None:
        universe = pd.read_csv(a.query_universe)
        if set(universe.columns) != {"query_id"}:
            raise ValueError("query-universe must contain only query_id")
        if universe["query_id"].duplicated().any() or universe["query_id"].isna().any():
            raise ValueError("query-universe query_id values must be unique and nonmissing")
        queries = sorted(universe["query_id"].astype(str))
        edges["query_id"] = edges["query_id"].astype(str)
        unexpected = set(edges["query_id"]) - set(queries)
        if unexpected:
            raise ValueError(f"edges contain queries outside frozen universe: {sorted(unexpected)[:5]}")
    else:
        queries = sorted(edges["query_id"].astype(str).unique())
        edges["query_id"] = edges["query_id"].astype(str)
    exclusion_sha = None
    if a.excluded_public_sources is not None:
        exclusions = pd.read_csv(a.excluded_public_sources, dtype=str)
        permitted = {"public_source_id", "public_raw_hash"}
        if not set(exclusions.columns) or set(exclusions.columns) - permitted:
            raise ValueError("excluded-public-sources accepts public_source_id/public_raw_hash only")
        if "public_source_id" in exclusions:
            overlap = set(edges["public_source_id"].astype(str)) & set(
                exclusions["public_source_id"].dropna().astype(str))
            if overlap:
                raise ValueError(f"excluded public sources are present in edges: {sorted(overlap)[:5]}")
        if "public_raw_hash" in exclusions:
            if "public_raw_hash" not in edges:
                raise ValueError("raw-hash exclusions supplied but edges lack public_raw_hash")
            overlap = set(edges["public_raw_hash"].dropna().astype(str)) & set(
                exclusions["public_raw_hash"].dropna().astype(str))
            if overlap:
                raise ValueError(f"excluded public raw hashes are present in edges: {sorted(overlap)[:5]}")
        exclusion_sha = sha256(a.excluded_public_sources)
    thresholds = json.loads(a.thresholds.read_text(encoding="utf-8"))
    families = set(edges["evidence_family"].astype(str))
    if families - set(thresholds):
        raise ValueError(f"unfrozen evidence families: {sorted(families-set(thresholds))}")
    edges["score"] = pd.to_numeric(edges["score"], errors="raise")
    edges["threshold"] = edges["evidence_family"].map(thresholds).astype(float)
    edges["active"] = edges["score"].ge(edges["threshold"])
    edges["excess"] = np.maximum(0.0, (edges["score"] - edges["threshold"]) /
                                 np.maximum(1e-12, 1.0 - edges["threshold"]))
    unit_rows = []
    for (query, unit), frame in edges.groupby(["query_id", "public_unit_id"], sort=True):
        active = frame.loc[frame["active"]]
        independent_spectra = active[["public_source_id", "public_spectrum_id"]].drop_duplicates()
        counts = independent_spectra["public_source_id"].value_counts()
        dominance = float(counts.iloc[0] / len(independent_spectra)) if len(independent_spectra) else 1.0
        unit_rows.append({
            "query_id": query, "public_unit_id": unit,
            "evidence_families": int(active["evidence_family"].nunique()),
            "public_sources": int(active["public_source_id"].nunique()),
            "public_instruments": int(active["public_instrument"].nunique()),
            "public_spectra": int(active["public_spectrum_id"].nunique()),
            "source_dominance": dominance,
            "certificate_score": float(active.groupby("evidence_family")["excess"].max().mean()) if len(active) else 0.0,
        })
    units = pd.DataFrame(unit_rows)
    units["eligible"] = ((units["evidence_families"] >= a.minimum_families) &
                         (units["public_sources"] >= a.minimum_sources) &
                         (units["source_dominance"] <= a.maximum_source_dominance))
    certificates = []
    for query in queries:
        options = units.loc[(units["query_id"] == query) & units["eligible"]].sort_values(
            ["certificate_score", "public_sources", "evidence_families", "public_unit_id"],
            ascending=[False, False, False, True])
        tie = len(options) > 1 and abs(options.iloc[0]["certificate_score"] - options.iloc[1]["certificate_score"]) < 1e-12
        winner = None if options.empty or tie else options.iloc[0]
        certificates.append({"query_id": query,
            "assigned_public_unit": "__NO_MATCH__" if winner is None else winner["public_unit_id"],
            "certificate_score": 0.0 if winner is None else float(winner["certificate_score"]),
            "eligible_units": int(len(options)), "abstention_reason":
            "no_eligible_unit" if options.empty else ("top_tie" if tie else "")})
    a.output.mkdir(parents=True)
    units.to_csv(a.output / "unit_evidence.csv.gz", index=False, compression="gzip")
    pd.DataFrame(certificates).to_csv(a.output / "certificates.csv", index=False)
    report = {"status": "PUBLIC_SPECTRAL_EVIDENCE_CERTIFICATES_COMPLETE", "formal": False,
              "queries": len(certificates), "assigned": sum(x["assigned_public_unit"] != "__NO_MATCH__" for x in certificates),
              "contracts": {"outcome_blind": True, "structure_truth_blind": True, "explicit_dustbin": True,
                            "multi_family_required": a.minimum_families, "multi_source_required": a.minimum_sources},
              "leakage_exclusion_ledger_supplied": a.excluded_public_sources is not None,
              "query_universe_supplied": a.query_universe is not None,
              "provenance": {"edges_sha256": sha256(a.edges), "thresholds_sha256": sha256(a.thresholds),
                             "query_universe_sha256": sha256(a.query_universe) if a.query_universe else None,
                             "excluded_public_sources_sha256": exclusion_sha},
              "claim_limit": "Evidence-unit assignment only; not molecular identity, reaction, function, or biological association."}
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

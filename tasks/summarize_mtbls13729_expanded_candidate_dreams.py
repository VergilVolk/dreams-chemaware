#!/usr/bin/env python
"""Summarize mass-constrained DreaMS candidates across expanded raw-MS2 links.

The input annotation tables are query-level library searches.  This audit joins
only precursor/RT-linked spectra, requires the annotation's exact-mass gate,
and then asks whether one candidate identity repeats across spectra and samples.
Consensus is annotation evidence, not identity truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LINKS = ROOT / "data/mtbls13729/expanded_ms2_links_v1/candidate_ms2_links.csv.gz"
COVERAGE = ROOT / "data/mtbls13729/expanded_ms2_links_v1/candidate_ms2_coverage.csv"
ANNOTATION = ROOT / "data/mtbls13729/annotation_native"
OUT = ROOT / "data/mtbls13729/expanded_candidate_dreams_consensus_v1"
SCAN_RE = re.compile(r"scan=(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if (OUT / "report.json").exists():
        raise RuntimeError(f"refusing to overwrite completed output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(LINKS)
    links["scan_number"] = links.native_id.astype(str).str.extract(SCAN_RE, expand=False).astype(int)
    # A scan can appear more than once for one feature only if the raw-link
    # builder emitted a duplicate.  Fail rather than let duplicate votes count.
    vote_key = ["panel", "feature_id", "sample_name", "scan_number"]
    if links.duplicated(vote_key).any():
        raise RuntimeError("duplicate feature/sample/scan links would inflate consensus votes")

    vote_cache = OUT / "mass_constrained_top1_votes.csv.gz"
    file_cache = OUT / "file_audit.csv"
    selected_parts: list[pd.DataFrame] = []
    file_audit: list[dict[str, object]] = []
    usecols = [
        "scan_number", "file_name", "ref_INCHIKEY", "ref_name", "ref_smiles",
        "topk", "DreaMS_similarity", "mz_pass", "dppm", "analog_hit",
    ]
    if vote_cache.exists() and file_cache.exists():
        votes = pd.read_csv(vote_cache)
        file_audit = pd.read_csv(file_cache).to_dict("records")
    else:
      for (panel, sample), subset in links.groupby(["panel", "sample_name"], sort=True):
          path = ANNOTATION / str(panel) / "per_file" / f"{sample}.tsv"
          if not path.exists():
              file_audit.append({"panel": panel, "sample_name": sample, "status": "missing_annotation"})
              continue
          scans = set(subset.scan_number.astype(int))
          annotation = pd.read_csv(path, sep="\t", usecols=usecols)
          annotation = annotation[annotation.scan_number.astype(int).isin(scans)].copy()
          annotation["mz_pass"] = annotation.mz_pass.astype(str).str.lower().isin({"true", "1"})
          annotation = annotation[annotation.mz_pass].copy()
          if annotation.empty:
              file_audit.append({"panel": panel, "sample_name": sample, "status": "complete", "linked_scans": len(scans), "mass_candidates": 0})
              continue
          # One vote per linked query spectrum: the highest-similarity candidate
          # among candidates that passed the upstream exact-mass filter.
          annotation = annotation.sort_values(
              ["scan_number", "DreaMS_similarity", "topk", "ref_INCHIKEY"],
              ascending=[True, False, True, True], kind="stable",
          ).groupby("scan_number", sort=False, as_index=False).head(1)
          annotation = annotation.merge(
              subset[["panel", "feature_id", "sample_name", "scan_number"]],
              on="scan_number", how="inner", validate="one_to_one",
          )
          selected_parts.append(annotation)
          file_audit.append({
              "panel": panel, "sample_name": sample, "status": "complete",
              "linked_scans": len(scans), "mass_candidates": int(len(annotation)),
          })
      votes = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    coverage = pd.read_csv(COVERAGE)
    if votes.empty:
        raise RuntimeError("no mass-constrained DreaMS votes were recovered")
    votes.to_csv(vote_cache, index=False)
    candidate = (
        votes.groupby(["panel", "feature_id", "ref_INCHIKEY", "ref_name", "ref_smiles"], dropna=False, as_index=False)
        .agg(
            supporting_spectra=("scan_number", "size"),
            supporting_samples=("sample_name", "nunique"),
            median_dreams_similarity=("DreaMS_similarity", "median"),
            maximum_dreams_similarity=("DreaMS_similarity", "max"),
        )
    )
    candidate = candidate.sort_values(
        ["panel", "feature_id", "supporting_samples", "supporting_spectra", "median_dreams_similarity", "maximum_dreams_similarity", "ref_INCHIKEY"],
        ascending=[True, True, False, False, False, False, True], kind="stable",
    )
    candidate["candidate_rank"] = candidate.groupby(["panel", "feature_id"]).cumcount() + 1
    candidate.to_csv(OUT / "all_consensus_candidates.csv.gz", index=False)
    best = candidate[candidate.candidate_rank.eq(1)].copy()
    vote_totals = votes.groupby(["panel", "feature_id"]).size().rename("mass_constrained_vote_spectra")
    best = best.merge(vote_totals, on=["panel", "feature_id"], how="left", validate="one_to_one")
    best["agreement_fraction"] = best.supporting_spectra / best.mass_constrained_vote_spectra
    best["dreams_consensus_evidence_tier"] = np.select(
        [
            best.maximum_dreams_similarity.ge(0.80) & best.supporting_samples.ge(2) & best.agreement_fraction.ge(0.60),
            best.maximum_dreams_similarity.ge(0.70) & best.supporting_samples.ge(2),
            best.maximum_dreams_similarity.ge(0.50),
        ],
        ["Level 2a-supported", "Level 2a-single/ambiguous", "Level 3-candidate"],
        default="unassigned",
    )
    result = coverage.merge(best, on=["panel", "feature_id"], how="left", validate="one_to_one")
    result.to_csv(OUT / "expanded_candidate_dreams_consensus.csv", index=False)
    pd.DataFrame(file_audit).to_csv(OUT / "file_audit.csv", index=False)
    report = {
        "status": "mtbls13729_expanded_candidate_dreams_consensus_complete",
        "formal": False,
        "candidates": int(len(result)),
        "candidates_with_raw_ms2": int(result.n_ms2_spectra.gt(0).sum()),
        "candidates_with_mass_constrained_vote": int(result.ref_INCHIKEY.notna().sum()),
        "level2a_supported": int(result.dreams_consensus_evidence_tier.eq("Level 2a-supported").sum()),
        "level2a_single_or_ambiguous": int(result.dreams_consensus_evidence_tier.eq("Level 2a-single/ambiguous").sum()),
        "level3_candidate": int(result.dreams_consensus_evidence_tier.eq("Level 3-candidate").sum()),
        "files": pd.Series([row["status"] for row in file_audit]).value_counts().to_dict(),
        "provenance": {"links_sha256": sha256(LINKS), "coverage_sha256": sha256(COVERAGE)},
        "claim_limit": "Consensus library candidates do not establish identity without compatible reference-spectrum and/or authentic-standard evidence.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

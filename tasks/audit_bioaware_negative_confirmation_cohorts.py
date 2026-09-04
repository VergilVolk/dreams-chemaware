#!/usr/bin/env python
"""Audit local cohorts for frozen BioAware-negative confirmation readiness.

The audit is deliberately conservative.  A cohort is a performance-confirmation
cohort only when the local files contain negative-mode MS/MS, independent
structure-resolved truth, enough truth identities, reconstructable candidate
search, sample context, and no exposure to the v2 development procedure.
Processed abundance tables and phenotype analyses are not substitutes for an
MS/MS identity benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


MINIMUM_TRUTH_IDENTITIES = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_files(root: Path, suffixes: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    wanted = tuple(value.lower() for value in suffixes)
    return sum(
        1 for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith(wanted)
    )


def mzml_polarity(path: Path, read_bytes: int = 2_000_000) -> str:
    """Return an observed mzML polarity without decoding binary arrays."""
    with path.open("rb") as handle:
        text = handle.read(read_bytes).decode("utf-8", errors="ignore")
    has_positive = 'name="positive scan"' in text
    has_negative = 'name="negative scan"' in text
    if has_positive and not has_negative:
        return "positive"
    if has_negative and not has_positive:
        return "negative"
    if has_positive and has_negative:
        return "mixed"
    return "unknown"


def maf_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0}
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    structure = pd.Series(False, index=table.index)
    for column in ("smiles", "inchi"):
        if column in table:
            structure |= table[column].str.strip().ne("")
    negative = pd.Series(False, index=table.index)
    for column in ("modifications", "charge", "ion_type", "adduct"):
        if column in table:
            negative |= table[column].str.contains(
                r"\[M-H\]-|negative|\bneg\b", case=False, regex=True, na=False
            )
    reliability = table.get("reliability", pd.Series("", index=table.index))
    level1 = reliability.str.contains(r"MSI\s*:?\s*1|Level\s*1", case=False, regex=True)
    return {
        "exists": True,
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(table)),
        "rows_with_structure": int(structure.sum()),
        "negative_like_rows": int(negative.sum()),
        "declared_level1_rows": int(level1.sum()),
        "unique_database_identifiers": int(
            table.get("database_identifier", pd.Series("", index=table.index))
            .str.strip().replace("", pd.NA).nunique()
        ),
    }


def assess_candidate(facts: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "negative_ms2_locally_available": bool(facts["negative_ms2_locally_available"]),
        "independent_structure_truth": bool(facts["independent_structure_truth"]),
        "truth_identities_ge_minimum": int(facts["truth_identities"]) >= MINIMUM_TRUTH_IDENTITIES,
        "candidate_search_reconstructable": bool(facts["candidate_search_reconstructable"]),
        "sample_network_context_available": bool(facts["sample_network_context_available"]),
        "not_used_for_v2_development": not bool(facts["used_for_v2_development"]),
    }
    return {
        **facts,
        "minimum_truth_identities": MINIMUM_TRUTH_IDENTITIES,
        "gates": gates,
        "ready_for_frozen_performance_confirmation": all(gates.values()),
    }


def mtbls1905(root: Path) -> dict[str, Any]:
    negative_maf = maf_summary(root / "metadata" / "negative.maf.tsv")
    qc_files = sorted((root / "qc_ms2").glob("*.mzML")) if (root / "qc_ms2").exists() else []
    polarities = {path.name: mzml_polarity(path) for path in qc_files}
    negative_ms2 = [name for name, polarity in polarities.items() if polarity in {"negative", "mixed"}]
    truths = int(negative_maf.get("declared_level1_rows", 0))
    return assess_candidate({
        "cohort": "MTBLS1905 local negative arm",
        "local_root": str(root),
        "negative_ms2_locally_available": bool(negative_ms2),
        "independent_structure_truth": truths > 0 and negative_maf.get("rows_with_structure", 0) >= truths,
        "truth_identities": truths,
        "candidate_search_reconstructable": bool(negative_ms2) and truths > 0,
        "sample_network_context_available": negative_maf.get("rows", 0) > 0,
        "used_for_v2_development": False,
        "raw_mzml_files": count_files(root, (".mzml",)),
        "qc_ms2_polarities": polarities,
        "negative_maf": negative_maf,
        "decision": "BLOCKED: local QC-MS2 is positive-mode only and the negative MSI:1 panel contains only three identities.",
        "salvage": "Obtain the deposited negative-mode QC-MS2 files and their acquisition mapping; even then treat the three identities as a tiny qualitative challenge, not a 3-4 pp confirmation benchmark.",
    })


def mtbls8090(root: Path) -> dict[str, Any]:
    reverse = maf_summary(root / "reverse_phase.maf.tsv")
    hilic = maf_summary(root / "hilic.maf.tsv")
    return assess_candidate({
        "cohort": "MTBLS8090 processed MAF",
        "local_root": str(root),
        "negative_ms2_locally_available": False,
        "independent_structure_truth": False,
        "truth_identities": 0,
        "candidate_search_reconstructable": False,
        "sample_network_context_available": True,
        "used_for_v2_development": False,
        "raw_ms_files": count_files(root, (".mzml", ".mzxml", ".mgf", ".msp")),
        "reverse_phase_maf": reverse,
        "hilic_maf": hilic,
        "decision": "BLOCKED: only processed MAF tables are local; reliability/Level-1 evidence is absent and no raw MS/MS can be replayed.",
        "salvage": "Retrieve raw negative-mode files plus the study's standard-confirmed identity table before considering a frozen performance test.",
    })


def mtbls13432(root: Path) -> dict[str, Any]:
    workbook = root / "author_repo" / "data" / "lcms" / "Supp_Tables_S1 to S4_v3.xlsx"
    manual_rows = 0
    level1_rows = 0
    if workbook.exists():
        table = pd.read_excel(workbook, sheet_name="Table_S3 Manually_annotated_PMs")
        manual_rows = int(len(table))
        loa = pd.to_numeric(table.get("LoA*", pd.Series(dtype=float)), errors="coerce")
        level1_rows = int((loa == 1).sum())
    return assess_candidate({
        "cohort": "MTBLS13432 SGME-HCC author snapshot",
        "local_root": str(root),
        "negative_ms2_locally_available": False,
        "independent_structure_truth": level1_rows > 0,
        "truth_identities": level1_rows,
        "candidate_search_reconstructable": False,
        "sample_network_context_available": True,
        "used_for_v2_development": False,
        "raw_ms_files": count_files(root, (".mzml", ".mzxml", ".mgf", ".msp")),
        "manual_annotation_rows": manual_rows,
        "level_of_annotation_1_rows": level1_rows,
        "workbook": str(workbook),
        "workbook_sha256": sha256(workbook) if workbook.exists() else None,
        "decision": "BLOCKED: the local author snapshot has no raw MS/MS and its manually annotated panel contains no LoA=1 entries.",
        "salvage": "Acquire the deposited LC-MS/MS data and an explicit standard-confirmed subset; LoA 2/3 annotations cannot serve as identity truth.",
    })


def oep00006137(root: Path) -> dict[str, Any]:
    negative_archives = count_files(root / "hilic_neg", (".mzxml.tar.gz",)) + count_files(
        root / "rp_neg", (".mzxml.tar.gz",)
    )
    return assess_candidate({
        "cohort": "OEP00006137 tissue raw deposit",
        "local_root": str(root),
        "negative_ms2_locally_available": False,
        "independent_structure_truth": False,
        "truth_identities": 0,
        "candidate_search_reconstructable": False,
        "sample_network_context_available": negative_archives > 0,
        "used_for_v2_development": False,
        "negative_raw_archives": negative_archives,
        "deposited_qc_ms2_scans": 0,
        "decision": "BLOCKED: negative-mode sample files are local, but deposited QC ddMS2 scans are absent; reported Level-1 coordinates cannot be re-tested as MS/MS identities.",
        "salvage": "Use only for phenotype-blind abundance/network deployment; do not count it as a retrieval-performance confirmation cohort.",
    })


def mtbls13729(root: Path) -> dict[str, Any]:
    mzml = count_files(root / "mzml", (".mzml",))
    return assess_candidate({
        "cohort": "MTBLS13729 biological application",
        "local_root": str(root),
        "negative_ms2_locally_available": mzml > 0,
        "independent_structure_truth": False,
        "truth_identities": 0,
        "candidate_search_reconstructable": False,
        "sample_network_context_available": mzml > 0,
        "used_for_v2_development": False,
        "raw_mzml_files": mzml,
        "decision": "DEPLOYMENT ONLY: raw MS1/MS2 and paired biology are available, but paper annotations lack structure-resolved confirmation truth.",
        "salvage": "Apply the already frozen expert without outcome tuning and report abstention, annotation yield, paths and biological coherence; never present this as performance confirmation.",
    })


def opened_development(root: Path) -> dict[str, Any]:
    return assess_candidate({
        "cohort": "MetDNA3 four-source negative development spectra",
        "local_root": str(root),
        "negative_ms2_locally_available": root.exists(),
        "independent_structure_truth": root.exists(),
        "truth_identities": 164 if root.exists() else 0,
        "candidate_search_reconstructable": root.exists(),
        "sample_network_context_available": root.exists(),
        "used_for_v2_development": True,
        "decision": "INELIGIBLE: this is the opened 548-query development protocol used to choose and audit the v2 expert.",
        "salvage": "Retain for mechanism and reproducibility only; never call another score on it independent confirmation.",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_negative_confirmation_cohort_readiness_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")

    candidates = [
        mtbls1905(Path("data/external/MTBLS1905")),
        mtbls8090(Path("data/external/MTBLS8090")),
        mtbls13432(Path("data/external/mtbls13432_sgme_hcc")),
        oep00006137(Path("data/external/OEP00006137_raw")),
        mtbls13729(Path("data/mtbls13729")),
        opened_development(Path("data/external/metdna3_2025")),
    ]
    ready = [item["cohort"] for item in candidates if item["ready_for_frozen_performance_confirmation"]]
    report = {
        "status": "bioaware_negative_confirmation_cohort_readiness_complete",
        "formal": True,
        "minimum_truth_identities": MINIMUM_TRUTH_IDENTITIES,
        "qualification_contract": (
            "locally available negative MS/MS + independent structure truth + at least 20 truth identities + "
            "reconstructable candidate search + sample network context + no v2 development exposure"
        ),
        "candidates": candidates,
        "ready_cohorts": ready,
        "ready_count": len(ready),
        "decision": (
            "LOCAL_CONFIRMATION_READY" if ready else
            "NO_LOCAL_COHORT_QUALIFIES; freeze v2 and acquire a new structure-confirmed negative-MS/MS cohort"
        ),
        "next_data_request": {
            "required": not bool(ready),
            "minimum": {
                "negative_mode_centroided_or_profile_msms": True,
                "truth_identity_fields": ["SMILES or InChI", "full InChIKey", "formula", "adduct"],
                "truth_standard": "MSI Level 1 or explicit authentic-standard confirmation",
                "minimum_unique_truth_identities": MINIMUM_TRUTH_IDENTITIES,
                "sample_membership_for_network_seeds": True,
                "candidate_library_or_reconstructable_mass_search": True,
            },
            "preferred": {
                "unique_truth_identities": 100,
                "multiple_biological_sources": True,
                "raw_ms1_and_ms2": True,
                "collision_energy_and_instrument_metadata": True,
            },
        },
        "claim_limit": (
            "This audit establishes local data readiness only. It does not evaluate BioAware performance, "
            "and processed annotations or abundance re-extraction do not substitute for independent MS/MS truth."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

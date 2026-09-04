#!/usr/bin/env python3
"""Fail-closed reproducibility audit for the BioAware network backbone.

This script does not run annotation and does not compare performance.  It only
answers which mature network algorithm is actually runnable from the assets on
disk.  That distinction prevents the public MetDNA3 core module from being
mistaken for the complete MetDNA3 workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


METDNA3_CODE = (
    "R/MRN3main.R",
    "R/MRN3annotation.R",
    "R/MRN3rmRedun.R",
    "R/MRN3preparation.R",
)

METDNA3_PRIVATE_OR_UNPUBLISHED_ASSETS = (
    "obj_mrn.rda",
    "info_mrn.rda",
    "md_mrn.rda",
    "obj_mrn_3x1.rda",
    "obj_mrn_3x2.rda",
    "info_mrn_3x.rda",
    "md_mrn_3x.rda",
)

METDNA2_PUBLIC_ASSETS = (
    "data/reaction_pair_network.rda",
    "data/md_mrn_emrn.rda",
    "data/lib_adduct_nl.rda",
    "data/lib_formula.rda",
    "data/lib_kegg.rda",
)

METDNA2_CODE = (
    "R/MetDNA2.R",
    "R/AnnotationCredential.R",
    "R/AnnotationCredentialFormula.R",
    "R/AnnotationCredentialPeakGroup.R",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inventory(root: Path, relative_paths: tuple[str, ...]) -> dict[str, Any]:
    present: list[str] = []
    missing: list[str] = []
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_file() and path.stat().st_size > 0:
            present.append(relative)
            hashes[relative] = sha256(path)
        else:
            missing.append(relative)
    return {
        "root": str(root.resolve()),
        "present": present,
        "missing": missing,
        "sha256": hashes,
        "pass": not missing,
    }


def author_workdir_inventory(workdir: Path | None) -> dict[str, Any]:
    required = (
        "02_result_MRN_annotation/table_ms2_edges.rda",
        "02_result_MRN_annotation/ms2_data.rda",
    )
    if workdir is None:
        return {
            "provided": False,
            "present": [],
            "missing": list(required),
            "pass": False,
        }
    result = inventory(workdir, required)
    result["provided"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metdna3", type=Path, default=Path("third_party/MrnAnnoAlgo3"))
    parser.add_argument("--metdna2", type=Path, default=Path("third_party/MetDNA2"))
    parser.add_argument("--metdna3-workdir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/metabolic_network_framework_reproducibility.json"),
    )
    args = parser.parse_args()

    metdna3_code = inventory(args.metdna3, METDNA3_CODE)
    metdna3_assets = inventory(args.metdna3, METDNA3_PRIVATE_OR_UNPUBLISHED_ASSETS)
    metdna3_workdir = author_workdir_inventory(args.metdna3_workdir)
    metdna2_code = inventory(args.metdna2, METDNA2_CODE)
    metdna2_assets = inventory(args.metdna2, METDNA2_PUBLIC_ASSETS)

    exact_metdna3_available = bool(
        metdna3_code["pass"]
        and (metdna3_assets["pass"] or metdna3_workdir["pass"])
    )
    metdna3_core_only = bool(metdna3_code["pass"] and not exact_metdna3_available)
    kgmn_metdna2_ready = bool(metdna2_code["pass"] and metdna2_assets["pass"])

    if exact_metdna3_available:
        decision = "exact_metdna3"
        next_step = (
            "Freeze an untouched author baseline, then score the exact author edge set "
            "with the four preregistered edge-reliability arms."
        )
    elif kgmn_metdna2_ready:
        decision = "kgmn_metdna2_reproducible_fallback"
        next_step = (
            "Run the public KGMN/MetDNA2 workflow as the mature reproducible baseline. "
            "Keep the MetDNA3 core-only branch blocked until official MRN assets or an "
            "author workdir are supplied."
        )
    else:
        decision = "blocked_no_complete_framework"
        next_step = "Acquire a complete official framework before any improvement claim."

    report = {
        "status": "metabolic_network_framework_reproducibility_audit_complete",
        "formal": True,
        "metdna3": {
            "source_commit": git_commit(args.metdna3),
            "public_core_code": metdna3_code,
            "complete_mrn_assets": metdna3_assets,
            "author_workdir": metdna3_workdir,
            "exact_reproduction_available": exact_metdna3_available,
            "core_only": metdna3_core_only,
            "claim_boundary": (
                "Public core code alone is not a complete MetDNA3 reproduction."
            ),
        },
        "kgmn_metdna2": {
            "source_commit": git_commit(args.metdna2),
            "public_code": metdna2_code,
            "public_network_assets": metdna2_assets,
            "reproducible_baseline_ready": kgmn_metdna2_ready,
        },
        "decision": decision,
        "next_step": next_step,
        "hard_stops": {
            "no_edge_calibration_before_author_baseline": True,
            "no_metdna3_claim_from_core_only": True,
            "no_raw_dreams_cosine_as_probability": True,
            "no_network_neighbor_as_identity_positive": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

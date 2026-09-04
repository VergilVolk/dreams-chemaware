#!/usr/bin/env python
"""Seal the untouched NetID external-validation contract without reading truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from fetch_bioaware_netid_external import REQUIRED_SUFFIXES, _locate_required


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/external/netid_v1/source")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_netid_external_contract_v1"),
    )
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    source_manifest_path = source / "bioaware_netid_source_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "bioaware_netid_source_installed":
        raise RuntimeError("NetID source is not sealed")
    located = _locate_required(source)
    for suffix in REQUIRED_SUFFIXES:
        expected = source_manifest["required_files"][suffix]["sha256"]
        observed = sha256(located[suffix])
        if expected != observed:
            raise RuntimeError(f"source file changed after installation: {suffix}")

    manual = located["FDR_example/manual_curate.csv"]
    report = {
        "status": "bioaware_netid_external_contract_frozen",
        "formal": True,
        "dataset": "NetID v1.0 / yeast negative-mode external benchmark",
        "source_record": "10.5281/zenodo.5508337",
        "source_manifest_sha256": sha256(source_manifest_path),
        "held_out_outcome": {
            "relative_path": manual.relative_to(source).as_posix(),
            "bytes": int(manual.stat().st_size),
            "sha256": sha256(manual),
            "content_read_during_lock": False,
        },
        "development_is_frozen": True,
        "fixed_method": {
            "spectral_baseline": "DreaMS-only candidate ranking",
            "network_feature": "dependency-corrected one-hop Rhea support",
            "dependency_rule": (
                "complete paths grouped by independent seed compound; incomplete "
                "paths sharing the same missing source-side signature counted once"
            ),
            "query_exclusion": "leave-query-out and leave-truth-identity-out",
            "phenotype_blind": True,
            "abstention": "network evidence may override only a low-margin spectral Top-1",
        },
        "construction_inputs": {
            suffix: {
                "relative_path": path.relative_to(source).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for suffix, path in located.items()
            if suffix != "FDR_example/manual_curate.csv"
        },
        "forbidden_before_candidate_graph_lock": [
            "read or inspect manual_curate.csv values",
            "select thresholds using NetID manual-curation outcomes",
            "use NetID/author annotations as BioAware seeds for the same query",
            "use phenotype, pathway enrichment, or differential abundance in identity ranking",
            "change dependency grouping after external outcomes are opened",
        ],
        "primary_endpoints": [
            "Recall@1 delta versus frozen DreaMS baseline",
            "MRR delta versus frozen DreaMS baseline",
            "corrected and introduced Top-1 transitions",
            "cluster bootstrap confidence interval",
            "degree-preserving network-decoy comparison",
        ],
        "success_gate": {
            "corrected_gt_introduced": True,
            "recall1_cluster_ci_low_gt_zero": True,
            "mrr_nonnegative": True,
            "beats_degree_preserving_decoy_p95": True,
            "minimum_evaluable_queries": 200,
        },
        "claim_limit": (
            "A passing test establishes incremental ranking evidence on one external "
            "yeast benchmark; it does not establish MSI Level 1 identity, human-tissue "
            "biology, flux, or shared-embedding improvement."
        ),
    }
    destination = output / "contract.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError(f"fail-closed: frozen contract differs: {destination}")
        print(f"[reuse] verified frozen external contract: {destination}", flush=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"fail-closed: non-empty contract directory: {output}")
        _write_atomic(destination, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

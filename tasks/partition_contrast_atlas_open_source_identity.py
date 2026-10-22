#!/usr/bin/env python3
"""Apply one strict phenotype-blind library policy to one open-response source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_msv000088023_hfd_unknown_response import library_partition
from build_contrast_atlas_open_response_collection import load_contract
from encode_contrast_atlas_open_response_entities import load_required_records_with_eligibility


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--library-manifest", required=True, type=Path)
    parser.add_argument("--library-mgf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument("--minimum-wse", type=float, default=0.70)
    parser.add_argument("--minimum-matched-peaks", type=int, default=6)
    parser.add_argument(
        "--mgf-override",
        type=Path,
        help="Phenotype-blind repaired MGF for the selected study.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.contract, args.library_manifest, args.library_mgf):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract, lineage = load_contract(args.contract.resolve(), ROOT)
    selected = [
        source for source in contract["sources"]
        if str(source["study_id"]) == str(args.study_id)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"study is absent or duplicated in contract: {args.study_id}")
    mgf_path = args.mgf_override or Path(str(selected[0]["mgf_path"]))
    if not mgf_path.is_file():
        raise FileNotFoundError(mgf_path)

    entities_path = args.collection / "response_entities.tsv.gz"
    source_manifest_path = args.collection / "source_manifest.tsv"
    collection_report_path = args.collection / "report.json"
    for path in (entities_path, source_manifest_path, collection_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    keys = pd.read_csv(
        entities_path,
        sep="\t",
        usecols=["study_id", "raw_entity_id", "entity_id"],
        dtype=str,
    ).drop_duplicates()
    keys = keys.loc[keys["study_id"].eq(str(args.study_id))].copy()
    if keys.empty or keys.duplicated("raw_entity_id").any() or keys.duplicated("entity_id").any():
        raise RuntimeError("study response registry is empty or non-unique")

    raw_ids, records, mgf_total, eligibility = load_required_records_with_eligibility(
        mgf_path,
        set(keys["raw_entity_id"].astype(str)),
        minimum_fragment_peaks=0,
    )
    if len(raw_ids) != len(keys) or not eligibility["embedding_eligible"].astype(bool).all():
        raise RuntimeError("identity partition loader failed to retain the complete spectrum registry")
    query_spectra = dict(zip(raw_ids, records))
    partition = library_partition(
        query_spectra,
        raw_ids,
        args.library_manifest,
        args.library_mgf,
        args.precursor_ppm,
        args.fragment_da,
        args.minimum_wse,
        args.minimum_matched_peaks,
    ).rename(columns={"feature_id": "raw_entity_id"})
    partition = keys.merge(partition, on="raw_entity_id", how="left", validate="one_to_one")
    if partition["library_supported"].isna().any():
        raise RuntimeError("strict library partition does not cover every response entity")
    partition["identity_status"] = "unresolved_entity"
    partition.loc[partition["library_supported"].astype(bool), "identity_status"] = "library_supported"
    partition.insert(0, "common_identity_policy", "gnps_gold_silver_10ppm_wse070_mp6_v1")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "identity_partition.tsv.gz"
    partition.to_csv(
        output, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    supported = int(partition["identity_status"].eq("library_supported").sum())
    report = {
        "schema": "contrast_atlas_open_source_common_identity_partition_v1",
        "status": "PHENOTYPE_BLIND_COMMON_STRICT_LIBRARY_PARTITION_FROZEN",
        "study_id": str(args.study_id),
        "denominators": {
            "representative_mgf_spectra": mgf_total,
            "response_entities": len(partition),
            "library_supported": supported,
            "unresolved_entity": len(partition) - supported,
        },
        "parameters": {
            "precursor_ppm": args.precursor_ppm,
            "fragment_da": args.fragment_da,
            "minimum_wse": args.minimum_wse,
            "minimum_matched_peaks": args.minimum_matched_peaks,
            "reference_library": "frozen GNPS Gold/Silver positive [M+H]+ development library",
        },
        "leakage_status": (
            "only study/entity identifiers, representative MS/MS and the frozen library were read; "
            "no beta, SE, contrast label, registered relation, rank or retrieval truth was loaded"
        ),
        "provenance": {
            "contract_sha256": sha256(args.contract),
            "contract_lineage_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in lineage
            },
            "response_entities_sha256": sha256(entities_path),
            "source_manifest_sha256": sha256(source_manifest_path),
            "collection_report_sha256": sha256(collection_report_path),
            "representative_mgf_sha256": sha256(mgf_path),
            "representative_mgf_override": bool(args.mgf_override),
            "library_manifest_sha256": sha256(args.library_manifest),
            "library_mgf_sha256": sha256(args.library_mgf),
            "identity_partition_sha256": sha256(output),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Library-supported is a strict common control, not MSI level-1 identity. "
            "Unresolved means no qualifying hit under this library contract, not novelty."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

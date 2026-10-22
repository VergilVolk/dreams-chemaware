#!/usr/bin/env python3
"""Build the frozen phenotype-blind MTBLS1866 positive-DDA measurement object.

This stage cannot read the design ledger.  It constructs the MS1 consensus
universe and links representative MS2 using only the 109-file acquisition
manifest.  Biological labels enter only in a separate response script after
the measurement files and their hashes exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from build_mtbls1066_ketogenic_response import (
    build_consensus,
    extract_or_reuse_features,
    link_representative_ms2,
    write_mgf,
)


EXPECTED_FILES = 109
EXPECTED_BYTES = 10_485_233_845
START_TIME_RE = re.compile(rb'startTimeStamp="([^"]+)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_staged_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "assay_name", "relative_path", "expected_bytes", "linkage_visibility",
        "local_path", "observed_bytes", "sha256", "status",
    }
    if missing := required - set(frame):
        raise RuntimeError(f"staged manifest lacks {sorted(missing)}")
    if len(frame) != EXPECTED_FILES or frame.assay_name.nunique() != EXPECTED_FILES:
        raise RuntimeError("expected 109 unique phenotype-blind assays")
    if set(frame.linkage_visibility) != {"phenotype_blind"}:
        raise RuntimeError("staging manifest exposes non-blind linkage fields")
    expected = pd.to_numeric(frame.expected_bytes, errors="raise").astype("int64")
    observed = pd.to_numeric(frame.observed_bytes, errors="raise").astype("int64")
    if int(expected.sum()) != EXPECTED_BYTES or not expected.equals(observed):
        raise RuntimeError("staged byte universe does not match the frozen manifest")
    if not set(frame.status).issubset({"downloaded", "reused"}):
        raise RuntimeError("staging is not complete")
    for value in frame.local_path:
        if not Path(value).is_file():
            raise FileNotFoundError(value)

    # Helper functions expect these names; both fields remain masked here.
    frame = frame.copy()
    frame["filename"] = frame.assay_name.astype(str)
    frame["group"] = "masked"
    frame["replicate"] = frame.assay_name.astype(str)
    return frame.sort_values("assay_name").reset_index(drop=True)


def acquisition_times(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in manifest.itertuples(index=False):
        with Path(row.local_path).open("rb") as handle:
            header = handle.read(4 << 20)
        match = START_TIME_RE.search(header)
        rows.append(
            {
                "assay_name": str(row.assay_name),
                "acquisition_start_time": (
                    match.group(1).decode("utf-8", errors="strict") if match else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-manifest",
        type=Path,
        default=Path("data/mtbls1866/positive_primary_mzml/staged_manifest.tsv"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/mtbls1866/positive_primary_work_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_mtbls1866_measurement_v1_20261022"
        ),
    )
    parser.add_argument("--noise-threshold", type=float, default=10_000.0)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    parser.add_argument("--consensus-mz-ppm", type=float, default=5.0)
    parser.add_argument("--consensus-rt-sec", type=float, default=15.0)
    parser.add_argument("--ms2-link-ppm", type=float, default=10.0)
    parser.add_argument("--ms2-link-rt-sec", type=float, default=15.0)
    parser.add_argument("--minimum-ms2-peaks", type=int, default=6)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_staged_manifest(args.staged_manifest)
    frames, extraction_audit = extract_or_reuse_features(
        manifest,
        args.work_dir,
        noise_threshold=args.noise_threshold,
        mass_error_ppm=args.mass_error_ppm,
        min_trace_sec=args.min_trace_sec,
        max_trace_sec=args.max_trace_sec,
    )
    metadata, intensities = build_consensus(
        frames, manifest, args.consensus_mz_ppm, args.consensus_rt_sec
    )
    metadata["raw_entity_id"] = metadata.feature_id.map(
        lambda value: f"mtbls1866_pos_rp_{int(value)}"
    )
    links, representatives, ms2_audit = link_representative_ms2(
        metadata,
        manifest,
        args.ms2_link_ppm,
        args.ms2_link_rt_sec,
        require_unique_candidate=True,
        minimum_ms2_peaks=args.minimum_ms2_peaks,
    )

    args.output_dir.mkdir(parents=True)
    metadata["has_representative_ms2"] = metadata.feature_id.isin(representatives)
    entities_path = args.output_dir / "consensus_entities.tsv.gz"
    matrix_path = args.output_dir / "discovery_intensity_matrix.tsv.gz"
    links_path = args.output_dir / "ms2_links.tsv.gz"
    mgf_path = args.output_dir / "representative_ms2.mgf"
    acquisition_path = args.output_dir / "phenotype_blind_acquisition_times.tsv"
    metadata.to_csv(entities_path, sep="\t", index=False)
    intensities.to_csv(matrix_path, sep="\t", index=False)
    pd.DataFrame(extraction_audit).to_csv(
        args.output_dir / "feature_extraction_audit.tsv", sep="\t", index=False
    )
    links.to_csv(links_path, sep="\t", index=False)
    ms2_audit.to_csv(args.output_dir / "ms2_file_audit.tsv", sep="\t", index=False)
    spectrum_index = write_mgf(mgf_path, metadata, representatives)
    spectrum_index.to_csv(
        args.output_dir / "spectrum_index.tsv.gz", sep="\t", index=False
    )
    acquisition = acquisition_times(manifest)
    acquisition.to_csv(acquisition_path, sep="\t", index=False)

    report = {
        "schema": "contrast_atlas_mtbls1866_phenotype_blind_measurement_v1",
        "status": "PHENOTYPE_BLIND_MEASUREMENT_OBJECT_FROZEN",
        "evidence_role": "development_consumed",
        "leakage_status": (
            "only the phenotype-blind 109-file acquisition manifest was read; "
            "the design ledger and COVID/control labels were not inputs"
        ),
        "denominators": {
            "biological_files": len(manifest),
            "consensus_entities": len(metadata),
            "entities_with_unique_representative_ms2": len(representatives),
            "accepted_ms2_links": len(links),
            "ambiguous_ms2_abstentions": int(
                ms2_audit.ambiguous_candidate_ms2_spectra.sum()
            ),
            "files_with_acquisition_start_time": int(
                acquisition.acquisition_start_time.ne("").sum()
            ),
        },
        "parameters": {
            "noise_threshold": args.noise_threshold,
            "mass_error_ppm": args.mass_error_ppm,
            "min_trace_sec": args.min_trace_sec,
            "max_trace_sec": args.max_trace_sec,
            "consensus_mz_ppm": args.consensus_mz_ppm,
            "consensus_rt_sec": args.consensus_rt_sec,
            "ms2_link_ppm": args.ms2_link_ppm,
            "ms2_link_rt_sec": args.ms2_link_rt_sec,
            "minimum_ms2_peaks": args.minimum_ms2_peaks,
            "multiple_candidate_policy": "abstain",
        },
        "provenance": {
            "staged_manifest_sha256": sha256(args.staged_manifest),
            "consensus_entities_sha256": sha256(entities_path),
            "discovery_intensity_matrix_sha256": sha256(matrix_path),
            "ms2_links_sha256": sha256(links_path),
            "representative_mgf_sha256": sha256(mgf_path),
            "phenotype_blind_acquisition_times_sha256": sha256(acquisition_path),
        },
        "claim_boundary": [
            "This is a measurement object, not a COVID biological result.",
            "Precursor/RT-linked MS2 is chemical evidence, not a unique molecular identity.",
            "No library-supported versus unresolved partition has yet been applied.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

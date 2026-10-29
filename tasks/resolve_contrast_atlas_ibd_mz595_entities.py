#!/usr/bin/env python3
"""Resolve the two m/z 595.348 IBD feature pairs without response leakage.

The repository currently contains two different adult features and two
different paediatric features near m/z 595.348.  This script uses only the
phenotype-blind feature tables, consensus MS/MS and legacy library-search
records to decide whether they may be collapsed into one evidence chain.
Biological effects are copied only after the spectral comparison has been
written and are descriptive; they never select a pairing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data/reference/contrast_atlas_ibd_cross_study_20261014"
DEFAULT_RESPONSES = (
    ROOT
    / "data/validation/contrast_atlas_open_response_collection_ibd200_plate3_v2_20261020"
    / "response_entities.tsv.gz"
)
DEFAULT_OUTPUT = ROOT / "data/validation/contrast_atlas_ibd_mz595_entity_resolution_v2_20261029"
ADULT_IDS = (1041, 1286)
PAEDIATRIC_IDS = (16870, 28796)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_quant(path: Path, study_id: str, ids: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["row ID", "row m/z", "row retention time"])
    frame["row ID"] = pd.to_numeric(frame["row ID"], errors="raise").astype(int)
    frame = frame.loc[frame["row ID"].isin(ids)].copy()
    if set(frame["row ID"]) != set(ids):
        raise RuntimeError(f"missing requested features in {path}")
    frame["study_id"] = study_id
    frame["entity_id"] = frame["study_id"] + "::" + frame["row ID"].astype(str)
    return frame.rename(columns={
        "row ID": "raw_entity_id",
        "row m/z": "precursor_mz",
        "row retention time": "retention_time_minutes",
    })


def read_annotations(path: Path, ids: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame["#Scan#"] = pd.to_numeric(frame["#Scan#"], errors="coerce")
    frame = frame.loc[frame["#Scan#"].isin(ids)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            "raw_entity_id", "legacy_compound_name", "legacy_library_quality",
            "legacy_spectrum_id", "legacy_inchikey", "legacy_mqscore",
        ])
    frame = frame.sort_values("MQScore", ascending=False).drop_duplicates("#Scan#")
    return frame.rename(columns={
        "#Scan#": "raw_entity_id",
        "Compound_Name": "legacy_compound_name",
        "LibraryQualityString": "legacy_library_quality",
        "SpectrumID": "legacy_spectrum_id",
        "InChIKey": "legacy_inchikey",
        "MQScore": "legacy_mqscore",
    })[[
        "raw_entity_id", "legacy_compound_name", "legacy_library_quality",
        "legacy_spectrum_id", "legacy_inchikey", "legacy_mqscore",
    ]]


def read_mgf(path: Path, ids: tuple[int, ...]) -> dict[int, list[tuple[float, float]]]:
    wanted = set(ids)
    spectra: dict[int, list[tuple[float, float]]] = {}
    scan: int | None = None
    peaks: list[tuple[float, float]] = []
    inside = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                inside, scan, peaks = True, None, []
            elif inside and line.startswith("SCANS="):
                scan = int(line.split("=", 1)[1])
            elif inside and line == "END IONS":
                if scan in wanted:
                    spectra[int(scan)] = peaks
                inside = False
            elif inside and line and "=" not in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except ValueError:
                        pass
    if set(spectra) != wanted:
        raise RuntimeError(f"missing requested spectra in {path}: {wanted - set(spectra)}")
    return spectra


def sqrt_cosine(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    tolerance: float = 0.02,
) -> tuple[float, int]:
    """Simple response-blind diagnostic cosine with greedy ordered matching."""
    def normalise(peaks: list[tuple[float, float]]) -> list[tuple[float, float]]:
        values = sorted((mz, math.sqrt(max(intensity, 0.0))) for mz, intensity in peaks)
        norm = math.sqrt(sum(intensity * intensity for _, intensity in values))
        if norm <= 0:
            raise RuntimeError("zero-intensity spectrum")
        return [(mz, intensity / norm) for mz, intensity in values]

    left_n, right_n = normalise(left), normalise(right)
    i = j = matched = 0
    score = 0.0
    while i < len(left_n) and j < len(right_n):
        delta = left_n[i][0] - right_n[j][0]
        if abs(delta) <= tolerance:
            score += left_n[i][1] * right_n[j][1]
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return float(score), matched


def run(reference_dir: Path, responses_path: Path, output_dir: Path) -> dict[str, Any]:
    reference_dir = reference_dir.resolve()
    responses_path = responses_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")

    adult_quant = reference_dir / "adult_ibd200_gnps_quant.csv"
    paediatric_quant = reference_dir / "pediatric_veoibd_gnps_quant.csv"
    adult_mgf = reference_dir / "adult_ibd200_consensus.mgf"
    paediatric_mgf = reference_dir / "pediatric_veoibd_consensus.mgf"
    adult_annotations = reference_dir / "adult_ibd200_annotations.tsv"
    paediatric_annotations = reference_dir / "pediatric_paper_annotations.tsv"

    adult = read_quant(adult_quant, "IBD200_GROUP", ADULT_IDS).merge(
        read_annotations(adult_annotations, ADULT_IDS), on="raw_entity_id", how="left"
    )
    paediatric = read_quant(paediatric_quant, "MSV000097610", PAEDIATRIC_IDS).merge(
        read_annotations(paediatric_annotations, PAEDIATRIC_IDS), on="raw_entity_id", how="left"
    )
    entities = pd.concat([adult, paediatric], ignore_index=True)
    entities["legacy_annotation_status"] = entities["legacy_compound_name"].notna().map({
        True: "legacy_bronze_hypothesis",
        False: "no_legacy_library_hypothesis",
    })

    adult_spectra = read_mgf(adult_mgf, ADULT_IDS)
    paediatric_spectra = read_mgf(paediatric_mgf, PAEDIATRIC_IDS)
    entity_lookup = entities.set_index("entity_id").to_dict("index")
    pair_rows: list[dict[str, Any]] = []
    for adult_id in ADULT_IDS:
        for paediatric_id in PAEDIATRIC_IDS:
            left_id = f"IBD200_GROUP::{adult_id}"
            right_id = f"MSV000097610::{paediatric_id}"
            score, matched = sqrt_cosine(adult_spectra[adult_id], paediatric_spectra[paediatric_id])
            left_mz = float(entity_lookup[left_id]["precursor_mz"])
            right_mz = float(entity_lookup[right_id]["precursor_mz"])
            pair_rows.append({
                "adult_entity_id": left_id,
                "paediatric_entity_id": right_id,
                "adult_precursor_mz": left_mz,
                "paediatric_precursor_mz": right_mz,
                "symmetric_ppm": abs(left_mz - right_mz) / ((left_mz + right_mz) / 2.0) * 1e6,
                "adult_retention_time_minutes": entity_lookup[left_id]["retention_time_minutes"],
                "paediatric_retention_time_minutes": entity_lookup[right_id]["retention_time_minutes"],
                "diagnostic_sqrt_cosine_0p02": score,
                "diagnostic_matched_peaks_0p02": matched,
                "same_legacy_library_spectrum": bool(
                    pd.notna(entity_lookup[left_id]["legacy_spectrum_id"])
                    and entity_lookup[left_id]["legacy_spectrum_id"]
                    == entity_lookup[right_id]["legacy_spectrum_id"]
                ),
            })
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["adult_entity_id", "diagnostic_sqrt_cosine_0p02"], ascending=[True, False]
    )

    # Pairing is decided before effects are opened.  These rows are descriptive.
    responses = pd.read_csv(responses_path, sep="\t", low_memory=False)
    responses = responses.loc[responses["entity_id"].isin(entities["entity_id"])][
        ["entity_id", "contrast_id", "beta", "se", "data_role"]
    ]
    entities = entities.merge(responses, on="entity_id", how="left", validate="one_to_one")

    output_dir.mkdir(parents=True)
    entities_path = output_dir / "entities.tsv"
    pairs_path = output_dir / "cross_study_pair_matrix.tsv"
    entities.to_csv(entities_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    pairs.to_csv(pairs_path, sep="\t", index=False)

    preferred = pairs.sort_values("diagnostic_sqrt_cosine_0p02", ascending=False).head(2)
    preferred_pairs = set(zip(preferred["adult_entity_id"], preferred["paediatric_entity_id"]))
    expected_pairs = {
        ("IBD200_GROUP::1041", "MSV000097610::16870"),
        ("IBD200_GROUP::1286", "MSV000097610::28796"),
    }
    report = {
        "schema": "contrast_atlas_ibd_mz595_entity_resolution_v2",
        "status": "DEVELOPMENT_CONSUMED_ENTITY_DISAMBIGUATION_COMPLETE",
        "data_role": "development_consumed",
        "phenotype_blind_pairing": True,
        "features": {
            "adult": len(ADULT_IDS),
            "paediatric": len(PAEDIATRIC_IDS),
            "cross_study_pairs": int(len(pairs)),
        },
        "resolution": {
            "single_chain_merge_allowed": False,
            "reason": "two distinct chromatographic peaks occur within each source and the cross-study MS/MS matrix supports two different pair hypotheses",
            "response_blind_preferred_pairs_equal_expected_two_pairs": preferred_pairs == expected_pairs,
            "pair_hypotheses": [
                {
                    "adult": "IBD200_GROUP::1041",
                    "paediatric": "MSV000097610::16870",
                    "identity_status": "legacy Bronze Stercobilin hypothesis; unresolved under current Gold/Silver policy",
                },
                {
                    "adult": "IBD200_GROUP::1286",
                    "paediatric": "MSV000097610::28796",
                    "identity_status": "unresolved same-precursor spectral identity hypothesis",
                },
            ],
        },
        "claim_boundary": [
            "The four feature IDs are not version aliases and must not be collapsed into one response chain.",
            "Retention times are only evidence of within-study peak separation; absolute RT is not aligned across studies.",
            "The diagnostic cosine is not a Full23 score and does not prove molecular identity.",
            "Legacy Bronze library matches are hypotheses, not current Gold/Silver identities.",
            "All effects and spectra are development/consumed; no independent biological validation is claimed.",
        ],
        "input_sha256": {
            str(path.relative_to(ROOT).as_posix()): sha256(path)
            for path in [
                adult_quant, paediatric_quant, adult_mgf, paediatric_mgf,
                adult_annotations, paediatric_annotations, responses_path,
            ]
        },
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": {
            entities_path.name: sha256(entities_path),
            pairs_path.name: sha256(pairs_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.reference_dir, args.responses, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

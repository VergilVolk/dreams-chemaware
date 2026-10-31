#!/usr/bin/env python3
"""Freeze complete, model-independent observed-ion candidate blocks.

Candidates are admitted before any spectral-model score is read.  The axis is
an observed-precursor identity *hypothesis*: explicit charge/polarity/adduct
conflicts are excluded, while missing ion chemistry is retained and labelled
unresolved.  It is not a neutral-mass identity graph and never pools effects.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_tsv_writer(path: Path, fields: list[str]):
    handle = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    text = __import__("io").TextIOWrapper(handle, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return handle, text, writer


def parse_charge(raw: str) -> tuple[int | None, str]:
    value = str(raw or "").strip()
    match = re.fullmatch(r"(\d+)\s*([+-]?)", value)
    if not match:
        return None, "unknown"
    magnitude = int(match.group(1))
    if magnitude <= 0 or magnitude > 10:
        return None, "unknown"
    sign = match.group(2)
    return magnitude, "positive" if sign == "+" else "negative" if sign == "-" else "unknown"


def normalized_adduct(metadata: dict[str, str]) -> str:
    for key in ("ADDUCT", "ION", "IONTYPE", "PRECURSORTYPE"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value.replace(" ", "")
    return "unknown"


def compatibility(query: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    q_charge, c_charge = query["charge"], candidate["charge"]
    q_pol, c_pol = query["polarity"], candidate["polarity"]
    q_adduct, c_adduct = query["adduct"], candidate["adduct"]
    if q_charge is not None and c_charge is not None and q_charge != c_charge:
        return False, "explicit_charge_conflict"
    if q_pol != "unknown" and c_pol != "unknown" and q_pol != c_pol:
        return False, "explicit_polarity_conflict"
    if q_adduct != "unknown" and c_adduct != "unknown" and q_adduct != c_adduct:
        return False, "explicit_adduct_conflict"
    if q_charge is None or c_charge is None:
        return True, "same_mz_charge_unresolved_adduct_unresolved"
    if q_adduct == "unknown" or c_adduct == "unknown":
        return True, "same_mz_charge_compatible_adduct_unresolved"
    if q_pol == "unknown" or c_pol == "unknown":
        return True, "same_mz_charge_adduct_compatible_polarity_unresolved"
    return True, "observed_ion_fields_compatible"


def write_plain_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_id(prefix: str, *parts: str) -> str:
    payload = "\t".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-dir",
        default="data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023",
    )
    parser.add_argument(
        "--source-report",
        default="data/validation/contrast_atlas_sparse_local_alignment_v3_chemprop_20261023/report.json",
    )
    parser.add_argument(
        "--identity-partition",
        default="data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/identity_partition.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation/contrast_atlas_model_independent_identity_axis_v1_20261027",
    )
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument(
        "--spectrum-asset-root",
        default="",
        help="Optional checkout containing historical repo-relative MGF assets.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tasks"))
    from run_contrast_atlas_sparse_fragment_measure_baseline import iter_mgf, spectrum_id

    collection = (root / args.collection_dir).resolve()
    source_report_path = (root / args.source_report).resolve()
    identity_partition_path = (root / args.identity_partition).resolve()
    output = (root / args.output_dir).resolve()
    spectrum_asset_root = Path(args.spectrum_asset_root).resolve() if args.spectrum_asset_root else None
    if args.precursor_ppm <= 0:
        raise ValueError("precursor ppm must be positive")
    if output.exists():
        raise FileExistsError(output)

    collection_report = json.loads((collection / "report.json").read_text(encoding="utf-8"))
    if collection_report.get("status") != "DEVELOPMENT_CONSUMED_FROZEN_BASE_EXTENSION_COMPLETE":
        raise ValueError("refusing a non-frozen-base response collection")
    response_path = collection / "response_entities.tsv.gz"
    expected_response_hash = collection_report["output_sha256"]["response_entities.tsv.gz"]
    if sha256(response_path) != expected_response_hash:
        raise ValueError("response collection hash mismatch")

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    expected_response_rows = int(collection_report["denominators"]["eligible_response_rows"])
    source_response_rows = source_report.get("denominators", {}).get("response_rows")
    if source_response_rows is None:
        source_response_rows = source_report.get("denominators", {}).get("eligible_response_rows")
    if source_response_rows is not None and int(source_response_rows) != expected_response_rows:
        raise ValueError(
            "source report response denominator differs from the response collection: "
            f"{source_response_rows} != {expected_response_rows}"
        )
    source_paths = {str(row["study_id"]): str(row["mgf_path"]) for row in source_report["sources"]}

    if not identity_partition_path.exists():
        raise FileNotFoundError(identity_partition_path)
    entity_status: dict[tuple[str, str], str] = {}
    with gzip.open(identity_partition_path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["study_id"], row["entity_id"])
            if key in entity_status:
                raise ValueError(f"duplicate common identity row: {key}")
            entity_status[key] = row["identity_status"]
    expected_entities = int(collection_report["denominators"]["unique_study_entities"])
    if len(entity_status) != expected_entities:
        raise ValueError(
            "common identity denominator differs from the response collection: "
            f"{len(entity_status)} != {expected_entities}"
        )

    raw_to_entity: dict[str, dict[str, str]] = defaultdict(dict)
    response_rows = 0
    with gzip.open(response_path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            response_rows += 1
            sid, raw_id, entity_id = row["study_id"], row["raw_entity_id"], row["entity_id"]
            previous = raw_to_entity[sid].setdefault(raw_id, entity_id)
            if previous != entity_id:
                raise ValueError(f"raw entity maps to multiple frozen entities: {sid}::{raw_id}")
            if (sid, entity_id) not in entity_status:
                raise ValueError(f"entity absent from common identity partition: {sid}::{entity_id}")

    if response_rows != expected_response_rows:
        raise ValueError(
            f"response row denominator differs from collection report: {response_rows} != {expected_response_rows}"
        )
    if sum(len(rows) for rows in raw_to_entity.values()) != expected_entities:
        raise ValueError("response entity denominator differs from collection report")

    if set(raw_to_entity) != set(source_paths):
        raise ValueError("response studies and source report studies differ")

    registries: dict[str, list[dict[str, Any]]] = {}
    mgf_hashes: dict[str, str] = {}
    for sid in sorted(source_paths):
        relative_path = Path(source_paths[sid].replace("\\", "/"))
        path = root / relative_path
        if not path.exists() and spectrum_asset_root is not None:
            path = spectrum_asset_root / relative_path
        if not path.exists():
            raise FileNotFoundError(path)
        found: dict[str, dict[str, Any]] = {}
        for metadata, _peaks in iter_mgf(path):
            raw_id = spectrum_id(metadata)
            entity_id = raw_to_entity[sid].get(raw_id)
            if entity_id is None:
                continue
            fields = str(metadata.get("PEPMASS", "")).split()
            try:
                mz = float(fields[0])
            except (IndexError, ValueError):
                continue
            if not math.isfinite(mz) or mz <= 0:
                continue
            charge, polarity = parse_charge(metadata.get("CHARGE", ""))
            row = {
                "study_id": sid,
                "raw_entity_id": raw_id,
                "entity_id": entity_id,
                "identity_status": entity_status[(sid, entity_id)],
                "precursor_mz": mz,
                "charge": charge,
                "polarity": polarity,
                "adduct": normalized_adduct(metadata),
            }
            if entity_id in found:
                raise ValueError(f"multiple valid representative spectra for {entity_id}")
            found[entity_id] = row
        expected = set(raw_to_entity[sid].values())
        if set(found) != expected:
            raise ValueError(f"{sid} has {len(expected - set(found))} frozen entities without valid precursor")
        registries[sid] = sorted(found.values(), key=lambda row: (row["precursor_mz"], row["entity_id"]))
        mgf_hashes[sid] = sha256(path)

    output.mkdir(parents=True)
    entity_path = output / "observed_ion_registry.tsv.gz"
    entity_fields = [
        "study_id", "raw_entity_id", "entity_id", "identity_status", "precursor_mz",
        "charge", "polarity", "adduct",
    ]
    raw_handle, text_handle, writer = gzip_tsv_writer(entity_path, entity_fields)
    try:
        for sid in sorted(registries):
            for row in registries[sid]:
                writer.writerow(row)
    finally:
        text_handle.close()

    pair_path = output / "directed_complete_candidate_pairs.tsv.gz"
    block_path = output / "query_target_source_blocks.tsv.gz"
    pair_fields = [
        "query_block_id", "query_block_index", "candidate_index_within_block",
        "directed_pair_id", "query_study_id", "query_entity_id",
        "candidate_study_id", "candidate_entity_id",
        "query_identity_status", "candidate_identity_status", "query_precursor_mz",
        "candidate_precursor_mz", "symmetric_ppm", "query_charge", "candidate_charge",
        "query_polarity", "candidate_polarity", "query_adduct", "candidate_adduct",
        "physicochemical_compatibility_grade", "candidate_axis",
    ]
    block_fields = [
        "query_block_id", "query_block_index", "candidate_pair_offset_start",
        "candidate_pair_offset_end", "query_study_id", "query_entity_id",
        "candidate_study_id", "query_identity_status",
        "mass_window_candidates", "explicit_conflicts_excluded", "eligible_candidates",
        "strict_observed_ion_candidates", "charge_compatible_adduct_unresolved",
        "charge_unresolved_candidates", "candidate_block_sha256", "result_class",
    ]
    pair_raw, pair_text, pair_writer = gzip_tsv_writer(pair_path, pair_fields)
    block_raw, block_text, block_writer = gzip_tsv_writer(block_path, block_fields)
    candidate_pairs = 0
    mass_window_pairs = 0
    conflicts = 0
    nonempty_blocks = 0
    all_blocks = 0
    grade_counts: dict[str, int] = defaultdict(int)
    try:
        for query_study in sorted(registries):
            query_rows = registries[query_study]
            for candidate_study in sorted(registries):
                if candidate_study == query_study:
                    continue
                candidate_rows = registries[candidate_study]
                candidate_mz = [row["precursor_mz"] for row in candidate_rows]
                for query in query_rows:
                    block_index = all_blocks
                    all_blocks += 1
                    block_id = stable_id(
                        "block", query_study, query["entity_id"], candidate_study
                    )
                    mz = float(query["precursor_mz"])
                    tolerance = mz * args.precursor_ppm * 1e-6
                    begin = bisect.bisect_left(candidate_mz, mz - tolerance)
                    end = bisect.bisect_right(candidate_mz, mz + tolerance)
                    mass_count = end - begin
                    mass_window_pairs += mass_count
                    eligible: list[tuple[dict[str, Any], float, str]] = []
                    local_conflicts = 0
                    for candidate in candidate_rows[begin:end]:
                        allowed, grade = compatibility(query, candidate)
                        if not allowed:
                            local_conflicts += 1
                            conflicts += 1
                            continue
                        candidate_mass = float(candidate["precursor_mz"])
                        ppm = abs(mz - candidate_mass) / ((mz + candidate_mass) / 2.0) * 1e6
                        eligible.append((candidate, ppm, grade))
                    digest = hashlib.sha256()
                    strict = charge_adduct_unresolved = charge_unresolved = 0
                    pair_offset_start = candidate_pairs
                    for candidate_index, (candidate, ppm, grade) in enumerate(eligible):
                        digest.update(
                            f'{candidate["entity_id"]}\t{candidate["precursor_mz"]:.10f}\t{grade}\n'.encode("utf-8")
                        )
                        grade_counts[grade] += 1
                        strict += int(grade == "observed_ion_fields_compatible")
                        charge_adduct_unresolved += int(grade == "same_mz_charge_compatible_adduct_unresolved")
                        charge_unresolved += int(grade == "same_mz_charge_unresolved_adduct_unresolved")
                        pair_writer.writerow(
                            {
                                "query_block_id": block_id,
                                "query_block_index": block_index,
                                "candidate_index_within_block": candidate_index,
                                "directed_pair_id": stable_id(
                                    "pair", query_study, query["entity_id"],
                                    candidate_study, candidate["entity_id"],
                                ),
                                "query_study_id": query_study,
                                "query_entity_id": query["entity_id"],
                                "candidate_study_id": candidate_study,
                                "candidate_entity_id": candidate["entity_id"],
                                "query_identity_status": query["identity_status"],
                                "candidate_identity_status": candidate["identity_status"],
                                "query_precursor_mz": f"{mz:.10f}",
                                "candidate_precursor_mz": f'{candidate["precursor_mz"]:.10f}',
                                "symmetric_ppm": f"{ppm:.8f}",
                                "query_charge": "unknown" if query["charge"] is None else query["charge"],
                                "candidate_charge": "unknown" if candidate["charge"] is None else candidate["charge"],
                                "query_polarity": query["polarity"],
                                "candidate_polarity": candidate["polarity"],
                                "query_adduct": query["adduct"],
                                "candidate_adduct": candidate["adduct"],
                                "physicochemical_compatibility_grade": grade,
                                "candidate_axis": "observed_precursor_10ppm_v1",
                            }
                        )
                    if eligible:
                        nonempty_blocks += 1
                    candidate_pairs += len(eligible)
                    block_writer.writerow(
                        {
                            "query_block_id": block_id,
                            "query_block_index": block_index,
                            "candidate_pair_offset_start": pair_offset_start,
                            "candidate_pair_offset_end": candidate_pairs,
                            "query_study_id": query_study,
                            "query_entity_id": query["entity_id"],
                            "candidate_study_id": candidate_study,
                            "query_identity_status": query["identity_status"],
                            "mass_window_candidates": mass_count,
                            "explicit_conflicts_excluded": local_conflicts,
                            "eligible_candidates": len(eligible),
                            "strict_observed_ion_candidates": strict,
                            "charge_compatible_adduct_unresolved": charge_adduct_unresolved,
                            "charge_unresolved_candidates": charge_unresolved,
                            "candidate_block_sha256": digest.hexdigest(),
                            "result_class": "score_complete_block" if eligible else "abstain_no_physicochemical_candidate",
                        }
                    )
    finally:
        pair_text.close()
        block_text.close()

    summary_rows = []
    for sid in sorted(registries):
        rows = registries[sid]
        summary_rows.append(
            {
                "study_id": sid,
                "entities": len(rows),
                "charge_known": sum(row["charge"] is not None for row in rows),
                "polarity_known": sum(row["polarity"] != "unknown" for row in rows),
                "adduct_known": sum(row["adduct"] != "unknown" for row in rows),
                "mgf_sha256": mgf_hashes[sid],
            }
        )
    summary_path = output / "study_observed_ion_coverage.tsv"
    write_plain_tsv(summary_path, summary_rows, list(summary_rows[0]))

    report = {
        "schema": "contrast_atlas_model_independent_identity_axis_v1",
        "status": "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN",
        "data_role": "development_consumed",
        "candidate_definition": {
            "precursor_tolerance_ppm": args.precursor_ppm,
            "source_disjoint": True,
            "model_scores_read": False,
            "response_table_read_for_entity_lineage": True,
            "response_beta_se_or_relation_truth_used_for_admission": False,
            "explicit_conflicts_excluded": ["charge", "polarity", "adduct"],
            "missing_ion_chemistry": "retained and labelled unresolved",
            "block_rule": "all eligible candidates in every query-by-target-source block; no top-k preselection",
            "block_indexing": "zero-based contiguous query_block_index; candidate offsets are half-open [start,end) into directed_complete_candidate_pairs.tsv.gz",
        },
        "denominators": {
            "studies": len(registries),
            "entities": sum(len(rows) for rows in registries.values()),
            "query_target_source_blocks": all_blocks,
            "nonempty_candidate_blocks": nonempty_blocks,
            "abstain_no_candidate_blocks": all_blocks - nonempty_blocks,
            "mass_window_directed_pairs_before_explicit_conflicts": mass_window_pairs,
            "explicit_conflict_pairs_excluded": conflicts,
            "directed_candidate_pairs": candidate_pairs,
            "compatibility_grades": dict(sorted(grade_counts.items())),
        },
        "claim_boundary": [
            "This is an observed-precursor candidate axis, not a confirmed molecular-identity graph.",
            "Adduct metadata are absent for the frozen response spectra, so cross-adduct neutral-mass identity is not claimed.",
            "Charge or polarity missingness is retained and labelled rather than silently treated as compatibility.",
            "Isomers, in-source fragments, isotopes, and ion-family redundancy remain possible.",
            "Every nonempty candidate block must be scored in full by a downstream model; singleton or mutual-top1 preselection is forbidden.",
            "No beta, SE, phenotype, registered relation, or identity truth was used to construct this axis.",
        ],
        "provenance": {
            "response_entities_sha256": sha256(response_path),
            "common_identity_partition_sha256": sha256(identity_partition_path),
            "collection_report_sha256": sha256(collection / "report.json"),
            "source_report_sha256": sha256(source_report_path),
            "implementation_sha256": sha256(Path(__file__)),
            "mgf_sha256_by_study": mgf_hashes,
        },
        "outputs": {
            "observed_ion_registry.tsv.gz": sha256(entity_path),
            "directed_complete_candidate_pairs.tsv.gz": sha256(pair_path),
            "query_target_source_blocks.tsv.gz": sha256(block_path),
            "study_observed_ion_coverage.tsv": sha256(summary_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (entity_path, pair_path, block_path, summary_path, report_path)
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

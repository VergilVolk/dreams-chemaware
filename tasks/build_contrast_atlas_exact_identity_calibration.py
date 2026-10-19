#!/usr/bin/env python3
"""Build a post-truth exact planar-InChIKey calibration ledger.

The calibration is intentionally separate from Contrast Atlas graph building
and scoring.  It opens strict library identities only after a frozen scoring
report is present and binds them to collection entities by exact source-local
entity IDs.  Names, precursor masses, and retention times are never used to
rescue a join.

This asset calibrates spectral entity retrieval.  It is not biological
relation truth, signed response truth, or evidence that unresolved entities
are useful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import permutations
from pathlib import Path
from typing import Any

import pandas as pd

from run_ibd_cross_study_contrast_query import library_hits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = ROOT / "data/validation/contrast_atlas_open_response_collection_v3_ibd_20261019"
DEFAULT_FREEZE_REPORT = ROOT / "data/validation/contrast_atlas_sparse_fragment_measure_baseline_v2_ibd_20261019/report.json"
DEFAULT_OUTPUT = ROOT / "data/validation/contrast_atlas_exact_identity_calibration_v3_20261019"
EXPECTED_COLLECTION_STATUS = "DEVELOPMENT_CONSUMED_OPEN_RESPONSE_COLLECTION_FROZEN"
EXPECTED_FREEZE_STATUS = "DEVELOPMENT_CONSUMED_TRUTH_BLIND_ALGORITHMIC_BASELINE_COMPLETE"
PLANAR_PATTERN = re.compile(r"^[A-Z]{14}(?:-[A-Z]{10}-[A-Z])?$")


PARTITION_SOURCES: dict[str, dict[str, Any]] = {
    "MSV000097426": {
        "path": "data/validation/msv097426_to_msv084112_hfd_obesity_query_20261014_v2/library_partition.csv.gz",
        "entity_column": "anchor_id",
        "key_column": "library_inchikey",
        "supported_column": "library_supported",
        "filter": {"study": "mouse"},
    },
    "MSV000084112": {
        "path": "data/validation/msv097426_to_msv084112_hfd_obesity_query_20261014_v2/library_partition.csv.gz",
        "entity_column": "anchor_id",
        "key_column": "library_inchikey",
        "supported_column": "library_supported",
        "filter": {"study": "human"},
    },
    "MSV000083077": {
        "path": "data/validation/msv084112_to_msv083077_obesity_query_20261014_v1/library_partition.csv.gz",
        "entity_column": "anchor_id",
        "key_column": "library_inchikey",
        "supported_column": "library_supported",
        "filter": {"study": "candidate"},
    },
    "MSV000088023": {
        "path": "data/validation/msv000088023_hfd_unknown_response_20261014_v3/library_partition.csv",
        "entity_column": "feature_id",
        "key_column": "library_inchikey",
        "supported_column": "library_supported",
        "filter": {},
    },
    "MSV000090788": {
        "path": "data/validation/public_fbmn_ad_identity_partition_20261017_v1/identity_partition.tsv.gz",
        "sep": "\t",
        "entity_column": "entity_id",
        "key_column": "InChIKey-Planar",
        "status_column": "identity_partition",
        "status_value": "library_supported",
        "filter": {},
    },
}

IBD_SOURCES: dict[str, dict[str, str]] = {
    "MSV000097610": {
        "path": "data/reference/contrast_atlas_ibd_cross_study_20261014/pediatric_paper_annotations.tsv",
    },
    "IBD200_GROUP": {
        "path": "data/reference/contrast_atlas_ibd_cross_study_20261014/adult_ibd200_annotations.tsv",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_entity_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+\.0+", text):
        return text.split(".", 1)[0]
    return text


def strict_planar_inchikey(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not PLANAR_PATTERN.fullmatch(text):
        return None
    return text[:14]


def parse_supported(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError("library-supported column contains invalid booleans")
    return values.isin({"true", "1"})


def resolve_unambiguous_evidence(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep an entity only when all strict evidence rows agree on one planar key."""
    required = {"raw_entity_id", "planar_inchikey", "evidence_source"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"evidence missing columns: {sorted(required - set(frame.columns))}")
    rows: list[dict[str, str]] = []
    ambiguous = 0
    for raw_id, block in frame.groupby("raw_entity_id", sort=True):
        keys = sorted(set(block["planar_inchikey"].dropna().astype(str)))
        if len(keys) != 1:
            ambiguous += 1
            continue
        rows.append({
            "raw_entity_id": str(raw_id),
            "planar_inchikey": keys[0],
            "evidence_source": ";".join(sorted(set(block["evidence_source"].astype(str)))),
        })
    return pd.DataFrame(rows, columns=["raw_entity_id", "planar_inchikey", "evidence_source"]), ambiguous


def load_partition_evidence(study_id: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], Path]:
    path = ROOT / config["path"]
    frame = pd.read_csv(path, sep=config.get("sep", ","), dtype={config["entity_column"]: str})
    for column, value in config.get("filter", {}).items():
        frame = frame.loc[frame[column].astype(str).eq(str(value))].copy()
    if config.get("supported_column"):
        frame = frame.loc[parse_supported(frame[config["supported_column"]])].copy()
    else:
        frame = frame.loc[
            frame[config["status_column"]].astype(str).eq(config["status_value"])
        ].copy()
    supported_rows = len(frame)
    evidence = pd.DataFrame({
        "raw_entity_id": frame[config["entity_column"]].map(normalize_entity_id),
        "planar_inchikey": frame[config["key_column"]].map(strict_planar_inchikey),
        "evidence_source": str(path.relative_to(ROOT).as_posix()),
    })
    invalid_key_rows = int(evidence["planar_inchikey"].isna().sum())
    evidence = evidence.dropna(subset=["planar_inchikey"]).copy()
    resolved, ambiguous = resolve_unambiguous_evidence(evidence)
    audit = {
        "supported_annotation_rows": int(supported_rows),
        "invalid_or_empty_planar_inchikey_rows": invalid_key_rows,
        "ambiguous_entity_ids_excluded": int(ambiguous),
        "strict_entity_keys": int(len(resolved)),
    }
    return resolved, audit, path


def load_ibd_evidence(study_id: str, config: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int], Path]:
    path = ROOT / config["path"]
    # Reuse the exact strict hit selection that defined the IBD local library
    # partition: MQScore >= 0.70, >= 6 shared peaks, best MQScore per scan.
    hits = library_hits(path)
    rows = []
    invalid = 0
    for scan, hit in hits.items():
        key = strict_planar_inchikey(hit.get("InChIKey-Planar") or hit.get("InChIKey"))
        if key is None:
            invalid += 1
            continue
        rows.append({
            "raw_entity_id": normalize_entity_id(scan),
            "planar_inchikey": key,
            "evidence_source": str(path.relative_to(ROOT).as_posix()),
        })
    evidence, ambiguous = resolve_unambiguous_evidence(pd.DataFrame(rows)) if rows else (
        pd.DataFrame(columns=["raw_entity_id", "planar_inchikey", "evidence_source"]), 0
    )
    audit = {
        "supported_annotation_rows": int(len(hits)),
        "invalid_or_empty_planar_inchikey_rows": int(invalid),
        "ambiguous_entity_ids_excluded": int(ambiguous),
        "strict_entity_keys": int(len(evidence)),
    }
    return evidence, audit, path


def validate_posttruth_freeze(freeze_report: Path, collection_entities: Path) -> dict[str, Any]:
    report = json.loads(freeze_report.read_text(encoding="utf-8"))
    if report.get("status") != EXPECTED_FREEZE_STATUS:
        raise RuntimeError("truth-blind graph/score freeze status is not complete")
    expected = report.get("input_sha256", {}).get(
        "data\\validation\\contrast_atlas_open_response_collection_v3_ibd_20261019\\response_entities.tsv.gz"
    )
    observed = sha256(collection_entities)
    if expected != observed:
        raise RuntimeError("frozen score report does not bind the current collection entities")
    checksum_path = freeze_report.parent / "checksums.sha256"
    score_path = freeze_report.parent / "fixed_candidate_scores.tsv.gz"
    if not checksum_path.exists() or not score_path.exists():
        raise RuntimeError("frozen score artifact or checksum manifest is missing")
    checksums: dict[str, str] = {}
    for raw in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, name = raw.split(maxsplit=1)
        checksums[name.strip()] = digest.strip()
    score_digest = sha256(score_path)
    if checksums.get(score_path.name) != score_digest:
        raise RuntimeError("frozen candidate score checksum mismatch")
    if checksums.get(freeze_report.name) != sha256(freeze_report):
        raise RuntimeError("frozen score report checksum mismatch")
    return {
        "freeze_report_sha256": sha256(freeze_report),
        "collection_entities_sha256": observed,
        "fixed_candidate_scores_sha256": score_digest,
        "freeze_checksums_manifest_sha256": sha256(checksum_path),
    }


def directed_cross_source_pairs(members: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "planar_inchikey", "left_study_id", "left_source_family", "left_entity_id",
        "left_raw_entity_id", "right_study_id", "right_source_family", "right_entity_id",
        "right_raw_entity_id", "truth_class", "truth_scope",
    ]
    rows: list[dict[str, str]] = []
    unique_members = members.drop_duplicates(
        ["planar_inchikey", "study_id", "source_family", "entity_id"]
    )
    for key, block in unique_members.groupby("planar_inchikey", sort=True):
        records = block.to_dict("records")
        for left, right in permutations(records, 2):
            if left["source_family"] == right["source_family"]:
                continue
            rows.append({
                "planar_inchikey": key,
                "left_study_id": left["study_id"],
                "left_source_family": left["source_family"],
                "left_entity_id": left["entity_id"],
                "left_raw_entity_id": left["raw_entity_id"],
                "right_study_id": right["study_id"],
                "right_source_family": right["source_family"],
                "right_entity_id": right["entity_id"],
                "right_raw_entity_id": right["raw_entity_id"],
                "truth_class": "exact_planar_inchikey_identity",
                "truth_scope": "posttruth_spectral_entity_retrieval_calibration_only",
            })
    return pd.DataFrame(rows, columns=columns)


def build(collection_dir: Path, freeze_report: Path, output_dir: Path) -> dict[str, Any]:
    collection_dir = collection_dir.resolve()
    freeze_report = freeze_report.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    collection_report = json.loads((collection_dir / "report.json").read_text(encoding="utf-8"))
    if collection_report.get("status") != EXPECTED_COLLECTION_STATUS:
        raise RuntimeError("open collection is not frozen")
    entities_path = collection_dir / "response_entities.tsv.gz"
    freeze_audit = validate_posttruth_freeze(freeze_report, entities_path)
    response = pd.read_csv(
        entities_path,
        sep="\t",
        dtype={"study_id": str, "source_family": str, "raw_entity_id": str, "entity_id": str, "contrast_id": str},
    )
    if set(response["study_id"]) != set(PARTITION_SOURCES) | set(IBD_SOURCES):
        raise RuntimeError("seven-source registry drift")
    entity_registry = response[
        ["study_id", "source_family", "raw_entity_id", "entity_id", "identity_status"]
    ].drop_duplicates().copy()
    entity_registry["raw_entity_id"] = entity_registry["raw_entity_id"].map(normalize_entity_id)
    if entity_registry.duplicated(["study_id", "raw_entity_id"]).any():
        raise RuntimeError("source-local raw entity IDs are not unique")

    evidence_blocks = []
    source_audit: dict[str, Any] = {}
    input_paths = [collection_dir / "report.json", entities_path, freeze_report]
    for study_id, config in PARTITION_SOURCES.items():
        evidence, audit, path = load_partition_evidence(study_id, config)
        evidence["study_id"] = study_id
        evidence_blocks.append(evidence)
        source_audit[study_id] = audit
        input_paths.append(path)
    for study_id, config in IBD_SOURCES.items():
        evidence, audit, path = load_ibd_evidence(study_id, config)
        evidence["study_id"] = study_id
        evidence_blocks.append(evidence)
        source_audit[study_id] = audit
        input_paths.append(path)
    evidence = pd.concat(evidence_blocks, ignore_index=True)
    strict = entity_registry.merge(evidence, on=["study_id", "raw_entity_id"], how="left", validate="one_to_one")
    strict["is_collection_library_supported"] = strict["identity_status"].eq("library_supported")
    strict["strict_planar_eligible"] = strict["is_collection_library_supported"] & strict["planar_inchikey"].notna()
    # A strict calibration identity must be supported by both the frozen
    # collection partition and a valid exact planar key.
    strict_eligible = strict.loc[strict["strict_planar_eligible"]].copy()
    for study_id, block in strict.groupby("study_id"):
        known = block["is_collection_library_supported"]
        mapped = block["strict_planar_eligible"]
        source_audit[study_id].update({
            "collection_unique_entities": int(len(block)),
            "collection_library_supported_entities": int(known.sum()),
            "collection_library_supported_with_strict_planar_key": int(mapped.sum()),
            "collection_library_supported_excluded_without_strict_key": int((known & ~mapped).sum()),
        })

    shared_keys = (
        strict_eligible.groupby("planar_inchikey")
        .agg(
            source_family_count=("source_family", "nunique"),
            study_count=("study_id", "nunique"),
            entity_count=("entity_id", "nunique"),
        )
        .reset_index()
    )
    shared_keys = shared_keys.loc[shared_keys["source_family_count"] >= 2].copy()
    shared_set = set(shared_keys["planar_inchikey"])
    shared_members = strict_eligible.loc[strict_eligible["planar_inchikey"].isin(shared_set)].copy()
    summary_extra = (
        shared_members.groupby("planar_inchikey")
        .agg(
            studies=("study_id", lambda values: ";".join(sorted(set(values)))),
            source_families=("source_family", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    shared_keys = shared_keys.merge(summary_extra, on="planar_inchikey", validate="one_to_one")
    pairs = directed_cross_source_pairs(shared_members)
    response_ledger = response.merge(
        shared_members[["study_id", "entity_id", "planar_inchikey", "evidence_source"]],
        on=["study_id", "entity_id"],
        how="inner",
        validate="many_to_one",
    )
    if not response_ledger[["beta", "se"]].notna().all().all() or (response_ledger["se"] <= 0).any():
        raise RuntimeError("shared-identity response ledger has invalid beta/SE")
    if pairs.empty:
        raise RuntimeError("no cross-source exact planar identities were found")

    output_dir.mkdir(parents=True)
    paths = {
        "strict_entity_identity_ledger.tsv.gz": strict.sort_values(
            ["study_id", "entity_id"], kind="stable"
        ),
        "shared_planar_identities.tsv": shared_keys.sort_values(
            ["planar_inchikey"], kind="stable"
        ),
        "shared_identity_members.tsv.gz": shared_members.sort_values(
            ["planar_inchikey", "study_id", "entity_id"], kind="stable"
        ),
        "directed_cross_source_entity_truth_pairs.tsv.gz": pairs.sort_values(
            ["planar_inchikey", "left_study_id", "left_entity_id", "right_study_id", "right_entity_id"],
            kind="stable",
        ),
        "shared_identity_response_ledger.tsv.gz": response_ledger.sort_values(
            ["planar_inchikey", "study_id", "contrast_id", "entity_id"], kind="stable"
        ),
    }
    for name, frame in paths.items():
        compression: str | dict[str, Any] | None = (
            {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        )
        frame.to_csv(output_dir / name, sep="\t", index=False, compression=compression)

    unordered_source_pairs = pairs.apply(
        lambda row: tuple(sorted((row["left_source_family"], row["right_source_family"]))), axis=1
    ).nunique()
    report = {
        "schema": "contrast_atlas_exact_planar_identity_posttruth_calibration_v1",
        "status": "DEVELOPMENT_CONSUMED_POSTTRUTH_EXACT_IDENTITY_CALIBRATION_COMPLETE",
        "opened_after_truth_blind_graph_and_score_freeze": True,
        "freeze_audit": freeze_audit,
        "join_policy": {
            "identity": "strict nonempty planar InChIKey from a frozen library-supported annotation",
            "entity_join": "exact study_id plus source-local raw_entity_id only",
            "forbidden_fallbacks": ["compound name", "precursor m/z", "retention time", "spectral score retuning"],
            "ambiguous_entity_policy": "exclude any source entity with conflicting strict planar keys",
        },
        "denominators": {
            "collection_source_families": int(response["source_family"].nunique()),
            "collection_contrasts": int(response[["study_id", "contrast_id"]].drop_duplicates().shape[0]),
            "collection_response_rows": int(len(response)),
            "collection_library_supported_unique_entities": int(entity_registry["identity_status"].eq("library_supported").sum()),
            "strict_planar_mapped_collection_entities": int(len(strict_eligible)),
            "shared_planar_identities": int(len(shared_keys)),
            "entities_in_shared_planar_identities": int(shared_members["entity_id"].nunique()),
            "source_family_pairs_with_truth": int(unordered_source_pairs),
            "directed_cross_source_entity_truth_pairs": int(len(pairs)),
            "shared_identity_response_rows": int(len(response_ledger)),
        },
        "per_source": source_audit,
        "ibd_boundary": {
            study_id: {
                "strict_planar_mapped": source_audit[study_id]["collection_library_supported_with_strict_planar_key"],
                "excluded_without_strict_key": source_audit[study_id]["collection_library_supported_excluded_without_strict_key"],
            }
            for study_id in IBD_SOURCES
        },
        "claim_boundary": [
            "This ledger is post-truth calibration for exact spectral entity retrieval only.",
            "It is not biological relation truth and does not label similar or opposite responses.",
            "It cannot establish value of unresolved entities because every truth item is library supported.",
            "It must never be read during graph construction, score fitting, threshold selection, or method choice.",
            "Planar InChIKey establishes connectivity identity, not stereochemical identity or unique analyte counting.",
        ],
        "input_sha256": {
            str(path.relative_to(ROOT).as_posix()): sha256(path)
            for path in sorted(set(input_paths))
        },
        "output_sha256": {name: sha256(output_dir / name) for name in paths},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--freeze-report", type=Path, default=DEFAULT_FREEZE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.collection_dir, args.freeze_report, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

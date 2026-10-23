#!/usr/bin/env python3
"""Append new sources/relations without rebuilding a corrected frozen base.

This protects post-contract source corrections (notably the IBD200 Plate3
restriction) that cannot be reconstructed by replaying an older base contract.
The extension build is used only as a staging source for explicitly new studies
and relation rows; every pre-existing study row comes byte-logically from the
frozen base collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_contrast_atlas_open_response_collection import sha256


FILES = (
    "response_entities.tsv.gz",
    "source_manifest.tsv",
    "contrast_manifest.tsv",
    "registered_relations.tsv",
)


def expected_hashes(report: dict[str, object]) -> dict[str, str]:
    hashes = report.get("output_sha256") or report.get("outputs_sha256")
    if not isinstance(hashes, dict):
        raise RuntimeError("collection report lacks output hashes")
    return {str(key): str(value) for key, value in hashes.items()}


def verify_collection(root: Path) -> dict[str, object]:
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = expected_hashes(report)
    for name in FILES:
        path = root / name
        if not path.is_file() or hashes.get(name) != sha256(path):
            raise RuntimeError(f"collection checksum drift: {path}")
    return report


def canonical_relation_key(frame: pd.DataFrame) -> pd.Series:
    return frame[[
        "relation_direction",
        "left_study_id",
        "left_contrast_id",
        "right_study_id",
        "right_contrast_id",
    ]].astype(str).agg("\x1f".join, axis=1)


def run(
    base: Path,
    staged_extension: Path,
    extension_contract: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base_report = verify_collection(base)
    staged_report = verify_collection(staged_extension)
    contract = json.loads(extension_contract.read_text(encoding="utf-8"))
    if contract.get("schema") != "contrast_atlas_open_collection_extension_v1":
        raise RuntimeError("not an append-only collection extension contract")
    new_studies = [str(source["study_id"]) for source in contract.get("sources", [])]
    if not new_studies or len(new_studies) != len(set(new_studies)):
        raise RuntimeError("extension has an empty or duplicate source list")

    base_sources = pd.read_csv(base / "source_manifest.tsv", sep="\t", dtype=str)
    staged_sources = pd.read_csv(
        staged_extension / "source_manifest.tsv", sep="\t", dtype=str
    )
    base_studies = set(base_sources["study_id"])
    if base_studies.intersection(new_studies):
        raise RuntimeError("new source IDs overlap the frozen base")
    if set(staged_sources["study_id"]) != base_studies.union(new_studies):
        raise RuntimeError("staged extension source universe is not base plus new studies")
    new_source_rows = staged_sources.loc[staged_sources["study_id"].isin(new_studies)]
    if len(new_source_rows) != len(new_studies):
        raise RuntimeError("staged extension does not contain each new source exactly once")

    base_entities = pd.read_csv(base / "response_entities.tsv.gz", sep="\t", low_memory=False)
    staged_entities = pd.read_csv(
        staged_extension / "response_entities.tsv.gz", sep="\t", low_memory=False
    )
    new_entities = staged_entities.loc[staged_entities["study_id"].isin(new_studies)]
    if new_entities.empty:
        raise RuntimeError("staged extension contains no new response entities")
    responses = pd.concat([base_entities, new_entities], ignore_index=True)

    base_contrasts = pd.read_csv(base / "contrast_manifest.tsv", sep="\t")
    staged_contrasts = pd.read_csv(staged_extension / "contrast_manifest.tsv", sep="\t")
    new_contrasts = staged_contrasts.loc[staged_contrasts["study_id"].isin(new_studies)]
    contrasts = pd.concat([base_contrasts, new_contrasts], ignore_index=True)
    contrasts["source_disjoint_candidate_contrasts"] = [
        int((contrasts["source_family"].astype(str) != str(row.source_family)).sum())
        for row in contrasts.itertuples(index=False)
    ]
    sources = pd.concat([base_sources, new_source_rows], ignore_index=True)

    base_relations = pd.read_csv(base / "registered_relations.tsv", sep="\t", dtype=str)
    staged_relations = pd.read_csv(
        staged_extension / "registered_relations.tsv", sep="\t", dtype=str
    )
    base_keys = set(canonical_relation_key(base_relations))
    staged_keys = canonical_relation_key(staged_relations)
    new_relations = staged_relations.loc[~staged_keys.isin(base_keys)].copy()
    relations = pd.concat([base_relations, new_relations], ignore_index=True)
    relation_keys = canonical_relation_key(relations)
    if relation_keys.duplicated().any():
        raise RuntimeError("duplicate directed registered relation after append")

    if responses.duplicated(["study_id", "contrast_id", "raw_entity_id"]).any():
        raise RuntimeError("duplicate response row after append")
    if sources["study_id"].duplicated().any():
        raise RuntimeError("duplicate source after append")
    if contrasts.duplicated(["study_id", "contrast_id"]).any():
        raise RuntimeError("duplicate contrast after append")
    contrast_keys = set(zip(contrasts["study_id"], contrasts["contrast_id"]))
    for row in relations.itertuples(index=False):
        if (row.left_study_id, row.left_contrast_id) not in contrast_keys:
            raise RuntimeError("registered left endpoint absent after append")
        if (row.right_study_id, row.right_contrast_id) not in contrast_keys:
            raise RuntimeError("registered right endpoint absent after append")

    output_dir.mkdir(parents=True)
    paths = {
        "response_entities.tsv.gz": output_dir / "response_entities.tsv.gz",
        "source_manifest.tsv": output_dir / "source_manifest.tsv",
        "contrast_manifest.tsv": output_dir / "contrast_manifest.tsv",
        "registered_relations.tsv": output_dir / "registered_relations.tsv",
    }
    responses.to_csv(
        paths["response_entities.tsv.gz"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    sources.to_csv(paths["source_manifest.tsv"], sep="\t", index=False)
    contrasts.to_csv(paths["contrast_manifest.tsv"], sep="\t", index=False)
    relations.to_csv(paths["registered_relations.tsv"], sep="\t", index=False)
    output_hashes = {name: sha256(path) for name, path in paths.items()}
    report = {
        "schema": "contrast_atlas_frozen_base_append_v1",
        "status": "DEVELOPMENT_CONSUMED_FROZEN_BASE_EXTENSION_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "independent_studies": int(sources["study_id"].nunique()),
            "source_families": int(sources["source_family"].nunique()),
            "study_level_contrasts": len(contrasts),
            "eligible_response_rows": len(responses),
            "unique_study_entities": int(
                responses[["study_id", "entity_id"]].drop_duplicates().shape[0]
            ),
            "base_response_rows_preserved": len(base_entities),
            "new_response_rows": len(new_entities),
            "registered_directional_relations": len(relations),
            "new_registered_directional_relations": len(new_relations),
            "source_disjoint_candidates_per_query_min": int(
                contrasts["source_disjoint_candidate_contrasts"].min()
            ),
            "source_disjoint_candidates_per_query_max": int(
                contrasts["source_disjoint_candidate_contrasts"].max()
            ),
        },
        "frozen_base_protection": {
            "rule": "all pre-existing source, contrast and response rows come from the frozen base; only explicitly new studies are selected from the staged contract replay",
            "base_status": base_report.get("status"),
            "staged_status": staged_report.get("status"),
            "new_studies": new_studies,
        },
        "leakage_status": (
            "mechanical append by registered study IDs; no candidate score, rank, response relation outcome, or model prediction was read"
        ),
        "provenance": {
            "base_report_sha256": sha256(base / "report.json"),
            "staged_extension_report_sha256": sha256(staged_extension / "report.json"),
            "extension_contract_sha256": sha256(extension_contract),
            **{f"base_{name}": sha256(base / name) for name in FILES},
            **{
                f"staged_extension_{name}": sha256(staged_extension / name)
                for name in FILES
            },
            "implementation_sha256": sha256(Path(__file__)),
        },
        "output_sha256": output_hashes,
        "claim_boundary": [
            "The pre-existing IBD200 Plate3 correction is preserved and not reconstructed from its older contract artifact.",
            "New exact-drug relations are development/consumed and come from one broader ChemProp project.",
            "Post-hoc obesity relations remain marked ineligible for primary relevance.",
            "This append operation contains no retrieval-performance claim.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--staged-extension", required=True, type=Path)
    parser.add_argument("--extension-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        args.base, args.staged_extension, args.extension_contract, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()

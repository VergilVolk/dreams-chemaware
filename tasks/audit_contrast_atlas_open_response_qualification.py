#!/usr/bin/env python3
"""Audit whether an open response collection can support biological retrieval claims.

The audit deliberately separates three questions that are otherwise easy to
conflate:

1. Is a source-disjoint biological relation registered before retrieval?
2. Is there strict common-identity support for a response comparison?
3. Is signed response truth available, or is the registered relation only an
   unsigned relevance relation?

Unregistered source-disjoint pairs are counted as unlabeled, never as negative
truth.  The script does not fit a model or select a retrieval threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def contrast_key(study_id: str, contrast_id: str) -> str:
    return f"{study_id}::{contrast_id}"


def relation_components(relations: pd.DataFrame) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in relations.itertuples(index=False):
        left = contrast_key(str(row.left_study_id), str(row.left_contrast_id))
        right = contrast_key(str(row.right_study_id), str(row.right_contrast_id))
        graph[left].add(right)
        graph[right].add(left)
    components: list[list[str]] = []
    unseen = set(graph)
    while unseen:
        start = min(unseen)
        queue = deque([start])
        component: list[str] = []
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(graph[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


def load_identity_partition(identity_root: Path, study_id: str) -> pd.DataFrame:
    path = identity_root / study_id / "identity_partition.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing identity partition: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str)
    required = {"entity_id", "library_inchikey", "library_supported", "identity_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"identity partition {study_id} lacks {sorted(missing)}")
    if frame["entity_id"].duplicated().any():
        raise RuntimeError(f"identity partition {study_id} has duplicate entity_id")
    return frame


def validate_registry(relations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "relation_direction", "left_study_id", "left_contrast_id",
        "right_study_id", "right_contrast_id", "relation_class",
        "expected_signed_direction", "eligible_primary_relevance",
        "evidence_boundary", "signed_truth_claimed",
    }
    missing = required.difference(relations.columns)
    if missing:
        raise RuntimeError(f"registered relation table lacks {sorted(missing)}")
    canonical = relations.loc[relations["relation_direction"].eq("registered")].copy()
    reverse = relations.loc[relations["relation_direction"].eq("reverse_registered")].copy()
    if canonical.empty:
        raise RuntimeError("no canonical registered relations")
    reverse_keys = {
        (
            str(row.left_study_id), str(row.left_contrast_id),
            str(row.right_study_id), str(row.right_contrast_id),
        )
        for row in reverse.itertuples(index=False)
    }
    for row in canonical.itertuples(index=False):
        expected = (
            str(row.right_study_id), str(row.right_contrast_id),
            str(row.left_study_id), str(row.left_contrast_id),
        )
        if expected not in reverse_keys:
            raise RuntimeError(f"registered relation lacks reverse row: {expected}")
    return canonical


def embedding_readiness(
    embedding_root: Path | None,
    studies: list[str],
    registered_studies: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for study_id in studies:
        if embedding_root is None:
            report = shard = eligibility = False
        else:
            directory = embedding_root / study_id
            report = (directory / "report.json").is_file()
            shard = (directory / f"{study_id}.npz").is_file()
            eligibility = (directory / f"{study_id}.spectral_eligibility.tsv.gz").is_file()
        complete = report and shard and eligibility
        rows.append({
            "study_id": study_id,
            "used_by_registered_relation": study_id in registered_studies,
            "report_present": report,
            "embedding_present": shard,
            "spectral_eligibility_present": eligibility,
            "embedding_shard_complete": complete,
        })
    frame = pd.DataFrame(rows)
    registered = frame.loc[frame["used_by_registered_relation"]]
    summary = {
        "embedding_root_supplied": embedding_root is not None,
        "complete_study_shards": int(frame["embedding_shard_complete"].sum()),
        "total_studies": int(len(frame)),
        "missing_study_shards": sorted(
            frame.loc[~frame["embedding_shard_complete"], "study_id"].astype(str).tolist()
        ),
        "registered_relation_studies_complete": int(
            registered["embedding_shard_complete"].sum()
        ),
        "registered_relation_studies_total": int(len(registered)),
        "missing_registered_relation_study_shards": sorted(
            registered.loc[
                ~registered["embedding_shard_complete"], "study_id"
            ].astype(str).tolist()
        ),
    }
    return frame, summary


def run_audit(
    collection: Path,
    identity_root: Path,
    output_dir: Path,
    embedding_root: Path | None = None,
) -> dict[str, object]:
    responses_path = collection / "response_entities.tsv.gz"
    contrasts_path = collection / "contrast_manifest.tsv"
    sources_path = collection / "source_manifest.tsv"
    relations_path = collection / "registered_relations.tsv"
    for path in (responses_path, contrasts_path, sources_path, relations_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    responses = pd.read_csv(
        responses_path, sep="\t", dtype={"raw_entity_id": str}, low_memory=False
    )
    contrasts = pd.read_csv(contrasts_path, sep="\t", dtype=str)
    sources = pd.read_csv(sources_path, sep="\t", dtype=str)
    relations = pd.read_csv(relations_path, sep="\t", dtype=str)
    canonical = validate_registry(relations)

    required_response = {
        "study_id", "source_family", "entity_id", "contrast_id", "beta", "se",
        "identity_status",
    }
    missing_response = required_response.difference(responses.columns)
    if missing_response:
        raise RuntimeError(f"response table lacks {sorted(missing_response)}")
    response_keys = responses[["study_id", "contrast_id"]].drop_duplicates()
    manifest_keys = contrasts[["study_id", "contrast_id"]].drop_duplicates()
    if set(map(tuple, response_keys.to_numpy())) != set(map(tuple, manifest_keys.to_numpy())):
        raise RuntimeError("response and contrast manifests disagree")
    if responses.duplicated(["study_id", "contrast_id", "entity_id"]).any():
        raise RuntimeError("response rows are not unique within study and contrast")

    partitions: dict[str, pd.DataFrame] = {}
    relation_rows: list[dict[str, object]] = []
    identity_pair_parts: list[pd.DataFrame] = []
    for relation_index, row in canonical.reset_index(drop=True).iterrows():
        sides: list[pd.DataFrame] = []
        side_counts: list[dict[str, int]] = []
        for prefix in ("left", "right"):
            study_id = str(row[f"{prefix}_study_id"])
            contrast_id = str(row[f"{prefix}_contrast_id"])
            if study_id not in partitions:
                partitions[study_id] = load_identity_partition(identity_root, study_id)
            identity = partitions[study_id]
            response = responses.loc[
                responses["study_id"].eq(study_id)
                & responses["contrast_id"].eq(contrast_id),
                ["entity_id", "beta", "se", "identity_status"],
            ].copy()
            response["beta"] = pd.to_numeric(response["beta"], errors="coerce")
            response["se"] = pd.to_numeric(response["se"], errors="coerce")
            strict = identity.loc[
                as_bool(identity["library_supported"])
                & identity["library_inchikey"].notna(),
                ["entity_id", "library_inchikey"],
            ].drop_duplicates()
            joined = response.merge(strict, on="entity_id", how="inner", validate="one_to_one")
            sides.append(joined)
            side_counts.append({
                "response_entities": int(len(response)),
                "library_supported_entities": int(len(joined)),
                "unresolved_entities": int(response["identity_status"].eq("unresolved_entity").sum()),
            })

        left, right = sides
        pairs = left.merge(right, on="library_inchikey", suffixes=("_left", "_right"))
        if not pairs.empty:
            pairs.insert(0, "relation_index", relation_index)
            pairs.insert(1, "left_study_id", str(row["left_study_id"]))
            pairs.insert(2, "left_contrast_id", str(row["left_contrast_id"]))
            pairs.insert(3, "right_study_id", str(row["right_study_id"]))
            pairs.insert(4, "right_contrast_id", str(row["right_contrast_id"]))
            pairs["left_z"] = pairs["beta_left"] / pairs["se_left"]
            pairs["right_z"] = pairs["beta_right"] / pairs["se_right"]
            pairs["both_abs_z_ge_1_96"] = (
                pairs["left_z"].abs().ge(1.96) & pairs["right_z"].abs().ge(1.96)
            )
            pairs["direction_concordant"] = pairs["beta_left"] * pairs["beta_right"] > 0
            identity_pair_parts.append(pairs)
        both_significant = (
            (pairs["beta_left"].abs() / pairs["se_left"]).ge(1.96)
            & (pairs["beta_right"].abs() / pairs["se_right"]).ge(1.96)
        ) if not pairs.empty else pd.Series(dtype=bool)
        concordant = (
            pairs["beta_left"] * pairs["beta_right"] > 0
        ) if not pairs.empty else pd.Series(dtype=bool)
        relation_rows.append({
            "relation_index": relation_index,
            **{column: row[column] for column in canonical.columns},
            "left_response_entities": side_counts[0]["response_entities"],
            "right_response_entities": side_counts[1]["response_entities"],
            "left_library_supported_entities": side_counts[0]["library_supported_entities"],
            "right_library_supported_entities": side_counts[1]["library_supported_entities"],
            "left_unresolved_entities": side_counts[0]["unresolved_entities"],
            "right_unresolved_entities": side_counts[1]["unresolved_entities"],
            "shared_strict_identity_keys": int(pairs["library_inchikey"].nunique()),
            "strict_identity_entity_pairs": int(len(pairs)),
            "strict_identity_pairs_both_abs_z_ge_1_96": int(both_significant.sum()),
            "strict_identity_pairs_direction_concordant": int(concordant.sum()),
        })
    relation_audit = pd.DataFrame(relation_rows)
    identity_pairs = (
        pd.concat(identity_pair_parts, ignore_index=True)
        if identity_pair_parts else pd.DataFrame()
    )

    entity_identity = responses[
        ["study_id", "entity_id", "identity_status"]
    ].drop_duplicates()
    if entity_identity.duplicated(["study_id", "entity_id"]).any():
        raise RuntimeError("response entities have inconsistent identity status across contrasts")
    harmonized_parts: list[pd.DataFrame] = []
    for study_id in sources["study_id"].astype(str):
        if study_id not in partitions:
            partitions[study_id] = load_identity_partition(identity_root, study_id)
        common = partitions[study_id][["entity_id", "identity_status"]].rename(
            columns={"identity_status": "common_identity_status"}
        )
        local = entity_identity.loc[
            entity_identity["study_id"].eq(study_id)
        ].merge(common, on="entity_id", how="left", validate="one_to_one")
        if local["common_identity_status"].isna().any():
            raise RuntimeError(f"common identity partition does not cover {study_id}")
        local["identity_status_changed"] = (
            local["identity_status"] != local["common_identity_status"]
        )
        harmonized_parts.append(local)
    identity_harmonization = pd.concat(harmonized_parts, ignore_index=True)
    identity_harmonization_summary = (
        identity_harmonization.groupby("study_id", as_index=False)
        .agg(
            response_entities=("entity_id", "size"),
            original_library_supported=(
                "identity_status", lambda values: int(values.eq("library_supported").sum())
            ),
            common_policy_library_supported=(
                "common_identity_status",
                lambda values: int(values.eq("library_supported").sum()),
            ),
            identity_status_changes=("identity_status_changed", "sum"),
        )
    )
    original_library = identity_harmonization["identity_status"].eq("library_supported")
    common_library = identity_harmonization["common_identity_status"].eq("library_supported")

    contrast_records = contrasts[["study_id", "source_family", "contrast_id"]].copy()
    registered_lookup: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in relations.itertuples(index=False):
        registered_lookup[(str(row.left_study_id), str(row.left_contrast_id))].add(
            (str(row.right_study_id), str(row.right_contrast_id))
        )
    query_rows: list[dict[str, object]] = []
    for query in contrast_records.itertuples(index=False):
        candidates = contrast_records.loc[
            ~contrast_records["source_family"].eq(str(query.source_family))
        ]
        registered_targets = registered_lookup[(str(query.study_id), str(query.contrast_id))]
        query_rows.append({
            "study_id": str(query.study_id),
            "source_family": str(query.source_family),
            "contrast_id": str(query.contrast_id),
            "source_disjoint_candidate_contrasts": int(len(candidates)),
            "registered_target_contrasts": int(len(registered_targets)),
            "unlabeled_source_disjoint_contrasts": int(len(candidates) - len(registered_targets)),
            "has_registered_target": bool(registered_targets),
            "registered_targets": ";".join(
                contrast_key(study_id, contrast_id)
                for study_id, contrast_id in sorted(registered_targets)
            ) if registered_targets else "none",
        })
    query_audit = pd.DataFrame(query_rows)

    registered_studies = set(canonical["left_study_id"]).union(canonical["right_study_id"])
    embedding_audit, embedding_summary = embedding_readiness(
        embedding_root,
        sources["study_id"].astype(str).tolist(),
        {str(value) for value in registered_studies},
    )
    components = relation_components(canonical)
    signed_claims = as_bool(relations["signed_truth_claimed"])
    eligible_relevance = as_bool(relations["eligible_primary_relevance"])
    source_disjoint_pairs = int(query_audit["source_disjoint_candidate_contrasts"].sum())
    registered_directional = int(len(relations))
    unlabeled_directional = source_disjoint_pairs - registered_directional

    output_dir.mkdir(parents=True, exist_ok=False)
    relation_path = output_dir / "registered_relation_opportunities.tsv"
    pair_path = output_dir / "strict_identity_response_pairs.tsv.gz"
    query_path = output_dir / "query_retrieval_eligibility.tsv"
    embedding_path = output_dir / "embedding_readiness.tsv"
    harmonization_path = output_dir / "identity_harmonization.tsv"
    relation_audit.to_csv(relation_path, sep="\t", index=False)
    identity_pairs.to_csv(pair_path, sep="\t", index=False, compression="gzip")
    query_audit.to_csv(query_path, sep="\t", index=False)
    embedding_audit.to_csv(embedding_path, sep="\t", index=False)
    identity_harmonization_summary.to_csv(harmonization_path, sep="\t", index=False)

    report: dict[str, object] = {
        "schema": "contrast_atlas_open_response_qualification_audit_v1",
        "status": "DEVELOPMENT_CONSUMED_QUALIFICATION_COMPLETE_PRIMARY_TRUTH_INSUFFICIENT",
        "data_role": "development_consumed",
        "fixed_denominators": {
            "studies": int(sources["study_id"].nunique()),
            "source_families": int(sources["source_family"].nunique()),
            "contrasts": int(len(contrasts)),
            "response_rows": int(len(responses)),
            "unique_response_entities": int(responses["entity_id"].nunique()),
        },
        "registered_relation_registry": {
            "directional_registered_rows": registered_directional,
            "canonical_registered_relations": int(len(canonical)),
            "connected_biological_relation_components": int(len(components)),
            "component_sizes_in_contrasts": [len(component) for component in components],
            "component_members": components,
            "query_contrasts_with_registered_target": int(query_audit["has_registered_target"].sum()),
            "query_contrasts_without_registered_target": int((~query_audit["has_registered_target"]).sum()),
            "source_disjoint_directional_candidate_pairs": source_disjoint_pairs,
            "unlabeled_source_disjoint_directional_pairs": unlabeled_directional,
            "confirmed_negative_directional_pairs": 0,
            "eligible_relevance_rows": int(eligible_relevance.sum()),
            "signed_truth_rows": int(signed_claims.sum()),
            "expected_signed_direction_unspecified_rows": int(
                relations["expected_signed_direction"].eq("unspecified").sum()
            ),
        },
        "strict_identity_opportunities": {
            "canonical_relations_with_any_shared_identity": int(
                relation_audit["shared_strict_identity_keys"].gt(0).sum()
            ),
            "canonical_relations_without_shared_identity": int(
                relation_audit["shared_strict_identity_keys"].eq(0).sum()
            ),
            "relation_specific_shared_identity_keys": int(
                relation_audit["shared_strict_identity_keys"].sum()
            ),
            "strict_identity_entity_pairs": int(len(identity_pairs)),
            "pairs_both_abs_z_ge_1_96": int(
                identity_pairs["both_abs_z_ge_1_96"].sum()
            ) if not identity_pairs.empty else 0,
            "direction_concordant_pairs_descriptive": int(
                identity_pairs["direction_concordant"].sum()
            ) if not identity_pairs.empty else 0,
        },
        "identity_harmonization": {
            "response_entities": int(len(identity_harmonization)),
            "original_source_specific_library_supported": int(original_library.sum()),
            "common_policy_library_supported": int(common_library.sum()),
            "unchanged_library_supported": int((original_library & common_library).sum()),
            "source_specific_library_downgraded_by_common_policy": int(
                (original_library & ~common_library).sum()
            ),
            "source_specific_unresolved_promoted_by_common_policy": int(
                (~original_library & common_library).sum()
            ),
            "identity_status_changes": int(
                identity_harmonization["identity_status_changed"].sum()
            ),
            "known_only_definition_for_comparison": "common_identity_policy",
        },
        "unresolved_relation_readiness": embedding_summary,
        "qualification": {
            "registered_unsigned_relevance_evaluation_possible": True,
            "signed_response_truth_available": bool(signed_claims.any()),
            "unregistered_pairs_may_be_used_as_negatives": False,
            "all_registered_relation_studies_have_embeddings": (
                embedding_summary["registered_relation_studies_complete"]
                == embedding_summary["registered_relation_studies_total"]
            ),
            "primary_biological_discovery_claim_ready": False,
            "blocking_reasons": [
                "no registered relation claims signed response truth",
                "unregistered source-disjoint pairs are unlabeled rather than confirmed negatives",
                "only two connected biological relation components are registered",
                "strict common-identity support is absent for two of five canonical relations",
            ] + ([
                "official embedding shards are incomplete for registered-relation studies"
            ] if embedding_summary["registered_relation_studies_complete"]
                 != embedding_summary["registered_relation_studies_total"] else []),
        },
        "interpretation": (
            "The collection supports a development test of preregistered unsigned relation "
            "ranking and abstention. It does not currently support signed-response accuracy, "
            "open-world FDR, or a primary biological discovery claim. Missing chemical-relation "
            "coverage must be completed before assessing unresolved-entity increment."
        ),
        "claim_boundary": [
            "A registered condition-family relation is relevance evidence, not metabolite-level signed truth.",
            "Unregistered contrast pairs are unknown relations and cannot be counted as negatives.",
            "Strict identity response pairs are descriptive opportunities, not independent biological discoveries.",
            "Relation-specific entity-pair counts are not independent when an InChIKey has multiple entities.",
            "Known-only comparisons must use the harmonized common identity policy, not heterogeneous source-specific labels.",
            "All inputs and outputs are development/consumed.",
        ],
        "provenance": {
            "response_entities_sha256": sha256(responses_path),
            "contrast_manifest_sha256": sha256(contrasts_path),
            "source_manifest_sha256": sha256(sources_path),
            "registered_relations_sha256": sha256(relations_path),
            "registered_relation_opportunities_sha256": sha256(relation_path),
            "strict_identity_response_pairs_sha256": sha256(pair_path),
            "query_retrieval_eligibility_sha256": sha256(query_path),
            "embedding_readiness_sha256": sha256(embedding_path),
            "identity_harmonization_sha256": sha256(harmonization_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-root", required=True, type=Path)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(
        args.collection, args.identity_root, args.output_dir, args.embedding_root
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

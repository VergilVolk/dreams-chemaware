#!/usr/bin/env python3
"""Trace candidate-ranker decisions into independent public IBD responses.

Candidate generation and scoring remain phenotype blind.  A selected candidate
may link an IBD200 query to strict library-supported response entities in the
two prespecified independent IBD studies.  The output retains all queries and
reports missing scores or absent public anchors as abstentions.

The response links are evidence consequences, not correctness labels: the
public studies differ in age and disease composition, and strict library
support is not MSI level-1 identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_ibd200_candidate_context_evidence import candidate_axis


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_score_spec(spec: str) -> tuple[str, Path, str]:
    if "=" not in spec or ":" not in spec.rsplit("=", 1)[1]:
        raise ValueError("--score must be METHOD=PATH:FIELD")
    method, payload = spec.split("=", 1)
    raw_path, field = payload.rsplit(":", 1)
    if not method or not raw_path or not field:
        raise ValueError("empty method, path or field in --score")
    return method, Path(raw_path), field


def load_scores(
    spec: tuple[str, Path, str],
    reference_axis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[str, np.ndarray, Path, str]:
    method, path, field = spec
    if not path.is_file():
        raise FileNotFoundError(path)
    axis = candidate_axis(path)
    for observed, expected in zip(axis, reference_axis):
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"candidate axis differs for method {method}")
    bundle = np.load(path, allow_pickle=False)
    if field not in bundle.files:
        raise RuntimeError(f"score field {field} is absent for method {method}")
    values = np.asarray(bundle[field], dtype=np.float64)
    if values.shape != (len(reference_axis[2]),):
        raise RuntimeError(f"score field length differs for method {method}")
    return method, values, path, field


def bool_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-shards-root", required=True, type=Path)
    parser.add_argument("--adduct-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--score", action="append", required=True,
        help="Candidate score METHOD=PATH:FIELD; repeat for B/integrated arms.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    reference_path = args.slice_dir / "wse_candidate_scores.npz"
    candidate_structures_path = args.slice_dir / "candidate_structures.tsv.gz"
    effects_path = args.slice_dir / "sealed_heldout_effects.tsv.gz"
    responses_path = args.collection / "response_entities.tsv.gz"
    relations_path = args.collection / "registered_relations.tsv"
    for path in (
        reference_path, candidate_structures_path, effects_path, responses_path,
        relations_path, args.adduct_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    axis = candidate_axis(reference_path)
    query_ptr, query_ids, candidate_ids = axis
    edge_queries = np.repeat(query_ids, np.diff(query_ptr))
    score_specs = [parse_score_spec(spec) for spec in args.score]
    if len({method for method, _, _ in score_specs}) != len(score_specs):
        raise RuntimeError("score method names must be unique")
    scores = [load_scores(spec, axis) for spec in score_specs]

    relations = pd.read_csv(relations_path, sep="\t")
    eligible = bool_column(relations["eligible_primary_relevance"])
    selected_relations = relations.loc[
        eligible & relations["left_study_id"].astype(str).eq("IBD200_GROUP")
    ].copy()
    if selected_relations.empty:
        raise RuntimeError("no prespecified IBD200 public response relations")
    if bool_column(selected_relations["signed_truth_claimed"]).any():
        raise RuntimeError("this evaluator expects unsigned design relevance, not signed truth")

    responses = pd.read_csv(responses_path, sep="\t", low_memory=False)
    query_identity_path = args.identity_shards_root / "IBD200_GROUP" / "identity_partition.tsv.gz"
    if not query_identity_path.is_file():
        raise FileNotFoundError(query_identity_path)
    query_identity = pd.read_csv(
        query_identity_path, sep="\t", usecols=["entity_id", "identity_status"], dtype=str,
    ).drop_duplicates("entity_id")
    if set(query_identity["entity_id"]) != set(query_ids):
        raise RuntimeError("IBD query identity partition differs from candidate query axis")
    query_identity_map = dict(zip(query_identity["entity_id"], query_identity["identity_status"]))
    anchor_parts: list[pd.DataFrame] = []
    identity_inputs: dict[str, str] = {}
    for relation in selected_relations.itertuples(index=False):
        study_id = str(relation.right_study_id)
        contrast_id = str(relation.right_contrast_id)
        identity_path = args.identity_shards_root / study_id / "identity_partition.tsv.gz"
        if not identity_path.is_file():
            raise FileNotFoundError(identity_path)
        identity_inputs[study_id] = sha256(identity_path)
        identity = pd.read_csv(identity_path, sep="\t")
        identity = identity.loc[bool_column(identity["library_supported"])].copy()
        identity = identity.loc[identity["library_inchikey"].notna()].copy()
        # One representative response entity per study/contrast/structure.
        identity["wse"] = pd.to_numeric(identity["wse"], errors="coerce")
        identity["matched_peaks"] = pd.to_numeric(identity["matched_peaks"], errors="coerce")
        identity = identity.sort_values(
            ["library_inchikey", "wse", "matched_peaks", "entity_id"],
            ascending=[True, False, False, True], kind="mergesort",
        ).drop_duplicates("library_inchikey", keep="first")
        response = responses.loc[
            responses["study_id"].astype(str).eq(study_id)
            & responses["contrast_id"].astype(str).eq(contrast_id),
            ["study_id", "contrast_id", "entity_id", "beta", "se", "effect_model"],
        ].copy()
        anchor = identity.merge(
            response, on=["study_id", "entity_id"], how="inner", validate="one_to_one",
        )
        anchor["target_contrast_id"] = contrast_id
        anchor["relation_class"] = str(relation.relation_class)
        anchor["relation_evidence_boundary"] = str(relation.evidence_boundary)
        anchor_parts.append(anchor)
    anchors = pd.concat(anchor_parts, ignore_index=True)
    if anchors.duplicated(["study_id", "target_contrast_id", "library_inchikey"]).any():
        raise RuntimeError("public response anchors are not unique")
    anchors = anchors.rename(columns={
        "study_id": "target_study_id",
        "entity_id": "target_entity_id",
        "library_inchikey": "candidate_id",
        "beta": "target_beta",
        "se": "target_se",
        "effect_model": "target_effect_model",
        "wse": "target_anchor_wse",
        "matched_peaks": "target_anchor_matched_peaks",
    })

    edge_axis = pd.DataFrame({
        "edge_index": np.arange(len(candidate_ids), dtype=np.int64),
        "entity_id": edge_queries,
        "candidate_id": candidate_ids,
        "query_identity_status": [query_identity_map[str(query_id)] for query_id in edge_queries],
    })
    candidate_structures = pd.read_csv(candidate_structures_path, sep="\t")
    candidate_structures = candidate_structures[[
        "entity_id", "candidate_id", "smiles", "formula", "exact_mass",
        "query_precursor_mz", "minimum_precursor_ppm",
    ]].copy()
    if candidate_structures.duplicated(["entity_id", "candidate_id"]).any():
        raise RuntimeError("candidate structure table has duplicate query-candidate pairs")
    edge_axis = edge_axis.merge(
        candidate_structures, on=["entity_id", "candidate_id"],
        how="left", validate="one_to_one",
    )
    if edge_axis["formula"].isna().any():
        raise RuntimeError("candidate structure metadata does not cover the score axis")
    adduct = pd.read_csv(args.adduct_audit, sep="\t")
    adduct = adduct[["entity_id", "candidate_id", "adduct_compatibility_status"]]
    edge_axis = edge_axis.merge(
        adduct, on=["entity_id", "candidate_id"], how="left", validate="one_to_one",
    )
    if edge_axis["adduct_compatibility_status"].isna().any():
        raise RuntimeError("adduct audit does not cover the score axis")
    edge_axis["strict_candidate_allowed"] = edge_axis[
        "adduct_compatibility_status"
    ].eq("explicit_neutral_compatible")
    candidate_links = edge_axis.merge(
        anchors[[
            "candidate_id", "target_study_id", "target_contrast_id", "target_entity_id",
            "target_beta", "target_se", "target_effect_model", "target_anchor_wse",
            "target_anchor_matched_peaks", "library_name", "relation_class",
            "relation_evidence_boundary",
        ]],
        on="candidate_id", how="inner", validate="many_to_many",
    )
    candidate_links = candidate_links.loc[candidate_links["strict_candidate_allowed"]].copy()
    opportunity_queries = set(candidate_links["entity_id"].astype(str))

    effects = pd.read_csv(effects_path, sep="\t")
    effects = effects.loc[bool_column(effects["effect_estimable"])].copy()
    effects = effects.rename(columns={"beta": "query_beta", "se": "query_se"})

    top_rows: list[dict[str, object]] = []
    response_link_parts: list[pd.DataFrame] = []
    method_summaries: dict[str, object] = {}
    robust_candidate_parts: list[pd.DataFrame] = []
    candidate_link_index = set(candidate_links["edge_index"].astype(int))
    for method, values, score_path, field in scores:
        selected_edge_indices: list[int] = []
        for query_index, query_id in enumerate(query_ids):
            start, stop = int(query_ptr[query_index]), int(query_ptr[query_index + 1])
            local_scores = values[start:stop]
            local_candidates = candidate_ids[start:stop]
            allowed = edge_axis["strict_candidate_allowed"].to_numpy()[start:stop]
            available = np.isfinite(local_scores) & allowed
            if not available.any():
                top_rows.append({
                    "method": method, "entity_id": str(query_id), "score_available": False,
                    "query_identity_status": query_identity_map[str(query_id)],
                    "top_candidate_id": "", "top_score": np.nan,
                    "public_response_opportunity": str(query_id) in opportunity_queries,
                    "selected_public_response_anchor": False,
                })
                continue
            local_indices = np.flatnonzero(available)
            order = np.lexsort((local_candidates[local_indices], -local_scores[local_indices]))
            edge_index = start + int(local_indices[order[0]])
            selected_edge_indices.append(edge_index)
            top_rows.append({
                "method": method, "entity_id": str(query_id), "score_available": True,
                "query_identity_status": query_identity_map[str(query_id)],
                "top_candidate_id": str(candidate_ids[edge_index]),
                "top_score": float(values[edge_index]),
                "public_response_opportunity": str(query_id) in opportunity_queries,
                "selected_public_response_anchor": edge_index in candidate_link_index,
            })
        method_links = candidate_links.loc[
            candidate_links["edge_index"].isin(selected_edge_indices)
        ].copy()
        method_links.insert(0, "method", method)
        method_links = method_links.merge(
            effects[["outer_fold", "entity_id", "query_beta", "query_se", "n_positive", "n_negative"]],
            on="entity_id", how="left", validate="many_to_many",
        )
        method_links["direction_concordant"] = (
            method_links["query_beta"] * method_links["target_beta"] > 0
        )
        total_se = np.sqrt(method_links["query_se"] ** 2 + method_links["target_se"] ** 2)
        method_links["standardized_effect_difference"] = (
            (method_links["query_beta"] - method_links["target_beta"]) / total_se
        )
        method_links["effect_compatible_95"] = method_links["standardized_effect_difference"].abs() <= 1.96
        response_link_parts.append(method_links)

        unresolved = method_links.loc[
            method_links["query_identity_status"].eq("unresolved_entity")
        ].copy()
        if not unresolved.empty:
            robust = (
                unresolved.groupby(
                    ["method", "entity_id", "candidate_id", "smiles", "formula", "exact_mass"],
                    as_index=False,
                ).agg(
                    target_studies=("target_study_id", "nunique"),
                    heldout_fold_response_links=("direction_concordant", "size"),
                    all_direction_concordant=("direction_concordant", "all"),
                    all_effect_compatible_95=("effect_compatible_95", "all"),
                    mean_query_beta=("query_beta", "mean"),
                    mean_target_beta=("target_beta", "mean"),
                    mean_absolute_standardized_difference=(
                        "standardized_effect_difference", lambda values: float(values.abs().mean())
                    ),
                )
            )
            robust = robust.loc[
                robust["target_studies"].eq(selected_relations["right_study_id"].nunique())
                & robust["heldout_fold_response_links"].eq(
                    selected_relations["right_study_id"].nunique() * effects["outer_fold"].nunique()
                )
                & robust["all_direction_concordant"]
                & robust["all_effect_compatible_95"]
            ].copy()
            robust_candidate_parts.append(robust)

        method_top = pd.DataFrame([row for row in top_rows if row["method"] == method])
        evaluable = method_links["query_beta"].notna() & method_links["target_beta"].notna()
        cluster_concordance = (
            method_links.loc[evaluable]
            .groupby(["candidate_id", "target_study_id"])["direction_concordant"].mean()
        )
        selected_by_status = (
            method_top.loc[method_top["selected_public_response_anchor"]]
            .groupby("query_identity_status").size().to_dict()
        )
        link_concordance_by_status = {}
        for status, group in method_links.loc[evaluable].groupby("query_identity_status"):
            link_concordance_by_status[str(status)] = {
                "heldout_fold_response_links": int(len(group)),
                "descriptive_direction_concordance": float(group["direction_concordant"].mean()),
                "descriptive_effect_compatible_95": float(group["effect_compatible_95"].mean()),
            }
        method_summaries[method] = {
            "score_path": str(score_path),
            "score_field": field,
            "score_sha256": sha256(score_path),
            "answered_queries": int(method_top["score_available"].sum()),
            "queries_with_public_response_opportunity": int(len(opportunity_queries)),
            "selected_public_response_anchor_queries": int(method_top["selected_public_response_anchor"].sum()),
            "selected_public_response_anchor_queries_by_identity_status": {
                str(key): int(value) for key, value in selected_by_status.items()
            },
            "heldout_fold_response_links": int(evaluable.sum()),
            "candidate_structure_target_clusters": int(len(cluster_concordance)),
            "descriptive_direction_concordance": (
                float(method_links.loc[evaluable, "direction_concordant"].mean()) if evaluable.any() else None
            ),
            "structure_target_weighted_direction_concordance": (
                float(cluster_concordance.mean()) if len(cluster_concordance) else None
            ),
            "descriptive_effect_compatible_95": (
                float(method_links.loc[evaluable, "effect_compatible_95"].mean()) if evaluable.any() else None
            ),
            "response_links_by_query_identity_status": link_concordance_by_status,
        }

    top = pd.DataFrame(top_rows)
    response_links = pd.concat(response_link_parts, ignore_index=True) if response_link_parts else pd.DataFrame()
    robust_candidates = (
        pd.concat(robust_candidate_parts, ignore_index=True)
        if robust_candidate_parts else pd.DataFrame()
    )
    args.output_dir.mkdir(parents=True)
    anchors_path = args.output_dir / "public_ibd_response_anchors.tsv.gz"
    candidates_path = args.output_dir / "candidate_public_response_links.tsv.gz"
    top_path = args.output_dir / "method_top_candidates.tsv.gz"
    response_links_path = args.output_dir / "method_heldout_response_links.tsv.gz"
    robust_candidates_path = args.output_dir / "development_unresolved_response_candidates.tsv"
    compression = {"method": "gzip", "mtime": 0}
    anchors.to_csv(anchors_path, sep="\t", index=False, compression=compression)
    candidate_links.to_csv(candidates_path, sep="\t", index=False, compression=compression)
    top.to_csv(top_path, sep="\t", index=False, compression=compression)
    response_links.to_csv(response_links_path, sep="\t", index=False, compression=compression)
    robust_candidates.to_csv(robust_candidates_path, sep="\t", index=False)

    report = {
        "schema": "ibd200_algorithm_public_response_link_evaluation_v1",
        "status": "DESCRIPTIVE_ALGORITHM_TO_PUBLIC_RESPONSE_LINKS_COMPLETE",
        "data_role": "development_consumed",
        "fixed_denominators": {
            "ibd200_queries": int(len(query_ids)),
            "candidate_edges": int(len(candidate_ids)),
            "prespecified_independent_ibd_target_studies": int(anchors["target_study_id"].nunique()),
            "public_response_anchor_entities": int(len(anchors)),
            "public_response_anchor_structures": int(anchors["candidate_id"].nunique()),
            "candidate_edges_with_public_response_anchor": int(candidate_links["edge_index"].nunique()),
            "queries_with_public_response_anchor_opportunity": int(len(opportunity_queries)),
        },
        "method_results": method_summaries,
        "development_unresolved_response_candidates": {
            str(method): int(count)
            for method, count in robust_candidates.groupby("method").size().to_dict().items()
        } if not robust_candidates.empty else {},
        "scientific_contract": {
            "candidate_selection": "scores only; no response effect is available to the ranker",
            "candidate_chemistry": "only explicit FBMN neutral-mass-compatible candidates; unresolved adduct is abstention",
            "target_selection": "two prespecified registered independent IBD-versus-control study relations",
            "response_evaluation": "IBD200 effects are estimated in held-out participant folds; public effects are fixed",
            "future_comparison": "B and integrated scores must preserve the exact query_ptr/query_ids/candidate_ids axis",
        },
        "provenance": {
            "reference_score_bundle_sha256": sha256(reference_path),
            "candidate_structures_sha256": sha256(candidate_structures_path),
            "candidate_adduct_audit_sha256": sha256(args.adduct_audit),
            "sealed_heldout_effects_sha256": sha256(effects_path),
            "response_collection_sha256": sha256(responses_path),
            "registered_relations_sha256": sha256(relations_path),
            "query_identity_partition_sha256": sha256(query_identity_path),
            "identity_partition_sha256": identity_inputs,
            "anchors_sha256": sha256(anchors_path),
            "candidate_links_sha256": sha256(candidates_path),
            "method_top_candidates_sha256": sha256(top_path),
            "method_response_links_sha256": sha256(response_links_path),
            "development_unresolved_candidates_sha256": sha256(robust_candidates_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "Direction concordance is a descriptive response-support endpoint, not identity truth.",
            "The target studies differ in age and IBD subtype; discordance is not automatically an algorithm error.",
            "Strict library support is not MSI level-1 identity, and unresolved query predictions are not metabolite identifications.",
            "Unregistered public studies are neither positives nor negatives and are excluded from the primary response-support summary.",
            "Development unresolved-response candidates are post-hoc leads and require orthogonal identity and prospective biological confirmation.",
            "Queries without an explicit neutral-mass-compatible candidate are strict chemistry abstentions, not negative results.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

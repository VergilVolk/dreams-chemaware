#!/usr/bin/env python3
"""Build a biological panorama and traceable evidence ledger for Contrast Atlas.

This is a descriptive, development/consumed data product.  It combines the
frozen study designs, entity-level beta/SE response objects and one retrieval
release without treating unregistered study pairs as negatives or spectral
identity hypotheses as confirmed molecular identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_contrast_atlas_open_local_edge_enrichment import (
    StudyProfiles,
    add_effect_statistics,
    build_profiles,
)


IDENTITY_TRACK = "same_precursor_10ppm_identity_hypothesis"
WORLDS = ("all", "known", "unresolved")
FACES = ("similar", "opposite")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_project_map(path: Path, studies: set[str]) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    required = {"study_id", "parent_project_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"project map missing {sorted(required - set(frame.columns))}")
    if frame["study_id"].duplicated().any():
        raise ValueError("project map has duplicate study_id values")
    lookup = dict(zip(frame["study_id"], frame["parent_project_id"]))
    if set(lookup) != studies:
        raise ValueError("project map does not exactly cover the response collection")
    return lookup


def response_summary(effects: pd.DataFrame) -> pd.DataFrame:
    frame = effects.copy()
    frame["absolute_beta"] = frame["beta"].abs()
    frame["absolute_z"] = frame["effect_z"].abs()
    frame["absolute_response_weight"] = frame["response_weight"].abs()
    frame["is_library_supported"] = frame["identity_status"].eq("library_supported")
    frame["is_unresolved"] = frame["identity_status"].eq("unresolved_entity")
    frame["diagnostic_up_q005"] = frame["response_qualified"] & frame["beta"].gt(0)
    frame["diagnostic_down_q005"] = frame["response_qualified"] & frame["beta"].lt(0)
    result = (
        frame.groupby(["study_id", "contrast_id"], sort=True)
        .agg(
            response_entities=("entity_id", "size"),
            library_supported_entities=("is_library_supported", "sum"),
            unresolved_entities=("is_unresolved", "sum"),
            positive_beta_entities=("beta", lambda values: int((values > 0).sum())),
            negative_beta_entities=("beta", lambda values: int((values < 0).sum())),
            diagnostic_up_q005=("diagnostic_up_q005", "sum"),
            diagnostic_down_q005=("diagnostic_down_q005", "sum"),
            median_absolute_beta=("absolute_beta", "median"),
            median_absolute_z=("absolute_z", "median"),
            median_absolute_response_weight=("absolute_response_weight", "median"),
            total_absolute_response_weight=("absolute_response_weight", "sum"),
        )
        .reset_index()
    )
    return result


def metadata_for_join(eligibility: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "study_id", "contrast_id", "project_dependence_group", "species",
        "source_documented_project_dependence_group",
        "biospecimen", "condition_family", "estimand", "biological_unit",
        "study_design", "repeated_measures", "independent_units", "time_or_dose",
        "covariates", "intentional_drug_exposure", "causal_status", "effect_model",
        "exclusion_or_limit_reasons",
    ]
    selected = eligibility[columns].copy()
    return selected.rename(columns={column: f"{prefix}_{column}" for column in columns})


def attach_side_metadata(
    frame: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    for prefix in ("query", "candidate"):
        metadata = metadata_for_join(eligibility, prefix)
        result = result.merge(
            metadata,
            how="left",
            left_on=[f"{prefix}_study_id", f"{prefix}_contrast_id"],
            right_on=[f"{prefix}_study_id", f"{prefix}_contrast_id"],
            validate="many_to_one",
        )
    if result[["query_project_dependence_group", "candidate_project_dependence_group"]].isna().any().any():
        raise RuntimeError("failed to attach contrast metadata")
    result["same_parent_project"] = result["query_project_dependence_group"].eq(
        result["candidate_project_dependence_group"]
    )
    result["same_species"] = result["query_species"].eq(result["candidate_species"])
    result["same_biospecimen"] = result["query_biospecimen"].eq(
        result["candidate_biospecimen"]
    )
    result["same_condition_family"] = result["query_condition_family"].eq(
        result["candidate_condition_family"]
    )
    result["biological_comparability_tier"] = np.select(
        [
            result["same_parent_project"],
            result["same_species"] & result["same_biospecimen"]
            & result["same_condition_family"],
            result["same_species"] & result["same_biospecimen"],
            result["same_condition_family"],
        ],
        [
            "same_parent_internal_control_or_shortcut",
            "cross_parent_domain_and_condition_matched",
            "cross_parent_species_and_biospecimen_matched",
            "cross_parent_condition_family_only",
        ],
        default="cross_parent_cross_domain_exploratory",
    )
    return result


def relation_lookup(relations: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"
    ]
    columns = [
        *keys, "relation_class", "expected_signed_direction",
        "eligible_primary_relevance", "evidence_boundary", "signed_truth_claimed",
    ]
    result = relations[columns].drop_duplicates(keys).copy()
    result = result.rename(columns={
        "left_study_id": "query_study_id",
        "left_contrast_id": "query_contrast_id",
        "right_study_id": "candidate_study_id",
        "right_contrast_id": "candidate_contrast_id",
    })
    return result


def attach_relation_status(frame: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "query_study_id", "query_contrast_id", "candidate_study_id",
        "candidate_contrast_id",
    ]
    result = frame.merge(relation_lookup(relations), how="left", on=keys, validate="many_to_one")
    result["relation_evaluation_status"] = np.where(
        result["relation_class"].notna(), "registered_unsigned_relation", "unregistered_unjudged"
    )
    result["relation_class"] = result["relation_class"].fillna("unregistered")
    result["expected_signed_direction"] = result["expected_signed_direction"].fillna(
        "unjudged"
    )
    result["signed_truth_claimed"] = (
        result["signed_truth_claimed"].astype("string").str.lower().eq("true")
    )
    return result


def combined_two_face_answers(top: pd.DataFrame) -> pd.DataFrame:
    selected = top.loc[top["evidence_track"].eq(IDENTITY_TRACK)].copy()
    rows: list[dict[str, object]] = []
    for keys, block in selected.groupby(
        ["query_study_id", "query_contrast_id", "evidence_world"], sort=True
    ):
        study, contrast, world = keys
        if set(block["response_face"]) != set(FACES) or len(block) != 2:
            raise RuntimeError(f"two-face top-hit contract drift for {keys}")
        eligible = block.loc[
            pd.to_numeric(block["weighted_evidence_edges"], errors="coerce").ge(3)
            & pd.to_numeric(block["enrichment_z"], errors="coerce").gt(0)
        ].copy()
        if len(eligible):
            chosen = eligible.sort_values(
                ["query_maxT_p", "response_face", "candidate_study_id", "candidate_contrast_id"]
            ).iloc[0]
            corrected = min(1.0, 2.0 * float(chosen["query_maxT_p"]))
            answer_class = (
                "calibrated_answer" if corrected <= 0.05 else "no_calibrated_match"
            )
        else:
            chosen = block.sort_values(
                ["query_maxT_p", "response_face", "candidate_study_id", "candidate_contrast_id"]
            ).iloc[0]
            corrected = 1.0
            answer_class = "insufficient_evidence"
        row = chosen.to_dict()
        row.update({
            "query_study_id": str(study),
            "query_contrast_id": str(contrast),
            "evidence_world": str(world),
            "two_face_bonferroni_p": corrected,
            "combined_answer_class": answer_class,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 51 * len(WORLDS):
        raise RuntimeError("combined query/world denominator drift")
    return result


def build_world_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    keys = ["query_study_id", "query_contrast_id"]
    value_columns = [
        "combined_answer_class", "response_face", "candidate_study_id",
        "candidate_contrast_id", "weighted_evidence_edges", "enrichment_z",
        "two_face_bonferroni_p",
    ]
    pieces = []
    for world in WORLDS:
        block = combined.loc[combined["evidence_world"].eq(world), [*keys, *value_columns]].copy()
        block = block.rename(columns={column: f"{world}_{column}" for column in value_columns})
        pieces.append(block)
    result = pieces[0]
    for block in pieces[1:]:
        result = result.merge(block, on=keys, validate="one_to_one")
    result["all_unresolved_same_selection"] = (
        result["all_candidate_study_id"].eq(result["unresolved_candidate_study_id"])
        & result["all_candidate_contrast_id"].eq(result["unresolved_candidate_contrast_id"])
        & result["all_response_face"].eq(result["unresolved_response_face"])
    )
    result["all_known_same_selection"] = (
        result["all_candidate_study_id"].eq(result["known_candidate_study_id"])
        & result["all_candidate_contrast_id"].eq(result["known_candidate_contrast_id"])
        & result["all_response_face"].eq(result["known_response_face"])
    )
    result["unresolved_supported_not_known_selection"] = (
        result["all_combined_answer_class"].eq("calibrated_answer")
        & result["unresolved_combined_answer_class"].eq("calibrated_answer")
        & result["all_unresolved_same_selection"]
        & ~(
            result["known_combined_answer_class"].eq("calibrated_answer")
            & result["all_known_same_selection"]
        )
    )
    result["known_corroborated_selection"] = (
        result["all_combined_answer_class"].eq("calibrated_answer")
        & result["known_combined_answer_class"].eq("calibrated_answer")
        & result["all_known_same_selection"]
    )
    return result


def annotate_scores(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    selected = scores.loc[scores["evidence_track"].eq(IDENTITY_TRACK)].copy()
    selected["best_face"] = np.where(
        selected["similar_query_maxT_p"].le(selected["opposite_query_maxT_p"]),
        "similar", "opposite",
    )
    selected["best_face_query_maxT_p"] = selected[
        ["similar_query_maxT_p", "opposite_query_maxT_p"]
    ].min(axis=1)
    selected["two_face_bonferroni_p"] = np.minimum(
        1.0, 2.0 * selected["best_face_query_maxT_p"]
    )
    selected = attach_side_metadata(selected, eligibility)
    selected = attach_relation_status(selected, relations)
    return selected


def top_support_evidence(
    top: pd.DataFrame,
    edges: pd.DataFrame,
    profiles: dict[str, StudyProfiles],
    registry: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    status = registry.set_index(["study_id", "entity_id"])["identity_status"].to_dict()
    profile_entity_rows = {
        study_id: {str(entity): index for index, entity in enumerate(profile.entities)}
        for study_id, profile in profiles.items()
    }
    edge_blocks = {
        (str(left), str(right), str(world)): block.reset_index(drop=True)
        for (left, right, world), block in edges.groupby(
            ["left_study_id", "right_study_id", "evidence_world"], sort=False
        )
    }
    records: list[dict[str, object]] = []
    selected = top.loc[top["evidence_track"].eq(IDENTITY_TRACK)].copy()
    for hit in selected.itertuples(index=False):
        query_study = str(hit.query_study_id)
        candidate_study = str(hit.candidate_study_id)
        low, high = sorted((query_study, candidate_study))
        block = edge_blocks.get((low, high, str(hit.evidence_world)))
        if block is None or not len(block):
            continue
        block = block.loc[block["precursor_compatible_10ppm"].astype(bool)].reset_index(drop=True)
        if query_study == low:
            query_entities = block["left_entity_id"].astype(str).to_numpy()
            candidate_entities = block["right_entity_id"].astype(str).to_numpy()
            query_mz = block["left_precursor_mz"].to_numpy(float)
            candidate_mz = block["right_precursor_mz"].to_numpy(float)
        else:
            query_entities = block["right_entity_id"].astype(str).to_numpy()
            candidate_entities = block["left_entity_id"].astype(str).to_numpy()
            query_mz = block["right_precursor_mz"].to_numpy(float)
            candidate_mz = block["left_precursor_mz"].to_numpy(float)
        query_profile = profiles[query_study]
        candidate_profile = profiles[candidate_study]
        query_rows = np.asarray(
            [profile_entity_rows[query_study][entity] for entity in query_entities],
            dtype=np.int64,
        )
        candidate_rows = np.asarray(
            [profile_entity_rows[candidate_study][entity] for entity in candidate_entities],
            dtype=np.int64,
        )
        query_column = query_profile.contrasts.index(str(hit.query_contrast_id))
        candidate_column = candidate_profile.contrasts.index(str(hit.candidate_contrast_id))
        query_sign = np.sign(query_profile.beta[query_rows, query_column])
        candidate_sign = np.sign(candidate_profile.beta[candidate_rows, candidate_column])
        if str(hit.response_face) == "similar":
            supported = (query_sign != 0) & (query_sign == candidate_sign)
        else:
            supported = (query_sign != 0) & (query_sign == -candidate_sign)
        contribution = (
            block["chemical_similarity"].to_numpy(float)
            * np.abs(query_profile.weight[query_rows, query_column])
            * np.abs(candidate_profile.weight[candidate_rows, candidate_column])
        )
        order = np.flatnonzero(supported)
        order = order[np.argsort(-contribution[order], kind="stable")[:top_n]]
        for rank, index in enumerate(order, start=1):
            query_q = float(query_profile.q[query_rows[index], query_column])
            candidate_q = float(candidate_profile.q[candidate_rows[index], candidate_column])
            records.append({
                "query_study_id": query_study,
                "query_contrast_id": str(hit.query_contrast_id),
                "candidate_study_id": candidate_study,
                "candidate_contrast_id": str(hit.candidate_contrast_id),
                "evidence_world": str(hit.evidence_world),
                "response_face": str(hit.response_face),
                "face_specific_result_class": str(hit.result_class),
                "evidence_rank": rank,
                "query_entity_id": query_entities[index],
                "candidate_entity_id": candidate_entities[index],
                "query_identity_status": status[(query_study, query_entities[index])],
                "candidate_identity_status": status[(candidate_study, candidate_entities[index])],
                "query_precursor_mz": float(query_mz[index]),
                "candidate_precursor_mz": float(candidate_mz[index]),
                "precursor_ppm_difference": float(block.iloc[index]["precursor_ppm_difference"]),
                "spectral_score": float(block.iloc[index]["chemical_similarity"]),
                "query_beta": float(query_profile.beta[query_rows[index], query_column]),
                "query_se": float(query_profile.se[query_rows[index], query_column]),
                "query_diagnostic_q": query_q,
                "candidate_beta": float(candidate_profile.beta[candidate_rows[index], candidate_column]),
                "candidate_se": float(candidate_profile.se[candidate_rows[index], candidate_column]),
                "candidate_diagnostic_q": candidate_q,
                "both_diagnostic_q005": bool(query_q <= 0.05 and candidate_q <= 0.05),
                "weighted_edge_contribution": float(contribution[index]),
                "identity_claim_status": "same_precursor_reciprocal_spectral_hypothesis_not_identity_proof",
            })
    return pd.DataFrame(records)


def candidate_questions(
    top_annotated: pd.DataFrame,
) -> pd.DataFrame:
    selected = top_annotated.loc[
        top_annotated["evidence_world"].eq("all")
        & top_annotated["result_class"].isin(["similar_response", "opposite_response"])
    ].copy()
    keys = [
        "query_study_id", "query_contrast_id", "candidate_study_id",
        "candidate_contrast_id", "response_face",
    ]
    selected_index = set(map(tuple, selected[keys].astype(str).to_numpy()))
    selected["reciprocal_same_face_selection"] = [
        (
            str(row.candidate_study_id), str(row.candidate_contrast_id),
            str(row.query_study_id), str(row.query_contrast_id), str(row.response_face),
        ) in selected_index
        for row in selected.itertuples(index=False)
    ]
    project_frequency = selected["candidate_project_dependence_group"].value_counts()
    selected["candidate_project_face_call_frequency"] = selected[
        "candidate_project_dependence_group"
    ].map(project_frequency)
    world_index = {
        (
            str(row.query_study_id), str(row.query_contrast_id), str(row.evidence_world),
            str(row.response_face), str(row.candidate_study_id), str(row.candidate_contrast_id),
        ): str(row.result_class)
        for row in top_annotated.itertuples(index=False)
    }
    selected["same_link_selected_in_unresolved"] = [
        world_index.get((
            str(row.query_study_id), str(row.query_contrast_id), "unresolved",
            str(row.response_face), str(row.candidate_study_id), str(row.candidate_contrast_id),
        ), "not_selected") in {"similar_response", "opposite_response"}
        for row in selected.itertuples(index=False)
    ]
    selected["same_link_selected_in_known"] = [
        world_index.get((
            str(row.query_study_id), str(row.query_contrast_id), "known",
            str(row.response_face), str(row.candidate_study_id), str(row.candidate_contrast_id),
        ), "not_selected") in {"similar_response", "opposite_response"}
        for row in selected.itertuples(index=False)
    ]
    selected["interpretation_priority"] = "exploratory_unjudged"
    selected.loc[selected["same_parent_project"], "interpretation_priority"] = (
        "same_parent_positive_control_or_shortcut"
    )
    selected.loc[
        ~selected["same_parent_project"]
        & selected["reciprocal_same_face_selection"],
        "interpretation_priority",
    ] = "reciprocal_cross_project_adjudication"
    selected.loc[
        ~selected["same_parent_project"]
        & selected["same_link_selected_in_unresolved"]
        & ~selected["same_link_selected_in_known"]
        & ~selected["reciprocal_same_face_selection"],
        "interpretation_priority",
    ] = "unresolved_supported_cross_project_adjudication"
    selected.loc[
        selected["relation_evaluation_status"].eq("registered_unsigned_relation")
        & ~selected["same_parent_project"],
        "interpretation_priority",
    ] = "registered_cross_project_audit"
    priority_order = {
        "registered_cross_project_audit": 0,
        "reciprocal_cross_project_adjudication": 1,
        "unresolved_supported_cross_project_adjudication": 2,
        "exploratory_unjudged": 3,
        "same_parent_positive_control_or_shortcut": 4,
    }
    selected["interpretation_priority_order"] = selected[
        "interpretation_priority"
    ].map(priority_order).astype(int)
    return selected.sort_values(
        ["interpretation_priority_order", "query_maxT_p", "candidate_project_face_call_frequency"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--eligibility", required=True, type=Path)
    parser.add_argument("--retrieval-dir", required=True, type=Path)
    parser.add_argument("--identity-registry", required=True, type=Path)
    parser.add_argument("--common-identity-partition", required=True, type=Path)
    parser.add_argument("--project-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--doc", required=True, type=Path)
    parser.add_argument("--top-evidence", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.doc.exists():
        raise FileExistsError(args.doc)

    paths = {
        "contrasts": args.collection / "contrast_manifest.tsv",
        "sources": args.collection / "source_manifest.tsv",
        "effects": args.collection / "response_entities.tsv.gz",
        "relations": args.collection / "registered_relations.tsv",
        "eligibility": args.eligibility,
        "top": args.retrieval_dir / "open_local_edge_top_hits.tsv",
        "scores": args.retrieval_dir / "open_local_edge_candidate_scores.tsv.gz",
        "edges": args.retrieval_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz",
        "registry": args.identity_registry,
        "common_identity_partition": args.common_identity_partition,
        "project_map": args.project_map,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    contrasts = pd.read_csv(paths["contrasts"], sep="\t")
    sources = pd.read_csv(paths["sources"], sep="\t")
    effects = pd.read_csv(
        paths["effects"], sep="\t", low_memory=False,
        dtype={
            "study_id": str, "raw_entity_id": str,
            "entity_id": str, "contrast_id": str,
        },
    )
    effects = add_effect_statistics(effects, 0.05)
    relations = pd.read_csv(paths["relations"], sep="\t")
    eligibility = pd.read_csv(paths["eligibility"], sep="\t")
    top = pd.read_csv(paths["top"], sep="\t", low_memory=False)
    scores = pd.read_csv(paths["scores"], sep="\t", low_memory=False)
    edges = pd.read_csv(paths["edges"], sep="\t", low_memory=False)
    registry = pd.read_csv(paths["registry"], sep="\t", low_memory=False)
    common_identity = pd.read_csv(
        paths["common_identity_partition"], sep="\t", dtype=str,
    )
    if (
        len(common_identity) != 54861
        or common_identity[["study_id", "entity_id"]].duplicated().any()
        or common_identity["identity_status"].value_counts().to_dict()
        != {"unresolved_entity": 53882, "library_supported": 979}
    ):
        raise RuntimeError("common identity partition denominator drift")
    effects = effects.rename(columns={"identity_status": "source_native_identity_status"})
    effects = effects.merge(
        common_identity[["study_id", "entity_id", "identity_status", "common_identity_policy"]],
        on=["study_id", "entity_id"], how="left", validate="many_to_one",
    )
    if effects["identity_status"].isna().any():
        raise RuntimeError("common identity partition failed to cover response rows")
    registry_status = registry[["study_id", "entity_id", "identity_status"]].merge(
        common_identity[["study_id", "entity_id", "identity_status"]],
        on=["study_id", "entity_id"], suffixes=("_registry", "_common"),
        validate="one_to_one",
    )
    if not registry_status["identity_status_registry"].eq(
        registry_status["identity_status_common"]
    ).all():
        raise RuntimeError("identity registry does not use the frozen common identity policy")
    project_lookup = load_project_map(paths["project_map"], set(contrasts["study_id"]))
    eligibility["source_documented_project_dependence_group"] = eligibility[
        "project_dependence_group"
    ]
    eligibility["project_dependence_group"] = eligibility["study_id"].map(project_lookup)

    expected = {
        "sources": 15, "contrasts": 51, "effects": 238412,
        "entities": 54861, "registered_relations": 42,
    }
    observed = {
        "sources": int(sources["study_id"].nunique()),
        "contrasts": int(len(contrasts)),
        "effects": int(len(effects)),
        "entities": int(effects[["study_id", "entity_id"]].drop_duplicates().shape[0]),
        "registered_relations": int(len(relations)),
    }
    if observed != expected:
        raise RuntimeError(f"collection denominator drift: {observed}")
    if len(eligibility) != 51 or eligibility[["study_id", "contrast_id"]].duplicated().any():
        raise RuntimeError("eligibility contract drift")
    if eligibility["project_dependence_group"].isna().any():
        raise RuntimeError("failed to attach frozen operational parent-project groups")

    summary = response_summary(effects)
    combined = combined_two_face_answers(top)
    top_annotated = attach_relation_status(
        attach_side_metadata(
            top.loc[top["evidence_track"].eq(IDENTITY_TRACK)].copy(), eligibility
        ),
        relations,
    )
    combined_annotated = attach_relation_status(
        attach_side_metadata(combined, eligibility), relations
    )
    world_comparison = build_world_comparison(combined)
    score_ledger = annotate_scores(scores, eligibility, relations)
    profiles = build_profiles(effects, contrasts, 0.05)
    support = top_support_evidence(top, edges, profiles, registry, args.top_evidence)
    support = attach_relation_status(
        attach_side_metadata(support, eligibility), relations
    )
    questions = candidate_questions(top_annotated)

    contrast_world_comparison = world_comparison.rename(columns={
        "query_study_id": "study_id",
        "query_contrast_id": "contrast_id",
    })
    contrast_panorama = eligibility.merge(
        summary, on=["study_id", "contrast_id"], how="left", validate="one_to_one"
    ).merge(
        contrast_world_comparison,
        on=["study_id", "contrast_id"],
        how="left",
        validate="one_to_one",
    )
    if contrast_panorama.isna().all(axis=1).any() or len(contrast_panorama) != 51:
        raise RuntimeError("contrast panorama denominator drift")

    source_design = (
        eligibility.groupby("study_id", sort=True)
        .agg(
            project_dependence_group=("project_dependence_group", "first"),
            condition_families=("condition_family", lambda values: ";".join(sorted(set(values)))),
            study_designs=("study_design", lambda values: ";".join(sorted(set(values)))),
            causal_statuses=("causal_status", lambda values: ";".join(sorted(set(values)))),
            contrast_level_units_min=("independent_units", "min"),
            contrast_level_units_max=("independent_units", "max"),
            intentional_drug_exposure=("intentional_drug_exposure", "max"),
        )
        .reset_index()
    )
    study_panorama = sources.merge(source_design, on="study_id", validate="one_to_one")

    args.output_dir.mkdir(parents=True)
    outputs = {
        "study_panorama": args.output_dir / "study_panorama.tsv",
        "contrast_panorama": args.output_dir / "contrast_panorama.tsv",
        "face_connections": args.output_dir / "face_connection_ledger.tsv",
        "combined_answers": args.output_dir / "combined_two_face_answer_ledger.tsv",
        "world_comparison": args.output_dir / "query_world_comparison.tsv",
        "candidate_scores": args.output_dir / "full_candidate_connection_ledger.tsv.gz",
        "support": args.output_dir / "top_connection_support_evidence.tsv.gz",
        "questions": args.output_dir / "candidate_biological_questions.tsv",
    }
    study_panorama.to_csv(outputs["study_panorama"], sep="\t", index=False)
    contrast_panorama.to_csv(outputs["contrast_panorama"], sep="\t", index=False)
    top_annotated.to_csv(outputs["face_connections"], sep="\t", index=False)
    combined_annotated.to_csv(outputs["combined_answers"], sep="\t", index=False)
    world_comparison.to_csv(outputs["world_comparison"], sep="\t", index=False)
    score_ledger.to_csv(
        outputs["candidate_scores"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    support.to_csv(
        outputs["support"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    questions.to_csv(outputs["questions"], sep="\t", index=False)

    all_combined = combined_annotated.loc[combined_annotated["evidence_world"].eq("all")]
    all_answered = all_combined.loc[
        all_combined["combined_answer_class"].eq("calibrated_answer")
    ].copy()
    cross_parent_answered = all_answered.loc[~all_answered["same_parent_project"]].copy()
    condition_counts = (
        eligibility["condition_family"].value_counts().rename_axis("condition_family")
        .reset_index(name="contrasts")
    )
    question_counts = questions["interpretation_priority"].value_counts()
    report = {
        "schema": "contrast_atlas_biological_panorama_v1",
        "status": "DEVELOPMENT_CONSUMED_BIOLOGICAL_PANORAMA_COMPLETE",
        "data_role": "development_consumed",
        "scale": {
            **observed,
            "operational_dependence_groups": int(
                eligibility["project_dependence_group"].nunique()
            ),
            "candidate_contrast_directions": int(
                score_ledger[["query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"]]
                .drop_duplicates().shape[0]
            ),
            "identity_candidate_score_rows": int(len(score_ledger)),
            "face_connection_rows": int(len(top_annotated)),
            "combined_query_world_rows": int(len(combined_annotated)),
            "traceable_support_rows": int(len(support)),
        },
        "contrast_composition": {
            "species": records(
                eligibility["species"].value_counts().rename_axis("species").reset_index(name="contrasts")
            ),
            "biospecimens": records(
                eligibility["biospecimen"].value_counts().rename_axis("biospecimen").reset_index(name="contrasts")
            ),
            "condition_families": records(condition_counts),
            "intentional_drug_exposure_contrasts": int(
                eligibility["intentional_drug_exposure"]
                .astype(str).str.lower().eq("true").sum()
            ),
        },
        "response_information": {
            "response_rows": int(len(effects)),
            "library_supported_rows": int(effects["identity_status"].eq("library_supported").sum()),
            "unresolved_rows": int(effects["identity_status"].eq("unresolved_entity").sum()),
            "library_supported_study_entities": int(
                common_identity["identity_status"].eq("library_supported").sum()
            ),
            "unresolved_study_entities": int(
                common_identity["identity_status"].eq("unresolved_entity").sum()
            ),
            "identity_policy": str(common_identity["common_identity_policy"].iloc[0]),
            "diagnostic_q005_rows": int(effects["response_qualified"].sum()),
            "diagnostic_q005_boundary": (
                "normal-approximation BH is descriptive only; heterogeneous small-n effect models do not share exact t degrees of freedom"
            ),
        },
        "complete_wse_identity_track": {
            "two_face_all_world_calibrated_answers": int(
                all_combined["combined_answer_class"].eq("calibrated_answer").sum()
            ),
            "two_face_known_world_calibrated_answers": int(
                combined_annotated.loc[combined_annotated["evidence_world"].eq("known"), "combined_answer_class"]
                .eq("calibrated_answer").sum()
            ),
            "two_face_unresolved_world_calibrated_answers": int(
                combined_annotated.loc[combined_annotated["evidence_world"].eq("unresolved"), "combined_answer_class"]
                .eq("calibrated_answer").sum()
            ),
            "all_world_same_parent_project_answers": int(
                (all_combined["combined_answer_class"].eq("calibrated_answer") & all_combined["same_parent_project"]).sum()
            ),
            "unresolved_supported_not_known_same_selection": int(
                world_comparison["unresolved_supported_not_known_selection"].sum()
            ),
            "known_corroborated_same_selection": int(
                world_comparison["known_corroborated_selection"].sum()
            ),
            "all_world_cross_parent_answers": int(len(cross_parent_answered)),
            "all_world_cross_parent_same_species_and_biospecimen": int(
                (
                    cross_parent_answered["same_species"]
                    & cross_parent_answered["same_biospecimen"]
                ).sum()
            ),
            "all_world_cross_parent_same_condition_family": int(
                cross_parent_answered["same_condition_family"].sum()
            ),
            "all_world_answers_targeting_chemprop_parent": int(
                all_answered["candidate_project_dependence_group"]
                .eq("CHEMPROP_SHARED_PARENT_PROJECT").sum()
            ),
            "candidate_parent_project_counts": records(
                all_answered["candidate_project_dependence_group"]
                .value_counts().rename_axis("candidate_project_dependence_group")
                .reset_index(name="answers")
            ),
            "biological_comparability_tier_counts": records(
                all_answered["biological_comparability_tier"]
                .value_counts().rename_axis("biological_comparability_tier")
                .reset_index(name="answers")
            ),
        },
        "candidate_question_classes": {
            str(key): int(value) for key, value in question_counts.items()
        },
        "decision": {
            "panorama_data_product_complete": True,
            "biological_validity_established": False,
            "unknown_information_increment_established": False,
            "in_house_algorithm_increment_established": False,
            "different_mass_relation_track_present": False,
        },
        "claim_boundary": [
            "The panorama displays all 51 contrasts and every complete-WSE identity-track candidate direction; it does not label unregistered pairs as negatives.",
            "Complete-system application yield and opportunity-matched explanatory controls are reported separately.",
            "A same-precursor reciprocal spectral edge is an observed-ion identity hypothesis, not confirmed molecular identity.",
            "The current release has no informative different-mass chemical-relation track and therefore does not decide the full Atlas hypothesis.",
            "Stable or reciprocal retrieval is a prioritisation property, not independent biological validation.",
            "The 13 dependence groups are operational groups and not proven fully independent projects.",
        ],
        "provenance": {
            **{name: sha256(path) for name, path in paths.items()},
            **{f"output_{name}": sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    examples = questions.loc[~questions["same_parent_project"]].head(8)
    example_lines = []
    for row in examples.itertuples(index=False):
        example_lines.append(
            f"| `{row.query_study_id}::{row.query_contrast_id}` | {row.response_face} | "
            f"`{row.candidate_study_id}::{row.candidate_contrast_id}` | "
            f"{int(row.weighted_evidence_edges)} | {row.interpretation_priority} |"
        )
    document = f"""# Contrast Atlas biological panorama v1

## What is now visible

This release organises all **15 sources, 13 operational dependence groups and 51 study-level contrasts** into one traceable biological panorama.  It contains {len(effects):,} entity-by-contrast response rows from {observed['entities']:,} study-entities, all {report['scale']['candidate_contrast_directions']:,} source-disjoint directed contrast candidates, and {len(support):,} top supporting entity-pair records with beta, SE, direction and spectral evidence.

The contrast-level unit count is not additive across contrasts: several contrasts reuse participants, organisms or experimental batches.  Every row retains its design, estimand, biological unit, covariates, causal boundary and dependence group.

## Complete-system view versus explanatory controls

The complete WSE identity graph yields **{report['complete_wse_identity_track']['two_face_all_world_calibrated_answers']}/51** all-world two-face-calibrated answers.  Known-only yields **{report['complete_wse_identity_track']['two_face_known_world_calibrated_answers']}/51** and unresolved-only yields **{report['complete_wse_identity_track']['two_face_unresolved_world_calibrated_answers']}/51**.  These are discovery yields, not numbers of correct biological links.

The anatomy of those 49 statistical answers is itself a global diagnostic: **{report['complete_wse_identity_track']['all_world_same_parent_project_answers']}** select a contrast from the same operational parent project, **{report['complete_wse_identity_track']['all_world_cross_parent_answers']}** are cross-parent hypotheses, and **{report['complete_wse_identity_track']['all_world_cross_parent_same_species_and_biospecimen']}** of the cross-parent answers preserve both species and biospecimen.  **{report['complete_wse_identity_track']['all_world_answers_targeting_chemprop_parent']}/49** select the shared ChemProp parent project.  These facts show that biological validity is unresolved; they do not identify whether the concentration is caused by coverage, chemistry, edge error, scoring or real nonspecific biology.

Opportunity-matched and parent-project-held-out experiments remain explanatory controls.  They do not replace this complete-system view, and their heavy edge overlap is not treated as independent biological replication.

## What each proposed connection contains

Every face-specific connection records the two study designs, parent projects, species, biospecimens, estimands, response face, evidence count, enrichment score, max-T value and registration status.  Its support ledger then identifies the observed-ion pairs, WSE score, precursor agreement, beta/SE on both sides, diagnostic q values and weighted contribution.  Identity hypotheses never merge beta/SE across different observed ions.

Current cross-project questions include:

| Query contrast | Face | Candidate contrast | Support edges | Status |
|---|---|---|---:|---|
{chr(10).join(example_lines)}

These rows are an adjudication queue, not discoveries.  A candidate can reflect biology, drug exposure, biospecimen, coverage, identity error or a project-level response hub; those explanations must be examined using the attached signal-level evidence.

## Scientific boundary

This is the first complete **presentation and evidence-traceability layer** for the existing response collection.  It does not yet establish biological validity, unknown-specific increment or Full23 increment.  It also covers only the same-precursor identity-hypothesis track: a valid different-mass chemical-relation layer remains absent.

The next scientific operation is not another null model.  It is systematic adjudication of the cross-project queue, followed by the same complete-candidate analysis with model-specific Official, Noise and Full23 connection decisions and score calibration.
"""
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(document, encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

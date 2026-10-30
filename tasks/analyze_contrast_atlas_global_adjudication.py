#!/usr/bin/env python3
"""Build a system-wide biological adjudication layer for Contrast Atlas.

The script never decides that an unregistered study relation is biologically
true or false.  It describes every selected answer using the complete
entity-edge contribution ledger, keeps identity and different-mass relation
tracks separate, and exposes source/domain concentration for adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_LAYER = "same_precursor_identity_hypothesis"
RELATION_LAYER = "legacy_different_mass_relation_hypothesis_exploratory"
RELATION_TRACK = "nonmass_chemical_relation_hypothesis"
WORLDS = ("all", "known", "unresolved")
FACES = ("similar", "opposite")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().eq("true")


def load_entity_library_evidence(shard_root: Path) -> pd.DataFrame:
    shard_paths = sorted(shard_root.glob("*/identity_partition.tsv.gz"))
    if len(shard_paths) != 15:
        raise RuntimeError(f"expected 15 common-identity shards, observed {len(shard_paths)}")
    frame = pd.concat(
        [pd.read_csv(path, sep="\t", low_memory=False) for path in shard_paths],
        ignore_index=True,
    )
    if len(frame) != 54861:
        raise RuntimeError(f"expected 54,861 Atlas entities, observed {len(frame)}")
    if frame.duplicated(["study_id", "entity_id"]).any():
        raise RuntimeError("common-identity endpoint key is not unique")
    frame["library_supported"] = bool_series(frame["library_supported"])
    supported = frame["library_supported"]
    required = [
        "library_name", "library_inchikey", "library_spectrum_id", "library_quality",
        "wse", "matched_peaks",
    ]
    if frame.loc[supported, required].isna().any().any():
        raise RuntimeError("qualified library-supported endpoint lacks required evidence")
    if not frame.loc[supported, "identity_status"].eq("library_supported").all():
        raise RuntimeError("library support flag and frozen identity status disagree")
    frame["library_inchikey_connectivity"] = (
        frame["library_inchikey"].astype("string").str.split("-").str[0]
    )
    frame["library_evidence_class"] = np.select(
        [
            supported,
            ~supported & pd.to_numeric(
                frame["library_candidates_10ppm"], errors="coerce"
            ).fillna(0).gt(0),
        ],
        ["library_supported", "below_policy_library_candidate"],
        default="no_10ppm_library_candidate",
    )
    return frame


def annotate_identity_method_edges(
    method_edges_path: Path,
    entity_evidence: pd.DataFrame,
) -> pd.DataFrame:
    edges = pd.read_csv(method_edges_path, sep="\t", low_memory=False)
    if len(edges) != 67465 or edges["method_edge_id"].duplicated().any():
        raise RuntimeError("identity method-edge denominator or key drift")
    endpoint_columns = [
        "study_id", "entity_id", "library_supported", "identity_status",
        "library_evidence_class", "library_name", "library_inchikey",
        "library_inchikey_connectivity", "library_spectrum_id", "library_quality",
        "wse", "matched_peaks", "library_candidates_10ppm",
        "best_overall_wse", "best_overall_matched_peaks",
    ]
    for prefix in ("left", "right"):
        endpoint = entity_evidence[endpoint_columns].rename(columns={
            column: f"{prefix}_{column}" for column in endpoint_columns
        })
        edges = edges.merge(
            endpoint,
            on=[f"{prefix}_study_id", f"{prefix}_entity_id"],
            how="left", validate="many_to_one",
        )
    if edges[["left_identity_status", "right_identity_status"]].isna().any().any():
        raise RuntimeError("identity edge endpoint failed common-identity join")
    left_supported = bool_series(edges["left_library_supported"])
    right_supported = bool_series(edges["right_library_supported"])
    both = left_supported & right_supported
    full_key_equal = (
        edges["left_library_inchikey"].astype("string").fillna("")
        .eq(edges["right_library_inchikey"].astype("string").fillna(""))
    )
    connectivity_equal = (
        edges["left_library_inchikey_connectivity"].astype("string").fillna("")
        .eq(
            edges["right_library_inchikey_connectivity"]
            .astype("string").fillna("")
        )
    )
    exact = both & full_key_equal
    same_connectivity = (
        both & ~exact & connectivity_equal
    )
    discordant = both & ~exact & ~same_connectivity
    mixed = left_supported ^ right_supported
    neither = ~left_supported & ~right_supported
    membership = pd.concat(
        [exact, same_connectivity, discordant, mixed, neither], axis=1
    ).sum(axis=1)
    if not membership.eq(1).all():
        raise RuntimeError("library edge classes are not mutually exclusive and exhaustive")
    edges["library_edge_class"] = np.select(
        [exact, same_connectivity, discordant, mixed],
        [
            "known_known_exact_inchikey",
            "known_known_same_connectivity_only",
            "known_known_discordant_identity",
            "mixed_one_named_anchor",
        ],
        default="unresolved_unresolved",
    )
    edges["identity_merge_allowed_by_library_evidence"] = exact
    edges["name_propagation_to_unresolved_allowed"] = False
    return edges


def connection_key(
    left_study: pd.Series,
    left_contrast: pd.Series,
    right_study: pd.Series,
    right_contrast: pd.Series,
) -> pd.Series:
    left = left_study.astype(str) + "::" + left_contrast.astype(str)
    right = right_study.astype(str) + "::" + right_contrast.astype(str)
    return pd.Series(
        np.where(left.le(right), left + "||" + right, right + "||" + left),
        index=left.index,
        dtype="string",
    )


def attach_connection_key(
    frame: pd.DataFrame,
    left_prefix: str,
    right_prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["connection_key"] = connection_key(
        result[f"{left_prefix}_study_id"], result[f"{left_prefix}_contrast_id"],
        result[f"{right_prefix}_study_id"], result[f"{right_prefix}_contrast_id"],
    )
    return result


def combine_two_faces(top: pd.DataFrame, evidence_track: str) -> pd.DataFrame:
    selected = top.loc[top["evidence_track"].eq(evidence_track)].copy()
    edge_column = (
        "weighted_evidence_edges"
        if "weighted_evidence_edges" in selected.columns
        else "mutual_edges"
    )
    rows: list[dict[str, object]] = []
    for keys, block in selected.groupby(
        ["query_study_id", "query_contrast_id", "evidence_world"], sort=True
    ):
        if len(block) != 2 or set(block["response_face"]) != set(FACES):
            raise RuntimeError(f"two-face result contract drift for {keys}")
        eligible = block.loc[
            pd.to_numeric(block[edge_column], errors="coerce").ge(3)
            & pd.to_numeric(block["enrichment_z"], errors="coerce").gt(0)
        ]
        source = eligible if len(eligible) else block
        chosen = source.sort_values(
            ["query_maxT_p", "response_face", "candidate_study_id", "candidate_contrast_id"]
        ).iloc[0]
        corrected = (
            min(1.0, 2.0 * float(chosen["query_maxT_p"])) if len(eligible) else 1.0
        )
        row = chosen.to_dict()
        row["two_face_bonferroni_p"] = corrected
        row["combined_answer_class"] = (
            "statistically_selected_hypothesis"
            if corrected <= 0.05
            else ("insufficient_evidence" if not len(eligible) else "no_selected_match")
        )
        row["selected_evidence_edges"] = int(chosen[edge_column])
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 51 * len(WORLDS):
        raise RuntimeError(f"expected 153 query/world answers, observed {len(result)}")
    return result


def node_metadata(nodes: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        "study_id", "contrast_id", "project_dependence_group", "species", "biospecimen",
        "condition_family", "estimand", "biological_unit", "study_design",
        "independent_units", "causal_status", "effect_model",
    ]
    return nodes[keep].rename(columns={column: f"{prefix}_{column}" for column in keep})


def attach_answer_metadata(answers: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    result = answers.copy()
    for prefix in ("query", "candidate"):
        metadata = node_metadata(nodes, prefix)
        result = result.merge(
            metadata,
            on=[f"{prefix}_study_id", f"{prefix}_contrast_id"],
            how="left", validate="many_to_one", suffixes=("", "_node"),
        )
        for column in metadata.columns:
            if column in {f"{prefix}_study_id", f"{prefix}_contrast_id"}:
                continue
            node_column = f"{column}_node"
            if node_column in result.columns:
                result[column] = result[column].where(result[column].notna(), result[node_column])
                result = result.drop(columns=node_column)
    required = [
        "query_project_dependence_group", "candidate_project_dependence_group",
        "query_species", "candidate_species", "query_biospecimen", "candidate_biospecimen",
    ]
    if result[required].isna().any().any():
        raise RuntimeError("failed to attach node metadata to answer")
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
    return attach_connection_key(result, "query", "candidate")


def attach_registered_relations(
    frame: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    keep = [
        "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id",
        "relation_class", "expected_signed_direction", "eligible_primary_relevance",
        "evidence_boundary", "signed_truth_claimed",
    ]
    lookup = relations[keep].rename(columns={
        "left_study_id": "query_study_id",
        "left_contrast_id": "query_contrast_id",
        "right_study_id": "candidate_study_id",
        "right_contrast_id": "candidate_contrast_id",
    })
    result = frame.drop(columns=[
        column for column in (
            "relation_class", "expected_signed_direction", "eligible_primary_relevance",
            "evidence_boundary", "signed_truth_claimed", "relation_evaluation_status",
        ) if column in frame.columns
    ]).merge(
        lookup, on=[
            "query_study_id", "query_contrast_id", "candidate_study_id",
            "candidate_contrast_id",
        ], how="left", validate="many_to_one",
    )
    result["relation_evaluation_status"] = np.where(
        result["relation_class"].notna(),
        "registered_unsigned_relevance",
        "unregistered_unjudged",
    )
    result["relation_class"] = result["relation_class"].fillna("unregistered")
    result["expected_signed_direction"] = result["expected_signed_direction"].fillna(
        "unspecified"
    )
    result["signed_truth_claimed"] = bool_series(result["signed_truth_claimed"])
    if result["signed_truth_claimed"].any():
        raise RuntimeError("signed truth unexpectedly present")
    return result


def safe_fraction(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(
        np.divide(
            numerator.to_numpy(float), denominator.to_numpy(float),
            out=np.zeros(len(numerator), dtype=float),
            where=denominator.to_numpy(float) > 0,
        ),
        index=numerator.index,
    )


def contribution_diagnostics(
    contribution_path: Path,
    strength_path: Path,
    evidence_layer: str,
    edge_library_annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "method_edge_id", "left_study_id", "left_contrast_id", "right_study_id",
        "right_contrast_id", "evidence_world", "left_common_identity_status",
        "right_common_identity_status", "left_effect_observed", "right_effect_observed",
        "same_direction_dual_fdr_support", "opposite_direction_dual_fdr_support",
        "same_direction_contribution", "opposite_direction_contribution",
    ]
    frame = pd.read_csv(contribution_path, sep="\t", usecols=columns, low_memory=False)
    exact_library_classes: tuple[tuple[str, str], ...] = ()
    if edge_library_annotation is not None:
        annotation = edge_library_annotation[[
            "method_edge_id", "library_edge_class",
        ]]
        frame = frame.merge(
            annotation, on="method_edge_id", how="left", validate="many_to_one"
        )
        if frame["library_edge_class"].isna().any():
            raise RuntimeError("contribution row failed identity library-edge join")
        exact_library_classes = (
            ("known_known_exact_inchikey", "library_known_known_exact_inchikey"),
            (
                "known_known_same_connectivity_only",
                "library_known_known_same_connectivity_only",
            ),
            (
                "known_known_discordant_identity",
                "library_known_known_discordant_identity",
            ),
            ("mixed_one_named_anchor", "library_mixed_one_named_anchor"),
            ("unresolved_unresolved", "library_unresolved_unresolved"),
        )
    frame = attach_connection_key(frame, "left", "right")
    for column in (
        "left_effect_observed", "right_effect_observed",
        "same_direction_dual_fdr_support", "opposite_direction_dual_fdr_support",
    ):
        frame[column] = bool_series(frame[column])
    frame["identity_composition"] = np.select(
        [
            frame["left_common_identity_status"].eq("library_supported")
            & frame["right_common_identity_status"].eq("library_supported"),
            frame["left_common_identity_status"].eq("unresolved_entity")
            & frame["right_common_identity_status"].eq("unresolved_entity"),
        ],
        ["known_known", "unresolved_unresolved"],
        default="mixed",
    )
    frame["both_effects_observed"] = (
        frame["left_effect_observed"] & frame["right_effect_observed"]
    )
    frame["any_effect_missing"] = ~frame["both_effects_observed"]
    keys = ["connection_key", "evidence_world"]
    base = frame.groupby(keys, sort=False).agg(
        contribution_rows=("method_edge_id", "size"),
        unique_method_edges=("method_edge_id", "nunique"),
        both_effects_observed_rows=("both_effects_observed", "sum"),
        any_effect_missing_rows=("any_effect_missing", "sum"),
    ).reset_index()
    base["missing_effect_row_fraction"] = safe_fraction(
        base["any_effect_missing_rows"], base["contribution_rows"]
    )
    for face in FACES:
        value_column = f"{face if face == 'opposite' else 'same'}_direction_contribution"
        support_column = f"{face if face == 'opposite' else 'same'}_direction_dual_fdr_support"
        work_columns = [*keys, "identity_composition", support_column, value_column]
        if exact_library_classes:
            work_columns.append("library_edge_class")
        work = frame[work_columns].copy()
        work = work.rename(columns={value_column: "value", support_column: "dual_q"})
        work["value"] = pd.to_numeric(work["value"], errors="coerce").fillna(0.0)
        work["value_squared"] = work["value"] ** 2
        work["contributing"] = work["value"].gt(0)
        work["dual_q_value"] = np.where(work["dual_q"], work["value"], 0.0)
        work["rank"] = work.groupby(keys, sort=False)["value"].rank(
            method="first", ascending=False
        )
        for n in (1, 5, 10):
            work[f"top{n}_value"] = np.where(work["rank"].le(n), work["value"], 0.0)
        for composition in ("known_known", "mixed", "unresolved_unresolved"):
            work[f"{composition}_value"] = np.where(
                work["identity_composition"].eq(composition), work["value"], 0.0
            )
            work[f"{composition}_edge"] = (
                work["identity_composition"].eq(composition) & work["contributing"]
            )
        aggregated = work.groupby(keys, sort=False).agg(
            total_strength=("value", "sum"),
            strength_squared_sum=("value_squared", "sum"),
            contributing_edges=("contributing", "sum"),
            dual_q_support_edges=("dual_q", "sum"),
            dual_q_support_strength=("dual_q_value", "sum"),
            top1_strength=("top1_value", "sum"),
            top5_strength=("top5_value", "sum"),
            top10_strength=("top10_value", "sum"),
            known_known_strength=("known_known_value", "sum"),
            mixed_strength=("mixed_value", "sum"),
            unresolved_unresolved_strength=("unresolved_unresolved_value", "sum"),
            known_known_contributing_edges=("known_known_edge", "sum"),
            mixed_contributing_edges=("mixed_edge", "sum"),
            unresolved_unresolved_contributing_edges=("unresolved_unresolved_edge", "sum"),
        ).reset_index()
        if exact_library_classes:
            for raw_category, metric_category in exact_library_classes:
                work[f"{metric_category}_value"] = np.where(
                    work["library_edge_class"].eq(raw_category), work["value"], 0.0
                )
                work[f"{metric_category}_edge"] = (
                    work["library_edge_class"].eq(raw_category) & work["contributing"]
                )
            exact_aggregation: dict[str, tuple[str, str]] = {}
            for _, metric_category in exact_library_classes:
                exact_aggregation[f"{metric_category}_strength"] = (
                    f"{metric_category}_value", "sum"
                )
                exact_aggregation[f"{metric_category}_contributing_edges"] = (
                    f"{metric_category}_edge", "sum"
                )
            exact = work.groupby(keys, sort=False).agg(**exact_aggregation).reset_index()
            aggregated = aggregated.merge(exact, on=keys, validate="one_to_one")
        aggregated[f"{face}_effective_edge_count"] = np.divide(
            aggregated["total_strength"] ** 2,
            aggregated["strength_squared_sum"],
            out=np.zeros(len(aggregated), dtype=float),
            where=aggregated["strength_squared_sum"].gt(0),
        )
        for name in (
            "top1", "top5", "top10", "dual_q_support", "known_known", "mixed",
            "unresolved_unresolved"
        ):
            aggregated[f"{face}_{name}_strength_fraction"] = safe_fraction(
                aggregated[f"{name}_strength"], aggregated["total_strength"]
            )
        for _, metric_category in exact_library_classes:
            aggregated[f"{face}_{metric_category}_strength_fraction"] = safe_fraction(
                aggregated[f"{metric_category}_strength"], aggregated["total_strength"]
            )
        rename = {
            column: f"{face}_{column}"
            for column in aggregated.columns
            if column not in keys and not column.startswith(f"{face}_")
        }
        aggregated = aggregated.rename(columns=rename)
        base = base.merge(aggregated, on=keys, validate="one_to_one")

    strengths = pd.read_csv(strength_path, sep="\t", low_memory=False)
    strengths = attach_connection_key(strengths, "left", "right")
    validation = strengths[[
        "connection_key", "evidence_world", "mutual_edges",
        "same_direction_strength", "opposite_direction_strength",
    ]].merge(base, on=keys, how="left", validate="one_to_one")
    if len(validation) != len(strengths):
        raise RuntimeError("connection diagnostic denominator drift")
    diagnostic_columns = [column for column in base.columns if column not in keys]
    missing_diagnostics = validation["unique_method_edges"].isna()
    if not validation.loc[missing_diagnostics, "mutual_edges"].eq(0).all():
        raise RuntimeError("nonempty connection is absent from contribution ledger")
    if int(missing_diagnostics.sum()) != int(strengths["mutual_edges"].eq(0).sum()):
        raise RuntimeError("explicit zero-edge connection count failed conservation")
    validation[diagnostic_columns] = validation[diagnostic_columns].fillna(0)
    if not np.allclose(
        validation["same_direction_strength"], validation["similar_total_strength"],
        rtol=1e-10, atol=1e-10,
    ):
        raise RuntimeError("similar contribution strength failed conservation")
    if not np.allclose(
        validation["opposite_direction_strength"], validation["opposite_total_strength"],
        rtol=1e-10, atol=1e-10,
    ):
        raise RuntimeError("opposite contribution strength failed conservation")
    if not validation["mutual_edges"].eq(validation["unique_method_edges"]).all():
        raise RuntimeError("method edge count failed conservation")
    strengths = strengths.merge(base, on=keys, how="left", validate="one_to_one")
    strengths[diagnostic_columns] = strengths[diagnostic_columns].fillna(0)
    strengths.insert(0, "evidence_layer", evidence_layer)
    return strengths


def attach_connection_metadata(
    connections: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    result = connections.copy()
    for prefix in ("left", "right"):
        metadata = node_metadata(nodes, prefix)
        result = result.merge(
            metadata, on=[f"{prefix}_study_id", f"{prefix}_contrast_id"],
            how="left", validate="many_to_one",
        )
    result["same_parent_project"] = result["left_project_dependence_group"].eq(
        result["right_project_dependence_group"]
    )
    result["same_species"] = result["left_species"].eq(result["right_species"])
    result["same_biospecimen"] = result["left_biospecimen"].eq(result["right_biospecimen"])
    result["same_condition_family"] = result["left_condition_family"].eq(
        result["right_condition_family"]
    )
    return result


def chosen_face_metrics(answers: pd.DataFrame) -> pd.DataFrame:
    result = answers.copy()
    suffixes = [
        "total_strength", "contributing_edges", "dual_q_support_edges",
        "dual_q_support_strength", "dual_q_support_strength_fraction",
        "effective_edge_count", "top1_strength_fraction", "top5_strength_fraction",
        "top10_strength_fraction", "known_known_strength_fraction",
        "mixed_strength_fraction", "unresolved_unresolved_strength_fraction",
        "known_known_contributing_edges", "mixed_contributing_edges",
        "unresolved_unresolved_contributing_edges",
    ]
    for category in (
        "library_known_known_exact_inchikey",
        "library_known_known_same_connectivity_only",
        "library_known_known_discordant_identity",
        "library_mixed_one_named_anchor",
        "library_unresolved_unresolved",
    ):
        suffixes.extend([
            f"{category}_strength",
            f"{category}_strength_fraction",
            f"{category}_contributing_edges",
        ])
    similar = result["response_face"].eq("similar")
    for suffix in suffixes:
        similar_column = f"similar_{suffix}"
        opposite_column = f"opposite_{suffix}"
        if similar_column in result.columns and opposite_column in result.columns:
            result[f"selected_face_{suffix}"] = np.where(
                similar, result[similar_column], result[opposite_column]
            )
    return result


def world_consistency(answers: pd.DataFrame, layer: str) -> pd.DataFrame:
    selected = answers.loc[answers["evidence_layer"].eq(layer)].copy()
    keys = ["query_study_id", "query_contrast_id"]
    values = [
        "combined_answer_class", "candidate_study_id", "candidate_contrast_id",
        "response_face",
    ]
    pieces = []
    for world in WORLDS:
        block = selected.loc[selected["evidence_world"].eq(world), [*keys, *values]].copy()
        block = block.rename(columns={column: f"{world}_{column}" for column in values})
        pieces.append(block)
    result = pieces[0]
    for block in pieces[1:]:
        result = result.merge(block, on=keys, validate="one_to_one")
    result["all_answered"] = result["all_combined_answer_class"].eq(
        "statistically_selected_hypothesis"
    )
    result["same_all_unresolved_target_and_face"] = (
        result["all_candidate_study_id"].eq(result["unresolved_candidate_study_id"])
        & result["all_candidate_contrast_id"].eq(result["unresolved_candidate_contrast_id"])
        & result["all_response_face"].eq(result["unresolved_response_face"])
    )
    result["same_all_known_target_and_face"] = (
        result["all_candidate_study_id"].eq(result["known_candidate_study_id"])
        & result["all_candidate_contrast_id"].eq(result["known_candidate_contrast_id"])
        & result["all_response_face"].eq(result["known_response_face"])
    )
    result["known_same_selection_and_answered"] = (
        result["same_all_known_target_and_face"]
        & result["known_combined_answer_class"].eq("statistically_selected_hypothesis")
    )
    result["unresolved_same_selection_and_answered"] = (
        result["same_all_unresolved_target_and_face"]
        & result["unresolved_combined_answer_class"].eq("statistically_selected_hypothesis")
    )
    return result


def classify_queue(queue: pd.DataFrame) -> pd.Series:
    answered = queue["combined_answer_class"].eq("statistically_selected_hypothesis")
    return pd.Series(
        np.select(
            [
                ~answered,
                queue["same_parent_project"],
                queue["relation_evaluation_status"].eq("registered_unsigned_relevance"),
                queue["biological_comparability_tier"].eq(
                    "cross_parent_domain_and_condition_matched"
                ),
                queue["biological_comparability_tier"].eq(
                    "cross_parent_species_and_biospecimen_matched"
                ),
                queue["biological_comparability_tier"].eq(
                    "cross_parent_condition_family_only"
                ),
                queue["cross_track_same_target_and_face"],
            ],
            [
                "no_statistically_selected_match",
                "same_parent_internal_control_or_shortcut",
                "registered_unsigned_cross_parent_audit",
                "domain_and_condition_matched_cross_parent_audit",
                "domain_matched_cross_parent_audit",
                "condition_family_only_cross_parent_audit",
                "cross_track_convergent_cross_domain_hypothesis",
            ],
            default="cross_domain_unregistered_hypothesis",
        ),
        index=queue.index,
        dtype="string",
    )


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def join_unique(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna()}))


def build_selected_link_evidence(
    all_queue: pd.DataFrame,
    connections: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    selected = all_queue.loc[
        all_queue["evidence_layer"].eq(IDENTITY_LAYER)
        & all_queue["combined_answer_class"].eq("statistically_selected_hypothesis")
    ].copy()
    selected["query_direction_key"] = (
        selected["query_study_id"].astype(str)
        + "::" + selected["query_contrast_id"].astype(str)
    )
    directional = selected.groupby("connection_key", sort=True).agg(
        selected_query_directions=("query_direction_key", "size"),
        selecting_query_keys=("query_direction_key", join_unique),
        selected_response_faces=("response_face", join_unique),
        minimum_two_face_p=("two_face_bonferroni_p", "min"),
        maximum_two_face_p=("two_face_bonferroni_p", "max"),
    ).reset_index()
    identity = connections.loc[
        connections["evidence_layer"].eq(IDENTITY_LAYER)
        & connections["evidence_world"].eq("all")
        & connections["connection_key"].isin(directional["connection_key"])
    ].copy()
    result = identity.merge(directional, on="connection_key", validate="one_to_one")
    if len(result) != 44:
        raise RuntimeError(f"expected 44 selected identity links, observed {len(result)}")

    relation = connections.loc[
        connections["evidence_layer"].eq(RELATION_LAYER)
        & connections["evidence_world"].eq("all"),
        [
            "connection_key", "mutual_edges", "same_direction_strength",
            "opposite_direction_strength", "similar_effective_edge_count",
            "opposite_effective_edge_count",
        ],
    ].rename(columns={
        column: f"relation_axis_{column}"
        for column in (
            "mutual_edges", "same_direction_strength", "opposite_direction_strength",
            "similar_effective_edge_count", "opposite_effective_edge_count",
        )
    })
    result = result.merge(relation, on="connection_key", how="left", validate="one_to_one")
    relation_same = result["relation_axis_same_direction_strength"]
    relation_opposite = result["relation_axis_opposite_direction_strength"]
    result["relation_axis_preferred_face"] = np.select(
        [relation_same.gt(relation_opposite), relation_opposite.gt(relation_same)],
        ["similar", "opposite"], default="tie_or_zero",
    )
    result["relation_axis_normalized_margin"] = np.divide(
        (relation_same - relation_opposite).abs(), relation_same + relation_opposite,
        out=np.zeros(len(result), dtype=float),
        where=(relation_same + relation_opposite).gt(0),
    )

    relation_keys = attach_connection_key(relations, "left", "right")
    relation_summary = relation_keys.groupby("connection_key", sort=True).agg(
        registered_relation_classes=("relation_class", join_unique),
        any_registered_primary_eligible=("eligible_primary_relevance", lambda x: bool_series(x).any()),
        signed_truth_claimed=("signed_truth_claimed", lambda x: bool_series(x).any()),
    ).reset_index()
    result = result.merge(
        relation_summary, on="connection_key", how="left", validate="one_to_one"
    )
    result["registered_relation_classes"] = result[
        "registered_relation_classes"
    ].fillna("unregistered")
    result["any_registered_primary_eligible"] = result[
        "any_registered_primary_eligible"
    ].astype("boolean").fillna(False).astype(bool)
    result["signed_truth_claimed"] = (
        result["signed_truth_claimed"].astype("boolean").fillna(False).astype(bool)
    )
    if result["signed_truth_claimed"].any():
        raise RuntimeError("selected link unexpectedly has signed truth")
    result["link_claim_tier"] = np.select(
        [
            result["same_parent_project"],
            result["any_registered_primary_eligible"],
            result["same_species"] & result["same_biospecimen"]
            & result["same_condition_family"],
            result["same_species"] & result["same_biospecimen"],
        ],
        [
            "same_parent_consumed_or_shortcut_control",
            "registered_unsigned_cross_parent_audit",
            "cross_parent_domain_and_condition_matched_hypothesis",
            "cross_parent_domain_matched_hypothesis",
        ],
        default="cross_domain_unregistered_hypothesis",
    )
    return result.sort_values(["link_claim_tier", "connection_key"]).reset_index(drop=True)


def build_registered_target_audit(
    relations: pd.DataFrame,
    all_queue: pd.DataFrame,
    connections: pd.DataFrame,
    identity_registered_ranks: pd.DataFrame,
) -> pd.DataFrame:
    result = attach_connection_key(relations, "left", "right")
    result["eligible_primary_relevance"] = bool_series(result["eligible_primary_relevance"])
    result["signed_truth_claimed"] = bool_series(result["signed_truth_claimed"])
    if result["signed_truth_claimed"].any():
        raise RuntimeError("registered target unexpectedly claims signed truth")
    rank_rows = identity_registered_ranks.loc[
        identity_registered_ranks["evidence_world"].eq("all")
        & identity_registered_ranks["evidence_track"].eq(
            "same_precursor_10ppm_identity_hypothesis"
        )
    ].copy()
    if len(rank_rows) != 42:
        raise RuntimeError(
            f"expected 42 all-world identity registered ranks, observed {len(rank_rows)}"
        )
    rank_columns = [
        "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id",
        "similar_z", "opposite_z", "similar_query_maxT_p",
        "opposite_query_maxT_p", "similar_rank", "opposite_rank",
        "candidate_universe_size", "best_face_rank", "best_face_query_maxT_p",
    ]
    rank_rows = rank_rows[rank_columns].rename(columns={
        column: f"identity_registered_target_{column}"
        for column in rank_columns
        if column not in {
            "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"
        }
    })
    result = result.merge(
        rank_rows,
        on=[
            "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"
        ],
        how="left", validate="one_to_one",
    )
    for layer, prefix in ((IDENTITY_LAYER, "identity"), (RELATION_LAYER, "relation")):
        chosen = all_queue.loc[
            all_queue["evidence_layer"].eq(layer),
            [
                "query_study_id", "query_contrast_id", "candidate_study_id",
                "candidate_contrast_id", "combined_answer_class", "response_face",
                "two_face_bonferroni_p", "selected_evidence_edges",
            ],
        ].rename(columns={
            "query_study_id": "left_study_id",
            "query_contrast_id": "left_contrast_id",
            "candidate_study_id": f"{prefix}_selected_study_id",
            "candidate_contrast_id": f"{prefix}_selected_contrast_id",
            "combined_answer_class": f"{prefix}_answer_class",
            "response_face": f"{prefix}_selected_face",
            "two_face_bonferroni_p": f"{prefix}_two_face_p",
            "selected_evidence_edges": f"{prefix}_selected_evidence_edges",
        })
        result = result.merge(
            chosen, on=["left_study_id", "left_contrast_id"],
            how="left", validate="many_to_one",
        )
        result[f"{prefix}_registered_target_selected"] = (
            result[f"{prefix}_answer_class"].eq("statistically_selected_hypothesis")
            & result[f"{prefix}_selected_study_id"].eq(result["right_study_id"])
            & result[f"{prefix}_selected_contrast_id"].eq(result["right_contrast_id"])
        )
        target = connections.loc[
            connections["evidence_layer"].eq(layer)
            & connections["evidence_world"].eq("all"),
            [
                "connection_key", "mutual_edges", "same_direction_strength",
                "opposite_direction_strength", "similar_dual_q_support_edges",
                "opposite_dual_q_support_edges", "missing_effect_row_fraction",
            ],
        ].rename(columns={
            column: f"{prefix}_registered_target_{column}"
            for column in (
                "mutual_edges", "same_direction_strength", "opposite_direction_strength",
                "similar_dual_q_support_edges", "opposite_dual_q_support_edges",
                "missing_effect_row_fraction",
            )
        })
        result = result.merge(target, on="connection_key", how="left", validate="many_to_one")
    if len(result) != 42:
        raise RuntimeError(f"expected 42 registered directions, observed {len(result)}")
    return result


def build_source_concentration(all_queue: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected = all_queue.loc[
        all_queue["combined_answer_class"].eq("statistically_selected_hypothesis")
    ].copy()
    grouped = selected.groupby(
        ["evidence_layer", "candidate_project_dependence_group"], sort=True
    ).agg(
        selected_query_directions=("query_study_id", "size"),
        same_parent_directions=("same_parent_project", "sum"),
    ).reset_index()
    grouped["cross_parent_directions"] = (
        grouped["selected_query_directions"] - grouped["same_parent_directions"]
    )
    totals = grouped.groupby("evidence_layer")["selected_query_directions"].transform("sum")
    grouped["selected_direction_fraction"] = grouped["selected_query_directions"] / totals
    summaries: list[dict[str, object]] = []
    for layer, block in grouped.groupby("evidence_layer", sort=True):
        hhi = float((block["selected_direction_fraction"] ** 2).sum())
        summaries.append({
            "evidence_layer": layer,
            "selected_directions": int(block["selected_query_directions"].sum()),
            "candidate_parent_projects": int(len(block)),
            "parent_project_hhi": hhi,
            "effective_parent_projects": float(1.0 / hhi),
        })
    return grouped, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panorama-dir", type=Path,
        default=Path("data/validation/contrast_atlas_biological_panorama_release_v1_20261029"),
    )
    parser.add_argument(
        "--identity-ledger-dir", type=Path,
        default=Path("data/validation/contrast_atlas_connection_evidence_ledger_release_v1_20261029"),
    )
    parser.add_argument(
        "--relation-ledger-dir", type=Path,
        default=Path("data/validation/contrast_atlas_legacy_relation_evidence_ledger_smoke_v1_20261029"),
    )
    parser.add_argument(
        "--legacy-retrieval-dir", type=Path,
        default=Path("data/validation/contrast_atlas_open_local_edge_enrichment_v5_20261027"),
    )
    parser.add_argument(
        "--identity-retrieval-dir", type=Path,
        default=Path(
            "data/validation/contrast_atlas_complete_identity_wse_open_retrieval_v3_20261029"
        ),
    )
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"),
    )
    parser.add_argument(
        "--common-identity-shards", type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_v9_frozenbase_common_identity_shards_20261023"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.doc.exists():
        raise FileExistsError(args.doc)

    paths = {
        "nodes": args.panorama_dir / "contrast_panorama.tsv",
        "identity_answers": args.panorama_dir / "combined_two_face_answer_ledger.tsv",
        "identity_contributions": args.identity_ledger_dir / "connection_edge_contributions.tsv.gz",
        "identity_strengths": args.identity_ledger_dir / "identity_connection_strengths.tsv.gz",
        "identity_method_edges": args.identity_ledger_dir / "method_entity_edges.tsv.gz",
        "relation_contributions": args.relation_ledger_dir / "legacy_relation_edge_contributions.tsv.gz",
        "relation_strengths": args.relation_ledger_dir / "legacy_relation_connection_strengths.tsv.gz",
        "relation_top": args.legacy_retrieval_dir / "open_local_edge_top_hits.tsv",
        "identity_registered_ranks": (
            args.identity_retrieval_dir / "registered_relation_ranks.tsv"
        ),
        "registered_relations": args.collection / "registered_relations.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    if not args.common_identity_shards.is_dir():
        raise FileNotFoundError(args.common_identity_shards)

    nodes = pd.read_csv(paths["nodes"], sep="\t", low_memory=False)
    relations = pd.read_csv(paths["registered_relations"], sep="\t", low_memory=False)
    identity_registered_ranks = pd.read_csv(
        paths["identity_registered_ranks"], sep="\t", low_memory=False
    )
    entity_library_evidence = load_entity_library_evidence(args.common_identity_shards)
    identity_method_edges = annotate_identity_method_edges(
        paths["identity_method_edges"], entity_library_evidence
    )

    identity_connections = contribution_diagnostics(
        paths["identity_contributions"], paths["identity_strengths"], IDENTITY_LAYER,
        edge_library_annotation=identity_method_edges,
    )
    relation_connections = contribution_diagnostics(
        paths["relation_contributions"], paths["relation_strengths"], RELATION_LAYER
    )
    connections = pd.concat(
        [
            attach_connection_metadata(identity_connections, nodes),
            attach_connection_metadata(relation_connections, nodes),
        ],
        ignore_index=True,
    )
    if len(connections) != 2 * 3429:
        raise RuntimeError("two-layer connection denominator drift")

    identity_answers = pd.read_csv(paths["identity_answers"], sep="\t", low_memory=False)
    identity_answers["combined_answer_class"] = identity_answers[
        "combined_answer_class"
    ].replace({"calibrated_answer": "statistically_selected_hypothesis"})
    identity_answers["selected_evidence_edges"] = identity_answers[
        "weighted_evidence_edges"
    ]
    identity_answers["evidence_layer"] = IDENTITY_LAYER
    identity_answers = attach_answer_metadata(identity_answers, nodes)
    identity_answers = attach_registered_relations(identity_answers, relations)

    relation_top = pd.read_csv(paths["relation_top"], sep="\t", low_memory=False)
    relation_answers = combine_two_faces(relation_top, RELATION_TRACK)
    relation_answers["evidence_layer"] = RELATION_LAYER
    relation_answers = attach_answer_metadata(relation_answers, nodes)
    relation_answers = attach_registered_relations(relation_answers, relations)

    answers = pd.concat([identity_answers, relation_answers], ignore_index=True, sort=False)
    diagnostic_columns = [
        column for column in connections.columns
        if column not in {
            "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id",
            "left_project_dependence_group", "right_project_dependence_group",
            "left_species", "right_species", "left_biospecimen", "right_biospecimen",
            "left_condition_family", "right_condition_family", "left_estimand", "right_estimand",
            "left_biological_unit", "right_biological_unit", "left_study_design",
            "right_study_design", "left_independent_units", "right_independent_units",
            "left_causal_status", "right_causal_status", "left_effect_model", "right_effect_model",
            "same_parent_project", "same_species", "same_biospecimen", "same_condition_family",
        }
    ]
    answers = answers.merge(
        connections[diagnostic_columns],
        on=["evidence_layer", "connection_key", "evidence_world"],
        how="left", validate="many_to_one",
    )
    if answers["contribution_rows"].isna().any():
        raise RuntimeError("failed to attach complete contribution diagnostics")
    answers = chosen_face_metrics(answers)

    all_queue = answers.loc[answers["evidence_world"].eq("all")].copy()
    for layer in (IDENTITY_LAYER, RELATION_LAYER):
        consistency = world_consistency(answers, layer)
        layer_mask = all_queue["evidence_layer"].eq(layer)
        merged = all_queue.loc[layer_mask].merge(
            consistency, on=["query_study_id", "query_contrast_id"],
            how="left", validate="one_to_one", suffixes=("", "_world"),
        )
        for column in (
            "same_all_unresolved_target_and_face", "same_all_known_target_and_face",
            "known_same_selection_and_answered", "unresolved_same_selection_and_answered",
        ):
            all_queue.loc[layer_mask, column] = merged[column].to_numpy()

    track_targets = all_queue[[
        "query_study_id", "query_contrast_id", "evidence_layer", "combined_answer_class",
        "candidate_study_id", "candidate_contrast_id", "response_face",
    ]].copy()
    identity_target = track_targets.loc[
        track_targets["evidence_layer"].eq(IDENTITY_LAYER)
    ].drop(columns="evidence_layer").rename(columns={
        column: f"identity_{column}" for column in (
            "combined_answer_class", "candidate_study_id", "candidate_contrast_id", "response_face"
        )
    })
    relation_target = track_targets.loc[
        track_targets["evidence_layer"].eq(RELATION_LAYER)
    ].drop(columns="evidence_layer").rename(columns={
        column: f"relation_{column}" for column in (
            "combined_answer_class", "candidate_study_id", "candidate_contrast_id", "response_face"
        )
    })
    convergence = identity_target.merge(
        relation_target, on=["query_study_id", "query_contrast_id"], validate="one_to_one"
    )
    convergence["both_tracks_answered"] = (
        convergence["identity_combined_answer_class"].eq("statistically_selected_hypothesis")
        & convergence["relation_combined_answer_class"].eq("statistically_selected_hypothesis")
    )
    convergence["cross_track_same_target"] = (
        convergence["both_tracks_answered"]
        & convergence["identity_candidate_study_id"].eq(convergence["relation_candidate_study_id"])
        & convergence["identity_candidate_contrast_id"].eq(
            convergence["relation_candidate_contrast_id"]
        )
    )
    convergence["cross_track_same_target_and_face"] = (
        convergence["cross_track_same_target"]
        & convergence["identity_response_face"].eq(convergence["relation_response_face"])
    )
    all_queue = all_queue.merge(
        convergence[[
            "query_study_id", "query_contrast_id", "both_tracks_answered",
            "cross_track_same_target", "cross_track_same_target_and_face",
        ]], on=["query_study_id", "query_contrast_id"], validate="many_to_one",
    )
    all_queue["adjudication_stratum"] = classify_queue(all_queue)
    all_queue["candidate_project_answer_frequency_within_layer"] = 0
    for layer in (IDENTITY_LAYER, RELATION_LAYER):
        mask = (
            all_queue["evidence_layer"].eq(layer)
            & all_queue["combined_answer_class"].eq("statistically_selected_hypothesis")
        )
        counts = all_queue.loc[mask, "candidate_project_dependence_group"].value_counts()
        all_queue.loc[mask, "candidate_project_answer_frequency_within_layer"] = (
            all_queue.loc[mask, "candidate_project_dependence_group"].map(counts).to_numpy()
        )
    all_queue["reciprocal_selected_answer"] = False
    for layer in (IDENTITY_LAYER, RELATION_LAYER):
        mask = (
            all_queue["evidence_layer"].eq(layer)
            & all_queue["combined_answer_class"].eq("statistically_selected_hypothesis")
        )
        selected_pairs = {
            (
                row.query_study_id, row.query_contrast_id,
                row.candidate_study_id, row.candidate_contrast_id,
            )
            for row in all_queue.loc[mask].itertuples()
        }
        reciprocal = []
        for row in all_queue.loc[mask].itertuples():
            reciprocal.append((
                row.candidate_study_id, row.candidate_contrast_id,
                row.query_study_id, row.query_contrast_id,
            ) in selected_pairs)
        all_queue.loc[mask, "reciprocal_selected_answer"] = reciprocal

    selected_link_evidence = build_selected_link_evidence(all_queue, connections, relations)
    registered_target_audit = build_registered_target_audit(
        relations, all_queue, connections, identity_registered_ranks
    )
    source_concentration, source_concentration_summary = build_source_concentration(
        all_queue
    )
    identity_global_answers = all_queue.loc[
        all_queue["evidence_layer"].eq(IDENTITY_LAYER)
    ].copy()
    relation_global_answers = all_queue.loc[
        all_queue["evidence_layer"].eq(RELATION_LAYER)
    ].copy()
    if len(identity_global_answers) != 51 or len(relation_global_answers) != 51:
        raise RuntimeError("global answer table denominator drift")

    args.output_dir.mkdir(parents=True)
    outputs = {
        "connections": args.output_dir / "atlas_connection_diagnostics.tsv.gz",
        "answers": args.output_dir / "atlas_selected_answer_diagnostics.tsv.gz",
        "queue": args.output_dir / "atlas_all_world_adjudication_queue.tsv",
        "identity_global_answers": args.output_dir / "identity_global_answer_adjudication.tsv",
        "relation_global_answers": args.output_dir / "relation_global_answer_adjudication.tsv",
        "selected_links": args.output_dir / "identity_selected_link_evidence.tsv.gz",
        "registered_targets": args.output_dir / "registered_target_audit.tsv",
        "source_concentration": args.output_dir / "source_concentration.tsv",
        "entity_library_evidence": args.output_dir / "entity_library_evidence.tsv.gz",
        "method_edge_library_annotation": (
            args.output_dir / "identity_method_edge_library_annotation.tsv.gz"
        ),
    }
    connections.to_csv(
        outputs["connections"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    answers.to_csv(
        outputs["answers"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    all_queue.to_csv(outputs["queue"], sep="\t", index=False)
    identity_global_answers.to_csv(
        outputs["identity_global_answers"], sep="\t", index=False
    )
    relation_global_answers.to_csv(
        outputs["relation_global_answers"], sep="\t", index=False
    )
    selected_link_evidence.to_csv(
        outputs["selected_links"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    registered_target_audit.to_csv(
        outputs["registered_targets"], sep="\t", index=False
    )
    source_concentration.to_csv(
        outputs["source_concentration"], sep="\t", index=False
    )
    entity_library_evidence.to_csv(
        outputs["entity_library_evidence"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    identity_method_edges.to_csv(
        outputs["method_edge_library_annotation"], sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    output_hashes = {path.name: sha256(path) for path in outputs.values()}
    checksums_path = args.output_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(output_hashes.items())
        ),
        encoding="utf-8",
    )

    all_answered = all_queue.loc[
        all_queue["combined_answer_class"].eq("statistically_selected_hypothesis")
    ]
    by_layer = []
    for layer, block in all_queue.groupby("evidence_layer", sort=True):
        answered = block.loc[
            block["combined_answer_class"].eq("statistically_selected_hypothesis")
        ]
        by_layer.append({
            "evidence_layer": layer,
            "queries": int(len(block)),
            "statistically_selected_hypotheses": int(len(answered)),
            "same_parent_answers": int(answered["same_parent_project"].sum()),
            "cross_parent_answers": int((~answered["same_parent_project"]).sum()),
            "cross_parent_same_species_and_biospecimen": int((
                ~answered["same_parent_project"]
                & answered["same_species"] & answered["same_biospecimen"]
            ).sum()),
            "cross_parent_same_condition_family": int((
                ~answered["same_parent_project"] & answered["same_condition_family"]
            ).sum()),
            "answers_targeting_chemprop_parent": int(
                answered["candidate_project_dependence_group"]
                .eq("CHEMPROP_SHARED_PARENT_PROJECT").sum()
            ),
            "registered_unsigned_answers": int(
                answered["relation_evaluation_status"]
                .eq("registered_unsigned_relevance").sum()
            ),
        })
    identity_answered = identity_global_answers.loc[
        identity_global_answers["combined_answer_class"].eq(
            "statistically_selected_hypothesis"
        )
    ].copy()
    identity_cross_parent = identity_answered.loc[~identity_answered["same_parent_project"]]
    identity_total_selected_strength = float(
        identity_answered["selected_face_total_strength"].sum()
    )
    library_strength_columns = {
        "exact_inchikey": "selected_face_library_known_known_exact_inchikey_strength",
        "same_connectivity_only": (
            "selected_face_library_known_known_same_connectivity_only_strength"
        ),
        "discordant_named_identity": (
            "selected_face_library_known_known_discordant_identity_strength"
        ),
        "one_named_anchor": "selected_face_library_mixed_one_named_anchor_strength",
        "unresolved_unresolved": (
            "selected_face_library_unresolved_unresolved_strength"
        ),
    }
    library_contribution = {
        name: {
            "strength": float(identity_answered[column].sum()),
            "fraction": (
                float(identity_answered[column].sum()) / identity_total_selected_strength
            ),
        }
        for name, column in library_strength_columns.items()
    }
    eligible_primary = registered_target_audit.loc[
        registered_target_audit["eligible_primary_relevance"]
    ]
    posthoc_registered = registered_target_audit.loc[
        ~registered_target_audit["eligible_primary_relevance"]
    ]
    registered_audit_summary = {
        "registered_directions": int(len(registered_target_audit)),
        "eligible_primary_directions": int(len(eligible_primary)),
        "identity_primary_targets_selected": int(
            eligible_primary["identity_registered_target_selected"].sum()
        ),
        "relation_primary_targets_selected": int(
            eligible_primary["relation_registered_target_selected"].sum()
        ),
        "identity_posthoc_targets_selected": int(
            posthoc_registered["identity_registered_target_selected"].sum()
        ),
        "relation_posthoc_targets_selected": int(
            posthoc_registered["relation_registered_target_selected"].sum()
        ),
        "identity_primary_registered_target_best_face_top1": int(
            pd.to_numeric(
                eligible_primary["identity_registered_target_best_face_rank"],
                errors="coerce",
            ).eq(1).sum()
        ),
        "identity_primary_registered_target_best_face_top5": int(
            pd.to_numeric(
                eligible_primary["identity_registered_target_best_face_rank"],
                errors="coerce",
            ).le(5).sum()
        ),
        "identity_primary_registered_target_median_best_face_rank": float(
            pd.to_numeric(
                eligible_primary["identity_registered_target_best_face_rank"],
                errors="coerce",
            ).median()
        ),
        "identity_primary_registered_target_maxT_005": int(
            pd.to_numeric(
                eligible_primary["identity_registered_target_best_face_query_maxT_p"],
                errors="coerce",
            ).le(0.05).sum()
        ),
        "identity_hfd_selected": int(registered_target_audit.loc[
            registered_target_audit["relation_class"].eq(
                "prespecified_exact_intervention_family"
            ), "identity_registered_target_selected"
        ].sum()),
        "identity_ibd_selected": int(registered_target_audit.loc[
            registered_target_audit["relation_class"].eq(
                "prespecified_near_exact_condition_family"
            ), "identity_registered_target_selected"
        ].sum()),
        "identity_exact_drug_selected": int(registered_target_audit.loc[
            registered_target_audit["relation_class"].eq(
                "consumed_exact_drug_cross_design"
            ), "identity_registered_target_selected"
        ].sum()),
    }
    identity_edge_class_counts = (
        identity_method_edges["library_edge_class"].value_counts()
        .rename_axis("library_edge_class").reset_index(name="method_edge_rows")
    )
    entity_evidence_counts = (
        entity_library_evidence["library_evidence_class"].value_counts()
        .rename_axis("library_evidence_class").reset_index(name="entities")
    )
    report = {
        "schema": "contrast_atlas_global_biological_adjudication_v1",
        "status": "DEVELOPMENT_CONSUMED_GLOBAL_ADJUDICATION_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "contrast_nodes": int(len(nodes)),
            "connection_diagnostic_rows": int(len(connections)),
            "selected_answer_rows": int(len(answers)),
            "all_world_queue_rows": int(len(all_queue)),
            "identity_contribution_rows": 1315906,
            "relation_contribution_rows": 315584,
            "entity_library_evidence_rows": int(len(entity_library_evidence)),
            "identity_method_edge_annotation_rows": int(len(identity_method_edges)),
            "identity_unique_selected_links": int(len(selected_link_evidence)),
            "registered_target_directions": int(len(registered_target_audit)),
        },
        "all_world_by_layer": by_layer,
        "identity_global_diagnostics": {
            "selected_directions": int(len(identity_answered)),
            "unique_undirected_links": int(len(selected_link_evidence)),
            "opposite_directions": int(identity_answered["response_face"].eq("opposite").sum()),
            "same_parent_directions": int(identity_answered["same_parent_project"].sum()),
            "cross_parent_directions": int(len(identity_cross_parent)),
            "cross_parent_species_biospecimen_condition_all_mismatch": int((
                ~identity_cross_parent["same_species"]
                & ~identity_cross_parent["same_biospecimen"]
                & ~identity_cross_parent["same_condition_family"]
            ).sum()),
            "cross_parent_same_species_and_biospecimen": int((
                identity_cross_parent["same_species"]
                & identity_cross_parent["same_biospecimen"]
            ).sum()),
            "reciprocal_selected_directions": int(
                identity_answered["reciprocal_selected_answer"].sum()
            ),
            "all_unresolved_same_target_and_face": int(
                identity_answered["unresolved_same_selection_and_answered"].sum()
            ),
            "all_known_same_target_and_face": int(
                identity_answered["known_same_selection_and_answered"].sum()
            ),
            "cross_parent_top1_fraction_median": float(
                identity_cross_parent["selected_face_top1_strength_fraction"].median()
            ),
            "cross_parent_top10_fraction_median": float(
                identity_cross_parent["selected_face_top10_strength_fraction"].median()
            ),
            "cross_parent_effective_edge_count_median": float(
                identity_cross_parent["selected_face_effective_edge_count"].median()
            ),
            "cross_parent_dual_q_strength_fraction": float(
                identity_cross_parent["selected_face_dual_q_support_strength"].sum()
                / identity_cross_parent["selected_face_total_strength"].sum()
            ),
        },
        "entity_library_evidence_counts": records(entity_evidence_counts),
        "identity_method_edge_library_classes": records(identity_edge_class_counts),
        "identity_selected_answer_library_contribution": library_contribution,
        "registered_target_audit": registered_audit_summary,
        "source_concentration": source_concentration_summary,
        "cross_track": {
            "both_tracks_answered": int(convergence["both_tracks_answered"].sum()),
            "same_target": int(convergence["cross_track_same_target"].sum()),
            "same_target_and_face": int(
                convergence["cross_track_same_target_and_face"].sum()
            ),
        },
        "adjudication_strata": records(
            all_queue["adjudication_stratum"].value_counts()
            .rename_axis("adjudication_stratum").reset_index(name="query_layer_rows")
        ),
        "decision": {
            "atlas_globally_traceable": True,
            "biological_validity_established": False,
            "unknown_increment_established": False,
            "in_house_algorithm_increment_established": False,
            "signed_external_truth_available": False,
        },
        "claim_boundaries": [
            "Statistically selected hypotheses are not counts of correct biological links.",
            "Unregistered study pairs are unjudged, never negative controls.",
            "Identity and different-mass relation layers remain separate and are never meta-analysed as one molecule.",
            "The relation layer is a legacy fragment-cosine exploratory graph, not a validated Full23 relation score.",
            "Contribution concentration and cross-track convergence are prioritisation diagnostics, not validation.",
            "All inputs are development/consumed; no sealed biological truth was opened.",
        ],
        "provenance": {
            "implementation_sha256": sha256(Path(__file__)),
            "inputs": {str(path): sha256(path) for path in paths.values()},
            "common_identity_shards": {
                str(path): sha256(path)
                for path in sorted(
                    args.common_identity_shards.glob("*/identity_partition.tsv.gz")
                )
            },
            "outputs": output_hashes,
            "checksums_manifest_sha256": sha256(checksums_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    identity_source = next(
        row for row in source_concentration_summary
        if row["evidence_layer"] == IDENTITY_LAYER
    )
    identity_diagnostics = report["identity_global_diagnostics"]
    exact_fraction = library_contribution["exact_inchikey"]["fraction"]
    discordant_fraction = library_contribution[
        "discordant_named_identity"
    ]["fraction"]
    mixed_fraction = library_contribution["one_named_anchor"]["fraction"]
    unresolved_fraction = library_contribution[
        "unresolved_unresolved"
    ]["fraction"]
    layer_rows = "\n".join(
        "| {evidence_layer} | {statistically_selected_hypotheses}/51 | {same_parent_answers} | "
        "{cross_parent_answers} | {cross_parent_same_species_and_biospecimen} | "
        "{answers_targeting_chemprop_parent} |".format(**row)
        for row in by_layer
    )
    document = f"""# Contrast Atlas global biological adjudication v1

## Outcome and denominator

This release evaluates the complete **51-contrast, 15-source, development/consumed** Atlas at two explicitly separate evidence layers and expands every connection into all supporting entity-response contributions. It contains {len(connections):,} connection/world diagnostics, {len(answers):,} query/world/layer answer records, {len(entity_library_evidence):,} entity library-evidence records and {len(identity_method_edges):,} method-edge annotations. No selected hypothesis is labelled correct or false.

| Evidence layer | Selected / 51 | Same parent | Cross parent | Cross-parent same species+matrix | Targets ChemProp parent |
|---|---:|---:|---:|---:|---:|
{layer_rows}

The identity layer is a same-precursor WSE identity-hypothesis graph. The relation layer is the older different-mass mutual fragment-cosine graph and remains exploratory. Their agreement is a prioritisation signal only: {report['cross_track']['both_tracks_answered']} queries are answered by both, but only {report['cross_track']['same_target']} select the same target and response face.

## Global identity-track result

- The all-world identity track selects a statistical hypothesis for **{len(identity_answered)}/51** queries, representing **{len(selected_link_evidence)} unique undirected links**; **{identity_diagnostics['opposite_directions']}/{len(identity_answered)}** choose the opposite-response face.
- **{identity_diagnostics['same_parent_directions']}** answers remain within one operational parent project and **{identity_diagnostics['cross_parent_directions']}** cross parent projects. Among cross-parent answers, **{identity_diagnostics['cross_parent_species_biospecimen_condition_all_mismatch']}/{identity_diagnostics['cross_parent_directions']}** mismatch species, biospecimen and condition family simultaneously; **{identity_diagnostics['cross_parent_same_species_and_biospecimen']}** preserve both species and biospecimen.
- Candidate-source concentration is severe: parent-project HHI is **{identity_source['parent_project_hhi']:.3f}** (effective parent projects **{identity_source['effective_parent_projects']:.2f}**), and 36/49 answers target the shared ChemProp parent.
- The median cross-parent Top-1 and Top-10 contribution fractions are **{identity_diagnostics['cross_parent_top1_fraction_median']:.3%}** and **{identity_diagnostics['cross_parent_top10_fraction_median']:.3%}**. Median effective edge count is **{identity_diagnostics['cross_parent_effective_edge_count_median']:.1f}**. Thus most answers are distributed rather than one-edge artefacts, but this does not make them biologically valid.
- Only **{identity_diagnostics['cross_parent_dual_q_strength_fraction']:.2%}** of cross-parent selected-face strength comes from edges significant at study-internal q<=0.05 at both endpoints.

## What the unresolved evidence actually contributes

Across the 49 selected identity answers, selected-face strength is composed of:

| Endpoint evidence class | Strength fraction | Interpretation |
|---|---:|---|
| exact full InChIKey at both endpoints | {exact_fraction:.3%} | shared qualified library candidate; not MSI level 1 |
| discordant qualified identities | {discordant_fraction:.3%} | explicit identity conflict; never merge effects |
| one named endpoint only | {mixed_fraction:.3%} | named anchor only; never propagate the name |
| unresolved at both endpoints | {unresolved_fraction:.3%} | structure-agnostic evidence; correctness unproven |

The all-world and unresolved-world target plus face agree for **{identity_diagnostics['all_unresolved_same_target_and_face']}/{len(identity_answered)}** answered queries; the corresponding all-versus-known agreement is only **{identity_diagnostics['all_known_same_target_and_face']}/{len(identity_answered)}**. This proves numerical dependence on unresolved evidence, not unknown-information validity or increment.

Qualified library support is not equivalent to exact cross-study identity. Of the 1,785 double-supported method-edge rows across the three worlds, 417 have discordant InChIKeys. The released edge annotation therefore blocks name propagation and effect merging for those edges.

## Prespecified relation audit

The 42 registered directions are all unsigned and development/consumed. Among **{registered_audit_summary['eligible_primary_directions']}** eligible-primary directions, the identity track selects the registered target for only **{registered_audit_summary['identity_primary_targets_selected']}/{registered_audit_summary['eligible_primary_directions']}**:

- exact-drug cross-design: **{registered_audit_summary['identity_exact_drug_selected']}/24**;
- HFD intervention family: **{registered_audit_summary['identity_hfd_selected']}/4**;
- IBD condition family: **{registered_audit_summary['identity_ibd_selected']}/6**.

The registered target itself ranks Top-1 on its better face for **{registered_audit_summary['identity_primary_registered_target_best_face_top1']}/34** directions and Top-5 for **{registered_audit_summary['identity_primary_registered_target_best_face_top5']}/34** (median best-face rank **{registered_audit_summary['identity_primary_registered_target_median_best_face_rank']:.1f}**). Although **{registered_audit_summary['identity_primary_registered_target_maxT_005']}/34** target-face tests have query-maxT p<=0.05 before the final two-face choice, competing candidates dominate the open answer. Complete absence of spectral edges is therefore not the sole explanation; source attraction, response nonspecificity and candidate discrimination remain unresolved alternatives.

The two exact-drug hits are the reciprocal cilnidipine pair inside the consumed ChemProp parent project; they are a technical control, not external biological confirmation. The identity track also selects 2/8 post-hoc obesity directions. Those are the only same-species, same-condition cross-parent reciprocal candidates, but use different biofluids and have no dual-q supporting edge; they remain hypotheses.

## Released Atlas tables

- `identity_global_answer_adjudication.tsv`: every identity query, including two abstentions.
- `relation_global_answer_adjudication.tsv`: every exploratory different-mass relation query, including thirteen abstentions.
- `identity_selected_link_evidence.tsv.gz`: 44 unique selected links with complete contribution, concentration, missingness, library evidence and relation-axis diagnostics.
- `registered_target_audit.tsv`: all 42 registered directions, including every miss.
- `source_concentration.tsv`: candidate-source indegree, same/cross-parent counts and concentration.
- `entity_library_evidence.tsv.gz` and `identity_method_edge_library_annotation.tsv.gz`: normalised endpoint and edge evidence without copying names into the 1.32-million-row contribution matrix.

## What is now auditable

For every contrast pair, evidence world and response face, the connection table reports total strength, contributing edges, dual-q support, effective edge count, missing-effect fraction, Top-1/5/10 strength concentration and the known-known/mixed/unresolved-unresolved contribution fractions.  The all-world queue retains every query, including abstentions, and separates same-parent controls, registered unsigned audits, domain-matched tests, condition-only matches, cross-track convergence and cross-domain hypotheses.

This is the Atlas-wide interpretation layer that was previously missing. It prevents a Top-10 display, one IBD feature or one drug example from standing in for the global result.

## Scientific decision

The Atlas is now globally traceable, but **biological validity is not established**. The immediate P0 is to explain why the prespecified HFD and IBD targets are 0/10 and why exact-drug recovery is only 2/24 before promoting unregistered cross-domain links. The post-hoc obesity reciprocal pair is the first cross-parent candidate for independent adjudication; the B. subtilis-to-AD reciprocal pair and the ChemProp hub are bias diagnostics, not discoveries.

## Scientific boundary

All inputs are development/consumed. The collection contains no signed entity-level truth and no sealed biological validation panel. Source concentration, contribution concentration and track convergence do not establish mechanism or correctness. Official, Noise and Full23 still require complete-axis method-specific score packages on all 291,330 directed pairs before algorithm increment can be evaluated; the old 41,082-row fixed-WSE-edge packages are not admissible substitutes.
"""
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(document, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

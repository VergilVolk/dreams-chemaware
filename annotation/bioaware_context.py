"""Outcome-blind reaction-context features for BioAware candidate evidence."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from annotation.bioaware import validate_reaction_participants


def _noisy_or(values: np.ndarray) -> float:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - values)) if values.size else 0.0


def extract_reaction_context_features(
    candidates: pd.DataFrame,
    paths: pd.DataFrame,
    participants: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    truth_col: str | None = None,
    exclude_truth_identity: bool = False,
    reaction_directions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe Rhea evidence without using ranking outcomes.

    ``truth_col`` is used only to remove held-out truth identities during an
    evaluation audit.  It is never emitted as a feature.  Deployment calls
    must leave ``exclude_truth_identity=False``.
    """

    candidate_required = {"query_id", "candidate_id", "spectral_score"}
    seed_required = {"seed_query_id", "seed_compound_id", "seed_score"}
    path_required = {
        "candidate_id",
        "seed_compound_id",
        "seed_query_id",
        "reaction_id",
        "seed_side",
        "contribution",
    }
    for frame, required, name in [
        (candidates, candidate_required, "candidates"),
        (seeds, seed_required, "seeds"),
        (paths, path_required, "paths"),
    ]:
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if exclude_truth_identity and (not truth_col or truth_col not in candidates):
        raise ValueError("truth exclusion requires truth_col")

    participant = validate_reaction_participants(participants)
    noncurrency = participant[~participant["is_currency"].astype(bool)]
    side_compounds = {
        (str(reaction), str(side)): frozenset(group["compound_id"].astype(str))
        for (reaction, side), group in noncurrency.groupby(
            ["reaction_id", "side"], sort=False
        )
    }
    side_stoichiometry = {
        (str(reaction), str(side)): float(
            pd.to_numeric(group.get("stoichiometry", 1.0), errors="coerce")
            .fillna(1.0)
            .sum()
        )
        for (reaction, side), group in noncurrency.groupby(
            ["reaction_id", "side"], sort=False
        )
    }
    # Rhea also contains transport/state-transition records whose chemical
    # participants are identical on both sides.  They are meaningful reactions
    # operationally, but they carry zero information in a molecular-identity
    # graph and must never support candidate A over candidate B.
    side_identity_stoichiometry = {
        (str(reaction), str(side)): tuple(
            sorted(
                (
                    str(compound),
                    float(
                        pd.to_numeric(
                            group.loc[
                                group["compound_id"].astype(str) == str(compound),
                                "stoichiometry",
                            ],
                            errors="coerce",
                        )
                        .fillna(1.0)
                        .sum()
                    ),
                )
                for compound in group["compound_id"].astype(str).unique()
            )
        )
        for (reaction, side), group in noncurrency.groupby(
            ["reaction_id", "side"], sort=False
        )
    }
    reaction_ids = set(participant["reaction_id"].astype(str))
    identity_noop_reactions = {
        reaction
        for reaction in reaction_ids
        if side_identity_stoichiometry.get((reaction, "left"), ())
        == side_identity_stoichiometry.get((reaction, "right"), ())
        and side_identity_stoichiometry.get((reaction, "left"), ())
    }
    direction_semantics = {
        str(reaction): str(group.get("direction_semantics", pd.Series(["unknown"])).iloc[0])
        for reaction, group in participant.groupby("reaction_id", sort=False)
    }
    curated_directions: dict[str, frozenset[str]] = {}
    if reaction_directions is not None:
        required_direction_columns = {"MASTER_ID", "DIRECTION"}
        missing = required_direction_columns - set(reaction_directions)
        if missing:
            raise ValueError(
                f"reaction_directions missing columns: {sorted(missing)}"
            )
        directions = reaction_directions.copy()
        directions["MASTER_ID"] = directions["MASTER_ID"].astype(str)
        directions["DIRECTION"] = directions["DIRECTION"].astype(str).str.upper()
        unexpected = set(directions["DIRECTION"]) - {"LR", "RL", "BI", "UN"}
        if unexpected:
            raise ValueError(
                f"reaction_directions contains unexpected directions: {sorted(unexpected)}"
            )
        curated_directions = {
            str(reaction): frozenset(
                value
                for value in group["DIRECTION"].astype(str)
                if value != "UN"
            )
            for reaction, group in directions.groupby("MASTER_ID", sort=False)
        }
    seed_table = seeds.copy()
    seed_table["seed_query_id"] = seed_table["seed_query_id"].astype(str)
    seed_table["seed_compound_id"] = seed_table["seed_compound_id"].astype(str)
    raw_paths_by_candidate = {
        str(candidate): group for candidate, group in paths.groupby("candidate_id", sort=False)
    }
    valid_paths = paths[
        ~paths["reaction_id"].astype(str).isin(identity_noop_reactions)
    ].copy()
    paths_by_candidate = {
        str(candidate): group
        for candidate, group in valid_paths.groupby("candidate_id", sort=False)
    }
    feature_rows: list[dict] = []
    detail_rows: list[pd.DataFrame] = []

    for query_id, query_candidates in candidates.groupby("query_id", sort=False):
        truth = None
        if exclude_truth_identity:
            truths = query_candidates[truth_col].dropna().astype(str).unique()
            if len(truths) != 1:
                raise ValueError(f"query {query_id} must have one truth identity")
            truth = truths[0]
        available = seed_table[seed_table["seed_query_id"] != str(query_id)].copy()
        if truth is not None:
            available = available[available["seed_compound_id"] != truth]
        available_compounds = set(available["seed_compound_id"])

        # A seed/reaction path is candidate-specific only if it does not support
        # many competing candidates in the same query.  This is computed before
        # the per-candidate loop and does not use the truth label.
        query_candidate_ids = set(query_candidates["candidate_id"].astype(str))
        path_competition: Counter[tuple[str, str, str]] = Counter()
        for competing_candidate in query_candidate_ids:
            competing = paths_by_candidate.get(competing_candidate)
            if competing is None:
                continue
            competing = competing[
                (competing["seed_query_id"].astype(str) != str(query_id))
                & (
                    True
                    if truth is None
                    else competing["seed_compound_id"].astype(str) != truth
                )
            ]
            for key in set(
                zip(
                    competing["seed_compound_id"].astype(str),
                    competing["seed_query_id"].astype(str),
                    competing["reaction_id"].astype(str),
                )
            ):
                path_competition[key] += 1

        for candidate_id in query_candidates["candidate_id"].astype(str):
            raw_selected = raw_paths_by_candidate.get(candidate_id)
            if raw_selected is None:
                excluded_identity_noop_path_count = 0
            else:
                raw_selected = raw_selected[
                    (raw_selected["seed_query_id"].astype(str) != str(query_id))
                    & (
                        True
                        if truth is None
                        else raw_selected["seed_compound_id"].astype(str) != truth
                    )
                ]
                excluded_identity_noop_path_count = int(
                    raw_selected["reaction_id"]
                    .astype(str)
                    .isin(identity_noop_reactions)
                    .sum()
                )
            selected = paths_by_candidate.get(candidate_id)
            if selected is None:
                selected = paths.iloc[0:0].copy()
            else:
                selected = selected[
                    (selected["seed_query_id"].astype(str) != str(query_id))
                    & (
                        True
                        if truth is None
                        else selected["seed_compound_id"].astype(str) != truth
                    )
                ].copy()
            if len(selected):
                selected = selected.sort_values("contribution", ascending=False).drop_duplicates(
                    ["seed_compound_id", "reaction_id"]
                )
                required_counts: list[int] = []
                missing_counts: list[int] = []
                completeness: list[float] = []
                target_required_counts: list[int] = []
                target_missing_counts: list[int] = []
                target_completeness: list[float] = []
                source_stoichiometry: list[float] = []
                target_stoichiometry: list[float] = []
                physiological_direction_available: list[bool] = []
                curated_direction_supported: list[bool] = []
                curated_direction_conflicted: list[bool] = []
                competing_candidate_counts: list[int] = []
                candidate_specificity: list[float] = []
                signatures: list[str] = []
                target_signatures: list[str] = []
                for row in selected.itertuples(index=False):
                    source = side_compounds.get(
                        (str(row.reaction_id), str(row.seed_side)), frozenset()
                    )
                    required = set(source) - {str(row.seed_compound_id)}
                    missing = sorted(required - available_compounds)
                    target = side_compounds.get(
                        (str(row.reaction_id), str(row.candidate_side)), frozenset()
                    )
                    required_target = set(target) - {candidate_id}
                    missing_target = sorted(required_target - available_compounds)
                    required_counts.append(len(required))
                    missing_counts.append(len(missing))
                    completeness.append(
                        1.0 if not required else (len(required) - len(missing)) / len(required)
                    )
                    target_required_counts.append(len(required_target))
                    target_missing_counts.append(len(missing_target))
                    target_completeness.append(
                        1.0
                        if not required_target
                        else (len(required_target) - len(missing_target))
                        / len(required_target)
                    )
                    source_stoichiometry.append(
                        side_stoichiometry.get(
                            (str(row.reaction_id), str(row.seed_side)), 0.0
                        )
                    )
                    target_stoichiometry.append(
                        side_stoichiometry.get(
                            (str(row.reaction_id), str(row.candidate_side)), 0.0
                        )
                    )
                    semantics = direction_semantics.get(str(row.reaction_id), "unknown")
                    curated = curated_directions.get(str(row.reaction_id), frozenset())
                    path_direction = (
                        "LR"
                        if str(row.seed_side) == "left"
                        and str(row.candidate_side) == "right"
                        else "RL"
                    )
                    has_curated_direction = bool(curated)
                    supported_direction = "BI" in curated or path_direction in curated
                    physiological_direction_available.append(
                        has_curated_direction
                        or semantics
                        not in {"", "unknown", "canonical_lr_not_physiological"}
                    )
                    curated_direction_supported.append(
                        bool(has_curated_direction and supported_direction)
                    )
                    curated_direction_conflicted.append(
                        bool(has_curated_direction and not supported_direction)
                    )
                    competition_key = (
                        str(row.seed_compound_id),
                        str(row.seed_query_id),
                        str(row.reaction_id),
                    )
                    competition = max(1, int(path_competition.get(competition_key, 1)))
                    competing_candidate_counts.append(competition)
                    candidate_specificity.append(1.0 / competition)
                    signatures.append(";".join(missing))
                    target_signatures.append(";".join(missing_target))
                selected["required_source_compound_count"] = required_counts
                selected["missing_source_compound_count"] = missing_counts
                selected["source_side_completeness"] = completeness
                selected["missing_source_signature"] = signatures
                selected["source_side_complete"] = np.asarray(missing_counts) == 0
                selected["required_target_coproduct_count"] = target_required_counts
                selected["missing_target_coproduct_count"] = target_missing_counts
                selected["target_side_completeness"] = target_completeness
                selected["missing_target_signature"] = target_signatures
                selected["target_side_complete"] = np.asarray(target_missing_counts) == 0
                selected["source_side_noncurrency_stoichiometry"] = source_stoichiometry
                selected["target_side_noncurrency_stoichiometry"] = target_stoichiometry
                selected["physiological_direction_available"] = physiological_direction_available
                selected["curated_direction_supported"] = curated_direction_supported
                selected["curated_direction_conflicted"] = curated_direction_conflicted
                selected["competing_query_candidate_count"] = competing_candidate_counts
                selected["candidate_specificity"] = candidate_specificity
                selected["specificity_weighted_contribution"] = (
                    selected["contribution"].astype(float)
                    * selected["candidate_specificity"].astype(float)
                )
                selected["direction_supported_contribution"] = (
                    selected["specificity_weighted_contribution"].astype(float)
                    * selected["curated_direction_supported"].astype(float)
                )
                selected.insert(0, "query_id", str(query_id))
                selected.insert(1, "query_candidate_id", candidate_id)
                detail_rows.append(selected)

            complete = selected[selected.get("source_side_complete", False)] if len(selected) else selected
            incomplete = selected[~selected.get("source_side_complete", True)] if len(selected) else selected
            fully_observed = (
                selected[
                    selected["source_side_complete"]
                    & selected["target_side_complete"]
                ]
                if len(selected)
                else selected
            )
            signature_counts = Counter(
                value for value in incomplete.get("missing_source_signature", []) if value
            )
            dominant_signature_fraction = (
                max(signature_counts.values()) / len(incomplete)
                if len(incomplete) and signature_counts
                else 0.0
            )
            dependency_groups: dict[str, float] = {}
            if len(selected):
                for row in selected.itertuples(index=False):
                    if bool(row.source_side_complete):
                        # Multiple Rhea records supported by the same observed
                        # seed are not independent biological observations.
                        key = f"complete_seed:{row.seed_compound_id}"
                    else:
                        # Paths sharing an absent co-substrate depend on the
                        # same unobserved event, even when their seed differs.
                        key = f"missing:{row.missing_source_signature}"
                    dependency_groups[key] = max(
                        dependency_groups.get(key, 0.0), float(row.contribution)
                    )
            feature_rows.append(
                {
                    "query_id": str(query_id),
                    "candidate_id": candidate_id,
                    "raw_network_support": _noisy_or(
                        selected["contribution"].to_numpy(float)
                    ) if len(selected) else 0.0,
                    "complete_network_support": _noisy_or(
                        complete["contribution"].to_numpy(float)
                    ) if len(complete) else 0.0,
                    "incomplete_network_support": _noisy_or(
                        incomplete["contribution"].to_numpy(float)
                    ) if len(incomplete) else 0.0,
                    "dependency_corrected_network_support": _noisy_or(
                        np.asarray(list(dependency_groups.values()), dtype=float)
                    ),
                    "candidate_specific_network_support": _noisy_or(
                        selected["specificity_weighted_contribution"].to_numpy(float)
                    ) if len(selected) else 0.0,
                    "complete_candidate_specific_network_support": _noisy_or(
                        complete["specificity_weighted_contribution"].to_numpy(float)
                    ) if len(complete) else 0.0,
                    "direction_supported_network_support": _noisy_or(
                        selected["direction_supported_contribution"].to_numpy(float)
                    ) if len(selected) else 0.0,
                    "complete_direction_supported_network_support": _noisy_or(
                        complete["direction_supported_contribution"].to_numpy(float)
                    ) if len(complete) else 0.0,
                    "fully_observed_direction_supported_network_support": _noisy_or(
                        fully_observed["direction_supported_contribution"].to_numpy(float)
                    ) if len(fully_observed) else 0.0,
                    "raw_path_count": int(len(selected)),
                    "excluded_identity_noop_path_count": excluded_identity_noop_path_count,
                    "complete_path_count": int(len(complete)),
                    "target_complete_path_count": int(
                        selected["target_side_complete"].sum()
                    ) if len(selected) else 0,
                    "fully_observed_hyperedge_path_count": int(
                        (
                            selected["source_side_complete"]
                            & selected["target_side_complete"]
                        ).sum()
                    ) if len(selected) else 0,
                    "incomplete_path_count": int(len(incomplete)),
                    "unique_seed_compounds": int(
                        selected["seed_compound_id"].astype(str).nunique()
                    ) if len(selected) else 0,
                    "complete_seed_compounds": int(
                        complete["seed_compound_id"].astype(str).nunique()
                    ) if len(complete) else 0,
                    "unique_reactions": int(
                        selected["reaction_id"].astype(str).nunique()
                    ) if len(selected) else 0,
                    "unique_missing_signatures": int(len(signature_counts)),
                    "dependency_group_count": int(len(dependency_groups)),
                    "largest_dependency_group_fraction": float(
                        max(Counter(
                            (
                                f"complete_seed:{row.seed_compound_id}"
                                if bool(row.source_side_complete)
                                else f"missing:{row.missing_source_signature}"
                            )
                            for row in selected.itertuples(index=False)
                        ).values()) / len(selected)
                    ) if len(selected) else 0.0,
                    "dominant_missing_signature_fraction": float(
                        dominant_signature_fraction
                    ),
                    "mean_source_side_completeness": float(
                        selected["source_side_completeness"].mean()
                    ) if len(selected) else 0.0,
                    "mean_target_side_completeness": float(
                        selected["target_side_completeness"].mean()
                    ) if len(selected) else 0.0,
                    "mean_candidate_specificity": float(
                        selected["candidate_specificity"].mean()
                    ) if len(selected) else 0.0,
                    "minimum_competing_query_candidates": int(
                        selected["competing_query_candidate_count"].min()
                    ) if len(selected) else 0,
                    "physiological_direction_path_fraction": float(
                        selected["physiological_direction_available"].mean()
                    ) if len(selected) else 0.0,
                    "curated_direction_supported_path_count": int(
                        selected["curated_direction_supported"].sum()
                    ) if len(selected) else 0,
                    "curated_direction_conflicted_path_count": int(
                        selected["curated_direction_conflicted"].sum()
                    ) if len(selected) else 0,
                    "minimum_missing_source_compounds": int(
                        selected["missing_source_compound_count"].min()
                    ) if len(selected) else 0,
                    "maximum_path_contribution": float(
                        selected["contribution"].max()
                    ) if len(selected) else 0.0,
                    "maximum_complete_path_contribution": float(
                        complete["contribution"].max()
                    ) if len(complete) else 0.0,
                }
            )
    features = pd.DataFrame(feature_rows)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if features.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("reaction-context extraction produced duplicate candidates")
    return features, details

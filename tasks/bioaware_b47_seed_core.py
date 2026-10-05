#!/usr/bin/env python
"""Pure tabular core for truth-blind B47 spectral seeds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeedPolicy:
    name: str
    minimum_score: float
    minimum_margin: float
    minimum_modal_samples: int
    minimum_modal_fraction: float


PRIMARY_POLICY = SeedPolicy("primary", 0.80, 0.05, 2, 0.80)
STRICT_POLICY = SeedPolicy("strict", 0.90, 0.10, 3, 0.90)


def query_summaries(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    required = {"query_id", "candidate_id", "spectral_score"}
    if not required.issubset(candidate_scores.columns):
        raise ValueError(f"candidate scores miss {sorted(required-set(candidate_scores.columns))}")
    rows: list[dict] = []
    for query_id, group in candidate_scores.groupby("query_id", sort=False):
        ordered = group.sort_values(
            ["spectral_score", "candidate_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        if len(ordered) < 2:
            raise ValueError(f"query has fewer than two candidates: {query_id}")
        top_score = float(ordered.loc[0, "spectral_score"])
        tied = np.isclose(
            ordered["spectral_score"].to_numpy(float), top_score, atol=1e-12, rtol=0
        )
        unique_top = int(tied.sum()) == 1
        second_score = float(ordered.loc[1, "spectral_score"])
        rows.append({
            "query_id": str(query_id),
            "top_candidate_id": str(ordered.loc[0, "candidate_id"]) if unique_top else "",
            "top_score": top_score,
            "second_score": second_score,
            "top_margin": top_score - second_score,
            "top_tie_count": int(tied.sum()),
            "unique_top1": unique_top,
            "candidate_count": int(len(ordered)),
        })
    return pd.DataFrame(rows)


def attach_absolute_gates(summary: pd.DataFrame, policies: tuple[SeedPolicy, ...]) -> pd.DataFrame:
    output = summary.copy()
    for policy in policies:
        output[f"{policy.name}_absolute_gate"] = (
            output["unique_top1"].astype(bool)
            & (output["top_score"].astype(float) >= policy.minimum_score)
            & (output["top_margin"].astype(float) >= policy.minimum_margin)
        )
    return output


def feature_consensus(
    query: pd.DataFrame, policies: tuple[SeedPolicy, ...]
) -> pd.DataFrame:
    required = {"study", "sample", "feature_id", "query_id", "top_candidate_id", "unique_top1"}
    if not required.issubset(query.columns):
        raise ValueError(f"query summaries miss {sorted(required-set(query.columns))}")
    rows: list[dict] = []
    for (study, feature_id), group in query.groupby(["study", "feature_id"], sort=False):
        if group["sample"].astype(str).duplicated().any():
            raise ValueError(f"feature has repeated sample event: {study}|{feature_id}")
        valid = group[group["unique_top1"].astype(bool) & group["top_candidate_id"].ne("")]
        counts = valid["top_candidate_id"].value_counts()
        modal = ""
        modal_support = 0
        modal_tied = False
        if len(counts):
            modal_support = int(counts.iloc[0])
            modal_tied = int((counts == modal_support).sum()) > 1
            if not modal_tied:
                modal = str(counts.index[0])
        row = {
            "study": str(study), "feature_id": str(feature_id),
            "observed_sample_events": int(group["sample"].nunique()),
            "unique_top1_events": int(len(valid)),
            "modal_candidate_id": modal,
            "modal_support_samples": modal_support,
            "modal_fraction_all_events": (
                float(modal_support / len(group)) if len(group) else 0.0
            ),
            "modal_identity_tied": modal_tied,
        }
        for policy in policies:
            absolute = f"{policy.name}_absolute_gate"
            if absolute not in group:
                raise ValueError(f"query summaries miss {absolute}")
            confident_modal = int(
                (group["top_candidate_id"].eq(modal) & group[absolute].astype(bool)).sum()
            ) if modal else 0
            row[f"{policy.name}_confident_modal_samples"] = confident_modal
            row[f"{policy.name}_feature_gate"] = bool(
                modal
                and not modal_tied
                and modal_support >= policy.minimum_modal_samples
                and confident_modal >= policy.minimum_modal_samples
                and row["modal_fraction_all_events"] >= policy.minimum_modal_fraction
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_sample_seeds(
    query: pd.DataFrame,
    consensus: pd.DataFrame,
    policy: SeedPolicy,
    graph_degree: dict[str, int],
    currency: set[str],
    maximum_degree: int = 250,
) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = query.merge(
        consensus, on=["study", "feature_id"], how="left", validate="many_to_one"
    )
    absolute = f"{policy.name}_absolute_gate"
    feature_gate = f"{policy.name}_feature_gate"
    selected = merged[
        merged[absolute].astype(bool)
        & merged[feature_gate].astype(bool)
        & merged["top_candidate_id"].eq(merged["modal_candidate_id"])
    ].copy()
    before_graph = len(selected)
    selected["reaction_degree"] = selected["top_candidate_id"].map(graph_degree).fillna(0).astype(int)
    selected["is_currency"] = selected["top_candidate_id"].isin(currency)
    selected = selected[
        (selected["reaction_degree"] > 0)
        & (selected["reaction_degree"] <= maximum_degree)
        & (~selected["is_currency"])
    ].copy()
    before_collapse = len(selected)
    # A candidate identity can be represented by isotope/adduct/in-source forms.
    # Until an explicit ion-family graph is available, retaining only its most
    # confident event per sample is the conservative independent-support unit.
    selected = (
        selected.sort_values(
            ["study", "sample", "top_candidate_id", "top_score", "top_margin", "query_id"],
            ascending=[True, True, True, False, False, True], kind="stable",
        )
        .drop_duplicates(["study", "sample", "top_candidate_id"], keep="first")
        .reset_index(drop=True)
    )
    output = pd.DataFrame({
        "study": selected["study"].astype(str),
        "sample": selected["sample"].astype(str),
        "seed_query_id": selected["query_id"].astype(str),
        "seed_feature_id": selected["feature_id"].astype(str),
        "seed_compound_id": selected["top_candidate_id"].astype(str),
        "seed_score": selected["top_score"].astype(float).clip(0, 1),
        "seed_margin": selected["top_margin"].astype(float),
        "feature_support_samples": selected["modal_support_samples"].astype(int),
        "feature_consensus_fraction": selected["modal_fraction_all_events"].astype(float),
        "reaction_degree": selected["reaction_degree"].astype(int),
        "seed_policy": policy.name,
        "independent_support_unit": "sample_candidate_identity",
    })
    audit = {
        "events_after_spectral_and_consensus_gates": int(before_graph),
        "events_after_reaction_graph_safety": int(before_collapse),
        "rows_after_sample_candidate_collapse": int(len(output)),
        "rows_removed_by_sample_candidate_collapse": int(before_collapse - len(output)),
    }
    return output, audit


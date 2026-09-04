"""Shared scoring primitives for the BioAware V6 identifiability router."""
from __future__ import annotations

import numpy as np
import pandas as pd

from develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES,
    add_family_features,
    score_queries,
)


EPS = 1e-12
BIOLOGICAL_MECHANISMS = {
    "reaction_network": ("family_known_reaction", "family_predicted_reaction"),
    "structure_network": ("family_structure_network",),
}


def apply_identifiable_router(
    ledger: pd.DataFrame, weights: np.ndarray, *, maximum_spectral_margin: float,
    minimum_fusion_advantage: float, minimum_support_families: int,
    minimum_unique_biological_mechanisms: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_family_features(ledger)
    predictions = score_queries(scored, weights)
    candidate_parts: list[pd.DataFrame] = []
    identifiability_rows: list[dict] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        group = group.copy()
        proposed_id = str(
            predictions.loc[predictions.query_id.astype(str) == str(query_id), "proposed_candidate_id"].iloc[0]
        )
        unique_mechanisms: list[str] = []
        for name, columns in BIOLOGICAL_MECHANISMS.items():
            values = group[list(columns)].max(axis=1).to_numpy(float)
            group[f"mechanism_{name}"] = values
            proposed_position = np.flatnonzero(group.candidate_id.astype(str).to_numpy() == proposed_id)
            if len(proposed_position) != 1:
                raise RuntimeError(f"{query_id}: proposed candidate is not unique in ledger")
            proposed_value = float(values[int(proposed_position[0])])
            competitors = np.delete(values, int(proposed_position[0]))
            second = float(np.max(competitors)) if len(competitors) else -np.inf
            if proposed_value > EPS and proposed_value > second + EPS:
                unique_mechanisms.append(name)
        candidate_parts.append(group)
        identifiability_rows.append({
            "query_id": str(query_id),
            "unique_biological_mechanism_count": len(unique_mechanisms),
            "unique_biological_mechanisms": ";".join(unique_mechanisms),
            "biologically_identifiable": len(unique_mechanisms) >= minimum_unique_biological_mechanisms,
        })
    scored = pd.concat(candidate_parts, ignore_index=True)
    identifiability = pd.DataFrame(identifiability_rows)
    result = predictions.merge(identifiability, on="query_id", how="left", validate="one_to_one")
    result["intervene"] = (
        result.changes_top1.astype(bool)
        & result.proposed_unique.astype(bool)
        & (result.spectral_margin.astype(float) <= maximum_spectral_margin + EPS)
        & (result.fusion_advantage.astype(float) >= minimum_fusion_advantage - EPS)
        & (result.support_count.astype(int) >= minimum_support_families)
        & result.biologically_identifiable.astype(bool)
    )
    result["final_candidate_id"] = result.baseline_candidate_id.astype(str).where(
        ~result.intervene, result.proposed_candidate_id.astype(str)
    )
    truth = result.truth_candidate_id.astype(str)
    # Exact fallback includes the frozen tie semantics.  A baseline tie that
    # counts against the positive cannot silently become correct merely because
    # the deterministic display candidate happens to equal the truth.
    result["final_correct"] = np.where(
        result.intervene.astype(bool),
        result.final_candidate_id.astype(str).eq(truth),
        result.baseline_correct.astype(bool),
    )
    result["corrected"] = ~result.baseline_correct.astype(bool) & result.final_correct.astype(bool)
    result["introduced"] = result.baseline_correct.astype(bool) & ~result.final_correct.astype(bool)
    result["delta"] = result.final_correct.astype(int) - result.baseline_correct.astype(int)
    return result, scored


def weights_from_artifact(artifact: dict) -> np.ndarray:
    raw = artifact["router"]["weights"]
    return np.asarray([float(raw[name]) for name in FAMILY_FEATURES], dtype=float)

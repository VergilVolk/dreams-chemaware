"""BioAware v1: conservative biochemical-network evidence after embedding.

This module deliberately does *not* train or alter the DreaMS/ChemAware
embedding.  It adds an auditable, cohort-contextual evidence layer to a frozen
candidate table.  The implementation is intentionally small and falsifiable:

* reactions remain explicit hyperedges (compound -> reaction -> compound);
* propagation is one reaction hop by default;
* high-degree/currency compounds are down-weighted or excluded as seeds;
* a query cannot support itself (leave-query-out), and formal evaluation can
  additionally remove every seed with the held-out truth identity;
* a network disagreement may change Top-1 only for a low-margin spectral query;
* every change has a reaction/seed explanation row.

Biological condition labels, differential-abundance statistics and pathway
enrichment results are not accepted by this API.  This avoids using a disease
hypothesis to choose an identity and then using that identity to prove the same
hypothesis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


REQUIRED_PARTICIPANT_COLUMNS = {"reaction_id", "side", "compound_id"}
REQUIRED_CANDIDATE_COLUMNS = {"query_id", "candidate_id", "spectral_score"}
REQUIRED_SEED_COLUMNS = {"seed_compound_id", "seed_score"}


@dataclass(frozen=True)
class BioAwareConfig:
    """Frozen v1 scoring policy.

    The default network weight is deliberately bounded.  A larger gain should
    come from better independent evidence, not from allowing the graph to
    overwhelm a confident spectrum.
    """

    directed: bool = False
    network_weight: float = 0.15
    maximum_spectral_margin_for_override: float = 0.05
    minimum_network_advantage: float = 0.10
    minimum_seed_score: float = 0.80
    maximum_seed_degree: int = 250
    degree_exponent: float = 0.50
    minimum_spectral_score_for_identity_claim: float = 0.50
    numerical_epsilon: float = 1e-12

    def to_dict(self) -> dict:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_reaction_participants(participants: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a reaction-participant hypergraph table."""

    _require_columns(participants, REQUIRED_PARTICIPANT_COLUMNS, "participants")
    out = participants.copy()
    out["reaction_id"] = out["reaction_id"].astype(str)
    out["compound_id"] = out["compound_id"].fillna("").astype(str)
    out["side"] = out["side"].astype(str).str.lower()
    if not set(out["side"]).issubset({"left", "right"}):
        bad = sorted(set(out["side"]) - {"left", "right"})
        raise ValueError(f"participants.side must be left/right, got {bad}")
    if (out["compound_id"].str.len() == 0).any():
        raise ValueError("participants contains empty compound_id")
    if "reaction_weight" not in out:
        out["reaction_weight"] = 1.0
    out["reaction_weight"] = pd.to_numeric(out["reaction_weight"], errors="raise")
    if (~np.isfinite(out["reaction_weight"]) | (out["reaction_weight"] <= 0)).any():
        raise ValueError("reaction_weight must be finite and positive")
    if "is_currency" not in out:
        out["is_currency"] = False
    out["is_currency"] = out["is_currency"].map(lambda value: bool(value) if pd.notna(value) else False)
    # Repeated participants caused by stoichiometry do not create independent
    # support.  Stoichiometry remains a metadata column when provided.
    out = out.drop_duplicates(["reaction_id", "side", "compound_id"]).reset_index(drop=True)
    valid_sides = out.groupby("reaction_id")["side"].nunique()
    invalid = set(valid_sides[valid_sides < 2].index)
    if invalid:
        out = out[~out["reaction_id"].isin(invalid)].reset_index(drop=True)
    if out.empty:
        raise ValueError("participants contains no reaction with both sides")
    return out


def compound_reaction_degree(participants: pd.DataFrame) -> pd.Series:
    """Number of distinct reactions containing each compound."""

    p = validate_reaction_participants(participants)
    return p.groupby("compound_id")["reaction_id"].nunique().astype(int)


def build_one_hop_evidence(
    participants: pd.DataFrame,
    seeds: pd.DataFrame,
    config: BioAwareConfig = BioAwareConfig(),
) -> pd.DataFrame:
    """Return seed->reaction->candidate evidence paths.

    The returned table is not yet query-specific.  Query/self-identity exclusion
    is applied later by :func:`aggregate_query_support`, so the same path cache
    can be reused for strict leave-one-out evaluation.
    """

    p = validate_reaction_participants(participants)
    _require_columns(seeds, REQUIRED_SEED_COLUMNS, "seeds")
    s = seeds.copy()
    s["seed_compound_id"] = s["seed_compound_id"].fillna("").astype(str)
    s["seed_score"] = pd.to_numeric(s["seed_score"], errors="raise")
    if (~np.isfinite(s["seed_score"]) | (s["seed_score"] < 0) | (s["seed_score"] > 1)).any():
        raise ValueError("seed_score must be finite and in [0, 1]")
    if "seed_query_id" not in s:
        s["seed_query_id"] = ""
    s["seed_query_id"] = s["seed_query_id"].fillna("").astype(str)

    degree = compound_reaction_degree(p)
    currency = set(p.loc[p["is_currency"], "compound_id"])
    s["seed_degree"] = s["seed_compound_id"].map(degree).fillna(0).astype(int)
    s = s[
        (s["seed_score"] >= config.minimum_seed_score)
        & (s["seed_degree"] > 0)
        & (s["seed_degree"] <= config.maximum_seed_degree)
        & (~s["seed_compound_id"].isin(currency))
    ].copy()
    if s.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id", "seed_compound_id", "seed_query_id", "reaction_id",
                "seed_side", "candidate_side", "seed_score", "contribution",
                "seed_degree", "candidate_degree", "reaction_size", "reaction_weight",
            ]
        )

    seeds_by_compound: dict[str, list[tuple[str, float]]] = {}
    for compound, group in s.groupby("seed_compound_id", sort=False):
        # Multiple independent source queries are retained for leave-query-out.
        seeds_by_compound[str(compound)] = [
            (str(row.seed_query_id), float(row.seed_score))
            for row in group.itertuples(index=False)
        ]

    rows: list[dict] = []
    for reaction_id, reaction in p.groupby("reaction_id", sort=False):
        left = reaction.loc[reaction["side"] == "left", "compound_id"].tolist()
        right = reaction.loc[reaction["side"] == "right", "compound_id"].tolist()
        reaction_size = len(set(left + right))
        reaction_weight = float(reaction["reaction_weight"].min())
        directions: list[tuple[str, list[str], str, list[str]]] = [
            ("left", left, "right", right)
        ]
        if not config.directed:
            directions.append(("right", right, "left", left))
        for seed_side, seed_compounds, candidate_side, candidates in directions:
            for seed_compound in set(seed_compounds):
                source_rows = seeds_by_compound.get(seed_compound)
                if not source_rows:
                    continue
                seed_degree = int(degree.get(seed_compound, 0))
                for candidate in set(candidates):
                    if candidate == seed_compound:
                        continue
                    candidate_degree = int(degree.get(candidate, 0))
                    if candidate_degree <= 0:
                        continue
                    degree_factor = (
                        max(1, seed_degree) * max(1, candidate_degree)
                    ) ** (-config.degree_exponent / 2.0)
                    size_factor = 1.0 / np.sqrt(max(1, reaction_size - 1))
                    for seed_query_id, seed_score in source_rows:
                        contribution = float(
                            np.clip(seed_score * reaction_weight * degree_factor * size_factor, 0.0, 1.0)
                        )
                        rows.append(
                            {
                                "candidate_id": candidate,
                                "seed_compound_id": seed_compound,
                                "seed_query_id": seed_query_id,
                                "reaction_id": str(reaction_id),
                                "seed_side": seed_side,
                                "candidate_side": candidate_side,
                                "seed_score": seed_score,
                                "contribution": contribution,
                                "seed_degree": seed_degree,
                                "candidate_degree": candidate_degree,
                                "reaction_size": reaction_size,
                                "reaction_weight": reaction_weight,
                            }
                        )
    return pd.DataFrame(rows)


def _noisy_or(values: Iterable[float]) -> float:
    arr = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
    if arr.size == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - arr))


def aggregate_query_support(
    candidates: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    truth_col: str | None = None,
    exclude_same_query: bool = True,
    exclude_truth_identity: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach leakage-safe network support to each query/candidate row.

    If ``exclude_truth_identity`` is enabled, ``truth_col`` must contain the
    held-out true compound identifier on every candidate row of a query.  This
    is an evaluation-only operation and must never be used in deployment.
    """

    _require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "candidates")
    c = candidates.copy()
    c["query_id"] = c["query_id"].astype(str)
    c["candidate_id"] = c["candidate_id"].astype(str)
    c["spectral_score"] = pd.to_numeric(c["spectral_score"], errors="raise")
    if c.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("candidates must have one row per query_id/candidate_id")
    if exclude_truth_identity:
        if truth_col is None or truth_col not in c:
            raise ValueError("exclude_truth_identity requires truth_col")
        truth_counts = c.groupby("query_id")[truth_col].nunique(dropna=False)
        if (truth_counts != 1).any():
            raise ValueError("truth_col must be constant within each query")

    if paths.empty:
        c["network_support"] = 0.0
        c["network_path_count"] = 0
        return c, paths.copy()
    required_paths = {
        "candidate_id", "seed_compound_id", "seed_query_id", "reaction_id", "contribution"
    }
    _require_columns(paths, required_paths, "paths")

    support_rows: list[dict] = []
    used_paths: list[pd.DataFrame] = []
    paths_by_candidate = {key: value for key, value in paths.groupby("candidate_id", sort=False)}
    for query_id, group in c.groupby("query_id", sort=False):
        truth = str(group[truth_col].iloc[0]) if exclude_truth_identity and truth_col else None
        for candidate_id in group["candidate_id"]:
            selected = paths_by_candidate.get(candidate_id)
            if selected is None:
                support_rows.append(
                    {"query_id": query_id, "candidate_id": candidate_id, "network_support": 0.0, "network_path_count": 0}
                )
                continue
            selected = selected.copy()
            if exclude_same_query:
                selected = selected[selected["seed_query_id"].astype(str) != str(query_id)]
            if truth is not None:
                selected = selected[selected["seed_compound_id"].astype(str) != truth]
            # The same seed through duplicate reaction records is one biological
            # item of evidence; retain its strongest path before noisy-or.
            selected = (
                selected.sort_values("contribution", ascending=False)
                .drop_duplicates(["seed_compound_id", "reaction_id"])
            )
            support_rows.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "network_support": _noisy_or(selected["contribution"]),
                    "network_path_count": int(len(selected)),
                }
            )
            if not selected.empty:
                selected.insert(0, "query_id", query_id)
                selected.insert(1, "query_candidate_id", candidate_id)
                used_paths.append(selected)
    support = pd.DataFrame(support_rows)
    out = c.merge(support, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    out["network_support"] = out["network_support"].fillna(0.0)
    out["network_path_count"] = out["network_path_count"].fillna(0).astype(int)
    explanations = pd.concat(used_paths, ignore_index=True) if used_paths else paths.iloc[0:0].copy()
    return out, explanations


def fuse_candidates(
    candidates_with_support: pd.DataFrame,
    config: BioAwareConfig = BioAwareConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a conservative, query-level BioAware gate and rerank candidates."""

    required = REQUIRED_CANDIDATE_COLUMNS | {"network_support", "network_path_count"}
    _require_columns(candidates_with_support, required, "candidates_with_support")
    c = candidates_with_support.copy()
    c["query_id"] = c["query_id"].astype(str)
    c["candidate_id"] = c["candidate_id"].astype(str)
    rows: list[pd.DataFrame] = []
    query_rows: list[dict] = []
    for query_id, group in c.groupby("query_id", sort=False):
        group = group.copy()
        # Deterministic conservative tie rule: candidate_id only makes ordering
        # reproducible; a tied positive is not considered uniquely Top-1 by the
        # formal evaluator.
        baseline = group.sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
        baseline_top = baseline.iloc[0]
        second_score = float(baseline["spectral_score"].iloc[1]) if len(baseline) > 1 else -np.inf
        spectral_margin = float(baseline_top.spectral_score - second_score)
        network = group.sort_values(["network_support", "candidate_id"], ascending=[False, True])
        network_top = network.iloc[0]
        second_network = float(network["network_support"].iloc[1]) if len(network) > 1 else 0.0
        network_advantage = float(network_top.network_support - second_network)
        network_available = bool(float(network_top.network_support) > 0 and int(network_top.network_path_count) > 0)
        disagreement = str(network_top.candidate_id) != str(baseline_top.candidate_id)
        low_margin = spectral_margin <= config.maximum_spectral_margin_for_override
        strong_network = network_advantage >= config.minimum_network_advantage
        gate_open = bool(network_available and low_margin and strong_network)

        group["raw_fused_score"] = (
            group["spectral_score"].astype(float)
            + config.network_weight * group["network_support"].astype(float)
        )
        raw_top = group.sort_values(["raw_fused_score", "candidate_id"], ascending=[False, True]).iloc[0]
        changed_by_raw = str(raw_top.candidate_id) != str(baseline_top.candidate_id)
        apply = bool(gate_open and changed_by_raw)
        group["final_score"] = group["raw_fused_score"] if apply else group["spectral_score"].astype(float)
        group["bioaware_applied"] = apply
        group["spectral_margin"] = spectral_margin
        group["network_advantage"] = network_advantage
        group["network_top_candidate"] = str(network_top.candidate_id)
        group["spectral_top_candidate"] = str(baseline_top.candidate_id)
        group["network_disagreement"] = disagreement
        if not network_available:
            state = "no_network_evidence"
        elif disagreement and not low_margin:
            state = "spectral_strong_network_conflict"
        elif disagreement and not strong_network:
            state = "network_ambiguous"
        elif apply:
            state = "network_supported_override"
        elif disagreement:
            state = "network_supported_no_rank_change"
        else:
            state = "spectral_network_agree"
        group["evidence_state"] = state
        final = group.sort_values(["final_score", "candidate_id"], ascending=[False, True])
        final_top = final.iloc[0]
        claim_allowed = float(final_top.spectral_score) >= config.minimum_spectral_score_for_identity_claim
        query_rows.append(
            {
                "query_id": query_id,
                "baseline_top_candidate": str(baseline_top.candidate_id),
                "network_top_candidate": str(network_top.candidate_id),
                "final_top_candidate": str(final_top.candidate_id),
                "spectral_margin": spectral_margin,
                "network_advantage": network_advantage,
                "network_available": network_available,
                "network_disagreement": disagreement,
                "bioaware_applied": apply,
                "evidence_state": state,
                "identity_claim_allowed": bool(claim_allowed),
            }
        )
        rows.append(group)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(query_rows)


def top1_transition_table(
    scored: pd.DataFrame,
    *,
    truth_col: str,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate unique Top-1 with ties counted against the truth."""

    if truth_col not in scored:
        raise ValueError(f"missing truth column {truth_col!r}")
    rows = []
    for query_id, group in scored.groupby("query_id", sort=False):
        truth_values = group[truth_col].dropna().astype(str).unique()
        if len(truth_values) != 1:
            raise ValueError(f"query {query_id} must have exactly one truth identity")
        truth = truth_values[0]

        def unique_top(score_col: str) -> tuple[str, bool]:
            maximum = float(group[score_col].max())
            top = group[np.isclose(group[score_col].astype(float), maximum, rtol=0, atol=1e-12)]
            candidate = str(top.sort_values("candidate_id").iloc[0].candidate_id)
            return candidate, bool(len(top) == 1 and candidate == truth)

        baseline_top, baseline_correct = unique_top("spectral_score")
        final_top, final_correct = unique_top("final_score")
        rows.append(
            {
                "query_id": str(query_id),
                "truth_candidate_id": truth,
                "baseline_top_candidate": baseline_top,
                "final_top_candidate": final_top,
                "baseline_correct": baseline_correct,
                "final_correct": final_correct,
                "corrected": (not baseline_correct) and final_correct,
                "introduced": baseline_correct and (not final_correct),
            }
        )
    per_query = pd.DataFrame(rows)
    n = len(per_query)
    summary = {
        "n_queries": n,
        "baseline_recall1": float(per_query["baseline_correct"].mean()) if n else np.nan,
        "bioaware_recall1": float(per_query["final_correct"].mean()) if n else np.nan,
        "delta_recall1": float(
            (per_query["final_correct"].astype(int) - per_query["baseline_correct"].astype(int)).mean()
        ) if n else np.nan,
        "corrected": int(per_query["corrected"].sum()),
        "introduced": int(per_query["introduced"].sum()),
    }
    return per_query, summary


def degree_preserving_reaction_decoy(
    participants: pd.DataFrame,
    *,
    seed: int,
    swaps_per_edge: int = 10,
) -> pd.DataFrame:
    """Rewire compound--reaction-side incidences while preserving both degrees.

    This is a negative-control graph, not a biological graph.  Each accepted
    double-edge swap preserves compound occurrence degree and reaction-side
    size.  Reaction metadata and side labels stay fixed.
    """

    p = validate_reaction_participants(participants)
    out = p.copy()
    side_node = out["reaction_id"].astype(str) + "|" + out["side"].astype(str)
    compounds = out["compound_id"].astype(str).to_numpy(copy=True)
    nodes = side_node.to_numpy()
    occupied = set(zip(nodes.tolist(), compounds.tolist()))
    rng = np.random.default_rng(seed)
    target_swaps = max(1, int(len(out) * swaps_per_edge))
    accepted = 0
    attempts = 0
    maximum_attempts = target_swaps * 50
    while accepted < target_swaps and attempts < maximum_attempts:
        attempts += 1
        i, j = rng.integers(0, len(out), size=2)
        if i == j or nodes[i] == nodes[j] or compounds[i] == compounds[j]:
            continue
        new_i = (nodes[i], compounds[j])
        new_j = (nodes[j], compounds[i])
        if new_i in occupied or new_j in occupied:
            continue
        occupied.remove((nodes[i], compounds[i]))
        occupied.remove((nodes[j], compounds[j]))
        compounds[i], compounds[j] = compounds[j], compounds[i]
        occupied.add(new_i)
        occupied.add(new_j)
        accepted += 1
    out["compound_id"] = compounds
    if accepted < max(1, target_swaps // 10):
        raise RuntimeError(
            f"degree-preserving decoy accepted too few swaps: {accepted}/{target_swaps}"
        )
    validated = validate_reaction_participants(out)
    validated.attrs["decoy_swaps_accepted"] = accepted
    validated.attrs["decoy_swaps_target"] = target_swaps
    return validated

"""Loss-aware N/P/A4 action-panel selection for direct-v3 training."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateBoundaryScore:
    """Molecule-max score plus the exact rows that define its boundary."""

    rank: int
    margin: float
    positive_row: int
    hard_negative_molecule_index: int
    hard_negative_row: int


def score_candidate_boundary(
    candidate_rows: np.ndarray,
    candidate_ptr: np.ndarray,
    candidate_embeddings: np.ndarray,
    vector: np.ndarray,
) -> CandidateBoundaryScore:
    """Score one view and retain its positive and hardest-negative rows.

    A scalar molecule margin is insufficient for direct fine-tuning: an action
    can change which candidate molecule is the active competitor.  Routers call
    this helper while the complete candidate block is resident so the later
    trainer can preserve that action-conditioned boundary without caching a
    dense action-by-candidate matrix.
    """
    rows = np.asarray(candidate_rows, dtype=np.int64)
    ptr = np.asarray(candidate_ptr, dtype=np.int64)
    embeddings = np.asarray(candidate_embeddings)
    vector = np.asarray(vector)
    if (
        rows.ndim != 1
        or ptr.ndim != 1
        or embeddings.ndim != 2
        or vector.ndim != 1
        or len(rows) != len(embeddings)
        or embeddings.shape[1] != len(vector)
        or len(ptr) < 3
        or int(ptr[0]) != 0
        or int(ptr[-1]) != len(rows)
        or np.any(np.diff(ptr) <= 0)
    ):
        raise ValueError("candidate boundary arrays are malformed")
    row_scores = embeddings @ vector
    if not np.all(np.isfinite(row_scores)):
        raise ValueError("candidate boundary score is non-finite")
    molecule_scores = np.maximum.reduceat(row_scores, ptr[:-1])
    hard_molecule = int(np.argmax(molecule_scores[1:])) + 1
    positive_left, positive_right = map(int, ptr[:2])
    hard_left, hard_right = map(int, ptr[hard_molecule:hard_molecule + 2])
    positive_position = positive_left + int(np.argmax(row_scores[positive_left:positive_right]))
    hard_position = hard_left + int(np.argmax(row_scores[hard_left:hard_right]))
    return CandidateBoundaryScore(
        rank=1 + int(np.sum(molecule_scores[1:] >= molecule_scores[0])),
        margin=float(molecule_scores[0] - molecule_scores[hard_molecule]),
        positive_row=int(rows[positive_position]),
        hard_negative_molecule_index=hard_molecule,
        hard_negative_row=int(rows[hard_position]),
    )


def _mechanism_block(source: str) -> str:
    source = str(source)
    if source in {"P_guided_original", "E10B", "E11", "E12B", "P"}:
        return "P"
    if source in {"N_mature", "V4_gradient_path", "N"}:
        return "N"
    if source in {"A4_exact", "A4"}:
        return "A4"
    raise RuntimeError(f"unregistered routed action source: {source}")


def _hierarchical_round_robin_indices(
    block: pd.DataFrame,
    *,
    maximum: int,
    score_column: str,
    descending: bool,
) -> list[int]:
    """Round-robin N/P/A4, then source/family, then score.

    A flat lexicographic source/family loop can exhaust a per-query cap on
    the many P recipe families before it ever reaches N_mature.  This helper
    makes the selector use the same hierarchy as the downstream loss.
    """
    if maximum < 1:
        raise ValueError("maximum must be positive")
    required = {"source", "family", "action_id", score_column}
    if missing := required - set(block.columns):
        raise KeyError(f"hierarchical diversity block misses {sorted(missing)}")
    leaves: dict[str, dict[tuple[str, str], list[int]]] = {}
    for (source, family), values in block.groupby(
        ["source", "family"], sort=True, dropna=False,
    ):
        mechanism = _mechanism_block(str(source))
        ordered = values.sort_values(
            [score_column, "action_id"],
            ascending=[not descending, True], kind="stable",
        )
        leaves.setdefault(mechanism, {})[(str(source), str(family))] = list(
            map(int, ordered.index)
        )

    mechanism_queues: dict[str, list[int]] = {}
    for mechanism, mechanism_leaves in leaves.items():
        queue: list[int] = []
        source_exposure: Counter[str] = Counter()
        family_exposure: Counter[str] = Counter()
        cursor = {leaf: 0 for leaf in mechanism_leaves}
        while True:
            candidates = []
            for (source, family), values in mechanism_leaves.items():
                position = cursor[(source, family)]
                if position >= len(values):
                    continue
                index = values[position]
                score = float(block.at[index, score_column])
                if not np.isfinite(score):
                    raise RuntimeError("hierarchical selector received a non-finite score")
                candidates.append((
                    source_exposure[source],
                    family_exposure[family],
                    -score if descending else score,
                    source,
                    family,
                    str(block.at[index, "action_id"]),
                    index,
                ))
            if not candidates:
                break
            *_, source, family, _, index = min(candidates)
            queue.append(int(index))
            cursor[(str(source), str(family))] += 1
            source_exposure[str(source)] += 1
            family_exposure[str(family)] += 1
        mechanism_queues[mechanism] = queue

    selected: list[int] = []
    cursor = 0
    # N/P/A4 have equal top-level opportunity.  Exhausted mechanisms are
    # skipped, so their unused capacity is automatically given to the rest.
    for _ in range(len(block) + 1):
        progressed = False
        for mechanism in ("N", "P", "A4"):
            queue = mechanism_queues.get(mechanism, [])
            if cursor < len(queue):
                selected.append(queue[cursor])
                progressed = True
                if len(selected) == min(maximum, len(block)):
                    return selected
        if not progressed:
            break
        cursor += 1
    return selected


def source_family_shuffled_action_bank(
    actions: pd.DataFrame,
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a chemistry-mismatched, exact-recipe-matched action control.

    Within every supervision/source/recipe stratum, each query receives the
    same intervention recipe from a different query.  A stratum containing
    only one query falls back to its paired control spectrum rather than
    leaking the true action or changing dose/action semantics.
    """
    required = {
        "query_index", "action_id", "source", "family", "recipe_id",
        "supervision_kind", "action_tensor_index",
    }
    if missing := required - set(actions.columns):
        raise KeyError(f"training action table misses {sorted(missing)}")
    indices = actions.action_tensor_index.to_numpy(np.int64)
    if (
        action_spectra.shape != control_spectra.shape
        or len(actions) != len(action_spectra)
        or not np.array_equal(indices, np.arange(len(actions), dtype=np.int64))
    ):
        raise RuntimeError("action/control tensor bank is not table-aligned")
    output = np.empty_like(action_spectra)
    rng = np.random.default_rng(int(seed))
    cross_query_rows = 0
    fallback_rows = 0
    strata = 0
    for _, block in actions.groupby(
        ["supervision_kind", "source", "recipe_id"],
        sort=True,
        dropna=False,
    ):
        strata += 1
        if block.family.astype(str).nunique(dropna=False) != 1:
            raise RuntimeError("one exact shuffled recipe maps to multiple families")
        by_query = {
            int(query): values.sort_values("action_id", kind="stable")
            for query, values in block.groupby("query_index", sort=True)
        }
        queries = list(by_query)
        rng.shuffle(queries)
        if len(queries) < 2:
            own = by_query[queries[0]].action_tensor_index.to_numpy(np.int64)
            output[own] = control_spectra[own]
            fallback_rows += len(own)
            continue
        for position, query in enumerate(queries):
            donor_query = queries[(position + 1) % len(queries)]
            target = by_query[query].action_tensor_index.to_numpy(np.int64)
            donor = by_query[donor_query].action_tensor_index.to_numpy(np.int64)
            for offset, target_index in enumerate(target):
                output[target_index] = action_spectra[donor[offset % len(donor)]]
                cross_query_rows += 1
    if cross_query_rows + fallback_rows != len(actions) or not np.isfinite(output).all():
        raise RuntimeError("shuffled action control did not cover the tensor bank")
    return output, {
        "strategy": "supervision_source_exact_recipe_matched_cross_query_cyclic_shuffle",
        "matching_columns": ["supervision_kind", "source", "recipe_id"],
        "dose_and_action_semantics_preserved": True,
        "all_nonfallback_donors_inside_input_panel": True,
        "seed": int(seed),
        "strata": strata,
        "rows": int(len(actions)),
        "cross_query_rows": int(cross_query_rows),
        "cross_query_fraction": float(cross_query_rows / len(actions)) if len(actions) else 0.0,
        "paired_control_fallback_rows": int(fallback_rows),
        "true_action_row_reused_for_same_query": False,
    }


def select_diverse_routed_actions_v3(
    frame: pd.DataFrame,
    *,
    maximum_corrective_per_query: int = 16,
    maximum_harmful_per_query: int = 8,
    maximum_robust_per_query: int = 8,
) -> pd.DataFrame:
    """Select bounded, source/family-diverse panels for all three semantics.

    Uncertain actions remain in the audit ledger but never enter an optimizer
    branch.  Robust actions receive their own action-view objective and are
    not relabelled as corrective examples.
    """
    required = {
        "query_index", "action_id", "source", "family", "route",
        "action_margin", "margin_change", "paired_advantage",
    }
    if missing := required - set(frame.columns):
        raise KeyError(f"routed action table misses {sorted(missing)}")
    limits = (
        maximum_corrective_per_query,
        maximum_harmful_per_query,
        maximum_robust_per_query,
    )
    if min(limits) < 1:
        raise ValueError("per-query action limits must be positive")
    if frame.action_id.astype(str).duplicated().any():
        raise RuntimeError("action IDs must be globally unique before selection")

    output = frame.copy()
    output["conservative_gain"] = np.minimum(
        output.margin_change.to_numpy(float),
        output.paired_advantage.to_numpy(float),
    )
    output["harm_strength"] = np.maximum(
        -output.margin_change.to_numpy(float),
        -output.paired_advantage.to_numpy(float),
    )
    # A robustness action is useful only insofar as its weakest guarantee is
    # strong.  This score does not turn it into a corrective target.
    output["robustness_score"] = np.minimum.reduce([
        output.action_margin.to_numpy(float),
        output.margin_change.to_numpy(float),
        output.paired_advantage.to_numpy(float),
    ])
    output["selected_corrective"] = False
    output["selected_harmful"] = False
    output["selected_robust"] = False
    for _, block in output.groupby("query_index", sort=True):
        definitions = (
            (
                "corrective", "selected_corrective",
                maximum_corrective_per_query, "conservative_gain", True,
            ),
            (
                "harmful", "selected_harmful",
                maximum_harmful_per_query, "harm_strength", True,
            ),
            (
                "robustness_only", "selected_robust",
                maximum_robust_per_query, "robustness_score", True,
            ),
        )
        for route, column, maximum, score, descending in definitions:
            candidates = block.loc[block.route.astype(str).eq(route)]
            if len(candidates):
                indices = _hierarchical_round_robin_indices(
                    candidates,
                    maximum=maximum,
                    score_column=score,
                    descending=descending,
                )
                output.loc[indices, column] = True

    expected = {
        "selected_corrective": "corrective",
        "selected_harmful": "harmful",
        "selected_robust": "robustness_only",
    }
    for column, route in expected.items():
        if (output[column] & ~output.route.astype(str).eq(route)).any():
            raise RuntimeError(f"{column} contains a non-{route} action")
    return output

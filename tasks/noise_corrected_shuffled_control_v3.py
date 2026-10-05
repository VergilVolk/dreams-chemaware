"""Training-only causal shuffled control for noise direct-v3.

This module is deliberately separate from ``noise_corrected_action_routing_v3``.
Formal N/P/A4 routes hash the routing module, while shuffled controls are built
only after those routes have been frozen. Keeping the implementation here lets
an already completed route bundle remain provenance-valid when the training-only
control implementation is repaired.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


SHUFFLED_CONTROL_VERSION = "family_scoped_maximum_row_matching_v3"


SHUFFLE_MATCHING_COLUMNS = (
    "supervision_kind",
    "source",
    "family",
    "recipe_id",
)


def source_family_shuffle_strata_report(
    actions: pd.DataFrame,
) -> dict[str, object]:
    """Audit formal shuffle feasibility without loading or copying spectra."""
    required = {
        "query_index", "action_id", "source", "family", "recipe_id",
        "supervision_kind",
    }
    if missing := required - set(actions.columns):
        raise KeyError(f"training action table misses {sorted(missing)}")
    cell_rows = []
    for _, block in actions.groupby(
        list(SHUFFLE_MATCHING_COLUMNS), sort=True, dropna=False,
    ):
        counts = block.groupby("query_index", sort=True).size().to_numpy(np.int64)
        rows = int(counts.sum())
        largest_query_rows = int(counts.max())
        # This is the exact maximum cardinality of a row-level bipartite
        # matching whose recipient and donor queries must differ.
        cross_rows = min(rows, 2 * (rows - largest_query_rows))
        cell_rows.append({
            "rows": rows,
            "queries": int(len(counts)),
            "cross_rows": int(cross_rows),
            "fallback_rows": int(rows - cross_rows),
        })
    cells = pd.DataFrame(cell_rows)
    cross_query_rows = int(cells.cross_rows.sum())
    fallback_rows = int(cells.fallback_rows.sum())
    legacy_columns = ["supervision_kind", "source", "recipe_id"]
    legacy = actions.groupby(
        legacy_columns, sort=True, dropna=False,
    ).agg(rows=("action_id", "size"), families=("family", "nunique"))
    collisions = legacy.loc[legacy.families.gt(1)]
    return {
        "matching_columns": list(SHUFFLE_MATCHING_COLUMNS),
        "strata": int(len(cells)),
        "rows": int(len(actions)),
        "cross_query_rows": cross_query_rows,
        "cross_query_fraction": (
            float(cross_query_rows / len(actions)) if len(actions) else 0.0
        ),
        "paired_control_fallback_rows": fallback_rows,
        "fully_cross_query_matchable_strata": int(cells.fallback_rows.eq(0).sum()),
        "partially_cross_query_matchable_strata": int(cells.fallback_rows.gt(0).sum()),
        "legacy_key_multi_family_strata": int(len(collisions)),
        "legacy_key_multi_family_rows": int(collisions.rows.sum()),
    }


def _maximum_cross_query_row_matching(
    block: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a maximum one-to-one different-query row matching.

    Rows are grouped contiguously by query with the largest group first, then
    the donor ordering is rotated by that largest multiplicity.  This is a
    full derangement when no query owns more than half the rows.  Otherwise it
    realizes the exact maximum ``2 * (n - max_count)`` cross-query matches;
    only the mathematically unmatched majority-query recipients fall back.
    """
    grouped: list[tuple[int, np.ndarray]] = []
    for query, values in block.groupby("query_index", sort=True):
        indices = values.sort_values("action_id", kind="stable").action_tensor_index.to_numpy(
            np.int64
        )
        indices = indices.copy()
        rng.shuffle(indices)
        grouped.append((int(query), indices))
    grouped.sort(key=lambda item: (-len(item[1]), item[0]))
    recipients = np.concatenate([indices for _, indices in grouped])
    recipient_queries = np.concatenate([
        np.full(len(indices), query, dtype=np.int64) for query, indices in grouped
    ])
    largest = len(grouped[0][1])
    donors = np.concatenate((recipients[largest:], recipients[:largest]))
    donor_query_by_index = {
        int(index): int(query)
        for query, indices in grouped
        for index in indices
    }
    donor_queries = np.asarray(
        [donor_query_by_index[int(index)] for index in donors], dtype=np.int64
    )
    cross = recipient_queries != donor_queries
    expected_cross = min(len(recipients), 2 * (len(recipients) - largest))
    if int(cross.sum()) != int(expected_cross):
        raise RuntimeError("maximum cross-query row matching cardinality drifted")
    return recipients[cross], donors[cross], recipients[~cross]


def source_family_shuffled_action_bank(
    actions: pd.DataFrame,
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a chemistry-mismatched, exact-action-cell-matched control.

    Every target receives an action spectrum from a different query inside the
    same supervision/source/family/recipe cell. ``family`` is an independent
    part of the cell: in particular, an A4 ``recipe_id`` stores token and dose,
    while its family stores peak role and gradient-rank bin. Omitting family
    would therefore either mix distinct A4 semantics or fail on the formal
    ledger. A cell containing only one query falls back to the target's paired
    control spectrum.
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
        or not np.isfinite(action_spectra).all()
        or not np.isfinite(control_spectra).all()
    ):
        raise RuntimeError("action/control tensor bank is not table-aligned")

    strata_report = source_family_shuffle_strata_report(actions)

    output = np.empty_like(action_spectra)
    donor_tensor_index = np.full(len(actions), -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    cross_query_rows = 0
    fallback_rows = 0
    strata = 0
    for key, block in actions.groupby(
        list(SHUFFLE_MATCHING_COLUMNS), sort=True, dropna=False,
    ):
        strata += 1
        observed = {
            tuple(map(str, values))
            for values in block[list(SHUFFLE_MATCHING_COLUMNS)].itertuples(
                index=False, name=None,
            )
        }
        if len(observed) != 1 or tuple(map(str, key)) not in observed:
            raise RuntimeError("shuffled exact action-cell grouping drifted")
        target, donor, fallback = _maximum_cross_query_row_matching(block, rng)
        output[target] = action_spectra[donor]
        donor_tensor_index[target] = donor
        cross_query_rows += len(target)
        if len(fallback):
            equal_fallback = np.asarray([
                np.array_equal(action_spectra[index], control_spectra[index])
                for index in fallback
            ])
            if np.any(equal_fallback):
                raise RuntimeError("paired-control fallback equals its targeted action")
            output[fallback] = control_spectra[fallback]
            fallback_rows += len(fallback)
        if len(fallback) == 0 and (
            set(map(int, target)) != set(map(int, donor))
        ):
            raise RuntimeError("fully matchable stratum did not preserve donor multiset")
    if cross_query_rows + fallback_rows != len(actions) or not np.isfinite(output).all():
        raise RuntimeError("shuffled action control did not cover the tensor bank")
    if (
        cross_query_rows != strata_report["cross_query_rows"]
        or fallback_rows != strata_report["paired_control_fallback_rows"]
        or strata != strata_report["strata"]
    ):
        raise RuntimeError("shuffled tensor assignment differs from strata preflight")
    if (
        int(np.sum(donor_tensor_index >= 0)) != cross_query_rows
        or np.any(donor_tensor_index >= len(actions))
    ):
        raise RuntimeError("shuffled donor index ledger escaped the input action panel")
    cross_donors = donor_tensor_index[donor_tensor_index >= 0]
    if len(np.unique(cross_donors)) != len(cross_donors):
        raise RuntimeError("shuffled action control reused a donor row")
    recipient_queries = actions.query_index.to_numpy(np.int64)
    cross_recipients = np.flatnonzero(donor_tensor_index >= 0)
    if np.any(
        recipient_queries[cross_recipients]
        == recipient_queries[donor_tensor_index[cross_recipients]]
    ):
        raise RuntimeError("shuffled action retained its own query")
    donor_digest = hashlib.sha256()
    donor_digest.update(donor_tensor_index.tobytes())
    targeted_equal_shuffled_rows = int(sum(
        np.array_equal(action_spectra[index], output[index])
        for index in range(len(actions))
    ))
    return output, {
        "strategy": (
            "supervision_source_family_exact_recipe_matched_"
            "cross_query_cyclic_shuffle"
        ),
        "assignment_algorithm": "maximum_cardinality_bijective_row_matching",
        **strata_report,
        "dose_and_action_semantics_preserved": True,
        "family_semantics_preserved": True,
        "seed": int(seed),
        "true_action_row_reused_for_same_query": False,
        "cross_query_donor_rows_unique": True,
        "cross_query_donor_reuse_rows": 0,
        "fallback_target_equal_rows": 0,
        "targeted_equal_shuffled_rows": targeted_equal_shuffled_rows,
        "targeted_equal_shuffled_fraction": (
            float(targeted_equal_shuffled_rows / len(actions)) if len(actions) else 0.0
        ),
        "maximum_cross_query_cardinality_realized": True,
        "all_nonfallback_donors_inside_input_panel": True,
        "donor_tensor_index_sha256": donor_digest.hexdigest(),
    }

"""Tests for current-E8 panel targeting and selector-frontier composition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from noise_corrected_action_panel import (
    lossless_selector_frontier,
    select_initial_e8_error_boundary_queries,
)
from noise_corrected_action_routing_v3 import select_diverse_routed_actions_v3


def _selected_ids(frame: pd.DataFrame) -> set[str]:
    selected = frame.selected_corrective | frame.selected_harmful | frame.selected_robust
    return set(frame.loc[selected, "action_id"].astype(str))


def main() -> None:
    query, report = select_initial_e8_error_boundary_queries(
        np.arange(9),
        np.asarray([2, 1, 1, 3, 1, 1, 1, 1, 1]),
        np.asarray([-.1, .01, .02, -.2, .03, .04, .05, .001, .002]),
        np.asarray(["e0", "a", "a", "e1", "b", "c", "d", "a", "b"]),
        boundary_multiplier=1.0,
        minimum_boundary_correct=3,
        maximum_boundary_correct=3,
    )
    assert set(query) == {0, 3, 5, 7, 8}
    assert report["initial_E8_error_queries"] == 2
    assert report["selected_boundary_correct_queries"] == 3

    rng = np.random.default_rng(20260907)
    rows = []
    sources = (("N_mature", "N"), ("E10B", "P"), ("E11", "P"), ("A4_exact", "A4"))
    routes = ("corrective", "harmful", "robustness_only", "uncertain")
    for query_index in range(7):
        for source, mechanism in sources:
            for action_index in range(19):
                rows.append({
                    "query_index": query_index,
                    "action_id": f"q{query_index}|{source}|{action_index}",
                    "source": source,
                    "mechanism": mechanism,
                    "family": f"f{action_index % 5}",
                    "route": routes[action_index % len(routes)],
                    "action_margin": float(rng.normal()),
                    "margin_change": float(rng.normal()),
                    "paired_advantage": float(rng.normal()),
                })
    actions = pd.DataFrame(rows)
    limits = {
        "maximum_corrective_per_query": 7,
        "maximum_harmful_per_query": 5,
        "maximum_robust_per_query": 4,
    }
    global_selected = select_diverse_routed_actions_v3(actions, **limits)
    frontiers = [
        lossless_selector_frontier(block.copy(), **limits)
        for _, block in actions.groupby("mechanism", sort=True)
    ]
    recomposed = select_diverse_routed_actions_v3(
        pd.concat(frontiers, ignore_index=True, sort=False), **limits,
    )
    assert _selected_ids(global_selected) == _selected_ids(recomposed)
    print("[noise corrected action-panel tests] PASS=5")


if __name__ == "__main__":
    main()

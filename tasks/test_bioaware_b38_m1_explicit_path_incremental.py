#!/usr/bin/env python
"""Unit checks for B38-M1 path-context aggregation."""
from __future__ import annotations

import pandas as pd

from audit_bioaware_b38_m1_explicit_path_incremental import (
    aggregate_path_contexts,
    empirical_upper_p,
)


def main() -> None:
    frame = pd.DataFrame({
        "query_id": ["q", "q", "q", "q"],
        "candidate_id": ["a", "a", "b", "b"],
        "seed_stratum": ["f0", "f1", "f0", "f1"],
        "has_direct_path": [1, 0, 1, 1],
        "direct_seed_count": [2, 0, 1, 1],
        "direct_event_count": [3, 0, 1, 2],
        "direct_rhea_event_count": [1, 0, 0, 2],
        "direct_kegg_event_count": [2, 0, 1, 0],
        "direction_supported_event_count": [1, 0, 0, 1],
        "rewire_support_00": [0, 1, 0, 0],
        "rewire_support_01": [0, 0, 1, 1],
    })
    result = aggregate_path_contexts(frame, 2).set_index("candidate_id")
    assert result.loc["a", "explicit_direct_path_context_fraction"] == 0.5
    assert result.loc["b", "explicit_direct_path_context_fraction"] == 1.0
    assert result.loc["a", "rewire_context_fraction_00"] == 0.5
    assert result.loc["b", "rewire_context_fraction_01"] == 1.0
    assert empirical_upper_p(0.2, [0.1] * 20) == 1 / 21
    print("[test_bioaware_b38_m1_explicit_path_incremental] PASS", flush=True)


if __name__ == "__main__":
    main()

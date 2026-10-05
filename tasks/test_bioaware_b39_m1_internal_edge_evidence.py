#!/usr/bin/env python
"""Fast unit checks for BioAware B39-M1 edge aggregation."""
from __future__ import annotations

from build_bioaware_b39_m1_internal_edge_evidence import (
    aggregate_coabundance,
    aggregate_spectral,
)


def main() -> None:
    rows = []
    for index, value in enumerate((0.2, 0.5)):
        row = {
            "spectral_available": True,
            "spectral_control_tier": index,
            "coabundance_available": True,
        }
        for view in (
            "truncated_direct", "neutral_loss", "modified_cosine", "dual_view",
            "kgmn_support", "matched_fragments",
        ):
            row[f"spectral_real_{view}"] = value
            row[f"spectral_control_{view}"] = 0.1
            row[f"spectral_excess_{view}"] = value - 0.1
        for name in ("signed", "absolute", "positive", "negative", "sign_stability"):
            row[f"coabundance_real_{name}"] = value
            for control in range(3):
                row[f"coabundance_control_{control}_{name}"] = 0.1 + control * 0.01
        rows.append(row)
    spectral = aggregate_spectral(rows)
    assert spectral["matched_edges"] == 2
    assert spectral["control_tier0_fraction"] == 0.5
    assert abs(spectral["truncated_direct_excess_mean"] - 0.25) < 1e-12
    coabundance = aggregate_coabundance(rows)
    assert coabundance["neighbours"] == 2
    assert abs(coabundance["actual_absolute"] - 0.35) < 1e-12
    assert abs(coabundance["random_absolute"] - 0.11) < 1e-12
    print("[test_bioaware_b39_m1_internal_edge_evidence] PASS", flush=True)


if __name__ == "__main__":
    main()

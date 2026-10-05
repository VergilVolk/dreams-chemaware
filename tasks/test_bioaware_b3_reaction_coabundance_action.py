#!/usr/bin/env python
"""Unit checks for BioAware B3 reaction-coabundance action."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b3_reaction_coabundance_action import (  # noqa: E402
    matched_controls,
    stable_correlation,
)


def main() -> None:
    increasing = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    positive = stable_correlation(increasing, increasing * 3)
    negative = stable_correlation(increasing, increasing[::-1])
    assert positive["signed"] > 0.99 and positive["sign_stability"] == 1.0
    assert negative["signed"] < -0.75 and negative["negative"] > 0.75

    profiles = {}
    for index, identity in enumerate(("N", "A", "B", "C", "D")):
        values = increasing + index
        profiles[f"{identity}|negative"] = {
            "identity": identity,
            "polarity": "negative",
            "profile": values,
            "mean": float(np.log1p(values).mean()),
            "spread": float(np.log1p(values).std()),
            "row": index,
        }
    relations = {"X": {"N"}}
    controls = matched_controls(
        "N", "X", "negative", {"N", "A", "B", "C", "D"}, profiles,
        relations, {name: 2 for name in ("N", "A", "B", "C", "D")}, 3,
    )
    assert controls == ["A", "B", "C"]
    print("[test_bioaware_b3_reaction_coabundance_action] PASS", flush=True)


if __name__ == "__main__":
    main()

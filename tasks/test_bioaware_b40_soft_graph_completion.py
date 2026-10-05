#!/usr/bin/env python
"""Fast unit checks for B40 graph projection and diffusion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_bioaware_b40_soft_graph_completion_cache import build_rhea_projection, diffuse


def main() -> None:
    participants = pd.DataFrame([
        {"compound_id": "A", "reaction_id": "R1", "side": "left", "is_currency": False},
        {"compound_id": "B", "reaction_id": "R1", "side": "right", "is_currency": False},
        {"compound_id": "B", "reaction_id": "R2", "side": "left", "is_currency": False},
        {"compound_id": "C", "reaction_id": "R2", "side": "right", "is_currency": False},
        {"compound_id": "W", "reaction_id": "R2", "side": "right", "is_currency": True},
    ])
    nodes, transition, report = build_rhea_projection(participants)
    assert nodes == ["A", "B", "C"]
    assert report["undirected_projection_edges"] == 2
    score, hops = diffuse([[0]], transition, hops=4, decay=0.65)
    assert hops[0][0, 1] > 0
    assert hops[1][0, 2] > 0
    assert score[0, 1] > score[0, 2] > 0
    assert np.isfinite(score).all()
    print("[test_bioaware_b40_soft_graph_completion] PASS", flush=True)


if __name__ == "__main__":
    main()

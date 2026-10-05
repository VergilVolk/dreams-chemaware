#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b14_reaction_context_action import (  # noqa: E402
    FEATURE_RECIPES,
    add_reaction_features,
)


def main() -> None:
    frame = pd.DataFrame({
        "known_path_fraction": [0.0, 0.6],
        "known_inverse_depth_mean": [0.0, 0.5],
        "known_log_seed_support_mean": [0.0, 1.2],
        "known_log_degree": [0.0, 2.0],
        "edge0_complete_fraction": [0.0, 0.5],
        "edge0_bottleneck_mean": [0.0, 0.8],
    })
    # Features inherited from B12 are only checked for finiteness here.
    for name in sorted(set().union(*FEATURE_RECIPES.values())):
        if name not in frame:
            frame[name] = 0.0
    enriched = add_reaction_features(frame)
    assert enriched["known_path_present"].tolist() == [0.0, 1.0]
    assert enriched["edge0_present"].tolist() == [0.0, 1.0]
    assert abs(float(enriched.loc[1, "known_path_per_degree"]) - 0.2) < 1e-12
    assert abs(float(enriched.loc[1, "known_seed_per_degree"]) - 0.4) < 1e-12
    assert abs(float(enriched.loc[1, "edge0_reliability"]) - 0.4) < 1e-12
    assert set(FEATURE_RECIPES) == {
        "linear_b4_replay",
        "spectral_catalog_interactions",
        "ambiguity_density_interactions",
        "reaction_availability",
        "reaction_strength",
    }
    print("[BioAware B14 reaction-context unit checks] PASS")


if __name__ == "__main__":
    main()

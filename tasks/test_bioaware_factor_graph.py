#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotation.bioaware_factor_graph import (  # noqa: E402
    UNKNOWN, CandidateVariable, PairFactor, TypedFactorGraph,
    identity_family_compatibility, relation_compatibility,
)


def variable(node: str, candidates: tuple[str, ...], scores: tuple[float, ...]) -> CandidateVariable:
    return CandidateVariable(node, candidates + (UNKNOWN,), np.asarray(scores + (-0.25,)))


def main() -> None:
    query = variable("query", ("truth", "wrong"), (-0.10, 0.0))
    seed = variable("seed", ("anchor",), (0.0,))
    pairs = {tuple(sorted(("truth", "anchor")))}
    factor = PairFactor("r1", "query", "seed", relation_compatibility(query, seed, pairs, reward=1.0), "reaction", 1.0)
    result = TypedFactorGraph([query, seed], [factor], damping=0.2).infer()
    assert result["decisions"]["query"]["candidate_id"] == "truth"

    # A disconnected variable exactly preserves its unary winner.
    alone = variable("alone", ("a", "b"), (0.1, 0.0))
    assert TypedFactorGraph([alone], []).infer()["decisions"]["alone"]["candidate_id"] == "a"

    # Ion-family conflict can make unknown safer than an incompatible identity.
    left = variable("left", ("same", "other"), (0.0, -0.1))
    right = CandidateVariable("right", ("different", UNKNOWN), np.asarray((0.0, -10.0)))
    compatibility = identity_family_compatibility(left, right, reward=1.0, conflict=2.0)
    family = PairFactor("family", "left", "right", compatibility, "ion_family", 1.0)
    family_result = TypedFactorGraph([left, right], [family], damping=0.0).infer()
    assert family_result["decisions"]["left"]["candidate_id"] == UNKNOWN

    # Repeated identical factors are bounded by degree normalization and message cap.
    hub = variable("hub", ("x", "y"), (-0.05, 0.0))
    variables = [hub]
    factors = []
    for index in range(20):
        anchor = variable(f"a{index}", (f"s{index}",), (0.0,))
        variables.append(anchor)
        supported = {tuple(sorted(("x", f"s{index}")))}
        factors.append(PairFactor(
            f"e{index}", "hub", f"a{index}", relation_compatibility(hub, anchor, supported, reward=1.0),
            "reaction", 1.0,
        ))
    hub_result = TypedFactorGraph(variables, factors, damping=0.3, message_cap=0.5).infer()
    belief = np.asarray(hub_result["decisions"]["hub"]["belief"])
    assert np.ptp(belief) <= 20 * 0.5 + 1.0
    print("[test_bioaware_factor_graph] PASS")


if __name__ == "__main__":
    main()

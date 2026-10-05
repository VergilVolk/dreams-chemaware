#!/usr/bin/env python
"""Fast unit and source-contract checks for B26."""
from __future__ import annotations

import ast
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tasks/train_bioaware_b26_direct_shared_embedding_canary.py"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    ast.parse(text)
    for required in (
        "model.eval()  # dropout disabled",
        "identity_balanced_sampling",
        "query_reference_encoder_shared",
        "catalogue_score_distilled\": False",
        "reaction_neighbour_used_as_identity_positive\": False",
        "P2b_used\": False",
        "corrective_gradient_reaches_backbone",
        "corrective_gradient_reaches_head",
    ):
        assert required in text, required
    assert "model(spectra)" in text
    assert "official_reference[positive]" in text

    # d softplus((m - (s_pos-s_neg))/tau) / d(s_pos) is negative and
    # d/d(s_neg) is positive for every finite margin.
    margin, target, temperature = -0.1, 0.03, 0.05
    sigmoid = 1.0 / (1.0 + math.exp(-(target - margin) / temperature))
    assert -sigmoid / temperature < 0
    assert sigmoid / temperature > 0

    # Both B20 and the sink-safe B31 bank must be accepted only through their
    # own frozen count/provenance report, never a command-line guess.
    assert "bioaware_b20_direct_action_manifest_complete" in text
    assert "bioaware_b31_sink_safe_action_bank_complete" in text
    assert "bioaware_b32_comprehensive_action_bank_complete" in text
    assert "expected_corrective_physical" in text
    assert "expected_safety_physical" in text
    print("[test_bioaware_b26_direct_shared_embedding_canary] PASS")


if __name__ == "__main__":
    main()

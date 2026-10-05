#!/usr/bin/env python
"""Static checks for the B20 direct-action builder."""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    source = (Path(__file__).resolve().parent / "build_bioaware_b20_direct_action_manifest.py").read_text(
        encoding="utf-8"
    )
    assert "catalogue_score_distilled\": False" in source
    assert "reaction_neighbour_used_as_identity_positive\": False" in source
    assert "physical_duplicate_weight" in source
    assert "identity_equal_weight" in source
    assert "#SBATCH" not in source
    print("[test_bioaware_b20_direct_action_manifest] PASS")


if __name__ == "__main__":
    main()

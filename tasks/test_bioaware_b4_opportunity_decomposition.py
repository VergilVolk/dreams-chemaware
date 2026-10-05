#!/usr/bin/env python
"""Static nested-recipe checks for B4."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b4_opportunity_decomposition import RECIPES  # noqa: E402


def main() -> None:
    assert set(RECIPES["combined_opportunity"]) == (
        set(RECIPES["reference_only"]) | set(RECIPES["graph_only"])
    )
    assert set(RECIPES["reference_only"]) - {"spectral_score"} == {
        "log_reference_spectra"
    }
    assert "log_reference_spectra" not in RECIPES["graph_only"]
    print("[test_bioaware_b4_opportunity_decomposition] PASS", flush=True)


if __name__ == "__main__":
    main()

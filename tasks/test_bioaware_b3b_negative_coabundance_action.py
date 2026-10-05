#!/usr/bin/env python
"""Static contract check for the one-coordinate B3b repair."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b3b_negative_coabundance_action import (  # noqa: E402
    ACTION_FEATURES,
    COMPARATOR_FEATURES,
)


def main() -> None:
    assert len(ACTION_FEATURES) == len(COMPARATOR_FEATURES) + 1
    assert set(ACTION_FEATURES) - set(COMPARATOR_FEATURES) == {
        "coabundance_abs_excess_top3_mean"
    }
    print("[test_bioaware_b3b_negative_coabundance_action] PASS", flush=True)


if __name__ == "__main__":
    main()

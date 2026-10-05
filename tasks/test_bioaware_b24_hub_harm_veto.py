#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]
from audit_bioaware_b24_hub_harm_veto import apply_veto  # noqa: E402


def main() -> None:
    base = pd.DataFrame([{
        "query_id": "q", "source": "d", "truth_candidate_id": "truth",
        "truth_formula": "F", "baseline_candidate_id": "truth",
        "baseline_correct": True, "final_candidate_id": "hub",
        "intervene": True, "final_correct": False, "corrected": False,
        "introduced": True, "delta": -1,
    }])
    candidates = pd.DataFrame([
        {"query_id": "q", "candidate_id": "truth", "known_log_degree": 0.0},
        {"query_id": "q", "candidate_id": "hub", "known_log_degree": 4.1},
    ])
    result = apply_veto(base, candidates, True).iloc[0]
    assert bool(result["hub_veto"])
    assert result["final_candidate_id"] == "truth"
    assert not bool(result["introduced"])
    assert bool(result["final_correct"])
    print("[test_bioaware_b24_hub_harm_veto] PASS")


if __name__ == "__main__":
    main()

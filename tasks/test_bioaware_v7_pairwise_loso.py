from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.develop_bioaware_v7_pairwise_loso import apply_gate, strict_top  # noqa: E402


def main() -> None:
    tied = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "spectral_score": [0.8, 0.8],
        "pairwise_score": [0.2, 0.1],
    })
    selected, unique, margin = strict_top(tied, "spectral_score")
    assert selected == "A" and not unique and margin == 0.0
    prediction = pd.DataFrame([{
        "query_id": "q", "truth_candidate_id": "A", "truth_formula": "F",
        "baseline_candidate_id": "A", "proposed_candidate_id": "B",
        "baseline_correct": False, "proposed_correct": False,
        "proposed_unique": True, "spectral_margin": 0.0,
        "pairwise_margin": 1.0, "support_count": 3, "changes_top1": True,
    }])
    # A no-intervention fallback must remain wrong despite displaying truth A.
    result = apply_gate(prediction, (0.0, float("inf"), 7))
    assert not bool(result.final_correct.iloc[0])
    assert not bool(result.corrected.iloc[0])
    print("[test_bioaware_v7_pairwise_loso] PASS")


if __name__ == "__main__":
    main()

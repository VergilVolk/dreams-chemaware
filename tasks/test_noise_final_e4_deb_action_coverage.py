"""Small deterministic tests for E4-DEB coverage accounting."""
from __future__ import annotations

import pandas as pd

from audit_noise_final_e4_deb_action_coverage import source_summary


def main() -> None:
    frame = pd.DataFrame({
        "query_index": [0, 0, 1, 2],
        "identity": ["a", "a", "b", "c"],
        "formula": ["f1", "f1", "f2", "f3"],
        "clean_rank": [2, 2, 1, 3],
        "paired_advantage": [0.02, -0.02, 0.03, 0.0],
        "corrective": [True, False, False, False],
        "introduced": [False, False, True, False],
    })
    result = source_summary(frame, 0.01)
    expected = {
        "actions": 4,
        "queries": 3,
        "action_covered_error_queries": 2,
        "strict_positive_actions": 1,
        "strict_positive_error_queries": 1,
        "corrected_error_queries": 1,
        "harmful_actions": 2,
        "harmful_queries": 2,
        "introduced_actions": 1,
    }
    observed = {key: result[key] for key in expected}
    if observed != expected:
        raise AssertionError({"expected": expected, "observed": observed})
    print("[test_noise_final_e4_deb_action_coverage] PASS")


if __name__ == "__main__":
    main()

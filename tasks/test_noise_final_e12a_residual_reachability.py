"""Unit tests for E12-A self-match exclusion."""
from __future__ import annotations

import numpy as np

from audit_noise_final_e12a_residual_reachability import PREVALENCE, POLICIES


def main() -> None:
    if POLICIES != ("top3", "farthest3", "maxmin6", "condition6", "maxmin12"):
        raise AssertionError("E12-A reference policies drifted")
    if PREVALENCE != (0.67, 0.50, 0.34) or not np.all(np.diff(PREVALENCE) < 0):
        raise AssertionError("E12-A prevalence ladder drifted")
    print("[test_noise_final_e12a_residual_reachability] PASS", flush=True)


if __name__ == "__main__":
    main()

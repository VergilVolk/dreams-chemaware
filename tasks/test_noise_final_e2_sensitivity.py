"""Unit tests for E2 sensitivity matching and joint max-T correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit_noise_final_e2_sensitivity import exact_control_match, max_t_adjusted_pvalues


def main() -> None:
    exact = pd.Series({
        "selector": "candidate_gradient", "control_count": 3,
        "control_match_levels": "role_intensity_mz,role_intensity_mz,role_intensity_mz",
    })
    fallback = exact.copy()
    fallback["control_match_levels"] = "role_intensity_mz,intensity_mz_role_fallback,role_intensity_mz"
    ordinary = pd.Series({
        "selector": "empirical_conditional_missingness", "control_count": 3,
        "control_match_levels": "intensity_mz,intensity_mz,intensity_mz",
    })
    assert exact_control_match(exact)
    assert not exact_control_match(fallback)
    assert exact_control_match(ordinary)

    rows = []
    for formula in range(30):
        for cell, shift in (("A", 1.0), ("B", 0.0)):
            rows.append({
                "query_formula": f"F{formula}", "cell_id": cell,
                "specific_margin_excess": shift + 0.01 * ((formula % 3) - 1),
            })
    result = max_t_adjusted_pvalues(pd.DataFrame(rows), ["A", "B"], 1000, 7)
    assert result["A"]["max_t_adjusted_p"] < 0.05
    assert result["B"]["max_t_adjusted_p"] >= result["A"]["max_t_adjusted_p"]
    print("[test_noise_final_e2_sensitivity] PASS")


if __name__ == "__main__":
    main()

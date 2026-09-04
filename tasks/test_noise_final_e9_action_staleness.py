"""Static/unit tests for E9 helpers without loading DreaMS."""
from __future__ import annotations
import numpy as np
from audit_noise_final_e9_action_staleness import action_cells, jaccard, parse_path


def main() -> None:
    cells = action_cells()
    if len(cells) != 9 or {cell[0] for cell in cells} != {"candidate_gradient", "role_confounder"}:
        raise AssertionError("E9 mature curriculum cell contract drifted")
    if parse_path("1,3,7") != (1, 3, 7):
        raise AssertionError("path parser drifted")
    for invalid in ("0", "2,2"):
        try:
            parse_path(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid path accepted: {invalid}")
    if not np.isclose(jaccard((1, 2), (2, 3)), 1 / 3):
        raise AssertionError("Jaccard implementation drifted")
    print("[test_noise_final_e9_action_staleness] PASS", flush=True)


if __name__ == "__main__":
    main()

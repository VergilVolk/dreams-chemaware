"""Pure classification contracts for the direct-action bank."""
from __future__ import annotations

import numpy as np

from chemaware_direct_action_core import ROLE_CODE, classify_action_roles


def main() -> None:
    old = np.asarray([2, 1, 1, 2, 1])
    new = np.asarray([1, 1, 1, 2, 2])
    delta = np.asarray([0.02, 0.01, 0.0, -0.002, -0.02])
    swap = np.asarray([0.01, 0.01, -0.01, -0.01, 0.01])
    perm = np.asarray([0.01, 0.01, -0.01, -0.01, 0.01])
    role = classify_action_roles(old, new, delta, swap, perm)
    assert role.tolist() == [ROLE_CODE["corrective_rank"], ROLE_CODE["corrective_margin"],
                             ROLE_CODE["robustness"], ROLE_CODE["robustness"], ROLE_CODE["harmful"]]
    print("direct action bank role contracts passed")


if __name__ == "__main__":
    main()

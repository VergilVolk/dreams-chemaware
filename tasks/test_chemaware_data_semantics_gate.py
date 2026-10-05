"""Small row-set invariants for the ChemAware data semantics gate."""
from __future__ import annotations

import numpy as np

from audit_chemaware_data_semantics_gate import exact_set


def main() -> None:
    exact_set("order-independent", np.asarray([3, 1, 2]), np.asarray([1, 2, 3]))
    for observed in (np.asarray([1, 2]), np.asarray([1, 2, 3, 4])):
        try:
            exact_set("must fail", observed, np.asarray([1, 2, 3]))
        except RuntimeError:
            pass
        else:
            raise AssertionError("row-set loss or contamination was accepted")
    print("data semantics exact-row-set contracts passed")


if __name__ == "__main__":
    main()

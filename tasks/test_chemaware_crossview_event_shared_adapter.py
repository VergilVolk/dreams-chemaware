from __future__ import annotations

import numpy as np

from audit_chemaware_crossview_event_shared_adapter import proposed_local_candidate


def main() -> None:
    body = {
        "query_ptr": np.asarray([0, 2, 4]),
        "molecule_ik14": np.asarray(["A", "B", "C", "D"]),
    }
    output = proposed_local_candidate(
        np.asarray([1, -1]),
        np.asarray([["A", "B"], ["C", "D"]]),
        np.asarray([0, 1]), body,
    )
    assert output.tolist() == [1, -1]
    print("PASS: ChemAware cross-view event transfer contracts")


if __name__ == "__main__":
    main()

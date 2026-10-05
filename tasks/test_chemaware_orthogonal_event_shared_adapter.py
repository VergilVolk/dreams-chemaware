from __future__ import annotations

import numpy as np

from audit_chemaware_orthogonal_event_shared_adapter import cache_events


def main() -> None:
    cache = {
        "baseline_rank": np.asarray([2, 1, 3]),
        "proposal_rank": np.asarray([[1, 2], [2, 1], [2, 3]]),
        "baseline_candidate": np.asarray([1, 0, 2]),
        "proposed_candidate": np.asarray([[0, 2], [1, 2], [1, 2]]),
        "correct_selected_candidate_slot": np.asarray([0, 0, 0]),
        "correct_candidate_utility": np.asarray([[3.0, 1.0], [2.5, 1.0], [2.2, 1.0]]),
        "selected_threshold": np.asarray(2.0),
    }
    events = cache_events(
        cache, "correct", np.asarray([0, 0, 0]), np.asarray([1, 0, 2]),
    )
    assert events["role"].tolist() == [1, 2, 0]
    assert np.allclose(events["confidence"], [1.0, 0.5, 0.2])
    print("PASS: ChemAware orthogonal event shared-adapter contracts")


if __name__ == "__main__":
    main()

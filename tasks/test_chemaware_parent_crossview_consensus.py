from __future__ import annotations

import numpy as np

from audit_chemaware_parent_crossview_consensus import promote_candidate_rank


def main() -> None:
    scores = np.asarray([0.9, 0.8, 0.7])
    labels = np.asarray([False, True, False])
    assert promote_candidate_rank(scores, labels, None) == 2
    assert promote_candidate_rank(scores, labels, 1) == 1
    assert promote_candidate_rank(scores, labels, 2) == 3
    print("PASS: ChemAware parent-plus-cross-view promotion contracts")


if __name__ == "__main__":
    main()

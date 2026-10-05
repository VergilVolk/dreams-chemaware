from __future__ import annotations

import numpy as np

from audit_chemaware_crossview_rankmax_rule_metric import (
    ranks_from_scores,
    selected_identity,
)


def main() -> None:
    slots = np.asarray([0, -1, 1])
    candidates = np.asarray([["A", "B"], ["C", "D"], ["E", "F"]])
    assert selected_identity(slots, candidates).tolist() == ["A", "", "F"]
    scores = np.asarray([[.5, .4, -1e4], [.1, .2, .3]], dtype=np.float32)
    valid = np.asarray([[1, 1, 0], [1, 1, 1]], dtype=bool)
    positive = np.asarray([0, 1])
    assert ranks_from_scores(scores, valid, positive).tolist() == [1, 2]
    print("PASS: ChemAware cross-view Rankmax rule-metric contracts")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Dependency-light unit checks for BioAware B35 ranking semantics."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from evaluate_bioaware_b35_real_library_retrieval import (  # noqa: E402
    mcnemar_exact,
    pooled_binary_auroc,
    strict_ranks,
)


def main() -> None:
    ids = np.asarray(["wrong_a", "truth", "wrong_b"])
    scores = np.asarray([0.9, 0.8, 0.7])
    rank, auc, credit, canonical = strict_ranks(ids, scores, "truth")
    assert rank == 2 and math.isclose(auc, 0.5) and math.isclose(credit, 1.0)
    promoted_rank, promoted_auc, promoted_credit, promoted = strict_ranks(
        ids, scores, "truth", promoted="truth"
    )
    assert promoted_rank == 1
    assert math.isclose(promoted_auc, 1.0) and math.isclose(promoted_credit, 2.0)
    assert promoted[1] > promoted[0] and promoted[0] == canonical[0]

    harmed_rank, harmed_auc, _, _ = strict_ranks(
        np.asarray(["truth", "wrong"]), np.asarray([0.9, 0.8]),
        "truth", promoted="wrong",
    )
    assert harmed_rank == 2 and math.isclose(harmed_auc, 0.0)

    tie_rank, tie_auc, _, _ = strict_ranks(
        np.asarray(["truth", "wrong"]), np.asarray([0.5, 0.5]), "truth"
    )
    assert tie_rank == 2 and math.isclose(tie_auc, 0.5)

    assert math.isclose(
        pooled_binary_auroc(np.asarray([0.9, 0.8, 0.2, 0.1]), np.asarray([1, 0, 1, 0])),
        0.75,
    )
    assert math.isclose(mcnemar_exact(10, 0), 2 / 1024)
    assert math.isclose(mcnemar_exact(0, 0), 1.0)
    print("[BioAware B35 unit checks] PASS", flush=True)


if __name__ == "__main__":
    main()

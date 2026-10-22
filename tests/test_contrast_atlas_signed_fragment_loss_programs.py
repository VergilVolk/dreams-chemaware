from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse


TASKS = Path(__file__).resolve().parents[1] / "tasks"
if str(TASKS) not in sys.path:
    sys.path.insert(0, str(TASKS))

from run_contrast_atlas_signed_fragment_loss_programs import (  # noqa: E402
    loss_spectrum,
    null_top1_summary,
    program_scores,
    signed_program,
)


def test_neutral_loss_uses_precursor_minus_fragment() -> None:
    indices, values = loss_spectrum(200.0, [(150.0, 9.0), (100.0, 16.0)], 1.0)
    assert indices == [50, 100]
    assert np.isclose(np.linalg.norm(values), 1.0)


def test_signed_program_separates_similar_and_opposite_response() -> None:
    matrix = sparse.eye(4, format="csr", dtype=np.float32)
    left = signed_program(matrix, np.asarray([2.0, 1.0, -1.0, -2.0]))
    similar = signed_program(matrix, np.asarray([3.0, 1.0, -1.0, -3.0]))
    opposite = signed_program(matrix, np.asarray([-3.0, -1.0, 1.0, 3.0]))
    same_score = program_scores(left, similar)
    reverse_score = program_scores(left, opposite)
    assert same_score["similar_score"] > same_score["opposite_score"]
    assert reverse_score["opposite_score"] > reverse_score["similar_score"]


def test_null_top1_summary_rejects_composition_preserved_hit() -> None:
    null = pd.DataFrame({
        "null_replicate": np.repeat(np.arange(31), 6),
        "alignment_strength_absolute_rank": np.tile([1, 1, 1, 4, 8, 12], 31),
    })
    result = null_top1_summary(3, null)
    assert result["null_top1_min"] == 3
    assert result["null_top1_max"] == 3
    assert result["plus_one_fraction_null_at_least_observed"] == 1.0
    assert result["specificity_gate"] == "FAIL_RESPONSE_REASSIGNMENT_SPECIFICITY"


def test_signed_program_explicitly_abstains_for_empty_known_world() -> None:
    program = signed_program(sparse.csr_matrix((0, 12)), np.asarray([]))
    assert program["n_entities"] == 0
    assert program["effective_n"] == 0.0
    assert program["up"].shape == (1, 12)
    assert program["composition"].nnz == 0

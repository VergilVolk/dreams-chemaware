#!/usr/bin/env python
"""Small deterministic checks for B6 candidate-level supervision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from annotation.bioaware_supervision import normalise_within_query_logits


def main() -> None:
    trainer_source = (
        ROOT / "tasks/train_bioaware_b4_direct_shared_embedding.py"
    ).read_text(encoding="utf-8")
    for token in (
        'choices=("direct_onehot", "bioaware_soft")',
        'args.supervision_mode == "bioaware_soft"',
        "normalise_within_query_logits",
        'candidate_teacher[query_id]',
        'held_formula_candidate_references_excluded_from_training',
        'training_schedule_sha256',
    ):
        if token not in trainer_source:
            raise AssertionError(f"B6 trainer lacks contract token: {token}")
    scores = np.asarray([3.0, 1.0, -2.0])
    target = normalise_within_query_logits(scores)
    if target.shape != (3,) or not np.isclose(float(target.mean()), 0.0):
        raise AssertionError("candidate logits were not centred")
    if not np.isclose(float(target.std()), 1.0):
        raise AssertionError("candidate logits were not standardised")
    if not bool(target[0] > target[1] > target[2]):
        raise AssertionError("candidate teacher ordering was not preserved")
    affine_target = normalise_within_query_logits(7.0 * scores + 19.0)
    if not np.allclose(target, affine_target, atol=1e-12, rtol=1e-12):
        raise AssertionError("within-query target is not invariant to fold logit scale")
    try:
        normalise_within_query_logits(np.asarray([1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("singleton candidate group did not fail closed")
    print("[BioAware B6 candidate supervision checks] PASS")


if __name__ == "__main__":
    main()

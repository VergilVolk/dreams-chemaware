from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_development_disagreement_core import arbitrate_exclusive_fallback  # noqa: E402


def main() -> None:
    baseline = np.asarray([2, 2, 2, 1], dtype=np.int16)
    direct_rank = np.asarray([1, 2, 2, 2], dtype=np.int16)
    direct_slot = np.asarray([0, -1, -1, 0], dtype=np.int16)
    residual_rank = np.asarray([1, 1, 1, 2], dtype=np.int16)
    residual_slot = np.asarray([0, 0, 1, 0], dtype=np.int16)
    controls = np.asarray([
        [-1, -1, -1], [-1, -1, -1], [-1, 0, -1], [-1, -1, -1],
    ], dtype=np.int16)
    rank, slot, source, exclusive = arbitrate_exclusive_fallback(
        baseline, direct_rank, direct_slot, residual_rank, residual_slot, controls,
    )
    assert rank.tolist() == [1, 1, 2, 2]
    assert slot.tolist() == [0, 0, -1, 0]
    assert source.tolist() == [1, 2, 0, 1]
    assert exclusive.tolist() == [True, True, False, True]

    audit = (ROOT / "tasks/audit_chemaware_counterfactual_exclusive_backoff.py").read_text(
        encoding="utf-8"
    )
    serializer = (ROOT / "tasks/audit_chemaware_orthogonal_rule_residual_policy.py").read_text(
        encoding="utf-8"
    )
    assert "--manifest" not in audit
    assert '"outer_role_4_accessed": False' in audit
    assert '"model_fit": False' in audit
    assert '"threshold_tuned": False' in audit
    assert "candidate_rotated_truthblind" in audit
    assert 'choices=("legacy_audit", "deployment_safe")' in serializer
    assert 'default="legacy_audit"' in serializer
    assert 'name != "candidate_rotated_truthblind"' in serializer
    print("PASS: ChemAware counterfactual-exclusive backoff contracts")


if __name__ == "__main__":
    main()

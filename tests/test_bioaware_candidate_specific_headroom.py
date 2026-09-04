from __future__ import annotations

import pytest

from tasks.summarize_bioaware_candidate_specific_headroom import reaches_requirement


def test_combined_headroom_requirement() -> None:
    assert reaches_requirement(8, 5, 12)
    assert not reaches_requirement(8, 3, 12)
    with pytest.raises(ValueError):
        reaches_requirement(-1, 5, 12)

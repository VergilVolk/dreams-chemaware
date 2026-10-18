"""Regression coverage for visible errors and the candidate-state contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import DEFAULT_SPECTRUM, DreaMSInterface


@pytest.mark.parametrize(
    "text,precursor,charge",
    [
        ("not a spectrum", 500.0, 1),
        (DEFAULT_SPECTRUM, 0.0, 1),
        (DEFAULT_SPECTRUM, 500.0, 2),
        (DEFAULT_SPECTRUM, None, 1),
    ],
)
def test_invalid_input_is_visible_and_structure_display_remains_safe(
    monkeypatch, text, precursor, charge
):
    interface = DreaMSInterface()

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before model loading")

    monkeypatch.setattr(interface, "ensure_model", unexpected_model_load)
    result = interface.process_spectrum(
        text, precursor, charge, "[M+H]+", False, "official_dreams"
    )

    assert len(result) == 10
    assert "输入错误" in result[0]
    assert "输入错误" in result[6]
    assert result[4] == []  # No stale peak table or candidate results.
    assert result[5] == []
    assert result[7:9] == (None, None)
    assert result[-1] == []  # State must remain a list, never an error string.
    assert "暂无可展示" in interface.render_molecules(3, result[-1])


def test_model_failure_is_visible_and_clears_candidates(monkeypatch):
    interface = DreaMSInterface()

    def failed_model_load(*args, **kwargs):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(interface, "ensure_model", failed_model_load)
    result = interface.process_spectrum(
        DEFAULT_SPECTRUM, 500.0, 1, "[M+H]+", False, "official_dreams"
    )

    assert len(result) == 10
    assert "分析失败: RuntimeError" in result[0]
    assert "分析失败: RuntimeError" in result[6]
    assert result[3] is None
    assert result[5] == []
    assert result[7:9] == (None, None)
    assert result[-1] == []
    assert "暂无可展示" in interface.render_molecules(3, result[-1])

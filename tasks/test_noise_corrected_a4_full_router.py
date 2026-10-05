"""Structural tests for the full A4 proposal router."""
from __future__ import annotations

from types import SimpleNamespace

from audit_noise_corrected_a4_full_router import (
    REGISTERED_FORMAL_A4_ROUTE_CONFIGURATION,
    _validate_registered_formal_configuration,
    action_identifier,
    gradient_rank_bin,
)


def main() -> None:
    assert [gradient_rank_bin(value) for value in (1, 2, 4, 7, 13, 26, 51)] == [
        "1", "2-3", "4-6", "7-12", "13-25", "26-50", ">50",
    ]
    assert action_identifier(1, 2, .5) != action_identifier(1, 2, .75)
    formal = {**REGISTERED_FORMAL_A4_ROUTE_CONFIGURATION, "formal": True}
    _validate_registered_formal_configuration(SimpleNamespace(**formal))
    formal["control_repeats"] = 3
    try:
        _validate_registered_formal_configuration(SimpleNamespace(**formal))
    except RuntimeError as error:
        assert "control_repeats" in str(error)
    else:
        raise AssertionError("formal A4 route accepted configuration drift")
    print("[noise corrected A4 full router tests] PASS=3")


if __name__ == "__main__":
    main()

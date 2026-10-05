"""Pure contract tests for corrected full action-bank publication."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from build_noise_corrected_full_action_bank import (
    REGISTERED_FORMAL_ACTION_BANK_CONFIGURATION,
    _validate_registered_formal_configuration,
    publish_rows,
)


def test_publish_rows_uses_only_complete_registered_prefixes() -> None:
    graph = SimpleNamespace(
        query_row=np.asarray([10]), query_ik14=np.asarray(["IK"]),
        query_formula=np.asarray(["F"]), query_has_near=np.asarray([True]),
    )
    roles = [np.asarray([-1, 1, 1, 1])] * 4
    rows = publish_rows(
        graph, 0, "candidate_gradient", 0.5,
        [1, 2, 3, 1], roles, [101, 102, 103, 104],
        [[3, 2, 1], [2, 3, 1]], (3, 4, 5, 6), 2,
    )
    assert len(rows) == 2 and rows[0]["step"] == 3 and rows[1]["step"] == 4
    assert rows[0]["target_path"] == "1,2,3"
    assert rows[0]["matched_control_paths"] == "3,2,1;2,3,1"
    assert rows[0]["hard_negative_row"] == 103
    assert rows[0]["matched_controls_complete"] is True
    assert rows[1]["matched_controls_complete"] is False

    target_only = publish_rows(
        graph, 0, "candidate_gradient", 0.5,
        [1, 2, 3, 1], roles, [101, 102, 103, 104],
        [[], []], (3, 4, 5, 6), 2,
    )
    assert len(target_only) == 2
    assert target_only[0]["matched_control_paths"] == ""
    assert target_only[0]["matched_controls_complete"] is False


def test_action_ids_are_stable_and_cell_specific() -> None:
    from build_noise_corrected_full_action_bank import action_id
    first = action_id(10, "candidate_gradient", 0.5, 3)
    assert first == action_id(10, "candidate_gradient", 0.5, 3)
    assert first != action_id(10, "candidate_gradient", 0.5, 4)


def test_formal_action_bank_configuration_fails_closed() -> None:
    values = dict(REGISTERED_FORMAL_ACTION_BANK_CONFIGURATION)
    _validate_registered_formal_configuration(SimpleNamespace(**values))
    values["softmax_temperature"] = 0.2
    try:
        _validate_registered_formal_configuration(SimpleNamespace(**values))
    except RuntimeError as error:
        assert "softmax_temperature" in str(error)
    else:
        raise AssertionError("formal mature-N action bank accepted configuration drift")


def main() -> None:
    test_publish_rows_uses_only_complete_registered_prefixes()
    test_action_ids_are_stable_and_cell_specific()
    test_formal_action_bank_configuration_fails_closed()
    print("[test_noise_corrected_full_action_bank] PASS tests=3")


if __name__ == "__main__":
    main()

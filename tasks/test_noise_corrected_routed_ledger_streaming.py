"""Numerical tests for bounded-memory routed-spectrum extraction."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from build_noise_corrected_routed_action_ledger import (
    _stream_selected_npz_rows,
    _validate_control_semantics,
    _validate_shared_clean_ranks,
)


def test_source_specific_control_semantics_are_fail_closed() -> None:
    valid = pd.DataFrame([
        {"source": "N_mature", "control_kind": "matched_path",
         "control_semantic": "matched_neutral"},
        {"source": "A4_exact", "control_kind": "clean_fallback",
         "control_semantic": "clean_fallback"},
        {"source": "V4_gradient_path", "control_kind": "matched_path",
         "control_semantic": "matched_neutral"},
        {"source": "E10B", "control_kind": "wrong_identity_direction",
         "control_semantic": "wrong_identity_direction"},
        {"source": "P_guided_original", "control_kind": "wrong_identity_direction",
         "control_semantic": "wrong_identity_direction"},
    ])
    _validate_control_semantics(valid, Path("valid"))
    invalid = valid.copy()
    invalid.loc[3, "control_semantic"] = "matched_neutral"
    try:
        _validate_control_semantics(invalid, Path("invalid"))
    except RuntimeError as error:
        assert "control semantics" in str(error)
    else:
        raise AssertionError("P wrong-identity control was relabelled as matched neutral")


def test_shared_clean_rank_validation_reports_and_rejects_source_drift() -> None:
    valid = pd.DataFrame([
        {"query_index": 1, "source": "N_mature", "clean_rank": 2},
        {"query_index": 1, "source": "V4_gradient_path", "clean_rank": 2},
        {"query_index": 2, "source": "E10B", "clean_rank": 1},
    ])
    report = _validate_shared_clean_ranks(valid)
    assert report == {
        "queries": 2, "multi_source_queries": 1,
        "rank_disagreement_queries": 0, "passed": True,
    }
    invalid = valid.copy()
    invalid.loc[1, "clean_rank"] = 3
    try:
        _validate_shared_clean_ranks(invalid)
    except RuntimeError as error:
        assert '"clean_ranks": [2, 3]' in str(error)
        assert '"sources": ["N_mature", "V4_gradient_path"]' in str(error)
    else:
        raise AssertionError("cross-source current-E8 rank drift was accepted")


def main() -> None:
    test_source_specific_control_semantics_are_fail_closed()
    test_shared_clean_rank_validation_reports_and_rejects_source_drift()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "actions.npz"
        identifiers = np.asarray([f"a{index}" for index in range(11)], dtype=str)
        actions = np.arange(11 * 6, dtype=np.float32).reshape(11, 3, 2)
        controls = -actions
        np.savez_compressed(
            path,
            action_ids=identifiers,
            action_spectra=actions,
            control_spectra=controls,
        )
        take = np.asarray([9, 1, 7], dtype=np.int64)
        observed_ids, id_shape = _stream_selected_npz_rows(
            path,
            "action_ids",
            take,
            expected_ids=pd.Series(identifiers),
            chunk_rows=4,
        )
        observed_actions, action_shape = _stream_selected_npz_rows(
            path, "action_spectra", take, chunk_rows=4,
        )
        observed_controls, control_shape = _stream_selected_npz_rows(
            path, "control_spectra", take, chunk_rows=4,
        )
        assert id_shape == (11,) and action_shape == control_shape == (11, 3, 2)
        assert np.array_equal(observed_ids, identifiers[take])
        assert np.array_equal(observed_actions, actions[take])
        assert np.array_equal(observed_controls, controls[take])

        try:
            _stream_selected_npz_rows(
                path,
                "action_ids",
                take,
                expected_ids=pd.Series(["drift", *identifiers[1:]]),
                chunk_rows=4,
            )
        except RuntimeError as error:
            assert "alignment failed" in str(error)
        else:
            raise AssertionError("full streamed ID alignment drift was not rejected")
    print("[noise corrected routed-ledger streaming tests] PASS=7")


if __name__ == "__main__":
    main()

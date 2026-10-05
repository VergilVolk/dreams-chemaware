"""CPU-only action-supplier and Injector-V1 integration tests for V11."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from noise_historical_best_action_bank_v1 import (
    HISTORICAL_BEST_ACTION_SOURCES,
    HistoricalBestActionBankV1,
    HistoricalBestActionBankV1Config,
    historical_best_action_bank_v1_contract_manifest,
)
from train_noise_corrected_routed_direct import (
    REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION,
    REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION,
    _materialize_admitted_action_bank,
    _select_corrective_admission,
    _validate_registered_formal_v3_configuration,
)


FROZEN_INJECTOR_V1_SHA256 = (
    "21e881f00942a61c2a192a4358806379215d55026f87c934c04f975da1887931"
)


def _row(
    action_id: str,
    query: int,
    source: str,
    margin: float,
    *,
    kind: str = "corrective",
    clean_rank: int = 2,
    action_rank: int = 1,
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "query_index": query,
        "query_row": 1000 + query,
        "query_ik14": f"IK{query:014d}"[-14:],
        "query_formula": f"F{query}",
        "source": source,
        "family": f"family-{source}",
        "recipe_id": f"recipe-{source}",
        "supervision_kind": kind,
        "clean_rank": clean_rank,
        "action_rank": action_rank,
        "action_margin": margin,
        "action_tensor_index": -1,
    }


def _panel() -> pd.DataFrame:
    rows = [
        _row(f"winner-{index}", index, source, 0.10 + index / 100)
        for index, source in enumerate(sorted(HISTORICAL_BEST_ACTION_SOURCES))
    ]
    rows.extend([
        _row("q0-lower", 0, "E12B", 0.07),
        _row("below-floor", 20, "N_mature", 1e-7),
        _row("not-top1", 21, "N_mature", 0.50, action_rank=2),
        _row("already-clean", 22, "N_mature", 0.50, clean_rank=1),
        _row("harmful", 23, "N_mature", -0.10, kind="harmful", action_rank=3),
        _row("robust", 24, "N_mature", 0.0, kind="robust", clean_rank=1),
    ])
    for index, row in enumerate(rows):
        row["action_tensor_index"] = index
    return pd.DataFrame(rows)


def test_historical_best_selects_exactly_one_max_margin_champion_per_query() -> None:
    actions = _panel()
    result = HistoricalBestActionBankV1(HistoricalBestActionBankV1Config(
        margin_floor=5e-6,
        require_all_registered_sources=True,
    )).select(actions)
    assert len(result.actions) == len(HISTORICAL_BEST_ACTION_SOURCES)
    assert result.actions.query_index.is_unique
    assert "q0-lower" not in set(result.actions.action_id)
    assert set(result.actions.source) == HISTORICAL_BEST_ACTION_SOURCES
    assert result.report["strict_top1_rows_before_margin_floor"] == 9
    assert result.report["strict_top1_rows_after_margin_floor"] == 8
    assert result.report["candidate_views_removed_after_champion_selection"] == 1
    assert result.report["outer_held_outcomes_used"] is False
    assert result.report["action_tensors_mutated"] is False


def test_trainer_adapter_matches_standalone_supplier_and_preserves_tensors() -> None:
    actions = _panel()
    standalone = HistoricalBestActionBankV1(HistoricalBestActionBankV1Config(
        margin_floor=5e-6,
    )).select(actions).actions
    selected, report = _select_corrective_admission(
        actions,
        mode="strict_top1",
        margin_floor=5e-6,
        action_bank_contract="historical_best_v1",
    )
    assert selected.action_id.tolist() == standalone.action_id.tolist()
    assert report["one_best_query_compression_used"] is True
    tensor = np.stack([
        np.full((4, 2), index, dtype=np.float32)
        for index in range(len(actions))
    ])
    original = tensor.copy()
    harmful = actions.loc[actions.supervision_kind.eq("harmful")].copy()
    robust = actions.loc[actions.supervision_kind.eq("robust")].copy()
    admitted, corrected, _, _, selected_tensor, _, materialized = (
        _materialize_admitted_action_bank(
            selected, harmful, robust, tensor, -tensor,
        )
    )
    expected_indices = np.concatenate([
        selected.action_tensor_index.to_numpy(np.int64),
        harmful.action_tensor_index.to_numpy(np.int64),
        robust.action_tensor_index.to_numpy(np.int64),
    ])
    assert admitted.action_id.tolist() == (
        selected.action_id.tolist()
        + harmful.action_id.tolist()
        + robust.action_id.tolist()
    )
    assert corrected.action_id.tolist() == selected.action_id.tolist()
    assert np.array_equal(selected_tensor, original[expected_indices])
    assert np.array_equal(tensor, original)
    assert materialized["rejected_corrective_rows_can_be_shuffle_donors"] is False


def test_v11_changes_only_the_upstream_action_supplier_from_v10() -> None:
    exact_diff = {
        key
        for key in (
            set(REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION)
        )
        if REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION.get(key)
    }
    float_diff = {
        key
        for key in (
            set(REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION)
        )
        if REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION.get(key)
    }
    assert exact_diff == {"direct_contract", "action_bank_contract"}
    assert float_diff == set()
    values = {
        **REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION,
        **REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION,
        "development": False,
        "corrective_objective_mode": "v3_direct",
    }
    _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    assert values["optimizer_restoration_scope"] == "safe_exact_corrective"
    assert values["action_bank_contract"] == "historical_best_v1"


def test_injector_v1_is_byte_frozen_and_has_no_action_supplier_dependency() -> None:
    injector = Path(__file__).with_name("noise_action_injector_v1.py")
    assert hashlib.sha256(injector.read_bytes()).hexdigest() == (
        FROZEN_INJECTOR_V1_SHA256
    )
    contract = historical_best_action_bank_v1_contract_manifest()
    assert contract["injector_dependency"] == "none_action_supplier_only"
    assert contract["teacher_embedding_or_distillation_target_used"] is False


def test_unknown_action_source_fails_closed() -> None:
    actions = _panel()
    actions.loc[0, "source"] = "invented_source"
    try:
        HistoricalBestActionBankV1(HistoricalBestActionBankV1Config()).select(actions)
    except RuntimeError as error:
        assert "unknown sources" in str(error)
    else:
        raise AssertionError("historical-best supplier accepted an unknown source")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_historical_best_action_bank_v1] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()

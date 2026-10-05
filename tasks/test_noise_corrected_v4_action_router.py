"""Fail-closed contract tests for the formal best-v4 action router."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from audit_noise_corrected_v4_action_router import (
    REGISTERED_FORMAL_V4_ROUTE_CONFIGURATION,
    _exact_tensor_deduplication_plan,
    _validate_registered_formal_configuration,
)
from noise_corrected_action_routing_v3 import _mechanism_block
from noise_corrected_best_action_v5 import (
    V4_SOURCE,
    action_lineage_contract,
    registered_v4_action_recipes,
)


def test_registered_formal_configuration_is_exact_and_fail_closed() -> None:
    values = dict(REGISTERED_FORMAL_V4_ROUTE_CONFIGURATION)
    _validate_registered_formal_configuration(
        SimpleNamespace(formal=True, **values),
    )
    values["maximum_corrective_frontier"] = 15
    try:
        _validate_registered_formal_configuration(
            SimpleNamespace(formal=True, **values),
        )
    except RuntimeError as error:
        assert "configuration drifted" in str(error)
    else:
        raise AssertionError("formal v4 router accepted a drifted action cap")


def test_exact_spectrum_deduplication_preserves_inverse_alignment() -> None:
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    second = first.clone()
    third = first + 1
    unique, inverse = _exact_tensor_deduplication_plan(
        [first, second, third, first],
    )
    assert len(unique) == 2
    assert inverse.tolist() == [0, 0, 1, 0]
    reconstructed = [unique[int(index)] for index in inverse]
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(reconstructed, [first, second, third, first])
    )


def test_v4_registry_is_direct_and_enters_the_n_mechanism() -> None:
    registry = registered_v4_action_recipes()
    assert len(registry) == 9
    assert len({recipe.name for recipe in registry}) == len(registry)
    assert _mechanism_block(V4_SOURCE) == "N"
    lineage = action_lineage_contract()
    assert lineage["teacher_embedding_or_distillation_target_used"] is False
    assert V4_SOURCE in lineage["mature_executable_sources"]
    assert lineage["E13"].startswith("training_attempt_reusing_E12B")


def test_router_never_recomputes_clean_rank_from_gradient_batch_forward() -> None:
    source = Path(__file__).with_name(
        "audit_noise_corrected_v4_action_router.py"
    ).read_text(encoding="utf-8")
    assert '"baseline_rank": int(ranks_by_query[query])' in source
    assert '"baseline_margin": float(margins_by_query[query])' in source
    assert "current[position].detach().float().cpu().numpy(),\n                )" not in source


def main() -> None:
    tests = [
        test_registered_formal_configuration_is_exact_and_fail_closed,
        test_exact_spectrum_deduplication_preserves_inverse_alignment,
        test_v4_registry_is_direct_and_enters_the_n_mechanism,
        test_router_never_recomputes_clean_rank_from_gradient_batch_forward,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_v4_action_router] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()

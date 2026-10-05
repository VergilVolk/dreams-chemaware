"""Regression tests for the v4 multi-peak action audit."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

import audit_noise_corrected_action_expansion_v4 as audit_module

from audit_noise_corrected_action_expansion_v4 import (
    build_recipe_views,
    build_sequential_supported_boost_views,
    formula_diverse_sample,
    summarize,
)
from noise_v3_core import CONFOUNDER_ONLY, IDENTITY_ONLY


def test_formula_diverse_sample_exhausts_first_round_before_repeats() -> None:
    frame = pd.DataFrame({
        "query_formula": ["A", "A", "B", "B", "C"],
        "query_row": [1, 2, 3, 4, 5],
    })
    selected = formula_diverse_sample(frame, 3, seed=7, panel="x")
    assert selected["query_formula"].nunique() == 3


def test_composite_control_is_same_role_and_disjoint() -> None:
    clean = torch.tensor([
        [500.0, 0.2],
        [50.0, 1.0], [60.0, 0.8], [70.0, 0.9], [80.0, 0.7],
        [90.0, 0.6], [100.0, 0.5], [110.0, 0.55], [120.0, 0.45],
        [0.0, 0.0],
    ])
    gradient = np.asarray([0.0, -1.0, -0.8, 0.0, 0.0, 1.0, 0.8, 0.0, 0.0, 0.0])
    roles = np.asarray([
        -1,
        CONFOUNDER_ONLY, CONFOUNDER_ONLY, CONFOUNDER_ONLY, CONFOUNDER_ONLY,
        IDENTITY_ONLY, IDENTITY_ONLY, IDENTITY_ONLY, IDENTITY_ONLY,
        -1,
    ])
    recipes = build_recipe_views(
        clean, gradient, roles, seed=11, baseline_margin=-0.05,
    )
    names = {str(recipe["recipe"]) for recipe in recipes}
    assert {
        "single_attenuation_1x50", "dual_attenuation_2x50",
        "supported_boost_2x50", "signed_multiplicative_2d2u_50",
        "conservative_exchange_2d2u_50",
        "adaptive_trust_down", "adaptive_trust_up", "adaptive_trust_joint",
        "adaptive_strong_down", "adaptive_strong_up", "adaptive_strong_joint",
    }.issubset(names)
    for recipe in recipes:
        if recipe["control"] is None:
            continue
        target = set(recipe["target_down"]) | set(recipe["target_up"])
        control = set(recipe["control_down"]) | set(recipe["control_up"])
        assert target.isdisjoint(control)
        assert all(roles[token] == roles[other] for token, other in zip(
            recipe["target_down"], recipe["control_down"],
        ))
        assert all(roles[token] == roles[other] for token, other in zip(
            recipe["target_up"], recipe["control_up"],
        ))


def test_summary_uses_current_rank_not_historical_panel_name() -> None:
    frame = pd.DataFrame({
        "recipe": ["r", "r"],
        "panel": ["residual_error", "correct_safety"],
        "baseline_rank": [1, 2],
        "baseline_margin": [0.1, -0.1],
        "target_rank": [2, 1],
        "target_margin": [-0.1, 0.1],
        "control_rank": [1, 2],
        "control_margin": [0.05, -0.05],
        "single_target_rank": [1, 2],
        "query_formula": ["A", "B"],
        "query_index": [1, 2],
        "total_fractional_dose": [0.5, 0.25],
        "target_gain_reached": [True, False],
    })
    report = summarize(frame, "r", 3)
    assert report["target_error_queries"] == 1
    assert report["target_corrected"] == 1
    assert report["target_correct_queries"] == 1
    assert report["target_introduced"] == 1
    assert report["target_risk_net"] == -1
    assert report["formal_route_counts"] == {
        "corrective": 1,
        "robustness_only": 0,
        "harmful": 1,
        "uncertain": 0,
    }
    assert report["strict_route_corrected_queries"] == [2]
    assert report["strict_route_introduced"] == 1
    assert report["adaptive_target_gain_reached_fraction"] == 0.5


def test_sequential_boost_refreshes_and_never_reuses_target_or_control() -> None:
    clean = torch.tensor([
        [500.0, 0.2],
        [50.0, 1.0], [60.0, 0.9], [70.0, 0.8],
        [80.0, 0.7], [90.0, 0.6], [100.0, 0.5],
        [0.0, 0.0],
    ])
    roles = np.asarray([-1] + [IDENTITY_ONLY] * 6 + [-1], dtype=np.int8)
    calls = {"context": 0}

    def fake_forward(model, block, amp):
        total = block[:, 1:, 1].sum(dim=1)
        return torch.stack((total, total * 0), dim=1)

    def fake_context(*args, **kwargs):
        calls["context"] += 1
        return SimpleNamespace(positive_row=1, negative_rows=(2,)), roles.copy()

    original_forward = audit_module.forward_embeddings
    original_context = audit_module.current_context
    audit_module.forward_embeddings = fake_forward
    audit_module.current_context = fake_context
    try:
        output = build_sequential_supported_boost_views(
            pd.DataFrame({"query_index": [0], "query_row": [10]}),
            torch.nn.Identity(), None, {10: clean},
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            {1: 1, 2: 2}, torch.device("cpu"),
            argparse.Namespace(
                sequential_boost_steps=3,
                sequential_boost_dose=0.5,
                gradient_batch_size=1,
                top_k_negatives=1,
                fragment_tolerance=0.02,
                softmax_temperature=0.1,
                n_highest_peaks=100,
                sample_seed=9,
            ),
        )
    finally:
        audit_module.forward_embeddings = original_forward
        audit_module.current_context = original_context
    assert calls["context"] == 3
    assert len(output[0]) == 3
    for step, recipe in enumerate(output[0], start=1):
        target = tuple(recipe["target_up"])
        control = tuple(recipe["control_up"])
        assert len(target) == step == len(set(target))
        assert len(control) == step == len(set(control))
        assert set(target).isdisjoint(control)
        assert recipe["total_fractional_dose"] == 0.5 * step


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        f"[test_noise_corrected_action_expansion_v4_audit] PASS tests={len(tests)}",
        flush=True,
    )

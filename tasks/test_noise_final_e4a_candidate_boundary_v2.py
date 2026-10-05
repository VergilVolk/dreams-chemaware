"""CPU tiny-overfit smoke for raw action views through boundary v2."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from train_noise_final_e4a_direct_augmentation import (
    DirectExample, candidate_boundary_action_loss, coverage_first_query_schedules,
    query_complete_action_batches,
)


class TinyStore:
    def __init__(self) -> None:
        # Token zero is the precursor and is never attenuated.
        self.values = {
            0: torch.tensor([[1.0, 1.0], [0.3, 0.8], [0.7, 0.2]]),
            1: torch.tensor([[1.0, 1.0], [0.2, 0.9], [0.8, 0.1]]),
            2: torch.tensor([[1.0, 1.0], [0.4, 0.6], [0.6, 0.4]]),
            3: torch.tensor([[1.0, 1.0], [0.7, 0.2], [0.3, 0.8]]),
            4: torch.tensor([[1.0, 1.0], [0.8, 0.1], [0.2, 0.9]]),
        }

    def one(self, row: int) -> torch.Tensor:
        return self.values[int(row)]

    def get(self, rows) -> torch.Tensor:
        return torch.stack([self.one(int(row)) for row in rows])


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(6, 3, bias=False)
        torch.manual_seed(9)
        torch.nn.init.normal_(self.linear.weight, std=0.2)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(spectra.reshape(len(spectra), -1)), dim=1)


def test_raw_target_and_control_reach_shared_encoder_with_clean_primary_loss() -> None:
    store = TinyStore()
    model = TinyEncoder()
    example = DirectExample(
        query_index=0, query_row=0, identity="A", formula="F",
        positive_rows=(1, 2), negative_rows=(3, 4),
        official_margin=-0.1, official_rank=2, sample_weight=1.0,
        policy="candidate_gradient|step=1", target_path=(1,), control_path=(2,),
        teacher_advantage=0.05, attenuation=0.5,
    )
    with torch.no_grad():
        anchors = {
            row: model(value.unsqueeze(0)).squeeze(0).numpy()
            for row, value in store.values.items()
        }
    args = SimpleNamespace(
        amp=False, candidate_boundary_version="v2_molecule_max",
        rank_margin=0.05, temperature=0.10,
        boundary_advantage_temperature=0.02, boundary_hard_temperature=0.10,
        boundary_topk_negatives=2, margin_floor_slack=0.005,
        lambda_boundary_clean=1.0, lambda_boundary_target=0.25,
        lambda_boundary_counterfactual=0.25, lambda_boundary_full_clean=0.25,
        lambda_boundary_action_safety=0.25,
        lambda_preserve=0.0,
    )
    loss, metrics = candidate_boundary_action_loss(
        model, store, [example], anchors, torch.device("cpu"), args,
    )
    loss.backward()
    gradient = model.linear.weight.grad
    assert torch.isfinite(loss) and gradient is not None
    assert float(gradient.norm()) > 0.0
    assert metrics["boundary_effective_query_fraction"] == 1.0
    assert metrics["boundary_clean"] > 0.0 and metrics["boundary_full_clean"] > 0.0


def test_query_action_sets_are_not_split_and_zero_weight_is_preserved() -> None:
    base = DirectExample(
        query_index=7, query_row=0, identity="A", formula="F",
        positive_rows=(1,), negative_rows=(3,), official_margin=-0.1,
        official_rank=2, sample_weight=1.0, policy="p1",
        target_path=(1,), control_path=(2,), attenuation=0.5,
    )
    second = DirectExample(
        **{**base.__dict__, "sample_weight": 0.0, "policy": "p2",
           "target_path": (2,), "control_path": (1,)}
    )
    third = DirectExample(
        **{**base.__dict__, "query_index": 8, "identity": "B", "policy": "p3"}
    )
    batches = query_complete_action_batches([base, third, second], maximum_actions=2)
    assert [[item.query_index for item in batch] for batch in batches] == [[7, 7], [8]]
    schedules = coverage_first_query_schedules([base, third, second], epochs=2, seed=3)
    assert sorted(index for schedule in schedules for index in schedule) == [0, 1, 2]
    assert sum(any(index in schedule for index in (0, 2)) for schedule in schedules) == 1

    store = TinyStore()
    model = TinyEncoder()
    with torch.no_grad():
        anchors = {
            row: model(value.unsqueeze(0)).squeeze(0).numpy()
            for row, value in store.values.items()
        }
    args = SimpleNamespace(
        amp=False, candidate_boundary_version="v2_molecule_max",
        rank_margin=0.05, temperature=0.10,
        boundary_advantage_temperature=0.02, boundary_hard_temperature=0.10,
        boundary_topk_negatives=1, margin_floor_slack=0.005,
        lambda_boundary_clean=1.0, lambda_boundary_target=0.25,
        lambda_boundary_counterfactual=0.25, lambda_boundary_full_clean=0.25,
        lambda_boundary_action_safety=0.25, lambda_preserve=0.0,
    )
    loss, metrics = candidate_boundary_action_loss(
        model, store, [base, second], anchors, torch.device("cpu"), args,
    )
    loss.backward()
    assert metrics["boundary_routed_actions"] == 2.0
    assert metrics["boundary_qualified_actions"] == 1.0
    assert metrics["boundary_unqualified_actions"] == 1.0
    assert metrics["boundary_effective_query_fraction"] == 1.0

    mismatched = DirectExample(
        **{**second.__dict__, "positive_rows": (2,), "negative_rows": (4,)}
    )
    try:
        candidate_boundary_action_loss(
            model, store, [base, mismatched], anchors, torch.device("cpu"), args,
        )
    except RuntimeError as error:
        assert "share candidate references" in str(error)
    else:
        raise AssertionError("semantic edge mismatch did not fail closed")


def main() -> None:
    test_raw_target_and_control_reach_shared_encoder_with_clean_primary_loss()
    test_query_action_sets_are_not_split_and_zero_weight_is_preserved()
    print("[test_noise_final_e4a_candidate_boundary_v2] PASS tests=2")


if __name__ == "__main__":
    main()

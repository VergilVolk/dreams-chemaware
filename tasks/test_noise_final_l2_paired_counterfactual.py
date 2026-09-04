"""Fast behavioural tests for the L2 paired counterfactual protocol."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from train_noise_final_l2_paired_counterfactual import (
    PairedAction, action_schedule, arm_paths, cell_id, deterministic_query_subset,
    paired_advantage_loss, select_oof_actions,
)


def fake_action(index: int, query: int) -> PairedAction:
    return PairedAction(
        action_index=index, query_index=query, query_row=query, identity=f"i{query}",
        formula=f"f{query}", selector="candidate_gradient", step=3, attenuation=0.5,
        target_path=(1, 2, 3), control_paths=((4, 5, 6), (7, 8, 9)),
        positive_rows=(10,), negative_rows=(11,), initial_margin=-0.1,
        predicted_gain=0.02 + index / 1000, positive_probability=0.8,
        harmful_probability=0.05,
    )


def main() -> None:
    frame = pd.DataFrame({
        "action_index": [0, 1, 2], "query_index": [0, 0, 1], "query_row": [10, 10, 11],
        "query_ik14": ["i0", "i0", "i1"], "query_formula": ["f0", "f0", "f1"],
        "formula_fold": [0, 0, 1], "selector": ["candidate_gradient", "candidate_gradient", "role_confounder"],
        "attenuation": [0.5, 0.5, 1.0], "step": [3, 4, 1],
        "target_path": ["1,2,3", "1,2,3,4", "1"],
        "matched_control_paths": ["4,5,6;7,8,9", "5,6,7,8;9,10,11,12", "2;3"],
        "clean_pred_gain": [0.02, 0.02, 0.02], "clean_p_positive": [0.8, 0.8, 0.8],
        "clean_p_harmful": [0.05, 0.05, 0.05], "advantage_label": ["positive"] * 3,
        "transition": ["corrected"] * 3,
    })
    assert cell_id(frame).tolist() == [
        "candidate_gradient|0.50000000|3", "candidate_gradient|0.50000000|4",
        "role_confounder|1.00000000|1",
    ]
    train, held = select_oof_actions(frame, 1)
    assert len(train) == 2 and len(held) == 1
    assert len(deterministic_query_subset(train, 1, 7)) == 2
    stratified = pd.concat([
        frame.iloc[[0]].assign(action_index=index, query_index=index, query_row=10 + index,
                               query_ik14=f"ic{index}", query_formula=f"fc{index}")
        for index in range(10)
    ] + [
        frame.iloc[[2]].assign(action_index=100, query_index=100, query_row=110,
                               query_ik14="ir", query_formula="fr", formula_fold=0)
    ], ignore_index=True)
    stratified_subset = deterministic_query_subset(stratified, 3, 11, minimum_per_selector=1)
    assert stratified_subset["query_index"].nunique() == 3
    assert set(stratified_subset["selector"]) == {"candidate_gradient", "role_confounder"}
    actions = [fake_action(index, 0) for index in range(4)] + [fake_action(10, 1)]
    schedule = action_schedule(actions, 2, 2, 13)
    assert all(len(group) <= 2 for epoch in schedule for group in epoch)
    assert {item.action_index for epoch in schedule for group in epoch for item in group} >= {0, 1, 2, 3, 10}
    same_identity = [
        fake_action(20, 20),
        PairedAction(**{**fake_action(21, 21).__dict__, "identity": "i20"}),
    ]
    identity_schedule = action_schedule(same_identity, 2, 1, 13)
    assert all(len(epoch) == 1 for epoch in identity_schedule)
    assert {group[0].query_index for epoch in identity_schedule for group in epoch} == {20, 21}
    target_primary, target_comparator = arm_paths(actions[0], "targeted")
    null_primary, null_comparator = arm_paths(actions[0], "matched_random")
    assert target_primary == actions[0].target_path
    assert null_primary in actions[0].control_paths and null_comparator in actions[0].control_paths
    assert target_comparator == null_primary and null_primary != null_comparator
    primary = torch.tensor([0.10, 0.20], requires_grad=True)
    comparator = torch.tensor([0.12, 0.18], requires_grad=True)
    weights = torch.ones(2)
    advantage = paired_advantage_loss(primary, comparator, 0.01, 0.10, weights)
    primary_gradient, comparator_gradient = torch.autograd.grad(
        advantage, (primary, comparator), allow_unused=True,
    )
    assert torch.all(primary_gradient < 0)
    assert comparator_gradient is None
    print("[test_noise_final_l2_paired_counterfactual] PASS", flush=True)


if __name__ == "__main__":
    main()

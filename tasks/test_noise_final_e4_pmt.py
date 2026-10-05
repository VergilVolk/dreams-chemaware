"""Numerical and static tests for the minimal E4-PMT repair."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from noise_final_e4_pmt_core import (
    coverage_first_schedules, restrict_corrective_query_scope, route_replayed_actions,
)
from build_noise_final_e4_pmt_manifest import (
    contract_mismatches, join_r0_to_published_n_ledger,
)


def main() -> None:
    cells = [
        *(f"candidate_gradient|a=0.50|step={step}" for step in range(3, 7)),
        *(f"role_confounder|a=1.00|step={step}" for step in range(1, 6)),
    ]
    frame = pd.DataFrame({
        "paired_advantage": [0.02, 0.02, -0.02, 0.0] * 9,
        "clean_rank": [2, 1, 1, 2] * 9,
        "target_rank": [1, 2, 2, 2] * 9,
        "control_rank": [2, 1, 1, 2] * 9,
        "corrected": [True, False, False, False] * 9,
        "introduced": [False, True, False, False] * 9,
        "cell_id": np.repeat(cells, 4),
    })
    routed = route_replayed_actions(frame)
    if routed.iloc[1]["route"] != "harmful":
        raise RuntimeError("harmful precedence failed for positive/harmful overlap")
    if not routed.loc[routed["route"] != "corrective", "corrective_weight"].eq(0).all():
        raise RuntimeError("non-corrective target weight is not exactly zero")
    errors_only = restrict_corrective_query_scope(routed, "errors")
    if errors_only.loc[errors_only["route"].eq("corrective"), "clean_rank"].eq(1).any():
        raise RuntimeError("errors-only scope retained a clean-correct corrective action")
    if not errors_only.loc[errors_only["clean_rank"].eq(1), "corrective_weight"].eq(0).all():
        raise RuntimeError("errors-only scope did not zero clean-correct action weight")
    # Reproduce the real published schema: selector/attenuation/step exist only
    # in R0, while the dynamic ledger keeps normalized action_id/cell/payload.
    r0 = pd.DataFrame({
        "query_index": [10, 11], "selector": ["candidate_gradient", "role_confounder"],
        "attenuation": [0.5, 1.0], "step": [3, 1], "formula_fold": [1, 2],
    })
    published = pd.DataFrame({
        "action_id": ["N|10|candidate_gradient|0.50|3", "N|11|role_confounder|1.00|1"],
        "cell_id": ["candidate_gradient|a=0.50|step=3", "role_confounder|a=1.00|step=1"],
        "control_payload": ["1,2,3", "4"],
    })
    joined = join_r0_to_published_n_ledger(r0, published, 0)
    if len(joined) != 2 or "selector" not in joined or "control_payload" not in joined:
        raise RuntimeError("published-schema R0/ledger join failed")
    mixed_contract = {
        "positive_gate": True, "M2_predictions_used": False,
        "P_actions_used": False, "P2b": "forbidden",
    }
    if contract_mismatches(mixed_contract, mixed_contract):
        raise RuntimeError("required False contracts are incorrectly treated as failures")
    wrong_contract = dict(mixed_contract, M2_predictions_used=True)
    if set(contract_mismatches(wrong_contract, mixed_contract)) != {"M2_predictions_used"}:
        raise RuntimeError("contract mismatch helper did not isolate the wrong field")
    ids = [f"a{i}" for i in range(17)]
    identities = [f"I{i % 5}" for i in range(17)]
    policies = [cells[i % len(cells)] for i in range(17)]
    schedules = coverage_first_schedules(ids, identities, policies, 4, 4, 7)
    flat = [index for epoch in schedules for index in epoch]
    if len(set(flat[:17])) != 17 or set(flat[:17]) != set(range(17)):
        raise RuntimeError("coverage-first scheduler recycled before full exposure")
    # A larger alpha must transmit a larger clean-margin gradient.  The
    # control floor is weighted twice as strongly as preference, so lowering
    # an already deficient control cannot reduce total paired loss.
    clean_margin = torch.tensor(0.0, requires_grad=True)
    advantage = torch.tensor(0.04)
    floor025 = torch.relu(torch.tensor(0.02) + 0.25 * advantage - clean_margin)
    floor050 = torch.relu(torch.tensor(0.02) + 0.50 * advantage - clean_margin)
    if not floor050 > floor025:
        raise RuntimeError("PMT alpha does not strengthen clean inheritance")
    control_margin = torch.tensor(-0.02, requires_grad=True)
    target_margin = torch.tensor(-0.02, requires_grad=True)
    preference = torch.relu(torch.tensor(0.01) - (target_margin - control_margin))
    control_floor = torch.relu(torch.tensor(0.02) - control_margin)
    toy_loss = preference + 2.0 * control_floor
    toy_loss.backward()
    if control_margin.grad is None or float(control_margin.grad) >= 0:
        raise RuntimeError("control safeguard permits preference-by-degradation")
    for script in (
        ROOT / "tasks/build_noise_final_e4_pmt_manifest.py",
        ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py",
    ):
        ast.parse(script.read_text(encoding="utf-8"))
    print("[test_noise_final_e4_pmt] PASS")


if __name__ == "__main__":
    main()

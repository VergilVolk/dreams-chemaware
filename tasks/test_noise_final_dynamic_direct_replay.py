"""Fail-fast static contract test for current-geometry replay."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))


def main() -> None:
    replay = (ROOT / "tasks/audit_noise_final_dynamic_direct_replay.py").read_text(encoding="utf-8")
    validator = (ROOT / "tasks/validate_noise_final_dynamic_direct_replay.py").read_text(encoding="utf-8")
    ast.parse(replay); ast.parse(validator)
    required = (
        "optimizer_steps\": 0", "full_candidate_scoring", "materialize_pair",
        "attenuate_sequence", "apply_action", "apply_transfer", "reference_profile",
        "recurrent_missing_peaks", "score_vector", "paired_advantage",
        "corrected", "introduced", "positive_harmful_overlap",
        "ledger_actions", "exact_current_geometry",
    )
    if missing := [token for token in required if token not in replay]:
        raise RuntimeError(f"replay contract drifted: {missing}")
    forbidden = ("optimizer =", ".backward(", ".step()", "P2b_score", "best_fixed_cell")
    if present := [token for token in forbidden if token in replay]:
        raise RuntimeError(f"replay contains forbidden training/selection code: {present}")
    if "training_actions.csv.gz" not in replay or "epoch_schedule.csv.gz" in replay:
        raise RuntimeError("replay must cover the full ledger before schedule sampling")
    try:
        import torch
        from audit_noise_final_dynamic_direct_replay import materialize_pair
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print("[test_noise_final_dynamic_direct_replay] static PASS; torch numeric test deferred")
            return
        raise

    clean = torch.tensor([[200.0, 1.0], [50.0, 1.0], [75.0, 0.5], [100.0, 0.2], [0.0, 0.0]])
    positive_a = torch.tensor([[200.0, 1.0], [50.0, 0.3], [75.0, 0.8], [125.0, 0.7], [0.0, 0.0]])
    positive_b = torch.tensor([[200.0, 1.0], [50.0, 0.4], [75.0, 0.7], [125.0, 0.6], [0.0, 0.0]])
    wrong_a = torch.tensor([[200.0, 1.0], [50.0, 0.9], [100.0, 0.8], [150.0, 0.7], [0.0, 0.0]])
    wrong_b = torch.tensor([[200.0, 1.0], [50.0, 0.8], [100.0, 0.7], [150.0, 0.6], [0.0, 0.0]])

    class Store:
        values = {0: clean, 1: positive_a, 2: positive_b, 3: wrong_a, 4: wrong_b}
        def one(self, row: int):
            return self.values[int(row)]

    args = SimpleNamespace(fragment_tolerance=0.02, recurrence_prevalence=0.67, recurrence_max_peaks=5)
    rows = [
        SimpleNamespace(source="N", query_row=0, query_index=0, target_payload="1", control_payload="2", dose=0.5, family="N:candidate_gradient"),
        SimpleNamespace(source="P_intensity", query_row=0, query_index=0, target_payload="1;2", control_payload="3;4", dose=0.5, family="P:consensus_projection"),
        SimpleNamespace(source="P_transfer", query_row=0, query_index=0, target_payload="1;2", control_payload="3;4", dose=0.5, family="P:recurrent_union_mix"),
    ]
    cache = {}
    for row in rows:
        target, control = materialize_pair(row, Store(), args, cache)
        if target.shape != clean.shape or control.shape != clean.shape:
            raise RuntimeError(f"{row.source} changed spectrum tensor shape")
        if torch.equal(target, control):
            raise RuntimeError(f"{row.source} target/control materialization collapsed")
    print("[test_noise_final_dynamic_direct_replay] PASS")


if __name__ == "__main__":
    main()

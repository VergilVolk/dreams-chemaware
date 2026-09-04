"""Fast regression tests for the E15-M2 executable panel and trainer."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import build_noise_final_e15_m2_executable_panel as panel
import train_noise_final_e15_m2_overfit as trainer


def test_e14_join_is_fail_closed() -> None:
    base = pd.DataFrame({
        "source": ["E14_mature_P", "E14_mature_P"],
        "query_index": [1, 2], "action_id": ["a", "b"],
    })
    exact = pd.DataFrame({
        "query_index": [1], "action_id": ["a"],
        "positive_reference_rows": ["10;11"], "teacher_positive_row": [10],
        "teacher_hard_negative_row": [20], "teacher_pair_clean_margin": [-0.1],
        "guided_family": ["consensus_projection"], "guided_dose": [0.5],
        "guided_auxiliary_dose": [0.0], "guided_recurrence_prevalence": [0.67],
        "guided_recurrence_max_peaks": [5], "guided_support_weighted": [False],
    })
    joined = panel.enrich_e14(base, exact)
    assert joined[["query_index", "action_id"]].to_dict("records") == [
        {"query_index": 1, "action_id": "a"}
    ]


def test_query_selection_is_source_balanced_and_keeps_multi_action() -> None:
    rows = []
    for source_index, source in enumerate(trainer.SOURCES):
        for local in range(3):
            query = 10 * source_index + local
            for action in range(3 - local):
                rows.append({
                    "source": source, "query_index": query,
                    "query_ik14": f"{source}-{local}", "query_formula": f"F{local}",
                    "source_kind_percentile": 1.0 - 0.1 * local,
                    "action_family": f"family-{action % 2}",
                    "action_id": f"{source}-{local}-{action}",
                })
    frame = pd.DataFrame(rows)
    ranks = np.full(40, 2, dtype=np.int16)
    margins = np.linspace(-0.2, 0.1, 40)
    selected = trainer.select_queries(frame, ranks, margins, 2, want_wrong=True)
    assert selected.groupby("source")["query_index"].nunique().eq(2).all()
    assert selected.groupby(["source", "query_index"]).size().min() >= 2
    bounded = trainer.limit_query_actions(selected, maximum=2, seed=1)
    assert bounded.groupby(["source", "query_index"]).size().eq(2).all()
    batches = trainer.query_batches(bounded, np.random.default_rng(1))
    assert set(batches) == set(trainer.SOURCES)
    assert all(len(source_batches) == 2 for source_batches in batches.values())
    assert all(batch["query_index"].nunique() == 1 for values in batches.values() for batch in values)


def test_query_selection_fills_with_real_single_action_errors() -> None:
    rows = []
    for source_index, source in enumerate(trainer.SOURCES):
        for local in range(8):
            query = 100 * source_index + local
            for action in range(2 if local < 6 else 1):
                rows.append({
                    "source": source, "query_index": query,
                    "query_ik14": f"{source}-{local}", "query_formula": f"F{local}",
                    "source_kind_percentile": 1.0 - 0.01 * local,
                    "action_family": f"family-{action}",
                    "action_id": f"{source}-{local}-{action}",
                })
    frame = pd.DataFrame(rows)
    ranks = np.full(400, 2, dtype=np.int16)
    margins = np.linspace(-0.2, 0.1, 400)
    selected = trainer.select_queries(frame, ranks, margins, 8, want_wrong=True)
    assert selected.groupby("source")["query_index"].nunique().eq(8).all()
    multi = selected.groupby(["source", "query_index"]).size().ge(2).groupby(level=0).sum()
    assert multi.eq(6).all()


def test_harmful_branch_cannot_execute_action_payload() -> None:
    source = inspect.getsource(trainer.risk_loss)
    assert "action_tensor(" not in source
    assert "action_payload" not in source
    assert "drop_duplicates" in source


def test_trainer_keeps_dropout_off_and_forbids_p2b() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "model.eval()  # gradients on, dropout off" in source
    assert '"P2b": "forbidden"' in source
    assert '"P3_consumed": False' in source
    assert "within epoch" in source
    assert "project_corrective_against_risk" in source
    assert "torch.autograd.grad(corr_loss" in source
    assert "torch.autograd.grad(safe_loss" in source
    assert 'default=0,' in source
    assert '"one_optimizer_step_per_source_query": True' in source
    assert "finite_mean" in source


def test_cosine_is_bounded_and_nan_mean_is_local() -> None:
    value = torch.linspace(-1.0, 1.0, 100_000)
    similarity = trainer.cosine(value, value)
    assert 0.999999999 <= similarity <= 1.0
    assert trainer.finite_mean([float("nan"), 1.0, 3.0]) == 2.0


def test_corrective_training_weight_contract() -> None:
    source = inspect.getsource(trainer.corrective_loss)
    assert 'row.get("training_weight", 1.0)' in source


class _Store:
    def __init__(self) -> None:
        self.values = {}
        for row in (1, 2, 3):
            value = torch.zeros(6, 2)
            value[0] = torch.tensor([100.0 + row, 1.0])
            value[1] = torch.tensor([20.0 + row, 1.0])
            value[2] = torch.tensor([40.0 + row, 0.5])
            self.values[row] = value

    def one(self, row: int) -> torch.Tensor:
        return self.values[int(row)]

    def get(self, rows) -> torch.Tensor:
        return torch.stack([self.one(int(row)) for row in rows])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(12, 8, bias=False)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.linear(spectra.flatten(1)), dim=1)


class _Args:
    amp = False
    rank_margin = 0.05
    temperature = 0.10
    target_delta_min = 0.01
    target_delta_max = 0.05
    lambda_action_rank = 1.0
    lambda_transfer = 0.5
    lambda_teacher = 0.5
    lambda_preserve = 1.0
    risk_margin_slack = 0.005


def test_losses_are_differentiable_and_risk_ignores_bad_payload() -> None:
    torch.manual_seed(1)
    model, store = _Model(), _Store()
    with torch.no_grad():
        initial = model(torch.stack([store.one(row) for row in (1, 2, 3)])).numpy()
    index = {1: 0, 2: 1, 3: 2}
    refs = {0: ((2,), (3,))}
    base = {
        "source": "A4_exact", "query_index": 0, "query_row": 1,
        "query_ik14": "I", "query_formula": "F", "action_id": "a",
        "action_payload": '{"token": 2, "mz": 42.0, "role": "x"}',
        "dose": 0.5, "margin_delta": 0.1, "initial_margin": -0.1,
        "source_kind_percentile": 0.75,
        "query_action_count": 1,
    }
    correction, _ = trainer.corrective_loss(
        model, store, pd.DataFrame([base]), refs, initial, index,
        torch.device("cpu"), _Args(),
    )
    correction.backward()
    assert torch.isfinite(correction)
    model.zero_grad(set_to_none=True)
    harmful = dict(base)
    harmful["action_payload"] = "this payload must never be parsed"
    risk, _ = trainer.risk_loss(
        model, store, pd.DataFrame([harmful]), refs, initial, index,
        torch.device("cpu"), _Args(),
    )
    risk.backward()
    assert torch.isfinite(risk)


def main() -> None:
    test_e14_join_is_fail_closed()
    test_query_selection_is_source_balanced_and_keeps_multi_action()
    test_query_selection_fills_with_real_single_action_errors()
    test_harmful_branch_cannot_execute_action_payload()
    test_trainer_keeps_dropout_off_and_forbids_p2b()
    test_cosine_is_bounded_and_nan_mean_is_local()
    test_corrective_training_weight_contract()
    test_losses_are_differentiable_and_risk_ignores_bad_payload()
    print("[test_noise_final_e15_m2] PASS", flush=True)


if __name__ == "__main__":
    main()

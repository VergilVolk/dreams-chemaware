"""Fast deterministic checks for the E4-B0 gradient-attribution audit."""
from __future__ import annotations

import ast
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
if str(TASKS) not in sys.path:
    sys.path.insert(0, str(TASKS))

from audit_noise_final_e4b0_gradient_attribution import (
    GradientExample, bootstrap_mean, grad_cosine, grad_norm, gradients,
    loss_bundle, select_formula_panel,
)


AUDIT = ROOT / "tasks/audit_noise_final_e4b0_gradient_attribution.py"
SBATCH = ROOT / "tasks/run_noise_final_e4b0_gradient_attribution.sbatch"


def synthetic_actions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    query = 0
    for formula_index in range(48):
        formula = f"C{formula_index + 10}H{formula_index + 20}O2"
        for selector, attenuation, steps in (
            ("candidate_gradient", 0.50, range(3, 7)),
            ("role_confounder", 1.00, range(1, 6)),
        ):
            for step in steps:
                rows.append({
                    "query_index": query,
                    "query_row": 10_000 + query,
                    "query_ik14": f"ID{query:06d}",
                    "query_formula": formula,
                    "selector": selector,
                    "attenuation": attenuation,
                    "step": step,
                    "target_path": ",".join(map(str, range(step))),
                    "matched_control_paths": (
                        ",".join(map(str, range(20, 20 + step))) + ";"
                        + ",".join(map(str, range(40, 40 + step)))
                    ),
                    "hard_negative_row": 20_000 + query,
                    "formula_fold": formula_index % 5,
                })
                query += 1
    return pd.DataFrame(rows)


class TinyStore:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(11)
        self.tensor: dict[int, torch.Tensor] = {}
        for row in range(13):
            spectrum = torch.zeros(100, 2)
            spectrum[0] = torch.tensor([500.0, 1.0])
            spectrum[1:16, 0] = torch.linspace(50.0, 190.0, 15)
            spectrum[1:16, 1] = torch.rand(15, generator=generator) + 0.2
            spectrum[:, 1] /= torch.linalg.vector_norm(spectrum[:, 1])
            self.tensor[row] = spectrum

    def one(self, row: int) -> torch.Tensor:
        return self.tensor[int(row)]

    def get(self, rows: tuple[int, ...]) -> torch.Tensor:
        return torch.stack([self.one(row) for row in rows])


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(200, 8, bias=False)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.head(spectra.flatten(1)), dim=1)


def main() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    sbatch = SBATCH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer_steps\": 0" in source
    assert "P2b\": \"forbidden\"" in source
    assert "P3_consumed\": False" in source
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem=" not in sbatch
    assert "audit_noise_final_e4b0_gradient_attribution.py" in sbatch
    assert "validate_noise_final_e4b0_gradient_attribution.py" in sbatch

    actions = synthetic_actions()
    first = select_formula_panel(actions, 32, 20260901)
    second = select_formula_panel(actions.sample(frac=1.0, random_state=19), 32, 20260901)
    columns = ["query_formula", "query_index", "selector", "attenuation", "step"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert len(first) == 128
    assert first["query_formula"].nunique() == 32
    assert first.groupby("query_formula")["query_index"].nunique().eq(4).all()
    assert first.groupby(["query_formula", "selector"]).size().eq(2).all()
    assert len(first[["selector", "attenuation", "step"]].drop_duplicates()) == 9

    left = [torch.tensor([3.0, 4.0]), None]
    same = [torch.tensor([6.0, 8.0]), None]
    opposite = [torch.tensor([-3.0, -4.0]), None]
    assert np.isclose(grad_norm(left), 5.0)
    assert np.isclose(grad_cosine(left, same), 1.0)
    assert np.isclose(grad_cosine(left, opposite), -1.0)
    interval = bootstrap_mean(np.asarray([1.0, 2.0, 3.0, 4.0]), 500, 7)
    assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]

    store = TinyStore()
    model = TinyEncoder()
    with torch.no_grad():
        official = {
            row: model(store.one(row)[None])[0].numpy() for row in range(13)
        }
    example = GradientExample(
        query_index=0, query_row=0, identity="I0", formula="C10H20O2",
        selector="candidate_gradient", step=3, attenuation=0.5,
        target_path=(1, 2, 3), control_paths=((4, 5, 6), (7, 8, 9)),
        selected_control=0, positive_rows=(1, 2, 3, 4),
        negative_rows=(5, 6, 7, 8, 9, 10, 11, 12),
        official_margin=-0.02, official_rank=2,
    )
    losses, outputs = loss_bundle(
        model, store, [example], official, torch.device("cpu"),
        Namespace(rank_margin=0.05, specificity_margin=0.01, temperature=0.10),
    )
    assert set(losses) == {
        "common", "current_target_branch", "current_target_minus_random",
        "paired_advantage",
    }
    assert outputs["paired_advantage"].shape == (1,)
    parameters = list(model.parameters())
    for position, name in enumerate(losses):
        values = gradients(losses[name], parameters, position < len(losses) - 1)
        assert grad_norm(values) > 0
    print("[test_noise_final_e4b0_gradient_attribution] PASS")


if __name__ == "__main__":
    main()

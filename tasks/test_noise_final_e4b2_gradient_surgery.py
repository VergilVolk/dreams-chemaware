"""Fast tests for E4-B2 paired panels and gradient transformations."""
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
sys.path.insert(0, str(TASKS))

from audit_noise_final_e4b2_gradient_surgery_screen import (  # noqa: E402
    rank_screen, scope_indices, select_discovery_confirmation_panel,
    transformed_gradient,
)
from train_noise_final_e4a_direct_augmentation import FIXED_POLICY  # noqa: E402

AUDIT = TASKS / "audit_noise_final_e4b2_gradient_surgery_screen.py"
SBATCH = TASKS / "run_noise_final_e4b2_gradient_surgery.sbatch"


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    actions: list[dict[str, object]] = []
    signature: list[dict[str, object]] = []
    query = 0
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        for formula_index in range(120):
            formula = f"C{formula_index + 10}H{formula_index + 20}O2"
            for correct in (False, True):
                actions.append({
                    "query_index": query, "query_row": 10_000 + query,
                    "query_ik14": f"ID{query:08d}", "query_formula": formula,
                    "selector": selector, "attenuation": attenuation, "step": step,
                    "target_path": ",".join(map(str, range(step))),
                    "matched_control_paths": (
                        ",".join(map(str, range(20, 20 + step))) + ";"
                        + ",".join(map(str, range(40, 40 + step)))
                    ),
                    "hard_negative_row": 50_000 + query, "formula_fold": 1,
                })
                signature.append({
                    "query_index": query, "dreams_correct": correct,
                    "has_near_candidate": formula_index % 2 == 0,
                    "score_error_family": "official_correct" if correct else "positive_deficit_only",
                })
                query += 1
    return pd.DataFrame(actions), pd.DataFrame(signature)


def main() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert '"optimizer_steps": 0' in source
    assert '"P2b": "forbidden"' in source
    assert 'json_dump(temporary / "report.json", report)' in source
    actions, signature = fixtures()
    b1_panel = actions.loc[
        actions["query_index"].map(lambda value: (int(value) // 2) % 120 < 32)
    ].merge(signature[["query_index", "dreams_correct"]], on="query_index", validate="one_to_one")
    b1_panel["baseline_state"] = np.where(
        b1_panel["dreams_correct"], "official_correct", "official_error",
    )
    b1_panel["cell_id"] = [
        f"{selector}|a={float(attenuation):.2f}|step={int(step)}"
        for selector, attenuation, step in b1_panel[
            ["selector", "attenuation", "step"]
        ].itertuples(index=False, name=None)
    ]
    args = Namespace(
        seed=20260902, screen_formulas_per_stratum=32,
        maximum_confirm_formulas_per_stratum=64,
    )
    first = select_discovery_confirmation_panel(actions, signature, b1_panel, args)
    second = select_discovery_confirmation_panel(
        actions.sample(frac=1.0, random_state=19), signature,
        b1_panel.sample(frac=1.0, random_state=23), args,
    )
    columns = ["panel_split", "cell_id", "baseline_state", "query_formula", "query_index"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert not set(first.loc[first["panel_split"].eq("screen"), "query_formula"]).intersection(
        set(first.loc[first["panel_split"].eq("confirm"), "query_formula"])
    )
    for (_, split, _), frame in first.groupby(
        ["cell_id", "panel_split", "baseline_state"]
    ):
        expected = 32 if split == "screen" else 64
        assert frame["query_formula"].nunique() == expected

    action = torch.tensor([1.0, -1.0])
    clean = torch.tensor([0.0, 1.0])
    scopes = scope_indices(
        ["backbone.weight", "head.weight"],
        [
            torch.nn.Parameter(torch.zeros(2, 3)),
            torch.nn.Parameter(torch.zeros(2, 2)),
        ],
    )
    assert torch.equal(scopes["joint"], torch.arange(10))
    assert torch.equal(scopes["backbone"], torch.arange(6))
    assert torch.equal(scopes["head"], torch.arange(6, 10))
    projected = transformed_gradient(action, clean, "pcgrad", 0.0)
    assert torch.allclose(projected, torch.tensor([1.0, 0.0]), atol=1e-7)
    assert float(torch.dot(projected, clean)) >= 0
    anchored = transformed_gradient(action, clean, "pcgrad_anchor", 0.1)
    assert float(torch.dot(anchored, clean)) > 0
    retention = float(torch.dot(action, anchored) / torch.dot(action, action))
    assert retention > 0

    summaries: list[dict[str, object]] = []
    for cell in range(9):
        for configuration in range(18):
            for state in ("official_error", "official_correct"):
                summaries.append({
                    "panel_split": "screen", "cell_id": f"cell-{cell}",
                    "baseline_state": state, "configuration": f"config-{configuration}",
                    "forward_margin_mean": 0.02 - configuration * 1e-5,
                    "forward_margin_ci_low": 0.01,
                    "gradient_consensus_mean": 0.03 - configuration * 1e-5,
                    "clean_alignment_mean": 0.04 - configuration * 1e-5,
                    "action_descent_retention_mean": 0.8,
                })
    ranked, selected = rank_screen(pd.DataFrame(summaries), 3)
    assert len(ranked) == 9 * 18
    assert len(selected) == 3
    assert len({row["cell_id"] for row in selected}) == 3
    assert all(bool(row["screen_gate_pass"]) for row in selected)

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem=" not in sbatch
    assert "audit_noise_final_e4b2_gradient_surgery_screen.py" in sbatch
    assert "validate_noise_final_e4b2_gradient_surgery.py" in sbatch
    assert "--screen-formulas-per-stratum 32" in sbatch
    assert "--maximum-confirm-formulas-per-stratum 64" in sbatch
    assert "g8r_noise_final_e4b2_gradient_surgery_expanded" in sbatch
    print("[test_noise_final_e4b2_gradient_surgery] PASS")


if __name__ == "__main__":
    main()

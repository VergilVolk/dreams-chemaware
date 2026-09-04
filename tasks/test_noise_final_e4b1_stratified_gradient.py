"""Fast deterministic tests for E4-B1 selection and gradient statistics."""
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

from audit_noise_final_e4b1_stratified_gradient import (  # noqa: E402
    bootstrap, loo_cosine, select_balanced_panel,
)
from train_noise_final_e4a_direct_augmentation import FIXED_POLICY  # noqa: E402


AUDIT = TASKS / "audit_noise_final_e4b1_stratified_gradient.py"
SBATCH = TASKS / "run_noise_final_e4b1_stratified_gradient.sbatch"


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    actions: list[dict[str, object]] = []
    signatures: list[dict[str, object]] = []
    query = 0
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        for correct in (False, True):
            for formula_index in range(40):
                formula = f"C{formula_index + 10}H{formula_index + 20}O2"
                actions.append({
                    "query_index": query,
                    "query_row": 10_000 + query,
                    "query_ik14": f"ID{query:07d}",
                    "query_formula": formula,
                    "selector": selector,
                    "attenuation": attenuation,
                    "step": step,
                    "target_path": ",".join(map(str, range(1, step + 1))),
                    "matched_control_paths": (
                        ",".join(map(str, range(20, 20 + step))) + ";"
                        + ",".join(map(str, range(40, 40 + step)))
                    ),
                    "hard_negative_row": 50_000 + query,
                    "formula_fold": 1,
                })
                signatures.append({
                    "query_index": query,
                    "dreams_correct": correct,
                    "has_near_candidate": formula_index % 2 == 0,
                    "score_error_family": (
                        "official_correct" if correct else "positive_deficit_only"
                    ),
                })
                query += 1
    return pd.DataFrame(actions), pd.DataFrame(signatures)


def main() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert '"optimizer_steps": 0' in source
    assert '"P2b": "forbidden"' in source
    actions, signature = fixtures()
    args = Namespace(seed=20260902, formulas_per_primary_stratum=32)
    first = select_balanced_panel(actions, signature, args)
    second = select_balanced_panel(
        actions.sample(frac=1.0, random_state=31), signature, args,
    )
    columns = ["primary_stratum", "query_formula", "query_index"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert len(first) == 576
    assert first["primary_stratum"].nunique() == 18
    assert first.groupby("primary_stratum")["query_formula"].nunique().eq(32).all()
    assert first.groupby(["cell_id", "baseline_state"]).ngroups == 18

    value = [torch.tensor([1.0, 0.0]), None]
    total = [torch.tensor([4.0, 0.0]), None]
    assert np.isclose(loo_cosine(value, total), 1.0)
    interval = bootstrap(
        np.asarray([0.1, 0.2, 0.3, 0.4]), 5000, 19, 0.001,
    )
    assert interval["multiplicity_adjusted_lower"] > 0
    assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]
    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem=" not in sbatch
    assert "audit_noise_final_e4b1_stratified_gradient.py" in sbatch
    assert "validate_noise_final_e4b1_stratified_gradient.py" in sbatch
    print("[test_noise_final_e4b1_stratified_gradient] PASS")


if __name__ == "__main__":
    main()

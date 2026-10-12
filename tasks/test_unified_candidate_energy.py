#!/usr/bin/env python
"""Small scientific contracts for the unified candidate-energy pipeline."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from train_unified_candidate_energy_crossfit import (
    AnchoredCandidateEnergy,
    crossfit,
    formula_fold,
)


ROOT = Path(__file__).resolve().parents[1]


def test_continuous_energy() -> None:
    names = (
        "noise_v1",
        "chemaware_stage1_encoder",
        "weighted_spectral_entropy",
        "p2b_noise_v1_frozen",
        "chemaware_v2_action",
    )
    model = AnchoredCandidateEnergy(names)
    with torch.no_grad():
        model.raw_alpha.zero_()
    evidence = torch.tensor([
        [3.0, -3.0], [2.0, -2.0], [4.0, -4.0],
        [3.5, -3.5], [-4.0, 4.0],
    ])
    availability = torch.ones_like(evidence)
    assert torch.equal(model(evidence, availability), evidence[2])
    assert torch.equal(model.coefficients(), torch.zeros(5))

    # Sparse chemical support is continuous evidence, not an unconditional
    # top-1 rewrite: a small coefficient cannot override a strong anchor.
    with torch.no_grad():
        model.raw_alpha[-1] = 0.1
    result = model(evidence, availability)
    assert int(torch.argmax(result)) == 0
    coefficients = model.coefficients()
    assert torch.all(coefficients >= 0)
    assert float(torch.sum(coefficients).detach()) <= 1.0 + 1e-7
    mixture = model.mixture_weights()
    assert torch.all(mixture >= 0)
    assert abs(float(torch.sum(mixture).detach()) - 1.0) < 1e-7


def test_formula_partition() -> None:
    formulas = ["C6H12O6", "C8H10N4O2", "C6H12O6"]
    first = [formula_fold(value, 5, 20261012) for value in formulas]
    second = [formula_fold(value, 5, 20261012) for value in formulas]
    assert first == second
    assert first[0] == first[2]
    assert all(0 <= value < 5 for value in first)


def test_sparse_adapter_never_needs_truth() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        panel = root / "panel.npz"
        ledger = root / "ledger.npz"
        output = root / "evidence.npz"
        # Object truth cannot be loaded with allow_pickle=False.  Successful
        # conversion therefore demonstrates that the adapter did not access it.
        np.savez_compressed(
            panel,
            query_ptr=np.asarray([0, 2, 5]),
            query_ik14=np.asarray(["Q1", "Q2"]),
            molecule_ik14=np.asarray(["A", "B", "C", "D", "E"]),
            molecule_label=np.asarray([object()], dtype=object),
        )
        np.savez_compressed(
            ledger,
            query_index=np.asarray([0, 1]),
            abstained=np.asarray([False, True]),
            selected_candidate=np.asarray([1, -1]),
        )
        subprocess.run(
            [sys.executable, str(ROOT / "tasks/build_chemaware_v2_sparse_action_evidence.py"),
             "--panel", str(panel), "--action-ledger", str(ledger),
             "--output", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        with np.load(output, allow_pickle=False) as body:
            assert body["scores"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
            assert body["availability"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_formula_target_truth_cannot_change_predictions() -> None:
    names = (
        "noise_v1", "chemaware_stage1_encoder", "weighted_spectral_entropy",
        "p2b_noise_v1_frozen", "chemaware_v2_action",
    )
    queries = 20
    candidates = 3
    ptr = np.arange(0, (queries + 1) * candidates, candidates, dtype=np.int64)
    rng = np.random.default_rng(20261012)
    common = {
        "evidence": rng.normal(size=(len(names), queries * candidates)).astype(np.float32),
        "availability": np.ones((len(names), queries * candidates), dtype=np.float32),
        "query_ptr": ptr,
        "formulas": np.asarray([f"F{index:02d}" for index in range(queries)]),
        "near": np.zeros(queries, dtype=bool),
    }
    train = dict(common)
    train["labels"] = np.tile(np.asarray([1, 0, 0], dtype=np.int8), queries)
    target_a = dict(common)
    target_a["labels"] = train["labels"].copy()
    target_b = dict(common)
    target_b["labels"] = np.tile(np.asarray([0, 0, 1], dtype=np.int8), queries)
    kwargs = dict(
        train=train,
        module_names=names,
        folds=5,
        epochs=1,
        batch_queries=8,
        learning_rate=2e-3,
        anchor_strength=0.01,
        seed=20261012,
        device=torch.device("cpu"),
    )
    prediction_a, _ = crossfit(targets={"formula": target_a}, **kwargs)
    prediction_b, _ = crossfit(targets={"formula": target_b}, **kwargs)
    assert np.array_equal(prediction_a["formula"], prediction_b["formula"])


if __name__ == "__main__":
    test_continuous_energy()
    test_formula_partition()
    test_sparse_adapter_never_needs_truth()
    test_formula_target_truth_cannot_change_predictions()
    print("PASS: unified candidate-energy scientific contracts")

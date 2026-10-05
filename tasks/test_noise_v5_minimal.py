#!/usr/bin/env python
"""Unit tests for the pure parts of the V5 minimal trainer.

Heavy runtime dependencies (torch, h5py, dreams, lightning-based trainers)
are stubbed so the pure ledger/dose logic is verified identically on any
machine, including inside the cluster job gate.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("train_noise_v5_minimal.py")


def install_stubs() -> None:
    def stub(name: str, **attributes: object) -> None:
        module = types.ModuleType(name)
        module.__path__ = []  # allow import traversal through this package stub
        for key, value in attributes.items():
            setattr(module, key, value)
        sys.modules.setdefault(name, module)

    stub("torch", nn=types.SimpleNamespace(), optim=types.SimpleNamespace())
    stub("torch.nn")
    stub("torch.nn.functional")
    stub("h5py")
    stub(
        "dreams.models.heads.heads",
        ContrastiveHead=object,
    )
    stub("dreams.models.heads")
    stub("dreams.models")
    stub("dreams")
    stub(
        "dreams.utils.data",
        SpectrumPreprocessor=object,
    )
    stub("dreams.utils")
    stub("dreams.utils.dformats", DataFormatA=object)
    stub("noise_final_core", sha256_file=lambda path: "stub")
    stub("train_noise_dreams_native", make_hdf5_spectrum=lambda row, mz: row)
    stub(
        "train_noise_dreams_native_residual_stage2",
        construct_native_model=lambda args: (None, "stub"),
    )
    # `import torch.nn.functional as F` binds through parent attributes,
    # so link the stub chain explicitly.
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]


def load_trainer_module():
    install_stubs()
    spec = importlib.util.spec_from_file_location(
        "train_noise_v5_minimal_under_test", MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = load_trainer_module()

    def make_selected(count: int = 3) -> dict[str, np.ndarray]:
        rows = np.arange(count, dtype=np.int64)
        return {name: rows * 10 + offset for name, offset in (
            ("query_index", 0), ("action_index", 1), ("query_row", 2),
            ("exact_positive_row", 3), ("exact_negative_row", 4),
            ("clean_positive_winner_row", 5), ("clean_top_rival_winner_row", 6),
        )}

    selected = make_selected()
    assert trainer.strict_selected_schema(selected) == 3
    for bad in (
        {**selected, "exact_positive_row": selected["exact_positive_row"][:-1]},
        {k: v.astype(np.int32) for k, v in selected.items()},
        {name: value for name, value in selected.items() if name != "query_row"},
        {**selected, "query_index": np.empty(0, dtype=np.int64)},
    ):
        try:
            trainer.strict_selected_schema(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed selected ledger passed the schema")

    assert trainer.dedupe_negative_rows(40, 50, 10) == [40, 50]
    assert trainer.dedupe_negative_rows(40, 40, 10) == [40]
    assert trainer.dedupe_negative_rows(10, 40, 10) == [40]
    assert trainer.dedupe_negative_rows(50, 10, 10) == [50]
    try:
        trainer.dedupe_negative_rows(10, 10, 10)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a unit without negatives must be rejected")

    first = trainer.event_order(3407, 1132)
    second = trainer.event_order(3407, 1132)
    other = trainer.event_order(3408, 1132)
    assert first.shape == (1132,)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert sorted(first.tolist()) == list(range(1132))

    print("[test_noise_v5_minimal] PASS")


if __name__ == "__main__":
    main()

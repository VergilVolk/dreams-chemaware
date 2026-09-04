"""Static and local unit checks for the P/N/S shared-encoder extension."""
from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tasks/build_noise_final_pn_positive_manifest.py"
TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"


def main() -> None:
    builder_source = BUILDER.read_text(encoding="utf-8")
    trainer_source = TRAINER.read_text(encoding="utf-8")
    ast.parse(builder_source)
    ast.parse(trainer_source)
    for token in (
        "same_instrument_cross_ce", "cross_instrument", "same_identity_same_adduct",
        '"teacher": "forbidden"', '"P2b": "forbidden"', "P3_identity_overlap",
    ):
        if token not in builder_source:
            raise RuntimeError(f"P-arm builder contract missing: {token}")
    for token in (
        "positive_arm_loss", "positive_stream_weight", "held_cross_condition_positive",
        "real_cross_condition_positive_pairs_train_same_encoder",
        "clean_and_augmented_raw_spectra_train_same_encoder", "final_shared_encoder.pt",
    ):
        if token not in trainer_source:
            raise RuntimeError(f"P/N/S trainer contract missing: {token}")
    for forbidden in ("p2b_frozen", "privileged_teacher", "corrective_teacher_actions"):
        if forbidden in trainer_source.lower():
            raise RuntimeError(f"downstream/teacher dependency entered P/N/S trainer: {forbidden}")

    # Regression test for the h5py failure encountered earlier: arbitrary
    # unsorted rows must be recovered in their original order.
    import sys
    sys.path.insert(0, str(ROOT / "tasks"))
    from build_noise_final_pn_positive_manifest import read_rows
    from train_noise_final_e4a_direct_augmentation import DirectExample, positive_arm_loss
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("x", data=np.arange(10, dtype=np.int64))
        with h5py.File(path, "r") as handle:
            observed = read_rows(handle, "x", np.asarray([7, 2, 7, 1]))
        if not np.array_equal(observed, np.asarray([7, 2, 7, 1])):
            raise RuntimeError("sorted HDF5 gather did not restore requested row order")

    class Store:
        def __init__(self):
            self.values = {
                0: torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                1: torch.tensor([[0.9, 0.1], [0.0, 0.0]]),
                2: torch.tensor([[0.1, 0.9], [0.0, 0.0]]),
            }
        def one(self, row):
            return self.values[int(row)]
        def get(self, rows):
            return torch.stack([self.values[int(row)] for row in rows])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
        def forward(self, spectra):
            return F.normalize(spectra[:, 0, :] @ self.weight, dim=1)

    from types import SimpleNamespace
    example = DirectExample(
        query_index=0, query_row=0, identity="A", formula="F",
        positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
        official_rank=1, sample_weight=1.0, policy="positive|cross_instrument",
    )
    official = {
        0: np.asarray([1.0, 0.0], dtype=np.float32),
        1: F.normalize(torch.tensor([0.9, 0.1]), dim=0).numpy(),
        2: F.normalize(torch.tensor([0.1, 0.9]), dim=0).numpy(),
    }
    model = Model()
    args = SimpleNamespace(
        amp=False, rank_margin=0.05, temperature=0.1,
        margin_floor_slack=0.005, lambda_positive_rank=1.0,
        lambda_positive_margin_floor=2.0, lambda_preserve=5.0,
    )
    loss, log = positive_arm_loss(model, Store(), [example], official, torch.device("cpu"), args)
    loss.backward()
    if not torch.isfinite(loss) or model.weight.grad is None or not torch.isfinite(model.weight.grad).all():
        raise RuntimeError("P-arm loss did not produce finite shared-encoder gradients")
    if log["positive_margin"] <= 0:
        raise RuntimeError("P-arm unit example has an unexpected ranking direction")
    print("[test_noise_final_pn_shared_encoder] PASS")


if __name__ == "__main__":
    main()

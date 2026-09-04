"""Fail-closed unit tests for E13 relaxed positive-noise transfer."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import train_noise_final_e4a_direct_augmentation as trainer  # noqa: E402


class FakeStore:
    def one(self, row: int) -> torch.Tensor:
        return torch.full((2, 4), float(row), dtype=torch.float32)


def main() -> None:
    source_path = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
    source = source_path.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "guided_recurrence_prevalence",
        "guided_recurrence_max_peaks",
        "guided_transfer_mode",
        "guided_action_authorization_report_sha256",
        "top3_by_frozen_mature_e8_embedding",
        "E13-reference-fp32",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"E13 contract missing: {missing}")

    # The relaxed recipe must reach the peak constructor without falling back
    # to the historical 0.67/max-5 constants.
    observed: dict[str, object] = {}
    old_profile = trainer.reference_profile
    old_missing = trainer.recurrent_missing_peaks
    old_transfer = trainer.apply_positive_peak_transfer
    try:
        trainer.reference_profile = lambda clean, refs, tol: (
            np.asarray([0.5], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )

        def fake_missing(clean, refs, tol, prevalence, maximum):
            observed["prevalence"] = prevalence
            observed["maximum"] = maximum
            return np.asarray([[100.0, 0.25, 0.75]], dtype=np.float32)

        def fake_transfer(clean, missing, prevalence, family, dose):
            observed["family"] = family
            observed["dose"] = dose
            return clean + 1.0, len(missing)

        trainer.recurrent_missing_peaks = fake_missing
        trainer.apply_positive_peak_transfer = fake_transfer
        example = trainer.GuidedNoiseExample(
            query_index=0, query_row=1, identity="A", formula="F",
            positive_rows=(2,), negative_rows=(3,), action_reference_rows=(4, 5),
            official_margin=0.0, official_rank=2, sample_weight=1.0,
            policy="positive_recurrent_peak_transfer",
            family="recurrent_union_mix", dose=0.50,
        )
        variant = trainer.guided_variant(FakeStore(), example, 0.50, 10)
        if observed != {
            "prevalence": 0.50, "maximum": 10,
            "family": "recurrent_union_mix", "dose": 0.50,
        }:
            raise RuntimeError(f"relaxed recurrence semantics drifted: {observed}")
        if not torch.equal(variant, FakeStore().one(1) + 1.0):
            raise RuntimeError("guided action variant was not returned")
    finally:
        trainer.reference_profile = old_profile
        trainer.recurrent_missing_peaks = old_missing
        trainer.apply_positive_peak_transfer = old_transfer

    # The mechanism control must block the action-side consistency gradient;
    # the E13 main arm must propagate it through both shared-encoder branches.
    clean = torch.tensor([[0.8, 0.6]], requires_grad=True)
    action = torch.tensor([[0.6, 0.8]], requires_grad=True)
    trainer.guided_consistency_values(clean, action, "stopgrad").sum().backward()
    if clean.grad is None or action.grad is not None:
        raise RuntimeError("stopgrad control has the wrong gradient graph")
    clean = torch.tensor([[0.8, 0.6]], requires_grad=True)
    action = torch.tensor([[0.6, 0.8]], requires_grad=True)
    trainer.guided_consistency_values(clean, action, "symmetric").sum().backward()
    if clean.grad is None or action.grad is None or float(action.grad.norm()) == 0.0:
        raise RuntimeError("symmetric E13 action branch receives no gradient")

    print("[test_noise_final_e13_relaxed_transfer] PASS")


if __name__ == "__main__":
    main()

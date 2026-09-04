"""Fast unit tests for the L0 full-candidate action ledger."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audit_noise_final_l0_action_learnability_ledger import (
    advantage_label, encode_action_variants, molecule_outcome, transition_label,
    validate_action_row,
)
import torch


def main() -> None:
    # Three molecules: positive max=0.8, negatives max=0.8 and 0.2.  The tie
    # must count against the positive, even though stable argmax diagnoses the
    # positive as the first top molecule.
    rank, margin, top = molecule_outcome(
        np.asarray([0.8, 0.7, 0.8, 0.2]), np.asarray([0, 2, 3, 4]),
    )
    assert rank == 2 and np.isclose(margin, 0.0) and top == 0
    rank, margin, top = molecule_outcome(
        np.asarray([0.81, 0.7, 0.8, 0.2]), np.asarray([0, 2, 3, 4]),
    )
    assert rank == 1 and np.isclose(margin, 0.01) and top == 0

    assert transition_label(1, 1) == "protected_correct"
    assert transition_label(1, 2) == "introduced"
    assert transition_label(3, 1) == "corrected"
    assert transition_label(3, 2) == "persistent_wrong"
    assert advantage_label(0.01, 0.01) == "positive"
    assert advantage_label(-0.01, 0.01) == "harmful"
    assert advantage_label(0.009, 0.01) == "neutral"

    row = SimpleNamespace(
        target_path="1,2,3", matched_control_paths="4,5,6;7,8,9", step=3,
    )
    target, controls = validate_action_row(row, 101)
    assert target == (1, 2, 3) and controls == ((4, 5, 6), (7, 8, 9))
    try:
        validate_action_row(
            SimpleNamespace(target_path="1,2", matched_control_paths="1,2;3,4", step=2), 101,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("target/control alias was accepted")

    class UnitModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = SimpleNamespace(out_features=2)

        def forward(self, spectra: torch.Tensor) -> torch.Tensor:
            values = spectra[:, 0, :2].float()
            return torch.nn.functional.normalize(values, p=2, dim=1)

    variants = [torch.tensor([[float(index + 1), 1.0]]) for index in range(7)]
    encoded = encode_action_variants(
        UnitModel(), variants, torch.device("cpu"), batch_size=3,
        fp32_retry_batch_size=1, amp=False,
    )
    assert encoded.shape == (7, 2)
    assert np.allclose(np.linalg.norm(encoded, axis=1), 1.0)
    print("[test_noise_final_l0_action_learnability_ledger] PASS", flush=True)


if __name__ == "__main__":
    main()

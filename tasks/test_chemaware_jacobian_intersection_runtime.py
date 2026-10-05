"""CPU runtime contracts for the Jacobian-intersection audit plumbing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_jacobian_intersection_actions import (  # noqa: E402
    JacobianSetting,
    build_common_scope_plans,
    input_jacobians,
    reference_deltas,
)
from chemaware_boundary_consensus_action_core import EvidenceProfile  # noqa: E402


class DummyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        intensity = spectra[:, 1:, 1]
        position = torch.arange(
            1,
            intensity.shape[1] + 1,
            dtype=intensity.dtype,
            device=intensity.device,
        )
        return (
            torch.stack(
                (intensity.sum(dim=1), (intensity * position).sum(dim=1)),
                dim=1,
            )
            * self.scale
        )


class FakeGraph:
    dreams_column = 0
    features = np.asarray([[0.9], [0.7], [0.8], [0.6]], dtype=np.float32)

    @staticmethod
    def query_block(_query: int):
        return (
            slice(0, 4),
            np.asarray([10, 11, 12, 13], dtype=np.int64),
            np.asarray([0, 2, 4], dtype=np.int64),
            0,
        )


def test_input_jacobian_is_log_intensity_derivative() -> None:
    model = DummyEncoder()
    spectra = torch.tensor(
        [
            [[500.0, 1.1], [50.0, 1.0], [75.0, 0.5]],
            [[400.0, 1.1], [60.0, 0.4], [90.0, 0.8]],
        ]
    )
    delta = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    value = input_jacobians(model, spectra, delta, torch.device("cpu"), batch_size=1)
    expected = np.asarray([[3.0, 2.5], [-0.2, 0.0]], dtype=np.float32)
    assert value.shape == (2, 2)
    assert np.all(np.isfinite(value))
    assert np.allclose(value, expected)
    assert not model.scale.requires_grad


def test_reference_delta_uses_max_reference_inside_fixed_molecules() -> None:
    graph = FakeGraph()
    official = np.asarray(
        [[1.0, 0.0], [0.5, 0.5], [0.2, 0.8], [0.0, 1.0]],
        dtype=np.float32,
    )
    row_position = {10: 0, 11: 1, 12: 2, 13: 3}
    value = reference_deltas(
        graph,
        np.asarray([0]),
        np.asarray([0]),
        [np.asarray([1])],
        official,
        row_position,
    )
    assert np.allclose(value, official[0] - official[2])


def test_four_arm_builder_runs_with_matched_capacity() -> None:
    signed = np.asarray([-0.8, -0.5, 0.7, 0.4, 0.3, -0.2], dtype=np.float32)
    target = EvidenceProfile(
        signed=signed,
        agreement=np.ones(6, dtype=np.float32),
        amplitude=np.ones(6, dtype=np.float32),
    )
    control_signed = np.asarray([-0.3, -0.6, 0.4, -0.2, 0.5, -0.7], dtype=np.float32)
    control = EvidenceProfile(
        signed=control_signed,
        agreement=np.ones(6, dtype=np.float32),
        amplitude=np.ones(6, dtype=np.float32),
    )
    clean = torch.tensor(
        [
            [
                [500.0, 1.1],
                [50.0, 0.7],
                [75.0, 0.6],
                [100.0, 0.65],
                [125.0, 1.0],
                [150.0, 0.5],
                [175.0, 0.4],
            ]
        ]
    )
    jacobian = np.asarray([-0.9, 0.4, 0.8, -0.5, 0.2, -0.1], dtype=np.float32)
    args = SimpleNamespace(
        minimum_agreement=0.75,
        minimum_observed_intensity=0.01,
        precursor_exclusion_da=1.1,
    )
    built = build_common_scope_plans(
        JacobianSetting("bidirectional_sharpen", 0.25, 1, 0.1),
        0,
        [target],
        [control],
        [control],
        clean,
        jacobian,
        args,
    )
    assert built is not None
    plans, errors = built
    assert len(plans) == 4 and len(errors) == 2
    assert plans[0].attenuated == plans[1].attenuated == plans[2].attenuated
    assert plans[0].boosted == plans[1].boosted == plans[2].boosted
    assert np.isclose(
        -sum(jacobian[plans[3].positions] * np.log(plans[3].factors)),
        sum(jacobian[plans[0].positions] * np.log(plans[0].factors)),
    )


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware Jacobian runtime contracts")


if __name__ == "__main__":
    main()

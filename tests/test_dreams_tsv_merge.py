from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_dreams_tsv", ROOT / "tasks/merge_dreams_tsv.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def official_reference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    rank = min(first.shape)
    keep = int(rank / 2)
    sum_u = torch.zeros(first.shape[0], rank)
    sum_s = torch.zeros(rank)
    sum_v = torch.zeros(rank, first.shape[1])
    for index, vector in enumerate((first, second)):
        u, singular, vh = torch.linalg.svd(vector, full_matrices=False)
        left, right = index * keep, (index + 1) * keep
        sum_u[:, left:right] = u[:, :keep]
        sum_s[left:right] = singular[:keep]
        sum_v[left:right, :] = vh[:keep, :]
    u_u, _s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))


def test_two_task_tsv_matches_author_operation_order() -> None:
    generator = torch.Generator().manual_seed(3407)
    first = torch.randn(7, 5, generator=generator)
    second = torch.randn(7, 5, generator=generator)
    observed = MODULE.official_tsv_two_vector_update(
        first, second, device=torch.device("cpu")
    )
    expected = official_reference(first, second)
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-5)


def test_non_matrix_update_is_arithmetic_mean() -> None:
    first = torch.tensor([1.0, 3.0, 5.0])
    second = torch.tensor([5.0, 1.0, -1.0])
    observed = MODULE.official_tsv_two_vector_update(
        first, second, device=torch.device("cpu")
    )
    assert torch.equal(observed, torch.tensor([3.0, 2.0, 2.0]))


def test_merge_section_preserves_buffers_and_builds_linear_mean() -> None:
    base = {
        "weight": torch.zeros(4, 4),
        "bias": torch.zeros(4),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    noise = {
        "weight": torch.eye(4),
        "bias": torch.ones(4),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    chem = {
        "weight": 2 * torch.eye(4),
        "bias": 3 * torch.ones(4),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    linear, _tsv, stats = MODULE.merge_section(
        base, noise, chem, device=torch.device("cpu")
    )
    assert torch.equal(linear["weight"], 1.5 * torch.eye(4))
    assert torch.equal(linear["bias"], 2 * torch.ones(4))
    assert torch.equal(linear["counter"], base["counter"])
    assert stats == {
        "two_dimensional_float_tensors": 1,
        "non_2d_float_tensors": 1,
        "non_float_tensors": 1,
    }


def test_incompatible_nonfloating_buffer_is_rejected() -> None:
    base = {"counter": torch.tensor(1, dtype=torch.int64)}
    noise = {"counter": torch.tensor(2, dtype=torch.int64)}
    chem = {"counter": torch.tensor(1, dtype=torch.int64)}
    try:
        MODULE.validate_states(base, noise, chem)
    except RuntimeError as error:
        assert "non-floating buffer differs" in str(error)
    else:
        raise AssertionError("buffer drift must fail")

from types import SimpleNamespace

import pytest
import torch

from dreams.models.dreams.layers import MultiheadAttention
from dreams.models.chem_aware.chem_aware_dreams import route_chemical_bias_to_layer
from dreams.models.chem_aware.peak_rule_attention_v3 import (
    PeakRuleBiasStore,
    deterministic_peak_permutation,
    deterministic_spectrum_permutation,
)


def _attention() -> MultiheadAttention:
    torch.manual_seed(1701)
    module = MultiheadAttention(SimpleNamespace(
        d_model=8,
        n_heads=2,
        att_dropout=0.0,
        no_transformer_bias=False,
        attn_mech="dot-product",
        d_graphormer_params=0,
    ))
    module.eval()
    return module


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1702)
    values = torch.randn(1, 4, 8)
    padding = torch.zeros(1, 4, dtype=torch.bool)
    return values, padding


def test_zero_chemical_bias_is_exactly_the_no_bias_path() -> None:
    module = _attention()
    values, padding = _inputs()
    without, weights_without = module(values, values, values, padding)
    with_zero, weights_zero = module(
        values, values, values, padding,
        chem_bias=torch.zeros(1, 1, 4, 4),
    )
    torch.testing.assert_close(with_zero, without, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weights_zero, weights_without, rtol=0.0, atol=0.0)


def test_nonuniform_chemical_bias_changes_attention_before_softmax() -> None:
    module = _attention()
    values, padding = _inputs()
    without, weights_without = module(values, values, values, padding)
    bias = torch.zeros(1, 1, 4, 4, requires_grad=True)
    with torch.no_grad():
        bias[0, 0, 0, 2] = 6.0
    with_bias, weights_bias = module(
        values, values, values, padding, chem_bias=bias
    )
    assert weights_bias[0, :, 0, 2].mean() > weights_without[0, :, 0, 2].mean()
    assert not torch.equal(with_bias, without)
    with_bias.square().sum().backward()
    assert bias.grad is not None
    assert torch.count_nonzero(bias.grad) > 0


@pytest.mark.parametrize(
    "bias",
    (
        torch.zeros(1, 4, 4),
        torch.zeros(1, 3, 4, 4),
        torch.zeros(2, 1, 4, 4),
        torch.zeros(1, 1, 3, 4),
        torch.full((1, 1, 4, 4), float("nan")),
    ),
)
def test_invalid_chemical_bias_fails_closed(bias: torch.Tensor) -> None:
    module = _attention()
    values, padding = _inputs()
    with pytest.raises((ValueError, RuntimeError)):
        module(values, values, values, padding, chem_bias=bias)


def test_chemical_bias_is_routed_to_exactly_the_requested_layer() -> None:
    bias = torch.randn(2, 1, 4, 4)
    routed, resolved = route_chemical_bias_to_layer(bias, n_layers=7, target_layer=-1)
    assert resolved == 6
    assert len(routed) == 7
    assert sum(item is not None for item in routed) == 1
    assert routed[6] is bias

    routed, resolved = route_chemical_bias_to_layer(bias, n_layers=7, target_layer=2)
    assert resolved == 2
    assert routed[2] is bias


@pytest.mark.parametrize("target", (-8, 7, 99))
def test_out_of_range_chemical_attention_layer_fails_closed(target: int) -> None:
    with pytest.raises(ValueError):
        route_chemical_bias_to_layer(torch.zeros(1, 1, 2, 2), 7, target)


def test_spectrum_permuted_control_is_a_deterministic_derangement() -> None:
    first = deterministic_spectrum_permutation(17, seed=1701)
    second = deterministic_spectrum_permutation(17, seed=1701)
    np = pytest.importorskip("numpy")
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(17))
    assert not np.any(first == np.arange(17))


def test_peak_permutation_is_deterministic_and_has_no_fragment_fixed_points() -> None:
    first = deterministic_peak_permutation(11, seed=1701)
    second = deterministic_peak_permutation(11, seed=1701)
    np = pytest.importorskip("numpy")
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(11))
    assert not np.any(first == np.arange(11))


def test_idf_precursor_peak_control_preserves_exact_per_spectrum_bias_values() -> None:
    np = pytest.importorskip("numpy")
    source = SimpleNamespace(
        rows=np.asarray([10, 20, 30], dtype=np.int64),
        tensor=torch.tensor([
            [[100.0, 1.1], [81.9894, 1.0], [43.0184, 0.8], [70.0651, 0.5], [0.0, 0.0]],
            [[150.0, 1.1], [131.9894, 1.0], [91.0542, 0.8], [98.9842, 0.5], [0.0, 0.0]],
            [[200.0, 1.1], [182.9735, 1.0], [136.0618, 0.8], [113.0346, 0.5], [0.0, 0.0]],
        ]),
    )
    correct = PeakRuleBiasStore(
        source, scale=0.5, control="correct", seed=17,
        categories=("NL", "CF"), bias_kind="idf_precursor",
    )
    permuted = PeakRuleBiasStore(
        source, scale=0.5, control="peak_permuted", seed=17,
        categories=("NL", "CF"), bias_kind="idf_precursor",
    )
    assert torch.count_nonzero(correct.base_bias[:, :, 1:, :]) == 0
    assert torch.count_nonzero(correct.base_bias[:, :, :, 0]) == 0
    assert not torch.equal(correct.base_bias, permuted.permuted_bias)
    torch.testing.assert_close(
        torch.sort(correct.base_bias.reshape(3, -1), dim=1).values,
        torch.sort(permuted.permuted_bias.reshape(3, -1), dim=1).values,
        rtol=0.0,
        atol=0.0,
    )
    assert permuted.audit["within_spectrum_topology_preserved"]
    assert permuted.audit["peak_permutation_fixed_points"] == 0
    assert permuted.audit["construction"]["precursor_query_only"]

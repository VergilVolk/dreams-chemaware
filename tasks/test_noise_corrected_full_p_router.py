from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from audit_noise_corrected_full_p_router import (
    REGISTERED_FORMAL_P_ROUTE_CONFIGURATION,
    _exact_tensor_deduplication_plan,
    _route,
    _validate_registered_formal_configuration,
)


def main() -> None:
    args = SimpleNamespace(
        paired_advantage_threshold=0.01,
        harm_margin_threshold=0.01,
        robustness_slack=0.005,
    )
    assert _route(2, -0.03, 1, 0.04, 0.00, args) == "corrective"
    assert _route(1, 0.04, 2, -0.02, 0.01, args) == "harmful"
    assert _route(1, 0.04, 1, 0.038, 0.03, args) == "robustness_only"
    assert _route(2, -0.03, 2, -0.025, -0.03, args) == "uncertain"
    values = [np.zeros((101, 2)), np.ones((101, 2))]
    stacked = np.stack(values).astype(np.float32)
    assert stacked.shape == (2, 101, 2) and stacked.dtype == np.float32
    first = torch.zeros((101, 2), dtype=torch.float32)
    second = torch.ones((101, 2), dtype=torch.float32)
    unique, inverse = _exact_tensor_deduplication_plan([
        first, second, first.clone(), second.clone(),
    ])
    assert len(unique) == 2
    assert inverse.tolist() == [0, 1, 0, 1]
    restored = torch.stack([unique[int(index)] for index in inverse])
    expected = torch.stack([first, second, first, second])
    assert torch.equal(restored, expected)
    formal = {**REGISTERED_FORMAL_P_ROUTE_CONFIGURATION, "formal": True}
    _validate_registered_formal_configuration(SimpleNamespace(**formal))
    formal["fragment_tolerance"] = 0.03
    try:
        _validate_registered_formal_configuration(SimpleNamespace(**formal))
    except RuntimeError as error:
        assert "fragment_tolerance" in str(error)
    else:
        raise AssertionError("formal P route accepted configuration drift")
    print("PASS=3")


if __name__ == "__main__":
    main()

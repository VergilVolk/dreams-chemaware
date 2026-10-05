from __future__ import annotations

import numpy as np
import torch

from audit_chemaware_crossview_routed_shared_adapter import identity_balanced_route_weights
from audit_noise_direct_shared_metric_reachability import SharedResidualMetric


def main() -> None:
    identity = np.asarray(["a", "a", "b", "c"])
    baseline = np.asarray([2, 1, 2, 1])
    correctable = np.asarray([True, False, False, False])
    mask = np.ones(4, dtype=bool)
    weight = identity_balanced_route_weights(
        identity, baseline, correctable, mask,
        correctable_weight=4.0, other_error_weight=0.75, correct_control_weight=0.5,
    )
    assert np.isclose(weight.mean(), 1.0)
    assert np.isclose(weight[:2].sum(), weight[2:3].sum())
    assert np.isclose(weight[2:3].sum(), weight[3:].sum())
    model = SharedResidualMetric(8, 3, 0.25)
    value = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    assert torch.equal(model(value), value)
    print("PASS: ChemAware cross-view routed shared-adapter contracts")


if __name__ == "__main__":
    main()

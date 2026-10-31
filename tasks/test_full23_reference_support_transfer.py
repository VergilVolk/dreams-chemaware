#!/usr/bin/env python
"""Small contracts for the large-source reference-support transfer trainer."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from train_enveda_unified_candidate_integrator import PanelData
import train_full23_reference_support_transfer as target


def panel(name: str) -> tuple[PanelData, np.ndarray]:
    ptr = np.asarray([0, 2, 5, 7], dtype=np.int64)
    labels = np.asarray([1, 0, 0, 1, 0, 1, 0], dtype=np.int8)
    features = np.zeros((7, 15), dtype=np.float32)
    features[:, 0] = np.asarray([0, 1, 1, 0, 0.5, 0, 1], dtype=np.float32)
    features[:, 1:] = np.linspace(-0.2, 0.2, 7 * 14, dtype=np.float32).reshape(7, 14)
    parent = np.asarray([0.1, 0.2, 0.3, 0.2, 0.0, 1.0, 0.0], dtype=np.float32)
    body = PanelData(
        name=name, graph=None, scalar=features, feature_names=(), labels=labels,
        query_ptr=ptr, formulas=np.asarray(["A", "B", "C"]),
        query_keys=np.asarray(["a", "b", "c"]), near=np.asarray([1, 1, 0], bool),
        query_weights=np.asarray([0.5, 1.0, 1.0], dtype=np.float32),
        scores={"full23_v1": parent},
    )
    return body, features


def main() -> None:
    panels, features = {}, {}
    for name in target.PANELS:
        panels[name], features[name] = panel(name)
    args = SimpleNamespace(
        learning_rate=0.01, weight_decay=0.002, epochs=3, batch_queries=2,
        residual_bound=0.25, boundary_l2=0.002,
    )
    occurrences = [(name, query) for name in target.PANELS for query in range(3)]
    models = {
        "count": target.fit_model(
            panels, features, occurrences, np.asarray([0]), "listwise", args, 1,
        ),
        "support": target.fit_model(
            panels, features, occurrences, np.arange(15), "boundary_top1", args, 2,
        ),
    }
    assert models["count"]["weight"].shape == (1,)
    assert models["support"]["weight"].shape == (15,)
    for model in models.values():
        assert np.all(np.isfinite(model["weight"]))
        for name in target.PANELS:
            score = target.predict(model, panels[name], features[name], args.residual_bound)
            assert score.shape == panels[name].labels.shape
            assert np.all(np.isfinite(score))
            assert np.max(np.abs(score - panels[name].scores["full23_v1"])) <= args.residual_bound + 1e-6
    print("PASS: Full23 reference-support fixed-transfer contracts")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from analyze_lcnec_global_remodeling_stage1 import (
    component_summary,
    decompose,
    mutual_knn_labels,
)


def main() -> None:
    embedding = np.asarray([
        [1.0, 0.0], [0.99, 0.01], [0.98, 0.02],
        [0.0, 1.0], [0.01, 0.99], [0.02, 0.98],
    ])
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    labels, adjacency = mutual_knn_labels(embedding, 2)
    assert adjacency.shape == (6, 6)
    assert np.all(adjacency == adjacency.T)
    assert len(np.unique(labels)) == 2

    values = np.asarray([
        [2.0, 3.0, 4.0, -1.0, -2.0, -3.0],
        [1.0, 1.5, 2.0, 4.0, 4.0, 4.0],
    ])
    family, within, isolated = decompose(values, labels)
    assert np.allclose(values, family + within + isolated)
    assert np.allclose(np.sum(family * within, axis=1), 0.0)
    assert np.allclose(isolated, 0.0)
    summary = component_summary(values.mean(axis=0), labels)
    assert summary["orthogonality_relative_error"] <= 1e-12

    singleton_labels = np.asarray([0, 0, 1])
    family, within, isolated = decompose(np.asarray([1.0, 3.0, 7.0]), singleton_labels)
    assert np.allclose(family, [2.0, 2.0, 0.0])
    assert np.allclose(within, [-1.0, 1.0, 0.0])
    assert np.allclose(isolated, [0.0, 0.0, 7.0])
    print("PASS: LCNEC global-remodelling stage-1 unit contracts")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "showspace"), str(ROOT / "tasks")]

from audit_e0_observability_residual import pair_features
from spectral_features import p2b_pair_features


class DeploymentContracts(unittest.TestCase):
    def test_p2b_feature_parity(self) -> None:
        query = np.asarray([[100.0, 1.0], [150.0, 0.8], [200.0, 0.4], [300.0, 0.2]])
        reference = np.asarray([[100.005, 0.9], [149.99, 0.7], [205.0, 0.5], [302.0, 0.3]])
        deployed = p2b_pair_features(query, 500.0, reference.T, 502.0, 0.73, 0.02)
        source = pair_features(query.T, 500.0, reference.T, 502.0, 0.02)
        expected = np.asarray([
            0.73, source["sqrt_cosine"], source["entropy_similarity"], source["neutral_loss_sqrt_cosine"]
        ])
        np.testing.assert_allclose(deployed, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from tasks.g8r_p2_listwise_core import (
    ResidualListwiseRanker,
    evaluate_query_scores,
    molecule_max_scores,
    query_listwise_loss,
)


class TestG8RP2ListwiseCore(unittest.TestCase):
    def test_zero_initialization_is_exact_dreams(self):
        model = ResidualListwiseRanker(4, hidden_dim=8, delta_bound=0.06)
        features = torch.randn(7, 4)
        baseline = torch.linspace(0.1, 0.7, 7)
        final, delta = model(features, baseline)
        self.assertTrue(torch.equal(final, baseline))
        self.assertEqual(int(torch.count_nonzero(delta)), 0)

    def test_residual_is_bounded(self):
        model = ResidualListwiseRanker(3, hidden_dim=0, delta_bound=0.04)
        with torch.no_grad():
            model.net.weight.fill_(100.0)
            model.net.bias.fill_(100.0)
        final, delta = model(torch.ones(5, 3), torch.zeros(5))
        self.assertTrue(bool(torch.all(delta <= 0.04)))
        self.assertTrue(bool(torch.all(delta >= -0.04)))
        self.assertTrue(torch.allclose(final, delta))

    def test_molecule_aggregation_matches_deployment_max(self):
        pair = torch.tensor([0.2, 0.8, 0.7, 0.1, 0.4])
        ptr = torch.tensor([0, 2, 3, 5])
        got = molecule_max_scores(pair, ptr)
        self.assertTrue(torch.allclose(got, torch.tensor([0.8, 0.7, 0.4])))

    def test_listwise_gradient_pushes_positive_up_and_negatives_down(self):
        model = ResidualListwiseRanker(3, hidden_dim=0, delta_bound=1.0)
        features = torch.eye(3)
        baseline = torch.zeros(3)
        ptr = torch.tensor([0, 1, 2, 3])
        loss = query_listwise_loss(
            model,
            features,
            baseline,
            ptr,
            positive_molecule=0,
            temperature=1.0,
            safety_weight=0.0,
            residual_weight=0.0,
        ).total
        loss.backward()
        gradient = model.net.weight.grad.squeeze(0)
        self.assertLess(float(gradient[0]), 0.0)
        self.assertGreater(float(gradient[1]), 0.0)
        self.assertGreater(float(gradient[2]), 0.0)

    def test_strict_tie_is_an_error(self):
        result = evaluate_query_scores(np.array([0.5, 0.5, 0.2]), [0, 1, 2, 3], 0)
        self.assertEqual(result["rank"], 2)
        self.assertFalse(result["top1"])

    def test_safety_penalizes_destroying_a_correct_baseline_margin(self):
        model = ResidualListwiseRanker(2, hidden_dim=0, delta_bound=0.2)
        with torch.no_grad():
            model.net.weight[:] = torch.tensor([[-4.0, 4.0]])
        features = torch.eye(2)
        baseline = torch.tensor([0.8, 0.6])
        ptr = torch.tensor([0, 1, 2])
        output = query_listwise_loss(
            model,
            features,
            baseline,
            ptr,
            0,
            safety_weight=1.0,
            allowed_margin_drop=0.0,
            residual_weight=0.0,
        )
        self.assertGreater(float(output.baseline_margin), 0.0)
        self.assertLess(float(output.final_margin), float(output.baseline_margin))
        self.assertGreater(float(output.safety), 0.0)


if __name__ == "__main__":
    unittest.main()

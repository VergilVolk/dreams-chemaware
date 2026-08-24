import sys
import tempfile
import unittest
import argparse
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from g8r_p2_listwise_core import FEATURE_NAMES, ResidualListwiseRanker  # noqa: E402
from train_g8r_p2_listwise import (  # noqa: E402
    Configuration,
    ListwiseCache,
    evaluate,
    fit_standardizer,
    selection_tuple,
    train_fixed_epochs,
)


class TestG8RP2Trainer(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cache.npz"
        # Two queries, each with [positive, negative] molecules and one spectrum
        # per molecule. Query 0 is baseline-correct; query 1 is baseline-wrong.
        features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
        features[:, 0] = [0.8, 0.6, 0.4, 0.7]
        features[:, 1:] = np.arange(4, dtype=np.float32)[:, None]
        np.savez_compressed(
            self.path,
            feature_names=np.asarray(FEATURE_NAMES, dtype=object),
            features=features,
            pair_candidate_row=np.arange(4, dtype=np.int64),
            query_ptr=np.asarray([0, 2, 4], dtype=np.int64),
            molecule_ptr=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            molecule_query=np.asarray([0, 0, 1, 1], dtype=np.int64),
            molecule_label=np.asarray([1, 0, 1, 0], dtype=np.int8),
            molecule_ik14=np.asarray(["A", "B", "C", "D"], dtype=object),
            molecule_formula=np.asarray(["F1", "F1", "F2", "F2"], dtype=object),
            molecule_mces_grade=np.asarray([-2, 0, -2, 1], dtype=np.int8),
            query_row=np.asarray([10, 20], dtype=np.int64),
            query_ik14=np.asarray(["A", "C"], dtype=object),
            query_formula=np.asarray(["F1", "F2"], dtype=object),
            query_has_near=np.asarray([True, False]),
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_zero_residual_evaluation_exactly_matches_dreams(self):
        cache = ListwiseCache(self.path)
        queries = np.arange(cache.n_queries)
        mean, scale = fit_standardizer(cache, queries)
        model = ResidualListwiseRanker(len(FEATURE_NAMES), hidden_dim=8, delta_bound=0.06)
        result = evaluate(cache, model, mean, scale, queries, torch.device("cpu"))
        self.assertEqual(result["corrected"], 0)
        self.assertEqual(result["introduced"], 0)
        self.assertEqual(result["delta_recall1"], 0.0)
        self.assertEqual(result["delta_mrr"], 0.0)

    def test_selection_gate_rejects_near_regression(self):
        metrics = {
            "delta_recall1": 0.04,
            "delta_near_recall1": -0.01,
            "delta_mrr": 0.02,
            "corrected": 10,
            "introduced": 1,
        }
        self.assertEqual(selection_tuple(metrics)[0], 0.0)

    def test_one_fixed_training_epoch_runs_on_query_groups(self):
        cache = ListwiseCache(self.path)
        queries = np.arange(cache.n_queries)
        mean, scale = fit_standardizer(cache, queries)
        arguments = argparse.Namespace(
            learning_rate=1e-3,
            weight_decay=0.0,
            temperature=0.1,
            allowed_margin_drop=0.003,
            residual_weight=0.02,
            query_batch_size=2,
        )
        model = train_fixed_epochs(
            cache,
            queries,
            mean,
            scale,
            Configuration("test", 0, 0.03, 4.0, 2.0),
            seed=7,
            epochs=1,
            a=arguments,
            device=torch.device("cpu"),
        )
        result = evaluate(cache, model, mean, scale, queries, torch.device("cpu"))
        self.assertEqual(result["n_queries"], 2)


if __name__ == "__main__":
    unittest.main()

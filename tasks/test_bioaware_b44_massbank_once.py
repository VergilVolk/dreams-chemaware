#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import torch

from evaluate_bioaware_b44_massbank_once import apply_expert, encode_rows, summarize


class DummyModel:
    def predict_proba(self, values):
        logits = values[:, 1] - 0.25 * values[:, 0]
        probability = 1.0 / (1.0 + np.exp(-4.0 * logits))
        return np.column_stack((1.0 - probability, probability))


class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))

    def forward(self, values):
        return self.head(values[:, 0, :])


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "portable.npz"
        spectra = np.zeros((2, 101, 2), dtype=np.float32)
        spectra[0, 0] = (1.0, 0.0)
        spectra[1, 0] = (0.0, 1.0)
        np.savez_compressed(
            cache, hdf5_rows=np.asarray([4, 9], np.int64), spectra=spectra,
            n_highest_peaks=np.asarray(100, np.int64),
            source_massbank_sha256=np.asarray("source"),
        )
        encoded = encode_rows(
            DummyEncoder(), None, np.asarray([4, 9]), torch.device("cpu"),
            batch_size=2, n_highest_peaks=100, spectrum_cache=cache,
        )
        assert np.allclose(encoded, np.eye(2), atol=1e-7)

    rows = []
    for query in range(4):
        truth = f"T{query}"
        for identity, positive, score, member in (
            (truth, True, 0.79, 2.0), (f"W{query}", False, 0.80, 0.0)
        ):
            rows.append({
                "query_id": f"q{query}", "candidate_ik14": identity,
                "truth_ik14": truth, "truth_formula": f"F{query // 2}",
                "ion_mode": "NEGATIVE", "is_positive": positive,
                "spectral_score": score, "reference_spectra": 1,
                "independent_member_count": member,
                "independent_member_intersection": member / 2,
                "independent_log_degree_mean": member,
                "independent_log_degree_min": member / 2,
            })
    candidates = pd.DataFrame(rows)
    _, queries = apply_expert(
        candidates, DummyModel(), {"margin": 0.05, "probability": 0.55},
        None, 7,
    )
    assert queries["baseline_rank"].eq(2).all()
    assert queries["final_rank"].eq(1).all()
    assert queries["corrected"].all() and not queries["introduced"].any()
    report = summarize(queries, candidates, 100, 7)
    assert report["paired"]["delta_recall_at_1"] == 1.0
    assert report["paired"]["delta_macro_query_auc"] == 1.0
    print("[test_bioaware_b44_massbank_once] PASS", flush=True)


if __name__ == "__main__":
    main()

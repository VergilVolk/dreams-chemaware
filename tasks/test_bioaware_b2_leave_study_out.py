#!/usr/bin/env python
"""Runtime smoke test for B2 evidence-context listwise training."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "tasks"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from annotation.bioaware_context_adapter import BiologicalEvidenceContextAdapter
from train_bioaware_b2_leave_study_out import (
    EVIDENCE_MODES, batch_loss, cluster_bootstrap, identity_balanced_epoch, score_query,
)


class FakeData:
    def __init__(self) -> None:
        torch.manual_seed(17)
        self.query = F.normalize(torch.randn(8, 16), dim=-1)
        self.offsets = np.arange(0, 25, 3, dtype=np.int64)
        self.positive = np.zeros(8, dtype=np.int64)
        candidates = []
        evidence = []
        context = []
        for index in range(8):
            candidates.append(F.normalize(self.query[index] + .2 * torch.randn(16), dim=0))
            candidates.extend(F.normalize(torch.randn(2, 16), dim=-1))
            evidence.extend([
                np.ones(12, dtype=np.float32),
                np.full(12, .25, dtype=np.float32),
                np.zeros(12, dtype=np.float32),
            ])
            context.extend([True, True, False])
        self.candidate = torch.stack(candidates)
        self.evidence_raw = np.stack(evidence)
        self.evidence = torch.from_numpy(self.evidence_raw)
        self.context = torch.tensor(context, dtype=torch.bool)

    def section(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


def main() -> None:
    assert EVIDENCE_MODES["full"][0] == tuple(range(12))
    assert EVIDENCE_MODES["full"][1] == tuple(range(10))
    assert EVIDENCE_MODES["smn_rt"][0] == (8, 9, 10, 11)
    assert EVIDENCE_MODES["smn_rt"][1] == (8, 9)
    assert EVIDENCE_MODES["rt_only"][1] == (10, 11)
    balanced = identity_balanced_epoch(
        np.asarray([0, 1, 2, 3]), np.asarray(["A", "A", "B", "C"]),
        np.random.default_rng(17),
    )
    assert len(balanced) == 3
    assert set(np.asarray(["A", "A", "B", "C"])[balanced]) == {"A", "B", "C"}
    data = FakeData()
    model = BiologicalEvidenceContextAdapter(16, 12, hidden_dim=16, update_rank=4)
    args = Namespace(
        temperature=.08, correct_query_weight=2.0, safety_weight=4.0,
        safety_slack=.005, preserve_weight=8.0, gate_weight=.005,
    )
    mean = torch.zeros(12)
    scale = torch.ones(12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, components = batch_loss(model, data, np.arange(8), mean, scale, args)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert model.output.weight.grad is not None
    assert float(model.output.weight.grad.abs().sum()) > 0
    optimizer.step()
    scores, preservation, gate = score_query(model, data, 0, mean, scale)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert preservation > .99
    assert 0 <= gate <= 1
    assert set(components) == {"rank", "safety", "preservation", "gate"}

    # The vectorised bootstrap must remain exactly equivalent to physically
    # concatenating each sampled formula cluster.
    bootstrap_frame = pd.DataFrame({
        "truth_formula": ["A", "A", "B", "C", "C", "C"],
        "delta": [1, 0, -1, 1, 1, 0],
    })
    repeats = 200
    seed = 31
    observed = cluster_bootstrap(bootstrap_frame, repeats, seed)
    groups = [
        group.delta.to_numpy(dtype=float)
        for _, group in bootstrap_frame.groupby("truth_formula", sort=True)
    ]
    rng = np.random.default_rng(seed)
    reference = []
    for _ in range(repeats):
        draw = rng.integers(0, len(groups), size=len(groups))
        reference.append(float(np.concatenate([groups[index] for index in draw]).mean()))
    assert observed["ci_low"] == float(np.quantile(reference, .025))
    assert observed["ci_high"] == float(np.quantile(reference, .975))
    print("[test_bioaware_b2_leave_study_out] PASS")


if __name__ == "__main__":
    main()

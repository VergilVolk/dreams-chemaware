#!/usr/bin/env python
"""CPU contracts for the score-blind MoNA polarity transfer benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_mona_transfer_core import build_panel, validate_panel  # noqa: E402
from evaluate_chemaware_mona_polarity_transfer import (  # noqa: E402
    evaluate_panel,
    paired_metrics,
    summarize,
)


def record(row: int, identity: str, formula: str, mass: float, signature: str) -> dict:
    return {
        "row": row,
        "ik14": identity,
        "formula": formula,
        "precursor_mz": mass,
        "spectrum_hash": signature * 64,
    }


def main() -> None:
    identity_a = "AAAAAAAAAAAAAA"
    identity_b = "BBBBBBBBBBBBBB"
    records = [
        record(0, identity_a, "C2H4O", 100.0000, "a"),
        record(1, identity_a, "C2H4O", 100.0001, "b"),
        record(2, identity_a, "C2H4O", 100.0002, "c"),
        record(3, identity_b, "C2H4O", 100.0000, "d"),
        record(4, identity_b, "C2H4O", 100.0001, "e"),
        record(5, identity_b, "C2H4O", 100.0002, "f"),
    ]
    panel, summary = build_panel(
        records, ppm=20.0, queries_per_identity=4, references_per_candidate=20,
    )
    validate_panel(panel)
    assert summary["queries"] == 6
    assert summary["query_identities"] == 2
    assert np.all(panel["molecule_label"][panel["query_ptr"][:-1]])
    for query in range(len(panel["query_row"])):
        molecule = int(panel["query_ptr"][query])
        left, right = map(int, panel["molecule_ptr"][molecule:molecule + 2])
        assert panel["query_spectrum_hash"][query] not in set(
            panel["candidate_spectrum_hash"][left:right]
        )

    rows = np.arange(6, dtype=np.int64)
    official = np.asarray([
        [1.00, 0.00], [0.99, 0.10], [0.98, 0.20],
        [0.99, 0.12], [0.98, 0.22], [0.97, 0.25],
    ], dtype=np.float32)
    official /= np.linalg.norm(official, axis=1, keepdims=True)
    improved = np.asarray([
        [1.00, 0.00], [1.00, 0.01], [1.00, 0.02],
        [0.00, 1.00], [0.01, 1.00], [0.02, 1.00],
    ], dtype=np.float32)
    improved /= np.linalg.norm(improved, axis=1, keepdims=True)
    old = evaluate_panel(panel, rows, official)
    new = evaluate_panel(panel, rows, improved)
    old_metrics = summarize([old])
    new_metrics = summarize([new])
    assert new_metrics["recall1"] >= old_metrics["recall1"]
    paired = paired_metrics(
        [old], [new], np.asarray(["positive:C2H4O"] * 6), 20, 7,
    )
    assert paired["delta_recall1"] >= 0
    assert paired["corrected_at_1"] >= paired["introduced_at_1"]

    sbatch = (ROOT / "tasks/run_chemaware_mona_polarity_transfer.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --gpus=2" not in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "srun --export=ALL --preserve-env python -u" in sbatch
    assert "build_chemaware_mona_polarity_transfer_panel.py" in sbatch
    assert "evaluate_chemaware_mona_polarity_transfer.py" in sbatch
    print("PASS: ChemAware MoNA polarity-transfer contracts", flush=True)


if __name__ == "__main__":
    main()

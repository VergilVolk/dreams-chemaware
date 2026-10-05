#!/usr/bin/env python
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import evaluate_gnps_gold_silver_10ppm_embeddings as evaluator


ROOT = Path(__file__).resolve().parents[1]


def test() -> None:
    benchmark = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
    used: list[int] = []
    for name in ("identity_disjoint", "formula_disjoint"):
        with np.load(benchmark / f"panel_{name}.npz", allow_pickle=False) as panel:
            used.extend(map(int, panel["query_row"]))
            used.extend(map(int, panel["candidate_row"]))
    rows = np.unique(np.asarray(used, dtype=np.int64))
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(len(rows), 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        baseline = temporary / "baseline.npz"
        candidate = temporary / "candidate.npz"
        output = temporary / "result"
        np.savez_compressed(baseline, rows=rows, embeddings=embeddings)
        np.savez_compressed(candidate, rows=rows, embeddings=embeddings.copy())
        previous = sys.argv
        try:
            sys.argv = [
                "evaluate_gnps_gold_silver_10ppm_embeddings.py",
                "--benchmark", str(benchmark),
                "--baseline-embeddings", str(baseline),
                "--candidate-embeddings", str(candidate),
                "--output", str(output),
                "--bootstrap-resamples", "100",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.main()
        finally:
            sys.argv = previous
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        for panel in report["panels"].values():
            assert panel["paired"]["corrected"] == 0
            assert panel["paired"]["introduced"] == 0
            assert panel["paired"]["risk_net_lambda2"] == 0
            for ci in panel["paired"]["formula_cluster_paired_ci"].values():
                assert ci["delta"] == 0
                assert ci["ci_low"] == 0
                assert ci["ci_high"] == 0


def test_npz_payload_with_npy_suffix_is_loaded_by_content() -> None:
    rows = np.asarray([1, 4], dtype=np.int64)
    embeddings = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "misnamed_embeddings.npy"
        with path.open("wb") as handle:
            np.savez(handle, rows=rows, embeddings=embeddings)
        loaded_rows, loaded_embeddings = evaluator.load_embeddings(path, 5)
    assert np.array_equal(loaded_rows, rows)
    assert np.allclose(np.linalg.norm(loaded_embeddings, axis=1), 1.0)


def main() -> None:
    test()
    test_npz_payload_with_npy_suffix_is_loaded_by_content()
    print("[test_evaluate_gnps_gold_silver_10ppm_embeddings] PASS")


if __name__ == "__main__":
    main()

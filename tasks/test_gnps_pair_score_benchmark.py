#!/usr/bin/env python
from __future__ import annotations

import json
import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate_gnps_gold_silver_10ppm_pair_scores as evaluator
from gnps_pair_score_cache import (
    PANELS,
    load_pair_score_cache,
    sha256_file,
    write_pair_score_cache,
)


def synthetic_panel(path: Path) -> None:
    np.savez_compressed(
        path,
        query_row=np.asarray([0, 1], dtype=np.int64),
        query_ik14=np.asarray(["TRUE_A", "TRUE_B"]),
        query_formula=np.asarray(["C2H4", "C3H6"]),
        near_query=np.asarray([True, True]),
        query_ptr=np.asarray([0, 3, 6], dtype=np.int64),
        molecule_ptr=np.arange(7, dtype=np.int64),
        molecule_label=np.asarray([1, 0, 0, 1, 0, 0], dtype=np.int8),
        molecule_ik14=np.asarray(["TRUE_A", "A1", "A2", "TRUE_B", "B1", "B2"]),
        molecule_formula=np.asarray(["C2H4", "C2H4", "C3H6", "C3H6", "C3H6", "C4H8"]),
        candidate_row=np.asarray([2, 3, 4, 5, 6, 7], dtype=np.int64),
    )


def synthetic_benchmark(root: Path) -> Path:
    benchmark = root / "benchmark"
    benchmark.mkdir()
    (benchmark / "manifest.csv.gz").write_bytes(b"frozen manifest")
    (benchmark / "report.json").write_text('{"formal": true}', encoding="utf-8")
    for name in PANELS:
        synthetic_panel(benchmark / f"panel_{name}.npz")
    return benchmark


def test_cache_and_evaluation_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark = synthetic_benchmark(root)
        baseline = np.asarray([0.80, 0.90, 0.10, 0.80, 0.70, 0.20], dtype=np.float32)
        candidate = np.asarray([0.95, 0.90, 0.10, 0.70, 0.80, 0.20], dtype=np.float32)
        baseline_dir = root / "baseline"
        candidate_dir = root / "candidate"
        write_pair_score_cache(
            baseline_dir, benchmark,
            {"name": "baseline", "input_contract": "spectrum_pair"},
            {name: baseline for name in PANELS},
        )
        write_pair_score_cache(
            candidate_dir, benchmark,
            {"name": "candidate", "input_contract": "spectrum_pair"},
            {name: candidate for name in PANELS},
        )
        _, loaded = load_pair_score_cache(candidate_dir, benchmark)
        assert np.array_equal(loaded["identity_disjoint"], candidate)

        graph = SimpleNamespace(
            pair_candidate_row=np.arange(6),
            molecule_ptr=np.arange(7),
        )
        scores = evaluator.graph_scores(graph, candidate)
        assert np.array_equal(scores.pair, candidate)
        assert np.array_equal(scores.molecule, candidate)

        previous = evaluator.arguments
        output = root / "evaluation"
        evaluator.arguments = lambda: SimpleNamespace(
            benchmark=benchmark,
            baseline_score_cache=baseline_dir,
            candidate_score_cache=candidate_dir,
            output=output,
            bootstrap_resamples=100,
            bootstrap_seed=7,
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.main()
        finally:
            evaluator.arguments = previous
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        for panel in report["panels"].values():
            assert panel["paired"]["corrected"] == 1
            assert panel["paired"]["introduced"] == 1
            assert panel["paired"]["risk_net_lambda2"] == -1
            assert panel["baseline"]["retrieval"]["recall@1"] == 0.5
            assert panel["candidate"]["retrieval"]["recall@1"] == 0.5


def test_provenance_and_score_tampering_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark = synthetic_benchmark(root)
        values = np.linspace(0, 1, 6, dtype=np.float32)
        cache = root / "cache"
        write_pair_score_cache(
            cache, benchmark, {"name": "test"}, {name: values for name in PANELS},
        )
        score_path = cache / "pair_scores_identity_disjoint.npy"
        original = score_path.read_bytes()
        score_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        try:
            load_pair_score_cache(cache, benchmark)
        except RuntimeError as error:
            assert "checksum mismatch" in str(error)
        else:
            raise AssertionError("tampered scores were accepted")

        score_path.write_bytes(original)
        (benchmark / "report.json").write_text('{"formal": false}', encoding="utf-8")
        try:
            load_pair_score_cache(cache, benchmark)
        except RuntimeError as error:
            assert "provenance mismatch" in str(error)
        else:
            raise AssertionError("tampered benchmark was accepted")


def main() -> None:
    test_cache_and_evaluation_contract()
    test_provenance_and_score_tampering_fail_closed()
    print("PASS: GNPS model-agnostic pair-score benchmark contracts")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""End-to-end smoke test for the BioAware unified benchmark CLI."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="bioaware_benchmark_test_") as raw:
        root = Path(raw)
        candidates = pd.DataFrame(
            [
                ("q1", "a", 1, "f1", "s1", "negative", 1),
                ("q1", "b", 0, "f1", "s1", "negative", 1),
                ("q2", "c", 1, "f2", "s2", "positive", 0),
                ("q2", "d", 0, "f2", "s2", "positive", 0),
            ],
            columns=[
                "query_id", "candidate_id", "is_truth", "formula_cluster",
                "source", "polarity", "near_query",
            ],
        )
        baseline = pd.DataFrame(
            [("q1", "a", 0.4), ("q1", "b", 0.5), ("q2", "c", 0.9), ("q2", "d", 0.1)],
            columns=["query_id", "candidate_id", "score"],
        )
        contender = pd.DataFrame(
            [("q1", "a", 0.6), ("q1", "b", 0.5), ("q2", "c", 0.9), ("q2", "d", 0.1)],
            columns=["query_id", "candidate_id", "score"],
        )
        candidate_path = root / "candidates.csv.gz"
        baseline_path = root / "baseline.csv.gz"
        contender_path = root / "contender.csv.gz"
        output = root / "result"
        candidates.to_csv(candidate_path, index=False)
        baseline.to_csv(baseline_path, index=False)
        contender.to_csv(contender_path, index=False)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tasks/evaluate_bioaware_unified_benchmark.py"),
                "--candidate-manifest", str(candidate_path),
                "--method", f"official_dreams={baseline_path}",
                "--method", f"bioaware={contender_path}",
                "--baseline-method", "official_dreams",
                "--benchmark-track", "sample_context",
                "--protocol-scope", "opened_development",
                "--expected-queries", "2",
                "--expected-candidate-rows", "4",
                "--bootstrap-resamples", "100",
                "--output-dir", str(output),
            ],
            check=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tasks/validate_bioaware_unified_benchmark.py"),
                "--output-dir", str(output),
            ],
            check=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        comparison = report["comparisons"]["bioaware_vs_official_dreams"]
        assert comparison["corrected"] == 1
        assert comparison["introduced"] == 0
        assert comparison["delta_recall_at_1"] == 0.5
    print("[test_evaluate_bioaware_unified_benchmark] PASS", flush=True)


if __name__ == "__main__":
    main()


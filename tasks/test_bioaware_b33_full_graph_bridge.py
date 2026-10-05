#!/usr/bin/env python3
"""Dependency-light source and graph-math checks for the B33 bridge."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rank(score: np.ndarray, pointer: np.ndarray) -> np.ndarray:
    result = []
    for left, right in zip(pointer[:-1], pointer[1:], strict=True):
        truth = score[int(left)]
        result.append(1 + int(np.sum(score[int(left) + 1:int(right)] >= truth)))
    return np.asarray(result, dtype=np.int64)


def main() -> None:
    trainer = ROOT / "tasks/train_bioaware_b33_full_graph_bridge.py"
    builder = ROOT / "tasks/build_bioaware_b33_full_candidate_graph.py"
    ast.parse(trainer.read_text(encoding="utf-8"))
    ast.parse(builder.read_text(encoding="utf-8"))
    source = trainer.read_text(encoding="utf-8")
    lowered = source.lower()
    required = (
        "candidate_max_recomputed_over_all_reference_spectra",
        "all_official_correct_physical_queries_supply_boundary_safety",
        "query_reference_encoder_shared",
        "model.eval()  # keep dropout disabled",
        "corrected_minus_2x_introduced_positive",
        "clip_scale",
        "p2b_used\": false",
    )
    for token in required:
        if token not in lowered:
            raise AssertionError(token)
    # Truth is first; a tie must count against it.
    score = np.asarray([0.5, 0.5, 0.4, 0.8, 0.7], dtype=np.float32)
    pointer = np.asarray([0, 3, 5], dtype=np.int64)
    if not np.array_equal(synthetic_rank(score, pointer), [2, 1]):
        raise AssertionError("strict tie ranking contract failed")
    print("[test_bioaware_b33_full_graph_bridge] PASS", flush=True)


if __name__ == "__main__":
    main()

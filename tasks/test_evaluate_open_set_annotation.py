from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    panel = tmp_path / "panel.npz"
    # Four queries, three molecules/query, one spectrum/molecule. The first
    # and third have an in-library truth; the other two are true no-match.
    np.savez_compressed(
        panel,
        query_row=np.arange(4),
        query_ik14=np.asarray(["A", "B", "C", "D"]),
        query_formula=np.asarray(["F1", "F2", "F3", "F4"]),
        query_has_match=np.asarray([1, 0, 1, 0], bool),
        query_ptr=np.asarray([0, 3, 6, 9, 12]),
        molecule_ptr=np.arange(13),
        molecule_label=np.asarray([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        molecule_ik14=np.asarray(list("abcdefghijkl")),
        molecule_formula=np.asarray(["F"] * 12),
        molecule_same_formula=np.zeros(12, bool),
        candidate_row=np.arange(12),
    )
    scores = tmp_path / "scores.npz"
    np.savez_compressed(
        scores,
        method_names=np.asarray(["unified", "wse"]),
        scores=np.asarray([
            [0.9, 0.2, 0.1, 0.30, 0.29, 0.1, 0.8, 0.7, 0.1, 0.3, 0.29, 0.2],
            [0.8, 0.4, 0.1, 0.60, 0.20, 0.1, 0.6, 0.7, 0.1, 0.5, 0.1, 0.0],
        ]),
    )
    return panel, scores


def test_calibration_then_external_fixed_threshold_evaluation(tmp_path: Path) -> None:
    panel, scores = fixture(tmp_path)
    calibration = tmp_path / "calibration.json"
    external = tmp_path / "external.json"
    base = [
        sys.executable, str(ROOT / "tasks/evaluate_open_set_annotation.py"),
        "--panel", str(panel), "--scores", str(scores), "--target-fdr", "0.5",
    ]
    subprocess.run(base + ["--mode", "calibrate", "--output", str(calibration)], check=True)
    calibrated = json.loads(calibration.read_text(encoding="utf-8"))
    subprocess.run(
        base + ["--mode", "evaluate", "--calibration", str(calibration),
                "--output", str(external)],
        check=True,
    )
    report = json.loads(external.read_text(encoding="utf-8"))
    assert report["status"] == "OPEN_SET_ANNOTATION_EXTERNAL_EVALUATION_COMPLETE"
    assert report["calibration_sha256"] is not None
    for method in ("unified", "wse"):
        assert report["methods"][method]["threshold"] == calibrated["methods"][method]["threshold"]
        assert "annotation_yield_all_queries" in report["methods"][method]

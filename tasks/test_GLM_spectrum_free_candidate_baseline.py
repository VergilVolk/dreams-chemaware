"""Contracts for GLM_spectrum_free_candidate_baseline.

Synthetic graph panels verify: reference-count ranking recall@1, the
truth-blind IK14 tiebreak, the uniform-random expectation, scattered manifest
panel gathering (no foreign molecules), and fail-closed structure checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import GLM_spectrum_free_candidate_baseline as baseline  # noqa: E402


def write_graph_panel(
    path: Path,
    groups: list[list[tuple[int, str, int]]],
) -> None:
    """groups: per query, per molecule: (label, ik14, reference_count)."""
    query_ptr = [0]
    molecule_ptr = [0]
    labels: list[int] = []
    ik14s: list[str] = []
    rows: list[int] = []
    for group in groups:
        for label, ik14, count in group:
            labels.append(label)
            ik14s.append(ik14)
            molecule_ptr.append(molecule_ptr[-1] + count)
            rows.extend(range(molecule_ptr[-2], molecule_ptr[-1]))
        query_ptr.append(len(labels))
    np.savez_compressed(
        path,
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
        molecule_label=np.asarray(labels, dtype=np.int8),
        molecule_ik14=np.asarray(ik14s, dtype="U6"),
        candidate_row=np.asarray(rows, dtype=np.int64),
    )


def test_graph_panel_ranking_and_tiebreak(tmp_path: Path) -> None:
    panel = tmp_path / "panel_graph.npz"
    write_graph_panel(panel, [
        # Truth wins on reference count 5 vs 2.
        [(1, "TRUTH", 5), (0, "NEGAT", 2)],
        # Truth loses: 1 vs 3.
        [(1, "TRUTH", 1), (0, "NEGAT", 3)],
        # Tie on 2 vs 2: IK14 lexicographic is truth-blind; here it favours
        # the truth, elsewhere it may not -- only determinism is guaranteed.
        [(1, "AAAA", 2), (0, "ZZZZ", 2)],
    ])
    graph = baseline.graph_arrays(panel)
    metrics = baseline.spectrum_free_metrics(graph)
    assert metrics["queries"] == 3
    assert metrics["spectrum_free_recall1"] == pytest.approx(2.0 / 3.0)
    assert metrics["random_ranking_expected_recall1"] == pytest.approx(0.5)
    assert metrics["candidates_per_query"] == {
        "min": 2, "median": 2.0, "max": 2,
    }
    assert metrics["truth_reference_count_mean"] == pytest.approx((5 + 1 + 2) / 3)
    assert metrics["negative_reference_count_mean"] == pytest.approx((2 + 3 + 2) / 3)
    assert metrics["ranking_rule"] == baseline.RANKING_RULE


def test_manifest_panel_gathers_scattered_queries(tmp_path: Path) -> None:
    # Manifest with three queries (2/3/2 molecules); the panel takes the
    # FIRST and THIRD only.  The middle query's molecules must not leak in.
    query_ptr = np.asarray([0, 2, 5, 7], dtype=np.int64)
    molecule_ptr = np.asarray([0, 2, 4, 7, 10, 13, 15, 17], dtype=np.int64)
    manifest = tmp_path / "manifest.npz"
    np.savez_compressed(
        manifest,
        query_row=np.asarray([100, 200, 300], dtype=np.int64),
        query_ptr=query_ptr,
        molecule_ptr=molecule_ptr,
        molecule_label=np.asarray([1, 0, 1, 0, 0, 1, 0], dtype=np.int8),
        molecule_ik14=np.asarray(
            ["QT0", "QN0", "QT1", "QN1", "QN2", "QT2", "QN2"], dtype="U6",
        ),
        pair_candidate_row=np.arange(17, dtype=np.int64),
    )
    role_panel = tmp_path / "role_panel.npz"
    np.savez_compressed(
        role_panel,
        query=np.asarray([0, 2], dtype=np.int64),
        formula=np.asarray(["F0", "F2"], dtype=str),
    )
    with np.load(manifest, allow_pickle=False) as loaded:
        manifest_arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(role_panel, allow_pickle=False) as loaded:
        queries = np.asarray(loaded["query"], dtype=np.int64)
    view = baseline.manifest_view(manifest_arrays, queries)
    assert view["query_ptr"].tolist() == [0, 2, 4]
    assert view["molecule_label"].tolist() == [1, 0, 1, 0]
    assert view["molecule_ik14"].tolist() == ["QT0", "QN0", "QT2", "QN2"]
    # Re-indexed molecule boundaries: 2+2+2+2 references total.
    assert view["molecule_ptr"].tolist() == [0, 2, 4, 6, 8]
    assert view["candidate_row"].tolist() == [0, 1, 2, 3, 13, 14, 15, 16]
    metrics = baseline.spectrum_free_metrics(view)
    assert metrics["queries"] == 2


def test_cli_end_to_end(tmp_path: Path) -> None:
    graph = tmp_path / "panel_graph.npz"
    write_graph_panel(graph, [
        [(1, "TRUTH", 4), (0, "NEGAT", 1)],
        [(1, "TRUTH", 1), (0, "NEGAT", 4)],
    ])
    output = tmp_path / "baseline" / "report.json"
    import subprocess
    result = subprocess.run(
        [
            sys.executable, str(
                Path(__file__).resolve().parent
                / "GLM_spectrum_free_candidate_baseline.py",
            ),
            "--graph-panel", f"identity_disjoint={graph}",
            "--output", str(output),
        ],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == baseline.BASELINE_STATUS
    assert report["panels"]["identity_disjoint"][
        "spectrum_free_recall1"
    ] == pytest.approx(0.5)
    assert "not a release gate" in report["interpretation"]
    del result


def test_structure_failures(tmp_path: Path) -> None:
    panel = tmp_path / "broken.npz"
    write_graph_panel(panel, [[(0, "ONLY", 2), (0, "NEGS", 1)]])
    with pytest.raises(RuntimeError, match="exactly one positive"):
        baseline.spectrum_free_metrics(baseline.graph_arrays(panel))
    empty = tmp_path / "empty.npz"
    np.savez_compressed(empty, query_ptr=np.asarray([0], dtype=np.int64))
    with pytest.raises(RuntimeError, match="lacks arrays"):
        baseline.graph_arrays(empty)


def test_refuses_overwrite_and_requires_panel(tmp_path: Path) -> None:
    import subprocess
    script = str(
        Path(__file__).resolve().parent / "GLM_spectrum_free_candidate_baseline.py",
    )
    output = tmp_path / "report.json"
    output.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, script, "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0  # refuses to overwrite AND lacks any panel
    result_no_panel = subprocess.run(
        [sys.executable, script, "--output", str(tmp_path / "other.json")],
        capture_output=True, text=True, check=False,
    )
    assert result_no_panel.returncode != 0
    assert "at least one panel" in result_no_panel.stderr

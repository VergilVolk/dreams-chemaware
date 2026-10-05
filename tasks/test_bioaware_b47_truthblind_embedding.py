#!/usr/bin/env python
"""Dependency-light checks for the B47 embedding I/O and execution contract."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile

import numpy as np

from bioaware_b47_truthblind_io import parse_mgf_records


ROOT = Path(__file__).resolve().parents[1]


def test_parser() -> None:
    content = """BEGIN IONS
TITLE=q2
PEPMASS=123.4 999
10 2
20 3
END IONS

BEGIN IONS
TITLE=q1
PEPMASS=456.7
30 4 annotation
END IONS
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.mgf"
        path.write_text(content, encoding="utf-8")
        titles, records = parse_mgf_records(path)
    assert titles == ["q2", "q1"]
    assert [record["precursor_mz"] for record in records] == [123.4, 456.7]
    assert np.asarray(records[0]["peaks"]).shape == (2, 2)


def test_source_contract() -> None:
    source_path = ROOT / "tasks/encode_bioaware_b47_truthblind_embeddings.py"
    source = source_path.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "candidate_scores_computed\": False",
        "seed_selection_performed\": False",
        "truth_opened\": False",
        "P2b_used\": False",
        "set(mgf_titles) != set(query_ids)",
        "refusing to overwrite output",
        "open_memmap",
        "candidate-reference count does not replay frozen graph report",
        "outside frozen HDF5 universe",
    ):
        assert required in source, required
    forbidden = ("truth_candidate", "corrected", "introduced", "recall1")
    lowered = source.casefold()
    for token in forbidden:
        assert token not in lowered, token


if __name__ == "__main__":
    test_parser()
    test_source_contract()
    print("[test_bioaware_b47_truthblind_embedding] PASS")

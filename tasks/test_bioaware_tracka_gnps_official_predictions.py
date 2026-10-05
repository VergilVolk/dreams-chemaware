#!/usr/bin/env python
"""Unit checks for official-DreaMS GNPS prediction construction."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_bioaware_tracka_gnps_official_predictions import build_frames  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        work = Path(raw_dir)
        panel_path = work / "panel.npz"
        embeddings_path = work / "embeddings.npz"

        # Two queries.  Query 0 has the truth plus one negative with two
        # reference spectra; query 1 has the truth plus one negative.
        np.savez(
            panel_path,
            query_row=np.array([10, 20], dtype=np.int64),
            query_ik14=np.array(["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]),
            query_formula=np.array(["C1H1", "C2H2"]),
            query_precursor_mz=np.array([100.0, 200.0]),
            query_ptr=np.array([0, 2, 4], dtype=np.int64),
            molecule_ik14=np.array(["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC",
                                    "BBBBBBBBBBBBBB", "DDDDDDDDDDDDDD"]),
            molecule_formula=np.array(["C1H1", "C1H2", "C2H2", "C2H3"]),
            molecule_label=np.array([True, False, True, False]),
            molecule_same_formula=np.array([True, True, True, True]),
            molecule_ptr=np.array([0, 1, 3, 4, 5], dtype=np.int64),
            candidate_row=np.array([11, 12, 13, 21, 22], dtype=np.int64),
            independent_positive=np.array([True, True]),
            near_query=np.array([False, True]),
        )

        rng = np.random.default_rng(3)
        universe = rng.normal(size=(30, 8)).astype(np.float32)
        # Truth of query 0 shares a direction with the query; the negative's
        # best reference outranks its weaker second reference.
        universe[10] = np.asarray([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        universe[11] = np.asarray([0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        universe[12] = np.asarray([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        universe[13] = np.asarray([0, 0, 1.0, 0, 0, 0, 0, 0], dtype=np.float32)
        universe[20] = np.asarray([0, 0, 0, 1.0, 0, 0, 0, 0], dtype=np.float32)
        universe[21] = np.asarray([0, 0, 0, 0, 1.0, 0, 0, 0], dtype=np.float32)
        universe[22] = np.asarray([0, 0, 0, 0.7, 0.7, 0, 0, 0], dtype=np.float32)
        rows = np.array([10, 11, 12, 13, 20, 21, 22], dtype=np.int64)
        np.savez(embeddings_path, rows=rows, embeddings=universe[rows])

        manifest, predictions, summary = build_frames(
            panel_path, embeddings_path, "unit_test"
        )
        assert summary["queries"] == 2
        assert summary["candidate_rows"] == 4

        scores = predictions.set_index(["query_id", "candidate_id"])["score"]
        # Molecule max: negative CCC keeps its best reference (row 12, cosine
        # 0.9/sqrt(0.82) against the query), not the weaker row 11.
        expected_ccc = 0.9 / np.sqrt(0.9 ** 2 + 0.1 ** 2)
        assert np.isclose(float(scores[("0", "CCCCCCCCCCCCCC")]), expected_ccc)
        assert np.isclose(float(scores[("0", "AAAAAAAAAAAAAA")]), 0.0)
        # Query 1: truth orthogonal exact 0; negative dominates via row 22
        # whose cosine is 0.7/sqrt(0.7^2+0.7^2).
        assert np.isclose(float(scores[("1", "BBBBBBBBBBBBBB")]), 0.0)
        assert np.isclose(
            float(scores[("1", "DDDDDDDDDDDDDD")]), 0.7 / np.sqrt(0.98)
        )

        manifest_row = manifest.set_index(["query_id", "candidate_id"])
        assert int(manifest_row.loc[("0", "AAAAAAAAAAAAAA"), "is_truth"]) == 1
        assert bool(manifest_row.loc[("1", "DDDDDDDDDDDDDD"), "near_query"])
        assert manifest_row.loc[("1", "DDDDDDDDDDDDDD"), "formula_cluster"] == "C2H2"

        # A panel referencing a missing embedding row must fail closed.
        np.savez(embeddings_path, rows=np.array([10, 11], dtype=np.int64),
                 embeddings=universe[[10, 11]])
        try:
            build_frames(panel_path, embeddings_path, "unit_test")
        except RuntimeError as error:
            assert "misses manifest row" in str(error)
        else:
            raise AssertionError("missing embedding row did not fail closed")

    print("[test_bioaware_tracka_gnps_official_predictions] PASS", flush=True)


if __name__ == "__main__":
    main()

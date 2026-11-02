#!/usr/bin/env python
"""Executable scientific contracts for the multi-source identity assembler."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from build_mega_identity_benchmark import MANIFEST_SCHEMA, assemble
from evaluate_mega_identity_benchmark import NpzMemmapStore, load_and_validate_panel


def write_panel(
    path: Path,
    query_identity: str,
    negative_identity: str,
    formula: str,
    query_row: int = 0,
    labels: tuple[int, int] = (1, 0),
    declared_n_p: int | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "query_ptr": np.asarray([0, 2], dtype=np.int64),
        "molecule_ptr": np.asarray([0, 2, 3], dtype=np.int64),
        "candidate_row": np.asarray([1, 2, 3], dtype=np.int64),
        "molecule_label": np.asarray(labels, dtype=bool),
        "query_ik14": np.asarray([query_identity], dtype="U14"),
        "molecule_ik14": np.asarray([query_identity, negative_identity], dtype="U14"),
        "query_formula": np.asarray([formula], dtype="U32"),
        "molecule_formula": np.asarray([formula, formula], dtype="U32"),
        "near_query": np.asarray([1], dtype=bool),
        "query_row": np.asarray([query_row], dtype=np.int64),
    }
    if declared_n_p is not None:
        payload["N_P"] = np.asarray(declared_n_p, dtype=np.int64)
    np.savez_compressed(path, **payload)


def write_manifest(path: Path, sources: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({
        "schema": MANIFEST_SCHEMA,
        "dataset_id": "synthetic_multisource_v1",
        "sources": sources,
    }), encoding="utf-8")


def source(source_id: str, panel: str, namespace_id: str, size: int = 10) -> dict[str, object]:
    return {
        "source_id": source_id,
        "panel": panel,
        "spectrum_row_namespace": {"id": namespace_id, "size": size},
    }


def expect_failure(manifest: Path, output: Path, fragment: str) -> None:
    try:
        assemble(manifest, output)
    except (RuntimeError, FileNotFoundError) as error:
        if fragment.lower() not in str(error).lower():
            raise AssertionError(f"expected {fragment!r}, observed {error!r}") from error
    else:
        raise AssertionError(f"expected failure containing {fragment!r}")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first.npz"
        second = root / "second.npz"
        write_panel(first, "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "C2H6O")
        write_panel(second, "CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD", "C2H6O")
        manifest = root / "manifest.json"
        write_manifest(manifest, [
            source("first", first.name, "first_rows"),
            source("second", second.name, "second_rows"),
        ])
        output = root / "assembled"
        report = assemble(manifest, output)
        assert report["counts"] == {
            "N_Q_labelled_queries": 2,
            "N_I_query_identity_clusters": 2,
            "N_F_query_formula_clusters": 1,
            "N_E_query_candidate_molecule_edges": 4,
            "N_P_query_reference_spectrum_comparisons": 6,
        }
        assert "not labelled queries" in report["unit_definitions"]["N_P"]

        work = root / "validate"
        work.mkdir()
        store = NpzMemmapStore(output / "panel.npz", work, "panel")
        try:
            panel = load_and_validate_panel(store)
            assert panel.n_queries == 2
            assert panel.n_molecule_edges == 4
            assert panel.n_pair_edges == 6
            assert np.array_equal(store.array("query_source"), [0, 1])
            assert np.array_equal(store.array("molecule_source"), [0, 0, 1, 1])
            assert np.array_equal(store.array("reference_source"), [0, 0, 0, 1, 1, 1])
            assert np.array_equal(store.array("query_row"), [0, 10])
            assert np.array_equal(store.array("candidate_row"), [1, 2, 3, 11, 12, 13])
            assert len(np.unique(np.concatenate((
                store.array("query_row"), store.array("candidate_row")
            )))) == 8
            assert int(store.scalar("N_Q")) == 2
            assert int(store.scalar("N_P")) == 6
        finally:
            store.close()
            shutil.rmtree(work)

        duplicate_manifest = root / "duplicate.json"
        write_manifest(duplicate_manifest, [
            source("same", first.name, "rows_a"),
            source("same", second.name, "rows_b"),
        ])
        expect_failure(duplicate_manifest, root / "duplicate_out", "duplicate source_id")

        bad_axis = root / "bad_axis.npz"
        write_panel(
            bad_axis, "EEEEEEEEEEEEEE", "FFFFFFFFFFFFFF", "C3H8O", declared_n_p=99
        )
        bad_axis_manifest = root / "bad_axis.json"
        write_manifest(bad_axis_manifest, [source("bad_axis", bad_axis.name, "bad_rows")])
        expect_failure(bad_axis_manifest, root / "bad_axis_out", "N_P metadata")

        bad_truth = root / "bad_truth.npz"
        write_panel(
            bad_truth, "GGGGGGGGGGGGGG", "HHHHHHHHHHHHHH", "C4H10", labels=(0, 1)
        )
        bad_truth_manifest = root / "bad_truth.json"
        write_manifest(
            bad_truth_manifest, [source("bad_truth", bad_truth.name, "truth_rows")]
        )
        expect_failure(bad_truth_manifest, root / "bad_truth_out", "labels must exactly equal")

        empty = root / "empty.npz"
        np.savez_compressed(
            empty,
            query_ptr=np.asarray([0, 0], dtype=np.int64),
            molecule_ptr=np.asarray([0], dtype=np.int64),
            candidate_row=np.asarray([], dtype=np.int64),
            molecule_label=np.asarray([], dtype=bool),
            query_ik14=np.asarray(["IIIIIIIIIIIIII"], dtype="U14"),
            molecule_ik14=np.asarray([], dtype="U14"),
            query_formula=np.asarray(["C5H12"], dtype="U16"),
            molecule_formula=np.asarray([], dtype="U16"),
            near_query=np.asarray([0], dtype=bool),
            query_row=np.asarray([0], dtype=np.int64),
        )
        empty_manifest = root / "empty.json"
        write_manifest(empty_manifest, [source("empty", empty.name, "empty_rows")])
        expect_failure(empty_manifest, root / "empty_out", "every query needs")

        narrow_manifest = root / "narrow.json"
        write_manifest(narrow_manifest, [source("narrow", first.name, "narrow_rows", size=3)])
        expect_failure(narrow_manifest, root / "narrow_out", "namespace size")

    print("PASS: mega multi-source identity benchmark assembler contracts")


if __name__ == "__main__":
    main()

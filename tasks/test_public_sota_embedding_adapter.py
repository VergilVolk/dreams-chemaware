from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from public_sota_embedding_adapter import (  # noqa: E402
    _lfs_pointer,
    _msbert_batches,
    _specembedding_batches,
    load_embedding_cache,
    main,
    score_molecules,
)
from evaluate_mega_identity_benchmark import (  # noqa: E402
    NpzMemmapStore,
    load_and_validate_panel,
    sha256_file,
    validate_score_contract,
)


def synthetic_records():
    yield 0, 100.0, np.asarray([[10.0, 20.0], [1.0, 4.0]], np.float32)
    yield 1, 101.0, np.asarray([[11.0, 21.0], [3.0, 2.0]], np.float32)


def test_official_preprocessing_shapes() -> None:
    s_rows, mz, intensity, mask = next(_specembedding_batches(synthetic_records(), 2))
    assert s_rows.tolist() == [0, 1]
    assert mz.shape == intensity.shape == mask.shape == (2, 100)
    assert mz[0, 0] == 100.0 and intensity[0, 0] == 2.0
    assert np.isclose(intensity[0, 2], 1.0)
    assert mask[:, :3].sum() == 0 and mask[:, 3:].all()

    m_rows, token, m_intensity = next(_msbert_batches(synthetic_records(), 2))
    assert m_rows.tolist() == [0, 1]
    assert token.shape == (2, 100) and m_intensity.shape == (2, 1, 100)
    assert token[0, 0] == 10002  # precursor 100.00 + PAD/MASK offset
    assert np.isclose(m_intensity[0, 0, 0], 0.5)  # official precursor weight 2 / peak max 4


def test_lfs_pointer_is_not_a_checkpoint(tmp_path: Path) -> None:
    pointer = tmp_path / "model.pkl"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\nsize 123\n",
        encoding="ascii",
    )
    assert _lfs_pointer(pointer) == {"pointer": True, "sha256": "a" * 64, "bytes": 123}


def test_standard_molecule_score_contract(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        rows=np.arange(6, dtype=np.int64),
        embeddings=np.asarray(
            [[1, 0], [0, 1], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
            np.float32,
        ),
    )
    rows, embeddings = load_embedding_cache(embeddings_path)
    panel = {
        "query_row": np.asarray([0, 1]),
        "query_ptr": np.asarray([0, 2, 4]),
        "molecule_ptr": np.asarray([0, 2, 3, 4, 6]),
        "candidate_row": np.asarray([2, 3, 4, 2, 4, 5]),
        "molecule_label": np.asarray([1, 0, 1, 0], dtype=bool),
        "query_ik14": np.asarray(["AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD"]),
        "molecule_ik14": np.asarray([
            "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "DDDDDDDDDDDDDD", "EEEEEEEEEEEEEE",
        ]),
        "query_formula": np.asarray(["C2H6O", "C3H8"]),
        "molecule_formula": np.asarray(["C2H6O", "C2H6O", "C3H8", "C4H10"]),
        "near_query": np.asarray([1, 0], dtype=bool),
    }
    scores = score_molecules(panel, rows, embeddings)
    assert scores.shape == (4,)
    assert scores[0] > scores[1]
    assert scores[3] > scores[2]

    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(panel_path, **panel)
    output = tmp_path / "scores.npz"
    report = tmp_path / "report.json"
    main([
        "score", "--panel", str(panel_path), "--embeddings", str(embeddings_path),
        "--method-name", "synthetic", "--output", str(output), "--report", str(report),
    ])
    with np.load(output, allow_pickle=False) as body:
        assert body["method_names"].tolist() == ["synthetic"]
        assert body["molecule_scores"].shape == (1, 4)
        assert int(body["N_Q"]) == 2
        assert int(body["N_E"]) == 4
        assert int(body["N_P"]) == 6
        assert len(str(body["panel_sha256"])) == 64
    assert json.loads(report.read_text(encoding="utf-8"))["candidate_molecules"] == 4

    extraction = tmp_path / "formal_contract_extract"
    extraction.mkdir()
    panel_store = NpzMemmapStore(panel_path, extraction, "panel")
    score_store = NpzMemmapStore(output, extraction, "scores")
    try:
        formal_panel = load_and_validate_panel(panel_store)
        formal_provenance = validate_score_contract(
            score_store, formal_panel, sha256_file(panel_path),
        )
        assert formal_provenance["N_Q"] == 2
        assert formal_provenance["N_E"] == 4
        assert formal_provenance["N_P"] == 6
    finally:
        panel_store.close()
        score_store.close()
        shutil.rmtree(extraction, ignore_errors=True)

    merged = tmp_path / "merged.npz"
    second = tmp_path / "scores_second.npz"
    with np.load(output, allow_pickle=False) as body:
        provenance = {
            key: np.asarray(body[key])
            for key in ("panel_sha256", "N_Q", "N_E", "N_P")
        }
    np.savez_compressed(
        second,
        **provenance,
        method_names=np.asarray(["second"]),
        molecule_scores=np.asarray([[0.1, 0.2, 0.3, 0.4]], np.float32),
    )
    main([
        "merge", "--score-cache", str(output), "--score-cache", str(second),
        "--output", str(merged),
    ])
    with np.load(merged, allow_pickle=False) as body:
        assert body["method_names"].tolist() == ["synthetic", "second"]
        assert body["molecule_scores"].shape == (2, 4)
        assert str(body["panel_sha256"]) == str(provenance["panel_sha256"])
        assert [int(body[key]) for key in ("N_Q", "N_E", "N_P")] == [2, 4, 6]

    merge_extract = tmp_path / "merged_contract_extract"
    merge_extract.mkdir()
    panel_store = NpzMemmapStore(panel_path, merge_extract, "panel")
    score_store = NpzMemmapStore(merged, merge_extract, "scores")
    try:
        formal_panel = load_and_validate_panel(panel_store)
        assert validate_score_contract(
            score_store, formal_panel, sha256_file(panel_path),
        )["N_E"] == 4
    finally:
        panel_store.close()
        score_store.close()
        shutil.rmtree(merge_extract, ignore_errors=True)

    wrong_axis = tmp_path / "wrong_axis.npz"
    np.savez_compressed(
        wrong_axis,
        panel_sha256=np.asarray("0" * 64),
        N_Q=np.asarray(2), N_E=np.asarray(4), N_P=np.asarray(6),
        method_names=np.asarray(["wrong"]),
        molecule_scores=np.zeros((1, 4), np.float32),
    )
    try:
        main([
            "merge", "--score-cache", str(output), "--score-cache", str(wrong_axis),
            "--output", str(tmp_path / "must_not_exist.npz"),
        ])
    except RuntimeError as error:
        assert "provenance/count mismatch" in str(error)
    else:
        raise AssertionError("merge accepted a score cache from a different panel axis")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_official_preprocessing_shapes()
        test_lfs_pointer_is_not_a_checkpoint(tmp)
        test_standard_molecule_score_contract(tmp)
    print("PASS: public SOTA embedding adapter contracts")

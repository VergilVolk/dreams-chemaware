from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from evaluate_mega_identity_benchmark import (  # noqa: E402
    NpzMemmapStore,
    load_and_validate_panel,
    sha256_file,
    validate_score_contract,
)
from transexion_same_axis_adapter import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_REPO,
    RefinedSpectrum,
    aggregate_molecules,
    inspect_asset,
    load_model,
    load_official_modules,
    load_selected_spectra,
    official_pair_features,
    panel_rows,
    refine_record,
    score_pairs,
    validate_panel,
    write_score_cache,
)


def synthetic_panel() -> dict[str, np.ndarray]:
    return {
        "query_row": np.asarray([0, 4], np.int64),
        "query_ptr": np.asarray([0, 2, 4], np.int64),
        "molecule_ptr": np.asarray([0, 2, 3, 4, 6], np.int64),
        "candidate_row": np.asarray([1, 2, 3, 1, 2, 3], np.int64),
        "molecule_label": np.asarray([1, 0, 1, 0], bool),
        "query_ik14": np.asarray(["AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD"]),
        "molecule_ik14": np.asarray([
            "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "DDDDDDDDDDDDDD", "EEEEEEEEEEEEEE",
        ]),
        "query_formula": np.asarray(["C2H6O", "C3H8"]),
        "molecule_formula": np.asarray(["C2H6O", "C2H6O", "C3H8", "C4H10"]),
        "near_query": np.asarray([1, 0], bool),
    }


def test_axis_and_max_aggregation() -> None:
    panel = synthetic_panel()
    validate_panel(panel)
    assert panel_rows(panel).tolist() == [0, 1, 2, 3, 4]
    pair = np.asarray([0.1, 0.7, 0.4, -0.2, 0.3, 0.2], np.float64)
    molecule = aggregate_molecules(pair, panel["molecule_ptr"])
    assert np.allclose(molecule, [0.7, 0.4, -0.2, 0.3])


def test_released_preprocessing_and_query_query_loss() -> None:
    modules = load_official_modules(DEFAULT_REPO)
    raw = np.asarray([
        [5.0, 50.0, 50.5, 80.0, 1001.0],
        [10.0, 100.0, 10.0, 50.0, 100.0],
    ])
    query = refine_record(modules, 0, 101.0, raw)
    assert np.all((query.fragment_mz >= 10) & (query.fragment_mz <= 1000))
    assert len(query.fragment_mz) < raw.shape[1]
    first_reference = RefinedSpectrum(
        1, 101.0, np.asarray([49.9, 79.9]), np.asarray([51.0]),
    )
    second_reference = RefinedSpectrum(
        2, 101.0, np.asarray([25.0, 125.0]), np.asarray([1.0, 99.0]),
    )
    first = official_pair_features(modules, query, first_reference)
    second = official_pair_features(modules, query, second_reference)
    # Fragment branch changes with reference; released neutral-loss branch does not.
    assert not np.array_equal(first[0].numpy(), second[0].numpy())
    assert np.array_equal(first[2].numpy(), second[2].numpy())


def test_real_checkpoint_strict_load_and_forward(tmp_path: Path) -> None:
    asset = inspect_asset(DEFAULT_REPO, DEFAULT_CHECKPOINT)
    assert asset["checkpoint_bytes"] == 8_403_857
    modules = load_official_modules(DEFAULT_REPO)
    model = load_model(modules, DEFAULT_CHECKPOINT, "cpu")
    query = RefinedSpectrum(
        0, 150.0, np.asarray([40.0, 75.0, 120.0]), np.asarray([30.0, 75.0]),
    )
    reference = RefinedSpectrum(
        1, 150.0, np.asarray([40.01, 80.0, 119.98]), np.asarray([30.02, 70.0]),
    )
    batch = modules.collate([official_pair_features(modules, query, reference)])
    # Deliberately no no_grad: released attention registers a gradient hook.
    with modules.torch.enable_grad():
        output = model(batch[0], batch[1], batch[2], batch[3])
    value = output.reshape(-1).detach().numpy()
    assert value.shape == (1,) and np.isfinite(value).all()

    mgf = tmp_path / "tiny.mgf"
    blocks = []
    for row in range(5):
        blocks.append(
            "BEGIN IONS\n"
            f"PEPMASS={150.0 + row * 0.001}\n"
            f"{40.0 + row * 0.01} 100\n"
            f"{75.0 + row * 0.01} 50\n"
            f"{120.0 + row * 0.01} 20\n"
            "END IONS\n"
        )
    mgf.write_text("".join(blocks), encoding="utf-8")
    panel = synthetic_panel()
    spectra = load_selected_spectra(modules, mgf, panel_rows(panel))
    pair_scores = score_pairs(modules, model, panel, spectra, "cpu", batch_size=2)
    assert pair_scores.shape == (6,) and np.isfinite(pair_scores).all()
    assert aggregate_molecules(pair_scores, panel["molecule_ptr"]).shape == (4,)


def test_formal_score_cache_contract(tmp_path: Path) -> None:
    panel = synthetic_panel()
    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(panel_path, **panel)
    output = tmp_path / "scores.npz"
    molecule = np.asarray([0.7, 0.4, -0.2, 0.3], np.float32)
    write_score_cache(output, panel_path, panel, "TransExION auxiliary", molecule)
    with np.load(output, allow_pickle=False) as body:
        assert str(body["schema_version"]) == "mega_identity_method_scores_v1"
        assert body["method_names"].tolist() == ["TransExION auxiliary"]
        assert body["molecule_scores"].shape == (1, 4)
        assert [int(body[key]) for key in ("N_Q", "N_E", "N_P")] == [2, 4, 6]

    extraction = tmp_path / "extract"
    extraction.mkdir()
    panel_store = NpzMemmapStore(panel_path, extraction, "panel")
    score_store = NpzMemmapStore(output, extraction, "scores")
    try:
        formal_panel = load_and_validate_panel(panel_store)
        provenance = validate_score_contract(score_store, formal_panel, sha256_file(panel_path))
        assert [provenance[key] for key in ("N_Q", "N_E", "N_P")] == [2, 4, 6]
    finally:
        panel_store.close()
        score_store.close()
        shutil.rmtree(extraction, ignore_errors=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_axis_and_max_aggregation()
        test_released_preprocessing_and_query_query_loss()
        test_real_checkpoint_strict_load_and_forward(tmp)
        test_formal_score_cache_contract(tmp)
    print(json.dumps({
        "status": "TRANSEXION_SAME_AXIS_ADAPTER_CONTRACT_PASS",
        "official_checkpoint_forward": True,
        "neutral_loss_query_query_retained": True,
        "no_no_grad": True,
        "mega_identity_method_scores_v1": True,
    }))

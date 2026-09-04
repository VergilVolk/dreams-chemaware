"""Small deterministic smoke tests; no DreaMS checkpoint or GPU required."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from noise_final_core import CandidateGraph, ZeroInitPeakAdapter, strict_rank


ROOT = Path(__file__).resolve().parent.parent


def build_fixture(directory: Path) -> tuple[Path, Path]:
    graph_path = directory / "graph.npz"
    data_path = directory / "data.hdf5"
    # Two queries; each has one positive and one negative molecule, one spectrum each.
    np.savez_compressed(
        graph_path,
        feature_names=np.asarray(["dreams_similarity"], dtype=object),
        features=np.asarray([[0.8], [0.2], [0.3], [0.4]], dtype=np.float32),
        pair_candidate_row=np.asarray([0, 2, 1, 2], dtype=np.int64),
        query_ptr=np.asarray([0, 2, 4], dtype=np.int64),
        molecule_ptr=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        molecule_label=np.asarray([1, 0, 1, 0], dtype=np.int8),
        molecule_ik14=np.asarray(["B" * 14, "C" * 14, "A" * 14, "C" * 14]),
        molecule_formula=np.asarray(["FB", "FC", "FA", "FC"]),
        molecule_mces_grade=np.asarray([-1, 0, -1, 1], dtype=np.int8),
        # Intentionally non-monotonic: this is the h5py regression case.
        query_row=np.asarray([1, 0], dtype=np.int64),
        query_ik14=np.asarray(["B" * 14, "A" * 14]),
        query_formula=np.asarray(["FB", "FA"]),
        query_has_near=np.asarray([True, False]),
    )
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("INCHIKEY", data=np.asarray([b"A" * 27, b"B" * 27, b"C" * 27]))
        handle.create_dataset("spectrum", data=np.zeros((3, 2, 128), dtype=np.float32))
        handle.create_dataset("precursor_mz", data=np.asarray([100, 101, 102], dtype=np.float32))
    return graph_path, data_path


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        graph_path, data_path = build_fixture(directory)
        graph = CandidateGraph(graph_path)
        assert graph.n_queries == 2
        assert strict_rank(graph.official_molecule_scores(0)) == 1
        assert strict_rank(graph.official_molecule_scores(1)) == 2
        output = directory / "d0"
        subprocess.run([
            sys.executable, str(ROOT / "tasks/build_noise_final_d0_manifest.py"),
            "--graph", str(graph_path), "--data", str(data_path),
            "--p3-dir", str(directory / "missing_p3"),
            "--c1-dir", str(directory / "missing_c1"),
            "--a4-dir", str(directory / "missing_a4"),
            "--output-dir", str(output), "--no-formal",
        ], check=True)
        with np.load(output / "manifest.npz") as body:
            assert body["baseline_rank"].tolist() == [1, 2]
            assert np.allclose(body["identity_weight"], 1.0)

    adapter = ZeroInitPeakAdapter(16, hidden_dim=8, delta_bound=0.1)
    official = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    tokens = torch.randn(4, 6, 16)
    mz = torch.rand(4, 6) * 500
    intensity = torch.rand(4, 6)
    mask = torch.ones(4, 6, dtype=torch.bool)
    adapted, delta, weights = adapter(official, tokens, mz, intensity, mask)
    assert torch.allclose(adapted, official, atol=1e-6)
    assert torch.count_nonzero(delta) == 0
    assert torch.allclose(weights.sum(dim=1), torch.ones(4))
    print("[test_noise_final_d0_d1] PASS")


if __name__ == "__main__":
    main()

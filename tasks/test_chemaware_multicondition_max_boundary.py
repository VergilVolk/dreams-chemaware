"""CPU contracts for multi-condition ChemAware native triplets and evaluation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from build_chemaware_max_boundary_native_triplets import DREAMS_NATIVE_REPLAY, PoolWriter
from build_chemaware_multicondition_max_boundary_triplets import (
    append_query_events,
    condition_signature,
    identity_equal_sampling_weights,
    pool_prefix_equal,
    pool_prefix_semantic_sha256,
    query_geometry,
    select_identity_anchors,
)
from evaluate_chemaware_full_role_native import (
    identity_equal_metrics,
    identity_equal_paired,
)


class FakeCache:
    def __init__(self, values: dict[int, list[float]]):
        self.values = {
            int(row): np.asarray(value, dtype=np.float32)
            for row, value in values.items()
        }

    def get(self, rows):
        return np.stack([self.values[int(row)] for row in rows])


def geometry_contract() -> None:
    manifest = {
        "query_row": np.asarray([0], dtype=np.int64),
        "query_ptr": np.asarray([0, 3], dtype=np.int64),
        "molecule_ptr": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "molecule_label": np.asarray([1, 0, 0], dtype=np.int8),
        "molecule_ik14": np.asarray(["TRUE", "NEG1", "NEG2"]),
        "pair_candidate_row": np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64),
    }
    cache = FakeCache({
        0: [1.0, 0.0],
        1: [0.80, 0.60],
        2: [0.90, 0.4358899],
        3: [0.95, 0.3122499],
        4: [0.10, 0.9949874],
        5: [0.85, 0.5267827],
        6: [0.20, 0.9797959],
    })
    geometry = query_geometry(manifest, cache, 0, 0.1)
    assert geometry["positive_row"] == 2
    assert geometry["hardest_candidate"] == 1
    assert geometry["hardest_identity"] == "NEG1"
    assert geometry["error"] is True
    writer = PoolWriter()
    counts = append_query_events(
        writer, geometry, {"NEG2": 2}, is_extra=True,
        negative_references=2, chemical_candidates=1,
    )
    assert counts == {"safe": 0, "official_error": 1, "chemical_error": 1}
    assert writer.curriculum_role == [6, 7]
    assert all(
        right - left == 1
        for left, right in zip(writer.positive_ptr[:-1], writer.positive_ptr[1:])
    )


def selection_contract() -> None:
    signatures = {
        10: ("Orbitrap", "2"),
        11: ("QTOF", "3"),
        12: ("Orbitrap", "2"),
        13: ("QTOF", "4"),
    }
    rows = [
        {"query": 10, "error": False, "margin": 0.20, "hardest_identity": "A"},
        {"query": 11, "error": True, "margin": -0.05, "hardest_identity": "B",
         "embedding_distance": 0.30},
        {"query": 12, "error": False, "margin": 0.30, "hardest_identity": "A",
         "embedding_distance": 0.50},
        {"query": 13, "error": False, "margin": 0.05, "hardest_identity": "C",
         "embedding_distance": 0.20},
    ]
    selected = select_identity_anchors(rows, 10, signatures, 3, 0.1)
    assert [int(row["query"]) for row in selected] == [10, 11, 13]
    assert condition_signature(b"Orbitrap", 35.0, 10.0) == ("Orbitrap", "3")
    assert condition_signature("QTOF", np.nan, 10.0) == ("QTOF", "missing")


def sampling_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / "small.h5"
        with h5py.File(data, "w") as handle:
            handle.create_dataset("INCHIKEY", data=np.asarray(
                [b"A", b"A", b"R", b"R", b"S"], dtype="S14",
            ))
        output = {
            "anchor_idx": np.arange(5, dtype=np.int64),
            "curriculum_role": np.asarray(
                [1, 2, DREAMS_NATIVE_REPLAY, DREAMS_NATIVE_REPLAY, DREAMS_NATIVE_REPLAY],
                dtype=np.int8,
            ),
        }
        weights, audit = identity_equal_sampling_weights(output, data, 0.2, 0.25)
    assert abs(weights[0] - 0.55) < 1e-12
    assert abs(weights[1] - 0.25) < 1e-12
    assert abs(weights[2:4].sum() - 0.1) < 1e-12
    assert abs(weights[4] - 0.1) < 1e-12
    assert audit["safety_boundary"]["identities"] == 1
    assert audit["error_boundary"]["identities"] == 1
    assert audit["replay"]["identities"] == 2


def immutable_prefix_contract() -> None:
    writer = PoolWriter()
    writer.append(10, [11], [12], 1, 0, 1, 1)
    writer.append(20, [21], [22], 2, 1, 2, 2)
    frozen = writer.arrays()
    expected_hash = pool_prefix_semantic_sha256(frozen, 2)
    writer.append(30, [31], [32], 3, 2, 3, 3)
    expanded = writer.arrays()
    assert pool_prefix_equal(frozen, expanded, 2)
    assert pool_prefix_semantic_sha256(expanded, 2) == expected_hash
    changed = {key: value.copy() for key, value in expanded.items()}
    changed["negative_idx"][0] = 99
    assert not pool_prefix_equal(frozen, changed, 2)
    assert pool_prefix_semantic_sha256(changed, 2) != expected_hash


def full_role_metric_contract() -> None:
    ranks = np.asarray([1, 3, 1, 2], dtype=np.int32)
    positive = np.asarray([0.9, 0.4, 0.8, 0.6], dtype=np.float32)
    negative = np.asarray([0.2, 0.5, 0.3, 0.7], dtype=np.float32)
    identities = np.asarray(["A", "A", "B", "C"])
    formulas = np.asarray(["FA", "FA", "FB", "FC"])
    metrics = identity_equal_metrics(ranks, positive, negative, identities)
    assert metrics["identities"] == 3
    assert abs(float(metrics["recall1"]) - 0.5) < 1e-12
    paired = identity_equal_paired(
        np.asarray([2, 3, 1, 2]), ranks, identities, formulas, 100, 7,
    )
    assert paired["identities_improved"] == 1
    assert paired["identities_harmed"] == 0
    assert paired["formula_clusters"] == 3


def source_isolation_contract() -> None:
    trainer = (
        Path(__file__).resolve().parent / "train_chemaware_weighted_native.py"
    ).read_text(encoding="utf-8")
    assert "WeightedRandomSampler" in trainer
    assert "train_chemaware_dreams_native as native" in trainer
    assert "sampling_weight" in trainer
    assert "Noise runtimes" in trainer

    sbatch = (
        Path(__file__).resolve().parent / "run_chemaware_multicondition_max_boundary.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "train_chemaware_weighted_native.py" in sbatch
    assert "evaluate_chemaware_full_role_native.py" in sbatch
    assert "freeze_chemaware_multicondition_artifact.py" in sbatch
    assert "protected_artifact/SHA256SUMS" in sbatch
    assert "--formula-role 2" in sbatch and "--formula-role 3" in sbatch
    assert "--formula-role 4" not in sbatch
    assert "/bin/rm" not in sbatch and "rm -f" not in sbatch
    assert "phaseA_base" in sbatch


def main() -> None:
    geometry_contract()
    selection_contract()
    sampling_contract()
    immutable_prefix_contract()
    full_role_metric_contract()
    source_isolation_contract()
    print("PASS: ChemAware multi-condition max-boundary native contracts", flush=True)


if __name__ == "__main__":
    main()

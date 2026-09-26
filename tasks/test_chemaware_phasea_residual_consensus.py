"""CPU contracts for Phase-A-current residual-consensus native triplets."""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from build_chemaware_max_boundary_native_triplets import PoolWriter
from build_chemaware_phasea_residual_consensus_triplets import (
    PROTECTION_SENTINEL,
    RESIDUAL_CONSENSUS,
    RESIDUAL_CONSENSUS_CHEMICAL,
    RESIDUAL_ISOLATED,
    append_protection_sentinel,
    append_residual_error,
    copy_pool,
    residual_sampling_weights,
)
from build_chemaware_multicondition_max_boundary_triplets import pool_prefix_equal


def geometry_contract() -> None:
    writer = PoolWriter()
    geometry = {
        "query": 7,
        "anchor": 10,
        "positive_row": 11,
        "hardest_candidate": 0,
        "hardest_identity": "FALSE",
        "margin": -0.02,
        "error": True,
        "candidates": {
            0: {
                "identity": "FALSE", "rows": np.asarray([12]),
                "active_rows": np.asarray([12]), "maximum_hinge": 0.12,
            },
            1: {
                "identity": "CHEM", "rows": np.asarray([13]),
                "active_rows": np.asarray([13]), "maximum_hinge": 0.08,
            },
        },
    }
    counts = append_residual_error(writer, geometry, {"CHEM": 17}, True, 1)
    assert counts == {"official": 1, "chemical": 1}
    assert writer.curriculum_role == [
        RESIDUAL_CONSENSUS, RESIDUAL_CONSENSUS_CHEMICAL,
    ]
    safe = {
        "query": 8,
        "anchor": 20,
        "positive_row": 21,
        "hardest_candidate": 0,
        "error": False,
        "candidates": {0: {"rows": np.asarray([22])}},
    }
    assert append_protection_sentinel(writer, safe)
    assert writer.curriculum_role[-1] == PROTECTION_SENTINEL


def prefix_and_sampling_contract() -> None:
    base_writer = PoolWriter()
    base_writer.append(0, [1], [2], 0, 0, 0, 1)
    base_writer.append(1, [0], [3], -1, -1, 0, 4)
    base = base_writer.arrays()
    writer = PoolWriter()
    copy_pool(writer, base)
    base_events = len(writer.anchor)
    writer.append(2, [3], [4], 2, 0, 0, RESIDUAL_CONSENSUS)
    writer.append(3, [2], [4], 3, 0, 0, RESIDUAL_ISOLATED)
    writer.append(4, [3], [2], 4, 0, 0, PROTECTION_SENTINEL)
    expanded = writer.arrays()
    assert pool_prefix_equal(base, expanded, base_events)
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / "small.h5"
        with h5py.File(data, "w") as handle:
            handle.create_dataset(
                "INCHIKEY", data=np.asarray([b"A", b"B", b"C", b"D", b"E"]),
            )
        weights, audit = residual_sampling_weights(
            expanded, base_events, data,
            {
                "phasea_base": 0.70,
                "residual_consensus": 0.18,
                "residual_isolated": 0.07,
                "protection_sentinel": 0.05,
            },
        )
    assert np.all(weights > 0)
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert abs(float(weights[:base_events].sum()) - 0.70) < 1e-12
    assert abs(float(audit["residual_consensus"]["sampling_mass"]) - 0.18) < 1e-12


def source_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    sbatch = (root / "tasks/run_chemaware_phasea_residual_consensus.sbatch").read_text()
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "--official-checkpoint \"$PHASEA\"" in sbatch
    assert "--lr 1e-6" in sbatch
    assert "--max-steps 800" in sbatch
    assert "--save-every-n-steps 100" in sbatch
    assert "--paired-reference phaseA_base --formula-role 2" in sbatch
    assert "--paired-reference phaseA_base --formula-role 3" in sbatch
    assert "--formula-role 4" not in sbatch
    assert "/bin/rm" not in sbatch and "rm -f" not in sbatch
    encoder = (root / "tasks/encode_chemaware_formula_role_checkpoint_rows.py").read_text()
    assert "set(roles) != {0, 1}" in encoder
    assert '"outer_roles_2_3_4_accessed": False' in encoder


def main() -> None:
    geometry_contract()
    prefix_and_sampling_contract()
    source_contract()
    print("PASS: ChemAware Phase-A residual-consensus contracts", flush=True)


if __name__ == "__main__":
    main()

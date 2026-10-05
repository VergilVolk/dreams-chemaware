"""Static contracts for the one-GPU multisource native route."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sbatch = (ROOT / "tasks/run_chemaware_multisource_native.sbatch").read_text()
    assert "#SBATCH --partition=gpu" in sbatch
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert '--official-checkpoint "$PHASEA"' in sbatch
    assert "build_chemaware_max_boundary_native_triplets.py" not in sbatch
    assert "build_chemaware_multisource_native_triplets.py" in sbatch
    assert "export_chemaware_iceberg_source_ledger.py" in sbatch
    assert "SIRIUS_SCORE_DIR" in sbatch
    assert "--paired-reference phaseA_2pp --formula-role 2" in sbatch
    assert "--require-positive-formula-ci" in sbatch
    assert "CHEMAWARE_MULTISOURCE_ROLE2_STOP" in sbatch
    print("PASS: ChemAware multisource one-GPU pipeline contracts", flush=True)


if __name__ == "__main__":
    main()

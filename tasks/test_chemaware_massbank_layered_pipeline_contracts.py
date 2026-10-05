"""Static contracts for the one-GPU MassBank-layered native route."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    text = (ROOT / "tasks/run_chemaware_massbank_layered_native.sbatch").read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "test_chemaware_massbank_layered_runtime.py" in text
    assert "test_chemaware_massbank_annotated_source.py" not in text
    assert "test_chemaware_dynamic_reference_native_triplets.py" not in text
    assert '--official-checkpoint "$PHASEA"' in text
    assert 'CORPUS="data/validation/chemaware_massbank_layered_12k_native_v1_20260928"' in text
    assert "build_chemaware_max_boundary_native_triplets.py" not in text
    assert "build_chemaware_multisource_native_triplets.py" not in text
    assert "--max-steps 1000" in text
    assert "--save-every-n-steps 250" in text
    assert "--paired-reference phaseA_2pp --formula-role 2" in text
    assert "--require-positive-formula-ci" in text
    assert "--paired-reference phaseA_2pp --formula-role 3" in text
    assert "CHEMAWARE_MASSBANK12K_ROLE2_STOP" in text
    assert "sha256sum" in text
    print("PASS: ChemAware MassBank-layered one-GPU pipeline contracts")


if __name__ == "__main__":
    main()

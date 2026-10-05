"""Static contracts for the one-GPU exact-boundary Slurm entrypoint."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tasks/run_chemaware_exact_boundary_residual.sbatch"
    text = path.read_text(encoding="utf-8")
    assert text.count("#SBATCH --gpus=1") == 1
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --mem" not in text and "--mem=" not in text
    assert "build_chemaware_exact_boundary_residual_triplets.py" in text
    assert "freeze_chemaware_exact_boundary_artifact.py" in text
    assert "--official-checkpoint \"$PHASEA\"" in text
    assert "--lr 1e-6" in text and "--max-steps 500" in text
    assert "--save-every-n-steps 100" in text
    assert "--paired-reference phaseA_base --formula-role 2" in text
    assert "CHEMAWARE_EXACT_BOUNDARY_ROLE2_STOP" in text
    assert "CHEMAWARE_EXACT_BOUNDARY_COVERAGE_STOP" in text
    assert "exit 0" in text
    assert "--paired-reference phaseA_base --formula-role 3" in text
    assert "--formula-role 2 3" in text
    assert "no_identity_broadcast" in text
    print("PASS: ChemAware exact-boundary one-GPU sbatch contracts")


if __name__ == "__main__":
    main()

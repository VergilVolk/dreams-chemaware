"""Contracts for context-only checkpoint selection and no-retrain resume."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    selector = (ROOT / "tasks/select_chemaware_residual_checkpoint.py").read_text(encoding="utf-8")
    resume = (ROOT / "tasks/resume_chemaware_fragment_massbank_union_selection.sbatch").read_text(encoding="utf-8")
    formal = (ROOT / "tasks/run_chemaware_fragment_massbank_union_continue.sbatch").read_text(encoding="utf-8")
    assert '"--exclude-name"' in selector
    assert "unknown_exclusions" in selector
    assert "--exclude-name phaseA_2pp" in formal
    assert "#SBATCH --gpus=1" in resume and "#SBATCH --mem" not in resume
    assert "train_chemaware_dreams_native.py" not in resume
    assert "build_chemaware_dynamic_reference_native_triplets.py" not in resume
    assert "--exclude-name phaseA_2pp" in resume
    assert "run_2346441" in resume
    assert "--formula-role 3" in resume
    print("PASS: ChemAware union selection resume contracts")


if __name__ == "__main__":
    main()

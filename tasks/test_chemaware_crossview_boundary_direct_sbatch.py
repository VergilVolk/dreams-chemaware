"""Static contracts for direct cross-view triplet fine-tuning."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    text = (ROOT / "tasks/run_chemaware_crossview_boundary_direct.sbatch").read_text(encoding="utf-8")
    assert text.count("#SBATCH --gpus=1") == 1
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --mem" not in text and "--mem=" not in text
    assert "CHEMAWARE_STORAGE_PREFLIGHT_STOP" in text
    assert "12 * 1024 * 1024" in text
    assert "build_chemaware_crossview_boundary_native_triplets.py" in text
    assert "train_chemaware_dreams_native.py" in text
    assert "--official-checkpoint \"$PHASEA\"" in text
    assert "--triplet-loss-margin 0.1" in text
    assert "--max-steps 500" in text and "--save-every-n-steps 100" in text
    assert "distillation" in text and "is False" in text
    assert "ordinary DreaMS spectrum triplets only" in text
    assert "CHEMAWARE_CROSSVIEW_BOUNDARY_COVERAGE_STOP" in text
    assert "--minimum-correction-events 100 --minimum-correction-queries 50" in text
    assert "--minimum-correction-formulas 40" in text
    assert "--paired-reference phaseA_base --formula-role 2" in text
    assert "--paired-reference phaseA_base --formula-role 3" in text
    print("PASS: ChemAware cross-view native direct-finetuning sbatch contracts")


if __name__ == "__main__":
    main()

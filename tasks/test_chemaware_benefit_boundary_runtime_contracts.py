"""Static contracts for the one-GPU benefit-boundary execution path."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sbatch = (ROOT / "tasks/run_chemaware_benefit_boundary_native.sbatch").read_text(
        encoding="utf-8",
    )
    trainer = (ROOT / "tasks/train_chemaware_dreams_native.py").read_text(
        encoding="utf-8",
    )
    builder = (
        ROOT / "tasks/build_chemaware_benefit_boundary_native_triplets.py"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "--official-checkpoint \"$PHASEA\"" in sbatch
    assert "--paired-reference phaseA_2pp" in sbatch
    assert "build_chemaware_benefit_boundary_native_triplets.py" in sbatch
    assert "--maximum-events-per-query 16" in sbatch
    assert "--minimum-benefit-proven-formulas 180" in sbatch
    assert "--minimum-active-correction-formulas 160" in sbatch
    assert 'background["old_error_events_removed"]' in builder
    assert '"exact_phasea_error_event_replacement"' in builder
    assert '"phasea_train_pool_cardinality_preserved"' in builder
    assert "train_chemaware_dreams_native.py" in sbatch
    assert "ContrastiveSpectraDataset" in trainer
    assert "n_pos_samples=1, n_neg_samples=1" in trainer
    assert "ContrastiveHead(" in trainer
    assert "custom_optimizer\": False" in trainer
    print("PASS: ChemAware benefit-boundary runtime contracts", flush=True)


if __name__ == "__main__":
    main()

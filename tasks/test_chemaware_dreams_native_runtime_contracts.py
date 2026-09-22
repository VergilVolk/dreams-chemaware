"""Static contracts: ChemAware changes triplets, not DreaMS training."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
trainer = (ROOT / "tasks/train_chemaware_dreams_native.py").read_text(encoding="utf-8")
blocked_sbatch = (ROOT / "tasks/run_chemaware_dreams_native.sbatch").read_text(encoding="utf-8")
sbatch = (ROOT / "tasks/run_chemaware_high_coverage_native.sbatch").read_text(encoding="utf-8")
resume = (ROOT / "tasks/run_chemaware_high_coverage_native_resume_2340524.sbatch").read_text(encoding="utf-8")

assert "from dreams.models.heads.heads import ContrastiveHead" in trainer
assert "from dreams.utils.data import ContrastiveSpectraDataset" in trainer
assert "model = ContrastiveHead(" in trainer
assert "ContrastiveSpectraDataset(" in trainer
assert "configure_optimizers" in trainer
assert "torch.optim.Adam(" not in trainer
assert "F.relu" not in trainer and "clamp_min" not in trainer
assert 'monitor="Train loss"' in trainer
assert "every_n_train_steps=args.save_every_n_steps" in trainer
assert '"--save-every-n-steps", type=int, default=1000' in trainer
assert "trainer.validate(" not in trainer
assert 'default=0' in trainer[trainer.index('"--num-workers"'):trainer.index('"--num-workers"') + 220]
assert "BLOCKED:" in blocked_sbatch and "exit 2" in blocked_sbatch
assert sbatch.count("#SBATCH --gpus=1") == 1
assert sbatch.count("#SBATCH --ntasks=1") == 1
assert "#SBATCH --mem" not in sbatch
assert "srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py" in sbatch
assert "SlurmLineProgress" in trainer
assert "enable_progress_bar=False" in trainer
assert "build_chemaware_action_hard_native_triplets.py" in sbatch
assert "--triplet-evidence-only" in sbatch
assert sbatch.count("confirmation_triplet_evidence.npz") == 2
assert sbatch.index("build_chemaware_action_hard_native_triplets.py") < sbatch.index("train_chemaware_dreams_native.py")
assert "--min-train-queries 3500" in sbatch
assert "--min-train-events 5000" in sbatch
assert resume.count("#SBATCH --gpus=1") == 1
assert resume.count("#SBATCH --ntasks=1") == 1
assert "#SBATCH --mem" not in resume
assert 'SOURCE="data/validation/chemaware_high_coverage_native/run_2340524"' in resume
assert '"$SOURCE/triplets/train_pool.npz"' in resume
assert "srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py" in resume
assert "--num-workers 0" in resume
for argument in (
    "--seed 3407", "--lr 5e-6", "--weight-decay 0",
    "--triplet-loss-margin 0.1", "--batch-size 4",
    "--num-workers 0",
    "--max-epochs 301", "--n-highest-peaks 100",
):
    assert argument in sbatch, argument
print("PASS: ChemAware uses native DreaMS training runtime")

"""Static contracts for the B-PMT trainer, summary, and one-GPU entrypoint."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
trainer = (ROOT / "tasks/train_chemaware_full_candidate_alignment.py").read_text()
summary = (ROOT / "tasks/summarize_chemaware_boundary_pmt.py").read_text()
sbatch = (ROOT / "tasks/run_chemaware_boundary_pmt_phase_a.sbatch").read_text()
ast.parse(trainer); ast.parse(summary)
for value in (
    '"clean_duplicate", "matched_formula_deranged", "alpha025", "alpha050"',
    "active_margin_transfer_loss",
    "inherited_clean_margin_target",
    "positive = model(positive_x).detach()",
    "negative = model(negative_x).detach()",
    '"bpmt_control_gradient_subtraction": False',
    '"bpmt_noncorrective_weight": 0.0',
    "for k in (5, 10, 20, 50)", "baseline_macro_auc", "baseline_micro_auc",
):
    assert value in trainer, value
for value in (
    "formula_cluster_bootstrap_95ci", "matched_formula_deranged",
    "recall5", "recall10", "recall20", "recall50", "macro_auc", "micro_auc",
):
    assert value in summary, value
assert sbatch.count("#SBATCH --partition=gpu") == 1
assert sbatch.count("#SBATCH --gpus=1") == 1
assert "#SBATCH --mem" not in sbatch and "#SBATCH --mem-per-cpu" not in sbatch
assert 'run_${SLURM_JOB_ID}' in sbatch and "run_233" not in sbatch
assert sbatch.count("train_chemaware_full_candidate_alignment.py") == 3
assert "for ARM in clean_duplicate matched_formula_deranged alpha025 alpha050" in sbatch
print("PASS: ChemAware B-PMT training and one-GPU sbatch contracts")

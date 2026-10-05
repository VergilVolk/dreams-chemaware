"""Static fail-closed and resource contracts for qualified action-delta Phase A."""
from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sbatch = (ROOT / "tasks/run_chemaware_qualified_action_delta_phase_a.sbatch").read_text()
    trainer = (ROOT / "tasks/train_chemaware_iceberg_direct_shared.py").read_text()
    core = (ROOT / "tasks/chemaware_action_delta_transfer_core.py").read_text()
    assert "#SBATCH --gpus=1" in sbatch
    assert not any(line.startswith("#SBATCH --mem") for line in sbatch.splitlines())
    assert not any(line.startswith("#SBATCH --array") for line in sbatch.splitlines())
    assert "set -euo pipefail" in sbatch
    assert "direct_action_delta_transfer" in sbatch
    # The immutable bank, never typed CLI values, supplies action mode/dose/k.
    common = sbatch[sbatch.index("COMMON_ARGS=("):sbatch.index(")\n\n# Validate")]
    assert "--differential-mode" not in common
    assert "--action-strength" not in common
    assert "--action-top-k" not in common
    assert "--action-generation-seed" not in common
    assert "--maximum-action-gradient-ratio" not in common
    assert "--gpus=1" in sbatch and "--grad-clip 5" in common
    assert "--max-eval-identities 0" in common
    clean = sbatch.index("run_arm clean_duplicate")
    correct = sbatch.index("run_arm correct_synthetic")
    stage = sbatch.index("--stage primary")
    swapped = sbatch.index("run_arm candidate_swapped")
    permuted = sbatch.index("run_arm peak_permuted")
    assert clean < correct < stage < swapped < permuted
    assert "pass_to_matched_controls" in sbatch and "STOPPED_EARLY" in sbatch
    assert "test_chemaware_action_delta_summary_runtime.py" in sbatch
    assert "test_chemaware_direct_runtime_contracts.py" in sbatch
    assert "test_chemaware_frozen_prefix_cache_import.py" in sbatch
    assert "QualifiedActionTensorSource" in trainer
    assert "evaluation_manifest" in trainer and "broad_inner_per_query.csv.gz" in trainer
    assert "evaluation manifest, official checkpoint, and embedding cache provenance drifted" in trainer
    assert "frozen_qualified_embedding" in trainer
    assert "clean_inherits_action_delta_loss" in trainer
    assert "role_calibrated_dose" in trainer
    assert "chemical_optimizer = torch.optim.AdamW" in trainer
    assert "separate_optimizer_moments" in trainer
    assert "optimizer_descent_geometry" in trainer
    assert "chemical_optimizer_signal_preserved" in trainer
    assert "all_losses_nonincreasing" in trainer
    assert "evaluate_action_delta_fit" in trainer
    assert "chemical_target_fit_improved" in trainer
    summary = (ROOT / "tasks/summarize_chemaware_action_delta_transfer.py").read_text()
    assert "return report, rows, target" in summary
    assert "for arm, (report, rows, target) in loaded.items()" in summary
    assert "action_view_trainable_forward\": False" in trainer
    assert "official_references.detach()" in core
    assert "candidate_center(action - clean).detach()" in core
    tree = ast.parse(trainer)
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }
    options.add("--no-amp")
    invoked = set(re.findall(r"--[a-z][a-z0-9-]*", common)) | {
        "--arm", "--output", "--preflight-only",
    }
    assert not (invoked - options), f"unknown trainer options: {sorted(invoked - options)}"
    print("PASS: qualified ChemAware action-delta Phase A is one-GPU and fail-closed")


if __name__ == "__main__":
    main()

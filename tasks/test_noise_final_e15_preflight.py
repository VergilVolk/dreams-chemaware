"""Regression tests for the E15 preflight audit itself."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import audit_noise_final_e15_preflight as audit  # noqa: E402


def main() -> None:
    audit_source = Path(audit.__file__).read_text(encoding="utf-8")
    ast.parse(audit_source)
    required_contract = {
        "multiple_corrective_actions_per_query": True,
        "harmful_actions_materialized_separately": True,
        "corrective_and_risk_losses_are_distinct": True,
        "within_epoch_action_recycling_forbidden": True,
        "gradient_calibration_minimum_stratified_microbatches": 32,
        "gradient_calibration_minimum_identity_action_observations": 128,
        "overfit_gate_before_identity_holdout": True,
        "identity_holdout_before_formula_fold": True,
        "formula_fold_before_multifold": True,
        "P2b_forbidden": True,
        "P3_consumed": False,
    }
    drifted = {
        key: (audit.E15_MANDATORY_CONTRACT.get(key), expected)
        for key, expected in required_contract.items()
        if audit.E15_MANDATORY_CONTRACT.get(key) != expected
    }
    if drifted:
        raise RuntimeError(f"E15 executable contract drifted: {drifted}")

    teacher = audit.E14_TEACHER.read_text(encoding="utf-8")
    trainer = audit.E14_TRAINER.read_text(encoding="utf-8")
    # These assertions are intentionally fail-closed: the audit must continue
    # detecting every legacy flaw until E15 is implemented in separate files.
    audit.require_tokens(teacher, (
        "best = int(correcting[np.argmax(result_margin[local, correcting])])",
        '"one_selected_action_per_query": True',
    ), "test single-action")
    audit.require_tokens(trainer, (
        "guided_batch = guided_batch + guided_risk_epoch[",
        '"p_corrective": guided_examples[: min(4, len(guided_examples))]',
        "guided_cursor = 0",
    ), "test legacy trainer")

    sbatch = (ROOT / "tasks/run_noise_final_e15_preflight.sbatch").read_text(
        encoding="utf-8"
    )
    required_sbatch = (
        "#SBATCH --partition=gpu",
        "#SBATCH --gpus=1",
        "set -euo pipefail",
        '[[ ! -e "$OUTPUT" ]]',
        "python -u tasks/test_noise_final_e15_preflight.py",
        "python -u tasks/test_noise_final_e4a_direct_augmentation.py",
        "python -u tasks/test_noise_final_e14_crossfit_teacher.py",
        "python -u tasks/audit_noise_final_e15_preflight.py",
    )
    missing_sbatch = [token for token in required_sbatch if token not in sbatch]
    if missing_sbatch:
        raise RuntimeError(f"E15 sbatch contract is incomplete: {missing_sbatch}")
    positions = [sbatch.index(token) for token in required_sbatch[4:]]
    if positions != sorted(positions):
        raise RuntimeError("E15 sbatch does not run tests before the formal audit")
    forbidden_sbatch = ("python -u tasks/train_noise", "--array=", "--mem=", "python - <<")
    found_sbatch = [token for token in forbidden_sbatch if token in sbatch]
    if found_sbatch:
        raise RuntimeError(
            f"E15 audit-only sbatch unexpectedly launches training or unsafe inline code: "
            f"{found_sbatch}"
        )
    print("[test_noise_final_e15_preflight] PASS", flush=True)


if __name__ == "__main__":
    main()

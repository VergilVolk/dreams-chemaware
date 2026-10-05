"""Static launch contracts for the Phase-A error-winner continuation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_chemaware_phasea_error_winner_native_continue.sbatch"


def test_exactly_one_gpu_and_no_manual_memory() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text
    assert "devices=1" not in text


def test_protected_phasea_weights_and_adam_are_same_checkpoint() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert '--official-checkpoint "$PHASEA" --restore-adam-from "$PHASEA"' in text
    assert "audit_chemaware_native_resume_checkpoint.py" in text
    assert '--expected-lr 5e-6 --expected-weight-decay 0' in text
    assert "step-750" not in text


def test_only_native_triplet_curriculum_changes() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "build_chemaware_phasea_error_winner_native_triplets.py" in text
    assert "train_chemaware_dreams_native.py" in text
    assert "--triplet-loss-margin 0.1" in text
    assert "listwise" not in text.lower()
    assert "layered_12k" not in text
    assert "dreams-replay-pool" not in text


def test_role2_selects_before_role3_is_opened() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    selection = text.index("select_chemaware_residual_checkpoint.py")
    stop = text.index("CHEMAWARE_ERROR_WINNER_ROLE2_STOP")
    role3 = text.index('--formula-role 3')
    assert selection < stop < role3
    assert "--base-name phaseA_2pp --require-positive-formula-ci" in text
    assert "--protected-baseline-name phaseA_2pp" in text


def test_fixed_step_checkpoints_cover_one_native_epoch() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "STEPS_PER_EPOCH=$((TOTAL_EVENTS / 4))" in text
    assert '--max-epochs 1 --max-steps "$STEPS_PER_EPOCH"' in text
    assert '--checkpoint-mode fixed_steps --save-every-n-steps "$SAVE_EVERY"' in text


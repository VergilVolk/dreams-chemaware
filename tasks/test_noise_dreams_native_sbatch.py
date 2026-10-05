"""Static Slurm contract for Noise on the proven native-triplet runtime."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "tasks/run_noise_dreams_native_2gpu.sbatch").read_text(
    encoding="utf-8"
)

assert text.count("#SBATCH --gpus=2") == 1
assert "#SBATCH --mem" not in text
assert "tasks/build_noise_reference_aligned_native_triplets.py" in text
assert "tasks/build_noise_action_hard_native_triplets.py" not in text
assert "tasks/train_noise_reference_native.py" in text
assert "tasks/train_chemaware_dreams_native.py" not in text
assert "tasks/build_chemaware_" not in text
assert "tasks/test_noise_action_hard_native_triplets.py" not in text
assert 'VERSION = "noise_reference_aligned_native_v3"' in text
assert text.index("build_noise_reference_aligned_native_triplets.py") < text.index(
    "train_noise_reference_native.py"
)
assert "--expected-actions 32114" in text
assert "--base-queries-per-formula 1" in text
assert "--negative-references-per-action-candidate 2" in text
assert "--positive-references-per-negative 2" in text
assert "--minimum-train-formulas 800" in text
assert "--minimum-triplet-events 6500" in text
assert "--minimum-active-events 4000" in text
assert "--minimum-exact-action-events 1000" in text
assert "--minimum-active-action-fraction 0.50" in text
assert '--official-checkpoint "$OFFICIAL_SLIM"' in text
assert '--official-slim-checkpoint "$OFFICIAL_SLIM"' in text
assert 'OFFICIAL_SLIM="data/e1/official_embedding_slim.pt"' in text
assert 'E8_ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"' in text
assert 'MATURE_E8="data/validation/g8r_noise_final_e8_direct_transfer/' in text
assert "--mature-e8-checkpoint \"$MATURE_E8\"" in text
for frozen in (
    "--lr 5e-6 --weight-decay 0",
    "--triplet-loss-margin 0.1",
    "--batch-size 4 --num-workers 0",
    "--max-epochs 2 --max-steps \"$TRAIN_STEPS\"",
    "--checkpoint-mode fixed_steps --save-every-n-steps \"$CHECKPOINT_INTERVAL\"",
    "--checkpoint-save-weights-only --no-save-last-checkpoint",
    "--keep-final-partial-batch",
):
    assert frozen in text, frozen
assert 'run_arm "${ALLOCATED_GPUS[0]}" primary 3407' in text
assert 'run_arm "${ALLOCATED_GPUS[1]}" replicate 3408' in text
assert "STATUS_PRIMARY" in text and "STATUS_REPLICATE" in text
assert "[native-train-heartbeat]" in text
assert "TRAIN_STEPS=3000" in text
assert 'CHECKPOINT_INTERVAL="$TRAIN_STEPS"' in text
assert 'TRAIN_CHECKPOINT="step-003000.ckpt"' in text
assert "TRAIN_EVENTS > 7500" in text
assert 'LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"' in text
assert 'ARMS="$LOCAL_ROOT/arms"' in text
assert "LOCAL_FREE_KB < 3145728" in text
assert "trap cleanup_local EXIT" in text
assert '"$ARMS/$arm/$TRAIN_CHECKPOINT"' in text
assert "best.ckpt" not in text
assert "last.ckpt" not in text
assert "evaluate_noise_dreams_native.py" in text
assert "summarize_noise_dreams_native_replicates.py" in text
assert "summarize_noise_dreams_native.py" not in text
assert "--minimum-delta-recall1-pp 2.0" in text
assert 'RESUME_ROOT="${RESUME_ROOT:-}"' in text
assert "requires a fresh run; unset RESUME_ROOT" in text
assert "train_noise_dreams_native.py" not in text
assert "audit_noise_dreams_native_official_replay.py" not in text
assert "train_pool_control.npz" not in text
trainer = (ROOT / "tasks/train_noise_reference_native.py").read_text(
    encoding="utf-8"
)
assert '"--keep-final-partial-batch", action="store_true"' in trainer
assert '"--checkpoint-save-weights-only", action="store_true"' in trainer
assert '"--no-save-last-checkpoint", action="store_true"' in trainer
assert "save_last=not args.no_save_last_checkpoint" in trainer
assert "save_weights_only=args.checkpoint_save_weights_only" in trainer
assert "drop_last=not args.keep_final_partial_batch" in trainer
assert 'SOURCE_MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"' in text
print("[test_noise_dreams_native_sbatch] PASS")

"""Static safeguards for the V10 job-2336553 storage-failure resume."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_noise_corrected_v10_safe_exact_resume_2336553_2gpu.sbatch"


def test_resume_reuses_frozen_snapshot_and_original_result_root() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "ORIGINAL_JOB_ID=2336553" in text
    assert 'SNAPSHOT_ROOT="$PWD/$RUN_ROOT/source_snapshot"' in text
    assert 'sha256sum -c sha256_manifest.txt' in text
    assert 'sha256sum -c tasks/noise_best_v10_source_manifest.sha256' in text
    assert '"$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py"' in text
    assert 'mv "$RUN_ROOT" "$FINAL_ROOT"' in text


def test_resume_preserves_scalar_and_runs_only_missing_arms() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "Preserving completed arm" in text
    assert "The completed scalar arm from job" in text
    assert "run_arm_async \"$GPU_ZERO\" full_action_view" in text
    assert "run_arm_async \"$GPU_ONE\" matched_shuffled" in text
    assert "run_arm_async \"$GPU_ZERO\" clean_control" in text
    assert 'run_arm_async "$GPU_ZERO" clean_control clean_control full_action_view &' in text
    assert 'wait "$CLEAN_PID"' in text
    assert "run_arm_async \"$GPU_ONE\" scalar_transfer_only" not in text
    assert "run_arm_async \"$GPU_ZERO\" scalar_transfer_only" not in text


def test_resume_has_storage_and_atomic_output_safeguards() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "AVAILABLE_BYTES=$(df -PB1" in text
    assert "REQUIRED_BYTES=" in text
    assert "final_shared_encoder_sha256" in text
    assert "hashlib.sha256" in text
    assert "Refusing to overwrite" in text
    assert "rm -" not in text


def test_resume_uses_two_gpus_without_manual_memory() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in text
    assert "#SBATCH --mem" not in text
    assert "#SBATCH --mem-per-gpu" not in text


def test_summary_only_recovery_never_retrains() -> None:
    summary = (
        ROOT / "tasks/run_noise_corrected_v10_summary_recovery_2336553.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in summary
    assert "#SBATCH --mem" not in summary
    assert "train_noise_corrected_routed_direct.py" not in summary
    assert "summarize_noise_corrected_v10_safe_exact_canary.py" in summary
    assert "validate_arm" in summary
    assert 'mv "$RUN_ROOT" "$FINAL_ROOT"' in summary


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_v10_resume_2336553_sbatch] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()

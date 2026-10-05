"""Fail-closed static checks for the E4 best-actions Injector V1 job."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_noise_e4_best_actions_injector_v1_2gpu.sbatch"


def main() -> None:
    source = SBATCH.read_text(encoding="utf-8")
    required = [
        "#SBATCH --gpus=2",
        "noise_corrected_best_v5_canary_fold_0_run_2332784",
        "train_noise_final_e4a_direct_augmentation.py",
        "--materialized-injection-mode multi_action_balanced",
        "--optimizer-boundary-mode action_injector_v1",
        "--injector-target-attributable-fraction 0.25",
        "--injector-minimum-protective-retention 0.90",
        "--injector-maximum-update-norm-ratio 1.50",
        "--direct-transfer-mode symmetric --rank-reference-mode shared",
        "--epochs 4 --batch-actions 4 --views-per-identity 4",
        "--backbone-lr 2e-6 --head-lr 1e-5",
        'run_arm "$GPU_ZERO" targeted &',
        'run_arm "$GPU_ONE" shuffled &',
        "selected.query_index.duplicated().any()",
        "len(selected) > len(one_best)",
        "all_actions_exposed_before_recycling",
        "source_snapshot",
        "sha256sum -c",
        "archive_partial_arm",
        "failed_attempts",
        "retrying alone on allocated GPU",
        'mv "$RUN_ROOT" "$FINAL_ROOT"',
    ]
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError(f"E4+InjectorV1 SBATCH missing: {missing}")
    forbidden = [
        "#SBATCH --mem", "train_noise_corrected_routed_direct.py",
        "--materialized-injection-mode one_best_e4", "--epochs 1",
        "--maximum-corrective-queries", "historical_best_v1",
    ]
    present = [value for value in forbidden if value in source]
    if present:
        raise AssertionError(f"E4+InjectorV1 SBATCH revived forbidden paths: {present}")
    if source.index("source_snapshot") > source.index("python -m py_compile"):
        raise AssertionError("E4+InjectorV1 tests do not run from the source snapshot")
    print("[test_noise_e4_best_actions_injector_v1_sbatch] PASS tests=1")


if __name__ == "__main__":
    main()

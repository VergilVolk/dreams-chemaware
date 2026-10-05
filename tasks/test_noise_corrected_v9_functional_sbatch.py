"""Static safety and causal-design checks for the V9 two-GPU job."""
from __future__ import annotations

from pathlib import Path


def test_v9_job_uses_exactly_two_gpus_and_no_manual_memory() -> None:
    source = Path(__file__).with_name(
        "run_noise_corrected_v9_functional_canary_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in source
    assert "#SBATCH --mem" not in source
    assert "--mem=" not in source
    assert "python -u \"$SNAPSHOT_ROOT/tasks/train_noise" in source


def test_v9_job_runs_only_the_two_new_matched_budget_arms() -> None:
    source = Path(__file__).with_name(
        "run_noise_corrected_v9_functional_canary_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "full_action_view full_action_view" in source
    assert "scalar_transfer_only scalar_transfer_only" in source
    assert "--corrective-gradient-locality query_action_only" in source
    assert "--target-corrective-to-risk-ratio 1.0" in source
    assert "--materialize-optimizer-update-restoration" in source
    assert "--minimum-optimizer-restoration-target-reached-fraction 0.90" in source


def test_v9_historical_controls_are_bridge_gated() -> None:
    source = Path(__file__).with_name(
        "run_noise_corrected_v9_functional_canary_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    summary = Path(__file__).with_name(
        "summarize_noise_corrected_v9_functional_canary.py"
    ).read_text(encoding="utf-8")
    assert "--v8-root \"$V8_ROOT\"" in source
    assert "historical_control_bridge" in summary
    assert 'if bridge["passed"]:' in summary
    assert "formula_cluster_ci_familywise" in summary


def test_v9_source_manifest_is_frozen_not_placeholder() -> None:
    source = Path(__file__).with_name(
        "run_noise_corrected_v9_functional_canary_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "__V9_SOURCE_MANIFEST_SHA256__" not in source
    assert "sha256sum -c \"$SOURCE_MANIFEST\"" in source
    assert "SOURCE_SNAPSHOT" in source


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_v9_functional_sbatch] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()

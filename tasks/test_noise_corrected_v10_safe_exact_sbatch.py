"""Static safety and resource checks for the V10 Slurm entrypoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def _source() -> str:
    return Path(__file__).with_name(
        "run_noise_corrected_v10_safe_exact_canary_2gpu.sbatch"
    ).read_text(encoding="utf-8")


def test_v10_uses_two_gpus_and_never_requests_memory_manually() -> None:
    source = _source()
    assert "#SBATCH --gpus=2" in source
    assert "#SBATCH --mem" not in source
    assert "--mem=" not in source


def test_v10_runs_four_causal_arms_in_two_gpu_waves() -> None:
    source = _source()
    assert "full_action_view routed_direct full_action_view" in source
    assert "scalar_transfer_only routed_direct scalar_transfer_only" in source
    assert "matched_shuffled shuffled_action_control full_action_view" in source
    assert "clean_control clean_control full_action_view" in source
    assert "V10 wave 1/2" in source and "V10 wave 2/2" in source


def test_v10_uses_hard_safe_exact_optimizer_contract() -> None:
    source = _source()
    required = (
        "--direct-contract best_action_v10_safe_exact",
        "--optimizer-restoration-scope safe_exact_corrective",
        "--optimizer-restoration-minimum-risk-component-retention 0.90",
        "--minimum-optimizer-restoration-target-reached-fraction 1.0",
        "--minimum-optimizer-action-attributable-fraction-p10 0.25",
        "--maximum-safe-exact-update-norm-ratio 1.50",
        "--reconcile-restored-adamw-first-moment",
    )
    assert all(value in source for value in required)


def test_v10_freezes_calibration_and_holds_out_formulas() -> None:
    source = _source()
    assert "--calibration-corrective-branch-mode full_action_view" in source
    assert "--arm-invariant-targeted-calibration" in source
    assert "--inner-holdout-unit formula" in source
    assert "--inner-holdout-fold 0" in source


def test_v10_is_atomic_and_manifest_closed() -> None:
    source = _source()
    assert "SOURCE_SNAPSHOT" in source
    assert 'sha256sum -c "$SOURCE_MANIFEST"' in source
    assert 'mv "$RUN_ROOT" "$FINAL_ROOT"' in source
    assert "__V10_SOURCE_MANIFEST_SHA256__" not in source


def test_v10_snapshots_and_tests_the_versioned_injector() -> None:
    source = _source()
    assert 'INJECTOR_SPEC_SOURCE="docs/NOISE_ACTION_INJECTOR_V1_SPEC_20260913.md"' in source
    assert 'test_noise_action_injector_v1.py' in source
    manifest_path = Path(__file__).with_name("noise_best_v10_source_manifest.sha256")
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "  tasks/noise_action_injector_v1.py" in manifest
    assert "  tasks/test_noise_action_injector_v1.py" in manifest
    expected = re.search(
        r'EXPECTED_SOURCE_MANIFEST_SHA256="([0-9a-f]{64})"', source,
    )
    assert expected is not None
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected.group(1)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_noise_corrected_v10_safe_exact_sbatch] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()

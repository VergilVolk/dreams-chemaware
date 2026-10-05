"""Static fail-closed checks for the V11 two-GPU Slurm entrypoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def _source() -> str:
    return Path(__file__).with_name(
        "run_noise_corrected_v11_historical_best_canary_2gpu.sbatch"
    ).read_text(encoding="utf-8")


def test_v11_uses_exactly_two_gpus_and_no_manual_memory_request() -> None:
    source = _source()
    assert "#SBATCH --gpus=2" in source
    assert "#SBATCH --mem" not in source
    assert "--mem=" not in source


def test_v11_changes_action_supplier_and_keeps_v10_injector_contract() -> None:
    source = _source()
    required = (
        "--direct-contract best_action_v11_historical_best",
        "--action-bank-contract historical_best_v1",
        "--optimizer-restoration-scope safe_exact_corrective",
        "--optimizer-restoration-minimum-risk-component-retention 0.90",
        "--minimum-optimizer-restoration-target-reached-fraction 1.0",
        "--minimum-optimizer-action-attributable-fraction-p10 0.25",
        "--maximum-safe-exact-update-norm-ratio 1.50",
        "--reconcile-restored-adamw-first-moment",
    )
    assert all(value in source for value in required)


def test_v11_runs_four_causal_arms_only_through_two_background_waves() -> None:
    source = _source()
    assert "full_action_view routed_direct full_action_view &" in source
    assert "scalar_transfer_only routed_direct scalar_transfer_only &" in source
    assert "matched_shuffled shuffled_action_control full_action_view &" in source
    assert "clean_control clean_control full_action_view &" in source
    assert "wait_pair_fail_fast" in source
    assert "V11 wave 1/2" in source and "V11 wave 2/2" in source


def test_v11_holds_out_formulas_before_action_preflight_and_training() -> None:
    source = _source()
    assert "audit_noise_historical_best_action_bank_v1.py" in source
    assert "--inner-holdout-unit formula" in source
    assert "--inner-holdout-fold 0" in source
    assert "--arm-invariant-targeted-calibration" in source


def test_v11_source_snapshot_is_atomic_and_manifest_closed() -> None:
    source = _source()
    assert "SOURCE_SNAPSHOT" in source
    assert 'sha256sum -c "$SOURCE_MANIFEST"' in source
    assert '"$V10_SOURCE_MANIFEST"' in source
    assert '"$V10_SBATCH_SOURCE"' in source
    assert 'mv "$RUN_ROOT" "$FINAL_ROOT"' in source
    assert "__V11_SOURCE_MANIFEST_SHA256__" not in source
    manifest = Path(__file__).with_name("noise_best_v11_source_manifest.sha256")
    expected = re.search(
        r'EXPECTED_SOURCE_MANIFEST_SHA256="([0-9a-f]{64})"', source,
    )
    assert expected is not None
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected.group(1)


def test_v11_guards_checkpoint_storage_before_gpu_work() -> None:
    source = _source()
    assert "AVAILABLE_KB" in source
    assert "6291456" in source
    assert source.index("AVAILABLE_KB") < source.index("V11 wave 1/2")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_noise_corrected_v11_historical_best_sbatch] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()

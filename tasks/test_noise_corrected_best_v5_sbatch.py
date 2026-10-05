"""Static fail-closed checks for the only best-v5 server entry point."""
from __future__ import annotations

from pathlib import Path
import re


SBATCH = Path(__file__).with_name(
    "run_noise_corrected_best_v5_canary_2gpu.sbatch"
)


def test_resource_contract_is_exactly_two_gpus_without_manual_memory() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert re.findall(r"^#SBATCH --gpus=.*$", text, flags=re.MULTILINE) == [
        "#SBATCH --gpus=2",
    ]
    assert not re.search(
        r"^#SBATCH --(?:mem|mem-per-cpu|mem-per-gpu)(?:=|\s)",
        text, flags=re.MULTILINE,
    )


def test_best_actions_and_low_loss_allocator_are_consumed_by_training() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    required = (
        "audit_noise_corrected_v4_action_router.py",
        '"$N_ROUTE" "$P_ROUTE" "$A4_ROUTE" "$V4_ROUTE"',
        '"N_mature", "P_guided_original", "E10B", "E11", "E12B"',
        '"A4_exact", "V4_gradient_path"',
        "--transfer-target-allocation mass_neutral_monotone",
        'v4_train.supervision_kind.eq("corrective").any()',
        'v4["contracts"]["current_E8_error_queries_all_retained_in_action_panel"]',
        'ledger["contracts"]["all_route_clean_ranks_match"]',
    )
    for value in required:
        assert value in text
    assert "--transfer-target-allocation hard_cap" not in text


def test_two_gpu_causal_triad_and_complete_summary_are_present() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert 'run_arm "$GPU_ZERO" routed_direct &' in text
    assert 'run_arm "$GPU_ONE" shuffled_action_control &' in text
    assert 'run_arm "$GPU_ZERO" clean_control' in text
    assert "summarize_noise_corrected_direct_v3_canary.py" in text
    assert "mv \"$RUN_ROOT\" \"$FINAL_ROOT\"" in text


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_best_v5_sbatch] PASS tests={len(tests)}")

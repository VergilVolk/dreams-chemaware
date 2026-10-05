#!/usr/bin/env python
"""Static launch-contract test for the two-GPU Stage-1 RRF confirmation."""
from pathlib import Path


def main() -> None:
    script = (Path(__file__).with_name("run_noise_stage1_rrf_confirmation_2gpu.sbatch"))
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in text
    assert "#SBATCH --mem" not in text
    assert "--dependency" not in text
    assert "evaluate_noise_stage1_rrf_confirmation.py" in text
    assert "evaluate_gnps_stage1_rrf_confirmation.py" in text
    assert "noise_dreams_hard_positive_native_fold_0_run_2344820" in text
    assert "rrf_confirmation" in text
    print("[test_noise_stage1_rrf_confirmation_sbatch] PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from pathlib import Path


def main() -> None:
    text = Path(__file__).with_name(
        "run_noise_relation_t1_t3_v3_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in text
    assert "#SBATCH --mem" not in text
    assert "--dependency" not in text
    assert "train_noise_relation_t1_t3_v3.py" in text
    assert 'run_arm "${ALLOCATED_GPUS[0]}" targeted' in text
    assert 'run_arm "${ALLOCATED_GPUS[1]}" control' in text
    assert "--action-fraction 0.5" in text
    assert "--rotation 0" in text
    assert "summarize_noise_relation_t1_t3_v3_fold1.py" in text
    assert "proceed_to_outer_held" in text
    assert "evaluate_noise_dreams_native.py" not in text
    assert "encode_gnps" not in text
    assert "NO-GO: outer held and GNPS remain untouched" in text
    assert text.index('if [[ "$PROCEED" != "yes" ]]') < text.index(
        'cp "$TARGETED/final_slim.pt"'
    )
    assert 'cp "$CONTROL/final_optimizer.pt"' not in text
    print("[test_noise_relation_t1_t3_v3_sbatch] PASS")


if __name__ == "__main__":
    main()

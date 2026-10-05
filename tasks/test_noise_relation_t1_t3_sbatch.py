#!/usr/bin/env python
from pathlib import Path


def main() -> None:
    text = Path(__file__).with_name("run_noise_relation_t1_t3_2gpu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in text
    assert "#SBATCH --mem" not in text
    assert "--dependency" not in text
    assert "train_noise_relation_t1_t3.py" in text
    assert "seed_3407" in text and "seed_3408" in text
    assert "GLM_probe" not in text and "rrf" not in text.lower()
    print("[test_noise_relation_t1_t3_sbatch] PASS")


if __name__ == "__main__":
    main()

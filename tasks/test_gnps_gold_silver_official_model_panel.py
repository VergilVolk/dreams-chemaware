#!/usr/bin/env python
"""Static contracts for the external official-model comparison panel."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = (ROOT / "tasks/run_gnps_gold_silver_official_model_panel_1gpu.sbatch").read_text()
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --mem" not in script
    assert "train_noise" not in script
    assert "c81a62766b10dd1d39fcda3edec5ef88623e5f6b" in script
    assert "d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832" in script
    assert "SLURM_TMPDIR" in script
    assert script.count("python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py") == 3
    assert script.count("python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py") == 2
    assert "max-epochs" not in script
    print("[test_gnps_gold_silver_official_model_panel] PASS")


if __name__ == "__main__":
    main()

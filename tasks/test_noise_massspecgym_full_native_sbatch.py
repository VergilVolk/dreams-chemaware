#!/usr/bin/env python
"""Static submission contract for the single full-MassSpecGym Noise run."""
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "tasks" / "run_noise_massspecgym_full_native_2gpu.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt" in script
    assert "MassSpecGym_MurckoHist_split.hdf5" in script
    assert "encode_noise_massspecgym_checkpoint.py" in script
    assert "--shard-count 2" in script
    assert "build_noise_massspecgym_full_triplets.py" in script
    assert "train_noise_massspecgym_full_native.py" in script
    assert script.count("evaluate_gnps_gold_silver_10ppm_embeddings.py") >= 3
    assert "official_vs_new" in script and "v1_vs_new" in script
    assert "evaluate_noise_dreams_native.py" not in script
    assert "--max-epochs 1" in script
    print("[test_noise_massspecgym_full_native_sbatch] PASS")


if __name__ == "__main__":
    main()

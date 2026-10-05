#!/usr/bin/env python
"""Static submission contract for the WSE-discordant native continuation."""
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "tasks" / "run_noise_wse_discordant_native_2gpu.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=2" in text
    assert "#SBATCH --mem" not in text
    # Pinned WSE backend, node-local install, interpreter guard.
    assert "ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64" in text
    assert "requires Python 3.11" in text
    # V1 champion warm start, full corrected graph, one triplet per query.
    assert "noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt" in text
    assert "MassSpecGym_MurckoHist_split.hdf5" in text
    assert "build_noise_wse_discordant_native_triplets.py" in text
    assert "--native-margin 0.1" in text
    assert "--fragment-tolerance 0.02" in text
    assert "train_noise_wse_discordant_native.py" in text
    assert "--seed 3407" not in text  # seeds passed via the train_seed shell function
    assert "train_seed" in text and "3407" in text and "3408" in text
    # Frozen GNPS baselines are reused, never re-derived.
    assert "gnps_absolute_baselines_run_2347472" in text
    assert "official_embeddings.npz" in text and "v1_embeddings.npz" in text
    # Both registered seeds are evaluated against official and V1, then summarized.
    assert text.count("evaluate_gnps_gold_silver_10ppm_embeddings.py") >= 3
    assert "gnps_official_vs_seed_3407" in text and "gnps_official_vs_seed_3408" in text
    assert "gnps_v1_vs_seed_3407" in text and "gnps_v1_vs_seed_3408" in text
    assert "summarize_noise_wse_discordant_gnps.py" in text
    # Discipline: no held-fold evaluation, contract tests run inside the job.
    assert "evaluate_noise_dreams_native" not in text
    assert "test_noise_wse_discordant_native.py" in text
    print("[test_noise_wse_discordant_native_sbatch] PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Contract test for the GNPS absolute-baseline job."""
from pathlib import Path


def main() -> None:
    text = Path(__file__).with_name(
        "run_gnps_absolute_baselines_1gpu.sbatch",
    ).read_text(encoding="utf-8")
    # One GPU, no training anywhere, no held fold, no GNPS spectra as
    # training material.
    assert "#SBATCH --gpus=1" in text
    assert "train_noise" not in text
    assert "evaluate_noise_dreams_native.py" not in text
    # The three reference checkpoints: official slim directly, V1 slim
    # directly, Stage-1 via the proven format-only slim conversion.
    assert 'encode official "$OFFICIAL"' in text
    assert 'encode stage1 "$STAGE1_SLIM"' in text
    assert 'encode v1 "$V1_SLIM"' in text
    assert "prepare_official_embedding_checkpoint.py" in text
    assert "noise_relation_t1_t3_run_2347055" in text
    # Paired evaluations against the official baseline; --output is a
    # directory (the evaluator writes report.json inside it).
    assert '--baseline-embeddings "$GNPS/official_embeddings.npz"' in text
    assert '--candidate-embeddings "$GNPS/stage1_embeddings.npz"' in text
    assert '--candidate-embeddings "$GNPS/v1_embeddings.npz"' in text
    assert '--output "$GNPS/paired_official_vs_stage1"' in text
    assert '--output "$GNPS/paired_official_vs_v1"' in text
    assert "/report.json\" \\\\" not in text
    # The evaluation-only charter is stated in the job itself.
    assert "evaluation-only by charter" in text
    assert "enters training" in text
    print("[test_gnps_absolute_baselines_sbatch] PASS")


if __name__ == "__main__":
    main()

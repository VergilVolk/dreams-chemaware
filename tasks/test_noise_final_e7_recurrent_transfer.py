"""Contract test for E7 direct recurrent peak-transfer augmentation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    batch = (ROOT / "tasks/run_noise_final_e7_recurrent_transfer.sbatch").read_text(
        encoding="utf-8",
    )
    trainer = (ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py").read_text(
        encoding="utf-8",
    )
    for token in (
        "#SBATCH --gpus=1", "#SBATCH --array=0-4", "recurrent_union_mix",
        '--action-selection fixed', '--action-scope errors',
        '--guided-noise-policy "${GPOLICY}"', '--safety-stream-weight 2.0',
    ):
        if token not in batch:
            raise AssertionError(f"E7 batch contract token missing: {token}")
    if batch.count("SUFFIX=") != 5:
        raise AssertionError("E7 must contain one control and four fixed weights")
    for token in (
        'example.family == "recurrent_union_mix"',
        'forward_embeddings(model, spectra.to(device), args.amp)',
        'shared_query_reference_encoder": True',
        'inference_clean_spectrum_only": True',
        'P2b": "forbidden"',
    ):
        if token not in trainer:
            raise AssertionError(f"E7 direct-encoder token missing: {token}")
    if "p2b_frozen" in trainer.lower():
        raise AssertionError("E7 trainer imports a downstream P2b artifact")
    print("[test_noise_final_e7_recurrent_transfer] PASS")


if __name__ == "__main__":
    main()

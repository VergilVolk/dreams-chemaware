"""Fail-closed static contract for the E8 direct-transfer factorial."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
SBATCH = ROOT / "tasks/run_noise_final_e8_direct_transfer_factorial.sbatch"


def main() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    ast.parse(source)
    batch = SBATCH.read_text(encoding="utf-8")
    required_source = {
        "encode_official_action_targets",
        "frozen_reference_margins",
        'choices=("symmetric", "student_action_stopgrad", "official_action")',
        'choices=("shared", "official")',
        "official_action_targets_frozen_before_optimizer",
        '"P2b": "forbidden"',
        "final_shared_encoder.pt",
    }
    missing = sorted(token for token in required_source if token not in source)
    if missing:
        raise RuntimeError(f"E8 trainer contract missing: {missing}")
    required_batch = {
        "#SBATCH --gpus=1", "--action-scope all", "--views-per-identity 4",
        "--backbone-lr 2e-6", "--head-lr 1e-5", "--outer-fold 0",
        "--guided-noise-policy none", "--action-selection fixed",
    }
    missing_batch = sorted(token for token in required_batch if token not in batch)
    if missing_batch:
        raise RuntimeError(f"E8 sbatch contract missing: {missing_batch}")
    # The array is a preregistered causal decomposition, not an arbitrary
    # hyperparameter sweep: one historical baseline, three mechanism isolates,
    # the combined correction, then two fixed-policy decompositions.
    for token in (
        'TRANSFER="symmetric"', 'TRANSFER="student_action_stopgrad"',
        'TRANSFER="official_action"', 'REFERENCE="shared"',
        'REFERENCE="official"', 'POLICY="candidate"', 'POLICY="combined"',
    ):
        if token not in batch:
            raise RuntimeError(f"E8 factorial arm missing: {token}")
    print("[test_noise_final_e8_direct_transfer_factorial] PASS")


if __name__ == "__main__":
    main()

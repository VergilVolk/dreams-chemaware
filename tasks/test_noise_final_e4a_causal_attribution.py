"""Unit tests for the strict E4-A clean/random/targeted attribution factor."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from noise_v3_core import attenuate_sequence
from train_noise_final_e4a_direct_augmentation import (
    DirectExample,
    identity_balanced_epoch,
    materialize_causal_arm,
    validate_causal_configuration,
)
from summarize_noise_final_e4a_causal_attribution import paired_summary


def synthetic_actions(rows: int = 128) -> pd.DataFrame:
    return pd.DataFrame({
        "query_index": np.arange(rows, dtype=np.int64),
        "query_row": np.arange(1000, 1000 + rows, dtype=np.int64),
        "query_ik14": [f"I{index % 17:02d}" for index in range(rows)],
        "query_formula": [f"F{index % 23:02d}" for index in range(rows)],
        "selector": ["candidate_gradient" if index % 2 == 0 else "role_confounder"
                     for index in range(rows)],
        "attenuation": np.full(rows, 0.50),
        "step": np.full(rows, 3, dtype=np.int8),
        "target_path": ["0,1,2"] * rows,
        "matched_control_paths": ["3,4,5;6,7,8"] * rows,
        "hard_negative_row": np.arange(2000, 2000 + rows, dtype=np.int64),
        "formula_fold": np.arange(rows, dtype=np.int64) % 5,
    })


def example_rows(frame: pd.DataFrame) -> list[DirectExample]:
    return [
        DirectExample(
            query_index=int(row.query_index), query_row=int(row.query_row),
            identity=str(row.query_ik14), formula=str(row.query_formula),
            positive_rows=(10,), negative_rows=(11,), official_margin=-0.1,
            official_rank=2, sample_weight=1.0,
            policy=f"{row.selector}|step={int(row.step)}",
            target_path=tuple(int(value) for value in str(row.target_path).split(",") if value),
            attenuation=float(row.attenuation),
        )
        for row in frame.itertuples(index=False)
    ]


def exact_args() -> SimpleNamespace:
    return SimpleNamespace(
        causal_arm="targeted", action_selection="fixed", policy="curriculum",
        action_scope="all", outer_fold=0, formula_fold_seed=20260825,
        epochs=4, batch_actions=4, views_per_identity=4,
        error_views_per_identity=0, positive_spectra=4, negative_molecules=8,
        unfreeze_blocks=1, direct_transfer_mode="symmetric",
        rank_reference_mode="shared", guided_noise_policy="none",
        backbone_lr=2e-6, head_lr=1e-5, weight_decay=1e-4,
        rank_margin=0.05, temperature=0.10, lambda_clean_rank=1.0,
        lambda_aug_rank=1.0, lambda_consistency=0.25,
        lambda_margin_floor=2.0, lambda_preserve=5.0,
        margin_floor_slack=0.005, safety_ratio=1.0,
        safety_stream_weight=1.0, positive_stream_weight=0.0,
        grad_clip=1.0, initial_student_checkpoint=None, amp=False,
        run_suffix="unit",
    )


def main() -> None:
    source = synthetic_actions()
    target, target_audit = materialize_causal_arm(source, "targeted")
    random_first, random_audit = materialize_causal_arm(source, "matched_random")
    random_second, _ = materialize_causal_arm(source, "matched_random")
    clean, clean_audit = materialize_causal_arm(source, "clean_duplicate")

    invariant = [
        "query_index", "query_row", "query_ik14", "query_formula", "selector",
        "attenuation", "step", "hard_negative_row", "formula_fold",
    ]
    for frame in (target, random_first, clean):
        if not frame[invariant].equals(source[invariant]):
            raise RuntimeError("causal arm changed a sampler or candidate invariant")
    if target["target_path"].tolist() != source["target_path"].tolist():
        raise RuntimeError("targeted arm did not preserve frozen target paths")
    if clean["target_path"].astype(str).ne("").any():
        raise RuntimeError("clean-duplicate arm contains a non-empty action path")
    allowed = {"3,4,5", "6,7,8"}
    if not set(random_first["target_path"].astype(str)) <= allowed:
        raise RuntimeError("matched-random arm used a path outside frozen controls")
    if not random_first["target_path"].equals(random_second["target_path"]):
        raise RuntimeError("matched-control assignment is not deterministic")
    counts = random_audit.get("matched_control_index_counts", {})
    if set(counts) != {"0", "1"}:
        raise RuntimeError("matched-control assignment did not exercise both controls")
    if target_audit["rows"] != random_audit["rows"] or target_audit["rows"] != clean_audit["rows"]:
        raise RuntimeError("causal arms have unequal row counts")

    schedules = []
    for frame in (target, random_first, clean):
        epoch = identity_balanced_epoch(
            example_rows(frame), np.random.default_rng(20260901), 4,
        )
        schedules.append([(item.query_index, item.identity, item.policy) for item in epoch])
    if schedules[0] != schedules[1] or schedules[0] != schedules[2]:
        raise RuntimeError("causal arm changed the identity-balanced batch schedule")

    clean_spectrum = torch.tensor([[0.0, 100.0, 200.0], [0.0, 0.2, 0.8]])
    if not torch.equal(attenuate_sequence(clean_spectrum, (), 0.5), clean_spectrum):
        raise RuntimeError("empty causal action path is not an exact clean duplicate")

    args = exact_args()
    validate_causal_configuration(args)
    args.policy = "candidate"
    try:
        validate_causal_configuration(args)
    except ValueError:
        pass
    else:
        raise RuntimeError("causal configuration failed to reject policy drift")

    reference = pd.DataFrame({
        "final_rank": [2, 1, 2, 1], "query_formula": ["A", "A", "B", "B"],
        "has_near": [True, True, False, False],
        "final_top_molecule_local": [1, 0, 2, 0],
        "final_full_margin": [-0.2, 0.1, -0.3, 0.2],
    })
    treatment = pd.DataFrame({
        "final_rank": [1, 1, 2, 2], "query_formula": ["A", "A", "B", "B"],
        "has_near": [True, True, False, False],
        "final_top_molecule_local": [0, 0, 3, 1],
        "final_full_margin": [0.1, 0.2, -0.1, -0.1],
    })
    paired = paired_summary(reference, treatment, 100, 11)
    if paired["corrected"] != 1 or paired["introduced"] != 1:
        raise RuntimeError("paired causal summary miscounted transitions")
    if paired["top_molecule_changed"] != 3 or paired["wrong_to_different_wrong"] != 1:
        raise RuntimeError("paired causal summary miscounted candidate switches")

    sbatch = (
        Path(__file__).resolve().parent / "run_noise_final_e4a_causal_attribution.sbatch"
    ).read_text(encoding="utf-8")
    required_sbatch = (
        "#SBATCH --array=0-2", "#SBATCH --gpus=1",
        "ARMS=(clean_duplicate matched_random targeted)",
        "--causal-arm \"$ARM\"", "--policy curriculum",
        "--action-selection fixed", "--action-scope all",
        "--guided-noise-policy none", "--rank-reference-mode shared",
        "summarize_noise_final_e4a_causal_attribution.py",
        "flock -n 9",
    )
    missing = [value for value in required_sbatch if value not in sbatch]
    if missing:
        raise RuntimeError(f"causal sbatch contract is incomplete: {missing}")
    recovery = (
        Path(__file__).resolve().parent
        / "run_noise_final_e4a_causal_attribution_summary.sbatch"
    ).read_text(encoding="utf-8")
    required_recovery = (
        "#SBATCH --gpus=1", "summarize_noise_final_e4a_causal_attribution.py",
        "validate_noise_final_e4a_causal_attribution.py",
    )
    missing_recovery = [value for value in required_recovery if value not in recovery]
    if missing_recovery:
        raise RuntimeError(f"causal summary recovery contract is incomplete: {missing_recovery}")
    summarizer = (
        Path(__file__).resolve().parent
        / "summarize_noise_final_e4a_causal_attribution.py"
    ).read_text(encoding="utf-8")
    if ".tmp_{os.getpid()}" not in summarizer or "temporary_output.rename(output)" not in summarizer:
        raise RuntimeError("causal summary is not atomically published")

    print("[test_noise_final_e4a_causal_attribution] PASS")


if __name__ == "__main__":
    main()

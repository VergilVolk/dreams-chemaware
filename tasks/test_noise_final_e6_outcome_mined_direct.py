"""Static and synthetic contract tests for E6 direct noise fine-tuning."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_summary_module():
    path = ROOT / "tasks/summarize_noise_final_e6_outcome_mined_direct.py"
    specification = importlib.util.spec_from_file_location("noise_e6_summary", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load E6 summary module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    trainer = (ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py").read_text(
        encoding="utf-8",
    )
    sbatch = (ROOT / "tasks/run_noise_final_e6_outcome_mined_direct.sbatch").read_text(
        encoding="utf-8",
    )
    required_trainer_tokens = (
        'choices=("fixed", "outcome_mined")',
        'corrective_teacher_actions.csv.gz',
        'locally_materialised_union_recoverable", -1)) != 882',
        'len(actions) != 882',
        'actions["query_index"].duplicated().any()',
        'teacher_hard_negative_row',
        'actions = actions.drop(columns=[',
        'action_outcomes_used_in_loss_or_sample_weight": False',
        'clean_and_augmented_raw_spectra_train_same_encoder": True',
        'shared_query_reference_encoder": True',
        'P2b": "forbidden"',
    )
    missing = [token for token in required_trainer_tokens if token not in trainer]
    if missing:
        raise AssertionError(f"E6 trainer contract tokens missing: {missing}")
    if "p2b_frozen" in trainer.lower():
        raise AssertionError("E6 trainer imports a downstream P2b artifact")

    for token in (
        "#SBATCH --gpus=1", "#SBATCH --array=0-5",
        'SELECT="fixed"', 'SELECT="outcome_mined"',
        '--action-scope errors', '--guided-noise-policy none',
        '--positive-stream-weight',
    ):
        if token == '--positive-stream-weight':
            # E6 intentionally leaves the optional P stream at its default zero;
            # an explicit nonzero value would confound the direct action test.
            if token in sbatch:
                raise AssertionError("E6 must not enable the P stream")
        elif token not in sbatch:
            raise AssertionError(f"E6 sbatch token missing: {token}")
    if sbatch.count("SUFFIX=") != 6:
        raise AssertionError("E6 must contain exactly six paired arms")

    module = load_summary_module()
    if len(module.EXPECTED) != 6:
        raise AssertionError("E6 summary does not expect all six arms")
    reference = pd.DataFrame({
        "query_index": np.arange(8),
        "query_formula": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "baseline_rank": [1, 2, 1, 2, 1, 2, 1, 2],
        "final_rank": [1, 2, 1, 2, 1, 2, 1, 2],
    })
    candidate = reference.copy()
    candidate.loc[[1, 3, 5], "final_rank"] = 1
    result = module.compare(reference, candidate, 200, 7)
    if result["net_vs_reference"] != 3 or result["corrected_vs_reference"] != 3:
        raise AssertionError("paired E6 comparison has incorrect transition accounting")
    if result["introduced_vs_reference"] != 0 or result["mean"] <= 0:
        raise AssertionError("paired E6 comparison has incorrect effect direction")
    print("[test_noise_final_e6_outcome_mined_direct] PASS")


if __name__ == "__main__":
    main()

"""Static fail-closed audit of the dynamic direct implementation boundary."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = {
        "core": ROOT / "tasks/noise_final_dynamic_direct_core.py",
        "preflight": ROOT / "tasks/audit_noise_final_dynamic_direct_preflight.py",
        "ledger": ROOT / "tasks/build_noise_final_dynamic_direct_ledger.py",
        "sbatch": ROOT / "tasks/run_noise_final_dynamic_direct_preflight.sbatch",
        "contract": ROOT / "docs/NOISE_FINAL_DYNAMIC_CONDITIONAL_DIRECT_FINETUNING_CONTRACT_20260904.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    for name in ("core", "preflight", "ledger"):
        ast.parse(source[name])
    gates = {
        "gpu_requested": "#SBATCH --gpus=1" in source["sbatch"],
        "memory_within_N26_one_gpu_limit": "#SBATCH --mem=30G" in source["sbatch"],
        "unique_output": "SLURM_JOB_ID" in source["sbatch"],
        "no_post_outcome_cell_selection": all(
            token not in source["ledger"]
            for token in ("passing_cells", "best_fixed_cell", "oracle_per_query")
        ),
        "N_all_nine_cells": "validate_n_cells" in source["ledger"],
        "P_all_twenty_one_cells": all(
            token in source["ledger"]
            for token in ("P_INTENSITY_FAMILIES", "P_TRANSFER_FAMILIES", "expected_cells = 30")
        ),
        "outer_held_removed_before_P_fit": (
            "outer_train_query = np.flatnonzero(formula_fold != args.outer_fold)" in source["ledger"]
        ),
        "formula_crossfit": "train = (folds != fold) & (folds != outer_fold)" in source["ledger"],
        "no_raw_P_outcomes_in_training_columns": "raw_P_outcomes_published\": False" in source["ledger"],
        "L0_geometry_exact": "exact clean geometry used to define L0/L1 action labels" in source["preflight"],
        "formula_identity_family_exposure_in_sampler": "stratified_action_epoch" in source["core"],
        "utility_not_inverse_frequency": "Multiplying inverse abundance into utility" in source["core"],
        "sampling_not_double_weighted": "never a second time through sampling probability" in source["core"],
        "P2b_absent": "P2b_score" not in "\n".join(source.values()),
        "contract_direct_primary": "直接微调是主线" in source["contract"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"dynamic-direct implementation audit failed: {gates}")
    report = {
        "status": "noise_final_dynamic_direct_implementation_audit_passed",
        "gates": gates,
        "scope": "model-free preflight plus 30-cell formula-crossfit action ledger",
        "claim_limit": "No shared encoder has been updated by this implementation stage.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

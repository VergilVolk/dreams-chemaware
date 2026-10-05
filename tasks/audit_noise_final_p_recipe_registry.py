"""Freeze reusable P recipes without importing historical outcome labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from audit_noise_final_e10_positive_residual_matrix import (
    REFERENCE_KINDS, positive_cells,
)
from audit_noise_final_e11_reference_diversity_matrix import (
    RECIPES as E11_RECIPES, REFERENCE_POLICIES,
)
from audit_noise_final_e12b_relaxed_recurrence_matrix import (
    POLICIES as E12_POLICIES, RECIPES as E12_RECIPES,
)
from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e8-checkpoint", type=Path,
        default=ROOT / (
            "data/validation/g8r_noise_final_e8_direct_transfer/"
            "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
            "e8_baseline_symmetric_shared/seed_20260830/fold_0/"
            "final_shared_encoder.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def recipe(family: str, dose: float, auxiliary: float) -> dict[str, object]:
    return {"family": family, "dose": float(dose), "auxiliary_dose": float(auxiliary)}


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    source_paths = [
        ROOT / "tasks/audit_noise_final_e10_positive_residual_matrix.py",
        ROOT / "tasks/audit_noise_final_e11_reference_diversity_matrix.py",
        ROOT / "tasks/audit_noise_final_e12b_relaxed_recurrence_matrix.py",
    ]
    if not args.e8_checkpoint.is_file() or any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("P recipe source or mature E8 checkpoint is missing")
    e10 = [recipe(*cell) for cell in positive_cells("expanded")]
    e11 = [
        {"reference_policy": policy, **recipe(*cell)}
        for policy in REFERENCE_POLICIES for cell in E11_RECIPES
    ]
    e12 = [
        {
            "reference_policy": policy, "maximum_transferred_peaks": int(maximum),
            "dose": float(dose), "support_weighted": bool(weighted),
            "minimum_reference_prevalence": 0.50,
        }
        for policy in E12_POLICIES for maximum, dose, weighted in E12_RECIPES
    ]
    body = {
        "status": "noise_final_p_recipe_registry_complete",
        "e10b": {
            "recipes": e10, "reference_kinds": list(REFERENCE_KINDS),
            "positive_cells": len(e10), "action_columns": len(e10) * len(REFERENCE_KINDS),
        },
        "e11": {"cells": e11, "cell_count": len(e11)},
        "e12b": {"cells": e12, "cell_count": len(e12)},
        "contracts": {
            "recipe_and_reference_selection_only": True,
            "historical_outcome_matrix_read": False,
            "historical_best_cell_read": False,
            "outer_held_label_read": False,
            "same_identity_references_required": True,
            "wrong_identity_reference_is_direction_control_only": True,
            "formal_training_requires_outer_train_rematerialization": True,
        },
        "provenance": {
            "source_sha256": {path.name: sha256_file(path) for path in source_paths},
            "mature_e8_checkpoint": str(args.e8_checkpoint),
            "mature_e8_checkpoint_sha256": sha256_file(args.e8_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Recipe registry only; contains no P action outcome or embedding result.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p_recipe_registry_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()

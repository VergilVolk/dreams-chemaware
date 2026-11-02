#!/usr/bin/env python3
"""Test remaining representation repairs directly on selected Full23 v1.3.1.

The harmful widened-bound intervention is deliberately absent. Each arm keeps
the selected ``01`` recipe (multiplier 1, raw-residual penalty, original bound)
and adds exactly one representation change. Arms are independently initialized
with the same seed and are never warm-started from one another.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
import json
import os
from pathlib import Path

import pandas as pd
import torch

from full23_tensor_set_readout_core import SCHEMA, TensorSetSettings
from train_enveda_unified_candidate_integrator import _group_occurrences, _occurrences
from train_full23_multireference_readout import (
    FULL23_OOF_SHA256,
    _embedding_score_alignment,
    _load_banks,
    _make_representation_panels,
    _reference_multiplicity,
    _training_input_hashes,
    arguments,
)
from train_full23_tensor_set_readout import EPOCHS, TRAIN_SEED
from train_full23_transfer_safe_tensor_readout import ARMS, fit_arm
from train_full23_v13_cumulative_repairs import (
    _atomic_json,
    _atomic_torch_save,
    _load_existing,
    training_bounds,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


ARM_ORDER = ("013", "014", "015")


@dataclass(frozen=True)
class ArmSpec:
    settings: TensorSetSettings
    intervention: str


def direct_arm_specs() -> dict[str, ArmSpec]:
    base = ARMS["no_count_query_centered"]
    return {
        "013": ArmSpec(
            replace(base, include_anchor_context=True),
            "add_query_relative_full23_anchor_context_directly_on_01",
        ),
        "014": ArmSpec(
            replace(base, pair_dimension=128),
            "increase_pair_dimension_64_to_128_directly_on_01",
        ),
        "015": ArmSpec(
            replace(base, include_signed_difference=True),
            "add_signed_query_minus_reference_channel_directly_on_01",
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.full23_oof is None:
        raise RuntimeError("training requires Full23 v1 formula-OOF parent")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 anchor identity drifted")
    if sha256_file(args.full23_oof) != FULL23_OOF_SHA256:
        raise RuntimeError("Full23 v1 OOF parent identity drifted")

    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(
        args, validate_checkpoint(args.full23_checkpoint), cpu_banks
    )
    bounds, bound_report = training_bounds(panels)
    residual_bound = bounds["legacy_median"]
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    groups = _group_occurrences(panels, occurrences)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()

    specs = direct_arm_specs()
    common_interventions = (
        "anchor_correct_multiplier_2_to_1",
        "penalize_query_centered_raw_residual",
    )
    partial = Path(str(args.output) + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(exist_ok=True)
    contract = json.loads(json.dumps({
        "schema": "full23_v131_direct_repairs_v1",
        "parent_recipe": "v13_01",
        "arm_order": list(ARM_ORDER),
        "arms": {
            name: {
                "settings": asdict(spec.settings),
                "residual_penalty_space": "raw",
                "bound_rule": "legacy_median",
                "interventions": [*common_interventions, spec.intervention],
            }
            for name, spec in specs.items()
        },
        "common_training": {
            "source": "Enveda-180 consumed meta-development",
            "epochs": EPOCHS,
            "seed": TRAIN_SEED,
            "anchor_correct_multiplier": 1.0,
            "warm_start": False,
        },
    }))
    contract_path = partial / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("partial direct-repair contract drifted")
    else:
        _atomic_json(contract, contract_path)

    upstream = _training_input_hashes(args)
    summaries = {}
    for arm in ARM_ORDER:
        spec = specs[arm]
        interventions = (*common_interventions, spec.intervention)
        checkpoint = partial / f"v13_{arm}.pt"
        if checkpoint.exists():
            payload = _load_existing(checkpoint)
            if (
                payload.get("model_role") != f"Full23_v131_direct_{arm}"
                or tuple(payload.get("interventions", ())) != interventions
            ):
                raise RuntimeError(f"partial checkpoint drifted for arm {arm}")
            summaries[arm] = payload["training_summary"]
            print(json.dumps({"arm": arm, "status": "RESUMED"}), flush=True)
            continue

        model, history = fit_arm(
            spec.settings,
            panels,
            occurrences,
            banks,
            args.device,
            residual_bound,
            anchor_correct_multiplier=1.0,
            residual_penalty_space="raw",
            arm_name=f"v13_{arm}",
            groups=groups,
        )
        summary = {
            "settings": asdict(spec.settings),
            "interventions": list(interventions),
            "residual_penalty_space": "raw",
            "bound_rule": "legacy_median",
            "residual_bound": residual_bound,
            "trainable_parameters": model.trainable_parameters,
            "training_history": history,
        }
        payload = {
            "schema": SCHEMA,
            "model_role": f"Full23_v131_direct_{arm}",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "full23_parent_oof_sha256": FULL23_OOF_SHA256,
            "official_embedding_direct_input": False,
            "settings": asdict(spec.settings),
            "residual_bound": residual_bound,
            "residual_bound_calibration": bound_report["legacy"],
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "interventions": interventions,
            "training_summary": summary,
            "training": {
                "source": "Enveda-180 consumed meta-development",
                "epochs": EPOCHS,
                "seed": TRAIN_SEED,
                "warm_start": False,
                "anchor_correct_multiplier": 1.0,
            },
            "upstream_sha256": upstream,
        }
        _atomic_torch_save(payload, checkpoint)
        summaries[arm] = summary
        _atomic_json(
            {
                "status": "FULL23_V131_DIRECT_REPAIRS_IN_PROGRESS",
                "completed_arms": [
                    name
                    for name in ARM_ORDER
                    if (partial / f"v13_{name}.pt").exists()
                ],
            },
            partial / "progress.json",
        )
        del model, payload
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    report = {
        "status": "FULL23_V131_DIRECT_REPAIRS_TRAINING_COMPLETE",
        "dataset": {
            "name": "Enveda-180",
            "role": "consumed meta-development",
            "queries": {
                name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
            },
        },
        "parent_recipe": "v13_01",
        "arm_order": list(ARM_ORDER),
        "arms": summaries,
        "bound_calibration": bound_report["legacy"],
        "reference_multiplicity": _reference_multiplicity(panels),
        "embedding_score_alignment_max_abs_drift": _embedding_score_alignment(panels),
        "cache_policy": "panels and four embedding banks loaded once; arms trained sequentially",
        "gnps_access": "none during fitting",
    }
    _atomic_json(report, partial / "report.json")
    (partial / "progress.json").unlink(missing_ok=True)
    os.replace(partial, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

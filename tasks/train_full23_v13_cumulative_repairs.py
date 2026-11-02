#!/usr/bin/env python3
"""Train six fixed cumulative repairs directly on formal Full23 v1.3.

The arms are nested by construction: 0, 01, 012, 0123, 01234, 012345.
Every arm starts independently from the same seed; later arms do not receive
extra optimization steps by warm-starting from earlier arms.
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
    calibrate_training_residual_bound,
)
from train_full23_tensor_set_readout import EPOCHS, TRAIN_SEED
from train_full23_transfer_safe_tensor_readout import ARMS, fit_arm
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


ARM_ORDER = ("0", "01", "012", "0123", "01234", "012345")


@dataclass(frozen=True)
class ArmSpec:
    settings: TensorSetSettings
    residual_penalty_space: str
    bound_rule: str
    interventions: tuple[str, ...]


def cumulative_arm_specs() -> dict[str, ArmSpec]:
    base = ARMS["no_count_query_centered"]
    loss = "anchor_correct_multiplier_2_to_1"
    raw = "penalize_query_centered_raw_residual"
    bound = "expand_pairwise_capacity_to_training_error_q75"
    context = "add_query_relative_full23_anchor_context"
    width = "increase_pair_dimension_64_to_128"
    signed = "add_signed_query_minus_reference_channel"
    return {
        "0": ArmSpec(base, "bounded", "legacy_median", (loss,)),
        "01": ArmSpec(base, "raw", "legacy_median", (loss, raw)),
        "012": ArmSpec(base, "raw", "error_q75", (loss, raw, bound)),
        "0123": ArmSpec(
            replace(base, include_anchor_context=True), "raw", "error_q75",
            (loss, raw, bound, context),
        ),
        "01234": ArmSpec(
            replace(base, include_anchor_context=True, pair_dimension=128),
            "raw", "error_q75", (loss, raw, bound, context, width),
        ),
        "012345": ArmSpec(
            replace(
                base, include_anchor_context=True, pair_dimension=128,
                include_signed_difference=True,
            ),
            "raw", "error_q75", (loss, raw, bound, context, width, signed),
        ),
    }


def training_bounds(panels) -> tuple[dict[str, float], dict]:
    legacy, calibration = calibrate_training_residual_bound(panels)
    error_q75 = float(calibration["weighted_parent_error_deficit"]["q75"])
    correct_median = float(calibration["weighted_correct_parent_margin"]["median"])
    expanded_capacity = min(error_q75, correct_median)
    expanded = 0.5 * expanded_capacity
    if not expanded > legacy:
        raise RuntimeError("fixed expanded bound is not wider than formal v1.3 bound")
    return {"legacy_median": legacy, "error_q75": expanded}, {
        "legacy": calibration,
        "expanded": {
            "rule": "half_min_training_error_q75_and_correct_margin_median_v1",
            "source": "Enveda-180 five-fold formula-OOF Full23 parent only",
            "gnps_accessed": False,
            "residual_bound_per_candidate": expanded,
            "maximum_pairwise_shift": expanded_capacity,
            "training_error_q75": error_q75,
            "training_correct_margin_median_cap": correct_median,
        },
    }


def _load_existing(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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
        args, validate_checkpoint(args.full23_checkpoint), cpu_banks,
    )
    bounds, bound_report = training_bounds(panels)
    occurrences = _occurrences({name: panel.base for name, panel in panels.items()})
    groups = _group_occurrences(panels, occurrences)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    banks = {
        name: torch.as_tensor(values, dtype=dtype, device=args.device)
        for name, (_, values) in cpu_banks.items()
    }
    del cpu_banks
    gc.collect()

    specs = cumulative_arm_specs()
    partial = Path(str(args.output) + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(exist_ok=True)
    contract = {
        "schema": "full23_v13_cumulative_repairs_v1",
        "arm_order": list(ARM_ORDER),
        "arms": {
            name: {
                "settings": asdict(specs[name].settings),
                "residual_penalty_space": specs[name].residual_penalty_space,
                "bound_rule": specs[name].bound_rule,
                "interventions": list(specs[name].interventions),
            }
            for name in ARM_ORDER
        },
        "common_training": {
            "source": "Enveda-180 consumed meta-development",
            "epochs": EPOCHS,
            "seed": TRAIN_SEED,
            "anchor_correct_multiplier": 1.0,
            "warm_start": False,
        },
    }
    # Normalize tuples to their JSON representation so a resumed run compares
    # the in-memory contract with the on-disk contract without false drift.
    contract = json.loads(json.dumps(contract))
    contract_path = partial / "contract.json"
    if contract_path.exists():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError("partial cumulative run contract differs from requested run")
    else:
        _atomic_json(contract, contract_path)

    summaries = {}
    upstream = _training_input_hashes(args)
    for arm in ARM_ORDER:
        spec = specs[arm]
        checkpoint = partial / f"v13_{arm}.pt"
        if checkpoint.exists():
            payload = _load_existing(checkpoint)
            if (
                payload.get("model_role") != f"Full23_v13_cumulative_{arm}"
                or tuple(payload.get("interventions", ())) != spec.interventions
            ):
                raise RuntimeError(f"partial checkpoint contract drifted for arm {arm}")
            summaries[arm] = payload["training_summary"]
            print(json.dumps({"arm": arm, "status": "RESUMED"}), flush=True)
            continue

        model, history = fit_arm(
            spec.settings, panels, occurrences, banks, args.device,
            bounds[spec.bound_rule], anchor_correct_multiplier=1.0,
            residual_penalty_space=spec.residual_penalty_space,
            arm_name=f"v13_{arm}", groups=groups,
        )
        summary = {
            "settings": asdict(spec.settings),
            "interventions": list(spec.interventions),
            "residual_penalty_space": spec.residual_penalty_space,
            "bound_rule": spec.bound_rule,
            "residual_bound": bounds[spec.bound_rule],
            "trainable_parameters": model.trainable_parameters,
            "training_history": history,
        }
        payload = {
            "schema": SCHEMA,
            "model_role": f"Full23_v13_cumulative_{arm}",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "full23_parent_oof_sha256": FULL23_OOF_SHA256,
            "official_embedding_direct_input": False,
            "settings": asdict(spec.settings),
            "residual_bound": bounds[spec.bound_rule],
            "residual_bound_calibration": bound_report[
                {"legacy_median": "legacy", "error_q75": "expanded"}[
                    spec.bound_rule
                ]
            ],
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "interventions": spec.interventions,
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
        _atomic_json({
            "status": "FULL23_V13_CUMULATIVE_REPAIRS_IN_PROGRESS",
            "completed_arms": [name for name in ARM_ORDER if (partial / f"v13_{name}.pt").exists()],
        }, partial / "progress.json")
        del model, payload
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    report = {
        "status": "FULL23_V13_CUMULATIVE_REPAIRS_TRAINING_COMPLETE",
        "dataset": {
            "name": "Enveda-180",
            "role": "consumed meta-development",
            "queries": {
                name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
            },
        },
        "arm_order": list(ARM_ORDER),
        "arms": summaries,
        "bound_calibration": bound_report,
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

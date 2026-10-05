"""Multi-action direct boundary transfer into one clean-input shared encoder.

This is the corrected successor to the compressed E14/E15 route.  Every
optimizer action batch is query-complete, keeps the exact action set selected
by its registered upstream supplier, and transfers only the conservative action-vs-clean AND
action-vs-direction-control margin gain to the deployed clean embedding.
Harmful actions are routing evidence for a separate clean-risk branch and are
never encoded as positive imitation targets.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from audit_noise_peak_gate_candidate_injection import formula_ci
from noise_corrected_fullgraph_evaluation import (
    full_metrics, held_metric_evidence, official_scores, paired_outcome_table,
    score_embedding_query_subset, score_embeddings,
)
from noise_final_core import CandidateGraph, seed_everything, sha256_file, stable_fold
from noise_final_direct_boundary_v2_core import direct_boundary_objective
from noise_corrected_direct_v3_core import (
    balanced_action_step_plan,
    cap_safe_corrective_repartition,
    corrective_recycle_scale,
    corrective_action_objective,
    harmful_boundary_objective,
    robust_action_objective,
    symmetric_live_action_consistency,
    v3_schedule_geometry,
)
from noise_corrected_shuffled_control_v3 import source_family_shuffled_action_bank
from noise_corrected_directional_update_v4 import rotate_update_toward_action
from noise_corrected_update_arbitration_v4 import (
    arbitrate_corrective_optimizer_updates_by_group,
    arbitrate_optimizer_updates,
    arbitrate_optimizer_updates_by_group,
    materialize_descent_updates_,
    reconcile_adamw_first_moments_to_materialized_updates_,
)
from noise_action_injector_v1 import (
    ActionInjectorV1,
    ActionInjectorV1Config,
    action_injector_v1_contract_manifest,
)
from noise_historical_best_action_bank_v1 import (
    HistoricalBestActionBankV1,
    HistoricalBestActionBankV1Config,
    historical_best_action_bank_v1_contract_manifest,
)
from noise_corrected_gradient_locality_v8 import (
    CORRECTIVE_GRADIENT_LOCALITIES,
    corrective_embedding_role_report,
    corrective_gradient_locality,
)
from noise_corrected_signal_v9 import (
    CORRECTIVE_BRANCH_MODES,
    corrective_branch_policy,
    minimum_gate_passed,
    restoration_coverage_gate,
)
from noise_final_e15_core import project_corrective_against_risk
from noise_final_e4_pmt_core import select_materialized_direct_action_panel
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


@dataclass(frozen=True)
class RoutedAction:
    action_id: str
    tensor_index: int
    group: str
    control_semantic: str = "matched_neutral"
    mechanism_weight: float = 1.0


@dataclass(frozen=True)
class BoundaryExample:
    query_index: int
    query_row: int
    identity: str
    formula: str
    actions: tuple[RoutedAction, ...]
    positive_rows: tuple[int, ...]
    negative_rows: tuple[tuple[int, ...], ...]
    epoch_weight: float = 1.0


@dataclass(frozen=True)
class ProtectExample:
    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[tuple[int, ...], ...]
    epoch_weight: float = 1.0


def _mechanism_block(source: str) -> str:
    source = str(source)
    if source in {"P_guided_original", "E10B", "E11", "E12B"}:
        return "P"
    if source in {"N_mature", "V4_gradient_path"}:
        return "N"
    if source == "A4_exact":
        return "A4"
    raise RuntimeError(f"unregistered routed action source: {source}")


def _global_mechanism_weights(
    actions: pd.DataFrame,
    query_weights: dict[int, float],
) -> tuple[dict[str, float], dict[str, object]]:
    """Equalize total N/P/A4 mass across queries without dropping actions.

    In-query family averaging alone cannot stop a high-coverage mechanism from
    dominating through many mechanism-only queries.  Each mechanism receives
    coefficient Q/(M*weighted_query_incidence), so its cumulative coefficient
    after frozen identity weights is exactly Q/M.  Absolute branch scale stays
    one and is still calibrated before training.
    """
    if actions.empty:
        return {}, {
            "mechanisms": [], "query_weight_sum": 0.0,
            "weighted_query_incidence": {}, "mechanism_coefficients": {},
            "effective_epoch_mass": {}, "balanced": True,
        }
    required = {"query_index", "source"}
    if missing := required - set(actions.columns):
        raise RuntimeError(f"mechanism balancing misses {sorted(missing)}")
    incidence: dict[str, float] = defaultdict(float)
    for query, block in actions.groupby("query_index", sort=True):
        query = int(query)
        if query not in query_weights:
            raise RuntimeError("mechanism balancing query lacks identity weight")
        for mechanism in {
            _mechanism_block(str(source)) for source in block.source.astype(str)
        }:
            incidence[mechanism] += float(query_weights[query])
    if not incidence or any(not np.isfinite(value) or value <= 0 for value in incidence.values()):
        raise RuntimeError("mechanism query incidence is empty or invalid")
    total_query_weight = float(sum(query_weights.values()))
    mechanisms = sorted(incidence)
    coefficients = {
        mechanism: total_query_weight / (len(mechanisms) * exposure)
        for mechanism, exposure in incidence.items()
    }
    effective = {
        mechanism: incidence[mechanism] * coefficients[mechanism]
        for mechanism in mechanisms
    }
    balanced = bool(
        np.allclose(list(effective.values()), total_query_weight / len(mechanisms),
                    rtol=1e-12, atol=1e-12)
    )
    if not balanced:
        raise RuntimeError("global mechanism mass equalization failed")
    return coefficients, {
        "mechanisms": mechanisms,
        "query_weight_sum": total_query_weight,
        "weighted_query_incidence": incidence,
        "mechanism_coefficients": coefficients,
        "effective_epoch_mass": effective,
        "target_epoch_mass_per_mechanism": total_query_weight / len(mechanisms),
        "balanced": balanced,
        "action_or_query_dropped": False,
    }


def _panel_mechanism_balance_report(
    examples: list[BoundaryExample],
) -> dict[str, object]:
    mechanisms = sorted({
        action.group.split("::", 1)[0]
        for example in examples for action in example.actions
    })
    effective = {
        mechanism: float(sum(
            example.epoch_weight * next(
                action.mechanism_weight for action in example.actions
                if action.group.split("::", 1)[0] == mechanism
            )
            for example in examples
            if any(
                action.group.split("::", 1)[0] == mechanism
                for action in example.actions
            )
        ))
        for mechanism in mechanisms
    }
    coefficient_range = {
        mechanism: {
            "minimum": float(min(values)),
            "maximum": float(max(values)),
        }
        for mechanism in mechanisms
        for values in [[
            action.mechanism_weight
            for example in examples for action in example.actions
            if action.group.split("::", 1)[0] == mechanism
        ]]
    }
    return {
        "mechanisms": mechanisms,
        "mechanism_coefficient_range": coefficient_range,
        "maximum_mechanism_coefficient": max(
            (values["maximum"] for values in coefficient_range.values()),
            default=0.0,
        ),
        "effective_epoch_mass": effective,
        "balanced": bool(
            not effective
            or np.allclose(list(effective.values()), next(iter(effective.values())),
                           rtol=1e-12, atol=1e-12)
        ),
    }


def _weighted_mechanism_mass(
    examples: list[BoundaryExample],
    scale: float,
) -> Counter[str]:
    output: Counter[str] = Counter()
    for example in examples:
        by_mechanism: dict[str, float] = {}
        for action in example.actions:
            mechanism = action.group.split("::", 1)[0]
            previous = by_mechanism.setdefault(
                mechanism, float(action.mechanism_weight)
            )
            if not np.isclose(previous, action.mechanism_weight):
                raise RuntimeError("mechanism weight differs inside one query block")
        for mechanism, coefficient in by_mechanism.items():
            output[mechanism] += (
                float(scale) * float(example.epoch_weight) * coefficient
            )
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--routed-ledger-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument(
        "--arm",
        choices=("routed_direct", "shuffled_action_control", "clean_control"),
        default="routed_direct",
    )
    parser.add_argument(
        "--corrective-objective-mode", choices=("paired_margin_only", "full", "v3_direct"),
        default="paired_margin_only",
    )
    parser.add_argument(
        "--corrective-gradient-locality",
        choices=CORRECTIVE_GRADIENT_LOCALITIES,
        default="shared",
        help=(
            "Gradient paths for the action-specific corrective residual only. "
            "The E4/E8 protective and auxiliary objectives remain shared."
        ),
    )
    parser.add_argument(
        "--corrective-branch-mode",
        choices=CORRECTIVE_BRANCH_MODES,
        default="full_action_view",
        help=(
            "Use the complete live action-view corrective gradient, or retain "
            "only the historical scalar candidate-margin transfer. Calibration "
            "matches the resulting total corrective gradient to the same risk "
            "budget."
        ),
    )
    parser.add_argument(
        "--calibration-corrective-branch-mode",
        choices=("same_as_training", *CORRECTIVE_BRANCH_MODES),
        default="same_as_training",
        help=(
            "Optional common branch policy used only to freeze calibration "
            "scales. V10 calibrates both causal arms with full_action_view, "
            "then masks disabled training branches before the first update."
        ),
    )
    parser.add_argument(
        "--direct-contract",
        choices=(
            "v3", "best_action_v6", "best_action_v6_restored",
            "best_action_v7_corrective_restored",
            "best_action_v10_safe_exact",
            "best_action_v11_historical_best",
        ),
        default="v3",
    )
    parser.add_argument(
        "--corrective-admission", choices=("all", "strict_top1"), default="all",
    )
    parser.add_argument(
        "--action-bank-contract",
        choices=("all_strict_top1", "historical_best_v1"),
        default="all_strict_top1",
        help=(
            "Select every strict Top-1 view or the frozen one-champion-per-query "
            "historical-best action supplier. This option does not alter the injector."
        ),
    )
    parser.add_argument("--corrective-margin-floor", type=float, default=0.0)
    parser.add_argument(
        "--arm-invariant-targeted-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--inner-identity-folds", type=int, default=5)
    parser.add_argument("--inner-holdout-fold", type=int, default=-1)
    parser.add_argument(
        "--inner-holdout-unit",
        choices=("identity", "formula"),
        default="identity",
        help="V10 uses formula to test transfer before outer-held evaluation.",
    )
    parser.add_argument("--maximum-corrective-queries", type=int, default=0)
    parser.add_argument("--maximum-risk-queries", type=int, default=0)
    parser.add_argument("--maximum-robust-queries", type=int, default=0)
    parser.add_argument("--maximum-clean-queries", type=int, default=0)
    parser.add_argument("--outer-held-eval-queries", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--protective-batch-queries", type=int, default=8)
    parser.add_argument("--positive-references", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--references-per-negative", type=int, default=2)
    parser.add_argument("--action-conditioned-positive-references", type=int, default=32)
    parser.add_argument("--action-conditioned-negative-molecules", type=int, default=32)
    parser.add_argument("--reference-refresh-every-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--advantage-temperature", type=float, default=0.05)
    parser.add_argument("--topk-negatives", type=int, default=8)
    parser.add_argument("--action-safety-slack", type=float, default=0.005)
    parser.add_argument("--lambda-action-clean", type=float, default=0.0)
    parser.add_argument("--lambda-corrective-clean", type=float, default=0.0)
    parser.add_argument("--lambda-action-rank", type=float, default=0.25)
    parser.add_argument("--lambda-counterfactual", type=float, default=0.25)
    parser.add_argument("--lambda-margin-transfer", type=float, default=1.0)
    parser.add_argument("--lambda-action-safety", type=float, default=0.25)
    parser.add_argument("--lambda-v3-corrective-consistency", type=float, default=0.25)
    parser.add_argument("--lambda-v3-robust-rank", type=float, default=0.25)
    parser.add_argument("--lambda-v3-robust-floor", type=float, default=0.25)
    parser.add_argument("--lambda-v3-harmful-rank", type=float, default=1.0)
    parser.add_argument("--lambda-v3-harmful-floor", type=float, default=1.0)
    parser.add_argument("--v3-harmful-damage-slack", type=float, default=0.005)
    parser.add_argument("--target-v3-payload-to-transfer-ratio", type=float, default=0.25)
    parser.add_argument("--target-v3-consistency-to-transfer-ratio", type=float, default=0.25)
    parser.add_argument("--target-v3-robust-to-transfer-ratio", type=float, default=0.10)
    parser.add_argument("--target-v3-harmful-to-risk-ratio", type=float, default=0.25)
    parser.add_argument("--v3-branch-scale-cap", type=float, default=16.0)
    parser.add_argument("--minimum-v3-internal-combination-retention", type=float, default=0.50)
    parser.add_argument("--minimum-v3-active-transfer-fraction", type=float, default=0.05)
    parser.add_argument("--v3-maximum-protective-microbatches-per-step", type=int, default=4)
    parser.add_argument("--v3-maximum-auxiliary-microbatches-per-step", type=int, default=4)
    parser.add_argument("--v3-maximum-corrective-recycle-factor", type=float, default=4.0)
    parser.add_argument(
        "--v3-corrective-recycle-full-dose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--margin-transfer-fraction", type=float, default=0.50)
    parser.add_argument("--margin-transfer-cap", type=float, default=0.10)
    parser.add_argument(
        "--transfer-target-allocation",
        choices=("hard_cap", "mass_neutral_monotone"),
        default="hard_cap",
    )
    parser.add_argument("--maximum-transfer-cap-factor", type=float, default=2.0)
    parser.add_argument("--target-margin-transfer-to-other-corrective-ratio", type=float, default=1.0)
    parser.add_argument("--margin-transfer-scale-cap", type=float, default=64.0)
    parser.add_argument("--lambda-clean-continuation", type=float, default=1.0)
    parser.add_argument("--lambda-risk-floor", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--risk-margin-slack", type=float, default=0.005)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--target-corrective-to-risk-ratio", type=float, default=1.0)
    parser.add_argument("--corrective-scale-cap", type=float, default=256.0)
    parser.add_argument("--minimum-corrective-scale", type=float, default=0.50)
    parser.add_argument("--target-preclip-gradient-norm", type=float, default=0.05)
    parser.add_argument("--minimum-action-retention-p10", type=float, default=0.50)
    parser.add_argument("--minimum-optimizer-action-alignment-p10", type=float, default=0.05)
    parser.add_argument(
        "--minimum-optimizer-action-attributable-fraction-p10",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-corrective-direction-retention-p10", type=float, default=0.10,
    )
    parser.add_argument(
        "--v4-audit-minimum-attributable-fraction", type=float, default=0.25,
    )
    parser.add_argument("--v4-audit-maximum-action-gain", type=float, default=4.0)
    parser.add_argument(
        "--materialize-optimizer-update-restoration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--optimizer-restoration-scope",
        choices=(
            "composite_action", "corrective_only", "safe_exact_corrective",
        ),
        default="composite_action",
    )
    parser.add_argument(
        "--reconcile-restored-adamw-first-moment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--optimizer-restoration-minimum-risk-component-retention",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--minimum-optimizer-restoration-target-reached-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-safe-exact-update-norm-ratio",
        type=float,
        default=1.50,
        help=(
            "Fail closed if hard-safe exact-dose composition would exceed this "
            "multiple of the ordinary combined AdamW update norm."
        ),
    )
    parser.add_argument(
        "--continue-after-signal-gate-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--v4-audit-minimum-update-action-alignment", type=float, default=0.25,
    )
    parser.add_argument(
        "--v4-audit-maximum-update-action-coefficient", type=float, default=0.50,
    )
    parser.add_argument("--optimizer-attribution-steps-per-epoch", type=int, default=16)
    parser.add_argument(
        "--progress-every-steps", type=int, default=50,
        help="Emit a compact training heartbeat at this optimizer-step interval",
    )
    parser.add_argument("--maximum-clip-event-fraction", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument(
        "--maximum-spectra-per-action-forward", type=int, default=512,
        help=(
            "Fail before a query-complete action forward exceeds this many "
            "spectra; queries are never split silently"
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


REGISTERED_V3_EXACT_CONFIGURATION: dict[str, object] = {
    "direct_contract": "v3",
    "corrective_admission": "all",
    "action_bank_contract": "all_strict_top1",
    "arm_invariant_targeted_calibration": False,
    "outer_fold": 0,
    "formula_fold_seed": 20260825,
    "inner_identity_folds": 5,
    "inner_holdout_fold": -1,
    "inner_holdout_unit": "identity",
    "maximum_corrective_queries": 0,
    "maximum_risk_queries": 0,
    "maximum_robust_queries": 0,
    "maximum_clean_queries": 0,
    "outer_held_eval_queries": 0,
    "epochs": 4,
    "batch_queries": 4,
    "protective_batch_queries": 8,
    "positive_references": 4,
    "negative_molecules": 8,
    "references_per_negative": 2,
    "action_conditioned_positive_references": 32,
    "action_conditioned_negative_molecules": 32,
    "reference_refresh_every_epochs": 1,
    "unfreeze_blocks": 1,
    "topk_negatives": 8,
    "calibration_batches": 32,
    "optimizer_attribution_steps_per_epoch": 16,
    "progress_every_steps": 50,
    "n_highest_peaks": 100,
    "maximum_spectra_per_action_forward": 512,
    "eval_batch_size": 256,
    "bootstrap_resamples": 10000,
    "v3_maximum_protective_microbatches_per_step": 4,
    "v3_maximum_auxiliary_microbatches_per_step": 4,
    "v3_corrective_recycle_full_dose": True,
    "transfer_target_allocation": "hard_cap",
    "materialize_optimizer_update_restoration": False,
    "optimizer_restoration_scope": "composite_action",
    "reconcile_restored_adamw_first_moment": False,
    "continue_after_signal_gate_failure": False,
    "maximum_safe_exact_update_norm_ratio": 1.50,
    "corrective_gradient_locality": "shared",
    "corrective_branch_mode": "full_action_view",
    "calibration_corrective_branch_mode": "same_as_training",
    "amp": False,
}

REGISTERED_V3_FLOAT_CONFIGURATION = {
    "corrective_margin_floor": 0.0,
    "head_lr": 1e-5,
    "backbone_lr": 2e-6,
    "weight_decay": 1e-4,
    "rank_margin": 0.05,
    "temperature": 0.10,
    "action_safety_slack": 0.005,
    "v3_harmful_damage_slack": 0.005,
    "lambda_margin_transfer": 1.0,
    "lambda_action_rank": 0.25,
    "lambda_action_safety": 0.25,
    "lambda_v3_corrective_consistency": 0.25,
    "lambda_v3_robust_rank": 0.25,
    "lambda_v3_robust_floor": 0.25,
    "lambda_v3_harmful_rank": 1.0,
    "lambda_v3_harmful_floor": 1.0,
    "target_v3_payload_to_transfer_ratio": 0.25,
    "target_v3_consistency_to_transfer_ratio": 0.25,
    "target_v3_robust_to_transfer_ratio": 0.10,
    "target_v3_harmful_to_risk_ratio": 0.25,
    "v3_branch_scale_cap": 16.0,
    "minimum_v3_internal_combination_retention": 0.50,
    "minimum_v3_active_transfer_fraction": 0.05,
    "v3_maximum_corrective_recycle_factor": 4.0,
    "margin_transfer_fraction": 0.50,
    "margin_transfer_cap": 0.10,
    "maximum_transfer_cap_factor": 2.0,
    "lambda_clean_continuation": 1.0,
    "lambda_risk_floor": 2.0,
    "lambda_preserve": 5.0,
    "risk_margin_slack": 0.005,
    "target_corrective_to_risk_ratio": 1.0,
    "corrective_scale_cap": 256.0,
    "minimum_corrective_scale": 0.50,
    "target_preclip_gradient_norm": 0.05,
    "minimum_action_retention_p10": 0.50,
    "minimum_optimizer_action_alignment_p10": 0.05,
    "minimum_optimizer_action_attributable_fraction_p10": 0.10,
    "minimum_corrective_direction_retention_p10": 0.10,
    "v4_audit_minimum_attributable_fraction": 0.25,
    "v4_audit_maximum_action_gain": 4.0,
    "optimizer_restoration_minimum_risk_component_retention": 0.90,
    "v4_audit_minimum_update_action_alignment": 0.25,
    "v4_audit_maximum_update_action_coefficient": 0.50,
    "maximum_clip_event_fraction": 0.10,
    "grad_clip": 1.0,
}


REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION: dict[str, object] = {
    **REGISTERED_V3_EXACT_CONFIGURATION,
    "direct_contract": "best_action_v6",
    "corrective_admission": "strict_top1",
    # V6 is an exact one-variable splice: only the corrective action admission
    # changes.  Calibration and transfer allocation remain the registered V3
    # injector, including post-shuffle arm-specific norm calibration.
    "arm_invariant_targeted_calibration": False,
    "transfer_target_allocation": "hard_cap",
    "seed": 20260908,
}

REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION: dict[str, float] = {
    **REGISTERED_V3_FLOAT_CONFIGURATION,
    "corrective_margin_floor": 5e-6,
}


REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION: dict[str, object] = {
    **REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
    "direct_contract": "best_action_v6_restored",
    "materialize_optimizer_update_restoration": True,
    # A failed diagnostic must still yield its checkpoint and frozen held
    # metric ledger; promotion remains impossible while any signal gate fails.
    "continue_after_signal_gate_failure": True,
    "seed": 20260911,
}

REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION: dict[str, float] = {
    **REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
    "optimizer_restoration_minimum_risk_component_retention": 0.90,
}


REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION: dict[str, object] = {
    **REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION,
    "direct_contract": "best_action_v7_corrective_restored",
    # V7 restores only the conditional corrective increment.  The later V5
    # allocator is now admitted because its local ledger invariants are exact
    # and it removes the recurrent hard-cap compression without changing the
    # total per-query target mass or maximum individual target.
    "transfer_target_allocation": "mass_neutral_monotone",
    "optimizer_restoration_scope": "corrective_only",
    "reconcile_restored_adamw_first_moment": True,
    # Keep the V6R seed so the old and repaired executions differ only in the
    # registered injection bundle, not stochastic initialization/order.
    "seed": 20260911,
}


REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION: dict[str, float] = {
    **REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION,
    # A 0.25 materialized target must not be judged by the legacy 0.10 gate.
    "minimum_optimizer_action_attributable_fraction_p10": 0.25,
    # Permit isolated safety-limited steps to finish and be evaluated, while
    # requiring both optimizer groups to attain the target on >=90% of steps.
    "minimum_optimizer_restoration_target_reached_fraction": 0.90,
}


REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION: dict[str, object] = {
    **REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION,
    "direct_contract": "best_action_v10_safe_exact",
    "optimizer_restoration_scope": "safe_exact_corrective",
    "calibration_corrective_branch_mode": "full_action_view",
    "arm_invariant_targeted_calibration": True,
    "inner_holdout_unit": "formula",
    "inner_holdout_fold": 0,
    "seed": 20260913,
}


REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION: dict[str, float] = {
    **REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION,
    # Every active head/backbone update must attain the exact target.  There is
    # no longer an accepted ten-percent tail of below-target steps.
    "minimum_optimizer_restoration_target_reached_fraction": 1.0,
}


REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION: dict[str, object] = {
    **REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION,
    "direct_contract": "best_action_v11_historical_best",
    "action_bank_contract": "historical_best_v1",
}


REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION: dict[str, float] = {
    **REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION,
}

BEST_ACTION_DIRECT_CONTRACTS = {
    "best_action_v6", "best_action_v6_restored",
    "best_action_v7_corrective_restored", "best_action_v10_safe_exact",
    "best_action_v11_historical_best",
}

SAFE_EXACT_DIRECT_CONTRACTS = {
    "best_action_v10_safe_exact", "best_action_v11_historical_best",
}


REGISTERED_BEST_ACTION_V6_INPUT_SHA256: dict[str, str] = {
    "spectrum_data_sha256": (
        "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f"
    ),
    "architecture_checkpoint_sha256": (
        "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2"
    ),
    "candidate_graph_sha256": (
        "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1"
    ),
    "graph_report_sha256": (
        "1a69db8f14d271adfa454900a206b5bc3cc4d1c51e91703e499ca74837bf557f"
    ),
    "routed_ledger_report_sha256": (
        "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349"
    ),
    "training_actions_sha256": (
        "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa"
    ),
    "action_spectra_sha256": (
        "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a"
    ),
    "initial_student_checkpoint_sha256": (
        "8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af"
    ),
    "initial_student_decision_sha256": (
        "0a7de098701e9ef5f6c3fa2ee62fcc2e1b301dab44878286bd0b61bc31e3e9d9"
    ),
    "official_checkpoint_sha256": (
        "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"
    ),
}

REGISTERED_BEST_ACTION_V6_COUNTS: dict[str, int] = {
    "graph_queries": 83619,
    "outer_train_queries": 65286,
    "outer_held_queries": 18333,
    "ledger_rows": 327678,
    "corrective_rows": 62430,
    "strict_corrective_rows_before_floor": 32127,
    "strict_corrective_rows": 32114,
    "strict_corrective_queries": 3482,
    "robust_queries": 39165,
    "harmful_queries": 16955,
    "protective_queries": 65286,
    "harmful_rows": 85959,
    "robust_rows": 179289,
    "admitted_rows": 297362,
}


REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY: dict[str, int | float | bool] = {
    "corrective_queries": 3482,
    "robust_queries": 39165,
    "harmful_queries": 16955,
    "protective_queries": 65286,
    "registered_corrective_batch_size": 4,
    "registered_protective_batch_size": 8,
    "original_corrective_batches": 871,
    "cap_safe_corrective_batches": 877,
    "corrective_batches_added_by_cap_safe_repartition": 6,
    "robust_batches": 9792,
    "harmful_batches": 4239,
    "auxiliary_batches": 14031,
    "protective_batches": 8161,
    "auxiliary_required_optimizer_steps": 3508,
    "protective_required_optimizer_steps": 2041,
    "required_optimizer_steps": 3508,
    "original_corrective_recycle_factor": 3508 / 871,
    "effective_corrective_recycle_factor": 4.0,
    "configured_maximum_corrective_recycle_factor": 4.0,
    "minimum_cap_safe_corrective_batch_size": 3,
    "maximum_cap_safe_corrective_batch_size": 4,
    "minimum_size_corrective_batches": 26,
    "maximum_size_corrective_batches": 851,
    "cap_safe_repartition_required": True,
    "cap_safe_repartition_feasible": True,
}


def _validate_registered_best_action_v6_schedule_geometry(
    observed: dict[str, object],
    *,
    context: str,
) -> None:
    """Fail closed if the exact V6 schedule repair or its inputs drift."""
    mismatches: dict[str, dict[str, object]] = {}
    if set(observed) != set(REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY):
        mismatches["keys"] = {
            "observed": sorted(observed),
            "expected": sorted(REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY),
        }
    for name, expected in REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY.items():
        value = observed.get(name)
        if isinstance(expected, float):
            matched = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                and np.isclose(float(value), expected, rtol=1e-12, atol=1e-12)
            )
        else:
            matched = value == expected and type(value) is type(expected)
        if not matched:
            mismatches[name] = {"observed": value, "expected": expected}
    if mismatches:
        raise RuntimeError(
            f"best-action v6 {context} schedule geometry drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _validate_registered_formal_v3_configuration(args: argparse.Namespace) -> None:
    """Fail before GPU work if a formal direct-v3 arm drifts from its contract."""
    if (
        args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS
        and args.corrective_objective_mode != "v3_direct"
    ):
        raise RuntimeError("best-action direct contracts require the v3_direct objective")
    if args.development or args.corrective_objective_mode != "v3_direct":
        return
    if args.direct_contract == "v3":
        exact_configuration = REGISTERED_V3_EXACT_CONFIGURATION
        float_configuration = REGISTERED_V3_FLOAT_CONFIGURATION
    elif args.direct_contract == "best_action_v6":
        exact_configuration = REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION
        float_configuration = REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION
    elif args.direct_contract == "best_action_v6_restored":
        exact_configuration = REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION
        float_configuration = REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION
    elif args.direct_contract == "best_action_v7_corrective_restored":
        exact_configuration = (
            REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION
        )
        float_configuration = (
            REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION
        )
    elif args.direct_contract == "best_action_v10_safe_exact":
        exact_configuration = REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION
        float_configuration = (
            REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION
        )
    elif args.direct_contract == "best_action_v11_historical_best":
        exact_configuration = (
            REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION
        )
        float_configuration = (
            REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION
        )
    else:
        raise RuntimeError(f"unregistered formal direct contract: {args.direct_contract}")
    mismatches = {}
    for name, expected in exact_configuration.items():
        observed = getattr(args, name)
        if observed != expected or type(observed) is not type(expected):
            mismatches[name] = {"observed": observed, "expected": expected}
    for name, expected in float_configuration.items():
        observed = float(getattr(args, name))
        if not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
            mismatches[name] = {"observed": observed, "expected": expected}
    if mismatches:
        raise RuntimeError(
            f"formal {args.direct_contract} configuration drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _registered_report_configuration(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Return every registered decision variable for the selected contract."""
    if args.direct_contract == "best_action_v7_corrective_restored":
        exact_configuration = (
            REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION
        )
        float_configuration = (
            REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION
        )
    elif args.direct_contract == "best_action_v10_safe_exact":
        exact_configuration = REGISTERED_BEST_ACTION_V10_SAFE_EXACT_CONFIGURATION
        float_configuration = (
            REGISTERED_BEST_ACTION_V10_SAFE_EXACT_FLOAT_CONFIGURATION
        )
    elif args.direct_contract == "best_action_v11_historical_best":
        exact_configuration = (
            REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_CONFIGURATION
        )
        float_configuration = (
            REGISTERED_BEST_ACTION_V11_HISTORICAL_BEST_FLOAT_CONFIGURATION
        )
    elif args.direct_contract == "best_action_v6_restored":
        exact_configuration = REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION
        float_configuration = REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION
    elif args.direct_contract == "best_action_v6":
        exact_configuration = REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION
        float_configuration = REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION
    else:
        exact_configuration = REGISTERED_V3_EXACT_CONFIGURATION
        float_configuration = REGISTERED_V3_FLOAT_CONFIGURATION
    return {
        name: getattr(args, name)
        for name in dict.fromkeys((
            *exact_configuration,
            *float_configuration,
        ))
    }


def _select_corrective_admission(
    train_actions: pd.DataFrame,
    *,
    mode: str,
    margin_floor: float,
    action_bank_contract: str = "all_strict_top1",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the rows allowed to carry the clean corrective gradient.

    The legacy best-action contracts keep every independently materialized
    strict Top-1 action.  V11 instead delegates to the frozen historical-best
    supplier, which keeps one maximum-margin champion per query.  Both routes
    select before tensor reindexing, shuffling, calibration and injection.
    """
    corrective = train_actions.loc[
        train_actions.supervision_kind.eq("corrective")
    ].copy()
    if not np.isfinite(margin_floor) or margin_floor < 0:
        raise ValueError("corrective margin floor must be finite and non-negative")
    if mode == "all":
        if action_bank_contract != "all_strict_top1":
            raise RuntimeError(
                "historical-best action bank requires strict_top1 admission"
            )
        selected = corrective
        strict_available = None
        action_bank_report = None
    elif mode == "strict_top1":
        if action_bank_contract == "all_strict_top1":
            selected, strict = select_materialized_direct_action_panel(
                train_actions,
                mode="multi_action_balanced",
                margin_floor=margin_floor,
            )
            strict_available = int(len(strict))
            action_bank_report = None
        elif action_bank_contract == "historical_best_v1":
            result = HistoricalBestActionBankV1(
                HistoricalBestActionBankV1Config(margin_floor=margin_floor)
            ).select(train_actions)
            selected = result.actions
            strict_available = int(
                result.report["strict_top1_rows_before_margin_floor"]
            )
            action_bank_report = result.report
        else:
            raise ValueError(f"unknown action-bank contract: {action_bank_contract}")
    else:
        raise ValueError(f"unknown corrective admission mode: {mode}")
    if selected.empty or selected.action_id.duplicated().any():
        raise RuntimeError("corrective admission is empty or duplicates an action")
    report = {
        "mode": mode,
        "action_bank_contract": action_bank_contract,
        "margin_floor": float(margin_floor),
        "candidate_corrective_rows": int(len(corrective)),
        "strict_top1_rows_before_margin_floor": strict_available,
        "selected_rows": int(len(selected)),
        "selected_queries": int(selected.query_index.nunique()),
        "all_selected_actions_preserved": True,
        "one_best_query_compression_used": bool(
            action_bank_contract == "historical_best_v1"
        ),
    }
    if action_bank_report is not None:
        report["historical_best_action_bank_v1"] = action_bank_report
    return selected.copy(), report


def _calibration_action_bank(
    targeted_action_spectra: np.ndarray,
    training_action_spectra: np.ndarray,
    *,
    arm_invariant_targeted: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose one frozen calibration bank without changing training views."""
    if targeted_action_spectra.shape != training_action_spectra.shape:
        raise RuntimeError("targeted and training action banks differ in shape")
    if arm_invariant_targeted:
        return targeted_action_spectra, {
            "policy": "targeted_action_bank_shared_by_all_causal_arms",
            "arm_invariant": True,
            "training_action_bank_mutated": False,
        }
    return training_action_spectra, {
        "policy": "arm_specific_training_action_bank",
        "arm_invariant": bool(
            np.array_equal(targeted_action_spectra, training_action_spectra)
        ),
        "training_action_bank_mutated": False,
    }


def _ndarray_sha256(values: np.ndarray) -> str:
    """Hash tensor content and layout without materializing a second byte copy."""
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _string_sequence_sha256(values: list[str] | np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _frame_columns_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    if missing := set(columns) - set(frame.columns):
        raise RuntimeError(f"cannot hash missing admitted-ledger columns: {sorted(missing)}")
    digest = hashlib.sha256()
    for column in columns:
        encoded = column.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for row in frame[columns].itertuples(index=False, name=None):
        for value in row:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _materialize_admitted_action_bank(
    corrective: pd.DataFrame,
    harmful: pd.DataFrame,
    robust: pd.DataFrame,
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Subset/reindex the exact training panel before any causal shuffle.

    The returned tensor banks contain only rows that can enter an optimizer
    branch.  Consequently a shuffled donor cannot be drawn from a rejected
    corrective row, an inner holdout, or a development query excluded by a
    query limit.
    """
    if action_spectra.shape != control_spectra.shape:
        raise RuntimeError("full action/control tensor banks differ in shape")
    panels = {
        "corrective": corrective.copy(),
        "harmful": harmful.copy(),
        "robust": robust.copy(),
    }
    for expected, panel in panels.items():
        if panel.empty:
            raise RuntimeError(f"admitted {expected} action panel is empty")
        if not panel.supervision_kind.astype(str).eq(expected).all():
            raise RuntimeError(f"admitted {expected} panel has mixed supervision")
        if panel.action_id.astype(str).duplicated().any():
            raise RuntimeError(f"admitted {expected} panel duplicates an action")
    admitted = pd.concat(list(panels.values()), ignore_index=True)
    action_ids = admitted.action_id.astype(str)
    if action_ids.duplicated().any():
        raise RuntimeError("admitted action panels overlap by action_id")
    original_index = admitted.action_tensor_index.to_numpy(np.int64)
    if (
        np.any(original_index < 0)
        or np.any(original_index >= len(action_spectra))
        or len(np.unique(original_index)) != len(original_index)
    ):
        raise RuntimeError("admitted action panel has invalid original tensor indices")

    # Preserve the immutable ledger order, then subset action and paired
    # control tensors with the same exact original index vector.
    order = np.argsort(original_index, kind="stable")
    admitted = admitted.iloc[order].reset_index(drop=True)
    original_index = original_index[order]
    admitted_action_spectra = np.ascontiguousarray(action_spectra[original_index])
    admitted_control_spectra = np.ascontiguousarray(control_spectra[original_index])
    admitted["action_tensor_index"] = np.arange(len(admitted), dtype=np.int64)
    if (
        admitted_action_spectra.shape != admitted_control_spectra.shape
        or not np.isfinite(admitted_action_spectra).all()
        or not np.isfinite(admitted_control_spectra).all()
    ):
        raise RuntimeError("admitted action/control tensor materialization failed")

    output_panels = {
        kind: admitted.loc[admitted.supervision_kind.astype(str).eq(kind)].copy()
        for kind in panels
    }
    for kind, original in panels.items():
        if set(output_panels[kind].action_id.astype(str)) != set(
            original.action_id.astype(str)
        ):
            raise RuntimeError(f"admitted {kind} panel changed during tensor reindexing")
    semantic_boundary_columns = [
        "action_id", "query_index", "query_row", "source", "family",
        "recipe_id", "supervision_kind", "control_semantic",
        "action_positive_row", "action_hard_negative_molecule_index",
        "action_hard_negative_row", "control_positive_row",
        "control_hard_negative_molecule_index", "control_hard_negative_row",
    ]
    semantic_boundary_hash = (
        _frame_columns_sha256(admitted, semantic_boundary_columns)
        if set(semantic_boundary_columns) <= set(admitted.columns) else None
    )
    report = {
        "strategy": "admission_then_exact_tensor_subset_then_reindex_before_shuffle",
        "full_tensor_rows": int(len(action_spectra)),
        "admitted_rows": int(len(admitted)),
        "admitted_rows_by_supervision": {
            kind: int(len(panel)) for kind, panel in output_panels.items()
        },
        "admitted_action_ids_sha256": _string_sequence_sha256(
            admitted.action_id.astype(str).tolist()
        ),
        "admitted_semantic_boundary_sha256": semantic_boundary_hash,
        "original_tensor_indices_sha256": _ndarray_sha256(original_index),
        "action_control_subset_uses_identical_original_indices": True,
        "action_tensor_indices_contiguous_after_admission": True,
        "shuffle_donor_pool_equals_optimizer_admitted_panel": True,
        "rejected_corrective_rows_can_be_shuffle_donors": False,
    }
    return (
        admitted,
        output_panels["corrective"],
        output_panels["harmful"],
        output_panels["robust"],
        admitted_action_spectra,
        admitted_control_spectra,
        report,
    )


def _audit_corrective_example_materialization(
    corrective: pd.DataFrame,
    examples: list[BoundaryExample],
) -> dict[str, object]:
    """Prove that query aggregation neither drops nor duplicates action rows."""
    expected_ids = corrective.action_id.astype(str).tolist()
    observed_ids = [
        action.action_id for example in examples for action in example.actions
    ]
    if len({example.query_index for example in examples}) != len(examples):
        raise RuntimeError("corrective query was split across BoundaryExamples")
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("corrective action was duplicated across BoundaryExamples")
    if set(observed_ids) != set(expected_ids) or len(observed_ids) != len(expected_ids):
        raise RuntimeError("corrective BoundaryExamples lost or invented action rows")
    expected_by_query = {
        int(query): set(block.action_id.astype(str))
        for query, block in corrective.groupby("query_index", sort=True)
    }
    observed_by_query = {
        int(example.query_index): {action.action_id for action in example.actions}
        for example in examples
    }
    if expected_by_query != observed_by_query:
        raise RuntimeError("corrective actions were assigned to the wrong query block")
    return {
        "selected_action_rows": int(len(expected_ids)),
        "materialized_action_rows": int(len(observed_ids)),
        "selected_queries": int(len(expected_by_query)),
        "materialized_query_complete_examples": int(len(examples)),
        "exact_action_id_set_preserved": True,
        "one_boundary_example_per_query": True,
    }


def _limit_queries(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(sorted(set(map(int, values))), dtype=np.int64)
    if maximum < 0:
        raise ValueError("query maximum cannot be negative")
    if maximum and len(values) > maximum:
        values = np.sort(np.random.default_rng(seed).choice(values, maximum, replace=False))
    return values


def _limit_risk_queries(harmful: pd.DataFrame, maximum: int, seed: int) -> np.ndarray:
    """Prefer clean-correct queries for the finite risk-preservation budget."""
    if "clean_rank" not in harmful.columns:
        raise KeyError("harmful action ledger misses clean_rank")
    values = np.asarray(sorted(set(map(int, harmful.query_index))), dtype=np.int64)
    if maximum < 0:
        raise ValueError("query maximum cannot be negative")
    if not maximum or len(values) <= maximum:
        return values
    clean_correct = np.asarray(sorted(set(map(
        int, harmful.loc[harmful.clean_rank.eq(1), "query_index"],
    ))), dtype=np.int64)
    clean_error = np.setdiff1d(values, clean_correct, assume_unique=True)
    rng = np.random.default_rng(seed)
    keep_correct = min(maximum, len(clean_correct))
    selected_correct = (
        np.sort(rng.choice(clean_correct, keep_correct, replace=False))
        if keep_correct < len(clean_correct) else clean_correct
    )
    remaining = maximum - len(selected_correct)
    selected_error = (
        np.sort(rng.choice(clean_error, remaining, replace=False))
        if remaining and remaining < len(clean_error)
        else clean_error[:remaining]
    )
    return np.sort(np.concatenate([selected_correct, selected_error])).astype(np.int64)


def select_references(
    graph: CandidateGraph,
    query: int,
    embeddings: np.ndarray,
    row_index: dict[int, int],
    positives: int,
    negatives: int,
    per_negative: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    _, rows, ptr, _ = graph.query_block(query)
    q = embeddings[row_index[int(graph.query_row[query])]]
    pair = embeddings[[row_index[int(row)] for row in rows]] @ q
    positive_order = np.argsort(-pair[int(ptr[0]):int(ptr[1])], kind="stable") + int(ptr[0])
    positive = tuple(map(int, rows[positive_order[:positives]]))
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for left, right in zip(ptr[1:-1], ptr[2:]):
        left, right = int(left), int(right)
        order = np.argsort(-pair[left:right], kind="stable") + left
        selected = tuple(map(int, rows[order[:per_negative]]))
        candidates.append((float(pair[order[0]]), selected))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if negatives > 0:
        candidates = candidates[:negatives]
    if not positive or not candidates:
        raise RuntimeError("query has no positive or negative reference")
    return positive, tuple(item[1] for item in candidates)


def select_references_from_pair_scores(
    graph: CandidateGraph,
    query: int,
    pair_scores: np.ndarray,
    positives: int,
    negatives: int,
    per_negative: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Select the same references from one vectorized full-graph score pass."""
    pair_slice, rows, ptr, _ = graph.query_block(query)
    scores = np.asarray(pair_scores[pair_slice], dtype=np.float32)
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise RuntimeError("pair-score reference cache is not graph-aligned")
    positive_order = np.argsort(
        -scores[int(ptr[0]):int(ptr[1])], kind="stable",
    ) + int(ptr[0])
    positive = tuple(map(int, rows[positive_order[:positives]]))
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for left, right in zip(ptr[1:-1], ptr[2:]):
        left, right = int(left), int(right)
        order = np.argsort(-scores[left:right], kind="stable") + left
        selected = tuple(map(int, rows[order[:per_negative]]))
        candidates.append((float(scores[order[0]]), selected))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if negatives > 0:
        candidates = candidates[:negatives]
    if not positive or not candidates:
        raise RuntimeError("query has no positive or negative reference")
    return positive, tuple(item[1] for item in candidates)


def _reference_margin(
    query_row: int,
    positive: tuple[int, ...],
    negatives: tuple[tuple[int, ...], ...],
    anchor: dict[int, np.ndarray],
) -> np.ndarray:
    q = anchor[query_row]
    pos = float(np.max(np.stack([anchor[row] for row in positive]) @ q))
    return np.asarray([
        pos - float(np.max(np.stack([anchor[row] for row in values]) @ q))
        for values in negatives
    ], dtype=np.float32)


def build_examples(
    graph: CandidateGraph,
    corrective: pd.DataFrame,
    protect_queries: np.ndarray,
    embeddings: np.ndarray,
    index: dict[int, int],
    args: argparse.Namespace,
    pair_scores: np.ndarray | None = None,
    *,
    global_mechanism_balance: bool = False,
) -> tuple[list[BoundaryExample], list[ProtectExample]]:
    queries = sorted(set(map(int, corrective.query_index)) | set(map(int, protect_queries)))
    references = {
        query: (
            select_references_from_pair_scores(
                graph, query, pair_scores, args.positive_references,
                args.negative_molecules, args.references_per_negative,
            )
            if pair_scores is not None else
            select_references(
                graph, query, embeddings, index, args.positive_references,
                args.negative_molecules, args.references_per_negative,
            )
        )
        for query in queries
    }
    def identity_weights(panel_queries: list[int]) -> dict[int, float]:
        if args.corrective_objective_mode != "v3_direct" or not panel_queries:
            return {int(query): 1.0 for query in panel_queries}
        identities = [str(graph.query_ik14[int(query)]) for query in panel_queries]
        counts = pd.Series(identities).value_counts().to_dict()
        mean_count = len(panel_queries) / len(counts)
        weights = {
            int(query): float(mean_count / counts[str(graph.query_ik14[int(query)])])
            for query in panel_queries
        }
        if not np.isclose(np.mean(list(weights.values())), 1.0, atol=1e-12):
            raise RuntimeError("identity-equal query weights do not have mean one")
        return weights

    boundary_queries = sorted(set(map(int, corrective.query_index)))
    boundary_weights = identity_weights(boundary_queries)
    mechanism_weights, _ = (
        _global_mechanism_weights(corrective, boundary_weights)
        if global_mechanism_balance else ({}, {})
    )
    query_local_mechanism_count = {
        int(query): len({
            _mechanism_block(str(source)) for source in block.source.astype(str)
        })
        for query, block in corrective.groupby("query_index", sort=True)
    }
    protect_query_list = list(map(int, protect_queries))
    protect_weights = identity_weights(protect_query_list)
    boundary: list[BoundaryExample] = []
    for query, block in corrective.groupby("query_index", sort=True):
        query = int(query)
        positive, negative = references[query]
        actions = tuple(RoutedAction(
            action_id=str(row.action_id), tensor_index=int(row.action_tensor_index),
            group=(
                f"{_mechanism_block(str(row.source))}::"
                f"{row.source}|{row.family}"
            ),
            control_semantic=str(row.control_semantic),
            mechanism_weight=float(
                mechanism_weights[_mechanism_block(str(row.source))]
                if global_mechanism_balance else
                1.0 / query_local_mechanism_count[query]
            ),
        ) for row in block.sort_values(["source", "family", "action_id"], kind="stable").itertuples())
        boundary.append(BoundaryExample(
            query_index=query, query_row=int(graph.query_row[query]),
            identity=str(graph.query_ik14[query]), formula=str(graph.query_formula[query]),
            actions=actions, positive_rows=positive, negative_rows=negative,
            epoch_weight=boundary_weights[query],
        ))
    protect: list[ProtectExample] = []
    for query in protect_queries:
        query = int(query)
        positive, negative = references[query]
        protect.append(ProtectExample(
            query_index=query, query_row=int(graph.query_row[query]),
            identity=str(graph.query_ik14[query]), formula=str(graph.query_formula[query]),
            positive_rows=positive, negative_rows=negative,
            epoch_weight=protect_weights[query],
        ))
    return boundary, protect


def _union_initial_negative_molecules(
    graph: CandidateGraph,
    initial: list[BoundaryExample] | list[ProtectExample],
    current: list[BoundaryExample] | list[ProtectExample],
) -> list[BoundaryExample] | list[ProtectExample]:
    """Keep current hard molecules plus initialization/action-hard references.

    References are de-duplicated by candidate-molecule index, not by spectrum
    row. If the same molecule is hard at both checkpoints, current rows are
    evaluated first but distinct initialization/action-switch rows remain in
    that molecule's max pool. The loss subsequently selects live top-k edges
    from the molecule union.
    """
    initial_by_query = {int(example.query_index): example for example in initial}
    if set(initial_by_query) != {int(example.query_index) for example in current}:
        raise RuntimeError("initial/current hard-reference query ledgers differ")

    output: list[BoundaryExample] | list[ProtectExample] = []
    for example in current:
        query = int(example.query_index)
        _, candidate_rows, ptr, _ = graph.query_block(query)
        row_to_molecule = {
            int(candidate_rows[position]): molecule
            for molecule, (left, right) in enumerate(zip(ptr[:-1], ptr[1:]))
            for position in range(int(left), int(right))
        }

        def indexed(
            groups: tuple[tuple[int, ...], ...],
        ) -> dict[int, tuple[int, ...]]:
            result: dict[int, tuple[int, ...]] = {}
            for selected_rows in groups:
                keys = {
                    row_to_molecule.get(int(row), -1) for row in selected_rows
                }
                if len(keys) != 1 or -1 in keys or next(iter(keys)) == 0:
                    raise RuntimeError(
                        "negative reference rows do not identify one negative molecule"
                    )
                result[next(iter(keys))] = tuple(map(int, selected_rows))
            return result

        current_by_molecule = {
            **indexed(example.negative_rows),
        }
        initial_by_molecule = {
            **indexed(initial_by_query[query].negative_rows),
        }
        merged = tuple(
            tuple(current_rows) + tuple(
                row for row in initial_by_molecule.get(molecule, ())
                if row not in set(current_rows)
            )
            for molecule, current_rows in current_by_molecule.items()
        ) + tuple(
            rows for molecule, rows in initial_by_molecule.items()
            if molecule not in current_by_molecule
        )
        if not (
            len(example.negative_rows) <= len(merged)
            <= len(example.negative_rows) + len(initial_by_query[query].negative_rows)
        ):
            raise RuntimeError("hard-negative molecule union has invalid cardinality")
        positive = tuple(map(int, example.positive_rows)) + tuple(
            int(row) for row in initial_by_query[query].positive_rows
            if int(row) not in set(map(int, example.positive_rows))
        )
        output.append(replace(
            example, positive_rows=positive, negative_rows=merged,
        ))
    return output


def _hierarchical_action_order(actions: tuple[RoutedAction, ...]) -> list[RoutedAction]:
    """Round-robin mechanism blocks and leaves without using action outcomes."""
    leaves: dict[str, dict[str, list[RoutedAction]]] = defaultdict(lambda: defaultdict(list))
    for action in sorted(actions, key=lambda value: (value.group, value.action_id)):
        mechanism, leaf = (
            action.group.split("::", 1)
            if "::" in action.group else (action.group, action.group)
        )
        leaves[mechanism][leaf].append(action)
    mechanism_queues: dict[str, list[RoutedAction]] = {}
    for mechanism, by_leaf in leaves.items():
        queue = []
        cursor = 0
        while True:
            progressed = False
            for leaf in sorted(by_leaf):
                if cursor < len(by_leaf[leaf]):
                    queue.append(by_leaf[leaf][cursor])
                    progressed = True
            if not progressed:
                break
            cursor += 1
        mechanism_queues[mechanism] = queue
    output = []
    cursor = 0
    while True:
        progressed = False
        for mechanism in ("N", "P", "A4"):
            queue = mechanism_queues.get(mechanism, [])
            if cursor < len(queue):
                output.append(queue[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    if len(output) != len(actions) or {item.action_id for item in output} != {
        item.action_id for item in actions
    }:
        raise RuntimeError("hierarchical action ordering lost a routed action")
    return output


def augment_action_conditioned_references(
    graph: CandidateGraph,
    examples: list[BoundaryExample],
    action_rows: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index: dict[int, int],
    *,
    maximum_positive_references: int,
    maximum_negative_molecules: int,
    references_per_negative: int,
) -> tuple[list[BoundaryExample], dict[str, object]]:
    """Add exact action/control candidate switches to a shared boundary.

    The router has already scored every real action and matched control on the
    complete E8 candidate graph.  It records the positive reference row and
    the precise hardest-negative molecule/row for each view.  This function
    takes a mechanism/source-balanced frontier of those rows and appends it to
    the clean E8 reference set.  All causal arms consume this same true-action
    boundary; no teacher embedding or held outcome is introduced.
    """
    required = {
        "action_id", "query_index", "action_positive_row",
        "action_hard_negative_molecule_index", "action_hard_negative_row",
        "control_positive_row", "control_hard_negative_molecule_index",
        "control_hard_negative_row",
    }
    if missing := required - set(action_rows.columns):
        raise RuntimeError(
            "routed action ledger lacks candidate-switch provenance: "
            f"{sorted(missing)}"
        )
    if min(
        maximum_positive_references,
        maximum_negative_molecules,
        references_per_negative,
    ) < 1:
        raise ValueError("action-conditioned reference limits must be positive")
    row_by_action = {
        str(row.action_id): row for row in action_rows.itertuples(index=False)
    }
    if len(row_by_action) != len(action_rows):
        raise RuntimeError("action IDs are not unique inside a supervision panel")

    output = []
    positive_added = []
    negative_added = []
    negative_rows_added = []
    queries_with_switch = 0
    for example in examples:
        ordered = _hierarchical_action_order(example.actions)
        routed = []
        for action in ordered:
            row = row_by_action.get(action.action_id)
            if row is None or int(row.query_index) != example.query_index:
                raise RuntimeError("boundary example and routed action table differ")
            routed.append(row)
        _, candidate_rows, ptr, _ = graph.query_block(example.query_index)
        candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
        ptr = np.asarray(ptr, dtype=np.int64)
        row_to_molecule = {
            int(candidate_rows[position]): molecule
            for molecule, (left, right) in enumerate(zip(ptr[:-1], ptr[1:]))
            for position in range(int(left), int(right))
        }

        positive = list(map(int, example.positive_rows))
        for column in ("action_positive_row", "control_positive_row"):
            for row in routed:
                value = int(getattr(row, column))
                if row_to_molecule.get(value) != 0:
                    raise RuntimeError("routed positive reference is outside molecule zero")
                if value not in positive:
                    positive.append(value)
        if len(positive) > len(example.positive_rows) + maximum_positive_references:
            raise RuntimeError(
                "action-conditioned positive-reference cap would drop a selected action"
            )

        existing_molecules: set[int] = set()
        existing_order: list[int] = []
        existing_rows: dict[int, tuple[int, ...]] = {}
        for rows in example.negative_rows:
            molecules = {row_to_molecule.get(int(row), -1) for row in rows}
            if len(molecules) != 1 or -1 in molecules or 0 in molecules:
                raise RuntimeError("clean negative reference has invalid molecule identity")
            molecule = next(iter(molecules))
            existing_molecules.add(molecule)
            existing_order.append(molecule)
            existing_rows[molecule] = tuple(map(int, rows))
        forced_rows: dict[int, list[int]] = defaultdict(list)
        candidate_order = []
        for row in routed:
            for prefix in ("action", "control"):
                molecule = int(getattr(row, f"{prefix}_hard_negative_molecule_index"))
                hard_row = int(getattr(row, f"{prefix}_hard_negative_row"))
                if not 1 <= molecule < len(ptr) - 1:
                    raise RuntimeError("routed hard-negative molecule index is invalid")
                if row_to_molecule.get(hard_row) != molecule:
                    raise RuntimeError("routed hard-negative row/molecule pair drifted")
                if hard_row not in forced_rows[molecule]:
                    forced_rows[molecule].append(hard_row)
                if molecule not in existing_molecules and molecule not in candidate_order:
                    candidate_order.append(molecule)
        if len(candidate_order) > maximum_negative_molecules:
            raise RuntimeError(
                "action-conditioned negative-molecule cap would drop a candidate switch"
            )
        selected_molecules = candidate_order
        negative = [
            tuple(existing_rows[molecule]) + tuple(
                row for row in forced_rows.get(molecule, [])
                if row not in set(existing_rows[molecule])
            )
            for molecule in existing_order
        ]
        query_vector = embeddings[embedding_index[int(example.query_row)]]
        for molecule in selected_molecules:
            left, right = map(int, ptr[molecule:molecule + 2])
            positions = np.arange(left, right, dtype=np.int64)
            scores = embeddings[
                [embedding_index[int(candidate_rows[position])] for position in positions]
            ] @ query_vector
            clean_order = positions[np.argsort(-scores, kind="stable")]
            selected_rows = list(forced_rows[molecule])
            for position in clean_order:
                value = int(candidate_rows[int(position)])
                if value not in selected_rows:
                    selected_rows.append(value)
                if len(selected_rows) >= references_per_negative:
                    break
            # Every exact action/control argmax row is mandatory.  The clean
            # E8 ordering only supplements a molecule that has fewer than the
            # usual number of references; it never truncates distinct routed
            # rows that happen to belong to the same candidate molecule.
            negative.append(tuple(selected_rows))
        added_positive = len(positive) - len(example.positive_rows)
        added_negative = len(negative) - len(example.negative_rows)
        added_negative_rows = sum(map(len, negative)) - sum(
            map(len, example.negative_rows)
        )
        positive_added.append(added_positive)
        negative_added.append(added_negative)
        negative_rows_added.append(added_negative_rows)
        queries_with_switch += int(added_negative_rows > 0)
        output.append(replace(
            example,
            positive_rows=tuple(positive),
            negative_rows=tuple(negative),
        ))
    return output, {
        "queries": len(output),
        "action_conditioned_candidate_switch_queries": queries_with_switch,
        "positive_references_added_mean": float(np.mean(positive_added)),
        "positive_references_added_maximum": int(max(positive_added, default=0)),
        "negative_molecules_added_mean": float(np.mean(negative_added)),
        "negative_molecules_added_maximum": int(max(negative_added, default=0)),
        "negative_spectrum_rows_added_mean": float(np.mean(negative_rows_added)),
        "negative_spectrum_rows_added_maximum": int(max(negative_rows_added, default=0)),
        "maximum_action_conditioned_positive_references": maximum_positive_references,
        "maximum_action_conditioned_negative_molecules": maximum_negative_molecules,
        "references_per_negative": references_per_negative,
        "routed_hard_rows_never_truncated_within_molecule": True,
        "selection_is_mechanism_source_balanced": True,
        "all_selected_action_control_boundaries_retained": True,
        "candidate_boundary_cap_truncation": False,
        "all_causal_arms_use_true_routed_boundary": True,
    }


def _boundary_example_spectra_count(
    example: BoundaryExample | ProtectExample,
    *,
    include_actions: bool,
    include_controls: bool,
) -> int:
    actions = len(example.actions) if isinstance(example, BoundaryExample) else 0
    return int(
        1
        + (actions if include_actions else 0)
        + (actions if include_controls else 0)
        + len(example.positive_rows)
        + sum(len(rows) for rows in example.negative_rows)
    )


def _guard_action_forward_spectra(
    examples: list[BoundaryExample] | list[ProtectExample],
    *,
    include_actions: bool,
    include_controls: bool,
    observed_spectra: int,
    args: argparse.Namespace,
    label: str,
) -> None:
    expected = sum(
        _boundary_example_spectra_count(
            example,
            include_actions=include_actions,
            include_controls=include_controls,
        )
        for example in examples
    )
    maximum = int(getattr(args, "maximum_spectra_per_action_forward", 512))
    if maximum < 1:
        raise ValueError("maximum spectra per action forward must be positive")
    if observed_spectra != expected:
        raise RuntimeError(
            f"{label} forward layout count drifted: observed={observed_spectra}, "
            f"expected={expected}"
        )
    if observed_spectra > maximum:
        raise RuntimeError(
            f"{label} query-complete forward requires {observed_spectra} spectra, "
            f"above the registered GPU-safe limit {maximum}; refusing an OOM-prone "
            "forward without splitting a query"
        )


def _forward_memory_contract(
    corrective: list[BoundaryExample],
    robust: list[BoundaryExample],
    harmful: list[BoundaryExample],
    protective: list[ProtectExample],
    args: argparse.Namespace,
) -> dict[str, object]:
    panel_spec = {
        "corrective": (corrective, args.batch_queries, True, True),
        "robust": (robust, args.batch_queries, True, False),
        "harmful": (harmful, args.batch_queries, True, False),
        "protective": (protective, args.protective_batch_queries, False, False),
    }
    panels: dict[str, dict[str, int]] = {}
    for name, (examples, batch_size, include_actions, include_controls) in panel_spec.items():
        counts = sorted((
            _boundary_example_spectra_count(
                example,
                include_actions=include_actions,
                include_controls=include_controls,
            )
            for example in examples
        ), reverse=True)
        panels[name] = {
            "queries": int(len(examples)),
            "largest_spectra_per_query": int(counts[0]) if counts else 0,
            "registered_query_batch_size": int(batch_size),
            "worst_case_spectra_per_forward": int(sum(counts[:batch_size])),
        }
    planned = max(
        values["worst_case_spectra_per_forward"] for values in panels.values()
    )
    maximum = int(getattr(args, "maximum_spectra_per_action_forward", 512))
    report: dict[str, object] = {
        "query_complete": True,
        "queries_split_across_forwards": False,
        "maximum_spectra_per_action_forward": maximum,
        "planned_worst_case_spectra_per_forward": int(planned),
        "panels": panels,
        "gate_passed": bool(maximum >= 1 and planned <= maximum),
    }
    if not report["gate_passed"]:
        raise RuntimeError(f"query-complete action-forward memory contract failed: {report}")
    return report


def corrective_batch_loss(
    model: torch.nn.Module,
    store: SpectrumStore,
    examples: list[BoundaryExample],
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    anchor: dict[int, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    margin_transfer_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    spectra: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    preserve_positions: list[int] = []
    preserve_rows: list[int] = []
    for example in examples:
        clean_index = len(spectra)
        spectra.append(store.one(example.query_row))
        preserve_positions.append(clean_index)
        preserve_rows.append(example.query_row)
        action_indices, control_indices, groups = [], [], []
        for action in example.actions:
            if not 0 <= action.tensor_index < len(action_spectra):
                raise RuntimeError("routed action tensor index is out of bounds")
            action_indices.append(len(spectra))
            spectra.append(torch.from_numpy(action_spectra[action.tensor_index]))
            control_indices.append(len(spectra))
            spectra.append(torch.from_numpy(control_spectra[action.tensor_index]))
            groups.append(action.group)
        positive_indices = []
        for row in example.positive_rows:
            positive_indices.append(len(spectra))
            spectra.append(store.one(row))
            preserve_positions.append(positive_indices[-1])
            preserve_rows.append(row)
        negative_indices = []
        for molecule in example.negative_rows:
            local = []
            for row in molecule:
                local.append(len(spectra))
                spectra.append(store.one(row))
                preserve_positions.append(local[-1])
                preserve_rows.append(row)
            negative_indices.append(local)
        layouts.append({
            "clean": clean_index, "actions": action_indices, "controls": control_indices,
            "groups": groups, "positive": positive_indices, "negative": negative_indices,
        })
    _guard_action_forward_spectra(
        examples,
        include_actions=True,
        include_controls=True,
        observed_spectra=len(spectra),
        args=args,
        label="corrective",
    )
    encoded = forward_embeddings(model, torch.stack(spectra).to(device), args.amp)
    clean_margin, action_margin, control_margin, action_weight, action_group = [], [], [], [], []
    for layout in layouts:
        positive = encoded[layout["positive"]]
        negatives = [encoded[index] for index in layout["negative"]]

        def margins(vector: torch.Tensor) -> torch.Tensor:
            pos = torch.max(positive @ vector)
            return torch.stack([pos - torch.max(values @ vector) for values in negatives])

        clean_margin.append(margins(encoded[int(layout["clean"])]))
        action_margin.append([margins(encoded[int(index)]) for index in layout["actions"]])
        control_margin.append([margins(encoded[int(index)]) for index in layout["controls"]])
        action_weight.append(torch.ones(
            len(layout["actions"]), device=device, dtype=encoded.dtype,
        ))
        action_group.append(list(map(str, layout["groups"])))
    result = direct_boundary_objective(
        clean_margin, action_margin, control_margin, action_weight,
        rank_margin=args.rank_margin, rank_temperature=args.temperature,
        advantage_temperature=args.advantage_temperature,
        topk_negatives=args.topk_negatives,
        action_safety_slack=args.action_safety_slack,
        lambda_clean=(args.lambda_action_clean if args.corrective_objective_mode == "full" else 0.0),
        lambda_corrective_clean=(args.lambda_corrective_clean if args.corrective_objective_mode == "full" else 0.0),
        lambda_action_rank=(args.lambda_action_rank if args.corrective_objective_mode == "full" else 0.0),
        lambda_counterfactual=(args.lambda_counterfactual if args.corrective_objective_mode == "full" else 0.0),
        lambda_action_safety=(args.lambda_action_safety if args.corrective_objective_mode == "full" else 0.0),
        action_group=action_group,
        margin_transfer_fraction=args.margin_transfer_fraction,
        margin_transfer_cap=args.margin_transfer_cap,
        lambda_margin_transfer=args.lambda_margin_transfer,
    )
    target = torch.as_tensor(
        np.stack([anchor[row] for row in preserve_rows]),
        device=device, dtype=encoded.dtype,
    )
    preservation = 1.0 - torch.sum(encoded[preserve_positions] * target, dim=1)
    components = {
        "clean": (
            args.lambda_action_clean * result.clean_rank
            if args.corrective_objective_mode == "full" else result.clean_rank * 0.0
        ),
        "corrective_clean": (
            args.lambda_corrective_clean * result.corrective_clean_rank
            if args.corrective_objective_mode == "full" else result.corrective_clean_rank * 0.0
        ),
        "action_rank": (
            args.lambda_action_rank * result.action_rank
            if args.corrective_objective_mode == "full" else result.action_rank * 0.0
        ),
        "counterfactual": (
            args.lambda_counterfactual * result.counterfactual
            if args.corrective_objective_mode == "full" else result.counterfactual * 0.0
        ),
        "margin_transfer": args.lambda_margin_transfer * result.margin_transfer,
        "action_safety": (
            args.lambda_action_safety * result.action_safety
            if args.corrective_objective_mode == "full" else result.action_safety * 0.0
        ),
    }
    loss = result.loss + (float(margin_transfer_multiplier) - 1.0) * components["margin_transfer"]
    return loss, {
        "loss": float(loss.detach()),
        "clean_rank": float(result.clean_rank.detach()),
        "action_rank": float(result.action_rank.detach()),
        "margin_transfer": float(
            margin_transfer_multiplier * result.margin_transfer.detach()
        ),
        "action_safety": float(result.action_safety.detach()),
        "reference_preservation": float((1.0 - preservation).mean().detach()),
        "queries": float(len(examples)),
        "actions": float(sum(len(example.actions) for example in examples)),
    }, components


def v3_action_panel_batch_loss(
    model: torch.nn.Module,
    store: SpectrumStore,
    examples: list[BoundaryExample],
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    anchor: dict[int, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    *,
    supervision_kind: str,
    diagnose_embedding_gradients: bool = False,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Evaluate one direct-v3 action panel with explicit gradient semantics."""
    if supervision_kind not in {"corrective", "robust", "harmful"}:
        raise ValueError(f"unsupported v3 action supervision: {supervision_kind}")
    if not examples:
        raise ValueError("v3 action panel batch cannot be empty")
    spectra: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    for example in examples:
        clean_index = len(spectra)
        spectra.append(store.one(example.query_row))
        action_indices, control_indices, groups = [], [], []
        control_semantics, mechanism_weights = [], []
        for action in example.actions:
            if not 0 <= action.tensor_index < len(action_spectra):
                raise RuntimeError("v3 routed action tensor index is out of bounds")
            action_indices.append(len(spectra))
            spectra.append(torch.from_numpy(action_spectra[action.tensor_index]))
            if supervision_kind == "corrective":
                control_indices.append(len(spectra))
                spectra.append(torch.from_numpy(control_spectra[action.tensor_index]))
            groups.append(action.group)
            control_semantics.append(action.control_semantic)
            mechanism_weights.append(float(action.mechanism_weight))
        positive_indices = []
        for row in example.positive_rows:
            positive_indices.append(len(spectra))
            spectra.append(store.one(row))
        negative_indices = []
        for molecule in example.negative_rows:
            local = []
            for row in molecule:
                local.append(len(spectra))
                spectra.append(store.one(row))
            negative_indices.append(local)
        layouts.append({
            "clean": clean_index,
            "actions": action_indices,
            "controls": control_indices,
            "groups": groups,
            "control_semantics": control_semantics,
            "mechanism_weights": mechanism_weights,
            "positive": positive_indices,
            "negative": negative_indices,
            "baseline": _reference_margin(
                example.query_row, example.positive_rows, example.negative_rows, anchor,
            ),
        })

    _guard_action_forward_spectra(
        examples,
        include_actions=True,
        include_controls=(supervision_kind == "corrective"),
        observed_spectra=len(spectra),
        args=args,
        label=f"v3-{supervision_kind}",
    )
    encoded = forward_embeddings(model, torch.stack(spectra).to(device), args.amp)
    corrective_locality = corrective_gradient_locality(
        getattr(args, "corrective_gradient_locality", "shared")
        if supervision_kind == "corrective" else "shared"
    )
    clean_margin: list[torch.Tensor] = []
    action_margin: list[list[torch.Tensor]] = []
    control_margin: list[list[torch.Tensor]] = []
    clean_embedding: list[torch.Tensor] = []
    action_embedding: list[list[torch.Tensor]] = []
    baseline_margin: list[torch.Tensor] = []
    action_group: list[list[str]] = []
    action_control_semantic: list[list[str]] = []
    action_block_weight: list[list[float]] = []
    for layout in layouts:
        positive = encoded[layout["positive"]]
        negatives = [encoded[index] for index in layout["negative"]]

        # V7 allowed the action-specific residual to be satisfied mostly by
        # candidate-reference updates.  V8 changes only this boundary: E4/E8
        # clean continuation, protective, robust and harmful objectives remain
        # shared, while corrective query_action_only detaches references from
        # the residual graph and keeps their exact forward values.
        if not corrective_locality.reference_live:
            positive = positive.detach()
            negatives = [value.detach() for value in negatives]

        # The default shared policy exactly reproduces V7.  Query-locality is
        # never a global reference freeze: it applies only to this corrective
        # residual and does not alter the shared encoder or its other losses.
        def shared_margin(vector: torch.Tensor) -> torch.Tensor:
            positive_score = torch.max(positive @ vector)
            return torch.stack([
                positive_score - torch.max(values @ vector)
                for values in negatives
            ])

        clean_vector = encoded[int(layout["clean"])]
        action_vectors = [encoded[int(index)] for index in layout["actions"]]
        clean_embedding.append(clean_vector)
        action_embedding.append(action_vectors)
        shared_clean = shared_margin(clean_vector)
        clean_margin.append(shared_clean)
        action_margin.append([shared_margin(vector) for vector in action_vectors])
        if supervision_kind == "corrective":
            control_margin.append([
                shared_margin(encoded[int(index)]) for index in layout["controls"]
            ])
        baseline_margin.append(torch.as_tensor(
            layout["baseline"], device=device, dtype=encoded.dtype,
        ))
        action_group.append(list(map(str, layout["groups"])))
        action_control_semantic.append(list(map(str, layout["control_semantics"])))
        action_block_weight.append(list(map(float, layout["mechanism_weights"])))

    if supervision_kind == "corrective":
        result = corrective_action_objective(
            clean_margin, action_margin, control_margin, action_group,
            rank_margin=args.rank_margin,
            rank_temperature=args.temperature,
            topk_negatives=args.topk_negatives,
            margin_transfer_fraction=args.margin_transfer_fraction,
            margin_transfer_cap=args.margin_transfer_cap,
            lambda_margin_transfer=args.lambda_margin_transfer,
            lambda_payload_rank=args.lambda_action_rank,
            action_safety_slack=args.action_safety_slack,
            lambda_payload_safety=args.lambda_action_safety,
            query_weight=[example.epoch_weight for example in examples],
            action_control_semantic=action_control_semantic,
            action_block_weight=action_block_weight,
            transfer_target_allocation=args.transfer_target_allocation,
            maximum_transfer_cap_factor=args.maximum_transfer_cap_factor,
        )
        consistency = symmetric_live_action_consistency(
            clean_embedding,
            action_embedding,
            action_group,
            query_weight=[example.epoch_weight for example in examples],
            action_block_weight=action_block_weight,
        )
        components = {
            "transfer": args.lambda_margin_transfer * result.margin_transfer,
            "payload": (
                args.lambda_action_rank * result.payload_rank
                + args.lambda_action_safety * result.payload_safety
            ),
            "consistency": args.lambda_v3_corrective_consistency * consistency,
        }
        total_loss = result.loss + components["consistency"]
        values = {
            "loss": float(total_loss.detach()),
            "margin_transfer": float(result.margin_transfer.detach()),
            "payload_rank": float(result.payload_rank.detach()),
            "payload_safety": float(result.payload_safety.detach()),
            "considered_transfer_edges": float(result.considered_transfer_edges),
            "active_transfer_edges": float(result.active_transfer_edges),
            "capped_transfer_edges": float(result.capped_transfer_edges),
            "active_safety_edges": float(result.active_safety_edges),
            "active_transfer_fraction": float(
                result.active_transfer_edges / result.considered_transfer_edges
            ),
            "capped_transfer_fraction": float(
                result.capped_transfer_edges / result.considered_transfer_edges
            ),
            "active_payload_safety_fraction": float(
                result.active_safety_edges / result.considered_transfer_edges
            ),
            "symmetric_live_consistency": float(consistency.detach()),
        }
        for scope, counts in result.transfer_edge_diagnostics.items():
            for metric, count in counts.items():
                values[f"transfer_edge_count::{scope}::{metric}"] = float(count)
    elif supervision_kind == "robust":
        result = robust_action_objective(
            clean_margin, action_margin, action_group,
            rank_margin=args.rank_margin,
            rank_temperature=args.temperature,
            topk_negatives=args.topk_negatives,
            safety_slack=args.action_safety_slack,
            lambda_payload_rank=args.lambda_v3_robust_rank,
            lambda_safety_floor=args.lambda_v3_robust_floor,
            query_weight=[example.epoch_weight for example in examples],
            action_block_weight=action_block_weight,
        )
        components = {
            "robust": result.loss,
        }
        total_loss = result.loss
        values = {
            "loss": float(result.loss.detach()),
            "payload_rank": float(result.payload_rank.detach()),
            "safety_floor": float(result.safety_floor.detach()),
        }
    else:
        result = harmful_boundary_objective(
            clean_margin, action_margin, baseline_margin, action_group,
            rank_margin=args.rank_margin,
            rank_temperature=args.temperature,
            topk_negatives=args.topk_negatives,
            harmful_damage_slack=args.v3_harmful_damage_slack,
            baseline_floor_slack=args.risk_margin_slack,
            lambda_boundary_rank=args.lambda_v3_harmful_rank,
            lambda_baseline_floor=args.lambda_v3_harmful_floor,
            query_weight=[example.epoch_weight for example in examples],
            action_block_weight=action_block_weight,
        )
        components = {
            "harmful": result.loss,
        }
        total_loss = result.loss
        values = {
            "loss": float(result.loss.detach()),
            "boundary_rank": float(result.boundary_rank.detach()),
            "baseline_floor": float(result.baseline_floor.detach()),
            "active_damage_edges": float(result.active_damage_edges),
        }
    values.update({
        "queries": float(len(examples)),
        "actions": float(sum(len(example.actions) for example in examples)),
        "corrective_reference_gradient_live": float(
            corrective_locality.reference_live
        ),
    })
    if diagnose_embedding_gradients:
        category_positions = {
            "query": [int(layout["clean"]) for layout in layouts],
            "action": [int(index) for layout in layouts for index in layout["actions"]],
            "control": [int(index) for layout in layouts for index in layout["controls"]],
            "reference": [
                int(index)
                for layout in layouts
                for index in (
                    list(layout["positive"])
                    + [item for group in layout["negative"] for item in group]
                )
            ],
        }
        for component_name, component in components.items():
            embedding_gradient = torch.autograd.grad(
                component, encoded, retain_graph=True, allow_unused=True,
            )[0]
            if embedding_gradient is None:
                continue
            total_norm = float(torch.linalg.vector_norm(embedding_gradient.detach().float()))
            values[f"embedding_grad_{component_name}_total_norm"] = total_norm
            for category, positions in category_positions.items():
                category_norm = (
                    float(torch.linalg.vector_norm(
                        embedding_gradient[positions].detach().float()
                    )) if positions else 0.0
                )
                values[f"embedding_grad_{component_name}_{category}_norm"] = category_norm
                values[f"embedding_grad_{component_name}_{category}_fraction"] = (
                    category_norm / total_norm if total_norm > 0 else 0.0
                )
    return total_loss, values, components


def protective_batch_loss(
    model: torch.nn.Module,
    store: SpectrumStore,
    examples: list[ProtectExample],
    anchor: dict[int, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    spectra: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    preserve_positions: list[int] = []
    preserve_rows: list[int] = []
    preserve_owner: list[int] = []
    for owner, example in enumerate(examples):
        clean_index = len(spectra)
        spectra.append(store.one(example.query_row))
        preserve_positions.append(clean_index)
        preserve_rows.append(example.query_row)
        preserve_owner.append(owner)
        positive_indices = []
        for row in example.positive_rows:
            positive_indices.append(len(spectra))
            spectra.append(store.one(row))
            preserve_positions.append(positive_indices[-1])
            preserve_rows.append(row)
            preserve_owner.append(owner)
        negative_indices = []
        for molecule in example.negative_rows:
            local = []
            for row in molecule:
                local.append(len(spectra))
                spectra.append(store.one(row))
                preserve_positions.append(local[-1])
                preserve_rows.append(row)
                preserve_owner.append(owner)
            negative_indices.append(local)
        baseline = _reference_margin(
            example.query_row, example.positive_rows, example.negative_rows,
            anchor,
        )
        layouts.append({
            "clean": clean_index, "positive": positive_indices,
            "negative": negative_indices, "baseline": baseline,
        })
    _guard_action_forward_spectra(
        examples,
        include_actions=False,
        include_controls=False,
        observed_spectra=len(spectra),
        args=args,
        label="protective",
    )
    encoded = forward_embeddings(model, torch.stack(spectra).to(device), args.amp)
    rank_terms, floor_terms = [], []
    for layout in layouts:
        clean = encoded[int(layout["clean"])]
        positive = torch.max(encoded[layout["positive"]] @ clean)
        margin = torch.stack([
            positive - torch.max(encoded[index] @ clean) for index in layout["negative"]
        ])
        baseline = torch.as_tensor(layout["baseline"], device=device, dtype=margin.dtype)
        take = torch.topk(
            -margin.detach(), k=min(args.topk_negatives, len(margin)), largest=True,
        ).indices
        rank_terms.append(F.softplus((args.rank_margin - margin) / args.temperature)[take].mean())
        floor_terms.append(F.relu(baseline - args.risk_margin_slack - margin)[take].mean())
    target = torch.as_tensor(
        np.stack([anchor[row] for row in preserve_rows]),
        device=device, dtype=encoded.dtype,
    )
    preserve = 1.0 - torch.sum(encoded[preserve_positions] * target, dim=1)
    query_weight = torch.as_tensor(
        [example.epoch_weight for example in examples],
        device=device, dtype=encoded.dtype,
    )
    rank_loss = torch.mean(torch.stack(rank_terms) * query_weight)
    floor_loss = torch.mean(torch.stack(floor_terms) * query_weight)
    preserve_terms = torch.stack([
        preserve[torch.as_tensor(
            [value == owner for value in preserve_owner],
            device=device, dtype=torch.bool,
        )].mean()
        for owner in range(len(examples))
    ])
    preserve_loss = torch.mean(preserve_terms * query_weight)
    loss = (
        args.lambda_clean_continuation * rank_loss
        + args.lambda_risk_floor * floor_loss
        + args.lambda_preserve * preserve_loss
    )
    return loss, {
        "loss": float(loss.detach()), "clean_rank": float(rank_loss.detach()),
        "risk_floor": float(floor_loss.detach()),
        "preservation": float((1.0 - preserve).mean().detach()),
        "weighted_preservation_loss": float(preserve_loss.detach()),
        "queries": float(len(examples)),
    }


def _gradient_norm(values: list[torch.Tensor | None]) -> float:
    terms = [
        torch.sum(value.detach().float() ** 2).double()
        for value in values if value is not None
    ]
    total = float(torch.stack(terms).sum().item()) if terms else 0.0
    return float(np.sqrt(total))


def _gradient_dot(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> float:
    terms = [
        torch.sum(one.detach().float() * two.detach().float()).double()
        for one, two in zip(left, right)
        if one is not None and two is not None
    ]
    return float(torch.stack(terms).sum().item()) if terms else 0.0


def _gradient_cosine(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> float | None:
    """Return a finite cosine only when both gradient ledgers are active."""
    left_norm = _gradient_norm(left)
    right_norm = _gradient_norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return None
    cosine = _gradient_dot(left, right) / (left_norm * right_norm)
    return float(np.clip(cosine, -1.0, 1.0))


def _gradient_direction_retention(
    candidate: list[torch.Tensor | None],
    reference: list[torch.Tensor | None],
) -> float | None:
    """Measure how much of a named reference direction remains in candidate."""
    denominator = _gradient_dot(reference, reference)
    if denominator <= 0:
        return None
    return float(_gradient_dot(candidate, reference) / denominator)


def _sum_gradients(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Add two aligned gradient ledgers without losing unused parameters."""
    if len(left) != len(right):
        raise ValueError("gradient ledgers must have the same length")
    return [
        two if one is None else one if two is None else one + two
        for one, two in zip(left, right)
    ]


def _subtract_gradients(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Subtract aligned ledgers while preserving explicit unused parameters."""
    if len(left) != len(right):
        raise ValueError("gradient ledgers must have the same length")
    return [
        None if one is None and two is None
        else -two if one is None
        else one if two is None
        else one - two
        for one, two in zip(left, right)
    ]


def _preserve_corrective_against_auxiliary(
    corrective: list[torch.Tensor | None],
    auxiliary: list[torch.Tensor | None],
    parameter_groups: dict[str, list[int]] | None = None,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], dict[str, object]]:
    """Remove only auxiliary components that oppose the primary correction.

    This is the inner semantic arbitration.  Full-graph risk projection remains
    a separate outer operation.  Consequently robust/harmful constraints can
    contribute agreeing or orthogonal information but cannot silently erase the
    dense corrective direction before the risk veto is measured.
    """
    if len(corrective) != len(auxiliary):
        raise ValueError("corrective and auxiliary gradient ledgers must align")
    groups = parameter_groups or {"all": list(range(len(corrective)))}
    flattened = [position for positions in groups.values() for position in positions]
    if sorted(flattened) != list(range(len(corrective))) or len(flattened) != len(
        set(flattened)
    ):
        raise ValueError("semantic-projection parameter groups must partition gradients")
    projected_auxiliary: list[torch.Tensor | None] = [None] * len(auxiliary)
    group_report: dict[str, dict[str, float | bool]] = {}
    for name, positions in groups.items():
        group_projected, projection = project_corrective_against_risk(
            _take_gradient_positions(auxiliary, positions),
            _take_gradient_positions(corrective, positions),
        )
        for position, gradient in zip(positions, group_projected):
            projected_auxiliary[position] = gradient
        group_combined = _sum_gradients(
            _take_gradient_positions(corrective, positions), group_projected,
        )
        group_corrective = _take_gradient_positions(corrective, positions)
        corrective_sq = _gradient_dot(group_corrective, group_corrective)
        ratio = (
            _gradient_dot(group_combined, group_corrective) / corrective_sq
            if corrective_sq > 0 else 1.0
        )
        group_report[name] = {
            **projection,
            "corrective_direction_preservation_ratio": ratio,
            "corrective_direction_preserved": bool(ratio >= 1.0 - 1e-6),
        }
    combined = _sum_gradients(corrective, projected_auxiliary)
    corrective_sq = _gradient_dot(corrective, corrective)
    primary_ratio = (
        _gradient_dot(combined, corrective) / corrective_sq
        if corrective_sq > 0 else 1.0
    )
    auxiliary_norm = _gradient_norm(auxiliary)
    projected_auxiliary_norm = _gradient_norm(projected_auxiliary)
    return combined, projected_auxiliary, {
        "conflict": bool(any(value["conflict"] for value in group_report.values())),
        "parameter_groups": group_report,
        "raw_auxiliary_gradient_norm": auxiliary_norm,
        "projected_auxiliary_gradient_norm": projected_auxiliary_norm,
        "auxiliary_projection_retention": (
            projected_auxiliary_norm / auxiliary_norm if auxiliary_norm > 0 else 1.0
        ),
        "corrective_direction_preservation_ratio": primary_ratio,
        "corrective_direction_preserved": bool(
            primary_ratio >= 1.0 - 1e-6
            and all(
                value["corrective_direction_preserved"]
                for value in group_report.values()
            )
        ),
    }


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor contents independently of ``torch.save`` container bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _optimizer_parameter_group_positions(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
) -> dict[str, list[int]]:
    """Map registered optimizer groups back onto the flat gradient ledger."""
    position = {id(parameter): index for index, parameter in enumerate(parameters)}
    output: dict[str, list[int]] = {}
    observed: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{group_index}"))
        if name in output:
            raise RuntimeError(f"duplicate optimizer parameter-group name: {name}")
        indices = []
        for parameter in group["params"]:
            index = position.get(id(parameter))
            if index is None or index in observed:
                raise RuntimeError("optimizer groups do not partition trainable parameters")
            observed.add(index)
            indices.append(index)
        if not indices:
            raise RuntimeError(f"optimizer parameter group is empty: {name}")
        output[name] = indices
    if observed != set(range(len(parameters))):
        raise RuntimeError("optimizer groups omit trainable parameters")
    return output


def _take_gradient_positions(
    values: list[torch.Tensor | None], positions: list[int],
) -> list[torch.Tensor | None]:
    return [values[index] for index in positions]


def _virtual_adamw_descent_updates(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Evaluate one AdamW descent update without mutating model/optimizer state.

    This is used to compare the update produced by the real combined gradient
    with a risk-only counterfactual under the *same* moments and step number.
    Their difference measures how much action information survives AdamW,
    including epsilon, momentum and decoupled weight decay effects.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("optimizer counterfactual currently requires AdamW")
    if len(parameters) != len(gradients):
        raise ValueError("virtual AdamW parameters and gradients must align")
    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if set(groups) != {id(parameter) for parameter in parameters}:
        raise RuntimeError("virtual AdamW did not receive the full optimizer parameter set")

    updates: list[torch.Tensor | None] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            updates.append(None)
            continue
        group = groups[id(parameter)]
        beta1, beta2 = map(float, group["betas"])
        if group.get("differentiable", False):
            raise RuntimeError("differentiable AdamW is unsupported by the audit")
        grad = gradient.detach()
        if group.get("maximize", False):
            grad = -grad
        state = optimizer.state.get(parameter, {})
        raw_step = state.get("step", 0)
        step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
        step += 1
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(parameter)
        if exp_avg_sq is None:
            exp_avg_sq = torch.zeros_like(parameter)
        # Mirror AdamW's in-place arithmetic, including the final parameter
        # write in the parameter dtype.  Returning ``decay + adaptive`` is
        # algebraically equivalent in real arithmetic, but it omits the FP32
        # rounding incurred by ``param.mul_`` and ``param.addcdiv_``.  On the
        # 117M-parameter model that subtraction noise was large enough to make
        # a correct virtual step miss the registered 1e-3 reproduction gate.
        next_avg = exp_avg.detach().clone()
        next_avg.lerp_(grad, 1.0 - beta1)
        next_sq = exp_avg_sq.detach().clone()
        next_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)
        if group.get("amsgrad", False):
            maximum = state.get("max_exp_avg_sq")
            if maximum is None:
                maximum = torch.zeros_like(parameter)
            denominator_sq = torch.maximum(maximum.detach(), next_sq)
        else:
            denominator_sq = next_sq
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denominator = denominator_sq.sqrt() / np.sqrt(bias2)
        denominator.add_(float(group["eps"]))
        after = parameter.detach().clone()
        after.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        after.addcdiv_(
            next_avg, denominator,
            value=-(float(group["lr"]) / bias1),
        )
        updates.append(parameter.detach() - after)
    return updates


def _weight_summary(examples: list[BoundaryExample] | list[ProtectExample]) -> dict[str, float | int]:
    values = np.asarray([example.epoch_weight for example in examples], dtype=float)
    if not len(values):
        return {"queries": 0}
    return {
        "queries": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sum": float(np.sum(values)),
    }


def _batches(values: list, size: int, rng: np.random.Generator) -> list[list]:
    order = rng.permutation(len(values))
    return [
        [values[int(index)] for index in order[left:left + size]]
        for left in range(0, len(order), size)
    ]


def _identity_stratified_batches(
    values: list[BoundaryExample] | list[ProtectExample],
    size: int,
    rng: np.random.Generator,
) -> list[list]:
    """Cover every query once while spreading views from repeated identities."""
    if size < 1:
        raise ValueError("identity-stratified batch size must be positive")
    grouped: dict[str, list[BoundaryExample | ProtectExample]] = defaultdict(list)
    for example in values:
        grouped[str(example.identity)].append(example)
    for identity in grouped:
        order = rng.permutation(len(grouped[identity]))
        grouped[identity] = [grouped[identity][int(index)] for index in order]
    ordered: list[BoundaryExample | ProtectExample] = []
    while grouped:
        identities = list(grouped)
        rng.shuffle(identities)
        for identity in identities:
            ordered.append(grouped[identity].pop())
            if not grouped[identity]:
                del grouped[identity]
    return [ordered[left:left + size] for left in range(0, len(ordered), size)]


def _formula_stratified_batches(
    examples: list[BoundaryExample] | list[ProtectExample],
    size: int,
    rng: np.random.Generator,
) -> list[list[BoundaryExample | ProtectExample]]:
    """Cover examples once while maximizing formula diversity per batch."""
    if size < 1:
        raise ValueError("batch size must be positive")
    grouped: dict[str, list[BoundaryExample | ProtectExample]] = defaultdict(list)
    for example in examples:
        grouped[str(example.formula)].append(example)
    for formula in grouped:
        order = rng.permutation(len(grouped[formula]))
        grouped[formula] = [grouped[formula][int(index)] for index in order]
    batches = []
    while grouped:
        batch = []
        formulas = list(grouped)
        rng.shuffle(formulas)
        for formula in formulas:
            batch.append(grouped[formula].pop())
            if not grouped[formula]:
                del grouped[formula]
            if len(batch) == size:
                break
        batches.append(batch)
    flattened = [example for batch in batches for example in batch]
    if len(flattened) != len(examples) or {
        id(example) for example in flattened
    } != {id(example) for example in examples}:
        raise RuntimeError("formula-stratified batching lost or repeated an example")
    return batches


def _prioritize_mechanism_coverage(
    batches: list[list[BoundaryExample | ProtectExample]],
) -> list[list[BoundaryExample | ProtectExample]]:
    """Order formula-diverse batches so rare action mechanisms appear early.

    This changes only the pretraining calibration prefix.  It never changes an
    epoch's training order or duplicates an example.
    """
    remaining = list(enumerate(batches))
    selected: list[list[BoundaryExample | ProtectExample]] = []
    exposure: Counter[str] = Counter()

    def mechanisms(batch: list[BoundaryExample | ProtectExample]) -> set[str]:
        return {
            action.group.split("::", 1)[0]
            for example in batch
            if isinstance(example, BoundaryExample)
            for action in example.actions
        }

    while remaining:
        best_position = max(
            range(len(remaining)),
            key=lambda position: (
                sum(
                    1.0 / (1.0 + exposure[mechanism])
                    for mechanism in mechanisms(remaining[position][1])
                ),
                -remaining[position][0],
            ),
        )
        _, batch = remaining.pop(best_position)
        selected.append(batch)
        exposure.update(mechanisms(batch))
    if {id(batch) for batch in selected} != {id(batch) for batch in batches}:
        raise RuntimeError("mechanism-prioritized calibration lost a batch")
    return selected


def _spread_batches(values: list[list], steps: int, rng: np.random.Generator) -> list[list[list]]:
    """Place batches at approximately uniform positions over an epoch."""
    if steps < 1:
        raise ValueError("optimizer step count must be positive")
    output: list[list[list]] = [[] for _ in range(steps)]
    if not values:
        return output
    order = rng.permutation(len(values))
    offset = int(rng.integers(0, steps))
    for position, index in enumerate(order):
        step = (offset + (position * steps) // len(values)) % steps
        output[step].append(values[int(index)])
    return output


def _single_exposure_auxiliary_schedule(
    values: list[list], steps: int, rng: np.random.Generator,
) -> list[list[list]]:
    """Spread sparse safety panels without globally amplifying them.

    Robust and harmful actions are auxiliary constraints, not a second dense
    training population.  Every original batch enters exactly once per epoch
    at unit scale.  Multiplying a rare batch by ``steps / len(values)`` would
    contradict the query-local contract and create large one-step gradients.
    """
    output = _spread_batches(values, steps, rng)
    flattened = [batch for step in output for batch in step]
    if len(flattened) != len(values) or {id(batch) for batch in flattened} != {
        id(batch) for batch in values
    }:
        raise RuntimeError("auxiliary schedule duplicated or dropped a batch")
    return output


def _interleaved_single_exposure_auxiliary_schedules(
    panels: dict[str, list[list]],
    steps: int,
    rng: np.random.Generator,
) -> dict[str, list[list[list]]]:
    """Interleave semantic panels without creating early-step spikes.

    Batches alternate between panels, then occupy uniformly spaced optimizer
    positions.  When the combined auxiliary batch count does not exceed the
    number of action-active steps, no step receives two auxiliary batches.
    Dense-panel loss scaling is applied separately after this lossless schedule.
    """
    if steps < 1 or not panels:
        raise ValueError("interleaved auxiliary scheduling needs steps and panels")
    output = {
        name: [[] for _ in range(steps)]
        for name in panels
    }
    queues = {
        name: [values[int(index)] for index in rng.permutation(len(values))]
        for name, values in panels.items()
    }
    ordered: list[tuple[str, list]] = []
    panel_order = sorted(panels)
    if panel_order:
        shift = int(rng.integers(0, len(panel_order)))
        panel_order = panel_order[shift:] + panel_order[:shift]
    cursor = {name: 0 for name in panels}
    while any(cursor[name] < len(queues[name]) for name in panels):
        for name in panel_order:
            position = cursor[name]
            if position < len(queues[name]):
                ordered.append((name, queues[name][position]))
                cursor[name] += 1
    if not ordered:
        return output
    offset = int(rng.integers(0, steps))
    for position, (name, batch) in enumerate(ordered):
        step = (offset + (position * steps) // len(ordered)) % steps
        output[name][step].append(batch)

    for name, values in panels.items():
        flattened = [batch for step in output[name] for batch in step]
        if len(flattened) != len(values) or {id(batch) for batch in flattened} != {
            id(batch) for batch in values
        }:
            raise RuntimeError(f"interleaved {name} schedule duplicated or dropped a batch")
    combined_load = [
        sum(len(output[name][step]) for name in output)
        for step in range(steps)
    ]
    if max(combined_load) - min(combined_load) > 1:
        raise RuntimeError("interleaved auxiliary schedule is not load-balanced")
    if len(ordered) <= steps and max(combined_load) > 1:
        raise RuntimeError("avoidable auxiliary-panel overlap remained")
    return output


def _batch_cardinality_scale(values: list, registered_size: int) -> float:
    """Convert an actual-batch mean to a registered-size query sum."""
    if registered_size < 1 or not values or len(values) > registered_size:
        raise ValueError("batch cardinality is outside the registered boundary")
    return float(len(values) / registered_size)


def _bounded_dense_auxiliary_scale(batch_count: int, steps: int) -> float:
    """Cap a dense panel at one batch of cumulative mass per optimizer step."""
    if batch_count < 1 or steps < 1:
        raise ValueError("auxiliary density scaling needs batches and optimizer steps")
    return float(min(1.0, steps / batch_count))


def _bounded_calibration_scale(
    requested: float,
    cap: float,
    label: str,
    *,
    require_exact: bool,
) -> tuple[float, bool]:
    """Apply a safety cap without silently accepting an under-dosed branch."""
    if (
        not np.isfinite(requested)
        or requested < 0
        or not np.isfinite(cap)
        or cap <= 0
    ):
        raise ValueError(f"invalid bounded calibration scale for {label}")
    truncated = bool(requested > cap * (1.0 + 1e-12))
    effective = float(min(requested, cap))
    if require_exact and truncated:
        raise RuntimeError(
            f"{label} calibration requires scale {requested:.6g}, above cap {cap:.6g}; "
            "refusing a silently under-dosed action branch"
        )
    return effective, truncated


def _finalize_transfer_edge_breakdown(
    totals: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    """Convert exact stratum counts into pooled edge fractions."""
    output: dict[str, dict[str, float | int]] = {}
    for scope, counts in sorted(totals.items()):
        considered = int(counts.get("considered", 0))
        values: dict[str, float | int] = {
            name: int(value) for name, value in counts.items()
        }
        for name in (
            "action_better_clean", "action_better_control", "active",
            "clean_limited", "control_limited", "capped",
        ):
            values[f"{name}_fraction"] = (
                float(counts.get(name, 0) / considered) if considered else 0.0
            )
        output[scope] = values
    return output


def _legacy_90pct_end_to_end_loss_reproduced(
    retention_p10: float,
    schedule_report: dict[str, object],
    *,
    active_v3_arm: bool,
    minimum_attributable_fraction_p10: float,
    minimum_corrective_direction_retention_p10: float,
    minimum_optimizer_action_alignment_p10: float,
) -> bool:
    """Diagnose the historical end-to-end signal-loss failure.

    This compatibility diagnostic follows the registered pass/fail comparisons
    exactly.  It must not silently tighten an accepted ``>= minimum`` gate to
    ``> minimum`` merely because a minimum happens to equal 0.10.
    """
    if float(retention_p10) <= 0.10:
        return True
    if not active_v3_arm:
        return False
    if not minimum_gate_passed(
        float(schedule_report.get(
            "optimizer_action_attributable_update_fraction_p10", 0.0,
        )),
        minimum_attributable_fraction_p10,
    ):
        return True
    if not minimum_gate_passed(
        float(schedule_report.get(
            "corrective_direction_retention_after_risk_and_clip_p10", 0.0,
        )),
        minimum_corrective_direction_retention_p10,
    ):
        return True
    if not minimum_gate_passed(
        float(schedule_report.get(
            "optimizer_action_attributable_corrective_alignment_p10", 0.0,
        )),
        minimum_optimizer_action_alignment_p10,
    ):
        return True
    return any(
        float(values.get("pcgrad_clip_action_retention_p10", 0.0)) <= 0.10
        or not minimum_gate_passed(
            float(values.get(
                "optimizer_action_attributable_update_fraction_p10", 0.0,
            )), minimum_attributable_fraction_p10,
        )
        or not minimum_gate_passed(
            float(values.get(
                "corrective_direction_retention_after_risk_and_clip_p10", 0.0,
            )), minimum_corrective_direction_retention_p10,
        )
        or not minimum_gate_passed(
            float(values.get(
                "optimizer_action_attributable_corrective_alignment_p10", 0.0,
            )), minimum_optimizer_action_alignment_p10,
        )
        for values in schedule_report.get(
            "parameter_group_action_signal", {}
        ).values()
    )


def _legacy_90pct_signal_boundary_observed(
    retention_p10: float,
    schedule_report: dict[str, object],
    *,
    active_v3_arm: bool,
) -> bool:
    """Report, without gating, any retained-fraction value at or below 0.10."""
    if float(retention_p10) <= 0.10:
        return True
    if not active_v3_arm:
        return False
    fraction_keys = (
        "optimizer_action_attributable_update_fraction_p10",
        "corrective_direction_retention_after_risk_and_clip_p10",
    )
    if any(float(schedule_report.get(key, 0.0)) <= 0.10 for key in fraction_keys):
        return True
    return any(
        float(values.get("pcgrad_clip_action_retention_p10", 0.0)) <= 0.10
        or any(float(values.get(key, 0.0)) <= 0.10 for key in fraction_keys)
        for values in schedule_report.get(
            "parameter_group_action_signal", {}
        ).values()
    )


def train_v3_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    store: SpectrumStore,
    corrective: list[BoundaryExample],
    robust: list[BoundaryExample],
    harmful: list[BoundaryExample],
    protective: list[ProtectExample],
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    anchor: dict[int, np.ndarray],
    trainable: list[torch.nn.Parameter],
    device: torch.device,
    args: argparse.Namespace,
    *,
    action_scale: float,
    global_gradient_scale: float,
    branch_scale: dict[str, float],
    reference_refresh: Callable[
        [], tuple[
            list[BoundaryExample], list[BoundaryExample],
            list[BoundaryExample], list[ProtectExample],
        ]
    ] | None = None,
    expected_schedule_geometry: dict[str, int | float | bool] | None = None,
) -> tuple[
    list[dict[str, object]], float, bool, float, bool, float, bool, dict[str, object]
]:
    """Train v3 with all protective work inside action-active updates."""
    if args.optimizer_attribution_steps_per_epoch < 1:
        raise ValueError("optimizer attribution steps per epoch must be positive")
    if args.progress_every_steps < 1:
        raise ValueError("progress heartbeat interval must be positive")
    if not 0 <= args.minimum_optimizer_action_attributable_fraction_p10 <= 1:
        raise ValueError("optimizer action-attributable fraction gate must be in [0, 1]")
    if not 0 <= args.minimum_corrective_direction_retention_p10 <= 1:
        raise ValueError("corrective direction-retention gate must be in [0, 1]")
    if not 0 < args.v4_audit_minimum_attributable_fraction < 1:
        raise ValueError("v4 audit attributable target must be in (0, 1)")
    if args.v4_audit_maximum_action_gain < 1:
        raise ValueError("v4 audit maximum action gain must be at least one")
    if not 0 <= args.optimizer_restoration_minimum_risk_component_retention <= 1:
        raise ValueError("optimizer restoration risk retention must be in [0, 1]")
    if not 0 <= args.minimum_optimizer_restoration_target_reached_fraction <= 1:
        raise ValueError("optimizer restoration target coverage must be in [0, 1]")
    if args.maximum_safe_exact_update_norm_ratio < 1:
        raise ValueError("safe exact update norm-ratio cap must be at least one")
    if (
        args.optimizer_restoration_scope == "safe_exact_corrective"
        and (
            not args.materialize_optimizer_update_restoration
            or not args.reconcile_restored_adamw_first_moment
        )
    ):
        raise RuntimeError(
            "safe exact corrective composition requires materialization and "
            "AdamW first-moment reconciliation"
        )
    if (
        args.materialize_optimizer_update_restoration
        and args.direct_contract not in {
            "best_action_v6_restored", "best_action_v7_corrective_restored",
            "best_action_v10_safe_exact", "best_action_v11_historical_best",
        }
        and not args.development
    ):
        raise RuntimeError(
            "optimizer update restoration is formal only in a registered restored contract"
        )
    if (
        args.optimizer_restoration_scope in {
            "corrective_only", "safe_exact_corrective",
        }
        and args.direct_contract not in {
            "best_action_v7_corrective_restored", "best_action_v10_safe_exact",
            "best_action_v11_historical_best",
        }
        and not args.development
    ):
        raise RuntimeError(
            "corrective-only optimizer restoration is formal only in "
            "best_action_v7_corrective_restored"
        )
    if (
        args.reconcile_restored_adamw_first_moment
        and args.direct_contract not in {
            "best_action_v7_corrective_restored", "best_action_v10_safe_exact",
            "best_action_v11_historical_best",
        }
        and not args.development
    ):
        raise RuntimeError(
            "restored AdamW first-moment reconciliation is formal only in "
            "best_action_v7_corrective_restored"
        )
    if not -1 < args.v4_audit_minimum_update_action_alignment < 1:
        raise ValueError("v4 audit update/action target must be in (-1, 1)")
    if args.v4_audit_maximum_update_action_coefficient < 0:
        raise ValueError("v4 audit update/action coefficient cap must be non-negative")
    rng = np.random.default_rng(args.seed + 701)
    history: list[dict[str, object]] = []
    all_retention: list[float] = []
    all_clip_retention: list[float] = []
    all_optimizer_alignment: list[float] = []
    all_optimizer_action_fraction: list[float] = []
    all_optimizer_attributable_alignment: list[float] = []
    all_corrective_direction_retention: list[float] = []
    all_optimizer_corrective_alignment: list[float] = []
    all_optimizer_virtual_error: list[float] = []
    all_restoration_target_reached: list[float] = []
    all_restoration_risk_retention: list[float] = []
    all_restoration_gain: list[float] = []
    all_restoration_hard_floor_enforced: list[float] = []
    all_restoration_exact_fraction_error: list[float] = []
    all_restoration_update_norm_ratio: list[float] = []
    all_restoration_reconciliation_error: list[float] = []
    all_restoration_materialization_error: list[float] = []
    all_restoration_fp32_replay_error: list[float] = []
    all_restoration_first_moment_change: list[float] = []
    all_protective_gradient_full_reach: list[float] = []
    parameter_group_positions = _optimizer_parameter_group_positions(
        optimizer, trainable,
    )
    safe_exact_injector = (
        ActionInjectorV1(ActionInjectorV1Config(
            target_attributable_fraction=(
                args.v4_audit_minimum_attributable_fraction
            ),
            minimum_protective_component_retention=(
                args.optimizer_restoration_minimum_risk_component_retention
            ),
            maximum_update_norm_ratio_to_original=(
                args.maximum_safe_exact_update_norm_ratio
            ),
        ))
        if args.optimizer_restoration_scope == "safe_exact_corrective"
        else None
    )
    parameter_group_diagnostics: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in parameter_group_positions
    }
    maximum_risk_microbatches = 0
    minimum_risk_microbatches = None
    epoch_schedule_geometries: list[dict[str, int | float | bool]] = []
    for epoch in range(args.epochs):
        epoch_started = time.time()
        original_corr_batches = _identity_stratified_batches(
            corrective, args.batch_queries, rng,
        )
        robust_batches = _identity_stratified_batches(robust, args.batch_queries, rng)
        harmful_batches = _identity_stratified_batches(harmful, args.batch_queries, rng)
        risk_batches = _identity_stratified_batches(
            protective, args.protective_batch_queries, rng,
        )
        if args.v3_maximum_auxiliary_microbatches_per_step < 1:
            raise ValueError("maximum auxiliary microbatches per step must be positive")
        schedule_geometry = v3_schedule_geometry(
            corrective_queries=len(corrective),
            robust_queries=len(robust),
            harmful_queries=len(harmful),
            protective_queries=len(protective),
            corrective_batch_size=int(args.batch_queries),
            protective_batch_size=int(args.protective_batch_queries),
            maximum_auxiliary_microbatches_per_step=int(
                args.v3_maximum_auxiliary_microbatches_per_step
            ),
            maximum_protective_microbatches_per_step=int(
                args.v3_maximum_protective_microbatches_per_step
            ),
            maximum_corrective_recycle_factor=float(
                args.v3_maximum_corrective_recycle_factor
            ),
        ).as_dict()
        if expected_schedule_geometry is not None and (
            schedule_geometry != expected_schedule_geometry
        ):
            raise RuntimeError(
                "v3 epoch schedule geometry differs from the pre-model preflight: "
                + json.dumps({
                    "epoch": epoch + 1,
                    "observed": schedule_geometry,
                    "expected": expected_schedule_geometry,
                }, sort_keys=True)
            )
        if len(original_corr_batches) != schedule_geometry[
            "original_corrective_batches"
        ]:
            raise RuntimeError("identity-stratified corrective batch count drifted")
        if (
            len(robust_batches) != schedule_geometry["robust_batches"]
            or len(harmful_batches) != schedule_geometry["harmful_batches"]
            or len(risk_batches) != schedule_geometry["protective_batches"]
        ):
            raise RuntimeError("v3 non-corrective batch geometry drifted")
        corr_batches = cap_safe_corrective_repartition(
            original_corr_batches,
            int(schedule_geometry["cap_safe_corrective_batches"]),
        )
        if (
            len(corr_batches) != schedule_geometry["cap_safe_corrective_batches"]
            or min(map(len, corr_batches))
            != schedule_geometry["minimum_cap_safe_corrective_batch_size"]
            or max(map(len, corr_batches))
            != schedule_geometry["maximum_cap_safe_corrective_batch_size"]
        ):
            raise RuntimeError("cap-safe corrective repartition geometry drifted")
        corrective_batch_size_counts = dict(sorted(Counter(
            map(len, corr_batches),
        ).items()))
        minimum_batch_size = int(
            schedule_geometry["minimum_cap_safe_corrective_batch_size"]
        )
        maximum_batch_size = int(
            schedule_geometry["maximum_cap_safe_corrective_batch_size"]
        )
        expected_corrective_batch_size_counts = (
            {minimum_batch_size: len(corr_batches)}
            if minimum_batch_size == maximum_batch_size else {
                minimum_batch_size: int(
                    schedule_geometry["minimum_size_corrective_batches"]
                ),
                maximum_batch_size: int(
                    schedule_geometry["maximum_size_corrective_batches"]
                ),
            }
        )
        if corrective_batch_size_counts != expected_corrective_batch_size_counts:
            raise RuntimeError(
                "cap-safe corrective repartition batch-size counts drifted: "
                f"{corrective_batch_size_counts} != "
                f"{expected_corrective_batch_size_counts}"
            )
        epoch_schedule_geometries.append(schedule_geometry)
        minimum_auxiliary_steps = int(
            schedule_geometry["auxiliary_required_optimizer_steps"]
        )
        plan = balanced_action_step_plan(
            corr_batches,
            risk_batches,
            seed=args.seed + 900 + epoch,
            maximum_protective_microbatches_per_step=(
                args.v3_maximum_protective_microbatches_per_step
            ),
            minimum_steps=max(1, minimum_auxiliary_steps),
        )
        steps = len(plan)
        if steps != schedule_geometry["required_optimizer_steps"]:
            raise RuntimeError("v3 planner differs from the preflight optimizer-step count")
        attribution_steps = set(map(int, np.linspace(
            0,
            steps - 1,
            num=min(args.optimizer_attribution_steps_per_epoch, steps),
            dtype=int,
        )))
        corrective_recycle_factor = steps / len(corr_batches)
        if not np.isclose(
            corrective_recycle_factor,
            float(schedule_geometry["effective_corrective_recycle_factor"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError("v3 corrective recycle factor differs from preflight")
        if corrective_recycle_factor > args.v3_maximum_corrective_recycle_factor:
            raise RuntimeError(
                "v3 corrective recycling exceeded the frozen factor cap: "
                f"{corrective_recycle_factor:.4f} > "
                f"{args.v3_maximum_corrective_recycle_factor:.4f}"
            )
        auxiliary_schedules = _interleaved_single_exposure_auxiliary_schedules(
            {"robust": robust_batches, "harmful": harmful_batches},
            steps,
            rng,
        )
        robust_schedule = auxiliary_schedules["robust"]
        harmful_schedule = auxiliary_schedules["harmful"]
        auxiliary_panel_scale = {
            "robust": _bounded_dense_auxiliary_scale(len(robust_batches), steps),
            "harmful": _bounded_dense_auxiliary_scale(len(harmful_batches), steps),
        }
        auxiliary_step_load = [
            len(robust_schedule[index]) + len(harmful_schedule[index])
            for index in range(steps)
        ]
        if max(auxiliary_step_load) > args.v3_maximum_auxiliary_microbatches_per_step:
            raise RuntimeError(
                "v3 auxiliary action panels exceed the per-step microbatch cap: "
                f"{max(auxiliary_step_load)} > "
                f"{args.v3_maximum_auxiliary_microbatches_per_step}"
            )
        robust_active_step_fraction = float(
            sum(bool(batches) for batches in robust_schedule) / steps
        )
        harmful_active_step_fraction = float(
            sum(bool(batches) for batches in harmful_schedule) / steps
        )
        protective_microbatch_scale = float(steps / len(risk_batches))
        risk_counts = [len(step.protective) for step in plan]
        maximum_risk_microbatches = max(maximum_risk_microbatches, max(risk_counts))
        epoch_minimum = min(risk_counts)
        minimum_risk_microbatches = (
            epoch_minimum if minimum_risk_microbatches is None
            else min(minimum_risk_microbatches, epoch_minimum)
        )
        if max(risk_counts) > args.v3_maximum_protective_microbatches_per_step:
            raise RuntimeError(
                "v3 action ledger is too sparse for balanced full-graph training: "
                f"maximum protective microbatches per action step={max(risk_counts)} > "
                f"{args.v3_maximum_protective_microbatches_per_step}"
            )

        logs: dict[str, list[float]] = defaultdict(list)
        epoch_transfer_edge_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        seen_queries = {"corrective": set(), "robust": set(), "harmful": set(), "protective": set()}
        seen_actions: set[str] = set()
        corrective_query_exposures: Counter[int] = Counter()
        corrective_action_exposures: Counter[str] = Counter()
        corrective_action_weighted_dose: Counter[str] = Counter()
        mechanism_epoch_mass: dict[str, Counter[str]] = {
            "corrective": Counter(), "robust": Counter(), "harmful": Counter(),
        }
        for step_index, scheduled in enumerate(plan):
            active_arm = args.arm in {"routed_direct", "shuffled_action_control"}
            materialize_optimizer_restoration = bool(
                active_arm and args.materialize_optimizer_update_restoration
            )
            corrective_only_restoration = bool(
                materialize_optimizer_restoration
                and args.optimizer_restoration_scope in {
                    "corrective_only", "safe_exact_corrective",
                }
            )
            safe_exact_restoration = bool(
                materialize_optimizer_restoration
                and args.optimizer_restoration_scope == "safe_exact_corrective"
            )
            audit_optimizer_attribution = bool(
                active_arm
                and (
                    materialize_optimizer_restoration
                    or step_index in attribution_steps
                )
            )
            corrective_grad: list[torch.Tensor | None] = [None] * len(trainable)
            robust_grad: list[torch.Tensor | None] = [None] * len(trainable)
            harmful_grad: list[torch.Tensor | None] = [None] * len(trainable)

            def accumulate_action_gradient(loss: torch.Tensor, *, branch: str) -> None:
                nonlocal corrective_grad, robust_grad, harmful_grad
                if not active_arm:
                    return
                local = list(torch.autograd.grad(
                    global_gradient_scale * action_scale * loss,
                    trainable,
                    allow_unused=True,
                ))
                if branch == "corrective":
                    corrective_grad = _sum_gradients(corrective_grad, local)
                elif branch == "robust":
                    robust_grad = _sum_gradients(robust_grad, local)
                elif branch == "harmful":
                    harmful_grad = _sum_gradients(harmful_grad, local)
                else:
                    raise ValueError(f"unknown v3 action-gradient branch: {branch}")

            corr = scheduled.corrective
            corrective_step_scale = corrective_recycle_scale(
                scheduled,
                full_dose=args.v3_corrective_recycle_full_dose,
                full_dose_epoch_multiplier=corrective_recycle_factor,
            )
            corrective_batch_scale = _batch_cardinality_scale(
                corr, args.batch_queries,
            )
            for example in corr:
                seen_queries["corrective"].add(example.query_index)
                corrective_query_exposures[example.query_index] += 1
                for action in example.actions:
                    seen_actions.add(action.action_id)
                    corrective_action_exposures[action.action_id] += 1
                    corrective_action_weighted_dose[action.action_id] += corrective_step_scale
            mechanism_epoch_mass["corrective"].update(
                _weighted_mechanism_mass(
                    corr, corrective_step_scale / args.batch_queries,
                )
            )
            if active_arm:
                _, values, corr_components = v3_action_panel_batch_loss(
                    model, store, corr, action_spectra, control_spectra,
                    anchor, device, args, supervision_kind="corrective",
                )
                for key, value in values.items():
                    if key.startswith("transfer_edge_count::"):
                        _, scope, metric = key.split("::", 2)
                        epoch_transfer_edge_totals[scope][metric] += int(
                            round(float(value))
                        )
                    else:
                        logs[f"corrective_{key}"].append(value)
                corrective_loss = (
                    corrective_step_scale * corrective_batch_scale * (
                        branch_scale["transfer"] * corr_components["transfer"]
                        + branch_scale["payload"] * corr_components["payload"]
                        + branch_scale["consistency"]
                        * corr_components["consistency"]
                    )
                )
                accumulate_action_gradient(corrective_loss, branch="corrective")
            logs["corrective_schedule_scale"].append(corrective_step_scale)
            logs["corrective_batch_cardinality_scale"].append(
                corrective_batch_scale
            )

            for kind, batches, panel_scale in (
                (
                    "robust", robust_schedule[step_index],
                    auxiliary_panel_scale["robust"],
                ),
                (
                    "harmful", harmful_schedule[step_index],
                    auxiliary_panel_scale["harmful"],
                ),
            ):
                for batch in batches:
                    batch_cardinality_scale = _batch_cardinality_scale(
                        batch, args.batch_queries,
                    )
                    mechanism_epoch_mass[kind].update(
                        _weighted_mechanism_mass(
                            batch, panel_scale / args.batch_queries,
                        )
                    )
                    for example in batch:
                        if example.query_index in seen_queries[kind]:
                            raise RuntimeError(f"v3 {kind} query repeated within an epoch")
                        seen_queries[kind].add(example.query_index)
                        for action in example.actions:
                            if action.action_id in seen_actions:
                                raise RuntimeError("v3 action repeated across supervision panels")
                            seen_actions.add(action.action_id)
                    if active_arm:
                        _, panel_values, components = v3_action_panel_batch_loss(
                            model, store, batch, action_spectra, control_spectra,
                            anchor, device, args, supervision_kind=kind,
                        )
                        for key, value in panel_values.items():
                            logs[f"{kind}_{key}"].append(value)
                        accumulate_action_gradient(
                            panel_scale * batch_cardinality_scale
                            * branch_scale[kind] * components[kind],
                            branch=kind,
                        )
                    logs[f"{kind}_batch_cardinality_scale"].append(
                        batch_cardinality_scale
                    )
            risk_grad: list[torch.Tensor | None] = [None] * len(trainable)
            for risk_batch in scheduled.protective:
                for example in risk_batch:
                    if example.query_index in seen_queries["protective"]:
                        raise RuntimeError("v3 protective query repeated within an epoch")
                    seen_queries["protective"].add(example.query_index)
                risk_loss, risk_values = protective_batch_loss(
                    model, store, risk_batch, anchor, device, args,
                )
                for key, value in risk_values.items():
                    logs[f"protective_{key}"].append(value)
                risk_batch_scale = (
                    protective_microbatch_scale
                    * _batch_cardinality_scale(
                        risk_batch, args.protective_batch_queries,
                    )
                )
                local = list(torch.autograd.grad(
                    global_gradient_scale * risk_batch_scale * risk_loss,
                    trainable,
                    allow_unused=True,
                ))
                logs["protective_effective_microbatch_scale"].append(
                    risk_batch_scale
                )
                risk_grad = [
                    right if left is None else left if right is None else left + right
                    for left, right in zip(risk_grad, local)
                ]

            auxiliary_grad = _sum_gradients(robust_grad, harmful_grad)
            action_grad, projected_auxiliary_grad, semantic_projection = (
                _preserve_corrective_against_auxiliary(
                    corrective_grad, auxiliary_grad, parameter_group_positions,
                )
            )
            if active_arm and not semantic_projection["corrective_direction_preserved"]:
                raise RuntimeError(
                    "auxiliary action constraints erased the corrective direction: "
                    f"{semantic_projection}"
                )
            projected, projection = project_corrective_against_risk(action_grad, risk_grad)
            projected_noncorrective_action, noncorrective_projection = (
                project_corrective_against_risk(
                    projected_auxiliary_grad, risk_grad,
                )
            )
            conditional_corrective_grad = _subtract_gradients(
                projected, projected_noncorrective_action,
            )
            noncorrective_grad = _sum_gradients(
                projected_noncorrective_action, risk_grad,
            )
            before_projection = _gradient_norm(action_grad)
            after_projection = _gradient_norm(projected)
            projection_retention = (
                after_projection / before_projection if before_projection > 0 else 1.0
            )
            corrective_direction_retention = _gradient_direction_retention(
                projected, corrective_grad,
            )
            if active_arm and corrective_direction_retention is None:
                raise RuntimeError("v3 corrective gradient vanished before risk arbitration")
            for parameter, left, right in zip(trainable, projected, risk_grad):
                if left is None and right is None:
                    parameter.grad = None
                elif left is None:
                    parameter.grad = right
                elif right is None:
                    parameter.grad = left
                else:
                    parameter.grad = left + right
            combined = _gradient_norm([parameter.grad for parameter in trainable])
            clip_retention = (
                min(1.0, args.grad_clip / (combined + 1e-6))
                if combined > 0 else 1.0
            )
            virtual_combined: list[torch.Tensor | None] = []
            virtual_risk: list[torch.Tensor | None] = []
            virtual_noncorrective: list[torch.Tensor | None] = []
            virtual_attribution_baseline: list[torch.Tensor | None] = []
            virtual_restoration = None
            groupwise_restoration = None
            prepared_action_injection = None
            applied_action_injection = None
            if active_arm:
                branch_gradients = {
                    "corrective": corrective_grad,
                    "robust": robust_grad,
                    "harmful": harmful_grad,
                    "protective": risk_grad,
                }
                for name, gradient in branch_gradients.items():
                    norm = _gradient_norm(gradient)
                    logs[f"raw_branch_gradient_norm_all_steps::{name}"].append(norm)
                    if norm > 0:
                        logs[f"raw_branch_gradient_norm_active_steps::{name}"].append(norm)
                for left, right in (
                    ("corrective", "robust"),
                    ("corrective", "harmful"),
                    ("corrective", "protective"),
                    ("robust", "protective"),
                    ("harmful", "protective"),
                ):
                    cosine = _gradient_cosine(
                        branch_gradients[left], branch_gradients[right],
                    )
                    if cosine is not None:
                        logs[f"raw_branch_gradient_cosine::{left}_vs_{right}"].append(
                            cosine
                        )
                action_retention = projection_retention * clip_retention
                clipped_corrective_direction_retention = float(
                    corrective_direction_retention * clip_retention
                )
                all_retention.append(action_retention)
                all_clip_retention.append(clip_retention)
                all_corrective_direction_retention.append(
                    clipped_corrective_direction_retention
                )
                logs["action_gradient_norm_preprojection"].append(before_projection)
                logs["action_gradient_norm_postprojection"].append(after_projection)
                logs["projection_retention"].append(projection_retention)
                logs["clip_retention"].append(clip_retention)
                logs["action_signal_retention"].append(action_retention)
                logs["corrective_direction_retention_after_risk_projection"].append(
                    float(corrective_direction_retention)
                )
                logs["corrective_direction_retention_after_risk_and_clip"].append(
                    clipped_corrective_direction_retention
                )
                logs["gradient_conflict"].append(float(bool(projection["conflict"])))
                logs["noncorrective_gradient_conflict"].append(float(bool(
                    noncorrective_projection["conflict"]
                )))
                logs["corrective_gradient_norm_before_auxiliary"].append(
                    _gradient_norm(corrective_grad)
                )
                logs["auxiliary_gradient_norm_before_inner_projection"].append(
                    float(semantic_projection["raw_auxiliary_gradient_norm"])
                )
                logs["auxiliary_gradient_norm_after_inner_projection"].append(
                    float(semantic_projection["projected_auxiliary_gradient_norm"])
                )
                logs["auxiliary_inner_projection_retention"].append(
                    float(semantic_projection["auxiliary_projection_retention"])
                )
                logs["corrective_direction_preservation_ratio"].append(
                    float(semantic_projection[
                        "corrective_direction_preservation_ratio"
                    ])
                )
                for name, values in semantic_projection["parameter_groups"].items():
                    logs[
                        f"corrective_direction_preservation_ratio::{name}"
                    ].append(float(values[
                        "corrective_direction_preservation_ratio"
                    ]))
                if audit_optimizer_attribution:
                    all_protective_gradient_full_reach.append(float(all(
                        gradient is not None for gradient in risk_grad
                    )))
                    for name, positions in parameter_group_positions.items():
                        group_before = _gradient_norm(
                            _take_gradient_positions(action_grad, positions)
                        )
                        group_after = _gradient_norm(
                            _take_gradient_positions(projected, positions)
                        )
                        group_projection_retention = (
                            group_after / group_before if group_before > 0 else 0.0
                        )
                        group_corrective_retention = _gradient_direction_retention(
                            _take_gradient_positions(projected, positions),
                            _take_gradient_positions(corrective_grad, positions),
                        )
                        parameter_group_diagnostics[name][
                            "action_gradient_preprojection_norm"
                        ].append(group_before)
                        parameter_group_diagnostics[name][
                            "action_gradient_postprojection_norm"
                        ].append(group_after)
                        parameter_group_diagnostics[name][
                            "pcgrad_retention"
                        ].append(group_projection_retention)
                        parameter_group_diagnostics[name][
                            "pcgrad_clip_retention"
                        ].append(group_projection_retention * clip_retention)
                        parameter_group_diagnostics[name][
                            "corrective_direction_risk_clip_retention"
                        ].append(
                            float(group_corrective_retention * clip_retention)
                            if group_corrective_retention is not None else 0.0
                        )
                        parameter_group_diagnostics[name][
                            "protective_gradient_full_parameter_reach"
                        ].append(float(all(
                            risk_grad[position] is not None
                            for position in positions
                        )))
                    if safe_exact_restoration:
                        if safe_exact_injector is None:
                            raise RuntimeError(
                                "ActionInjectorV1 was not constructed for V10"
                            )
                        prepared_action_injection = safe_exact_injector.prepare(
                            optimizer,
                            trainable,
                            clipped_combined_gradients=[
                                None if parameter.grad is None
                                else parameter.grad.detach() * clip_retention
                                for parameter in trainable
                            ],
                            clipped_noncorrective_gradients=[
                                None if gradient is None
                                else gradient.detach() * clip_retention
                                for gradient in noncorrective_grad
                            ],
                            clipped_protective_gradients=[
                                None if gradient is None
                                else gradient.detach() * clip_retention
                                for gradient in risk_grad
                            ],
                            parameter_group_positions=parameter_group_positions,
                        )
                        virtual_combined = (
                            prepared_action_injection.virtual_combined_updates
                        )
                        virtual_noncorrective = (
                            prepared_action_injection.virtual_noncorrective_updates
                        )
                        virtual_risk = (
                            prepared_action_injection.virtual_protective_updates
                        )
                        groupwise_restoration = (
                            prepared_action_injection.composition
                        )
                        virtual_attribution_baseline = (
                            prepared_action_injection.composition
                            .counterfactual_baseline_updates
                        )
                    else:
                        virtual_combined = _virtual_adamw_descent_updates(
                            optimizer,
                            trainable,
                            [
                                None if parameter.grad is None
                                else parameter.grad.detach() * clip_retention
                                for parameter in trainable
                            ],
                        )
                        virtual_risk = _virtual_adamw_descent_updates(
                            optimizer,
                            trainable,
                            [
                                None if gradient is None
                                else gradient.detach() * clip_retention
                                for gradient in risk_grad
                            ],
                        )
                        if corrective_only_restoration:
                            virtual_noncorrective = _virtual_adamw_descent_updates(
                                optimizer,
                                trainable,
                                [
                                    None if gradient is None
                                    else gradient.detach() * clip_retention
                                    for gradient in noncorrective_grad
                                ],
                            )
                            virtual_attribution_baseline = virtual_noncorrective
                        else:
                            virtual_attribution_baseline = virtual_risk
                    if materialize_optimizer_restoration:
                        if not safe_exact_restoration:
                            if corrective_only_restoration:
                                groupwise_restoration = (
                                    arbitrate_corrective_optimizer_updates_by_group(
                                        virtual_combined,
                                        virtual_noncorrective,
                                        virtual_risk,
                                        parameter_group_positions,
                                        minimum_attributable_fraction=(
                                            args.v4_audit_minimum_attributable_fraction
                                        ),
                                        maximum_corrective_gain=(
                                            args.v4_audit_maximum_action_gain
                                        ),
                                        minimum_protective_component_retention=(
                                            args.optimizer_restoration_minimum_risk_component_retention
                                        ),
                                        materialize_updates=True,
                                    )
                                )
                            else:
                                groupwise_restoration = arbitrate_optimizer_updates_by_group(
                                    virtual_combined,
                                    virtual_risk,
                                    parameter_group_positions,
                                    minimum_attributable_fraction=(
                                        args.v4_audit_minimum_attributable_fraction
                                    ),
                                    maximum_action_gain=(
                                        args.v4_audit_maximum_action_gain
                                    ),
                                    minimum_risk_component_retention=(
                                        args.optimizer_restoration_minimum_risk_component_retention
                                    ),
                                    materialize_updates=True,
                                )
                    else:
                        virtual_restoration = arbitrate_optimizer_updates(
                            virtual_combined,
                            virtual_risk,
                            minimum_attributable_fraction=(
                                args.v4_audit_minimum_attributable_fraction
                            ),
                            maximum_action_gain=args.v4_audit_maximum_action_gain,
                            materialize_updates=False,
                        )
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if safe_exact_restoration:
                if (
                    safe_exact_injector is None
                    or prepared_action_injection is None
                ):
                    raise RuntimeError(
                        "ActionInjectorV1 preparation was not completed"
                    )
                before_parameters: list[torch.Tensor] = []
                applied_action_injection = safe_exact_injector.step_and_inject_(
                    optimizer, trainable, prepared_action_injection,
                )
            else:
                before_parameters = (
                    [parameter.detach().clone() for parameter in trainable]
                    if audit_optimizer_attribution else []
                )
                optimizer.step()
            if audit_optimizer_attribution:
                standard_updates: list[torch.Tensor | None] = (
                    applied_action_injection.standard_adamw_updates
                    if applied_action_injection is not None else []
                )
                clipped_action: list[torch.Tensor | None] = []
                for position, action in enumerate(
                    conditional_corrective_grad
                    if corrective_only_restoration else projected
                ):
                    effective_action = (
                        None if action is None
                        else action.detach() * clip_retention
                    )
                    if applied_action_injection is None:
                        before = before_parameters[position]
                        parameter = trainable[position]
                        standard_updates.append(before - parameter.detach())
                    clipped_action.append(effective_action)
                virtual_residual = [
                    None if actual is None or expected is None
                    else actual - expected
                    for actual, expected in zip(standard_updates, virtual_combined)
                ]
                standard_update_norm = _gradient_norm(standard_updates)
                virtual_error = (
                    applied_action_injection.virtual_adamw_relative_error
                    if applied_action_injection is not None else (
                        _gradient_norm(virtual_residual) / standard_update_norm
                        if standard_update_norm > 0 else 0.0
                    )
                )
                if materialize_optimizer_restoration:
                    if groupwise_restoration is None:
                        raise RuntimeError(
                            "registered optimizer restoration was not constructed"
                        )
                    if safe_exact_restoration:
                        if applied_action_injection is None:
                            raise RuntimeError(
                                "ActionInjectorV1 did not return an applied receipt"
                            )
                        reconciliation = applied_action_injection.reconciliation
                    else:
                        materialize_descent_updates_(
                            trainable,
                            before_parameters,
                            groupwise_restoration.updates,
                        )
                    if (
                        args.reconcile_restored_adamw_first_moment
                        and not safe_exact_restoration
                    ):
                        reconciliation = (
                            reconcile_adamw_first_moments_to_materialized_updates_(
                                optimizer,
                                trainable,
                                before_parameters,
                                groupwise_restoration.updates,
                            )
                        )
                    if args.reconcile_restored_adamw_first_moment:
                        all_restoration_reconciliation_error.append(float(
                            reconciliation[
                                "same_step_reconstruction_relative_error"
                            ]
                        ))
                        all_restoration_materialization_error.append(float(
                            reconciliation["materialized_update_relative_error"]
                        ))
                        all_restoration_fp32_replay_error.append(float(
                            reconciliation[
                                "same_step_fp32_parameter_replay_relative_error"
                            ]
                        ))
                        all_restoration_first_moment_change.append(float(
                            reconciliation["first_moment_change_fraction"]
                        ))
                    actual_updates = groupwise_restoration.updates
                    all_restoration_target_reached.append(float(
                        groupwise_restoration.all_groups_target_reached
                    ))
                    all_restoration_risk_retention.append(float(
                        groupwise_restoration.minimum_group_risk_component_retention
                    ))
                    all_restoration_gain.append(float(max(
                        report.action_gain
                        for report in groupwise_restoration.parameter_groups.values()
                    )))
                    if safe_exact_restoration:
                        all_restoration_hard_floor_enforced.append(float(
                            groupwise_restoration
                            .all_groups_protective_floor_enforced
                        ))
                        all_restoration_exact_fraction_error.append(float(
                            groupwise_restoration
                            .maximum_group_attributable_fraction_abs_error
                        ))
                        all_restoration_update_norm_ratio.append(float(
                            groupwise_restoration
                            .maximum_group_update_norm_ratio_to_original
                        ))
                    logs["optimizer_restoration_materialized"].append(1.0)
                    if safe_exact_restoration:
                        logs[
                            "optimizer_restoration_hard_protective_floor_enforced"
                        ].append(float(
                            groupwise_restoration
                            .all_groups_protective_floor_enforced
                        ))
                        logs[
                            "optimizer_restoration_exact_fraction_max_abs_error"
                        ].append(float(
                            groupwise_restoration
                            .maximum_group_attributable_fraction_abs_error
                        ))
                        logs[
                            "optimizer_restoration_max_update_norm_ratio"
                        ].append(float(
                            groupwise_restoration
                            .maximum_group_update_norm_ratio_to_original
                        ))
                    logs[
                        "optimizer_restoration_all_groups_target_reached"
                    ].append(float(
                        groupwise_restoration.all_groups_target_reached
                    ))
                    logs[
                        "optimizer_restoration_minimum_group_risk_component_retention"
                    ].append(float(
                        groupwise_restoration.minimum_group_risk_component_retention
                    ))
                    for name, report in (
                        groupwise_restoration.parameter_groups.items()
                    ):
                        logs[
                            f"optimizer_restoration_action_gain::{name}"
                        ].append(report.action_gain)
                        logs[
                            f"optimizer_restoration_final_attributable_fraction::{name}"
                        ].append(report.final_attributable_fraction)
                        logs[
                            f"optimizer_restoration_risk_component_retention::{name}"
                        ].append(report.risk_component_retention)
                        logs[
                            f"optimizer_restoration_target_reached::{name}"
                        ].append(float(report.target_reached))
                        logs[
                            f"optimizer_restoration_risk_constraint_active::{name}"
                        ].append(float(report.risk_constraint_active))
                else:
                    actual_updates = standard_updates
                update_norm = _gradient_norm(actual_updates)
                action_norm = _gradient_norm(clipped_action)
                update_action_dot = _gradient_dot(
                    actual_updates, clipped_action,
                )
                alignment = (
                    update_action_dot / (update_norm * action_norm)
                    if update_norm > 0 and action_norm > 0 else 0.0
                )
                all_optimizer_alignment.append(alignment)
                logs["optimizer_update_norm"].append(update_norm)
                logs["optimizer_action_alignment_cosine"].append(alignment)
                attributable = [
                    None if actual_update is None else (
                        actual_update if risk_update is None
                        else actual_update - risk_update
                    )
                    for actual_update, risk_update in zip(
                        actual_updates, virtual_attribution_baseline,
                    )
                ]
                attributable_norm = _gradient_norm(attributable)
                virtual_norm = update_norm
                attributable_fraction = (
                    attributable_norm / virtual_norm if virtual_norm > 0 else 0.0
                )
                attributable_alignment = (
                    _gradient_dot(attributable, clipped_action)
                    / (attributable_norm * action_norm)
                    if attributable_norm > 0 and action_norm > 0 else 0.0
                )
                clipped_corrective = [
                    None if gradient is None
                    else gradient.detach() * clip_retention
                    for gradient in corrective_grad
                ]
                clipped_corrective_norm = _gradient_norm(clipped_corrective)
                attributable_corrective_alignment = (
                    _gradient_dot(attributable, clipped_corrective)
                    / (attributable_norm * clipped_corrective_norm)
                    if attributable_norm > 0 and clipped_corrective_norm > 0 else 0.0
                )
                clipped_risk = [
                    None if gradient is None
                    else gradient.detach() * clip_retention
                    for gradient in risk_grad
                ]
                directional_restoration = rotate_update_toward_action(
                    actual_updates,
                    clipped_action,
                    risk_gradients=clipped_risk,
                    minimum_action_alignment=(
                        args.v4_audit_minimum_update_action_alignment
                    ),
                    maximum_action_coefficient=(
                        args.v4_audit_maximum_update_action_coefficient
                    ),
                    materialize_updates=False,
                )
                all_optimizer_action_fraction.append(attributable_fraction)
                all_optimizer_attributable_alignment.append(attributable_alignment)
                all_optimizer_corrective_alignment.append(
                    attributable_corrective_alignment
                )
                all_optimizer_virtual_error.append(virtual_error)
                for name, positions in parameter_group_positions.items():
                    group_actual = _take_gradient_positions(
                        actual_updates, positions,
                    )
                    group_action = _take_gradient_positions(
                        clipped_action, positions,
                    )
                    group_attributable = _take_gradient_positions(
                        attributable, positions,
                    )
                    group_update_norm = _gradient_norm(group_actual)
                    group_action_norm = _gradient_norm(group_action)
                    group_attributable_norm = _gradient_norm(group_attributable)
                    group_virtual_norm = group_update_norm
                    group_alignment = (
                        _gradient_dot(group_actual, group_action)
                        / (group_update_norm * group_action_norm)
                        if group_update_norm > 0 and group_action_norm > 0 else 0.0
                    )
                    group_attributable_fraction = (
                        group_attributable_norm / group_virtual_norm
                        if group_virtual_norm > 0 else 0.0
                    )
                    group_attributable_alignment = (
                        _gradient_dot(group_attributable, group_action)
                        / (group_attributable_norm * group_action_norm)
                        if group_attributable_norm > 0 and group_action_norm > 0 else 0.0
                    )
                    group_corrective = _take_gradient_positions(
                        clipped_corrective, positions,
                    )
                    group_corrective_norm = _gradient_norm(group_corrective)
                    group_attributable_corrective_alignment = (
                        _gradient_dot(group_attributable, group_corrective)
                        / (group_attributable_norm * group_corrective_norm)
                        if group_attributable_norm > 0 and group_corrective_norm > 0
                        else 0.0
                    )
                    parameter_group_diagnostics[name][
                        "optimizer_action_alignment"
                    ].append(group_alignment)
                    parameter_group_diagnostics[name][
                        "optimizer_action_attributable_fraction"
                    ].append(group_attributable_fraction)
                    parameter_group_diagnostics[name][
                        "optimizer_action_attributable_alignment"
                    ].append(group_attributable_alignment)
                    parameter_group_diagnostics[name][
                        "optimizer_action_attributable_corrective_alignment"
                    ].append(group_attributable_corrective_alignment)
                logs["optimizer_action_attributable_update_norm"].append(
                    attributable_norm
                )
                logs["optimizer_action_attributable_update_fraction"].append(
                    attributable_fraction
                )
                logs["optimizer_action_attributable_alignment_cosine"].append(
                    attributable_alignment
                )
                logs[
                    "optimizer_action_attributable_corrective_alignment_cosine"
                ].append(attributable_corrective_alignment)
                logs["optimizer_virtual_step_relative_error"].append(virtual_error)
                if not materialize_optimizer_restoration:
                    if virtual_restoration is None:
                        raise RuntimeError("v4 optimizer-space audit was not constructed")
                    logs["v4_audit_action_gain"].append(
                        virtual_restoration.action_gain
                    )
                    logs["v4_audit_norm_rescale"].append(
                        virtual_restoration.norm_rescale
                    )
                    logs["v4_audit_original_attributable_fraction"].append(
                        virtual_restoration.original_attributable_fraction
                    )
                    logs["v4_audit_restored_attributable_fraction"].append(
                        virtual_restoration.final_attributable_fraction
                    )
                    logs["v4_audit_target_reached"].append(
                        float(virtual_restoration.target_reached)
                    )
                    logs["v4_audit_gain_cap_hit"].append(
                        float(virtual_restoration.gain_cap_hit)
                    )
                logs["v4_audit_directional_action_coefficient"].append(
                    directional_restoration.action_coefficient
                )
                logs["v4_audit_directional_norm_rescale"].append(
                    directional_restoration.norm_rescale
                )
                logs["v4_audit_directional_alignment_before"].append(
                    directional_restoration.action_alignment_before
                )
                logs["v4_audit_directional_alignment_after"].append(
                    directional_restoration.action_alignment_after
                )
                logs["v4_audit_directional_target_reached"].append(
                    float(directional_restoration.target_reached)
                )
                logs["v4_audit_directional_risk_constraint_active"].append(
                    float(directional_restoration.risk_constraint_active)
                )
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            completed_steps = step_index + 1
            if (
                completed_steps == 1
                or completed_steps % args.progress_every_steps == 0
                or completed_steps == steps
            ):
                elapsed = time.time() - epoch_started
                print(json.dumps({
                    "status": "noise_corrected_v3_training_progress",
                    "arm": args.arm,
                    "seed": args.seed,
                    "epoch": epoch + 1,
                    "epochs": args.epochs,
                    "optimizer_step": completed_steps,
                    "optimizer_steps": steps,
                    "epoch_fraction": float(completed_steps / steps),
                    "elapsed_seconds": elapsed,
                    "seconds_per_step": float(elapsed / completed_steps),
                    "estimated_epoch_seconds_remaining": float(
                        elapsed / completed_steps * (steps - completed_steps)
                    ),
                }), flush=True)

        expected_actions = sum(
            len(example.actions) for panel in (corrective, robust, harmful) for example in panel
        )
        expected_queries = {
            "corrective": len(corrective),
            "robust": len(robust),
            "harmful": len(harmful),
            "protective": len(protective),
        }
        for kind, expected in expected_queries.items():
            if len(seen_queries[kind]) != expected:
                raise RuntimeError(f"v3 epoch did not cover every {kind} query")
        if len(seen_actions) != expected_actions:
            raise RuntimeError("v3 epoch did not cover every selected action")
        if min(corrective_action_exposures.values(), default=0) < 1:
            raise RuntimeError("v3 corrective recycling lost an action")
        if not args.v3_corrective_recycle_full_dose and any(
            not np.isclose(value, 1.0)
            for value in corrective_action_weighted_dose.values()
        ):
            raise RuntimeError("normalized corrective recycling changed epoch action dose")
        if args.v3_corrective_recycle_full_dose and any(
            not np.isclose(value, corrective_recycle_factor)
            for value in corrective_action_weighted_dose.values()
        ):
            raise RuntimeError("full-dose corrective recycling lost batch-equal epoch dose")
        corrective_exposure_values = np.asarray(
            list(corrective_action_exposures.values()), dtype=float,
        )
        corrective_dose_values = np.asarray(
            list(corrective_action_weighted_dose.values()), dtype=float,
        )
        if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development:
            _validate_registered_best_action_v6_schedule_geometry(
                schedule_geometry,
                context=f"epoch-{epoch + 1}",
            )
            if (
                not np.all(corrective_exposure_values == 4)
                or not np.allclose(
                    corrective_dose_values, 4.0, rtol=1e-12, atol=1e-12,
                )
            ):
                raise RuntimeError(
                    "best-action v6 cap-safe schedule did not expose every "
                    "corrective action at exactly the registered four-dose"
                )
        mechanism_mass_balanced = {
            kind: bool(
                values
                and np.allclose(
                    list(values.values()), next(iter(values.values())),
                    rtol=1e-10, atol=1e-10,
                )
            )
            for kind, values in mechanism_epoch_mass.items()
        }
        if not mechanism_mass_balanced["corrective"]:
            raise RuntimeError(
                "actual scheduled corrective N/P/A4 epoch mass drifted: "
                f"{mechanism_epoch_mass}"
            )
        epoch_transfer_edge_breakdown = _finalize_transfer_edge_breakdown(
            epoch_transfer_edge_totals
        )
        if active_arm and (
            "all" not in epoch_transfer_edge_breakdown
            or not any(key.startswith("family=") for key in epoch_transfer_edge_breakdown)
            or not any(
                key.startswith("source_family=")
                for key in epoch_transfer_edge_breakdown
            )
        ):
            raise RuntimeError("v3 epoch transfer-edge strata are incomplete")
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "optimizer_steps": steps,
            "optimizer_counterfactual_attribution_steps": (
                (
                    steps if materialize_optimizer_restoration
                    else len(attribution_steps)
                ) if active_arm else 0
            ),
            "optimizer_counterfactual_attribution_step_fraction": (
                (
                    1.0 if materialize_optimizer_restoration
                    else float(len(attribution_steps) / steps)
                ) if active_arm else 0.0
            ),
            "optimizer_counterfactual_attribution_is_sampled_not_exhaustive": bool(
                active_arm
                and not materialize_optimizer_restoration
                and len(attribution_steps) < steps
            ),
            "action_active_steps": steps if args.arm != "clean_control" else 0,
            "action_active_fraction": 1.0 if args.arm != "clean_control" else 0.0,
            "protective_microbatches": len(risk_batches),
            "protective_microbatches_per_step_min": min(risk_counts),
            "protective_microbatches_per_step_max": max(risk_counts),
            "robust_auxiliary_scale": auxiliary_panel_scale["robust"],
            "harmful_auxiliary_scale": auxiliary_panel_scale["harmful"],
            "robust_active_step_fraction": robust_active_step_fraction,
            "harmful_active_step_fraction": harmful_active_step_fraction,
            "auxiliary_active_step_fraction": float(
                np.mean(np.asarray(auxiliary_step_load) > 0)
            ),
            "auxiliary_overlap_step_fraction": float(
                np.mean(np.asarray(auxiliary_step_load) > 1)
            ),
            "maximum_auxiliary_batches_per_step": int(max(auxiliary_step_load)),
            "configured_maximum_auxiliary_microbatches_per_step": int(
                args.v3_maximum_auxiliary_microbatches_per_step
            ),
            "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step": bool(
                all(
                    scale <= 1.0
                    and np.isclose(
                        scale * count,
                        min(count, steps),
                    )
                    for scale, count in (
                        (auxiliary_panel_scale["robust"], len(robust_batches)),
                        (auxiliary_panel_scale["harmful"], len(harmful_batches)),
                    )
                )
            ),
            "sparse_auxiliary_batches_evenly_interleaved": True,
            "auxiliary_batch_count": int(len(robust_batches) + len(harmful_batches)),
            "avoidable_auxiliary_step_overlap": bool(
                len(robust_batches) + len(harmful_batches) <= steps
                and max(auxiliary_step_load) > 1
            ),
            "inner_semantic_projection": {
                "auxiliary_projection_retention_p10": float(np.quantile(
                    logs["auxiliary_inner_projection_retention"], 0.10,
                )) if active_arm else 1.0,
                "corrective_direction_preservation_minimum": float(min(
                    logs["corrective_direction_preservation_ratio"]
                )) if active_arm else 1.0,
                "corrective_direction_preserved_every_step": bool(
                    not active_arm or min(
                        logs["corrective_direction_preservation_ratio"]
                    ) >= 1.0 - 1e-6
                ),
                "parameter_group_corrective_direction_preservation_minimum": {
                    name: float(min(logs[
                        f"corrective_direction_preservation_ratio::{name}"
                    ])) if active_arm else 1.0
                    for name in parameter_group_positions
                },
            },
            "semantic_branch_gradient_diagnostics": {
                "raw_norm": {
                    name: {
                        "all_step_observations": len(logs[
                            f"raw_branch_gradient_norm_all_steps::{name}"
                        ]),
                        "nonzero_step_observations": len(logs[
                            f"raw_branch_gradient_norm_active_steps::{name}"
                        ]),
                        "all_step_median": float(np.median(logs[
                            f"raw_branch_gradient_norm_all_steps::{name}"
                        ])) if active_arm else 0.0,
                        "active_step_median": float(np.median(logs[
                            f"raw_branch_gradient_norm_active_steps::{name}"
                        ])) if active_arm and logs[
                            f"raw_branch_gradient_norm_active_steps::{name}"
                        ] else 0.0,
                    }
                    for name in ("corrective", "robust", "harmful", "protective")
                },
                "raw_cosine": {
                    pair: {
                        "coactive_step_observations": len(logs[
                            f"raw_branch_gradient_cosine::{pair}"
                        ]),
                        "median": float(np.median(logs[
                            f"raw_branch_gradient_cosine::{pair}"
                        ])) if logs[f"raw_branch_gradient_cosine::{pair}"] else None,
                    }
                    for pair in (
                        "corrective_vs_robust",
                        "corrective_vs_harmful",
                        "corrective_vs_protective",
                        "robust_vs_protective",
                        "harmful_vs_protective",
                    )
                },
            },
            "protective_equal_microbatch_epoch_scale": protective_microbatch_scale,
            "sparse_auxiliary_global_duty_compensation_applied": False,
            "epoch_corrective_queries": len(seen_queries["corrective"]),
            "epoch_corrective_query_exposures": int(sum(corrective_query_exposures.values())),
            "corrective_action_exposures": int(sum(corrective_action_exposures.values())),
            "corrective_action_exposure_min": float(np.min(corrective_exposure_values)),
            "corrective_action_exposure_median": float(np.median(corrective_exposure_values)),
            "corrective_action_exposure_max": float(np.max(corrective_exposure_values)),
            "corrective_action_weighted_dose_min": float(np.min(corrective_dose_values)),
            "corrective_action_weighted_dose_median": float(np.median(corrective_dose_values)),
            "corrective_action_weighted_dose_max": float(np.max(corrective_dose_values)),
            "corrective_batch_recycle_factor": float(corrective_recycle_factor),
            "original_corrective_batches": int(
                schedule_geometry["original_corrective_batches"]
            ),
            "cap_safe_corrective_batches": int(
                schedule_geometry["cap_safe_corrective_batches"]
            ),
            "corrective_batches_added_by_cap_safe_repartition": int(
                schedule_geometry[
                    "corrective_batches_added_by_cap_safe_repartition"
                ]
            ),
            "cap_safe_corrective_batch_size_counts": {
                str(size): count
                for size, count in corrective_batch_size_counts.items()
            },
            "cap_safe_corrective_repartition_preserved_query_order_and_coverage": True,
            "schedule_geometry": schedule_geometry,
            "mechanism_epoch_mass": {
                kind: dict(values) for kind, values in mechanism_epoch_mass.items()
            },
            "mechanism_epoch_mass_balanced": mechanism_mass_balanced,
            "corrective_transfer_edge_breakdown": epoch_transfer_edge_breakdown,
            "corrective_pooled_active_transfer_fraction": float(
                epoch_transfer_edge_breakdown.get("all", {}).get(
                    "active_fraction", 0.0,
                )
            ),
            "epoch_robust_queries": len(seen_queries["robust"]),
            "epoch_harmful_queries": len(seen_queries["harmful"]),
            "epoch_protective_queries": len(seen_queries["protective"]),
            "epoch_selected_actions": len(seen_actions),
            **{key: float(np.mean(value)) for key, value in logs.items() if value},
        }
        history.append(record)
        if (
            reference_refresh is not None
            and args.reference_refresh_every_epochs > 0
            and epoch + 1 < args.epochs
            and (epoch + 1) % args.reference_refresh_every_epochs == 0
        ):
            corrective, robust, harmful, protective = reference_refresh()
            if not all((corrective, robust, harmful, protective)):
                raise RuntimeError("v3 hard-reference refresh returned an empty panel")
            record["hard_references_refreshed_for_next_epoch"] = True
        print(json.dumps(record), flush=True)

    active_arm = args.arm in {"routed_direct", "shuffled_action_control"}
    retention_p10 = float(np.quantile(all_retention, 0.10)) if all_retention else 1.0
    signal_gate = bool(retention_p10 >= args.minimum_action_retention_p10)
    clip_event_fraction = (
        float(np.mean(np.asarray(all_clip_retention) < 1.0 - 1e-7))
        if all_clip_retention else 0.0
    )
    clip_gate = bool(clip_event_fraction <= args.maximum_clip_event_fraction)
    optimizer_alignment_p10 = (
        float(np.quantile(all_optimizer_alignment, 0.10))
        if all_optimizer_alignment else 0.0
    )
    optimizer_alignment_gate = (
        minimum_gate_passed(
            optimizer_alignment_p10, args.minimum_optimizer_action_alignment_p10,
        )
        if active_arm else True
    )
    attributable_fraction_p10 = (
        float(np.quantile(all_optimizer_action_fraction, 0.10))
        if all_optimizer_action_fraction else 0.0
    )
    attributable_alignment_p10 = (
        float(np.quantile(all_optimizer_attributable_alignment, 0.10))
        if all_optimizer_attributable_alignment else 0.0
    )
    corrective_direction_retention_p10 = (
        float(np.quantile(all_corrective_direction_retention, 0.10))
        if all_corrective_direction_retention else 0.0
    )
    optimizer_corrective_alignment_p10 = (
        float(np.quantile(all_optimizer_corrective_alignment, 0.10))
        if all_optimizer_corrective_alignment else 0.0
    )
    attributable_fraction_gate = (
        minimum_gate_passed(
            attributable_fraction_p10,
            args.minimum_optimizer_action_attributable_fraction_p10,
        )
        if active_arm else True
    )
    attributable_alignment_gate = (
        minimum_gate_passed(
            attributable_alignment_p10,
            args.minimum_optimizer_action_alignment_p10,
        )
        if active_arm else True
    )
    corrective_direction_retention_gate = (
        minimum_gate_passed(
            corrective_direction_retention_p10,
            args.minimum_corrective_direction_retention_p10,
        ) if active_arm else True
    )
    optimizer_corrective_alignment_gate = (
        minimum_gate_passed(
            optimizer_corrective_alignment_p10,
            args.minimum_optimizer_action_alignment_p10,
        ) if active_arm else True
    )
    virtual_step_max_relative_error = max(all_optimizer_virtual_error, default=0.0)
    virtual_step_gate = bool(virtual_step_max_relative_error <= 1e-3)
    minimum_group_retention = float(
        args.minimum_optimizer_action_attributable_fraction_p10
    )
    parameter_group_signal: dict[str, dict[str, float | int | bool]] = {}
    for name, values in parameter_group_diagnostics.items():
        preprojection = np.asarray(
            values["action_gradient_preprojection_norm"], dtype=float,
        )
        group_retention = np.asarray(
            values["pcgrad_clip_retention"], dtype=float,
        )
        group_alignment = np.asarray(
            values["optimizer_action_alignment"], dtype=float,
        )
        group_fraction = np.asarray(
            values["optimizer_action_attributable_fraction"], dtype=float,
        )
        group_attributable_alignment = np.asarray(
            values["optimizer_action_attributable_alignment"], dtype=float,
        )
        group_corrective_retention = np.asarray(
            values["corrective_direction_risk_clip_retention"], dtype=float,
        )
        group_corrective_alignment = np.asarray(
            values["optimizer_action_attributable_corrective_alignment"],
            dtype=float,
        )
        group_protective_reach = np.asarray(
            values["protective_gradient_full_parameter_reach"], dtype=float,
        )
        observations = int(len(preprojection))
        present = bool(observations and np.all(preprojection > 0))
        retention_value = (
            float(np.quantile(group_retention, 0.10)) if observations else 0.0
        )
        alignment_value = (
            float(np.quantile(group_alignment, 0.10)) if observations else 0.0
        )
        fraction_value = (
            float(np.quantile(group_fraction, 0.10)) if observations else 0.0
        )
        attributable_alignment_value = (
            float(np.quantile(group_attributable_alignment, 0.10))
            if observations else 0.0
        )
        corrective_retention_value = (
            float(np.quantile(group_corrective_retention, 0.10))
            if observations else 0.0
        )
        corrective_alignment_value = (
            float(np.quantile(group_corrective_alignment, 0.10))
            if observations else 0.0
        )
        gate = bool(
            present
            and minimum_gate_passed(retention_value, minimum_group_retention)
            and minimum_gate_passed(
                alignment_value, args.minimum_optimizer_action_alignment_p10,
            )
            and minimum_gate_passed(
                fraction_value,
                args.minimum_optimizer_action_attributable_fraction_p10,
            )
            and minimum_gate_passed(
                attributable_alignment_value,
                args.minimum_optimizer_action_alignment_p10,
            )
            and minimum_gate_passed(
                corrective_retention_value,
                args.minimum_corrective_direction_retention_p10,
            )
            and minimum_gate_passed(
                corrective_alignment_value,
                args.minimum_optimizer_action_alignment_p10,
            )
            and (
                not args.materialize_optimizer_update_restoration
                or (
                    len(group_protective_reach) == observations
                    and np.all(group_protective_reach == 1.0)
                )
            )
        )
        parameter_group_signal[name] = {
            "observations": observations,
            "action_gradient_preprojection_norm_median": (
                float(np.median(preprojection)) if observations else 0.0
            ),
            "action_gradient_reaches_group_on_every_audit_step": present,
            "pcgrad_clip_action_retention_p10": retention_value,
            "minimum_pcgrad_clip_action_retention_p10": minimum_group_retention,
            "optimizer_action_alignment_p10": alignment_value,
            "optimizer_action_attributable_update_fraction_p10": fraction_value,
            "optimizer_action_attributable_alignment_p10": (
                attributable_alignment_value
            ),
            "corrective_direction_retention_after_risk_and_clip_p10": (
                corrective_retention_value
            ),
            "optimizer_action_attributable_corrective_alignment_p10": (
                corrective_alignment_value
            ),
            "protective_gradient_reaches_every_parameter_on_every_audit_step": bool(
                observations
                and len(group_protective_reach) == observations
                and np.all(group_protective_reach == 1.0)
            ),
            "gate_passed": gate if active_arm else True,
        }
    parameter_group_signal_gate = bool(
        not active_arm
        or (
            set(parameter_group_signal) == {"head", "backbone"}
            and all(value["gate_passed"] for value in parameter_group_signal.values())
        )
    )
    restoration_target_reached_fraction = (
        float(np.mean(all_restoration_target_reached))
        if all_restoration_target_reached else 1.0
    )
    restoration_target_coverage_gate = restoration_coverage_gate(
        active_arm=active_arm,
        materialized=bool(args.materialize_optimizer_update_restoration),
        observed_fraction=restoration_target_reached_fraction,
        minimum_fraction=args.minimum_optimizer_restoration_target_reached_fraction,
    )
    minimum_observed_protective_retention = (
        float(min(all_restoration_risk_retention))
        if all_restoration_risk_retention else 1.0
    )
    hard_protective_floor_gate = bool(
        args.optimizer_restoration_scope != "safe_exact_corrective"
        or minimum_gate_passed(
            minimum_observed_protective_retention,
            args.optimizer_restoration_minimum_risk_component_retention,
        )
    )
    exact_fraction_error_max = float(
        max(all_restoration_exact_fraction_error, default=0.0)
    )
    exact_fraction_gate = bool(
        args.optimizer_restoration_scope != "safe_exact_corrective"
        or exact_fraction_error_max <= 2e-6
    )
    safe_exact_norm_ratio_max = float(
        max(all_restoration_update_norm_ratio, default=1.0)
    )
    safe_exact_norm_gate = bool(
        args.optimizer_restoration_scope != "safe_exact_corrective"
        or safe_exact_norm_ratio_max
        <= args.maximum_safe_exact_update_norm_ratio + 1e-7
    )
    signal_gate_failure = bool(active_arm and (
        not signal_gate
        or not clip_gate
        or not optimizer_alignment_gate
        or not attributable_fraction_gate
        or not attributable_alignment_gate
        or not corrective_direction_retention_gate
        or not optimizer_corrective_alignment_gate
        or not virtual_step_gate
        or not parameter_group_signal_gate
        or not restoration_target_coverage_gate
        or not hard_protective_floor_gate
        or not exact_fraction_gate
        or not safe_exact_norm_gate
    ))
    if signal_gate_failure and not args.continue_after_signal_gate_failure:
        raise RuntimeError(
            f"v3 signal gate failed: retention p10={retention_p10:.4f}, "
            f"clip-event fraction={clip_event_fraction:.4f}, "
            f"optimizer action alignment p10={optimizer_alignment_p10:.4f}, "
            f"action-attributable update fraction p10="
            f"{attributable_fraction_p10:.4f}, "
            f"action-attributable alignment p10="
            f"{attributable_alignment_p10:.4f}, "
            f"corrective direction retention p10="
            f"{corrective_direction_retention_p10:.4f}, "
            f"optimizer corrective alignment p10="
            f"{optimizer_corrective_alignment_p10:.4f}, "
            f"virtual AdamW error={virtual_step_max_relative_error:.6g}, "
            f"parameter groups={parameter_group_signal}"
        )
    if not epoch_schedule_geometries or any(
        geometry != epoch_schedule_geometries[0]
        for geometry in epoch_schedule_geometries[1:]
    ):
        raise RuntimeError("v3 schedule geometry changed across epochs")
    final_schedule_geometry = epoch_schedule_geometries[0]
    schedule_report = {
        "mode": "bounded_accumulation_action_active_with_corrective_recycling",
        "schedule_geometry": final_schedule_geometry,
        "schedule_geometry_by_epoch": epoch_schedule_geometries,
        "schedule_geometry_matches_pre_model_preflight": bool(
            expected_schedule_geometry is None
            or final_schedule_geometry == expected_schedule_geometry
        ),
        "cap_safe_corrective_repartition_used": bool(
            final_schedule_geometry["cap_safe_repartition_required"]
        ),
        "cap_safe_corrective_repartition_preserved_query_order_and_coverage": True,
        "original_corrective_batches": int(
            final_schedule_geometry["original_corrective_batches"]
        ),
        "cap_safe_corrective_batches": int(
            final_schedule_geometry["cap_safe_corrective_batches"]
        ),
        "corrective_batches_added_by_cap_safe_repartition": int(
            final_schedule_geometry[
                "corrective_batches_added_by_cap_safe_repartition"
            ]
        ),
        "required_optimizer_steps": int(
            final_schedule_geometry["required_optimizer_steps"]
        ),
        "effective_corrective_recycle_factor": float(
            final_schedule_geometry["effective_corrective_recycle_factor"]
        ),
        "all_unique_action_panels_covered_before_recycling": True,
        "all_protective_queries_consumed_once_per_epoch": True,
        "maximum_protective_microbatches_per_step": maximum_risk_microbatches,
        "minimum_protective_microbatches_per_step": minimum_risk_microbatches,
        "configured_maximum_protective_microbatches_per_step": (
            args.v3_maximum_protective_microbatches_per_step
        ),
        "configured_maximum_corrective_recycle_factor": (
            args.v3_maximum_corrective_recycle_factor
        ),
        "protect_only_optimizer_tail": False,
        "corrective_recycle_full_dose": bool(args.v3_corrective_recycle_full_dose),
        "full_dose_recycling_equalizes_cumulative_original_batch_mass": True,
        "corrective_recycling_prevents_sparse_action_optimizer_schedule": True,
        "maximum_corrective_action_exposure_per_epoch": float(max(
            record["corrective_action_exposure_max"] for record in history
        )),
        "minimum_corrective_action_exposure_per_epoch": float(min(
            record["corrective_action_exposure_min"] for record in history
        )),
        "robust_harmful_single_exposure_bounded_scale": True,
        "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step": all(
            record["dense_auxiliary_epoch_mass_capped_at_one_batch_per_step"]
            for record in history
        ),
        "maximum_auxiliary_microbatches_per_step": int(max(
            record["maximum_auxiliary_batches_per_step"] for record in history
        )),
        "configured_maximum_auxiliary_microbatches_per_step": (
            args.v3_maximum_auxiliary_microbatches_per_step
        ),
        "robust_harmful_evenly_interleaved_across_epoch": True,
        "robust_harmful_avoidable_step_overlap": bool(any(
            record["avoidable_auxiliary_step_overlap"] for record in history
        )),
        "sparse_auxiliary_global_duty_compensation_applied": False,
        "partial_batch_query_mass_scaled_to_registered_size": True,
        "protective_microbatches_have_equal_epoch_weight": True,
        "clean_control_skips_no_gradient_action_forwards": True,
        "active_arm_epoch_transfer_edge_strata_recorded": bool(
            not active_arm
            or all(
                "all" in record["corrective_transfer_edge_breakdown"]
                and any(
                    key.startswith("family=")
                    for key in record["corrective_transfer_edge_breakdown"]
                )
                and any(
                    key.startswith("source_family=")
                    for key in record["corrective_transfer_edge_breakdown"]
                )
                for record in history
            )
        ),
        "actual_scheduled_corrective_mechanism_epoch_mass_equalized": all(
            record["mechanism_epoch_mass_balanced"]["corrective"]
            for record in history
        ),
        "auxiliary_projected_against_corrective_before_fullgraph_risk": True,
        "corrective_robust_harmful_protective_gradients_separately_recorded": all(
            all(
                record["semantic_branch_gradient_diagnostics"]["raw_norm"][name][
                    "all_step_observations"
                ] == record["optimizer_steps"]
                for name in ("corrective", "robust", "harmful", "protective")
            )
            for record in history
        ) if active_arm else True,
        "each_action_branch_vs_protective_cosine_recorded": all(
            all(
                record["semantic_branch_gradient_diagnostics"]["raw_cosine"][pair][
                    "coactive_step_observations"
                ] > 0
                or record["semantic_branch_gradient_diagnostics"]["raw_norm"][
                    pair.split("_vs_", 1)[0]
                ]["nonzero_step_observations"] == 0
                for pair in (
                    "corrective_vs_protective",
                    "robust_vs_protective",
                    "harmful_vs_protective",
                )
            )
            for record in history
        ) if active_arm else True,
        "corrective_direction_preserved_through_inner_semantic_projection": all(
            record["inner_semantic_projection"][
                "corrective_direction_preserved_every_step"
            ]
            for record in history
        ),
        "auxiliary_inner_projection_retention_p10": float(np.quantile([
            record["inner_semantic_projection"][
                "auxiliary_projection_retention_p10"
            ]
            for record in history
        ], 0.10)),
        "action_branch_graphs_backpropagated_sequentially": True,
        "simultaneous_corrective_robust_harmful_graph_retention": False,
        "hard_reference_refresh_every_epochs": args.reference_refresh_every_epochs,
        "hard_reference_refresh_uses_outer_training_graph_only": True,
        "hard_negative_reference_strategy": (
            "initial_clean_action_control_row_union_epoch_current_by_molecule"
        ),
        "optimizer_action_alignment_p10": optimizer_alignment_p10,
        "minimum_optimizer_action_alignment_p10": args.minimum_optimizer_action_alignment_p10,
        "optimizer_action_alignment_gate_passed": optimizer_alignment_gate,
        "optimizer_action_alignment_steps": len(all_optimizer_alignment),
        "optimizer_action_alignment_sampling": (
            "every_action_active_optimizer_step"
            if args.materialize_optimizer_update_restoration and active_arm
            else "uniform_epoch_step_positions_including_endpoints"
        ),
        "optimizer_action_attributable_update_fraction_p10": attributable_fraction_p10,
        "minimum_optimizer_action_attributable_fraction_p10": (
            args.minimum_optimizer_action_attributable_fraction_p10
        ),
        "optimizer_action_attributable_update_fraction_gate_passed": (
            attributable_fraction_gate
        ),
        "optimizer_action_attributable_alignment_p10": attributable_alignment_p10,
        "optimizer_action_attributable_alignment_gate_passed": (
            attributable_alignment_gate
        ),
        "corrective_direction_retention_after_risk_and_clip_p10": (
            corrective_direction_retention_p10
        ),
        "minimum_corrective_direction_retention_p10": (
            args.minimum_corrective_direction_retention_p10
        ),
        "corrective_direction_retention_gate_passed": (
            corrective_direction_retention_gate
        ),
        "optimizer_action_attributable_corrective_alignment_p10": (
            optimizer_corrective_alignment_p10
        ),
        "optimizer_action_attributable_corrective_alignment_gate_passed": (
            optimizer_corrective_alignment_gate
        ),
        "optimizer_counterfactual_attribution_steps": len(
            all_optimizer_virtual_error
        ),
        "optimizer_counterfactual_attribution_steps_per_epoch": (
            int(final_schedule_geometry["required_optimizer_steps"])
            if args.materialize_optimizer_update_restoration and active_arm
            else args.optimizer_attribution_steps_per_epoch
        ),
        "optimizer_counterfactual_attribution_actual_steps_by_epoch": [
            int(record["optimizer_counterfactual_attribution_steps"])
            for record in history
        ],
        "optimizer_counterfactual_attribution_total_optimizer_steps_by_epoch": [
            int(record["optimizer_steps"]) for record in history
        ],
        "optimizer_counterfactual_attribution_step_fraction_by_epoch": [
            float(record["optimizer_counterfactual_attribution_step_fraction"])
            for record in history
        ],
        "optimizer_counterfactual_attribution_is_sampled_not_exhaustive": bool(
            active_arm and any(
                record[
                    "optimizer_counterfactual_attribution_is_sampled_not_exhaustive"
                ]
                for record in history
            )
        ),
        "optimizer_counterfactual_attribution_scope": (
            (
                (
                    "hard_safe_exact_corrective_increment_over_safety_projected_"
                    "protective_plus_auxiliary_on_every_action_active_optimizer_step"
                    if args.optimizer_restoration_scope == "safe_exact_corrective"
                    else "corrective_increment_over_protective_plus_auxiliary_on_"
                    "every_action_active_optimizer_step"
                )
                if args.optimizer_restoration_scope in {
                    "corrective_only", "safe_exact_corrective",
                }
                else "composite_action_over_protective_on_every_action_active_optimizer_step"
            )
            if args.materialize_optimizer_update_restoration and active_arm
            else (
                "uniformly sampled optimizer steps only; not an all-step guarantee"
                if active_arm else "not_applicable"
            )
        ),
        "optimizer_counterfactual_attribution_sampling": (
            "every_action_active_optimizer_step"
            if args.materialize_optimizer_update_restoration and active_arm
            else "uniform_epoch_step_positions_including_endpoints"
        ),
        "optimizer_counterfactual_virtual_step_max_relative_error": (
            virtual_step_max_relative_error
        ),
        "optimizer_counterfactual_virtual_step_gate_passed": virtual_step_gate,
        "parameter_group_action_signal": parameter_group_signal,
        "parameter_group_action_signal_gate_passed": parameter_group_signal_gate,
        "parameter_group_action_signal_sampling": (
            "every_action_active_optimizer_step"
            if args.materialize_optimizer_update_restoration and active_arm
            else "uniform_epoch_step_positions_including_endpoints"
        ),
        "parameter_group_action_signal_scope": (
            "every action-active optimizer step"
            if args.materialize_optimizer_update_restoration and active_arm
            else (
                "same sampled optimizer steps; not an all-step guarantee"
                if active_arm else "not_applicable"
            )
        ),
        "optimizer_update_restoration_materialized_every_active_step": bool(
            args.materialize_optimizer_update_restoration and active_arm
        ),
        "action_injector_v1": (
            {
                **(
                    safe_exact_injector.audit_manifest()
                    if safe_exact_injector is not None else {}
                ),
                "used_on_every_action_active_optimizer_step": bool(active_arm),
                "clean_control_noop": bool(not active_arm),
            }
            if args.optimizer_restoration_scope == "safe_exact_corrective"
            else None
        ),
        "optimizer_update_restoration_clean_control_noop": bool(
            args.materialize_optimizer_update_restoration and not active_arm
        ),
        "optimizer_update_restoration_target_attributable_fraction": float(
            args.v4_audit_minimum_attributable_fraction
        ),
        "optimizer_update_restoration_scope": args.optimizer_restoration_scope,
        "optimizer_update_restoration_corrective_only": bool(
            args.optimizer_restoration_scope in {
                "corrective_only", "safe_exact_corrective",
            }
        ),
        "optimizer_update_restoration_safe_exact_corrective": bool(
            args.optimizer_restoration_scope == "safe_exact_corrective"
        ),
        "optimizer_update_restoration_noncorrective_baseline": (
            (
                "safety_projected_protective_plus_projected_robust_harmful"
                if args.optimizer_restoration_scope == "safe_exact_corrective"
                else "protective_plus_projected_robust_harmful"
            )
            if args.optimizer_restoration_scope in {
                "corrective_only", "safe_exact_corrective",
            }
            else "protective_only"
        ),
        "optimizer_update_hard_protective_floor": bool(
            args.optimizer_restoration_scope == "safe_exact_corrective"
        ),
        "optimizer_update_exact_per_group_corrective_fraction": bool(
            args.optimizer_restoration_scope == "safe_exact_corrective"
        ),
        "maximum_safe_exact_update_norm_ratio": float(
            args.maximum_safe_exact_update_norm_ratio
        ),
        "protective_gradient_reaches_every_parameter_on_every_active_step": bool(
            active_arm
            and all_protective_gradient_full_reach
            and np.all(np.asarray(all_protective_gradient_full_reach) == 1.0)
        ),
        "optimizer_update_restoration_maximum_action_gain": float(
            args.v4_audit_maximum_action_gain
        ) if args.optimizer_restoration_scope != "safe_exact_corrective" else None,
        "optimizer_update_restoration_gain_cap_applicable": bool(
            args.optimizer_restoration_scope != "safe_exact_corrective"
        ),
        "optimizer_update_restoration_minimum_risk_component_retention": float(
            args.optimizer_restoration_minimum_risk_component_retention
        ),
        "optimizer_update_restoration_all_group_targets_reached_fraction": (
            restoration_target_reached_fraction
        ),
        "minimum_optimizer_restoration_target_reached_fraction": float(
            args.minimum_optimizer_restoration_target_reached_fraction
        ),
        "optimizer_update_restoration_target_coverage_gate_passed": (
            restoration_target_coverage_gate
        ),
        "optimizer_update_restoration_minimum_observed_group_risk_retention": (
            minimum_observed_protective_retention
        ),
        "optimizer_update_hard_protective_floor_gate_passed": (
            hard_protective_floor_gate
        ),
        "optimizer_update_hard_protective_floor_enforced_fraction": (
            float(np.mean(all_restoration_hard_floor_enforced))
            if all_restoration_hard_floor_enforced else 1.0
        ),
        "optimizer_update_exact_fraction_max_abs_error": (
            exact_fraction_error_max
        ),
        "optimizer_update_exact_fraction_gate_passed": exact_fraction_gate,
        "optimizer_update_safe_exact_max_norm_ratio": safe_exact_norm_ratio_max,
        "optimizer_update_safe_exact_norm_ratio_gate_passed": (
            safe_exact_norm_gate
        ),
        "optimizer_update_restoration_maximum_observed_action_gain": (
            float(max(all_restoration_gain)) if all_restoration_gain else 1.0
        ),
        "restored_adamw_first_moment_reconciled_every_active_step": bool(
            args.reconcile_restored_adamw_first_moment
            and active_arm
            and len(all_restoration_reconciliation_error)
            == len(all_optimizer_virtual_error)
        ),
        "restored_adamw_first_moment_max_reconstruction_relative_error": (
            float(max(all_restoration_reconciliation_error))
            if all_restoration_reconciliation_error else 0.0
        ),
        "restored_adamw_max_materialized_update_relative_error": (
            float(max(all_restoration_materialization_error))
            if all_restoration_materialization_error else 0.0
        ),
        "restored_adamw_max_fp32_parameter_replay_relative_error": (
            float(max(all_restoration_fp32_replay_error))
            if all_restoration_fp32_replay_error else 0.0
        ),
        "restored_adamw_reconstruction_target": (
            "realized_materialized_parameter_displacement"
            if args.reconcile_restored_adamw_first_moment else "not_applicable"
        ),
        "restored_adamw_verification_arithmetic": (
            "stable_decay_plus_adaptive_displacement"
            if args.reconcile_restored_adamw_first_moment else "not_applicable"
        ),
        "restored_adamw_first_moment_max_change_fraction": (
            float(max(all_restoration_first_moment_change))
            if all_restoration_first_moment_change else 0.0
        ),
        "restored_adamw_second_moment_source": (
            "actual_combined_gradient"
            if args.reconcile_restored_adamw_first_moment else "unmodified"
        ),
        "signal_gate_failure": signal_gate_failure,
        "all_signal_gates_passed": bool(not signal_gate_failure),
        "signal_gate_failure_deferred_to_final_decision": bool(
            signal_gate_failure and args.continue_after_signal_gate_failure
        ),
    }
    return (
        history, retention_p10, signal_gate,
        clip_event_fraction, clip_gate,
        optimizer_alignment_p10, optimizer_alignment_gate,
        schedule_report,
    )


def calibrate(
    model: torch.nn.Module,
    store: SpectrumStore,
    corrective: list[BoundaryExample],
    protective: list[ProtectExample],
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    anchor: dict[int, np.ndarray],
    trainable: list[torch.nn.Parameter],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float, float, dict[str, object]]:
    rng = np.random.default_rng(args.seed + 101)
    corr_batches = _batches(corrective, args.batch_queries, rng)
    risk_batches = _batches(protective, args.batch_queries, rng)
    available = min(args.calibration_batches, len(corr_batches), len(risk_batches))
    if not args.development and (
        available < 32 or available * args.batch_queries < 128
    ):
        raise RuntimeError("formal calibration requires >=32 unique query batches and >=128 observations")
    if available < 1:
        raise RuntimeError("gradient calibration has no paired batches")
    corr_norms, risk_norms, transfer_norms, other_norms = [], [], [], []
    for corr, risk in zip(corr_batches[:available], risk_batches[:available]):
        corr_loss, _, components = corrective_batch_loss(
            model, store, corr, action_spectra, control_spectra, anchor, device, args,
        )
        risk_loss, _ = protective_batch_loss(model, store, risk, anchor, device, args)
        corr_grad = list(torch.autograd.grad(
            corr_loss, trainable, retain_graph=True, allow_unused=True,
        ))
        risk_grad = list(torch.autograd.grad(
            risk_loss, trainable, retain_graph=True, allow_unused=True,
        ))
        transfer_grad = list(torch.autograd.grad(
            components["margin_transfer"], trainable, retain_graph=True,
            allow_unused=True,
        ))
        other_grad = list(torch.autograd.grad(
            corr_loss - components["margin_transfer"], trainable, allow_unused=True,
        ))
        corr_norms.append(_gradient_norm(corr_grad))
        risk_norms.append(_gradient_norm(risk_grad))
        transfer_norms.append(_gradient_norm(transfer_grad))
        other_norms.append(_gradient_norm(other_grad))
        model.zero_grad(set_to_none=True)
    corr_median = float(np.median(corr_norms))
    risk_median = float(np.median(risk_norms))
    transfer_median = float(np.median(transfer_norms))
    other_median = float(np.median(other_norms))
    if (
        corr_median <= 0 or transfer_median <= 0
        or not np.isfinite(corr_median + risk_median + other_median)
    ):
        raise RuntimeError("corrective/margin-transfer calibration gradient is zero or non-finite")
    if (
        args.target_margin_transfer_to_other_corrective_ratio <= 0
        or args.margin_transfer_scale_cap < 1
    ):
        raise ValueError("margin-transfer calibration targets are invalid")
    margin_multiplier = (
        min(
            args.margin_transfer_scale_cap,
            args.target_margin_transfer_to_other_corrective_ratio
            * other_median / transfer_median,
        )
        if other_median > 0 else 1.0
    )
    effective_corr_norms = [
        other + margin_multiplier * transfer
        for other, transfer in zip(other_norms, transfer_norms)
    ]
    effective_corr_median = float(np.median(effective_corr_norms))
    scale = (
        1.0 if risk_median <= 0 else
        min(
            args.corrective_scale_cap,
            args.target_corrective_to_risk_ratio * risk_median / effective_corr_median,
        )
    )
    if args.arm == "routed_direct" and scale < args.minimum_corrective_scale:
        raise RuntimeError(
            f"corrective calibration scale {scale:.4f} would recreate excessive signal loss"
        )
    conservative_combined_maximum = float(max(
        scale * corr + risk for corr, risk in zip(effective_corr_norms, risk_norms)
    ))
    if args.target_preclip_gradient_norm <= 0:
        raise ValueError("target preclip gradient norm must be positive")
    # This is one common multiplier after corrective/risk ratio calibration.
    # It preserves every relative direction and is frozen before the first
    # update.  Unlike clipping, uniform scaling does not preferentially erase
    # the large corrective branch.
    global_scale = min(
        1.0, args.target_preclip_gradient_norm / conservative_combined_maximum,
    )
    return float(scale), float(global_scale), float(margin_multiplier), {
        "microbatches": available,
        "query_observations": int(available * args.batch_queries),
        "formal_32x128_gate": bool(available >= 32 and available * args.batch_queries >= 128),
        "corrective_gradient_median": corr_median,
        "risk_gradient_median": risk_median,
        "margin_transfer_gradient_median": transfer_median,
        "other_corrective_gradient_median": other_median,
        "raw_margin_transfer_fraction_of_corrective": transfer_median / corr_median,
        "target_margin_transfer_to_other_corrective_ratio": args.target_margin_transfer_to_other_corrective_ratio,
        "effective_margin_transfer_multiplier": float(margin_multiplier),
        "effective_margin_transfer_to_other_ratio": (
            float(margin_multiplier * transfer_median / other_median)
            if other_median > 0 else None
        ),
        "margin_transfer_scale_hit_cap": bool(
            np.isclose(margin_multiplier, args.margin_transfer_scale_cap)
        ),
        "effective_corrective_gradient_median_upper_bound": effective_corr_median,
        "target_corrective_to_risk_ratio": args.target_corrective_to_risk_ratio,
        "effective_corrective_scale": float(scale),
        "minimum_corrective_scale": args.minimum_corrective_scale,
        "conservative_combined_gradient_maximum": conservative_combined_maximum,
        "target_preclip_gradient_norm": args.target_preclip_gradient_norm,
        "effective_global_gradient_scale": float(global_scale),
        "global_scale_is_common_to_all_branches": True,
        "legacy_lower_clamp_at_one_used": False,
    }


def calibrate_v3(
    model: torch.nn.Module,
    store: SpectrumStore,
    corrective: list[BoundaryExample],
    robust: list[BoundaryExample],
    harmful: list[BoundaryExample],
    protective: list[ProtectExample],
    action_spectra: np.ndarray,
    control_spectra: np.ndarray,
    anchor: dict[int, np.ndarray],
    trainable: list[torch.nn.Parameter],
    device: torch.device,
    args: argparse.Namespace,
    *,
    parameter_group_positions: dict[str, list[int]],
) -> tuple[float, float, dict[str, float], dict[str, object]]:
    """Freeze branch scales from real gradients before the first v3 update."""
    rng = np.random.default_rng(args.seed + 303)
    corr_batches = _prioritize_mechanism_coverage(
        _formula_stratified_batches(corrective, args.batch_queries, rng)
    )
    robust_batches = _prioritize_mechanism_coverage(
        _formula_stratified_batches(robust, args.batch_queries, rng)
    )
    harmful_batches = _prioritize_mechanism_coverage(
        _formula_stratified_batches(harmful, args.batch_queries, rng)
    )
    risk_batches = _formula_stratified_batches(
        protective, args.protective_batch_queries, rng,
    )
    if not all((corr_batches, robust_batches, harmful_batches, risk_batches)):
        raise RuntimeError("v3 calibration requires corrective/robust/harmful/protective panels")
    available = min(
        args.calibration_batches,
        len(corr_batches),
        len(robust_batches),
        len(harmful_batches),
        len(risk_batches),
    )
    if not args.development and (
        available < 32 or available * args.batch_queries < 128
    ):
        raise RuntimeError("formal v3 calibration requires >=32 batches and >=128 observations")
    if available < 1:
        raise RuntimeError("v3 calibration has no paired batches")
    calibration_batches = {
        "corrective": corr_batches[:available],
        "robust": robust_batches[:available],
        "harmful": harmful_batches[:available],
        "risk": risk_batches[:available],
    }
    calibration_mechanism_coverage = {}
    for name, full_panel, selected_batches in (
        ("corrective", corrective, calibration_batches["corrective"]),
        ("robust", robust, calibration_batches["robust"]),
        ("harmful", harmful, calibration_batches["harmful"]),
    ):
        available_mechanisms = sorted({
            action.group.split("::", 1)[0]
            for example in full_panel for action in example.actions
        })
        selected_mechanisms = sorted({
            action.group.split("::", 1)[0]
            for batch in selected_batches for example in batch for action in example.actions
        })
        calibration_mechanism_coverage[name] = {
            "available": available_mechanisms,
            "selected": selected_mechanisms,
            "all_available_present": selected_mechanisms == available_mechanisms,
        }
    if not args.development and not all(
        values["all_available_present"]
        for values in calibration_mechanism_coverage.values()
    ):
        raise RuntimeError(
            "formal calibration prefix omitted an available action mechanism: "
            f"{calibration_mechanism_coverage}"
        )
    formula_diversity = {
        name: {
            "minimum_unique_formulas_per_batch": min(
                len({example.formula for example in batch}) for batch in batches
            ),
            "mean_unique_formulas_per_batch": float(np.mean([
                len({example.formula for example in batch}) for batch in batches
            ])),
        }
        for name, batches in calibration_batches.items()
    }
    if not args.development:
        required_diversity = {
            "corrective": args.batch_queries,
            "robust": args.batch_queries,
            "harmful": args.batch_queries,
            "risk": args.protective_batch_queries,
        }
        failed = {
            name: values["minimum_unique_formulas_per_batch"]
            for name, values in formula_diversity.items()
            if values["minimum_unique_formulas_per_batch"] < required_diversity[name]
        }
        if failed:
            raise RuntimeError(
                f"formal v3 calibration is not formula-diverse: {failed}"
            )

    norms: dict[str, list[float]] = defaultdict(list)
    embedding_diagnostics: dict[str, list[float]] = defaultdict(list)
    semantic_diagnostics: dict[str, list[float]] = defaultdict(list)
    transfer_edge_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for index in range(available):
        corr_batch_scale = _batch_cardinality_scale(
            corr_batches[index], args.batch_queries,
        )
        _, corr_values, corr_components = v3_action_panel_batch_loss(
            model, store, corr_batches[index], action_spectra, control_spectra,
            anchor, device, args, supervision_kind="corrective",
            diagnose_embedding_gradients=index < 4,
        )
        for key, value in corr_values.items():
            if key.startswith("embedding_grad_"):
                embedding_diagnostics[f"corrective.{key}"].append(float(value))
            if key.startswith("transfer_edge_count::"):
                _, scope, metric = key.split("::", 2)
                transfer_edge_totals[scope][metric] += int(round(float(value)))
        for key in (
            "active_transfer_fraction", "capped_transfer_fraction",
            "active_payload_safety_fraction",
        ):
            semantic_diagnostics[key].append(float(corr_values[key]))
        corr_order = (
            ("transfer", corr_components["transfer"]),
            ("payload", corr_components["payload"]),
            ("consistency", corr_components["consistency"]),
        )
        for position, (name, loss) in enumerate(corr_order):
            gradient = list(torch.autograd.grad(
                corr_batch_scale * loss,
                trainable,
                retain_graph=position < len(corr_order) - 1,
                allow_unused=True,
            ))
            norms[name].append(_gradient_norm(gradient))

        robust_batch_scale = _batch_cardinality_scale(
            robust_batches[index], args.batch_queries,
        )
        _, robust_values, robust_components = v3_action_panel_batch_loss(
            model, store, robust_batches[index],
            action_spectra, control_spectra, anchor, device, args,
            supervision_kind="robust",
            diagnose_embedding_gradients=index < 4,
        )
        for key, value in robust_values.items():
            if key.startswith("embedding_grad_"):
                embedding_diagnostics[f"robust.{key}"].append(float(value))
        robust_gradient = list(torch.autograd.grad(
            robust_batch_scale * robust_components["robust"],
            trainable, allow_unused=True,
        ))
        norms["robust"].append(_gradient_norm(robust_gradient))

        harmful_batch_scale = _batch_cardinality_scale(
            harmful_batches[index], args.batch_queries,
        )
        _, harmful_values, harmful_components = v3_action_panel_batch_loss(
            model, store, harmful_batches[index],
            action_spectra, control_spectra, anchor, device, args,
            supervision_kind="harmful",
            diagnose_embedding_gradients=index < 4,
        )
        for key, value in harmful_values.items():
            if key.startswith("embedding_grad_"):
                embedding_diagnostics[f"harmful.{key}"].append(float(value))
        harmful_gradient = list(torch.autograd.grad(
            harmful_batch_scale * harmful_components["harmful"],
            trainable, allow_unused=True,
        ))
        norms["harmful"].append(_gradient_norm(harmful_gradient))

        risk_batch_scale = _batch_cardinality_scale(
            risk_batches[index], args.protective_batch_queries,
        )
        risk_loss, _ = protective_batch_loss(
            model, store, risk_batches[index], anchor, device, args,
        )
        risk_gradient = list(torch.autograd.grad(
            risk_batch_scale * risk_loss, trainable, allow_unused=True,
        ))
        norms["risk"].append(_gradient_norm(risk_gradient))
        model.zero_grad(set_to_none=True)

    median = {name: float(np.median(values)) for name, values in norms.items()}
    diagnostic_median = {
        key: float(np.median(values)) for key, values in embedding_diagnostics.items()
    }
    semantic_median = {
        key: float(np.median(values)) for key, values in semantic_diagnostics.items()
    }
    transfer_edge_breakdown = _finalize_transfer_edge_breakdown(
        transfer_edge_totals
    )
    required_mechanisms = sorted({
        action.group.split("::", 1)[0]
        for example in corrective for action in example.actions
    })
    mechanism_active_fractions = {
        mechanism: float(
            transfer_edge_breakdown.get(f"mechanism={mechanism}", {}).get(
                "active_fraction", 0.0,
            )
        )
        for mechanism in required_mechanisms
    }
    mechanism_transfer_gate = bool(
        required_mechanisms
        and all(
            value >= args.minimum_v3_active_transfer_fraction
            for value in mechanism_active_fractions.values()
        )
    )
    active_transfer_gate = bool(
        semantic_median.get("active_transfer_fraction", 0.0)
        >= args.minimum_v3_active_transfer_fraction
    )
    if any(not np.isfinite(value) or value < 0 for value in median.values()):
        raise RuntimeError(f"v3 calibration contains negative/non-finite branch gradient: {median}")
    calibration_branch_mode = (
        args.corrective_branch_mode
        if args.calibration_corrective_branch_mode == "same_as_training"
        else args.calibration_corrective_branch_mode
    )
    corrective_policy = corrective_branch_policy(calibration_branch_mode)
    essential_branches = {"risk", "robust"}
    if corrective_policy.payload:
        essential_branches.add("payload")
    if corrective_policy.consistency:
        essential_branches.add("consistency")
    if any(median[name] <= 0 for name in essential_branches):
        raise RuntimeError(f"v3 essential branch gradient is zero: {median}")
    allow_semantic_zero = args.arm == "shuffled_action_control"
    if not allow_semantic_zero and (median["transfer"] <= 0 or median["harmful"] <= 0):
        raise RuntimeError(f"v3 routed semantic branch gradient is zero: {median}")
    if not allow_semantic_zero and not active_transfer_gate:
        raise RuntimeError(
            "v3 routed corrective transfer is too sparse at calibration: "
            f"{semantic_median.get('active_transfer_fraction', 0.0):.4f} < "
            f"{args.minimum_v3_active_transfer_fraction:.4f}"
        )
    if not allow_semantic_zero and not mechanism_transfer_gate:
        raise RuntimeError(
            "v3 routed corrective transfer collapsed inside a mechanism: "
            f"{mechanism_active_fractions}; minimum="
            f"{args.minimum_v3_active_transfer_fraction:.4f}"
        )
    corrective_gradient_locality_name = getattr(
        args, "corrective_gradient_locality", "shared"
    )
    query_local_corrective = (
        corrective_gradient_locality_name == "query_action_only"
    )
    required_live_paths = [
        "robust.embedding_grad_robust_action_norm",
        "robust.embedding_grad_robust_reference_norm",
    ]
    if corrective_policy.payload:
        required_live_paths.append(
            "corrective.embedding_grad_payload_action_norm"
        )
    if corrective_policy.consistency:
        required_live_paths.extend([
            "corrective.embedding_grad_consistency_query_norm",
            "corrective.embedding_grad_consistency_action_norm",
        ])
    if not query_local_corrective:
        required_live_paths.append(
            "corrective.embedding_grad_payload_reference_norm"
        )
    if median["transfer"] > 0:
        required_live_paths.append(
            "corrective.embedding_grad_transfer_query_norm"
        )
        if not query_local_corrective:
            required_live_paths.append(
                "corrective.embedding_grad_transfer_reference_norm"
            )
    if median["harmful"] > 0:
        required_live_paths.extend([
            "harmful.embedding_grad_harmful_query_norm",
            "harmful.embedding_grad_harmful_reference_norm",
        ])
    forbidden_paths = (
        "corrective.embedding_grad_payload_query_norm",
        "corrective.embedding_grad_payload_control_norm",
        "corrective.embedding_grad_transfer_action_norm",
        "corrective.embedding_grad_transfer_control_norm",
        "corrective.embedding_grad_consistency_control_norm",
        "corrective.embedding_grad_consistency_reference_norm",
        "robust.embedding_grad_robust_query_norm",
        "robust.embedding_grad_robust_control_norm",
        "harmful.embedding_grad_harmful_action_norm",
    )
    if query_local_corrective:
        forbidden_paths += (
            "corrective.embedding_grad_payload_reference_norm",
            "corrective.embedding_grad_transfer_reference_norm",
        )
    if any(diagnostic_median.get(key, 0.0) <= 0 for key in required_live_paths):
        raise RuntimeError(f"v3 required embedding-gradient path is absent: {diagnostic_median}")
    if any(diagnostic_median.get(key, 0.0) > 1e-12 for key in forbidden_paths):
        raise RuntimeError(f"v3 forbidden embedding-gradient path is active: {diagnostic_median}")
    corrective_role_report = corrective_embedding_role_report(
        diagnostic_median,
        locality=corrective_gradient_locality_name,
    )
    if query_local_corrective and corrective_role_report[
        "query_action_only_locality_gate_passed"
    ] is not True:
        raise RuntimeError(
            "query-local corrective gradient escaped its registered path: "
            f"{corrective_role_report}"
        )
    targets = {
        "payload": (
            args.target_v3_payload_to_transfer_ratio
            if corrective_policy.payload else 0.0
        ),
        "consistency": (
            args.target_v3_consistency_to_transfer_ratio
            if corrective_policy.consistency else 0.0
        ),
        "robust": args.target_v3_robust_to_transfer_ratio,
        "harmful": args.target_v3_harmful_to_risk_ratio,
    }
    if (
        targets["robust"] <= 0
        or targets["harmful"] <= 0
        or any(
            targets[name] <= 0
            for name in ("payload", "consistency")
            if corrective_policy.scale_enabled(name)
        )
        or args.v3_branch_scale_cap < 1
        or not 0 <= args.minimum_v3_active_transfer_fraction <= 1
    ):
        raise ValueError("v3 branch calibration targets are invalid")
    if median["transfer"] > 0:
        transfer_scale = 1.0 if corrective_policy.transfer else 0.0
        requested_branch_scale = {
            "payload": (
                targets["payload"] * median["transfer"] / median["payload"]
                if corrective_policy.payload else 0.0
            ),
            "consistency": (
                targets["consistency"] * median["transfer"] / median["consistency"]
                if corrective_policy.consistency else 0.0
            ),
            "robust": targets["robust"] * median["transfer"] / median["robust"],
        }
    else:
        # Destroying query/action semantics can correctly remove the transfer
        # branch.  Keep the control active and total-norm matched using its
        # payload branch; never invent a nonzero corrective target.
        if not corrective_policy.payload:
            raise RuntimeError(
                "scalar-transfer-only control has zero semantic transfer; use the "
                "full-action-view shuffled arm for a matched negative control"
            )
        transfer_scale = 0.0
        requested_branch_scale = {
            "payload": 1.0,
            "consistency": (
                targets["consistency"] * median["payload"] / median["consistency"]
                if corrective_policy.consistency else 0.0
            ),
            "robust": targets["robust"] * median["payload"] / median["robust"],
        }
    requested_branch_scale["harmful"] = (
        targets["harmful"] * median["risk"] / median["harmful"]
        if median["harmful"] > 0 else 0.0
    )
    active_arm = args.arm in {"routed_direct", "shuffled_action_control"}
    bounded_branch_scale = {
        name: _bounded_calibration_scale(
            requested,
            args.v3_branch_scale_cap,
            f"v3 {name}",
            require_exact=active_arm,
        )
        for name, requested in requested_branch_scale.items()
    }
    branch_scale = {
        "transfer": transfer_scale,
        **{name: value[0] for name, value in bounded_branch_scale.items()},
    }
    branch_scale_hit_cap = {
        name: value[1] for name, value in bounded_branch_scale.items()
    }
    effective_action_upper_bound = (
        branch_scale["transfer"] * median["transfer"]
        + branch_scale["payload"] * median["payload"]
        + branch_scale["consistency"] * median["consistency"]
        + branch_scale["robust"] * median["robust"]
        + branch_scale["harmful"] * median["harmful"]
    )
    corrective_action_norms = []
    corrective_upper_bounds = []
    all_branch_action_norms = []
    all_branch_upper_bounds = []
    combined_risk_cosines = []
    raw_branch_pairwise_cosines: dict[str, list[float]] = defaultdict(list)
    for index in range(available):
        corr_batch_scale = _batch_cardinality_scale(
            corr_batches[index], args.batch_queries,
        )
        _, _, corr_components = v3_action_panel_batch_loss(
            model, store, corr_batches[index], action_spectra, control_spectra,
            anchor, device, args, supervision_kind="corrective",
        )
        corr_loss = (
            corr_batch_scale * (
                branch_scale["transfer"] * corr_components["transfer"]
                + branch_scale["payload"] * corr_components["payload"]
                + branch_scale["consistency"] * corr_components["consistency"]
            )
        )
        corrective_gradient = list(torch.autograd.grad(
            corr_loss, trainable, allow_unused=True,
        ))
        robust_batch_scale = _batch_cardinality_scale(
            robust_batches[index], args.batch_queries,
        )
        _, _, robust_components = v3_action_panel_batch_loss(
            model, store, robust_batches[index],
            action_spectra, control_spectra, anchor, device, args,
            supervision_kind="robust",
        )
        robust_gradient = list(torch.autograd.grad(
            robust_batch_scale * branch_scale["robust"]
            * robust_components["robust"],
            trainable,
            allow_unused=True,
        ))
        harmful_batch_scale = _batch_cardinality_scale(
            harmful_batches[index], args.batch_queries,
        )
        _, _, harmful_components = v3_action_panel_batch_loss(
            model, store, harmful_batches[index],
            action_spectra, control_spectra, anchor, device, args,
            supervision_kind="harmful",
        )
        harmful_gradient = list(torch.autograd.grad(
            harmful_batch_scale * branch_scale["harmful"]
            * harmful_components["harmful"],
            trainable,
            allow_unused=True,
        ))
        auxiliary_gradient = _sum_gradients(robust_gradient, harmful_gradient)
        action_gradient, _, semantic_projection = (
            _preserve_corrective_against_auxiliary(
                corrective_gradient, auxiliary_gradient, parameter_group_positions,
            )
        )
        if not semantic_projection["corrective_direction_preserved"]:
            raise RuntimeError(
                "calibration auxiliary constraints erased corrective direction: "
                f"{semantic_projection}"
            )

        risk_batch_scale = _batch_cardinality_scale(
            risk_batches[index], args.protective_batch_queries,
        )
        risk_loss, _ = protective_batch_loss(
            model, store, risk_batches[index], anchor, device, args,
        )
        risk_gradient = list(torch.autograd.grad(
            risk_batch_scale * risk_loss, trainable, allow_unused=True,
        ))
        raw_branch_gradients = {
            "corrective": corrective_gradient,
            "robust": robust_gradient,
            "harmful": harmful_gradient,
            "protective": risk_gradient,
        }
        for left, right in (
            ("corrective", "robust"),
            ("corrective", "harmful"),
            ("corrective", "protective"),
            ("robust", "harmful"),
            ("robust", "protective"),
            ("harmful", "protective"),
        ):
            cosine = _gradient_cosine(
                raw_branch_gradients[left], raw_branch_gradients[right],
            )
            if cosine is not None:
                raw_branch_pairwise_cosines[f"{left}_vs_{right}"].append(cosine)
        corrective_norm = _gradient_norm(corrective_gradient)
        action_norm = _gradient_norm(action_gradient)
        risk_norm = _gradient_norm(risk_gradient)
        corrective_upper = (
            branch_scale["transfer"] * norms["transfer"][index]
            + branch_scale["payload"] * norms["payload"][index]
            + branch_scale["consistency"] * norms["consistency"][index]
        )
        all_branch_upper = (
            corrective_upper
            + branch_scale["robust"] * norms["robust"][index]
            + branch_scale["harmful"] * norms["harmful"][index]
        )
        cosine = (
            _gradient_dot(action_gradient, risk_gradient) / (action_norm * risk_norm)
            if action_norm > 0 and risk_norm > 0 else 0.0
        )
        corrective_action_norms.append(corrective_norm)
        corrective_upper_bounds.append(corrective_upper)
        all_branch_action_norms.append(action_norm)
        all_branch_upper_bounds.append(all_branch_upper)
        combined_risk_cosines.append(cosine)
        norms["auxiliary_projection_retention"].append(float(
            semantic_projection["auxiliary_projection_retention"]
        ))
        norms["corrective_direction_preservation"].append(float(
            semantic_projection["corrective_direction_preservation_ratio"]
        ))
        for name, values in semantic_projection["parameter_groups"].items():
            norms[f"corrective_direction_preservation::{name}"].append(float(
                values["corrective_direction_preservation_ratio"]
            ))
        model.zero_grad(set_to_none=True)
    effective_corrective_action = float(np.median(corrective_action_norms))
    corrective_combination_retention = np.asarray(
        corrective_action_norms
    ) / np.maximum(
        np.asarray(corrective_upper_bounds), np.finfo(float).eps,
    )
    combination_retention_p10 = float(np.quantile(
        corrective_combination_retention, 0.10,
    ))
    all_branch_combination_retention = np.asarray(
        all_branch_action_norms
    ) / np.maximum(
        np.asarray(all_branch_upper_bounds), np.finfo(float).eps,
    )
    all_branch_combination_retention_p10 = float(np.quantile(
        all_branch_combination_retention, 0.10,
    ))
    if (
        args.minimum_v3_internal_combination_retention <= 0
        or combination_retention_p10 < args.minimum_v3_internal_combination_retention
    ):
        raise RuntimeError(
            "v3 corrective branches cancel before risk projection: "
            f"combination retention p10={combination_retention_p10:.4f}"
        )
    if effective_corrective_action <= 0:
        raise RuntimeError("v3 calibrated corrective action gradient is zero")
    requested_action_scale = (
        args.target_corrective_to_risk_ratio
        * median["risk"] / effective_corrective_action
    )
    action_scale, action_scale_hit_cap = _bounded_calibration_scale(
        requested_action_scale,
        args.corrective_scale_cap,
        "v3 dense corrective action/risk",
        require_exact=active_arm,
    )
    if args.arm != "clean_control" and action_scale < args.minimum_corrective_scale:
        raise RuntimeError(
            f"v3 action calibration scale {action_scale:.4f} would erase the action signal"
        )
    conservative_combined = float(max(
        action_scale * action + risk
        for action, risk in zip(all_branch_action_norms, norms["risk"])
    ))
    if args.target_preclip_gradient_norm <= 0:
        raise ValueError("target preclip gradient norm must be positive")
    global_scale = min(1.0, args.target_preclip_gradient_norm / conservative_combined)
    report: dict[str, object] = {
        "microbatches": available,
        "query_observations": int(available * args.batch_queries),
        "formal_32x128_gate": bool(available >= 32 and available * args.batch_queries >= 128),
        "formula_stratified_without_replacement": True,
        "formula_diversity": formula_diversity,
        "mechanism_prioritized_formula_diverse_prefix": True,
        "calibration_mechanism_coverage": calibration_mechanism_coverage,
        "branch_graphs_backpropagated_sequentially": True,
        "simultaneous_corrective_robust_harmful_risk_graph_retention": False,
        "branch_gradient_median": median,
        "raw_branch_pairwise_cosine_before_arbitration": {
            pair: {
                "observations": len(raw_branch_pairwise_cosines[pair]),
                "median": float(np.median(raw_branch_pairwise_cosines[pair]))
                if raw_branch_pairwise_cosines[pair] else None,
                "p10": float(np.quantile(
                    raw_branch_pairwise_cosines[pair], 0.10,
                )) if raw_branch_pairwise_cosines[pair] else None,
            }
            for pair in (
                "corrective_vs_robust",
                "corrective_vs_harmful",
                "corrective_vs_protective",
                "robust_vs_harmful",
                "robust_vs_protective",
                "harmful_vs_protective",
            )
        },
        "corrective_robust_harmful_protective_gradients_separately_calibrated": True,
        "zero_semantic_branches": [
            name for name in ("transfer", "harmful") if median[name] <= 0
        ],
        "embedding_gradient_path_median_first_four_batches": diagnostic_median,
        "embedding_gradient_path_gate_passed": True,
        "corrective_embedding_role_energy": corrective_role_report,
        "corrective_semantic_edge_fraction_median": semantic_median,
        "corrective_transfer_edge_breakdown": transfer_edge_breakdown,
        "required_corrective_mechanisms": required_mechanisms,
        "corrective_mechanism_active_fraction": mechanism_active_fractions,
        "mechanism_active_transfer_gate_observed_passed": mechanism_transfer_gate,
        "mechanism_active_transfer_gate_applicable": not allow_semantic_zero,
        "mechanism_active_transfer_gate_passed": (
            mechanism_transfer_gate if not allow_semantic_zero else None
        ),
        "minimum_active_transfer_fraction": args.minimum_v3_active_transfer_fraction,
        "active_transfer_fraction_observed_gate_passed": active_transfer_gate,
        "active_transfer_fraction_gate_applicable": not allow_semantic_zero,
        "active_transfer_fraction_gate_passed": (
            active_transfer_gate if not allow_semantic_zero else None
        ),
        "target_branch_ratios": targets,
        "corrective_branch_mode": args.corrective_branch_mode,
        "calibration_corrective_branch_mode": calibration_branch_mode,
        "corrective_branch_policy": corrective_policy.as_dict(),
        "calibration_corrective_branch_policy": corrective_policy.as_dict(),
        "disabled_corrective_branches_have_zero_effective_scale": bool(all(
            branch_scale[name] == 0.0
            for name in ("transfer", "payload", "consistency")
            if not corrective_policy.scale_enabled(name)
        )),
        "requested_branch_scale": requested_branch_scale,
        "effective_branch_scale": branch_scale,
        "branch_scale_hit_cap": branch_scale_hit_cap,
        "effective_action_gradient_upper_bound": float(effective_action_upper_bound),
        "corrective_combined_gradient_median": effective_corrective_action,
        "all_branch_combined_action_gradient_median": float(np.median(
            all_branch_action_norms
        )),
        "internal_action_combination_retention_p10": combination_retention_p10,
        "internal_action_combination_scope": (
            "dense_corrective_" + "_".join(
                name for name in ("transfer", "payload", "consistency")
                if corrective_policy.scale_enabled(name)
            )
        ),
        "all_branch_combination_retention_p10_diagnostic": (
            all_branch_combination_retention_p10
        ),
        "auxiliary_projection_retention_p10": float(np.quantile(
            norms["auxiliary_projection_retention"], 0.10,
        )),
        "corrective_direction_preservation_minimum": float(min(
            norms["corrective_direction_preservation"]
        )),
        "parameter_group_corrective_direction_preservation_minimum": {
            name: float(min(norms[
                f"corrective_direction_preservation::{name}"
            ]))
            for name in parameter_group_positions
        },
        "corrective_direction_preserved_through_inner_semantic_projection": bool(
            min(norms["corrective_direction_preservation"]) >= 1.0 - 1e-6
        ),
        "inner_semantic_projection_order": (
            "project_auxiliary_against_corrective_then_project_action_against_risk"
        ),
        "minimum_internal_action_combination_retention": (
            args.minimum_v3_internal_combination_retention
        ),
        "internal_action_combination_gate_passed": True,
        "combined_action_vs_risk_cosine_median_before_PCGrad": float(
            np.median(combined_risk_cosines)
        ),
        "effective_action_to_risk_scale": float(action_scale),
        "requested_action_to_risk_scale": float(requested_action_scale),
        "action_to_risk_scale_hit_cap": action_scale_hit_cap,
        "effective_dense_corrective_to_risk_gradient_ratio": float(
            action_scale * effective_corrective_action / median["risk"]
        ),
        "dense_corrective_to_risk_target_exactly_reached": bool(
            np.isclose(
                action_scale * effective_corrective_action / median["risk"],
                args.target_corrective_to_risk_ratio,
                rtol=1e-10,
                atol=1e-12,
            )
        ),
        "corrective_scale_uses_dense_corrective_branches_only": True,
        "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator": True,
        "minimum_action_scale": args.minimum_corrective_scale,
        "target_preclip_gradient_norm": args.target_preclip_gradient_norm,
        "effective_global_gradient_scale": float(global_scale),
        "calibration_is_pretraining_and_frozen": True,
        "calibration_is_branch_specific": bool(
            args.calibration_corrective_branch_mode == "same_as_training"
        ),
        "calibration_is_common_full_action_reference": bool(
            calibration_branch_mode == "full_action_view"
            and args.calibration_corrective_branch_mode == "full_action_view"
        ),
    }
    return float(action_scale), float(global_scale), branch_scale, report


def _query_metrics(
    graph: CandidateGraph,
    queries: np.ndarray,
    embeddings: np.ndarray,
    index: dict[int, int],
) -> pd.DataFrame:
    records = []
    for query in queries:
        query = int(query)
        _, rows, ptr, _ = graph.query_block(query)
        q = embeddings[index[int(graph.query_row[query])]]
        pair = embeddings[[index[int(row)] for row in rows]] @ q
        molecule = np.maximum.reduceat(pair, ptr[:-1])
        rank = 1 + int(np.sum(molecule[1:] >= molecule[0]))
        records.append({
            "query_index": query, "rank": rank, "mrr": 1.0 / rank,
            "margin": float(molecule[0] - np.max(molecule[1:])),
        })
    return pd.DataFrame(records)


def _paired_subset(
    graph: CandidateGraph,
    queries: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    index: dict[int, int],
) -> tuple[dict[str, object], pd.DataFrame]:
    queries = np.asarray(sorted(set(map(int, queries))), dtype=np.int64)
    if not len(queries):
        return {"queries": 0}, pd.DataFrame()
    left = _query_metrics(graph, queries, initial, index)
    right = _query_metrics(graph, queries, final, index)
    paired = left.merge(right, on="query_index", suffixes=("_initial", "_final"), validate="one_to_one")
    paired.insert(
        1,
        "query_formula",
        [str(graph.query_formula[int(query)]) for query in paired.query_index],
    )
    corrected = paired.rank_initial.ne(1) & paired.rank_final.eq(1)
    introduced = paired.rank_initial.eq(1) & paired.rank_final.ne(1)
    return {
        "queries": int(len(paired)),
        "initial_recall1": float(paired.rank_initial.eq(1).mean()),
        "final_recall1": float(paired.rank_final.eq(1).mean()),
        "delta_recall1_pp": float(100 * (paired.rank_final.eq(1).mean() - paired.rank_initial.eq(1).mean())),
        "initial_mrr": float(paired.mrr_initial.mean()),
        "final_mrr": float(paired.mrr_final.mean()),
        "delta_mrr_pp": float(100 * (paired.mrr_final.mean() - paired.mrr_initial.mean())),
        "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
        "risk_net_lambda2": int(corrected.sum() - 2 * introduced.sum()),
        "mean_margin_change": float((paired.margin_final - paired.margin_initial).mean()),
    }, paired


def main() -> None:
    args = arguments()
    _validate_registered_formal_v3_configuration(args)
    started = time.time()
    seed_everything(args.seed)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if (
        args.outer_fold not in range(5)
        or args.batch_queries < 1
        or args.protective_batch_queries < 1
        or args.epochs < 1
        or args.reference_refresh_every_epochs < 0
    ):
        raise ValueError("fold/batch/epoch configuration is invalid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.development and any((
        args.maximum_corrective_queries, args.maximum_risk_queries,
        args.maximum_robust_queries, args.maximum_clean_queries,
        args.outer_held_eval_queries,
    )):
        raise RuntimeError("formal training cannot use development query limits")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    ledger_path = args.routed_ledger_dir / "training_actions.csv.gz"
    spectra_path = args.routed_ledger_dir / "action_spectra.npz"
    ledger_report_path = args.routed_ledger_dir / "report.json"
    source_manifest_path = args.source_manifest_dir / "manifest.npz"
    initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
    required = (
        graph_path, graph_report_path, ledger_path, spectra_path, ledger_report_path,
        source_manifest_path, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.initial_student_checkpoint,
        initial_decision_path,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    ledger_report = json.loads(ledger_report_path.read_text(encoding="utf-8"))
    if (
        graph_report.get("formal_training_authorized") is not True
        or ledger_report.get("status") != "noise_corrected_routed_action_ledger_complete"
        or int(ledger_report.get("outer_formula_fold", -1)) != args.outer_fold
        or (not args.development and ledger_report.get("formal_training_authorized") is not True)
        or ledger_report.get("contracts", {}).get("teacher_embedding_target_used") is not False
        or ledger_report.get("contracts", {}).get("P3_consumed") is not False
        or (
            args.corrective_objective_mode == "v3_direct"
            and ledger_report.get("contracts", {}).get(
                "exact_action_control_candidate_switch_rows_recorded"
            ) is not True
        )
        or (
            args.corrective_objective_mode == "v3_direct"
            and ledger_report.get("contracts", {}).get(
                "control_semantics_explicit_and_source_validated"
            ) is not True
        )
        or (
            args.corrective_objective_mode == "v3_direct"
            and ledger_report.get("contracts", {}).get(
                "all_route_formal_configurations_verified"
            ) is not True
        )
        or (
            args.corrective_objective_mode == "v3_direct"
            and ledger_report.get("contracts", {}).get(
                "source_then_family_exposure_balanced_before_recipe_score"
            ) is not True
        )
        or (
            args.corrective_objective_mode == "v3_direct"
            and ledger_report.get("contracts", {}).get(
                "all_route_formula_fold_seeds_match"
            ) is not True
        )
    ):
        raise RuntimeError("corrected graph/routed ledger contract failed")
    observed_input_sha256 = {
        key: sha256_file(path)
        for key, path in (
        ("spectrum_data_sha256", args.data),
        ("architecture_checkpoint_sha256", args.architecture_checkpoint),
        ("candidate_graph_sha256", graph_path),
        ("graph_report_sha256", graph_report_path),
        ("routed_ledger_report_sha256", ledger_report_path),
        ("training_actions_sha256", ledger_path),
        ("action_spectra_sha256", spectra_path),
        ("initial_student_checkpoint_sha256", args.initial_student_checkpoint),
        ("initial_student_decision_sha256", initial_decision_path),
        ("official_checkpoint_sha256", args.official_checkpoint),
        )
    }
    for key in (
        "candidate_graph_sha256", "graph_report_sha256",
        "training_actions_sha256", "action_spectra_sha256",
    ):
        if ledger_report.get("provenance", {}).get(key) != observed_input_sha256[key]:
            raise RuntimeError(f"routed ledger provenance drifted: {key}")
    if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development:
        mismatched_inputs = {
            key: {"observed": observed_input_sha256.get(key), "expected": expected}
            for key, expected in REGISTERED_BEST_ACTION_V6_INPUT_SHA256.items()
            if observed_input_sha256.get(key) != expected
        }
        if mismatched_inputs:
            raise RuntimeError(
                "best-action v6 immutable input drifted: "
                + json.dumps(mismatched_inputs, sort_keys=True)
            )
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if graph_report.get("provenance", {}).get(
        "source_manifest_sha256"
    ) != source_manifest_sha256:
        raise RuntimeError("evaluation metadata manifest differs from graph provenance")
    official_checkpoint_sha256 = observed_input_sha256["official_checkpoint_sha256"]
    if graph_report.get("provenance", {}).get(
        "official_checkpoint_sha256"
    ) != official_checkpoint_sha256:
        raise RuntimeError("official checkpoint differs from candidate graph provenance")

    graph = CandidateGraph(graph_path)
    with np.load(source_manifest_path, allow_pickle=False) as source:
        required_source = {
            "query_row", "query_ik14", "query_formula", "query_adduct",
            "query_ptr", "molecule_ptr", "molecule_label", "molecule_ik14",
            "molecule_formula", "pair_candidate_row",
        }
        if missing_source := required_source - set(source.files):
            raise RuntimeError(
                f"evaluation metadata manifest misses {sorted(missing_source)}"
            )
        for name in required_source - {"query_adduct"}:
            if not np.array_equal(np.asarray(source[name]), getattr(graph, name)):
                raise RuntimeError(
                    f"evaluation metadata manifest is not graph-aligned: {name}"
                )
        query_adduct = np.asarray(source["query_adduct"], dtype=str)
    if query_adduct.shape != (graph.n_queries,):
        raise RuntimeError("evaluation metadata query adduct is not graph-aligned")
    actions = pd.read_csv(ledger_path, low_memory=False)
    required_columns = {
        "action_id", "query_index", "query_row", "query_ik14", "query_formula",
        "formula_fold", "source", "family", "supervision_kind", "action_tensor_index",
        "control_semantic", "recipe_id",
    }
    if missing := required_columns - set(actions.columns):
        raise RuntimeError(f"training action ledger misses {sorted(missing)}")
    candidate_switch_columns = {
        "action_positive_row", "action_hard_negative_molecule_index",
        "action_hard_negative_row", "control_positive_row",
        "control_hard_negative_molecule_index", "control_hard_negative_row",
    }
    if (
        args.corrective_objective_mode == "v3_direct"
        and (missing_switch := candidate_switch_columns - set(actions.columns))
    ):
        raise RuntimeError(
            "v3 direct training requires exact action candidate-switch rows: "
            f"{sorted(missing_switch)}"
        )
    allowed_supervision = {"corrective", "harmful", "robust"}
    observed_supervision = set(actions.supervision_kind.astype(str))
    if not observed_supervision <= allowed_supervision:
        raise RuntimeError(f"unknown routed supervision kinds: {observed_supervision}")
    if args.corrective_objective_mode == "v3_direct" and (
        not {"corrective", "harmful", "robust"} <= observed_supervision
        or ledger_report.get("contracts", {}).get("three_semantics_are_separate") is not True
    ):
        raise RuntimeError("v3 direct training requires an explicit three-semantics ledger")
    if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development:
        expected_sources = {
            "N_mature", "P_guided_original", "E10B", "E11", "E12B",
            "A4_exact", "V4_gradient_path",
        }
        observed_counts = actions.supervision_kind.astype(str).value_counts().to_dict()
        expected_counts = {
            "corrective": REGISTERED_BEST_ACTION_V6_COUNTS["corrective_rows"],
            "harmful": REGISTERED_BEST_ACTION_V6_COUNTS["harmful_rows"],
            "robust": REGISTERED_BEST_ACTION_V6_COUNTS["robust_rows"],
        }
        if (
            len(actions) != REGISTERED_BEST_ACTION_V6_COUNTS["ledger_rows"]
            or observed_counts != expected_counts
            or set(actions.source.astype(str)) != expected_sources
        ):
            raise RuntimeError(
                "best-action v6 ledger counts/source set drifted: "
                + json.dumps({
                    "rows": len(actions),
                    "supervision": observed_counts,
                    "sources": sorted(set(actions.source.astype(str))),
                }, sort_keys=True)
            )
    with np.load(spectra_path, allow_pickle=False) as body:
        if set(body.files) != {"action_ids", "action_spectra", "control_spectra"}:
            raise RuntimeError("routed action spectra schema failed")
        ids = np.asarray(body["action_ids"], dtype=str)
        action_spectra = np.asarray(body["action_spectra"], dtype=np.float32)
        control_spectra = np.asarray(body["control_spectra"], dtype=np.float32)
    if (
        action_spectra.shape != control_spectra.shape
        or action_spectra.shape[1:] != (args.n_highest_peaks + 1, 2)
        or not np.array_equal(actions.action_id.astype(str).to_numpy(), ids)
        or not np.array_equal(actions.action_tensor_index.to_numpy(np.int64), np.arange(len(actions)))
    ):
        raise RuntimeError("training action tensor alignment failed")
    if actions.formula_fold.eq(args.outer_fold).any():
        raise RuntimeError("outer-held action leaked into training ledger")
    query = actions.query_index.to_numpy(np.int64)
    if (
        not np.array_equal(actions.query_row.to_numpy(np.int64), graph.query_row[query])
        or not np.array_equal(actions.query_ik14.astype(str).to_numpy(), graph.query_ik14[query])
        or not np.array_equal(actions.query_formula.astype(str).to_numpy(), graph.query_formula[query])
    ):
        raise RuntimeError("training action graph alignment failed")

    inner_holdout_key = (
        actions.query_formula.astype(str)
        if args.inner_holdout_unit == "formula"
        else actions.query_ik14.astype(str)
    )
    identity_fold = inner_holdout_key.map(
        lambda value: stable_fold(value, args.inner_identity_folds, args.seed + 17)
    ).to_numpy(np.int8)
    if args.inner_holdout_fold >= 0:
        if args.inner_holdout_fold >= args.inner_identity_folds:
            raise ValueError("inner holdout fold is out of range")
        train_actions = actions.loc[identity_fold != args.inner_holdout_fold].copy()
        held_actions = actions.loc[identity_fold == args.inner_holdout_fold].copy()
    else:
        train_actions = actions.copy()
        held_actions = actions.iloc[:0].copy()
    if args.inner_holdout_fold >= 0 and args.inner_holdout_unit == "formula":
        train_formulas = set(train_actions.query_formula.astype(str))
        held_formulas = set(held_actions.query_formula.astype(str))
        overlap = train_formulas & held_formulas
        if overlap:
            raise RuntimeError(
                "formula-disjoint inner holdout leaked formulas: "
                f"{sorted(overlap)[:5]}"
            )
    corrective, corrective_admission_report = _select_corrective_admission(
        train_actions,
        mode=args.corrective_admission,
        margin_floor=args.corrective_margin_floor,
        action_bank_contract=args.action_bank_contract,
    )
    held_corrective, held_corrective_admission_report = (
        _select_corrective_admission(
            held_actions,
            mode=args.corrective_admission,
            margin_floor=args.corrective_margin_floor,
            action_bank_contract=args.action_bank_contract,
        )
        if len(held_actions)
        else (
            held_actions.copy(),
            {
                "mode": args.corrective_admission,
                "input_rows": 0,
                "selected_rows": 0,
            },
        )
    )
    harmful = train_actions.loc[train_actions.supervision_kind.eq("harmful")].copy()
    robust = train_actions.loc[train_actions.supervision_kind.eq("robust")].copy()
    corrective_queries = _limit_queries(
        corrective.query_index.unique(), args.maximum_corrective_queries, args.seed + 1,
    )
    risk_queries = _limit_risk_queries(
        harmful, args.maximum_risk_queries, args.seed + 2,
    )
    robust_queries = _limit_queries(
        robust.query_index.unique(), args.maximum_robust_queries, args.seed + 5,
    )
    corrective = corrective.loc[corrective.query_index.isin(corrective_queries)].copy()
    harmful = harmful.loc[harmful.query_index.isin(risk_queries)].copy()
    robust = robust.loc[robust.query_index.isin(robust_queries)].copy()
    (
        admitted_actions,
        corrective,
        harmful,
        robust,
        action_spectra,
        control_spectra,
        admitted_action_report,
    ) = _materialize_admitted_action_bank(
        corrective, harmful, robust, action_spectra, control_spectra,
    )
    if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development:
        expected_admitted = {
            "corrective": REGISTERED_BEST_ACTION_V6_COUNTS["strict_corrective_rows"],
            "harmful": REGISTERED_BEST_ACTION_V6_COUNTS["harmful_rows"],
            "robust": REGISTERED_BEST_ACTION_V6_COUNTS["robust_rows"],
        }
        if (
            corrective_admission_report["strict_top1_rows_before_margin_floor"]
            != REGISTERED_BEST_ACTION_V6_COUNTS[
                "strict_corrective_rows_before_floor"
            ]
            or corrective.query_index.nunique()
            != REGISTERED_BEST_ACTION_V6_COUNTS["strict_corrective_queries"]
            or robust.query_index.nunique()
            != REGISTERED_BEST_ACTION_V6_COUNTS["robust_queries"]
            or harmful.query_index.nunique()
            != REGISTERED_BEST_ACTION_V6_COUNTS["harmful_queries"]
            or set(corrective.source.astype(str)) != expected_sources
            or int(corrective.groupby("query_index").size().max()) > 16
            or admitted_action_report["admitted_rows"]
            != REGISTERED_BEST_ACTION_V6_COUNTS["admitted_rows"]
            or admitted_action_report["admitted_rows_by_supervision"]
            != expected_admitted
            or not admitted_action_report["admitted_semantic_boundary_sha256"]
        ):
            raise RuntimeError(
                "best-action v6 admitted-panel counts drifted: "
                + json.dumps({
                    "admission": corrective_admission_report,
                    "admitted": admitted_action_report,
                }, sort_keys=True)
            )
    action_control_report: dict[str, object] = {
        "strategy": "true_routed_action_bank",
        "rows": int(len(admitted_actions)),
        "donor_pool_rows": int(len(admitted_actions)),
        "donors_restricted_to_optimizer_admitted_panel": True,
    }
    training_action_spectra = action_spectra
    if args.arm == "shuffled_action_control":
        if args.corrective_objective_mode != "v3_direct":
            raise RuntimeError("shuffled action control is defined only for v3 direct training")
        training_action_spectra, action_control_report = source_family_shuffled_action_bank(
            admitted_actions,
            action_spectra,
            control_spectra,
            seed=args.seed + 1701,
        )
        action_control_report.update({
            "donor_pool_rows": int(len(admitted_actions)),
            "donors_restricted_to_optimizer_admitted_panel": True,
            "rejected_corrective_rows_can_be_donors": False,
            "inner_holdout_rows_excluded_before_donor_pool": bool(
                args.inner_holdout_fold >= 0
            ),
        })
        if (
            not args.development
            and (
                float(action_control_report["cross_query_fraction"]) < 0.95
                or action_control_report.get(
                    "all_nonfallback_donors_inside_input_panel"
                ) is not True
                or not action_control_report.get("donor_tensor_index_sha256")
            )
        ):
            raise RuntimeError(
                "formal source/family/exact-recipe shuffled control retains less "
                f"than 95% cross-query rows: {action_control_report}"
            )
    calibration_action_spectra, calibration_action_report = _calibration_action_bank(
        action_spectra,
        training_action_spectra,
        arm_invariant_targeted=args.arm_invariant_targeted_calibration,
    )
    targeted_action_bank_sha256 = _ndarray_sha256(action_spectra)
    control_action_bank_sha256 = _ndarray_sha256(control_spectra)
    training_action_bank_sha256 = _ndarray_sha256(training_action_spectra)
    calibration_action_bank_sha256 = _ndarray_sha256(calibration_action_spectra)
    calibration_action_report.update({
        "targeted_action_bank_sha256": targeted_action_bank_sha256,
        "control_action_bank_sha256": control_action_bank_sha256,
        "training_action_bank_sha256": training_action_bank_sha256,
        "calibration_action_bank_sha256": calibration_action_bank_sha256,
        "admitted_action_ids_sha256": admitted_action_report[
            "admitted_action_ids_sha256"
        ],
        "calibration_occurs_after_admission_and_shuffle": True,
    })
    if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS:
        safe_exact_common_calibration = bool(
            args.direct_contract in SAFE_EXACT_DIRECT_CONTRACTS
        )
        if (
            args.arm_invariant_targeted_calibration
            != safe_exact_common_calibration
        ):
            raise RuntimeError(
                "best-action calibration policy drifted from its registered contract"
            )
        expected_calibration_sha = (
            targeted_action_bank_sha256
            if safe_exact_common_calibration else training_action_bank_sha256
        )
        if calibration_action_bank_sha256 != expected_calibration_sha:
            raise RuntimeError("best-action calibration bank is not the registered bank")
        training_differs = training_action_bank_sha256 != targeted_action_bank_sha256
        if training_differs != (args.arm == "shuffled_action_control"):
            raise RuntimeError(
                "best-action v6 changed more or less than the shuffled training bank"
            )
    outer_folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula
    ], dtype=np.int8)
    outer_train = np.flatnonzero(outer_folds != args.outer_fold)
    inner_held_formulas = set(held_actions.query_formula.astype(str))
    clean_training_pool = outer_train
    if args.inner_holdout_fold >= 0 and args.inner_holdout_unit == "formula":
        clean_training_pool = np.asarray([
            int(query)
            for query in outer_train
            if str(graph.query_formula[int(query)]) not in inner_held_formulas
        ], dtype=np.int64)
        if any(
            str(graph.query_formula[int(query)]) in inner_held_formulas
            for query in clean_training_pool
        ):
            raise RuntimeError("inner-held formula leaked into clean training pool")
    registered_outer_held = int(np.sum(outer_folds == args.outer_fold))
    if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development and (
        graph.n_queries != REGISTERED_BEST_ACTION_V6_COUNTS["graph_queries"]
        or len(outer_train) != REGISTERED_BEST_ACTION_V6_COUNTS["outer_train_queries"]
        or registered_outer_held
        != REGISTERED_BEST_ACTION_V6_COUNTS["outer_held_queries"]
    ):
        raise RuntimeError(
            "best-action v6 candidate-graph fold geometry drifted: "
            + json.dumps({
                "graph_queries": graph.n_queries,
                "outer_train_queries": len(outer_train),
                "outer_held_queries": registered_outer_held,
            }, sort_keys=True)
        )
    required_clean = np.asarray(sorted(
        set(map(int, corrective_queries))
        | set(map(int, risk_queries))
        | set(map(int, robust_queries))
    ), dtype=np.int64)
    if args.maximum_clean_queries:
        if len(required_clean) > args.maximum_clean_queries:
            raise RuntimeError("clean query limit is smaller than routed training queries")
        pool = np.setdiff1d(
            clean_training_pool, required_clean, assume_unique=False,
        )
        add = min(args.maximum_clean_queries - len(required_clean), len(pool))
        chosen = (
            np.sort(np.random.default_rng(args.seed + 3).choice(pool, add, replace=False))
            if add else np.empty(0, dtype=np.int64)
        )
        clean_queries = np.sort(np.concatenate([required_clean, chosen]))
    else:
        clean_queries = clean_training_pool
    protect_queries = np.asarray(sorted(set(map(int, clean_queries)) | set(map(int, risk_queries))), dtype=np.int64)
    held_corrective_queries = held_corrective.query_index.unique().astype(
        np.int64
    )
    held_robust_queries = held_actions.loc[
        held_actions.supervision_kind.eq("robust"), "query_index"
    ].unique().astype(np.int64)
    held_outer = np.flatnonzero(outer_folds == args.outer_fold)
    held_outer = _limit_queries(held_outer, args.outer_held_eval_queries, args.seed + 4)
    evaluation_queries = np.asarray(sorted(
        set(map(int, corrective_queries)) | set(map(int, held_corrective_queries))
        | set(map(int, risk_queries)) | set(map(int, robust_queries))
        | set(map(int, held_robust_queries)) | set(map(int, held_outer))
    ), dtype=np.int64)
    if not len(corrective_queries) or not len(protect_queries):
        raise RuntimeError("training requires corrective and protective queries")
    if args.corrective_objective_mode == "v3_direct" and (
        not len(risk_queries) or not len(robust_queries)
    ):
        raise RuntimeError("v3 direct training requires non-empty harmful and robust panels")

    pre_model_schedule_geometry: dict[str, int | float | bool] = {}
    if args.corrective_objective_mode == "v3_direct":
        pre_model_schedule_geometry = v3_schedule_geometry(
            corrective_queries=int(len(corrective_queries)),
            robust_queries=int(len(robust_queries)),
            harmful_queries=int(len(risk_queries)),
            protective_queries=int(len(protect_queries)),
            corrective_batch_size=int(args.batch_queries),
            protective_batch_size=int(args.protective_batch_queries),
            maximum_auxiliary_microbatches_per_step=int(
                args.v3_maximum_auxiliary_microbatches_per_step
            ),
            maximum_protective_microbatches_per_step=int(
                args.v3_maximum_protective_microbatches_per_step
            ),
            maximum_corrective_recycle_factor=float(
                args.v3_maximum_corrective_recycle_factor
            ),
        ).as_dict()
        if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development:
            _validate_registered_best_action_v6_schedule_geometry(
                pre_model_schedule_geometry,
                context="pre-model",
            )
        print(json.dumps({
            "status": "noise_corrected_v3_pre_model_schedule_geometry_pass",
            "arm": args.arm,
            **pre_model_schedule_geometry,
            "model_loaded": False,
            "gpu_forward_started": False,
        }), flush=True)

    if args.development:
        needed_queries = np.asarray(sorted(set(map(int, protect_queries)) | set(map(int, evaluation_queries))), dtype=np.int64)
        needed: set[int] = set(map(int, graph.query_row[needed_queries]))
        for value in needed_queries:
            _, rows, _, _ = graph.query_block(int(value))
            needed.update(map(int, rows))
        reachable = np.asarray(sorted(needed), dtype=np.int64)
    else:
        reachable = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    model, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks,
    )
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    initial_decision = json.loads(initial_decision_path.read_text(encoding="utf-8"))
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or package.get("inference_clean_only") is not True
        or package.get("P2b_used") is not False
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or initial_decision.get("formal") is not True
        or ledger_report.get("model_provenance", {}).get("initial_student_checkpoint_sha256")
        != sha256_file(args.initial_student_checkpoint)
    ):
        raise RuntimeError("routed ledger and mature E8 initialization differ")
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    initial = encode_rows(
        model, store, reachable, device, args.eval_batch_size, args.amp,
        "corrected-routed-initial",
    )
    initial_graph_scores = (
        score_embeddings(graph, reachable, initial) if not args.development else None
    )
    initial_pair_scores = (
        initial_graph_scores.pair if initial_graph_scores is not None else None
    )
    index = {int(row): position for position, row in enumerate(reachable)}
    anchor = {int(row): initial[position] for position, row in enumerate(reachable)}
    boundary_examples, protect_examples = build_examples(
        graph, corrective, protect_queries, initial, index, args, initial_pair_scores,
        global_mechanism_balance=(args.corrective_objective_mode == "v3_direct"),
    )
    robust_examples: list[BoundaryExample] = []
    harmful_examples: list[BoundaryExample] = []
    action_boundary_report: dict[str, object] = {"enabled": False}
    mechanism_balance_report: dict[str, object] = {}
    if args.corrective_objective_mode == "v3_direct":
        robust_examples, _ = build_examples(
            graph, robust, np.empty(0, dtype=np.int64), initial, index, args,
            initial_pair_scores,
            global_mechanism_balance=False,
        )
        harmful_examples, _ = build_examples(
            graph, harmful, np.empty(0, dtype=np.int64), initial, index, args,
            initial_pair_scores,
            global_mechanism_balance=False,
        )
        boundary_examples, corrective_boundary_report = augment_action_conditioned_references(
            graph, boundary_examples, corrective, initial, index,
            maximum_positive_references=args.action_conditioned_positive_references,
            maximum_negative_molecules=args.action_conditioned_negative_molecules,
            references_per_negative=args.references_per_negative,
        )
        robust_examples, robust_boundary_report = augment_action_conditioned_references(
            graph, robust_examples, robust, initial, index,
            maximum_positive_references=args.action_conditioned_positive_references,
            maximum_negative_molecules=args.action_conditioned_negative_molecules,
            references_per_negative=args.references_per_negative,
        )
        harmful_examples, harmful_boundary_report = augment_action_conditioned_references(
            graph, harmful_examples, harmful, initial, index,
            maximum_positive_references=args.action_conditioned_positive_references,
            maximum_negative_molecules=args.action_conditioned_negative_molecules,
            references_per_negative=args.references_per_negative,
        )
        action_boundary_report = {
            "enabled": True,
            "corrective": corrective_boundary_report,
            "robust": robust_boundary_report,
            "harmful": harmful_boundary_report,
            "initial_E8_clean_and_action_control_candidate_boundary_union": True,
        }
        mechanism_balance_report = {
            "corrective": _panel_mechanism_balance_report(boundary_examples),
            "robust": _panel_mechanism_balance_report(robust_examples),
            "harmful": _panel_mechanism_balance_report(harmful_examples),
            "strategy": {
                "corrective": "global_identity_weighted_inverse_mechanism_incidence",
                "robust": "query_local_mechanism_mean",
                "harmful": "query_local_mechanism_mean",
            },
            "control_semantic_action_rows": dict(Counter(
                action.control_semantic
                for panel in (boundary_examples, robust_examples, harmful_examples)
                for example in panel for action in example.actions
            )),
            "query_or_action_dropped_for_balance": False,
        }
        if not bool(mechanism_balance_report["corrective"]["balanced"]):
            raise RuntimeError("global corrective N/P/A4 mechanism mass is not balanced")
        mechanism_balance_report["maximum_allowed_coefficient"] = (
            args.v3_branch_scale_cap
        )
        mechanism_balance_report["coefficient_cap_passed"] = (
            float(mechanism_balance_report["corrective"][
                "maximum_mechanism_coefficient"
            ]) <= args.v3_branch_scale_cap
        )
        if not args.development and not mechanism_balance_report[
            "coefficient_cap_passed"
        ]:
            raise RuntimeError(
                "global mechanism equalization requires an unsafe coefficient: "
                f"{mechanism_balance_report}"
            )
    corrective_materialization_report = _audit_corrective_example_materialization(
        corrective, boundary_examples,
    )
    forward_memory_contract = _forward_memory_contract(
        boundary_examples,
        robust_examples,
        harmful_examples,
        protect_examples,
        args,
    )
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            "params": head, "lr": args.head_lr,
            "weight_decay": args.weight_decay, "group_name": "head",
        },
        {
            "params": backbone, "lr": args.backbone_lr,
            "weight_decay": 0.0, "group_name": "backbone",
        },
    ])
    optimizer_parameter_group_positions = _optimizer_parameter_group_positions(
        optimizer, trainable,
    )
    v3_branch_scale: dict[str, float] = {}
    if args.corrective_objective_mode == "v3_direct":
        (
            corrective_scale,
            global_gradient_scale,
            v3_branch_scale,
            calibration,
        ) = calibrate_v3(
            model, store, boundary_examples, robust_examples, harmful_examples,
            protect_examples, calibration_action_spectra, control_spectra,
            anchor, trainable, device, args,
            parameter_group_positions=optimizer_parameter_group_positions,
        )
        calibrated_v3_branch_scale = dict(v3_branch_scale)
        training_corrective_policy = corrective_branch_policy(
            args.corrective_branch_mode
        )
        v3_branch_scale = {
            name: (
                float(value)
                if name in {"robust", "harmful"}
                or training_corrective_policy.scale_enabled(name)
                else 0.0
            )
            for name, value in calibrated_v3_branch_scale.items()
        }
        calibration["calibrated_effective_branch_scale"] = (
            calibrated_v3_branch_scale
        )
        calibration["effective_branch_scale"] = dict(v3_branch_scale)
        calibration["training_corrective_branch_policy"] = (
            training_corrective_policy.as_dict()
        )
        calibration[
            "disabled_training_corrective_branches_have_zero_effective_scale"
        ] = bool(all(
            v3_branch_scale[name] == 0.0
            for name in ("transfer", "payload", "consistency")
            if not training_corrective_policy.scale_enabled(name)
        ))
        margin_transfer_multiplier = 1.0
    else:
        (
            corrective_scale, global_gradient_scale,
            margin_transfer_multiplier, calibration,
        ) = calibrate(
            model, store, boundary_examples, protect_examples, action_spectra,
            control_spectra, anchor, trainable, device, args,
        )
    v3_mode = args.corrective_objective_mode == "v3_direct"
    schedule_report: dict[str, object] = {
        "mode": "legacy_front_loaded",
        "protect_only_optimizer_tail": True,
    }
    if v3_mode:
        refresh_counter = 0

        def refresh_v3_examples() -> tuple[
            list[BoundaryExample], list[BoundaryExample],
            list[BoundaryExample], list[ProtectExample],
        ]:
            nonlocal refresh_counter
            refresh_counter += 1
            refreshed = encode_rows(
                model, store, reachable, device, args.eval_batch_size, args.amp,
                f"v3-hard-reference-refresh-{refresh_counter}",
            )
            refreshed_pair_scores = (
                score_embeddings(graph, reachable, refreshed).pair
                if not args.development else None
            )
            refreshed_boundary, refreshed_protect = build_examples(
                graph, corrective, protect_queries, refreshed, index, args,
                refreshed_pair_scores,
                global_mechanism_balance=True,
            )
            refreshed_robust, _ = build_examples(
                graph, robust, np.empty(0, dtype=np.int64), refreshed, index, args,
                refreshed_pair_scores,
                global_mechanism_balance=False,
            )
            refreshed_harmful, _ = build_examples(
                graph, harmful, np.empty(0, dtype=np.int64), refreshed, index, args,
                refreshed_pair_scores,
                global_mechanism_balance=False,
            )
            return (
                _union_initial_negative_molecules(
                    graph, boundary_examples, refreshed_boundary,
                ),
                _union_initial_negative_molecules(
                    graph, robust_examples, refreshed_robust,
                ),
                _union_initial_negative_molecules(
                    graph, harmful_examples, refreshed_harmful,
                ),
                _union_initial_negative_molecules(
                    graph, protect_examples, refreshed_protect,
                ),
            )

        (
            history,
            retention_p10,
            signal_gate,
            clip_event_fraction,
            clip_gate,
            optimizer_alignment_p10,
            optimizer_alignment_gate,
            schedule_report,
        ) = train_v3_epochs(
            model, optimizer, store,
            boundary_examples, robust_examples, harmful_examples, protect_examples,
            training_action_spectra, control_spectra, anchor, trainable, device, args,
            action_scale=corrective_scale,
            global_gradient_scale=global_gradient_scale,
            branch_scale=v3_branch_scale,
            reference_refresh=refresh_v3_examples,
            expected_schedule_geometry=pre_model_schedule_geometry,
        )
        if schedule_report.get(
            "schedule_geometry_matches_pre_model_preflight"
        ) is not True:
            raise RuntimeError("v3 training schedule escaped the pre-model geometry gate")
        if args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS and not args.development:
            _validate_registered_best_action_v6_schedule_geometry(
                schedule_report.get("schedule_geometry", {}),
                context="final-report",
            )
    else:
        history = []
    rng = np.random.default_rng(args.seed + 500)
    all_retention: list[float] = []
    all_clip_retention: list[float] = []
    for epoch in ([] if v3_mode else range(args.epochs)):
        corr_batches = _batches(boundary_examples, args.batch_queries, rng)
        risk_batches = _batches(protect_examples, args.batch_queries, rng)
        steps = max(len(corr_batches), len(risk_batches))
        logs: dict[str, list[float]] = defaultdict(list)
        seen_actions: set[str] = set()
        seen_corr_queries: set[int] = set()
        seen_protect_queries: set[int] = set()
        for step in range(steps):
            corr = corr_batches[step] if step < len(corr_batches) else []
            risk = risk_batches[step] if step < len(risk_batches) else []
            corr_loss = None
            if corr:
                duplicate = seen_corr_queries.intersection(example.query_index for example in corr)
                if duplicate:
                    raise RuntimeError("corrective query repeated within an epoch")
                seen_corr_queries.update(example.query_index for example in corr)
                for example in corr:
                    for action in example.actions:
                        if action.action_id in seen_actions:
                            raise RuntimeError("corrective action repeated within an epoch")
                        seen_actions.add(action.action_id)
                corr_loss, values, _ = corrective_batch_loss(
                    model, store, corr, action_spectra, control_spectra,
                    anchor, device, args, margin_transfer_multiplier,
                )
                for key, value in values.items():
                    logs[f"corrective_{key}"].append(value)
            risk_loss = None
            if risk:
                duplicate = seen_protect_queries.intersection(example.query_index for example in risk)
                if duplicate:
                    raise RuntimeError("protective query repeated within an epoch")
                seen_protect_queries.update(example.query_index for example in risk)
                risk_loss, values = protective_batch_loss(
                    model, store, risk, anchor, device, args,
                )
                for key, value in values.items():
                    logs[f"protective_{key}"].append(value)
            corr_grad = (
                list(torch.autograd.grad(
                    global_gradient_scale * corrective_scale * corr_loss,
                    trainable, allow_unused=True,
                )) if (corr_loss is not None and args.arm == "routed_direct")
                else [None] * len(trainable)
            )
            risk_grad = (
                list(torch.autograd.grad(
                    global_gradient_scale * risk_loss, trainable, allow_unused=True,
                ))
                if risk_loss is not None else [None] * len(trainable)
            )
            projected, projection = project_corrective_against_risk(corr_grad, risk_grad)
            before_projection = _gradient_norm(corr_grad)
            after_projection = _gradient_norm(projected)
            projection_retention = (
                after_projection / before_projection if before_projection > 0 else 1.0
            )
            for parameter, left, right in zip(trainable, projected, risk_grad):
                if left is None and right is None:
                    parameter.grad = None
                elif left is None:
                    parameter.grad = right
                elif right is None:
                    parameter.grad = left
                else:
                    parameter.grad = left + right
            combined = _gradient_norm([
                parameter.grad for parameter in trainable
            ])
            clip_retention = min(1.0, args.grad_clip / combined) if combined > 0 else 1.0
            action_retention = projection_retention * clip_retention
            if corr_loss is not None and args.arm == "routed_direct":
                all_retention.append(action_retention)
                all_clip_retention.append(clip_retention)
                logs["projection_retention"].append(projection_retention)
                logs["clip_retention"].append(clip_retention)
                logs["action_signal_retention"].append(action_retention)
                logs["gradient_conflict"].append(float(bool(projection["conflict"])))
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.eval()
        if len(seen_corr_queries) != len(boundary_examples):
            raise RuntimeError("epoch did not consume every corrective query exactly once")
        if len(seen_actions) != sum(len(example.actions) for example in boundary_examples):
            raise RuntimeError("epoch did not consume every corrective action exactly once")
        if len(seen_protect_queries) != len(protect_examples):
            raise RuntimeError("epoch did not consume every protective query exactly once")
        record = {
            "epoch": epoch + 1, "optimizer_steps": steps,
            "epoch_corrective_queries": len(seen_corr_queries),
            "epoch_corrective_actions": len(seen_actions),
            "epoch_protective_queries": len(seen_protect_queries),
            **{
                key: float(np.mean(value)) for key, value in logs.items() if value
            },
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    if not v3_mode:
        retention_p10 = (
            float(np.quantile(all_retention, 0.10))
            if all_retention and args.arm == "routed_direct" else 1.0
        )
        signal_gate = bool(retention_p10 >= args.minimum_action_retention_p10)
        clip_event_fraction = (
            float(np.mean(np.asarray(all_clip_retention) < 1.0 - 1e-7))
            if all_clip_retention else 0.0
        )
        clip_gate = bool(clip_event_fraction <= args.maximum_clip_event_fraction)
        optimizer_alignment_p10 = 0.0
        optimizer_alignment_gate = True
        if args.arm == "routed_direct" and (not signal_gate or not clip_gate):
            raise RuntimeError(
                f"signal gate failed: retention p10={retention_p10:.4f}, "
                f"clip-event fraction={clip_event_fraction:.4f}"
            )
    final = encode_rows(
        model, store, reachable, device, args.eval_batch_size, args.amp,
        "corrected-routed-final",
    )
    subset_reports = {}
    subset_tables = []
    panels = {
        "train_corrective": corrective_queries,
        "inner_held_corrective": held_corrective_queries,
        "risk": risk_queries,
        "robust": robust_queries,
        "inner_held_robust": held_robust_queries,
        "outer_held": held_outer,
    }
    for name, queries in panels.items():
        report, table = _paired_subset(graph, queries, initial, final, index)
        subset_reports[name] = report
        if len(table):
            table.insert(0, "panel", name)
            subset_tables.append(table)
    evaluation: dict[str, object] = {"development_panels": subset_reports}
    paired = None
    held_metric_evidence_payload: dict[str, np.ndarray] | None = None
    if args.development and len(held_outer):
        initial_scores = score_embedding_query_subset(
            graph, reachable, initial, held_outer,
        )
        final_scores = score_embedding_query_subset(
            graph, reachable, final, held_outer,
        )
        official_metric, _ = full_metrics(
            graph, official_scores(graph), query_adduct, held_outer,
        )
        initial_metric, initial_query = full_metrics(
            graph, initial_scores, query_adduct, held_outer,
        )
        candidate_metric, candidate_query = full_metrics(
            graph, final_scores, query_adduct, held_outer,
        )
        paired = paired_outcome_table(initial_query, candidate_query)
        paired = paired.rename(columns={
            column: column.replace("official_", "initial_E8_", 1)
            for column in paired.columns if column.startswith("official_")
        })
        ci = formula_ci(paired, args.bootstrap_resamples, args.seed)
        near_paired = paired.loc[paired.near].reset_index(drop=True)
        near_ci = (
            formula_ci(
                near_paired, args.bootstrap_resamples, args.seed + 1,
            )
            if len(near_paired) else None
        )
        evaluation["development_held_graph"] = {
            "comparison_contract": {
                "bounded_outer_formula_held_query_panel": True,
                "primary_baseline": "exact_initial_E8_on_same_held_queries",
                "secondary_baseline": "official_embedding_slim_on_same_held_queries",
                "all_metrics_use_clean_input": True,
                "massspecgym_pairwise_is_not_exact_NIST20_replication": True,
                "not_a_formal_promotion_result": True,
            },
            "official": official_metric,
            "initial_E8": initial_metric,
            "candidate": candidate_metric,
            "candidate_vs_initial_E8": {
                "corrected": int(paired.corrected.sum()),
                "introduced": int(paired.introduced.sum()),
                "risk_net_lambda2": int(
                    paired.corrected.sum() - 2 * paired.introduced.sum()
                ),
                "near": {
                    "corrected": int(paired.loc[paired.near, "corrected"].sum()),
                    "introduced": int(paired.loc[paired.near, "introduced"].sum()),
                    "risk_net_lambda2": int(
                        paired.loc[paired.near, "corrected"].sum()
                        - 2 * paired.loc[paired.near, "introduced"].sum()
                    ),
                },
                "formula_cluster_delta_recall1": ci,
                "near_formula_cluster_delta_recall1": near_ci,
            },
        }
    elif not args.development:
        initial_scores = initial_graph_scores
        if initial_scores is None:
            raise RuntimeError("formal initial full-graph scores were not cached")
        final_scores = score_embeddings(graph, reachable, final)
        held = np.flatnonzero(outer_folds == args.outer_fold)
        official_graph_scores = official_scores(graph)
        official_metric, official_query = full_metrics(
            graph, official_graph_scores, query_adduct, held,
        )
        initial_metric, initial_query = full_metrics(
            graph, initial_scores, query_adduct, held,
        )
        candidate_metric, candidate_query = full_metrics(
            graph, final_scores, query_adduct, held,
        )
        held_metric_evidence_payload = held_metric_evidence(
            graph,
            {
                "official": official_graph_scores,
                "initial_E8": initial_scores,
                "candidate": final_scores,
            },
            query_adduct,
            held,
        )
        paired = paired_outcome_table(initial_query, candidate_query)
        paired = paired.rename(columns={
            column: column.replace("official_", "initial_E8_", 1)
            for column in paired.columns
            if column.startswith("official_")
        })
        ci = formula_ci(paired, args.bootstrap_resamples, args.seed)
        near_ci = formula_ci(
            paired.loc[paired.near].reset_index(drop=True),
            args.bootstrap_resamples, args.seed + 1,
        )
        evaluation["formal_held_graph"] = {
            "comparison_contract": {
                "primary_baseline": "exact_initial_E8_on_same_held_graph",
                "secondary_baseline": "official_embedding_slim_on_same_held_graph",
                "all_metrics_use_clean_input": True,
                "massspecgym_pairwise_is_not_exact_NIST20_replication": True,
            },
            "official": official_metric, "initial_E8": initial_metric,
            "candidate": candidate_metric,
            "candidate_vs_initial_E8": {
                "corrected": int(paired.corrected.sum()),
                "introduced": int(paired.introduced.sum()),
                "risk_net_lambda2": int(paired.corrected.sum() - 2 * paired.introduced.sum()),
                "near": {
                    "corrected": int(paired.loc[paired.near, "corrected"].sum()),
                    "introduced": int(paired.loc[paired.near, "introduced"].sum()),
                    "risk_net_lambda2": int(
                        paired.loc[paired.near, "corrected"].sum()
                        - 2 * paired.loc[paired.near, "introduced"].sum()
                    ),
                },
                "formula_cluster_delta_recall1": ci,
                "near_formula_cluster_delta_recall1": near_ci,
            },
        }
    registered_configuration = _registered_report_configuration(args)
    report = {
        "status": "noise_corrected_routed_direct_complete",
        "formal": bool(not args.development), "arm": args.arm,
        "corrective_objective_mode": args.corrective_objective_mode,
        "corrective_branch_mode": args.corrective_branch_mode,
        "direct_contract": args.direct_contract,
        "outer_formula_fold": args.outer_fold,
        "graph_scope": {
            "all_queries": int(graph.n_queries),
            "outer_train_queries": int(len(outer_train)),
            "outer_held_queries": int(registered_outer_held),
            "evaluated_outer_held_queries": int(len(held_outer)),
        },
        "configuration": {
            # Preserve every registered decision variable under its canonical
            # argparse name.  The explicit entries below retain the richer
            # report used by earlier V3 runs, but cannot silently omit a V6
            # registry field.
            **registered_configuration,
            "direct_contract": args.direct_contract,
            "corrective_admission": args.corrective_admission,
            "corrective_branch_mode": args.corrective_branch_mode,
            "corrective_branch_policy": corrective_branch_policy(
                args.corrective_branch_mode
            ).as_dict(),
            "calibration_corrective_branch_mode": (
                args.calibration_corrective_branch_mode
            ),
            "inner_holdout_unit": args.inner_holdout_unit,
            "corrective_margin_floor": args.corrective_margin_floor,
            "arm_invariant_targeted_calibration": bool(
                args.arm_invariant_targeted_calibration
            ),
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "protective_batch_queries": args.protective_batch_queries,
            "positive_references": args.positive_references,
            "negative_molecules": args.negative_molecules,
            "references_per_negative": args.references_per_negative,
            "action_conditioned_positive_references": (
                args.action_conditioned_positive_references
            ),
            "action_conditioned_negative_molecules": (
                args.action_conditioned_negative_molecules
            ),
            "reference_refresh_every_epochs": args.reference_refresh_every_epochs,
            "hard_negative_reference_strategy": (
                "initial_clean_action_control_row_union_epoch_current_by_molecule"
            ),
            "unfreeze_blocks": args.unfreeze_blocks,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "rank_margin": args.rank_margin,
            "temperature": args.temperature,
            "topk_negatives": args.topk_negatives,
            "action_safety_slack": args.action_safety_slack,
            "lambda_action_rank": args.lambda_action_rank,
            "lambda_action_safety": args.lambda_action_safety,
            "lambda_v3_corrective_consistency": (
                args.lambda_v3_corrective_consistency
            ),
            "lambda_v3_robust_rank": args.lambda_v3_robust_rank,
            "lambda_v3_robust_floor": args.lambda_v3_robust_floor,
            "lambda_v3_harmful_rank": args.lambda_v3_harmful_rank,
            "lambda_v3_harmful_floor": args.lambda_v3_harmful_floor,
            "v3_harmful_damage_slack": args.v3_harmful_damage_slack,
            "lambda_clean_continuation": args.lambda_clean_continuation,
            "lambda_risk_floor": args.lambda_risk_floor,
            "lambda_preserve": args.lambda_preserve,
            "risk_margin_slack": args.risk_margin_slack,
            "margin_transfer_fraction": args.margin_transfer_fraction,
            "margin_transfer_cap": args.margin_transfer_cap,
            "transfer_target_allocation": args.transfer_target_allocation,
            "maximum_transfer_cap_factor": args.maximum_transfer_cap_factor,
            "target_v3_payload_to_transfer_ratio": args.target_v3_payload_to_transfer_ratio,
            "target_v3_consistency_to_transfer_ratio": (
                args.target_v3_consistency_to_transfer_ratio
            ),
            "target_v3_robust_to_transfer_ratio": args.target_v3_robust_to_transfer_ratio,
            "target_v3_harmful_to_risk_ratio": args.target_v3_harmful_to_risk_ratio,
            "minimum_v3_internal_combination_retention": (
                args.minimum_v3_internal_combination_retention
            ),
            "minimum_v3_active_transfer_fraction": (
                args.minimum_v3_active_transfer_fraction
            ),
            "v3_maximum_protective_microbatches_per_step": (
                args.v3_maximum_protective_microbatches_per_step
            ),
            "v3_maximum_auxiliary_microbatches_per_step": (
                args.v3_maximum_auxiliary_microbatches_per_step
            ),
            "v3_maximum_corrective_recycle_factor": (
                args.v3_maximum_corrective_recycle_factor
            ),
            "v3_corrective_recycle_full_dose": bool(
                args.v3_corrective_recycle_full_dose
            ),
            "target_dense_corrective_to_risk_ratio": (
                args.target_corrective_to_risk_ratio
            ),
            "target_preclip_gradient_norm": args.target_preclip_gradient_norm,
            "minimum_action_retention_p10": args.minimum_action_retention_p10,
            "maximum_clip_event_fraction": args.maximum_clip_event_fraction,
            "minimum_optimizer_action_alignment_p10": (
                args.minimum_optimizer_action_alignment_p10
            ),
            "minimum_optimizer_action_attributable_fraction_p10": (
                args.minimum_optimizer_action_attributable_fraction_p10
            ),
            "minimum_corrective_direction_retention_p10": (
                args.minimum_corrective_direction_retention_p10
            ),
            "v4_audit_minimum_attributable_fraction": (
                args.v4_audit_minimum_attributable_fraction
            ),
            "v4_audit_maximum_action_gain": args.v4_audit_maximum_action_gain,
            "v4_audit_minimum_update_action_alignment": (
                args.v4_audit_minimum_update_action_alignment
            ),
            "v4_audit_maximum_update_action_coefficient": (
                args.v4_audit_maximum_update_action_coefficient
            ),
            "optimizer_attribution_steps_per_epoch": (
                args.optimizer_attribution_steps_per_epoch
            ),
            "progress_every_steps": args.progress_every_steps,
            "maximum_spectra_per_action_forward": (
                args.maximum_spectra_per_action_forward
            ),
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "examples": {
            "corrective_queries": len(boundary_examples),
            "corrective_actions": int(sum(len(value.actions) for value in boundary_examples)),
            "robust_queries": len(robust_examples),
            "robust_actions": int(sum(len(value.actions) for value in robust_examples)),
            "harmful_queries": len(harmful_examples),
            "harmful_actions": int(sum(len(value.actions) for value in harmful_examples)),
            "protective_queries": len(protect_examples),
            "inner_held_corrective_queries": int(len(held_corrective_queries)),
            "inner_held_robust_queries": int(len(held_robust_queries)),
            "identity_epoch_weights": {
                "corrective": _weight_summary(boundary_examples),
                "robust": _weight_summary(robust_examples),
                "harmful": _weight_summary(harmful_examples),
                "protective": _weight_summary(protect_examples),
            },
        },
        "corrective_admission": corrective_admission_report,
        "inner_held_corrective_admission": held_corrective_admission_report,
        "admitted_action_bank": admitted_action_report,
        "corrective_query_materialization": corrective_materialization_report,
        "action_forward_memory_contract": forward_memory_contract,
        "capacity": capacity,
        "gradient_calibration": calibration,
        "calibration_action_bank": calibration_action_report,
        "v3_branch_scale": v3_branch_scale,
        "schedule": schedule_report,
        "action_control": action_control_report,
        "action_conditioned_boundary": action_boundary_report,
        "mechanism_balance": mechanism_balance_report,
        "signal_transmission": {
            "action_retention_p10": retention_p10,
            "minimum_required": args.minimum_action_retention_p10,
            "passed": signal_gate,
            "clip_event_fraction": clip_event_fraction,
            "maximum_clip_event_fraction": args.maximum_clip_event_fraction,
            "clip_gate_passed": clip_gate,
            "optimizer_action_alignment_p10": optimizer_alignment_p10,
            "minimum_optimizer_action_alignment_p10": (
                args.minimum_optimizer_action_alignment_p10
            ),
            "optimizer_action_alignment_gate_passed": optimizer_alignment_gate,
            "optimizer_action_alignment_steps": schedule_report.get(
                "optimizer_action_alignment_steps", 0
            ),
            "optimizer_action_alignment_sampling": schedule_report.get(
                "optimizer_action_alignment_sampling", "not_applicable"
            ),
            "optimizer_action_attributable_update_fraction_p10": (
                schedule_report.get(
                    "optimizer_action_attributable_update_fraction_p10", 0.0
                )
            ),
            "minimum_optimizer_action_attributable_fraction_p10": (
                args.minimum_optimizer_action_attributable_fraction_p10
            ),
            "optimizer_action_attributable_update_fraction_gate_passed": (
                schedule_report.get(
                    "optimizer_action_attributable_update_fraction_gate_passed",
                    not v3_mode,
                )
            ),
            "optimizer_action_attributable_alignment_p10": (
                schedule_report.get(
                    "optimizer_action_attributable_alignment_p10", 0.0
                )
            ),
            "optimizer_action_attributable_alignment_gate_passed": (
                schedule_report.get(
                    "optimizer_action_attributable_alignment_gate_passed",
                    not v3_mode,
                )
            ),
            "corrective_direction_retention_after_risk_and_clip_p10": (
                schedule_report.get(
                    "corrective_direction_retention_after_risk_and_clip_p10", 0.0,
                )
            ),
            "minimum_corrective_direction_retention_p10": (
                args.minimum_corrective_direction_retention_p10
            ),
            "corrective_direction_retention_gate_passed": schedule_report.get(
                "corrective_direction_retention_gate_passed", not v3_mode,
            ),
            "optimizer_action_attributable_corrective_alignment_p10": (
                schedule_report.get(
                    "optimizer_action_attributable_corrective_alignment_p10", 0.0,
                )
            ),
            "optimizer_action_attributable_corrective_alignment_gate_passed": (
                schedule_report.get(
                    "optimizer_action_attributable_corrective_alignment_gate_passed",
                    not v3_mode,
                )
            ),
            "optimizer_counterfactual_virtual_step_max_relative_error": (
                schedule_report.get(
                    "optimizer_counterfactual_virtual_step_max_relative_error", 0.0
                )
            ),
            "optimizer_counterfactual_virtual_step_gate_passed": (
                schedule_report.get(
                    "optimizer_counterfactual_virtual_step_gate_passed", not v3_mode
                )
            ),
            "optimizer_counterfactual_attribution_steps": schedule_report.get(
                "optimizer_counterfactual_attribution_steps", 0
            ),
            "optimizer_counterfactual_attribution_steps_per_epoch": (
                schedule_report.get(
                    "optimizer_counterfactual_attribution_steps_per_epoch", 0
                )
            ),
            "optimizer_counterfactual_attribution_actual_steps_by_epoch": (
                schedule_report.get(
                    "optimizer_counterfactual_attribution_actual_steps_by_epoch", []
                )
            ),
            "optimizer_counterfactual_attribution_total_optimizer_steps_by_epoch": (
                schedule_report.get(
                    "optimizer_counterfactual_attribution_total_optimizer_steps_by_epoch",
                    [],
                )
            ),
            "optimizer_counterfactual_attribution_step_fraction_by_epoch": (
                schedule_report.get(
                    "optimizer_counterfactual_attribution_step_fraction_by_epoch", []
                )
            ),
            "optimizer_counterfactual_attribution_is_sampled_not_exhaustive": (
                schedule_report.get(
                    "optimizer_counterfactual_attribution_is_sampled_not_exhaustive",
                    False,
                )
            ),
            "optimizer_counterfactual_attribution_scope": schedule_report.get(
                "optimizer_counterfactual_attribution_scope", "not_applicable"
            ),
            "optimizer_counterfactual_attribution_sampling": schedule_report.get(
                "optimizer_counterfactual_attribution_sampling", "not_applicable"
            ),
            "parameter_group_action_signal": schedule_report.get(
                "parameter_group_action_signal", {}
            ),
            "parameter_group_action_signal_gate_passed": schedule_report.get(
                "parameter_group_action_signal_gate_passed", not v3_mode
            ),
            "parameter_group_action_signal_sampling": schedule_report.get(
                "parameter_group_action_signal_sampling", "not_applicable"
            ),
            "parameter_group_action_signal_scope": schedule_report.get(
                "parameter_group_action_signal_scope", "not_applicable"
            ),
            "optimizer_update_restoration_materialized_every_active_step": (
                schedule_report.get(
                    "optimizer_update_restoration_materialized_every_active_step",
                    False,
                )
            ),
            "optimizer_update_restoration_clean_control_noop": (
                schedule_report.get(
                    "optimizer_update_restoration_clean_control_noop", False,
                )
            ),
            "optimizer_update_restoration_scope": schedule_report.get(
                "optimizer_update_restoration_scope", "composite_action"
            ),
            "optimizer_update_restoration_corrective_only": schedule_report.get(
                "optimizer_update_restoration_corrective_only", False
            ),
            "optimizer_update_restoration_safe_exact_corrective": (
                schedule_report.get(
                    "optimizer_update_restoration_safe_exact_corrective", False
                )
            ),
            "optimizer_update_restoration_noncorrective_baseline": (
                schedule_report.get(
                    "optimizer_update_restoration_noncorrective_baseline",
                    "protective_only",
                )
            ),
            "protective_gradient_reaches_every_parameter_on_every_active_step": (
                schedule_report.get(
                    "protective_gradient_reaches_every_parameter_on_every_active_step",
                    False,
                )
            ),
            "optimizer_update_restoration_target_attributable_fraction": (
                schedule_report.get(
                    "optimizer_update_restoration_target_attributable_fraction",
                    0.0,
                )
            ),
            "optimizer_update_restoration_minimum_risk_component_retention": (
                schedule_report.get(
                    "optimizer_update_restoration_minimum_risk_component_retention",
                    1.0,
                )
            ),
            "optimizer_update_restoration_all_group_targets_reached_fraction": (
                schedule_report.get(
                    "optimizer_update_restoration_all_group_targets_reached_fraction",
                    1.0,
                )
            ),
            "minimum_optimizer_restoration_target_reached_fraction": (
                schedule_report.get(
                    "minimum_optimizer_restoration_target_reached_fraction", 0.0,
                )
            ),
            "optimizer_update_restoration_target_coverage_gate_passed": (
                schedule_report.get(
                    "optimizer_update_restoration_target_coverage_gate_passed", True,
                )
            ),
            "optimizer_update_restoration_minimum_observed_group_risk_retention": (
                schedule_report.get(
                    "optimizer_update_restoration_minimum_observed_group_risk_retention",
                    1.0,
                )
            ),
            "optimizer_update_hard_protective_floor_gate_passed": (
                schedule_report.get(
                    "optimizer_update_hard_protective_floor_gate_passed", True
                )
            ),
            "optimizer_update_hard_protective_floor_enforced_fraction": (
                schedule_report.get(
                    "optimizer_update_hard_protective_floor_enforced_fraction",
                    1.0,
                )
            ),
            "optimizer_update_exact_fraction_max_abs_error": (
                schedule_report.get(
                    "optimizer_update_exact_fraction_max_abs_error", 0.0
                )
            ),
            "optimizer_update_exact_fraction_gate_passed": schedule_report.get(
                "optimizer_update_exact_fraction_gate_passed", True
            ),
            "optimizer_update_safe_exact_max_norm_ratio": schedule_report.get(
                "optimizer_update_safe_exact_max_norm_ratio", 1.0
            ),
            "optimizer_update_safe_exact_norm_ratio_gate_passed": (
                schedule_report.get(
                    "optimizer_update_safe_exact_norm_ratio_gate_passed", True
                )
            ),
            "restored_adamw_first_moment_reconciled_every_active_step": (
                schedule_report.get(
                    "restored_adamw_first_moment_reconciled_every_active_step",
                    False,
                )
            ),
            "restored_adamw_first_moment_max_reconstruction_relative_error": (
                schedule_report.get(
                    "restored_adamw_first_moment_max_reconstruction_relative_error",
                    0.0,
                )
            ),
            "restored_adamw_max_materialized_update_relative_error": (
                schedule_report.get(
                    "restored_adamw_max_materialized_update_relative_error", 0.0,
                )
            ),
            "restored_adamw_max_fp32_parameter_replay_relative_error": (
                schedule_report.get(
                    "restored_adamw_max_fp32_parameter_replay_relative_error", 0.0,
                )
            ),
            "restored_adamw_reconstruction_target": schedule_report.get(
                "restored_adamw_reconstruction_target", "not_applicable"
            ),
            "restored_adamw_verification_arithmetic": schedule_report.get(
                "restored_adamw_verification_arithmetic", "not_applicable"
            ),
            "restored_adamw_first_moment_max_change_fraction": (
                schedule_report.get(
                    "restored_adamw_first_moment_max_change_fraction", 0.0,
                )
            ),
            "restored_adamw_second_moment_source": schedule_report.get(
                "restored_adamw_second_moment_source", "unmodified"
            ),
            "signal_gate_failure": schedule_report.get(
                "signal_gate_failure", False,
            ),
            "all_signal_gates_passed": schedule_report.get(
                "all_signal_gates_passed", True,
            ),
            "signal_gate_failure_deferred_to_final_decision": schedule_report.get(
                "signal_gate_failure_deferred_to_final_decision", False,
            ),
            "legacy_90pct_gradient_retention_loss_reproduced": bool(
                retention_p10 <= 0.10
            ),
            "legacy_90pct_end_to_end_loss_reproduced": (
                _legacy_90pct_end_to_end_loss_reproduced(
                    retention_p10,
                    schedule_report,
                    active_v3_arm=(
                        v3_mode
                        and args.arm in {
                            "routed_direct", "shuffled_action_control"
                        }
                    ),
                    minimum_attributable_fraction_p10=(
                        args.minimum_optimizer_action_attributable_fraction_p10
                    ),
                    minimum_corrective_direction_retention_p10=(
                        args.minimum_corrective_direction_retention_p10
                    ),
                    minimum_optimizer_action_alignment_p10=(
                        args.minimum_optimizer_action_alignment_p10
                    ),
                )
            ),
            "legacy_90pct_signal_boundary_observed": (
                _legacy_90pct_signal_boundary_observed(
                    retention_p10,
                    schedule_report,
                    active_v3_arm=(
                        v3_mode
                        and args.arm in {
                            "routed_direct", "shuffled_action_control"
                        }
                    ),
                )
            ),
        },
        "history": history, "evaluation": evaluation,
        "contracts": {
            "mature_E8_initialization": True,
            "query_complete_multi_action_step": True,
            "selected_corrective_action_ids_exactly_preserved_in_query_blocks": bool(
                corrective_materialization_report["exact_action_id_set_preserved"]
                and corrective_materialization_report[
                    "one_boundary_example_per_query"
                ]
            ),
            "maximum_actions_per_query": int(
                max(
                    len(value.actions)
                    for panel in (boundary_examples, robust_examples, harmful_examples)
                    for value in panel
                )
            ),
            "action_multiplicity_is_not_optimizer_dose": True,
            "corrective_recycling_is_schedule_dose_not_source_multiplicity": bool(
                v3_mode
            ),
            "schedule_geometry_validated_before_model_load": bool(
                v3_mode
                and schedule_report.get(
                    "schedule_geometry_matches_pre_model_preflight"
                ) is True
            ),
            "cap_safe_corrective_repartition_preserves_query_action_panel": bool(
                v3_mode
                and schedule_report.get(
                    "cap_safe_corrective_repartition_preserved_query_order_and_coverage"
                ) is True
            ),
            "best_action_v6_exact_877_batch_schedule_verified": bool(
                args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS
                and not args.development
                and schedule_report.get("cap_safe_corrective_batches") == 877
                and np.isclose(
                    float(schedule_report.get(
                        "effective_corrective_recycle_factor", np.nan,
                    )),
                    4.0,
                    rtol=1e-12,
                    atol=1e-12,
                )
            ),
            "family_source_equalized_inside_query": True,
            "P_repeated_generation_source_does_not_multiply_family_dose": bool(
                v3_mode
            ),
            "selector_balances_source_then_family_before_recipe_multiplicity": bool(
                v3_mode
            ),
            "N_P_A4_mechanism_blocks_equalized_inside_query": bool(v3_mode),
            "corrective_N_P_A4_global_epoch_mass_equalized_across_queries": bool(v3_mode),
            "robust_harmful_use_query_local_not_sparse_global_equalization": bool(
                v3_mode
                and schedule_report[
                    "robust_harmful_single_exposure_bounded_scale"
                ]
                and not schedule_report[
                    "sparse_auxiliary_global_duty_compensation_applied"
                ]
            ),
            "dense_auxiliary_panels_cannot_multiply_calibrated_step_mass": bool(
                v3_mode
                and schedule_report[
                    "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step"
                ]
                and schedule_report["maximum_auxiliary_microbatches_per_step"]
                <= schedule_report[
                    "configured_maximum_auxiliary_microbatches_per_step"
                ]
            ),
            "sparse_auxiliary_batches_evenly_interleaved": bool(
                v3_mode
                and schedule_report["robust_harmful_evenly_interleaved_across_epoch"]
                and not schedule_report["robust_harmful_avoidable_step_overlap"]
            ),
            "auxiliary_projected_against_corrective_before_fullgraph_risk": bool(
                v3_mode
                and schedule_report[
                    "auxiliary_projected_against_corrective_before_fullgraph_risk"
                ]
            ),
            "semantic_action_branches_have_separate_gradient_ledgers": bool(
                v3_mode
                and calibration.get(
                    "corrective_robust_harmful_protective_gradients_separately_calibrated"
                ) is True
                and schedule_report[
                    "corrective_robust_harmful_protective_gradients_separately_recorded"
                ]
                and schedule_report[
                    "each_action_branch_vs_protective_cosine_recorded"
                ]
            ),
            "corrective_direction_preserved_through_inner_semantic_projection": bool(
                v3_mode
                and calibration.get(
                    "corrective_direction_preserved_through_inner_semantic_projection"
                ) is True
                and schedule_report[
                    "corrective_direction_preserved_through_inner_semantic_projection"
                ]
            ),
            "control_semantics_explicit_and_separate": bool(v3_mode),
            "global_mechanism_coefficient_cap_passed": bool(
                v3_mode and mechanism_balance_report.get("coefficient_cap_passed")
            ),
            "paired_margin_transfer_action_and_control_targets_detached": True,
            "paired_margin_transfer_updates_shared_clean_query_and_references": bool(
                v3_mode and args.corrective_gradient_locality == "shared"
            ),
            "corrective_gradient_locality": args.corrective_gradient_locality,
            "action_specific_corrective_reference_gradient_disabled": bool(
                v3_mode
                and args.corrective_gradient_locality == "query_action_only"
                and calibration.get("corrective_embedding_role_energy", {}).get(
                    "corrective_reference_gradient_exact_zero"
                ) is True
            ),
            "E4_shared_protective_and_auxiliary_reference_updates_preserved": bool(
                v3_mode
            ),
            "corrective_objective_mode": args.corrective_objective_mode,
            "harmful_action_positive_imitation": False,
            "harmful_action_content_consumed": bool(v3_mode),
            "robust_action_content_consumed": bool(v3_mode),
            "real_action_view_receives_gradient": bool(
                v3_mode
                and args.corrective_branch_mode == "full_action_view"
            ),
            "corrective_branch_mode": args.corrective_branch_mode,
            "scalar_candidate_margin_transfer_receives_gradient": bool(
                v3_mode
                and corrective_branch_policy(
                    args.corrective_branch_mode
                ).transfer
            ),
            "shared_query_reference_rank_updates_preserved": bool(
                v3_mode and args.corrective_gradient_locality == "shared"
            ),
            "initial_and_current_hard_negative_molecule_union": bool(v3_mode),
            "action_and_control_candidate_switch_molecules_retained": bool(v3_mode),
            "action_control_hard_rows_survive_epoch_refresh": bool(v3_mode),
            "clean_action_control_topk_edge_union": bool(v3_mode),
            "action_conditioned_positive_references_retained": bool(v3_mode),
            "E8_symmetric_live_action_consistency_restored": bool(
                v3_mode
                and args.corrective_branch_mode == "full_action_view"
            ),
            "mature_E8_gradient_clip_restored": bool(
                v3_mode and np.isclose(args.grad_clip, 1.0)
            ),
            "official_reference_anchors_frozen": False,
            "protect_only_optimizer_tail": bool(schedule_report["protect_only_optimizer_tail"]),
            "branch_scales_frozen_before_training": bool(v3_mode),
            "all_causal_arms_calibrated_on_same_targeted_action_bank": bool(
                v3_mode
                and calibration_action_report.get("arm_invariant") is True
                and args.arm_invariant_targeted_calibration
            ),
            "each_arm_calibrated_on_actual_post_shuffle_training_bank": bool(
                v3_mode
                and calibration_action_report.get("policy")
                == "arm_specific_training_action_bank"
                and not args.arm_invariant_targeted_calibration
                and calibration_action_bank_sha256 == training_action_bank_sha256
                and calibration_action_report.get(
                    "calibration_occurs_after_admission_and_shuffle"
                ) is True
            ),
            "action_admission_precedes_tensor_subset_reindex_and_shuffle": bool(
                admitted_action_report.get("strategy")
                == "admission_then_exact_tensor_subset_then_reindex_before_shuffle"
                and admitted_action_report.get(
                    "shuffle_donor_pool_equals_optimizer_admitted_panel"
                ) is True
                and admitted_action_report.get(
                    "rejected_corrective_rows_can_be_shuffle_donors"
                ) is False
            ),
            "causal_arm_training_action_bank_hash_contract": bool(
                args.direct_contract not in BEST_ACTION_DIRECT_CONTRACTS
                or (
                    calibration_action_bank_sha256
                    == training_action_bank_sha256
                    and (
                        training_action_bank_sha256
                        != targeted_action_bank_sha256
                    ) == (args.arm == "shuffled_action_control")
                )
            ),
            "formula_stratified_gradient_calibration": bool(v3_mode),
            "clean_control_skips_no_gradient_action_forwards": bool(
                v3_mode and schedule_report[
                    "clean_control_skips_no_gradient_action_forwards"
                ]
            ),
            "training_epoch_transfer_edge_strata_are_pooled_not_batch_averaged": bool(
                v3_mode
                and schedule_report["active_arm_epoch_transfer_edge_strata_recorded"]
            ),
            "corrective_scale_uses_dense_corrective_branches_only": bool(
                v3_mode and calibration.get(
                    "corrective_scale_uses_dense_corrective_branches_only"
                ) is True
            ),
            "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator": bool(
                v3_mode and calibration.get(
                    "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator"
                ) is True
            ),
            "action_calibration_caps_not_silently_truncated": bool(
                v3_mode
                and (
                    args.arm == "clean_control"
                    or (
                        calibration.get("action_to_risk_scale_hit_cap") is False
                        and not any(
                            calibration.get("branch_scale_hit_cap", {}).values()
                        )
                        and calibration.get(
                            "dense_corrective_to_risk_target_exactly_reached"
                        ) is True
                    )
                )
            ),
            "bounded_memory_sequential_action_branch_backprop": bool(
                v3_mode and forward_memory_contract["gate_passed"]
            ),
            "query_complete_action_forward_has_hard_spectra_limit": bool(
                v3_mode
                and forward_memory_contract["gate_passed"]
                and not forward_memory_contract["queries_split_across_forwards"]
            ),
            "registered_formal_v3_configuration_verified": bool(
                v3_mode and not args.development and args.direct_contract == "v3"
            ),
            "registered_formal_best_action_v6_configuration_verified": bool(
                v3_mode
                and not args.development
                and args.direct_contract in BEST_ACTION_DIRECT_CONTRACTS
            ),
            "registered_formal_best_action_v6_restored_configuration_verified": bool(
                v3_mode
                and not args.development
                and args.direct_contract == "best_action_v6_restored"
                and args.materialize_optimizer_update_restoration
                and args.continue_after_signal_gate_failure
            ),
            "registered_formal_best_action_v7_corrective_restored_configuration_verified": bool(
                v3_mode
                and not args.development
                and args.direct_contract == "best_action_v7_corrective_restored"
                and args.materialize_optimizer_update_restoration
                and args.optimizer_restoration_scope == "corrective_only"
                and args.reconcile_restored_adamw_first_moment
                and args.continue_after_signal_gate_failure
            ),
            "registered_formal_best_action_v10_safe_exact_configuration_verified": bool(
                v3_mode
                and not args.development
                and args.direct_contract in SAFE_EXACT_DIRECT_CONTRACTS
                and args.materialize_optimizer_update_restoration
                and args.optimizer_restoration_scope == "safe_exact_corrective"
                and args.reconcile_restored_adamw_first_moment
                and args.inner_holdout_unit == "formula"
                and args.inner_holdout_fold >= 0
            ),
            "historical_best_action_bank_v1_selected_before_injection": bool(
                args.direct_contract != "best_action_v11_historical_best"
                or (
                    args.action_bank_contract == "historical_best_v1"
                    and corrective_admission_report.get(
                        "historical_best_action_bank_v1", {}
                    ).get("exactly_one_champion_per_query") is True
                    and historical_best_action_bank_v1_contract_manifest().get(
                        "injector_dependency"
                    ) == "none_action_supplier_only"
                )
            ),
            "strict_top1_corrective_admission": bool(
                args.corrective_admission == "strict_top1"
                and corrective_admission_report["selected_rows"]
                == len(corrective)
            ),
            "formal_query_truncation_disabled": bool(
                v3_mode
                and not args.development
                and all(getattr(args, name) == 0 for name in (
                    "maximum_corrective_queries", "maximum_risk_queries",
                    "maximum_robust_queries", "maximum_clean_queries",
                    "outer_held_eval_queries",
                ))
            ),
            "source_manifest_exact_graph_alignment": True,
            "route_and_training_formula_fold_seed_exactly_aligned": bool(
                v3_mode
                and args.formula_fold_seed == 20260825
                and ledger_report.get("contracts", {}).get(
                    "all_route_formula_fold_seeds_match"
                ) is True
            ),
            "source_family_shuffled_control_available": True,
            "shuffled_control_matches_supervision_source_family_and_exact_recipe": bool(
                v3_mode
            ),
            "risk_branch_separate": True,
            "PCGrad_risk_veto": True,
            "gradient_clipping_retention_measured": True,
            "optimizer_counterfactual_action_attribution_measured": bool(
                v3_mode and args.arm in {"routed_direct", "shuffled_action_control"}
            ),
            "optimizer_parameter_clones_limited_to_registered_audit_steps": bool(
                v3_mode and not args.materialize_optimizer_update_restoration
            ),
            "optimizer_update_restoration_materialized_every_active_step": bool(
                v3_mode
                and (
                    (
                        args.arm in {
                            "routed_direct", "shuffled_action_control"
                        }
                        and schedule_report.get(
                            "optimizer_update_restoration_materialized_every_active_step"
                        ) is True
                    )
                    or (
                        args.arm == "clean_control"
                        and schedule_report.get(
                            "optimizer_update_restoration_clean_control_noop"
                        ) is True
                    )
                )
            ),
            "head_and_backbone_action_signal_measured_separately": bool(
                v3_mode
                and set(schedule_report.get(
                    "parameter_group_action_signal", {}
                )) == {"head", "backbone"}
            ),
            "corrective_direction_traced_through_risk_clip_and_adamw": bool(
                v3_mode
                and schedule_report.get(
                    "corrective_direction_retention_gate_passed"
                ) is True
                and schedule_report.get(
                    "optimizer_action_attributable_corrective_alignment_gate_passed"
                ) is True
                and all(
                    minimum_gate_passed(
                        values.get(
                            "corrective_direction_retention_after_risk_and_clip_p10",
                            0.0,
                        ),
                        args.minimum_corrective_direction_retention_p10,
                    )
                    and minimum_gate_passed(
                        values.get(
                            "optimizer_action_attributable_corrective_alignment_p10",
                            0.0,
                        ),
                        args.minimum_optimizer_action_alignment_p10,
                    )
                    for values in schedule_report.get(
                        "parameter_group_action_signal", {}
                    ).values()
                )
            ),
            "corrective_only_optimizer_residual_restored": bool(
                v3_mode
                and args.materialize_optimizer_update_restoration
                and args.optimizer_restoration_scope in {
                    "corrective_only", "safe_exact_corrective",
                }
                and schedule_report.get(
                    "optimizer_update_restoration_corrective_only"
                ) is True
            ),
            "safe_exact_optimizer_composition_enforced": bool(
                v3_mode
                and args.optimizer_restoration_scope == "safe_exact_corrective"
                and schedule_report.get(
                    "optimizer_update_restoration_safe_exact_corrective"
                ) is True
                and schedule_report.get(
                    "optimizer_update_hard_protective_floor"
                ) is True
                and schedule_report.get(
                    "optimizer_update_exact_per_group_corrective_fraction"
                ) is True
            ),
            "versioned_action_injector_v1_boundary_used": bool(
                v3_mode
                and args.optimizer_restoration_scope == "safe_exact_corrective"
                and schedule_report.get("action_injector_v1", {}).get(
                    "semantic_contract_sha256"
                ) == action_injector_v1_contract_manifest()[
                    "semantic_contract_sha256"
                ]
            ),
            "restored_adamw_first_moment_reconciled": bool(
                v3_mode
                and args.materialize_optimizer_update_restoration
                and args.reconcile_restored_adamw_first_moment
                and (
                    schedule_report.get(
                        "restored_adamw_first_moment_reconciled_every_active_step"
                    ) is True
                    if args.arm in {
                        "routed_direct", "shuffled_action_control",
                    }
                    else True
                )
            ),
            "outer_held_action_consumed": False,
            "near_corrected_introduced_risk_reported": True,
            "bounded_development_held_full_metric_suite_reported": bool(
                args.development and len(held_outer)
                and "development_held_graph" in evaluation
            ),
            "teacher_embedding_or_distillation_target_used": False,
            "inference_clean_only": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "runtime_seconds": time.time() - started,
        "provenance": {
            "spectrum_data_sha256": observed_input_sha256[
                "spectrum_data_sha256"
            ],
            "architecture_checkpoint_sha256": observed_input_sha256[
                "architecture_checkpoint_sha256"
            ],
            "candidate_graph_sha256": sha256_file(graph_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "source_manifest_sha256": source_manifest_sha256,
            "routed_ledger_report_sha256": sha256_file(ledger_report_path),
            "training_actions_sha256": sha256_file(ledger_path),
            "action_spectra_sha256": sha256_file(spectra_path),
            "initial_student_checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
            "initial_student_decision_sha256": sha256_file(initial_decision_path),
            "official_checkpoint_sha256": official_checkpoint_sha256,
            "script_sha256": sha256_file(Path(__file__)),
            "v3_core_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_direct_v3_core.py")
            ),
            "v3_transfer_objective_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_transfer_objective_v4.py")
            ),
            "optimizer_update_arbitration_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_update_arbitration_v4.py")
            ),
            "action_injector_v1_sha256": sha256_file(
                Path(__file__).with_name("noise_action_injector_v1.py")
            ),
            "gradient_locality_sha256": sha256_file(
                Path(__file__).with_name(
                    "noise_corrected_gradient_locality_v8.py"
                )
            ),
            "v3_action_expansion_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_expansion_v4.py")
            ),
            "noise_v3_core_sha256": sha256_file(
                Path(__file__).with_name("noise_v3_core.py")
            ),
            "v3_action_router_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing_v3.py")
            ),
            "v3_shuffled_control_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_shuffled_control_v3.py")
            ),
            "best_action_selector_sha256": sha256_file(
                Path(__file__).with_name("noise_final_e4_pmt_core.py")
            ),
            "v3_action_panel_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_panel.py")
            ),
            "fullgraph_evaluator_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_fullgraph_evaluation.py")
            ),
        },
        "claim_limit": (
            "Development injection diagnostic only; not a held performance claim."
            if args.development else
            "Outer-formula-held development result; P3 remains untouched."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        final_model_state = model.state_dict()
        report["provenance"]["final_model_state_sha256"] = state_sha256(
            final_model_state
        )
        torch.save({
            "status": "noise_corrected_routed_direct_shared_encoder",
            "model_state": final_model_state, "outer_fold": args.outer_fold,
            "arm": args.arm, "inference_clean_only": True, "P2b_used": False,
            "corrective_objective_mode": args.corrective_objective_mode,
            "direct_contract": args.direct_contract,
            "corrective_branch_mode": args.corrective_branch_mode,
        }, staging / "final_shared_encoder.pt")
        report["provenance"]["final_shared_encoder_sha256"] = sha256_file(
            staging / "final_shared_encoder.pt"
        )
        if subset_tables:
            pd.concat(subset_tables, ignore_index=True).to_csv(
                staging / "development_per_query.csv.gz", index=False, compression="gzip",
            )
        if paired is not None:
            held_per_query_path = staging / "held_per_query.csv.gz"
            paired.to_csv(held_per_query_path, index=False, compression="gzip")
            report["provenance"]["held_per_query_sha256"] = sha256_file(
                held_per_query_path
            )
            report["provenance"]["held_per_query_rows"] = int(len(paired))
        if held_metric_evidence_payload is not None:
            held_metric_evidence_path = staging / "held_metric_evidence.npz"
            np.savez_compressed(
                held_metric_evidence_path, **held_metric_evidence_payload,
            )
            report["provenance"]["held_metric_evidence_sha256"] = sha256_file(
                held_metric_evidence_path
            )
            report["provenance"]["held_metric_evidence_rows"] = {
                "held_queries": int(
                    len(held_metric_evidence_payload["query_index"])
                ),
                "molecules": int(
                    len(held_metric_evidence_payload["molecule_label"])
                ),
                "spectrum_pairs": int(
                    len(held_metric_evidence_payload["pair_label"])
                ),
                "mh_spectrum_pairs": int(np.sum(
                    held_metric_evidence_payload["pair_is_mh"]
                )),
            }
        (staging / "decision.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

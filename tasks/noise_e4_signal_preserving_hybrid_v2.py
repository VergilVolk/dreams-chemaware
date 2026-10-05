"""Signal-preserving bridge between historical E4 and later best actions.

This module does not select actions and does not define a teacher target.  It
owns three narrow, auditable boundaries that were conflated in Hybrid V1:

* every identity contributes exactly four semantic action bags per epoch;
* only the augmented-rank plus symmetric-consistency residual is attributed to
  later actions, with candidate references detached inside that residual; and
* the complete historical E4 update is the protective optimizer baseline.

All action spectra remain inputs to the same shared encoder.  Inference remains
clean-spectrum-only direct fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from noise_action_injector_v1 import (
    _ledger_norm,
    validate_action_injector_v1_frozen_dependencies,
    virtual_adamw_descent_updates_v1,
)
from noise_corrected_update_arbitration_v10 import (
    compose_safe_exact_corrective_updates_by_group,
)
from noise_corrected_update_arbitration_v4 import (
    materialize_descent_updates_,
    reconcile_adamw_first_moments_to_materialized_updates_,
)


@dataclass(frozen=True)
class IdentityActionBag:
    """One identity-equal semantic unit containing one or more action rows."""

    identity: str
    action_indices: tuple[int, ...]
    action_weights: tuple[float, ...]
    effective_weight: float = 1.0


@dataclass(frozen=True)
class IdentityActionBagPlan:
    """Four identity-equal, optimizer-distinct bags per E4-base epoch."""

    epoch_steps: tuple[tuple[tuple[IdentityActionBag, ...], ...], ...]
    action_count: int
    identity_count: int
    identity_source_family_count: int
    maximum_source_families_per_identity: int
    physical_action_exposures: int
    maximum_actions_per_bag: int
    bags_per_identity_per_epoch: int
    minimum_identity_effective_weight_per_epoch: float
    maximum_identity_effective_weight_per_epoch: float
    maximum_step_effective_weight: float
    active_optimizer_steps_per_epoch: int
    zero_semantic_optimizer_steps_per_epoch: int

    def audit_manifest(self) -> dict[str, object]:
        epoch_step_counts = [len(value) for value in self.epoch_steps]
        epoch_bag_counts = [sum(map(len, value)) for value in self.epoch_steps]
        return {
            "scheduler": "identity_source_family_equal_complete_action_bags_v2",
            "action_count": int(self.action_count),
            "identity_count": int(self.identity_count),
            "identity_source_family_count": int(self.identity_source_family_count),
            "maximum_source_families_per_identity": int(
                self.maximum_source_families_per_identity
            ),
            "epochs": int(len(self.epoch_steps)),
            "optimizer_steps_per_epoch": epoch_step_counts,
            "identity_bags_per_epoch": epoch_bag_counts,
            "bags_per_identity_per_epoch": int(self.bags_per_identity_per_epoch),
            "physical_action_exposures": int(self.physical_action_exposures),
            "maximum_actions_per_bag": int(self.maximum_actions_per_bag),
            "minimum_identity_effective_weight_per_epoch": float(
                self.minimum_identity_effective_weight_per_epoch
            ),
            "maximum_identity_effective_weight_per_epoch": float(
                self.maximum_identity_effective_weight_per_epoch
            ),
            "maximum_step_effective_weight": float(
                self.maximum_step_effective_weight
            ),
            "active_optimizer_steps_per_epoch": int(
                self.active_optimizer_steps_per_epoch
            ),
            "zero_semantic_optimizer_steps_per_epoch": int(
                self.zero_semantic_optimizer_steps_per_epoch
            ),
            "all_unique_actions_exposed": True,
            "recycling_only_after_per_identity_source_family_complete_coverage": True,
            "identity_semantic_dose_equal": True,
            "source_family_semantic_dose_equal_within_identity": True,
            "every_source_family_present_on_every_identity_optimizer_step": True,
        }


def build_identity_equal_action_bag_plan(
    action_ids: Sequence[str],
    identities: Sequence[str],
    source_families: Sequence[str],
    *,
    epochs: int,
    bags_per_identity_per_epoch: int,
    optimizer_steps_per_epoch: int,
    seed: int,
) -> IdentityActionBagPlan:
    """Cover all rows with equal identity and within-identity source/family dose.

    Each identity owns exactly ``epochs * bags_per_identity_per_epoch`` bags.
    Every source/family available to that identity is present in every bag and
    receives an equal share of its mass.  Actions are partitioned within their
    source/family; a small family recycles only after all of its unique actions
    have appeared. This prevents a high-row-count P-like family from taking
    more of the fixed-attribution optimizer opportunities than N/A4/V4. The
    bags are distributed over the already-frozen historical E4 optimizer steps.
    """
    n = len(action_ids)
    if not (n == len(identities) == len(source_families)) or n == 0:
        raise ValueError("action bag inputs must be equally sized and non-empty")
    if len(set(map(str, action_ids))) != n:
        raise RuntimeError("action ids must be unique")
    if epochs < 1 or bags_per_identity_per_epoch < 1 or optimizer_steps_per_epoch < 1:
        raise ValueError("action bag schedule dimensions must be positive")

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        grouped.setdefault(str(identity), []).append(index)
    slots = epochs * bags_per_identity_per_epoch
    epoch_bags: list[list[IdentityActionBag]] = [[] for _ in range(epochs)]
    seen: set[int] = set()
    physical_exposures = 0
    maximum_actions_per_bag = 0
    per_identity_epoch_counts: dict[tuple[str, int], int] = {}
    identity_source_family_count = 0
    maximum_source_families_per_identity = 0

    for identity in sorted(grouped):
        family_buckets: dict[str, list[int]] = {}
        for index in grouped[identity]:
            family_buckets.setdefault(str(source_families[index]), []).append(index)
        families = sorted(family_buckets)
        identity_source_family_count += len(families)
        maximum_source_families_per_identity = max(
            maximum_source_families_per_identity, len(families)
        )
        bags: list[list[int]] = [[] for _ in range(slots)]
        bag_weights: list[list[float]] = [[] for _ in range(slots)]
        family_mass = 1.0 / len(families)
        for family in families:
            values = np.asarray(family_buckets[family], dtype=np.int64)
            ordered = values[rng.permutation(len(values))].tolist()
            family_slot_rows: list[list[int]] = [[] for _ in range(slots)]
            if len(ordered) >= slots:
                for position, index in enumerate(ordered):
                    family_slot_rows[position % slots].append(int(index))
            else:
                for position in range(slots):
                    family_slot_rows[position].append(
                        int(ordered[position % len(ordered)])
                    )
            flattened_family = [
                index for slot_rows in family_slot_rows for index in slot_rows
            ]
            if len(ordered) >= slots:
                if sorted(flattened_family) != sorted(ordered):
                    raise RuntimeError("non-recycled source/family changed multiplicity")
            elif set(flattened_family[:len(ordered)]) != set(ordered):
                raise RuntimeError(
                    "a source/family recycled before its complete action coverage"
                )
            for slot, slot_rows in enumerate(family_slot_rows):
                action_mass = family_mass / len(slot_rows)
                bags[slot].extend(slot_rows)
                bag_weights[slot].extend([action_mass] * len(slot_rows))
        if any(not values for values in bags):
            raise RuntimeError("identity action bag scheduler emitted an empty bag")
        for slot, (values, weights) in enumerate(zip(bags, bag_weights)):
            if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("source/family-equal bag mass drifted")
            epoch = slot // bags_per_identity_per_epoch
            epoch_bags[epoch].append(
                IdentityActionBag(
                    identity=identity,
                    action_indices=tuple(values),
                    action_weights=tuple(weights),
                )
            )
            per_identity_epoch_counts[(identity, epoch)] = (
                per_identity_epoch_counts.get((identity, epoch), 0) + 1
            )
            seen.update(values)
            physical_exposures += len(values)
            maximum_actions_per_bag = max(maximum_actions_per_bag, len(values))

    expected_keys = {
        (identity, epoch) for identity in grouped for epoch in range(epochs)
    }
    if set(per_identity_epoch_counts) != expected_keys or any(
        value != bags_per_identity_per_epoch
        for value in per_identity_epoch_counts.values()
    ):
        raise RuntimeError("identity action-bag epoch dose drifted")
    if seen != set(range(n)):
        raise RuntimeError("identity action bags lost a unique action")

    active_steps_per_epoch = len(grouped) * bags_per_identity_per_epoch
    if active_steps_per_epoch > optimizer_steps_per_epoch:
        raise RuntimeError(
            "identity-equal semantic dose needs one optimizer step per action bag: "
            f"{active_steps_per_epoch} > {optimizer_steps_per_epoch}"
        )

    epoch_steps: list[tuple[tuple[IdentityActionBag, ...], ...]] = []
    identity_epoch_effective: list[float] = []
    maximum_step_effective_weight = 0.0
    for epoch, bags in enumerate(epoch_bags):
        order = np.arange(len(bags), dtype=np.int64)
        epoch_rng = np.random.default_rng(seed + 104729 * (epoch + 1))
        order = order[epoch_rng.permutation(len(order))]
        ordered_bags = [bags[int(index)] for index in order]
        expected_bags = len(grouped) * bags_per_identity_per_epoch
        if len(ordered_bags) != expected_bags:
            raise RuntimeError("semantic epoch bag count drifted")

        # Keep each of an identity's four E4-style semantic views on a distinct
        # optimizer step.  Combining four bags before an update is not
        # equivalent once the optimizer-boundary injector fixes the semantic
        # fraction per step: it silently collapses four corrective update
        # opportunities into one.  The corrected full E4 base has enough steps
        # for all bags; all remaining steps execute an exact zero-residual E4
        # base update.
        active_positions = np.sort(epoch_rng.choice(
            optimizer_steps_per_epoch,
            size=active_steps_per_epoch,
            replace=False,
        ))
        mutable_steps: list[list[IdentityActionBag]] = [
            [] for _ in range(optimizer_steps_per_epoch)
        ]
        cursor = 0
        for step_position in active_positions:
            bag = ordered_bags[cursor]
            mutable_steps[int(step_position)] = [IdentityActionBag(
                identity=bag.identity,
                action_indices=bag.action_indices,
                action_weights=bag.action_weights,
                effective_weight=1.0,
            )]
            cursor += 1
        if cursor != len(ordered_bags):
            raise RuntimeError("semantic bags did not fit the frozen E4 steps")
        steps = tuple(tuple(step) for step in mutable_steps)
        effective_by_identity: dict[str, float] = {}
        for step in steps:
            step_weight = sum(bag.effective_weight for bag in step)
            maximum_step_effective_weight = max(
                maximum_step_effective_weight, step_weight,
            )
            if step_weight > 1.0 + 1e-12:
                raise RuntimeError("semantic step effective weight exceeds one")
            for bag in step:
                effective_by_identity[bag.identity] = (
                    effective_by_identity.get(bag.identity, 0.0)
                    + bag.effective_weight
                )
        observed_effective = np.asarray(
            [effective_by_identity[identity] for identity in sorted(grouped)],
            dtype=np.float64,
        )
        if not np.allclose(
            observed_effective,
            float(bags_per_identity_per_epoch),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "identity semantic effective dose must equal the E4 view count"
            )
        identity_epoch_effective.extend(observed_effective.tolist())
        epoch_steps.append(steps)

    return IdentityActionBagPlan(
        epoch_steps=tuple(epoch_steps),
        action_count=n,
        identity_count=len(grouped),
        identity_source_family_count=identity_source_family_count,
        maximum_source_families_per_identity=maximum_source_families_per_identity,
        physical_action_exposures=physical_exposures,
        maximum_actions_per_bag=maximum_actions_per_bag,
        bags_per_identity_per_epoch=bags_per_identity_per_epoch,
        minimum_identity_effective_weight_per_epoch=float(
            min(identity_epoch_effective)
        ),
        maximum_identity_effective_weight_per_epoch=float(
            max(identity_epoch_effective)
        ),
        maximum_step_effective_weight=float(maximum_step_effective_weight),
        active_optimizer_steps_per_epoch=active_steps_per_epoch,
        zero_semantic_optimizer_steps_per_epoch=(
            optimizer_steps_per_epoch - active_steps_per_epoch
        ),
    )


def flatten_weighted_action_bags(
    bags: Sequence[IdentityActionBag],
) -> tuple[list[int], list[float]]:
    """Flatten bags so every identity-bag has equal gradient mass in one step."""
    if not bags:
        raise ValueError("at least one action bag is required")
    indices: list[int] = []
    weights: list[float] = []
    for bag in bags:
        if not bag.action_indices:
            raise RuntimeError("action bag is empty")
        if len(bag.action_indices) != len(bag.action_weights):
            raise RuntimeError("action bag indices and solved weights do not align")
        if not math.isclose(
            sum(bag.action_weights), 1.0, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError("action bag source/family weights do not sum to one")
        indices.extend(map(int, bag.action_indices))
        weights.extend(
            float(bag.effective_weight) * float(value)
            for value in bag.action_weights
        )
    if not 0 < sum(weights) <= 1.0 + 1e-12:
        raise RuntimeError("flattened identity-bag weights are not bounded")
    return indices, weights


def bounded_semantic_microbatches(
    action_indices: Sequence[int],
    action_weights: Sequence[float],
    spectra_per_action: Sequence[int],
    *,
    maximum_spectra_per_forward: int,
) -> list[tuple[list[int], list[float]]]:
    """Pack one optimizer step without renormalizing its solved action weights."""
    if not (
        len(action_indices) == len(action_weights)
        and len(spectra_per_action) > max(map(int, action_indices), default=-1)
    ):
        raise ValueError("semantic microbatch ledgers do not align")
    if maximum_spectra_per_forward < 1:
        raise ValueError("maximum spectra per forward must be positive")
    output: list[tuple[list[int], list[float]]] = []
    indices: list[int] = []
    weights: list[float] = []
    spectra = 0
    for action_index, action_weight in zip(action_indices, action_weights):
        action_index = int(action_index)
        count = int(spectra_per_action[action_index])
        if count < 1 or count > maximum_spectra_per_forward:
            raise RuntimeError(
                f"one semantic action needs {count} spectra; cap="
                f"{maximum_spectra_per_forward}"
            )
        if indices and spectra + count > maximum_spectra_per_forward:
            output.append((indices, weights))
            indices, weights, spectra = [], [], 0
        indices.append(action_index)
        weights.append(float(action_weight))
        spectra += count
    if indices:
        output.append((indices, weights))
    if [index for batch, _ in output for index in batch] != list(
        map(int, action_indices)
    ):
        raise RuntimeError("semantic microbatch packing lost action order")
    flattened_weights = [weight for _, batch in output for weight in batch]
    if not np.allclose(
        flattened_weights, np.asarray(action_weights, dtype=np.float64),
        rtol=0.0, atol=0.0,
    ):
        raise RuntimeError("semantic microbatch packing renormalized action weights")
    return output


def query_local_e4_semantic_residual(
    encoded: torch.Tensor,
    layouts: Sequence[Mapping[str, object]],
    action_weights: Sequence[float],
    *,
    rank_margin: float,
    temperature: float,
    lambda_clean_rank: float,
    lambda_aug_rank: float,
    lambda_consistency: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the clean-boundary E4 residual with the reference bypass closed.

    The later panel contains actions that already make their transformed view
    Top-1.  Continuing to reward every such action view recreates the historical
    easy-action gradient bug.  The residual therefore keeps E4's clean rank on
    the exact selected molecule boundary and stops action-rank pressure once
    the live action clears the registered E4 margin. E4's validated symmetric
    clean/action consistency remains unchanged. Candidate references are
    detached only inside this residual; the complete historical E4 base stream
    remains live and unchanged.
    """
    if len(layouts) != len(action_weights) or not layouts:
        raise ValueError("semantic residual layouts and weights must align")
    weights = torch.as_tensor(
        action_weights, device=encoded.device, dtype=encoded.dtype,
    )
    weight_sum = weights.sum()
    if (
        not torch.isfinite(weights).all()
        or torch.any(weights <= 0)
        or weight_sum > 1.0 + 1e-6
    ):
        raise ValueError("semantic residual weights must be positive with total at most one")

    clean = torch.stack([encoded[int(layout["clean"])] for layout in layouts])
    action = torch.stack([encoded[int(layout["action"])] for layout in layouts])
    clean_margins: list[torch.Tensor] = []
    action_margins: list[torch.Tensor] = []
    for clean_embedding, action_embedding, layout in zip(clean, action, layouts):
        positive = encoded[list(map(int, layout["positive"]))].detach()
        negative = encoded[list(map(int, layout["negative"]))].detach()
        if len(positive) == 0 or len(negative) == 0:
            raise RuntimeError("semantic residual requires positive and negative references")
        clean_margins.append(
            torch.max(positive @ clean_embedding)
            - torch.max(negative @ clean_embedding)
        )
        action_margins.append(
            torch.max(positive @ action_embedding)
            - torch.max(negative @ action_embedding)
        )
    clean_margin = torch.stack(clean_margins)
    action_margin = torch.stack(action_margins)
    clean_rank_each = F.softplus(
        (float(rank_margin) - clean_margin) / float(temperature)
    )
    raw_aug_rank_each = F.softplus(
        (float(rank_margin) - action_margin) / float(temperature)
    )
    action_rank_active = action_margin.detach() < float(rank_margin)
    aug_rank_each = torch.where(
        action_rank_active, raw_aug_rank_each, torch.zeros_like(raw_aug_rank_each)
    )
    consistency_each = 1.0 - torch.sum(clean * action, dim=1)
    clean_rank = torch.sum(weights * clean_rank_each)
    aug_rank = torch.sum(weights * aug_rank_each)
    consistency = torch.sum(weights * consistency_each)
    loss = (
        float(lambda_clean_rank) * clean_rank
        + float(lambda_aug_rank) * aug_rank
        + float(lambda_consistency) * consistency
    )
    return loss, {
        "semantic_clean_rank": float(clean_rank.detach()),
        "semantic_aug_rank": float(aug_rank.detach()),
        "semantic_consistency": float(consistency.detach()),
        "semantic_clean_margin": float(
            torch.sum(weights * clean_margin).detach()
        ),
        "semantic_action_margin": float(
            torch.sum(weights * action_margin).detach()
        ),
        "semantic_action_rank_active_fraction": float(
            (
                torch.sum(weights * action_rank_active.to(weights.dtype))
                / weight_sum
            ).detach()
        ),
        "semantic_action_margin_pass_fraction": float(
            (
                torch.sum(weights * (action_margin > 0).to(weights.dtype))
                / weight_sum
            ).detach()
        ),
        "semantic_reference_gradient_detached": True,
        "semantic_clean_boundary_rank_active": True,
        "semantic_satisfied_action_rank_gradient_gated": True,
        "semantic_symmetric_clean_action_consistency": True,
        "semantic_weight_sum": float(weight_sum.detach()),
    }


def _clone_gradients(
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _gradient_norm(values: Sequence[torch.Tensor | None]) -> float:
    terms = [
        torch.sum(value.detach().float() ** 2).double()
        for value in values if value is not None
    ]
    return float(torch.sqrt(torch.stack(terms).sum())) if terms else 0.0


def _scaled(
    values: Sequence[torch.Tensor | None], scale: float,
) -> list[torch.Tensor | None]:
    return [None if value is None else value.detach() * float(scale) for value in values]


def _sum_gradients(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    if len(left) != len(right):
        raise ValueError("gradient ledgers differ")
    output: list[torch.Tensor | None] = []
    for base, semantic in zip(left, right):
        if base is None and semantic is None:
            output.append(None)
        elif base is None:
            output.append(semantic.detach().clone())
        elif semantic is None:
            output.append(base.detach().clone())
        else:
            output.append(base.detach() + semantic.detach())
    return output


def _subtract_ledgers(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    if len(left) != len(right):
        raise ValueError("update ledgers differ")
    output: list[torch.Tensor | None] = []
    for actual, expected in zip(left, right):
        if actual is None and expected is None:
            output.append(None)
        elif actual is None:
            output.append(-expected.detach())
        elif expected is None:
            output.append(actual.detach().clone())
        else:
            output.append(actual.detach() - expected.detach())
    return output


def _set_gradients(
    parameters: Sequence[torch.nn.Parameter], values: Sequence[torch.Tensor | None],
) -> None:
    if len(parameters) != len(values):
        raise ValueError("parameters and gradient ledger differ")
    for parameter, value in zip(parameters, values):
        parameter.grad = None if value is None else value.detach().clone()


@dataclass(frozen=True)
class SignalPreservingInjectionV2Step:
    semantic_active: bool
    target_attributable_fraction: float
    historical_e4_gradient_norm_before_clip: float
    semantic_gradient_norm_before_clip: float
    combined_gradient_norm_before_clip: float
    historical_e4_clip_retention: float
    combined_clip_retention: float
    optimizer_action_fraction_by_group: dict[str, float]
    historical_e4_component_retention_by_group: dict[str, float]
    final_to_shadow_e4_update_norm_ratio_by_group: dict[str, float]
    final_to_historical_e4_update_norm_ratio: float
    minimum_historical_e4_retention: float
    maximum_historical_e4_update_norm_ratio: float
    maximum_fraction_abs_error: float
    virtual_adamw_relative_error: float
    first_moment_reconstruction_relative_error: float
    fp32_parameter_replay_relative_error: float
    actual_vs_shadow_first_moment_relative_error_by_group: dict[str, float]
    actual_vs_shadow_second_moment_relative_error_by_group: dict[str, float]
    zero_semantic_exact_e4_equivalence: bool
    zero_semantic_shadow_update_materialized: bool


class E4SignalPreservingInjectorBridgeV2:
    """Inject a semantic residual around an independent E4-only AdamW stream.

    The shadow optimizer never owns the live model parameters.  Before each
    step its cloned parameters are synchronized to the current model while its
    first/second moments continue using only the historical E4 gradient.  This
    prevents semantic second-moment history from silently shrinking the next
    step's E4 baseline.
    """

    def __init__(
        self,
        optimizer: torch.optim.AdamW,
        parameters: Sequence[torch.nn.Parameter],
        *,
        target_attributable_fraction: float = 0.25,
        minimum_historical_e4_retention: float = 0.90,
        maximum_historical_e4_update_norm_ratio: float = 1.50,
        verify_frozen_dependencies: bool = True,
    ) -> None:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("E4 signal-preserving bridge requires AdamW")
        self.optimizer = optimizer
        self.parameters = list(parameters)
        if not self.parameters or len({id(value) for value in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("bridge parameters must be non-empty and unique")
        position = {id(parameter): index for index, parameter in enumerate(self.parameters)}
        groups: dict[str, list[int]] = {}
        for index, group in enumerate(optimizer.param_groups):
            name = str(group.get("group_name", f"group_{index}"))
            if name in groups:
                raise ValueError(f"duplicate optimizer group: {name}")
            groups[name] = [position[id(parameter)] for parameter in group["params"]]
        observed = [position for values in groups.values() for position in values]
        if set(observed) != set(range(len(self.parameters))) or len(observed) != len(
            self.parameters
        ):
            raise ValueError("optimizer groups do not partition bridge parameters")
        self.parameter_group_positions = groups
        if not 0 < float(target_attributable_fraction) < 1:
            raise ValueError("target attributable fraction must be in (0, 1)")
        if not 0 <= float(minimum_historical_e4_retention) <= 1:
            raise ValueError("minimum historical E4 retention must be in [0, 1]")
        if float(maximum_historical_e4_update_norm_ratio) < 1:
            raise ValueError("maximum update norm ratio must be at least one")
        self.target_attributable_fraction = float(target_attributable_fraction)
        self.minimum_historical_e4_retention = float(minimum_historical_e4_retention)
        self.maximum_historical_e4_update_norm_ratio = float(
            maximum_historical_e4_update_norm_ratio
        )
        # Construction imports the frozen V1 virtual-AdamW implementation.  The
        # old V1 class itself is deliberately not used because all three of its
        # counterfactuals share the semantic-contaminated optimizer state.
        self.validated_dependency_sha256 = (
            validate_action_injector_v1_frozen_dependencies()
            if verify_frozen_dependencies else {}
        )
        self.shadow_parameters = [
            torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)
            for parameter in self.parameters
        ]
        shadow_groups: list[dict[str, object]] = []
        for group in self.optimizer.param_groups:
            options = {
                key: value for key, value in group.items()
                if key != "params"
            }
            options["params"] = [
                self.shadow_parameters[position[id(parameter)]]
                for parameter in group["params"]
            ]
            shadow_groups.append(options)
        self.shadow_optimizer = torch.optim.AdamW(shadow_groups)
        self._semantic_gradients: list[torch.Tensor | None] | None = None

    def audit_manifest(self) -> dict[str, object]:
        return {
            "bridge": "historical_E4_baseline_plus_query_local_semantic_residual_v2",
            "historical_e4_baseline_is_protective_axis": True,
            "independent_historical_e4_shadow_adamw_state": True,
            "shadow_parameter_values_resynchronized_before_each_step": True,
            "shadow_first_second_moments_receive_only_historical_e4_gradient": True,
            "historical_e4_and_semantic_gradients_clipped_independently": True,
            "one_real_adamw_step": True,
            "minimum_historical_e4_update_norm_ratio": (
                self.minimum_historical_e4_retention
            ),
            "target_optimizer_action_fraction": self.target_attributable_fraction,
            "maximum_update_norm_ratio_to_independent_historical_e4": (
                self.maximum_historical_e4_update_norm_ratio
            ),
            "composition_kernel": "V10_hard_safe_exact_groupwise",
            "virtual_adamw_kernel": "ActionInjectorV1_frozen_virtual_adamw",
            "validated_dependency_sha256": dict(
                self.validated_dependency_sha256
            ),
        }

    def capture_semantic_corrective_(self, *, allow_zero: bool = False) -> None:
        if self._semantic_gradients is not None:
            raise RuntimeError("semantic gradient was captured twice")
        captured = _clone_gradients(self.parameters)
        if _gradient_norm(captured) <= 0 and not allow_zero:
            raise RuntimeError("semantic corrective gradient is zero")
        self._semantic_gradients = captured

    def _clipped(
        self, gradients: Sequence[torch.Tensor | None], maximum: float,
    ) -> tuple[list[torch.Tensor | None], float, float]:
        _set_gradients(self.parameters, gradients)
        raw_norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, maximum))
        if not math.isfinite(raw_norm):
            raise RuntimeError("non-finite gradient norm")
        clipped = _clone_gradients(self.parameters)
        retention = min(1.0, maximum / (raw_norm + 1e-6))
        return clipped, raw_norm, float(retention)

    def _advance_shadow(
        self, historical_clipped: Sequence[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        for live, shadow in zip(self.parameters, self.shadow_parameters):
            shadow.data.copy_(live.detach())
        before = [parameter.detach().clone() for parameter in self.parameters]
        _set_gradients(self.shadow_parameters, historical_clipped)
        self.shadow_optimizer.step()
        self.shadow_optimizer.zero_grad(set_to_none=True)
        return [
            None if gradient is None else live_before - shadow.detach()
            for live_before, shadow, gradient in zip(
                before, self.shadow_parameters, historical_clipped
            )
        ]

    def _optimizer_state_divergence(self) -> tuple[dict[str, float], dict[str, float]]:
        first: dict[str, float] = {}
        second: dict[str, float] = {}
        for name, positions in self.parameter_group_positions.items():
            first_num = first_den = second_num = second_den = 0.0
            for position in positions:
                actual_state = self.optimizer.state.get(self.parameters[position], {})
                shadow_state = self.shadow_optimizer.state.get(
                    self.shadow_parameters[position], {}
                )
                for key, numerator, denominator in (
                    ("exp_avg", "first_num", "first_den"),
                    ("exp_avg_sq", "second_num", "second_den"),
                ):
                    actual = actual_state.get(key)
                    shadow = shadow_state.get(key)
                    if actual is None or shadow is None:
                        continue
                    difference = actual.detach().float() - shadow.detach().float()
                    num = float(torch.sum(difference * difference).double())
                    den = float(torch.sum(shadow.detach().float() ** 2).double())
                    if numerator == "first_num":
                        first_num += num
                        first_den += den
                    else:
                        second_num += num
                        second_den += den
            first[name] = math.sqrt(first_num / max(first_den, 1e-30))
            second[name] = math.sqrt(second_num / max(second_den, 1e-30))
        return first, second

    @staticmethod
    def _group_norm_ratios(
        final: Sequence[torch.Tensor | None],
        baseline: Sequence[torch.Tensor | None],
        groups: Mapping[str, Sequence[int]],
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, positions in groups.items():
            final_norm = _ledger_norm([final[int(position)] for position in positions])
            baseline_norm = _ledger_norm([
                baseline[int(position)] for position in positions
            ])
            output[str(name)] = final_norm / max(baseline_norm, 1e-30)
        return output

    @staticmethod
    def _group_projection_retention(
        final: Sequence[torch.Tensor | None],
        baseline: Sequence[torch.Tensor | None],
        groups: Mapping[str, Sequence[int]],
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, positions in groups.items():
            numerator = 0.0
            denominator = 0.0
            for position in positions:
                reference = baseline[int(position)]
                if reference is None:
                    continue
                value = final[int(position)]
                if value is None:
                    value = torch.zeros_like(reference)
                numerator += float(torch.sum(
                    value.detach().float() * reference.detach().float()
                ).double())
                denominator += float(torch.sum(
                    reference.detach().float() ** 2
                ).double())
            output[str(name)] = (
                numerator / denominator if denominator > 0 else 1.0
            )
        return output

    def _actual_and_shadow_states_equal(self) -> bool:
        for position in range(len(self.parameters)):
            actual_state = self.optimizer.state.get(self.parameters[position], {})
            shadow_state = self.shadow_optimizer.state.get(
                self.shadow_parameters[position], {}
            )
            if set(actual_state) != set(shadow_state):
                return False
            for key in actual_state:
                left = actual_state[key]
                right = shadow_state[key]
                if torch.is_tensor(left) and torch.is_tensor(right):
                    if not torch.equal(left, right):
                        return False
                elif left != right:
                    return False
        return True

    def step_and_inject_(
        self, *, maximum_gradient_norm: float,
    ) -> SignalPreservingInjectionV2Step:
        if self._semantic_gradients is None:
            raise RuntimeError("semantic corrective gradient was not captured")
        if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm <= 0:
            raise ValueError("maximum gradient norm must be finite and positive")
        historical_raw = _clone_gradients(self.parameters)
        semantic_norm = _gradient_norm(self._semantic_gradients)
        if _gradient_norm(historical_raw) <= 0:
            raise RuntimeError("historical E4 gradient is zero")
        historical_clipped, historical_norm, historical_scale = self._clipped(
            historical_raw, maximum_gradient_norm,
        )
        shadow_updates = self._advance_shadow(historical_clipped)
        combined_raw = _sum_gradients(historical_raw, self._semantic_gradients)
        combined_clipped, combined_norm, combined_scale = self._clipped(
            combined_raw, maximum_gradient_norm,
        )
        virtual_combined = virtual_adamw_descent_updates_v1(
            self.optimizer, self.parameters, combined_clipped,
        )
        before = [parameter.detach().clone() for parameter in self.parameters]

        if semantic_norm <= 0:
            _set_gradients(self.parameters, historical_clipped)
            self.optimizer.step()
            standard_updates = [
                None if gradient is None else left - parameter.detach()
                for left, parameter, gradient in zip(
                    before, self.parameters, historical_clipped
                )
            ]
            exact_updates = all(
                (update is None and shadow is None)
                or (
                    update is not None
                    and shadow is not None
                    and torch.equal(update, shadow)
                )
                for update, shadow in zip(standard_updates, shadow_updates)
            )
            exact_states = self._actual_and_shadow_states_equal()
            virtual_residual = _subtract_ledgers(
                standard_updates, virtual_combined,
            )
            virtual_error = _ledger_norm(virtual_residual) / max(
                _ledger_norm(standard_updates), 1e-30,
            )
            if exact_updates and exact_states:
                final_updates = standard_updates
                reconstruction = {
                    "gate_passed": True,
                    "same_step_reconstruction_relative_error": 0.0,
                    "same_step_fp32_parameter_replay_relative_error": 0.0,
                }
            else:
                # A previous semantic step leaves the real AdamW moments
                # intentionally different from the independent E4-only stream.
                # A later zero-semantic step must therefore materialize the
                # shadow E4 displacement; directly accepting optimizer.step()
                # would silently reintroduce the contaminated baseline.
                materialize_descent_updates_(
                    self.parameters, before, shadow_updates,
                )
                reconstruction = reconcile_adamw_first_moments_to_materialized_updates_(
                    self.optimizer, self.parameters, before, shadow_updates,
                )
                if not reconstruction["gate_passed"]:
                    raise RuntimeError(
                        "zero-semantic shadow-E4 reconciliation failed: "
                        f"{reconstruction}"
                    )
                final_updates = [
                    None if update is None else left - parameter.detach()
                    for left, parameter, update in zip(
                        before, self.parameters, shadow_updates
                    )
                ]
            group_ratios = self._group_norm_ratios(
                final_updates, shadow_updates, self.parameter_group_positions,
            )
            group_projection = self._group_projection_retention(
                final_updates, shadow_updates, self.parameter_group_positions,
            )
            zero_materialized = bool(
                min(group_ratios.values()) + 1e-6 >= 1.0
                and min(group_projection.values()) + 1e-6 >= 1.0
            )
            if not zero_materialized:
                raise RuntimeError(
                    "zero-semantic step did not materialize the shadow E4 update: "
                    f"norm={group_ratios}, projection={group_projection}"
                )
            first_divergence, second_divergence = self._optimizer_state_divergence()
            self._semantic_gradients = None
            return SignalPreservingInjectionV2Step(
                semantic_active=False,
                target_attributable_fraction=self.target_attributable_fraction,
                historical_e4_gradient_norm_before_clip=historical_norm,
                semantic_gradient_norm_before_clip=0.0,
                combined_gradient_norm_before_clip=combined_norm,
                historical_e4_clip_retention=historical_scale,
                combined_clip_retention=combined_scale,
                optimizer_action_fraction_by_group={name: 0.0 for name in group_ratios},
                historical_e4_component_retention_by_group=group_projection,
                final_to_shadow_e4_update_norm_ratio_by_group=group_ratios,
                final_to_historical_e4_update_norm_ratio=(
                    _ledger_norm(final_updates)
                    / max(_ledger_norm(shadow_updates), 1e-30)
                ),
                minimum_historical_e4_retention=self.minimum_historical_e4_retention,
                maximum_historical_e4_update_norm_ratio=(
                    self.maximum_historical_e4_update_norm_ratio
                ),
                maximum_fraction_abs_error=0.0,
                virtual_adamw_relative_error=float(virtual_error),
                first_moment_reconstruction_relative_error=float(
                    reconstruction["same_step_reconstruction_relative_error"]
                ),
                fp32_parameter_replay_relative_error=float(
                    reconstruction["same_step_fp32_parameter_replay_relative_error"]
                ),
                actual_vs_shadow_first_moment_relative_error_by_group=first_divergence,
                actual_vs_shadow_second_moment_relative_error_by_group=second_divergence,
                zero_semantic_exact_e4_equivalence=bool(exact_updates and exact_states),
                zero_semantic_shadow_update_materialized=zero_materialized,
            )

        composition = compose_safe_exact_corrective_updates_by_group(
            virtual_combined,
            shadow_updates,
            shadow_updates,
            self.parameter_group_positions,
            target_attributable_fraction=self.target_attributable_fraction,
            minimum_protective_component_retention=(
                self.minimum_historical_e4_retention
            ),
            materialize_updates=True,
        )
        if not composition.all_groups_target_reached:
            raise RuntimeError("exact semantic update fraction was not reached")
        if not composition.all_groups_protective_floor_enforced:
            raise RuntimeError("historical E4 protective floor was not enforced")
        _set_gradients(self.parameters, combined_clipped)
        self.optimizer.step()
        standard_updates = [
            None if gradient is None else left - parameter.detach()
            for left, parameter, gradient in zip(
                before, self.parameters, combined_clipped
            )
        ]
        virtual_residual = _subtract_ledgers(standard_updates, virtual_combined)
        virtual_error = _ledger_norm(virtual_residual) / max(
            _ledger_norm(standard_updates), 1e-30,
        )
        materialize_descent_updates_(
            self.parameters, before, composition.updates,
        )
        reconciliation = reconcile_adamw_first_moments_to_materialized_updates_(
            self.optimizer, self.parameters, before, composition.updates,
        )
        if not reconciliation["gate_passed"]:
            raise RuntimeError(
                "signal-preserving AdamW first-moment reconciliation failed: "
                f"{reconciliation}"
            )
        historical_update_norm = _ledger_norm(shadow_updates)
        final_update_norm = _ledger_norm(composition.updates)
        norm_ratio = final_update_norm / historical_update_norm
        if norm_ratio + 1e-6 < self.minimum_historical_e4_retention:
            raise RuntimeError(
                "absolute historical E4 update retention failed: "
                f"{norm_ratio:.6f} < {self.minimum_historical_e4_retention:.6f}"
            )
        if norm_ratio > self.maximum_historical_e4_update_norm_ratio + 1e-6:
            raise RuntimeError(
                "signal-preserving update exceeds the absolute E4 norm cap: "
                f"{norm_ratio:.6f} > {self.maximum_historical_e4_update_norm_ratio:.6f}"
            )
        group_reports = composition.parameter_groups
        group_norm_ratios = self._group_norm_ratios(
            composition.updates, shadow_updates, self.parameter_group_positions,
        )
        if min(group_norm_ratios.values()) + 1e-6 < self.minimum_historical_e4_retention:
            raise RuntimeError(
                "per-group absolute historical E4 update retention failed: "
                f"{group_norm_ratios}"
            )
        if max(group_norm_ratios.values()) > self.maximum_historical_e4_update_norm_ratio + 1e-6:
            raise RuntimeError(
                "per-group signal-preserving update exceeds the absolute E4 norm cap: "
                f"{group_norm_ratios}"
            )
        first_divergence, second_divergence = self._optimizer_state_divergence()
        step = SignalPreservingInjectionV2Step(
            semantic_active=True,
            target_attributable_fraction=self.target_attributable_fraction,
            historical_e4_gradient_norm_before_clip=historical_norm,
            semantic_gradient_norm_before_clip=semantic_norm,
            combined_gradient_norm_before_clip=combined_norm,
            historical_e4_clip_retention=float(historical_scale),
            combined_clip_retention=float(combined_scale),
            optimizer_action_fraction_by_group={
                name: float(report.final_attributable_fraction)
                for name, report in group_reports.items()
            },
            historical_e4_component_retention_by_group={
                name: float(report.risk_component_retention)
                for name, report in group_reports.items()
            },
            final_to_shadow_e4_update_norm_ratio_by_group=group_norm_ratios,
            final_to_historical_e4_update_norm_ratio=float(norm_ratio),
            minimum_historical_e4_retention=self.minimum_historical_e4_retention,
                maximum_historical_e4_update_norm_ratio=(
                self.maximum_historical_e4_update_norm_ratio
            ),
            maximum_fraction_abs_error=float(
                composition.maximum_group_attributable_fraction_abs_error
            ),
            virtual_adamw_relative_error=float(virtual_error),
            first_moment_reconstruction_relative_error=float(
                reconciliation["same_step_reconstruction_relative_error"]
            ),
            fp32_parameter_replay_relative_error=float(
                reconciliation["same_step_fp32_parameter_replay_relative_error"]
            ),
            actual_vs_shadow_first_moment_relative_error_by_group=first_divergence,
            actual_vs_shadow_second_moment_relative_error_by_group=second_divergence,
            zero_semantic_exact_e4_equivalence=False,
            zero_semantic_shadow_update_materialized=False,
        )
        self._semantic_gradients = None
        return step


def summarize_signal_preserving_v2_steps(
    steps: Sequence[SignalPreservingInjectionV2Step],
) -> dict[str, object]:
    if not steps:
        return {"enabled": False, "steps": 0, "gate_passed": False}
    groups = sorted(steps[0].optimizer_action_fraction_by_group)
    if any(sorted(step.optimizer_action_fraction_by_group) != groups for step in steps):
        raise RuntimeError("signal-preserving parameter groups drifted")
    configured_floors = {
        float(step.minimum_historical_e4_retention) for step in steps
    }
    if len(configured_floors) != 1:
        raise RuntimeError("historical E4 retention floor changed within the run")
    configured_floor = configured_floors.pop()
    configured_caps = {
        float(step.maximum_historical_e4_update_norm_ratio) for step in steps
    }
    if len(configured_caps) != 1:
        raise RuntimeError("historical E4 update norm cap changed within the run")
    configured_cap = configured_caps.pop()
    configured_targets = {
        float(step.target_attributable_fraction) for step in steps
    }
    if len(configured_targets) != 1:
        raise RuntimeError("semantic attributable target changed within the run")
    configured_target = configured_targets.pop()
    active_steps = [step for step in steps if step.semantic_active]
    zero_steps = [step for step in steps if not step.semantic_active]
    fraction_p10 = {
        group: float(np.quantile([
            step.optimizer_action_fraction_by_group[group] for step in steps
            if step.semantic_active
        ], 0.10))
        for group in groups
    } if active_steps else {group: 0.0 for group in groups}
    retention_min = {
        group: float(min(
            step.historical_e4_component_retention_by_group[group] for step in steps
        ))
        for group in groups
    }
    group_norm_ratio_min = {
        group: float(min(
            step.final_to_shadow_e4_update_norm_ratio_by_group[group]
            for step in steps
        ))
        for group in groups
    }
    group_norm_ratio_max = {
        group: float(max(
            step.final_to_shadow_e4_update_norm_ratio_by_group[group]
            for step in steps
        ))
        for group in groups
    }
    report = {
        "enabled": True,
        "steps": int(len(steps)),
        "semantic_active_steps": int(len(active_steps)),
        "zero_semantic_steps": int(len(zero_steps)),
        "attribution_sampling": "every_optimizer_step",
        "target_optimizer_action_fraction": float(configured_target),
        "optimizer_action_fraction_p10_by_group": fraction_p10,
        "optimizer_action_fraction_max_abs_error": float(max(
            step.maximum_fraction_abs_error for step in steps
            if step.semantic_active
        )) if active_steps else 0.0,
        "minimum_historical_e4_component_retention_by_group": retention_min,
        "minimum_final_to_shadow_e4_update_norm_ratio_by_group": (
            group_norm_ratio_min
        ),
        "maximum_final_to_shadow_e4_update_norm_ratio_by_group": (
            group_norm_ratio_max
        ),
        "minimum_final_to_historical_e4_update_norm_ratio": float(min(
            step.final_to_historical_e4_update_norm_ratio for step in steps
        )),
        "median_final_to_historical_e4_update_norm_ratio": float(np.median([
            step.final_to_historical_e4_update_norm_ratio for step in steps
        ])),
        "maximum_final_to_historical_e4_update_norm_ratio": float(max(
            step.final_to_historical_e4_update_norm_ratio for step in steps
        )),
        "minimum_historical_e4_clip_retention": float(min(
            step.historical_e4_clip_retention for step in steps
        )),
        "minimum_combined_clip_retention": float(min(
            step.combined_clip_retention for step in steps
        )),
        "maximum_virtual_adamw_relative_error": float(max(
            step.virtual_adamw_relative_error for step in steps
        )),
        "maximum_first_moment_reconstruction_relative_error": float(max(
            step.first_moment_reconstruction_relative_error for step in steps
        )),
        "maximum_fp32_parameter_replay_relative_error": float(max(
            step.fp32_parameter_replay_relative_error for step in steps
        )),
        "historical_e4_absolute_update_floor": float(configured_floor),
        "historical_e4_absolute_update_cap": float(configured_cap),
        "maximum_actual_vs_shadow_first_moment_relative_error_by_group": {
            group: float(max(
                step.actual_vs_shadow_first_moment_relative_error_by_group[group]
                for step in steps
            ))
            for group in groups
        },
        "maximum_actual_vs_shadow_second_moment_relative_error_by_group": {
            group: float(max(
                step.actual_vs_shadow_second_moment_relative_error_by_group[group]
                for step in steps
            ))
            for group in groups
        },
        "all_zero_semantic_shadow_updates_materialized": bool(all(
            step.zero_semantic_shadow_update_materialized for step in zero_steps
        )),
        "all_zero_only_steps_bitwise_e4_equivalent": bool(all(
            step.zero_semantic_exact_e4_equivalence for step in zero_steps
        )),
    }
    report["gate_passed"] = bool(
        (
            not active_steps
            or (
                report["optimizer_action_fraction_max_abs_error"] <= 2e-6
                and all(
                    abs(value - configured_target) <= 2e-6
                    for value in fraction_p10.values()
                )
            )
        )
        and min(retention_min.values()) + 1e-6 >= configured_floor
        and min(group_norm_ratio_min.values()) + 1e-6 >= configured_floor
        and max(group_norm_ratio_max.values()) <= configured_cap + 1e-6
        and report["minimum_final_to_historical_e4_update_norm_ratio"] + 1e-6
        >= configured_floor
        and report["maximum_final_to_historical_e4_update_norm_ratio"]
        <= configured_cap + 1e-6
        and report["maximum_virtual_adamw_relative_error"] <= 1e-3
        and report["maximum_first_moment_reconstruction_relative_error"] <= 1e-6
        and report["all_zero_semantic_shadow_updates_materialized"]
    )
    return report

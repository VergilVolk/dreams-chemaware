"""Frozen best-action inventory for the next direct noise fine-tuning route.

This module deliberately separates experiment names from executable spectrum
actions.  E4 and E8 are training/initialization foundations, while E13 reused
the E12B relaxed-recurrence action and therefore is not a seventh action bank.
The executable union is the mature N/P/A4 bank plus the small set of v4
gradient-path actions that passed either a broad paired gate or an exact
per-instance routing gate on the frozen development panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


V4_SOURCE = "V4_gradient_path"


@dataclass(frozen=True)
class V4ActionRecipe:
    name: str
    family: str
    evidence_role: str
    development_evidence: str

    @property
    def recipe_id(self) -> str:
        return f"{V4_SOURCE}|{self.name}"


# Do not add every explored transform.  Signed multiplicative, conservative
# exchange, dual attenuation, strong up/joint and sequential 3/5/6 either had
# non-positive risk-net, crossing CIs, no unique strict correction, or a worse
# eligible-panel tradeoff.  Mixed families below are still routed per instance;
# their adverse rows can only enter the harmful branch.
V4_BEST_ACTION_RECIPES = (
    V4ActionRecipe(
        "quad_attenuation_4x25", "gradient_attenuation", "broad_safe",
        "3 corrected, 0 introduced; paired formula CI strictly positive",
    ),
    V4ActionRecipe(
        "quad_attenuation_4x50", "gradient_attenuation", "broad_safe",
        "5 corrected, 0 introduced; paired formula CI strictly positive",
    ),
    V4ActionRecipe(
        "supported_boost_2x50", "supported_boost", "broad_safe",
        "5 corrected, 0 introduced; paired formula CI strictly positive",
    ),
    V4ActionRecipe(
        "adaptive_trust_up", "adaptive_supported_boost", "broad_safe",
        "5 corrected, 0 introduced; paired formula CI strictly positive",
    ),
    V4ActionRecipe(
        "adaptive_trust_down", "adaptive_gradient_attenuation",
        "exact_routed_extension",
        "mixed family; contributes a strict correction outside the broad union",
    ),
    V4ActionRecipe(
        "adaptive_strong_down", "adaptive_gradient_attenuation",
        "exact_routed_extension",
        "mixed family; adds three strict corrections beyond mild actions",
    ),
    V4ActionRecipe(
        "sequential_supported_boost_1x", "sequential_supported_boost",
        "exact_routed_extension",
        "mixed prefix; contributes one unique strict correction",
    ),
    V4ActionRecipe(
        "sequential_supported_boost_2x", "sequential_supported_boost",
        "broad_safe",
        "5 corrected, 0 introduced; paired formula CI strictly positive",
    ),
    V4ActionRecipe(
        "sequential_supported_boost_4x", "sequential_supported_boost",
        "exact_routed_extension",
        "4 corrected, 0 introduced on its eligible panel; strict positive CI",
    ),
)


def registered_v4_action_recipes() -> tuple[V4ActionRecipe, ...]:
    names = [recipe.name for recipe in V4_BEST_ACTION_RECIPES]
    ids = [recipe.recipe_id for recipe in V4_BEST_ACTION_RECIPES]
    if len(names) != 9 or len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise RuntimeError("best-action v4 registry is malformed")
    return V4_BEST_ACTION_RECIPES


def v4_recipe_map() -> dict[str, V4ActionRecipe]:
    return {recipe.name: recipe for recipe in registered_v4_action_recipes()}


def select_registered_v4_views(
    views: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Keep only frozen v4 recipes while preserving generator order."""
    registry = v4_recipe_map()
    selected: list[dict[str, object]] = []
    observed: set[str] = set()
    for raw in views:
        name = str(raw.get("recipe", ""))
        if name not in registry:
            continue
        if name in observed:
            raise RuntimeError(f"v4 generator emitted duplicate recipe: {name}")
        observed.add(name)
        item = dict(raw)
        item["family"] = registry[name].family
        item["evidence_role"] = registry[name].evidence_role
        item["recipe_id"] = registry[name].recipe_id
        selected.append(item)
    return selected


def action_lineage_contract() -> dict[str, object]:
    """Machine-readable correction to the historical experiment/action mix-up."""
    return {
        "E4": "direct_finetuning_foundation_not_action_bank",
        "E8": "mature_shared_encoder_and_curriculum_not_action_bank",
        "E12B": "relaxed_recurrence_spectrum_actions_in_P_bank",
        "E13": "training_attempt_reusing_E12B_not_new_action_bank",
        "mature_executable_sources": [
            "N_mature", "P_guided_original", "E10B", "E11", "E12B",
            "A4_exact", V4_SOURCE,
        ],
        "teacher_embedding_or_distillation_target_used": False,
    }

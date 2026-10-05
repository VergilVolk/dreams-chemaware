"""Complete frozen positive-action registry for direct shared-encoder routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from audit_noise_final_e10_positive_residual_matrix import cell_variant
from audit_noise_final_e11_reference_diversity_matrix import select_rows
from audit_noise_final_positive_guided_matrix import reference_profile
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks


@dataclass(frozen=True)
class PRecipe:
    source: str
    reference_policy: str
    family: str
    dose: float
    auxiliary_dose: float
    minimum_reference_prevalence: float
    maximum_transferred_peaks: int
    support_weighted: bool

    @property
    def recipe_id(self) -> str:
        return (
            f"{self.source}|{self.reference_policy}|{self.family}|dose={self.dose:.2f}|"
            f"aux={self.auxiliary_dose:.2f}|prevalence={self.minimum_reference_prevalence:.2f}|"
            f"max={self.maximum_transferred_peaks}|weighted={int(self.support_weighted)}"
        )


E10_RECIPES = (
    *(("consensus_projection", dose, 0.0) for dose in (0.25, 0.50, 0.75, 1.00)),
    *(("recurrent_union_mix", dose, 0.0) for dose in (0.10, 0.25, 0.50)),
    *(("matched_intensity_transport", dose, 0.0) for dose in (0.50, 1.00)),
    *(("recurrent_peak_graft", dose, 0.0) for dose in (0.10, 0.25, 0.50)),
    *(("balanced_peak_exchange", dose, 0.0) for dose in (0.10, 0.25, 0.50)),
    ("transport_then_union", 0.50, 0.50),
    ("transport_then_union", 1.00, 0.50),
    ("consensus_then_union", 0.50, 0.50),
    ("consensus_then_union", 0.75, 0.50),
)
# The original full-graph positive-guided matrix contained all 12 combinations
# of three existing-peak intensity families and four doses.  E10B re-used all
# four consensus cells and two transport cells, but did not repeat the four
# prevalence cells or transport doses 0.25/0.75.  Keep only those six unique
# recipes here so the union is complete without duplicating an action tensor.
P_GUIDED_ORIGINAL_UNIQUE_RECIPES = (
    *(('prevalence_attenuation', dose, 0.0) for dose in (0.25, 0.50, 0.75, 1.00)),
    ('matched_intensity_transport', 0.25, 0.0),
    ('matched_intensity_transport', 0.75, 0.0),
)
E11_POLICIES = ("farthest3", "maxmin6", "condition6", "maxmin12")
E11_RECIPES = (
    ("recurrent_union_mix", 0.50, 0.0),
    ("balanced_peak_exchange", 0.50, 0.0),
    ("consensus_then_union", 0.75, 0.50),
    ("transport_then_union", 1.00, 0.50),
)
E12_POLICIES = ("top3",) + E11_POLICIES
P_MISSING_PEAK_FAMILIES = {
    "recurrent_union_mix", "recurrent_peak_graft", "balanced_peak_exchange",
    "transport_then_union", "consensus_then_union",
}


def registered_p_recipes() -> tuple[PRecipe, ...]:
    recipes: list[PRecipe] = []
    for family, dose, auxiliary in P_GUIDED_ORIGINAL_UNIQUE_RECIPES:
        recipes.append(PRecipe(
            "P_guided_original", "top3", family, dose, auxiliary, 0.67, 5, False,
        ))
    for family, dose, auxiliary in E10_RECIPES:
        recipes.append(PRecipe("E10B", "top3", family, dose, auxiliary, 0.67, 5, False))
    for policy in E11_POLICIES:
        for family, dose, auxiliary in E11_RECIPES:
            recipes.append(PRecipe("E11", policy, family, dose, auxiliary, 0.67, 5, False))
    for policy in E12_POLICIES:
        for maximum in (5, 10):
            for dose in (0.25, 0.50):
                recipes.append(PRecipe(
                    "E12B", policy, "recurrent_union_mix", dose, 0.0, 0.50, maximum, False,
                ))
        recipes.append(PRecipe(
            "E12B", policy, "recurrent_union_mix", 0.50, 0.0, 0.50, 10, True,
        ))
    ids = [recipe.recipe_id for recipe in recipes]
    if len(recipes) != 66 or len(set(ids)) != len(ids):
        raise RuntimeError("complete positive-action registry must contain exactly 66 unique cells")
    return tuple(recipes)


def choose_reference_rows(
    rows: np.ndarray,
    scores: np.ndarray,
    vectors: np.ndarray,
    policy: str,
    *,
    query_instrument: str,
    query_collision_energy: float,
    instruments: Mapping[int, str],
    collision_energy: Mapping[int, float],
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if policy == "top3":
        return rows[np.argsort(-scores, kind="stable")[:3]]
    return select_rows(
        rows, scores, vectors, policy, query_instrument, query_collision_energy,
        dict(instruments), dict(collision_energy),
    )


def materialize_p_action(
    clean: torch.Tensor,
    references: list[torch.Tensor],
    recipe: PRecipe,
    fragment_tolerance: float,
    *,
    profile: tuple[np.ndarray, np.ndarray] | None = None,
    missing: np.ndarray | None = None,
) -> torch.Tensor:
    if profile is None:
        profile = reference_profile(clean, references, fragment_tolerance)
    if missing is None:
        missing = (
            recurrent_missing_peaks(
                clean, references, fragment_tolerance,
                recipe.minimum_reference_prevalence,
                recipe.maximum_transferred_peaks,
            )
            if recipe.family in P_MISSING_PEAK_FAMILIES
            else np.empty((0, 3), dtype=np.float32)
        )
    missing = np.asarray(missing, dtype=np.float32).copy()
    if recipe.support_weighted and len(missing):
        missing[:, 1] *= missing[:, 2]
    return cell_variant(
        clean, profile, missing, recipe.family, recipe.dose, recipe.auxiliary_dose,
    )

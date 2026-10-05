"""Frozen historical-best action supplier for direct noise fine-tuning.

The supplier is deliberately separate from the optimizer-boundary injector.
It reproduces the no-op-aware E12-B / E4-native champion rule on the current
training geometry: among all mature, strict Top-1 corrective actions for one
query, retain the action with the largest realized positive margin.  It never
reads an outer-held outcome and never changes an action tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from noise_final_e4_pmt_core import select_materialized_best_action_union


HISTORICAL_BEST_ACTION_SOURCES = frozenset({
    "N_mature",
    "P_guided_original",
    "E10B",
    "E11",
    "E12B",
    "A4_exact",
    "V4_gradient_path",
})

REGISTERED_FULL_LEDGER_COUNTS = {
    "outer_train_queries": 65286,
    "strict_top1_rows_before_margin_floor": 32127,
    "strict_top1_champions_before_margin_floor": 3483,
    "selected_rows": 3482,
    "selected_queries": 3482,
}


def _ordered_text_sha256(values: pd.Series) -> str:
    payload = "\n".join(values.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class HistoricalBestActionBankV1Config:
    margin_floor: float = 5e-6
    require_all_registered_sources: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.margin_floor) or self.margin_floor < 0:
            raise ValueError("historical-best margin floor must be finite and non-negative")


@dataclass(frozen=True)
class HistoricalBestActionBankV1Selection:
    actions: pd.DataFrame
    report: dict[str, object]


class HistoricalBestActionBankV1:
    """Select one deterministic champion action for every correctable query."""

    VERSION = "noise_historical_best_action_bank_v1"
    POLICY = "current_E8_strict_top1_max_margin_per_query"

    def __init__(self, config: HistoricalBestActionBankV1Config) -> None:
        self.config = config

    def select(self, actions: pd.DataFrame) -> HistoricalBestActionBankV1Selection:
        observed_sources = set(actions["source"].astype(str))
        unknown = observed_sources - HISTORICAL_BEST_ACTION_SOURCES
        if unknown:
            raise RuntimeError(
                f"historical-best action bank contains unknown sources: {sorted(unknown)}"
            )
        if self.config.require_all_registered_sources and (
            observed_sources != HISTORICAL_BEST_ACTION_SOURCES
        ):
            raise RuntimeError(
                "historical-best action bank source closure drifted: "
                f"{sorted(observed_sources)}"
            )

        champions, strict = select_materialized_best_action_union(actions)
        strict_margin = pd.to_numeric(strict["action_margin"], errors="raise")
        strong_strict = strict.loc[
            strict_margin.gt(float(self.config.margin_floor))
        ].copy()
        champion_margin = pd.to_numeric(champions["action_margin"], errors="raise")
        selected = champions.loc[
            champion_margin.gt(float(self.config.margin_floor))
        ].copy().reset_index(drop=True)
        if selected.empty:
            raise RuntimeError("historical-best action selection is empty")
        if selected["action_id"].duplicated().any():
            raise RuntimeError("historical-best action selection duplicates action IDs")
        if selected["query_index"].duplicated().any():
            raise RuntimeError("historical-best action selection repeats a query")
        if set(selected["action_id"].astype(str)) - set(
            strong_strict["action_id"].astype(str)
        ):
            raise RuntimeError("historical-best champion escaped the strict strong panel")

        selected_sources = set(selected["source"].astype(str))
        report: dict[str, object] = {
            "version": self.VERSION,
            "selection_policy": self.POLICY,
            "margin_floor": float(self.config.margin_floor),
            "input_rows": int(len(actions)),
            "strict_top1_rows_before_margin_floor": int(len(strict)),
            "strict_top1_champions_before_margin_floor": int(len(champions)),
            "strict_top1_rows_after_margin_floor": int(len(strong_strict)),
            "selected_rows": int(len(selected)),
            "selected_queries": int(selected["query_index"].nunique()),
            "candidate_views_removed_after_champion_selection": int(
                len(strong_strict) - len(selected)
            ),
            "exactly_one_champion_per_query": True,
            "all_selected_actions_preserved": True,
            "one_best_query_compression_used": True,
            "registered_sources": sorted(HISTORICAL_BEST_ACTION_SOURCES),
            "observed_input_sources": sorted(observed_sources),
            "selected_sources": sorted(selected_sources),
            "selected_rows_by_source": {
                str(key): int(value)
                for key, value in selected["source"].astype(str).value_counts(
                    sort=False
                ).sort_index().items()
            },
            "selected_action_ids_sha256": _ordered_text_sha256(
                selected["action_id"]
            ),
            "selection_uses_current_training_geometry_only": True,
            "outer_held_outcomes_used": False,
            "historical_outcomes_used_as_row_weights": False,
            "action_tensors_mutated": False,
            "teacher_embedding_or_distillation_target_used": False,
        }
        return HistoricalBestActionBankV1Selection(selected, report)


def historical_best_action_bank_v1_contract_manifest() -> dict[str, object]:
    return {
        "version": HistoricalBestActionBankV1.VERSION,
        "selection_policy": HistoricalBestActionBankV1.POLICY,
        "registered_sources": sorted(HISTORICAL_BEST_ACTION_SOURCES),
        "one_champion_per_query": True,
        "outer_held_outcomes_used": False,
        "action_tensors_mutated": False,
        "injector_dependency": "none_action_supplier_only",
        "teacher_embedding_or_distillation_target_used": False,
    }

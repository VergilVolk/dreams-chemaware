"""Small, dependency-free contracts for the V9 direct-injection ablation.

V9 does not introduce a new teacher or a new action bank.  It separates the
historical scalar candidate-margin transfer from the live action-view gradient
so that their clean-query contribution can be measured under a matched total
corrective-gradient budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


CORRECTIVE_BRANCH_MODES = (
    "full_action_view",
    "scalar_transfer_only",
)


@dataclass(frozen=True)
class CorrectiveBranchPolicy:
    name: str
    transfer: bool
    payload: bool
    consistency: bool

    def scale_enabled(self, branch: str) -> bool:
        if branch not in {"transfer", "payload", "consistency"}:
            raise ValueError(f"unknown corrective branch: {branch}")
        return bool(getattr(self, branch))

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "name": self.name,
            "transfer": self.transfer,
            "payload": self.payload,
            "consistency": self.consistency,
        }


def corrective_branch_policy(mode: str) -> CorrectiveBranchPolicy:
    if mode == "full_action_view":
        return CorrectiveBranchPolicy(
            name=mode, transfer=True, payload=True, consistency=True,
        )
    if mode == "scalar_transfer_only":
        return CorrectiveBranchPolicy(
            name=mode, transfer=True, payload=False, consistency=False,
        )
    raise ValueError(f"unsupported corrective branch mode: {mode}")


def minimum_gate_passed(
    observed: float,
    minimum: float,
    *,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-6,
) -> bool:
    """Numerically stable inclusive minimum comparison.

    A target such as 0.25 must not fail because float32/AdamW accounting
    returns 0.24999998.  The tolerance is deliberately tiny: it cannot turn a
    scientifically material miss into a pass.
    """
    observed = float(observed)
    minimum = float(minimum)
    if not math.isfinite(observed) or not math.isfinite(minimum):
        return False
    tolerance = max(
        float(absolute_tolerance),
        abs(minimum) * float(relative_tolerance),
    )
    return observed + tolerance >= minimum


def restoration_coverage_gate(
    *,
    active_arm: bool,
    materialized: bool,
    observed_fraction: float,
    minimum_fraction: float,
) -> bool:
    """Gate optimizer restoration by executed behavior, never a label string."""
    if not active_arm or not materialized:
        return True
    return minimum_gate_passed(observed_fraction, minimum_fraction)

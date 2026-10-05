"""Gradient-locality contracts for direct noise-action fine-tuning.

The mature E4/E8 continuation remains a shared query/reference objective.  The
only configurable boundary here is the *action-specific corrective residual*:
``shared`` reproduces V7, while ``query_action_only`` stops that residual from
being satisfied by reference-side gradients.  No teacher embedding or margin
regression is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


CORRECTIVE_GRADIENT_LOCALITIES = ("shared", "query_action_only")


@dataclass(frozen=True)
class CorrectiveGradientLocality:
    name: str
    clean_query_live: bool
    action_view_live: bool
    reference_live: bool


def corrective_gradient_locality(name: str) -> CorrectiveGradientLocality:
    """Return the exact live-path policy for one corrective residual."""
    if name == "shared":
        return CorrectiveGradientLocality(
            name=name,
            clean_query_live=True,
            action_view_live=True,
            reference_live=True,
        )
    if name == "query_action_only":
        return CorrectiveGradientLocality(
            name=name,
            clean_query_live=True,
            action_view_live=True,
            reference_live=False,
        )
    raise ValueError(f"unsupported corrective gradient locality: {name}")


def _energy_fraction(norm: float, total_norm: float) -> float:
    norm = float(norm)
    total_norm = float(total_norm)
    if not math.isfinite(norm) or not math.isfinite(total_norm):
        raise ValueError("gradient norms must be finite")
    if norm < 0 or total_norm < 0:
        raise ValueError("gradient norms must be non-negative")
    return (norm / total_norm) ** 2 if total_norm > 0 else 0.0


def corrective_embedding_role_report(
    diagnostics: Mapping[str, float],
    *,
    locality: str,
    zero_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Convert first-batch embedding-gradient norms into role energy shares.

    The input keys are the medians emitted by ``v3_action_panel_batch_loss``.
    Squared norm fractions are reported because role slices are disjoint rows
    of the same encoded tensor; their squared norms, unlike raw norm ratios,
    have an additive energy interpretation.
    """
    policy = corrective_gradient_locality(locality)
    if zero_tolerance < 0:
        raise ValueError("zero tolerance must be non-negative")
    branches: dict[str, object] = {}
    for branch in ("transfer", "payload", "consistency"):
        prefix = f"corrective.embedding_grad_{branch}_"
        total = float(diagnostics.get(prefix + "total_norm", 0.0))
        roles = {
            role: {
                "norm": float(diagnostics.get(prefix + f"{role}_norm", 0.0)),
                "energy_fraction": _energy_fraction(
                    float(diagnostics.get(prefix + f"{role}_norm", 0.0)),
                    total,
                ),
            }
            for role in ("query", "action", "control", "reference")
        }
        branches[branch] = {
            "total_norm": total,
            "roles": roles,
            "disjoint_role_energy_sum": float(sum(
                float(value["energy_fraction"]) for value in roles.values()
            )),
        }

    transfer = branches["transfer"]
    payload = branches["payload"]
    consistency = branches["consistency"]
    assert isinstance(transfer, dict)
    assert isinstance(payload, dict)
    assert isinstance(consistency, dict)
    transfer_roles = transfer["roles"]
    payload_roles = payload["roles"]
    consistency_roles = consistency["roles"]
    assert isinstance(transfer_roles, dict)
    assert isinstance(payload_roles, dict)
    assert isinstance(consistency_roles, dict)
    reference_norms = (
        float(transfer_roles["reference"]["norm"]),
        float(payload_roles["reference"]["norm"]),
        float(consistency_roles["reference"]["norm"]),
    )
    required_live_norms = (
        float(transfer_roles["query"]["norm"]),
        float(payload_roles["action"]["norm"]),
        float(consistency_roles["query"]["norm"]),
        float(consistency_roles["action"]["norm"]),
    )
    reference_zero = bool(max(reference_norms, default=0.0) <= zero_tolerance)
    required_live = bool(min(required_live_norms, default=0.0) > zero_tolerance)
    query_local_gate = bool(
        locality == "query_action_only" and reference_zero and required_live
    )
    return {
        "corrective_gradient_locality": locality,
        "policy": {
            "clean_query_live": policy.clean_query_live,
            "action_view_live": policy.action_view_live,
            "reference_live": policy.reference_live,
            "shared_E4_protective_and_auxiliary_paths_unchanged": True,
        },
        "branches": branches,
        "transfer_reference_dominates_query_energy": bool(
            float(transfer_roles["reference"]["energy_fraction"])
            > float(transfer_roles["query"]["energy_fraction"])
        ),
        "payload_reference_dominates_action_energy": bool(
            float(payload_roles["reference"]["energy_fraction"])
            > float(payload_roles["action"]["energy_fraction"])
        ),
        "corrective_reference_gradient_exact_zero": reference_zero,
        "required_clean_and_action_paths_live": required_live,
        "query_action_only_locality_gate_applicable": locality == "query_action_only",
        "query_action_only_locality_gate_passed": (
            query_local_gate if locality == "query_action_only" else None
        ),
        "zero_tolerance": float(zero_tolerance),
    }

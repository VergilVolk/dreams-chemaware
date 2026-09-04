"""Pure, fail-closed utilities for dynamic conditional direct noise training.

This module contains no model code and no downstream candidate expert.  It
turns formula-crossfit clean-visible scores and one-epoch-lagged action
advantages into bounded action weights.  The functions are deliberately pure
so every sampling and weighting invariant can be tested before loading DreaMS.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


PHASE_A_ARMS = (
    "clean_continuation",
    "matched_random",
    "static_target",
    "dynamic_np",
)

N_CELLS = frozenset(
    [("candidate_gradient", 0.50, step) for step in range(3, 7)]
    + [("role_confounder", 1.00, step) for step in range(1, 6)]
)

WEIGHT_COLUMNS = frozenset({
    "action_id", "query_index", "identity", "formula", "family",
    "p_clean", "lagged_advantage", "risk",
})


@dataclass(frozen=True)
class WeightConfig:
    """Frozen first-pass weighting parameters.

    The numeric values bound exposure; they are not performance hyperparameter
    search dimensions.  Any later change must create a new named protocol.
    """

    tau_floor: float = 0.005
    max_action_weight: float = 1.0
    max_query_weight: float = 1.0
    minimum_family_ess: float = 32.0

    def validate(self) -> None:
        if self.tau_floor <= 0:
            raise ValueError("tau_floor must be positive")
        if self.max_action_weight <= 0 or self.max_query_weight <= 0:
            raise ValueError("weight caps must be positive")
        if self.minimum_family_ess <= 0:
            raise ValueError("minimum_family_ess must be positive")


def stable_control_index(action_id: str, controls: int = 2, seed: int = 20260904) -> int:
    """Choose a frozen matched control without reading an action outcome."""
    if controls < 1:
        raise ValueError("controls must be positive")
    payload = f"{seed}|{action_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % controls


def robust_temperature(values: Iterable[float], floor: float) -> float:
    """Return a deterministic robust scale for the lagged advantage sigmoid."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("lagged advantage has no finite observations")
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return float(max(float(q75 - q25), float(floor)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _validate_weight_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = WEIGHT_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"dynamic weight frame misses columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("dynamic weight frame is empty")
    if frame["action_id"].astype(str).duplicated().any():
        raise RuntimeError("action_id must be globally unique")
    if frame[["query_index", "identity", "formula", "family"]].isna().any().any():
        raise RuntimeError("action grouping columns contain missing values")
    output = frame.copy()
    for column in ("p_clean", "lagged_advantage", "risk"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(np.float64)
        if not np.all(np.isfinite(output[column])):
            raise RuntimeError(f"{column} contains a non-finite value")
    if not output["p_clean"].between(0.0, 1.0).all():
        raise RuntimeError("p_clean must lie in [0, 1]")
    if not output["risk"].between(0.0, 1.0).all():
        raise RuntimeError("risk must lie in [0, 1]")
    return output


def _cap_queries(frame: pd.DataFrame, weights: np.ndarray, cap: float) -> np.ndarray:
    work = pd.DataFrame({"query_index": frame["query_index"].to_numpy(), "weight": weights})
    totals = work.groupby("query_index", sort=False)["weight"].transform("sum").to_numpy(np.float64)
    scale = np.minimum(1.0, cap / np.clip(totals, 1e-12, None))
    return weights * scale


def _normalize_within_family(frame: pd.DataFrame, utility: np.ndarray) -> np.ndarray:
    """Use a common scale without changing within-family action ordering.

    Formula, identity and family equal exposure belongs in the stratified
    sampler.  Multiplying inverse abundance into utility can erase or reverse
    the conditional action signal that this protocol is designed to test.
    """
    output = utility.copy()
    families = frame["family"].astype(str).to_numpy()
    for family in sorted(set(families)):
        mask = families == family
        positive = utility[mask] > 0
        if not np.any(positive):
            output[mask] = 0.0
            continue
        scale = float(np.mean(utility[mask][positive]))
        output[mask] = utility[mask] / max(scale, 1e-12)
    return output


def build_action_weights(
    frame: pd.DataFrame,
    mode: str,
    config: WeightConfig = WeightConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build static or dynamic target weights with identical exposure bounds.

    ``dynamic`` uses crossfit clean probability, lagged target-minus-control
    advantage and explicit risk.  ``static`` ignores those outcomes and gives
    all registered target actions equal raw mass.  Both then pass through the
    same within-family scale normalization and query/action caps.  Equal
    identity/formula/family exposure is a sampler invariant, not folded into
    action utility.
    """
    config.validate()
    work = _validate_weight_frame(frame)
    if mode not in {"dynamic", "static"}:
        raise ValueError("mode must be 'dynamic' or 'static'")
    tau = robust_temperature(work["lagged_advantage"], config.tau_floor)
    if mode == "dynamic":
        raw = (
            work["p_clean"].to_numpy(np.float64)
            * _sigmoid(work["lagged_advantage"].to_numpy(np.float64) / tau)
            * (1.0 - work["risk"].to_numpy(np.float64))
        )
    else:
        raw = np.ones(len(work), dtype=np.float64)

    # Preserve conditional utility.  Dataset-abundance correction is deferred
    # to the stratified sampler so it cannot reverse action ordering here.
    work["raw_utility"] = raw.astype(np.float32)
    weights = _normalize_within_family(work, raw)
    weights = np.minimum(weights, config.max_action_weight)
    weights = _cap_queries(work, weights, config.max_query_weight)

    work["weight"] = weights.astype(np.float32)
    family_rows = []
    for family, block in work.groupby("family", sort=True):
        values = block["weight"].to_numpy(np.float64)
        total = float(np.sum(values))
        ess = float(total * total / np.sum(values * values)) if np.any(values > 0) else 0.0
        family_rows.append({
            "family": str(family),
            "actions": int(len(block)),
            "positive_weight_actions": int(np.sum(values > 0)),
            "weight_sum": total,
            "weight_mean": float(np.mean(values)),
            "effective_sample_size": ess,
            "learnable": bool(ess >= config.minimum_family_ess),
        })
    query_totals = work.groupby("query_index", sort=False)["weight"].sum().to_numpy(np.float64)
    report: dict[str, object] = {
        "mode": mode,
        "actions": int(len(work)),
        "queries": int(work["query_index"].nunique()),
        "identities": int(work["identity"].nunique()),
        "formulas": int(work["formula"].nunique()),
        "families": family_rows,
        "tau": tau,
        "zero_weight_fraction": float(np.mean(weights == 0)),
        "maximum_action_weight": float(np.max(weights)),
        "maximum_query_weight": float(np.max(query_totals)),
        "all_family_ess_pass": bool(all(row["learnable"] for row in family_rows)),
        "exposure_equalization": "stratified sampler, not action utility",
    }
    tolerance = 2e-6
    if report["maximum_action_weight"] > config.max_action_weight + tolerance:
        raise RuntimeError("action weight cap failed")
    if report["maximum_query_weight"] > config.max_query_weight + tolerance:
        raise RuntimeError("query weight cap failed")
    return work, report


def validate_n_cells(frame: pd.DataFrame) -> None:
    required = {"selector", "attenuation", "step"}
    if missing := required - set(frame.columns):
        raise RuntimeError(f"N action table misses columns: {sorted(missing)}")
    observed = {
        (str(row.selector), round(float(row.attenuation), 2), int(row.step))
        for row in frame[["selector", "attenuation", "step"]].drop_duplicates().itertuples(index=False)
    }
    if observed != N_CELLS:
        raise RuntimeError(
            f"mature N cells drifted; missing={sorted(N_CELLS-observed)}, "
            f"unexpected={sorted(observed-N_CELLS)}"
        )


def assert_outer_formula_disjoint(frame: pd.DataFrame, outer_fold: int) -> None:
    required = {"formula", "formula_fold", "split"}
    if missing := required - set(frame.columns):
        raise RuntimeError(f"split table misses columns: {sorted(missing)}")
    train = set(frame.loc[frame["split"].eq("train"), "formula"].astype(str))
    held = set(frame.loc[frame["split"].eq("held"), "formula"].astype(str))
    if train & held:
        raise RuntimeError("outer train/held formula overlap is nonzero")
    observed_held = set(frame.loc[frame["formula_fold"].astype(int).eq(outer_fold), "formula"].astype(str))
    if observed_held != held:
        raise RuntimeError("held split does not exactly equal the requested formula fold")


def formula_equal_weights(formulas: Iterable[str]) -> np.ndarray:
    """Give every formula equal total mass while retaining every observation."""
    values = pd.Series(list(map(str, formulas)), dtype=str)
    if values.empty:
        raise ValueError("formula weights require observations")
    counts = values.groupby(values, sort=False).transform("size").to_numpy(np.float64)
    weights = 1.0 / counts
    return weights / float(np.mean(weights))


def stratified_action_epoch(
    frame: pd.DataFrame,
    weight_column: str,
    seed: int,
    actions_per_identity_family: int,
) -> pd.DataFrame:
    """Bounded, without-replacement action schedule.

    Each identity contributes at most K actions to each family.  Formula and
    family quotas are interleaved deterministically; rows are never recycled
    within an epoch.  Action utility changes selection probability inside a
    stratum but cannot create duplicate exposure.
    """
    required = {"action_id", "identity", "formula", "family", weight_column}
    if missing := required - set(frame.columns):
        raise RuntimeError(f"sampler frame misses columns: {sorted(missing)}")
    if actions_per_identity_family < 1:
        raise ValueError("actions_per_identity_family must be positive")
    if frame["action_id"].astype(str).duplicated().any():
        raise RuntimeError("sampler action ids must be unique")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for (_, _), block in frame.groupby(["family", "identity"], sort=True):
        indices = block.index.to_numpy()
        weights = np.clip(block[weight_column].to_numpy(np.float64), 0.0, None)
        positive = weights > 0
        indices = indices[positive]
        weights = weights[positive]
        if not len(indices):
            continue
        take = min(actions_per_identity_family, len(indices))
        # Schedule membership is arm-invariant. Dynamic utility is applied by
        # the loss, never a second time through sampling probability.
        chosen = rng.choice(indices, size=take, replace=False)
        selected.extend(map(int, chosen))
    output = frame.loc[selected].copy()
    if output["action_id"].duplicated().any():
        raise RuntimeError("within-epoch action recycling detected")
    # Interleave formula/family queues so large groups cannot form long runs.
    queues: dict[tuple[str, str], list[int]] = {}
    for key, block in output.groupby(["formula", "family"], sort=True):
        order = rng.permutation(block.index.to_numpy())
        queues[(str(key[0]), str(key[1]))] = list(map(int, order))
    schedule: list[int] = []
    keys = sorted(queues)
    while keys:
        next_keys: list[tuple[str, str]] = []
        for key in np.asarray(keys, dtype=object)[rng.permutation(len(keys))].tolist():
            key_tuple = tuple(key) if isinstance(key, list) else key
            schedule.append(queues[key_tuple].pop())
            if queues[key_tuple]:
                next_keys.append(key_tuple)
        keys = sorted(next_keys)
    return output.loc[schedule].reset_index(drop=True)

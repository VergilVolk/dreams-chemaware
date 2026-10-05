"""Pure routing and schedule helpers for the minimal E4-PMT experiment."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


MATURE_N_CELLS = {
    *(f"candidate_gradient|a=0.50|step={step}" for step in range(3, 7)),
    *(f"role_confounder|a=1.00|step={step}" for step in range(1, 6)),
}


def select_materialized_best_action_union(
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one maximum-margin strict Top-1 corrective action per query."""
    required = {
        "action_id", "query_index", "source", "family", "recipe_id",
        "supervision_kind", "clean_rank", "action_rank", "action_margin",
    }
    if missing := required - set(actions.columns):
        raise KeyError(f"materialized action table misses {sorted(missing)}")
    clean_rank = pd.to_numeric(actions["clean_rank"], errors="raise").to_numpy(
        np.float64,
    )
    action_rank = pd.to_numeric(actions["action_rank"], errors="raise").to_numpy(
        np.float64,
    )
    if (
        not np.isfinite(clean_rank).all()
        or not np.isfinite(action_rank).all()
        or not np.equal(clean_rank, np.floor(clean_rank)).all()
        or not np.equal(action_rank, np.floor(action_rank)).all()
        or np.any(clean_rank < 1)
        or np.any(action_rank < 1)
    ):
        raise RuntimeError("materialized clean/action ranks must be finite positive integers")
    strict = actions.loc[
        actions["supervision_kind"].astype(str).eq("corrective")
        & pd.Series(clean_rank, index=actions.index).ne(1)
        & pd.Series(action_rank, index=actions.index).eq(1)
    ].copy()
    if strict.empty or strict["action_id"].duplicated().any():
        raise RuntimeError("strict Top-1 corrective action set is empty or duplicated")
    strict["_action_margin"] = pd.to_numeric(
        strict["action_margin"], errors="raise",
    )
    if not np.isfinite(strict["_action_margin"].to_numpy(np.float64)).all():
        raise RuntimeError("strict Top-1 corrective action margin is non-finite")
    strict = strict.sort_values(
        ["query_index", "_action_margin", "source", "family", "recipe_id", "action_id"],
        ascending=[True, False, True, True, True, True],
        kind="stable",
    )
    best = strict.drop_duplicates("query_index", keep="first").drop(
        columns=["_action_margin"],
    ).copy()
    if best.empty or best["query_index"].duplicated().any():
        raise RuntimeError("best-action union is empty or repeats a query")
    return best, strict


def select_materialized_direct_action_panel(
    actions: pd.DataFrame,
    *,
    mode: str,
    margin_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the materialized actions that enter direct shared training.

    ``one_best_e4`` preserves the historical one-action-per-query experiment.
    ``multi_action_balanced`` retains every numerically robust strict Top-1
    corrective action.  The latter deliberately separates *action-oracle
    selection* from *training coverage*: a maximum-margin action is sufficient
    to count a corrected query, but it is not sufficient to represent all of
    the source/family/path invariances that the clean encoder must learn.
    """
    if mode not in {"one_best_e4", "multi_action_balanced"}:
        raise ValueError(f"unknown materialized direct mode: {mode}")
    if not np.isfinite(margin_floor) or margin_floor < 0:
        raise ValueError("materialized action margin floor must be finite and non-negative")
    best, strict = select_materialized_best_action_union(actions)
    selected = best if mode == "one_best_e4" else strict
    margin = pd.to_numeric(selected["action_margin"], errors="raise")
    selected = selected.loc[margin.gt(float(margin_floor))].copy()
    if selected.empty or selected["action_id"].duplicated().any():
        raise RuntimeError("materialized direct action panel is empty or duplicated")
    if mode == "one_best_e4" and selected["query_index"].duplicated().any():
        raise RuntimeError("one-best materialized panel repeats a query")
    return selected.reset_index(drop=True), strict.reset_index(drop=True)


def identity_family_balanced_action_weights(
    actions: pd.DataFrame,
    *,
    total_weight_per_identity: float,
) -> np.ndarray:
    """Give every identity and every source/family equal total direct dose.

    All physical actions remain present.  Within an identity, total dose is
    split equally across observed ``(source, family)`` groups and then equally
    across the actions in that group.  Consequently copying a recipe or adding
    many actions from one large family cannot increase either query-family or
    identity dose.
    """
    required = {"query_ik14", "source", "family", "action_id"}
    if missing := required - set(actions.columns):
        raise KeyError(f"balanced materialized panel misses {sorted(missing)}")
    if (
        actions.empty
        or actions["action_id"].duplicated().any()
        or not np.isfinite(total_weight_per_identity)
        or total_weight_per_identity <= 0
    ):
        raise ValueError("balanced materialized action weights are underspecified")
    frame = actions.reset_index(drop=True)
    weights = np.zeros(len(frame), dtype=np.float64)
    identity = frame["query_ik14"].astype(str)
    source = frame["source"].astype(str)
    family = frame["family"].astype(str)
    for identity_value, identity_index in frame.groupby(identity, sort=True).groups.items():
        positions = np.asarray(list(identity_index), dtype=np.int64)
        keys = np.asarray([
            f"{source.iloc[position]}|{family.iloc[position]}" for position in positions
        ], dtype=object)
        groups = sorted(set(map(str, keys)))
        family_weight = float(total_weight_per_identity) / len(groups)
        for key in groups:
            local = positions[keys == key]
            weights[local] = family_weight / len(local)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("balanced materialized action weights are non-positive")
    observed = pd.Series(weights).groupby(identity.reset_index(drop=True)).sum()
    if not np.allclose(
        observed.to_numpy(np.float64), float(total_weight_per_identity),
        rtol=0.0, atol=1e-10,
    ):
        raise RuntimeError("balanced materialized identity dose drifted")
    return weights.astype(np.float32)


def identity_family_balanced_exposure_weights(
    actions: pd.DataFrame,
    exposure_indices: list[int],
    *,
    total_weight_per_identity: float,
) -> np.ndarray:
    """Balance effective dose after coverage-first physical scheduling.

    Every action must occur at least once. Under-covered identities may recycle
    actions to preserve E4's number of optimizer contacts. A static per-action
    weight is then solved from the actual exposure multiplicities so each
    identity receives ``total_weight_per_identity`` and each source/family
    observed for that identity receives an equal share of that total.
    """
    required = {"query_ik14", "source", "family", "action_id"}
    if missing := required - set(actions.columns):
        raise KeyError(f"balanced exposure panel misses {sorted(missing)}")
    frame = actions.reset_index(drop=True)
    n = len(frame)
    exposure = np.asarray(exposure_indices, dtype=np.int64)
    if (
        n == 0
        or exposure.ndim != 1
        or not len(exposure)
        or np.any((exposure < 0) | (exposure >= n))
        or not np.isfinite(total_weight_per_identity)
        or total_weight_per_identity <= 0
    ):
        raise ValueError("balanced exposure weights are underspecified")
    multiplicity = np.bincount(exposure, minlength=n).astype(np.float64)
    if np.any(multiplicity < 1):
        raise RuntimeError("an action was not physically exposed before weighting")
    weights = np.zeros(n, dtype=np.float64)
    identity = frame["query_ik14"].astype(str).reset_index(drop=True)
    source = frame["source"].astype(str).reset_index(drop=True)
    family = frame["family"].astype(str).reset_index(drop=True)
    for _, identity_index in frame.groupby(identity, sort=True).groups.items():
        positions = np.asarray(list(identity_index), dtype=np.int64)
        keys = np.asarray([
            f"{source.iloc[position]}|{family.iloc[position]}" for position in positions
        ], dtype=object)
        groups = sorted(set(map(str, keys)))
        target_group_dose = float(total_weight_per_identity) / len(groups)
        for key in groups:
            local = positions[keys == key]
            physical_exposures = float(np.sum(multiplicity[local]))
            weights[local] = target_group_dose / physical_exposures
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("balanced exposure weights are non-positive")
    effective = weights * multiplicity
    identity_dose = pd.Series(effective).groupby(identity).sum()
    if not np.allclose(
        identity_dose.to_numpy(np.float64), float(total_weight_per_identity),
        rtol=0.0, atol=1e-10,
    ):
        raise RuntimeError("balanced scheduled identity dose drifted")
    for _, identity_index in frame.groupby(identity, sort=True).groups.items():
        positions = np.asarray(list(identity_index), dtype=np.int64)
        group_dose = pd.Series(effective[positions]).groupby(pd.MultiIndex.from_arrays([
            source.iloc[positions].to_numpy(), family.iloc[positions].to_numpy(),
        ])).sum()
        if not np.allclose(
            group_dose.to_numpy(np.float64), group_dose.iloc[0],
            rtol=0.0, atol=1e-10,
        ):
            raise RuntimeError("balanced scheduled source/family dose drifted")
    return weights.astype(np.float32)


def strict_bool(values: pd.Series, name: str) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{name} is not a strict boolean column")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def route_replayed_actions(
    frame: pd.DataFrame, advantage_threshold: float = 0.01,
) -> pd.DataFrame:
    """Assign exactly one route; harm has precedence over apparent benefit."""
    required = {
        "paired_advantage", "clean_rank", "target_rank", "control_rank",
        "corrected", "introduced", "cell_id",
    }
    if missing := required - set(frame.columns):
        raise RuntimeError(f"PMT replay lacks columns: {sorted(missing)}")
    if advantage_threshold <= 0:
        raise ValueError("advantage threshold must be positive")
    output = frame.copy()
    advantage = output["paired_advantage"].to_numpy(np.float64)
    introduced = strict_bool(output["introduced"], "introduced")
    harmful = introduced | (advantage <= -advantage_threshold)
    corrective = (~harmful) & (advantage >= advantage_threshold)
    # Written with NumPy masks so the four routes are explicit and auditable.
    robust = (
        (~harmful) & (~corrective)
        & (output["clean_rank"].to_numpy(int) == 1)
        & (output["target_rank"].to_numpy(int) == 1)
        & (output["control_rank"].to_numpy(int) == 1)
    )
    uncertain = ~(harmful | corrective | robust)
    route = np.full(len(output), "uncertain", dtype=object)
    route[robust] = "robustness_only"
    route[corrective] = "corrective"
    route[harmful] = "harmful"
    output["route"] = route
    output["corrective_weight"] = corrective.astype(np.float32)
    output["teacher_advantage"] = np.where(corrective, advantage, 0.0).astype(np.float32)
    membership = np.column_stack((harmful, corrective, robust, uncertain)).sum(axis=1)
    if not np.all(membership == 1):
        raise RuntimeError("PMT routes are not mutually exclusive and exhaustive")
    if np.any(output.loc[output["route"] != "corrective", "corrective_weight"] != 0):
        raise RuntimeError("non-corrective action received target weight")
    observed_cells = set(output["cell_id"].astype(str))
    if observed_cells != MATURE_N_CELLS:
        raise RuntimeError(
            f"PMT requires exactly nine mature N cells; observed={sorted(observed_cells)}"
        )
    return output


def restrict_corrective_query_scope(
    routed: pd.DataFrame, scope: str,
) -> pd.DataFrame:
    """Apply target-exposure scope to the published all-routed table itself."""
    if scope not in {"all", "errors"}:
        raise ValueError(f"unknown corrective query scope: {scope}")
    output = routed.copy()
    if scope == "all":
        return output
    required = {"route", "clean_rank", "target_rank", "control_rank",
                "corrective_weight", "teacher_advantage"}
    if missing := required - set(output.columns):
        raise RuntimeError(f"routed actions lack scope columns: {sorted(missing)}")
    demote = output["route"].eq("corrective") & output["clean_rank"].astype(int).eq(1)
    robust = (
        demote
        & output["target_rank"].astype(int).eq(1)
        & output["control_rank"].astype(int).eq(1)
    )
    output.loc[demote, "route"] = "uncertain"
    output.loc[robust, "route"] = "robustness_only"
    output.loc[demote, "corrective_weight"] = np.float32(0.0)
    output.loc[demote, "teacher_advantage"] = np.float32(0.0)
    if not output.loc[~output["route"].eq("corrective"), "corrective_weight"].eq(0).all():
        raise RuntimeError("corrective query scope left a non-corrective positive weight")
    if output.loc[output["route"].eq("corrective"), "clean_rank"].astype(int).eq(1).any():
        raise RuntimeError("errors-only corrective scope retained a clean-correct query")
    return output


def coverage_first_schedules(
    action_ids: list[str], identities: list[str], policies: list[str],
    epochs: int, views_per_identity: int, seed: int,
) -> list[list[int]]:
    """Cover every action before recycling, then fill the historical E4 budget.

    The returned indices are deterministic. The first ``n_actions`` positions
    are a permutation without replacement, stratified round-robin by identity
    and policy. Repeats can only occur after that prefix.
    """
    n = len(action_ids)
    if not (n == len(identities) == len(policies)) or n == 0:
        raise ValueError("coverage schedule inputs must be equally sized and non-empty")
    if len(set(action_ids)) != n:
        raise RuntimeError("action ids must be unique")
    if epochs < 1 or views_per_identity < 1:
        raise ValueError("epochs and views-per-identity must be positive")
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (identity, policy) in enumerate(zip(identities, policies)):
        buckets[(str(identity), str(policy))].append(index)
    for key in buckets:
        values = np.asarray(buckets[key], dtype=np.int64)
        buckets[key] = values[rng.permutation(len(values))].tolist()
    keys = sorted(buckets)
    key_order = np.asarray(keys, dtype=object)[rng.permutation(len(keys))].tolist()
    unique_order: list[int] = []
    while len(unique_order) < n:
        for raw_key in key_order:
            key = tuple(raw_key) if not isinstance(raw_key, tuple) else raw_key
            if buckets[key]:
                unique_order.append(buckets[key].pop())
    identities_count = len(set(map(str, identities)))
    historical_budget = epochs * identities_count * views_per_identity
    total = max(n, historical_budget)
    order = list(unique_order)
    while len(order) < total:
        cycle = np.asarray(unique_order, dtype=np.int64)
        order.extend(cycle[rng.permutation(n)].tolist()[: total - len(order)])
    if len(set(order[:n])) != n:
        raise RuntimeError("PMT schedule recycled before complete coverage")
    sizes = [total // epochs + int(epoch < total % epochs) for epoch in range(epochs)]
    schedules: list[list[int]] = []
    left = 0
    for size in sizes:
        schedules.append(order[left:left + size])
        left += size
    if left != total:
        raise RuntimeError("PMT schedule partition is incomplete")
    return schedules


def coverage_first_identity_balanced_schedules(
    action_ids: list[str], identities: list[str], policies: list[str],
    epochs: int, views_per_identity: int, seed: int,
) -> list[list[int]]:
    """Cover every action, then restore E4's equal identity exposure budget.

    Every action appears exactly once in the global prefix.  Only after that
    prefix do repeats fill each identity to ``epochs * views_per_identity``.
    An identity with more unique actions than that budget is never truncated.
    """
    n = len(action_ids)
    if not (n == len(identities) == len(policies)) or n == 0:
        raise ValueError("identity-balanced schedule inputs must be equally sized and non-empty")
    if len(set(action_ids)) != n:
        raise RuntimeError("action ids must be unique")
    if epochs < 1 or views_per_identity < 1:
        raise ValueError("epochs and views-per-identity must be positive")
    rng = np.random.default_rng(seed)
    by_identity_policy: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (identity, policy) in enumerate(zip(identities, policies)):
        by_identity_policy[str(identity)][str(policy)].append(index)

    identity_order = sorted(by_identity_policy)
    identity_order = np.asarray(identity_order, dtype=object)[
        rng.permutation(len(identity_order))
    ].tolist()
    unique_by_identity: dict[str, list[int]] = {}
    for identity in identity_order:
        policy_buckets = by_identity_policy[identity]
        policy_order = sorted(policy_buckets)
        policy_order = np.asarray(policy_order, dtype=object)[
            rng.permutation(len(policy_order))
        ].tolist()
        for policy in policy_order:
            values = np.asarray(policy_buckets[policy], dtype=np.int64)
            policy_buckets[policy] = values[rng.permutation(len(values))].tolist()
        ordered: list[int] = []
        identity_action_count = sum(len(values) for values in policy_buckets.values())
        while len(ordered) < identity_action_count:
            for policy in policy_order:
                if policy_buckets[policy]:
                    ordered.append(policy_buckets[policy].pop())
        unique_by_identity[identity] = ordered

    # Round-robin identities so the no-replacement prefix is not dominated by
    # molecules with many query spectra.
    remaining = {identity: list(values) for identity, values in unique_by_identity.items()}
    unique_order: list[int] = []
    while len(unique_order) < n:
        for identity in identity_order:
            if remaining[identity]:
                unique_order.append(remaining[identity].pop(0))
    if set(unique_order) != set(range(n)) or len(unique_order) != n:
        raise RuntimeError("identity-balanced scheduler failed complete action coverage")

    historical_budget = epochs * views_per_identity
    filler: dict[str, list[int]] = {}
    targets: dict[str, int] = {}
    for identity in identity_order:
        unique = unique_by_identity[identity]
        targets[identity] = max(len(unique), historical_budget)
        deficit = targets[identity] - len(unique)
        values: list[int] = []
        while len(values) < deficit:
            cycle = np.asarray(unique, dtype=np.int64)
            values.extend(cycle[rng.permutation(len(cycle))].tolist()[: deficit - len(values)])
        filler[identity] = values
    filler_order: list[int] = []
    while any(filler.values()):
        for identity in identity_order:
            if filler[identity]:
                filler_order.append(filler[identity].pop(0))
    order = unique_order + filler_order
    observed = {
        identity: sum(str(identities[index]) == identity for index in order)
        for identity in identity_order
    }
    if observed != targets or len(set(order[:n])) != n:
        raise RuntimeError("identity-balanced exposure budget drifted")

    sizes = [len(order) // epochs + int(epoch < len(order) % epochs) for epoch in range(epochs)]
    schedules: list[list[int]] = []
    left = 0
    for size in sizes:
        schedules.append(order[left:left + size])
        left += size
    if left != len(order):
        raise RuntimeError("identity-balanced schedule partition is incomplete")
    return schedules

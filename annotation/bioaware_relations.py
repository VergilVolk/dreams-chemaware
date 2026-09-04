"""Typed, audit-friendly biochemical relation utilities.

The functions in this module never score a spectrum and never use phenotype
labels.  Their only purpose is to turn an offline reaction-participant table
into typed molecule-pair supervision while removing reactions that do not
change molecular identity after currency compounds are excluded.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_PARTICIPANT_COLUMNS = {
    "compound_id", "reaction_id", "side", "stoichiometry",
    "direction_semantics", "is_currency",
}


def _normalise_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def noncurrency_signature(group: pd.DataFrame, side: str) -> tuple[tuple[str, float], ...]:
    """Return a stoichiometry-aware signature for one reaction side."""
    selected = group[
        (group["side"].astype(str) == side)
        & ~group["is_currency"].map(_normalise_bool)
    ]
    totals: dict[str, float] = defaultdict(float)
    for row in selected.itertuples(index=False):
        totals[str(row.compound_id)[:14]] += float(row.stoichiometry)
    return tuple(sorted((key, round(value, 8)) for key, value in totals.items()))


def identity_noop_reactions(participants: pd.DataFrame) -> set[str]:
    """Find transport/state records with identical noncurrency participants."""
    missing = REQUIRED_PARTICIPANT_COLUMNS - set(participants.columns)
    if missing:
        raise RuntimeError(f"participant table missing columns: {sorted(missing)}")
    output: set[str] = set()
    for reaction, group in participants.groupby("reaction_id", sort=False):
        left = noncurrency_signature(group, "left")
        right = noncurrency_signature(group, "right")
        if left and left == right:
            output.add(str(reaction))
    return output


def _direction_label(semantics: str, source_side: str, target_side: str) -> str:
    value = semantics.strip().lower()
    reversible = any(token in value for token in ("bidirectional", "reversible", "equilibrium"))
    if reversible:
        return "reaction_bidirectional"
    physiological_lr = (
        ("physiological_lr" in value or "reactome_consensus_lr" in value)
        and "not_physiological" not in value
    )
    physiological_rl = (
        ("physiological_rl" in value or "reactome_consensus_rl" in value)
        and "not_physiological" not in value
    )
    if physiological_lr:
        return "reaction_forward" if source_side == "left" else "reaction_reverse"
    if physiological_rl:
        return "reaction_forward" if source_side == "right" else "reaction_reverse"
    return "reaction_direction_unknown"


@dataclass(frozen=True)
class TypedReactionPair:
    identity_a: str
    identity_b: str
    relation_type: str
    reaction_ids: tuple[str, ...]
    evidence_count: int


def typed_reaction_pairs(
    participants: pd.DataFrame,
    eligible_identities: Iterable[str],
) -> tuple[list[TypedReactionPair], dict[str, int]]:
    """Build typed cross-side identity pairs from valid Rhea reactions.

    Pair orientation is retained for curated physiological direction.  Unknown
    and bidirectional pairs are canonicalised lexicographically.  Multiple
    reactions supporting the same typed pair are aggregated.
    """
    missing = REQUIRED_PARTICIPANT_COLUMNS - set(participants.columns)
    if missing:
        raise RuntimeError(f"participant table missing columns: {sorted(missing)}")
    eligible = {str(value)[:14] for value in eligible_identities}
    noop = identity_noop_reactions(participants)
    aggregate: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    reaction_count = 0
    for reaction, group in participants.groupby("reaction_id", sort=False):
        reaction = str(reaction)
        if reaction in noop:
            continue
        left = sorted({
            str(value)[:14] for value in group.loc[
                (group["side"].astype(str) == "left")
                & ~group["is_currency"].map(_normalise_bool), "compound_id"
            ] if str(value)[:14] in eligible
        })
        right = sorted({
            str(value)[:14] for value in group.loc[
                (group["side"].astype(str) == "right")
                & ~group["is_currency"].map(_normalise_bool), "compound_id"
            ] if str(value)[:14] in eligible
        })
        if not left or not right:
            continue
        semantics = str(group["direction_semantics"].iloc[0])
        reaction_count += 1
        for source in left:
            for target in right:
                if source == target:
                    continue
                relation = _direction_label(semantics, "left", "right")
                identity_a, identity_b = source, target
                if relation in {"reaction_bidirectional", "reaction_direction_unknown"}:
                    identity_a, identity_b = sorted((source, target))
                aggregate[(identity_a, identity_b, relation)].add(reaction)
    pairs = [
        TypedReactionPair(
            identity_a=left,
            identity_b=right,
            relation_type=relation,
            reaction_ids=tuple(sorted(reactions)),
            evidence_count=len(reactions),
        )
        for (left, right, relation), reactions in sorted(aggregate.items())
    ]
    return pairs, {
        "identity_noop_reactions": len(noop),
        "eligible_reactions": reaction_count,
        "typed_identity_pairs": len(pairs),
    }

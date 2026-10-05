"""Truth-blind, sample-local atomic reaction events for BioAware B47-U3."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from annotation.bioaware import validate_reaction_participants
from annotation.bioaware_relations import _direction_label


PROTON_ADDUCT_MASS = 1.007276466621
SODIUM_ADDUCT_MASS = 22.989218
SUPPORTED_POSITIVE_ADDUCTS = {
    "[M+H]+": PROTON_ADDUCT_MASS,
    "[M+NA]+": SODIUM_ADDUCT_MASS,
}
MONOISOTOPIC_MASS = {
    "H": 1.00782503223, "B": 11.00930536, "C": 12.0,
    "N": 14.00307400443, "O": 15.99491461957, "F": 18.99840316273,
    "Na": 22.9897692820, "Mg": 23.985041697, "P": 30.97376199842,
    "S": 31.9720711744, "Cl": 34.968852682, "K": 38.9637064864,
    "Ca": 39.962590863, "Si": 27.97692653465, "Fe": 55.93493633, "Br": 78.9183376,
    "I": 126.904468,
}
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
# Rhea encodes charge as a terminal sign, optionally followed by magnitude
# (e.g. ``C5H9O2-`` or ``C21H25N7O17P3-3``). Digits before the sign belong to
# the final element count and must never be stripped.
FORMULA_CHARGE_SUFFIX = re.compile(r"[+-]\d*$")


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "t", "yes", "y"}


def normalize_ik14(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.upper().str[:14]
    return values.where(values.str.fullmatch(r"[A-Z]{14}"), "")


def neutral_mass_from_mz(mz: pd.Series, adduct: pd.Series) -> pd.Series:
    """Convert the two adducts admitted by the frozen B47 graph to neutral mass."""

    shifts = adduct.fillna("").astype(str).str.upper().map(SUPPORTED_POSITIVE_ADDUCTS)
    return pd.to_numeric(mz, errors="coerce") - shifts


def formula_mass(formula: object) -> float:
    text = str(formula).strip()
    if not text or text.casefold() == "nan":
        return float("nan")
    position = 0
    total = 0.0
    for match in FORMULA_TOKEN.finditer(text):
        if match.start() != position or match.group(1) not in MONOISOTOPIC_MASS:
            return float("nan")
        count = int(match.group(2)) if match.group(2) else 1
        if count <= 0:
            return float("nan")
        total += MONOISOTOPIC_MASS[match.group(1)] * count
        position = match.end()
    return total if position == len(text) and position > 0 else float("nan")


def formula_composition(formula: object) -> dict[str, int] | None:
    """Parse an elemental formula while ignoring terminal ionic charge.

    Rhea participants frequently use formulas such as ``C5H9O2-`` whereas the
    spectral candidate is represented by a neutral formula.  Hydrogen and
    formal charge are acquisition-state dependent, so reaction specificity is
    frozen on the non-hydrogen elemental transformation.
    """

    text = str(formula).strip().replace(" ", "")
    if not text or text.casefold() == "nan":
        return None
    text = FORMULA_CHARGE_SUFFIX.sub("", text)
    position = 0
    output: dict[str, int] = {}
    for match in FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            return None
        count = int(match.group(2)) if match.group(2) else 1
        if count <= 0:
            return None
        output[match.group(1)] = output.get(match.group(1), 0) + count
        position = match.end()
    return output if position == len(text) and position > 0 else None


def transformation_signature(candidate_formula: object, seed_formula: object) -> str:
    """Return candidate-minus-seed non-hydrogen elemental delta."""

    candidate = formula_composition(candidate_formula)
    seed = formula_composition(seed_formula)
    if candidate is None or seed is None:
        return ""
    elements = sorted((set(candidate) | set(seed)) - {"H"})
    changes = [
        f"{element}:{candidate.get(element, 0) - seed.get(element, 0):+d}"
        for element in elements
        if candidate.get(element, 0) != seed.get(element, 0)
    ]
    return "|".join(changes) if changes else "NO_HEAVY_ELEMENT_CHANGE"


def _degree_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        np.log1p(pd.to_numeric(values, errors="coerce").fillna(0).astype(float)),
        bins=[-np.inf, math.log1p(2), math.log1p(8), math.log1p(32), np.inf],
        labels=False,
    )


def _score_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 0.85, 0.90, 0.95, np.inf],
        labels=False,
    )


def safe_participants(participants: pd.DataFrame) -> pd.DataFrame:
    frame = participants.copy()
    if "formula" not in frame:
        raise ValueError("reaction participants require endpoint formulas")
    if "is_currency" in frame:
        frame["is_currency"] = frame["is_currency"].map(parse_bool)
    frame = validate_reaction_participants(frame)
    frame["compound_id"] = normalize_ik14(frame["compound_id"])
    frame = frame[frame["compound_id"].ne("")].copy()
    if "stoichiometry" not in frame:
        frame["stoichiometry"] = 1.0
    frame["stoichiometry"] = pd.to_numeric(frame["stoichiometry"], errors="coerce").fillna(1.0)
    if "direction_semantics" not in frame:
        frame["direction_semantics"] = "unknown"
    return frame


def reaction_pair_table(
    participants: pd.DataFrame,
    seed_identities: set[str],
    candidate_identities: set[str],
    maximum_seed_degree: int = 250,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build changed-participant one-hop pairs without query outcomes.

    A participant present with the same stoichiometry on both sides is a
    catalyst/carrier, not a molecular transformation endpoint.  Pairing all
    cross-side participants would therefore manufacture many false edges.
    We retain only compounds with a non-zero net stoichiometric change.
    """

    frame = safe_participants(participants)
    noncurrency = frame[~frame["is_currency"].astype(bool)].copy()
    degree = frame.groupby("compound_id")["reaction_id"].nunique().astype(int)
    rows: list[dict] = []
    identity_noop = 0
    unchanged_participants_removed = 0
    reactions_used = 0
    for reaction_id, group in noncurrency.groupby("reaction_id", sort=False):
        stoichiometry = group.pivot_table(
            index="compound_id", columns="side", values="stoichiometry",
            aggfunc="sum", fill_value=0.0,
        )
        left_amount = stoichiometry.get("left", pd.Series(0.0, index=stoichiometry.index))
        right_amount = stoichiometry.get("right", pd.Series(0.0, index=stoichiometry.index))
        net = right_amount.astype(float) - left_amount.astype(float)
        unchanged_participants_removed += int(np.isclose(net.to_numpy(float), 0.0).sum())
        left = sorted(net.index[net < -1e-8].astype(str))
        right = sorted(net.index[net > 1e-8].astype(str))
        if not left or not right:
            identity_noop += 1
            continue
        semantics = str(group["direction_semantics"].iloc[0])
        participant_formula = (
            group.sort_values(["compound_id", "side"], kind="stable")
            .drop_duplicates(["compound_id", "side"])
            .set_index(["compound_id", "side"])["formula"]
            .astype(str)
            .to_dict()
        )
        reaction_size = len(set(left + right))
        reaction_weight = float(pd.to_numeric(group["reaction_weight"], errors="raise").min())
        used = False
        for seed_side, seeds, candidate_side, candidates in (
            ("left", left, "right", right),
            ("right", right, "left", left),
        ):
            for seed_identity in seeds:
                if seed_identity not in seed_identities:
                    continue
                seed_degree = int(degree.get(seed_identity, 0))
                if seed_degree <= 0 or seed_degree > maximum_seed_degree:
                    continue
                for candidate_identity in candidates:
                    if (
                        candidate_identity == seed_identity
                        or candidate_identity not in candidate_identities
                    ):
                        continue
                    candidate_degree = int(degree.get(candidate_identity, 0))
                    if candidate_degree <= 0:
                        continue
                    reaction_seed_formula = participant_formula.get(
                        (seed_identity, seed_side), ""
                    )
                    reaction_candidate_formula = participant_formula.get(
                        (candidate_identity, candidate_side), ""
                    )
                    rows.append({
                        "candidate_id": candidate_identity,
                        "seed_compound_id": seed_identity,
                        "reaction_id": str(reaction_id),
                        "seed_side": seed_side,
                        "candidate_side": candidate_side,
                        "direction_label": _direction_label(
                            semantics, seed_side, candidate_side
                        ),
                        "direction_semantics": semantics,
                        "seed_degree": seed_degree,
                        "candidate_degree": candidate_degree,
                        "reaction_size": reaction_size,
                        "reaction_weight": reaction_weight,
                        "reaction_seed_formula": reaction_seed_formula,
                        "reaction_candidate_formula": reaction_candidate_formula,
                        "reaction_transform_signature": transformation_signature(
                            reaction_candidate_formula, reaction_seed_formula,
                        ),
                    })
                    used = True
        reactions_used += int(used)
    columns = [
        "candidate_id", "seed_compound_id", "reaction_id", "seed_side",
        "candidate_side", "direction_label", "direction_semantics",
        "seed_degree", "candidate_degree", "reaction_size", "reaction_weight",
        "reaction_seed_formula", "reaction_candidate_formula",
        "reaction_transform_signature",
    ]
    pairs = pd.DataFrame(rows, columns=columns).drop_duplicates(
        ["candidate_id", "seed_compound_id", "reaction_id", "seed_side", "candidate_side"]
    )
    return pairs, {
        "safe_participant_rows": int(len(noncurrency)),
        "identity_noop_reactions": int(identity_noop),
        "unchanged_participants_removed": int(unchanged_participants_removed),
        "reactions_used": int(reactions_used),
        "typed_pairs": int(len(pairs)),
    }


def materialize_sample_events(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    seeds: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    seed_reference_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join exact pairs to observed same-sample seed events and candidate sets."""

    candidate_required = {
        "query_id", "candidate_id", "candidate_formula", "spectral_score",
        "candidate_reference_mz", "best_reference_adduct",
    }
    query_required = {"query_id", "study", "sample", "feature_id", "feature_mz"}
    seed_required = {
        "study", "sample", "seed_query_id", "seed_feature_id",
        "seed_compound_id", "seed_score", "seed_margin",
    }
    for frame, required, name in (
        (candidates, candidate_required, "candidates"),
        (queries, query_required, "queries"),
        (seeds, seed_required, "seeds"),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("candidates must be unique by query and identity")
    q = queries[list(query_required)].copy()
    q["query_id"] = q["query_id"].astype(str)
    q["study"] = q["study"].astype(str)
    q["sample"] = q["sample"].astype(str)
    q["feature_mz"] = pd.to_numeric(q["feature_mz"], errors="raise")
    c = candidates.copy()
    c["query_id"] = c["query_id"].astype(str)
    c["candidate_id"] = normalize_ik14(c["candidate_id"])
    c = c.merge(q, on="query_id", how="left", validate="many_to_one")
    if c[["study", "sample"]].isna().any().any():
        raise RuntimeError("candidate lost query study/sample metadata")

    # A context-permutation null moves an intact seed event between samples.
    # Its query/identity/reference tuple must remain authentic, so seed lookup
    # may use the unpermuted candidate table while target candidates vary.
    reference_source = candidates if seed_reference_candidates is None else seed_reference_candidates
    reference_source = reference_source.copy()
    reference_source["query_id"] = reference_source["query_id"].astype(str)
    reference_source["candidate_id"] = normalize_ik14(reference_source["candidate_id"])
    reference_source = reference_source.merge(q, on="query_id", how="left", validate="many_to_one")
    seed_reference = reference_source[[
        "query_id", "candidate_id", "feature_mz", "candidate_reference_mz",
        "best_reference_adduct", "candidate_formula",
    ]].rename(columns={
        "query_id": "seed_query_id",
        "candidate_id": "seed_compound_id",
        "feature_mz": "seed_feature_mz",
        "candidate_reference_mz": "seed_reference_mz",
        "best_reference_adduct": "seed_reference_adduct",
        "candidate_formula": "seed_formula",
    })
    s = seeds.copy()
    for column in ("study", "sample", "seed_query_id"):
        s[column] = s[column].astype(str)
    s["seed_compound_id"] = normalize_ik14(s["seed_compound_id"])
    s = s.merge(
        seed_reference,
        on=["seed_query_id", "seed_compound_id"],
        how="left",
        validate="one_to_one",
    )
    if s[["seed_feature_mz", "seed_reference_mz"]].isna().any().any():
        raise RuntimeError("seed reference replay failed")

    events = c.merge(pairs, on="candidate_id", how="inner", validate="many_to_many")
    events = events.merge(
        s,
        on=["study", "sample", "seed_compound_id"],
        how="inner",
        validate="many_to_many",
    )
    events = events[
        events["query_id"].astype(str).ne(events["seed_query_id"].astype(str))
    ].copy()
    if events.empty:
        return events
    events = events.sort_values(
        ["query_id", "candidate_id", "seed_compound_id", "reaction_id", "seed_query_id"],
        kind="stable",
    ).drop_duplicates([
        "query_id", "candidate_id", "seed_compound_id", "reaction_id", "seed_query_id",
    ])

    events["candidate_neutral_mass"] = neutral_mass_from_mz(
        events["feature_mz"], events["best_reference_adduct"],
    )
    events["candidate_reference_neutral_mass"] = neutral_mass_from_mz(
        events["candidate_reference_mz"], events["best_reference_adduct"],
    )
    events["seed_neutral_mass"] = neutral_mass_from_mz(
        events["seed_feature_mz"], events["seed_reference_adduct"],
    )
    events["seed_reference_neutral_mass"] = neutral_mass_from_mz(
        events["seed_reference_mz"], events["seed_reference_adduct"],
    )
    events["observed_neutral_delta"] = (
        events["candidate_neutral_mass"] - events["seed_neutral_mass"]
    )
    events["candidate_formula_mass"] = events["candidate_formula"].map(formula_mass)
    events["seed_formula_mass"] = events["seed_formula"].map(formula_mass)
    events["expected_reaction_delta"] = (
        events["candidate_formula_mass"] - events["seed_formula_mass"]
    )
    events["observed_transform_signature"] = [
        transformation_signature(candidate_formula, seed_formula)
        for candidate_formula, seed_formula in zip(
            events["candidate_formula"], events["seed_formula"], strict=True,
        )
    ]
    events["reaction_transform_signature_match"] = (
        events["reaction_transform_signature"].astype(str).ne("")
        & events["observed_transform_signature"].astype(str).ne("")
        & events["reaction_transform_signature"].astype(str).eq(
            events["observed_transform_signature"].astype(str)
        )
    )
    events["candidate_formula_residual_da"] = (
        events["candidate_neutral_mass"] - events["candidate_formula_mass"]
    )
    events["candidate_reference_formula_residual_da"] = (
        events["candidate_reference_neutral_mass"] - events["candidate_formula_mass"]
    )
    events["seed_formula_residual_da"] = (
        events["seed_neutral_mass"] - events["seed_formula_mass"]
    )
    events["seed_reference_formula_residual_da"] = (
        events["seed_reference_neutral_mass"] - events["seed_formula_mass"]
    )
    events["transform_residual_da"] = (
        events["observed_neutral_delta"] - events["expected_reaction_delta"]
    )
    tolerance = 10e-6 * (
        events["candidate_neutral_mass"].abs()
        + events["seed_neutral_mass"].abs()
    )
    events["supported_adduct_pair"] = (
        events["candidate_neutral_mass"].notna()
        & events["candidate_reference_neutral_mass"].notna()
        & events["seed_neutral_mass"].notna()
        & events["seed_reference_neutral_mass"].notna()
    )
    candidate_tolerance = 10e-6 * events["candidate_formula_mass"].abs()
    seed_tolerance = 10e-6 * events["seed_formula_mass"].abs()
    events["identity_mass_contract_within_10ppm"] = (
        events["supported_adduct_pair"]
        & events["candidate_formula_mass"].notna()
        & events["seed_formula_mass"].notna()
        & (events["candidate_formula_residual_da"].abs() <= candidate_tolerance + 1e-12)
        & (
            events["candidate_reference_formula_residual_da"].abs()
            <= candidate_tolerance + 1e-12
        )
        & (events["seed_formula_residual_da"].abs() <= seed_tolerance + 1e-12)
        & (
            events["seed_reference_formula_residual_da"].abs()
            <= seed_tolerance + 1e-12
        )
    )
    events["transform_within_combined_10ppm"] = (
        events["identity_mass_contract_within_10ppm"]
        & events["reaction_transform_signature_match"]
        & events["expected_reaction_delta"].notna()
        & (events["transform_residual_da"].abs() <= tolerance + 1e-12)
    )
    events["event_eligible"] = events["transform_within_combined_10ppm"].astype(bool)

    # Candidate specificity is a property of the observed seed against the
    # complete candidate set.  Reaction IDs are deliberately excluded: one
    # seed supporting two candidates through two catalogue records is not two
    # exclusive pieces of evidence.
    competition_key = ["query_id", "seed_query_id", "seed_compound_id"]
    eligible = events[events["event_eligible"]].copy()
    competition = (
        eligible.groupby(competition_key)["candidate_id"].nunique()
        if len(eligible) else pd.Series(dtype=int)
    )
    event_keys = pd.MultiIndex.from_frame(events[competition_key])
    events["competing_query_candidate_count"] = (
        competition.reindex(event_keys).fillna(0).to_numpy(dtype=int)
    )
    events["candidate_specificity"] = np.where(
        events["competing_query_candidate_count"].gt(0),
        1.0 / events["competing_query_candidate_count"].clip(lower=1).astype(float),
        0.0,
    )
    degree_factor = (
        events["seed_degree"].clip(lower=1).astype(float)
        * events["candidate_degree"].clip(lower=1).astype(float)
    ) ** -0.25
    size_factor = 1.0 / np.sqrt(events["reaction_size"].clip(lower=2).astype(float) - 1.0)
    events["contribution"] = np.where(events["event_eligible"], np.clip(
        events["seed_score"].astype(float)
        * events["reaction_weight"].astype(float)
        * degree_factor * size_factor,
        0.0, 1.0,
    ), 0.0)
    events["specificity_weighted_contribution"] = (
        events["contribution"] * events["candidate_specificity"]
    )
    events["exclusive_event"] = (
        events["event_eligible"]
        & events["competing_query_candidate_count"].eq(1)
    )
    return events.reset_index(drop=True)


def _noisy_or(values: pd.Series) -> float:
    array = np.clip(values.to_numpy(float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - array)) if len(array) else 0.0


def aggregate_candidate_features(
    candidates: pd.DataFrame, events: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "query_id", "candidate_id", "candidate_formula", "spectral_score",
    ]
    for optional in (
        "reference_spectra", "best_reference_adduct", "candidate_catalogue_degree",
    ):
        if optional in candidates:
            base_columns.append(optional)
    base = candidates[base_columns].copy()
    if events.empty:
        for name in (
            "raw_event_count", "event_count", "exclusive_seed_count",
            "unique_seed_identities", "unique_reactions", "forward_event_count",
            "reverse_event_count", "unknown_direction_event_count",
        ):
            base[name] = 0
        for name in (
            "transform_consistent_fraction", "network_support",
            "candidate_specific_network_support",
        ):
            base[name] = 0.0
        return base
    raw_grouped = events.groupby(["query_id", "candidate_id"], sort=False)
    raw = raw_grouped.agg(
        raw_event_count=("reaction_id", "size"),
        transform_consistent_fraction=("event_eligible", "mean"),
    ).reset_index()
    eligible = events[events["event_eligible"]].copy()
    if eligible.empty:
        output = aggregate_candidate_features(candidates, events.iloc[0:0].copy())
        output = output.drop(columns=["raw_event_count", "transform_consistent_fraction"])
        output = output.merge(
            raw, on=["query_id", "candidate_id"], how="left", validate="one_to_one",
        ).fillna({"raw_event_count": 0, "transform_consistent_fraction": 0.0})
        output["raw_event_count"] = output["raw_event_count"].astype(int)
        output["transform_consistent_fraction"] = output["transform_consistent_fraction"].astype(float)
        return output
    grouped = eligible.groupby(["query_id", "candidate_id"], sort=False)
    feature = grouped.agg(
        event_count=("reaction_id", "size"),
        unique_seed_identities=("seed_compound_id", "nunique"),
        unique_reactions=("reaction_id", "nunique"),
        forward_event_count=(
            "direction_label", lambda x: int((x.astype(str) == "reaction_forward").sum())
        ),
        reverse_event_count=(
            "direction_label", lambda x: int((x.astype(str) == "reaction_reverse").sum())
        ),
        unknown_direction_event_count=(
            "direction_label",
            lambda x: int((x.astype(str) == "reaction_direction_unknown").sum()),
        ),
    ).reset_index()
    # Multiple Rhea records for the same observed seed are catalogue evidence,
    # not independent biological observations.  Collapse them before noisy-OR.
    independent_seed = eligible.groupby(
        ["query_id", "candidate_id", "seed_query_id", "seed_compound_id"],
        sort=False,
    ).agg(
        contribution=("contribution", "max"),
        specificity_weighted_contribution=("specificity_weighted_contribution", "max"),
        exclusive_event=("exclusive_event", "max"),
    ).reset_index()
    support = independent_seed.groupby(["query_id", "candidate_id"], sort=False).agg(
        network_support=("contribution", _noisy_or),
        candidate_specific_network_support=("specificity_weighted_contribution", _noisy_or),
        exclusive_seed_count=("exclusive_event", "sum"),
    ).reset_index()
    feature = feature.merge(support, on=["query_id", "candidate_id"], validate="one_to_one")
    feature = feature.merge(raw, on=["query_id", "candidate_id"], validate="one_to_one")
    output = base.merge(feature, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    integer_columns = (
        "raw_event_count", "event_count", "exclusive_seed_count",
        "unique_seed_identities", "unique_reactions", "forward_event_count",
        "reverse_event_count", "unknown_direction_event_count",
    )
    output[list(integer_columns)] = output[list(integer_columns)].fillna(0).astype(int)
    float_columns = (
        "transform_consistent_fraction", "network_support",
        "candidate_specific_network_support",
    )
    output[list(float_columns)] = output[list(float_columns)].fillna(0.0).astype(float)
    return output


def query_opportunities(features: pd.DataFrame, query_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    meta = query_metadata.set_index("query_id")
    for query_id, group in features.groupby("query_id", sort=False):
        if len(group) < 2:
            raise RuntimeError(f"U3 query has fewer than two candidates: {query_id}")
        scores = group["spectral_score"].to_numpy(float)
        maximum = float(scores.max())
        baseline_top = group[np.isclose(scores, maximum, atol=1e-12, rtol=0)]
        ordered_scores = np.sort(scores)[::-1]
        spectral_margin = float(ordered_scores[0] - ordered_scores[1])
        event_score = group["candidate_specific_network_support"].to_numpy(float)
        event_max = float(event_score.max())
        event_top = group[np.isclose(event_score, event_max, atol=1e-12, rtol=0)]
        event_available = event_max > 0
        event_unique = event_available and len(event_top) == 1
        second_event = float(np.sort(event_score)[-2])
        baseline_unique = len(baseline_top) == 1
        baseline_id = (
            str(baseline_top.iloc[0]["candidate_id"]) if baseline_unique else ""
        )
        event_id = str(event_top.sort_values("candidate_id").iloc[0]["candidate_id"])
        exclusive_candidates = int(group["exclusive_seed_count"].gt(0).sum())
        event_top_has_exclusive = bool(
            event_unique and int(event_top.iloc[0]["exclusive_seed_count"]) > 0
        )
        event_disagrees = bool(
            baseline_unique and event_unique and event_id != baseline_id
        )
        # Evidence that merely agrees with a unique spectral winner cannot
        # contribute a Recall@1 correction.
        intervention_opportunity = bool(
            event_top_has_exclusive
            and ((not baseline_unique) or event_id != baseline_id)
        )
        metadata = meta.loc[str(query_id)]
        rows.append({
            "query_id": str(query_id),
            "study": str(metadata["study"]),
            "sample": str(metadata["sample"]),
            "feature_id": str(metadata["feature_id"]),
            "candidate_count": int(len(group)),
            "baseline_top_candidate": baseline_id,
            "baseline_top_unique": bool(baseline_unique),
            "spectral_margin": spectral_margin,
            "event_available": bool(event_available),
            "event_top_candidate": event_id,
            "event_top_formula": (
                str(event_top.iloc[0]["candidate_formula"]) if event_unique else ""
            ),
            "event_top_unique": bool(event_unique),
            "event_advantage": float(event_max - second_event),
            "event_disagrees_with_baseline": event_disagrees,
            "event_resolves_baseline_tie": bool(
                (not baseline_unique) and event_top_has_exclusive
            ),
            "exclusive_supported_candidates": exclusive_candidates,
            "event_top_has_exclusive_support": event_top_has_exclusive,
            "candidate_specific_opportunity": event_top_has_exclusive,
            "candidate_specific_intervention_opportunity": intervention_opportunity,
            # These candidates are already in the frozen query candidate set,
            # but U3 has no eligible exact-catalogue event for them. Calling
            # them "mass-matched non-neighbours" overstates what is known.
            "unsupported_candidates_in_frozen_set": int(group["event_count"].eq(0).sum()),
            "catalogue_covered_candidates": int(
                group.get("candidate_catalogue_degree", pd.Series(0, index=group.index)).gt(0).sum()
            ),
            "event_top_reference_spectra": int(
                event_top.iloc[0].get("reference_spectra", 0)
            ) if event_unique else 0,
            "event_top_catalogue_degree": int(
                event_top.iloc[0].get("candidate_catalogue_degree", 0)
            ) if event_unique else 0,
        })
    return pd.DataFrame(rows)


def effective_count(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(float)
    if not len(counts) or counts.sum() <= 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(1.0 / np.square(probabilities).sum())


def summarize_opportunity(
    events: pd.DataFrame, opportunities: pd.DataFrame,
) -> dict[str, object]:
    by_study = {}
    for study, group in opportunities.groupby("study", sort=True):
        by_study[str(study)] = {
            "queries": int(len(group)),
            "event_supported_queries": int(group["event_available"].sum()),
            "candidate_specific_queries": int(group["candidate_specific_opportunity"].sum()),
            "candidate_specific_intervention_queries": int(
                group["candidate_specific_intervention_opportunity"].sum()
            ),
            "event_baseline_disagreements": int(group["event_disagrees_with_baseline"].sum()),
            "baseline_unique_queries": int(group["baseline_top_unique"].sum()),
        }
    eligible = events[events.get("event_eligible", False)].copy() if len(events) else events.copy()
    independent = (
        eligible.drop_duplicates(["query_id", "candidate_id", "seed_query_id", "seed_compound_id"])
        if len(eligible) else eligible
    )
    seed_counts = independent["seed_compound_id"].value_counts() if len(independent) else pd.Series(dtype=int)
    reaction_counts = eligible["reaction_id"].value_counts() if len(eligible) else pd.Series(dtype=int)
    actionable = opportunities[
        opportunities["candidate_specific_intervention_opportunity"].astype(bool)
    ].copy()
    action_candidate_counts = (
        actionable["event_top_candidate"].value_counts()
        if len(actionable) else pd.Series(dtype=int)
    )
    action_formula_counts = (
        actionable["event_top_formula"].value_counts()
        if len(actionable) else pd.Series(dtype=int)
    )
    return {
        "atomic_events": int(len(events)),
        "eligible_atomic_events": int(len(eligible)),
        "queries": int(len(opportunities)),
        "event_supported_queries": int(opportunities["event_available"].sum()),
        "candidate_specific_queries": int(opportunities["candidate_specific_opportunity"].sum()),
        "candidate_specific_intervention_queries": int(
            opportunities["candidate_specific_intervention_opportunity"].sum()
        ),
        "maximum_intervention_fraction": float(
            opportunities["candidate_specific_intervention_opportunity"].mean()
        ) if len(opportunities) else 0.0,
        "intervention_candidate_identities": int(
            actionable["event_top_candidate"].nunique()
        ) if len(actionable) else 0,
        "intervention_candidate_formulas": int(
            actionable["event_top_formula"].nunique()
        ) if len(actionable) else 0,
        "effective_intervention_candidate_identities": effective_count(
            actionable["event_top_candidate"]
        ) if len(actionable) else 0.0,
        "largest_intervention_candidate_fraction": float(
            action_candidate_counts.iloc[0] / len(actionable)
        ) if len(actionable) else 0.0,
        "largest_intervention_formula_fraction": float(
            action_formula_counts.iloc[0] / len(actionable)
        ) if len(actionable) else 0.0,
        "intervention_reference_spectra_median": float(
            actionable["event_top_reference_spectra"].median()
        ) if len(actionable) else 0.0,
        "intervention_reference_spectra_p90": float(
            actionable["event_top_reference_spectra"].quantile(0.9)
        ) if len(actionable) else 0.0,
        "intervention_catalogue_degree_median": float(
            actionable["event_top_catalogue_degree"].median()
        ) if len(actionable) else 0.0,
        "intervention_catalogue_degree_p90": float(
            actionable["event_top_catalogue_degree"].quantile(0.9)
        ) if len(actionable) else 0.0,
        "event_baseline_disagreements": int(opportunities["event_disagrees_with_baseline"].sum()),
        "queries_with_unsupported_candidates": int(
            opportunities["unsupported_candidates_in_frozen_set"].gt(0).sum()
        ),
        "seed_identities_used": int(eligible["seed_compound_id"].nunique()) if len(eligible) else 0,
        "reactions_used": int(eligible["reaction_id"].nunique()) if len(eligible) else 0,
        "candidate_identities_supported": int(eligible["candidate_id"].nunique()) if len(eligible) else 0,
        "effective_seed_identities": effective_count(independent["seed_compound_id"]) if len(independent) else 0.0,
        "effective_reactions": effective_count(eligible["reaction_id"]) if len(eligible) else 0.0,
        "largest_seed_event_fraction": float(seed_counts.iloc[0] / len(independent)) if len(independent) else 0.0,
        "largest_reaction_event_fraction": float(reaction_counts.iloc[0] / len(eligible)) if len(eligible) else 0.0,
        "exclusive_event_fraction": float(eligible["exclusive_event"].mean()) if len(eligible) else 0.0,
        "supported_adduct_pair_fraction": float(
            events["supported_adduct_pair"].mean()
        ) if len(events) else 0.0,
        "formula_mass_pair_fraction": float(
            (
                events["candidate_formula_mass"].notna()
                & events["seed_formula_mass"].notna()
            ).mean()
        ) if len(events) else 0.0,
        "identity_mass_contract_fraction": float(
            events["identity_mass_contract_within_10ppm"].mean()
        ) if len(events) else 0.0,
        "reaction_transform_signature_match_fraction": float(
            events["reaction_transform_signature_match"].mean()
        ) if len(events) else 0.0,
        "transform_consistent_fraction": float(
            events["event_eligible"].mean()
        ) if len(events) else 0.0,
        "direction_labels": (
            eligible["direction_label"].astype(str).value_counts().sort_index().astype(int).to_dict()
            if len(eligible) else {}
        ),
        "by_study": by_study,
    }


def permute_seed_events_across_samples(
    seeds: pd.DataFrame, degree: pd.Series, seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Move intact seed events between samples within matched study strata.

    Only permuting ``seed_compound_id`` breaks the frozen seed-query/reference
    identity and makes the null impossible to replay.  Here the entire seed
    payload moves while the target study/sample slots remain fixed.  This
    preserves per-sample seed counts and authentic spectrum--identity tuples,
    while breaking the biological sample context being tested.
    """

    required_payload = [
        "seed_query_id", "seed_feature_id", "seed_compound_id",
        "seed_score", "seed_margin",
    ]
    missing = set(["study", "sample", *required_payload]) - set(seeds)
    if missing:
        raise ValueError(f"seeds missing permutation columns: {sorted(missing)}")
    output = seeds.reset_index(drop=True).copy()
    output["_degree"] = normalize_ik14(output["seed_compound_id"]).map(degree).fillna(0)
    output["_degree_bin"] = _degree_bin(output["_degree"])
    output["_score_bin"] = _score_bin(output["seed_score"])
    original_sample = output["sample"].astype(str).copy()
    original_query = output["seed_query_id"].astype(str).copy()
    rng = np.random.default_rng(seed)
    moved_rows = 0
    eligible_rows = 0
    for _, labels in output.groupby(
        ["study", "_degree_bin", "_score_bin"], dropna=False, sort=False,
    ).groups.items():
        positions = np.asarray(list(labels), dtype=int)
        source_samples = original_sample.iloc[positions].to_numpy()
        if len(positions) < 2 or len(set(source_samples)) < 2:
            continue
        eligible_rows += len(positions)
        best_order = positions.copy()
        best_moved = -1
        for _ in range(64):
            order = rng.permutation(positions)
            moved = int(np.sum(original_sample.iloc[order].to_numpy() != source_samples))
            if moved > best_moved:
                best_order, best_moved = order, moved
            if moved == len(positions):
                break
        output.loc[positions, required_payload] = (
            output.loc[best_order, required_payload].to_numpy(copy=True)
        )
        moved_rows += max(0, best_moved)
    audit = {
        "rows": int(len(output)),
        "eligible_rows": int(eligible_rows),
        "moved_rows": int(moved_rows),
        "moved_fraction": float(moved_rows / len(output)) if len(output) else 0.0,
        "query_identity_payload_preserved": int(
            output["seed_query_id"].astype(str).nunique() == original_query.nunique()
        ),
    }
    return output.drop(columns=["_degree", "_degree_bin", "_score_bin"]), audit


def permute_candidate_identities_within_query_adduct(
    candidates: pd.DataFrame, seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build a mass/adduct-matched wrong-identity transformation control.

    Candidate identity/formula/degree are cyclically reassigned only among
    candidates from the same query and best-reference adduct.  The spectral
    score, reference multiplicity and precursor observation stay in place.
    Hence any reaction event produced by the permuted identity is a wrong
    identity-to-spectrum transformation under an otherwise matched candidate
    competition.
    """

    required = {
        "query_id", "candidate_id", "candidate_formula", "best_reference_adduct",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidates missing permutation columns: {sorted(missing)}")
    output = candidates.reset_index(drop=True).copy()
    payload = ["candidate_id", "candidate_formula"]
    if "candidate_catalogue_degree" in output:
        payload.append("candidate_catalogue_degree")
    rng = np.random.default_rng(seed)
    moved_rows = 0
    eligible_rows = 0
    eligible_queries: set[str] = set()
    for (query_id, _), labels in output.groupby(
        ["query_id", "best_reference_adduct"], dropna=False, sort=False,
    ).groups.items():
        positions = np.asarray(list(labels), dtype=int)
        if len(positions) < 2:
            continue
        eligible_rows += len(positions)
        eligible_queries.add(str(query_id))
        order = rng.permutation(positions)
        # A cyclic shift guarantees a derangement even when RNG returns the
        # original order or a permutation with fixed points.
        order = np.roll(order, 1)
        original_ids = output.loc[positions, "candidate_id"].astype(str).to_numpy()
        for _ in range(len(positions)):
            proposed = output.loc[order, "candidate_id"].astype(str).to_numpy()
            if np.all(proposed != original_ids):
                break
            order = np.roll(order, 1)
        if not np.all(output.loc[order, "candidate_id"].astype(str).to_numpy() != original_ids):
            order = np.roll(positions, 1)
        output.loc[positions, payload] = output.loc[order, payload].to_numpy(copy=True)
        moved_rows += len(positions)
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("candidate permutation created duplicate query identities")
    audit = {
        "rows": int(len(output)),
        "eligible_rows": int(eligible_rows),
        "eligible_queries": int(len(eligible_queries)),
        "moved_rows": int(moved_rows),
        "moved_fraction": float(moved_rows / len(output)) if len(output) else 0.0,
    }
    return output, audit

"""Truth-blind feature construction and inference for ChemAware candidates.

The functions in this module deliberately accept no truth labels, positive
candidate index, baseline correctness, or post-hoc correction outcomes.  They
are therefore safe to reuse at deployment.  Training and evaluation code must
attach labels only after this module has returned frozen candidate scores.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from chemaware_orthogonal_rule_policy_core import rotate_candidate_contrast_truthblind


STAT_NAMES = ("max", "second", "top2_mean", "mean", "median", "std", "q75", "max_gap")
CHANNELS = ("official", "mass", "rule")
DESCRIPTOR_NAMES = (
    ["log_reference_count"]
    + [f"{channel}_{stat}" for channel in CHANNELS for stat in STAT_NAMES]
    + ["corr_official_mass", "corr_official_rule", "corr_mass_rule"]
    + ["argmax_official_mass", "argmax_official_rule", "argmax_mass_rule"]
    + ["mass_at_official_best", "rule_at_official_best", "official_at_mass_best", "official_at_rule_best"]
)
ACTION_NAMES = [
    "action_top_fraction", "action_largest_region_fraction",
    "action_same_neighbor_fraction", "action_margin_max", "action_margin_mean",
    "action_min_total_beta", "action_max_total_beta",
    "action_best_advantage_over_baseline", "global_action_advantage_over_baseline",
    "global_action_selects_candidate",
]
CONTEXT_NAMES = [
    "log_candidate_count", "baseline_official_margin",
    "candidate_official_rank_fraction", "candidate_mass_rank_fraction",
    "candidate_rule_rank_fraction",
]
FEATURE_NAMES = (
    CONTEXT_NAMES
    + [f"candidate_{name}" for name in DESCRIPTOR_NAMES]
    + [f"delta_{name}" for name in DESCRIPTOR_NAMES]
    + ACTION_NAMES
)


def truthblind_scored_view(
    scored: Mapping[str, np.ndarray], rule_key: str,
) -> dict[str, np.ndarray]:
    """Project a labelled training ledger onto deployment-visible channels."""
    allowed = ("global", "mass", "reference_ptr", str(rule_key))
    missing = [key for key in allowed if key not in scored]
    if missing:
        raise KeyError(f"truth-blind scored channels absent: {missing}")
    return {key: scored[key] for key in allowed}


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    output = np.empty(len(order), dtype=np.float32)
    output[order] = np.arange(len(order), dtype=np.float32)
    return output / max(1, len(order) - 1)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if np.isfinite(result) else 0.0


def statistics(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)
    maximum = float(ordered[-1])
    second = float(ordered[-2]) if len(ordered) > 1 else maximum
    return [
        maximum, second, float(np.mean(ordered[-min(2, len(ordered)):])),
        float(np.mean(values)), float(np.median(values)), float(np.std(values)),
        float(np.quantile(values, 0.75)), maximum - second,
    ]


def reference_descriptor(official: np.ndarray, mass: np.ndarray, rule: np.ndarray) -> np.ndarray:
    values = [np.asarray(x, dtype=np.float64) for x in (official, mass, rule)]
    result: list[float] = [float(np.log1p(len(official)))]
    for channel in values:
        result.extend(statistics(channel))
    result.extend([
        safe_correlation(values[0], values[1]),
        safe_correlation(values[0], values[2]),
        safe_correlation(values[1], values[2]),
        float(np.argmax(values[0]) == np.argmax(values[1])),
        float(np.argmax(values[0]) == np.argmax(values[2])),
        float(np.argmax(values[1]) == np.argmax(values[2])),
        float(values[1][np.argmax(values[0])]),
        float(values[2][np.argmax(values[0])]),
        float(values[0][np.argmax(values[1])]),
        float(values[0][np.argmax(values[2])]),
    ])
    output = np.asarray(result, dtype=np.float32)
    if len(output) != len(DESCRIPTOR_NAMES):
        raise RuntimeError("reference descriptor schema drifted")
    return output


def action_topology(actions: Sequence[tuple[float, float]]) -> list[list[int]]:
    mass_values = sorted({x[0] for x in actions} | {0.0})
    rule_values = sorted({x[1] for x in actions} | {0.0})
    coordinate = {
        (mass_values.index(mass), rule_values.index(rule)): index
        for index, (mass, rule) in enumerate(actions)
    }
    neighbors: list[list[int]] = [[] for _ in actions]
    for (left, right), index in coordinate.items():
        for delta_left, delta_right in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = coordinate.get((left + delta_left, right + delta_right))
            if other is not None:
                neighbors[index].append(other)
    return neighbors


def connected_size(top: np.ndarray, candidate: int, neighbors: list[list[int]]) -> tuple[int, float]:
    remaining = set(np.flatnonzero(top == candidate).tolist())
    largest = 0
    supports: list[float] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            degree = len(neighbors[current])
            supports.append(sum(int(top[x]) == candidate for x in neighbors[current]) / max(1, degree))
            for other in neighbors[current]:
                if other in remaining and int(top[other]) == candidate:
                    remaining.remove(other)
                    stack.append(other)
        largest = max(largest, size)
    return largest, (float(np.mean(supports)) if supports else 0.0)


def _validate_scored(scored: Mapping[str, np.ndarray], rule_key: str) -> int:
    forbidden = {
        "labels", "old_rank", "truth", "truth_candidate", "true_candidate",
        "query_identity", "query_ik14", "corrected", "introduced",
    }
    leaked = sorted(forbidden & set(scored))
    if leaked:
        raise ValueError(f"truth fields reached deployment inference: {leaked}")
    required = ("global", "mass", "reference_ptr", rule_key)
    missing = [key for key in required if key not in scored]
    if missing:
        raise KeyError(f"truth-blind scored channels absent: {missing}")
    query_count = len(scored["reference_ptr"])
    if any(len(scored[key]) != query_count for key in required):
        raise ValueError("truth-blind scored channels have inconsistent query counts")
    return query_count


def build_truthblind_candidate_features(
    scored: Mapping[str, np.ndarray], actions: Sequence[tuple[float, float]],
    global_action: int, rule_key: str,
) -> dict[str, np.ndarray]:
    """Construct candidate features without accepting any ground-truth field."""
    query_count = _validate_scored(scored, rule_key)
    candidate_counts = [len(np.asarray(ptr)) - 1 for ptr in scored["reference_ptr"]]
    if not candidate_counts or min(candidate_counts) < 1:
        raise ValueError("every query must contain at least one candidate")
    maximum_challengers = max(candidate_counts) - 1
    feature = np.zeros((query_count, maximum_challengers, len(FEATURE_NAMES)), dtype=np.float32)
    proposed_candidate = np.full((query_count, maximum_challengers), -1, dtype=np.int16)
    valid = np.zeros((query_count, maximum_challengers), dtype=bool)
    baseline_candidate = np.empty(query_count, dtype=np.int16)
    neighbors = action_topology(actions)
    action_array = np.asarray(actions, dtype=np.float64)
    if not len(actions) or not (0 <= int(global_action) < len(actions)):
        raise ValueError("global action index is invalid")

    for query in range(query_count):
        pointer = np.asarray(scored["reference_ptr"][query], dtype=np.int64)
        if pointer.ndim != 1 or len(pointer) < 2 or pointer[0] != 0 or np.any(np.diff(pointer) <= 0):
            raise ValueError(f"invalid reference pointer for query {query}")
        pair_channels = [
            np.asarray(scored["global"][query], dtype=np.float32),
            np.asarray(scored["mass"][query], dtype=np.float32),
            np.asarray(scored[rule_key][query], dtype=np.float32),
        ]
        if any(len(channel) != int(pointer[-1]) for channel in pair_channels):
            raise ValueError(f"reference score length drift for query {query}")
        molecule_channels = [np.maximum.reduceat(channel, pointer[:-1]) for channel in pair_channels]
        official, mass, rule = molecule_channels
        baseline = int(np.argmax(official))
        baseline_candidate[query] = baseline
        descriptors = np.stack([
            reference_descriptor(*[
                channel[int(pointer[molecule]):int(pointer[molecule + 1])]
                for channel in pair_channels
            ])
            for molecule in range(len(pointer) - 1)
        ])
        baseline_descriptor = descriptors[baseline]
        channel_ranks = [ranks(channel) for channel in molecule_channels]
        combined_all = []
        top = np.empty(len(actions), dtype=np.int16)
        margin = np.empty(len(actions), dtype=np.float32)
        for index, (mass_beta, rule_beta) in enumerate(actions):
            pair = pair_channels[0] + mass_beta * pair_channels[1] + rule_beta * pair_channels[2]
            combined = np.maximum.reduceat(pair, pointer[:-1])
            combined_all.append(combined)
            order = np.sort(combined)
            top[index] = int(np.argmax(combined))
            margin[index] = float(order[-1] - order[-2]) if len(order) > 1 else 1.0
        combined_all = np.stack(combined_all)
        official_sorted = np.sort(official)
        baseline_margin = float(official_sorted[-1] - official_sorted[-2]) if len(official) > 1 else 1.0
        slot = 0
        for candidate in range(len(official)):
            if candidate == baseline:
                continue
            selected = top == candidate
            largest, neighbor_support = connected_size(top, candidate, neighbors)
            selected_totals = action_array[selected].sum(axis=1)
            selected_margins = margin[selected]
            advantage = combined_all[:, candidate] - combined_all[:, baseline]
            action_features = np.asarray([
                float(np.mean(selected)), largest / len(actions), neighbor_support,
                float(np.max(selected_margins)) if selected.any() else 0.0,
                float(np.mean(selected_margins)) if selected.any() else 0.0,
                float(np.min(selected_totals)) if selected.any() else float(np.max(action_array.sum(axis=1))),
                float(np.max(selected_totals)) if selected.any() else 0.0,
                float(np.max(advantage)), float(advantage[global_action]),
                float(top[global_action] == candidate),
            ], dtype=np.float32)
            context = np.asarray([
                float(np.log1p(len(official))), baseline_margin,
                channel_ranks[0][candidate], channel_ranks[1][candidate], channel_ranks[2][candidate],
            ], dtype=np.float32)
            feature[query, slot] = np.concatenate((
                context, descriptors[candidate], descriptors[candidate] - baseline_descriptor,
                action_features,
            ))
            proposed_candidate[query, slot] = candidate
            valid[query, slot] = True
            slot += 1
    return {
        "feature": feature,
        "proposed_candidate": proposed_candidate,
        "baseline_candidate": baseline_candidate,
        "valid": valid,
        "reference_ptr": np.asarray([
            np.asarray(pointer, dtype=np.int32) for pointer in scored["reference_ptr"]
        ], dtype=object),
    }


def rank_candidates_from_utility(
    table: Mapping[str, np.ndarray], utility: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected candidate index and abstention flag per query."""
    valid = np.asarray(table["valid"], dtype=bool)
    values = np.asarray(utility, dtype=np.float64)
    if values.shape != valid.shape:
        raise ValueError("utility/valid shapes differ")
    masked = np.where(valid, values, -np.inf)
    slot = np.argmax(masked, axis=1)
    best = masked[np.arange(len(masked)), slot]
    selected = np.asarray(table["proposed_candidate"])[np.arange(len(masked)), slot].astype(np.int16)
    abstained = (~np.isfinite(best)) | (best < float(threshold))
    selected[abstained] = np.asarray(table["baseline_candidate"], dtype=np.int16)[abstained]
    return selected, abstained


def _feature_indices(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    base = []
    chemical = []
    for index, name in enumerate(names):
        lowered = str(name).lower()
        (chemical if "rule" in lowered or "action" in lowered else base).append(index)
    if not base or not chemical:
        raise ValueError("both nuisance and chemical feature blocks are required")
    return np.asarray(base, dtype=np.int64), np.asarray(chemical, dtype=np.int64)


def _restore_prior(probability: np.ndarray, prevalence: float) -> np.ndarray:
    if not (0.0 < float(prevalence) < 1.0):
        raise ValueError("training prevalence must lie strictly between zero and one")
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    balanced_logit = np.log(probability / (1.0 - probability))
    prior_logit = np.log(float(prevalence) / (1.0 - float(prevalence)))
    return 1.0 / (1.0 + np.exp(-(balanced_logit + prior_logit)))


def _predict_channel(
    channel: Mapping[str, object], base: np.ndarray, contrast: np.ndarray,
    valid: np.ndarray, dose: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nuisance = np.zeros(valid.shape, dtype=np.float64)
    residual = np.zeros(valid.shape, dtype=np.float64)
    nuisance[valid] = _restore_prior(
        channel["nuisance_model"].predict_proba(np.asarray(base)[valid])[:, 1],
        float(channel["prevalence"]),
    )
    active = np.asarray(channel["active"], dtype=np.int64)
    residual[valid] = channel["residual_model"].predict(
        np.asarray(contrast)[valid][:, active],
    )
    return nuisance + float(dose) * residual, nuisance, residual


def _truthblind_policy_context(
    bundle: Mapping[str, object], correct_scored: Mapping[str, np.ndarray],
    control_scored: Sequence[Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build the matched deployment-visible base and chemical contrast."""
    required = (
        "schema", "feature_names", "actions", "global_action", "channels",
        "dose", "threshold", "risk_penalty", "rule_key", "control_rule_keys",
    )
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"frozen policy fields absent: {missing}")
    if bundle["schema"] != "chemaware_truthblind_candidate_policy_v1":
        raise ValueError("unsupported frozen policy schema")
    names = tuple(bundle["feature_names"])
    if names != tuple(FEATURE_NAMES):
        raise ValueError("frozen policy feature schema drifted")
    keys = tuple(bundle["control_rule_keys"])
    if len(keys) != len(control_scored) or not keys:
        raise ValueError("control score channels do not match the frozen policy")
    actions = [tuple(map(float, action)) for action in bundle["actions"]]
    correct = build_truthblind_candidate_features(
        correct_scored, actions, int(bundle["global_action"]), str(bundle["rule_key"]),
    )
    controls = [
        build_truthblind_candidate_features(scored, actions, int(bundle["global_action"]), key)
        for scored, key in zip(control_scored, keys, strict=True)
    ]
    for control in controls:
        for key in ("valid", "proposed_candidate", "baseline_candidate"):
            if not np.array_equal(correct[key], control[key]):
                raise ValueError(f"correct/control candidate rows drifted at {key}")
        if len(correct["reference_ptr"]) != len(control["reference_ptr"]) or any(
            not np.array_equal(left, right)
            for left, right in zip(
                correct["reference_ptr"], control["reference_ptr"], strict=True,
            )
        ):
            raise ValueError("correct/control candidate rows drifted at reference_ptr")
    base_indices, chemical_indices = _feature_indices(names)
    base = np.asarray(correct["feature"], dtype=np.float32)[..., base_indices]
    correct_chemical = np.asarray(correct["feature"], dtype=np.float32)[..., chemical_indices]
    representation = str(bundle.get("contrast_representation", "mean"))
    if representation == "mean":
        control_mean = np.mean(
            np.stack([
                np.asarray(control["feature"], dtype=np.float32)
                for control in controls
            ]),
            axis=0,
        )
        contrast = (correct_chemical - control_mean[..., chemical_indices]).astype(
            np.float32
        )
    elif representation == "symmetric_summary":
        if len(controls) < 3:
            raise ValueError("symmetric summary requires at least three control centers")
        differences = np.stack([
            correct_chemical
            - np.asarray(control["feature"], dtype=np.float32)[..., chemical_indices]
            for control in controls
        ], axis=0)
        contrast = np.concatenate((
            np.mean(differences, axis=0),
            np.min(differences, axis=0),
            np.max(differences, axis=0),
            np.std(differences, axis=0),
            np.mean(differences > 0, axis=0),
            np.mean(differences < 0, axis=0),
        ), axis=-1).astype(np.float32)
    else:
        raise ValueError(f"unsupported contrast representation: {representation}")
    contrast[~correct["valid"]] = 0.0
    return correct, base, contrast


def _score_truthblind_contrast(
    bundle: Mapping[str, object], correct: Mapping[str, np.ndarray],
    base: np.ndarray, contrast: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score one frozen contrast arm against a matched candidate table."""
    benefit, benefit_base, benefit_residual = _predict_channel(
        bundle["channels"]["benefit"], base, contrast, correct["valid"], float(bundle["dose"]),
    )
    harm, harm_base, harm_residual = _predict_channel(
        bundle["channels"]["harmful"], base, contrast, correct["valid"], float(bundle["dose"]),
    )
    utility = benefit - float(bundle["risk_penalty"]) * harm
    utility[~correct["valid"]] = -np.inf
    selected, abstained = rank_candidates_from_utility(correct, utility, float(bundle["threshold"]))
    selected_slot = np.argmax(utility, axis=1).astype(np.int16)
    selected_slot[abstained] = -1
    return {
        **correct,
        "utility": utility,
        "benefit": benefit,
        "harm": harm,
        "benefit_base": benefit_base,
        "harm_base": harm_base,
        "benefit_residual": benefit_residual,
        "harm_residual": harm_residual,
        "selected_candidate": selected,
        "selected_candidate_slot": selected_slot,
        "abstained": abstained,
    }


def predict_truthblind_policy(
    bundle: Mapping[str, object], correct_scored: Mapping[str, np.ndarray],
    control_scored: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Apply a frozen policy while making truth access structurally impossible."""
    correct, base, contrast = _truthblind_policy_context(
        bundle, correct_scored, control_scored,
    )
    return _score_truthblind_contrast(bundle, correct, base, contrast)


def predict_truthblind_policy_arms(
    bundle: Mapping[str, object], correct_scored: Mapping[str, np.ndarray],
    control_scored: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Score correct chemistry and deployment-safe counterfactual null arms."""
    correct, base, contrast = _truthblind_policy_context(
        bundle, correct_scored, control_scored,
    )
    rotated, _ = rotate_candidate_contrast_truthblind(contrast, correct["valid"])
    contrasts = {
        "correct": contrast,
        "zero_contrast": np.zeros_like(contrast),
        "reversed_contrast": -contrast,
        "candidate_rotated_truthblind": rotated,
    }
    return {
        name: _score_truthblind_contrast(bundle, correct, base, arm)
        for name, arm in contrasts.items()
    }


def predict_truthblind_baselines(
    bundle: Mapping[str, object], correct_scored: Mapping[str, np.ndarray],
    control_scored: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Apply the two frozen mechanism baselines without accepting truth.

    ``same_feature_direct`` receives exactly the nuisance block and the
    correct-minus-control chemical block used by the primary policy, but fits
    the labels directly.  ``nuisance_only`` reuses the primary channel's
    official-DreaMS/mass nuisance models and receives no rule contrast.
    """
    if "baselines" not in bundle:
        raise KeyError("frozen policy does not contain mechanism baselines")
    baselines = bundle["baselines"]
    if not isinstance(baselines, Mapping):
        raise TypeError("frozen mechanism baseline registry is invalid")
    required_baselines = ("same_feature_direct", "nuisance_only")
    missing_baselines = [name for name in required_baselines if name not in baselines]
    if missing_baselines:
        raise KeyError(f"frozen mechanism baselines absent: {missing_baselines}")

    correct, base, contrast = _truthblind_policy_context(
        bundle, correct_scored, control_scored,
    )
    risk_penalty = float(bundle["risk_penalty"])

    direct_contract = baselines["same_feature_direct"]
    direct_feature = np.concatenate((base, contrast), axis=-1)
    direct_predictions: dict[str, np.ndarray] = {}
    for target in ("benefit", "harmful"):
        channel = direct_contract["channels"][target]
        values = np.zeros(correct["valid"].shape, dtype=np.float64)
        values[correct["valid"]] = _restore_prior(
            channel["model"].predict_proba(direct_feature[correct["valid"]])[:, 1],
            float(channel["prevalence"]),
        )
        direct_predictions[target] = values
    direct_utility = (
        direct_predictions["benefit"] - risk_penalty * direct_predictions["harmful"]
    )
    direct_utility[~correct["valid"]] = -np.inf

    nuisance_contract = baselines["nuisance_only"]
    nuisance_predictions: dict[str, np.ndarray] = {}
    for target in ("benefit", "harmful"):
        channel = bundle["channels"][target]
        values = np.zeros(correct["valid"].shape, dtype=np.float64)
        values[correct["valid"]] = _restore_prior(
            channel["nuisance_model"].predict_proba(base[correct["valid"]])[:, 1],
            float(channel["prevalence"]),
        )
        nuisance_predictions[target] = values
    nuisance_utility = (
        nuisance_predictions["benefit"] - risk_penalty * nuisance_predictions["harmful"]
    )
    nuisance_utility[~correct["valid"]] = -np.inf

    output: dict[str, dict[str, np.ndarray]] = {}
    for name, contract, utility, predictions in (
        ("same_feature_direct", direct_contract, direct_utility, direct_predictions),
        ("nuisance_only", nuisance_contract, nuisance_utility, nuisance_predictions),
    ):
        selected, abstained = rank_candidates_from_utility(
            correct, utility, float(contract["threshold"]),
        )
        selected_slot = np.argmax(utility, axis=1).astype(np.int16)
        selected_slot[abstained] = -1
        output[name] = {
            "selected_candidate": selected,
            "selected_candidate_slot": selected_slot,
            "abstained": abstained,
            "utility": utility,
            "benefit": predictions["benefit"],
            "harm": predictions["harmful"],
            "valid": correct["valid"],
            "proposed_candidate": correct["proposed_candidate"],
            "baseline_candidate": correct["baseline_candidate"],
        }
    return output


def arbitrate_truthblind_predictions(
    primary: Mapping[str, np.ndarray], fallback: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compose two frozen policies without labels: primary action, then fallback.

    The primary policy always wins when it acts, including when both policies
    propose different candidates.  The fallback can only fill primary
    abstentions.  This keeps arbitration independent of correctness outcomes.
    """
    required = (
        "selected_candidate", "selected_candidate_slot", "abstained",
        "utility", "valid", "proposed_candidate", "baseline_candidate",
    )
    for name, prediction in (("primary", primary), ("fallback", fallback)):
        missing = [key for key in required if key not in prediction]
        if missing:
            raise KeyError(f"{name} prediction fields absent: {missing}")
    for key in ("valid", "proposed_candidate", "baseline_candidate"):
        if not np.array_equal(primary[key], fallback[key]):
            raise ValueError(f"arbitration candidate layout drifted at {key}")
    primary_abstained = np.asarray(primary["abstained"], dtype=bool)
    fallback_abstained = np.asarray(fallback["abstained"], dtype=bool)
    if primary_abstained.shape != fallback_abstained.shape:
        raise ValueError("arbitration abstention arrays are not aligned")
    use_primary = ~primary_abstained
    use_fallback = primary_abstained & ~fallback_abstained
    selected_candidate = np.asarray(primary["baseline_candidate"], dtype=np.int16).copy()
    selected_slot = np.full(len(selected_candidate), -1, dtype=np.int16)
    source = np.zeros(len(selected_candidate), dtype=np.int8)
    selected_candidate[use_primary] = np.asarray(
        primary["selected_candidate"], dtype=np.int16,
    )[use_primary]
    selected_slot[use_primary] = np.asarray(
        primary["selected_candidate_slot"], dtype=np.int16,
    )[use_primary]
    source[use_primary] = 1
    selected_candidate[use_fallback] = np.asarray(
        fallback["selected_candidate"], dtype=np.int16,
    )[use_fallback]
    selected_slot[use_fallback] = np.asarray(
        fallback["selected_candidate_slot"], dtype=np.int16,
    )[use_fallback]
    source[use_fallback] = 2
    return {
        "selected_candidate": selected_candidate,
        "selected_candidate_slot": selected_slot,
        "abstained": ~(use_primary | use_fallback),
        "action_source": source,
        "primary_acted": use_primary,
        "fallback_acted": use_fallback,
        "valid": np.asarray(primary["valid"], dtype=bool),
        "proposed_candidate": np.asarray(primary["proposed_candidate"], dtype=np.int16),
        "baseline_candidate": np.asarray(primary["baseline_candidate"], dtype=np.int16),
    }


def predict_truthblind_direct_first_backoff(
    bundle: Mapping[str, object], correct_scored: Mapping[str, np.ndarray],
    control_scored: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Direct same-feature expert first; residual ChemAware fills abstentions."""
    residual = predict_truthblind_policy(bundle, correct_scored, control_scored)
    baselines = predict_truthblind_baselines(bundle, correct_scored, control_scored)
    return arbitrate_truthblind_predictions(baselines["same_feature_direct"], residual)


def predict_truthblind_direct_first_exclusive_backoff(
    bundle: Mapping[str, object], correct_scored: Mapping[str, np.ndarray],
    control_scored: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Use ChemAware fallback only when every deployable null arm abstains.

    The gate is candidate-label-free.  It asks whether correct chemical
    alignment is necessary for the frozen policy to act, rather than merely
    whether the correct arm has a high score.
    """
    arms = predict_truthblind_policy_arms(bundle, correct_scored, control_scored)
    residual = dict(arms["correct"])
    control_names = (
        "zero_contrast", "reversed_contrast", "candidate_rotated_truthblind",
    )
    exclusive = ~np.asarray(residual["abstained"], dtype=bool)
    for name in control_names:
        exclusive &= np.asarray(arms[name]["abstained"], dtype=bool)
    baseline = np.asarray(residual["baseline_candidate"], dtype=np.int16)
    selected = np.asarray(residual["selected_candidate"], dtype=np.int16).copy()
    slot = np.asarray(residual["selected_candidate_slot"], dtype=np.int16).copy()
    selected[~exclusive] = baseline[~exclusive]
    slot[~exclusive] = -1
    residual["selected_candidate"] = selected
    residual["selected_candidate_slot"] = slot
    residual["abstained"] = ~exclusive
    residual["chemical_exclusive"] = exclusive
    baselines = predict_truthblind_baselines(bundle, correct_scored, control_scored)
    output = arbitrate_truthblind_predictions(
        baselines["same_feature_direct"], residual,
    )
    output["chemical_exclusive"] = exclusive
    return output

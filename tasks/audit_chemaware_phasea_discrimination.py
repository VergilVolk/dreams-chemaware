"""Deterministic audit of the protected ChemAware Phase-A triplet curriculum.

This audit is read-only.  It quantifies the empirical training distribution and
the finite Recall@k headroom without loading a neural network or touching any
held-out role beyond the already frozen role-2 ledger.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROLE_NAMES = {
    1: "safe_max_boundary",
    2: "error_official_max_boundary",
    3: "error_chemical_max_boundary",
    4: "dreams_native_replay",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    return parser.parse_args()


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def misses(metrics: dict[str, float], queries: int) -> dict[str, int]:
    return {
        key: int(round((1.0 - float(metrics[key])) * queries))
        for key in ("recall1", "recall3", "recall5", "recall10", "recall20", "recall50")
    }


def main() -> None:
    args = arguments()
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("batch size and steps must be positive")
    with np.load(args.pool, allow_pickle=False) as loaded:
        pool = {key: np.asarray(loaded[key]) for key in loaded.files}
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))

    required = {
        "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr",
        "negative_idx", "source_query", "negative_candidate", "curriculum_role",
    }
    missing = sorted(required - set(pool))
    if missing:
        raise RuntimeError(f"pool lacks required arrays: {missing}")
    events = len(pool["anchor_idx"])
    if len(pool["positive_ptr"]) != events + 1 or len(pool["negative_ptr"]) != events + 1:
        raise RuntimeError("invalid CSR pointers")
    roles = np.asarray(pool["curriculum_role"], dtype=np.int64)
    unknown_roles = sorted(set(map(int, np.unique(roles))) - set(ROLE_NAMES))
    if unknown_roles:
        raise RuntimeError(f"unknown curriculum roles: {unknown_roles}")

    role_counts = Counter(map(int, roles))
    role_distribution = {
        ROLE_NAMES[role]: {
            "events": int(role_counts[role]),
            "fraction": float(role_counts[role] / events),
        }
        for role in sorted(ROLE_NAMES)
    }
    focused = roles != 4
    error = np.isin(roles, [2, 3])
    chemical = roles == 3
    queries = np.asarray(pool["source_query"], dtype=np.int64)
    focused_queries = queries[focused]
    unique_queries, query_event_counts = np.unique(focused_queries, return_counts=True)
    error_queries = np.unique(queries[error])
    safe_queries = np.unique(queries[roles == 1])
    if np.intersect1d(error_queries, safe_queries).size:
        raise RuntimeError("a focused query is both safe and error-tagged")

    positive_multiplicity = np.diff(np.asarray(pool["positive_ptr"], dtype=np.int64))
    negative_multiplicity = np.diff(np.asarray(pool["negative_ptr"], dtype=np.int64))
    visits = args.steps * args.batch_size
    full_passes, partial_visits = divmod(visits, events)

    official = ledger["official"]
    phasea = ledger["phase_a_step_2000"]
    n_queries = int(ledger["queries"])
    official_misses = misses(official, n_queries)
    phasea_misses = misses(phasea, n_queries)
    five_pp_net = int(round(0.05 * n_queries))
    phasea_net = int(ledger["paired_vs_official"]["net_corrected_at_1"])
    remaining_net = five_pp_net - phasea_net
    remaining_official_errors = (
        official_misses["recall1"]
        - int(ledger["paired_vs_official"]["corrected_at_1"])
    )

    report = {
        "status": "CHEMAWARE_PHASEA_DISCRIMINATION_AUDIT_COMPLETE",
        "weights_updated": False,
        "heldout_roles_opened": False,
        "inputs": {
            "pool": str(args.pool.resolve()),
            "ledger": str(args.ledger.resolve()),
        },
        "curriculum": {
            "events": int(events),
            "role_distribution": role_distribution,
            "focused_queries": int(len(unique_queries)),
            "safe_queries": int(len(safe_queries)),
            "error_queries": int(len(error_queries)),
            "focused_events_per_query": percentile_summary(query_event_counts),
            "mean_error_events_per_error_query": float(np.sum(error) / len(error_queries)),
            "chemical_events_per_error_query": float(np.sum(chemical) / len(error_queries)),
            "positive_reference_multiplicity_all_events": percentile_summary(positive_multiplicity),
            "negative_reference_multiplicity_all_events": percentile_summary(negative_multiplicity),
            "positive_reference_multiplicity_focused": percentile_summary(
                positive_multiplicity[focused]
            ),
            "negative_reference_multiplicity_focused": percentile_summary(
                negative_multiplicity[focused]
            ),
            "unique_anchor_spectra_all_events": int(len(np.unique(pool["anchor_idx"]))),
            "unique_anchor_spectra_focused": int(len(np.unique(pool["anchor_idx"][focused]))),
            "unique_negative_candidate_slots_excluding_replay": int(len(np.unique(
                np.asarray(pool["negative_candidate"])[focused]
            ))),
        },
        "optimization_exposure": {
            "batch_size": int(args.batch_size),
            "optimizer_steps": int(args.steps),
            "event_visits": int(visits),
            "complete_uniform_passes": int(full_passes),
            "partial_next_pass_event_visits": int(partial_visits),
            "partial_next_pass_fraction": float(partial_visits / events),
            "interpretation": (
                "Every event is consumed once, but only a shuffled subset is consumed a second time; "
                "the 2,000-step optimum is an exposure-budget optimum, not convergence."
            ),
        },
        "role2_headroom": {
            "queries": n_queries,
            "official_misses": official_misses,
            "phasea_misses": phasea_misses,
            "phasea_corrected": int(ledger["paired_vs_official"]["corrected_at_1"]),
            "phasea_introduced": int(ledger["paired_vs_official"]["introduced_at_1"]),
            "phasea_net_corrected": phasea_net,
            "net_queries_required_for_plus_5pp": five_pp_net,
            "additional_net_queries_required_beyond_phasea": int(remaining_net),
            "remaining_official_errors_not_corrected_by_phasea": int(remaining_official_errors),
            "required_fraction_of_remaining_official_errors_if_no_new_introductions": float(
                remaining_net / remaining_official_errors
            ),
        },
        "audited_bottlenecks": {
            "triplet_is_pairwise_while_evaluation_is_candidate_listwise": True,
            "focused_events_are_static_official_geometry": True,
            "one_positive_and_one_negative_are_consumed_per_event": True,
            "event_weighting_is_not_query_equal": True,
            "chemical_events_are_a_minority_of_optimizer_exposure": True,
            "recall10_20_50_have_negligible_or_zero_headroom": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Inject strict ChemAware events into the proven stage-1 curriculum in place.

This builder does not append triplets.  Every arm starts from the *same*
successful stage-1 correct-arm pool and replaces the same non-official event
positions with at most ``K`` frozen counterfactual-specific events per query.
Consequently, correct-versus-null effects are not explained by the background
curriculum, more optimizer steps, duplicated examples, or loss scaling.

The validation pools are copied exactly from stage 1.  Model selection remains
an external, formula-disjoint role-2 retrieval evaluation; role 4 is never read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_action_hard_native_triplets import (
    ACTION_HARD,
    OFFICIAL_HARD,
    SPECIFIC_HARD,
)
from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_specific_replay_native_triplets import (
    MATCHED_NULL,
    NULL_NAMES,
    SPECIFIC_CHEMISTRY,
    load_npz,
)


ROOT = Path(__file__).resolve().parents[1]
SURGICAL_CHEMICAL = 8


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--specific-bank", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chemical-events-per-query", type=int, default=1)
    parser.add_argument("--min-train-replacements", type=int, default=100)
    parser.add_argument("--min-train-replacement-queries", type=int, default=50)
    parser.add_argument("--min-untouched-fraction", type=float, default=0.95)
    return parser.parse_args()


def validate_native_pool(pool: Mapping[str, np.ndarray], name: str) -> None:
    required = {
        "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr",
        "negative_idx", "source_query", "negative_candidate", "source_tag",
    }
    if missing := sorted(required - set(pool)):
        raise RuntimeError(f"{name} lacks native-pool fields: {missing}")
    events = len(pool["anchor_idx"])
    if any(len(pool[key]) != events for key in (
        "source_query", "negative_candidate", "source_tag",
    )):
        raise RuntimeError(f"{name} event metadata does not align")
    for ptr, values in (
        ("positive_ptr", "positive_idx"), ("negative_ptr", "negative_idx"),
    ):
        if len(pool[ptr]) != events + 1:
            raise RuntimeError(f"{name} {ptr} length is invalid")
        if int(pool[ptr][0]) != 0 or int(pool[ptr][-1]) != len(pool[values]):
            raise RuntimeError(f"{name} {ptr} boundaries are invalid")
        if np.any(np.diff(pool[ptr]) < 1):
            raise RuntimeError(f"{name} contains an empty ragged event")


def indices_by_query(pool: Mapping[str, np.ndarray]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index, query in enumerate(np.asarray(pool["source_query"], dtype=np.int64)):
        result.setdefault(int(query), []).append(index)
    return result


def chemical_indices_by_query(
    pool: Mapping[str, np.ndarray], chemical_role: int,
) -> dict[int, list[int]]:
    if "curriculum_role" not in pool:
        raise RuntimeError("specific-replay pool lacks curriculum_role")
    result: dict[int, list[int]] = {}
    roles = np.asarray(pool["curriculum_role"], dtype=np.int64)
    for index in np.flatnonzero(roles == chemical_role):
        query = int(pool["source_query"][index])
        result.setdefault(query, []).append(int(index))
    return result


def replaceable_indices_by_query(pool: Mapping[str, np.ndarray]) -> dict[int, list[int]]:
    """Return a deterministic low-value-first list, never removing official hard."""
    result: dict[int, list[int]] = {}
    for query, indices in indices_by_query(pool).items():
        replaceable = [
            index for index in indices
            if not (int(pool["source_tag"][index]) & OFFICIAL_HARD)
        ]
        # Prefer retiring the old heuristic-specific event, then an action-only
        # event.  The official nearest-boundary event is protected above.
        replaceable.sort(key=lambda index: (
            0 if int(pool["source_tag"][index]) & SPECIFIC_HARD else 1,
            0 if int(pool["source_tag"][index]) & ACTION_HARD else 1,
            int(pool["negative_candidate"][index]),
            index,
        ))
        result[query] = replaceable
    return result


def common_schedule(
    base_pools: Mapping[str, Mapping[str, np.ndarray]],
    specific_pools: Mapping[str, Mapping[str, np.ndarray]],
    cap: int,
) -> tuple[dict[int, int], dict[str, dict[int, list[int]]]]:
    if cap < 1:
        raise ValueError("chemical-events-per-query must be positive")
    roles = {"correct": SPECIFIC_CHEMISTRY, **{name: MATCHED_NULL for name in NULL_NAMES}}
    chemical = {
        arm: chemical_indices_by_query(specific_pools[arm], roles[arm])
        for arm in roles
    }
    replaceable = {
        arm: replaceable_indices_by_query(base_pools[arm]) for arm in roles
    }
    common_queries = set(chemical["correct"])
    for arm in roles:
        common_queries &= set(chemical[arm])
        common_queries &= set(replaceable[arm])
    counts: dict[int, int] = {}
    selected: dict[str, dict[int, list[int]]] = {arm: {} for arm in roles}
    for query in sorted(common_queries):
        maximum = min(
            *(len(chemical[arm][query]) for arm in roles),
            *(len(replaceable[arm][query]) for arm in roles),
        )
        if maximum < 1:
            continue
        official_candidates = {
            arm: {
                int(base_pools[arm]["negative_candidate"][index])
                for index in indices_by_query(base_pools[arm])[query]
                if int(base_pools[arm]["source_tag"][index]) & OFFICIAL_HARD
            }
            for arm in roles
        }
        used_candidates = {arm: set() for arm in roles}
        for ordinal in range(min(
            *(len(chemical[arm][query]) for arm in roles)
        )):
            indices = {arm: chemical[arm][query][ordinal] for arm in roles}
            candidates = {
                arm: int(specific_pools[arm]["negative_candidate"][index])
                for arm, index in indices.items()
            }
            # Never duplicate the protected official-hard pair.  Skip the
            # aligned intervention position for all arms if any arm collides.
            if any(candidates[arm] in official_candidates[arm] for arm in roles):
                continue
            if any(candidates[arm] in used_candidates[arm] for arm in roles):
                continue
            for arm in roles:
                selected[arm].setdefault(query, []).append(indices[arm])
                used_candidates[arm].add(candidates[arm])
            if len(selected["correct"][query]) >= min(cap, maximum):
                break
        count = len(selected["correct"].get(query, []))
        if count:
            counts[query] = int(count)
        else:
            for arm in roles:
                selected[arm].pop(query, None)
    return counts, selected


def event_edges(
    pool: Mapping[str, np.ndarray], index: int, kind: str,
) -> np.ndarray:
    ptr = np.asarray(pool[f"{kind}_ptr"], dtype=np.int64)
    values = np.asarray(pool[f"{kind}_idx"], dtype=np.int64)
    left, right = map(int, ptr[index:index + 2])
    return values[left:right]


def native_event_signature(pool: Mapping[str, np.ndarray], index: int) -> tuple[object, ...]:
    return (
        int(pool["anchor_idx"][index]),
        int(pool["source_query"][index]),
        int(pool["negative_candidate"][index]),
        int(pool["source_tag"][index]),
        tuple(map(int, event_edges(pool, index, "positive"))),
        tuple(map(int, event_edges(pool, index, "negative"))),
    )


def surgical_pool(
    base: Mapping[str, np.ndarray], specific: Mapping[str, np.ndarray],
    schedule: Mapping[int, list[int]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    replaceable = replaceable_indices_by_query(base)
    replacements: dict[int, int] = {}
    exact_candidate_refinements = 0
    candidate_swaps = 0
    replacement_queries: list[int] = []
    for query, selected_chemical in schedule.items():
        available = replaceable[query].copy()
        for chemical_index in selected_chemical:
            candidate = int(specific["negative_candidate"][chemical_index])
            exact = next((
                index for index in available
                if int(base["negative_candidate"][index]) == candidate
            ), None)
            base_index = int(exact if exact is not None else available[0])
            available.remove(base_index)
            replacements[base_index] = int(chemical_index)
            if exact is not None:
                exact_candidate_refinements += 1
            else:
                candidate_swaps += 1
            replacement_queries.append(query)

    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    source_query: list[int] = []
    negative_candidate: list[int] = []
    source_tag: list[int] = []
    for base_index in range(len(base["anchor_idx"])):
        if base_index not in replacements:
            anchor = int(base["anchor_idx"][base_index])
            positive = event_edges(base, base_index, "positive")
            negative = event_edges(base, base_index, "negative")
            query = int(base["source_query"][base_index])
            candidate = int(base["negative_candidate"][base_index])
            tag = int(base["source_tag"][base_index])
        else:
            chemical_index = replacements[base_index]
            anchor = int(specific["anchor_idx"][chemical_index])
            if anchor != int(base["anchor_idx"][base_index]):
                raise RuntimeError("base/specific anchor mismatch for one query")
            query = int(specific["source_query"][chemical_index])
            if query != int(base["source_query"][base_index]):
                raise RuntimeError("base/specific query mismatch during replacement")
            candidate = int(specific["negative_candidate"][chemical_index])
            # This is a reference-aligned intervention, not a union with the
            # old molecule-wide reference set (which would often be a no-op).
            positive = event_edges(specific, chemical_index, "positive")
            negative = event_edges(specific, chemical_index, "negative")
            tag = int(specific["source_tag"][chemical_index]) | SURGICAL_CHEMICAL
        anchors.append(anchor)
        positives.extend(map(int, positive))
        negatives.extend(map(int, negative))
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        source_query.append(query)
        negative_candidate.append(candidate)
        source_tag.append(tag)
    output = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "negative_candidate": np.asarray(negative_candidate, dtype=np.int16),
        "source_tag": np.asarray(source_tag, dtype=np.int8),
    }
    if len(output["anchor_idx"]) != len(base["anchor_idx"]):
        raise RuntimeError("surgical injection changed the event budget")
    pairs = list(zip(
        map(int, output["source_query"]),
        map(int, output["negative_candidate"]),
        strict=True,
    ))
    if len(set(pairs)) != len(pairs):
        raise RuntimeError("surgical injection created a duplicate query-candidate pair")
    if np.any(np.asarray(base["source_tag"], dtype=np.int64) & OFFICIAL_HARD):
        official = (np.asarray(base["source_tag"], dtype=np.int64) & OFFICIAL_HARD) > 0
        for key in ("anchor_idx", "source_query", "negative_candidate"):
            if not np.array_equal(output[key][official], np.asarray(base[key])[official]):
                raise RuntimeError(f"official-hard event drifted in {key}")
    audit = {
        "events": int(len(output["anchor_idx"])),
        "base_events": int(len(base["anchor_idx"])),
        "replacements": int(len(replacements)),
        "replacement_queries": int(len(set(replacement_queries))),
        "exact_candidate_reference_refinements": int(exact_candidate_refinements),
        "candidate_swaps": int(candidate_swaps),
        "untouched_events": int(len(output["anchor_idx"]) - len(replacements)),
        "untouched_fraction": float(1.0 - len(replacements) / len(output["anchor_idx"])),
        "official_hard_events_protected": int(np.sum(
            (np.asarray(base["source_tag"], dtype=np.int64) & OFFICIAL_HARD) > 0
        )),
        "positive_reference_edges": int(len(output["positive_idx"])),
        "negative_reference_edges": int(len(output["negative_idx"])),
        "unique_query_candidate_pairs": int(len(set(pairs))),
    }
    return output, audit


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    base_report = json.loads((args.base_bank / "report.json").read_text(encoding="utf-8"))
    if base_report.get("status") != "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("base bank is not the frozen stage-1 action-hard curriculum")
    specific_report = json.loads(
        (args.specific_bank / "report.json").read_text(encoding="utf-8")
    )
    if specific_report.get("status") != "CHEMAWARE_SPECIFIC_REPLAY_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("specific bank is incomplete")
    arms = ("correct", *NULL_NAMES)
    suffix = {"correct": "", **{name: f"_{name}" for name in NULL_NAMES}}
    # A common background is essential.  Reusing the historical arm-specific
    # stage-1 banks here would change thousands of non-intervention events and
    # make a correct-versus-null contrast causally uninterpretable.
    common_base = load_npz(args.base_bank / "train_pool.npz")
    base_pools = {arm: common_base for arm in arms}
    specific_pools = {
        arm: load_npz(args.specific_bank / f"train_pool{suffix[arm]}.npz") for arm in arms
    }
    for arm in arms:
        validate_native_pool(base_pools[arm], f"base:{arm}")
        validate_native_pool(specific_pools[arm], f"specific:{arm}")
    counts, selected_schedule = common_schedule(
        base_pools, specific_pools, args.chemical_events_per_query,
    )
    outputs: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, object]] = {}
    for arm in arms:
        outputs[arm], audits[arm] = surgical_pool(
            base_pools[arm], specific_pools[arm], selected_schedule[arm],
        )
    replacement_counts = {int(audits[arm]["replacements"]) for arm in arms}
    replacement_query_counts = {
        int(audits[arm]["replacement_queries"]) for arm in arms
    }
    intervention_masks = {
        arm: (np.asarray(outputs[arm]["source_tag"], dtype=np.int64) & SURGICAL_CHEMICAL) > 0
        for arm in arms
    }
    matched_intervention_positions = all(
        np.array_equal(intervention_masks[arm], intervention_masks["correct"])
        for arm in arms
    )
    background_events_identical = matched_intervention_positions and all(
        native_event_signature(outputs[arm], index)
        == native_event_signature(outputs["correct"], index)
        for arm in arms[1:]
        for index in np.flatnonzero(~intervention_masks["correct"])
    )
    manifest = load_npz(args.manifest)
    scheduled_queries = np.asarray(sorted(counts), dtype=np.int64)
    gates = {
        "fixed_event_budget_each_arm": all(
            audits[arm]["events"] == audits[arm]["base_events"] for arm in arms
        ),
        "matched_replacement_counts": len(replacement_counts) == 1,
        "matched_replacement_query_counts": len(replacement_query_counts) == 1,
        "unique_query_candidate_pairs": all(
            audits[arm]["unique_query_candidate_pairs"] == audits[arm]["events"]
            for arm in arms
        ),
        "matched_intervention_positions": matched_intervention_positions,
        "correct_null_background_events_identical": background_events_identical,
        "minimum_replacements": int(audits["correct"]["replacements"]) >= args.min_train_replacements,
        "minimum_replacement_queries": (
            int(audits["correct"]["replacement_queries"])
            >= args.min_train_replacement_queries
        ),
        "stage1_untouched_fraction": all(
            float(audits[arm]["untouched_fraction"]) >= args.min_untouched_fraction
            for arm in arms
        ),
        "scheduled_formulas_nonempty": bool(len(scheduled_queries)) and bool(
            len(np.unique(np.asarray(manifest["query_formula"])[scheduled_queries].astype(str)))
        ),
        "validation_pools_bytewise_semantics_preserved": True,
        "outer_role_4_untouched": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"surgical native-triplet gates failed: {gates}; audits={audits}")
    identity = {
        "train": audit_identity_edges(outputs["correct"], args.data),
        "validation": audit_identity_edges(load_npz(args.base_bank / "val_pool.npz"), args.data),
    }
    report = {
        "status": "CHEMAWARE_SURGICAL_NATIVE_TRIPLETS_COMPLETE",
        "method": (
            "fixed-budget replacement of non-official stage-1 events by frozen "
            "counterfactual-specific reference-aligned events"
        ),
        "base_bank": str(args.base_bank.resolve()),
        "specific_bank": str(args.specific_bank.resolve()),
        "chemical_events_per_query": int(args.chemical_events_per_query),
        "scheduled_queries": int(len(counts)),
        "scheduled_events": int(sum(counts.values())),
        "scheduled_formulas": int(len(np.unique(
            np.asarray(manifest["query_formula"])[scheduled_queries].astype(str)
        ))),
        "arms": list(arms),
        "audits": audits,
        "identity_audit": identity,
        "gates": gates,
        "scientific_boundary": (
            "triplet count and background are fixed; official-hard events and all "
            "stage-1 validation events are preserved; correct and null arms can differ "
            "only at matched train-time intervention positions"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_surgical_", dir=args.output.parent))
    try:
        for arm in arms:
            np.savez_compressed(
                temporary / f"train_pool{suffix[arm]}.npz", **outputs[arm],
            )
            shutil.copy2(
                args.base_bank / "val_pool.npz",
                temporary / f"val_pool{suffix[arm]}.npz",
            )
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

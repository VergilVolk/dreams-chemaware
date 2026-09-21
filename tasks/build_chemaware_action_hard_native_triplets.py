"""Build full-coverage native DreaMS triplets with ChemAware hard negatives.

Every retained event has a true same-identity positive and a false candidate
negative.  ChemAware changes only which hard negative molecules are exposed:
the official nearest false candidate, up to K action-hard false candidates,
and one center-vs-multinull-specific false candidate.  Query/candidate pairs
are deduplicated, so reported scale never comes from copying a triplet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from chemaware_native_triplet_mining_core import validate_evidence


ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_HARD = 1
ACTION_HARD = 2
SPECIFIC_HARD = 4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-hard-k", type=int, default=2)
    parser.add_argument("--specific-hard-k", type=int, default=1)
    parser.add_argument("--min-train-queries", type=int, default=3500)
    parser.add_argument("--min-train-events", type=int, default=5000)
    parser.add_argument("--min-train-formulas", type=int, default=2000)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def metric(evidence: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(evidence["metric_names"]).tolist()
    )
    return np.asarray(evidence["arm_metric"])[..., names.index(name)]


def true_locals(manifest: Mapping[str, np.ndarray], query: int) -> np.ndarray:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    return np.flatnonzero(np.asarray(manifest["molecule_label"][left:right], dtype=bool))


def descending_lexicographic(slots: np.ndarray, keys: tuple[np.ndarray, ...]) -> np.ndarray:
    """Order slots by the key tuple, first key most important, descending."""
    return np.asarray(sorted(
        map(int, slots),
        key=lambda slot: tuple(float(key[slot]) for key in keys),
        reverse=True,
    ), dtype=np.int64)


def selected_negative_candidates(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    metrics: Mapping[str, np.ndarray], row: int, center: int,
    action_hard_k: int, specific_hard_k: int,
) -> dict[int, int]:
    query = int(np.asarray(evidence["query"])[row])
    truth = set(map(int, true_locals(manifest, query)))
    baseline = int(np.asarray(evidence["baseline_candidate"])[row])
    valid = np.asarray(evidence["valid"], dtype=bool)[row]
    proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)[row]
    false_slots = np.asarray([
        slot for slot in np.flatnonzero(valid) if int(proposed[slot]) not in truth
    ], dtype=np.int64)
    selected: dict[int, int] = {}

    official_rank = metrics["candidate_official_rank_fraction"][0, row]
    if baseline not in truth:
        selected[baseline] = OFFICIAL_HARD
    elif len(false_slots):
        slot = min(map(int, false_slots), key=lambda value: float(official_rank[value]))
        selected[int(proposed[slot])] = OFFICIAL_HARD

    support = metrics["action_top_fraction"][:, row]
    region = metrics["action_largest_region_fraction"][:, row]
    neighbor = metrics["action_same_neighbor_fraction"][:, row]
    advantage = metrics["action_best_advantage_over_baseline"][:, row]
    global_advantage = metrics["global_action_advantage_over_baseline"][:, row]
    global_select = metrics["global_action_selects_candidate"][:, row]
    rule_rank = metrics["candidate_rule_rank_fraction"][:, row]
    rule_max = metrics["candidate_rule_max"][:, row]
    rule_top2 = metrics["candidate_rule_top2_mean"][:, row]
    delta_rule_max = metrics["delta_rule_max"][:, row]
    delta_rule_top2 = metrics["delta_rule_top2_mean"][:, row]
    # A useful hard negative need not already win an action-grid point.  The
    # closest false candidate below the chemical decision boundary is exactly
    # where a triplet gradient is informative, so rank every false candidate
    # and treat actual grid winners as the highest-priority subset.
    eligible = false_slots
    ordered = descending_lexicographic(eligible, (
        -rule_rank[center], delta_rule_max[center], delta_rule_top2[center],
        rule_max[center], rule_top2[center], global_select[center],
        support[center], region[center], neighbor[center],
        global_advantage[center], advantage[center], -official_rank,
    ))
    for slot in ordered[:max(0, int(action_hard_k))]:
        candidate = int(proposed[slot])
        selected[candidate] = selected.get(candidate, 0) | ACTION_HARD

    others = np.asarray([index for index in range(len(support)) if index != center])
    support_delta = support[center] - np.median(support[others], axis=0)
    advantage_delta = advantage[center] - np.median(advantage[others], axis=0)
    rule_max_delta = rule_max[center] - np.median(rule_max[others], axis=0)
    delta_rule_delta = delta_rule_max[center] - np.median(delta_rule_max[others], axis=0)
    specific = false_slots[
        (rule_max_delta[false_slots] > 0.0)
        | (delta_rule_delta[false_slots] > 0.0)
        | (support_delta[false_slots] > 0.0)
        | (advantage_delta[false_slots] > 0.0)
    ]
    ordered_specific = descending_lexicographic(specific, (
        rule_max_delta, delta_rule_delta, support_delta, advantage_delta,
        rule_max[center], delta_rule_max[center], support[center], region[center],
        neighbor[center], -official_rank,
    ))
    for slot in ordered_specific[:max(0, int(specific_hard_k))]:
        candidate = int(proposed[slot])
        selected[candidate] = selected.get(candidate, 0) | SPECIFIC_HARD
    if not selected:
        raise RuntimeError(f"query {query} has no false candidate for a triplet")
    return selected


def build_arm_pool(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    center: int, action_hard_k: int, specific_hard_k: int,
) -> tuple[dict[str, np.ndarray], dict[str, object], set[tuple[int, int]]]:
    validate_evidence(evidence)
    metric_cache = {
        name: metric(evidence, name)
        for name in (
            "candidate_official_rank_fraction", "action_top_fraction",
            "action_largest_region_fraction", "action_same_neighbor_fraction",
            "action_best_advantage_over_baseline",
            "global_action_advantage_over_baseline",
            "global_action_selects_candidate",
            "candidate_rule_rank_fraction", "candidate_rule_max",
            "candidate_rule_top2_mean", "delta_rule_max",
            "delta_rule_top2_mean",
        )
    }
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    source_query: list[int] = []
    negative_candidate: list[int] = []
    source_tag: list[int] = []
    skipped_no_distinct_positive = 0
    pairs: set[tuple[int, int]] = set()
    for row, query_value in enumerate(np.asarray(evidence["query"], dtype=np.int64)):
        query = int(query_value)
        truth = true_locals(manifest, query)
        positive = np.unique(np.concatenate([
            molecule_rows(manifest, query, int(candidate)) for candidate in truth
        ]))
        anchor = int(manifest["query_row"][query])
        positive = positive[positive != anchor]
        if not len(positive):
            skipped_no_distinct_positive += 1
            continue
        selected = selected_negative_candidates(
            evidence, manifest, metric_cache, row, center,
            action_hard_k, specific_hard_k,
        )
        for candidate, tag in sorted(selected.items()):
            pair = (query, int(candidate))
            if pair in pairs:
                raise RuntimeError("query-negative pair deduplication failed")
            pairs.add(pair)
            negative = np.unique(molecule_rows(manifest, query, int(candidate)))
            if not len(negative):
                raise RuntimeError("selected hard-negative molecule has no reference spectrum")
            anchors.append(anchor)
            positives.extend(map(int, positive))
            negatives.extend(map(int, negative))
            positive_ptr.append(len(positives))
            negative_ptr.append(len(negatives))
            source_query.append(query)
            negative_candidate.append(int(candidate))
            source_tag.append(int(tag))
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "negative_candidate": np.asarray(negative_candidate, dtype=np.int16),
        "source_tag": np.asarray(source_tag, dtype=np.int8),
    }
    query_array = np.asarray(source_query, dtype=np.int64)
    formulas = np.asarray(manifest["query_formula"])[query_array].astype(str)
    tags = np.asarray(source_tag, dtype=np.int8)
    audit = {
        "triplet_events": int(len(anchors)),
        "unique_query_negative_pairs": int(len(pairs)),
        "unique_anchor_queries": int(len(np.unique(query_array))),
        "unique_formulas": int(len(np.unique(formulas))),
        "official_hard_events": int(np.sum((tags & OFFICIAL_HARD) > 0)),
        "action_hard_events": int(np.sum((tags & ACTION_HARD) > 0)),
        "specific_hard_events": int(np.sum((tags & SPECIFIC_HARD) > 0)),
        "multi_source_events": int(np.sum(np.asarray([int(tag).bit_count() for tag in tags]) > 1)),
        "positive_reference_edges": int(len(positives)),
        "negative_reference_edges": int(len(negatives)),
        "skipped_no_distinct_positive": int(skipped_no_distinct_positive),
    }
    return pool, audit, pairs


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    evidence = {
        "train": load_npz(args.evidence_dir / "train_triplet_evidence.npz"),
        "selection": load_npz(args.evidence_dir / "selection_triplet_evidence.npz"),
        "confirmation": load_npz(args.evidence_dir / "confirmation_triplet_evidence.npz"),
    }
    manifest = load_npz(args.manifest)
    arm_names = tuple(map(str, evidence["train"]["arm_names"].tolist()))
    if any(tuple(map(str, body["arm_names"].tolist())) != arm_names for body in evidence.values()):
        raise RuntimeError("chemistry arm names drifted across formula roles")
    pools: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, object]] = {}
    pair_sets: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for role in ("train", "selection", "confirmation"):
        for center, arm in enumerate(arm_names):
            pool, audit, pairs = build_arm_pool(
                evidence[role], manifest, center,
                args.action_hard_k, args.specific_hard_k,
            )
            pools[(role, arm)] = pool
            audits[f"{role}:{arm}"] = audit
            pair_sets[(role, arm)] = pairs
    correct_train = audits["train:correct"]
    gates = {
        "train_unique_queries": int(correct_train["unique_anchor_queries"]) >= args.min_train_queries,
        "train_unique_query_negative_pairs": int(correct_train["unique_query_negative_pairs"]) >= args.min_train_events,
        "train_unique_formulas": int(correct_train["unique_formulas"]) >= args.min_train_formulas,
        "every_event_is_unique_query_negative_pair": (
            correct_train["triplet_events"] == correct_train["unique_query_negative_pairs"]
        ),
        "outer_role_4_untouched": True,
    }
    overlaps = {}
    correct_pairs = pair_sets[("selection", "correct")]
    for arm in arm_names[1:]:
        null_pairs = pair_sets[("selection", arm)]
        overlaps[arm] = {
            "intersection": int(len(correct_pairs & null_pairs)),
            "union": int(len(correct_pairs | null_pairs)),
            "jaccard": float(len(correct_pairs & null_pairs) / max(1, len(correct_pairs | null_pairs))),
            "correct_only": int(len(correct_pairs - null_pairs)),
            "null_only": int(len(null_pairs - correct_pairs)),
        }
    if not all(gates.values()):
        raise RuntimeError(f"action-hard triplet gates failed: {gates}; train={correct_train}")
    identity_audit = {
        "train": audit_identity_edges(pools[("train", "correct")], args.data),
        "validation": audit_identity_edges(pools[("confirmation", "correct")], args.data),
    }
    report = {
        "status": "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE",
        "custom_component": "truth-valid action-conditioned hard-negative construction only",
        "training_runtime": "unmodified DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "action_hard_k": int(args.action_hard_k),
        "specific_hard_k": int(args.specific_hard_k),
        "deduplication_unit": "global query row plus local negative molecule",
        "roles": {
            "optimization": "formula roles 0-1",
            "construction_audit": "formula role 2",
            "model_validation": "formula role 3",
            "outer": "formula role 4 untouched",
        },
        "arms": list(arm_names),
        "audits": audits,
        "role2_correct_vs_null_pair_overlap": overlaps,
        "identity_audit": identity_audit,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_action_hard_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **pools[("train", "correct")])
        np.savez_compressed(temporary / "val_pool.npz", **pools[("confirmation", "correct")])
        for arm in arm_names[1:]:
            safe = arm.removeprefix("rule_response_")
            np.savez_compressed(temporary / f"train_pool_{safe}.npz", **pools[("train", arm)])
            np.savez_compressed(temporary / f"val_pool_{safe}.npz", **pools[("confirmation", arm)])
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

"""Re-mine a balanced ChemAware triplet curriculum from an adapted checkpoint.

The encoder and loss remain the native DreaMS implementation.  This builder
changes only the identity-valid negative curriculum.  Each query contributes
at most two candidate molecules: its hardest current false candidate and one
chemically informative false candidate inside a fixed hardness window.  A
strict correct-vs-three-null recipe, frozen on formula role 2, receives first
priority but is never used alone because its coverage is too small.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_action_hard_native_triplets import load_npz, metric, true_locals
from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from chemaware_native_triplet_mining_core import (
    directional_masks,
    recipe_dict,
    select_recipe,
    validate_evidence,
)


ROOT = Path(__file__).resolve().parents[1]

ADAPTIVE_HARD = 1
CHEMICAL_HARD = 2
STRICT_SPECIFIC = 4
FALLBACK_HARD = 8
MARGIN_VIOLATION = 16


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument(
        "--chemical-hardness-window", type=float, default=0.10,
        help="Chemical candidate must be within this cosine gap of the hardest false candidate.",
    )
    parser.add_argument("--events-per-query", type=int, default=2)
    parser.add_argument("--min-train-queries", type=int, default=3500)
    parser.add_argument("--min-train-events", type=int, default=6500)
    parser.add_argument("--min-train-formulas", type=int, default=2000)
    parser.add_argument("--min-chemical-events", type=int, default=1000)
    return parser.parse_args()


def cache_arrays(directory: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    rows = np.load(directory / "rows.npy", mmap_mode="r")
    embedding_path = directory / "embeddings_f32.npy"
    if not embedding_path.is_file():
        legacy_official = directory / "official_embeddings_f32.npy"
        if not legacy_official.is_file():
            raise FileNotFoundError(embedding_path)
        embedding_path = legacy_official
    embedding = np.load(embedding_path, mmap_mode="r")
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or embedding.ndim != 2 or len(rows) != len(embedding):
        raise RuntimeError("checkpoint embedding cache has incompatible arrays")
    if np.any(np.diff(rows) <= 0):
        raise RuntimeError("checkpoint cache rows must be strictly increasing")
    norms = np.linalg.norm(np.asarray(embedding[::max(1, len(embedding) // 4096)]), axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-4:
        raise RuntimeError("checkpoint cache embeddings are not normalized")
    return rows, embedding, embedding_path


def positions(cache_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    found = np.searchsorted(cache_rows, rows)
    bounded = np.minimum(found, len(cache_rows) - 1)
    missing = (found >= len(cache_rows)) | (cache_rows[bounded] != rows)
    if np.any(missing):
        absent = rows[missing]
        raise RuntimeError(f"spectrum rows absent from checkpoint cache: {absent[:8].tolist()}")
    return found


def molecule_scores(
    manifest: Mapping[str, np.ndarray], query: int,
    cache_rows: np.ndarray, embedding: np.ndarray,
) -> np.ndarray:
    mleft, mright = map(int, manifest["query_ptr"][query:query + 2])
    pleft = int(manifest["molecule_ptr"][mleft])
    pright = int(manifest["molecule_ptr"][mright])
    reference_rows = np.asarray(manifest["pair_candidate_row"][pleft:pright], dtype=np.int64)
    query_row = int(manifest["query_row"][query])
    query_z = np.asarray(embedding[positions(cache_rows, np.asarray([query_row]))[0]])
    reference_z = np.asarray(embedding[positions(cache_rows, reference_rows)])
    pair = reference_z @ query_z
    local_ptr = np.asarray(manifest["molecule_ptr"][mleft:mright + 1], dtype=np.int64) - pleft
    return np.maximum.reduceat(pair, local_ptr[:-1]).astype(np.float32, copy=False)


def metric_cache(evidence: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: metric(evidence, name)
        for name in (
            "action_top_fraction", "action_largest_region_fraction",
            "action_same_neighbor_fraction", "action_best_advantage_over_baseline",
            "candidate_rule_rank_fraction", "candidate_rule_max",
            "candidate_rule_top2_mean", "delta_rule_max", "delta_rule_top2_mean",
        )
    }


def strict_negative_candidates(
    evidence: Mapping[str, np.ndarray], row: int, center: int,
    correction: np.ndarray, protection: np.ndarray,
) -> set[int]:
    result: set[int] = set()
    if np.any(correction[row]):
        result.add(int(np.asarray(evidence["baseline_candidate"])[row]))
    proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)[row]
    result.update(map(int, proposed[np.asarray(protection[row], dtype=bool)]))
    return result


def chemical_order_key(
    slot: int, row: int, center: int, metrics: Mapping[str, np.ndarray],
    current_score: float, strict: bool,
) -> tuple[float, ...]:
    others = np.asarray([index for index in range(metrics["action_top_fraction"].shape[0]) if index != center])
    support = metrics["action_top_fraction"]
    region = metrics["action_largest_region_fraction"]
    neighbor = metrics["action_same_neighbor_fraction"]
    advantage = metrics["action_best_advantage_over_baseline"]
    rank = metrics["candidate_rule_rank_fraction"]
    rule_max = metrics["candidate_rule_max"]
    rule_top2 = metrics["candidate_rule_top2_mean"]
    delta_max = metrics["delta_rule_max"]
    delta_top2 = metrics["delta_rule_top2_mean"]
    contrasts = (
        abs(float(support[center, row, slot] - np.median(support[others, row, slot]))),
        abs(float(region[center, row, slot] - np.median(region[others, row, slot]))),
        abs(float(advantage[center, row, slot] - np.median(advantage[others, row, slot]))),
        abs(float(rule_max[center, row, slot] - np.median(rule_max[others, row, slot]))),
        abs(float(delta_max[center, row, slot] - np.median(delta_max[others, row, slot]))),
    )
    # Lexicographic ordering avoids inventing an uncalibrated weighted chemical
    # score.  Strict semantic specificity wins first, followed by repeated-arm
    # contrast, absolute rule evidence and current checkpoint hardness.
    return (
        float(strict),
        float(sum(value > 0.0 for value in contrasts)),
        *contrasts,
        -float(rank[center, row, slot]),
        float(rule_max[center, row, slot]),
        float(rule_top2[center, row, slot]),
        float(delta_max[center, row, slot]),
        float(delta_top2[center, row, slot]),
        float(support[center, row, slot]),
        float(region[center, row, slot]),
        float(neighbor[center, row, slot]),
        float(current_score),
    )


def select_query_negatives(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    metrics: Mapping[str, np.ndarray], row: int, center: int,
    scores: np.ndarray, correction: np.ndarray, protection: np.ndarray,
    margin: float, hardness_window: float, events_per_query: int,
) -> list[tuple[int, int]]:
    query = int(np.asarray(evidence["query"])[row])
    truth = set(map(int, true_locals(manifest, query)))
    false = np.asarray([candidate for candidate in range(len(scores)) if candidate not in truth], dtype=np.int64)
    if not len(false):
        raise RuntimeError(f"query {query} has no false molecule candidate")
    true_score = float(np.max(scores[np.asarray(sorted(truth), dtype=np.int64)]))
    false_order = false[np.argsort(-scores[false], kind="stable")]
    hardest = int(false_order[0])
    selected: dict[int, int] = {hardest: ADAPTIVE_HARD}

    strict = strict_negative_candidates(evidence, row, center, correction, protection)
    if hardest in strict:
        selected[hardest] |= STRICT_SPECIFIC
    valid = np.asarray(evidence["valid"], dtype=bool)[row]
    proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)[row]
    candidate_slot = {
        int(proposed[slot]): int(slot)
        for slot in np.flatnonzero(valid)
        if int(proposed[slot]) not in truth
    }
    hardest_score = float(scores[hardest])
    eligible = [
        candidate for candidate in candidate_slot
        if candidate != hardest
        and float(scores[candidate]) >= hardest_score - float(hardness_window)
    ]
    # Strict correction can point to the official baseline, which is not part
    # of proposed_candidate.  It remains eligible if it is genuinely hard.
    baseline = int(np.asarray(evidence["baseline_candidate"])[row])
    strict_baseline = (
        baseline in strict and baseline not in truth and baseline != hardest
        and float(scores[baseline]) >= hardest_score - hardness_window
    )
    chemical = None
    if eligible:
        chemical = max(
            eligible,
            key=lambda candidate: chemical_order_key(
                candidate_slot[candidate], row, center, metrics,
                float(scores[candidate]), candidate in strict,
            ),
        )
    # A strict baseline has priority over a merely broad chemical candidate;
    # if both are strict, retain the currently harder molecule.
    if strict_baseline and (
        chemical is None
        or chemical not in strict
        or float(scores[baseline]) > float(scores[chemical])
    ):
        chemical = baseline
    if chemical is not None:
        selected[chemical] = selected.get(chemical, 0) | CHEMICAL_HARD
        if chemical in strict:
            selected[chemical] |= STRICT_SPECIFIC

    # Fill a fixed per-query budget with the next current hard candidates.  No
    # pair is copied, and easy candidates never displace an available chemical
    # candidate inside the predeclared hardness window.
    for candidate in false_order:
        if len(selected) >= events_per_query:
            break
        candidate = int(candidate)
        selected[candidate] = selected.get(candidate, 0) | FALLBACK_HARD
    ordered = sorted(selected.items(), key=lambda item: (-float(scores[item[0]]), item[0]))
    ordered = ordered[:events_per_query]
    return [
        (
            candidate,
            tag | (MARGIN_VIOLATION if margin + float(scores[candidate]) - true_score > 0.0 else 0),
        )
        for candidate, tag in ordered
    ]


def build_pool(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    score_by_query: Mapping[int, np.ndarray], center: int,
    recipe, margin: float, hardness_window: float, events_per_query: int,
) -> tuple[dict[str, np.ndarray], dict[str, object], set[tuple[int, int]]]:
    validate_evidence(evidence)
    correction, protection, _ = directional_masks(evidence, recipe, center)
    metrics = metric_cache(evidence)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    source_query: list[int] = []
    negative_candidate: list[int] = []
    source_tag: list[int] = []
    current_negative_score: list[float] = []
    current_positive_score: list[float] = []
    pairs: set[tuple[int, int]] = set()
    for row, query_value in enumerate(np.asarray(evidence["query"], dtype=np.int64)):
        query = int(query_value)
        score = np.asarray(score_by_query[query], dtype=np.float32)
        truth = true_locals(manifest, query)
        positive = np.unique(np.concatenate([
            molecule_rows(manifest, query, int(candidate)) for candidate in truth
        ]))
        anchor = int(manifest["query_row"][query])
        positive = positive[positive != anchor]
        if not len(positive):
            raise RuntimeError(f"query {query} has no distinct positive reference")
        positive_score = float(np.max(score[truth]))
        chosen = select_query_negatives(
            evidence, manifest, metrics, row, center, score,
            correction, protection, margin, hardness_window, events_per_query,
        )
        for candidate, tag in chosen:
            pair = (query, candidate)
            if pair in pairs:
                raise RuntimeError("residual query-negative pair was duplicated")
            pairs.add(pair)
            negative = np.unique(molecule_rows(manifest, query, candidate))
            if not len(negative):
                raise RuntimeError("residual negative molecule has no spectrum")
            anchors.append(anchor)
            positives.extend(map(int, positive))
            negatives.extend(map(int, negative))
            positive_ptr.append(len(positives))
            negative_ptr.append(len(negatives))
            source_query.append(query)
            negative_candidate.append(candidate)
            source_tag.append(tag)
            current_negative_score.append(float(score[candidate]))
            current_positive_score.append(positive_score)
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "negative_candidate": np.asarray(negative_candidate, dtype=np.int16),
        "source_tag": np.asarray(source_tag, dtype=np.int8),
        "mining_positive_score": np.asarray(current_positive_score, dtype=np.float32),
        "mining_negative_score": np.asarray(current_negative_score, dtype=np.float32),
    }
    tags = np.asarray(source_tag, dtype=np.int8)
    query_array = np.asarray(source_query, dtype=np.int64)
    formulas = np.asarray(manifest["query_formula"])[query_array].astype(str)
    gaps = np.asarray(current_positive_score) - np.asarray(current_negative_score)
    _, per_query_counts = np.unique(query_array, return_counts=True)
    audit = {
        "triplet_events": int(len(anchors)),
        "unique_query_negative_pairs": int(len(pairs)),
        "anchor_queries": int(len(np.unique(query_array))),
        "unique_formulas": int(len(np.unique(formulas))),
        "adaptive_hard_events": int(np.sum((tags & ADAPTIVE_HARD) > 0)),
        "chemical_hard_events": int(np.sum((tags & CHEMICAL_HARD) > 0)),
        "strict_specific_events": int(np.sum((tags & STRICT_SPECIFIC) > 0)),
        "fallback_events": int(np.sum((tags & FALLBACK_HARD) > 0)),
        "margin_violating_events": int(np.sum((tags & MARGIN_VIOLATION) > 0)),
        "margin_violating_fraction": float(np.mean((tags & MARGIN_VIOLATION) > 0)),
        "mean_current_positive_minus_negative": float(np.mean(gaps)),
        "positive_reference_edges": int(len(positives)),
        "negative_reference_edges": int(len(negatives)),
        "events_per_query_min": int(np.min(per_query_counts)),
        "events_per_query_max": int(np.max(per_query_counts)),
    }
    return pool, audit, pairs


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.events_per_query < 1:
        raise ValueError("--events-per-query must be positive")
    if args.chemical_hardness_window < 0.0:
        raise ValueError("--chemical-hardness-window must be nonnegative")
    evidence = {
        role: load_npz(args.evidence_dir / f"{file_role}_triplet_evidence.npz")
        for role, file_role in (
            ("train", "train"), ("selection", "selection"),
            ("confirmation", "confirmation"),
        )
    }
    manifest = load_npz(args.manifest)
    cache_rows, embedding, embedding_path = cache_arrays(args.embedding_cache)
    role_formulas = {role: set(body["formula"].astype(str)) for role, body in evidence.items()}
    if any(
        role_formulas[left] & role_formulas[right]
        for left, right in (("train", "selection"), ("train", "confirmation"), ("selection", "confirmation"))
    ):
        raise RuntimeError("formula roles overlap in residual triplet evidence")
    recipe, selection, grid = select_recipe(evidence["selection"])
    arm_names = tuple(map(str, evidence["train"]["arm_names"].tolist()))
    score_cache = {
        role: {
            int(query): molecule_scores(manifest, int(query), cache_rows, embedding)
            for query in np.asarray(body["query"], dtype=np.int64)
        }
        for role, body in evidence.items()
    }
    pools: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, object]] = {}
    pair_sets: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for role, body in evidence.items():
        for center, arm in enumerate(arm_names):
            pool, audit, pairs = build_pool(
                body, manifest, score_cache[role], center, recipe,
                args.margin, args.chemical_hardness_window, args.events_per_query,
            )
            pools[(role, arm)] = pool
            audits[f"{role}:{arm}"] = audit
            pair_sets[(role, arm)] = pairs
    train = audits["train:correct"]
    gates = {
        "train_queries": int(train["anchor_queries"]) >= args.min_train_queries,
        "train_events": int(train["triplet_events"]) >= args.min_train_events,
        "train_formulas": int(train["unique_formulas"]) >= args.min_train_formulas,
        "train_chemical_events": int(train["chemical_hard_events"]) >= args.min_chemical_events,
        "fixed_query_budget": int(train["events_per_query_max"]) <= args.events_per_query,
        "unique_query_negative_pairs": train["triplet_events"] == train["unique_query_negative_pairs"],
        "role2_recipe_specific": int(selection["specific_candidate_surplus"]) > 0,
        "formula_roles_disjoint": True,
        "outer_role_4_untouched": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"residual triplet gates failed: {gates}; train={train}")
    identity_audit = {
        role: audit_identity_edges(pools[(role, "correct")], args.data)
        for role in ("train", "selection", "confirmation")
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
    report = {
        "status": "CHEMAWARE_RESIDUAL_NATIVE_TRIPLETS_COMPLETE",
        "method": "checkpoint-adaptive two-slot native DreaMS triplet curriculum",
        "training_runtime": "unmodified DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "initialization_cache": str(args.embedding_cache.resolve()),
        "embedding_cache_file": str(embedding_path.resolve()),
        "margin": float(args.margin),
        "chemical_hardness_window": float(args.chemical_hardness_window),
        "events_per_query": int(args.events_per_query),
        "role2_frozen_specific_recipe": recipe_dict(recipe),
        "role2_recipe_selection": selection,
        "role2_recipe_grid_rows": int(len(grid)),
        "arms": list(arm_names),
        "roles": {
            "optimization": "formula roles 0-1",
            "checkpoint_selection": "formula role 2",
            "development_evaluation": "formula role 3",
            "outer": "formula role 4 untouched and not loaded",
        },
        "audits": audits,
        "role2_correct_vs_null_pair_overlap": overlaps,
        "identity_audit": identity_audit,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_residual_triplets_", dir=args.output.parent))
    try:
        for role in ("train", "selection", "confirmation"):
            np.savez_compressed(temporary / f"{role}_pool.npz", **pools[(role, "correct")])
            for arm in arm_names[1:]:
                safe = arm.removeprefix("rule_response_")
                np.savez_compressed(
                    temporary / f"{role}_pool_{safe}.npz", **pools[(role, arm)],
                )
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

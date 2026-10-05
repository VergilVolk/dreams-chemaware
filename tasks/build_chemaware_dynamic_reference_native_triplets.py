"""Expand qualified chemical candidate relations into active native references.

The chemical source still selects only an identity-level ``(query, true,
false)`` relation.  This builder then uses the native DreaMS reference-sampling
semantics: for each qualified relation it retains the highest-similarity true
reference spectra that still have at least one positive triplet hinge, and the
highest-similarity false references active against each retained positive.
Every sampled combination therefore has non-zero loss under the frozen Phase-A
geometry.  No chemical score, teacher target, loss weight or custom optimizer
enters training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from build_chemaware_max_boundary_native_triplets import FrozenEmbeddings, PoolWriter
from build_chemaware_multisource_native_triplets import pair_evidence, read_source_ledgers
from build_chemaware_sirius_native_triplets import copy_phasea, load_npz, verify_phasea_prefix


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_ERROR_REFERENCE = 12
DYNAMIC_MARGIN_REFERENCE = 13
MATCHED_CONTROL_POLICIES = {"pairwise_dominance", "family_qualified"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, action="append", required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument("--embedding-rows", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--embedding-report", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--maximum-candidates-per-query", type=int, default=6)
    parser.add_argument("--maximum-positive-references-per-relation", type=int, default=1)
    parser.add_argument("--maximum-negative-references-per-event", type=int, default=1)
    parser.add_argument("--maximum-events-per-formula", type=int, default=8)
    parser.add_argument("--maximum-events-per-identity", type=int, default=3)
    parser.add_argument("--minimum-supporting-sources", type=int, default=1)
    parser.add_argument(
        "--matched-control-policy", choices=sorted(MATCHED_CONTROL_POLICIES),
        default="pairwise_dominance",
        help=(
            "pairwise_dominance preserves the historical per-relation null test; "
            "family_qualified treats matched controls as the formula-disjoint source-family "
            "qualification gate and retains every strictly positive admitted-family delta"
        ),
    )
    parser.add_argument(
        "--geometry-label", default="protected Phase-A +2.1266 pp checkpoint",
    )
    return parser.parse_args()


def active_reference_events(
    positive_rows: np.ndarray,
    positive_scores: np.ndarray,
    negative_rows: np.ndarray,
    negative_scores: np.ndarray,
    margin: float,
    maximum_positive: int,
    maximum_negative: int,
) -> list[dict[str, object]]:
    """Return one active native event for one distinct false candidate.

    Positives are considered from highest to lowest similarity, so a lower-
    quality hard positive is never preferred over a higher-similarity positive
    that already yields a gradient.  The hardest active negative is paired
    with it.  Extra references of the same false candidate are deliberately
    not converted into extra training evidence.
    """
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_rows = np.asarray(negative_rows, dtype=np.int64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if len(positive_rows) != len(positive_scores) or len(negative_rows) != len(negative_scores):
        raise ValueError("reference rows and scores have different cardinalities")
    positive_order = np.lexsort((positive_rows, -positive_scores))
    negative_order = np.lexsort((negative_rows, -negative_scores))
    if maximum_positive != 1 or maximum_negative != 1:
        raise ValueError(
            "one relation must yield exactly one positive and one negative reference"
        )
    for positive_index in positive_order:
        hinge = margin + negative_scores - positive_scores[int(positive_index)]
        active = [int(index) for index in negative_order if hinge[int(index)] > 0]
        if not active:
            continue
        negative_index = active[0]
        return [{
            "positive_row": int(positive_rows[int(positive_index)]),
            "positive_score": float(positive_scores[int(positive_index)]),
            "negative_rows": np.asarray([negative_rows[negative_index]], dtype=np.int64),
            "negative_scores": np.asarray(
                [negative_scores[negative_index]], dtype=np.float64,
            ),
            "hinges": np.asarray([hinge[negative_index]], dtype=np.float64),
        }]
    return []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def validate_embedding_provenance(
    report_path: Path, checkpoint: Path, manifest: Path,
    rows: np.ndarray, embeddings: np.ndarray,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "status": "CHEMAWARE_CHECKPOINT_MANIFEST_EMBEDDINGS_COMPLETE",
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(embeddings),
        "formula_role_4_accessed": False,
    }
    drift = {key: (report.get(key), value) for key, value in expected.items()
             if report.get(key) != value}
    if drift:
        raise RuntimeError(f"embedding provenance drift: {drift}")
    return report


def family_qualified_pair_evidence(
    candidates: Mapping[int, Mapping[str, Mapping[str, object]]],
    families: Mapping[str, Mapping[str, object]],
    true_candidate: int,
    false_candidate: int,
) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    """Use admitted-family direction without reusing nulls as sample labels.

    Every family reaching this function has already passed the formula-disjoint
    matched-control qualification enforced by ``read_source_ledgers``.  The
    controls establish that the *family* carries chemical specificity.  At the
    relation level we therefore require only the chemically meaningful strict
    ordering ``score(true) > score(false)``.  The historical, more conservative
    per-pair null dominance remains available as a confidence tier and as the
    default policy; it is not silently weakened for other pipelines.
    """
    supports_with_rank: list[tuple[str, float, int]] = []
    opposes_with_rank: list[tuple[str, int]] = []
    for family in sorted(families):
        if family not in candidates.get(true_candidate, {}) or family not in candidates.get(false_candidate, {}):
            continue
        true = candidates[true_candidate][family]
        false = candidates[false_candidate][family]
        true_score, false_score = float(true["score"]), float(false["score"])
        if not (np.isfinite(true_score) and np.isfinite(false_score)):
            continue
        same_formula = str(true["formula"]) == str(false["formula"])
        scope = str(true["scope"])
        applicable = (
            scope == "all_candidates"
            or (scope == "cross_formula" and not same_formula)
            or (scope == "within_formula" and same_formula)
        )
        if not applicable:
            continue
        tier = str(families[family]["confidence_tier"])[0]
        # read_source_ledgers has already validated the admitted tier; use its
        # ordering without importing private module constants here.
        rank = {"A": 3, "B": 2, "C": 1}[tier]
        delta = true_score - false_score
        if delta > 0:
            supports_with_rank.append((family, delta, rank))
        else:
            opposes_with_rank.append((family, rank))
    if not supports_with_rank:
        return [], [family for family, _ in opposes_with_rank], []
    strongest_support = max(rank for _, _, rank in supports_with_rank)
    blocking = [family for family, rank in opposes_with_rank if rank >= strongest_support]
    nonblocking = [family for family, rank in opposes_with_rank if rank < strongest_support]
    return (
        [(family, delta) for family, delta, _ in supports_with_rank],
        blocking,
        nonblocking,
    )


def relation_evidence(
    candidates: Mapping[int, Mapping[str, Mapping[str, object]]],
    families: Mapping[str, Mapping[str, object]],
    true_candidate: int,
    false_candidate: int,
    matched_control_policy: str,
) -> tuple[list[tuple[str, float]], list[str], list[str], bool]:
    """Return relation evidence plus its strict per-pair confidence flag."""
    strict_supports, strict_blocking, strict_nonblocking = pair_evidence(
        candidates, families, true_candidate, false_candidate,
    )
    strict = bool(strict_supports) and not strict_blocking
    if matched_control_policy == "pairwise_dominance":
        return strict_supports, strict_blocking, strict_nonblocking, strict
    if matched_control_policy != "family_qualified":
        raise ValueError(f"unknown matched-control policy: {matched_control_policy}")
    supports, blocking, nonblocking = family_qualified_pair_evidence(
        candidates, families, true_candidate, false_candidate,
    )
    return supports, blocking, nonblocking, strict


def reference_geometry(
    manifest: Mapping[str, np.ndarray], cache: FrozenEmbeddings, query: int,
) -> dict[str, object]:
    anchor = int(manifest["query_row"][query])
    anchor_embedding = cache.get([anchor])[0]
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
    truth = np.flatnonzero(labels)
    if len(truth) != 1:
        raise RuntimeError(f"query {query} does not have exactly one true candidate")
    rows_by_candidate: dict[int, np.ndarray] = {}
    scores_by_candidate: dict[int, np.ndarray] = {}
    for candidate in range(right - left):
        rows = np.unique(molecule_rows(manifest, query, candidate))
        if candidate == int(truth[0]):
            rows = rows[rows != anchor]
        if not len(rows):
            raise RuntimeError(f"query {query} candidate {candidate} has no usable reference")
        rows_by_candidate[candidate] = rows
        scores_by_candidate[candidate] = cache.get(rows) @ anchor_embedding
    return {
        "anchor": anchor,
        "truth": int(truth[0]),
        "rows": rows_by_candidate,
        "scores": scores_by_candidate,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.margin <= 0
        or args.maximum_candidates_per_query < 1
        or args.maximum_positive_references_per_relation < 1
        or args.maximum_negative_references_per_event < 1
        or args.maximum_events_per_formula < 1
        or args.maximum_events_per_identity < 1
        or args.minimum_supporting_sources < 1
    ):
        raise ValueError("dynamic-reference budgets and margin must be positive")

    phasea = load_npz(args.phasea_pool)
    manifest = load_npz(args.manifest)
    scores, families = read_source_ledgers(args.source_ledger)
    cache = FrozenEmbeddings(args.embedding_rows, args.phasea_embeddings)
    embedding_provenance = validate_embedding_provenance(
        args.embedding_report, args.geometry_checkpoint, args.manifest,
        np.load(args.embedding_rows, mmap_mode="r"), cache.embeddings,
    )
    writer = PoolWriter()
    copy_phasea(phasea, writer)
    phasea_events = len(writer.anchor)
    records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    conflict_patterns: Counter[str] = Counter()
    audit = {
        "qualified_candidate_relations": 0,
        "relations_with_active_references": 0,
        "relations_rejected_by_source_conflict": 0,
        "relations_rejected_by_source_support": 0,
        "events_colliding_with_phasea": 0,
        "pairwise_control_dominant_relations": 0,
        "family_qualified_positive_relations": 0,
        "family_qualified_relations_added_beyond_pairwise_control": 0,
        "relations_pruned_by_formula_cap": 0,
        "relations_pruned_by_identity_cap": 0,
    }
    current_error_queries: set[int] = set()
    source_correctable_winners: set[int] = set()
    pending: list[dict[str, object]] = []

    for query in sorted(scores):
        if query < 0 or query >= len(manifest["query_row"]):
            raise RuntimeError(f"source ledger query out of manifest range: {query}")
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        candidate_count = right - left
        for candidate, source_body in scores[query].items():
            if candidate < 0 or candidate >= candidate_count:
                raise RuntimeError(f"source candidate out of range: query={query} candidate={candidate}")
            molecule = left + candidate
            for body in source_body.values():
                if str(manifest["molecule_ik14"][molecule]) != str(body["ik14"]):
                    raise RuntimeError(f"source IK14 drift: query={query} candidate={candidate}")
                if str(manifest["molecule_formula"][molecule]) != str(body["formula"]):
                    raise RuntimeError(f"source formula drift: query={query} candidate={candidate}")

        geometry = reference_geometry(manifest, cache, query)
        truth = int(geometry["truth"])
        rows_by_candidate = geometry["rows"]
        scores_by_candidate = geometry["scores"]
        assert isinstance(rows_by_candidate, dict) and isinstance(scores_by_candidate, dict)
        true_best = float(np.max(scores_by_candidate[truth]))
        false_candidates = [candidate for candidate in range(candidate_count) if candidate != truth]
        winner = max(false_candidates, key=lambda candidate: np.max(scores_by_candidate[candidate]))
        current_error = float(np.max(scores_by_candidate[winner])) >= true_best
        if current_error:
            current_error_queries.add(query)

        relations: list[dict[str, object]] = []
        for candidate in false_candidates:
            if candidate not in scores[query] or truth not in scores[query]:
                continue
            supports, blocking, nonblocking, strict_pairwise = relation_evidence(
                scores[query], families, truth, candidate,
                args.matched_control_policy,
            )
            if blocking:
                audit["relations_rejected_by_source_conflict"] += 1
                pattern = ",".join(sorted(blocking))
                conflict_patterns[pattern] += 1
                rejected_records.append({
                    "manifest_query": query,
                    "true_candidate": truth,
                    "false_candidate": candidate,
                    "reason": "blocking_source_conflict",
                    "supporting_sources": ",".join(family for family, _ in supports),
                    "blocking_sources": pattern,
                    "nonblocking_sources": ",".join(nonblocking),
                })
                continue
            if len(supports) < args.minimum_supporting_sources:
                audit["relations_rejected_by_source_support"] += 1
                rejected_records.append({
                    "manifest_query": query,
                    "true_candidate": truth,
                    "false_candidate": candidate,
                    "reason": "insufficient_pairwise_control_support",
                    "supporting_sources": ",".join(family for family, _ in supports),
                    "blocking_sources": "",
                    "nonblocking_sources": ",".join(nonblocking),
                })
                continue
            audit["qualified_candidate_relations"] += 1
            audit["family_qualified_positive_relations"] += 1
            if strict_pairwise:
                audit["pairwise_control_dominant_relations"] += 1
            else:
                audit["family_qualified_relations_added_beyond_pairwise_control"] += 1
            events = active_reference_events(
                rows_by_candidate[truth], scores_by_candidate[truth],
                rows_by_candidate[candidate], scores_by_candidate[candidate],
                args.margin, args.maximum_positive_references_per_relation,
                args.maximum_negative_references_per_event,
            )
            if not events:
                continue
            audit["relations_with_active_references"] += 1
            relations.append({
                "candidate": candidate,
                "supports": supports,
                "nonblocking": nonblocking,
                "events": events,
                "is_winner": candidate == winner,
                "best_negative": float(np.max(scores_by_candidate[candidate])),
                "strict_pairwise_control_dominance": strict_pairwise,
            })
        relations.sort(key=lambda body: (
            -int(bool(body["is_winner"])),
            -int(bool(body["strict_pairwise_control_dominance"])),
            -len(body["supports"]),
            -float(body["best_negative"]),
            int(body["candidate"]),
        ))
        selected_relations = relations[:args.maximum_candidates_per_query]
        for relation in selected_relations:
            candidate = int(relation["candidate"])
            supports = relation["supports"]
            for event in relation["events"]:
                pending.append({
                    "query": query, "truth": truth, "candidate": candidate,
                    "anchor": int(geometry["anchor"]), "current_error": current_error,
                    "is_winner": bool(relation["is_winner"]), "supports": supports,
                    "nonblocking": relation["nonblocking"], "event": event,
                    "strict": bool(relation["strict_pairwise_control_dominance"]),
                    "best_negative": float(relation["best_negative"]),
                    "formula": str(manifest["query_formula"][query]),
                    "identity": str(manifest["query_ik14"][query]),
                })

    # Allocate scarce training mass to distinct chemical relations, not to the
    # arbitrary manifest order or repeated reference spectra.
    pending.sort(key=lambda body: (
        -int(bool(body["current_error"]) and bool(body["is_winner"])),
        -int(bool(body["current_error"])),
        -int(bool(body["strict"])),
        -len(body["supports"]),
        -float(body["best_negative"]),
        int(body["query"]), int(body["candidate"]),
    ))
    formula_count: dict[str, int] = {}
    identity_count: dict[str, int] = {}
    for body in pending:
        formula = str(body["formula"])
        identity = str(body["identity"])
        if formula_count.get(formula, 0) >= args.maximum_events_per_formula:
            audit["relations_pruned_by_formula_cap"] += 1
            continue
        if identity_count.get(identity, 0) >= args.maximum_events_per_identity:
            audit["relations_pruned_by_identity_cap"] += 1
            continue
        query = int(body["query"])
        candidate = int(body["candidate"])
        event = body["event"]
        positive = int(event["positive_row"])
        negatives = np.asarray(event["negative_rows"], dtype=np.int64)
        before = len(writer.anchor)
        writer.append(
            int(body["anchor"]), [positive], negatives, query, candidate, 0,
            DYNAMIC_ERROR_REFERENCE if bool(body["current_error"])
            else DYNAMIC_MARGIN_REFERENCE,
        )
        if len(writer.anchor) == before:
            audit["events_colliding_with_phasea"] += 1
            continue
        formula_count[formula] = formula_count.get(formula, 0) + 1
        identity_count[identity] = identity_count.get(identity, 0) + 1
        if bool(body["current_error"]) and bool(body["is_winner"]):
            source_correctable_winners.add(query)
        hinges = np.asarray(event["hinges"], dtype=np.float64)
        supports = body["supports"]
        records.append({
            "manifest_query": query,
            "true_candidate": int(body["truth"]),
            "false_candidate": candidate,
            "positive_row": positive,
            "negative_rows": ";".join(map(str, negatives)),
            "sampled_triplet_capacity": len(negatives),
            "support_count": len(supports),
            "supporting_sources": ",".join(family for family, _ in supports),
            "source_deltas": ",".join(f"{delta:.9g}" for _, delta in supports),
            "source_confidence_tier": (
                "pairwise_control_dominant" if bool(body["strict"])
                else "formula_disjoint_family_qualified_positive"
            ),
            "nonblocking_lower_tier_opposition": ",".join(body["nonblocking"]),
            "positive_score": f"{float(event['positive_score']):.17g}",
            "minimum_negative_score": f"{float(np.min(event['negative_scores'])):.17g}",
            "minimum_native_hinge": f"{float(np.min(hinges)):.17g}",
            "maximum_native_hinge": f"{float(np.max(hinges)):.17g}",
            "phasea_current_error": int(bool(body["current_error"])),
        })

    output = writer.arrays()
    chemical_slice = slice(phasea_events, len(output["anchor_idx"]))
    positive_multiplicity = np.diff(output["positive_ptr"])[chemical_slice]
    negative_multiplicity = np.diff(output["negative_ptr"])[chemical_slice]
    gates = {
        "phasea_pool_is_exact_prefix": verify_phasea_prefix(phasea, output),
        "one_positive_reference_per_dynamic_event": bool(np.all(positive_multiplicity == 1)),
        "bounded_negative_reference_multiplicity": bool(np.all(
            negative_multiplicity == 1
        )),
        "one_event_per_distinct_false_candidate": len(records) == len({
            (int(row["manifest_query"]), int(row["false_candidate"])) for row in records
        }),
        "formula_event_cap_respected": max(formula_count.values(), default=0) <= args.maximum_events_per_formula,
        "identity_event_cap_respected": max(identity_count.values(), default=0) <= args.maximum_events_per_identity,
        "every_dynamic_reference_pair_active_at_mining_geometry": bool(all(
            float(row["minimum_native_hinge"]) > 0 for row in records
        )),
        "no_numerical_source_fusion": True,
        "validation_pool_unchanged": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"dynamic-reference native-triplet gates failed: {gates}")

    report = {
        "status": "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE",
        "initialization": args.geometry_label,
        "embedding_provenance": embedding_provenance,
        "source_families": families,
        "phasea_events_preserved": phasea_events,
        "dynamic_chemical_events_added": len(records),
        "dynamic_chemical_queries": len({int(row["manifest_query"]) for row in records}),
        "dynamic_chemical_formulas": len({
            str(manifest["query_formula"][int(row["manifest_query"])]) for row in records
        }),
        "qualified_candidate_relations_selected": len({
            (int(row["manifest_query"]), int(row["false_candidate"])) for row in records
        }),
        "sampled_native_triplet_capacity": sum(
            int(row["sampled_triplet_capacity"]) for row in records
        ),
        "current_error_queries": len(current_error_queries),
        "source_correctable_current_winners": len(source_correctable_winners),
        "reference_multiplicity": {
            "maximum_positive_references_per_relation": args.maximum_positive_references_per_relation,
            "maximum_negative_references_per_event": args.maximum_negative_references_per_event,
            "maximum_candidates_per_query": args.maximum_candidates_per_query,
            "maximum_events_per_formula": args.maximum_events_per_formula,
            "maximum_events_per_identity": args.maximum_events_per_identity,
        },
        "matched_control_policy": args.matched_control_policy,
        "audit": audit,
        "source_conflict_patterns": dict(sorted(conflict_patterns.items())),
        "identity_audit": audit_identity_edges(output, args.data),
        "gates": gates,
        "scientific_contract": {
            "changed_component": "qualified triplet reference selection only",
            "phasea_pool": "copied exactly before dynamic chemical events",
            "source": "formula-disjoint confirmed truth-blind candidate relation evidence",
            "matched_control_use": (
                "source-family qualification plus an explicit per-relation confidence tier; "
                "matched null scores are not reused as candidate labels"
                if args.matched_control_policy == "family_qualified"
                else "source-family qualification and strict per-relation null dominance"
            ),
            "reference_selection": (
                "highest-similarity true references with active native hinge; highest-similarity "
                "active false references; every sampled pair is active"
            ),
            "loss": "unchanged DreaMS native triplet loss with unit event weights",
            "teacher_or_distillation": False,
            "chemical_score_used_as_training_target": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_dynamic_reference_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        with (temporary / "dynamic_source_event_ledger.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer_csv = csv.DictWriter(
                handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n",
            )
            writer_csv.writeheader()
            writer_csv.writerows(records)
        with (temporary / "rejected_relation_ledger.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            fieldnames = [
                "manifest_query", "true_candidate", "false_candidate", "reason",
                "supporting_sources", "blocking_sources", "nonblocking_sources",
            ]
            writer_csv = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n",
            )
            writer_csv.writeheader()
            writer_csv.writerows(rejected_records)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

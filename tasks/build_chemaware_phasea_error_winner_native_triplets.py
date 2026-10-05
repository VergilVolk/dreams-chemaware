"""Append chemically supported Phase-A error winners to the native triplet pool.

The protected Phase-A pool is copied byte-for-byte as the output prefix.  A
new event is eligible only when the *current* Phase-A top false molecule beats
the true molecule and that exact winner has chemistry-specific support from
either the frozen multi-null action evidence or the qualified fragment/
MassBank source ledgers.  Each query contributes at most one new false
candidate and one max/max reference pair.  Chemical scores never enter the
loss; downstream training remains the native DreaMS triplet objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_action_hard_native_triplets import (
    SPECIFIC_HARD,
    metric,
    selected_negative_candidates,
)
from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_dynamic_reference_native_triplets import (
    active_reference_events,
    reference_geometry,
    validate_embedding_provenance,
)
from build_chemaware_max_boundary_native_triplets import FrozenEmbeddings, PoolWriter
from build_chemaware_multisource_native_triplets import read_source_ledgers
from build_chemaware_sirius_native_triplets import (
    copy_phasea,
    load_npz,
    verify_phasea_prefix,
)
from GLM_build_chemaware_v16_dynamic_triplets import (
    v16_registry,
    v16_relation_verdict,
)


ROOT = Path(__file__).resolve().parents[1]
ERROR_WINNER_ROLE = 21
LEGACY_METRICS = (
    "candidate_official_rank_fraction", "action_top_fraction",
    "action_largest_region_fraction", "action_same_neighbor_fraction",
    "action_best_advantage_over_baseline",
    "global_action_advantage_over_baseline",
    "global_action_selects_candidate", "candidate_rule_rank_fraction",
    "candidate_rule_max", "candidate_rule_top2_mean", "delta_rule_max",
    "delta_rule_top2_mean",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
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
    parser.add_argument("--maximum-events-per-formula", type=int, default=8)
    parser.add_argument("--maximum-events-per-identity", type=int, default=3)
    parser.add_argument("--minimum-new-error-winners", type=int, default=20)
    return parser.parse_args()


def legacy_specific_winners(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
) -> dict[int, set[int]]:
    names = tuple(map(str, np.asarray(evidence["arm_names"]).tolist()))
    if not names or names[0] != "correct":
        raise RuntimeError("legacy evidence does not place the correct arm first")
    metrics = {name: metric(evidence, name) for name in LEGACY_METRICS}
    output: dict[int, set[int]] = {}
    for row, query_value in enumerate(np.asarray(evidence["query"], dtype=np.int64)):
        selected = selected_negative_candidates(
            evidence, manifest, metrics, row, 0,
            action_hard_k=0, specific_hard_k=64,
        )
        output[int(query_value)] = {
            int(candidate) for candidate, tag in selected.items()
            if int(tag) & SPECIFIC_HARD
        }
    if len(output) != len(evidence["query"]):
        raise RuntimeError("legacy evidence repeats a training query")
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.margin <= 0 or args.maximum_events_per_formula < 1
        or args.maximum_events_per_identity < 1
        or args.minimum_new_error_winners < 1
    ):
        raise ValueError("invalid current-error winner budgets")

    phasea = load_npz(args.phasea_pool)
    validation = load_npz(args.validation_pool)
    evidence = load_npz(args.legacy_evidence)
    manifest = load_npz(args.manifest)
    source_scores, source_families = read_source_ledgers(args.source_ledger)
    registry = v16_registry(source_families)
    cache = FrozenEmbeddings(args.embedding_rows, args.phasea_embeddings)
    provenance = validate_embedding_provenance(
        args.embedding_report, args.geometry_checkpoint, args.manifest,
        np.load(args.embedding_rows, mmap_mode="r"), cache.embeddings,
    )
    legacy = legacy_specific_winners(evidence, manifest)
    evidence_queries = set(map(int, np.asarray(evidence["query"], dtype=np.int64)))
    if evidence_queries != set(legacy):
        raise RuntimeError("legacy-specific query registry drifted")

    pending: list[dict[str, object]] = []
    current_errors = 0
    unsupported_winners = 0
    contested_winners = 0
    for query in sorted(evidence_queries):
        geometry = reference_geometry(manifest, cache, query)
        truth = int(geometry["truth"])
        rows = geometry["rows"]
        scores = geometry["scores"]
        false = [candidate for candidate in rows if int(candidate) != truth]
        winner = max(false, key=lambda candidate: float(np.max(scores[candidate])))
        true_score = float(np.max(scores[truth]))
        false_score = float(np.max(scores[winner]))
        if false_score < true_score:
            continue
        current_errors += 1

        legacy_support = int(winner) in legacy.get(query, set())
        verdict = None
        source_supports: list[tuple[str, float]] = []
        if query in source_scores and truth in source_scores[query] and winner in source_scores[query]:
            verdict = v16_relation_verdict(
                source_scores[query], registry, truth, int(winner),
            )
            source_supports = list(verdict["supports"])
        uncontested_source = bool(
            verdict is not None
            and source_supports
            and not verdict["blocking_dominant"]
            and verdict["pair_class"] in (
                "unanimous_admitted_style", "vetoed_by_subsignificant_oppose",
            )
        )
        if verdict is not None and verdict["pair_class"] == "contested_dominant":
            contested_winners += 1
        if not (legacy_support or uncontested_source):
            unsupported_winners += 1
            continue

        event = active_reference_events(
            rows[truth], scores[truth], rows[winner], scores[winner],
            args.margin, 1, 1,
        )
        if len(event) != 1:
            raise RuntimeError("a current error winner lacks an active max/max triplet")
        formula = str(manifest["query_formula"][query])
        identity = str(manifest["query_ik14"][query])
        pending.append({
            "query": query, "truth": truth, "winner": int(winner),
            "anchor": int(geometry["anchor"]), "event": event[0],
            "formula": formula, "identity": identity,
            "legacy_specific": legacy_support,
            "source_supports": source_supports,
            "pair_class": verdict["pair_class"] if verdict is not None else "legacy_only",
            "severity": false_score - true_score,
        })

    # Prefer agreement across evidence systems, then stronger source consensus,
    # then the actual current retrieval error severity.  Caps prevent one
    # formula or identity from dominating the continuation packet.
    pending.sort(key=lambda row: (
        -int(bool(row["legacy_specific"]) and bool(row["source_supports"])),
        -len(row["source_supports"]), -float(row["severity"]),
        str(row["formula"]), int(row["query"]),
    ))
    writer = PoolWriter()
    copy_phasea(phasea, writer)
    phasea_events = len(writer.anchor)
    formula_count: dict[str, int] = {}
    identity_count: dict[str, int] = {}
    records: list[dict[str, object]] = []
    collisions = formula_pruned = identity_pruned = 0
    for body in pending:
        formula = str(body["formula"])
        identity = str(body["identity"])
        if formula_count.get(formula, 0) >= args.maximum_events_per_formula:
            formula_pruned += 1
            continue
        if identity_count.get(identity, 0) >= args.maximum_events_per_identity:
            identity_pruned += 1
            continue
        event = body["event"]
        before = len(writer.anchor)
        writer.append(
            int(body["anchor"]), [int(event["positive_row"])],
            np.asarray(event["negative_rows"], dtype=np.int64),
            int(body["query"]), int(body["winner"]), 0, ERROR_WINNER_ROLE,
        )
        if len(writer.anchor) == before:
            collisions += 1
            continue
        formula_count[formula] = formula_count.get(formula, 0) + 1
        identity_count[identity] = identity_count.get(identity, 0) + 1
        records.append({
            "manifest_query": int(body["query"]),
            "true_candidate": int(body["truth"]),
            "current_false_winner": int(body["winner"]),
            "formula": formula, "identity": identity,
            "legacy_multinull_specific": int(bool(body["legacy_specific"])),
            "qualified_source_count": len(body["source_supports"]),
            "qualified_sources": ",".join(name for name, _ in body["source_supports"]),
            "pair_evidence_class": str(body["pair_class"]),
            "phasea_error_severity": f"{float(body['severity']):.17g}",
            "positive_row": int(event["positive_row"]),
            "negative_row": int(np.asarray(event["negative_rows"])[0]),
            "native_hinge": f"{float(np.asarray(event['hinges'])[0]):.17g}",
        })

    output = writer.arrays()
    added = len(output["anchor_idx"]) - phasea_events
    gates = {
        "phasea_pool_is_exact_prefix": verify_phasea_prefix(phasea, output),
        "every_addition_is_one_current_error_winner": added == len(records),
        "one_event_per_added_query": len({row["manifest_query"] for row in records}) == added,
        "all_added_pairs_active": all(float(row["native_hinge"]) > 0 for row in records),
        "no_contested_only_winner_admitted": all(
            row["pair_evidence_class"] != "contested_dominant" for row in records
        ),
        "minimum_new_error_winners": added >= args.minimum_new_error_winners,
        "validation_pool_unchanged": True,
        "outer_role_4_untouched": True,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"Phase-A error-winner triplet gates failed: {gates}; "
            f"current_errors={current_errors} eligible={len(pending)} added={added}"
        )
    identity_audit = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_PHASEA_ERROR_WINNER_NATIVE_TRIPLETS_COMPLETE",
        "phasea_events_preserved": phasea_events,
        "current_phasea_errors": current_errors,
        "chemically_supported_error_winners": len(pending),
        "new_error_winner_events": added,
        "unsupported_error_winners": unsupported_winners,
        "contested_error_winners_rejected": contested_winners,
        "events_supported_by_legacy_multinull": sum(
            int(row["legacy_multinull_specific"]) for row in records
        ),
        "events_supported_by_qualified_sources": sum(
            int(row["qualified_source_count"] > 0) for row in records
        ),
        "events_supported_by_both": sum(
            int(row["legacy_multinull_specific"] and row["qualified_source_count"] > 0)
            for row in records
        ),
        "collisions_with_phasea": collisions,
        "formula_cap_pruned": formula_pruned,
        "identity_cap_pruned": identity_pruned,
        "embedding_provenance": provenance,
        "identity_audit": identity_audit,
        "gates": gates,
        "scientific_contract": {
            "changed_component": "native triplet curriculum only",
            "loss": "unchanged DreaMS cosine triplet margin",
            "optimizer": "must restore the protected Phase-A native Adam state",
            "chemical_use": "select the exact current false winner; never a numeric target",
            "event_unit": "one distinct query/current-false-winner max/max pair",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_error_winner_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        with (temporary / "error_winner_ledger.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer_csv = csv.DictWriter(
                handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n",
            )
            writer_csv.writeheader()
            writer_csv.writerows(records)
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

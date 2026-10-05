"""GLM V16 pre-registered dynamic-triplet builder: symmetric-significance veto.

Scientific rationale (pre-registered before the evidence-class diagnostic is
run on the server): the production dynamic builder discards a relation when
ANY tier-qualifying source opposes on the raw sign of its delta, even though
support must clear a much stronger bar (beat every matched-control delta).
The veto is therefore asymmetric.  V16 makes the veto symmetric: an
opposition blocks only when it is dominantly significant under the same
standard support must meet -- the reversed delta must beat every reversed
control delta.  Sub-significant opposition is recorded, never silently
dropped.  Relations where a dominant support AND a dominant blocking
opposition coexist (chemically real two-sided conflict) are routed to a
separate contested arm pool instead of being discarded, so the conflict mass
is trainable as an explicitly labelled experiment rather than lost.

Launch gate (executable pre-registration): the builder refuses to produce any
pool unless the frozen-geometry evidence-class diagnostic (GLM_diagnose_)
reports (a) matching sanity anchors, (b) decision eligibility, and (c) at
least --minimum-launch-boundaries error boundaries in
vetoed_by_subsignificant_oppose + contested_dominant, and its provenance
hashes match the manifest and every source ledger passed here.

Arm A (this builder's main output) is a strict superset of the production
admitted mass under the same reference-sampling semantics: every relation the
production builder admits is unanimous here; the increment is exactly the
sub-significant-veto recovery.  Identity supervision is unchanged native
DreaMS triplet training; no chemical score enters the loss.
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

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_dynamic_reference_native_triplets import (
    active_reference_events,
    reference_geometry,
    validate_embedding_provenance,
)
from build_chemaware_max_boundary_native_triplets import FrozenEmbeddings, PoolWriter
from build_chemaware_multisource_native_triplets import (
    CONFIDENCE_RANKS,
    read_source_ledgers,
)
from build_chemaware_sirius_native_triplets import (
    copy_phasea,
    load_npz,
    verify_phasea_prefix,
)
from GLM_diagnose_chemaware_evidence_class_errors import (
    file_sha256,
    family_stance,
    classify_pair,
)

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_ERROR_REFERENCE = 12
DYNAMIC_MARGIN_REFERENCE = 13
V16_CONTESTED_REFERENCE = 14
DIAGNOSTIC_STATUS = "GLM_CHEMAWARE_EVIDENCE_CLASS_DIAGNOSTIC_COMPLETE"
V16_MINIMUM_LAUNCH_BOUNDARIES = 50


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
    parser.add_argument("--evidence-class-diagnostic", type=Path, required=True)
    parser.add_argument("--minimum-launch-boundaries", type=int,
                        default=V16_MINIMUM_LAUNCH_BOUNDARIES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contested-output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--maximum-candidates-per-query", type=int, default=6)
    parser.add_argument("--maximum-positive-references-per-relation", type=int, default=1)
    parser.add_argument("--maximum-negative-references-per-event", type=int, default=1)
    parser.add_argument("--maximum-events-per-formula", type=int, default=8)
    parser.add_argument("--maximum-events-per-identity", type=int, default=3)
    parser.add_argument("--minimum-supporting-sources", type=int, default=1)
    parser.add_argument(
        "--geometry-label", default="protected Phase-A +2.1266 pp checkpoint",
    )
    return parser.parse_args()


def enforce_launch_gate(
    diagnostic_path: Path, manifest_path: Path, ledger_paths: list[Path],
    minimum_launch_boundaries: int,
) -> dict[str, object]:
    """Executable pre-registration: refuse to build unless the frozen-geometry
    diagnostic authorizes the V16 route and describes exactly these inputs."""
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic.get("status") != DIAGNOSTIC_STATUS:
        raise RuntimeError(
            f"evidence-class diagnostic is not complete: {diagnostic_path}"
        )
    anchors = diagnostic.get("sanity_anchors", {})
    if not anchors.get("all_match", False):
        raise RuntimeError(
            "evidence-class diagnostic anchors do not match the preflight; "
            "geometry or ledger drift suspected"
        )
    if not diagnostic.get("decision_summary", {}).get("decision_eligible", False):
        raise RuntimeError("evidence-class diagnostic is not decision eligible")
    boundaries = diagnostic.get("error_boundary_counts", {})
    recoverable = int(boundaries.get("vetoed_by_subsignificant_oppose", 0))
    contested = int(boundaries.get("contested_dominant", 0))
    if recoverable + contested < minimum_launch_boundaries:
        raise RuntimeError(
            f"V16 launch gate unmet: recoverable({recoverable}) + contested({contested}) "
            f"< {minimum_launch_boundaries} error boundaries"
        )
    provenance = diagnostic.get("provenance", {})
    if provenance.get("manifest_sha256") != file_sha256(manifest_path):
        raise RuntimeError("diagnostic manifest provenance does not match this run")
    known = {
        (entry.get("report_sha256"), entry.get("tsv_sha256"))
        for entry in provenance.get("ledgers", [])
    }
    if not known:
        raise RuntimeError("diagnostic provenance carries no ledger hashes")
    ledger_digests: list[dict[str, str]] = []
    for path in ledger_paths:
        directory = path if path.is_dir() else path.parent
        report_sha = file_sha256(directory / "report.json")
        tsv_sha = file_sha256(directory / "candidate_scores.tsv")
        if (report_sha, tsv_sha) not in known:
            raise RuntimeError(
                f"source ledger was not part of the diagnostic: {directory}"
            )
        ledger_digests.append({
            "ledger": str(directory.resolve()),
            "report_sha256": report_sha,
            "tsv_sha256": tsv_sha,
        })
    return {
        "diagnostic": str(diagnostic_path.resolve()),
        "diagnostic_sha256": file_sha256(diagnostic_path),
        "minimum_launch_boundaries": int(minimum_launch_boundaries),
        "diagnostic_recoverable_boundaries": recoverable,
        "diagnostic_contested_boundaries": contested,
        "ledgers": ledger_digests,
    }


def v16_registry(
    families: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Adapt builder family contracts to the diagnostic stance engine."""
    registry: dict[str, dict[str, object]] = {}
    for name, body in families.items():
        tier = str(body["confidence_tier"])[0]
        if tier not in CONFIDENCE_RANKS:
            raise RuntimeError(f"invalid confidence tier for family {name}: {tier!r}")
        registry[name] = {
            "scope": str(body["scope"]),
            "confidence_tier": str(body["confidence_tier"]),
            "tier_rank": CONFIDENCE_RANKS[tier],
            "matched_controls": list(body.get("matched_controls") or []),
            "larger_is_better": True,
        }
    return registry


def v16_relation_verdict(
    candidates: Mapping[int, Mapping[str, Mapping[str, object]]],
    registry: Mapping[str, Mapping[str, object]],
    true_candidate: int, false_candidate: int,
) -> dict[str, object]:
    """Stances, five-class verdict, and the dominant/sub opposition split."""
    stances: dict[str, str] = {}
    for family in sorted(registry):
        if (
            family not in candidates.get(true_candidate, {})
            or family not in candidates.get(false_candidate, {})
        ):
            continue
        stances[family] = family_stance(
            candidates[true_candidate][family],
            candidates[false_candidate][family],
            family, registry,
        )
    pair_class = classify_pair(stances, registry)
    supports = sorted(
        (family, float(candidates[true_candidate][family]["score"])
         - float(candidates[false_candidate][family]["score"]))
        for family, stance in stances.items() if stance == "support"
    )
    dominant_opposition = sorted(
        family for family, stance in stances.items()
        if stance in ("oppose_dominant", "oppose_unassessed")
    )
    subsignificant_opposition = sorted(
        family for family, stance in stances.items()
        if stance == "oppose_subsignificant"
    )
    if supports:
        strongest = max(int(registry[family]["tier_rank"]) for family, _ in supports)
        blocking_subsignificant = [
            f for f in subsignificant_opposition
            if int(registry[f]["tier_rank"]) >= strongest
        ]
        blocking_dominant = [
            f for f in dominant_opposition
            if int(registry[f]["tier_rank"]) >= strongest
        ]
    else:
        blocking_subsignificant = []
        blocking_dominant = []
    support_names = [family for family, _ in supports]
    return {
        "stances": stances,
        "pair_class": pair_class,
        "supports": supports,
        "support_names": support_names,
        "blocking_dominant": blocking_dominant,
        "blocking_subsignificant": blocking_subsignificant,
        "nonblocking_opposition": sorted(
            (set(dominant_opposition) | set(subsignificant_opposition))
            - set(blocking_dominant) - set(blocking_subsignificant)
        ),
    }


def allocate_pending(
    pending: list[dict[str, object]],
    writer: PoolWriter,
    contested_role: int | None,
    margin_roles: tuple[int, int],
) -> tuple[list[dict[str, object]], dict[str, int], set[int]]:
    """Shared allocation pass: formula/identity caps, dedup, roles."""
    records: list[dict[str, object]] = []
    formula_count: dict[str, int] = {}
    identity_count: dict[str, int] = {}
    correctable: set[int] = set()
    pruned_formula = pruned_identity = collisions = 0
    for body in pending:
        formula = str(body["formula"])
        identity = str(body["identity"])
        if formula_count.get(formula, 0) >= body["formula_cap"]:
            pruned_formula += 1
            continue
        if identity_count.get(identity, 0) >= body["identity_cap"]:
            pruned_identity += 1
            continue
        event = body["event"]
        positive = int(event["positive_row"])
        negatives = np.asarray(event["negative_rows"], dtype=np.int64)
        before = len(writer.anchor)
        role = (
            contested_role
            if contested_role is not None
            else (
                margin_roles[0] if bool(body["current_error"]) else margin_roles[1]
            )
        )
        writer.append(
            int(body["anchor"]), [positive], negatives,
            int(body["query"]), int(body["candidate"]), 0, role,
        )
        if len(writer.anchor) == before:
            collisions += 1
            continue
        formula_count[formula] = formula_count.get(formula, 0) + 1
        identity_count[identity] = identity_count.get(identity, 0) + 1
        if bool(body["current_error"]) and bool(body["is_winner"]):
            correctable.add(int(body["query"]))
        hinges = np.asarray(event["hinges"], dtype=np.float64)
        supports = body["supports"]
        verdict = body["verdict"]
        records.append({
            "manifest_query": int(body["query"]),
            "true_candidate": int(body["truth"]),
            "false_candidate": int(body["candidate"]),
            "positive_row": positive,
            "negative_rows": ";".join(map(str, negatives)),
            "sampled_triplet_capacity": len(negatives),
            "support_count": len(supports),
            "supporting_sources": ",".join(
                family for family, _ in supports
            ),
            "source_deltas": ",".join(
                f"{delta:.9g}" for _, delta in supports
            ),
            "pair_evidence_class": verdict["pair_class"],
            "v16_confidence_tier": (
                "uncontested_pairwise_dominant"
                if verdict["pair_class"] == "unanimous_admitted_style"
                else "recovered_subsignificant_veto"
                if verdict["pair_class"] == "vetoed_by_subsignificant_oppose"
                else "contested_dominant_arm"
            ),
            "blocking_dominant_opposition": ",".join(verdict["blocking_dominant"]),
            "blocking_subsignificant_opposition": ",".join(
                verdict["blocking_subsignificant"]
            ),
            "nonblocking_lower_tier_opposition": ",".join(
                verdict["nonblocking_opposition"]
            ),
            "positive_score": f"{float(event['positive_score']):.17g}",
            "minimum_negative_score": f"{float(np.min(event['negative_scores'])):.17g}",
            "minimum_native_hinge": f"{float(np.min(hinges)):.17g}",
            "maximum_native_hinge": f"{float(np.max(hinges)):.17g}",
            "phasea_current_error": int(bool(body["current_error"])),
        })
    audit = {
        "relations_pruned_by_formula_cap": pruned_formula,
        "relations_pruned_by_identity_cap": pruned_identity,
        "events_colliding_with_prior_pool": collisions,
    }
    return records, audit, correctable


def write_ledger(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if records:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]), delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
        else:
            handle.write("manifest_query\ttrue_candidate\tfalse_candidate\n")


def main() -> None:
    args = arguments()
    if args.output.exists() or args.contested_output.exists():
        raise FileExistsError("refusing to overwrite V16 outputs")
    if (
        args.margin <= 0
        or args.maximum_candidates_per_query < 1
        or args.maximum_positive_references_per_relation != 1
        or args.maximum_negative_references_per_event != 1
        or args.maximum_events_per_formula < 1
        or args.maximum_events_per_identity < 1
        or args.minimum_supporting_sources < 1
        or args.minimum_launch_boundaries < 0
    ):
        raise ValueError("V16 budgets, caps, and margin must be positive")

    launch_gate = enforce_launch_gate(
        args.evidence_class_diagnostic, args.manifest, args.source_ledger,
        args.minimum_launch_boundaries,
    )
    phasea = load_npz(args.phasea_pool)
    manifest = load_npz(args.manifest)
    scores, families = read_source_ledgers(args.source_ledger)
    registry = v16_registry(families)
    cache = FrozenEmbeddings(args.embedding_rows, args.phasea_embeddings)
    embedding_provenance = validate_embedding_provenance(
        args.embedding_report, args.geometry_checkpoint, args.manifest,
        np.load(args.embedding_rows, mmap_mode="r"), cache.embeddings,
    )

    writer = PoolWriter()
    contested_writer = PoolWriter()
    copy_phasea(phasea, writer)
    copy_phasea(phasea, contested_writer)
    phasea_events = len(writer.anchor)
    if len(contested_writer.anchor) != phasea_events:
        raise RuntimeError("contested-arm Phase-A prefix diverged")

    pending: list[dict[str, object]] = []
    contested_pending: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    audit = {
        "qualified_candidate_relations": 0,
        "relations_with_active_references": 0,
        "relations_rejected_no_dominant_support": 0,
        "relations_recovered_from_subsignificant_veto": 0,
        "relations_uncontested_admitted": 0,
        "relations_routed_to_contested_arm": 0,
        "relations_rejected_by_source_support": 0,
        "current_error_queries": 0,
    }
    current_error_queries: set[int] = set()

    for query in sorted(scores):
        if query < 0 or query >= len(manifest["query_row"]):
            raise RuntimeError(f"source ledger query out of manifest range: {query}")
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        candidate_count = right - left
        for candidate, source_body in scores[query].items():
            if candidate < 0 or candidate >= candidate_count:
                raise RuntimeError(
                    f"source candidate out of range: query={query} candidate={candidate}"
                )
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
        true_best = float(np.max(scores_by_candidate[truth]))
        false_candidates = [c for c in range(candidate_count) if c != truth]
        winner = max(
            false_candidates, key=lambda c: float(np.max(scores_by_candidate[c])),
        )
        current_error = float(np.max(scores_by_candidate[winner])) >= true_best
        if current_error:
            current_error_queries.add(query)

        admitted: list[dict[str, object]] = []
        contested: list[dict[str, object]] = []
        for candidate in false_candidates:
            if candidate not in scores[query] or truth not in scores[query]:
                continue
            verdict = v16_relation_verdict(
                scores[query], registry, truth, candidate,
            )
            pair_class = verdict["pair_class"]
            if pair_class in ("dominant_opposition_only", "no_dominant_signal"):
                audit["relations_rejected_no_dominant_support"] += 1
                rejected_records.append({
                    "manifest_query": query, "true_candidate": truth,
                    "false_candidate": candidate,
                    "reason": "no_dominant_support",
                    "pair_evidence_class": pair_class,
                    "supporting_sources": ",".join(verdict["support_names"]),
                    "blocking_sources": ",".join(
                        verdict["blocking_dominant"]
                        + verdict["blocking_subsignificant"]
                    ),
                    "nonblocking_sources": ",".join(verdict["nonblocking_opposition"]),
                })
                continue
            if len(verdict["supports"]) < args.minimum_supporting_sources:
                audit["relations_rejected_by_source_support"] += 1
                rejected_records.append({
                    "manifest_query": query, "true_candidate": truth,
                    "false_candidate": candidate,
                    "reason": "insufficient_dominant_support",
                    "pair_evidence_class": pair_class,
                    "supporting_sources": ",".join(verdict["support_names"]),
                    "blocking_sources": "", 
                    "nonblocking_sources": ",".join(verdict["nonblocking_opposition"]),
                })
                continue
            events = active_reference_events(
                rows_by_candidate[truth], scores_by_candidate[truth],
                rows_by_candidate[candidate], scores_by_candidate[candidate],
                args.margin, args.maximum_positive_references_per_relation,
                args.maximum_negative_references_per_event,
            )
            if not events:
                continue
            audit["qualified_candidate_relations"] += 1
            body = {
                "candidate": candidate, "verdict": verdict,
                "supports": list(verdict["supports"]),
                "events": events, "is_winner": candidate == winner,
                "best_negative": float(np.max(scores_by_candidate[candidate])),
            }
            if pair_class == "contested_dominant":
                audit["relations_routed_to_contested_arm"] += 1
                contested.append(body)
            else:
                audit["relations_with_active_references"] += 1
                if pair_class == "vetoed_by_subsignificant_oppose":
                    audit["relations_recovered_from_subsignificant_veto"] += 1
                else:
                    audit["relations_uncontested_admitted"] += 1
                admitted.append(body)

        for arm, arm_list in (("A", admitted), ("B", contested)):
            arm_list.sort(key=lambda body: (
                -int(bool(body["is_winner"])),
                -int(
                    bool(body["verdict"]["pair_class"]
                         == "unanimous_admitted_style")
                ),
                -len(body["supports"]),
                -float(body["best_negative"]),
                int(body["candidate"]),
            ))
            for relation in arm_list[:args.maximum_candidates_per_query]:
                for event in relation["events"]:
                    target = pending if arm == "A" else contested_pending
                    target.append({
                        "query": query, "truth": truth,
                        "candidate": int(relation["candidate"]),
                        "anchor": int(geometry["anchor"]),
                        "current_error": current_error,
                        "is_winner": bool(relation["is_winner"]),
                        "supports": relation["supports"],
                        "verdict": relation["verdict"],
                        "event": event,
                        "best_negative": float(relation["best_negative"]),
                        "formula": str(manifest["query_formula"][query]),
                        "identity": str(manifest["query_ik14"][query]),
                        "formula_cap": args.maximum_events_per_formula,
                        "identity_cap": args.maximum_events_per_identity,
                    })

    for body in pending + contested_pending:
        body["sort_key"] = (
            -int(bool(body["current_error"]) and bool(body["is_winner"])),
            -int(bool(body["current_error"])),
            -int(bool(
                body["verdict"]["pair_class"] == "unanimous_admitted_style"
            )),
            -len(body["supports"]),
            -float(body["best_negative"]),
            int(body["query"]), int(body["candidate"]),
        )
    pending.sort(key=lambda body: body["sort_key"])
    contested_pending.sort(key=lambda body: body["sort_key"])

    records, arm_audit, correctable = allocate_pending(
        pending, writer, None,
        (DYNAMIC_ERROR_REFERENCE, DYNAMIC_MARGIN_REFERENCE),
    )
    contested_records, contested_audit, contested_correctable = allocate_pending(
        contested_pending, contested_writer, V16_CONTESTED_REFERENCE,
        (V16_CONTESTED_REFERENCE, V16_CONTESTED_REFERENCE),
    )
    audit["current_error_queries"] = len(current_error_queries)
    audit.update(arm_audit)

    output = writer.arrays()
    contested_output = contested_writer.arrays()
    chemical_slice = slice(phasea_events, len(output["anchor_idx"]))
    contested_slice = slice(phasea_events, len(contested_output["anchor_idx"]))
    gates = {
        "phasea_pool_is_exact_prefix": verify_phasea_prefix(phasea, output),
        "contested_phasea_prefix_exact": verify_phasea_prefix(
            phasea, contested_output,
        ),
        "one_positive_reference_per_event": bool(np.all(
            np.diff(output["positive_ptr"])[chemical_slice] == 1
        )),
        "one_negative_reference_per_event": bool(np.all(
            np.diff(output["negative_ptr"])[chemical_slice] == 1
        )),
        "one_event_per_distinct_false_candidate": len(records) == len({
            (int(row["manifest_query"]), int(row["false_candidate"]))
            for row in records
        }),
        "every_dynamic_reference_pair_active_at_mining_geometry": bool(all(
            float(row["minimum_native_hinge"]) > 0 for row in records
        )),
        "arm_a_roles_are_dynamic_reference_roles": bool(np.all(np.isin(
            np.asarray(output["curriculum_role"])[chemical_slice],
            [DYNAMIC_ERROR_REFERENCE, DYNAMIC_MARGIN_REFERENCE],
        ))),
        "contested_arm_roles_are_v16_contested": bool(np.all(
            np.asarray(contested_output["curriculum_role"])[contested_slice]
            == V16_CONTESTED_REFERENCE
        )),
        "contested_events_active_at_mining_geometry": bool(all(
            float(row["minimum_native_hinge"]) > 0
            for row in contested_records
        )),
        "no_numerical_source_fusion": True,
        "validation_pool_unchanged": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V16 dynamic-reference gates failed: {gates}")

    report = {
        "status": "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE",
        "variant": "GLM_v16_symmetric_significance_pre_registered",
        "initialization": args.geometry_label,
        "launch_gate": launch_gate,
        "embedding_provenance": embedding_provenance,
        "source_families": families,
        "phasea_events_preserved": phasea_events,
        "dynamic_chemical_events_added": len(records),
        "dynamic_chemical_queries": len({
            int(row["manifest_query"]) for row in records
        }),
        "dynamic_chemical_formulas": len({
            str(manifest["query_formula"][int(row["manifest_query"])])
            for row in records
        }),
        "source_correctable_current_winners": len(correctable),
        "v16_audit": audit,
        "contested_arm": {
            "output": str(args.contested_output.resolve()),
            "relations_routed": audit["relations_routed_to_contested_arm"],
            "events": len(contested_records),
            "queries": len({
                int(row["manifest_query"]) for row in contested_records
            }),
            "correctable_current_winners": len(contested_correctable),
            "pruned_by_formula_cap":
                contested_audit["relations_pruned_by_formula_cap"],
            "pruned_by_identity_cap":
                contested_audit["relations_pruned_by_identity_cap"],
            "events_colliding_with_prior_pool":
                contested_audit["events_colliding_with_prior_pool"],
        },
        "reference_multiplicity": {
            "maximum_positive_references_per_relation":
                args.maximum_positive_references_per_relation,
            "maximum_negative_references_per_event":
                args.maximum_negative_references_per_event,
            "maximum_candidates_per_query": args.maximum_candidates_per_query,
            "maximum_events_per_formula": args.maximum_events_per_formula,
            "maximum_events_per_identity": args.maximum_events_per_identity,
        },
        "matched_control_policy": "v16_symmetric_significance_veto",
        "identity_audit": audit_identity_edges(output, args.data),
        "contested_identity_audit": audit_identity_edges(
            contested_output, args.data,
        ),
        "gates": gates,
        "scientific_contract": {
            "changed_component": "veto symmetry of qualified triplet selection only",
            "phasea_pool": "copied exactly before V16 dynamic chemical events",
            "source": "formula-disjoint confirmed truth-blind candidate relation evidence",
            "veto_rule": (
                "an opposition blocks only when its reversed delta beats every "
                "reversed matched-control delta (the same standard support must "
                "meet); sub-significant opposition is retained in the ledger"
            ),
            "contested_arm": (
                "dominant support + dominant blocking opposition relations are "
                "routed to a separate labelled pool, never silently discarded"
            ),
            "arm_a_superset_claim": (
                "every production-admitted relation is admitted here unchanged; "
                "the increment is exactly sub-significant-veto recovery"
            ),
            "loss": "unchanged DreaMS native triplet loss with unit event weights",
            "teacher_or_distillation": False,
            "chemical_score_used_as_training_target": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.contested_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix="glm_v16_dynamic_", dir=args.output.parent,
    ))
    contested_temporary = Path(tempfile.mkdtemp(
        prefix="glm_v16_contested_", dir=args.contested_output.parent,
    ))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        write_ledger(temporary / "dynamic_source_event_ledger.tsv", records)
        write_ledger(temporary / "rejected_relation_ledger.tsv", rejected_records)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        np.savez_compressed(
            contested_temporary / "train_pool.npz", **contested_output,
        )
        shutil.copy2(
            args.validation_pool, contested_temporary / "val_pool.npz",
        )
        write_ledger(
            contested_temporary / "dynamic_source_event_ledger.tsv",
            contested_records,
        )
        (contested_temporary / "report.json").write_text(
            json.dumps({
                "status": "GLM_V16_CONTESTED_ARM_POOL_COMPLETE",
                "variant": "GLM_v16_contested_arm_pre_registered",
                "derived_from": str(args.output.resolve()),
                "launch_gate": launch_gate,
                "phasea_events_preserved": phasea_events,
                "contested_events": len(contested_records),
                "contested_queries": len({
                    int(row["manifest_query"]) for row in contested_records
                }),
                "identity_audit": report["contested_identity_audit"],
                "scientific_contract": (
                    "dominant-vs-dominant chemical conflicts; exploratory arm; "
                    "identical native trainer settings if ever trained"
                ),
            }, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
        contested_temporary.replace(args.contested_output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(contested_temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

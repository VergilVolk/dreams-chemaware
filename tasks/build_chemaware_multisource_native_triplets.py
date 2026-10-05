"""Append source-proven candidate boundaries to the protected Phase-A pool.

Sources are never numerically fused.  Each source must independently rank the
truth strictly above the applicable candidate set.  A pair is rejected when
any comparable source ranks the false candidate at least as high as the truth.
For sources carrying matched controls, the correct pairwise advantage must
also strictly exceed every control advantage.  The surviving relation only
selects a native identity triplet; no teacher value enters the DreaMS loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from build_chemaware_max_boundary_native_triplets import FrozenEmbeddings, PoolWriter, edges
from build_chemaware_sirius_native_triplets import copy_phasea, load_npz, verify_phasea_prefix


ROOT = Path(__file__).resolve().parents[1]
MULTISOURCE_ERROR_BOUNDARY = 7
MULTISOURCE_MARGIN_BOUNDARY = 8
SCOPES = {"all_candidates", "cross_formula", "within_formula"}
CONFIDENCE_RANKS = {"A": 3, "B": 2, "C": 1, "Q": 0}


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
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--maximum-source-events-per-query", type=int, default=2)
    parser.add_argument("--minimum-supporting-sources", type=int, default=1)
    parser.add_argument(
        "--geometry-label", default="protected Phase-A +2.1266 pp checkpoint",
        help="Provenance label only; never changes geometry or optimization.",
    )
    return parser.parse_args()


def optional_float(value: str) -> float:
    if not value.strip():
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_ledgers(
    paths: list[Path],
) -> tuple[dict[int, dict[int, dict[str, dict[str, object]]]], dict[str, dict[str, object]]]:
    scores: dict[int, dict[int, dict[str, dict[str, object]]]] = {}
    families: dict[str, dict[str, object]] = {}
    for path in paths:
        directory = path if path.is_dir() else path.parent
        table = directory / "candidate_scores.tsv" if path.is_dir() else path
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE":
            raise RuntimeError(f"candidate source ledger is incomplete: {directory}")
        if report.get("truth_fields_exported") is not False:
            raise RuntimeError(f"candidate source ledger is not truth-blind: {directory}")
        if report.get("candidate_scores_sha256") != file_sha256(table):
            raise RuntimeError(f"candidate source ledger hash drift: {directory}")
        for family, body in report["source_families"].items():
            if family in families and families[family] != body:
                raise RuntimeError(f"source family contract drift: {family}")
            if body.get("scope") not in SCOPES or body.get("larger_is_better") is not True:
                raise RuntimeError(f"invalid source family contract: {family}")
            if body.get("specificity_gate_passed") is not True:
                raise RuntimeError(f"unqualified source family reached triplet mining: {family}")
            tier = str(body.get("confidence_tier", ""))[:1]
            if tier not in CONFIDENCE_RANKS or tier == "Q":
                raise RuntimeError(f"invalid admitted confidence tier for {family}: {tier!r}")
            families[family] = body
        with table.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise RuntimeError(f"candidate source ledger has no header: {table}")
            control_columns = sorted(
                name for name in reader.fieldnames
                if name.startswith("control_") and name.endswith("_score")
            )
            if not control_columns:
                raise RuntimeError(f"candidate source ledger has no controls: {table}")
            for row in reader:
                query = int(row["manifest_query"])
                candidate = int(row["local_candidate"])
                family = row["source_family"]
                if family not in families or row["scope"] != families[family]["scope"]:
                    raise RuntimeError(f"source family/schema mismatch: {family}")
                body = {
                    "ik14": row["ik14"], "formula": row["formula"],
                    "scope": row["scope"], "score": optional_float(row["source_score"]),
                    "control_scores": tuple(optional_float(row[name]) for name in control_columns),
                    "control_names": tuple(control_columns),
                    "controls": bool(int(row["controls_available"])),
                }
                slot = scores.setdefault(query, {}).setdefault(candidate, {})
                if family in slot:
                    raise RuntimeError(
                        f"duplicate source score: query={query} candidate={candidate} family={family}"
                    )
                slot[family] = body
    if not scores or not families:
        raise RuntimeError("candidate source ledgers are empty")
    return scores, families


def pair_evidence(
    candidates: Mapping[int, Mapping[str, Mapping[str, object]]],
    families: Mapping[str, Mapping[str, object]],
    true_candidate: int, false_candidate: int,
) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    supports_with_rank: list[tuple[str, float, int]] = []
    opposes_with_rank: list[tuple[str, int]] = []
    for family in sorted(families):
        if family not in candidates.get(true_candidate, {}) or family not in candidates.get(false_candidate, {}):
            continue
        true = candidates[true_candidate][family]
        false = candidates[false_candidate][family]
        true_score, false_score = float(true["score"]), float(false["score"])
        if not (math.isfinite(true_score) and math.isfinite(false_score)):
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
        delta = true_score - false_score
        if delta <= 0:
            tier = str(families[family]["confidence_tier"])[0]
            opposes_with_rank.append((family, CONFIDENCE_RANKS[tier]))
            continue
        tier = str(families[family]["confidence_tier"])[0]
        requires_controls = bool(families[family].get("matched_controls")) or tier == "C"
        if requires_controls and not (bool(true["controls"]) and bool(false["controls"])):
            continue
        if bool(true["controls"]) or bool(false["controls"]):
            if not (bool(true["controls"]) and bool(false["controls"])):
                continue
            if true["control_names"] != false["control_names"]:
                raise RuntimeError(f"control schema mismatch for source family {family}")
            control_deltas = tuple(
                float(true_value) - float(false_value)
                for true_value, false_value in zip(
                    true["control_scores"], false["control_scores"], strict=True,
                )
            )
            if not all(math.isfinite(value) and delta > value for value in control_deltas):
                continue
        supports_with_rank.append((family, delta, CONFIDENCE_RANKS[tier]))
    if not supports_with_rank:
        return [], [family for family, _ in opposes_with_rank], []
    strongest_support = max(rank for _, _, rank in supports_with_rank)
    blocking = [
        family for family, rank in opposes_with_rank if rank >= strongest_support
    ]
    nonblocking = [
        family for family, rank in opposes_with_rank if rank < strongest_support
    ]
    supports = [(family, delta) for family, delta, _ in supports_with_rank]
    return supports, blocking, nonblocking


def candidate_geometry(
    manifest: Mapping[str, np.ndarray], cache: FrozenEmbeddings, query: int,
) -> dict[str, object]:
    anchor = int(manifest["query_row"][query])
    anchor_embedding = cache.get([anchor])[0]
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
    true_values = np.flatnonzero(labels)
    if len(true_values) != 1:
        raise RuntimeError(f"query {query} does not have exactly one true candidate")
    true_candidate = int(true_values[0])
    top_row: dict[int, int] = {}
    top_score: dict[int, float] = {}
    for candidate in range(right - left):
        candidate_rows = np.unique(molecule_rows(manifest, query, candidate))
        if candidate == true_candidate:
            candidate_rows = candidate_rows[candidate_rows != anchor]
        if not len(candidate_rows):
            raise RuntimeError(f"query {query} candidate {candidate} has no usable reference")
        values = cache.get(candidate_rows) @ anchor_embedding
        best = int(np.argmax(values))
        top_row[candidate] = int(candidate_rows[best])
        top_score[candidate] = float(values[best])
    return {
        "anchor": anchor, "true_candidate": true_candidate,
        "top_row": top_row, "top_score": top_score,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.margin <= 0 or args.maximum_source_events_per_query < 1:
        raise ValueError("margin and maximum source events must be positive")
    if args.minimum_supporting_sources < 1:
        raise ValueError("minimum supporting sources must be positive")
    phasea = load_npz(args.phasea_pool)
    manifest = load_npz(args.manifest)
    scores, families = read_source_ledgers(args.source_ledger)
    cache = FrozenEmbeddings(args.embedding_rows, args.phasea_embeddings)
    writer = PoolWriter()
    copy_phasea(phasea, writer)
    phasea_events = len(writer.anchor)
    records: list[dict[str, object]] = []
    current_errors = corrected_current_winners = active_boundaries = 0
    rejected_conflicts = rejected_insufficient_support = duplicates = 0
    nonblocking_lower_tier_conflicts = 0

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
        geometry = candidate_geometry(manifest, cache, query)
        truth = int(geometry["true_candidate"])
        top_score = geometry["top_score"]
        top_row = geometry["top_row"]
        assert isinstance(top_score, dict) and isinstance(top_row, dict)
        true_score = float(top_score[truth])
        false_candidates = [candidate for candidate in range(candidate_count) if candidate != truth]
        winner = max(false_candidates, key=lambda candidate: top_score[candidate])
        current_error = float(top_score[winner]) >= true_score
        current_errors += int(current_error)
        eligible: list[
            tuple[int, int, float, int, list[tuple[str, float]], list[str]]
        ] = []
        for candidate in false_candidates:
            hinge = args.margin + float(top_score[candidate]) - true_score
            if hinge <= 0 or candidate not in scores[query] or truth not in scores[query]:
                continue
            active_boundaries += 1
            supports, blocking_opposes, nonblocking_opposes = pair_evidence(
                scores[query], families, truth, candidate,
            )
            if blocking_opposes:
                rejected_conflicts += 1
                continue
            if len(supports) < args.minimum_supporting_sources:
                rejected_insufficient_support += 1
                continue
            strongest_support_rank = max(
                CONFIDENCE_RANKS[str(families[family]["confidence_tier"])[0]]
                for family, _ in supports
            )
            eligible.append((
                strongest_support_rank, len(supports), float(top_score[candidate]), candidate,
                supports, nonblocking_opposes,
            ))
        eligible.sort(key=lambda value: (-value[0], -value[1], -value[2], value[3]))
        corrected_current_winners += int(
            current_error and any(candidate == winner for _, _, _, candidate, _, _ in eligible)
        )
        for strongest_rank, support_count, false_score, candidate, supports, nonblocking_opposes in eligible[
            :args.maximum_source_events_per_query
        ]:
            before = len(writer.anchor)
            writer.append(
                int(geometry["anchor"]), [int(top_row[truth])], [int(top_row[candidate])],
                query, candidate, 0,
                MULTISOURCE_ERROR_BOUNDARY if current_error else MULTISOURCE_MARGIN_BOUNDARY,
            )
            if len(writer.anchor) == before:
                duplicates += 1
                continue
            nonblocking_lower_tier_conflicts += len(nonblocking_opposes)
            records.append({
                "manifest_query": query,
                "true_candidate": truth,
                "false_candidate": candidate,
                "support_count": support_count,
                "strongest_support_tier": next(
                    tier for tier, rank in CONFIDENCE_RANKS.items() if rank == strongest_rank
                ),
                "supporting_sources": ",".join(family for family, _ in supports),
                "source_deltas": ",".join(f"{delta:.9g}" for _, delta in supports),
                "nonblocking_lower_tier_opposition": ",".join(nonblocking_opposes),
                "phasea_true_score": f"{true_score:.17g}",
                "phasea_false_score": f"{false_score:.17g}",
                "native_hinge": f"{args.margin + false_score - true_score:.17g}",
                "phasea_current_error": int(current_error),
            })

    output = writer.arrays()
    source_roles = np.asarray(output["curriculum_role"], dtype=np.int64)[phasea_events:]
    gates = {
        "phasea_pool_is_exact_prefix": verify_phasea_prefix(phasea, output),
        "source_events_are_singleton_native_triplets": all(
            int(output["positive_ptr"][event + 1] - output["positive_ptr"][event]) == 1
            and int(output["negative_ptr"][event + 1] - output["negative_ptr"][event]) == 1
            for event in range(phasea_events, len(output["anchor_idx"]))
        ),
        "source_event_roles_are_explicit": bool(np.all(np.isin(
            source_roles, [MULTISOURCE_ERROR_BOUNDARY, MULTISOURCE_MARGIN_BOUNDARY],
        ))),
        "no_numerical_source_fusion": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"multisource native-triplet gates failed: {gates}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_multisource_triplets_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        with (temporary / "source_event_ledger.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            if records:
                writer_csv = csv.DictWriter(
                    handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n",
                )
                writer_csv.writeheader()
                writer_csv.writerows(records)
            else:
                handle.write("manifest_query\ttrue_candidate\tfalse_candidate\n")
        report = {
            "status": "CHEMAWARE_MULTISOURCE_NATIVE_TRIPLETS_COMPLETE",
            "initialization": args.geometry_label,
            "source_families": families,
            "phasea_events_preserved": phasea_events,
            "source_events_added": len(output["anchor_idx"]) - phasea_events,
            "source_error_events": int(np.sum(source_roles == MULTISOURCE_ERROR_BOUNDARY)),
            "source_margin_events": int(np.sum(source_roles == MULTISOURCE_MARGIN_BOUNDARY)),
            "source_event_queries": len({int(row["manifest_query"]) for row in records}),
            "source_event_formulas": len({
                str(manifest["query_formula"][int(row["manifest_query"])]) for row in records
            }),
            "consensus_events": sum(int(row["support_count"]) > 1 for row in records),
            "single_source_events": sum(int(row["support_count"]) == 1 for row in records),
            "events_by_strongest_support_tier": {
                tier: sum(row["strongest_support_tier"] == tier for row in records)
                for tier in ("A", "B", "C")
            },
            "phasea_current_errors_in_source_coverage": current_errors,
            "source_corrected_phasea_current_winners": corrected_current_winners,
            "active_source_covered_boundaries": active_boundaries,
            "rejected_source_conflicts": rejected_conflicts,
            "nonblocking_lower_tier_conflicts": nonblocking_lower_tier_conflicts,
            "rejected_insufficient_support": rejected_insufficient_support,
            "duplicate_events_already_in_phasea": duplicates,
            "conditional_current_winner_headroom": (
                corrected_current_winners / current_errors if current_errors else 0.0
            ),
            "margin": args.margin,
            "maximum_source_events_per_query": args.maximum_source_events_per_query,
            "minimum_supporting_sources": args.minimum_supporting_sources,
            "identity_audit": audit_identity_edges(output, args.data),
            "gates": gates,
            "scientific_contract": {
                "changed_component": "triplet source selection only",
                "phasea_reproduction": False,
                "phasea_pool": "copied exactly before source events",
                "boundary": f"only active max-reference candidates under {args.geometry_label}",
                "source_rule": (
                    "candidate-pair advantage must be positive and exceed every matched-control "
                    "advantage; source families are qualified independently before mining"
                ),
                "conflict_rule": "discard when any comparable source prefers the false candidate",
                "confidence_rule": (
                    "same-or-higher-tier opposition vetoes; lower-tier opposition is retained "
                    "in the event ledger but cannot veto stronger qualified support"
                ),
                "fusion": "rank/consensus only; source values are never added or averaged",
                "trainer": "unchanged native DreaMS triplet fine-tuning",
            },
        }
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

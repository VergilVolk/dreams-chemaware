"""Build one ChemAware-native event for every eligible MassSpecGym anchor.

The complete official strict-10-ppm DreaMS train pool is the curriculum.  Its
anchor order and cardinality are immutable: every eligible anchor occurs once.
For an anchor with qualified candidate-level chemical evidence, only the
reference pair is replaced by the retrieval-aligned molecule-max positive and
one chemically supported active negative.  Every other anchor retains its
official positive and negative candidate lists byte-for-byte.

Chemistry therefore selects a supervised negative; it is never a regression
target, loss weight, teacher score, or source of duplicated events.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
)
from build_chemaware_max_boundary_native_triplets import FrozenEmbeddings
from build_chemaware_multisource_native_triplets import read_source_ledgers
from build_chemaware_sirius_native_triplets import load_npz
from GLM_build_chemaware_v16_dynamic_triplets import (
    v16_registry,
    v16_relation_verdict,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE_EVENT = 0
CHEMICAL_WINNER_EVENT = 1
CHEMICAL_SEMIHARD_EVENT = 2
ADMITTED_CLASSES = {
    "unanimous_admitted_style",
    "vetoed_by_subsignificant_oppose",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-train-pool", type=Path, action="append", required=True,
    )
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-rows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-report", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--minimum-supporting-sources", type=int, default=1)
    return parser.parse_args()


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


def validate_geometry_cache(
    report_path: Path, checkpoint: Path, manifest_path: Path,
    rows: np.ndarray, embeddings: np.ndarray, manifest: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Validate either the all-manifest token cache or the newer encode cache."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(checkpoint)
    manifest_sha = file_sha256(manifest_path)
    status = str(report.get("status"))
    if status == "CHEMAWARE_CHECKPOINT_MANIFEST_EMBEDDINGS_COMPLETE":
        expected = {
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "rows_array_sha256": array_sha256(rows),
            "embeddings_array_sha256": array_sha256(embeddings),
            "formula_role_4_accessed": False,
        }
        drift = {key: (report.get(key), value) for key, value in expected.items()
                 if report.get(key) != value}
        if drift:
            raise RuntimeError(f"embedding provenance drift: {drift}")
    elif status == "chemaware_corrected_manifest_token_cache_complete":
        provenance = report.get("provenance", {})
        if provenance.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("all-manifest cache does not match the candidate manifest")
        if provenance.get("official_checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("all-manifest embeddings do not match the geometry checkpoint")
        if int(report.get("spectra", -1)) != len(rows):
            raise RuntimeError("all-manifest cache row count drifted")
    else:
        raise RuntimeError(f"unsupported embedding cache report: {status}")
    required = np.unique(np.concatenate((
        np.asarray(manifest["query_row"], dtype=np.int64),
        np.asarray(manifest["pair_candidate_row"], dtype=np.int64),
    )))
    if not np.array_equal(required, rows):
        raise RuntimeError("geometry cache does not cover the complete candidate manifest")
    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise RuntimeError("geometry cache row/embedding shapes disagree")
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 5e-5:
        raise RuntimeError("geometry embeddings are not normalized")
    return {
        "report": str(report_path.resolve()),
        "report_status": status,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(embeddings),
        "rows": len(rows),
        "dimension": int(embeddings.shape[1]),
    }


def edge_slice(pool: Mapping[str, np.ndarray], event: int, kind: str) -> np.ndarray:
    pointer = np.asarray(pool[f"{kind}_ptr"], dtype=np.int64)
    values = np.asarray(pool[f"{kind}_idx"], dtype=np.int64)
    left, right = map(int, pointer[event:event + 2])
    return values[left:right]


def validate_native_pool(pool: Mapping[str, np.ndarray], name: str) -> None:
    required = {"anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx"}
    missing = sorted(required - set(pool))
    if missing:
        raise RuntimeError(f"{name} pool lacks arrays: {missing}")
    anchors = np.asarray(pool["anchor_idx"], dtype=np.int64)
    if len(anchors) == 0 or len(np.unique(anchors)) != len(anchors):
        raise RuntimeError(f"{name} anchors must be nonempty and unique")
    for kind in ("positive", "negative"):
        pointer = np.asarray(pool[f"{kind}_ptr"], dtype=np.int64)
        values = np.asarray(pool[f"{kind}_idx"], dtype=np.int64)
        if (
            len(pointer) != len(anchors) + 1
            or pointer[0] != 0
            or pointer[-1] != len(values)
            or np.any(np.diff(pointer) < 1)
        ):
            raise RuntimeError(f"{name} {kind} pointers are invalid or contain empty events")


def select_chemical_overrides(
    manifest: Mapping[str, np.ndarray],
    cache: FrozenEmbeddings,
    scores: Mapping[int, Mapping[int, Mapping[str, Mapping[str, object]]]],
    families: Mapping[str, Mapping[str, object]],
    official_anchors: set[int],
    margin: float,
    minimum_sources: int,
) -> tuple[dict[int, dict[str, object]], dict[str, int]]:
    """Choose at most one chemically supported retrieval boundary per anchor."""
    registry = v16_registry(families)
    selected: dict[int, dict[str, object]] = {}
    audit = {
        "ledger_queries": len(scores),
        "queries_outside_official_pool": 0,
        "queries_without_qualified_active_relation": 0,
        "contested_relations_rejected": 0,
        "no_signal_relations_rejected": 0,
        "qualified_active_relations": 0,
        "current_error_winner_overrides": 0,
        "boundary_winner_overrides": 0,
        "chemical_semihard_overrides": 0,
    }
    for query in sorted(scores):
        if query < 0 or query >= len(manifest["query_row"]):
            raise RuntimeError(f"ledger query outside manifest: {query}")
        anchor = int(manifest["query_row"][query])
        if anchor not in official_anchors:
            audit["queries_outside_official_pool"] += 1
            continue
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        candidate_count = right - left
        for candidate, source_body in scores[query].items():
            if candidate < 0 or candidate >= candidate_count:
                raise RuntimeError(
                    f"ledger candidate outside manifest: query={query} candidate={candidate}"
                )
            molecule = left + candidate
            for body in source_body.values():
                if str(manifest["molecule_ik14"][molecule]) != str(body["ik14"]):
                    raise RuntimeError(
                        f"ledger identity drift: query={query} candidate={candidate}"
                    )

        geometry = reference_geometry(manifest, cache, query)
        truth = int(geometry["truth"])
        rows_by_candidate = geometry["rows"]
        similarity = geometry["scores"]
        true_best = float(np.max(similarity[truth]))
        false_candidates = [candidate for candidate in range(candidate_count) if candidate != truth]
        winner = max(false_candidates, key=lambda candidate: float(np.max(similarity[candidate])))
        current_error = float(np.max(similarity[winner])) >= true_best
        candidates: list[dict[str, object]] = []
        for candidate in false_candidates:
            if candidate not in scores[query] or truth not in scores[query]:
                continue
            verdict = v16_relation_verdict(scores[query], registry, truth, candidate)
            pair_class = str(verdict["pair_class"])
            if pair_class == "contested_dominant":
                audit["contested_relations_rejected"] += 1
                continue
            if pair_class not in ADMITTED_CLASSES:
                audit["no_signal_relations_rejected"] += 1
                continue
            if len(verdict["supports"]) < minimum_sources or verdict["blocking_dominant"]:
                continue
            active = active_reference_events(
                rows_by_candidate[truth], similarity[truth],
                rows_by_candidate[candidate], similarity[candidate],
                margin, 1, 1,
            )
            if not active:
                continue
            audit["qualified_active_relations"] += 1
            candidates.append({
                "candidate": int(candidate),
                "event": active[0],
                "verdict": verdict,
                "is_winner": candidate == winner,
                "current_error": current_error,
                "negative_score": float(np.max(similarity[candidate])),
            })
        if not candidates:
            audit["queries_without_qualified_active_relation"] += 1
            continue
        candidates.sort(key=lambda body: (
            -int(bool(body["current_error"]) and bool(body["is_winner"])),
            -int(bool(body["is_winner"])),
            -int(body["verdict"]["pair_class"] == "unanimous_admitted_style"),
            -len(body["verdict"]["supports"]),
            -float(body["negative_score"]),
            int(body["candidate"]),
        ))
        choice = candidates[0]
        if bool(choice["current_error"]) and bool(choice["is_winner"]):
            role = CHEMICAL_WINNER_EVENT
            audit["current_error_winner_overrides"] += 1
        elif bool(choice["is_winner"]):
            role = CHEMICAL_WINNER_EVENT
            audit["boundary_winner_overrides"] += 1
        else:
            role = CHEMICAL_SEMIHARD_EVENT
            audit["chemical_semihard_overrides"] += 1
        choice["role"] = role
        choice["query"] = query
        choice["truth"] = truth
        selected[anchor] = choice
    return selected, audit


def merge_official_pools(pools: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate official pools in order, relocating global edge pointers.

    Duplicate anchors are not merged here on purpose: the caller re-runs
    ``validate_native_pool`` on the result, so any overlap between pools
    fails closed instead of silently double-training an anchor.
    """
    if len(pools) == 1:
        return pools[0]
    positive_index = np.concatenate(
        [np.asarray(pool["positive_idx"]) for pool in pools],
    )
    negative_index = np.concatenate(
        [np.asarray(pool["negative_idx"]) for pool in pools],
    )
    positive_pointers = [np.asarray([0], dtype=np.int64)]
    negative_pointers = [np.asarray([0], dtype=np.int64)]
    positive_total = 0
    negative_total = 0
    for pool in pools:
        positive_ptr = np.asarray(pool["positive_ptr"], dtype=np.int64)
        negative_ptr = np.asarray(pool["negative_ptr"], dtype=np.int64)
        positive_pointers.append(positive_ptr[1:] + positive_total)
        negative_pointers.append(negative_ptr[1:] + negative_total)
        positive_total += int(positive_ptr[-1])
        negative_total += int(negative_ptr[-1])
    return {
        "anchor_idx": np.concatenate(
            [np.asarray(pool["anchor_idx"], dtype=np.int64) for pool in pools],
        ),
        "positive_ptr": np.concatenate(positive_pointers),
        "positive_idx": positive_index,
        "negative_ptr": np.concatenate(negative_pointers),
        "negative_idx": negative_index,
    }


def construct_pool(
    official: Mapping[str, np.ndarray], overrides: Mapping[int, Mapping[str, object]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    anchors = np.asarray(official["anchor_idx"], dtype=np.int64)
    positive_ptr = [0]
    negative_ptr = [0]
    positive_idx: list[int] = []
    negative_idx: list[int] = []
    source_role = np.zeros(len(anchors), dtype=np.int8)
    source_query = np.full(len(anchors), -1, dtype=np.int64)
    negative_candidate = np.full(len(anchors), -1, dtype=np.int32)
    records: list[dict[str, object]] = []
    for event, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        body = overrides.get(anchor)
        if body is None:
            positive = edge_slice(official, event, "positive")
            negative = edge_slice(official, event, "negative")
        else:
            ref = body["event"]
            positive = np.asarray([ref["positive_row"]], dtype=np.int64)
            negative = np.asarray(ref["negative_rows"], dtype=np.int64)
            source_role[event] = int(body["role"])
            source_query[event] = int(body["query"])
            negative_candidate[event] = int(body["candidate"])
            verdict = body["verdict"]
            records.append({
                "anchor_row": anchor,
                "manifest_query": int(body["query"]),
                "true_candidate": int(body["truth"]),
                "false_candidate": int(body["candidate"]),
                "positive_row": int(positive[0]),
                "negative_row": int(negative[0]),
                "curriculum_role": int(body["role"]),
                "pair_evidence_class": str(verdict["pair_class"]),
                "supporting_sources": ",".join(family for family, _ in verdict["supports"]),
                "minimum_native_hinge": float(np.min(ref["hinges"])),
            })
        positive_idx.extend(map(int, positive))
        negative_idx.extend(map(int, negative))
        positive_ptr.append(len(positive_idx))
        negative_ptr.append(len(negative_idx))
    return {
        "anchor_idx": anchors.copy(),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positive_idx, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negative_idx, dtype=np.int64),
        "curriculum_role": source_role,
        "source_query": source_query,
        "negative_candidate": negative_candidate,
    }, records


def unchanged_native_events(
    official: Mapping[str, np.ndarray], output: Mapping[str, np.ndarray],
    overridden: set[int],
) -> bool:
    for event, anchor in enumerate(np.asarray(official["anchor_idx"], dtype=np.int64)):
        if int(anchor) in overridden:
            continue
        if not np.array_equal(edge_slice(official, event, "positive"), edge_slice(output, event, "positive")):
            return False
        if not np.array_equal(edge_slice(official, event, "negative"), edge_slice(output, event, "negative")):
            return False
    return True


def write_ledger(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not records:
            handle.write("anchor_row\tmanifest_query\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.margin <= 0 or args.minimum_supporting_sources < 1:
        raise ValueError("margin and supporting-source threshold must be positive")
    loaded_pools = [load_npz(path) for path in args.official_train_pool]
    for path, pool in zip(args.official_train_pool, loaded_pools):
        validate_native_pool(pool, f"official pool {path.name}")
    official = merge_official_pools(loaded_pools)
    validate_native_pool(official, "merged official")
    validation = load_npz(args.validation_pool)
    manifest = load_npz(args.manifest)
    validate_native_pool(validation, "validation")
    official_anchors = set(map(int, official["anchor_idx"]))
    manifest_anchors = set(map(int, manifest["query_row"]))
    covered_manifest_anchors = official_anchors & manifest_anchors
    if not covered_manifest_anchors:
        raise RuntimeError("candidate manifest has no overlap with the official train pool")

    rows = np.load(args.embedding_rows, mmap_mode="r")
    cache = FrozenEmbeddings(args.embedding_rows, args.embeddings)
    embedding_provenance = validate_geometry_cache(
        args.embedding_report, args.geometry_checkpoint, args.manifest,
        rows, cache.embeddings, manifest,
    )
    scores, families = read_source_ledgers(args.source_ledger)
    overrides, selection_audit = select_chemical_overrides(
        manifest, cache, scores, families, official_anchors,
        args.margin, args.minimum_supporting_sources,
    )
    output, records = construct_pool(official, overrides)
    anchors = np.asarray(output["anchor_idx"], dtype=np.int64)
    roles = np.asarray(output["curriculum_role"], dtype=np.int8)
    chemical = roles != NATIVE_EVENT
    gates = {
        "all_official_massspecgym_anchors_present_once": bool(
            np.array_equal(anchors, official["anchor_idx"])
            and len(np.unique(anchors)) == len(official["anchor_idx"])
        ),
        "event_count_equals_complete_official_pool": len(anchors) == len(official["anchor_idx"]),
        "native_events_are_exactly_unchanged": unchanged_native_events(
            official, output, set(overrides),
        ),
        "one_event_per_anchor_no_duplication": len(anchors) == len(set(map(int, anchors))),
        "chemical_events_have_one_positive": bool(np.all(np.diff(output["positive_ptr"])[chemical] == 1)),
        "chemical_events_have_one_negative": bool(np.all(np.diff(output["negative_ptr"])[chemical] == 1)),
        "every_chemical_pair_active_at_frozen_geometry": bool(records) and all(
            float(row["minimum_native_hinge"]) > 0 for row in records
        ),
        "no_contested_relation_trained": all(
            row["pair_evidence_class"] != "contested_dominant" for row in records
        ),
        "validation_pool_is_unmodified": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"full-MassSpecGym triplet gates failed: {gates}")
    identity_audit = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_FULL_MASSSPECGYM_NATIVE_TRIPLETS_COMPLETE",
        "official_pools": [str(path.resolve()) for path in args.official_train_pool],
        "official_pools_sha256": [
            file_sha256(path) for path in args.official_train_pool
        ],
        "validation_pool": str(args.validation_pool.resolve()),
        "validation_pool_sha256": file_sha256(args.validation_pool),
        "complete_official_events": len(anchors),
        "complete_official_unique_anchors": len(np.unique(anchors)),
        "manifest_anchors_inside_official_pool": len(covered_manifest_anchors),
        "chemical_overrides": int(np.sum(chemical)),
        "native_events_preserved": int(np.sum(~chemical)),
        "chemical_override_fraction": float(np.mean(chemical)),
        "selection_audit": selection_audit,
        "source_families": families,
        "embedding_provenance": embedding_provenance,
        "identity_audit": identity_audit,
        "gates": gates,
        "scientific_contract": {
            "coverage": (
                "every eligible official MassSpecGym anchor exactly once "
                "(all provided official pools merged: train and validation "
                "splits)"
            ),
            "native_fallback": "official positive and negative candidate lists unchanged",
            "chemical_event": "one query-local, identity-correct, molecule-max active (q,p*,n*)",
            "chemical_evidence": "matched-control-qualified symmetric-significance relation",
            "excluded": ["identity broadcast", "repeated-event weighting", "contested evidence", "teacher targets", "distillation"],
            "training": "unchanged DreaMS ContrastiveSpectraDataset and ContrastiveHead triplet loss",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_full_msg_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        write_ledger(temporary / "chemical_override_ledger.tsv", records)
        (temporary / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

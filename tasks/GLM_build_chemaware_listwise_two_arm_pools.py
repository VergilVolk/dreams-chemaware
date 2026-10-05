"""GLM listwise two-arm pool builder (pre-registered 2026-09-30).

Builds ONE shared candidate-group training pool for the two-arm decisive
experiment of docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md.
Both arms consume identical arrays; they differ only in the margin vector
(``arm1_margin`` is identically zero, ``arm2_margin`` carries the sealed V16
chemical relation verdicts as per-candidate margins).

Design contracts (all fail-closed):
- Training queries are exactly the frozen formula-role-0/1 training panel from
  ``train_triplet_evidence.npz``; query sets and formula sets must be disjoint
  from the frozen role-2 selection and role-3 confirmation panels.
- Candidate groups come from the corrected candidate manifest (positive
  molecule first). Negative molecules are ranked by frozen Phase-A
  molecule-max similarity (desc, then molecule index asc) and capped at
  ``--maximum-candidates-per-query`` total molecules; reference spectra per
  molecule are ranked by frozen Phase-A cosine (desc, then row asc) and capped
  at ``--maximum-reference-spectra-per-molecule``. The query's own HDF5 row is
  excluded from its reference lists (training-only hygiene; the frozen
  evaluation panels are untouched) and every exclusion is audited.
- Arm-2 margins are recomputed with the exact sealed V16 verdict logic (same
  ledgers, same frozen geometry, margin 0.1, 1/1 reference sampling for the
  qualification filter) and ANCHORED to the frozen run_2347164 audit:
  qualified 1,684 / admitted 1,202 (602 uncontested + 600 recovered) /
  contested 482. Any drift fails the build.
- The builder refuses to overwrite existing outputs and writes atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from build_chemaware_dynamic_reference_native_triplets import (  # noqa: E402
    active_reference_events,
    reference_geometry,
    validate_embedding_provenance,
)
from build_chemaware_max_boundary_native_triplets import FrozenEmbeddings  # noqa: E402
from build_chemaware_multisource_native_triplets import read_source_ledgers  # noqa: E402
from build_chemaware_sirius_native_triplets import load_npz  # noqa: E402
from GLM_build_chemaware_v16_dynamic_triplets import (  # noqa: E402
    enforce_launch_gate,
    v16_registry,
    v16_relation_verdict,
)

POOL_STATUS = "GLM_LISTWISE_TWO_ARM_POOL_BUILT"
# The qualification filter of the sealed V16 computation: native margin and
# 1/1 reference sampling. These are NOT free parameters of this experiment.
SEALED_VERDICT_MARGIN = 0.1
SEALED_POSITIVE_REFERENCES = 1
SEALED_NEGATIVE_REFERENCES = 1
SEALED_LAUNCH_BOUNDARIES = 40  # amended, disclosed value of the retired route


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--confirmation-evidence", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, action="append", required=True)
    parser.add_argument("--embedding-rows", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--embedding-report", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-class-diagnostic", type=Path, required=True)
    parser.add_argument("--delta-main", type=float, default=0.05)
    parser.add_argument("--delta-contested", type=float, default=0.025)
    parser.add_argument("--maximum-candidates-per-query", type=int, default=8)
    parser.add_argument("--maximum-reference-spectra-per-molecule", type=int, default=3)
    parser.add_argument("--expected-qualified-relations", type=int, default=1684)
    parser.add_argument("--expected-admitted-relations", type=int, default=1202)
    parser.add_argument("--expected-uncontested-relations", type=int, default=602)
    parser.add_argument("--expected-recovered-relations", type=int, default=600)
    parser.add_argument("--expected-contested-relations", type=int, default=482)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_evidence_queries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        if "query" not in loaded or "formula" not in loaded:
            raise RuntimeError(f"evidence panel lacks query/formula registry: {path}")
        queries = np.asarray(loaded["query"], dtype=np.int64)
        formulas = np.asarray(loaded["formula"]).astype(str)
    if len(np.unique(queries)) != len(queries):
        raise RuntimeError(f"evidence panel has duplicate queries: {path}")
    return queries, formulas


def verify_panels(
    manifest: dict[str, np.ndarray],
    train: tuple[np.ndarray, np.ndarray],
    selection: tuple[np.ndarray, np.ndarray],
    confirmation: tuple[np.ndarray, np.ndarray],
) -> dict[str, int]:
    manifest_queries = set(range(len(manifest["query_row"])))
    train_queries, train_formulas = train
    selection_queries, selection_formulas = selection
    confirmation_queries, confirmation_formulas = confirmation
    train_set = {int(q) for q in train_queries}
    if not train_set <= manifest_queries:
        raise RuntimeError("training panel queries fall outside the manifest")
    if len(train_set) != len(train_queries):
        raise RuntimeError("training panel queries are not unique")
    selection_set = {int(q) for q in selection_queries}
    confirmation_set = {int(q) for q in confirmation_queries}
    if train_set & selection_set:
        raise RuntimeError("training panel intersects the role-2 selection panel")
    if train_set & confirmation_set:
        raise RuntimeError("training panel intersects the role-3 confirmation panel")
    train_formula_set = {str(f) for f in train_formulas}
    if train_formula_set & {str(f) for f in selection_formulas}:
        raise RuntimeError("training formulas intersect the selection panel formulas")
    if train_formula_set & {str(f) for f in confirmation_formulas}:
        raise RuntimeError("training formulas intersect the confirmation panel formulas")
    manifest_train_formulas = {
        str(manifest["query_formula"][q]) for q in sorted(train_set)
    }
    if manifest_train_formulas != train_formula_set:
        raise RuntimeError("training panel formulas disagree with the manifest")
    return {
        "train_panel_queries": len(train_set),
        "selection_panel_queries": len(selection_set),
        "confirmation_panel_queries": len(confirmation_set),
    }


def recompute_sealed_margins(
    manifest: dict[str, np.ndarray],
    cache: FrozenEmbeddings,
    scores: dict[int, dict[int, dict[str, dict[str, object]]]],
    registry: dict[str, dict[str, object]],
    expected: dict[str, int],
) -> tuple[dict[tuple[int, int], str], dict[str, int]]:
    """Re-derive the sealed V16 relation verdicts with identical logic."""
    margin_of_relation: dict[tuple[int, int], str] = {}
    audit = {
        "ledger_queries": len(scores),
        "qualified_candidate_relations": 0,
        "relations_rejected_no_dominant_support": 0,
        "relations_rejected_by_source_support": 0,
        "relations_with_active_references": 0,
        "relations_recovered_from_subsignificant_veto": 0,
        "relations_uncontested_admitted": 0,
        "relations_routed_to_contested_arm": 0,
    }
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
        false_candidates = [c for c in range(candidate_count) if c != truth]
        for candidate in false_candidates:
            if candidate not in scores[query] or truth not in scores[query]:
                continue
            verdict = v16_relation_verdict(scores[query], registry, truth, candidate)
            pair_class = verdict["pair_class"]
            if pair_class in ("dominant_opposition_only", "no_dominant_signal"):
                audit["relations_rejected_no_dominant_support"] += 1
                continue
            if not verdict["supports"]:
                audit["relations_rejected_by_source_support"] += 1
                continue
            events = active_reference_events(
                rows_by_candidate[truth], scores_by_candidate[truth],
                rows_by_candidate[candidate], scores_by_candidate[candidate],
                SEALED_VERDICT_MARGIN,
                SEALED_POSITIVE_REFERENCES,
                SEALED_NEGATIVE_REFERENCES,
            )
            if not events:
                continue
            audit["qualified_candidate_relations"] += 1
            molecule = left + candidate
            if pair_class == "contested_dominant":
                audit["relations_routed_to_contested_arm"] += 1
                margin_of_relation[(query, molecule)] = "contested"
            else:
                audit["relations_with_active_references"] += 1
                if pair_class == "vetoed_by_subsignificant_oppose":
                    audit["relations_recovered_from_subsignificant_veto"] += 1
                else:
                    audit["relations_uncontested_admitted"] += 1
                margin_of_relation[(query, molecule)] = "main"
    for key, value in expected.items():
        if int(audit[key]) != int(value):
            raise RuntimeError(
                f"sealed V16 relation anchor drifted: {key} observed={audit[key]} "
                f"expected={value}; refusing to build margins"
            )
    return margin_of_relation, audit


def ranked_rows(rows: np.ndarray, similarities: np.ndarray) -> np.ndarray:
    """Reference rows ordered by frozen similarity desc, then row asc."""
    order = sorted(
        range(len(rows)),
        key=lambda i: (-float(similarities[i]), int(rows[i])),
    )
    return np.asarray([int(rows[i]) for i in order], dtype=np.int64)


def is_validation_query(query_row: int) -> bool:
    digest = hashlib.sha256(f"glm-listwise-val-{int(query_row)}".encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") % 10 == 0


def main() -> None:
    args = arguments()
    if args.delta_main <= 0 or not 0 < args.delta_contested < args.delta_main:
        raise ValueError("margin deltas must satisfy 0 < delta_contested < delta_main")
    if args.maximum_candidates_per_query < 2:
        raise ValueError("at least one positive and one negative candidate are required")
    if args.maximum_reference_spectra_per_molecule < 1:
        raise ValueError("each candidate molecule needs at least one reference")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    provenance = enforce_launch_gate(
        args.evidence_class_diagnostic, args.manifest, args.source_ledger,
        SEALED_LAUNCH_BOUNDARIES,
    )
    manifest = load_npz(args.manifest)
    train = load_evidence_queries(args.train_evidence)
    selection = load_evidence_queries(args.selection_evidence)
    confirmation = load_evidence_queries(args.confirmation_evidence)
    panel_audit = verify_panels(manifest, train, selection, confirmation)

    cache = FrozenEmbeddings(args.embedding_rows, args.phasea_embeddings)
    embedding_provenance = validate_embedding_provenance(
        args.embedding_report, args.geometry_checkpoint, args.manifest,
        np.load(args.embedding_rows, mmap_mode="r"), cache.embeddings,
    )
    scores, families = read_source_ledgers(args.source_ledger)
    registry = v16_registry(families)
    margin_of_relation, relation_audit = recompute_sealed_margins(
        manifest, cache, scores, registry,
        {
            "qualified_candidate_relations": args.expected_qualified_relations,
            "relations_with_active_references": args.expected_admitted_relations,
            "relations_uncontested_admitted": args.expected_uncontested_relations,
            "relations_recovered_from_subsignificant_veto": args.expected_recovered_relations,
            "relations_routed_to_contested_arm": args.expected_contested_relations,
        },
    )

    with h5py.File(args.data, "r") as handle:
        total_rows = int(len(handle["INCHIKEY"]))

    train_queries = sorted(int(q) for q in train[0])
    query_row: list[int] = []
    group_ptr = [0]
    molecule_ref_ptr = [0]
    ref_row: list[int] = []
    molecule_label: list[int] = []
    arm2_margin: list[float] = []
    arm2_margin_kind: list[str] = []
    val_query_mask: list[bool] = []
    query_formula: list[str] = []
    molecule_ik14: list[str] = []
    excluded_groups = 0
    query_self_rows_excluded = 0
    candidate_cap_applied_groups = 0
    reference_cap_applied_molecules = 0
    margin_outside_candidate_cap = {"main": 0, "contested": 0}
    margin_applied = {"main": 0, "contested": 0}

    for query in train_queries:
        geometry = reference_geometry(manifest, cache, query)
        truth = int(geometry["truth"])
        rows_by_candidate = geometry["rows"]
        scores_by_candidate = geometry["scores"]
        candidate_count = len(rows_by_candidate)
        molecule_max = [
            float(np.max(scores_by_candidate[c])) for c in range(candidate_count)
        ]
        negatives = sorted(
            (c for c in range(candidate_count) if c != truth),
            key=lambda c: (-molecule_max[c], c),
        )
        keep = [truth, *negatives[: args.maximum_candidates_per_query - 1]]
        if len(keep) < candidate_count:
            candidate_cap_applied_groups += 1
        global_base = int(manifest["query_ptr"][query])
        kept_locals = set(keep)
        for local in range(candidate_count):
            if local in kept_locals:
                continue
            key = (query, global_base + local)
            if key in margin_of_relation:
                margin_outside_candidate_cap[margin_of_relation[key]] += 1

        query_hdf5_row = int(manifest["query_row"][query])
        group_refs: list[list[int]] = []
        group_labels: list[int] = []
        group_margins: list[float] = []
        group_kinds: list[str] = []
        group_ik14: list[str] = []
        valid = True
        for local in keep:
            ordered = ranked_rows(
                rows_by_candidate[local], scores_by_candidate[local],
            )
            filtered = [int(row) for row in ordered if int(row) != query_hdf5_row]
            query_self_rows_excluded += len(ordered) - len(filtered)
            if len(filtered) > args.maximum_reference_spectra_per_molecule:
                filtered = filtered[: args.maximum_reference_spectra_per_molecule]
                reference_cap_applied_molecules += 1
            if not filtered:
                valid = False
                break
            label = int(local == truth)
            kind = "none"
            margin = 0.0
            if label == 0:
                kind = margin_of_relation.get((query, global_base + local), "none")
                if kind == "main":
                    margin = float(args.delta_main)
                    margin_applied["main"] += 1
                elif kind == "contested":
                    margin = float(args.delta_contested)
                    margin_applied["contested"] += 1
                else:
                    kind = "none"
            group_refs.append(filtered)
            group_labels.append(label)
            group_margins.append(margin)
            group_kinds.append(kind)
            group_ik14.append(str(manifest["molecule_ik14"][global_base + local]))
        if (
            not valid
            or group_labels[0] != 1
            or sum(group_labels) != 1
            or len(group_labels) < 2
        ):
            excluded_groups += 1
            continue

        for rows, label, margin, kind, ik14 in zip(
            group_refs, group_labels, group_margins, group_kinds, group_ik14,
            strict=True,
        ):
            ref_row.extend(rows)
            molecule_ref_ptr.append(len(ref_row))
            molecule_label.append(label)
            arm2_margin.append(margin)
            arm2_margin_kind.append(kind)
            molecule_ik14.append(ik14)
        group_ptr.append(len(molecule_label))
        query_row.append(query_hdf5_row)
        query_formula.append(str(manifest["query_formula"][query]))
        val_query_mask.append(is_validation_query(query_hdf5_row))

    query_row_a = np.asarray(query_row, dtype=np.int64)
    group_ptr_a = np.asarray(group_ptr, dtype=np.int64)
    molecule_ref_ptr_a = np.asarray(molecule_ref_ptr, dtype=np.int64)
    ref_row_a = np.asarray(ref_row, dtype=np.int64)
    molecule_label_a = np.asarray(molecule_label, dtype=np.int8)
    arm2_margin_a = np.asarray(arm2_margin, dtype=np.float32)
    val_query_mask_a = np.asarray(val_query_mask, dtype=bool)
    arm1_margin_a = np.zeros(len(molecule_label_a), dtype=np.float32)

    if np.any((query_row_a < 0) | (query_row_a >= total_rows)):
        raise RuntimeError("query row out of HDF5 range")
    if np.any((ref_row_a < 0) | (ref_row_a >= total_rows)):
        raise RuntimeError("reference row out of HDF5 range")
    if group_ptr_a[-1] != len(molecule_label_a):
        raise RuntimeError("group pointer does not span molecules")
    if molecule_ref_ptr_a[-1] != len(ref_row_a):
        raise RuntimeError("reference pointer does not span rows")
    if np.any(np.diff(molecule_ref_ptr_a) < 1):
        raise RuntimeError("a candidate molecule carries no reference spectrum")
    for left, right in zip(group_ptr_a[:-1], group_ptr_a[1:]):
        labels = molecule_label_a[left:right]
        if len(labels) < 2 or labels[0] != 1 or int(labels.sum()) != 1:
            raise RuntimeError("every group needs exactly one first-position positive")
    if np.any(arm2_margin_a[molecule_label_a == 1] != 0.0):
        raise RuntimeError("a positive molecule received a chemical margin")
    if np.any(arm1_margin_a != 0.0):
        raise RuntimeError("arm-1 margin must be identically zero")
    if int((arm2_margin_a > 0).sum()) != (
        margin_applied["main"] + margin_applied["contested"]
    ):
        raise RuntimeError("arm-2 margin accounting mismatch")

    shared_digest = hashlib.sha256()
    for array in (
        query_row_a.tobytes(), group_ptr_a.tobytes(),
        molecule_ref_ptr_a.tobytes(), ref_row_a.tobytes(),
        molecule_label_a.tobytes(), val_query_mask_a.tobytes(),
    ):
        shared_digest.update(array)
    molecules_per_query = np.diff(group_ptr_a)
    refs_per_molecule = np.diff(molecule_ref_ptr_a)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        query_row=query_row_a,
        group_ptr=group_ptr_a,
        molecule_ref_ptr=molecule_ref_ptr_a,
        ref_row=ref_row_a,
        molecule_label=molecule_label_a,
        arm1_margin=arm1_margin_a,
        arm2_margin=arm2_margin_a,
        arm2_margin_kind=np.asarray(arm2_margin_kind, dtype=str),
        val_query_mask=val_query_mask_a,
        query_formula=np.asarray(query_formula, dtype=str),
        molecule_ik14=np.asarray(molecule_ik14, dtype=str),
    )
    temporary.replace(args.output)

    report = {
        "status": POOL_STATUS,
        "preregistration": (
            "docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md"
        ),
        "panels": panel_audit,
        "counts": {
            "training_queries": int(len(query_row_a)),
            "validation_queries": int(np.sum(val_query_mask_a)),
            "candidate_molecules": int(len(molecule_label_a)),
            "reference_spectrum_rows": int(len(ref_row_a)),
            "unique_reference_rows": int(len(np.unique(ref_row_a))),
        },
        "molecules_per_query": {
            "min": int(molecules_per_query.min()),
            "median": float(np.median(molecules_per_query)),
            "p90": float(np.quantile(molecules_per_query, 0.9)),
            "max": int(molecules_per_query.max()),
        },
        "reference_spectra_per_molecule": {
            "min": int(refs_per_molecule.min()),
            "median": float(np.median(refs_per_molecule)),
            "p90": float(np.quantile(refs_per_molecule, 0.9)),
            "max": int(refs_per_molecule.max()),
        },
        "caps": {
            "maximum_candidates_per_query": int(args.maximum_candidates_per_query),
            "maximum_reference_spectra_per_molecule": int(
                args.maximum_reference_spectra_per_molecule
            ),
            "groups_where_candidate_cap_applied": int(candidate_cap_applied_groups),
            "molecules_where_reference_cap_applied": int(
                reference_cap_applied_molecules
            ),
        },
        "margins": {
            "delta_main": float(args.delta_main),
            "delta_contested": float(args.delta_contested),
            "arm2_molecules_with_main_margin": int(margin_applied["main"]),
            "arm2_molecules_with_contested_margin": int(margin_applied["contested"]),
            "relations_dropped_by_candidate_cap": dict(margin_outside_candidate_cap),
            "arm1_margin_is_identically_zero": True,
        },
        "hygiene": {
            "query_self_reference_rows_excluded": int(query_self_rows_excluded),
            "groups_excluded_for_empty_references": int(excluded_groups),
        },
        "sealed_v16_relation_recompute": relation_audit,
        "launch_gate_provenance": provenance,
        "embedding_provenance": embedding_provenance,
        "shared_arrays_sha256": shared_digest.hexdigest(),
        "outputs": {"pool": str(args.output.resolve())},
        "arm_identity_contract": (
            "both arms read the same pool file; arm1 selects arm1_margin "
            "(all zero), arm2 selects arm2_margin; every other array is "
            "byte-identical by construction and summarized by "
            "shared_arrays_sha256"
        ),
    }
    report_path = args.output.with_suffix(".json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {report_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "status": POOL_STATUS,
        "training_queries": report["counts"]["training_queries"],
        "margins": report["margins"],
        "shared_arrays_sha256": report["shared_arrays_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

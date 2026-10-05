"""Append SIRIUS-proven max-boundary events to the protected Phase-A pool.

This is deliberately a small intervention.  The successful Phase-A triplet
pool is copied intact.  A new event is added only when (1) the false candidate
is active at the protected Phase-A retrieval boundary and (2) independent
SIRIUS evidence ranks the true chemistry above that false candidate.  Formula
comparisons use fragmentation-tree scores; same-formula structure comparisons
use CSI:FingerID scores.  The native DreaMS sampler, loss, model and optimizer
are not changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from build_chemaware_max_boundary_native_triplets import (
    FrozenEmbeddings,
    PoolWriter,
    edges,
)


ROOT = Path(__file__).resolve().parents[1]
SIRIUS_ERROR_BOUNDARY = 5
SIRIUS_MARGIN_BOUNDARY = 6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
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
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def optional_float(value: str) -> float:
    if not value.strip():
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def read_candidate_scores(
    path: Path,
) -> dict[int, dict[int, dict[str, object]]]:
    output: dict[int, dict[int, dict[str, object]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query = int(row["manifest_query"])
            candidate = int(row["local_candidate"])
            by_candidate = output.setdefault(query, {})
            if candidate in by_candidate:
                raise RuntimeError(
                    f"SIRIUS score table repeats query {query} candidate {candidate}"
                )
            by_candidate[candidate] = {
                "ik14": row["ik14"],
                "formula": row["formula"],
                "tree": optional_float(row["tree_score"]),
                "csi": optional_float(row["csi_fingerid_score"]),
            }
    if not output:
        raise RuntimeError("SIRIUS candidate score table is empty")
    return output


def source_relation(
    true_formula: str, false_formula: str,
    true_tree: float, false_tree: float,
    true_csi: float, false_csi: float,
    true_formula_is_top: bool, true_structure_is_top: bool,
) -> tuple[str, float] | None:
    """Return a strict, scale-separated chemical proof for one candidate pair."""
    if true_formula != false_formula:
        if (
            true_formula_is_top
            and math.isfinite(true_tree) and math.isfinite(false_tree)
            and true_tree > false_tree
        ):
            return "cross_formula_tree", true_tree - false_tree
        return None
    if (
        true_structure_is_top
        and math.isfinite(true_csi) and math.isfinite(false_csi)
        and true_csi > false_csi
    ):
        return "within_formula_csi", true_csi - false_csi
    return None


def copy_phasea(
    phasea: Mapping[str, np.ndarray], writer: PoolWriter,
) -> None:
    for event in range(len(phasea["anchor_idx"])):
        writer.append(
            int(phasea["anchor_idx"][event]),
            edges(phasea, event, "positive"),
            edges(phasea, event, "negative"),
            int(phasea["source_query"][event]),
            int(phasea["negative_candidate"][event]),
            int(phasea["source_tag"][event]),
            int(phasea["curriculum_role"][event]),
        )
    if len(writer.anchor) != len(phasea["anchor_idx"]):
        raise RuntimeError("protected Phase-A pool contains duplicate signatures")


def verify_phasea_prefix(
    phasea: Mapping[str, np.ndarray], output: Mapping[str, np.ndarray],
) -> bool:
    events = len(phasea["anchor_idx"])
    scalar = (
        "anchor_idx", "source_query", "negative_candidate", "source_tag",
        "curriculum_role",
    )
    if any(not np.array_equal(phasea[name], output[name][:events]) for name in scalar):
        return False
    for kind in ("positive", "negative"):
        edge_count = len(phasea[f"{kind}_idx"])
        if not np.array_equal(
            phasea[f"{kind}_idx"], output[f"{kind}_idx"][:edge_count],
        ):
            return False
        if not np.array_equal(
            phasea[f"{kind}_ptr"], output[f"{kind}_ptr"][:events + 1],
        ):
            return False
    return True


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
    rows_by_candidate: dict[int, np.ndarray] = {}
    top_row: dict[int, int] = {}
    top_score: dict[int, float] = {}
    for candidate in range(right - left):
        rows = np.unique(molecule_rows(manifest, query, candidate))
        if candidate == true_candidate:
            rows = rows[rows != anchor]
        if not len(rows):
            raise RuntimeError(f"query {query} candidate {candidate} has no usable reference")
        values = cache.get(rows) @ anchor_embedding
        best = int(np.argmax(values))
        rows_by_candidate[candidate] = rows
        top_row[candidate] = int(rows[best])
        top_score[candidate] = float(values[best])
    return {
        "anchor": anchor,
        "true_candidate": true_candidate,
        "rows": rows_by_candidate,
        "top_row": top_row,
        "top_score": top_score,
    }


def top_source_flags(
    scores: Mapping[int, Mapping[str, object]], true_candidate: int,
) -> tuple[bool, bool]:
    true = scores[true_candidate]
    true_formula = str(true["formula"])
    formula_values: dict[str, float] = {}
    for body in scores.values():
        formula = str(body["formula"])
        value = float(body["tree"])
        if math.isfinite(value):
            prior = formula_values.get(formula, -math.inf)
            formula_values[formula] = max(prior, value)
    true_tree = float(true["tree"])
    alternative_formula_values = [
        value for formula, value in formula_values.items()
        if formula != true_formula
    ]
    formula_top = (
        math.isfinite(true_tree)
        and all(true_tree > value for value in alternative_formula_values)
    )
    same_formula_false = [
        float(body["csi"])
        for candidate, body in scores.items()
        if candidate != true_candidate and str(body["formula"]) == true_formula
        and math.isfinite(float(body["csi"]))
    ]
    true_csi = float(true["csi"])
    structure_top = (
        math.isfinite(true_csi)
        and all(true_csi > value for value in same_formula_false)
    )
    return formula_top, structure_top


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.margin <= 0 or args.maximum_source_events_per_query < 1:
        raise ValueError("margin and maximum source events must be positive")
    phasea = load_npz(args.phasea_pool)
    validation = load_npz(args.validation_pool)
    manifest = load_npz(args.manifest)
    scores = read_candidate_scores(args.candidate_scores)
    cache = FrozenEmbeddings(args.embedding_rows, args.phasea_embeddings)
    writer = PoolWriter()
    copy_phasea(phasea, writer)
    phasea_events = len(writer.anchor)

    selected_records: list[dict[str, object]] = []
    phasea_errors = comparable_current_winners = source_corrected_current_winners = 0
    source_disagrees_current_winners = duplicate_source_events = 0
    active_candidates = source_proven_candidates = 0
    scored_candidate_rows = 0

    for query in sorted(scores):
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        candidate_count = right - left
        expected = set(range(candidate_count))
        if set(scores[query]) != expected:
            raise RuntimeError(
                f"query {query} SIRIUS candidate coverage mismatch: "
                f"expected {candidate_count}, got {len(scores[query])}"
            )
        scored_candidate_rows += candidate_count
        for candidate, body in scores[query].items():
            molecule = left + candidate
            if str(manifest["molecule_ik14"][molecule]) != str(body["ik14"]):
                raise RuntimeError(f"query {query} candidate {candidate} IK14 drift")
            if str(manifest["molecule_formula"][molecule]) != str(body["formula"]):
                raise RuntimeError(f"query {query} candidate {candidate} formula drift")

        geometry = candidate_geometry(manifest, cache, query)
        true_candidate = int(geometry["true_candidate"])
        top_score = geometry["top_score"]
        assert isinstance(top_score, dict)
        true_score = float(top_score[true_candidate])
        false_candidates = [candidate for candidate in expected if candidate != true_candidate]
        current_winner = max(false_candidates, key=lambda candidate: top_score[candidate])
        current_error = float(top_score[current_winner]) >= true_score
        phasea_errors += int(current_error)
        formula_top, structure_top = top_source_flags(scores[query], true_candidate)
        true_source = scores[query][true_candidate]

        winner_source = scores[query][current_winner]
        winner_comparable = (
            math.isfinite(float(true_source["tree"]))
            and math.isfinite(float(winner_source["tree"]))
            if str(true_source["formula"]) != str(winner_source["formula"])
            else math.isfinite(float(true_source["csi"]))
            and math.isfinite(float(winner_source["csi"]))
        )
        comparable_current_winners += int(winner_comparable)
        winner_relation = source_relation(
            str(true_source["formula"]), str(winner_source["formula"]),
            float(true_source["tree"]), float(winner_source["tree"]),
            float(true_source["csi"]), float(winner_source["csi"]),
            formula_top, structure_top,
        )
        source_corrected_current_winners += int(current_error and winner_relation is not None)
        source_disagrees_current_winners += int(winner_comparable and winner_relation is None)

        eligible: list[tuple[float, int, str, float]] = []
        for candidate in false_candidates:
            hinge = args.margin + float(top_score[candidate]) - true_score
            if hinge <= 0:
                continue
            active_candidates += 1
            body = scores[query][candidate]
            relation = source_relation(
                str(true_source["formula"]), str(body["formula"]),
                float(true_source["tree"]), float(body["tree"]),
                float(true_source["csi"]), float(body["csi"]),
                formula_top, structure_top,
            )
            if relation is None:
                continue
            source_proven_candidates += 1
            kind, source_delta = relation
            eligible.append((float(top_score[candidate]), candidate, kind, source_delta))
        eligible.sort(key=lambda row: (-row[0], row[1]))
        for false_score, candidate, kind, source_delta in eligible[
            :args.maximum_source_events_per_query
        ]:
            top_row = geometry["top_row"]
            assert isinstance(top_row, dict)
            before = len(writer.anchor)
            writer.append(
                int(geometry["anchor"]), [int(top_row[true_candidate])],
                [int(top_row[candidate])], query, candidate, 0,
                SIRIUS_ERROR_BOUNDARY if current_error else SIRIUS_MARGIN_BOUNDARY,
            )
            if len(writer.anchor) == before:
                duplicate_source_events += 1
                continue
            selected_records.append({
                "manifest_query": query,
                "true_candidate": true_candidate,
                "false_candidate": candidate,
                "source_kind": kind,
                "source_delta": f"{source_delta:.17g}",
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
            source_roles, [SIRIUS_ERROR_BOUNDARY, SIRIUS_MARGIN_BOUNDARY],
        ))),
        "source_candidates_cover_exact_score_table": (
            scored_candidate_rows == sum(len(value) for value in scores.values())
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"SIRIUS native-triplet gates failed: {gates}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_sirius_triplets_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        with (temporary / "source_event_ledger.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            if selected_records:
                writer_csv = csv.DictWriter(
                    handle, fieldnames=list(selected_records[0]), delimiter="\t",
                    lineterminator="\n",
                )
                writer_csv.writeheader()
                writer_csv.writerows(selected_records)
            else:
                handle.write(
                    "manifest_query\ttrue_candidate\tfalse_candidate\tsource_kind\n"
                )
        report = {
            "status": "CHEMAWARE_SIRIUS_NATIVE_TRIPLETS_COMPLETE",
            "initialization": "protected Phase-A +2.1266 pp checkpoint directly",
            "phasea_events_preserved": phasea_events,
            "source_events_added": len(output["anchor_idx"]) - phasea_events,
            "source_error_events": int(np.sum(source_roles == SIRIUS_ERROR_BOUNDARY)),
            "source_margin_events": int(np.sum(source_roles == SIRIUS_MARGIN_BOUNDARY)),
            "source_event_queries": len({int(row["manifest_query"]) for row in selected_records}),
            "source_event_formulas": len({
                str(manifest["query_formula"][int(row["manifest_query"])])
                for row in selected_records
            }),
            "cross_formula_tree_events": sum(
                row["source_kind"] == "cross_formula_tree" for row in selected_records
            ),
            "within_formula_csi_events": sum(
                row["source_kind"] == "within_formula_csi" for row in selected_records
            ),
            "phasea_error_queries_in_source_panel": phasea_errors,
            "comparable_current_winner_queries": comparable_current_winners,
            "source_corrected_phasea_current_winner_queries": source_corrected_current_winners,
            "source_noncorrective_comparable_current_winner_queries": source_disagrees_current_winners,
            "conditional_source_headroom_fraction": (
                source_corrected_current_winners / phasea_errors
                if phasea_errors else 0.0
            ),
            "active_phasea_candidate_boundaries": active_candidates,
            "source_proven_active_candidates": source_proven_candidates,
            "duplicate_source_events_already_in_phasea": duplicate_source_events,
            "margin": args.margin,
            "maximum_source_events_per_query": args.maximum_source_events_per_query,
            "identity_audit": audit_identity_edges(output, args.data),
            "gates": gates,
            "scientific_contract": {
                "changed_component": "triplet source selection only",
                "phasea_reproduction": False,
                "phasea_pool": "copied exactly, then SIRIUS events appended",
                "formula_rule": "true formula must be strict top TreeScore",
                "structure_rule": "true structure must be strict top CSI score within formula",
                "boundary_rule": "only active Phase-A max-reference candidates",
                "reference_rule": "one top positive and one top negative reference",
                "score_fusion": "none; formula and structure scores are never mixed",
                "trainer": "unchanged native DreaMS triplet fine-tuning",
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8", newline="\n",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

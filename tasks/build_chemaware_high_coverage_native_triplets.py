"""Build high-coverage ChemAware pools for the unmodified DreaMS trainer.

Role 2 freezes a small rule-specificity recipe.  That recipe is applied once
to roles 0--1 for optimization and to role 3 for model validation.  Role 4 is
not loaded.  Candidate directions from the same query are grouped into one
native DreaMS anchor with dynamically sampled positive/negative references.
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
from chemaware_native_triplet_mining_core import (
    RecipeSelectionError,
    directional_masks,
    recipe_dict,
    select_recipe,
    summarize_masks,
    validate_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "data/validation/chemaware_multinull_triplet_evidence_v1"
DEFAULT_MANIFEST = ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/chemaware_high_coverage_native_triplets_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-selection-anchors", type=int, default=128)
    parser.add_argument("--min-selection-formulas", type=int, default=64)
    parser.add_argument("--min-specificity-ratio", type=float, default=1.20)
    parser.add_argument("--min-train-anchors", type=int, default=512)
    parser.add_argument("--min-train-formulas", type=int, default=256)
    parser.add_argument("--min-train-candidate-directions", type=int, default=768)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def stop_with_report(output: Path, report: dict[str, object]) -> None:
    """Persist a scientific no-go result before stopping the Slurm pipeline."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_triplet_no_go_", dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(2)


def _true_local_candidates(manifest: Mapping[str, np.ndarray], query: int) -> np.ndarray:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
    result = np.flatnonzero(labels).astype(np.int16)
    if not len(result):
        raise RuntimeError(f"query {query} has no true molecule")
    return result


def build_pool(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    correction: np.ndarray, protection: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    validate_evidence(evidence)
    selected = np.asarray(correction, dtype=bool) | np.asarray(protection, dtype=bool)
    anchors: list[int] = []
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    source_query: list[int] = []
    source_candidate: list[int] = []
    source_candidate_ptr = [0]
    source_event_type: list[int] = []
    skipped_no_distinct_positive = 0
    for evidence_row in np.flatnonzero(selected.any(axis=1)):
        query = int(np.asarray(evidence["query"])[evidence_row])
        anchor = int(manifest["query_row"][query])
        true_locals = _true_local_candidates(manifest, query)
        positives = np.unique(np.concatenate([
            molecule_rows(manifest, query, int(local)) for local in true_locals
        ]))
        positives = positives[positives != anchor]
        if not len(positives):
            skipped_no_distinct_positive += 1
            continue
        negatives_local: list[int] = []
        directions_local: list[int] = []
        directions_type: list[int] = []
        baseline = int(np.asarray(evidence["baseline_candidate"])[evidence_row])
        if np.any(correction[evidence_row]):
            promoted = np.asarray(evidence["proposed_candidate"])[evidence_row, correction[evidence_row]]
            if not np.all(np.isin(promoted, true_locals)):
                raise RuntimeError("correction direction does not promote a true candidate")
            if baseline in true_locals:
                raise RuntimeError("correction query already has a true baseline")
            negatives_local.append(baseline)
            directions_local.extend(map(int, promoted))
            directions_type.extend([1] * len(promoted))
        protected = np.asarray(evidence["proposed_candidate"])[evidence_row, protection[evidence_row]]
        for candidate in protected:
            candidate = int(candidate)
            if candidate in true_locals:
                raise RuntimeError("protection direction selected a true candidate as negative")
            negatives_local.append(candidate)
            directions_local.append(candidate)
            directions_type.append(2)
        negatives_local = sorted(set(negatives_local))
        if not negatives_local:
            continue
        negatives = np.unique(np.concatenate([
            molecule_rows(manifest, query, local) for local in negatives_local
        ]))
        anchors.append(anchor)
        positive_rows.extend(map(int, positives))
        negative_rows.extend(map(int, negatives))
        positive_ptr.append(len(positive_rows))
        negative_ptr.append(len(negative_rows))
        source_query.append(query)
        source_candidate.extend(directions_local)
        source_event_type.extend(directions_type)
        source_candidate_ptr.append(len(source_candidate))
    if not anchors:
        raise RuntimeError("the frozen recipe produced no native DreaMS triplets")
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positive_rows, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negative_rows, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "source_candidate_ptr": np.asarray(source_candidate_ptr, dtype=np.int64),
        "source_candidate": np.asarray(source_candidate, dtype=np.int16),
        "source_event_type": np.asarray(source_event_type, dtype=np.int8),
    }
    formulas = np.asarray(manifest["query_formula"])[pool["source_query"]].astype(str)
    audit = {
        "anchor_queries": int(len(anchors)),
        "candidate_directions": int(len(source_candidate)),
        "correction_directions": int(np.sum(pool["source_event_type"] == 1)),
        "protection_directions": int(np.sum(pool["source_event_type"] == 2)),
        "unique_formulas": int(len(np.unique(formulas))),
        "positive_reference_edges": int(len(positive_rows)),
        "negative_reference_edges": int(len(negative_rows)),
        "skipped_no_distinct_positive": int(skipped_no_distinct_positive),
    }
    return pool, audit


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {
        "train": args.evidence_dir / "train_triplet_evidence.npz",
        "selection": args.evidence_dir / "selection_triplet_evidence.npz",
        "confirmation": args.evidence_dir / "confirmation_triplet_evidence.npz",
    }
    evidence = {name: load_npz(path) for name, path in paths.items()}
    manifest = load_npz(args.manifest)
    for body in evidence.values():
        validate_evidence(body)
    role_formulas = {name: set(body["formula"].astype(str)) for name, body in evidence.items()}
    if any(
        role_formulas[left] & role_formulas[right]
        for left, right in (("train", "selection"), ("train", "confirmation"), ("selection", "confirmation"))
    ):
        raise RuntimeError("formula roles overlap in triplet evidence")

    try:
        recipe, selection, grid = select_recipe(
            evidence["selection"],
            min_anchor_queries=args.min_selection_anchors,
            min_formulas=args.min_selection_formulas,
            min_specificity_ratio=args.min_specificity_ratio,
        )
    except RecipeSelectionError as error:
        ranked = sorted(
            error.rows,
            key=lambda row: (
                int(row["specific_candidate_surplus"]),
                int(row["correct"]["unique_formulas"]),
                int(row["correct"]["anchor_queries"]),
            ),
            reverse=True,
        )
        stop_with_report(args.output, {
            "status": "CHEMAWARE_HIGH_COVERAGE_NATIVE_TRIPLETS_NO_GO",
            "stage": "role2_recipe_selection",
            "reason": str(error),
            "training_started": False,
            "outer_role_4_untouched": True,
            "required": {
                "min_selection_anchors": args.min_selection_anchors,
                "min_selection_formulas": args.min_selection_formulas,
                "min_specificity_ratio": args.min_specificity_ratio,
            },
            "best_rejected_recipes": ranked[:10],
        })
    masks = {
        name: directional_masks(body, recipe, 0)[:2]
        for name, body in evidence.items()
    }
    summaries = {
        name: summarize_masks(evidence[name], *masks[name])
        for name in evidence
    }
    train_summary = summaries["train"]
    gates = {
        "role2_correct_exceeds_every_null": int(selection["specific_candidate_surplus"]) > 0,
        "role2_specificity_ratio": float(selection["specificity_ratio"]) >= args.min_specificity_ratio,
        "train_anchor_queries": train_summary["anchor_queries"] >= args.min_train_anchors,
        "train_candidate_directions": train_summary["candidate_directions"] >= args.min_train_candidate_directions,
        "train_unique_formulas": train_summary["unique_formulas"] >= args.min_train_formulas,
        "formula_roles_disjoint": True,
        "outer_role_4_untouched": True,
    }
    if not all(gates.values()):
        stop_with_report(args.output, {
            "status": "CHEMAWARE_HIGH_COVERAGE_NATIVE_TRIPLETS_NO_GO",
            "stage": "roles01_coverage_gate",
            "reason": "the chemically specific bank is still too small for native DreaMS continuation",
            "training_started": False,
            "outer_role_4_untouched": True,
            "recipe": recipe_dict(recipe),
            "role2_selection": selection,
            "directional_evidence": summaries,
            "gates": gates,
        })

    train_pool, train_pool_audit = build_pool(evidence["train"], manifest, *masks["train"])
    val_pool, val_pool_audit = build_pool(
        evidence["confirmation"], manifest, *masks["confirmation"],
    )
    identity_audit = {
        "train": audit_identity_edges(train_pool, args.data),
        "validation": audit_identity_edges(val_pool, args.data),
    }
    report = {
        "status": "CHEMAWARE_HIGH_COVERAGE_NATIVE_TRIPLETS_COMPLETE",
        "training_runtime": "unmodified DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "custom_component": "triplet mining only",
        "role_contract": {
            "optimization": "formula roles 0-1",
            "recipe_selection": "formula role 2",
            "model_validation": "formula role 3",
            "outer": "formula role 4 untouched and not loaded",
        },
        "recipe": recipe_dict(recipe),
        "role2_selection": selection,
        "selection_grid_rows": int(len(grid)),
        "directional_evidence": summaries,
        "native_pools": {"train": train_pool_audit, "validation": val_pool_audit},
        "identity_audit": identity_audit,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_triplet_bank_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **train_pool)
        np.savez_compressed(temporary / "val_pool.npz", **val_pool)
        (temporary / "recipe.json").write_text(
            json.dumps(recipe_dict(recipe), indent=2), encoding="utf-8",
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

"""D0: freeze the data/evaluation contract for final embedding fine-tuning."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold, strict_metrics, strict_rank


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--p3-dir", type=Path, default=ROOT / "data/validation/g8r_p3_test")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--a4-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_a4_action_teacher")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--formula-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--formal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_allowed(path: Path) -> tuple[set[str], dict]:
    body = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(map(str, body["real_train_primary"]["ik14"]))
    if body.get("p3_query_overlap") != 0 or not allowed:
        raise RuntimeError("P3 training allow-list is malformed")
    return allowed, body


def main() -> None:
    args = arguments()
    required = [args.graph, args.data]
    if args.formal:
        required.extend([
            args.p3_dir / "p3_lock_summary.json",
            args.p3_dir / "p3_p2_allowed_training_ik14.json",
            args.c1_dir / "crossfit_examples.csv.gz",
            args.c1_dir / "decision.json",
            args.a4_dir / "oof_selected_actions.csv.gz",
            args.a4_dir / "decision.json",
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"D0 missing required inputs: {missing}")
    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite immutable D0: {args.output_dir}")

    graph = CandidateGraph(args.graph)
    if args.formal and graph.n_queries != 23876:
        raise RuntimeError(f"formal D0 expects 23,876 queries, observed {graph.n_queries}")

    baseline_ranks = np.asarray([
        strict_rank(graph.official_molecule_scores(query)) for query in range(graph.n_queries)
    ], dtype=np.int16)
    baseline = strict_metrics(baseline_ranks, graph.query_has_near)
    if args.formal and int(baseline["errors"]) != 1805:
        # The integer error count is the exact gate; Recall@1 is derived from it.
        raise RuntimeError(f"official baseline mismatch: {baseline}")

    allowed: set[str] | None = None
    allow_body = None
    if (args.p3_dir / "p3_p2_allowed_training_ik14.json").exists():
        allowed, allow_body = load_allowed(args.p3_dir / "p3_p2_allowed_training_ik14.json")
        overlap_or_outside = set(graph.query_ik14) - allowed
        if overlap_or_outside:
            raise RuntimeError(f"{len(overlap_or_outside)} D0 query identities are outside sealed P3 allow-list")

    identity_counts = {value: int(np.sum(graph.query_ik14 == value)) for value in np.unique(graph.query_ik14)}
    identity_weight = np.asarray([1.0 / identity_counts[value] for value in graph.query_ik14], dtype=np.float32)
    identity_weight /= identity_weight.mean()
    formula_fold = np.asarray([
        stable_fold(str(value), args.formula_folds, args.seed) for value in graph.query_formula
    ], dtype=np.int8)

    with h5py.File(args.data, "r") as handle:
        n_rows = len(handle["INCHIKEY"])
        if np.any(graph.query_row < 0) or np.any(graph.query_row >= n_rows):
            raise RuntimeError("query rows fall outside HDF5")
        # h5py fancy indices must be strictly increasing.  The candidate graph
        # deliberately retains evaluation order, so read sorted unique rows and
        # restore the graph order through the inverse index.
        unique_query_rows, inverse_query_rows = np.unique(
            graph.query_row, return_inverse=True,
        )
        query_identity_values = handle["INCHIKEY"][unique_query_rows][inverse_query_rows]
        hdf5_ik14 = np.asarray([
            (value.decode() if isinstance(value, bytes) else str(value))[:14]
            for value in query_identity_values
        ])
        if not np.array_equal(hdf5_ik14, graph.query_ik14):
            raise RuntimeError("graph query identity does not match HDF5")

    p_arm = {"available": False, "examples": 0, "identities": 0, "row_disjoint": None}
    if (args.c1_dir / "crossfit_examples.csv.gz").exists():
        c1 = pd.read_csv(args.c1_dir / "crossfit_examples.csv.gz")
        c1_decision = json.loads((args.c1_dir / "decision.json").read_text(encoding="utf-8"))
        if args.formal and not c1_decision.get("pass_to_candidate_aware_student"):
            raise RuntimeError("C1 did not pass its pre-registered teacher-space gates")
        needed = {"query_index", "evaluation_positive_row", "teacher_rows", "query_ik14"}
        if not needed.issubset(c1.columns):
            raise RuntimeError(f"C1 schema missing: {sorted(needed - set(c1.columns))}")
        disjoint = True
        for row in c1.itertuples(index=False):
            teacher = {int(value) for value in str(row.teacher_rows).split(";") if value}
            if int(row.evaluation_positive_row) in teacher:
                disjoint = False
                break
        if not disjoint:
            raise RuntimeError("C1 evaluation-positive row leaks into teacher rows")
        if np.any((c1["query_index"].to_numpy() < 0) | (c1["query_index"].to_numpy() >= graph.n_queries)):
            raise RuntimeError("C1 query index is outside D0 graph")
        # Teacher spectra are same-identity evidence, never hidden labels from a
        # different molecule.  Read all relevant HDF5 identities in one sorted pass.
        all_teacher_rows = []
        for value in c1["teacher_rows"].astype(str):
            all_teacher_rows.extend(int(item) for item in value.split(";") if item)
        relevant_rows = np.unique(np.concatenate((
            c1["evaluation_positive_row"].to_numpy(np.int64),
            np.asarray(all_teacher_rows, dtype=np.int64),
        )))
        with h5py.File(args.data, "r") as handle:
            identity_values = handle["INCHIKEY"][relevant_rows]
        row_identity = {
            int(row): (value.decode() if isinstance(value, bytes) else str(value))[:14]
            for row, value in zip(relevant_rows, identity_values)
        }
        for row in c1.itertuples(index=False):
            query_identity = str(row.query_ik14)
            if row_identity[int(row.evaluation_positive_row)] != query_identity:
                raise RuntimeError("C1 evaluation-positive identity mismatch")
            teacher = [int(value) for value in str(row.teacher_rows).split(";") if value]
            if any(row_identity[value] != query_identity for value in teacher):
                raise RuntimeError("C1 teacher contains a different identity")
            if allowed is not None and query_identity not in allowed:
                raise RuntimeError("C1 teacher query is outside sealed P3 allow-list")
        p_arm = {
            "available": True, "examples": int(len(c1)),
            "identities": int(c1["query_ik14"].nunique()), "row_disjoint": True,
        }
        if args.formal and (p_arm["examples"] != 80250 or p_arm["identities"] != 1217):
            raise RuntimeError(f"formal C1 cardinality mismatch: {p_arm}")

    n_arm = {"available": False, "actions": 0, "queries": 0}
    if (args.a4_dir / "oof_selected_actions.csv.gz").exists():
        a4 = pd.read_csv(args.a4_dir / "oof_selected_actions.csv.gz")
        needed = {"query_index", "token", "role", "attenuation", "predicted_utility"}
        if not needed.issubset(a4.columns):
            raise RuntimeError(f"A4 schema missing: {sorted(needed - set(a4.columns))}")
        if np.any((a4["query_index"].to_numpy() < 0) | (a4["query_index"].to_numpy() >= graph.n_queries)):
            raise RuntimeError("A4 query index is outside D0 graph")
        a4_decision = json.loads((args.a4_dir / "decision.json").read_text(encoding="utf-8"))
        if args.formal and (
            not a4_decision.get("formal")
            or a4_decision.get("integrity", {}).get("formula_fold_overlap") != 0
        ):
            raise RuntimeError("A4 action teacher is not formal formula-OOF evidence")
        n_arm = {
            "available": True, "actions": int(len(a4)),
            "queries": int(a4["query_index"].nunique()),
        }

    provenance = {
        "graph": sha256_file(args.graph), "hdf5": sha256_file(args.data),
        "build_script": sha256_file(Path(__file__)),
    }
    for label, path in {
        "p3_lock": args.p3_dir / "p3_lock_summary.json",
        "p3_allow": args.p3_dir / "p3_p2_allowed_training_ik14.json",
        "c1_examples": args.c1_dir / "crossfit_examples.csv.gz",
        "c1_decision": args.c1_dir / "decision.json",
        "a4_actions": args.a4_dir / "oof_selected_actions.csv.gz",
        "a4_decision": args.a4_dir / "decision.json",
    }.items():
        if path.exists():
            provenance[label] = sha256_file(path)

    report = {
        "status": "noise_final_d0_manifest_complete", "formal": args.formal,
        "n_queries": graph.n_queries, "n_identities": int(len(np.unique(graph.query_ik14))),
        "n_formulas": int(len(np.unique(graph.query_formula))),
        "n_near": int(np.sum(graph.query_has_near)), "baseline": baseline,
        "identity_weight_mean": float(identity_weight.mean()),
        "identity_weight_sum_by_identity_minmax": [
            float(min(sum(identity_weight[graph.query_ik14 == value]) for value in np.unique(graph.query_ik14))),
            float(max(sum(identity_weight[graph.query_ik14 == value]) for value in np.unique(graph.query_ik14))),
        ],
        "formula_folds": {str(fold): int(np.sum(formula_fold == fold)) for fold in range(args.formula_folds)},
        "p3_query_identity_overlap": 0 if allowed is not None else None,
        "p_arm": p_arm, "n_arm": n_arm, "contains_p2b_fields": False,
        "provenance": provenance,
        "contract": {
            "positive_molecule": "unique and first", "ties": "count against positive",
            "training_unit": "query with identity-equal weights",
            "candidate_references": "frozen official DreaMS embeddings",
            "P2b": "forbidden as training feature, score, teacher, or loss",
        },
    }
    if args.formal and (not p_arm["available"] or not n_arm["available"]):
        raise RuntimeError("formal D0 requires both audited P-arm and N-arm artifacts")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_final_d0_", dir=args.output_dir.parent))
    try:
        np.savez_compressed(
            staging / "manifest.npz", query_index=np.arange(graph.n_queries, dtype=np.int32),
            query_row=graph.query_row, query_ik14=graph.query_ik14,
            query_formula=graph.query_formula, query_has_near=graph.query_has_near,
            baseline_rank=baseline_ranks, identity_weight=identity_weight,
            formula_fold=formula_fold,
        )
        json_dump(staging / "decision.json", report)
        if args.output_dir.exists():
            if not args.overwrite:
                raise RuntimeError("D0 output appeared while building")
            shutil.rmtree(args.output_dir)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Build an outcome-free fixed positive-evidence action bank on the corrected graph.

The frozen P1 recipe is the historical E10-B top3 transport-then-union action:
matched intensity transport at dose 1.00 followed by recurrent union at dose
0.50, prevalence 0.67, maximum five inserted peaks.  No action outcome, rank,
or no-op decision is computed by this builder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from audit_noise_final_e10_positive_residual_matrix import cell_variant
from audit_noise_final_positive_guided_matrix import reference_profile
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, stable_fold
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows


RECIPE = {
    "selector": "p_top3_transport_then_union",
    "family": "transport_then_union",
    "dose": 1.0,
    "auxiliary_dose": 0.5,
    "minimum_reference_prevalence": 0.67,
    "maximum_transferred_peaks": 5,
    "positive_references": 3,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--development-query-scope", choices=("all", "official_errors"), default="all")
    parser.add_argument("--development-sample-seed", type=int, default=20260906)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def choose_queries(graph: CandidateGraph, folds: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    eligible = np.flatnonzero(folds != args.outer_fold)
    if args.max_queries:
        if args.development_query_scope == "official_errors":
            errors = np.asarray([
                np.sum(graph.official_molecule_scores(int(query))[1:] >= graph.official_molecule_scores(int(query))[0]) > 0
                for query in eligible
            ], dtype=bool)
            eligible = eligible[errors]
        if args.max_queries < len(eligible):
            eligible = np.sort(np.random.default_rng(args.development_sample_seed).choice(
                eligible, args.max_queries, replace=False,
            ))
    return np.asarray(eligible, dtype=np.int64)


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    graph_report_path = args.graph_dir / "report.json"
    required = (graph_path, cache_path, graph_report_path, args.data, args.official_checkpoint,
                args.architecture_checkpoint, args.initial_student_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("corrected graph is not training-authorized")
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be 0..4")

    started = time.time()
    graph = CandidateGraph(graph_path)
    folds = np.asarray([stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula], dtype=np.int8)
    queries = choose_queries(graph, folds, args)
    if len(queries) == 0:
        raise RuntimeError("no P-action queries selected")
    needed: set[int] = set(map(int, graph.query_row[queries]))
    for query in queries:
        _, rows, _, _ = graph.query_block(int(query))
        needed.update(map(int, rows))
    reachable = np.asarray(sorted(needed), dtype=np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    model, _ = load_base_model(args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks)
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
    if not initial_decision_path.is_file():
        raise FileNotFoundError(initial_decision_path)
    initial_decision = json.loads(initial_decision_path.read_text(encoding="utf-8"))
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or package.get("inference_clean_only") is not True
        or package.get("P2b_used") is not False
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or initial_decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
        or initial_decision.get("formal") is not True
        or initial_decision.get("pass_to_multifold") is not True
    ):
        raise RuntimeError("initial E4 checkpoint contract failed")
    model.load_state_dict(package["model_state"], strict=True)
    embeddings = encode_rows(model, store, reachable, device, args.encode_batch_size, args.amp, "fixed-P1-geometry")
    index = {int(row): pos for pos, row in enumerate(reachable)}

    spectra: list[np.ndarray] = []
    rows_out: list[dict[str, object]] = []
    for local, query_value in enumerate(queries):
        query = int(query_value)
        _, candidate_rows, ptr, _ = graph.query_block(query)
        qvector = embeddings[index[int(graph.query_row[query])]]
        pair_scores = embeddings[[index[int(row)] for row in candidate_rows]] @ qvector
        positive_rows = np.asarray(candidate_rows[:int(ptr[1])], dtype=np.int64)
        order = np.argsort(-pair_scores[:int(ptr[1])], kind="stable")[:RECIPE["positive_references"]]
        selected = positive_rows[order]
        clean = store.one(int(graph.query_row[query]))
        references = [store.one(int(row)) for row in selected]
        profile = reference_profile(clean, references, args.fragment_tolerance)
        missing = recurrent_missing_peaks(
            clean, references, args.fragment_tolerance,
            RECIPE["minimum_reference_prevalence"], RECIPE["maximum_transferred_peaks"],
        )
        variant = cell_variant(
            clean, profile, missing, RECIPE["family"], RECIPE["dose"], RECIPE["auxiliary_dose"],
        )
        action_id = f"q{query}|p1_top3_transport_union"
        tensor_index = len(spectra)
        spectra.append(variant.numpy().astype(np.float32, copy=False))
        rows_out.append({
            "action_id": action_id, "query_index": query,
            "query_row": int(graph.query_row[query]), "query_ik14": str(graph.query_ik14[query]),
            "query_formula": str(graph.query_formula[query]), "formula_fold": int(folds[query]),
            "selector": RECIPE["selector"], "attenuation": 0.0, "step": 1,
            "target_path": "", "action_kind": "precomputed_spectrum",
            "action_tensor_index": tensor_index,
            "positive_reference_rows": ",".join(map(str, selected)),
            "available_missing_peaks": int(len(missing)),
        })
        if (local + 1) % 1000 == 0 or local + 1 == len(queries):
            print(f"[fixed-P1-build] {local + 1:,}/{len(queries):,}", flush=True)

    frame = pd.DataFrame(rows_out)
    if frame.action_id.duplicated().any() or np.any(frame.formula_fold.to_numpy(np.int8) == args.outer_fold):
        raise RuntimeError("P action uniqueness or outer-fold exclusion failed")
    formal = args.max_queries == 0
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "training_actions.csv.gz", index=False)
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=np.asarray(frame.action_id.astype(str).tolist(), dtype=str),
            action_spectra=np.stack(spectra).astype(np.float32),
        )
        report = {
            "status": "noise_corrected_fixed_p_action_bank_complete",
            "formal": formal, "formal_training_authorized": formal,
            "outer_formula_fold": args.outer_fold,
            "source_queries": int(len(queries)), "action_rows": int(len(frame)),
            "recipe": RECIPE,
            "contracts": {
                "action_outcomes_computed": False,
                "outer_held_formulas_published": False,
                "real_same_identity_positive_references": True,
                "fixed_recipe_no_outcome_routing": True,
                "action_multiplicity_is_not_training_dose": True,
                "teacher_embedding_or_margin_used": False,
                "P2b": "forbidden", "P3_consumed": False,
            },
            "model_provenance": {
                "initialization": "mature_e4_current_geometry",
                "initial_student_checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
                "initial_decision_sha256": sha256_file(initial_decision_path),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            },
            "provenance": {
                "candidate_graph_sha256": sha256_file(graph_path),
                "embedding_cache_sha256": sha256_file(cache_path),
                "graph_report_sha256": sha256_file(graph_report_path),
                "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "runtime_seconds": time.time() - started,
            "claim_limit": "Outcome-free P action construction only; no encoder improvement is claimed.",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

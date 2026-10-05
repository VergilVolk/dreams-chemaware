"""Evaluate an unchanged historical E4 checkpoint on the corrected graph.

Training is deliberately kept in ``train_noise_e4_faithful_v1.py``.  This
separate process cannot influence gradients, sampling, or checkpoint choice;
it only encodes clean spectra and applies the frozen corrected-graph metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from noise_corrected_fullgraph_evaluation import (
    full_metrics,
    held_metric_evidence,
    official_scores,
    paired_outcome_table,
    score_embedding_query_subset,
)
from noise_final_core import (
    CandidateGraph,
    load_embedding_cache,
    sha256_file,
    stable_fold,
)
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import (
    SpectrumStore,
    encode_rows,
    formula_bootstrap_delta,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--training-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def nested_delta(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in candidate.items():
        reference = baseline.get(key)
        if isinstance(value, dict) and isinstance(reference, dict):
            output[key] = nested_delta(value, reference)
        elif (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and isinstance(reference, (int, float)) and not isinstance(reference, bool)
        ):
            output[key] = float(value - reference)
    return output


def paired_counts(frame: pd.DataFrame) -> dict[str, object]:
    corrected = frame["corrected"].to_numpy(bool)
    introduced = frame["introduced"].to_numpy(bool)
    near = frame["near"].to_numpy(bool)
    return {
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_net_lambda2": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "near": {
            "queries": int(np.sum(near)),
            "corrected": int(np.sum(corrected & near)),
            "introduced": int(np.sum(introduced & near)),
            "risk_net_lambda2": int(
                np.sum(corrected & near) - 2 * np.sum(introduced & near)
            ),
        },
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("corrected-graph E4 evaluation requires an allocated GPU")
    required = (
        args.graph, args.source_manifest, args.embedding_cache, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
        args.student_checkpoint, args.training_decision,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    decision = json.loads(args.training_decision.read_text(encoding="utf-8"))
    configuration = decision.get("configuration", {})
    # These are exactly the fields emitted by the 28c3 historical E4 trainer.
    # Later causal/materialized/injector fields must not be expected here: their
    # presence would mean that the wrong trainer was executed.
    expected = {
        "policy": "curriculum", "action_scope": "all",
        "outer_fold": args.outer_fold,
        "formula_fold_seed": args.formula_fold_seed,
        "epochs": 4, "batch_actions": 4, "views_per_identity": 4,
        "positive_spectra": 4, "negative_molecules": 8,
        "unfreeze_blocks": 1, "run_suffix": "highlr_multifold",
        "eval_batch_size": 128, "n_highest_peaks": 100,
        "bootstrap_resamples": 2000, "amp": False, "smoke": False,
    }
    drift = {
        name: {"expected": value, "observed": configuration.get(name)}
        for name, value in expected.items() if configuration.get(name) != value
    }
    expected_floats = {
        "backbone_lr": 2e-6, "head_lr": 1e-5, "weight_decay": 1e-4,
        "rank_margin": 0.05, "temperature": 0.10,
        "lambda_clean_rank": 1.0, "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25, "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0, "margin_floor_slack": 0.005,
        "safety_ratio": 1.0, "grad_clip": 1.0,
    }
    for name, value in expected_floats.items():
        observed = configuration.get(name)
        if observed is None or not math.isclose(float(observed), value, rel_tol=0, abs_tol=1e-15):
            drift[name] = {"expected": value, "observed": observed}
    if drift:
        raise RuntimeError(f"student is not an unchanged E4 run: {drift}")
    if decision.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError("training decision is incomplete")
    provenance = decision.get("provenance", {})
    expected_provenance = {
        "script_sha256": "28c3b375d270fc2030783938d9390710c2c5ab8f0926a0b4ff507d62afa26885",
        "r0_report_sha256": "4f68bfd950d44d02664f67ae9d7ac6700fd3ea33ae0f914f6e623146c57749b6",
        "r0_actions_sha256": "35d52f13e4441141d622c3d60208e3ae7f11dc554bfc129fef2baa4a1cd27843",
        "graph_sha256": "5f2340751c7521c5a93114e2b134d5796f157148736ad9162d545b84c11d9f71",
        "official_checkpoint_sha256": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    }
    provenance_drift = {
        key: {"expected": value, "observed": provenance.get(key)}
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if provenance_drift:
        raise RuntimeError(f"training provenance is not historical E4: {provenance_drift}")
    contracts = decision.get("contracts", {})
    if (
        contracts.get("teacher") != "forbidden"
        or contracts.get("P2b") != "forbidden"
        or contracts.get("P3_consumed") is not False
        or contracts.get("clean_and_augmented_raw_spectra_train_same_encoder") is not True
        or contracts.get("action_outcomes_used_for_weights_or_selection") is not False
    ):
        raise RuntimeError("historical E4 direct-training contract was not preserved")

    graph = CandidateGraph(args.graph)
    held = np.asarray([
        query for query, formula in enumerate(graph.query_formula)
        if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
    ], dtype=np.int64)
    if not len(held):
        raise RuntimeError("corrected graph has no held queries")
    with np.load(args.source_manifest, allow_pickle=False) as source:
        required_source = {"query_row", "query_ik14", "query_formula", "query_adduct"}
        if required_source - set(source.files):
            raise RuntimeError("corrected source manifest lacks registered metadata")
        for name, observed in (
            ("query_row", graph.query_row),
            ("query_ik14", graph.query_ik14),
            ("query_formula", graph.query_formula),
        ):
            if not np.array_equal(np.asarray(source[name]), observed):
                raise RuntimeError(f"source manifest {name} differs from corrected graph")
        query_adduct = np.asarray(source["query_adduct"], dtype=str)

    # Held formula queries are interleaved.  Select their complete pair blocks
    # explicitly instead of relying on a contiguous suffix.
    held_candidate: list[np.ndarray] = []
    for query in held:
        m_left, m_right = map(int, graph.query_ptr[int(query):int(query) + 2])
        p_left, p_right = map(int, graph.molecule_ptr[[m_left, m_right]])
        held_candidate.append(graph.pair_candidate_row[p_left:p_right])
    reachable = np.unique(np.concatenate((graph.query_row[held], *held_candidate))).astype(np.int64)

    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device(args.device), args.n_highest_peaks,
    )
    package = torch_load_compat(args.student_checkpoint, map_location="cpu")
    checkpoint_identity = {
        "policy": configuration["policy"],
        "action_scope": configuration["action_scope"],
        "seed": int(configuration["seed"]),
        "outer_fold": int(configuration["outer_fold"]),
    }
    checkpoint_identity_drift = {
        key: {"expected": value, "observed": package.get(key)}
        for key, value in checkpoint_identity.items()
        if package.get(key) != value
    }
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or not package.get("inference_clean_only")
        or package.get("P2b_used")
        or checkpoint_identity_drift
    ):
        raise RuntimeError(
            "student checkpoint violates the historical E4 output contract: "
            f"{checkpoint_identity_drift}"
        )
    model.load_state_dict(package["model_state"], strict=True)
    del package
    model.eval()
    encoded = encode_rows(
        model, store, reachable, torch.device(args.device),
        args.eval_batch_size, args.amp, "E4-faithful-corrected",
    )

    candidate_scores = score_embedding_query_subset(graph, reachable, encoded, held)
    baseline_scores = official_scores(graph)
    official_metrics, official_table = full_metrics(
        graph, baseline_scores, query_adduct=query_adduct, queries=held,
    )
    student_metrics, student_table = full_metrics(
        graph, candidate_scores, query_adduct=query_adduct, queries=held,
    )
    paired = paired_outcome_table(official_table, student_table)
    formula_ci = formula_bootstrap_delta(
        official_table["rank"].to_numpy(np.int64),
        student_table["rank"].to_numpy(np.int64),
        graph.query_formula[held],
        args.bootstrap_resamples,
        int(configuration.get("seed", 20260830)),
    )
    evidence = held_metric_evidence(
        graph, {"official": baseline_scores, "student": candidate_scores},
        query_adduct, held,
    )

    # Recompute every held official edge from the registered cache.  A simple
    # self-dot normalization check cannot detect a cache from the wrong graph.
    cache_rows, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    del cache_rows
    try:
        official_reachable = np.asarray([
            cache_embeddings[cache_index[int(row)]] for row in reachable
        ], dtype=np.float32)
    except KeyError as error:
        raise RuntimeError("official corrected cache misses a held graph row") from error
    cache_scores = score_embedding_query_subset(
        graph, reachable, official_reachable, held,
    )
    cache_edge_errors: list[float] = []
    for query in held:
        molecule_left, molecule_right = map(
            int, graph.query_ptr[int(query):int(query) + 2],
        )
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        cache_edge_errors.append(float(np.max(np.abs(
            cache_scores.pair[pair_left:pair_right]
            - baseline_scores.pair[pair_left:pair_right]
        ))))
    official_cache_graph_max_abs_error = float(max(cache_edge_errors))
    if official_cache_graph_max_abs_error > 2e-5:
        raise RuntimeError(
            "official corrected cache does not reproduce frozen graph scores: "
            f"max_abs_error={official_cache_graph_max_abs_error:.8g}"
        )

    report = {
        "status": "noise_e4_faithful_v1_corrected_evaluation_complete",
        "formal": False,
        "evaluation_role": "registered train-side development graph",
        "training_was_external_and_unchanged": True,
        "held_queries": int(len(held)),
        "official": official_metrics,
        "student": student_metrics,
        "student_minus_official": nested_delta(student_metrics, official_metrics),
        "paired_top1_vs_official": {
            **paired_counts(paired),
            "formula_cluster_delta_recall1": formula_ci,
        },
        "contracts": {
            "evaluation_cannot_modify_training": True,
            "clean_spectra_only_at_inference": True,
            "complete_corrected_candidate_blocks": True,
            "same_shared_encoder_for_queries_and_candidates": True,
            "query_formula_fold_exact": True,
            "massspecgym_pairwise_not_nist20_replication": True,
            "teacher_embedding_or_margin_target_used": False,
            "checkpoint_selection_from_held_metrics": False,
            "query_formula_held_out_only": True,
            "reference_spectrum_formula_isolation_enforced": False,
            "source_and_registered_artifacts_byte_exact": True,
            "runtime_cuda_stack_byte_exact": False,
        },
        "configuration": vars(args),
        "runtime_seconds": time.time() - started,
        "model_initialization_loader": initialization,
        "official_cache_graph_max_abs_error": official_cache_graph_max_abs_error,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
            "training_decision_sha256": sha256_file(args.training_decision),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Development evaluation on the registered train-side corrected graph, not a P3 "
            "test claim. E4 holds out query formulas only; spectra on positive/negative "
            "reference sides were not formula-isolated by the historical trainer. Source and "
            "registered artifacts are byte-locked, but the CUDA/runtime stack is not."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_e4_faithful_eval_", dir=args.output_dir.parent))
    try:
        paired.to_csv(staging / "paired_per_query.csv.gz", index=False, compression="gzip")
        np.savez_compressed(staging / "held_metric_evidence.npz", **evidence)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

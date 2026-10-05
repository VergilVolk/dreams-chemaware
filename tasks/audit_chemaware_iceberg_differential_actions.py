"""Frozen-DreaMS audit of candidate-differential ICEBERG peak actions.

Only the formula-held-out inner portion of the identity-unique teacher panel is
evaluated.  Candidate structures and ICEBERG predictions are privileged audit
inputs.  The deployed shared encoder still receives one experimental spectrum.
No weights are changed in this program.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_iceberg_synthetic_embedding import peak_permute  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from chemaware_iceberg_peak_action_core import (  # noqa: E402
    apply_peak_action, differential_evidence, hard_negative_indices,
)
from noise_final_core import CandidateGraph, sha256_file, strict_rank  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "data/validation/chemaware_iceberg_teacher_ledger_260_unique_v2")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic/tokens")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_iceberg_differential_actions_inner_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260904)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def molecule_rank_margin(graph: CandidateGraph, query: int, query_embedding: np.ndarray,
                         row_embedding: np.ndarray, row_position: dict[int, int]) -> tuple[int, float]:
    _, rows, ptr, _ = graph.query_block(query)
    pair = row_embedding[[row_position[int(row)] for row in rows]] @ query_embedding
    molecule = np.maximum.reduceat(pair, ptr[:-1])
    return strict_rank(molecule), float(molecule[0] - np.max(molecule[1:]))


@torch.no_grad()
def encode(model, spectra: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    for left in range(0, len(spectra), batch_size):
        output.append(model(spectra[left:left + batch_size].to(device)).float().cpu().numpy())
    return np.concatenate(output)


def summary(old_rank: np.ndarray, old_margin: np.ndarray, new_rank: np.ndarray,
            new_margin: np.ndarray, near: np.ndarray) -> dict:
    old_ok, new_ok = old_rank == 1, new_rank == 1
    return {
        "queries": int(len(old_rank)),
        "baseline_recall1": float(np.mean(old_ok)),
        "recall1": float(np.mean(new_ok)),
        "delta_recall1": float(np.mean(new_ok) - np.mean(old_ok)),
        "corrected": int(np.sum(~old_ok & new_ok)),
        "introduced": int(np.sum(old_ok & ~new_ok)),
        "rank_changed": int(np.sum(old_rank != new_rank)),
        "delta_mean_margin": float(np.mean(new_margin - old_margin)),
        "near_queries": int(np.sum(near)),
        "baseline_near_recall1": float(np.mean(old_ok[near])) if np.any(near) else None,
        "near_recall1": float(np.mean(new_ok[near])) if np.any(near) else None,
    }


def main() -> None:
    args = arguments()
    started = time.time()
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report["selected"].get("unique_identities") != report["selected"].get("queries"):
        raise RuntimeError("requires the passed globally identity-unique teacher ledger")
    if report["inputs"]["graph_sha256"] != sha256_file(args.graph):
        raise RuntimeError("teacher/graph provenance mismatch")

    graph = CandidateGraph(args.graph)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    prediction = np.load(args.teacher_dir / "iceberg_predictions_f16.npy").astype(np.float32)
    scores = np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True)
    folds = stable_formula_folds(graph.query_formula[selected], args.folds, args.fold_seed)
    if args.inner_fold == args.outer_fold:
        raise RuntimeError("inner and outer folds must differ")
    evaluate_position = np.flatnonzero(folds == args.inner_fold)
    evaluate_query = selected[evaluate_position]

    token_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    row_embedding = np.load(args.token_dir / "official_embeddings_f32.npy").astype(np.float32)
    row_embedding /= np.clip(np.linalg.norm(row_embedding, axis=1, keepdims=True), 1e-12, None)
    row_position = {int(row): index for index, row in enumerate(token_rows)}
    store = SpectrumStore(args.data, graph.query_row[selected], args.n_highest_peaks)

    controls = {
        "correct": (prediction, np.asarray(scores["correct_score"], dtype=np.float32)),
        "candidate_swapped": (
            np.concatenate([np.roll(prediction[int(left):int(right)], 1, axis=0) for left, right in zip(ptr[:-1], ptr[1:])]),
            np.asarray(scores["candidate_swapped_score"], dtype=np.float32),
        ),
        "peak_permuted": (
            peak_permute(prediction, int(report["protocol"]["seed"]) + 41),
            np.asarray(scores["peak_permuted_score"], dtype=np.float32),
        ),
    }
    settings = [
        ("signed_exp", value, 0) for value in (0.25, 0.5, 1.0, 2.0)
    ] + [
        (mode, strength, top_k)
        for mode in ("conflict_attenuate", "support_boost")
        for top_k in (5, 10, 20)
        for strength in (0.25, 0.5, 0.75)
    ]

    clean = torch.stack([store.one(int(graph.query_row[query])) for query in evaluate_query])
    action_tensors: list[torch.Tensor] = []
    labels: list[tuple[str, str, float, int]] = []
    for arm, (arm_prediction, arm_score) in controls.items():
        hard = hard_negative_indices(ptr, arm_score)
        for mode, strength, top_k in settings:
            one_setting = []
            for local_position, teacher_position in enumerate(evaluate_position):
                left = int(ptr[teacher_position])
                evidence = differential_evidence(
                    arm_prediction[left], arm_prediction[int(hard[teacher_position])],
                    clean[local_position, 1:, 0].numpy(),
                )
                one_setting.append(apply_peak_action(
                    clean[local_position], evidence, mode, strength, top_k,
                ))
            action_tensors.extend(one_setting)
            labels.append((arm, mode, strength, top_k))

    device = torch.device(args.device)
    model, _ = load_base_model(args.official_checkpoint, args.architecture_checkpoint,
                               device, args.n_highest_peaks)
    stacked = torch.stack(action_tensors)
    encoded = encode(model, stacked, device, args.batch_size)
    per_setting = len(evaluate_query)

    old_rank = np.empty(per_setting, dtype=np.int16)
    old_margin = np.empty(per_setting, dtype=np.float32)
    for index, query in enumerate(evaluate_query):
        qz = row_embedding[row_position[int(graph.query_row[query])]]
        old_rank[index], old_margin[index] = molecule_rank_margin(
            graph, int(query), qz, row_embedding, row_position,
        )
    results = []
    for setting_index, (arm, mode, strength, top_k) in enumerate(labels):
        block = encoded[setting_index * per_setting:(setting_index + 1) * per_setting]
        new_rank = np.empty(per_setting, dtype=np.int16)
        new_margin = np.empty(per_setting, dtype=np.float32)
        for index, query in enumerate(evaluate_query):
            new_rank[index], new_margin[index] = molecule_rank_margin(
                graph, int(query), block[index], row_embedding, row_position,
            )
        results.append({
            "arm": arm, "mode": mode, "strength": strength, "top_k": top_k,
            **summary(old_rank, old_margin, new_rank, new_margin,
                      graph.query_has_near[evaluate_query]),
        })
    results.sort(key=lambda value: (
        value["delta_recall1"], value["corrected"] - value["introduced"],
        value["delta_mean_margin"]), reverse=True)
    output = {
        "status": "AUDIT_COMPLETE",
        "scope": {
            "weights_changed": False,
            "inner_formula_fold_only": True,
            "outer_fold_evaluated": False,
            "candidate_information_training_only": True,
            "action_changes_observed_intensity_only": True,
        },
        "inputs": {
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(args.teacher_dir / "iceberg_predictions_f16.npy"),
        },
        "split": {
            "inner_queries": int(len(evaluate_query)),
            "inner_unique_identities": int(len(np.unique(graph.query_ik14[evaluate_query]))),
            "inner_unique_formulas": int(len(np.unique(graph.query_formula[evaluate_query]))),
        },
        "protocol": {
            "hard_negative": "minimum ICEBERG entropy distance among non-positive candidates",
            "evidence": "sqrt(max-normalized true prediction) minus sqrt(max-normalized hard-negative prediction), sampled only at observed peaks",
            "retrieval": "modified query embedding versus unchanged official real-reference embeddings; molecule score is max over reference spectra",
            "controls": ["candidate_swapped", "peak_permuted"],
        },
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"top": results[:12], "runtime_seconds": output["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

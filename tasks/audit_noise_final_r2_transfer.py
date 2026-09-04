"""Post-hoc audit of R2 privileged-action transfer into clean shared embeddings.

This is diagnostic only.  It does not tune a checkpoint or touch P3.  It
separates four quantities that the R2 training log alone cannot distinguish:

1. the frozen official action teacher;
2. the trained student's action-view ranking;
3. the trained student's clean-view ranking;
4. train-formula versus held-formula transfer.

The audit fails closed on any provenance, finiteness, or R1 teacher-replay
drift.  P2b and downstream reranker scores are never loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import torch_load_compat  # noqa: E402
from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, sha256_file, strict_rank,
)
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, forward_embeddings, parse_path,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r1_privileged_teacher")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def encode_actions(model, store: SpectrumStore, frame: pd.DataFrame,
                   device: torch.device, batch_size: int, label: str) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for left in range(0, len(frame), batch_size):
        batch = frame.iloc[left:left + batch_size]
        tensors = [
            attenuate_sequence(
                store.one(int(row.query_row)), parse_path(row.target_path),
                float(row.attenuation),
            )
            for row in batch.itertuples(index=False)
        ]
        encoded = forward_embeddings(model, torch.stack(tensors).to(device), False)
        encoded = encoded.float().cpu().numpy()
        if not np.all(np.isfinite(encoded)):
            bad = np.flatnonzero(~np.all(np.isfinite(encoded), axis=1))
            raise RuntimeError(
                f"{label} action embeddings non-finite for table rows "
                f"{(left + bad[:50]).tolist()}"
            )
        output.append(encoded)
        right = min(left + batch_size, len(frame))
        if right == len(frame) or right % (batch_size * 10) == 0:
            print(f"[{label}] {right:,}/{len(frame):,} actions", flush=True)
    return np.concatenate(output, axis=0)


def candidate_scores(graph: CandidateGraph, query: int, query_z: np.ndarray,
                     row_z: np.ndarray, row_position: dict[int, int]) -> np.ndarray:
    left, right = map(int, graph.query_ptr[query:query + 2])
    pair_left = int(graph.molecule_ptr[left])
    pair_right = int(graph.molecule_ptr[right])
    rows = graph.pair_candidate_row[pair_left:pair_right]
    positions = np.asarray([row_position[int(row)] for row in rows], dtype=np.int64)
    pair_scores = row_z[positions] @ query_z
    local_ptr = graph.molecule_ptr[left:right + 1] - pair_left
    scores = np.maximum.reduceat(pair_scores, local_ptr[:-1])
    if len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise RuntimeError(f"invalid candidate scores for query {query}")
    return scores.astype(np.float64, copy=False)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = values / temperature
    shifted -= shifted.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def kl_divergence(teacher: np.ndarray, student: np.ndarray, temperature: float) -> float:
    p = softmax(teacher, temperature)
    q = softmax(student, temperature)
    return float(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12)))))


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"queries": 0}
    baseline_correct = frame["baseline_rank"].to_numpy(int) == 1
    clean_correct = frame["student_clean_rank"].to_numpy(int) == 1
    return {
        "queries": int(len(frame)),
        "identities": int(frame["query_ik14"].nunique()),
        "formulas": int(frame["query_formula"].nunique()),
        "baseline_accuracy": float(np.mean(baseline_correct)),
        "fixed_teacher_accuracy": float(np.mean(frame["fixed_teacher_rank"].to_numpy(int) == 1)),
        "student_action_accuracy": float(np.mean(frame["student_action_rank"].to_numpy(int) == 1)),
        "student_clean_accuracy": float(np.mean(clean_correct)),
        "clean_corrected": int(np.sum(~baseline_correct & clean_correct)),
        "clean_introduced": int(np.sum(baseline_correct & ~clean_correct)),
        "mean_fixed_teacher_margin": float(frame["fixed_teacher_margin"].mean()),
        "mean_student_clean_margin": float(frame["student_clean_margin"].mean()),
        "mean_student_action_margin": float(frame["student_action_margin"].mean()),
        "mean_official_clean_to_fixed_action_cos": float(frame["official_clean_to_fixed_action_cos"].mean()),
        "mean_student_clean_to_fixed_action_cos": float(frame["student_clean_to_fixed_action_cos"].mean()),
        "mean_student_clean_to_student_action_cos": float(frame["student_clean_to_student_action_cos"].mean()),
        "mean_fixed_teacher_to_student_clean_kl": float(frame["fixed_teacher_to_student_clean_kl"].mean()),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite audit: {args.output_dir}")
    if args.outer_fold not in range(5) or args.temperature <= 0:
        raise ValueError("invalid outer fold or temperature")
    required = [
        args.checkpoint, args.graph, args.data, args.embedding_cache,
        args.official_checkpoint, args.architecture_checkpoint,
        args.r1_dir / "report.json", args.r1_dir / "corrective_teacher_actions.csv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    graph = CandidateGraph(args.graph)
    action = pd.read_csv(args.r1_dir / "corrective_teacher_actions.csv.gz")
    if len(action) != 882 or action["query_index"].duplicated().any():
        raise RuntimeError("formal R1 corrective table drifted")
    reachable = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    _, official_cache, cache_position = load_embedding_cache(args.embedding_cache)
    if set(map(int, reachable)) - set(cache_position):
        raise RuntimeError("official cache does not cover graph")
    official_rows = np.stack([official_cache[cache_position[int(row)]] for row in reachable])
    row_position = {int(row): index for index, row in enumerate(reachable)}

    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks,
    )
    fixed_action = encode_actions(model, store, action, device, args.batch_size, "fixed-teacher")
    package = torch_load_compat(args.checkpoint, map_location="cpu")
    if package.get("status") != "noise_final_r2_shared_dreams_encoder" or package.get("P2b_used"):
        raise RuntimeError("checkpoint is not a P2b-free R2 shared encoder")
    if int(package.get("outer_fold", -1)) != args.outer_fold:
        raise RuntimeError("checkpoint outer fold mismatch")
    model.load_state_dict(package["model_state"], strict=True)
    model.to(device).eval()
    student_rows = encode_rows(model, store, reachable, device, args.batch_size, False, "student-clean")
    student_action = encode_actions(model, store, action, device, args.batch_size, "student-action")

    records = []
    for index, row in enumerate(action.itertuples(index=False)):
        query = int(row.query_index)
        query_row = int(row.query_row)
        qpos = row_position[query_row]
        teacher_scores = candidate_scores(
            graph, query, fixed_action[index], official_rows, row_position,
        )
        student_clean_scores = candidate_scores(
            graph, query, student_rows[qpos], student_rows, row_position,
        )
        student_action_scores = candidate_scores(
            graph, query, student_action[index], student_rows, row_position,
        )
        record = {
            "query_index": query,
            "query_row": query_row,
            "query_ik14": str(row.query_ik14),
            "query_formula": str(row.query_formula),
            "formula_fold": int(row.formula_fold),
            "teacher_source": str(row.teacher_source),
            "selector": str(row.selector),
            "has_near": bool(row.has_near),
            "baseline_rank": int(row.baseline_rank),
            "recorded_teacher_rank": int(row.teacher_rank),
            "fixed_teacher_rank": strict_rank(teacher_scores),
            "student_clean_rank": strict_rank(student_clean_scores),
            "student_action_rank": strict_rank(student_action_scores),
            "fixed_teacher_margin": float(teacher_scores[0] - teacher_scores[1:].max()),
            "student_clean_margin": float(student_clean_scores[0] - student_clean_scores[1:].max()),
            "student_action_margin": float(student_action_scores[0] - student_action_scores[1:].max()),
            "official_clean_to_fixed_action_cos": float(official_rows[qpos] @ fixed_action[index]),
            "student_clean_to_fixed_action_cos": float(student_rows[qpos] @ fixed_action[index]),
            "student_clean_to_student_action_cos": float(student_rows[qpos] @ student_action[index]),
            "fixed_teacher_to_student_clean_kl": kl_divergence(
                teacher_scores, student_clean_scores, args.temperature,
            ),
        }
        records.append(record)
    result = pd.DataFrame(records)
    replay_mismatch = result["fixed_teacher_rank"].ne(result["recorded_teacher_rank"])
    replay_mismatch_count = int(replay_mismatch.sum())
    replay_mismatch_fraction = replay_mismatch_count / len(result)
    if replay_mismatch_fraction > 0.005:
        raise RuntimeError(
            f"fixed R1 teacher replay drift exceeds 0.5%: "
            f"{replay_mismatch_count}/{len(result)} queries"
        )

    train = result.loc[result["formula_fold"].ne(args.outer_fold)]
    held = result.loc[result["formula_fold"].eq(args.outer_fold)]
    by_source = {
        f"{split_name}|{source}": summarize(part)
        for split_name, split in (("train", train), ("held", held))
        for source, part in split.groupby("teacher_source", sort=True)
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output_dir / "per_query.csv.gz", index=False)
    report = {
        "status": "noise_final_r2_transfer_audit_complete",
        "formal": True,
        "checkpoint": str(args.checkpoint),
        "initialization": initialization,
        "fixed_teacher_replay": {
            "mismatches": replay_mismatch_count,
            "mismatch_fraction": replay_mismatch_fraction,
            "mismatch_query_indices": result.loc[
                replay_mismatch, "query_index"
            ].astype(int).tolist(),
            "mismatch_recomputed_margins": result.loc[
                replay_mismatch, "fixed_teacher_margin"
            ].astype(float).tolist(),
            "policy": "reported explicitly; tolerated only when fraction <= 0.005",
        },
        "train_formula_actions": summarize(train),
        "held_formula_actions": summarize(held),
        "by_teacher_source": by_source,
        "diagnostic_questions": {
            "did_clean_training_actions_become_correct": float(np.mean(train["student_clean_rank"] == 1)),
            "did_clean_held_actions_become_correct": float(np.mean(held["student_clean_rank"] == 1)),
            "did_student_preserve_action_teacher": float(np.mean(result["student_action_rank"] == 1)),
            "did_clean_move_toward_fixed_action_target": float(
                np.mean(result["student_clean_to_fixed_action_cos"] - result["official_clean_to_fixed_action_cos"])
            ),
        },
        "contracts": {
            "diagnostic_only": True,
            "P2b": "forbidden",
            "P3_consumed": False,
            "fixed_teacher_is_official_DreaMS_on_R1_action_view": True,
        },
        "provenance": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "graph_sha256": sha256_file(args.graph),
            "r1_report_sha256": sha256_file(args.r1_dir / "report.json"),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "post-hoc transfer diagnosis; no model selection or test-set claim",
    }
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

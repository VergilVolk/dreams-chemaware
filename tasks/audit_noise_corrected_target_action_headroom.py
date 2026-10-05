"""Measure target-only action headroom without feeding outcomes to training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, strict_rank
from noise_v3_core import attenuate_sequence
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, forward_embeddings, parse_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--action-bank-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rank_margin(
    graph: CandidateGraph, query: int, vector: np.ndarray,
    embeddings: np.ndarray, row_index: dict[int, int],
) -> tuple[int, float]:
    _, rows, ptr, _ = graph.query_block(query)
    score = embeddings[[row_index[int(row)] for row in rows]] @ vector
    molecule = np.maximum.reduceat(score, ptr[:-1])
    return strict_rank(molecule), float(molecule[0] - np.max(molecule[1:]))


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    action_path = args.action_bank_dir / "training_actions.csv.gz"
    action_report_path = args.action_bank_dir / "report.json"
    for path in (graph_path, cache_path, action_path, action_report_path, args.data,
                 args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph = CandidateGraph(graph_path)
    rows, clean_embeddings, row_index = load_embedding_cache(cache_path)
    actions = pd.read_csv(action_path, low_memory=False)
    action_report = json.loads(action_report_path.read_text(encoding="utf-8"))
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device(args.device), args.n_highest_peaks,
    )
    model_provenance: dict[str, object] = {"kind": initialization}
    expected = action_report.get("model_provenance", {})
    if args.initial_student_checkpoint is not None:
        package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
        if (
            package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or expected.get("initial_student_checkpoint_sha256")
            != sha256_file(args.initial_student_checkpoint)
        ):
            raise RuntimeError("headroom model differs from action-bank geometry")
        model.load_state_dict(package["model_state"], strict=True)
        # A mature checkpoint needs its own clean reference bank.  This audit
        # intentionally refuses to score it against official candidates.
        raise RuntimeError("mature headroom requires a separately encoded current-geometry cache")
    if expected.get("official_checkpoint_sha256") != sha256_file(args.official_checkpoint):
        raise RuntimeError("official model differs from action-bank geometry")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    store = SpectrumStore(args.data, np.unique(actions.query_row.to_numpy(np.int64)), args.n_highest_peaks)
    metadata = []
    variants = []
    for row in actions.itertuples(index=False):
        variants.append(attenuate_sequence(
            store.one(int(row.query_row)), parse_path(row.target_path), float(row.attenuation),
        ))
        metadata.append(row)
    vectors = []
    device = torch.device(args.device)
    with torch.inference_mode():
        for left in range(0, len(variants), args.batch_size):
            block = torch.stack(variants[left:left + args.batch_size]).to(device)
            vectors.append(forward_embeddings(model, block, args.amp).float().cpu().numpy())
    encoded = np.concatenate(vectors)
    records = []
    clean_cache: dict[int, tuple[int, float]] = {}
    for row, vector in zip(metadata, encoded):
        query = int(row.query_index)
        if query not in clean_cache:
            clean_cache[query] = rank_margin(
                graph, query, clean_embeddings[row_index[int(row.query_row)]],
                clean_embeddings, row_index,
            )
        clean_rank, clean_margin = clean_cache[query]
        target_rank, target_margin = rank_margin(
            graph, query, vector, clean_embeddings, row_index,
        )
        records.append({
            "action_id": str(row.action_id), "query_index": query,
            "query_row": int(row.query_row), "query_ik14": str(row.query_ik14),
            "query_formula": str(row.query_formula), "selector": str(row.selector),
            "attenuation": float(row.attenuation), "step": int(row.step),
            "clean_rank": clean_rank, "clean_margin": clean_margin,
            "target_rank": target_rank, "target_margin": target_margin,
            "margin_change": target_margin - clean_margin,
            "corrected": bool(clean_rank > 1 and target_rank == 1),
            "introduced": bool(clean_rank == 1 and target_rank > 1),
        })
    frame = pd.DataFrame(records)
    per_query = frame.groupby("query_index", sort=True).agg(
        clean_rank=("clean_rank", "first"), best_target_rank=("target_rank", "min"),
        best_margin_change=("margin_change", "max"), actions=("action_id", "size"),
        query_formula=("query_formula", "first"),
    ).reset_index()
    per_query["any_corrected"] = per_query.clean_rank.gt(1) & per_query.best_target_rank.eq(1)
    per_query["all_targets_safe_top1"] = [
        bool(group.target_rank.eq(1).all()) if int(group.clean_rank.iloc[0]) == 1 else True
        for _, group in frame.groupby("query_index", sort=True)
    ]
    cells = frame.groupby(["selector", "step"], as_index=False).agg(
        actions=("action_id", "size"), queries=("query_index", "nunique"),
        corrected=("corrected", "sum"), introduced=("introduced", "sum"),
        mean_margin_change=("margin_change", "mean"),
    )
    report = {
        "status": "noise_corrected_target_action_headroom_complete",
        "formal": False,
        "actions": int(len(frame)), "queries": int(len(per_query)),
        "baseline_error_queries": int(per_query.clean_rank.gt(1).sum()),
        "queries_with_any_correcting_target": int(per_query.any_corrected.sum()),
        "baseline_correct_queries_with_any_introduction": int(frame.loc[
            frame.introduced, "query_index"
        ].nunique()),
        "mean_action_margin_change": float(frame.margin_change.mean()),
        "positive_margin_change_fraction": float(frame.margin_change.gt(0).mean()),
        "cells": cells.to_dict("records"),
        "contracts": {
            "outcomes_written_separately_from_action_bank": True,
            "outcomes_not_training_authorization": True,
            "teacher_or_distillation_used": False,
            "candidate_reference_geometry_matches_action_bank": True,
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "embedding_cache_sha256": sha256_file(cache_path),
            "action_bank_report_sha256": sha256_file(action_report_path),
            "action_bank_sha256": sha256_file(action_path),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Small development action-headroom audit; not encoder improvement or a training manifest.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_action_headroom_", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "action_outcomes.csv.gz", index=False, compression="gzip")
        per_query.to_csv(staging / "per_query.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

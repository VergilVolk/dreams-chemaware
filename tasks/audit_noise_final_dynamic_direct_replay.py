"""Replay the common Phase-A schedule under the exact current encoder geometry.

This stage performs no optimizer step.  It replays every outer-train ledger
action before any epoch schedule is sampled, then records full-candidate
molecule ranks and margins under the same mature shared encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_l0_action_learnability_ledger import (
    encode_action_variants, prepare_candidate_blocks, score_vector,
)
from audit_noise_final_positive_guided_matrix import apply_action, reference_profile
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks, apply_transfer
from noise_final_core import CandidateGraph, sha256_file
from noise_v3_core import attenuate_sequence
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, parse_path
from train_e1_identity import load_base_model, torch_load_compat


def load_replay_model(
    args: argparse.Namespace, device: torch.device, preflight: dict[str, object],
) -> tuple[torch.nn.Module, dict[str, str]]:
    """Load the exact preflight-authorized E4 shared encoder.

    L0's helper is intentionally clean-duplicate-only.  This replay also allows
    a passing fold-matched mature E4 checkpoint, but only because the complete
    action universe is rescored below before any crossfit weight or optimizer
    step is created.
    """
    decision_path = args.clean_checkpoint.parent / "decision.json"
    package = torch_load_compat(args.clean_checkpoint, map_location="cpu")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    contract = str(preflight.get("initialization", {}).get("contract", "l0_exact"))
    configuration = decision.get("configuration", {})
    common = (
        package.get("status") == "noise_final_e4a_direct_shared_dreams_encoder"
        and not package.get("P2b_used")
        and package.get("inference_clean_only") is True
        and decision.get("status") == "noise_final_e4a_direct_augmentation_complete"
        and decision.get("formal") is True
        and int(package.get("outer_fold", -1)) == int(configuration.get("outer_fold", -2))
        and int(configuration.get("outer_fold", -1)) == int(preflight.get("outer_formula_fold", -2))
    )
    if not common:
        raise RuntimeError("replay checkpoint violates the fold-matched E4 shared-encoder contract")
    if contract == "l0_exact":
        if package.get("causal_arm") != "clean_duplicate" or configuration.get("causal_arm") != "clean_duplicate":
            raise RuntimeError("l0_exact replay requires the clean-duplicate checkpoint")
    elif contract == "mature_e4_current_replay":
        if decision.get("pass_to_multifold") is not True:
            raise RuntimeError("mature E4 replay checkpoint did not pass its original gates")
    else:
        raise RuntimeError(f"unknown initialization contract: {contract}")
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    model.load_state_dict(package["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, {
        "initialization_kind": str(initialization),
        "clean_checkpoint_sha256": sha256_file(args.clean_checkpoint),
        "clean_decision_sha256": sha256_file(decision_path),
        "initialization_contract": contract,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation = ROOT / "data/validation"
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=validation / "g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp32-retry-batch-size", type=int, default=16)
    parser.add_argument("--actions-per-chunk", type=int, default=256)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--recurrence-prevalence", type=float, default=0.67)
    parser.add_argument("--recurrence-max-peaks", type=int, default=5)
    parser.add_argument("--advantage-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_rows(value: object) -> tuple[int, ...]:
    rows = tuple(int(part) for part in str(value).split(";") if part.strip())
    if not rows or len(rows) != len(set(rows)):
        raise RuntimeError("P action reference payload must be non-empty and unique")
    return rows


def materialize_pair(
    row: object, store: SpectrumStore, args: argparse.Namespace,
    cache: dict[tuple[object, ...], object],
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = store.one(int(row.query_row))
    if row.source == "N":
        return (
            attenuate_sequence(clean, parse_path(row.target_payload), float(row.dose)),
            attenuate_sequence(clean, parse_path(row.control_payload), float(row.dose)),
        )
    def profile(payload: object) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray]:
        key = ("profile", int(row.query_index), str(payload))
        if key not in cache:
            refs = [store.one(item) for item in parse_rows(payload)]
            prevalence, intensity = reference_profile(clean, refs, args.fragment_tolerance)
            cache[key] = (refs, prevalence, intensity)
        return cache[key]  # type: ignore[return-value]

    target_refs, target_prevalence, target_intensity = profile(row.target_payload)
    control_refs, control_prevalence, control_intensity = profile(row.control_payload)
    if row.source == "P_intensity":
        family = str(row.family).removeprefix("P:")
        return (
            apply_action(clean, target_prevalence, target_intensity, family, float(row.dose)),
            apply_action(clean, control_prevalence, control_intensity, family, float(row.dose)),
        )
    if row.source == "P_transfer":
        def missing(payload: object, refs: list[torch.Tensor]) -> np.ndarray:
            key = ("missing", int(row.query_index), str(payload), args.recurrence_prevalence,
                   args.recurrence_max_peaks)
            if key not in cache:
                cache[key] = recurrent_missing_peaks(
                    clean, refs, args.fragment_tolerance,
                    args.recurrence_prevalence, args.recurrence_max_peaks,
                )
            return cache[key]  # type: ignore[return-value]
        target_missing = missing(row.target_payload, target_refs)
        control_missing = missing(row.control_payload, control_refs)
        family = str(row.family).removeprefix("P:")
        return (
            apply_transfer(clean, target_missing, target_prevalence, family, float(row.dose))[0],
            apply_transfer(clean, control_missing, control_prevalence, family, float(row.dose))[0],
        )
    raise RuntimeError(f"unknown dynamic-direct source: {row.source}")


def main() -> None:
    args = arguments()
    if (args.batch_size < 1 or args.fp32_retry_batch_size < 1
            or args.fp32_retry_batch_size > args.batch_size or args.actions_per_chunk < 1):
        raise ValueError("invalid bounded replay batch sizes")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite dynamic-direct replay: {args.output_dir}")
    required = [
        args.ledger_dir / "report.json", args.ledger_dir / "training_actions.csv.gz",
        args.preflight_dir / "report.json", args.graph, args.data,
        args.official_checkpoint, args.architecture_checkpoint, args.clean_checkpoint,
        args.clean_checkpoint.parent / "decision.json",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    ledger_report = json.loads(required[0].read_text(encoding="utf-8"))
    preflight = json.loads((args.preflight_dir / "report.json").read_text(encoding="utf-8"))
    if (ledger_report.get("status") != "noise_final_dynamic_direct_action_ledger_complete"
            or ledger_report.get("pass_to_gpu_replay") is not True):
        raise RuntimeError("complete outer-train action ledger has not passed")
    if preflight.get("initialization", {}).get("sha256") != sha256_file(args.clean_checkpoint):
        raise RuntimeError("replay checkpoint differs from the exact preflight geometry")
    schedule = pd.read_csv(args.ledger_dir / "training_actions.csv.gz", low_memory=False)
    schedule.insert(0, "schedule_index", np.arange(len(schedule), dtype=np.int64))
    if len(schedule) != int(ledger_report.get("actions", -1)) or schedule["action_id"].duplicated().any():
        raise RuntimeError("replay ledger count or action uniqueness failed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph = CandidateGraph(args.graph)
    payload_rows: set[int] = set()
    for row in schedule.itertuples(index=False):
        if row.source != "N":
            payload_rows.update(parse_rows(row.target_payload))
            payload_rows.update(parse_rows(row.control_payload))
    reachable = np.unique(np.concatenate((
        graph.query_row, graph.pair_candidate_row, np.asarray(sorted(payload_rows), dtype=np.int64),
    ))).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    model, model_provenance = load_replay_model(args, device, preflight)
    current_embeddings = encode_rows(
        model, store, reachable, device, args.batch_size, args.amp, "dynamic-replay-clean",
    )
    row_position = {int(row): index for index, row in enumerate(reachable)}
    query_indices = schedule["query_index"].to_numpy(np.int64)
    blocks = prepare_candidate_blocks(graph, row_position, query_indices)
    clean_rank: dict[int, int] = {}
    clean_margin: dict[int, float] = {}
    for query in np.unique(query_indices):
        vector = current_embeddings[row_position[int(graph.query_row[int(query)])]]
        rank, margin, _, _ = score_vector(vector, current_embeddings, blocks[int(query)], graph)
        clean_rank[int(query)] = int(rank)
        clean_margin[int(query)] = float(margin)

    records: list[dict[str, object]] = []
    action_cache: dict[tuple[object, ...], object] = {}
    started = time.time()
    for left in range(0, len(schedule), args.actions_per_chunk):
        block_frame = schedule.iloc[left:left + args.actions_per_chunk]
        variants: list[torch.Tensor] = []
        for row in block_frame.itertuples(index=False):
            target, control = materialize_pair(row, store, args, action_cache)
            variants.extend((target, control))
        encoded = encode_action_variants(
            model, variants, device, args.batch_size, args.fp32_retry_batch_size, args.amp,
        )
        for local, row in enumerate(block_frame.itertuples(index=False)):
            query = int(row.query_index)
            target_rank, target_margin, _, _ = score_vector(
                encoded[2 * local], current_embeddings, blocks[query], graph,
            )
            control_rank, control_margin, _, _ = score_vector(
                encoded[2 * local + 1], current_embeddings, blocks[query], graph,
            )
            advantage = float(target_margin - control_margin)
            corrected = clean_rank[query] != 1 and target_rank == 1
            introduced = clean_rank[query] == 1 and target_rank != 1
            positive = bool(corrected or (not introduced and advantage >= args.advantage_threshold))
            harmful = bool(introduced or advantage <= -args.advantage_threshold)
            records.append({
                "action_id": str(row.action_id), "schedule_index": int(row.schedule_index),
                "query_index": query, "identity": str(row.identity), "formula": str(row.formula),
                "source": str(row.source), "family": str(row.family), "cell_id": str(row.cell_id),
                "clean_rank": clean_rank[query], "clean_margin": clean_margin[query],
                "target_rank": int(target_rank), "target_margin": float(target_margin),
                "control_rank": int(control_rank), "control_margin": float(control_margin),
                "paired_advantage": advantage, "corrected": corrected,
                "introduced": introduced, "positive": positive, "harmful": harmful,
            })
        right = min(left + args.actions_per_chunk, len(schedule))
        if right == len(schedule) or right % 2048 == 0:
            print(f"[dynamic replay] {right:,}/{len(schedule):,}; {time.time()-started:.0f}s", flush=True)
    outcomes = pd.DataFrame(records).sort_values("schedule_index", kind="stable").reset_index(drop=True)
    if outcomes["action_id"].tolist() != schedule["action_id"].astype(str).tolist():
        raise RuntimeError("replay output order differs from the common schedule")
    overlap = outcomes["positive"] & outcomes["harmful"]
    source_summary = outcomes.groupby("source", sort=True).agg(
        actions=("action_id", "size"), queries=("query_index", "nunique"),
        identities=("identity", "nunique"), formulas=("formula", "nunique"),
        corrected=("corrected", "sum"), introduced=("introduced", "sum"),
        positive=("positive", "sum"), harmful=("harmful", "sum"),
        mean_advantage=("paired_advantage", "mean"),
    ).reset_index()
    report = {
        "status": "noise_final_dynamic_direct_current_geometry_replay_complete",
        "formal": True, "optimizer_steps": 0, "actions": int(len(outcomes)),
        "queries": int(outcomes["query_index"].nunique()),
        "identities": int(outcomes["identity"].nunique()),
        "formulas": int(outcomes["formula"].nunique()),
        "positive_harmful_overlap": int(overlap.sum()),
        "source_summary": source_summary.to_dict(orient="records"),
        "parameters": {
            "fragment_tolerance": args.fragment_tolerance,
            "recurrence_prevalence": args.recurrence_prevalence,
            "recurrence_max_peaks": args.recurrence_max_peaks,
            "advantage_threshold": args.advantage_threshold,
        },
        "contracts": {
            "exact_current_geometry": True, "full_candidate_scoring": True,
            "same_encoder_for_query_and_references": True,
            "same_ledger_membership_target_and_control": True,
            "labels_mutually_nonexclusive_and_overlap_reported": True,
            "optimizer_steps": 0, "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            **model_provenance,
            "ledger_report": sha256_file(args.ledger_dir / "report.json"),
            "ledger_actions": sha256_file(args.ledger_dir / "training_actions.csv.gz"),
            "preflight": sha256_file(args.preflight_dir / "report.json"),
            "graph": sha256_file(args.graph), "script": sha256_file(Path(__file__)),
        },
        "pass_to_current_geometry_crossfit": bool(
            len(outcomes) == len(schedule) and outcomes["source"].nunique() == 3
            and outcomes["family"].nunique() == 8
        ),
        "claim_limit": "No-update current-geometry action/control replay; not a trained embedding result.",
    }
    if not report["pass_to_current_geometry_crossfit"]:
        raise RuntimeError("current-geometry replay coverage gate failed")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dynamic_direct_replay_", dir=args.output_dir.parent))
    try:
        outcomes.to_csv(staging / "current_geometry_outcomes.csv.gz", index=False, compression="gzip")
        source_summary.to_csv(staging / "source_summary.csv", index=False)
        np.savez(
            staging / "current_geometry_embeddings.npz",
            rows=reachable, embeddings=current_embeddings.astype(np.float32),
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""L0: build a full-candidate action-learnability label ledger.

This is Experiment B of the frozen noise-finetuning plan.  It performs no
optimizer step.  Every R0 target action and both frozen matched-random actions
are replayed against the complete candidate list under one shared mature
clean-continuation encoder.  The resulting labels are outcomes for the later
formula-crossfit learnability audit; they are never inference features.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import torch_load_compat  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file, strict_rank  # noqa: E402
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, forward_embeddings, parse_controls, parse_path,
)


EXPECTED_R0_STATUS = "noise_final_r0_faithful_s3a_manifest_complete"
EXPECTED_CHECKPOINT_STATUS = "noise_final_e4a_direct_shared_dreams_encoder"
EXPECTED_DECISION_STATUS = "noise_final_e4a_direct_augmentation_complete"
ADVANTAGE_THRESHOLD = 0.01


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz",
    )
    parser.add_argument(
        "--r0-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=(
            ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution/"
            "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
            "e4a_causal_v1_20260901_causal_clean_duplicate/"
            "seed_20260828/fold_0/final_shared_encoder.pt"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_l0_action_learnability_ledger",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="maximum spectra per GPU forward; this is not the number of three-view actions",
    )
    parser.add_argument(
        "--actions-per-chunk", type=int, default=256,
        help="CPU bookkeeping chunk; each action creates target plus two controls",
    )
    parser.add_argument(
        "--fp32-retry-batch-size", type=int, default=8,
        help="bounded retry size for non-finite AMP embeddings",
    )
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--advantage-threshold", type=float, default=ADVANTAGE_THRESHOLD)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-actions", type=int, default=0, help="debug only; 0 is formal")
    return parser.parse_args()


def molecule_outcome(pair_scores: np.ndarray, local_ptr: np.ndarray) -> tuple[int, float, int]:
    """Return strict rank, positive-minus-best-negative margin, and top molecule."""
    scores = np.asarray(pair_scores, dtype=np.float64)
    ptr = np.asarray(local_ptr, dtype=np.int64)
    if ptr.ndim != 1 or len(ptr) < 3 or ptr[0] != 0 or ptr[-1] != len(scores):
        raise RuntimeError("invalid local candidate pointer")
    molecule_scores = np.maximum.reduceat(scores, ptr[:-1])
    rank = strict_rank(molecule_scores)
    best_negative = float(np.max(molecule_scores[1:]))
    margin = float(molecule_scores[0] - best_negative)
    # Stable first maximum is only a diagnostic identifier. Strict rank above
    # still counts a positive/negative tie against the positive.
    top = int(np.argmax(molecule_scores))
    return rank, margin, top


def transition_label(baseline_rank: int, target_rank: int) -> str:
    baseline_correct = int(baseline_rank) == 1
    target_correct = int(target_rank) == 1
    if baseline_correct and target_correct:
        return "protected_correct"
    if baseline_correct and not target_correct:
        return "introduced"
    if not baseline_correct and target_correct:
        return "corrected"
    return "persistent_wrong"


def advantage_label(value: float, threshold: float) -> str:
    if not np.isfinite(value) or threshold <= 0:
        raise ValueError("invalid advantage or threshold")
    if value >= threshold:
        return "positive"
    if value <= -threshold:
        return "harmful"
    return "neutral"


def validate_action_row(row: object, n_tokens: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    target = parse_path(getattr(row, "target_path"))
    controls = parse_controls(getattr(row, "matched_control_paths"))
    step = int(getattr(row, "step"))
    if len(target) != step or any(len(path) != step for path in controls):
        raise RuntimeError("target/control path length does not equal the frozen step")
    flattened_controls = [token for path in controls for token in path]
    if (
        target in controls or controls[0] == controls[1]
        or set(target).intersection(flattened_controls)
        or len(flattened_controls) != len(set(flattened_controls))
    ):
        raise RuntimeError("target and matched-random paths must be mutually disjoint")
    if any(token <= 0 or token >= n_tokens for path in (target, *controls) for token in path):
        raise RuntimeError("action path contains an invalid fragment-token index")
    return target, controls


def formula_cluster_mean_ci(
    values: np.ndarray, formulas: np.ndarray, resamples: int, seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    formulas = np.asarray(formulas, dtype=str)
    if len(values) == 0 or len(values) != len(formulas) or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid formula-cluster bootstrap input")
    unique, inverse = np.unique(formulas, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        draws[draw] = sums[chosen].sum() / counts[chosen].sum()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def load_clean_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, str]]:
    decision_path = args.clean_checkpoint.parent / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    package = torch_load_compat(args.clean_checkpoint, map_location="cpu")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        package.get("status") != EXPECTED_CHECKPOINT_STATUS
        or package.get("causal_arm") != "clean_duplicate"
        or package.get("P2b_used")
        or not package.get("inference_clean_only")
        or decision.get("status") != EXPECTED_DECISION_STATUS
        or not decision.get("formal")
        or decision.get("configuration", {}).get("causal_arm") != "clean_duplicate"
    ):
        raise RuntimeError("clean-continuation checkpoint violates the frozen E4-A contract")
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
    }


def prepare_candidate_blocks(
    graph: CandidateGraph, row_position: dict[int, int], queries: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    output: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for query in np.unique(queries):
        _, rows, ptr, molecule_left = graph.query_block(int(query))
        positions = np.asarray([row_position[int(row)] for row in rows], dtype=np.int64)
        output[int(query)] = (positions, np.asarray(ptr, dtype=np.int64), int(molecule_left))
    return output


def score_vector(
    vector: np.ndarray, candidate_embeddings: np.ndarray,
    block: tuple[np.ndarray, np.ndarray, int], graph: CandidateGraph,
) -> tuple[int, float, int, str]:
    positions, ptr, molecule_left = block
    pair_scores = candidate_embeddings[positions] @ np.asarray(vector, dtype=np.float32)
    rank, margin, top_local = molecule_outcome(pair_scores, ptr)
    top_global = molecule_left + top_local
    return rank, margin, top_local, str(graph.molecule_ik14[top_global])


@torch.inference_mode()
def encode_action_variants(
    model: torch.nn.Module, variants: list[torch.Tensor], device: torch.device,
    batch_size: int, fp32_retry_batch_size: int, amp: bool,
) -> np.ndarray:
    """Encode action views with bounded AMP forwards and bounded FP32 retries.

    The caller may assemble hundreds of actions on CPU, but this function
    never turns that bookkeeping chunk into one unbounded GPU batch.
    """
    if not variants or batch_size < 1 or fp32_retry_batch_size < 1:
        raise ValueError("invalid action-variant batch")
    dimension = int(model.head.out_features)
    output = np.empty((len(variants), dimension), dtype=np.float32)
    for left in range(0, len(variants), batch_size):
        right = min(left + batch_size, len(variants))
        cpu_batch = torch.stack(variants[left:right])
        device_batch = cpu_batch.to(device)
        encoded = forward_embeddings(model, device_batch, amp).float().cpu().numpy()
        del device_batch
        invalid = ~np.all(np.isfinite(encoded), axis=1)
        if np.any(invalid) and amp:
            bad_local = np.flatnonzero(invalid)
            # The original implementation retried every invalid view in one
            # FP32 forward. That can be much larger than the successful AMP
            # batch and caused the observed 31.7 GiB OOM. Retry in a strict
            # bounded loop after releasing the AMP temporaries.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for retry_left in range(0, len(bad_local), fp32_retry_batch_size):
                retry_indices = bad_local[retry_left:retry_left + fp32_retry_batch_size]
                retry_batch = cpu_batch[retry_indices].to(device)
                retry_encoded = forward_embeddings(model, retry_batch, False).float().cpu().numpy()
                encoded[retry_indices] = retry_encoded
                del retry_batch, retry_encoded
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            invalid = ~np.all(np.isfinite(encoded), axis=1)
        if np.any(invalid):
            raise RuntimeError(
                "non-finite L0 action embeddings after bounded FP32 retry at global view indices "
                f"{(left + np.flatnonzero(invalid)).tolist()}"
            )
        norms = np.linalg.norm(encoded, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 2e-3):
            bad = np.flatnonzero(~np.isfinite(norms) | (np.abs(norms - 1.0) > 2e-3))
            raise RuntimeError(
                "L0 action encoder produced non-unit embeddings at global view indices "
                f"{(left + bad).tolist()}"
            )
        output[left:right] = encoded
        del cpu_batch, encoded
    return output


def summarize_cell(frame: pd.DataFrame, threshold: float, resamples: int, seed: int) -> dict[str, object]:
    transition_counts = frame["transition"].value_counts().to_dict()
    label_counts = frame["advantage_label"].value_counts().to_dict()
    return {
        "actions": int(len(frame)),
        "queries": int(frame["query_index"].nunique()),
        "identities": int(frame["query_ik14"].nunique()),
        "formulas": int(frame["query_formula"].nunique()),
        "mean_target_margin_gain": float(frame["target_margin_gain"].mean()),
        "mean_control_margin_gain": float(frame["control_mean_margin_gain"].mean()),
        "paired_advantage": formula_cluster_mean_ci(
            frame["paired_advantage"].to_numpy(float),
            frame["query_formula"].to_numpy(str), resamples, seed,
        ),
        "positive_actions": int(label_counts.get("positive", 0)),
        "neutral_actions": int(label_counts.get("neutral", 0)),
        "harmful_actions": int(label_counts.get("harmful", 0)),
        "corrected": int(transition_counts.get("corrected", 0)),
        "introduced": int(transition_counts.get("introduced", 0)),
        "protected_correct": int(transition_counts.get("protected_correct", 0)),
        "persistent_wrong": int(transition_counts.get("persistent_wrong", 0)),
        "threshold": float(threshold),
    }


def main() -> None:
    args = arguments()
    formal = args.max_actions == 0
    if (
        args.batch_size < 1 or args.actions_per_chunk < 1 or args.fp32_retry_batch_size < 1
        or args.fp32_retry_batch_size > args.batch_size
        or args.bootstrap_resamples < 100 or args.advantage_threshold <= 0
    ):
        raise ValueError("invalid L0 numeric arguments")
    required = [
        args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
        args.clean_checkpoint, args.clean_checkpoint.parent / "decision.json",
        args.r0_dir / "report.json", args.r0_dir / "training_actions.csv.gz",
        args.r0_dir / "outcome_audit_only.csv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"completed-output path already exists: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph = CandidateGraph(args.graph)
    if formal and graph.n_queries != 23876:
        raise RuntimeError(f"formal L0 requires 23,876 graph queries, observed {graph.n_queries}")
    r0_report = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    if (
        r0_report.get("status") != EXPECTED_R0_STATUS
        or not r0_report.get("formal")
        or int(r0_report.get("contracts", {}).get("matched_controls_preserved", -1)) != 2
        or r0_report.get("contracts", {}).get("P2b") != "forbidden"
    ):
        raise RuntimeError("R0 source violates the frozen action-manifest contract")
    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    historical = pd.read_csv(args.r0_dir / "outcome_audit_only.csv.gz", low_memory=False)
    required_columns = {
        "query_index", "query_row", "query_ik14", "query_formula", "has_near",
        "selector", "attenuation", "step", "target_path", "matched_control_paths",
        "formula_fold",
    }
    if missing_columns := required_columns - set(actions.columns):
        raise RuntimeError(f"R0 actions miss columns: {sorted(missing_columns)}")
    if actions.duplicated(["query_index", "selector", "attenuation", "step"]).any():
        raise RuntimeError("R0 contains duplicate action definitions")
    if formal and len(actions) != int(r0_report.get("training_action_rows", -1)):
        raise RuntimeError("formal L0 must score every frozen R0 action")
    historical_required = {
        "query_index", "selector", "attenuation", "step", "baseline_rank",
        "baseline_margin", "target_rank", "target_margin", "random_margin",
        "corrected", "introduced",
    }
    if missing_historical := historical_required - set(historical.columns):
        raise RuntimeError(f"R0 historical audit misses columns: {sorted(missing_historical)}")
    historical_keys = ["query_index", "selector", "attenuation", "step"]
    if len(historical) != len(actions) or not historical[historical_keys].equals(actions[historical_keys]):
        raise RuntimeError("R0 historical outcomes do not align one-to-one with action definitions")
    if args.max_actions:
        actions = actions.iloc[: args.max_actions].copy()
        historical = historical.iloc[: args.max_actions].copy()
    actions = actions.reset_index(drop=True)
    if actions.empty:
        raise RuntimeError("no R0 actions selected")
    query_index = actions["query_index"].to_numpy(np.int64)
    if np.any(query_index < 0) or np.any(query_index >= graph.n_queries):
        raise RuntimeError("R0 query index falls outside the candidate graph")
    if not np.array_equal(actions["query_row"].to_numpy(np.int64), graph.query_row[query_index]):
        raise RuntimeError("R0 query rows do not match the candidate graph")
    if not np.array_equal(actions["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query_index]):
        raise RuntimeError("R0 identities do not match the candidate graph")
    if not np.array_equal(actions["query_formula"].astype(str).to_numpy(), graph.query_formula[query_index]):
        raise RuntimeError("R0 formulas do not match the candidate graph")

    parsed: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for row in actions.itertuples(index=False):
        parsed.append(validate_action_row(row, args.n_highest_peaks + 1))

    reachable_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable_rows, args.n_highest_peaks)
    model, model_provenance = load_clean_model(args, device)
    clean_embeddings = encode_rows(
        model, store, reachable_rows, device, args.batch_size, args.amp, "L0-clean",
    )
    row_position = {int(row): index for index, row in enumerate(reachable_rows)}
    blocks = prepare_candidate_blocks(graph, row_position, query_index)

    unique_queries = np.unique(query_index)
    clean_outcomes: dict[int, tuple[int, float, int, str]] = {}
    for count, query in enumerate(unique_queries, start=1):
        vector = clean_embeddings[row_position[int(graph.query_row[int(query)])]]
        clean_outcomes[int(query)] = score_vector(vector, clean_embeddings, blocks[int(query)], graph)
        if count % 5000 == 0 or count == len(unique_queries):
            print(f"[L0 clean ranks] {count:,}/{len(unique_queries):,}", flush=True)

    records: list[dict[str, object]] = []
    started = time.time()
    with torch.inference_mode():
        for left in range(0, len(actions), args.actions_per_chunk):
            right = min(left + args.actions_per_chunk, len(actions))
            variants: list[torch.Tensor] = []
            for local in range(left, right):
                row = actions.iloc[local]
                clean = store.one(int(row["query_row"]))
                target, controls = parsed[local]
                variants.append(attenuate_sequence(clean, target, float(row["attenuation"])))
                variants.extend(
                    attenuate_sequence(clean, path, float(row["attenuation"])) for path in controls
                )
            vectors = encode_action_variants(
                model, variants, device, args.batch_size,
                args.fp32_retry_batch_size, args.amp,
            )

            for batch_local, action_index in enumerate(range(left, right)):
                row = actions.iloc[action_index]
                query = int(row["query_index"])
                baseline_rank, baseline_margin, baseline_top_local, baseline_top_ik14 = clean_outcomes[query]
                outcomes = [
                    score_vector(vectors[3 * batch_local + offset], clean_embeddings, blocks[query], graph)
                    for offset in range(3)
                ]
                target_rank, target_margin, target_top_local, target_top_ik14 = outcomes[0]
                control_ranks = [int(outcome[0]) for outcome in outcomes[1:]]
                control_margins = [float(outcome[1]) for outcome in outcomes[1:]]
                target_gain = float(target_margin - baseline_margin)
                control_gains = [float(value - baseline_margin) for value in control_margins]
                paired_advantage = float(target_gain - np.mean(control_gains))
                records.append({
                    "action_index": int(action_index),
                    "query_index": query,
                    "query_row": int(row["query_row"]),
                    "query_ik14": str(row["query_ik14"]),
                    "query_formula": str(row["query_formula"]),
                    "has_near": bool(graph.query_has_near[query]),
                    "formula_fold": int(row["formula_fold"]),
                    "selector": str(row["selector"]),
                    "attenuation": float(row["attenuation"]),
                    "step": int(row["step"]),
                    "target_path": str(row["target_path"]),
                    "matched_control_paths": str(row["matched_control_paths"]),
                    "baseline_rank": int(baseline_rank),
                    "baseline_margin": float(baseline_margin),
                    "baseline_top_molecule_local": int(baseline_top_local),
                    "baseline_top_ik14": baseline_top_ik14,
                    "target_rank": int(target_rank),
                    "target_margin": float(target_margin),
                    "target_top_molecule_local": int(target_top_local),
                    "target_top_ik14": target_top_ik14,
                    "control0_rank": control_ranks[0],
                    "control0_margin": control_margins[0],
                    "control0_top_ik14": outcomes[1][3],
                    "control1_rank": control_ranks[1],
                    "control1_margin": control_margins[1],
                    "control1_top_ik14": outcomes[2][3],
                    "target_margin_gain": target_gain,
                    "control0_margin_gain": control_gains[0],
                    "control1_margin_gain": control_gains[1],
                    "control_mean_margin_gain": float(np.mean(control_gains)),
                    "paired_advantage": paired_advantage,
                    "advantage_label": advantage_label(paired_advantage, args.advantage_threshold),
                    "transition": transition_label(baseline_rank, target_rank),
                    "target_changes_top": bool(target_top_local != baseline_top_local),
                    "control0_changes_top": bool(outcomes[1][2] != baseline_top_local),
                    "control1_changes_top": bool(outcomes[2][2] != baseline_top_local),
                    "historical_official_baseline_rank": int(historical.iloc[action_index]["baseline_rank"]),
                    "historical_official_baseline_margin": float(historical.iloc[action_index]["baseline_margin"]),
                    "historical_official_target_rank": int(historical.iloc[action_index]["target_rank"]),
                    "historical_official_target_margin": float(historical.iloc[action_index]["target_margin"]),
                    "historical_official_random_margin": float(historical.iloc[action_index]["random_margin"]),
                    "historical_official_paired_advantage": float(
                        historical.iloc[action_index]["target_margin"]
                        - historical.iloc[action_index]["random_margin"]
                    ),
                })
            if right == len(actions) or right % (args.actions_per_chunk * 20) == 0:
                print(f"[L0 actions] {right:,}/{len(actions):,}; {time.time() - started:.0f}s", flush=True)

    ledger = pd.DataFrame.from_records(records)
    if len(ledger) != len(actions) or ledger["action_index"].duplicated().any():
        raise RuntimeError("L0 action ledger lost or duplicated actions")
    if not np.allclose(
        ledger["paired_advantage"],
        ledger["target_margin_gain"] - ledger[["control0_margin_gain", "control1_margin_gain"]].mean(axis=1),
        atol=1e-8,
    ):
        raise RuntimeError("L0 paired-advantage arithmetic mismatch")

    cells: list[dict[str, object]] = []
    for cell_index, ((selector, attenuation, step), frame) in enumerate(
        ledger.groupby(["selector", "attenuation", "step"], sort=True), start=1
    ):
        summary = summarize_cell(
            frame, args.advantage_threshold, args.bootstrap_resamples, args.seed + cell_index,
        )
        summary.update({
            "cell_id": f"{selector}|a={float(attenuation):.2f}|step={int(step)}",
            "selector": str(selector), "attenuation": float(attenuation), "step": int(step),
        })
        cells.append(summary)
    cell_frame = pd.json_normalize(cells)

    baseline_by_query = ledger.drop_duplicates("query_index")
    positive = ledger["advantage_label"].eq("positive")
    harmful = ledger["advantage_label"].eq("harmful")
    historical_advantage = ledger["historical_official_paired_advantage"].to_numpy(float)
    current_advantage = ledger["paired_advantage"].to_numpy(float)
    historical_labels = np.asarray([
        advantage_label(value, args.advantage_threshold) for value in historical_advantage
    ])
    current_labels = ledger["advantage_label"].astype(str).to_numpy()
    non_neutral_both = (historical_labels != "neutral") & (current_labels != "neutral")
    report = {
        "status": "noise_final_l0_action_learnability_ledger_complete",
        "formal": bool(formal),
        "checkpoint_geometry": "E4-A clean-duplicate continuation; shared query/reference encoder",
        "actions": int(len(ledger)),
        "queries": int(ledger["query_index"].nunique()),
        "identities": int(ledger["query_ik14"].nunique()),
        "formulas": int(ledger["query_formula"].nunique()),
        "candidate_reference_spectra": int(len(reachable_rows)),
        "baseline_errors_among_action_covered_queries": int(np.sum(baseline_by_query["baseline_rank"] != 1)),
        "advantage_threshold": float(args.advantage_threshold),
        "label_counts": {key: int(value) for key, value in ledger["advantage_label"].value_counts().items()},
        "transition_counts": {key: int(value) for key, value in ledger["transition"].value_counts().items()},
        "positive_action_queries": int(ledger.loc[positive, "query_index"].nunique()),
        "positive_action_identities": int(ledger.loc[positive, "query_ik14"].nunique()),
        "positive_action_formulas": int(ledger.loc[positive, "query_formula"].nunique()),
        "harmful_action_queries": int(ledger.loc[harmful, "query_index"].nunique()),
        "historical_to_mature_action_staleness": {
            "paired_advantage_pearson": float(np.corrcoef(historical_advantage, current_advantage)[0, 1]),
            "three_class_label_agreement": float(np.mean(historical_labels == current_labels)),
            "non_neutral_in_both": int(np.sum(non_neutral_both)),
            "non_neutral_direction_agreement": (
                float(np.mean(historical_labels[non_neutral_both] == current_labels[non_neutral_both]))
                if np.any(non_neutral_both) else None
            ),
            "historical_target_rank_equals_mature_target_rank": float(np.mean(
                ledger["historical_official_target_rank"].to_numpy(np.int64)
                == ledger["target_rank"].to_numpy(np.int64)
            )),
            "interpretation": (
                "Historical outcomes are diagnostics only. L1 labels are defined exclusively in the "
                "mature clean-continuation geometry."
            ),
        },
        "cells": cells,
        "contracts": {
            "complete_candidate_list_scored": True,
            "query_and_reference_encoder_shared": True,
            "target_and_two_frozen_matched_controls_scored": True,
            "local_positive_negative_surrogate_used": False,
            "optimizer_steps": 0,
            "outcome_labels_are_not_features": True,
            "next_stage_features_must_be_clean_query_visible": True,
            "formula_crossfit_required_next": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "hdf5_sha256": sha256_file(args.data),
            "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
            "r0_actions_sha256": sha256_file(args.r0_dir / "training_actions.csv.gz"),
            "r0_historical_outcomes_sha256": sha256_file(args.r0_dir / "outcome_audit_only.csv.gz"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            **model_provenance,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "next_stage": (
            "L1 formula-crossfit clean-input action learnability: predict positive/neutral/harmful "
            "and paired advantage using only clean-query-visible covariates"
        ),
        "claim_limit": (
            "Outcome-aware full-candidate label ledger under a frozen mature encoder. It is not a "
            "deployable selector, trained embedding, or retrieval gain."
        ),
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_l0_", dir=args.output_dir.parent))
    try:
        ledger.to_csv(temporary / "action_labels.csv.gz", index=False, compression="gzip")
        cell_frame.to_csv(temporary / "cell_summary.csv", index=False)
        json_dump(temporary / "report.json", report)
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

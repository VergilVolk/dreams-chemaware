"""Audit whether qualified ChemAware actions create useful clean-query gradients.

This is a bounded, no-optimizer mechanism audit.  It keeps the passed action
bank fixed and compares three non-distillation injection operators against the
same clean-query deployment boundary:

* action-query-only: block the much larger shared-candidate gradient;
* action-forward/clean-backward: use action geometry but update the clean path;
* action-routed clean pair: let chemistry select a real positive/negative
  reference edge and train that edge on the clean query.

Correct, candidate-swapped and peak-permuted actions are rebuilt with the
action-bank seed and evaluated at matched dose.  A route can advance only when
its predicted clean-margin gain is positive and exceeds both controls under a
formula-cluster bootstrap.  No weights are updated or saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_action_transfer_core import (  # noqa: E402
    action_routed_clean_pair_loss,
    detached_gradients,
    first_order_loss_decrease,
    first_order_metric_gain,
    gradient_cosine,
    gradient_norm,
    routed_listwise_loss,
    subtract_gradients,
)
from chemaware_direct_action_core import formula_bootstrap  # noqa: E402
from chemaware_iceberg_peak_action_core import (  # noqa: E402
    apply_peak_action,
    differential_evidence,
    hard_negative_indices,
)
from chemaware_retrieval_graph import RetrievalGraph  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, forward_embeddings  # noqa: E402


ARMS = ("correct", "candidate_swapped", "peak_permuted")
ROUTES = (
    "action_query_only",
    "action_forward_clean_backward",
    "action_routed_clean_pair",
    "naive_action_minus_clean",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz",
    )
    parser.add_argument(
        "--teacher-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--action-bank", type=Path,
        default=ROOT / "data/validation/chemaware_direct_action_bank_v1",
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
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_action_transfer_gradient_screen_v1",
    )
    parser.add_argument(
        "--max-actions", type=int, default=24,
        help="Formula-stratified action count; 0 means every eligible action.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260906)
    parser.add_argument(
        "--action-generation-seed", type=int, default=20260935,
        help="Seed used when the frozen action-bank peak-permuted control was generated.",
    )
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--pair-target-margin", type=float, default=0.0)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--minimum-positive-fraction", type=float, default=0.60)
    parser.add_argument("--replay-tolerance", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--routes", nargs="+", choices=ROUTES, default=ROUTES,
        help="Injection operators to audit; full confirmation may replay only the screen winner.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, body: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def unfreeze_last_blocks(model, blocks: int) -> dict[str, object]:
    """Minimal local copy to avoid importing an unrelated training pipeline."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    encoder = model.backbone.transformer_encoder
    if blocks < 1 or blocks > int(encoder.n_layers):
        raise ValueError(f"unfreeze-blocks must be in 1..{int(encoder.n_layers)}")
    layers = list(range(int(encoder.n_layers) - blocks, int(encoder.n_layers)))
    for layer in layers:
        for module in (
            encoder.atts[layer], encoder.ffs[layer],
            encoder.scales[2 * layer], encoder.scales[2 * layer + 1],
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if getattr(encoder, "pre_norm", False):
        for parameter in encoder.scales[-1].parameters():
            parameter.requires_grad = True
    return {
        "transformer_layers": int(encoder.n_layers),
        "unfrozen_layers": layers,
        "trainable_parameters": int(sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def formula_stratified_positions(
    eligible: np.ndarray, formula: np.ndarray, maximum: int, seed: int,
) -> np.ndarray:
    eligible = np.asarray(eligible, dtype=np.int64)
    formula = np.asarray(formula).astype(str)
    if not len(eligible):
        raise ValueError("action-transfer audit received zero eligible actions")
    if maximum < 0:
        raise ValueError("max-actions must be non-negative")
    if maximum == 0 or maximum >= len(eligible):
        return np.sort(eligible)
    grouped: dict[str, list[int]] = {}
    for position in eligible:
        grouped.setdefault(str(formula[int(position)]), []).append(int(position))
    rng = np.random.default_rng(seed)
    formulas = np.asarray(sorted(grouped), dtype=object)
    rng.shuffle(formulas)
    chosen: list[int] = []
    for name in formulas:
        rows = sorted(grouped[str(name)])
        chosen.append(rows[int(rng.integers(0, len(rows)))])
        if len(chosen) == maximum:
            return np.asarray(sorted(chosen), dtype=np.int64)
    remaining = sorted(set(map(int, eligible)) - set(chosen))
    rng.shuffle(remaining)
    chosen.extend(remaining[: maximum - len(chosen)])
    return np.asarray(sorted(chosen), dtype=np.int64)


def permuted_prediction_row(prediction: np.ndarray, index: int, seed: int) -> np.ndarray:
    """Memory-bounded exact replay of ``peak_permute`` for one global row."""
    output = np.asarray(prediction[int(index)], dtype=np.float32).copy()
    nonzero = np.flatnonzero(output > 0)
    if len(nonzero) < 2:
        return output
    rng = np.random.default_rng(seed + 1_000_003 * (int(index) + 1))
    permutation = rng.permutation(len(nonzero))
    if np.array_equal(permutation, np.arange(len(nonzero))):
        permutation = np.roll(permutation, 1)
    output[nonzero] = output[nonzero][permutation]
    return output


def swapped_prediction_row(
    prediction: np.ndarray, index: int, left: int, right: int,
) -> np.ndarray:
    """Exact one-row replay of ``np.roll(block, 1, axis=0)``."""
    if not left <= index < right:
        raise ValueError("candidate-swapped index lies outside its query block")
    source = right - 1 if index == left else index - 1
    return np.asarray(prediction[source], dtype=np.float32)


def action_tensor(
    arm: str,
    position: int,
    clean: torch.Tensor,
    prediction: np.ndarray,
    query_ptr: np.ndarray,
    hard: dict[str, np.ndarray],
    mode: str,
    strength: float,
    top_k: int,
    action_generation_seed: int,
) -> torch.Tensor:
    left, right = map(int, query_ptr[position:position + 2])
    true_index = left
    negative_index = int(hard[arm][position])
    if arm == "correct":
        true_prediction = np.asarray(prediction[true_index], dtype=np.float32)
        negative_prediction = np.asarray(prediction[negative_index], dtype=np.float32)
    elif arm == "candidate_swapped":
        true_prediction = swapped_prediction_row(prediction, true_index, left, right)
        negative_prediction = swapped_prediction_row(prediction, negative_index, left, right)
    elif arm == "peak_permuted":
        true_prediction = permuted_prediction_row(
            prediction, true_index, action_generation_seed + 41,
        )
        negative_prediction = permuted_prediction_row(
            prediction, negative_index, action_generation_seed + 41,
        )
    else:
        raise ValueError(f"unknown action arm: {arm}")
    evidence = differential_evidence(
        true_prediction, negative_prediction, clean[1:, 0].numpy(),
    )
    return apply_peak_action(clean, evidence, mode, strength, top_k)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def gradient_record(
    raw_gradient,
    update_direction,
    margin_gradient,
    clean_loss_gradient,
) -> dict[str, float | None]:
    raw_norm = gradient_norm(raw_gradient)
    norm = gradient_norm(update_direction)
    margin_norm = gradient_norm(margin_gradient)
    clean_loss_norm = gradient_norm(clean_loss_gradient)
    margin_gain = first_order_metric_gain(update_direction, margin_gradient)
    clean_loss_decrease = first_order_loss_decrease(update_direction, clean_loss_gradient)
    return {
        "raw_gradient_norm": finite_or_none(raw_norm),
        "learning_rate_scaled_update_norm": finite_or_none(norm),
        "predicted_clean_margin_gain": finite_or_none(margin_gain),
        "clean_margin_gain_per_unit_update_norm": finite_or_none(
            margin_gain / norm if norm > 0 else float("nan")
        ),
        "clean_margin_descent_cosine": finite_or_none(
            margin_gain / (norm * margin_norm)
            if norm > 0 and margin_norm > 0 else float("nan")
        ),
        "clean_loss_decrease_per_lr": finite_or_none(clean_loss_decrease),
        "clean_loss_decrease_per_unit_update_norm": finite_or_none(
            clean_loss_decrease / norm if norm > 0 else float("nan")
        ),
        "clean_loss_gradient_cosine": finite_or_none(
            gradient_cosine(update_direction, clean_loss_gradient)
            if clean_loss_norm > 0 else float("nan")
        ),
    }


def summarize(
    records: list[dict], draws: int, seed: int, minimum_fraction: float,
    routes: tuple[str, ...] | list[str] = ROUTES,
) -> dict:
    metric = "clean_margin_gain_per_unit_update_norm"
    by_key = {
        (int(row["action_position"]), str(row["route"]), str(row["arm"])): row
        for row in records
    }
    positions = sorted({int(row["action_position"]) for row in records})
    formulas = {
        int(row["action_position"]): str(row["formula"]) for row in records
    }
    output: dict[str, dict] = {}
    for route_index, route in enumerate(routes):
        correct = np.asarray([
            by_key[(position, route, "correct")][metric] for position in positions
        ], dtype=np.float64)
        swapped = np.asarray([
            by_key[(position, route, "candidate_swapped")][metric] for position in positions
        ], dtype=np.float64)
        permuted = np.asarray([
            by_key[(position, route, "peak_permuted")][metric] for position in positions
        ], dtype=np.float64)
        finite = np.isfinite(correct) & np.isfinite(swapped) & np.isfinite(permuted)
        if not np.all(finite):
            output[route] = {
                "metric": metric,
                "finite_fraction": float(np.mean(finite)),
                "failure": "non-finite or zero-norm gradient influence",
                "gates": {
                    "all_influences_finite": False,
                    "correct_absolute_formula_ci_positive": False,
                    "correct_minus_candidate_swapped_formula_ci_positive": False,
                    "correct_minus_peak_permuted_formula_ci_positive": False,
                    "correct_positive_fraction": False,
                },
                "pass": False,
            }
            continue
        formula = np.asarray([formulas[position] for position in positions])
        absolute = formula_bootstrap(correct, formula, seed + 100 * route_index, draws)
        minus_swapped = formula_bootstrap(
            correct - swapped, formula, seed + 100 * route_index + 1, draws,
        )
        minus_permuted = formula_bootstrap(
            correct - permuted, formula, seed + 100 * route_index + 2, draws,
        )
        gates = {
            "correct_absolute_formula_ci_positive": (
                absolute["formula_cluster_bootstrap_95ci"][0] > 0
            ),
            "correct_minus_candidate_swapped_formula_ci_positive": (
                minus_swapped["formula_cluster_bootstrap_95ci"][0] > 0
            ),
            "correct_minus_peak_permuted_formula_ci_positive": (
                minus_permuted["formula_cluster_bootstrap_95ci"][0] > 0
            ),
            "correct_positive_fraction": float(np.mean(correct > 0)) >= minimum_fraction,
        }
        output[route] = {
            "metric": metric,
            "correct": absolute,
            "correct_minus_candidate_swapped": minus_swapped,
            "correct_minus_peak_permuted": minus_permuted,
            "correct_positive_fraction": float(np.mean(correct > 0)),
            "correct_median": float(np.median(correct)),
            "candidate_swapped_mean": float(np.mean(swapped)),
            "peak_permuted_mean": float(np.mean(permuted)),
            "gates": gates,
            "pass": bool(all(gates.values())),
        }
    return output


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    if args.bootstrap_draws < 10_000:
        raise ValueError("formula-cluster confirmation requires at least 10,000 draws")
    if not 0 < args.minimum_positive_fraction <= 1:
        raise ValueError("minimum-positive-fraction must be in (0, 1]")
    if min(args.backbone_lr, args.head_lr) <= 0:
        raise ValueError("gradient-audit learning rates must be positive")
    if args.temperature <= 0 or args.pair_target_margin < 0 or args.replay_tolerance <= 0:
        raise ValueError("temperature/tolerance must be positive and pair margin non-negative")
    args.routes = tuple(dict.fromkeys(args.routes))
    required = [
        args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
        args.teacher_dir / "report.json", args.teacher_dir / "selected_queries.npy",
        args.teacher_dir / "query_ptr.npy", args.teacher_dir / "iceberg_predictions_f16.npy",
        args.teacher_dir / "scores_and_ranks.npz", args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.action_bank / "report.json", args.action_bank / "action_bank.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    bank_report = json.loads((args.action_bank / "report.json").read_text(encoding="utf-8"))
    teacher_report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if (
        bank_report.get("status") != "CHEMAWARE_DIRECT_ACTION_BANK_PASS"
        or bank_report.get("pass_to_direct_action_pilot") is not True
    ):
        raise RuntimeError("gradient audit requires the passed ChemAware action bank")
    if bank_report.get("provenance", {}).get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("action bank graph provenance drifted")
    if (
        teacher_report.get("status") != "PASS"
        or bank_report.get("provenance", {}).get("teacher_report_sha256")
        != sha256_file(args.teacher_dir / "report.json")
    ):
        raise RuntimeError("action bank teacher provenance drifted")
    if bank_report.get("provenance", {}).get("action_bank_sha256") != sha256_file(
        args.action_bank / "action_bank.npz"
    ):
        raise RuntimeError("action bank payload hash drifted")

    graph = RetrievalGraph(args.graph)
    selected_queries = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    if (
        selected_queries.ndim != 1
        or not len(selected_queries)
        or np.any(selected_queries < 0)
        or np.any(selected_queries >= graph.n_queries)
    ):
        raise RuntimeError("teacher selected-query ledger is invalid")
    query_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    prediction = np.load(
        args.teacher_dir / "iceberg_predictions_f16.npy", mmap_mode="r",
    )
    if (
        prediction.ndim != 2
        or query_ptr.shape != (len(selected_queries) + 1,)
        or int(query_ptr[0]) != 0
        or int(query_ptr[-1]) != len(prediction)
        or np.any(np.diff(query_ptr) < 2)
    ):
        raise RuntimeError("teacher query pointer and prediction matrix are not aligned")
    with np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True) as score_file:
        required_scores = {
            "correct_score", "candidate_swapped_score", "peak_permuted_score",
        }
        if required_scores - set(score_file.files):
            raise RuntimeError(
                f"teacher score ledger lacks controls: {sorted(required_scores - set(score_file.files))}"
            )
        score_values = {
            name: np.asarray(score_file[name], dtype=np.float32) for name in required_scores
        }
    if any(value.shape != (len(prediction),) for value in score_values.values()):
        raise RuntimeError("teacher score vectors do not align with predictions")
    hard = {
        "correct": hard_negative_indices(query_ptr, score_values["correct_score"]),
        "candidate_swapped": hard_negative_indices(
            query_ptr, score_values["candidate_swapped_score"],
        ),
        "peak_permuted": hard_negative_indices(
            query_ptr, score_values["peak_permuted_score"],
        ),
    }
    with np.load(args.action_bank / "action_bank.npz", allow_pickle=False) as payload:
        bank = {name: payload[name] for name in payload.files}
    required_bank = {
        "selected_query", "formula", "setting_mode", "setting_strength", "setting_top_k",
        "selected_setting", "direct_training_eligible", "margins", "old_margin",
    }
    if required_bank - set(bank):
        raise RuntimeError(f"action bank is missing arrays: {sorted(required_bank - set(bank))}")
    if not np.array_equal(np.asarray(bank["selected_query"], dtype=np.int64), selected_queries):
        raise RuntimeError("action bank and teacher query ledgers are not aligned")
    formula = np.asarray(bank["formula"]).astype(str)
    if not np.array_equal(formula, graph.query_formula[selected_queries].astype(str)):
        raise RuntimeError("action bank formula labels drifted from the retrieval graph")
    selected_setting_array = np.asarray(bank["selected_setting"])
    if selected_setting_array.shape != (1,):
        raise RuntimeError("action bank selected_setting must be a scalar vector")
    selected_setting = int(selected_setting_array[0])
    modes = np.asarray(bank["setting_mode"]).astype(str)
    strengths = np.asarray(bank["setting_strength"], dtype=np.float64)
    top_ks = np.asarray(bank["setting_top_k"], dtype=np.int64)
    if not (0 <= selected_setting < len(modes) == len(strengths) == len(top_ks)):
        raise RuntimeError("action bank selected action setting is invalid")
    mode = str(modes[selected_setting])
    strength = float(strengths[selected_setting])
    top_k = int(top_ks[selected_setting])
    if "action_generation_seed" in bank:
        recorded_seed = np.asarray(bank["action_generation_seed"])
        if recorded_seed.shape != (1,) or int(recorded_seed[0]) != args.action_generation_seed:
            raise RuntimeError("requested action-generation seed differs from the frozen bank")
    report_seed = bank_report.get("action_space", {}).get("action_generation_seed")
    if report_seed is not None and int(report_seed) != args.action_generation_seed:
        raise RuntimeError("action-bank report and requested action-generation seed differ")
    report_setting = bank_report.get("action_space", {}).get("selected_setting", {})
    if (
        int(report_setting.get("setting_id", -1)) != selected_setting
        or str(report_setting.get("mode")) != mode
        or not math.isclose(float(report_setting.get("strength", -1)), strength, abs_tol=1e-6)
        or int(report_setting.get("top_k", -1)) != top_k
    ):
        raise RuntimeError("action bank report and payload disagree on the selected action")
    eligible_mask = np.asarray(bank["direct_training_eligible"], dtype=bool)
    if eligible_mask.shape != (len(selected_queries),):
        raise RuntimeError("action bank eligibility mask is not aligned")
    eligible = np.flatnonzero(eligible_mask)
    positions = formula_stratified_positions(
        eligible, formula, args.max_actions, args.selection_seed,
    )
    if len(np.unique(graph.query_ik14[selected_queries[positions]])) != len(positions):
        raise RuntimeError("gradient audit selection is not identity unique")
    bank_margins = np.asarray(bank["margins"], dtype=np.float32)
    old_margins = np.asarray(bank["old_margin"], dtype=np.float32)
    if (
        bank_margins.ndim != 3
        or bank_margins.shape[0] != len(modes)
        or bank_margins.shape[1] != len(ARMS)
        or bank_margins.shape[2] != len(selected_queries)
        or old_margins.shape != (len(selected_queries),)
    ):
        raise RuntimeError("action bank margin arrays are not aligned")
    if (
        not np.all(np.isfinite(old_margins[positions]))
        or not np.all(np.isfinite(bank_margins[selected_setting, :, positions]))
    ):
        raise RuntimeError("selected qualified actions lack frozen replay margins")

    token_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    if len(token_rows) != len(official):
        raise RuntimeError("official embedding cache rows and values are not aligned")
    if official.ndim != 2 or len(np.unique(token_rows)) != len(token_rows):
        raise RuntimeError("official embedding cache is not a unique row-by-embedding matrix")
    row_position = {int(row): index for index, row in enumerate(token_rows)}
    required_rows = set(map(int, graph.query_row[selected_queries[positions]]))
    for position in positions:
        _, rows, _, _ = graph.query_block(int(selected_queries[int(position)]))
        required_rows.update(map(int, rows))
    absent = required_rows - set(row_position)
    if absent:
        raise RuntimeError(f"official embedding cache lacks audit rows: {sorted(absent)[:20]}")
    with h5py.File(args.data, "r") as handle:
        if not {"spectrum", "precursor_mz"}.issubset(handle):
            raise RuntimeError("spectrum HDF5 lacks spectrum/precursor_mz datasets")
        if len(handle["spectrum"]) != len(handle["precursor_mz"]):
            raise RuntimeError("spectrum and precursor HDF5 rows are not aligned")
        query_rows = graph.query_row[selected_queries[positions]]
        if np.any(query_rows < 0) or np.any(query_rows >= len(handle["spectrum"])):
            raise RuntimeError("audit query row lies outside spectrum HDF5")

    preflight = {
        "status": "CHEMAWARE_ACTION_TRANSFER_GRADIENT_PREFLIGHT_PASS",
        "weights_updated": False,
        "eligible_action_bank_actions": int(len(eligible)),
        "audited_actions": int(len(positions)),
        "audited_formulas": int(len(np.unique(formula[positions]))),
        "all_eligible_actions_audited": bool(len(positions) == len(eligible)),
        "positions": positions.tolist(),
        "setting": {
            "setting_id": selected_setting, "mode": mode,
            "strength": strength, "top_k": top_k,
            "action_generation_seed": args.action_generation_seed,
        },
        "routes": list(args.routes),
        "controls": list(ARMS[1:]),
        "contracts": {
            "candidate_embeddings_frozen": True,
            "deployment_molecule_max_scoring": True,
            "correct_and_controls_matched": True,
            "optimizer_steps": 0,
            "teacher_score_regression": False,
            "teacher_embedding_regression": False,
            "teacher_margin_regression": False,
            "outer_fold_evaluated": False,
            "formal_training_authorized": False,
            "first_order_update_uses_layerwise_learning_rates": True,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(
                args.teacher_dir / "iceberg_predictions_f16.npy"
            ),
            "action_bank_sha256": sha256_file(args.action_bank / "action_bank.npz"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        args.output / "preflight.json",
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.set_num_threads(args.torch_threads)
    random.seed(args.selection_seed)
    np.random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.selection_seed)
    device = torch.device(args.device)
    model, initialization_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    learning_rate_scale = [
        args.head_lr if id(parameter) in head_ids else args.backbone_lr
        for parameter in trainable
    ]

    store = SpectrumStore(
        args.data, graph.query_row[selected_queries[positions]], args.n_highest_peaks,
    )
    result_path = args.output / "per_action.jsonl"
    records: list[dict] = []
    replay_errors: list[float] = []
    clean_replay_cosines: list[float] = []

    for audit_index, position_value in enumerate(positions, start=1):
        position = int(position_value)
        query_index = int(selected_queries[position])
        query_row = int(graph.query_row[query_index])
        clean_tensor = store.one(query_row)
        action_tensors = [
            action_tensor(
                arm, position, clean_tensor, prediction, query_ptr, hard,
                mode, strength, top_k, args.action_generation_seed,
            )
            for arm in ARMS
        ]
        spectra = torch.stack([clean_tensor, *action_tensors]).to(device)
        encoded = forward_embeddings(model, spectra, args.amp).float()
        clean_query = encoded[0]
        action_queries = {arm: encoded[index + 1] for index, arm in enumerate(ARMS)}

        _, candidate_rows, molecule_ptr, _ = graph.query_block(query_index)
        candidate_embeddings = torch.from_numpy(np.asarray([
            official[row_position[int(row)]] for row in candidate_rows
        ], dtype=np.float32)).to(device)
        official_clean = torch.from_numpy(
            np.asarray(official[row_position[query_row]], dtype=np.float32)
        ).to(device)
        clean_cosine = float(torch.sum(clean_query.detach() * official_clean).cpu())
        clean_replay_cosines.append(clean_cosine)
        if clean_cosine < 0.999:
            raise RuntimeError(
                "official clean-query initialization replay drifted at action "
                f"position {position}: cosine={clean_cosine}"
            )

        clean_objective = routed_listwise_loss(
            clean_query, clean_query, candidate_embeddings, molecule_ptr,
            route="clean_query_only", temperature=args.temperature,
        )
        clean_replay_error = abs(float(clean_objective.margin.detach().cpu()) - old_margins[position])
        replay_errors.append(clean_replay_error)
        if clean_replay_error > args.replay_tolerance:
            raise RuntimeError(
                "clean boundary does not replay the qualified bank at action "
                f"position {position}: margin error={clean_replay_error:.6g}"
            )
        margin_gradient = detached_gradients(
            clean_objective.margin, trainable, retain_graph=True,
        )
        clean_loss_gradient = detached_gradients(
            clean_objective.loss, trainable, retain_graph=True,
        )
        if gradient_norm(margin_gradient) <= 0 or gradient_norm(clean_loss_gradient) <= 0:
            raise RuntimeError(f"zero clean validation gradient at action position {position}")

        for arm_index, arm in enumerate(ARMS):
            action_query = action_queries[arm]
            action_objective = routed_listwise_loss(
                clean_query, action_query, candidate_embeddings, molecule_ptr,
                route="action_query_only", temperature=args.temperature,
            )
            action_replay_error = abs(
                float(action_objective.margin.detach().cpu())
                - float(bank_margins[selected_setting, arm_index, position])
            )
            replay_errors.append(action_replay_error)
            if action_replay_error > args.replay_tolerance:
                raise RuntimeError(
                    f"{arm} action does not replay the qualified bank at action "
                    f"position {position}: margin error={action_replay_error:.6g}; "
                    "check the frozen action-generation seed"
                )
            action_gradient = (
                detached_gradients(action_objective.loss, trainable, retain_graph=True)
                if {"action_query_only", "naive_action_minus_clean"} & set(args.routes)
                else None
            )
            straight_through = routed_listwise_loss(
                clean_query, action_query, candidate_embeddings, molecule_ptr,
                route="action_forward_clean_backward", temperature=args.temperature,
            )
            straight_gradient = (
                detached_gradients(straight_through.loss, trainable, retain_graph=True)
                if "action_forward_clean_backward" in args.routes else None
            )
            pair = action_routed_clean_pair_loss(
                clean_query, action_query, candidate_embeddings, molecule_ptr,
                temperature=args.temperature, target_margin=args.pair_target_margin,
            )
            pair_gradient = (
                detached_gradients(pair.loss, trainable, retain_graph=True)
                if "action_routed_clean_pair" in args.routes else None
            )
            naive_gradient = (
                subtract_gradients(action_gradient, clean_loss_gradient)
                if "naive_action_minus_clean" in args.routes else None
            )
            route_values = {}
            if "action_query_only" in args.routes:
                route_values["action_query_only"] = (
                    action_gradient, float(action_objective.loss.detach().cpu()),
                    float(action_objective.margin.detach().cpu()), None, None,
                )
            if "action_forward_clean_backward" in args.routes:
                route_values["action_forward_clean_backward"] = (
                    straight_gradient, float(straight_through.loss.detach().cpu()),
                    float(straight_through.margin.detach().cpu()), None, None,
                )
            if "action_routed_clean_pair" in args.routes:
                route_values["action_routed_clean_pair"] = (
                    pair_gradient, float(pair.loss.detach().cpu()),
                    float(pair.margin.detach().cpu()),
                    pair.positive_reference, pair.negative_reference,
                )
            if "naive_action_minus_clean" in args.routes:
                route_values["naive_action_minus_clean"] = (
                    naive_gradient, float(action_objective.loss.detach().cpu()),
                    float(action_objective.margin.detach().cpu()), None, None,
                )
            for route, (gradient, route_loss, route_margin, positive_ref, negative_ref) in (
                route_values.items()
            ):
                update_direction = [
                    value * scale if value is not None else None
                    for value, scale in zip(gradient, learning_rate_scale)
                ]
                record = {
                    "audit_index": audit_index,
                    "action_position": position,
                    "query_index": query_index,
                    "query_row": query_row,
                    "identity": str(graph.query_ik14[query_index]),
                    "formula": str(formula[position]),
                    "arm": arm,
                    "route": route,
                    "route_loss": route_loss,
                    "route_forward_margin": route_margin,
                    "clean_forward_margin": float(clean_objective.margin.detach().cpu()),
                    "positive_reference": positive_ref,
                    "negative_reference": negative_ref,
                    **gradient_record(
                        gradient, update_direction, margin_gradient, clean_loss_gradient,
                    ),
                }
                append_jsonl(result_path, record)
                records.append(record)
                del update_direction
            del action_gradient, straight_gradient, pair_gradient, naive_gradient
        del margin_gradient, clean_loss_gradient, encoded, spectra, candidate_embeddings
        model.zero_grad(set_to_none=True)
        if device.type == "cuda" and audit_index % 4 == 0:
            torch.cuda.empty_cache()
        print(
            f"[gradient audit] {audit_index}/{len(positions)} "
            f"position={position} max_replay_error={max(replay_errors):.3e}",
            flush=True,
        )

    maximum_replay_error = float(max(replay_errors))
    minimum_clean_cosine = float(min(clean_replay_cosines))
    if maximum_replay_error > args.replay_tolerance:
        raise RuntimeError(
            "action reconstruction does not replay the qualified bank: "
            f"max margin error={maximum_replay_error:.6g}"
        )
    if minimum_clean_cosine < 0.999:
        raise RuntimeError(
            f"official clean-query initialization replay drifted: min cosine={minimum_clean_cosine}"
        )

    route_summary = summarize(
        records, args.bootstrap_draws, args.selection_seed + 9001,
        args.minimum_positive_fraction, args.routes,
    )
    passing = [route for route in args.routes if route_summary[route]["pass"]]
    non_naive_passing = [route for route in passing if route != "naive_action_minus_clean"]
    def minimum_lower(route: str) -> float:
        body = route_summary[route]
        return float(min(
            body["correct"]["formula_cluster_bootstrap_95ci"][0],
            body["correct_minus_candidate_swapped"]["formula_cluster_bootstrap_95ci"][0],
            body["correct_minus_peak_permuted"]["formula_cluster_bootstrap_95ci"][0],
        ))
    winner = max(non_naive_passing, key=minimum_lower) if non_naive_passing else None
    all_actions = len(positions) == len(eligible)
    report = {
        "status": (
            "CHEMAWARE_ACTION_TRANSFER_FULL_CONFIRMATION_PASS"
            if winner is not None and all_actions else
            "CHEMAWARE_ACTION_TRANSFER_SCREEN_PASS"
            if winner is not None else
            "CHEMAWARE_ACTION_TRANSFER_GRADIENT_FAIL"
        ),
        "weights_updated": False,
        "optimizer_steps": 0,
        "winner": winner,
        "pass_to_full_gradient_confirmation": bool(winner is not None and not all_actions),
        "pass_to_bounded_embedding_pilot": bool(winner is not None and all_actions),
        "formal_training_authorized": False,
        "audited_actions": int(len(positions)),
        "audited_formulas": int(len(np.unique(formula[positions]))),
        "eligible_action_bank_actions": int(len(eligible)),
        "all_eligible_actions_audited": all_actions,
        "action_replay": {
            "maximum_margin_abs_error": maximum_replay_error,
            "tolerance": args.replay_tolerance,
            "minimum_clean_embedding_cosine": minimum_clean_cosine,
            "action_generation_seed_exactly_replayed": True,
        },
        "capacity": capacity,
        "first_order_update": {
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "optimizer_interpretation": (
                "layerwise learning-rate-scaled gradient descent direction; "
                "Adam moment preconditioning is intentionally not inferred before training"
            ),
        },
        "initialization_kind": initialization_kind,
        "routes": route_summary,
        "interpretation": {
            "positive_metric": (
                "predicted clean positive-vs-hardest-negative margin gain per unit "
                "parameter-update norm"
            ),
            "matched_controls_required": True,
            "same_identity_mechanism_test_only": True,
            "held_out_embedding_performance_established": False,
            "performance_pp_claim": None,
        },
        "runtime_seconds": time.time() - started,
        "preflight": preflight,
    }
    atomic_write_text(
        args.output / "report.json",
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    csv_path = args.output / "per_action.csv"
    with csv_path.with_name(csv_path.name + ".tmp").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    csv_path.with_name(csv_path.name + ".tmp").replace(csv_path)
    atomic_write_text(
        args.output / "COMPLETE.json",
        json.dumps({"status": report["status"], "winner": winner}, indent=2) + "\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

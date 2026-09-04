"""Learn peak-local ICEBERG structure evidence with a spectrum-only branch.

For every experimental query peak, the aligned target has two channels: the
ICEBERG intensity predicted for the true structure and the maximum intensity
predicted by its same-formula competitors.  Candidate structures are used only
to build this offline target.  A raw-spectrum branch must recover the aligned
targets on molecular-formula-held-out spectra better than candidate-swapped
and peak-permuted teachers before the representation may be connected to the
shared embedding residual.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_shared_v2_core import formula_folds  # noqa: E402
from dreams.models.chem_aware.hierarchical_chemical_adapter import (  # noqa: E402
    HierarchicalChemicalResidualAdapter,
)
from noise_final_core import CandidateGraph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz"),
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/tokens"),
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_teacher_ledger_inner_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_peak_evidence_inner_v1"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=20260904)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--evaluate-only-from",
        type=Path,
        default=None,
        help="Evaluate fixed final checkpoints from an earlier run without retraining.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def peak_permute(predicted: np.ndarray, seed: int) -> np.ndarray:
    output = predicted.copy()
    for index in range(len(output)):
        nonzero = np.flatnonzero(output[index] > 0)
        if len(nonzero) < 2:
            continue
        rng = np.random.default_rng(seed + 1_000_003 * (index + 1))
        permutation = rng.permutation(len(nonzero))
        if np.array_equal(permutation, np.arange(len(nonzero))):
            permutation = np.roll(permutation, 1)
        output[index, nonzero] = output[index, nonzero][permutation]
    return output


def normalize_max(prediction: np.ndarray) -> np.ndarray:
    maximum = prediction.max(axis=1, keepdims=True)
    return prediction / np.clip(maximum, 1e-12, None)


def make_target(
    predicted: np.ndarray,
    query_ptr: np.ndarray,
    query_mz: np.ndarray,
    query_valid: np.ndarray,
) -> np.ndarray:
    bins = np.linspace(0.0, 1500.0, 15_000)
    output = np.zeros((*query_mz.shape, 2), dtype=np.float32)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        candidates = normalize_max(predicted[int(left) : int(right)])
        peak_bin = np.digitize(query_mz[query], bins=bins)
        peak_bin = np.clip(peak_bin, 0, len(bins) - 1)
        output[query, :, 0] = np.sqrt(np.maximum(candidates[0, peak_bin], 0.0))
        output[query, :, 1] = np.sqrt(
            np.maximum(np.max(candidates[1:, peak_bin], axis=0), 0.0)
        )
    output[~query_valid] = 0.0
    return output


def load_arrays(args: argparse.Namespace, graph: CandidateGraph) -> dict[str, np.ndarray | dict]:
    report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("teacher ledger failed its prerequisite gate")
    if report["inputs"].get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("teacher/graph provenance mismatch")
    query = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    query_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    predicted = np.load(args.teacher_dir / "iceberg_predictions_f16.npy").astype(np.float32)
    if query_ptr.shape != (len(query) + 1,) or query_ptr[-1] != len(predicted):
        raise RuntimeError("teacher arrays are not aligned")
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    position = {int(row): index for index, row in enumerate(rows)}
    take = np.asarray([position[int(graph.query_row[value])] for value in query], dtype=np.int64)
    arrays = {
        "report": report,
        "query": query,
        "query_ptr": query_ptr,
        "formulas": graph.query_formula[query],
        "rows": graph.query_row[query],
        "official": np.load(args.token_dir / "official_embeddings_f32.npy")[take].astype(np.float32),
        "mz": np.load(args.token_dir / "mz_f32.npy")[take].astype(np.float32),
        "intensity": np.load(args.token_dir / "intensity_f32.npy")[take].astype(np.float32),
        "valid": np.load(args.token_dir / "valid.npy")[take].astype(bool),
        "precursor": np.load(args.token_dir / "precursor_mz_f32.npy")[take].astype(np.float32),
    }
    correct = make_target(predicted, query_ptr, arrays["mz"], arrays["valid"])
    swapped_prediction = predicted.copy()
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        swapped_prediction[int(left) : int(right)] = np.roll(
            predicted[int(left) : int(right)], 1, axis=0
        )
    permuted_prediction = peak_permute(predicted, int(report["protocol"]["seed"]) + 41)
    arrays["targets"] = {
        "correct": correct,
        "candidate_swapped": make_target(
            swapped_prediction, query_ptr, arrays["mz"], arrays["valid"]
        ),
        "peak_permuted": make_target(
            permuted_prediction, query_ptr, arrays["mz"], arrays["valid"]
        ),
    }
    return arrays


def forward_positions(
    model: HierarchicalChemicalResidualAdapter,
    arrays: dict,
    positions: np.ndarray,
) -> torch.Tensor:
    return model(
        torch.from_numpy(arrays["official"][positions]),
        torch.from_numpy(arrays["mz"][positions]),
        torch.from_numpy(arrays["intensity"][positions]),
        torch.from_numpy(arrays["precursor"][positions]),
        torch.from_numpy(arrays["valid"][positions]),
    ).formula_logits


def loss_function(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    intensity: torch.Tensor,
) -> torch.Tensor:
    error = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.1).mean(dim=-1)
    weight = torch.sqrt(intensity.clamp_min(1e-4)) * valid.float()
    return torch.sum(error * weight) / torch.sum(weight).clamp_min(1e-8)


@torch.no_grad()
def evaluate(
    model: HierarchicalChemicalResidualAdapter,
    arrays: dict,
    positions: np.ndarray,
    batch_size: int,
) -> dict:
    model.eval()
    correct_target = arrays["targets"]["correct"]
    per_spectrum_mae = []
    per_spectrum_evidence_mae = []
    per_spectrum_evidence_correlation = []
    per_spectrum_sign_agreement = []
    pooled_prediction = []
    pooled_target = []
    pooled_weight = []
    for left in range(0, len(positions), batch_size):
        index = positions[left : left + batch_size]
        prediction = forward_positions(model, arrays, index)
        target = torch.from_numpy(correct_target[index])
        valid = torch.from_numpy(arrays["valid"][index])
        intensity = torch.from_numpy(arrays["intensity"][index])
        for local in range(len(index)):
            keep = valid[local]
            weight = torch.sqrt(intensity[local, keep].clamp_min(1e-4))
            channel_error = torch.abs(prediction[local, keep] - target[local, keep]).mean(dim=1)
            pred_evidence = prediction[local, keep, 0] - prediction[local, keep, 1]
            true_evidence = target[local, keep, 0] - target[local, keep, 1]
            evidence_error = torch.abs(pred_evidence - true_evidence)
            per_spectrum_mae.append(float(torch.sum(channel_error * weight) / torch.sum(weight)))
            per_spectrum_evidence_mae.append(
                float(torch.sum(evidence_error * weight) / torch.sum(weight))
            )
            pred_np = pred_evidence.numpy()
            true_np = true_evidence.numpy()
            weight_np = weight.numpy()
            pred_center = pred_np - np.average(pred_np, weights=weight_np)
            true_center = true_np - np.average(true_np, weights=weight_np)
            denominator = np.sqrt(
                np.sum(weight_np * pred_center**2)
                * np.sum(weight_np * true_center**2)
            )
            per_spectrum_evidence_correlation.append(
                float(np.sum(weight_np * pred_center * true_center) / denominator)
                if denominator > 1e-12
                else 0.0
            )
            direction_weight = weight_np * np.abs(true_np)
            per_spectrum_sign_agreement.append(
                float(
                    np.sum(direction_weight * (np.sign(pred_np) == np.sign(true_np)))
                    / np.sum(direction_weight)
                )
                if np.sum(direction_weight) > 1e-12
                else 0.0
            )
            pooled_prediction.extend(pred_evidence.tolist())
            pooled_target.extend(true_evidence.tolist())
            pooled_weight.extend(weight.tolist())
    pred = np.asarray(pooled_prediction)
    target = np.asarray(pooled_target)
    weight = np.asarray(pooled_weight)
    pred_center = pred - np.average(pred, weights=weight)
    target_center = target - np.average(target, weights=weight)
    correlation = float(
        np.sum(weight * pred_center * target_center)
        / np.sqrt(
            np.sum(weight * pred_center**2) * np.sum(weight * target_center**2) + 1e-22
        )
    )
    return {
        "spectra": int(len(positions)),
        "macro_channel_mae": float(np.mean(per_spectrum_mae)),
        "macro_evidence_mae": float(np.mean(per_spectrum_evidence_mae)),
        "macro_evidence_correlation": float(np.mean(per_spectrum_evidence_correlation)),
        "macro_weighted_sign_agreement": float(np.mean(per_spectrum_sign_agreement)),
        "peak_weighted_evidence_correlation": correlation,
        "per_spectrum_channel_mae": np.asarray(per_spectrum_mae, dtype=np.float32),
        "per_spectrum_evidence_mae": np.asarray(per_spectrum_evidence_mae, dtype=np.float32),
        "per_spectrum_evidence_correlation": np.asarray(
            per_spectrum_evidence_correlation, dtype=np.float32
        ),
        "per_spectrum_sign_agreement": np.asarray(
            per_spectrum_sign_agreement, dtype=np.float32
        ),
    }


def train_arm(
    name: str,
    initial_state: dict[str, torch.Tensor],
    arrays: dict,
    train_positions: np.ndarray,
    inner_positions: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, torch.Tensor], dict]:
    set_seed(args.seed)
    model = HierarchicalChemicalResidualAdapter(
        dropout=0.0, formula_dimensions=2, use_formula_moments=False
    )
    model.load_state_dict(copy.deepcopy(initial_state))
    for parameter in model.residual_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed + 311)
    history = []
    target_array = arrays["targets"][name]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_positions)
        losses = []
        for left in range(0, len(order), args.batch_size):
            index = order[left : left + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_positions(model, arrays, index)
            loss = loss_function(
                prediction,
                torch.from_numpy(target_array[index]),
                torch.from_numpy(arrays["valid"][index]),
                torch.from_numpy(arrays["intensity"][index]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            inner = evaluate(model, arrays, inner_positions, args.batch_size)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "inner_macro_channel_mae": inner["macro_channel_mae"],
                    "inner_macro_evidence_mae": inner["macro_evidence_mae"],
                    "inner_evidence_correlation": inner["peak_weighted_evidence_correlation"],
                }
            )
            print(
                f"arm={name} epoch={epoch}/{args.epochs} "
                f"evidence_mae={inner['macro_evidence_mae']:.5f} "
                f"correlation={inner['peak_weighted_evidence_correlation']:.4f}",
                flush=True,
            )
    final = evaluate(model, arrays, inner_positions, args.batch_size)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {
        "arm": name,
        "final": {key: value for key, value in final.items() if not key.startswith("per_spectrum")},
        "history": history,
    }, state, final


def bootstrap_improvement(
    correct_error: np.ndarray,
    control_error: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    difference = control_error - correct_error
    unique = np.unique(formulas)
    values = np.asarray(
        [np.mean(difference[formulas == formula]) for formula in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "formula_macro_mae_improvement": float(np.mean(values)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def bootstrap_advantage(
    correct_value: np.ndarray,
    control_value: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    difference = correct_value - control_value
    unique = np.unique(formulas)
    values = np.asarray(
        [np.mean(difference[formulas == formula]) for formula in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "formula_macro_advantage": float(np.mean(values)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def main() -> None:
    args = parse_args()
    if args.inner_fold == args.outer_fold:
        raise ValueError("inner and outer folds must differ")
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    graph = CandidateGraph(args.graph)
    arrays = load_arrays(args, graph)
    folds = formula_folds(arrays["formulas"], args.folds, args.fold_seed)
    train = np.flatnonzero((folds != args.inner_fold) & (folds != args.outer_fold))
    inner = np.flatnonzero(folds == args.inner_fold)
    outer = np.flatnonzero(folds == args.outer_fold)
    if min(len(train), len(inner), len(outer)) < 1:
        raise RuntimeError("empty formula partition")
    train_formula = set(arrays["formulas"][train])
    inner_formula = set(arrays["formulas"][inner])
    outer_formula = set(arrays["formulas"][outer])
    if train_formula & inner_formula or train_formula & outer_formula or inner_formula & outer_formula:
        raise RuntimeError("formula leakage")

    set_seed(args.seed)
    template = HierarchicalChemicalResidualAdapter(
        dropout=0.0, formula_dimensions=2, use_formula_moments=False
    )
    initial_state = {key: value.detach().cpu() for key, value in template.state_dict().items()}
    untrained = evaluate(template, arrays, inner, args.batch_size)
    summaries = []
    evaluations = {}
    for arm in ("correct", "candidate_swapped", "peak_permuted"):
        if args.evaluate_only_from is None:
            summary, state, final = train_arm(
                arm, initial_state, arrays, train, inner, args
            )
            torch.save(state, args.output / f"{arm}_peak_evidence_branch.pt")
        else:
            checkpoint = args.evaluate_only_from / f"{arm}_peak_evidence_branch.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = HierarchicalChemicalResidualAdapter(
                dropout=0.0, formula_dimensions=2, use_formula_moments=False
            )
            model.load_state_dict(state, strict=True)
            final = evaluate(model, arrays, inner, args.batch_size)
            summary = {
                "arm": arm,
                "fixed_checkpoint": str(checkpoint.resolve()),
                "fixed_checkpoint_sha256": sha256_file(checkpoint),
                "final": {
                    key: value for key, value in final.items()
                    if not key.startswith("per_spectrum")
                },
                "history": [],
            }
        summaries.append(summary)
        evaluations[arm] = final
    formulas = arrays["formulas"][inner]
    comparisons = {}
    for control in ("candidate_swapped", "peak_permuted"):
        comparisons[f"correct_vs_{control}_channel_mae"] = bootstrap_improvement(
            evaluations["correct"]["per_spectrum_channel_mae"],
            evaluations[control]["per_spectrum_channel_mae"],
            formulas,
            args.seed + (17 if control == "candidate_swapped" else 19),
            args.bootstrap_draws,
        )
        comparisons[f"correct_vs_{control}_evidence_mae"] = bootstrap_improvement(
            evaluations["correct"]["per_spectrum_evidence_mae"],
            evaluations[control]["per_spectrum_evidence_mae"],
            formulas,
            args.seed + (23 if control == "candidate_swapped" else 29),
            args.bootstrap_draws,
        )
        comparisons[f"correct_vs_{control}_evidence_correlation"] = bootstrap_advantage(
            evaluations["correct"]["per_spectrum_evidence_correlation"],
            evaluations[control]["per_spectrum_evidence_correlation"],
            formulas,
            args.seed + (31 if control == "candidate_swapped" else 37),
            args.bootstrap_draws,
        )
        comparisons[f"correct_vs_{control}_sign_agreement"] = bootstrap_advantage(
            evaluations["correct"]["per_spectrum_sign_agreement"],
            evaluations[control]["per_spectrum_sign_agreement"],
            formulas,
            args.seed + (41 if control == "candidate_swapped" else 43),
            args.bootstrap_draws,
        )
    zero = np.zeros_like(evaluations["correct"]["per_spectrum_evidence_correlation"])
    comparisons["correct_evidence_correlation_above_zero"] = bootstrap_advantage(
        evaluations["correct"]["per_spectrum_evidence_correlation"],
        zero,
        formulas,
        args.seed + 47,
        args.bootstrap_draws,
    )
    required_keys = [
        "correct_vs_candidate_swapped_channel_mae",
        "correct_vs_peak_permuted_channel_mae",
        "correct_vs_candidate_swapped_evidence_correlation",
        "correct_vs_peak_permuted_evidence_correlation",
        "correct_evidence_correlation_above_zero",
    ]
    passed = bool(all(
        comparisons[key]["formula_cluster_bootstrap_95ci"][0] > 0
        for key in required_keys
    ))
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "Peak-local structure-differential teacher is learnable; residual transfer may be tested."
            if passed
            else "Peak-local teacher did not beat matched controls; do not connect it to the embedding."
        ),
        "scope": {
            "non_formal": True,
            "residual_projection_frozen": True,
            "dreaMS_embedding_changed": False,
            "teacher_structures_training_only": True,
            "outer_fold_evaluated": False,
            "massspecgym_overlap_warning": True,
            "fixed_endpoint_re_evaluation": args.evaluate_only_from is not None,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(
                args.teacher_dir / "iceberg_predictions_f16.npy"
            ),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
        },
        "split": {
            "train_queries": int(len(train)),
            "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(len(outer)),
            "train_formulas": len(train_formula),
            "inner_formulas": len(inner_formula),
            "outer_formulas": len(outer_formula),
            "formula_disjoint": True,
        },
        "target": {
            "channels": ["sqrt true-structure max-normalized predicted intensity", "sqrt max competitor predicted intensity"],
            "locations": "experimental query peaks mapped to ICEBERG 0.1-Da bins",
            "loss": "sqrt-experimental-intensity weighted smooth-L1",
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "matched_initialization": True,
            "matched_steps": True,
            "evaluate_only_from": (
                str(args.evaluate_only_from.resolve()) if args.evaluate_only_from else None
            ),
        },
        "untrained": {
            key: value for key, value in untrained.items() if not key.startswith("per_spectrum")
        },
        "arms": summaries,
        "comparisons": comparisons,
        "gate": {
            "required_metrics": required_keys,
            "rule": (
                "Formula-cluster-bootstrap lower bound must exceed zero for correct-arm "
                "two-channel MAE improvement against both controls, evidence-correlation "
                "advantage against both controls, and correct-arm correlation above zero."
            ),
            "evidence_mae_not_a_gate": (
                "Signed evidence is concentrated near zero, so an uninformative shrink-to-zero "
                "predictor can improve absolute error while losing chemical direction."
            ),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2, ensure_ascii=False), flush=True)
    print(f"decision={report['status']} report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

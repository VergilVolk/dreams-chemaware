"""Matched non-formal pilot for the candidate-independent raw chemical branch.

The official DreaMS embedding is frozen.  Two identically initialized branches
receive either correctly aligned subformula targets or within-spectrum
peak-permuted targets.  Formula folds are inherited from the frozen label cache;
the outer fold is never loaded into training or evaluation metrics.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dreams.models.chem_aware.hierarchical_chemical_adapter import (
    HierarchicalChemicalResidualAdapter,
    deployable_parameter_count,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_aligned(token_dir: Path, label_dir: Path) -> dict[str, np.ndarray]:
    token_rows = np.load(token_dir / "rows.npy").astype(np.int64)
    label_rows = np.load(label_dir / "rows.npy").astype(np.int64)
    positions = {int(row): index for index, row in enumerate(token_rows)}
    if any(int(row) not in positions for row in label_rows):
        raise RuntimeError("label cache contains rows absent from token cache")
    take = np.asarray([positions[int(row)] for row in label_rows], dtype=np.int64)
    return {
        "rows": label_rows,
        "formulas": np.load(label_dir / "formulas.npy").astype(str),
        "row_fold": np.load(label_dir / "row_fold.npy").astype(np.int8),
        "target": np.load(label_dir / "target_f16.npy").astype(np.float32),
        "target_mask": np.load(label_dir / "target_mask.npy").astype(bool),
        "official": np.load(token_dir / "official_embeddings_f32.npy")[take].astype(np.float32),
        "mz": np.load(token_dir / "mz_f32.npy")[take].astype(np.float32),
        "intensity": np.load(token_dir / "intensity_f32.npy")[take].astype(np.float32),
        "valid": np.load(token_dir / "valid.npy")[take].astype(bool),
        "precursor": np.load(token_dir / "precursor_mz_f32.npy")[take].astype(np.float32),
    }


def permute_within_spectrum(
    target: np.ndarray, target_mask: np.ndarray, rows: np.ndarray, seed: int,
) -> np.ndarray:
    result = target.copy()
    for position in rows:
        peaks = np.flatnonzero(target_mask[position])
        if len(peaks) > 1:
            # A row-specific generator makes this invariant to batch order.
            rng = np.random.default_rng(seed + 104_729 * int(position))
            result[position, peaks] = result[position, rng.permutation(peaks)]
    return result


def formula_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    assigned: torch.Tensor,
    intensity: torch.Tensor,
    dimensions: torch.Tensor,
) -> torch.Tensor:
    selected_prediction = prediction[assigned][:, dimensions]
    selected_target = target[assigned][:, dimensions]
    peak_loss = F.smooth_l1_loss(
        selected_prediction, selected_target, reduction="none", beta=0.1,
    ).mean(dim=1)
    weight = torch.sqrt(intensity[assigned].clamp_min(1e-4))
    return (peak_loss * weight).sum() / weight.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(
    model: HierarchicalChemicalResidualAdapter,
    arrays: dict[str, np.ndarray],
    positions: np.ndarray,
    dimensions: np.ndarray,
    batch_size: int,
) -> dict:
    model.eval()
    spectrum_errors: list[float] = []
    spectrum_cosines: list[float] = []
    spectrum_weights: list[float] = []
    spectrum_positions: list[int] = []
    for start in range(0, len(positions), batch_size):
        index = positions[start : start + batch_size]
        official = torch.from_numpy(arrays["official"][index])
        mz = torch.from_numpy(arrays["mz"][index])
        intensity = torch.from_numpy(arrays["intensity"][index])
        precursor = torch.from_numpy(arrays["precursor"][index])
        valid = torch.from_numpy(arrays["valid"][index])
        assigned = torch.from_numpy(arrays["target_mask"][index])
        target = torch.from_numpy(arrays["target"][index])
        prediction = model(official, mz, intensity, precursor, valid).formula_logits
        for local in range(len(index)):
            keep = assigned[local]
            if not torch.any(keep):
                continue
            pred = prediction[local, keep][:, dimensions]
            truth = target[local, keep][:, dimensions]
            weights = torch.sqrt(intensity[local, keep].clamp_min(1e-4))
            error = torch.abs(pred - truth).mean(dim=1)
            cosine = F.cosine_similarity(pred, truth, dim=1, eps=1e-8)
            spectrum_errors.append(float((error * weights).sum() / weights.sum()))
            spectrum_cosines.append(float((cosine * weights).sum() / weights.sum()))
            spectrum_weights.append(float(weights.sum()))
            spectrum_positions.append(int(index[local]))
    weights_np = np.asarray(spectrum_weights, dtype=np.float64)
    return {
        "spectra": len(spectrum_errors),
        "macro_spectrum_mean_absolute_error": float(np.mean(spectrum_errors)),
        "peak_weighted_mean_absolute_error": float(np.average(spectrum_errors, weights=weights_np)),
        "macro_spectrum_mean_cosine": float(np.mean(spectrum_cosines)),
        "peak_weighted_mean_cosine": float(np.average(spectrum_cosines, weights=weights_np)),
        "per_spectrum_mae": spectrum_errors,
        "spectrum_positions": spectrum_positions,
    }


def train_arm(
    name: str,
    initial_state: dict,
    arrays: dict[str, np.ndarray],
    supervision: np.ndarray,
    train_positions: np.ndarray,
    inner_positions: np.ndarray,
    dimensions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[HierarchicalChemicalResidualAdapter, dict]:
    set_seed(seed)
    model = HierarchicalChemicalResidualAdapter(dropout=0.0)
    model.load_state_dict(copy.deepcopy(initial_state))
    # Formula pretraining cannot update or benefit from the residual projection.
    for parameter in model.residual_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    trajectory = []
    generator = np.random.default_rng(seed + 91)
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(train_positions)
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            official = torch.from_numpy(arrays["official"][index])
            mz = torch.from_numpy(arrays["mz"][index])
            intensity = torch.from_numpy(arrays["intensity"][index])
            precursor = torch.from_numpy(arrays["precursor"][index])
            valid = torch.from_numpy(arrays["valid"][index])
            assigned = torch.from_numpy(arrays["target_mask"][index])
            target = torch.from_numpy(supervision[index])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(official, mz, intensity, precursor, valid).formula_logits
            loss = formula_loss(prediction, target, assigned, intensity, torch.from_numpy(dimensions))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        inner = evaluate(model, arrays, inner_positions, dimensions, batch_size)
        trajectory.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "inner_macro_spectrum_mae": inner["macro_spectrum_mean_absolute_error"],
            "inner_macro_spectrum_cosine": inner["macro_spectrum_mean_cosine"],
        })
    final = evaluate(model, arrays, inner_positions, dimensions, batch_size)
    final.pop("per_spectrum_mae")
    final.pop("spectrum_positions")
    return model, {"name": name, "final": final, "trajectory": trajectory}


def cluster_bootstrap_difference(
    correct: np.ndarray,
    control: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int = 10_000,
) -> dict:
    if correct.shape != control.shape or correct.shape != formulas.shape:
        raise RuntimeError("paired metrics are not aligned")
    rng = np.random.default_rng(seed)
    difference = control - correct
    unique = np.unique(formulas)
    cluster_means = np.asarray([
        np.mean(difference[formulas == formula]) for formula in unique
    ], dtype=np.float64)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(cluster_means), len(cluster_means))
        estimates[draw] = np.mean(cluster_means[sampled])
    return {
        "formula_macro_mae_improvement": float(np.mean(cluster_means)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--permutation-seed", type=int, default=9062)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    set_seed(args.seed)
    arrays = load_aligned(args.token_dir, args.label_dir)
    train_positions = np.flatnonzero(
        (arrays["row_fold"] != args.outer_fold) & (arrays["row_fold"] != args.inner_fold)
    )
    inner_positions = np.flatnonzero(arrays["row_fold"] == args.inner_fold)
    outer_positions = np.flatnonzero(arrays["row_fold"] == args.outer_fold)
    training_values = arrays["target"][arrays["target_mask"] & np.isin(
        np.arange(len(arrays["rows"]))[:, None], train_positions
    )]
    dimensions = np.flatnonzero(np.ptp(training_values, axis=0) > 0)
    if len(dimensions) < 2:
        raise RuntimeError("too few variable formula dimensions")

    base = HierarchicalChemicalResidualAdapter(dropout=0.0)
    initial_state = copy.deepcopy(base.state_dict())
    untrained = evaluate(base, arrays, inner_positions, dimensions, args.batch_size)
    untrained_errors = np.asarray(untrained.pop("per_spectrum_mae"), dtype=np.float64)
    inner_metric_positions = np.asarray(untrained.pop("spectrum_positions"), dtype=np.int64)
    inner_formulas = arrays["formulas"][inner_metric_positions]
    permuted_target = permute_within_spectrum(
        arrays["target"], arrays["target_mask"], train_positions, args.permutation_seed,
    )
    correct_model, correct_report = train_arm(
        "correct_subformula", initial_state, arrays, arrays["target"], train_positions,
        inner_positions, dimensions, args.epochs, args.batch_size, args.learning_rate,
        args.weight_decay, args.seed,
    )
    permuted_model, permuted_report = train_arm(
        "within_spectrum_peak_permuted", initial_state, arrays, permuted_target,
        train_positions, inner_positions, dimensions, args.epochs, args.batch_size,
        args.learning_rate, args.weight_decay, args.seed,
    )
    correct_eval = evaluate(correct_model, arrays, inner_positions, dimensions, args.batch_size)
    permuted_eval = evaluate(permuted_model, arrays, inner_positions, dimensions, args.batch_size)
    correct_errors = np.asarray(correct_eval.pop("per_spectrum_mae"), dtype=np.float64)
    correct_positions = np.asarray(correct_eval.pop("spectrum_positions"), dtype=np.int64)
    permuted_errors = np.asarray(permuted_eval.pop("per_spectrum_mae"), dtype=np.float64)
    permuted_positions = np.asarray(permuted_eval.pop("spectrum_positions"), dtype=np.int64)
    if not (
        np.array_equal(inner_metric_positions, correct_positions)
        and np.array_equal(correct_positions, permuted_positions)
    ):
        raise RuntimeError("evaluation arms are not spectrum-aligned")
    comparison = cluster_bootstrap_difference(
        correct_errors, permuted_errors, inner_formulas, args.seed + 313,
    )
    comparison_untrained = cluster_bootstrap_difference(
        correct_errors, untrained_errors, inner_formulas, args.seed + 314,
    )
    gate = bool(
        comparison["formula_macro_mae_improvement"] > 0.005
        and comparison["formula_cluster_bootstrap_95ci"][0] > 0
    )

    args.output.mkdir(parents=True)
    torch.save(initial_state, args.output / "initial_formula_branch.pt")
    torch.save(correct_model.state_dict(), args.output / "correct_formula_branch.pt")
    torch.save(permuted_model.state_dict(), args.output / "peak_permuted_formula_branch.pt")
    report = {
        "status": "nonformal_raw_formula_branch_pilot",
        "formal_training_authorized": False,
        "official_dreams_parameters_updated": False,
        "outer_fold_loaded_for_metrics": False,
        "selection": {
            "train_spectra": int(len(train_positions)),
            "inner_spectra": int(len(inner_positions)),
            "outer_spectra_untouched": int(len(outer_positions)),
            "variable_target_dimensions": dimensions.tolist(),
            "formula_fold_overlap": 0,
        },
        "architecture": {
            "deployable_parameters": deployable_parameter_count(base),
            "total_parameters_with_training_head": sum(p.numel() for p in base.parameters()),
            "candidate_inputs_used": False,
            "official_embedding_exact_at_initialization": True,
            "initial_state_sha256": state_sha256(initial_state),
        },
        "optimization": {
            "epochs_fixed_in_advance": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "shared_initialization_and_batch_order": True,
            "checkpoint_selection": "none_final_epoch_only",
        },
        "untrained": untrained,
        "correct": {**correct_report, "final_recomputed": correct_eval},
        "peak_permuted": {**permuted_report, "final_recomputed": permuted_eval},
        "correct_vs_peak_permuted": comparison,
        "correct_vs_untrained": comparison_untrained,
        "pass_to_retrieval_residual_pilot": gate,
        "gate": (
            "correct formula-macro MAE improvement >0.005 and formula-cluster bootstrap "
            "95% lower bound >0"
        ),
        "claim_limit": (
            "A pass only shows that the deployable raw-spectrum branch learns peak-local "
            "formula information. It does not establish retrieval or isomer-ranking gain."
        ),
        "provenance": {
            "label_report_sha256": sha256_file(args.label_dir / "report.json"),
            "adapter_source_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "dreams/models/chem_aware/hierarchical_chemical_adapter.py"
            ),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "untrained": report["untrained"],
        "correct": correct_eval,
        "peak_permuted": permuted_eval,
        "correct_vs_peak_permuted": comparison,
        "pass_to_retrieval_residual_pilot": gate,
    }, indent=2))


if __name__ == "__main__":
    main()

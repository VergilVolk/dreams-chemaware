"""Audit whether chemical-rule teachers create identifiable encoder gradients.

This is a small mechanism check, not a retrieval result.  It uses one official
error and one nearest-boundary official-correct training query, runs the real
official DreaMS model, and compares gradients induced by the correct rule-mass
teacher with a mass-shifted rule control on the exact same candidate batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_mass_kernel_embedding import KernelCache  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    listwise_losses, official_outcomes, sample_training_batch,
)
from train_chemaware_full_candidate_direct import kernel_teacher_loss  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, forward_embeddings,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--teacher-beta", type=float, default=0.2)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def gradient_summary(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> dict:
    left_sq = right_sq = diff_sq = dot = 0.0
    nonzero_left = nonzero_right = nonzero_diff = 0
    tensors = 0
    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        tensors += 1
        af = a.detach().float(); bf = b.detach().float(); df = af - bf
        av = float(torch.sum(af * af)); bv = float(torch.sum(bf * bf))
        dv = float(torch.sum(df * df)); cv = float(torch.sum(af * bf))
        left_sq += av; right_sq += bv; diff_sq += dv; dot += cv
        nonzero_left += int(av > 0); nonzero_right += int(bv > 0)
        nonzero_diff += int(dv > 0)
    left_norm = left_sq ** 0.5; right_norm = right_sq ** 0.5
    return {
        "parameter_tensors": tensors,
        "correct_nonzero_tensors": nonzero_left,
        "shifted_nonzero_tensors": nonzero_right,
        "differential_nonzero_tensors": nonzero_diff,
        "correct_gradient_norm": left_norm,
        "shifted_gradient_norm": right_norm,
        "differential_gradient_norm": diff_sq ** 0.5,
        "correct_shifted_cosine": dot / max(left_norm * right_norm, 1e-30),
    }


def gradient_relation(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> dict:
    summary = gradient_summary(left, right)
    return {
        "left_norm": summary["correct_gradient_norm"],
        "right_norm": summary["shifted_gradient_norm"],
        "difference_norm": summary["differential_gradient_norm"],
        "cosine": summary["correct_shifted_cosine"],
        "right_to_left_norm_ratio": (
            summary["shifted_gradient_norm"]
            / max(summary["correct_gradient_norm"], 1e-30)
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}

    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    molecule_fold = stable_formula_folds(body["molecule_formula"], args.folds, args.fold_seed)
    molecule_allowed = (molecule_fold != args.inner_fold) & (molecule_fold != args.outer_fold)
    allowed_count = np.add.reduceat(molecule_allowed.astype(np.int32), body["query_ptr"][:-1])
    train = np.flatnonzero(
        (fold != args.inner_fold) & (fold != args.outer_fold) & (allowed_count >= 2)
    )
    error, margin = official_outcomes(body, train, official, row_position, molecule_allowed)
    error_query = int(train[error[train]][np.argmax(margin[train[error[train]]])])
    correct_query = int(train[~error[train]][np.argmin(margin[train[~error[train]]])])
    queries = np.asarray([error_query, correct_query], dtype=np.int64)
    batch = sample_training_batch(
        body, queries, row_position, None, 1,
        np.random.default_rng(args.seed), molecule_allowed,
    )
    query_rows = rows[batch["query_cache"]]
    reference_rows = rows[batch["reference_cache"]]
    store = SpectrumStore(
        args.data, np.unique(np.r_[query_rows, reference_rows]), args.n_highest_peaks,
    )
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    capacity = unfreeze_last_blocks(model, 1)
    model.eval()
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    spectra = torch.cat((store.get(query_rows), store.get(reference_rows))).to(device)
    encoded = forward_embeddings(model, spectra, False)
    query_z = encoded[:len(queries)]
    reference_z = encoded[len(queries):]
    official_query = torch.from_numpy(
        np.array(official[batch["query_cache"]], copy=True),
    ).to(device)
    official_reference = torch.from_numpy(
        np.array(official[batch["reference_cache"]], copy=True),
    ).to(device)
    cache = KernelCache(
        args, row_position,
        variants=("rule_mass", "rule_mass_shifted"),
    )
    weight = torch.ones(len(queries), device=device)
    loss_correct = kernel_teacher_loss(
        query_z, reference_z, official_query, official_reference,
        query_rows, reference_rows, batch["candidate_ptr"], batch["reference_ptr"],
        batch["reference_edge"], cache, "rule_mass", args.teacher_beta,
        args.teacher_temperature, weight,
    )
    loss_rank_equivalent = kernel_teacher_loss(
        query_z, reference_z, official_query, official_reference,
        query_rows, reference_rows, batch["candidate_ptr"], batch["reference_ptr"],
        batch["reference_edge"], cache, "rule_mass", args.teacher_beta,
        args.teacher_temperature, weight, "rank_equivalent_kl",
    )
    loss_margin_transfer = kernel_teacher_loss(
        query_z, reference_z, official_query, official_reference,
        query_rows, reference_rows, batch["candidate_ptr"], batch["reference_ptr"],
        batch["reference_edge"], cache, "rule_mass", args.teacher_beta,
        args.teacher_temperature, weight, "positive_margin_transfer", 0.5, 0.05, 0.01,
    )
    spectrum_loss, _, inbatch_spectrum, _, margin_floor = listwise_losses(
        query_z, reference_z, None, batch["candidate_ptr"], batch["reference_ptr"],
        batch["reference_edge"], 0.10, official_query, official_reference,
        0.005, weight,
    )
    preserve = torch.cat((
        1 - torch.sum(query_z * official_query, dim=1),
        1 - torch.sum(reference_z * official_reference, dim=1),
    )).mean()
    base_loss = spectrum_loss + 0.25 * inbatch_spectrum + 2.0 * margin_floor + 5.0 * preserve
    loss_shifted = kernel_teacher_loss(
        query_z, reference_z, official_query, official_reference,
        query_rows, reference_rows, batch["candidate_ptr"], batch["reference_ptr"],
        batch["reference_edge"], cache, "rule_mass_shifted", args.teacher_beta,
        args.teacher_temperature, weight,
    )
    grad_base = torch.autograd.grad(
        base_loss, trainable, retain_graph=True, allow_unused=True,
    )
    grad_correct = torch.autograd.grad(loss_correct, trainable, retain_graph=True, allow_unused=True)
    grad_rank_equivalent = torch.autograd.grad(
        loss_rank_equivalent, trainable, retain_graph=True, allow_unused=True,
    )
    grad_margin_transfer = torch.autograd.grad(
        loss_margin_transfer, trainable, retain_graph=True, allow_unused=True,
    )
    grad_shifted = torch.autograd.grad(
        loss_shifted, trainable, allow_unused=True,
    )
    gradients = gradient_summary(grad_correct, grad_shifted)
    gates = {
        "correct_teacher_gradient_nonzero": gradients["correct_gradient_norm"] > 0,
        "shifted_teacher_gradient_nonzero": gradients["shifted_gradient_norm"] > 0,
        "teacher_differential_gradient_nonzero": gradients["differential_gradient_norm"] > 0,
        "teacher_gradients_not_identical": gradients["correct_shifted_cosine"] < 0.999999,
    }
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "two-query real-model mechanism audit; not a retrieval result",
        "queries": {
            "official_error": error_query,
            "official_error_margin": float(margin[error_query]),
            "official_boundary_correct": correct_query,
            "official_boundary_correct_margin": float(margin[correct_query]),
        },
        "loss": {
            "base": float(base_loss.detach()),
            "rule_mass": float(loss_correct.detach()),
            "rule_mass_rank_equivalent": float(loss_rank_equivalent.detach()),
            "rule_mass_positive_margin_transfer": float(loss_margin_transfer.detach()),
            "rule_mass_shifted": float(loss_shifted.detach()),
            "difference": float((loss_correct - loss_shifted).detach()),
        },
        "gradients": gradients,
        "transfer_diagnostics": {
            "base_vs_normalized_kl": gradient_relation(grad_base, grad_correct),
            "base_vs_rank_equivalent_kl": gradient_relation(grad_base, grad_rank_equivalent),
            "base_vs_positive_margin_transfer": gradient_relation(grad_base, grad_margin_transfer),
            "normalized_vs_rank_equivalent_kl": gradient_relation(
                grad_correct, grad_rank_equivalent,
            ),
        },
        "capacity": capacity,
        "initialization": str(initialization),
        "gates": gates,
        "contracts": {
            "same_candidate_batch": True,
            "same_official_initialization": True,
            "same_trainable_parameters": True,
            "only_rule_mass_values_shifted_in_control": True,
            "outer_fold_evaluated": False,
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Noise V5 minimal residual round (pre-registered, GNPS-certified design).

Trains exactly one unit per V1-current residual query selected by the atlas
(``selected_relations.npz``): the anchor is the targeted action view (or the
matched control view in the control arm), the positive is the event's exact
positive row, and the negatives are the event's exact negative row plus the
query's current molecule-max top-rival winner row.  There is deliberately no
clean stream: the shared-clean drift that killed v2/v3 cannot occur.  One
Adam step per unit, fresh native Adam at the frozen 5e-6, native 0.1 margin,
hinge averaged over negatives.  Both arms share the seed and the unit order,
so per-unit gradients are directly comparable between arms.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from noise_final_core import sha256_file
from train_noise_dreams_native import make_hdf5_spectrum
from train_noise_dreams_native_residual_stage2 import construct_native_model

SELECTED_REQUIRED = (
    "query_index", "action_index", "query_row",
    "exact_positive_row", "exact_negative_row",
    "clean_positive_winner_row", "clean_top_rival_winner_row",
)
BANK_REQUIRED = (
    "query_index", "source", "family",
    "targeted_action_spectra", "control_action_spectra",
    "native_action_view_representable",
)


def strict_selected_schema(arrays: dict[str, np.ndarray]) -> int:
    """Validate the atlas selected-relations ledger without touching torch."""
    missing = [name for name in SELECTED_REQUIRED if name not in arrays]
    if missing:
        raise RuntimeError(f"selected relations miss arrays: {missing}")
    count = len(arrays["query_index"])
    if count == 0:
        raise RuntimeError("selected relations ledger is empty")
    for name in SELECTED_REQUIRED:
        array = arrays[name]
        if array.ndim != 1 or array.dtype != np.int64 or len(array) != count:
            raise RuntimeError(f"selected relations array {name} is malformed")
    return count


def dedupe_negative_rows(
    exact_negative_row: int, rival_row: int, positive_row: int,
) -> list[int]:
    """Collect the unit's negative rows, dropping duplicates and the positive."""
    rows: list[int] = []
    for row in (int(exact_negative_row), int(rival_row)):
        if row == int(positive_row):
            continue
        if row not in rows:
            rows.append(row)
    if not rows:
        raise RuntimeError("V5 unit lost every negative row")
    return rows


def event_order(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(count)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-relations", type=Path, required=True)
    parser.add_argument("--action-bank", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("targeted", "control"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    frozen = {
        "seed": 3407, "lr": 5e-6, "weight_decay": 0.0,
        "triplet_loss_margin": 0.1, "n_highest_peaks": 100,
    }
    observed = {key: getattr(args, key) for key in frozen}
    if observed != frozen:
        raise RuntimeError(f"V5 minimal settings are frozen: {observed} != {frozen}")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("V5 minimal training requires an allocated GPU")

    with np.load(args.selected_relations, allow_pickle=False) as body:
        selected = {name: np.asarray(body[name]) for name in body.files}
    unit_count = strict_selected_schema(selected)
    with np.load(args.action_bank, allow_pickle=False) as body:
        bank = {name: np.asarray(body[name]) for name in body.files}
    missing_bank = [name for name in BANK_REQUIRED if name not in bank]
    if missing_bank:
        raise RuntimeError(f"action bank misses arrays: {missing_bank}")
    bank_count = len(bank["query_index"])
    targeted_spectra = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_spectra = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    representable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if (
        targeted_spectra.shape != (bank_count, 101, 2)
        or control_spectra.shape != targeted_spectra.shape
    ):
        raise RuntimeError("V5 action bank spectra are not (N, 101, 2) views")
    action_index = np.asarray(selected["action_index"], dtype=np.int64)
    query_index = np.asarray(selected["query_index"], dtype=np.int64)
    if (
        np.any((action_index < 0) | (action_index >= bank_count))
        or not np.all(representable[action_index])
        or not np.array_equal(
            np.asarray(bank["query_index"], dtype=np.int64)[action_index], query_index,
        )
    ):
        raise RuntimeError("a selected relation does not attach to its action view")
    positive_rows = np.asarray(selected["exact_positive_row"], dtype=np.int64)
    negative_rows = np.asarray(selected["exact_negative_row"], dtype=np.int64)
    rival_rows = np.asarray(selected["clean_top_rival_winner_row"], dtype=np.int64)
    if (
        np.any(positive_rows == negative_rows)
        or np.any(positive_rows == rival_rows)
        or np.any((positive_rows < 0) | (negative_rows < 0) | (rival_rows < 0))
    ):
        raise RuntimeError("a selected relation reuses its positive row as a negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("V5 did not reconstruct the native DreaMS ContrastiveHead")
    device = torch.device("cuda")
    model.backbone.unfreeze()
    model.to(device).train()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V5 failed to unfreeze the complete shared encoder")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise RuntimeError("V5 optimizer is not the single native Adam group")
    group = optimizer.param_groups[0]
    if (
        abs(float(group["lr"]) - args.lr) > 1e-15
        or float(group["weight_decay"]) != args.weight_decay
        or {id(p) for p in group["params"]} != {id(p) for p in model.parameters()}
    ):
        raise RuntimeError("V5 native Adam does not own the exact model")
    if optimizer.state:
        raise RuntimeError("V5 must start from a fresh Adam state")

    initial_backbone = {
        key: value.detach().to("cpu", copy=True)
        for key, value in model.backbone.state_dict().items()
    }
    initial_head = {
        key: value.detach().to("cpu", copy=True)
        for key, value in model.head.state_dict().items()
    }
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    anchor_bank = targeted_spectra if args.arm == "targeted" else control_spectra
    order = event_order(args.seed, unit_count)

    stats = {
        "units": 0, "active_units": 0, "hinge_sum": 0.0, "hinge_max": 0.0,
        "negative_row_sum": 0, "grad_norm_sum": 0.0,
    }
    first_grad_norm: float | None = None
    sources: Counter[str] = Counter()
    families: Counter[str] = Counter()
    started = time.time()
    with h5py.File(args.data, "r") as handle:
        spectrum_dataset = handle["spectrum"]
        row_limit = len(spectrum_dataset)
        if np.any(positive_rows >= row_limit) or np.any(negative_rows >= row_limit) \
                or np.any(rival_rows >= row_limit):
            raise RuntimeError("a selected relation references a row beyond the hdf5")
        for position in order:
            unit = int(position)
            action = int(action_index[unit])
            positive_row = int(positive_rows[unit])
            unit_negatives = dedupe_negative_rows(
                negative_rows[unit], rival_rows[unit], positive_row,
            )
            sources[str(bank["source"][action])] += 1
            families[str(bank["family"][action])] += 1
            spectra = [torch.from_numpy(anchor_bank[action])]
            for row in (positive_row, *unit_negatives):
                spectrum = make_hdf5_spectrum(
                    spectrum_dataset[row], float(handle["precursor_mz"][row]),
                )
                spectra.append(torch.from_numpy(preprocessor(
                    spectrum.get_peak_list(),
                    prec_mz=spectrum.get_precursor_mz(),
                    high_form=False, augment=False,
                )))
            tensor = torch.from_numpy(
                np.stack([np.asarray(item, dtype=np.float32) for item in spectra]),
            ).to(device)
            encoded = F.normalize(model(tensor).float(), dim=1)
            anchor, positive = encoded[0], encoded[1]
            negatives = encoded[2:]
            hinge = torch.relu(
                args.triplet_loss_margin + (anchor @ negatives) - (anchor @ positive),
            ).mean()
            if not torch.isfinite(hinge):
                raise RuntimeError("non-finite V5 hinge")
            optimizer.zero_grad(set_to_none=True)
            hinge.backward()
            grad_norm = float(torch.sqrt(torch.stack([
                torch.sum(parameter.grad.detach() ** 2)
                for parameter in optimizer.param_groups[0]["params"]
                if parameter.grad is not None
            ]).sum()))
            if not np.isfinite(grad_norm):
                raise RuntimeError("non-finite V5 gradient norm")
            if first_grad_norm is None:
                first_grad_norm = grad_norm
            optimizer.step()
            stats["units"] += 1
            stats["active_units"] += int(float(hinge.detach()) > 0.0)
            stats["hinge_sum"] += float(hinge.detach())
            stats["hinge_max"] = max(stats["hinge_max"], float(hinge.detach()))
            stats["negative_row_sum"] += len(unit_negatives)
            stats["grad_norm_sum"] += grad_norm
            if stats["units"] % 200 == 0 or stats["units"] == unit_count:
                print(
                    f"[V5 arm={args.arm}] {stats['units']:,}/{unit_count:,} units "
                    f"elapsed={time.time()-started:.0f}s", flush=True,
                )

    final_steps = {
        index: int(state["step"])
        for index, state in optimizer.state.items()
    }
    if len(final_steps) != len(list(model.parameters())) or set(
        final_steps.values()
    ) != {unit_count}:
        raise RuntimeError("V5 Adam did not advance exactly once per unit")
    drift = 0.0
    with torch.no_grad():
        for key, value in model.backbone.state_dict().items():
            drift += float((value - initial_backbone[key].to(device)).abs().sum())
        for key, value in model.head.state_dict().items():
            drift += float((value - initial_head[key].to(device)).abs().sum())
    if drift <= 0.0:
        raise RuntimeError("V5 training did not move any parameter")

    args.output.mkdir(parents=True)
    slim = {
        "format": "official_embedding_slim_v1",
        "source_checkpoint": str(args.warm_start_checkpoint),
        "source_size_bytes": args.warm_start_checkpoint.stat().st_size,
        "backbone_state_dict": {
            key: value.detach().cpu()
            for key, value in model.backbone.state_dict().items()
        },
        "head_state_dict": {
            key: value.detach().cpu()
            for key, value in model.head.state_dict().items()
        },
    }
    checkpoint = args.output / "final_slim.pt"
    torch.save(slim, checkpoint)
    optimizer_checkpoint = args.output / "final_optimizer.pt"
    torch.save({
        "format": "noise_v5_minimal_adam_v1",
        "optimizer_state_dict": optimizer.state_dict(),
        "units": unit_count,
        "warm_start_checkpoint": str(args.warm_start_checkpoint),
    }, optimizer_checkpoint)
    report = {
        "status": "noise_v5_minimal_training_complete",
        "arm": args.arm,
        "seed": args.seed,
        "initialization": "V1 champion (T1/T3 v1 primary seed 3407)",
        "initialization_kind": initialization_kind,
        "training": {
            "units": stats["units"],
            "active_fraction": stats["active_units"] / stats["units"],
            "mean_hinge": stats["hinge_sum"] / stats["units"],
            "max_hinge": stats["hinge_max"],
            "mean_negative_rows_per_unit": stats["negative_row_sum"] / stats["units"],
            "first_unit_grad_norm_l2": first_grad_norm,
            "mean_unit_grad_norm_l2": stats["grad_norm_sum"] / stats["units"],
            "total_abs_parameter_drift": drift,
            "optimizer": (
                "fresh native Adam (the v1 trainer did not retain its final "
                "moments; one step per unit, no schedule)"
            ),
            "clean_stream": "none",
            "hinge_aggregation": "mean over negatives",
        },
        "trained_sources": dict(sorted(sources.items())),
        "trained_families": dict(sorted(families.items())),
        "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
        "selected_relations_sha256": sha256_file(args.selected_relations),
        "atlas_report_sha256": sha256_file(
            args.selected_relations.parent / "report.json",
        ),
        "action_bank_sha256": sha256_file(args.action_bank),
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "final_optimizer_sha256": sha256_file(optimizer_checkpoint),
        "outer_performance_claimed": False,
        "claim_limit": (
            "Train-side residual round only; promotion is decided externally by "
            "the pre-registered GNPS paired gates, never by this report."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

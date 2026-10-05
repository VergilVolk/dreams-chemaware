#!/usr/bin/env python
"""One-pass Stage-1 continuation with the relation-complete T1/T3 objective."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from e1_checkpoint_io import torch_load_compat
from noise_final_core import sha256_file
from noise_relation_t1_t3_core import batch_relation_complete_loss
from train_noise_dreams_native import make_hdf5_spectrum
from train_noise_dreams_native_residual_stage2 import adam_state_steps, construct_native_model


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {name: np.asarray(body[name]) for name in body.files}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--action-bank", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--listwise-temperature", type=float, default=0.07)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--maximum-spectra-per-step", type=int, default=64)
    parser.add_argument("--maximum-queries-per-step", type=int, default=4)
    return parser.parse_args()


def query_span(corpus: dict[str, np.ndarray], position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0, a1 = map(int, corpus["action_ptr"][position:position + 2])
    m0, m1 = map(int, corpus["molecule_ptr"][position:position + 2])
    r0 = int(corpus["reference_ptr"][m0])
    r1 = int(corpus["reference_ptr"][m1])
    return (
        np.asarray(corpus["action_index"][a0:a1], dtype=np.int64),
        np.asarray(corpus["reference_row"][r0:r1], dtype=np.int64),
        np.asarray(corpus["reference_ptr"][m0:m1 + 1] - r0, dtype=np.int64),
    )


def packed_batches(
    corpus: dict[str, np.ndarray], seed: int,
    maximum_spectra: int, maximum_queries: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus["query_index"]))
    batches: list[list[int]] = []
    current: list[int] = []
    current_size = 0
    for value in order:
        position = int(value)
        actions, references, _ = query_span(corpus, position)
        size = 1 + len(actions) + len(references)
        if size > maximum_spectra:
            raise RuntimeError(f"T1/T3 query {position} exceeds per-step spectrum budget: {size}")
        if current and (
            len(current) >= maximum_queries or current_size + size > maximum_spectra
        ):
            batches.append(current)
            current, current_size = [], 0
        current.append(position)
        current_size += size
    if current:
        batches.append(current)
    flattened = [value for batch in batches for value in batch]
    if sorted(flattened) != list(range(len(corpus["query_index"]))):
        raise RuntimeError("T1/T3 packed schedule lost or duplicated a query")
    return batches


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("T1/T3 training requires an allocated GPU")
    for path in (
        args.data, args.corpus / "train.npz", args.corpus / "report.json",
        args.action_bank, args.warm_start_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.lr != 5e-6 or args.weight_decay != 0.0
        or args.triplet_loss_margin != 0.1 or args.listwise_temperature != 0.07
        or args.n_highest_peaks != 100
    ):
        raise RuntimeError("T1/T3 changes relation content/objective, not native optimizer settings")
    corpus_report = json.loads((args.corpus / "report.json").read_text(encoding="utf-8"))
    if corpus_report.get("status") != "noise_relation_t1_t3_corpus_complete":
        raise RuntimeError("T1/T3 corpus is incomplete")
    corpus = load_npz(args.corpus / "train.npz")
    bank = load_npz(args.action_bank)
    actions = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    representable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if len(corpus["action_index"]) and not np.all(representable[corpus["action_index"]]):
        raise RuntimeError("unrepresentable Stage-1 action entered T1/T3")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("T1/T3 did not reconstruct the native DreaMS ContrastiveHead")
    device = torch.device("cuda")
    # Native ContrastiveHead unfreezes the backbone at epoch 0 through its
    # Lightning hook.  This direct one-pass loop must make that same action
    # explicitly before constructing the optimizer.
    model.backbone.unfreeze()
    model.to(device).train()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("T1/T3 failed to unfreeze the complete shared encoder")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam:
        raise RuntimeError("T1/T3 optimizer is not native Adam")
    warm = torch_load_compat(args.warm_start_checkpoint, map_location="cpu")
    states = warm.get("optimizer_states")
    if not isinstance(states, list) or len(states) != 1:
        raise RuntimeError("Stage-1 warm start lacks its native Adam state")
    optimizer.load_state_dict(states[0])
    # load_state_dict restores the checkpoint's param_groups, including its
    # final learning rate.  A Stage-1 scheduler residue would otherwise make
    # this continuation silently train at a non-native step size.
    if (
        len(optimizer.param_groups) != 1
        or abs(float(optimizer.param_groups[0]["lr"]) - args.lr) > 1e-15
        or float(optimizer.param_groups[0]["weight_decay"]) != args.weight_decay
    ):
        raise RuntimeError("restored Stage-1 Adam hyperparameters drifted")
    initial_steps = adam_state_steps(optimizer.state_dict())
    del states, warm
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    batches = packed_batches(
        corpus, args.seed, args.maximum_spectra_per_step, args.maximum_queries_per_step,
    )
    component_sum: dict[str, float] = {}
    maximum_observed_spectra = 0
    started = time.time()
    with h5py.File(args.data, "r") as handle:
        for step, positions in enumerate(batches, start=1):
            payload = [query_span(corpus, position) for position in positions]
            hdf5_rows = sorted(set(
                [int(corpus["query_row"][position]) for position in positions]
                + [int(row) for _, references, _ in payload for row in references]
            ))
            action_indices = sorted(set(
                int(action) for action_set, _, _ in payload for action in action_set
            ))
            spectra = []
            for row in hdf5_rows:
                spectrum = make_hdf5_spectrum(
                    handle["spectrum"][row], float(handle["precursor_mz"][row]),
                )
                spectra.append(preprocessor(
                    spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
                    high_form=False, augment=False,
                ))
            spectra.extend(actions[index] for index in action_indices)
            maximum_observed_spectra = max(maximum_observed_spectra, len(spectra))
            if len(spectra) > args.maximum_spectra_per_step:
                raise RuntimeError("deduplicated T1/T3 batch exceeded spectrum budget")
            tensor = torch.from_numpy(np.stack(spectra).astype(np.float32)).to(device)
            encoded = F.normalize(model(tensor).float(), dim=1)
            row_position = {row: index for index, row in enumerate(hdf5_rows)}
            action_position = {
                action: len(hdf5_rows) + index for index, action in enumerate(action_indices)
            }
            query_losses = []
            for position, (action_set, references, pointer) in zip(positions, payload, strict=True):
                anchor_positions = [row_position[int(corpus["query_row"][position])]]
                anchor_positions.extend(action_position[int(action)] for action in action_set)
                reference_positions = [row_position[int(row)] for row in references]
                if len(action_set):
                    anchor_weights = torch.full(
                        (1 + len(action_set),), 0.5 / len(action_set),
                        device=device, dtype=torch.float32,
                    )
                    anchor_weights[0] = 0.5
                else:
                    anchor_weights = torch.ones(1, device=device, dtype=torch.float32)
                query_losses.append((
                    encoded[torch.as_tensor(anchor_positions, device=device)],
                    encoded[torch.as_tensor(reference_positions, device=device)],
                    torch.as_tensor(pointer, device=device, dtype=torch.long),
                    anchor_weights,
                ))
            loss, detail = batch_relation_complete_loss(
                query_losses,
                triplet_margin=args.triplet_loss_margin,
                listwise_temperature=args.listwise_temperature,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite T1/T3 objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Weight per-batch means by their query counts so the reported
            # averages are over queries, matching the training dose.
            component_sum["loss"] = (
                component_sum.get("loss", 0.0) + float(loss.detach()) * len(positions)
            )
            for name, value in detail.items():
                component_sum[name] = (
                    component_sum.get(name, 0.0)
                    + float(value.detach()) * len(positions)
                )
            if step % 250 == 0 or step == len(batches):
                print(
                    f"[T1/T3 seed={args.seed}] {step:,}/{len(batches):,} steps "
                    f"loss={float(loss.detach()):.5f} elapsed={time.time()-started:.0f}s",
                    flush=True,
                )

    final_steps = adam_state_steps(optimizer.state_dict())
    if set(final_steps) != set(initial_steps):
        raise RuntimeError("T1/T3 Adam parameter registry drifted")
    deltas = {key: final_steps[key] - initial_steps[key] for key in final_steps}
    if set(deltas.values()) != {len(batches)}:
        raise RuntimeError("T1/T3 Adam did not advance exactly once per packed batch")
    args.output.mkdir(parents=True)
    slim = {
        "format": "official_embedding_slim_v1",
        "source_checkpoint": str(args.warm_start_checkpoint),
        "source_size_bytes": args.warm_start_checkpoint.stat().st_size,
        "backbone_state_dict": {key: value.detach().cpu() for key, value in model.backbone.state_dict().items()},
        "head_state_dict": {key: value.detach().cpu() for key, value in model.head.state_dict().items()},
    }
    checkpoint = args.output / "final_slim.pt"
    torch.save(slim, checkpoint)
    report = {
        "status": "noise_relation_t1_t3_training_complete",
        "seed": args.seed,
        "initialization": "Stage-1 targeted champion",
        "initialization_kind": initialization_kind,
        "shared_encoder_updated": True,
        "training": {
            "optimizer": "native Adam with restored Stage-1 moments",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "precision": "FP32",
            "epochs": 1,
            "query_dose": (
                "each eligible query exactly once; clean anchor has 0.5 weight and "
                "all available action anchors share 0.5, or clean has 1.0 when absent"
            ),
            "queries": int(len(corpus["query_index"])),
            "optimizer_steps": len(batches),
            "maximum_spectra_per_step": args.maximum_spectra_per_step,
            "maximum_observed_unique_spectra": maximum_observed_spectra,
            "objective": "0.5*T1 multi-relation native hinge + 0.5*T3 exact molecule-max listwise",
            "triplet_margin": args.triplet_loss_margin,
            "listwise_temperature": args.listwise_temperature,
        },
        "mean_components": {
            name: value / int(len(corpus["query_index"]))
            for name, value in component_sum.items()
        },
        "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
        "corpus_report_sha256": sha256_file(args.corpus / "report.json"),
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "outer_performance_claimed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

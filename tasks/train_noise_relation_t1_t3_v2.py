"""v2 relation trainer: clean T1/T3 stream plus one-action-per-step rotation.

Arms (for the train-side comparison, never model selection on held):

* ``clean-only``        -- the v2 clean relation stream alone.
* ``action-rotation``   -- the same clean stream interleaved with every exact
  Stage-1 hard-positive bridge, one action per optimizer step, actions
  rotated in a seeded permutation.  One action is never averaged with
  another action nor with the clean anchors, restoring the boundary each
  action was mined for.  The interleaving is proportional and deterministic:
  action count is a unit of schedule, not a unit of dose.

Additionally the trainer saves the final Adam state (v1 discarded it) and
measures fold-1 relation diagnostics before and after training so the arm
comparison can be made without touching the outer held fold.

Author: GLM-5.3 via DeepSeek Harness, taking over the GPT-authored pipeline.
"""
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
from noise_relation_t1_t3_core import (
    batch_relation_complete_loss,
    proportional_interleave,
)
from train_noise_dreams_native import make_hdf5_spectrum
from train_noise_dreams_native_residual_stage2 import (
    adam_state_steps,
    construct_native_model,
)
from train_noise_relation_t1_t3 import packed_batches, query_span

ENCODE_BATCH = 256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--action-bank", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("clean-only", "action-rotation"), required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--listwise-temperature", type=float, default=0.07)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--maximum-spectra-per-step", type=int, default=64)
    parser.add_argument("--maximum-queries-per-step", type=int, default=4)
    return parser.parse_args()


def fold1_relation_diagnostics(
    model, preprocessor, handle, corpus: dict[str, np.ndarray], device,
) -> dict[str, float]:
    """Strict molecule-max ranking metrics on the fold-1 diagnostic corpus."""
    model.eval()
    rows_needed = sorted(set(
        [int(row) for row in corpus["query_row"]]
        + [int(row) for row in corpus["reference_row"]]
    ))
    row_position = {row: index for index, row in enumerate(rows_needed)}
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows_needed), ENCODE_BATCH):
            batch_rows = rows_needed[start:start + ENCODE_BATCH]
            spectra = []
            for row in batch_rows:
                spectrum = make_hdf5_spectrum(
                    handle["spectrum"][row], float(handle["precursor_mz"][row]),
                )
                spectra.append(preprocessor(
                    spectrum.get_peak_list(),
                    prec_mz=spectrum.get_precursor_mz(),
                    high_form=False, augment=False,
                ))
            tensor = torch.from_numpy(np.stack(spectra).astype(np.float32)).to(device)
            encoded = F.normalize(model(tensor).float(), dim=1).cpu().numpy()
            chunks.append(encoded.astype(np.float32))
    embeddings = np.concatenate(chunks, axis=0)

    strict_top1 = 0
    margins = []
    reciprocal = []
    for position in range(len(corpus["query_index"])):
        query = row_position[int(corpus["query_row"][position])]
        molecule_left = int(corpus["molecule_ptr"][position])
        molecule_right = int(corpus["molecule_ptr"][position + 1])
        molecule_scores = []
        for molecule in range(molecule_left, molecule_right):
            r0 = int(corpus["reference_ptr"][molecule])
            r1 = int(corpus["reference_ptr"][molecule + 1])
            positions = [row_position[int(row)] for row in corpus["reference_row"][r0:r1]]
            molecule_scores.append(float(np.max(
                embeddings[positions] @ embeddings[query],
            )))
        positive = molecule_scores[0]
        others = molecule_scores[1:]
        best_other = max(others) if others else float("-inf")
        strict_top1 += int(positive > best_other)
        margins.append(positive - best_other)
        better = 1 + sum(1 for value in others if value >= positive)
        reciprocal.append(1.0 / better)
    model.train()
    queries = len(corpus["query_index"])
    return {
        "queries": queries,
        "recall_at_1_strict": strict_top1 / queries,
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("T1/T3 v2 training requires an allocated GPU")
    for path in (
        args.data, args.corpus / "train.npz", args.corpus / "report.json",
        args.corpus / "validation.npz", args.corpus / "action_events.npz",
        args.action_bank, args.warm_start_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.lr != 5e-6 or args.weight_decay != 0.0
        or args.triplet_loss_margin != 0.1 or args.listwise_temperature != 0.07
        or args.n_highest_peaks != 100
    ):
        raise RuntimeError("T1/T3 v2 changes relation content, not native settings")

    corpus_report = json.loads((args.corpus / "report.json").read_text(encoding="utf-8"))
    if corpus_report.get("status") != "noise_relation_t1_t3_corpus_v2_complete":
        raise RuntimeError("T1/T3 v2 corpus is incomplete")
    with np.load(args.corpus / "train.npz", allow_pickle=False) as body:
        corpus = {name: np.asarray(body[name]) for name in body.files}
    with np.load(args.corpus / "validation.npz", allow_pickle=False) as body:
        validation_corpus = {name: np.asarray(body[name]) for name in body.files}
    with np.load(args.corpus / "action_events.npz", allow_pickle=False) as body:
        training_events = {
            name[len("train_"):]: np.asarray(body[name]) for name in body.files
            if name.startswith("train_")
        }
    with np.load(args.action_bank, allow_pickle=False) as body:
        actions = np.asarray(body["targeted_action_spectra"], dtype=np.float32)
        representable = np.asarray(body["native_action_view_representable"], dtype=bool)
    if len(training_events["action_index"]) and not np.all(
        representable[training_events["action_index"]]
    ):
        raise RuntimeError("unrepresentable Stage-1 action entered T1/T3 v2")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("T1/T3 v2 did not reconstruct the native DreaMS ContrastiveHead")
    device = torch.device("cuda")
    model.backbone.unfreeze()
    model.to(device).train()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("T1/T3 v2 failed to unfreeze the complete shared encoder")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam:
        raise RuntimeError("T1/T3 v2 optimizer is not native Adam")
    warm = torch_load_compat(args.warm_start_checkpoint, map_location="cpu")
    states = warm.get("optimizer_states")
    if not isinstance(states, list) or len(states) != 1:
        raise RuntimeError("Stage-1 warm start lacks its native Adam state")
    optimizer.load_state_dict(states[0])
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

    clean_batches = packed_batches(
        corpus, args.seed, args.maximum_spectra_per_step,
        args.maximum_queries_per_step,
    )
    rng = np.random.default_rng(args.seed)
    action_order = rng.permutation(len(training_events["action_index"]))
    if args.arm == "clean-only":
        schedule = np.zeros(len(clean_batches), dtype=np.int8)
    else:
        schedule = proportional_interleave(len(clean_batches), len(action_order))

    started = time.time()
    with h5py.File(args.data, "r") as handle:
        fold1_before = fold1_relation_diagnostics(
            model, preprocessor, handle, validation_corpus, device,
        )
    clean_index = 0
    action_index = 0
    component_sum: dict[str, float] = {}
    action_stats = {"units": 0, "active_units": 0, "hinge_sum": 0.0}
    maximum_observed_spectra = 0
    with h5py.File(args.data, "r") as handle:
        for step, unit in enumerate(schedule, start=1):
            if unit == 0:
                positions = clean_batches[clean_index]
                clean_index += 1
                payload = [query_span(corpus, position) for position in positions]
                hdf5_rows = sorted(set(
                    [int(corpus["query_row"][position]) for position in positions]
                    + [int(row) for _, references, _ in payload for row in references]
                ))
                spectra = []
                for row in hdf5_rows:
                    spectrum = make_hdf5_spectrum(
                        handle["spectrum"][row], float(handle["precursor_mz"][row]),
                    )
                    spectra.append(preprocessor(
                        spectrum.get_peak_list(),
                        prec_mz=spectrum.get_precursor_mz(),
                        high_form=False, augment=False,
                    ))
                tensor = torch.from_numpy(np.stack(spectra).astype(np.float32)).to(device)
                encoded = F.normalize(model(tensor).float(), dim=1)
                row_position = {row: index for index, row in enumerate(hdf5_rows)}
                query_losses = []
                for position, (action_set, references, pointer) in zip(
                    positions, payload, strict=True,
                ):
                    anchors = [row_position[int(corpus["query_row"][position])]]
                    anchor_positions = torch.as_tensor(anchors, device=device)
                    reference_positions = torch.as_tensor(
                        [row_position[int(row)] for row in references], device=device,
                    )
                    query_losses.append((
                        encoded[anchor_positions],
                        encoded[reference_positions],
                        torch.as_tensor(pointer, device=device, dtype=torch.long),
                        torch.ones(1, device=device, dtype=torch.float32),
                    ))
                loss, _ = batch_relation_complete_loss(
                    query_losses,
                    triplet_margin=args.triplet_loss_margin,
                    listwise_temperature=args.listwise_temperature,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite T1/T3 v2 objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                component_sum["clean_loss"] = (
                    component_sum.get("clean_loss", 0.0)
                    + float(loss.detach()) * len(positions)
                )
                component_sum["clean_queries_seen"] = (
                    component_sum.get("clean_queries_seen", 0.0) + len(positions)
                )
                maximum_observed_spectra = max(maximum_observed_spectra, len(spectra))
            else:
                event = int(action_order[action_index])
                action_index += 1
                action_row = int(training_events["action_index"][event])
                positive_row = int(training_events["positive_row"][event])
                negative_row = int(training_events["negative_row"][event])
                spectra = [actions[action_row]]
                for row in (positive_row, negative_row):
                    spectrum = make_hdf5_spectrum(
                        handle["spectrum"][row], float(handle["precursor_mz"][row]),
                    )
                    spectra.append(preprocessor(
                        spectrum.get_peak_list(),
                        prec_mz=spectrum.get_precursor_mz(),
                        high_form=False, augment=False,
                    ))
                tensor = torch.from_numpy(np.stack(spectra).astype(np.float32)).to(device)
                encoded = F.normalize(model(tensor).float(), dim=1)
                anchor, positive, negative = encoded[0], encoded[1], encoded[2]
                hinge = torch.relu(
                    args.triplet_loss_margin + (anchor @ negative) - (anchor @ positive)
                )
                if not torch.isfinite(hinge):
                    raise RuntimeError("non-finite T1/T3 v2 action hinge")
                optimizer.zero_grad(set_to_none=True)
                hinge.backward()
                optimizer.step()
                action_stats["units"] += 1
                action_stats["active_units"] += int(float(hinge.detach()) > 0.0)
                action_stats["hinge_sum"] += float(hinge.detach())
                maximum_observed_spectra = max(maximum_observed_spectra, len(spectra))
            if step % 500 == 0 or step == len(schedule):
                print(
                    f"[T1/T3 v2 seed={args.seed} arm={args.arm}] "
                    f"{step:,}/{len(schedule):,} units "
                    f"elapsed={time.time()-started:.0f}s", flush=True,
                )

    final_steps = adam_state_steps(optimizer.state_dict())
    if set(final_steps) != set(initial_steps):
        raise RuntimeError("T1/T3 v2 Adam parameter registry drifted")
    deltas = {key: final_steps[key] - initial_steps[key] for key in final_steps}
    if set(deltas.values()) != {len(schedule)}:
        raise RuntimeError("T1/T3 v2 Adam did not advance exactly once per unit")
    with h5py.File(args.data, "r") as handle:
        fold1_after = fold1_relation_diagnostics(
            model, preprocessor, handle, validation_corpus, device,
        )

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
    optimizer_checkpoint = args.output / "final_optimizer.pt"
    torch.save({
        "format": "noise_relation_t1_t3_v2_adam_v1",
        "optimizer_state_dict": optimizer.state_dict(),
        "units": len(schedule),
        "warm_start_checkpoint": str(args.warm_start_checkpoint),
    }, optimizer_checkpoint)
    report = {
        "status": "noise_relation_t1_t3_v2_training_complete",
        "arm": args.arm,
        "seed": args.seed,
        "initialization": "Stage-1 targeted champion",
        "initialization_kind": initialization_kind,
        "training": {
            "clean_batches": len(clean_batches),
            "action_units": int(action_stats["units"]),
            "total_units": len(schedule),
            "action_unit_share": (
                action_stats["units"] / len(schedule) if len(schedule) else 0.0
            ),
            "action_active_fraction": (
                action_stats["active_units"] / action_stats["units"]
                if action_stats["units"] else None
            ),
            "action_mean_hinge": (
                action_stats["hinge_sum"] / action_stats["units"]
                if action_stats["units"] else None
            ),
            "clean_mean_loss_per_query": (
                component_sum.get("clean_loss", 0.0)
                / max(1.0, component_sum.get("clean_queries_seen", 0.0))
            ),
            "maximum_observed_unique_spectra": maximum_observed_spectra,
            "optimizer": "native Adam with restored Stage-1 moments and saved final state",
        },
        "fold1_diagnostics": {
            "before": fold1_before,
            "after": fold1_after,
            "recall1_delta": fold1_after["recall_at_1_strict"] - fold1_before["recall_at_1_strict"],
            "margin_delta": fold1_after["mean_margin"] - fold1_before["mean_margin"],
        },
        "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
        "corpus_report_sha256": sha256_file(args.corpus / "report.json"),
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "final_optimizer_sha256": sha256_file(optimizer_checkpoint),
        "outer_performance_claimed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

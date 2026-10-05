#!/usr/bin/env python
"""Query-balanced exact-bridge continuation of the Stage-1 DreaMS encoder.

Every clean training query is presented exactly once.  An action-bearing query
receives one exact registered bridge in the same query objective; targeted and
same-query control arms differ only in the action spectrum tensor.
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
from e1_checkpoint_io import torch_load_compat
from noise_final_core import sha256_file
from noise_relation_t1_t3_core import relation_complete_loss
from noise_relation_t1_t3_v3_core import (
    combine_query_losses,
    exact_cosine_triplet_hinge,
    query_balanced_packed_batches,
    query_balanced_event_rotation,
)
from train_noise_dreams_native import make_hdf5_spectrum
from train_noise_dreams_native_residual_stage2 import (
    adam_state_steps,
    construct_native_model,
)
from train_noise_relation_t1_t3 import query_span
from train_noise_relation_t1_t3_v2 import fold1_relation_diagnostics


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
    parser.add_argument("--arm", choices=("targeted", "control"), required=True)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--action-fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--listwise-temperature", type=float, default=0.07)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--maximum-spectra-per-step", type=int, default=64)
    parser.add_argument("--maximum-queries-per-step", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("T1/T3 V3 training requires an allocated GPU")
    for path in (
        args.data, args.corpus / "train.npz", args.corpus / "validation.npz",
        args.corpus / "action_events.npz", args.corpus / "report.json",
        args.action_bank, args.warm_start_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.lr != 5e-6 or args.weight_decay != 0.0
        or args.triplet_loss_margin != 0.1 or args.listwise_temperature != 0.07
        or args.n_highest_peaks != 100 or args.action_fraction != 0.5
    ):
        raise RuntimeError("V3 changes exact relations and dose, not native settings")

    corpus_report = json.loads((args.corpus / "report.json").read_text(encoding="utf-8"))
    if corpus_report.get("status") != "noise_relation_t1_t3_corpus_v2_complete":
        raise RuntimeError("V3 requires the completed exact-event V2 corpus")
    corpus = load_npz(args.corpus / "train.npz")
    validation = load_npz(args.corpus / "validation.npz")
    archive = load_npz(args.corpus / "action_events.npz")
    events = {
        name[len("train_"):]: value for name, value in archive.items()
        if name.startswith("train_")
    }
    bank = load_npz(args.action_bank)
    action_key = (
        "targeted_action_spectra" if args.arm == "targeted"
        else "control_action_spectra"
    )
    required_bank = {
        "targeted_action_spectra", "control_action_spectra",
        "native_action_view_representable", "query_index", "source", "family",
    }
    missing = required_bank - set(bank)
    if missing:
        raise RuntimeError(f"V3 action bank lacks fields: {sorted(missing)}")
    action_spectra = np.asarray(bank[action_key], dtype=np.float32)
    targeted_shape = np.asarray(bank["targeted_action_spectra"]).shape
    control_shape = np.asarray(bank["control_action_spectra"]).shape
    if targeted_shape != control_shape or action_spectra.shape != targeted_shape:
        raise RuntimeError("V3 targeted/control action tensors are not aligned")

    selected, selection_report = query_balanced_event_rotation(
        events["query"], events["action_index"], corpus["query_index"],
        seed=args.seed, rotation=args.rotation,
    )
    selected_event_indices = np.asarray(list(selected.values()), dtype=np.int64)
    selected_actions = np.asarray(events["action_index"], dtype=np.int64)[selected_event_indices]
    representable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if (
        np.any(selected_actions < 0)
        or np.any(selected_actions >= len(action_spectra))
        or not np.all(representable[selected_actions])
    ):
        raise RuntimeError("V3 selected an unrepresentable action")
    if not np.array_equal(
        np.asarray(bank["query_index"], dtype=np.int64)[selected_actions],
        np.asarray(events["query"], dtype=np.int64)[selected_event_indices],
    ):
        raise RuntimeError("V3 selected action/query provenance drifted")
    selected_targeted = np.asarray(
        bank["targeted_action_spectra"], dtype=np.float32,
    )[selected_actions]
    selected_control = np.asarray(
        bank["control_action_spectra"], dtype=np.float32,
    )[selected_actions]
    distinct = np.any(selected_targeted != selected_control, axis=(1, 2))
    if not np.all(distinct):
        raise RuntimeError("V3 selected a targeted/control-identical action")
    registered_sources = set(map(str, np.asarray(bank["source"])))
    selected_source_set = set(
        str(np.asarray(bank["source"])[action]) for action in selected_actions
    )
    if selected_source_set != registered_sources:
        raise RuntimeError(
            "V3 query-balanced selection lost an action source: "
            f"selected={sorted(selected_source_set)} registered={sorted(registered_sources)}"
        )

    batches, schedule_report = query_balanced_packed_batches(
        corpus, events, selected, seed=args.seed,
        maximum_spectra=args.maximum_spectra_per_step,
        maximum_queries=args.maximum_queries_per_step,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("V3 did not reconstruct the native DreaMS ContrastiveHead")
    device = torch.device("cuda")
    model.backbone.unfreeze()
    model.to(device).train()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V3 failed to unfreeze the complete shared encoder")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam:
        raise RuntimeError("V3 optimizer is not native Adam")
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
    with h5py.File(args.data, "r") as handle:
        fold1_before = fold1_relation_diagnostics(
            model, preprocessor, handle, validation, device,
        )

    totals = Counter()
    maximum_observed_spectra = 0
    started = time.time()
    with h5py.File(args.data, "r") as handle:
        for step, positions in enumerate(batches, start=1):
            payload = [query_span(corpus, position) for position in positions]
            event_for_position: dict[int, int] = {}
            hdf5_rows = {
                int(corpus["query_row"][position]) for position in positions
            }
            for local, (position, (_unused, references, _pointer)) in enumerate(
                zip(positions, payload, strict=True)
            ):
                hdf5_rows.update(map(int, references))
                query = int(corpus["query_index"][position])
                event = selected.get(query)
                if event is not None:
                    event_for_position[local] = event
                    hdf5_rows.add(int(events["positive_row"][event]))
                    hdf5_rows.add(int(events["negative_row"][event]))
            hdf5_rows = sorted(hdf5_rows)
            action_indices = sorted({
                int(events["action_index"][event])
                for event in event_for_position.values()
            })
            spectra = []
            for row in hdf5_rows:
                spectrum = make_hdf5_spectrum(
                    handle["spectrum"][row], float(handle["precursor_mz"][row]),
                )
                spectra.append(preprocessor(
                    spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
                    high_form=False, augment=False,
                ))
            spectra.extend(action_spectra[index] for index in action_indices)
            if len(spectra) > args.maximum_spectra_per_step:
                raise RuntimeError("V3 runtime batch exceeded the spectrum budget")
            maximum_observed_spectra = max(maximum_observed_spectra, len(spectra))
            tensor = torch.from_numpy(np.stack(spectra).astype(np.float32)).to(device)
            encoded = F.normalize(model(tensor).float(), dim=1)
            row_position = {row: index for index, row in enumerate(hdf5_rows)}
            action_position = {
                action: len(hdf5_rows) + index
                for index, action in enumerate(action_indices)
            }

            clean_losses: list[torch.Tensor] = []
            action_losses: dict[int, torch.Tensor] = {}
            for local, (position, (_unused, references, pointer)) in enumerate(
                zip(positions, payload, strict=True)
            ):
                clean_loss, _detail = relation_complete_loss(
                    encoded[torch.as_tensor(
                        [row_position[int(corpus["query_row"][position])]], device=device,
                    )],
                    encoded[torch.as_tensor(
                        [row_position[int(row)] for row in references], device=device,
                    )],
                    torch.as_tensor(pointer, device=device, dtype=torch.long),
                    triplet_margin=args.triplet_loss_margin,
                    listwise_temperature=args.listwise_temperature,
                )
                clean_losses.append(clean_loss)
                event = event_for_position.get(local)
                if event is not None:
                    action = int(events["action_index"][event])
                    positive = int(events["positive_row"][event])
                    negative = int(events["negative_row"][event])
                    hinge = exact_cosine_triplet_hinge(
                        encoded[torch.as_tensor([action_position[action]], device=device)],
                        encoded[torch.as_tensor([row_position[positive]], device=device)],
                        encoded[torch.as_tensor([row_position[negative]], device=device)],
                        margin=args.triplet_loss_margin,
                    )[0]
                    action_losses[local] = hinge
                    totals["action_events"] += 1
                    totals["active_action_events"] += int(float(hinge.detach()) > 0)
                    totals["action_hinge_sum"] += float(hinge.detach())

            loss = combine_query_losses(
                clean_losses, action_losses, action_fraction=args.action_fraction,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V3 objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["queries"] += len(positions)
            totals["loss_sum"] += float(loss.detach()) * len(positions)
            if step % 250 == 0 or step == len(batches):
                print(
                    f"[T1/T3 V3 seed={args.seed} arm={args.arm}] "
                    f"{step:,}/{len(batches):,} steps elapsed={time.time()-started:.0f}s",
                    flush=True,
                )

    if totals["queries"] != len(corpus["query_index"]):
        raise RuntimeError("V3 query dose drifted during training")
    if totals["action_events"] != len(selected):
        raise RuntimeError("V3 did not expose exactly one action per action query")
    final_steps = adam_state_steps(optimizer.state_dict())
    if set(final_steps) != set(initial_steps):
        raise RuntimeError("V3 Adam parameter registry drifted")
    deltas = {key: final_steps[key] - initial_steps[key] for key in final_steps}
    if set(deltas.values()) != {len(batches)}:
        raise RuntimeError("V3 targeted/control step budget drifted")

    with h5py.File(args.data, "r") as handle:
        fold1_after = fold1_relation_diagnostics(
            model, preprocessor, handle, validation, device,
        )
    args.output.mkdir(parents=True)
    slim = {
        "format": "official_embedding_slim_v1",
        "source_checkpoint": str(args.warm_start_checkpoint),
        "source_size_bytes": args.warm_start_checkpoint.stat().st_size,
        "backbone_state_dict": {
            key: value.detach().cpu() for key, value in model.backbone.state_dict().items()
        },
        "head_state_dict": {
            key: value.detach().cpu() for key, value in model.head.state_dict().items()
        },
    }
    checkpoint = args.output / "final_slim.pt"
    torch.save(slim, checkpoint)
    optimizer_checkpoint = args.output / "final_optimizer.pt"
    torch.save({
        "format": "noise_relation_t1_t3_v3_adam_v1",
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_steps": len(batches),
        "warm_start_checkpoint": str(args.warm_start_checkpoint),
    }, optimizer_checkpoint)

    selected_sources = Counter(
        str(np.asarray(bank["source"])[action]) for action in selected_actions
    )
    selected_families = Counter(
        str(np.asarray(bank["family"])[action]) for action in selected_actions
    )
    report = {
        "status": "noise_relation_t1_t3_v3_training_complete",
        "arm": args.arm,
        "seed": args.seed,
        "rotation": args.rotation,
        "initialization": "Stage-1 targeted champion",
        "initialization_kind": initialization_kind,
        "shared_encoder_updated": True,
        "training": {
            "objective": (
                "query-balanced clean relation-complete T1/T3 plus one exact "
                "native action triplet per action query"
            ),
            "action_fraction_inside_action_query": args.action_fraction,
            "optimizer": "native Adam with restored Stage-1 moments",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "precision": "FP32",
            "queries": int(totals["queries"]),
            "action_queries": int(totals["action_events"]),
            "optimizer_steps": len(batches),
            "mean_loss_per_query": totals["loss_sum"] / totals["queries"],
            "action_active_fraction": (
                totals["active_action_events"] / totals["action_events"]
            ),
            "action_mean_hinge": totals["action_hinge_sum"] / totals["action_events"],
            "maximum_observed_unique_spectra": maximum_observed_spectra,
        },
        "selection": selection_report,
        "schedule": schedule_report,
        "selected_source_counts": dict(sorted(selected_sources.items())),
        "selected_family_counts": dict(sorted(selected_families.items())),
        "fold1_diagnostics": {
            "before": fold1_before,
            "after": fold1_after,
            "recall1_delta": (
                fold1_after["recall_at_1_strict"] - fold1_before["recall_at_1_strict"]
            ),
            "mrr_delta": (
                fold1_after["mean_reciprocal_rank"]
                - fold1_before["mean_reciprocal_rank"]
            ),
            "margin_delta": fold1_after["mean_margin"] - fold1_before["mean_margin"],
        },
        "contracts": {
            "targeted_and_control_share_event_selection": True,
            "targeted_and_control_share_positive_negative_rows": True,
            "targeted_and_control_share_query_batches": True,
            "targeted_and_control_action_tensors_are_distinct": True,
            "all_registered_action_sources_selected": True,
            "action_multiplicity_is_not_optimizer_dose": True,
            "one_action_per_action_query": True,
            "outer_held_evaluated": False,
        },
        "provenance": {
            "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
            "corpus_report_sha256": sha256_file(args.corpus / "report.json"),
            "action_bank_sha256": sha256_file(args.action_bank),
            "final_checkpoint_sha256": sha256_file(checkpoint),
            "final_optimizer_sha256": sha256_file(optimizer_checkpoint),
        },
        "claim_limit": (
            "Train-side fold-1 diagnostic only.  This run makes no outer-held "
            "or independent-GNPS performance claim."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

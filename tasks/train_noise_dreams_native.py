"""Continue the official DreaMS embedding on Noise-selected native triplets.

The production path deliberately imports and executes the repository's native
``ContrastiveSpectraDataset`` and ``ContrastiveHead``.  Noise contributes only
the spectrum registry and positive/negative memberships built by
``build_noise_dreams_native_triplets.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import inspect
import json
import random
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreams.models.heads.heads import ContrastiveHead  # noqa: E402
from dreams.utils.data import ContrastiveSpectraDataset, SpectrumPreprocessor  # noqa: E402
from dreams.utils.dformats import DataFormatA  # noqa: E402
from dreams.utils.spectra import MSnSpectrum  # noqa: E402
from noise_dreams_native_spectrum import (  # noqa: E402
    NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION,
    action_fragment_profile,
    make_action_spectrum,
)
from train_e1_identity import load_base_model  # noqa: E402

NATIVE_LOADER_COMPAT_VERSION = "chemaware_slim_to_native_contrastive_v5"
NATIVE_SCHEDULE_VERSION = "native_hard_positive_only_query_disjoint_v8"
NATIVE_MAX_EPOCHS = 1
NATIVE_MAX_OPTIMIZER_STEPS = 9000


@contextmanager
def trusted_torch_load_scope():
    """Apply ``weights_only=False`` to nested trusted checkpoint loads.

    Lightning forwards the flag for the outer ``ContrastiveHead`` checkpoint,
    but the native head constructor performs a second, otherwise implicit load
    of its DreaMS backbone.  Torch 2.6 changes that nested call to
    ``weights_only=True`` unless it is supplied explicitly.  Keep the override
    local to construction of the two repository-owned official checkpoints.
    """
    original_torch_load = torch.load
    supports_weights_only = "weights_only" in inspect.signature(
        original_torch_load
    ).parameters

    def trusted_torch_load(*args, **kwargs):
        # Lightning 2.5 passes ``weights_only=None`` explicitly for the nested
        # backbone load.  Under Torch 2.6, None selects the new safe default
        # just like omitting the argument, so both cases must become False.
        if supports_weights_only and kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--official-finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--arm", choices=("targeted", "control"), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=NATIVE_MAX_EPOCHS)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_one_pass_training_indices(
    pool: dict[str, np.ndarray],
    event_dataset_indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return one native shuffled pass with every semantic event exactly once.

    The official DreaMS fine-tuning loader uses ``shuffle=True`` and
    ``drop_last=True``.  The Noise event ledger is not necessarily divisible by
    four, so append only the minimum number of deterministic action-free clean
    events.  Every action event remains present exactly once; no query-balanced
    oversampling or custom batch ordering is introduced.
    """
    validate_pool(pool)
    event_dataset_indices = np.asarray(event_dataset_indices, dtype=np.int64)
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    queries = np.asarray(pool["event_query"], dtype=np.int64)
    actions = np.asarray(pool["event_action_index"], dtype=np.int64)
    if (
        batch_size < 2
        or event_dataset_indices.shape != kinds.shape
        or len(np.unique(event_dataset_indices)) != len(event_dataset_indices)
    ):
        raise RuntimeError("native one-pass event indices are invalid")
    action_mask = kinds > 0
    clean_mask = kinds == 0
    if (
        np.any(~np.isin(kinds, np.asarray([0, 1, 2], dtype=np.int8)))
        or np.any(actions[clean_mask] != -1)
        or np.any(actions[action_mask] < 0)
    ):
        raise RuntimeError("native one-pass event roles are invalid")
    action_ids = np.unique(actions[action_mask])
    if not np.array_equal(action_ids, np.arange(len(action_ids), dtype=np.int64)):
        raise RuntimeError("native one-pass action units are not contiguous")
    action_view_units = 0
    clean_only_units = 0
    for action in action_ids:
        modes = kinds[actions == action]
        mode_set = set(map(int, modes))
        if (mode_set == {1} and len(modes) == 1):
            clean_only_units += 1
        elif mode_set == {2} and len(modes) == 1:
            action_view_units += 1
        else:
            raise RuntimeError("native one-pass action unit has invalid triplet modes")

    padding = (-len(event_dataset_indices)) % int(batch_size)
    action_queries = np.unique(queries[action_mask])
    filler_candidates = np.flatnonzero(
        clean_mask & ~np.isin(queries, action_queries)
    ).astype(np.int64)
    if len(filler_candidates) < padding:
        raise RuntimeError("native one-pass lacks action-free clean batch fillers")
    rng = np.random.default_rng(int(seed))
    filler_events = (
        filler_candidates[rng.permutation(len(filler_candidates))[:padding]]
        if padding else np.empty(0, dtype=np.int64)
    )
    output = np.concatenate(
        (event_dataset_indices, event_dataset_indices[filler_events])
    ).astype(np.int64, copy=False)
    if len(output) % batch_size or len(output) // batch_size < 1:
        raise RuntimeError("native one-pass training events do not tile batches")
    audit = {
        "scheduler": "official_native_shuffle_drop_last_one_triplet_v5",
        "base_events": int(len(event_dataset_indices)),
        "clean_events": int(np.sum(clean_mask)),
        "action_events": int(np.sum(action_mask)),
        "semantic_action_units": int(len(action_ids)),
        "action_view_units": int(action_view_units),
        "clean_boundary_only_units": int(clean_only_units),
        "action_events_per_representable_semantic_unit": 1,
        "action_events_per_unrepresentable_semantic_unit": 1,
        "every_action_event_exposed_exactly_once": True,
        "query_balanced_oversampling": False,
        "clean_batch_padding_events": int(padding),
        "padding_events_are_action_free_clean": True,
        "total_events": int(len(output)),
        "batches_per_epoch": int(len(output) // batch_size),
        "maximum_distinct_actions_per_query": int(max(
            np.bincount(queries[action_mask], minlength=int(queries.max()) + 1),
            default=0,
        )),
        "dose_interpretation": (
            "one native shuffled pass; each effective unit contributes exactly "
            "one clean-anchor/action-positive or clean-only boundary triplet"
        ),
    }
    return output, filler_events, audit


def native_query_disjoint_one_pass_batches(
    pool: dict[str, np.ndarray],
    event_dataset_indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[list[list[int]], np.ndarray, dict[str, object]]:
    """Expose each native event once while keeping queries distinct per batch."""
    validate_pool(pool)
    event_dataset_indices = np.asarray(event_dataset_indices, dtype=np.int64)
    queries = np.asarray(pool["event_query"], dtype=np.int64)
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    actions = np.asarray(pool["event_action_index"], dtype=np.int64)
    if event_dataset_indices.shape != queries.shape or batch_size < 2:
        raise RuntimeError("query-disjoint schedule inputs are invalid")
    if np.any(~np.isin(kinds, np.asarray([0, 1, 2], dtype=np.int8))):
        raise RuntimeError("query-disjoint schedule received an unknown event kind")
    action_ids = np.unique(actions[actions >= 0])
    if not np.array_equal(action_ids, np.arange(len(action_ids), dtype=np.int64)):
        raise RuntimeError("query-disjoint action units are not contiguous")
    for action in action_ids:
        modes = kinds[actions == action]
        if not (
            (len(modes) == 1 and set(map(int, modes)) == {1})
            or (len(modes) == 1 and set(map(int, modes)) == {2})
        ):
            raise RuntimeError("query-disjoint action unit lacks its sole native triplet")

    rng = np.random.default_rng(int(seed))
    by_query: dict[int, list[int]] = {}
    for query in np.unique(queries):
        positions = np.flatnonzero(queries == query)
        by_query[int(query)] = list(map(int, positions[rng.permutation(len(positions))]))

    # Largest queues are consumed first. This prevents high-action queries
    # from becoming an unschedulable tail while never putting two events from
    # the same query into one optimizer batch.
    random_tie = {query: float(rng.random()) for query in by_query}
    batches: list[list[int]] = []
    filler_positions: list[int] = []
    clean_action_free = np.flatnonzero(
        (kinds == 0) & ~np.isin(queries, np.unique(queries[kinds > 0]))
    ).astype(np.int64)
    if len(clean_action_free) < batch_size - 1:
        raise RuntimeError("query-disjoint schedule lacks action-free clean fillers")
    filler_cursor = 0

    heap = [
        (-len(queue), random_tie[query], query)
        for query, queue in by_query.items() if queue
    ]
    heapq.heapify(heap)
    while heap:
        chosen = [heapq.heappop(heap) for _ in range(min(batch_size, len(heap)))]
        chosen_queries = [int(body[2]) for body in chosen]
        positions = [by_query[query].pop() for query in chosen_queries]
        used_queries = set(chosen_queries)
        while len(positions) < batch_size:
            found = False
            for _ in range(len(clean_action_free)):
                candidate = int(clean_action_free[filler_cursor % len(clean_action_free)])
                filler_cursor += 1
                if int(queries[candidate]) not in used_queries:
                    positions.append(candidate)
                    filler_positions.append(candidate)
                    used_queries.add(int(queries[candidate]))
                    found = True
                    break
            if not found:
                raise RuntimeError("could not complete a query-disjoint native batch")
        batches.append(list(map(int, event_dataset_indices[positions])))
        for _, tie, query in chosen:
            if by_query[int(query)]:
                heapq.heappush(
                    heap, (-len(by_query[int(query)]), float(tie), int(query))
                )

    flattened = np.asarray([index for batch in batches for index in batch], dtype=np.int64)
    if len(filler_positions) >= batch_size:
        raise RuntimeError(
            "query-disjoint scheduling required more than final-batch padding"
        )
    # Event dataset indices are contiguous and sorted in production. Keep an
    # explicit mapping so the invariant also holds in synthetic tests.
    position_by_dataset_index = {
        int(dataset_index): position
        for position, dataset_index in enumerate(event_dataset_indices)
    }
    counts = np.zeros(len(event_dataset_indices), dtype=np.int64)
    for dataset_index in flattened:
        counts[position_by_dataset_index[int(dataset_index)]] += 1
    filler_set = set(filler_positions)
    if any(counts[position] != 1 for position in range(len(counts)) if position not in filler_set):
        raise RuntimeError("query-disjoint schedule lost or duplicated a native event")
    if np.any(counts < 1):
        raise RuntimeError("query-disjoint schedule did not expose every native event")
    for batch in batches:
        positions = [position_by_dataset_index[int(index)] for index in batch]
        if len(set(map(int, queries[positions]))) != len(batch):
            raise RuntimeError("one query appears twice in a native optimizer batch")

    action_counts = np.bincount(actions[actions >= 0], minlength=len(action_ids))
    return batches, np.asarray(filler_positions, dtype=np.int64), {
        "scheduler": "native_query_disjoint_hard_positive_only_v8",
        "base_events": int(len(event_dataset_indices)),
        "clean_events": int(np.sum(kinds == 0)),
        "action_events": int(np.sum(kinds > 0)),
        "semantic_action_units": int(len(action_counts)),
        "hard_positive_action_events": int(np.sum(kinds == 2)),
        "forbidden_clean_to_action_events": int(np.sum(kinds == 3)),
        "clean_boundary_fallback_events": int(np.sum(kinds == 1)),
        "every_base_event_exposed_exactly_once": True,
        "same_query_events_never_share_an_optimizer_batch": True,
        "query_balanced_oversampling": False,
        "synthetic_query_equalization_events": 0,
        "clean_batch_padding_events": int(len(filler_positions)),
        "padding_events_are_action_free_clean": True,
        "padding_is_final_batch_only": bool(len(filler_positions) < batch_size),
        "total_events": int(len(flattened)),
        "batches_per_epoch": int(len(batches)),
        "minimum_action_events_per_semantic_unit": int(action_counts.min()),
        "maximum_action_events_per_semantic_unit": int(action_counts.max()),
    }


def load_trusted_native_contrastive_head(
    checkpoint: Path,
    *,
    architecture_checkpoint: Path,
    map_location: torch.device,
    strict: bool = True,
    **overrides,
) -> ContrastiveHead:
    """Build the native head from the frozen official slim initialization.

    This is the same adapter used by the successful ChemAware native run:
    ``load_base_model`` reconstructs the official backbone/head from
    ``official_embedding_slim.pt`` plus the raw architecture package, then the
    repository's unmodified ``ContrastiveHead`` owns those exact weights.
    Calling Lightning ``load_from_checkpoint`` on the slim package is invalid
    because the slim package intentionally has no trainer metadata.
    """
    if strict is not True:
        raise RuntimeError("native official initialization requires strict weights")
    if torch.device(map_location).type != "cpu":
        raise RuntimeError("native official initialization must be reconstructed on CPU")
    required = {"lr", "weight_decay", "triplet_loss_margin"}
    if set(overrides) != required:
        raise RuntimeError(
            f"native ContrastiveHead overrides {sorted(overrides)} != {sorted(required)}"
        )
    initialized, kind = load_base_model(
        checkpoint,
        architecture_checkpoint,
        torch.device("cpu"),
        n_highest_peaks=100,
    )
    if kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError(f"official initialization has unsupported format: {kind}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute 'backbone.*already saved during checkpointing.*",
        )
        model = ContrastiveHead(
            initialized.backbone,
            float(overrides["lr"]),
            float(overrides["weight_decay"]),
            triplet_loss_margin=float(overrides["triplet_loss_margin"]),
        )
    model.head.load_state_dict(initialized.head.state_dict(), strict=True)
    model.unfreeze_backbone_at_epoch = 0
    del initialized
    return model


def make_hdf5_spectrum(raw: np.ndarray, precursor_mz: float) -> MSnSpectrum:
    raw = np.asarray(raw, dtype=np.float32)
    keep = (raw[0] > 0) & (raw[1] > 0)
    if not np.any(keep):
        raise RuntimeError("empty HDF5 spectrum reached native DreaMS dataset")
    return MSnSpectrum(
        peak_list=raw[:, keep],
        precursor_mz=float(precursor_mz),
        precursor_charge=1,
        assert_is_valid=False,
    )


def validate_pool(pool: dict[str, np.ndarray]) -> None:
    required = {
        "registry_kind",
        "registry_source_index",
        "anchor_idx",
        "positive_ptr",
        "positive_idx",
        "negative_ptr",
        "negative_idx",
        "event_kind",
        "event_query",
        "event_action_index",
        "event_formula",
    }
    if set(pool) != required:
        raise RuntimeError(
            f"native triplet pool keys {sorted(pool)} != {sorted(required)}"
        )
    n_events = len(pool["anchor_idx"])
    n_registry = len(pool["registry_kind"])
    if (
        len(pool["registry_source_index"]) != n_registry
        or len(pool["positive_ptr"]) != n_events + 1
        or len(pool["negative_ptr"]) != n_events + 1
        or len(pool["event_kind"]) != n_events
        or len(pool["event_query"]) != n_events
        or len(pool["event_action_index"]) != n_events
        or len(pool["event_formula"]) != n_events
    ):
        raise RuntimeError("native triplet pool arrays are not aligned")
    if (
        np.any(pool["anchor_idx"] < 0)
        or np.any(pool["anchor_idx"] >= n_registry)
        or np.any(pool["positive_idx"] < 0)
        or np.any(pool["positive_idx"] >= n_registry)
        or np.any(pool["negative_idx"] < 0)
        or np.any(pool["negative_idx"] >= n_registry)
        or np.any(np.diff(pool["positive_ptr"]) < 1)
        or np.any(np.diff(pool["negative_ptr"]) < 1)
    ):
        raise RuntimeError("native triplet pool contains an empty or invalid membership")


def native_dataset(
    pool: dict[str, np.ndarray],
    data: Path,
    action_spectra: np.ndarray,
    spec_preproc,
) -> tuple[ContrastiveSpectraDataset, np.ndarray, dict[str, float | int]]:
    if type(spec_preproc) is not SpectrumPreprocessor:
        raise RuntimeError(
            "native Noise dataset requires the unmodified SpectrumPreprocessor"
        )
    validate_pool(pool)
    kind = np.asarray(pool["registry_kind"], dtype=np.int8)
    source = np.asarray(pool["registry_source_index"], dtype=np.int64)
    spectra: list[MSnSpectrum | None] = [None] * len(kind)
    hdf5_positions = np.flatnonzero((kind == 0) | (kind == 2))
    action_positions = np.flatnonzero(kind == 1)
    if np.any((kind != 0) & (kind != 1) & (kind != 2)):
        raise RuntimeError("native spectrum registry has an unknown source kind")
    with h5py.File(data, "r") as handle:
        rows = source[hdf5_positions]
        if np.any((rows < 0) | (rows >= len(handle["spectrum"]))):
            raise RuntimeError("native spectrum registry has an out-of-range HDF5 row")
        for position, row in zip(hdf5_positions, rows, strict=True):
            spectra[int(position)] = make_hdf5_spectrum(
                handle["spectrum"][int(row)],
                handle["precursor_mz"][int(row)],
            )
    action_indices = source[action_positions]
    if len(action_indices) and (
        np.any(action_indices < 0) or np.any(action_indices >= len(action_spectra))
    ):
        raise RuntimeError("native spectrum registry has an invalid action index")
    maximum_roundtrip_error = 0.0
    canonicalized_internal_padding_actions = 0
    for position, action_index in zip(action_positions, action_indices, strict=True):
        source_tensor = action_spectra[int(action_index)]
        profile = action_fragment_profile(source_tensor)
        if not profile["native_action_view_representable"]:
            raise RuntimeError("unrepresentable all-zero action reached native dataset")
        spectrum = make_action_spectrum(source_tensor)
        replay = spec_preproc(
            spectrum.get_peak_list(),
            prec_mz=spectrum.get_precursor_mz(),
            high_form=False,
        )
        # Native preprocessing always moves true [0, 0] padding behind real
        # peaks.  Action operators can leave such holes internally; ignore only
        # that padding layout and require every real token to replay exactly.
        source_tokens = source_tensor[source_tensor[:, 0] > 0].copy()
        replay_tokens = replay[replay[:, 0] > 0].copy()
        if source_tokens.shape != replay_tokens.shape:
            raise RuntimeError("Noise action lost a real token in native preprocessing")
        error = float(np.max(np.abs(replay_tokens - source_tokens)))
        maximum_roundtrip_error = max(maximum_roundtrip_error, error)
        nonpadding = np.flatnonzero(source_tensor[:, 0] > 0)
        if len(nonpadding) and not np.array_equal(
            nonpadding, np.arange(len(nonpadding), dtype=np.int64)
        ):
            canonicalized_internal_padding_actions += 1
        spectra[int(position)] = spectrum
    if maximum_roundtrip_error != 0.0:
        raise RuntimeError(
            "Noise action does not round-trip through native SpectrumPreprocessor: "
            f"{maximum_roundtrip_error}"
        )
    if any(spectrum is None for spectrum in spectra):
        raise RuntimeError("native spectrum registry was not fully materialized")

    # ContrastiveSpectraDataset stores memberships on the same DataFrame row as
    # the anchor.  A registry spectrum can anchor several distinct event-level
    # triplets (the broad clean pool plus one clean/action positive relation
    # for every representable action, or one exact clean fallback).
    # Writing those memberships onto the registry row would make later events
    # overwrite earlier ones.  Append one anchor row per event instead: the
    # immutable spectrum object is shared, while positive/negative membership
    # remains event-specific. Positive and negative indices still address the
    # canonical registry rows at the front of the DataFrame.
    registry_count = len(spectra)
    event_anchors = pool["anchor_idx"].astype(np.int64)
    event_spectra = [spectra[int(anchor)] for anchor in event_anchors]
    event_positive_lists: list[list[int]] = []
    event_negative_lists: list[list[int]] = []
    for event in range(len(event_anchors)):
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        event_positive_lists.append(
            list(map(int, pool["positive_idx"][p0:p1]))
        )
        event_negative_lists.append(
            list(map(int, pool["negative_idx"][n0:n1]))
        )
    frame = pd.DataFrame({
        "MSnSpectrum": spectra + event_spectra,
        "pos_idx": ([[] for _ in spectra] + event_positive_lists),
        "neg_idx": ([[] for _ in spectra] + event_negative_lists),
    })
    dataset = ContrastiveSpectraDataset(
        frame,
        spec_preproc=spec_preproc,
        n_pos_samples=1,
        n_neg_samples=1,
        return_smiles=False,
    )
    event_dataset_indices = np.arange(
        registry_count, registry_count + len(event_anchors), dtype=np.int64
    )
    return dataset, event_dataset_indices, {
        "registry_spectra": int(len(spectra)),
        "event_anchor_rows": int(len(event_anchors)),
        "dataset_rows": int(len(frame)),
        "event_memberships_are_independent": True,
        "hdf5_spectra": int(len(hdf5_positions)),
        "action_spectra": int(len(action_positions)),
        "action_native_preprocessor_max_abs_error": maximum_roundtrip_error,
        "action_internal_padding_layouts_canonicalized": int(
            canonicalized_internal_padding_actions
        ),
        "unmodified_native_preprocessor": True,
        "action_spectrum_adapter_version": NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION,
    }


def load_official_continuation(args: argparse.Namespace) -> ContrastiveHead:
    model = load_trusted_native_contrastive_head(
        args.official_finetuned_checkpoint,
        architecture_checkpoint=args.architecture_checkpoint,
        lr=args.lr,
        weight_decay=args.weight_decay,
        triplet_loss_margin=args.triplet_loss_margin,
        map_location=torch.device("cpu"), strict=True,
    )
    model.unfreeze_backbone_at_epoch = 0
    if type(model) is not ContrastiveHead:
        raise RuntimeError("official continuation did not load the native ContrastiveHead")
    if not isinstance(model.head, torch.nn.Linear):
        raise RuntimeError("native DreaMS contrastive projection is not Linear")
    if (
        model.head.in_features != 1024
        or model.head.out_features != 1024
        or model.head.bias is None
    ):
        raise RuntimeError("native DreaMS projection is not the official 1024D biased head")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("native DreaMS model contains a non-FP32 parameter")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise RuntimeError("native DreaMS optimizer is not one Adam parameter group")
    group = optimizer.param_groups[0]
    if abs(float(group["lr"]) - args.lr) > 1e-15 or float(group["weight_decay"]) != 0:
        raise RuntimeError("native DreaMS Adam hyperparameters drifted")
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {id(parameter) for parameter in group["params"]}
    if optimizer_ids != parameter_ids:
        raise RuntimeError("native DreaMS Adam does not own every model parameter")
    del optimizer
    return model


def main() -> None:
    args = arguments()
    if args.resume_checkpoint is None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
    else:
        if not args.output.is_dir() or not args.resume_checkpoint.is_file():
            raise FileNotFoundError(
                "native resume requires the existing arm directory and checkpoint"
            )
        if args.resume_checkpoint.parent.resolve() != args.output.resolve():
            raise RuntimeError("resume checkpoint must belong to the requested arm output")
    if not torch.cuda.is_available():
        raise RuntimeError("formal native DreaMS continuation requires a CUDA allocation")
    expected = {
        "seed": 3407,
        "lr": 5e-6,
        "weight_decay": 0.0,
        "triplet_loss_margin": 0.1,
        "batch_size": 4,
        "max_epochs": NATIVE_MAX_EPOCHS,
        "n_highest_peaks": 100,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"formal native DreaMS hyperparameters drifted: {observed} != {expected}"
        )
    for path in (
        args.data,
        args.triplet_dir / "train_pool.npz",
        args.triplet_dir / "validation_pool.npz",
        args.triplet_dir / "action_spectra.npz",
        args.triplet_dir / "selected_actions.csv.gz",
        args.triplet_dir / "qualified_actions.csv.gz",
        args.triplet_dir / "action_aliases.csv.gz",
        args.triplet_dir / "official_replay.json",
        args.triplet_dir / "native_official_embeddings.npz",
        args.triplet_dir / "report.json",
        args.official_finetuned_checkpoint,
        args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    triplet_report = json.loads(
        (args.triplet_dir / "report.json").read_text(encoding="utf-8")
    )
    gates = triplet_report.get("gates")
    if (
        triplet_report.get("status") != "NOISE_DREAMS_NATIVE_TRIPLETS_COMPLETE"
        or triplet_report.get("implementation", {}).get("builder_version")
        != "native_hard_positive_only_v16"
        or triplet_report.get("implementation", {}).get(
            "matched_control_version"
        ) != "same_query_registered_control_spectrum_v1"
        or triplet_report.get("implementation", {}).get(
            "action_spectrum_adapter_version"
        ) != NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise RuntimeError("native Noise triplet artifact did not pass its scientific gates")
    artifact_paths = {
        "train_pool_sha256": args.triplet_dir / "train_pool.npz",
        "validation_pool_sha256": args.triplet_dir / "validation_pool.npz",
        "action_spectra_sha256": args.triplet_dir / "action_spectra.npz",
        "selected_actions_sha256": args.triplet_dir / "selected_actions.csv.gz",
        "qualified_actions_sha256": args.triplet_dir / "qualified_actions.csv.gz",
        "action_aliases_sha256": args.triplet_dir / "action_aliases.csv.gz",
    }
    registered_artifacts = triplet_report.get("output_artifacts")
    if not isinstance(registered_artifacts, dict):
        raise RuntimeError("native triplet report does not bind its output artifacts")
    for key, path in artifact_paths.items():
        if registered_artifacts.get(key) != sha256_file(path):
            raise RuntimeError(f"native triplet artifact drifted: {key}")
    replay_report = json.loads(
        (args.triplet_dir / "official_replay.json").read_text(encoding="utf-8")
    )
    replay_gates = replay_report.get("gates")
    replay_schedule = replay_report.get("exact_native_training_schedule", {})
    if (
        replay_report.get("status") != "NOISE_DREAMS_NATIVE_OFFICIAL_REPLAY_PASS"
        or replay_report.get("replay_geometry_version")
        != "native_hard_positive_only_official_v16"
        or replay_report.get("initialization_adapter_version")
        != NATIVE_LOADER_COMPAT_VERSION
        or not isinstance(replay_gates, dict)
        or not replay_gates
        or not all(value is True for value in replay_gates.values())
        or replay_report.get("provenance", {}).get("triplet_report_sha256")
        != sha256_file(args.triplet_dir / "report.json")
        or replay_report.get("provenance", {}).get("selected_actions_sha256")
        != sha256_file(args.triplet_dir / "selected_actions.csv.gz")
        or replay_report.get("provenance", {}).get("qualified_actions_sha256")
        != sha256_file(args.triplet_dir / "qualified_actions.csv.gz")
        or replay_report.get("provenance", {}).get("action_aliases_sha256")
        != sha256_file(args.triplet_dir / "action_aliases.csv.gz")
        or replay_report.get("provenance", {}).get("action_spectra_sha256")
        != sha256_file(args.triplet_dir / "action_spectra.npz")
        or replay_report.get("provenance", {}).get(
            "native_official_embedding_cache_sha256"
        ) != sha256_file(args.triplet_dir / "native_official_embeddings.npz")
        or replay_schedule.get("weighting_version") != NATIVE_SCHEDULE_VERSION
        or replay_schedule.get("seed") != args.seed
        or replay_schedule.get("epochs") != args.max_epochs
        or replay_schedule.get("batch_size") != args.batch_size
        or int(replay_schedule.get("optimizer_steps_per_arm", 0)) < 1
    ):
        raise RuntimeError("official-initialization action replay did not pass")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = load_official_continuation(args)
    # This mirrors dreams/training/train.py:41-46.  The preprocessing object
    # stored inside the SSL backbone checkpoint describes its historical SSL
    # loader and is not the contrastive fine-tuning dataset preprocessor.
    spec_preproc = SpectrumPreprocessor(
        dformat=DataFormatA(),
        prec_intens=1.1,
        n_highest_peaks=args.n_highest_peaks,
        spec_entropy_cleaning=False,
        precision=32,
        mz_shift_aug_p=0,
        mz_shift_aug_max=0,
    )
    action_bank = load_npz(args.triplet_dir / "action_spectra.npz")
    action_key = (
        "targeted_action_spectra"
        if args.arm == "targeted" else "control_action_spectra"
    )
    if action_key not in action_bank or "native_action_view_representable" not in action_bank:
        raise RuntimeError(f"triplet artifact lacks {action_key}")
    action_spectra = np.asarray(action_bank[action_key], dtype=np.float32)
    action_capable = np.asarray(
        action_bank["native_action_view_representable"], dtype=bool
    )
    if action_capable.shape != (len(action_spectra),):
        raise RuntimeError("native action representability mask is not aligned")
    train_pool = load_npz(args.triplet_dir / "train_pool.npz")
    validation_pool = load_npz(args.triplet_dir / "validation_pool.npz")
    materialized_action_indices = np.asarray(
        train_pool["event_action_index"], dtype=np.int64
    )[np.asarray(train_pool["event_kind"], dtype=np.int8) == 2]
    if (
        np.any(materialized_action_indices < 0)
        or np.any(materialized_action_indices >= len(action_capable))
        or not np.all(action_capable[materialized_action_indices])
    ):
        raise RuntimeError("unrepresentable action was registered as a materialized event")
    train_dataset, train_indices, train_materialization = native_dataset(
        train_pool,
        args.data,
        action_spectra,
        spec_preproc,
    )
    validation_dataset, validation_indices, validation_materialization = native_dataset(
        validation_pool,
        args.data,
        action_spectra,
        spec_preproc,
    )
    train_batches, train_filler_events, schedule_audit = (
        native_query_disjoint_one_pass_batches(
        train_pool,
        batch_size=args.batch_size,
        seed=args.seed,
        event_dataset_indices=train_indices,
    ))
    action_event_counts = np.bincount(
        np.asarray(train_pool["event_action_index"], dtype=np.int64)[
            np.asarray(train_pool["event_action_index"], dtype=np.int64) >= 0
        ],
        minlength=schedule_audit["semantic_action_units"],
    ).astype(np.int64)
    schedule_exposure_sha256 = hashlib.sha256(
        np.ascontiguousarray(action_event_counts).tobytes()
    ).hexdigest()
    if (
        schedule_audit["every_base_event_exposed_exactly_once"] is not True
        or schedule_audit["same_query_events_never_share_an_optimizer_batch"] is not True
        or schedule_audit["query_balanced_oversampling"] is not False
        or schedule_audit["synthetic_query_equalization_events"] != 0
        or schedule_audit["forbidden_clean_to_action_events"] != 0
        or schedule_audit["padding_is_final_batch_only"] is not True
        or schedule_audit["minimum_action_events_per_semantic_unit"] != 1
        or schedule_audit["maximum_action_events_per_semantic_unit"] != 1
        or schedule_audit["batches_per_epoch"] > NATIVE_MAX_OPTIMIZER_STEPS
        or schedule_audit["batches_per_epoch"]
        != replay_schedule.get("optimizer_steps_per_arm")
        or schedule_exposure_sha256
        != replay_schedule.get("action_unit_exposure_sha256")
    ):
        raise RuntimeError(f"native one-pass schedule failed: {schedule_audit}")
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batches,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    if not len(train_loader) or not len(validation_loader):
        raise RuntimeError("native DreaMS train or validation loader is empty")

    args.output.mkdir(parents=True, exist_ok=args.resume_checkpoint is not None)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output,
        filename="last",
        # The published command's save_top_k=-1 would create several thousand
        # 1.2-GB files at this scale.  Overwrite one native resumable checkpoint;
        # this changes storage cadence, never model/loss/optimizer semantics.
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        precision="32-true",
        logger=False,
        callbacks=[callback],
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    if args.resume_checkpoint is None:
        trainer.validate(model, dataloaders=validation_loader)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=(
            str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None else None
        ),
    )
    final_checkpoint = args.output / "final.ckpt"
    last_checkpoint = Path(callback.last_model_path) if callback.last_model_path else (
        args.output / "last.ckpt"
    )
    if not last_checkpoint.is_file():
        raise RuntimeError("native ModelCheckpoint produced no resumable last.ckpt")
    if final_checkpoint.exists():
        raise FileExistsError(final_checkpoint)
    # last.ckpt already contains the final native model and Adam state.  An
    # atomic rename avoids writing a second ~1.2GB archive, the failure mode
    # that previously exhausted the validation filesystem.
    last_checkpoint.replace(final_checkpoint)
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    required_checkpoint_keys = {
        "state_dict",
        "optimizer_states",
        "hyper_parameters",
        "epoch",
        "global_step",
    }
    if missing := required_checkpoint_keys - set(checkpoint):
        raise RuntimeError(f"native final checkpoint lacks {sorted(missing)}")
    state_keys = set(checkpoint["state_dict"])
    if not any(key.startswith("backbone.") for key in state_keys) or not any(
        key.startswith("head.") for key in state_keys
    ):
        raise RuntimeError("native final checkpoint lacks backbone or head state")
    report = {
        "status": "NOISE_DREAMS_NATIVE_TRAINING_COMPLETE",
        "arm": args.arm,
        "training_runtime": {
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "preprocessor": type(spec_preproc).__name__,
            "loss": "ContrastiveHead.step cosine triplet hinge",
            "optimizer": "ContrastiveHead.configure_optimizers torch.optim.Adam",
            "custom_loss": False,
            "custom_model": False,
            "custom_optimizer": False,
            "custom_triplet_content": True,
            "initialization_adapter": NATIVE_LOADER_COMPAT_VERSION,
        },
        "frozen_hyperparameters": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "triplet_loss_margin": args.triplet_loss_margin,
            "batch_size": args.batch_size,
            "n_pos_samples": 1,
            "n_neg_samples": 1,
            "precision": 32,
            "unfreeze_backbone_at_epoch": 0,
            "max_epochs": args.max_epochs,
            "schedule_version": NATIVE_SCHEDULE_VERSION,
            "native_shuffle": False,
            "native_drop_last": False,
            "query_disjoint_batch_order_only": True,
            "n_highest_peaks": args.n_highest_peaks,
            "precursor_intensity": 1.1,
        },
        "initialization": (
            "official_embedding_slim reconstructed into native ContrastiveHead "
            "with the frozen ssl_model_server architecture"
        ),
        "resumed_from": (
            str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None else None
        ),
        "train_anchors": int(len(train_indices)),
        "validation_anchors": int(len(validation_indices)),
        "train_materialization": train_materialization,
        "validation_materialization": validation_materialization,
        "native_one_pass_schedule": schedule_audit,
        "train_filler_event_indices": list(map(int, train_filler_events)),
        "official_initialization_action_replay": replay_report["exact_boundary"],
        "official_initialization_schedule_replay": replay_schedule,
        "final_checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_format": "native PyTorch Lightning ContrastiveHead .ckpt",
        "final_epoch": int(checkpoint["epoch"]),
        "final_global_step": int(checkpoint["global_step"]),
        "outer_performance_claimed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

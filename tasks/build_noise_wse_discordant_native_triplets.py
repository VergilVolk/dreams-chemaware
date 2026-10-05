#!/usr/bin/env python
"""Build one information-dense native DreaMS triplet per MassSpecGym query.

The frozen Noise V1 encoder supplies current cosine geometry.  Weighted
spectral entropy (WSE) is an independent, label-blind spectrum score used only
to select training relations on MassSpecGym.  WSE never enters the loss.
Every corrected-graph query contributes exactly one native triplet, in this
priority order:

1. a measured WSE-supported relation that is still active under V1;
2. one exact historical Stage-1 action relation;
3. a measured high-margin V1 preservation relation; or
4. for singleton identities, a light same-identity noise-to-clean relation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from build_noise_massspecgym_full_triplets import (
    Registry,
    load_complete_embeddings,
    make_noise_view,
    stage1_exact_actions,
)
from noise_final_core import CandidateGraph, sha256_file
from noise_gnps_article_spectral_scores import (
    prepare_spectrum,
    weighted_entropy_backend,
    weighted_entropy_similarity,
)
from noise_massspecgym_full_triplet_core import decode_array, ordered_unique
from noise_wse_discordant_triplet_core import (
    candidate_molecule_max_rows,
    choose_v1_preservation_triplet,
    choose_wse_supported_triplet,
)
from train_noise_dreams_native import make_hdf5_spectrum


WSE_RELATION = 0
STAGE1_REPLAY = 1
MEASURED_PRESERVATION = 2
SINGLETON_NOISE_PRESERVATION = 3
EVENT_NAMES = {
    WSE_RELATION: "wse_supported_relation",
    STAGE1_REPLAY: "stage1_exact_replay",
    MEASURED_PRESERVATION: "measured_v1_preservation",
    SINGLETON_NOISE_PRESERVATION: "singleton_noise_preservation",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embedding-shard", type=Path, action="append", required=True)
    parser.add_argument("--stage1-triplets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--native-margin", type=float, default=0.1)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--max-positive-candidates", type=int, default=6)
    parser.add_argument("--max-negative-candidates", type=int, default=8)
    return parser.parse_args()


def prepared_spectra(handle: h5py.File) -> list[np.ndarray]:
    """Read HDF5 sequentially once; later mining is memory-local."""
    output: list[np.ndarray] = []
    for row in range(len(handle["spectrum"])):
        spectrum = prepare_spectrum(np.asarray(handle["spectrum"][row]), n_peaks=100)
        if spectrum.shape[1] == 0:
            raise RuntimeError(f"MassSpecGym row {row} has no usable fragment peak")
        output.append(spectrum)
        if row and row % 20000 == 0:
            print(f"[wse-prepare] {row:,}/{len(handle['spectrum']):,} spectra", flush=True)
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.native_margin != 0.1
        or args.fragment_tolerance != 0.02
        or min(args.max_positive_candidates, args.max_negative_candidates) < 1
    ):
        raise ValueError("registered WSE mining settings drifted")
    required = [
        args.graph, args.data, *args.embedding_shard,
        args.stage1_triplets / "train_pool.npz",
        args.stage1_triplets / "action_spectra.npz",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    backend = weighted_entropy_backend()
    if not backend.startswith("ms_entropy_"):
        raise RuntimeError("formal WSE mining requires the pinned ms_entropy backend")

    graph = CandidateGraph(args.graph)
    with h5py.File(args.data, "r") as handle:
        n_rows = len(handle["spectrum"])
        identities = np.asarray(
            [value[:14] for value in decode_array(handle["INCHIKEY"][:])], dtype="U14",
        )
        adducts = decode_array(handle["adduct"][:])
        precursor_mz = np.asarray(handle["precursor_mz"][:], dtype=np.float64)
        spectra = prepared_spectra(handle)
    if not (
        len(identities) == len(adducts) == len(precursor_mz) == len(spectra) == n_rows
        and np.all(np.isfinite(precursor_mz))
    ):
        raise RuntimeError("MassSpecGym HDF5 arrays are not aligned")
    if not np.array_equal(identities[graph.query_row], graph.query_ik14.astype(str)):
        raise RuntimeError("candidate graph query identity drifted from MassSpecGym")

    embeddings = load_complete_embeddings(args.embedding_shard, n_rows)
    actions, actions_by_query, action_report = stage1_exact_actions(args.stage1_triplets)
    identity_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row, (identity, adduct) in enumerate(zip(identities, adducts, strict=True)):
        identity_rows[(str(adduct), str(identity))].append(row)

    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    registry = Registry()
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    event_query: list[int] = []
    event_kind: list[int] = []
    event_action: list[int] = []
    noise_spectra: list[np.ndarray] = []

    ledger_kind: list[int] = []
    ledger_positive: list[int] = []
    ledger_negative: list[int] = []
    ledger_action: list[int] = []
    ledger_v1_margin: list[float] = []
    ledger_wse_margin: list[float] = []
    wse_misranked = 0
    wse_boundary = 0
    stage1_actions_available = sum(map(len, actions_by_query.values()))
    stage1_actions_selected = 0
    short_noise_views = 0

    def append_event(
        *, anchor: int, positive: int, negative: int,
        query: int, kind: int, action: int,
        v1_margin: float, wse_margin: float,
    ) -> None:
        if len({anchor, positive, negative}) != 3:
            raise RuntimeError("native WSE triplet roles overlap")
        anchors.append(int(anchor))
        positives.append(int(positive)); positive_ptr.append(len(positives))
        negatives.append(int(negative)); negative_ptr.append(len(negatives))
        event_query.append(int(query)); event_kind.append(int(kind))
        event_action.append(int(action))
        ledger_kind.append(int(kind)); ledger_positive.append(int(positive))
        ledger_negative.append(int(negative)); ledger_action.append(int(action))
        ledger_v1_margin.append(float(v1_margin)); ledger_wse_margin.append(float(wse_margin))

    with h5py.File(args.data, "r") as handle:
        for query in range(graph.n_queries):
            query_row = int(graph.query_row[query])
            query_identity = str(identities[query_row])
            query_adduct = str(adducts[query_row])
            query_embedding = embeddings[query_row]
            _, pair_rows, local_ptr, _ = graph.query_block(query)
            pair_rows = np.asarray(pair_rows, dtype=np.int64)
            pair_scores = embeddings[pair_rows] @ query_embedding
            molecule_rows, molecule_scores = candidate_molecule_max_rows(
                pair_rows, local_ptr, pair_scores,
            )
            if str(identities[molecule_rows[0]]) != query_identity:
                raise RuntimeError("positive candidate molecule identity drifted")
            if np.any(identities[molecule_rows[1:]] == query_identity):
                raise RuntimeError("negative candidate block contains the true identity")

            positive_block = ordered_unique(map(
                int, pair_rows[int(local_ptr[0]):int(local_ptr[1])],
            ))
            positive_rows = np.asarray(
                [row for row in positive_block if row != query_row], dtype=np.int64,
            )
            if not len(positive_rows):
                positive_rows = np.asarray([
                    row for row in identity_rows[(query_adduct, query_identity)]
                    if row != query_row
                ], dtype=np.int64)
            if len(positive_rows):
                positive_scores = embeddings[positive_rows] @ query_embedding
                order = np.argsort(positive_scores, kind="stable")
                positive_rows = positive_rows[order[:args.max_positive_candidates]]
                positive_scores = positive_scores[order[:args.max_positive_candidates]]
            else:
                positive_scores = np.empty(0, dtype=np.float32)

            negative_rows_all = molecule_rows[1:]
            negative_scores_all = molecule_scores[1:]
            hard_order = np.argsort(-negative_scores_all, kind="stable")
            hard_order = hard_order[:args.max_negative_candidates]
            negative_rows = negative_rows_all[hard_order]
            negative_scores = negative_scores_all[hard_order]
            if not len(negative_rows):
                raise RuntimeError("corrected graph query has no negative molecule")

            choice = None
            if len(positive_rows):
                query_spectrum = spectra[query_row]
                positive_wse = np.asarray([
                    weighted_entropy_similarity(
                        query_spectrum, spectra[int(row)], args.fragment_tolerance,
                    ) for row in positive_rows
                ], dtype=np.float64)
                negative_wse = np.asarray([
                    weighted_entropy_similarity(
                        query_spectrum, spectra[int(row)], args.fragment_tolerance,
                    ) for row in negative_rows
                ], dtype=np.float64)
                choice = choose_wse_supported_triplet(
                    positive_rows, positive_scores, positive_wse,
                    negative_rows, negative_scores, negative_wse,
                    native_margin=args.native_margin,
                )
            if choice is not None:
                append_event(
                    anchor=registry.add(Registry.HDF5, query_row),
                    positive=registry.add(Registry.HDF5, choice.positive_row),
                    negative=registry.add(Registry.HDF5, choice.negative_row),
                    query=query, kind=WSE_RELATION, action=-1,
                    v1_margin=choice.v1_margin, wse_margin=choice.wse_margin,
                )
                wse_misranked += int(choice.v1_margin <= 0)
                wse_boundary += int(choice.v1_margin > 0)
            elif query in actions_by_query:
                action, positive_row, negative_row = min(
                    actions_by_query[query], key=lambda body: body[0],
                )
                if (
                    str(identities[positive_row]) != query_identity
                    or str(identities[negative_row]) == query_identity
                ):
                    raise RuntimeError("Stage-1 replay identity boundary drifted")
                margin = float(
                    embeddings[query_row] @ embeddings[positive_row]
                    - embeddings[query_row] @ embeddings[negative_row]
                )
                append_event(
                    anchor=registry.add(Registry.ACTION, action),
                    positive=registry.add(Registry.HDF5, positive_row),
                    negative=registry.add(Registry.HDF5, negative_row),
                    query=query, kind=STAGE1_REPLAY, action=action,
                    v1_margin=margin, wse_margin=np.nan,
                )
                stage1_actions_selected += 1
            elif len(positive_rows):
                positive_row, negative_row, margin = choose_v1_preservation_triplet(
                    positive_rows, positive_scores,
                    negative_rows_all, negative_scores_all,
                )
                append_event(
                    anchor=registry.add(Registry.HDF5, query_row),
                    positive=registry.add(Registry.HDF5, positive_row),
                    negative=registry.add(Registry.HDF5, negative_row),
                    query=query, kind=MEASURED_PRESERVATION, action=-1,
                    v1_margin=margin, wse_margin=np.nan,
                )
            else:
                negative_row = int(negative_rows_all[np.argmin(negative_scores_all)])
                spectrum = make_hdf5_spectrum(
                    handle["spectrum"][query_row], float(precursor_mz[query_row]),
                )
                clean = preprocessor(
                    spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
                    high_form=False, augment=False,
                ).astype(np.float32, copy=False)
                view, short = make_noise_view(clean, "easy", query_row, args.seed)
                short_noise_views += int(short)
                noise_index = len(noise_spectra); noise_spectra.append(view)
                margin = float(1.0 - embeddings[query_row] @ embeddings[negative_row])
                append_event(
                    anchor=registry.add(Registry.NOISE, noise_index),
                    positive=registry.add(Registry.HDF5, query_row),
                    negative=registry.add(Registry.HDF5, negative_row),
                    query=query, kind=SINGLETON_NOISE_PRESERVATION, action=-1,
                    v1_margin=margin, wse_margin=np.nan,
                )
            if query and query % 10000 == 0:
                print(f"[wse-triplets] {query:,}/{graph.n_queries:,} queries", flush=True)

    event_query_array = np.asarray(event_query, dtype=np.int64)
    if not np.array_equal(event_query_array, np.arange(graph.n_queries, dtype=np.int64)):
        raise RuntimeError("WSE route did not emit exactly one event per graph query")
    kinds = np.asarray(event_kind, dtype=np.int8)
    counts = Counter(map(int, kinds))
    if sum(counts.values()) != graph.n_queries or set(counts) - set(EVENT_NAMES):
        raise RuntimeError("WSE event accounting drifted")
    pool = {
        "registry_kind": np.asarray(registry.kind, dtype=np.int8),
        "registry_source_index": np.asarray(registry.source, dtype=np.int64),
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "event_query": event_query_array,
        "event_kind": kinds,
        "event_action_index": np.asarray(event_action, dtype=np.int64),
    }
    noise_array = (
        np.asarray(noise_spectra, dtype=np.float32)
        if noise_spectra else np.empty((0, 101, 2), dtype=np.float32)
    )
    selection_ledger = {
        "query_index": event_query_array,
        "event_kind": kinds,
        "positive_registry_index": np.asarray(ledger_positive, dtype=np.int64),
        "negative_registry_index": np.asarray(ledger_negative, dtype=np.int64),
        "action_index": np.asarray(ledger_action, dtype=np.int64),
        "v1_margin": np.asarray(ledger_v1_margin, dtype=np.float32),
        "wse_margin": np.asarray(ledger_wse_margin, dtype=np.float32),
    }
    report = {
        "status": "NOISE_WSE_DISCORDANT_NATIVE_TRIPLETS_COMPLETE",
        "algorithm": {
            "unit": "every corrected MassSpecGym query exactly once",
            "primary_relation": "measured hard positive and actual candidate-graph hard negative",
            "selector": "WSE positive ordering plus active V1 native margin",
            "selector_role": "triplet membership only; WSE is absent from optimization",
            "fallback_order": [
                "one exact Stage-1 action replay",
                "largest-margin measured V1 preservation",
                "light singleton noise-to-clean preservation",
            ],
            "positive_sampling": "singleton after mining; native dynamic sampling cannot dilute the relation",
            "negative_sampling": "singleton candidate-molecule-max row after mining",
        },
        "coverage": {
            "massspecgym_spectra_reference_universe": int(n_rows),
            "corrected_graph_queries": int(graph.n_queries),
            "queries_represented": int(len(event_query_array)),
            "events": {EVENT_NAMES[key]: int(counts.get(key, 0)) for key in EVENT_NAMES},
            "wse_misranked_relations": int(wse_misranked),
            "wse_native_hinge_boundary_relations": int(wse_boundary),
            "stage1_actions_available_in_source_ledger": int(stage1_actions_available),
            "stage1_action_queries_available": int(action_report["queries_with_actions"]),
            "stage1_action_queries_replayed": int(stage1_actions_selected),
            "stage1_action_policy": "one representative only when no WSE-supported relation; V1 already contains full Stage-1 training",
            "singleton_short_spectrum_views": int(short_noise_views),
        },
        "native_contract": {
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "loss": "native cosine triplet-margin loss",
            "margin": 0.1,
            "optimizer": "native Adam",
            "learning_rate": 5e-6,
            "batch_size": 4,
            "precision": "FP32",
        },
        "independence": {
            "gnps_enters_triplet_mining": False,
            "gnps_enters_training": False,
            "final_judge": "frozen GNPS identity-disjoint and formula-disjoint panels",
            "wse_backend": backend,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "data_sha256": sha256_file(args.data),
            "embedding_shard_sha256": [sha256_file(path) for path in args.embedding_shard],
            "stage1_train_pool_sha256": sha256_file(args.stage1_triplets / "train_pool.npz"),
            "stage1_action_bank_sha256": sha256_file(args.stage1_triplets / "action_spectra.npz"),
        },
        "claim_limit": "Training-corpus construction only; no encoder gain is claimed before frozen GNPS evaluation.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        np.savez_compressed(staging / "noise_spectra.npz", spectra=noise_array)
        np.savez_compressed(staging / "selection_ledger.npz", **selection_ledger)
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build one native triplet library over every corrected MassSpecGym query.

The shared encoder, loss and optimizer are untouched.  This builder only
defines triplet content.  Every corrected-graph query receives three bounded
same-identity Noise anchors (easy/medium/hard).  Every proven representable
Stage-1 action remains an exact action-anchor triplet candidate.  Measured
positives and V1-mined real negatives are drawn from all MassSpecGym spectra.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from noise_dreams_native_spectrum import action_fragment_profile
from noise_final_core import CandidateGraph, sha256_file
from noise_massspecgym_full_triplet_core import (
    decode_array,
    diverse_by_similarity,
    nearest_different_identity_rows,
    ordered_unique,
    strict_ppm_rows,
)
from build_noise_dreams_native_scale_stage6 import identity_preserving_noise_view
from train_noise_dreams_native import make_hdf5_spectrum


SEVERITIES = ("easy", "medium", "hard")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embedding-shard", type=Path, action="append", required=True)
    parser.add_argument("--stage1-triplets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--max-measured-positives", type=int, default=6)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--fallback-neighbours", type=int, default=64)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


def load_complete_embeddings(paths: list[Path], rows: int) -> np.ndarray:
    output: np.ndarray | None = None
    seen = np.zeros(rows, dtype=bool)
    dimension = None
    for path in paths:
        body = load_npz(path)
        shard_rows = np.asarray(body["rows"], dtype=np.int64)
        shard_embeddings = np.asarray(body["embeddings"], dtype=np.float32)
        if (
            shard_embeddings.ndim != 2
            or len(shard_rows) != len(shard_embeddings)
            or np.any((shard_rows < 0) | (shard_rows >= rows))
            or np.any(seen[shard_rows])
        ):
            raise RuntimeError(f"invalid or overlapping embedding shard: {path}")
        if dimension is None:
            dimension = int(shard_embeddings.shape[1])
            output = np.empty((rows, dimension), dtype=np.float32)
        if shard_embeddings.shape[1] != dimension:
            raise RuntimeError("MassSpecGym embedding shard dimensions disagree")
        norms = np.linalg.norm(shard_embeddings, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise RuntimeError("MassSpecGym embedding shard is non-finite")
        output[shard_rows] = shard_embeddings / norms[:, None]
        seen[shard_rows] = True
    if output is None or not np.all(seen):
        raise RuntimeError(
            f"embedding shards cover {int(seen.sum())}/{rows} MassSpecGym rows"
        )
    return output


class Registry:
    HDF5 = 0
    NOISE = 1
    ACTION = 2

    def __init__(self) -> None:
        self.position: dict[tuple[int, int], int] = {}
        self.kind: list[int] = []
        self.source: list[int] = []

    def add(self, kind: int, source: int) -> int:
        key = (int(kind), int(source))
        if key not in self.position:
            self.position[key] = len(self.kind)
            self.kind.append(key[0])
            self.source.append(key[1])
        return self.position[key]


def fallback_short_spectrum_view(
    clean: np.ndarray, severity: str, source_key: int, seed: int,
) -> np.ndarray:
    """Intensity-only nuisance for the rare native tensor with <8 fragments."""
    scale = {"easy": 0.05, "medium": 0.10, "hard": 0.20}[severity]
    output = np.asarray(clean, dtype=np.float32).copy()
    fragment_positions = np.flatnonzero(
        (output[1:, 0] > 0) & (output[1:, 1] > 0)
    ) + 1
    if not len(fragment_positions):
        raise RuntimeError("MassSpecGym query has no positive native fragment")
    rng = np.random.default_rng(int(seed) ^ (int(source_key) * 0x9E3779B1) ^ len(severity))
    output[fragment_positions, 1] *= rng.uniform(
        1.0 - scale, 1.0 + scale, len(fragment_positions)
    )
    output[fragment_positions, 1] = np.maximum(output[fragment_positions, 1], 1e-5)
    output[fragment_positions, 1] /= float(output[fragment_positions, 1].max())
    return output


def make_noise_view(
    clean: np.ndarray, severity: str, source_key: int, seed: int,
) -> tuple[np.ndarray, bool]:
    fragments = int(np.sum((clean[1:, 0] > 0) & (clean[1:, 1] > 0)))
    if fragments >= 8:
        return identity_preserving_noise_view(
            clean, source_key=source_key, severity=severity, seed=seed,
        ), False
    return fallback_short_spectrum_view(clean, severity, source_key, seed), True


def stage1_exact_actions(
    stage1: Path,
) -> tuple[np.ndarray, dict[int, list[tuple[int, int, int]]], dict[str, int]]:
    pool = load_npz(stage1 / "train_pool.npz")
    bank = load_npz(stage1 / "action_spectra.npz")
    actions = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    representable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    registry_kind = np.asarray(pool["registry_kind"], dtype=np.int8)
    registry_source = np.asarray(pool["registry_source_index"], dtype=np.int64)
    event_kind = np.asarray(pool["event_kind"], dtype=np.int8)
    event_query = np.asarray(pool["event_query"], dtype=np.int64)
    event_action = np.asarray(pool["event_action_index"], dtype=np.int64)
    by_query: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    used: list[int] = []
    for event in np.flatnonzero(event_kind == 2):
        action = int(event_action[event])
        if action < 0 or action >= len(actions) or not representable[action]:
            raise RuntimeError("Stage-1 action event is not natively representable")
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        if p1 - p0 != 1 or n1 - n0 != 1:
            raise RuntimeError("Stage-1 exact action triplet is not singleton")
        p_registry = int(pool["positive_idx"][p0])
        n_registry = int(pool["negative_idx"][n0])
        if registry_kind[p_registry] != 0 or registry_kind[n_registry] != 0:
            raise RuntimeError("Stage-1 action boundary is not measured HDF5 data")
        by_query[int(event_query[event])].append((
            action, int(registry_source[p_registry]), int(registry_source[n_registry]),
        ))
        used.append(action)
    if len(used) != len(set(used)):
        raise RuntimeError("Stage-1 action entered the exact triplet ledger twice")
    return actions, by_query, {
        "available_action_spectra": int(len(actions)),
        "exact_representable_action_triplets": int(len(used)),
        "queries_with_actions": int(len(by_query)),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.max_measured_positives, args.max_negatives) < 1:
        raise ValueError("positive and negative caps must be positive")
    required = [
        args.graph, args.data, *args.embedding_shard,
        args.stage1_triplets / "train_pool.npz",
        args.stage1_triplets / "action_spectra.npz",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    graph = CandidateGraph(args.graph)
    with h5py.File(args.data, "r") as handle:
        n_rows = len(handle["spectrum"])
        identities = np.asarray(
            [value[:14] for value in decode_array(handle["INCHIKEY"][:])],
            dtype="U14",
        )
        adducts = decode_array(handle["adduct"][:])
        precursor_mz = np.asarray(handle["precursor_mz"][:], dtype=np.float64)
    if not (
        len(identities) == len(adducts) == len(precursor_mz) == n_rows
        and np.all(np.isfinite(precursor_mz))
    ):
        raise RuntimeError("MassSpecGym HDF5 metadata is malformed")
    if np.any((graph.query_row < 0) | (graph.query_row >= n_rows)):
        raise RuntimeError("corrected graph query row is outside MassSpecGym")
    if not np.array_equal(identities[graph.query_row], graph.query_ik14.astype(str)):
        raise RuntimeError("corrected graph query identities disagree with MassSpecGym")

    embeddings = load_complete_embeddings(args.embedding_shard, n_rows)
    actions, actions_by_query, action_report = stage1_exact_actions(args.stage1_triplets)
    for query in actions_by_query:
        if query < 0 or query >= graph.n_queries:
            raise RuntimeError("Stage-1 action query is outside corrected graph")

    identity_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    adduct_sorted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row, (identity, adduct) in enumerate(zip(identities, adducts, strict=True)):
        identity_rows[(str(adduct), str(identity))].append(row)
    for adduct in np.unique(adducts):
        rows = np.flatnonzero(adducts == adduct).astype(np.int64)
        order = np.argsort(precursor_mz[rows], kind="stable")
        adduct_sorted[str(adduct)] = (rows[order], precursor_mz[rows[order]])

    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    registry = Registry()
    noise_spectra: list[np.ndarray] = []
    noise_query: list[int] = []
    noise_severity: list[str] = []
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    event_query: list[int] = []
    event_kind: list[int] = []
    event_action: list[int] = []

    measured_positive_queries = 0
    strict_negative_queries = 0
    fallback_negative_queries = 0
    short_noise_views = 0
    measured_positive_memberships = 0
    generic_negative_memberships = 0

    def append_event(
        anchor: int, positive: list[int], negative: list[int], *,
        query: int, kind: int, action: int,
    ) -> None:
        if not positive or not negative:
            raise RuntimeError("full-MassSpecGym triplet has an empty role")
        if anchor in positive or anchor in negative or set(positive) & set(negative):
            raise RuntimeError("full-MassSpecGym triplet roles overlap")
        anchors.append(int(anchor))
        positives.extend(map(int, positive))
        negatives.extend(map(int, negative))
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        event_query.append(int(query))
        event_kind.append(int(kind))
        event_action.append(int(action))

    with h5py.File(args.data, "r") as handle:
        for query in range(graph.n_queries):
            query_row = int(graph.query_row[query])
            query_identity = str(identities[query_row])
            query_adduct = str(adducts[query_row])
            query_embedding = embeddings[query_row]

            same_identity = np.asarray(
                identity_rows[(query_adduct, query_identity)], dtype=np.int64,
            )
            distinct = same_identity[same_identity != query_row]
            if len(distinct):
                measured_positive_queries += 1
                chosen_positive = diverse_by_similarity(
                    distinct, embeddings[distinct] @ query_embedding,
                    args.max_measured_positives, hard_first=False,
                )
            else:
                chosen_positive = np.empty(0, dtype=np.int64)
            # The measured clean query is a legitimate positive for every
            # independently perturbed Noise anchor, including singleton identities.
            positive_rows = ordered_unique([query_row, *map(int, chosen_positive)])
            measured_positive_memberships += len(positive_rows)
            positive_registry = [registry.add(Registry.HDF5, row) for row in positive_rows]

            sorted_rows, sorted_mz = adduct_sorted[query_adduct]
            strict = strict_ppm_rows(
                sorted_rows, sorted_mz, float(precursor_mz[query_row]), args.ppm,
            )
            strict = strict[identities[strict] != query_identity]
            strict_negative_queries += int(len(strict) > 0)
            candidates = list(map(int, strict))
            if len(ordered_unique(candidates)) < args.max_negatives:
                fallback = nearest_different_identity_rows(
                    sorted_rows, sorted_mz, identities,
                    float(precursor_mz[query_row]), query_identity,
                    args.fallback_neighbours,
                )
                candidates.extend(map(int, fallback))
                fallback_negative_queries += int(len(strict) == 0)
            candidates_array = np.asarray(ordered_unique(candidates), dtype=np.int64)
            if not len(candidates_array):
                raise RuntimeError(f"query {query} has no real negative spectrum")
            chosen_negative = diverse_by_similarity(
                candidates_array, embeddings[candidates_array] @ query_embedding,
                args.max_negatives, hard_first=True,
            )
            negative_registry = [
                registry.add(Registry.HDF5, row) for row in chosen_negative
            ]
            generic_negative_memberships += len(negative_registry)

            spectrum = make_hdf5_spectrum(
                handle["spectrum"][query_row], float(precursor_mz[query_row]),
            )
            clean = preprocessor(
                spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
                high_form=False, augment=False,
            ).astype(np.float32, copy=False)
            for severity in SEVERITIES:
                view, short = make_noise_view(clean, severity, query_row, args.seed)
                short_noise_views += int(short)
                source = len(noise_spectra)
                noise_spectra.append(view)
                noise_query.append(query)
                noise_severity.append(severity)
                append_event(
                    registry.add(Registry.NOISE, source),
                    positive_registry,
                    negative_registry,
                    query=query, kind=0, action=-1,
                )

            for action, positive_row, negative_row in actions_by_query.get(query, []):
                if (
                    str(identities[positive_row]) != query_identity
                    or str(identities[negative_row]) == query_identity
                ):
                    raise RuntimeError("Stage-1 action exact identity boundary drifted")
                append_event(
                    registry.add(Registry.ACTION, action),
                    [registry.add(Registry.HDF5, positive_row)],
                    [registry.add(Registry.HDF5, negative_row)],
                    query=query, kind=1, action=action,
                )

            if query and query % 10000 == 0:
                print(f"[full-triplets] {query:,}/{graph.n_queries:,} queries", flush=True)

    event_query_array = np.asarray(event_query, dtype=np.int64)
    represented = np.unique(event_query_array)
    if not np.array_equal(represented, np.arange(graph.n_queries, dtype=np.int64)):
        raise RuntimeError("full-MassSpecGym triplet library lost a corrected-graph query")
    action_ids = np.asarray(event_action, dtype=np.int64)
    action_ids = action_ids[action_ids >= 0]
    expected_actions = sorted(
        action for values in actions_by_query.values() for action, _, _ in values
    )
    if not np.array_equal(np.sort(action_ids), np.asarray(expected_actions, dtype=np.int64)):
        raise RuntimeError("full-MassSpecGym library lost a Stage-1 action triplet")

    pool = {
        "registry_kind": np.asarray(registry.kind, dtype=np.int8),
        "registry_source_index": np.asarray(registry.source, dtype=np.int64),
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "event_query": event_query_array,
        "event_kind": np.asarray(event_kind, dtype=np.int8),
        "event_action_index": np.asarray(event_action, dtype=np.int64),
    }
    noise_archive = {
        "spectra": np.asarray(noise_spectra, dtype=np.float32),
        "query_index": np.asarray(noise_query, dtype=np.int64),
        "severity": np.asarray(noise_severity, dtype="U8"),
    }
    report = {
        "status": "NOISE_MASSSPECGYM_FULL_NATIVE_TRIPLETS_COMPLETE",
        "algorithm": {
            "unit": "one corrected candidate-graph query",
            "generic_anchor": "same-identity acquisition-noise view at easy/medium/hard severity",
            "action_anchor": "exact Stage-1 representable targeted action view",
            "positive": "same-identity measured MassSpecGym spectrum; clean query included for every noise anchor",
            "negative": "real different-identity same-adduct spectra, strict 10 ppm first, nearest-mass fallback only when needed",
            "difficulty": "V1 cosine hard-to-easy coverage; never hardest-only",
            "training_dose": "one rotating library event per query per epoch",
        },
        "coverage": {
            "massspecgym_spectra_reference_universe": int(n_rows),
            "corrected_graph_queries": int(graph.n_queries),
            "queries_represented": int(len(represented)),
            "generic_noise_events": int(np.sum(pool["event_kind"] == 0)),
            "easy_medium_hard_views_per_query": 3,
            "measured_distinct_positive_queries": measured_positive_queries,
            "singleton_positive_queries_carried_by_noise_to_clean_relation": int(
                graph.n_queries - measured_positive_queries
            ),
            "strict_10ppm_negative_queries": strict_negative_queries,
            "nearest_mass_fallback_queries": fallback_negative_queries,
            "measured_positive_memberships": measured_positive_memberships,
            "generic_negative_memberships": generic_negative_memberships,
            "short_spectrum_intensity_only_views": short_noise_views,
            **action_report,
            "actions_retained_in_library": int(len(action_ids)),
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
            "backbone_unfrozen_from_epoch": 0,
        },
        "split_contract": {
            "massspecgym_internal_holdout": False,
            "reason": "all MassSpecGym is training content; only frozen identity/formula-disjoint GNPS is the final judge",
            "gnps_enters_training": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "data_sha256": sha256_file(args.data),
            "embedding_shard_sha256": [sha256_file(path) for path in args.embedding_shard],
            "stage1_train_pool_sha256": sha256_file(args.stage1_triplets / "train_pool.npz"),
            "stage1_action_bank_sha256": sha256_file(args.stage1_triplets / "action_spectra.npz"),
        },
        "claim_limit": "Triplet construction only; no performance gain is claimed before frozen GNPS evaluation.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        np.savez_compressed(staging / "noise_spectra.npz", **noise_archive)
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Audit ChemAware training/evaluation lineage before any new optimization.

The audit answers three narrow questions with row-level evidence:

1. What fraction of historical pools belongs to the spectrum-simulation task?
2. How large is the strict-10-ppm, P3-disjoint training remainder?
3. Did any prior audit incorrectly treat ``simulation_challenge`` as provenance?

No model is trained and no score threshold is selected here.  The output is a
data-contract report; it is not a performance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--e1-pool",
        type=Path,
        default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz",
    )
    parser.add_argument(
        "--corrected-pool",
        action="append",
        type=Path,
        default=None,
        help="Corrected P3-disjoint control pool to audit. Repeat for multiple adducts.",
    )
    parser.add_argument(
        "--p3-allow",
        type=Path,
        default=(
            ROOT
            / "data/validation/g8r_p3_allow_recovered_corrected_v3_20260902"
            / "p3_p2_allowed_training_ik14.json"
        ),
    )
    parser.add_argument(
        "--graph",
        action="append",
        type=Path,
        default=None,
        help="Graph NPZ to audit. Repeat for multiple graphs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_training_lineage_audit_v3/report.json",
    )
    return parser.parse_args()


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def counts(values: np.ndarray) -> dict[str, int]:
    keys, values_n = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(value) for key, value in zip(keys, values_n)}


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"min": None, "median": None, "p90": None, "max": None, "mean": None}
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
    }


def load_allow(path: Path) -> set[str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = body["train_primary_all"]["ik14"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"unexpected P3 allow-list schema: {path}") from error
    return set(map(str, values))


def edge_source(
    anchors: np.ndarray,
    ptr: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Expand anchor rows for a contiguous edge slice."""
    left = int(np.searchsorted(ptr, start, side="right") - 1)
    right = int(np.searchsorted(ptr, stop - 1, side="right") - 1)
    anchor_positions = np.arange(left, right + 1, dtype=np.int64)
    expanded = np.repeat(anchors[anchor_positions], np.diff(ptr[left:right + 2]))
    offset = start - int(ptr[left])
    return expanded[offset:offset + (stop - start)]


def audit_edges(
    anchors: np.ndarray,
    ptr: np.ndarray,
    targets: np.ndarray,
    simulation_challenge: np.ndarray,
    allowed_row: np.ndarray,
    instrument: np.ndarray,
    collision: np.ndarray,
    chunk: int = 1_000_000,
) -> dict:
    type_counter: Counter[str] = Counter()
    allowed_edges = 0
    allowed_cross_instrument = 0
    allowed_distinct_collision = 0
    for start in range(0, len(targets), chunk):
        stop = min(start + chunk, len(targets))
        source = edge_source(anchors, ptr, start, stop)
        target = targets[start:stop]
        pair_types = np.char.add(
            np.char.add(simulation_challenge[source].astype(str), "->"),
            simulation_challenge[target].astype(str),
        )
        type_counter.update(map(str, pair_types))
        keep = allowed_row[source] & allowed_row[target]
        allowed_edges += int(np.sum(keep))
        if np.any(keep):
            source_keep, target_keep = source[keep], target[keep]
            known_instrument = (
                (instrument[source_keep] != "nan")
                & (instrument[target_keep] != "nan")
            )
            allowed_cross_instrument += int(np.sum(
                known_instrument
                & (instrument[source_keep] != instrument[target_keep])
            ))
            finite = np.isfinite(collision[source_keep]) & np.isfinite(collision[target_keep])
            allowed_distinct_collision += int(np.sum(
                finite & (np.abs(collision[source_keep] - collision[target_keep]) > 1e-9)
            ))
    return {
        "edges": int(len(targets)),
        "simulation_challenge_membership_transition_counts": dict(sorted(type_counter.items())),
        "p3_disjoint_edges": allowed_edges,
        "p3_disjoint_cross_known_instrument_edges": allowed_cross_instrument,
        "p3_disjoint_distinct_observed_collision_energy_edges": (
            allowed_distinct_collision
        ),
    }


def audit_pool(
    path: Path,
    simulation_challenge: np.ndarray,
    identity: np.ndarray,
    adduct: np.ndarray,
    fold: np.ndarray,
    instrument: np.ndarray,
    collision: np.ndarray,
    allow: set[str],
) -> dict:
    with np.load(path) as pool:
        anchors = pool["anchor_idx"].astype(np.int64)
        positive_ptr = pool["positive_ptr"].astype(np.int64)
        positive_idx = pool["positive_idx"].astype(np.int64)
        negative_ptr = pool["negative_ptr"].astype(np.int64)
        negative_idx = pool["negative_idx"].astype(np.int64)

    allowed_identity = np.isin(identity, np.asarray(sorted(allow), dtype=str))
    anchor_adducts = np.unique(adduct[anchors])
    if len(anchor_adducts) != 1:
        raise RuntimeError(f"pool mixes adducts: {path}: {anchor_adducts.tolist()}")
    pool_adduct = str(anchor_adducts[0])
    allowed_row = (
        (fold == "train")
        & (adduct == pool_adduct)
        & allowed_identity
    )
    positive_keep = allowed_row[positive_idx]
    negative_keep = allowed_row[negative_idx]
    positive_prefix = np.concatenate(([0], np.cumsum(positive_keep, dtype=np.int64)))
    negative_prefix = np.concatenate(([0], np.cumsum(negative_keep, dtype=np.int64)))
    positive_count = positive_prefix[positive_ptr[1:]] - positive_prefix[positive_ptr[:-1]]
    negative_count = negative_prefix[negative_ptr[1:]] - negative_prefix[negative_ptr[:-1]]
    anchor_keep = allowed_row[anchors]
    eligible = anchor_keep & (positive_count > 0) & (negative_count > 0)
    anchor_identity = identity[anchors]

    return {
        "path": str(path),
        "sha256": sha256(path),
        "historical_contract": {
            "fold": counts(fold[anchors]),
            "adduct": counts(adduct[anchors]),
            "anchor_simulation_challenge_membership_counts": counts(simulation_challenge[anchors]),
            "anchors": int(len(anchors)),
            "simulation_challenge_member_fraction": float(np.mean(simulation_challenge[anchors] == "True")),
            "unique_anchor_identities": int(len(set(map(str, anchor_identity)))),
        },
        "positive_edges": audit_edges(
            anchors, positive_ptr, positive_idx, simulation_challenge, allowed_row, instrument, collision
        ),
        "negative_edges": audit_edges(
            anchors, negative_ptr, negative_idx, simulation_challenge, allowed_row, instrument, collision
        ),
        "p3_disjoint_restriction": {
            "disallowed_anchors": int(np.sum(~anchor_keep)),
            "eligible_anchors": int(np.sum(eligible)),
            "unique_eligible_identities": int(len(set(map(str, anchor_identity[eligible])))),
            "positive_candidates_per_eligible_anchor": describe(positive_count[eligible]),
            "negative_candidates_per_eligible_anchor": describe(negative_count[eligible]),
            "identity_equal_weight_required": True,
            "source_provenance_available": False,
            "note": (
                "Exact restriction of the historical strict-10-ppm pool. The original "
                "peak-hash exclusion is retained, and identities outside the corrected "
                "P3-disjoint train allow-list are removed."
            ),
        },
    }


def audit_graph(path: Path, simulation_challenge: np.ndarray, fold: np.ndarray) -> dict:
    with np.load(path, allow_pickle=True) as graph:
        query = graph["query_row"].astype(np.int64)
        pair = graph["pair_candidate_row"].astype(np.int64)
    reachable = np.unique(np.concatenate((query, pair)))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "queries": int(len(query)),
        "query_simulation_challenge_membership_counts": counts(simulation_challenge[query]),
        "query_simulation_challenge_member_fraction": float(np.mean(simulation_challenge[query] == "True")),
        "pair_edges": int(len(pair)),
        "pair_edge_simulation_challenge_membership_counts": counts(simulation_challenge[pair]),
        "reachable_rows": int(len(reachable)),
        "reachable_row_simulation_challenge_membership_counts": counts(simulation_challenge[reachable]),
        "reachable_fold_counts": counts(fold[reachable]),
        "membership_is_not_spectrum_provenance": True,
    }


def main() -> None:
    args = arguments()
    graph_paths = args.graph or [
        ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz",
        ROOT / "data/validation/chemaware_shared_v2_cached_real_confirmation/graph.npz",
        ROOT / "data/validation/chemaware_shared_v2_local_real_pilot/graph.npz",
    ]
    corrected_pools = args.corrected_pool or [
        ROOT / "data/e1/chemaware_control_train_mh_triplet_pool_10ppm_p3disjoint_v3.npz",
        ROOT / "data/e1/chemaware_control_train_mna_triplet_pool_10ppm_p3disjoint_v3.npz",
    ]
    required = [args.hdf5, args.e1_pool, args.p3_allow, *graph_paths, *corrected_pools]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite lineage audit: {args.output}")
    allow = load_allow(args.p3_allow)
    with h5py.File(args.hdf5, "r") as handle:
        simulation_challenge = np.asarray(handle["SIMULATION_CHALLENGE"].asstr()[:], dtype=str)
        identity = np.asarray(handle["INCHIKEY"].asstr()[:], dtype=str)
        adduct = np.asarray(handle["adduct"].asstr()[:], dtype=str)
        fold = np.asarray(handle["fold"].asstr()[:], dtype=str)
        instrument = np.asarray(handle["INSTRUMENT_TYPE"].asstr()[:], dtype=str)
        collision = np.asarray(handle["COLLISION_ENERGY"][:], dtype=np.float64)

    train_allow = (
        (fold == "train")
        & np.isin(identity, np.asarray(sorted(allow), dtype=str))
    )
    report = {
        "status": "chemaware_training_lineage_audit_complete",
        "formal": False,
        "scope": "data lineage and supervision semantics; no model performance claim",
        "hdf5": {
            "path": str(args.hdf5),
            "sha256": sha256(args.hdf5),
            "rows": int(len(identity)),
            "simulation_challenge_membership_counts": counts(simulation_challenge),
            "simulation_challenge_semantics": "benchmark eligibility mask, not provenance",
            "p3_disjoint_train_rows": int(np.sum(train_allow)),
            "p3_disjoint_train_identities": int(len(set(identity[train_allow]))),
            "p3_disjoint_train_adduct_counts": counts(adduct[train_allow]),
        },
        "p3_allow": {
            "path": str(args.p3_allow),
            "sha256": sha256(args.p3_allow),
            "identities": int(len(allow)),
            "provenance_limit": (
                "Corrected local semantic allow-list; byte identity to the unavailable server "
                "seal is not claimed."
            ),
        },
        "historical_e1_pool": audit_pool(
            args.e1_pool, simulation_challenge, identity, adduct, fold, instrument, collision, allow
        ),
        "corrected_identity_continuation_control_pools": [
            audit_pool(
                path, simulation_challenge, identity, adduct, fold, instrument, collision, allow
            )
            for path in corrected_pools
        ],
        "local_graphs": [audit_graph(path, simulation_challenge, fold) for path in graph_paths],
        "decisions": [
            "SIMULATION_CHALLENGE must never again be used as an experimental-versus-synthetic provenance label.",
            "Historical E1 is not invalidated by membership=True rows; those rows are spectra eligible for the simulation benchmark, not simulated spectra.",
            "The next trainable bank must fail closed on fold=train, same adduct, strict 10 ppm, distinct peak hash, and P3-disjoint allowed identity.",
            "A corrected identity-only continuation is a control, not the ChemAware scientific contribution, because released DreaMS already used same-structure positives and 0.05-Da hard negatives from MoNA.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

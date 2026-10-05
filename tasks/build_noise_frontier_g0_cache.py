"""Build the frozen multiview readout cache for frontier experiment G0.

The cache augments the corrected 23,876-query candidate graph with four
candidate-independent observation channels:

* contextual peak-token late interaction (mass/loss/rule constrained),
* a leave-query-out candidate consensus in official embedding space,
* an A1-faithful raw-multichannel landmark response profile,
* matched negative controls that destroy token/consensus/landmark alignment.

No retrieval outcome is used to choose a peak, landmark, score, or weight.
Identity is used only for replicate grouping and to exclude self landmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_frontier_g0_core import (  # noqa: E402
    FrozenCandidateGraph,
    landmark_response_concordance,
    load_embedding_cache,
    pair_query_index,
    sha256_file,
)
from pilot_reference_anchored_multi_probe_a1 import CHANNELS, pair_features  # noqa: E402


TOKEN_FEATURES = (
    "g0_token_mass",
    "g0_token_rule_mass",
    "g0_token_unmasked",
    "g0_token_peak_permuted_control",
    "g0_mass_overlap_control",
)
ADDED_FEATURES = (
    *TOKEN_FEATURES,
    "g0_candidate_consensus",
    "g0_random_consensus_control",
    "g0_landmark_profile",
    "g0_random_landmark_control",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--rules", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/noise_frontier_g0_20260914/cache.npz")
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--top-peaks", type=int, default=16)
    parser.add_argument("--sketch-dim", type=int, default=128)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--abs-tolerance", type=float, default=0.01)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--raw-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ], dtype=str)


def ion_mode(adduct: str) -> str:
    value = str(adduct).strip()
    if value.endswith("+"):
        return "positive"
    if value.endswith("-"):
        return "negative"
    return "unknown"


def stable_key(seed: int, *values: object) -> bytes:
    return hashlib.sha256((str(seed) + "|" + "|".join(map(str, values))).encode("utf-8")).digest()


def load_rule_masses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    body = json.loads(path.read_text(encoding="utf-8"))
    neutral = [
        float(rule["value"]) for rule in body["rules"]
        if rule.get("category") == "NL" and rule.get("match_type") == "mass_diff"
    ]
    fragment = [
        float(rule["value"]) for rule in body["rules"]
        if rule.get("category") == "CF" and rule.get("match_type") == "peak_mz"
    ]
    if not neutral or not fragment:
        raise RuntimeError("chemical rule library contains no usable NL/CF masses")
    return np.sort(np.asarray(neutral, dtype=np.float32)), np.sort(np.asarray(fragment, dtype=np.float32))


def rule_hits(
    mz: np.ndarray,
    precursor: float,
    valid: np.ndarray,
    neutral: np.ndarray,
    fragment: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    output = np.zeros((len(mz), len(neutral) + len(fragment)), dtype=bool)
    if np.any(valid):
        loss = precursor - mz[valid]
        output[valid, :len(neutral)] = np.abs(loss[:, None] - neutral[None, :]) <= tolerance
        output[valid, len(neutral):] = np.abs(mz[valid, None] - fragment[None, :]) <= tolerance
    return output


def rule_bitsets(
    mz: np.ndarray,
    precursor: float,
    valid: np.ndarray,
    neutral: np.ndarray,
    fragment: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Compact rule memberships without a peak x 3,000-rule dense matrix."""

    rule_count = len(neutral) + len(fragment)
    words = (rule_count + 63) // 64
    output = np.zeros((len(mz), words), dtype=np.uint64)
    for peak in np.flatnonzero(valid):
        loss = float(precursor - mz[peak])
        for offset, values, target in (
            (0, neutral, loss),
            (len(neutral), fragment, float(mz[peak])),
        ):
            left = int(np.searchsorted(values, target - tolerance, side="left"))
            right = int(np.searchsorted(values, target + tolerance, side="right"))
            for rule in range(left, right):
                absolute = offset + rule
                output[peak, absolute // 64] |= np.uint64(1) << np.uint64(absolute % 64)
    return output


def symmetric(mask: np.ndarray, similarity: np.ndarray, qweight: np.ndarray, rweight: np.ndarray) -> np.ndarray:
    values = np.where(mask, np.maximum(similarity, 0.0), 0.0)
    query_side = np.sum(np.max(values, axis=2) * qweight[None, :], axis=1)
    reference_side = np.sum(np.max(values, axis=1) * rweight, axis=1)
    return 0.5 * (query_side + reference_side)


class PeakStore:
    def __init__(
        self,
        token_dir: Path,
        rows: np.ndarray,
        precursor: np.ndarray,
        top_peaks: int,
        sketch_dim: int,
        seed: int,
    ) -> None:
        if 1024 % sketch_dim:
            raise ValueError("sketch dimension must divide 1024")
        self.top = int(top_peaks)
        self.dim = int(sketch_dim)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.tokens = np.load(token_dir / "tokens_f16.npy", mmap_mode="r")
        self.mz = np.load(token_dir / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(token_dir / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(token_dir / "valid.npy", mmap_mode="r")
        self.precursor = np.asarray(precursor, dtype=np.float32)
        rng = np.random.default_rng(seed + 411)
        self.permutation = rng.permutation(1024)
        self.sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=1024)
        self.group = 1024 // self.dim
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}

    def get(self, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        position = self.position[row]
        available = np.flatnonzero(self.valid[position])
        order = available[
            np.argsort(-np.asarray(self.intensity[position, available]), kind="stable")[:self.top]
        ]
        count = len(order)
        token = np.zeros((self.top, self.dim), dtype=np.float32)
        mz = np.zeros(self.top, dtype=np.float32)
        weight = np.zeros(self.top, dtype=np.float32)
        mask = np.zeros(self.top, dtype=bool)
        if count:
            raw = np.asarray(self.tokens[position, order], dtype=np.float32)
            raw = raw[:, self.permutation] * self.sign
            raw = raw.reshape(count, self.dim, self.group).sum(axis=2) / np.sqrt(self.group)
            raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
            token[:count] = raw
            mz[:count] = self.mz[position, order]
            values = np.maximum(np.asarray(self.intensity[position, order], dtype=np.float32), 0.0)
            weight[:count] = values / max(float(np.sum(values)), 1e-8)
            mask[:count] = True
        result = (token, mz, weight, mask, float(self.precursor[position]))
        self.cache[row] = result
        return result


def token_reference_scores(
    store: PeakStore,
    query_row: int,
    reference_rows: np.ndarray,
    neutral_rules: np.ndarray,
    fragment_rules: np.ndarray,
    ppm: float,
    abs_tolerance: float,
    rule_tolerance: float,
) -> np.ndarray:
    qz, qmz, qw, qvalid, qprecursor = store.get(query_row)
    n = len(reference_rows)
    rz = np.empty((n, store.top, store.dim), dtype=np.float32)
    rmz = np.empty((n, store.top), dtype=np.float32)
    rw = np.empty((n, store.top), dtype=np.float32)
    rvalid = np.empty((n, store.top), dtype=bool)
    rprecursor = np.empty(n, dtype=np.float32)
    rpermuted = np.empty_like(rz)
    for index, row in enumerate(map(int, reference_rows)):
        token, mz, weight, valid, precursor = store.get(row)
        rz[index] = token
        rmz[index] = mz
        rw[index] = weight
        rvalid[index] = valid
        rprecursor[index] = precursor
        rpermuted[index] = token
        count = int(np.sum(valid))
        if count > 1:
            shift = 1 + (int(row) % (count - 1))
            rpermuted[index, :count] = np.roll(token[:count], shift, axis=0)
    similarity = np.einsum("kd,njd->nkj", qz, rz, optimize=True)
    permuted_similarity = np.einsum("kd,njd->nkj", qz, rpermuted, optimize=True)
    valid_pair = qvalid[None, :, None] & rvalid[:, None, :]
    delta = np.abs(qmz[None, :, None] - rmz[:, None, :])
    tolerance = np.maximum(
        abs_tolerance,
        ppm * np.maximum(np.abs(qmz[None, :, None]), np.abs(rmz[:, None, :])) / 1e6,
    )
    direct = valid_pair & (delta <= tolerance)
    query_loss = qprecursor - qmz
    reference_loss = rprecursor[:, None] - rmz
    loss_delta = np.abs(query_loss[None, :, None] - reference_loss[:, None, :])
    loss_tolerance = np.maximum(
        abs_tolerance,
        ppm * np.maximum(np.abs(query_loss[None, :, None]), np.abs(reference_loss[:, None, :])) / 1e6,
    )
    mass_mask = direct | (valid_pair & (loss_delta <= loss_tolerance))
    query_rule = rule_bitsets(
        qmz, qprecursor, qvalid, neutral_rules, fragment_rules, rule_tolerance,
    )
    reference_rule = np.empty((n, store.top, query_rule.shape[1]), dtype=np.uint64)
    for index in range(n):
        reference_rule[index] = rule_bitsets(
            rmz[index], float(rprecursor[index]), rvalid[index],
            neutral_rules, fragment_rules, rule_tolerance,
        )
    rule_mask = np.any(
        np.bitwise_and(
            query_rule[None, :, None, :],
            reference_rule[:, None, :, :],
        ),
        axis=-1,
    )
    rule_mass = mass_mask | (valid_pair & rule_mask)
    return np.column_stack((
        symmetric(mass_mask, similarity, qw, rw),
        symmetric(rule_mass, similarity, qw, rw),
        symmetric(valid_pair, similarity, qw, rw),
        symmetric(rule_mass, permuted_similarity, qw, rw),
        symmetric(mass_mask, np.ones_like(similarity), qw, rw),
    )).astype(np.float32)


def choose_landmarks(
    rows: np.ndarray,
    identities: np.ndarray,
    spectra: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    by_identity: dict[str, list[int]] = {}
    for position, identity in enumerate(identities):
        by_identity.setdefault(str(identity), []).append(position)
    ordered = sorted(by_identity, key=lambda value: stable_key(seed, value))
    selected = []
    for identity in ordered[:count]:
        candidates = by_identity[identity]
        best = max(
            candidates,
            key=lambda position: (
                int(np.sum((spectra[position, 0] > 0) & (spectra[position, 1] > 0))),
                -int(rows[position]),
            ),
        )
        selected.append(best)
    if len(selected) < min(count, 32):
        raise RuntimeError("too few unique identities for landmark panel")
    return np.asarray(selected, dtype=np.int64)


def build_landmark_responses(
    spectra: np.ndarray,
    precursor: np.ndarray,
    landmark_position: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    responses = np.empty(
        (len(spectra), len(landmark_position), len(CHANNELS)), dtype=np.float16,
    )
    for node in range(len(spectra)):
        for landmark_index, landmark in enumerate(landmark_position):
            values = pair_features(
                spectra[node], float(precursor[node]),
                spectra[int(landmark)], float(precursor[int(landmark)]), tolerance,
            )
            responses[node, landmark_index] = [values[name] for name in CHANNELS]
        if (node + 1) % 256 == 0 or node + 1 == len(spectra):
            print(f"[G0 landmark] {node + 1:,}/{len(spectra):,} spectra", flush=True)
    return responses


def main() -> None:
    args = arguments()
    started = time.time()
    required = [
        args.graph,
        args.embeddings,
        args.data,
        args.rules,
        args.token_dir / "report.json",
        args.token_dir / "rows.npy",
        args.token_dir / "tokens_f16.npy",
        args.token_dir / "mz_f32.npy",
        args.token_dir / "intensity_f32.npy",
        args.token_dir / "valid.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"G0 cache inputs missing: {missing}")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise RuntimeError(f"refusing to overwrite G0 cache: {args.output}")
    if args.landmarks < 32 or args.top_peaks < 4 or args.sketch_dim < 32:
        raise ValueError("G0 capacity parameters are below the preregistered minimum")

    graph = FrozenCandidateGraph(args.graph)
    if graph.n_queries != 23876:
        raise RuntimeError(f"formal G0 requires 23,876 graph queries, observed {graph.n_queries:,}")
    reachable = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if (
        token_report.get("status") != "noise_final_f1_full_token_cache_complete"
        or token_report.get("formal") is not True
        or token_report.get("provenance", {}).get("graph_sha256") != sha256_file(args.graph)
    ):
        raise RuntimeError("G0 requires the formal F1 token cache for this exact graph")
    token_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    if not np.array_equal(token_rows, reachable):
        raise RuntimeError("F1 token rows do not exactly match graph-reachable spectra")

    embedding_rows, embeddings, embedding_position = load_embedding_cache(args.embeddings)
    missing_embedding_rows = np.setdiff1d(reachable, embedding_rows)
    if len(missing_embedding_rows):
        raise RuntimeError(
            f"official embedding cache misses {len(missing_embedding_rows)} graph rows"
        )
    extra_embedding_rows = int(len(embedding_rows) - len(reachable))
    node_embedding_position = np.asarray([embedding_position[int(row)] for row in reachable], dtype=np.int64)
    node_embedding = embeddings[node_embedding_position]
    row_position = {int(row): index for index, row in enumerate(reachable)}

    print(f"[G0] loading {len(reachable):,} reachable spectra and metadata", flush=True)
    with h5py.File(args.data, "r") as handle:
        spectra = np.asarray(handle["spectrum"][reachable], dtype=np.float32)
        precursor = np.asarray(handle["precursor_mz"][reachable], dtype=np.float32)
        identities = decode(handle["INCHIKEY"][reachable])
        identities = np.asarray([value[:14] for value in identities], dtype=str)
        adduct = decode(handle["adduct"][reachable])
        instrument = decode(handle["INSTRUMENT_TYPE"][reachable])
    if not np.all(np.isfinite(precursor)) or np.any(precursor <= 0):
        raise RuntimeError("reachable spectra contain invalid precursor masses")

    pair_query = pair_query_index(graph.query_ptr, graph.molecule_ptr)
    pair_query_node = np.asarray(
        [row_position[int(row)] for row in graph.query_row[pair_query]], dtype=np.int64,
    )
    pair_reference_node = np.asarray(
        [row_position[int(row)] for row in graph.pair_candidate_row], dtype=np.int64,
    )
    observed = np.einsum(
        "ij,ij->i", node_embedding[pair_query_node], node_embedding[pair_reference_node],
        optimize=True,
    )
    score_error = float(np.max(np.abs(observed - graph.features[:, graph.dreams_column])))
    if score_error > 5e-4:
        raise RuntimeError(f"embedding cache/graph score mismatch: max={score_error:.6g}")

    neutral_rules, fragment_rules = load_rule_masses(args.rules)
    store = PeakStore(
        args.token_dir, reachable, precursor, args.top_peaks, args.sketch_dim, args.seed,
    )
    token_feature = np.empty((len(graph.features), len(TOKEN_FEATURES)), dtype=np.float32)
    for query in range(graph.n_queries):
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        token_feature[pair_left:pair_right] = token_reference_scores(
            store,
            int(graph.query_row[query]),
            graph.pair_candidate_row[pair_left:pair_right],
            neutral_rules,
            fragment_rules,
            args.ppm,
            args.abs_tolerance,
            args.rule_tolerance,
        )
        if (query + 1) % 500 == 0 or query + 1 == graph.n_queries:
            print(
                f"[G0 token] {query + 1:,}/{graph.n_queries:,} queries; {pair_right:,} pairs",
                flush=True,
            )

    consensus = np.empty(len(graph.features), dtype=np.float32)
    random_consensus = np.empty(len(graph.features), dtype=np.float32)
    for query in range(graph.n_queries):
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        query_node = row_position[int(graph.query_row[query])]
        molecule_values = []
        for molecule in range(molecule_left, molecule_right):
            left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
            positions = pair_reference_node[left:right]
            centre = np.mean(node_embedding[positions], axis=0)
            centre /= max(float(np.linalg.norm(centre)), 1e-12)
            molecule_values.append(float(node_embedding[query_node] @ centre))
            consensus[left:right] = molecule_values[-1]
        shift = 1 + int.from_bytes(stable_key(args.seed + 29, query)[:2], "little") % (len(molecule_values) - 1)
        control_values = np.roll(np.asarray(molecule_values, dtype=np.float32), shift)
        for local, molecule in enumerate(range(molecule_left, molecule_right)):
            left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
            random_consensus[left:right] = control_values[local]

    landmark_position = choose_landmarks(
        reachable, identities, spectra, args.landmarks, args.seed + 101,
    )
    landmark_rows = reachable[landmark_position]
    landmark_identity = identities[landmark_position]
    responses = build_landmark_responses(
        spectra, precursor, landmark_position, args.raw_tolerance,
    )
    landmark_score = np.empty(len(graph.features), dtype=np.float32)
    random_landmark_score = np.empty(len(graph.features), dtype=np.float32)
    rng = np.random.default_rng(args.seed + 307)
    coordinate_permutation = rng.permutation(len(landmark_position))
    for pair in range(len(graph.features)):
        query_node = int(pair_query_node[pair])
        reference_node = int(pair_reference_node[pair])
        valid = (landmark_identity != identities[query_node]) & (landmark_identity != identities[reference_node])
        landmark_score[pair] = landmark_response_concordance(
            responses[query_node], responses[reference_node], valid,
        )
        control_valid = valid & (
            (landmark_identity[coordinate_permutation] != identities[query_node])
            & (landmark_identity[coordinate_permutation] != identities[reference_node])
        )
        random_landmark_score[pair] = landmark_response_concordance(
            responses[query_node], responses[reference_node][coordinate_permutation], control_valid,
        )
        if (pair + 1) % 100000 == 0 or pair + 1 == len(graph.features):
            print(f"[G0 profile] {pair + 1:,}/{len(graph.features):,} pairs", flush=True)

    added = np.column_stack((
        token_feature,
        consensus,
        random_consensus,
        landmark_score,
        random_landmark_score,
    )).astype(np.float32)
    if added.shape != (len(graph.features), len(ADDED_FEATURES)) or not np.all(np.isfinite(added)):
        raise RuntimeError("G0 added feature matrix is malformed")

    with np.load(args.graph, allow_pickle=True) as source:
        payload = {name: source[name] for name in source.files}
    payload["features"] = np.concatenate((graph.features, added), axis=1).astype(np.float32)
    payload["feature_names"] = np.asarray([*graph.feature_names, *ADDED_FEATURES], dtype=object)
    query_node = np.asarray([row_position[int(row)] for row in graph.query_row], dtype=np.int64)
    payload["query_instrument"] = instrument[query_node].astype(object)
    payload["query_adduct"] = adduct[query_node].astype(object)
    payload["query_ion_mode"] = np.asarray([ion_mode(value) for value in adduct[query_node]], dtype=object)
    payload["g0_landmark_rows"] = landmark_rows
    payload["g0_landmark_ik14"] = landmark_identity.astype(object)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_frontier_g0_", dir=args.output.parent))
    try:
        temporary = staging / args.output.name
        np.savez_compressed(temporary, **payload)
        temporary.replace(args.output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    report = {
        "status": "noise_frontier_g0_cache_complete",
        "formal": True,
        "queries": int(graph.n_queries),
        "molecules": int(len(graph.molecule_label)),
        "spectrum_pairs": int(len(graph.features)),
        "reachable_spectra": int(len(reachable)),
        "landmarks": int(len(landmark_rows)),
        "added_features": list(ADDED_FEATURES),
        "official_graph_max_abs_error": score_error,
        "embedding_cache_extra_rows_ignored": extra_embedding_rows,
        "feature_standard_deviation": {
            name: float(value) for name, value in zip(ADDED_FEATURES, np.std(added.astype(np.float64), axis=0))
        },
        "contracts": {
            "retrieval_outcomes_used_for_feature_construction": False,
            "identity_use": "replicate grouping and self-landmark exclusion only",
            "same_pair_fusion_before_molecule_max": True,
            "token_control": "candidate peak-token rows cyclically displaced from peak masses",
            "consensus_control": "candidate consensus values cyclically permuted within query",
            "landmark_control": "candidate response coordinates globally permuted",
            "DreaMS_parameters_updated": False,
            "P2b_used": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "embedding_cache_sha256": sha256_file(args.embeddings),
            "hdf5_sha256": sha256_file(args.data),
            "rules_sha256": sha256_file(args.rules),
            "cache_sha256": sha256_file(args.output),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "parameters": {
            "landmarks": args.landmarks,
            "top_peaks": args.top_peaks,
            "sketch_dim": args.sketch_dim,
            "ppm": args.ppm,
            "abs_tolerance": args.abs_tolerance,
            "rule_tolerance": args.rule_tolerance,
            "raw_tolerance": args.raw_tolerance,
            "seed": args.seed,
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Frozen readout cache only; no shared-embedding improvement has been tested.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Causal peak-occlusion audit for large residual pair mechanisms.

For overaggregated different-identity pairs, matched/shared peaks are deleted
from one spectrum and the pair cosine should decrease more than after matched
random deletion. For unstable same-identity pairs, unmatched/condition-specific
peaks are deleted and pair cosine should increase more than after matched
random deletion. Both directions of every pair are evaluated with the official
DreaMS backbone frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

import pilot_multilevel_factor_activations as multi
from audit_e0_observability_residual import greedy_matches, peaks
from e1_checkpoint_io import official_head_state
from pilot_paired_layer_cka import preprocess_spectrum


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "data/validation/dreams_residual_peak_mechanisms_large"
    / "discovery_peak_mechanisms.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv"
)
DEFAULT_EMBEDDINGS = (
    ROOT / "data/validation/large_observability_embeddings_discovery/official_embeddings.npy"
)
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/dreams_residual_pair_occlusion"

SUPPORTED = {
    "shared_major_peak_overaggregation_candidate": "shared",
    "cross_instrument_identity_instability": "unique",
    "large_ce_shift_identity_instability": "unique",
    "fragmentation_divergence_identity_instability": "unique",
    "unresolved_identity_instability": "unique",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--token-match-tolerance", type=float, default=0.005)
    parser.add_argument("--max-target-peaks", type=int, default=12)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-pairs-per-mechanism", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def matched_and_unique_mz(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_mz, _ = peaks(source)
    target_mz, _ = peaks(target)
    matches = greedy_matches(source_mz, target_mz, tolerance)
    shared = np.asarray([source_mz[i] for i, _ in matches], dtype=float)
    matched_source = {i for i, _ in matches}
    unique = np.asarray(
        [source_mz[i] for i in range(len(source_mz)) if i not in matched_source], dtype=float
    )
    return shared, unique


def target_tokens(clean: torch.Tensor, target_mz: np.ndarray, tolerance: float) -> np.ndarray:
    values = clean.numpy()
    eligible = np.flatnonzero(
        (np.arange(len(values)) > 0) & (values[:, 0] > 0)
        & (values[:, 1] > 0) & (values[:, 1] < 1.0 - 1e-7)
    )
    chosen, used = [], set()
    for mz in target_mz:
        candidates = [
            idx for idx in eligible
            if idx not in used and abs(float(values[idx, 0]) - float(mz)) <= tolerance
        ]
        if candidates:
            idx = min(candidates, key=lambda item: abs(float(values[item, 0]) - float(mz)))
            chosen.append(idx)
            used.add(idx)
    return np.asarray(chosen, dtype=np.int64)


def matched_random_tokens(
    clean: torch.Tensor,
    targeted: np.ndarray,
    excluded: set[int],
    seed: int,
) -> np.ndarray:
    values = clean.numpy()
    pool = np.asarray([
        idx for idx in range(1, len(values))
        if values[idx, 0] > 0 and values[idx, 1] > 0 and idx not in excluded
    ], dtype=np.int64)
    if not len(targeted) or len(pool) < len(targeted):
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    valid_mz = values[1:, 0][values[1:, 0] > 0]
    mz_scale = max(float(np.std(valid_mz)), 25.0)
    log_intensity = np.log10(np.clip(values[:, 1].astype(float), 1e-6, None))
    cost = np.empty((len(targeted), len(pool)), dtype=float)
    for row, target in enumerate(targeted):
        log_diff = np.abs(log_intensity[pool] - log_intensity[target])
        linear_diff = np.abs(values[pool, 1].astype(float) - float(values[target, 1]))
        mz_diff = np.abs(values[pool, 0].astype(float) - float(values[target, 0])) / mz_scale
        cost[row] = 4.0 * log_diff + 8.0 * linear_diff + 0.15 * mz_diff
    cost += rng.gumbel(0.0, 0.015, size=cost.shape)
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(targeted):
        return np.empty(0, dtype=np.int64)
    return pool[columns].astype(np.int64)


def select_target_subset(
    clean: torch.Tensor,
    all_targeted: np.ndarray,
    max_target_peaks: int,
) -> tuple[np.ndarray, set[int]]:
    """Keep the most intense target peaks while preserving a disjoint control pool."""
    values = clean.numpy()
    excluded = set(all_targeted.tolist())
    control_capacity = sum(
        1 for idx in range(1, len(values))
        if values[idx, 0] > 0 and values[idx, 1] > 0 and idx not in excluded
    )
    capacity = min(len(all_targeted), max_target_peaks, control_capacity)
    if capacity <= 0:
        return np.empty(0, dtype=np.int64), excluded
    order = np.argsort(values[all_targeted, 1].astype(float))[::-1]
    return all_targeted[order[:capacity]].astype(np.int64), excluded


def perturb(clean: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    result = clean.clone()
    result[indices] = 0.0
    return result


def encode(
    model,
    weight: torch.Tensor,
    bias: torch.Tensor,
    tensors: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.stack(tensors)), batch_size=batch_size, shuffle=False)
    dtype = next(model.parameters()).dtype
    output = []
    with torch.inference_mode():
        for (batch,) in loader:
            precursor = model(batch.to(device=device, dtype=dtype), None)[:, 0]
            vector = F.normalize(F.linear(precursor, weight, bias), dim=-1)
            output.append(vector.float().cpu().numpy())
    return np.concatenate(output)


def read_spectra(handle: h5py.File, rows: np.ndarray) -> dict[int, np.ndarray]:
    unique = np.unique(rows.astype(np.int64))
    loaded = np.asarray(handle["spectrum"][unique])
    return {int(row): loaded[pos] for pos, row in enumerate(unique)}


def choose_pairs(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    selected = []
    for position, mechanism in enumerate(SUPPORTED):
        group = frame.loc[frame["mechanism_screen"] == mechanism].copy()
        if mechanism == "shared_major_peak_overaggregation_candidate":
            group = group.loc[group["pair_type"] == "different_identity"]
        elif mechanism.endswith("identity_instability"):
            group = group.loc[group["pair_type"] == "same_identity"]
        if cap > 0 and len(group) > cap:
            group = group.sample(n=cap, random_state=seed + position)
        selected.append(group)
    return pd.concat(selected, ignore_index=True) if selected else frame.iloc[:0].copy()


def cluster_bootstrap(values: pd.DataFrame, column: str, iterations: int, seed: int) -> list[float] | None:
    grouped = values.groupby("pair_key", sort=False)[column].mean().to_numpy(float)
    if not len(grouped) or iterations <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        draws[idx] = rng.choice(grouped, len(grouped), replace=True).mean()
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def summarize(paired: pd.DataFrame, bootstrap: int, seed: int) -> dict[str, object]:
    output = {}
    for position, (mechanism, group) in enumerate(paired.groupby("mechanism_screen", sort=True)):
        output[str(mechanism)] = {
            "directed_pairs": int(len(group)),
            "unique_molecule_pairs": int(group["pair_key"].nunique()),
            "mean_target_minus_random_cosine_change": float(group["target_minus_random_cosine_change"].mean()),
            "mean_directional_support": float(group["directional_support"].mean()),
            "median_directional_support": float(group["directional_support"].median()),
            "fraction_directionally_supportive": float((group["directional_support"] > 0).mean()),
            "pair_cluster_bootstrap_95ci": cluster_bootstrap(
                group, "directional_support", bootstrap, seed + position
            ),
            "median_removed_peaks": float(group["removed_count"].median()),
        }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    source = pd.read_csv(args.input)
    selected = choose_pairs(source, args.max_pairs_per_mechanism, args.seed)
    manifest = pd.read_csv(args.manifest)
    library = np.load(args.embeddings).astype(np.float32)
    required_rows = np.concatenate([
        selected["hdf5_row_a"].to_numpy(np.int64),
        selected["hdf5_row_b"].to_numpy(np.int64),
    ])
    with h5py.File(args.data, "r") as handle:
        spectra = read_spectra(handle, required_rows)

    tensors: list[torch.Tensor] = []
    metadata: list[dict[str, object]] = []
    for pair_position, row in enumerate(selected.itertuples(index=False)):
        pair_key = "|".join(sorted((str(row.ik_a), str(row.ik_b))))
        mode = SUPPORTED[str(row.mechanism_screen)]
        for side, source_index, target_index, source_hdf, target_hdf in (
            ("a_to_b", int(row.row_a), int(row.row_b), int(row.hdf5_row_a), int(row.hdf5_row_b)),
            ("b_to_a", int(row.row_b), int(row.row_a), int(row.hdf5_row_b), int(row.hdf5_row_a)),
        ):
            raw_source = spectra[source_hdf]
            raw_target = spectra[target_hdf]
            shared_mz, unique_mz = matched_and_unique_mz(
                raw_source, raw_target, args.fragment_tolerance
            )
            target_mz = shared_mz if mode == "shared" else unique_mz
            clean = preprocess_spectrum(
                raw_source, float(manifest.at[source_index, "precursor_mz"]), args.n_highest_peaks
            )
            all_targeted = target_tokens(clean, target_mz, args.token_match_tolerance)
            targeted, excluded = select_target_subset(
                clean, all_targeted, args.max_target_peaks
            )
            if not len(targeted):
                continue
            base = {
                "pair_position": int(pair_position),
                "pair_key": pair_key,
                "ik_a": row.ik_a,
                "ik_b": row.ik_b,
                "pair_type": row.pair_type,
                "mechanism_screen": row.mechanism_screen,
                "source_side": side,
                "source_index": source_index,
                "target_index": target_index,
                "removed_count": int(len(targeted)),
                "target_class_peak_count": int(len(all_targeted)),
                "target_mode": mode,
            }
            tensors.append(perturb(clean, targeted))
            metadata.append(base | {"condition": "targeted", "repeat": -1})
            for repeat in range(args.random_repeats):
                random_indices = matched_random_tokens(
                    clean, targeted, excluded,
                    stable_seed(args.seed, pair_key, side, repeat),
                )
                if len(random_indices) != len(targeted):
                    continue
                tensors.append(perturb(clean, random_indices))
                metadata.append(base | {"condition": "matched_random", "repeat": repeat})

    if not tensors:
        raise RuntimeError("No perturbations survived preprocessing")
    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    encoded = encode(model, weight, bias, tensors, args.batch_size, device)
    del model, raw_package, official_package, tensors
    gc.collect()

    rows = []
    for vector, item in zip(encoded, metadata):
        source_index = int(item["source_index"])
        target_index = int(item["target_index"])
        clean_pair_cosine = float(library[source_index] @ library[target_index])
        perturbed_pair_cosine = float(vector @ library[target_index])
        rows.append(item | {
            "clean_pair_cosine": clean_pair_cosine,
            "perturbed_pair_cosine": perturbed_pair_cosine,
            "cosine_change": perturbed_pair_cosine - clean_pair_cosine,
            "embedding_cosine_to_clean_source": float(vector @ library[source_index]),
        })
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "perturbation_results.csv", index=False)

    targeted = results.loc[results["condition"] == "targeted"].copy()
    random = results.loc[results["condition"] == "matched_random"].groupby(
        ["pair_position", "pair_key", "source_side", "mechanism_screen"], as_index=False
    ).agg(
        random_cosine_change=("cosine_change", "mean"),
        random_embedding_cosine_to_clean=("embedding_cosine_to_clean_source", "mean"),
        random_repeats_observed=("repeat", "count"),
    )
    paired = targeted.merge(
        random,
        on=["pair_position", "pair_key", "source_side", "mechanism_screen"],
        how="inner",
    )
    paired["target_minus_random_cosine_change"] = (
        paired["cosine_change"] - paired["random_cosine_change"]
    )
    is_overaggregation = paired["mechanism_screen"].eq(
        "shared_major_peak_overaggregation_candidate"
    )
    paired["directional_support"] = paired["target_minus_random_cosine_change"]
    paired.loc[is_overaggregation, "directional_support"] *= -1.0
    paired.to_csv(args.output_dir / "paired_effects.csv", index=False)
    report = {
        "status": "residual_pair_peak_occlusion",
        "checkpoint": str(args.official_checkpoint),
        "pairs_selected": int(len(selected)),
        "variants_encoded": int(len(metadata)),
        "random_repeats_requested": int(args.random_repeats),
        "interventions": {
            "overaggregation": "delete shared fragment peaks from one spectrum; expect pair cosine decrease",
            "identity_instability": "delete unmatched peaks from one spectrum; expect pair cosine increase",
        },
        "random_control": "same count with intensity and approximate m/z matching",
        "results": summarize(paired, args.bootstrap, args.seed),
        "claim_boundary": (
            "A positive directional effect supports a causal contribution of the selected peak class. "
            "It does not prove a unique chemical mechanism or justify training without confirmation."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

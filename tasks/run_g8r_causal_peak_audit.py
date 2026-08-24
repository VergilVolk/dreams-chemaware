"""Causal peak audit for the full real-error atlas.

Every selected query is tested in two independent arms:
* positive-deficit arm: delete condition-specific/unmatched peaks from one
  same-identity spectrum and expect similarity to increase;
* negative-excess arm: delete shared peaks from a different-identity hard
  negative and expect similarity to decrease.

Each targeted deletion is paired with count/intensity/mz-matched random peak
deletions.  Results are screening evidence, never structure labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# Project helpers live both at repository root and under tasks/.  Add both
# explicitly so direct execution and importlib/pytest loading behave identically.
ROOT = Path(__file__).resolve().parent.parent
for search_path in (ROOT, ROOT / "tasks"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import pilot_multilevel_factor_activations as multi
from e1_checkpoint_io import official_head_state
from pilot_paired_layer_cka import preprocess_spectrum
from run_residual_pair_peak_occlusion import (
    encode, matched_and_unique_mz, matched_random_tokens, perturb,
    read_spectra, select_target_subset, target_tokens,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signatures", type=Path, default=ROOT / "data/validation/g8r_real_error_analysis/query_error_signatures.csv.gz")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_causal_peak_audit")
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--protected-controls", type=int, default=2000)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--max-target-peaks", type=int, default=12)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--token-tolerance", type=float, default=0.005)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def select_cases(frame: pd.DataFrame, protected_cap: int, seed: int) -> pd.DataFrame:
    active = frame.loc[frame["transition"] != "protected_correct"].copy()
    protected = frame.loc[frame["transition"] == "protected_correct"].copy()
    if protected_cap and len(protected) > protected_cap:
        order = protected["query_row"].map(lambda row: stable_seed(seed, "control", int(row)))
        protected = protected.loc[order.sort_values(kind="mergesort").index[:protected_cap]]
    selected = pd.concat([active, protected], ignore_index=True)
    if selected["query_index"].duplicated().any():
        raise RuntimeError("selected causal cases contain duplicate query indices")
    return selected


def load_embedding_map(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as body:
        rows = body["rows"].astype(np.int64)
        embeddings = body["embeddings"].astype(np.float32)
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    if len(set(map(int, rows))) != len(rows):
        raise RuntimeError("embedding cache contains duplicate HDF5 rows")
    return {int(row): embeddings[index] for index, row in enumerate(rows)}


def cluster_ci(
    frame: pd.DataFrame, column: str, cluster: str, n: int, seed: int,
) -> list[float] | None:
    values = frame.groupby(cluster, sort=False)[column].mean().to_numpy(float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(frame: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    output = {}
    for position, ((arm, transition), group) in enumerate(frame.groupby(["arm", "transition"], sort=True)):
        output[f"{arm}|{transition}"] = {
            "directed_cases": int(len(group)),
            "query_identities": int(group["query_ik14"].nunique()),
            "mean_directional_support": float(group["directional_support"].mean()),
            "median_directional_support": float(group["directional_support"].median()),
            "supportive_fraction": float((group["directional_support"] > 0).mean()),
            "identity_cluster_bootstrap_95ci": cluster_ci(
                group, "directional_support", "query_ik14", bootstrap, seed + position,
            ),
            "formula_cluster_bootstrap_95ci": cluster_ci(
                group, "directional_support", "query_formula", bootstrap,
                seed + 10_000 + position,
            ),
            "median_removed_peaks": float(group["removed_count"].median()),
        }
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (args.signatures, args.embedding_cache, args.data, args.raw_checkpoint, args.official_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    frame = pd.read_csv(args.signatures)
    selected = select_cases(frame, args.protected_controls, args.seed)
    required_columns = {
        "query_row", "positive_dreams_pair_row", "hard_negative_dreams_pair_row",
        "query_ik14", "query_formula", "transition",
    }
    missing = required_columns - set(selected.columns)
    if missing:
        raise RuntimeError(f"signature table missing columns: {sorted(missing)}")
    row_columns = ["query_row", "positive_dreams_pair_row", "hard_negative_dreams_pair_row"]
    selected[row_columns] = selected[row_columns].astype(np.int64)
    rows = np.unique(np.concatenate([
        selected["query_row"].to_numpy(np.int64),
        selected["positive_dreams_pair_row"].to_numpy(np.int64),
        selected["hard_negative_dreams_pair_row"].to_numpy(np.int64),
    ]))
    embeddings = load_embedding_map(args.embedding_cache)
    missing_embeddings = sorted(set(map(int, rows)) - set(embeddings))
    if missing_embeddings:
        raise RuntimeError(f"{len(missing_embeddings)} required rows absent from embeddings")
    with h5py.File(args.data, "r") as handle:
        spectra = read_spectra(handle, rows)
        precursor = {int(row): float(handle["precursor_mz"][int(row)]) for row in rows}

    tensors: list[torch.Tensor] = []
    metadata: list[dict] = []
    for item in selected.itertuples(index=False):
        query_row = int(item.query_row)
        for arm, reference_row, mode in (
            ("positive_deficit", int(item.positive_dreams_pair_row), "unique"),
            ("negative_excess", int(item.hard_negative_dreams_pair_row), "shared"),
        ):
            for direction, source_row, target_row in (
                ("query_to_reference", query_row, reference_row),
                ("reference_to_query", reference_row, query_row),
            ):
                shared_mz, unique_mz = matched_and_unique_mz(
                    spectra[source_row], spectra[target_row], args.fragment_tolerance,
                )
                target_mz = shared_mz if mode == "shared" else unique_mz
                clean = preprocess_spectrum(
                    spectra[source_row], precursor[source_row], args.n_highest_peaks,
                )
                all_targeted = target_tokens(clean, target_mz, args.token_tolerance)
                targeted, excluded = select_target_subset(clean, all_targeted, args.max_target_peaks)
                if not len(targeted):
                    continue
                base = {
                    "query_index": int(item.query_index),
                    "query_ik14": str(item.query_ik14),
                    "query_formula": str(item.query_formula),
                    "transition": str(item.transition),
                    "arm": arm,
                    "direction": direction,
                    "source_row": source_row,
                    "target_row": target_row,
                    "removed_count": int(len(targeted)),
                    "target_class_peak_count": int(len(all_targeted)),
                }
                tensors.append(perturb(clean, targeted))
                metadata.append(base | {"condition": "targeted", "repeat": -1})
                for repeat in range(args.random_repeats):
                    control = matched_random_tokens(
                        clean, targeted, excluded,
                        stable_seed(args.seed, item.query_index, arm, direction, repeat),
                    )
                    if len(control) != len(targeted):
                        continue
                    tensors.append(perturb(clean, control))
                    metadata.append(base | {"condition": "matched_random", "repeat": repeat})
    if not tensors:
        raise RuntimeError("no peak interventions survived preprocessing")
    print(f"[occlusion] selected={len(selected):,} variants={len(tensors):,}", flush=True)

    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device,
    )
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    encoded = encode(model, weight, bias, tensors, args.batch_size, device)
    del model, raw_package, official_package, tensors
    gc.collect()

    records = []
    for vector, item in zip(encoded, metadata):
        source = embeddings[int(item["source_row"])]
        target = embeddings[int(item["target_row"])]
        clean_similarity = float(source @ target)
        perturbed_similarity = float(vector @ target)
        records.append(item | {
            "clean_similarity": clean_similarity,
            "perturbed_similarity": perturbed_similarity,
            "similarity_change": perturbed_similarity - clean_similarity,
            "embedding_preservation": float(vector @ source),
        })
    results = pd.DataFrame(records)
    key = ["query_index", "query_ik14", "query_formula", "transition", "arm", "direction"]
    targeted = results.loc[results["condition"] == "targeted"].copy()
    controls = results.loc[results["condition"] == "matched_random"].groupby(key, as_index=False).agg(
        random_similarity_change=("similarity_change", "mean"),
        random_embedding_preservation=("embedding_preservation", "mean"),
        random_repeats_observed=("repeat", "count"),
    )
    paired = targeted.merge(controls, on=key, validate="one_to_one")
    incomplete_controls = paired["random_repeats_observed"] != args.random_repeats
    n_incomplete_controls = int(incomplete_controls.sum())
    paired = paired.loc[~incomplete_controls].copy()
    if paired.empty:
        raise RuntimeError("no interventions retained a complete matched-random control set")
    paired["target_minus_random_change"] = (
        paired["similarity_change"] - paired["random_similarity_change"]
    )
    paired["directional_support"] = paired["target_minus_random_change"]
    paired.loc[paired["arm"] == "negative_excess", "directional_support"] *= -1.0
    report = {
        "status": "g8r_causal_peak_audit_complete",
        "selected_queries": int(len(selected)),
        "selected_identities": int(selected["query_ik14"].nunique()),
        "variants_encoded": int(len(results)),
        "random_repeats": int(args.random_repeats),
        "directed_interventions_with_complete_controls": int(len(paired)),
        "directed_interventions_excluded_for_incomplete_controls": n_incomplete_controls,
        "results": summarize(paired, args.bootstrap, args.seed),
        "interpretation": {
            "positive_deficit": "positive support means targeted unique-peak deletion raises same-identity similarity more than matched random deletion",
            "negative_excess": "positive support means targeted shared-peak deletion lowers different-identity similarity more than matched random deletion",
        },
        "training_gate": "Only identity/formula-cluster replicated arms with CI above zero may define the next fine-tuning pool.",
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent))
    try:
        selected.to_csv(staging / "selected_queries.csv.gz", index=False, compression="gzip")
        results.to_csv(staging / "perturbation_results.csv.gz", index=False, compression="gzip")
        paired.to_csv(staging / "paired_effects.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

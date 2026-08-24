"""M1b single-peak candidate-set headroom audit for directional noise V2.

This is an oracle audit, not a training result.  Each consensus-derived
conditional peak is deleted one at a time.  The best targeted deletion is
compared with the best of an equally sized set of intensity/mz-matched random
single-peak deletions, repeated independently.  Comparing best with best
controls the multiple-choice advantage of searching several candidate peaks.
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


ROOT = Path(__file__).resolve().parent.parent
for search_path in (ROOT, ROOT / "tasks"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import pilot_multilevel_factor_activations as multi
from e1_checkpoint_io import official_head_state
from pilot_paired_layer_cka import preprocess_spectrum
from run_directional_noise_v2_margin_audit import (
    as_bool,
    load_embeddings,
    parse_float_list,
    stable_seed,
)
from run_residual_pair_peak_occlusion import encode, matched_random_tokens, perturb, target_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m1")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m1b_headroom")
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--max-target-peaks", type=int, default=12)
    parser.add_argument("--max-mask-fraction", type=float, default=0.20)
    parser.add_argument("--token-tolerance", type=float, default=0.005)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_ci(frame: pd.DataFrame, cluster: str, column: str, n: int, seed: int) -> list[float] | None:
    values = frame.groupby(cluster, sort=False)[column].mean().dropna().to_numpy(float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for index in range(n):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(group: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    return {
        "queries": int(len(group)),
        "identities": int(group["ik14"].nunique()),
        "formulas": int(group["formula"].nunique()),
        "baseline_accuracy": float(group["baseline_top1"].mean()),
        "oracle_target_accuracy": float(group["target_top1"].mean()),
        "mean_best_random_accuracy": float(group["random_top1_mean"].mean()),
        "oracle_target_corrected": int(((group["baseline_top1"] == 0) & (group["target_top1"] == 1)).sum()),
        "oracle_target_introduced": int(((group["baseline_top1"] == 1) & (group["target_top1"] == 0)).sum()),
        "expected_best_random_corrected": float(((group["baseline_top1"] == 0) * group["random_top1_mean"]).sum()),
        "expected_best_random_introduced": float(((group["baseline_top1"] == 1) * (1.0 - group["random_top1_mean"])).sum()),
        "mean_best_target_minus_best_random_margin": float(group["target_minus_random_best_margin"].mean()),
        "median_best_target_minus_best_random_margin": float(group["target_minus_random_best_margin"].median()),
        "mean_target_minus_random_top1_delta": float(group["target_minus_random_top1_delta"].mean()),
        "identity_margin_specificity_95ci": cluster_ci(group, "ik14", "target_minus_random_best_margin", bootstrap, seed),
        "formula_margin_specificity_95ci": cluster_ci(group, "formula", "target_minus_random_best_margin", bootstrap, seed + 10_000),
        "identity_top1_specificity_95ci": cluster_ci(group, "ik14", "target_minus_random_top1_delta", bootstrap, seed + 20_000),
        "formula_top1_specificity_95ci": cluster_ci(group, "formula", "target_minus_random_top1_delta", bootstrap, seed + 30_000),
        "median_candidate_peaks": float(group["candidate_count"].median()),
    }


def aggregate_best_of_k(results: pd.DataFrame, random_repeats: int) -> pd.DataFrame:
    keys = [
        "query_row", "positive_row", "negative_row", "ik14", "formula", "adduct",
        "cross_condition_positive", "baseline_margin", "candidate_count",
    ]
    targeted = results.loc[results["condition"] == "targeted"].copy()
    targeted = targeted.sort_values(
        ["query_row", "perturbed_margin", "target_slot"],
        ascending=[True, False, True], kind="mergesort",
    ).groupby("query_row", sort=False).head(1)
    targeted = targeted[keys + ["target_slot", "target_token", "target_mz", "perturbed_margin"]].rename(
        columns={
            "target_slot": "best_target_slot", "target_token": "best_target_token",
            "target_mz": "best_target_mz", "perturbed_margin": "best_target_margin",
        }
    )
    random = results.loc[results["condition"] == "matched_random"].copy()
    random_best = random.groupby(keys + ["repeat"], as_index=False)["perturbed_margin"].max()
    random_summary = random_best.groupby(keys, as_index=False).agg(
        mean_best_random_margin=("perturbed_margin", "mean"),
        random_top1_mean=("perturbed_margin", lambda values: float((values > 0).mean())),
        random_repeats_observed=("repeat", "nunique"),
    )
    paired = targeted.merge(random_summary, on=keys, validate="one_to_one")
    paired = paired.loc[paired["random_repeats_observed"] == random_repeats].copy()
    paired["baseline_top1"] = (paired["baseline_margin"] > 0).astype(int)
    paired["target_top1"] = (paired["best_target_margin"] > 0).astype(int)
    paired["target_minus_random_best_margin"] = paired["best_target_margin"] - paired["mean_best_random_margin"]
    paired["target_minus_random_top1_delta"] = paired["target_top1"] - paired["random_top1_mean"]
    paired["baseline_status"] = np.where(paired["baseline_top1"] == 1, "baseline_correct", "baseline_wrong")
    return paired


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    selected_path = args.m1_dir / "selected_triples.csv.gz"
    for path in (selected_path, args.embedding_cache, args.data, args.raw_checkpoint, args.official_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.random_repeats < 3:
        raise ValueError("random-repeats must be at least 3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    selected = pd.read_csv(selected_path)
    selected["cross_condition_positive"] = as_bool(selected["cross_condition_positive"])
    _, embeddings, embedding_index = load_embeddings(args.embedding_cache)
    required_rows = np.unique(selected[["query_row", "positive_row", "negative_row"]].to_numpy(np.int64))
    missing_embeddings = set(map(int, required_rows)) - set(embedding_index)
    if missing_embeddings:
        raise RuntimeError(f"M1b triples miss {len(missing_embeddings)} official embeddings")

    query_rows = np.unique(selected["query_row"].to_numpy(np.int64))
    with h5py.File(args.data, "r") as handle:
        spectra = {int(row): np.asarray(handle["spectrum"][int(row)], dtype=float) for row in query_rows}
        precursor = np.asarray(handle["precursor_mz"][:], dtype=float)

    tensors: list[torch.Tensor] = []
    metadata: list[dict] = []
    retained_queries = 0
    for item in selected.itertuples(index=False):
        clean = preprocess_spectrum(spectra[int(item.query_row)], float(precursor[int(item.query_row)]), args.n_highest_peaks)
        targets = target_tokens(clean, parse_float_list(item.conditional_mz), args.token_tolerance)
        core = target_tokens(clean, parse_float_list(item.core_mz), args.token_tolerance)
        values = clean.numpy()
        valid = np.flatnonzero((np.arange(len(values)) > 0) & (values[:, 0] > 0) & (values[:, 1] > 0))
        strong = valid[values[valid, 1] >= 0.20]
        excluded = set(targets.tolist()) | set(core.tolist()) | set(strong.tolist())
        targets = np.asarray([
            index for index in targets if index not in set(core.tolist()) and index not in set(strong.tolist())
        ], dtype=np.int64)
        if not len(targets) or not any(index not in excluded for index in valid):
            continue
        fraction_cap = max(1, int(np.floor(len(valid) * args.max_mask_fraction)))
        capacity = min(len(targets), args.max_target_peaks, fraction_cap)
        order = np.argsort(values[targets, 1].astype(float))[::-1]
        targets = targets[order[:capacity]].astype(np.int64)
        controls: dict[tuple[int, int], int] = {}
        complete = True
        # Match all k controls jointly so each repeat contains k distinct
        # random peaks. This gives targeted and random arms identical search
        # multiplicity for the best-of-k comparison.
        for repeat in range(args.random_repeats):
            matched = matched_random_tokens(
                clean, targets, excluded,
                stable_seed(args.seed, item.query_row, "single-peak-random", repeat),
            )
            if len(matched) != len(targets):
                complete = False
                break
            for slot, control in enumerate(matched):
                controls[(slot, repeat)] = int(control)
        if not complete:
            continue
        q = embeddings[embedding_index[int(item.query_row)]]
        p = embeddings[embedding_index[int(item.positive_row)]]
        n = embeddings[embedding_index[int(item.negative_row)]]
        base = {
            "query_row": int(item.query_row), "positive_row": int(item.positive_row),
            "negative_row": int(item.negative_row), "ik14": str(item.ik14),
            "formula": str(item.formula), "adduct": str(item.adduct),
            "cross_condition_positive": bool(item.cross_condition_positive),
            "baseline_margin": float(q @ p - q @ n), "candidate_count": int(len(targets)),
        }
        for slot, target in enumerate(targets):
            tensors.append(perturb(clean, np.asarray([target], dtype=np.int64)))
            metadata.append(base | {
                "condition": "targeted", "repeat": -1, "target_slot": int(slot),
                "target_token": int(target), "target_mz": float(values[target, 0]),
            })
            for repeat in range(args.random_repeats):
                control = controls[(slot, repeat)]
                tensors.append(perturb(clean, np.asarray([control], dtype=np.int64)))
                metadata.append(base | {
                    "condition": "matched_random", "repeat": int(repeat), "target_slot": int(slot),
                    "target_token": int(control), "target_mz": float(values[control, 0]),
                })
        retained_queries += 1
    if not tensors:
        raise RuntimeError("no M1b single-peak interventions retained complete controls")
    print(f"[M1b] retained_queries={retained_queries:,} variants={len(tensors):,}", flush=True)

    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw_package, multi.official_backbone_state(official_package), device)
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    encoded = encode(model, weight, bias, tensors, args.batch_size, device)
    del model, raw_package, official_package, tensors
    gc.collect()

    records = []
    for vector, item in zip(encoded, metadata):
        p = embeddings[embedding_index[item["positive_row"]]]
        n = embeddings[embedding_index[item["negative_row"]]]
        records.append(item | {"perturbed_margin": float(vector @ p - vector @ n)})
    variants = pd.DataFrame(records)
    paired = aggregate_best_of_k(variants, args.random_repeats)
    if paired.empty:
        raise RuntimeError("no M1b query retained all best-of-k random controls")

    groups = {
        "overall": paired,
        "baseline_correct": paired.loc[paired["baseline_status"] == "baseline_correct"],
        "baseline_wrong": paired.loc[paired["baseline_status"] == "baseline_wrong"],
        "cross_condition": paired.loc[paired["cross_condition_positive"]],
    }
    summaries = {
        name: summarize(group, args.bootstrap, args.seed + position)
        for position, (name, group) in enumerate(groups.items()) if len(group)
    }
    wrong = summaries["baseline_wrong"]
    gates = {
        "baseline_wrong_identities_ge_200": wrong["identities"] >= 200,
        "wrong_margin_identity_ci_positive": wrong["identity_margin_specificity_95ci"][0] > 0,
        "wrong_margin_formula_ci_positive": wrong["formula_margin_specificity_95ci"][0] > 0,
        "wrong_top1_identity_ci_positive": wrong["identity_top1_specificity_95ci"][0] > 0,
        "wrong_top1_formula_ci_positive": wrong["formula_top1_specificity_95ci"][0] > 0,
    }
    report = {
        "status": "directional_noise_v2_m1b_candidate_headroom_complete",
        "selected_queries": int(len(selected)),
        "paired_queries": int(len(paired)),
        "fraction_retained": float(len(paired) / len(selected)),
        "random_repeats": int(args.random_repeats),
        "results": summaries,
        "gates": gates,
        "candidate_set_headroom_pass": bool(all(gates.values())),
        "decision": (
            "This is a multiplicity-matched oracle audit. A pass only permits development of an independent peak selector; "
            "it does not permit choosing peaks by their observed test-query outcome or starting fine-tuning."
        ),
        "provenance": {
            "selected_triples_sha256": sha256(selected_path),
            "embedding_cache_sha256": sha256(args.embedding_cache),
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
        },
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent))
    try:
        variants.to_csv(staging / "single_peak_variants.csv.gz", index=False, compression="gzip")
        paired.to_csv(staging / "best_of_k_query_results.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

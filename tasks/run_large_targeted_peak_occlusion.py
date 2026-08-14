"""Paired causal peak-occlusion audit on large DreaMS residual failures.

Two evidence classes come from the descriptive localization audit:
identity-supporting query peaks match the best identity spectrum only, while
confounder-supporting peaks match the DreaMS top same-formula error only.
Each targeted deletion is compared with query-only random deletions matched on
peak count, intensity and approximate m/z.  The clean library is never changed.
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
from e1_checkpoint_io import official_head_state
from pilot_paired_layer_cka import preprocess_spectrum


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def parse_values(value: object) -> np.ndarray:
    if pd.isna(value) or str(value).strip() == "":
        return np.empty(0, dtype=float)
    return np.asarray([float(item) for item in str(value).split(";") if item], dtype=float)


def target_tokens(clean: torch.Tensor, target_mz: np.ndarray, tolerance: float) -> np.ndarray:
    values = clean.numpy()
    # The base peak has no equally intense fragment control by definition.
    eligible = np.flatnonzero(
        (np.arange(len(values)) > 0) & (values[:, 0] > 0)
        & (values[:, 1] > 0) & (values[:, 1] < 1.0 - 1e-7)
    )
    chosen = []
    used: set[int] = set()
    for mz in target_mz:
        candidates = [i for i in eligible if i not in used and abs(float(values[i, 0]) - mz) <= tolerance]
        if candidates:
            index = min(candidates, key=lambda i: abs(float(values[i, 0]) - mz))
            chosen.append(index)
            used.add(index)
    return np.asarray(chosen, dtype=int)


def matched_random_tokens(
    clean: torch.Tensor, targeted: np.ndarray, excluded: set[int], seed: int,
) -> np.ndarray:
    values = clean.numpy()
    pool = np.asarray([
        i for i in range(1, len(values))
        if values[i, 0] > 0 and values[i, 1] > 0 and i not in excluded
    ], dtype=int)
    if not len(targeted) or len(pool) < len(targeted):
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    # Global assignment strongly prioritizes intensity, then approximate m/z.
    # Small seeded jitter yields several closely matched controls per query.
    valid_mz = values[1:, 0][values[1:, 0] > 0]
    mz_scale = max(float(np.std(valid_mz)), 25.0)
    log_intensity = np.log10(np.clip(values[:, 1].astype(float), 1e-6, None))
    cost = np.empty((len(targeted), len(pool)), dtype=float)
    for row, target in enumerate(targeted):
        log_difference = np.abs(log_intensity[pool] - log_intensity[target])
        linear_difference = np.abs(values[pool, 1].astype(float) - float(values[target, 1]))
        mz_difference = np.abs(values[pool, 0].astype(float) - float(values[target, 0])) / mz_scale
        cost[row] = 4.0 * log_difference + 8.0 * linear_difference + 0.15 * mz_difference
    cost += rng.gumbel(0.0, 0.015, size=cost.shape)
    row_indices, column_indices = linear_sum_assignment(cost)
    if len(row_indices) != len(targeted):
        return np.empty(0, dtype=int)
    return pool[column_indices].astype(int)


def perturb(clean: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    output = clean.clone()
    output[indices] = 0.0
    return output


def encode(
    model, weight: torch.Tensor, bias: torch.Tensor, tensors: list[torch.Tensor],
    batch_size: int, device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.stack(tensors)), batch_size=batch_size, shuffle=False)
    dtype = next(model.parameters()).dtype
    output = []
    with torch.inference_mode():
        for (batch,) in loader:
            precursor = model(batch.to(device=device, dtype=dtype), None)[:, 0]
            output.append(F.normalize(F.linear(precursor, weight, bias), dim=-1).float().cpu().numpy())
    return np.concatenate(output)


def retrieval_score(
    vector: np.ndarray, query_index: int, manifest: pd.DataFrame, library: np.ndarray,
    ppm: float,
) -> dict[str, object]:
    formula = manifest.at[query_index, "formula"]
    indices = manifest.index[manifest["formula"] == formula].to_numpy(np.int64)
    indices = indices[indices != query_index]
    query_mass = float(manifest.at[query_index, "precursor_mz"])
    candidate_mass = manifest.loc[indices, "precursor_mz"].to_numpy(float)
    delta_ppm = np.abs(candidate_mass - query_mass) / ((candidate_mass + query_mass) / 2.0) * 1e6
    query_hash = manifest.at[query_index, "spectrum_hash"]
    distinct_hash = manifest.loc[indices, "spectrum_hash"].to_numpy() != query_hash
    indices = indices[(delta_ppm <= ppm) & distinct_hash]
    query_ik = manifest.at[query_index, "ik14"]
    positive = indices[manifest.loc[indices, "ik14"].to_numpy() == query_ik]
    negative = indices[manifest.loc[indices, "ik14"].to_numpy() != query_ik]
    positive_score = float((library[positive] @ vector).max())
    molecule_scores = []
    for _, group in manifest.loc[negative].groupby("ik14", sort=False):
        molecule_scores.append(float((library[group.index.to_numpy(np.int64)] @ vector).max()))
    negative_scores = np.asarray(molecule_scores)
    best_negative = float(negative_scores.max())
    return {
        "positive_similarity": positive_score,
        "best_negative_similarity": best_negative,
        "margin": positive_score - best_negative,
        "top1_correct": bool(positive_score > best_negative),
        "pairwise_accuracy": float(np.mean(positive_score > negative_scores)),
    }


def formula_bootstrap(values: pd.DataFrame, column: str, iterations: int, seed: int) -> list[float]:
    formula_values = values.groupby("formula", sort=False)[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        draws[i] = rng.choice(formula_values, len(formula_values), replace=True).mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def summarize_one(results: pd.DataFrame, repeats: int, bootstrap: int, seed: int) -> dict[str, object]:
    output: dict[str, object] = {}
    for evidence in sorted(results["evidence"].unique()):
        targeted = results.loc[(results["evidence"] == evidence) & (results["condition"] == "targeted")].copy()
        random = results.loc[(results["evidence"] == evidence) & (results["condition"] == "matched_random")]
        random_mean = random.groupby(["split", "query_index"], as_index=False).agg(
            random_margin=("margin", "mean"), random_top1=("top1_correct", "mean"),
            random_pairwise=("pairwise_accuracy", "mean"),
        )
        paired = targeted.merge(random_mean, on=["split", "query_index"], how="inner")
        paired["target_minus_random_margin"] = paired["margin"] - paired["random_margin"]
        paired["target_minus_random_top1"] = paired["top1_correct"].astype(float) - paired["random_top1"]
        paired.to_csv(Path(results.attrs["output_dir"]) / f"{evidence}_paired_summary.csv", index=False)
        output[evidence] = {
            "queries": len(paired), "molecules": int(paired["ik14"].nunique()),
            "formulas": int(paired["formula"].nunique()), "random_repeats": repeats,
            "clean_top1": float(paired["clean_top1"].mean()),
            "targeted_top1": float(paired["top1_correct"].mean()),
            "matched_random_top1": float(paired["random_top1"].mean()),
            "mean_target_minus_random_margin": float(paired["target_minus_random_margin"].mean()),
            "formula_balanced_target_minus_random_margin": float(
                paired.groupby("formula")["target_minus_random_margin"].mean().mean()
            ),
            "formula_bootstrap_ci95": formula_bootstrap(
                paired, "target_minus_random_margin", bootstrap, seed + (0 if evidence == "identity" else 1)
            ),
            "mean_target_minus_random_top1": float(paired["target_minus_random_top1"].mean()),
            "mean_embedding_cosine_to_clean_targeted": float(paired["embedding_cosine_to_clean"].mean()),
            "median_removed_peaks": float(paired["removed_count"].median()),
        }
    return output


def summarize(results: pd.DataFrame, repeats: int, bootstrap: int, seed: int) -> dict[str, object]:
    output = {"combined": summarize_one(results, repeats, bootstrap, seed), "by_split": {}}
    for position, (split, group) in enumerate(results.groupby("split", sort=True)):
        group = group.copy()
        group.attrs["output_dir"] = results.attrs["output_dir"]
        output["by_split"][split] = summarize_one(group, repeats, bootstrap, seed + 10 + position)
    if "audit_quadrant" in results.columns:
        output["by_failure_type"] = {}
        for position, (label, group) in enumerate(results.groupby("audit_quadrant", sort=True)):
            group = group.copy()
            group.attrs["output_dir"] = results.attrs["output_dir"]
            output["by_failure_type"][label] = summarize_one(group, repeats, bootstrap, seed + 20 + position)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--localization-dir", type=Path, default=Path("data/validation/large_residual_peak_localization"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_targeted_peak_occlusion"))
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--match-tolerance", type=float, default=0.005)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--evidence", nargs="+", choices=("identity", "confounder"), default=["identity", "confounder"])
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    tensors: list[torch.Tensor] = []
    metadata: list[dict[str, object]] = []
    split_resources: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    with h5py.File(args.data, "r") as handle:
        for split in args.splits:
            manifest = pd.read_csv(args.embedding_root / f"large_observability_embeddings_{split}" / "manifest.csv")
            library = np.load(args.embedding_root / f"large_observability_embeddings_{split}" / "official_embeddings.npy").astype(np.float32)
            evidence = pd.read_csv(args.localization_dir / f"{split}_peak_evidence.csv")
            split_resources[split] = (manifest, library)
            for row in evidence.itertuples(index=False):
                query_index = int(row.query_index)
                raw = np.asarray(handle["spectrum"][int(row.query_hdf5_row)])
                clean = preprocess_spectrum(raw, float(row.query_precursor_mz), args.n_highest_peaks)
                identity = target_tokens(clean, parse_values(row.fragment_identity_support_mz), args.match_tolerance)
                confounder = target_tokens(clean, parse_values(row.fragment_confounder_support_mz), args.match_tolerance)
                all_confounder_mz = (
                    parse_values(row.fragment_all_confounder_support_mz)
                    if hasattr(row, "fragment_all_confounder_support_mz")
                    else parse_values(row.fragment_confounder_support_mz)
                )
                all_confounder = target_tokens(clean, all_confounder_mz, args.match_tolerance)
                excluded = set(identity.tolist()) | set(all_confounder.tolist())
                for label, targeted in (("identity", identity), ("confounder", confounder)):
                    if label not in args.evidence:
                        continue
                    if not len(targeted):
                        continue
                    tensors.append(perturb(clean, targeted))
                    metadata.append({
                        "split": split, "query_index": query_index, "ik14": row.ik14,
                        "formula": row.formula, "ring_class": row.ring_class,
                        "audit_quadrant": row.audit_quadrant,
                        "robust_model_residual_candidate": bool(row.robust_model_residual_candidate),
                        "evidence": label, "condition": "targeted", "repeat": -1,
                        "removed_count": len(targeted),
                        "removed_mz": ";".join(f"{float(clean[i, 0]):.5f}" for i in targeted),
                    })
                    for repeat in range(args.random_repeats):
                        random_indices = matched_random_tokens(
                            clean, targeted, excluded,
                            stable_seed(args.seed, split, query_index, label, repeat),
                        )
                        if len(random_indices) != len(targeted):
                            continue
                        tensors.append(perturb(clean, random_indices))
                        metadata.append({
                            "split": split, "query_index": query_index, "ik14": row.ik14,
                            "formula": row.formula, "ring_class": row.ring_class,
                            "audit_quadrant": row.audit_quadrant,
                            "robust_model_residual_candidate": bool(row.robust_model_residual_candidate),
                            "evidence": label, "condition": "matched_random", "repeat": repeat,
                            "removed_count": len(random_indices),
                            "removed_mz": ";".join(f"{float(clean[i, 0]):.5f}" for i in random_indices),
                        })
    if not tensors:
        raise RuntimeError("No targeted peaks survived model preprocessing")

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

    rows = []
    clean_cache: dict[tuple[str, int], dict[str, object]] = {}
    for vector, item in zip(encoded, metadata):
        split, query_index = str(item["split"]), int(item["query_index"])
        manifest, library = split_resources[split]
        key = (split, query_index)
        if key not in clean_cache:
            clean_cache[key] = retrieval_score(library[query_index], query_index, manifest, library, args.ppm)
        clean_score = clean_cache[key]
        noisy_score = retrieval_score(vector, query_index, manifest, library, args.ppm)
        rows.append(item | noisy_score | {
            "clean_margin": clean_score["margin"], "clean_top1": clean_score["top1_correct"],
            "clean_pairwise_accuracy": clean_score["pairwise_accuracy"],
            "margin_change": float(noisy_score["margin"] - clean_score["margin"]),
            "embedding_cosine_to_clean": float(vector @ library[query_index]),
        })
    results = pd.DataFrame(rows)
    results.attrs["output_dir"] = str(args.output_dir)
    results.to_csv(args.output_dir / "perturbation_results.csv", index=False)
    summary = summarize(results, args.random_repeats, args.bootstrap, args.seed)
    report = {
        "status": "large_targeted_peak_occlusion",
        "splits": args.splits,
        "variants_encoded": len(metadata),
        "intervention": "query-only peak dropout; clean candidate library",
        "random_control": "same peak count, intensity and approximate m/z matched; target classes excluded",
        "statistical_unit": "formula-cluster bootstrap",
        "candidate_protocol": f"same formula, precursor delta <= {args.ppm:g} ppm, duplicate hashes excluded",
        "interpretation": {
            "identity_negative_excess": "identity peaks carry DreaMS-used true evidence",
            "confounder_positive_excess": "confounder peaks carry DreaMS-overweighted misleading evidence",
        },
        "results": summary,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

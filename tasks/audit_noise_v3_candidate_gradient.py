"""Full candidate-conditioned, input-gradient directional-noise audit.

This is a headroom/causal-selection audit.  It trains no model and never uses
chemical rules as labels.  Every query is evaluated against its complete
strict-10ppm candidate graph after one query peak is attenuated or deleted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from build_g8r_real_error_atlas import Cache, load_p3_identities, sha256_file  # noqa: E402
from noise_v3_core import (  # noqa: E402
    CONFOUNDER_ONLY, IDENTITY_ONLY, ROLE_NAMES, attenuate_and_renormalize,
    candidate_peak_roles_from_mz,
    candidate_representatives, matched_control_tokens, predicted_gain,
    rank_gradient_targets, rank_role_targets, stable_seed,
)
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


DEFAULT_CACHE = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_EMBEDDINGS = ROOT / "data/validation/g8r_p2_official_embeddings.npz"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCHITECTURE = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_noise_v3_candidate_gradient"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--top-k-negatives", type=int, default=5)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--selection-attenuation", type=float, default=0.50)
    parser.add_argument("--attenuations", type=float, nargs="+", default=[0.25, 0.50, 1.0])
    parser.add_argument(
        "--selectors", nargs="+",
        choices=["candidate_gradient", "role_confounder", "role_identity"],
        default=["candidate_gradient", "role_confounder"],
        help="Preregistered single-peak selectors to evaluate in one shared run.",
    )
    parser.add_argument("--candidate-gradient-top-k", type=int, default=1)
    parser.add_argument("--role-confounder-top-k", type=int, default=1)
    parser.add_argument("--control-repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke only; 0 is formal full graph")
    return parser.parse_args()


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    with np.load(path) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(rows) != len(embeddings):
        raise RuntimeError("embedding cache is malformed")
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("embedding cache has duplicate HDF5 rows")
    return rows, embeddings, {int(row): index for index, row in enumerate(rows)}


def query_candidate_block(cache: Cache, query: int, score_column: int):
    molecule_left, molecule_right = map(int, cache.query_ptr[query:query + 2])
    pair_left = int(cache.molecule_ptr[molecule_left])
    pair_right = int(cache.molecule_ptr[molecule_right])
    local_ptr = cache.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
    return (
        cache.features[pair_left:pair_right, score_column],
        cache.pair_candidate_row[pair_left:pair_right],
        local_ptr,
    )


def strict_metrics(scores: np.ndarray, molecule_ptr: np.ndarray) -> tuple[int, float, float]:
    molecule_scores = np.maximum.reduceat(np.asarray(scores, float), molecule_ptr[:-1])
    positive = float(molecule_scores[0])
    negative = float(molecule_scores[1:].max())
    rank = 1 + int(np.sum(molecule_scores[1:] >= positive))
    return rank, 1.0 / rank, positive - negative


def top_reference_rows(
    scores: np.ndarray, rows: np.ndarray, molecule_ptr: np.ndarray, molecule_index: int, count: int = 3,
) -> np.ndarray:
    left, right = map(int, molecule_ptr[molecule_index:molecule_index + 2])
    order = np.argsort(-np.asarray(scores[left:right]), kind="mergesort")[:count] + left
    return np.asarray(rows[order], dtype=np.int64)


def cluster_ci(frame: pd.DataFrame, value: str, cluster: str, n: int, seed: int) -> list[float] | None:
    values = frame.groupby(cluster, sort=False)[value].mean().dropna().to_numpy(float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def summarize_paired(frame: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    output = {}
    group_columns = ["selector", "attenuation", "baseline_state", "near_state"]
    for position, (key, group) in enumerate(frame.groupby(group_columns, sort=True, dropna=False)):
        selector, attenuation, baseline_state, near_state = key
        target_correct = group["target_rank"].eq(1)
        random_correct = group["random_top1"].astype(float)
        target_minus_random_top1 = target_correct.astype(float) - random_correct
        local = group.assign(target_minus_random_top1=target_minus_random_top1)
        name = f"{selector}|a={float(attenuation):.2f}|{baseline_state}|{near_state}"
        output[name] = {
            "queries": int(len(group)),
            "identities": int(group["query_ik14"].nunique()),
            "formulas": int(group["query_formula"].nunique()),
            "target_margin_change": float(group["target_margin_change"].mean()),
            "random_margin_change": float(group["random_margin_change"].mean()),
            "target_minus_random_margin_change": float(group["target_minus_random_margin_change"].mean()),
            "target_minus_random_top1": float(target_minus_random_top1.mean()),
            "target_corrected": int(((group["baseline_rank"] > 1) & target_correct).sum()),
            "target_introduced": int(((group["baseline_rank"] == 1) & ~target_correct).sum()),
            "identity_margin_ci": cluster_ci(
                group, "target_minus_random_margin_change", "query_ik14", bootstrap, seed + position,
            ),
            "formula_margin_ci": cluster_ci(
                group, "target_minus_random_margin_change", "query_formula", bootstrap,
                seed + 10_000 + position,
            ),
            "identity_top1_ci": cluster_ci(
                local, "target_minus_random_top1", "query_ik14", bootstrap,
                seed + 20_000 + position,
            ),
            "formula_top1_ci": cluster_ci(
                local, "target_minus_random_top1", "query_formula", bootstrap,
                seed + 30_000 + position,
            ),
        }
    return output


def summarize_paired_descriptive(frame: pd.DataFrame) -> dict:
    """All matrix cells without extra resampling for sensitivity panels."""
    output = {}
    group_columns = ["selector", "attenuation", "baseline_state", "near_state"]
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        selector, attenuation, baseline_state, near_state = key
        target_correct = group["target_rank"].eq(1)
        target_minus_random_top1 = (
            target_correct.astype(float) - group["random_top1"].astype(float)
        )
        name = f"{selector}|a={float(attenuation):.2f}|{baseline_state}|{near_state}"
        output[name] = {
            "queries": int(len(group)),
            "identities": int(group["query_ik14"].nunique()),
            "formulas": int(group["query_formula"].nunique()),
            "target_minus_random_margin_change": float(
                group["target_minus_random_margin_change"].mean()
            ),
            "target_minus_random_top1": float(target_minus_random_top1.mean()),
            "target_corrected": int(((group["baseline_rank"] > 1) & target_correct).sum()),
            "target_introduced": int(((group["baseline_rank"] == 1) & ~target_correct).sum()),
        }
    return output


def gate_summary(group: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    """One preregistered aggregate, without hiding signal in near strata."""
    target_correct = group["target_rank"].eq(1)
    target_minus_random_top1 = target_correct.astype(float) - group["random_top1"].astype(float)
    local = group.assign(target_minus_random_top1=target_minus_random_top1)
    return {
        "queries": int(len(group)),
        "identities": int(group["query_ik14"].nunique()),
        "formulas": int(group["query_formula"].nunique()),
        "mean_target_minus_random_margin": float(group["target_minus_random_margin_change"].mean()),
        "mean_target_minus_random_top1": float(target_minus_random_top1.mean()),
        "corrected_vs_baseline": int(((group["baseline_rank"] > 1) & target_correct).sum()),
        "introduced_vs_baseline": int(((group["baseline_rank"] == 1) & ~target_correct).sum()),
        "identity_margin_ci": cluster_ci(
            group, "target_minus_random_margin_change", "query_ik14", bootstrap, seed,
        ),
        "formula_margin_ci": cluster_ci(
            group, "target_minus_random_margin_change", "query_formula", bootstrap, seed + 10_000,
        ),
        "identity_top1_ci": cluster_ci(
            local, "target_minus_random_top1", "query_ik14", bootstrap, seed + 20_000,
        ),
        "formula_top1_ci": cluster_ci(
            local, "target_minus_random_top1", "query_formula", bootstrap, seed + 30_000,
        ),
    }


def main() -> None:
    args = parse_args()
    formal = args.max_queries == 0
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (
        args.cache, args.embedding_cache, args.data, args.official_checkpoint,
        args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.p3_dir.is_dir():
        raise FileNotFoundError(args.p3_dir)
    if not (0 < args.selection_attenuation <= 1):
        raise ValueError("invalid selection attenuation")
    if any(not 0 < value <= 1 for value in args.attenuations):
        raise ValueError("all attenuations must lie in (0, 1]")
    if len(set(args.attenuations)) != len(args.attenuations):
        raise ValueError("attenuation doses must be unique")
    if len(set(args.selectors)) != len(args.selectors):
        raise ValueError("selectors must be unique")
    if args.candidate_gradient_top_k < 1 or args.role_confounder_top_k < 1:
        raise ValueError("selector top-k values must be positive")
    if args.control_repeats < 1 or args.top_k_negatives < 1:
        raise ValueError("control repeats and top-k must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cache = Cache(args.cache)
    count = cache.n_queries if formal else min(args.max_queries, cache.n_queries)
    p3 = load_p3_identities(args.p3_dir)
    overlap = set(map(str, cache.query_ik14[:count])) & p3
    if overlap:
        raise RuntimeError(f"P3 leakage: {len(overlap)} query identities")
    score_column = cache.feature_names.index("dreams_similarity")
    embedding_rows, embeddings, embedding_index = load_embeddings(args.embedding_cache)
    needed = set(map(int, cache.query_row[:count]))
    for query in range(count):
        _, pair_rows, _ = query_candidate_block(cache, query, score_column)
        needed.update(map(int, pair_rows))
    missing = needed - set(embedding_index)
    if missing:
        raise RuntimeError(f"embedding cache misses {len(missing)} candidate-graph rows")

    representatives = []
    for query in range(count):
        scores, rows, ptr = query_candidate_block(cache, query, score_column)
        representatives.append(candidate_representatives(scores, rows, ptr, args.top_k_negatives))

    # Preprocess every reachable row exactly once. The candidate graph reuses the
    # same spectra heavily, so per-query HDF5 preprocessing would be both slow and
    # an avoidable source of drift.
    tensor_cache: dict[int, torch.Tensor] = {}
    with h5py.File(args.data, "r") as handle:
        for position, row in enumerate(sorted(needed)):
            tensor_cache[int(row)] = preprocess_spectrum(
                np.asarray(handle["spectrum"][row]), float(handle["precursor_mz"][row]),
                args.n_highest_peaks,
            )
            if (position + 1) % 5000 == 0:
                print(f"[spectra] {position + 1:,}/{len(needed):,}", flush=True)
    query_tensors = [tensor_cache[int(row)] for row in cache.query_row[:count]]

    # Roles are defined against small unions of the three best positive spectra
    # and three best spectra from the hardest negative molecule. This protects a
    # true peak supported by any close experimental replicate instead of trusting
    # one arbitrary representative.
    role_arrays: list[np.ndarray] = []
    for query, rep in enumerate(representatives):
        scores, rows, ptr = query_candidate_block(cache, query, score_column)
        positive_rows = top_reference_rows(scores, rows, ptr, 0, 3)
        negative_molecule = next(
            index for index, (left, right) in enumerate(zip(ptr[:-1], ptr[1:]))
            if int(rep.negative_rows[0]) in set(map(int, rows[int(left):int(right)]))
        )
        negative_rows = top_reference_rows(scores, rows, ptr, negative_molecule, 3)
        positive_mz = np.concatenate([
            tensor_cache[int(row)][1:, 0].numpy()[tensor_cache[int(row)][1:, 1].numpy() > 0]
            for row in positive_rows
        ])
        negative_mz = np.concatenate([
            tensor_cache[int(row)][1:, 0].numpy()[tensor_cache[int(row)][1:, 1].numpy() > 0]
            for row in negative_rows
        ])
        role_arrays.append(candidate_peak_roles_from_mz(
            query_tensors[query], positive_mz, negative_mz, args.fragment_tolerance,
        ))

    model, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("audit requires official fine-tuned DreaMS weights")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    selections: list[dict] = []
    query_clean_embeddings = np.empty((count, embeddings.shape[1]), dtype=np.float32)
    for left in range(0, count, args.gradient_batch_size):
        right = min(left + args.gradient_batch_size, count)
        clean = torch.stack(query_tensors[left:right]).to(device)
        clean.requires_grad_(True)
        current = model(clean)
        pos = torch.as_tensor(np.stack([
            embeddings[embedding_index[representatives[index].positive_row]]
            for index in range(left, right)
        ]), device=device)
        negative_rows = [representatives[index].negative_rows for index in range(left, right)]
        max_k = max(map(len, negative_rows))
        neg = torch.zeros((right - left, max_k, embeddings.shape[1]), device=device)
        valid = torch.zeros((right - left, max_k), dtype=torch.bool, device=device)
        for row_index, rows in enumerate(negative_rows):
            values = np.stack([embeddings[embedding_index[item]] for item in rows])
            neg[row_index, :len(rows)] = torch.as_tensor(values, device=device)
            valid[row_index, :len(rows)] = True
        pos_similarity = (current * pos).sum(1)
        neg_similarity = torch.einsum("bd,bkd->bk", current, neg)
        neg_similarity = neg_similarity.masked_fill(~valid, -1e9)
        weights = torch.softmax(neg_similarity / args.softmax_temperature, dim=1).detach()
        objective = pos_similarity - (weights * neg_similarity).sum(1)
        intensity_gradient = torch.autograd.grad(objective.sum(), clean)[0][:, :, 1]
        query_clean_embeddings[left:right] = current.detach().float().cpu().numpy()
        for offset, query in enumerate(range(left, right)):
            clean_cpu = query_tensors[query]
            roles = role_arrays[query]
            gradient = intensity_gradient[offset].detach().float().cpu().numpy()
            targets = {
                "candidate_gradient": rank_gradient_targets(
                    clean_cpu, gradient, roles, args.selection_attenuation,
                    max_targets=args.candidate_gradient_top_k, protect_identity=True,
                ),
                "role_confounder": rank_role_targets(
                    clean_cpu, roles, CONFOUNDER_ONLY,
                    max_targets=args.role_confounder_top_k,
                ),
                "role_identity": rank_role_targets(
                    clean_cpu, roles, IDENTITY_ONLY, max_targets=1,
                ),
            }
            values = clean_cpu.numpy()
            for selector_family in args.selectors:
                for selector_rank, target in enumerate(targets[selector_family], start=1):
                    selector = (
                        selector_family if selector_rank == 1
                        else f"{selector_family}_r{selector_rank}"
                    )
                    controls = matched_control_tokens(
                        clean_cpu, int(target), roles, args.control_repeats,
                        stable_seed(args.seed, query, selector), same_role=True,
                    )
                    if len(controls) != args.control_repeats:
                        continue
                    selections.append({
                        "query_index": query,
                        "selector": selector,
                        "selector_family": selector_family,
                        "selector_rank": selector_rank,
                        "target": int(target),
                        "controls": controls,
                        "control_tokens": ",".join(map(str, map(int, controls))),
                        "target_role": str(ROLE_NAMES[roles[target]]),
                        "target_mz": float(values[target, 0]),
                        "target_intensity": float(values[target, 1]),
                        "target_predicted_gain": float(predicted_gain(
                            values[:, 1], gradient, args.selection_attenuation,
                        )[target]),
                        "controls_all_same_role": bool(np.all(roles[controls] == roles[target])),
                    })
        print(f"[gradient] {right:,}/{count:,}; selections={len(selections):,}", flush=True)

    # Exact implementation-identity gate: forward(clean) must reproduce the cached official query vector.
    cached_query = np.stack([embeddings[embedding_index[int(row)]] for row in cache.query_row[:count]])
    reproduction = np.sum(query_clean_embeddings * cached_query, axis=1)
    if formal and float(np.quantile(reproduction, 0.01)) < 0.999:
        raise RuntimeError(
            f"official forward/cache mismatch: p01 preservation={np.quantile(reproduction, 0.01):.6f}"
        )

    if not selections:
        raise RuntimeError("no query retained a target plus complete matched controls")
    variants = []
    for selection in selections:
        for attenuation in args.attenuations:
            variants.append(selection | {
                "attenuation": float(attenuation), "condition": "target", "repeat": -1,
                "token": int(selection["target"]),
            })
            for repeat, token in enumerate(selection["controls"]):
                variants.append(selection | {
                    "attenuation": float(attenuation), "condition": "matched_random",
                    "repeat": repeat, "token": int(token),
                })
    print(f"[interventions] variants={len(variants):,}", flush=True)

    records = []
    with torch.inference_mode():
        for left in range(0, len(variants), args.encode_batch_size):
            block = variants[left:left + args.encode_batch_size]
            spectra = torch.stack([
                attenuate_and_renormalize(
                    query_tensors[item["query_index"]], item["token"], item["attenuation"],
                ) for item in block
            ]).to(device)
            vectors = model(spectra).detach().float().cpu().numpy()
            for vector, item in zip(vectors, block):
                query = int(item["query_index"])
                _, candidate_rows, molecule_ptr = query_candidate_block(cache, query, score_column)
                candidate_vectors = embeddings[[embedding_index[int(row)] for row in candidate_rows]]
                scores = candidate_vectors @ vector
                rank, mrr, margin = strict_metrics(scores, molecule_ptr)
                base_rank = 1 + int(representatives[query].hardest_negative_score >= representatives[query].positive_score)
                records.append({
                    "query_index": query,
                    "query_row": int(cache.query_row[query]),
                    "query_ik14": str(cache.query_ik14[query]),
                    "query_formula": str(cache.query_formula[query]),
                    "has_near": bool(cache.query_has_near[query]),
                    "baseline_rank": base_rank,
                    "baseline_margin": (
                        representatives[query].positive_score - representatives[query].hardest_negative_score
                    ),
                    "selector": item["selector"],
                    "selector_family": item["selector_family"],
                    "selector_rank": item["selector_rank"],
                    "target_role": item["target_role"],
                    "target_token": int(item["target"]),
                    "control_tokens": item["control_tokens"],
                    "target_mz": item["target_mz"],
                    "target_intensity": item["target_intensity"],
                    "target_predicted_gain": item["target_predicted_gain"],
                    "controls_all_same_role": item["controls_all_same_role"],
                    "attenuation": item["attenuation"],
                    "condition": item["condition"],
                    "repeat": item["repeat"],
                    "rank": rank,
                    "mrr": mrr,
                    "margin": margin,
                })
            if (left + len(block)) % 10000 < args.encode_batch_size:
                print(f"[encode] {left + len(block):,}/{len(variants):,}", flush=True)

    del model
    gc.collect()
    results = pd.DataFrame(records)
    key = ["query_index", "selector", "attenuation"]
    target = results.loc[results["condition"] == "target"].copy()
    random = results.loc[results["condition"] == "matched_random"].groupby(key, as_index=False).agg(
        random_rank=("rank", "mean"), random_top1=("rank", lambda x: float(np.mean(np.asarray(x) == 1))),
        random_mrr=("mrr", "mean"), random_margin=("margin", "mean"),
        random_repeats=("repeat", "count"),
    )
    paired = target.merge(random, on=key, validate="one_to_one")
    paired = paired.loc[paired["random_repeats"] == args.control_repeats].copy()
    if paired.empty:
        raise RuntimeError("no intervention retained complete controls")
    paired["target_rank"] = paired["rank"]
    paired["target_mrr"] = paired["mrr"]
    paired["target_margin"] = paired["margin"]
    paired["target_margin_change"] = paired["target_margin"] - paired["baseline_margin"]
    paired["random_margin_change"] = paired["random_margin"] - paired["baseline_margin"]
    paired["target_minus_random_margin_change"] = (
        paired["target_margin_change"] - paired["random_margin_change"]
    )
    paired["baseline_state"] = np.where(paired["baseline_rank"] == 1, "baseline_correct", "baseline_wrong")
    paired["near_state"] = np.where(paired["has_near"], "near", "non_near")

    summary = summarize_paired(paired, args.bootstrap, args.seed)
    strict_same_role = paired.loc[paired["controls_all_same_role"]].copy()
    strict_same_role_summary = summarize_paired_descriptive(strict_same_role)
    active_selectors = list(dict.fromkeys(item["selector"] for item in selections))
    aggregate = {}
    for selector in active_selectors:
        for attenuation in args.attenuations:
            block = paired.loc[
                (paired["selector"] == selector)
                & np.isclose(paired["attenuation"], attenuation)
            ]
            for state, subset in (
                ("all", block),
                ("baseline_wrong", block.loc[block["baseline_rank"] > 1]),
                ("baseline_correct", block.loc[block["baseline_rank"] == 1]),
                ("near", block.loc[block["has_near"]]),
            ):
                if len(subset):
                    aggregate[f"{selector}|a={attenuation:.2f}|{state}"] = gate_summary(
                        subset, args.bootstrap,
                        args.seed + len(aggregate) * 100_000,
                    )
    main_wrong = aggregate.get("candidate_gradient|a=0.50|baseline_wrong", {})
    main_correct = aggregate.get("candidate_gradient|a=0.50|baseline_correct", {})
    main_all = aggregate.get("candidate_gradient|a=0.50|all", {})
    main_near = aggregate.get("candidate_gradient|a=0.50|near", {})
    enough = main_wrong.get("queries", 0) >= 1000
    enough_identities = main_wrong.get("identities", 0) >= 500
    positive_margin_ci = bool(main_wrong) and all(
        main_wrong[name] is not None and main_wrong[name][0] > 0
        for name in ("identity_margin_ci", "formula_margin_ci")
    )
    positive_top1_ci = bool(main_wrong) and all(
        main_wrong[name] is not None and main_wrong[name][0] > 0
        for name in ("identity_top1_ci", "formula_top1_ci")
    )
    safe_correct = bool(main_correct) and (
        main_correct["mean_target_minus_random_top1"] >= 0
        and main_correct["introduced_vs_baseline"] <= max(10, int(0.01 * main_correct["queries"]))
    )
    near_nonnegative = bool(main_near) and main_near["mean_target_minus_random_margin"] >= 0
    dose_consistent = all(
        aggregate.get(f"candidate_gradient|a={value:.2f}|baseline_wrong", {}).get(
            "mean_target_minus_random_margin", float("-inf")
        ) > 0
        for value in (0.25, 0.50)
        if value in args.attenuations
    )
    net_positive = bool(main_all) and (
        main_all["corrected_vs_baseline"] > main_all["introduced_vs_baseline"]
    )
    selector_coverage = {}
    for selector in active_selectors:
        local = paired.loc[np.isclose(paired["attenuation"], args.selection_attenuation)]
        local = local.loc[local["selector"] == selector]
        selector_coverage[selector] = {
            "queries": int(local["query_index"].nunique()),
            "identities": int(local["query_ik14"].nunique()),
            "formulas": int(local["query_formula"].nunique()),
            "strict_same_role_control_fraction": float(
                local["controls_all_same_role"].mean()
            ) if len(local) else None,
        }
    intersection = paired.loc[
        np.isclose(paired["attenuation"], args.selection_attenuation)
    ].groupby("query_index")["selector"].nunique()
    all_selector_queries = set(intersection.index[intersection == len(active_selectors)])
    intersection_frame = paired.loc[paired["query_index"].isin(all_selector_queries)].copy()
    intersection_summary = summarize_paired_descriptive(intersection_frame)
    matrix_control = {}
    if "role_identity" in active_selectors:
        identity = paired.loc[
            (paired["selector"] == "role_identity")
            & np.isclose(paired["attenuation"], args.selection_attenuation)
        ]
        identity_gate = aggregate.get(
            f"role_identity|a={args.selection_attenuation:.2f}|all", {}
        )
        identity_ci_negative = bool(identity_gate) and all(
            identity_gate.get(name) is not None and identity_gate[name][1] < 0
            for name in ("identity_margin_ci", "formula_margin_ci")
        )
        matrix_control = {
            "queries": int(len(identity)),
            "expected_direction": "attenuating identity-only evidence should reduce margin",
            "mean_target_minus_random_margin": float(
                identity["target_minus_random_margin_change"].mean()
            ) if len(identity) else None,
            "direction_control_pass": bool(
                len(identity)
                and identity["target_minus_random_margin_change"].mean() < 0
                and identity_ci_negative
            ),
            "direction_control_rule": (
                "mean effect and both identity/formula clustered margin CI upper bounds < 0"
            ),
            "identity_margin_ci": identity_gate.get("identity_margin_ci"),
            "formula_margin_ci": identity_gate.get("formula_margin_ci"),
        }
    report = {
        "status": "g8r_noise_v3_candidate_gradient_audit_complete",
        "formal": formal,
        "queries": count,
        "query_identities": int(len(set(map(str, cache.query_ik14[:count])))),
        "selection_records": int(len(selections)),
        "orthogonal_matrix": {
            "selector_families": list(args.selectors),
            "selectors": active_selectors,
            "attenuation_doses": sorted(map(float, args.attenuations)),
            "fixed_selection_attenuation": float(args.selection_attenuation),
            "same_target_and_controls_reused_across_doses": True,
            "selector_coverage_at_selection_dose": selector_coverage,
            "queries_eligible_for_every_selector": int(len(all_selector_queries)),
            "shared_eligibility_stratified_results": intersection_summary,
            "identity_only_direction_control": matrix_control,
            "selection_rule": (
                "No selector or dose is chosen from this matrix. All cells are reported; "
                "a later training action requires replicated rescue, safety, and dose consistency."
            ),
        },
        "selection_control_role_match_fraction": float(np.mean([
            item["controls_all_same_role"] for item in selections
        ])) if selections else float("nan"),
        "intervention_variants": int(len(variants)),
        "official_forward_cache_preservation": {
            "mean": float(reproduction.mean()), "p01": float(np.quantile(reproduction, 0.01)),
            "minimum": float(reproduction.min()),
        },
        "stratified_results": summary,
        "strict_same_role_control_results": strict_same_role_summary,
        "aggregate_results": aggregate,
        "gates": {
            "baseline_wrong_queries_ge_1000": enough,
            "baseline_wrong_identities_ge_500": enough_identities,
            "target_vs_random_margin_identity_and_formula_ci_positive": positive_margin_ci,
            "target_vs_random_top1_identity_and_formula_ci_positive": positive_top1_ci,
            "baseline_correct_safety": safe_correct,
            "near_margin_nonnegative": near_nonnegative,
            "attenuation_25_and_50_direction_consistent": dose_consistent,
            "absolute_corrected_gt_introduced": net_positive,
            "pass_to_training": bool(
                enough and enough_identities and positive_margin_ci and positive_top1_ci
                and safe_correct and near_nonnegative and dose_consistent and net_positive
            ),
        },
        "claim_limit": (
            "This is a P3-disjoint training-data orthogonal headroom audit. Selector coverage "
            "differs, so raw cell counts are not causal selector comparisons. It does not "
            "establish that fine-tuning improves clean-spectrum retrieval."
        ),
        "provenance": {
            "cache": str(args.cache), "cache_sha256": sha256_file(args.cache),
            "embedding_cache": str(args.embedding_cache),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "hdf5": str(args.data), "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint": str(args.official_checkpoint),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
            "noise_v3_core_sha256": sha256_file(ROOT / "tasks/noise_v3_core.py"),
        },
        "parameters": vars(args) | {"output_dir": str(args.output_dir)},
    }
    # Convert Paths before JSON serialization.
    report["parameters"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["parameters"].items()}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_v3_", dir=args.output_dir.parent))
    try:
        paired.to_csv(temporary / "paired_interventions.csv.gz", index=False)
        pd.DataFrame(selections).drop(columns=["controls"]).to_csv(
            temporary / "selected_peaks.csv.gz", index=False,
        )
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

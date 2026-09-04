"""Exact all-peak intervention scan for the official-DreaMS error space.

The script scans every real fragment token at four preregistered attenuation
doses.  Input gradients, peak roles and chemical-rule summaries are metadata;
the action outcome is always obtained from a fresh official-DreaMS forward and
the complete strict-10ppm candidate graph.

Formal mode includes every official Top-1 error plus deterministic matched
official-correct safety controls.  It trains no model and touches no P3 query
identity.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
    IDENTITY_ONLY, ROLE_NAMES, attenuate_and_renormalize,
    candidate_peak_roles_from_mz, candidate_representatives, predicted_gain,
)
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


DEFAULT_CACHE = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_EMBEDDINGS = ROOT / "data/validation/g8r_p2_official_embeddings.npz"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCHITECTURE = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_SIGNATURES = ROOT / "data/validation/g8r_real_error_analysis/query_error_signatures.csv.gz"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--error-signatures", type=Path, default=DEFAULT_SIGNATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--top-k-negatives", type=int, default=5)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--selection-attenuation", type=float, default=0.50)
    parser.add_argument(
        "--attenuations", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00],
    )
    parser.add_argument("--safety-controls-per-error", type=int, default=3)
    parser.add_argument("--maximum-control-reuse", type=int, default=3)
    parser.add_argument("--max-errors", type=int, default=0, help="Smoke only; 0 is formal all errors")
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    with np.load(path) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(rows) != len(embeddings):
        raise RuntimeError("embedding cache is malformed")
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("embedding cache has duplicate rows")
    return rows, embeddings, {int(row): position for position, row in enumerate(rows)}


def query_candidate_block(cache: Cache, query: int, score_column: int):
    molecule_left, molecule_right = map(int, cache.query_ptr[query:query + 2])
    pair_left = int(cache.molecule_ptr[molecule_left])
    pair_right = int(cache.molecule_ptr[molecule_right])
    local_ptr = cache.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
    return (
        cache.features[pair_left:pair_right, score_column],
        cache.pair_candidate_row[pair_left:pair_right],
        local_ptr,
        molecule_left,
    )


def strict_detail(
    scores: np.ndarray, rows: np.ndarray, molecule_ptr: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=float)
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    molecule_scores = np.maximum.reduceat(values, ptr[:-1])
    positive = float(molecule_scores[0])
    negative_local = int(np.argmax(molecule_scores[1:])) + 1
    negative = float(molecule_scores[negative_local])
    rank = 1 + int(np.sum(molecule_scores[1:] >= positive))
    left, right = map(int, ptr[negative_local:negative_local + 2])
    best_pair = left + int(np.argmax(values[left:right]))
    return {
        "rank": rank,
        "mrr": 1.0 / rank,
        "positive": positive,
        "negative": negative,
        "margin": positive - negative,
        "adversarial_molecule_local": negative_local,
        "adversarial_pair_row": int(rows[best_pair]),
    }


def top_reference_rows(
    scores: np.ndarray, rows: np.ndarray, ptr: np.ndarray, molecule: int, count: int = 3,
) -> np.ndarray:
    left, right = map(int, ptr[molecule:molecule + 2])
    order = np.argsort(-np.asarray(scores[left:right]), kind="mergesort")[:count] + left
    return np.asarray(rows[order], dtype=np.int64)


def raw_peak_count(spectrum: np.ndarray) -> int:
    array = np.asarray(spectrum)
    if array.ndim != 2:
        raise ValueError(f"unexpected spectrum shape {array.shape}")
    intensity = array[1] if array.shape[0] == 2 else array[:, 1]
    return int(np.sum(np.isfinite(intensity) & (intensity > 0)))


def build_baseline_table(cache: Cache, score_column: int, data: Path) -> pd.DataFrame:
    records: list[dict] = []
    with h5py.File(data, "r") as handle:
        for query in range(cache.n_queries):
            scores, rows, ptr, molecule_left = query_candidate_block(cache, query, score_column)
            detail = strict_detail(scores, rows, ptr)
            query_row = int(cache.query_row[query])
            records.append({
                "query_index": query,
                "query_row": query_row,
                "query_ik14": str(cache.query_ik14[query]),
                "query_formula": str(cache.query_formula[query]),
                "has_near": bool(cache.query_has_near[query]),
                "baseline_rank": int(detail["rank"]),
                "baseline_margin": float(detail["margin"]),
                "candidate_molecules": int(len(ptr) - 1),
                "peak_count": raw_peak_count(np.asarray(handle["spectrum"][query_row])),
                "baseline_adversarial_molecule_local": int(detail["adversarial_molecule_local"]),
                "baseline_adversarial_pair_row": int(detail["adversarial_pair_row"]),
                "baseline_adversarial_mces_grade": int(
                    cache.molecule_mces_grade[
                        molecule_left + int(detail["adversarial_molecule_local"])
                    ]
                ),
            })
    return pd.DataFrame(records)


def select_matched_controls(
    baseline: pd.DataFrame, errors: pd.DataFrame, per_error: int, reuse_cap: int,
) -> pd.DataFrame:
    """Greedy deterministic matching with recorded fallback and reuse."""
    correct = baseline.loc[baseline["baseline_rank"] == 1].copy()
    if correct.empty:
        raise RuntimeError("no official-correct controls")
    numeric = ["baseline_margin", "candidate_molecules", "peak_count"]
    transformed = correct[numeric].astype(float).copy()
    transformed["candidate_molecules"] = np.log1p(transformed["candidate_molecules"])
    transformed["peak_count"] = np.log1p(transformed["peak_count"])
    center = transformed.median()
    scale = (transformed.quantile(0.75) - transformed.quantile(0.25)).clip(lower=1e-6)
    correct_z = (transformed - center) / scale
    reuse: defaultdict[int, int] = defaultdict(int)
    matches: list[dict] = []
    for error in errors.sort_values("query_index").itertuples(index=False):
        masks = [
            (correct["query_formula"] == error.query_formula) & (correct["has_near"] == error.has_near),
            correct["query_formula"] == error.query_formula,
            correct["has_near"] == error.has_near,
            pd.Series(True, index=correct.index),
        ]
        labels = ["formula+near", "formula", "near", "global"]
        chosen: list[int] = []
        error_values = pd.Series({
            "baseline_margin": float(error.baseline_margin),
            "candidate_molecules": math.log1p(float(error.candidate_molecules)),
            "peak_count": math.log1p(float(error.peak_count)),
        })
        error_z = (error_values - center) / scale
        for level, mask in zip(labels, masks):
            pool = correct.loc[mask & (correct["query_ik14"] != error.query_ik14)]
            pool = pool.loc[~pool["query_index"].isin(chosen)]
            if pool.empty:
                continue
            distance = ((correct_z.loc[pool.index] - error_z) ** 2).sum(axis=1)
            order = sorted(
                pool.index,
                key=lambda index: (
                    reuse[int(correct.at[index, "query_index"])] >= reuse_cap,
                    float(distance.at[index]) + 0.10 * reuse[int(correct.at[index, "query_index"])] ,
                    int(correct.at[index, "query_index"]),
                ),
            )
            for index in order:
                control_query = int(correct.at[index, "query_index"])
                if reuse[control_query] >= reuse_cap and len(chosen) < per_error:
                    continue
                chosen.append(control_query)
                reuse[control_query] += 1
                matches.append({
                    "error_query_index": int(error.query_index),
                    "control_query_index": control_query,
                    "match_level": level,
                    "match_distance": float(distance.at[index]),
                    "control_reuse_after_selection": int(reuse[control_query]),
                })
                if len(chosen) == per_error:
                    break
            if len(chosen) == per_error:
                break
        if len(chosen) < per_error:
            raise RuntimeError(
                f"error query {error.query_index} retained only {len(chosen)}/{per_error} controls"
            )
    return pd.DataFrame(matches)


def create_dataset(handle: h5py.File, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    chunks = (min(max(len(array), 1), 262_144),) + array.shape[1:]
    handle.create_dataset(
        name, data=array, compression="gzip", compression_opts=4, shuffle=True,
        chunks=chunks,
    )


def main() -> None:
    args = parse_args()
    formal = args.max_errors == 0
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
    doses = np.asarray(args.attenuations, dtype=np.float32)
    if len(doses) != len(np.unique(doses)) or np.any((doses <= 0) | (doses > 1)):
        raise ValueError("attenuation doses must be unique and in (0, 1]")
    if args.safety_controls_per_error < 1 or args.maximum_control_reuse < 1:
        raise ValueError("control parameters must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cache = Cache(args.cache)
    if formal and cache.n_queries != 23876:
        raise RuntimeError(f"formal A4 expects 23,876 queries, observed {cache.n_queries}")
    p3 = load_p3_identities(args.p3_dir)
    overlap = set(map(str, cache.query_ik14)) & p3
    if overlap:
        raise RuntimeError(f"P3 leakage: {len(overlap)} identities")
    score_column = cache.feature_names.index("dreams_similarity")
    embedding_rows, embeddings, embedding_index = load_embeddings(args.embedding_cache)
    needed_graph_rows = set(map(int, cache.query_row)) | set(map(int, cache.pair_candidate_row))
    missing = needed_graph_rows - set(embedding_index)
    if missing:
        raise RuntimeError(f"embedding cache misses {len(missing)} candidate-graph rows")

    print("[A4] building official baseline table", flush=True)
    baseline = build_baseline_table(cache, score_column, args.data)
    errors = baseline.loc[baseline["baseline_rank"] > 1].copy()
    if formal and len(errors) != 1805:
        raise RuntimeError(f"formal A4 expects 1,805 official errors, observed {len(errors)}")
    if args.max_errors:
        errors = errors.sort_values("query_index").head(args.max_errors).copy()
    matches = select_matched_controls(
        baseline, errors, args.safety_controls_per_error, args.maximum_control_reuse,
    )
    control_queries = np.sort(matches["control_query_index"].unique())
    selected_queries = np.concatenate((
        errors["query_index"].to_numpy(np.int64), control_queries.astype(np.int64),
    ))
    if len(selected_queries) != len(np.unique(selected_queries)):
        raise RuntimeError("error/control scan query overlap")
    selected = baseline.set_index("query_index").loc[selected_queries].reset_index()
    selected["scan_kind"] = np.where(selected["baseline_rank"] > 1, "official_error", "safety_control")
    selected["matched_error_count"] = selected["query_index"].map(
        matches.groupby("control_query_index").size(),
    ).fillna(0).astype(int)
    print(
        f"[A4] errors={len(errors):,}; unique controls={len(control_queries):,}; "
        f"scan queries={len(selected):,}", flush=True,
    )

    # Optional mechanism labels are descriptive joins, never action labels.
    if args.error_signatures.is_file():
        signatures = pd.read_csv(args.error_signatures)
        keep = [column for column in (
            "query_index", "score_error_family", "positive_deficit", "negative_excess",
            "shared_major_peak_screen", "neutral_loss_convergence_screen",
            "rules_favor_positive", "rules_favor_wrong",
        ) if column in signatures.columns]
        if "query_index" in keep:
            selected = selected.merge(
                signatures[keep].drop_duplicates("query_index"), on="query_index", how="left",
                validate="one_to_one",
            )

    representatives = []
    reference_rows: set[int] = set()
    for query in selected_queries:
        scores, rows, ptr, _ = query_candidate_block(cache, int(query), score_column)
        rep = candidate_representatives(scores, rows, ptr, args.top_k_negatives)
        representatives.append(rep)
        reference_rows.update(map(int, top_reference_rows(scores, rows, ptr, 0, 3)))
        negative_molecule = int(selected.loc[
            selected["query_index"] == query, "baseline_adversarial_molecule_local"
        ].iloc[0])
        reference_rows.update(map(int, top_reference_rows(scores, rows, ptr, negative_molecule, 3)))

    tensor_cache: dict[int, torch.Tensor] = {}
    required_tensors = set(map(int, selected["query_row"])) | reference_rows
    with h5py.File(args.data, "r") as handle:
        for position, row in enumerate(sorted(required_tensors), start=1):
            tensor_cache[row] = preprocess_spectrum(
                np.asarray(handle["spectrum"][row]), float(handle["precursor_mz"][row]),
                args.n_highest_peaks,
            )
            if position % 2000 == 0 or position == len(required_tensors):
                print(f"[A4 spectra] {position:,}/{len(required_tensors):,}", flush=True)
    query_tensors = [tensor_cache[int(row)] for row in selected["query_row"]]

    roles: list[np.ndarray] = []
    for position, query in enumerate(selected_queries):
        scores, rows, ptr, _ = query_candidate_block(cache, int(query), score_column)
        positive_rows = top_reference_rows(scores, rows, ptr, 0, 3)
        negative_molecule = int(selected.iloc[position]["baseline_adversarial_molecule_local"])
        negative_rows = top_reference_rows(scores, rows, ptr, negative_molecule, 3)
        positive_mz = np.concatenate([
            tensor_cache[int(row)][1:, 0].numpy()[tensor_cache[int(row)][1:, 1].numpy() > 0]
            for row in positive_rows
        ])
        negative_mz = np.concatenate([
            tensor_cache[int(row)][1:, 0].numpy()[tensor_cache[int(row)][1:, 1].numpy() > 0]
            for row in negative_rows
        ])
        roles.append(candidate_peak_roles_from_mz(
            query_tensors[position], positive_mz, negative_mz, args.fragment_tolerance,
        ))

    model, checkpoint_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if checkpoint_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("A4 requires official fine-tuned DreaMS weights")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    gradients: list[np.ndarray] = []
    clean_vectors = np.empty((len(selected), embeddings.shape[1]), dtype=np.float32)
    for left in range(0, len(selected), args.gradient_batch_size):
        right = min(left + args.gradient_batch_size, len(selected))
        clean = torch.stack(query_tensors[left:right]).to(device)
        clean.requires_grad_(True)
        current = model(clean)
        positive = torch.as_tensor(np.stack([
            embeddings[embedding_index[representatives[index].positive_row]]
            for index in range(left, right)
        ]), device=device)
        negative_rows = [representatives[index].negative_rows for index in range(left, right)]
        max_k = max(map(len, negative_rows))
        negative = torch.zeros((right - left, max_k, embeddings.shape[1]), device=device)
        valid = torch.zeros((right - left, max_k), dtype=torch.bool, device=device)
        for offset, rows in enumerate(negative_rows):
            negative[offset, :len(rows)] = torch.as_tensor(np.stack([
                embeddings[embedding_index[int(row)]] for row in rows
            ]), device=device)
            valid[offset, :len(rows)] = True
        pos_similarity = (current * positive).sum(1)
        neg_similarity = torch.einsum("bd,bkd->bk", current, negative).masked_fill(~valid, -1e9)
        weights = torch.softmax(neg_similarity / args.softmax_temperature, dim=1).detach()
        objective = pos_similarity - (weights * neg_similarity).sum(1)
        gradient = torch.autograd.grad(objective.sum(), clean)[0][:, :, 1]
        clean_vectors[left:right] = current.detach().float().cpu().numpy()
        gradients.extend(gradient.detach().float().cpu().numpy())
        print(f"[A4 gradient] {right:,}/{len(selected):,}", flush=True)

    cached = np.stack([
        embeddings[embedding_index[int(row)]] for row in selected["query_row"]
    ])
    reproduction = np.sum(cached * clean_vectors, axis=1)
    if formal and float(np.quantile(reproduction, 0.01)) < 0.999:
        raise RuntimeError(f"official forward/cache mismatch p01={np.quantile(reproduction, 0.01)}")

    action_query: list[int] = []
    action_token: list[int] = []
    action_role: list[int] = []
    action_mz: list[float] = []
    action_intensity: list[float] = []
    action_gradient: list[float] = []
    action_gain: list[float] = []
    action_gradient_rank: list[int] = []
    action_policy_eligible: list[bool] = []
    query_action_ptr = [0]
    for position, (spectrum, gradient, role) in enumerate(zip(query_tensors, gradients, roles)):
        values = spectrum.numpy()
        tokens = np.flatnonzero(
            (np.arange(len(values)) > 0) & (values[:, 0] > 0) & (values[:, 1] > 0)
        )
        gain = predicted_gain(values[:, 1], gradient, args.selection_attenuation)
        eligible = tokens[role[tokens] != IDENTITY_ONLY]
        order = sorted(eligible, key=lambda token: (-float(gain[token]), int(token)))
        rank = {int(token): index + 1 for index, token in enumerate(order)}
        for token in tokens:
            action_query.append(position)
            action_token.append(int(token))
            action_role.append(int(role[token]))
            action_mz.append(float(values[token, 0]))
            action_intensity.append(float(values[token, 1]))
            action_gradient.append(float(gradient[token]))
            action_gain.append(float(gain[token]))
            action_gradient_rank.append(int(rank.get(int(token), -1)))
            action_policy_eligible.append(bool(role[token] != IDENTITY_ONLY))
        query_action_ptr.append(len(action_query))
    action_query_array = np.asarray(action_query, dtype=np.int32)
    action_token_array = np.asarray(action_token, dtype=np.int16)
    n_actions = len(action_query_array)
    n_variants = n_actions * len(doses)
    print(f"[A4 actions] actions={n_actions:,}; variants={n_variants:,}", flush=True)

    result_rank = np.empty(n_variants, dtype=np.int16)
    result_mrr = np.empty(n_variants, dtype=np.float32)
    result_positive = np.empty(n_variants, dtype=np.float32)
    result_negative = np.empty(n_variants, dtype=np.float32)
    result_margin = np.empty(n_variants, dtype=np.float32)
    result_adversarial_molecule = np.empty(n_variants, dtype=np.int16)
    result_adversarial_pair_row = np.empty(n_variants, dtype=np.int64)
    candidate_rows_by_scan: list[np.ndarray] = []
    candidate_ptr_by_scan: list[np.ndarray] = []
    candidate_embedding_indices_by_scan: list[np.ndarray] = []
    for graph_query in selected_queries:
        _, candidate_rows, ptr, _ = query_candidate_block(cache, int(graph_query), score_column)
        candidate_rows_by_scan.append(np.asarray(candidate_rows, dtype=np.int64))
        candidate_ptr_by_scan.append(np.asarray(ptr, dtype=np.int64))
        candidate_embedding_indices_by_scan.append(np.asarray([
            embedding_index[int(row)] for row in candidate_rows
        ], dtype=np.int64))
    with torch.inference_mode():
        for left in range(0, n_variants, args.encode_batch_size):
            right = min(left + args.encode_batch_size, n_variants)
            action_indices = np.arange(left, right, dtype=np.int64) // len(doses)
            dose_indices = np.arange(left, right, dtype=np.int64) % len(doses)
            spectra = torch.stack([
                attenuate_and_renormalize(
                    query_tensors[int(action_query_array[action])],
                    int(action_token_array[action]), float(doses[dose]),
                )
                for action, dose in zip(action_indices, dose_indices)
            ]).to(device)
            vectors = model(spectra).detach().float().cpu().numpy()
            scan_queries = action_query_array[action_indices]
            for scan_query in np.unique(scan_queries):
                local_offsets = np.flatnonzero(scan_queries == scan_query)
                candidate_rows = candidate_rows_by_scan[int(scan_query)]
                ptr = candidate_ptr_by_scan[int(scan_query)]
                candidate_vectors = embeddings[
                    candidate_embedding_indices_by_scan[int(scan_query)]
                ]
                score_matrix = vectors[local_offsets] @ candidate_vectors.T
                for local_offset, scores in zip(local_offsets, score_matrix):
                    output_index = left + int(local_offset)
                    detail = strict_detail(scores, candidate_rows, ptr)
                    result_rank[output_index] = int(detail["rank"])
                    result_mrr[output_index] = float(detail["mrr"])
                    result_positive[output_index] = float(detail["positive"])
                    result_negative[output_index] = float(detail["negative"])
                    result_margin[output_index] = float(detail["margin"])
                    result_adversarial_molecule[output_index] = int(
                        detail["adversarial_molecule_local"]
                    )
                    result_adversarial_pair_row[output_index] = int(
                        detail["adversarial_pair_row"]
                    )
            if right % 10000 < args.encode_batch_size or right == n_variants:
                print(f"[A4 exact] {right:,}/{n_variants:,}", flush=True)

    del model
    gc.collect()
    selected["scan_position"] = np.arange(len(selected), dtype=np.int64)
    selected["clean_embedding_preservation"] = reproduction
    temporary = Path(tempfile.mkdtemp(prefix="noise_v3_a4_", dir=args.output_dir.parent))
    try:
        selected.to_csv(temporary / "scan_queries.csv.gz", index=False, compression="gzip")
        matches.to_csv(temporary / "safety_control_matches.csv.gz", index=False, compression="gzip")
        with h5py.File(temporary / "exact_peak_scan.h5", "w") as output:
            output.attrs["status"] = "noise_v3_a4_exact_peak_scan_complete"
            output.attrs["attenuations_json"] = json.dumps([float(x) for x in doses])
            output.attrs["query_count"] = len(selected)
            output.attrs["error_count"] = len(errors)
            output.attrs["action_count"] = n_actions
            output.attrs["variant_count"] = n_variants
            create_dataset(output, "query_action_ptr", np.asarray(query_action_ptr, dtype=np.int64))
            create_dataset(output, "action_query", action_query_array)
            create_dataset(output, "action_token", action_token_array)
            create_dataset(output, "action_role", np.asarray(action_role, dtype=np.int8))
            create_dataset(output, "action_mz", np.asarray(action_mz, dtype=np.float32))
            create_dataset(output, "action_intensity", np.asarray(action_intensity, dtype=np.float32))
            create_dataset(output, "action_gradient", np.asarray(action_gradient, dtype=np.float32))
            create_dataset(output, "action_predicted_gain", np.asarray(action_gain, dtype=np.float32))
            create_dataset(output, "action_gradient_rank", np.asarray(action_gradient_rank, dtype=np.int16))
            create_dataset(output, "action_policy_eligible", np.asarray(action_policy_eligible, dtype=np.bool_))
            create_dataset(output, "result_rank", result_rank)
            create_dataset(output, "result_mrr", result_mrr)
            create_dataset(output, "result_positive", result_positive)
            create_dataset(output, "result_negative", result_negative)
            create_dataset(output, "result_margin", result_margin)
            create_dataset(output, "result_adversarial_molecule_local", result_adversarial_molecule)
            create_dataset(output, "result_adversarial_pair_row", result_adversarial_pair_row)
        report = {
            "status": "noise_v3_a4_exact_peak_scan_complete",
            "formal": formal,
            "full_graph_queries": int(cache.n_queries),
            "official_errors_scanned": int(len(errors)),
            "unique_safety_controls": int(len(control_queries)),
            "scan_queries": int(len(selected)),
            "fragment_actions": int(n_actions),
            "exact_variants": int(n_variants),
            "attenuations": [float(x) for x in doses],
            "policy_eligible_actions": int(np.sum(action_policy_eligible)),
            "identity_only_negative_control_actions": int(np.sum(~np.asarray(action_policy_eligible))),
            "control_matching": {
                "requested_per_error": args.safety_controls_per_error,
                "maximum_reuse": args.maximum_control_reuse,
                "match_level_counts": {
                    str(key): int(value)
                    for key, value in matches["match_level"].value_counts().items()
                },
                "errors_with_complete_controls": int(matches.groupby("error_query_index").size().eq(
                    args.safety_controls_per_error
                ).sum()),
            },
            "official_forward_cache_preservation": {
                "mean": float(np.mean(reproduction)),
                "p01": float(np.quantile(reproduction, 0.01)),
                "minimum": float(np.min(reproduction)),
            },
            "claim_limit": (
                "Exact per-action outcome headroom on P3-disjoint training data. It is not a "
                "deployable policy and not evidence that fine-tuning improves retrieval."
            ),
            "provenance": {
                "cache_sha256": sha256_file(args.cache),
                "embedding_cache_sha256": sha256_file(args.embedding_cache),
                "hdf5_sha256": sha256_file(args.data),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

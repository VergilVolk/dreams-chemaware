#!/usr/bin/env python
"""E2-M1: execute the frozen corrective/noise negative-control matrix.

Every target action is compared with three deterministic peak-count,
intensity, m/z and (when relevant) candidate-role matched controls.  This stage
uses the official frozen encoder and candidate library; it trains no model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_noise_v3_a4_exact_peak_scan import (  # noqa: E402
    load_embeddings, query_candidate_block, strict_detail,
)
from build_g8r_real_error_atlas import Cache, load_p3_identities, sha256_file  # noqa: E402
from noise_v3_core import IDENTITY_ONLY, attenuate_sequence, stable_seed  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_matrix_manifest"))
    parser.add_argument("--e1-dir", type=Path, default=Path("data/validation/g8r_noise_final_e1_empirical_calibration"))
    parser.add_argument("--a4-dir", type=Path, default=Path("data/validation/g8r_noise_v3_a4_exact_peak_scan"))
    parser.add_argument("--cache", type=Path, default=Path("data/validation/g8r_error_atlas_listwise_cache.npz"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/validation/g8r_p2_official_embeddings.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--official-checkpoint", type=Path, default=Path("data/e1/official_embedding_slim.pt"))
    parser.add_argument("--architecture-checkpoint", type=Path, default=Path("dreams/models/pretrained/ssl_model_server.pt"))
    parser.add_argument("--p3-dir", type=Path, default=Path("data/validation/g8r_p3_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_corrective_scan"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-clean-preservation-p01", type=float, default=0.999)
    parser.add_argument("--maximum-clean-rank-mismatch-fraction", type=float, default=0.001)
    parser.add_argument("--max-scan-queries", type=int, default=0, help="smoke only")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def decode(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def closest_cluster_probabilities(
    mz: np.ndarray, cluster_mz: np.ndarray, probabilities: np.ndarray, tolerance: float,
) -> np.ndarray:
    output = np.full(len(mz), np.nan, dtype=np.float32)
    if not len(cluster_mz):
        return output
    order = np.argsort(cluster_mz, kind="mergesort")
    cmz, prob = cluster_mz[order], probabilities[order]
    position = np.searchsorted(cmz, mz)
    left = np.clip(position - 1, 0, len(cmz) - 1)
    right = np.clip(position, 0, len(cmz) - 1)
    choose_right = np.abs(cmz[right] - mz) < np.abs(cmz[left] - mz)
    nearest = np.where(choose_right, right, left)
    distance = np.abs(cmz[nearest] - mz)
    output[distance <= tolerance] = prob[nearest[distance <= tolerance]]
    return output


def matched_control_sequences(
    spectrum: torch.Tensor,
    targets: np.ndarray,
    roles: np.ndarray,
    repeats: int,
    seed: int,
    same_role: bool,
) -> tuple[list[np.ndarray], list[str]]:
    """Choose independent matched paths and disclose any role-match fallback."""
    values = spectrum.detach().cpu().numpy()
    valid_all = np.flatnonzero(
        (np.arange(len(values)) > 0) & (values[:, 0] > 0) & (values[:, 1] > 0)
    )
    target_set = set(map(int, targets))
    output: list[list[int]] = [[] for _ in range(repeats)]
    levels = ["role_intensity_mz" if same_role else "intensity_mz" for _ in range(repeats)]
    log_intensity = np.log(np.clip(values[:, 1].astype(float), 1e-8, None))
    mz_scale = max(float(np.std(values[valid_all, 0])), 25.0)
    for target_position, target in enumerate(map(int, targets)):
        for repeat in range(repeats):
            blocked = target_set | set(output[repeat])
            pool = np.asarray([token for token in valid_all if int(token) not in blocked], dtype=np.int64)
            if same_role:
                role_pool = pool[roles[pool] == roles[target]]
                if len(role_pool):
                    pool = role_pool
                else:
                    levels[repeat] = "intensity_mz_role_fallback"
            if not len(pool):
                return [], []
            cost = (
                4.0 * np.abs(log_intensity[pool] - log_intensity[target])
                + 0.15 * np.abs(values[pool, 0] - values[target, 0]) / mz_scale
            )
            rng = np.random.default_rng(stable_seed(seed, target_position, repeat, target))
            cost += rng.gumbel(0.0, 1e-8, len(cost))
            output[repeat].append(int(pool[int(np.argmin(cost))]))
    return [np.asarray(path, dtype=np.int16) for path in output], levels


def action_targets(
    selector: str,
    step: int,
    dose: float,
    spectrum: torch.Tensor,
    tokens: np.ndarray,
    roles: np.ndarray,
    gains: np.ndarray,
    gradient_ranks: np.ndarray,
    missingness: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, bool]:
    values = spectrum.detach().cpu().numpy()
    real = tokens[(tokens > 0) & (values[tokens, 0] > 0) & (values[tokens, 1] > 0)]
    if selector == "candidate_gradient":
        choices = real[(roles[real] != IDENTITY_ONLY) & (gains[real] > 0) & (gradient_ranks[real] > 0)]
        choices = choices[np.lexsort((choices, gradient_ranks[choices], -gains[choices]))]
        return choices[:step].astype(np.int16), True
    if selector in {"role_confounder", "role_shared"}:
        wanted = 1 if selector == "role_confounder" else 2
        choices = real[roles[real] == wanted]
        choices = choices[np.lexsort((choices, -values[choices, 1].astype(float)))]
        return choices[:step].astype(np.int16), True
    if selector == "uniform_random":
        count = max(1, int(math.ceil(float(dose) * len(real))))
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(real, size=min(count, len(real)), replace=False)).astype(np.int16), False

    mapped = real[np.isfinite(missingness[real]) & (missingness[real] > 0)]
    count = max(1, int(math.ceil(float(dose) * len(real))))
    if selector == "empirical_conditional_missingness":
        score = missingness[mapped]
        same_role = False
    elif selector == "conditional_missingness_x_confounder":
        mapped = mapped[roles[mapped] == 1]
        score = missingness[mapped] * values[mapped, 1].astype(float)
        same_role = True
    elif selector == "conditional_missingness_x_positive_gradient":
        mapped = mapped[(roles[mapped] != IDENTITY_ONLY) & (gains[mapped] > 0)]
        if not len(mapped):
            return np.empty(0, dtype=np.int16), True
        gain_scale = gains[mapped] / max(float(np.max(gains[mapped])), 1e-12)
        score = missingness[mapped] * gain_scale
        same_role = True
    else:
        raise ValueError(f"unsupported E2 corrective selector: {selector}")
    if not len(mapped):
        return np.empty(0, dtype=np.int16), same_role
    order = np.lexsort((mapped, -values[mapped, 1].astype(float), -score))
    return mapped[order[: min(count, len(mapped))]].astype(np.int16), same_role


def main() -> None:
    args = parse_args()
    for name in (
        "manifest_dir", "e1_dir", "a4_dir", "cache", "embedding_cache", "data",
        "official_checkpoint", "architecture_checkpoint", "p3_dir", "output_dir",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E2 scan: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest_path = args.manifest_dir / "e2_manifest.json"
    cells_path = args.manifest_dir / "e2_preregistered_cells.csv"
    e1_report_path = args.e1_dir / "e1_report.json"
    consensus_path = args.e1_dir / "consensus_peak_calibration.csv.gz"
    pairwise_path = args.e1_dir / "pairwise_spectrum_variation.csv.gz"
    a4_h5_path = args.a4_dir / "exact_peak_scan.h5"
    scan_path = args.a4_dir / "scan_queries.csv.gz"
    for path in (
        manifest_path, cells_path, e1_report_path, consensus_path, pairwise_path,
        a4_h5_path, scan_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    e1_report = json.loads(e1_report_path.read_text(encoding="utf-8"))
    if not manifest.get("formal") or not e1_report.get("formal") or not e1_report.get("pass_to_e2"):
        raise RuntimeError("formal E2/E1 contracts are not passing")
    cells = pd.read_csv(cells_path)
    cells = cells.loc[cells["arm"].isin(["corrective", "negative_control"])].reset_index(drop=True)
    if len(cells) != manifest["corrective_cells"] + manifest["negative_control_cells"]:
        raise RuntimeError("E2 corrective cell count changed")

    cache = Cache(args.cache)
    if cache.n_queries != 23876:
        raise RuntimeError(f"formal E2 expects 23,876 graph queries, got {cache.n_queries}")
    overlap = set(map(str, cache.query_ik14)) & load_p3_identities(args.p3_dir)
    if overlap:
        raise RuntimeError(f"P3 leakage: {len(overlap)} identities")
    score_column = cache.feature_names.index("dreams_similarity")
    embedding_rows, embeddings, embedding_index = load_embeddings(args.embedding_cache)
    needed = set(map(int, cache.query_row)) | set(map(int, cache.pair_candidate_row))
    if needed - set(embedding_index):
        raise RuntimeError("official embedding cache does not cover the candidate graph")

    scan = pd.read_csv(scan_path).sort_values("scan_position", kind="stable").reset_index(drop=True)
    if args.max_scan_queries:
        scan = scan.head(args.max_scan_queries).copy()
    if scan["scan_position"].tolist() != list(range(len(scan))):
        raise RuntimeError("smoke mode must retain the leading contiguous A4 scan positions")
    with h5py.File(a4_h5_path, "r") as handle:
        ptr = np.asarray(handle["query_action_ptr"][: len(scan) + 1], dtype=np.int64)
        n_actions = int(ptr[-1])
        action_query = np.asarray(handle["action_query"][:n_actions], dtype=np.int32)
        action_token = np.asarray(handle["action_token"][:n_actions], dtype=np.int16)
        action_role = np.asarray(handle["action_role"][:n_actions], dtype=np.int8)
        action_gain = np.asarray(handle["action_predicted_gain"][:n_actions], dtype=np.float32)
        action_gradient_rank = np.asarray(handle["action_gradient_rank"][:n_actions], dtype=np.int16)
    if len(ptr) != len(scan) + 1 or (len(action_query) and int(action_query.max()) >= len(scan)):
        raise RuntimeError("A4 action cache and selected scan queries disagree")
    for position in range(len(scan)):
        left, right = map(int, ptr[position:position + 2])
        if right < left or (right > left and not np.all(action_query[left:right] == position)):
            raise RuntimeError(f"A4 action pointer corruption at scan position {position}")

    consensus = pd.read_csv(
        consensus_path,
        usecols=["ik14", "adduct", "mz", "dropout_probability"],
    )
    consensus_groups = {
        (str(ik14), str(adduct)): (
            part["mz"].to_numpy(dtype=np.float64),
            part["dropout_probability"].to_numpy(dtype=np.float32),
        )
        for (ik14, adduct), part in consensus.groupby(["ik14", "adduct"], sort=False)
    }
    pairwise = pd.read_csv(
        pairwise_path, usecols=["ik14", "adduct", "relation", "reliable_for_dose"],
    )
    reliable_mask = pairwise["reliable_for_dose"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1"}
    )
    reliable = pairwise.loc[reliable_mask]
    relation_supported_groups = set(zip(
        reliable["ik14"].astype(str), reliable["adduct"].astype(str),
        reliable["relation"].astype(str),
    ))

    tensors: list[torch.Tensor] = []
    adducts: list[str] = []
    with h5py.File(args.data, "r") as handle:
        for position, row in enumerate(scan["query_row"].astype(int), start=1):
            tensors.append(preprocess_spectrum(
                np.asarray(handle["spectrum"][row]), float(handle["precursor_mz"][row]),
                args.n_highest_peaks,
            ))
            adducts.append(decode(handle["adduct"][row]))
            if position % 1000 == 0 or position == len(scan):
                print(f"[E2 spectra] {position:,}/{len(scan):,}", flush=True)

    # Expand action arrays into token-index lookup tables per query.
    roles_by_query: list[np.ndarray] = []
    gains_by_query: list[np.ndarray] = []
    ranks_by_query: list[np.ndarray] = []
    missingness_by_query: list[np.ndarray] = []
    for position, tensor in enumerate(tensors):
        size = len(tensor)
        roles = np.full(size, -1, dtype=np.int8)
        gains = np.full(size, -np.inf, dtype=np.float32)
        ranks = np.full(size, -1, dtype=np.int16)
        left, right = map(int, ptr[position:position + 2])
        tokens = action_token[left:right].astype(int)
        roles[tokens] = action_role[left:right]
        gains[tokens] = action_gain[left:right]
        ranks[tokens] = action_gradient_rank[left:right]
        group = consensus_groups.get((str(scan.iloc[position]["query_ik14"]), adducts[position]))
        missingness = np.full(size, np.nan, dtype=np.float32)
        if group is not None:
            mz = tensor[:, 0].numpy().astype(float)
            missingness = closest_cluster_probabilities(mz, group[0], group[1], args.fragment_tolerance)
            missingness[0] = np.nan
        roles_by_query.append(roles)
        gains_by_query.append(gains)
        ranks_by_query.append(ranks)
        missingness_by_query.append(missingness)

    # One plan holds one target and its three matched controls.
    plans: list[dict[str, Any]] = []
    variants: list[tuple[int, int, np.ndarray, float]] = []
    for position, row in enumerate(scan.itertuples(index=False)):
        left, right = map(int, ptr[position:position + 2])
        tokens = action_token[left:right].astype(np.int64)
        for cell in cells.itertuples(index=False):
            relation = str(cell.acquisition_relation)
            if relation != "not_applicable" and (
                str(row.query_ik14), adducts[position], relation
            ) not in relation_supported_groups:
                continue
            targets, same_role = action_targets(
                str(cell.selector), int(cell.step), float(cell.dose), tensors[position], tokens,
                roles_by_query[position], gains_by_query[position], ranks_by_query[position],
                missingness_by_query[position], stable_seed(args.seed, position, cell.cell_id),
            )
            if not len(targets):
                continue
            controls, control_match_levels = matched_control_sequences(
                tensors[position], targets, roles_by_query[position],
                int(cell.matched_random_controls),
                stable_seed(args.seed, "controls", position, cell.cell_id), same_role,
            )
            if str(cell.selector) == "uniform_random":
                controls = []
                control_match_levels = []
            elif len(controls) != int(cell.matched_random_controls):
                continue
            plan_index = len(plans)
            plans.append({
                "cell_id": str(cell.cell_id), "arm": str(cell.arm),
                "selector": str(cell.selector), "operator": str(cell.operator),
                "acquisition_relation": str(cell.acquisition_relation),
                "dose": float(cell.dose), "step": int(cell.step),
                "scan_position": position, "query_index": int(row.query_index),
                "query_row": int(row.query_row), "query_ik14": str(row.query_ik14),
                "query_formula": str(row.query_formula), "has_near": bool(row.has_near),
                "scan_kind": str(row.scan_kind), "baseline_rank": int(row.baseline_rank),
                "baseline_margin": float(row.baseline_margin),
                "target_count": int(len(targets)), "target_tokens": ",".join(map(str, targets)),
                "control_match_levels": ",".join(control_match_levels),
                "target_result": None, "control_results": [],
            })
            attenuation = float(cell.dose) if str(cell.operator) == "attenuate" else 1.0
            variants.append((plan_index, -1, targets, attenuation))
            for control_index, control in enumerate(controls):
                variants.append((plan_index, control_index, control, attenuation))
        if (position + 1) % 500 == 0 or position + 1 == len(scan):
            print(f"[E2 plan] {position + 1:,}/{len(scan):,}; variants={len(variants):,}", flush=True)
    if not plans or not variants:
        raise RuntimeError("E2 produced no eligible corrective action")

    model, checkpoint_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if checkpoint_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("E2 requires official fine-tuned DreaMS weights")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    # Cache each candidate block once.  All clean and perturbed scores below use
    # exactly these arrays, so the effect estimate cannot drift between paths.
    candidate_rows_by_scan: list[np.ndarray] = []
    candidate_ptr_by_scan: list[np.ndarray] = []
    candidate_vectors_by_scan: list[np.ndarray] = []
    for position, row in enumerate(scan.itertuples(index=False)):
        query = int(row.query_index)
        scores, candidate_rows, local_ptr, _ = query_candidate_block(cache, query, score_column)
        detail = strict_detail(scores, candidate_rows, local_ptr)
        if int(detail["rank"]) != int(row.baseline_rank):
            raise RuntimeError(f"baseline rank mismatch at graph query {query}")
        candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
        candidate_rows_by_scan.append(candidate_rows)
        candidate_ptr_by_scan.append(np.asarray(local_ptr, dtype=np.int64))
        candidate_vectors_by_scan.append(embeddings[np.asarray([
            embedding_index[int(candidate_row)] for candidate_row in candidate_rows
        ], dtype=np.int64)])

    # Recompute the clean query with the same model invocation and candidate
    # arrays used for every perturbation.  Cached embeddings are only a
    # provenance/reproduction target, never the numerical baseline of an
    # intervention effect.
    clean_vectors = np.empty((len(scan), embeddings.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(tensors), args.batch_size):
            right = min(left + args.batch_size, len(tensors))
            clean_vectors[left:right] = model(
                torch.stack(tensors[left:right]).to(device)
            ).detach().float().cpu().numpy()
    cached_query_vectors = embeddings[np.asarray([
        embedding_index[int(row)] for row in scan["query_row"].astype(int)
    ], dtype=np.int64)]
    clean_preservation = np.einsum("ij,ij->i", clean_vectors, cached_query_vectors)
    clean_details: list[dict[str, Any]] = []
    clean_rank_mismatches = 0
    for position, vector in enumerate(clean_vectors):
        detail = strict_detail(
            vector @ candidate_vectors_by_scan[position].T,
            candidate_rows_by_scan[position], candidate_ptr_by_scan[position],
        )
        clean_details.append(detail)
        clean_rank_mismatches += int(int(detail["rank"]) != int(scan.iloc[position]["baseline_rank"]))
    clean_p01 = float(np.quantile(clean_preservation, 0.01))
    mismatch_fraction = clean_rank_mismatches / max(len(scan), 1)
    if clean_p01 < args.minimum_clean_preservation_p01:
        raise RuntimeError(
            f"clean forward/cache preservation p01={clean_p01:.6f} below "
            f"{args.minimum_clean_preservation_p01:.6f}"
        )
    if mismatch_fraction > args.maximum_clean_rank_mismatch_fraction:
        raise RuntimeError(
            f"clean forward changes {clean_rank_mismatches}/{len(scan)} cached ranks "
            f"({mismatch_fraction:.6f})"
        )
    for plan in plans:
        clean = clean_details[plan["scan_position"]]
        plan["cached_baseline_rank"] = plan["baseline_rank"]
        plan["cached_baseline_margin"] = plan["baseline_margin"]
        plan["baseline_rank"] = int(clean["rank"])
        plan["baseline_margin"] = float(clean["margin"])

    with torch.inference_mode():
        for batch_left in range(0, len(variants), args.batch_size):
            batch_right = min(batch_left + args.batch_size, len(variants))
            block = variants[batch_left:batch_right]
            batch = torch.stack([
                attenuate_sequence(
                    tensors[plans[plan_index]["scan_position"]], path, attenuation,
                )
                for plan_index, _, path, attenuation in block
            ]).to(device)
            vectors = model(batch).detach().float().cpu().numpy()
            for vector, (plan_index, control_index, _, _) in zip(vectors, block):
                plan = plans[plan_index]
                scan_position = plan["scan_position"]
                candidate_rows = candidate_rows_by_scan[scan_position]
                local_ptr = candidate_ptr_by_scan[scan_position]
                candidate_vectors = candidate_vectors_by_scan[scan_position]
                detail = strict_detail(vector @ candidate_vectors.T, candidate_rows, local_ptr)
                result = {"rank": int(detail["rank"]), "margin": float(detail["margin"])}
                if control_index < 0:
                    plan["target_result"] = result
                else:
                    plan["control_results"].append(result)
            if batch_right % 10_000 < args.batch_size or batch_right == len(variants):
                print(f"[E2 forward] {batch_right:,}/{len(variants):,}", flush=True)

    del model
    gc.collect()
    records = []
    for plan in plans:
        if plan["target_result"] is None:
            raise RuntimeError("E2 target result is missing")
        controls = plan.pop("control_results")
        target = plan.pop("target_result")
        control_margin = float(np.mean([value["margin"] for value in controls])) if controls else np.nan
        control_top1 = float(np.mean([value["rank"] == 1 for value in controls])) if controls else np.nan
        baseline_correct = plan["baseline_rank"] == 1
        target_correct = target["rank"] == 1
        records.append({
            **plan,
            "target_rank": target["rank"], "target_margin": target["margin"],
            "target_margin_change": target["margin"] - plan["baseline_margin"],
            "mean_control_margin": control_margin,
            "mean_control_margin_change": control_margin - plan["baseline_margin"] if controls else np.nan,
            "specific_margin_excess": target["margin"] - control_margin if controls else np.nan,
            "mean_control_top1": control_top1,
            "specific_top1_excess": float(target_correct) - control_top1 if controls else np.nan,
            "corrected": bool((not baseline_correct) and target_correct),
            "introduced": bool(baseline_correct and not target_correct),
            "control_count": len(controls),
        })
    frame = pd.DataFrame(records)
    match_level_counts: dict[str, int] = {}
    for value in frame["control_match_levels"].fillna("").astype(str):
        for level in filter(None, value.split(",")):
            match_level_counts[level] = match_level_counts.get(level, 0) + 1
    temporary = Path(tempfile.mkdtemp(prefix="noise_e2_scan_", dir=args.output_dir.parent))
    try:
        frame.to_csv(temporary / "paired_corrective_interventions.csv.gz", index=False, compression="gzip")
        report = {
            "status": "noise_final_e2_corrective_scan_complete",
            "formal": args.max_scan_queries == 0,
            "scan_queries": int(len(scan)),
            "eligible_query_cells": int(len(frame)),
            "encoded_variants": int(len(variants)),
            "cells_attempted": int(len(cells)),
            "cells_with_results": int(frame["cell_id"].nunique()),
            "control_match_level_counts": match_level_counts,
            "relation_supported_identity_adduct_relation_groups": int(len(relation_supported_groups)),
            "official_errors": int(scan["baseline_rank"].gt(1).sum()),
            "safety_controls": int(scan["baseline_rank"].eq(1).sum()),
            "clean_forward_reproduction": {
                "preservation_mean": float(np.mean(clean_preservation)),
                "preservation_p01": clean_p01,
                "preservation_minimum": float(np.min(clean_preservation)),
                "rank_mismatches": int(clean_rank_mismatches),
                "rank_mismatch_fraction": float(mismatch_fraction),
                "effect_baseline": "fresh clean forward from the same executor",
            },
            "contracts": {
                "model": "frozen official DreaMS",
                "candidate_references": "frozen official DreaMS embeddings",
                "matched_controls": 3,
                "P3_identity_overlap": 0,
                "P2b": "forbidden",
                "outcome_used_for_action_selection": False,
                "empirical_missingness_semantics": (
                    "identity-adduct peak prevalence chooses peaks; reliable acquisition-relation "
                    "pairs determine eligibility and the frozen relation-specific dose"
                ),
            },
            "provenance": {
                "manifest_sha256": sha256_file(manifest_path),
                "cells_sha256": sha256_file(cells_path),
                "e1_report_sha256": sha256_file(e1_report_path),
                "e1_consensus_sha256": sha256_file(consensus_path),
                "e1_pairwise_variation_sha256": sha256_file(pairwise_path),
                "a4_h5_sha256": sha256_file(a4_h5_path),
                "graph_sha256": sha256_file(args.cache),
                "embeddings_sha256": sha256_file(args.embedding_cache),
                "script_sha256": sha256_file(Path(__file__)),
                "interventions_sha256": sha256_file(temporary / "paired_corrective_interventions.csv.gz"),
            },
            "claim_limit": (
                "E2 measures frozen-encoder action specificity. It is not shared-encoder fine-tuning performance."
            ),
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

"""Noise-v3 S2: dynamic sequential peak interventions on the full candidate graph.

This is a headroom and specificity audit, not model training. After every
intervention it re-runs official DreaMS, rebuilds the current best-positive and
hard-negative candidate context, and re-computes either the input-gradient or
a registered candidate-role target. Complete same-role/intensity/mz matched random paths
are evaluated at the same dose and path length.
"""
from __future__ import annotations

import argparse
import gc
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
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_noise_v3_candidate_gradient import (  # noqa: E402
    cluster_ci, load_embeddings, query_candidate_block, strict_metrics,
    top_reference_rows,
)
from build_g8r_real_error_atlas import Cache, load_p3_identities, sha256_file  # noqa: E402
from noise_v3_core import (  # noqa: E402
    CONFOUNDER_ONLY, IDENTITY_ONLY, ROLE_NAMES, SHARED, UNMATCHED,
    attenuate_and_renormalize,
    candidate_peak_roles_from_mz, candidate_representatives,
    matched_control_tokens_strict_excluding,
    rank_gradient_targets, rank_role_targets, stable_seed,
)
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


DEFAULT_CACHE = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_EMBEDDINGS = ROOT / "data/validation/g8r_p2_official_embeddings.npz"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCHITECTURE = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_noise_v3_s2_sequential"


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
    parser.add_argument("--selection-batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--top-k-negatives", type=int, default=5)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--attenuations", type=float, nargs="+", default=[0.50, 1.00])
    parser.add_argument(
        "--selectors", nargs="+",
        choices=[
            "candidate_gradient", "role_confounder", "role_shared",
            "role_unmatched",
        ],
        default=["candidate_gradient", "role_confounder"],
    )
    parser.add_argument(
        "--action-specs", nargs="+", default=None,
        help=(
            "Optional selector:dose pairs. When supplied, only these registered "
            "actions are run instead of the selectors x attenuations Cartesian product."
        ),
    )
    parser.add_argument("--maximum-steps", type=int, default=3)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke only; 0 is formal")
    return parser.parse_args()


def current_context(
    cache: Cache,
    query: int,
    vector: np.ndarray,
    score_column: int,
    embeddings: np.ndarray,
    embedding_index: dict[int, int],
    tensor_cache: dict[int, torch.Tensor],
    query_tensor: torch.Tensor,
    top_k_negatives: int,
    fragment_tolerance: float,
) -> tuple[np.ndarray, object, np.ndarray]:
    """Current full-graph scores, candidate representatives and peak roles."""
    _, rows, ptr = query_candidate_block(cache, query, score_column)
    candidate_vectors = embeddings[[embedding_index[int(row)] for row in rows]]
    scores = candidate_vectors @ np.asarray(vector, dtype=np.float32)
    representatives = candidate_representatives(scores, rows, ptr, top_k_negatives)
    positive_rows = top_reference_rows(scores, rows, ptr, 0, 3)
    hardest_row = int(representatives.negative_rows[0])
    hardest_positions = np.flatnonzero(rows == hardest_row)
    if len(hardest_positions) != 1:
        raise RuntimeError("hard-negative representative is not unique in query graph")
    negative_molecule = int(np.searchsorted(ptr[1:], hardest_positions[0], side="right"))
    negative_rows = top_reference_rows(scores, rows, ptr, negative_molecule, 3)
    positive_mz = np.concatenate([
        tensor_cache[int(row)][1:, 0].numpy()[tensor_cache[int(row)][1:, 1].numpy() > 0]
        for row in positive_rows
    ])
    negative_mz = np.concatenate([
        tensor_cache[int(row)][1:, 0].numpy()[tensor_cache[int(row)][1:, 1].numpy() > 0]
        for row in negative_rows
    ])
    roles = candidate_peak_roles_from_mz(
        query_tensor, positive_mz, negative_mz, fragment_tolerance,
    )
    return scores, representatives, roles


def first_unused(values: np.ndarray, used: set[int]) -> int | None:
    for value in values:
        if int(value) not in used:
            return int(value)
    return None


ROLE_SELECTORS = {
    "role_confounder": CONFOUNDER_ONLY,
    "role_shared": SHARED,
    "role_unmatched": UNMATCHED,
}


def registered_actions(args: argparse.Namespace) -> list[tuple[str, float]]:
    if args.action_specs is None:
        return [
            (selector, float(attenuation))
            for selector in args.selectors for attenuation in args.attenuations
        ]
    output: list[tuple[str, float]] = []
    allowed = {"candidate_gradient", *ROLE_SELECTORS}
    for specification in args.action_specs:
        try:
            selector, dose_text = str(specification).rsplit(":", 1)
            dose = float(dose_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid action specification: {specification}") from error
        if selector not in allowed or not 0 < dose <= 1:
            raise ValueError(f"invalid action specification: {specification}")
        output.append((selector, dose))
    if len(set(output)) != len(output):
        raise ValueError("action specifications must be unique")
    return output


def encode_records(
    model: torch.nn.Module,
    device: torch.device,
    spectra: list[torch.Tensor],
    metadata: list[dict],
    records: list[dict],
    cache: Cache,
    score_column: int,
    embeddings: np.ndarray,
    embedding_index: dict[int, int],
) -> None:
    if not spectra:
        return
    with torch.inference_mode():
        vectors = model(torch.stack(spectra).to(device)).detach().float().cpu().numpy()
    for vector, item in zip(vectors, metadata):
        query = int(item["query_index"])
        _, rows, ptr = query_candidate_block(cache, query, score_column)
        candidate_vectors = embeddings[[embedding_index[int(row)] for row in rows]]
        pair_scores = candidate_vectors @ vector
        rank, mrr, margin = strict_metrics(pair_scores, ptr)
        molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
        winner_local = int(np.argmax(molecule_scores))
        winner_pair_local = int(ptr[winner_local] + np.argmax(
            pair_scores[ptr[winner_local]:ptr[winner_local + 1]]
        ))
        molecule_global = int(cache.query_ptr[query]) + winner_local
        records.append(item | {
            "rank": rank,
            "mrr": mrr,
            "margin": margin,
            "positive_score": float(molecule_scores[0]),
            "hardest_negative_score": float(np.max(molecule_scores[1:])),
            "winner_local_molecule": winner_local,
            "winner_pair_row": int(rows[winner_pair_local]),
            "winner_ik14": str(cache.molecule_ik14[molecule_global]),
            "winner_formula": str(cache.molecule_formula[molecule_global]),
            "winner_mces_grade": int(cache.molecule_mces_grade[molecule_global]),
        })
    spectra.clear()
    metadata.clear()


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
    if args.maximum_steps < 2 or args.control_repeats < 1:
        raise ValueError("S2 requires >=2 steps and at least one control repeat")
    if any(not 0 < value <= 1 for value in args.attenuations):
        raise ValueError("attenuation doses must lie in (0, 1]")
    if len(set(args.attenuations)) != len(args.attenuations):
        raise ValueError("attenuation doses must be unique")
    if len(set(args.selectors)) != len(args.selectors):
        raise ValueError("selectors must be unique")
    actions = registered_actions(args)
    if not actions:
        raise ValueError("at least one action must be registered")
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
    _, embeddings, embedding_index = load_embeddings(args.embedding_cache)
    needed = set(map(int, cache.query_row[:count]))
    for query in range(count):
        _, rows, _ = query_candidate_block(cache, query, score_column)
        needed.update(map(int, rows))
    missing = needed - set(embedding_index)
    if missing:
        raise RuntimeError(f"embedding cache misses {len(missing)} graph rows")

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

    baseline = []
    for query in range(count):
        scores, _, ptr = query_candidate_block(cache, query, score_column)
        rank, mrr, margin = strict_metrics(scores, ptr)
        baseline.append((rank, mrr, margin))

    model, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    if kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("S2 requires official fine-tuned DreaMS weights")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    sequences: list[dict] = []
    reproduction: list[float] = []
    for action_index, (selector, attenuation) in enumerate(actions):
            states = [tensor.clone() for tensor in query_tensors]
            active = np.ones(count, dtype=bool)
            paths: list[list[int]] = [[] for _ in range(count)]
            roles_by_step: list[list[np.ndarray]] = [[] for _ in range(count)]
            hard_rows: list[list[int]] = [[] for _ in range(count)]
            for step in range(1, args.maximum_steps + 1):
                active_indices = np.flatnonzero(active)
                for left in range(0, len(active_indices), args.selection_batch_size):
                    indices = active_indices[left:left + args.selection_batch_size]
                    if not len(indices):
                        continue
                    block = torch.stack([states[int(query)] for query in indices]).to(device)
                    block.requires_grad_(selector == "candidate_gradient")
                    current = model(block)
                    current_np = current.detach().float().cpu().numpy()
                    contexts = [current_context(
                        cache, int(query), vector, score_column, embeddings, embedding_index,
                        tensor_cache, states[int(query)], args.top_k_negatives,
                        args.fragment_tolerance,
                    ) for query, vector in zip(indices, current_np)]
                    if action_index == 0 and step == 1:
                        for query_value, vector in zip(indices, current_np):
                            cached = embeddings[
                                embedding_index[int(cache.query_row[int(query_value)])]
                            ]
                            reproduction.append(float(np.dot(vector, cached)))
                    gradients = None
                    if selector == "candidate_gradient":
                        positive = torch.as_tensor(np.stack([
                            embeddings[embedding_index[context[1].positive_row]]
                            for context in contexts
                        ]), device=device)
                        max_k = max(len(context[1].negative_rows) for context in contexts)
                        negatives = torch.zeros((len(indices), max_k, embeddings.shape[1]), device=device)
                        valid = torch.zeros((len(indices), max_k), dtype=torch.bool, device=device)
                        for row_index, context in enumerate(contexts):
                            rows = context[1].negative_rows
                            negatives[row_index, :len(rows)] = torch.as_tensor(np.stack([
                                embeddings[embedding_index[int(row)]] for row in rows
                            ]), device=device)
                            valid[row_index, :len(rows)] = True
                        pos_similarity = (current * positive).sum(1)
                        neg_similarity = torch.einsum("bd,bkd->bk", current, negatives)
                        neg_similarity = neg_similarity.masked_fill(~valid, -1e9)
                        weights = torch.softmax(
                            neg_similarity / args.softmax_temperature, dim=1,
                        ).detach()
                        objective = pos_similarity - (weights * neg_similarity).sum(1)
                        gradients = torch.autograd.grad(objective.sum(), block)[0][:, :, 1]

                    for offset, query_value in enumerate(indices):
                        query = int(query_value)
                        roles = contexts[offset][2]
                        used = set(paths[query])
                        if selector == "candidate_gradient":
                            gradient = gradients[offset].detach().float().cpu().numpy()
                            ranked = rank_gradient_targets(
                                states[query], gradient, roles, attenuation,
                                max_targets=args.n_highest_peaks, protect_identity=True,
                            )
                        elif selector in ROLE_SELECTORS:
                            ranked = rank_role_targets(
                                states[query], roles, ROLE_SELECTORS[selector],
                                max_targets=args.n_highest_peaks,
                            )
                        else:  # pragma: no cover - protected by registered_actions
                            raise RuntimeError(f"unhandled selector: {selector}")
                        target = first_unused(ranked, used)
                        if target is None:
                            active[query] = False
                            continue
                        if target <= 0 or roles[target] == IDENTITY_ONLY:
                            raise RuntimeError("identity/precursor protection failed")
                        if selector in ROLE_SELECTORS and roles[target] != ROLE_SELECTORS[selector]:
                            raise RuntimeError(
                                f"{selector} chose peak role {int(roles[target])}"
                            )
                        paths[query].append(target)
                        roles_by_step[query].append(roles.copy())
                        hard_rows[query].append(int(contexts[offset][1].negative_rows[0]))
                        states[query] = attenuate_and_renormalize(
                            states[query], target, attenuation,
                        )
                    done = min(left + len(indices), len(active_indices))
                    if done % 5000 < args.selection_batch_size:
                        print(
                            f"[select] {selector} dose={attenuation:.2f} step={step} "
                            f"{done:,}/{len(active_indices):,}", flush=True,
                        )

            for query in range(count):
                target_tokens = paths[query]
                if not target_tokens:
                    continue
                target_state = query_tensors[query].clone()
                control_paths = [[] for _ in range(args.control_repeats)]
                blocked = set(target_tokens)
                complete_steps = 0
                for step_index, target in enumerate(target_tokens):
                    controls = matched_control_tokens_strict_excluding(
                        target_state, target, roles_by_step[query][step_index],
                        args.control_repeats,
                        stable_seed(args.seed, query, selector, attenuation, step_index),
                        excluded=blocked | {
                            token for path in control_paths for token in path
                        },
                    )
                    if len(controls) != args.control_repeats:
                        break
                    for repeat, token in enumerate(controls):
                        control_paths[repeat].append(int(token))
                    target_state = attenuate_and_renormalize(
                        target_state, target, attenuation,
                    )
                    complete_steps += 1
                if complete_steps:
                    sequences.append({
                        "query_index": query,
                        "selector": selector,
                        "attenuation": float(attenuation),
                        "target_tokens": target_tokens[:complete_steps],
                        "target_roles": [
                            str(ROLE_NAMES[roles_by_step[query][index][target_tokens[index]]])
                            for index in range(complete_steps)
                        ],
                        "hard_negative_rows": hard_rows[query][:complete_steps],
                        "control_paths": [path[:complete_steps] for path in control_paths],
                    })
            print(
                f"[paths] {selector} dose={attenuation:.2f}; total={len(sequences):,}",
                flush=True,
            )

    if formal and (not reproduction or float(np.quantile(reproduction, 0.01)) < 0.999):
        raise RuntimeError("official clean-forward/cache reproduction failed")
    if not sequences:
        raise RuntimeError("no complete sequential path")

    records: list[dict] = []
    spectra_batch: list[torch.Tensor] = []
    metadata_batch: list[dict] = []
    variants = 0
    for sequence in sequences:
        query = int(sequence["query_index"])
        base_rank, base_mrr, base_margin = baseline[query]
        states = [query_tensors[query].clone() for _ in range(1 + args.control_repeats)]
        for step_index, target in enumerate(sequence["target_tokens"], start=1):
            states[0] = attenuate_and_renormalize(
                states[0], int(target), float(sequence["attenuation"]),
            )
            tokens = [int(target)]
            for repeat in range(args.control_repeats):
                control = int(sequence["control_paths"][repeat][step_index - 1])
                states[repeat + 1] = attenuate_and_renormalize(
                    states[repeat + 1], control, float(sequence["attenuation"]),
                )
                tokens.append(control)
            for condition_index, state in enumerate(states):
                condition = "target" if condition_index == 0 else "matched_random"
                repeat = -1 if condition_index == 0 else condition_index - 1
                spectra_batch.append(state.clone())
                metadata_batch.append({
                    "query_index": query,
                    "query_row": int(cache.query_row[query]),
                    "query_ik14": str(cache.query_ik14[query]),
                    "query_formula": str(cache.query_formula[query]),
                    "has_near": bool(cache.query_has_near[query]),
                    "baseline_rank": int(base_rank),
                    "baseline_mrr": float(base_mrr),
                    "baseline_margin": float(base_margin),
                    "selector": sequence["selector"],
                    "attenuation": float(sequence["attenuation"]),
                    "step": step_index,
                    "target_path": ",".join(map(str, sequence["target_tokens"][:step_index])),
                    "control_path": (
                        "" if repeat < 0 else ",".join(map(
                            str, sequence["control_paths"][repeat][:step_index],
                        ))
                    ),
                    "target_role": sequence["target_roles"][step_index - 1],
                    "hard_negative_row": int(sequence["hard_negative_rows"][step_index - 1]),
                    "condition": condition,
                    "repeat": repeat,
                    "applied_token": tokens[condition_index],
                })
                variants += 1
                if len(spectra_batch) >= args.encode_batch_size:
                    encode_records(
                        model, device, spectra_batch, metadata_batch, records,
                        cache, score_column, embeddings, embedding_index,
                    )
                    if variants % 10000 < args.encode_batch_size:
                        print(f"[encode] {variants:,}", flush=True)
    encode_records(
        model, device, spectra_batch, metadata_batch, records,
        cache, score_column, embeddings, embedding_index,
    )
    del model
    gc.collect()

    results = pd.DataFrame(records)
    key = ["query_index", "selector", "attenuation", "step"]
    target = results.loc[results["condition"] == "target"].copy()
    random = results.loc[results["condition"] == "matched_random"].groupby(
        key, as_index=False,
    ).agg(
        random_rank=("rank", "mean"),
        random_top1=("rank", lambda values: float(np.mean(np.asarray(values) == 1))),
        random_mrr=("mrr", "mean"), random_margin=("margin", "mean"),
        random_repeats=("repeat", "count"),
    )
    paired = target.merge(random, on=key, validate="one_to_one")
    paired = paired.loc[paired["random_repeats"] == args.control_repeats].copy()
    if paired.empty:
        raise RuntimeError("no paired sequential outcomes")
    paired["target_rank"] = paired["rank"]
    paired["target_mrr"] = paired["mrr"]
    paired["target_margin"] = paired["margin"]
    paired["target_margin_change"] = paired["target_margin"] - paired["baseline_margin"]
    paired["random_margin_change"] = paired["random_margin"] - paired["baseline_margin"]
    paired["target_minus_random_margin_change"] = (
        paired["target_margin_change"] - paired["random_margin_change"]
    )
    paired["target_minus_random_top1"] = (
        paired["target_rank"].eq(1).astype(float) - paired["random_top1"]
    )
    paired["corrected"] = paired["baseline_rank"].gt(1) & paired["target_rank"].eq(1)
    paired["introduced"] = paired["baseline_rank"].eq(1) & paired["target_rank"].gt(1)

    cells = {}
    for position, (key_value, group) in enumerate(paired.groupby(
        ["selector", "attenuation", "step"], sort=True,
    )):
        selector, attenuation, step = key_value
        wrong = group.loc[group["baseline_rank"] > 1]
        name = f"{selector}|a={float(attenuation):.2f}|step={int(step)}"
        cells[name] = {
            "queries": int(len(group)),
            "identities": int(group["query_ik14"].nunique()),
            "formulas": int(group["query_formula"].nunique()),
            "corrected": int(group["corrected"].sum()),
            "introduced": int(group["introduced"].sum()),
            "mean_target_minus_random_margin": float(
                group["target_minus_random_margin_change"].mean()
            ),
            "baseline_wrong_target_minus_random_top1": float(
                wrong["target_minus_random_top1"].mean()
            ) if len(wrong) else None,
            "baseline_wrong_identity_top1_ci": cluster_ci(
                wrong, "target_minus_random_top1", "query_ik14", args.bootstrap,
                args.seed + position,
            ) if len(wrong) else None,
            "baseline_wrong_formula_top1_ci": cluster_ci(
                wrong, "target_minus_random_top1", "query_formula", args.bootstrap,
                args.seed + 10_000 + position,
            ) if len(wrong) else None,
        }

    selected_table = pd.DataFrame([{
        "query_index": item["query_index"], "selector": item["selector"],
        "attenuation": item["attenuation"], "steps": len(item["target_tokens"]),
        "target_tokens": ",".join(map(str, item["target_tokens"])),
        "target_roles": ",".join(item["target_roles"]),
        "hard_negative_rows": ",".join(map(str, item["hard_negative_rows"])),
        "control_paths": ";".join(
            ",".join(map(str, path)) for path in item["control_paths"]
        ),
    } for item in sequences])
    report = {
        "status": "noise_v3_s2_sequential_matrix_complete",
        "formal": formal,
        "queries": count,
        "query_identities": int(len(set(map(str, cache.query_ik14[:count])))),
        "official_errors": int(sum(rank > 1 for rank, _, _ in baseline)),
        "official_recall1": float(np.mean([rank == 1 for rank, _, _ in baseline])),
        "official_near_errors": int(sum(
            rank > 1 and bool(cache.query_has_near[index])
            for index, (rank, _, _) in enumerate(baseline)
        )),
        "sequences": int(len(sequences)),
        "paired_cells": int(len(paired)),
        "intervention_variants": int(variants),
        "matrix": {
            "selectors": sorted({selector for selector, _ in actions}),
            "attenuations": sorted({float(dose) for _, dose in actions}),
            "registered_actions": [
                {"selector": selector, "attenuation": float(dose)}
                for selector, dose in actions
            ],
            "maximum_steps": int(args.maximum_steps),
            "control_repeats": int(args.control_repeats),
            "dynamic_candidate_recalculation_every_step": True,
            "dynamic_peak_role_recalculation_every_step": True,
            "identity_only_peaks_protected": True,
            "complete_matched_random_paths_only": True,
            "no_outcome_based_action_selection": True,
        },
        "official_forward_cache_preservation": {
            "mean": float(np.mean(reproduction)),
            "p01": float(np.quantile(reproduction, 0.01)),
            "minimum": float(np.min(reproduction)),
        },
        "cell_results": cells,
        "claim_limit": (
            "This is a P3-disjoint action-space audit. It uses structural identity to define "
            "the positive candidate and reports every preregistered path length. It does not "
            "train a policy or establish clean-spectrum fine-tuning gains."
        ),
        "provenance": {
            "cache_sha256": sha256_file(args.cache),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
            "noise_v3_core_sha256": sha256_file(ROOT / "tasks/noise_v3_core.py"),
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_v3_s2_", dir=args.output_dir.parent))
    try:
        paired.to_csv(temporary / "paired_interventions.csv.gz", index=False)
        selected_table.to_csv(temporary / "selected_sequences.csv.gz", index=False)
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

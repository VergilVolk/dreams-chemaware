"""Run a bounded real-encoder scan of new multi-peak direct-noise actions.

The discovery panel is fixed before any new action is executed.  Error queries
are historical A4 single-peak residuals; correct queries are a separate safety
panel.  Both are restricted to the current outer-training formula split.  The
script recomputes the complete candidate geometry, peak roles and input
gradients under the supplied shared encoder, then compares each target action
with a strict same-role, intensity/mz-matched composite control.

This is a development action-space experiment, not a shared-encoder result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import torch

from build_noise_corrected_full_action_bank import load_initial_model, current_context
from noise_corrected_action_expansion_v4 import (
    attenuate_tokens_and_renormalize,
    boost_tokens_and_renormalize,
    conservative_intensity_exchange,
    rank_attenuation_tokens,
    rank_supported_boost_tokens,
    signed_multiplicative_action,
)
from noise_corrected_trust_region_action_v4 import (
    apply_trust_region_action,
    minimal_supported_margin_action,
)
from noise_corrected_action_routing import RoutingThresholds, route_action
from noise_final_core import CandidateGraph, sha256_file, stable_fold, strict_rank
from noise_v3_core import matched_control_tokens_strict_excluding, stable_seed
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E8 = ROOT / (
    "data/validation/g8r_noise_final_e8_direct_transfer/"
    "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/"
    "seed_20260830/fold_0/final_shared_encoder.pt"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir", type=Path,
        default=ROOT / "data/validation/noise_corrected_candidate_graph_v1_20260906",
    )
    parser.add_argument(
        "--a4-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--initial-student-checkpoint", type=Path, default=DEFAULT_E8)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--sample-seed", type=int, default=20260907)
    parser.add_argument("--error-queries", type=int, default=16)
    parser.add_argument("--correct-queries", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--top-k-negatives", type=int, default=5)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--gradient-batch-size", type=int, default=2)
    parser.add_argument(
        "--recipe-set", choices=("baseline", "sequential", "all"), default="all",
        help=(
            "baseline runs the frozen one-shot panel; sequential runs its "
            "single-peak comparator plus refreshed supported-boost paths"
        ),
    )
    parser.add_argument("--sequential-boost-steps", type=int, default=6)
    parser.add_argument("--sequential-boost-dose", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/noise_corrected_action_expansion_v4_local_e8_20260907",
    )
    return parser.parse_args()


def formula_diverse_sample(
    frame: pd.DataFrame, count: int, *, seed: int, panel: str,
) -> pd.DataFrame:
    if count < 0:
        raise ValueError("sample count must be nonnegative")
    if not count or frame.empty:
        return frame.iloc[:0].copy()
    local = frame.copy()
    local["_tie"] = [
        stable_seed(seed, panel, formula, int(row))
        for formula, row in zip(local["query_formula"], local["query_row"])
    ]
    local = local.sort_values(["query_formula", "_tie"], kind="mergesort")
    local["_round"] = local.groupby("query_formula", sort=False).cumcount()
    local = local.sort_values(["_round", "_tie"], kind="mergesort")
    return local.head(min(count, len(local))).drop(columns=["_tie", "_round"])


def candidate_detail(
    graph: CandidateGraph,
    query: int,
    vector: np.ndarray,
    embeddings: np.ndarray,
    row_index: dict[int, int],
) -> dict[str, float | int]:
    _, rows, ptr, _ = graph.query_block(query)
    candidate = embeddings[[row_index[int(row)] for row in rows]]
    pair_scores = candidate @ np.asarray(vector, dtype=np.float32)
    molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
    return {
        "rank": int(strict_rank(molecule_scores)),
        "positive": float(molecule_scores[0]),
        "negative": float(np.max(molecule_scores[1:])),
        "margin": float(molecule_scores[0] - np.max(molecule_scores[1:])),
    }


def matched_path(
    clean: torch.Tensor,
    targets: tuple[int, ...],
    roles: np.ndarray,
    *,
    seed: int,
    excluded: tuple[int, ...] = (),
) -> tuple[int, ...] | None:
    blocked = set(map(int, targets)) | set(map(int, excluded))
    output: list[int] = []
    for position, target in enumerate(targets):
        control = matched_control_tokens_strict_excluding(
            clean, int(target), roles, 1,
            stable_seed(seed, position, int(target)),
            excluded=blocked | set(output),
        )
        if len(control) != 1:
            return None
        output.append(int(control[0]))
    return tuple(output)


def build_recipe_views(
    clean: torch.Tensor,
    gradient: np.ndarray,
    roles: np.ndarray,
    *,
    seed: int,
    baseline_margin: float,
) -> list[dict[str, object]]:
    down = tuple(map(int, rank_attenuation_tokens(clean, gradient, roles, 4)))
    up = tuple(map(int, rank_supported_boost_tokens(clean, gradient, roles, 2)))
    recipes: list[tuple[str, tuple[int, ...], tuple[int, ...], object]] = []
    if len(down) >= 1:
        recipes.append((
            "single_attenuation_1x50", down[:1], (),
            lambda d, u: attenuate_tokens_and_renormalize(clean, d, 0.50),
        ))
    if len(down) >= 2:
        recipes.append((
            "dual_attenuation_2x50", down[:2], (),
            lambda d, u: attenuate_tokens_and_renormalize(clean, d, 0.50),
        ))
    if len(down) >= 4:
        recipes.extend((
            (
                "quad_attenuation_4x25", down[:4], (),
                lambda d, u: attenuate_tokens_and_renormalize(clean, d, 0.25),
            ),
            (
                "quad_attenuation_4x50", down[:4], (),
                lambda d, u: attenuate_tokens_and_renormalize(clean, d, 0.50),
            ),
        ))
    if len(up) >= 2:
        recipes.append((
            "supported_boost_2x50", (), up[:2],
            lambda d, u: boost_tokens_and_renormalize(clean, u, 0.50),
        ))
    if len(down) >= 2 and len(up) >= 2:
        recipes.extend((
            (
                "signed_multiplicative_2d2u_50", down[:2], up[:2],
                lambda d, u: signed_multiplicative_action(
                    clean, d, u, attenuation=0.50, boost=0.50,
                ),
            ),
            (
                "conservative_exchange_2d2u_50", down[:2], up[:2],
                lambda d, u: conservative_intensity_exchange(
                    clean, d, u, attenuation=0.50,
                ),
            ),
        ))
    output: list[dict[str, object]] = []
    for recipe, down_target, up_target, transform in recipes:
        control_down = matched_path(
            clean, down_target, roles, seed=stable_seed(seed, recipe, "down"),
            excluded=up_target,
        ) if down_target else ()
        blocked = down_target + (() if control_down is None else control_down)
        control_up = matched_path(
            clean, up_target, roles, seed=stable_seed(seed, recipe, "up"),
            excluded=blocked,
        ) if up_target else ()
        control_complete = control_down is not None and control_up is not None
        output.append({
            "recipe": recipe,
            "target_down": down_target,
            "target_up": up_target,
            "target": transform(down_target, up_target),
            "control_down": control_down,
            "control_up": control_up,
            "control": (
                transform(control_down, control_up) if control_complete else None
            ),
        })
    # A fixed 50% edit can under-treat a deep error and over-treat an already
    # correct query.  Freeze a minimal first-order dose from the clean margin
    # and input gradient only; action outcomes remain unavailable here.
    for recipe, directions, desired_margin, maximum_gain, maximum_peaks, per_peak, total in (
        ("adaptive_trust_down", frozenset({"down"}), 0.05, 0.12, 6, 0.75, 2.0),
        ("adaptive_trust_up", frozenset({"up"}), 0.05, 0.12, 6, 0.75, 2.0),
        ("adaptive_trust_joint", frozenset({"down", "up"}), 0.05, 0.12, 6, 0.75, 2.0),
        ("adaptive_strong_down", frozenset({"down"}), 0.15, 0.30, 8, 0.90, 3.0),
        ("adaptive_strong_up", frozenset({"up"}), 0.15, 0.30, 8, 0.90, 3.0),
        ("adaptive_strong_joint", frozenset({"down", "up"}), 0.15, 0.30, 8, 0.90, 3.0),
    ):
        required_gain = float(np.clip(
            float(desired_margin) - float(baseline_margin), 0.02, maximum_gain,
        ))
        action = minimal_supported_margin_action(
            clean, gradient, roles,
            target_gain=required_gain,
            maximum_peaks=maximum_peaks,
            maximum_fraction_per_peak=per_peak,
            maximum_total_fraction=total,
            directions=directions,
        )
        if not action.attenuation_tokens and not action.boost_tokens:
            continue
        control_down = matched_path(
            clean, action.attenuation_tokens, roles,
            seed=stable_seed(seed, recipe, "down"),
            excluded=action.boost_tokens,
        ) if action.attenuation_tokens else ()
        blocked = action.attenuation_tokens + (
            () if control_down is None else control_down
        )
        control_up = matched_path(
            clean, action.boost_tokens, roles,
            seed=stable_seed(seed, recipe, "up"),
            excluded=blocked,
        ) if action.boost_tokens else ()
        control_complete = control_down is not None and control_up is not None
        control_action = (
            replace(
                action,
                attenuation_tokens=control_down,
                boost_tokens=control_up,
            )
            if control_complete else None
        )
        output.append({
            "recipe": recipe,
            "target_down": action.attenuation_tokens,
            "target_up": action.boost_tokens,
            "target": apply_trust_region_action(clean, action),
            "control_down": control_down,
            "control_up": control_up,
            "control": (
                apply_trust_region_action(clean, control_action)
                if control_action is not None else None
            ),
            "predicted_gain": action.predicted_gain,
            "target_gain": action.target_gain,
            "total_fractional_dose": action.total_fractional_dose,
            "target_gain_reached": action.target_reached,
        })
    return output


def build_sequential_supported_boost_views(
    query_meta: pd.DataFrame,
    model: torch.nn.Module,
    graph: CandidateGraph,
    tensors: dict[int, torch.Tensor],
    embeddings: np.ndarray,
    row_index: dict[int, int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[int, list[dict[str, object]]]:
    """Build direct boost prefixes with a freshly mined boundary each step.

    The target path is mined without seeing target/control rank outcomes.  Each
    step re-encodes the current edited spectrum, refreshes the positive and hard
    negative representatives, recomputes peak roles and the input gradient, and
    selects one previously unused identity/shared supported peak.  Matched
    controls are constructed only after the complete target path is frozen, so
    no control token can later collide with a target token.
    """
    if args.sequential_boost_steps < 1 or args.sequential_boost_dose <= 0:
        raise ValueError("sequential boost steps and dose must be positive")
    meta = {
        int(row.query_index): row for row in query_meta.itertuples(index=False)
    }
    states = {
        query: tensors[int(row.query_row)].clone() for query, row in meta.items()
    }
    paths: dict[int, list[int]] = {query: [] for query in meta}
    role_history: dict[int, list[np.ndarray]] = {query: [] for query in meta}
    target_views: dict[int, list[torch.Tensor]] = {query: [] for query in meta}
    active = set(meta)
    for step in range(1, args.sequential_boost_steps + 1):
        ordered = np.asarray(sorted(active), dtype=np.int64)
        for left in range(0, len(ordered), args.gradient_batch_size):
            local = ordered[left:left + args.gradient_batch_size]
            block = torch.stack([states[int(query)] for query in local]).to(device)
            block.requires_grad_(True)
            current = forward_embeddings(model, block, False)
            contexts = [
                current_context(
                    graph, int(query), vector, embeddings, row_index, tensors,
                    states[int(query)], args.top_k_negatives,
                    args.fragment_tolerance,
                )
                for query, vector in zip(
                    local, current.detach().float().cpu().numpy(),
                )
            ]
            positive = torch.as_tensor(np.stack([
                embeddings[row_index[int(context[0].positive_row)]]
                for context in contexts
            ]), device=device, dtype=current.dtype)
            maximum = max(len(context[0].negative_rows) for context in contexts)
            negative = torch.zeros(
                (len(local), maximum, embeddings.shape[1]),
                device=device, dtype=current.dtype,
            )
            valid = torch.zeros(
                (len(local), maximum), device=device, dtype=torch.bool,
            )
            for position, context in enumerate(contexts):
                rows = context[0].negative_rows
                negative[position, :len(rows)] = torch.as_tensor(np.stack([
                    embeddings[row_index[int(row)]] for row in rows
                ]), device=device, dtype=current.dtype)
                valid[position, :len(rows)] = True
            positive_score = torch.sum(current * positive, dim=1)
            negative_score = torch.einsum(
                "bd,bkd->bk", current, negative,
            ).masked_fill(~valid, -1e9)
            weight = torch.softmax(
                negative_score / args.softmax_temperature, dim=1,
            ).detach()
            objective = positive_score - torch.sum(weight * negative_score, dim=1)
            gradients = torch.autograd.grad(objective.sum(), block)[0][:, :, 1]
            for position, query_value in enumerate(local):
                query = int(query_value)
                roles = contexts[position][1]
                ranked = rank_supported_boost_tokens(
                    states[query],
                    gradients[position].detach().float().cpu().numpy(),
                    roles,
                    args.n_highest_peaks,
                )
                target = next(
                    (int(token) for token in ranked if int(token) not in paths[query]),
                    None,
                )
                if target is None:
                    active.discard(query)
                    continue
                paths[query].append(target)
                role_history[query].append(roles.copy())
                states[query] = boost_tokens_and_renormalize(
                    states[query], (target,), args.sequential_boost_dose,
                )
                target_views[query].append(states[query].clone())
        print(
            f"[sequential-supported-boost] step={step} active={len(active):,}",
            flush=True,
        )

    output: dict[int, list[dict[str, object]]] = defaultdict(list)
    for query, row in meta.items():
        clean = tensors[int(row.query_row)]
        target_state = clean.clone()
        control_state = clean.clone()
        control_path: list[int] = []
        controls_complete = True
        blocked = set(paths[query])
        for position, target in enumerate(paths[query]):
            selected = (
                matched_control_tokens_strict_excluding(
                    target_state,
                    target,
                    role_history[query][position],
                    1,
                    stable_seed(
                        args.sample_seed, int(row.query_row),
                        "sequential_supported_boost", position,
                    ),
                    excluded=blocked | set(control_path),
                )
                if controls_complete else []
            )
            target_state = boost_tokens_and_renormalize(
                target_state, (target,), args.sequential_boost_dose,
            )
            if not torch.equal(target_state, target_views[query][position]):
                raise RuntimeError("sequential target reconstruction drifted")
            if len(selected) == 1:
                control_path.append(int(selected[0]))
                control_state = boost_tokens_and_renormalize(
                    control_state, (int(selected[0]),),
                    args.sequential_boost_dose,
                )
            else:
                controls_complete = False
            step = position + 1
            output[query].append({
                "recipe": f"sequential_supported_boost_{step}x",
                "target_down": (),
                "target_up": tuple(paths[query][:step]),
                "target": target_views[query][position],
                "control_down": (),
                "control_up": (
                    tuple(control_path) if controls_complete else None
                ),
                "control": control_state.clone() if controls_complete else None,
                "predicted_gain": None,
                "target_gain": None,
                "total_fractional_dose": float(
                    step * args.sequential_boost_dose
                ),
                "target_gain_reached": None,
            })
    return output


def cluster_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    resamples: int = 2000,
) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    if not len(values):
        return None
    unique = np.unique(clusters)
    grouped = [values[clusters == cluster] for cluster in unique]
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.integers(0, len(grouped), len(grouped))
        boot[index] = np.concatenate([grouped[position] for position in selected]).mean()
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def summarize(frame: pd.DataFrame, recipe: str, seed: int) -> dict[str, object]:
    local = frame[frame["recipe"] == recipe].copy()
    # Historical A4 strata select the panel without looking at the new action,
    # but all endpoints must use the recomputed rank in the current encoder.
    error = local[local["baseline_rank"] != 1]
    correct = local[local["baseline_rank"] == 1]
    paired = local[local["control_rank"].notna()].copy()
    thresholds = RoutingThresholds()
    paired["route"] = [
        route_action(
            clean_rank=int(row.baseline_rank),
            clean_margin=float(row.baseline_margin),
            action_rank=int(row.target_rank),
            action_margin=float(row.target_margin),
            control_margin=float(row.control_margin),
            thresholds=thresholds,
        )
        for row in paired.itertuples(index=False)
    ]
    route_counts = paired["route"].value_counts()
    strict_corrected = paired[
        (paired["baseline_rank"] != 1)
        & (paired["target_rank"] == 1)
        & paired["route"].eq("corrective")
    ]
    strict_introduced = paired[
        (paired["baseline_rank"] == 1)
        & (paired["target_rank"] != 1)
    ]
    dose = local["total_fractional_dose"].dropna().to_numpy(float)
    reached = local["target_gain_reached"].dropna().astype(bool).to_numpy()
    delta = (
        (paired["target_rank"] == 1).astype(float)
        - (paired["control_rank"] == 1).astype(float)
    ).to_numpy()
    return {
        "target_error_queries": int(len(error)),
        "target_corrected": int((error["target_rank"] == 1).sum()),
        "target_unique_corrected_beyond_single": int((
            (error["target_rank"] == 1) & (error["single_target_rank"] != 1)
        ).sum()),
        "target_correct_queries": int(len(correct)),
        "target_introduced": int((correct["target_rank"] != 1).sum()),
        "target_risk_net": int(
            (error["target_rank"] == 1).sum()
            - 2 * (correct["target_rank"] != 1).sum()
        ),
        "matched_control_complete": int(len(paired)),
        "formal_route_counts": {
            label: int(route_counts.get(label, 0)) for label in (
                "corrective", "robustness_only", "harmful", "uncertain",
            )
        },
        "strict_route_corrected": int(len(strict_corrected)),
        "strict_route_corrected_queries": sorted(
            map(int, strict_corrected["query_index"].unique())
        ),
        "strict_route_introduced": int(len(strict_introduced)),
        "target_minus_control_top1_mean": float(delta.mean()) if len(delta) else None,
        "target_minus_control_top1_formula_ci": cluster_interval(
            delta, paired["query_formula"].to_numpy(), seed=stable_seed(seed, recipe),
        ),
        "target_margin_change_median": (
            float((local["target_margin"] - local["baseline_margin"]).median())
            if len(local) else None
        ),
        "target_minus_control_margin_median": (
            float((paired["target_margin"] - paired["control_margin"]).median())
            if len(paired) else None
        ),
        "adaptive_total_fractional_dose_p10_median_p90": (
            [float(value) for value in np.quantile(dose, [0.1, 0.5, 0.9])]
            if len(dose) else None
        ),
        "adaptive_target_gain_reached_fraction": (
            float(reached.mean()) if len(reached) else None
        ),
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph_path = args.graph_dir / "candidate_graph.npz"
    embedding_path = args.graph_dir / "official_embeddings.npz"
    scan_path = args.a4_dir / "scan_queries.csv.gz"
    action_path = args.a4_dir / "exact_peak_scan.h5"
    required = [
        graph_path, embedding_path, scan_path, action_path, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
    ]
    if args.initial_student_checkpoint is not None:
        required.extend([
            args.initial_student_checkpoint,
            args.initial_student_checkpoint.parent / "decision.json",
        ])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    graph = CandidateGraph(graph_path)
    scan = pd.read_csv(scan_path)
    row_to_query = {int(row): index for index, row in enumerate(graph.query_row)}
    scan["graph_query"] = [row_to_query.get(int(row), -1) for row in scan["query_row"]]
    scan = scan[scan["graph_query"] >= 0].copy()
    scan["formula_fold"] = [
        stable_fold(str(formula), 5, args.formula_fold_seed)
        for formula in scan["query_formula"]
    ]
    scan = scan[scan["formula_fold"] != args.outer_fold].copy()
    with h5py.File(action_path, "r") as handle:
        ptr = np.asarray(handle["query_action_ptr"], dtype=np.int64)
        rank = np.asarray(handle["result_rank"], dtype=np.int16).reshape(-1, 4)
        gradient = np.asarray(handle["action_gradient"], dtype=np.float32)
        intensity = np.asarray(handle["action_intensity"], dtype=np.float32)
        roles = np.asarray(handle["action_role"], dtype=np.int8)
        eligible = np.asarray(handle["action_policy_eligible"], dtype=bool)
        best_rank = np.asarray([
            int(rank[int(ptr[position]):int(ptr[position + 1])].min())
            if ptr[position + 1] > ptr[position] else 32767
            for position in range(len(ptr) - 1)
        ])
        historical_down_count = np.asarray([
            int(np.sum(
                eligible[int(ptr[position]):int(ptr[position + 1])]
                & (
                    -intensity[int(ptr[position]):int(ptr[position + 1])]
                    * gradient[int(ptr[position]):int(ptr[position + 1])]
                    > 0
                )
            ))
            for position in range(len(ptr) - 1)
        ])
        historical_up_count = np.asarray([
            int(np.sum(
                np.isin(
                    roles[int(ptr[position]):int(ptr[position + 1])], [0, 2],
                )
                & (
                    intensity[int(ptr[position]):int(ptr[position + 1])]
                    * gradient[int(ptr[position]):int(ptr[position + 1])]
                    > 0
                )
            ))
            for position in range(len(ptr) - 1)
        ])
    scan["historical_single_best_rank"] = best_rank[scan["scan_position"].to_numpy(int)]
    scan["historical_down_count"] = historical_down_count[
        scan["scan_position"].to_numpy(int)
    ]
    scan["historical_supported_up_count"] = historical_up_count[
        scan["scan_position"].to_numpy(int)
    ]
    executable = (
        (scan["historical_down_count"] >= 2)
        & (scan["historical_supported_up_count"] >= 2)
    )
    residual = scan[
        (scan["baseline_rank"] != 1) & (scan["historical_single_best_rank"] != 1)
        & executable
    ]
    safety = scan[(scan["baseline_rank"] == 1) & executable]
    residual = formula_diverse_sample(
        residual, args.error_queries, seed=args.sample_seed, panel="residual_error",
    )
    safety = formula_diverse_sample(
        safety, args.correct_queries, seed=args.sample_seed, panel="correct_safety",
    )
    residual["panel"] = "residual_error"
    safety["panel"] = "correct_safety"
    selected = pd.concat([residual, safety], ignore_index=True)
    if selected.empty or residual.empty or safety.empty:
        raise RuntimeError("both residual-error and correct-safety panels are required")

    needed: set[int] = set(map(int, selected["query_row"]))
    for query in selected["graph_query"]:
        _, rows, _, _ = graph.query_block(int(query))
        needed.update(map(int, rows))
    store = SpectrumStore(args.data, np.asarray(sorted(needed), dtype=np.int64), args.n_highest_peaks)
    tensors = {int(row): store.tensor[index] for index, row in enumerate(store.rows)}
    device = torch.device(args.device)
    model, model_provenance = load_initial_model(args, device)
    embeddings = encode_rows(
        model, store, store.rows, device, args.encode_batch_size, args.amp,
        "action-expansion-current",
    )
    row_index = {int(row): index for index, row in enumerate(store.rows)}

    query_meta: list[dict[str, object]] = []
    gradients: dict[int, np.ndarray] = {}
    roles_by_query: dict[int, np.ndarray] = {}
    selected_records = list(selected.itertuples(index=False))
    for left in range(0, len(selected_records), args.gradient_batch_size):
        local = selected_records[left:left + args.gradient_batch_size]
        clean = torch.stack([tensors[int(row.query_row)] for row in local]).to(device)
        clean.requires_grad_(True)
        current = forward_embeddings(model, clean, False)
        contexts = [
            current_context(
                graph, int(row.graph_query), vector, embeddings, row_index,
                tensors, tensors[int(row.query_row)], args.top_k_negatives,
                args.fragment_tolerance,
            )
            for row, vector in zip(local, current.detach().float().cpu().numpy())
        ]
        positive = torch.as_tensor(np.stack([
            embeddings[row_index[int(context[0].positive_row)]] for context in contexts
        ]), device=device, dtype=current.dtype)
        maximum = max(len(context[0].negative_rows) for context in contexts)
        negative = torch.zeros((len(local), maximum, embeddings.shape[1]), device=device)
        valid = torch.zeros((len(local), maximum), device=device, dtype=torch.bool)
        for position, context in enumerate(contexts):
            rows = context[0].negative_rows
            negative[position, :len(rows)] = torch.as_tensor(np.stack([
                embeddings[row_index[int(row)]] for row in rows
            ]), device=device, dtype=current.dtype)
            valid[position, :len(rows)] = True
        positive_score = torch.sum(current * positive, dim=1)
        negative_score = torch.einsum("bd,bkd->bk", current, negative).masked_fill(~valid, -1e9)
        weight = torch.softmax(negative_score / args.softmax_temperature, dim=1).detach()
        objective = positive_score - torch.sum(weight * negative_score, dim=1)
        local_gradient = torch.autograd.grad(objective.sum(), clean)[0][:, :, 1]
        for row, vector, gradient, context in zip(
            local, current.detach().float().cpu().numpy(),
            local_gradient.detach().float().cpu().numpy(), contexts,
        ):
            query = int(row.graph_query)
            gradients[query] = gradient
            roles_by_query[query] = context[1]
            detail = candidate_detail(graph, query, vector, embeddings, row_index)
            query_meta.append({
                "query_index": query,
                "query_row": int(row.query_row),
                "query_ik14": str(row.query_ik14),
                "query_formula": str(row.query_formula),
                "panel": str(row.panel),
                "baseline_rank": detail["rank"],
                "baseline_margin": detail["margin"],
                "historical_single_best_rank": int(row.historical_single_best_rank),
            })
        print(f"[action-expansion-gradient] {min(left + len(local), len(selected_records))}/{len(selected_records)}", flush=True)

    query_meta_frame = pd.DataFrame(query_meta)
    sequential_by_query = (
        build_sequential_supported_boost_views(
            query_meta_frame, model, graph, tensors, embeddings, row_index,
            device, args,
        )
        if args.recipe_set in {"sequential", "all"} else {}
    )
    view_records: list[dict[str, object]] = []
    view_tensors: list[torch.Tensor] = []
    for row in query_meta_frame.itertuples(index=False):
        query = int(row.query_index)
        clean = tensors[int(row.query_row)]
        recipes = build_recipe_views(
            clean, gradients[query], roles_by_query[query],
            seed=stable_seed(args.sample_seed, int(row.query_row)),
            baseline_margin=float(row.baseline_margin),
        )
        if args.recipe_set == "sequential":
            recipes = [
                recipe for recipe in recipes
                if recipe["recipe"] == "single_attenuation_1x50"
            ]
        recipes.extend(sequential_by_query.get(query, ()))
        for recipe in recipes:
            for kind in ("target", "control"):
                tensor = recipe[kind]
                if tensor is None:
                    continue
                view_records.append({
                    **row._asdict(),
                    "recipe": recipe["recipe"],
                    "view_kind": kind,
                    "target_down": json.dumps(recipe["target_down"]),
                    "target_up": json.dumps(recipe["target_up"]),
                    "control_down": json.dumps(recipe["control_down"]),
                    "control_up": json.dumps(recipe["control_up"]),
                    "predicted_gain": recipe.get("predicted_gain"),
                    "target_gain": recipe.get("target_gain"),
                    "total_fractional_dose": recipe.get("total_fractional_dose"),
                    "target_gain_reached": recipe.get("target_gain_reached"),
                })
                view_tensors.append(tensor)
    if not view_tensors:
        raise RuntimeError("no action-expansion views were constructed")
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for left in range(0, len(view_tensors), args.encode_batch_size):
            block = torch.stack(view_tensors[left:left + args.encode_batch_size]).to(device)
            vectors.extend(forward_embeddings(model, block, False).float().cpu().numpy())
            print(f"[action-expansion-encode] {min(left + len(block), len(view_tensors))}/{len(view_tensors)}", flush=True)
    scored: list[dict[str, object]] = []
    for record, vector in zip(view_records, vectors):
        detail = candidate_detail(
            graph, int(record["query_index"]), vector, embeddings, row_index,
        )
        scored.append({**record, **{f"view_{key}": value for key, value in detail.items()}})
    long = pd.DataFrame(scored)
    recipe_metadata = long.groupby(
        ["query_index", "recipe"], as_index=False, dropna=False,
    )[[
        "predicted_gain", "target_gain", "total_fractional_dose",
        "target_gain_reached",
    ]].first()
    index_columns = [
        "query_index", "query_row", "query_ik14", "query_formula", "panel",
        "baseline_rank", "baseline_margin", "historical_single_best_rank",
        "recipe", "target_down", "target_up", "control_down", "control_up",
    ]
    wide = long.pivot(index=index_columns, columns="view_kind", values=[
        "view_rank", "view_positive", "view_negative", "view_margin",
    ])
    wide.columns = [f"{kind}_{metric.removeprefix('view_')}" for metric, kind in wide.columns]
    wide = wide.reset_index()
    wide = wide.merge(
        recipe_metadata, on=["query_index", "recipe"],
        how="left", validate="one_to_one",
    )
    wide["margin_change"] = wide["target_margin"] - wide["baseline_margin"]
    wide["paired_advantage"] = wide["target_margin"] - wide["control_margin"]
    wide["route"] = "unpaired"
    complete = wide["control_rank"].notna()
    wide.loc[complete, "route"] = [
        route_action(
            clean_rank=int(row.baseline_rank),
            clean_margin=float(row.baseline_margin),
            action_rank=int(row.target_rank),
            action_margin=float(row.target_margin),
            control_margin=float(row.control_margin),
        )
        for row in wide.loc[complete].itertuples(index=False)
    ]
    wide["strict_route_top1_corrected"] = (
        (wide["baseline_rank"] != 1)
        & (wide["target_rank"] == 1)
        & wide["route"].eq("corrective")
    )
    wide["introduced"] = (
        (wide["baseline_rank"] == 1) & (wide["target_rank"] != 1)
    )
    single = wide[wide["recipe"] == "single_attenuation_1x50"][[
        "query_index", "target_rank",
    ]].rename(columns={"target_rank": "single_target_rank"})
    wide = wide.merge(single, on="query_index", how="left", validate="many_to_one")
    recipes = sorted(wide["recipe"].unique())
    report = {
        "status": "noise_corrected_action_expansion_v4_complete",
        "formal": False,
        "shared_encoder_trained": False,
        "teacher_or_distillation_target_used": False,
        "selection": {
            "panel_fixed_before_new_action_execution": True,
            "outer_formula_fold_excluded": args.outer_fold,
            "formula_fold_seed": args.formula_fold_seed,
            "historical_single_peak_residual_queries": int(len(residual)),
            "correct_safety_queries": int(len(safety)),
            "distinct_formulas": int(selected["query_formula"].nunique()),
            "historical_executability_requires_two_attenuation_and_two_supported_boost_peaks": True,
            "current_E8_error_queries_after_recompute": int(
                (query_meta_frame["baseline_rank"] != 1).sum()
            ),
            "current_E8_correct_queries_after_recompute": int(
                (query_meta_frame["baseline_rank"] == 1).sum()
            ),
            "recipe_set": args.recipe_set,
            "sequential_supported_boost_steps": args.sequential_boost_steps,
            "sequential_supported_boost_dose": args.sequential_boost_dose,
        },
        "model": model_provenance,
        "recipes": {
            recipe: summarize(wide, recipe, args.sample_seed) for recipe in recipes
        },
        "contracts": {
            "all_actions_edit_real_fragment_tokens_only": True,
            "identity_only_peaks_never_attenuated": True,
            "boosts_restricted_to_identity_or_shared_support": True,
            "conservative_exchange_preserves_pre_normalization_intensity_sum": True,
            "matched_controls_require_same_peak_role": True,
            "new_action_outcomes_not_used_to_select_panel": True,
            "adaptive_action_dose_uses_clean_margin_and_input_gradient_only": True,
            "sequential_boost_refreshes_current_boundary_and_input_gradient_each_step": bool(
                args.recipe_set in {"sequential", "all"}
            ),
            "result_is_action_space_development_not_encoder_performance": True,
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "input_embedding_cache_sha256": sha256_file(embedding_path),
            "a4_scan_sha256": sha256_file(action_path),
            "a4_query_table_sha256": sha256_file(scan_path),
            "action_core_sha256": sha256_file(
                ROOT / "tasks/noise_corrected_action_expansion_v4.py"
            ),
            "trust_region_action_core_sha256": sha256_file(
                ROOT / "tasks/noise_corrected_trust_region_action_v4.py"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "elapsed_seconds": time.time() - started,
        "claim_limit": (
            "Bounded outer-train action-space discovery. Corrections and controls are "
            "not a clean-input shared-encoder gain and cannot guarantee 4 pp."
        ),
    }
    staging = Path(tempfile.mkdtemp(prefix="noise_action_expansion_v4_", dir=args.output_dir.parent))
    try:
        wide.to_csv(staging / "per_query_action.csv.gz", index=False)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

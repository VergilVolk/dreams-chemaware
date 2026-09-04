"""L2: paired counterfactual transfer of the two mature noise policies.

The treatment and null-control encoders share initialization, query/action
schedule, candidate representatives, optimizer and losses.  Their sole
difference is the primary action path: the treatment receives the frozen
targeted path and the null arm receives one frozen matched-random path.  Both
arms optimize the same primary-versus-comparator advantage loss.

Action eligibility is determined only by formula-OOF L1 predictions from the
clean spectrum.  L0 outcomes are used only to choose clean risk-protection
queries on the outer-training side.  Inference and held evaluation use clean
spectra and one shared query/reference encoder; P2b and P3 are forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_l0_action_learnability_ledger import load_clean_model  # noqa: E402
from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, seed_everything, sha256_file, stable_fold, strict_metrics, strict_rank,
)
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, evaluate_embeddings, formula_bootstrap_delta,
    forward_embeddings, parse_controls, parse_path,
)


PRIMARY_THRESHOLDS = {
    "positive_probability": 0.70,
    "harmful_probability": 0.10,
    "predicted_gain": 0.01,
}
MATCHED_CONTROL_ASSIGNMENT_SEED = 20260903
ALLOWED_CELLS = {
    *(f"candidate_gradient|0.50000000|{step}" for step in range(3, 7)),
    *(f"role_confounder|1.00000000|{step}" for step in range(1, 6)),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_l1_clean_action_learnability")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=(ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution/"
                 "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
                 "e4a_causal_v1_20260901_causal_clean_duplicate/"
                 "seed_20260828/fold_0/final_shared_encoder.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--maximum-train-queries", type=int, default=1024)
    parser.add_argument("--actions-per-query-per-epoch", type=int, default=2)
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--advantage-margin", type=float, default=0.01)
    parser.add_argument("--lambda-clean-rank", type=float, default=1.0)
    parser.add_argument("--lambda-action-rank", type=float, default=1.0)
    parser.add_argument("--lambda-advantage", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.05)
    parser.add_argument("--lambda-margin-floor", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class PairedAction:
    action_index: int
    query_index: int
    query_row: int
    identity: str
    formula: str
    selector: str
    step: int
    attenuation: float
    target_path: tuple[int, ...]
    control_paths: tuple[tuple[int, ...], tuple[int, ...]]
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    initial_margin: float
    predicted_gain: float
    positive_probability: float
    harmful_probability: float


@dataclass(frozen=True)
class CleanProtection:
    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    initial_margin: float


def stable_key(seed: int, *values: object) -> int:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def cell_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["selector"].astype(str) + "|"
        + frame["attenuation"].astype(float).map(lambda value: f"{value:.8f}") + "|"
        + frame["step"].astype(int).astype(str)
    )


def select_oof_actions(frame: pd.DataFrame, outer_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "action_index", "query_index", "query_row", "query_ik14", "query_formula",
        "formula_fold", "selector", "attenuation", "step", "target_path",
        "matched_control_paths", "clean_pred_gain", "clean_p_positive",
        "clean_p_harmful", "advantage_label", "transition",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"L1 OOF table lacks columns: {sorted(missing)}")
    if frame["action_index"].duplicated().any():
        raise RuntimeError("L1 OOF table repeats an action index")
    if not cell_id(frame).isin(ALLOWED_CELLS).all():
        raise RuntimeError("L1 contains an action outside the two frozen mature policies")
    prediction = frame[["clean_pred_gain", "clean_p_positive", "clean_p_harmful"]].to_numpy(float)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("L1 OOF predictions are incomplete")
    passing = (
        (frame["clean_p_positive"].to_numpy(float) >= PRIMARY_THRESHOLDS["positive_probability"])
        & (frame["clean_p_harmful"].to_numpy(float) <= PRIMARY_THRESHOLDS["harmful_probability"])
        & (frame["clean_pred_gain"].to_numpy(float) >= PRIMARY_THRESHOLDS["predicted_gain"])
    )
    selected = frame.loc[passing].copy()
    train = selected.loc[selected["formula_fold"].astype(int).ne(outer_fold)].copy()
    held = selected.loc[selected["formula_fold"].astype(int).eq(outer_fold)].copy()
    return train, held


def deterministic_query_subset(
    frame: pd.DataFrame, maximum: int, seed: int, minimum_per_selector: int = 64,
) -> pd.DataFrame:
    if maximum <= 0 or frame["query_index"].nunique() <= maximum:
        return frame.copy()
    selectors = sorted(frame["selector"].astype(str).unique())
    if maximum < len(selectors):
        raise ValueError("maximum query count cannot retain every mature selector")
    queries = frame[["query_index", "query_formula"]].drop_duplicates("query_index").copy()
    queries["order"] = [stable_key(seed, q, f) for q, f in queries.itertuples(index=False, name=None)]
    keep: set[int] = set()
    # Reserve deterministic query coverage for each selector before global
    # filling.  This prevents a low-frequency mature family from disappearing
    # solely because of the pilot-size hash truncation.
    for selector in selectors:
        selector_queries = frame.loc[
            frame["selector"].astype(str).eq(selector), ["query_index", "query_formula"]
        ].drop_duplicates("query_index").copy()
        selector_queries["order"] = [
            stable_key(seed, selector, query, formula)
            for query, formula in selector_queries.itertuples(index=False, name=None)
        ]
        quota = min(minimum_per_selector, len(selector_queries))
        keep.update(selector_queries.sort_values(["order", "query_index"]).head(quota)["query_index"].astype(int))
    remaining = maximum - len(keep)
    if remaining < 0:
        raise RuntimeError("per-selector pilot reservations exceed the maximum query count")
    fill = queries.loc[~queries["query_index"].astype(int).isin(keep)].sort_values(["order", "query_index"])
    keep.update(fill.head(remaining)["query_index"].astype(int))
    return frame.loc[frame["query_index"].astype(int).isin(keep)].copy()


def current_representatives(
    graph: CandidateGraph, query: int, embedding_by_row: dict[int, np.ndarray], positives: int,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    left, right = map(int, graph.query_ptr[query:query + 2])
    query_z = embedding_by_row[int(graph.query_row[query])]
    selected: list[tuple[int, ...]] = []
    molecule_scores: list[float] = []
    for molecule in range(left, right):
        pair_left, pair_right = map(int, graph.molecule_ptr[molecule:molecule + 2])
        rows = graph.pair_candidate_row[pair_left:pair_right].astype(np.int64)
        scores = np.asarray([float(np.dot(query_z, embedding_by_row[int(row)])) for row in rows])
        order = np.argsort(-scores, kind="stable")
        take = positives if molecule == left else 1
        selected.append(tuple(map(int, rows[order[:take]])))
        molecule_scores.append(float(scores[order[0]]))
    if len(selected) < 2 or not selected[0] or any(not rows for rows in selected[1:]):
        raise RuntimeError(f"query {query} lacks a complete candidate molecule list")
    negative_rows = tuple(rows[0] for rows in selected[1:])
    margin = molecule_scores[0] - max(molecule_scores[1:])
    return selected[0], negative_rows, float(margin)


def build_actions(
    graph: CandidateGraph, frame: pd.DataFrame, embedding_by_row: dict[int, np.ndarray], positives: int,
) -> list[PairedAction]:
    output: list[PairedAction] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        target = parse_path(row.target_path)
        controls = parse_controls(row.matched_control_paths)
        if len(target) != int(row.step) or any(len(path) != int(row.step) for path in controls):
            raise RuntimeError(f"action {row.action_index} path length differs from step")
        if target in controls or controls[0] == controls[1]:
            raise RuntimeError(f"action {row.action_index} has non-distinct target/controls")
        positive_rows, negative_rows, margin = current_representatives(
            graph, query, embedding_by_row, positives,
        )
        output.append(PairedAction(
            action_index=int(row.action_index), query_index=query, query_row=int(row.query_row),
            identity=str(row.query_ik14), formula=str(row.query_formula), selector=str(row.selector),
            step=int(row.step), attenuation=float(row.attenuation), target_path=target,
            control_paths=(controls[0], controls[1]), positive_rows=positive_rows,
            negative_rows=negative_rows, initial_margin=margin,
            predicted_gain=float(row.clean_pred_gain),
            positive_probability=float(row.clean_p_positive),
            harmful_probability=float(row.clean_p_harmful),
        ))
    return output


def build_protections(
    graph: CandidateGraph, queries: list[int], embedding_by_row: dict[int, np.ndarray], positives: int,
) -> list[CleanProtection]:
    output: list[CleanProtection] = []
    for query in queries:
        positive_rows, negative_rows, margin = current_representatives(
            graph, int(query), embedding_by_row, positives,
        )
        output.append(CleanProtection(
            query_index=int(query), query_row=int(graph.query_row[query]),
            identity=str(graph.query_ik14[query]), formula=str(graph.query_formula[query]),
            positive_rows=positive_rows, negative_rows=negative_rows, initial_margin=margin,
        ))
    return output


def action_schedule(
    actions: list[PairedAction], epochs: int, per_query: int, seed: int,
) -> list[list[list[PairedAction]]]:
    if epochs < 1 or per_query < 1:
        raise ValueError("epochs and actions-per-query-per-epoch must be positive")
    grouped: dict[int, list[PairedAction]] = {}
    for action in actions:
        grouped.setdefault(action.query_index, []).append(action)
    for query in grouped:
        grouped[query].sort(key=lambda item: (
            -item.predicted_gain, item.harmful_probability, -item.positive_probability,
            item.selector, item.step, item.action_index,
        ))
    identities: dict[str, list[int]] = {}
    for query, values in grouped.items():
        identities.setdefault(values[0].identity, []).append(query)
    for identity in identities:
        identities[identity].sort(key=lambda query: (stable_key(seed, identity, query), query))
    visits = {query: 0 for query in grouped}
    schedules: list[list[list[PairedAction]]] = []
    for epoch in range(epochs):
        groups: list[list[PairedAction]] = []
        for identity, queries in identities.items():
            query = queries[epoch % len(queries)]
            values = grouped[query]
            offset = (visits[query] * per_query) % len(values)
            count = min(per_query, len(values))
            groups.append([values[(offset + index) % len(values)] for index in range(count)])
            visits[query] += 1
        groups.sort(key=lambda values: stable_key(seed + epoch, values[0].query_index))
        schedules.append(groups)
    return schedules


def arm_paths(action: PairedAction, arm: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    index = stable_key(MATCHED_CONTROL_ASSIGNMENT_SEED, action.action_index, action.query_index) & 1
    selected_control = action.control_paths[index]
    alternate_control = action.control_paths[1 - index]
    if arm == "targeted":
        return action.target_path, selected_control
    if arm == "matched_random":
        return selected_control, alternate_control
    raise ValueError(f"unknown L2 arm: {arm}")


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(1e-12)


def paired_advantage_loss(
    primary_margin: torch.Tensor, comparator_margin: torch.Tensor,
    advantage_margin: float, temperature: float, weights: torch.Tensor,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("paired-advantage temperature must be positive")
    return weighted_mean(
        F.softplus(
            (comparator_margin.detach() + advantage_margin - primary_margin) / temperature
        ),
        weights,
    )


def paired_components(
    model: torch.nn.Module, store: SpectrumStore, actions: list[PairedAction], arm: str,
    initial_by_row: dict[int, np.ndarray], device: torch.device, args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    tensors: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    clean_rows: list[int] = []
    for action in actions:
        primary_path, comparator_path = arm_paths(action, arm)
        layout: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(action.query_row)); clean_rows.append(action.query_row)
        layout["primary"] = len(tensors)
        tensors.append(attenuate_sequence(store.one(action.query_row), primary_path, action.attenuation))
        layout["comparator"] = len(tensors)
        tensors.append(attenuate_sequence(store.one(action.query_row), comparator_path, action.attenuation))
        layout["positive"] = list(range(len(tensors), len(tensors) + len(action.positive_rows)))
        tensors.extend(store.get(action.positive_rows)); clean_rows.extend(action.positive_rows)
        layout["negative"] = list(range(len(tensors), len(tensors) + len(action.negative_rows)))
        tensors.extend(store.get(action.negative_rows)); clean_rows.extend(action.negative_rows)
        layouts.append(layout)
    encoded = forward_embeddings(model, torch.stack(tensors).to(device), args.amp)

    def margins(view: str) -> torch.Tensor:
        output = []
        for layout in layouts:
            query_z = encoded[int(layout[view])]
            positive = torch.max(encoded[layout["positive"]] @ query_z)
            negative = torch.max(encoded[layout["negative"]] @ query_z)
            output.append(positive - negative)
        return torch.stack(output)

    clean_margin = margins("clean")
    primary_margin = margins("primary")
    comparator_margin = margins("comparator")
    weights = torch.full_like(clean_margin, 1.0 / len(actions))
    clean_rank = weighted_mean(F.softplus((args.rank_margin - clean_margin) / args.temperature), weights)
    action_rank = weighted_mean(F.softplus((args.rank_margin - primary_margin) / args.temperature), weights)
    # A smooth paired hinge is intentional.  A hard ReLU made the branch silently
    # disappear for already-satisfied examples, which recreates the E4-A failure:
    # the targeted path then contributes no explicit advantage gradient at all.
    advantage = paired_advantage_loss(
        primary_margin, comparator_margin, args.advantage_margin, args.temperature, weights,
    )
    clean_z = torch.stack([encoded[int(layout["clean"])] for layout in layouts])
    primary_z = torch.stack([encoded[int(layout["primary"])] for layout in layouts])
    consistency = weighted_mean(1.0 - torch.sum(clean_z * primary_z.detach(), dim=1), weights)
    floors = torch.as_tensor(
        [action.initial_margin - args.margin_floor_slack for action in actions],
        dtype=clean_margin.dtype, device=device,
    )
    floor = weighted_mean(F.relu(floors - clean_margin), weights)
    clean_indices: list[int] = []
    for layout in layouts:
        clean_indices.append(int(layout["clean"]))
        clean_indices.extend(layout["positive"]); clean_indices.extend(layout["negative"])
    initial = torch.as_tensor(np.stack([initial_by_row[row] for row in clean_rows]), device=device,
                              dtype=encoded.dtype)
    preserve = 1.0 - torch.sum(encoded[clean_indices] * initial, dim=1).mean()
    components = {
        "clean_rank": clean_rank, "action_rank": action_rank, "paired_advantage": advantage,
        "consistency": consistency, "margin_floor": floor, "preservation": preserve,
    }
    logs = {
        "clean_margin": float(clean_margin.mean().detach()),
        "primary_margin": float(primary_margin.mean().detach()),
        "comparator_margin": float(comparator_margin.mean().detach()),
        **{name: float(value.detach()) for name, value in components.items()},
    }
    return components, logs


def protection_components(
    model: torch.nn.Module, store: SpectrumStore, examples: list[CleanProtection],
    initial_by_row: dict[int, np.ndarray], device: torch.device, args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    tensors: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    rows: list[int] = []
    for example in examples:
        layout: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(example.query_row)); rows.append(example.query_row)
        layout["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows)); rows.extend(example.positive_rows)
        layout["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows)); rows.extend(example.negative_rows)
        layouts.append(layout)
    encoded = forward_embeddings(model, torch.stack(tensors).to(device), args.amp)
    margins = []
    for layout in layouts:
        query_z = encoded[int(layout["clean"])]
        margins.append(torch.max(encoded[layout["positive"]] @ query_z)
                       - torch.max(encoded[layout["negative"]] @ query_z))
    margin = torch.stack(margins)
    floors = torch.as_tensor(
        [example.initial_margin - args.margin_floor_slack for example in examples],
        dtype=margin.dtype, device=device,
    )
    floor = F.relu(floors - margin).mean()
    rank = F.softplus((args.rank_margin - margin) / args.temperature).mean()
    initial = torch.as_tensor(np.stack([initial_by_row[row] for row in rows]), device=device,
                              dtype=encoded.dtype)
    indices: list[int] = []
    for layout in layouts:
        indices.append(int(layout["clean"])); indices.extend(layout["positive"]); indices.extend(layout["negative"])
    preserve = 1.0 - torch.sum(encoded[indices] * initial, dim=1).mean()
    components = {"risk_rank": rank, "risk_floor": floor, "risk_preservation": preserve}
    return components, {name: float(value.detach()) for name, value in components.items()}


def total_loss(parts: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    return (
        args.lambda_clean_rank * parts["clean_rank"]
        + args.lambda_action_rank * parts["action_rank"]
        + args.lambda_advantage * parts["paired_advantage"]
        + args.lambda_consistency * parts["consistency"]
        + args.lambda_margin_floor * parts["margin_floor"]
        + args.lambda_preserve * parts["preservation"]
    )


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return math.sqrt(sum(float(torch.sum(gradient.float() ** 2).detach())
                         for gradient in gradients if gradient is not None))


def gradient_cosine(
    left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...],
) -> float:
    dot = 0.0; left_norm = 0.0; right_norm = 0.0
    for left_gradient, right_gradient in zip(left, right):
        left_float = left_gradient.float() if left_gradient is not None else None
        right_float = right_gradient.float() if right_gradient is not None else None
        if left_float is not None:
            left_norm += float(torch.sum(left_float ** 2).detach())
        if right_float is not None:
            right_norm += float(torch.sum(right_float ** 2).detach())
        if left_float is not None and right_float is not None:
            dot += float(torch.sum(left_float * right_float).detach())
    if left_norm <= 0 or right_norm <= 0:
        return float("nan")
    return dot / math.sqrt(left_norm * right_norm)


def initial_gradient_audit(
    model: torch.nn.Module, store: SpectrumStore, groups: list[list[PairedAction]], arm: str,
    initial_by_row: dict[int, np.ndarray], device: torch.device, args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    """Audit several independent query groups instead of trusting the first batch."""
    names = ("clean_rank", "action_rank", "paired_advantage", "margin_floor")
    values: dict[str, list[float]] = {name: [] for name in names}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    first_cosines: dict[str, float] = {}
    for group_index, group in enumerate(groups[: min(8, len(groups))]):
        parts, _ = paired_components(model, store, group, arm, initial_by_row, device, args)
        if group_index == 0:
            gradient_sets: dict[str, tuple[torch.Tensor | None, ...]] = {}
            for index, name in enumerate(names):
                gradients = torch.autograd.grad(
                    parts[name], trainable, retain_graph=index + 1 < len(names), allow_unused=True,
                )
                values[name].append(math.sqrt(sum(
                    float(torch.sum(gradient.float() ** 2).detach())
                    for gradient in gradients if gradient is not None
                )))
                if name in {"clean_rank", "action_rank", "paired_advantage"}:
                    gradient_sets[name] = gradients
            first_cosines = {
                "paired_advantage_vs_clean_rank": gradient_cosine(
                    gradient_sets["paired_advantage"], gradient_sets["clean_rank"]
                ),
                "paired_advantage_vs_action_rank": gradient_cosine(
                    gradient_sets["paired_advantage"], gradient_sets["action_rank"]
                ),
                "clean_rank_vs_action_rank": gradient_cosine(
                    gradient_sets["clean_rank"], gradient_sets["action_rank"]
                ),
            }
            if not all(np.isfinite(value) for value in first_cosines.values()):
                raise RuntimeError(f"{arm} branch-gradient cosine is undefined")
            del gradient_sets
        else:
            for index, name in enumerate(names):
                values[name].append(gradient_norm(parts[name], trainable, index + 1 < len(names)))
        model.zero_grad(set_to_none=True)
        del parts
    output = {
        name: {
            "groups": int(len(norms)),
            "minimum": float(np.min(norms)),
            "mean": float(np.mean(norms)),
            "maximum": float(np.max(norms)),
            "nonzero_groups": int(np.sum(np.asarray(norms) > 1e-10)),
        }
        for name, norms in values.items()
    }
    if output["paired_advantage"]["nonzero_groups"] != output["paired_advantage"]["groups"]:
        raise RuntimeError(f"{arm} paired-advantage gradient vanished in an audited query group")
    output["cosines_first_audited_group"] = first_cosines
    return output


def initial_risk_gradient_audit(
    model: torch.nn.Module, store: SpectrumStore, protections: list[CleanProtection],
    initial_by_row: dict[int, np.ndarray], device: torch.device, args: argparse.Namespace,
) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norms: list[float] = []
    for protection in protections[: min(8, len(protections))]:
        parts, _ = protection_components(model, store, [protection], initial_by_row, device, args)
        loss = (
            args.lambda_clean_rank * parts["risk_rank"]
            + args.lambda_margin_floor * parts["risk_floor"]
            + args.lambda_preserve * parts["risk_preservation"]
        )
        norms.append(gradient_norm(loss, parameters, False))
        model.zero_grad(set_to_none=True)
        del parts, loss
    output = {
        "groups": int(len(norms)), "minimum": float(np.min(norms)),
        "mean": float(np.mean(norms)), "maximum": float(np.max(norms)),
        "nonzero_groups": int(np.sum(np.asarray(norms) > 1e-10)),
    }
    if output["nonzero_groups"] != output["groups"]:
        raise RuntimeError("clean risk/no-op protection gradient vanished in an audited group")
    return output


def train_arm(
    arm: str, args: argparse.Namespace, store: SpectrumStore,
    schedules: list[list[list[PairedAction]]], protections: list[CleanProtection],
    initial_by_row: dict[int, np.ndarray], reachable_rows: np.ndarray, graph: CandidateGraph,
    held_queries: np.ndarray,
) -> tuple[dict[str, object], dict[str, torch.Tensor], np.ndarray]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, clean_provenance = load_clean_model(args, device)
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    protection_order = sorted(protections, key=lambda item: stable_key(args.seed, item.query_index))
    gradient_audit = initial_gradient_audit(
        model, store, schedules[0], arm, initial_by_row, device, args,
    )
    gradient_audit["risk_total"] = initial_risk_gradient_audit(
        model, store, protection_order, initial_by_row, device, args,
    )

    history: list[dict[str, object]] = []
    for epoch, groups in enumerate(schedules, start=1):
        totals: dict[str, float] = {}; clips = 0; maximum_group = 0
        started = time.time()
        for step, action_group in enumerate(groups):
            maximum_group = max(maximum_group, len(action_group))
            risk = [protection_order[(step + epoch - 1) % len(protection_order)]]
            optimizer.zero_grad(set_to_none=True)
            parts, logs = paired_components(model, store, action_group, arm, initial_by_row, device, args)
            loss = total_loss(parts, args)
            # Backpropagate the corrective graph before constructing the risk
            # graph.  This preserves the exact summed gradient while avoiding
            # two full DreaMS graphs resident on a 32-GiB GPU.
            scaler.scale(loss).backward()
            loss_value = float(loss.detach())
            del parts, loss
            risk_parts, risk_logs = protection_components(
                model, store, risk, initial_by_row, device, args,
            )
            risk_loss = (
                args.lambda_clean_rank * risk_parts["risk_rank"]
                + args.lambda_margin_floor * risk_parts["risk_floor"]
                + args.lambda_preserve * risk_parts["risk_preservation"]
            )
            scaler.scale(risk_loss).backward()
            risk_loss_value = float(risk_loss.detach())
            del risk_parts, risk_loss
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
            clips += int(norm > args.grad_clip)
            scaler.step(optimizer); scaler.update()
            record = {"loss": loss_value + risk_loss_value, "gradient_norm": norm, **logs, **risk_logs}
            for name, value in record.items(): totals[name] = totals.get(name, 0.0) + float(value)
            if (step + 1) % 100 == 0 or step + 1 == len(groups):
                print(f"[L2 {arm} epoch={epoch}] {step + 1}/{len(groups)} loss={totals['loss']/(step+1):.5f}", flush=True)
        history.append({
            "epoch": epoch, "steps": len(groups), "seconds": time.time() - started,
            "clip_fraction": clips / max(len(groups), 1), "maximum_actions_in_query_step": maximum_group,
            **{name: value / max(len(groups), 1) for name, value in totals.items()},
        })

    encoded = encode_rows(model, store, reachable_rows, device, args.eval_batch_size, False, f"L2-{arm}-final")
    rank, summary = evaluate_embeddings(graph, reachable_rows, encoded, held_queries)
    initialization = np.stack([initial_by_row[int(row)] for row in reachable_rows])
    preservation = np.sum(encoded * initialization, axis=1)
    summary["initialization_preservation"] = {
        "mean": float(np.mean(preservation)),
        "p01": float(np.quantile(preservation, 0.01)),
        "minimum": float(np.min(preservation)),
    }
    checkpoint = {
        "status": f"noise_final_l2_{arm}_shared_encoder", "arm": arm,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "clean_provenance": clean_provenance, "capacity": capacity, "seed": args.seed,
        "outer_fold": args.outer_fold, "P2b_used": False, "P3_consumed": False,
        "inference_clean_only": True,
    }
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"summary": summary, "history": history, "gradient_audit": gradient_audit,
            "capacity": capacity, "clean_provenance": clean_provenance}, checkpoint, rank


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5): raise ValueError("outer-fold must be 0..4")
    if args.outer_fold != 0:
        raise ValueError("the frozen L2 pilot and its clean-continuation checkpoint are outer fold 0 only")
    if args.formula_fold_seed != 20260825:
        raise ValueError("L2 must use the frozen D0 formula-fold seed 20260825")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("L2 requires CUDA")
    if args.output_dir.exists(): raise RuntimeError(f"completed L2 output already exists: {args.output_dir}")
    if args.bootstrap_resamples < 100 or args.actions_per_query_per_epoch < 1: raise ValueError("invalid L2 numeric argument")
    if args.head_lr < args.backbone_lr or args.backbone_lr <= 0: raise ValueError("invalid learning rates")
    required = [
        args.l1_dir / "report.json", args.l1_dir / "action_oof_predictions.csv.gz",
        args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
        args.clean_checkpoint, args.clean_checkpoint.parent / "decision.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    report = json.loads((args.l1_dir / "report.json").read_text(encoding="utf-8"))
    if (report.get("status") != "noise_final_l1_clean_action_learnability_complete"
            or not report.get("formal") or not report.get("pass_to_l2_small_causal_pilot")
            or report.get("feature_contract", {}).get("P2b") != "forbidden"
            or report.get("feature_contract", {}).get("P3_consumed")):
        raise RuntimeError("L2 requires the formal passing P2b/P3-free L1 result")
    if report.get("fixed_policy_thresholds", {}).get("primary") != PRIMARY_THRESHOLDS:
        raise RuntimeError("L1 primary thresholds differ from the frozen L2 contract")
    l1_provenance = report.get("provenance", {})
    expected_provenance = {
        "graph_sha256": sha256_file(args.graph),
        "hdf5_sha256": sha256_file(args.data),
        "clean_checkpoint_sha256": sha256_file(args.clean_checkpoint),
    }
    mismatched = {
        name: {"reported": l1_provenance.get(name), "expected": expected}
        for name, expected in expected_provenance.items()
        if l1_provenance.get(name) != expected
    }
    if mismatched:
        raise RuntimeError(f"L1 provenance differs from L2 inputs: {mismatched}")

    frame = pd.read_csv(args.l1_dir / "action_oof_predictions.csv.gz", low_memory=False)
    graph = CandidateGraph(args.graph)
    query = frame["query_index"].to_numpy(np.int64)
    if np.any((query < 0) | (query >= graph.n_queries)):
        raise RuntimeError("L1 query index outside candidate graph")
    if not np.array_equal(frame["query_row"].to_numpy(np.int64), graph.query_row[query]):
        raise RuntimeError("L1 query rows drifted from candidate graph")
    if not np.array_equal(frame["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query].astype(str)):
        raise RuntimeError("L1 identities drifted from candidate graph")
    if not np.array_equal(frame["query_formula"].astype(str).to_numpy(), graph.query_formula[query].astype(str)):
        raise RuntimeError("L1 formulas drifted from candidate graph")
    formula_fold_counts = frame.groupby("query_formula")["formula_fold"].nunique()
    if not formula_fold_counts.eq(1).all() or sorted(frame["formula_fold"].astype(int).unique()) != list(range(5)):
        raise RuntimeError("L1 formula-fold assignment is inconsistent or incomplete")
    expected_action_folds = frame["query_formula"].astype(str).map(
        lambda formula: stable_fold(formula, 5, args.formula_fold_seed)
    ).to_numpy(np.int8)
    if not np.array_equal(expected_action_folds, frame["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("L1 action folds differ from the frozen D0 formula-fold rule")
    train_selected, held_selected = select_oof_actions(frame, args.outer_fold)
    train_selector_queries = train_selected.groupby("selector")["query_index"].nunique().astype(int).to_dict()
    # L0 establishes that both action families can work when their outcomes are
    # known. L1 is the deployability gate: a family with zero clean-OOF primary
    # selections becomes no-op rather than being forced into encoder training.
    if not train_selector_queries:
        raise RuntimeError("L1 primary thresholds select no outer-train action")
    pilot_selected = deterministic_query_subset(train_selected, args.maximum_train_queries, args.seed)
    if pilot_selected["query_index"].nunique() < 256 or pilot_selected["query_formula"].nunique() < 100:
        raise RuntimeError("L2 pilot lacks 256 queries or 100 outer-train formulas")
    if pilot_selected["query_ik14"].nunique() < 300:
        raise RuntimeError("L2 pilot lacks 300 outer-train identities")
    active_selectors = set(pilot_selected["selector"].astype(str))
    if not active_selectors or not active_selectors.issubset({"candidate_gradient", "role_confounder"}):
        raise RuntimeError(f"L2 contains an invalid active selector set: {sorted(active_selectors)}")

    reachable_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable_rows, args.n_highest_peaks)
    device = torch.device(args.device)
    initial_model, _ = load_clean_model(args, device)
    initial_encoded = encode_rows(initial_model, store, reachable_rows, device, args.eval_batch_size,
                                  False, "L2-initial-clean")
    initial_by_row = {int(row): initial_encoded[index] for index, row in enumerate(reachable_rows)}
    del initial_model
    if device.type == "cuda": torch.cuda.empty_cache()

    actions = build_actions(graph, pilot_selected, initial_by_row, args.positive_spectra)
    schedules = action_schedule(actions, 1 if args.smoke else args.epochs,
                                args.actions_per_query_per_epoch, args.seed)
    if args.smoke:
        schedules = [[group for group in schedules[0][:8]]]
    train_fold = frame["formula_fold"].astype(int).ne(args.outer_fold)
    selected_query_set = set(pilot_selected["query_index"].astype(int))
    harmful = frame.loc[
        train_fold & (frame["advantage_label"].astype(str).eq("harmful")
                      | frame["transition"].astype(str).eq("introduced")), "query_index"
    ].astype(int).tolist()
    no_op = frame.loc[train_fold & ~frame["query_index"].astype(int).isin(
        set(train_selected["query_index"].astype(int))), "query_index"].astype(int).tolist()
    protection_queries = sorted((set(harmful) | set(no_op)) - selected_query_set,
                                key=lambda value: stable_key(args.seed, value))
    if len(protection_queries) < 256: raise RuntimeError("L2 lacks 256 independent clean risk/no-op queries")
    protections = build_protections(graph, protection_queries[:2048], initial_by_row, args.positive_spectra)
    # L1 contains only action-covered queries.  Final retrieval must instead use
    # every query in the frozen D0 formula fold, including action-uncovered ones.
    graph_folds = np.asarray([
        stable_fold(str(formula), 5, args.formula_fold_seed) for formula in graph.query_formula
    ], dtype=np.int8)
    held_queries = np.flatnonzero(graph_folds == args.outer_fold).astype(np.int64)
    held_formulas = set(graph.query_formula[held_queries].astype(str))
    if set(pilot_selected["query_formula"].astype(str)) & held_formulas:
        raise RuntimeError("L2 training actions overlap held formulas")
    initial_rank, initial_summary = evaluate_embeddings(graph, reachable_rows, initial_encoded, held_queries)
    official_rank = np.asarray([
        strict_rank(graph.official_molecule_scores(int(query_index))) for query_index in held_queries
    ], dtype=np.int16)
    official_summary = strict_metrics(official_rank, graph.query_has_near[held_queries])

    exposure_by_action: dict[int, int] = {}
    exposure_by_query: dict[int, int] = {}
    exposure_by_identity: dict[str, int] = {}
    for epoch in schedules:
        seen_in_epoch: set[int] = set()
        for group in epoch:
            for action in group:
                if action.action_index in seen_in_epoch:
                    raise RuntimeError(f"action {action.action_index} is recycled within one epoch")
                seen_in_epoch.add(action.action_index)
                exposure_by_action[action.action_index] = exposure_by_action.get(action.action_index, 0) + 1
                exposure_by_query[action.query_index] = exposure_by_query.get(action.query_index, 0) + 1
                exposure_by_identity[action.identity] = exposure_by_identity.get(action.identity, 0) + 1
    exposure = {
        "epochs": int(len(schedules)),
        "optimizer_steps_per_arm": int(sum(len(epoch) for epoch in schedules)),
        "maximum_action_exposure": int(max(exposure_by_action.values())),
        "maximum_query_exposure": int(max(exposure_by_query.values())),
        "maximum_identity_exposure": int(max(exposure_by_identity.values())),
        "within_epoch_action_recycling": False,
        "exposed_unique_actions": int(len(exposure_by_action)),
        "ledger_actions_not_exposed_in_pilot": int(len(actions) - len(exposure_by_action)),
    }
    pilot_export = pilot_selected.copy()
    control_indices = np.asarray([
        stable_key(MATCHED_CONTROL_ASSIGNMENT_SEED, int(action_index), int(query_index)) & 1
        for action_index, query_index in pilot_export[["action_index", "query_index"]].itertuples(
            index=False, name=None,
        )
    ], dtype=np.int8)
    parsed_controls = [parse_controls(value) for value in pilot_export["matched_control_paths"]]
    pilot_export["selected_control_index"] = control_indices
    pilot_export["selected_control_path"] = [
        ",".join(map(str, controls[int(index)]))
        for controls, index in zip(parsed_controls, control_indices)
    ]
    pilot_export["alternate_control_path"] = [
        ",".join(map(str, controls[1 - int(index)]))
        for controls, index in zip(parsed_controls, control_indices)
    ]
    exposed_mask = pilot_selected["action_index"].astype(int).isin(exposure_by_action)

    arm_outputs: dict[str, dict[str, object]] = {}; checkpoints: dict[str, dict[str, torch.Tensor]] = {}; ranks = {}
    for arm in ("matched_random", "targeted"):
        result, checkpoint, rank = train_arm(
            arm, args, store, schedules, protections, initial_by_row, reachable_rows, graph, held_queries,
        )
        arm_outputs[arm] = result; checkpoints[arm] = checkpoint; ranks[arm] = rank
    control_rank = ranks["matched_random"]; targeted_rank = ranks["targeted"]
    formulas = graph.query_formula[held_queries].astype(str); near = graph.query_has_near[held_queries]
    paired_ci = formula_bootstrap_delta(control_rank, targeted_rank, formulas,
                                        args.bootstrap_resamples, args.seed + 4000)
    control_correct = control_rank == 1; targeted_correct = targeted_rank == 1
    paired = {
        "delta_recall1": float(np.mean(targeted_correct) - np.mean(control_correct)),
        "corrected": int(np.sum(~control_correct & targeted_correct)),
        "introduced": int(np.sum(control_correct & ~targeted_correct)),
        "risk_net_lambda2": int(np.sum(~control_correct & targeted_correct)
                                - 2 * np.sum(control_correct & ~targeted_correct)),
        "near_delta_recall1": float(np.mean(targeted_correct[near]) - np.mean(control_correct[near])),
        "formula_cluster_ci": paired_ci,
    }
    for arm in arm_outputs:
        arm_rank = ranks[arm]
        summary = arm_outputs[arm]["summary"]
        summary.update({
            "initial_recall1": initial_summary["recall1"],
            "delta_recall1_vs_initial": float(np.mean(arm_rank == 1) - np.mean(initial_rank == 1)),
            "corrected_vs_initial": int(np.sum((initial_rank != 1) & (arm_rank == 1))),
            "introduced_vs_initial": int(np.sum((initial_rank == 1) & (arm_rank != 1))),
            "formula_ci_vs_initial": formula_bootstrap_delta(
                initial_rank, arm_rank, formulas, args.bootstrap_resamples,
                args.seed + (100 if arm == "targeted" else 200),
            ),
            "official_recall1": official_summary["recall1"],
            "delta_recall1_vs_official": float(np.mean(arm_rank == 1) - np.mean(official_rank == 1)),
            "corrected_vs_official": int(np.sum((official_rank != 1) & (arm_rank == 1))),
            "introduced_vs_official": int(np.sum((official_rank == 1) & (arm_rank != 1))),
            "formula_ci_vs_official": formula_bootstrap_delta(
                official_rank, arm_rank, formulas, args.bootstrap_resamples,
                args.seed + (300 if arm == "targeted" else 400),
            ),
        })
    gates = {
        "paired_formula_ci_positive": bool(paired_ci["ci_low"] > 0),
        "paired_corrected_gt_introduced": bool(paired["corrected"] > paired["introduced"]),
        "paired_risk_net_positive": bool(paired["risk_net_lambda2"] > 0),
        "paired_near_nonnegative": bool(paired["near_delta_recall1"] >= 0),
        "both_arms_advantage_gradient_nonzero": bool(all(
            arm_outputs[arm]["gradient_audit"]["paired_advantage"]["nonzero_groups"]
            == arm_outputs[arm]["gradient_audit"]["paired_advantage"]["groups"]
            for arm in arm_outputs
        )),
        "both_arms_risk_gradient_nonzero": bool(all(
            arm_outputs[arm]["gradient_audit"]["risk_total"]["nonzero_groups"]
            == arm_outputs[arm]["gradient_audit"]["risk_total"]["groups"]
            for arm in arm_outputs
        )),
        "both_arms_preservation_ge_0_995": bool(all(
            arm_outputs[arm]["summary"]["initialization_preservation"]["mean"] >= 0.995
            for arm in arm_outputs
        )),
    }
    report_out = {
        "status": "noise_final_l2_paired_counterfactual_complete", "formal": not args.smoke,
        "outer_formula_fold": args.outer_fold, "seed": args.seed,
        "formula_fold_seed": args.formula_fold_seed,
        "frozen_strategies": sorted(ALLOWED_CELLS),
        "active_l1_deployable_selectors": sorted(active_selectors),
        "inactive_mature_selectors_routed_to_no_op": sorted(
            {"candidate_gradient", "role_confounder"} - active_selectors
        ),
        "selection": {
            "all_l1_actions": int(len(frame)), "all_oof_passing_train_actions": int(len(train_selected)),
            "all_oof_passing_train_queries": int(train_selected["query_index"].nunique()),
            "all_oof_passing_train_queries_by_selector": train_selector_queries,
            "pilot_actions": int(len(pilot_selected)), "pilot_queries": int(pilot_selected["query_index"].nunique()),
            "pilot_identities": int(pilot_selected["query_ik14"].nunique()),
            "pilot_formulas": int(pilot_selected["query_formula"].nunique()),
            "held_oof_passing_actions_not_trained": int(len(held_selected)),
            "full_held_fold_queries": int(len(held_queries)),
            "full_held_fold_formulas": int(len(held_formulas)),
            "action_cell_counts": pilot_selected.assign(cell=cell_id(pilot_selected))["cell"].value_counts().astype(int).to_dict(),
            "exposed_action_cell_counts": pilot_selected.loc[exposed_mask].assign(
                cell=cell_id(pilot_selected.loc[exposed_mask])
            )["cell"].value_counts().astype(int).to_dict(),
            "maximum_actions_per_query_in_ledger": int(pilot_selected.groupby("query_index").size().max()),
            "actions_per_query_per_epoch": args.actions_per_query_per_epoch,
            "protection_queries": int(len(protections)),
            "exposure": exposure,
        },
        "official": official_summary,
        "initial": {
            **initial_summary,
            "delta_recall1_vs_official": float(np.mean(initial_rank == 1) - np.mean(official_rank == 1)),
            "formula_ci_vs_official": formula_bootstrap_delta(
                official_rank, initial_rank, formulas, args.bootstrap_resamples, args.seed + 500,
            ),
        },
        "arms": arm_outputs, "targeted_vs_matched_random": paired,
        "gates": gates, "pass_to_second_seed": bool((not args.smoke) and all(gates.values())),
        "contracts": {
            "mature_n_arm_action_universe_only": True,
            "zero_clean_oof_selector_is_routed_to_no_op": True,
            "selection_uses_formula_oof_clean_input_predictions": True,
            "multiple_actions_retained_in_ledger": True,
            "one_query_optimizer_step_with_bounded_rotating_actions": True,
            "one_query_per_identity_per_epoch": True,
            "action_priority_uses_clean_oof_predictions_only": True,
            "same_schedule_optimizer_and_loss_for_both_arms": True,
            "arm_difference_is_primary_action_path_only": True,
            "matched_random_paths_frozen_before_training": True,
            "matched_control_assignment_seed": MATCHED_CONTROL_ASSIGNMENT_SEED,
            "all_candidate_molecules_represented_in_training": True,
            "full_candidate_spectra_used_in_final_evaluation": True,
            "risk_and_no_op_payloads_not_used_as_positive_teacher": True,
            "inference_clean_spectrum_only": True, "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            "l1_report_sha256": sha256_file(args.l1_dir / "report.json"),
            "l1_predictions_sha256": sha256_file(args.l1_dir / "action_oof_predictions.csv.gz"),
            "graph_sha256": sha256_file(args.graph), "hdf5_sha256": sha256_file(args.data),
            "clean_checkpoint_sha256": sha256_file(args.clean_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "One paired development-fold pilot; only targeted-minus-matched-random clean full-list retrieval is causal evidence. Not P3 or multifold performance.",
    }
    per_query = pd.DataFrame({
        "query_index": held_queries, "query_formula": formulas, "has_near": near,
        "initial_rank": initial_rank, "matched_random_rank": control_rank, "targeted_rank": targeted_rank,
    })
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_l2_", dir=args.output_dir.parent))
    try:
        pilot_export.to_csv(temporary / "training_actions.csv.gz", index=False, compression="gzip")
        per_query.to_csv(temporary / "held_per_query.csv.gz", index=False, compression="gzip")
        torch.save(checkpoints["matched_random"], temporary / "matched_random_shared_encoder.pt")
        torch.save(checkpoints["targeted"], temporary / "targeted_shared_encoder.pt")
        json_dump(temporary / "report.json", report_out)
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    print(json.dumps(report_out, indent=2), flush=True)


if __name__ == "__main__":
    main()

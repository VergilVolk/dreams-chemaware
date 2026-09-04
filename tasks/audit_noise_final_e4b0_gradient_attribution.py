"""E4-B0: paired target/random gradient attribution without optimizer steps.

This audit is deliberately downstream of the failed E4-A attribution gate and
upstream of any new training.  It asks whether the target-specific gradient is
non-zero, cross-formula coherent, and aligned with improving the clean-query
ranking loss.  It never updates model weights or consumes P3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything, sha256_file,
)
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_e4a_direct_augmentation import (  # noqa: E402
    FIXED_POLICY, SpectrumStore, _stable_control_index, official_rank_margin,
    unfreeze_last_blocks,
)
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    forward_embeddings, parse_controls, parse_path, representatives,
)


@dataclass(frozen=True)
class GradientExample:
    query_index: int
    query_row: int
    identity: str
    formula: str
    selector: str
    step: int
    attenuation: float
    target_path: tuple[int, ...]
    control_paths: tuple[tuple[int, ...], tuple[int, ...]]
    selected_control: int
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    official_margin: float
    official_rank: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz",
    )
    parser.add_argument(
        "--r0-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--embedding-cache", type=Path,
        default=ROOT / "data/validation/g8r_p2_official_embeddings.npz",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=(
            ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution/"
            "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
            "e4a_causal_v1_20260901_causal_clean_duplicate/"
            "seed_20260828/fold_0/final_shared_encoder.pt"
        ),
    )
    parser.add_argument(
        "--attribution-summary", type=Path,
        default=(
            ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution_summary/"
            "e4a_causal_v1_20260901/seed_20260828/fold_0/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4b0_gradient_attribution",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--panel-formulas", type=int, default=32)
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--specificity-margin", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stable_key(seed: int, *values: object) -> str:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_formula_panel(
    actions: pd.DataFrame, panel_formulas: int, seed: int,
) -> pd.DataFrame:
    """Select four distinct queries per formula without using action outcomes."""
    required = {
        "query_index", "query_row", "query_ik14", "query_formula", "selector",
        "attenuation", "step", "target_path", "matched_control_paths",
        "hard_negative_row", "formula_fold",
    }
    missing = required - set(actions.columns)
    if missing:
        raise RuntimeError(f"R0 action ledger lacks columns: {sorted(missing)}")
    work = actions.copy()
    work["_order"] = [
        stable_key(seed, row.query_formula, row.query_index, row.selector, row.step)
        for row in work.itertuples(index=False)
    ]
    selected_by_formula: dict[str, pd.DataFrame] = {}
    for formula, group in work.groupby("query_formula", sort=True):
        group = group.sort_values("_order", kind="stable")
        if group["query_index"].nunique() < 4:
            continue
        if set(group["selector"].astype(str)) != {"candidate_gradient", "role_confounder"}:
            continue
        candidate = group.loc[group["selector"].astype(str).eq("candidate_gradient")]
        confounder = group.loc[group["selector"].astype(str).eq("role_confounder")]
        if candidate["query_index"].nunique() < 2 or confounder["query_index"].nunique() < 2:
            continue
        chosen: list[pd.Series] = []
        used_queries: set[int] = set()
        formula_offset = int(stable_key(seed, formula)[:8], 16)
        desired = {
            "candidate_gradient": (
                3 + formula_offset % 4, 3 + (formula_offset + 1) % 4,
            ),
            "role_confounder": (
                1 + formula_offset % 5, 1 + (formula_offset + 1) % 5,
            ),
        }
        for selector in ("candidate_gradient", "role_confounder"):
            block = group.loc[group["selector"].astype(str).eq(selector)]
            for step in desired[selector]:
                options = block.loc[
                    block["step"].astype(int).eq(step)
                    & ~block["query_index"].astype(int).isin(used_queries)
                ]
                if options.empty:
                    options = block.loc[
                        ~block["query_index"].astype(int).isin(used_queries)
                    ]
                if options.empty:
                    break
                row = options.sort_values("_order", kind="stable").iloc[0]
                chosen.append(row)
                used_queries.add(int(row["query_index"]))
        if len(chosen) == 4 and len(used_queries) == 4:
            selected_by_formula[str(formula)] = pd.DataFrame(chosen)
    eligible = sorted(
        selected_by_formula,
        key=lambda formula: stable_key(seed + 1, formula),
    )
    if len(eligible) < panel_formulas:
        raise RuntimeError(
            f"only {len(eligible)} formulas have four paired actions; need {panel_formulas}"
        )
    panel = pd.concat(
        [selected_by_formula[formula] for formula in eligible[:panel_formulas]],
        ignore_index=True,
    ).drop(columns="_order")
    if (
        len(panel) != 4 * panel_formulas
        or panel.groupby("query_formula")["query_index"].nunique().ne(4).any()
        or panel.groupby(["query_formula", "selector"]).size().ne(2).any()
    ):
        raise RuntimeError("formula panel lost its frozen 2x2 query/action structure")
    return panel.sort_values(
        ["query_formula", "selector", "step", "query_index"], kind="stable"
    ).reset_index(drop=True)


def make_gradient_examples(
    graph: CandidateGraph, panel: pd.DataFrame,
    official_rank: np.ndarray, official_margin: np.ndarray,
    positives: int, negatives: int,
) -> list[GradientExample]:
    examples: list[GradientExample] = []
    for _, row in panel.iterrows():
        query = int(row["query_index"])
        positive_rows, negative_rows = representatives(
            graph, query, positives, negatives, int(row["hard_negative_row"]),
        )
        target = parse_path(row["target_path"])
        controls = parse_controls(row["matched_control_paths"])
        step = int(row["step"])
        if len(target) != step or any(len(path) != step for path in controls):
            raise RuntimeError("panel target/control path does not reproduce frozen step")
        if target in controls or controls[0] == controls[1]:
            raise RuntimeError("panel target and matched controls are not distinct")
        examples.append(GradientExample(
            query_index=query,
            query_row=int(row["query_row"]),
            identity=str(row["query_ik14"]),
            formula=str(row["query_formula"]),
            selector=str(row["selector"]),
            step=step,
            attenuation=float(row["attenuation"]),
            target_path=target,
            control_paths=(controls[0], controls[1]),
            selected_control=_stable_control_index(row),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            official_margin=float(official_margin[query]),
            official_rank=int(official_rank[query]),
        ))
    return examples


def flatten_formula_batch(
    store: SpectrumStore, examples: list[GradientExample],
) -> tuple[torch.Tensor, list[dict[str, object]], list[int]]:
    tensors: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    clean_rows: list[int] = []
    for example in examples:
        clean = store.one(example.query_row)
        layout: dict[str, object] = {"clean": len(tensors)}
        tensors.append(clean)
        clean_rows.append(example.query_row)
        layout["target"] = len(tensors)
        tensors.append(attenuate_sequence(clean, example.target_path, example.attenuation))
        control_indices: list[int] = []
        for path in example.control_paths:
            control_indices.append(len(tensors))
            tensors.append(attenuate_sequence(clean, path, example.attenuation))
        layout["controls"] = control_indices
        layout["positive"] = list(range(
            len(tensors), len(tensors) + len(example.positive_rows)
        ))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        layout["negative"] = list(range(
            len(tensors), len(tensors) + len(example.negative_rows)
        ))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layouts.append(layout)
    return torch.stack(tensors), layouts, clean_rows


def view_margins(
    encoded: torch.Tensor, layouts: list[dict[str, object]], view: str,
    control_offset: int | None = None,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for layout in layouts:
        index = (
            int(layout["controls"][int(control_offset)])
            if control_offset is not None else int(layout[view])
        )
        query = encoded[index]
        positive = torch.max(encoded[layout["positive"]] @ query)
        negative = torch.max(encoded[layout["negative"]] @ query)
        values.append(positive - negative)
    return torch.stack(values)


def loss_bundle(
    model: torch.nn.Module, store: SpectrumStore, examples: list[GradientExample],
    official_by_row: dict[int, np.ndarray], device: torch.device, args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    spectra, layouts, clean_rows = flatten_formula_batch(store, examples)
    encoded = forward_embeddings(model, spectra.to(device), False)
    clean_margin = view_margins(encoded, layouts, "clean")
    target_margin = view_margins(encoded, layouts, "target")
    control_margin = torch.stack([
        view_margins(encoded, layouts, "control", 0),
        view_margins(encoded, layouts, "control", 1),
    ], dim=1)
    selected_index = torch.tensor(
        [example.selected_control for example in examples], device=device, dtype=torch.long,
    )
    selected_random_margin = control_margin.gather(1, selected_index[:, None]).squeeze(1)
    mean_random_margin = control_margin.mean(dim=1)
    rank = lambda margin: F.softplus(
        (args.rank_margin - margin) / args.temperature
    ).mean()
    clean_rank = rank(clean_margin)
    target_rank = rank(target_margin)
    selected_random_rank = rank(selected_random_margin)
    clean_z = torch.stack([encoded[int(layout["clean"])] for layout in layouts])
    target_z = torch.stack([encoded[int(layout["target"])] for layout in layouts])
    selected_random_z = torch.stack([
        encoded[int(layout["controls"][example.selected_control])]
        for layout, example in zip(layouts, examples)
    ])
    target_consistency = (1.0 - torch.sum(clean_z * target_z, dim=1)).mean()
    random_consistency = (1.0 - torch.sum(clean_z * selected_random_z, dim=1)).mean()
    floors = torch.tensor(
        [example.official_margin - 0.005 for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor = F.relu(floors - clean_margin).mean()
    clean_indices: list[int] = []
    for layout in layouts:
        clean_indices.append(int(layout["clean"]))
        clean_indices.extend(layout["positive"])
        clean_indices.extend(layout["negative"])
    official = torch.from_numpy(np.stack([
        official_by_row[int(row)] for row in clean_rows
    ])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    common_loss = clean_rank + 2.0 * floor + 5.0 * preserve
    current_target_branch = target_rank + 0.25 * target_consistency
    current_random_branch = selected_random_rank + 0.25 * random_consistency
    current_specific_difference = (
        current_target_branch - current_random_branch
        + 0.25 * (target_consistency - random_consistency)
    )
    paired_advantage_loss = F.softplus(
        (selected_random_margin + args.specificity_margin - target_margin)
        / args.temperature
    ).mean()
    outputs: dict[str, object] = {
        "clean_margin": clean_margin,
        "target_margin": target_margin,
        "selected_random_margin": selected_random_margin,
        "mean_random_margin": mean_random_margin,
        "paired_advantage": target_margin - selected_random_margin,
        "paired_advantage_two_control": target_margin - mean_random_margin,
        "paired_violation": target_margin < selected_random_margin + args.specificity_margin,
        "clean_rank": clean_rank,
        "target_rank": target_rank,
        "selected_random_rank": selected_random_rank,
        "target_consistency": target_consistency,
        "random_consistency": random_consistency,
    }
    return {
        "common": common_loss,
        "current_target_branch": current_target_branch,
        "current_target_minus_random": current_specific_difference,
        "paired_advantage": paired_advantage_loss,
    }, outputs


def gradients(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool,
) -> list[torch.Tensor | None]:
    return [
        value.detach() if value is not None else None
        for value in torch.autograd.grad(
            loss, parameters, retain_graph=retain_graph,
            create_graph=False, allow_unused=True,
        )
    ]


def grad_dot(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None],
    mask: np.ndarray | None = None,
) -> float:
    total = 0.0
    for index, (left, right) in enumerate(zip(first, second)):
        if mask is not None and not bool(mask[index]):
            continue
        if left is not None and right is not None:
            total += float(torch.sum(left.float() * right.float()).detach())
    return total


def grad_norm(values: list[torch.Tensor | None], mask: np.ndarray | None = None) -> float:
    return math.sqrt(max(grad_dot(values, values, mask), 0.0))


def grad_cosine(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None],
    mask: np.ndarray | None = None,
) -> float:
    denominator = grad_norm(first, mask) * grad_norm(second, mask)
    return grad_dot(first, second, mask) / denominator if denominator > 0 else float("nan")


def bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("formula bootstrap received invalid values")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def load_checkpoint_model(
    label: str, args: argparse.Namespace, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object], list[str], list[torch.nn.Parameter]]:
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    provenance: dict[str, object] = {"initialization": initialization}
    if label == "clean_continuation":
        package = torch_load_compat(args.clean_checkpoint, map_location="cpu")
        decision_path = args.clean_checkpoint.parent / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or package.get("causal_arm") != "clean_duplicate"
            or package.get("P2b_used")
            or not package.get("inference_clean_only")
            or decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
            or not decision.get("formal")
            or decision.get("configuration", {}).get("causal_arm") != "clean_duplicate"
        ):
            raise RuntimeError("clean-continuation checkpoint violates E4-A causal contract")
        model.load_state_dict(package["model_state"], strict=True)
        provenance.update({
            "checkpoint_sha256": sha256_file(args.clean_checkpoint),
            "decision_sha256": sha256_file(decision_path),
        })
    elif label != "official_initialization":
        raise ValueError(label)
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    names: list[str] = []
    parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
            parameters.append(parameter)
    if not parameters or any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("trainable gradient parameter state is invalid")
    provenance["capacity"] = capacity
    return model, provenance, names, parameters


def audit_checkpoint(
    label: str, args: argparse.Namespace, store: SpectrumStore,
    examples_by_formula: dict[str, list[GradientExample]],
    official_by_row: dict[int, np.ndarray], device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    model, provenance, names, parameters = load_checkpoint_model(label, args, device)
    head_mask = np.asarray([name.startswith("head.") for name in names], dtype=bool)
    backbone_mask = ~head_mask
    accumulated: list[torch.Tensor | None] | None = None
    records: list[dict[str, object]] = []
    ordered = sorted(examples_by_formula)
    for index, formula in enumerate(ordered):
        batch = examples_by_formula[formula]
        losses, outputs = loss_bundle(model, store, batch, official_by_row, device, args)
        common = gradients(losses["common"], parameters, True)
        target_branch = gradients(losses["current_target_branch"], parameters, True)
        common_norm = grad_norm(common)
        target_branch_norm = grad_norm(target_branch)
        common_target_dot = grad_dot(common, target_branch)
        current_norm = math.sqrt(max(
            common_norm ** 2 + target_branch_norm ** 2 + 2.0 * common_target_dot,
            0.0,
        ))
        target_branch_alignment = (
            common_target_dot / (common_norm * target_branch_norm)
            if common_norm > 0 and target_branch_norm > 0 else float("nan")
        )
        del target_branch
        specific = gradients(losses["current_target_minus_random"], parameters, True)
        specific_norm = grad_norm(specific)
        specific_alignment = grad_cosine(common, specific)
        del specific
        advantage = gradients(losses["paired_advantage"], parameters, False)
        if accumulated is None:
            accumulated = [
                torch.zeros_like(value, device="cpu") if value is not None else None
                for value in advantage
            ]
        for destination, value in zip(accumulated, advantage):
            if destination is not None and value is not None:
                destination.add_(value.float().cpu())
        advantage_norm = grad_norm(advantage)
        record = {
            "checkpoint": label,
            "formula": formula,
            "actions": len(batch),
            "identities": len({item.identity for item in batch}),
            "official_errors": sum(item.official_rank != 1 for item in batch),
            "candidate_actions": sum(item.selector == "candidate_gradient" for item in batch),
            "confounder_actions": sum(item.selector == "role_confounder" for item in batch),
            "mean_clean_margin": float(outputs["clean_margin"].mean().detach()),
            "mean_target_margin": float(outputs["target_margin"].mean().detach()),
            "mean_selected_random_margin": float(
                outputs["selected_random_margin"].mean().detach()
            ),
            "mean_target_minus_selected_random": float(
                outputs["paired_advantage"].mean().detach()
            ),
            "mean_target_minus_two_control_mean": float(
                outputs["paired_advantage_two_control"].mean().detach()
            ),
            "specificity_violation_fraction": float(
                outputs["paired_violation"].float().mean().detach()
            ),
            "common_gradient_norm": common_norm,
            "current_target_gradient_norm": current_norm,
            "current_target_branch_gradient_norm": target_branch_norm,
            "current_target_minus_random_gradient_norm": specific_norm,
            "paired_advantage_gradient_norm": advantage_norm,
            "current_target_branch_to_total_norm_ratio": (
                target_branch_norm / current_norm if current_norm > 0 else float("nan")
            ),
            "current_specific_to_current_norm_ratio": (
                specific_norm / current_norm if current_norm > 0 else float("nan")
            ),
            "paired_advantage_to_current_norm_ratio": (
                advantage_norm / current_norm if current_norm > 0 else float("nan")
            ),
            "current_target_branch_clean_alignment": target_branch_alignment,
            "current_specific_clean_alignment": specific_alignment,
            "paired_advantage_clean_alignment": grad_cosine(common, advantage),
            "paired_advantage_clean_alignment_head": grad_cosine(
                common, advantage, head_mask,
            ),
            "paired_advantage_clean_alignment_backbone": grad_cosine(
                common, advantage, backbone_mask,
            ),
            "current_target_clip_scale_at_1": min(
                1.0, args.grad_clip / current_norm
            ) if current_norm > 0 else 1.0,
        }
        if not all(
            np.isfinite(value) for key, value in record.items()
            if isinstance(value, (float, np.floating))
        ):
            raise RuntimeError(f"non-finite gradient diagnostic for formula {formula}")
        records.append(record)
        del losses, outputs, common, advantage
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[E4-B0 {label}] first pass {index + 1}/{len(ordered)}", flush=True)
    if accumulated is None:
        raise RuntimeError("gradient accumulator was never initialized")
    total_mean = [
        value.to(device) if value is not None else None for value in accumulated
    ]
    consensus: dict[str, float] = {}
    for index, formula in enumerate(ordered):
        losses, _ = loss_bundle(
            model, store, examples_by_formula[formula], official_by_row, device, args,
        )
        advantage = gradients(losses["paired_advantage"], parameters, False)
        leave_one_out = [
            (mean - value.float()) if mean is not None and value is not None else None
            for mean, value in zip(total_mean, advantage)
        ]
        consensus[formula] = grad_cosine(advantage, leave_one_out)
        del losses, advantage, leave_one_out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[E4-B0 {label}] consensus {index + 1}/{len(ordered)}", flush=True)
    frame = pd.DataFrame(records)
    frame["paired_advantage_gradient_leave_one_formula_out_cosine"] = frame["formula"].map(
        consensus
    )
    report = {
        "checkpoint": label,
        "provenance": provenance,
        "formulas": int(len(frame)),
        "actions": int(frame["actions"].sum()),
        "identities": int(len({
            item.identity for values in examples_by_formula.values() for item in values
        })),
        "official_errors": int(frame["official_errors"].sum()),
        "margin_advantage_selected_random_formula_ci": bootstrap_mean(
            frame["mean_target_minus_selected_random"].to_numpy(),
            args.bootstrap_resamples, args.seed + 10,
        ),
        "margin_advantage_two_control_formula_ci": bootstrap_mean(
            frame["mean_target_minus_two_control_mean"].to_numpy(),
            args.bootstrap_resamples, args.seed + 11,
        ),
        "current_specific_norm_ratio_formula_ci": bootstrap_mean(
            frame["current_specific_to_current_norm_ratio"].to_numpy(),
            args.bootstrap_resamples, args.seed + 12,
        ),
        "current_target_branch_norm_ratio_formula_ci": bootstrap_mean(
            frame["current_target_branch_to_total_norm_ratio"].to_numpy(),
            args.bootstrap_resamples, args.seed + 17,
        ),
        "current_target_branch_clean_alignment_formula_ci": bootstrap_mean(
            frame["current_target_branch_clean_alignment"].to_numpy(),
            args.bootstrap_resamples, args.seed + 18,
        ),
        "paired_advantage_norm_ratio_formula_ci": bootstrap_mean(
            frame["paired_advantage_to_current_norm_ratio"].to_numpy(),
            args.bootstrap_resamples, args.seed + 13,
        ),
        "current_specific_clean_alignment_formula_ci": bootstrap_mean(
            frame["current_specific_clean_alignment"].to_numpy(),
            args.bootstrap_resamples, args.seed + 14,
        ),
        "paired_advantage_clean_alignment_formula_ci": bootstrap_mean(
            frame["paired_advantage_clean_alignment"].to_numpy(),
            args.bootstrap_resamples, args.seed + 15,
        ),
        "paired_advantage_gradient_consensus_formula_ci": bootstrap_mean(
            frame["paired_advantage_gradient_leave_one_formula_out_cosine"].to_numpy(),
            args.bootstrap_resamples, args.seed + 16,
        ),
        "formula_fraction_advantage_gradient_consensus_positive": float(np.mean(
            frame["paired_advantage_gradient_leave_one_formula_out_cosine"] > 0
        )),
        "formula_fraction_advantage_clean_alignment_positive": float(np.mean(
            frame["paired_advantage_clean_alignment"] > 0
        )),
        "formula_fraction_current_target_would_clip": float(np.mean(
            frame["current_target_clip_scale_at_1"] < 1.0
        )),
        "median_current_target_clip_scale": float(np.median(
            frame["current_target_clip_scale_at_1"]
        )),
    }
    del model, total_mean, accumulated
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, frame


def main() -> None:
    args = arguments()
    if args.outer_fold != 0 or args.panel_formulas != 32:
        raise ValueError("E4-B0 freezes fold=0 and a 32-formula/128-action panel")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal E4-B0 gradient attribution requires CUDA")
    required = [
        args.graph, args.r0_dir / "report.json", args.r0_dir / "training_actions.csv.gz",
        args.data, args.embedding_cache, args.official_checkpoint,
        args.architecture_checkpoint, args.clean_checkpoint,
        args.clean_checkpoint.parent / "decision.json", args.attribution_summary,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E4-B0 output: {args.output_dir}")
    seed_everything(args.seed)
    device = torch.device(args.device)
    attribution = json.loads(args.attribution_summary.read_text(encoding="utf-8"))
    if (
        attribution.get("status") != "noise_final_e4a_causal_attribution_complete"
        or not attribution.get("formal")
        or attribution.get("pass_to_action_learnability") is not False
        or attribution.get("contracts", {}).get("P2b") != "forbidden"
        or attribution.get("contracts", {}).get("P3_consumed") is not False
    ):
        raise RuntimeError("E4-A attribution failure contract is not the frozen input")
    expected_clean_sha = str(
        attribution.get("arms", {}).get("clean_duplicate", {}).get(
            "checkpoint_sha256", ""
        )
    )
    observed_clean_sha = sha256_file(args.clean_checkpoint)
    if not expected_clean_sha or observed_clean_sha != expected_clean_sha:
        raise RuntimeError(
            "clean-continuation checkpoint does not match the frozen E4-A summary"
        )
    r0_report = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    if (
        not r0_report.get("formal")
        or r0_report.get("contracts", {}).get("P2b") != "forbidden"
        or r0_report.get("contracts", {}).get(
            "action_outcomes_absent_from_training_manifest"
        ) is not True
        or int(r0_report.get("contracts", {}).get("matched_controls_preserved", -1)) != 2
    ):
        raise RuntimeError("R0 violates the outcome-free paired-control contract")
    graph = CandidateGraph(args.graph)
    official_rank, official_margin = official_rank_margin(graph)
    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    cells: list[pd.DataFrame] = []
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        cell = actions.loc[
            actions["selector"].astype(str).eq(selector)
            & np.isclose(actions["attenuation"].astype(float), attenuation)
            & actions["step"].astype(int).eq(step)
            & actions["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        if cell.empty:
            raise RuntimeError(f"missing R0 cell {selector}|{attenuation}|{step}")
        cells.append(cell)
    train_actions = pd.concat(cells, ignore_index=True)
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden.intersection(train_actions.columns):
        raise RuntimeError("outcome columns leaked into E4-B0 action panel")
    panel = select_formula_panel(train_actions, args.panel_formulas, args.seed)
    if len(panel[["selector", "attenuation", "step"]].drop_duplicates()) != 9:
        raise RuntimeError("E4-B0 deterministic panel does not cover all nine curriculum cells")
    examples = make_gradient_examples(
        graph, panel, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    )
    examples_by_formula = {
        formula: [item for item in examples if item.formula == formula]
        for formula in sorted({item.formula for item in examples})
    }
    if len(examples) != 128 or len(examples_by_formula) != 32:
        raise RuntimeError("E4-B0 panel does not contain 128 actions in 32 formulas")
    reachable_rows = sorted({
        row
        for item in examples
        for row in (item.query_row, *item.positive_rows, *item.negative_rows)
    })
    store = SpectrumStore(args.data, np.asarray(reachable_rows), args.n_highest_peaks)
    cache_rows, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    del cache_rows
    missing_embedding = [row for row in reachable_rows if row not in cache_index]
    if missing_embedding:
        raise RuntimeError(f"official embedding cache misses rows: {missing_embedding[:20]}")
    official_by_row = {
        int(row): cache_embeddings[cache_index[int(row)]] for row in reachable_rows
    }
    checkpoint_reports: dict[str, object] = {}
    frames: list[pd.DataFrame] = []
    for label in ("official_initialization", "clean_continuation"):
        checkpoint_report, frame = audit_checkpoint(
            label, args, store, examples_by_formula, official_by_row, device,
        )
        checkpoint_reports[label] = checkpoint_report
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    clean = checkpoint_reports["clean_continuation"]
    gates = {
        "panel_has_32_formulas": len(examples_by_formula) == 32,
        "panel_has_128_actions": len(examples) == 128,
        "panel_has_both_action_families": set(panel["selector"].astype(str))
        == {"candidate_gradient", "role_confounder"},
        "all_nine_curriculum_cells_present": len(
            panel[["selector", "attenuation", "step"]].drop_duplicates()
        ) == 9,
        "target_advantage_survives_clean_continuation": bool(
            clean["margin_advantage_selected_random_formula_ci"]["ci_low"] > 0
        ),
        "advantage_gradient_cross_formula_consensus_positive": bool(
            clean["paired_advantage_gradient_consensus_formula_ci"]["ci_low"] > 0
        ),
        "advantage_gradient_clean_alignment_positive": bool(
            clean["paired_advantage_clean_alignment_formula_ci"]["ci_low"] > 0
        ),
        "P2b_forbidden": True,
        "P3_not_consumed": True,
    }
    pass_to_pilot = bool(all(gates.values()))
    report = {
        "status": "noise_final_e4b0_gradient_attribution_complete",
        "formal": True,
        "panel": {
            "formulas": len(examples_by_formula),
            "actions": len(examples),
            "identities": len({item.identity for item in examples}),
            "official_errors": sum(item.official_rank != 1 for item in examples),
            "cell_counts": panel.groupby(
                ["selector", "attenuation", "step"]
            ).size().rename("actions").reset_index().to_dict("records"),
        },
        "checkpoints": checkpoint_reports,
        "gates": gates,
        "pass_to_paired_advantage_pilot": pass_to_pilot,
        "decision": (
            "authorize one small paired counterfactual-advantage pilot"
            if pass_to_pilot else
            "do not train; localize the failed gradient availability/consensus/alignment gate"
        ),
        "contracts": {
            "optimizer_steps": 0,
            "weights_changed": False,
            "same_query_target_and_two_frozen_controls": True,
            "one_formula_per_gradient_microbatch": True,
            "leave_one_formula_out_gradient_consensus": True,
            "outcomes_used_for_panel_selection": False,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "attribution_summary_sha256": sha256_file(args.attribution_summary),
            "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
            "r0_actions_sha256": sha256_file(
                args.r0_dir / "training_actions.csv.gz"
            ),
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "clean_checkpoint_sha256": observed_clean_sha,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Mechanistic gradient audit on a fixed outer-train panel. It contains no "
            "optimizer step, held-fold model selection, new embedding gain, or P3 result."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    panel.to_csv(temporary / "panel.csv.gz", index=False, compression="gzip")
    combined.to_csv(
        temporary / "per_formula_gradient.csv.gz", index=False, compression="gzip"
    )
    report["artifacts"] = {
        "panel_sha256": sha256_file(temporary / "panel.csv.gz"),
        "per_formula_gradient_sha256": sha256_file(
            temporary / "per_formula_gradient.csv.gz"
        ),
    }
    json_dump(temporary / "report.json", report)
    temporary.rename(args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

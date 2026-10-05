"""Zero-update audit of the frozen Noise native multi-difficulty Stage-3 bank.

The audit never calls an optimizer.  It measures independent chemical coverage,
target/control geometry for every registered action triplet, and exact
pre-optimizer gradient directions on deterministic query-stratified probes.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from build_noise_dreams_native_residual_stage2 import sha256_file
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from train_noise_dreams_native import load_npz, native_dataset
from train_noise_dreams_native_residual_stage2 import construct_native_model


AUDIT_VERSION = "noise_native_stage3_zero_update_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--gradient-events-per-stratum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def distribution(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(float)
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def coverage_audit(selected: pd.DataFrame, pool: dict[str, np.ndarray]) -> dict:
    required = {
        "query_index", "query_ik14", "query_formula", "source", "family",
        "difficulty_tier", "action_positive_row", "action_negative_row",
        "stage1_action_index", "action_id", "positive_pool_size",
    }
    if missing := required - set(selected):
        raise RuntimeError(f"Stage-3 selected actions lack {sorted(missing)}")
    if len(selected) == 0:
        raise RuntimeError("Stage-3 selected action table is empty")
    action_mask = np.asarray(pool["event_kind"], dtype=np.int8) == 2
    clean_mask = np.asarray(pool["event_kind"], dtype=np.int8) == 0
    action_ids = np.asarray(pool["event_action_index"], dtype=np.int64)[action_mask]
    if not np.array_equal(action_ids, np.arange(len(selected), dtype=np.int64)):
        raise RuntimeError("Stage-3 action events no longer align to selected rows")

    boundary_columns = [
        "query_index", "action_positive_row", "action_negative_row",
    ]
    query_counts = selected.groupby("query_index", sort=False).size()
    identity_counts = selected.groupby("query_ik14", sort=False).size()
    formula_counts = selected.groupby("query_formula", sort=False).size()
    boundary_counts = selected.groupby(boundary_columns, sort=False).size()
    negative_counts = selected.groupby("action_negative_row", sort=False).size()
    clean_queries = set(map(
        int, np.asarray(pool["event_query"], dtype=np.int64)[clean_mask],
    ))
    action_queries = set(map(int, selected["query_index"]))

    source_tier = pd.crosstab(
        selected["source"].astype(str), selected["difficulty_tier"].astype(str),
    )
    source_tier = {
        str(source): {str(tier): int(value) for tier, value in row.items()}
        for source, row in source_tier.to_dict(orient="index").items()
    }
    top_formula_rows = int(formula_counts.sort_values(ascending=False).head(10).sum())
    top_identity_rows = int(identity_counts.sort_values(ascending=False).head(10).sum())
    return {
        "action_rows": int(len(selected)),
        "unique_queries": int(selected["query_index"].nunique()),
        "unique_identities": int(selected["query_ik14"].nunique()),
        "unique_formulas": int(selected["query_formula"].nunique()),
        "unique_positive_rows": int(selected["action_positive_row"].nunique()),
        "unique_negative_rows": int(selected["action_negative_row"].nunique()),
        "positive_pool_size": distribution(selected["positive_pool_size"]),
        "rows_with_at_least_two_measured_positives": int(
            (selected["positive_pool_size"].astype(int) >= 2).sum()
        ),
        "rows_with_at_least_three_measured_positives": int(
            (selected["positive_pool_size"].astype(int) >= 3).sum()
        ),
        "fraction_with_at_least_two_measured_positives": float(
            (selected["positive_pool_size"].astype(int) >= 2).mean()
        ),
        "fraction_with_at_least_three_measured_positives": float(
            (selected["positive_pool_size"].astype(int) >= 3).mean()
        ),
        "unique_query_positive_negative_boundaries": int(
            selected[boundary_columns].drop_duplicates().shape[0]
        ),
        "duplicate_boundary_rows": int(len(selected) - len(boundary_counts)),
        "clean_event_queries": int(len(clean_queries)),
        "selected_action_queries": int(len(action_queries)),
        "extra_clean_only_queries": int(len(clean_queries - action_queries)),
        "all_action_queries_have_clean_event": action_queries <= clean_queries,
        "rows_per_query": distribution(query_counts),
        "rows_per_identity": distribution(identity_counts),
        "rows_per_formula": distribution(formula_counts),
        "rows_per_unique_boundary": distribution(boundary_counts),
        "rows_per_negative_row": distribution(negative_counts),
        "top_10_formula_row_fraction": float(top_formula_rows / len(selected)),
        "top_10_identity_row_fraction": float(top_identity_rows / len(selected)),
        "source_by_tier": source_tier,
    }


def make_preprocessor() -> SpectrumPreprocessor:
    return SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )


def make_model(args: argparse.Namespace) -> tuple[torch.nn.Module, str]:
    namespace = SimpleNamespace(
        warm_start_checkpoint=args.warm_start_checkpoint,
        architecture_checkpoint=args.architecture_checkpoint,
        n_highest_peaks=100,
        lr=5e-6,
        weight_decay=0.0,
        triplet_loss_margin=0.1,
    )
    model, kind = construct_native_model(namespace)
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, kind


def action_geometry(
    model: torch.nn.Module,
    dataset,
    event_dataset_indices: np.ndarray,
    action_positions: np.ndarray,
    *,
    batch_size: int,
) -> pd.DataFrame:
    indices = event_dataset_indices[action_positions]
    loader = DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            anchor = model(batch["spec"].cuda(non_blocking=True))
            pos = model(batch["pos_specs"][:, 0].cuda(non_blocking=True))
            neg = model(batch["neg_specs"][:, 0].cuda(non_blocking=True))
            positive.append(F.cosine_similarity(anchor, pos, dim=-1).cpu().numpy())
            negative.append(F.cosine_similarity(anchor, neg, dim=-1).cpu().numpy())
    p = np.concatenate(positive)
    n = np.concatenate(negative)
    if len(p) != len(action_positions):
        raise RuntimeError("Stage-3 geometry scoring lost action rows")
    return pd.DataFrame({
        "positive_similarity": p,
        "negative_similarity": n,
        "margin": p - n,
        "hinge_loss": np.maximum(0.1 - p + n, 0.0),
        "hinge_active": (p - n) < 0.1,
    })


def geometry_summary(frame: pd.DataFrame) -> dict:
    return {
        "events": int(len(frame)),
        "positive_similarity": distribution(frame["positive_similarity"]),
        "negative_similarity": distribution(frame["negative_similarity"]),
        "margin": distribution(frame["margin"]),
        "hinge_loss": distribution(frame["hinge_loss"]),
        "hinge_active_fraction": float(frame["hinge_active"].mean()),
    }


def materialize_batch(dataset, event_dataset_indices: np.ndarray, positions: list[int]):
    return default_collate([
        dataset[int(event_dataset_indices[int(position)])] for position in positions
    ])


def objective(
    model: torch.nn.Module, batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    anchor = model(batch["spec"].cuda())
    positive = model(batch["pos_specs"][:, 0].cuda())
    negative = model(batch["neg_specs"][:, 0].cuda())
    s_pos = F.cosine_similarity(anchor, positive, dim=-1)
    s_neg = F.cosine_similarity(anchor, negative, dim=-1)
    loss = torch.clamp_min(0.1 - s_pos + s_neg, 0.0).mean()
    role_gradients = torch.autograd.grad(
        loss, (anchor, positive, negative), retain_graph=True,
    )
    role_norms = {
        role: float(gradient.detach().norm().cpu())
        for role, gradient in zip(
            ("anchor", "positive", "negative"), role_gradients, strict=True,
        )
    }
    return loss, {
        "loss": float(loss.detach().cpu()),
        "positive_similarity_mean": float(s_pos.detach().mean().cpu()),
        "negative_similarity_mean": float(s_neg.detach().mean().cpu()),
        "margin_mean": float((s_pos - s_neg).detach().mean().cpu()),
        "active_fraction": float(((s_pos - s_neg) < 0.1).float().mean().cpu()),
        "role_gradient_norms": role_norms,
    }


def parameter_groups(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> dict[str, list[int]]:
    groups = {"head": [], "backbone": [], "all": []}
    for index, (name, _) in enumerate(named_parameters):
        groups["all"].append(index)
        groups["head" if name.startswith("head.") else "backbone"].append(index)
    return groups


def capture_gradients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    model.zero_grad(set_to_none=True)
    loss, observation = objective(model, batch)
    loss.backward()
    gradients = [
        torch.zeros_like(parameter, device="cpu")
        if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for parameter in model.parameters()
    ]
    model.zero_grad(set_to_none=True)
    return gradients, observation


def compare_gradients(
    left: list[torch.Tensor],
    right: list[torch.Tensor],
    groups: dict[str, list[int]],
) -> dict[str, dict[str, float]]:
    if len(left) != len(right):
        raise RuntimeError("gradient vectors have different parameter structure")
    report: dict[str, dict[str, float]] = {}
    for label, indices in groups.items():
        dot = sum(float(torch.sum(left[i] * right[i])) for i in indices)
        left_sq = sum(float(torch.sum(left[i] * left[i])) for i in indices)
        right_sq = sum(float(torch.sum(right[i] * right[i])) for i in indices)
        denom = max((left_sq * right_sq) ** 0.5, 1e-30)
        report[label] = {
            "left_norm": float(left_sq ** 0.5),
            "right_norm": float(right_sq ** 0.5),
            "cosine": float(dot / denom),
            "difference_norm": float(max(left_sq + right_sq - 2 * dot, 0.0) ** 0.5),
        }
    return report


def select_probe_actions(
    selected: pd.DataFrame, mask: pd.Series, maximum: int, seed: int,
) -> list[int]:
    eligible = selected.loc[mask].sort_values(
        ["query_index", "difficulty_tier", "source", "action_id"], kind="stable",
    )
    eligible = eligible.drop_duplicates("query_index", keep="first")
    if eligible.empty:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))[:maximum]
    return list(map(int, eligible.iloc[order].index))


def gradient_audit(
    model: torch.nn.Module,
    selected: pd.DataFrame,
    pool: dict[str, np.ndarray],
    event_dataset_indices_targeted: np.ndarray,
    targeted_dataset,
    event_dataset_indices_control: np.ndarray,
    control_dataset,
    *,
    maximum: int,
    seed: int,
) -> dict:
    event_kind = np.asarray(pool["event_kind"], dtype=np.int8)
    event_query = np.asarray(pool["event_query"], dtype=np.int64)
    event_action = np.asarray(pool["event_action_index"], dtype=np.int64)
    action_positions = np.flatnonzero(event_kind == 2)
    clean_positions = np.flatnonzero(event_kind == 0)
    action_position_by_id = {
        int(event_action[position]): int(position) for position in action_positions
    }
    clean_position_by_query: dict[int, int] = {}
    for position in clean_positions:
        clean_position_by_query.setdefault(int(event_query[position]), int(position))

    strata: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=selected.index))]
    strata.extend(
        (f"tier:{tier}", selected["difficulty_tier"].astype(str) == str(tier))
        for tier in sorted(selected["difficulty_tier"].astype(str).unique())
    )
    strata.extend(
        (f"source:{source}", selected["source"].astype(str) == str(source))
        for source in sorted(selected["source"].astype(str).unique())
    )
    named = list(model.named_parameters())
    groups = parameter_groups(named)
    output: dict[str, object] = {}
    for offset, (label, mask) in enumerate(strata):
        action_ids = select_probe_actions(selected, mask, maximum, seed + offset)
        if not action_ids:
            continue
        action_event_positions = [action_position_by_id[action] for action in action_ids]
        queries = list(map(int, selected.loc[action_ids, "query_index"]))
        clean_event_positions = [clean_position_by_query[query] for query in queries]

        random.seed(seed + 1000 + offset)
        targeted_batch = materialize_batch(
            targeted_dataset, event_dataset_indices_targeted, action_event_positions,
        )
        random.seed(seed + 1000 + offset)
        control_batch = materialize_batch(
            control_dataset, event_dataset_indices_control, action_event_positions,
        )
        random.seed(seed + 2000 + offset)
        clean_batch = materialize_batch(
            targeted_dataset, event_dataset_indices_targeted, clean_event_positions,
        )

        targeted_gradient, targeted_observation = capture_gradients(
            model, targeted_batch,
        )
        control_gradient, control_observation = capture_gradients(model, control_batch)
        targeted_control = compare_gradients(targeted_gradient, control_gradient, groups)
        del control_gradient
        clean_gradient, clean_observation = capture_gradients(model, clean_batch)
        targeted_clean = compare_gradients(targeted_gradient, clean_gradient, groups)
        del targeted_gradient, clean_gradient
        output[label] = {
            "events": int(len(action_ids)),
            "action_ids": action_ids,
            "queries": queries,
            "targeted": targeted_observation,
            "control": control_observation,
            "clean": clean_observation,
            "targeted_vs_control_parameter_gradient": targeted_control,
            "targeted_vs_clean_parameter_gradient": targeted_clean,
        }
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-3 zero-update gradient audit requires an allocated GPU")
    if args.score_batch_size < 1 or not 1 <= args.gradient_events_per_stratum <= 16:
        raise RuntimeError("invalid bounded Stage-3 audit batch configuration")
    required = (
        args.triplet_dir / "report.json",
        args.triplet_dir / "train_pool.npz",
        args.triplet_dir / "action_spectra.npz",
        args.triplet_dir / "selected_actions.csv.gz",
        args.warm_start_checkpoint,
        args.architecture_checkpoint,
        args.data,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    triplet_report = json.loads(
        (args.triplet_dir / "report.json").read_text(encoding="utf-8")
    )
    if triplet_report.get("status") != "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_COMPLETE":
        raise RuntimeError("zero-update audit received a non-Stage-3 triplet bank")
    selected = pd.read_csv(args.triplet_dir / "selected_actions.csv.gz", low_memory=False)
    pool = load_npz(args.triplet_dir / "train_pool.npz")
    bank = load_npz(args.triplet_dir / "action_spectra.npz")
    targeted_actions = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_actions = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    if len(selected) != len(targeted_actions) or targeted_actions.shape != control_actions.shape:
        raise RuntimeError("Stage-3 zero-update action artifacts are misaligned")

    coverage = coverage_audit(selected, pool)
    model, initialization_kind = make_model(args)
    preprocessor = make_preprocessor()
    targeted_dataset, targeted_indices, targeted_materialization = native_dataset(
        pool, args.data, targeted_actions, preprocessor,
    )
    control_dataset, control_indices, control_materialization = native_dataset(
        pool, args.data, control_actions, preprocessor,
    )
    action_positions = np.flatnonzero(np.asarray(pool["event_kind"], dtype=np.int8) == 2)
    targeted_geometry = action_geometry(
        model, targeted_dataset, targeted_indices, action_positions,
        batch_size=args.score_batch_size,
    )
    control_geometry = action_geometry(
        model, control_dataset, control_indices, action_positions,
        batch_size=args.score_batch_size,
    )
    geometry = selected.copy()
    for prefix, frame in (("targeted", targeted_geometry), ("control", control_geometry)):
        for column in frame:
            geometry[f"{prefix}_{column}"] = frame[column].to_numpy()
    geometry["targeted_minus_control_margin"] = (
        geometry["targeted_margin"] - geometry["control_margin"]
    )

    by_tier = {
        str(label): {
            "targeted": geometry_summary(group[[
                "targeted_positive_similarity", "targeted_negative_similarity",
                "targeted_margin", "targeted_hinge_loss", "targeted_hinge_active",
            ]].rename(columns=lambda value: value.removeprefix("targeted_"))),
            "control": geometry_summary(group[[
                "control_positive_similarity", "control_negative_similarity",
                "control_margin", "control_hinge_loss", "control_hinge_active",
            ]].rename(columns=lambda value: value.removeprefix("control_"))),
            "targeted_minus_control_margin": distribution(
                group["targeted_minus_control_margin"]
            ),
            "targeted_margin_better_fraction": float(
                (group["targeted_minus_control_margin"] > 0).mean()
            ),
        }
        for label, group in geometry.groupby("difficulty_tier", sort=True)
    }
    by_source = {
        str(label): {
            "events": int(len(group)),
            "targeted_minus_control_margin": distribution(
                group["targeted_minus_control_margin"]
            ),
            "targeted_margin_better_fraction": float(
                (group["targeted_minus_control_margin"] > 0).mean()
            ),
        }
        for label, group in geometry.groupby("source", sort=True)
    }
    gradients = gradient_audit(
        model, selected, pool, targeted_indices, targeted_dataset,
        control_indices, control_dataset,
        maximum=args.gradient_events_per_stratum, seed=args.seed,
    )
    del model

    gates = {
        "no_optimizer_constructed_or_stepped": True,
        "all_action_queries_have_clean_event": bool(
            coverage["all_action_queries_have_clean_event"]
        ),
        "all_action_rows_scored_in_both_arms": bool(
            len(geometry) == len(selected)
            and geometry.filter(like="targeted_").notna().all().all()
            and geometry.filter(like="control_").notna().all().all()
        ),
        "targeted_and_control_geometry_are_distinct": bool(
            np.any(np.abs(geometry["targeted_minus_control_margin"].to_numpy()) > 1e-8)
        ),
        "every_registered_tier_and_source_has_gradient_probe": bool(
            {f"tier:{value}" for value in selected["difficulty_tier"].astype(str).unique()}
            | {f"source:{value}" for value in selected["source"].astype(str).unique()}
            <= set(gradients)
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-3 zero-update audit gates failed: {gates}")

    report = {
        "status": "NOISE_DREAMS_NATIVE_STAGE3_ZERO_UPDATE_AUDIT_COMPLETE",
        "audit_version": AUDIT_VERSION,
        "weights_updated": False,
        "initialization_kind": initialization_kind,
        "coverage": coverage,
        "geometry": {
            "all": {
                "targeted": geometry_summary(targeted_geometry),
                "control": geometry_summary(control_geometry),
                "targeted_minus_control_margin": distribution(
                    geometry["targeted_minus_control_margin"]
                ),
                "targeted_margin_better_fraction": float(
                    (geometry["targeted_minus_control_margin"] > 0).mean()
                ),
            },
            "by_tier": by_tier,
            "by_source": by_source,
        },
        "preoptimizer_gradients": gradients,
        "materialization": {
            "targeted": targeted_materialization,
            "control": control_materialization,
        },
        "gates": gates,
        "provenance": {
            "triplet_report_sha256": sha256_file(args.triplet_dir / "report.json"),
            "selected_actions_sha256": sha256_file(
                args.triplet_dir / "selected_actions.csv.gz"
            ),
            "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
        },
        "claim_limit": (
            "Frozen pre-optimizer coverage, geometry and gradient audit only; "
            "no held performance or improved checkpoint is claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_stage3_zero_", dir=args.output.parent))
    try:
        geometry.to_csv(
            staging / "action_geometry.csv.gz", index=False, compression="gzip",
        )
        report["artifacts"] = {
            "action_geometry_sha256": sha256_file(staging / "action_geometry.csv.gz"),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, default=scalar), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=scalar), flush=True)


if __name__ == "__main__":
    main()

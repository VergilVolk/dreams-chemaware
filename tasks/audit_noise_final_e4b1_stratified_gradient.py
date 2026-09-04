"""E4-B1: stratify the E4-B0 target/random gradient failure without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_e4b0_gradient_attribution import (  # noqa: E402
    grad_cosine, grad_dot, grad_norm, gradients, load_checkpoint_model,
    loss_bundle, make_gradient_examples,
)
from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything, sha256_file,
)
from train_noise_final_e4a_direct_augmentation import (  # noqa: E402
    FIXED_POLICY, SpectrumStore, official_rank_margin,
)


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
        "--error-analysis", type=Path,
        default=ROOT / "data/validation/g8r_real_error_analysis",
    )
    parser.add_argument(
        "--e4b0-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4b0_gradient_attribution",
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
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4b1_stratified_gradient",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formulas-per-primary-stratum", type=int, default=32)
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--specificity-margin", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stable_key(seed: int, *values: object) -> str:
    text = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strict_bool(series: pd.Series, label: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    normalized = series.astype(str).str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{label} is not a strict boolean column")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def load_signature(args: argparse.Namespace, graph: CandidateGraph,
                   official_rank: np.ndarray) -> pd.DataFrame:
    report_path = args.error_analysis / "report.json"
    signature_path = args.error_analysis / "query_error_signatures.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "g8r_real_error_analysis_complete":
        raise RuntimeError("frozen real-error analysis is incomplete")
    columns = [
        "query_index", "dreams_correct", "has_near_candidate", "score_error_family",
    ]
    signature = pd.read_csv(signature_path, usecols=columns, low_memory=False)
    if (
        len(signature) != graph.n_queries
        or signature["query_index"].duplicated().any()
        or set(signature["query_index"].astype(int)) != set(range(graph.n_queries))
    ):
        raise RuntimeError("error signature is not one-to-one with the locked graph")
    signature["dreams_correct"] = strict_bool(
        signature["dreams_correct"], "dreams_correct",
    )
    signature["has_near_candidate"] = strict_bool(
        signature["has_near_candidate"], "has_near_candidate",
    )
    ordered = signature.sort_values("query_index", kind="stable")
    if not np.array_equal(
        ordered["dreams_correct"].to_numpy(bool), official_rank == 1,
    ):
        raise RuntimeError("error signature does not reproduce official DreaMS rank")
    return signature


def select_balanced_panel(actions: pd.DataFrame, signature: pd.DataFrame,
                          args: argparse.Namespace) -> pd.DataFrame:
    merged = actions.merge(signature, on="query_index", how="inner", validate="many_to_one")
    if len(merged) != len(actions):
        raise RuntimeError("error signature failed to cover every action")
    merged["baseline_state"] = np.where(
        merged["dreams_correct"].to_numpy(bool), "official_correct", "official_error",
    )
    merged["near_state"] = np.where(
        merged["has_near_candidate"].to_numpy(bool), "near", "non_near",
    )
    merged["cell_id"] = [
        f"{selector}|a={float(attenuation):.2f}|step={int(step)}"
        for selector, attenuation, step in merged[
            ["selector", "attenuation", "step"]
        ].itertuples(index=False, name=None)
    ]
    blocks: list[pd.DataFrame] = []
    expected_strata: list[str] = []
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        cell_id = f"{selector}|a={float(attenuation):.2f}|step={int(step)}"
        for state in ("official_error", "official_correct"):
            stratum_id = f"{cell_id}|{state}"
            expected_strata.append(stratum_id)
            block = merged.loc[
                merged["cell_id"].eq(cell_id) & merged["baseline_state"].eq(state)
            ].copy()
            block["_query_order"] = [
                stable_key(args.seed, stratum_id, formula, query)
                for formula, query in block[["query_formula", "query_index"]]
                .itertuples(index=False, name=None)
            ]
            one_per_formula = (
                block.sort_values("_query_order", kind="stable")
                .drop_duplicates("query_formula", keep="first")
            )
            formulas = sorted(
                one_per_formula["query_formula"].astype(str),
                key=lambda value: stable_key(args.seed + 1, stratum_id, value),
            )
            if len(formulas) < args.formulas_per_primary_stratum:
                raise RuntimeError(
                    f"{stratum_id} has {len(formulas)} formulas; "
                    f"need {args.formulas_per_primary_stratum}"
                )
            chosen = set(formulas[:args.formulas_per_primary_stratum])
            selected = one_per_formula.loc[
                one_per_formula["query_formula"].astype(str).isin(chosen)
            ].copy()
            selected["primary_stratum"] = stratum_id
            blocks.append(selected.drop(columns="_query_order"))
    panel = pd.concat(blocks, ignore_index=True)
    counts = panel.groupby("primary_stratum")["query_formula"].nunique()
    if (
        set(counts.index) != set(expected_strata)
        or counts.ne(args.formulas_per_primary_stratum).any()
        or panel.duplicated(["primary_stratum", "query_formula"]).any()
    ):
        raise RuntimeError("balanced E4-B1 panel lost its cell/state/formula structure")
    return panel.sort_values(
        ["primary_stratum", "query_formula", "query_index"], kind="stable",
    ).reset_index(drop=True)


def gradient_sum_add(destination: list[torch.Tensor | None],
                     source: list[torch.Tensor | None]) -> None:
    for left, right in zip(destination, source):
        if left is not None and right is not None:
            left.add_(right.float().cpu())


def loo_cosine(value: list[torch.Tensor | None], total: list[torch.Tensor | None],
               mask: np.ndarray | None = None) -> float:
    value_norm_sq = grad_dot(value, value, mask)
    total_norm_sq = grad_dot(total, total, mask)
    value_total = grad_dot(value, total, mask)
    remainder_norm_sq = max(total_norm_sq + value_norm_sq - 2.0 * value_total, 0.0)
    denominator = math.sqrt(max(value_norm_sq, 0.0) * remainder_norm_sq)
    return (value_total - value_norm_sq) / denominator if denominator > 0 else float("nan")


def bootstrap(values: np.ndarray, resamples: int, seed: int,
              adjusted_alpha: float | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid formula values for bootstrap")
    rng = np.random.default_rng(seed)
    sampled = values[
        rng.integers(0, len(values), size=(resamples, len(values)))
    ].mean(axis=1)
    output = {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }
    if adjusted_alpha is not None:
        output["multiplicity_adjusted_lower"] = float(
            np.quantile(sampled, adjusted_alpha)
        )
    return output


def summarize_group(frame: pd.DataFrame, group_type: str, group_id: str,
                    resamples: int, seed: int,
                    adjusted_alpha: float | None = None) -> dict[str, object]:
    # Several action cells can contribute to a descriptive group.  Collapse to
    # one value per formula before resampling so formula remains the unit.
    metrics = [
        "target_minus_selected_random", "paired_advantage_clean_alignment",
        "paired_advantage_clean_alignment_head",
        "paired_advantage_clean_alignment_backbone",
        "primary_gradient_consensus", "primary_gradient_consensus_head",
        "primary_gradient_consensus_backbone",
        "current_specific_clean_alignment", "current_specific_norm_ratio",
        "paired_advantage_norm_ratio", "current_target_clip_scale",
    ]
    formula = frame.groupby("query_formula", sort=False)[metrics].mean()
    row: dict[str, object] = {
        "group_type": group_type,
        "group_id": group_id,
        "actions": int(len(frame)),
        "formulas": int(len(formula)),
        "identities": int(frame["identity"].nunique()),
        "official_errors": int(frame["baseline_state"].eq("official_error").sum()),
    }
    for position, metric in enumerate(metrics):
        interval = bootstrap(
            formula[metric].to_numpy(), resamples, seed + position,
            adjusted_alpha if metric in {
                "target_minus_selected_random", "paired_advantage_clean_alignment",
                "primary_gradient_consensus",
            } else None,
        )
        row[f"{metric}_mean"] = interval["mean"]
        row[f"{metric}_ci_low"] = interval["ci_low"]
        row[f"{metric}_ci_high"] = interval["ci_high"]
        if "multiplicity_adjusted_lower" in interval:
            row[f"{metric}_adjusted_low"] = interval[
                "multiplicity_adjusted_lower"
            ]
    return row


def main() -> None:
    args = arguments()
    if (
        args.outer_fold != 0
        or args.formulas_per_primary_stratum != 32
        or args.bootstrap_resamples != 50000
    ):
        raise ValueError("E4-B1 freezes fold=0, 32 formulas/stratum and 50k bootstraps")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal E4-B1 requires CUDA")
    required = [
        args.graph, args.r0_dir / "report.json",
        args.r0_dir / "training_actions.csv.gz",
        args.error_analysis / "report.json",
        args.error_analysis / "query_error_signatures.csv.gz",
        args.e4b0_dir / "report.json", args.data, args.embedding_cache,
        args.official_checkpoint, args.architecture_checkpoint, args.clean_checkpoint,
        args.clean_checkpoint.parent / "decision.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E4-B1 output: {args.output_dir}")
    seed_everything(args.seed)
    device = torch.device(args.device)
    e4b0 = json.loads((args.e4b0_dir / "report.json").read_text(encoding="utf-8"))
    if (
        e4b0.get("status") != "noise_final_e4b0_gradient_attribution_complete"
        or not e4b0.get("formal")
        or e4b0.get("pass_to_paired_advantage_pilot") is not False
        or e4b0.get("gates", {}).get("target_advantage_survives_clean_continuation")
        is not True
        or e4b0.get("gates", {}).get(
            "advantage_gradient_cross_formula_consensus_positive"
        ) is not False
        or e4b0.get("gates", {}).get(
            "advantage_gradient_clean_alignment_positive"
        ) is not False
        or e4b0.get("contracts", {}).get("optimizer_steps") != 0
    ):
        raise RuntimeError("E4-B0 is not the frozen failed pooled-gradient input")
    clean_sha = sha256_file(args.clean_checkpoint)
    if clean_sha != e4b0.get("provenance", {}).get("clean_checkpoint_sha256"):
        raise RuntimeError("E4-B1 clean checkpoint differs from frozen E4-B0")
    r0_report = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    if (
        not r0_report.get("formal")
        or r0_report.get("contracts", {}).get("P2b") != "forbidden"
        or r0_report.get("contracts", {}).get(
            "action_outcomes_absent_from_training_manifest"
        ) is not True
    ):
        raise RuntimeError("R0 does not satisfy the outcome-free action contract")

    graph = CandidateGraph(args.graph)
    official_rank, official_margin = official_rank_margin(graph)
    signature = load_signature(args, graph, official_rank)
    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden.intersection(actions.columns):
        raise RuntimeError("action outcomes leaked into the E4-B1 manifest")
    selected_cells: list[pd.DataFrame] = []
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        block = actions.loc[
            actions["selector"].astype(str).eq(selector)
            & np.isclose(actions["attenuation"].astype(float), attenuation)
            & actions["step"].astype(int).eq(step)
            & actions["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        if block.empty:
            raise RuntimeError(f"missing R0 curriculum cell {selector}|{attenuation}|{step}")
        selected_cells.append(block)
    panel = select_balanced_panel(
        pd.concat(selected_cells, ignore_index=True), signature, args,
    )
    examples = make_gradient_examples(
        graph, panel, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    )
    if len(examples) != 18 * args.formulas_per_primary_stratum:
        raise RuntimeError("E4-B1 did not materialize the frozen 576 actions")
    metadata = panel.set_index(
        ["query_index", "selector", "step"], verify_integrity=True,
    )
    reachable_rows = sorted({
        row for example in examples
        for row in (example.query_row, *example.positive_rows, *example.negative_rows)
    })
    store = SpectrumStore(args.data, np.asarray(reachable_rows), args.n_highest_peaks)
    _, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    missing_embedding = [row for row in reachable_rows if row not in cache_index]
    if missing_embedding:
        raise RuntimeError(f"official embedding cache misses rows: {missing_embedding[:20]}")
    official_by_row = {
        int(row): cache_embeddings[cache_index[int(row)]] for row in reachable_rows
    }
    model, model_provenance, names, parameters = load_checkpoint_model(
        "clean_continuation", args, device,
    )
    head_mask = np.asarray([name.startswith("head.") for name in names], dtype=bool)
    backbone_mask = ~head_mask
    group_sums: dict[str, list[torch.Tensor | None]] = {}
    records: list[dict[str, object]] = []

    for position, example in enumerate(examples):
        row = metadata.loc[(example.query_index, example.selector, example.step)]
        stratum = str(row["primary_stratum"])
        losses, outputs = loss_bundle(
            model, store, [example], official_by_row, device, args,
        )
        common = gradients(losses["common"], parameters, True)
        target_branch = gradients(losses["current_target_branch"], parameters, True)
        common_norm = grad_norm(common)
        target_norm = grad_norm(target_branch)
        common_target_dot = grad_dot(common, target_branch)
        total_norm = math.sqrt(max(
            common_norm ** 2 + target_norm ** 2 + 2.0 * common_target_dot, 0.0,
        ))
        target_alignment = (
            common_target_dot / (common_norm * target_norm)
            if common_norm > 0 and target_norm > 0 else float("nan")
        )
        del target_branch
        specific = gradients(losses["current_target_minus_random"], parameters, True)
        specific_norm = grad_norm(specific)
        specific_alignment = grad_cosine(common, specific)
        specific_alignment_head = grad_cosine(common, specific, head_mask)
        specific_alignment_backbone = grad_cosine(common, specific, backbone_mask)
        del specific
        advantage = gradients(losses["paired_advantage"], parameters, False)
        advantage_norm = grad_norm(advantage)
        if stratum not in group_sums:
            group_sums[stratum] = [
                torch.zeros_like(value, device="cpu") if value is not None else None
                for value in advantage
            ]
        gradient_sum_add(group_sums[stratum], advantage)
        record = {
            "query_index": example.query_index,
            "query_row": example.query_row,
            "identity": example.identity,
            "query_formula": example.formula,
            "selector": example.selector,
            "attenuation": example.attenuation,
            "step": example.step,
            "cell_id": str(row["cell_id"]),
            "primary_stratum": stratum,
            "baseline_state": str(row["baseline_state"]),
            "score_error_family": str(row["score_error_family"]),
            "near_state": str(row["near_state"]),
            "official_rank": example.official_rank,
            "target_minus_selected_random": float(outputs[
                "paired_advantage"
            ][0].detach()),
            "target_minus_two_control_mean": float(outputs[
                "paired_advantage_two_control"
            ][0].detach()),
            "specificity_violation": bool(outputs["paired_violation"][0].detach()),
            "common_gradient_norm": common_norm,
            "current_target_total_gradient_norm": total_norm,
            "current_target_branch_gradient_norm": target_norm,
            "current_specific_gradient_norm": specific_norm,
            "paired_advantage_gradient_norm": advantage_norm,
            "current_target_branch_clean_alignment": target_alignment,
            "current_specific_clean_alignment": specific_alignment,
            "current_specific_clean_alignment_head": specific_alignment_head,
            "current_specific_clean_alignment_backbone": specific_alignment_backbone,
            "paired_advantage_clean_alignment": grad_cosine(common, advantage),
            "paired_advantage_clean_alignment_head": grad_cosine(
                common, advantage, head_mask,
            ),
            "paired_advantage_clean_alignment_backbone": grad_cosine(
                common, advantage, backbone_mask,
            ),
            "current_specific_norm_ratio": (
                specific_norm / total_norm if total_norm > 0 else float("nan")
            ),
            "paired_advantage_norm_ratio": (
                advantage_norm / total_norm if total_norm > 0 else float("nan")
            ),
            "current_target_clip_scale": (
                min(1.0, args.grad_clip / total_norm) if total_norm > 0 else 1.0
            ),
        }
        if not all(
            np.isfinite(value) for value in record.values()
            if isinstance(value, (float, np.floating))
        ):
            raise RuntimeError(f"non-finite first-pass diagnostic: {stratum}")
        records.append(record)
        del losses, outputs, common, advantage
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (position + 1) % 32 == 0 or position + 1 == len(examples):
            print(f"[E4-B1 first pass] {position + 1}/{len(examples)}", flush=True)

    consensus_full: list[float] = []
    consensus_head: list[float] = []
    consensus_backbone: list[float] = []
    for position, example in enumerate(examples):
        row = metadata.loc[(example.query_index, example.selector, example.step)]
        stratum = str(row["primary_stratum"])
        losses, _ = loss_bundle(model, store, [example], official_by_row, device, args)
        advantage = gradients(losses["paired_advantage"], parameters, False)
        advantage_cpu = [
            value.float().cpu() if value is not None else None for value in advantage
        ]
        consensus_full.append(loo_cosine(advantage_cpu, group_sums[stratum]))
        consensus_head.append(loo_cosine(
            advantage_cpu, group_sums[stratum], head_mask,
        ))
        consensus_backbone.append(loo_cosine(
            advantage_cpu, group_sums[stratum], backbone_mask,
        ))
        del losses, advantage, advantage_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (position + 1) % 32 == 0 or position + 1 == len(examples):
            print(f"[E4-B1 consensus] {position + 1}/{len(examples)}", flush=True)

    per_action = pd.DataFrame(records)
    per_action["primary_gradient_consensus"] = consensus_full
    per_action["primary_gradient_consensus_head"] = consensus_head
    per_action["primary_gradient_consensus_backbone"] = consensus_backbone
    numeric = per_action.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise RuntimeError("E4-B1 per-action ledger contains non-finite diagnostics")

    primary_count = 18
    primary_endpoints = 3
    adjusted_alpha = 0.05 / (primary_count * primary_endpoints)
    summaries: list[dict[str, object]] = []
    for index, (group_id, frame) in enumerate(
        per_action.groupby("primary_stratum", sort=True)
    ):
        summaries.append(summarize_group(
            frame, "primary_cell_state", str(group_id),
            args.bootstrap_resamples, args.seed + 1000 * index, adjusted_alpha,
        ))
    descriptive_axes = [
        ("selector_state", ["selector", "baseline_state"]),
        ("selector_error_family", ["selector", "score_error_family"]),
        ("selector_near_state", ["selector", "near_state", "baseline_state"]),
    ]
    offset = 100_000
    for axis, columns in descriptive_axes:
        for index, (key, frame) in enumerate(per_action.groupby(columns, sort=True)):
            key_values = key if isinstance(key, tuple) else (key,)
            group_id = "|".join(map(str, key_values))
            summaries.append(summarize_group(
                frame, axis, group_id, args.bootstrap_resamples,
                args.seed + offset + 1000 * index,
            ))
        offset += 100_000
    group_summary = pd.DataFrame(summaries)
    primary = group_summary.loc[
        group_summary["group_type"].eq("primary_cell_state")
    ].copy()
    primary["eligible_error_cell"] = (
        primary["group_id"].astype(str).str.endswith("|official_error")
        & primary["target_minus_selected_random_adjusted_low"].gt(0)
        & primary["primary_gradient_consensus_adjusted_low"].gt(0)
        & primary["paired_advantage_clean_alignment_adjusted_low"].gt(0)
    )
    eligibility = dict(zip(primary["group_id"], primary["eligible_error_cell"]))
    group_summary["eligible_error_cell"] = group_summary.apply(
        lambda row: bool(eligibility.get(row["group_id"], False))
        if row["group_type"] == "primary_cell_state" else False,
        axis=1,
    )
    eligible = sorted(primary.loc[primary["eligible_error_cell"], "group_id"].astype(str))
    primary_report_columns = [
        "group_id", "actions", "formulas", "identities", "official_errors",
        "target_minus_selected_random_mean",
        "target_minus_selected_random_ci_low",
        "target_minus_selected_random_ci_high",
        "target_minus_selected_random_adjusted_low",
        "primary_gradient_consensus_mean", "primary_gradient_consensus_ci_low",
        "primary_gradient_consensus_ci_high",
        "primary_gradient_consensus_adjusted_low",
        "paired_advantage_clean_alignment_mean",
        "paired_advantage_clean_alignment_ci_low",
        "paired_advantage_clean_alignment_ci_high",
        "paired_advantage_clean_alignment_adjusted_low",
        "paired_advantage_clean_alignment_head_mean",
        "paired_advantage_clean_alignment_head_ci_low",
        "paired_advantage_clean_alignment_backbone_mean",
        "paired_advantage_clean_alignment_backbone_ci_low",
        "primary_gradient_consensus_head_mean",
        "primary_gradient_consensus_head_ci_low",
        "primary_gradient_consensus_backbone_mean",
        "primary_gradient_consensus_backbone_ci_low",
        "current_specific_clean_alignment_mean",
        "current_specific_clean_alignment_ci_low",
        "eligible_error_cell",
    ]

    report = {
        "status": "noise_final_e4b1_stratified_gradient_complete",
        "formal": True,
        "panel": {
            "actions": int(len(panel)),
            "primary_strata": int(panel["primary_stratum"].nunique()),
            "formulas_per_primary_stratum": args.formulas_per_primary_stratum,
            "unique_formulas": int(panel["query_formula"].nunique()),
            "unique_identities": int(panel["query_ik14"].nunique()),
            "official_errors": int(panel["baseline_state"].eq("official_error").sum()),
            "official_correct": int(panel["baseline_state"].eq("official_correct").sum()),
        },
        "multiplicity": {
            "primary_groups": primary_count,
            "primary_endpoints": primary_endpoints,
            "one_sided_bonferroni_alpha": adjusted_alpha,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "eligible_error_cells": eligible,
        "eligible_error_cell_count": len(eligible),
        "primary_group_results": primary[primary_report_columns].to_dict("records"),
        "has_conditionally_coherent_error_cells": bool(eligible),
        "pass_to_training": False,
        "decision": (
            "design, but do not yet run, one small spectrum-conditional overfit audit"
            if eligible else
            "stop pooled mature-action transfer; no cell has multiplicity-controlled margin, consensus and clean alignment"
        ),
        "model": model_provenance,
        "gates": {
            "balanced_18x32_panel": bool(
                len(panel) == 576 and panel["primary_stratum"].nunique() == 18
            ),
            "all_nine_cells_in_both_states": bool(
                panel.groupby(["cell_id", "baseline_state"]).ngroups == 18
            ),
            "error_and_correct_counts_equal": bool(
                panel["baseline_state"].value_counts().nunique() == 1
            ),
            "all_primary_groups_reported": len(primary) == 18,
            "all_head_backbone_metrics_finite": bool(np.isfinite(
                per_action[[
                    "paired_advantage_clean_alignment_head",
                    "paired_advantage_clean_alignment_backbone",
                    "primary_gradient_consensus_head",
                    "primary_gradient_consensus_backbone",
                ]].to_numpy(np.float64)
            ).all()),
            "optimizer_steps_zero": True,
            "P2b_forbidden": True,
            "P3_not_consumed": True,
        },
        "contracts": {
            "action_outcomes_used_for_panel_selection": False,
            "official_baseline_state_used_for_preregistered_stratification": True,
            "same_query_target_and_two_frozen_controls": True,
            "one_query_per_formula_per_primary_stratum": True,
            "model_eval_mode": True,
            "optimizer_steps": 0,
            "weights_changed": False,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "e4b0_report_sha256": sha256_file(args.e4b0_dir / "report.json"),
            "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
            "r0_actions_sha256": sha256_file(
                args.r0_dir / "training_actions.csv.gz"
            ),
            "error_report_sha256": sha256_file(args.error_analysis / "report.json"),
            "error_signatures_sha256": sha256_file(
                args.error_analysis / "query_error_signatures.csv.gz"
            ),
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "clean_checkpoint_sha256": clean_sha,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Stratified no-update gradient diagnosis on outer-train actions. It is not "
            "a trained embedding, retrieval gain, deployable selector, or P3 result."
        ),
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"E4-B1 structural gates failed: {report['gates']}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    panel.to_csv(temporary / "panel.csv.gz", index=False, compression="gzip")
    per_action.to_csv(
        temporary / "per_action_gradient.csv.gz", index=False, compression="gzip",
    )
    group_summary.to_csv(
        temporary / "group_summary.csv.gz", index=False, compression="gzip",
    )
    report["artifacts"] = {
        "panel_sha256": sha256_file(temporary / "panel.csv.gz"),
        "per_action_gradient_sha256": sha256_file(
            temporary / "per_action_gradient.csv.gz"
        ),
        "group_summary_sha256": sha256_file(temporary / "group_summary.csv.gz"),
    }
    json_dump(temporary / "report.json", report)
    temporary.rename(args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

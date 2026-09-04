"""E4-B2: systematic zero-update gradient-surgery screen and confirmation."""
from __future__ import annotations

import argparse
import gc
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
    gradients, load_checkpoint_model, loss_bundle, make_gradient_examples,
)
from audit_noise_final_e4b1_stratified_gradient import (  # noqa: E402
    bootstrap, load_signature,
)
from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything, sha256_file,
)
from train_noise_final_e4a_direct_augmentation import (  # noqa: E402
    FIXED_POLICY, SpectrumStore, official_rank_margin,
)

METHODS = (
    ("raw", "raw", 0.0),
    ("pcgrad0", "pcgrad", 0.0),
    ("anchor_0.10", "anchor", 0.10),
    ("anchor_0.25", "anchor", 0.25),
    ("pcgrad_anchor_0.05", "pcgrad_anchor", 0.05),
    ("pcgrad_anchor_0.10", "pcgrad_anchor", 0.10),
)
SCOPES = ("joint", "head", "backbone")
METRICS = (
    "forward_margin", "clean_alignment", "gradient_consensus",
    "action_descent_retention", "action_fidelity", "norm_ratio",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--error-analysis", type=Path, default=ROOT / "data/validation/g8r_real_error_analysis")
    parser.add_argument("--e4b0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4b0_gradient_attribution")
    parser.add_argument("--e4b1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4b1_stratified_gradient")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=ROOT / (
            "data/validation/g8r_noise_final_e4a_causal_attribution/"
            "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
            "e4a_causal_v1_20260901_causal_clean_duplicate/"
            "seed_20260828/fold_0/final_shared_encoder.pt"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4b2_gradient_surgery_expanded",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--screen-formulas-per-stratum", type=int, default=32)
    parser.add_argument("--maximum-confirm-formulas-per-stratum", type=int, default=64)
    parser.add_argument("--maximum-confirm-candidates", type=int, default=3)
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--specificity-margin", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--screen-bootstraps", type=int, default=5000)
    parser.add_argument("--confirm-bootstraps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stable_key(seed: int, *values: object) -> str:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_discovery_confirmation_panel(
    actions: pd.DataFrame,
    signature: pd.DataFrame,
    b1_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Reuse B1 for discovery and reserve every unseen formula for confirmation."""
    merged = actions.merge(signature, on="query_index", how="inner", validate="many_to_one")
    if len(merged) != len(actions):
        raise RuntimeError("signature does not cover every R0 action")
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

    required_b1 = {
        "query_index", "query_formula", "selector", "attenuation", "step",
        "baseline_state", "cell_id",
    }
    if required_b1 - set(b1_panel.columns):
        raise RuntimeError("E4-B1 panel lacks fields required for discovery reuse")
    screen = b1_panel.copy()
    screen["panel_split"] = "screen"
    screen["paired_group"] = [
        f"screen|{cell_id}|{state}"
        for cell_id, state in screen[["cell_id", "baseline_state"]]
        .itertuples(index=False, name=None)
    ]
    screen_counts = screen.groupby(["cell_id", "baseline_state"])["query_formula"].nunique()
    if (
        len(screen_counts) != 18
        or screen_counts.ne(args.screen_formulas_per_stratum).any()
        or screen.duplicated(["paired_group", "query_formula"]).any()
    ):
        raise RuntimeError("E4-B1 discovery panel is not the frozen 9x2x32 design")

    excluded_formulas = set(screen["query_formula"].astype(str))
    confirmation = merged.loc[
        ~merged["query_formula"].astype(str).isin(excluded_formulas)
    ].copy()
    blocks: list[pd.DataFrame] = [screen]
    availability: list[dict[str, object]] = []
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        cell_id = f"{selector}|a={float(attenuation):.2f}|step={int(step)}"
        cell = confirmation.loc[confirmation["cell_id"].eq(cell_id)].copy()
        for state in ("official_error", "official_correct"):
            block = cell.loc[cell["baseline_state"].eq(state)].copy()
            block["_query_order"] = [
                stable_key(args.seed + 1, "confirm", cell_id, state, formula, query)
                for formula, query in block[["query_formula", "query_index"]]
                .itertuples(index=False, name=None)
            ]
            one_per_formula = (
                block.sort_values("_query_order", kind="stable")
                .drop_duplicates("query_formula", keep="first")
            )
            formulas = sorted(
                one_per_formula["query_formula"].astype(str),
                key=lambda value: stable_key(args.seed + 2, "confirm", cell_id, state, value),
            )
            selected_count = min(args.maximum_confirm_formulas_per_stratum, len(formulas))
            availability.append({
                "cell_id": cell_id,
                "baseline_state": state,
                "panel_split": "confirm",
                "available_formulas": len(formulas),
                "selected_formulas": selected_count,
            })
            if selected_count == 0:
                raise RuntimeError(
                    f"{cell_id}|{state} has no confirmation formula after B1 exclusion"
                )
            chosen = set(formulas[:selected_count])
            selected = one_per_formula.loc[
                one_per_formula["query_formula"].astype(str).isin(chosen)
            ].copy()
            selected["panel_split"] = "confirm"
            selected["paired_group"] = f"confirm|{cell_id}|{state}"
            blocks.append(selected.drop(columns="_query_order"))

    panel = pd.concat(blocks, ignore_index=True)
    if panel.duplicated(["paired_group", "query_formula"]).any():
        raise RuntimeError("panel has a duplicate formula within a cell/state stratum")
    screen_formulas = set(
        panel.loc[panel["panel_split"].eq("screen"), "query_formula"].astype(str)
    )
    confirm_formulas = set(
        panel.loc[panel["panel_split"].eq("confirm"), "query_formula"].astype(str)
    )
    if screen_formulas.intersection(confirm_formulas):
        raise RuntimeError("B1 discovery and confirmation formulas overlap")
    result = panel.sort_values(
        ["panel_split", "cell_id", "baseline_state", "query_formula", "query_index"],
        kind="stable",
    ).reset_index(drop=True)
    result.attrs["availability"] = availability
    return result


def flatten_gradient(values: list[torch.Tensor | None]) -> torch.Tensor:
    if not values or any(value is None for value in values):
        raise RuntimeError("E4-B2 requires every unfrozen parameter to receive a gradient")
    parts = [value.detach().float().reshape(-1).cpu() for value in values]
    return torch.cat(parts)


def scope_indices(names: list[str], parameters: list[torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    labels: list[torch.Tensor] = []
    for name, parameter in zip(names, parameters):
        labels.append(torch.full(
            (parameter.numel(),), name.startswith("head."), dtype=torch.bool,
        ))
    head = torch.cat(labels)
    if not head.any() or head.all():
        raise RuntimeError("head/backbone parameter partition is empty")
    return {
        "joint": torch.arange(len(head), dtype=torch.long),
        "head": torch.nonzero(head, as_tuple=False).flatten(),
        "backbone": torch.nonzero(~head, as_tuple=False).flatten(),
    }


def transformed_gradient(action: torch.Tensor, clean: torch.Tensor,
                         family: str, beta: float) -> torch.Tensor:
    action_norm = float(torch.linalg.vector_norm(action))
    clean_norm = float(torch.linalg.vector_norm(clean))
    if action_norm <= 0 or clean_norm <= 0:
        raise RuntimeError("zero gradient in E4-B2 transformation")
    output = action.clone()
    if family in {"pcgrad", "pcgrad_anchor"}:
        dot = float(torch.dot(output, clean))
        if dot < 0:
            output.add_(clean, alpha=-dot / (clean_norm ** 2))
    if family in {"anchor", "pcgrad_anchor"}:
        output.add_(clean, alpha=beta * action_norm / clean_norm)
    return output


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    # A projection can deliberately collapse an exactly anti-parallel action to
    # no-op.  Record zero rather than an undefined cosine; retention/norm_ratio
    # then make the collapse fail the action-preservation gate.
    return float(torch.dot(left, right)) / denominator if denominator > 0 else 0.0


def summarize_values(frame: pd.DataFrame, resamples: int, seed: int,
                     adjusted_alpha: float | None = None) -> dict[str, object]:
    output: dict[str, object] = {
        "actions": int(len(frame)),
        "formulas": int(frame["query_formula"].nunique()),
        "identities": int(frame["identity"].nunique()),
    }
    if len(frame) != frame["query_formula"].nunique():
        raise RuntimeError("summary does not have one action per formula")
    for position, metric in enumerate(METRICS):
        interval = bootstrap(
            frame[metric].to_numpy(np.float64), resamples, seed + position,
            adjusted_alpha,
        )
        output[f"{metric}_mean"] = interval["mean"]
        output[f"{metric}_ci_low"] = interval["ci_low"]
        output[f"{metric}_ci_high"] = interval["ci_high"]
        if "multiplicity_adjusted_lower" in interval:
            output[f"{metric}_adjusted_low"] = interval["multiplicity_adjusted_lower"]
    return output


def rank_screen(summary: pd.DataFrame, maximum: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    error = summary.loc[
        summary["panel_split"].eq("screen")
        & summary["baseline_state"].eq("official_error")
    ].copy()
    correct = summary.loc[
        summary["panel_split"].eq("screen")
        & summary["baseline_state"].eq("official_correct"),
        ["cell_id", "configuration", "clean_alignment_mean"],
    ].rename(columns={"clean_alignment_mean": "paired_correct_clean_alignment_mean"})
    error = error.merge(correct, on=["cell_id", "configuration"], validate="one_to_one")
    error["screen_gate_pass"] = (
        error["forward_margin_ci_low"].gt(0)
        & error["gradient_consensus_mean"].gt(0)
        & error["clean_alignment_mean"].gt(0)
        & error["action_descent_retention_mean"].ge(0.5)
        & error["paired_correct_clean_alignment_mean"].gt(0)
    )
    ranking_metrics = [
        "forward_margin_mean", "gradient_consensus_mean", "clean_alignment_mean",
        "action_descent_retention_mean", "paired_correct_clean_alignment_mean",
    ]
    error["screen_rank_sum"] = sum(
        error[column].rank(method="min", ascending=False) for column in ranking_metrics
    )
    passing = error.loc[error["screen_gate_pass"]].sort_values(
        ["screen_rank_sum", "cell_id", "configuration"], kind="stable",
    )
    per_cell = passing.drop_duplicates("cell_id", keep="first")
    selected = per_cell.head(maximum).copy()
    columns = [
        "cell_id", "configuration", "screen_rank_sum", "screen_gate_pass",
        *ranking_metrics,
    ]
    return error, selected[columns].to_dict("records")


def main() -> None:
    args = arguments()
    if (
        args.outer_fold != 0 or args.screen_formulas_per_stratum != 32
        or args.maximum_confirm_formulas_per_stratum != 64
        or args.maximum_confirm_candidates != 3
        or args.screen_bootstraps != 5000 or args.confirm_bootstraps != 50000
    ):
        raise ValueError("formal E4-B2 freezes fold/panel/candidate/bootstrap settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal E4-B2 requires CUDA")
    required = [
        args.graph, args.r0_dir / "report.json", args.r0_dir / "training_actions.csv.gz",
        args.error_analysis / "report.json", args.error_analysis / "query_error_signatures.csv.gz",
        args.e4b0_dir / "report.json", args.e4b1_dir / "report.json",
        args.e4b1_dir / "panel.csv.gz", args.data, args.embedding_cache,
        args.official_checkpoint, args.architecture_checkpoint, args.clean_checkpoint,
        args.clean_checkpoint.parent / "decision.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E4-B2 output: {args.output_dir}")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device(args.device)

    b1_report = json.loads((args.e4b1_dir / "report.json").read_text(encoding="utf-8"))
    if (
        b1_report.get("status") != "noise_final_e4b1_stratified_gradient_complete"
        or not b1_report.get("formal") or b1_report.get("pass_to_training") is not False
        or b1_report.get("contracts", {}).get("optimizer_steps") != 0
        or b1_report.get("contracts", {}).get("P2b") != "forbidden"
    ):
        raise RuntimeError("E4-B1 is not the frozen zero-update input")
    b0_report = json.loads((args.e4b0_dir / "report.json").read_text(encoding="utf-8"))
    clean_sha = sha256_file(args.clean_checkpoint)
    if clean_sha != b0_report.get("provenance", {}).get("clean_checkpoint_sha256"):
        raise RuntimeError("clean checkpoint differs from frozen E4-B0")
    b1_panel = pd.read_csv(args.e4b1_dir / "panel.csv.gz", low_memory=False)
    excluded_formulas = set(b1_panel["query_formula"].astype(str))

    graph = CandidateGraph(args.graph)
    official_rank, official_margin = official_rank_margin(graph)
    signature = load_signature(args, graph, official_rank)
    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden.intersection(actions.columns):
        raise RuntimeError("action outcome leaked into R0 manifest")
    selected_cells: list[pd.DataFrame] = []
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        block = actions.loc[
            actions["selector"].astype(str).eq(selector)
            & np.isclose(actions["attenuation"].astype(float), attenuation)
            & actions["step"].astype(int).eq(step)
            & actions["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        if block.empty:
            raise RuntimeError(f"missing action cell {selector}|{attenuation}|{step}")
        selected_cells.append(block)
    panel = select_discovery_confirmation_panel(
        pd.concat(selected_cells, ignore_index=True), signature,
        b1_panel, args,
    )
    availability = panel.attrs.get("availability", [])
    print(json.dumps({
        "status": "E4-B2 expanded discovery/confirmation preflight passed",
        "actions": int(len(panel)),
        "B1_excluded_formulas": int(len(excluded_formulas)),
        "availability": availability,
        "note": "GPU gradients alternate with CPU gradient-surgery algebra; temporary zero GPU utilization is expected while configuration progress continues",
    }, indent=2), flush=True)
    examples = make_gradient_examples(
        graph, panel, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    )
    metadata = panel.set_index(["query_index", "selector", "step"], verify_integrity=True)
    example_groups: dict[str, list[object]] = {}
    for example in examples:
        row = metadata.loc[(example.query_index, example.selector, example.step)]
        example_groups.setdefault(str(row["paired_group"]), []).append(example)

    reachable = sorted({
        row for example in examples
        for row in (example.query_row, *example.positive_rows, *example.negative_rows)
    })
    store = SpectrumStore(args.data, np.asarray(reachable), args.n_highest_peaks)
    _, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    missing_embeddings = [row for row in reachable if row not in cache_index]
    if missing_embeddings:
        raise RuntimeError(f"embedding cache misses rows: {missing_embeddings[:20]}")
    official_by_row = {
        int(row): cache_embeddings[cache_index[int(row)]] for row in reachable
    }
    model, model_provenance, names, parameters = load_checkpoint_model(
        "clean_continuation", args, device,
    )
    indices = scope_indices(names, parameters)

    formula_records: list[dict[str, object]] = []
    for group_position, (group_id, group_examples) in enumerate(sorted(example_groups.items())):
        print(
            f"[E4-B2 group-start] {group_position + 1}/{len(example_groups)} "
            f"{group_id} formulas={len(group_examples)}",
            flush=True,
        )
        action_vectors: list[torch.Tensor] = []
        clean_vectors: list[torch.Tensor] = []
        margins: list[float] = []
        rows: list[pd.Series] = []
        ordered_examples = sorted(
            group_examples, key=lambda item: (item.formula, item.query_index)
        )
        for example_position, example in enumerate(ordered_examples):
            row = metadata.loc[(example.query_index, example.selector, example.step)]
            losses, outputs = loss_bundle(model, store, [example], official_by_row, device, args)
            clean_grad = gradients(losses["common"], parameters, True)
            action_grad = gradients(losses["paired_advantage"], parameters, False)
            clean_vectors.append(flatten_gradient(clean_grad))
            action_vectors.append(flatten_gradient(action_grad))
            margins.append(float(outputs["paired_advantage"][0].detach()))
            rows.append(row)
            del losses, outputs, clean_grad, action_grad
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (example_position + 1) % 8 == 0 or example_position + 1 == len(ordered_examples):
                print(
                    f"[E4-B2 gradients] {group_id} "
                    f"{example_position + 1}/{len(ordered_examples)}",
                    flush=True,
                )
        for scope in SCOPES:
            index = indices[scope]
            scoped_action = [value.index_select(0, index) for value in action_vectors]
            scoped_clean = [value.index_select(0, index) for value in clean_vectors]
            for method_position, (method_name, family, beta) in enumerate(METHODS):
                safe = [
                    transformed_gradient(action, clean, family, beta)
                    for action, clean in zip(scoped_action, scoped_clean)
                ]
                total = torch.stack(safe, dim=0).sum(dim=0)
                configuration = f"{method_name}|scope={scope}"
                for formula_position, (example, row, action, clean, safe_value, margin) in enumerate(zip(
                    ordered_examples,
                    rows, scoped_action, scoped_clean, safe, margins,
                )):
                    action_norm = float(torch.linalg.vector_norm(action))
                    clean_norm = float(torch.linalg.vector_norm(clean))
                    safe_norm = float(torch.linalg.vector_norm(safe_value))
                    remainder = total - safe_value
                    formula_records.append({
                        "panel_split": str(row["panel_split"]),
                        "cell_id": str(row["cell_id"]),
                        "baseline_state": str(row["baseline_state"]),
                        "configuration": configuration,
                        "method": method_name,
                        "scope": scope,
                        "query_index": int(example.query_index),
                        "query_formula": str(example.formula),
                        "identity": str(example.identity),
                        "forward_margin": margin,
                        "clean_alignment": cosine(safe_value, clean),
                        "gradient_consensus": cosine(safe_value, remainder),
                        "action_descent_retention": (
                            float(torch.dot(action, safe_value)) / (action_norm ** 2)
                        ),
                        "action_fidelity": cosine(safe_value, action),
                        "norm_ratio": safe_norm / action_norm,
                    })
                del safe, total
                print(
                    f"[E4-B2 surgery] {group_id} scope={scope} "
                    f"method={method_name} {method_position + 1}/{len(METHODS)}",
                    flush=True,
                )
            del scoped_action, scoped_clean
        del action_vectors, clean_vectors
        gc.collect()
        print(
            f"[E4-B2 groups] {group_position + 1}/{len(example_groups)} {group_id}",
            flush=True,
        )

    per_formula = pd.DataFrame(formula_records)
    numeric = per_formula.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise RuntimeError("non-finite E4-B2 formula diagnostic")
    summaries: list[dict[str, object]] = []
    for position, (key, frame) in enumerate(per_formula.groupby(
        ["panel_split", "cell_id", "baseline_state", "configuration"], sort=True,
    )):
        split, cell_id, state, configuration = map(str, key)
        row = {
            "panel_split": split, "cell_id": cell_id,
            "baseline_state": state, "configuration": configuration,
        }
        row.update(summarize_values(
            frame, args.screen_bootstraps, args.seed + 100 * position,
        ))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    screen_table, selected = rank_screen(summary, args.maximum_confirm_candidates)

    confirm_results: list[dict[str, object]] = []
    confirm_alpha = (
        0.05 / (len(selected) * 5) if selected else None
    )
    for position, candidate in enumerate(selected):
        cell_id = str(candidate["cell_id"])
        configuration = str(candidate["configuration"])
        error_frame = per_formula.loc[
            per_formula["panel_split"].eq("confirm")
            & per_formula["cell_id"].eq(cell_id)
            & per_formula["baseline_state"].eq("official_error")
            & per_formula["configuration"].eq(configuration)
        ]
        correct_frame = per_formula.loc[
            per_formula["panel_split"].eq("confirm")
            & per_formula["cell_id"].eq(cell_id)
            & per_formula["baseline_state"].eq("official_correct")
            & per_formula["configuration"].eq(configuration)
        ]
        error_stats = summarize_values(
            error_frame, args.confirm_bootstraps, args.seed + 1_000_000 + position,
            confirm_alpha,
        )
        correct_stats = summarize_values(
            correct_frame, args.confirm_bootstraps, args.seed + 2_000_000 + position,
            confirm_alpha,
        )
        gates = {
            "error_forward_margin_positive": error_stats["forward_margin_adjusted_low"] > 0,
            "error_consensus_positive": error_stats["gradient_consensus_adjusted_low"] > 0,
            "error_clean_alignment_positive": error_stats["clean_alignment_adjusted_low"] > 0,
            "error_action_retention_ge_0_5": error_stats["action_descent_retention_adjusted_low"] >= 0.5,
            "paired_correct_clean_alignment_positive": correct_stats["clean_alignment_adjusted_low"] > 0,
        }
        confirm_results.append({
            "cell_id": cell_id, "configuration": configuration,
            "screen_rank_sum": candidate["screen_rank_sum"],
            "error": error_stats, "paired_correct": correct_stats,
            "gates": gates, "confirm_pass": bool(all(gates.values())),
        })
    confirmed = [
        f"{row['cell_id']}|{row['configuration']}"
        for row in confirm_results if row["confirm_pass"]
    ]

    report = {
        "status": "noise_final_e4b2_gradient_surgery_complete",
        "formal": True,
        "selection_requirements": {
            "screen_formulas_per_stratum": int(args.screen_formulas_per_stratum),
            "maximum_confirm_formulas_per_stratum": int(
                args.maximum_confirm_formulas_per_stratum
            ),
            "minimum_confirm_formulas_per_stratum": 1,
            "screen_bootstraps": int(args.screen_bootstraps),
            "confirm_bootstraps": int(args.confirm_bootstraps),
        },
        "panel": {
            "actions": int(len(panel)),
            "B1_excluded_formulas": int(len(excluded_formulas)),
            "screen_formulas": int(panel.loc[panel["panel_split"].eq("screen"), "query_formula"].nunique()),
            "confirm_formulas": int(panel.loc[panel["panel_split"].eq("confirm"), "query_formula"].nunique()),
            "screen_confirm_formula_overlap": 0,
            "availability": availability,
        },
        "matrix": {
            "cells": 9, "methods": [value[0] for value in METHODS],
            "scopes": list(SCOPES), "configurations_per_cell": len(METHODS) * len(SCOPES),
        },
        "screen": {
            "passing_configurations": int(screen_table["screen_gate_pass"].sum()),
            "selected_candidates": selected,
            "maximum_candidates": args.maximum_confirm_candidates,
            "selection_used_confirm_outcomes": False,
        },
        "confirmation": {
            "candidates": len(selected), "endpoints": 5,
            "one_sided_bonferroni_alpha": confirm_alpha,
            "results": confirm_results,
            "confirmed_configurations": confirmed,
        },
        "has_confirmed_gradient_surgery": bool(confirmed),
        "pass_to_training": False,
        "decision": (
            "freeze confirmed configuration(s) and write a separate tiny overfit protocol"
            if confirmed else
            "no gradient-surgery configuration independently confirmed; do not train"
        ),
        "model": model_provenance,
        "gates": {
            "B1_formulas_excluded_from_confirmation": not bool(
                set(panel.loc[panel["panel_split"].eq("confirm"), "query_formula"].astype(str))
                .intersection(excluded_formulas)
            ),
            "screen_confirm_globally_disjoint": not bool(
                set(panel.loc[panel["panel_split"].eq("screen"), "query_formula"].astype(str)).intersection(
                    set(panel.loc[panel["panel_split"].eq("confirm"), "query_formula"].astype(str))
                )
            ),
            "screen_reuses_frozen_B1_discovery_panel": bool(
                set(
                    panel.loc[panel["panel_split"].eq("screen"), [
                        "query_index", "selector", "attenuation", "step",
                    ]].itertuples(index=False, name=None)
                )
                == set(
                    b1_panel[["query_index", "selector", "attenuation", "step"]]
                    .itertuples(index=False, name=None)
                )
            ),
            "all_nine_cells": panel["cell_id"].nunique() == 9,
            "all_confirmation_strata_nonempty": bool(
                len(availability) == 18
                and all(
                    int(row["selected_formulas"]) >= 1
                    for row in availability
                )
            ),
            "all_18_configurations": per_formula["configuration"].nunique() == 18,
            "optimizer_steps_zero": True,
            "P2b_forbidden": True,
            "P3_not_consumed": True,
        },
        "contracts": {
            "action_outcomes_used_for_panel_selection": False,
            "official_baseline_state_used_for_stratification": True,
            "screen_source": "frozen E4-B1 discovery formulas",
            "confirmation_source": (
                "B1-formula-disjoint R0 actions, up to 64 formulas per cell/state"
            ),
            "error_correct_formula_pairing_required": False,
            "confirm_outcomes_used_for_screen_selection": False,
            "same_query_target_and_two_controls": True,
            "model_eval_mode": True,
            "optimizer_steps": 0,
            "weights_changed": False,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            "e4b0_report_sha256": sha256_file(args.e4b0_dir / "report.json"),
            "e4b1_report_sha256": sha256_file(args.e4b1_dir / "report.json"),
            "e4b1_panel_sha256": sha256_file(args.e4b1_dir / "panel.csv.gz"),
            "r0_actions_sha256": sha256_file(args.r0_dir / "training_actions.csv.gz"),
            "graph_sha256": sha256_file(args.graph),
            "clean_checkpoint_sha256": clean_sha,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Independent zero-update gradient screen/confirmation. It is not a trained "
            "embedding, retrieval gain, deployable policy, or P3 result."
        ),
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"E4-B2 structural gates failed: {report['gates']}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    panel.to_csv(temporary / "panel.csv.gz", index=False, compression="gzip")
    per_formula.to_csv(temporary / "per_formula.csv.gz", index=False, compression="gzip")
    summary.to_csv(temporary / "screen_summary.csv.gz", index=False, compression="gzip")
    screen_table.to_csv(temporary / "screen_ranking.csv.gz", index=False, compression="gzip")
    report["artifacts"] = {
        "panel_sha256": sha256_file(temporary / "panel.csv.gz"),
        "per_formula_sha256": sha256_file(temporary / "per_formula.csv.gz"),
        "screen_summary_sha256": sha256_file(temporary / "screen_summary.csv.gz"),
        "screen_ranking_sha256": sha256_file(temporary / "screen_ranking.csv.gz"),
    }
    json_dump(temporary / "report.json", report)
    temporary.replace(args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

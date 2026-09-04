#!/usr/bin/env python
"""Build the auditable historical noise-action ledger used by the final campaign.

E0 is deliberately descriptive.  It does not train a model, select an action
using its outcome, or claim clean-spectrum embedding improvement.  It brings
the S1a/S1c/S2/S3A/A4 experiments under one schema so that later experiments
cannot silently change the denominator, outcome definition, or control arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SPECS = (
    ("S1a", Path("data/validation/g8r_noise_v3_s1a_single_peak_matrix")),
    ("S1c", Path("data/validation/g8r_noise_v3_s1c_topk_matrix")),
    ("S2", Path("data/validation/g8r_noise_v3_s2_sequential")),
    ("S3A", Path("data/validation/g8r_noise_v3_s3a_extended_matrix")),
)
A4_DIR = Path("data/validation/g8r_noise_v3_a4_exact_peak_scan")
A4_TEACHER_DIR = Path("data/validation/g8r_noise_v3_a4_action_teacher")

FORBIDDEN_FIELDS = ("p2b", "rank_fusion", "reranker", "raw_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/g8r_noise_final_e0_unified_matrix"),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Local forensic mode. Formal server execution must omit this flag.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def column(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def outcome(corrected: pd.Series, introduced: pd.Series, baseline_rank: pd.Series) -> pd.Series:
    result = np.full(len(corrected), "stable", dtype=object)
    result[np.asarray(corrected, dtype=bool)] = "corrected"
    result[np.asarray(introduced, dtype=bool)] = "introduced"
    stable = ~(np.asarray(corrected, dtype=bool) | np.asarray(introduced, dtype=bool))
    baseline_correct = pd.to_numeric(baseline_rank, errors="coerce").eq(1).to_numpy()
    result[stable & baseline_correct] = "protected_correct"
    result[stable & ~baseline_correct] = "persistent_wrong"
    return pd.Series(result, index=corrected.index)


LEDGER_COLUMNS = [
    "stage",
    "evidence_level",
    "query_index",
    "query_row",
    "query_ik14",
    "query_formula",
    "has_near",
    "baseline_rank",
    "baseline_margin",
    "scan_kind",
    "error_family",
    "positive_deficit",
    "negative_excess",
    "selector",
    "operator",
    "attenuation",
    "step",
    "role",
    "token",
    "action_index",
    "hard_negative_row",
    "result_rank",
    "result_margin",
    "margin_change",
    "random_margin_change",
    "specific_margin_excess",
    "specific_top1_excess",
    "corrected",
    "introduced",
    "outcome",
    "risk_net_lambda2",
    "source_file",
]


def normalize_sequential(frame: pd.DataFrame, stage: str, source: Path) -> pd.DataFrame:
    corrected = as_bool(column(frame, "corrected", False))
    introduced = as_bool(column(frame, "introduced", False))
    baseline_rank = pd.to_numeric(column(frame, "baseline_rank"), errors="coerce")
    target_rank = pd.to_numeric(column(frame, "target_rank", column(frame, "rank")), errors="coerce")
    target_margin = pd.to_numeric(column(frame, "target_margin", column(frame, "margin")), errors="coerce")
    baseline_margin = pd.to_numeric(column(frame, "baseline_margin"), errors="coerce")
    margin_change = pd.to_numeric(
        column(frame, "target_margin_change", target_margin - baseline_margin), errors="coerce"
    )
    random_margin_change = pd.to_numeric(column(frame, "random_margin_change"), errors="coerce")
    specific_margin = pd.to_numeric(
        column(frame, "target_minus_random_margin_change", margin_change - random_margin_change),
        errors="coerce",
    )
    specific_top1 = pd.to_numeric(column(frame, "target_minus_random_top1"), errors="coerce")
    selector = column(frame, "selector", stage).astype(str)
    out = pd.DataFrame(
        {
            "stage": stage,
            "evidence_level": "paired_direct_action_with_matched_random_control",
            "query_index": pd.to_numeric(column(frame, "query_index"), errors="coerce"),
            "query_row": pd.to_numeric(column(frame, "query_row"), errors="coerce"),
            "query_ik14": column(frame, "query_ik14", "").astype(str),
            "query_formula": column(frame, "query_formula", "").astype(str),
            "has_near": as_bool(column(frame, "has_near", False)),
            "baseline_rank": baseline_rank,
            "baseline_margin": baseline_margin,
            "scan_kind": "candidate_graph",
            "error_family": "unknown",
            "positive_deficit": False,
            "negative_excess": False,
            "selector": selector,
            "operator": "peak_attenuation",
            "attenuation": pd.to_numeric(column(frame, "attenuation"), errors="coerce"),
            "step": pd.to_numeric(column(frame, "step", 1), errors="coerce"),
            "role": column(frame, "target_role", "unknown").astype(str),
            "token": column(frame, "applied_token", column(frame, "target_path", "")).astype(str),
            "action_index": np.nan,
            "hard_negative_row": pd.to_numeric(column(frame, "hard_negative_row"), errors="coerce"),
            "result_rank": target_rank,
            "result_margin": target_margin,
            "margin_change": margin_change,
            "random_margin_change": random_margin_change,
            "specific_margin_excess": specific_margin,
            "specific_top1_excess": specific_top1,
            "corrected": corrected,
            "introduced": introduced,
            "outcome": outcome(corrected, introduced, baseline_rank),
            "risk_net_lambda2": corrected.astype(int) - 2 * introduced.astype(int),
            "source_file": str(source.as_posix()),
        }
    )
    return out[LEDGER_COLUMNS]


def normalize_a4(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    corrected = as_bool(column(frame, "corrected", False))
    introduced = as_bool(column(frame, "introduced", False))
    baseline_rank = pd.to_numeric(column(frame, "baseline_rank"), errors="coerce")
    result_rank = pd.to_numeric(column(frame, "result_rank"), errors="coerce")
    result_margin = pd.to_numeric(column(frame, "result_margin"), errors="coerce")
    baseline_margin = pd.to_numeric(column(frame, "baseline_margin"), errors="coerce")
    margin_change = pd.to_numeric(column(frame, "margin_change", result_margin - baseline_margin), errors="coerce")
    out = pd.DataFrame(
        {
            "stage": "A4",
            "evidence_level": "exact_single_peak_action_without_random_specificity_claim",
            "query_index": pd.to_numeric(column(frame, "query_index"), errors="coerce"),
            "query_row": np.nan,
            "query_ik14": column(frame, "query_ik14", "").astype(str),
            "query_formula": column(frame, "query_formula", "").astype(str),
            "has_near": as_bool(column(frame, "has_near", False)),
            "baseline_rank": baseline_rank,
            "baseline_margin": baseline_margin,
            "scan_kind": column(frame, "scan_kind", "unknown").astype(str),
            "error_family": column(frame, "score_error_family", "unknown").astype(str),
            "positive_deficit": as_bool(column(frame, "positive_deficit", False)),
            "negative_excess": as_bool(column(frame, "negative_excess", False)),
            "selector": "exact_peak_scan",
            "operator": "single_peak_attenuation",
            "attenuation": pd.to_numeric(column(frame, "attenuation"), errors="coerce"),
            "step": 1,
            "role": column(frame, "role", "unknown").astype(str),
            "token": column(frame, "token", "").astype(str),
            "action_index": pd.to_numeric(column(frame, "action_index"), errors="coerce"),
            "hard_negative_row": np.nan,
            "result_rank": result_rank,
            "result_margin": result_margin,
            "margin_change": margin_change,
            "random_margin_change": np.nan,
            "specific_margin_excess": np.nan,
            "specific_top1_excess": np.nan,
            "corrected": corrected,
            "introduced": introduced,
            "outcome": outcome(corrected, introduced, baseline_rank),
            "risk_net_lambda2": corrected.astype(int) - 2 * introduced.astype(int),
            "source_file": str(source.as_posix()),
        }
    )
    return out[LEDGER_COLUMNS]


def parse_cell(cell: str) -> tuple[str, float, int]:
    parts = cell.split("|")
    selector = parts[0]
    attenuation = float(next(part.split("=", 1)[1] for part in parts if part.startswith("a=")))
    step = int(next(part.split("=", 1)[1] for part in parts if part.startswith("step=")))
    return selector, attenuation, step


def stage_cell_rows(stage: str, directory: Path) -> list[dict[str, Any]]:
    report_path = directory / "report.json"
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cells = report.get("cell_results", {})
    rows: list[dict[str, Any]] = []
    for cell, values in cells.items():
        selector, attenuation, step = parse_cell(cell)
        corrected = int(values.get("corrected", 0))
        introduced = int(values.get("introduced", 0))
        rows.append(
            {
                "stage": stage,
                "cell": cell,
                "selector": selector,
                "attenuation": attenuation,
                "step": step,
                "queries": int(values.get("queries", 0)),
                "identities": int(values.get("identities", 0)),
                "formulas": int(values.get("formulas", 0)),
                "corrected": corrected,
                "introduced": introduced,
                "net": corrected - introduced,
                "risk_net_lambda2": corrected - 2 * introduced,
                "mean_specific_margin": values.get("mean_target_minus_random_margin", np.nan),
                "identity_specific_top1_ci_low": (values.get("baseline_wrong_identity_top1_ci") or [np.nan])[0],
                "formula_specific_top1_ci_low": (values.get("baseline_wrong_formula_top1_ci") or [np.nan])[0],
                "evidence_level": "report_cell_with_matched_random_control",
            }
        )
    return rows


def a4_cell_rows(directory: Path) -> list[dict[str, Any]]:
    decision_path = directory / "decision.json"
    if not decision_path.exists():
        return []
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    rows = []
    for values in decision.get("dose_results", []):
        corrected = int(values["unique_recoverable_errors"])
        introduced = int(values["unique_at_risk_controls"])
        attenuation = float(values["attenuation"])
        rows.append(
            {
                "stage": "A4",
                "cell": f"exact_peak_scan|a={attenuation:.2f}|step=1",
                "selector": "exact_peak_scan",
                "attenuation": attenuation,
                "step": 1,
                "queries": np.nan,
                "identities": np.nan,
                "formulas": np.nan,
                "corrected": corrected,
                "introduced": introduced,
                "net": corrected - introduced,
                "risk_net_lambda2": corrected - 2 * introduced,
                "mean_specific_margin": np.nan,
                "identity_specific_top1_ci_low": np.nan,
                "formula_specific_top1_ci_low": np.nan,
                "gradient_exact_spearman": values.get("gradient_exact_spearman", np.nan),
                "evidence_level": "post_outcome_exact_action_oracle",
            }
        )
    return rows


def update_aggregates(aggregates: dict[tuple[Any, ...], dict[str, float]], frame: pd.DataFrame) -> None:
    keys = ["stage", "selector", "attenuation", "step", "role", "has_near", "error_family", "scan_kind"]
    numeric = frame.copy()
    numeric["margin_valid"] = numeric["margin_change"].notna().astype(int)
    numeric["specific_valid"] = numeric["specific_margin_excess"].notna().astype(int)
    numeric["margin_sum"] = numeric["margin_change"].fillna(0.0)
    numeric["specific_sum"] = numeric["specific_margin_excess"].fillna(0.0)
    grouped = numeric.groupby(keys, dropna=False, observed=True).agg(
        action_rows=("query_index", "size"),
        corrected=("corrected", "sum"),
        introduced=("introduced", "sum"),
        margin_sum=("margin_sum", "sum"),
        margin_valid=("margin_valid", "sum"),
        specific_sum=("specific_sum", "sum"),
        specific_valid=("specific_valid", "sum"),
    )
    for key, values in grouped.iterrows():
        target = aggregates[key]
        for name, value in values.items():
            target[name] += float(value)


def finalize_aggregates(aggregates: dict[tuple[Any, ...], dict[str, float]]) -> pd.DataFrame:
    rows = []
    names = ["stage", "selector", "attenuation", "step", "role", "has_near", "error_family", "scan_kind"]
    for key, values in aggregates.items():
        row = dict(zip(names, key))
        corrected = int(values["corrected"])
        introduced = int(values["introduced"])
        row.update(
            {
                "action_rows": int(values["action_rows"]),
                "corrected": corrected,
                "introduced": introduced,
                "net": corrected - introduced,
                "risk_net_lambda2": corrected - 2 * introduced,
                "correction_precision": corrected / max(corrected + introduced, 1),
                "mean_margin_change": values["margin_sum"] / max(values["margin_valid"], 1),
                "mean_specific_margin_excess": values["specific_sum"] / max(values["specific_valid"], 1),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(names, kind="stable").reset_index(drop=True)


def append_csv(frame: pd.DataFrame, path: Path, first: bool) -> None:
    frame.to_csv(path, index=False, mode="w" if first else "a", header=first, compression="gzip")


def plot_cell_summary(cells: pd.DataFrame, output: Path) -> None:
    usable = cells[cells["stage"].isin(["S1a", "S1c", "S2", "S3A"])].copy()
    if usable.empty:
        return
    labels = (usable["stage"] + ":" + usable["selector"] + "@" + usable["attenuation"].map(lambda x: f"{x:.2f}"))
    usable["row_label"] = labels
    pivot = usable.pivot_table(index="row_label", columns="step", values="risk_net_lambda2", aggfunc="max")
    pivot = pivot.sort_index()
    data = pivot.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(data))), 1.0)
    fig_h = max(5.0, 0.34 * len(pivot))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    image = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound)
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Sequential attenuation step")
    ax.set_ylabel("Stage and preregistered selector")
    ax.set_title("Historical noise-action safety matrix: corrected - 2 x introduced")
    fig.colorbar(image, ax=ax, label="Risk-weighted net")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_correction_tradeoff(cells: pd.DataFrame, output: Path) -> None:
    if cells.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    palette = {stage: color for stage, color in zip(sorted(cells.stage.unique()), plt.cm.tab10.colors)}
    for stage, part in cells.groupby("stage"):
        ax.scatter(part["introduced"], part["corrected"], s=42, alpha=0.8, label=stage, color=palette[stage])
    maximum = max(float(cells[["introduced", "corrected"]].to_numpy().max()), 1.0)
    x = np.linspace(0, maximum, 100)
    ax.plot(x, x, color="black", linestyle="--", linewidth=1, label="corrected = introduced")
    ax.plot(x, 2 * x, color="firebrick", linestyle=":", linewidth=1.5, label="corrected = 2 x introduced")
    ax.set_xlim(left=-0.02 * maximum)
    ax.set_ylim(bottom=-0.02 * maximum)
    ax.set_xlabel("Introduced errors")
    ax.set_ylabel("Corrected errors")
    ax.set_title("Direct action headroom is a correction-risk trade-off, not model gain")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_a4_dose(cells: pd.DataFrame, output: Path) -> None:
    part = cells[cells["stage"].eq("A4")].sort_values("attenuation")
    if part.empty:
        return
    fig, left = plt.subplots(figsize=(9.0, 5.8))
    left.plot(part["attenuation"], part["corrected"], marker="o", label="Recoverable errors", color="#2b6cb0")
    left.plot(part["attenuation"], part["introduced"], marker="o", label="At-risk controls", color="#c53030")
    left.set_xlabel("Single-peak attenuation")
    left.set_ylabel("Unique queries")
    left.set_title("A4 exact action dose-response: larger gradients also enlarge collateral risk")
    right = left.twinx()
    if "gradient_exact_spearman" in part:
        right.plot(
            part["attenuation"], part["gradient_exact_spearman"], marker="s", linestyle="--", color="#2f855a",
            label="Gradient-exact Spearman",
        )
        right.set_ylabel("Gradient-exact Spearman")
    handles, labels = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    left.legend(handles + handles2, labels + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E0 output: {args.output_dir}")

    resolved_specs = [(stage, ROOT / rel) for stage, rel in SOURCE_SPECS]
    a4_dir = ROOT / A4_DIR
    teacher_dir = ROOT / A4_TEACHER_DIR
    missing = [str(directory) for _, directory in resolved_specs if not (directory / "report.json").exists()]
    if not (a4_dir / "decision.json").exists():
        missing.append(str(a4_dir))
    if not (teacher_dir / "decision.json").exists():
        missing.append(str(teacher_dir))
    if missing and not args.allow_partial:
        raise RuntimeError("formal E0 requires all historical artifacts; missing: " + ", ".join(missing))

    temporary = Path(tempfile.mkdtemp(prefix="noise_e0_", dir=args.output_dir.parent))
    source_manifest: dict[str, Any] = {}
    cell_rows: list[dict[str, Any]] = []
    aggregates: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ledger_path = temporary / "unified_query_action_ledger.csv.gz"
    first_write = True
    total_rows = 0

    try:
        for stage, directory in resolved_specs:
            report_path = directory / "report.json"
            paired_path = directory / "paired_interventions.csv.gz"
            available = report_path.exists() and paired_path.exists()
            source_manifest[stage] = {
                "directory": str(directory),
                "available": available,
                "report_sha256": sha256(report_path) if report_path.exists() else None,
                "paired_sha256": sha256(paired_path) if paired_path.exists() else None,
            }
            if not available:
                continue
            report = read_json(report_path)
            if not report.get("formal", False):
                raise RuntimeError(f"{stage} report is not formal")
            cell_rows.extend(stage_cell_rows(stage, directory))
            stage_rows = 0
            for chunk in pd.read_csv(paired_path, chunksize=args.chunk_size):
                normalized = normalize_sequential(chunk, stage, paired_path.relative_to(ROOT))
                append_csv(normalized, ledger_path, first_write)
                first_write = False
                stage_rows += len(normalized)
                total_rows += len(normalized)
                update_aggregates(aggregates, normalized)
            source_manifest[stage]["ledger_rows"] = stage_rows

        a4_policy = a4_dir / "policy_candidate_actions.csv.gz"
        source_manifest["A4"] = {
            "directory": str(a4_dir),
            "available": (a4_dir / "decision.json").exists() and a4_policy.exists(),
            "decision_sha256": sha256(a4_dir / "decision.json") if (a4_dir / "decision.json").exists() else None,
            "policy_actions_sha256": sha256(a4_policy) if a4_policy.exists() else None,
        }
        if source_manifest["A4"]["available"]:
            cell_rows.extend(a4_cell_rows(a4_dir))
            a4_rows = 0
            for chunk in pd.read_csv(a4_policy, chunksize=args.chunk_size):
                normalized = normalize_a4(chunk, a4_policy.relative_to(ROOT))
                append_csv(normalized, ledger_path, first_write)
                first_write = False
                a4_rows += len(normalized)
                total_rows += len(normalized)
                update_aggregates(aggregates, normalized)
            source_manifest["A4"]["ledger_rows"] = a4_rows

        teacher_decision = teacher_dir / "decision.json"
        source_manifest["A4_teacher"] = {
            "directory": str(teacher_dir),
            "available": teacher_decision.exists(),
            "decision_sha256": sha256(teacher_decision) if teacher_decision.exists() else None,
        }
        if teacher_decision.exists():
            teacher = read_json(teacher_decision)
            source_manifest["A4_teacher"]["pass_to_counterfactual_training"] = bool(
                teacher.get("pass_to_counterfactual_training", False)
            )
            source_manifest["A4_teacher"]["oof_policy"] = teacher.get("oof_policy", {})

        if first_write:
            raise RuntimeError("no complete historical query-action source was available")

        cells = pd.DataFrame(cell_rows)
        cells.to_csv(temporary / "historical_cell_summary.csv", index=False)
        effect_matrix = finalize_aggregates(aggregates)
        effect_matrix.to_csv(temporary / "unified_action_effect_matrix.csv", index=False)

        # Materialise changed-outcome explanations without pretending stable rows need a cause.
        changed_parts: list[pd.DataFrame] = []
        transition_path = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix/transition_audit.csv.gz"
        if transition_path.exists():
            transition = pd.read_csv(transition_path)
            keep = [
                "query_index", "query_row", "query_ik14", "query_formula", "has_near", "baseline_rank",
                "baseline_margin", "selector", "attenuation", "step", "target_role", "target_path",
                "target_rank", "target_margin", "target_margin_change", "target_minus_random_margin_change",
                "corrected", "introduced", "transition", "winner_mces_grade_name", "winner_same_formula",
                "baseline_rule_jaccard", "target_rule_jaccard", "baseline_rule_jaccard_core",
                "target_rule_jaccard_core", "baseline_rule_jaccard_massbank", "target_rule_jaccard_massbank",
            ]
            transition = transition[[name for name in keep if name in transition]].copy()
            transition.insert(0, "stage", "S3A")
            changed_parts.append(transition)
            source_manifest["S3A"]["transition_sha256"] = sha256(transition_path)
        if a4_policy.exists():
            changes = []
            for chunk in pd.read_csv(a4_policy, chunksize=args.chunk_size):
                mask = as_bool(column(chunk, "corrected", False)) | as_bool(column(chunk, "introduced", False))
                if mask.any():
                    part = chunk.loc[mask].copy()
                    part.insert(0, "stage", "A4")
                    changes.append(part)
            if changes:
                changed_parts.append(pd.concat(changes, ignore_index=True))
        if changed_parts:
            all_columns = sorted(set().union(*(part.columns for part in changed_parts)))
            aligned = [part.reindex(columns=all_columns) for part in changed_parts]
            pd.concat(aligned, ignore_index=True).to_csv(
                temporary / "changed_outcome_explanations.csv.gz", index=False, compression="gzip"
            )

        plot_cell_summary(cells, temporary / "e0_risk_weighted_action_matrix.png")
        plot_correction_tradeoff(cells, temporary / "e0_correction_risk_tradeoff.png")
        plot_a4_dose(cells, temporary / "e0_a4_dose_response.png")

        ledger_header = pd.read_csv(ledger_path, nrows=0).columns.tolist()
        forbidden = [name for name in ledger_header if any(token in name.lower() for token in FORBIDDEN_FIELDS)]
        if forbidden:
            raise RuntimeError(f"downstream-expert fields leaked into E0 ledger: {forbidden}")
        if total_rows <= 0 or cells.empty or effect_matrix.empty:
            raise RuntimeError("E0 output is unexpectedly empty")

        formal = not args.allow_partial
        required_available = all(source_manifest.get(stage, {}).get("available", False) for stage in ["S1a", "S1c", "S2", "S3A", "A4", "A4_teacher"])
        if formal and not required_available:
            raise RuntimeError("formal E0 lost one or more required historical sources")

        s3a_decision_path = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix/decision.json"
        a4_decision_path = a4_dir / "decision.json"
        headroom = {}
        if s3a_decision_path.exists():
            s3a_decision = read_json(s3a_decision_path)
            headroom["S3A"] = s3a_decision.get("no_op_aware_headroom", {})
        if a4_decision_path.exists():
            a4_decision = read_json(a4_decision_path)
            headroom["A4"] = a4_decision.get("exact_action_oracle", {})

        manifest = {
            "status": "noise_final_e0_unified_matrix_complete",
            "formal": formal,
            "historical_sources_complete": required_available,
            "ledger_rows": total_rows,
            "cell_rows": len(cells),
            "effect_matrix_rows": len(effect_matrix),
            "sources": source_manifest,
            "historical_headroom": headroom,
            "contracts": {
                "scope": "noise fine-tuning action evidence only",
                "output_target": "shared query/reference DreaMS embedding space",
                "p2b_or_downstream_expert_fields": "forbidden",
                "direct_action_effect": "not clean-spectrum model gain",
                "random_control_specificity": "kept separate from raw action effect",
                "oracle_selection": "headroom only and never a training label",
                "ties": "count against positive where inherited from source experiment",
            },
            "outputs": {
                "ledger_sha256": sha256(ledger_path),
                "cell_summary_sha256": sha256(temporary / "historical_cell_summary.csv"),
                "effect_matrix_sha256": sha256(temporary / "unified_action_effect_matrix.csv"),
            },
            "claim_limit": (
                "E0 unifies historical direct interventions, matched-control specificity and transition risks. "
                "It does not establish that any action transfers to clean-spectrum shared-encoder retrieval."
            ),
        }
        (temporary / "e0_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

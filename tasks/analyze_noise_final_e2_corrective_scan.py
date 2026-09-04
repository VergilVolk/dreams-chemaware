"""Preregistered cell-wise decision analysis for the E2 corrective scan."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_corrective_scan"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_matrix_manifest"))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--minimum-error-identities", type=int, default=100)
    parser.add_argument("--minimum-error-formulas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def cluster_bootstrap(frame: pd.DataFrame, value: str, cluster: str, resamples: int, seed: int) -> dict[str, float]:
    local = frame[[cluster, value]].dropna().copy()
    grouped = local.groupby(cluster, sort=False)[value].agg(["sum", "count"])
    if len(grouped) < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "clusters": int(len(grouped))}
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draw = rng.integers(0, len(sums), size=len(sums))
        distribution[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(sums.sum() / counts.sum()),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "clusters": int(len(grouped)),
    }


def main() -> None:
    args = parse_args()
    args.scan_dir, args.manifest_dir = resolve(args.scan_dir), resolve(args.manifest_dir)
    report = json.loads((args.scan_dir / "report.json").read_text(encoding="utf-8"))
    if not report.get("formal") or report.get("status") != "noise_final_e2_corrective_scan_complete":
        raise RuntimeError("formal E2 corrective scan is not complete")
    frame = pd.read_csv(args.scan_dir / "paired_corrective_interventions.csv.gz")
    cells = pd.read_csv(args.manifest_dir / "e2_preregistered_cells.csv")
    cells = cells.loc[cells["arm"].isin(["corrective", "negative_control"])].copy()
    if cells["cell_id"].duplicated().any() or len(cells) != 32:
        raise RuntimeError("expected 32 frozen E2-M1 cells")

    rows = []
    for cell_number, cell in enumerate(cells.itertuples(index=False)):
        part = frame.loc[frame["cell_id"].eq(cell.cell_id)].copy()
        errors = part.loc[part["baseline_rank"].gt(1)].copy()
        safety = part.loc[part["baseline_rank"].eq(1)].copy()
        near = part.loc[part["has_near"].astype(bool)].copy()
        corrected = int(part["corrected"].sum())
        introduced = int(part["introduced"].sum())
        identity_ci = cluster_bootstrap(
            errors, "specific_margin_excess", "query_ik14", args.bootstrap_resamples,
            args.seed + 2 * cell_number,
        ) if len(errors) else {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "clusters": 0}
        formula_ci = cluster_bootstrap(
            errors, "specific_margin_excess", "query_formula", args.bootstrap_resamples,
            args.seed + 2 * cell_number + 1,
        ) if len(errors) else {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "clusters": 0}
        near_corrected = int(near["corrected"].sum())
        near_introduced = int(near["introduced"].sum())
        is_corrective = str(cell.arm) == "corrective"
        coverage_gate = (
            int(errors["query_ik14"].nunique()) >= args.minimum_error_identities
            and int(errors["query_formula"].nunique()) >= args.minimum_error_formulas
        )
        specificity_gate = bool(
            np.isfinite(identity_ci["ci_low"]) and identity_ci["ci_low"] > 0
            and np.isfinite(formula_ci["ci_low"]) and formula_ci["ci_low"] > 0
        )
        safety_gate = bool(corrected > introduced and corrected - 2 * introduced > 0)
        near_gate = bool(near_corrected - near_introduced >= 0)
        passed = bool(is_corrective and coverage_gate and specificity_gate and safety_gate and near_gate)
        rows.append({
            **cell._asdict(),
            "eligible_queries": int(len(part)),
            "eligible_errors": int(len(errors)),
            "eligible_safety": int(len(safety)),
            "error_identities": int(errors["query_ik14"].nunique()),
            "error_formulas": int(errors["query_formula"].nunique()),
            "corrected": corrected,
            "introduced": introduced,
            "net": corrected - introduced,
            "risk_net": corrected - 2 * introduced,
            "correction_precision": corrected / max(corrected + introduced, 1),
            "near_corrected": near_corrected,
            "near_introduced": near_introduced,
            "near_net": near_corrected - near_introduced,
            "specific_margin_identity_mean": identity_ci["mean"],
            "specific_margin_identity_ci_low": identity_ci["ci_low"],
            "specific_margin_identity_ci_high": identity_ci["ci_high"],
            "specific_margin_formula_mean": formula_ci["mean"],
            "specific_margin_formula_ci_low": formula_ci["ci_low"],
            "specific_margin_formula_ci_high": formula_ci["ci_high"],
            "coverage_gate": coverage_gate,
            "specificity_gate": specificity_gate,
            "safety_gate": safety_gate,
            "near_gate": near_gate,
            "pass_to_e3": passed,
        })
    decision = pd.DataFrame(rows)
    if set(decision["cell_id"]) != set(cells["cell_id"]):
        raise RuntimeError("not every frozen E2-M1 cell was reported")

    temporary = Path(tempfile.mkdtemp(prefix="noise_e2_analysis_", dir=args.scan_dir.parent))
    try:
        decision.to_csv(temporary / "cell_decisions.csv", index=False)
        selected = decision.loc[decision["pass_to_e3"]].sort_values(
            ["risk_net", "specific_margin_formula_ci_low"], ascending=False,
        )
        selected.to_csv(temporary / "e3_eligible_cells.csv", index=False)
        summary = {
            "status": "noise_final_e2_corrective_decision_complete",
            "formal": True,
            "cells_reported": int(len(decision)),
            "corrective_cells_reported": int(decision["arm"].eq("corrective").sum()),
            "negative_controls_reported": int(decision["arm"].eq("negative_control").sum()),
            "cells_passing_to_e3": int(decision["pass_to_e3"].sum()),
            "passing_cell_ids": selected["cell_id"].astype(str).tolist(),
            "decision_rule": {
                "minimum_error_identities": args.minimum_error_identities,
                "minimum_error_formulas": args.minimum_error_formulas,
                "identity_and_formula_specific_margin_ci_low_gt_zero": True,
                "corrected_gt_introduced": True,
                "corrected_minus_2x_introduced_gt_zero": True,
                "near_net_nonnegative": True,
                "negative_controls_never_eligible": True,
            },
            "claim_limit": "E2 selects perturbation policies for E3. Only E4 can establish improved shared-embedding retrieval.",
        }
        (temporary / "decision.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for source in (args.scan_dir / "report.json", args.scan_dir / "paired_corrective_interventions.csv.gz"):
            shutil.copy2(source, temporary / source.name)
        shutil.rmtree(args.scan_dir)
        temporary.replace(args.scan_dir)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

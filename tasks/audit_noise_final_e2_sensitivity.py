"""E2-M1b: multiplicity, exact-control and error-transition sensitivity audit.

This stage never changes an action outcome.  It reuses the frozen E2-M1
paired interventions and asks whether a discovered cell survives:
  1. exact candidate-role matched controls;
  2. identity/formula clustered uncertainty;
  3. a joint formula-cluster sign-flip max-T correction across all corrective cells.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_noise_v3_a4b_positive_evidence import cluster_bootstrap


ROOT = Path(__file__).resolve().parents[1]
ROLE_MATCHED_SELECTORS = {
    "candidate_gradient",
    "role_confounder",
    "conditional_missingness_x_confounder",
    "conditional_missingness_x_positive_gradient",
    "role_shared",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_corrective_scan"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_sensitivity"))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--max-t-permutations", type=int, default=10000)
    parser.add_argument("--minimum-error-identities", type=int, default=100)
    parser.add_argument("--minimum-error-formulas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def family_name(selector: str, relation: str, arm: str) -> str:
    if arm == "negative_control":
        return f"negative_control|{selector}"
    if selector in {"candidate_gradient", "role_confounder"}:
        return selector
    return f"{relation}|{selector}"


def exact_control_match(row: pd.Series) -> bool:
    selector = str(row["selector"])
    levels = [value for value in str(row.get("control_match_levels", "")).split(",") if value]
    if selector == "uniform_random":
        return True
    if len(levels) != int(row["control_count"]):
        return False
    expected = "role_intensity_mz" if selector in ROLE_MATCHED_SELECTORS else "intensity_mz"
    return bool(levels and all(level == expected for level in levels))


def bootstrap_or_nan(
    frame: pd.DataFrame, cluster: str, resamples: int, seed: int,
) -> dict[str, float]:
    if frame[cluster].nunique() < 2:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    return cluster_bootstrap(
        frame, frame["specific_margin_excess"].to_numpy(float), cluster, resamples, seed,
    )


def max_t_adjusted_pvalues(
    errors: pd.DataFrame, cell_ids: list[str], permutations: int, seed: int,
) -> dict[str, dict[str, float]]:
    """One-sided joint formula sign-flip test, preserving cross-cell dependence."""
    grouped = errors.groupby(["query_formula", "cell_id"], sort=False)["specific_margin_excess"].mean()
    matrix_frame = grouped.unstack("cell_id").reindex(columns=cell_ids)
    values = matrix_frame.to_numpy(float)
    valid = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.0)
    counts = valid.sum(axis=0).astype(float)
    sums = filled.sum(axis=0)
    sumsq = np.square(filled).sum(axis=0)

    def t_stat(current_sums: np.ndarray) -> np.ndarray:
        mean = np.divide(current_sums, counts, out=np.full_like(current_sums, np.nan), where=counts > 0)
        variance = np.divide(
            sumsq - counts * np.square(mean), counts - 1,
            out=np.full_like(mean, np.nan), where=counts > 1,
        )
        se = np.sqrt(np.maximum(variance, 0.0) / counts)
        return np.divide(mean, se, out=np.full_like(mean, np.nan), where=se > 0)

    observed = t_stat(sums)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(cell_ids), dtype=np.int64)
    for _ in range(permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(matrix_frame))
        permuted = t_stat(signs @ filled)
        maximum = float(np.nanmax(permuted))
        exceed += maximum >= observed
    adjusted = (exceed + 1) / (permutations + 1)
    adjusted[~np.isfinite(observed)] = 1.0
    return {
        cell_id: {
            "formula_equal_t": float(observed[index]),
            "max_t_adjusted_p": float(adjusted[index]),
            "formula_clusters": int(counts[index]),
        }
        for index, cell_id in enumerate(cell_ids)
    }


def main() -> None:
    args = parse_args()
    args.input_dir, args.output_dir = resolve(args.input_dir), resolve(args.output_dir)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E2 sensitivity audit: {args.output_dir}")
    report = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
    decision = json.loads((args.input_dir / "decision.json").read_text(encoding="utf-8"))
    if not report.get("formal") or not decision.get("formal"):
        raise RuntimeError("E2-M1 formal inputs are not passing")
    frame = pd.read_csv(args.input_dir / "paired_corrective_interventions.csv.gz")
    frozen = pd.read_csv(args.input_dir / "cell_decisions.csv")
    if len(frozen) != 32 or frozen["cell_id"].nunique() != 32:
        raise RuntimeError("E2-M1 did not report all 32 frozen cells")
    frame["exact_control_match"] = frame.apply(exact_control_match, axis=1)
    frame["family"] = [
        family_name(selector, relation, arm)
        for selector, relation, arm in zip(frame["selector"], frame["acquisition_relation"], frame["arm"])
    ]

    corrective_ids = frozen.loc[frozen["arm"].eq("corrective"), "cell_id"].astype(str).tolist()
    exact_errors = frame.loc[
        frame["arm"].eq("corrective") & frame["baseline_rank"].gt(1) & frame["exact_control_match"]
    ].copy()
    adjusted = max_t_adjusted_pvalues(
        exact_errors, corrective_ids, args.max_t_permutations, args.seed,
    )

    rows = []
    for index, cell in frozen.iterrows():
        part = frame.loc[frame["cell_id"].eq(cell["cell_id"])].copy()
        exact = part.loc[part["exact_control_match"]].copy()
        errors = exact.loc[exact["baseline_rank"].gt(1)].copy()
        near = exact.loc[exact["has_near"].astype(bool)].copy()
        identity_ci = bootstrap_or_nan(errors, "query_ik14", args.bootstrap_resamples, args.seed + 2 * index)
        formula_ci = bootstrap_or_nan(errors, "query_formula", args.bootstrap_resamples, args.seed + 2 * index + 1)
        corrected = int(exact["corrected"].sum())
        introduced = int(exact["introduced"].sum())
        near_corrected = int(near["corrected"].sum())
        near_introduced = int(near["introduced"].sum())
        multi = adjusted.get(str(cell["cell_id"]), {})
        coverage = (
            errors["query_ik14"].nunique() >= args.minimum_error_identities
            and errors["query_formula"].nunique() >= args.minimum_error_formulas
        )
        ci_gate = bool(identity_ci["ci_low"] > 0 and formula_ci["ci_low"] > 0)
        risk_gate = bool(corrected > introduced and corrected - 2 * introduced > 0)
        near_gate = bool(near_corrected - near_introduced >= 0)
        multiplicity_gate = bool(multi.get("max_t_adjusted_p", 1.0) < 0.05)
        passed = bool(
            cell["arm"] == "corrective" and cell["pass_to_e3"]
            and coverage and ci_gate and risk_gate and near_gate and multiplicity_gate
        )
        rows.append({
            **cell.to_dict(),
            "family": family_name(str(cell["selector"]), str(cell["acquisition_relation"]), str(cell["arm"])),
            "all_control_rows": int(len(part)),
            "exact_control_rows": int(len(exact)),
            "exact_control_fraction": float(len(exact) / max(len(part), 1)),
            "exact_error_identities": int(errors["query_ik14"].nunique()),
            "exact_error_formulas": int(errors["query_formula"].nunique()),
            "exact_corrected": corrected,
            "exact_introduced": introduced,
            "exact_risk_net": corrected - 2 * introduced,
            "exact_near_net": near_corrected - near_introduced,
            "exact_identity_margin_mean": identity_ci["mean"],
            "exact_identity_margin_ci_low": identity_ci["ci_low"],
            "exact_identity_margin_ci_high": identity_ci["ci_high"],
            "exact_formula_margin_mean": formula_ci["mean"],
            "exact_formula_margin_ci_low": formula_ci["ci_low"],
            "exact_formula_margin_ci_high": formula_ci["ci_high"],
            **multi,
            "exact_coverage_gate": bool(coverage),
            "exact_ci_gate": ci_gate,
            "exact_risk_gate": risk_gate,
            "exact_near_gate": near_gate,
            "multiplicity_gate": multiplicity_gate,
            "pass_to_e3_after_sensitivity": passed,
        })
    cells = pd.DataFrame(rows)

    family_rows = []
    corrective = frame.loc[frame["arm"].eq("corrective") & frame["exact_control_match"]].copy()
    for family_index, (family, part) in enumerate(corrective.groupby("family", sort=False)):
        errors = part.loc[part["baseline_rank"].gt(1)].copy()
        per_query = errors.groupby(
            ["query_index", "query_ik14", "query_formula"], as_index=False,
        )["specific_margin_excess"].mean()
        identity_ci = bootstrap_or_nan(per_query, "query_ik14", args.bootstrap_resamples, args.seed + 1000 + family_index)
        formula_ci = bootstrap_or_nan(per_query, "query_formula", args.bootstrap_resamples, args.seed + 2000 + family_index)
        family_cells = cells.loc[cells["family"].eq(family)]
        family_rows.append({
            "family": family,
            "cells": int(family_cells["cell_id"].nunique()),
            "cells_originally_passing": int(family_cells["pass_to_e3"].sum()),
            "cells_passing_sensitivity": int(family_cells["pass_to_e3_after_sensitivity"].sum()),
            "error_queries": int(per_query["query_index"].nunique()),
            "error_identities": int(per_query["query_ik14"].nunique()),
            "error_formulas": int(per_query["query_formula"].nunique()),
            "dose_step_positive_fraction": float((family_cells["exact_formula_margin_mean"] > 0).mean()),
            "identity_consensus_mean": identity_ci["mean"],
            "identity_consensus_ci_low": identity_ci["ci_low"],
            "identity_consensus_ci_high": identity_ci["ci_high"],
            "formula_consensus_mean": formula_ci["mean"],
            "formula_consensus_ci_low": formula_ci["ci_low"],
            "formula_consensus_ci_high": formula_ci["ci_high"],
        })
    families = pd.DataFrame(family_rows)

    introduced = frame.loc[frame["introduced"].astype(bool)].copy()
    introduced_summary = introduced.groupby(
        ["cell_id", "family", "selector", "acquisition_relation", "query_formula", "has_near"],
        dropna=False, as_index=False,
    ).agg(
        introduced_queries=("query_index", "nunique"),
        introduced_identities=("query_ik14", "nunique"),
        mean_specific_margin_excess=("specific_margin_excess", "mean"),
        exact_control_fraction=("exact_control_match", "mean"),
    ).sort_values(["introduced_queries", "cell_id"], ascending=[False, True])

    selected = cells.loc[cells["pass_to_e3_after_sensitivity"]].sort_values(
        ["exact_risk_net", "exact_formula_margin_ci_low"], ascending=False,
    )
    temporary = Path(tempfile.mkdtemp(prefix="noise_e2_sensitivity_", dir=args.output_dir.parent))
    try:
        cells.to_csv(temporary / "cell_sensitivity.csv", index=False)
        families.to_csv(temporary / "family_consensus.csv", index=False)
        selected.to_csv(temporary / "e3_candidate_cells.csv", index=False)
        introduced_summary.to_csv(temporary / "introduced_error_atlas.csv.gz", index=False, compression="gzip")
        summary = {
            "status": "noise_final_e2_sensitivity_complete",
            "formal": True,
            "cells_audited": int(len(cells)),
            "corrective_cells": int(cells["arm"].eq("corrective").sum()),
            "originally_passing_cells": int(cells["pass_to_e3"].sum()),
            "sensitivity_passing_cells": int(cells["pass_to_e3_after_sensitivity"].sum()),
            "sensitivity_passing_cell_ids": selected["cell_id"].astype(str).tolist(),
            "families_with_positive_formula_consensus": families.loc[
                families["formula_consensus_ci_low"] > 0, "family"
            ].astype(str).tolist(),
            "exact_control_fraction_overall": float(frame["exact_control_match"].mean()),
            "multiplicity": (
                "one-sided joint formula-cluster sign-flip max-T across all 28 corrective cells"
            ),
            "next_stage": (
                "E3 computes gradient compatibility only for sensitivity-passing cells; nested cells are "
                "clustered before any E4 combination."
            ),
            "claim_limit": "Sensitivity audit of frozen action outcomes; no shared-embedding training result.",
        }
        (temporary / "sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

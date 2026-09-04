"""R0: freeze a faithful S3A training manifest before shared-encoder training.

This stage does not invent new peak actions and does not use P2b.  It copies
the already validated dynamic S3A trajectories, preserving selector, dose,
step, ordered peak path, step-specific hard negative and matched controls.
Action outcomes are written to a separate audit file and are never permitted
as student sample weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import json_dump, sha256_file, stable_fold


ROOT = Path(__file__).resolve().parents[1]
S3A = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix"

# These are the fixed policies supported by the original S3A decision, not an
# outcome-selected per-query policy.
POLICIES = {
    "candidate_gradient": {"dose": 0.50, "steps": (3, 4, 5, 6)},
    "role_confounder": {"dose": 1.00, "steps": (1, 2, 3, 4, 5)},
}

EXPECTED = {
    "candidate_gradient|a=0.50|step=1": (112, 90),
    "candidate_gradient|a=0.50|step=2": (127, 80),
    "candidate_gradient|a=0.50|step=3": (142, 66),
    "candidate_gradient|a=0.50|step=4": (145, 54),
    "candidate_gradient|a=0.50|step=5": (140, 47),
    "candidate_gradient|a=0.50|step=6": (140, 42),
    "role_confounder|a=1.00|step=1": (19, 1),
    "role_confounder|a=1.00|step=2": (22, 1),
    "role_confounder|a=1.00|step=3": (25, 1),
    "role_confounder|a=1.00|step=4": (23, 1),
    "role_confounder|a=1.00|step=5": (24, 0),
    "role_confounder|a=1.00|step=6": (20, 0),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3a-dir", type=Path, default=S3A)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a",
    )
    parser.add_argument("--formula-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def tokens_prefix(value: object, step: int) -> str:
    tokens = [token for token in str(value).split(",") if token]
    if len(tokens) < step:
        raise RuntimeError(f"trajectory contains {len(tokens)} tokens, needs {step}")
    return ",".join(tokens[:step])


def controls_prefix(value: object, step: int) -> str:
    paths = str(value).split(";")
    if len(paths) != 2:
        raise RuntimeError("S3A faithful manifest requires two matched controls")
    return ";".join(tokens_prefix(path, step) for path in paths)


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R0: {args.output_dir}")
    required = [
        args.s3a_dir / "report.json", args.s3a_dir / "decision.json",
        args.s3a_dir / "matrix_validation.json",
        args.s3a_dir / "selected_sequences.csv.gz",
        args.s3a_dir / "paired_interventions.csv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    decision = json.loads(required[1].read_text(encoding="utf-8"))
    validation = json.loads(required[2].read_text(encoding="utf-8"))
    if not report.get("formal") or report.get("queries") != 23876:
        raise RuntimeError("R0 requires the formal 23,876-query S3A matrix")
    if validation.get("status") != "noise_v3_s3a_matrix_validation_passed":
        raise RuntimeError("S3A fail-closed validation did not pass")
    for cell, (corrected, introduced) in EXPECTED.items():
        observed = decision["action_results"].get(cell)
        if observed is None or (int(observed["corrected"]), int(observed["introduced"])) != (corrected, introduced):
            raise RuntimeError(f"S3A fidelity mismatch for {cell}: {observed}")

    sequences = pd.read_csv(args.s3a_dir / "selected_sequences.csv.gz")
    outcomes = pd.read_csv(args.s3a_dir / "paired_interventions.csv.gz")
    # S3A is an action-eligibility table, not the full clean-query ledger:
    # queries for which an action cannot be constructed are intentionally absent.
    # Fail closed on out-of-range indices here; exact per-cell eligibility and
    # outcomes are checked against the frozen S3A decision below.
    outcome_queries = outcomes["query_index"].astype(np.int64)
    if outcome_queries.lt(0).any() or outcome_queries.ge(23876).any():
        raise RuntimeError("paired S3A outcomes contain a query outside the locked graph")
    manifest_rows: list[pd.DataFrame] = []
    audit_rows: list[pd.DataFrame] = []
    for selector, spec in POLICIES.items():
        dose = float(spec["dose"])
        base = sequences.loc[
            sequences["selector"].astype(str).eq(selector)
            & np.isclose(sequences["attenuation"].astype(float), dose)
        ].copy()
        if base["query_index"].duplicated().any():
            raise RuntimeError(f"duplicate S3A sequence for {selector}")
        for step in spec["steps"]:
            eligible = base.loc[base["steps"].astype(int).ge(step)].copy()
            eligible["step"] = int(step)
            eligible["target_path"] = eligible["target_tokens"].map(lambda value: tokens_prefix(value, step))
            eligible["hard_negative_row"] = eligible["hard_negative_rows"].map(
                lambda value: int(tokens_prefix(value, step).split(",")[-1])
            )
            eligible["matched_control_paths"] = eligible["control_paths"].map(
                lambda value: controls_prefix(value, step)
            )
            action = outcomes.loc[
                outcomes["selector"].astype(str).eq(selector)
                & np.isclose(outcomes["attenuation"].astype(float), dose)
                & outcomes["step"].astype(int).eq(step)
            ].copy()
            if len(action) != len(eligible):
                raise RuntimeError(f"trajectory/outcome count mismatch for {selector} step {step}")
            join = action[[
                "query_index", "query_row", "query_ik14", "query_formula", "has_near",
                "baseline_rank", "baseline_margin", "target_rank", "target_margin",
                "random_margin", "corrected", "introduced",
            ]].merge(
                eligible[["query_index", "target_path", "hard_negative_row", "matched_control_paths"]],
                on="query_index", how="inner", validate="one_to_one",
            )
            join["selector"] = selector
            join["attenuation"] = dose
            join["step"] = int(step)
            join["formula_fold"] = join["query_formula"].astype(str).map(
                lambda value: stable_fold(value, args.formula_folds, args.seed)
            ).astype(np.int8)
            audit_rows.append(join.copy())
            # Outcomes are intentionally stripped from the student manifest.
            manifest_rows.append(join[[
                "query_index", "query_row", "query_ik14", "query_formula", "has_near",
                "selector", "attenuation", "step", "target_path",
                "hard_negative_row", "matched_control_paths", "formula_fold",
            ]].copy())

    manifest = pd.concat(manifest_rows, ignore_index=True)
    audit = pd.concat(audit_rows, ignore_index=True)
    if manifest.duplicated(["query_index", "selector", "attenuation", "step"]).any():
        raise RuntimeError("R0 manifest contains duplicate query-policy-step rows")
    if any(column in manifest for column in ("corrected", "introduced", "target_rank", "target_margin")):
        raise RuntimeError("action outcomes leaked into the student manifest")

    cell = audit.groupby(["selector", "attenuation", "step"], as_index=False).agg(
        queries=("query_index", "size"), identities=("query_ik14", "nunique"),
        formulas=("query_formula", "nunique"), corrected=("corrected", "sum"),
        introduced=("introduced", "sum"),
    )
    for row in cell.itertuples(index=False):
        key = f"{row.selector}|a={float(row.attenuation):.2f}|step={int(row.step)}"
        expected = EXPECTED[key]
        if (int(row.corrected), int(row.introduced)) != expected:
            raise RuntimeError(f"R0 recomputation mismatch for {key}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_r0_", dir=args.output_dir.parent))
    try:
        manifest.to_csv(temporary / "training_actions.csv.gz", index=False, compression="gzip")
        audit.to_csv(temporary / "outcome_audit_only.csv.gz", index=False, compression="gzip")
        cell.to_csv(temporary / "cell_fidelity.csv", index=False)
        body = {
            "status": "noise_final_r0_faithful_s3a_manifest_complete",
            "formal": True,
            "queries_in_locked_graph": 23876,
            "official_errors": 1805,
            "training_action_rows": int(len(manifest)),
            "training_identities": int(manifest["query_ik14"].nunique()),
            "training_formulas": int(manifest["query_formula"].nunique()),
            "policies": POLICIES,
            "fidelity_cells": cell.to_dict("records"),
            "historical_headroom": {
                "fixed_candidate_step6_net": 98,
                "combined_s1c_s2_s3a_oracle_delta": float(
                    decision["no_op_aware_headroom"]["combined_delta_recall1_upper_bound"]
                ),
                "claim_limit": "oracle selects action/stop using outcomes; not student performance",
            },
            "contracts": {
                "dynamic_sequences_copied_without_recomputation": True,
                "ordered_peak_paths_preserved": True,
                "step_specific_hard_negative_preserved": True,
                "matched_controls_preserved": 2,
                "action_outcomes_absent_from_training_manifest": True,
                "P2b": "forbidden",
            },
            "provenance": {
                "s3a_report_sha256": sha256_file(args.s3a_dir / "report.json"),
                "s3a_decision_sha256": sha256_file(args.s3a_dir / "decision.json"),
                "s3a_sequences_sha256": sha256_file(args.s3a_dir / "selected_sequences.csv.gz"),
                "s3a_paired_sha256": sha256_file(args.s3a_dir / "paired_interventions.csv.gz"),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "next_stage": "R1 trains shared encoder with noisy->clean identity and privileged margin objectives",
        }
        json_dump(temporary / "report.json", body)
        temporary.replace(args.output_dir)
        print(json.dumps(body, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

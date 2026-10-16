#!/usr/bin/env python
"""Post-fit capacity and exchange diagnosis for a finished WSE residual run.

This entrypoint exists so a run started before the diagnostic fields were
added need not be cancelled or retrained.  It only reads frozen bundles,
continuous-chemistry evidence, candidate predictions, and the existing
cross-fit report; it writes one descriptive JSON file and never fits a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_grand_unified_evidence_model import load_bundle
from train_unified_candidate_energy_crossfit import formula_fold
from train_wse_conditional_residual import build_panel, residual_diagnostics, sha256


def bounds_for_panel(report: dict, formulas: np.ndarray) -> np.ndarray:
    folds = int(report["formula_crossfit_folds"])
    seed = int(report["optimization"]["seed"])
    fold_bounds = {
        int(record["fold"]): float(
            record["arms"]["B_joint_residual"]["residual_bound_raw_wse_units"]
        )
        for record in report["folds"]
    }
    if set(fold_bounds) != set(range(folds)) or any(value <= 0 for value in fold_bounds.values()):
        raise RuntimeError("incomplete B residual-bound ledger in model report")
    return np.asarray([
        fold_bounds[formula_fold(str(formula), folds, seed)] for formula in formulas
    ], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--identity-chem", type=Path, required=True)
    parser.add_argument("--formula-chem", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    report = json.loads(args.model_report.read_text(encoding="utf-8"))
    if report.get("status") != "WSE_CONDITIONAL_RESIDUAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("unsupported WSE residual report")
    bundles = {
        "identity": load_bundle(args.identity_bundle),
        "formula": load_bundle(args.formula_bundle),
    }
    panels = {
        "identity": build_panel(bundles["identity"], args.identity_chem),
        "formula": build_panel(bundles["formula"], args.formula_chem),
    }
    diagnostics = {}
    with np.load(args.candidate_predictions, allow_pickle=False) as predictions:
        for name, panel in panels.items():
            prefix = f"{name}__"
            for key, expected in (
                ("query_ptr", panel["query_ptr"]),
                ("labels", panel["labels"]),
                ("query_ids", panel["query_ids"]),
                ("candidate_ids", panel["candidate_ids"]),
            ):
                observed = np.asarray(predictions[prefix + key])
                if not np.array_equal(observed.astype(str), np.asarray(expected).astype(str)):
                    raise RuntimeError(f"prediction axis mismatch: {name}/{key}")
            scores = {
                method: np.asarray(predictions[prefix + method], dtype=np.float32)
                for method in ("wse", "old_four_module", "A", "B")
            }
            diagnostics[name] = residual_diagnostics(
                panel, scores, bounds_for_panel(report, panel["formulas"]),
            )
    output = {
        "status": "WSE_CONDITIONAL_RESIDUAL_POSTFIT_DIAGNOSTICS_COMPLETE",
        "dataset_role": "development_consumed",
        "enveda_accessed": False,
        "model_report": str(args.model_report),
        "candidate_predictions": str(args.candidate_predictions),
        "interpretation": (
            "descriptive capacity/exchange analysis only; no parameter fitting, "
            "threshold selection, residual-bound tuning, or routing"
        ),
        "panels": diagnostics,
        "input_sha256": {
            "model_report": sha256(args.model_report),
            "candidate_predictions": sha256(args.candidate_predictions),
            "identity_bundle": sha256(args.identity_bundle),
            "formula_bundle": sha256(args.formula_bundle),
            "identity_chem": sha256(args.identity_chem),
            "formula_chem": sha256(args.formula_chem),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

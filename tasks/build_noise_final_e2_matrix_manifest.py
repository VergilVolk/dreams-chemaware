#!/usr/bin/env python
"""Freeze the preregistered E2 noise-action matrix before model forwards.

E2 separates corrective interventions from acquisition-robustness views.  It
does not train an encoder and it never uses P2b.  Historical outcomes justify
which hypotheses are carried forward, but cannot alter cells after E2 begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e0-dir", type=Path,
        default=Path("data/validation/g8r_noise_final_e0_unified_matrix"),
    )
    parser.add_argument(
        "--e1-dir", type=Path,
        default=Path("data/validation/g8r_noise_final_e1_empirical_calibration"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/g8r_noise_final_e2_matrix_manifest"),
    )
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_cell(rows: list[dict[str, Any]], **values: Any) -> None:
    rows.append({"cell_id": f"E2-{len(rows):03d}", **values})


def main() -> None:
    args = parse_args()
    args.e0_dir, args.e1_dir, args.output_dir = map(
        resolve, (args.e0_dir, args.e1_dir, args.output_dir)
    )
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite frozen E2 manifest: {args.output_dir}")
    e0_path = args.e0_dir / "e0_manifest.json"
    e1_path = args.e1_dir / "e1_report.json"
    recipe_path = args.e1_dir / "frozen_empirical_noise_recipe.json"
    history_path = args.e0_dir / "historical_cell_summary.csv"
    for path in (e0_path, e1_path, recipe_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    e0 = json.loads(e0_path.read_text(encoding="utf-8"))
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not e0.get("formal") or not e0.get("historical_sources_complete"):
        if not args.allow_smoke:
            raise RuntimeError("E2 requires complete formal E0")
    if not e1.get("formal") or not e1.get("pass_to_e2"):
        if not args.allow_smoke:
            raise RuntimeError("E2 requires passing formal E1")
    if e1["contracts"].get("P3_identity_overlap") != 0:
        raise RuntimeError("E1 P3 isolation contract failed")
    if e1["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("downstream expert leaked into E1")
    if sha256(recipe_path) != e1["provenance"]["recipe_sha256"]:
        raise RuntimeError("E1 frozen recipe hash mismatch")

    history = pd.read_csv(history_path)
    required_history = {
        ("candidate_gradient", 0.50),
        ("role_confounder", 1.00),
        ("role_shared", 1.00),
    }
    observed = set(zip(history["selector"].astype(str), history["attenuation"].astype(float)))
    missing_history = required_history - observed
    if missing_history:
        raise RuntimeError(f"E0 misses required historical controls: {sorted(missing_history)}")

    dropout_grid = recipe["peak_dropout"]["e2_screening_grid"]
    eligible_relations = {
        relation: values for relation, values in dropout_grid.items()
        if values["eligible_for_e2"]
    }
    if len(eligible_relations) < 2:
        raise RuntimeError("E1 provides fewer than two eligible acquisition relations")

    rows: list[dict[str, Any]] = []
    common = {
        "matched_random_controls": 3,
        "query_scope": "all official errors plus matched official-correct safety controls",
        "selection_uses_action_outcome": False,
        "p2b_allowed": False,
    }

    # Corrective arm: previously promising selectors, newly crossed with the
    # empirical conditional-missingness selector.  Each fixed cell is reported.
    for step in (3, 4, 5, 6):
        append_cell(
            rows, arm="corrective", selector="candidate_gradient", operator="attenuate",
            acquisition_relation="not_applicable", dose=0.50, step=step,
            hypothesis="remove the highest positive first-order candidate-margin gradients",
            training_eligibility="requires E2 specificity and safety gates", **common,
        )
    for step in (1, 2, 3, 4, 5, 6):
        append_cell(
            rows, arm="corrective", selector="role_confounder", operator="dropout",
            acquisition_relation="not_applicable", dose=1.00, step=step,
            hypothesis="remove peaks supported by the hardest wrong candidate but not the identity candidate",
            training_eligibility="requires E2 specificity and safety gates", **common,
        )
    for relation, values in sorted(eligible_relations.items()):
        for dose in values["doses"]:
            for selector in (
                "empirical_conditional_missingness",
                "conditional_missingness_x_confounder",
                "conditional_missingness_x_positive_gradient",
            ):
                append_cell(
                    rows, arm="corrective", selector=selector, operator="dropout",
                    acquisition_relation=relation, dose=float(dose), step=0,
                    hypothesis="condition-calibrated peak removal improves the true-candidate margin",
                    training_eligibility="requires E2 specificity and safety gates", **common,
                )

    # Robustness arm: these are observation views, not correction labels.
    # Immediate Top-1 correction is descriptive only; E4 decides whether their
    # consistency loss improves the clean shared embedding.
    for relation, values in sorted(eligible_relations.items()):
        for dose in values["doses"]:
            append_cell(
                rows, arm="robustness", selector="empirical_missingness_weighted_random",
                operator="dropout", acquisition_relation=relation, dose=float(dose), step=0,
                hypothesis="model should preserve identity ranking under empirically plausible missing peaks",
                training_eligibility="only as a consistency view after E2 non-destructiveness gate",
                **{**common, "query_scope": "all P3-disjoint training queries"},
            )
    for quantile in ("q25", "q50"):
        append_cell(
            rows, arm="robustness", selector="all_real_fragments", operator="symmetric_log_intensity_jitter",
            acquisition_relation="pooled_matched_replicates",
            dose=float(recipe["abs_log_intensity_jitter"][quantile]), step=0,
            hypothesis="model should preserve identity ranking under empirical intensity variation",
            training_eligibility="only as a consistency view after E2 non-destructiveness gate",
            **{**common, "query_scope": "all P3-disjoint training queries"},
        )
        append_cell(
            rows, arm="robustness", selector="all_real_fragments", operator="symmetric_mz_jitter_ppm",
            acquisition_relation="pooled_matched_replicates",
            dose=float(recipe["abs_mz_jitter_ppm"][quantile]), step=0,
            hypothesis="model should preserve identity ranking under empirical mass variation",
            training_eligibility="only as a consistency view after E2 non-destructiveness gate",
            **{**common, "query_scope": "all P3-disjoint training queries"},
        )
    for quantile in ("q25", "q50"):
        append_cell(
            rows, arm="robustness", selector="same_identity_absent_consensus_peak",
            operator="low_intensity_peak_addition", acquisition_relation="same_identity_same_adduct",
            dose=float(recipe["low_intensity_addition"][quantile]), step=0,
            hypothesis="model should ignore low-intensity peaks observed only in another real replicate",
            training_eligibility="requires cross-fit support rows and E2 non-destructiveness gate",
            **{**common, "query_scope": "all P3-disjoint identities with replicate support"},
        )

    # Negative controls are deliberately retained but can never enter training.
    for dose in (0.25, 0.50, 1.00):
        append_cell(
            rows, arm="negative_control", selector="role_shared", operator="attenuate",
            acquisition_relation="not_applicable", dose=dose, step=1,
            hypothesis="historically unsafe shared-peak deletion must reproduce collateral errors",
            training_eligibility="forbidden", **common,
        )
    append_cell(
        rows, arm="negative_control", selector="uniform_random", operator="dropout",
        acquisition_relation="not_applicable", dose=0.20, step=0,
        hypothesis="generic random masking baseline", training_eligibility="forbidden", **common,
    )

    cells = pd.DataFrame(rows)
    if cells["cell_id"].duplicated().any() or cells.empty:
        raise RuntimeError("E2 cell identifiers are invalid")
    if cells["p2b_allowed"].any() or cells["selection_uses_action_outcome"].any():
        raise RuntimeError("E2 scope contract violated")
    if (cells.loc[cells["arm"] == "negative_control", "training_eligibility"] != "forbidden").any():
        raise RuntimeError("E2 negative control became trainable")

    temporary = Path(tempfile.mkdtemp(prefix="noise_e2_manifest_", dir=args.output_dir.parent))
    try:
        cells.to_csv(temporary / "e2_preregistered_cells.csv", index=False)
        manifest = {
            "status": "noise_final_e2_matrix_manifest_frozen",
            "formal": bool(
                e0.get("formal") and e0.get("historical_sources_complete")
                and e1.get("formal") and e1.get("pass_to_e2")
                and not args.allow_smoke
            ),
            "cells": int(len(cells)),
            "corrective_cells": int(cells["arm"].eq("corrective").sum()),
            "robustness_cells": int(cells["arm"].eq("robustness").sum()),
            "negative_control_cells": int(cells["arm"].eq("negative_control").sum()),
            "eligible_acquisition_relations": sorted(eligible_relations),
            "matched_random_controls_per_target": 3,
            "decision_contract": {
                "corrective": (
                    "target-minus-matched-random margin CI lower bounds must exceed zero at identity and formula "
                    "levels; corrected > introduced and corrected - 2*introduced > 0; near must not degrade"
                ),
                "robustness": (
                    "not a correction label; clean-vs-noisy consistency is evaluated later against the same-dose "
                    "uniform/matched control and must not degrade clean ranking"
                ),
                "cell_selection": "all fixed cells reported; no post-outcome cell deletion or relabelling",
                "shared_encoder": "E2 discovers actions; E4 alone tests transfer to a clean shared embedding",
            },
            "forbidden": [
                "P2b or another downstream candidate expert",
                "rule overlap as identity or distance label",
                "post-outcome oracle action as a deployable result",
                "unfiltered E1 q50/q75 peak-dropout dose",
                "shared-only action entering training",
            ],
            "provenance": {
                "e0_manifest_sha256": sha256(e0_path),
                "e0_history_sha256": sha256(history_path),
                "e1_report_sha256": sha256(e1_path),
                "e1_recipe_sha256": sha256(recipe_path),
                "script_sha256": sha256(Path(__file__)),
            },
            "claim_limit": (
                "This freezes the E2 experiment family. It contains no action outcome and no embedding gain."
            ),
        }
        (temporary / "e2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["provenance"]["cells_sha256"] = sha256(temporary / "e2_preregistered_cells.csv")
        (temporary / "e2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

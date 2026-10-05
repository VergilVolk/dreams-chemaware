"""Import correct and matched-control SIRIUS scores as a source ledger.

TreeScore and CSI:FingerID remain separate evidence families.  The correct
score is never numerically fused with another source.  Intensity permutation,
mass shifting and query-local candidate-role swapping remain three independent
controls.  The output remains unqualified until a formula-disjoint audit passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from import_chemaware_sirius_scores import (
    formula_scores,
    read_tsv,
    structure_scores,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--correct-summary", type=Path, action="append", required=True)
    parser.add_argument("--intensity-summary", type=Path, action="append", required=True)
    parser.add_argument("--mass-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cyclic_map(values: list[int]) -> dict[int, int]:
    if not values:
        return {}
    if len(values) == 1:
        return {values[0]: values[0]}
    return {value: values[(index + 1) % len(values)] for index, value in enumerate(values)}


def finite_or_nan(value: float | None) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else math.nan


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = json.loads((args.source_panel / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "CHEMAWARE_SIRIUS_SOURCE_PANEL_COMPLETE":
        raise RuntimeError("SIRIUS source panel is incomplete")
    if report.get("gates", {}).get("matched_control_cardinality") is not True:
        raise RuntimeError("SIRIUS source panel predates matched spectral controls")
    if report.get("gates", {}).get("instrument_profile_cardinality") is not True:
        raise RuntimeError("SIRIUS source panel was not instrument-profile matched")
    if report.get("gates", {}).get(
        "all_global_formula_structures_fit_top_k_50_summary"
    ) is not True:
        raise RuntimeError("SIRIUS top-k summary would truncate the candidate graph")
    registry = read_tsv(args.source_panel / "formula_feature_registry.tsv")
    candidates = read_tsv(args.source_panel / "candidate_ledger.tsv")
    feature_ids = {
        (int(row["manifest_query"]), row["candidate_formula"]): {
            "correct": row["formula_feature_id"],
            "intensity": row["intensity_rank_permuted_feature_id"],
            "mass": row["mass_shifted_feature_id"],
        }
        for row in registry
    }
    if len(feature_ids) != len(registry):
        raise RuntimeError("duplicate query-formula registry key")

    correct_tree = formula_scores(args.correct_summary)
    intensity_tree = formula_scores(args.intensity_summary)
    mass_tree = formula_scores(args.mass_summary)
    correct_csi = structure_scores(args.correct_summary)
    intensity_csi = structure_scores(args.intensity_summary)
    mass_csi = structure_scores(args.mass_summary)
    if not correct_tree or not correct_csi:
        raise RuntimeError("correct SIRIUS summaries contain no usable scores")

    by_query: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_query[int(row["manifest_query"])].append(row)
    output_rows = []
    missing = defaultdict(int)
    for query, query_rows in sorted(by_query.items()):
        query_rows.sort(key=lambda row: int(row["local_candidate"]))
        local_values = [int(row["local_candidate"]) for row in query_rows]
        if local_values != list(range(len(query_rows))):
            raise RuntimeError(f"non-contiguous candidates for query {query}")
        formulas = sorted({row["formula"] for row in query_rows})
        formula_swap = cyclic_map(list(range(len(formulas))))
        swapped_formula = {
            formula: formulas[formula_swap[index]] for index, formula in enumerate(formulas)
        }
        within_formula_swap = {}
        for formula in formulas:
            locals_for_formula = [
                int(row["local_candidate"]) for row in query_rows if row["formula"] == formula
            ]
            within_formula_swap.update(cyclic_map(locals_for_formula))

        row_by_local = {int(row["local_candidate"]): row for row in query_rows}
        representative_by_formula = {
            formula: next(row for row in query_rows if row["formula"] == formula)
            for formula in formulas
        }
        for row in query_rows:
            local = int(row["local_candidate"])
            formula = row["formula"]
            ids = feature_ids[(query, formula)]
            tree = finite_or_nan(correct_tree.get((ids["correct"], formula)))
            tree_intensity = finite_or_nan(intensity_tree.get((ids["intensity"], formula)))
            tree_mass = finite_or_nan(mass_tree.get((ids["mass"], formula)))
            swap_formula = swapped_formula[formula]
            swap_ids = feature_ids[(query, swap_formula)]
            tree_candidate_swap = finite_or_nan(
                correct_tree.get((swap_ids["correct"], swap_formula))
            )

            csi_key = (ids["correct"], formula, row["ik14"])
            csi = finite_or_nan(correct_csi.get(csi_key))
            csi_intensity = finite_or_nan(
                intensity_csi.get((ids["intensity"], formula, row["ik14"]))
            )
            csi_mass = finite_or_nan(
                mass_csi.get((ids["mass"], formula, row["ik14"]))
            )
            swap_row = row_by_local[within_formula_swap[local]]
            csi_candidate_swap = finite_or_nan(
                correct_csi.get((ids["correct"], formula, swap_row["ik14"]))
            )

            for name, value in (
                ("tree", tree), ("tree_intensity", tree_intensity),
                ("tree_mass", tree_mass), ("tree_candidate_swap", tree_candidate_swap),
                ("csi", csi), ("csi_intensity", csi_intensity),
                ("csi_mass", csi_mass), ("csi_candidate_swap", csi_candidate_swap),
            ):
                missing[name] += int(not math.isfinite(value))

            common = {
                "manifest_query": query, "local_candidate": local,
                "ik14": row["ik14"], "formula": formula,
                "controls_available": 1,
            }
            output_rows.append({
                **common,
                "source_family": "sirius_tree_controlled",
                "scope": "cross_formula",
                "source_score": "" if not math.isfinite(tree) else f"{tree:.17g}",
                "control_a_score": "" if not math.isfinite(tree_intensity) else f"{tree_intensity:.17g}",
                "control_b_score": "" if not math.isfinite(tree_mass) else f"{tree_mass:.17g}",
                "control_c_score": "" if not math.isfinite(tree_candidate_swap) else f"{tree_candidate_swap:.17g}",
            })
            output_rows.append({
                **common,
                "source_family": "sirius_csi_controlled",
                "scope": "within_formula",
                "source_score": "" if not math.isfinite(csi) else f"{csi:.17g}",
                "control_a_score": "" if not math.isfinite(csi_intensity) else f"{csi_intensity:.17g}",
                "control_b_score": "" if not math.isfinite(csi_mass) else f"{csi_mass:.17g}",
                "control_c_score": "" if not math.isfinite(csi_candidate_swap) else f"{csi_candidate_swap:.17g}",
            })

    source_families = {
        "sirius_tree_controlled": {
            "scope": "cross_formula", "larger_is_better": True,
            "confidence_tier": "A_external_fragmentation_tree",
            "matched_controls": [
                "intensity_rank_permuted", "mass_shifted", "candidate_formula_cyclic",
            ],
            "specificity_gate_passed": False,
        },
        "sirius_csi_controlled": {
            "scope": "within_formula", "larger_is_better": True,
            "confidence_tier": "A_external_structure_score",
            "matched_controls": [
                "intensity_rank_permuted", "mass_shifted", "candidate_structure_cyclic",
            ],
            "specificity_gate_passed": False,
        },
    }
    output_report = {
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED",
        "truth_fields_exported": False,
        "source_families": source_families,
        "queries": len(by_query), "candidates": len(candidates),
        "candidate_source_rows": len(output_rows),
        "missing_scores": dict(missing),
        "formal_triplet_mining_authorized": False,
        "scientific_contract": {
            "score_fusion": "none",
            "tree_and_csi_separate": True,
            "matched_controls_required": True,
            "candidate_role_swap_truth_blind": True,
            "next_gate": "formula-disjoint specificity qualification",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_sirius_controlled_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer.writeheader(); writer.writerows(output_rows)
        (temporary / "report.json").write_text(
            json.dumps(output_report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(output_report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Map residual BioAware action headroom beyond the frozen B17 router.

This is a descriptive opened-development atlas.  It applies each explicitly
listed deployment-visible candidate feature in both directions, under the
three already-open DreaMS margin gates.  It records corrections, harms and
incremental coverage beyond B17.  Truth is used only to score the frozen
actions, so no row from this file is a deployable model result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256
from audit_bioaware_b12_multicohort_catalog_action import build_universe
from audit_bioaware_b14_reaction_context_action import add_reaction_features


FEATURES = (
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
    "log_reference_spectra",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "known_path_per_degree",
    "known_seed_per_degree",
    "edge0_reliability",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
    "coabundance_available_fraction",
    "coabundance_log_neighbours_mean",
    "coabundance_multiwitness_fraction",
    "coabundance_actual_abs_top3_mean",
    "coabundance_actual_positive_top3_mean",
    "coabundance_actual_negative_top3_mean",
    "coabundance_actual_sign_stability_top3_mean",
    "coabundance_abs_excess_top3_mean",
    "coabundance_positive_excess_top3_mean",
    "coabundance_negative_excess_top3_mean",
)
MARGINS = (0.04, 0.05, 0.08)


def evaluate_action(
    candidates: pd.DataFrame,
    b17: pd.DataFrame,
    feature: str,
    direction: str,
    margin: float,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        baseline = str(group["baseline_candidate_id"].iloc[0])
        values = group[feature].to_numpy(float)
        optimum = float(np.max(values) if direction == "max" else np.min(values))
        top = group.loc[np.isclose(group[feature], optimum, rtol=0, atol=1e-12)]
        unique = len(top) == 1
        proposal = str(top["candidate_id"].iloc[0]) if unique else baseline
        intervene = bool(
            unique
            and proposal != baseline
            and float(group["baseline_gap"].iloc[0]) <= margin + 1e-15
        )
        final = proposal if intervene else baseline
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline_correct = bool(group["baseline_correct"].iloc[0])
        final_correct = final == truth
        rows.append({
            "query_id": str(query_id),
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": truth,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "baseline_correct": baseline_correct,
            "proposal_candidate_id": proposal,
            "intervene": intervene,
            "final_correct": final_correct,
            "corrected": (not baseline_correct) and final_correct,
            "introduced": baseline_correct and (not final_correct),
        })
    result = pd.DataFrame(rows).merge(
        b17[["query_id", "corrected", "introduced", "final_correct"]],
        on="query_id", suffixes=("", "_b17"), validate="one_to_one",
    )
    corrected = int(result["corrected"].sum())
    introduced = int(result["introduced"].sum())
    new_corrected = result["corrected"] & ~result["corrected_b17"]
    shared_corrected = result["corrected"] & result["corrected_b17"]
    new_introduced = result["introduced"] & ~result["introduced_b17"]
    shared_introduced = result["introduced"] & result["introduced_b17"]
    by_domain = {}
    for source, local in result.groupby("source", sort=True):
        c = int(local["corrected"].sum())
        i = int(local["introduced"].sum())
        by_domain[str(source)] = {
            "corrected": c,
            "introduced": i,
            "risk_net_lambda2": c - 2 * i,
        }
    summary = {
        "feature": feature,
        "direction": direction,
        "margin": margin,
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "interventions": int(result["intervene"].sum()),
        "new_corrected_beyond_B17": int(new_corrected.sum()),
        "shared_corrected_with_B17": int(shared_corrected.sum()),
        "new_introduced_beyond_B17": int(new_introduced.sum()),
        "shared_introduced_with_B17": int(shared_introduced.sum()),
        "new_corrected_identities": int(
            result.loc[new_corrected, "truth_candidate_id"].nunique()
        ),
        "new_corrected_formulas": int(
            result.loc[new_corrected, "truth_formula"].nunique()
        ),
        "every_domain_risk_nonnegative": all(
            item["risk_net_lambda2"] >= 0 for item in by_domain.values()
        ),
        "by_domain": by_domain,
    }
    return summary, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument(
        "--b17-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2/nested_domain_loso_transitions.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.b17_transitions,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, provenance = build_universe(args)
    candidates = add_reaction_features(candidates)
    candidates = candidates.loc[candidates["polarity"].eq("negative")].copy()
    for feature in FEATURES:
        if feature not in candidates:
            raise RuntimeError(f"B19 missing feature: {feature}")
        candidates[feature] = pd.to_numeric(
            candidates[feature], errors="coerce"
        ).fillna(0.0)
    if not np.isfinite(candidates[list(FEATURES)].to_numpy(float)).all():
        raise RuntimeError("B19 non-finite feature values")
    b17 = pd.read_csv(args.b17_transitions)
    if len(b17) != 860 or int(b17["corrected"].sum()) != 57 or int(b17["introduced"].sum()) != 7:
        raise RuntimeError("B19 frozen B17 comparator changed")

    ledger: list[dict] = []
    best_rows: dict[tuple[str, str, float], pd.DataFrame] = {}
    for feature in FEATURES:
        for direction in ("max", "min"):
            for margin in MARGINS:
                summary, rows = evaluate_action(
                    candidates, b17, feature, direction, margin
                )
                ledger.append(summary)
                best_rows[(feature, direction, margin)] = rows
    ledger.sort(key=lambda item: (
        item["new_corrected_beyond_B17"] - 2 * item["new_introduced_beyond_B17"],
        item["risk_net_lambda2"],
        item["new_corrected_beyond_B17"],
        -item["introduced"],
    ), reverse=True)
    top = ledger[:20]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "action_ledger.json"
    atomic_json(ledger_path, {"actions": ledger})
    top_key = (top[0]["feature"], top[0]["direction"], float(top[0]["margin"]))
    top_path = args.output_dir / "top_action_transitions.csv.gz"
    best_rows[top_key].to_csv(top_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b19_residual_action_atlas_complete",
        "formal": True,
        "actions_audited": int(len(ledger)),
        "features": list(FEATURES),
        "directions": ["max", "min"],
        "margins": list(MARGINS),
        "B17_comparator": {
            "corrected": 57,
            "introduced": 7,
            "risk_net_lambda2": 43,
            "residual_official_errors": int((~b17["baseline_correct"] & ~b17["final_correct"]).sum()),
        },
        "top_20_by_incremental_headroom_then_risk": top,
        "provenance": {
            **provenance["provenance"],
            "B17_transitions_sha256": sha256(args.b17_transitions),
            "ledger_sha256": sha256(ledger_path),
            "top_transitions_sha256": sha256(top_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "contracts": {
            "truth_used_to_select_deployable_action": False,
            "purpose": "descriptive residual action headroom only",
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "claim_limit": (
            "B19 is an opened feature-action atlas. Ranking actions by their "
            "observed truth outcome is hypothesis generation, not nested OOF "
            "performance and not permission to train a shared encoder."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Apply the preregistered B47 Track-C gate remediation to frozen U3 outputs.

Implements docs/BIOAWARE_B47_TRACKC_GATE_REMEDIATION_PROTOCOL_20261003.md
exactly: operations O1-O3 in fixed order, then the repaired authorization
predicate R1-R5.  Reads only truth-blind U3 tables; opens no truth; tunes
nothing on outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_PROTOCOL_VERSION = "U3-v3-reaction-signature-20260922"
TOTAL_FROZEN_QUERIES = 51_976
HEADROOM_FRACTION = 0.03
CONCENTRATION_CAP = 0.25
MIN_EFFECTIVE_CANDIDATES = 20
MIN_STUDY_SHARE = 0.25
POOL_PERCENTILE = 75.0

REQUIRED_OPPORTUNITY_COLUMNS = {
    "query_id", "study", "event_top_candidate", "event_top_formula",
    "event_advantage", "candidate_specific_intervention_opportunity",
    "event_top_reference_spectra", "event_top_catalogue_degree",
}
REQUIRED_FEATURE_COLUMNS = {"query_id", "candidate_id"}


def effective_count(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(float)
    if not len(counts) or counts.sum() <= 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(1.0 / np.square(probabilities).sum())


def apply_caps(actions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep at most ceil(cap * N) actions per group, dropping the weakest."""
    limit = math.ceil(CONCENTRATION_CAP * len(actions))
    if limit <= 0:
        return actions.iloc[0:0].copy()
    # Smallest advantage is removed first; ties break by query_id so the
    # operation is deterministic.
    ranked = actions.sort_values(
        ["event_advantage", "query_id"], ascending=[False, True], kind="stable"
    )
    kept = ranked.groupby(column, sort=False).cumcount()
    return ranked[kept < limit].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    u3_dir = args.u3_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite remediation output: {output}")

    report_path = u3_dir / "report.json"
    opportunities_path = u3_dir / "query_event_opportunities.csv.gz"
    features_path = u3_dir / "candidate_event_features.csv.gz"
    for path in (report_path, opportunities_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise RuntimeError(
            "U3 provenance mismatch: expected "
            f"{EXPECTED_PROTOCOL_VERSION}, observed {report.get('protocol_version')}"
        )

    opportunities = pd.read_csv(opportunities_path)
    features = pd.read_csv(features_path)
    missing = REQUIRED_OPPORTUNITY_COLUMNS - set(opportunities.columns)
    if missing:
        raise RuntimeError(f"opportunities missing columns: {sorted(missing)}")
    if not REQUIRED_FEATURE_COLUMNS.issubset(features.columns):
        raise RuntimeError("candidate features lack identity columns")

    pool_reference_q75 = float(np.percentile(
        features.get("reference_spectra", pd.Series(dtype=float)), POOL_PERCENTILE
    )) if "reference_spectra" in features else None
    pool_degree_q75 = float(np.percentile(
        features.get("candidate_catalogue_degree", pd.Series(dtype=float)),
        POOL_PERCENTILE,
    )) if "candidate_catalogue_degree" in features else None
    pool_reference_median = float(np.percentile(
        features.get("reference_spectra", pd.Series(dtype=float)), 50.0
    )) if "reference_spectra" in features else None
    pool_degree_median = float(np.percentile(
        features.get("candidate_catalogue_degree", pd.Series(dtype=float)), 50.0
    )) if "candidate_catalogue_degree" in features else None

    actions = opportunities[
        opportunities["candidate_specific_intervention_opportunity"].astype(bool)
    ].copy()
    raw_count = int(len(actions))

    repaired = apply_caps(actions, "event_top_candidate")
    repaired = apply_caps(repaired, "event_top_formula")
    if pool_reference_q75 is not None:
        repaired = repaired[
            repaired["event_top_reference_spectra"] <= pool_reference_q75
        ].copy()
    if pool_degree_q75 is not None:
        repaired = repaired[
            repaired["event_top_catalogue_degree"] <= pool_degree_q75
        ].copy()
    repaired_count = int(len(repaired))

    effective_candidates = effective_count(repaired["event_top_candidate"]) if repaired_count else 0.0
    candidate_counts = repaired["event_top_candidate"].value_counts()
    formula_counts = repaired["event_top_formula"].value_counts()
    largest_candidate_fraction = (
        float(candidate_counts.iloc[0] / repaired_count) if repaired_count else 1.0
    )
    largest_formula_fraction = (
        float(formula_counts.iloc[0] / repaired_count) if repaired_count else 1.0
    )
    r1 = bool(
        effective_candidates >= MIN_EFFECTIVE_CANDIDATES
        and largest_candidate_fraction <= CONCENTRATION_CAP
        and largest_formula_fraction <= CONCENTRATION_CAP
    )

    if repaired_count and pool_reference_median is not None and pool_degree_median is not None:
        winner_reference_median = float(repaired["event_top_reference_spectra"].median())
        winner_degree_median = float(repaired["event_top_catalogue_degree"].median())
        r2 = bool(
            winner_reference_median <= pool_reference_median
            and winner_degree_median <= pool_degree_median
        )
    else:
        winner_reference_median = winner_degree_median = None
        r2 = False

    required_actions = math.ceil(HEADROOM_FRACTION * TOTAL_FROZEN_QUERIES)
    r3 = bool(repaired_count >= required_actions)

    study_shares = (
        (repaired.groupby("study", sort=True).size() / repaired_count)
        if repaired_count else pd.Series(dtype=float)
    )
    r4 = bool(repaired_count and (study_shares >= MIN_STUDY_SHARE).all())

    authorization = report.get("authorization", {})
    r5 = bool(
        authorization.get("pass_to_frozen_event_ranking_evaluation_structural_nulls")
        or authorization.get("null_supremacy_passed")
    )
    if not r5:
        # The structural-null verdict lives in different report layouts;
        # inherit it only when it is explicitly recorded as passed.
        r5 = bool(report.get("nulls_passed"))

    output.mkdir(parents=True)
    payload = {
        "status": "bioaware_b47_gate_remediation_complete",
        "protocol": "docs/BIOAWARE_B47_TRACKC_GATE_REMEDIATION_PROTOCOL_20261003.md",
        "u3_dir": str(u3_dir),
        "u3_protocol_version": report.get("protocol_version"),
        "raw_intervention_actions": raw_count,
        "repaired_intervention_actions": repaired_count,
        "operations": {
            "O1_candidate_cap": CONCENTRATION_CAP,
            "O2_formula_cap": CONCENTRATION_CAP,
            "O3_pool_percentile": POOL_PERCENTILE,
            "pool_reference_q75": pool_reference_q75,
            "pool_degree_q75": pool_degree_q75,
        },
        "gates": {
            "R1_concentration": {
                "pass": r1,
                "effective_candidates": effective_candidates,
                "largest_candidate_fraction": largest_candidate_fraction,
                "largest_formula_fraction": largest_formula_fraction,
            },
            "R2_identity_quality": {
                "pass": r2,
                "winner_reference_median": winner_reference_median,
                "pool_reference_median": pool_reference_median,
                "winner_degree_median": winner_degree_median,
                "pool_degree_median": pool_degree_median,
            },
            "R3_headroom": {
                "pass": r3,
                "required_actions": required_actions,
            },
            "R4_study_materiality": {
                "pass": r4,
                "study_shares": {str(k): float(v) for k, v in study_shares.items()},
            },
            "R5_null_supremacy_inherited": r5,
        },
        "authorization_after_repair": bool(r1 and r2 and r3 and r4 and r5),
        "claim_limit": (
            "Truth-blind gate repair only. Authorization, if granted, licenses "
            "exactly one confirmatory evaluation opening; it claims no gain."
        ),
    }
    repaired.to_csv(
        output / "repaired_action_ledger.csv.gz", index=False, compression="gzip"
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temporary = Path(handle.name)
    temporary.replace(output / "report.json")
    print(json.dumps({
        "status": payload["status"],
        "raw_actions": raw_count,
        "repaired_actions": repaired_count,
        "authorization_after_repair": payload["authorization_after_repair"],
        "gates_passed": {
            k: v["pass"] if isinstance(v, dict) else v
            for k, v in payload["gates"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

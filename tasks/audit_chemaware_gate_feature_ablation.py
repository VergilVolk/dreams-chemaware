"""Exploratory feature ablation for the frozen spectral-evidence gate.

All variants are trained only on discovery with formula-grouped OOF threshold
selection and evaluated on the already consumed confirmation split.  The raw
consensus route itself is held fixed, so a DreaMS-only confidence model still
depends on raw spectra to define the alternative candidate.  This audit only
asks which information makes that alternative safe; it does not create a new
frozen gate and must not be evaluated on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
FEATURE_SETS = {
    "full": (
        "dreams_top2_margin", "consensus_votes", "distinct_raw_winners",
        "raw_winners_equal_dreams", "raw_top2_margin_mean", "raw_top2_margin_min",
        "raw_top2_margin_max", "consensus_candidate_gap_mean",
        "consensus_candidate_gap_min", "candidate_molecules",
        "candidate_reference_spectra",
    ),
    "raw_confidence_without_dreams_margin": (
        "consensus_votes", "distinct_raw_winners", "raw_winners_equal_dreams",
        "raw_top2_margin_mean", "raw_top2_margin_min", "raw_top2_margin_max",
        "consensus_candidate_gap_mean", "consensus_candidate_gap_min",
        "candidate_molecules", "candidate_reference_spectra",
    ),
    "dreams_geometry_only": (
        "dreams_top2_margin", "candidate_molecules", "candidate_reference_spectra",
    ),
    "raw_agreement_only": (
        "consensus_votes", "distinct_raw_winners", "raw_winners_equal_dreams",
        "candidate_molecules", "candidate_reference_spectra",
    ),
    "raw_margin_only": (
        "raw_top2_margin_mean", "raw_top2_margin_min", "raw_top2_margin_max",
        "consensus_candidate_gap_mean", "consensus_candidate_gap_min",
        "candidate_molecules", "candidate_reference_spectra",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit(frame: pd.DataFrame, features: tuple[str, ...], labels: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=3000, random_state=20260902),
    )
    formula_counts = frame.groupby("formula")["query_index"].transform("size").to_numpy(float)
    weights = 1.0 / formula_counts
    weights *= len(weights) / weights.sum()
    model.fit(frame[list(features)], labels, logisticregression__sample_weight=weights)
    return model


def crossfit(frame: pd.DataFrame, features: tuple[str, ...], folds: int):
    labels = frame["beneficial_route"].astype(int).to_numpy()
    groups = frame["formula"].astype(str).to_numpy()
    probability = np.empty(len(frame), dtype=float)
    for train, valid in GroupKFold(n_splits=folds).split(frame, groups=groups):
        model = fit(frame.iloc[train], features, labels[train])
        probability[valid] = model.predict_proba(frame.iloc[valid][list(features)])[:, 1]
    return probability, fit(frame, features, labels)


def metric(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict:
    active = frame["route_candidate"].to_numpy(bool) & (probability >= threshold)
    old = frame["dreams_correct"].to_numpy(bool)
    new = old.copy()
    new[active] = frame.loc[active, "consensus_correct"].to_numpy(bool)
    corrected = (~old) & new
    introduced = old & (~new)
    return {
        "threshold": float(threshold),
        "activated": int(active.sum()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "utility": int(corrected.sum() - 2 * introduced.sum()),
        "delta_pp": float(100.0 * np.mean(new.astype(float) - old.astype(float))),
    }


def choose_threshold(frame: pd.DataFrame, probability: np.ndarray) -> tuple[float, dict]:
    rows = [metric(frame, probability, float(value)) for value in np.r_[np.linspace(.05, .95, 91), 1.01]]
    best = max(rows, key=lambda row: (row["utility"], -row["introduced"], row["corrected"], row["threshold"]))
    return float(best["threshold"]), best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_gate_feature_ablation_20260903/report.json",
    )
    args = parser.parse_args()
    source = ROOT / "data/validation/chemaware_spectral_consensus_applicability_v4_frozen"
    discovery_path = source / "discovery_gate_ledger.csv.gz"
    confirmation_path = source / "confirmation_gate_ledger.csv.gz"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    discovery = pd.read_csv(discovery_path)
    confirmation = pd.read_csv(confirmation_path)
    route_discovery = discovery.loc[discovery["route_candidate"]].reset_index(drop=True)
    route_confirmation = confirmation.loc[confirmation["route_candidate"]].reset_index(drop=True)
    results = {}
    for name, features in FEATURE_SETS.items():
        oof, model = crossfit(route_discovery, features, args.folds)
        threshold, discovery_result = choose_threshold(route_discovery, oof)
        confirmation_probability = model.predict_proba(
            route_confirmation[list(features)]
        )[:, 1]
        results[name] = {
            "features": list(features),
            "discovery_oof": discovery_result,
            "confirmation": metric(route_confirmation, confirmation_probability, threshold),
        }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "chemaware_gate_feature_ablation_complete",
        "exploratory_post_hoc": True,
        "test_split_read": False,
        "route_definition_held_fixed": "raw consensus differs from DreaMS with at least 3/5 votes",
        "important_limit": (
            "Every ablation still uses the raw-spectral route definition. The DreaMS-only "
            "variant removes raw features from confidence estimation, not from the action."
        ),
        "discovery_route_candidates": int(len(route_discovery)),
        "confirmation_route_candidates": int(len(route_confirmation)),
        "results": results,
        "provenance": {
            "discovery_ledger_sha256": sha256(discovery_path),
            "confirmation_ledger_sha256": sha256(confirmation_path),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

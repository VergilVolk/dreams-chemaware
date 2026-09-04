#!/usr/bin/env python
"""Identity-isolated HILIC RT evidence audit for the BioAware expert.

MetDNA3 requires predicted RT in addition to MS1, MS2 and reaction topology.
This script tests that missing evidence arm without tuning on annotation
outcomes.  For every held-out rotation it trains a fixed random-forest RT
model only on that rotation's Level-1 seed identities, predicts every DreaMS
candidate from structure, and applies the published 30% relative-RT window.

The deployable rule is intentionally conservative: keep DreaMS unless its
Top-1 fails the RT window; only then choose the highest-DreaMS candidate that
passes.  Truth is used after decisions are frozen, never as a feature or gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from audit_bioaware_metdna3_smn_headroom import (
        aggregate_majority,
        bootstrap_delta,
        load_smiles_by_ik14,
        sha256,
    )
except ModuleNotFoundError:
    from tasks.audit_bioaware_metdna3_smn_headroom import (
        aggregate_majority,
        bootstrap_delta,
        load_smiles_by_ik14,
        sha256,
    )


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def descriptor_matrix(smiles_by_id: dict[str, str]) -> tuple[list[str], np.ndarray, list[str]]:
    """Compute the fixed RDKit 2D descriptor panel for all valid structures."""
    descriptor_names = [name for name, _ in Descriptors.descList]
    functions = [function for _, function in Descriptors.descList]
    identities: list[str] = []
    rows: list[list[float]] = []
    for identity, smiles in sorted(smiles_by_id.items()):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        values: list[float] = []
        for function in functions:
            try:
                value = float(function(molecule))
            except Exception:
                value = np.nan
            values.append(value if np.isfinite(value) else np.nan)
        identities.append(identity)
        rows.append(values)
    return identities, np.asarray(rows, dtype=np.float64), descriptor_names


def fit_predict_rt(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fixed RF recipe; no annotation-outcome tuning is permitted."""
    usable = np.mean(np.isfinite(train_x), axis=0) >= 0.5
    variance = np.nanstd(train_x[:, usable], axis=0) > 0
    selected = np.flatnonzero(usable)[variance]
    if len(selected) < 10:
        raise RuntimeError(f"only {len(selected)} usable RT descriptors")
    imputer = SimpleImputer(strategy="median")
    fitted_train = imputer.fit_transform(train_x[:, selected])
    fitted_test = imputer.transform(test_x[:, selected])
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(fitted_train, train_y)
    return model.predict(fitted_test), {
        "usable_descriptors": int(len(selected)),
        "trees": 500,
        "minimum_leaf": 2,
        "maximum_features": "sqrt",
    }


def rt_score(predicted: float, observed: float, tolerance: float) -> tuple[float, float, bool]:
    relative_error = abs(float(predicted) - float(observed)) / max(abs(float(observed)), 1e-12)
    return relative_error, max(0.0, 1.0 - relative_error / tolerance), relative_error <= tolerance


def conservative_rt_candidate(group: pd.DataFrame, baseline_id: str) -> tuple[str, bool]:
    baseline = group[group["candidate_id"].eq(baseline_id)]
    if len(baseline) != 1:
        raise RuntimeError(f"missing baseline candidate {baseline_id}")
    if bool(baseline.iloc[0].rt_pass):
        return baseline_id, False
    eligible = group[group["rt_pass"]].sort_values(
        ["spectral_score", "candidate_id"], ascending=[False, True]
    )
    if eligible.empty:
        return baseline_id, False
    best_score = float(eligible.iloc[0].spectral_score)
    winners = eligible[np.isclose(eligible["spectral_score"], best_score, atol=1e-12)]
    if len(winners) != 1:
        return baseline_id, False
    winner = str(winners.iloc[0].candidate_id)
    return winner, winner != baseline_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--development-dir", type=Path, default=Path("data/validation/bioaware_metdna3_development_v1"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"))
    parser.add_argument("--candidate-scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--baseline-transitions", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"))
    parser.add_argument("--mrn-decision-dir", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_decision_v1"))
    parser.add_argument("--smn-dir", type=Path, default=Path("data/validation/bioaware_metdna3_smn_headroom_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_rt_headroom_v1"))
    parser.add_argument("--relative-tolerance", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()

    files = {
        "truth": args.development_dir / args.truth_name,
        "splits": args.development_dir / "identity_splits.csv.gz",
        "queries": args.query_cache,
        "candidate_scores": args.candidate_scores,
        "baseline": args.baseline_transitions,
        "mrn_queries": args.mrn_decision_dir / "query_transitions.csv.gz",
        "smn_queries": args.smn_dir / "query_transitions.csv.gz",
        "hdf5": args.hdf5,
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    truth = pd.read_csv(files["truth"])
    splits = pd.read_csv(files["splits"])
    queries = pd.read_csv(files["queries"])
    candidates = pd.read_csv(files["candidate_scores"])
    baseline_frame = pd.read_csv(files["baseline"])
    expected_queries = int(queries.query_id.nunique())
    if candidates["query_id"].nunique() != expected_queries:
        raise RuntimeError("frozen candidate query coverage changed")
    baseline_by_query = baseline_frame.groupby("query_id")["baseline_top_candidate"].first()
    query_meta = queries.set_index("query_id")

    target_ids = set(candidates["candidate_id"].astype(str))
    level1_ids = set(truth["ik14"].astype(str))
    smiles_by_id = load_smiles_by_ik14(files["hdf5"], target_ids | level1_ids, truth)
    identities, matrix, descriptor_names = descriptor_matrix(smiles_by_id)
    matrix_by_id = {identity: matrix[index] for index, identity in enumerate(identities)}
    level1_rt = truth.groupby("ik14")["rt"].median().to_dict()

    candidate_rows: list[dict] = []
    fold_rows: list[dict] = []
    validation_rows: list[dict] = []
    recipe: dict | None = None
    for fold in range(10):
        seed_ids = set(
            splits[(splits["fold"].eq(fold)) & splits["role"].eq("seed")]["ik14"].astype(str)
        ) & set(matrix_by_id) & set(level1_rt)
        heldout_ids = set(
            splits[(splits["fold"].eq(fold)) & splits["role"].eq("heldout")]["ik14"].astype(str)
        )
        train_ids = sorted(seed_ids)
        predict_ids = sorted((target_ids | (heldout_ids & level1_ids)) & set(matrix_by_id))
        train_x = np.stack([matrix_by_id[item] for item in train_ids])
        train_y = np.asarray([level1_rt[item] for item in train_ids], dtype=float)
        test_x = np.stack([matrix_by_id[item] for item in predict_ids])
        predictions, recipe = fit_predict_rt(train_x, train_y, test_x, args.seed + fold)
        predicted = dict(zip(predict_ids, predictions, strict=True))
        for identity in sorted(heldout_ids & set(level1_rt) & set(predicted)):
            validation_rows.append({
                "fold": fold,
                "ik14": identity,
                "observed_rt": float(level1_rt[identity]),
                "predicted_rt": float(predicted[identity]),
                "absolute_error": abs(float(predicted[identity]) - float(level1_rt[identity])),
            })
        fold_candidates = candidates[candidates["truth_candidate_id"].isin(heldout_ids)]
        for query_id, group in fold_candidates.groupby("query_id", sort=True):
            observed_rt = float(query_meta.loc[str(query_id)].feature_rt_sec)
            frozen_rows: list[dict] = []
            for row in group.itertuples(index=False):
                candidate = str(row.candidate_id)
                if candidate in predicted:
                    error, score, passed = rt_score(
                        predicted[candidate], observed_rt, args.relative_tolerance
                    )
                    predicted_rt = float(predicted[candidate])
                else:
                    error, score, passed, predicted_rt = np.nan, 0.0, False, np.nan
                record = {
                    "fold": fold,
                    "query_id": str(query_id),
                    "candidate_id": candidate,
                    "truth_candidate_id": str(row.truth_candidate_id),
                    "truth_formula": str(row.truth_formula),
                    "spectral_score": float(row.spectral_score),
                    "observed_rt": observed_rt,
                    "predicted_rt": predicted_rt,
                    "relative_rt_error": error,
                    "rt_score": score,
                    "rt_pass": bool(passed),
                }
                candidate_rows.append(record)
                frozen_rows.append(record)
            frozen = pd.DataFrame(frozen_rows)
            baseline_id = str(baseline_by_query.loc[str(query_id)])
            truth_id = str(frozen["truth_candidate_id"].iloc[0])
            final_id, intervene = conservative_rt_candidate(frozen, baseline_id)
            truth_row = frozen[frozen["candidate_id"].eq(truth_id)]
            baseline_row = frozen[frozen["candidate_id"].eq(baseline_id)]
            if len(truth_row) != 1 or len(baseline_row) != 1:
                raise RuntimeError(f"candidate contract failed for {query_id} fold {fold}")
            truth_row, baseline_row = truth_row.iloc[0], baseline_row.iloc[0]
            fold_rows.append({
                "fold": fold,
                "query_id": str(query_id),
                "truth_candidate_id": truth_id,
                "truth_formula": str(truth_row.truth_formula),
                "baseline_candidate_id": baseline_id,
                "network_candidate_id": final_id,
                "intervene": bool(intervene),
                "baseline_correct": baseline_id == truth_id,
                "network_correct": final_id == truth_id,
                "truth_strict_advantage": (
                    bool(truth_row.rt_pass) and not bool(baseline_row.rt_pass)
                ),
                "truth_rt_score_advantage": float(truth_row.rt_score - baseline_row.rt_score),
            })

    candidate_frame = pd.DataFrame(candidate_rows)
    fold_frame = pd.DataFrame(fold_rows)
    query_frame = aggregate_majority(fold_frame)
    validation = pd.DataFrame(validation_rows)
    if len(query_frame) != expected_queries:
        raise RuntimeError(f"expected {expected_queries} query decisions, got {len(query_frame)}")
    candidate_path = output / "candidate_rt_evidence.csv.gz"
    fold_path = output / "fold_transitions.csv.gz"
    query_path = output / "query_transitions.csv.gz"
    validation_path = output / "heldout_level1_rt_predictions.csv.gz"
    candidate_frame.to_csv(candidate_path, index=False, compression="gzip")
    fold_frame.to_csv(fold_path, index=False, compression="gzip")
    query_frame.to_csv(query_path, index=False, compression="gzip")
    validation.to_csv(validation_path, index=False, compression="gzip")

    mrn = pd.read_csv(files["mrn_queries"])
    mrn = mrn[mrn["maximum_depth"].eq(3)]
    mrn_corrected = set(mrn.loc[(~mrn["baseline_correct"]) & mrn["final_correct"], "query_id"].astype(str))
    smn = pd.read_csv(files["smn_queries"])
    smn_headroom = set(smn.loc[smn["truth_headroom"], "query_id"].astype(str))
    rt_corrected = set(query_frame.loc[query_frame["corrected"], "query_id"].astype(str))
    rt_headroom = set(query_frame.loc[query_frame["truth_headroom"], "query_id"].astype(str))
    required = int(math.ceil(0.10 * len(query_frame)))
    safe_union = mrn_corrected | rt_corrected
    optimistic_union = mrn_corrected | smn_headroom | rt_headroom
    report = {
        "status": "bioaware_metdna3_rt_headroom_complete",
        "formal": True,
        "scope": args.scope,
        "rt_model": {
            "training": "fold-specific Level-1 seed identities only",
            "structure_features": f"RDKit 2D descriptors ({len(descriptor_names)} attempted)",
            "recipe": recipe,
            "relative_rt_tolerance": args.relative_tolerance,
            "tolerance_source": "MetDNA3 published HILIC protocol; not tuned here",
            "heldout_predictions": int(len(validation)),
            "heldout_identities": int(validation["ik14"].nunique()),
            "heldout_mae_sec": float(mean_absolute_error(validation["observed_rt"], validation["predicted_rt"])),
            "heldout_r2": float(r2_score(validation["observed_rt"], validation["predicted_rt"])),
        },
        "development": {
            "queries": int(len(query_frame)),
            "baseline_recall1": float(query_frame["baseline_correct"].mean()),
            "rt_gated_recall1": float(query_frame["final_correct"].mean()),
            "delta_recall1": float(query_frame["final_correct"].mean() - query_frame["baseline_correct"].mean()),
            "corrected": int(query_frame["corrected"].sum()),
            "introduced": int(query_frame["introduced"].sum()),
            "truth_headroom_errors": int(query_frame["truth_headroom"].sum()),
            "formula_cluster_bootstrap": bootstrap_delta(query_frame, args.seed),
        },
        "ten_point_feasibility": {
            "required_net_corrections": required,
            "existing_mrn_raw_ms2_corrected_queries": int(len(mrn_corrected)),
            "rt_corrected_queries": int(len(rt_corrected)),
            "rt_truth_headroom_queries": int(len(rt_headroom)),
            "safe_mrn_plus_rt_union": int(len(safe_union)),
            "optimistic_mrn_smn_rt_union": int(len(optimistic_union)),
            "safe_union_reaches_ten_points": len(safe_union) >= required,
            "optimistic_union_reaches_ten_points": len(optimistic_union) >= required,
        },
        "contracts": {
            "heldout_identity_rt_used_for_training": False,
            "truth_used_to_choose_override": False,
            "outcome_used_to_tune_tolerance": False,
            "P2b_used": False,
            "external_test_opened": False,
        },
        "provenance": {name: sha256(path) for name, path in files.items()},
        "outputs": {
            "candidate_rt_sha256": sha256(candidate_path),
            "fold_transitions_sha256": sha256(fold_path),
            "query_transitions_sha256": sha256(query_path),
            "heldout_rt_predictions_sha256": sha256(validation_path),
        },
        "claim_limit": (
            "Consumed-development identity-isolated RT audit. The RDKit descriptor panel is an "
            "independent implementation, not a bitwise reproduction of MetDNA3's proprietary descriptor table."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Audit whether a clean-visible peak gate can recover A4 action headroom.

This is deliberately a *reachability audit*, not an encoder result.  It uses
frozen contextual fragment tokens from the official encoder and excludes all
candidate-conditioned A4 fields (gradient, role, candidate counts, error arm,
and exact baseline margins) from the model inputs.  Development is restricted
to the discovery formulas.  A single configuration chosen by formula-OOF
performance is then evaluated once on the disjoint confirmation formulas.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


DOSES = (0.25, 0.50, 0.75, 1.00)
SCORES = ("margin", "top1", "hybrid")
COVERAGES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a4-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_a4_exact_peak_scan"),
    )
    parser.add_argument(
        "--token-root", type=Path,
        default=Path("data/validation/official_peak_tokens"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/noise_clean_visible_peak_gate_audit_20260905"),
    )
    parser.add_argument("--trees", type=int, default=192)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_token_index(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    manifests = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("discovery", "confirmation"):
        split_dir = root / split
        manifest = pd.read_csv(split_dir / "manifest.csv")
        manifest = manifest.copy()
        manifest["audit_split"] = split
        manifest["token_row"] = np.arange(len(manifest), dtype=np.int32)
        manifests.append(manifest)
        arrays[split] = {
            "tokens": np.load(split_dir / "peak_tokens_f16.npy", mmap_mode="r"),
            "mz": np.load(split_dir / "peak_mz.npy", mmap_mode="r"),
            "intensity": np.load(split_dir / "peak_intensity.npy", mmap_mode="r"),
            "valid": np.load(split_dir / "peak_valid.npy", mmap_mode="r"),
        }
    manifest = pd.concat(manifests, ignore_index=True)
    if manifest["hdf5_row"].duplicated().any():
        raise RuntimeError("token manifest contains duplicate HDF5 rows")
    return manifest, arrays


def build_table(a4_dir: Path, token_root: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    queries = pd.read_csv(a4_dir / "scan_queries.csv.gz")
    with h5py.File(a4_dir / "exact_peak_scan.h5", "r") as handle:
        doses = np.asarray(json.loads(handle.attrs["attenuations_json"]), dtype=np.float32)
        if not np.allclose(doses, DOSES):
            raise RuntimeError(f"unexpected A4 attenuation grid: {doses.tolist()}")
        n_actions = len(handle["action_query"])
        action = pd.DataFrame({
            "action_index": np.repeat(np.arange(n_actions, dtype=np.int64), len(doses)),
            "scan_position": np.repeat(handle["action_query"][:], len(doses)),
            "token": np.repeat(handle["action_token"][:], len(doses)),
            "mz": np.repeat(handle["action_mz"][:], len(doses)),
            "intensity": np.repeat(handle["action_intensity"][:], len(doses)),
            "attenuation": np.tile(doses, n_actions),
            "result_rank": handle["result_rank"][:],
            "result_margin": handle["result_margin"][:],
        })
    # Unlike policy_candidate_actions.csv.gz, this is the complete exact action
    # table.  The policy CSV is forbidden here because its top-50 truncation is
    # sorted by the true correction and margin outcome.
    action = action.merge(
        queries[[
            "scan_position", "query_index", "query_row", "query_formula",
            "scan_kind", "baseline_rank", "baseline_margin",
        ]],
        on="scan_position", how="left", validate="many_to_one",
    )
    action["margin_change"] = action.result_margin - action.baseline_margin
    action["corrected"] = action.scan_kind.eq("official_error") & action.result_rank.eq(1)
    action["introduced"] = action.scan_kind.eq("safety_control") & action.result_rank.gt(1)
    manifest, arrays = load_token_index(token_root)
    query = queries.merge(
        manifest[["hdf5_row", "audit_split", "token_row"]],
        left_on="query_row", right_on="hdf5_row", how="inner", validate="one_to_one",
    )
    if set(query.loc[query.audit_split.eq("discovery"), "query_formula"]) & set(
        query.loc[query.audit_split.eq("confirmation"), "query_formula"]
    ):
        raise RuntimeError("discovery and confirmation formulas overlap")
    action = action.merge(
        query[["query_index", "audit_split", "token_row"]],
        on="query_index", how="inner", validate="many_to_one",
    )
    # A4 token indices include the precursor token at zero.  The saved fragment
    # tensor has already removed it, hence token-1 below.
    fragment_index = action["token"].to_numpy(np.int64) - 1
    if np.any((fragment_index < 0) | (fragment_index >= 100)):
        raise RuntimeError("A4 token index is outside the frozen fragment tensor")
    features = np.empty((len(action), 70), dtype=np.float32)
    for split in ("discovery", "confirmation"):
        take = np.flatnonzero(action["audit_split"].to_numpy() == split)
        row = action.iloc[take]["token_row"].to_numpy(np.int64)
        peak = fragment_index[take]
        valid = arrays[split]["valid"][row, peak]
        if not np.all(valid):
            raise RuntimeError(f"{split} contains A4 actions on invalid frozen peaks")
        token = np.asarray(arrays[split]["tokens"][row, peak], dtype=np.float32)
        mz = np.asarray(arrays[split]["mz"][row, peak], dtype=np.float32)
        intensity = np.asarray(arrays[split]["intensity"][row, peak], dtype=np.float32)
        features[take, :64] = token
        features[take, 64] = mz / 1000.0
        features[take, 65] = np.log1p(mz) / 8.0
        features[take, 66] = intensity
        features[take, 67] = np.log(np.clip(intensity, 1e-7, None)) / 16.0
        features[take, 68] = fragment_index[take] / 100.0
        features[take, 69] = action.iloc[take]["attenuation"].to_numpy(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite clean-visible features")
    names = [f"context_token_{index:02d}" for index in range(64)] + [
        "mz_scaled", "log_mz_scaled", "intensity", "log_intensity_scaled",
        "token_position_scaled", "attenuation",
    ]
    # These are retained only for evaluation.  None is present in `features`.
    action["corrected"] = action["corrected"].astype(bool)
    action["introduced"] = action["introduced"].astype(bool)
    return action.reset_index(drop=True), features, names


def fit_model(
    x: np.ndarray, y: np.ndarray, query_index: np.ndarray, args: argparse.Namespace,
) -> ExtraTreesRegressor:
    counts = pd.Series(query_index).value_counts()
    weight = np.asarray([1.0 / counts[int(value)] for value in query_index], dtype=np.float64)
    weight *= len(weight) / weight.sum()
    model = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=0.75,
        bootstrap=False,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def score_predictions(prediction: np.ndarray, margin_scale: float, kind: str) -> np.ndarray:
    margin, correction, harm = prediction.T
    if kind == "margin":
        return margin
    if kind == "top1":
        return correction - 2.0 * harm
    if kind == "hybrid":
        return correction - 2.0 * harm + 0.20 * margin / margin_scale
    raise ValueError(kind)


def select_one(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    work = frame[[
        "query_index", "query_formula", "audit_split", "scan_kind", "baseline_rank",
        "token", "attenuation", "result_rank", "margin_change", "corrected", "introduced",
    ]].copy()
    work["gate_score"] = score
    # No exact outcome is used in tie breaking.
    work = work.sort_values(
        ["query_index", "gate_score", "token"], ascending=[True, False, True],
        kind="mergesort",
    )
    return work.drop_duplicates("query_index", keep="first").reset_index(drop=True)


def cluster_ci(
    selected: pd.DataFrame, applied: np.ndarray, resamples: int, seed: int,
) -> dict[str, float]:
    value = applied * (
        selected["corrected"].to_numpy(np.float64)
        - selected["introduced"].to_numpy(np.float64)
    )
    grouped = pd.DataFrame({
        "formula": selected["query_formula"].astype(str), "value": value,
    }).groupby("formula")["value"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=np.float64)
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    for index in range(resamples):
        take = rng.integers(0, len(grouped), len(grouped))
        boot[index] = 100.0 * sums[take].sum() / counts[take].sum()
    return {
        "delta_pp": float(100.0 * value.sum() / len(selected)),
        "ci_low_pp": float(np.quantile(boot, 0.025)),
        "ci_high_pp": float(np.quantile(boot, 0.975)),
    }


def evaluate_grid(
    selected: pd.DataFrame, score_name: str, dose: float, args: argparse.Namespace,
) -> list[dict]:
    order = np.lexsort((
        selected["query_index"].to_numpy(np.int64),
        -selected["gate_score"].to_numpy(float),
    ))
    records = []
    for coverage in COVERAGES:
        count = max(1, int(math.ceil(coverage * len(selected))))
        applied = np.zeros(len(selected), dtype=bool)
        applied[order[:count]] = True
        corrected = int((applied & selected.corrected.to_numpy(bool)).sum())
        introduced = int((applied & selected.introduced.to_numpy(bool)).sum())
        ci = cluster_ci(selected, applied, args.bootstrap_resamples, args.seed + count)
        records.append({
            "dose": dose, "score": score_name, "coverage": coverage,
            "queries": int(len(selected)), "interventions": count,
            "corrected": corrected, "introduced": introduced,
            "risk_net_lambda2": corrected - 2 * introduced,
            **ci,
        })
    return records


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.folds < 2 or args.trees < 1 or args.min_samples_leaf < 1:
        raise ValueError("invalid model parameters")
    frame, features, feature_names = build_table(args.a4_dir, args.token_root)
    discovery = frame.audit_split.eq("discovery").to_numpy()
    confirmation = frame.audit_split.eq("confirmation").to_numpy()
    y = np.column_stack((
        frame.margin_change.to_numpy(np.float32),
        frame.corrected.to_numpy(np.float32),
        frame.introduced.to_numpy(np.float32),
    ))
    margin_scale = float(max(np.quantile(np.abs(y[discovery, 0]), 0.90), 1e-4))
    oof_prediction = np.full((len(frame), 3), np.nan, dtype=np.float32)
    discovery_rows = np.flatnonzero(discovery)
    group = frame.loc[discovery, "query_formula"].astype(str).to_numpy()
    splitter = GroupKFold(args.folds)
    for fold, (train_local, test_local) in enumerate(splitter.split(discovery_rows, groups=group)):
        train = discovery_rows[train_local]
        test = discovery_rows[test_local]
        model = fit_model(
            features[train], y[train], frame.iloc[train].query_index.to_numpy(), args,
        )
        oof_prediction[test] = model.predict(features[test]).astype(np.float32)
        print(f"[clean-visible gate] discovery fold {fold + 1}/{args.folds}", flush=True)
    if not np.isfinite(oof_prediction[discovery]).all():
        raise RuntimeError("incomplete discovery OOF predictions")

    development_records = []
    development_selected: dict[tuple[float, str], pd.DataFrame] = {}
    for dose in DOSES:
        take = discovery & np.isclose(frame.attenuation.to_numpy(float), dose)
        for score_name in SCORES:
            score = score_predictions(oof_prediction[take], margin_scale, score_name)
            selected = select_one(frame.loc[take].reset_index(drop=True), score)
            development_selected[(dose, score_name)] = selected
            development_records.extend(evaluate_grid(selected, score_name, dose, args))
    development = pd.DataFrame(development_records)
    # Selection is confined to discovery OOF results.  Prefer risk-adjusted net,
    # then raw top-1 delta, lower CI, and lower intervention coverage.
    choice_row = development.sort_values(
        ["risk_net_lambda2", "delta_pp", "ci_low_pp", "coverage"],
        ascending=[False, False, False, True],
    ).iloc[0]
    choice = {
        "dose": float(choice_row.dose), "score": str(choice_row.score),
        "coverage": float(choice_row.coverage),
    }

    train = np.flatnonzero(discovery & np.isclose(frame.attenuation.to_numpy(float), choice["dose"]))
    test = np.flatnonzero(confirmation & np.isclose(frame.attenuation.to_numpy(float), choice["dose"]))
    final_model = fit_model(
        features[train], y[train], frame.iloc[train].query_index.to_numpy(), args,
    )
    confirmation_prediction = final_model.predict(features[test]).astype(np.float32)
    confirmation_score = score_predictions(confirmation_prediction, margin_scale, choice["score"])
    confirmation_selected = select_one(frame.iloc[test].reset_index(drop=True), confirmation_score)
    confirmation_result = evaluate_grid(
        confirmation_selected, choice["score"], choice["dose"], args,
    )
    confirmation_result = next(
        item for item in confirmation_result if item["coverage"] == choice["coverage"]
    )

    def cohort_summary(mask: np.ndarray) -> dict[str, int]:
        local = frame.loc[mask].drop_duplicates("query_index")
        return {
            "queries": int(len(local)),
            "formulas": int(local.query_formula.nunique()),
            "official_errors": int(local.scan_kind.eq("official_error").sum()),
            "safety_controls": int(local.scan_kind.eq("safety_control").sum()),
        }

    report = {
        "status": "clean_visible_peak_gate_reachability_audit",
        "claim_limit": (
            "This is formula-held action-selection evidence on a frozen A4 subset, "
            "not a trained shared-encoder retrieval improvement."
        ),
        "excluded_privileged_inputs": [
            "gradient", "predicted_gain", "role", "policy_eligible",
            "candidate_molecules", "baseline_rank", "baseline_margin", "scan_kind",
        ],
        "feature_names": feature_names,
        "margin_scale": margin_scale,
        "discovery": cohort_summary(discovery),
        "confirmation": cohort_summary(confirmation),
        "development_choice": choice,
        "development_choice_metrics": choice_row.to_dict(),
        "confirmation_once": confirmation_result,
        "confirmation_pass_4pp_raw": bool(confirmation_result["delta_pp"] >= 4.0),
        "confirmation_pass_positive_formula_ci": bool(confirmation_result["ci_low_pp"] > 0.0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    development.to_csv(args.output_dir / "discovery_oof_grid.csv", index=False)
    development_selected[(choice["dose"], choice["score"])].to_csv(
        args.output_dir / "discovery_oof_selected_actions.csv", index=False,
    )
    confirmation_selected.to_csv(
        args.output_dir / "confirmation_selected_actions.csv", index=False,
    )
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

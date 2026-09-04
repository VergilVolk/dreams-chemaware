"""Build the formula-crossfit 30-cell N/P ledger for direct fine-tuning.

The output contains executable action recipes and continuous training weights,
not a downstream selector.  Outer-held formulas are removed before any P
outcome model is fit.  Historical P outcomes are used only to generate
formula-OOF clean-visible probabilities on outer-train; raw outcome columns are
never published in the training ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from noise_final_dynamic_direct_core import (
    WeightConfig, build_action_weights, formula_equal_weights,
    stable_control_index, validate_n_cells,
)


ROOT = Path(__file__).resolve().parents[1]
P_INTENSITY_FAMILIES = (
    "matched_intensity_transport", "prevalence_attenuation", "consensus_projection",
)
P_TRANSFER_FAMILIES = (
    "recurrent_peak_graft", "balanced_peak_exchange", "recurrent_union_mix",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation = ROOT / "data/validation"
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=validation / "g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embedding-cache", type=Path, default=validation / "g8r_p2_official_embeddings.npz")
    parser.add_argument("--r0-dir", type=Path, default=validation / "g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--l0-dir", type=Path, default=validation / "g8r_noise_final_l0_action_learnability_ledger")
    parser.add_argument("--l1-dir", type=Path, default=validation / "g8r_noise_final_l1_clean_action_learnability")
    parser.add_argument("--token-dir", type=Path, default=validation / "g8r_noise_final_f1_full_tokens")
    parser.add_argument("--p-intensity-dir", type=Path, default=validation / "g8r_noise_final_positive_guided_matrix")
    parser.add_argument("--p-transfer-dir", type=Path, default=validation / "g8r_noise_final_positive_peak_transfer")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--embedding-projection-dim", type=int, default=32)
    parser.add_argument("--token-projection-dim", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection(source: int, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((source, target)) / np.sqrt(target)).astype(np.float32)


def clean_query_features(args: argparse.Namespace, query_rows: np.ndarray) -> np.ndarray:
    with np.load(args.embedding_cache) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    index = {int(row): position for position, row in enumerate(rows)}
    positions = np.asarray([index[int(row)] for row in query_rows], dtype=np.int64)
    embedding = embeddings[positions]
    embedding /= np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12, None)
    embedding_feature = embedding @ projection(embedding.shape[1], args.embedding_projection_dim, args.seed)

    token_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    token_index = {int(row): position for position, row in enumerate(token_rows)}
    token_positions = np.asarray([token_index[int(row)] for row in query_rows], dtype=np.int64)
    tokens = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
    mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    valid = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    token_projection = projection(tokens.shape[2], args.token_projection_dim, args.seed + 1)
    blocks: list[np.ndarray] = []
    for left in range(0, len(query_rows), 512):
        pos = token_positions[left:left + 512]
        mask = np.asarray(valid[pos], dtype=bool)
        contextual = np.asarray(tokens[pos], dtype=np.float32) @ token_projection
        count = np.clip(mask.sum(axis=1, keepdims=True), 1, None).astype(np.float32)
        masked = contextual * mask[..., None]
        mean = masked.sum(axis=1) / count
        variance = ((contextual - mean[:, None, :]) ** 2 * mask[..., None]).sum(axis=1) / count
        maximum = np.where(mask[..., None], contextual, -np.inf).max(axis=1)
        maximum[~np.isfinite(maximum)] = 0.0
        intensity_block = np.asarray(intensity[pos], dtype=np.float32) * mask
        weight = intensity_block / np.clip(intensity_block.sum(axis=1, keepdims=True), 1e-12, None)
        weighted = (contextual * weight[..., None]).sum(axis=1)
        mz_block = np.asarray(mz[pos], dtype=np.float32)
        mz_mean = (mz_block * mask).sum(axis=1, keepdims=True) / count
        mz_var = (((mz_block - mz_mean) ** 2) * mask).sum(axis=1, keepdims=True) / count
        probability = weight
        entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, None)), axis=1, keepdims=True)
        raw = np.concatenate([
            count / mask.shape[1], mz_mean / 1000.0, np.sqrt(mz_var) / 1000.0,
            intensity_block.mean(axis=1, keepdims=True),
            intensity_block.max(axis=1, keepdims=True), entropy,
        ], axis=1)
        blocks.append(np.concatenate([mean, np.sqrt(variance), maximum, weighted, raw], axis=1))
        right = min(left + 512, len(query_rows))
        if right == len(query_rows) or right % 4096 == 0:
            print(f"[dynamic ledger clean features] {right:,}/{len(query_rows):,}", flush=True)
    token_feature = np.concatenate(blocks, axis=0).astype(np.float32)
    features = np.concatenate([embedding_feature, token_feature], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("clean query feature matrix contains non-finite values")
    return features


def fit_crossfit(
    x: np.ndarray,
    formulas: np.ndarray,
    folds: np.ndarray,
    gain: np.ndarray,
    positive: np.ndarray,
    harmful: np.ndarray,
    outer_fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    pred_gain = np.full(len(x), np.nan, dtype=np.float32)
    p_positive = np.full(len(x), np.nan, dtype=np.float32)
    p_harmful = np.full(len(x), np.nan, dtype=np.float32)
    for fold in sorted(set(map(int, folds)) - {outer_fold}):
        test = folds == fold
        train = (folds != fold) & (folds != outer_fold)
        if not np.any(test) or len(set(formulas[train])) < 50:
            raise RuntimeError(f"P crossfit fold {fold} has insufficient formula support")
        weights = formula_equal_weights(formulas[train])
        common = dict(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=1.0, random_state=seed + fold,
        )
        regressor = HistGradientBoostingRegressor(**common).fit(x[train], gain[train], sample_weight=weights)
        positive_model = HistGradientBoostingClassifier(**common).fit(
            x[train], positive[train].astype(np.int8), sample_weight=weights,
        )
        harmful_model = HistGradientBoostingClassifier(**common).fit(
            x[train], harmful[train].astype(np.int8), sample_weight=weights,
        )
        pred_gain[test] = regressor.predict(x[test]).astype(np.float32)
        p_positive[test] = positive_model.predict_proba(x[test])[:, 1].astype(np.float32)
        p_harmful[test] = harmful_model.predict_proba(x[test])[:, 1].astype(np.float32)
        print(
            f"[dynamic ledger formula-crossfit] fold={fold} "
            f"train={int(train.sum()):,} test={int(test.sum()):,}",
            flush=True,
        )
    active = folds != outer_fold
    if not np.isfinite(np.column_stack([pred_gain[active], p_positive[active], p_harmful[active]])).all():
        raise RuntimeError("P formula-crossfit predictions are incomplete")
    clipped_positive = np.clip(p_positive, 1e-4, 1 - 1e-4)
    clipped_harmful = np.clip(p_harmful, 1e-4, 1 - 1e-4)
    metrics = {
        "actions": int(active.sum()),
        "formulas": int(len(set(formulas[active]))),
        "positive_prevalence": float(np.mean(positive[active])),
        "harmful_prevalence": float(np.mean(harmful[active])),
        "positive_auprc": float(average_precision_score(positive[active], clipped_positive[active])),
        "harmful_auprc": float(average_precision_score(harmful[active], clipped_harmful[active])),
        "gain_pearson": float(np.corrcoef(gain[active], pred_gain[active])[0, 1]),
    }
    return pred_gain, clipped_positive, clipped_harmful, metrics


def p_action_rows(
    args: argparse.Namespace,
    query_rows: np.ndarray,
    query_ik14: np.ndarray,
    query_formula: np.ndarray,
    formula_fold: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = pd.read_csv(args.p_intensity_dir / "action_manifest.csv.gz", low_memory=False)
    transfer = pd.read_csv(args.p_transfer_dir / "action_manifest.csv.gz", low_memory=False)
    base = base.sort_values("query_index", kind="stable").reset_index(drop=True)
    transfer = transfer.sort_values("query_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(base["query_row"].to_numpy(np.int64), query_rows):
        raise RuntimeError("P manifest and graph query rows differ")
    features_by_query = clean_query_features(args, query_rows)
    outer_train_query = np.flatnonzero(formula_fold != args.outer_fold)
    frames: list[pd.DataFrame] = []
    branch_reports: dict[str, object] = {}

    with h5py.File(args.p_intensity_dir / "matrix_results.h5", "r") as handle:
        families = tuple(json.loads(handle.attrs["families_json"]))
        doses = tuple(map(float, json.loads(handle.attrs["doses_json"])))
        if families != P_INTENSITY_FAMILIES:
            raise RuntimeError(f"P intensity families drifted: {families}")
        result_rank = np.asarray(handle["result_rank"], dtype=np.int16)
        result_margin = np.asarray(handle["result_margin"], dtype=np.float32)
    intensity_rows: list[dict[str, object]] = []
    intensity_x: list[np.ndarray] = []
    intensity_gain: list[float] = []
    intensity_positive: list[bool] = []
    intensity_harmful: list[bool] = []
    for query in outer_train_query:
        for family_index, family in enumerate(families):
            for dose_index, dose in enumerate(doses):
                target_cell = (family_index * len(doses) + dose_index) * 2
                control_cell = target_cell + 1
                gain = float(result_margin[query, target_cell] - result_margin[query, control_cell])
                top1 = int(result_rank[query, target_cell] == 1) - int(result_rank[query, control_cell] == 1)
                descriptor = np.zeros(len(families) + 2, dtype=np.float32)
                descriptor[family_index] = 1.0; descriptor[-2] = dose; descriptor[-1] = 0.0
                intensity_x.append(np.concatenate([features_by_query[query], descriptor]))
                intensity_gain.append(gain)
                intensity_positive.append(top1 > 0 or gain >= 0.01)
                intensity_harmful.append(top1 < 0 or gain <= -0.01)
                intensity_rows.append({
                    "query_index": int(query), "query_row": int(query_rows[query]),
                    "identity": str(query_ik14[query]), "formula": str(query_formula[query]),
                    "formula_fold": int(formula_fold[query]), "source": "P_intensity",
                    "family": f"P:{family}", "cell_id": f"{family}|dose={dose:.2f}",
                    "payload_kind": "positive_intensity", "dose": dose,
                    "target_payload": str(base.iloc[query]["positive_reference_rows"]),
                    "control_payload": str(base.iloc[query]["hardest_wrong_reference_rows"]),
                    "evidence_availability": "observed",
                })
        if (query + 1) % 5000 == 0:
            print(f"[dynamic ledger P-intensity recipes] query={query + 1:,}", flush=True)
    intensity_frame = pd.DataFrame(intensity_rows)
    arrays = fit_crossfit(
        np.asarray(intensity_x, dtype=np.float32), intensity_frame["formula"].to_numpy(str),
        intensity_frame["formula_fold"].to_numpy(np.int8), np.asarray(intensity_gain),
        np.asarray(intensity_positive), np.asarray(intensity_harmful), args.outer_fold, args.seed + 10,
    )
    intensity_frame["lagged_advantage"] = np.asarray(intensity_gain, dtype=np.float32)
    intensity_frame["p_clean"] = arrays[1]; intensity_frame["risk"] = arrays[2]
    frames.append(intensity_frame); branch_reports["P_intensity"] = arrays[3]

    with h5py.File(args.p_transfer_dir / "matrix_results.h5", "r") as handle:
        families = tuple(json.loads(handle.attrs["families_json"]))
        doses = tuple(map(float, json.loads(handle.attrs["doses_json"])))
        if families != P_TRANSFER_FAMILIES:
            raise RuntimeError(f"P transfer families drifted: {families}")
        result_rank = np.asarray(handle["result_rank"], dtype=np.int16)
    transfer_rows: list[dict[str, object]] = []
    transfer_x: list[np.ndarray] = []
    transfer_gain: list[float] = []
    transfer_positive: list[bool] = []
    transfer_harmful: list[bool] = []
    for query in outer_train_query:
        missing_count = int(transfer.iloc[query]["positive_missing_peak_count"])
        if missing_count <= 0:
            continue
        for family_index, family in enumerate(families):
            for dose_index, dose in enumerate(doses):
                target_cell = (family_index * len(doses) + dose_index) * 2
                control_cell = target_cell + 1
                signed_top1 = int(result_rank[query, target_cell] == 1) - int(result_rank[query, control_cell] == 1)
                descriptor = np.zeros(len(families) + 2, dtype=np.float32)
                descriptor[family_index] = 1.0; descriptor[-2] = dose
                descriptor[-1] = min(np.log1p(missing_count) / np.log(101.0), 1.0)
                transfer_x.append(np.concatenate([features_by_query[query], descriptor]))
                transfer_gain.append(0.05 * signed_top1)
                transfer_positive.append(signed_top1 > 0)
                transfer_harmful.append(signed_top1 < 0)
                transfer_rows.append({
                    "query_index": int(query), "query_row": int(query_rows[query]),
                    "identity": str(query_ik14[query]), "formula": str(query_formula[query]),
                    "formula_fold": int(formula_fold[query]), "source": "P_transfer",
                    "family": f"P:{family}", "cell_id": f"{family}|dose={dose:.2f}",
                    "payload_kind": "positive_peak_transfer", "dose": dose,
                    "target_payload": str(base.iloc[query]["positive_reference_rows"]),
                    "control_payload": str(base.iloc[query]["hardest_wrong_reference_rows"]),
                    "evidence_availability": "predictable_missing",
                    "positive_missing_peak_count": missing_count,
                })
        if (query + 1) % 5000 == 0:
            print(f"[dynamic ledger P-transfer recipes] query={query + 1:,}", flush=True)
    transfer_frame = pd.DataFrame(transfer_rows)
    arrays = fit_crossfit(
        np.asarray(transfer_x, dtype=np.float32), transfer_frame["formula"].to_numpy(str),
        transfer_frame["formula_fold"].to_numpy(np.int8), np.asarray(transfer_gain),
        np.asarray(transfer_positive), np.asarray(transfer_harmful), args.outer_fold, args.seed + 20,
    )
    # No continuous margin was stored by the historical transfer matrix.  Do
    # not invent one: epoch-0 uses the neutral sigmoid value and online replay
    # supplies the first current-geometry margin advantage for epoch 1.
    transfer_frame["lagged_advantage"] = 0.0
    transfer_frame["p_clean"] = arrays[1]; transfer_frame["risk"] = arrays[2]
    frames.append(transfer_frame); branch_reports["P_transfer"] = arrays[3]
    return pd.concat(frames, ignore_index=True), branch_reports


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be in 0..4")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite dynamic action ledger: {args.output_dir}")
    preflight = json.loads((args.preflight_dir / "report.json").read_text(encoding="utf-8"))
    if (preflight.get("status") != "noise_final_dynamic_direct_preflight_complete"
            or not preflight.get("pass_to_gpu_replay")
            or int(preflight.get("outer_formula_fold", -1)) != args.outer_fold):
        raise RuntimeError("ledger requires the matching completed preflight")

    with np.load(args.graph, allow_pickle=True) as body:
        query_rows = np.asarray(body["query_row"], dtype=np.int64)
        query_ik14 = np.asarray(body["query_ik14"], dtype=str)
        query_formula = np.asarray(body["query_formula"], dtype=str)
    formula_fold = np.asarray([
        int.from_bytes(hashlib.sha256(f"20260825|{value}".encode()).digest()[:8], "little") % 5
        for value in query_formula
    ], dtype=np.int8)

    r0 = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    l0 = pd.read_csv(args.l0_dir / "action_labels.csv.gz", low_memory=False)
    l1 = pd.read_csv(args.l1_dir / "action_oof_predictions.csv.gz", low_memory=False)
    validate_n_cells(r0[["selector", "attenuation", "step"]])
    keep = ["query_index", "selector", "attenuation", "step"]
    n = r0.merge(
        l0[keep + ["paired_advantage"]], on=keep, how="inner", validate="one_to_one",
    ).merge(
        l1[keep + ["clean_p_positive", "clean_p_harmful"]],
        on=keep, how="inner", validate="one_to_one",
    )
    if len(n) != len(r0):
        raise RuntimeError("N R0/L0/L1 action join is incomplete")
    n = n.loc[n["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    n["identity"] = n["query_ik14"].astype(str); n["formula"] = n["query_formula"].astype(str)
    n["source"] = "N"; n["family"] = "N:" + n["selector"].astype(str)
    n["cell_id"] = n.apply(
        lambda row: f"{row.selector}|a={float(row.attenuation):.2f}|step={int(row.step)}", axis=1,
    )
    n["payload_kind"] = "negative_attenuation"; n["dose"] = n["attenuation"].astype(float)
    n["target_payload"] = n["target_path"].astype(str)
    n["action_id"] = n.apply(
        lambda row: f"N|{int(row.query_index)}|{row.selector}|{float(row.attenuation):.2f}|{int(row.step)}", axis=1,
    )
    controls: list[str] = []
    for row in n.itertuples(index=False):
        values = str(row.matched_control_paths).split(";")
        if len(values) != 2 or values[0] == values[1]:
            raise RuntimeError("N action lacks two distinct matched controls")
        controls.append(values[stable_control_index(str(row.action_id))])
    n["control_payload"] = controls; n["evidence_availability"] = "observed"
    n["p_clean"] = np.clip(n["clean_p_positive"].to_numpy(float), 1e-4, 1 - 1e-4)
    n["risk"] = np.clip(n["clean_p_harmful"].to_numpy(float), 1e-4, 1 - 1e-4)
    n["lagged_advantage"] = n["paired_advantage"].astype(float)

    p, p_reports = p_action_rows(
        args, query_rows, query_ik14, query_formula, formula_fold,
    )
    p["action_id"] = p.apply(
        lambda row: f"P|{int(row.query_index)}|{row.cell_id}", axis=1,
    )
    columns = [
        "action_id", "query_index", "query_row", "identity", "formula", "formula_fold",
        "source", "family", "cell_id", "payload_kind", "dose", "target_payload",
        "control_payload", "evidence_availability", "p_clean", "lagged_advantage", "risk",
    ]
    actions = pd.concat([n[columns], p[columns]], ignore_index=True)
    if actions["action_id"].duplicated().any():
        raise RuntimeError("unified ledger action ids are not unique")
    held_formulas = set(query_formula[formula_fold == args.outer_fold])
    if set(actions["formula"].astype(str)) & held_formulas:
        raise RuntimeError("outer-held formula leaked into dynamic action ledger")
    dynamic, dynamic_report = build_action_weights(actions, "dynamic", WeightConfig())
    static, static_report = build_action_weights(actions, "static", WeightConfig())
    actions["dynamic_weight"] = dynamic["weight"].to_numpy(np.float32)
    actions["static_weight"] = static["weight"].to_numpy(np.float32)
    if not np.isclose(actions["dynamic_weight"].sum(), actions["static_weight"].sum(), rtol=0.02):
        raise RuntimeError("dynamic/static total target exposure differs by more than 2%")
    cell_counts = actions.groupby(["source", "family", "cell_id"], as_index=False).agg(
        actions=("action_id", "size"), queries=("query_index", "nunique"),
        identities=("identity", "nunique"), formulas=("formula", "nunique"),
        dynamic_weight=("dynamic_weight", "sum"), static_weight=("static_weight", "sum"),
    )
    expected_cells = 30
    if cell_counts["cell_id"].nunique() != expected_cells:
        raise RuntimeError(f"expected all 30 mature N/P cells, observed {cell_counts['cell_id'].nunique()}")
    report = {
        "status": "noise_final_dynamic_direct_action_ledger_complete", "formal": True,
        "outer_formula_fold": int(args.outer_fold), "actions": int(len(actions)),
        "queries": int(actions["query_index"].nunique()), "identities": int(actions["identity"].nunique()),
        "formulas": int(actions["formula"].nunique()), "cells": int(len(cell_counts)),
        "source_counts": actions["source"].value_counts().to_dict(),
        "P_crossfit": p_reports, "dynamic_weight": dynamic_report, "static_weight": static_report,
        "contracts": {
            "all_30_cells_retained": True, "multiple_actions_per_query_retained": True,
            "no_op_implicit_and_always_available": True,
            "outer_held_formulas_absent": True, "raw_P_outcomes_published": False,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            "preflight": sha256_file(args.preflight_dir / "report.json"),
            "r0": sha256_file(args.r0_dir / "training_actions.csv.gz"),
            "l0": sha256_file(args.l0_dir / "action_labels.csv.gz"),
            "l1": sha256_file(args.l1_dir / "action_oof_predictions.csv.gz"),
            "P_intensity": sha256_file(args.p_intensity_dir / "matrix_results.h5"),
            "P_transfer": sha256_file(args.p_transfer_dir / "matrix_results.h5"),
            "script": sha256_file(Path(__file__)),
        },
        "pass_to_gpu_replay": bool(
            dynamic_report["all_family_ess_pass"] and static_report["all_family_ess_pass"]
        ),
    }
    if not report["pass_to_gpu_replay"]:
        raise RuntimeError("unified action ledger failed family ESS")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dynamic_direct_ledger_", dir=args.output_dir.parent))
    try:
        actions.to_csv(temporary / "training_actions.csv.gz", index=False, compression="gzip")
        cell_counts.to_csv(temporary / "cell_summary.csv", index=False)
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

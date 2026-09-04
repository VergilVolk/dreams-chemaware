"""L1: formula-crossfit clean-input learnability of mature noise actions.

The outcome is the L0 full-candidate target-minus-matched-random advantage.
Inputs contain only the unmodified query spectrum, label-free contextual peak
tokens, the mature clean embedding, acquisition metadata, and the fixed action
family/step identifier. Candidate scores, identities, formulas, target paths,
and every L0 outcome are forbidden as features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_l0_action_learnability_ledger import load_clean_model  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows  # noqa: E402


PRIMARY_POLICY = {"positive_probability": 0.70, "harmful_probability": 0.10, "predicted_gain": 0.01}
POLICY_SENSITIVITY = {
    "moderate": {"positive_probability": 0.60, "harmful_probability": 0.15, "predicted_gain": 0.005},
    "primary": PRIMARY_POLICY,
    "strict": {"positive_probability": 0.80, "harmful_probability": 0.05, "predicted_gain": 0.015},
}
METADATA_KEYS = (
    "instrument_type", "instrument", "collision_energy", "collision_energy_normed",
    "adduct", "precursor_type", "ion_mode",
)
RAW_FEATURE_NAMES = (
    "precursor_mz", "peak_count", "peak_fraction", "mz_mean", "mz_std",
    "mz_q10", "mz_q25", "mz_q50", "mz_q75", "mz_q90",
    "intensity_mean", "intensity_std", "intensity_q10", "intensity_q25",
    "intensity_q50", "intensity_q75", "intensity_q90", "intensity_entropy",
    "top1_intensity", "top5_intensity_share", "neutral_loss_mean",
    "neutral_loss_std", "neutral_loss_q25", "neutral_loss_q50", "neutral_loss_q75",
    "positive_neutral_loss_fraction", *(f"peak_pair_diff_bin_{index}" for index in range(10)),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_l0_action_learnability_ledger")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=(ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution/"
                 "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
                 "e4a_causal_v1_20260901_causal_clean_duplicate/"
                 "seed_20260828/fold_0/final_shared_encoder.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_l1_clean_action_learnability")
    parser.add_argument("--embedding-projection-dim", type=int, default=64)
    parser.add_argument("--token-projection-dim", type=int, default=16)
    parser.add_argument("--metadata-hash-bins", type=int, default=8)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def stable_hash(value: object, bins: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % bins


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return "|".join(decode(item) for item in value.reshape(-1))
    return str(value)


def random_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    if input_dim < 1 or output_dim < 1:
        raise ValueError("invalid random-projection dimensions")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(input_dim, output_dim)).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
    return matrix


def spectrum_descriptors(mz: np.ndarray, intensity: np.ndarray, precursor: float) -> np.ndarray:
    valid = (np.asarray(mz) > 0) & (np.asarray(intensity) > 0)
    mz = np.asarray(mz, dtype=np.float64)[valid]
    intensity = np.asarray(intensity, dtype=np.float64)[valid]
    if not len(mz):
        return np.zeros(36, dtype=np.float32)
    intensity = intensity / max(float(intensity.max()), 1e-12)
    probability = intensity / max(float(intensity.sum()), 1e-12)
    entropy = -float(np.sum(probability * np.log(np.clip(probability, 1e-12, None))))
    top = np.sort(intensity)[::-1]
    losses = float(precursor) - mz
    positive_losses = losses[losses > 0]
    if len(positive_losses):
        loss_stats = [float(positive_losses.mean()), float(positive_losses.std()),
                      *np.quantile(positive_losses, [0.25, 0.50, 0.75]).tolist(),
                      float(len(positive_losses) / len(losses))]
    else:
        loss_stats = [0.0] * 6
    top_indices = np.argsort(-intensity, kind="stable")[:20]
    top_mz = mz[top_indices]
    differences = np.abs(top_mz[:, None] - top_mz[None, :])
    differences = differences[np.triu_indices(len(top_mz), k=1)]
    pair_hist = np.histogram(differences, bins=np.linspace(0, 100, 11))[0].astype(np.float64)
    pair_hist /= max(float(pair_hist.sum()), 1.0)
    features = np.asarray([
        float(precursor), float(len(mz)), float(len(mz) / len(valid)),
        float(mz.mean()), float(mz.std()), *np.quantile(mz, [0.10, 0.25, 0.50, 0.75, 0.90]),
        float(intensity.mean()), float(intensity.std()),
        *np.quantile(intensity, [0.10, 0.25, 0.50, 0.75, 0.90]),
        entropy, float(top[0]), float(top[: min(5, len(top))].sum() / max(intensity.sum(), 1e-12)),
        *loss_stats, *pair_hist.tolist(),
    ], dtype=np.float32)
    if len(features) != 36 or not np.all(np.isfinite(features)):
        raise RuntimeError("clean spectrum descriptor construction failed")
    return features


def metadata_vector(handle: h5py.File, row: int, keys: tuple[str, ...], bins: int, seed: int) -> np.ndarray:
    output: list[float] = []
    for key_index, key in enumerate(keys):
        value = handle[key][int(row)]
        text = decode(value).strip()
        missing = not text or text.lower() in {"nan", "none", "unknown"}
        numeric = 0.0
        try:
            numeric = float(np.asarray(value).reshape(-1)[0])
            if not np.isfinite(numeric):
                numeric = 0.0
        except (TypeError, ValueError):
            numeric = 0.0
        hashed = [0.0] * bins
        if not missing:
            hashed[stable_hash(text, bins, seed + key_index)] = 1.0
        output.extend([float(missing), numeric, *hashed])
    return np.asarray(output, dtype=np.float32)


def formula_bootstrap(values: np.ndarray, formulas: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    formulas = np.asarray(formulas, dtype=str)
    unique, inverse = np.unique(formulas, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, len(unique), len(unique))
        draws[index] = sums[selected].sum() / counts[selected].sum()
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975))}


def formula_equal_weights(formulas: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(np.asarray(formulas, dtype=str), return_inverse=True)
    counts = np.bincount(inverse)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def fit_predict(x_train: np.ndarray, frame_train: pd.DataFrame, x_test: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = formula_equal_weights(frame_train["query_formula"].to_numpy(str))
    common = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                  min_samples_leaf=50, l2_regularization=1.0,
                  early_stopping=False, random_state=seed)
    regressor = HistGradientBoostingRegressor(loss="squared_error", **common)
    positive = HistGradientBoostingClassifier(loss="log_loss", **common)
    harmful = HistGradientBoostingClassifier(loss="log_loss", **common)
    regressor.fit(x_train, frame_train["paired_advantage"].to_numpy(float), sample_weight=weights)
    positive.fit(x_train, frame_train["advantage_label"].eq("positive").to_numpy(np.int8), sample_weight=weights)
    harmful.fit(x_train, frame_train["advantage_label"].eq("harmful").to_numpy(np.int8), sample_weight=weights)
    return (regressor.predict(x_test).astype(np.float32),
            positive.predict_proba(x_test)[:, 1].astype(np.float32),
            harmful.predict_proba(x_test)[:, 1].astype(np.float32))


def cell_only_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = ["selector", "attenuation", "step"]
    # Give every formula equal influence inside a cell, matching the primary
    # model's training weights and the formula-cluster evaluation endpoint.
    by_formula = train.assign(
        is_positive=train["advantage_label"].eq("positive").astype(float),
        is_harmful=train["advantage_label"].eq("harmful").astype(float),
    ).groupby([*keys, "query_formula"], sort=True).agg(
        gain=("paired_advantage", "mean"),
        positive=("is_positive", "mean"), harmful=("is_harmful", "mean"),
    )
    grouped = by_formula.groupby(level=keys, sort=True).mean()
    fallback_by_formula = train.assign(
        is_positive=train["advantage_label"].eq("positive").astype(float),
        is_harmful=train["advantage_label"].eq("harmful").astype(float),
    ).groupby("query_formula").agg(
        gain=("paired_advantage", "mean"), positive=("is_positive", "mean"),
        harmful=("is_harmful", "mean"),
    ).mean()
    fallback = (float(fallback_by_formula["gain"]), float(fallback_by_formula["positive"]),
                float(fallback_by_formula["harmful"]))
    gain, positive, harmful = [], [], []
    for row in test.itertuples(index=False):
        key = (str(row.selector), float(row.attenuation), int(row.step))
        if key in grouped.index:
            item = grouped.loc[key]
            gain.append(float(item["gain"])); positive.append(float(item["positive"])); harmful.append(float(item["harmful"]))
        else:
            gain.append(fallback[0]); positive.append(fallback[1]); harmful.append(fallback[2])
    return np.asarray(gain), np.asarray(positive), np.asarray(harmful)


def permuted_train_frame(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = train.copy()
    rng = np.random.default_rng(seed)
    columns = ["paired_advantage", "advantage_label"]
    for _, indices in output.groupby(["selector", "attenuation", "step"], sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=np.int64)
        source = indices.copy(); rng.shuffle(source)
        output.loc[indices, columns] = train.loc[source, columns].to_numpy()
    return output


def prediction_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    truth_positive = frame["advantage_label"].eq("positive").to_numpy(np.int8)
    truth_harmful = frame["advantage_label"].eq("harmful").to_numpy(np.int8)
    p_positive = frame[f"{prefix}_p_positive"].to_numpy(float)
    p_harmful = frame[f"{prefix}_p_harmful"].to_numpy(float)
    gain = frame[f"{prefix}_pred_gain"].to_numpy(float)
    return {
        "positive_auprc": float(average_precision_score(truth_positive, p_positive)),
        "positive_auroc": float(roc_auc_score(truth_positive, p_positive)),
        "positive_brier": float(brier_score_loss(truth_positive, p_positive)),
        "harmful_auprc": float(average_precision_score(truth_harmful, p_harmful)),
        "harmful_auroc": float(roc_auc_score(truth_harmful, p_harmful)),
        "harmful_brier": float(brier_score_loss(truth_harmful, p_harmful)),
        "gain_pearson": float(np.corrcoef(gain, frame["paired_advantage"].to_numpy(float))[0, 1]),
    }


def select_policy(frame: pd.DataFrame, prefix: str, thresholds: dict[str, float],
                  repeats: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    eligible = (frame[f"{prefix}_p_positive"].ge(thresholds["positive_probability"])
                & frame[f"{prefix}_p_harmful"].le(thresholds["harmful_probability"])
                & frame[f"{prefix}_pred_gain"].ge(thresholds["predicted_gain"]))
    candidates = frame.loc[eligible].copy()
    candidates = candidates.sort_values(
        ["query_index", f"{prefix}_pred_gain", f"{prefix}_p_positive", f"{prefix}_p_harmful"],
        ascending=[True, False, False, True], kind="mergesort").drop_duplicates("query_index")
    base_columns = ["query_index", "query_ik14", "query_formula", "has_near", "baseline_rank"]
    population = frame[base_columns].drop_duplicates("query_index").copy()
    chosen_columns = ["query_index", "selector", "attenuation", "step", "target_rank",
                      "target_margin", "paired_advantage", "advantage_label", "transition",
                      "target_changes_top", "baseline_top_ik14", "target_top_ik14",
                      f"{prefix}_pred_gain", f"{prefix}_p_positive", f"{prefix}_p_harmful"]
    population = population.merge(candidates[chosen_columns], on="query_index", how="left", validate="one_to_one")
    selected = population["target_rank"].notna().to_numpy()
    baseline_correct = population["baseline_rank"].to_numpy(np.int64) == 1
    target_rank = population["target_rank"].fillna(population["baseline_rank"]).to_numpy(np.int64)
    target_correct = target_rank == 1
    effect = target_correct.astype(float) - baseline_correct.astype(float)
    paired = population["paired_advantage"].fillna(0.0).to_numpy(float)
    near = population["has_near"].astype(bool).to_numpy()
    corrected = int(np.sum(~baseline_correct & target_correct)); introduced = int(np.sum(baseline_correct & ~target_correct))
    summary = {
        "population_queries": int(len(population)), "selected_queries": int(np.sum(selected)),
        "selected_fraction": float(np.mean(selected)),
        "selected_identities": int(population.loc[selected, "query_ik14"].nunique()),
        "selected_formulas": int(population.loc[selected, "query_formula"].nunique()),
        "selected_positive_fraction": float(population.loc[selected, "advantage_label"].eq("positive").mean()) if np.any(selected) else None,
        "selected_harmful_fraction": float(population.loc[selected, "advantage_label"].eq("harmful").mean()) if np.any(selected) else None,
        "corrected": corrected, "introduced": introduced, "risk_net_lambda2": corrected - 2 * introduced,
        "delta_recall1": float(effect.mean()),
        "near_delta_recall1": float(effect[near].mean()) if np.any(near) else None,
        "population_paired_advantage": formula_bootstrap(paired, population["query_formula"].to_numpy(str), repeats, seed),
        "population_top1_delta": formula_bootstrap(effect, population["query_formula"].to_numpy(str), repeats, seed + 1),
    }
    return summary, population


def paired_policy_delta(left: pd.DataFrame, right: pd.DataFrame, repeats: int, seed: int) -> dict[str, float]:
    if not np.array_equal(left["query_index"], right["query_index"]):
        raise RuntimeError("policy comparison query order mismatch")
    def effect(frame: pd.DataFrame) -> np.ndarray:
        final = frame["target_rank"].fillna(frame["baseline_rank"]).to_numpy(np.int64) == 1
        initial = frame["baseline_rank"].to_numpy(np.int64) == 1
        return final.astype(float) - initial.astype(float)
    return formula_bootstrap(effect(left) - effect(right), left["query_formula"].to_numpy(str), repeats, seed)


def main() -> None:
    args = arguments()
    required = [args.l0_dir / "report.json", args.l0_dir / "action_labels.csv.gz",
                args.token_dir / "report.json", args.token_dir / "rows.npy",
                args.token_dir / "tokens_f16.npy", args.token_dir / "mz_f32.npy",
                args.token_dir / "intensity_f32.npy", args.token_dir / "valid.npy",
                args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
                args.clean_checkpoint, args.clean_checkpoint.parent / "decision.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    if args.output_dir.exists(): raise RuntimeError(f"completed-output path already exists: {args.output_dir}")
    if args.bootstrap_resamples < 100 or args.batch_size < 1: raise ValueError("invalid L1 numeric arguments")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")

    l0_report = json.loads((args.l0_dir / "report.json").read_text(encoding="utf-8"))
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if l0_report.get("status") != "noise_final_l0_action_learnability_ledger_complete" or not l0_report.get("formal"):
        raise RuntimeError("L1 requires a formal L0 ledger")
    l0_provenance = l0_report.get("provenance", {})
    if (l0_provenance.get("graph_sha256") != sha256_file(args.graph)
            or l0_provenance.get("hdf5_sha256") != sha256_file(args.data)
            or l0_provenance.get("clean_checkpoint_sha256") != sha256_file(args.clean_checkpoint)):
        raise RuntimeError("L0 and L1 data/checkpoint provenance differ")
    if token_report.get("status") != "noise_final_f1_full_token_cache_complete" or not token_report.get("formal"):
        raise RuntimeError("L1 requires the formal label-free contextual token cache")
    token_provenance = token_report.get("provenance", {})
    if token_provenance.get("graph_sha256") != sha256_file(args.graph) or token_provenance.get("hdf5_sha256") != sha256_file(args.data):
        raise RuntimeError("token cache data provenance differs from L1")
    if token_provenance.get("official_checkpoint_sha256") != sha256_file(args.official_checkpoint):
        raise RuntimeError("token cache official checkpoint differs")

    labels = pd.read_csv(args.l0_dir / "action_labels.csv.gz", low_memory=False)
    if len(labels) != int(l0_report.get("actions", -1)) or labels.empty: raise RuntimeError("L0 labels are incomplete")
    fold_counts = labels.groupby("query_formula")["formula_fold"].nunique()
    if not fold_counts.eq(1).all() or sorted(labels["formula_fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("formula-fold contract is invalid")

    graph = CandidateGraph(args.graph)
    query_rows = labels[["query_index", "query_row"]].drop_duplicates("query_index").sort_values("query_index")
    if not np.array_equal(query_rows["query_row"].to_numpy(np.int64),
                          graph.query_row[query_rows["query_index"].to_numpy(np.int64)]):
        raise RuntimeError("L1 clean query rows differ from candidate graph")
    rows = query_rows["query_row"].to_numpy(np.int64)
    if int(args.n_highest_peaks) != int(token_report.get("tokens_per_spectrum", -1)):
        raise RuntimeError("L1 spectrum length differs from the contextual token cache")
    store = SpectrumStore(args.data, rows, args.n_highest_peaks)
    model, clean_provenance = load_clean_model(args, device)
    mature = encode_rows(model, store, rows, device, args.batch_size, args.amp, "L1-mature-clean")
    del model
    if device.type == "cuda": torch.cuda.empty_cache()

    token_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    token_index = {int(row): index for index, row in enumerate(token_rows)}
    missing_token_rows = set(map(int, rows)) - set(token_index)
    if missing_token_rows: raise RuntimeError(f"token cache misses {len(missing_token_rows)} L1 query rows")
    tokens = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
    intensity_cache = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    valid_cache = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    embedding_projection = random_projection(mature.shape[1], args.embedding_projection_dim, args.seed)
    token_projection = random_projection(tokens.shape[2], args.token_projection_dim, args.seed + 1)
    with h5py.File(args.data, "r") as handle:
        present_metadata = tuple(
            key for key in METADATA_KEYS
            if key in handle and handle[key].shape and handle[key].shape[0] > int(rows.max())
        )
        query_features: list[np.ndarray] = []
        for position, row in enumerate(rows):
            cache_position = token_index[int(row)]
            valid = np.asarray(valid_cache[cache_position], dtype=bool)
            contextual = np.asarray(tokens[cache_position][valid], dtype=np.float32)
            intensity = np.asarray(intensity_cache[cache_position][valid], dtype=np.float32)
            if len(contextual):
                projected = contextual @ token_projection
                weights = intensity / max(float(intensity.sum()), 1e-12)
                pooled = np.concatenate([projected.mean(axis=0), projected.std(axis=0),
                                         projected.max(axis=0), np.sum(projected * weights[:, None], axis=0)]).astype(np.float32)
            else:
                pooled = np.zeros(4 * args.token_projection_dim, dtype=np.float32)
            spectrum = store.one(int(row)).numpy()
            raw = spectrum_descriptors(spectrum[1:, 0], spectrum[1:, 1], float(spectrum[0, 0]))
            metadata = metadata_vector(handle, int(row), present_metadata, args.metadata_hash_bins, args.seed + 2)
            query_features.append(np.concatenate([mature[position] @ embedding_projection, pooled, raw, metadata]).astype(np.float32))
    query_feature_matrix = np.stack(query_features)
    if not np.all(np.isfinite(query_feature_matrix)): raise RuntimeError("L1 clean-query features contain non-finite values")
    query_to_position = {int(query): index for index, query in enumerate(query_rows["query_index"])}
    row_positions = np.asarray([query_to_position[int(query)] for query in labels["query_index"]], dtype=np.int64)
    clean_features = query_feature_matrix[row_positions]
    selector_candidate = labels["selector"].eq("candidate_gradient").to_numpy(float)
    selector_role = labels["selector"].eq("role_confounder").to_numpy(float)
    step = labels["step"].to_numpy(np.int64)
    step_one_hot = np.eye(6, dtype=np.float32)[step - 1]
    action_features = np.column_stack([selector_candidate, selector_role,
                                      labels["attenuation"].to_numpy(float), step / 6.0,
                                      step_one_hot, step_one_hot * selector_candidate[:, None],
                                      step_one_hot * selector_role[:, None]]).astype(np.float32)
    features = np.concatenate([clean_features, action_features], axis=1)
    query_feature_names = (
        [f"mature_embedding_rp_{index}" for index in range(args.embedding_projection_dim)]
        + [f"contextual_token_{stat}_rp_{index}" for stat in ("mean", "std", "max", "intensity_weighted_mean")
           for index in range(args.token_projection_dim)]
        + list(RAW_FEATURE_NAMES)
        + [name for key in present_metadata
           for name in ([f"metadata_{key}_missing", f"metadata_{key}_numeric"]
                        + [f"metadata_{key}_hash_{index}" for index in range(args.metadata_hash_bins)])]
    )
    action_feature_names = (
        ["action_family_candidate_gradient", "action_family_role_confounder",
         "action_attenuation", "action_step_scaled"]
        + [f"action_step_{index}" for index in range(1, 7)]
        + [f"action_candidate_step_{index}" for index in range(1, 7)]
        + [f"action_role_step_{index}" for index in range(1, 7)]
    )
    if len(query_feature_names) != query_feature_matrix.shape[1] or len(action_feature_names) != action_features.shape[1]:
        raise RuntimeError("L1 feature-name matrix mismatch")

    for prefix in ("clean", "family", "permuted"):
        labels[f"{prefix}_pred_gain"] = np.nan; labels[f"{prefix}_p_positive"] = np.nan; labels[f"{prefix}_p_harmful"] = np.nan
    fold_reports: list[dict[str, object]] = []
    for fold in range(5):
        test_mask = labels["formula_fold"].to_numpy(np.int64) == fold; train_mask = ~test_mask
        train = labels.loc[train_mask].copy(); test = labels.loc[test_mask].copy()
        predictions = {
            "clean": fit_predict(features[train_mask], train, features[test_mask], args.seed + fold),
            "family": cell_only_predict(train, test),
            "permuted": fit_predict(features[train_mask], permuted_train_frame(train, args.seed + 100 + fold),
                                     features[test_mask], args.seed + 200 + fold),
        }
        for prefix, values in predictions.items():
            labels.loc[test_mask, f"{prefix}_pred_gain"] = values[0]
            labels.loc[test_mask, f"{prefix}_p_positive"] = values[1]
            labels.loc[test_mask, f"{prefix}_p_harmful"] = values[2]
        fold_reports.append({"fold": fold, "train_actions": int(train_mask.sum()), "held_actions": int(test_mask.sum()),
                             "train_formulas": int(train["query_formula"].nunique()), "held_formulas": int(test["query_formula"].nunique()),
                             "formula_overlap": int(len(set(train["query_formula"]) & set(test["query_formula"]))),
                             "identity_overlap": int(len(set(train["query_ik14"]) & set(test["query_ik14"])))})
    prediction_columns = [column for column in labels if column.endswith(("_pred_gain", "_p_positive", "_p_harmful"))]
    if not np.all(np.isfinite(labels[prediction_columns].to_numpy(float))): raise RuntimeError("L1 OOF predictions are incomplete")

    metrics = {prefix: prediction_metrics(labels, prefix) for prefix in ("clean", "family", "permuted")}
    policies: dict[str, dict[str, object]] = {}; policy_frames: dict[str, pd.DataFrame] = {}
    for prefix_index, prefix in enumerate(("clean", "family", "permuted")):
        policies[prefix] = {}
        for offset, (name, threshold) in enumerate(POLICY_SENSITIVITY.items()):
            summary, per_query = select_policy(labels, prefix, threshold, args.bootstrap_resamples,
                                               args.seed + 1000 + offset + 10 * prefix_index)
            policies[prefix][name] = summary
            if name == "primary": policy_frames[prefix] = per_query

    clean_primary = policies["clean"]["primary"]
    clean_vs_family = paired_policy_delta(policy_frames["clean"], policy_frames["family"],
                                          args.bootstrap_resamples, args.seed + 3000)
    clean_vs_permuted = paired_policy_delta(policy_frames["clean"], policy_frames["permuted"],
                                            args.bootstrap_resamples, args.seed + 3001)
    selected_clean = policy_frames["clean"]
    false_positive = selected_clean.loc[selected_clean["target_rank"].notna()
                                        & (selected_clean["advantage_label"].eq("harmful")
                                           | selected_clean["transition"].eq("introduced"))].copy()
    gates = {
        "positive_auprc_beats_family": bool(metrics["clean"]["positive_auprc"] > metrics["family"]["positive_auprc"]),
        "positive_auprc_beats_permutation": bool(metrics["clean"]["positive_auprc"] > metrics["permuted"]["positive_auprc"]),
        "primary_selected_queries_ge_500": bool(int(clean_primary["selected_queries"]) >= 500),
        "primary_selected_identities_ge_300": bool(int(clean_primary["selected_identities"]) >= 300),
        "primary_selected_formulas_ge_150": bool(int(clean_primary["selected_formulas"]) >= 150),
        "primary_margin_formula_ci_positive": bool(clean_primary["population_paired_advantage"]["ci_low"] > 0),
        "primary_top1_formula_ci_positive": bool(clean_primary["population_top1_delta"]["ci_low"] > 0),
        "primary_corrected_gt_introduced": bool(int(clean_primary["corrected"]) > int(clean_primary["introduced"])),
        "primary_risk_net_positive": bool(int(clean_primary["risk_net_lambda2"]) > 0),
        "primary_near_nonnegative": bool(float(clean_primary["near_delta_recall1"]) >= 0),
        "primary_top1_beats_family_formula_ci": bool(clean_vs_family["ci_low"] > 0),
        "primary_top1_beats_permutation_formula_ci": bool(clean_vs_permuted["ci_low"] > 0),
        "every_outer_fold_formula_disjoint": bool(all(item["formula_overlap"] == 0 for item in fold_reports)),
    }
    pass_to_l2 = bool(all(gates.values()))
    report = {
        "status": "noise_final_l1_clean_action_learnability_complete", "formal": True,
        "actions": int(len(labels)), "queries": int(labels["query_index"].nunique()),
        "identities": int(labels["query_ik14"].nunique()), "formulas": int(labels["query_formula"].nunique()),
        "clean_query_feature_dimension": int(features.shape[1]),
        "contextual_token_projection_dimension": int(args.token_projection_dim),
        "mature_embedding_projection_dimension": int(args.embedding_projection_dim),
        "acquisition_metadata_fields_present": list(present_metadata),
        "query_feature_names": query_feature_names,
        "action_descriptor_feature_names": action_feature_names,
        "feature_contract": {"clean_spectrum_only": True, "contextual_peak_tokens_label_free": True,
                             "target_path_used": False, "candidate_scores_used": False,
                             "baseline_rank_or_margin_used": False, "identity_or_formula_used_as_feature": False,
                             "identity_and_formula_used_only_for_weighting_split_and_audit": True,
                             "action_family_and_step_used": True, "P2b": "forbidden", "P3_consumed": False},
        "model": {"primary": "HistGradientBoosting gain + positive + harmful models",
                  "fixed_parameters": {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15,
                                       "min_samples_leaf": 50, "l2_regularization": 1.0},
                  "formula_equal_training_weights": True, "outer_formula_folds": 5},
        "prediction_metric_contrasts": {
            "positive_auprc_minus_family": float(metrics["clean"]["positive_auprc"] - metrics["family"]["positive_auprc"]),
            "positive_auprc_minus_permuted": float(metrics["clean"]["positive_auprc"] - metrics["permuted"]["positive_auprc"]),
        },
        "folds": fold_reports, "prediction_metrics": metrics,
        "fixed_policy_thresholds": POLICY_SENSITIVITY, "policy_results": policies,
        "primary_clean_vs_family_top1": clean_vs_family,
        "primary_clean_vs_permuted_top1": clean_vs_permuted,
        "primary_false_positive_actions": int(len(false_positive)), "gates": gates,
        "pass_to_l2_small_causal_pilot": pass_to_l2,
        "decision": ("authorize one paired targeted-vs-matched-random shared-encoder L2 pilot" if pass_to_l2
                     else "do not train L2; retain predictable strata only for action redesign or explanation"),
        "provenance": {"l0_report_sha256": sha256_file(args.l0_dir / "report.json"),
                       "l0_labels_sha256": sha256_file(args.l0_dir / "action_labels.csv.gz"),
                       "token_report_sha256": sha256_file(args.token_dir / "report.json"),
                       "token_rows_sha256": sha256_file(args.token_dir / "rows.npy"),
                       "graph_sha256": sha256_file(args.graph), "hdf5_sha256": sha256_file(args.data),
                       **clean_provenance, "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Formula-OOF clean-input action predictability and no-op routing audit; not a trained embedding or deployable result.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_l1_", dir=args.output_dir.parent))
    try:
        labels.to_csv(temporary / "action_oof_predictions.csv.gz", index=False, compression="gzip")
        policy_frames["clean"].to_csv(temporary / "primary_per_query.csv.gz", index=False, compression="gzip")
        false_positive.to_csv(temporary / "primary_false_positive_audit.csv.gz", index=False, compression="gzip")
        np.savez_compressed(temporary / "clean_query_features.npz",
                            query_index=query_rows["query_index"].to_numpy(np.int64), query_row=rows,
                            features=query_feature_matrix,
                            feature_names=np.asarray(query_feature_names, dtype=str))
        json_dump(temporary / "report.json", report)
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

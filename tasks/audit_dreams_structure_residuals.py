"""Build a structure-similarity residual atlas for DreaMS embeddings.

This is an evaluation-only stage.  It does not update model weights and it
does not use chemical rules as labels.  The script mirrors the structural
correlation question in DreaMS Figure 4a, then goes one step further by
calibrating embedding similarity against Morgan Tanimoto on a molecule-
disjoint discovery cohort and auditing residuals on a confirmation cohort.

Default inputs reuse official fine-tuned embeddings already encoded for the
large observability cohorts.  The cached raw-SSL E0 embeddings are aligned by
spectrum identifier so that raw and official weights are compared on exactly
the same spectrum pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISCOVERY = ROOT / "data/validation/large_observability_embeddings_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/large_observability_embeddings_confirmation"
DEFAULT_RAW = ROOT / "data/validation/e0_baseline"
DEFAULT_P0 = ROOT / "data/validation/e0_failure_audit/e0_query_audit.csv"
DEFAULT_MCES = ROOT / "data/validation/e0_failure_audit/deduplicated_error_pairs.csv"
DEFAULT_OUTPUT = ROOT / "data/validation/dreams_structure_residual_atlas"

TANIMOTO_EDGES = np.linspace(0.0, 1.0, 11)
TANIMOTO_LABELS = [f"{TANIMOTO_EDGES[i]:.1f}-{TANIMOTO_EDGES[i + 1]:.1f}" for i in range(10)]
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-dir", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation-dir", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--p0-query-audit", type=Path, default=DEFAULT_P0)
    parser.add_argument("--mces-pairs", type=Path, default=DEFAULT_MCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-per-bin-discovery", type=int, default=3000)
    parser.add_argument("--pairs-per-bin-confirmation", type=int, default=1200)
    parser.add_argument("--identity-pairs-discovery", type=int, default=3000)
    parser.add_argument("--identity-pairs-confirmation", type=int, default=1200)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_cohort(directory: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    manifest_path = directory / "manifest.csv"
    embedding_path = directory / "official_embeddings.npy"
    report_path = directory / "report.json"
    if not manifest_path.exists() or not embedding_path.exists():
        raise FileNotFoundError(f"Incomplete cohort directory: {directory}")
    frame = pd.read_csv(manifest_path)
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or len(frame) != embeddings.shape[0]:
        raise ValueError(f"Manifest/embedding mismatch in {directory}")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return frame, embeddings, report


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def align_raw_embeddings(frame: pd.DataFrame, raw_dir: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads((raw_dir / "e0_manifest.json").read_text(encoding="utf-8"))
    raw = np.load(raw_dir / "e0_embeddings.npy", mmap_mode="r")
    identifier_to_idx = {str(row["spectrum_id"]): int(row["embedding_idx"]) for row in manifest}
    aligned = np.full((len(frame), raw.shape[1]), np.nan, dtype=np.float32)
    missing = []
    for row_idx, identifier in enumerate(frame["identifier"].astype(str)):
        source_idx = identifier_to_idx.get(identifier)
        if source_idx is None:
            missing.append(identifier)
            continue
        aligned[row_idx] = raw[source_idx]
    valid = np.isfinite(aligned).all(axis=1)
    if np.any(valid):
        aligned[valid] = l2_normalize(aligned[valid])
    return aligned, {
        "source": str(raw_dir / "e0_embeddings.npy"),
        "matched_spectra": int(valid.sum()),
        "missing_spectra": int((~valid).sum()),
        "first_missing_identifiers": missing[:10],
    }


def modal_smiles(group: pd.DataFrame) -> str:
    values = group["smiles"].dropna().astype(str)
    if values.empty:
        return ""
    counts = values.value_counts()
    return str(counts.index[0])


def molecule_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    rows = []
    spectra_by_ik: dict[str, np.ndarray] = {}
    ambiguous_smiles = 0
    for ik14, group in frame.groupby("ik14", sort=True):
        counts = group["smiles"].dropna().astype(str).value_counts()
        if len(counts) > 1:
            ambiguous_smiles += 1
        rows.append({
            "ik14": str(ik14),
            "smiles": modal_smiles(group),
            "formula": str(group["formula"].mode().iloc[0]) if group["formula"].notna().any() else "",
            "n_spectra": int(len(group)),
        })
        spectra_by_ik[str(ik14)] = group.index.to_numpy(dtype=np.int64)
    return pd.DataFrame(rows), spectra_by_ik, ambiguous_smiles


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)


def tanimoto_bin(value: float) -> int:
    return min(9, max(0, int(math.floor(float(value) * 10.0))))


def reservoir_structure_pairs(
    molecules: pd.DataFrame,
    per_bin: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    valid_rows, fps = [], []
    invalid = []
    for row in molecules.itertuples(index=False):
        fp = fingerprint(row.smiles)
        if fp is None:
            invalid.append(row.ik14)
            continue
        valid_rows.append(row)
        fps.append(fp)

    reservoirs: list[list[tuple[int, int, float]]] = [[] for _ in range(10)]
    seen = np.zeros(10, dtype=np.int64)
    for i, fp in enumerate(fps[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
        for offset, value in enumerate(similarities, start=i + 1):
            bin_idx = tanimoto_bin(value)
            seen[bin_idx] += 1
            bucket = reservoirs[bin_idx]
            candidate = (i, offset, float(value))
            if len(bucket) < per_bin:
                bucket.append(candidate)
            else:
                replacement = int(rng.integers(0, seen[bin_idx]))
                if replacement < per_bin:
                    bucket[replacement] = candidate

    pairs = []
    for bin_idx, bucket in enumerate(reservoirs):
        for i, j, value in bucket:
            pairs.append({
                "pair_type": "different_identity",
                "ik_a": str(valid_rows[i].ik14),
                "ik_b": str(valid_rows[j].ik14),
                "smiles_a": str(valid_rows[i].smiles),
                "smiles_b": str(valid_rows[j].smiles),
                "formula_a": str(valid_rows[i].formula),
                "formula_b": str(valid_rows[j].formula),
                "same_formula": bool(str(valid_rows[i].formula) == str(valid_rows[j].formula)),
                "tanimoto": value,
                "tanimoto_bin": TANIMOTO_LABELS[bin_idx],
            })
    return pairs, {
        "valid_molecules": len(valid_rows),
        "invalid_smiles": len(invalid),
        "available_pairs_by_bin": {TANIMOTO_LABELS[i]: int(seen[i]) for i in range(10)},
        "sampled_pairs_by_bin": {TANIMOTO_LABELS[i]: int(len(reservoirs[i])) for i in range(10)},
    }


def attach_spectrum_indices(
    pair_rows: list[dict],
    spectra_by_ik: dict[str, np.ndarray],
    identity_cap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for row in pair_rows:
        a = spectra_by_ik[row["ik_a"]]
        b = spectra_by_ik[row["ik_b"]]
        rows.append(row | {
            "row_a": int(rng.choice(a)),
            "row_b": int(rng.choice(b)),
        })

    identity_candidates = [ik for ik, indices in spectra_by_ik.items() if len(indices) >= 2]
    rng.shuffle(identity_candidates)
    for ik in identity_candidates[:identity_cap]:
        indices = spectra_by_ik[ik]
        a, b = rng.choice(indices, size=2, replace=False)
        rows.append({
            "pair_type": "same_identity",
            "ik_a": ik,
            "ik_b": ik,
            "smiles_a": "",
            "smiles_b": "",
            "formula_a": "",
            "formula_b": "",
            "same_formula": True,
            "tanimoto": 1.0,
            "tanimoto_bin": "identity",
            "row_a": int(a),
            "row_b": int(b),
        })
    return pd.DataFrame(rows)


def sample_pairs(
    frame: pd.DataFrame,
    per_bin: int,
    identity_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    molecules, spectra_by_ik, ambiguous = molecule_table(frame)
    rng = np.random.default_rng(seed)
    pair_rows, audit = reservoir_structure_pairs(molecules, per_bin, rng)
    audit["ambiguous_smiles_ik14"] = int(ambiguous)
    pairs = attach_spectrum_indices(pair_rows, spectra_by_ik, identity_cap, rng)
    molecule_smiles = molecules.set_index("ik14")["smiles"].to_dict()
    molecule_formula = molecules.set_index("ik14")["formula"].to_dict()
    blank_a = pairs["smiles_a"].eq("")
    blank_b = pairs["smiles_b"].eq("")
    pairs.loc[blank_a, "smiles_a"] = pairs.loc[blank_a, "ik_a"].map(molecule_smiles)
    pairs.loc[blank_b, "smiles_b"] = pairs.loc[blank_b, "ik_b"].map(molecule_smiles)
    pairs.loc[pairs["formula_a"].eq(""), "formula_a"] = pairs["ik_a"].map(molecule_formula)
    pairs.loc[pairs["formula_b"].eq(""), "formula_b"] = pairs["ik_b"].map(molecule_formula)
    row_a = pairs["row_a"].to_numpy(dtype=np.int64)
    row_b = pairs["row_b"].to_numpy(dtype=np.int64)
    instrument_a = frame.iloc[row_a]["instrument"].fillna("").astype(str).to_numpy()
    instrument_b = frame.iloc[row_b]["instrument"].fillna("").astype(str).to_numpy()
    ce_a = pd.to_numeric(frame.iloc[row_a]["collision_energy"], errors="coerce").to_numpy(float)
    ce_b = pd.to_numeric(frame.iloc[row_b]["collision_energy"], errors="coerce").to_numpy(float)
    pairs["instrument_a"] = instrument_a
    pairs["instrument_b"] = instrument_b
    pairs["same_instrument"] = (instrument_a == instrument_b) & (instrument_a != "")
    pairs["ce_delta"] = np.abs(ce_a - ce_b)
    audit["identity_pairs"] = int((pairs["pair_type"] == "same_identity").sum())
    audit["total_sampled_pairs"] = int(len(pairs))
    return pairs, audit


def rowwise_cosine(matrix: np.ndarray, pair_frame: pd.DataFrame) -> np.ndarray:
    a = np.asarray(matrix[pair_frame["row_a"].to_numpy(dtype=np.int64)], dtype=np.float32)
    b = np.asarray(matrix[pair_frame["row_b"].to_numpy(dtype=np.int64)], dtype=np.float32)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    result = np.full(len(pair_frame), np.nan, dtype=np.float32)
    result[valid] = np.einsum("ij,ij->i", a[valid], b[valid], optimize=True)
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[valid], float), np.asarray(y[valid], float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "pearson_r": None, "spearman_rho": None}
    return {
        "n": int(len(x)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }


def fast_pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[valid], float), np.asarray(y[valid], float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def paired_anchor_bootstrap(
    frame: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Paired comparison with resampling clustered on the first molecule.

    This is stricter than an independent row bootstrap, although a molecule can
    still occur as the second member of another pair.  The limitation is kept
    explicit in the output and the interval is used as a stability check, not a
    population-level confidence guarantee.
    """
    rng = np.random.default_rng(seed)
    masks = {
        "all_pairs": np.ones(len(frame), dtype=bool),
        "different_identity": frame["pair_type"].eq("different_identity").to_numpy(),
        "same_formula_different_identity": (
            frame["pair_type"].eq("different_identity") & frame["same_formula"].fillna(False)
        ).to_numpy(),
    }
    output = {
        "method": "paired anchor-molecule clustered bootstrap",
        "limitations": "Pairs are clustered by ik_a; the same molecule may also occur as ik_b.",
        "n_bootstrap": int(n_bootstrap),
        "pearson_delta_official_minus_raw": {},
    }
    for label, mask in masks.items():
        subset = frame.loc[mask].reset_index(drop=True)
        groups = [group.index.to_numpy(dtype=np.int64) for _, group in subset.groupby("ik_a", sort=False)]
        point = fast_pearson(
            subset["tanimoto"].to_numpy(float), subset[f"cosine_{model_a}"].to_numpy(float)
        ) - fast_pearson(
            subset["tanimoto"].to_numpy(float), subset[f"cosine_{model_b}"].to_numpy(float)
        )
        draws = []
        if n_bootstrap > 0 and len(groups) >= 2:
            for _ in range(n_bootstrap):
                chosen = rng.integers(0, len(groups), size=len(groups))
                indices = np.concatenate([groups[idx] for idx in chosen])
                x = subset.loc[indices, "tanimoto"].to_numpy(float)
                delta = fast_pearson(x, subset.loc[indices, f"cosine_{model_a}"].to_numpy(float))
                delta -= fast_pearson(x, subset.loc[indices, f"cosine_{model_b}"].to_numpy(float))
                if np.isfinite(delta):
                    draws.append(delta)
        output["pearson_delta_official_minus_raw"][label] = {
            "n_pairs": int(len(subset)),
            "n_anchor_clusters": int(len(groups)),
            "point": float(point),
            "bootstrap_95ci": (
                [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                if draws else None
            ),
        }
    identity = frame.loc[frame["pair_type"] == "same_identity"].copy()
    delta = identity[f"cosine_{model_a}"] - identity[f"cosine_{model_b}"]
    if len(delta):
        draws = []
        for _ in range(n_bootstrap):
            sample = rng.choice(delta.to_numpy(float), len(delta), replace=True)
            draws.append(float(np.mean(sample)))
        output["identity_cosine_delta_official_minus_raw"] = {
            "n_pairs": int(len(delta)),
            "point": float(delta.mean()),
            "bootstrap_95ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
            if draws else None,
        }
    return output


def fit_calibrator(frame: pd.DataFrame, score_col: str) -> IsotonicRegression:
    # Structural calibration is a different-identity task. Same-identity
    # consistency is audited separately and must not dominate the Tanimoto=1
    # end of the isotonic curve.
    valid = (
        np.isfinite(frame[score_col])
        & np.isfinite(frame["tanimoto"])
        & frame["pair_type"].eq("different_identity")
    )
    return IsotonicRegression(y_min=-1.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(
        frame.loc[valid, "tanimoto"], frame.loc[valid, score_col]
    )


def add_residuals(
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    models: list[str],
) -> tuple[dict[str, IsotonicRegression], dict]:
    calibrators = {}
    thresholds = {}
    for model in models:
        score_col = f"cosine_{model}"
        calibrator = fit_calibrator(discovery, score_col)
        calibrators[model] = calibrator
        for frame in (discovery, confirmation):
            valid = np.isfinite(frame[score_col])
            expected = np.full(len(frame), np.nan, dtype=float)
            expected[valid] = calibrator.predict(frame.loc[valid, "tanimoto"].to_numpy(float))
            frame[f"expected_{model}"] = expected
            frame[f"residual_{model}"] = frame[score_col] - expected
        structural_discovery = discovery.loc[
            discovery["pair_type"].eq("different_identity")
        ]
        residual = structural_discovery[f"residual_{model}"].dropna().to_numpy(float)
        low, high = np.quantile(residual, [0.10, 0.90])
        identity_score = discovery.loc[
            discovery["pair_type"] == "same_identity", score_col
        ].dropna().to_numpy(float)
        identity_low = float(np.quantile(identity_score, 0.10)) if len(identity_score) else float("nan")
        thresholds[model] = {
            "residual_q10": float(low),
            "residual_q90": float(high),
            "identity_cosine_q10": identity_low,
        }
        residual_col = f"residual_{model}"
        for frame in (discovery, confirmation):
            family = np.full(len(frame), "calibrated_middle", dtype=object)
            different_identity = frame["pair_type"].eq("different_identity").to_numpy(bool)
            residual_values = frame[residual_col].to_numpy(float)
            family[different_identity & (residual_values <= low)] = (
                "lower_than_structure_expected"
            )
            family[different_identity & (residual_values >= high)] = (
                "higher_than_structure_expected"
            )
            identity_unstable = (
                (frame["pair_type"] == "same_identity")
                & (frame[score_col] <= identity_low)
            )
            family[identity_unstable.to_numpy(bool)] = "same_identity_instability"
            frame[f"residual_family_{model}"] = family
    return calibrators, thresholds


def model_summary(frame: pd.DataFrame, model: str) -> dict:
    score = frame[f"cosine_{model}"].to_numpy(float)
    tanimoto = frame["tanimoto"].to_numpy(float)
    different = frame["pair_type"].eq("different_identity").to_numpy()
    same_formula = frame["same_formula"].fillna(False).to_numpy(bool) & different
    identity = frame["pair_type"].eq("same_identity").to_numpy()
    residual = frame[f"residual_{model}"].to_numpy(float)
    valid_residual = np.isfinite(residual)
    output = {
        "all_pairs": correlation(tanimoto, score),
        "different_identity_pairs": correlation(tanimoto[different], score[different]),
        "same_formula_different_identity_pairs": correlation(tanimoto[same_formula], score[same_formula]),
        "residual_mae": float(np.mean(np.abs(residual[valid_residual]))),
        "residual_rmse": float(np.sqrt(np.mean(residual[valid_residual] ** 2))),
        "identity_cosine_mean": float(np.nanmean(score[identity])),
        "identity_cosine_p10": float(np.nanquantile(score[identity], 0.10)),
    }
    score_sd = float(np.nanstd(score))
    output["residual_nrmse_by_score_sd"] = (
        float(output["residual_rmse"] / score_sd) if score_sd > 0 else None
    )
    output["calibration_r2"] = (
        float(1.0 - np.mean(residual[valid_residual] ** 2) / np.nanvar(score))
        if score_sd > 0 else None
    )
    identity_frame = frame.loc[identity].copy()
    if not identity_frame.empty:
        condition_rows = {}
        for label, mask in {
            "same_instrument": identity_frame["same_instrument"].fillna(False),
            "cross_or_unknown_instrument": ~identity_frame["same_instrument"].fillna(False),
            "ce_delta_le_10": identity_frame["ce_delta"].fillna(np.inf) <= 10,
            "ce_delta_gt_10": identity_frame["ce_delta"].fillna(-np.inf) > 10,
        }.items():
            values = identity_frame.loc[mask, f"cosine_{model}"].dropna().to_numpy(float)
            condition_rows[label] = {
                "n": int(len(values)),
                "mean_cosine": float(np.mean(values)) if len(values) else None,
                "p10_cosine": float(np.quantile(values, 0.10)) if len(values) else None,
            }
        output["identity_condition_strata"] = condition_rows
    family_col = f"residual_family_{model}"
    if family_col in frame:
        output["residual_families"] = {
            str(k): int(v) for k, v in frame[family_col].value_counts().items()
        }
    return output


def attach_cached_mces(top1: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists() or top1.empty:
        top1["mces"] = np.nan
        top1["mces_bin"] = "missing"
        return top1
    cached = pd.read_csv(path)
    key_to_mces = {}
    for row in cached.itertuples(index=False):
        key_to_mces[tuple(sorted((str(row.ik_a), str(row.ik_b))))] = (row.mces, row.mces_bin)
    values = [key_to_mces.get(tuple(sorted((str(a), str(b)))), (np.nan, "missing"))
              for a, b in zip(top1["query_ik14"], top1["best_negative_ik14"])]
    top1["mces"] = [value[0] for value in values]
    top1["mces_bin"] = [value[1] for value in values]
    return top1


def top1_intersection(
    cohort_name: str,
    frame: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    calibrators: dict[str, IsotonicRegression],
    thresholds: dict,
    audit_path: Path,
    mces_path: Path,
) -> pd.DataFrame:
    if not audit_path.exists():
        return pd.DataFrame()
    audit = pd.read_csv(audit_path)
    index = {str(identifier): int(i) for i, identifier in enumerate(frame["identifier"].astype(str))}
    needed = ["query_spectrum_id", "correct_best_spectrum_id", "best_negative_spectrum_id"]
    keep = np.ones(len(audit), dtype=bool)
    for column in needed:
        keep &= audit[column].astype(str).isin(index).to_numpy()
    output = audit.loc[keep].copy()
    if output.empty:
        return output
    output.insert(0, "cohort", cohort_name)
    q_idx = np.asarray([index[str(x)] for x in output["query_spectrum_id"]], dtype=np.int64)
    p_idx = np.asarray([index[str(x)] for x in output["correct_best_spectrum_id"]], dtype=np.int64)
    n_idx = np.asarray([index[str(x)] for x in output["best_negative_spectrum_id"]], dtype=np.int64)
    t_negative = output["morgan_tanimoto"].to_numpy(float)
    for model, matrix in matrices.items():
        q = np.asarray(matrix[q_idx], dtype=np.float32)
        p = np.asarray(matrix[p_idx], dtype=np.float32)
        n = np.asarray(matrix[n_idx], dtype=np.float32)
        positive = np.einsum("ij,ij->i", q, p, optimize=True)
        negative = np.einsum("ij,ij->i", q, n, optimize=True)
        expected_positive = calibrators[model].predict(np.ones(len(output)))
        expected_negative = calibrators[model].predict(t_negative)
        output[f"positive_cosine_{model}"] = positive
        output[f"negative_cosine_{model}"] = negative
        output[f"margin_{model}"] = positive - negative
        output[f"top1_correct_{model}"] = positive > negative
        output[f"positive_residual_{model}"] = positive - expected_positive
        output[f"negative_residual_{model}"] = negative - expected_negative
        low = thresholds[model]["residual_q10"]
        high = thresholds[model]["residual_q90"]
        labels = np.full(len(output), "other_local_ranking_error", dtype=object)
        labels[(negative - expected_negative) >= high] = "wrong_candidate_overaggregated"
        labels[(positive - expected_positive) <= low] = "identity_positive_underaggregated"
        both = ((negative - expected_negative) >= high) & ((positive - expected_positive) <= low)
        labels[both] = "both_overaggregation_and_underaggregation"
        labels[(positive - negative) > 0] = "locally_correct"
        output[f"local_error_family_{model}"] = labels
    return attach_cached_mces(output, mces_path)


def plot_atlas(
    confirmation: pd.DataFrame,
    models: list[str],
    calibrators: dict[str, IsotonicRegression],
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(models), 3, figsize=(18, 5.4 * len(models)), squeeze=False)
    x_curve = np.linspace(0.0, 1.0, 201)
    for row_idx, model in enumerate(models):
        score_col = f"cosine_{model}"
        residual_col = f"residual_{model}"
        family_col = f"residual_family_{model}"
        ax = axes[row_idx, 0]
        valid = np.isfinite(confirmation[score_col])
        image = ax.hexbin(
            confirmation.loc[valid, "tanimoto"], confirmation.loc[valid, score_col],
            gridsize=45, bins="log", mincnt=1, cmap="Blues",
        )
        ax.plot(x_curve, calibrators[model].predict(x_curve), color="#b2182b", lw=2.5,
                label="discovery isotonic calibration")
        stats = correlation(
            confirmation.loc[valid, "tanimoto"].to_numpy(float),
            confirmation.loc[valid, score_col].to_numpy(float),
        )
        ax.text(0.03, 0.96,
                f"Pearson r={stats['pearson_r']:.3f}\nSpearman rho={stats['spearman_rho']:.3f}\nN={stats['n']:,}",
                transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.9})
        ax.set(xlabel="Morgan Tanimoto (radius=2, 4096 bits)", ylabel="Embedding cosine",
               title=f"{model}: structure vs embedding")
        ax.legend(loc="lower right")
        fig.colorbar(image, ax=ax, label="pair density (log)")

        ax = axes[row_idx, 1]
        order = TANIMOTO_LABELS + ["identity"]
        data = [confirmation.loc[confirmation["tanimoto_bin"] == label, residual_col].dropna().to_numpy()
                for label in order]
        positions = [idx + 1 for idx, values in enumerate(data) if len(values)]
        present = [values for values in data if len(values)]
        present_labels = [label for label, values in zip(order, data) if len(values)]
        ax.boxplot(present, positions=positions, showfliers=False, widths=0.65)
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xticks(positions, present_labels, rotation=45, ha="right")
        ax.set(xlabel="Tanimoto stratum", ylabel="Cosine residual",
               title="Residual after discovery calibration")

        ax = axes[row_idx, 2]
        counts = confirmation[family_col].value_counts()
        labels = list(counts.index)
        values = counts.to_numpy()
        ax.barh(labels, values, color=["#6baed6", "#fdae6b", "#e6550d", "#9e9ac8"][: len(values)])
        for idx, value in enumerate(values):
            ax.text(value, idx, f" {value:,}", va="center")
        ax.set(xlabel="Confirmation pairs", title="Residual error families")
    fig.suptitle("DreaMS structure-similarity residual atlas", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_summary(summary: dict, path: Path) -> None:
    paired = summary.get("paired_common_support", {})
    if not paired:
        return
    metrics = paired["confirmation_metrics"]
    official = metrics["official_finetuned"]
    raw = metrics["raw_ssl"]
    transition = paired.get("top1_transitions_official_vs_raw", {}).get("confirmation", {})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scopes = ["all_pairs", "different_identity_pairs", "same_formula_different_identity_pairs"]
    labels = ["All balanced pairs", "Different identities", "Same-formula identities"]
    x = np.arange(len(scopes))
    width = 0.36
    axes[0, 0].bar(x - width / 2, [raw[key]["pearson_r"] for key in scopes], width,
                   label="Raw SSL", color="#9ecae1")
    axes[0, 0].bar(x + width / 2, [official[key]["pearson_r"] for key in scopes], width,
                   label="Official fine-tuned", color="#6a3d9a")
    axes[0, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0, 0].set_ylabel("Pearson r")
    axes[0, 0].set_title("Fine-tuning changes global and local structure alignment differently")
    axes[0, 0].legend()

    condition_keys = ["same_instrument", "cross_or_unknown_instrument", "ce_delta_le_10", "ce_delta_gt_10"]
    condition_labels = ["Same instrument", "Cross/unknown instrument", "CE delta <=10", "CE delta >10"]
    raw_condition = raw["identity_condition_strata"]
    official_condition = official["identity_condition_strata"]
    axes[0, 1].bar(x=np.arange(4) - width / 2,
                   height=[raw_condition[key]["mean_cosine"] for key in condition_keys],
                   width=width, label="Raw SSL", color="#9ecae1")
    axes[0, 1].bar(x=np.arange(4) + width / 2,
                   height=[official_condition[key]["mean_cosine"] for key in condition_keys],
                   width=width, label="Official fine-tuned", color="#6a3d9a")
    axes[0, 1].set_xticks(np.arange(4), condition_labels, rotation=15, ha="right")
    axes[0, 1].set_ylabel("Mean same-identity cosine")
    axes[0, 1].set_title("Acquisition mismatch remains a major source of spread")
    axes[0, 1].legend()

    transition_labels = ["Fixed by official", "Regressed", "Both correct", "Both wrong"]
    transition_values = [
        transition.get("fixed_by_official", 0), transition.get("regressed_by_official", 0),
        transition.get("both_correct", 0), transition.get("both_wrong", 0),
    ]
    axes[1, 0].bar(transition_labels, transition_values,
                   color=["#31a354", "#de2d26", "#9ecae1", "#756bb1"])
    for idx, value in enumerate(transition_values):
        axes[1, 0].text(idx, value, str(value), ha="center", va="bottom")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].set_ylabel("P0 confirmation queries")
    axes[1, 0].set_title("Official fine-tuning improves local Top-1 ranking")

    mces = transition.get("official_error_mces_bins", {})
    mces_labels = ["0-2", "3-5", "6-10", ">10_or_lower_bound", "missing"]
    present = [(label, mces.get(label, 0)) for label in mces_labels if mces.get(label, 0)]
    axes[1, 1].bar([item[0] for item in present], [item[1] for item in present], color="#fdae6b")
    for idx, (_, value) in enumerate(present):
        axes[1, 1].text(idx, value, str(value), ha="center", va="bottom")
    axes[1, 1].set_xlabel("MCES to strongest wrong candidate")
    axes[1, 1].set_ylabel("Remaining official errors")
    axes[1, 1].set_title("Remaining errors are concentrated in structural neighbours")

    fig.suptitle("What still limits DreaMS after official fine-tuning?", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# DreaMS structure-similarity residual atlas",
        "",
        "**Status:** evaluation-only first pass; no model weights were changed.",
        "",
        "## Method boundary",
        "",
        "Morgan Tanimoto defines the structural reference used in DreaMS Figure 4a. "
        "The isotonic curve is fitted only on the molecule-disjoint discovery cohort; "
        "all reported residuals and error families are evaluated on confirmation molecules.",
        "Chemical rules are not used as labels or as the calibration target.",
        "",
        "## Confirmation results",
        "",
        "| Model | N pairs | Pearson r | Spearman rho | Residual MAE | Residual RMSE | Identity cosine mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in summary["confirmation_metrics"].items():
        corr = values["all_pairs"]
        lines.append(
            f"| {model} | {corr['n']:,} | {corr['pearson_r']:.4f} | "
            f"{corr['spearman_rho']:.4f} | {values['residual_mae']:.4f} | "
            f"{values['residual_rmse']:.4f} | {values['identity_cosine_mean']:.4f} |"
        )
    lines += [
        "",
        "The paper's Figure 4a reference values are zero-shot DreaMS r=0.63 and "
        "fine-tuned DreaMS r=0.70. They are not direct reproduction targets here because "
        "the dataset and sampling cohort differ; they are included only as external context.",
        "",
        "## Top-1 intersection",
        "",
    ]
    if summary["top1_intersection"]:
        for cohort, values in summary["top1_intersection"].items():
            lines.append(f"- **{cohort}:** {values['queries']:,} P0 queries have all three spectra in this cohort.")
            for model, model_values in values["models"].items():
                lines.append(
                    f"  - {model}: Top-1={model_values['top1']:.4f}; "
                    f"errors={model_values['errors']:,}; families={model_values['error_families']}"
                )
    else:
        lines.append("No P0 queries had all required spectra in the selected cohorts.")
    paired = summary.get("paired_common_support")
    if paired:
        lines += [
            "",
            "## Paired raw-vs-official comparison",
            "",
            "The historical raw-SSL cache covers only the validation fold. Therefore raw and "
            "official weights are compared separately on their exact common spectra; raw metrics "
            "are never mixed into the larger train+validation official analysis.",
            "",
            "| Model | N pairs | Pearson r | Spearman rho | Residual MAE |",
            "|---|---:|---:|---:|---:|",
        ]
        for model, values in paired["confirmation_metrics"].items():
            corr = values["all_pairs"]
            lines.append(
                f"| {model} | {corr['n']:,} | {corr['pearson_r']:.4f} | "
                f"{corr['spearman_rho']:.4f} | {values['residual_mae']:.4f} |"
            )
        lines += [
            "",
            "Residual MAE is model-scale dependent; use Pearson/Spearman and normalized residual "
            "metrics for model comparison. Raw and official embeddings can have different cosine ranges.",
        ]
        bootstrap = paired.get("paired_model_bootstrap", {}).get(
            "pearson_delta_official_minus_raw", {}
        )
        if bootstrap:
            lines += ["", "Paired Pearson deltas (official minus raw):"]
            for label, values in bootstrap.items():
                lines.append(
                    f"- {label}: {values['point']:+.4f}; "
                    f"anchor-cluster bootstrap 95% CI={values['bootstrap_95ci']}"
                )
        transitions = paired.get("top1_transitions_official_vs_raw", {})
        if transitions:
            lines += ["", "Paired Top-1 transitions:"]
            for cohort, values in transitions.items():
                lines.append(
                    f"- {cohort}: fixed={values['fixed_by_official']}, "
                    f"regressed={values['regressed_by_official']}, "
                    f"both wrong={values['both_wrong']}"
                )
    lines += [
        "",
        "## Decision rule",
        "",
        "This atlas is diagnostic. An intervention becomes trainable only when a residual "
        "family is reproducible in confirmation data and can be linked to a controlled "
        "mechanism (identity instability, local structural confusion, or peak-level causal evidence).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_top1(top1: pd.DataFrame, models: list[str]) -> dict:
    output = {}
    if top1.empty:
        return output
    for cohort, group in top1.groupby("cohort", sort=False):
        cohort_summary = {"queries": int(len(group)), "models": {}}
        for model in models:
            correct = group[f"top1_correct_{model}"].astype(bool)
            errors = group.loc[~correct, f"local_error_family_{model}"].value_counts().to_dict()
            cohort_summary["models"][model] = {
                "top1": float(correct.mean()),
                "errors": int((~correct).sum()),
                "error_families": {str(k): int(v) for k, v in errors.items()},
            }
        output[str(cohort)] = cohort_summary
    return output


def paired_top1_transitions(top1: pd.DataFrame, model_a: str, model_b: str) -> dict:
    output = {}
    if top1.empty:
        return output
    for cohort, group in top1.groupby("cohort", sort=False):
        a = group[f"top1_correct_{model_a}"].astype(bool).to_numpy()
        b = group[f"top1_correct_{model_b}"].astype(bool).to_numpy()
        official_errors = group.loc[~a]
        output[str(cohort)] = {
            "queries": int(len(group)),
            "fixed_by_official": int(np.sum(a & ~b)),
            "regressed_by_official": int(np.sum(~a & b)),
            "both_correct": int(np.sum(a & b)),
            "both_wrong": int(np.sum(~a & ~b)),
            "official_error_mces_bins": {
                str(k): int(v) for k, v in official_errors["mces_bin"].value_counts().items()
            },
        }
    return output


def run_pair_analysis(
    disc_manifest: pd.DataFrame,
    conf_manifest: pd.DataFrame,
    matrices_discovery: dict[str, np.ndarray],
    matrices_confirmation: dict[str, np.ndarray],
    args: argparse.Namespace,
    file_prefix: str,
    seed_offset: int = 0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, IsotonicRegression]]:
    discovery, discovery_sampling = sample_pairs(
        disc_manifest, args.pairs_per_bin_discovery,
        args.identity_pairs_discovery, args.seed + seed_offset,
    )
    confirmation, confirmation_sampling = sample_pairs(
        conf_manifest, args.pairs_per_bin_confirmation,
        args.identity_pairs_confirmation, args.seed + seed_offset + 1,
    )
    models = list(matrices_discovery)
    for model in models:
        discovery[f"cosine_{model}"] = rowwise_cosine(matrices_discovery[model], discovery)
        confirmation[f"cosine_{model}"] = rowwise_cosine(matrices_confirmation[model], confirmation)
    calibrators, thresholds = add_residuals(discovery, confirmation, models)

    top1_frames = []
    for cohort_name, manifest, matrices in (
        ("discovery", disc_manifest, matrices_discovery),
        ("confirmation", conf_manifest, matrices_confirmation),
    ):
        intersect = top1_intersection(
            cohort_name, manifest, matrices, calibrators, thresholds,
            args.p0_query_audit, args.mces_pairs,
        )
        if not intersect.empty:
            top1_frames.append(intersect)
    top1 = pd.concat(top1_frames, ignore_index=True) if top1_frames else pd.DataFrame()
    analysis = {
        "sampling": {"discovery": discovery_sampling, "confirmation": confirmation_sampling},
        "residual_thresholds_from_discovery": thresholds,
        "discovery_metrics": {model: model_summary(discovery, model) for model in models},
        "confirmation_metrics": {model: model_summary(confirmation, model) for model in models},
        "top1_intersection": summarize_top1(top1, models),
    }
    discovery.to_csv(args.output_dir / f"{file_prefix}discovery_structure_pairs.csv", index=False)
    confirmation.to_csv(args.output_dir / f"{file_prefix}confirmation_structure_pairs.csv", index=False)
    combined_priority = []
    for cohort_name, source, allowed_use in (
        ("discovery", discovery, "method_development_or_training"),
        ("confirmation", confirmation, "locked_validation_only"),
    ):
        priority_frames = []
        for model in models:
            family_col = f"residual_family_{model}"
            residual_col = f"residual_{model}"
            priority = source.loc[source[family_col] != "calibrated_middle"].copy()
            priority.insert(0, "model", model)
            priority.insert(0, "allowed_use", allowed_use)
            priority.insert(0, "cohort", cohort_name)
            priority["absolute_residual"] = priority[residual_col].abs()
            priority_frames.append(priority.sort_values("absolute_residual", ascending=False))
        if priority_frames:
            cohort_priority = pd.concat(priority_frames, ignore_index=True)
            cohort_priority.to_csv(
                args.output_dir / f"{file_prefix}{cohort_name}_residual_priority_cases.csv",
                index=False,
            )
            combined_priority.append(cohort_priority)
    if combined_priority:
        pd.concat(combined_priority, ignore_index=True).to_csv(
            args.output_dir / f"{file_prefix}residual_review_manifest.csv", index=False
        )
    if not top1.empty:
        top1.to_csv(args.output_dir / f"{file_prefix}p0_top1_residual_intersection.csv", index=False)
    plot_atlas(
        confirmation, models, calibrators,
        args.output_dir / f"{file_prefix}structure_residual_atlas.png",
    )
    return analysis, discovery, confirmation, top1, calibrators


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        args.pairs_per_bin_discovery = min(args.pairs_per_bin_discovery, 40)
        args.pairs_per_bin_confirmation = min(args.pairs_per_bin_confirmation, 30)
        args.identity_pairs_discovery = min(args.identity_pairs_discovery, 40)
        args.identity_pairs_confirmation = min(args.identity_pairs_confirmation, 30)
        args.n_bootstrap = min(args.n_bootstrap, 50)

    print("Loading molecule-disjoint cohorts...")
    disc_manifest, disc_official, disc_report = read_cohort(args.discovery_dir)
    conf_manifest, conf_official, conf_report = read_cohort(args.confirmation_dir)
    matrices_discovery = {"official_finetuned": l2_normalize(disc_official)}
    matrices_confirmation = {"official_finetuned": l2_normalize(conf_official)}
    raw_alignment = {}
    print("Sampling structure-balanced pairs for the full official cohort...")
    primary, _, _, _, _ = run_pair_analysis(
        disc_manifest, conf_manifest, matrices_discovery, matrices_confirmation,
        args, file_prefix="", seed_offset=0,
    )

    summary = {
        "status": "evaluation_only_structure_residual_atlas",
        "paper_context_not_direct_reproduction": {
            "zero_shot_dreams_tanimoto_pearson": 0.63,
            "finetuned_dreams_tanimoto_pearson": 0.70,
            "paper_pair_sampling": "NIST20, approximately 82k entropy-balanced spectrum pairs",
        },
        "provenance": {
            "discovery_report": disc_report,
            "confirmation_report": conf_report,
            "raw_alignment": raw_alignment,
            "fingerprint": "Morgan radius=2, 4096 bits",
            "calibration": "isotonic regression fitted on discovery molecules only",
        },
        "sampling": primary["sampling"],
        "residual_thresholds_from_discovery": primary["residual_thresholds_from_discovery"],
        "discovery_metrics": primary["discovery_metrics"],
        "confirmation_metrics": primary["confirmation_metrics"],
        "top1_intersection": primary["top1_intersection"],
    }

    if not args.skip_raw:
        print("Building an exact common-support raw-vs-official comparison...")
        disc_raw, disc_raw_audit = align_raw_embeddings(disc_manifest, args.raw_cache_dir)
        conf_raw, conf_raw_audit = align_raw_embeddings(conf_manifest, args.raw_cache_dir)
        raw_alignment = {"discovery": disc_raw_audit, "confirmation": conf_raw_audit}
        summary["provenance"]["raw_alignment"] = raw_alignment
        disc_valid = np.isfinite(disc_raw).all(axis=1)
        conf_valid = np.isfinite(conf_raw).all(axis=1)
        paired_disc_manifest = disc_manifest.loc[disc_valid].reset_index(drop=True)
        paired_conf_manifest = conf_manifest.loc[conf_valid].reset_index(drop=True)
        paired_disc_matrices = {
            "official_finetuned": l2_normalize(np.asarray(disc_official[disc_valid])),
            "raw_ssl": disc_raw[disc_valid],
        }
        paired_conf_matrices = {
            "official_finetuned": l2_normalize(np.asarray(conf_official[conf_valid])),
            "raw_ssl": conf_raw[conf_valid],
        }
        paired, _, paired_confirmation, paired_top1, _ = run_pair_analysis(
            paired_disc_manifest, paired_conf_manifest,
            paired_disc_matrices, paired_conf_matrices,
            args, file_prefix="paired_common_", seed_offset=100,
        )
        paired["paired_model_bootstrap"] = paired_anchor_bootstrap(
            paired_confirmation, "official_finetuned", "raw_ssl",
            args.n_bootstrap, args.seed + 500,
        )
        paired["top1_transitions_official_vs_raw"] = paired_top1_transitions(
            paired_top1, "official_finetuned", "raw_ssl"
        )
        summary["paired_common_support"] = paired

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, args.output_dir / "REPORT.md")
    plot_mechanism_summary(summary, args.output_dir / "mechanism_summary.png")

    print(f"Saved residual atlas to {args.output_dir}")
    for model, values in summary["confirmation_metrics"].items():
        corr = values["all_pairs"]
        print(
            f"  {model}: Pearson={corr['pearson_r']:.4f}, "
            f"Spearman={corr['spearman_rho']:.4f}, residual MAE={values['residual_mae']:.4f}"
        )


if __name__ == "__main__":
    main()

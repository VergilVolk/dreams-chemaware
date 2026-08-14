"""Large, formula-clustered observability–DreaMS residual audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_e0_observability_residual import pair_features


FEATURES = [
    "sqrt_cosine", "linear_cosine", "entropy_similarity",
    "intensity_coverage_min", "intensity_coverage_mean",
    "matched_peak_fraction_min", "top10_match_fraction",
    "neutral_loss_sqrt_cosine", "neutral_loss_coverage_min",
    "neutral_loss_coverage_mean", "peak_count_ratio",
]
CONSENSUS = [
    "entropy_similarity", "sqrt_cosine", "linear_cosine",
    "top10_match_fraction", "intensity_coverage_min",
]


def symmetric_features(
    spectrum_a: np.ndarray, precursor_a: float,
    spectrum_b: np.ndarray, precursor_b: float, tolerance: float,
) -> dict[str, float]:
    raw = pair_features(spectrum_a, precursor_a, spectrum_b, precursor_b, tolerance)
    return {
        "sqrt_cosine": raw["sqrt_cosine"],
        "linear_cosine": raw["linear_cosine"],
        "entropy_similarity": raw["entropy_similarity"],
        "intensity_coverage_min": min(raw["query_intensity_coverage"], raw["candidate_intensity_coverage"]),
        "intensity_coverage_mean": 0.5 * (raw["query_intensity_coverage"] + raw["candidate_intensity_coverage"]),
        "matched_peak_fraction_min": raw["matched_peak_fraction_min"],
        "top10_match_fraction": raw["top10_match_fraction"],
        "neutral_loss_sqrt_cosine": raw["neutral_loss_sqrt_cosine"],
        "neutral_loss_coverage_min": min(raw["neutral_loss_query_coverage"], raw["neutral_loss_candidate_coverage"]),
        "neutral_loss_coverage_mean": 0.5 * (raw["neutral_loss_query_coverage"] + raw["neutral_loss_candidate_coverage"]),
        "peak_count_ratio": raw["peak_count_ratio"],
    }


def load_embeddings(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(directory / "manifest.csv")
    values = np.load(directory / "official_embeddings.npy").astype(np.float32)
    if len(manifest) != len(values):
        raise RuntimeError(f"Embedding alignment failure in {directory}")
    values /= np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    return manifest, values


def build_pairs(
    manifest: pd.DataFrame, embeddings: np.ndarray, data: Path,
    tolerance: float, ppm: float,
) -> pd.DataFrame:
    rows = manifest["hdf5_row"].to_numpy(np.int64)
    with h5py.File(data, "r") as handle:
        order = np.argsort(rows)
        inverse = np.argsort(order)
        spectra = np.asarray(handle["spectrum"][rows[order]])[inverse]
    output = []
    formulas = list(manifest.groupby("formula", sort=True))
    for position, (formula, group) in enumerate(formulas, start=1):
        idx = group.index.to_numpy(np.int64)
        for left_pos in range(len(idx) - 1):
            i = int(idx[left_pos])
            for right_pos in range(left_pos + 1, len(idx)):
                j = int(idx[right_pos])
                mass = 0.5 * (float(manifest.at[i, "precursor_mz"]) + float(manifest.at[j, "precursor_mz"]))
                delta_ppm = abs(float(manifest.at[i, "precursor_mz"]) - float(manifest.at[j, "precursor_mz"])) / mass * 1e6
                if delta_ppm > ppm or manifest.at[i, "spectrum_hash"] == manifest.at[j, "spectrum_hash"]:
                    continue
                features = symmetric_features(
                    spectra[i], float(manifest.at[i, "precursor_mz"]),
                    spectra[j], float(manifest.at[j, "precursor_mz"]), tolerance,
                )
                output.append({
                    "split": manifest.at[i, "audit_split"], "formula": formula,
                    "left": i, "right": j,
                    "left_ik14": manifest.at[i, "ik14"], "right_ik14": manifest.at[j, "ik14"],
                    "label": int(manifest.at[i, "ik14"] == manifest.at[j, "ik14"]),
                    "precursor_delta_ppm": delta_ppm,
                    "dreams_similarity": float(embeddings[i] @ embeddings[j]),
                } | features)
        if position % 100 == 0 or position == len(formulas):
            print(f"  pair features: {position}/{len(formulas)} formulas; {len(output):,} pairs", flush=True)
    return pd.DataFrame(output)


def fit_raw_proxy(frame: pd.DataFrame):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight=None, max_iter=3000, random_state=20260813),
    )
    counts = frame.groupby(["formula", "label"])["left"].transform("size").astype(float)
    formula_count = frame["formula"].nunique()
    weights = 1.0 / (2.0 * formula_count * counts)
    weights *= len(weights) / weights.sum()
    model.fit(frame[FEATURES], frame["label"], logisticregression__sample_weight=weights)
    return model


def score_lookup(pair_table: pd.DataFrame, column: str) -> dict[tuple[int, int], float]:
    lookup = {}
    for row in pair_table[["left", "right", column]].itertuples(index=False):
        lookup[(int(row.left), int(row.right))] = float(getattr(row, column))
    return lookup


def retrieval(manifest: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    score_columns = ["dreams_similarity", "raw_proxy_score"] + CONSENSUS
    lookups = {column: score_lookup(pair_table, column) for column in score_columns}
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(manifest))}
    for row in pair_table[["left", "right"]].itertuples(index=False):
        adjacency[int(row.left)].append(int(row.right))
        adjacency[int(row.right)].append(int(row.left))
    rows = []
    for query in range(len(manifest)):
        candidates = adjacency[query]
        if not candidates:
            continue
        positive = [j for j in candidates if manifest.at[j, "ik14"] == manifest.at[query, "ik14"]]
        negative = [j for j in candidates if manifest.at[j, "ik14"] != manifest.at[query, "ik14"]]
        if not positive or not negative:
            continue
        row = {
            "split": manifest.at[query, "audit_split"], "query_index": query,
            "hdf5_row": int(manifest.at[query, "hdf5_row"]),
            "ik14": manifest.at[query, "ik14"], "formula": manifest.at[query, "formula"],
            "smiles": manifest.at[query, "smiles"], "ring_class": manifest.at[query, "ring_class"],
            "n_positive_spectra": len(positive),
            "n_negative_molecules": len({manifest.at[j, "ik14"] for j in negative}),
        }
        for column in score_columns:
            lookup = lookups[column]
            def value(j: int) -> float:
                return lookup[(min(query, j), max(query, j))]
            best_positive_index = max(positive, key=value)
            pos = value(best_positive_index)
            molecule_best: dict[str, tuple[float, int]] = {}
            for j in negative:
                key, score = manifest.at[j, "ik14"], value(j)
                if key not in molecule_best or score > molecule_best[key][0]:
                    molecule_best[key] = (score, j)
            best_ik, (best_neg, best_j) = max(molecule_best.items(), key=lambda item: item[1][0])
            prefix = "dreams" if column == "dreams_similarity" else ("raw" if column == "raw_proxy_score" else column)
            row[f"{prefix}_positive"] = pos
            row[f"{prefix}_best_negative"] = best_neg
            row[f"{prefix}_margin"] = pos - best_neg
            row[f"{prefix}_top1_correct"] = bool(pos > best_neg)
            row[f"{prefix}_best_positive_index"] = int(best_positive_index)
            row[f"{prefix}_best_negative_index"] = int(best_j)
            if column in ("dreams_similarity", "raw_proxy_score"):
                all_negative_scores = np.asarray([item[0] for item in molecule_best.values()])
                row[f"{prefix}_pairwise_accuracy"] = float(np.mean(pos > all_negative_scores))
                row[f"{prefix}_best_negative_ik14"] = best_ik
                row[f"{prefix}_best_negative_smiles"] = manifest.at[best_j, "smiles"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    vote_columns = [f"{feature}_top1_correct" for feature in CONSENSUS]
    frame["raw_metric_consensus_votes"] = frame[vote_columns].sum(axis=1).astype(int)
    frame["audit_quadrant"] = np.select(
        [
            frame["dreams_top1_correct"] & frame["raw_top1_correct"],
            (~frame["dreams_top1_correct"]) & frame["raw_top1_correct"],
            (~frame["dreams_top1_correct"]) & (~frame["raw_top1_correct"]),
        ],
        ["both_correct", "model_residual_candidate", "shared_or_spectrum_limited"],
        default="dreams_only_correct",
    )
    frame["robust_model_residual_candidate"] = (
        (~frame["dreams_top1_correct"]) & (frame["raw_metric_consensus_votes"] >= 3)
    )
    return frame


def formula_bootstrap(frame: pd.DataFrame, iterations: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    summary = frame.groupby("formula", sort=False).agg(
        n=("query_index", "size"), raw=("raw_top1_correct", "sum"),
        dreams=("dreams_top1_correct", "sum"),
    )
    n = summary["n"].to_numpy(float)
    difference = (summary["raw"] - summary["dreams"]).to_numpy(float)
    draws = np.empty(iterations, dtype=float)
    formula_count = len(summary)
    # Chunked vectorization avoids materializing a very large index matrix.
    for start in range(0, iterations, 500):
        stop = min(start + 500, iterations)
        indices = rng.integers(0, formula_count, size=(stop - start, formula_count))
        draws[start:stop] = difference[indices].sum(axis=1) / n[indices].sum(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def summarize(frame: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    hard_scores = np.r_[frame["dreams_positive"], frame["dreams_best_negative"]]
    hard_labels = np.r_[np.ones(len(frame)), np.zeros(len(frame))]
    molecule = frame.groupby("ik14", sort=False).agg(
        dreams_failed_any=("dreams_top1_correct", lambda x: bool((~x).any())),
        robust_residual_any=("robust_model_residual_candidate", "any"),
    )
    return {
        "query_spectra": len(frame), "molecules": int(frame["ik14"].nunique()),
        "formulas": int(frame["formula"].nunique()),
        "dreams_hard_negative_roc_auc": float(roc_auc_score(hard_labels, hard_scores)),
        "dreams_top1": float(frame["dreams_top1_correct"].mean()),
        "dreams_pairwise_accuracy": float(frame["dreams_pairwise_accuracy"].mean()),
        "raw_proxy_top1": float(frame["raw_top1_correct"].mean()),
        "raw_minus_dreams_top1": float(frame["raw_top1_correct"].mean() - frame["dreams_top1_correct"].mean()),
        "raw_minus_dreams_formula_bootstrap_ci95": formula_bootstrap(frame, bootstrap, seed),
        "view_quadrants": {str(k): int(v) for k, v in frame["audit_quadrant"].value_counts().items()},
        "molecules_dreams_failed_any": int(molecule["dreams_failed_any"].sum()),
        "robust_residual_query_spectra": int(frame["robust_model_residual_candidate"].sum()),
        "robust_residual_molecules": int(molecule["robust_residual_any"].sum()),
        "robust_residual_formulas": int(frame.loc[frame["robust_model_residual_candidate"], "formula"].nunique()),
        "pearson_margin_correlation": float(frame[["dreams_margin", "raw_margin"]].corr().iloc[0, 1]),
        "spearman_margin_correlation": float(frame[["dreams_margin", "raw_margin"]].corr(method="spearman").iloc[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--discovery-embeddings", type=Path, default=Path("data/validation/large_observability_embeddings_discovery"))
    parser.add_argument("--confirmation-embeddings", type=Path, default=Path("data/validation/large_observability_embeddings_confirmation"))
    parser.add_argument("--test-embeddings", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests, embeddings, pairs = {}, {}, {}
    split_directories = [
        ("discovery", args.discovery_embeddings),
        ("confirmation", args.confirmation_embeddings),
    ]
    if args.test_embeddings is not None:
        split_directories.append(("test", args.test_embeddings))
    for split, directory in split_directories:
        manifests[split], embeddings[split] = load_embeddings(directory)
        cache = args.output_dir / f"{split}_pair_features.csv"
        if cache.exists():
            pairs[split] = pd.read_csv(cache)
            print(f"  reused {cache}: {len(pairs[split]):,} pairs", flush=True)
        else:
            pairs[split] = build_pairs(manifests[split], embeddings[split], args.data, args.tolerance, args.ppm)
            pairs[split].to_csv(cache, index=False)
    model = fit_raw_proxy(pairs["discovery"])
    for split in pairs:
        pairs[split]["raw_proxy_score"] = model.predict_proba(pairs[split][FEATURES])[:, 1]
    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "standardized_logistic_coefficient": model.named_steps["logisticregression"].coef_[0],
    }).sort_values("standardized_logistic_coefficient", ascending=False)
    coefficients.to_csv(args.output_dir / "raw_proxy_coefficients.csv", index=False)
    queries = {}
    for split in pairs:
        queries[split] = retrieval(manifests[split], pairs[split])
        queries[split].to_csv(args.output_dir / f"{split}_query_audit.csv", index=False)
    report = {
        "status": "large_observability_residual_audit",
        "protocol": "[M+H]+, same formula, measured precursor difference <=10 ppm, duplicate hashes excluded",
        "raw_proxy_fit": "Discovery formulas only; formula-and-class balanced sample weights.",
        "splits": {
            split: summarize(frame, args.bootstrap, 20260813 + i)
            for i, (split, frame) in enumerate(queries.items())
        },
        "test_split": (
            "Consumed once after freezing the evidence panel."
            if "test" in queries else "Untouched and not encoded."
        ),
        "claim_limit": (
            "MassSpecGym-derived failure discovery. A raw proxy success marks a model-residual candidate, "
            "not proof that a structural edit is identifiable or causal; peak-level validation remains required."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

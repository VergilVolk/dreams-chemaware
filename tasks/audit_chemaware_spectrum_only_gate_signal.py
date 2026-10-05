"""Audit whether ChemAware corrections versus harms are spectrum-predictable."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_chemaware_mass_kernel_embedding import KernelCache
from audit_chemaware_whitened_centered_rule_kernel import FourCenterCache, fit_transforms
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_shrinkage_whitening_core import apply_whitener
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-ranks", type=Path,
        default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_spectrum_only_gate_signal_v1",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--final-fit-identities", type=int, default=8192)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--null-draws", type=int, default=100)
    return parser.parse_args()


def spectrum_scalars(cache: KernelCache, rows: np.ndarray) -> np.ndarray:
    output = np.empty((len(rows), 9), dtype=np.float32)
    for index, row in enumerate(map(int, rows)):
        position = cache.row_position[row]
        valid = np.asarray(cache.valid[position], dtype=bool)
        mz = np.asarray(cache.mz[position][valid], dtype=np.float64)
        intensity = np.asarray(cache.intensity[position][valid], dtype=np.float64)
        intensity = np.maximum(intensity, 0.0)
        total = max(float(intensity.sum()), 1e-12)
        probability = intensity / total
        mean_mz = float(np.sum(probability * mz)) if len(mz) else 0.0
        output[index] = (
            float(cache.precursor[position]), float(len(mz)),
            float(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))),
            float(np.max(probability)) if len(probability) else 0.0,
            mean_mz,
            float(np.sqrt(np.sum(probability * (mz - mean_mz) ** 2))) if len(mz) else 0.0,
            float(np.sum(probability[: min(5, len(probability))])),
            float(np.min(mz)) if len(mz) else 0.0,
            float(np.max(mz)) if len(mz) else 0.0,
        )
    return output


def oof_probability(
    feature: np.ndarray,
    label: np.ndarray,
    group: np.ndarray,
    pca_dim: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    unique_groups = np.unique(group)
    folds = min(5, len(unique_groups))
    if folds < 3:
        raise ValueError("spectrum gate audit needs at least three formula groups")
    probability = np.empty(len(label), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train, test in splitter.split(feature, label, group):
        if len(np.unique(label[train])) != 2:
            raise RuntimeError("one gate-training fold contains a single outcome class")
        components = min(pca_dim, feature.shape[1], len(train) - 2)
        steps = [StandardScaler()]
        if feature.shape[1] > components:
            steps.append(PCA(n_components=components, whiten=True, random_state=seed))
        steps.append(LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=seed,
        ))
        model = make_pipeline(*steps)
        model.fit(feature[train], label[train])
        probability[test] = model.predict_proba(feature[test])[:, 1]
    return probability, folds


def evaluate_feature_set(
    feature: np.ndarray,
    label: np.ndarray,
    group: np.ndarray,
    args: argparse.Namespace,
    offset: int,
) -> dict[str, object]:
    probability, folds = oof_probability(feature, label, group, args.pca_dim, args.seed + offset)
    auc = float(roc_auc_score(label, probability))
    ap = float(average_precision_score(label, probability))
    null_auc = []
    rng = np.random.default_rng(args.seed + 1000 + offset)
    for draw in range(args.null_draws):
        permuted = label[rng.permutation(len(label))]
        null_probability, _ = oof_probability(
            feature, permuted, group, args.pca_dim, args.seed + 2000 + offset + draw,
        )
        null_auc.append(float(roc_auc_score(permuted, null_probability)))
    null_array = np.asarray(null_auc)
    return {
        "formula_grouped_oof_folds": int(folds),
        "roc_auc_helpful_vs_harmful": auc,
        "average_precision_helpful": ap,
        "helpful_prevalence": float(label.mean()),
        "null_auc_quantiles": np.quantile(null_array, (0.0, 0.5, 0.90, 0.95, 1.0)).astype(float).tolist(),
        "empirical_one_sided_p": float((1 + np.sum(null_array >= auc)) / (1 + len(null_array))),
        "beats_null_95th_percentile": bool(auc > np.quantile(null_array, 0.95)),
        "probability_quantiles_helpful": np.quantile(
            probability[label == 1], (0, 0.25, 0.5, 0.75, 1),
        ).astype(float).tolist(),
        "probability_quantiles_harmful": np.quantile(
            probability[label == 0], (0, 0.25, 0.5, 0.75, 1),
        ).astype(float).tolist(),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    final_fit = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"],
        np.random.default_rng(20260905 + 23), args.final_fit_identities,
    )
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library,
        top_peaks=32, kernel_dim=2048, bin_width=0.02, grid_offsets=4,
        intensity_power=0.5, mass_shift_da=0.137, pair_weight=0.25,
        multi_bin_widths=(0.01, 0.02, 0.05), uniform_channel_weight=1.0,
        rule_tolerance=0.02, rule_channel_weight=1.0,
    )
    base = KernelCache(
        kernel_args, row_position,
        variants=(
            "mass", "rule_response", "rule_response_local_background_a",
            "rule_response_local_background_b", "rule_response_local_background_c",
        ),
    )
    centers = FourCenterCache(base)
    transforms, variants, fit_report = fit_transforms(final_fit, body, centers, (0.50,))
    variant = "whitened_true_s0p5"
    mean, transform = transforms[variant]
    with np.load(args.parent_ranks, allow_pickle=False) as parent:
        query = np.asarray(parent["query"], dtype=np.int64)
        formula = np.asarray(parent["formula"]).astype(str)
        baseline_rank = np.asarray(parent["baseline_rank"])
        chemical_rank = np.asarray(parent["whitened_true_rank"])
    query_rows = np.asarray(body["query_row"][query], dtype=np.int64)
    positions = np.asarray([row_position[int(row)] for row in query_rows], dtype=np.int64)
    corrected = (baseline_rank > 1) & (chemical_rank == 1)
    introduced = (baseline_rank == 1) & (chemical_rank > 1)
    changed = corrected | introduced
    if int(corrected.sum()) != 74 or int(introduced.sum()) != 26:
        raise RuntimeError("frozen parent correction/harm counts changed")
    changed_rows = query_rows[changed]
    official_feature = np.asarray(official[positions[changed]], dtype=np.float32)
    centered_feature = np.stack([
        np.asarray(centers.get(int(row))["true"], dtype=np.float32) for row in changed_rows
    ])
    whitened_feature = apply_whitener(centered_feature, mean, transform)
    scalar = spectrum_scalars(base, changed_rows)
    center_norm = np.linalg.norm(centered_feature, axis=1, keepdims=True)
    scalar = np.concatenate((scalar, center_norm), axis=1)
    label = corrected[changed].astype(np.int8)
    group = formula[changed]
    feature_sets = {
        "spectrum_scalars": scalar,
        "chemical_rule_response": np.concatenate((whitened_feature, scalar), axis=1),
        "official_dreams": np.concatenate((official_feature, scalar), axis=1),
        "combined": np.concatenate((official_feature, whitened_feature, scalar), axis=1),
    }
    results = {}
    for offset, (name, feature) in enumerate(feature_sets.items()):
        print(f"auditing spectrum-only gate feature set: {name}", flush=True)
        results[name] = evaluate_feature_set(feature, label, group, args, 100 * offset)
    best_name = max(results, key=lambda name: results[name]["roc_auc_helpful_vs_harmful"])
    report = {
        "status": "CHEMAWARE_SPECTRUM_ONLY_GATE_SIGNAL_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "exploratory formula-grouped OOF predictability audit inside already-used inner fold 3; outer fold 4 sealed",
        "claim_limit": "This tests gate learnability only; it is not a deployable gated embedding or a performance claim.",
        "outcomes": {
            "queries": int(len(query)), "corrected": int(corrected.sum()),
            "introduced": int(introduced.sum()), "changed": int(changed.sum()),
        },
        "features": {
            "candidate_formula_identity_or_rank_used": False,
            "single_clean_spectrum_only": True,
            "feature_dimensions": {name: int(value.shape[1]) for name, value in feature_sets.items()},
            "classifier": "standardize + at-most-16D PCA + fixed-C balanced logistic regression",
        },
        "results": results,
        "best_feature_set": best_name,
        "gate_development_eligible": bool(results[best_name]["beats_null_95th_percentile"]),
        "outer_fold_untouched": True,
        "whitener_fit": fit_report,
        "provenance": {
            "parent_ranks_sha256": sha256_file(args.parent_ranks),
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_gate_signal_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "outcomes": report["outcomes"],
        "results": results, "best_feature_set": best_name,
        "gate_development_eligible": report["gate_development_eligible"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

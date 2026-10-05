"""Audit whether a clean query spectrum predicts rule-over-mass benefit.

This is a necessary pre-training gate, not an embedding result.  Model inputs
are built independently from one unmodified query spectrum: a fixed projection
of the official DreaMS embedding, a fixed projection of the mass kernel, the
rule-response vector, and elementary spectrum descriptors.  Candidate scores,
ranks, molecule identities, and molecular formulae are forbidden as features.

In retrospective mode one rank ledger is evaluated by formula-grouped OOF and
can never authorize GPU training.  Formal mode additionally requires a frozen,
formula-disjoint confirmation ledger; discovery OOF selects the routing
threshold once and the fitted discovery model is evaluated once on confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-ranks", type=Path,
        default=ROOT / "data/validation/chemaware_rule_mass_pairfirst_full_inner_v1/inner_per_query.npz",
    )
    parser.add_argument("--confirmation-ranks", type=Path, default=None)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--permutation-controls", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--official-projection-dim", type=int, default=48)
    parser.add_argument("--mass-projection-dim", type=int, default=48)
    parser.add_argument("--min-discovery-positive", type=int, default=50)
    parser.add_argument("--min-discovery-harmful", type=int, default=20)
    parser.add_argument("--min-confirmation-positive", type=int, default=20)
    parser.add_argument("--min-confirmation-harmful", type=int, default=8)
    parser.add_argument("--min-selected-formulas", type=int, default=25)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    return parser.parse_args()


def random_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(input_dim, output_dim)).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
    return matrix


def spectrum_descriptors(mz: np.ndarray, intensity: np.ndarray, precursor: float) -> np.ndarray:
    valid = (np.asarray(mz) > 0) & (np.asarray(intensity) > 0)
    mz = np.asarray(mz, dtype=np.float64)[valid]
    intensity = np.asarray(intensity, dtype=np.float64)[valid]
    if not len(mz):
        return np.zeros(26, dtype=np.float32)
    intensity /= max(float(intensity.max()), 1e-12)
    probability = intensity / max(float(intensity.sum()), 1e-12)
    losses = float(precursor) - mz
    positive_losses = losses[losses > 0]
    loss_quantile = np.quantile(positive_losses, [0.10, 0.25, 0.50, 0.75, 0.90]) if len(positive_losses) else np.zeros(5)
    top = np.sort(intensity)[::-1]
    output = np.asarray([
        float(precursor), float(len(mz)), float(mz.mean()), float(mz.std()),
        *np.quantile(mz, [0.10, 0.25, 0.50, 0.75, 0.90]),
        float(intensity.mean()), float(intensity.std()),
        *np.quantile(intensity, [0.10, 0.25, 0.50, 0.75, 0.90]),
        -float(np.sum(probability * np.log(np.clip(probability, 1e-12, None)))),
        float(top[0]), float(top[: min(5, len(top))].sum() / max(float(intensity.sum()), 1e-12)),
        float(positive_losses.mean()) if len(positive_losses) else 0.0,
        float(positive_losses.std()) if len(positive_losses) else 0.0,
        *loss_quantile,
    ], dtype=np.float32)
    if len(output) != 26 or not np.all(np.isfinite(output)):
        raise RuntimeError("clean spectrum descriptor construction failed")
    return output


def formula_weights(formulas: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(np.asarray(formulas, dtype=str), return_inverse=True)
    counts = np.bincount(inverse)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def fit_binary(features: np.ndarray, target: np.ndarray, formulas: np.ndarray, seed: int):
    target = np.asarray(target, dtype=np.int8)
    if len(np.unique(target)) < 2:
        return float(np.mean(target))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000, solver="liblinear", random_state=seed),
    )
    model.fit(features, target, logisticregression__sample_weight=formula_weights(formulas))
    return model


def predict_binary(model, features: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(features), model, dtype=np.float64)
    return model.predict_proba(features)[:, 1]


def crossfit_predictions(
    features: np.ndarray, formulas: np.ndarray, positive: np.ndarray,
    harmful: np.ndarray, folds: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    unique_formulas = np.unique(formulas)
    if len(unique_formulas) < folds:
        raise ValueError("fewer formula groups than requested folds")
    positive_probability = np.empty(len(features), dtype=np.float64)
    harmful_probability = np.empty(len(features), dtype=np.float64)
    reports = []
    split = GroupKFold(n_splits=folds)
    for fold, (train, held) in enumerate(split.split(features, groups=formulas)):
        positive_model = fit_binary(features[train], positive[train], formulas[train], seed + fold)
        harmful_model = fit_binary(features[train], harmful[train], formulas[train], seed + 100 + fold)
        positive_probability[held] = predict_binary(positive_model, features[held])
        harmful_probability[held] = predict_binary(harmful_model, features[held])
        reports.append({
            "fold": fold, "train_queries": int(len(train)), "held_queries": int(len(held)),
            "train_formulas": int(len(np.unique(formulas[train]))),
            "held_formulas": int(len(np.unique(formulas[held]))),
            "formula_overlap": int(len(set(formulas[train]) & set(formulas[held]))),
        })
    return positive_probability, harmful_probability, reports


def safe_auprc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(target, dtype=np.int8), probability))


def choose_threshold(
    formulas: np.ndarray, mass_hit: np.ndarray, rule_hit: np.ndarray,
    positive_probability: np.ndarray, harmful_probability: np.ndarray,
    min_selected_formulas: int,
) -> tuple[float, dict]:
    candidates = np.r_[np.linspace(0.05, 0.95, 91), 1.01]
    rows = []
    for threshold in candidates:
        active = (positive_probability >= threshold) & (harmful_probability <= 1.0 - threshold)
        selected_formulas = len(np.unique(formulas[active]))
        final = np.where(active, rule_hit, mass_hit)
        corrected = int(np.sum((~mass_hit) & final))
        introduced = int(np.sum(mass_hit & (~final)))
        rows.append({
            "threshold": float(threshold), "selected": int(np.sum(active)),
            "selected_formulas": int(selected_formulas), "corrected": corrected,
            "introduced": introduced, "risk_utility": corrected - 2 * introduced,
            "delta_recall1": float(np.mean(final.astype(float) - mass_hit.astype(float))),
        })
    eligible = [row for row in rows if row["selected_formulas"] >= min_selected_formulas]
    if not eligible:
        eligible = [row for row in rows if row["selected"] == 0]
    best = max(eligible, key=lambda row: (row["risk_utility"], -row["introduced"], row["corrected"], row["threshold"]))
    return float(best["threshold"]), best


def route_outcome(
    formulas: np.ndarray, mass_hit: np.ndarray, rule_hit: np.ndarray,
    positive_probability: np.ndarray, harmful_probability: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    active = (positive_probability >= threshold) & (harmful_probability <= 1.0 - threshold)
    final = np.where(active, rule_hit, mass_hit)
    corrected = int(np.sum((~mass_hit) & final)); introduced = int(np.sum(mass_hit & (~final)))
    return ({
        "queries": int(len(final)), "selected": int(np.sum(active)),
        "selected_formulas": int(len(np.unique(formulas[active]))),
        "corrected": corrected, "introduced": introduced,
        "risk_utility": corrected - 2 * introduced,
        "mass_recall1": float(np.mean(mass_hit)), "routed_recall1": float(np.mean(final)),
        "delta_recall1": float(np.mean(final.astype(float) - mass_hit.astype(float))),
    }, final)


def formula_bootstrap_delta(
    formulas: np.ndarray, mass_hit: np.ndarray, final_hit: np.ndarray,
    draws: int, seed: int,
) -> list[float]:
    unique, inverse = np.unique(formulas, return_inverse=True)
    sums = np.bincount(inverse, weights=final_hit.astype(float) - mass_hit.astype(float))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed); values = np.empty(draws, dtype=np.float64)
    for left in range(0, draws, 500):
        right = min(left + 500, draws)
        selected = rng.integers(0, len(unique), size=(right - left, len(unique)))
        values[left:right] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def labels_from_ranks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        required = {"query", "formula", "mass_rank", "rule_mass_rank"}
        missing = required - set(loaded.files)
        if missing:
            raise RuntimeError(f"rank ledger misses {sorted(missing)}: {path}")
        output = {key: np.array(loaded[key], copy=True) for key in required}
    output["formula"] = output["formula"].astype(str)
    output["mass_hit"] = output["mass_rank"] == 1
    output["rule_hit"] = output["rule_mass_rank"] == 1
    output["positive"] = (~output["mass_hit"]) & output["rule_hit"]
    output["harmful"] = output["mass_hit"] & (~output["rule_hit"])
    return output


def build_clean_features(
    ledger: dict[str, np.ndarray], body: dict[str, np.ndarray], rows: np.ndarray,
    official: np.ndarray, args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    # Keep the pure OOF/statistical audit importable on CPU-only machines.
    # KernelCache itself needs torch and is required only for real feature
    # extraction, not for audit_arrays or its contract tests.
    from audit_chemaware_mass_kernel_embedding import KernelCache  # noqa: PLC0415

    row_position = {int(row): index for index, row in enumerate(rows)}
    query = ledger["query"].astype(np.int64)
    query_rows = body["query_row"][query].astype(np.int64)
    positions = np.asarray([row_position[int(row)] for row in query_rows], dtype=np.int64)
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library, top_peaks=args.top_peaks,
        kernel_dim=args.kernel_dim, bin_width=args.bin_width, grid_offsets=args.grid_offsets,
        intensity_power=args.intensity_power, mass_shift_da=args.mass_shift_da,
        pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=args.rule_tolerance,
        rule_channel_weight=1.0,
    )
    cache = KernelCache(kernel_args, row_position, variants=("mass", "rule_response"))
    official_projection = random_projection(official.shape[1], args.official_projection_dim, args.seed + 11)
    mass_projection = random_projection(args.kernel_dim, args.mass_projection_dim, args.seed + 12)
    features = []
    for index, (row, position) in enumerate(zip(query_rows, positions)):
        vectors = cache.get(int(row))
        token_position = row_position[int(row)]
        valid = np.asarray(cache.valid[token_position], dtype=bool)
        raw = spectrum_descriptors(
            np.asarray(cache.mz[token_position][valid]), np.asarray(cache.intensity[token_position][valid]),
            float(cache.precursor[token_position]),
        )
        features.append(np.concatenate((
            np.asarray(official[position], dtype=np.float32) @ official_projection,
            vectors["mass"].astype(np.float32) @ mass_projection,
            vectors["rule_response"].astype(np.float32), raw,
        )).astype(np.float32))
        if (index + 1) % 512 == 0:
            print(f"clean features {index + 1}/{len(query_rows)}", flush=True)
    matrix = np.stack(features)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("non-finite clean feature")
    feature_names = (
        [f"official_embedding_fixed_rp_{i}" for i in range(args.official_projection_dim)]
        + [f"mass_kernel_fixed_rp_{i}" for i in range(args.mass_projection_dim)]
        + [f"rule_response_{i}" for i in range(matrix.shape[1] - args.official_projection_dim - args.mass_projection_dim - 26)]
        + [f"raw_spectrum_descriptor_{i}" for i in range(26)]
    )
    return matrix, feature_names


def audit_arrays(
    discovery_features: np.ndarray, discovery: dict[str, np.ndarray],
    confirmation_features: np.ndarray | None, confirmation: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray]]:
    dp, dh, fold_reports = crossfit_predictions(
        discovery_features, discovery["formula"], discovery["positive"], discovery["harmful"],
        args.folds, args.seed,
    )
    threshold, threshold_selection = choose_threshold(
        discovery["formula"], discovery["mass_hit"], discovery["rule_hit"], dp, dh,
        args.min_selected_formulas,
    )
    discovery_route, discovery_final = route_outcome(
        discovery["formula"], discovery["mass_hit"], discovery["rule_hit"], dp, dh, threshold,
    )
    discovery_metrics = {
        "positive_prevalence": float(np.mean(discovery["positive"])),
        "harmful_prevalence": float(np.mean(discovery["harmful"])),
        "positive_auprc": safe_auprc(discovery["positive"], dp),
        "harmful_auprc": safe_auprc(discovery["harmful"], dh),
    }
    report: dict = {
        "mode": "formal_discovery_confirmation" if confirmation is not None else "retrospective_oof_only",
        "discovery": {
            "queries": int(len(dp)), "formulas": int(len(np.unique(discovery["formula"]))),
            "positive_events": int(np.sum(discovery["positive"])),
            "harmful_events": int(np.sum(discovery["harmful"])),
            "prediction": discovery_metrics, "threshold_selection": threshold_selection,
            "route": discovery_route,
            "route_formula_bootstrap_ci95": formula_bootstrap_delta(
                discovery["formula"], discovery["mass_hit"], discovery_final,
                args.bootstrap_draws, args.seed + 300,
            ),
        },
        "folds": fold_reports,
    }
    arrays = {"discovery_positive_probability": dp, "discovery_harmful_probability": dh}
    if confirmation is None or confirmation_features is None:
        report["status"] = "RETROSPECTIVE_DIAGNOSTIC_ONLY"
        report["gates"] = {
            "discovery_positive_events_sufficient": int(np.sum(discovery["positive"])) >= args.min_discovery_positive,
            "discovery_harmful_events_sufficient": int(np.sum(discovery["harmful"])) >= args.min_discovery_harmful,
            "formula_disjoint_confirmation_present": False,
        }
        report["pass_to_gpu_training"] = False
        return report, arrays

    overlap = set(discovery["formula"]) & set(confirmation["formula"])
    positive_model = fit_binary(discovery_features, discovery["positive"], discovery["formula"], args.seed + 500)
    harmful_model = fit_binary(discovery_features, discovery["harmful"], discovery["formula"], args.seed + 501)
    cp = predict_binary(positive_model, confirmation_features)
    ch = predict_binary(harmful_model, confirmation_features)
    confirmation_route, confirmation_final = route_outcome(
        confirmation["formula"], confirmation["mass_hit"], confirmation["rule_hit"], cp, ch, threshold,
    )
    observed_positive_auprc = safe_auprc(confirmation["positive"], cp)
    observed_harmful_auprc = safe_auprc(confirmation["harmful"], ch)
    rng = np.random.default_rng(args.seed + 900)
    null_positive = []; null_harmful = []; null_utility = []
    for index in range(args.permutation_controls):
        permutation = rng.permutation(len(discovery_features))
        pm = fit_binary(discovery_features, discovery["positive"][permutation], discovery["formula"], args.seed + 1000 + index)
        hm = fit_binary(discovery_features, discovery["harmful"][permutation], discovery["formula"], args.seed + 2000 + index)
        pp = predict_binary(pm, confirmation_features); hp = predict_binary(hm, confirmation_features)
        null_positive.append(safe_auprc(confirmation["positive"], pp))
        null_harmful.append(safe_auprc(confirmation["harmful"], hp))
        null_utility.append(route_outcome(
            confirmation["formula"], confirmation["mass_hit"], confirmation["rule_hit"], pp, hp, threshold,
        )[0]["risk_utility"])
    confirmation_ci = formula_bootstrap_delta(
        confirmation["formula"], confirmation["mass_hit"], confirmation_final,
        args.bootstrap_draws, args.seed + 301,
    )
    gates = {
        "discovery_positive_events_sufficient": int(np.sum(discovery["positive"])) >= args.min_discovery_positive,
        "discovery_harmful_events_sufficient": int(np.sum(discovery["harmful"])) >= args.min_discovery_harmful,
        "confirmation_positive_events_sufficient": int(np.sum(confirmation["positive"])) >= args.min_confirmation_positive,
        "confirmation_harmful_events_sufficient": int(np.sum(confirmation["harmful"])) >= args.min_confirmation_harmful,
        "formula_disjoint": len(overlap) == 0,
        "positive_auprc_beats_null95": observed_positive_auprc > float(np.quantile(null_positive, 0.95)),
        "harmful_auprc_beats_null95": observed_harmful_auprc > float(np.quantile(null_harmful, 0.95)),
        "selected_formula_support": confirmation_route["selected_formulas"] >= args.min_selected_formulas,
        "risk_utility_positive": confirmation_route["risk_utility"] > 0,
        "formula_bootstrap_ci_positive": confirmation_ci[0] > 0,
        "utility_beats_null95": confirmation_route["risk_utility"] > float(np.quantile(null_utility, 0.95)),
    }
    report.update({
        "status": "CLEAN_RULE_OBSERVABILITY_PASS" if all(gates.values()) else "CLEAN_RULE_OBSERVABILITY_FAIL",
        "confirmation": {
            "queries": int(len(cp)), "formulas": int(len(np.unique(confirmation["formula"]))),
            "positive_events": int(np.sum(confirmation["positive"])),
            "harmful_events": int(np.sum(confirmation["harmful"])),
            "positive_auprc": observed_positive_auprc, "harmful_auprc": observed_harmful_auprc,
            "route": confirmation_route, "route_formula_bootstrap_ci95": confirmation_ci,
        },
        "permutation_control": {
            "count": args.permutation_controls,
            "positive_auprc_null95": float(np.quantile(null_positive, 0.95)),
            "harmful_auprc_null95": float(np.quantile(null_harmful, 0.95)),
            "utility_null95": float(np.quantile(null_utility, 0.95)),
        },
        "formula_overlap": int(len(overlap)), "gates": gates,
        "pass_to_gpu_training": bool(all(gates.values())),
    })
    arrays.update({"confirmation_positive_probability": cp, "confirmation_harmful_probability": ch})
    return report, arrays


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.folds < 2 or args.permutation_controls < 1 or args.bootstrap_draws < 100:
        raise ValueError("invalid audit size")
    required = [args.discovery_ranks, args.manifest, args.token_dir / "rows.npy",
                args.token_dir / "official_embeddings_f32.npy", args.rule_library]
    if args.confirmation_ranks is not None:
        required.append(args.confirmation_ranks)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    with np.load(args.manifest) as loaded:
        body = {key: np.array(loaded[key], copy=True) for key in ("query_row",)}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    discovery = labels_from_ranks(args.discovery_ranks)
    discovery_features, feature_names = build_clean_features(discovery, body, rows, official, args)
    confirmation = None; confirmation_features = None
    if args.confirmation_ranks is not None:
        confirmation = labels_from_ranks(args.confirmation_ranks)
        confirmation_features, confirmation_names = build_clean_features(confirmation, body, rows, official, args)
        if feature_names != confirmation_names:
            raise RuntimeError("discovery/confirmation feature contract mismatch")
    result, prediction_arrays = audit_arrays(
        discovery_features, discovery, confirmation_features, confirmation, args,
    )
    result.update({
        "formal": args.confirmation_ranks is not None,
        "feature_contract": {
            "one_unmodified_query_spectrum_only": True,
            "official_embedding_fixed_projection": True,
            "mass_kernel_fixed_projection": True,
            "rule_response_vector": True,
            "candidate_scores_used_as_features": False,
            "candidate_ranks_used_as_features": False,
            "identity_used_as_feature": False,
            "formula_used_as_feature": False,
            "formula_used_only_for_grouping_weighting_and_audit": True,
            "outcomes_used_only_as_labels": True,
        },
        "feature_dimension": int(discovery_features.shape[1]),
        "feature_names": feature_names,
        "claim_limit": (
            "Necessary clean-input rule-applicability evidence only; not an embedding result. "
            "Retrospective mode never authorizes GPU training."
        ),
        "provenance": {
            "discovery_ranks_sha256": sha256_file(args.discovery_ranks),
            "confirmation_ranks_sha256": sha256_file(args.confirmation_ranks) if args.confirmation_ranks else None,
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
            "script_sha256": sha256_file(Path(__file__)),
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_observability_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "predictions.npz", **prediction_arrays,
            discovery_query=discovery["query"], discovery_formula=discovery["formula"],
            **({"confirmation_query": confirmation["query"], "confirmation_formula": confirmation["formula"]}
               if confirmation is not None else {}),
        )
        np.savez_compressed(temporary / "clean_features.npz", discovery=discovery_features,
                            **({"confirmation": confirmation_features} if confirmation_features is not None else {}))
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Component-cross-fitted DreaMS edge calibration for NetID positive mode."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_component_folds(components: np.ndarray, sizes: np.ndarray, folds: int) -> dict[int, int]:
    if folds < 2 or folds > len(components):
        raise ValueError("invalid number of component folds")
    totals = np.zeros(folds, dtype=np.int64)
    assignment: dict[int, int] = {}
    order = np.lexsort((components, -sizes))
    for index in order:
        fold = int(np.argmin(totals))
        component = int(components[index])
        assignment[component] = fold
        totals[fold] += int(sizes[index])
    return assignment


def fixed_fdr_threshold(
    positive: np.ndarray, decoy: np.ndarray, controls_per_edge: int, target_fdr: float
) -> dict[str, float]:
    candidates = np.unique(np.concatenate([positive, decoy]))[::-1]
    best: dict[str, float] | None = None
    for threshold in candidates:
        selected_positive = int(np.sum(positive >= threshold))
        if selected_positive == 0:
            continue
        selected_decoy = int(np.sum(decoy >= threshold))
        estimated_false = selected_decoy / controls_per_edge
        estimated_fdr = estimated_false / selected_positive
        if estimated_fdr <= target_fdr:
            record = {
                "threshold": float(threshold),
                "selected_positive": selected_positive,
                "selected_decoy": selected_decoy,
                "estimated_fdr": float(estimated_fdr),
            }
            if best is None or selected_positive > best["selected_positive"]:
                best = record
    if best is None:
        return {
            "threshold": float("inf"),
            "selected_positive": 0,
            "selected_decoy": 0,
            "estimated_fdr": 0.0,
        }
    return best


def evaluate_selection(
    positive: np.ndarray, decoy: np.ndarray, threshold: float, controls_per_edge: int
) -> dict[str, float]:
    selected_positive = int(np.sum(positive >= threshold))
    selected_decoy = int(np.sum(decoy >= threshold))
    estimated_false = selected_decoy / controls_per_edge
    fdr = estimated_false / selected_positive if selected_positive else 0.0
    return {
        "positive_edges": int(len(positive)),
        "decoy_edges": int(len(decoy)),
        "selected_positive": selected_positive,
        "selected_decoy": selected_decoy,
        "positive_coverage": float(selected_positive / len(positive)) if len(positive) else 0.0,
        "target_decoy_fdr_proxy": float(fdr),
    }


def aggregate_embedding_metadata(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        raw_ids = np.asarray(cache["netid_peak_id"], dtype=np.int64)
        raw_vectors = np.asarray(cache["embeddings"], dtype=np.float32)
        raw_mz = np.asarray(cache["precursor_mz"], dtype=float)
        raw_rt = np.asarray(cache["raw_rt_min"], dtype=float)
        raw_counts = np.asarray(cache["n_fragment_peaks"], dtype=np.int64)
    feature_ids = np.unique(raw_ids)
    vectors, mz, rt, counts = [], [], [], []
    for feature in feature_ids:
        mask = raw_ids == feature
        vector = raw_vectors[mask].mean(axis=0)
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
        mz.append(float(np.median(raw_mz[mask])))
        rt.append(float(np.median(raw_rt[mask])))
        counts.append(int(np.max(raw_counts[mask])))
    return {
        "feature_ids": feature_ids,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "mz": np.asarray(mz),
        "rt": np.asarray(rt),
        "counts": np.asarray(counts),
    }


def build_component_isolated_decoy_similarities(
    metadata: dict[str, np.ndarray],
    train_positive_pairs: np.ndarray,
    all_positive_pairs: np.ndarray,
    held_features: set[int],
    degree_by_feature: dict[int, int],
    controls_per_edge: int,
) -> np.ndarray:
    """Build exact descriptor-nearest nonedges inside the training universe."""

    all_ids = metadata["feature_ids"]
    allowed_mask = np.asarray([int(value) not in held_features for value in all_ids])
    feature_ids = all_ids[allowed_mask]
    vectors = metadata["vectors"][allowed_mask]
    mz = metadata["mz"][allowed_mask]
    rt = metadata["rt"][allowed_mask]
    counts = metadata["counts"][allowed_mask]
    index = {int(value): position for position, value in enumerate(feature_ids)}
    if any(int(value) not in index for value in train_positive_pairs.ravel()):
        raise RuntimeError("training positive contains held feature")
    left, right = np.triu_indices(len(feature_ids), k=1)
    positive_set = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in all_positive_pairs
    }
    allowed = np.asarray(
        [
            (min(int(feature_ids[a]), int(feature_ids[b])), max(int(feature_ids[a]), int(feature_ids[b])))
            not in positive_set
            for a, b in zip(left, right, strict=True)
        ],
        dtype=bool,
    )
    candidates = np.stack([left[allowed], right[allowed]], axis=1)
    positive_indices = np.asarray(
        [[index[int(a)], index[int(b)]] for a, b in train_positive_pairs], dtype=np.int64
    )
    degree = np.asarray([degree_by_feature.get(int(value), 0) for value in feature_ids])

    def descriptors(pairs: np.ndarray) -> np.ndarray:
        a, b = pairs[:, 0], pairs[:, 1]
        return np.stack(
            [
                np.log1p(np.abs(mz[a] - mz[b])),
                np.log1p(np.abs(rt[a] - rt[b])),
                np.log1p(degree[a] + degree[b]),
                np.log1p(np.minimum(counts[a], counts[b])),
            ],
            axis=1,
        )

    candidate_x = descriptors(candidates)
    positive_x = descriptors(positive_indices)
    median = np.median(candidate_x, axis=0)
    scale = np.subtract(*np.percentile(candidate_x, [75, 25], axis=0))
    scale[scale <= 1e-12] = 1.0
    candidate_x = (candidate_x - median) / scale
    positive_x = (positive_x - median) / scale
    tree = cKDTree(candidate_x)
    similarities = np.empty((len(positive_indices), controls_per_edge), dtype=float)
    for row_index, target in enumerate(positive_x):
        distance, _ = tree.query(target, k=controls_per_edge, eps=0.0, workers=1)
        radius = float(np.atleast_1d(distance)[-1]) + 1e-12
        pool = np.asarray(tree.query_ball_point(target, radius), dtype=np.int64)
        exact_distance = np.sum((candidate_x[pool] - target) ** 2, axis=1)
        order = np.lexsort(
            (
                feature_ids[candidates[pool, 1]],
                feature_ids[candidates[pool, 0]],
                exact_distance,
            )
        )[:controls_per_edge]
        chosen = candidates[pool[order]]
        similarities[row_index] = np.sum(
            vectors[chosen[:, 0]] * vectors[chosen[:, 1]], axis=1
        )
    return similarities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-dir",
        type=Path,
        default=Path("data/validation/netid_positive_dreams_edge_signal_20260901"),
    )
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=Path("data/validation/netid_positive_edge_robustness_20260901"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_positive_edge_calibration_20260901"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/validation/netid_mouse_liver_positive_dreams_20260901"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fdr", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    predictions_path = args.output_dir / "component_oof_predictions.csv.gz"
    model_path = args.output_dir / "full_positive_mode_calibrator.pkl"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") not in {
            "netid_positive_edge_calibration_passed",
            "netid_positive_edge_calibration_failed",
        }:
            raise RuntimeError("invalid existing calibration report")
        for name, path in (("predictions", predictions_path), ("model", model_path)):
            if sha256(path) != report["provenance"][f"{name}_sha256"]:
                raise RuntimeError(f"existing calibration {name} changed")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {args.output_dir}")

    edge_report_path = args.edge_dir / "report.json"
    edge_table_path = args.edge_dir / "edge_matched_nonedges.csv.gz"
    robustness_path = args.robustness_dir / "report.json"
    edge_report = json.loads(edge_report_path.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if edge_report.get("status") != "netid_dreams_edge_signal_passed":
        raise RuntimeError("positive edge signal did not pass")
    if robustness.get("status") != "netid_positive_edge_robustness_passed":
        raise RuntimeError("positive edge robustness did not pass")
    if sha256(edge_table_path) != edge_report["provenance"]["pair_table_sha256"]:
        raise RuntimeError("positive edge pair table changed")

    frame = pd.read_csv(edge_table_path)
    controls = int(edge_report["controls_per_edge"])
    embedding_report_path = args.embedding_dir / "report.json"
    embedding_path = args.embedding_dir / "official_dreams_embeddings.npz"
    embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
    if sha256(embedding_path) != embedding_report["provenance"]["embeddings_sha256"]:
        raise RuntimeError("positive-mode embedding cache changed")
    metadata = aggregate_embedding_metadata(embedding_path)
    component_sizes = frame["component"].value_counts().sort_index()
    assignment = balanced_component_folds(
        component_sizes.index.to_numpy(int), component_sizes.to_numpy(int), args.folds
    )
    frame["fold"] = frame["component"].map(assignment).astype(int)
    all_feature_component: dict[int, int] = {}
    for row in frame.itertuples(index=False):
        all_feature_component[int(row.feature1)] = int(row.component)
        all_feature_component[int(row.feature2)] = int(row.component)
    all_positive_pairs = frame[["feature1", "feature2"]].to_numpy(dtype=np.int64)
    degree_counts = pd.concat([frame["feature1"], frame["feature2"]]).value_counts()
    degree_by_feature = {int(key): int(value) for key, value in degree_counts.items()}

    predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(args.folds):
        held_components = set(frame.loc[frame["fold"].eq(fold), "component"].astype(int))
        train = frame[~frame["component"].isin(held_components)]
        test = frame[frame["component"].isin(held_components)]
        train_positive = train["dreams_similarity"].to_numpy(float)
        held_features = {
            feature
            for feature, component in all_feature_component.items()
            if component in held_components
        }
        train_decoy = build_component_isolated_decoy_similarities(
            metadata,
            train[["feature1", "feature2"]].to_numpy(dtype=np.int64),
            all_positive_pairs,
            held_features,
            degree_by_feature,
            controls,
        ).ravel()
        if len(train_positive) < 100 or len(train_decoy) < 100:
            raise RuntimeError(f"fold {fold} has insufficient component-isolated training rows")
        x_train = np.concatenate([train_positive, train_decoy])[:, None]
        y_train = np.concatenate(
            [np.ones(len(train_positive), dtype=int), np.zeros(len(train_decoy), dtype=int)]
        )
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=args.seed)
        model.fit(x_train, y_train)
        train_positive_probability = model.predict_proba(train_positive[:, None])[:, 1]
        train_decoy_probability = model.predict_proba(train_decoy[:, None])[:, 1]
        thresholds = {
            str(target): fixed_fdr_threshold(
                train_positive_probability, train_decoy_probability, controls, target
            )
            for target in args.target_fdr
        }
        for row in test.itertuples(index=False):
            positive_probability = float(
                model.predict_proba(np.array([[float(row.dreams_similarity)]]))[0, 1]
            )
            predictions.append(
                {
                    "fold": fold,
                    "component": int(row.component),
                    "edge_id": row.edge_id,
                    "category": row.category,
                    "author_explicit_ms2": bool(pd.notna(row.score_MS2_similarity)),
                    "label": 1,
                    "dreams_similarity": float(row.dreams_similarity),
                    "probability": positive_probability,
                }
            )
            for control in range(controls):
                similarity = float(getattr(row, f"decoy_similarity_{control}"))
                predictions.append(
                    {
                        "fold": fold,
                        "component": int(row.component),
                        "edge_id": f"{row.edge_id}__decoy{control}",
                        "category": row.category,
                        "author_explicit_ms2": bool(pd.notna(row.score_MS2_similarity)),
                        "label": 0,
                        "dreams_similarity": similarity,
                        "probability": float(model.predict_proba(np.array([[similarity]]))[0, 1]),
                    }
                )
        fold_reports.append(
            {
                "fold": fold,
                "held_components": len(held_components),
                "test_positive_edges": len(test),
                "train_positive_edges": len(train_positive),
                "train_component_isolated_decoys": len(train_decoy),
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
                "training_thresholds": thresholds,
            }
        )

    oof = pd.DataFrame.from_records(predictions)
    if len(oof[oof["label"].eq(1)]) != len(frame):
        raise RuntimeError("OOF positive edge coverage mismatch")
    oof_auc = float(roc_auc_score(oof["label"], oof["probability"]))
    fdr_reports: dict[str, Any] = {}
    for target in args.target_fdr:
        selected = []
        for fold in range(args.folds):
            threshold = fold_reports[fold]["training_thresholds"][str(target)]["threshold"]
            part = oof[oof["fold"].eq(fold)].copy()
            part["selected"] = part["probability"] >= threshold
            selected.append(part)
        joined = pd.concat(selected, ignore_index=True)
        positive = joined[joined["label"].eq(1)]
        decoy = joined[joined["label"].eq(0)]
        overall = evaluate_selection(
            positive["probability"].to_numpy(float),
            decoy["probability"].to_numpy(float),
            -np.inf,
            controls,
        )
        overall["selected_positive"] = int(positive["selected"].sum())
        overall["selected_decoy"] = int(decoy["selected"].sum())
        overall["positive_coverage"] = float(positive["selected"].mean())
        overall["target_decoy_fdr_proxy"] = float(
            (decoy["selected"].sum() / controls) / max(positive["selected"].sum(), 1)
        )
        no_ms2 = positive[~positive["author_explicit_ms2"]]
        decoy_no_ms2 = decoy[~decoy["author_explicit_ms2"]]
        fdr_reports[str(target)] = {
            "overall": overall,
            "without_author_explicit_ms2": {
                "positive_edges": int(len(no_ms2)),
                "selected_positive": int(no_ms2["selected"].sum()),
                "selected_decoy": int(decoy_no_ms2["selected"].sum()),
                "positive_coverage": float(no_ms2["selected"].mean()),
                "target_decoy_fdr_proxy": float(
                    (decoy_no_ms2["selected"].sum() / controls)
                    / max(no_ms2["selected"].sum(), 1)
                ),
            },
        }

    full_positive = frame["dreams_similarity"].to_numpy(float)
    full_decoy = frame[[f"decoy_similarity_{i}" for i in range(controls)]].to_numpy(float).ravel()
    full_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=args.seed)
    full_model.fit(
        np.concatenate([full_positive, full_decoy])[:, None],
        np.concatenate([np.ones(len(full_positive)), np.zeros(len(full_decoy))]),
    )
    frozen_thresholds = {
        str(target): fixed_fdr_threshold(
            full_model.predict_proba(full_positive[:, None])[:, 1],
            full_model.predict_proba(full_decoy[:, None])[:, 1],
            controls,
            target,
        )
        for target in args.target_fdr
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    oof.to_csv(predictions_path, index=False, compression="gzip")
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": full_model,
                "thresholds": frozen_thresholds,
                "panel": "Mouse_liver_pos",
                "feature": "official_dreams_centroid_cosine",
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    primary = fdr_reports["0.1"]
    gates = {
        "oof_auc_ge_0_60": oof_auc >= 0.60,
        "ten_percent_fdr_proxy_le_0_15": primary["overall"]["target_decoy_fdr_proxy"] <= 0.15,
        "ten_percent_edge_coverage_ge_0_10": primary["overall"]["positive_coverage"] >= 0.10,
        "no_author_ms2_coverage_ge_0_05": primary["without_author_explicit_ms2"]["positive_coverage"] >= 0.05,
        "all_fold_coefficients_positive": all(item["coefficient"] > 0 for item in fold_reports),
    }
    gates["pass_positive_mode_calibrator_development"] = all(gates.values())
    status = (
        "netid_positive_edge_calibration_passed"
        if gates["pass_positive_mode_calibrator_development"]
        else "netid_positive_edge_calibration_failed"
    )
    report = {
        "status": status,
        "formal": True,
        "panel": "Mouse_liver_pos",
        "feature": "official DreaMS centroid cosine only",
        "component_folds": args.folds,
        "oof_auc": oof_auc,
        "folds": fold_reports,
        "fixed_fdr_oof": fdr_reports,
        "frozen_full_model": {
            "coefficient": float(full_model.coef_[0, 0]),
            "intercept": float(full_model.intercept_[0]),
            "thresholds": frozen_thresholds,
        },
        "gates": gates,
        "contracts": {
            "outer_test_components_seen_in_training_positives": False,
            "training_decoys_containing_held_component_features": False,
            "threshold_selected_on_outer_test": False,
            "negative_mode_supported": False,
            "annotation_accuracy_claim": False,
            "pre_solution_netid_overlay_executed": False,
        },
        "provenance": {
            "edge_report_sha256": sha256(edge_report_path),
            "edge_table_sha256": sha256(edge_table_path),
            "robustness_report_sha256": sha256(robustness_path),
            "embedding_report_sha256": sha256(embedding_report_path),
            "embeddings_sha256": sha256(embedding_path),
            "predictions_sha256": sha256(predictions_path),
            "model_sha256": sha256(model_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": (
            "Component-cross-fitted reproduction of positive-mode author edge membership. "
            "It is not independent edge truth, annotation accuracy, a NetID solver rerun, "
            "negative-mode generalization, or SOTA evidence."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

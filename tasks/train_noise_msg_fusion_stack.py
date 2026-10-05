#!/usr/bin/env python
"""Train and freeze the formula-isolated pair-evidence fusion ranker.

Recipe (preregistered, single shot, no tuning on GNPS):

* features: one molecule-max row from the frozen deployment-visible channels;
* folds: ``stable_fold(query_formula, 5, 20260825)`` so every molecular-formula
  family is entirely inside one fold — no formula leaks across the OOF split;
* model: sklearn HistGradientBoostingClassifier with query-balanced sample
  weights, predicting the positive candidate molecule from deployment-visible
  scores only;
* OOF report: per-query top-1 accuracy of the fused ranking plus the V1, WSE
  and frozen-P2b references on the same folds;
* final artifact: the identical recipe refit on all folds, frozen with joblib
  and a full provenance chain.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from noise_final_core import sha256_file, stable_fold
from noise_msg_pair_evidence_core import (
    MOLECULE_FEATURES,
    build_molecule_features,
    query_balanced_molecule_weights,
)

FOLD_SEED = 20260825
FOLD_COUNT = 5
MODEL_SETTINGS = dict(
    max_iter=400,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=False,
    random_state=20261003,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def top1_accuracy(
    scores: np.ndarray, molecule_label: np.ndarray, query_ptr: np.ndarray,
    query_folds: np.ndarray | None = None, fold: int | None = None,
) -> float:
    """Molecule-max retrieval accuracy with strict ties counting against.

    When ``fold`` is given, only queries whose formula fold equals it are
    scored; edges of other queries have no valid OOF score and must not be
    counted as errors.
    """
    total = 0
    correct = 0
    for query in range(len(query_ptr) - 1):
        if fold is not None and int(query_folds[query]) != fold:
            continue
        total += 1
        m_left, m_right = int(query_ptr[query]), int(query_ptr[query + 1])
        molecule_scores = np.asarray(scores[m_left:m_right], dtype=np.float64)
        best = float(np.max(molecule_scores))
        winners = np.flatnonzero(molecule_scores == best)
        labels = np.asarray(molecule_label[m_left:m_right], dtype=np.int8)
        if len(winners) == 1 and int(labels[winners[0]]) == 1:
            correct += 1
    if total == 0:
        raise RuntimeError("top-1 evaluation selected no queries")
    return correct / total


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.evidence / "evidence.npz", allow_pickle=False) as body:
        evidence = {name: np.asarray(body[name]) for name in body.files}
    report_path = args.evidence / "report.json"
    evidence_report = json.loads(report_path.read_text(encoding="utf-8"))
    if evidence_report.get("status") != "NOISE_MSG_PAIR_EVIDENCE_COMPLETE":
        raise RuntimeError("pair evidence artifact is incomplete")

    features = build_molecule_features(
        evidence, evidence["query_ptr"], evidence["molecule_ptr"],
    )
    labels = np.asarray(evidence["molecule_label"], dtype=np.int8)
    if len(labels) != len(features):
        raise RuntimeError("molecule evidence label alignment drifted")
    molecule_query = np.repeat(
        np.arange(len(evidence["query_ptr"]) - 1, dtype=np.int64),
        np.diff(np.asarray(evidence["query_ptr"], dtype=np.int64)),
    )
    formulas = np.asarray([str(value) for value in evidence["query_formula"]])
    if len(formulas) != len(evidence["query_ptr"]) - 1 or len(molecule_query) != len(labels):
        raise RuntimeError("formula/query/molecule ledgers are not aligned")
    query_folds = np.asarray(
        [stable_fold(formula, FOLD_COUNT, FOLD_SEED) for formula in formulas],
        dtype=np.int8,
    )
    molecule_folds = query_folds[molecule_query]
    # Every query owns total weight 1.0.  Its positive molecule owns 0.5 and
    # all negative molecules share 0.5.  This exactly removes candidate-count
    # and reference-spectrum multiplicity as accidental training dose.
    sample_weight = query_balanced_molecule_weights(labels, evidence["query_ptr"])

    molecule_scores = {
        name: np.maximum.reduceat(
            np.asarray(evidence[name], dtype=np.float64),
            np.asarray(evidence["molecule_ptr"], dtype=np.int64)[:-1],
        )
        for name in ("v1_cosine", "weighted_entropy", "p2b_fused")
    }

    oof_scores = np.zeros(len(features), dtype=np.float64)
    fold_reports = []
    for fold in range(FOLD_COUNT):
        held = molecule_folds == fold
        model = HistGradientBoostingClassifier(**MODEL_SETTINGS)
        model.fit(features[~held], labels[~held], sample_weight=sample_weight[~held])
        probabilities = model.predict_proba(features[held])[:, 1]
        oof_scores[held] = probabilities
        reference = {
            "v1": top1_accuracy(
                molecule_scores["v1_cosine"], labels, evidence["query_ptr"],
                query_folds, fold,
            ),
            "wse": top1_accuracy(
                molecule_scores["weighted_entropy"], labels, evidence["query_ptr"],
                query_folds, fold,
            ),
            "p2b_fused": top1_accuracy(
                molecule_scores["p2b_fused"], labels, evidence["query_ptr"],
                query_folds, fold,
            ),
        }
        fold_reports.append({
            "fold": fold,
            "held_queries": int(np.unique(molecule_query[held]).size),
            "held_molecules": int(held.sum()),
            "fused_oof_top1": top1_accuracy(
                oof_scores, labels, evidence["query_ptr"], query_folds, fold,
            ),
            "references_held_fold_top1": reference,
        })
        print(json.dumps(fold_reports[-1]), flush=True)

    final_model = HistGradientBoostingClassifier(**MODEL_SETTINGS)
    final_model.fit(features, labels, sample_weight=sample_weight)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fusion_freeze.", dir=args.output.parent))
    try:
        joblib.dump(
            {"model": final_model, "feature_names": list(MOLECULE_FEATURES),
             "settings": MODEL_SETTINGS, "fold_seed": FOLD_SEED,
             "fold_count": FOLD_COUNT},
            staging / "fusion_model.joblib",
        )
        # The grand router must see formula-OOF predictions on its training
        # side, never in-sample final-model predictions.  Expand each molecule
        # probability to its reference edges so it follows the common expert
        # evidence contract without changing candidate rankings.
        oof_pair_scores = np.repeat(
            oof_scores.astype(np.float32),
            np.diff(np.asarray(evidence["molecule_ptr"], dtype=np.int64)),
        )
        np.savez_compressed(
            staging / "oof_pair_scores.npz",
            pair_scores=oof_pair_scores,
            evidence_sha256=np.asarray(sha256_file(args.evidence / "evidence.npz")),
        )
        reference_all = {
            name: top1_accuracy(
                values, labels, evidence["query_ptr"],
            )
            for name, values in molecule_scores.items()
        }
        fused_all = top1_accuracy(
            oof_scores, labels, evidence["query_ptr"],
        )
        report = {
            "status": "NOISE_MSG_FUSION_FROZEN",
            "recipe": {
                "features": list(MOLECULE_FEATURES),
                "training_unit": "candidate molecule after reference-spectrum max pooling",
                "model": "HistGradientBoostingClassifier",
                "settings": MODEL_SETTINGS,
                "sample_weights": "each query totals 1.0; positive 0.5; negatives share 0.5",
                "folds": "stable_fold(query_formula, 5, 20260825), formula-disjoint",
            },
            "oof": {
                "fused_top1_all_folds": fused_all,
                "references_top1_all_folds": reference_all,
                "fused_minus_reference_pp": {
                    name: 100.0 * (fused_all - value)
                    for name, value in reference_all.items()
                },
                "per_fold": fold_reports,
            },
            "gnps_usage": "none; GNPS is untouched by training, freezing and selection",
            "provenance": {
                "evidence_sha256": sha256_file(args.evidence / "evidence.npz"),
                "evidence_report_sha256": sha256_file(report_path),
                "oof_pair_scores_sha256": sha256_file(staging / "oof_pair_scores.npz"),
            },
            "claim_limit": (
                "OOF numbers are MassSpecGym train-side development evidence; the "
                "deployable claim is decided only by the frozen GNPS evaluation."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Calibrate or evaluate open-set molecular annotation at fixed error risk.

Development mode selects one confidence threshold per method on a consumed
panel.  External mode only applies those saved thresholds; it never optimizes
against external labels.  Pair-level library scores and molecule-level
structure-retrieval scores share the same candidate contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {name: np.asarray(body[name]) for name in body.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_match_mask(panel: dict[str, np.ndarray], fraction: float, seed: int) -> np.ndarray:
    if "query_has_match" in panel:
        return np.asarray(panel["query_has_match"], dtype=bool)
    if not 0.0 < fraction < 1.0:
        raise ValueError("match fraction must be in (0, 1)")
    threshold = int(fraction * (1 << 64))
    return np.asarray([
        int.from_bytes(hashlib.sha256(
            f"{seed}|{identity}".encode("utf-8")
        ).digest()[:8], "big") < threshold
        for identity in panel["query_ik14"].astype(str)
    ], dtype=bool)


def molecule_scores(panel: dict[str, np.ndarray], scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == len(panel["molecule_label"]):
        return scores
    if len(scores) != len(panel["candidate_row"]):
        raise RuntimeError(
            "method scores must have one value per candidate spectrum or candidate molecule"
        )
    pointer = np.asarray(panel["molecule_ptr"], dtype=np.int64)
    return np.asarray([
        np.max(scores[int(left):int(right)])
        for left, right in zip(pointer[:-1], pointer[1:])
    ])


def query_predictions(
    panel: dict[str, np.ndarray], scores: np.ndarray, has_match: np.ndarray,
) -> dict[str, np.ndarray]:
    pointer = np.asarray(panel["query_ptr"], dtype=np.int64)
    labels = np.asarray(panel["molecule_label"], dtype=bool)
    confidence = np.empty(len(has_match), dtype=np.float64)
    correct = np.zeros(len(has_match), dtype=bool)
    rank = np.full(len(has_match), -1, dtype=np.int64)
    for query, (left, right) in enumerate(zip(pointer[:-1], pointer[1:])):
        left_i, right_i = int(left), int(right)
        local_scores = np.asarray(scores[left_i:right_i], dtype=np.float64).copy()
        local_labels = labels[left_i:right_i]
        if has_match[query]:
            positive = np.flatnonzero(local_labels)
            if len(positive) != 1:
                raise RuntimeError(f"match query {query} does not have one positive")
            positive_score = local_scores[int(positive[0])]
            rank[query] = 1 + int(np.sum(local_scores[~local_labels] >= positive_score))
            correct[query] = rank[query] == 1
        else:
            # Some development panels retain the positive physically; emulate
            # deployment by removing it before prediction. External open-set
            # panels already have no positive, so this is a no-op there.
            local_scores[local_labels] = -np.inf
        finite = np.sort(local_scores[np.isfinite(local_scores)])
        if len(finite) < 2:
            raise RuntimeError(f"query {query} needs at least two deployable candidates")
        confidence[query] = float(finite[-1] - finite[-2])
    return {"confidence": confidence, "correct": correct, "rank": rank}


def operating_point(
    prediction: dict[str, np.ndarray], has_match: np.ndarray, threshold: float,
) -> dict[str, float | int]:
    accepted = prediction["confidence"] >= threshold
    correct = accepted & has_match & prediction["correct"]
    wrong_match = accepted & has_match & ~prediction["correct"]
    false_nomatch = accepted & ~has_match
    assigned = int(accepted.sum())
    correct_count = int(correct.sum())
    false_count = int(wrong_match.sum() + false_nomatch.sum())
    return {
        "threshold": float(threshold),
        "queries": int(len(has_match)),
        "match_queries": int(has_match.sum()),
        "no_match_queries": int((~has_match).sum()),
        "accepted": assigned,
        "correct_annotations": correct_count,
        "wrong_match_assignments": int(wrong_match.sum()),
        "false_no_match_assignments": int(false_nomatch.sum()),
        "fdr": float(false_count / assigned) if assigned else 0.0,
        "precision": float(correct_count / assigned) if assigned else 0.0,
        "match_coverage": float(correct_count / max(int(has_match.sum()), 1)),
        "annotation_yield_all_queries": float(correct_count / len(has_match)),
        "abstention_rate": float(1.0 - assigned / len(has_match)),
        "match_recall_at_1": float(np.mean(prediction["rank"][has_match] == 1)),
        "match_mrr": float(np.mean(1.0 / prediction["rank"][has_match])),
    }


def calibrate_threshold(
    prediction: dict[str, np.ndarray], has_match: np.ndarray, target_fdr: float,
) -> dict[str, float | int]:
    upper = np.nextafter(float(np.max(prediction["confidence"])), np.inf)
    candidates = np.unique(np.r_[prediction["confidence"], upper])
    best = None
    for threshold in candidates:
        point = operating_point(prediction, has_match, float(threshold))
        if point["fdr"] <= target_fdr:
            if best is None or (
                point["correct_annotations"], point["precision"], -point["threshold"]
            ) > (
                best["correct_annotations"], best["precision"], -best["threshold"]
            ):
                best = point
    if best is None:
        raise RuntimeError("no feasible open-set operating point")
    return best


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument("--development-match-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20261008)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target FDR must be in (0, 1)")
    panel = load_npz(args.panel)
    bundle = load_npz(args.scores)
    methods = [str(value) for value in bundle["method_names"]]
    matrix = np.asarray(bundle[args.score_key], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(methods):
        raise RuntimeError("score bundle must be [method, pair-or-molecule]")
    has_match = deterministic_match_mask(
        panel, args.development_match_fraction, args.seed,
    )
    predictions = {
        method: query_predictions(
            panel, molecule_scores(panel, matrix[index]), has_match,
        )
        for index, method in enumerate(methods)
    }
    if args.mode == "calibrate":
        results = {
            method: calibrate_threshold(prediction, has_match, args.target_fdr)
            for method, prediction in predictions.items()
        }
        status = "OPEN_SET_ANNOTATION_CALIBRATION_COMPLETE"
        role = "development_consumed"
    else:
        if args.calibration is None or not args.calibration.is_file():
            raise FileNotFoundError("external evaluation requires --calibration")
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        if calibration.get("status") != "OPEN_SET_ANNOTATION_CALIBRATION_COMPLETE":
            raise RuntimeError("calibration artifact is incomplete")
        if abs(float(calibration.get("target_fdr", -1.0)) - args.target_fdr) > 1e-12:
            raise RuntimeError("external target FDR differs from frozen calibration")
        missing = sorted(set(methods) - set(calibration["methods"]))
        if missing:
            raise RuntimeError(f"external methods lack frozen thresholds: {missing}")
        results = {
            method: operating_point(
                prediction, has_match,
                float(calibration["methods"][method]["threshold"]),
            )
            for method, prediction in predictions.items()
        }
        status = "OPEN_SET_ANNOTATION_EXTERNAL_EVALUATION_COMPLETE"
        role = "sealed_external_fixed_thresholds"
    report = {
        "status": status,
        "role": role,
        "target_fdr": args.target_fdr,
        "confidence": "top1_minus_top2_molecule_score",
        "panel_sha256": sha256(args.panel),
        "score_bundle_sha256": sha256(args.scores),
        "calibration_sha256": (
            sha256(args.calibration) if args.mode == "evaluate" else None
        ),
        "methods": results,
        "metric_definitions": {
            "match_coverage": "correct accepted annotations / all match queries",
            "annotation_yield_all_queries": "correct accepted annotations / all match and no-match queries",
            "fdr": "accepted wrong-match plus no-match assignments / all accepted assignments",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

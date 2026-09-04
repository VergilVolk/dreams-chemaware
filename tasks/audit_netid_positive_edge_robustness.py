#!/usr/bin/env python
"""Robustness audit for the positive-mode NetID DreaMS edge signal."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_bootstrap(
    delta: np.ndarray,
    clusters: np.ndarray,
    repeats: int,
    seed: int,
    equal_component_weight: bool,
) -> dict[str, Any]:
    unique = np.unique(clusters)
    means = np.asarray([np.mean(delta[clusters == value]) for value in unique])
    grouped = {value: delta[clusters == value] for value in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        selected = rng.choice(unique, size=len(unique), replace=True)
        if equal_component_weight:
            index = {value: i for i, value in enumerate(unique)}
            draws[repeat] = float(np.mean([means[index[value]] for value in selected]))
        else:
            draws[repeat] = float(np.mean(np.concatenate([grouped[value] for value in selected])))
    return {
        "mean": float(np.mean(means) if equal_component_weight else np.mean(delta)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(unique)),
        "resamples": int(repeats),
        "weighting": "equal_component" if equal_component_weight else "edge_weighted_cluster",
    }


def matched_randomization(scores: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    observed = float(np.mean(scores[:, 0] - scores[:, 1:].mean(axis=1)))
    total = scores.sum(axis=1)
    rng = np.random.default_rng(seed)
    null = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        selected = rng.integers(0, scores.shape[1], size=len(scores))
        chosen = scores[np.arange(len(scores)), selected]
        null[repeat] = float(np.mean(chosen - (total - chosen) / (scores.shape[1] - 1)))
    return {
        "observed_delta": observed,
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "one_sided_empirical_p": float((1 + np.sum(null >= observed)) / (repeats + 1)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-dir",
        type=Path,
        default=Path("data/validation/netid_positive_dreams_edge_signal_20260901"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_positive_edge_robustness_20260901"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--randomization-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") not in {
            "netid_positive_edge_robustness_passed",
            "netid_positive_edge_robustness_failed",
        }:
            raise RuntimeError("invalid existing robustness report")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {args.output_dir}")

    edge_report_path = args.edge_dir / "report.json"
    edge_table_path = args.edge_dir / "edge_matched_nonedges.csv.gz"
    edge_report = json.loads(edge_report_path.read_text(encoding="utf-8"))
    if edge_report.get("status") != "netid_dreams_edge_signal_passed":
        raise RuntimeError("positive-mode edge gate did not pass")
    if edge_report.get("panel") != "Mouse_liver_pos":
        raise RuntimeError("robustness audit is locked to Mouse_liver_pos")
    if sha256(edge_table_path) != edge_report["provenance"]["pair_table_sha256"]:
        raise RuntimeError("positive-mode pair table hash mismatch")

    frame = pd.read_csv(edge_table_path)
    decoy_columns = [f"decoy_similarity_{index}" for index in range(edge_report["controls_per_edge"])]
    decoy = frame[decoy_columns].to_numpy(float)
    frame["delta"] = frame["dreams_similarity"].to_numpy(float) - decoy.mean(axis=1)
    frame["author_explicit_ms2"] = frame["score_MS2_similarity"].notna()
    component = frame.groupby("component")["delta"].agg(["size", "mean"])
    largest = int(component["size"].idxmax())
    without_largest = frame[frame["component"].ne(largest)]
    no_author_ms2 = frame[~frame["author_explicit_ms2"]]
    with_author_ms2 = frame[frame["author_explicit_ms2"]]

    decoy_pairs: list[tuple[int, int]] = []
    for control in range(edge_report["controls_per_edge"]):
        decoy_pairs.extend(
            map(
                tuple,
                frame[[f"decoy_feature1_{control}", f"decoy_feature2_{control}"]]
                .to_numpy(int)
                .tolist(),
            )
        )
    reuse = Counter(decoy_pairs)
    score_matrix = np.column_stack([frame["dreams_similarity"].to_numpy(float), decoy])
    overall_randomization = matched_randomization(
        score_matrix, args.randomization_resamples, args.seed
    )
    no_ms2_randomization = matched_randomization(
        np.column_stack(
            [
                no_author_ms2["dreams_similarity"].to_numpy(float),
                no_author_ms2[decoy_columns].to_numpy(float),
            ]
        ),
        args.randomization_resamples,
        args.seed + 1,
    )

    summaries = {
        "overall_edge_weighted": cluster_bootstrap(
            frame["delta"].to_numpy(float),
            frame["component"].to_numpy(int),
            args.bootstrap_resamples,
            args.seed,
            False,
        ),
        "overall_equal_component": cluster_bootstrap(
            frame["delta"].to_numpy(float),
            frame["component"].to_numpy(int),
            args.bootstrap_resamples,
            args.seed + 1,
            True,
        ),
        "without_author_explicit_ms2": cluster_bootstrap(
            no_author_ms2["delta"].to_numpy(float),
            no_author_ms2["component"].to_numpy(int),
            args.bootstrap_resamples,
            args.seed + 2,
            False,
        ),
        "with_author_explicit_ms2": cluster_bootstrap(
            with_author_ms2["delta"].to_numpy(float),
            with_author_ms2["component"].to_numpy(int),
            args.bootstrap_resamples,
            args.seed + 3,
            False,
        ),
    }
    gates = {
        "overall_equal_component_ci_low_gt_zero": summaries["overall_equal_component"]["ci_low"] > 0,
        "without_author_ms2_ci_low_gt_zero": summaries["without_author_explicit_ms2"]["ci_low"] > 0,
        "largest_component_delta_gt_zero": float(component.loc[largest, "mean"]) > 0,
        "leave_largest_component_delta_gt_zero": float(without_largest["delta"].mean()) > 0,
        "positive_component_fraction_ge_0_75": float((component["mean"] > 0).mean()) >= 0.75,
        "overall_randomization_p_le_0_01": overall_randomization["one_sided_empirical_p"] <= 0.01,
        "no_author_ms2_randomization_p_le_0_01": no_ms2_randomization["one_sided_empirical_p"] <= 0.01,
    }
    gates["pass_to_component_crossfit_calibration"] = all(gates.values())
    status = (
        "netid_positive_edge_robustness_passed"
        if gates["pass_to_component_crossfit_calibration"]
        else "netid_positive_edge_robustness_failed"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": status,
        "formal": True,
        "edges": int(len(frame)),
        "components": int(len(component)),
        "largest_component": {
            "component": largest,
            "edges": int(component.loc[largest, "size"]),
            "edge_fraction": float(component.loc[largest, "size"] / len(frame)),
            "mean_delta": float(component.loc[largest, "mean"]),
        },
        "leave_largest_component": {
            "edges": int(len(without_largest)),
            "mean_delta": float(without_largest["delta"].mean()),
        },
        "positive_component_fraction": float((component["mean"] > 0).mean()),
        "author_ms2_strata": {
            "without_explicit_author_ms2": int(len(no_author_ms2)),
            "with_explicit_author_ms2": int(len(with_author_ms2)),
            "summaries": summaries,
        },
        "decoy_reuse": {
            "rows": int(len(decoy_pairs)),
            "unique_pairs": int(len(reuse)),
            "maximum_reuse": int(max(reuse.values())),
            "pairs_reused_at_least_five_times": int(sum(value >= 5 for value in reuse.values())),
        },
        "matched_randomization": {
            "overall": overall_randomization,
            "without_author_explicit_ms2": no_ms2_randomization,
        },
        "gates": gates,
        "contracts": {
            "author_graph_is_independent_truth": False,
            "calibration_target": "author post-solution edge membership only",
            "annotation_accuracy_claim": False,
            "negative_mode_generalization_claim": False,
        },
        "provenance": {
            "edge_report_sha256": sha256(edge_report_path),
            "edge_table_sha256": sha256(edge_table_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": (
            "Robust positive-mode development signal, including edges lacking the author's "
            "explicit MS2 score. This permits calibrator development only; it does not "
            "establish independent edge truth, annotation gain, or SOTA performance."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

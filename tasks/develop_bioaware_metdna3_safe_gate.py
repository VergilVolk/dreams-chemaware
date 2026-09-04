#!/usr/bin/env python
"""Development-only safe-gate and network-headroom audit for MetDNA3 HILIC."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def noisy_or(values) -> float:
    value = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - value)) if len(value) else 0.0


def formula_fold(value: str, folds: int = 5) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % folds


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def support_from_paths(paths: pd.DataFrame) -> pd.DataFrame:
    frame = paths.copy()
    complete = frame["source_side_complete"].astype(bool)
    signature = frame["missing_source_signature"].fillna("").astype(str)
    frame["dependency_key"] = np.where(
        complete,
        "complete_seed:" + frame["seed_compound_id"].astype(str),
        "missing:" + signature,
    )
    grouped = (
        frame.groupby(["fold", "query_id", "query_candidate_id", "dependency_key"], sort=False)
        ["contribution"].max().reset_index()
    )
    return (
        grouped.groupby(["fold", "query_id", "query_candidate_id"], sort=False)
        ["contribution"].apply(noisy_or).rename("network_support").reset_index()
        .rename(columns={"query_candidate_id": "candidate_id"})
    )


def evaluate_instances(frame: pd.DataFrame, config: tuple[float, float, float]) -> pd.DataFrame:
    weight, maximum_margin, minimum_advantage = config
    rows: list[dict] = []
    for (fold, query_id), group in frame.groupby(["fold", "query_id"], sort=False):
        group = group.copy()
        truth = str(group["truth_candidate_id"].iloc[0])
        ordered = group.sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
        base_max = float(ordered["spectral_score"].iloc[0])
        base_tied = ordered[np.isclose(ordered["spectral_score"], base_max, rtol=0, atol=1e-12)]
        baseline_top = str(base_tied.iloc[0].candidate_id)
        baseline_correct = len(base_tied) == 1 and baseline_top == truth
        second = float(ordered["spectral_score"].iloc[1])
        margin = base_max - second
        network = group.sort_values(["network_support", "candidate_id"], ascending=[False, True])
        network_top = str(network.iloc[0].candidate_id)
        network_max = float(network.iloc[0].network_support)
        network_second = float(network.iloc[1].network_support)
        advantage = network_max - network_second
        gate = bool(
            network_max > 0 and margin <= maximum_margin and advantage >= minimum_advantage
        )
        group["final_score"] = (
            group["spectral_score"] + weight * group["network_support"]
            if gate else group["spectral_score"]
        )
        final_max = float(group["final_score"].max())
        final_tied = group[np.isclose(group["final_score"], final_max, rtol=0, atol=1e-12)]
        final_top = str(final_tied.sort_values("candidate_id").iloc[0].candidate_id)
        final_correct = len(final_tied) == 1 and final_top == truth
        rows.append({
            "fold": int(fold), "query_id": str(query_id), "truth_formula": str(group["truth_formula"].iloc[0]),
            "truth_candidate_id": truth, "baseline_top_candidate": baseline_top,
            "network_top_candidate": network_top, "final_top_candidate": final_top,
            "baseline_correct": bool(baseline_correct), "final_correct": bool(final_correct),
            "corrected": bool(not baseline_correct and final_correct),
            "introduced": bool(baseline_correct and not final_correct),
            "intervened": bool(gate and final_top != baseline_top),
            "spectral_margin": float(margin), "network_advantage": float(advantage),
            "delta": int(final_correct) - int(baseline_correct),
        })
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> dict:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    return {
        "instances": int(len(frame)), "delta_recall1": float(frame["delta"].mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_net_corrected_minus_2x_introduced": corrected - 2 * introduced,
        "intervention_rate": float(frame["intervened"].mean()),
        "mcnemar_exact_p": float(
            binomtest(min(corrected, introduced), corrected + introduced, 0.5).pvalue
        ) if corrected + introduced else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"),
    )
    parser.add_argument(
        "--transitions", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_eval_v1/dependency_corrected_transitions.csv.gz"),
    )
    parser.add_argument(
        "--paths", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_eval_v1/evidence_paths.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_safe_gate_development_v1"),
    )
    args = parser.parse_args()
    scores = pd.read_csv(args.scores)
    transitions = pd.read_csv(args.transitions)
    paths = pd.read_csv(args.paths)
    support = support_from_paths(paths)
    instances = transitions[["fold", "query_id", "truth_formula"]].drop_duplicates()
    frame = instances.merge(scores, on=["query_id", "truth_formula"], validate="many_to_many")
    frame = frame.merge(support, on=["fold", "query_id", "candidate_id"], how="left", validate="one_to_one")
    frame["network_support"] = frame["network_support"].fillna(0.0)

    headroom_rows: list[dict] = []
    for (fold, query_id), group in frame.groupby(["fold", "query_id"], sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        ordered = group.sort_values("spectral_score", ascending=False)
        baseline = str(ordered.iloc[0].candidate_id)
        if baseline == truth:
            continue
        truth_row = group[group["candidate_id"].astype(str) == truth].iloc[0]
        truth_support = float(truth_row.network_support)
        necessary = 0.0
        possible = True
        for competitor in group[group["candidate_id"].astype(str) != truth].itertuples(index=False):
            spectral_gap = float(competitor.spectral_score - truth_row.spectral_score)
            if spectral_gap < 0:
                continue
            support_gap = truth_support - float(competitor.network_support)
            if support_gap <= 0:
                possible = False
                necessary = np.inf
                break
            necessary = max(necessary, spectral_gap / support_gap)
        headroom_rows.append({
            "fold": int(fold), "query_id": str(query_id), "truth_formula": str(group["truth_formula"].iloc[0]),
            "truth_candidate_id": truth, "baseline_top_candidate": baseline,
            "truth_network_support": truth_support,
            "baseline_network_support": float(group[group["candidate_id"].astype(str) == baseline].iloc[0].network_support),
            "network_can_rank_truth_first": bool(possible),
            "minimum_unbounded_network_weight": None if not np.isfinite(necessary) else float(necessary),
        })
    headroom = pd.DataFrame(headroom_rows)

    configurations = list(itertools.product(
        [0.05, 0.10, 0.15, 0.20, 0.30],
        [0.02, 0.05, 0.10, 0.20],
        [0.00, 0.02, 0.05, 0.10],
    ))
    predictions = {config: evaluate_instances(frame, config) for config in configurations}
    oof_frames: list[pd.DataFrame] = []
    selected: list[dict] = []
    for heldout_fold in range(5):
        train_formulas = {
            formula for formula in frame["truth_formula"].unique()
            if formula_fold(formula) != heldout_fold
        }
        candidates_for_selection = []
        for config, result in predictions.items():
            train = result[result["truth_formula"].isin(train_formulas)]
            score = summary(train)
            candidates_for_selection.append((
                score["risk_net_corrected_minus_2x_introduced"],
                score["corrected"], -score["introduced"], -score["intervention_rate"],
                -config[0], -config[1], config,
            ))
        best = max(candidates_for_selection)[-1]
        heldout = predictions[best][
            predictions[best]["truth_formula"].map(formula_fold) == heldout_fold
        ].copy()
        heldout["formula_oof_fold"] = heldout_fold
        oof_frames.append(heldout)
        selected.append({
            "formula_oof_fold": heldout_fold,
            "weight": best[0], "maximum_margin": best[1], "minimum_advantage": best[2],
            "heldout": summary(heldout),
        })
    oof = pd.concat(oof_frames, ignore_index=True)
    oof_result = summary(oof)

    full_ranked = []
    for config, result in predictions.items():
        s = summary(result)
        full_ranked.append((
            s["risk_net_corrected_minus_2x_introduced"], s["corrected"],
            -s["introduced"], -s["intervention_rate"], -config[0], -config[1], config, s,
        ))
    full_best = max(full_ranked)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: safe-gate development exists: {output}")
    headroom.to_csv(output / "error_headroom.csv", index=False)
    oof.to_csv(output / "formula_oof_transitions.csv.gz", index=False, compression="gzip")
    report = {
        "status": "bioaware_metdna3_safe_gate_development_complete", "formal": True,
        "error_headroom": {
            "error_rotation_instances": int(len(headroom)),
            "unique_error_queries": int(headroom["query_id"].nunique()),
            "network_can_rank_truth_first_instances": int(headroom["network_can_rank_truth_first"].sum()),
            "network_can_rank_truth_first_unique_queries": int(
                headroom[headroom["network_can_rank_truth_first"]]["query_id"].nunique()
            ),
        },
        "formula_group_oof": {**oof_result, "selected_configs": selected},
        "full_development_selection_for_future_freeze": {
            "weight": full_best[-2][0], "maximum_margin": full_best[-2][1],
            "minimum_advantage": full_best[-2][2], **full_best[-1],
        },
        "gates": {
            "oof_corrected_gt_introduced": oof_result["corrected"] > oof_result["introduced"],
            "oof_risk_net_positive": oof_result["risk_net_corrected_minus_2x_introduced"] > 0,
            "oof_every_formula_fold_nonnegative": all(item["heldout"]["delta_recall1"] >= 0 for item in selected),
        },
        "contracts": {
            "development_only": True, "P2b": "forbidden",
            "RP_or_external_test_opened": False,
            "selection": "five-fold formula-group OOF; risk objective corrected - 2*introduced",
        },
        "claim_limit": "Development gate selection and headroom only; no locked validation claim.",
    }
    report["gates"]["pass_to_freeze"] = all(report["gates"].values())
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


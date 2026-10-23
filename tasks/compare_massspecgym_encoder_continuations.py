#!/usr/bin/env python
"""Compare Noise V1, matched continuation and ChemAware continuation on held MSG."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--chemistry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    return parser.parse_args()


def read(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "query_index", "query_row", "query_ik14", "query_formula", "near",
        "candidate_rank", "candidate_reciprocal_rank",
    }
    missing = sorted(required - set(table))
    if missing:
        raise RuntimeError(f"{path} lacks held columns: {missing}")
    if len(table) != 18333 or table.query_index.duplicated().any():
        raise RuntimeError(f"{path} is not the complete 18,333-query held panel")
    return table.sort_values("query_index", kind="stable").reset_index(drop=True)


def cluster_ci(
    formula: np.ndarray, value: np.ndarray, repeats: int, seed: int,
) -> dict[str, float | int]:
    grouped = pd.DataFrame({"formula": formula, "value": value}).groupby(
        "formula", sort=True,
    ).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), size=len(grouped))
        samples[index] = sums[take].sum() / counts[take].sum()
    return {
        "delta_pp": float(100 * np.mean(value)),
        "ci_low_pp": float(100 * np.quantile(samples, 0.025)),
        "ci_high_pp": float(100 * np.quantile(samples, 0.975)),
        "formula_clusters": int(len(grouped)),
        "bootstrap_resamples": int(repeats),
    }


def metrics(table: pd.DataFrame) -> dict[str, object]:
    rank = table.candidate_rank.to_numpy(int)
    near = table.near.to_numpy(bool)
    return {
        "queries": int(len(rank)),
        "recall@1": float(np.mean(rank <= 1)),
        "recall@2": float(np.mean(rank <= 2)),
        "recall@3": float(np.mean(rank <= 3)),
        "recall@5": float(np.mean(rank <= 5)),
        "recall@10": float(np.mean(rank <= 10)),
        "mrr": float(np.mean(1.0 / rank)),
        "near_queries": int(np.sum(near)),
        "near_recall@1": float(np.mean(rank[near] <= 1)),
        "near_mrr": float(np.mean(1.0 / rank[near])),
    }


def paired(
    baseline: pd.DataFrame, candidate: pd.DataFrame, repeats: int, seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline_rank = baseline.candidate_rank.to_numpy(int)
    candidate_rank = candidate.candidate_rank.to_numpy(int)
    corrected = (baseline_rank != 1) & (candidate_rank == 1)
    introduced = (baseline_rank == 1) & (candidate_rank != 1)
    delta = (candidate_rank == 1).astype(float) - (baseline_rank == 1).astype(float)
    near = baseline.near.to_numpy(bool)
    output = baseline[[
        "query_index", "query_row", "query_ik14", "query_formula", "near",
    ]].copy()
    output["baseline_rank"] = baseline_rank
    output["candidate_rank"] = candidate_rank
    output["corrected"] = corrected
    output["introduced"] = introduced
    summary = {
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_net_lambda2": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "mrr_delta": float(np.mean(1.0 / candidate_rank - 1.0 / baseline_rank)),
        "recall1_formula_cluster_ci": cluster_ci(
            baseline.query_formula.to_numpy(str), delta, repeats, seed,
        ),
        "near_recall1_delta_pp": float(100 * np.mean(delta[near])),
        "near_corrected": int(np.sum(corrected[near])),
        "near_introduced": int(np.sum(introduced[near])),
    }
    return summary, output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    tables = {
        "noise_v1": read(args.noise),
        "matched_continuation": read(args.control),
        "chemical_continuation": read(args.chemistry),
    }
    keys = ("query_index", "query_row", "query_ik14", "query_formula", "near")
    reference = tables["noise_v1"]
    for name, table in tables.items():
        if not table[list(keys)].equals(reference[list(keys)]):
            raise RuntimeError(f"held query axis drifted for {name}")
    comparisons: dict[str, object] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for index, (name, baseline, candidate) in enumerate((
        ("chemistry_vs_noise_v1", "noise_v1", "chemical_continuation"),
        ("chemistry_vs_matched_continuation", "matched_continuation", "chemical_continuation"),
        ("matched_continuation_vs_noise_v1", "noise_v1", "matched_continuation"),
    )):
        comparisons[name], ledgers[name] = paired(
            tables[baseline], tables[candidate], args.bootstrap_resamples,
            args.bootstrap_seed + 100 * index,
        )
    chem_noise = comparisons["chemistry_vs_noise_v1"]
    chem_control = comparisons["chemistry_vs_matched_continuation"]
    pass_noise = chem_noise["recall1_formula_cluster_ci"]["ci_low_pp"] > 0
    pass_control = chem_control["recall1_formula_cluster_ci"]["ci_low_pp"] > 0
    pass_near = chem_noise["near_recall1_delta_pp"] >= 0
    advance = bool(pass_noise and pass_control and pass_near)
    report = {
        "status": "MASSSPECGYM_ENCODER_CONTINUATION_COMPARISON_COMPLETE",
        "dataset": "MassSpecGym corrected outer-held fold 0",
        "development_queries": 18333,
        "training_rows_in_evaluation": False,
        "methods": {name: metrics(table) for name, table in tables.items()},
        "comparisons": comparisons,
        "advance_to_gnps": advance,
        "decision": (
            "ADVANCE_CHEMISTRY_ENCODER_TO_GNPS"
            if advance else "STOP_BEFORE_GNPS"
        ),
        "advance_requirements": {
            "chemistry_vs_noise_recall1_ci_low_positive": bool(pass_noise),
            "chemistry_vs_matched_continuation_recall1_ci_low_positive": bool(pass_control),
            "near_recall1_noninferior_point_estimate": bool(pass_near),
        },
        "claim_boundary": (
            "MassSpecGym development selection only; GNPS is not read by this comparison"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="msg_encoder_compare_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        for name, ledger in ledgers.items():
            ledger.to_csv(
                temporary / f"{name}.csv.gz", index=False, compression="gzip",
            )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

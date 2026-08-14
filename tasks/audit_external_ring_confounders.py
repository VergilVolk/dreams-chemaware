"""Audit whether ring-stratified E0 differences survive measurable confounders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def bootstrap_logit(frame: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    molecule = frame.groupby("ik14", sort=False).agg(
        top1=("top1_correct", "mean"),
        precursor_mz=("precursor_mz", "first"),
        peak_count=("peak_count", "mean"),
        positive_proxy_similarity=("positive_proxy_similarity", "first"),
        n_negative_molecules=("n_negative_molecules", "mean"),
        ring_class=("ring_class", "first"),
    ).reset_index()
    ring = pd.get_dummies(molecule["ring_class"], dtype=float)
    for name in ("acyclic", "single_ring", "multi_ring"):
        if name not in ring: ring[name] = 0.0
    continuous = molecule[["precursor_mz", "peak_count", "positive_proxy_similarity", "n_negative_molecules"]].to_numpy(float)
    continuous = StandardScaler().fit_transform(continuous)
    # multi_ring is the reference class.
    design = np.column_stack((ring["acyclic"], ring["single_ring"], continuous))
    names = ["acyclic_vs_multi_ring", "single_ring_vs_multi_ring", "precursor_mz_z", "peak_count_z", "positive_proxy_similarity_z", "n_negative_molecules_z"]
    # Two views produce a molecule-level success fraction {0, .5, 1}; expand
    # back to two Bernoulli observations with the same molecule covariates.
    query = frame.merge(molecule[["ik14"]], on="ik14")
    row_index = {ik: i for i, ik in enumerate(molecule["ik14"])}
    x = np.stack([design[row_index[ik]] for ik in frame["ik14"]])
    y = frame["top1_correct"].astype(int).to_numpy()
    model = LogisticRegression(C=1.0, max_iter=2000).fit(x, y)
    point = model.coef_[0]
    rng = np.random.default_rng(seed)
    molecules = molecule["ik14"].to_numpy()
    coefficients = []
    for _ in range(iterations):
        sampled = rng.choice(molecules, size=len(molecules), replace=True)
        indices = np.concatenate([np.flatnonzero(frame["ik14"].to_numpy() == ik) for ik in sampled])
        xb, yb = x[indices], y[indices]
        if len(np.unique(yb)) < 2: continue
        coefficients.append(LogisticRegression(C=1.0, max_iter=1000).fit(xb, yb).coef_[0])
    coefficients = np.asarray(coefficients)
    return pd.DataFrame({
        "term": names,
        "coefficient": point,
        "odds_ratio": np.exp(point),
        "ci_low": np.quantile(coefficients, 0.025, axis=0),
        "ci_high": np.quantile(coefficients, 0.975, axis=0),
        "odds_ratio_ci_low": np.exp(np.quantile(coefficients, 0.025, axis=0)),
        "odds_ratio_ci_high": np.exp(np.quantile(coefficients, 0.975, axis=0)),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--e0-dir", type=Path, default=Path("data/validation/external_ring_balanced_e0"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_ring_confounder_audit"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.e0_dir / "query_results.csv")
    reports = {}
    all_coefficients = []
    all_summary = []
    for split in ("discovery", "confirmation"):
        manifest = json.loads((args.pilot_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        units = pd.DataFrame(manifest)
        spectra = np.load(args.pilot_dir / f"{split}_spectra.npz")["spectra"]
        peak_counts = (spectra[:, :, 0, :] > 0).sum(axis=-1)
        count_rows = []
        for pair_id in range(len(spectra)):
            for view in (0, 1): count_rows.append({"pair_id": pair_id, "query_view": view, "peak_count": int(peak_counts[pair_id, view])})
        counts = pd.DataFrame(count_rows)
        frame = results.loc[(results["split"] == split) & (results["candidate_protocol"] == "negative_pair_ids")].copy()
        frame = frame.merge(units[["pair_id", "precursor_mz", "positive_proxy_similarity"]], on="pair_id", validate="many_to_one")
        frame = frame.merge(counts, on=["pair_id", "query_view"], validate="one_to_one")
        summary = frame.groupby("ring_class").agg(
            molecules=("ik14", "nunique"), top1=("top1_correct", "mean"),
            precursor_mz_median=("precursor_mz", "median"),
            peak_count_median=("peak_count", "median"),
            positive_proxy_median=("positive_proxy_similarity", "median"),
            candidates_median=("n_negative_molecules", "median"),
            positive_embedding_similarity=("positive_similarity", "mean"),
            best_negative_similarity=("best_negative_similarity", "mean"),
        ).reset_index()
        summary["split"] = split
        all_summary.append(summary)
        coefficients = bootstrap_logit(frame, args.bootstrap, args.seed + (0 if split == "discovery" else 1))
        coefficients["split"] = split
        all_coefficients.append(coefficients)
        reports[split] = {
            "observations": len(frame), "molecules": int(frame["ik14"].nunique()),
            "model": "L2-regularized logistic regression; molecule-cluster bootstrap; multi-ring reference",
        }
    pd.concat(all_summary, ignore_index=True).to_csv(args.output_dir / "ring_summary.csv", index=False)
    coefficient_frame = pd.concat(all_coefficients, ignore_index=True)
    coefficient_frame.to_csv(args.output_dir / "adjusted_logistic_coefficients.csv", index=False)
    report = {
        "status": "external_ring_confounder_audit",
        "splits": reports,
        "interpretation_limit": "Small balanced pilot; coefficients test robustness to measured covariates, not causal effects of ring count.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(pd.concat(all_summary, ignore_index=True).to_string(index=False))
    print(coefficient_frame.to_string(index=False))


if __name__ == "__main__":
    main()

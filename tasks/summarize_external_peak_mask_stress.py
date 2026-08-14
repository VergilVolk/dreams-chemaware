"""Molecule-clustered summary of the external peak masking stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def molecule_metric(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return frame.groupby("ik14", sort=False)[metric].mean().to_numpy(float)


def ci(values: np.ndarray, seed: int, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def difference_ci(left: np.ndarray, right: np.ndarray, seed: int, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    a = left[rng.integers(0, len(left), size=(n, len(left)))].mean(axis=1)
    b = right[rng.integers(0, len(right), size=(n, len(right)))].mean(axis=1)
    return np.quantile(a - b, [0.025, 0.975]).tolist()


def summarize_group(frame: pd.DataFrame, seed: int) -> dict:
    clean_correct = frame.loc[frame["clean_top1_correct"].astype(bool)]
    flip = molecule_metric(clean_correct, "correct_to_wrong")
    margin = molecule_metric(frame, "margin_drop")
    cosine = molecule_metric(frame, "embedding_cosine_to_clean")
    return {
        "molecules": int(frame["ik14"].nunique()),
        "clean_correct_molecules": int(clean_correct["ik14"].nunique()),
        "conditional_correct_to_wrong_rate": float(flip.mean()),
        "conditional_correct_to_wrong_ci95": ci(flip, seed),
        "margin_drop_mean": float(margin.mean()),
        "margin_drop_ci95": ci(margin, seed + 1),
        "embedding_cosine_mean": float(cosine.mean()),
        "embedding_cosine_ci95": ci(cosine, seed + 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, default=Path("data/validation/external_peak_mask_discovery/perturbation_results.csv"))
    parser.add_argument("--confirmation", type=Path, default=Path("data/validation/external_peak_mask_confirmation/perturbation_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_peak_mask_final"))
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    report = {
        "status": "external_peak_mask_stress_final",
        "selection": "20% native masking selected on discovery because 30% caused larger non-specific embedding displacement; confirmation was not used for rate selection.",
        "confirmation_hypothesis": "The high-risk acyclic query domain is more sensitive to peak masking than single- or multi-ring queries.",
        "splits": {},
    }
    sources = {"discovery": pd.read_csv(args.discovery), "confirmation": pd.read_csv(args.confirmation)}
    for split, frame in sources.items():
        frame = frame.loc[frame["candidate_protocol"] == "negative_pair_ids"].copy()
        if split == "discovery":
            frame = frame.loc[(frame["mode"] == "native_mask") & np.isclose(frame["mask_rate"], 0.2)]
        report["splits"][split] = {}
        for mode, mode_frame in frame.groupby("mode"):
            report["splits"][split][mode] = {}
            grouped_arrays = {}
            for position, ring in enumerate(("acyclic", "single_ring", "multi_ring")):
                subset = mode_frame.loc[mode_frame["ring_class"] == ring]
                summary = summarize_group(subset, args.seed + 100 * position + (0 if split == "discovery" else 1000))
                report["splits"][split][mode][ring] = summary
                correct = subset.loc[subset["clean_top1_correct"].astype(bool)]
                grouped_arrays[ring] = molecule_metric(correct, "correct_to_wrong")
                outputs.append({"split": split, "mode": mode, "ring_class": ring} | summary)
            report["splits"][split][mode]["acyclic_minus_multi_ring_flip_ci95"] = difference_ci(
                grouped_arrays["acyclic"], grouped_arrays["multi_ring"], args.seed + 4000
            )
            report["splits"][split][mode]["acyclic_minus_single_ring_flip_ci95"] = difference_ci(
                grouped_arrays["acyclic"], grouped_arrays["single_ring"], args.seed + 5000
            )
    confirmation = report["splits"]["confirmation"]["native_mask"]
    replicated = (
        confirmation["acyclic_minus_multi_ring_flip_ci95"][0] > 0
        and confirmation["acyclic_minus_single_ring_flip_ci95"][0] > 0
    )
    report["hypothesis_supported"] = bool(replicated)
    report["decision"] = (
        "Acyclic-specific masking sensitivity replicated; eligible for targeted robustness training."
        if replicated else
        "Acyclic-specific sensitivity did not replicate. Masking is a valid general stress test, but does not justify acyclic-targeted training."
    )
    pd.DataFrame(outputs).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

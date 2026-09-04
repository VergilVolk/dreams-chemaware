"""Render full query/reference mirror spectra for the frozen LCNEC priorities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
from audit_e0_observability_residual import greedy_matches, peaks  # noqa: E402
from encode_mona_neg_library import parse_mgf  # noqa: E402


PRIORITY = {
    "XTWYTFMLZFPYCI": "ADP family",
    "SRNWOUGRCWSEMX": "ADP-ribose family",
    "CIWBSHSKHKDKBQ": "Ascorbate",
    "GJAWHXHKYYXBSV": "Quinolinate",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/validation/lcnec_hsst3n_all_robust_annotation/priority_annotation_primary20.csv")
    parser.add_argument("--top5", type=Path, default=ROOT / "data/validation/lcnec_hsst3n_all_robust_annotation/priority_annotation_top5.csv")
    parser.add_argument("--query-mgf", type=Path, default=ROOT / "data/validation/lcnec_hsst3n_all_robust_ms2/priority_dark_modules.mgf")
    parser.add_argument("--library-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/lcnec_hsst3n_manuscript_figures")
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations)
    annotations = annotations.loc[
        annotations["p2b_top_ik14"].isin(PRIORITY)
        & annotations["annotation_confidence"].str.contains("consistency", na=False)
    ].copy()
    top5 = pd.read_csv(args.top5)
    queries = parse_mgf(args.query_mgf)
    references = parse_mgf(args.library_mgf)

    order = ["XTWYTFMLZFPYCI", "SRNWOUGRCWSEMX", "CIWBSHSKHKDKBQ", "GJAWHXHKYYXBSV"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, ik14 in zip(axes.flat, order, strict=True):
        annotation = annotations.loc[annotations["p2b_top_ik14"] == ik14]
        if len(annotation) != 1:
            raise RuntimeError(f"expected one annotation for {ik14}, found {len(annotation)}")
        row = annotation.iloc[0]
        qidx = int(row["query_index"])
        selected = top5.loc[
            top5["query_index"].eq(qidx)
            & top5["ppm_window"].eq(20)
            & top5["rank"].eq(1)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"expected one 20-ppm rank-1 reference for query {qidx}")
        ridx = int(selected.iloc[0]["reference_index"])
        q_mz, q_intensity = peaks(queries[qidx]["peaks"])
        r_mz, r_intensity = peaks(references[ridx]["peaks"])
        q_intensity = q_intensity / max(float(q_intensity.max()), 1e-12)
        r_intensity = r_intensity / max(float(r_intensity.max()), 1e-12)
        matches = greedy_matches(q_mz, r_mz, args.fragment_tolerance)
        q_match = np.zeros(len(q_mz), dtype=bool)
        r_match = np.zeros(len(r_mz), dtype=bool)
        for qi, ri in matches:
            q_match[qi] = True
            r_match[ri] = True

        ax.vlines(q_mz[~q_match], 0, q_intensity[~q_match], color="#AAB4BC", lw=0.8, alpha=0.65)
        ax.vlines(r_mz[~r_match], 0, -r_intensity[~r_match], color="#C2B7AE", lw=0.8, alpha=0.65)
        ax.vlines(q_mz[q_match], 0, q_intensity[q_match], color="#2F6B9A", lw=1.5)
        ax.vlines(r_mz[r_match], 0, -r_intensity[r_match], color="#D97925", lw=1.5)
        ax.axhline(0, color="#4C5660", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"{PRIORITY[ik14]}  |  {float(row['target_mz']):.5f} @ {float(row['target_rt_sec']):.1f}s\n"
            f"DreaMS={float(row['dreams_top_score']):.3f}; matched={len(matches)}"
        )
        ax.set_xlabel("Fragment m/z")
        ax.set_ylabel("QC query (+) / library reference (−)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.12)

    fig.suptitle("Frozen LCNEC priority spectra: full mirrors with matched peaks highlighted", fontsize=14, y=1.01)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "priority_full_mirror_spectra.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "priority_full_mirror_spectra.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] {args.output_dir / 'priority_full_mirror_spectra.png'}")


if __name__ == "__main__":
    main()

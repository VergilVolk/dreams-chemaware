"""Summarize the P0 dropout single-variable experiment across all 3 seeds.

Run on the server where the sbatch wrote its output:
  python tasks/summarize_p0_dropout.py

Reports per-seed and per-mode (dropout_on vs dropout_off): margin delta (+CI),
near-margin delta, pairwise-accuracy delta, Recall@1 / macro-AUC deltas (+CI),
corrected/introduced, plus mean/std/direction consistency across seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path("data/validation/m1b_p0_dropout")
SEEDS = [20260821, 20260822, 20260823]
MODES = ["dropout_on", "dropout_off"]


def load(mode, seed):
    f = BASE / mode / f"seed_{seed}" / "m1b_rank_gate.json"
    return json.loads(f.read_text(encoding="utf-8"))


def main() -> None:
    rows = {}
    for mode in MODES:
        for s in SEEDS:
            d = load(mode, s)
            m = d["margin"]
            strat = d["stratification"]
            ret = d["retrieval"]
            et = d["error_transition"]
            rb = d["retrieval_delta_bootstrap"]
            nb = strat["baseline"].get("grade=near", {})
            nc = strat["candidate"].get("grade=near", {})
            rows[f"{mode}_{s}"] = {
                "margin_delta": m["delta_candidate_minus_baseline"]["mean_delta"],
                "margin_ci_low": m["delta_candidate_minus_baseline"]["ci_low"],
                "margin_ci_high": m["delta_candidate_minus_baseline"]["ci_high"],
                "pairwise_delta": m["candidate"]["pairwise_accuracy"] - m["baseline"]["pairwise_accuracy"],
                "near_margin_delta": nc.get("margin_mean", 0.0) - nb.get("margin_mean", 0.0),
                "recall1_delta": rb["recall1"]["mean_delta"],
                "macro_auc_delta": rb["macro_auc"]["mean_delta"],
                "macro_auc_ci_low": rb["macro_auc"]["ci_low"],
                "macro_auc_ci_high": rb["macro_auc"]["ci_high"],
                "corrected": et["corrected"],
                "introduced": et["introduced"],
            }

    hdr = f"{'run':20} {'margin_d':>9} {'marginCI':>18} {'pairw_d':>8} {'near_d':>8} {'rec1_d':>8} {'auc_d':>8} {'aucCI':>18} {'corr':>5} {'intro':>6}"
    print(hdr)
    for k, v in rows.items():
        print(f"{k:20} {v['margin_delta']:9.4f} "
              f"[{v['margin_ci_low']:.4f},{v['margin_ci_high']:.4f}] {v['pairwise_delta']:8.4f} "
              f"{v['near_margin_delta']:8.4f} {v['recall1_delta']:8.4f} {v['macro_auc_delta']:8.4f} "
              f"[{v['macro_auc_ci_low']:.4f},{v['macro_auc_ci_high']:.4f}] {v['corrected']:5d} {v['introduced']:6d}")

    print("\n--- per-mode summary (mean / std / sign-consistent) ---")
    for mode in MODES:
        ms = [v for k, v in rows.items() if k.startswith(mode)]
        print(f"\n{mode}:")
        for key in ["margin_delta", "pairwise_delta", "near_margin_delta", "recall1_delta", "macro_auc_delta"]:
            vals = np.array([v[key] for v in ms])
            sign = all(x >= 0 for x in vals) or all(x <= 0 for x in vals)
            print(f"  {key:18} mean={vals.mean():.4f} std={vals.std():.4f} sign_consistent={sign}")
        corr = sum(v["corrected"] for v in ms)
        intro = sum(v["introduced"] for v in ms)
        print(f"  corrected_total={corr} introduced_total={intro} net={corr - intro:+d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot the frozen matched-background null for the modified-guanosine module."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


root = Path("data/mtbls13729/modified_guanosine_matched_background_v1")
null = pd.read_csv(root / "matched_background_null.csv.gz")
report = json.loads((root / "matched_background_report.json").read_text(encoding="utf-8"))
observed = {row["normalization"]: row["observed_mean_log2fc"] for row in report["results"]}
order = ["raw", "global_pqn_prev60", "global_pqn_prev80", "global_pqn_prev90"]
labels = ["Raw", "PQN >=60%", "PQN >=80%", "PQN >=90%"]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
for ax, name, label in zip(axes.ravel(), order, labels):
    block = null.loc[(null.normalization == name) & (null.n_pairs == 10), "mean_log2fc"].dropna()
    ax.hist(block, bins=35, color="#9ecae1", edgecolor="white", alpha=0.95)
    ax.axvline(observed[name], color="#b2182b", linewidth=2.4, label=f"Observed {observed[name]:.2f}")
    ax.axvline(block.quantile(0.99), color="#4d4d4d", linestyle="--", linewidth=1.4, label=f"Null P99 {block.quantile(0.99):.2f}")
    ax.set_title(label)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
fig.supxlabel("Mean paired Rmu/RN module log2 fold change", y=0.055)
fig.supylabel("Strictly matched random modules")
fig.suptitle("Modified-guanosine module exceeds the matched technical background", fontsize=14, weight="bold")
fig.text(0.5, 0.018, "1,412/2,000 random panels had all 10 patient pairs; empirical one-sided p = 0.000708 in every normalization", ha="center", fontsize=9)
fig.tight_layout(rect=(0.02, 0.10, 1, 0.95))
output = root / "modified_guanosine_matched_background.png"
fig.savefig(output, dpi=220, bbox_inches="tight")
print(output.resolve())

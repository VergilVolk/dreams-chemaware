#!/usr/bin/env python
"""Create a manuscript-ready structural evidence and identity-ceiling panel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONSENSUS = ROOT / "data/mtbls13729/frozen_candidate_ms2_consensus_v1/report.json"
TRANSITIONS = ROOT / "data/mtbls13729/fragmentation_standard_consistency_v1/fragmentation_standard_consistency.csv"
MASSBANK = ROOT / "data/mtbls13729/massbank_isomer_spectral_audit_v1/report.json"
OUTPUT = ROOT / "data/mtbls13729/structural_evidence_figure_v1"

LABELS = {
    1597: "1597\nmethylguanosine family",
    3019: "3019\ndimethylguanosine family",
    1717: "1717\ndiacetylspermidine-like",
    3222: "3222\nlong-chain acylcarnitine-like",
    4966: "4966\npurine-like",
}
EFFECT = {1597: 3.721, 3019: 2.401, 1717: 3.009, 3222: 1.776, 4966: 2.440}
CEILING = {
    1597: "Family supported; positional isomer unresolved",
    3019: "Family supported; positional isomer unresolved",
    1717: "Strong candidate; exact identity unconfirmed",
    3222: "Class supported; acyl-chain isomer unresolved",
    4966: "Purine-like family only",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(CONSENSUS.read_text(encoding="utf-8"))
    transitions = pd.read_csv(TRANSITIONS).set_index("feature_id")
    massbank = json.loads(MASSBANK.read_text(encoding="utf-8"))
    summaries = {int(item["feature_id"]): item for item in report["summary"]}
    features = list(LABELS)

    rows = []
    for feature in features:
        item = summaries[feature]
        transition = transitions.loc[feature] if feature in transitions.index else None
        fallback = item["top_fragments"][0]
        rows.append(
            {
                "feature_id": feature,
                "label": LABELS[feature],
                "rmu_log2fc": EFFECT[feature],
                "spectra": item["spectra"],
                "samples": item["samples"],
                "diagnostic_support": float(transition.support_fraction) if transition is not None else float(fallback["support_fraction"]),
                "diagnostic_mz": float(transition.observed_fragment_mz) if transition is not None else float(fallback["mz"]),
                "identity_ceiling": CEILING[feature],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "structural_evidence_table.csv", index=False)

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.0, 1.15])

    ax = fig.add_subplot(grid[0, 0])
    colors = ["#3266a8", "#4f86c6", "#ba5a31", "#4a8f6b", "#7963a8"]
    y = np.arange(len(frame))
    ax.barh(y, frame.rmu_log2fc, color=colors, height=0.67)
    ax.set_yticks(y, frame.label)
    ax.invert_yaxis()
    ax.set_xlabel("Rmu vs paired normal mean log2 fold-change")
    ax.set_title("A  Patient-paired abundance effect", loc="left", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.7)
    for i, value in enumerate(frame.rmu_log2fc):
        ax.text(value + 0.06, i, f"+{value:.2f}", va="center")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    width = 0.38
    x = np.arange(len(frame))
    ax.bar(x - width / 2, frame.spectra, width, label="peak-resolved MS2 spectra", color="#355f8a")
    ax.bar(x + width / 2, frame.samples, width, label="independent samples", color="#78a6c8")
    ax.set_xticks(x, [str(v) for v in frame.feature_id])
    ax.set_ylabel("Count")
    ax.set_xlabel("Feature ID")
    ax.set_title("B  Raw-MS2 recurrence", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    support = frame.diagnostic_support.fillna(0).to_numpy()
    ax.barh(y, support, color=colors, height=0.67)
    ax.set_yticks(y, [f"{f}: m/z {m:.4f}" if np.isfinite(m) else str(f) for f, m in zip(frame.feature_id, frame.diagnostic_mz)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Fraction of peak-resolved spectra containing diagnostic ion")
    ax.set_title("C  Diagnostic-fragment consistency", loc="left", fontweight="bold")
    for i, value in enumerate(support):
        ax.text(min(value + 0.02, 1.02), i, f"{value:.0%}", va="center")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    massbank_gap = massbank["best_isomer_score_gap"]
    lines = [
        ("Identity ceiling", "What the present evidence allows"),
        ("1597", f"MassBank authentic spectra support the family, but m7G vs m2G score gap = {massbank_gap:.4f}; unresolved."),
        ("3019", "312→180 supports a dimethylguanosine family; positional isomers require same-method standards."),
        ("1717", "230→100 occurs in 73/73 spectra and matches an authentic-standard transition; RT/full-spectrum mirror is absent."),
        ("3222", "85.028 and carnitine-related products support the class; C20:4 double-bond/position identity is unresolved."),
        ("4966", "Recurrent purine-like products support a companion family, not a unique structure."),
        ("Required upgrade", "Same-method authentic-standard RT + collision-energy series + sample spike-in coelution."),
    ]
    ax.text(0, 1.02, "D  Structural claim ceiling", transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")
    y0 = 0.93
    for key, value in lines:
        ax.text(0.01, y0, key, transform=ax.transAxes, fontweight="bold", va="top")
        ax.text(0.30, y0, value, transform=ax.transAxes, va="top", wrap=True)
        y0 -= 0.13 if key != "Identity ceiling" else 0.09

    fig.suptitle(
        "Structural evidence supports metabolite families while preserving isomer uncertainty",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "structural_evidence.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT / "structural_evidence.pdf", bbox_inches="tight")
    payload = {
        "status": "mtbls13729_structural_evidence_figure_complete",
        "features": features,
        "massbank_m7g_vs_m2g_gap": massbank_gap,
        "claim_limit": "Family-level structural evidence; no positional-isomer or MSI Level-1 claim.",
    }
    (OUTPUT / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

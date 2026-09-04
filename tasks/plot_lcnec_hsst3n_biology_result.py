"""Render the frozen LCNEC biology result without re-fitting annotations."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PALETTE = {
    "blue": "#2F6B9A",
    "orange": "#D97925",
    "green": "#3A8D5D",
    "red": "#B94A48",
    "gray": "#69737D",
    "light": "#EDF3F7",
}


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def cross_platform_plot(ledger: pd.DataFrame, output: Path) -> dict[str, float]:
    frame = ledger.loc[
        (ledger["author_status"] == "published_atlas_overlap")
        & ledger["author_beta_log10fc"].notna()
    ].copy()
    frame["author_log2fc"] = frame["author_beta_log10fc"].astype(float) / np.log10(2.0)
    frame["dark_log2fc"] = frame["dark_effect_log2fc"].astype(float)
    rho = float(frame[["author_log2fc", "dark_log2fc"]].corr(method="spearman").iloc[0, 1])
    concordance = float(
        np.mean(np.sign(frame["author_log2fc"]) == np.sign(frame["dark_log2fc"]))
    )

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(
        frame["author_log2fc"],
        frame["dark_log2fc"],
        s=65,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    label_offsets = {
        "CREATINE": (5, 12),
        "PANTOTHENIC ACID": (5, -14),
        "Glutathione (oxidized)": (5, 0),
        "Guanosine": (-42, -12),
        "PHENYLACETYL-GLUTAMINE": (5, 10),
        "Guanine": (5, -12),
    }
    for row in frame.itertuples():
        label = html.unescape(str(row.spectral_hypothesis)).replace("ADENOSINE 5'-", "")
        dx, dy = label_offsets.get(html.unescape(str(row.spectral_hypothesis)), (4, 4))
        ax.annotate(
            label.title(),
            (row.author_log2fc, row.dark_log2fc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
        )
    low = float(min(frame["author_log2fc"].min(), frame["dark_log2fc"].min(), 0))
    high = float(max(frame["author_log2fc"].max(), frame["dark_log2fc"].max(), 0))
    ax.axhline(0, color="#AAB2B8", lw=0.8)
    ax.axvline(0, color="#AAB2B8", lw=0.8)
    ax.plot([low, high], [low, high], ls="--", lw=1, color="#9AA4AB")
    ax.set_xlabel("Published atlas effect (converted to log2 scale)")
    ax.set_ylabel("Re-extracted HSST3n paired effect (log2FC)")
    ax.set_title(
        f"Orthogonal cross-platform reproduction\n"
        f"n={len(frame)}, direction={concordance:.0%}, Spearman ρ={rho:.3f}"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16, zorder=0)
    save_figure(fig, output, "cross_platform_reproduction")
    return {"n": int(len(frame)), "direction_concordance": concordance, "spearman_rho": rho}


def pair_effect_plot(effects: pd.DataFrame, output: Path) -> dict[str, dict[str, float]]:
    order = [
        "adenosine_diphosphate_family",
        "adenosine_diphosphoribose_family",
        "ascorbate",
        "quinolinate",
    ]
    titles = ["ADP family", "ADP-ribose family", "Ascorbate", "Quinolinate"]
    colors = [PALETTE["blue"], "#5D78B5", PALETTE["orange"], PALETTE["green"]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=False)
    report: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(20260831)
    for ax, name, title, color in zip(axes.flat, order, titles, colors, strict=True):
        values = effects.loc[effects["priority_name"] == name, "per_mg_log2fc_tumor_vs_normal"].to_numpy(float)
        x = rng.normal(0.0, 0.045, size=len(values))
        ax.scatter(x, values, s=28, alpha=0.72, color=color, edgecolor="white", linewidth=0.35)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ci = 2.0345 * sem
        ax.errorbar(0, mean, yerr=ci, fmt="D", ms=6, color="#1F2933", capsize=4, lw=1.5)
        ax.axhline(0, color="#8F989F", ls="--", lw=0.9)
        concordance = float(np.mean(values > 0))
        ax.set_xlim(-0.22, 0.22)
        ax.set_xticks([])
        ax.set_title(f"{title}\nmean={mean:+.2f}; positive pairs={np.sum(values > 0)}/{len(values)}")
        ax.set_ylabel("Tumor − adjacent tissue (log2, per mg)")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        report[name] = {"n": int(len(values)), "mean_log2fc": mean, "positive_fraction": concordance}
    fig.suptitle("Patient-pair consistency of four author-unreported hypotheses", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, output, "priority_pair_effects")
    return report


def evidence_map(ledger: pd.DataFrame, output: Path) -> None:
    selected = ledger.loc[
        ledger["module"].isin(["free_nucleoside_or_base", "phosphorylated_nucleotide_or_sugar", "nad_adenylate_turnover", "redox_buffering"])
    ].copy()
    selected["name"] = selected["spectral_hypothesis"].map(lambda value: html.unescape(str(value)))
    selected["effect"] = selected["dark_effect_log2fc"].astype(float)

    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    groups = [
        ("Free nucleoside\n/ base", "free_nucleoside_or_base", 0.4, 3.0, 2.4, 3.1),
        ("Phosphorylated\nnucleotide / sugar", "phosphorylated_nucleotide_or_sugar", 3.0, 3.0, 3.3, 3.1),
        ("NAD-related\ncontext", "nad_adenylate_turnover", 6.55, 3.0, 2.4, 3.1),
        ("Antioxidant pools", "redox_buffering", 9.2, 3.0, 3.35, 3.1),
    ]
    for label, module, x, y, width, height in groups:
        box = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=PALETTE["light"], edgecolor="#9BAAB5", linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(x + 0.12, y + height - 0.28, label, fontsize=10.2, weight="bold", va="top", linespacing=0.95)
        rows = selected.loc[selected["module"] == module].sort_values("effect", ascending=False)
        yy = y + height - (0.95 if "\n" in label else 0.75)
        for row in rows.itertuples():
            display_names = {
                "ADENOSINE 5'-DIPHOSPHORIBOSE": "ADP-ribose",
                "ADENOSINE 5'-DIPHOSPHATE": "ADP family",
                "Uridine-5-diphosphoacetylgalactosamine": "UDP-HexNAc",
                "Guanosine-5'-monophosphate": "GMP",
                "Adenosine 5'-monophosphate": "AMP",
                "[(2R,3S,4R,5R)-5-(6-aminopurin-9-yl)-4-hydroxy-2-(hydroxymethyl)oxolan-3-yl] dihydrogen phosphate": "AMP family",
            }
            short = display_names.get(row.name, row.name)
            if row.author_status == "published_atlas_overlap":
                evidence_tag = "[R]"
                font_weight = "normal"
            elif str(row.priority_novel_hypothesis).lower() == "true":
                evidence_tag = "[N]"
                font_weight = "bold"
            else:
                evidence_tag = "[H]"
                font_weight = "normal"
            color = PALETTE["red"] if row.effect < 0 else PALETTE["green"]
            ax.text(
                x + 0.18,
                yy,
                f"{evidence_tag} {short}: {row.effect:+.2f}",
                fontsize=9.0,
                color=color,
                va="top",
                weight=font_weight,
            )
            yy -= 0.38

    ax.text(0.45, 7.35, "Measured abundance evidence", fontsize=16, weight="bold")
    ax.text(
        0.45, 6.98,
        "Boxes summarize co-occurring pool changes; their placement does not imply reaction direction or flux.",
        fontsize=10, color=PALETTE["gray"],
    )
    ax.text(
        0.45,
        6.65,
        "[R] source-atlas metabolite reproduced across platforms    "
        "[N] author-unreported priority hypothesis    [H] other Level-2/family hypothesis",
        fontsize=9.4,
        color="#34495E",
    )
    ax.text(
        6.65, 2.62,
        "BioAware context\nQuinolinate: de novo NAD anchor\nADP-ribose: NUDT context\nADP: hub abstention",
        fontsize=9.2, color="#34495E", va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#8FA4B2", "linestyle": "--"},
    )
    ax.text(
        0.45, 0.48,
        "Allowed interpretation: phosphorylated-nucleotide/NAD-related pool redistribution and expanded antioxidant pools.\n"
        "Identity boundary: [N]/[H] remain MSI Level 2 or molecular-family hypotheses. "
        "Forbidden: ATP energy charge, enzyme activity, pathway flux, or causal adaptation.",
        fontsize=10.2, color="#313A40",
    )
    save_figure(fig, output, "abundance_evidence_map")


def matched_fragment_mirrors(fragments: pd.DataFrame, output: Path) -> dict[str, int]:
    order = [104, 102, 109, 169]
    titles = {104: "ADP family", 102: "ADP-ribose family", 109: "Ascorbate", 169: "Quinolinate"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    counts: dict[str, int] = {}
    for ax, family_id in zip(axes.flat, order, strict=True):
        frame = fragments.loc[fragments["family_id"] == family_id].sort_values("query_fragment_mz")
        ax.vlines(frame["query_fragment_mz"], 0, frame["query_relative_intensity"], color=PALETTE["blue"], lw=1.6)
        ax.vlines(frame["reference_fragment_mz"], 0, -frame["reference_relative_intensity"], color=PALETTE["orange"], lw=1.6)
        ax.axhline(0, color="#4C5660", lw=0.8)
        ax.set_title(f"{titles[family_id]} — {len(frame)} matched fragments")
        ax.set_xlabel("Fragment m/z")
        ax.set_ylabel("Query (+) / library reference (−)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.12)
        counts[titles[family_id]] = int(len(frame))
    fig.suptitle("Direct matched-fragment evidence (unmatched peaks intentionally omitted)", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, output, "priority_matched_fragment_mirrors")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--biology-ledger", type=Path, default=Path("data/validation/lcnec_hsst3n_annotation_biology/identity_evidence_ledger.csv"))
    parser.add_argument("--pair-effects", type=Path, default=Path("data/validation/lcnec_hsst3n_priority_pair_consistency/per_patient_effects.csv"))
    parser.add_argument("--matched-fragments", type=Path, default=Path("data/validation/lcnec_hsst3n_priority_structure/matched_fragments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lcnec_hsst3n_manuscript_figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.biology_ledger)
    effects = pd.read_csv(args.pair_effects)
    fragments = pd.read_csv(args.matched_fragments)
    report = {
        "status": "lcnec_hsst3n_manuscript_figures_complete",
        "cross_platform": cross_platform_plot(ledger, args.output_dir),
        "priority_pair_effects": pair_effect_plot(effects, args.output_dir),
        "matched_fragment_counts": matched_fragment_mirrors(fragments, args.output_dir),
        "contracts": {
            "annotations_refit": False,
            "phenotype_used_for_identity": False,
            "network_edges_are_causal": False,
        },
    }
    evidence_map(ledger, args.output_dir)
    (args.output_dir / "figure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

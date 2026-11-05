#!/usr/bin/env python3
"""Summarize paired CALMS advantages from frozen GNPS development reports.

This script never recomputes scores or selects a model.  It only changes the
reference orientation of existing paired, formula-cluster bootstrap results
so that positive values consistently mean that frozen CALMS is better.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CALMS_INTERNAL = "full23_v132"
CALMS_RELEASE = "CALMS v1.0"
FULL_PANEL_QUERIES = {
    "identity_disjoint": 10_995,
    "formula_disjoint": 5_261,
}
DISPLAY = {
    "official_dreams": "DreaMS",
    "noise_v1": "Noise V1",
    "cosine_greedy": "Cosine",
    "modified_cosine": "Modified cosine",
    "weighted_spectral_entropy": "WSE",
    "p2b_sqrt_cosine": "P2b sqrt cosine",
    "p2b_unweighted_entropy": "P2b entropy",
    "neutral_loss_sqrt_cosine": "Neutral-loss cosine",
    "p2b_official_frozen": "P2b on DreaMS",
    "p2b_noise_v1_frozen": "P2b on Noise",
    "ms2deepscore_2x_public": "MS2DeepScore 2.x",
    "spec2vec_gnps_public": "Spec2Vec (public)",
    "entropy_raw_public": "Entropy search",
    "denoising_search_public": "Denoising search",
    "spec2vec_2026_retrained": "Spec2Vec (retrained)",
    "specembedding_official": "SpecEmbedding",
    "msbert_official": "MSBERT",
}
FIGURE_METHODS = (
    "official_dreams",
    "weighted_spectral_entropy",
    "noise_v1",
    "ms2deepscore_2x_public",
    "spec2vec_gnps_public",
    "msbert_official",
    "specembedding_official",
)


def load_panel(path: Path, panel: str) -> tuple[list[dict], dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise RuntimeError(f"{panel}: paired report is incomplete")
    if report.get("reference_method") != CALMS_INTERNAL:
        raise RuntimeError(f"{panel}: paired report is not CALMS-referenced")
    if report.get("evaluation_role") != "development" or report.get("truth_status") != "consumed":
        raise RuntimeError(f"{panel}: expected consumed development evidence")
    if report.get("allow_final_claim"):
        raise RuntimeError(f"{panel}: development report cannot allow a final claim")
    definitions = report.get("paired_delta_definitions", {})
    if definitions.get("recall/mrr/auroc/average_precision") != (
        "candidate minus reference; positive is better"
    ):
        raise RuntimeError(f"{panel}: paired delta orientation is not recognized")
    methods = report["methods"]
    calms = methods[CALMS_INTERNAL]
    panel_meta = report["panel"]
    queries = int(calms["retrieval"]["queries"])
    if int(panel_meta["N_Q"]) != queries:
        raise RuntimeError(f"{panel}: panel and CALMS query counts differ")
    if panel not in FULL_PANEL_QUERIES or queries > FULL_PANEL_QUERIES[panel]:
        raise RuntimeError(f"{panel}: unexpected common-domain denominator")
    rows: list[dict] = []
    for method, paired in report["paired_comparisons"].items():
        if method not in methods:
            raise RuntimeError(f"{panel}: paired method {method} is absent")
        if paired.get("reference_method") != CALMS_INTERNAL:
            raise RuntimeError(f"{panel}/{method}: wrong paired reference")
        if paired.get("candidate_method") != method:
            raise RuntimeError(f"{panel}/{method}: wrong paired candidate")
        if int(paired.get("queries", -1)) != queries:
            raise RuntimeError(f"{panel}/{method}: paired query count differs")
        if int(methods[method]["retrieval"]["queries"]) != queries:
            raise RuntimeError(f"{panel}/{method}: method query count differs")
        bootstrap = paired["formula_cluster_bootstrap"]
        if bootstrap.get("cluster_unit") != "query_formula":
            raise RuntimeError(f"{panel}/{method}: wrong cluster unit")
        if int(bootstrap["clusters"]) != int(panel_meta["N_F"]):
            raise RuntimeError(f"{panel}/{method}: formula-cluster count differs")
        metric = bootstrap["metrics"]["recall@1"]
        delta = float(metric["delta"])
        ci_low = float(metric["ci_low"])
        ci_high = float(metric["ci_high"])
        if not ci_low <= delta <= ci_high:
            raise RuntimeError(f"{panel}/{method}: invalid interval ordering")
        # Stored deltas are candidate minus CALMS.  Reverse point and bounds.
        rows.append({
            "panel": panel,
            "queries": queries,
            "method_id": method,
            "method": DISPLAY.get(method, method),
            "calms_recall_at_1": float(calms["retrieval"]["recall@1"]),
            "competitor_recall_at_1": float(methods[method]["retrieval"]["recall@1"]),
            "calms_advantage_pp": -100.0 * delta,
            "ci_low_pp": -100.0 * ci_high,
            "ci_high_pp": -100.0 * ci_low,
            "calms_corrected": int(paired["introduced"]),
            "calms_introduced": int(paired["corrected"]),
            "calms_mrr": float(calms["retrieval"]["mrr"]),
            "competitor_mrr": float(methods[method]["retrieval"]["mrr"]),
            "calms_micro_auroc": float(calms["micro_candidate"]["auroc"]),
            "competitor_micro_auroc": float(methods[method]["micro_candidate"]["auroc"]),
        })
    return rows, report


def forest_plot(table: pd.DataFrame, path: Path) -> None:
    selected = table.loc[table["method_id"].isin(FIGURE_METHODS)].copy()
    order = [DISPLAY[name] for name in FIGURE_METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for axis, panel, title in zip(
        axes,
        ("identity_disjoint", "formula_disjoint"),
        ("Identity-disjoint", "Formula-disjoint"),
    ):
        block = selected.loc[selected["panel"].eq(panel)].set_index("method").loc[order]
        y = list(range(len(block)))
        value = block["calms_advantage_pp"].to_numpy()
        low = block["ci_low_pp"].to_numpy()
        high = block["ci_high_pp"].to_numpy()
        axis.errorbar(
            value,
            y,
            xerr=[value - low, high - value],
            fmt="o",
            color="#1f5f99",
            ecolor="#7d9fc1",
            elinewidth=1.6,
            capsize=3,
            markersize=5.5,
        )
        axis.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        axis.set_title(f"{title} (n={int(block['queries'].iloc[0]):,})", fontsize=11)
        axis.set_xlabel("CALMS Recall@1 advantage (percentage points)")
        axis.grid(axis="x", color="#e4e8ec", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_yticks(y)
        if axis is axes[0]:
            axis.set_yticklabels(block.index)
        else:
            axis.tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle("Paired CALMS advantage on consumed GNPS development panels", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Within-comparator formula-cluster intervals; positive values favor CALMS. Not external confirmation.",
        ha="center",
        fontsize=8.5,
        color="#4c5661",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--formula-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows: list[dict] = []
    reports = {}
    for panel, path in (
        ("identity_disjoint", args.identity_report),
        ("formula_disjoint", args.formula_report),
    ):
        panel_rows, report = load_panel(path, panel)
        rows.extend(panel_rows)
        reports[panel] = {
            "report": path.as_posix(),
            "reference_method": report["reference_method"],
            "evaluation_role": report["evaluation_role"],
            "truth_status": report["truth_status"],
            "queries": int(report["panel"]["N_Q"]),
            "molecule_edges": int(report["panel"]["N_E"]),
            "formula_clusters": int(report["panel"]["N_F"]),
            "axis_digest": report["panel"]["axis_digest"],
            "full_panel_queries": FULL_PANEL_QUERIES[panel],
            "common_domain_retention": (
                int(report["panel"]["N_Q"]) / FULL_PANEL_QUERIES[panel]
            ),
            "competitors": int(len(report["paired_comparisons"])),
        }
    table = pd.DataFrame(rows).sort_values(["panel", "calms_advantage_pp"])
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "calms_paired_superiority.tsv", sep="\t", index=False)
    forest_plot(table, args.output_dir / "calms_paired_superiority.png")
    tsv_path = args.output_dir / "calms_paired_superiority.tsv"
    png_path = args.output_dir / "calms_paired_superiority.png"
    pdf_path = args.output_dir / "calms_paired_superiority.pdf"
    payload = {
        "status": "CALMS_PAIRED_SUPERIORITY_SUMMARY_COMPLETE",
        "method_name": CALMS_RELEASE,
        "internal_method_id": CALMS_INTERNAL,
        "internal_method_version": "Full23 v1.3.2",
        "reports": reports,
        "methods_total_per_panel": int(table["method_id"].nunique() + 1),
        "competitors_per_panel": int(table["method_id"].nunique()),
        "panel_comparisons": int(len(table)),
        "interval_semantics": (
            "Each interval is formula-cluster bootstrap and familywise-adjusted "
            "over 20 endpoints within one CALMS-versus-competitor comparison. "
            "It is not simultaneous across competitors or panels and does not "
            "account for prior CALMS development or scale selection on GNPS."
        ),
        "artifacts": {
            "table": tsv_path.name,
            "figure_png": png_path.name,
            "figure_pdf": pdf_path.name,
            "table_sha256": hashlib.sha256(tsv_path.read_bytes()).hexdigest(),
            "figure_png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "figure_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        },
        "claim_boundary": (
            "Consumed GNPS development evidence. Paired superiority and intervals "
            "do not constitute unopened external confirmation or a field-wide SOTA claim."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

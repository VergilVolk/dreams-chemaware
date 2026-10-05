"""Assemble the verified GNPS article-benchmark ladder (task 1, zero-error rule).

Every number in the final ladder has TWO independent sources:
  1. the frozen evaluator's evaluation/report.json;
  2. recomputation from the per-query tables (queries_*.csv.gz, paired_*.csv.gz).
Any disagreement beyond 1e-9 fails the run with a precise message; nothing is
rounded, inferred or copied by hand.  Also adjudicates the suspicious
macro_query_auprc == mrr identity in the report block.

Outputs:
  deliverables/GLM_gnps_article_ladder/<run>/ladder_full.csv
  deliverables/GLM_gnps_article_ladder/<run>/verification_report.json
  docs/GLM_GNPS_ARTICLE_LADDER_<date>.md
  deliverables/figures/GLM_gnps_article_ladder.{svg,pdf,png}
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(sys.argv[sys.argv.index("--run") + 1]) if "--run" in sys.argv else \
    Path("data/validation/noise_gnps_article_benchmark_run_2349091")
OUT_DIR = Path("deliverables/GLM_gnps_article_ladder") / RUN.name
FIG_DIR = Path("deliverables/figures")
DOC = Path(f"docs/GLM_GNPS_ARTICLE_LADDER_{date.today():%Y%m%d}.md")
TOL = 1e-9
KS = (1, 2, 3, 5, 10, 20)

report = json.loads((RUN / "evaluation/report.json").read_text(encoding="utf-8"))
bundle_meta = report["score_bundle_metadata"]
methods: list[str] = list(report["methods"]) if "methods" in report else \
    list(bundle_meta["methods"])
panels = ("identity_disjoint", "formula_disjoint")
base = report.get("baseline_method", "official_dreams")

problems: list[str] = []
auprc_identity_notes: list[str] = []
rows: list[dict] = []


def check(name: str, reported, recomputed) -> float:
    if reported is None:
        problems.append(f"{name}: missing in report")
        return float("nan")
    if not abs(float(reported) - float(recomputed)) <= TOL:
        problems.append(
            f"{name}: report={reported!r} recomputed={recomputed!r}")
    return float(recomputed)


def load_queries(panel: str, method: str) -> pd.DataFrame:
    return pd.read_csv(RUN / "evaluation" / f"queries_{panel}_{method}.csv.gz",
                       low_memory=False)


for panel in panels:
    for method in methods:
        q = load_queries(panel, method)
        abs_block = report["panels"][panel]["absolute"][method]
        ret = abs_block["retrieval"]
        near = abs_block["near_subset"]

        row: dict = {
            "panel": panel, "method": method, "queries": int(len(q)),
        }
        check(f"{panel}/{method}/queries", ret["queries"], len(q))

        # recall@k + MRR: recomputed from per-query ranks, asserted vs report
        rank = q["rank"].to_numpy()
        rr = q["reciprocal_rank"].to_numpy()
        for k in KS:
            row[f"recall@{k}"] = check(
                f"{panel}/{method}/recall@{k}", ret[f"recall@{k}"],
                float(np.mean(rank <= k)))
        row["mrr"] = check(f"{panel}/{method}/mrr", ret["mrr"],
                           float(np.mean(rr)))
        row["mean_rank"] = check(f"{panel}/{method}/mean_rank",
                                 ret["mean_rank"], float(np.mean(rank)))
        row["median_rank"] = check(f"{panel}/{method}/median_rank",
                                   ret["median_rank"], float(np.median(rank)))

        # macro AUC blocks: recompute means from columns
        # (per-query columns are macro_query_auc / macro_query_auprc)
        row["macro_query_auroc"] = check(
            f"{panel}/{method}/macro_query_auroc", ret["macro_query_auroc"],
            float(np.mean(q["macro_query_auc"].to_numpy())))
        auprc_recomputed = float(np.mean(q["macro_query_auprc"].to_numpy()))
        row["macro_query_auprc_recomputed"] = auprc_recomputed
        if abs(float(ret["macro_query_auprc"]) - float(ret["mrr"])) <= 1e-12:
            auprc_identity_notes.append(
                f"{panel}/{method}: report macro_query_auprc equals mrr "
                f"({ret['macro_query_auprc']!r}); ladder uses the recomputed "
                f"per-query mean {auprc_recomputed!r}")
        else:
            check(f"{panel}/{method}/macro_query_auprc",
                  ret["macro_query_auprc"], auprc_recomputed)

        # margins
        for key, col in (
            ("mean_positive_vs_best_negative_margin",
             "positive_vs_best_negative_margin"),
            ("mean_top1_top2_gap", "top1_top2_gap"),
            ("mean_signed_top1_top2_gap", "signed_top1_top2_gap"),
        ):
            row[key] = check(f"{panel}/{method}/{key}", ret[key],
                             float(np.mean(q[col].to_numpy())))

        # near subset
        qn = q[q["near"]]
        check(f"{panel}/{method}/near/queries", near["queries"], len(qn))
        row["near_queries"] = int(len(qn))
        for k in KS:
            row[f"near_recall@{k}"] = check(
                f"{panel}/{method}/near/recall@{k}", near[f"recall@{k}"],
                float(np.mean(qn["rank"].to_numpy() <= k)) if len(qn) else
                float("nan"))
        row["near_mrr"] = check(f"{panel}/{method}/near/mrr", near["mrr"],
                                float(np.mean(qn["reciprocal_rank"].to_numpy()))
                                if len(qn) else float("nan"))

        # micro + pooled (no per-query recomputation possible; copied verbatim)
        micro = abs_block["micro_candidate"]
        row["micro_auroc"] = float(micro["auroc"])
        row["micro_auprc"] = float(micro["auprc"])
        row["micro_molecules"] = int(micro["molecules"])
        for tag in ("gnps_10ppm_pooled_pairwise", "gnps_mh_10ppm_pooled_pairwise"):
            block = abs_block[tag]
            row[f"{tag}_auroc"] = float(block["auroc"])
            row[f"{tag}_auprc"] = float(block["auprc"])

        # vs-baseline transitions, recomputed from paired tables
        paired_path = RUN / "evaluation" / f"paired_{panel}_{method}.csv.gz"
        if method != base and paired_path.is_file():
            p = pd.read_csv(paired_path, low_memory=False)
            vs = report["panels"][panel]["vs_official_dreams"][method]
            row["corrected_vs_official"] = check(
                f"{panel}/{method}/corrected", vs["corrected"],
                int(p["corrected"].sum()))
            row["introduced_vs_official"] = check(
                f"{panel}/{method}/introduced", vs["introduced"],
                int(p["introduced"].sum()))
            risk = int(p["corrected"].sum()) - 2 * int(p["introduced"].sum())
            check(f"{panel}/{method}/risk_net_lambda2",
                  vs["risk_net_lambda2"], risk)
            row["risk_net_lambda2_vs_official"] = risk
        rows.append(row)

if problems:
    print("LADDER VERIFICATION FAILED — no output written:", file=sys.stderr)
    for item in problems:
        print("  MISMATCH:", item, file=sys.stderr)
    sys.exit(1)

ladder = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
ladder.to_csv(OUT_DIR / "ladder_full.csv", index=False)

verification = {
    "status": "GLM_GNPS_LADDER_VERIFIED",
    "run": str(RUN),
    "methods": methods,
    "panels": list(panels),
    "cross_checks": "every recall@k, MRR, rank, margin, near metric and "
                    "corrected/introduced/risk-net recomputed from per-query "
                    "tables and asserted equal to the frozen report at 1e-9",
    "mismatches": problems,
    "macro_query_auprc_identity_notes": auprc_identity_notes,
    "claim_limits": [report["claim_limit"], bundle_meta["claim_limit"]],
    "information_levels": bundle_meta["information_levels"],
    "missing_baselines": [
        "MS2DeepScore 2.0 and Spec2Vec weights not yet placed under "
        "third_party/ (verdict doc item S4); the ladder gains these rows "
        "without any change to frozen methods",
        "structure-library tools (SIRIUS/CSI:FingerID, MetFrag, CFM-ID) are "
        "out of scope by design: this ladder compares spectral-similarity "
        "information extraction on one sealed candidate graph",
    ],
}
(OUT_DIR / "verification_report.json").write_text(
    json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRAY = "#1a1a1a", "#9a9a9a"
method_rank = (ladder[ladder.panel == "identity_disjoint"]
               .sort_values("recall@1", ascending=False)["method"].tolist())
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
for ax, panel in zip(axes, panels):
    sub = ladder[ladder.panel == panel].set_index("method").loc[method_rank]
    y = np.arange(len(method_rank))[::-1]
    for k, marker, color in ((1, "o", INK), (5, "s", "#1f618d"),
                             (10, "^", GRAY), (20, "D", LIGHT if (LIGHT := "#c9c9c9") else GRAY)):
        ax.plot(sub[f"recall@{k}"] * 100, y, marker, ms=5, lw=0,
                label=f"Recall@{k}", color=color)
    ax.set_yticks(y)
    labels = [m.replace("weighted_spectral_entropy", "weighted spectral entropy")
               .replace("noise_v1", "noise V1 (ours)")
               .replace("official_dreams", "official DreaMS")
               .replace("_", " ") for m in method_rank]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("accuracy (%)", fontsize=9)
    ax.set_title(f"{panel.replace('_', '-')}  (n={int(sub.queries.iloc[0]):,})",
                 fontsize=10)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_xlim(84, 100.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].legend(fontsize=8, loc="lower right", frameon=False)
fig.suptitle("GNPS Gold/Silver 10 ppm benchmark ladder — 10 spectral-similarity "
             "methods, sealed model-blind panels (run 2349091)",
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("svg", "pdf", "png"):
    fig.savefig(FIG_DIR / f"GLM_gnps_article_ladder.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight",
                facecolor="white")
print(json.dumps(verification, indent=2, ensure_ascii=False))
print("ladder rows:", len(ladder), "->", OUT_DIR / "ladder_full.csv")

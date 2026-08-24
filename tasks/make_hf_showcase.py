"""Generate hf_space/data/showcase.json from the MTBLS13729 smoke results.

This is the *curated* data bundle that the static Chem-aware DreaMS platform UI
renders. It is deliberately small (a few KB, not the 33 MB full CSV) because a
static HF Space must carry everything in the repo and the browser only needs a
showcase, not the full table.

Everything emitted here is *real* -- read straight off the smoke CSVs produced by
the annotation pipeline (annotation/). The only hand-authored blocks are the
COSMIC / frozen-probe headline numbers, which come from the SLURM log (they are
not materialised as a CSV), and are labelled as such.

Usage (conda dreams_env):
    python tasks/make_hf_showcase.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "data/mtbls13729/smoke"
OUT = ROOT / "hf_space/data/showcase.json"

CONFIDENT_COS = 0.7


def _round(x, nd=4):
    try:
        return None if pd.isna(x) else round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _sanitize(obj):
    """Make the bundle strict-JSON-safe: float('inf')/-inf/NaN are not valid
    JSON and would break JSON.parse in the browser, so they become None."""
    import math
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def main() -> int:
    fdr = pd.read_csv(SMOKE / "ann/annotations_fdr.csv")
    diff = pd.read_csv(SMOKE / "ann/diff_tumor_vs_normal.csv")
    novel = pd.read_csv(SMOKE / "author_compare/novel_pos_rp.csv")
    summ = pd.read_csv(SMOKE / "author_compare/_summary.csv")
    report = json.loads((SMOKE / "ann/report.json").read_text(encoding="utf-8"))

    top1 = fdr[fdr["rank"] == 1].copy()
    confident = top1[(top1["cosine"] >= CONFIDENT_COS) & (top1["mz_pass"] == True)]  # noqa: E712

    def _row(r):
        return {
            "query_file": r.get("query_file"),
            "query_scan": r.get("query_scan"),
            "query_precursor_mz": _round(r.get("query_precursor_mz")),
            "cosine": _round(r.get("cosine")),
            "dppm": _round(r.get("dppm")),
            "mz_pass": bool(r.get("mz_pass")),
            "schymanski_level": int(r.get("schymanski_level")),
            "lib_name": r.get("lib_name"),
            "lib_inchikey": r.get("lib_inchikey"),
            "lib_smiles": r.get("lib_smiles"),
            "lib_precursor_mz": _round(r.get("lib_precursor_mz")),
        }

    confident_matches = confident.sort_values("cosine", ascending=False).head(50)
    matches = [{"rank": 1, **_row(r)} for _, r in confident_matches.iterrows()]

    # Schymanski level distribution over top-1 hits (the confidence ladder).
    schymanski = (
        top1["schymanski_level"].value_counts().sort_index().to_dict()
    )
    schymanski = {int(k): int(v) for k, v in schymanski.items()}

    # q-value / FDR summary.
    n_total_top1 = int(len(top1))
    n_fdr_pass = int(top1["fdr_pass"].sum()) if "fdr_pass" in top1.columns else 0
    qvals = top1["qvalue"].dropna() if "qvalue" in top1.columns else pd.Series(dtype=float)
    fdr_summary = {
        "n_top1": n_total_top1,
        "n_fdr_pass": n_fdr_pass,
        "q_median": _round(qvals.median()) if len(qvals) else None,
        "q_min": _round(qvals.min()) if len(qvals) else None,
        "n_q_le_0_05": int((qvals <= 0.05).sum()) if len(qvals) else 0,
    }

    diff_rows = []
    for _, r in diff.iterrows():
        diff_rows.append({
            "lib_inchikey": r.get("lib_inchikey"),
            "lib_name": r.get("lib_name"),
            "n_samples_tumor": int(r.get("n_samples_Tumor", 0)),
            "n_samples_normal": int(r.get("n_samples_Normal", 0)),
            "n_spectra_tumor": int(r.get("n_spectra_Tumor", 0)),
            "n_spectra_normal": int(r.get("n_spectra_Normal", 0)),
            "odds_ratio": _round(r.get("odds_ratio")),
            "p_value": _round(r.get("p_value")),
            "q_value": _round(r.get("q_value")),
        })

    novel_rows = []
    for _, r in novel.iterrows():
        novel_rows.append({k: (int(v) if isinstance(v, (int, float)) and not pd.isna(v) else v)
                           for k, v in r.items() if v is not None and str(v) != ""})

    summ_rows = []
    for _, r in summ.iterrows():
        summ_rows.append({k: v for k, v in r.items()})

    # Frozen concept-probe + COSMIC headline numbers (from SLURM logs, not a CSV).
    cosmic_probe = {
        "_note": "Headline numbers from the COSMIC Layer-1 SLURM log "
                 "(tasks/run_cosmic_full.sbatch), not materialised as a CSV.",
        "n_rules": 266,
        "probe_train": {"n_molecules": 3994},
        "probe_val": {"n_molecules": 856},
        "probe_test": {"n_molecules": 856, "auprc": 0.63, "baseline_auprc": 0.20},
        "coherence_decoy": {
            "note": "shuffle decoys are WEAK: structure-space decoy coherence does "
                    "not separate targets from decoys (Mann-Whitney P=0.44).",
            "p_value": 0.44,
            "verdict": "null -- coherence score is a self-consistency signal, NOT P(correct)",
        },
        "self_retrieval_fdr": {
            "mz_constrained": 4.19,
            "cosine_only": 10.10,
            "note": "ground-truth FDR measured on the full MassSpecGym library -- the "
                    "reliable confidence scale.",
        },
    }

    overview = {
        "n_queries": int(report.get("n_queries", top1["query_idx"].nunique())),
        "n_top1_annotated": n_total_top1,
        "n_confident": int(len(confident)),
        "n_distinct_confident_inchikey": int(confident["lib_inchikey"].nunique()),
        "annotation_rate": report.get("annotation_rate"),
        "confident_rate": round(len(confident) / n_total_top1, 6) if n_total_top1 else None,
        "schymanski_distribution": schymanski,
        "fdr": fdr_summary,
    }

    bundle = _sanitize({
        "_source": "generated by tasks/make_hf_showcase.py from data/mtbls13729/smoke/*",
        "overview": overview,
        "confident_matches": matches,
        "differential_analysis": diff_rows,
        "novel_findings": novel_rows,
        "author_compare_summary": summ_rows,
        "cosmic_probe": cosmic_probe,
        "report_ladder": report,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"     overview.n_queries={overview['n_queries']} "
          f"confident={overview['n_confident']} distinct={overview['n_distinct_confident_inchikey']}")
    print(f"     diff_rows={len(diff_rows)} novel_rows={len(novel_rows)} "
          f"matches={len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

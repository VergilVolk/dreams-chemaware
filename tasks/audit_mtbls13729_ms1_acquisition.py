#!/usr/bin/env python
"""Read-only acquisition audit for the four MTBLS13729 LC-MS panels.

This script intentionally stops before feature detection.  It verifies that the
raw mzML files support a defensible paired MS1 analysis and produces a compact
CSV/JSON/Markdown audit bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lxml import etree


PANELS = ("neg_hilic", "neg_rp", "pos_hilic", "pos_rp")
TIMESTAMP_RE = re.compile(r"startTimeStamp=[\"']([^\"']+)[\"']")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _percentile(values: list[float], q: float) -> float:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.percentile(vals, q)) if vals.size else math.nan


def _median(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(statistics.median(vals)) if vals else math.nan


def _read_start_timestamp(path: Path) -> str | None:
    # The run element occurs near the beginning. Read incrementally so no binary
    # arrays are touched merely to recover acquisition order.
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        text = ""
        for _ in range(400):
            line = handle.readline()
            if not line:
                break
            text += line
            match = TIMESTAMP_RE.search(text)
            if match:
                return match.group(1)
    return None


def audit_file(payload: tuple[str, str]) -> dict[str, Any]:
    panel, path_str = payload
    path = Path(path_str)
    started = time.time()
    row: dict[str, Any] = {
        "panel": panel,
        "file": path.name,
        "sample_name": path.stem,
        "path": str(path),
        "file_size_mib": path.stat().st_size / (1024**2),
        "start_timestamp": _read_start_timestamp(path),
        "parse_ok": False,
        "error": "",
    }
    ms1_rts: list[float] = []
    ms1_tics: list[float] = []
    ms1_bpis: list[float] = []
    ms1_points: list[float] = []
    ms2_rts: list[float] = []
    polarity_positive = 0
    polarity_negative = 0

    try:
        # Stream only spectrum metadata.  Generic mzML readers materialize a
        # rich Python object for every spectrum and are unnecessarily slow for
        # this audit; lxml skips decoding the large binary peak arrays.
        spectrum_tag = "{http://psi.hupo.org/ms/mzml}spectrum"
        context = etree.iterparse(str(path), events=("end",), tag=spectrum_tag, huge_tree=True)
        for _, spec in context:
            level = 0
            rt = math.nan
            tic = math.nan
            bpi = math.nan
            for cv in spec.iterfind(".//{http://psi.hupo.org/ms/mzml}cvParam"):
                accession = cv.get("accession")
                if accession == "MS:1000511":
                    level = int(float(cv.get("value", "0")))
                elif accession == "MS:1000016":
                    rt = _as_float(cv.get("value"))
                    # mzML permits seconds, though these files use minutes.
                    if cv.get("unitAccession") == "UO:0000010":
                        rt /= 60.0
                elif accession == "MS:1000285":
                    tic = _as_float(cv.get("value"))
                elif accession == "MS:1000505":
                    bpi = _as_float(cv.get("value"))
                elif accession == "MS:1000130":
                    polarity_positive += 1
                elif accession == "MS:1000129":
                    polarity_negative += 1
            points = _as_float(spec.get("defaultArrayLength"))
            if level == 1:
                ms1_rts.append(rt)
                ms1_tics.append(tic)
                ms1_bpis.append(bpi)
                ms1_points.append(points)
            elif level == 2:
                ms2_rts.append(rt)
            spec.clear()
            while spec.getprevious() is not None:
                del spec.getparent()[0]
        del context
        row["parse_ok"] = True
    except Exception as exc:  # keep auditing the other 239 files
        row["error"] = f"{type(exc).__name__}: {exc}"

    row.update(
        {
            "n_ms1": len(ms1_rts),
            "n_ms2": len(ms2_rts),
            "ms1_rt_min": min(ms1_rts) if ms1_rts else math.nan,
            "ms1_rt_max": max(ms1_rts) if ms1_rts else math.nan,
            "ms1_rt_span": (max(ms1_rts) - min(ms1_rts)) if ms1_rts else math.nan,
            "ms1_tic_median": _median(ms1_tics),
            "ms1_tic_p05": _percentile(ms1_tics, 5),
            "ms1_tic_p95": _percentile(ms1_tics, 95),
            "ms1_bpi_median": _median(ms1_bpis),
            "ms1_points_median": _median(ms1_points),
            "positive_scan_count": polarity_positive,
            "negative_scan_count": polarity_negative,
            "audit_seconds": time.time() - started,
        }
    )
    return row


def robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return 0.67448975 * (x - med) / mad


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def summarize_panel(panel_df: pd.DataFrame) -> dict[str, Any]:
    df = panel_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_timestamp"], errors="coerce", utc=True)
    if df["start_dt"].notna().any():
        df = df.sort_values(["start_dt", "file"]).reset_index(drop=True)
    else:
        df = df.sort_values("file").reset_index(drop=True)
    df["injection_order"] = np.arange(1, len(df) + 1)

    metrics = ("n_ms1", "n_ms2", "ms1_rt_span", "ms1_tic_median", "ms1_bpi_median", "ms1_points_median")
    for metric in metrics:
        df[f"rz_{metric}"] = robust_z(df[metric])
    rz_cols = [f"rz_{m}" for m in metrics]
    df["acquisition_outlier"] = df[rz_cols].abs().max(axis=1) > 3.5

    tumor = (df["tissue"].astype(str).str.lower() == "tumor").astype(int)
    order_tissue_r = safe_spearman(df["injection_order"], tumor)
    metric_order_r = {metric: safe_spearman(df["injection_order"], df[metric]) for metric in metrics}

    pair_gaps: list[int] = []
    pair_minutes: list[float] = []
    pair_log2_tic: list[float] = []
    pair_log2_bpi: list[float] = []
    tumor_before = 0
    tumor_after = 0
    for _, group in df.groupby("patient", dropna=True):
        if len(group) != 2:
            continue
        tumor_row = group[group["tissue"].astype(str).str.lower() == "tumor"]
        normal_row = group[group["tissue"].astype(str).str.lower() == "normal"]
        if len(tumor_row) != 1 or len(normal_row) != 1:
            continue
        signed_gap = int(tumor_row["injection_order"].iloc[0] - normal_row["injection_order"].iloc[0])
        pair_gaps.append(signed_gap)
        tumor_before += int(signed_gap < 0)
        tumor_after += int(signed_gap > 0)
        if group["start_dt"].notna().all():
            dt = abs((group["start_dt"].iloc[0] - group["start_dt"].iloc[1]).total_seconds()) / 60
            pair_minutes.append(float(dt))
        for metric, sink in (("ms1_tic_median", pair_log2_tic), ("ms1_bpi_median", pair_log2_bpi)):
            t_val = _as_float(tumor_row[metric].iloc[0])
            n_val = _as_float(normal_row[metric].iloc[0])
            if t_val > 0 and n_val > 0:
                sink.append(float(math.log2(t_val / n_val)))

    polarity_expected = "positive" if df["panel"].iloc[0].startswith("pos_") else "negative"
    wrong_polarity = int(
        (df["negative_scan_count"] > 0).sum()
        if polarity_expected == "positive"
        else (df["positive_scan_count"] > 0).sum()
    )

    return {
        "panel": str(df["panel"].iloc[0]),
        "n_files": int(len(df)),
        "n_parse_ok": int(df["parse_ok"].sum()),
        "n_with_timestamp": int(df["start_dt"].notna().sum()),
        "n_acquisition_outliers": int(df["acquisition_outlier"].sum()),
        "wrong_polarity_files": wrong_polarity,
        "ms1_count_median": float(df["n_ms1"].median()),
        "ms1_count_min": int(df["n_ms1"].min()),
        "ms1_count_max": int(df["n_ms1"].max()),
        "ms2_count_median": float(df["n_ms2"].median()),
        "rt_span_median_min": float(df["ms1_rt_span"].median()),
        "tic_median_across_files": float(df["ms1_tic_median"].median()),
        "order_vs_tumor_spearman": order_tissue_r,
        "metric_vs_order_spearman": metric_order_r,
        "pair_order_gap_median": _median([float(v) for v in pair_gaps]),
        "pair_abs_order_gap_median": _median([float(abs(v)) for v in pair_gaps]),
        "pair_abs_order_gap_p95": _percentile([float(abs(v)) for v in pair_gaps], 95),
        "pair_time_gap_median_min": _median(pair_minutes),
        "pair_time_gap_p95_min": _percentile(pair_minutes, 95),
        "tumor_before_normal_pairs": tumor_before,
        "tumor_after_normal_pairs": tumor_after,
        "adjacent_pair_fraction": float(np.mean([abs(v) == 1 for v in pair_gaps])) if pair_gaps else math.nan,
        "median_log2_tumor_normal_tic": _median(pair_log2_tic),
        "fraction_tumor_higher_tic": float(np.mean([v > 0 for v in pair_log2_tic])) if pair_log2_tic else math.nan,
        "median_log2_tumor_normal_bpi": _median(pair_log2_bpi),
        "fraction_tumor_higher_bpi": float(np.mean([v > 0 for v in pair_log2_bpi])) if pair_log2_bpi else math.nan,
        "rows": df,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(summaries: list[dict[str, Any]], df: pd.DataFrame) -> str:
    parse_failures = int((~df["parse_ok"]).sum())
    missing_ms1 = int((df["n_ms1"] == 0).sum())
    wrong_counts = sum(int(s["n_files"] != 60) for s in summaries)
    strong_group_confounding = any(
        math.isfinite(float(s["order_vs_tumor_spearman"])) and abs(float(s["order_vs_tumor_spearman"])) >= 0.5
        for s in summaries
    )
    strong_drift = any(
        any(math.isfinite(float(v)) and abs(float(v)) >= 0.5 for v in s["metric_vs_order_spearman"].values())
        for s in summaries
    )
    fixed_pair_order = any(
        max(int(s["tumor_before_normal_pairs"]), int(s["tumor_after_normal_pairs"])) >= 0.9 * int(s["n_files"] / 2)
        for s in summaries
    )
    hard_pass = parse_failures == 0 and missing_ms1 == 0 and wrong_counts == 0
    conditional_pass = hard_pass and not strong_group_confounding and not (strong_drift and fixed_pair_order)

    lines = [
        "# MTBLS13729 MS1 acquisition audit",
        "",
        "This is a read-only acquisition audit. It does not perform feature detection, normalization, imputation, annotation, or differential testing.",
        "",
        "## Gate decision",
        "",
        f"- File/parse/MS1 completeness: **{'PASS' if hard_pass else 'FAIL'}**",
        f"- Global tumor-status vs injection-order correlation: **{'PASS' if not strong_group_confounding else 'REVIEW'}**",
        f"- Within-pair acquisition order: **{'REVIEW: fixed tissue order' if fixed_pair_order else 'PASS'}**",
        f"- Run-order signal drift screen: **{'REVIEW' if strong_drift else 'no strong monotonic drift detected'}**",
        f"- Proceed to prototype feature extraction: **{'YES, conditionally' if conditional_pass else 'NO / manual review first'}**",
        "",
        "The paired design controls stable patient-to-patient differences; it does not by itself correct run-order drift.",
        "",
        "## Panel summary",
        "",
        "| Panel | Files | Parse | MS1 median (min–max) | MS2 median | RT span min | Order–tumor rho | Absolute pair gap, median/P95 | Pair gap min, median/P95 | Outliers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {panel} | {n_files} | {n_parse_ok} | {ms1_count_median:.0f} ({ms1_count_min}–{ms1_count_max}) | "
            "{ms2_count_median:.0f} | {rt_span_median_min:.2f} | {order} | {gap}/{gap95} | {mins}/{mins95} | {outliers} |".format(
                panel=s["panel"],
                n_files=s["n_files"],
                n_parse_ok=s["n_parse_ok"],
                ms1_count_median=s["ms1_count_median"],
                ms1_count_min=s["ms1_count_min"],
                ms1_count_max=s["ms1_count_max"],
                ms2_count_median=s["ms2_count_median"],
                rt_span_median_min=s["rt_span_median_min"],
                order=fmt(s["order_vs_tumor_spearman"]),
                gap=fmt(s["pair_abs_order_gap_median"], 1),
                gap95=fmt(s["pair_abs_order_gap_p95"], 1),
                mins=fmt(s["pair_time_gap_median_min"], 1),
                mins95=fmt(s["pair_time_gap_p95_min"], 1),
                outliers=s["n_acquisition_outliers"],
            )
        )

    lines += ["", "## Monotonic run-order correlations", ""]
    for s in summaries:
        correlations = ", ".join(f"{k}={fmt(v)}" for k, v in s["metric_vs_order_spearman"].items())
        lines.append(f"- **{s['panel']}**: {correlations}")

    lines += [
        "",
        "## Paired-order audit",
        "",
        "| Panel | Tumor before/after normal | Adjacent pairs | Median log2(T/N) TIC | T>N TIC | Median log2(T/N) BPI | T>N BPI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['panel']} | {s['tumor_before_normal_pairs']}/{s['tumor_after_normal_pairs']} | "
            f"{100 * s['adjacent_pair_fraction']:.1f}% | {fmt(s['median_log2_tumor_normal_tic'])} | "
            f"{100 * s['fraction_tumor_higher_tic']:.1f}% | {fmt(s['median_log2_tumor_normal_bpi'])} | "
            f"{100 * s['fraction_tumor_higher_bpi']:.1f}% |"
        )
    lines += [
        "",
        "A globally alternating tumor/normal sequence can have low rank correlation while still being confounded if every tumor is acquired before its matched normal. When signal also changes monotonically with run order, raw paired tumor/normal ratios can inherit a small systematic acquisition bias.",
    ]

    outliers = df[df["acquisition_outlier"] | (~df["parse_ok"])].copy()
    lines += ["", "## Files requiring review", ""]
    if outliers.empty:
        lines.append("No files crossed the robust |z| > 3.5 acquisition-level threshold.")
    else:
        lines += [
            "| Panel | File | Parse | MS1 | MS2 | TIC median | RT span | Error |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for _, r in outliers.sort_values(["panel", "file"]).iterrows():
            lines.append(
                f"| {r['panel']} | {r['file']} | {bool(r['parse_ok'])} | {int(r['n_ms1'])} | {int(r['n_ms2'])} | "
                f"{fmt(float(r['ms1_tic_median']))} | {fmt(float(r['ms1_rt_span']))} | "
                f"{'' if pd.isna(r['error']) else r['error']} |"
            )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- TIC/BPI summaries are acquisition-level diagnostics, not metabolite abundance estimates.",
        "- Without pooled QC and blanks, this audit cannot estimate analytical CV or remove contaminants.",
        "- A monotonic correlation screen can reveal obvious drift, but absence of correlation does not prove absence of batch effects.",
        "- Feature extraction should begin with RP positive and RP negative panels as complementary sphingolipid views.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--sample-groups", type=Path, default=Path("data/mtbls13729/sample_groups.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_acquisition_audit"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-panel", type=int, default=None, help="Smoke-test only; omit for the formal audit")
    parser.add_argument("--reuse-file-audit", action="store_true", help="Regenerate summaries from an existing file_audit.csv")
    args = parser.parse_args()

    root = args.mzml_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(args.sample_groups, sep="\t")
    groups["patient"] = groups["sample_name"].str.extract(r"^(P\d+)")

    csv_path = out / "file_audit.csv"
    if args.reuse_file_audit:
        if not csv_path.exists():
            raise SystemExit(f"Cannot reuse missing audit CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Reusing: {csv_path}", flush=True)
    else:
        payloads: list[tuple[str, str]] = []
        for panel in PANELS:
            panel_dir = root / panel
            paths = sorted(panel_dir.glob("*.mzML"))
            if args.limit_per_panel is not None:
                paths = paths[: args.limit_per_panel]
            for path in paths:
                payloads.append((panel, str(path)))
        if not payloads:
            raise SystemExit(f"No mzML files found under {root}")

        print(f"Auditing {len(payloads)} mzML files with {args.workers} workers...", flush=True)
        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(audit_file, p): p for p in payloads}
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                print(f"[{i:03d}/{len(payloads)}] {row['panel']}/{row['file']} MS1={row['n_ms1']} MS2={row['n_ms2']} ok={row['parse_ok']}", flush=True)
        df = pd.DataFrame(rows).merge(groups, on="sample_name", how="left", validate="many_to_one")
    summaries: list[dict[str, Any]] = []
    ordered_parts: list[pd.DataFrame] = []
    for panel in PANELS:
        summary = summarize_panel(df[df["panel"] == panel])
        ordered_parts.append(summary.pop("rows"))
        summaries.append(summary)
    df = pd.concat(ordered_parts, ignore_index=True)

    json_path = out / "panel_summary.json"
    report_path = out / "REPORT.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(build_report(summaries, df), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

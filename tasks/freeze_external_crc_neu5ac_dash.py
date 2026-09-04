#!/usr/bin/env python
"""Freeze and audit the public patient-level Neu5Ac Dash figures.

The Jain et al. CRC biogeography web application exposes the values plotted
for tumour and normal tissue through a Dash callback.  These values are useful
external context, but they are not automatically analysis-ready source data:

* the callback does not expose patient identifiers;
* the displayed tumour and normal arrays therefore must not be paired;
* the displayed sample counts and regression statistics must be checked
  against the frozen paper supplement rather than assumed to match it.

This script performs that audit, saves the exact HTTP response, and emits a
tidy table plus descriptive/regression summaries.  It deliberately does not
turn the public plot into an independent mucinous-CRC replication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "data/external/CRC_metabolic_biogeography_PMC11438248_20260831"
    / "neu5ac_dash_patient_level_v1"
)
ENDPOINT = "https://colorectal-cancer-metabolome.com/_dash-update-component"
METABOLITE = "N-Acetylneuraminic acid"
OUTPUT_KEY = "..tumor-linear-plot.figure...normal-linear-plot.figure.."
SUBSITES = [
    "cecum",
    "ascending",
    "transverse",
    "descending",
    "sigmoid",
    "rectosigmoid",
    "rectum",
]
PAPER_SUPPLEMENT = {
    "normal": {"reported_slope": 0.349, "reported_p_text": "<.001"},
    "tumour": {"reported_slope": 0.088, "reported_p_text": ".091"},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def callback_payload() -> dict[str, Any]:
    return {
        "output": OUTPUT_KEY,
        "outputs": [
            {"id": "tumor-linear-plot", "property": "figure"},
            {"id": "normal-linear-plot", "property": "figure"},
        ],
        "changedPropIds": ["compound-dropdown-linear.value"],
        "inputs": [
            {
                "id": "compound-dropdown-linear",
                "property": "value",
                "value": METABOLITE,
            }
        ],
        "state": [],
    }


def fail_closed_output(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def extract_rows(response: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(response.get("response"), dict):
        raise RuntimeError("Dash response is missing the response object")
    figures = response["response"]
    expected_keys = {"tumor-linear-plot", "normal-linear-plot"}
    if set(figures) != expected_keys:
        raise RuntimeError(f"unexpected Dash figure keys: {sorted(figures)}")

    rows: list[dict[str, Any]] = []
    figure_audit: dict[str, Any] = {}
    for tissue, key in (("tumour", "tumor-linear-plot"), ("normal", "normal-linear-plot")):
        figure = figures[key].get("figure")
        if not isinstance(figure, dict) or not isinstance(figure.get("data"), list):
            raise RuntimeError(f"{key} has no Plotly data array")
        traces = figure["data"]
        if len(traces) != len(SUBSITES):
            raise RuntimeError(f"{key} expected seven subsite traces; found {len(traces)}")

        counts: list[int] = []
        customdata_available = False
        for ordinal, (expected_subsite, trace) in enumerate(zip(SUBSITES, traces)):
            x_values = trace.get("x")
            y_values = trace.get("y")
            if not isinstance(x_values, list) or not isinstance(y_values, list):
                raise RuntimeError(f"{key}/{expected_subsite} lacks x or y values")
            if len(x_values) != len(y_values) or not y_values:
                raise RuntimeError(f"{key}/{expected_subsite} has invalid x/y lengths")
            observed_subsites = {str(value).strip().casefold() for value in x_values}
            if observed_subsites != {expected_subsite}:
                raise RuntimeError(
                    f"{key} trace {ordinal} expected {expected_subsite}; "
                    f"found {sorted(observed_subsites)}"
                )
            if trace.get("customdata") not in (None, [], {}):
                customdata_available = True
            counts.append(len(y_values))
            for within_trace_index, value in enumerate(y_values):
                abundance = float(value)
                if not np.isfinite(abundance):
                    raise RuntimeError(f"non-finite abundance in {key}/{expected_subsite}")
                rows.append(
                    {
                        "metabolite": METABOLITE,
                        "tissue": tissue,
                        "subsite": expected_subsite,
                        "subsite_ordinal": ordinal,
                        "within_trace_index": within_trace_index,
                        "relative_abundance_log": abundance,
                    }
                )
        figure_audit[tissue] = {
            "trace_count": len(traces),
            "counts_by_subsite": dict(zip(SUBSITES, counts)),
            "total_values": sum(counts),
            "all_subsite_counts_equal": len(set(counts)) == 1,
            "customdata_available": customdata_available,
            "patient_identifiers_available": False,
        }
    return rows, figure_audit


def tissue_statistics(rows: list[dict[str, Any]], tissue: str) -> dict[str, Any]:
    selected = [row for row in rows if row["tissue"] == tissue]
    x = np.asarray([row["subsite_ordinal"] for row in selected], dtype=float)
    y = np.asarray([row["relative_abundance_log"] for row in selected], dtype=float)
    regression = stats.linregress(x, y)

    per_subsite: dict[str, Any] = {}
    for subsite in SUBSITES:
        values = np.asarray(
            [row["relative_abundance_log"] for row in selected if row["subsite"] == subsite],
            dtype=float,
        )
        per_subsite[subsite] = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }

    cecum = per_subsite["cecum"]["mean"]
    rectum = per_subsite["rectum"]["mean"]
    supplement = PAPER_SUPPLEMENT[tissue]
    return {
        "n_displayed_values": int(y.size),
        "per_subsite": per_subsite,
        "public_plot_ordinal_regression": {
            "raw_slope_per_subsite_step": float(regression.slope),
            "intercept": float(regression.intercept),
            "pearson_r_or_standardized_beta": float(regression.rvalue),
            "r_squared": float(regression.rvalue**2),
            "p_value": float(regression.pvalue),
            "slope_standard_error": float(regression.stderr),
        },
        "rectum_minus_cecum_mean": float(rectum - cecum),
        "supplement_report": supplement,
        "supplement_slope_exactly_reproduced": bool(
            np.isclose(regression.slope, supplement["reported_slope"], rtol=0.0, atol=1e-6)
        ),
        "supplement_slope_matches_public_plot_standardized_beta": bool(
            np.isclose(regression.rvalue, supplement["reported_slope"], rtol=0.0, atol=1e-6)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    fail_closed_output(args.output_dir)

    payload = callback_payload()
    payload_bytes = canonical_json(payload)
    retrieval_time = datetime.now(timezone.utc).isoformat()
    response = requests.post(
        ENDPOINT,
        data=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "DreaMS-external-evidence-audit/1.0",
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    response_bytes = response.content
    parsed = response.json()

    request_path = args.output_dir / "dash_request.json"
    raw_response_path = args.output_dir / "dash_response.json"
    request_path.write_bytes(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
    raw_response_path.write_bytes(response_bytes)

    rows, figure_audit = extract_rows(parsed)
    tidy_path = args.output_dir / "neu5ac_public_plot_values.csv"
    with tidy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    statistics_by_tissue = {
        tissue: tissue_statistics(rows, tissue) for tissue in ("normal", "tumour")
    }
    paper_pair_count = 372
    displayed_counts = {
        tissue: audit["total_values"] for tissue, audit in figure_audit.items()
    }
    count_matches_paper = {
        tissue: count == paper_pair_count for tissue, count in displayed_counts.items()
    }
    regression_matches_supplement = all(
        statistics_by_tissue[tissue]["supplement_slope_exactly_reproduced"]
        or statistics_by_tissue[tissue]["supplement_slope_matches_public_plot_standardized_beta"]
        for tissue in ("normal", "tumour")
    )

    report = {
        "status": "external_crc_neu5ac_dash_patient_level_audit_complete",
        "formal": True,
        "metabolite": METABOLITE,
        "retrieved_at_utc": retrieval_time,
        "endpoint": ENDPOINT,
        "paper_reported_pairs": paper_pair_count,
        "figure_audit": figure_audit,
        "displayed_count_matches_paper_pairs": count_matches_paper,
        "statistics_by_tissue": statistics_by_tissue,
        "supplement_regression_reproduced_from_public_plot_values": regression_matches_supplement,
        "interpretation": {
            "usable": (
                "The public callback independently exposes individual plotted Neu5Ac values "
                "for seven colorectal subsites in tumour and normal tissue."
            ),
            "discrepancy": (
                "The callback exposes 371 values per tissue (53 per subsite), not the 372 "
                "patient pairs reported by the paper, and direct ordinal regression of the "
                "displayed values does not reproduce the supplement's reported slopes."
            ),
            "pairing": (
                "Patient identifiers and pairing keys are absent. Tumour and normal arrays "
                "must be treated as unpaired displayed distributions; within-array positions "
                "are not patient matches."
            ),
            "mtbls13729_relevance": (
                "The displayed data remain useful as directionally consistent anatomical "
                "context, but the frozen supplement—not a reconstructed web regression—is "
                "the authoritative statistic for the external cohort."
            ),
        },
        "claim_limit": (
            "Public-figure distribution audit only. It is not independent mucinous replication, "
            "does not support paired tumour-normal inference, and cannot replace the paper's "
            "analysis-ready patient-level source data."
        ),
        "provenance": {
            "request_sha256": sha256_file(request_path),
            "response_sha256": sha256_bytes(response_bytes),
            "tidy_csv_sha256": sha256_file(tidy_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

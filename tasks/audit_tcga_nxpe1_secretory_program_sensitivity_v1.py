#!/usr/bin/env python
"""Test whether NXPE1 attenuation depends on one secretory-mucin marker."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_tcga_proline_sialic_lineage_sensitivity_v1 import fit_hc3


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/nxpe1_free_donor_v3_secretory"
OUT = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/nxpe1_secretory_sensitivity_v1"
MARKERS = ("MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2")
LINEAGES = (
    "lineage__epithelial", "lineage__myeloid", "lineage__b_plasma",
    "lineage__t_nk", "lineage__endothelial", "lineage__fibroblast",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit(frame: pd.DataFrame, markers: tuple[str, ...]) -> dict:
    work = frame.copy()
    numerical = ["age", *LINEAGES]
    if markers:
        column = "secretory_sensitivity"
        work[column] = work[[f"gene__{gene}" for gene in markers]].mean(axis=1)
        numerical.append(column)
    return fit_hc3(
        work,
        "gene__NXPE1",
        numerical,
        ["side", "stage_group", "gender", "msi"],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    unit_reports: dict[str, dict] = {}
    for unit in ("tpm", "fpkm_uq"):
        source = SOURCE / f"analysis_samples_{unit}_locked.csv"
        frame = pd.read_csv(source)
        if len(frame) != 371 or int(frame["mucinous"].sum()) != 42:
            raise RuntimeError(f"{unit}: locked cohort changed")
        baseline = fit(frame, ())
        configurations: list[tuple[str, tuple[str, ...]]] = [("none", ())]
        configurations.append(("all_five", MARKERS))
        configurations.extend(
            (f"leave_out_{gene}", tuple(marker for marker in MARKERS if marker != gene))
            for gene in MARKERS
        )
        configurations.extend((f"single_{gene}", (gene,)) for gene in MARKERS)
        configurations.extend(
            ("pair_" + "_".join(pair), pair)
            for pair in itertools.combinations(MARKERS, 2)
        )
        for name, markers in configurations:
            result = fit(frame, markers)
            attenuation = (
                1.0 - result["beta"] / baseline["beta"]
                if baseline["beta"] != 0 else np.nan
            )
            rows.append({
                "unit": unit,
                "configuration": name,
                "markers": ";".join(markers),
                "n_markers": len(markers),
                "n": result["n"],
                "beta": result["beta"],
                "p": result["p"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "attenuation_fraction_vs_no_secretory": attenuation,
            })

        unit_rows = [row for row in rows if row["unit"] == unit]
        leave_one_out = [row for row in unit_rows if row["configuration"].startswith("leave_out_")]
        singles = [row for row in unit_rows if row["configuration"].startswith("single_")]
        pairs = [row for row in unit_rows if row["configuration"].startswith("pair_")]
        unit_reports[unit] = {
            "baseline": next(row for row in unit_rows if row["configuration"] == "none"),
            "all_five": next(row for row in unit_rows if row["configuration"] == "all_five"),
            "leave_one_out": {
                "all_p_gt_0_05": all(row["p"] > 0.05 for row in leave_one_out),
                "beta_range": [min(row["beta"] for row in leave_one_out), max(row["beta"] for row in leave_one_out)],
                "attenuation_range": [min(row["attenuation_fraction_vs_no_secretory"] for row in leave_one_out), max(row["attenuation_fraction_vs_no_secretory"] for row in leave_one_out)],
            },
            "single_marker": {
                "significant_after_adjustment": [row["configuration"] for row in singles if row["p"] < 0.05],
                "beta_range": [min(row["beta"] for row in singles), max(row["beta"] for row in singles)],
            },
            "pair_marker": {
                "significant_after_adjustment": [row["configuration"] for row in pairs if row["p"] < 0.05],
                "beta_range": [min(row["beta"] for row in pairs), max(row["beta"] for row in pairs)],
            },
        }

    result_table = pd.DataFrame(rows)
    result_table.to_csv(OUT / "nxpe1_secretory_sensitivity.csv", index=False)
    report = {
        "status": "tcga_nxpe1_secretory_program_sensitivity_complete",
        "formal": False,
        "locked_tumours": 371,
        "mucinous": 42,
        "conventional": 329,
        "secretory_markers": list(MARKERS),
        "models": "clinical + broad-lineage + MSI; secretory configurations are sensitivity covariates",
        "units": unit_reports,
        "interpretation_rule": (
            "If all leave-one-out models strongly attenuate NXPE1, the result supports a distributed "
            "secretory-carrier state rather than dependence on one marker. Single/pair results are "
            "descriptive and are not used for post-hoc model selection."
        ),
        "claim_limit": (
            "Covariate attenuation is not causal mediation. Secretory markers may be lineage proxies, "
            "mediators or correlated outputs; the audit tests robustness of the descriptive carrier-state interpretation."
        ),
        "provenance": {
            unit: sha256(SOURCE / f"analysis_samples_{unit}_locked.csv")
            for unit in ("tpm", "fpkm_uq")
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

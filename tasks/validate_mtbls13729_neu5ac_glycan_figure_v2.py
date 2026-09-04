#!/usr/bin/env python
"""Fail-closed validation of the hybrid mucin glycome publication figure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/neu5ac_glycan_publication_figure_v2_final"


def main() -> None:
    required = [
        "neu5ac_hybrid_glycome_figure_v2.png",
        "neu5ac_hybrid_glycome_figure_v2.pdf",
        "neu5ac_targeted_eic_patient_deltas.csv",
        "tcga_glycan_branch_effects.csv",
        "external_mucinous_oglycan_structures.csv",
        "report.json",
        "README.md",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing figure-v2 outputs: {missing}")
    if (OUT / required[0]).stat().st_size < 100_000 or (OUT / required[1]).stat().st_size < 10_000:
        raise RuntimeError("rendered figure artifacts are unexpectedly small")

    patient = pd.read_csv(OUT / required[2])
    rmu = patient.loc[patient.cohort.eq("Rmu"), "log2_tumour_minus_normal"].dropna()
    if len(rmu) != 10 or int(rmu.gt(0).sum()) != 10:
        raise RuntimeError("locked Rmu Neu5Ac direction no longer reproduces 10/10")
    if not np.isclose(rmu.mean(), 1.9350668127509976, atol=1e-12):
        raise RuntimeError(f"locked Rmu Neu5Ac mean drifted: {rmu.mean()}")

    tcga = pd.read_csv(OUT / required[3])
    expected = {
        "neu5ac_donor_supply_transport",
        "secretory_mucin_program",
        "normal_mucosal_core3_sda",
        "core2_slex_biosynthesis",
        "alpha23_o_glycan_sialylation",
        "ST6GAL1",
        "ST6GALNAC1",
        "GCNT3",
    }
    if set(tcga.outcome) != expected or len(tcga) != len(expected):
        raise RuntimeError("TCGA branch panel is incomplete or duplicated")
    q = tcga.set_index("outcome").bh_q
    if not (q["neu5ac_donor_supply_transport"] < 1e-6 and q["secretory_mucin_program"] < 1e-8):
        raise RuntimeError("positive donor/carrier anchors did not reproduce")
    if not (q["core2_slex_biosynthesis"] > 0.5 and q["ST6GAL1"] < 1e-3):
        raise RuntimeError("key branch-decoupling controls did not reproduce")

    ogly = pd.read_csv(OUT / required[4]).set_index("feature")
    if len(ogly) != 5:
        raise RuntimeError("external O-glycomics panel must contain five frozen structures")
    if not (ogly.loc["core_2", ["T2_delta", "T3_delta"]].astype(float) > 0).all():
        raise RuntimeError("external core-2 paired direction drifted")
    if not (ogly.loc["alpha2_6_sialylation", ["T2_delta", "T3_delta"]].astype(float) < 0).all():
        raise RuntimeError("external alpha2-6 paired direction drifted")
    if not (ogly.loc["core_3", ["T2_delta", "T3_delta"]].astype(float) < 0).all():
        raise RuntimeError("external core-3 paired direction drifted")

    report = json.loads((OUT / required[5]).read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_neu5ac_hybrid_glycome_figure_v2_complete":
        raise RuntimeError("unexpected figure-v2 report status")
    boundary = report.get("claim_limit", "")
    for phrase in ("not flux", "enzyme causality", "independent abundance replication"):
        if phrase not in boundary:
            raise RuntimeError(f"claim boundary lost required phrase: {phrase}")

    print(json.dumps({
        "status": "mtbls13729_neu5ac_hybrid_glycome_figure_v2_validation_passed",
        "rmu_n": int(len(rmu)),
        "rmu_positive": int(rmu.gt(0).sum()),
        "rmu_mean_log2fc": float(rmu.mean()),
        "tcga_branches": int(len(tcga)),
        "external_structures": int(len(ogly)),
    }, indent=2))


if __name__ == "__main__":
    main()

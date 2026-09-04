#!/usr/bin/env python
"""Freeze the four UHPLC-HRMS/MS sphingolipids used in the source-paper story.

Coordinates come from the authors' Table S4.  This script only creates a
targeted-EIC execution panel; it does not accept the published identities or
significance calls as truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
SAMPLES = ROOT / "data/mtbls13729/ms1_consensus"
OUT = ROOT / "data/mtbls13729/author_sphingolipid_targets_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    # Observed precursor m/z and RT are copied verbatim from Table S4.
    # RT is converted from minutes to seconds for raw-mzML extraction.
    records = [
        {
            "panel": "pos_rp", "feature_id": 9902782, "mz": 568.56486,
            "rt_sec": 12.168 * 60.0, "published_name": "Cer(d18:0/18:0)",
            "published_hmdb": "HMDB0011761", "published_msi_level": "Level 1",
        },
        {
            "panel": "neg_rp", "feature_id": 9900264, "mz": 314.27007,
            "rt_sec": 9.029 * 60.0, "published_name": "Dehydrophytosphingosine",
            "published_hmdb": "HMDB0038057", "published_msi_level": "Level 2",
        },
        {
            "panel": "pos_rp", "feature_id": 9900055, "mz": 302.30511,
            "rt_sec": 10.592 * 60.0, "published_name": "Sphinganine",
            "published_hmdb": "HMDB0000252", "published_msi_level": "Level 1",
        },
        {
            "panel": "pos_hilic", "feature_id": 9900175, "mz": 300.28586,
            "rt_sec": 0.884 * 60.0, "published_name": "Sphingosine",
            "published_hmdb": "HMDB0062807", "published_msi_level": "Level 1",
        },
    ]
    frame = pd.DataFrame(records)
    for panel in ("neg_rp", "pos_rp", "pos_hilic"):
        target = frame[frame.panel.eq(panel)].drop(columns="panel").copy()
        target.to_csv(OUT / f"{panel}__requantification_targets.csv.gz", index=False)
        sample_path = SAMPLES / f"{panel}__samples.csv"
        if sample_path.exists():
            samples = pd.read_csv(sample_path)
        else:
            # HILIC was not part of the discovery-wide MS1 consensus stage.
            # Build the execution manifest from deposited raw files only.
            names = sorted(path.stem for path in (ROOT / f"data/mtbls13729/mzml/{panel}").glob("*.mzML"))
            samples = pd.DataFrame({"sample_name": names})
            samples.insert(0, "map_index", range(len(samples)))
            samples["patient"] = samples.sample_name.str.extract(r"^(P\d{2})")
            samples["tissue"] = np.where(samples.sample_name.str.endswith(("LN", "RN")), "normal", "tumor")
            samples["histology"] = np.select(
                [samples.sample_name.str.endswith("Rmu"), samples.sample_name.str.endswith(("Ltu", "Rtu"))],
                ["mucinous", "tubular"], default="normal",
            )
        samples.to_csv(OUT / f"{panel}__samples.csv", index=False)
    frame.to_csv(OUT / "author_sphingolipid_panel.csv", index=False)
    report = {
        "status": "mtbls13729_author_sphingolipid_panel_frozen",
        "formal": False,
        "targets": int(len(frame)),
        "panels": frame.groupby("panel").size().astype(int).to_dict(),
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "coordinate_contract": "verbatim Table S4 precursor m/z and RT; RT converted min to sec",
        "claim_limit": "Published names and MSI levels are hypotheses under raw-data re-extraction, not target truth.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

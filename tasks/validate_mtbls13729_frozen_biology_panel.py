#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, default=Path("data/mtbls13729/frozen_biology_panel_v1"))
    args = parser.parse_args()
    report = json.loads((args.panel_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "mtbls13729_frozen_biology_panel_complete"
    assert report["formal"] and report["selection_is_phenotype_blind"]
    for panel in ("neg_rp", "pos_rp"):
        frame = pd.read_csv(args.panel_dir / f"{panel}__requantification_targets.csv.gz")
        assert len(frame) == report["panels"][panel]["selected_targets"]
        assert not frame.feature_id.duplicated().any()
        assert frame[["mz", "rt_sec"]].notna().all().all()
        assert frame.selected_for_targeted_requantification.all()
        assert not any("qvalue" in column.lower() or "log2fc" in column.lower() for column in frame.columns)
    pos = pd.read_csv(args.panel_dir / "pos_rp__requantification_targets.csv.gz")
    assert int(pos.predeclared_c20_4_anchor.sum()) == 1
    print("[validate_mtbls13729_frozen_biology_panel] PASS", flush=True)


if __name__ == "__main__":
    main()

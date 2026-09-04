"""Fail-closed validator for the formal E2 corrective scan and decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_corrective_scan"))
    args = parser.parse_args()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(output / "cell_decisions.csv")
    interventions = pd.read_csv(output / "paired_corrective_interventions.csv.gz")
    if report.get("status") != "noise_final_e2_corrective_scan_complete" or not report.get("formal"):
        raise RuntimeError("formal E2 scan report failed")
    if decision.get("status") != "noise_final_e2_corrective_decision_complete" or not decision.get("formal"):
        raise RuntimeError("formal E2 decision failed")
    if len(cells) != 32 or cells["cell_id"].nunique() != 32:
        raise RuntimeError("E2 decision must report all 32 frozen M1 cells")
    if cells["arm"].eq("corrective").sum() != 28 or cells["arm"].eq("negative_control").sum() != 4:
        raise RuntimeError("E2 arm counts changed")
    if cells.loc[cells["arm"].eq("negative_control"), "pass_to_e3"].any():
        raise RuntimeError("negative control entered E3")
    if report["clean_forward_reproduction"]["preservation_p01"] < 0.999:
        raise RuntimeError("clean official-forward preservation failed")
    if report["clean_forward_reproduction"]["rank_mismatch_fraction"] > 0.001:
        raise RuntimeError("clean official-forward rank reproduction failed")
    corrective = interventions["arm"].eq("corrective")
    if not interventions.loc[corrective, "control_count"].eq(3).all():
        raise RuntimeError("corrective intervention lacks three matched controls")
    print(f"[validate_noise_final_e2_corrective_scan] PASS; E3 cells={decision['cells_passing_to_e3']}")


if __name__ == "__main__":
    main()

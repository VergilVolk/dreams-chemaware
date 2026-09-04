"""Fail-closed validation for E2-M1b sensitivity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_sensitivity"))
    args = parser.parse_args()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    summary = json.loads((output / "sensitivity.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(output / "cell_sensitivity.csv")
    candidates = pd.read_csv(output / "e3_candidate_cells.csv")
    families = pd.read_csv(output / "family_consensus.csv")
    if summary.get("status") != "noise_final_e2_sensitivity_complete" or not summary.get("formal"):
        raise RuntimeError("formal E2 sensitivity summary failed")
    if len(cells) != 32 or cells["cell_id"].nunique() != 32:
        raise RuntimeError("sensitivity audit lost frozen cells")
    if cells["arm"].eq("corrective").sum() != 28 or cells["arm"].eq("negative_control").sum() != 4:
        raise RuntimeError("sensitivity arm counts changed")
    if cells.loc[cells["arm"].eq("negative_control"), "pass_to_e3_after_sensitivity"].any():
        raise RuntimeError("negative control passed sensitivity")
    if set(candidates["cell_id"].astype(str)) != set(summary["sensitivity_passing_cell_ids"]):
        raise RuntimeError("E3 candidate list disagrees with sensitivity summary")
    if families.empty:
        raise RuntimeError("family consensus report is empty")
    print(
        f"[validate_noise_final_e2_sensitivity] PASS; "
        f"E3 candidates={len(candidates)} families={len(families)}"
    )


if __name__ == "__main__":
    main()

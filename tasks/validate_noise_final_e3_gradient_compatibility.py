"""Fail-closed validation for the formal E3 gradient audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e3_gradient_compatibility"))
    args = parser.parse_args()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(output / "cell_gradient_summary.csv")
    cell_pairs = pd.read_csv(output / "cell_gradient_compatibility.csv")
    family_pairs = pd.read_csv(output / "family_gradient_compatibility.csv")
    if report.get("status") != "noise_final_e3_gradient_compatibility_complete" or not report.get("formal"):
        raise RuntimeError("formal E3 report failed")
    n_cells = int(report["candidate_cells"])
    n_families = int(report["mechanism_families"])
    if len(cells) != n_cells or len(cell_pairs) != n_cells * n_cells:
        raise RuntimeError("E3 cell matrix is incomplete")
    if len(family_pairs) != n_families * n_families:
        raise RuntimeError("E3 family matrix is incomplete")
    if n_families < 2 or report["identities"] < 100 or report["formulas"] < 100:
        raise RuntimeError("E3 chemical coverage is insufficient")
    if not report["contracts"]["shared_embedding_target"] or report["contracts"]["P2b"] != "forbidden":
        raise RuntimeError("E3 scope contract failed")
    print(f"[validate_noise_final_e3_gradient_compatibility] PASS; cells={n_cells} families={n_families}")


if __name__ == "__main__":
    main()

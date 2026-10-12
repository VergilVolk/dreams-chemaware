#!/usr/bin/env python
"""Audit the experimental-unit structure of ST003325 before effect modelling."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PANELS = ("AN005444", "AN005445")
STUDY_RE = re.compile(
    r"^(?P<clone>(?P<edit>CE|KO|PTC|WT)_[A-Z0-9]+)_plate_(?P<plate>[1-6])_"
    r"(?P<panel>HILICneg|RPpos)_"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_group(text: str) -> tuple[str, str]:
    group = text.split("GROUP:", 1)[1]
    edit, condition = group.split(",", 1)
    return edit, condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    factor_payload = json.loads(args.factors.read_text(encoding="utf-8"))
    factors = {row["raw_data"].removesuffix(".mzML"): row for row in factor_payload.values()}
    reports = []
    all_design_rows = []

    for analysis in PANELS:
        table_path = args.tables / f"ST003325_{analysis}_Results.txt"
        columns = pd.read_csv(table_path, sep="\t", nrows=0).columns.tolist()[1:]
        parsed_rows = []
        for column in columns:
            match = STUDY_RE.match(column)
            if match is None:
                continue
            if column not in factors:
                raise KeyError(f"No factor row for {analysis}:{column}")
            edit_factor, condition = parse_group(factors[column]["factors"])
            edit = match.group("edit")
            expected_edit = "PT" if edit == "PTC" else edit
            if edit_factor != expected_edit:
                raise ValueError(f"Edit mismatch for {column}: {edit_factor} != {expected_edit}")
            parsed_rows.append({
                "analysis_id": analysis,
                "sample_id": column,
                "edit_class": edit,
                "clone": match.group("clone"),
                "plate": int(match.group("plate")),
                "condition": condition,
            })
        frame = pd.DataFrame(parsed_rows)
        if len(frame) != 72:
            raise ValueError(f"{analysis}: expected 72 study injections, found {len(frame)}")
        frame.to_csv(args.output / f"{analysis}_design.csv", index=False)
        all_design_rows.extend(parsed_rows)

        clone_counts = Counter(frame["clone"])
        cells = Counter(zip(frame["edit_class"], frame["condition"]))
        plate_condition = Counter(zip(frame["plate"], frame["condition"]))
        basal_plates = sorted(frame.loc[frame["condition"] == "Basal", "plate"].unique().tolist())
        dheas_plates = sorted(frame.loc[frame["condition"] == "Basal+DHEAS", "plate"].unique().tolist())
        reports.append({
            "analysis_id": analysis,
            "study_injections": int(len(frame)),
            "edit_classes": sorted(frame["edit_class"].unique().tolist()),
            "clones": sorted(frame["clone"].unique().tolist()),
            "unique_clones": int(frame["clone"].nunique()),
            "injections_per_clone": dict(sorted(clone_counts.items())),
            "edit_condition_cells": {f"{k[0]}|{k[1]}": v for k, v in sorted(cells.items())},
            "plate_condition_cells": {f"plate{k[0]}|{k[1]}": v for k, v in sorted(plate_condition.items())},
            "basal_plates": basal_plates,
            "dheas_plates": dheas_plates,
            "condition_fully_confounded_with_plate_number": not set(basal_plates).intersection(dheas_plates),
            "table_sha256": sha256(table_path),
        })

    design = pd.DataFrame(all_design_rows)
    report = {
        "status": "ST003325_PERTURBATION_DESIGN_AUDITED",
        "formal": True,
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed",
        "sealed": False,
        "phenotype_values_used": True,
        "effect_values_used": False,
        "panels": reports,
        "combined": {
            "study_injections": int(len(design)),
            "unique_physical_sample_keys_across_panels": int(
                design[["clone", "plate"]].drop_duplicates().shape[0]
            ),
            "unique_clones": int(design["clone"].nunique()),
            "edit_classes": int(design["edit_class"].nunique()),
            "panels_measure_the_same_72_physical_conditions": True,
        },
        "hard_boundary": (
            "Plate number 1-3 encodes Basal and plate number 4-6 encodes Basal+DHEAS. "
            "DHEA-S cannot be separated from a generic plate-number effect by injection-level inference. "
            "The 72 injections per panel are not 72 independent biological replicates; clone and edit "
            "class are the admissible replication axes."
        ),
        "admissible_next_test": (
            "A structure-hidden development test may ask whether a response direction replicates across "
            "independently edited clones and across CE/KO/PTC edit classes, using clone-blocked or "
            "edit-class-held-out inference. It may not claim an unconfounded causal DHEA-S effect."
        ),
        "provenance": {"factors_sha256": sha256(args.factors)},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

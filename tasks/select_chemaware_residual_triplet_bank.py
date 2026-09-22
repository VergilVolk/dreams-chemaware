"""Freeze a residual-triplet hardness window without fitting model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank", action="append", required=True,
        help="Named bank as NAME=DIRECTORY; repeat for the frozen window grid.",
    )
    parser.add_argument("--min-relative-violation", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_bank(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--bank must be NAME=DIRECTORY")
    name, path = raw.split("=", 1)
    return name, Path(path)


def main() -> None:
    args = arguments()
    rows = []
    for raw in args.bank:
        name, directory = parse_bank(raw)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("status") != "CHEMAWARE_RESIDUAL_NATIVE_TRIPLETS_COMPLETE":
            raise RuntimeError(f"residual triplet bank is incomplete: {directory}")
        role2 = report["audits"]["selection:correct"]
        overlap = report["role2_correct_vs_null_pair_overlap"]
        secondary_overlap = report.get(
            "role2_correct_vs_null_secondary_slot_overlap", overlap,
        )
        rows.append({
            "name": name,
            "directory": str(directory.resolve()),
            "hardness_window": float(report["chemical_hardness_window"]),
            "margin_violating_fraction": float(role2["margin_violating_fraction"]),
            "mean_positive_minus_negative": float(role2["mean_current_positive_minus_negative"]),
            "chemical_hard_events": int(role2["chemical_hard_events"]),
            "strict_specific_events": int(role2["strict_specific_events"]),
            "maximum_correct_null_pair_jaccard": max(
                float(value["jaccard"]) for value in overlap.values()
            ),
            "maximum_correct_null_secondary_slot_jaccard": max(
                float(value["jaccard"]) for value in secondary_overlap.values()
            ),
        })
    maximum_violation = max(row["margin_violating_fraction"] for row in rows)
    for row in rows:
        row["relative_violation_retention"] = (
            row["margin_violating_fraction"] / maximum_violation
            if maximum_violation > 0.0 else 1.0
        )
        row["admissible"] = row["relative_violation_retention"] >= args.min_relative_violation
    admissible = [row for row in rows if row["admissible"]]
    if not admissible:
        raise RuntimeError("no residual triplet bank retained the required active-boundary fraction")
    selected = min(admissible, key=lambda row: (
        float(row["maximum_correct_null_secondary_slot_jaccard"]),
        -int(row["strict_specific_events"]),
        -int(row["chemical_hard_events"]),
        float(row["mean_positive_minus_negative"]),
    ))
    output = {
        "status": "CHEMAWARE_RESIDUAL_TRIPLET_BANK_SELECTED",
        "selection_scope": "formula role 2 only; no model weights fitted",
        "selection_rule": (
            "retain at least the configured fraction of the most active bank, "
            "then minimize correct-vs-null secondary-slot pair overlap"
        ),
        "min_relative_violation": float(args.min_relative_violation),
        "selected": selected,
        "banks": rows,
        "outer_role_4_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output.with_suffix(".path.txt").write_text(
        str(selected["directory"]) + "\n", encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

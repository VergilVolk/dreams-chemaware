"""Split the mass-dense factor cohort without molecule or neighbor leakage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/validation/mass_dense_factor_cohort_audit.json"
DEFAULT_OUTPUT = ROOT / "data/validation/mass_dense_factor_cohort_split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


class DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    units = {int(unit["unit_id"]): unit for unit in report["units"]}
    dsu = DisjointSet(units)
    by_molecule = {}
    for unit_id, unit in units.items():
        by_molecule.setdefault(unit["ik14"], []).append(unit_id)
        for neighbor in unit["negative_unit_ids"]:
            if neighbor in units:
                dsu.union(unit_id, neighbor)
    for molecule_units in by_molecule.values():
        for unit_id in molecule_units[1:]:
            dsu.union(molecule_units[0], unit_id)

    components = {}
    for unit_id in units:
        components.setdefault(dsu.find(unit_id), []).append(unit_id)
    component_list = sorted(
        components.values(), key=lambda values: (-len(values), min(values))
    )

    assignments = {}
    counts = {"discovery": Counter(), "confirmation": Counter()}
    totals = Counter(unit["adduct"] for unit in units.values())
    for component in component_list:
        component_counts = Counter(units[idx]["adduct"] for idx in component)
        penalties = {}
        for candidate_split in ("discovery", "confirmation"):
            projected = {
                split: counts[split].copy()
                for split in ("discovery", "confirmation")
            }
            projected[candidate_split].update(component_counts)
            total_penalty = sum(
                (sum(projected[split].values()) - len(units) / 2) ** 2
                / max(len(units) / 2, 1)
                for split in projected
            )
            adduct_penalty = sum(
                (projected[split][adduct] - total / 2) ** 2
                / max(total / 2, 1)
                for split in projected
                for adduct, total in totals.items()
            )
            penalties[candidate_split] = total_penalty + adduct_penalty
        split = min(
            penalties,
            key=lambda name: (penalties[name], sum(counts[name].values()), name),
        )
        for unit_id in component:
            assignments[unit_id] = split
        counts[split].update(component_counts)

    molecule_splits = {}
    cross_split_links = 0
    for unit_id, unit in units.items():
        molecule_splits.setdefault(unit["ik14"], set()).add(assignments[unit_id])
        for neighbor in unit["negative_unit_ids"]:
            if neighbor in assignments and assignments[neighbor] != assignments[unit_id]:
                cross_split_links += 1
    molecule_overlap = sum(len(splits) > 1 for splits in molecule_splits.values())
    output_units = []
    for unit_id, unit in units.items():
        item = dict(unit)
        item["split"] = assignments[unit_id]
        item["negative_unit_ids"] = [
            neighbor for neighbor in item["negative_unit_ids"]
            if neighbor in assignments and assignments[neighbor] == assignments[unit_id]
        ]
        output_units.append(item)
    result = {
        "status": "mass_dense_factor_cohort_split",
        "source": str(args.input),
        "split_method": (
            "Connected components combine all 10 ppm neighbor links and all units of "
            "the same IK14; components are greedily balanced by size and adduct."
        ),
        "n_components": len(component_list),
        "largest_component": max(map(len, component_list)),
        "counts": {
            split: {"total": sum(counter.values()), "by_adduct": dict(counter)}
            for split, counter in counts.items()
        },
        "molecule_overlap": molecule_overlap,
        "cross_split_negative_links": cross_split_links,
        "units": output_units,
    }
    if molecule_overlap or cross_split_links:
        raise RuntimeError(
            f"Leakage detected: molecule_overlap={molecule_overlap}, "
            f"cross_split_links={cross_split_links}"
        )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))
    print(
        f"components={result['n_components']}; largest={result['largest_component']}; "
        f"molecule_overlap={molecule_overlap}; cross_links={cross_split_links}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Select ring-balanced query anchors while preserving all 10-ppm candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=Path("data/validation/external_ring_stratified_cohort"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--queries-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cohort / "manifest.json").read_text(encoding="utf-8"))
    source = np.load(args.cohort / "spectra.npz")
    source_ids = source["unit_id"].astype(int)
    source_position = {unit_id: i for i, unit_id in enumerate(source_ids)}
    units = {int(unit["unit_id"]): unit for unit in manifest["units"]}
    rng = np.random.default_rng(args.seed)
    report = {"status": "external_ring_balanced_query_pilot", "splits": {}}
    for split in ("discovery", "confirmation"):
        anchors = []
        for ring_class in ("acyclic", "single_ring", "multi_ring"):
            candidates = [unit for unit in units.values()
                          if unit["split"] == split and unit["ring_class"] == ring_class]
            tie = {int(unit["unit_id"]): float(rng.random()) for unit in candidates}
            candidates.sort(key=lambda unit: (
                not bool(unit["same_formula_negative_ids"]),
                -len(unit["same_formula_negative_ids"]),
                -len(unit["negative_unit_ids"]),
                tie[int(unit["unit_id"])],
            ))
            chosen = candidates[:args.queries_per_class]
            if len(chosen) < args.queries_per_class:
                raise RuntimeError(f"Insufficient {split}/{ring_class}: {len(chosen)}")
            anchors.extend(int(unit["unit_id"]) for unit in chosen)
        included = set(anchors)
        for anchor in anchors:
            included.update(int(value) for value in units[anchor]["negative_unit_ids"])
        included = sorted(included)
        remap = {old: new for new, old in enumerate(included)}
        spectra = np.stack([source["spectra"][source_position[old]] for old in included])
        precursor = np.stack([source["precursor_mz"][source_position[old]] for old in included])
        np.savez_compressed(
            args.output_dir / f"{split}_spectra.npz",
            spectra=spectra, precursor_mz=precursor,
            original_unit_id=np.asarray(included, dtype=np.int64),
        )
        output_units = []
        for old in included:
            unit = dict(units[old])
            unit.update({
                "pair_id": remap[old],
                "is_query_anchor": old in anchors,
                "negative_pair_ids": [remap[value] for value in unit["negative_unit_ids"] if value in remap],
                "same_formula_negative_pair_ids": [remap[value] for value in unit["same_formula_negative_ids"] if value in remap],
            })
            output_units.append(unit)
        (args.output_dir / f"{split}_manifest.json").write_text(
            json.dumps({"split": split, "units": output_units}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        anchor_units = [units[value] for value in anchors]
        report["splits"][split] = {
            "query_anchors": len(anchors),
            "query_by_ring": dict(Counter(unit["ring_class"] for unit in anchor_units)),
            "anchors_with_same_formula_negative": int(sum(bool(unit["same_formula_negative_ids"]) for unit in anchor_units)),
            "candidate_units_including_anchors": len(included),
            "query_candidate_links": int(sum(len(unit["negative_unit_ids"]) for unit in anchor_units)),
            "same_formula_query_candidate_links": int(sum(len(unit["same_formula_negative_ids"]) for unit in anchor_units)),
        }
    discovery = {unit["ik14"] for unit in json.loads((args.output_dir / "discovery_manifest.json").read_text(encoding="utf-8"))["units"]}
    confirmation = {unit["ik14"] for unit in json.loads((args.output_dir / "confirmation_manifest.json").read_text(encoding="utf-8"))["units"]}
    report["all_included_molecule_overlap"] = len(discovery & confirmation)
    if discovery & confirmation:
        raise RuntimeError("Discovery/confirmation molecule leakage")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

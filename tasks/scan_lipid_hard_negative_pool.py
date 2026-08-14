"""Raw-MGF quality gate for the index-level lipid hard-negative pool."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_external_ring_stratified_cohort_v2 import coarse_cosine, stream_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", type=Path, default=Path("data/annotated01.mgf"))
    parser.add_argument("--index-candidates", type=Path, default=Path("data/validation/lipid_hard_negative_pool_gate/index_level_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lipid_hard_negative_pool_gate"))
    parser.add_argument("--adduct-ppm", type=float, default=20.0)
    parser.add_argument("--max-spectra", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.index_candidates.open(encoding="utf-8") as stream:
        indexed = list(csv.DictReader(stream))
    targets = {
        row["ik14"]: {
            **row,
            "expected_precursor_mz": float(row["expected_precursor_mz"]),
        }
        for row in indexed
    }
    pools, scan_audit = stream_targets(args.mgf, targets, args.max_spectra, args.adduct_ppm)

    viable = {}
    rejection = Counter()
    for ik14, target in targets.items():
        spectra = pools.get(ik14, [])
        if len(spectra) < 2:
            rejection["fewer_than_two_quality_spectra"] += 1
            continue
        pairs = [
            (coarse_cosine(spectra[i]["peaks"], spectra[j]["peaks"]), i, j)
            for i in range(len(spectra) - 1)
            for j in range(i + 1, len(spectra))
        ]
        nonduplicate = [value for value in pairs if 0.05 <= value[0] < 0.995]
        if not nonduplicate:
            rejection["no_nonduplicate_positive_pair"] += 1
            continue
        similarity, left, right = min(nonduplicate, key=lambda value: abs(value[0] - 0.5))
        viable[ik14] = {
            **target,
            "retained_quality_spectra": len(spectra),
            "selected_positive_proxy_similarity": float(similarity),
            "spectra": [spectra[left], spectra[right]],
        }

    by_formula = defaultdict(list)
    for value in viable.values():
        by_formula[value["formula"]].append(value)
    dense = {formula: values for formula, values in by_formula.items() if len(values) >= 2}
    kept = [value for values in dense.values() for value in values]
    kept.sort(key=lambda value: (value["formula"], value["ik14"]))

    spectra_array = np.zeros((len(kept), 2, 2, 128), dtype=np.float32)
    precursor_array = np.zeros((len(kept), 2), dtype=np.float32)
    records = []
    id_by_ik = {value["ik14"]: i for i, value in enumerate(kept)}
    for i, value in enumerate(kept):
        for view, spectrum in enumerate(value["spectra"]):
            peaks = spectrum["peaks"][:128]
            spectra_array[i, view, :, : len(peaks)] = peaks.T
            precursor_array[i, view] = spectrum["precursor_mz"]
        alternatives = [other for other in dense[value["formula"]] if other["ik14"] != value["ik14"]]
        records.append({
            key: item for key, item in value.items() if key != "spectra"
        } | {
            "unit_id": i,
            "same_formula_negative_unit_ids": [id_by_ik[other["ik14"]] for other in alternatives],
            "same_formula_negative_ik14": [other["ik14"] for other in alternatives],
        })
    np.savez_compressed(
        args.output_dir / "quality_screened_spectra.npz",
        spectra=spectra_array,
        precursor_mz=precursor_array,
        unit_id=np.arange(len(kept), dtype=np.int64),
    )
    (args.output_dir / "quality_screened_manifest.json").write_text(
        json.dumps({"units": records}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = len(kept)
    directed = sum(len(values) * (len(values) - 1) for values in dense.values())
    largest = max((len(values) for values in dense.values()), default=0)
    gates = {
        "at_least_100_molecules": total >= 100,
        "at_least_30_formula_groups": len(dense) >= 30,
        "largest_formula_at_most_15_percent": (largest / total <= 0.15) if total else False,
        "at_least_500_directed_identity_negative_choices": directed >= 500,
        "at_least_two_nonduplicate_quality_spectra_each": True,
    }
    report = {
        "status": "raw_quality_lipid_hard_negative_pool_gate",
        "index_candidates": len(indexed),
        "scan_audit": scan_audit,
        "rejection": dict(rejection),
        "viable_before_formula_filter": len(viable),
        "eligible_same_formula_molecules": total,
        "independent_formula_groups": len(dense),
        "largest_formula_group": largest,
        "directed_identity_negative_choices": directed,
        "formula_group_size_distribution": dict(Counter(len(values) for values in dense.values())),
        "positive_proxy_similarity": {
            "median": float(np.median([v["selected_positive_proxy_similarity"] for v in kept])) if kept else None,
            "min": float(np.min([v["selected_positive_proxy_similarity"] for v in kept])) if kept else None,
            "max": float(np.max([v["selected_positive_proxy_similarity"] for v in kept])) if kept else None,
        },
        "gates": gates,
        "gate_pass": all(gates.values()),
        "decision_if_failed": "Do not fine-tune. Expand an independent lipid corpus or relax the chemical domain only after a new failure audit supports it.",
    }
    (args.output_dir / "raw_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

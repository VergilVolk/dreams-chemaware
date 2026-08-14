"""Efficient external ring-stratified annotated01 cohort builder.

All chemistry and 10-ppm graph operations are performed once per molecule.
The 2.5-GB MGF is then streamed once, while peak parsing is restricted to a
preselected target set.  MassSpecGym molecules are excluded globally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


PROTON = 1.007276466621
SODIUM = 22.989218


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", type=Path, default=Path("data/annotated01.mgf"))
    parser.add_argument("--indices", type=Path, default=Path("tasks/_cache/indices.json"))
    parser.add_argument("--massspecgym-metadata", type=Path, default=Path("data/massspecgym/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_ring_stratified_cohort"))
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--adduct-ppm", type=float, default=20.0)
    parser.add_argument("--preselect-per-class-per-split", type=int, default=220)
    parser.add_argument("--max-spectra-per-unit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def decode_exclusions(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as stream:
        return {row["inchikey14"][:14] for row in csv.DictReader(stream)}


def molecule_table(index: dict, excluded: set[str], adduct_ppm: float) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for key, smiles in index["ik_to_smi"].items():
        ik14 = key[:14]
        if ik14 in result or ik14 in excluded:
            continue
        if int(index["ik_counts"].get(key, 0)) < 2 or key not in index["ik_to_pm"]:
            continue
        formula = index["ik_to_fm"].get(key)
        mol = Chem.MolFromSmiles(smiles)
        if not formula or mol is None:
            continue
        neutral = Descriptors.ExactMolWt(mol)
        first_mass = float(index["ik_to_pm"][key])
        candidates = {"[M+H]+": neutral + PROTON, "[M+Na]+": neutral + SODIUM}
        adduct, expected = min(candidates.items(), key=lambda item: abs(item[1] - first_mass))
        if abs(expected - first_mass) / expected * 1e6 > adduct_ppm:
            continue
        rings = rdMolDescriptors.CalcNumRings(mol)
        ring = "acyclic" if rings == 0 else "single_ring" if rings == 1 else "multi_ring"
        result[ik14] = {
            "ik14": ik14, "smiles": smiles, "formula": formula,
            "ring_class": ring, "adduct": adduct,
            "expected_precursor_mz": expected,
        }
    return result


def dense_components(molecules: dict[str, dict], ppm: float) -> list[list[str]]:
    components: list[list[str]] = []
    by_adduct: dict[str, list[str]] = defaultdict(list)
    for ik14, item in molecules.items():
        by_adduct[item["adduct"]].append(ik14)
    for identifiers in by_adduct.values():
        identifiers.sort(key=lambda ik: molecules[ik]["expected_precursor_mz"])
        current: list[str] = []
        maximum_reach = -math.inf
        for ik14 in identifiers:
            mass = molecules[ik14]["expected_precursor_mz"]
            low, high = mass * (1 - ppm * 1e-6), mass * (1 + ppm * 1e-6)
            if current and low > maximum_reach:
                if len(current) > 1:
                    components.append(current)
                current = []
                maximum_reach = -math.inf
            current.append(ik14)
            maximum_reach = max(maximum_reach, high)
        if len(current) > 1:
            components.append(current)
    return components


def allocate_components(
    components: list[list[str]], molecules: dict[str, dict], target: int, seed: int,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    rng = np.random.default_rng(seed)
    random_tie = rng.random(len(components))
    selected = {"discovery": [], "confirmation": []}
    assignment: dict[str, str] = {}
    counts = {split: Counter() for split in selected}
    remaining = set(range(len(components)))
    classes = ("acyclic", "single_ring", "multi_ring")
    component_counts = [Counter(molecules[ik]["ring_class"] for ik in component)
                        for component in components]
    # Allocate rare chemical classes first.  Whole mass-neighbor components
    # remain intact, so no candidate link can cross discovery/confirmation.
    for focus in classes:
        while remaining and min(counts[split][focus] for split in selected) < target:
            split = min(selected, key=lambda name: (counts[name][focus], len(selected[name]), name))
            candidates = []
            for component_id in remaining:
                component = components[component_id]
                comp = component_counts[component_id]
                if comp[focus] == 0 or len(component) > 40:
                    continue
                deficit = max(target - counts[split][focus], 0)
                useful = min(comp[focus], deficit)
                contamination = len(component) - comp[focus]
                overshoot = max(comp[focus] - deficit, 0)
                # Prefer high focus purity, then useful support, while the
                # seeded tie avoids deterministic mass-order artifacts.
                score = (contamination / len(component), overshoot, -useful,
                         len(component), random_tie[component_id])
                candidates.append((score, component_id))
            if not candidates:
                break
            _, component_id = min(candidates)
            remaining.remove(component_id)
            component = components[component_id]
            selected[split].extend(component)
            counts[split].update(component_counts[component_id])
            assignment.update({ik: split for ik in component})
    return selected, assignment


def preprocess_peaks(peaks: list[tuple[float, float]]) -> np.ndarray | None:
    if len(peaks) < 3:
        return None
    values = np.asarray(peaks, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1) & (values[:, 0] > 0) & (values[:, 1] > 0)]
    if len(values) < 3 or values[:, 0].max() > 1000:
        return None
    maximum, minimum = float(values[:, 1].max()), float(values[:, 1].min())
    if maximum <= 0 or maximum / minimum < 20:
        return None
    values[:, 1] /= maximum
    if int(np.sum(values[:, 1] >= 0.1)) < 3:
        return None
    values = values[np.argsort(values[:, 1])[::-1][:128]]
    return values[np.argsort(values[:, 0])].astype(np.float32)


def signature(peaks: np.ndarray) -> str:
    strongest = peaks[np.argsort(peaks[:, 1])[::-1][:32]]
    packed = np.column_stack((np.rint(strongest[:, 0] * 100), np.rint(strongest[:, 1] * 1000))).astype(np.int32)
    packed = packed[np.argsort(packed[:, 0])]
    return hashlib.sha1(packed.tobytes()).hexdigest()


def coarse_cosine(a: np.ndarray, b: np.ndarray, width: float = 0.02) -> float:
    left = {int(round(float(mz) / width)): float(value) for mz, value in a}
    right = {int(round(float(mz) / width)): float(value) for mz, value in b}
    dot = sum(left[key] * right[key] for key in set(left) & set(right))
    norm = math.sqrt(sum(x * x for x in left.values()) * sum(x * x for x in right.values()))
    return dot / norm if norm else 0.0


def stream_targets(
    path: Path, targets: dict[str, dict], max_spectra: int, adduct_ppm: float,
) -> tuple[dict[str, list[dict]], dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    audit = Counter()
    current = {"ik14": "", "precursor": math.nan, "ionmode": "", "peaks": [], "wanted": False}

    def finish() -> None:
        ik14 = current["ik14"]
        if not current["wanted"] or current["ionmode"] != "POSITIVE" or not math.isfinite(current["precursor"]):
            return
        audit["target_positive_spectra"] += 1
        expected = targets[ik14]["expected_precursor_mz"]
        if abs(float(current["precursor"]) - expected) / expected * 1e6 > adduct_ppm:
            audit["adduct_mass_rejected"] += 1
            return
        peaks = preprocess_peaks(current["peaks"])
        if peaks is None:
            audit["quality_rejected"] += 1
            return
        token = signature(peaks)
        if token in seen[ik14]:
            audit["duplicate_rejected"] += 1
            return
        seen[ik14].add(token)
        if len(pools[ik14]) < max_spectra:
            pools[ik14].append({"precursor_mz": float(current["precursor"]), "peaks": peaks})
            audit["retained_spectra"] += 1

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw in stream:
            line = raw.strip()
            if line == "BEGIN IONS":
                current = {"ik14": "", "precursor": math.nan, "ionmode": "", "peaks": [], "wanted": False}
            elif line == "END IONS":
                finish()
            elif line.startswith("INCHIKEY="):
                current["ik14"] = line[9:].strip()[:14]
                current["wanted"] = current["ik14"] in targets
            elif current["wanted"] and line.startswith("PEPMASS="):
                try: current["precursor"] = float(line[8:].split()[0].split("/")[0])
                except ValueError: current["precursor"] = math.nan
            elif current["wanted"] and line.startswith("IONMODE="):
                current["ionmode"] = line[8:].strip().upper()
            elif current["wanted"] and line and (line[0].isdigit() or line[0] == "-"):
                fields = line.split()
                if len(fields) >= 2:
                    try: current["peaks"].append((float(fields[0]), float(fields[1])))
                    except ValueError: pass
    return pools, dict(audit)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.indices.read_text(encoding="utf-8"))
    exclusions = decode_exclusions(args.massspecgym_metadata)
    molecules = molecule_table(index, exclusions, args.adduct_ppm)
    components = dense_components(molecules, args.ppm)
    selected, assignment = allocate_components(
        components, molecules, args.preselect_per_class_per_split, args.seed
    )
    target_ids = set(assignment)
    targets = {ik: molecules[ik] for ik in target_ids}
    pools, scan_audit = stream_targets(args.mgf, targets, args.max_spectra_per_unit, args.adduct_ppm)

    viable: dict[str, dict] = {}
    for ik14, spectra in pools.items():
        if len(spectra) < 2:
            continue
        pairs = [(coarse_cosine(spectra[i]["peaks"], spectra[j]["peaks"]), i, j)
                 for i in range(len(spectra) - 1) for j in range(i + 1, len(spectra))]
        pairs = [item for item in pairs if 0.05 <= item[0] < 0.995]
        if not pairs:
            continue
        similarity, left, right = min(pairs, key=lambda item: abs(item[0] - 0.5))
        viable[ik14] = dict(molecules[ik14]) | {
            "split": assignment[ik14], "positive_proxy_similarity": similarity,
            "spectra": [spectra[left], spectra[right]],
            "precursor_mz": float(np.mean([spectra[left]["precursor_mz"], spectra[right]["precursor_mz"]])),
        }

    by_split_adduct: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ik14, item in viable.items():
        by_split_adduct[(item["split"], item["adduct"])].append(ik14)
    neighbors: dict[str, list[str]] = defaultdict(list)
    for identifiers in by_split_adduct.values():
        identifiers.sort(key=lambda ik: viable[ik]["precursor_mz"])
        masses = np.asarray([viable[ik]["precursor_mz"] for ik in identifiers])
        for position, ik14 in enumerate(identifiers):
            mass = masses[position]; tolerance = mass * args.ppm * 1e-6
            left = np.searchsorted(masses, mass - tolerance, "left")
            right = np.searchsorted(masses, mass + tolerance, "right")
            neighbors[ik14] = [identifiers[j] for j in range(left, right) if identifiers[j] != ik14]
    kept = sorted(ik for ik in viable if neighbors[ik])
    remap = {ik: i for i, ik in enumerate(kept)}
    units = []
    spectra_array = np.zeros((len(kept), 2, 2, 128), dtype=np.float32)
    precursor_array = np.zeros((len(kept), 2), dtype=np.float32)
    for i, ik14 in enumerate(kept):
        item = viable[ik14]
        for view, spectrum in enumerate(item["spectra"]):
            peaks = spectrum["peaks"][:128]
            spectra_array[i, view, :, :len(peaks)] = peaks.T
            precursor_array[i, view] = spectrum["precursor_mz"]
        units.append({key: value for key, value in item.items() if key != "spectra"} | {
            "unit_id": i,
            "negative_unit_ids": [remap[other] for other in neighbors[ik14] if other in remap],
            "same_formula_negative_ids": [remap[other] for other in neighbors[ik14]
                                           if other in remap and viable[other]["formula"] == item["formula"]],
        })
    np.savez_compressed(
        args.output_dir / "spectra.npz", spectra=spectra_array,
        precursor_mz=precursor_array, unit_id=np.arange(len(kept)),
    )
    split_counts = {split: dict(Counter(item["ring_class"] for item in units if item["split"] == split))
                    for split in ("discovery", "confirmation")}
    report = {
        "status": "external_ring_stratified_cohort_v2",
        "protocol_limit": (
            "annotated01 has no source, instrument, collision energy, or adduct header. "
            "Positives are non-identical same-molecule spectra with mass-inferred adduct, "
            "not proven cross-condition replicates."
        ),
        "leakage_control": f"Excluded all {len(exclusions)} MassSpecGym IK14 molecules.",
        "candidate_molecules_after_exclusion_and_adduct_inference": len(molecules),
        "dense_components": len(components),
        "preselected_by_split": {split: dict(Counter(molecules[ik]["ring_class"] for ik in values))
                                 for split, values in selected.items()},
        "scan_audit": scan_audit,
        "selected_units": len(units),
        "selected_by_split_and_ring": split_counts,
        "discovery_confirmation_ik_overlap": 0,
        "units": units,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "units"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

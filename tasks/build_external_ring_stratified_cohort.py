"""Build a leakage-resistant, ring-stratified annotated01 retrieval cohort.

This is an external-source pilot.  All molecules present anywhere in
MassSpecGym metadata are excluded.  annotated01 lacks source, instrument and
collision-energy headers, so same-molecule positive spectra are selected as
non-identical spectral replicates and must not be described as cross-condition
replicates.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", type=Path, default=Path("data/annotated01.mgf"))
    parser.add_argument("--indices", type=Path, default=Path("tasks/_cache/indices.json"))
    parser.add_argument("--massspecgym-metadata", type=Path, default=Path("data/massspecgym/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_ring_stratified_cohort"))
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--adduct-ppm", type=float, default=20.0)
    parser.add_argument("--max-spectra-per-unit", type=int, default=5)
    parser.add_argument("--target-per-class-per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def infer_adduct(smiles: str, precursor: float, ppm: float) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    neutral = Descriptors.ExactMolWt(mol)
    candidates = {"[M+H]+": neutral + PROTON, "[M+Na]+": neutral + SODIUM}
    adduct, expected = min(candidates.items(), key=lambda item: abs(item[1] - precursor))
    error = abs(expected - precursor) / expected * 1e6
    return adduct if error <= ppm else None


def ring_class(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rings = rdMolDescriptors.CalcNumRings(mol)
    return "acyclic" if rings == 0 else "single_ring" if rings == 1 else "multi_ring"


def top_peaks(peaks: list[tuple[float, float]], n: int = 128) -> np.ndarray | None:
    if not peaks:
        return None
    values = np.asarray(peaks, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1) & (values[:, 0] > 0) & (values[:, 1] > 0)]
    if len(values) < 3 or values[:, 0].max() > 1000:
        return None
    order = np.argsort(values[:, 1])[::-1][:n]
    values = values[order]
    maximum = float(values[:, 1].max())
    if maximum <= 0:
        return None
    values[:, 1] /= maximum
    if int(np.sum(values[:, 1] >= 0.1)) < 3:
        return None
    positive = values[:, 1][values[:, 1] > 0]
    if maximum / max(float(positive.min() * maximum), 1e-12) < 20:
        return None
    return values[np.argsort(values[:, 0])].astype(np.float32)


def spectrum_signature(peaks: np.ndarray) -> str:
    strongest = peaks[np.argsort(peaks[:, 1])[::-1][:32]]
    packed = np.column_stack([np.rint(strongest[:, 0] * 100), np.rint(strongest[:, 1] * 1000)]).astype(np.int32)
    packed = packed[np.argsort(packed[:, 0])]
    return hashlib.sha1(packed.tobytes()).hexdigest()


def binned_cosine(left: np.ndarray, right: np.ndarray, width: float = 0.02) -> float:
    def vector(values: np.ndarray) -> dict[int, float]:
        result: dict[int, float] = {}
        for mz, intensity in values:
            key = int(round(float(mz) / width))
            result[key] = max(result.get(key, 0.0), float(intensity))
        return result
    a, b = vector(left), vector(right)
    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)
    norm = math.sqrt(sum(value * value for value in a.values()) * sum(value * value for value in b.values()))
    return dot / norm if norm else 0.0


class DisjointSet:
    def __init__(self, values): self.parent = {value: value for value in values}
    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[b] = a


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_index = json.loads(args.indices.read_text(encoding="utf-8"))
    excluded = set()
    with args.massspecgym_metadata.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            excluded.add(row["inchikey14"][:14])

    molecule_info: dict[str, dict] = {}
    for key, smiles in raw_index["ik_to_smi"].items():
        ik14 = key[:14]
        if ik14 in excluded or ik14 in molecule_info or int(raw_index["ik_counts"].get(key, 0)) < 2:
            continue
        formula = raw_index["ik_to_fm"].get(key)
        cls = ring_class(smiles)
        if formula and cls:
            molecule_info[ik14] = {"smiles": smiles, "formula": formula, "ring_class": cls}

    retained: dict[tuple[str, str], list[dict]] = defaultdict(list)
    signatures: dict[tuple[str, str], set[str]] = defaultdict(set)
    current = {"ik14": "", "smiles": "", "formula": "", "precursor": math.nan, "ionmode": "", "peaks": []}
    spectra_seen = 0

    def finish() -> None:
        nonlocal spectra_seen
        ik14 = current["ik14"]
        if ik14 not in molecule_info or current["ionmode"] != "POSITIVE" or not math.isfinite(current["precursor"]):
            return
        precursor = float(current["precursor"])
        if precursor <= 0 or precursor > 1000:
            return
        smiles = molecule_info[ik14]["smiles"]
        adduct = infer_adduct(smiles, precursor, args.adduct_ppm)
        if adduct is None:
            return
        peaks = top_peaks(current["peaks"])
        if peaks is None:
            return
        spectra_seen += 1
        key = (ik14, adduct)
        signature = spectrum_signature(peaks)
        if signature in signatures[key]:
            return
        signatures[key].add(signature)
        item = {"precursor_mz": precursor, "peaks": peaks, "signature": signature}
        pool = retained[key]
        pool.append(item)
        if len(pool) > args.max_spectra_per_unit:
            # Keep a diverse, high-information subset.
            scores = []
            for i, candidate in enumerate(pool):
                other = [1 - binned_cosine(candidate["peaks"], value["peaks"]) for j, value in enumerate(pool) if i != j]
                scores.append((float(np.mean(other)), len(candidate["peaks"]), i))
            keep = {item[2] for item in sorted(scores, reverse=True)[:args.max_spectra_per_unit]}
            retained[key] = [value for i, value in enumerate(pool) if i in keep]

    with args.mgf.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw in stream:
            line = raw.strip()
            if line == "BEGIN IONS":
                current = {"ik14": "", "smiles": "", "formula": "", "precursor": math.nan, "ionmode": "", "peaks": []}
            elif line == "END IONS":
                finish()
            elif line.startswith("INCHIKEY="):
                current["ik14"] = line[9:].strip()[:14]
            elif line.startswith("SMILES="):
                current["smiles"] = line[7:].strip()
            elif line.startswith("FORMULA="):
                current["formula"] = line[8:].strip()
            elif line.startswith("PEPMASS="):
                try: current["precursor"] = float(line[8:].split()[0].split("/")[0])
                except ValueError: current["precursor"] = math.nan
            elif line.startswith("IONMODE="):
                current["ionmode"] = line[8:].strip().upper()
            elif line and (line[0].isdigit() or line[0] == "-"):
                fields = line.split()
                if len(fields) >= 2:
                    try: current["peaks"].append((float(fields[0]), float(fields[1])))
                    except ValueError: pass

    units = []
    for (ik14, adduct), pool in retained.items():
        if len(pool) < 2:
            continue
        candidates = []
        for i in range(len(pool) - 1):
            for j in range(i + 1, len(pool)):
                similarity = binned_cosine(pool[i]["peaks"], pool[j]["peaks"])
                candidates.append((similarity, i, j))
        # Avoid exact duplicates, while not deliberately selecting unrelated extremes.
        viable = [item for item in candidates if item[0] < 0.995 and item[0] >= 0.05]
        if not viable:
            continue
        similarity, left, right = min(viable, key=lambda item: abs(item[0] - 0.5))
        info = molecule_info[ik14]
        units.append({
            "unit_id": len(units), "ik14": ik14, "adduct": adduct,
            "ring_class": info["ring_class"], "smiles": info["smiles"], "formula": info["formula"],
            "positive_proxy_similarity": similarity,
            "spectra": [pool[left], pool[right]],
            "precursor_mz": float(np.mean([pool[left]["precursor_mz"], pool[right]["precursor_mz"]])),
        })

    by_adduct: dict[str, list[int]] = defaultdict(list)
    for i, unit in enumerate(units): by_adduct[unit["adduct"]].append(i)
    links = set()
    for indices in by_adduct.values():
        order = sorted(indices, key=lambda i: units[i]["precursor_mz"])
        masses = np.asarray([units[i]["precursor_mz"] for i in order])
        for pos, index in enumerate(order):
            mass = masses[pos]; tolerance = mass * args.ppm * 1e-6
            left = np.searchsorted(masses, mass - tolerance, "left")
            right = np.searchsorted(masses, mass + tolerance, "right")
            neighbors = [order[j] for j in range(left, right) if order[j] != index and units[order[j]]["ik14"] != units[index]["ik14"]]
            units[index]["negative_unit_ids"] = neighbors
            for neighbor in neighbors: links.add(tuple(sorted((index, neighbor))))

    eligible = {i for i, unit in enumerate(units) if unit.get("negative_unit_ids")}
    dsu = DisjointSet(eligible)
    for a, b in links:
        if a in eligible and b in eligible: dsu.union(a, b)
    components: dict[int, list[int]] = defaultdict(list)
    for i in eligible: components[dsu.find(i)].append(i)
    rng = np.random.default_rng(args.seed)
    component_list = list(components.values()); rng.shuffle(component_list)
    component_list.sort(key=len, reverse=True)
    selected = {"discovery": [], "confirmation": []}
    class_counts = {split: Counter() for split in selected}
    for component in component_list:
        comp_counts = Counter(units[i]["ring_class"] for i in component)
        best = None
        for split in selected:
            overflow = sum(max(class_counts[split][cls] + count - args.target_per_class_per_split, 0) for cls, count in comp_counts.items())
            deficit_gain = sum(min(count, max(args.target_per_class_per_split - class_counts[split][cls], 0)) for cls, count in comp_counts.items())
            score = (overflow, -deficit_gain, len(selected[split]))
            if best is None or score < best[0]: best = (score, split)
        split = best[1]
        if best[0][1] == 0: continue
        selected[split].extend(component); class_counts[split].update(comp_counts)
        if all(class_counts[s][c] >= args.target_per_class_per_split for s in selected for c in ("acyclic", "single_ring", "multi_ring")):
            break

    selected_ids = set(selected["discovery"]) | set(selected["confirmation"])
    output_units = []
    remap = {old: new for new, old in enumerate(sorted(selected_ids))}
    split_by_old = {old: split for split, values in selected.items() for old in values}
    for old in sorted(selected_ids):
        unit = units[old]
        same_split_neighbors = [n for n in unit["negative_unit_ids"] if n in selected_ids and split_by_old[n] == split_by_old[old]]
        if not same_split_neighbors:
            continue
        output_units.append({
            key: value for key, value in unit.items() if key != "spectra"
        } | {
            "unit_id": remap[old], "split": split_by_old[old],
            "negative_unit_ids": [remap[n] for n in same_split_neighbors],
            "same_formula_negative_ids": [remap[n] for n in same_split_neighbors if units[n]["formula"] == unit["formula"]],
        })

    # Store only selected spectra in compact padded arrays.
    kept_old = [old for old in sorted(selected_ids) if any(item["unit_id"] == remap[old] for item in output_units)]
    spectra = np.zeros((len(kept_old), 2, 2, 128), dtype=np.float32)
    precursor = np.zeros((len(kept_old), 2), dtype=np.float32)
    unit_order = []
    for position, old in enumerate(kept_old):
        unit_order.append(remap[old])
        for view, spectrum in enumerate(units[old]["spectra"]):
            peaks = spectrum["peaks"][:128]
            spectra[position, view, :, :len(peaks)] = peaks.T
            precursor[position, view] = spectrum["precursor_mz"]
    np.savez_compressed(args.output_dir / "spectra.npz", spectra=spectra, precursor_mz=precursor, unit_id=np.asarray(unit_order))
    report = {
        "status": "external_ring_stratified_cohort",
        "protocol_limit": "annotated01 has no source, instrument, collision energy, or adduct header. Positives are non-identical same-molecule spectra with inferred adduct, not proven cross-condition replicates.",
        "leakage_control": f"Excluded all {len(excluded)} IK14 molecules in MassSpecGym metadata.",
        "eligible_molecules_before_spectrum_scan": len(molecule_info),
        "quality_positive_spectra_seen": spectra_seen,
        "same_adduct_replicate_units": len(units),
        "selected_units": len(output_units),
        "selected_by_split_and_ring": {split: dict(Counter(units[i]["ring_class"] for i in values if i in kept_old)) for split, values in selected.items()},
        "molecule_overlap_between_splits": 0,
        "ppm": args.ppm,
        "units": output_units,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "units"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Link validated peak-localized factors to molecule-local environments.

The comparison universe is restricted to molecules whose spectra contain the
same fixed target fragment/loss.  A radius-2 Morgan environment is selected on
discovery molecules and tested unchanged on molecule-disjoint confirmation.
The curated fragmentation-rule library is not read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import fisher_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-audit", type=Path, required=True)
    parser.add_argument("--localization-audit", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--minimum-positive-support", type=int, default=5)
    parser.add_argument("--minimum-negative-support", type=int, default=5)
    return parser.parse_args()


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def load_table(directory: Path, codes_path: Path) -> dict:
    spectra = json.loads((directory / "spectra.json").read_text(encoding="utf-8"))
    mask = np.load(directory / "peak_mask.npy")
    values = np.load(directory / "peak_values.npy")
    codes = np.load(codes_path, mmap_mode="r").astype(np.float32)
    counts = mask.sum(axis=1)
    spectrum_index = np.repeat(np.arange(len(spectra)), counts)
    mz = values[:, :, 0][mask].astype(float)
    precursor = np.asarray([item["precursor_mz"] for item in spectra], dtype=float)[spectrum_index]
    return {
        "spectra": spectra,
        "spectrum_index": spectrum_index,
        "mz": mz,
        "neutral_loss": precursor - mz,
        "codes": codes,
    }


def molecule_target_labels(table: dict, factor: int, kind: str, mass: float, width: float) -> list[dict]:
    values = table["mz"] if kind == "fragment_mz" else table["neutral_loss"]
    target = np.rint(values / width).astype(np.int64) == int(round(mass / width))
    rows: dict[str, dict] = {}
    for spectrum, record in enumerate(table["spectra"]):
        peak_rows = (table["spectrum_index"] == spectrum) & target
        if not np.any(peak_rows):
            continue
        ik14 = record["ik14"]
        item = rows.setdefault(ik14, {
            "ik14": ik14,
            "smiles": record["smiles"],
            "target_occurrences": 0,
            "active_occurrences": 0,
            "max_activation": 0.0,
        })
        score = table["codes"][peak_rows, factor]
        item["target_occurrences"] += int(len(score))
        item["active_occurrences"] += int(np.sum(score > 0))
        item["max_activation"] = max(item["max_activation"], float(score.max()))
    for item in rows.values():
        item["factor_positive"] = item["active_occurrences"] > 0
    return list(rows.values())


def fingerprint(smiles: str, n_bits: int, radius: int) -> tuple[np.ndarray, dict, Chem.Mol | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=bool), {}, None
    bit_info: dict[int, list[tuple[int, int]]] = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, bitInfo=bit_info
    )
    array = np.zeros(n_bits, dtype=bool)
    array[list(fp.GetOnBits())] = True
    return array, bit_info, mol


def environment_smiles(mol: Chem.Mol, center: int, radius: int) -> str:
    if radius == 0:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[center], canonical=True)
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    atoms = {center}
    for bond_idx in bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(atoms), bondsToUse=list(bonds), canonical=True
    )


def matrix(records: list[dict], n_bits: int, radius: int) -> tuple[np.ndarray, list[dict]]:
    output = np.zeros((len(records), n_bits), dtype=bool)
    metadata = []
    for i, record in enumerate(records):
        fp, bit_info, mol = fingerprint(record["smiles"], n_bits, radius)
        output[i] = fp
        metadata.append({"bit_info": bit_info, "mol": mol})
    return output, metadata


def effect(labels: np.ndarray, present: np.ndarray) -> dict:
    a = int(np.sum(labels & present))
    b = int(np.sum(labels & ~present))
    c = int(np.sum(~labels & present))
    d = int(np.sum(~labels & ~present))
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    pos = (a + 0.5) / (a + b + 1)
    neg = (c + 0.5) / (c + d + 1)
    return {
        "positive_with_environment": a,
        "positive_without_environment": b,
        "negative_with_environment": c,
        "negative_without_environment": d,
        "log2_enrichment": float(np.log2(pos / neg)),
        "odds_ratio": float(odds),
        "p": float(p),
    }


def select_environment(records: list[dict], x: np.ndarray, metadata: list[dict], args: argparse.Namespace) -> dict:
    labels = np.asarray([item["factor_positive"] for item in records], dtype=bool)
    candidates = []
    for bit in range(x.shape[1]):
        present = x[:, bit]
        result = effect(labels, present)
        if (
            result["positive_with_environment"] < args.minimum_positive_support
            or result["negative_with_environment"] < args.minimum_negative_support
        ):
            continue
        result["bit"] = bit
        candidates.append(result)
    if not candidates:
        return {"found": False, "screened_bits": 0}
    best = max(candidates, key=lambda item: (item["log2_enrichment"], item["positive_with_environment"]))
    representative = ""
    radius = None
    for i, record in enumerate(records):
        if labels[i] and x[i, best["bit"]] and metadata[i]["mol"] is not None:
            center, radius = metadata[i]["bit_info"][best["bit"]][0]
            representative = environment_smiles(metadata[i]["mol"], center, radius)
            break
    best.update({
        "found": True,
        "screened_bits": len(candidates),
        "representative_environment_smiles": representative,
        "environment_radius": radius,
    })
    return best


def main() -> None:
    args = parse_args()
    spectral = json.loads(args.spectral_audit.read_text(encoding="utf-8"))
    localization = json.loads(args.localization_audit.read_text(encoding="utf-8"))
    width = float(spectral["bin_width_da"])
    discovery = load_table(args.discovery, args.run / "discovery_codes.npy")
    confirmation = load_table(args.confirmation, args.run / "confirmation_codes.npy")
    passed = []
    for factor in localization["factors"]:
        factor_id = int(factor["factor"])
        for kind, result in factor["confirmation"].items():
            if result.get("localization_pass"):
                passed.append((factor_id, kind, float(result["fixed_mass_da"])))
    rows = []
    confirmation_p = []
    for factor, kind, mass in passed:
        d_records = molecule_target_labels(discovery, factor, kind, mass, width)
        c_records = molecule_target_labels(confirmation, factor, kind, mass, width)
        d_x, d_meta = matrix(d_records, args.n_bits, args.radius)
        c_x, _ = matrix(c_records, args.n_bits, args.radius)
        candidate = select_environment(d_records, d_x, d_meta, args)
        if candidate.get("found"):
            c_labels = np.asarray([item["factor_positive"] for item in c_records], dtype=bool)
            confirmation_result = effect(c_labels, c_x[:, candidate["bit"]])
            confirmation_result["tested"] = True
            confirmation_p.append(confirmation_result)
        else:
            confirmation_result = {"tested": False}
        rows.append({
            "factor": factor,
            "spectral_kind": kind,
            "fixed_mass_da": mass,
            "discovery_target_present_molecules": len(d_records),
            "discovery_factor_positive_molecules": int(sum(item["factor_positive"] for item in d_records)),
            "confirmation_target_present_molecules": len(c_records),
            "confirmation_factor_positive_molecules": int(sum(item["factor_positive"] for item in c_records)),
            "discovery_selected_environment": candidate,
            "confirmation_fixed_environment_test": confirmation_result,
        })
    if confirmation_p:
        q = bh(np.asarray([item["p"] for item in confirmation_p]))
        for item, value in zip(confirmation_p, q):
            item["bh_q"] = float(value)
    report = {
        "status": "peak_factor_structure_context_audit",
        "rules_read": False,
        "comparison_universe": "Only molecules containing the same fixed target fragment/loss.",
        "structure_feature": f"Morgan radius {args.radius}, {args.n_bits} hashed bits; selected on discovery and fixed on confirmation.",
        "factors": rows,
        "claim_limit": "An enriched local environment is a structural context association, not proof of a bond-cleavage mechanism.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

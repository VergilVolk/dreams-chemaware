"""Create data-driven local structure labels aligned to molecule-mean DreaMS embeddings."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parent.parent


def atom_environment_tokens(mol: Chem.Mol, radii: tuple[int, ...]) -> set[str]:
    tokens = set()
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        for radius in radii:
            bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_index))
            atoms = {atom_index}
            for bond_index in bonds:
                bond = mol.GetBondWithIdx(int(bond_index))
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            fragment = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=sorted(atoms),
                bondsToUse=sorted(bonds),
                rootedAtAtom=atom_index,
                canonical=True,
                isomericSmiles=False,
            )
            if fragment:
                tokens.add(f"r{radius}:{fragment}")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_manifest.json",
    )
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_embeddings.npy",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/double_mapping/structure_environment_probe_data.npz",
    )
    parser.add_argument("--radii", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--min-molecules", type=int, default=30)
    parser.add_argument("--max-prevalence", type=float, default=0.50)
    parser.add_argument("--max-environments", type=int, default=800)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    embeddings = np.load(args.embeddings, mmap_mode="r")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    smiles_by_ik: dict[str, str] = {}
    for row in manifest:
        ik = str(row["inchikey_14"])
        grouped_indices[ik].append(int(row["embedding_idx"]))
        smiles_by_ik.setdefault(ik, str(row["smiles"]))
    molecules = sorted(grouped_indices)
    molecule_embeddings = np.stack([
        np.asarray(embeddings[grouped_indices[ik]], dtype=np.float32).mean(axis=0)
        for ik in molecules
    ]).astype(np.float32)
    tokens_by_molecule = []
    scaffolds = []
    support = Counter()
    valid_molecules, valid_embeddings = [], []
    for index, ik in enumerate(molecules):
        mol = Chem.MolFromSmiles(smiles_by_ik[ik])
        if mol is None:
            continue
        tokens = atom_environment_tokens(mol, tuple(args.radii))
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaffold:
            scaffold = f"ACYCLIC:{Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)}"
        valid_molecules.append(ik)
        valid_embeddings.append(molecule_embeddings[index])
        tokens_by_molecule.append(tokens)
        scaffolds.append(scaffold)
        support.update(tokens)
    n = len(valid_molecules)
    selected = [
        token for token, count in support.items()
        if count >= args.min_molecules and count / n <= args.max_prevalence
    ]
    selected.sort(key=lambda token: (-support[token], token))
    selected = selected[: args.max_environments]
    token_index = {token: index for index, token in enumerate(selected)}
    labels = np.zeros((n, len(selected)), dtype=np.uint8)
    for row, tokens in enumerate(tokens_by_molecule):
        for token in tokens:
            if token in token_index:
                labels[row, token_index[token]] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=np.asarray(valid_embeddings, dtype=np.float32),
        labels=labels,
        ik14=np.asarray(valid_molecules, dtype="U14"),
        smiles=np.asarray([smiles_by_ik[ik] for ik in valid_molecules], dtype="U512"),
        scaffold=np.asarray(scaffolds, dtype="U512"),
        environment=np.asarray(selected, dtype="U512"),
        support=np.asarray([support[token] for token in selected], dtype=np.int32),
    )
    report = {
        "status": "structure_environment_probe_data_complete",
        "molecules": n,
        "source_spectra": len(manifest),
        "embedding_aggregation": "mean official DreaMS embedding per IK14",
        "environment_definition": f"canonical atom-centered fragments at radii {args.radii}",
        "selected_environments": len(selected),
        "minimum_molecule_support": args.min_molecules,
        "maximum_prevalence": args.max_prevalence,
        "selection": "frequency thresholds only; no embedding or outcome labels used",
        "claim_limit": "Environment SMILES are local contexts, not complete candidate substructures.",
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

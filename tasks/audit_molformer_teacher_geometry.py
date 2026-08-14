"""Audit MolFormer geometry on the strict 10 ppm molecular-neighbor graph."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOLFORMER = ROOT / "data/validation/molformer_factor_embeddings"
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_OUTPUT = ROOT / "data/validation/molformer_teacher_geometry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molformer", type=Path, default=DEFAULT_MOLFORMER)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "mean": float(np.mean(array)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "q25": float(np.quantile(array, 0.25)) if len(array) else None,
        "q75": float(np.quantile(array, 0.75)) if len(array) else None,
    }


def analyze(split: str, directory: Path, molformer: Path) -> dict:
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    records = load_records(molformer / f"{split}_records.csv")
    embedding = normalize(np.load(molformer / f"{split}.npy").astype(np.float64))
    if not (len(pairs) == len(records) == len(embedding)):
        raise RuntimeError(f"Length mismatch for {split}")
    molecules = [Chem.MolFromSmiles(record["canonical_isomeric_smiles"]) for record in records]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for mol in molecules]
    formulas = [rdMolDescriptors.CalcMolFormula(mol) for mol in molecules]
    scaffolds = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        for mol in molecules
    ]
    links = set()
    for left, pair in enumerate(pairs):
        for right in pair["negative_pair_ids"]:
            links.add(tuple(sorted((left, int(right)))))
    rows = []
    for left, right in sorted(links):
        rows.append({
            "molformer_cosine": float(embedding[left] @ embedding[right]),
            "morgan_tanimoto": float(DataStructs.TanimotoSimilarity(
                fingerprints[left], fingerprints[right]
            )),
            "same_formula": formulas[left] == formulas[right],
            "same_scaffold": bool(scaffolds[left]) and scaffolds[left] == scaffolds[right],
            "same_nonisomeric_smiles": (
                records[left]["molformer_smiles"] == records[right]["molformer_smiles"]
            ),
        })
    cosine = np.asarray([row["molformer_cosine"] for row in rows])
    tanimoto = np.asarray([row["morgan_tanimoto"] for row in rows])
    same_formula = [row["molformer_cosine"] for row in rows if row["same_formula"]]
    different_formula = [row["molformer_cosine"] for row in rows if not row["same_formula"]]
    same_scaffold = [row["molformer_cosine"] for row in rows if row["same_scaffold"]]
    different_scaffold = [row["molformer_cosine"] for row in rows if not row["same_scaffold"]]
    return {
        "n_molecules": len(records),
        "n_unique_undirected_10ppm_links": len(rows),
        "molformer_vs_morgan_spearman": float(spearmanr(cosine, tanimoto).statistic),
        "molformer_cosine_overall": summarize(cosine.tolist()),
        "molformer_cosine_same_formula": summarize(same_formula),
        "molformer_cosine_different_formula": summarize(different_formula),
        "molformer_cosine_same_scaffold": summarize(same_scaffold),
        "molformer_cosine_different_scaffold": summarize(different_scaffold),
        "morgan_tanimoto": summarize(tanimoto.tolist()),
        "same_formula_links": int(sum(row["same_formula"] for row in rows)),
        "same_scaffold_links": int(sum(row["same_scaffold"] for row in rows)),
        "collapsed_nonisomeric_links": int(sum(
            row["same_nonisomeric_smiles"] for row in rows
        )),
    }


def main() -> None:
    args = parse_args()
    result = {
        "status": "molformer_teacher_geometry_audit",
        "protocol": (
            "MolFormer cosine similarity is compared with radius-2 Morgan "
            "Tanimoto on the exact different-molecule 10 ppm neighbor graph."
        ),
        "discovery": analyze("discovery", args.discovery, args.molformer),
        "confirmation": analyze("confirmation", args.confirmation, args.molformer),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split in ("discovery", "confirmation"):
        values = result[split]
        print(
            f"{split}: links={values['n_unique_undirected_10ppm_links']}; "
            f"MolFormer-vs-Morgan rho={values['molformer_vs_morgan_spearman']:.3f}; "
            f"same-formula={values['same_formula_links']}; "
            f"same-scaffold={values['same_scaffold_links']}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

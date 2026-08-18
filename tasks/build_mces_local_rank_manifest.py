"""Collapse strict-10ppm spectrum edges into leakage-safe molecule pairs.

This is the inexpensive first half of MCES local-rank preparation.  It creates
the complete unique molecule-pair universe and audits fold/adduct/ppm/identity
constraints.  MCES values are deliberately computed in a separate resumable
step so no RDKit/myopic-MCES work is performed online during model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from train_e1_identity import CandidatePool


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"


def decode(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {"min": None, "median": None, "p90": None, "max": None, "mean": None}
    values = np.asarray(values, dtype=float)
    return {
        "min": float(values.min()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)), "max": float(values.max()),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-pool", type=Path, default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz")
    parser.add_argument("--val-pool", type=Path, default=ROOT / "data/e1/e1_val_triplet_pool_10ppm.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e2/mces_local_rank")
    parser.add_argument("--max-edges", type=int, default=0, help="Debug cap per split; 0 uses all edges.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as handle:
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])])
        smiles = decode(handle["smiles"][:])
        formula = decode(handle["FORMULA"][:])
        folds = decode(handle["fold"][:])
        adducts = decode(handle["adduct"][:])
        precursor = np.asarray(handle["precursor_mz"][:], dtype=float)

    unique_ik, molecule_code = np.unique(ik14, return_inverse=True)
    reports = {}
    split_iks = {}
    for split, path in (("train", args.train_pool), ("val", args.val_pool)):
        pool = CandidatePool(path)
        counts = np.diff(pool.negative_ptr).astype(np.int64)
        anchor_rows = np.repeat(pool.anchor_idx.astype(np.int64), counts)
        negative_rows = pool.negative_idx.astype(np.int64)
        if args.max_edges:
            anchor_rows = anchor_rows[:args.max_edges]
            negative_rows = negative_rows[:args.max_edges]
        a_code = molecule_code[anchor_rows].astype(np.uint64)
        b_code = molecule_code[negative_rows].astype(np.uint64)
        low, high = np.minimum(a_code, b_code), np.maximum(a_code, b_code)
        packed = (low << np.uint64(32)) | high
        unique_packed, first = np.unique(packed, return_index=True)
        low_unique = (unique_packed >> np.uint64(32)).astype(np.int64)
        high_unique = (unique_packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
        rep_a, rep_b = anchor_rows[first], negative_rows[first]
        rep_a_code = molecule_code[rep_a]
        swap = rep_a_code != low_unique
        spectrum_a = np.where(swap, rep_b, rep_a)
        spectrum_b = np.where(swap, rep_a, rep_b)
        ppm = np.abs(precursor[spectrum_a] - precursor[spectrum_b]) / precursor[spectrum_a] * 1e6

        frame = pd.DataFrame({
            "split": split,
            "molecule_a_code": low_unique,
            "molecule_b_code": high_unique,
            "ik14_a": unique_ik[low_unique],
            "ik14_b": unique_ik[high_unique],
            "smiles_a": smiles[spectrum_a],
            "smiles_b": smiles[spectrum_b],
            "formula_a": formula[spectrum_a],
            "formula_b": formula[spectrum_b],
            "same_formula": formula[spectrum_a] == formula[spectrum_b],
            "representative_spectrum_a": spectrum_a,
            "representative_spectrum_b": spectrum_b,
            "precursor_ppm": ppm,
            "adduct_a": adducts[spectrum_a],
            "adduct_b": adducts[spectrum_b],
            "fold_a": folds[spectrum_a],
            "fold_b": folds[spectrum_b],
            "mces": np.nan,
            "mces_status": "pending",
        })
        output = args.output_dir / f"{split}_unique_molecule_pairs.csv"
        frame.to_csv(output, index=False)

        degree = np.bincount(
            np.concatenate([low_unique, high_unique]), minlength=len(unique_ik)
        )
        involved = np.unique(np.concatenate([low_unique, high_unique]))
        split_iks[split] = set(unique_ik[involved])
        reports[split] = {
            "source_pool": str(path.resolve()),
            "spectrum_negative_edges": int(len(anchor_rows)),
            "unique_molecule_pairs": int(len(frame)),
            "unique_molecules": int(len(involved)),
            "molecules_with_at_least_two_local_candidates": int(np.sum(degree[involved] >= 2)),
            "same_formula_pairs": int(frame["same_formula"].sum()),
            "same_formula_fraction": float(frame["same_formula"].mean()),
            "ppm": describe(ppm),
            "identity_violations": int(np.sum(frame["ik14_a"] == frame["ik14_b"])),
            "adduct_mismatches": int(np.sum(frame["adduct_a"] != frame["adduct_b"])),
            "fold_mismatches": int(np.sum(frame["fold_a"] != frame["fold_b"])),
            "ppm_violations": int(np.sum(ppm > 10.0 + 1e-8)),
            "output": str(output.resolve()),
        }
        print(json.dumps({split: reports[split]}, ensure_ascii=False, indent=2), flush=True)

    report = {
        "status": "mces_candidate_manifest_complete_values_pending",
        "protocol": "unique different-IK14 molecule pairs inherited from exact 10-ppm [M+H]+ pools",
        "train_val_ik14_overlap": int(len(split_iks["train"] & split_iks["val"])),
        "splits": reports,
        "next_step": "compute each unique pair MCES once into a resumable cache, then form relative-rank triplets only for anchors with >=2 candidates",
    }
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

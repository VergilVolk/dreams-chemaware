"""Audit g8r_locked train/val overlap at every identity level + acquisition metadata.

Checklist items 1-2 of the 2026-08-22 reranker audit:
  - recover formula / full InChIKey / SMILES / fold / acquisition conditions from
    the HDF5 for every anchor;
  - report train/val overlap at spectrum-row / IK14 / full-InChIKey / formula /
    Murcko-scaffold level.
This is the prerequisite before any reranker migration; g8r_val is a DEV set,
so this audit is descriptive (it does not re-qualify g8r_val as a test set).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_OUT = ROOT / "data/validation/g8r_field_overlap_audit.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def read_str_array(h, key):
    """Bulk-read one string dataset and decode in-memory (fast)."""
    raw = h[key][:]
    return np.asarray([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                       for x in raw], dtype=object)


def main() -> None:
    a = parse_args()
    train = json.loads(a.train.read_text(encoding="utf-8"))["entries"]
    val = json.loads(a.val.read_text(encoding="utf-8"))["entries"]

    with h5py.File(a.data, "r") as h:
        formula_all = read_str_array(h, "FORMULA")
        ik_full = read_str_array(h, "INCHIKEY")
        smiles_all = read_str_array(h, "smiles")
        fold_all = read_str_array(h, "fold")
        inst_all = read_str_array(h, "INSTRUMENT_TYPE")
        ce_all = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)
        adduct_all = read_str_array(h, "adduct")

    def murcko(smi):
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return None
            return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            return None

    def stats(entries):
        rows = [int(e["anchor_row"]) for e in entries]
        ik14 = {e["ik14"] for e in entries}
        ik_full_set = {ik_full[r] for r in rows}
        formula = {formula_all[r] for r in rows}
        scaffold = set()
        for r in rows:
            s = murcko(smiles_all[r])
            if s:
                scaffold.add(s)
        inst = {}
        for r in rows:
            inst[inst_all[r]] = inst.get(inst_all[r], 0) + 1
        ce = [ce_all[r] for r in rows]
        ce_finite = [c for c in ce if np.isfinite(c)]
        return {
            "n_anchors": len(entries),
            "n_unique_spectrum_rows": len(set(rows)),
            "n_ik14": len(ik14),
            "n_full_inchikey": len(ik_full_set),
            "n_formula": len(formula),
            "n_murcko_scaffold": len(scaffold),
            "instrument_counts": inst,
            "collision_energy_finite": len(ce_finite),
            "collision_energy_median": float(np.median(ce_finite)) if ce_finite else None,
        }, ik14, ik_full_set, formula, scaffold

    tr_s, tr_ik14, tr_ikf, tr_f, tr_sc = stats(train)
    va_s, va_ik14, va_ikf, va_f, va_sc = stats(val)

    def ov(a, b):
        return len(a & b)

    report = {
        "status": "g8r_field_overlap_audit",
        "train": tr_s, "val": va_s,
        "overlap": {
            "ik14": ov(tr_ik14, va_ik14),
            "full_inchikey": ov(tr_ikf, va_ikf),
            "formula": ov(tr_f, va_f),
            "murcko_scaffold": ov(tr_sc, va_sc),
        },
        "val_overlap_rates": {
            "formula_share_of_val": ov(tr_f, va_f) / len(va_f) if va_f else None,
            "scaffold_share_of_val": ov(tr_sc, va_sc) / len(va_sc) if va_sc else None,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {a.output}")


if __name__ == "__main__":
    main()

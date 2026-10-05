#!/usr/bin/env python
from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from build_gnps_gold_silver_10ppm_benchmark import PROTON, STATUS, build


def write_record(handle, smiles: str, quality: int, index: int, filename: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    inchi = Chem.MolToInchi(mol)
    precursor = Descriptors.ExactMolWt(mol) + PROTON
    handle.write("BEGIN IONS\n")
    handle.write(f"PEPMASS={precursor:.8f}\n")
    handle.write("IONMODE=Positive\nMSLEVEL=2\n")
    handle.write(f"LIBRARYQUALITY={quality}\n")
    handle.write(f"SMILES={smiles}\nINCHI={inchi}\n")
    handle.write(f"FILENAME={filename}\n")
    handle.write(f"SPECTRUMID=TEST{index}\nUSI=mzspec:TEST:{index}\n")
    for peak in range(10):
        handle.write(f"{20 + peak + index * 0.001:.6f} {10 + peak:.6f}\n")
    handle.write("END IONS\n")


def test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        gnps = root / "gnps.mgf"
        with gnps.open("w", encoding="utf-8") as handle:
            # Two same-formula identities, each with independent replicate files.
            write_record(handle, "CCO", 1, 0, "ethanol_a.raw")
            write_record(handle, "CCO", 2, 1, "ethanol_b.raw")
            write_record(handle, "COC", 1, 2, "ether_a.raw")
            write_record(handle, "COC", 2, 3, "ether_b.raw")
            # Rejected because it is Bronze.
            write_record(handle, "CCO", 3, 4, "bronze.raw")
            # Rejected because the identity appears in the development HDF5.
            write_record(handle, "CC(=O)C", 1, 5, "excluded.raw")

        hdf5 = root / "development.hdf5"
        excluded = Chem.MolToInchiKey(Chem.MolFromSmiles("CC(=O)C"))
        dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(hdf5, "w") as handle:
            handle.create_dataset("INCHIKEY", data=np.asarray([excluded], dtype=object), dtype=dtype)
            handle.create_dataset("FORMULA", data=np.asarray(["C3H6O"], dtype=object), dtype=dtype)

        mona = root / "mona.mgf"
        with mona.open("w", encoding="utf-8") as handle:
            write_record(handle, "c1ccccc1", 1, 6, "mona.raw")

        output = root / "benchmark"
        args = Namespace(
            gnps=gnps,
            massspecgym_hdf5=hdf5,
            mona_mgf=[mona],
            output_dir=output,
            ppm=10.0,
            precursor_validation_ppm=30.0,
            precursor_abs_da=0.01,
            min_peaks=10,
            queries_per_identity=1,
            references_per_identity=2,
            min_negative_identities=1,
            max_negative_identities=20,
            require_independent_positive=True,
            minimum_queries=2,
            minimum_pairs=4,
            minimum_formula_queries=2,
            seed=20260928,
            progress_every=1000,
            overwrite=False,
        )
        report = build(args)
        assert report["status"] == STATUS
        assert report["formal"] is True
        assert report["scan"]["accepted_spectra"] == 4
        assert report["scan"]["quality_rejected"] == 1
        assert report["scan"]["identity_overlap"] == 1
        assert report["identity_disjoint"]["queries"] == 2
        assert report["formula_disjoint"]["queries"] == 2
        assert report["identity_disjoint"]["positive_pairs"] == 2
        assert report["identity_disjoint"]["negative_pairs"] == 4
        with np.load(output / "panel_identity_disjoint.npz") as panel:
            assert panel["query_row"].shape == (2,)
            assert np.all(panel["independent_positive"])
            assert np.all(panel["near_query"])
            assert panel["molecule_label"].tolist() == [True, False, True, False]
        with np.load(output / "pairs_identity_disjoint.npz") as pairs:
            assert pairs["label"].tolist().count(True) == 2
            assert pairs["label"].tolist().count(False) == 4
        assert (output / "checksums.sha256").is_file()


def main() -> None:
    test()
    print("[test_build_gnps_gold_silver_10ppm_benchmark] PASS")


if __name__ == "__main__":
    main()

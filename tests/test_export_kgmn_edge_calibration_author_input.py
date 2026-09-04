from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_author_export_preserves_order_and_all_valid_fragments(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    triples_path = manifest / "paired_reaction_decoy_triples.csv.gz"
    triples = pd.DataFrame(
        {
            "source_row": [2, 0],
            "positive_row": [0, 2],
            "decoy_row": [1, 1],
        }
    )
    triples.to_csv(triples_path, index=False, compression="gzip")
    (manifest / "report.json").write_text(
        json.dumps(
            {
                "status": "kgmn_dreams_edge_calibration_manifest_frozen",
                "provenance": {"triples_sha256": sha256(triples_path)},
            }
        ),
        encoding="utf-8",
    )

    hdf5_path = tmp_path / "spectra.hdf5"
    spectra = np.zeros((3, 2, 4), dtype=np.float32)
    spectra[0, :, :2] = [[100.0, 50.0], [1.0, 0.5]]
    spectra[1, :, :3] = [[90.0, 120.0, 75.0], [0.2, 0.8, 0.4]]
    spectra[2, :, :2] = [[200.0, 125.0], [0.9, 0.3]]
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("spectrum", data=spectra)
        handle.create_dataset("precursor_mz", data=np.asarray([150.0, 130.0, 250.0]))

    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tasks" / "export_kgmn_edge_calibration_author_input.py"),
            "--manifest-dir",
            str(manifest),
            "--data",
            str(hdf5_path),
            "--output-dir",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    pairs = pd.read_csv(output / "pairs.csv.gz")
    assert pairs["triple_index"].tolist() == [0, 1]
    assert pairs[["source_row", "positive_row", "decoy_row"]].to_dict("records") == triples.to_dict("records")

    fragments = pd.read_csv(output / "spectra_long.csv.gz")
    assert len(fragments) == 7
    assert fragments.groupby("hdf5_row").size().to_dict() == {0: 2, 1: 3, 2: 2}
    assert fragments.groupby("hdf5_row")["fragment_mz"].apply(list).to_dict() == {
        0: [50.0, 100.0],
        1: [75.0, 90.0, 120.0],
        2: [125.0, 200.0],
    }
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["contracts"]["author_score_not_computed_in_python"] is True
    assert report["provenance"]["spectra_long_sha256"] == sha256(output / "spectra_long.csv.gz")
    assert report["provenance"]["pairs_sha256"] == sha256(output / "pairs.csv.gz")

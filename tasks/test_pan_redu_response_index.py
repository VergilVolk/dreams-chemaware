from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from encode_pan_redu_panel_spectra import load_mgf


def test_mgf_anchor_ids_are_preserved(tmp_path: Path) -> None:
    mgf = tmp_path / "spectra.mgf"
    mgf.write_text(
        "BEGIN IONS\nTITLE=A1\nFEATURE_ID=A1\nPEPMASS=100.1\n"
        "20 1\n30 2\n40 3\nEND IONS\n",
        encoding="utf-8",
    )
    ids, records = load_mgf(mgf)
    assert ids == ["A1"]
    assert records[0]["precursor_mz"] == 100.1
    assert np.asarray(records[0]["peaks"]).shape == (2, 3)


def test_effects_and_embeddings_assemble_without_calling_missing_identity_unknown(
    tmp_path: Path,
) -> None:
    jobs = pd.DataFrame([{
        "job_index": 0, "panel_id": "P1", "dataset_accession": "S1",
        "sample_files": 12, "biological_units": 12, "expected_bytes": 1,
        "quantification_manifest": "unused", "output_dir": "unused",
    }])
    jobs_path = tmp_path / "jobs.tsv"
    jobs.to_csv(jobs_path, sep="\t", index=False)
    topology = pd.DataFrame([{
        "panel_id": "P1", "contrast_id": "C1", "dataset_accession": "S1",
        "preliminary_source_family": "F1", "comparison_stratum": "diet_mouse_plasma",
        "contrast_field": "Diet", "level_a": "normal", "level_b": "high_fat",
        "reported_units_crossing_levels": 0,
        "topology_status": "PASS_REPORTED_UNIT_TOPOLOGY_METADATA_ONLY",
    }])
    topology_path = tmp_path / "topology.tsv"
    topology.to_csv(topology_path, sep="\t", index=False)
    effects_root = tmp_path / "effects"
    embeddings_root = tmp_path / "embeddings"
    effect_dir, embedding_dir = effects_root / "P1", embeddings_root / "P1"
    effect_dir.mkdir(parents=True)
    embedding_dir.mkdir(parents=True)
    (effect_dir / "report.json").write_text(json.dumps({
        "status": "development_effect_objects_not_cross_study_evidence"
    }), encoding="utf-8")
    (embedding_dir / "report.json").write_text(json.dumps({
        "status": "phenotype_blind_frozen_spectral_embeddings"
    }), encoding="utf-8")
    effects = pd.DataFrame({
        "contrast_id": ["C1", "C1"], "anchor_id": ["A1", "A2"],
        "beta_b_minus_a": [1.0, -1.0], "se": [0.5, 0.5],
        "unit_prevalence": [1.0, 1.0], "passes_effect_qc": [True, True],
    })
    effects.to_csv(effect_dir / "contrast_effects.tsv.gz", sep="\t", index=False, compression="gzip")
    np.savez_compressed(
        embedding_dir / "anchor_embeddings.npz",
        anchor_id=np.asarray(["A1", "A2"]),
        embeddings=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    output = tmp_path / "index"
    script = Path(__file__).with_name("assemble_pan_redu_signed_response_index.py")
    subprocess.run([
        sys.executable, str(script), "--panel-jobs", str(jobs_path),
        "--topology", str(topology_path), "--effects-root", str(effects_root),
        "--embeddings-root", str(embeddings_root), "--output-dir", str(output),
        "--minimum-entities", "1",
    ], check=True, capture_output=True, text=True)
    manifest = pd.read_csv(output / "contrast_manifest.tsv", sep="\t")
    assert manifest.loc[0, "n_entities_all"] == 2
    assert manifest.loc[0, "n_entities_known"] == 0
    assert manifest.loc[0, "n_entities_unresolved"] == 0
    assert not bool(manifest.loc[0, "identity_partition_available"])
    with np.load(output / "contrast_sketches.npz") as packed:
        assert np.linalg.norm(packed["positive_all"][0]) > 0
        assert np.linalg.norm(packed["negative_all"][0]) > 0
        assert np.linalg.norm(packed["positive_unresolved"][0]) == 0


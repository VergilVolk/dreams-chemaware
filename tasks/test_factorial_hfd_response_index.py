from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def test_factorial_index_equal_panel_mixture_and_missing_identity_is_not_unknown(tmp_path: Path) -> None:
    jobs = pd.DataFrame([
        {"job_index": 0, "panel_id": "P1", "dataset_accession": "S1"},
        {"job_index": 1, "panel_id": "P2", "dataset_accession": "S1"},
    ])
    jobs_path = tmp_path / "jobs.tsv"
    jobs.to_csv(jobs_path, sep="\t", index=False)
    registry = pd.DataFrame([
        {"panel_id": "P1", "dataset_accession": "S1", "source_family": "F1"},
        {"panel_id": "P2", "dataset_accession": "S1", "source_family": "F1"},
    ])
    registry_path = tmp_path / "registry.tsv"
    registry.to_csv(registry_path, sep="\t", index=False)
    effects_root = tmp_path / "effects"
    embeddings_root = tmp_path / "embeddings"
    for panel_id, embeddings in (
        ("P1", np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32)),
        ("P2", np.asarray([[0.0, 1.0], [1.0, 0.0]], np.float32)),
    ):
        effect_dir = effects_root / panel_id
        embedding_dir = embeddings_root / panel_id
        effect_dir.mkdir(parents=True)
        embedding_dir.mkdir(parents=True)
        (effect_dir / "report.json").write_text(json.dumps({
            "status": "development_factorial_effect_objects_not_cross_study_evidence"
        }), encoding="utf-8")
        (embedding_dir / "report.json").write_text(json.dumps({
            "status": "phenotype_blind_frozen_spectral_embeddings"
        }), encoding="utf-8")
        rows = []
        for contrast in (
            "diet_reference", "diet_modified", "genotype_control",
            "genotype_hfd", "genotype_by_hfd",
        ):
            rows.extend([
                {"panel_id": panel_id, "anchor_id": "A1", "contrast_type": contrast,
                 "beta": 1.0, "se_hc3": 0.5, "passes_effect_qc": True, "unit_prevalence": 1.0},
                {"panel_id": panel_id, "anchor_id": "A2", "contrast_type": contrast,
                 "beta": -1.0, "se_hc3": 0.5, "passes_effect_qc": True, "unit_prevalence": 1.0},
            ])
        pd.DataFrame(rows).to_csv(
            effect_dir / "factorial_contrast_effects.tsv.gz", sep="\t", index=False, compression="gzip"
        )
        np.savez_compressed(
            embedding_dir / "anchor_embeddings.npz",
            anchor_id=np.asarray(["A1", "A2"]), embeddings=embeddings,
        )

    output = tmp_path / "index"
    script = Path(__file__).with_name("assemble_factorial_hfd_response_index.py")
    subprocess.run([
        sys.executable, str(script), "--panel-jobs", str(jobs_path),
        "--panel-registry", str(registry_path), "--effects-root", str(effects_root),
        "--embeddings-root", str(embeddings_root), "--output-dir", str(output),
        "--minimum-entities", "1",
    ], check=True, capture_output=True, text=True)
    manifest = pd.read_csv(output / "study_contrast_manifest.tsv", sep="\t")
    assert len(manifest) == 5
    assert (manifest.n_entities_all == 4).all()
    assert (manifest.n_entities_unresolved == 0).all()
    assert not manifest.identity_partition_available.astype(bool).any()
    with np.load(output / "study_contrast_sketches.npz") as packed:
        # Equal-panel averaging makes the two coordinate-swapped panels symmetric.
        np.testing.assert_allclose(packed["positive_all"][0], [1 / 4, 1 / 4], atol=1e-6)
        np.testing.assert_allclose(packed["negative_all"][0], [1 / 4, 1 / 4], atol=1e-6)
        np.testing.assert_allclose(packed["chemistry_all"][0], [1 / 2, 1 / 2], atol=1e-6)
        assert np.linalg.norm(packed["positive_unresolved"][0]) == 0


def test_prepare_jobs_keeps_design_out_of_quantification_manifest(tmp_path: Path) -> None:
    wave = tmp_path / "wave"
    (wave / "phenotype_blind").mkdir(parents=True)
    acquisition = pd.DataFrame([
        {"panel_id": "P1", "dataset_accession": "S1", "sample_id": "A",
         "biological_unit_id": "A", "filename": "a.mzML", "download_url": "https://x/a",
         "local_path": "raw/a.mzML", "deposited_bytes": 10},
        {"panel_id": "P1", "dataset_accession": "S1", "sample_id": "B",
         "biological_unit_id": "B", "filename": "b.mzML", "download_url": "https://x/b",
         "local_path": "raw/b.mzML", "deposited_bytes": 20},
    ])
    acquisition.to_csv(wave / "phenotype_blind" / "acquisition_manifest.tsv", sep="\t", index=False)
    pd.DataFrame([{
        "panel_id": "P1", "dataset_accession": "S1", "source_family": "F1",
        "biological_units": 2, "cell_00": 1, "cell_01": 1, "cell_10": 1, "cell_11": 1,
    }]).to_csv(wave / "panel_registry.tsv", sep="\t", index=False)
    output = tmp_path / "jobs"
    script = Path(__file__).with_name("prepare_factorial_hfd_panel_jobs.py")
    subprocess.run([
        sys.executable, str(script), "--wave-dir", str(wave), "--output-dir", str(output)
    ], check=True, capture_output=True, text=True)
    blind = pd.read_csv(
        output / "phenotype_blind_panels" / "P1" / "quantification_manifest.tsv", sep="\t"
    )
    assert set(blind) == {
        "sample_id", "biological_unit_id", "dataset_accession", "filename",
        "download_url", "local_path", "deposited_bytes",
    }
    assert not {"genotype", "diet", "genotype_code", "diet_code"} & set(blind)


def test_truth_blind_query_scores_direction_and_chemistry_separately(tmp_path: Path) -> None:
    manifest = pd.DataFrame([
        {"study_id": "S1", "contrast_id": "interaction", "n_entities_all": 20,
         "n_entities_known": 0, "n_entities_unresolved": 0},
        {"study_id": "S2", "contrast_id": "interaction", "n_entities_all": 20,
         "n_entities_known": 0, "n_entities_unresolved": 0},
        {"study_id": "S2", "contrast_id": "main", "n_entities_all": 20,
         "n_entities_known": 0, "n_entities_unresolved": 0},
    ])
    manifest_path = tmp_path / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    zero = np.zeros((3, 2), dtype=np.float32)
    arrays = {}
    for world in ("all", "known", "unresolved"):
        arrays[f"positive_{world}"] = zero.copy()
        arrays[f"negative_{world}"] = zero.copy()
        arrays[f"chemistry_{world}"] = zero.copy()
    arrays["positive_all"] = np.asarray([[1, 0], [1, 0], [0, 1]], np.float32)
    arrays["negative_all"] = np.asarray([[0, 0], [0, 0], [1, 0]], np.float32)
    arrays["chemistry_all"] = np.asarray([[1, 1], [1, 1], [1, 1]], np.float32)
    sketches_path = tmp_path / "sketches.npz"
    np.savez_compressed(sketches_path, **arrays)
    output = tmp_path / "query"
    script = Path(__file__).with_name("query_factorial_hfd_response_index.py")
    subprocess.run([
        sys.executable, str(script), "--manifest", str(manifest_path),
        "--sketches", str(sketches_path), "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    scores = pd.read_csv(output / "retrieval_scores.tsv.gz", sep="\t")
    block = scores.loc[(scores.query_study_id == "S1") & (scores.evidence_world == "all")]
    interaction = block.loc[block.target_contrast_id == "interaction"].iloc[0]
    main = block.loc[block.target_contrast_id == "main"].iloc[0]
    assert interaction.signed_response_score > main.signed_response_score
    assert interaction.rank_signed_response_score == 1
    assert interaction.chemistry_only_score == main.chemistry_only_score
    assert interaction.rank_chemistry_only_score == main.rank_chemistry_only_score == 1

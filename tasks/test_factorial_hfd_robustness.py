from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


CONTRASTS = (
    "diet_reference", "diet_modified", "genotype_control",
    "genotype_hfd", "genotype_by_hfd",
)


def test_reciprocal_gate_uses_max_null_and_panel_jackknife(tmp_path: Path) -> None:
    panel_rows = []
    positive = []
    negative = []
    chemistry = []
    for study in ("MTBLS135", "MTBLS3316"):
        for panel_id in (f"{study}_P1", f"{study}_P2"):
            for contrast in CONTRASTS:
                panel_rows.append({
                    "sketch_row": len(panel_rows), "study_id": study,
                    "panel_id": panel_id, "contrast_id": contrast,
                    "n_entities_all": 20, "n_entities_known": 0, "n_entities_unresolved": 0,
                })
                positive.append([1.0, 0.0] if contrast == "genotype_by_hfd" else [0.0, 0.5])
                negative.append([0.0, 0.0])
                chemistry.append([0.5, 0.5])
    panel_manifest = pd.DataFrame(panel_rows)
    panel_manifest_path = tmp_path / "panel_manifest.tsv"
    panel_manifest.to_csv(panel_manifest_path, sep="\t", index=False)
    arrays = {}
    for world in ("all", "known", "unresolved"):
        if world == "all":
            arrays[f"positive_{world}"] = np.asarray(positive, np.float32)
            arrays[f"negative_{world}"] = np.asarray(negative, np.float32)
            arrays[f"chemistry_{world}"] = np.asarray(chemistry, np.float32)
        else:
            arrays[f"positive_{world}"] = np.zeros((len(panel_rows), 2), np.float32)
            arrays[f"negative_{world}"] = np.zeros((len(panel_rows), 2), np.float32)
            arrays[f"chemistry_{world}"] = np.zeros((len(panel_rows), 2), np.float32)
    panel_sketches_path = tmp_path / "panel_sketches.npz"
    np.savez_compressed(panel_sketches_path, **arrays)

    study_rows = []
    study_arrays = {name: [] for name in arrays}
    for study in ("MTBLS135", "MTBLS3316"):
        for contrast in CONTRASTS:
            block = panel_manifest.loc[
                panel_manifest.study_id.eq(study) & panel_manifest.contrast_id.eq(contrast)
            ]
            indices = block.sketch_row.to_numpy(int)
            study_rows.append({
                "sketch_row": len(study_rows), "study_id": study,
                "contrast_id": contrast, "n_entities_all": 40,
                "n_entities_known": 0, "n_entities_unresolved": 0,
            })
            for name, values in arrays.items():
                study_arrays[name].append(values[indices].mean(axis=0))
    study_manifest = pd.DataFrame(study_rows)
    study_manifest_path = tmp_path / "study_manifest.tsv"
    study_manifest.to_csv(study_manifest_path, sep="\t", index=False)
    study_sketches_path = tmp_path / "study_sketches.npz"
    np.savez_compressed(
        study_sketches_path,
        **{name: np.asarray(values, np.float32) for name, values in study_arrays.items()},
    )

    query_dir = tmp_path / "query"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("query_factorial_hfd_response_index.py")),
        "--manifest", str(study_manifest_path), "--sketches", str(study_sketches_path),
        "--output-dir", str(query_dir), "--minimum-entities", "12",
    ], check=True, capture_output=True, text=True)
    jackknife_dir = tmp_path / "jackknife"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("audit_factorial_hfd_panel_jackknife.py")),
        "--panel-manifest", str(panel_manifest_path), "--panel-sketches", str(panel_sketches_path),
        "--output-dir", str(jackknife_dir), "--minimum-entities", "12",
    ], check=True, capture_output=True, text=True)

    null_path = tmp_path / "null.npz"
    np.savez_compressed(
        null_path,
        query_sketch_row=study_manifest.sketch_row.to_numpy(int),
        max_candidate_score=np.zeros((len(study_manifest), 99), dtype=np.float32),
    )
    adjudication_dir = tmp_path / "adjudication"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("evaluate_factorial_hfd_retrieval.py")),
        "--scores", str(query_dir / "retrieval_scores.tsv.gz"),
        "--null", str(null_path),
        "--jackknife", str(jackknife_dir / "single_panel_deletion_scores.tsv.gz"),
        "--output-dir", str(adjudication_dir),
    ], check=True, capture_output=True, text=True)
    report = json.loads((adjudication_dir / "report.json").read_text(encoding="utf-8"))
    assert report["rank_gate_pass"]
    assert report["max_candidate_null_gate_pass"]
    assert report["single_panel_deletion_gate_pass"]
    assert report["full_primary_pass"]
    adjudication = pd.read_csv(adjudication_dir / "primary_reciprocal_adjudication.tsv", sep="\t")
    assert (adjudication.max_candidate_permutation_p == 0.01).all()
    assert (adjudication.single_panel_deletion_pass_fraction == 1.0).all()


def test_response_assignment_null_is_truth_blind_and_candidate_maxed(tmp_path: Path) -> None:
    registry = pd.DataFrame([
        {"panel_id": "P1", "dataset_accession": "S1", "source_family": "F1"},
        {"panel_id": "P2", "dataset_accession": "S2", "source_family": "F2"},
    ])
    registry_path = tmp_path / "registry.tsv"
    registry.to_csv(registry_path, sep="\t", index=False)
    effects_root = tmp_path / "effects"
    embeddings_root = tmp_path / "embeddings"
    for panel_id in ("P1", "P2"):
        effect_dir = effects_root / panel_id
        embedding_dir = embeddings_root / panel_id
        effect_dir.mkdir(parents=True)
        embedding_dir.mkdir(parents=True)
        rows = []
        for contrast in CONTRASTS:
            rows.extend([
                {"anchor_id": "A1", "contrast_type": contrast, "beta": 1.0,
                 "se_hc3": 0.5, "unit_prevalence": 1.0, "passes_effect_qc": True},
                {"anchor_id": "A2", "contrast_type": contrast, "beta": -0.5,
                 "se_hc3": 0.5, "unit_prevalence": 0.8, "passes_effect_qc": True},
            ])
        pd.DataFrame(rows).to_csv(
            effect_dir / "factorial_contrast_effects.tsv.gz", sep="\t", index=False, compression="gzip"
        )
        np.savez_compressed(
            embedding_dir / "anchor_embeddings.npz",
            anchor_id=np.asarray(["A1", "A2"]),
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32),
        )
    manifest_rows = []
    for study in ("S1", "S2"):
        for contrast in CONTRASTS:
            manifest_rows.append({
                "sketch_row": len(manifest_rows), "study_id": study,
                "contrast_id": contrast, "assay_panels": 1,
            })
    manifest_path = tmp_path / "study.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    output = tmp_path / "null"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("calibrate_factorial_hfd_response_null.py")),
        "--study-manifest", str(manifest_path), "--panel-registry", str(registry_path),
        "--effects-root", str(effects_root), "--embeddings-root", str(embeddings_root),
        "--output-dir", str(output), "--permutations", "99", "--seed", "7",
    ], check=True, capture_output=True, text=True)
    with np.load(output / "response_assignment_null.npz") as packed:
        assert packed["max_candidate_score"].shape == (10, 99)
        assert packed["pair_scores"].shape == (10, 10, 99)
        assert np.isnan(packed["pair_scores"][0, 1]).all()  # same-study candidates are excluded
        assert np.isfinite(packed["pair_scores"][0, 5]).all()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["truth_opened"] is False

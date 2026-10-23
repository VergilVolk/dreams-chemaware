import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("query_factorial_hfd_local_directed_response.py")
SPEC = importlib.util.spec_from_file_location("local_response", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mutual_top1_returns_only_reciprocal_pairs():
    left = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    li, ri, similarity = MODULE.mutual_top1(left, right, 0.70, block_size=1)
    assert list(zip(li.tolist(), ri.tolist())) == [(0, 0), (1, 1)]
    assert np.allclose(similarity, 1.0)


def test_local_score_preserves_direction_and_penalizes_missing_response_mass():
    left_response = np.asarray([0.9, -0.8, 0.7])
    right_same = np.asarray([0.8, -0.9, 0.6])
    right_opposed = -right_same
    edge = np.asarray([0, 1])
    similarity = np.asarray([0.95, 0.90])
    same = MODULE.local_response_statistic(
        left_response, right_same, edge, edge, similarity, minimum_edges=2,
    )
    opposed = MODULE.local_response_statistic(
        left_response, right_opposed, edge, edge, similarity, minimum_edges=2,
    )
    assert same["answerable"] and opposed["answerable"]
    assert 0 < same["score"] < 1
    assert -1 < opposed["score"] < 0
    assert np.isclose(same["score"], -opposed["score"])
    assert same["left_coverage"] < 1.0


def test_minimum_edge_rule_abstains_instead_of_inventing_support():
    result = MODULE.local_response_statistic(
        np.asarray([0.8]), np.asarray([0.7]),
        np.asarray([0]), np.asarray([0]), np.asarray([0.9]), minimum_edges=2,
    )
    assert not result["answerable"]
    assert np.isnan(result["score"])


def test_end_to_end_keeps_identity_and_relation_tracks_separate(tmp_path: Path):
    studies = ("MTBLS135", "MTBLS3316", "MTBLS2085")
    jobs, registry, identity_rows = [], [], []
    effects_root = tmp_path / "effects"
    embeddings_root = tmp_path / "embeddings"
    families_root = tmp_path / "families"
    panel_root = tmp_path / "panels"
    anchors = ("a", "b", "c", "d")
    embedding = np.eye(4, dtype=np.float32)
    for job_index, study in enumerate(studies):
        panel_id = f"{study}_POS"
        panel_object = panel_root / panel_id
        panel_object.mkdir(parents=True)
        (effects_root / panel_id).mkdir(parents=True)
        (embeddings_root / panel_id).mkdir(parents=True)
        (families_root / panel_id).mkdir(parents=True)
        jobs.append({
            "job_index": job_index, "panel_id": panel_id,
            "panel_object": str(panel_object),
        })
        registry.append({
            "panel_id": panel_id, "dataset_accession": study,
            "source_family": study, "assay_file": f"{study}_POS.mzML",
        })
        mz = [100.0, 200.0 if study != "MTBLS3316" else 250.0, 300.0, 400.0]
        pd.DataFrame({"anchor_id": anchors, "precursor_mz": mz}).to_csv(
            panel_object / "anchor_manifest.tsv", sep="\t", index=False,
        )
        pd.DataFrame({
            "panel_id": panel_id, "anchor_id": anchors,
            "family_id": [f"f{x}" for x in anchors], "representative": True,
        }).to_csv(
            families_root / panel_id / "ion_family_membership.tsv", sep="\t", index=False,
        )
        np.savez_compressed(
            embeddings_root / panel_id / "anchor_embeddings.npz",
            anchor_id=np.asarray(anchors), embeddings=embedding,
        )
        rows = []
        for contrast_index, contrast in enumerate(MODULE.CONTRAST_ORDER):
            if contrast == "genotype_by_hfd":
                beta = np.asarray([1.0, -1.0, 0.8, -0.8])
                if study == "MTBLS2085":
                    beta = -beta
            else:
                beta = np.roll(np.asarray([0.7, 0.2, -0.3, -0.8]), contrast_index)
            for anchor, value in zip(anchors, beta):
                rows.append({
                    "anchor_id": anchor, "contrast_type": contrast,
                    "beta": value, "se_hc3": 0.25, "unit_prevalence": 1.0,
                    "passes_effect_qc": True,
                })
        pd.DataFrame(rows).to_csv(
            effects_root / panel_id / "factorial_contrast_effects.tsv.gz",
            sep="\t", index=False, compression="gzip",
        )
        identity_rows.extend([
            {"panel_id": panel_id, "anchor_id": anchor,
             "identity_status": "library_supported" if anchor in {"a", "b"} else "unresolved_entity"}
            for anchor in anchors
        ])
    jobs_path, registry_path, identity_path = (
        tmp_path / "jobs.tsv", tmp_path / "registry.tsv", tmp_path / "identity.tsv",
    )
    pd.DataFrame(jobs).to_csv(jobs_path, sep="\t", index=False)
    pd.DataFrame(registry).to_csv(registry_path, sep="\t", index=False)
    pd.DataFrame(identity_rows).to_csv(identity_path, sep="\t", index=False)
    output = tmp_path / "local"
    subprocess.run([
        sys.executable, str(MODULE_PATH),
        "--panel-jobs", str(jobs_path), "--panel-registry", str(registry_path),
        "--effects-root", str(effects_root), "--embeddings-root", str(embeddings_root),
        "--families-root", str(families_root), "--identity-table", str(identity_path),
        "--output-dir", str(output), "--minimum-local-edges", "1",
        "--minimum-track-edges", "1", "--block-size", "2",
    ], check=True, capture_output=True, text=True)
    scores = pd.read_csv(output / "local_directed_retrieval_scores.tsv.gz", sep="\t")
    assert len(scores) == 15 * 10 * 3
    reciprocal = scores.loc[
        scores.query_study_id.eq("MTBLS135")
        & scores.query_contrast_id.eq("genotype_by_hfd")
        & scores.target_study_id.eq("MTBLS3316")
        & scores.target_contrast_id.eq("genotype_by_hfd")
        & scores.evidence_world.eq("all")
    ].iloc[0]
    assert reciprocal.local_all_score > 0
    assert reciprocal.identity_compatible_edges == 3
    assert reciprocal.chemical_relation_edges == 1
    edges = pd.read_csv(output / "local_edge_contributions.tsv.gz", sep="\t")
    assert set(edges.evidence_track) == {"identity_compatible", "chemical_relation"}

    mean = scores[[
        "query_study_id", "query_contrast_id", "target_study_id",
        "target_contrast_id", "evidence_world",
    ]].copy()
    mean["answerable"] = scores.local_all_score.notna()
    mean["signed_response_score"] = scores.local_all_score
    mean["rank_signed_response_score"] = mean.groupby([
        "query_study_id", "query_contrast_id", "evidence_world",
    ])["signed_response_score"].rank(ascending=False, method="min")
    mean_path = tmp_path / "mean.tsv.gz"
    mean.to_csv(mean_path, sep="\t", index=False, compression="gzip")
    adjudication = tmp_path / "adjudication"
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("evaluate_factorial_hfd_local_response.py")),
        "--local-scores", str(output / "local_directed_retrieval_scores.tsv.gz"),
        "--edge-contributions", str(output / "local_edge_contributions.tsv.gz"),
        "--mean-scores", str(mean_path), "--output-dir", str(adjudication),
    ], check=True, capture_output=True, text=True)
    compared = pd.read_csv(
        adjudication / "local_vs_mean_reciprocal_results.tsv", sep="\t",
    )
    assert len(compared) == 2 * 3 * 3
    assert (adjudication / "top_supporting_local_edges.tsv").stat().st_size > 0

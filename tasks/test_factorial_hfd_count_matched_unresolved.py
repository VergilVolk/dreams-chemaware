import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = (
    "diet_reference", "diet_modified", "genotype_control",
    "genotype_hfd", "genotype_by_hfd",
)
STUDIES = ("MTBLS135", "MTBLS3316", "MTBLS2085")


def test_count_matched_builder_is_truth_blind_and_complete(tmp_path: Path) -> None:
    jobs, registry, identity = [], [], []
    effects_root, embeddings_root, families_root = [tmp_path / name for name in ("effects", "embeddings", "families")]
    for index, study in enumerate(STUDIES):
        panel = f"P{index}"
        jobs.append({"job_index": index, "panel_id": panel})
        registry.append({"panel_id": panel, "dataset_accession": study})
        for root in (effects_root, embeddings_root, families_root):
            (root / panel).mkdir(parents=True)
        anchors = np.asarray(["k1", "k2", "u1", "u2"])
        rows = []
        for contrast_index, contrast in enumerate(CONTRASTS):
            for anchor_index, anchor in enumerate(anchors):
                rows.append({
                    "anchor_id": anchor, "contrast_type": contrast,
                    "beta": float((anchor_index + 1) * (contrast_index + 1)),
                    "se_hc3": 1.0, "unit_prevalence": 1.0, "passes_effect_qc": True,
                })
        pd.DataFrame(rows).to_csv(
            effects_root / panel / "factorial_contrast_effects.tsv.gz",
            sep="\t", index=False, compression="gzip",
        )
        np.savez_compressed(
            embeddings_root / panel / "anchor_embeddings.npz",
            anchor_id=anchors, embeddings=np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32),
        )
        pd.DataFrame({
            "anchor_id": anchors, "representative": True,
        }).to_csv(families_root / panel / "ion_family_membership.tsv", sep="\t", index=False)
        identity.extend([
            {"panel_id": panel, "anchor_id": anchor,
             "identity_status": "library_supported" if anchor.startswith("k") else "unresolved_entity"}
            for anchor in anchors
        ])
    jobs_path, registry_path, identity_path = [tmp_path / name for name in ("jobs.tsv", "registry.tsv", "identity.tsv")]
    pd.DataFrame(jobs).to_csv(jobs_path, sep="\t", index=False)
    pd.DataFrame(registry).to_csv(registry_path, sep="\t", index=False)
    pd.DataFrame(identity).to_csv(identity_path, sep="\t", index=False)
    output = tmp_path / "output"
    subprocess.run([
        sys.executable, str(ROOT / "tasks/build_factorial_hfd_count_matched_unresolved.py"),
        "--panel-jobs", str(jobs_path), "--panel-registry", str(registry_path),
        "--effects-root", str(effects_root), "--embeddings-root", str(embeddings_root),
        "--families-root", str(families_root), "--identity-table", str(identity_path),
        "--output-dir", str(output), "--resamples", "2", "--minimum-entities", "1",
    ], check=True, cwd=ROOT)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(output / "count_matched_retrieval_scores.tsv.gz", sep="\t")
    assert report["truth_opened"] is False
    assert report["denominators"]["response_objects_per_resample"] == 15
    assert len(scores) == 2 * 15 * 10


def test_count_matched_adjudication_requires_joint_80_percent(tmp_path: Path) -> None:
    rows = []
    for resample in range(16):
        for query, target in (("MTBLS135", "MTBLS3316"), ("MTBLS3316", "MTBLS135")):
            other = "MTBLS2085"
            candidates = [(target, contrast) for contrast in CONTRASTS] + [(other, contrast) for contrast in CONTRASTS]
            for rank, (target_study, contrast) in enumerate(candidates, start=1):
                expected = target_study == target and contrast == "genotype_by_hfd"
                rows.append({
                    "resample": resample, "query_study_id": query,
                    "query_contrast_id": "genotype_by_hfd", "target_study_id": target_study,
                    "target_contrast_id": contrast, "answerable": True,
                    "signed_response_score": 2.0 if expected else 1.0 - rank / 100,
                    "rank_signed_response_score": 1 if expected else rank + 1,
                })
    scores_path = tmp_path / "scores.tsv.gz"
    pd.DataFrame(rows).to_csv(scores_path, sep="\t", index=False, compression="gzip")
    known_rows = []
    for query, target in (("MTBLS135", "MTBLS3316"), ("MTBLS3316", "MTBLS135")):
        known_rows.append({
            "evidence_world": "known", "query_study_id": query,
            "query_contrast_id": "genotype_by_hfd", "target_study_id": target,
            "target_contrast_id": "genotype_by_hfd", "answerable": True,
            "rank_signed_response_score": 2,
        })
    known_path = tmp_path / "known.tsv.gz"
    pd.DataFrame(known_rows).to_csv(known_path, sep="\t", index=False, compression="gzip")
    output = tmp_path / "adjudication"
    subprocess.run([
        sys.executable, str(ROOT / "tasks/evaluate_factorial_hfd_count_matched_unresolved.py"),
        "--scores", str(scores_path), "--known-scores", str(known_path),
        "--output-dir", str(output),
    ], check=True, cwd=ROOT)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["denominators"]["required_joint_passes"] == 13
    assert report["result"]["joint_reciprocal_rank_gate_passes"] == 16
    assert report["result"]["passes_80_percent_sensitivity"] is True

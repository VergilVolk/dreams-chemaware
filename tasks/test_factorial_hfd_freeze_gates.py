import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_download_verifier_requires_complete_exact_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "sample.mzML"
    raw.write_bytes(b"<mzML>" + b" " * 2048 + b"</mzML>")
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{
        "panel_id": "p1", "dataset_accession": "S1", "sample_id": "u1",
        "local_path": raw.as_posix(), "deposited_bytes": raw.stat().st_size,
    }]).to_csv(manifest, sep="\t", index=False)
    reports = tmp_path / "reports"
    write_json(reports / "batch_000.json", {
        "records": [{
            "local_path": raw.as_posix(), "observed_bytes": raw.stat().st_size,
            "sha256": file_sha256(raw), "status": "downloaded_valid",
        }]
    })
    output = tmp_path / "verified"
    subprocess.run([
        sys.executable, str(ROOT / "tasks/verify_factorial_hfd_download.py"),
        "--manifest", str(manifest), "--report-dir", str(reports),
        "--output-dir", str(output),
    ], check=True, cwd=ROOT)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FROZEN_DOWNLOAD_COMPLETE"
    assert report["denominators"]["sample_files"] == 1


def test_pretruth_freeze_accepts_complete_contract(tmp_path: Path) -> None:
    index, retrieval, null, jackknife = [tmp_path / name for name in ("index", "retrieval", "null", "jackknife")]
    for path in (index, retrieval, null, jackknife):
        path.mkdir()
    identity = tmp_path / "identity.tsv"
    identity.write_text("panel_id\tanchor_id\tidentity_status\np\ta\tlibrary_supported\n", encoding="utf-8")
    write_json(index / "report.json", {
        "status": "development_pretruth_factorial_response_index",
        "identity_table_sha256": file_sha256(identity),
    })
    write_json(retrieval / "report.json", {"status": "pretruth_scores_frozen", "truth_opened": False})
    write_json(null / "report.json", {"status": "pretruth_max_candidate_null_frozen", "truth_opened": False, "permutations": 1999})
    write_json(jackknife / "report.json", {"status": "pretruth_single_panel_deletion_scores_frozen", "truth_opened": False, "response_objects": 15, "scored_rows": 123})

    manifest = pd.DataFrame({
        "study_id": np.repeat(["A", "B", "C"], 5),
        "contrast_id": [f"c{i}" for _ in range(3) for i in range(5)],
        "identity_partition_available": True,
        "n_entities_known": 2,
        "n_entities_unresolved": 3,
    })
    manifest.to_csv(index / "study_contrast_manifest.tsv", sep="\t", index=False)
    pd.DataFrame({"x": [1]}).to_csv(index / "panel_contrast_manifest.tsv", sep="\t", index=False)
    np.savez_compressed(index / "study_contrast_sketches.npz", x=np.zeros((15, 2)))
    np.savez_compressed(index / "panel_contrast_sketches.npz", x=np.zeros((1, 2)))
    scores = pd.DataFrame({"evidence_world": np.repeat(["all", "known", "unresolved"], 150)})
    scores.to_csv(retrieval / "retrieval_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    np.savez_compressed(null / "response_assignment_null.npz", x=np.zeros(1))
    pd.DataFrame({"x": [1]}).to_csv(jackknife / "single_panel_deletion_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    output = tmp_path / "frozen"
    subprocess.run([
        sys.executable, str(ROOT / "tasks/freeze_factorial_hfd_pretruth.py"),
        "--index-dir", str(index), "--retrieval-dir", str(retrieval),
        "--null-dir", str(null), "--jackknife-dir", str(jackknife),
        "--identity-table", str(identity),
        "--output-dir", str(output),
    ], check=True, cwd=ROOT)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PRETRUTH_ARTIFACTS_HASH_FROZEN"
    assert report["truth_opened"] is False
    assert report["denominators"]["retrieval_rows"] == 450

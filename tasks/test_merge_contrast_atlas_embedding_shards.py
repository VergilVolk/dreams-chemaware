from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import merge_contrast_atlas_embedding_shards as merger


def _write_fixture(root: Path) -> tuple[Path, Path]:
    collection = root / "collection"
    shards = root / "shards"
    collection.mkdir()
    entities = pd.DataFrame({
        "study_id": ["A", "A", "B"],
        "entity_id": ["A:1", "A:2", "B:1"],
    })
    entities.to_csv(
        collection / "response_entities.tsv.gz",
        sep="\t", index=False, compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame({"study_id": ["A", "B"]}).to_csv(
        collection / "source_manifest.tsv", sep="\t", index=False,
    )
    (collection / "report.json").write_text("{}\n", encoding="utf-8")
    response_sha = merger.sha256(collection / "response_entities.tsv.gz")

    for study, ids in (("A", ["A:1", "A:2"]), ("B", ["B:1"])):
        shard = shards / study
        shard.mkdir(parents=True)
        vectors = np.zeros((len(ids), 3), dtype=np.float32)
        for index in range(len(ids)):
            vectors[index, index % 3] = 1.0
        archive = shard / f"{study}.npz"
        np.savez_compressed(archive, entity_id=np.asarray(ids), embeddings=vectors)
        pd.DataFrame({
            "study_id": [study],
            "embedding_path": [str(archive)],
            "method_id": ["official"],
        }).to_csv(shard / "embedding_manifest.tsv", sep="\t", index=False)
        report = {
            "schema": merger.SCHEMA,
            "status": merger.STATUS,
            "data_role": "development_consumed",
            "model_kind": "official",
            "checkpoint_kind": "official_embedding_slim",
            "leakage_status": "phenotype blind",
            "preprocessing": {"n_highest_peaks": 100},
            "denominators": {
                "independent_studies": 1,
                "response_entities_encoded": len(ids),
                "embedding_dimensions": 3,
            },
            "sources": [{
                "study_id": study,
                "response_entities_encoded": len(ids),
                "embedding_dimensions": 3,
                "embedding_npz_sha256": merger.sha256(archive),
            }],
            "provenance": {
                "contract_sha256": "contract",
                "contract_lineage_sha256": {"contract.json": "lineage"},
                "response_entities_sha256": response_sha,
                "raw_checkpoint_sha256": "raw",
                "encoder_checkpoint_sha256": "encoder",
                "encoder_model_state_sha256": "state",
                "implementation_sha256": "implementation",
            },
        }
        (shard / "report.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8",
        )
    return collection, shards


def test_merge_validates_and_preserves_all_studies(tmp_path, monkeypatch):
    collection, shards = _write_fixture(tmp_path)
    output = tmp_path / "merged"
    monkeypatch.setattr(sys, "argv", [
        "merge_contrast_atlas_embedding_shards.py",
        "--collection", str(collection),
        "--shards-root", str(shards),
        "--output-dir", str(output),
    ])
    merger.main()
    manifest = pd.read_csv(output / "embedding_manifest.tsv", sep="\t", dtype=str)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert list(manifest.study_id) == ["A", "B"]
    assert report["denominators"] == {
        "independent_studies": 2,
        "response_entities_encoded": 3,
        "embedding_dimensions": 3,
    }
    assert report["status"].endswith("SHARDS_MERGED")


def test_merge_rejects_entity_inventory_drift(tmp_path, monkeypatch):
    collection, shards = _write_fixture(tmp_path)
    archive = shards / "B" / "B.npz"
    np.savez_compressed(
        archive,
        entity_id=np.asarray(["B:wrong"]),
        embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    report_path = shards / "B" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["sources"][0]["embedding_npz_sha256"] = merger.sha256(archive)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "merge_contrast_atlas_embedding_shards.py",
        "--collection", str(collection),
        "--shards-root", str(shards),
        "--output-dir", str(tmp_path / "merged"),
    ])
    try:
        merger.main()
    except RuntimeError as error:
        assert "registry mismatch" in str(error)
    else:
        raise AssertionError("entity inventory drift should fail closed")

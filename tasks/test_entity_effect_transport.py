#!/usr/bin/env python
"""Contract tests for analyze_entity_effect_transport.py."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks/analyze_entity_effect_transport.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(root: Path, contaminated_link: bool = False) -> dict[str, Path]:
    paths = {name: root / f"{name}.csv" for name in (
        "discovery_manifest", "validation_manifest", "links",
        "discovery_effects", "validation_effects",
    )}
    pd.DataFrame([
        {"entity_id": "D1", "identity_status": "confirmed_known", "qc_pass": True},
        {"entity_id": "D2", "identity_status": "unresolved_entity", "qc_pass": True},
        {"entity_id": "D3", "identity_status": "unresolved_entity", "qc_pass": True},
        {"entity_id": "D4", "identity_status": "library_supported", "qc_pass": True},
    ]).to_csv(paths["discovery_manifest"], index=False)
    pd.DataFrame([
        {"entity_id": "V1", "qc_pass": True},
        {"entity_id": "V2", "qc_pass": True},
        {"entity_id": "V3", "qc_pass": True},
        {"entity_id": "V4", "qc_pass": True},
    ]).to_csv(paths["validation_manifest"], index=False)
    links = pd.DataFrame([
        # Strong, unique correspondence and same-direction effect.
        {"discovery_entity_id": "D1", "validation_entity_id": "V1", "match_p": 1e-8, "match_weight": 1.0},
        # Ambiguous candidates: one is same-direction, but weights impose the selection penalty.
        {"discovery_entity_id": "D2", "validation_entity_id": "V2", "match_p": 1e-7, "match_weight": 0.5},
        {"discovery_entity_id": "D2", "validation_entity_id": "V3", "match_p": 1e-7, "match_weight": 0.5},
        # Strong link but reversed validation effect.
        {"discovery_entity_id": "D3", "validation_entity_id": "V4", "match_p": 1e-8, "match_weight": 1.0},
    ])
    if contaminated_link:
        links["phenotype"] = "case"
    links.to_csv(paths["links"], index=False)
    pd.DataFrame([
        {"entity_id": "D1", "beta": 1.0, "se": 0.05},
        {"entity_id": "D2", "beta": 0.8, "se": 0.06},
        {"entity_id": "D3", "beta": 0.9, "se": 0.05},
        # No link: must remain non-replicated.
        {"entity_id": "D4", "beta": 1.1, "se": 0.05},
    ]).to_csv(paths["discovery_effects"], index=False)
    pd.DataFrame([
        {"entity_id": "V1", "beta": 0.9, "se": 0.05},
        {"entity_id": "V2", "beta": 0.7, "se": 0.05},
        {"entity_id": "V3", "beta": -0.8, "se": 0.05},
        {"entity_id": "V4", "beta": -0.9, "se": 0.05},
    ]).to_csv(paths["validation_effects"], index=False)
    freeze = root / "link_freeze.json"
    freeze.write_text(json.dumps({
        "phenotype_blind": True,
        "validation_effects_unseen": True,
        "discovery_manifest_sha256": sha256(paths["discovery_manifest"]),
        "validation_manifest_sha256": sha256(paths["validation_manifest"]),
        "links_sha256": sha256(paths["links"]),
        "note": "synthetic contract fixture",
    }), encoding="utf-8")
    paths["link_freeze"] = freeze
    return paths


def command(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable, str(SCRIPT),
        "--discovery-manifest", str(paths["discovery_manifest"]),
        "--validation-manifest", str(paths["validation_manifest"]),
        "--links", str(paths["links"]),
        "--link-freeze", str(paths["link_freeze"]),
        "--discovery-effects", str(paths["discovery_effects"]),
        "--validation-effects", str(paths["validation_effects"]),
        "--effect-scale", "paired_log2fc",
        "--min-abs-effect", "0.2",
        "--analysis-role", "technical_replication",
        "--output-dir", str(output),
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="entity_effect_transport_") as raw:
        root = Path(raw)
        clean = root / "clean"
        clean.mkdir()
        paths = write_inputs(clean)
        output = root / "output"
        subprocess.run(command(paths, output), check=True, capture_output=True, text=True)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        entities = pd.read_csv(output / "entity_replication_results.csv").set_index("entity_id")
        pairs = pd.read_csv(output / "pair_replication_audit.csv")
        assert report["status"] == "ENTITY_EFFECT_TRANSPORT_COMPLETE"
        assert report["universe"]["discovery_qc_entities"] == 4
        assert bool(entities.loc["D1", "replicated_full"])
        assert bool(entities.loc["D2", "replicated_full"])
        assert not bool(entities.loc["D3", "replicated_full"])
        assert not bool(entities.loc["D4", "replicated_full"])
        assert entities.loc["D4", "link_state"] == "no_evaluable_link"
        d2 = pairs.loc[pairs["discovery_entity_id"].eq("D2")]
        assert len(d2) == 2 and abs(float(d2["match_weight"].sum()) - 1.0) < 1e-12
        assert report["outcomes"]["replicated_unresolved_in_full_universe"] == 1
        assert report["truth_and_leakage"]["links_declared_phenotype_blind"] is True
        assert report["truth_and_leakage"]["validation_effects_unseen_declared_in_freeze"] is True

        dirty = root / "dirty"
        dirty.mkdir()
        dirty_paths = write_inputs(dirty, contaminated_link=True)
        failed = subprocess.run(
            command(dirty_paths, root / "dirty_output"), capture_output=True, text=True,
        )
        assert failed.returncode != 0
        assert "phenotype/effect columns" in failed.stderr

        # Modifying a frozen input must invalidate the hash contract.
        paths["links"].write_text(paths["links"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
        failed_hash = subprocess.run(
            command(paths, root / "hash_output"), capture_output=True, text=True,
        )
        assert failed_hash.returncode != 0
        assert "hash mismatch" in failed_hash.stderr
    print("PASS: entity effect transport contracts")


if __name__ == "__main__":
    main()


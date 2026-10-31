#!/usr/bin/env python3
"""Fail-closed preflight for the frozen 16-source Contrast Atlas Full23 run.

This program performs no embedding or pair scoring.  It binds the exact v10
model-independent candidate axis, the truth-blind spectrum-source manifest,
and every frozen Full23 upstream artifact before GPU work can begin.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


AXIS_STATUS = "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN"
AXIS_SCHEMA = "contrast_atlas_model_independent_identity_axis_v1"
SOURCE_STATUS = "DEVELOPMENT_CONSUMED_SPECTRUM_SOURCE_MANIFEST_COMPLETE"
SOURCE_SCHEMA = "contrast_atlas_spectrum_source_manifest_v1"
EXPECTED_DENOMINATORS = {
    "studies": 16,
    "entities": 55_127,
    "query_target_source_blocks": 826_905,
    "nonempty_candidate_blocks": 141_332,
    "abstain_no_candidate_blocks": 685_573,
    "directed_candidate_pairs": 294_236,
}
EXPECTED_AXIS_OUTPUTS = {
    "observed_ion_registry.tsv.gz": "b47d11b81e4a479c51882b4adbfb1deaf345f15165e325132f2be63ba070fab9",
    "directed_complete_candidate_pairs.tsv.gz": "f7854880bba4431531ee95095360209f7f0a3c650ecb1612a2be0cb713416860",
    "query_target_source_blocks.tsv.gz": "4bee8470d24e7eb097131b4216cdd8e1d7066617df0dc819b4b81b8fd1448623",
    "study_observed_ion_coverage.tsv": "8e0e22937112fef893cfa4faa1ecc448b38138a1b173c25efb9621187a5fd82a",
}
EXPECTED_SOURCE_MANIFEST_SHA256 = "188e1b0387db85b4908e94ee8a8502d1c657082f9e20e19d7636fdab3f9a4a51"
EXPECTED_RELEASE_MANIFEST_SHA256 = "5c4c879de220f67e632c2f15c9d186dacd4d22d7f26eb2efd24e474ab510114e"
EXPECTED_RUNTIME_EQUIVALENCE_ADDENDUM_SHA256 = "9529b9ff64dedecd8a199a2ba5945507b30ab3842d761112f6c76ea086f3ffec"
EXPECTED_MS_ENTROPY_WHEEL_SHA256 = "f0a9a83359118f0fdb3260e1a6ba891ba51f76523bbfa0310e065c9ee5b9e7a7"
EXPECTED_FULL23_SHA256 = "7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717"
RELEASE_FROZEN_ENVEDA_CORE_SHA256 = "b8c86591b1314321ad113440d8796b3deefefa96a874474a520335e3eab46a79"
ACCEPTED_INFERENCE_EQUIVALENT_ENVEDA_CORE_SHA256 = "f67fb23867123ca3a6d6a413fa793ed7db0ae4ec43bc8f23039bed680402a33a"
EXPECTED_IMPLEMENTATIONS = {
    "tasks/score_crossstudy_identity_multiencoder.py": "8575e2c8a691fbb4602790ce6dcca1bf2c1e6e4a709d3591b103a0f21ad54ee1",
    "tasks/encode_crossstudy_mgfs_checkpoint.py": "6220df6122e6dfdae90d746dd3ddcfa975a9f245e0b53996bcc7f3537fce3e29",
    "tasks/encode_pan_redu_panel_spectra.py": "57432c2e2974fdd50147564b4b131a83bed0467892e054a6a671e469fd4ff00b",
    "tasks/prepare_official_embedding_checkpoint.py": "dd40738097f4ae0ece95fe8d912aa4dd974ce66f85684856362041b807bebcdc",
    "tasks/unified_additive_release.py": "52aa2130dbd290d4cc7e3c63c3edc463daf8c95eecd1c4b664fadba68c76a3ab",
    "tasks/enveda_unified_candidate_integrator_core.py": "f67fb23867123ca3a6d6a413fa793ed7db0ae4ec43bc8f23039bed680402a33a",
    "tasks/massspecgym_embedding_readout_core.py": "213bbd4b687b560c948247e6c325ab712fceaef558cb56cff00cf28c951a1833",
    "tasks/noise_gnps_article_spectral_scores.py": "ff2a8ad8a1ff0ec7edf2f882f5e201b34792a3db52993910cd8c7c787488caf2",
    "tasks/noise_msg_pair_evidence_core.py": "0eec2ee4ae5caca3a6e1a6eacf1d47618d9b2d3fb2a2f2a2de58c4e06a2c27b5",
    "tasks/encode_contrast_atlas_v10_axis_checkpoint.py": "7a214a4796b0a3b8aa1bde1930b2c4655a95bfad5d6804c80f69c7917921a404",
    "tasks/score_contrast_atlas_v10_complete_axis_full23.py": "ca17e796064e3a31e6eb18ea180e684d23842260d328138874e7f7ccb6e6e026",
    "tasks/package_contrast_atlas_v10_method_scores.py": "3d3878305800d529acc99f8b2d33693618d52890252ad31604a254e1ec54f6e6",
    "tasks/contrast_atlas_v10_method_pair_score_contract_v1.json": "e5e0e2a3f34fa58536c6da6c5aa57a13d58b9fda1520ab1cb47ed00904bedde9",
    "tasks/test_unified_additive_release.py": "d8c928afb32ef41225e431f10e205f19585facc2eded8b9bfdd2cd1bb4c1a79c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def parse_artifacts(items: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--artifact must use NAME=PATH")
        name, raw = item.split("=", 1)
        require(bool(name and raw and name not in result), f"invalid/duplicate artifact: {name!r}")
        result[name] = Path(raw)
    return result


def expected_artifacts(manifest: dict[str, Any]) -> dict[str, set[str]]:
    upstream = manifest["upstream_assets"]
    official = upstream["official_finetuned_dreams_comparator"]
    return {
        "architecture": {str(upstream["dreams_ssl_architecture"]["sha256"])},
        "official_dreams": {
            str(official["canonical_container_sha256"]),
            str(official["local_repacked_container_sha256"]),
        },
        "noise_v1": {str(upstream["noise_v1"]["sha256"])},
        "chemaware_stage1": {str(upstream["chemaware_stage1"]["sha256"])},
        "chemaware_phasea": {str(upstream["chemaware_phasea"]["sha256"])},
        "noise_chemaware_fullmsg": {str(upstream["noise_chemaware_fullmsg"]["sha256"])},
        "noise_msg_model": {str(upstream["noise_msg_fusion_stack_v1"]["sha256"])},
        "noise_msg_report": {str(upstream["noise_msg_fusion_stack_v1"]["report_sha256"])},
        "full23_v1": {EXPECTED_FULL23_SHA256},
    }


def validate(
    axis_dir: Path,
    source_manifest_path: Path,
    release_manifest_path: Path,
    runtime_equivalence_addendum_path: Path,
    runtime_wheel_path: Path,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    axis_report_path = axis_dir / "report.json"
    axis = read_json(axis_report_path)
    require(axis.get("schema") == AXIS_SCHEMA, "v10 candidate-axis schema drifted")
    require(axis.get("status") == AXIS_STATUS, "v10 candidate axis is not frozen")
    require(axis.get("data_role") == "development_consumed", "v10 data role drifted")
    for key, expected in EXPECTED_DENOMINATORS.items():
        require(int(axis["denominators"].get(key, -1)) == expected, f"axis denominator drifted: {key}")
    for name, expected_hash in EXPECTED_AXIS_OUTPUTS.items():
        path = axis_dir / name
        require(path.is_file(), f"frozen axis output is missing: {path}")
        require(axis["outputs"].get(name) == expected_hash, f"axis-declared hash drifted: {name}")
        require(sha256(path) == expected_hash, f"frozen axis byte hash drifted: {name}")

    source_manifest = read_json(source_manifest_path)
    require(sha256(source_manifest_path) == EXPECTED_SOURCE_MANIFEST_SHA256, "source manifest hash drifted")
    require(source_manifest.get("schema") == SOURCE_SCHEMA, "source manifest schema drifted")
    require(source_manifest.get("status") == SOURCE_STATUS, "source manifest is not complete")
    require(source_manifest.get("data_role") == "development_consumed", "source data role drifted")
    require(int(source_manifest["denominators"].get("studies", -1)) == 16, "source count drifted")
    require(int(source_manifest["denominators"].get("study_entities", -1)) == 55_127, "source entity count drifted")
    axis_mgfs = axis["provenance"]["mgf_sha256_by_study"]
    observed_sources: dict[str, dict[str, Any]] = {}
    source_bytes = 0
    for source in source_manifest.get("sources", []):
        study = str(source.get("study_id", ""))
        require(study and study not in observed_sources, f"duplicate/empty source study: {study!r}")
        path = Path(str(source.get("mgf_path", "")).replace("\\", "/"))
        require(path.is_file(), f"source MGF is missing: {study}: {path}")
        observed_hash = sha256(path)
        require(observed_hash == str(source.get("mgf_sha256", "")), f"source manifest MGF hash drifted: {study}")
        require(observed_hash == str(axis_mgfs.get(study, "")), f"axis/source MGF hash differs: {study}")
        source_bytes += path.stat().st_size
        observed_sources[study] = {"path": str(path), "sha256": observed_hash, "bytes": path.stat().st_size}
    require(set(observed_sources) == set(axis_mgfs), "axis and source-manifest study sets differ")

    require(sha256(release_manifest_path) == EXPECTED_RELEASE_MANIFEST_SHA256, "Full23 release manifest hash drifted")
    release = read_json(release_manifest_path)
    require(release.get("schema") == "dreams_unified_additive_full23_v1", "Full23 release manifest drifted")

    require(
        sha256(runtime_equivalence_addendum_path) == EXPECTED_RUNTIME_EQUIVALENCE_ADDENDUM_SHA256,
        "Full23 runtime-equivalence addendum hash drifted",
    )
    runtime_equivalence = read_json(runtime_equivalence_addendum_path)
    require(
        runtime_equivalence.get("schema") == "unified_additive_full23_runtime_equivalence_addendum_v1"
        and runtime_equivalence.get("status") == "FROZEN_INFERENCE_EQUIVALENCE_ACCEPTED",
        "Full23 runtime-equivalence addendum is not accepted",
    )
    require(
        runtime_equivalence.get("release_manifest", {}).get("sha256") == EXPECTED_RELEASE_MANIFEST_SHA256,
        "runtime-equivalence addendum refers to a different release manifest",
    )
    component = runtime_equivalence.get("component", {})
    require(
        component.get("release_frozen_sha256") == RELEASE_FROZEN_ENVEDA_CORE_SHA256
        and component.get("accepted_runtime_sha256") == ACCEPTED_INFERENCE_EQUIVALENT_ENVEDA_CORE_SHA256,
        "runtime-equivalence component hashes drifted",
    )

    require(runtime_wheel_path.is_file(), f"pinned ms_entropy wheel is missing: {runtime_wheel_path}")
    require(sha256(runtime_wheel_path) == EXPECTED_MS_ENTROPY_WHEEL_SHA256, "pinned ms_entropy wheel hash drifted")
    required = expected_artifacts(release)
    require(set(artifacts) == set(required), f"artifact registry incomplete: required={sorted(required)}, supplied={sorted(artifacts)}")
    observed_artifacts: dict[str, dict[str, Any]] = {}
    for name, accepted_hashes in required.items():
        path = artifacts[name]
        require(path.is_file(), f"frozen artifact is missing: {name}: {path}")
        observed_hash = sha256(path)
        require(observed_hash in accepted_hashes, f"frozen artifact hash drifted: {name}: {observed_hash}")
        observed_artifacts[name] = {"path": str(path), "sha256": observed_hash, "bytes": path.stat().st_size}

    observed_implementations: dict[str, dict[str, object]] = {}
    for raw_path, expected_hash in EXPECTED_IMPLEMENTATIONS.items():
        path = Path(raw_path)
        require(path.is_file(), f"registered implementation is missing: {path}")
        observed_hash = sha256(path)
        require(observed_hash == expected_hash, f"registered implementation drifted: {path}: {observed_hash}")
        observed_implementations[raw_path] = {"sha256": observed_hash, "bytes": path.stat().st_size}

    require(importlib.metadata.version("ms_entropy") == "1.5.2", "Full23 requires ms_entropy==1.5.2")
    return {
        "schema": "contrast_atlas_v10_full23_preflight_v1",
        "status": "PASS_READY_FOR_GPU_EVIDENCE_GENERATION",
        "data_role": "development_consumed",
        "truth_blinding": {
            "response_beta_read": False,
            "response_se_read": False,
            "phenotype_read": False,
            "registered_relations_read": False,
            "identity_truth_read": False,
        },
        "axis": {
            "report": str(axis_report_path),
            "report_sha256": sha256(axis_report_path),
            "denominators": EXPECTED_DENOMINATORS,
            "outputs": EXPECTED_AXIS_OUTPUTS,
        },
        "sources": {"manifest": str(source_manifest_path), "manifest_sha256": sha256(source_manifest_path), "mgfs": observed_sources, "total_bytes": source_bytes},
        "release": {
            "manifest": str(release_manifest_path),
            "manifest_sha256": sha256(release_manifest_path),
            "runtime_equivalence_addendum": str(runtime_equivalence_addendum_path),
            "runtime_equivalence_addendum_sha256": sha256(runtime_equivalence_addendum_path),
            "release_frozen_enveda_core_sha256": RELEASE_FROZEN_ENVEDA_CORE_SHA256,
            "accepted_inference_equivalent_enveda_core_sha256": ACCEPTED_INFERENCE_EQUIVALENT_ENVEDA_CORE_SHA256,
        },
        "artifacts": observed_artifacts,
        "implementations": observed_implementations,
        "runtime": {
            "ms_entropy": "1.5.2",
            "wheel": str(runtime_wheel_path),
            "wheel_sha256": sha256(runtime_wheel_path),
        },
        "claim_boundary": [
            "Preflight proves byte identity and input completeness only; it is not a performance or biological result.",
            "The 294236-pair model-independent axis is scored in full; WSE-selected edges are not an input.",
            "ChemAware V2 remains the frozen unavailable no-op because candidate structures are absent.",
            "BioAware is not fabricated without an event-level sample-context handoff.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-equivalence-addendum", type=Path, required=True)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], help="Repeated NAME=PATH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(
        args.axis_dir,
        args.source_manifest,
        args.release_manifest,
        args.runtime_equivalence_addendum,
        args.runtime_wheel,
        parse_artifacts(args.artifact),
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

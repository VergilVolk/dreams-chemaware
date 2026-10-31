#!/usr/bin/env python3
"""Score the complete 16-source Atlas axis with frozen spectral experts.

The model-independent 294,236-row axis is the sole candidate universe.  No
response, phenotype, registered relation, WSE-selected edge, or identity truth
is read.  Missing evidence abstains an entire nonempty block in Full23.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from encode_pan_redu_panel_spectra import load_mgf
from noise_gnps_article_spectral_scores import (
    frozen_p2b_pair_features,
    prepare_spectrum,
    weighted_entropy_backend,
    weighted_entropy_similarity,
)
from score_crossstudy_identity_multiencoder import (
    MSG_FAMILY,
    apply_unified_model,
    frozen_p2b_block,
    matched_peaks,
    score_frozen_noise_msg,
)
from unified_additive_release import validate_checkpoint


ENCODERS = (
    "noise_v1",
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
)
METHODS = (
    "weighted_spectral_entropy",
    "official_dreams",
    "noise_v1",
    "p2b_noise_v1_frozen",
    MSG_FAMILY,
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
    "full23_v1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_paths(items: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{label} must use NAME=PATH")
        name, raw = item.split("=", 1)
        if not name or not raw or name in result:
            raise ValueError(f"invalid/duplicate {label}: {name!r}")
        result[name] = Path(raw)
    return result


def load_embeddings(
    paths: dict[str, Path], axis_report_sha: str, source_manifest_sha: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, object]]]:
    caches: dict[str, dict[str, np.ndarray]] = {}
    provenance: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        sidecar_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"embedding cache or sidecar is absent: {path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("status") != "CONTRAST_ATLAS_V10_FROZEN_ENCODER_EMBEDDINGS_COMPLETE"
            or sidecar.get("encoder_name") != name
            or sidecar.get("labels_or_responses_accessed") is not False
            or sidecar.get("inputs", {}).get("axis_report", {}).get("sha256") != axis_report_sha
            or sidecar.get("inputs", {}).get("source_manifest", {}).get("sha256") != source_manifest_sha
            or sidecar.get("output", {}).get("sha256") != sha256(path)
        ):
            raise RuntimeError(f"embedding cache provenance drifted: {name}")
        with np.load(path, allow_pickle=False) as body:
            ids = np.asarray(body["entity_id"]).astype("U")
            matrix = np.asarray(body["embeddings"], dtype=np.float32)
        if matrix.shape != (len(ids), 1024) or len(ids) != len(set(ids.tolist())):
            raise RuntimeError(f"invalid embedding cache: {name}")
        caches[name] = dict(zip(ids.tolist(), matrix, strict=True))
        provenance[name] = {
            "path": str(path), "sha256": sha256(path),
            "sidecar": str(sidecar_path), "sidecar_sha256": sha256(sidecar_path),
            "canonical_artifact": sidecar["inputs"]["canonical_artifact"],
            "encoded_entities": len(ids),
        }
    return caches, provenance


def load_spectra(
    registry: pd.DataFrame, source_manifest: dict[str, object], axis_mgf_hashes: dict[str, str],
) -> tuple[dict[str, tuple[float, np.ndarray]], dict[str, dict[str, object]]]:
    spectra: dict[str, tuple[float, np.ndarray]] = {}
    provenance: dict[str, dict[str, object]] = {}
    sources = {str(row["study_id"]): row for row in source_manifest["sources"]}
    if set(sources) != set(registry["study_id"].astype(str)):
        raise RuntimeError("source manifest and frozen entity registry studies differ")
    for study in sorted(sources):
        source = sources[study]
        path = Path(str(source["mgf_path"]).replace("\\", "/"))
        observed_hash = sha256(path)
        if observed_hash != str(source["mgf_sha256"]) or observed_hash != axis_mgf_hashes[study]:
            raise RuntimeError(f"frozen MGF provenance drifted: {study}")
        frame = registry.loc[registry["study_id"].astype(str).eq(study)]
        required = dict(zip(frame["raw_entity_id"].astype(str), frame["entity_id"].astype(str), strict=True))
        ids, records = load_mgf(path, minimum_peaks=0, skip_ineligible=False)
        found: set[str] = set()
        for raw_id, record in zip(ids, records, strict=True):
            entity_id = required.get(str(raw_id))
            if entity_id is None:
                continue
            if entity_id in found:
                raise RuntimeError(f"duplicate representative spectrum: {entity_id}")
            found.add(entity_id)
            spectra[entity_id] = (
                float(record["precursor_mz"]),
                prepare_spectrum(np.asarray(record["peaks"]), n_peaks=100),
            )
        missing = set(required.values()) - found
        if missing:
            raise RuntimeError(f"{study} misses {len(missing)} frozen representative spectra")
        provenance[study] = {"path": str(path), "sha256": observed_hash, "entities": len(found)}
    if len(spectra) != len(registry):
        raise RuntimeError("loaded spectrum denominator differs from frozen registry")
    return spectra, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--embedding", action="append", default=[], help="Repeated encoder NAME=NPZ")
    parser.add_argument("--official-embedding", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], help="Repeated method NAME=PATH")
    parser.add_argument("--msg-fusion-model", type=Path, required=True)
    parser.add_argument("--full23-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fragment_da <= 0:
        raise ValueError("fragment tolerance must be positive")

    axis_report_path = args.axis_dir / "report.json"
    registry_path = args.axis_dir / "observed_ion_registry.tsv.gz"
    pairs_path = args.axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    blocks_path = args.axis_dir / "query_target_source_blocks.tsv.gz"
    for path in (axis_report_path, registry_path, pairs_path, blocks_path, args.source_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    axis_report = json.loads(axis_report_path.read_text(encoding="utf-8"))
    if axis_report.get("status") != "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN":
        raise RuntimeError("refusing a non-frozen identity axis")
    denominators = axis_report["denominators"]
    if (
        int(denominators.get("studies", -1)) != 16
        or int(denominators.get("entities", -1)) != 55_127
        or int(denominators.get("query_target_source_blocks", -1)) != 826_905
        or int(denominators.get("directed_candidate_pairs", -1)) != 294_236
    ):
        raise RuntimeError("v10 complete-axis denominator drifted")
    for filename, path in (
        ("observed_ion_registry.tsv.gz", registry_path),
        ("directed_complete_candidate_pairs.tsv.gz", pairs_path),
        ("query_target_source_blocks.tsv.gz", blocks_path),
    ):
        if sha256(path) != axis_report["outputs"][filename]:
            raise RuntimeError(f"frozen axis hash drifted: {filename}")

    registry = pd.read_csv(registry_path, sep="\t", dtype=str)
    pairs = pd.read_csv(pairs_path, sep="\t", low_memory=False, dtype={
        "query_block_id": str, "directed_pair_id": str,
        "query_entity_id": str, "candidate_entity_id": str,
    })
    blocks = pd.read_csv(blocks_path, sep="\t", dtype={"query_block_id": str})
    starts = blocks["candidate_pair_offset_start"].to_numpy(np.int64)
    stops = blocks["candidate_pair_offset_end"].to_numpy(np.int64)
    if starts[0] != 0 or stops[-1] != len(pairs) or not np.array_equal(starts[1:], stops[:-1]):
        raise RuntimeError("frozen candidate blocks are not contiguous")
    query_ptr = np.concatenate((starts[:1], stops)).astype(np.int64)
    query_ids = blocks["query_block_id"].astype(str).to_numpy(dtype="U")
    if not np.array_equal(np.repeat(query_ids, np.diff(query_ptr)), pairs["query_block_id"].astype(str).to_numpy(dtype="U")):
        raise RuntimeError("candidate rows do not follow frozen block order")
    pairs["query_id"] = pairs["query_entity_id"].astype(str)
    pairs["candidate_id"] = pairs["candidate_entity_id"].astype(str)

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    spectra, spectrum_provenance = load_spectra(
        registry, source_manifest, axis_report["provenance"]["mgf_sha256_by_study"],
    )
    embedding_paths = parse_named_paths(args.embedding, "--embedding")
    if set(embedding_paths) != set(ENCODERS):
        raise RuntimeError(f"encoder registry incomplete: {sorted(embedding_paths)}")
    embedding_paths["official_dreams"] = args.official_embedding
    caches, embedding_provenance = load_embeddings(
        embedding_paths, sha256(axis_report_path), sha256(args.source_manifest),
    )

    for name in (*ENCODERS, "official_dreams"):
        values = np.full(len(pairs), np.nan, dtype=np.float32)
        cache = caches[name]
        for index, (query_id, candidate_id) in enumerate(zip(pairs["query_id"], pairs["candidate_id"], strict=True)):
            if query_id in cache and candidate_id in cache:
                values[index] = float(np.dot(cache[query_id], cache[candidate_id]))
        pairs[f"{name}_score"] = values

    wse = np.full(len(pairs), np.nan, dtype=np.float32)
    matched = np.zeros(len(pairs), dtype=np.int16)
    sqrt_cosine = np.full(len(pairs), np.nan, dtype=np.float32)
    entropy = np.full(len(pairs), np.nan, dtype=np.float32)
    neutral_loss = np.full(len(pairs), np.nan, dtype=np.float32)
    for index, row in enumerate(pairs.itertuples(index=False)):
        left_precursor, left = spectra[str(row.query_entity_id)]
        right_precursor, right = spectra[str(row.candidate_entity_id)]
        if left.shape[1] and right.shape[1]:
            wse[index] = weighted_entropy_similarity(left, right, args.fragment_da)
            matched[index] = matched_peaks(left.T, right.T, args.fragment_da)
            values = frozen_p2b_pair_features(
                left, left_precursor, right, right_precursor, tolerance=args.fragment_da,
            )
            sqrt_cosine[index], entropy[index], neutral_loss[index] = values
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"scored frozen spectral pairs: {index + 1}/{len(pairs)}", flush=True)
    pairs["WSE_score"] = wse
    pairs["WSE_available"] = np.isfinite(wse)
    pairs["WSE_matched_fragments"] = matched
    pairs["P2b_sqrt_cosine"] = sqrt_cosine
    pairs["P2b_entropy"] = entropy
    pairs["P2b_neutral_loss"] = neutral_loss

    p2b = np.full(len(pairs), np.nan, dtype=np.float32)
    noise = pairs["noise_v1_score"].to_numpy(np.float64)
    for start, stop in zip(query_ptr[:-1], query_ptr[1:]):
        start, stop = int(start), int(stop)
        if stop == start:
            continue
        complete = np.column_stack((
            noise[start:stop], sqrt_cosine[start:stop], entropy[start:stop], neutral_loss[start:stop],
        ))
        if np.isfinite(complete).all():
            p2b[start:stop] = frozen_p2b_block(
                noise[start:stop], sqrt_cosine[start:stop], entropy[start:stop], neutral_loss[start:stop],
            )
    pairs["P2b_score"] = p2b
    msg, msg_provenance = score_frozen_noise_msg(pairs, query_ptr, args.msg_fusion_model)
    full23_payload = validate_checkpoint(args.full23_checkpoint)
    full23, full23_provenance = apply_unified_model(
        pairs,
        {name: caches[name] for name in ENCODERS},
        args.full23_checkpoint,
        torch.device(args.device),
        query_ptr,
        query_ids,
        auxiliary_scores={MSG_FAMILY: msg},
        enveda_integrator_arm="additive_linear_control",
    )
    del full23_payload

    score_arrays = {
        "weighted_spectral_entropy": wse,
        "official_dreams": pairs["official_dreams_score"].to_numpy(np.float32),
        "noise_v1": pairs["noise_v1_score"].to_numpy(np.float32),
        "p2b_noise_v1_frozen": p2b,
        MSG_FAMILY: msg,
        "chemaware_stage1": pairs["chemaware_stage1_score"].to_numpy(np.float32),
        "chemaware_phasea": pairs["chemaware_phasea_score"].to_numpy(np.float32),
        "noise_chemaware_fullmsg": pairs["noise_chemaware_fullmsg_score"].to_numpy(np.float32),
        "full23_v1": full23,
    }
    artifact_paths = parse_named_paths(args.artifact, "--artifact")
    if set(artifact_paths) != set(METHODS):
        raise RuntimeError(f"method artifact registry incomplete: required={sorted(METHODS)}, supplied={sorted(artifact_paths)}")
    artifact_hashes = {name: sha256(path) for name, path in artifact_paths.items()}

    args.output_dir.mkdir(parents=True)
    bundle_path = args.output_dir / "candidate_score_bundle.npz"
    np.savez_compressed(
        bundle_path,
        schema=np.asarray("contrast_atlas_v10_complete_axis_full23_score_bundle_v1"),
        query_ptr=query_ptr,
        query_ids=query_ids,
        directed_pair_ids=pairs["directed_pair_id"].astype(str).to_numpy(dtype="U"),
        wse_matched_fragments=matched,
        **score_arrays,
    )
    report = {
        "schema": "contrast_atlas_v10_complete_axis_full23_scoring_v1",
        "status": "DEVELOPMENT_CONSUMED_CONTRAST_ATLAS_V10_COMPLETE_AXIS_FULL23_SCORED",
        "data_role": "development_consumed",
        "truth_blinding": {
            "response_beta_read": False, "response_se_read": False,
            "phenotype_read": False, "registered_relations_read": False,
            "identity_truth_read": False,
        },
        "denominators": {
            "studies": 16, "entities": 55_127, "query_blocks": len(blocks),
            "nonempty_blocks": int(np.sum(np.diff(query_ptr) > 0)),
            "directed_candidate_pairs": len(pairs),
            "available_scores_by_method": {name: int(np.isfinite(values).sum()) for name, values in score_arrays.items()},
        },
        "methods": list(METHODS),
        "method_artifacts": {name: {"path": str(artifact_paths[name]), "sha256": artifact_hashes[name]} for name in METHODS},
        "inputs": {
            "axis_report": {"path": str(axis_report_path), "sha256": sha256(axis_report_path)},
            "pair_axis": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "block_axis": {"path": str(blocks_path), "sha256": sha256(blocks_path)},
            "source_manifest": {"path": str(args.source_manifest), "sha256": sha256(args.source_manifest)},
            "embeddings": embedding_provenance,
            "spectra": spectrum_provenance,
        },
        "frozen_semantics": {
            "weighted_spectral_entropy_backend": weighted_entropy_backend(),
            "fragment_tolerance_da": args.fragment_da,
            "maximum_peaks": 100,
            "p2b": "registered one-reference Noise-V1 support-gated composite",
            "noise_msg": msg_provenance,
            "full23": full23_provenance,
            "chemaware_v2": "unavailable_exact_noop",
            "bioaware": "not_supplied_no_event_context_fabricated",
        },
        "outputs": {"candidate_score_bundle.npz": sha256(bundle_path)},
        "claim_boundary": [
            "All methods were scored on the complete model-independent 294236-pair axis; WSE-selected edges were not inherited.",
            "Raw scores are ranking evidence, not calibrated identity probabilities.",
            "Singleton blocks remain hypotheses and Full23 has no open-set identity acceptance rule.",
            "This development-consumed run cannot establish independent biological validation.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir), "available": report["denominators"]["available_scores_by_method"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

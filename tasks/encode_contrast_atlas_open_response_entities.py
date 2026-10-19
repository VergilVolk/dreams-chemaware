#!/usr/bin/env python3
"""Phenotype-blind frozen embeddings for an open response collection.

Only the study and entity identifier columns are read from the response
collection.  Effect sizes, standard errors, contrast labels, registered
relations, and retrieval truth are never loaded by this encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dreams_checkpoint_semantic_hash import slim_model_state_sha256  # noqa: E402
from build_contrast_atlas_open_response_collection import load_contract  # noqa: E402
from encode_ad_mgf_frozen_checkpoint import (  # noqa: E402
    MODEL_CHECKPOINT_SHA256,
    OFFICIAL_MODEL_STATE_SHA256,
    RAW_CHECKPOINT_SHA256,
)
from run_contrast_atlas_sparse_fragment_measure_baseline import (  # noqa: E402
    iter_mgf,
    normalise_entity_id,
    spectrum_id,
)


MINIMUM_FRAGMENT_PEAKS = 3
N_HIGHEST_PEAKS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_required_records(
    mgf_path: Path, required_ids: set[str]
) -> tuple[list[str], list[dict[str, object]], int]:
    ids: list[str] = []
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    total = 0
    for metadata, peaks in iter_mgf(mgf_path):
        total += 1
        entity_id = spectrum_id(metadata)
        if entity_id not in required_ids:
            continue
        if entity_id in seen:
            raise ValueError(f"duplicate representative spectrum {entity_id} in {mgf_path}")
        seen.add(entity_id)
        if len(peaks) < MINIMUM_FRAGMENT_PEAKS:
            raise ValueError(
                f"required entity {entity_id} has fewer than {MINIMUM_FRAGMENT_PEAKS} peaks"
            )
        if "PEPMASS" not in metadata:
            raise ValueError(f"required entity {entity_id} lacks PEPMASS")
        peak_array = np.asarray(peaks, dtype=np.float32).T
        ids.append(entity_id)
        records.append(
            {
                "peaks": peak_array,
                "precursor_mz": float(metadata["PEPMASS"].split()[0]),
            }
        )
    missing = required_ids - seen
    if missing:
        raise ValueError(
            f"{len(missing)} required response entities lack spectra in {mgf_path}: "
            f"{sorted(missing)[:10]}"
        )
    return ids, records, total


def embed_slim_records(
    records: list[dict[str, object]], model, *, device_name: str, batch_size: int
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from train_e1_identity import preprocess_spectrum

    device = torch.device(device_name)

    class SpectrumDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int):
            record = records[index]
            return preprocess_spectrum(
                np.asarray(record["peaks"]),
                float(record["precursor_mz"]),
                N_HIGHEST_PEAKS,
            )

    loader = DataLoader(
        SpectrumDataset(), batch_size=batch_size, shuffle=False, num_workers=0
    )
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batches.append(model(batch.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=sorted(MODEL_CHECKPOINT_SHA256))
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--study-id",
        dest="study_ids",
        action="append",
        help="Encode only this study from the contract; repeat for a subset.",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    for path in (args.contract, args.encoder_checkpoint, args.raw_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    entities_path = args.collection / "response_entities.tsv.gz"
    if not entities_path.is_file():
        raise FileNotFoundError(entities_path)

    raw_sha = sha256(args.raw_checkpoint)
    if raw_sha != RAW_CHECKPOINT_SHA256:
        raise RuntimeError(f"raw checkpoint SHA256 mismatch: {raw_sha}")
    encoder_sha = sha256(args.encoder_checkpoint)
    expected_encoder_sha = MODEL_CHECKPOINT_SHA256[args.model_kind]
    encoder_state_sha: str | None = None
    if args.model_kind == "official":
        encoder_state_sha = slim_model_state_sha256(args.encoder_checkpoint)
        if (
            encoder_sha != expected_encoder_sha
            and encoder_state_sha != OFFICIAL_MODEL_STATE_SHA256
        ):
            raise RuntimeError(
                "official checkpoint differs in container bytes and tensor state: "
                f"container={encoder_sha}, state={encoder_state_sha}"
            )
    elif encoder_sha != expected_encoder_sha:
        raise RuntimeError(
            f"{args.model_kind} checkpoint SHA256 mismatch: {encoder_sha}"
        )

    # Deliberately do not load beta, SE, contrast labels, identity status, or truth.
    entity_keys = pd.read_csv(
        entities_path,
        sep="\t",
        usecols=["study_id", "raw_entity_id", "entity_id"],
        dtype=str,
    ).drop_duplicates()
    entity_keys["raw_entity_id"] = entity_keys["raw_entity_id"].map(normalise_entity_id)
    entity_keys["entity_id"] = entity_keys["entity_id"].map(normalise_entity_id)
    if entity_keys.duplicated(["study_id", "raw_entity_id"]).any():
        raise ValueError("raw entity IDs are not unique within study")
    if entity_keys["entity_id"].duplicated().any():
        raise ValueError("collection entity IDs are not globally unique")
    contract, contract_paths = load_contract(args.contract.resolve(), ROOT)
    all_sources = contract["sources"]
    all_studies = {str(source["study_id"]) for source in all_sources}
    if set(entity_keys["study_id"]) != all_studies:
        raise ValueError("contract sources and response-entity source set differ")
    if args.study_ids:
        requested = {str(study_id) for study_id in args.study_ids}
        if not requested.issubset(all_studies):
            raise ValueError(f"requested studies are absent from contract: {sorted(requested - all_studies)}")
        sources = [source for source in all_sources if str(source["study_id"]) in requested]
    else:
        sources = all_sources

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without an allocated CUDA device")
    official_head: tuple[object, object] | None = None
    if args.model_kind == "official":
        # This repository-native path reconstructs the official backbone without
        # importing matchms; it is also the path used by the validated annotation
        # baseline.
        from annotation.embed import load_embedder

        model, head_weight, head_bias = load_embedder(
            device=device,
            raw_path=args.raw_checkpoint,
            official_path=args.encoder_checkpoint,
            n_highest_peaks=N_HIGHEST_PEAKS,
        )
        official_head = (head_weight, head_bias)
        checkpoint_kind = "official_embedding_slim"
    else:
        # Noise V1 is packaged as the same slim interface but uses the training
        # loader to recover its adapted backbone/head.
        from train_e1_identity import load_base_model

        model, checkpoint_kind = load_base_model(
            args.encoder_checkpoint,
            args.raw_checkpoint,
            device,
            N_HIGHEST_PEAKS,
        )
        if checkpoint_kind != "official_embedding_slim":
            raise RuntimeError(f"unexpected checkpoint package kind: {checkpoint_kind}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.tmp_", dir=args.output_dir.parent)
    )
    source_reports: list[dict[str, object]] = []
    started = time.time()
    try:
        for source in sources:
            study_id = str(source["study_id"])
            mgf_path = Path(source["mgf_path"])
            source_keys = entity_keys.loc[entity_keys["study_id"].eq(study_id)].copy()
            required = set(source_keys["raw_entity_id"].astype(str))
            raw_ids, records, mgf_total = load_required_records(mgf_path, required)
            raw_to_entity = source_keys.set_index("raw_entity_id")["entity_id"].to_dict()
            ids = [str(raw_to_entity[raw_id]) for raw_id in raw_ids]
            source_started = time.time()
            if official_head is not None:
                from annotation.embed import embed_records as embed_official_records

                embeddings = embed_official_records(
                    records,
                    model,
                    official_head[0],
                    official_head[1],
                    device=device,
                    n_highest_peaks=N_HIGHEST_PEAKS,
                    batch_size=args.batch_size,
                )
            else:
                embeddings = embed_slim_records(
                    records,
                    model,
                    device_name=args.device,
                    batch_size=args.batch_size,
                )
            if embeddings.ndim != 2 or len(embeddings) != len(ids):
                raise RuntimeError(f"invalid embedding shape for {study_id}: {embeddings.shape}")
            norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
            if not np.isfinite(embeddings).all() or float(np.max(np.abs(norms - 1))) > 2e-5:
                raise RuntimeError(f"non-finite or non-unit embeddings for {study_id}")
            output_path = temporary / f"{study_id}.npz"
            np.savez_compressed(
                output_path,
                entity_id=np.asarray(ids, dtype=str),
                embeddings=embeddings,
            )
            source_reports.append(
                {
                    "study_id": study_id,
                    "mgf_path": str(mgf_path),
                    "mgf_sha256": sha256(mgf_path),
                    "mgf_spectra_total": mgf_total,
                    "response_entities_encoded": len(ids),
                    "embedding_dimensions": int(embeddings.shape[1]),
                    "embedding_npz": output_path.name,
                    "embedding_npz_sha256": sha256(output_path),
                    "encoding_seconds": time.time() - source_started,
                }
            )
        manifest = pd.DataFrame(
            {
                "study_id": [str(row["study_id"]) for row in source_reports],
                "embedding_path": [
                    str(args.output_dir / str(row["embedding_npz"]))
                    for row in source_reports
                ],
                "method_id": [args.model_kind] * len(source_reports),
            }
        )
        manifest_path = temporary / "embedding_manifest.tsv"
        manifest.to_csv(manifest_path, sep="\t", index=False)
        report = {
            "schema": "contrast_atlas_open_response_frozen_embeddings_v1",
            "status": "PHENOTYPE_BLIND_FROZEN_OPEN_RESPONSE_EMBEDDINGS",
            "data_role": "development_consumed",
            "model_kind": args.model_kind,
            "checkpoint_kind": checkpoint_kind,
            "denominators": {
                "independent_studies": len(sources),
                "response_entities_encoded": int(
                    sum(int(row["response_entities_encoded"]) for row in source_reports)
                ),
                "embedding_dimensions": int(source_reports[0]["embedding_dimensions"]),
            },
            "leakage_status": (
                "encoder loaded only study_id, raw_entity_id, and entity_id from the response table; "
                "no beta, SE, contrast label, identity status, registered relation, or retrieval truth was read"
            ),
            "preprocessing": {
                "minimum_fragment_peaks": MINIMUM_FRAGMENT_PEAKS,
                "n_highest_peaks": N_HIGHEST_PEAKS,
                "output_order": "representative MGF order restricted to the frozen response-entity registry",
            },
            "sources": source_reports,
            "provenance": {
                "contract_sha256": sha256(args.contract),
                "contract_lineage_sha256": {
                    str(path.relative_to(ROOT)): sha256(path) for path in contract_paths
                },
                "response_entities_sha256": sha256(entities_path),
                "raw_checkpoint_sha256": raw_sha,
                "encoder_checkpoint_sha256": encoder_sha,
                "encoder_model_state_sha256": encoder_state_sha,
                "implementation_sha256": sha256(Path(__file__)),
            },
            "elapsed_seconds": time.time() - started,
            "claim_boundary": (
                "Frozen spectral coordinates only; this artifact contains no response score, "
                "cross-study rank, biological truth, or performance claim."
            ),
        }
        report_path = temporary / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (temporary / "checksums.sha256").write_text(
            "".join(
                f"{sha256(path)}  {path.name}\n"
                for path in sorted(temporary.iterdir())
                if path.name != "checksums.sha256"
            ),
            encoding="utf-8",
        )
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Encode several application MGFs with one frozen DreaMS-family checkpoint.

The output key is ``study_id::raw_entity_id`` so the same raw identifier may
occur in more than one study without collision.  No phenotype, response or
cross-study relation is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from encode_pan_redu_panel_spectra import load_mgf


MINIMUM_FRAGMENT_PEAKS = 3
N_HIGHEST_PEAKS = 100


def load_eligible_mgf(path: Path) -> tuple[list[str], list[dict[str, object]], list[str]]:
    all_ids, all_records = load_mgf(path, minimum_peaks=0, skip_ineligible=False)
    entity_ids, records = load_mgf(
        path, minimum_peaks=MINIMUM_FRAGMENT_PEAKS, skip_ineligible=True,
    )
    expected = [
        entity for entity, record in zip(all_ids, all_records, strict=True)
        if np.asarray(record["peaks"]).shape[-1] >= MINIMUM_FRAGMENT_PEAKS
    ]
    if entity_ids != expected:
        raise RuntimeError("MGF eligibility changed between parser passes")
    eligible = set(entity_ids)
    return entity_ids, records, [entity for entity in all_ids if entity not in eligible]


def embed_records(
    records: list[dict[str, object]], checkpoint: Path, raw_checkpoint: Path,
    *, device_name: str, batch_size: int,
) -> tuple[np.ndarray, str]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from train_e1_identity import load_base_model, preprocess_spectrum

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA encoding requested without an allocated GPU")

    class SpectrumDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int):
            record = records[index]
            return preprocess_spectrum(
                np.asarray(record["peaks"]), float(record["precursor_mz"]),
                N_HIGHEST_PEAKS,
            )

    model, checkpoint_kind = load_base_model(
        checkpoint, raw_checkpoint, device, N_HIGHEST_PEAKS,
    )
    if checkpoint_kind != "official_embedding_slim":
        raise RuntimeError(f"application encoder must be a slim embedding checkpoint: {checkpoint_kind}")
    model.eval()
    loader = DataLoader(SpectrumDataset(), batch_size=batch_size, shuffle=False, num_workers=0)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batches.append(model(batch.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False), checkpoint_kind


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-mgf", action="append", required=True,
                        help="Repeated STUDY_ID=PATH input")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError(args.output)

    studies: list[str] = []
    paths: list[Path] = []
    for item in args.study_mgf:
        if "=" not in item:
            raise ValueError(f"invalid --study-mgf value: {item}")
        study, raw_path = item.split("=", 1)
        if not study or study in studies:
            raise ValueError(f"duplicate or empty study: {study!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        studies.append(study); paths.append(path)

    global_ids: list[str] = []
    records: list[dict[str, object]] = []
    abstained: dict[str, int] = {}
    counts: dict[str, int] = {}
    for study, path in zip(studies, paths, strict=True):
        entity_ids, study_records, absent = load_eligible_mgf(path)
        global_ids.extend(f"{study}::{entity}" for entity in entity_ids)
        records.extend(study_records)
        counts[study] = len(entity_ids)
        abstained[study] = len(absent)
    if len(global_ids) != len(set(global_ids)):
        raise RuntimeError("study-qualified spectrum IDs are not unique")

    embeddings, checkpoint_kind = embed_records(
        records, args.checkpoint, args.raw_checkpoint,
        device_name=args.device, batch_size=args.batch_size,
    )
    if embeddings.shape[0] != len(global_ids):
        raise RuntimeError("embedding rows do not match spectrum registry")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, entity_id=np.asarray(global_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
    )
    report = {
        "status": "CROSSSTUDY_FROZEN_ENCODER_EMBEDDINGS_COMPLETE",
        "encoder_name": args.encoder_name,
        "checkpoint_kind": checkpoint_kind,
        "labels_or_responses_accessed": False,
        "counts_encoded": counts,
        "counts_abstained_lt3_peaks": abstained,
        "embedding_dimension": int(embeddings.shape[1]),
        "inputs": {
            "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
            "raw_checkpoint": {"path": str(args.raw_checkpoint), "sha256": sha256(args.raw_checkpoint)},
            "mgfs": {study: {"path": str(path), "sha256": sha256(path)}
                     for study, path in zip(studies, paths, strict=True)},
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Encode a frozen Pan-ReDU panel's representative MS2 spectra.

The encoder never reads contrast membership, effects, or cross-study truth.
It creates a strict anchor-id keyed interface so a frozen spectral model can be
changed without changing the biological response estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mgf(
    path: Path, *, minimum_peaks: int = 3, skip_ineligible: bool = False,
) -> tuple[list[str], list[dict[str, object]]]:
    ids: list[str] = []
    records: list[dict[str, object]] = []
    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    inside = False
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                if inside:
                    raise ValueError("nested BEGIN IONS")
                inside, fields, peaks = True, {}, []
            elif line == "END IONS":
                if not inside:
                    raise ValueError("END IONS without BEGIN IONS")
                anchor_id = fields.get("FEATURE_ID") or fields.get("TITLE") or fields.get("SCANS")
                if not anchor_id or "PEPMASS" not in fields:
                    raise ValueError("spectrum lacks FEATURE_ID/TITLE/SCANS or PEPMASS")
                if len(peaks) < minimum_peaks:
                    if skip_ineligible:
                        inside = False
                        continue
                    raise ValueError(f"{anchor_id} has fewer than {minimum_peaks} fragment peaks")
                array = np.asarray(peaks, dtype=np.float32).T
                ids.append(anchor_id)
                records.append({"peaks": array, "precursor_mz": float(fields["PEPMASS"].split()[0])})
                inside = False
            elif inside and line:
                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key.upper()] = value
                else:
                    pieces = line.split()
                    if len(pieces) < 2:
                        raise ValueError(f"invalid peak line: {line}")
                    peaks.append((float(pieces[0]), float(pieces[1])))
    if inside:
        raise ValueError("unterminated MGF spectrum")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate anchor ids in MGF")
    if not records:
        raise ValueError("MGF contains no spectra")
    return ids, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument(
        "--panel-id",
        help="Stable panel identifier; defaults to the panel-object stem for legacy callers.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--encoder-name", default="official_dreams")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    panel_report_path = args.panel_object / "report.json"
    panel_report = json.loads(panel_report_path.read_text(encoding="utf-8"))
    if panel_report.get("status") != "development_consumed_abundance_matrix_frozen":
        raise RuntimeError("panel object is not frozen")
    mgf_path = args.panel_object / "representative_spectra.mgf"
    anchor_ids, records = load_mgf(mgf_path)

    from annotation.embed import embed_records, load_embedder
    model, weight, bias = load_embedder(
        device=args.device, raw_path=args.raw_checkpoint,
        official_path=args.official_checkpoint, n_highest_peaks=100,
    )
    embeddings = embed_records(
        records, model, weight, bias, device=args.device,
        n_highest_peaks=100, batch_size=args.batch_size,
    )
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.ndim != 2 or len(embeddings) != len(anchor_ids):
        raise RuntimeError("embedding shape does not match anchors")
    if not np.isfinite(embeddings).all() or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError("invalid embedding values or norms")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "anchor_embeddings.npz"
    np.savez_compressed(
        output_path, anchor_id=np.asarray(anchor_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
    )
    report = {
        "schema": "pan_redu_anchor_embeddings_v1",
        "status": "phenotype_blind_frozen_spectral_embeddings",
        "panel_id": args.panel_id or args.panel_object.stem,
        "encoder_name": args.encoder_name,
        "denominators": {"anchors": len(anchor_ids), "embedding_dimensions": int(embeddings.shape[1])},
        "leakage_status": "no contrast membership_effect_or_cross_study_truth_read",
        "provenance": {
            "panel_report_sha256": sha256(panel_report_path),
            "representative_mgf_sha256": sha256(mgf_path),
            "raw_checkpoint_sha256": sha256(args.raw_checkpoint),
            "encoder_checkpoint_sha256": sha256(args.official_checkpoint),
            "embeddings_sha256": sha256(output_path),
        },
        "claim_boundary": "Frozen spectral coordinates only; no biological response or retrieval result.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

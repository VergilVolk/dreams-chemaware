#!/usr/bin/env python
"""Append externally computed expert scores to frozen MassSpecGym evidence.

Each expert NPZ must contain ``pair_scores`` and scalar ``evidence_sha256``.
The latter must equal the SHA256 of the exact input evidence.npz, preventing a
plausible-looking but row-misaligned ChemAware/RRF score vector from entering
router training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from noise_final_core import sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expert", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_npz = args.evidence / "evidence.npz"
    source_report = args.evidence / "report.json"
    evidence_hash = sha256_file(source_npz)
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if report.get("status") != "NOISE_MSG_PAIR_EVIDENCE_COMPLETE":
        raise RuntimeError("source evidence is not frozen")
    with np.load(source_npz, allow_pickle=False) as body:
        arrays = {name: np.asarray(body[name]) for name in body.files}
    edge_count = len(arrays["v1_cosine"])
    appended: dict[str, dict] = {}
    for spec in args.expert:
        if "=" not in spec:
            raise ValueError(f"expert must be NAME=NPZ: {spec}")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not name or name in arrays:
            raise RuntimeError(f"invalid or colliding expert name: {name!r}")
        with np.load(path, allow_pickle=False) as body:
            if set(body.files) != {"pair_scores", "evidence_sha256"}:
                raise RuntimeError(f"expert NPZ has unexpected fields: {path}")
            declared = str(np.asarray(body["evidence_sha256"]).item())
            values = np.asarray(body["pair_scores"], dtype=np.float32)
        if declared != evidence_hash:
            raise RuntimeError(f"expert evidence hash mismatch: {name}")
        if values.shape != (edge_count,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"expert score vector is malformed: {name}")
        arrays[name] = values
        appended[name] = {"path": str(path), "sha256": sha256_file(path), "pairs": edge_count}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".grand_evidence.", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "evidence.npz", **arrays)
        output_report = dict(report)
        output_report["grand_fusion_extension"] = {
            "source_evidence_sha256": evidence_hash,
            "experts": appended,
        }
        output_report["status"] = "NOISE_MSG_PAIR_EVIDENCE_COMPLETE"
        (staging / "report.json").write_text(json.dumps(output_report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(output_report["grand_fusion_extension"], indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Normalize a project-owned DreaMS checkpoint to official_embedding_slim_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e1_checkpoint_io import (
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    torch_load_compat,
)
from noise_final_core import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    package = torch_load_compat(args.input, map_location="cpu")
    kind = checkpoint_kind(package)
    if kind not in {"official_embedding", "official_embedding_slim", "e1_identity"}:
        raise RuntimeError(f"checkpoint cannot be normalized as a shared DreaMS encoder: {kind}")
    slim = {
        "format": "official_embedding_slim_v1",
        "source_checkpoint": str(args.input.resolve()),
        "source_size_bytes": args.input.stat().st_size,
        "backbone_state_dict": official_backbone_state(package),
        "head_state_dict": official_head_state(package),
        "normalization_provenance": {
            "source_kind": kind,
            "source_sha256": sha256_file(args.input),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(slim, temporary)
    temporary.replace(args.output)
    report = {
        "status": "DREAMS_SLIM_NORMALIZATION_COMPLETE",
        "source_kind": kind,
        "source_sha256": sha256_file(args.input),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

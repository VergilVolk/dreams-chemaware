"""Remove optimizer/trainer state from the large official Lightning checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from e1_checkpoint_io import (
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    torch_load_compat,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "dreams/models/pretrained/embedding_model.ckpt"
DEFAULT_OUTPUT = ROOT / "data/e1/official_embedding_slim.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.is_file() and not args.force:
        print(f"Slim official checkpoint already exists: {args.output}")
        return
    print(f"Loading official Lightning checkpoint once: {args.source}", flush=True)
    print("This one-time conversion can take several minutes on Windows.", flush=True)
    package = torch_load_compat(args.source, map_location="cpu")
    if checkpoint_kind(package) != "official_embedding":
        raise ValueError("Source is not the official ContrastiveHead checkpoint")
    slim = {
        "format": "official_embedding_slim_v1",
        "source_checkpoint": str(args.source.resolve()),
        "source_size_bytes": args.source.stat().st_size,
        "backbone_state_dict": official_backbone_state(package),
        "head_state_dict": official_head_state(package),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, args.output)
    print(f"Saved slim checkpoint: {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()

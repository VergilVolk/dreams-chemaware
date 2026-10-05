#!/usr/bin/env python
"""Compose Noise and ChemAware slim checkpoints by task-vector arithmetic.

This utility is deliberately small and strict.  It accepts only compatible
``official_embedding_slim_v1`` checkpoints, checks every key and shape, keeps
non-floating buffers unchanged, and writes hashes plus coefficients.  It does
not claim that a merge is useful; the frozen external benchmark decides that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--chem", type=Path, required=True)
    parser.add_argument("--mode", choices=("linear", "ties"), default="linear")
    parser.add_argument("--alpha-noise", type=float, default=1.0)
    parser.add_argument("--alpha-chem", type=float, default=1.0)
    parser.add_argument("--ties-density", type=float, default=0.20)
    parser.add_argument("--ties-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_slim(path: Path) -> dict:
    body = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(body, dict) or body.get("format") != "official_embedding_slim_v1":
        raise RuntimeError(f"not an official_embedding_slim_v1 checkpoint: {path}")
    for key in ("backbone_state_dict", "head_state_dict"):
        if not isinstance(body.get(key), dict):
            raise RuntimeError(f"checkpoint misses {key}: {path}")
    return body


def merge_state(
    base: dict[str, torch.Tensor], noise: dict[str, torch.Tensor], chem: dict[str, torch.Tensor],
    alpha_noise: float, alpha_chem: float,
) -> dict[str, torch.Tensor]:
    if set(base) != set(noise) or set(base) != set(chem):
        raise RuntimeError("state-dict keys are incompatible")
    result: dict[str, torch.Tensor] = {}
    for key in base:
        b, n, c = base[key].detach().cpu(), noise[key].detach().cpu(), chem[key].detach().cpu()
        if b.shape != n.shape or b.shape != c.shape or b.dtype != n.dtype or b.dtype != c.dtype:
            raise RuntimeError(f"tensor contract differs at {key}")
        if torch.is_floating_point(b):
            result[key] = b + alpha_noise * (n - b) + alpha_chem * (c - b)
        else:
            if not torch.equal(b, n) or not torch.equal(b, c):
                raise RuntimeError(f"non-floating buffer differs at {key}")
            result[key] = b.clone()
    return result


def _trim(vector: torch.Tensor, density: float) -> torch.Tensor:
    flat = vector.reshape(-1)
    keep = max(1, int(round(density * flat.numel())))
    if keep >= flat.numel():
        return vector.clone()
    threshold = torch.topk(flat.abs(), keep, sorted=False).values.min()
    return torch.where(vector.abs() >= threshold, vector, torch.zeros_like(vector))


def ties_merge_state(
    base: dict[str, torch.Tensor], noise: dict[str, torch.Tensor], chem: dict[str, torch.Tensor],
    density: float, scale: float,
) -> dict[str, torch.Tensor]:
    """Two-vector TIES: trim, elect sign, merge sign-aligned deltas."""
    if not 0.0 < density <= 1.0:
        raise ValueError("TIES density must be in (0, 1]")
    if set(base) != set(noise) or set(base) != set(chem):
        raise RuntimeError("state-dict keys are incompatible")
    result: dict[str, torch.Tensor] = {}
    for key in base:
        b, n, c = base[key].detach().cpu(), noise[key].detach().cpu(), chem[key].detach().cpu()
        if b.shape != n.shape or b.shape != c.shape or b.dtype != n.dtype or b.dtype != c.dtype:
            raise RuntimeError(f"tensor contract differs at {key}")
        if not torch.is_floating_point(b):
            if not torch.equal(b, n) or not torch.equal(b, c):
                raise RuntimeError(f"non-floating buffer differs at {key}")
            result[key] = b.clone()
            continue
        vectors = torch.stack((_trim(n - b, density), _trim(c - b, density)))
        elected = torch.sign(vectors.sum(dim=0))
        aligned = (torch.sign(vectors) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
        count = aligned.sum(dim=0)
        merged = torch.where(
            count > 0,
            (vectors * aligned).sum(dim=0) / count.clamp_min(1),
            torch.zeros_like(b),
        )
        result[key] = b + scale * merged
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    if not (-2.0 <= args.alpha_noise <= 2.0 and -2.0 <= args.alpha_chem <= 2.0):
        raise ValueError("task-vector coefficients outside preregistration safety range [-2, 2]")
    base, noise, chem = load_slim(args.base), load_slim(args.noise), load_slim(args.chem)
    merged = dict(base)
    for section in ("backbone_state_dict", "head_state_dict"):
        if args.mode == "linear":
            merged[section] = merge_state(
                base[section], noise[section], chem[section], args.alpha_noise, args.alpha_chem,
            )
        else:
            merged[section] = ties_merge_state(
                base[section], noise[section], chem[section], args.ties_density, args.ties_scale,
            )
    merged["grand_fusion_provenance"] = {
        "schema": "dreams_task_vector_merge_v1",
        "mode": args.mode,
        "operation": (
            "base + alpha_noise*(noise-base) + alpha_chem*(chem-base)"
            if args.mode == "linear" else "TIES trim-elect-sign-merge over Noise and Chem task vectors"
        ),
        "alpha_noise": args.alpha_noise,
        "alpha_chem": args.alpha_chem,
        "ties_density": args.ties_density,
        "ties_scale": args.ties_scale,
        "base": {"path": str(args.base), "sha256": sha256(args.base)},
        "noise": {"path": str(args.noise), "sha256": sha256(args.noise)},
        "chem": {"path": str(args.chem), "sha256": sha256(args.chem)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(merged, temporary)
    temporary.replace(args.output)
    report = dict(merged["grand_fusion_provenance"])
    report["output"] = {"path": str(args.output), "sha256": sha256(args.output)}
    sidecar.write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

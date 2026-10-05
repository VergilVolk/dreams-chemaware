#!/usr/bin/env python
"""Measure Noise/Chem task-vector alignment before any model merge is scored."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from merge_dreams_task_vectors import load_slim, sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--chem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(base: dict, noise: dict, chem: dict) -> dict:
    dot = norm_n = norm_c = 0.0
    sign_same = sign_total = 0
    top_overlap_values: list[float] = []
    layers = []
    for section in ("backbone_state_dict", "head_state_dict"):
        if set(base[section]) != set(noise[section]) or set(base[section]) != set(chem[section]):
            raise RuntimeError(f"incompatible keys in {section}")
        for key in base[section]:
            b, n, c = base[section][key], noise[section][key], chem[section][key]
            if b.shape != n.shape or b.shape != c.shape:
                raise RuntimeError(f"shape mismatch at {section}/{key}")
            if not torch.is_floating_point(b):
                continue
            dn = (n.detach().cpu().double() - b.detach().cpu().double()).reshape(-1)
            dc = (c.detach().cpu().double() - b.detach().cpu().double()).reshape(-1)
            local_dot = float(torch.dot(dn, dc))
            local_n = float(torch.dot(dn, dn))
            local_c = float(torch.dot(dc, dc))
            dot += local_dot
            norm_n += local_n
            norm_c += local_c
            active = (dn != 0) & (dc != 0)
            sign_same += int((torch.sign(dn[active]) == torch.sign(dc[active])).sum())
            sign_total += int(active.sum())
            keep = max(1, int(round(0.01 * len(dn))))
            top_n = set(torch.topk(dn.abs(), keep, sorted=False).indices.tolist())
            top_c = set(torch.topk(dc.abs(), keep, sorted=False).indices.tolist())
            overlap = len(top_n & top_c) / keep
            top_overlap_values.append(overlap)
            cosine = local_dot / np.sqrt(local_n * local_c) if local_n > 0 and local_c > 0 else 0.0
            layers.append({
                "section": section, "tensor": key, "parameters": int(len(dn)),
                "cosine": cosine, "noise_l2": np.sqrt(local_n), "chem_l2": np.sqrt(local_c),
                "top1pct_overlap": overlap,
            })
    return {
        "global_cosine": dot / np.sqrt(norm_n * norm_c) if norm_n > 0 and norm_c > 0 else 0.0,
        "noise_l2": np.sqrt(norm_n),
        "chem_l2": np.sqrt(norm_c),
        "active_sign_agreement": sign_same / sign_total if sign_total else None,
        "mean_tensor_top1pct_overlap": float(np.mean(top_overlap_values)),
        "tensors": layers,
        "interpretation": (
            "Positive cosine supports a small linear merge scan; near-zero or negative cosine and low sign agreement prioritize TIES or separate expert routing. This diagnostic is label-free and is not a performance claim."
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    base, noise, chem = load_slim(args.base), load_slim(args.noise), load_slim(args.chem)
    report = {
        "status": "DREAMS_TASK_VECTOR_INTERFERENCE_COMPLETE",
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
            ("base", args.base), ("noise", args.noise), ("chem", args.chem),
        )},
        "alignment": summarize(base, noise, chem),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report["alignment"].items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main()

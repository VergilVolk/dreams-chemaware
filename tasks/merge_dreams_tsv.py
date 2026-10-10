#!/usr/bin/env python
"""Merge Noise V1 and ChemAware Phase-A with the official TSV-Merge rule.

This is a direct DreaMS adaptation of ``compute_and_sum_svd_mem_reduction``
from AntoAndGar/task_singular_vectors (commit a6c9718).  With two task
vectors, each two-dimensional update contributes its leading half-rank SVD
components; the concatenated left and right singular vectors are separately
orthogonalised before reconstruction.  Non-2D floating updates are averaged.

Both fixed outputs are produced from the same three checkpoints:

* linear: base + mean(noise - base, chem - base)
* tsv:    base + TSV(noise - base, chem - base)

The TSV scaling coefficient is deliberately fixed to 1.0.  No data, labels,
gradients, or coefficient search are used during checkpoint construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


OFFICIAL_TSV_REPOSITORY = "https://github.com/AntoAndGar/task_singular_vectors"
OFFICIAL_TSV_COMMIT = "a6c97188f7aa0bb20753de4f2939be89dd64f1c3"
TSV_SCALE = 1.0
N_TASKS = 2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--chem", type=Path, required=True)
    parser.add_argument("--linear-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_slim(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    body = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(body, dict) or body.get("format") != "official_embedding_slim_v1":
        raise RuntimeError(f"not an official_embedding_slim_v1 checkpoint: {path}")
    for section in ("backbone_state_dict", "head_state_dict"):
        if not isinstance(body.get(section), dict):
            raise RuntimeError(f"checkpoint misses {section}: {path}")
    return body


def validate_states(
    base: dict[str, torch.Tensor],
    noise: dict[str, torch.Tensor],
    chem: dict[str, torch.Tensor],
) -> None:
    if set(base) != set(noise) or set(base) != set(chem):
        raise RuntimeError("state-dict keys are incompatible")
    for key in base:
        tensors = (base[key], noise[key], chem[key])
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise RuntimeError(f"non-tensor state entry at {key}")
        if len({value.shape for value in tensors}) != 1:
            raise RuntimeError(f"tensor shape differs at {key}")
        if len({value.dtype for value in tensors}) != 1:
            raise RuntimeError(f"tensor dtype differs at {key}")
        if not torch.is_floating_point(base[key]) and not (
            torch.equal(base[key], noise[key]) and torch.equal(base[key], chem[key])
        ):
            raise RuntimeError(f"non-floating buffer differs at {key}")


def official_tsv_two_vector_update(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Mirror the official two-task TSV core for one floating task tensor."""
    if first.shape != second.shape or first.dtype != second.dtype:
        raise RuntimeError("TSV task tensors are incompatible")
    if first.ndim != 2:
        return ((first.to(device) + second.to(device)) / N_TASKS).cpu()

    # The author implementation uses the native tensor dtype and
    # full_matrices=False.  For two tasks, sv_reduction is exactly 1/2.
    vectors = (first, second)
    sum_u = sum_s = sum_v = None
    reduced_rank = None
    with torch.no_grad():
        for index, vector in enumerate(vectors):
            u, singular, vh = torch.linalg.svd(vector.to(device), full_matrices=False)
            if sum_u is None:
                sum_u = torch.zeros_like(u, device=device)
                sum_s = torch.zeros_like(singular, device=device)
                sum_v = torch.zeros_like(vh, device=device)
                reduced_rank = int(singular.shape[0] / N_TASKS)
            assert sum_u is not None and sum_s is not None and sum_v is not None
            assert reduced_rank is not None
            left = index * reduced_rank
            right = (index + 1) * reduced_rank
            sum_u[:, left:right] = u[:, :reduced_rank]
            sum_s[left:right] = singular[:reduced_rank]
            sum_v[left:right, :] = vh[:reduced_rank, :]

        # This is the exact orthogonalisation/reconstruction order used by
        # compute_and_sum_svd_mem_reduction in the author repository.
        u_u, _s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
    return merged.cpu()


def merge_section(
    base: dict[str, torch.Tensor],
    noise: dict[str, torch.Tensor],
    chem: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    validate_states(base, noise, chem)
    linear: dict[str, torch.Tensor] = {}
    tsv: dict[str, torch.Tensor] = {}
    matrix_keys = 0
    non_matrix_float_keys = 0
    non_float_keys = 0
    for index, key in enumerate(base):
        base_tensor = base[key].detach().cpu()
        noise_tensor = noise[key].detach().cpu()
        chem_tensor = chem[key].detach().cpu()
        if not torch.is_floating_point(base_tensor):
            linear[key] = base_tensor.clone()
            tsv[key] = base_tensor.clone()
            non_float_keys += 1
            continue
        noise_update = noise_tensor - base_tensor
        chem_update = chem_tensor - base_tensor
        mean_update = (noise_update + chem_update) / N_TASKS
        linear[key] = base_tensor + mean_update
        if base_tensor.ndim == 2:
            matrix_keys += 1
            print(
                f"[TSV {index + 1}/{len(base)}] {key} shape={tuple(base_tensor.shape)}",
                flush=True,
            )
        else:
            non_matrix_float_keys += 1
        tsv_update = official_tsv_two_vector_update(
            noise_update, chem_update, device=device,
        )
        tsv[key] = base_tensor + TSV_SCALE * tsv_update.to(base_tensor.dtype)
    stats = {
        "two_dimensional_float_tensors": matrix_keys,
        "non_2d_float_tensors": non_matrix_float_keys,
        "non_float_tensors": non_float_keys,
    }
    return linear, tsv, stats


def atomic_save(package: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(package, temporary)
    os.replace(temporary, output)


def main() -> None:
    args = arguments()
    if args.linear_output.resolve() == args.tsv_output.resolve():
        raise ValueError("linear and TSV outputs must differ")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires an allocated CUDA device")
    device = torch.device(args.device)
    base = load_slim(args.base)
    noise = load_slim(args.noise)
    chem = load_slim(args.chem)

    linear_package = dict(base)
    tsv_package = dict(base)
    section_stats: dict[str, Any] = {}
    for section in ("backbone_state_dict", "head_state_dict"):
        linear_state, tsv_state, stats = merge_section(
            base[section], noise[section], chem[section], device=device,
        )
        linear_package[section] = linear_state
        tsv_package[section] = tsv_state
        section_stats[section] = stats

    inputs = {
        "base": {"path": str(args.base), "sha256": sha256(args.base)},
        "noise": {"path": str(args.noise), "sha256": sha256(args.noise)},
        "chem": {"path": str(args.chem), "sha256": sha256(args.chem)},
    }
    common = {
        "schema": "dreams_tsv_merge_v1",
        "inputs": inputs,
        "n_tasks": N_TASKS,
        "rank_fraction_per_task": 1 / N_TASKS,
        "scale": TSV_SCALE,
        "device": str(device),
        "official_source": {
            "repository": OFFICIAL_TSV_REPOSITORY,
            "commit": OFFICIAL_TSV_COMMIT,
            "function": "compute_and_sum_svd_mem_reduction",
        },
        "adaptation": (
            "All DreaMS 2D backbone and contrastive-head updates use TSV; the CLIP-only "
            "text_projection exclusion is inapplicable. Non-2D floating updates are averaged."
        ),
        "section_stats": section_stats,
    }
    linear_package["model_merge_provenance"] = {
        **common,
        "method": "linear_task_vector_mean",
        "operation": "base + mean(noise-base, chem-base)",
    }
    tsv_package["model_merge_provenance"] = {
        **common,
        "method": "tsv_merge",
        "operation": "base + 1.0 * official_two_task_tsv(noise-base, chem-base)",
    }
    atomic_save(linear_package, args.linear_output)
    atomic_save(tsv_package, args.tsv_output)

    report = {
        "status": "DREAMS_FIXED_LINEAR_AND_TSV_MERGE_COMPLETE",
        **common,
        "outputs": {
            "linear": {"path": str(args.linear_output), "sha256": sha256(args.linear_output)},
            "tsv": {"path": str(args.tsv_output), "sha256": sha256(args.tsv_output)},
        },
        "selection_data_accessed": False,
        "massspecgym_accessed": False,
        "enveda_accessed": False,
    }
    report_path = args.tsv_output.parent / "merge_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

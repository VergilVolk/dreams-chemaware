#!/usr/bin/env python
"""B26 engineering canary for direct BioAware action injection.

This stage does not estimate generalisation.  It asks a narrower, necessary
question: can one shared DreaMS encoder receive the exact B17/B20 spectrum-level
actions and move the clean query/reference margin in the requested direction?

Only two supervised action types are used:

* corrective: true molecule versus the official DreaMS wrong Top-1;
* safety: true molecule versus the B17 candidate that would introduce an error.

The reaction graph and catalogue scores route examples upstream, but neither is
available to the loss or to inference.  Query and reference spectra are always
encoded by the same trainable model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-corrective-identities", type=int, default=8)
    parser.add_argument("--batch-safety-identities", type=int, default=4)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=0.03)
    parser.add_argument("--lambda-corrective", type=float, default=1.0)
    parser.add_argument("--lambda-safety", type=float, default=4.0)
    parser.add_argument("--lambda-preserve", type=float, default=10.0)
    parser.add_argument("--safety-slack", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=96)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_manifest(
    path: Path, expected_corrective_rows: int, expected_safety_rows: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    required = {
        "query_id", "source", "physical_query_id", "truth_candidate_id",
        "truth_formula", "baseline_candidate_id", "final_candidate_id",
        "query_tensor", "reference_tensor_rows", "reference_tensor",
        "truth_reference_position", "baseline_reference_position",
        "final_reference_position", "baseline_correct", "final_correct",
        "direct_corrective", "direct_safety", "physical_duplicate_weight",
        "identity_equal_weight",
    }
    missing = required - set(body)
    if missing:
        raise RuntimeError(f"B20 manifest lacks {sorted(missing)}")
    n = len(body["query_id"])
    if n != 860 or body["query_tensor"].shape != (n, 101, 2):
        raise RuntimeError("B20 query tensor contract changed")
    if body["reference_tensor"].ndim != 3 or body["reference_tensor"].shape[1:] != (101, 2):
        raise RuntimeError("B20 reference tensor contract changed")
    for name in (
        "truth_reference_position", "baseline_reference_position",
        "final_reference_position",
    ):
        values = np.asarray(body[name], dtype=np.int64)
        if len(values) != n or np.any(values < 0) or np.any(values >= len(body["reference_tensor"])):
            raise RuntimeError(f"invalid B20 {name}")
    if (int(np.sum(body["direct_corrective"])) != expected_corrective_rows
            or int(np.sum(body["direct_safety"])) != expected_safety_rows):
        raise RuntimeError("direct action-bank row counts changed")
    if np.any(body["direct_corrective"] & body["direct_safety"]):
        raise RuntimeError("B20 corrective/safety actions overlap")
    return body


def collapse_physical_actions(
    frame: pd.DataFrame, expected_corrective: int, expected_safety: int,
) -> pd.DataFrame:
    """Collapse duplicated opened-cohort views without changing an action."""
    selected: list[pd.Series] = []
    for _, group in frame.groupby(["source", "physical_query_id"], sort=True):
        corrective = bool(group["direct_corrective"].astype(bool).any())
        safety = bool(group["direct_safety"].astype(bool).any())
        if corrective and safety:
            raise RuntimeError("one physical spectrum has contradictory B20 roles")
        if not (corrective or safety):
            continue
        columns = [
            "truth_candidate_id", "baseline_candidate_id", "final_candidate_id",
            "truth_reference_position", "baseline_reference_position",
            "final_reference_position",
        ]
        for column in columns:
            if group[column].nunique(dropna=False) != 1:
                raise RuntimeError(
                    f"physical duplicate disagrees on {column}: "
                    f"{group[['source', 'physical_query_id']].iloc[0].to_dict()}"
                )
        row = group.iloc[0].copy()
        row["direct_corrective"] = corrective
        row["direct_safety"] = safety
        row["manifest_position"] = int(group.index[0])
        selected.append(row)
    output = pd.DataFrame(selected).reset_index(drop=True)
    if (len(output) != expected_corrective + expected_safety
            or int(output["direct_corrective"].sum()) != expected_corrective
            or int(output["direct_safety"].sum()) != expected_safety):
        raise RuntimeError("direct action-bank physical collapse changed")
    return output


@torch.no_grad()
def encode(
    model: torch.nn.Module,
    spectra: torch.Tensor,
    device: torch.device,
    batch_size: int,
    label: str,
) -> np.ndarray:
    model.eval()
    result = []
    for left in range(0, len(spectra), batch_size):
        result.append(model(spectra[left:left + batch_size].to(device)).float().cpu().numpy())
    output = np.concatenate(result).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite embeddings during {label}")
    print(f"[{label}] encoded={len(output):,}", flush=True)
    return output


def action_margins(
    actions: pd.DataFrame,
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
) -> np.ndarray:
    margins = []
    for position, row in enumerate(actions.itertuples(index=False)):
        positive = int(row.truth_reference_position)
        negative = (
            int(row.baseline_reference_position)
            if bool(row.direct_corrective)
            else int(row.final_reference_position)
        )
        margins.append(float(
            reference_embedding[positive] @ query_embedding[position]
            - reference_embedding[negative] @ query_embedding[position]
        ))
    return np.asarray(margins, dtype=np.float32)


def pools_by_identity(actions: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    output = []
    for column in ("direct_corrective", "direct_safety"):
        local: dict[str, list[int]] = {}
        for index, row in actions.loc[actions[column].astype(bool)].iterrows():
            local.setdefault(str(row["truth_candidate_id"]), []).append(int(index))
        output.append({key: np.asarray(value, dtype=np.int64) for key, value in local.items()})
    return output[0], output[1]


def sample_identity_balanced(
    pool: dict[str, np.ndarray], count: int, rng: np.random.Generator,
) -> list[int]:
    identities = sorted(pool)
    if not identities:
        raise RuntimeError("empty identity-balanced action pool")
    chosen = []
    for _ in range(count):
        identity = identities[int(rng.integers(0, len(identities)))]
        values = pool[identity]
        chosen.append(int(values[int(rng.integers(0, len(values)))]))
    return chosen


def component_gradient(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool,
) -> list[torch.Tensor | None]:
    return list(torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True,
    ))


def gradient_summary(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None] | None = None,
) -> dict[str, float]:
    norm_first = sum(float(torch.sum(value.detach().float() ** 2)) for value in first if value is not None)
    result = {"norm": math.sqrt(max(norm_first, 0.0))}
    if second is not None:
        norm_second = sum(float(torch.sum(value.detach().float() ** 2)) for value in second if value is not None)
        dot = sum(
            float(torch.sum(a.detach().float() * b.detach().float()))
            for a, b in zip(first, second, strict=True)
            if a is not None and b is not None
        )
        denominator = math.sqrt(max(norm_first * norm_second, 0.0))
        result["cosine"] = float(dot / denominator) if denominator > 0 else 0.0
    return result


def make_live_batch(
    actions: pd.DataFrame,
    indices: list[int],
    query_tensor: torch.Tensor,
    reference_tensor: torch.Tensor,
    official_query: np.ndarray,
    official_reference: np.ndarray,
) -> tuple[torch.Tensor, list[dict], torch.Tensor]:
    spectra: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    layout = []
    for index in indices:
        row = actions.iloc[int(index)]
        query_position = int(row["manifest_position"])
        positive = int(row["truth_reference_position"])
        negative = (
            int(row["baseline_reference_position"])
            if bool(row["direct_corrective"])
            else int(row["final_reference_position"])
        )
        q = len(spectra)
        spectra.append(query_tensor[query_position])
        targets.append(torch.from_numpy(official_query[int(index)]))
        p = len(spectra)
        spectra.append(reference_tensor[positive])
        targets.append(torch.from_numpy(official_reference[positive]))
        n = len(spectra)
        spectra.append(reference_tensor[negative])
        targets.append(torch.from_numpy(official_reference[negative]))
        layout.append({
            "action_index": int(index), "query": q, "positive": p, "negative": n,
            "corrective": bool(row["direct_corrective"]),
        })
    return torch.stack(spectra), layout, torch.stack(targets)


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    if (args.steps < 1 or args.batch_corrective_identities < 1
            or args.batch_safety_identities < 1 or args.head_lr < args.backbone_lr
            or args.temperature <= 0 or args.lambda_corrective <= 0
            or args.lambda_safety <= 0 or args.lambda_preserve <= 0):
        raise ValueError("invalid B26 optimization configuration")
    required = [
        args.manifest_dir / "report.json",
        args.manifest_dir / "direct_actions.csv.gz",
        args.manifest_dir / "direct_action_manifest.npz",
        args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    manifest_report = json.loads((args.manifest_dir / "report.json").read_text(encoding="utf-8"))
    if (manifest_report.get("status") not in {
        "bioaware_b20_direct_action_manifest_complete",
        "bioaware_b31_sink_safe_action_bank_complete",
        "bioaware_b32_comprehensive_action_bank_complete",
        } or manifest_report.get("pass_to_direct_gradient_canary") is not True):
        raise RuntimeError("direct action bank did not pass its spectrum gate")
    summary_key = (
        "B17_replay"
        if manifest_report["status"] == "bioaware_b20_direct_action_manifest_complete"
        else "action_bank"
    )
    action_summary = manifest_report[summary_key]
    expected_corrective_rows = int(
        action_summary.get("corrected_rows", action_summary.get("corrective_rows"))
    )
    expected_safety_rows = int(
        action_summary.get("introduced_rows", action_summary.get("safety_rows"))
    )
    expected_corrective_physical = int(
        action_summary.get(
            "corrected_physical_queries", action_summary.get("corrective_physical_queries")
        )
    )
    expected_safety_physical = int(
        action_summary.get(
            "introduced_physical_queries", action_summary.get("safety_physical_queries")
        )
    )
    manifest_path = args.manifest_dir / "direct_action_manifest.npz"
    if manifest_report["provenance"].get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("B20 manifest provenance mismatch")

    body = load_manifest(
        manifest_path, expected_corrective_rows, expected_safety_rows,
    )
    frame = pd.DataFrame({
        "query_id": body["query_id"].astype(str),
        "source": body["source"].astype(str),
        "physical_query_id": body["physical_query_id"].astype(str),
        "truth_candidate_id": body["truth_candidate_id"].astype(str),
        "truth_formula": body["truth_formula"].astype(str),
        "baseline_candidate_id": body["baseline_candidate_id"].astype(str),
        "final_candidate_id": body["final_candidate_id"].astype(str),
        "truth_reference_position": body["truth_reference_position"].astype(np.int64),
        "baseline_reference_position": body["baseline_reference_position"].astype(np.int64),
        "final_reference_position": body["final_reference_position"].astype(np.int64),
        "direct_corrective": body["direct_corrective"].astype(bool),
        "direct_safety": body["direct_safety"].astype(bool),
    })
    frame.index = np.arange(len(frame), dtype=np.int64)
    # A physical spectrum may appear in several opened evaluation views.  It
    # must not silently become several different training inputs.
    for _, group in frame.groupby(["source", "physical_query_id"], sort=False):
        positions = group.index.to_numpy(np.int64)
        if len(positions) > 1:
            reference = np.asarray(body["query_tensor"][positions[0]], dtype=np.float32)
            if any(
                not np.array_equal(
                    reference, np.asarray(body["query_tensor"][position], dtype=np.float32)
                )
                for position in positions[1:]
            ):
                raise RuntimeError("one physical query id maps to non-identical tensors")
    actions = collapse_physical_actions(
        frame, expected_corrective_physical, expected_safety_physical,
    )
    corrective_pool, safety_pool = pools_by_identity(actions)

    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal B26 canary requires CUDA")
    query_tensor = torch.from_numpy(np.asarray(body["query_tensor"], dtype=np.float32))
    reference_tensor = torch.from_numpy(np.asarray(body["reference_tensor"], dtype=np.float32))
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    model.eval()
    action_query_tensor = query_tensor[
        actions["manifest_position"].to_numpy(np.int64)
    ]
    official_query = encode(
        model, action_query_tensor, device, args.encode_batch_size, "official-action-query-fp32",
    )
    official_reference = encode(
        model, reference_tensor, device, args.encode_batch_size, "official-reference-fp32",
    )
    pre_margin = action_margins(actions, official_query, official_reference)
    corrective_mask = actions["direct_corrective"].to_numpy(bool)
    safety_mask = actions["direct_safety"].to_numpy(bool)
    # These are exact chosen reference spectra for the official top identities.
    if np.any(pre_margin[corrective_mask] >= 1e-5):
        raise RuntimeError("B20 corrective route does not replay official wrong pairwise ordering")
    if np.any(pre_margin[safety_mask] <= 0):
        raise RuntimeError("B20 safety route does not replay official correct pairwise ordering")

    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()  # dropout disabled; autograd remains enabled
    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    parameters = backbone_parameters + head_parameters
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": head_parameters, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    initial_backbone = [parameter.detach().cpu().clone() for parameter in backbone_parameters]
    initial_head = [parameter.detach().cpu().clone() for parameter in head_parameters]
    rng = np.random.default_rng(args.seed)
    history = []
    first_gradient_audit: dict | None = None
    for step in range(args.steps):
        chosen = sample_identity_balanced(
            corrective_pool, args.batch_corrective_identities, rng,
        ) + sample_identity_balanced(
            safety_pool, args.batch_safety_identities, rng,
        )
        permutation = rng.permutation(len(chosen))
        chosen = [chosen[int(position)] for position in permutation]
        spectra, layout, preserve_target = make_live_batch(
            actions, chosen, query_tensor, reference_tensor,
            official_query, official_reference,
        )
        spectra = spectra.to(device)
        preserve_target = preserve_target.to(device)
        z = model(spectra)
        if not bool(torch.isfinite(z).all().item()):
            raise RuntimeError(f"non-finite embedding at B26 step {step}")
        corrective_losses, safety_losses = [], []
        batch_margin = []
        for item in layout:
            margin = (
                z[item["query"]] * z[item["positive"]]
            ).sum() - (
                z[item["query"]] * z[item["negative"]]
            ).sum()
            batch_margin.append(margin)
            if item["corrective"]:
                corrective_losses.append(F.softplus(
                    (args.rank_margin - margin) / args.temperature,
                ))
            else:
                floor = float(pre_margin[item["action_index"]] - args.safety_slack)
                safety_losses.append(F.relu(
                    torch.as_tensor(floor, device=device, dtype=z.dtype) - margin,
                ))
        corrective_loss = torch.stack(corrective_losses).mean()
        safety_loss = torch.stack(safety_losses).mean()
        preserve_loss = (1.0 - torch.sum(z * preserve_target, dim=1)).mean()
        loss = (
            args.lambda_corrective * corrective_loss
            + args.lambda_safety * safety_loss
            + args.lambda_preserve * preserve_loss
        )
        if first_gradient_audit is None:
            corrective_gradient = component_gradient(corrective_loss, parameters, True)
            safety_gradient = component_gradient(safety_loss, parameters, True)
            preserve_gradient = component_gradient(preserve_loss, parameters, True)
            split = len(backbone_parameters)
            first_gradient_audit = {
                "corrective_all": gradient_summary(corrective_gradient),
                "corrective_backbone": gradient_summary(corrective_gradient[:split]),
                "corrective_head": gradient_summary(corrective_gradient[split:]),
                "safety_all": gradient_summary(safety_gradient),
                "preserve_all": gradient_summary(preserve_gradient),
                "corrective_vs_safety": gradient_summary(corrective_gradient, safety_gradient),
                "corrective_vs_preserve": gradient_summary(corrective_gradient, preserve_gradient),
            }
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in parameters
        ):
            raise RuntimeError(f"non-finite gradient at B26 step {step}")
        raw_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
        if not np.isfinite(raw_norm):
            raise RuntimeError(f"non-finite gradient norm at B26 step {step}")
        optimizer.step()
        record = {
            "step": int(step + 1),
            "loss": float(loss.detach()),
            "corrective": float(corrective_loss.detach()),
            "safety": float(safety_loss.detach()),
            "preserve": float(preserve_loss.detach()),
            "mean_margin": float(torch.stack(batch_margin).mean().detach()),
            "raw_grad_norm": raw_norm,
            "clipped": bool(raw_norm > args.grad_clip),
        }
        history.append(record)
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == args.steps:
            print(f"[B26 step {step + 1}/{args.steps}] {record}", flush=True)

    post_query = encode(
        model, action_query_tensor, device, args.encode_batch_size, "adapted-action-query-fp32",
    )
    post_reference = encode(
        model, reference_tensor, device, args.encode_batch_size, "adapted-reference-fp32",
    )
    post_margin = action_margins(actions, post_query, post_reference)
    query_preservation = np.sum(official_query * post_query, axis=1)
    reference_preservation = np.sum(official_reference * post_reference, axis=1)
    backbone_parameter_delta = math.sqrt(sum(
        float(torch.sum((parameter.detach().cpu() - initial) ** 2))
        for parameter, initial in zip(backbone_parameters, initial_backbone, strict=True)
    ))
    head_parameter_delta = math.sqrt(sum(
        float(torch.sum((parameter.detach().cpu() - initial) ** 2))
        for parameter, initial in zip(head_parameters, initial_head, strict=True)
    ))
    corrected = int(np.sum(post_margin[corrective_mask] > 0))
    introduced = int(np.sum(post_margin[safety_mask] <= 0))
    corrective_delta = post_margin[corrective_mask] - pre_margin[corrective_mask]
    safety_delta = post_margin[safety_mask] - pre_margin[safety_mask]
    gates = {
        "official_corrective_pairwise_replay_exact": bool(np.all(pre_margin[corrective_mask] < 0)),
        "official_safety_pairwise_replay_exact": bool(np.all(pre_margin[safety_mask] > 0)),
        "corrective_gradient_reaches_backbone": bool(
            first_gradient_audit["corrective_backbone"]["norm"] > 0
        ),
        "corrective_gradient_reaches_head": bool(
            first_gradient_audit["corrective_head"]["norm"] > 0
        ),
        "backbone_parameters_changed": backbone_parameter_delta > 0,
        "head_parameters_changed": head_parameter_delta > 0,
        "corrective_mean_margin_delta_ge_0_005": bool(np.mean(corrective_delta) >= 0.005),
        "corrective_actions_realised_ge_5": corrected >= 5,
        "known_harm_introduced_le_1": introduced <= 1,
        "minimum_query_preservation_ge_0_98": bool(np.min(query_preservation) >= 0.98),
        "minimum_reference_preservation_ge_0_98": bool(np.min(reference_preservation) >= 0.98),
    }
    report = {
        "status": "bioaware_b26_direct_shared_embedding_canary_complete",
        "formal": True,
        "scope": "same-action engineering canary; no held-out performance claim",
        "actions": {
            "physical_queries": int(len(actions)),
            "corrective_queries": int(np.sum(corrective_mask)),
            "corrective_identities": int(actions.loc[corrective_mask, "truth_candidate_id"].nunique()),
            "corrective_formulas": int(actions.loc[corrective_mask, "truth_formula"].nunique()),
            "safety_queries": int(np.sum(safety_mask)),
            "safety_identities": int(actions.loc[safety_mask, "truth_candidate_id"].nunique()),
        },
        "official": {
            "corrective_pairwise_accuracy": float(np.mean(pre_margin[corrective_mask] > 0)),
            "safety_pairwise_accuracy": float(np.mean(pre_margin[safety_mask] > 0)),
            "corrective_mean_margin": float(np.mean(pre_margin[corrective_mask])),
            "safety_mean_margin": float(np.mean(pre_margin[safety_mask])),
        },
        "adapted": {
            "corrective_pairwise_accuracy": float(np.mean(post_margin[corrective_mask] > 0)),
            "safety_pairwise_accuracy": float(np.mean(post_margin[safety_mask] > 0)),
            "corrective_mean_margin": float(np.mean(post_margin[corrective_mask])),
            "corrective_mean_margin_delta": float(np.mean(corrective_delta)),
            "safety_mean_margin": float(np.mean(post_margin[safety_mask])),
            "safety_mean_margin_delta": float(np.mean(safety_delta)),
            "corrected": corrected,
            "introduced": introduced,
            "risk_net_lambda2": int(corrected - 2 * introduced),
            "query_preservation_mean": float(np.mean(query_preservation)),
            "query_preservation_minimum": float(np.min(query_preservation)),
            "reference_preservation_mean": float(np.mean(reference_preservation)),
            "reference_preservation_minimum": float(np.min(reference_preservation)),
            "backbone_parameter_l2_delta": backbone_parameter_delta,
            "head_parameter_l2_delta": head_parameter_delta,
        },
        "first_step_gradient_audit": first_gradient_audit,
        "capacity": capacity,
        "history": history,
        "gates": gates,
        "pass_to_formula_isolated_training": bool(all(gates.values())),
        "contracts": {
            "query_reference_encoder_shared": True,
            "inference_candidate_independent": True,
            "catalogue_score_distilled": False,
            "reaction_neighbour_used_as_identity_positive": False,
            "physical_query_duplicates_collapsed": True,
            "identity_balanced_sampling": True,
            "dropout_disabled": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "parameters": vars(args) | {
            "manifest_dir": str(args.manifest_dir),
            "official_checkpoint": str(args.official_checkpoint),
            "architecture_checkpoint": str(args.architecture_checkpoint),
            "output": str(args.output),
        },
        "provenance": {
            "initialization": initialization,
            "action_manifest_sha256": sha256_file(manifest_path),
            "action_report_sha256": sha256_file(args.manifest_dir / "report.json"),
            "action_manifest_status": manifest_report["status"],
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "elapsed_seconds": float(time.time() - started),
        "claim_limit": (
            "B26 is an on-action gradient and memorisation canary. Passing proves that the "
            "shared encoder can receive the exact direct action; it does not establish "
            "formula-isolated, external-cohort, blind, or SOTA retrieval gain."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "report.json", report)
    actions.assign(
        official_margin=pre_margin,
        adapted_margin=post_margin,
        margin_delta=post_margin - pre_margin,
    ).to_csv(args.output / "per_action.csv.gz", index=False, compression="gzip")
    if not report["pass_to_formula_isolated_training"]:
        raise RuntimeError(f"B26 canary gate failed: {gates}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

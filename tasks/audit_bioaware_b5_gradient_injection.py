#!/usr/bin/env python
"""Prove that candidate-level BioAware targets inject a distinct DreaMS gradient.

This is a one-step mechanism gate, not a performance experiment.  Only teacher-
corrected, nested-OOF actions are used.  The exact teacher candidate distribution
is distilled; a one-hot truth objective is computed only as a gradient control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file  # noqa: E402
from train_bioaware_b4_direct_shared_embedding import (  # noqa: E402
    ManifestReferenceStore,
    load_base_model,
    load_manifest,
    query_candidates,
    unfreeze_last_blocks,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b4-manifest", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest/manifest.npz",
    )
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("full_bioaware_safe", "full_no_edge_high_recall"),
        default="full_bioaware_safe",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--student-temperature", type=float, default=0.10)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def encode(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device,
           batch_size: int, grad: bool) -> torch.Tensor:
    outputs = []
    context = torch.enable_grad if grad else torch.no_grad
    with context():
        for left in range(0, len(tensor), batch_size):
            outputs.append(model(tensor[left:left + batch_size].to(device)))
    return torch.cat(outputs)


def candidate_scores(
    body: dict[str, np.ndarray], query_indices: list[int], query_z: torch.Tensor,
    reference_z: torch.Tensor, reference_position: dict[int, int],
) -> list[torch.Tensor]:
    result = []
    for qpos, query in enumerate(query_indices):
        local = []
        for molecule in query_candidates(body, query):
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            positions = [reference_position[int(row)] for row in body["reference_rows"][left:right]]
            local.append(torch.max(reference_z[positions] @ query_z[qpos]))
        result.append(torch.stack(local))
    return result


def gradient_cosine(left: tuple[torch.Tensor | None, ...],
                    right: tuple[torch.Tensor | None, ...]) -> tuple[float, float, float]:
    dot = torch.zeros((), device=next(value.device for value in left if value is not None))
    norm_left = torch.zeros_like(dot)
    norm_right = torch.zeros_like(dot)
    for lvalue, rvalue in zip(left, right, strict=True):
        if lvalue is None or rvalue is None:
            continue
        dot = dot + torch.sum(lvalue * rvalue)
        norm_left = norm_left + torch.sum(lvalue * lvalue)
        norm_right = norm_right + torch.sum(rvalue * rvalue)
    cosine = dot / torch.sqrt(norm_left * norm_right).clamp_min(1e-30)
    return float(cosine), float(torch.sqrt(norm_left)), float(torch.sqrt(norm_right))


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    action_path = args.teacher_dir / "teacher_actions.csv.gz"
    candidate_path = args.teacher_dir / "candidate_teacher_scores.csv.gz"
    for path in (args.b4_manifest, action_path, candidate_path,
                 args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    body = load_manifest(args.b4_manifest)
    actions = pd.read_csv(action_path)
    candidates = pd.read_csv(candidate_path)
    actions = actions[
        actions["arm"].eq(args.arm)
        & actions["outer_fold"].eq(args.outer_fold)
        & actions["corrected"].astype(bool)
    ].copy()
    candidates = candidates[
        candidates["arm"].eq(args.arm)
        & candidates["outer_fold"].eq(args.outer_fold)
        & candidates["query_id"].astype(str).isin(actions["query_id"].astype(str))
    ].copy()
    if len(actions) < 8 or actions["query_id"].duplicated().any():
        raise RuntimeError(f"insufficient or duplicate corrected actions: {len(actions)}")
    query_position = {str(value): index for index, value in enumerate(body["query_id"])}
    query_indices = [query_position[str(value)] for value in actions["query_id"]]
    if any(int(body["formula_fold"][query]) == args.outer_fold for query in query_indices):
        raise RuntimeError("outer-held formula entered the gradient audit")

    store = ManifestReferenceStore(body["reference_tensor_rows"], body["reference_tensor"])
    selected_rows = set()
    for query in query_indices:
        for molecule in query_candidates(body, query):
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            selected_rows.update(map(int, body["reference_rows"][left:right]))
    reference_rows = np.asarray(sorted(selected_rows), dtype=np.int64)
    reference_position = {int(row): position for position, row in enumerate(reference_rows)}
    query_tensor = torch.from_numpy(body["query_tensor"][query_indices].astype(np.float32))
    reference_tensor = store.get(reference_rows)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B5 gradient injection audit requires CUDA")
    model, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, 100,
    )
    model.eval()
    capacity = unfreeze_last_blocks(model, 1)
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": head, "lr": args.head_lr, "weight_decay": 1e-4},
    ])

    query_z = encode(model, query_tensor, device, args.batch_size, True)
    reference_z = encode(model, reference_tensor, device, args.batch_size, True)
    student_groups = candidate_scores(body, query_indices, query_z, reference_z, reference_position)
    teacher_groups = []
    baseline_positions = []
    for query, action in zip(query_indices, actions.itertuples(index=False), strict=True):
        group = candidates[candidates["query_id"].astype(str).eq(str(action.query_id))]
        score_by_candidate = dict(zip(
            group["candidate_id"].astype(str), group["teacher_model_score"].astype(float), strict=True,
        ))
        molecule_indices = query_candidates(body, query)
        ids = [str(body["molecule_id"][molecule]) for molecule in molecule_indices]
        if ids[0] != str(action.truth_candidate_id) or str(action.proposed_candidate_id) != ids[0]:
            raise RuntimeError(f"{action.query_id}: corrected teacher target is not exact truth")
        if set(ids) != set(score_by_candidate):
            raise RuntimeError(f"{action.query_id}: candidate target universe mismatch")
        teacher_groups.append(torch.tensor(
            [score_by_candidate[value] for value in ids], device=device, dtype=query_z.dtype,
        ))
        baseline_positions.append(ids.index(str(action.baseline_candidate_id)))

    bio_losses = []
    generic_losses = []
    initial_margins = []
    for student, teacher, baseline_position in zip(
            student_groups, teacher_groups, baseline_positions, strict=True):
        target = F.softmax(teacher / args.teacher_temperature, dim=0).detach()
        bio_losses.append(-torch.sum(target * F.log_softmax(
            student / args.student_temperature, dim=0,
        )))
        generic_losses.append(-F.log_softmax(
            student / args.student_temperature, dim=0,
        )[0])
        initial_margins.append(float((student[0] - student[baseline_position]).detach()))
    bio_loss = torch.stack(bio_losses).mean()
    generic_loss = torch.stack(generic_losses).mean()
    official_query = torch.from_numpy(body["query_official_embedding"][query_indices]).to(device)
    store_positions = [store.position[int(row)] for row in reference_rows]
    official_reference = torch.from_numpy(
        body["reference_official_embedding"][store_positions]
    ).to(device)
    official_error = max(
        float(torch.max(torch.abs(query_z.detach() - official_query))),
        float(torch.max(torch.abs(reference_z.detach() - official_reference))),
    )
    preserve = torch.mean(1 - torch.sum(query_z * official_query, dim=1)) + torch.mean(
        1 - torch.sum(reference_z * official_reference, dim=1)
    )
    bio_grad = torch.autograd.grad(bio_loss, parameters, retain_graph=True, allow_unused=True)
    generic_grad = torch.autograd.grad(generic_loss, parameters, retain_graph=True, allow_unused=True)
    gradient_alignment, bio_grad_norm, generic_grad_norm = gradient_cosine(bio_grad, generic_grad)
    loss = bio_loss + args.lambda_preserve * preserve
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm_before_clip = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
    if not np.isfinite(gradient_norm_before_clip):
        raise RuntimeError("non-finite B5 gradient")
    optimizer.step()

    model.eval()
    adapted_query = encode(model, query_tensor, device, args.batch_size, False)
    adapted_reference = encode(model, reference_tensor, device, args.batch_size, False)
    adapted_groups = candidate_scores(
        body, query_indices, adapted_query, adapted_reference, reference_position,
    )
    final_bio_losses = []
    final_margins = []
    for student, teacher, baseline_position in zip(
            adapted_groups, teacher_groups, baseline_positions, strict=True):
        target = F.softmax(teacher / args.teacher_temperature, dim=0)
        final_bio_losses.append(float(-torch.sum(target * F.log_softmax(
            student / args.student_temperature, dim=0,
        ))))
        final_margins.append(float(student[0] - student[baseline_position]))
    margin_delta = np.asarray(final_margins) - np.asarray(initial_margins)
    report = {
        "status": "bioaware_b5_gradient_injection_audit_complete",
        "formal": False,
        "arm": args.arm,
        "outer_fold": args.outer_fold,
        "corrected_actions": int(len(actions)),
        "corrected_identities": int(actions["truth_candidate_id"].nunique()),
        "corrected_formulas": int(actions["truth_formula"].nunique()),
        "official_replay_max_element_error": official_error,
        "bioaware_candidate_ce_before": float(bio_loss.detach()),
        "bioaware_candidate_ce_after": float(np.mean(final_bio_losses)),
        "bioaware_gradient_norm": bio_grad_norm,
        "generic_onehot_gradient_norm": generic_grad_norm,
        "bioaware_vs_generic_gradient_cosine": gradient_alignment,
        "optimizer_gradient_norm_before_clip": gradient_norm_before_clip,
        "capacity": capacity,
        "truth_vs_exact_baseline_wrong_margin_delta": {
            "mean": float(np.mean(margin_delta)),
            "median": float(np.median(margin_delta)),
            "supportive_fraction": float(np.mean(margin_delta > 0)),
            "minimum": float(np.min(margin_delta)),
        },
        "gates": {
            "official_replay": official_error <= 5e-4,
            "bioaware_gradient_finite_nonzero": np.isfinite(bio_grad_norm) and bio_grad_norm > 1e-8,
            "bioaware_gradient_not_generic_onehot": gradient_alignment < 0.99999,
            "teacher_candidate_ce_decreased": float(np.mean(final_bio_losses)) < float(bio_loss.detach()),
            "mean_exact_action_margin_increased": float(np.mean(margin_delta)) > 0,
            "median_exact_action_margin_increased": float(np.median(margin_delta)) > 0,
            "supportive_fraction_ge_0_60": float(np.mean(margin_delta > 0)) >= 0.60,
        },
        "contracts": {
            "teacher_input_is_candidate_level_nested_oof": True,
            "exact_baseline_wrong_candidate_used": True,
            "bioaware_logits_enter_loss": True,
            "generic_onehot_is_control_only": True,
            "shared_query_reference_encoder": True,
            "outer_held_formula_excluded": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.b4_manifest),
            "actions_sha256": sha256_file(action_path),
            "candidate_teacher_sha256": sha256_file(candidate_path),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "One-step gradient mechanism audit only; no retrieval improvement claim.",
    }
    report["pass_to_shared_embedding_pilot"] = all(report["gates"].values())
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass_to_shared_embedding_pilot"]:
        raise RuntimeError("B5 candidate-level gradient injection gate failed")


if __name__ == "__main__":
    main()

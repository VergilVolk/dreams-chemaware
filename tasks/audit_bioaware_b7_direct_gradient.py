#!/usr/bin/env python
"""Fail-closed one-step gate for direct BioAware shared-encoder gradients.

The graph prior selects nested-OOF correction boundaries.  The optimization
target is only the true molecular identity: no graph logit is distilled.
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
    parser.add_argument("--b4-manifest", type=Path, required=True)
    parser.add_argument("--action-router-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--audit-queries", type=int, default=8)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--lambda-listwise", type=float, default=0.5)
    parser.add_argument("--lambda-corrective", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def encode(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device,
           batch_size: int, grad: bool) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    context = torch.enable_grad if grad else torch.no_grad
    with context():
        for left in range(0, len(tensor), batch_size):
            outputs.append(model(tensor[left:left + batch_size].to(device)))
    return torch.cat(outputs)


def group_scores(
    body: dict[str, np.ndarray], query_indices: list[int], query_z: torch.Tensor,
    reference_z: torch.Tensor, reference_position: dict[int, int],
) -> list[torch.Tensor]:
    groups: list[torch.Tensor] = []
    for qpos, query in enumerate(query_indices):
        scores: list[torch.Tensor] = []
        for molecule in query_candidates(body, query):
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            positions = [
                reference_position[int(row)]
                for row in body["reference_rows"][left:right]
            ]
            scores.append(torch.max(reference_z[positions] @ query_z[qpos]))
        groups.append(torch.stack(scores))
    return groups


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    if args.audit_queries < 4:
        raise ValueError("at least four direct actions are required")
    action_path = args.action_router_dir / "action_routes.csv.gz"
    router_report_path = args.action_router_dir / "report.json"
    for path in (
        args.b4_manifest, action_path, router_report_path,
        args.official_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
    if (router_report.get("status") != "bioaware_b7_graph_action_router_frozen"
            or not router_report.get("pass_to_direct_gradient_gate")):
        raise RuntimeError("B7 action router is not eligible for direct injection")

    body = load_manifest(args.b4_manifest)
    actions = pd.read_csv(action_path)
    actions = actions[
        actions["arm"].eq("graph_prior_direct")
        & actions["outer_fold"].eq(args.outer_fold)
        & actions["corrected"].astype(bool)
    ].copy()
    actions = actions.sort_values("query_id", kind="stable").drop_duplicates(
        "truth_candidate_id", keep="first"
    ).head(args.audit_queries)
    if len(actions) < args.audit_queries:
        raise RuntimeError(f"only {len(actions)} identity-distinct corrective actions")
    query_position = {str(value): index for index, value in enumerate(body["query_id"])}
    try:
        query_indices = [query_position[str(value)] for value in actions["query_id"]]
    except KeyError as error:
        raise RuntimeError(f"action query absent from B4 manifest: {error.args[0]}") from error
    if any(int(body["formula_fold"][query]) == args.outer_fold for query in query_indices):
        raise RuntimeError("outer-held formula entered direct gradient audit")

    store = ManifestReferenceStore(body["reference_tensor_rows"], body["reference_tensor"])
    reference_rows: set[int] = set()
    baseline_positions: list[int] = []
    for query, action in zip(query_indices, actions.itertuples(index=False), strict=True):
        molecules = query_candidates(body, query)
        candidate_ids = [str(body["molecule_id"][molecule]) for molecule in molecules]
        if candidate_ids[0] != str(action.truth_candidate_id):
            raise RuntimeError(f"{action.query_id}: truth is not the unique first candidate")
        if str(action.proposed_candidate_id) != candidate_ids[0]:
            raise RuntimeError(f"{action.query_id}: selected graph action does not propose truth")
        if str(action.baseline_candidate_id) not in candidate_ids[1:]:
            raise RuntimeError(f"{action.query_id}: baseline wrong candidate is not in group")
        baseline_positions.append(candidate_ids.index(str(action.baseline_candidate_id)))
        for molecule in molecules:
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            reference_rows.update(map(int, body["reference_rows"][left:right]))

    reference_rows_array = np.asarray(sorted(reference_rows), dtype=np.int64)
    reference_position = {
        int(row): position for position, row in enumerate(reference_rows_array)
    }
    query_tensor = torch.from_numpy(body["query_tensor"][query_indices].astype(np.float32))
    reference_tensor = store.get(reference_rows_array)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal B7 direct-gradient audit requires CUDA")
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
    official_query = torch.from_numpy(
        body["query_official_embedding"][query_indices].astype(np.float32)
    ).to(device)
    store_positions = [store.position[int(row)] for row in reference_rows_array]
    official_reference = torch.from_numpy(
        body["reference_official_embedding"][store_positions].astype(np.float32)
    ).to(device)
    official_error = max(
        float(torch.max(torch.abs(query_z.detach() - official_query))),
        float(torch.max(torch.abs(reference_z.detach() - official_reference))),
    )
    if official_error > 5e-4:
        raise RuntimeError(f"official replay failed: max element error={official_error}")

    groups = group_scores(body, query_indices, query_z, reference_z, reference_position)
    listwise = torch.stack([
        -F.log_softmax(group / args.temperature, dim=0)[0]
        for group in groups
    ]).mean()
    corrective = torch.stack([
        F.softplus((args.rank_margin - (group[0] - torch.max(group[1:]))) / args.temperature)
        for group in groups
    ]).mean()
    preserve = (
        torch.mean(1 - torch.sum(query_z * official_query, dim=1))
        + torch.mean(1 - torch.sum(reference_z * official_reference, dim=1))
    )
    loss = (
        args.lambda_listwise * listwise
        + args.lambda_corrective * corrective
        + args.lambda_preserve * preserve
    )
    initial_loss = float(loss.detach())
    initial_margins = np.asarray([
        float((group[0] - group[position]).detach())
        for group, position in zip(groups, baseline_positions, strict=True)
    ])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
    if not np.isfinite(gradient_norm) or gradient_norm <= 1e-8:
        raise RuntimeError(f"direct gradient is invalid: {gradient_norm}")
    optimizer.step()

    model.eval()
    adapted_query = encode(model, query_tensor, device, args.batch_size, False)
    adapted_reference = encode(model, reference_tensor, device, args.batch_size, False)
    adapted_groups = group_scores(
        body, query_indices, adapted_query, adapted_reference, reference_position,
    )
    final_listwise = np.mean([
        float(-F.log_softmax(group / args.temperature, dim=0)[0])
        for group in adapted_groups
    ])
    final_corrective = np.mean([
        float(F.softplus((args.rank_margin - (group[0] - torch.max(group[1:]))) / args.temperature))
        for group in adapted_groups
    ])
    final_loss_without_preserve = (
        args.lambda_listwise * final_listwise
        + args.lambda_corrective * final_corrective
    )
    initial_loss_without_preserve = float(
        args.lambda_listwise * listwise.detach()
        + args.lambda_corrective * corrective.detach()
    )
    final_margins = np.asarray([
        float(group[0] - group[position])
        for group, position in zip(adapted_groups, baseline_positions, strict=True)
    ])
    margin_delta = final_margins - initial_margins
    preservation = float(torch.mean(torch.cat((
        torch.sum(adapted_query * official_query, dim=1),
        torch.sum(adapted_reference * official_reference, dim=1),
    ))))
    gates = {
        "official_replay": official_error <= 5e-4,
        "direct_gradient_finite_nonzero": np.isfinite(gradient_norm) and gradient_norm > 1e-8,
        "direct_truth_objective_decreased": final_loss_without_preserve < initial_loss_without_preserve,
        "mean_exact_wrong_margin_increased": float(margin_delta.mean()) > 0,
        "median_exact_wrong_margin_increased": float(np.median(margin_delta)) > 0,
        "supportive_fraction_ge_0_60": float(np.mean(margin_delta > 0)) >= 0.60,
        "one_step_preservation_ge_0_999": preservation >= 0.999,
    }
    # NumPy comparisons return np.bool_, which the standard JSON encoder does
    # not accept.  Normalize the complete gate map at the report boundary so a
    # successful GPU audit can never be lost while serializing its result.
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "bioaware_b7_direct_gradient_audit_complete",
        "formal": True,
        "outer_fold": args.outer_fold,
        "audit_queries": int(len(actions)),
        "audit_identities": int(actions["truth_candidate_id"].nunique()),
        "supervision": "direct one-hot truth identity; graph prior is router only",
        "official_maximum_element_error": official_error,
        "initial_objective": initial_loss,
        "initial_truth_objective_without_preservation": initial_loss_without_preserve,
        "final_truth_objective_without_preservation": float(final_loss_without_preserve),
        "gradient_norm_before_clip": gradient_norm,
        "exact_baseline_wrong_margin_delta": {
            "mean": float(margin_delta.mean()),
            "median": float(np.median(margin_delta)),
            "minimum": float(margin_delta.min()),
            "supportive_fraction": float(np.mean(margin_delta > 0)),
        },
        "one_step_preservation": preservation,
        "capacity": capacity,
        "gates": gates,
        "pass_to_direct_shared_embedding_pilot": bool(all(gates.values())),
        "contracts": {
            "graph_score_used_only_for_action_routing": True,
            "graph_score_or_probability_in_loss": False,
            "true_identity_directly_supervises_ranking": True,
            "shared_query_reference_encoder": True,
            "outer_held_formula_excluded": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.b4_manifest),
            "action_router_report_sha256": sha256_file(router_report_path),
            "actions_sha256": sha256_file(action_path),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "One-step training-mechanism gate; held-fold retrieval is required for an embedding-improvement claim.",
    }
    args.output_dir.mkdir(parents=True)
    serialized = json.dumps(report, indent=2)
    (args.output_dir / "report.json").write_text(
        serialized, encoding="utf-8"
    )
    print(serialized, flush=True)
    if not report["pass_to_direct_shared_embedding_pilot"]:
        raise RuntimeError("B7 direct-gradient gate failed; training was not authorized")


if __name__ == "__main__":
    main()

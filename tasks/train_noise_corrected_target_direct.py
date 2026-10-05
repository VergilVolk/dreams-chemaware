"""Minimal corrected-graph raw-target direct fine-tuning.

This trainer intentionally has no dependency on historical R0, old held
ledgers, teacher scores, matched controls, or PMT route labels.  Every batch is
query-complete.  Clean and target views optimize the same live molecule-max
identity boundary, and a stop-gradient action-to-clean term injects the raw
action direction into the deployed clean representation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from audit_noise_peak_gate_candidate_injection import formula_ci
from noise_corrected_fullgraph_evaluation import (
    full_metrics, official_scores, paired_outcome_table, score_embeddings,
)
from noise_final_direct_boundary_v2_core import (
    direct_action_to_clean_invariance, direct_target_boundary_objective,
)
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, stable_fold
from noise_v3_core import attenuate_sequence
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings, parse_path


@dataclass(frozen=True)
class ActionView:
    kind: str
    family: str
    path: tuple[int, ...] = ()
    attenuation: float = 0.0
    tensor_index: int = -1


@dataclass(frozen=True)
class QueryExample:
    query_index: int
    query_row: int
    formula: str
    actions: tuple[ActionView, ...]
    positive_rows: tuple[int, ...]
    negative_molecule_rows: tuple[tuple[int, ...], ...]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--action-bank-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, default=None)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--positive-references", type=int, default=3)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--references-per-negative", type=int, default=2)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--topk-negatives", type=int, default=8)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--lambda-clean", type=float, default=0.25)
    parser.add_argument("--lambda-action-conditioned-clean", type=float, default=1.0)
    parser.add_argument("--lambda-action-rank", type=float, default=0.5)
    parser.add_argument("--lambda-action-safety", type=float, default=0.25)
    parser.add_argument("--lambda-action-to-clean", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-preclip-gradient-norm", type=float, default=0.5)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--target-transfer-to-clean-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--transfer-scale-cap", type=float, default=8.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_references(
    graph: CandidateGraph, query: int, embeddings: np.ndarray,
    row_index: dict[int, int], positives: int, negatives: int, per_negative: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    _, rows, ptr, _ = graph.query_block(query)
    q = embeddings[row_index[int(graph.query_row[query])]]
    pair_score = embeddings[[row_index[int(row)] for row in rows]] @ q
    chosen: list[tuple[float, tuple[int, ...]]] = []
    for molecule, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        order = np.argsort(-pair_score[left:right], kind="stable") + left
        count = positives if molecule == 0 else per_negative
        selected = tuple(map(int, rows[order[:count]]))
        chosen.append((float(pair_score[order[0]]), selected))
    positive_rows = chosen[0][1]
    wrong = sorted(chosen[1:], key=lambda item: (-item[0], item[1]))[:negatives]
    return positive_rows, tuple(item[1] for item in wrong)


def build_examples(
    graph: CandidateGraph, actions: pd.DataFrame, embeddings: np.ndarray,
    row_index: dict[int, int], args: argparse.Namespace,
) -> list[QueryExample]:
    output = []
    for query, group in actions.groupby("query_index", sort=True):
        query = int(query)
        positive, negative = select_references(
            graph, query, embeddings, row_index, args.positive_references,
            args.negative_molecules, args.references_per_negative,
        )
        action_values = []
        for row in group.sort_values(["selector", "step"], kind="stable").itertuples():
            kind = str(getattr(row, "action_kind", "attenuation_path"))
            if kind == "attenuation_path":
                action_values.append(ActionView(
                    kind=kind, family=str(row.selector), path=parse_path(row.target_path),
                    attenuation=float(row.attenuation),
                ))
            elif kind == "precomputed_spectrum":
                action_values.append(ActionView(
                    kind=kind, family=str(row.selector), tensor_index=int(row.action_tensor_index),
                ))
            else:
                raise RuntimeError(f"unregistered action view kind: {kind}")
        action = tuple(action_values)
        if not action:
            continue
        output.append(QueryExample(
            query_index=query, query_row=int(graph.query_row[query]),
            formula=str(graph.query_formula[query]), actions=action,
            positive_rows=positive, negative_molecule_rows=negative,
        ))
    return output


def evaluate_query_subset(
    graph: CandidateGraph, examples: list[QueryExample], rows: np.ndarray,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    index = {int(row): position for position, row in enumerate(rows)}
    output = []
    for example in examples:
        _, candidate_rows, ptr, _ = graph.query_block(example.query_index)
        query = embeddings[index[example.query_row]]
        candidate = embeddings[[index[int(row)] for row in candidate_rows]]
        scores = np.maximum.reduceat(candidate @ query, ptr[:-1])
        rank = 1 + int(np.sum(scores[1:] >= scores[0]))
        output.append({
            "query_index": example.query_index, "query_formula": example.formula,
            "rank": rank, "margin": float(scores[0] - np.max(scores[1:])),
        })
    return pd.DataFrame(output)


def encode_query_batch(
    model: torch.nn.Module, store: SpectrumStore, examples: list[QueryExample],
    anchor: dict[int, np.ndarray], device: torch.device, args: argparse.Namespace,
    action_spectra: np.ndarray | None = None,
    transfer_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    spectra: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []
    preserve_rows: list[int] = []
    preserve_indices: list[int] = []
    for example in examples:
        clean_index = len(spectra)
        spectra.append(store.one(example.query_row))
        preserve_rows.append(example.query_row); preserve_indices.append(clean_index)
        action_indices = []
        for action in example.actions:
            action_indices.append(len(spectra))
            if action.kind == "attenuation_path":
                spectra.append(attenuate_sequence(
                    store.one(example.query_row), action.path, action.attenuation,
                ))
            elif action.kind == "precomputed_spectrum":
                if action_spectra is None or not 0 <= action.tensor_index < len(action_spectra):
                    raise RuntimeError("precomputed action tensor is unavailable or out of bounds")
                spectra.append(torch.from_numpy(np.asarray(
                    action_spectra[action.tensor_index], dtype=np.float32,
                )))
            else:
                raise RuntimeError(f"unregistered action view kind: {action.kind}")
        positive_indices = []
        for row in example.positive_rows:
            positive_indices.append(len(spectra)); spectra.append(store.one(row))
            preserve_rows.append(row); preserve_indices.append(positive_indices[-1])
        negative_indices = []
        for molecule_rows in example.negative_molecule_rows:
            local = []
            for row in molecule_rows:
                local.append(len(spectra)); spectra.append(store.one(row))
                preserve_rows.append(row); preserve_indices.append(local[-1])
            negative_indices.append(local)
        layout.append({
            "clean": clean_index, "actions": action_indices,
            "action_families": [action.family for action in example.actions],
            "positive": positive_indices, "negative": negative_indices,
        })
    encoded = forward_embeddings(model, torch.stack(spectra).to(device), args.amp)
    clean_margin: list[torch.Tensor] = []
    action_margin: list[list[torch.Tensor]] = []
    clean_embedding: list[torch.Tensor] = []
    action_embedding: list[list[torch.Tensor]] = []
    action_group: list[list[str]] = []
    for item in layout:
        positive = encoded[item["positive"]]
        negatives = [encoded[index] for index in item["negative"]]
        clean = encoded[int(item["clean"])]
        actions = [encoded[int(index)] for index in item["actions"]]
        def margins(vector: torch.Tensor) -> torch.Tensor:
            pos = torch.max(positive @ vector)
            return torch.stack([pos - torch.max(negative @ vector) for negative in negatives])
        clean_margin.append(margins(clean))
        action_margin.append([margins(action) for action in actions])
        clean_embedding.append(clean)
        action_embedding.append(actions)
        action_group.append(list(map(str, item["action_families"])))
    target = direct_target_boundary_objective(
        clean_margin, action_margin, rank_margin=args.rank_margin,
        rank_temperature=args.temperature, topk_negatives=args.topk_negatives,
        action_safety_slack=args.margin_floor_slack, lambda_clean=args.lambda_clean,
        lambda_action_conditioned_clean=args.lambda_action_conditioned_clean,
        lambda_action_rank=args.lambda_action_rank,
        lambda_action_safety=args.lambda_action_safety,
        action_group=action_group,
    )
    transfer = direct_action_to_clean_invariance(
        clean_embedding, action_embedding, action_group=action_group,
    )
    anchor_tensor = torch.as_tensor(
        np.stack([anchor[int(row)] for row in preserve_rows]),
        device=device, dtype=encoded.dtype,
    )
    preserve = 1.0 - torch.sum(encoded[preserve_indices] * anchor_tensor, dim=1)
    preserve_loss = preserve.mean()
    components = {
        "clean_boundary": (
            args.lambda_clean * target.clean_rank
            + args.lambda_action_conditioned_clean * target.action_conditioned_clean_rank
        ),
        "action_rank": args.lambda_action_rank * target.action_rank,
        "action_safety": args.lambda_action_safety * target.action_safety,
        "action_to_clean": args.lambda_action_to_clean * transfer,
        "preserve": args.lambda_preserve * preserve_loss,
    }
    loss = sum(
        value * (transfer_multiplier if name == "action_to_clean" else 1.0)
        for name, value in components.items()
    )
    return loss, {
        "loss": float(loss.detach()), "clean_rank": float(target.clean_rank.detach()),
        "conditioned_clean_rank": float(target.action_conditioned_clean_rank.detach()),
        "action_rank": float(target.action_rank.detach()),
        "action_safety": float(target.action_safety.detach()),
        "action_to_clean": float(transfer.detach()), "preserve": float(preserve_loss.detach()),
        "queries": float(target.queries), "actions": float(target.actions),
    }, components


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(np.sqrt(sum(
        float(torch.sum(gradient.detach().float() ** 2))
        for gradient in gradients if gradient is not None
    )))


def frozen_gradient_calibration(
    model: torch.nn.Module, store: SpectrumStore, examples: list[QueryExample],
    anchor: dict[int, np.ndarray], trainable: list[torch.nn.Parameter],
    device: torch.device, args: argparse.Namespace,
    action_spectra: np.ndarray | None = None,
) -> tuple[float, float, dict[str, object]]:
    if (
        args.calibration_batches < 1 or args.target_preclip_gradient_norm <= 0
        or args.target_transfer_to_clean_gradient_ratio < 0
        or args.transfer_scale_cap < 1
    ):
        raise ValueError("gradient calibration dimensions must be positive")
    rng = np.random.default_rng(args.seed + 991)
    order = rng.permutation(len(examples))
    batches = [
        [examples[int(index)] for index in order[left:left + args.batch_queries]]
        for left in range(0, min(len(order), args.calibration_batches * args.batch_queries), args.batch_queries)
    ]
    norms: dict[str, list[float]] = {name: [] for name in (
        "total", "clean_boundary", "action_rank", "action_safety",
        "action_to_clean", "preserve",
    )}
    for batch in batches:
        loss, _, components = encode_query_batch(
            model, store, batch, anchor, device, args, action_spectra=action_spectra,
        )
        norms["total"].append(gradient_norm(loss, trainable))
        for name, component in components.items():
            norms[name].append(gradient_norm(component, trainable))
        model.zero_grad(set_to_none=True)
    total_p90 = float(np.quantile(norms["total"], 0.90))
    if not np.isfinite(total_p90) or total_p90 <= 0:
        raise RuntimeError("calibration total gradient is zero or non-finite")
    summary = {
        name: {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(np.max(values)),
        }
        for name, values in norms.items()
    }
    clean_median = float(summary["clean_boundary"]["median"])
    transfer_median = float(summary["action_to_clean"]["median"])
    if clean_median <= 0:
        raise RuntimeError("clean calibration gradient is zero")
    if args.lambda_action_to_clean == 0 or args.target_transfer_to_clean_gradient_ratio == 0:
        transfer_multiplier = 0.0
    else:
        if transfer_median <= 0:
            raise RuntimeError("enabled action-to-clean calibration gradient is zero")
        transfer_multiplier = min(
            float(args.transfer_scale_cap),
            float(args.target_transfer_to_clean_gradient_ratio) * clean_median / transfer_median,
        )
    # Triangle inequality over the observed per-branch maxima is conservative:
    # every calibration batch remains below this bound regardless of gradient
    # alignment.  The multiplier and global scale are frozen before updates.
    conservative_maximum = sum(
        float(summary[name]["maximum"])
        * (transfer_multiplier if name == "action_to_clean" else 1.0)
        for name in ("clean_boundary", "action_rank", "action_safety", "action_to_clean", "preserve")
    )
    scale = min(
        1.0, float(args.target_preclip_gradient_norm) / conservative_maximum,
    )
    return scale, transfer_multiplier, {
        "batches": len(batches), "query_complete": True,
        "branch_norms_after_configured_weights": summary,
        "target_preclip_gradient_norm": float(args.target_preclip_gradient_norm),
        "target_transfer_to_clean_gradient_ratio": float(
            args.target_transfer_to_clean_gradient_ratio
        ),
        "effective_transfer_multiplier": float(transfer_multiplier),
        "expected_transfer_to_clean_gradient_ratio": float(
            transfer_multiplier * transfer_median / clean_median
        ),
        "conservative_combined_gradient_maximum": float(conservative_maximum),
        "effective_global_loss_scale": float(scale),
        "scale_can_exceed_one": False, "outcomes_used": False,
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    graph_report_path = args.graph_dir / "report.json"
    action_path = args.action_bank_dir / "training_actions.csv.gz"
    action_report_path = args.action_bank_dir / "report.json"
    source_manifest_path = args.source_manifest_dir / "manifest.npz"
    for path in (graph_path, cache_path, graph_report_path, action_path, action_report_path,
                 source_manifest_path, args.data, args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    action_report = json.loads(action_report_path.read_text(encoding="utf-8"))
    accepted_action_status = {
        "noise_corrected_full_action_bank_complete",
        "noise_corrected_fixed_p_action_bank_complete",
        "noise_corrected_combined_np_action_bank_complete",
    }
    if (
        graph_report.get("formal_training_authorized") is not True
        or action_report.get("status") not in accepted_action_status
        or int(action_report.get("outer_formula_fold", -1)) != args.outer_fold
        or action_report.get("contracts", {}).get("action_outcomes_computed") is not False
        or (not args.smoke and action_report.get("formal_training_authorized") is not True)
    ):
        raise RuntimeError("corrected graph/action-bank contract failed")
    if not args.smoke and args.initial_student_checkpoint is None:
        raise RuntimeError("formal corrected training must start from the fold-aligned mature E4 checkpoint")
    graph = CandidateGraph(graph_path)
    official_rows, official_embeddings, official_index = load_embedding_cache(cache_path)
    actions = pd.read_csv(action_path, low_memory=False)
    required = {"action_id", "query_index", "query_row", "query_formula", "formula_fold",
                "selector", "attenuation", "step", "target_path"}
    if required - set(actions.columns) or actions.action_id.duplicated().any():
        raise RuntimeError("full action bank schema or uniqueness failed")
    action_spectra = None
    if "action_kind" in actions.columns:
        kinds = set(actions.action_kind.astype(str))
        if not kinds <= {"attenuation_path", "precomputed_spectrum"}:
            raise RuntimeError(f"unregistered action kinds: {sorted(kinds)}")
        if "precomputed_spectrum" in kinds:
            spectrum_path = args.action_bank_dir / "action_spectra.npz"
            if not spectrum_path.is_file():
                raise FileNotFoundError(spectrum_path)
            with np.load(spectrum_path, allow_pickle=False) as body:
                if set(body.files) != {"action_ids", "action_spectra"}:
                    raise RuntimeError("precomputed action spectrum artifact schema failed")
                stored_ids = np.asarray(body["action_ids"], dtype=str)
                action_spectra = np.asarray(body["action_spectra"], dtype=np.float32)
            if action_spectra.ndim != 3 or action_spectra.shape[1:] != (args.n_highest_peaks + 1, 2):
                raise RuntimeError("precomputed action tensor shape failed")
            precomputed = actions.action_kind.astype(str).eq("precomputed_spectrum").to_numpy()
            indices = actions.loc[precomputed, "action_tensor_index"].to_numpy(np.int64)
            if (
                len(stored_ids) != len(action_spectra)
                or np.any(indices < 0) or np.any(indices >= len(action_spectra))
                or not np.array_equal(
                    actions.loc[precomputed, "action_id"].astype(str).to_numpy(), stored_ids[indices]
                )
            ):
                raise RuntimeError("precomputed action tensor index/ID alignment failed")
            expected_sha = action_report.get("provenance", {}).get("action_spectra_sha256")
            if expected_sha != sha256_file(spectrum_path):
                raise RuntimeError("precomputed action spectrum hash failed")
    query = actions.query_index.to_numpy(np.int64)
    if np.any(actions.formula_fold.to_numpy(np.int8) == args.outer_fold):
        raise RuntimeError("outer-held formula leaked into action bank")
    if not np.array_equal(actions.query_row.to_numpy(np.int64), graph.query_row[query]):
        raise RuntimeError("action query rows differ from graph")
    device = torch.device(args.device)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    initialization_provenance: dict[str, object] = {"kind": initialization}
    expected_model = action_report.get("model_provenance", {})
    if args.initial_student_checkpoint is not None:
        package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
        if (
            package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or package.get("inference_clean_only") is not True
            or package.get("P2b_used") is not False
            or int(package.get("outer_fold", -1)) != args.outer_fold
            or expected_model.get("initial_student_checkpoint_sha256")
            != sha256_file(args.initial_student_checkpoint)
        ):
            raise RuntimeError("initial checkpoint differs from action-bank geometry")
        model.load_state_dict(package["model_state"], strict=True)
        initialization_provenance = {
            "kind": "mature_e4_current_geometry",
            "checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
            "historical_held_ledger_reused": False,
        }
    elif expected_model.get("official_checkpoint_sha256") != sha256_file(args.official_checkpoint):
        raise RuntimeError("official initialization differs from action-bank geometry")
    folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula
    ], dtype=np.int8)
    held_queries = np.flatnonzero(folds == args.outer_fold)
    if args.smoke:
        needed = set(map(int, actions.query_row))
        for value in np.unique(query):
            _, candidate_rows, _, _ = graph.query_block(int(value))
            needed.update(map(int, candidate_rows))
        reachable = np.asarray(sorted(needed), dtype=np.int64)
    else:
        reachable = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    if args.initial_student_checkpoint is None:
        initial_encoded = np.stack([
            official_embeddings[official_index[int(row)]] for row in reachable
        ]).astype(np.float32, copy=False)
        probe_rows = reachable[:min(32, len(reachable))]
        probe_encoded = encode_rows(
            model, store, probe_rows, device, args.eval_batch_size, args.amp,
            "corrected-target-init-probe",
        )
        probe_official = np.stack([
            official_embeddings[official_index[int(row)]] for row in probe_rows
        ])
        probe_cosine = np.einsum("ij,ij->i", probe_encoded, probe_official)
        if float(np.quantile(probe_cosine, 0.01)) < 0.999:
            raise RuntimeError("official forward/cache initialization probe failed")
        initialization_provenance["official_cache_probe_p01_cosine"] = float(
            np.quantile(probe_cosine, 0.01)
        )
    else:
        initial_encoded = encode_rows(
            model, store, reachable, device, args.eval_batch_size, args.amp,
            "corrected-target-init",
        )
    initial_index = {int(row): index for index, row in enumerate(reachable)}
    anchor = {int(row): initial_encoded[index] for index, row in enumerate(reachable)}
    examples = build_examples(graph, actions, initial_encoded, initial_index, args)
    if not examples:
        raise RuntimeError("no query-complete target examples")
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    effective_loss_scale, effective_transfer_multiplier, gradient_calibration = frozen_gradient_calibration(
        model, store, examples, anchor, trainable, device, args,
        action_spectra=action_spectra,
    )
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(examples))
        total: dict[str, float] = {}
        steps = 0
        for left in range(0, len(order), args.batch_queries):
            batch = [examples[int(index)] for index in order[left:left + args.batch_queries]]
            loss, parts, _ = encode_query_batch(
                model, store, batch, anchor, device, args,
                action_spectra=action_spectra,
                transfer_multiplier=effective_transfer_multiplier,
            )
            scaled_loss = effective_loss_scale * loss
            optimizer.zero_grad(set_to_none=True)
            scaled_loss.backward()
            unclipped = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
            optimizer.step()
            for key, value in parts.items():
                total[key] = total.get(key, 0.0) + float(value)
            total["unclipped_gradient_norm"] = total.get("unclipped_gradient_norm", 0.0) + unclipped
            total["clip_fraction"] = total.get("clip_fraction", 0.0) + float(unclipped > args.grad_clip)
            steps += 1
        epoch_report = {key: value / steps for key, value in total.items()} | {"epoch": epoch + 1, "steps": steps}
        history.append(epoch_report)
        print(f"[corrected target direct] {json.dumps(epoch_report)}", flush=True)
    final_rows = reachable
    final_encoded = encode_rows(
        model, store, final_rows, device, args.eval_batch_size, args.amp,
        "corrected-target-final",
    )
    evaluation: dict[str, object]
    paired = None
    if args.smoke:
        initial_probe = np.stack([anchor[int(row)] for row in final_rows])
        movement = np.einsum("ij,ij->i", initial_probe, final_encoded)
        initial_query = evaluate_query_subset(graph, examples, reachable, initial_encoded)
        final_query = evaluate_query_subset(graph, examples, final_rows, final_encoded)
        paired_query = initial_query.merge(
            final_query, on=["query_index", "query_formula"], suffixes=("_initial", "_final"),
            validate="one_to_one",
        )
        corrected = (paired_query.rank_initial != 1) & (paired_query.rank_final == 1)
        introduced = (paired_query.rank_initial == 1) & (paired_query.rank_final != 1)
        evaluation = {
            "smoke_only": True, "reachable_rows": int(len(reachable)),
            "movement_probe_rows": int(len(final_rows)),
            "mean_initial_final_cosine": float(np.mean(movement)),
            "minimum_initial_final_cosine": float(np.min(movement)),
            "clean_query_transfer_gate": {
                "queries": int(len(paired_query)),
                "initial_recall1": float(np.mean(paired_query.rank_initial == 1)),
                "final_recall1": float(np.mean(paired_query.rank_final == 1)),
                "delta_recall1_pp": float(100.0 * np.mean(
                    (paired_query.rank_final == 1).astype(float)
                    - (paired_query.rank_initial == 1).astype(float)
                )),
                "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
                "risk_net": int(corrected.sum() - 2 * introduced.sum()),
                "mean_margin_change": float(np.mean(
                    paired_query.margin_final - paired_query.margin_initial
                )),
                "all_metrics_are_on_clean_input": True,
            },
        }
    else:
        initial_score = score_embeddings(graph, reachable, initial_encoded)
        candidate_score = score_embeddings(graph, reachable, final_encoded)
        with np.load(source_manifest_path, allow_pickle=False) as source:
            query_adduct = np.asarray(source["query_adduct"], dtype=str)
        official_metric, official_query = full_metrics(
            graph, official_scores(graph), query_adduct, held_queries,
        )
        initial_metric, initial_query = full_metrics(
            graph, initial_score, query_adduct, held_queries,
        )
        final_metric, final_query = full_metrics(
            graph, candidate_score, query_adduct, held_queries,
        )
        paired = paired_outcome_table(official_query, final_query)
        initial_paired = paired_outcome_table(initial_query, final_query)
        paired["initial_e4_rank"] = initial_paired["official_rank"].to_numpy(np.int64)
        paired["corrected_vs_initial_e4"] = initial_paired["corrected"].to_numpy(bool)
        paired["introduced_vs_initial_e4"] = initial_paired["introduced"].to_numpy(bool)
        paired["net_vs_initial_e4"] = initial_paired["risk_net"].to_numpy(np.int8)
        for name in (
            "reciprocal_rank", "macro_query_auc", "macro_query_auprc",
            "positive_vs_best_negative_margin", "top1_top2_gap",
        ):
            paired[f"initial_e4_{name}"] = initial_paired[f"official_{name}"].to_numpy(
                np.float64
            )
        official_ci = formula_ci(paired, args.bootstrap_resamples, args.seed)
        initial_ci_frame = paired.assign(
            corrected=paired["corrected_vs_initial_e4"],
            introduced=paired["introduced_vs_initial_e4"],
        )
        initial_ci = formula_ci(initial_ci_frame, args.bootstrap_resamples, args.seed)
        near_initial_ci = formula_ci(
            initial_ci_frame.loc[initial_ci_frame.near].reset_index(drop=True),
            args.bootstrap_resamples,
            args.seed + 1,
        )
        corrected_e4 = int(initial_paired.corrected.sum())
        introduced_e4 = int(initial_paired.introduced.sum())
        evaluation = {
            "comparison_contract": {
                "primary_baseline": "exact_initial_e4_checkpoint_on_same_held_graph",
                "secondary_baseline": "official_embedding_slim_on_same_held_graph",
                "all_metrics_use_clean_input": True,
                "risk_net_lambda": 2,
            },
            "official": official_metric,
            "initial_e4": initial_metric,
            "candidate": final_metric,
            "candidate_vs_initial_e4": {
                "corrected": corrected_e4,
                "introduced": introduced_e4,
                "net_corrected_minus_introduced": corrected_e4 - introduced_e4,
                "risk_net_lambda2": corrected_e4 - 2 * introduced_e4,
                "formula_cluster_delta_recall1": initial_ci,
                "near_formula_cluster_delta_recall1": near_initial_ci,
            },
            "candidate_vs_official": {
                "corrected": int(paired.corrected.sum()),
                "introduced": int(paired.introduced.sum()),
                "net_corrected_minus_introduced": int(paired.risk_net.sum()),
                "risk_net_lambda2": int(paired.corrected.sum() - 2 * paired.introduced.sum()),
                "formula_cluster_delta_recall1": official_ci,
            },
        }
    report = {
        "status": "noise_corrected_target_direct_complete",
        "formal": bool(not args.smoke), "outer_formula_fold": args.outer_fold,
        "examples": {"queries": len(examples), "actions": int(sum(len(item.actions) for item in examples))},
        "initialization": initialization_provenance, "capacity": capacity,
        "gradient_calibration": gradient_calibration,
        "history": history, "evaluation": evaluation,
        "contracts": {
            "historical_r0_used": False, "historical_held_ledger_used": False,
            "teacher_or_distillation_target_used": False, "matched_control_training_gate": False,
            "query_complete_query_equal": True, "molecule_max_boundary": True,
            "query_family_action_equal": True,
            "action_to_clean_stop_gradient_direction": True,
            "outer_held_formula_actions_seen": False, "inference_clean_only": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "runtime_seconds": time.time() - started,
        "provenance": ({
            "candidate_graph_sha256": sha256_file(graph_path),
            "embedding_cache_sha256": sha256_file(cache_path),
            "action_bank_report_sha256": sha256_file(action_report_path),
            "action_bank_sha256": sha256_file(action_path),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        } | ({
            "action_spectra_sha256": sha256_file(args.action_bank_dir / "action_spectra.npz"),
        } if action_spectra is not None else {})),
        "claim_limit": (
            "Local execution/gradient smoke only; no performance claim."
            if args.smoke else "Train-side formula-OOF development result; not a P3 test result."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_corrected_train_", dir=args.output_dir.parent))
    try:
        torch.save({
            "status": "noise_corrected_target_direct_shared_encoder",
            "model_state": model.state_dict(), "outer_fold": args.outer_fold,
            "inference_clean_only": True, "P2b_used": False,
        }, staging / "final_shared_encoder.pt")
        if paired is not None:
            paired.to_csv(staging / "held_per_query.csv.gz", index=False, compression="gzip")
        (staging / "decision.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

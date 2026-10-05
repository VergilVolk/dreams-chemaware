"""Learn a bounded shared rule metric from validated cross-view events.

The official DreaMS and mass channels are frozen.  Only non-negative weights
on the 316 whitened chemical-rule coordinates are learned.  Cross-view action
outcomes are privileged training-time routing information; inference uses one
clean spectrum and the same PSD map for query and reference spectra.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from audit_chemaware_whitened_centered_rule_kernel import FourCenterCache, TransformCache
from chemaware_crossview_consensus_core import crossview_policy
from chemaware_crossview_rankmax_core import (
    boundary_coordinate_gradients,
    boundary_symmetric_operator_gradients,
    gradient_coherence_report,
    gradient_transfer_report,
    identity_balanced_weights,
    identity_risk_route,
    matched_error_identities,
    rankmax_top1_loss,
)
from chemaware_formula_rule_core import rows_for_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_shrinkage_whitening_core import fit_shrinkage_whitener
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("restore_only", "matched_random", "alignment_permuted", "correct")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--parent-report", type=Path, default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/report.json")
    parser.add_argument("--parent-consensus-ranks", type=Path, default=ROOT / "data/validation/chemaware_parent_crossview_consensus_v1/confirmation_ranks.npz")
    parser.add_argument("--repeat-ledger", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v3/repeat_ledger.npz")
    parser.add_argument("--consensus-report", type=Path, default=ROOT / "data/validation/chemaware_crossview_consensus_teacher_v2/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_crossview_rankmax_rule_metric_v1")
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--final-fit-identities", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-margin", type=float, default=0.005)
    parser.add_argument("--maximum-coordinate-change", type=float, default=0.50)
    parser.add_argument("--restore-weight", type=float, default=1.0)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--safety-slack", type=float, default=0.002)
    parser.add_argument("--preserve-weight", type=float, default=0.10)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_identity(
    selected_slot: np.ndarray, candidate_identity: np.ndarray,
) -> np.ndarray:
    output = np.full(len(selected_slot), "", dtype="U14")
    active = np.flatnonzero(np.asarray(selected_slot) >= 0)
    output[active] = candidate_identity[active, selected_slot[active].astype(np.int64)]
    return output


def compile_fixed_representatives(
    panel: np.ndarray,
    body: dict[str, np.ndarray],
    scored: dict[str, np.ndarray],
    cache: TransformCache,
    variant: str,
    mass_beta: float,
    rule_beta: float,
) -> dict[str, np.ndarray]:
    max_molecules = int(max(len(value) for value in scored["labels"]))
    rule_dimension = len(np.asarray(cache.get(int(body["query_row"][panel[0]]))[variant]))
    fixed = np.full((len(panel), max_molecules), -1e4, dtype=np.float64)
    product = np.zeros((len(panel), max_molecules, rule_dimension), dtype=np.float16)
    valid = np.zeros((len(panel), max_molecules), dtype=bool)
    positive = np.empty(len(panel), dtype=np.int16)
    reference_row = np.full((len(panel), max_molecules), -1, dtype=np.int64)
    parent_rank = np.empty(len(panel), dtype=np.int16)
    parent_margin = np.empty(len(panel), dtype=np.float32)
    for index, query in enumerate(map(int, panel)):
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        labels = np.asarray(scored["labels"][index], dtype=bool)
        qrow = int(body["query_row"][query])
        qrule = np.asarray(cache.get(qrow)[variant], dtype=np.float32)
        left, right = map(int, body["query_ptr"][query : query + 2])
        edge_left = int(body["molecule_ptr"][left])
        reference_rows = body["pair_candidate_row"][edge_left:int(body["molecule_ptr"][right])]
        global_pair = np.asarray(scored["global"][index], dtype=np.float64)
        mass_pair = np.asarray(scored["mass"][index], dtype=np.float64)
        rule_pair = np.asarray(scored[variant][index], dtype=np.float64)
        parent_pair = global_pair + float(mass_beta) * mass_pair + float(rule_beta) * rule_pair
        molecule_score = np.empty(len(labels), dtype=np.float64)
        for molecule, (start, stop) in enumerate(zip(pointer[:-1], pointer[1:])):
            winner = int(start + np.argmax(parent_pair[start:stop]))
            row = int(reference_rows[winner])
            reference_row[index, molecule] = row
            # Store the frozen parent score itself.  Training adds only the
            # weighted-rule residual around w=1; this avoids reconstructing a
            # near-tied parent boundary with a different accumulation order.
            fixed[index, molecule] = parent_pair[winner]
            rrule = np.asarray(cache.get(row)[variant], dtype=np.float32)
            product[index, molecule] = (qrule * rrule).astype(np.float16)
            molecule_score[molecule] = parent_pair[winner]
            valid[index, molecule] = True
        pos = int(np.flatnonzero(labels)[0]); positive[index] = pos
        parent_rank[index] = strict_rank(molecule_score, labels)
        parent_margin[index] = molecule_score[pos] - float(np.max(molecule_score[~labels]))
        if (index + 1) % 1024 == 0:
            print(f"compiled Rankmax representatives {index + 1}/{len(panel)}", flush=True)
    return {
        "fixed": fixed, "product": product, "valid": valid, "positive": positive,
        "reference_row": reference_row,
        "parent_rank": parent_rank, "parent_margin": parent_margin,
    }


def scores_from_weights(
    tensors: dict[str, np.ndarray], weights: np.ndarray, rule_beta: float,
) -> np.ndarray:
    return (
        tensors["fixed"].astype(np.float64)
        + float(rule_beta) * np.einsum(
            "qmd,d->qm", tensors["product"].astype(np.float64),
            np.asarray(weights, dtype=np.float64) - 1.0, optimize=True,
        )
    )


def ranks_from_scores(scores: np.ndarray, valid: np.ndarray, positive: np.ndarray) -> np.ndarray:
    output = np.empty(len(scores), dtype=np.int16)
    for index in range(len(scores)):
        pos = int(positive[index]); value = float(scores[index, pos])
        mask = valid[index].copy(); mask[pos] = False
        output[index] = 1 + int(np.sum(scores[index, mask] >= value))
    return output


def train_arm(
    tensors: dict[str, np.ndarray],
    active_weight: np.ndarray,
    restore_weight: np.ndarray,
    safety_mask: np.ndarray,
    train_mask: np.ndarray,
    rule_beta: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dimension = tensors["product"].shape[-1]
    delta = torch.nn.Parameter(torch.zeros(dimension))
    optimizer = torch.optim.AdamW(
        [delta], lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    active_index = np.flatnonzero(active_weight > 0)
    restore_index = np.flatnonzero(restore_weight > 0)
    safety_index = np.flatnonzero(train_mask & safety_mask)
    if not len(restore_index) or not len(safety_index):
        raise RuntimeError("Rankmax restoration or safety pool is empty")
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        safety_draw = rng.choice(
            safety_index, size=min(args.batch_size, len(safety_index)), replace=False,
        )
        index = np.unique(np.concatenate((active_index, restore_index, safety_draw)))
        fixed = torch.from_numpy(tensors["fixed"][index])
        product = torch.from_numpy(tensors["product"][index].astype(np.float64))
        valid = torch.from_numpy(tensors["valid"][index])
        pos_index = torch.from_numpy(tensors["positive"][index].astype(np.int64))
        coordinate = 1.0 + args.maximum_coordinate_change * torch.tanh(delta)
        score = fixed + float(rule_beta) * torch.einsum(
            "bmd,d->bm", product, coordinate.to(torch.float64) - 1.0,
        )
        row = torch.arange(len(index)); positive = score[row, pos_index]
        negative_mask = valid.clone(); negative_mask[row, pos_index] = False
        negative = score.masked_fill(~negative_mask, -1e4)
        active = torch.from_numpy(active_weight[index])
        restore = torch.from_numpy(restore_weight[index])
        event_loss = rankmax_top1_loss(
            positive, negative, active, args.target_margin,
        )
        repair_loss = rankmax_top1_loss(
            positive, negative, restore, args.target_margin,
        )
        current_margin = positive - torch.max(negative, dim=1).values
        safe = torch.from_numpy(np.isin(index, safety_draw))
        target = torch.from_numpy(tensors["parent_margin"][index]) - args.safety_slack
        safety = torch.relu(target[safe] - current_margin[safe]).mean()
        preserve = torch.mean((coordinate - 1.0) ** 2)
        loss = event_loss + args.restore_weight * repair_loss + args.safety_weight * safety + args.preserve_weight * preserve
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_([delta], 5.0)
        optimizer.step()
        coordinate = 1.0 + args.maximum_coordinate_change * torch.tanh(delta.detach())
        item = {
            "epoch": epoch + 1, "loss": float(loss.detach()),
            "event": float(event_loss.detach()), "restore": float(repair_loss.detach()),
            "safety": float(safety.detach()), "gradient": float(gradient),
            "coordinate_min": float(coordinate.min()), "coordinate_max": float(coordinate.max()),
            "coordinate_rms_change": float(torch.sqrt(torch.mean((coordinate - 1) ** 2))),
            "optimizer_steps": 1,
        }
        history.append(item)
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 20 == 0:
            print(json.dumps(item), flush=True)
    return coordinate.cpu().numpy().astype(np.float32), history


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.smoke:
        args.epochs = min(args.epochs, 5); args.bootstrap_draws = min(args.bootstrap_draws, 300)
    if not (0 < args.target_margin <= 0.02 and 0 < args.maximum_coordinate_change < 1):
        raise ValueError("event margin or coordinate trust region is invalid")
    torch.set_num_threads(max(1, args.torch_threads))
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    consensus_report = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    parent_setting = parent_report["fold2_selection"]["whitened"]["true"]
    consensus_setting = consensus_report["selection"]
    if parent_setting["variant"] != "whitened_true_s0p5":
        raise RuntimeError("unexpected frozen parent")
    if consensus_report.get("status") != "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_PASS":
        raise RuntimeError("validated cross-view teacher required")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.repeat_ledger, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    panel = ledger["query"].astype(np.int64)
    formula = ledger["formula"].astype(str); identity = ledger["identity"].astype(str)
    official_rank = ledger["baseline_rank"].astype(np.int64)
    role = stable_formula_folds(formula, 3, args.split_seed)
    train_mask = role < 2; confirmation_mask = role == 2
    if set(formula[train_mask]) & set(formula[confirmation_mask]):
        raise RuntimeError("training and confirmation formulas overlap")

    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    if not np.all(fold[panel] == 3):
        raise RuntimeError("repeat panel escaped inner fold 3")
    final_fit = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, (0, 1, 2))), body["query_ik14"],
        np.random.default_rng(args.fold_seed + 23), args.final_fit_identities,
    )
    if set(body["query_formula"][final_fit].astype(str)) & set(formula):
        raise RuntimeError("parent whitening fit leaked panel formulas")
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library,
        top_peaks=args.top_peaks, kernel_dim=args.kernel_dim,
        bin_width=args.bin_width, grid_offsets=args.grid_offsets,
        intensity_power=args.intensity_power, mass_shift_da=args.mass_shift_da,
        pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=args.rule_tolerance,
        rule_channel_weight=1.0,
    )
    raw = KernelCache(
        kernel_args, row_position,
        variants=(
            "mass", "rule_response", "rule_response_local_background_a",
            "rule_response_local_background_b", "rule_response_local_background_c",
        ),
    )
    centers = FourCenterCache(raw)
    fit_rows = rows_for_queries(final_fit, body)
    fit_feature = np.stack([
        np.asarray(centers.get(int(row))["true"], dtype=np.float32) for row in fit_rows
    ])
    mean, transform, transform_report = fit_shrinkage_whitener(fit_feature, 0.5)
    del fit_feature; gc.collect()
    variant = str(parent_setting["variant"])
    cache = TransformCache(centers, {variant: (mean, transform)}, {variant: "true"})
    scored = score_queries(panel, body, official, row_position, cache, ("mass", variant))
    if not np.array_equal(np.asarray(scored["old_rank"], dtype=np.int64), official_rank):
        raise RuntimeError("official repeat baseline did not replay")
    tensors = compile_fixed_representatives(
        panel, body, scored, cache, variant,
        float(parent_setting["mass_beta"]), float(parent_setting["rule_beta"]),
    )
    parent_rank = tensors["parent_rank"].astype(np.int64)
    with np.load(args.parent_consensus_ranks, allow_pickle=False) as frozen:
        expected = {int(query): int(rank) for query, rank in zip(frozen["query"], frozen["parent_rank"])}
    observed = np.asarray([parent_rank[i] for i in np.flatnonzero(confirmation_mask)])
    wanted = np.asarray([expected[int(panel[i])] for i in np.flatnonzero(confirmation_mask)])
    if not np.array_equal(observed, wanted):
        raise RuntimeError(f"parent fixed-representative replay mismatch: {np.sum(observed != wanted)}")

    selected = {}
    selected_name = {}
    for arm in ("correct", "alignment_permuted"):
        _rank, slot, _score, _support = crossview_policy(
            official_rank, ledger["candidate_identity"], ledger["candidate_valid"],
            ledger["proposal_rank"], ledger[f"{arm}_candidate_utility"], identity,
            threshold=float(consensus_setting["threshold"]),
            aggregation=str(consensus_setting["aggregation"]),
            min_context=int(consensus_setting["min_context"]),
        )
        selected[arm] = slot
        selected_name[arm] = selected_identity(slot, ledger["candidate_identity"])
    correct_identities, correct_audit = identity_risk_route(
        identity, train_mask, parent_rank, selected_name["correct"],
        selected_name["alignment_permuted"],
    )
    alignment_identities, alignment_audit = identity_risk_route(
        identity, train_mask, parent_rank, selected_name["alignment_permuted"],
        selected_name["correct"],
    )
    matched_identities = matched_error_identities(
        correct_identities, identity, formula, train_mask, parent_rank > 1,
        tensors["parent_margin"],
    )
    active_identity = {
        "restore_only": np.empty(0, dtype="U14"),
        "matched_random": matched_identities,
        "alignment_permuted": alignment_identities,
        "correct": correct_identities,
    }
    active_weight = {}
    active_mask = {}
    for arm, identities in active_identity.items():
        active = train_mask & (parent_rank > 1) & np.isin(identity, identities)
        active_mask[arm] = active
        active_weight[arm] = identity_balanced_weights(identity, active)
    restore = train_mask & (official_rank == 1) & (parent_rank > 1)
    restore_weight = identity_balanced_weights(identity, restore)
    safety = train_mask & (parent_rank == 1)

    boundary_gradient, top_negative = boundary_coordinate_gradients(
        tensors["fixed"], tensors["product"], tensors["valid"],
        tensors["positive"], float(parent_setting["rule_beta"]),
    )
    direct_correct = active_mask["correct"] & (selected_name["correct"] == identity)
    direct_alignment = active_mask["alignment_permuted"] & (
        selected_name["alignment_permuted"] == identity
    )
    confirmation_direct_correct = (
        confirmation_mask & (parent_rank > 1) & (selected_name["correct"] == identity)
    )
    gradient_diagnostic = {
        "correct_identity_expanded": gradient_coherence_report(
            boundary_gradient, identity, active_mask["correct"],
        ),
        "correct_direct_support_only": gradient_coherence_report(
            boundary_gradient, identity, direct_correct,
        ),
        "alignment_identity_expanded": gradient_coherence_report(
            boundary_gradient, identity, active_mask["alignment_permuted"],
        ),
        "alignment_direct_support_only": gradient_coherence_report(
            boundary_gradient, identity, direct_alignment,
        ),
        "matched_random": gradient_coherence_report(
            boundary_gradient, identity, active_mask["matched_random"],
        ),
        "correct_direct_train_to_confirmation": gradient_transfer_report(
            boundary_gradient, identity, direct_correct, confirmation_direct_correct,
        ),
        "support_expansion": {
            "expanded_queries": int(np.sum(active_mask["correct"])),
            "direct_supported_queries": int(np.sum(direct_correct)),
            "unsupported_queries": int(np.sum(
                active_mask["correct"] & (selected_name["correct"] == "")
            )),
            "wrong_candidate_selected_queries": int(np.sum(
                active_mask["correct"] & (selected_name["correct"] != "")
                & (selected_name["correct"] != identity)
            )),
        },
    }
    operator_scope = (
        active_mask["correct"] | active_mask["alignment_permuted"]
        | active_mask["matched_random"] | confirmation_direct_correct
    )
    operator_index = np.flatnonzero(operator_scope)
    query_rule = np.stack([
        np.asarray(cache.get(int(body["query_row"][int(panel[index])]))[variant], dtype=np.float32)
        for index in operator_index
    ])
    positive_rule = np.stack([
        np.asarray(cache.get(int(tensors["reference_row"][index, tensors["positive"][index]]))[variant], dtype=np.float32)
        for index in operator_index
    ])
    negative_rule = np.stack([
        np.asarray(cache.get(int(tensors["reference_row"][index, top_negative[index]]))[variant], dtype=np.float32)
        for index in operator_index
    ])
    operator_gradient = boundary_symmetric_operator_gradients(
        query_rule, positive_rule, negative_rule, float(parent_setting["rule_beta"]),
    )
    operator_identity = identity[operator_index]

    def operator_coherence(mask: np.ndarray) -> dict:
        return gradient_coherence_report(
            operator_gradient, operator_identity, np.asarray(mask)[operator_index],
        )

    gradient_diagnostic["full_symmetric_operator"] = {
        "correct_identity_expanded": operator_coherence(active_mask["correct"]),
        "correct_direct_support_only": operator_coherence(direct_correct),
        "alignment_identity_expanded": operator_coherence(active_mask["alignment_permuted"]),
        "alignment_direct_support_only": operator_coherence(direct_alignment),
        "matched_random": operator_coherence(active_mask["matched_random"]),
        "correct_direct_train_to_confirmation": gradient_transfer_report(
            operator_gradient, operator_identity,
            direct_correct[operator_index], confirmation_direct_correct[operator_index],
        ),
    }

    ranks = {}; coordinate = {}; histories = {}; route_report = {}
    for arm in ARMS:
        weight, history = train_arm(
            tensors, active_weight[arm], restore_weight, safety, train_mask,
            float(parent_setting["rule_beta"]), args,
        )
        score = scores_from_weights(tensors, weight, float(parent_setting["rule_beta"]))
        ranks[arm] = ranks_from_scores(score, tensors["valid"], tensors["positive"])
        coordinate[arm] = weight; histories[arm] = history
        mask = active_weight[arm] > 0
        route_report[arm] = {
            "identities": int(len(active_identity[arm])),
            "queries": int(np.sum(mask)), "formulas": int(len(np.unique(formula[mask]))),
            "total_identity_balanced_dose": float(active_weight[arm].sum()),
        }
        print(f"completed Rankmax rule-metric arm {arm}", flush=True)

    confirmation = {}
    for index, arm in enumerate(("parent", *ARMS)):
        rank = parent_rank if arm == "parent" else ranks[arm]
        confirmation[arm] = {
            "absolute_from_official": retrieval(official_rank[confirmation_mask], rank[confirmation_mask]),
            "increment_over_parent": retrieval(parent_rank[confirmation_mask], rank[confirmation_mask]),
            "absolute_formula_ci95": bootstrap(
                formula[confirmation_mask], official_rank[confirmation_mask], rank[confirmation_mask],
                args.bootstrap_draws, args.seed + 100 + index,
            ),
            "increment_formula_ci95": bootstrap(
                formula[confirmation_mask], parent_rank[confirmation_mask], rank[confirmation_mask],
                args.bootstrap_draws, args.seed + 200 + index,
            ),
        }
    paired = {
        f"correct_minus_{arm}": paired_rank_comparison(
            ranks["correct"][confirmation_mask], ranks[arm][confirmation_mask],
            formula[confirmation_mask], draws=args.bootstrap_draws,
            seed=args.seed + 300 + index,
        )
        for index, arm in enumerate(ARMS[:-1])
    }
    target = confirmation["correct"]
    gates = {
        "absolute_formula_ci_positive": target["absolute_formula_ci95"][0] > 0,
        "increment_parent_formula_ci_positive": target["increment_formula_ci95"][0] > 0,
        "increment_corrected_exceeds_twice_introduced": (
            target["increment_over_parent"]["corrected_at_1"]
            > 2 * target["increment_over_parent"]["introduced_at_1"]
        ),
        "beats_restore_only_ci": paired["correct_minus_restore_only"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "beats_matched_random_ci": paired["correct_minus_matched_random"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "beats_alignment_permuted_ci": paired["correct_minus_alignment_permuted"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "shared_psd_embedding": bool(np.all(coordinate["correct"] > 0)),
        "single_clean_spectrum_inference": True,
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_CROSSVIEW_RANKMAX_RULE_METRIC_PASS"
            if all(gates.values()) else "CHEMAWARE_CROSSVIEW_RANKMAX_RULE_METRIC_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": True,
        "shared_embedding_result": True,
        "backbone_updated": False,
        "scope": "Frozen parent rule map; repeat roles 0-1 training and role 2 confirmation; outer fold 4 untouched.",
        "claim_limit": "One-seed rule-coordinate metric-learning result on an already-used inner fold; not DreaMS backbone fine-tuning or external confirmation.",
        "method": {
            "base": parent_report["method"]["map"],
            "learned_map": "[official, sqrt(0.4)*mass, sqrt(0.8)*sqrt(w)*whitened_centered_rule]",
            "rule_coordinate_constraint": f"w=1+{args.maximum_coordinate_change}*tanh(delta), hence w>0",
            "loss": "event-calibrated Top-1 Rankmax on validated identity-level residual errors",
            "restoration": "official-correct parent errors shared identically by every arm",
            "controls": ["restoration only", "difficulty-matched random errors", "alignment-permuted cross-view events"],
            "nonselected_chemical_event_weight": 0.0,
            "same_initialization_schedule_optimizer_total_event_dose": True,
        },
        "data": {
            "queries": int(len(panel)), "identities": int(len(np.unique(identity))),
            "training_queries": int(np.sum(train_mask)),
            "confirmation_queries": int(np.sum(confirmation_mask)),
            "confirmation_formulas": int(len(np.unique(formula[confirmation_mask]))),
            "parent_fit_queries": int(len(final_fit)), "parent_fit_rows": int(len(fit_rows)),
            "formula_overlap": 0,
        },
        "routes": route_report,
        "gradient_diagnostic": gradient_diagnostic,
        "restoration_queries": int(np.sum(restore)),
        "confirmation": confirmation,
        "paired_confirmation": paired,
        "coordinate_summary": {
            arm: {
                "minimum": float(value.min()), "maximum": float(value.max()),
                "mean": float(value.mean()),
                "rms_change": float(np.sqrt(np.mean((value - 1.0) ** 2))),
            }
            for arm, value in coordinate.items()
        },
        "gates": gates,
        "runtime_seconds": time.time() - started,
        "parent_transform": transform_report,
        "configuration": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "route_audit": {
            "correct": correct_audit, "alignment_permuted": alignment_audit,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "repeat_ledger_sha256": sha256(args.repeat_ledger),
            "consensus_report_sha256": sha256(args.consensus_report),
            "parent_report_sha256": sha256(args.parent_report),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_rankmax_rule_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "ranks_and_weights.npz", query=panel, formula=formula,
            identity=identity, official_rank=official_rank, parent_rank=parent_rank,
            parent_top_negative=top_negative,
            **{f"{arm}_rank": ranks[arm] for arm in ARMS},
            **{f"{arm}_rule_weight": coordinate[arm] for arm in ARMS},
        )
        torch.save({"format": "chemaware_crossview_rankmax_rule_metric_v1", "history": histories}, temporary / "history.pt")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "routes": route_report,
        "confirmation": confirmation, "paired": paired, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

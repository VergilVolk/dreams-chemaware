"""CPU reachability audit for event-calibrated orthogonal ChemAware transfer.

This is a cheap decision experiment before DreaMS PEFT.  It reuses the
repository's bounded, identity-initialized shared residual adapter and the
existing full-candidate evaluation graph.  The new factor is the objective:
the frozen orthogonal policy selects a true-vs-official-top1 boundary, while
the required displacement is computed from the official margin itself.
Candidates and rules are training-only; inference remains one shared clean-
spectrum map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_crossview_routed_shared_adapter import (
    build_retrieval_arrays,
    evaluate,
    transform,
)
from audit_noise_direct_shared_metric_reachability import SharedResidualMetric
from audit_chemaware_mass_kernel_embedding import KernelCache
from audit_chemaware_whitened_centered_rule_kernel import FourCenterCache
from chemaware_candidate_residual_distillation_core import (
    event_calibrated_rankmax_loss,
    teacher_boundary_events,
)
from chemaware_direct_training_core import projected_guarded_auxiliary
from chemaware_iceberg_direct_core import stable_formula_folds
from dreams.models.chem_aware.global_embedding_adapter import (
    RuleConditionedResidualEmbeddingAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("zero_contrast", "reversed_contrast", "alignment_permuted", "correct")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_rule_residual_policy_v4")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_event_shared_adapter_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--train-fold", type=int, default=2)
    parser.add_argument("--evaluation-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--residual-strength", type=float, default=0.25)
    parser.add_argument(
        "--conditioning", choices=("official_only", "centered_rule"),
        default="official_only",
    )
    parser.add_argument(
        "--rule-center", choices=("true", "local_a", "local_b", "local_c"),
        default="true",
    )
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--event-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--target-margin", type=float, default=0.005)
    parser.add_argument("--protection-slack", type=float, default=0.002)
    parser.add_argument("--event-huber", type=float, default=0.02)
    parser.add_argument("--lambda-event", type=float, default=1.0)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--safety-slack", type=float, default=0.005)
    parser.add_argument("--preserve-weight", type=float, default=0.10)
    parser.add_argument("--maximum-event-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_representatives(
    panel: np.ndarray,
    body: dict[str, np.ndarray],
    graph: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frozen-official winning reference for every candidate molecule."""
    local = {int(row): index for index, row in enumerate(graph["rows"])}
    base = np.asarray(graph["base"], dtype=np.float32)
    maximum = max(
        int(body["query_ptr"][int(query) + 1] - body["query_ptr"][int(query)])
        for query in panel
    )
    representative = np.full((len(panel), maximum), -1, dtype=np.int64)
    valid = np.zeros((len(panel), maximum), dtype=bool)
    positive = np.empty(len(panel), dtype=np.int64)
    baseline = np.empty(len(panel), dtype=np.int64)
    for index, query in enumerate(map(int, panel)):
        qpos = int(graph["query"][index])
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        positive[index] = int(np.flatnonzero(labels)[0])
        molecule_scores = []
        for slot, molecule in enumerate(range(left, right)):
            pleft, pright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            rows = body["pair_candidate_row"][pleft:pright].astype(np.int64)
            positions = np.asarray([local[int(row)] for row in rows], dtype=np.int64)
            scores = base[positions] @ base[qpos]
            winner = int(np.argmax(scores))
            representative[index, slot] = int(positions[winner])
            valid[index, slot] = True
            molecule_scores.append(float(scores[winner]))
        baseline[index] = int(np.argmax(molecule_scores))
    return representative, valid, positive, baseline


def cache_events(
    cache: dict[str, np.ndarray],
    arm: str,
    positive: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, np.ndarray]:
    if not np.array_equal(cache["baseline_candidate"].astype(np.int64), baseline):
        raise RuntimeError(f"{arm} cache baseline candidate does not replay official embeddings")
    events = teacher_boundary_events(
        cache["baseline_rank"], cache["proposal_rank"],
        cache["baseline_candidate"], cache["proposed_candidate"],
        cache[f"{arm}_selected_candidate_slot"], positive,
    )
    utility = cache[f"{arm}_candidate_utility"].astype(np.float64)
    selected = cache[f"{arm}_selected_candidate_slot"].astype(np.int64)
    confidence = np.zeros(len(selected), dtype=np.float32)
    active = selected >= 0
    if np.any(active):
        confidence[active] = np.maximum(
            utility[np.arange(len(selected))[active], selected[active]]
            - float(cache["selected_threshold"]),
            0.0,
        ).astype(np.float32)
    # Confidence is an audit value, not a gradient weight.  This prevents an
    # uncalibrated tree-regressor scale from silently becoming the loss scale.
    return {
        "positive": events.positive,
        "negative": events.negative,
        "role": events.role,
        "active": events.active,
        "confidence": confidence,
    }


def event_loss_for_positions(
    model: torch.nn.Module,
    base: np.ndarray,
    rule_feature: np.ndarray | None,
    qpos: np.ndarray,
    representative: np.ndarray,
    valid: np.ndarray,
    events: dict[str, np.ndarray],
    index: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q0 = torch.from_numpy(base[qpos[index]])
    refs0 = torch.from_numpy(base[representative[index].clip(min=0)])
    if rule_feature is None:
        q = model(q0)
        refs = model(refs0.reshape(-1, refs0.shape[-1])).reshape_as(refs0)
    else:
        q = model(q0, torch.from_numpy(rule_feature[qpos[index]]))
        reference_position = representative[index].clip(min=0)
        refs = model(
            refs0.reshape(-1, refs0.shape[-1]),
            torch.from_numpy(rule_feature[reference_position].reshape(-1, rule_feature.shape[-1])),
        ).reshape_as(refs0)
    student = torch.einsum("bd,bmd->bm", q, refs)
    official = torch.einsum("bd,bmd->bm", q0, refs0)
    return event_calibrated_rankmax_loss(
        student, official, torch.from_numpy(valid[index]),
        torch.from_numpy(events["positive"][index]),
        torch.from_numpy(events["negative"][index]),
        torch.from_numpy(events["role"][index]),
        torch.ones(len(index)),
        target_margin=args.target_margin,
        protection_slack=args.protection_slack,
        huber_beta=args.event_huber,
    )


@torch.no_grad()
def event_fit(
    model: torch.nn.Module,
    base: np.ndarray,
    rule_feature: np.ndarray | None,
    qpos: np.ndarray,
    representative: np.ndarray,
    valid: np.ndarray,
    events: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, float]:
    index = np.flatnonzero(events["role"] == 1)
    if not len(index):
        return {"corrections": 0, "loss": 0.0, "satisfied_fraction": 1.0,
                "required_residual_median": 0.0, "required_residual_q90": 0.0}
    model.eval()
    loss, audit = event_loss_for_positions(
        model, base, rule_feature, qpos, representative, valid, events, index, args,
    )
    required = audit["required_residual"].cpu().numpy()
    shortfall = audit["shortfall"].cpu().numpy()
    return {
        "corrections": int(len(index)),
        "loss": float(loss),
        "satisfied_fraction": float(np.mean(shortfall <= 0)),
        "required_residual_median": float(np.median(required)),
        "required_residual_q90": float(np.quantile(required, 0.9)),
        "required_residual_maximum": float(np.max(required)),
    }


def train_arm(
    base: np.ndarray,
    rule_feature: np.ndarray | None,
    graph: dict[str, object],
    representative: np.ndarray,
    candidate_valid: np.ndarray,
    baseline_rank: np.ndarray,
    events: dict[str, np.ndarray],
    clean_schedules: list[list[np.ndarray]],
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, float], dict[str, float]]:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model: torch.nn.Module
    if rule_feature is None:
        model = SharedResidualMetric(base.shape[1], args.hidden_dim, args.residual_strength)
    else:
        model = RuleConditionedResidualEmbeddingAdapter(
            base.shape[1], rule_feature.shape[1], args.hidden_dim, args.residual_strength,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    parameters = [value for value in model.parameters() if value.requires_grad]
    qpos = graph["query"]; ppos = graph["positive"]; npos = graph["negative"]
    correction = np.flatnonzero(events["role"] == 1)
    event_rng = np.random.default_rng(args.seed + 700)
    initial_fit = event_fit(
        model, base, rule_feature, qpos, representative, candidate_valid, events, args,
    )
    history: list[dict[str, float]] = []
    for epoch, batches in enumerate(clean_schedules, start=1):
        total = {key: 0.0 for key in (
            "loss", "rank", "safety", "preserve", "event",
            "event_raw_cosine", "event_safe_ratio", "event_projected",
        )}
        seen = 0
        model.train()
        for index in batches:
            q0 = torch.from_numpy(base[qpos[index]])
            p0 = torch.from_numpy(base[ppos[index]])
            n0 = torch.from_numpy(base[npos[index]])
            if rule_feature is None:
                q = model(q0); p = model(p0)
                n = model(n0.reshape(-1, n0.shape[-1])).reshape_as(n0)
            else:
                q = model(q0, torch.from_numpy(rule_feature[qpos[index]]))
                p = model(p0, torch.from_numpy(rule_feature[ppos[index]]))
                n = model(
                    n0.reshape(-1, n0.shape[-1]),
                    torch.from_numpy(rule_feature[npos[index]].reshape(-1, rule_feature.shape[-1])),
                ).reshape_as(n0)
            positive = torch.sum(q * p, dim=1)
            negative = torch.einsum("bd,bkd->bk", q, n)
            logits = torch.cat((positive[:, None], negative), dim=1) / args.temperature
            rank_loss = F.cross_entropy(logits, torch.zeros(len(index), dtype=torch.long))
            official_margin = torch.sum(q0 * p0, dim=1) - torch.max(
                torch.einsum("bd,bkd->bk", q0, n0), dim=1,
            ).values
            student_margin = positive - torch.max(negative, dim=1).values
            safe = torch.from_numpy(baseline_rank[index] == 1)
            safety = (
                F.relu(official_margin[safe].detach() - args.safety_slack - student_margin[safe]).mean()
                if bool(safe.any()) else rank_loss * 0.0
            )
            adapted = torch.cat((q, p, n.reshape(-1, n.shape[-1])))
            initial = torch.cat((q0, p0, n0.reshape(-1, n0.shape[-1])))
            preserve = (1.0 - torch.sum(adapted * initial, dim=1)).mean()
            primary = rank_loss + args.safety_weight * safety + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            event_value = primary * 0.0
            geometry = {
                "auxiliary_raw_cosine": 0.0,
                "auxiliary_safe_to_primary_ratio": 0.0,
                "auxiliary_conflict_projected": False,
            }
            if len(correction):
                event_index = event_rng.choice(
                    correction,
                    size=min(args.event_batch_size, len(correction)),
                    replace=len(correction) < args.event_batch_size,
                ).astype(np.int64)
                event_value, _ = event_loss_for_positions(
                    model, base, rule_feature, qpos, representative, candidate_valid,
                    events, event_index, args,
                )
                primary_gradient = torch.autograd.grad(
                    primary, parameters, retain_graph=True, allow_unused=True,
                )
                event_gradient = torch.autograd.grad(
                    args.lambda_event * event_value, parameters,
                    retain_graph=False, allow_unused=True,
                )
                event_norm = sum(
                    float(torch.sum(value.detach().float() ** 2))
                    for value in event_gradient if value is not None
                )
                if event_norm > 0:
                    combined, geometry = projected_guarded_auxiliary(
                        primary_gradient, event_gradient, parameters,
                        maximum_auxiliary_ratio=args.maximum_event_gradient_ratio,
                    )
                    for parameter, gradient in zip(parameters, combined):
                        parameter.grad = gradient
                else:
                    primary.backward()
            else:
                primary.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            count = len(index); seen += count
            for key, value in (
                ("loss", primary + args.lambda_event * event_value),
                ("rank", rank_loss), ("safety", safety),
                ("preserve", preserve), ("event", event_value),
            ):
                total[key] += float(value.detach()) * count
            total["event_raw_cosine"] += float(geometry["auxiliary_raw_cosine"]) * count
            total["event_safe_ratio"] += float(geometry["auxiliary_safe_to_primary_ratio"]) * count
            total["event_projected"] += float(geometry["auxiliary_conflict_projected"]) * count
        record = {key: value / max(1, seen) for key, value in total.items()}
        record.update({"epoch": epoch, "gradient": float(gradient)})
        history.append(record)
        if epoch == 1 or epoch == len(clean_schedules) or epoch % 5 == 0:
            print(json.dumps(record), flush=True)
    final_fit = event_fit(
        model, base, rule_feature, qpos, representative, candidate_valid, events, args,
    )
    return model, history, initial_fit, final_fit


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    if len({args.train_fold, args.evaluation_fold, args.outer_fold}) != 3:
        raise ValueError("training, evaluation and outer folds must differ")
    if not (0 < args.maximum_event_gradient_ratio <= 1 and args.lambda_event >= 0):
        raise ValueError("invalid event-gradient configuration")
    torch.set_num_threads(max(1, args.torch_threads))
    report_path = args.policy_dir / "report.json"
    teacher_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (teacher_report.get("status") != "CHEMAWARE_ORTHOGONAL_RULE_RESIDUAL_POLICY_COMPLETE"
            or not all(teacher_report.get("gates", {}).values())):
        raise RuntimeError("a fully passing orthogonal development teacher is required")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.policy_dir / "validation_policy.npz", allow_pickle=False) as loaded:
        train_cache = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.policy_dir / "inner_policy.npz", allow_pickle=False) as loaded:
        eval_cache = {key: np.asarray(loaded[key]) for key in loaded.files}
    train_query = train_cache["query"].astype(np.int64)
    eval_query = eval_cache["query"].astype(np.int64)
    panel = np.concatenate((train_query, eval_query))
    formula = body["query_formula"][panel].astype(str)
    identity = body["query_ik14"][panel].astype(str)
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    if np.any(fold[train_query] != args.train_fold) or np.any(fold[eval_query] != args.evaluation_fold):
        raise RuntimeError("teacher cache escaped its registered formula fold")
    if set(formula[:len(train_query)]) & set(formula[len(train_query):]):
        raise RuntimeError("training and evaluation formulas overlap")
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    graph = build_retrieval_arrays(panel, body, rows, official, args.top_negatives)
    base = graph["base"]
    rule_feature = None
    if args.conditioning == "centered_rule":
        kernel_args = SimpleNamespace(
            token_dir=args.token_dir, rule_library=args.rule_library,
            top_peaks=args.top_peaks, kernel_dim=args.kernel_dim,
            bin_width=args.bin_width, grid_offsets=args.grid_offsets,
            intensity_power=args.intensity_power, mass_shift_da=args.mass_shift_da,
            pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
            uniform_channel_weight=1.0, rule_tolerance=args.rule_tolerance,
            rule_channel_weight=1.0,
        )
        row_position = {int(row): index for index, row in enumerate(rows)}
        raw_rule = KernelCache(
            kernel_args, row_position,
            variants=(
                "mass", "rule_response", "rule_response_local_background_a",
                "rule_response_local_background_b", "rule_response_local_background_c",
            ),
        )
        centered_rule = FourCenterCache(raw_rule)
        rule_feature = np.stack([
            np.asarray(centered_rule.get(int(row))[args.rule_center], dtype=np.float32)
            for row in graph["rows"]
        ])
        print(
            f"built {args.rule_center} centered-rule conditioning for {len(rule_feature)} spectra",
            flush=True,
        )
    representative, candidate_valid, positive, baseline_candidate = candidate_representatives(
        panel, body, graph,
    )
    baseline_rank = evaluate(base, graph, identity)
    expected_rank = np.concatenate((train_cache["baseline_rank"], eval_cache["baseline_rank"]))
    if not np.array_equal(baseline_rank, expected_rank):
        raise RuntimeError(f"official full-candidate replay mismatch: {np.sum(baseline_rank != expected_rank)}")
    cached_baseline = train_cache["baseline_candidate"].astype(np.int64)
    baseline_mismatch = np.flatnonzero(
        baseline_candidate[:len(train_query)] != cached_baseline
    )
    maximum_tie_gap = 0.0
    for index in baseline_mismatch:
        q = int(graph["query"][index])
        observed = int(baseline_candidate[index]); frozen = int(cached_baseline[index])
        observed_score = float(base[representative[index, observed]] @ base[q])
        frozen_score = float(base[representative[index, frozen]] @ base[q])
        maximum_tie_gap = max(maximum_tie_gap, abs(observed_score - frozen_score))
    if maximum_tie_gap > 5e-7:
        raise RuntimeError(
            "training cache baseline candidate differs beyond float32 tie tolerance: "
            f"count={len(baseline_mismatch)} max_gap={maximum_tie_gap}"
        )
    # Candidate identity at a float32 tie is part of the frozen teacher
    # contract.  Preserve it rather than allowing a BLAS reduction order to
    # silently change the event pair on another machine.
    baseline_candidate[:len(train_query)] = cached_baseline
    train_rep = representative[:len(train_query)]
    train_valid = candidate_valid[:len(train_query)]
    train_positive = positive[:len(train_query)]
    train_baseline = baseline_candidate[:len(train_query)]
    arm_events = {
        arm: cache_events(train_cache, arm, train_positive, train_baseline)
        for arm in ARMS
    }
    rng = np.random.default_rng(args.seed + 101)
    clean_schedules = []
    train_index = np.arange(len(train_query), dtype=np.int64)
    for _ in range(args.epochs):
        order = train_index[rng.permutation(len(train_index))]
        clean_schedules.append([
            order[left:left + args.batch_size]
            for left in range(0, len(order), args.batch_size)
        ])
    # Training graph is a prefix of the combined graph, so all local row
    # positions remain valid and no evaluation query enters a gradient.
    train_graph = {
        "query": graph["query"][:len(train_query)],
        "positive": graph["positive"][:len(train_query)],
        "negative": graph["negative"][:len(train_query)],
    }
    models = {}; histories = {}; fit = {}; ranks = {}
    for arm in ARMS:
        model, history, initial_fit, final_fit = train_arm(
            base, rule_feature, train_graph, train_rep, train_valid,
            baseline_rank[:len(train_query)], arm_events[arm], clean_schedules, args,
        )
        if rule_feature is None:
            adapted = transform(model, base, 512)
        else:
            model.eval(); adapted = np.empty_like(base)
            with torch.no_grad():
                for left in range(0, len(base), 512):
                    right = min(left + 512, len(base))
                    adapted[left:right] = model(
                        torch.from_numpy(base[left:right]),
                        torch.from_numpy(rule_feature[left:right]),
                    ).cpu().numpy()
        ranks[arm] = evaluate(adapted, graph, identity)
        models[arm] = model.state_dict(); histories[arm] = history
        fit[arm] = {"initial": initial_fit, "final": final_fit}
        print(f"completed event-calibrated shared-adapter arm {arm}", flush=True)
    evaluation = np.arange(len(train_query), len(panel))
    evaluation_formula = formula[evaluation]
    results = {}
    for index, arm in enumerate(ARMS):
        results[arm] = {
            "retrieval": retrieval(baseline_rank[evaluation], ranks[arm][evaluation]),
            "formula_cluster_bootstrap_ci95": bootstrap(
                evaluation_formula, baseline_rank[evaluation], ranks[arm][evaluation],
                args.bootstrap_draws, args.seed + 200 + index,
            ),
        }
    paired = {
        f"correct_minus_{arm}": paired_rank_comparison(
            ranks["correct"][evaluation], ranks[arm][evaluation], evaluation_formula,
            draws=args.bootstrap_draws, seed=args.seed + 300 + index,
        )
        for index, arm in enumerate(ARMS[:-1])
    }
    primary = results["correct"]
    gates = {
        "correct_point_gain_positive": primary["retrieval"]["delta_recall1"] > 0,
        "correct_formula_ci_positive": primary["formula_cluster_bootstrap_ci95"][0] > 0,
        "corrected_exceeds_twice_introduced": (
            primary["retrieval"]["corrected_at_1"]
            > 2 * primary["retrieval"]["introduced_at_1"]
        ),
        **{
            f"beats_{arm}_ci": paired[f"correct_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in ARMS[:-1]
        },
        "shared_query_reference_map": True,
        "single_clean_spectrum_inference": True,
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_ORTHOGONAL_EVENT_SHARED_ADAPTER_PASS"
            if all(gates.values()) else "CHEMAWARE_ORTHOGONAL_EVENT_SHARED_ADAPTER_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": True,
        "shared_embedding_result": True,
        "backbone_updated": False,
        "scope": "Fold 2 cached-embedding train, previously used fold 3 evaluation; fold 4 untouched.",
        "claim_limit": "CPU post-adapter reachability audit, not DreaMS PEFT or untouched outer confirmation.",
        "method": {
            "teacher": "frozen orthogonal correct-minus-content-permuted candidate policy",
            "chemical_target": "label-oriented selected true-vs-official-top1 boundary",
            "target_amplitude": "epsilon minus frozen official margin; no fixed action cap",
            "student": "existing bounded zero-initialized SharedResidualMetric",
            "conditioning": args.conditioning,
            "rule_center": args.rule_center if rule_feature is not None else None,
            "conditioning_inputs": (
                "clean spectrum-local centered registered-mass responses only"
                if rule_feature is not None else "official embedding only"
            ),
            "primary": "clean listwise retrieval plus official-correct safety and preservation",
            "gradient_injection": "existing projected_guarded_auxiliary with fixed norm ratio",
            "nonselected_or_unsupported_chemical_weight": 0.0,
            "utility_magnitude_used_as_loss_weight": False,
            "same_clean_schedule_initialization_optimizer_across_arms": True,
        },
        "data": {
            "train_queries": int(len(train_query)),
            "evaluation_queries": int(len(eval_query)),
            "train_formulas": int(len(np.unique(formula[:len(train_query)]))),
            "evaluation_formulas": int(len(np.unique(evaluation_formula))),
            "formula_overlap": 0,
            "outer_fold_untouched": int(np.sum(fold == args.outer_fold)),
            "float32_tied_baseline_candidates_replayed_from_teacher": int(len(baseline_mismatch)),
            "maximum_tied_baseline_score_gap": float(maximum_tie_gap),
        },
        "events": {
            arm: {
                "selected": int(np.sum(train_cache[f"{arm}_selected_candidate_slot"] >= 0)),
                "corrections": int(np.sum(events["role"] == 1)),
                "protections": int(np.sum(events["role"] == 2)),
                "unsupported_zero_weight": int(np.sum(
                    (train_cache[f"{arm}_selected_candidate_slot"] >= 0)
                    & (events["role"] == 0)
                )),
                "median_selection_confidence": float(np.median(
                    events["confidence"][events["confidence"] > 0]
                )) if np.any(events["confidence"] > 0) else 0.0,
            }
            for arm, events in arm_events.items()
        },
        "event_fit": fit,
        "held_evaluation": results,
        "paired_evaluation": paired,
        "gates": gates,
        "configuration": {
            key: value for key, value in vars(args).items() if not isinstance(value, Path)
        },
        "runtime_seconds": time.time() - started,
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "teacher_report_sha256": sha256(report_path),
            "teacher_validation_cache_sha256": sha256(args.policy_dir / "validation_policy.npz"),
            "teacher_inner_cache_sha256": sha256(args.policy_dir / "inner_policy.npz"),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_event_adapter_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "ranks.npz", query=panel, formula=formula, identity=identity,
            baseline_rank=baseline_rank, **{f"{arm}_rank": ranks[arm] for arm in ARMS},
        )
        torch.save({
            "format": "chemaware_orthogonal_event_shared_adapter_v1",
            "state_dict_by_arm": models, "histories": histories,
            "hidden_dim": args.hidden_dim, "residual_strength": args.residual_strength,
        }, temporary / "adapters.pt")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "events": report["events"],
        "event_fit": report["event_fit"], "held_evaluation": results,
        "paired_evaluation": paired, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

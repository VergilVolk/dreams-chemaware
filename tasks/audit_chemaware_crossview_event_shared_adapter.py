"""Multi-view event-calibrated transfer into one shared spectrum embedding.

This reuses the already-passed cross-view orthogonal teacher.  Formula roles
0-1 provide training actions and role 2 is never used by the optimizer.  The
teacher's candidate identity is converted into a label-oriented boundary only
during training; deployment remains a single candidate-free spectrum map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_crossview_routed_shared_adapter import (
    build_retrieval_arrays,
    evaluate,
    transform,
)
from audit_chemaware_orthogonal_event_shared_adapter import (
    candidate_representatives,
    train_arm,
)
from chemaware_candidate_residual_distillation_core import teacher_boundary_events
from chemaware_crossview_consensus_core import crossview_policy
from chemaware_iceberg_direct_core import stable_formula_folds


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("zero_contrast", "reversed_contrast", "alignment_permuted", "correct")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--repeat-ledger", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v3/repeat_ledger.npz")
    parser.add_argument("--consensus-report", type=Path, default=ROOT / "data/validation/chemaware_crossview_consensus_teacher_v2/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_crossview_event_shared_adapter_v1")
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--residual-strength", type=float, default=0.25)
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


def proposed_local_candidate(
    selected_slot: np.ndarray,
    candidate_identity: np.ndarray,
    query: np.ndarray,
    body: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.full(len(query), -1, dtype=np.int64)
    for index in np.flatnonzero(selected_slot >= 0):
        identity = str(candidate_identity[index, int(selected_slot[index])])
        left, right = map(int, body["query_ptr"][int(query[index]):int(query[index]) + 2])
        candidates = body["molecule_ik14"][left:right].astype(str)
        match = np.flatnonzero(candidates == identity)
        if len(match) != 1:
            raise RuntimeError("cross-view candidate identity does not map uniquely into its query")
        output[index] = int(match[0])
    return output


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    torch.set_num_threads(max(1, args.torch_threads))
    consensus = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    if (consensus.get("status") != "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_PASS"
            or not all(consensus.get("gates", {}).values())):
        raise RuntimeError("a fully passing cross-view teacher is required")
    setting = consensus["selection"]
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.repeat_ledger, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    formula = ledger["formula"].astype(str)
    identity = ledger["identity"].astype(str)
    role = stable_formula_folds(formula, 3, args.split_seed)
    train_original = np.flatnonzero(role < 2)
    evaluation_original = np.flatnonzero(role == 2)
    order = np.concatenate((train_original, evaluation_original))
    if set(formula[train_original]) & set(formula[evaluation_original]):
        raise RuntimeError("cross-view training and evaluation formulas overlap")

    selected = {}; teacher_rank = {}; teacher_score = {}; teacher_support = {}
    for arm in ARMS:
        rank, slot, score, support = crossview_policy(
            ledger["baseline_rank"], ledger["candidate_identity"],
            ledger["candidate_valid"], ledger["proposal_rank"],
            ledger[f"{arm}_candidate_utility"], identity,
            threshold=float(setting["threshold"]),
            aggregation=str(setting["aggregation"]),
            min_context=int(setting["min_context"]),
        )
        selected[arm] = slot
        teacher_rank[arm] = rank
        teacher_score[arm] = score
        teacher_support[arm] = support
        expected = consensus["held_confirmation"][arm]["retrieval"]
        observed = retrieval(
            ledger["baseline_rank"][evaluation_original], rank[evaluation_original],
        )
        for key in ("corrected_at_1", "introduced_at_1", "risk_utility_at_1"):
            if int(observed[key]) != int(expected[key]):
                raise RuntimeError(f"cross-view teacher replay drifted at {arm}/{key}")

    panel = ledger["query"][order].astype(np.int64)
    formula = formula[order]; identity = identity[order]
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    graph = build_retrieval_arrays(panel, body, rows, official, args.top_negatives)
    base = graph["base"]
    representative, candidate_valid, positive, baseline_candidate = candidate_representatives(
        panel, body, graph,
    )
    baseline_rank = evaluate(base, graph, identity)
    expected_baseline = ledger["baseline_rank"][order].astype(np.int64)
    if not np.array_equal(baseline_rank, expected_baseline):
        raise RuntimeError(f"cross-view official replay mismatch: {np.sum(baseline_rank != expected_baseline)}")
    train_count = len(train_original)
    teacher_events = {}
    for arm in ARMS:
        slot = selected[arm][order][:train_count].astype(np.int64)
        proposal = proposed_local_candidate(
            selected[arm], ledger["candidate_identity"], ledger["query"], body,
        )[order][:train_count]
        proposal_matrix = proposal[:, None]
        proposal_rank = teacher_rank[arm][order][:train_count, None]
        normalized_slot = np.where(slot >= 0, 0, -1)
        event = teacher_boundary_events(
            baseline_rank[:train_count], proposal_rank,
            baseline_candidate[:train_count], proposal_matrix,
            normalized_slot, positive[:train_count],
        )
        teacher_events[arm] = {
            "positive": event.positive, "negative": event.negative,
            "role": event.role, "active": event.active,
            "confidence": np.maximum(
                teacher_score[arm][order][:train_count] - float(setting["threshold"]),
                0.0,
            ).astype(np.float32),
        }

    rng = np.random.default_rng(args.seed + 101)
    train_index = np.arange(train_count, dtype=np.int64)
    clean_schedules = []
    for _ in range(args.epochs):
        shuffled = train_index[rng.permutation(train_count)]
        clean_schedules.append([
            shuffled[left:left + args.batch_size]
            for left in range(0, len(shuffled), args.batch_size)
        ])
    train_graph = {
        "query": graph["query"][:train_count],
        "positive": graph["positive"][:train_count],
        "negative": graph["negative"][:train_count],
    }
    models = {}; histories = {}; fit = {}; ranks = {}
    for arm in ARMS:
        model, history, initial_fit, final_fit = train_arm(
            base, None, train_graph,
            representative[:train_count], candidate_valid[:train_count],
            baseline_rank[:train_count], teacher_events[arm], clean_schedules, args,
        )
        adapted = transform(model, base, 512)
        ranks[arm] = evaluate(adapted, graph, identity)
        models[arm] = model.state_dict(); histories[arm] = history
        fit[arm] = {"initial": initial_fit, "final": final_fit}
        print(f"completed cross-view event shared-adapter arm {arm}", flush=True)

    evaluation = np.arange(train_count, len(panel))
    results = {}
    for index, arm in enumerate(ARMS):
        results[arm] = {
            "retrieval": retrieval(baseline_rank[evaluation], ranks[arm][evaluation]),
            "formula_cluster_bootstrap_ci95": bootstrap(
                formula[evaluation], baseline_rank[evaluation], ranks[arm][evaluation],
                args.bootstrap_draws, args.seed + 200 + index,
            ),
        }
    paired = {
        f"correct_minus_{arm}": paired_rank_comparison(
            ranks["correct"][evaluation], ranks[arm][evaluation], formula[evaluation],
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
        "formula_role2_optimizer_untouched": True,
        "outer_fold4_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_CROSSVIEW_EVENT_SHARED_ADAPTER_PASS"
            if all(gates.values()) else "CHEMAWARE_CROSSVIEW_EVENT_SHARED_ADAPTER_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": True,
        "shared_embedding_result": True,
        "backbone_updated": False,
        "scope": "Cross-view formula roles 0-1 train, role 2 evaluation; outer fold 4 untouched.",
        "claim_limit": "CPU post-adapter reachability audit, not DreaMS PEFT or untouched outer confirmation.",
        "method": {
            "teacher": "passed leave-one-spectrum-out orthogonal cross-view consensus",
            "candidate_alignment": "molecular identity, never candidate slot",
            "target": "event-calibrated true-vs-official-top1 residual boundary",
            "training_mass": "all repeat views with identity-level context; clean schedule fixed across arms",
            "utility_magnitude_used_as_loss_weight": False,
            "gradient_injection": "existing projected_guarded_auxiliary",
            "nonselected_or_unsupported_chemical_weight": 0.0,
        },
        "data": {
            "train_queries": int(train_count),
            "evaluation_queries": int(len(evaluation)),
            "train_identities": int(len(np.unique(identity[:train_count]))),
            "evaluation_identities": int(len(np.unique(identity[evaluation]))),
            "formula_overlap": 0,
        },
        "events": {
            arm: {
                "selected": int(np.sum(selected[arm][train_original] >= 0)),
                "corrections": int(np.sum(event["role"] == 1)),
                "protections": int(np.sum(event["role"] == 2)),
                "unsupported_zero_weight": int(np.sum(
                    (selected[arm][train_original] >= 0) & (event["role"] == 0)
                )),
                "identities_with_correction": int(len(np.unique(
                    identity[:train_count][event["role"] == 1]
                ))),
            }
            for arm, event in teacher_events.items()
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
            "repeat_ledger_sha256": sha256(args.repeat_ledger),
            "consensus_report_sha256": sha256(args.consensus_report),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_crossview_event_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "ranks.npz", query=panel, formula=formula, identity=identity,
            baseline_rank=baseline_rank, **{f"{arm}_rank": ranks[arm] for arm in ARMS},
        )
        torch.save({
            "format": "chemaware_crossview_event_shared_adapter_v1",
            "state_dict_by_arm": models, "histories": histories,
            "hidden_dim": args.hidden_dim, "residual_strength": args.residual_strength,
        }, temporary / "adapters.pt")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "events": report["events"],
        "event_fit": fit, "held_evaluation": results,
        "paired_evaluation": paired, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

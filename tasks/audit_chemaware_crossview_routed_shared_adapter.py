"""CPU reachability test for cross-view ChemAware supervision in one shared map.

The adapter and loss are reused from the existing Noise direct-shared probe.
Correct, alignment-permuted and uniform arms share formula splits, candidate
representatives, initialization, optimizer steps and identity-balanced dose.
Only the training curriculum weights differ.  Evaluation uses one clean
spectrum and the same adapter for query and reference embeddings.
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
import torch.nn.functional as F

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_noise_direct_shared_metric_reachability import SharedResidualMetric
from chemaware_crossview_consensus_core import crossview_policy
from chemaware_iceberg_direct_core import stable_formula_folds


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("uniform", "alignment_permuted", "correct")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--repeat-ledger", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v3/repeat_ledger.npz")
    parser.add_argument("--consensus-report", type=Path, default=ROOT / "data/validation/chemaware_crossview_consensus_teacher_v2/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_crossview_routed_shared_adapter_v1")
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--residual-strength", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--correctable-weight", type=float, default=4.0)
    parser.add_argument("--other-error-weight", type=float, default=0.75)
    parser.add_argument("--correct-control-weight", type=float, default=0.50)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--safety-slack", type=float, default=0.005)
    parser.add_argument("--preserve-weight", type=float, default=0.10)
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


def identity_balanced_route_weights(
    identity: np.ndarray,
    baseline_rank: np.ndarray,
    correctable: np.ndarray,
    mask: np.ndarray,
    *,
    correctable_weight: float,
    other_error_weight: float,
    correct_control_weight: float,
) -> np.ndarray:
    identity = np.asarray(identity).astype(str)
    baseline_rank = np.asarray(baseline_rank, dtype=np.int64)
    correctable = np.asarray(correctable, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    weight = np.where(
        correctable, correctable_weight,
        np.where(baseline_rank > 1, other_error_weight, correct_control_weight),
    ).astype(np.float64)
    weight[~mask] = 0.0
    for value in np.unique(identity[mask]):
        positions = np.flatnonzero(mask & (identity == value))
        weight[positions] /= np.sum(weight[positions])
    weight[mask] *= np.sum(mask) / np.sum(weight[mask])
    return weight.astype(np.float32)


def build_retrieval_arrays(
    panel: np.ndarray,
    body: dict[str, np.ndarray],
    rows: np.ndarray,
    official: np.ndarray,
    top_negatives: int,
) -> dict[str, object]:
    row_position = {int(row): index for index, row in enumerate(rows)}
    reachable = [body["query_row"][panel].astype(np.int64)]
    for query in panel:
        left, right = map(int, body["query_ptr"][query : query + 2])
        reachable.append(body["pair_candidate_row"][body["molecule_ptr"][left] : body["molecule_ptr"][right]])
    reachable_rows = np.unique(np.concatenate(reachable)).astype(np.int64)
    base = np.asarray(official[[row_position[int(row)] for row in reachable_rows]], dtype=np.float32)
    local = {int(row): index for index, row in enumerate(reachable_rows)}
    query_position = np.empty(len(panel), dtype=np.int64)
    positive_position = np.empty(len(panel), dtype=np.int64)
    negative_position = np.empty((len(panel), top_negatives), dtype=np.int64)
    candidate_position: list[np.ndarray] = []
    candidate_identity: list[np.ndarray] = []
    for index, query in enumerate(panel):
        qrow = int(body["query_row"][query]); qpos = local[qrow]
        query_position[index] = qpos
        left, right = map(int, body["query_ptr"][query : query + 2])
        molecule_best = []
        all_rows = []; all_identity = []
        labels = body["molecule_label"][left:right].astype(bool)
        for molecule in range(left, right):
            pleft, pright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            reference_rows = body["pair_candidate_row"][pleft:pright].astype(np.int64)
            positions = np.asarray([local[int(row)] for row in reference_rows], dtype=np.int64)
            scores = base[positions] @ base[qpos]
            winner = int(np.argmax(scores))
            molecule_best.append((float(scores[winner]), int(positions[winner]), molecule - left))
            all_rows.extend(positions.tolist())
            all_identity.extend([str(body["molecule_ik14"][molecule])] * len(positions))
        positive_local = int(np.flatnonzero(labels)[0])
        positive_position[index] = molecule_best[positive_local][1]
        negatives = sorted(
            [item for item in molecule_best if item[2] != positive_local],
            key=lambda item: (-item[0], item[1]),
        )
        if not negatives:
            raise RuntimeError("query has no negative molecule")
        chosen = [item[1] for item in negatives[:top_negatives]]
        chosen.extend([chosen[-1]] * (top_negatives - len(chosen)))
        negative_position[index] = chosen
        candidate_position.append(np.asarray(all_rows, dtype=np.int64))
        candidate_identity.append(np.asarray(all_identity, dtype="U14"))
        if (index + 1) % 1024 == 0:
            print(f"built shared-adapter graph {index + 1}/{len(panel)}", flush=True)
    return {
        "rows": reachable_rows, "base": base, "query": query_position,
        "positive": positive_position, "negative": negative_position,
        "candidate": candidate_position, "candidate_identity": candidate_identity,
    }


@torch.inference_mode()
def transform(model: SharedResidualMetric, values: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval(); output = np.empty_like(values)
    for left in range(0, len(values), batch_size):
        right = min(left + batch_size, len(values))
        output[left:right] = model(torch.from_numpy(values[left:right])).cpu().numpy()
    return output


def evaluate(
    embeddings: np.ndarray,
    graph: dict[str, object],
    panel_identity: np.ndarray,
) -> np.ndarray:
    rank = np.empty(len(panel_identity), dtype=np.int16)
    for index, truth in enumerate(panel_identity.astype(str)):
        q = embeddings[graph["query"][index]]
        positions = graph["candidate"][index]
        identities = graph["candidate_identity"][index]
        scores = embeddings[positions] @ q
        molecules = np.unique(identities)
        molecule_score = np.asarray([np.max(scores[identities == value]) for value in molecules])
        positive = float(molecule_score[molecules == truth][0])
        rank[index] = 1 + int(np.sum(molecule_score[molecules != truth] >= positive))
    return rank


def train_arm(
    base: np.ndarray,
    graph: dict[str, object],
    train_mask: np.ndarray,
    baseline_correct: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[SharedResidualMetric, list[dict[str, float]]]:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = SharedResidualMetric(base.shape[1], args.hidden_dim, args.residual_strength)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train = np.flatnonzero(train_mask); rng = np.random.default_rng(args.seed)
    qpos = graph["query"]; ppos = graph["positive"]; npos = graph["negative"]
    history = []
    for epoch in range(args.epochs):
        order = train[rng.permutation(len(train))]
        total = np.zeros(5, dtype=np.float64); seen = 0
        model.train()
        for left in range(0, len(order), args.batch_size):
            index = order[left:left + args.batch_size]
            q0 = torch.from_numpy(base[qpos[index]])
            p0 = torch.from_numpy(base[ppos[index]])
            n0 = torch.from_numpy(base[npos[index]])
            q = model(q0); p = model(p0)
            n = model(n0.reshape(-1, n0.shape[-1])).reshape_as(n0)
            positive = torch.sum(q * p, dim=1)
            negative = torch.einsum("bd,bkd->bk", q, n)
            logits = torch.cat((positive[:, None], negative), dim=1) / args.temperature
            per_query = F.cross_entropy(
                logits, torch.zeros(len(index), dtype=torch.long), reduction="none",
            )
            weight = torch.from_numpy(weights[index])
            rank_loss = torch.sum(weight * per_query) / weight.sum().clamp_min(1e-8)
            initial_margin = torch.sum(q0 * p0, dim=1) - torch.max(
                torch.einsum("bd,bkd->bk", q0, n0), dim=1,
            ).values
            current_margin = positive - torch.max(negative, dim=1).values
            safe = torch.from_numpy(baseline_correct[index])
            safety = (
                F.relu(initial_margin[safe].detach() - args.safety_slack - current_margin[safe]).mean()
                if bool(safe.any()) else rank_loss * 0.0
            )
            adapted = torch.cat((q, p, n.reshape(-1, n.shape[-1])))
            initial = torch.cat((q0, p0, n0.reshape(-1, n0.shape[-1])))
            preserve = (1.0 - torch.sum(adapted * initial, dim=1)).mean()
            loss = rank_loss + args.safety_weight * safety + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(index); seen += count
            total += np.asarray([
                float(loss.detach()), float(rank_loss.detach()), float(safety.detach()),
                float(preserve.detach()), float(gradient),
            ]) * count
        row = {
            "epoch": epoch + 1, "loss": float(total[0] / seen),
            "rank_loss": float(total[1] / seen), "safety": float(total[2] / seen),
            "preserve": float(total[3] / seen), "gradient": float(total[4] / seen),
        }
        history.append(row)
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 7 == 0:
            print(json.dumps(row), flush=True)
    return model, history


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.smoke:
        args.epochs = min(args.epochs, 2); args.bootstrap_draws = min(args.bootstrap_draws, 300)
    torch.set_num_threads(max(1, args.torch_threads))
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.repeat_ledger, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    consensus = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    if consensus.get("status") != "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_PASS":
        raise RuntimeError("validated cross-view teacher is required")
    setting = consensus["selection"]
    panel = ledger["query"].astype(np.int64); formula = ledger["formula"].astype(str)
    identity = ledger["identity"].astype(str); baseline = ledger["baseline_rank"].astype(np.int64)
    role = stable_formula_folds(formula, 3, args.split_seed)
    train_mask = role == 0; validation_mask = role == 1; confirmation_mask = role == 2
    if any(set(formula[role == i]) & set(formula[role == j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula role leakage")
    consensus_rank = {}
    for arm in ("correct", "alignment_permuted"):
        rank, _selected, _score, _support = crossview_policy(
            baseline, ledger["candidate_identity"], ledger["candidate_valid"],
            ledger["proposal_rank"], ledger[f"{arm}_candidate_utility"], identity,
            threshold=float(setting["threshold"]), aggregation=str(setting["aggregation"]),
            min_context=int(setting["min_context"]),
        )
        consensus_rank[arm] = rank
    correctable = {
        "uniform": np.zeros(len(panel), dtype=bool),
        **{
            arm: (baseline > 1) & (consensus_rank[arm] == 1)
            for arm in ("correct", "alignment_permuted")
        },
    }
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    graph = build_retrieval_arrays(panel, body, rows, official, args.top_negatives)
    base = graph["base"]
    replay = evaluate(base, graph, identity)
    if not np.array_equal(replay, baseline):
        raise RuntimeError(f"shared adapter graph baseline mismatch: {np.sum(replay != baseline)}")

    ranks = {}; models = {}; histories = {}; weights_report = {}
    for arm in ARMS:
        if arm == "uniform":
            weights = np.zeros(len(panel), dtype=np.float32); weights[train_mask] = 1.0
            for value in np.unique(identity[train_mask]):
                positions = np.flatnonzero(train_mask & (identity == value))
                weights[positions] /= len(positions)
            weights[train_mask] *= np.sum(train_mask) / np.sum(weights[train_mask])
        else:
            weights = identity_balanced_route_weights(
                identity, baseline, correctable[arm], train_mask,
                correctable_weight=args.correctable_weight,
                other_error_weight=args.other_error_weight,
                correct_control_weight=args.correct_control_weight,
            )
        model, history = train_arm(
            base, graph, train_mask, baseline == 1, weights, args,
        )
        adapted = transform(model, base, 512)
        ranks[arm] = evaluate(adapted, graph, identity)
        models[arm] = model.state_dict(); histories[arm] = history
        weights_report[arm] = {
            "train_correctable_queries": int(np.sum(train_mask & correctable[arm])),
            "mean": float(np.mean(weights[train_mask])),
            "minimum": float(np.min(weights[train_mask])),
            "maximum": float(np.max(weights[train_mask])),
        }
        print(f"completed shared adapter arm {arm}", flush=True)

    def role_metrics(mask: np.ndarray, offset: int) -> dict[str, object]:
        result = {}
        for index, arm in enumerate(ARMS):
            result[arm] = {
                "retrieval": retrieval(baseline[mask], ranks[arm][mask]),
                "formula_cluster_bootstrap_ci95": bootstrap(
                    formula[mask], baseline[mask], ranks[arm][mask],
                    args.bootstrap_draws, args.seed + offset + index,
                ),
            }
        return result

    validation = role_metrics(validation_mask, 100)
    confirmation = role_metrics(confirmation_mask, 200)
    paired = {
        f"correct_minus_{arm}": paired_rank_comparison(
            ranks["correct"][confirmation_mask], ranks[arm][confirmation_mask],
            formula[confirmation_mask], draws=args.bootstrap_draws,
            seed=args.seed + 300 + index,
        )
        for index, arm in enumerate(("uniform", "alignment_permuted"))
    }
    primary = confirmation["correct"]
    gates = {
        "confirmation_absolute_ci_positive": primary["formula_cluster_bootstrap_ci95"][0] > 0,
        "confirmation_corrected_exceeds_twice_introduced": (
            primary["retrieval"]["corrected_at_1"] > 2 * primary["retrieval"]["introduced_at_1"]
        ),
        **{
            f"confirmation_beats_{arm}_ci": paired[f"correct_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in ("uniform", "alignment_permuted")
        },
        "shared_query_reference_map": True,
        "single_clean_spectrum_inference": True,
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_CROSSVIEW_ROUTED_SHARED_ADAPTER_PASS"
            if all(gates.values()) else "CHEMAWARE_CROSSVIEW_ROUTED_SHARED_ADAPTER_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": True,
        "shared_embedding_result": True,
        "backbone_updated": False,
        "scope": "Low-rank cached-embedding reachability; formula role 0 train, role 1 diagnostic, role 2 confirmation; outer fold 4 untouched.",
        "claim_limit": "One-seed post-embedding adapter result, not a DreaMS backbone fine-tuning claim.",
        "method": {
            "adapter": "existing zero-initialized bounded Noise SharedResidualMetric",
            "loss": "identity listwise cross-entropy plus official-margin safety and embedding preservation",
            "chemical_injection": "identity-balanced training weights from cross-view orthogonal teacher correctability",
            "controls": ["uniform curriculum", "alignment-permuted teacher curriculum"],
            "same_initialization_schedule_candidates_optimizer": True,
        },
        "data": {
            "queries": int(len(panel)), "identities": int(len(np.unique(identity))),
            "train_queries": int(np.sum(train_mask)), "validation_queries": int(np.sum(validation_mask)),
            "confirmation_queries": int(np.sum(confirmation_mask)),
            "reachable_spectrum_rows": int(len(graph["rows"])), "formula_overlap": 0,
        },
        "route_weights": weights_report,
        "validation_diagnostic": validation,
        "held_confirmation": confirmation,
        "paired_confirmation": paired,
        "gates": gates,
        "configuration": {
            key: value for key, value in vars(args).items() if not isinstance(value, Path)
        },
        "runtime_seconds": time.time() - started,
        "provenance": {
            "manifest_sha256": sha256(args.manifest), "repeat_ledger_sha256": sha256(args.repeat_ledger),
            "consensus_report_sha256": sha256(args.consensus_report),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_shared_adapter_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "ranks.npz", query=panel, formula=formula, identity=identity,
            baseline_rank=baseline, **{f"{arm}_rank": ranks[arm] for arm in ARMS},
        )
        torch.save({
            "format": "chemaware_crossview_routed_shared_adapter_v1",
            "state_dict_by_arm": models, "histories": histories,
            "hidden_dim": args.hidden_dim, "residual_strength": args.residual_strength,
        }, temporary / "adapters.pt")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "validation": validation,
        "confirmation": confirmation, "paired": paired, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

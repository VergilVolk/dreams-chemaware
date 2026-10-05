"""Transfer the passed cross-view ChemAware actions into DreaMS PEFT.

The script reuses the official DreaMS loader, repository PEFT parametrization,
frozen-prefix accelerator, Noise/BioAware projected-gradient guard, and the
frozen cross-view action ledger.  Candidate information exists only in the
training loss.  The saved model is a single shared clean-spectrum encoder.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chemaware_candidate_residual_distillation_core import (
    event_calibrated_rankmax_loss,
    logmeanexp_segments,
    teacher_boundary_events,
)
from chemaware_crossview_consensus_core import crossview_policy
from chemaware_direct_training_core import (
    gradient_dot,
    gradient_norm,
    projected_guarded_auxiliary,
    virtual_adamw_descent_updates,
)
from chemaware_frozen_prefix_cache import FrozenPrefixSpectrumStore
from chemaware_iceberg_direct_core import stable_formula_folds
from dreams.models.chem_aware.peft_v3 import (
    DreaMSPEFTConfig,
    install_dreams_peft,
    load_peft_state_dict,
    peft_state_dict,
)
from noise_final_core import sha256_file
from train_e1_identity import load_base_model
from train_noise_final_r2_shared_encoder import SpectrumStore, forward_embeddings


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ARMS = ("zero_contrast", "reversed_contrast", "alignment_permuted", "correct")
ARMS = (
    "zero_contrast", "matched_random", "correct",
    "reversed_contrast", "alignment_permuted",
)
CONTROL_ARMS = tuple(arm for arm in ARMS if arm != "correct")


def retrieval(old_rank: np.ndarray, new_rank: np.ndarray) -> dict[str, object]:
    """Paired retrieval metrics kept local to avoid importing audit programs."""
    old_rank = np.asarray(old_rank)
    new_rank = np.asarray(new_rank)
    result: dict[str, object] = {
        "queries": int(len(old_rank)),
        "baseline_mrr": float(np.mean(1.0 / old_rank)),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank - 1.0 / old_rank)),
        "corrected_at_1": int(np.sum((old_rank > 1) & (new_rank == 1))),
        "introduced_at_1": int(np.sum((old_rank == 1) & (new_rank > 1))),
        "rank_improved": int(np.sum(new_rank < old_rank)),
        "rank_worsened": int(np.sum(new_rank > old_rank)),
    }
    for k in (1, 3, 5, 10, 20, 50):
        result[f"baseline_recall{k}"] = float(np.mean(old_rank <= k))
        result[f"recall{k}"] = float(np.mean(new_rank <= k))
        result[f"delta_recall{k}"] = float(
            np.mean(new_rank <= k) - np.mean(old_rank <= k)
        )
    result["risk_utility_at_1"] = (
        int(result["corrected_at_1"]) - 2 * int(result["introduced_at_1"])
    )
    return result


def bootstrap(
    formula: np.ndarray,
    old_rank: np.ndarray,
    new_rank: np.ndarray,
    draws: int,
    seed: int,
) -> list[float]:
    """Formula-cluster bootstrap for paired Recall@1 changes."""
    unique, inverse = np.unique(np.asarray(formula, dtype=str), return_inverse=True)
    delta = (new_rank == 1).astype(float) - (old_rank == 1).astype(float)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for left in range(0, draws, 500):
        right = min(left + 500, draws)
        selected = rng.integers(0, len(unique), size=(right - left, len(unique)))
        output[left:right] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(value) for value in np.quantile(output, (0.025, 0.975))]


def paired_rank_comparison(
    left_rank: np.ndarray,
    right_rank: np.ndarray,
    formula: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Compare two arms on exactly the same queries and formula clusters."""
    result = retrieval(right_rank, left_rank)
    result["formula_cluster_bootstrap_delta_recall1_ci95"] = bootstrap(
        formula, right_rank, left_rank, draws, seed,
    )
    return result


def proposed_local_candidate(
    selected_slot: np.ndarray,
    candidate_identity: np.ndarray,
    query: np.ndarray,
    body: dict[str, np.ndarray],
) -> np.ndarray:
    """Map a teacher identity slot to the unique local candidate molecule."""
    output = np.full(len(query), -1, dtype=np.int64)
    for index in np.flatnonzero(selected_slot >= 0):
        identity = str(candidate_identity[index, int(selected_slot[index])])
        left, right = map(
            int, body["query_ptr"][int(query[index]):int(query[index]) + 2],
        )
        candidates = body["molecule_ik14"][left:right].astype(str)
        match = np.flatnonzero(candidates == identity)
        if len(match) != 1:
            raise RuntimeError(
                "cross-view candidate identity does not map uniquely into its query"
            )
        output[index] = int(match[0])
    return output


def build_retrieval_arrays(
    panel: np.ndarray,
    body: dict[str, np.ndarray],
    rows: np.ndarray,
    official: np.ndarray,
    top_negatives: int,
) -> dict[str, object]:
    """Build the frozen query/candidate graph without importing audit drivers."""
    row_position = {int(row): index for index, row in enumerate(rows)}
    reachable = [body["query_row"][panel].astype(np.int64)]
    for query in panel:
        left, right = map(int, body["query_ptr"][query:query + 2])
        reachable.append(
            body["pair_candidate_row"][body["molecule_ptr"][left]:body["molecule_ptr"][right]]
        )
    reachable_rows = np.unique(np.concatenate(reachable)).astype(np.int64)
    base = np.asarray(
        official[[row_position[int(row)] for row in reachable_rows]], dtype=np.float32,
    )
    local = {int(row): index for index, row in enumerate(reachable_rows)}
    query_position = np.empty(len(panel), dtype=np.int64)
    positive_position = np.empty(len(panel), dtype=np.int64)
    negative_position = np.empty((len(panel), top_negatives), dtype=np.int64)
    candidate_position: list[np.ndarray] = []
    candidate_identity: list[np.ndarray] = []
    for index, query in enumerate(panel):
        qrow = int(body["query_row"][query])
        qpos = local[qrow]
        query_position[index] = qpos
        left, right = map(int, body["query_ptr"][query:query + 2])
        molecule_best = []
        all_rows = []
        all_identity = []
        labels = body["molecule_label"][left:right].astype(bool)
        for molecule in range(left, right):
            pleft, pright = map(int, body["molecule_ptr"][molecule:molecule + 2])
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
            print(f"built PEFT retrieval graph {index + 1}/{len(panel)}", flush=True)
    return {
        "rows": reachable_rows,
        "base": base,
        "query": query_position,
        "positive": positive_position,
        "negative": negative_position,
        "candidate": candidate_position,
        "candidate_identity": candidate_identity,
    }


def evaluate(
    embeddings: np.ndarray,
    graph: dict[str, object],
    panel_identity: np.ndarray,
) -> np.ndarray:
    """Compute molecule-level ranks with max aggregation over reference spectra."""
    rank = np.empty(len(panel_identity), dtype=np.int16)
    for index, truth in enumerate(panel_identity.astype(str)):
        q = embeddings[graph["query"][index]]
        positions = graph["candidate"][index]
        identities = graph["candidate_identity"][index]
        scores = embeddings[positions] @ q
        molecules = np.unique(identities)
        molecule_score = np.asarray([
            np.max(scores[identities == value]) for value in molecules
        ])
        positive = float(molecule_score[molecules == truth][0])
        rank[index] = 1 + int(np.sum(molecule_score[molecules != truth] >= positive))
    return rank


def numerical_replay_audit(
    panel: np.ndarray,
    formula: np.ndarray,
    expected_rank: np.ndarray,
    observed_rank: np.ndarray,
    observed_margin: np.ndarray,
    *,
    maximum_fraction: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, object]]]:
    """Fail closed on data drift while isolating rare cross-BLAS boundaries."""
    panel = np.asarray(panel, dtype=np.int64)
    formula = np.asarray(formula).astype(str)
    expected_rank = np.asarray(expected_rank, dtype=np.int64)
    observed_rank = np.asarray(observed_rank, dtype=np.int64)
    observed_margin = np.asarray(observed_margin, dtype=np.float64)
    if not (
        len(panel) == len(formula) == len(expected_rank)
        == len(observed_rank) == len(observed_margin)
    ):
        raise ValueError("numerical replay audit arrays do not align")
    if not 0.0 < maximum_fraction <= 0.01:
        raise ValueError("numerical replay allowance must be in (0, 0.01]")
    stable = observed_rank == expected_rank
    mismatch = np.flatnonzero(~stable)
    maximum = max(1, int(np.ceil(maximum_fraction * len(panel))))
    audit: list[dict[str, object]] = [
        {
            "panel_position": int(index),
            "manifest_query": int(panel[index]),
            "formula": str(formula[index]),
            "expected_rank": int(expected_rank[index]),
            "observed_rank": int(observed_rank[index]),
            "observed_top1_margin": float(observed_margin[index]),
        }
        for index in mismatch
    ]
    if len(mismatch) > maximum:
        raise RuntimeError(
            "official retrieval replay drift exceeds the frozen numerical-boundary "
            f"allowance: observed={len(mismatch)} maximum={maximum}; first="
            + json.dumps(audit[:10], sort_keys=True)
        )
    return stable, mismatch, maximum, audit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--repeat-ledger", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v3/repeat_ledger.npz")
    parser.add_argument("--repeat-report", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v3/report.json")
    parser.add_argument("--consensus-report", type=Path, default=ROOT / "data/validation/chemaware_crossview_consensus_teacher_v2/report.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all"), required=True)
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--event-batch-size", type=int, default=16)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--references-per-candidate", type=int, default=4)
    parser.add_argument("--reference-temperature", type=float, default=0.002)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--last-blocks", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--target-margin", type=float, default=0.005)
    parser.add_argument("--protection-slack", type=float, default=0.002)
    parser.add_argument("--event-huber", type=float, default=0.02)
    parser.add_argument("--lambda-event", type=float, default=1.0)
    parser.add_argument("--lambda-safety", type=float, default=2.0)
    parser.add_argument("--safety-slack", type=float, default=0.005)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--maximum-event-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--prefix-cache-batch-size", type=int, default=8)
    parser.add_argument("--frozen-prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_rows(
    model: torch.nn.Module,
    store: SpectrumStore | FrozenPrefixSpectrumStore,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    if isinstance(store, FrozenPrefixSpectrumStore):
        return store.forward(model, rows, device, batch_size, amp).float()
    output = []
    for left in range(0, len(rows), batch_size):
        spectra = store.get(rows[left:left + batch_size]).to(device)
        output.append(forward_embeddings(model, spectra, amp).float())
    return torch.cat(output, dim=0)


def encode_positions(
    model: torch.nn.Module,
    store: SpectrumStore | FrozenPrefixSpectrumStore,
    graph_rows: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, np.ndarray]:
    unique, inverse = np.unique(np.asarray(positions, dtype=np.int64), return_inverse=True)
    encoded = forward_rows(
        model, store, graph_rows[unique], device, args.eval_batch_size, args.amp,
    )
    return encoded, inverse


def clean_loss(
    model: torch.nn.Module,
    store: SpectrumStore | FrozenPrefixSpectrumStore,
    graph_rows: np.ndarray,
    base: np.ndarray,
    graph: dict[str, object],
    baseline_rank: np.ndarray,
    index: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    qpos = graph["query"][index]
    ppos = graph["positive"][index]
    npos = graph["negative"][index]
    joined = np.concatenate((qpos, ppos, npos.reshape(-1)))
    encoded, inverse = encode_positions(
        model, store, graph_rows, joined, device, args,
    )
    nq = len(index); nn = npos.shape[1]
    q = encoded[torch.from_numpy(inverse[:nq]).to(device)]
    p = encoded[torch.from_numpy(inverse[nq:2 * nq]).to(device)]
    n = encoded[torch.from_numpy(inverse[2 * nq:]).to(device)].reshape(nq, nn, -1)
    q0 = torch.from_numpy(base[qpos]).to(device)
    p0 = torch.from_numpy(base[ppos]).to(device)
    n0 = torch.from_numpy(base[npos]).to(device)
    positive = torch.sum(q * p, dim=1)
    negative = torch.einsum("bd,bkd->bk", q, n)
    rank = F.cross_entropy(
        torch.cat((positive[:, None], negative), dim=1) / args.temperature,
        torch.zeros(nq, dtype=torch.long, device=device),
    )
    official_margin = torch.sum(q0 * p0, dim=1) - torch.max(
        torch.einsum("bd,bkd->bk", q0, n0), dim=1,
    ).values
    student_margin = positive - torch.max(negative, dim=1).values
    safe = torch.from_numpy(baseline_rank[index] == 1).to(device)
    safety = (
        F.relu(official_margin[safe] - args.safety_slack - student_margin[safe]).mean()
        if bool(safe.any()) else rank * 0.0
    )
    official_unique = torch.from_numpy(base[np.unique(joined)]).to(device)
    preserve = torch.clamp(
        1.0 - torch.sum(encoded * official_unique, dim=1), min=0.0,
    ).mean()
    value = rank + args.lambda_safety * safety + args.lambda_preserve * preserve
    return value, {"rank": rank, "safety": safety, "preserve": preserve}


def event_loss(
    model: torch.nn.Module,
    store: SpectrumStore | FrozenPrefixSpectrumStore,
    graph_rows: np.ndarray,
    base: np.ndarray,
    query_position: np.ndarray,
    candidate_reference: np.ndarray,
    reference_valid: np.ndarray,
    event: dict[str, np.ndarray],
    index: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    qpos = query_position[index]
    role = event["role"][index]
    if np.any(role == 0):
        raise RuntimeError("event mini-batch contains an unsupported zero-weight row")
    chosen_slot = np.stack(
        (event["positive"][index], event["negative"][index]), axis=1,
    ).astype(np.int64)
    row = np.arange(len(index))[:, None]
    rpos = candidate_reference[index[:, None], chosen_slot]
    rvalid = reference_valid[index[:, None], chosen_slot]
    if np.any(np.sum(rvalid, axis=2) == 0):
        raise RuntimeError("an active boundary candidate has no reference spectrum")
    flat_position = []
    flat_query = []
    segment_ptr = [0]
    for query_index in range(len(index)):
        for side in range(2):
            values = rpos[query_index, side, rvalid[query_index, side]]
            flat_position.extend(values.tolist())
            flat_query.extend([query_index] * len(values))
            segment_ptr.append(len(flat_position))
    flat_position = np.asarray(flat_position, dtype=np.int64)
    joined = np.concatenate((qpos, flat_position))
    encoded, inverse = encode_positions(
        model, store, graph_rows, joined, device, args,
    )
    nq = len(index)
    q = encoded[torch.from_numpy(inverse[:nq]).to(device)]
    refs = encoded[torch.from_numpy(inverse[nq:]).to(device)]
    q0 = torch.from_numpy(base[qpos]).to(device)
    refs0 = torch.from_numpy(base[flat_position]).to(device)
    flat_query_tensor = torch.as_tensor(flat_query, dtype=torch.long, device=device)
    student_pair_score = torch.sum(q[flat_query_tensor] * refs, dim=1)
    official_pair_score = torch.sum(q0[flat_query_tensor] * refs0, dim=1)
    student_score = logmeanexp_segments(
        student_pair_score, segment_ptr, args.reference_temperature,
    ).reshape(nq, 2)
    official_score = logmeanexp_segments(
        official_pair_score, segment_ptr, args.reference_temperature,
    ).reshape(nq, 2)
    return event_calibrated_rankmax_loss(
        student_score, official_score,
        torch.ones((nq, 2), dtype=torch.bool, device=device),
        torch.zeros(nq, dtype=torch.long, device=device),
        torch.ones(nq, dtype=torch.long, device=device),
        torch.from_numpy(role).to(device),
        torch.ones(nq, device=device),
        target_margin=args.target_margin,
        protection_slack=args.protection_slack,
        huber_beta=args.event_huber,
    )


def candidate_reference_views(
    panel: np.ndarray,
    body: dict[str, np.ndarray],
    graph: dict[str, object],
    views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep several competitive references while replaying the exact baseline.

    Candidate slots retain manifest order, which is also the teacher action
    coordinate system.  Selection of the small reference set uses only frozen
    official similarities; scoring within that set is recomputed by the
    student at every optimization step.
    """
    if views < 1:
        raise ValueError("references per candidate must be positive")
    local = {int(row): index for index, row in enumerate(graph["rows"])}
    base = np.asarray(graph["base"], dtype=np.float32)
    maximum = max(
        int(body["query_ptr"][int(query) + 1] - body["query_ptr"][int(query)])
        for query in panel
    )
    reference = np.full((len(panel), maximum, views), -1, dtype=np.int64)
    reference_valid = np.zeros_like(reference, dtype=bool)
    positive = np.empty(len(panel), dtype=np.int64)
    baseline = np.empty(len(panel), dtype=np.int64)
    candidate_score = np.full((len(panel), maximum), -np.inf, dtype=np.float32)
    for index, query in enumerate(map(int, panel)):
        qpos = int(graph["query"][index])
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        positive[index] = int(np.flatnonzero(labels)[0])
        molecule_scores = []
        for slot, molecule in enumerate(range(left, right)):
            pair_left, pair_right = map(
                int, body["molecule_ptr"][molecule:molecule + 2],
            )
            rows = body["pair_candidate_row"][pair_left:pair_right].astype(np.int64)
            positions = np.asarray([local[int(row)] for row in rows], dtype=np.int64)
            scores = base[positions] @ base[qpos]
            # Stable ordering makes exact ties reproducible across platforms.
            selected = np.argsort(-scores, kind="stable")[:views]
            count = len(selected)
            reference[index, slot, :count] = positions[selected]
            reference_valid[index, slot, :count] = True
            candidate_score[index, slot] = float(np.max(scores))
            molecule_scores.append(float(candidate_score[index, slot]))
        baseline[index] = int(np.argmax(molecule_scores))
    return reference, reference_valid, positive, baseline, candidate_score


def matched_random_events(
    correct_event: dict[str, np.ndarray],
    baseline_rank: np.ndarray,
    baseline_candidate: np.ndarray,
    positive: np.ndarray,
    candidate_score: np.ndarray,
    identity: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Construct a dose- and difficulty-matched non-chemical curriculum.

    Matching is performed only inside the optimizer-visible formula roles.
    Correct-teacher identities are excluded from the pool.  This separates
    chemistry-based event selection from the easier explanation that any
    equally hard collection of labelled retrieval errors would work.
    """
    baseline_rank = np.asarray(baseline_rank, dtype=np.int64)
    baseline_candidate = np.asarray(baseline_candidate, dtype=np.int64)
    positive = np.asarray(positive, dtype=np.int64)
    identity = np.asarray(identity).astype(str)
    n = len(baseline_rank)
    if candidate_score.shape[0] != n:
        raise ValueError("matched-random candidate scores do not align")
    row = np.arange(n)
    score = np.asarray(candidate_score, dtype=np.float64)
    negative_score = score.copy()
    negative_score[row, positive] = -np.inf
    top_negative = np.argmax(negative_score, axis=1).astype(np.int64)
    active_correct = np.asarray(correct_event["active"], dtype=bool)
    blocked_identity = set(identity[active_correct])
    repeat_count = {value: int(np.sum(identity == value)) for value in np.unique(identity)}
    candidate_count = np.sum(np.isfinite(score), axis=1)

    def margin_for(index: np.ndarray, role_code: int, negative: np.ndarray) -> np.ndarray:
        return score[index, positive[index]] - score[index, negative]

    rng = np.random.default_rng(seed)
    selected_by_role: dict[int, np.ndarray] = {}
    matching_audit = {}
    for role_code in (1, 2):
        target = np.flatnonzero(np.asarray(correct_event["role"]) == role_code)
        eligible_role = baseline_rank > 1 if role_code == 1 else baseline_rank == 1
        pool = np.flatnonzero(
            eligible_role & ~active_correct
            & np.asarray([value not in blocked_identity for value in identity])
        )
        if len(pool) < len(target):
            raise RuntimeError(f"insufficient matched-random role-{role_code} pool")
        target_negative = np.asarray(correct_event["negative"])[target]
        pool_negative = (
            baseline_candidate[pool] if role_code == 1 else top_negative[pool]
        )
        target_feature = np.column_stack((
            margin_for(target, role_code, target_negative),
            np.log1p(candidate_count[target]),
            np.log1p([repeat_count[value] for value in identity[target]]),
            np.log1p(baseline_rank[target]),
        ))
        pool_feature = np.column_stack((
            margin_for(pool, role_code, pool_negative),
            np.log1p(candidate_count[pool]),
            np.log1p([repeat_count[value] for value in identity[pool]]),
            np.log1p(baseline_rank[pool]),
        ))
        centre = np.mean(pool_feature, axis=0)
        scale = np.std(pool_feature, axis=0)
        scale[scale < 1e-8] = 1.0
        target_z = (target_feature - centre) / scale
        pool_z = (pool_feature - centre) / scale
        available = np.ones(len(pool), dtype=bool)
        chosen = []
        # Randomized-but-seeded order prevents a fixed ledger ordering from
        # resolving nearest-neighbour conflicts in favour of one acquisition run.
        for target_position in rng.permutation(len(target)):
            distance = np.sum((pool_z - target_z[target_position]) ** 2, axis=1)
            distance[~available] = np.inf
            winner = int(np.argmin(distance))
            if not np.isfinite(distance[winner]):
                raise RuntimeError("matched-random greedy assignment exhausted")
            chosen.append((int(target_position), winner))
            available[winner] = False
        chosen.sort()
        selected = pool[np.asarray([winner for _, winner in chosen], dtype=np.int64)]
        selected_by_role[role_code] = selected
        selected_feature = pool_feature[
            np.asarray([winner for _, winner in chosen], dtype=np.int64)
        ]
        pooled_scale = np.sqrt(
            0.5 * (np.var(target_feature, axis=0) + np.var(selected_feature, axis=0))
        )
        smd = np.divide(
            np.abs(np.mean(target_feature, axis=0) - np.mean(selected_feature, axis=0)),
            pooled_scale,
            out=np.zeros_like(pooled_scale), where=pooled_scale > 1e-8,
        )
        matching_audit[str(role_code)] = {
            "target": int(len(target)),
            "matched": int(len(selected)),
            "standardized_mean_difference": {
                key: float(value) for key, value in zip(
                    ("official_margin", "candidate_count", "identity_repeat_count", "baseline_rank"),
                    smd,
                )
            },
            "maximum_absolute_smd": float(np.max(smd)),
        }

    role = np.zeros(n, dtype=np.int8)
    event_positive = np.full(n, -1, dtype=np.int64)
    event_negative = np.full(n, -1, dtype=np.int64)
    correction = selected_by_role[1]
    protection = selected_by_role[2]
    role[correction] = 1; role[protection] = 2
    event_positive[role > 0] = positive[role > 0]
    event_negative[correction] = baseline_candidate[correction]
    event_negative[protection] = top_negative[protection]
    event = {
        "positive": event_positive,
        "negative": event_negative,
        "role": role,
        "active": role > 0,
        "confidence": np.zeros(n, dtype=np.float32),
    }
    return event, {
        "kind": "greedy_nearest_without_replacement",
        "correct_teacher_identities_excluded": True,
        "features": [
            "official_margin", "candidate_count",
            "identity_repeat_count", "baseline_rank",
        ],
        "roles": matching_audit,
    }


@torch.inference_mode()
def encode_all(
    model: torch.nn.Module,
    store: SpectrumStore | FrozenPrefixSpectrumStore,
    graph_rows: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    label: str,
) -> np.ndarray:
    model.eval(); output = np.empty((len(graph_rows), model.head.out_features), dtype=np.float32)
    started = time.time()
    for left in range(0, len(graph_rows), args.eval_batch_size):
        right = min(left + args.eval_batch_size, len(graph_rows))
        output[left:right] = forward_rows(
            model, store, graph_rows[left:right], device,
            args.eval_batch_size, args.amp,
        ).cpu().numpy()
        if right == len(graph_rows) or right % (20 * args.eval_batch_size) == 0:
            print(f"[{label}] {right}/{len(graph_rows)} rows elapsed={time.time()-started:.0f}s", flush=True)
    return output


def retrieval_diagnostics(
    embeddings: np.ndarray,
    graph: dict[str, object],
    panel_identity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact molecule-level rank, AUC credit, and top-1 margin."""
    n = len(panel_identity)
    rank = np.empty(n, dtype=np.int16)
    auc_credit = np.empty(n, dtype=np.float64)
    comparisons = np.empty(n, dtype=np.int16)
    margin = np.empty(n, dtype=np.float32)
    for index, truth in enumerate(panel_identity.astype(str)):
        query = embeddings[graph["query"][index]]
        positions = graph["candidate"][index]
        identities = graph["candidate_identity"][index]
        pair_score = embeddings[positions] @ query
        molecules = np.unique(identities)
        molecule_score = np.asarray([
            np.max(pair_score[identities == value]) for value in molecules
        ])
        positive = float(molecule_score[molecules == truth][0])
        negative = molecule_score[molecules != truth]
        rank[index] = 1 + int(np.sum(negative >= positive))
        comparisons[index] = len(negative)
        auc_credit[index] = float(
            np.sum(positive > negative) + 0.5 * np.sum(positive == negative)
        )
        margin[index] = positive - float(np.max(negative))
    return {
        "rank": rank,
        "auc_credit": auc_credit,
        "comparisons": comparisons,
        "query_auc": auc_credit / comparisons,
        "top1_margin": margin,
    }


def auc_metrics(
    diagnostics: dict[str, np.ndarray],
    formula: np.ndarray,
    baseline: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    formula = np.asarray(formula).astype(str)
    query_auc = np.asarray(diagnostics["query_auc"], dtype=np.float64)
    result = {
        "micro_auc": float(
            np.sum(diagnostics["auc_credit"]) / np.sum(diagnostics["comparisons"])
        ),
        "macro_query_auc": float(np.mean(query_auc)),
        "macro_formula_auc": float(np.mean([
            np.mean(query_auc[formula == value]) for value in np.unique(formula)
        ])),
        "mean_top1_margin": float(np.mean(diagnostics["top1_margin"])),
    }
    if baseline is not None:
        base = auc_metrics(baseline, formula)
        result.update({f"delta_{key}": value - base[key] for key, value in result.items()})
    return result


def bootstrap_auc_delta(
    formula: np.ndarray,
    baseline: dict[str, np.ndarray],
    adapted: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, list[float]]:
    """Formula-cluster bootstrap for micro and macro-formula AUC deltas."""
    formula = np.asarray(formula).astype(str)
    unique = np.unique(formula)
    credit_delta = np.asarray(adapted["auc_credit"] - baseline["auc_credit"])
    comparisons = np.asarray(baseline["comparisons"], dtype=np.float64)
    query_delta = np.asarray(adapted["query_auc"] - baseline["query_auc"])
    cluster_credit = np.asarray([
        np.sum(credit_delta[formula == value]) for value in unique
    ])
    cluster_comparisons = np.asarray([
        np.sum(comparisons[formula == value]) for value in unique
    ])
    cluster_macro = np.asarray([
        np.mean(query_delta[formula == value]) for value in unique
    ])
    rng = np.random.default_rng(seed)
    micro = np.empty(draws, dtype=np.float64)
    macro = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(unique), size=len(unique))
        micro[draw] = (
            np.sum(cluster_credit[selected]) / np.sum(cluster_comparisons[selected])
        )
        macro[draw] = np.mean(cluster_macro[selected])
    return {
        "delta_micro_auc_ci95": np.quantile(micro, (0.025, 0.975)).tolist(),
        "delta_macro_formula_auc_ci95": np.quantile(macro, (0.025, 0.975)).tolist(),
    }


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.epochs < 1 or args.batch_queries < 1 or args.event_batch_size < 1
            or args.references_per_candidate < 1 or args.reference_temperature <= 0
            or args.last_blocks != 1 or not 0 < args.maximum_event_gradient_ratio <= 1):
        raise ValueError("invalid bounded PEFT pilot configuration")
    required = (
        args.manifest, args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.repeat_ledger,
        args.repeat_report, args.consensus_report, args.data, args.official_checkpoint,
        args.architecture_checkpoint,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    torch.set_num_threads(max(1, args.torch_threads)); seed_everything(args.seed)
    consensus = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    if (consensus.get("status") != "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_PASS"
            or not all(consensus.get("gates", {}).values())):
        raise RuntimeError("a fully passing cross-view teacher is required")
    repeat_report = json.loads(args.repeat_report.read_text(encoding="utf-8"))
    if repeat_report.get("status") != "CHEMAWARE_ORTHOGONAL_TEACHER_REPEAT_CONSISTENCY_COMPLETE":
        raise RuntimeError("repeat teacher report has an unsupported status")
    expected_provenance = repeat_report.get("provenance", {})
    observed_provenance = {
        "manifest_sha256": sha256_file(args.manifest),
        "official_embeddings_sha256": sha256_file(
            args.token_dir / "official_embeddings_f32.npy"
        ),
        "repeat_ledger_sha256": sha256_file(args.repeat_ledger),
        "repeat_report_sha256": sha256_file(args.repeat_report),
    }
    provenance_gates = {
        "manifest_matches_repeat_teacher": (
            observed_provenance["manifest_sha256"]
            == expected_provenance.get("manifest_sha256")
        ),
        "official_embeddings_match_repeat_teacher": (
            observed_provenance["official_embeddings_sha256"]
            == expected_provenance.get("official_embeddings_sha256")
        ),
        "repeat_ledger_matches_consensus": (
            observed_provenance["repeat_ledger_sha256"]
            == consensus.get("provenance", {}).get("repeat_ledger_sha256")
        ),
        "repeat_report_matches_consensus": (
            observed_provenance["repeat_report_sha256"]
            == consensus.get("provenance", {}).get("teacher_report_sha256")
        ),
    }
    if not all(provenance_gates.values()):
        raise RuntimeError(
            "frozen ChemAware provenance mismatch: "
            + json.dumps(provenance_gates, sort_keys=True)
        )
    requested_arms = ARMS if args.arm == "all" else (args.arm,)
    setting = consensus["selection"]
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.repeat_ledger, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    formula0 = ledger["formula"].astype(str); identity0 = ledger["identity"].astype(str)
    role = stable_formula_folds(formula0, 3, args.split_seed)
    train_original = np.flatnonzero(role < 2); evaluation_original = np.flatnonzero(role == 2)
    if set(formula0[train_original]) & set(formula0[evaluation_original]):
        raise RuntimeError("training and evaluation formulas overlap")
    order = np.concatenate((train_original, evaluation_original))
    panel = ledger["query"][order].astype(np.int64)
    formula = formula0[order]; identity = identity0[order]
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    retrieval_graph = build_retrieval_arrays(panel, body, rows, official, args.top_negatives)
    graph_rows = retrieval_graph["rows"].astype(np.int64)
    base = retrieval_graph["base"].astype(np.float32)
    (
        candidate_reference, reference_valid, positive,
        baseline_candidate, candidate_score,
    ) = candidate_reference_views(
        panel, body, retrieval_graph, args.references_per_candidate,
    )
    baseline_rank = evaluate(base, retrieval_graph, identity)
    baseline_diagnostics = retrieval_diagnostics(base, retrieval_graph, identity)
    if not np.array_equal(baseline_rank, baseline_diagnostics["rank"]):
        raise RuntimeError("baseline rank and AUC diagnostic implementations disagree")
    expected_baseline = ledger["baseline_rank"][order].astype(np.int64)
    (
        replay_stable,
        replay_mismatch,
        maximum_platform_mismatches,
        replay_mismatch_audit,
    ) = numerical_replay_audit(
        panel, formula, expected_baseline, baseline_rank,
        baseline_diagnostics["top1_margin"],
    )
    train_count = len(train_original)
    evaluation = np.flatnonzero(
        (np.arange(len(panel), dtype=np.int64) >= train_count) & replay_stable
    )
    # One clean query per identity prevents acquisition-rich molecules from
    # increasing the base loss, while every verified action view remains in
    # the separate identity-routed chemical stream.
    rng = np.random.default_rng(args.seed + 101)
    clean_query = []
    for value in sorted(np.unique(identity[:train_count])):
        candidates = np.flatnonzero(identity[:train_count] == value)
        clean_query.append(int(candidates[int(rng.integers(len(candidates)))]))
    clean_query = np.asarray(clean_query, dtype=np.int64)
    schedules = []
    for _ in range(args.epochs):
        shuffled = clean_query[rng.permutation(len(clean_query))]
        schedules.append([
            shuffled[left:left + args.batch_queries]
            for left in range(0, len(shuffled), args.batch_queries)
        ])

    teacher_by_arm: dict[str, dict[str, object]] = {}
    preflight_by_arm = {}
    needed_teacher_arms = set(requested_arms) & set(TEACHER_ARMS)
    if "matched_random" in requested_arms:
        needed_teacher_arms.add("correct")
    for arm in TEACHER_ARMS:
        if arm not in needed_teacher_arms:
            continue
        rank, selected, score, support = crossview_policy(
            ledger["baseline_rank"], ledger["candidate_identity"],
            ledger["candidate_valid"], ledger["proposal_rank"],
            ledger[f"{arm}_candidate_utility"], identity0,
            threshold=float(setting["threshold"]),
            aggregation=str(setting["aggregation"]),
            min_context=int(setting["min_context"]),
        )
        expected = consensus["held_confirmation"][arm]["retrieval"]
        teacher_replay = retrieval(
            ledger["baseline_rank"][evaluation_original], rank[evaluation_original],
        )
        if any(
            int(teacher_replay[key]) != int(expected[key])
            for key in ("corrected_at_1", "introduced_at_1", "risk_utility_at_1")
        ):
            raise RuntimeError(f"cross-view teacher arm {arm} did not exactly replay")
        proposal = proposed_local_candidate(
            selected, ledger["candidate_identity"], ledger["query"], body,
        )[order][:train_count]
        selected_train = selected[order][:train_count]
        normalized_slot = np.where(selected_train >= 0, 0, -1)
        event_batch = teacher_boundary_events(
            baseline_rank[:train_count], rank[order][:train_count, None],
            baseline_candidate[:train_count], proposal[:, None],
            normalized_slot, positive[:train_count],
        )
        event = {
            "positive": event_batch.positive,
            "negative": event_batch.negative,
            "role": event_batch.role,
            "active": event_batch.active,
            # Retained only for audit.  Magnitude is deliberately not used as
            # a gradient weight because utilities across arms are not calibrated.
            "confidence": np.maximum(
                score[order][:train_count] - float(setting["threshold"]), 0.0,
            ).astype(np.float32),
        }
        # The legacy repeat ledger froze ranks but not raw pair scores.  A tiny
        # number of BLAS-dependent boundary queries can therefore replay to a
        # different rank on another platform.  They receive exactly zero
        # chemical supervision and are absent from the paired held evaluation;
        # they can never masquerade as an embedding improvement.
        unstable_train = ~replay_stable[:train_count]
        event["positive"][unstable_train] = -1
        event["negative"][unstable_train] = -1
        event["role"][unstable_train] = 0
        event["active"][unstable_train] = False
        event["confidence"][unstable_train] = 0.0
        active_event = np.flatnonzero(event["active"])
        teacher_by_arm[arm] = {
            "rank": rank,
            "selected": selected,
            "support": support,
            "event": event,
            "active_event": active_event,
        }
        preflight_by_arm[arm] = {
            "teacher_selected": int(np.sum(selected_train >= 0)),
            "teacher_corrections": int(np.sum(event["role"] == 1)),
            "teacher_protections": int(np.sum(event["role"] == 2)),
            "teacher_active_events": int(len(active_event)),
            "unsupported_zero_weight": int(
                np.sum((selected_train >= 0) & (event["role"] == 0))
            ),
            "teacher_retrieval_replayed": teacher_replay,
        }

    if "matched_random" in requested_arms:
        matched_event, matched_audit = matched_random_events(
            teacher_by_arm["correct"]["event"], baseline_rank[:train_count],
            baseline_candidate[:train_count], positive[:train_count],
            candidate_score[:train_count], identity[:train_count], args.seed + 509,
        )
        maximum_matching_smd = max(
            value["maximum_absolute_smd"]
            for value in matched_audit["roles"].values()
        )
        if maximum_matching_smd >= 0.20:
            raise RuntimeError(
                f"matched-random balance failed: maximum SMD={maximum_matching_smd}"
            )
        active_event = np.flatnonzero(matched_event["active"])
        teacher_by_arm["matched_random"] = {
            "rank": None, "selected": None, "support": None,
            "event": matched_event, "active_event": active_event,
        }
        preflight_by_arm["matched_random"] = {
            "teacher_selected": int(len(active_event)),
            "teacher_corrections": int(np.sum(matched_event["role"] == 1)),
            "teacher_protections": int(np.sum(matched_event["role"] == 2)),
            "teacher_active_events": int(len(active_event)),
            "unsupported_zero_weight": 0,
            "teacher_retrieval_replayed": None,
            "matching": matched_audit,
        }
    preflight_by_arm = {
        arm: preflight_by_arm[arm] for arm in requested_arms
    }

    preflight = {
        "status": "CHEMAWARE_CROSSVIEW_EVENT_PEFT_PREFLIGHT_PASS",
        "requested_arm": args.arm,
        "arms": list(requested_arms),
        "train_queries": int(train_count),
        "train_identities": int(len(clean_query)),
        "evaluation_queries": int(len(evaluation)),
        "evaluation_queries_before_numerical_boundary_exclusion": int(
            len(evaluation_original)
        ),
        "numerical_boundary_exclusions": int(len(replay_mismatch)),
        "numerical_boundary_exclusion_fraction": float(
            len(replay_mismatch) / len(panel)
        ),
        "numerical_boundary_exclusion_limit": int(maximum_platform_mismatches),
        "numerical_boundary_audit": replay_mismatch_audit,
        "provenance": observed_provenance,
        "reachable_spectra": int(len(graph_rows)),
        "references_per_candidate": int(args.references_per_candidate),
        "reference_temperature": float(args.reference_temperature),
        "per_arm": preflight_by_arm,
        "optimizer_steps_per_arm": int(sum(map(len, schedules))),
        "optimizer_steps_total": int(sum(map(len, schedules)) * len(requested_arms)),
        "contracts": {
            "teacher_replayed": True,
            "formula_role2_optimizer_untouched": True,
            "candidate_inputs_at_inference": False,
            "query_reference_encoder_shared": True,
            "nonselected_chemical_weight_zero": True,
            "utility_magnitude_not_loss_weight": True,
            "matched_random_has_correct_arm_event_dose": (
                "matched_random" not in requested_arms
                or preflight_by_arm["matched_random"]["teacher_active_events"]
                == preflight_by_arm["correct"]["teacher_active_events"]
            ),
            "matched_random_maximum_smd_below_0_20": (
                "matched_random" not in requested_arms
                or max(
                    value["maximum_absolute_smd"]
                    for value in preflight_by_arm["matched_random"]["matching"]["roles"].values()
                ) < 0.20
            ),
            "correction_and_protection_events_trained": True,
            "all_arms_share_clean_schedule": True,
            "all_arms_start_same_peft_state": True,
            "all_arms_share_one_frozen_prefix_cache": True,
            "outer_fold4_untouched": True,
            "frozen_provenance_exact": all(provenance_gates.values()),
            "numerical_boundary_queries_have_zero_chemical_weight": bool(
                all(
                    not np.any(
                        teacher_by_arm[arm]["event"]["active"][~replay_stable[:train_count]]
                    )
                    for arm in requested_arms
                )
            ),
            "numerical_boundary_queries_excluded_from_paired_evaluation": bool(
                np.all(replay_stable[evaluation])
            ),
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2)); return
    device = torch.device(args.device)
    if (device.type != "cuda" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1):
        raise RuntimeError("DreaMS PEFT pilot requires exactly one available CUDA device")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir()
    raw_store = SpectrumStore(args.data, graph_rows, args.n_highest_peaks)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    model.eval()
    capacity = install_dreams_peft(model, DreaMSPEFTConfig(
        last_blocks=args.last_blocks, rank=args.rank, alpha=args.alpha,
        adapt_attention=True, adapt_feed_forward=True, adapt_head=True,
    ))
    store: SpectrumStore | FrozenPrefixSpectrumStore = raw_store
    if args.frozen_prefix_cache:
        store = FrozenPrefixSpectrumStore(
            model, raw_store, device, args.prefix_cache_batch_size,
            last_blocks=args.last_blocks,
        )
    initial_peft_state = peft_state_dict(model)
    parameters = [value for value in model.parameters() if value.requires_grad]
    histories = {}; held_results = {}; held_rank_by_arm = {}; held_diagnostics_by_arm = {}
    preservation_by_arm = {}; artifact_gates_by_arm = {}; arm_states = {}
    transmission_by_arm = {}
    early_stop = {"triggered": False, "stage": "not_reached"}
    replay_positions = np.linspace(0, len(graph_rows) - 1, 64, dtype=np.int64)
    for arm_index, arm in enumerate(requested_arms):
        load_peft_state_dict(model, initial_peft_state)
        seed_everything(args.seed)
        replay = forward_rows(
            model, store, graph_rows[replay_positions], device,
            args.eval_batch_size, False,
        ).detach().cpu().numpy()
        replay_cosine = np.sum(replay * base[replay_positions], axis=1)
        if float(np.min(replay_cosine)) < 0.999:
            raise RuntimeError(
                f"zero-init PEFT official replay drift for {arm}: "
                f"{float(np.min(replay_cosine))}"
            )
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay,
        )
        event_rng = np.random.default_rng(args.seed + 701)
        event = teacher_by_arm[arm]["event"]
        active_event = teacher_by_arm[arm]["active_event"]
        history = []
        transmission_values = {key: [] for key in (
            "projection_retention", "clip_event", "optimizer_update_action_fraction",
            "optimizer_update_action_alignment", "optimizer_attributable_alignment",
            "virtual_step_relative_error",
        )}
        for epoch, batches in enumerate(schedules, start=1):
            totals = {key: 0.0 for key in (
                "primary", "rank", "safety", "preserve", "event",
                "event_raw_cosine", "event_safe_ratio", "event_projected",
                "event_capped", "event_projection_retention", "grad_norm",
            )}
            seen = 0; model.eval()
            for index in batches:
                primary, pieces = clean_loss(
                    model, store, graph_rows, base, retrieval_graph,
                    baseline_rank, index, device, args,
                )
                optimizer.zero_grad(set_to_none=True)
                event_value = primary * 0.0
                geometry = {
                    "auxiliary_raw_cosine": 0.0,
                    "auxiliary_safe_to_primary_ratio": 0.0,
                    "auxiliary_conflict_projected": False,
                    "auxiliary_norm_capped": False,
                    "auxiliary_raw_norm": 0.0,
                    "auxiliary_safe_norm": 0.0,
                }
                audit_action = False
                safe_gradient: list[torch.Tensor | None] = []
                primary_gradient: tuple[torch.Tensor | None, ...] = tuple()
                if len(active_event):
                    event_index = event_rng.choice(
                        active_event,
                        size=min(args.event_batch_size, len(active_event)),
                        replace=len(active_event) < args.event_batch_size,
                    ).astype(np.int64)
                    event_value, _ = event_loss(
                        model, store, graph_rows, base,
                        retrieval_graph["query"][:train_count],
                        candidate_reference[:train_count], reference_valid[:train_count],
                        event, event_index, device, args,
                    )
                    primary_gradient = torch.autograd.grad(
                        primary, parameters, retain_graph=True, allow_unused=True,
                    )
                    event_gradient = torch.autograd.grad(
                        args.lambda_event * event_value, parameters, allow_unused=True,
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
                        primary_filled = [
                            torch.zeros_like(parameter) if gradient is None else gradient
                            for parameter, gradient in zip(parameters, primary_gradient)
                        ]
                        safe_gradient = [
                            total.detach() - clean.detach()
                            for total, clean in zip(combined, primary_filled)
                        ]
                        audit_action = True
                    else:
                        primary.backward()
                else:
                    primary.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                clip_retention = min(
                    1.0, args.grad_clip / max(float(norm), 1e-30),
                )
                before_parameters = []
                virtual_combined = []; virtual_primary = []; safe_clipped = []
                if audit_action:
                    combined_clipped = [
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in parameters
                    ]
                    primary_clipped = [
                        gradient.detach() * clip_retention
                        for gradient in primary_filled
                    ]
                    safe_clipped = [
                        gradient.detach() * clip_retention
                        for gradient in safe_gradient
                    ]
                    virtual_combined = virtual_adamw_descent_updates(
                        optimizer, parameters, combined_clipped,
                    )
                    virtual_primary = virtual_adamw_descent_updates(
                        optimizer, parameters, primary_clipped,
                    )
                    before_parameters = [
                        parameter.detach().clone() for parameter in parameters
                    ]
                optimizer.step()
                if audit_action:
                    actual_update = [
                        before - parameter.detach()
                        for before, parameter in zip(before_parameters, parameters)
                    ]
                    virtual_error = gradient_norm([
                        actual - virtual
                        for actual, virtual in zip(actual_update, virtual_combined)
                        if virtual is not None
                    ]) / max(gradient_norm(actual_update), 1e-30)
                    attributable = [
                        None if total is None or clean is None else total - clean
                        for total, clean in zip(virtual_combined, virtual_primary)
                    ]
                    update_norm = gradient_norm(virtual_combined)
                    attributable_norm = gradient_norm(attributable)
                    safe_norm = gradient_norm(safe_clipped)
                    full_alignment = (
                        gradient_dot(virtual_combined, safe_clipped)
                        / (update_norm * safe_norm)
                        if update_norm > 0 and safe_norm > 0 else 0.0
                    )
                    attributable_alignment = (
                        gradient_dot(attributable, safe_clipped)
                        / (attributable_norm * safe_norm)
                        if attributable_norm > 0 and safe_norm > 0 else 0.0
                    )
                    transmission_values["projection_retention"].append(
                        float(geometry["auxiliary_safe_norm"])
                        / max(float(geometry["auxiliary_raw_norm"]), 1e-30)
                    )
                    transmission_values["clip_event"].append(float(clip_retention < 1.0))
                    transmission_values["optimizer_update_action_fraction"].append(
                        attributable_norm / max(update_norm, 1e-30)
                    )
                    transmission_values["optimizer_update_action_alignment"].append(
                        full_alignment
                    )
                    transmission_values["optimizer_attributable_alignment"].append(
                        attributable_alignment
                    )
                    transmission_values["virtual_step_relative_error"].append(virtual_error)
                count = len(index); seen += count
                for key, value in (("primary", primary), ("rank", pieces["rank"]),
                                   ("safety", pieces["safety"]),
                                   ("preserve", pieces["preserve"]),
                                   ("event", event_value)):
                    totals[key] += float(value.detach()) * count
                totals["event_raw_cosine"] += float(geometry["auxiliary_raw_cosine"]) * count
                totals["event_safe_ratio"] += float(geometry["auxiliary_safe_to_primary_ratio"]) * count
                totals["event_projected"] += float(geometry["auxiliary_conflict_projected"]) * count
                totals["event_capped"] += float(geometry["auxiliary_norm_capped"]) * count
                totals["event_projection_retention"] += (
                    float(geometry["auxiliary_safe_norm"])
                    / max(float(geometry["auxiliary_raw_norm"]), 1e-30)
                ) * count
                totals["grad_norm"] += float(norm) * count
            record = {key: value / max(1, seen) for key, value in totals.items()}
            record["epoch"] = epoch; history.append(record)
            print(json.dumps({"arm": arm, **record}), flush=True)

        if len(active_event):
            if len(transmission_values["virtual_step_relative_error"]) < 16:
                raise RuntimeError(
                    f"{arm} produced fewer than 16 nonzero chemical optimizer audits"
                )
            transmission = {
                "audited_steps": len(transmission_values["virtual_step_relative_error"]),
                "projection_retention_p10": float(np.quantile(
                    transmission_values["projection_retention"], 0.10,
                )),
                "clip_event_fraction": float(np.mean(transmission_values["clip_event"])),
                "optimizer_action_attributable_fraction_p10": float(np.quantile(
                    transmission_values["optimizer_update_action_fraction"], 0.10,
                )),
                "optimizer_full_update_action_alignment_p10": float(np.quantile(
                    transmission_values["optimizer_update_action_alignment"], 0.10,
                )),
                "optimizer_attributable_action_alignment_p10": float(np.quantile(
                    transmission_values["optimizer_attributable_alignment"], 0.10,
                )),
                "virtual_adamw_max_relative_error": float(np.max(
                    transmission_values["virtual_step_relative_error"],
                )),
            }
            transmission_gates = {
                "audited_steps_at_least_16": transmission["audited_steps"] >= 16,
                "clip_event_fraction_at_most_0_10": (
                    transmission["clip_event_fraction"] <= 0.10
                ),
                "optimizer_action_fraction_p10_at_least_0_10": (
                    transmission["optimizer_action_attributable_fraction_p10"] >= 0.10
                ),
                "optimizer_attributable_alignment_p10_at_least_0_05": (
                    transmission["optimizer_attributable_action_alignment_p10"] >= 0.05
                ),
                "virtual_adamw_error_at_most_1e_3": (
                    transmission["virtual_adamw_max_relative_error"] <= 1e-3
                ),
            }
        else:
            transmission = {"audited_steps": 0, "not_applicable_clean_only": True}
            transmission_gates = {"clean_only_no_action_transmission_gate": True}

        encoded = encode_all(model, store, graph_rows, device, args, f"{arm}-final")
        new_diagnostics = retrieval_diagnostics(encoded, retrieval_graph, identity)
        new_rank = new_diagnostics["rank"]
        metric = retrieval(baseline_rank[evaluation], new_rank[evaluation])
        ci = bootstrap(
            formula[evaluation], baseline_rank[evaluation], new_rank[evaluation],
            args.bootstrap_draws, args.seed + 901 + arm_index,
        )
        preservation = np.sum(encoded * base, axis=1)
        artifact_gates = {
            "finite_normalized_embeddings": bool(
                np.all(np.isfinite(encoded))
                and np.max(np.abs(np.linalg.norm(encoded, axis=1) - 1.0)) <= 2e-3
            ),
            "mean_preservation": float(np.mean(preservation)) >= 0.995,
            "minimum_preservation": float(np.min(preservation)) >= 0.95,
            "formula_role2_optimizer_untouched": True,
            "outer_fold4_untouched": True,
            **transmission_gates,
        }
        histories[arm] = history
        held_baseline_diagnostics = {
            key: value[evaluation] for key, value in baseline_diagnostics.items()
        }
        held_new_diagnostics = {
            key: value[evaluation] for key, value in new_diagnostics.items()
        }
        held_results[arm] = {
            "retrieval": metric,
            "formula_cluster_bootstrap_ci95": ci,
            "auc": auc_metrics(
                held_new_diagnostics, formula[evaluation], held_baseline_diagnostics,
            ),
            "auc_formula_cluster_bootstrap": bootstrap_auc_delta(
                formula[evaluation], held_baseline_diagnostics, held_new_diagnostics,
                args.bootstrap_draws, args.seed + 1001 + arm_index,
            ),
        }
        held_rank_by_arm[arm] = new_rank[evaluation].astype(np.int64)
        held_diagnostics_by_arm[arm] = held_new_diagnostics
        preservation_by_arm[arm] = {
            "mean": float(np.mean(preservation)),
            "minimum": float(np.min(preservation)),
        }
        artifact_gates_by_arm[arm] = artifact_gates
        transmission_by_arm[arm] = transmission
        arm_states[arm] = peft_state_dict(model)
        arm_dir = partial / arm
        arm_dir.mkdir()
        (arm_dir / "result.json").write_text(json.dumps({
            "arm": arm, "preflight": preflight_by_arm[arm],
            "history": history, "held_evaluation": held_results[arm],
            "preservation": preservation_by_arm[arm],
            "artifact_gates": artifact_gates,
            "signal_transmission": transmission,
        }, indent=2), encoding="utf-8")
        np.savez_compressed(
            arm_dir / "held_predictions.npz",
            query=panel[evaluation], formula=formula[evaluation],
            identity=identity[evaluation], old_rank=baseline_rank[evaluation],
            new_rank=held_rank_by_arm[arm],
            old_query_auc=baseline_diagnostics["query_auc"][evaluation],
            new_query_auc=held_diagnostics_by_arm[arm]["query_auc"],
        )
        torch.save({
            "format": "chemaware_crossview_event_peft_arm_v1",
            "arm": arm, "peft_state": arm_states[arm], "capacity": capacity,
            "single_clean_spectrum_inference": True,
            "candidate_inputs_at_inference": False,
        }, arm_dir / "peft.pt")
        (partial / "run_state.json").write_text(json.dumps({
            "status": "CHEMAWARE_CROSSVIEW_EVENT_PEFT_PARTIAL",
            "completed_arms": list(held_results),
            "requested_arms": list(requested_arms),
            "recoverable_after_node_or_time_failure": True,
        }, indent=2), encoding="utf-8")
        del optimizer, encoded
        torch.cuda.empty_cache()
        print(json.dumps({
            "arm_complete": arm, "held_evaluation": metric,
            "held_formula_bootstrap_ci95": ci,
            "preservation": preservation_by_arm[arm],
        }), flush=True)

        if args.arm == "all" and arm == "correct":
            early_comparisons = {
                control: paired_rank_comparison(
                    held_rank_by_arm["correct"], held_rank_by_arm[control],
                    formula[evaluation], draws=args.bootstrap_draws,
                    seed=args.seed + 1101 + index,
                )
                for index, control in enumerate(("zero_contrast", "matched_random"))
            }
            early_gates = {
                "correct_absolute_formula_ci_positive": (
                    held_results["correct"]["formula_cluster_bootstrap_ci95"][0] > 0
                ),
                "corrected_exceeds_twice_introduced": (
                    held_results["correct"]["retrieval"]["corrected_at_1"]
                    > 2 * held_results["correct"]["retrieval"]["introduced_at_1"]
                ),
                "correct_artifact_and_signal_gates": all(
                    artifact_gates_by_arm["correct"].values()
                ),
                "matched_random_artifact_and_signal_gates": all(
                    artifact_gates_by_arm["matched_random"].values()
                ),
                **{
                    f"correct_beats_{control}_formula_ci": early_comparisons[control][
                        "formula_cluster_bootstrap_delta_recall1_ci95"
                    ][0] > 0
                    for control in ("zero_contrast", "matched_random")
                },
            }
            early_stop = {
                "triggered": not all(early_gates.values()),
                "stage": "correct_vs_clean_and_difficulty_matched_random",
                "gates": early_gates,
                "comparisons": early_comparisons,
                "saved_control_gpu_arms_if_failed": [
                    "reversed_contrast", "alignment_permuted",
                ],
            }
            (partial / "early_gate.json").write_text(
                json.dumps(early_stop, indent=2), encoding="utf-8",
            )
            if early_stop["triggered"]:
                print(json.dumps({
                    "status": "CHEMAWARE_CROSSVIEW_EVENT_PEFT_EARLY_STOP",
                    **early_stop,
                }, indent=2), flush=True)
                break

    paired = {}; paired_auc = {}
    scientific_gates = {"all_five_arms_present": set(held_rank_by_arm) == set(ARMS)}
    if set(held_rank_by_arm) == set(ARMS):
        paired = {
            f"correct_minus_{arm}": paired_rank_comparison(
                held_rank_by_arm["correct"], held_rank_by_arm[arm],
                formula[evaluation], draws=args.bootstrap_draws,
                seed=args.seed + 1201 + index,
            )
            for index, arm in enumerate(CONTROL_ARMS)
        }
        paired_auc = {
            f"correct_minus_{arm}": {
                "auc": auc_metrics(
                    held_diagnostics_by_arm["correct"], formula[evaluation],
                    held_diagnostics_by_arm[arm],
                ),
                "formula_cluster_bootstrap": bootstrap_auc_delta(
                    formula[evaluation], held_diagnostics_by_arm[arm],
                    held_diagnostics_by_arm["correct"], args.bootstrap_draws,
                    args.seed + 1401 + index,
                ),
            }
            for index, arm in enumerate(CONTROL_ARMS)
        }
        primary = held_results["correct"]
        scientific_gates.update({
            "correct_absolute_formula_ci_positive": (
                primary["formula_cluster_bootstrap_ci95"][0] > 0
            ),
            "corrected_exceeds_twice_introduced": (
                primary["retrieval"]["corrected_at_1"]
                > 2 * primary["retrieval"]["introduced_at_1"]
            ),
            "correct_mrr_positive": primary["retrieval"]["delta_mrr"] > 0,
            "correct_micro_auc_positive": primary["auc"]["delta_micro_auc"] > 0,
            "correct_macro_formula_auc_positive": (
                primary["auc"]["delta_macro_formula_auc"] > 0
            ),
            "correct_macro_formula_auc_ci_positive": (
                primary["auc_formula_cluster_bootstrap"][
                    "delta_macro_formula_auc_ci95"
                ][0] > 0
            ),
            "correct_higher_recall_nondecreasing": all(
                primary["retrieval"][f"delta_recall{k}"] >= 0
                for k in (3, 5, 10, 20, 50)
            ),
            **{
                f"correct_beats_{arm}_formula_ci": paired[f"correct_minus_{arm}"][
                    "formula_cluster_bootstrap_delta_recall1_ci95"
                ][0] > 0
                for arm in CONTROL_ARMS
            },
        })
    all_artifact_gates = all(
        all(gates.values()) for gates in artifact_gates_by_arm.values()
    )
    report = {
        "status": (
            "CHEMAWARE_CROSSVIEW_EVENT_PEFT_PASS"
            if all_artifact_gates and all(scientific_gates.values())
            else "CHEMAWARE_CROSSVIEW_EVENT_PEFT_FAIL"
        ),
        "formal": False,
        "release_eligible": False,
        "causal_chemical_claim": bool(
            all_artifact_gates and all(scientific_gates.values())
        ),
        "requested_arm": args.arm,
        "arms": list(requested_arms),
        "initialization": initialization,
        "capacity": capacity,
        "preflight": preflight,
        "optimization": {
            "objective": "clean listwise plus event-calibrated candidate-centred residual",
            "event_target": "epsilon minus official true-vs-top1 margin",
            "gradient_guard": "project conflicting event gradient and cap at fixed primary ratio",
            "epochs": args.epochs, "lr": args.lr,
            "maximum_event_gradient_ratio": args.maximum_event_gradient_ratio,
            "utility_magnitude_used_as_loss_weight": False,
            "event_roles_trained": ["correction", "protection"],
            "candidate_reference_score": (
                "dynamic count-normalized low-temperature logmeanexp over frozen-official top views"
            ),
            "references_per_candidate": args.references_per_candidate,
            "reference_temperature": args.reference_temperature,
            "arm_pairing": "same clean schedule, event RNG, PEFT initialization, and prefix cache",
        },
        "history": histories,
        "held_evaluation": held_results,
        "paired_chemical_controls": paired,
        "paired_auc_controls": paired_auc,
        "preservation": preservation_by_arm,
        "artifact_gates": artifact_gates_by_arm,
        "signal_transmission": transmission_by_arm,
        "scientific_gates": scientific_gates,
        "early_stop": early_stop,
        "contracts": {
            "single_clean_spectrum_inference": True,
            "candidate_inputs_at_inference": False,
            "query_reference_encoder_shared": True,
            "teacher_discarded_at_inference": True,
            "nonselected_chemical_gradient_zero": True,
            "fold4_evaluated": False,
            "one_prefix_cache_for_all_arms": True,
            "identical_arm_initialization": True,
        },
        "runtime_seconds": time.time() - started,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "repeat_ledger_sha256": sha256_file(args.repeat_ledger),
            "consensus_report_sha256": sha256_file(args.consensus_report),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
        },
    }
    (partial / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    prediction_body = {
        "query": panel[evaluation], "formula": formula[evaluation],
        "identity": identity[evaluation], "old_rank": baseline_rank[evaluation],
        "old_query_auc": baseline_diagnostics["query_auc"][evaluation],
        "old_top1_margin": baseline_diagnostics["top1_margin"][evaluation],
    }
    prediction_body.update({f"{arm}_rank": value for arm, value in held_rank_by_arm.items()})
    prediction_body.update({
        f"{arm}_query_auc": value["query_auc"]
        for arm, value in held_diagnostics_by_arm.items()
    })
    prediction_body.update({
        f"{arm}_top1_margin": value["top1_margin"]
        for arm, value in held_diagnostics_by_arm.items()
    })
    np.savez_compressed(partial / "held_predictions.npz", **prediction_body)
    torch.save({
        "format": "chemaware_crossview_event_peft_v1",
        "peft_state_by_arm": arm_states, "capacity": capacity,
        "arms": list(requested_arms), "single_clean_spectrum_inference": True,
        "candidate_inputs_at_inference": False, "release_eligible": False,
        "provenance": report["provenance"],
    }, partial / "peft_checkpoints.pt")
    partial.replace(args.output)
    print(json.dumps({
        "status": report["status"], "arms": list(requested_arms),
        "held_evaluation": held_results,
        "paired_chemical_controls": paired,
        "paired_auc_controls": paired_auc,
        "preservation": report["preservation"],
        "scientific_gates": scientific_gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

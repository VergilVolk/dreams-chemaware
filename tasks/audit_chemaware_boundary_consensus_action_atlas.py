"""Discover same-boundary, multi-negative ChemAware peak actions.

This is a frozen-model A1 action audit, not an embedding trainer.  It targets
the exact official-DreaMS retrieval boundary among same-formula candidates,
requires peak evidence to agree across several high-scoring negatives, and
abstains when that evidence is weak.  Candidate-role-reversed and
intensity-rank-permuted arms share the target query scope, action counts, and
multiplicative doses.  Formula folds 3 and 4 remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_boundary_consensus_action_core import (
    EvidenceProfile,
    PairEvidenceProfile,
    PeakActionPlan,
    apply_action_plan,
    build_pair_logratio_action_plan,
    build_peak_action_plan,
    capacity_matched_pair_logratio_plan,
    capacity_matched_plan,
    consensus_evidence,
    intensity_rank_permuted_pair_profile,
    intensity_rank_permuted_profile,
    invert_action_plan,
    pairwise_consensus_evidence,
    top_official_negative_positions,
)
from chemaware_direct_action_core import formula_bootstrap
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_retrieval_graph import RetrievalGraph
from noise_final_core import sha256_file, strict_rank
from train_e1_identity import load_base_model
from train_noise_final_r2_shared_encoder import SpectrumStore

ARMS = (
    "correct",
    "candidate_role_reversed",
    "intensity_rank_permuted",
    "direction_reversed",
)
RECALL_K = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class ActionSetting:
    mode: str
    strength: float
    top_k: int
    minimum_abs_evidence: float


POINT_SETTINGS = tuple(
    ActionSetting(mode, strength, top_k, threshold)
    for threshold in (0.05, 0.10)
    for mode, strengths, top_values in (
        ("conflict_attenuate", (0.50, 0.75), (1, 3)),
        ("support_boost", (0.25, 0.50), (1, 3)),
        ("bidirectional_sharpen", (0.25, 0.50), (1, 2)),
    )
    for strength in strengths
    for top_k in top_values
)
PAIR_SETTINGS = tuple(
    ActionSetting("pair_logratio_sharpen", dose, top_pairs, threshold)
    for threshold in (0.25, 0.50)
    for dose in (0.25, 0.50)
    for top_pairs in (1, 2)
)
SETTINGS = POINT_SETTINGS + PAIR_SETTINGS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT
        / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz",
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1",
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint",
        type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_boundary_consensus_action_atlas_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--discovery-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--embedding-evaluation-fold", type=int, default=3)
    parser.add_argument("--reserve-fold", type=int, default=4)
    parser.add_argument("--top-negative-count", type=int, default=5)
    parser.add_argument("--minimum-negative-count", type=int, default=2)
    parser.add_argument("--agreement-quantile", type=float, default=0.75)
    parser.add_argument("--minimum-prediction", type=float, default=0.10)
    parser.add_argument("--minimum-agreement", type=float, default=0.75)
    parser.add_argument("--minimum-observed-intensity", type=float, default=0.01)
    parser.add_argument("--precursor-exclusion-da", type=float, default=1.1)
    parser.add_argument("--minimum-action-queries", type=int, default=20)
    parser.add_argument("--minimum-action-formulas", type=int, default=10)
    parser.add_argument("--minimum-action-fraction", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _empty_profile(width: int) -> EvidenceProfile:
    zeros = np.zeros(width, dtype=np.float32)
    return EvidenceProfile(zeros.copy(), zeros.copy(), zeros.copy())


def _empty_pair_profile(width: int) -> PairEvidenceProfile:
    zeros = np.zeros((width, width), dtype=np.float32)
    return PairEvidenceProfile(zeros.copy(), zeros.copy(), zeros.copy())


def _empty_plan() -> PeakActionPlan:
    return PeakActionPlan(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int8),
    )


def molecule_rank_margin(
    graph: RetrievalGraph,
    query: int,
    query_embedding: np.ndarray,
    reference: np.ndarray,
    row_position: dict[int, int],
) -> tuple[int, float, int]:
    _, rows, ptr, _ = graph.query_block(query)
    pair = reference[[row_position[int(row)] for row in rows]] @ query_embedding
    molecule = np.maximum.reduceat(pair, ptr[:-1])
    return (
        strict_rank(molecule),
        float(molecule[0] - np.max(molecule[1:])),
        len(molecule),
    )


def ranking_summary(
    old_rank: np.ndarray,
    old_margin: np.ndarray,
    new_rank: np.ndarray,
    new_margin: np.ndarray,
    candidate_count: np.ndarray,
) -> dict:
    old_rank = np.asarray(old_rank, dtype=np.int64)
    new_rank = np.asarray(new_rank, dtype=np.int64)
    count = np.asarray(candidate_count, dtype=np.int64)
    old_auc = (count - old_rank) / np.maximum(count - 1, 1)
    new_auc = (count - new_rank) / np.maximum(count - 1, 1)
    output = {
        "queries": len(old_rank),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank - 1.0 / old_rank)),
        "macro_auc": float(np.mean(new_auc)),
        "delta_macro_auc": float(np.mean(new_auc - old_auc)),
        "micro_auc": float(np.sum(count - new_rank) / np.sum(count - 1)),
        "delta_micro_auc": float(
            (np.sum(count - new_rank) - np.sum(count - old_rank)) / np.sum(count - 1)
        ),
        "delta_mean_margin": float(np.mean(new_margin - old_margin)),
    }
    for k in RECALL_K:
        old_hit, new_hit = old_rank <= k, new_rank <= k
        output[f"recall_at_{k}"] = float(np.mean(new_hit))
        output[f"delta_recall_at_{k}"] = float(np.mean(new_hit) - np.mean(old_hit))
        output[f"corrected_at_{k}"] = int(np.sum(~old_hit & new_hit))
        output[f"introduced_at_{k}"] = int(np.sum(old_hit & ~new_hit))
    return output


def encode(model, spectra, device, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    output = []
    with torch.no_grad():
        for left in range(0, len(spectra), batch_size):
            output.append(
                model(spectra[left : left + batch_size].to(device))
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(output)


def _capacity_signature(plan: PeakActionPlan) -> tuple[int, int, tuple[float, ...]]:
    return plan.attenuated, plan.boosted, tuple(np.sort(plan.factors).round(7))


def main() -> None:
    import torch

    args = arguments()
    started = time.time()
    if args.output.exists() and not args.preflight_only:
        raise FileExistsError(args.output)
    roles = (
        *args.discovery_folds,
        args.confirmation_fold,
        args.embedding_evaluation_fold,
        args.reserve_fold,
    )
    if len(set(roles)) != len(roles) or min(roles) < 0 or max(roles) >= args.folds:
        raise ValueError("all formula-fold roles must be distinct and in range")
    if args.bootstrap_draws < 10_000:
        raise ValueError("formula-cluster bootstrap requires at least 10,000 draws")
    if (
        args.minimum_negative_count < 2
        or args.top_negative_count < args.minimum_negative_count
    ):
        raise ValueError("multi-negative consensus requires at least two negatives")
    if not 0.5 <= args.agreement_quantile < 1.0:
        raise ValueError("invalid agreement quantile")

    required = [
        args.graph,
        args.teacher_dir / "report.json",
        args.teacher_dir / "selected_queries.npy",
        args.teacher_dir / "query_ptr.npy",
        args.teacher_dir / "candidate_molecule_index.npy",
        args.teacher_dir / "iceberg_predictions_f16.npy",
        args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.data,
        args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"required ChemAware assets are absent: {missing}")
    teacher_report = json.loads(
        (args.teacher_dir / "report.json").read_text(encoding="utf-8")
    )
    if teacher_report.get("status") != "PASS":
        raise RuntimeError("requires a passed ICEBERG teacher ledger")
    if teacher_report.get("inputs", {}).get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("teacher and retrieval graph provenance differ")

    graph = RetrievalGraph(args.graph)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    teacher_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    teacher_molecule = np.load(
        args.teacher_dir / "candidate_molecule_index.npy"
    ).astype(np.int64)
    prediction = np.load(
        args.teacher_dir / "iceberg_predictions_f16.npy", mmap_mode="r"
    )
    if (
        teacher_ptr.shape != (len(selected) + 1,)
        or int(teacher_ptr[0]) != 0
        or int(teacher_ptr[-1]) != len(prediction)
        or len(teacher_molecule) != len(prediction)
    ):
        raise RuntimeError("ICEBERG teacher arrays are not aligned")
    for position, query in enumerate(selected):
        left, right = map(int, teacher_ptr[position : position + 2])
        qleft, qright = map(int, graph.query_ptr[int(query) : int(query) + 2])
        if not np.array_equal(teacher_molecule[left:right], np.arange(qleft, qright)):
            raise RuntimeError("teacher candidate order differs from retrieval graph")

    formula = graph.query_formula[selected].astype(str)
    fold = stable_formula_folds(formula, args.folds, args.fold_seed)
    discovery = np.flatnonzero(np.isin(fold, args.discovery_folds))
    confirmation = np.flatnonzero(fold == args.confirmation_fold)
    embedding_evaluation = np.flatnonzero(fold == args.embedding_evaluation_fold)
    reserve = np.flatnonzero(fold == args.reserve_fold)
    if min(map(len, (discovery, confirmation, embedding_evaluation, reserve))) == 0:
        raise RuntimeError("one or more formula-fold roles are empty")

    candidate_negative: list[np.ndarray] = []
    formula_observable = np.zeros(len(selected), dtype=bool)
    for position, query in enumerate(selected):
        scores = graph.official_molecule_scores(int(query))
        qleft, qright = map(int, graph.query_ptr[int(query) : int(query) + 2])
        negatives = top_official_negative_positions(
            scores,
            graph.molecule_formula[qleft:qright],
            graph.query_formula[int(query)],
            args.top_negative_count,
            args.minimum_negative_count,
        )
        candidate_negative.append(negatives)
        formula_observable[position] = len(negatives) >= args.minimum_negative_count

    preflight = {
        "status": "CHEMAWARE_BOUNDARY_CONSENSUS_ACTION_PREFLIGHT_PASS",
        "weights_updated": False,
        "action_settings": len(SETTINGS),
        "arms": list(ARMS),
        "teacher_queries": len(selected),
        "same_formula_multi_negative_queries": int(np.sum(formula_observable)),
        "same_formula_multi_negative_fraction": float(np.mean(formula_observable)),
        "fold_roles": {
            "discovery_queries": len(discovery),
            "confirmation_queries": len(confirmation),
            "embedding_evaluation_queries_untouched": len(embedding_evaluation),
            "reserve_queries_untouched": len(reserve),
        },
        "planned_discovery_encodes": int(len(SETTINGS) * len(ARMS) * len(discovery)),
        "graph_sha256": sha256_file(args.graph),
        "teacher_predictions_sha256": sha256_file(
            args.teacher_dir / "iceberg_predictions_f16.npy"
        ),
    }
    torch.set_num_threads(args.torch_threads)
    store = SpectrumStore(args.data, graph.query_row[selected], args.n_highest_peaks)
    clean = torch.stack([store.one(int(graph.query_row[query])) for query in selected])

    target_profile: list[EvidenceProfile] = []
    reversed_profile: list[EvidenceProfile] = []
    peak_permuted_profile: list[EvidenceProfile] = []
    target_pair_profile: list[PairEvidenceProfile] = []
    reversed_pair_profile: list[PairEvidenceProfile] = []
    peak_permuted_pair_profile: list[PairEvidenceProfile] = []
    for position in range(len(selected)):
        width = len(clean[position]) - 1
        negatives = candidate_negative[position]
        if len(negatives) < args.minimum_negative_count:
            target_profile.append(_empty_profile(width))
            reversed_profile.append(_empty_profile(width))
            peak_permuted_profile.append(_empty_profile(width))
            target_pair_profile.append(_empty_pair_profile(width))
            reversed_pair_profile.append(_empty_pair_profile(width))
            peak_permuted_pair_profile.append(_empty_pair_profile(width))
            continue
        left, right = map(int, teacher_ptr[position : position + 2])
        block = np.asarray(prediction[left:right], dtype=np.float32)
        observed_mz = clean[position, 1:, 0].numpy()
        target = consensus_evidence(
            block,
            0,
            int(negatives[0]),
            negatives,
            observed_mz,
            args.agreement_quantile,
            args.minimum_prediction,
        )
        swapped_negatives = np.r_[0, negatives[1:]].astype(np.int64)
        reversed_value = consensus_evidence(
            block,
            int(negatives[0]),
            0,
            swapped_negatives,
            observed_mz,
            args.agreement_quantile,
            args.minimum_prediction,
        )
        target_pair = pairwise_consensus_evidence(
            block,
            0,
            int(negatives[0]),
            negatives,
            observed_mz,
            args.agreement_quantile,
            args.minimum_prediction,
        )
        reversed_pair = pairwise_consensus_evidence(
            block,
            int(negatives[0]),
            0,
            swapped_negatives,
            observed_mz,
            args.agreement_quantile,
            args.minimum_prediction,
        )
        target_profile.append(target)
        reversed_profile.append(reversed_value)
        peak_permuted_profile.append(
            intensity_rank_permuted_profile(
                target,
                clean[position, 1:, 1].numpy(),
                args.seed + position,
            )
        )
        target_pair_profile.append(target_pair)
        reversed_pair_profile.append(reversed_pair)
        peak_permuted_pair_profile.append(
            intensity_rank_permuted_pair_profile(
                target_pair,
                clean[position, 1:, 1].numpy(),
                args.seed + position,
            )
        )

    def target_plan(position: int, setting: ActionSetting) -> PeakActionPlan:
        if not formula_observable[position]:
            return _empty_plan()
        if setting.mode == "pair_logratio_sharpen":
            return build_pair_logratio_action_plan(
                target_pair_profile[position],
                clean[position, 1:, 0].numpy(),
                clean[position, 1:, 1].numpy(),
                float(clean[position, 0, 0]),
                setting.strength,
                setting.top_k,
                setting.minimum_abs_evidence,
                args.minimum_agreement,
                args.minimum_observed_intensity,
                args.precursor_exclusion_da,
            )
        return build_peak_action_plan(
            target_profile[position],
            clean[position, 1:, 0].numpy(),
            clean[position, 1:, 1].numpy(),
            float(clean[position, 0, 0]),
            setting.mode,
            setting.strength,
            setting.top_k,
            setting.minimum_abs_evidence,
            args.minimum_agreement,
            args.minimum_observed_intensity,
            args.precursor_exclusion_da,
        )

    def plans_for_position(
        position: int,
        setting: ActionSetting,
    ) -> tuple[PeakActionPlan, PeakActionPlan, PeakActionPlan, PeakActionPlan]:
        target = target_plan(position, setting)
        if target.abstained:
            return target, _empty_plan(), _empty_plan(), _empty_plan()
        common = (
            clean[position, 1:, 0].numpy(),
            clean[position, 1:, 1].numpy(),
            float(clean[position, 0, 0]),
        )
        if setting.mode == "pair_logratio_sharpen":
            reversed_value = capacity_matched_pair_logratio_plan(
                reversed_pair_profile[position],
                *common,
                target,
                args.minimum_observed_intensity,
                args.precursor_exclusion_da,
            )
            permuted_value = capacity_matched_pair_logratio_plan(
                peak_permuted_pair_profile[position],
                *common,
                target,
                args.minimum_observed_intensity,
                args.precursor_exclusion_da,
            )
        else:
            reversed_value = capacity_matched_plan(
                reversed_profile[position],
                *common,
                target,
                args.minimum_observed_intensity,
                args.precursor_exclusion_da,
            )
            permuted_value = capacity_matched_plan(
                peak_permuted_profile[position],
                *common,
                target,
                args.minimum_observed_intensity,
                args.precursor_exclusion_da,
            )
        direction_reversed = invert_action_plan(target)
        signature = _capacity_signature(target)
        if (
            reversed_value.abstained
            or permuted_value.abstained
            or _capacity_signature(reversed_value) != signature
            or _capacity_signature(permuted_value) != signature
        ):
            raise RuntimeError("matched control lost target action capacity")
        if (
            not np.array_equal(direction_reversed.positions, target.positions)
            or not np.array_equal(direction_reversed.roles, -target.roles)
            or not np.allclose(direction_reversed.factors, 1.0 / target.factors)
        ):
            raise RuntimeError("direction-reversed control lost exact action support")
        return target, reversed_value, permuted_value, direction_reversed

    if args.preflight_only:
        audited_positions = np.concatenate((discovery, confirmation))
        active_queries = np.zeros(len(SETTINGS), dtype=np.int32)
        active_formulas = np.zeros(len(SETTINGS), dtype=np.int32)
        modified_peaks = np.zeros(len(SETTINGS), dtype=np.int32)
        for setting_id, setting in enumerate(SETTINGS):
            active_formula_values = []
            for position in audited_positions:
                target, *_ = plans_for_position(int(position), setting)
                if not target.abstained:
                    active_queries[setting_id] += 1
                    modified_peaks[setting_id] += len(target.positions)
                    active_formula_values.append(formula[position])
            active_formulas[setting_id] = len(np.unique(active_formula_values))
        if np.any(active_queries == 0):
            empty = np.flatnonzero(active_queries == 0).tolist()
            raise RuntimeError(f"action settings without any eligible query: {empty}")
        preflight["deep_action_contracts"] = {
            "positions_audited": int(len(audited_positions)),
            "plans_constructed": int(len(SETTINGS) * len(audited_positions)),
            "matched_control_capacity_validated": True,
            "settings_with_at_least_one_action": int(np.sum(active_queries > 0)),
            "active_query_range": [
                int(np.min(active_queries)),
                int(np.max(active_queries)),
            ],
            "active_formula_range": [
                int(np.min(active_formulas)),
                int(np.max(active_formulas)),
            ],
            "modified_peak_range": [
                int(np.min(modified_peaks)),
                int(np.max(modified_peaks)),
            ],
        }
        print(json.dumps(preflight, indent=2))
        return

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    token_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(token_rows)}
    required_rows = set(map(int, graph.query_row[selected]))
    if not required_rows.issubset(row_position):
        raise RuntimeError(
            "teacher queries are absent from the official embedding cache"
        )

    old_rank = np.empty(len(selected), dtype=np.int16)
    old_margin = np.empty(len(selected), dtype=np.float32)
    candidate_count = np.empty(len(selected), dtype=np.int32)
    for position, query in enumerate(selected):
        embedding = official[row_position[int(graph.query_row[query])]]
        old_rank[position], old_margin[position], candidate_count[position] = (
            molecule_rank_margin(
                graph,
                int(query),
                embedding,
                official,
                row_position,
            )
        )

    action_tensors = []
    action_count = np.zeros((len(SETTINGS), len(selected)), dtype=np.int16)
    action_formula_count = np.zeros(len(SETTINGS), dtype=np.int32)
    for setting_id, setting in enumerate(SETTINGS):
        per_arm = [[] for _ in ARMS]
        active_formula = []
        for position in discovery:
            plans = plans_for_position(int(position), setting)
            action_count[setting_id, position] = len(plans[0].positions)
            if not plans[0].abstained:
                active_formula.append(formula[position])
            for arm_id, plan in enumerate(plans):
                per_arm[arm_id].append(apply_action_plan(clean[position], plan))
        action_formula_count[setting_id] = len(np.unique(active_formula))
        for arm_block in per_arm:
            action_tensors.extend(arm_block)

    model, _ = load_base_model(
        args.official_checkpoint,
        args.architecture_checkpoint,
        device,
        args.n_highest_peaks,
    )
    encoded = encode(model, torch.stack(action_tensors), device, args.batch_size)
    ranks = np.broadcast_to(old_rank, (len(SETTINGS), len(ARMS), len(selected))).copy()
    margins = np.broadcast_to(
        old_margin, (len(SETTINGS), len(ARMS), len(selected))
    ).copy()
    block_size = len(discovery)
    block_id = 0
    for setting_id in range(len(SETTINGS)):
        for arm_id in range(len(ARMS)):
            block = encoded[block_id * block_size : (block_id + 1) * block_size]
            block_id += 1
            for local, position in enumerate(discovery):
                query = selected[position]
                (
                    ranks[setting_id, arm_id, position],
                    margins[setting_id, arm_id, position],
                    _,
                ) = molecule_rank_margin(
                    graph,
                    int(query),
                    block[local],
                    official,
                    row_position,
                )

    setting_reports = []
    qualified = []
    for setting_id, setting in enumerate(SETTINGS):
        comparisons = {
            "absolute_margin": formula_bootstrap(
                (margins[setting_id, 0] - old_margin)[discovery],
                formula[discovery],
                args.seed + setting_id,
                args.bootstrap_draws,
            ),
            "minus_candidate_role_reversed_margin": formula_bootstrap(
                (margins[setting_id, 0] - margins[setting_id, 1])[discovery],
                formula[discovery],
                args.seed + 100 + setting_id,
                args.bootstrap_draws,
            ),
            "minus_intensity_rank_permuted_margin": formula_bootstrap(
                (margins[setting_id, 0] - margins[setting_id, 2])[discovery],
                formula[discovery],
                args.seed + 200 + setting_id,
                args.bootstrap_draws,
            ),
            "minus_direction_reversed_margin": formula_bootstrap(
                (margins[setting_id, 0] - margins[setting_id, 3])[discovery],
                formula[discovery],
                args.seed + 300 + setting_id,
                args.bootstrap_draws,
            ),
        }
        active = action_count[setting_id, discovery] > 0
        metrics = ranking_summary(
            old_rank[discovery],
            old_margin[discovery],
            ranks[setting_id, 0, discovery],
            margins[setting_id, 0, discovery],
            candidate_count[discovery],
        )
        gates = {
            "minimum_action_queries": int(np.sum(active))
            >= args.minimum_action_queries,
            "minimum_action_formulas": int(action_formula_count[setting_id])
            >= args.minimum_action_formulas,
            "minimum_action_fraction": float(np.mean(active))
            >= args.minimum_action_fraction,
            "absolute_margin_ci_positive": comparisons["absolute_margin"][
                "formula_cluster_bootstrap_95ci"
            ][0]
            > 0,
            "candidate_control_ci_positive": comparisons[
                "minus_candidate_role_reversed_margin"
            ]["formula_cluster_bootstrap_95ci"][0]
            > 0,
            "peak_control_ci_positive": comparisons[
                "minus_intensity_rank_permuted_margin"
            ]["formula_cluster_bootstrap_95ci"][0]
            > 0,
            "direction_control_ci_positive": comparisons[
                "minus_direction_reversed_margin"
            ]["formula_cluster_bootstrap_95ci"][0]
            > 0,
            "recall1_nonnegative": metrics["delta_recall_at_1"] >= 0,
            "risk_nonnegative": metrics["corrected_at_1"] >= metrics["introduced_at_1"],
        }
        minimum_advantage = min(
            value["formula_macro_mean"] for value in comparisons.values()
        )
        item = {
            "setting_id": setting_id,
            **asdict(setting),
            "active_queries": int(np.sum(active)),
            "active_fraction": float(np.mean(active)),
            "active_formulas": int(action_formula_count[setting_id]),
            "mean_modified_peaks_when_active": float(
                np.mean(action_count[setting_id, discovery][active])
                if np.any(active)
                else 0.0
            ),
            "ranking": metrics,
            "comparisons": comparisons,
            "minimum_formula_macro_margin_advantage": float(minimum_advantage),
            "gates": gates,
            "qualified_for_confirmation": bool(all(gates.values())),
        }
        setting_reports.append(item)
        if item["qualified_for_confirmation"]:
            qualified.append(setting_id)

    if qualified:
        selected_setting_id = max(
            qualified,
            key=lambda index: (
                setting_reports[index]["minimum_formula_macro_margin_advantage"],
                setting_reports[index]["ranking"]["delta_recall_at_1"],
                -SETTINGS[index].strength,
                -SETTINGS[index].top_k,
            ),
        )
    else:
        selected_setting_id = max(
            range(len(SETTINGS)),
            key=lambda index: setting_reports[index][
                "minimum_formula_macro_margin_advantage"
            ],
        )

    confirmation_report = None
    pass_to_transfer = False
    if qualified:
        setting = SETTINGS[selected_setting_id]
        confirmation_tensors = [[] for _ in ARMS]
        active_formula = []
        for position in confirmation:
            plans = plans_for_position(int(position), setting)
            action_count[selected_setting_id, position] = len(plans[0].positions)
            if not plans[0].abstained:
                active_formula.append(formula[position])
            for arm_id, plan in enumerate(plans):
                confirmation_tensors[arm_id].append(
                    apply_action_plan(clean[position], plan)
                )
        confirmation_encoded = encode(
            model,
            torch.stack([tensor for block in confirmation_tensors for tensor in block]),
            device,
            args.batch_size,
        )
        for arm_id in range(len(ARMS)):
            block = confirmation_encoded[
                arm_id * len(confirmation) : (arm_id + 1) * len(confirmation)
            ]
            for local, position in enumerate(confirmation):
                query = selected[position]
                (
                    ranks[selected_setting_id, arm_id, position],
                    margins[selected_setting_id, arm_id, position],
                    _,
                ) = molecule_rank_margin(
                    graph,
                    int(query),
                    block[local],
                    official,
                    row_position,
                )
        comparisons = {
            "absolute_margin": formula_bootstrap(
                (margins[selected_setting_id, 0] - old_margin)[confirmation],
                formula[confirmation],
                args.seed + 501,
                args.bootstrap_draws,
            ),
            "minus_candidate_role_reversed_margin": formula_bootstrap(
                (margins[selected_setting_id, 0] - margins[selected_setting_id, 1])[
                    confirmation
                ],
                formula[confirmation],
                args.seed + 502,
                args.bootstrap_draws,
            ),
            "minus_intensity_rank_permuted_margin": formula_bootstrap(
                (margins[selected_setting_id, 0] - margins[selected_setting_id, 2])[
                    confirmation
                ],
                formula[confirmation],
                args.seed + 503,
                args.bootstrap_draws,
            ),
            "minus_direction_reversed_margin": formula_bootstrap(
                (margins[selected_setting_id, 0] - margins[selected_setting_id, 3])[
                    confirmation
                ],
                formula[confirmation],
                args.seed + 504,
                args.bootstrap_draws,
            ),
        }
        active = action_count[selected_setting_id, confirmation] > 0
        metrics = ranking_summary(
            old_rank[confirmation],
            old_margin[confirmation],
            ranks[selected_setting_id, 0, confirmation],
            margins[selected_setting_id, 0, confirmation],
            candidate_count[confirmation],
        )
        gates = {
            "minimum_action_queries": int(np.sum(active))
            >= args.minimum_action_queries,
            "minimum_action_formulas": len(np.unique(active_formula))
            >= args.minimum_action_formulas,
            "minimum_action_fraction": float(np.mean(active))
            >= args.minimum_action_fraction,
            "absolute_margin_ci_positive": comparisons["absolute_margin"][
                "formula_cluster_bootstrap_95ci"
            ][0]
            > 0,
            "candidate_control_ci_positive": comparisons[
                "minus_candidate_role_reversed_margin"
            ]["formula_cluster_bootstrap_95ci"][0]
            > 0,
            "peak_control_ci_positive": comparisons[
                "minus_intensity_rank_permuted_margin"
            ]["formula_cluster_bootstrap_95ci"][0]
            > 0,
            "direction_control_ci_positive": comparisons[
                "minus_direction_reversed_margin"
            ]["formula_cluster_bootstrap_95ci"][0]
            > 0,
            "recall1_nonnegative": metrics["delta_recall_at_1"] >= 0,
            "risk_nonnegative": metrics["corrected_at_1"] >= metrics["introduced_at_1"],
        }
        pass_to_transfer = bool(all(gates.values()))
        confirmation_report = {
            "active_queries": int(np.sum(active)),
            "active_fraction": float(np.mean(active)),
            "active_formulas": len(np.unique(active_formula)),
            "ranking": metrics,
            "comparisons": comparisons,
            "gates": gates,
            "passed": pass_to_transfer,
        }

    report = {
        "status": (
            "CHEMAWARE_BOUNDARY_CONSENSUS_ACTION_CONFIRMATION_PASS"
            if pass_to_transfer
            else "CHEMAWARE_BOUNDARY_CONSENSUS_ACTION_CONFIRMATION_FAIL"
            if qualified
            else "CHEMAWARE_BOUNDARY_CONSENSUS_ACTION_DISCOVERY_FAIL"
        ),
        "formal_training_authorized": False,
        "pass_to_action_transfer_audit": pass_to_transfer,
        "preflight": preflight,
        "scope": {
            "weights_updated": False,
            "candidate_information_training_or_audit_only": True,
            "same_formula_negatives_only": True,
            "official_dreams_boundary_frozen_across_arms": True,
            "multi_negative_consensus": True,
            "pair_logratio_scale_invariant_action": True,
            "weak_evidence_abstains": True,
            "observed_peak_mz_unchanged": True,
            "precursor_region_excluded": True,
            "base_peak_and_max_normalization_fixed": True,
            "direction_reversed_same_peak_control": True,
            "embedding_evaluation_fold_inspected": False,
            "reserve_fold_inspected": False,
        },
        "protocol": {
            "arms": list(ARMS),
            "settings": len(SETTINGS),
            "top_negative_count": args.top_negative_count,
            "minimum_negative_count": args.minimum_negative_count,
            "agreement_quantile": args.agreement_quantile,
            "minimum_prediction": args.minimum_prediction,
            "minimum_agreement": args.minimum_agreement,
            "minimum_observed_intensity": args.minimum_observed_intensity,
            "precursor_exclusion_da": args.precursor_exclusion_da,
            "control_capacity": "candidate/peak controls use the same query, candidate set, attenuated/boosted counts and factors; direction control uses the same peak slots and reciprocal factors",
            "selection": "discovery only; confirmation is evaluated once only if every discovery gate passes",
        },
        "folds": {
            "fold_seed": args.fold_seed,
            "discovery_folds": list(args.discovery_folds),
            "confirmation_fold": args.confirmation_fold,
            "embedding_evaluation_fold": args.embedding_evaluation_fold,
            "reserve_fold": args.reserve_fold,
        },
        "discovery": {
            "qualified_settings": qualified,
            "selected_setting_id": selected_setting_id,
            "selected_setting_was_qualified": selected_setting_id in qualified,
            "settings": setting_reports,
        },
        "confirmation": confirmation_report,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(
                args.teacher_dir / "iceberg_predictions_f16.npy"
            ),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    screen_path = args.output / "screen.npz"
    np.savez_compressed(
        screen_path,
        selected_query=selected,
        formula=formula,
        formula_fold=fold,
        formula_observable=formula_observable,
        setting_mode=np.asarray([value.mode for value in SETTINGS]),
        setting_strength=np.asarray(
            [value.strength for value in SETTINGS], dtype=np.float32
        ),
        setting_top_k=np.asarray([value.top_k for value in SETTINGS], dtype=np.int16),
        setting_minimum_abs_evidence=np.asarray(
            [value.minimum_abs_evidence for value in SETTINGS], dtype=np.float32
        ),
        old_rank=old_rank,
        old_margin=old_margin,
        candidate_count=candidate_count,
        ranks=ranks,
        margins=margins,
        action_count=action_count,
        selected_setting=np.asarray([selected_setting_id], dtype=np.int16),
        discovery_position=discovery,
        confirmation_position=confirmation,
        embedding_evaluation_position=embedding_evaluation,
        reserve_position=reserve,
    )
    report["provenance"]["screen_sha256"] = sha256_file(screen_path)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "formal_training_authorized": False,
                "pass_to_action_transfer_audit": pass_to_transfer,
                "selected_setting_id": selected_setting_id,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

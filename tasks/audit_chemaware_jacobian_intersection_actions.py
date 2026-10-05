"""Screen ChemAware actions at the chemistry-by-input-Jacobian intersection.

This frozen-model audit does not encode perturbed spectra and never updates a
weight.  It asks a cheaper prerequisite question: after the official DreaMS
same-formula boundary is fixed, do chemically directed edits retain a positive
first-order margin advantage over candidate-role and peak-identity controls
that are matched on edit capacity, intensity and absolute input-Jacobian
magnitude?  Folds 3 and 4 remain untouched.
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

from chemaware_boundary_consensus_action_core import (  # noqa: E402
    EvidenceProfile,
    PeakActionPlan,
    consensus_evidence,
    intensity_rank_permuted_profile,
    invert_action_plan,
    top_official_negative_positions,
)
from chemaware_direct_action_core import formula_bootstrap  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from chemaware_jacobian_intersection_action_core import (  # noqa: E402
    build_jacobian_intersection_plan,
    first_order_log_intensity_gain,
    jacobian_match_error,
    matched_jacobian_control_plan,
)
from chemaware_retrieval_graph import RetrievalGraph  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore  # noqa: E402


ARMS = (
    "correct_chemistry",
    "candidate_role_reversed",
    "intensity_rank_permuted",
    "direction_reversed_same_peaks",
)


@dataclass(frozen=True)
class JacobianSetting:
    mode: str
    log_dose: float
    top_k: int
    minimum_abs_evidence: float


SETTINGS = tuple(
    JacobianSetting(mode, 0.25, top_k, threshold)
    for threshold in (0.05, 0.10)
    for mode in ("conflict_attenuate", "support_boost", "bidirectional_sharpen")
    for top_k in (1, 2)
)


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
        default=ROOT / "data/validation/chemaware_jacobian_intersection_actions_v1",
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
    parser.add_argument(
        "--maximum-mean-log-jacobian-match-error", type=float, default=0.50
    )
    parser.add_argument(
        "--maximum-mean-log-intensity-match-error", type=float, default=0.50
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def empty_profile(width: int) -> EvidenceProfile:
    zeros = np.zeros(width, dtype=np.float32)
    return EvidenceProfile(zeros.copy(), zeros.copy(), zeros.copy())


def empty_plan() -> PeakActionPlan:
    return PeakActionPlan(
        positions=np.empty(0, dtype=np.int64),
        factors=np.empty(0, dtype=np.float32),
        evidence=np.empty(0, dtype=np.float32),
        roles=np.empty(0, dtype=np.int8),
    )


def capacity_signature(plan: PeakActionPlan) -> tuple[int, int, tuple[float, ...]]:
    return plan.attenuated, plan.boosted, tuple(np.sort(plan.factors).round(7))


def reference_deltas(
    graph: RetrievalGraph,
    selected: np.ndarray,
    positions: np.ndarray,
    candidate_negative: list[np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
) -> np.ndarray:
    output = []
    for position in positions:
        query = int(selected[position])
        pair_slice, rows, ptr, _ = graph.query_block(query)
        pair_scores = graph.features[pair_slice, graph.dreams_column]
        negative = int(candidate_negative[position][0])
        positive_pair = int(np.argmax(pair_scores[ptr[0] : ptr[1]])) + int(ptr[0])
        negative_pair = int(
            np.argmax(pair_scores[ptr[negative] : ptr[negative + 1]])
        ) + int(ptr[negative])
        positive_embedding = official[row_position[int(rows[positive_pair])]]
        negative_embedding = official[row_position[int(rows[negative_pair])]]
        output.append(positive_embedding - negative_embedding)
    return np.asarray(output, dtype=np.float32)


def input_jacobians(model, spectra, deltas: np.ndarray, device, batch_size: int):
    import torch

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = []
    for left in range(0, len(spectra), batch_size):
        batch = (
            spectra[left : left + batch_size].to(device).detach().requires_grad_(True)
        )
        reference_delta = torch.as_tensor(
            deltas[left : left + batch_size],
            dtype=batch.dtype,
            device=device,
        )
        embedding = model(batch)
        margin = torch.sum(embedding * reference_delta, dim=1)
        gradient = torch.autograd.grad(margin.sum(), batch, create_graph=False)[0]
        # Convert d margin / d intensity into d margin / d log(intensity).
        output.append((gradient[:, 1:, 1] * batch[:, 1:, 1]).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def sparse_role_jacobians(
    model,
    graph: RetrievalGraph,
    selected: np.ndarray,
    positions: np.ndarray,
    observable: np.ndarray,
    candidate_negative: list[np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    clean,
    device,
    batch_size: int,
) -> np.ndarray:
    """Differentiate only positions with an observable same-formula boundary."""
    output = np.zeros((len(positions), clean.shape[1] - 1), dtype=np.float32)
    active_local = np.flatnonzero(observable[positions])
    if not len(active_local):
        return output
    active_positions = positions[active_local]
    deltas = reference_deltas(
        graph,
        selected,
        active_positions,
        candidate_negative,
        official,
        row_position,
    )
    output[active_local] = input_jacobians(
        model,
        clean[active_positions],
        deltas,
        device,
        batch_size,
    )
    return output


def build_common_scope_plans(
    setting: JacobianSetting,
    position: int,
    target_profile: list[EvidenceProfile],
    reversed_profile: list[EvidenceProfile],
    permuted_profile: list[EvidenceProfile],
    clean,
    jacobian: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[PeakActionPlan], list[dict[str, float]]] | None:
    common = (
        jacobian,
        clean[position, 1:, 0].numpy(),
        clean[position, 1:, 1].numpy(),
        float(clean[position, 0, 0]),
    )
    target = build_jacobian_intersection_plan(
        target_profile[position],
        *common,
        setting.mode,
        setting.log_dose,
        setting.top_k,
        setting.minimum_abs_evidence,
        args.minimum_agreement,
        args.minimum_observed_intensity,
        args.precursor_exclusion_da,
    )
    if target.abstained:
        return None
    controls = []
    errors = []
    for profile in (reversed_profile[position], permuted_profile[position]):
        control = matched_jacobian_control_plan(
            profile,
            *common,
            target,
            args.minimum_agreement,
            setting.minimum_abs_evidence,
            args.minimum_observed_intensity,
            args.precursor_exclusion_da,
        )
        if control.abstained or capacity_signature(control) != capacity_signature(
            target
        ):
            return None
        controls.append(control)
        errors.append(
            jacobian_match_error(
                target,
                control,
                jacobian,
                clean[position, 1:, 1].numpy(),
            )
        )
    inverse = invert_action_plan(target)
    if (
        not np.array_equal(inverse.positions, target.positions)
        or not np.allclose(inverse.factors, 1.0 / target.factors)
        or not np.array_equal(inverse.roles, -target.roles)
    ):
        raise RuntimeError("direction inverse contract failed")
    return [target, *controls, inverse], errors


def screen_positions(
    settings: tuple[JacobianSetting, ...],
    positions: np.ndarray,
    formula: np.ndarray,
    profiles: tuple[
        list[EvidenceProfile], list[EvidenceProfile], list[EvidenceProfile]
    ],
    clean,
    jacobians: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, float]]]]:
    gains = np.zeros((len(settings), len(ARMS), len(positions)), dtype=np.float32)
    active = np.zeros((len(settings), len(positions)), dtype=bool)
    match_errors: list[list[dict[str, float]]] = [[] for _ in settings]
    for setting_id, setting in enumerate(settings):
        for local, position in enumerate(positions):
            built = build_common_scope_plans(
                setting,
                int(position),
                *profiles,
                clean,
                jacobians[local],
                args,
            )
            if built is None:
                continue
            plans, errors = built
            active[setting_id, local] = True
            match_errors[setting_id].extend(errors)
            gains[setting_id, :, local] = [
                first_order_log_intensity_gain(plan, jacobians[local]) for plan in plans
            ]
            if not np.isclose(
                gains[setting_id, 0, local], -gains[setting_id, 3, local]
            ):
                raise RuntimeError(
                    "same-peak direction reversal lost first-order antisymmetry"
                )
    return gains, active, match_errors


def mean_error(rows: list[dict[str, float]], key: str) -> float | None:
    return float(np.mean([row[key] for row in rows])) if rows else None


def summarize_setting(
    setting_id: int,
    setting: JacobianSetting,
    positions: np.ndarray,
    formula: np.ndarray,
    gains: np.ndarray,
    active: np.ndarray,
    match_errors: list[dict[str, float]],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict:
    comparisons = {
        "absolute_first_order_gain": formula_bootstrap(
            gains[0],
            formula[positions],
            args.seed + seed_offset + setting_id,
            args.bootstrap_draws,
        ),
        "minus_candidate_role_reversed": formula_bootstrap(
            gains[0] - gains[1],
            formula[positions],
            args.seed + seed_offset + 100 + setting_id,
            args.bootstrap_draws,
        ),
        "minus_intensity_rank_permuted": formula_bootstrap(
            gains[0] - gains[2],
            formula[positions],
            args.seed + seed_offset + 200 + setting_id,
            args.bootstrap_draws,
        ),
    }
    active_formulas = len(np.unique(formula[positions][active]))
    mean_j_error = mean_error(match_errors, "mean_abs_log_jacobian_error")
    mean_i_error = mean_error(match_errors, "mean_abs_log_intensity_error")
    gates = {
        "minimum_action_queries": int(np.sum(active)) >= args.minimum_action_queries,
        "minimum_action_formulas": active_formulas >= args.minimum_action_formulas,
        "minimum_action_fraction": float(np.mean(active))
        >= args.minimum_action_fraction,
        "absolute_gain_ci_positive": comparisons["absolute_first_order_gain"][
            "formula_cluster_bootstrap_95ci"
        ][0]
        > 0,
        "candidate_control_ci_positive": comparisons["minus_candidate_role_reversed"][
            "formula_cluster_bootstrap_95ci"
        ][0]
        > 0,
        "peak_control_ci_positive": comparisons["minus_intensity_rank_permuted"][
            "formula_cluster_bootstrap_95ci"
        ][0]
        > 0,
        "jacobian_match_error_bounded": mean_j_error is not None
        and mean_j_error <= args.maximum_mean_log_jacobian_match_error,
        "intensity_match_error_bounded": mean_i_error is not None
        and mean_i_error <= args.maximum_mean_log_intensity_match_error,
    }
    return {
        "setting_id": setting_id,
        **asdict(setting),
        "active_queries": int(np.sum(active)),
        "active_fraction": float(np.mean(active)),
        "active_formulas": active_formulas,
        "mean_gain_by_arm": dict(zip(ARMS, map(float, np.mean(gains, axis=1)))),
        "matched_control_error": {
            "mean_abs_log_jacobian_error": mean_j_error,
            "mean_abs_log_intensity_error": mean_i_error,
        },
        "comparisons": comparisons,
        "gates": gates,
        "qualified": bool(all(gates.values())),
    }


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
        raise ValueError("formula-fold roles must be distinct and in range")
    if args.bootstrap_draws < 10_000:
        raise ValueError("formula-cluster bootstrap requires at least 10,000 draws")
    if args.minimum_negative_count < 2:
        raise ValueError("Jacobian screen requires multiple same-formula negatives")
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
        or teacher_ptr[0] != 0
        or teacher_ptr[-1] != len(prediction)
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
    observable = np.zeros(len(selected), dtype=bool)
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
        observable[position] = len(negatives) >= args.minimum_negative_count

    torch.set_num_threads(args.torch_threads)
    store = SpectrumStore(args.data, graph.query_row[selected], args.n_highest_peaks)
    clean = torch.stack([store.one(int(graph.query_row[query])) for query in selected])
    target_profile: list[EvidenceProfile] = []
    reversed_profile: list[EvidenceProfile] = []
    permuted_profile: list[EvidenceProfile] = []
    for position in range(len(selected)):
        width = len(clean[position]) - 1
        negatives = candidate_negative[position]
        if not observable[position]:
            target_profile.append(empty_profile(width))
            reversed_profile.append(empty_profile(width))
            permuted_profile.append(empty_profile(width))
            continue
        left, right = map(int, teacher_ptr[position : position + 2])
        block = np.asarray(prediction[left:right], dtype=np.float32)
        mz = clean[position, 1:, 0].numpy()
        target = consensus_evidence(
            block,
            0,
            int(negatives[0]),
            negatives,
            mz,
            args.agreement_quantile,
            args.minimum_prediction,
        )
        swapped_negatives = np.r_[0, negatives[1:]].astype(np.int64)
        reversed_value = consensus_evidence(
            block,
            int(negatives[0]),
            0,
            swapped_negatives,
            mz,
            args.agreement_quantile,
            args.minimum_prediction,
        )
        target_profile.append(target)
        reversed_profile.append(reversed_value)
        permuted_profile.append(
            intensity_rank_permuted_profile(
                target,
                clean[position, 1:, 1].numpy(),
                args.seed + position,
            )
        )

    preflight = {
        "status": "CHEMAWARE_JACOBIAN_INTERSECTION_PREFLIGHT_PASS",
        "weights_updated": False,
        "perturbed_spectra_encoded": False,
        "settings": len(SETTINGS),
        "arms": list(ARMS),
        "teacher_queries": len(selected),
        "same_formula_multi_negative_queries": int(np.sum(observable)),
        "fold_roles": {
            "discovery_queries": len(discovery),
            "confirmation_queries": len(confirmation),
            "embedding_evaluation_queries_untouched": len(embedding_evaluation),
            "reserve_queries_untouched": len(reserve),
        },
        "planned_query_backward_passes_before_confirmation": len(discovery),
        "graph_sha256": sha256_file(args.graph),
        "teacher_predictions_sha256": sha256_file(
            args.teacher_dir / "iceberg_predictions_f16.npy"
        ),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    token_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(token_rows)}
    needed_rows = set(map(int, graph.query_row[selected]))
    for query in selected:
        _, rows, _, _ = graph.query_block(int(query))
        needed_rows.update(map(int, rows))
    if not needed_rows.issubset(row_position):
        raise RuntimeError("query or candidate rows are absent from the official cache")

    model, _ = load_base_model(
        args.official_checkpoint,
        args.architecture_checkpoint,
        device,
        args.n_highest_peaks,
    )
    discovery_jacobian = sparse_role_jacobians(
        model,
        graph,
        selected,
        discovery,
        observable,
        candidate_negative,
        official,
        row_position,
        clean,
        device,
        args.batch_size,
    )
    profiles = (target_profile, reversed_profile, permuted_profile)
    discovery_gain, discovery_active, discovery_errors = screen_positions(
        SETTINGS,
        discovery,
        formula,
        profiles,
        clean,
        discovery_jacobian,
        args,
    )
    discovery_reports = [
        summarize_setting(
            setting_id,
            setting,
            discovery,
            formula,
            discovery_gain[setting_id],
            discovery_active[setting_id],
            discovery_errors[setting_id],
            args,
            0,
        )
        for setting_id, setting in enumerate(SETTINGS)
    ]
    qualified = [row["setting_id"] for row in discovery_reports if row["qualified"]]
    selected_setting = (
        max(
            qualified,
            key=lambda index: min(
                discovery_reports[index]["comparisons"][name]["formula_macro_mean"]
                for name in (
                    "absolute_first_order_gain",
                    "minus_candidate_role_reversed",
                    "minus_intensity_rank_permuted",
                )
            ),
        )
        if qualified
        else None
    )

    confirmation_report = None
    pass_to_nonlinear_action_audit = False
    confirmation_gain = np.empty((0, len(ARMS), len(confirmation)), dtype=np.float32)
    confirmation_active = np.empty((0, len(confirmation)), dtype=bool)
    if selected_setting is not None:
        confirmation_jacobian = sparse_role_jacobians(
            model,
            graph,
            selected,
            confirmation,
            observable,
            candidate_negative,
            official,
            row_position,
            clean,
            device,
            args.batch_size,
        )
        setting_tuple = (SETTINGS[selected_setting],)
        confirmation_gain, confirmation_active, confirmation_errors = screen_positions(
            setting_tuple,
            confirmation,
            formula,
            profiles,
            clean,
            confirmation_jacobian,
            args,
        )
        confirmation_report = summarize_setting(
            selected_setting,
            SETTINGS[selected_setting],
            confirmation,
            formula,
            confirmation_gain[0],
            confirmation_active[0],
            confirmation_errors[0],
            args,
            1000,
        )
        pass_to_nonlinear_action_audit = confirmation_report["qualified"]

    report = {
        "status": (
            "CHEMAWARE_JACOBIAN_INTERSECTION_CONFIRMATION_PASS"
            if pass_to_nonlinear_action_audit
            else "CHEMAWARE_JACOBIAN_INTERSECTION_CONFIRMATION_FAIL"
            if selected_setting is not None
            else "CHEMAWARE_JACOBIAN_INTERSECTION_DISCOVERY_FAIL"
        ),
        "formal_training_authorized": False,
        "pass_to_nonlinear_action_audit": pass_to_nonlinear_action_audit,
        "preflight": preflight,
        "scope": {
            "weights_updated": False,
            "perturbed_spectra_encoded": False,
            "official_same_formula_boundary": True,
            "candidate_references_frozen": True,
            "input_jacobian_variable": "log observed peak intensity",
            "chemistry_and_positive_local_gain_required": True,
            "base_peak_and_max_normalization_fixed": True,
            "controls_match_edit_count_factor_intensity_and_abs_jacobian": True,
            "direction_control_uses_same_peak_slots": True,
            "confirmation_computed_only_after_discovery_pass": True,
            "embedding_evaluation_fold_inspected": False,
            "reserve_fold_inspected": False,
        },
        "discovery": {
            "qualified_settings": qualified,
            "selected_setting_id": selected_setting,
            "settings": discovery_reports,
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
        discovery_position=discovery,
        confirmation_position=confirmation,
        embedding_evaluation_position=embedding_evaluation,
        reserve_position=reserve,
        discovery_gain=discovery_gain,
        discovery_active=discovery_active,
        confirmation_gain=confirmation_gain,
        confirmation_active=confirmation_active,
        selected_setting=np.asarray(
            [-1 if selected_setting is None else selected_setting], dtype=np.int16
        ),
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
                "pass_to_nonlinear_action_audit": pass_to_nonlinear_action_audit,
                "selected_setting_id": selected_setting,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

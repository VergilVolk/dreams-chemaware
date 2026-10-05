"""Direct full-candidate fine-tuning of the official shared DreaMS encoder.

This is the capacity-upgrade stage after the cheap post-embedding screen.  It
updates one final Transformer block and the official projection head while
using exactly the same encoder for query and real reference spectra.  Formula
held-out candidates never enter training gradients; molecules and candidate
lists are training/evaluation scaffolding and are absent at deployment.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from audit_chemaware_mass_kernel_embedding import KernelCache as MassKernelCache  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    error_curriculum_queries, evaluate, formula_bootstrap,
    formula_identity_epoch_weights, identity_balanced_queries, listwise_losses,
    official_outcomes, sample_training_batch,
)
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, forward_embeddings,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-authorization", type=Path, default=None,
        help="Mandatory signed-by-evidence pilot ledger for any non-smoke GPU training.",
    )
    parser.add_argument(
        "--clean-observability-report", type=Path, default=None,
        help=(
            "Mandatory formal clean-spectrum observability report for rule_* teachers. "
            "The report must explicitly pass and match manifest/token/rule provenance."
        ),
    )
    parser.add_argument(
        "--data-semantics-report", type=Path,
        default=ROOT / "data/validation/chemaware_data_semantics_gate_v1/report.json",
        help="Mandatory membership-semantics and exact-row-set gate for every formal run.",
    )
    parser.add_argument(
        "--rule-library-admission-report", type=Path,
        default=ROOT / "data/validation/chemaware_rule_library_admission_v1/report.json",
        help=(
            "Mandatory empirical, formula-disjoint admission report for a formal rule teacher. "
            "A schema-only observation registry is insufficient."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-queries", type=int, default=2)
    parser.add_argument("--references-per-molecule", type=int, default=1)
    parser.add_argument("--max-train-identities", type=int, default=2048)
    parser.add_argument("--error-identity-fraction", type=float, default=0.5)
    parser.add_argument("--clean-safety-selection", choices=("random", "boundary_mixture"), default="random")
    parser.add_argument("--training-mass", choices=("identity", "formula_identity"), default="identity")
    parser.add_argument("--max-eval-identities", type=int, default=0)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-spectrum", type=float, default=1.0)
    parser.add_argument("--lambda-inbatch-spectrum", type=float, default=0.25)
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument(
        "--teacher-arm",
        choices=(
            "none", "mass", "intensity_permuted", "mass_shifted",
            "rule_response", "rule_mass", "rule_mass_shifted",
            "rule_response_shifted", "rule_response_row_permuted", "rule_mass_row_permuted",
        ),
        default="none",
    )
    parser.add_argument("--teacher-beta", type=float, default=0.10)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument(
        "--teacher-objective",
        choices=(
            "normalized_kl", "rank_equivalent_kl", "positive_margin_transfer",
            "candidate_residual_huber",
        ),
        default="rank_equivalent_kl",
        help=(
            "normalized_kl reproduces the first formal experiment. "
            "rank_equivalent_kl removes the ranking-irrelevant 1+beta score "
            "compression. positive_margin_transfer distils only clipped, "
            "strictly beneficial chemistry margin increments. candidate_residual_huber "
            "preserves every query-reference chemistry residual with molecule-balanced dose."
        ),
    )
    parser.add_argument("--margin-transfer-alpha", type=float, default=0.50)
    parser.add_argument("--margin-transfer-cap", type=float, default=0.05)
    parser.add_argument("--margin-transfer-huber", type=float, default=0.01)
    parser.add_argument("--candidate-residual-alpha", type=float, default=0.50)
    parser.add_argument(
        "--candidate-residual-cap", type=float, default=0.0,
        help="Symmetric target cap; zero disables clipping to prevent silent saturation.",
    )
    parser.add_argument("--candidate-residual-huber", type=float, default=0.02)
    parser.add_argument(
        "--legacy-teacher-objective-reproduction-only", action="store_true",
        help=(
            "Permit obsolete KL/scalar-PMT rule objectives only to reproduce historical runs; "
            "never treats their output as a new formal ChemAware experiment."
        ),
    )
    parser.add_argument(
        "--teacher-gradient-ratio", type=float, default=0.0,
        help=(
            "If positive, rescale the chemical loss so its embedding-space "
            "gradient norm is this fraction of the non-chemical objective."
        ),
    )
    parser.add_argument("--teacher-gradient-scale-cap", type=float, default=4.0)
    parser.add_argument(
        "--teacher-scope",
        choices=("all", "official_error", "official_error_or_boundary"),
        default="all",
    )
    parser.add_argument("--teacher-boundary-margin", type=float, default=0.02)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_clean_observability_report(
    report_path: Path | None, manifest: Path, token_dir: Path, rule_library: Path,
) -> dict:
    if report_path is None:
        raise RuntimeError(
            "rule_* GPU training is blocked: --clean-observability-report is mandatory"
        )
    path = report_path / "report.json" if report_path.is_dir() else report_path
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "CLEAN_RULE_OBSERVABILITY_PASS"
            or report.get("formal") is not True
            or report.get("pass_to_gpu_training") is not True):
        raise RuntimeError(
            "rule_* GPU training is blocked: clean observability did not formally pass"
        )
    contract = report.get("feature_contract", {})
    required_contract = (
        "one_unmodified_query_spectrum_only", "candidate_scores_used_as_features",
        "candidate_ranks_used_as_features", "identity_used_as_feature", "formula_used_as_feature",
    )
    expected = (True, False, False, False, False)
    if tuple(contract.get(key) for key in required_contract) != expected:
        raise RuntimeError("clean observability feature contract is invalid")
    provenance = report.get("provenance", {})
    expected_hashes = {
        "manifest_sha256": sha256_file(manifest),
        "token_report_sha256": sha256_file(token_dir / "report.json"),
        "rule_library_sha256": sha256_file(rule_library),
    }
    if any(provenance.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("clean observability provenance differs from this training run")
    return {"path": str(path), "sha256": sha256_file(path), "status": report["status"]}


def validate_data_semantics_report(
    path: Path, manifest: Path, token_dir: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"formal training is blocked: missing data-semantics report {path}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "CHEMAWARE_DATA_SEMANTICS_PASS"
            or report.get("development_training_admissible") is not True
            or report.get("schema_contract", {}).get("membership_used_to_select_rows") is not False
            or report.get("schema_contract", {}).get("membership_blind_row_reconstruction_exact") is not True):
        raise RuntimeError("formal training is blocked: data-semantics gate did not pass")
    provenance = report.get("provenance", {})
    if provenance.get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("data-semantics report refers to a different candidate manifest")
    token_report = json.loads((token_dir / "report.json").read_text(encoding="utf-8"))
    token_manifest_hash = token_report.get("provenance", {}).get("manifest_sha256")
    if token_manifest_hash is not None and token_manifest_hash != provenance["manifest_sha256"]:
        raise RuntimeError("official token cache refers to a different candidate manifest")
    return {
        "path": str(path), "sha256": sha256_file(path), "status": report["status"],
        "release_eligible": bool(report.get("release_eligible", False)),
    }


def validate_rule_library_admission(path: Path, rule_library: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"formal rule training is blocked: missing empirical rule-library admission {path}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "CHEMAWARE_RULE_LIBRARY_EMPIRICALLY_ADMITTED"
            or report.get("formal_training_authorized") is not True
            or report.get("formula_disjoint_confirmation_pass") is not True
            or report.get("negative_mode_excluded") is not True
            or report.get("mass_degenerate_aliases_merged") is not True):
        raise RuntimeError("formal rule training is blocked: rule-library admission did not pass")
    if report.get("admitted_library_sha256") != sha256_file(rule_library):
        raise RuntimeError("rule-library admission refers to a different library")
    return {"path": str(path), "sha256": sha256_file(path), "status": report["status"]}


def evaluation_rows(body: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    rows = [body["query_row"][np.asarray(queries, dtype=np.int64)]]
    candidate_rows = []
    for query in np.asarray(queries, dtype=np.int64):
        left, right = map(int, body["query_ptr"][query:query + 2])
        edge_left = int(body["molecule_ptr"][left])
        edge_right = int(body["molecule_ptr"][right])
        candidate_rows.append(body["pair_candidate_row"][edge_left:edge_right])
    if candidate_rows:
        rows.append(np.concatenate(candidate_rows))
    return np.unique(np.concatenate(rows)).astype(np.int64)


def kernel_teacher_loss(
    query_z: torch.Tensor, reference_z: torch.Tensor,
    official_query: torch.Tensor, official_reference: torch.Tensor,
    query_rows: np.ndarray, reference_rows: np.ndarray,
    candidate_ptr: np.ndarray, reference_ptr: np.ndarray,
    reference_edge: np.ndarray, cache: MassKernelCache, variant: str,
    beta: float, temperature: float, query_weight: torch.Tensor,
    objective: str = "normalized_kl", margin_transfer_alpha: float = 0.50,
    margin_transfer_cap: float = 0.05, margin_transfer_huber: float = 0.01,
    candidate_residual_alpha: float = 0.50, candidate_residual_cap: float = 0.0,
    candidate_residual_huber: float = 0.02,
    return_active_fraction: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Distil a clean-visible shared mass-kernel teacher into DreaMS geometry.

    ``normalized_kl`` is retained only to reproduce the first formal run.  Its
    division by ``1 + beta`` preserves teacher ranks but changes softmax scale,
    so it mixes chemical supervision with a ranking-irrelevant temperature
    target.  ``rank_equivalent_kl`` uses the same ranks without that scale
    confounder.  ``positive_margin_transfer`` goes one step further: it asks
    the student to inherit only the teacher's positive-vs-hardest-negative
    margin increment and assigns zero chemical weight to harmful/no-op rows.
    ``candidate_residual_huber`` supervises every query-reference residual,
    with equal weight per molecule, instead of reducing chemistry to one scalar.
    """
    if variant == "none":
        zero = query_z.sum() * 0.0
        return (zero, 0.0) if return_active_fraction else zero
    if beta < 0 or temperature <= 0:
        raise ValueError("teacher beta must be non-negative and temperature positive")
    if (margin_transfer_alpha < 0 or margin_transfer_cap < 0
            or margin_transfer_huber <= 0):
        raise ValueError("invalid positive-margin-transfer hyperparameters")
    if (candidate_residual_alpha < 0 or candidate_residual_cap < 0
            or candidate_residual_huber <= 0):
        raise ValueError("invalid candidate-residual hyperparameters")
    device = query_z.device
    query_kernel = torch.from_numpy(np.stack([
        cache.get(int(row))[variant] for row in query_rows
    ]).astype(np.float32)).to(device)
    reference_kernel = torch.from_numpy(np.stack([
        cache.get(int(row))[variant] for row in reference_rows
    ]).astype(np.float32)).to(device)
    edge = torch.as_tensor(reference_edge, device=device)
    losses = []
    active = []
    for index, (left, right) in enumerate(zip(candidate_ptr[:-1], candidate_ptr[1:])):
        student, official, fused = [], [], []
        molecule_residual_losses = []
        for molecule in range(int(left), int(right)):
            rleft, rright = map(int, reference_ptr[molecule:molecule + 2])
            selected = edge[rleft:rright]
            student.append((reference_z[selected] @ query_z[index]).max())
            official_pair = official_reference[selected] @ official_query[index]
            kernel_pair = reference_kernel[selected] @ query_kernel[index]
            # Pair-first fusion: official and kernel evidence must come from
            # the same reference spectrum before the molecule-level maximum.
            official.append(official_pair.max())
            fused.append((official_pair + beta * kernel_pair).max())
            target_residual = candidate_residual_alpha * beta * kernel_pair
            if candidate_residual_cap > 0:
                target_residual = torch.clamp(
                    target_residual, min=-candidate_residual_cap, max=candidate_residual_cap,
                )
            observed_residual = reference_z[selected] @ query_z[index] - official_pair.detach()
            molecule_residual_losses.append(F.smooth_l1_loss(
                observed_residual, target_residual.detach(),
                beta=candidate_residual_huber, reduction="mean",
            ))
        student_score = torch.stack(student)
        official_score = torch.stack(official)
        rank_equivalent_teacher = torch.stack(fused)

        if objective in ("normalized_kl", "rank_equivalent_kl"):
            teacher_score = rank_equivalent_teacher
            if objective == "normalized_kl":
                teacher_score = teacher_score / (1.0 + beta)
            teacher_probability = F.softmax(
                teacher_score / temperature, dim=0,
            ).detach()
            losses.append(-(
                teacher_probability
                * F.log_softmax(student_score / temperature, dim=0)
            ).sum())
            active.append(student_score.new_tensor(1.0))
        elif objective == "positive_margin_transfer":
            if len(student_score) < 2:
                raise ValueError("margin transfer requires at least one negative candidate")
            student_margin = student_score[0] - torch.max(student_score[1:])
            official_margin = official_score[0] - torch.max(official_score[1:])
            teacher_margin = (
                rank_equivalent_teacher[0]
                - torch.max(rank_equivalent_teacher[1:])
            )
            advantage = torch.clamp(
                teacher_margin - official_margin,
                min=0.0, max=margin_transfer_cap,
            ).detach()
            is_active = (advantage > 0).to(student_score.dtype)
            target_delta = margin_transfer_alpha * advantage
            losses.append(F.smooth_l1_loss(
                student_margin - official_margin.detach(), target_delta,
                beta=margin_transfer_huber, reduction="none",
            ))
            active.append(is_active)
        elif objective == "candidate_residual_huber":
            if not molecule_residual_losses:
                raise ValueError("candidate residual requires at least one candidate molecule")
            losses.append(torch.stack(molecule_residual_losses).mean())
            active.append(student_score.new_tensor(float(
                beta > 0 and candidate_residual_alpha > 0
            )))
        else:
            raise ValueError(f"unknown teacher objective: {objective}")
    weight = query_weight.to(device=device, dtype=query_z.dtype)
    active_weight = weight * torch.stack(active)
    if float(active_weight.sum().detach()) <= 0:
        zero = query_z.sum() * 0.0
        return (zero, 0.0) if return_active_fraction else zero
    # Preserve the requested mean teacher dose after routing and PMT's strict
    # positive-advantage gate.  This avoids silently weakening sparse batches.
    active_weight = active_weight * (weight.sum() / active_weight.sum()).detach()
    loss = torch.mean(torch.stack(losses) * active_weight)
    active_fraction = float(torch.count_nonzero(active_weight).detach()) / max(1, len(weight))
    return (loss, active_fraction) if return_active_fraction else loss


def chemical_teacher_query_weights(
    base_weight: torch.Tensor,
    official_error: np.ndarray,
    official_margin: np.ndarray,
    scope: str,
    boundary_margin: float,
) -> torch.Tensor:
    """Concentrate privileged chemistry on errors without changing mean dose.

    Candidate-derived error and margin labels are training-only routing
    variables.  They never enter the shared encoder.  Rescaling the active
    weights to the original sum keeps ``lambda_teacher`` comparable across
    routing arms; a batch without an eligible query receives zero chemistry.
    """
    if boundary_margin < 0:
        raise ValueError("teacher boundary margin must be non-negative")
    if scope == "all":
        return base_weight
    error = torch.as_tensor(
        np.asarray(official_error, dtype=bool), device=base_weight.device,
    )
    if scope == "official_error":
        active = error
    elif scope == "official_error_or_boundary":
        margin = torch.as_tensor(
            np.asarray(official_margin), device=base_weight.device,
            dtype=base_weight.dtype,
        )
        active = error | (margin <= boundary_margin)
    else:
        raise ValueError(f"unknown teacher scope: {scope}")
    routed = base_weight * active.to(base_weight.dtype)
    routed_sum = routed.sum()
    if float(routed_sum.detach()) <= 0:
        return torch.zeros_like(base_weight)
    return routed * (base_weight.sum() / routed_sum).detach()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, store: SpectrumStore, eval_rows: np.ndarray,
    body: dict[str, np.ndarray], inner: np.ndarray, official: np.ndarray,
    row_position: dict[int, int], device: torch.device, args: argparse.Namespace,
    label: str,
) -> tuple[dict, float, np.ndarray]:
    encoded = encode_rows(
        model, store, eval_rows, device, args.eval_batch_size, args.amp, label,
    )
    adapted = np.array(official, copy=True)
    positions = np.asarray([row_position[int(row)] for row in eval_rows], dtype=np.int64)
    adapted[positions] = encoded
    result = evaluate(body, inner, official, adapted, row_position)
    preservation = float(np.mean(np.sum(encoded * official[positions], axis=1)))
    return result, preservation, adapted


def main() -> None:
    args = arguments()
    started = time.time()
    if (args.inner_fold == args.outer_fold or args.head_lr < args.backbone_lr
            or args.teacher_beta < 0 or args.teacher_temperature <= 0
            or args.lambda_teacher < 0 or args.teacher_boundary_margin < 0
            or args.margin_transfer_alpha < 0 or args.margin_transfer_cap < 0
            or args.margin_transfer_huber <= 0 or args.teacher_gradient_ratio < 0
            or args.candidate_residual_alpha < 0 or args.candidate_residual_cap < 0
            or args.candidate_residual_huber <= 0
            or args.teacher_gradient_scale_cap <= 0):
        raise ValueError("invalid fold or learning-rate configuration")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    is_rule_teacher = args.teacher_arm.startswith("rule_")
    if is_rule_teacher and not args.smoke:
        if (args.teacher_objective != "candidate_residual_huber"
                and not args.legacy_teacher_objective_reproduction_only):
            raise RuntimeError(
                "formal rule-teacher training is blocked: KL and scalar PMT discard candidate-specific "
                "chemistry; use --teacher-objective candidate_residual_huber"
            )
        if args.teacher_scope != "all" and not args.legacy_teacher_objective_reproduction_only:
            raise RuntimeError(
                "formal rule-teacher training is blocked: candidate-derived routing is privileged; "
                "use --teacher-scope all"
            )
    required = [
        args.manifest, args.token_dir / "report.json", args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.data,
        args.official_checkpoint, args.architecture_checkpoint,
    ]
    if is_rule_teacher:
        required.append(args.rule_library)
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if token_report.get("status") != "chemaware_corrected_manifest_token_cache_complete":
        raise RuntimeError("direct training requires the complete official cache")
    data_semantics = None if args.smoke else validate_data_semantics_report(
        args.data_semantics_report, args.manifest, args.token_dir,
    )
    observability = None
    rule_library_admission = None
    if is_rule_teacher and not args.smoke:
        rule_library_admission = validate_rule_library_admission(
            args.rule_library_admission_report, args.rule_library,
        )
        observability = validate_clean_observability_report(
            args.clean_observability_report, args.manifest, args.token_dir, args.rule_library,
        )
    gpu_authorization = None
    if not args.smoke and not args.preflight_only:
        if args.gpu_authorization is None:
            raise RuntimeError(
                "formal GPU training is blocked: --gpu-authorization is mandatory; "
                "submit through the bounded pilot wrapper"
            )
        from verify_chemaware_gpu_authorization import verify as verify_gpu_authorization  # noqa: PLC0415
        verified = verify_gpu_authorization(
            args.gpu_authorization, args.clean_observability_report,
        )
        gpu_authorization = {
            "path": str(args.gpu_authorization),
            "sha256": sha256_file(args.gpu_authorization),
            "status": verified["status"],
        }
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    reachable = set(map(int, np.unique(np.r_[body["query_row"], body["pair_candidate_row"]])))
    if reachable != set(map(int, rows)):
        raise RuntimeError("official cache and manifest-reachable rows differ")

    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    molecule_fold = stable_formula_folds(body["molecule_formula"], args.folds, args.fold_seed)
    molecule_allowed = (molecule_fold != args.inner_fold) & (molecule_fold != args.outer_fold)
    allowed_count = np.add.reduceat(molecule_allowed.astype(np.int32), body["query_ptr"][:-1])
    train_pool = np.flatnonzero(
        (fold != args.inner_fold) & (fold != args.outer_fold) & (allowed_count >= 2)
    )
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer_pool = np.flatnonzero(fold == args.outer_fold)
    train_error, train_margin = official_outcomes(
        body, train_pool, official, row_position, molecule_allowed,
    )
    effective_eval_identities = (
        min(args.max_eval_identities, 8) if args.smoke and args.max_eval_identities > 0
        else 8 if args.smoke else args.max_eval_identities
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        effective_eval_identities,
    )
    eval_rows = evaluation_rows(body, inner)
    preflight = {
        "status": "chemaware_full_candidate_direct_preflight_passed",
        "manifest_queries": int(len(body["query_row"])),
        "reachable_spectra": int(len(rows)),
        "train_query_pool": int(len(train_pool)),
        "train_official_errors": int(np.sum(train_error[train_pool])),
        "inner_eval_identities": int(len(inner)),
        "inner_eval_reachable_spectra": int(len(eval_rows)),
        "outer_queries_reserved_not_evaluated": int(len(outer_pool)),
        "contracts": {
            "formula_disjoint": True,
            "held_formula_candidates_excluded_from_training_gradients": True,
            "same_raw_spectrum_encoder_query_reference": True,
            "official_checkpoint_initialization": True,
            "candidate_input_at_deployment": False,
            "teacher_input_clean_spectrum_only": args.teacher_arm != "none",
            "teacher_discarded_at_deployment": True,
            "outer_fold_evaluated": False,
            "outer_split_membership_used_only_for_gradient_exclusion": True,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "clean_observability_report": observability,
            "data_semantics_report": data_semantics,
            "rule_library_admission_report": rule_library_admission,
            "gpu_authorization": gpu_authorization,
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    torch.set_num_threads(args.torch_threads); seed_everything(args.seed)
    device = torch.device(args.device)
    # Freeze the complete identity/candidate schedule before loading raw spectra.
    # Besides making arms reproducible, this avoids preprocessing tens of
    # thousands of rows that cannot be touched by this run.
    rng = np.random.default_rng(args.seed + 101)
    replay_count = 8 if args.smoke else 64
    replay_rows = rows[np.linspace(0, len(rows) - 1, replay_count, dtype=np.int64)]
    scheduled_rows = set(map(int, np.concatenate((eval_rows, replay_rows))))
    epoch_schedules: list[list[tuple[np.ndarray, dict, np.ndarray]]] = []
    batch_reference_counts: list[int] = []
    batch_candidate_counts: list[int] = []
    effective_epochs = 1 if args.smoke else args.epochs
    effective_train_identities = (
        min(args.max_train_identities, 4) if args.smoke and args.max_train_identities > 0
        else 4 if args.smoke else args.max_train_identities
    )
    for _epoch in range(1, effective_epochs + 1):
        epoch_query = error_curriculum_queries(
            train_pool, body["query_ik14"], train_error, train_margin, rng,
            effective_train_identities, args.error_identity_fraction,
            args.clean_safety_selection,
        )
        rng.shuffle(epoch_query)
        epoch_weight = formula_identity_epoch_weights(
            epoch_query, body["query_formula"], args.training_mass,
        )
        weight_by_query = dict(zip(map(int, epoch_query), map(float, epoch_weight)))
        batches = []
        for left in range(0, len(epoch_query), args.batch_queries):
            queries = epoch_query[left:left + args.batch_queries]
            batch = sample_training_batch(
                body, queries, row_position, None, args.references_per_molecule,
                rng, molecule_allowed,
            )
            weights = np.asarray(
                [weight_by_query[int(query)] for query in queries], dtype=np.float32,
            )
            scheduled_rows.update(map(int, rows[batch["query_cache"]]))
            scheduled_rows.update(map(int, rows[batch["reference_cache"]]))
            batch_reference_counts.append(int(len(batch["reference_cache"])))
            batch_candidate_counts.append(int(np.max(np.diff(batch["candidate_ptr"]))))
            batches.append((queries.copy(), batch, weights))
        epoch_schedules.append(batches)
    scheduled_rows_array = np.asarray(sorted(scheduled_rows), dtype=np.int64)
    preflight["schedule"] = {
        "epochs": int(effective_epochs),
        "requested_epochs": int(args.epochs),
        "max_train_identities": int(effective_train_identities),
        "max_eval_identities": int(effective_eval_identities),
        "optimizer_steps": int(sum(map(len, epoch_schedules))),
        "raw_spectra_materialized": int(len(scheduled_rows_array)),
        "maximum_unique_references_per_batch": int(max(batch_reference_counts, default=0)),
        "p95_unique_references_per_batch": float(
            np.quantile(batch_reference_counts, 0.95) if batch_reference_counts else 0
        ),
        "maximum_candidate_molecules_per_query": int(max(batch_candidate_counts, default=0)),
        "frozen_before_training": True,
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2)); return
    if (not args.smoke and
            (not args.device.startswith("cuda") or not torch.cuda.is_available())):
        raise RuntimeError("formal direct shared-encoder training requires CUDA")
    store = SpectrumStore(args.data, scheduled_rows_array, args.n_highest_peaks)
    kernel_variants = () if args.teacher_arm == "none" else (args.teacher_arm,)
    kernel_cache = MassKernelCache(args, row_position, variants=kernel_variants)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    # Cheap replay audit before any parameter is trainable.
    replay = encode_rows(model, store, replay_rows, device, args.eval_batch_size, False, "official-replay")
    replay_pos = np.asarray([row_position[int(row)] for row in replay_rows], dtype=np.int64)
    replay_cosine = np.sum(replay * official[replay_pos], axis=1)
    if float(np.min(replay_cosine)) < 0.999:
        raise RuntimeError(f"official checkpoint replay drift: min cosine={float(np.min(replay_cosine))}")
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()  # keep dropout off without disabling autograd
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    initial = evaluate(body, inner, official, official, row_position)
    best_state = None
    best_step = 0
    best_score = (initial["summary"]["recall1"], initial["summary"]["mrr"], 0.0, 1.0)
    best_eval = initial
    best_preservation = 1.0
    history = []
    global_step = 0
    for epoch, epoch_batches in enumerate(epoch_schedules, start=1):
        totals = {"loss": 0.0, "spectrum": 0.0, "inbatch_spectrum": 0.0,
                  "teacher": 0.0,
                  "teacher_active_fraction": 0.0,
                  "teacher_gradient_scale": 0.0,
                  "margin_floor": 0.0, "preserve": 0.0, "grad_norm": 0.0,
                  "clip_fraction": 0.0}
        batches = 0
        for queries, batch, weights in epoch_batches:
            query_rows = rows[batch["query_cache"]]
            reference_rows = rows[batch["reference_cache"]]
            spectra = torch.cat((store.get(query_rows), store.get(reference_rows))).to(device)
            encoded = forward_embeddings(model, spectra, args.amp)
            query_z = encoded[:len(queries)]
            reference_z = encoded[len(queries):]
            query_x = torch.from_numpy(np.array(official[batch["query_cache"]], copy=True)).to(device)
            reference_x = torch.from_numpy(np.array(official[batch["reference_cache"]], copy=True)).to(device)
            query_weight = torch.as_tensor(
                weights, device=device, dtype=query_z.dtype,
            )
            teacher_query_weight = chemical_teacher_query_weights(
                query_weight, train_error[queries], train_margin[queries],
                args.teacher_scope, args.teacher_boundary_margin,
            )
            spectrum_loss, _, inbatch_spectrum, _, margin_floor = listwise_losses(
                query_z, reference_z, None, batch["candidate_ptr"], batch["reference_ptr"],
                batch["reference_edge"], args.temperature, query_x, reference_x,
                args.margin_floor_slack, query_weight,
            )
            teacher_loss, teacher_active_fraction = kernel_teacher_loss(
                query_z, reference_z, query_x, reference_x,
                query_rows, reference_rows, batch["candidate_ptr"],
                batch["reference_ptr"], batch["reference_edge"], kernel_cache,
                args.teacher_arm, args.teacher_beta, args.teacher_temperature,
                teacher_query_weight, args.teacher_objective,
                args.margin_transfer_alpha, args.margin_transfer_cap,
                args.margin_transfer_huber,
                args.candidate_residual_alpha, args.candidate_residual_cap,
                args.candidate_residual_huber, True,
            )
            preserve = torch.cat((
                1 - torch.sum(query_z * query_x, dim=1),
                1 - torch.sum(reference_z * reference_x, dim=1),
            )).mean()
            base_loss = (args.lambda_spectrum * spectrum_loss
                         + args.lambda_inbatch_spectrum * inbatch_spectrum
                         + args.lambda_margin_floor * margin_floor
                         + args.lambda_preserve * preserve)
            teacher_gradient_scale = 1.0 if teacher_active_fraction > 0 else 0.0
            if args.teacher_gradient_ratio > 0 and teacher_active_fraction > 0:
                base_embedding_grad = torch.autograd.grad(
                    base_loss, (query_z, reference_z), retain_graph=True,
                    allow_unused=True,
                )
                teacher_embedding_grad = torch.autograd.grad(
                    args.lambda_teacher * teacher_loss,
                    (query_z, reference_z), retain_graph=True, allow_unused=True,
                )
                base_norm_sq = sum(
                    torch.sum(value.detach().float() ** 2)
                    for value in base_embedding_grad if value is not None
                )
                teacher_norm_sq = sum(
                    torch.sum(value.detach().float() ** 2)
                    for value in teacher_embedding_grad if value is not None
                )
                teacher_norm = torch.sqrt(teacher_norm_sq)
                if float(teacher_norm) > 0:
                    desired = (
                        args.teacher_gradient_ratio * torch.sqrt(base_norm_sq)
                        / teacher_norm
                    )
                    teacher_gradient_scale = float(torch.clamp(
                        desired, min=0.0, max=args.teacher_gradient_scale_cap,
                    ))
                else:
                    teacher_gradient_scale = 0.0
            loss = base_loss + (
                args.lambda_teacher * teacher_gradient_scale * teacher_loss
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            ))
            optimizer.step(); global_step += 1; batches += 1
            for key, value in (("loss", loss), ("spectrum", spectrum_loss),
                               ("inbatch_spectrum", inbatch_spectrum),
                               ("teacher", teacher_loss),
                               ("margin_floor", margin_floor), ("preserve", preserve)):
                totals[key] += float(value.detach())
            totals["teacher_active_fraction"] += teacher_active_fraction
            totals["teacher_gradient_scale"] += teacher_gradient_scale
            totals["grad_norm"] += norm
            totals["clip_fraction"] += float(norm > args.grad_clip)
            if global_step % 250 == 0:
                print(f"[direct epoch={epoch}] step={global_step} loss={totals['loss']/batches:.5f}", flush=True)
        current, preservation, _ = evaluate_model(
            model, store, eval_rows, body, inner, official, row_position,
            device, args, f"direct-epoch-{epoch}",
        )
        record = {
            "epoch": epoch, "global_step": global_step,
            "train": {key: value / max(1, batches) for key, value in totals.items()},
            "inner": current["summary"], "preservation": preservation,
        }
        history.append(record)
        score = (current["summary"]["recall1"], current["summary"]["mrr"],
                 current["summary"]["delta_mean_margin"], preservation)
        if preservation >= 0.995 and score > best_score:
            best_score = score; best_step = global_step; best_eval = current
            best_preservation = preservation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"direct epoch={epoch}/{effective_epochs} delta={current['summary']['delta_recall1']:+.4f} "
            f"corrected={current['summary']['corrected']} introduced={current['summary']['introduced']} "
            f"preserve={preservation:.6f} clip={record['train']['clip_fraction']:.3f}", flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    delta = ((best_eval["new_rank"] == 1).astype(float)
             - (best_eval["old_rank"] == 1).astype(float))
    ci = formula_bootstrap(
        delta, body["query_formula"][inner], args.seed + 901, args.bootstrap_draws,
    )
    mean_clip = float(np.mean([row["train"]["clip_fraction"] for row in history]))
    absolute_gates = {
        "inner_recall_positive": best_eval["summary"]["delta_recall1"] > 0,
        "formula_ci_positive": ci["formula_cluster_bootstrap_95ci"][0] > 0,
        "corrected_exceeds_introduced": best_eval["summary"]["corrected"] > best_eval["summary"]["introduced"],
        "mean_preservation": best_preservation >= 0.995,
        "clip_not_saturated": mean_clip < 0.9,
    }
    artifact_gates = {
        "selected_trained_step": best_step > 0,
        "mean_preservation": best_preservation >= 0.995,
        "clip_not_saturated": mean_clip < 0.9,
    }
    status = (
        "SMOKE_ONLY" if args.smoke else
        ("LEGACY_REPRODUCTION_ONLY" if args.legacy_teacher_objective_reproduction_only else
        ("ARM_COMPLETE" if all(artifact_gates.values()) else "ARM_SAFETY_FAIL")
        )
    )
    checkpoint = {
        # A single arm can establish artifact integrity and an absolute change
        # versus official DreaMS, but it cannot establish a chemical increment.
        # Keep it quarantined from the shared inference loader until a matched
        # multi-arm causal decision and an untouched confirmation explicitly
        # promote a separate release checkpoint.
        "status": "chemaware_full_candidate_direct_arm_checkpoint",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initialization": f"official_dreams:{initialization}",
        "capacity": capacity, "inner_fold": args.inner_fold,
        "outer_fold": args.outer_fold, "seed": args.seed,
        "inference_clean_spectrum_only": True,
        "inference_clean_only": True,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "mass_kernel_teacher_arm": args.teacher_arm,
        "mass_kernel_teacher_beta": args.teacher_beta,
        "mass_kernel_teacher_objective": args.teacher_objective,
        "mass_kernel_margin_transfer_alpha": args.margin_transfer_alpha,
        "mass_kernel_margin_transfer_cap": args.margin_transfer_cap,
        "mass_kernel_margin_transfer_huber": args.margin_transfer_huber,
        "candidate_residual_alpha": args.candidate_residual_alpha,
        "candidate_residual_cap": args.candidate_residual_cap,
        "candidate_residual_huber": args.candidate_residual_huber,
        "mass_kernel_teacher_gradient_ratio": args.teacher_gradient_ratio,
        "mass_kernel_teacher_gradient_scale_cap": args.teacher_gradient_scale_cap,
        "mass_kernel_teacher_scope": args.teacher_scope,
        "mass_kernel_teacher_boundary_margin": args.teacher_boundary_margin,
        "mass_kernel_teacher_discarded_at_inference": True,
        "P2b_used": False,
        "formal": (not args.smoke) and not args.legacy_teacher_objective_reproduction_only,
        "legacy_reproduction_only": args.legacy_teacher_objective_reproduction_only,
        "artifact_validation_pass": (not args.smoke) and all(artifact_gates.values()),
        "absolute_development_pass": (not args.smoke) and all(absolute_gates.values()),
        "causal_chemistry_pass": False,
        "release_eligible": False,
        "validation_pass": False,
        "provenance": {
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "raw_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "manifest_sha256": sha256_file(args.manifest),
        },
    }
    torch.save(checkpoint, args.output / "final_shared_encoder.pt")
    report = {
        "status": status, "preflight": preflight, "optimization": vars(args),
        "initial_inner": initial["summary"], "final_inner": best_eval["summary"],
        "formula_bootstrap": ci, "selected_step": best_step,
        "preservation": best_preservation, "mean_clip_fraction": mean_clip,
        "artifact_gates": artifact_gates,
        "absolute_vs_official_gates": absolute_gates,
        "causal_chemistry_status": "NOT_EVALUATED_SINGLE_ARM",
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    report["optimization"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["optimization"].items()
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "inner_per_query.npz",
        query=inner.astype(np.int64),
        formula=body["query_formula"][inner],
        old_rank=best_eval["old_rank"].astype(np.int16),
        new_rank=best_eval["new_rank"].astype(np.int16),
    )
    print(json.dumps({
        "status": status, "selected_step": best_step,
        "causal_chemistry_status": "NOT_EVALUATED_SINGLE_ARM",
        "release_eligible": False,
        "final_inner": best_eval["summary"], "formula_bootstrap": ci,
        "preservation": best_preservation, "mean_clip_fraction": mean_clip,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

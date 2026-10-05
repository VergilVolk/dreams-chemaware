"""Replay every exact Noise triplet under the official native DreaMS start.

The mature action ledger was selected in the later E8 geometry.  Native
continuation starts from the published fine-tuned checkpoint, so this audit
measures every canonical action-positive-negative margin again before any
optimizer step. Exact aliases were already collapsed outcome-blind by the
triplet builder and remain fully represented in its provenance mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from train_noise_dreams_native import (
    NATIVE_LOADER_COMPAT_VERSION,
    NATIVE_MAX_EPOCHS,
    NATIVE_MAX_OPTIMIZER_STEPS,
    NATIVE_SCHEDULE_VERSION,
    load_trusted_native_contrastive_head,
    make_hdf5_spectrum,
    native_query_disjoint_one_pass_batches,
)
from noise_dreams_native_spectrum import (
    NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION,
    native_action_model_input,
)


NATIVE_REPLAY_GEOMETRY_VERSION = "native_hard_positive_only_official_v16"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--native-embedding-cache-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


@torch.no_grad()
def encode_actions(
    model: ContrastiveHead,
    spectra: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    label: str,
    preprocessor: SpectrumPreprocessor,
) -> np.ndarray:
    if spectra.ndim != 3 or spectra.shape[1:] != (101, 2):
        raise RuntimeError(f"{label} action bank has invalid shape {spectra.shape}")
    output = np.empty((len(spectra), 1024), dtype=np.float32)
    model.eval()
    for left in range(0, len(spectra), batch_size):
        right = min(left + batch_size, len(spectra))
        replay = []
        for source_tensor in spectra[left:right]:
            replay.append(native_action_model_input(source_tensor, preprocessor))
        tensor = torch.from_numpy(np.stack(replay).astype(np.float32)).to(device)
        encoded = F.normalize(model(tensor).float(), dim=1)
        output[left:right] = encoded.cpu().numpy()
        if right == len(spectra) or right % (batch_size * 100) == 0:
            print(f"[{label}-official-replay] {right:,}/{len(spectra):,}", flush=True)
    if not np.all(np.isfinite(output)):
        raise RuntimeError(f"{label} official replay produced non-finite embeddings")
    return output


def row_embeddings(
    cache: dict[str, np.ndarray], requested: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(cache["rows"], dtype=np.int64)
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    if (
        rows.ndim != 1
        or embeddings.shape != (len(rows), 1024)
        or len(np.unique(rows)) != len(rows)
    ):
        raise RuntimeError("official embedding cache schema is invalid")
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    at = np.searchsorted(sorted_rows, requested)
    if np.any(at >= len(sorted_rows)) or not np.array_equal(
        sorted_rows[at], requested
    ):
        raise RuntimeError("official cache misses an exact action boundary row")
    values = embeddings[order[at]]
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError("official boundary embeddings are not finite unit vectors")
    return values


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Mean under the exact production action-opportunity distribution."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise RuntimeError("weighted replay arrays are not aligned")
    total = float(np.sum(weights))
    if total <= 0 or not np.all(np.isfinite(values)) or not np.all(weights >= 0):
        raise RuntimeError("weighted replay received invalid values or weights")
    return float(np.dot(values, weights) / total)


@torch.no_grad()
def encode_measured_rows(
    model: ContrastiveHead,
    rows: np.ndarray,
    data: Path,
    *,
    batch_size: int,
    device: torch.device,
    preprocessor: SpectrumPreprocessor,
    label: str,
) -> np.ndarray:
    """Encode measured spectra with the exact model/preprocessor used in training."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise RuntimeError(f"{label} rows are not a unique one-dimensional registry")
    output = np.empty((len(rows), 1024), dtype=np.float32)
    model.eval()
    with h5py.File(data, "r") as handle:
        for left in range(0, len(rows), batch_size):
            right = min(left + batch_size, len(rows))
            spectra = []
            for row in rows[left:right]:
                spectrum = make_hdf5_spectrum(
                    handle["spectrum"][int(row)],
                    float(handle["precursor_mz"][int(row)]),
                )
                spectra.append(preprocessor(
                    spectrum.get_peak_list(),
                    prec_mz=spectrum.get_precursor_mz(),
                    high_form=False,
                ))
            tensor = torch.from_numpy(
                np.stack(spectra).astype(np.float32)
            ).to(device)
            output[left:right] = F.normalize(
                model(tensor).float(), dim=1
            ).cpu().numpy()
            if right == len(rows) or right % (batch_size * 100) == 0:
                print(f"[{label}] {right:,}/{len(rows):,}", flush=True)
    norms = np.linalg.norm(output, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError(f"{label} produced invalid native embeddings")
    return output


def compare_selection_and_native_geometry(
    selection_cache: dict[str, np.ndarray],
    native_rows: np.ndarray,
    native_embeddings: np.ndarray,
) -> dict[str, float | int | bool]:
    """Diagnose, but never mix, the action-selection and native geometries."""
    selection_rows = np.asarray(selection_cache["rows"], dtype=np.int64)
    selection_embeddings = np.asarray(
        selection_cache["embeddings"], dtype=np.float32
    )
    if (
        not np.array_equal(selection_rows, native_rows)
        or selection_embeddings.shape != native_embeddings.shape
    ):
        raise RuntimeError("selection and native embedding registries are not aligned")
    positions = np.unique(
        np.linspace(0, len(native_rows) - 1, 64, dtype=np.int64)
    )
    observed = native_embeddings[positions]
    expected = selection_embeddings[positions]
    cosine = np.einsum("ij,ij->i", observed, expected)
    maximum_error = float(np.max(np.abs(observed - expected)))
    return {
        "rows": int(len(positions)),
        "minimum_cosine": float(np.min(cosine)),
        "mean_cosine": float(np.mean(cosine)),
        "maximum_abs_error": maximum_error,
        "numerically_identical": bool(
            np.min(cosine) >= 0.99999 and maximum_error <= 2e-4
        ),
        "role": (
            "diagnostic_only; selection cache defines historical action mining, "
            "native cache defines replay, training initialization and official comparator"
        ),
    }


def native_tensor_from_action(
    source: np.ndarray, preprocessor: SpectrumPreprocessor,
) -> np.ndarray:
    return native_action_model_input(source, preprocessor)


def native_tensors_from_rows(
    data: Path,
    rows: np.ndarray,
    preprocessor: SpectrumPreprocessor,
) -> np.ndarray:
    output = []
    with h5py.File(data, "r") as handle:
        for row in np.asarray(rows, dtype=np.int64):
            spectrum = make_hdf5_spectrum(
                handle["spectrum"][int(row)],
                float(handle["precursor_mz"][int(row)]),
            )
            output.append(preprocessor(
                spectrum.get_peak_list(),
                prec_mz=spectrum.get_precursor_mz(),
                high_form=False,
            ))
    return np.stack(output).astype(np.float32)


def actual_action_gradient_audit(
    model: ContrastiveHead,
    actions: pd.DataFrame,
    targeted_spectra: np.ndarray,
    clean_identity_hinge: np.ndarray,
    hard_positive_hinge: np.ndarray,
    *,
    data: Path,
    preprocessor: SpectrumPreprocessor,
    device: torch.device,
) -> dict[str, object]:
    """Audit the clean-preservation and hard-positive production paths."""
    clean_active = np.flatnonzero(clean_identity_hinge > 0)
    hard_positive_active = np.flatnonzero(hard_positive_hinge > 0)

    def choose(active: np.ndarray) -> list[int]:
        # Prefer source diversity, then fill using distinct queries. Gradient
        # magnitude is never used for selection.
        chosen: list[int] = []
        used_queries: set[int] = set()
        used_sources: set[str] = set()
        for index in active:
            query = int(actions.at[int(index), "query_index"])
            source = str(actions.at[int(index), "source"])
            if query in used_queries or source in used_sources:
                continue
            chosen.append(int(index))
            used_queries.add(query)
            used_sources.add(source)
            if len(chosen) == 4:
                return chosen
        for index in active:
            query = int(actions.at[int(index), "query_index"])
            if query not in used_queries:
                chosen.append(int(index))
                used_queries.add(query)
            if len(chosen) == 4:
                return chosen
        return chosen

    chosen_by_mode = {
        "clean_identity": choose(clean_active),
        "hard_positive": choose(hard_positive_active),
    }
    if any(len(chosen) != 4 for chosen in chosen_by_mode.values()):
        return {
            "clean_identity_active_actions": int(len(clean_active)),
            "hard_positive_active_actions": int(len(hard_positive_active)),
            "gate_passed": False,
            "reason": "fewer than four distinct real queries activate one native path",
        }

    model.backbone.unfreeze()
    # Keep deterministic evaluation-mode geometry while retaining autograd.
    # Production training mode can add dropout noise; it must not be used to
    # manufacture or erase the pre-optimizer reachability proof.
    model.eval()

    def run_step(mode: str) -> dict[str, object]:
        chosen = chosen_by_mode[mode]
        selected = actions.iloc[chosen]
        clean_np = native_tensors_from_rows(
            data, selected["query_row"].to_numpy(np.int64), preprocessor
        )
        positive_np = native_tensors_from_rows(
            data, selected["action_positive_row"].to_numpy(np.int64), preprocessor
        )
        negative_np = native_tensors_from_rows(
            data, selected["action_hard_negative_row"].to_numpy(np.int64), preprocessor
        )
        action_np = np.stack([
            native_tensor_from_action(targeted_spectra[index], preprocessor)
            for index in chosen
        ])
        role_tensors = {
            "clean": torch.from_numpy(clean_np).to(device).requires_grad_(True),
            "action": torch.from_numpy(action_np).to(device).requires_grad_(True),
            "positive": torch.from_numpy(positive_np).to(device).requires_grad_(True),
            "negative": torch.from_numpy(negative_np).to(device).requires_grad_(True),
        }
        model.zero_grad(set_to_none=True)
        if mode == "clean_identity":
            roles = ("clean", "positive", "negative")
            batch = {
                "spec": role_tensors["clean"],
                "pos_specs": role_tensors["positive"][:, None],
                "neg_specs": role_tensors["negative"][:, None],
            }
        else:
            roles = ("action", "positive", "negative")
            batch = {
                "spec": role_tensors["action"],
                "pos_specs": role_tensors["positive"][:, None],
                "neg_specs": role_tensors["negative"][:, None],
            }
        _, loss = model.step(batch, 0)
        loss.backward()
        role_per_example_norms = {
            role: (
                role_tensors[role].grad.float().flatten(1).norm(dim=1)
                .detach().cpu().numpy().tolist()
                if role_tensors[role].grad is not None else [0.0] * len(chosen)
            )
            for role in roles
        }
        role_norms = {
            role: float(np.linalg.norm(values))
            for role, values in role_per_example_norms.items()
        }
        head_norm = float(torch.sqrt(sum(
            torch.sum(parameter.grad.float() ** 2)
            for parameter in model.head.parameters()
            if parameter.grad is not None
        )).detach().cpu())
        backbone_norm = float(torch.sqrt(sum(
            torch.sum(parameter.grad.float() ** 2)
            for parameter in model.backbone.parameters()
            if parameter.grad is not None
        )).detach().cpu())
        return {
            "audited_action_indices": chosen,
            "audited_action_ids": selected["action_id"].astype(str).tolist(),
            "audited_sources": selected["source"].astype(str).tolist(),
            "loss": float(loss.detach().cpu()),
            "role_gradient_norms": role_norms,
            "role_per_example_gradient_norms": role_per_example_norms,
            "head_gradient_norm": head_norm,
            "backbone_gradient_norm": backbone_norm,
            "all_expected_roles_live": bool(all(
                value > 0
                for values in role_per_example_norms.values()
                for value in values
            )),
            "head_and_backbone_live": bool(head_norm > 0 and backbone_norm > 0),
        }

    clean_identity = run_step("clean_identity")
    hard_positive = run_step("hard_positive")
    model.zero_grad(set_to_none=True)
    model.eval()
    passed = bool(
        clean_identity["loss"] > 0
        and hard_positive["loss"] > 0
        and clean_identity["all_expected_roles_live"]
        and hard_positive["all_expected_roles_live"]
        and clean_identity["head_and_backbone_live"]
        and hard_positive["head_and_backbone_live"]
    )
    return {
        "clean_identity_active_actions": int(len(clean_active)),
        "hard_positive_active_actions": int(len(hard_positive_active)),
        "clean_identity_step": clean_identity,
        "hard_positive_step": hard_positive,
        "all_four_roles_live_across_production_paths": bool(
            clean_identity["all_expected_roles_live"]
            and hard_positive["all_expected_roles_live"]
        ),
        "gate_passed": passed,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.native_embedding_cache_output.exists():
        raise FileExistsError(args.native_embedding_cache_output)
    if not torch.cuda.is_available():
        raise RuntimeError("official action replay requires an allocated GPU")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    actions_path = args.triplet_dir / "selected_actions.csv.gz"
    qualified_actions_path = args.triplet_dir / "qualified_actions.csv.gz"
    aliases_path = args.triplet_dir / "action_aliases.csv.gz"
    bank_path = args.triplet_dir / "action_spectra.npz"
    train_pool_path = args.triplet_dir / "train_pool.npz"
    report_path = args.triplet_dir / "report.json"
    for path in (
        actions_path,
        qualified_actions_path,
        aliases_path,
        bank_path,
        train_pool_path,
        report_path,
        args.embedding_cache,
        args.data,
        args.official_finetuned_checkpoint,
        args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    build_report = json.loads(report_path.read_text(encoding="utf-8"))
    implementation = build_report.get("implementation", {})
    if (
        implementation.get("builder_version") != "native_hard_positive_only_v16"
        or implementation.get("matched_control_version")
        != "same_query_registered_control_spectrum_v1"
        or implementation.get("action_spectrum_adapter_version")
        != NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION
    ):
        raise RuntimeError(
            "official replay requires hard-positive-only v16 artifacts"
        )
    artifacts = build_report.get("output_artifacts", {})
    if (
        artifacts.get("selected_actions_sha256") != sha256_file(actions_path)
        or artifacts.get("qualified_actions_sha256")
        != sha256_file(qualified_actions_path)
        or artifacts.get("action_aliases_sha256") != sha256_file(aliases_path)
        or artifacts.get("action_spectra_sha256") != sha256_file(bank_path)
        or artifacts.get("train_pool_sha256") != sha256_file(train_pool_path)
    ):
        raise RuntimeError("official replay inputs differ from triplet build report")
    actions = pd.read_csv(actions_path, low_memory=False)
    qualified_actions = pd.read_csv(qualified_actions_path, low_memory=False)
    aliases = pd.read_csv(aliases_path, low_memory=False)
    bank = load_npz(bank_path)
    train_pool = load_npz(train_pool_path)
    selection = build_report.get("action_selection", {})
    expected_units = int(selection.get("semantic_training_units", -1))
    if (
        len(qualified_actions) != 32114
        or len(aliases) != 32114
        or len(actions) != expected_units
        or len(bank["targeted_action_spectra"]) != len(actions)
        or len(bank["control_action_spectra"]) != len(actions)
        or aliases["qualified_action_id"].astype(str).nunique() != 32114
        or aliases["canonical_action_index"].nunique() != len(actions)
        or np.any(aliases["canonical_action_index"].to_numpy(np.int64) < 0)
        or np.any(aliases["canonical_action_index"].to_numpy(np.int64) >= len(actions))
    ):
        raise RuntimeError("official replay action provenance/canonical units drifted")
    if not np.array_equal(
        actions["action_id"].astype(str).to_numpy(), bank["action_ids"].astype(str)
    ):
        raise RuntimeError("official replay action IDs are not aligned")
    _, filler_events, production_schedule = native_query_disjoint_one_pass_batches(
        train_pool,
        np.arange(len(train_pool["event_kind"]), dtype=np.int64),
        batch_size=4,
        seed=3407,
    )
    action_event_indices = np.asarray(
        train_pool["event_action_index"], dtype=np.int64
    )
    schedule_exposures = np.bincount(
        action_event_indices[action_event_indices >= 0], minlength=len(actions)
    ).astype(np.int64)
    if (
        schedule_exposures.shape != (len(actions),)
        or int(schedule_exposures.sum())
        != int(production_schedule["action_events"])
    ):
        raise RuntimeError("official replay schedule weights drifted from production")
    schedule_weights = schedule_exposures.astype(np.float64)
    schedule_exposure_sha256 = hashlib.sha256(
        np.ascontiguousarray(schedule_exposures, dtype=np.int64).tobytes()
    ).hexdigest()

    selection_cache = load_npz(args.embedding_cache)
    model = load_trusted_native_contrastive_head(
        args.official_finetuned_checkpoint,
        architecture_checkpoint=args.architecture_checkpoint,
        lr=5e-6,
        weight_decay=0.0,
        triplet_loss_margin=0.1,
        map_location=torch.device("cpu"), strict=True,
    ).to(torch.device("cuda"))
    training_preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    selection_rows = np.asarray(selection_cache["rows"], dtype=np.int64)
    native_rows = selection_rows.copy()
    native_embeddings = encode_measured_rows(
        model,
        native_rows,
        args.data,
        batch_size=args.batch_size,
        device=torch.device("cuda"),
        preprocessor=training_preprocessor,
        label="native-official-measured-cache",
    )
    native_cache = {
        "rows": native_rows,
        "embeddings": native_embeddings,
    }
    geometry_diagnostic = compare_selection_and_native_geometry(
        selection_cache, native_rows, native_embeddings
    )
    # The legacy values have now served their only legitimate role: a labelled
    # diagnostic of the action-mining geometry. Release them before replaying
    # actions; every training-role embedding below comes from ``native_cache``.
    del selection_cache
    if "native_action_view_representable" not in actions:
        raise RuntimeError("selected actions lack native representability routing")
    action_capable = actions["native_action_view_representable"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if action_capable.isna().any():
        raise RuntimeError("native representability routing is not boolean")
    action_capable = action_capable.to_numpy(bool)
    bank_action_capable = np.asarray(
        bank.get("native_action_view_representable", np.empty(0)), dtype=bool
    )
    if not np.array_equal(bank_action_capable, action_capable):
        raise RuntimeError("action archive representability mask drifted")
    capable_actions = actions.loc[action_capable].reset_index(drop=True)
    targeted_bank = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_bank = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    targeted = encode_actions(
        model,
        targeted_bank[action_capable],
        batch_size=args.batch_size,
        device=torch.device("cuda"),
        label="targeted",
        preprocessor=training_preprocessor,
    )
    control = encode_actions(
        model,
        control_bank[action_capable],
        batch_size=args.batch_size,
        device=torch.device("cuda"),
        label="matched-control",
        preprocessor=training_preprocessor,
    )
    positive = row_embeddings(
        native_cache, capable_actions["action_positive_row"].to_numpy(np.int64)
    )
    negative = row_embeddings(
        native_cache, capable_actions["action_hard_negative_row"].to_numpy(np.int64)
    )
    clean = row_embeddings(
        native_cache, capable_actions["query_row"].to_numpy(np.int64)
    )
    clean_all = row_embeddings(
        native_cache, actions["query_row"].to_numpy(np.int64)
    )
    positive_all = row_embeddings(
        native_cache, actions["action_positive_row"].to_numpy(np.int64)
    )
    negative_all = row_embeddings(
        native_cache, actions["action_hard_negative_row"].to_numpy(np.int64)
    )
    # Bridge 1 keeps the deployed clean query on the action-specific boundary.
    targeted_deployment_margin = np.einsum("ij,ij->i", clean, targeted) - np.einsum(
        "ij,ij->i", clean, negative
    )
    control_deployment_margin = np.einsum("ij,ij->i", clean, control) - np.einsum(
        "ij,ij->i", clean, negative
    )
    # Bridge 2 is the hard-positive relation: modified same-query view to an
    # independently measured true spectrum, against the same exact negative.
    targeted_hard_positive_margin = np.einsum(
        "ij,ij->i", targeted, positive
    ) - np.einsum("ij,ij->i", targeted, negative)
    control_hard_positive_margin = np.einsum(
        "ij,ij->i", control, positive
    ) - np.einsum("ij,ij->i", control, negative)
    clean_margin = np.einsum("ij,ij->i", clean, positive) - np.einsum(
        "ij,ij->i", clean, negative
    )
    clean_margin_all = np.einsum("ij,ij->i", clean_all, positive_all) - np.einsum(
        "ij,ij->i", clean_all, negative_all
    )
    # The action-discovery boundary is retained as a diagnostic only.  It is
    # not substituted for the actual triplet optimized by production.
    deployment_paired_advantage = (
        targeted_deployment_margin - control_deployment_margin
    )
    hard_positive_paired_advantage = (
        targeted_hard_positive_margin - control_hard_positive_margin
    )
    targeted_clean_change = targeted_hard_positive_margin - clean_margin
    clean_identity_hinge = np.maximum(0.1 - clean_margin, 0.0)
    clean_boundary_hinge = np.maximum(0.1 - clean_margin_all, 0.0)
    targeted_deployment_hinge = np.maximum(
        0.1 - targeted_deployment_margin, 0.0
    )
    control_deployment_hinge = np.maximum(
        0.1 - control_deployment_margin, 0.0
    )
    targeted_hard_positive_hinge = np.maximum(
        0.1 - targeted_hard_positive_margin, 0.0
    )
    control_hard_positive_hinge = np.maximum(
        0.1 - control_hard_positive_margin, 0.0
    )
    action_schedule_weights = schedule_weights[action_capable]
    alias_source_memberships = [
        set(value.split("|"))
        for value in actions["qualified_alias_sources"].astype(str)
    ]
    alias_source_unit_counts = {
        source: int(sum(source in memberships for memberships in alias_source_memberships))
        for source in sorted(set().union(*alias_source_memberships))
    }
    representative_source = actions["source"].astype(str).to_numpy()
    capable_source = capable_actions["source"].astype(str).to_numpy()
    source_diagnostics: dict[str, dict[str, float | int | bool]] = {}
    for source in sorted(np.unique(representative_source)):
        # The registered control belongs to the same query as its targeted
        # action. Cross-source aliases remain provenance-only.
        clean_mask = representative_source == source
        action_mask = capable_source == source
        source_weights = action_schedule_weights[action_mask]
        has_action_view = bool(np.any(action_mask))
        body = {
            "effective_units": int(np.sum(clean_mask)),
            "native_action_view_units": int(np.sum(action_mask)),
            "scheduled_action_opportunities": int(np.sum(source_weights)),
            "schedule_weighted_clean_query_relation_active_fraction": weighted_mean(
                clean_boundary_hinge[clean_mask] > 0, schedule_weights[clean_mask]
            ),
            "diagnostic_unoptimized_clean_to_action_hinge_active_fraction": weighted_mean(
                targeted_deployment_hinge[action_mask] > 0, source_weights
            ) if has_action_view else None,
            "schedule_weighted_hard_positive_bridge_active_fraction": weighted_mean(
                targeted_hard_positive_hinge[action_mask] > 0, source_weights
            ) if has_action_view else None,
            "diagnostic_unoptimized_targeted_minus_control_clean_to_action_margin": weighted_mean(
                deployment_paired_advantage[action_mask], source_weights
            ) if has_action_view else None,
            "schedule_weighted_targeted_minus_control_hard_positive_margin": weighted_mean(
                hard_positive_paired_advantage[action_mask], source_weights
            ) if has_action_view else None,
            "schedule_weighted_discovery_targeted_minus_clean_margin": weighted_mean(
                targeted_clean_change[action_mask], source_weights
            ) if has_action_view else None,
        }
        body["has_native_hinge_signal"] = bool(
            has_action_view
            and body["schedule_weighted_hard_positive_bridge_active_fraction"] > 0
        )
        source_diagnostics[source] = body
    gradient_audit = actual_action_gradient_audit(
        model,
        capable_actions,
        targeted_bank[action_capable],
        clean_identity_hinge,
        targeted_hard_positive_hinge,
        data=args.data,
        preprocessor=training_preprocessor,
        device=torch.device("cuda"),
    )
    del model
    gates = {
        "all_32114_qualified_rows_preserved_in_alias_ledger": (
            len(qualified_actions) == len(aliases) == 32114
        ),
        "all_effective_training_units_replayed": (
            len(clean_margin_all) == expected_units == len(actions)
            and len(targeted_deployment_margin) == int(np.sum(action_capable))
        ),
        "all_margins_finite": bool(
            np.all(np.isfinite(targeted_deployment_margin))
            and np.all(np.isfinite(control_deployment_margin))
            and np.all(np.isfinite(targeted_hard_positive_margin))
            and np.all(np.isfinite(control_hard_positive_margin))
            and np.all(np.isfinite(clean_margin))
            and np.all(np.isfinite(clean_margin_all))
        ),
        "clean_action_measured_positive_negative_share_native_checkpoint_geometry": True,
        "native_official_cache_covers_complete_selection_registry": bool(
            len(native_rows) == len(selection_rows)
            and np.array_equal(native_rows, selection_rows)
        ),
        "clean_only_units_excluded_from_action_effect_denominator": bool(
            len(targeted_deployment_margin) == int(np.sum(action_capable))
        ),
        "clean_and_hard_positive_native_paths_have_hinge_signal": bool(
            np.any(clean_identity_hinge > 0)
            and np.any(targeted_hard_positive_hinge > 0)
        ),
        "every_effective_action_is_exposed_exactly_once": bool(
            production_schedule["minimum_action_events_per_semantic_unit"] == 1
            and production_schedule["maximum_action_events_per_semantic_unit"] == 1
        ),
        "harmful_clean_to_action_training_path_is_absent": bool(
            production_schedule["forbidden_clean_to_action_events"] == 0
        ),
        "synthetic_query_equalization_is_absent": bool(
            production_schedule["synthetic_query_equalization_events"] == 0
            and len(filler_events) < 4
        ),
        "optimizer_step_budget_is_bounded": bool(
            0 < production_schedule["batches_per_epoch"]
            <= NATIVE_MAX_OPTIMIZER_STEPS
        ),
        "targeted_hard_positive_relation_beats_same_query_control": bool(
            weighted_mean(
                hard_positive_paired_advantage, action_schedule_weights
            ) > 0
            and weighted_mean(
                hard_positive_paired_advantage > 0, action_schedule_weights
            ) > 0.5
        ),
        "real_materialized_actions_reach_all_roles_head_and_backbone": bool(
            gradient_audit["gate_passed"]
        ),
    }
    report = {
        "status": "NOISE_DREAMS_NATIVE_OFFICIAL_REPLAY_PASS"
        if all(gates.values()) else "NOISE_DREAMS_NATIVE_OFFICIAL_REPLAY_FAIL",
        "selection_geometry": "mature_current_E8",
        "training_initialization_geometry": "official_finetuned_DreaMS",
        "qualified_action_rows": int(len(qualified_actions)),
        "semantic_training_units": int(len(actions)),
        "native_action_view_units": int(np.sum(action_capable)),
        "clean_boundary_only_units": int(np.sum(~action_capable)),
        "replay_geometry_version": NATIVE_REPLAY_GEOMETRY_VERSION,
        "initialization_adapter_version": NATIVE_LOADER_COMPAT_VERSION,
        "selection_cache_vs_native_checkpoint_diagnostic": geometry_diagnostic,
        "boundary_embedding_geometry": (
            "all clean/action/positive/negative roles encoded by the same native "
            "official ContrastiveHead before optimization"
        ),
        "exact_boundary": {
            "targeted_deployment_margin": distribution(targeted_deployment_margin),
            "control_deployment_margin": distribution(control_deployment_margin),
            "targeted_hard_positive_margin": distribution(targeted_hard_positive_margin),
            "control_hard_positive_margin": distribution(control_hard_positive_margin),
            "clean_margin_for_action_view_units": distribution(clean_margin),
            "clean_margin_all_units": distribution(clean_margin_all),
            "targeted_minus_control_deployment": distribution(
                deployment_paired_advantage
            ),
            "targeted_minus_control_hard_positive": distribution(
                hard_positive_paired_advantage
            ),
            "targeted_minus_clean": distribution(targeted_clean_change),
            "targeted_deployment_win_fraction": float(
                np.mean(deployment_paired_advantage > 0)
            ),
            "targeted_hard_positive_win_fraction": float(
                np.mean(hard_positive_paired_advantage > 0)
            ),
            "targeted_clean_improvement_fraction": float(
                np.mean(targeted_clean_change > 0)
            ),
            "clean_to_action_relation_is_diagnostic_and_not_optimized": True,
            "margin_comparisons_are_diagnostic_not_training_gates": True,
            "clean_boundary_native_hinge": distribution(clean_boundary_hinge),
            "targeted_deployment_native_hinge": distribution(
                targeted_deployment_hinge
            ),
            "control_deployment_native_hinge": distribution(
                control_deployment_hinge
            ),
            "targeted_hard_positive_native_hinge": distribution(
                targeted_hard_positive_hinge
            ),
            "control_hard_positive_native_hinge": distribution(
                control_hard_positive_hinge
            ),
        },
        "exact_native_training_schedule": {
            "weighting_version": NATIVE_SCHEDULE_VERSION,
            "epochs": NATIVE_MAX_EPOCHS,
            "seed": 3407,
            "batch_size": 4,
            "optimizer_steps_per_arm": int(
                production_schedule["batches_per_epoch"]
            ),
            "base_events": int(production_schedule["base_events"]),
            "clean_batch_padding_events": int(len(filler_events)),
            "padding_is_final_batch_only": bool(
                production_schedule["padding_is_final_batch_only"]
            ),
            "same_query_events_never_share_an_optimizer_batch": bool(
                production_schedule[
                    "same_query_events_never_share_an_optimizer_batch"
                ]
            ),
            "query_balanced_oversampling": False,
            "synthetic_query_equalization_events": int(
                production_schedule["synthetic_query_equalization_events"]
            ),
            "maximum_optimizer_steps_gate": NATIVE_MAX_OPTIMIZER_STEPS,
            "scheduled_action_opportunities": int(schedule_exposures.sum()),
            "scheduled_materialized_action_views": int(np.sum(action_capable)),
            "action_unit_exposure_sha256": schedule_exposure_sha256,
            "minimum_exposures_per_semantic_unit": int(schedule_exposures.min()),
            "maximum_exposures_per_semantic_unit": int(schedule_exposures.max()),
            "forbidden_clean_to_action_events": int(
                production_schedule["forbidden_clean_to_action_events"]
            ),
            "targeted_minus_control_deployment_mean_margin": weighted_mean(
                deployment_paired_advantage, action_schedule_weights
            ),
            "targeted_deployment_win_fraction": weighted_mean(
                deployment_paired_advantage > 0, action_schedule_weights
            ),
            "targeted_minus_control_hard_positive_mean_margin": weighted_mean(
                hard_positive_paired_advantage, action_schedule_weights
            ),
            "targeted_hard_positive_win_fraction": weighted_mean(
                hard_positive_paired_advantage > 0, action_schedule_weights
            ),
            "targeted_minus_clean_mean_margin": weighted_mean(
                targeted_clean_change, action_schedule_weights
            ),
            "targeted_clean_improvement_fraction": weighted_mean(
                targeted_clean_change > 0, action_schedule_weights
            ),
            "clean_boundary_native_hinge_active_fraction": weighted_mean(
                clean_boundary_hinge > 0, schedule_weights
            ),
            "targeted_deployment_native_hinge_active_fraction": weighted_mean(
                targeted_deployment_hinge > 0, action_schedule_weights
            ),
            "targeted_hard_positive_native_hinge_active_fraction": weighted_mean(
                targeted_hard_positive_hinge > 0, action_schedule_weights
            ),
        },
        "real_materialized_action_gradient_audit": gradient_audit,
        "per_canonical_representative_source_native_signal": source_diagnostics,
        "qualified_alias_source_membership_unit_counts": alias_source_unit_counts,
        "gates": gates,
        "provenance": {
            "triplet_report_sha256": sha256_file(report_path),
            "selected_actions_sha256": sha256_file(actions_path),
            "qualified_actions_sha256": sha256_file(qualified_actions_path),
            "action_aliases_sha256": sha256_file(aliases_path),
            "action_spectra_sha256": sha256_file(bank_path),
            "train_pool_sha256": sha256_file(train_pool_path),
            "selection_embedding_cache_sha256": sha256_file(args.embedding_cache),
            "spectrum_data_sha256": sha256_file(args.data),
            "official_native_checkpoint_sha256": sha256_file(
                args.official_finetuned_checkpoint
            ),
            "architecture_checkpoint_sha256": sha256_file(
                args.architecture_checkpoint
            ),
        },
        "claim_limit": (
            "Pre-optimization exact-triplet replay only; this is not held encoder performance."
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"official action replay failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.native_embedding_cache_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.native_embedding_cache_output,
        rows=native_rows,
        embeddings=native_embeddings,
    )
    reloaded_native_cache = load_npz(args.native_embedding_cache_output)
    if (
        set(reloaded_native_cache) != {"rows", "embeddings"}
        or not np.array_equal(reloaded_native_cache["rows"], native_rows)
        or not np.array_equal(
            reloaded_native_cache["embeddings"], native_embeddings
        )
    ):
        raise RuntimeError("native official embedding cache roundtrip drifted")
    report["provenance"]["native_official_embedding_cache_sha256"] = (
        sha256_file(args.native_embedding_cache_output)
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

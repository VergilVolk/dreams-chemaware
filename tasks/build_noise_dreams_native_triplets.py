"""Materialize Noise actions as native DreaMS contrastive triplets.

This is the only Noise-specific training boundary.  The output contains
ordinary anchor/positive/negative membership lists consumed by the repository's
``ContrastiveSpectraDataset``.  It does not define a loss, model, optimizer,
gradient injector, teacher target, or candidate-graph objective.

Every qualified seven-source row is retained in provenance. Rows that produce
the identical native model input at the identical positive/negative boundary
share one optimizer unit, so rediscovery cannot silently multiply dose. Every
distinct representable unit contributes exactly one native hard-positive
triplet: action-to-independent-measurement recognition. Unrepresentable units contribute one
measured fallback. Clean anchors remain an identity-preserving base. A
formula-disjoint validation split is drawn only from formulas with no selected
action so that no distinct action is withheld from optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import pandas as pd

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_dreams_native_spectrum import (
    NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION,
    action_fragment_profile,
    native_action_model_input as adapt_native_action_model_input,
)

NATIVE_TRIPLET_BUILDER_VERSION = "native_hard_positive_only_v16"
MATCHED_CONTROL_VERSION = "same_query_registered_control_spectrum_v1"

NATIVE_ACTION_PREPROCESSOR = SpectrumPreprocessor(
    DataFormatA(),
    prec_intens=1.1,
    n_highest_peaks=100,
    spec_entropy_cleaning=False,
    precision=32,
    mz_shift_aug_p=0,
    mz_shift_aug_max=0,
)

REGISTERED_SOURCES = frozenset({
    "N_mature",
    "P_guided_original",
    "E10B",
    "E11",
    "E12B",
    "A4_exact",
    "V4_gradient_path",
})
REGISTERED_FORMAL_INPUT_SHA256 = {
    "ledger_report_sha256": (
        "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349"
    ),
    "training_actions_sha256": (
        "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa"
    ),
    "action_spectra_sha256": (
        "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a"
    ),
    "candidate_graph_sha256": (
        "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1"
    ),
    "embedding_cache_sha256": (
        "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda"
    ),
}
REGISTERED_FORMAL_COUNTS = {
    "graph_queries": 83619,
    "ledger_rows": 327678,
    "corrective_rows": 62430,
    "strict_corrective_rows_before_floor": 32127,
    "strict_corrective_rows": 32114,
    "strict_corrective_queries": 3482,
}
REQUIRED_ACTION_COLUMNS = {
    "action_id",
    "query_index",
    "query_row",
    "query_ik14",
    "query_formula",
    "formula_fold",
    "source",
    "family",
    "recipe_id",
    "supervision_kind",
    "clean_rank",
    "action_rank",
    "action_margin",
    "action_tensor_index",
    "action_positive_row",
    "action_hard_negative_row",
    "control_positive_row",
    "control_hard_negative_row",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--validation-folds", type=int, default=10)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=20260920)
    parser.add_argument("--shuffle-seed", type=int, default=20260920)
    parser.add_argument("--margin-floor", type=float, default=5e-6)
    parser.add_argument("--hard-negative-molecules", type=int, default=5)
    parser.add_argument("--max-positive-pool", type=int, default=32)
    parser.add_argument("--max-negative-pool", type=int, default=32)
    parser.add_argument("--expected-actions", type=int, default=32114)
    parser.add_argument("--minimum-train-clean-anchors", type=int, default=50000)
    parser.add_argument("--minimum-validation-clean-anchors", type=int, default=1000)
    parser.add_argument("--minimum-train-formulas", type=int, default=4000)
    parser.add_argument(
        "--clean-anchor-scope",
        choices=("all", "action_queries_plus_formula"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def stable_fold(value: str, folds: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def fixed_unicode(values: Iterable[object]) -> np.ndarray:
    """Return a pickle-free fixed-width Unicode array for formal NPZ files."""
    text = [str(value) for value in values]
    width = max((len(value) for value in text), default=1)
    output = np.asarray(text, dtype=f"<U{max(width, 1)}")
    if output.dtype.kind != "U":
        raise RuntimeError("formal string array is not fixed-width Unicode")
    return output


def audit_pickle_free_arrays(
    arrays: Mapping[str, np.ndarray], *, label: str,
) -> None:
    object_keys = [
        key for key, value in arrays.items()
        if np.asarray(value).dtype.kind == "O"
    ]
    if object_keys:
        raise RuntimeError(
            f"{label} contains pickle-requiring object arrays: {object_keys}"
        )


def _report_allows_training(report: Mapping[str, object]) -> bool:
    contracts = report.get("contracts")
    contracts = contracts if isinstance(contracts, Mapping) else {}
    return (
        report.get("status") == "noise_corrected_routed_action_ledger_complete"
        and report.get("formal") is True
        and report.get("formal_training_authorized") is True
        and contracts.get("outer_held_formula_consumed") is False
        and contracts.get("teacher_embedding_target_used") is False
        and contracts.get("P3_consumed") is False
        and contracts.get("action_multiplicity_is_not_training_dose") is True
    )


def audit_formal_inputs(
    report: Mapping[str, object],
    *,
    report_path: Path,
    actions_path: Path,
    spectra_path: Path,
    graph_path: Path,
    embedding_cache_path: Path,
) -> dict[str, str]:
    observed = {
        "ledger_report_sha256": sha256_file(report_path),
        "training_actions_sha256": sha256_file(actions_path),
        "action_spectra_sha256": sha256_file(spectra_path),
        "candidate_graph_sha256": sha256_file(graph_path),
        "embedding_cache_sha256": sha256_file(embedding_cache_path),
    }
    drift = {
        key: {"observed": observed[key], "expected": expected}
        for key, expected in REGISTERED_FORMAL_INPUT_SHA256.items()
        if observed[key] != expected
    }
    if drift:
        raise RuntimeError(
            "formal native-triplet input drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    provenance = report.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    for key in (
        "training_actions_sha256",
        "action_spectra_sha256",
        "candidate_graph_sha256",
    ):
        if provenance.get(key) != observed[key]:
            raise RuntimeError(f"formal ledger internal provenance drifted: {key}")
    return observed


def select_actions(
    frame: pd.DataFrame,
    *,
    margin_floor: float,
    outer_fold: int,
    formula_fold_seed: int,
    expected_actions: int,
) -> pd.DataFrame:
    missing = REQUIRED_ACTION_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(
            "formal action ledger lacks exact native-triplet fields: "
            f"{sorted(missing)}"
        )
    unknown = set(frame["source"].astype(str)) - REGISTERED_SOURCES
    if unknown:
        raise RuntimeError(f"unregistered action sources: {sorted(unknown)}")
    formal_count_gate = int(expected_actions) == REGISTERED_FORMAL_COUNTS[
        "strict_corrective_rows"
    ]
    if formal_count_gate:
        if len(frame) != REGISTERED_FORMAL_COUNTS["ledger_rows"]:
            raise RuntimeError(
                f"formal ledger rows drifted: {len(frame)} != "
                f"{REGISTERED_FORMAL_COUNTS['ledger_rows']}"
            )
        if int(frame["supervision_kind"].astype(str).eq("corrective").sum()) != (
            REGISTERED_FORMAL_COUNTS["corrective_rows"]
        ):
            raise RuntimeError("formal corrective-row count drifted")
    clean_rank = pd.to_numeric(frame["clean_rank"], errors="raise")
    action_rank = pd.to_numeric(frame["action_rank"], errors="raise")
    action_margin = pd.to_numeric(frame["action_margin"], errors="raise")
    strict_before_floor = frame.loc[
        frame["supervision_kind"].astype(str).eq("corrective")
        & clean_rank.ne(1)
        & action_rank.eq(1)
    ].copy()
    if formal_count_gate and len(strict_before_floor) != REGISTERED_FORMAL_COUNTS[
        "strict_corrective_rows_before_floor"
    ]:
        raise RuntimeError("strict corrective count before numerical floor drifted")
    selected = strict_before_floor.loc[
        action_margin.loc[strict_before_floor.index].gt(float(margin_floor))
    ].copy()
    selected = selected.sort_values(
        ["query_index", "source", "family", "recipe_id", "action_id"],
        kind="stable",
    ).reset_index(drop=True)
    if len(selected) != int(expected_actions):
        raise RuntimeError(
            f"strict seven-source action count drifted: {len(selected)} != "
            f"{expected_actions}"
        )
    if selected["action_id"].astype(str).duplicated().any():
        raise RuntimeError("strict action panel duplicates action_id")
    if formal_count_gate and selected["query_index"].nunique() != REGISTERED_FORMAL_COUNTS[
        "strict_corrective_queries"
    ]:
        raise RuntimeError("strict corrective query count drifted")
    if set(selected["source"].astype(str)) != REGISTERED_SOURCES:
        raise RuntimeError("strict action panel lost a registered mature source")
    expected_folds = selected["query_formula"].astype(str).map(
        lambda value: stable_fold(value, 5, formula_fold_seed)
    )
    observed_folds = pd.to_numeric(selected["formula_fold"], errors="raise").astype(int)
    if not np.array_equal(expected_folds.to_numpy(int), observed_folds.to_numpy(int)):
        raise RuntimeError("action ledger formula folds do not replay")
    if np.any(observed_folds.to_numpy(int) == int(outer_fold)):
        raise RuntimeError("outer-held query action reached native triplet mining")
    return selected


def align_action_tensor_archive(
    actions: pd.DataFrame,
    tensor_bank: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    required = {"action_ids", "action_spectra", "control_spectra"}
    if set(tensor_bank) != required:
        raise RuntimeError(
            f"action tensor archive keys {sorted(tensor_bank)} != {sorted(required)}"
        )
    ids = np.asarray(tensor_bank["action_ids"]).astype(str)
    targeted = np.asarray(tensor_bank["action_spectra"], dtype=np.float32)
    control = np.asarray(tensor_bank["control_spectra"], dtype=np.float32)
    if (
        len(ids) != len(targeted)
        or targeted.shape != control.shape
        or targeted.ndim != 3
        or targeted.shape[1:] != (101, 2)
        or not np.isfinite(targeted).all()
        or not np.isfinite(control).all()
    ):
        raise RuntimeError("action/control tensor archive has invalid geometry")
    position = {value: index for index, value in enumerate(ids)}
    if len(position) != len(ids):
        raise RuntimeError("action tensor archive duplicates action IDs")
    try:
        take = np.asarray(
            [position[value] for value in actions["action_id"].astype(str)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(f"selected action is absent from tensor archive: {error}") from error
    targeted = np.ascontiguousarray(targeted[take])
    control = np.ascontiguousarray(control[take])
    return targeted, control


def align_action_tensors(
    actions: pd.DataFrame,
    tensor_bank: Mapping[str, np.ndarray],
    *,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return the registered same-query targeted/control action views.

    ``shuffle_seed`` remains in the compatibility signature for callers from
    the preceding builder.  It has no effect: a cross-query tensor permutation
    is not a valid same-identity positive control.
    """
    del shuffle_seed
    targeted, control = align_action_tensor_archive(actions, tensor_bank)
    distinct = np.asarray([
        not np.array_equal(targeted[index], control[index])
        for index in range(len(actions))
    ], dtype=bool)
    if not np.any(distinct):
        raise RuntimeError("all registered matched controls equal targeted actions")
    return targeted, np.ascontiguousarray(control), {
        "version": MATCHED_CONTROL_VERSION,
        "rows": int(len(actions)),
        "same_query_by_construction": True,
        "cross_query_tensor_donors": 0,
        "targeted_control_distinct_rows": int(np.sum(distinct)),
        "targeted_control_distinct_fraction": float(np.mean(distinct)),
    }


def native_action_model_input(tensor: np.ndarray) -> np.ndarray:
    """Replay the exact deterministic native action preprocessing path."""
    return adapt_native_action_model_input(tensor, NATIVE_ACTION_PREPROCESSOR)


def native_action_semantic_sha256(tensor: np.ndarray) -> str:
    """Hash encoder tokens, or raw content when native-unrepresentable."""
    profile = action_fragment_profile(tensor)
    digest = hashlib.sha256()
    if profile["native_action_view_representable"]:
        tokens = native_action_model_input(tensor)
        digest.update(b"native_encoder_tokens_v1|")
    else:
        tokens = np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))
        digest.update(b"native_unrepresentable_raw_action_v1|")
    digest.update(np.asarray(tokens.shape, dtype=np.int64).tobytes())
    digest.update(tokens.tobytes())
    return digest.hexdigest()


def audit_native_action_bank(tensors: np.ndarray) -> dict[str, int | float | bool]:
    unrepresentable = 0
    maximum_real_tokens_in_unrepresentable_action = 0
    maximum_real_token_error = 0.0
    for tensor in np.asarray(tensors, dtype=np.float32):
        profile = action_fragment_profile(tensor)
        if not profile["native_action_view_representable"]:
            unrepresentable += 1
            maximum_real_tokens_in_unrepresentable_action = max(
                maximum_real_tokens_in_unrepresentable_action,
                int(profile["real_fragment_tokens"]),
            )
            continue
        replay = native_action_model_input(tensor)
        source_tokens = tensor[tensor[:, 0] > 0].copy()
        replay_tokens = replay[replay[:, 0] > 0].copy()
        if source_tokens.shape != replay_tokens.shape:
            raise RuntimeError("native action preflight lost a real model token")
        maximum_real_token_error = max(
            maximum_real_token_error,
            float(np.max(np.abs(source_tokens - replay_tokens))),
        )
    return {
        "actions": int(len(tensors)),
        "native_action_view_representable_actions": int(len(tensors) - unrepresentable),
        "all_zero_fragment_intensity_actions": int(unrepresentable),
        "maximum_real_fragment_tokens_in_all_zero_action": int(
            maximum_real_tokens_in_unrepresentable_action
        ),
        "maximum_real_token_roundtrip_abs_error": float(maximum_real_token_error),
        "all_representable_actions_roundtrip_exactly": bool(
            maximum_real_token_error == 0.0
        ),
        "unmodified_native_preprocessor": True,
        "all_zero_actions_require_clean_boundary_only": True,
    }


def audit_same_identity_action_views(
    actions: pd.DataFrame,
    targeted: np.ndarray,
    control: np.ndarray,
    *,
    data: Path,
) -> dict[str, object]:
    """Audit frozen query provenance without erasing counterfactual payload.

    Candidate-gradient actions may attenuate/remove measured peaks, while the
    registered P/A4/V4 actions may graft or shift peaks selected from the true
    identity/candidate boundary.  Those introduced peaks are the Noise signal,
    not evidence that the action belongs to another molecule.  Identity is
    bound by the frozen action-id -> query-row ledger, exact query precursor,
    and the graph-verified positive/negative relation.  Cross-query tensor
    shuffling remains forbidden by ``align_action_tensors``.
    """
    if len(actions) != len(targeted) or targeted.shape != control.shape:
        raise RuntimeError("identity-preservation audit is not tensor aligned")
    query_rows = actions["query_row"].to_numpy(np.int64)
    clean_by_row: dict[int, np.ndarray] = {}
    with h5py.File(data, "r") as handle:
        n_rows = len(handle["spectrum"])
        for row in np.unique(query_rows):
            if row < 0 or row >= n_rows:
                raise RuntimeError("action query row is outside the HDF5 spectrum table")
            raw = np.asarray(handle["spectrum"][int(row)], dtype=np.float32)
            keep = (raw[0] > 0) & (raw[1] > 0)
            if not np.any(keep):
                raise RuntimeError("action query has no measured fragment peaks")
            clean_by_row[int(row)] = NATIVE_ACTION_PREPROCESSOR(
                raw[:, keep],
                prec_mz=float(handle["precursor_mz"][int(row)]),
                high_form=False,
            )

    def inspect(label: str, bank: np.ndarray) -> dict[str, object]:
        precursor_mismatches = 0
        introduced_fragment_rows = 0
        changed_rows = 0
        representable_rows = 0
        maximum_introduced_fragments = 0
        for index, tensor in enumerate(np.asarray(bank, dtype=np.float32)):
            clean = clean_by_row[int(query_rows[index])]
            if not np.array_equal(tensor[0], clean[0]):
                precursor_mismatches += 1
            clean_mz = set(map(float, clean[1:, 0][clean[1:, 0] > 0]))
            action_mz = set(map(float, tensor[1:, 0][tensor[1:, 0] > 0]))
            introduced = action_mz - clean_mz
            if introduced:
                introduced_fragment_rows += 1
                maximum_introduced_fragments = max(
                    maximum_introduced_fragments, len(introduced)
                )
            changed_rows += int(not np.array_equal(tensor, clean))
            representable_rows += int(
                action_fragment_profile(tensor)["native_action_view_representable"]
            )
        report = {
            "label": label,
            "rows": int(len(bank)),
            "precursor_mismatches": int(precursor_mismatches),
            "rows_with_introduced_or_shifted_fragment_mz": int(
                introduced_fragment_rows
            ),
            "maximum_introduced_or_shifted_fragments_per_row": int(
                maximum_introduced_fragments
            ),
            "changed_from_clean_rows": int(changed_rows),
            "native_representable_rows": int(representable_rows),
            "counterfactual_fragment_payload_is_allowed": True,
            "query_provenance_gate": bool(precursor_mismatches == 0),
        }
        if not report["query_provenance_gate"]:
            raise RuntimeError(
                f"{label} action bank violates frozen query provenance: {report}"
            )
        return report

    targeted_report = inspect("targeted", targeted)
    control_report = inspect("matched_control", control)
    return {
        "targeted": targeted_report,
        "matched_control": control_report,
        "both_arms_keep_registered_query_provenance": True,
        "cross_query_tensor_positive_control_forbidden": True,
        "introduced_counterfactual_peaks_are_not_treated_as_identity_drift": True,
    }


def canonicalize_semantic_action_aliases(
    actions: pd.DataFrame,
    targeted: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Collapse exact optimizer aliases while retaining all provenance rows."""
    if len(actions) != len(targeted):
        raise RuntimeError("semantic alias audit is not action-tensor aligned")
    keys = [
        (
            int(actions.at[index, "query_index"]),
            int(actions.at[index, "action_positive_row"]),
            int(actions.at[index, "action_hard_negative_row"]),
            native_action_semantic_sha256(targeted[index]),
        )
        for index in range(len(actions))
    ]
    members: dict[tuple[int, int, int, str], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        members[key].append(index)
    aliased = [indices for indices in members.values() if len(indices) > 1]
    source_sets = Counter(
        "+".join(sorted(set(actions.iloc[indices].source.astype(str))))
        for indices in aliased
    )
    duplicate_alias_rows = int(sum(len(indices) - 1 for indices in aliased))
    profiles = [action_fragment_profile(tensor) for tensor in targeted]
    unrepresentable_profiles = [
        profile for profile in profiles
        if not profile["native_action_view_representable"]
    ]
    representatives: list[int] = []
    representative_by_key: dict[tuple[int, int, int, str], int] = {}
    for key, indices in members.items():
        representative = min(
            indices,
            key=lambda index: hashlib.sha256(
                (
                    "native_semantic_alias_representative_v1|"
                    + str(actions.at[index, "action_id"])
                ).encode("utf-8")
            ).digest(),
        )
        representatives.append(int(representative))
        representative_by_key[key] = int(representative)
    representatives.sort()
    canonical_position = {
        representative: position
        for position, representative in enumerate(representatives)
    }
    canonical = actions.iloc[representatives].copy().reset_index(drop=True)
    canonical_targeted = np.ascontiguousarray(targeted[representatives])
    alias_count = []
    alias_sources = []
    alias_families = []
    for representative in representatives:
        key = keys[representative]
        indices = members[key]
        alias_count.append(len(indices))
        alias_sources.append("|".join(sorted(set(actions.iloc[indices].source.astype(str)))))
        alias_families.append("|".join(sorted(set(actions.iloc[indices].family.astype(str)))))
    canonical["qualified_alias_count"] = np.asarray(alias_count, dtype=np.int64)
    canonical["qualified_alias_sources"] = alias_sources
    canonical["qualified_alias_families"] = alias_families
    canonical["qualified_action_tensor_index"] = canonical[
        "action_tensor_index"
    ].to_numpy(np.int64)
    canonical["action_tensor_index"] = np.arange(len(canonical), dtype=np.int64)

    alias_rows = []
    for index, key in enumerate(keys):
        representative = representative_by_key[key]
        canonical_index = canonical_position[representative]
        alias_rows.append({
            "qualified_action_index": int(index),
            "qualified_action_id": str(actions.at[index, "action_id"]),
            "qualified_source": str(actions.at[index, "source"]),
            "qualified_family": str(actions.at[index, "family"]),
            "qualified_recipe_id": str(actions.at[index, "recipe_id"]),
            "canonical_action_index": int(canonical_index),
            "canonical_action_id": str(actions.at[representative, "action_id"]),
            "is_canonical_representative": bool(index == representative),
            "semantic_key_sha256": hashlib.sha256(
                "|".join(map(str, key)).encode("utf-8")
            ).hexdigest(),
        })
    aliases = pd.DataFrame(alias_rows)
    report = {
        "qualified_provenance_rows": int(len(actions)),
        "unique_semantic_training_units": int(len(canonical)),
        "duplicate_alias_rows": duplicate_alias_rows,
        "all_zero_fragment_intensity_rows": int(len(unrepresentable_profiles)),
        "maximum_real_fragment_tokens_in_all_zero_row": int(max(
            (profile["real_fragment_tokens"] for profile in unrepresentable_profiles),
            default=0,
        )),
        "all_zero_rows_native_action_view_representable": False,
        "aliased_relation_groups": int(len(aliased)),
        "maximum_aliases_per_relation": int(
            max(map(len, members.values()), default=0)
        ),
        "queries_with_aliased_relations": int(len({
            int(actions.at[indices[0], "query_index"]) for indices in aliased
        })),
        "aliased_relation_source_sets": dict(sorted(source_sets.items())),
        "all_qualified_rows_mapped_once": bool(
            len(aliases) == len(actions)
            and aliases["qualified_action_id"].nunique() == len(actions)
        ),
        "canonical_relations_are_unique": bool(len(canonical) == len(members)),
        "representative_policy": (
            "minimum_sha256(native_semantic_alias_representative_v1|action_id)"
        ),
        "canonical_control_policy": (
            "The same outcome-blind representative supplies source/family/recipe "
            "and its registered same-query paired control."
        ),
        "interpretation": (
            "All qualified source rows remain in the alias ledger, while exact "
            "query/positive/negative/action-tensor aliases contribute one native "
            "optimizer unit and therefore cannot gain dose from rediscovery."
        ),
    }
    return canonical, canonical_targeted, aliases, report


def canonicalize_effective_native_triplets(
    actions: pd.DataFrame,
    targeted: np.ndarray,
    control: np.ndarray,
    aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Canonicalize by the triplets that can actually reach native DreaMS.

    A materialized action view is admitted only when both causal arms are
    representable by the unmodified native preprocessor. Otherwise both arms
    receive the same clean-boundary triplet, whose key intentionally ignores
    the invisible action tensor so multiplicity cannot become optimizer dose.
    """
    if len(actions) != len(targeted) or len(actions) != len(control):
        raise RuntimeError("effective native canonicalization is not tensor aligned")
    targeted_ok = np.asarray([
        action_fragment_profile(value)["native_action_view_representable"]
        for value in targeted
    ], dtype=bool)
    control_ok = np.asarray([
        action_fragment_profile(value)["native_action_view_representable"]
        for value in control
    ], dtype=bool)
    action_capable = targeted_ok & control_ok
    keys: list[tuple[object, ...]] = []
    for index in range(len(actions)):
        boundary = (
            int(actions.at[index, "query_index"]),
            int(actions.at[index, "action_positive_row"]),
            int(actions.at[index, "action_hard_negative_row"]),
        )
        if action_capable[index]:
            keys.append(("clean_and_action", *boundary,
                         native_action_semantic_sha256(targeted[index])))
        else:
            keys.append(("clean_boundary_only", *boundary))
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    representatives = sorted(
        min(
            indices,
            key=lambda index: hashlib.sha256(
                ("effective_native_triplet_v1|" + str(actions.at[index, "action_id"]))
                .encode("utf-8")
            ).digest(),
        )
        for indices in groups.values()
    )
    old_to_new: dict[int, int] = {}
    representative_for_old: dict[int, int] = {}
    for new_index, representative in enumerate(representatives):
        for old_index in groups[keys[representative]]:
            old_to_new[int(old_index)] = int(new_index)
            representative_for_old[int(old_index)] = int(representative)
    canonical = actions.iloc[representatives].copy().reset_index(drop=True)
    canonical_targeted = np.ascontiguousarray(targeted[representatives])
    canonical_control = np.ascontiguousarray(control[representatives])
    canonical["native_action_view_representable"] = action_capable[representatives]
    canonical["action_tensor_index"] = np.arange(len(canonical), dtype=np.int64)

    remapped = aliases.copy()
    old_indices = remapped["canonical_action_index"].to_numpy(np.int64)
    remapped["canonical_action_index"] = np.asarray(
        [old_to_new[int(index)] for index in old_indices], dtype=np.int64
    )
    new_ids = []
    new_representative = []
    semantic_hashes = []
    for old_index in old_indices:
        representative = representative_for_old[int(old_index)]
        new_ids.append(str(actions.at[representative, "action_id"]))
        new_representative.append(
            str(remapped.iloc[len(new_ids) - 1]["qualified_action_id"])
            == str(actions.at[representative, "action_id"])
        )
        semantic_hashes.append(hashlib.sha256(
            "|".join(map(str, keys[representative])).encode("utf-8")
        ).hexdigest())
    remapped["canonical_action_id"] = new_ids
    remapped["is_canonical_representative"] = np.asarray(new_representative, dtype=bool)
    remapped["semantic_key_sha256"] = semantic_hashes

    alias_counts = remapped.groupby("canonical_action_index").size()
    alias_sources = remapped.groupby("canonical_action_index")["qualified_source"].apply(
        lambda values: "|".join(sorted(set(map(str, values))))
    )
    alias_families = remapped.groupby("canonical_action_index")["qualified_family"].apply(
        lambda values: "|".join(sorted(set(map(str, values))))
    )
    canonical["qualified_alias_count"] = np.asarray([
        int(alias_counts.loc[index]) for index in range(len(canonical))
    ], dtype=np.int64)
    canonical["qualified_alias_sources"] = [
        str(alias_sources.loc[index]) for index in range(len(canonical))
    ]
    canonical["qualified_alias_families"] = [
        str(alias_families.loc[index]) for index in range(len(canonical))
    ]
    return canonical, canonical_targeted, canonical_control, remapped, {
        "input_semantic_units": int(len(actions)),
        "effective_native_training_units": int(len(canonical)),
        "action_view_units": int(canonical["native_action_view_representable"].sum()),
        "clean_boundary_only_units": int(
            (~canonical["native_action_view_representable"]).sum()
        ),
        "clean_only_duplicates_removed": int(len(actions) - len(canonical)),
        "targeted_and_same_query_control_must_be_native_representable": True,
        "clean_only_key_ignores_invisible_action_tensor": True,
    }


class Registry:
    """Deduplicated spectrum registry shared by native anchor memberships."""

    HDF5 = 0
    ACTION = 1
    HDF5_CLONE = 2

    def __init__(self) -> None:
        self._position: dict[tuple[int, int], int] = {}
        self.kind: list[int] = []
        self.source_index: list[int] = []

    def add(self, kind: int, source_index: int) -> int:
        key = (int(kind), int(source_index))
        if key not in self._position:
            self._position[key] = len(self.kind)
            self.kind.append(key[0])
            self.source_index.append(key[1])
        return self._position[key]

    def add_hdf5_clone(self, source_index: int) -> int:
        """Add an event-local copy so one clean spectrum can own another pool."""
        position = len(self.kind)
        self.kind.append(self.HDF5_CLONE)
        self.source_index.append(int(source_index))
        return position

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.kind, dtype=np.int8),
            np.asarray(self.source_index, dtype=np.int64),
        )


def query_molecules(
    graph: Mapping[str, np.ndarray], query: int,
) -> list[tuple[int, bool, np.ndarray]]:
    left, right = map(int, graph["query_ptr"][query:query + 2])
    result: list[tuple[int, bool, np.ndarray]] = []
    for molecule in range(left, right):
        pair_left, pair_right = map(
            int, graph["molecule_ptr"][molecule:molecule + 2]
        )
        result.append((
            molecule,
            bool(graph["molecule_label"][molecule]),
            np.asarray(
                graph["pair_candidate_row"][pair_left:pair_right], dtype=np.int64
            ),
        ))
    return result


def locate_candidate_row(
    molecules: Iterable[tuple[int, bool, np.ndarray]], row: int,
) -> tuple[int, bool, np.ndarray]:
    matches = [body for body in molecules if int(row) in set(map(int, body[2]))]
    if len(matches) != 1:
        raise RuntimeError(f"candidate row {row} maps to {len(matches)} molecules")
    return matches[0]


def embedding_positions(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = np.asarray(cache["rows"], dtype=np.int64)
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    if (
        len(rows) != len(embeddings)
        or embeddings.ndim != 2
        or len(np.unique(rows)) != len(rows)
        or not np.isfinite(embeddings).all()
    ):
        raise RuntimeError("official embedding cache is invalid")
    positions = np.full(int(rows.max()) + 1, -1, dtype=np.int64)
    positions[rows] = np.arange(len(rows), dtype=np.int64)
    return positions


def hard_negative_rows(
    graph: Mapping[str, np.ndarray],
    query: int,
    cache: Mapping[str, np.ndarray],
    row_position: np.ndarray,
    *,
    maximum_molecules: int,
) -> list[int]:
    query_row = int(graph["query_row"][query])
    if query_row >= len(row_position) or row_position[query_row] < 0:
        raise RuntimeError(f"query row {query_row} is absent from embedding cache")
    query_embedding = np.asarray(
        cache["embeddings"][row_position[query_row]], dtype=np.float32
    )
    ranked: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for molecule, label, rows in query_molecules(graph, query):
        if label:
            continue
        if np.any(rows >= len(row_position)) or np.any(row_position[rows] < 0):
            raise RuntimeError("candidate row is absent from official embedding cache")
        scores = np.asarray(cache["embeddings"][row_position[rows]]) @ query_embedding
        ranked.append((float(np.max(scores)), int(molecule), rows, scores))
    if not ranked:
        raise RuntimeError(f"query {query} has no negative candidate molecule")
    ranked.sort(key=lambda body: (-body[0], body[1]))
    output: list[int] = []
    for _, _, rows, scores in ranked[:maximum_molecules]:
        order = np.argsort(-scores, kind="stable")
        output.extend(map(int, rows[order]))
    return output


def ordered_unique(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def bounded(values: Iterable[int], maximum: int) -> list[int]:
    output = ordered_unique(values)
    if not output:
        raise RuntimeError("native triplet pool member list is empty")
    return output[: int(maximum)]


def expected_unique_draws(pool_size: np.ndarray, epochs: int) -> np.ndarray:
    pool_size = np.asarray(pool_size, dtype=np.float64)
    return pool_size * (1.0 - np.power(1.0 - 1.0 / pool_size, int(epochs)))


def pool_report(pool: Mapping[str, np.ndarray], *, epochs: int) -> dict[str, object]:
    positive_count = np.diff(pool["positive_ptr"]).astype(np.int64)
    negative_count = np.diff(pool["negative_ptr"]).astype(np.int64)
    combinations = positive_count * negative_count
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    action_mask = kinds > 0
    clean_mask = kinds == 0

    def stats(values: np.ndarray) -> dict[str, float | int]:
        return {
            "minimum": int(np.min(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": int(np.max(values)),
        }

    action_positive_count = positive_count[action_mask]
    action_negative_count = negative_count[action_mask]
    clean_positive_count = positive_count[clean_mask]
    clean_negative_count = negative_count[clean_mask]
    return {
        "anchors": int(len(pool["anchor_idx"])),
        "clean_anchors": int(np.sum(clean_mask)),
        "action_events": int(np.sum(action_mask)),
        "action_units": int(len(np.unique(
            pool["event_action_index"][action_mask]
        ))) if np.any(action_mask) else 0,
        "action_clean_boundary_triplets": int(np.sum(kinds == 1)),
        "action_measured_positive_hard_triplets": int(np.sum(kinds == 2)),
        "forbidden_clean_to_action_triplets": int(np.sum(kinds == 3)),
        "unique_queries": int(len(np.unique(pool["event_query"]))),
        "unique_formulas": int(len(np.unique(pool["event_formula"].astype(str)))),
        "positive_memberships": int(len(pool["positive_idx"])),
        "negative_memberships": int(len(pool["negative_idx"])),
        "positive_pool": stats(positive_count),
        "negative_pool": stats(negative_count),
        "clean_positive_pool": stats(clean_positive_count),
        "clean_negative_pool": stats(clean_negative_count),
        "action_positive_pool": (
            stats(action_positive_count) if len(action_positive_count) else None
        ),
        "action_negative_pool": (
            stats(action_negative_count) if len(action_negative_count) else None
        ),
        "all_action_events_are_singleton_triplets": bool(
            len(action_positive_count)
            and np.all(action_positive_count == 1)
            and np.all(action_negative_count == 1)
        ),
        "possible_triplets": int(np.sum(combinations, dtype=np.int64)),
        "possible_triplets_per_anchor": stats(combinations),
        "expected_unique_positive_members_seen_per_anchor_median": float(
            np.median(expected_unique_draws(positive_count, epochs))
        ),
        "expected_unique_negative_members_seen_per_anchor_median": float(
            np.median(expected_unique_draws(negative_count, epochs))
        ),
        "epochs_for_sampling_audit": int(epochs),
    }


def build_pools(
    graph: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    actions: pd.DataFrame,
    *,
    outer_fold: int,
    formula_fold_seed: int,
    validation_folds: int,
    validation_fold: int,
    validation_seed: int,
    hard_negative_molecules: int,
    max_positive_pool: int,
    max_negative_pool: int,
    clean_anchor_scope: str = "all",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    if "native_action_view_representable" not in actions.columns:
        raise RuntimeError("actions lack native representability routing")
    query_count = len(graph["query_row"])
    if len(graph["query_ptr"]) != query_count + 1:
        raise RuntimeError("candidate graph query pointers are invalid")
    query_formula = np.asarray(graph["query_formula"]).astype(str)
    query_ik14 = np.asarray(graph["query_ik14"]).astype(str)
    outer = np.asarray([
        stable_fold(value, 5, formula_fold_seed) for value in query_formula
    ], dtype=np.int8)
    outer_train = np.flatnonzero(outer != int(outer_fold))
    action_queries = set(map(int, actions["query_index"]))
    action_formulas = set(actions["query_formula"].astype(str))
    clean_only_formulas = set(query_formula[outer_train]) - action_formulas
    validation_formulas = {
        value for value in clean_only_formulas
        if stable_fold(value, validation_folds, validation_seed) == validation_fold
    }
    eligible_train_queries = np.asarray([
        query for query in outer_train if query_formula[query] not in validation_formulas
    ], dtype=np.int64)
    validation_queries = np.asarray([
        query for query in outer_train if query_formula[query] in validation_formulas
    ], dtype=np.int64)
    if action_queries - set(map(int, eligible_train_queries)):
        raise RuntimeError("selected action query was withheld from optimization")
    if set(query_formula[eligible_train_queries]) & set(query_formula[validation_queries]):
        raise RuntimeError("native train/validation formulas overlap")

    if clean_anchor_scope == "all":
        train_queries = eligible_train_queries
    elif clean_anchor_scope == "action_queries_plus_formula":
        # Keep every action query.  For formulas without actions, retain one
        # deterministic measured query so the native clean-identity stream
        # still spans the complete outer-train formula support without
        # drowning the hard-positive bridge in tens of thousands of duplicate
        # clean anchors.
        selected = set(action_queries)
        first_by_formula: dict[str, int] = {}
        for query in map(int, eligible_train_queries):
            formula = str(query_formula[query])
            first_by_formula.setdefault(formula, query)
        selected.update(first_by_formula.values())
        train_queries = np.asarray(sorted(selected), dtype=np.int64)
    else:
        raise ValueError(f"unknown clean anchor scope: {clean_anchor_scope}")

    action_by_query: dict[int, list[int]] = defaultdict(list)
    for action_index, query in enumerate(actions["query_index"].to_numpy(np.int64)):
        action_by_query[int(query)].append(int(action_index))
    maximum_actions_per_query = max(map(len, action_by_query.values()), default=0)

    row_position = embedding_positions(cache)
    train_registry = Registry()
    val_registry = Registry()

    def make_pool(
        queries: np.ndarray,
        registry: Registry,
        *,
        allow_actions: bool,
    ) -> dict[str, np.ndarray]:
        anchors: list[int] = []
        positive: list[int] = []
        negative: list[int] = []
        positive_ptr = [0]
        negative_ptr = [0]
        event_kind: list[int] = []
        event_query: list[int] = []
        event_action: list[int] = []
        event_formula: list[str] = []

        def append_event(
            anchor: int,
            positives: Iterable[int],
            negatives: Iterable[int],
            *,
            kind: int,
            query: int,
            action_index: int,
        ) -> None:
            p = bounded(positives, max_positive_pool)
            n = bounded(negatives, max_negative_pool)
            if anchor in p or anchor in n or set(p) & set(n):
                raise RuntimeError("native triplet event has overlapping semantic roles")
            anchors.append(int(anchor))
            positive.extend(p)
            negative.extend(n)
            positive_ptr.append(len(positive))
            negative_ptr.append(len(negative))
            event_kind.append(int(kind))
            event_query.append(int(query))
            event_action.append(int(action_index))
            event_formula.append(str(query_formula[query]))

        for query in map(int, queries):
            molecules = query_molecules(graph, query)
            query_row = int(graph["query_row"][query])
            if str(graph["query_ik14"][query]) != query_ik14[query]:
                raise RuntimeError("candidate graph query identity is internally inconsistent")
            positive_rows = ordered_unique(
                row for _, label, rows in molecules if label for row in rows
            )
            positive_rows = [row for row in positive_rows if row != query_row]
            if not positive_rows:
                raise RuntimeError(f"query {query} has no distinct positive spectrum")
            clean_hard_rows = hard_negative_rows(
                graph,
                query,
                cache,
                row_position,
                maximum_molecules=hard_negative_molecules,
            )
            clean_anchor = registry.add(Registry.HDF5, query_row)
            clean_positive = [
                registry.add(Registry.HDF5, row) for row in positive_rows
            ]
            clean_negative = [
                registry.add(Registry.HDF5, row)
                for row in ordered_unique(clean_hard_rows)
            ]
            append_event(
                clean_anchor,
                clean_positive,
                clean_negative,
                kind=0,
                query=query,
                action_index=-1,
            )

            if not allow_actions:
                continue
            for action_index in action_by_query.get(query, []):
                if int(actions.at[action_index, "query_row"]) != query_row:
                    raise RuntimeError("action query_row disagrees with candidate graph")
                if str(actions.at[action_index, "query_ik14"]) != query_ik14[query]:
                    raise RuntimeError("action query identity disagrees with candidate graph")
                if str(actions.at[action_index, "query_formula"]) != query_formula[query]:
                    raise RuntimeError("action query formula disagrees with candidate graph")
                exact_positive = int(actions.at[action_index, "action_positive_row"])
                exact_negative = int(actions.at[action_index, "action_hard_negative_row"])
                _, positive_label, _ = locate_candidate_row(
                    molecules, exact_positive
                )
                _, negative_label, _ = locate_candidate_row(
                    molecules, exact_negative
                )
                if not positive_label or negative_label:
                    raise RuntimeError("action exact boundary has invalid labels")
                if exact_positive == query_row:
                    raise RuntimeError(
                        "hard-positive bridge requires an independent measured positive"
                    )
                if bool(actions.at[action_index, "native_action_view_representable"]):
                    action_position = registry.add(Registry.ACTION, action_index)
                    # The only admitted action relation is the empirically
                    # positive hard-positive bridge. The preceding experiment
                    # proved that clean -> action is directionally harmful and
                    # it is therefore forbidden here.
                    # the modified action view must recognize an independently
                    # measured spectrum of the same molecule while rejecting
                    # the exact action-mined false candidate.
                    append_event(
                        action_position,
                        [registry.add(Registry.HDF5, exact_positive)],
                        [registry.add(Registry.HDF5, exact_negative)],
                        kind=2,
                        query=query,
                        action_index=action_index,
                    )
                else:
                    # The unmodified native preprocessor cannot consume an
                    # all-zero action view. Preserve its verified boundary once
                    # as an ordinary measured clean-anchor triplet. Effective
                    # clean-only aliases were already collapsed above, so this
                    # fallback cannot turn invisible action multiplicity into dose.
                    clean_boundary_anchor = registry.add_hdf5_clone(query_row)
                    append_event(
                        clean_boundary_anchor,
                        [registry.add(Registry.HDF5, exact_positive)],
                        [registry.add(Registry.HDF5, exact_negative)],
                        kind=1,
                        query=query,
                        action_index=action_index,
                    )

        registry_kind, registry_source_index = registry.arrays()
        return {
            "registry_kind": registry_kind,
            "registry_source_index": registry_source_index,
            "anchor_idx": np.asarray(anchors, dtype=np.int64),
            "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
            "positive_idx": np.asarray(positive, dtype=np.int64),
            "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
            "negative_idx": np.asarray(negative, dtype=np.int64),
            "event_kind": np.asarray(event_kind, dtype=np.int8),
            "event_query": np.asarray(event_query, dtype=np.int64),
            "event_action_index": np.asarray(event_action, dtype=np.int64),
            "event_formula": np.asarray(event_formula),
        }

    train_pool = make_pool(train_queries, train_registry, allow_actions=True)
    validation_pool = make_pool(
        validation_queries, val_registry, allow_actions=False
    )
    action_events = train_pool["event_action_index"]
    action_events = action_events[action_events >= 0]
    action_counts = np.bincount(action_events, minlength=len(actions))
    expected_action_counts = np.ones(len(actions), dtype=np.int64)
    if not np.array_equal(action_counts, expected_action_counts):
        raise RuntimeError(
            "each effective action must contribute exactly one native triplet"
        )
    split_report = {
        "outer_train_queries": int(len(outer_train)),
        "eligible_train_queries": int(len(eligible_train_queries)),
        "train_queries": int(len(train_queries)),
        "clean_anchor_scope": clean_anchor_scope,
        "synthetic_gradient_bearing_query_equalization_fillers": 0,
        "validation_queries": int(len(validation_queries)),
        "train_formulas": int(len(set(query_formula[train_queries]))),
        "validation_formulas": int(len(validation_formulas)),
        "validation_formulas_have_no_selected_actions": not bool(
            validation_formulas & action_formulas
        ),
        "train_validation_formula_disjoint": True,
        "outer_held_query_actions_used": False,
        "all_selected_actions_in_optimization": True,
        "action_queries": int(len(action_queries)),
        "maximum_actions_per_query": int(maximum_actions_per_query),
    }
    return train_pool, validation_pool, split_report


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.validation_folds < 2 or not 0 <= args.validation_fold < args.validation_folds:
        raise ValueError("invalid validation fold configuration")
    if min(args.max_positive_pool, args.max_negative_pool) < 2:
        raise ValueError("clean native dynamic pools must permit at least two members")
    report_path = args.ledger_dir / "report.json"
    actions_path = args.ledger_dir / "training_actions.csv.gz"
    spectra_path = args.ledger_dir / "action_spectra.npz"
    for path in (
        report_path, actions_path, spectra_path, args.graph,
        args.embedding_cache, args.data,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    ledger_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not _report_allows_training(ledger_report):
        raise RuntimeError("action ledger is not a formal outer-safe training artifact")
    if int(ledger_report.get("outer_formula_fold", -1)) != args.outer_fold:
        raise RuntimeError("action ledger outer fold drifted")
    observed_input_sha256 = audit_formal_inputs(
        ledger_report,
        report_path=report_path,
        actions_path=actions_path,
        spectra_path=spectra_path,
        graph_path=args.graph,
        embedding_cache_path=args.embedding_cache,
    )
    complete_actions = pd.read_csv(actions_path, low_memory=False)
    qualified_actions = select_actions(
        complete_actions,
        margin_floor=args.margin_floor,
        outer_fold=args.outer_fold,
        formula_fold_seed=args.formula_fold_seed,
        expected_actions=args.expected_actions,
    )
    qualified_targeted, qualified_control = align_action_tensor_archive(
        qualified_actions, load_npz(spectra_path),
    )
    actions, targeted, action_aliases, semantic_aliases = (
        canonicalize_semantic_action_aliases(
            qualified_actions, qualified_targeted,
        )
    )
    representative_rows = action_aliases.loc[
        action_aliases["is_canonical_representative"]
    ].sort_values("canonical_action_index", kind="stable")
    if not np.array_equal(
        representative_rows["canonical_action_index"].to_numpy(np.int64),
        np.arange(len(actions), dtype=np.int64),
    ):
        raise RuntimeError("semantic alias representatives are not canonical-aligned")
    control = np.ascontiguousarray(qualified_control[
        representative_rows["qualified_action_index"].to_numpy(np.int64)
    ])
    distinct_control = np.asarray([
        not np.array_equal(targeted[index], control[index])
        for index in range(len(actions))
    ], dtype=bool)
    matched_control_report = {
        "version": MATCHED_CONTROL_VERSION,
        "rows": int(len(actions)),
        "same_query_by_construction": True,
        "cross_query_tensor_donors": 0,
        "targeted_control_distinct_rows": int(np.sum(distinct_control)),
        "targeted_control_distinct_fraction": float(np.mean(distinct_control)),
    }
    if not np.any(distinct_control):
        raise RuntimeError("all canonical matched controls equal targeted actions")
    actions, targeted, control, action_aliases, effective_native = (
        canonicalize_effective_native_triplets(
            actions, targeted, control, action_aliases,
        )
    )
    effective_distinct_control = np.asarray([
        not np.array_equal(targeted[index], control[index])
        for index in range(len(actions))
    ], dtype=bool)
    matched_control_report.update({
        "effective_rows": int(len(actions)),
        "effective_targeted_control_distinct_rows": int(
            np.sum(effective_distinct_control)
        ),
        "effective_targeted_control_distinct_fraction": float(
            np.mean(effective_distinct_control)
        ),
    })
    semantic_aliases = dict(semantic_aliases)
    semantic_aliases["pre_native_effective_semantic_units"] = int(
        semantic_aliases["unique_semantic_training_units"]
    )
    semantic_aliases["unique_semantic_training_units"] = int(len(actions))
    semantic_aliases["duplicate_alias_rows"] = int(
        len(qualified_actions) - len(actions)
    )
    semantic_aliases["effective_native_triplets"] = effective_native
    semantic_aliases["all_qualified_rows_mapped_once"] = bool(
        len(action_aliases) == len(qualified_actions)
        and action_aliases["qualified_action_id"].nunique() == len(qualified_actions)
    )
    semantic_aliases["canonical_relations_are_unique"] = bool(
        action_aliases["canonical_action_index"].nunique() == len(actions)
    )
    targeted_native_materialization = audit_native_action_bank(targeted)
    control_native_materialization = audit_native_action_bank(control)
    identity_preservation = audit_same_identity_action_views(
        actions, targeted, control, data=args.data,
    )
    duplicate_semantic_actions = int(semantic_aliases["duplicate_alias_rows"])
    graph = load_npz(args.graph)
    if len(graph.get("query_row", [])) != REGISTERED_FORMAL_COUNTS["graph_queries"]:
        raise RuntimeError("registered candidate-graph query count drifted")
    cache = load_npz(args.embedding_cache)
    train_pool, validation_pool, split_report = build_pools(
        graph,
        cache,
        actions,
        outer_fold=args.outer_fold,
        formula_fold_seed=args.formula_fold_seed,
        validation_folds=args.validation_folds,
        validation_fold=args.validation_fold,
        validation_seed=args.validation_seed,
        hard_negative_molecules=args.hard_negative_molecules,
        max_positive_pool=args.max_positive_pool,
        max_negative_pool=args.max_negative_pool,
        clean_anchor_scope=args.clean_anchor_scope,
    )
    train_report = pool_report(train_pool, epochs=args.epochs)
    validation_report = pool_report(validation_pool, epochs=args.epochs)
    qualified_source_counts = Counter(qualified_actions["source"].astype(str))
    qualified_family_counts = Counter(
        qualified_actions["source"].astype(str)
        + "|" + qualified_actions["family"].astype(str)
    )
    canonical_source_counts = Counter(actions["source"].astype(str))
    canonical_family_counts = Counter(
        actions["source"].astype(str) + "|" + actions["family"].astype(str)
    )
    gates = {
        "formal_outer_safe_ledger": True,
        "registered_qualified_source_closure": (
            set(qualified_source_counts) == REGISTERED_SOURCES
        ),
        "all_qualified_aliases_map_to_unique_training_units": (
            int(semantic_aliases["qualified_provenance_rows"])
            == len(qualified_actions) == int(args.expected_actions)
            and int(semantic_aliases["unique_semantic_training_units"])
            == len(actions)
            and len(actions) + duplicate_semantic_actions == len(qualified_actions)
            and bool(semantic_aliases["all_qualified_rows_mapped_once"])
            and bool(semantic_aliases["canonical_relations_are_unique"])
        ),
        "every_effective_action_has_exactly_one_native_triplet": (
            train_report["action_units"] == len(actions)
            and train_report["action_clean_boundary_triplets"]
            == int((~actions["native_action_view_representable"]).sum())
            and train_report["action_measured_positive_hard_triplets"]
            == int(actions["native_action_view_representable"].sum())
            and train_report["forbidden_clean_to_action_triplets"] == 0
            and train_report["action_events"] == len(actions)
        ),
        "all_action_triplets_preserve_exact_boundary": bool(
            train_report["all_action_events_are_singleton_triplets"]
        ),
        "query_disjoint_native_one_pass_is_required": True,
        "no_synthetic_gradient_bearing_query_fillers": bool(
            split_report[
                "synthetic_gradient_bearing_query_equalization_fillers"
            ] == 0
            and train_report["clean_anchors"] == split_report["train_queries"]
        ),
        "train_clean_anchor_scale": (
            train_report["clean_anchors"] >= args.minimum_train_clean_anchors
        ),
        "validation_clean_anchor_scale": (
            validation_report["clean_anchors"]
            >= args.minimum_validation_clean_anchors
        ),
        "train_formula_scale": (
            train_report["unique_formulas"] >= args.minimum_train_formulas
        ),
        "clean_positive_dynamic_membership": (
            train_report["clean_positive_pool"]["median"] >= 2
        ),
        "clean_negative_dynamic_membership": (
            train_report["clean_negative_pool"]["median"] >= 2
        ),
        "targeted_control_distinct_on_native_action_views": not np.array_equal(
            targeted[actions["native_action_view_representable"].to_numpy(bool)],
            control[actions["native_action_view_representable"].to_numpy(bool)],
        ),
        "targeted_representable_actions_roundtrip_exactly": bool(
            targeted_native_materialization[
                "all_representable_actions_roundtrip_exactly"
            ]
        ),
        "control_representable_actions_roundtrip_exactly": bool(
            control_native_materialization[
                "all_representable_actions_roundtrip_exactly"
            ]
        ),
        "both_action_arms_keep_registered_query_provenance": bool(
            identity_preservation["both_arms_keep_registered_query_provenance"]
        ),
        "train_validation_formula_disjoint": bool(
            split_report["train_validation_formula_disjoint"]
        ),
        "all_actions_in_train_not_validation": bool(
            split_report["all_selected_actions_in_optimization"]
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"native Noise triplet gates failed: {gates}")
    report = {
        "status": "NOISE_DREAMS_NATIVE_TRIPLETS_COMPLETE",
        "implementation": {
            "builder_version": NATIVE_TRIPLET_BUILDER_VERSION,
            "matched_control_version": MATCHED_CONTROL_VERSION,
            "action_spectrum_adapter_version": (
                NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION
            ),
        },
        "scientific_contract": (
            "Noise changes native triplet membership/spectra and uses one "
            "pre-registered pass; DreaMS dataset/preprocessor/model/loss/optimizer "
            "remain unmodified."
        ),
        "action_selection": {
            "policy": (
                "all qualified provenance rows retained; exact effective native "
                "triplet aliases collapse to one optimizer unit"
            ),
            "qualified_rows": int(len(qualified_actions)),
            "semantic_training_units": int(len(actions)),
            "semantic_alias_rows_collapsed_from_optimizer_dose": int(
                duplicate_semantic_actions
            ),
            "semantic_alias_audit": semantic_aliases,
            "queries": int(qualified_actions["query_index"].nunique()),
            "identities": int(qualified_actions["query_ik14"].astype(str).nunique()),
            "formulas": int(qualified_actions["query_formula"].astype(str).nunique()),
            "qualified_sources": dict(sorted(qualified_source_counts.items())),
            "qualified_source_families": dict(sorted(qualified_family_counts.items())),
            "canonical_representative_sources": dict(
                sorted(canonical_source_counts.items())
            ),
            "canonical_representative_source_families": dict(
                sorted(canonical_family_counts.items())
            ),
            "one_best_query_compression_used": False,
            "representable_units_use_one_hard_positive_triplet": True,
            "clean_to_action_deployment_training_forbidden": True,
            "unrepresentable_units_use_clean_anchor": True,
            "native_action_view_units": int(
                actions["native_action_view_representable"].sum()
            ),
            "clean_boundary_only_units": int(
                (~actions["native_action_view_representable"]).sum()
            ),
            "all_qualified_action_ids_preserved_in_alias_ledger": True,
            "exact_action_positive_and_hard_negative_retained": True,
            "historical_e4_training_kernel_reused": False,
            "e4_candidate_and_role_confounder_content": (
                "represented through the qualified N_mature source; the 190324-row "
                "outcome-free construction bank is not relabelled as a positive action bank"
            ),
        },
        "triplet_design": {
            "clean_anchor_base": args.clean_anchor_scope,
            "action_clean_boundary_triplet": (
                "unrepresentable action only: clean query -> exact positive -> exact hard negative"
            ),
            "deployment_bridge_triplet": "forbidden by run 2344688 evidence",
            "hard_positive_bridge_triplet": (
                "same-identity action view -> independent measured positive -> exact hard negative"
            ),
            "unrepresentable_action_policy": (
                "if targeted or same-query matched control has no positive fragment intensity, "
                "both arms use one deduplicated clean-boundary triplet only"
            ),
            "clean_positive_pool": "same-identity measured candidate spectra",
            "clean_negative_pool": "official-embedding hard candidate molecules only",
            "dynamic_samples_per_getitem": {"positive": 1, "negative": 1},
            "dose": (
                "one query-disjoint native pass; exactly one hard-positive triplet "
                "per representable action and one clean-boundary fallback per "
                "unrepresentable action; one ordinary clean identity event per "
                "selected clean query; no action replay and no synthetic "
                "gradient-bearing query equalization"
            ),
            "hard_negative_molecules": int(args.hard_negative_molecules),
            "maximum_positive_pool": int(args.max_positive_pool),
            "maximum_negative_pool": int(args.max_negative_pool),
        },
        "split": split_report,
        "train": train_report,
        "validation": validation_report,
        "same_query_matched_control": matched_control_report,
        "same_identity_action_view_audit": identity_preservation,
        "native_action_materialization": {
            "targeted": targeted_native_materialization,
            "matched_control": control_native_materialization,
        },
        "gates": gates,
        "provenance": observed_input_sha256,
        "claim_limit": (
            "Triplet coverage and semantic validity only; no encoder improvement "
            "or absence of overfitting is claimed before native training and held evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_native_triplets_", dir=args.output.parent))
    try:
        action_archive = {
            "action_ids": fixed_unicode(actions["action_id"]),
            "query_index": actions["query_index"].to_numpy(np.int64),
            "source": fixed_unicode(actions["source"]),
            "family": fixed_unicode(actions["family"]),
            "recipe_id": fixed_unicode(actions["recipe_id"]),
            "native_action_view_representable": actions[
                "native_action_view_representable"
            ].to_numpy(bool),
            "targeted_action_spectra": targeted,
            "control_action_spectra": control,
        }
        audit_pickle_free_arrays(train_pool, label="native train pool")
        audit_pickle_free_arrays(validation_pool, label="native validation pool")
        audit_pickle_free_arrays(action_archive, label="native action archive")
        np.savez_compressed(staging / "train_pool.npz", **train_pool)
        np.savez_compressed(staging / "validation_pool.npz", **validation_pool)
        np.savez_compressed(staging / "action_spectra.npz", **action_archive)
        # Reload every formal archive with the exact production safety policy.
        # This is the gate that the v7 writer lacked.
        archive_sources = {
            "train_pool.npz": train_pool,
            "validation_pool.npz": validation_pool,
            "action_spectra.npz": action_archive,
        }
        for archive_name, original in archive_sources.items():
            replayed = load_npz(staging / archive_name)
            audit_pickle_free_arrays(replayed, label=f"reloaded {archive_name}")
            if set(replayed) != set(original) or any(
                not np.array_equal(replayed[key], np.asarray(original[key]))
                for key in original
            ):
                raise RuntimeError(f"formal NPZ roundtrip drifted: {archive_name}")
        actions.to_csv(staging / "selected_actions.csv.gz", index=False, compression="gzip")
        qualified_actions.to_csv(
            staging / "qualified_actions.csv.gz", index=False, compression="gzip"
        )
        action_aliases.to_csv(
            staging / "action_aliases.csv.gz", index=False, compression="gzip"
        )
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "selected_actions_sha256": sha256_file(staging / "selected_actions.csv.gz"),
            "qualified_actions_sha256": sha256_file(
                staging / "qualified_actions.csv.gz"
            ),
            "action_aliases_sha256": sha256_file(staging / "action_aliases.csv.gz"),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

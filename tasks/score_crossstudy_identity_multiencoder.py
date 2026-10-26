#!/usr/bin/env python3
"""Score the frozen cross-study identity-candidate handoff.

The handoff is the sole candidate universe.  Candidate admission, the
canonical WSE score and the zero/single/multiple-candidate query blocks are
loaded from it before any query-relative feature is computed.  This prevents a
larger convenience-scoring universe from changing percentiles, margins or
missingness on the fixed biological comparison axis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import joblib

from enveda_unified_candidate_integrator_core import (
    ARMS as ENVEDA_INTEGRATOR_ARMS,
    CHECKPOINT_SCHEMA as ENVEDA_INTEGRATOR_SCHEMA,
    PRIMARY_ARM as ENVEDA_PRIMARY_ARM,
    predict_from_payload as predict_enveda_integrator,
)
from encode_pan_redu_panel_spectra import load_mgf
from massspecgym_embedding_readout_core import (
    FULL_INTEGRATION_ARMS, REQUIRED_ENCODERS, FrozenEmbeddingReadout,
    ReadoutSettings, build_independent_scalar_features,
)
from noise_gnps_article_spectral_scores import (
    P2B_WEIGHTS, frozen_p2b_pair_features,
)
from noise_msg_pair_evidence_core import MOLECULE_FEATURES, build_molecule_features


NEW_STUDY = "MSV000095331"
DEFAULT_IDENTITY_HANDOFF = Path(
    "data/validation/msv000095331_complete_identity_handoff_v3_20261023"
)
PRODUCTION_QUERY_COUNT = 38_628
PRODUCTION_CANDIDATE_COUNT = 22_046
MSG_FAMILY = "noise_msg_fusion_stack_v1"
NEW_MGF = Path("data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf")
CHEMPROP_MGF = {
    "MSV000094899": Path("data/reference/chemprop2_m1_20261011/consensus.mgf"),
    "MSV000096724": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_23drug/spectra/specs_ms.mgf"
    ),
    "MSV000093571": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_45drug/spectra/specs_ms.mgf"
    ),
}


def prepare_spectrum(spectrum: np.ndarray, n_peaks: int = 100) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("spectrum must have shape (2, peaks)")
    keep = (
        np.isfinite(values[0]) & np.isfinite(values[1])
        & (values[0] > 0) & (values[1] > 0)
    )
    values = values[:, keep]
    if values.shape[1] > n_peaks:
        selected = np.argpartition(values[1], -n_peaks)[-n_peaks:]
        values = values[:, selected]
    return values[:, np.argsort(values[0], kind="stable")]


def peak_rows(peaks: np.ndarray) -> np.ndarray:
    """Return MGF peaks as ``(peaks, 2)``, including an empty spectrum."""
    values = np.asarray(peaks, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("MGF peak matrix must have shape (2, peaks)")
    return values.T


def valid_precursor_mz(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        ids = np.asarray(body["entity_id"]).astype("U")
        matrix = np.asarray(body["embeddings"], dtype=np.float32)
    if matrix.ndim != 2 or len(ids) != len(matrix) or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid embedding cache: {path}")
    return dict(zip(ids.tolist(), matrix, strict=True))


def load_fixed_identity_axis(
    handoff: Path,
    *,
    expected_queries: int | None = None,
    expected_candidates: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, object]]:
    """Load and cross-check the truth-free application candidate axis.

    The NPZ controls ordering and query boundaries.  The TSV files contribute
    stable identifiers and metadata only; their candidate rows are reordered
    to the NPZ ``candidate_pair_ids`` before they are returned.
    """
    pair_path = handoff / "complete_identity_candidate_pairs.tsv.gz"
    query_path = handoff / "directional_query_registry.tsv.gz"
    axis_path = handoff / "wse_identity_inference_axis.npz"
    for path in (pair_path, query_path, axis_path):
        if not path.is_file():
            raise FileNotFoundError(f"fixed identity handoff input is absent: {path}")

    pair_table = pd.read_csv(
        pair_path, sep="\t",
        dtype={
            "candidate_pair_id": str, "query_id": str, "candidate_id": str,
            "query_study_id": str, "query_raw_entity_id": str,
            "candidate_study_id": str, "candidate_raw_entity_id": str,
        },
    )
    query_table = pd.read_csv(
        query_path, sep="\t",
        dtype={
            "query_id": str, "query_study_id": str,
            "query_raw_entity_id": str, "candidate_study_id": str,
        },
    )
    required_pairs = {
        "candidate_pair_id", "query_id", "candidate_id", "query_study_id",
        "query_raw_entity_id", "candidate_study_id", "candidate_raw_entity_id",
        "query_precursor_mz", "candidate_precursor_mz", "wse_score",
        "wse_matched_fragments",
    }
    required_queries = {
        "query_id", "query_study_id", "query_raw_entity_id",
        "candidate_study_id", "candidate_count", "query_status",
    }
    if not required_pairs.issubset(pair_table.columns):
        raise RuntimeError(
            f"fixed pair table lacks columns: {sorted(required_pairs - set(pair_table.columns))}"
        )
    if not required_queries.issubset(query_table.columns):
        raise RuntimeError(
            f"fixed query table lacks columns: {sorted(required_queries - set(query_table.columns))}"
        )
    if pair_table["candidate_pair_id"].duplicated().any():
        raise RuntimeError("fixed identity handoff has duplicate candidate_pair_id")
    if query_table.duplicated(["query_id", "candidate_study_id"]).any():
        raise RuntimeError("fixed identity handoff has duplicate composite query blocks")

    with np.load(axis_path, allow_pickle=False) as body:
        schema = str(np.asarray(body["schema"]).item())
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        query_ids = np.asarray(body["query_ids"]).astype("U")
        candidate_ids = np.asarray(body["candidate_ids"]).astype("U")
        candidate_pair_ids = np.asarray(body["candidate_pair_ids"]).astype("U")
        wse_score = np.asarray(body["wse_score"], dtype=np.float32)
        wse_available = np.asarray(body["wse_availability"], dtype=bool)
    if schema != "msv000095331_complete_identity_candidate_axis_v1":
        raise RuntimeError(f"unexpected fixed identity axis schema: {schema}")
    if (
        query_ptr.ndim != 1 or len(query_ptr) != len(query_ids) + 1
        or query_ptr[0] != 0 or np.any(np.diff(query_ptr) < 0)
        or int(query_ptr[-1]) != len(candidate_pair_ids)
    ):
        raise RuntimeError("fixed identity handoff has an invalid zero/single/multi query pointer")
    if not (
        len(candidate_ids) == len(candidate_pair_ids)
        == len(wse_score) == len(wse_available)
    ):
        raise RuntimeError("fixed identity candidate arrays are not aligned")
    if len(set(candidate_pair_ids.tolist())) != len(candidate_pair_ids):
        raise RuntimeError("fixed identity inference NPZ has duplicate candidate_pair_ids")
    if expected_queries is not None and len(query_ids) != expected_queries:
        raise RuntimeError(
            f"fixed identity query count drifted: {len(query_ids)} != {expected_queries}"
        )
    if expected_candidates is not None and len(candidate_pair_ids) != expected_candidates:
        raise RuntimeError(
            "fixed identity candidate count drifted: "
            f"{len(candidate_pair_ids)} != {expected_candidates}"
        )

    table_query_ids = (
        query_table["query_id"].astype(str) + "|"
        + query_table["candidate_study_id"].astype(str)
    ).to_numpy(dtype="U")
    if not np.array_equal(table_query_ids, query_ids):
        raise RuntimeError("fixed query registry differs from canonical NPZ block order")
    candidate_count = pd.to_numeric(query_table["candidate_count"], errors="raise").to_numpy(np.int64)
    if not np.array_equal(candidate_count, np.diff(query_ptr)):
        raise RuntimeError("fixed query registry candidate counts differ from canonical query_ptr")

    indexed = pair_table.set_index("candidate_pair_id", drop=False)
    if set(indexed.index.astype(str)) != set(candidate_pair_ids.tolist()):
        missing = int(np.sum(~np.isin(candidate_pair_ids, indexed.index.astype(str))))
        outside = int(np.sum(~indexed.index.astype(str).isin(set(candidate_pair_ids.tolist()))))
        raise RuntimeError(
            f"fixed pair TSV and NPZ axes differ: missing={missing}, outside={outside}"
        )
    pair_table = indexed.loc[candidate_pair_ids.tolist()].reset_index(drop=True)
    if not np.array_equal(pair_table["candidate_id"].astype(str).to_numpy(dtype="U"), candidate_ids):
        raise RuntimeError("fixed candidate IDs differ between pair TSV and NPZ")
    repeated_query_ids = np.repeat(query_ids, np.diff(query_ptr))
    pair_query_ids = (
        pair_table["query_id"].astype(str) + "|"
        + pair_table["candidate_study_id"].astype(str)
    ).to_numpy(dtype="U")
    if not np.array_equal(pair_query_ids, repeated_query_ids):
        raise RuntimeError("fixed candidate rows are not contiguous on the composite query axis")
    if not np.allclose(
        pd.to_numeric(pair_table["wse_score"], errors="raise").to_numpy(np.float64),
        wse_score.astype(np.float64), rtol=0.0, atol=5e-8,
    ):
        raise RuntimeError("canonical WSE differs between fixed pair TSV and inference NPZ")

    # The NPZ value is the deployed baseline.  No application-side entropy
    # implementation is allowed to replace or recompute it.
    pair_table["query_block_id"] = repeated_query_ids
    pair_table["WSE_score"] = wse_score
    pair_table["WSE_matched_fragments"] = pd.to_numeric(
        pair_table["wse_matched_fragments"], errors="raise"
    ).to_numpy(np.int64)
    pair_table["WSE_available"] = wse_available
    query_table["query_block_id"] = query_ids
    query_table["candidate_cardinality"] = np.select(
        [candidate_count == 0, candidate_count == 1],
        ["zero", "single"], default="multiple",
    )
    provenance = {
        "handoff": str(handoff),
        "pair_table": {"path": str(pair_path), "sha256": sha256(pair_path)},
        "query_registry": {"path": str(query_path), "sha256": sha256(query_path)},
        "inference_axis": {"path": str(axis_path), "sha256": sha256(axis_path)},
        "query_blocks": int(len(query_ids)),
        "candidate_pairs": int(len(candidate_pair_ids)),
        "zero_candidate_blocks": int(np.sum(candidate_count == 0)),
        "single_candidate_blocks": int(np.sum(candidate_count == 1)),
        "multiple_candidate_blocks": int(np.sum(candidate_count > 1)),
        "wse_source": "canonical_wse_identity_inference_axis_npz",
    }
    return pair_table, query_table, query_ptr, query_ids, provenance


def load_auxiliary_candidate_scores(
    specifications: list[str], candidate_pair_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Align optional complete-expert scores to the already-fixed pair axis.

    Syntax is ``NAME=PATH::COLUMN``.  Superset rows are discarded *before*
    any query-relative feature is built.  Missing or explicitly unavailable
    rows remain NaN and therefore make that exact query block abstain for a
    checkpoint that requires the family.
    """
    axis = pd.Index(np.asarray(candidate_pair_ids).astype("U"), name="candidate_pair_id")
    output: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, object]] = {}
    for item in specifications:
        if "=" not in item or "::" not in item.split("=", 1)[1]:
            raise ValueError("--auxiliary-score must use NAME=PATH::COLUMN")
        name, raw = item.split("=", 1)
        raw_path, column = raw.rsplit("::", 1)
        if not name or name in output or not column:
            raise ValueError(f"invalid or duplicate auxiliary family: {name!r}")
        path = Path(raw_path)
        table = pd.read_csv(path, sep="\t", dtype={"candidate_pair_id": str})
        if "method_name" in table.columns:
            table = table.loc[table["method_name"].astype(str).eq(name)].copy()
        if "candidate_pair_id" not in table or column not in table:
            raise RuntimeError(f"auxiliary family {name} lacks candidate_pair_id/{column}")
        if table["candidate_pair_id"].duplicated().any():
            raise RuntimeError(f"auxiliary family {name} has duplicate candidate_pair_id")
        fixed = table.loc[table["candidate_pair_id"].isin(axis)].copy()
        values = pd.to_numeric(
            fixed.set_index("candidate_pair_id")[column].reindex(axis), errors="coerce"
        ).to_numpy(np.float64)
        if "evidence_available" in fixed.columns:
            availability = fixed.set_index("candidate_pair_id")["evidence_available"].reindex(axis)
            if availability.dropna().dtype == bool:
                available = availability.fillna(False).to_numpy(bool)
            else:
                normalised = availability.astype("string").str.strip().str.lower()
                invalid = normalised.notna() & ~normalised.isin({"true", "false", "1", "0"})
                if invalid.any():
                    raise RuntimeError(
                        f"auxiliary family {name} has invalid evidence_available values"
                    )
                available = normalised.isin({"true", "1"}).to_numpy(bool)
            values[~available] = np.nan
        output[name] = values
        provenance[name] = {
            "path": str(path), "sha256": sha256(path), "column": column,
            "source_rows": int(len(table)), "fixed_axis_rows": int(len(fixed)),
            "outside_axis_rows_discarded": int(len(table) - len(fixed)),
            "available_fixed_axis_rows": int(np.isfinite(values).sum()),
        }
    return output, provenance


def complete_query_blocks(
    row_complete: np.ndarray, query_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack complete non-empty fixed-axis blocks without changing membership."""
    complete = np.asarray(row_complete, dtype=bool)
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        complete.ndim != 1 or ptr.ndim != 1 or len(ptr) < 1
        or ptr[0] != 0 or ptr[-1] != len(complete)
        or np.any(np.diff(ptr) < 0)
    ):
        raise RuntimeError("fixed query blocks do not align with candidate completeness")
    query_indices: list[int] = []
    parts: list[np.ndarray] = []
    counts: list[int] = []
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        if right > left and bool(complete[left:right].all()):
            query_indices.append(query)
            parts.append(np.arange(left, right, dtype=np.int64))
            counts.append(right - left)
    rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    packed_ptr = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(np.asarray(counts, dtype=np.int64)),
    ))
    return rows, np.asarray(query_indices, dtype=np.int64), packed_ptr


def score_frozen_noise_msg(
    pairs: pd.DataFrame, query_ptr: np.ndarray, model_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build one frozen MSG composite; raw MSG inputs never leave this helper."""
    root = model_path if model_path.is_dir() else model_path.parent
    artifact = root / "fusion_model.joblib" if model_path.is_dir() else model_path
    report_path = root / "report.json"
    if not artifact.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "frozen Noise-MSG requires fusion_model.joblib and report.json: "
            f"{root}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "NOISE_MSG_FUSION_FROZEN":
        raise RuntimeError("Noise-MSG artifact is not a registered frozen fusion model")
    frozen = joblib.load(artifact)
    if tuple(frozen.get("feature_names", ())) != tuple(MOLECULE_FEATURES):
        raise RuntimeError("frozen Noise-MSG feature contract drifted")
    if "model" not in frozen or not hasattr(frozen["model"], "predict_proba"):
        raise RuntimeError("frozen Noise-MSG model lacks predict_proba")

    columns = {
        "v1_cosine": "noise_v1_score",
        "official_cosine": "official_dreams_score",
        "weighted_entropy": "WSE_score",
        "sqrt_cosine": "P2b_sqrt_cosine",
        "entropy_similarity": "P2b_entropy",
        "neutral_loss_sqrt_cosine": "P2b_neutral_loss",
        "p2b_fused": "P2b_score",
    }
    missing = sorted(set(columns.values()) - set(pairs.columns))
    if missing:
        raise RuntimeError(f"Noise-MSG application evidence is absent: {missing}")
    numeric = {
        name: pd.to_numeric(pairs[column], errors="coerce").to_numpy(np.float64)
        for name, column in columns.items()
    }
    row_complete = np.column_stack([
        np.isfinite(values) for values in numeric.values()
    ]).all(axis=1)
    if "WSE_available" in pairs:
        row_complete &= pairs["WSE_available"].to_numpy(bool)
    rows, queries, packed_ptr = complete_query_blocks(row_complete, query_ptr)
    output = np.full(len(pairs), np.nan, dtype=np.float32)
    if len(rows):
        edges = {name: values[rows] for name, values in numeric.items()}
        features = build_molecule_features(
            edges, packed_ptr, np.arange(len(rows) + 1, dtype=np.int64),
        )
        model = frozen["model"]
        classes = np.asarray(getattr(model, "classes_", (0, 1)))
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            raise RuntimeError("frozen Noise-MSG model lacks one positive class")
        probability = np.asarray(model.predict_proba(features), dtype=np.float64)
        if probability.ndim != 2 or probability.shape[0] != len(rows):
            raise RuntimeError("frozen Noise-MSG probability axis drifted")
        values = probability[:, int(positive[0])]
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise RuntimeError("frozen Noise-MSG produced invalid probabilities")
        output[rows] = values.astype(np.float32)
    return output, {
        "family": MSG_FAMILY,
        "model_path": str(artifact),
        "model_sha256": sha256(artifact),
        "report_path": str(report_path),
        "report_sha256": sha256(report_path),
        "feature_names": list(MOLECULE_FEATURES),
        "input_semantics": (
            "single frozen composite from WSE, Noise V1, official DreaMS, "
            "and the three registered P2b raw channels plus P2b composite"
        ),
        "raw_channels_passed_to_unified_integrator": False,
        "scored_pairs": int(np.isfinite(output).sum()),
        "scored_queries": int(len(queries)),
        "zero_candidate_queries": int(np.sum(np.diff(query_ptr) == 0)),
        "incomplete_nonempty_queries": int(np.sum(np.diff(query_ptr) > 0) - len(queries)),
    }


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.min(values)), float(np.max(values))
    return np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)


def matched_peaks(left: np.ndarray, right: np.ndarray, tolerance: float) -> int:
    a = np.sort(np.asarray(left, dtype=np.float64)[:, 0])
    b = np.sort(np.asarray(right, dtype=np.float64)[:, 0])
    i = j = total = 0
    while i < len(a) and j < len(b):
        delta = a[i] - b[j]
        if abs(delta) <= tolerance:
            total += 1; i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return total


def frozen_p2b_block(
    noise: np.ndarray, sqrt_cosine: np.ndarray, entropy: np.ndarray,
    neutral_loss: np.ndarray,
) -> np.ndarray:
    """Apply the registered one-reference P2b support gate to one query."""
    noise = np.asarray(noise, dtype=np.float64)
    raw = np.stack((sqrt_cosine, entropy, neutral_loss), axis=1).clip(0, 1)
    channels = np.column_stack((np.clip((noise + 1) / 2, 0, 1), raw))
    fused = channels @ P2B_WEIGHTS

    def unique_top(values: np.ndarray) -> int | None:
        winners = np.flatnonzero(values >= float(np.max(values)) - 1e-12)
        return int(winners[0]) if len(winners) == 1 else None

    winner = unique_top(fused)
    if winner is None:
        return noise.astype(np.float32)
    support = sum(unique_top(raw[:, column]) == winner for column in range(3))
    baseline = unique_top(noise)
    advantage = math.inf if baseline is None else (
        0.0 if winner == baseline else float(fused[winner] - fused[baseline])
    )
    return (fused if support >= 1 and advantage >= -1e-12 else noise).astype(np.float32)


def build_enveda_application_features(
    pairs: pd.DataFrame, packed_query_ptr: np.ndarray, msg_scores: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Rebuild the Enveda integrator's exact scalar evidence contract."""
    msg = np.asarray(msg_scores, dtype=np.float64)
    if msg.shape != (len(pairs),) or not np.all(np.isfinite(msg)):
        raise RuntimeError("application MSG composite is malformed")
    graph = SimpleNamespace(
        query_ptr=np.asarray(packed_query_ptr, dtype=np.int64),
        molecule_ptr=np.arange(len(pairs) + 1, dtype=np.int64),
        molecule_label=np.zeros(len(pairs), dtype=np.int8),
    )
    encoder_scores = {
        name: pd.to_numeric(pairs[f"{name}_score"], errors="raise").to_numpy(np.float64)
        for name in REQUIRED_ENCODERS
    }
    evidence = {
        "v1_cosine": encoder_scores["noise_v1"],
        "weighted_entropy": pd.to_numeric(
            pairs["WSE_score"], errors="raise",
        ).to_numpy(np.float64),
        "p2b_fused": pd.to_numeric(
            pairs["P2b_score"], errors="raise",
        ).to_numpy(np.float64),
    }
    zeros = np.zeros(len(pairs), dtype=np.float64)
    unavailable_chemistry = {
        "chem_valid": np.zeros(len(pairs), dtype=bool),
        "candidate_utility_correct": zeros.copy(),
        "candidate_utility_zero": zeros.copy(),
        "candidate_utility_reversed": zeros.copy(),
        "candidate_utility_rotated": zeros.copy(),
    }
    return build_independent_scalar_features(
        evidence, graph, encoder_scores, "noise_v1", unavailable_chemistry,
        auxiliary_scores={MSG_FAMILY: msg},
        include_reliability_context=True,
    )


def apply_enveda_unified_model(
    pairs: pd.DataFrame, payload: dict, model_path: Path, device: torch.device,
    query_ptr: np.ndarray, query_ids: np.ndarray,
    auxiliary_scores: dict[str, np.ndarray] | None,
    arm: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the consumed-Enveda scalar integrator to the fixed application axis."""
    if payload.get("selection_class") != "candidate_evidence_integration":
        raise RuntimeError("Enveda integrator lacks candidate-evidence selection contract")
    if tuple(payload.get("encoder_names", ())) != tuple(REQUIRED_ENCODERS):
        raise RuntimeError("Enveda integrator encoder registry drifted")
    requested_arm = arm
    checkpoint_primary_arm = str(payload.get("primary_arm", ENVEDA_PRIMARY_ARM))
    arm = str(arm or checkpoint_primary_arm)
    if arm not in ENVEDA_INTEGRATOR_ARMS:
        raise RuntimeError(f"unknown Enveda integrator arm requested: {arm}")
    if arm not in payload.get("arms", {}):
        raise RuntimeError(f"Enveda integrator checkpoint does not contain requested arm: {arm}")
    supplied = auxiliary_scores or {}
    if set(supplied) != {MSG_FAMILY}:
        raise RuntimeError(
            "Enveda integrator requires exactly one frozen auxiliary family: "
            f"{MSG_FAMILY}; supplied={sorted(supplied)}"
        )
    msg = np.asarray(supplied[MSG_FAMILY], dtype=np.float64)
    if msg.shape != (len(pairs),):
        raise RuntimeError("Enveda integrator MSG composite differs from fixed candidate axis")

    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    query_ids = np.asarray(query_ids).astype("U")
    if (
        len(query_ptr) != len(query_ids) + 1 or query_ptr[0] != 0
        or query_ptr[-1] != len(pairs) or np.any(np.diff(query_ptr) < 0)
    ):
        raise RuntimeError("Enveda application inference received an invalid query pointer")
    if not np.array_equal(
        pairs["query_block_id"].astype(str).to_numpy(dtype="U"),
        np.repeat(query_ids, np.diff(query_ptr)),
    ):
        raise RuntimeError("Enveda candidate rows differ from canonical composite query blocks")
    required = [
        "WSE_score", "P2b_score",
        *[f"{name}_score" for name in REQUIRED_ENCODERS],
    ]
    row_complete = pairs[required].apply(
        pd.to_numeric, errors="coerce",
    ).notna().all(axis=1).to_numpy(bool)
    row_complete &= np.isfinite(msg)
    if "WSE_available" in pairs:
        row_complete &= pairs["WSE_available"].to_numpy(bool)
    rows, queries, packed_ptr = complete_query_blocks(row_complete, query_ptr)
    output = np.full(len(pairs), np.nan, dtype=np.float32)
    if len(rows):
        selected = pairs.iloc[rows].reset_index(drop=True)
        scalar, names = build_enveda_application_features(
            selected, packed_ptr, msg[rows],
        )
        output[rows] = predict_enveda_integrator(
            payload, names, scalar, arm=arm, device=device,
        )
    return output, {
        "schema": ENVEDA_INTEGRATOR_SCHEMA,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "arm": arm,
        "checkpoint_primary_arm": checkpoint_primary_arm,
        "arm_selection_source": "explicit_cli" if requested_arm is not None else "checkpoint_primary",
        "selection_class": str(payload["selection_class"]),
        "encoder_names": list(REQUIRED_ENCODERS),
        "auxiliary_family_names": [MSG_FAMILY],
        "auxiliary_semantics": "one frozen MSG composite; no MSG raw-channel re-entry",
        "fixed_axis_queries": int(len(query_ids)),
        "zero_candidate_queries": int(np.sum(np.diff(query_ptr) == 0)),
        "scored_pairs": int(len(rows)),
        "scored_queries": int(len(queries)),
        "incomplete_nonempty_queries": int(np.sum(np.diff(query_ptr) > 0) - len(queries)),
        "query_key": "query_id|candidate_study_id",
        "missing_rule": "complete fixed-axis composite blocks only; zero blocks retained externally",
        "chemaware_v2": "unavailable_without_candidate_structures_availability_0_continuous_0",
        "bioaware": "not_supplied_no_sample_context_fabricated",
    }


def apply_unified_model(
    pairs: pd.DataFrame, caches: dict[str, dict[str, np.ndarray]], model_path: Path,
    device: torch.device, query_ptr: np.ndarray, query_ids: np.ndarray,
    auxiliary_scores: dict[str, np.ndarray] | None = None,
    enveda_integrator_arm: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a selected MassSpecGym ranker on fixed composite query blocks."""
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        payload = torch.load(model_path, map_location="cpu")
    if payload.get("schema") == ENVEDA_INTEGRATOR_SCHEMA:
        return apply_enveda_unified_model(
            pairs, payload, model_path, device, query_ptr, query_ids,
            auxiliary_scores, arm=enveda_integrator_arm,
        )
    if enveda_integrator_arm is not None:
        raise RuntimeError(
            "--enveda-integrator-arm is only valid for an Enveda candidate-integrator checkpoint"
        )
    settings = ReadoutSettings(**payload["settings"])
    encoder_names = tuple(payload["encoder_names"])
    if encoder_names != settings.encoder_names:
        raise RuntimeError("unified model encoder registry drifted")
    if str(payload["arm"]) not in FULL_INTEGRATION_ARMS:
        raise RuntimeError("application deployment requires a full-embedding integration arm")
    if payload.get("selection_class") != "full_embedding_integration":
        raise RuntimeError("unified model lacks the full-integration selection contract")
    if encoder_names != REQUIRED_ENCODERS:
        raise RuntimeError("unified model does not contain the exact four registered encoders")
    missing = sorted(set(encoder_names) - set(caches))
    if missing:
        raise RuntimeError(f"unified model requires missing encoder caches: {missing}")
    expected_auxiliary = tuple(payload.get("auxiliary_family_names", ()))
    supplied_auxiliary = auxiliary_scores or {}
    missing_auxiliary = sorted(set(expected_auxiliary) - set(supplied_auxiliary))
    if missing_auxiliary:
        raise RuntimeError(
            "unified auxiliary-family contract is incomplete: "
            f"missing={missing_auxiliary}, supplied={list(supplied_auxiliary)}"
        )
    if any(np.asarray(values).shape != (len(pairs),) for values in supplied_auxiliary.values()):
        raise RuntimeError("unified auxiliary scores do not match the fixed candidate axis")
    model = FrozenEmbeddingReadout(str(payload["arm"]), settings)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    query_ids = np.asarray(query_ids).astype("U")
    if (
        query_ptr.ndim != 1 or len(query_ptr) != len(query_ids) + 1
        or query_ptr[0] != 0 or np.any(np.diff(query_ptr) < 0)
        or int(query_ptr[-1]) != len(pairs)
    ):
        raise RuntimeError("unified inference received an invalid fixed query pointer")
    if "query_block_id" not in pairs:
        raise RuntimeError("unified inference requires canonical query_block_id")
    if not np.array_equal(
        pairs["query_block_id"].astype(str).to_numpy(dtype="U"),
        np.repeat(query_ids, np.diff(query_ptr)),
    ):
        raise RuntimeError("candidate rows differ from canonical composite query blocks")

    required_columns = ["WSE_score", "P2b_score", *[f"{name}_score" for name in encoder_names]]
    row_complete = pairs[required_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if expected_auxiliary:
        row_complete &= np.column_stack([
            np.isfinite(np.asarray(supplied_auxiliary[name], dtype=np.float64))
            for name in expected_auxiliary
        ]).all(axis=1)
    selected_queries: list[int] = []
    selected_parts: list[np.ndarray] = []
    for query_index, (start, stop) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        start, stop = int(start), int(stop)
        if stop > start and bool(row_complete.iloc[start:stop].all()):
            selected_queries.append(query_index)
            selected_parts.append(np.arange(start, stop, dtype=np.int64))
    selected_indices = (
        np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    )
    full_output = np.full(len(pairs), np.nan, dtype=np.float32)
    selected_pairs = pairs.iloc[selected_indices].copy().reset_index(drop=True)
    if selected_pairs.empty:
        return full_output, {
            "model_path": str(model_path), "model_sha256": sha256(model_path),
            "arm": str(payload["arm"]), "encoder_names": list(encoder_names),
            "auxiliary_family_names": list(expected_auxiliary),
            "unused_supplied_auxiliary_family_names": sorted(
                set(supplied_auxiliary) - set(expected_auxiliary)
            ),
            "selection_class": str(payload["selection_class"]),
            "representation_increment_verdict": str(
                payload["representation_increment_verdict"]
            ),
            "fixed_axis_queries": int(len(query_ids)),
            "zero_candidate_queries": int(np.sum(np.diff(query_ptr) == 0)),
            "scored_pairs": 0, "scored_queries": 0,
            "chemaware_v2": "unavailable_without_candidate_structures_exact_zero",
        }

    counts = np.diff(query_ptr)[np.asarray(selected_queries, dtype=np.int64)]
    packed_query_ptr = np.concatenate(([0], np.cumsum(counts)))
    graph = SimpleNamespace(
        query_ptr=packed_query_ptr,
        molecule_ptr=np.arange(len(selected_pairs) + 1, dtype=np.int64),
        molecule_label=np.zeros(len(selected_pairs), dtype=np.int8),
    )
    encoder_scores = {
        name: pd.to_numeric(selected_pairs[f"{name}_score"], errors="raise").to_numpy(np.float64)
        for name in encoder_names
    }
    evidence = {
        "v1_cosine": encoder_scores["noise_v1"],
        "weighted_entropy": pd.to_numeric(selected_pairs["WSE_score"], errors="raise").to_numpy(float),
        "p2b_fused": pd.to_numeric(selected_pairs["P2b_score"], errors="raise").to_numpy(float),
    }
    unavailable_chemistry = {
        "chem_valid": np.zeros(len(selected_pairs), dtype=bool),
        "candidate_utility_correct": np.zeros(len(selected_pairs), dtype=np.float64),
        "candidate_utility_zero": np.zeros(len(selected_pairs), dtype=np.float64),
        "candidate_utility_reversed": np.zeros(len(selected_pairs), dtype=np.float64),
        "candidate_utility_rotated": np.zeros(len(selected_pairs), dtype=np.float64),
    }
    selected_auxiliary = {
        name: np.asarray(supplied_auxiliary[name], dtype=np.float64)[selected_indices]
        for name in expected_auxiliary
    }
    scalar, names = build_independent_scalar_features(
        evidence, graph, encoder_scores, "noise_v1", unavailable_chemistry,
        auxiliary_scores=selected_auxiliary,
        include_reliability_context=bool(payload.get("reliability_context", False)),
    )
    if list(names) != list(payload["scalar_feature_names"]):
        raise RuntimeError("application scalar feature contract differs from selected model")
    mean = np.asarray(payload["scalar_mean"], dtype=np.float32)
    scale = np.asarray(payload["scalar_scale"], dtype=np.float32)

    entity_ids = sorted(
        set(selected_pairs["query_id"].astype(str))
        | set(selected_pairs["candidate_id"].astype(str))
    )
    entity_index = {entity: index for index, entity in enumerate(entity_ids)}
    banks = {
        name: torch.as_tensor(
            np.stack([caches[name][entity] for entity in entity_ids]).astype(np.float32),
            device=device,
        )
        for name in encoder_names
    }
    query_rows = np.asarray(
        [entity_index[value] for value in selected_pairs["query_id"].astype(str)], dtype=np.int64
    )
    candidate_rows = np.asarray(
        [entity_index[value] for value in selected_pairs["candidate_id"].astype(str)], dtype=np.int64
    )
    reference_rows = np.repeat(candidate_rows[:, None], len(encoder_names), axis=1)
    output = np.empty(len(selected_pairs), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(selected_pairs), 8192):
            right = min(left + 8192, len(selected_pairs))
            scalar_tensor = torch.as_tensor((scalar[left:right] - mean) / scale, device=device)
            output[left:right] = model(
                scalar_tensor,
                torch.as_tensor(query_rows[left:right], device=device),
                torch.as_tensor(reference_rows[left:right], device=device),
                banks,
            ).float().cpu().numpy()
    full_output[selected_indices] = output
    return full_output, {
        "model_path": str(model_path), "model_sha256": sha256(model_path),
        "arm": str(payload["arm"]), "encoder_names": list(encoder_names),
        "auxiliary_family_names": list(expected_auxiliary),
        "unused_supplied_auxiliary_family_names": sorted(
            set(supplied_auxiliary) - set(expected_auxiliary)
        ),
        "selection_class": str(payload["selection_class"]),
        "representation_increment_verdict": str(
            payload["representation_increment_verdict"]
        ),
        "fixed_axis_queries": int(len(query_ids)),
        "zero_candidate_queries": int(np.sum(np.diff(query_ptr) == 0)),
        "scored_pairs": int(len(selected_pairs)),
        "scored_queries": int(len(selected_queries)),
        "incomplete_nonempty_queries": int(
            np.sum(np.diff(query_ptr) > 0) - len(selected_queries)
        ),
        "query_key": "query_id|candidate_study_id",
        "missing_rule": "score only complete fixed-axis composite query blocks; never shrink a query selectively",
        "chemaware_v2": "unavailable_without_candidate_structures_exact_zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", action="append", required=True,
                        help="Repeated METHOD=NPZ input")
    parser.add_argument(
        "--baseline-embedding", action="append", default=[],
        help="Repeated METHOD=NPZ comparator; excluded from simple/unified encoder fusion",
    )
    parser.add_argument(
        "--identity-handoff", type=Path, default=DEFAULT_IDENTITY_HANDOFF,
        help="Frozen complete-identity handoff; this is the sole candidate axis",
    )
    parser.add_argument(
        "--auxiliary-score", action="append", default=[],
        help="Optional frozen expert score as NAME=PATH::COLUMN",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument(
        "--msg-fusion-model", type=Path,
        help="Frozen Noise-MSG fusion_model directory (or fusion_model.joblib)",
    )
    parser.add_argument("--unified-model", type=Path)
    parser.add_argument(
        "--enveda-integrator-arm", choices=ENVEDA_INTEGRATOR_ARMS,
        help=(
            "Explicit arm from an Enveda candidate-integrator checkpoint. "
            "Omit to preserve the checkpoint's predeclared primary arm."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    pairs, queries, query_ptr, query_ids, axis_provenance = load_fixed_identity_axis(
        args.identity_handoff,
        expected_queries=PRODUCTION_QUERY_COUNT,
        expected_candidates=PRODUCTION_CANDIDATE_COUNT,
    )
    auxiliary_scores, auxiliary_provenance = load_auxiliary_candidate_scores(
        args.auxiliary_score, pairs["candidate_pair_id"].astype(str).to_numpy(dtype="U"),
    )

    methods: list[str] = []
    caches: dict[str, dict[str, np.ndarray]] = {}
    cache_paths: dict[str, Path] = {}
    for item in args.embedding:
        if "=" not in item:
            raise ValueError(f"invalid --embedding value: {item}")
        method, raw_path = item.split("=", 1)
        if not method or method in caches:
            raise ValueError(f"duplicate or empty method: {method!r}")
        path = Path(raw_path)
        caches[method] = load_embedding(path)
        cache_paths[method] = path
        methods.append(method)

    baseline_methods: list[str] = []
    baseline_caches: dict[str, dict[str, np.ndarray]] = {}
    baseline_cache_paths: dict[str, Path] = {}
    for item in args.baseline_embedding:
        if "=" not in item:
            raise ValueError(f"invalid --baseline-embedding value: {item}")
        method, raw_path = item.split("=", 1)
        if not method or method in caches or method in baseline_caches:
            raise ValueError(f"duplicate or empty baseline method: {method!r}")
        path = Path(raw_path)
        baseline_caches[method] = load_embedding(path)
        baseline_cache_paths[method] = path
        baseline_methods.append(method)

    mgfs = {NEW_STUDY: NEW_MGF, **CHEMPROP_MGF}
    spectra: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    for study, path in mgfs.items():
        ids, records = load_mgf(path, minimum_peaks=0, skip_ineligible=False)
        spectra[study] = {
            str(entity): (float(record["precursor_mz"]), peak_rows(record["peaks"]))
            for entity, record in zip(ids, records, strict=True)
        }

    # Candidate scores are computed only after the exact 22,046-row handoff is
    # loaded and ordered.  No spectrum or embedding can add a candidate.
    query_entities = pairs["query_id"].astype(str).to_numpy()
    candidate_entities = pairs["candidate_id"].astype(str).to_numpy()
    scoring_caches = {**caches, **baseline_caches}
    for name in [*methods, *baseline_methods]:
        values = np.full(len(pairs), np.nan, dtype=np.float32)
        cache = scoring_caches[name]
        for index, (query_id, candidate_id) in enumerate(
            zip(query_entities, candidate_entities, strict=True)
        ):
            if query_id in cache and candidate_id in cache:
                values[index] = float(np.dot(cache[query_id], cache[candidate_id]))
        pairs[f"{name}_score"] = values

    sqrt_cosine = np.full(len(pairs), np.nan, dtype=np.float32)
    entropy = np.full(len(pairs), np.nan, dtype=np.float32)
    neutral_loss = np.full(len(pairs), np.nan, dtype=np.float32)
    for index, row in enumerate(pairs.itertuples(index=False)):
        try:
            query_peaks = spectra[str(row.query_study_id)][str(row.query_raw_entity_id)][1]
            candidate_peaks = spectra[str(row.candidate_study_id)][str(row.candidate_raw_entity_id)][1]
        except KeyError as error:
            raise RuntimeError(
                f"fixed candidate spectrum is absent at candidate_pair_id={row.candidate_pair_id}"
            ) from error
        values = frozen_p2b_pair_features(
            prepare_spectrum(query_peaks.T), float(row.query_precursor_mz),
            prepare_spectrum(candidate_peaks.T), float(row.candidate_precursor_mz),
            tolerance=args.fragment_da,
        )
        sqrt_cosine[index], entropy[index], neutral_loss[index] = values
    pairs["P2b_sqrt_cosine"] = sqrt_cosine
    pairs["P2b_entropy"] = entropy
    pairs["P2b_neutral_loss"] = neutral_loss

    if "noise_v1" not in methods:
        raise RuntimeError("cross-study unified evidence requires noise_v1 embeddings")
    p2b = np.full(len(pairs), np.nan, dtype=np.float32)
    simple = np.full(len(pairs), np.nan, dtype=np.float64)
    for start, stop in zip(query_ptr[:-1], query_ptr[1:]):
        start, stop = int(start), int(stop)
        if stop == start:
            continue
        noise = pd.to_numeric(
            pairs.iloc[start:stop]["noise_v1_score"], errors="coerce"
        ).to_numpy(np.float64)
        p2b_inputs = np.column_stack((
            noise, sqrt_cosine[start:stop], entropy[start:stop], neutral_loss[start:stop],
        ))
        if np.isfinite(p2b_inputs).all():
            p2b[start:stop] = frozen_p2b_block(
                noise, sqrt_cosine[start:stop], entropy[start:stop], neutral_loss[start:stop],
            )
        components = [
            pd.to_numeric(pairs.iloc[start:stop]["WSE_score"], errors="coerce").to_numpy(float),
            *[
                pd.to_numeric(
                    pairs.iloc[start:stop][f"{name}_score"], errors="coerce"
                ).to_numpy(float)
                for name in methods
            ],
        ]
        if np.isfinite(np.stack(components)).all():
            simple[start:stop] = np.mean(
                np.stack([minmax(values) for values in components]), axis=0
            )
    pairs["P2b_score"] = p2b
    pairs["simple_fusion_score"] = simple
    msg_provenance = None
    if args.msg_fusion_model is not None:
        if MSG_FAMILY in auxiliary_scores:
            raise RuntimeError(
                f"{MSG_FAMILY} was supplied twice (generic auxiliary and frozen model)"
            )
        msg_scores, msg_provenance = score_frozen_noise_msg(
            pairs, query_ptr, args.msg_fusion_model,
        )
        auxiliary_scores[MSG_FAMILY] = msg_scores
        auxiliary_provenance[MSG_FAMILY] = msg_provenance
    for name, values in auxiliary_scores.items():
        pairs[f"auxiliary__{name}_score"] = values

    unified_provenance = None
    if args.unified_model is not None and not pairs.empty:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("unified application inference requested CUDA without an allocated GPU")
        pairs["unified_candidate_ranker_score"], unified_provenance = apply_unified_model(
            pairs, caches, args.unified_model, device, query_ptr, query_ids,
            auxiliary_scores=auxiliary_scores,
            enveda_integrator_arm=args.enveda_integrator_arm,
        )
    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    pair_path = args.output_dir / "identity_candidate_scores.tsv.gz"
    long_path = args.output_dir / "method_scores_long.tsv.gz"
    query_path = args.output_dir / "directional_query_registry.tsv.gz"
    pairs.to_csv(pair_path, sep="\t", index=False, compression=compression)
    queries.to_csv(query_path, sep="\t", index=False, compression=compression)
    long_rows: list[pd.DataFrame] = []
    score_columns = {
        "weighted_spectral_entropy": "WSE_score",
        "p2b_noise_v1_frozen": "P2b_score",
        **{name: f"{name}_score" for name in methods},
        **{name: f"{name}_score" for name in baseline_methods},
        **{name: f"auxiliary__{name}_score" for name in auxiliary_scores},
        "simple_query_minmax_mean": "simple_fusion_score",
    }
    if args.unified_model is not None:
        score_columns["unified_candidate_ranker"] = "unified_candidate_ranker_score"
    for method, column in score_columns.items():
        block = pairs[[
            "candidate_pair_id", "query_block_id", "query_id", "candidate_id",
            "query_study_id", "candidate_study_id",
        ]].copy()
        block["method_name"] = method
        block["method_score"] = pd.to_numeric(pairs[column], errors="coerce")
        if method == "weighted_spectral_entropy":
            block["evidence_available"] = pairs["WSE_available"].to_numpy(bool)
        else:
            block["evidence_available"] = block["method_score"].notna()
        long_rows.append(block)
    long_table = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=["candidate_pair_id", "query_id", "candidate_id", "method_name",
                 "method_score", "evidence_available"]
    )
    long_table.to_csv(long_path, sep="\t", index=False, compression=compression)
    report = {
        "status": "CROSSSTUDY_IDENTITY_MULTIENCODER_COMPLETE",
        "labels_or_responses_accessed": False,
        "candidate_admission": "frozen complete-identity handoff only; no application-side candidate generation",
        "methods": list(score_columns),
        "weighted_entropy_backend": "canonical_handoff_wse_score_no_recomputation",
        "parameters": {"fragment_da_for_p2b_only": args.fragment_da},
        "denominators": {
            "directional_queries": int(len(queries)),
            "queries_with_identity_candidates": int(queries["candidate_count"].gt(0).sum()),
            "queries_without_identity_candidates": int(queries["candidate_count"].eq(0).sum()),
            "queries_with_one_identity_candidate": int(queries["candidate_count"].eq(1).sum()),
            "queries_with_multiple_identity_candidates": int(queries["candidate_count"].gt(1).sum()),
            "identity_candidate_pairs": int(len(pairs)),
        },
        "inputs": {
            "fixed_identity_axis": axis_provenance,
            "embeddings": {name: {"path": str(cache_paths[name]), "sha256": sha256(cache_paths[name])}
                           for name in methods},
            "baseline_embeddings": {
                name: {
                    "path": str(baseline_cache_paths[name]),
                    "sha256": sha256(baseline_cache_paths[name]),
                    "role": "independent_comparator_not_a_simple_fusion_encoder",
                }
                for name in baseline_methods
            },
            "auxiliary_scores": auxiliary_provenance,
            "noise_msg_composite": msg_provenance,
            "unified_model": unified_provenance,
            "mgfs": {study: {"path": str(path), "sha256": sha256(path)} for study, path in mgfs.items()},
        },
        "outputs": {
            "candidate_scores": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "method_scores_long": {"path": str(long_path), "sha256": sha256(long_path)},
            "query_registry": {"path": str(query_path), "sha256": sha256(query_path)},
        },
        "claim_boundary": (
            "Fixed same-observed-precursor identity hypotheses only. Zero and singleton blocks "
            "remain explicit; a finite singleton score is not a calibrated identity acceptance. "
            "Out-of-window chemical relationships, BioAware context and structure-dependent V2 "
            "evidence are separate optional tracks. In this application V2 is an exact unavailable "
            "no-op (availability=0 and all continuous quantities=0), and BioAware is not fabricated."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

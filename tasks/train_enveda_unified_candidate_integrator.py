#!/usr/bin/env python
"""Cross-fit the unified candidate-evidence integrator on consumed Enveda.

This is meta-development, not a new external test.  Every upstream encoder and
reranker is frozen before this program starts.  The two Enveda primary panels
are kept as separate candidate ledgers, while a global molecular-formula fold
assignment prevents a formula (and therefore any repeated query occurrence)
from crossing train/evaluation folds.  A query repeated across panels receives
total weight one.

The predeclared primary is one small nonlinear reliability head.  A linear
additive head with identical scalar inputs is a capacity-lower comparator,
not a capacity-matched mechanism proof; the OOF table is not used to choose
between architectures.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from enveda_unified_candidate_integrator_core import (
    ARMS,
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    PRIMARY_ARM,
    CandidateEvidenceIntegrator,
    IntegratorSettings,
    candidate_row_weights,
    fit_weighted_normalization,
    formula_fold,
    occurrence_weights,
    transform_features,
    validate_global_query_formula,
    weighted_listwise_loss,
    weighted_query_residual_penalty,
)
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from gnps_pair_score_cache import load_pair_score_cache, sha256_file
from massspecgym_embedding_readout_core import build_independent_scalar_features
from noise_corrected_fullgraph_evaluation import score_embeddings


PANELS = ("identity_disjoint", "formula_disjoint")
ENCODERS = (
    "noise_v1",
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
)
SCORE_METHODS = (
    "official_dreams",
    "noise_v1",
    "weighted_spectral_entropy",
    "p2b_noise_v1_frozen",
)
WSE = "weighted_spectral_entropy"
MSG = "noise_msg_fusion_stack_v1"


@dataclass
class PanelData:
    name: str
    graph: object
    scalar: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray
    query_ptr: np.ndarray
    formulas: np.ndarray
    query_keys: np.ndarray
    near: np.ndarray
    query_weights: np.ndarray
    scores: dict[str, np.ndarray]


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw = value.split("=", 1)
    name, raw = name.strip(), raw.strip()
    if not name or not raw:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--msg-cache", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--fullmsg-embeddings", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument(
        "--comparator-embedding", action="append", type=named_path, default=[],
        metavar="NAME=PATH", help="Embedding comparator; never enters fitted features.",
    )
    parser.add_argument(
        "--comparator-score", action="append", type=named_path, default=[],
        metavar="NAME=PATH", help="Pair/molecule score comparator; never enters fitted features.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-strength", type=float, default=1e-2)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _load_panel_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {name: np.asarray(body[name]) for name in body.files}


def _query_keys(raw: dict[str, np.ndarray]) -> np.ndarray:
    rows = np.asarray(raw["query_row"], dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise RuntimeError("query rows must be unique within each panel")
    # Both panels index the same benchmark manifest.  Source row is therefore
    # the strongest occurrence identity; IK14/adduct consistency is checked
    # separately below rather than being allowed to split one row into two.
    return np.asarray([f"row={int(row)}" for row in rows], dtype="U32")


def _validate_query_descriptors(raw_panels: dict[str, dict[str, np.ndarray]]) -> None:
    observed: dict[int, tuple[str, str]] = {}
    for name, raw in raw_panels.items():
        rows = np.asarray(raw["query_row"], dtype=np.int64)
        identities = np.asarray(raw["query_ik14"]).astype("U")
        adducts = np.asarray(
            raw.get("query_adduct", np.full(len(rows), "", dtype="U1")),
        ).astype("U")
        if not (rows.shape == identities.shape == adducts.shape):
            raise RuntimeError(f"query descriptor axis is malformed in {name}")
        for row, identity, adduct in zip(rows, identities, adducts):
            descriptor = (str(identity), str(adduct))
            previous = observed.setdefault(int(row), descriptor)
            if previous != descriptor:
                raise RuntimeError(
                    f"query row {int(row)} changed IK14/adduct across panels: "
                    f"{previous!r} != {descriptor!r}"
                )


def _aggregate_pair(values: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if values.ndim != 1 or len(values) != int(ptr[-1]) or np.any(np.diff(ptr) < 1):
        raise RuntimeError("pair score does not align to non-empty molecule blocks")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("pair score is non-finite")
    return np.maximum.reduceat(values, ptr[:-1]).astype(np.float32)


def _load_chemistry(path: Path, raw: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        chem = {field: np.asarray(body[field]) for field in body.files}
    required = {
        "schema", "query_ptr", "query_ids", "candidate_ids", "query_formula",
        "near_query", "availability", "utility_correct", "utility_zero",
        "utility_reversed", "utility_candidate_mismatch",
    }
    missing = required - chem.keys()
    if missing:
        raise RuntimeError(f"ChemAware evidence misses {sorted(missing)}: {path}")
    if str(np.asarray(chem["schema"]).item()) != "gnps_chemaware_continuous_null_evidence_v1":
        raise RuntimeError(f"unsupported continuous ChemAware schema: {path}")
    checks = {
        "query_ptr": np.array_equal(chem["query_ptr"], raw["query_ptr"]),
        "query_ids": np.array_equal(chem["query_ids"].astype("U"), raw["query_ik14"].astype("U")),
        "candidate_ids": np.array_equal(
            chem["candidate_ids"].astype("U"), raw["molecule_ik14"].astype("U")
        ),
        "query_formula": np.array_equal(
            chem["query_formula"].astype("U"), raw["query_formula"].astype("U")
        ),
        "near": np.array_equal(chem["near_query"].astype(bool), raw["near_query"].astype(bool)),
    }
    failed = [field for field, valid in checks.items() if not valid]
    if failed:
        raise RuntimeError(f"ChemAware candidate axis drifted for {name}: {failed}")
    available = np.asarray(chem["availability"], dtype=bool)
    count = len(raw["molecule_label"])
    if available.shape != (count,):
        raise RuntimeError(f"ChemAware availability shape drifted for {name}")
    output = {
        "chem_valid": available,
        "candidate_utility_correct": np.asarray(chem["utility_correct"], dtype=np.float32),
        "candidate_utility_zero": np.asarray(chem["utility_zero"], dtype=np.float32),
        "candidate_utility_reversed": np.asarray(chem["utility_reversed"], dtype=np.float32),
        "candidate_utility_rotated": np.asarray(
            chem["utility_candidate_mismatch"], dtype=np.float32,
        ),
    }
    if any(value.shape != (count,) for value in output.values()):
        raise RuntimeError(f"ChemAware continuous utility shape drifted for {name}")
    for field in output:
        if field != "chem_valid" and not np.all(np.isfinite(output[field][available])):
            raise RuntimeError(f"ChemAware {field} is non-finite for {name}")
    return output


def _load_score_bundle(
    path: Path, benchmark: Path,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as body:
        methods = tuple(map(str, body["method_names"]))
        scores = {name: np.asarray(body[f"scores_{name}"], dtype=np.float32) for name in PANELS}
    missing = [name for name in SCORE_METHODS if name not in methods]
    if missing:
        raise RuntimeError(f"score bundle misses required methods: {missing}")
    expected_rows: dict[str, int] = {}
    for name in PANELS:
        with np.load(benchmark / f"panel_{name}.npz", allow_pickle=False) as panel:
            expected_rows[name] = int(len(panel["candidate_row"]))
        if scores[name].shape != (len(methods), expected_rows[name]):
            raise RuntimeError(
                f"score bundle {name} shape {scores[name].shape} does not match "
                f"{len(methods)} methods and {expected_rows[name]} frozen edges"
            )
    metadata_path = path.with_suffix(path.suffix + ".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        declared_benchmark = metadata.get("benchmark")
        if declared_benchmark is not None and Path(declared_benchmark) != benchmark:
            raise RuntimeError(
                f"score bundle benchmark drifted: {declared_benchmark} != {benchmark}"
            )
        declared_rows = metadata.get("score_rows")
        if declared_rows is not None and any(
            declared_rows.get(name) != count for name, count in expected_rows.items()
        ):
            raise RuntimeError(
                f"score bundle sidecar edge axis drifted: {declared_rows} != {expected_rows}"
            )
    return methods, scores


def _load_embedding_banks(
    args: argparse.Namespace, manifest_rows: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    model_paths = {
        "chemaware_stage1": args.stage1_embeddings,
        "chemaware_phasea": args.phasea_embeddings,
        "noise_chemaware_fullmsg": args.fullmsg_embeddings,
    }
    model_banks = {
        name: load_embeddings(path, manifest_rows) for name, path in model_paths.items()
    }
    comparators: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, path in args.comparator_embedding:
        if name in comparators or name in model_paths or name in SCORE_METHODS:
            raise ValueError(f"duplicate or reserved comparator name: {name}")
        comparators[name] = load_embeddings(path, manifest_rows)
    return model_banks, comparators


def _candidate_score_from_archive(
    archive: Path, name: str, panel: str, raw: dict[str, np.ndarray],
    edge_count: int, molecule_ptr: np.ndarray,
) -> np.ndarray:
    with np.load(archive, allow_pickle=False) as body:
        aliases = (panel, panel.replace("_disjoint", ""))
        prefix = panel.replace("_disjoint", "")
        # A same-length score vector is not sufficient evidence of alignment.
        # Standalone comparator archives must carry the complete frozen
        # candidate axis and match it exactly before any score is accepted.
        axis = {
            f"{prefix}__query_ptr": np.asarray(raw["query_ptr"]),
            f"{prefix}__labels": np.asarray(raw["molecule_label"]),
            f"{prefix}__query_ids": np.asarray(raw["query_ik14"]).astype("U"),
            f"{prefix}__candidate_ids": np.asarray(raw["molecule_ik14"]).astype("U"),
            f"{prefix}__formula": np.asarray(raw["query_formula"]).astype("U"),
            f"{prefix}__near": np.asarray(raw["near_query"], dtype=bool),
        }
        missing_axis = [key for key in axis if key not in body.files]
        if missing_axis:
            raise RuntimeError(
                f"comparator {name}/{panel} lacks frozen axis fields in {archive}: "
                f"{missing_axis}"
            )
        drifted = []
        for key, expected in axis.items():
            observed = np.asarray(body[key])
            if expected.dtype.kind in "USO":
                observed = observed.astype("U")
            elif expected.dtype == np.dtype(bool):
                observed = observed.astype(bool)
            if not np.array_equal(observed, expected):
                drifted.append(key)
        if drifted:
            raise RuntimeError(
                f"comparator {name}/{panel} candidate axis drifted in {archive}: {drifted}"
            )
        candidates: list[str] = []
        for alias in aliases:
            candidates.extend((
                f"{alias}__{name}", f"{alias}__B", f"molecule_scores_{alias}",
                f"pair_scores_{alias}", f"scores_{alias}", alias,
            ))
        key = next((key for key in candidates if key in body.files), None)
        if key is None:
            raise RuntimeError(
                f"comparator {name} has no recognized {panel} score in {archive}; "
                f"available={body.files}"
            )
        value = np.asarray(body[key], dtype=np.float32)
    molecule_count = len(molecule_ptr) - 1
    if value.shape == (edge_count,):
        return _aggregate_pair(value, molecule_ptr)
    if value.shape == (molecule_count,):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite comparator {name}/{panel}")
        return value
    raise RuntimeError(
        f"comparator {name}/{panel} shape {value.shape} is neither pair nor molecule axis"
    )


def _load_score_comparators(
    specs: list[tuple[str, Path]], benchmark: Path, graphs: dict[str, object],
    raw_panels: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {panel: {} for panel in PANELS}
    names: set[str] = set()
    for name, path in specs:
        if name in names or name in SCORE_METHODS or name in ENCODERS or name == MSG:
            raise ValueError(f"duplicate or reserved comparator name: {name}")
        names.add(name)
        archive = path
        if path.is_dir() and (path / "candidate_predictions.npz").is_file():
            archive = path / "candidate_predictions.npz"
        if path.is_dir() and (path / "report.json").is_file() and not archive.is_file():
            _, pair_scores = load_pair_score_cache(path, benchmark)
            for panel in PANELS:
                result[panel][name] = _aggregate_pair(
                    pair_scores[panel], graphs[panel].molecule_ptr,
                )
            continue
        if not archive.is_file():
            raise FileNotFoundError(path)
        for panel in PANELS:
            result[panel][name] = _candidate_score_from_archive(
                archive, name, panel, raw_panels[panel],
                len(graphs[panel].pair_candidate_row),
                graphs[panel].molecule_ptr,
            )
    return result


def _make_panels(args: argparse.Namespace) -> tuple[dict[str, PanelData], dict[str, Path]]:
    panel_paths = {name: args.benchmark / f"panel_{name}.npz" for name in PANELS}
    required = [
        args.benchmark / "manifest.csv.gz", args.score_bundle, args.msg_cache / "report.json",
        args.stage1_embeddings, args.phasea_embeddings, args.fullmsg_embeddings,
        args.chem_dir / "report.json", *panel_paths.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"unified integrator misses inputs: {missing}")
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    manifest_rows = len(manifest)
    if not np.array_equal(manifest["row"].to_numpy(np.int64), np.arange(manifest_rows)):
        raise RuntimeError("benchmark manifest row registry is not contiguous")
    methods, method_scores = _load_score_bundle(args.score_bundle, args.benchmark)
    _, msg_pairs = load_pair_score_cache(args.msg_cache, args.benchmark)
    model_banks, embedding_comparators = _load_embedding_banks(args, manifest_rows)
    graphs = {name: graph_from_panel(path) for name, path in panel_paths.items()}
    raw_panels = {name: _load_panel_raw(path) for name, path in panel_paths.items()}
    _validate_query_descriptors(raw_panels)
    score_comparators = _load_score_comparators(
        args.comparator_score, args.benchmark, graphs, raw_panels,
    )

    provisional: dict[str, dict] = {}
    for name in PANELS:
        graph = graphs[name]
        raw = raw_panels[name]
        pair_matrix = method_scores[name]
        if pair_matrix.shape != (len(methods), len(graph.pair_candidate_row)):
            raise RuntimeError(f"score-bundle edge axis drifted for {name}")
        pairs = {method: pair_matrix[methods.index(method)] for method in SCORE_METHODS}
        molecule = {
            method: _aggregate_pair(values, graph.molecule_ptr)
            for method, values in pairs.items()
        }
        msg_molecule = _aggregate_pair(msg_pairs[name], graph.molecule_ptr)
        encoder_scores: dict[str, np.ndarray] = {"noise_v1": molecule["noise_v1"]}
        for encoder, (rows, embeddings) in model_banks.items():
            encoder_scores[encoder] = score_embeddings(graph, rows, embeddings).molecule
        chemical = _load_chemistry(
            args.chem_dir / f"continuous_chem_{name}.npz", raw, name,
        )
        evidence = {
            "v1_cosine": pairs["noise_v1"],
            "weighted_entropy": pairs[WSE],
            "p2b_fused": pairs["p2b_noise_v1_frozen"],
        }
        scalar, feature_names = build_independent_scalar_features(
            evidence, graph, encoder_scores, "noise_v1", chemical,
            auxiliary_scores={MSG: msg_molecule},
            include_reliability_context=True,
        )
        labels = np.asarray(raw["molecule_label"], dtype=np.int8)
        ptr = np.asarray(raw["query_ptr"], dtype=np.int64)
        # Formula strings are upstream-normalized; this trainer performs only
        # the declared whitespace trim before global grouping.
        formulas = np.char.strip(np.asarray(raw["query_formula"]).astype("U"))
        if np.any(formulas == ""):
            raise RuntimeError(f"empty normalized formula in {name}")
        near = np.asarray(raw["near_query"], dtype=bool)
        if (
            scalar.shape[0] != len(labels)
            or int(ptr[-1]) != len(labels)
            or len(formulas) != len(ptr) - 1
            or np.any(np.diff(ptr) < 2)
        ):
            raise RuntimeError(f"candidate/query axis drifted for {name}")
        if any(np.sum(labels[left:right] > 0) != 1 for left, right in zip(ptr[:-1], ptr[1:])):
            raise RuntimeError(f"every {name} query must have exactly one positive")
        all_scores = {
            **molecule,
            MSG: msg_molecule,
            "chemaware_stage1": encoder_scores["chemaware_stage1"],
            "chemaware_phasea": encoder_scores["chemaware_phasea"],
            "noise_chemaware_fullmsg": encoder_scores["noise_chemaware_fullmsg"],
        }
        for comparator, (rows, embeddings) in embedding_comparators.items():
            all_scores[comparator] = score_embeddings(graph, rows, embeddings).molecule
        all_scores.update(score_comparators[name])
        provisional[name] = {
            "graph": graph, "scalar": scalar, "feature_names": tuple(feature_names),
            "labels": labels, "query_ptr": ptr, "formulas": formulas,
            "query_keys": _query_keys(raw), "near": near, "scores": all_scores,
        }

    feature_contracts = {data["feature_names"] for data in provisional.values()}
    if len(feature_contracts) != 1:
        raise RuntimeError("scalar feature contract differs between Enveda panels")
    keys = {name: data["query_keys"] for name, data in provisional.items()}
    formulas = {name: data["formulas"] for name, data in provisional.items()}
    validate_global_query_formula(keys, formulas)
    weights = occurrence_weights(keys)
    panels = {
        name: PanelData(name=name, query_weights=weights[name], **data)
        for name, data in provisional.items()
    }
    return panels, panel_paths


def _occurrences(panels: dict[str, PanelData]) -> list[tuple[str, int]]:
    return [
        (name, query)
        for name in PANELS
        for query in range(len(panels[name].query_ptr) - 1)
    ]


def _group_occurrences(
    panels: dict[str, PanelData], occurrences: list[tuple[str, int]],
) -> list[list[tuple[str, int]]]:
    """Keep all cross-panel occurrences of one source query indivisible."""
    grouped: dict[str, list[tuple[str, int]]] = {}
    for occurrence in occurrences:
        name, query = occurrence
        key = str(panels[name].query_keys[query])
        grouped.setdefault(key, []).append(occurrence)
    for key, group in grouped.items():
        mass = sum(float(panels[name].query_weights[query]) for name, query in group)
        if not np.isclose(mass, 1.0, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"unique query {key} has training mass {mass}, not one")
    return list(grouped.values())


def _fit_normalization(
    panels: dict[str, PanelData], occurrences: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    by_panel: dict[str, list[int]] = {name: [] for name in PANELS}
    for panel, query in occurrences:
        by_panel[panel].append(query)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for name in PANELS:
        if not by_panel[name]:
            continue
        panel = panels[name]
        rows, weights = candidate_row_weights(
            panel.query_ptr, np.asarray(by_panel[name], dtype=np.int64), panel.query_weights,
        )
        blocks.append((panel.scalar[rows], weights))
    return fit_weighted_normalization(blocks, panels[PANELS[0]].feature_names)


def _pack(
    panels: dict[str, PanelData], occurrences: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    weights: list[float] = []
    lengths: list[int] = []
    for name, query in occurrences:
        panel = panels[name]
        left, right = int(panel.query_ptr[query]), int(panel.query_ptr[query + 1])
        feature_blocks.append(panel.scalar[left:right])
        label_blocks.append(panel.labels[left:right])
        weights.append(float(panel.query_weights[query]))
        lengths.append(right - left)
    raw = np.concatenate(feature_blocks) if feature_blocks else np.empty((0, len(mean)))
    labels = np.concatenate(label_blocks) if label_blocks else np.asarray([], dtype=np.int8)
    ptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return (
        transform_features(raw, mean, scale), labels, ptr,
        np.asarray(weights, dtype=np.float32),
    )


def _fit_arm(
    arm: str, panels: dict[str, PanelData], occurrences: list[tuple[str, int]],
    mean: np.ndarray, scale: np.ndarray, args: argparse.Namespace, seed: int,
) -> tuple[CandidateEvidenceIntegrator, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    names = panels[PANELS[0]].feature_names
    # Preserve continuous WSE matching strength as the ranking anchor.  Its
    # query percentile remains an auxiliary feature, never the base score.
    anchor_name = WSE
    if anchor_name not in names:
        raise RuntimeError(f"WSE anchor feature is missing: {anchor_name}")
    settings = IntegratorSettings(len(names), names.index(anchor_name), args.hidden)
    model = CandidateEvidenceIntegrator(arm, settings).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    groups = _group_occurrences(panels, occurrences)
    history: list[float] = []
    for _ in range(args.epochs):
        order = rng.permutation(len(groups))
        total, mass = 0.0, 0.0
        for start in range(0, len(order), args.batch_queries):
            chosen = [
                occurrence
                for index in order[start:start + args.batch_queries]
                for occurrence in groups[index]
            ]
            features, labels, ptr, weights = _pack(panels, chosen, mean, scale)
            tensor = torch.as_tensor(features, dtype=torch.float32, device=args.device)
            logits = model(tensor)
            loss = weighted_listwise_loss(logits, ptr, labels, weights)
            residual = model.residual(tensor)
            loss = loss + args.anchor_strength * weighted_query_residual_penalty(
                residual, ptr, weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_mass = float(np.sum(weights))
            total += float(loss.detach().cpu()) * batch_mass
            mass += batch_mass
        history.append(total / mass)
    return model, history


@torch.inference_mode()
def _predict_occurrences(
    model: CandidateEvidenceIntegrator,
    panels: dict[str, PanelData],
    occurrences: list[tuple[str, int]],
    mean: np.ndarray,
    scale: np.ndarray,
    device: str,
) -> dict[tuple[str, int], np.ndarray]:
    output: dict[tuple[str, int], np.ndarray] = {}
    batch_size = 512
    for start in range(0, len(occurrences), batch_size):
        batch = occurrences[start:start + batch_size]
        features, _, ptr, _ = _pack(panels, batch, mean, scale)
        logits = model(torch.as_tensor(features, dtype=torch.float32, device=device))
        values = logits.cpu().numpy().astype(np.float32)
        for index, occurrence in enumerate(batch):
            output[occurrence] = values[int(ptr[index]):int(ptr[index + 1])]
    return output


def _strict_ranks(scores: np.ndarray, panel: PanelData) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != panel.labels.shape or not np.all(np.isfinite(values)):
        raise RuntimeError(f"invalid candidate score for {panel.name}")
    ranks = []
    for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
        left, right = int(left), int(right)
        positive = np.flatnonzero(panel.labels[left:right] > 0)
        if len(positive) != 1:
            raise RuntimeError("query does not have one positive")
        target = float(values[left + positive[0]])
        # Strict policy: every tied candidate counts ahead of the positive,
        # including the positive itself as rank one.
        ranks.append(int(np.sum(values[left:right] >= target)))
    return np.asarray(ranks, dtype=np.int32)


def _metric_summary(scores: np.ndarray, panel: PanelData) -> tuple[dict, np.ndarray]:
    ranks = _strict_ranks(scores, panel)
    auc, margins = [], []
    for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
        left, right = int(left), int(right)
        local_label = panel.labels[left:right]
        local = np.asarray(scores[left:right], dtype=np.float64)
        positive = float(local[np.flatnonzero(local_label > 0)[0]])
        negative = local[local_label == 0]
        auc.append(float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative)))
        ordered = np.sort(local)[::-1]
        margins.append(float(ordered[0] - ordered[1]))
    result = {
        "queries": int(len(ranks)),
        **{f"recall@{cutoff}": float(np.mean(ranks <= cutoff)) for cutoff in (1, 2, 3, 5, 10, 20)},
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "macro_query_auroc": float(np.mean(auc)),
        "macro_query_auprc": float(np.mean(1.0 / ranks)),
        "micro_candidate_auroc": float(roc_auc_score(panel.labels, scores)),
        "micro_candidate_auprc": float(average_precision_score(panel.labels, scores)),
        "mean_top1_top2_margin": float(np.mean(margins)),
    }
    if np.any(panel.near):
        near = ranks[panel.near]
        result["near"] = {
            "queries": int(len(near)),
            "recall@1": float(np.mean(near == 1)),
            "mrr": float(np.mean(1.0 / near)),
        }
    else:
        result["near"] = {"queries": 0}
    return result, ranks


def _formula_ci(
    delta: np.ndarray, formulas: np.ndarray, resamples: int, seed: int,
) -> dict:
    formulas = np.asarray(formulas).astype("U")
    delta = np.asarray(delta, dtype=np.float64)
    unique = np.unique(formulas)
    sums = np.asarray([np.sum(delta[formulas == value]) for value in unique])
    counts = np.asarray([np.sum(formulas == value) for value in unique])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        bootstrap[index] = np.sum(sums[chosen]) / np.sum(counts[chosen])
    return {
        "delta": float(np.mean(delta)),
        "delta_pp": float(100 * np.mean(delta)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "ci_low_pp": float(100 * np.quantile(bootstrap, 0.025)),
        "ci_high_pp": float(100 * np.quantile(bootstrap, 0.975)),
        "formula_clusters": int(len(unique)),
        "bootstrap_resamples": int(resamples),
    }


def _comparison(
    candidate: np.ndarray, baseline: np.ndarray, panel: PanelData,
    resamples: int, seed: int,
) -> dict:
    candidate_rank = _strict_ranks(candidate, panel)
    baseline_rank = _strict_ranks(baseline, panel)
    candidate_ok, baseline_ok = candidate_rank == 1, baseline_rank == 1
    delta = candidate_ok.astype(float) - baseline_ok.astype(float)
    corrected = int(np.sum(candidate_ok & ~baseline_ok))
    introduced = int(np.sum(~candidate_ok & baseline_ok))
    return {
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "recall1_formula_cluster_ci": _formula_ci(delta, panel.formulas, resamples, seed),
        "mrr_delta": float(np.mean(1.0 / candidate_rank - 1.0 / baseline_rank)),
    }


def _joint_query_balanced(
    panels: dict[str, PanelData], method_scores: dict[str, dict[str, np.ndarray]],
    method: str,
) -> tuple[dict, dict[str, tuple[float, float, str]]]:
    records: dict[str, list[tuple[float, float, str]]] = {}
    for name in PANELS:
        _, ranks = _metric_summary(method_scores[name][method], panels[name])
        for key, formula, rank in zip(panels[name].query_keys, panels[name].formulas, ranks):
            records.setdefault(str(key), []).append((float(rank == 1), float(1.0 / rank), str(formula)))
    collapsed: dict[str, tuple[float, float, str]] = {}
    for key, values in records.items():
        formulas = {row[2] for row in values}
        if len(formulas) != 1:
            raise RuntimeError(f"repeated query formula drifted during joint metric: {key}")
        collapsed[key] = (
            float(np.mean([row[0] for row in values])),
            float(np.mean([row[1] for row in values])),
            next(iter(formulas)),
        )
    return {
        "unique_queries": int(len(collapsed)),
        "top1_utility": float(np.mean([row[0] for row in collapsed.values()])),
        "mrr_utility": float(np.mean([row[1] for row in collapsed.values()])),
        "semantics": (
            "each unique query has total mass one; a query present in k panels contributes "
            "the mean of its k panel-specific outcomes"
        ),
        "naive_union_metric_reported": False,
    }, collapsed


def _sha(path: Path) -> str:
    return sha256_file(path)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.folds < 2 or args.epochs < 1 or args.batch_queries < 1
        or args.hidden < 1 or args.learning_rate <= 0 or args.bootstrap_resamples < 100
    ):
        raise ValueError("invalid fixed training configuration")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    panels, panel_paths = _make_panels(args)
    feature_names = panels[PANELS[0]].feature_names
    anchor_index = feature_names.index(WSE)
    occurrences = _occurrences(panels)
    folds_by_occurrence = np.asarray([
        formula_fold(panels[name].formulas[query], args.folds, args.seed)
        for name, query in occurrences
    ], dtype=np.int16)
    fold_by_key: dict[str, int] = {}
    for index, (name, query) in enumerate(occurrences):
        key = str(panels[name].query_keys[query])
        fold = int(folds_by_occurrence[index])
        previous = fold_by_key.setdefault(key, fold)
        if previous != fold:
            raise RuntimeError(f"repeated query crossed global folds: {key}")

    oof = {
        arm: {name: np.full(len(panels[name].labels), np.nan, dtype=np.float32) for name in PANELS}
        for arm in ARMS
    }
    fold_reports: list[dict] = []
    for fold in range(args.folds):
        train_occ = [occ for index, occ in enumerate(occurrences) if folds_by_occurrence[index] != fold]
        valid_occ = [occ for index, occ in enumerate(occurrences) if folds_by_occurrence[index] == fold]
        train_formulas = {str(panels[name].formulas[query]) for name, query in train_occ}
        valid_formulas = {str(panels[name].formulas[query]) for name, query in valid_occ}
        if not train_occ or not valid_occ or train_formulas & valid_formulas:
            raise RuntimeError(f"global formula fold {fold} is empty or leaked")
        mean, scale = _fit_normalization(panels, train_occ)
        arm_report = {}
        for arm_index, arm in enumerate(ARMS):
            model, history = _fit_arm(
                arm, panels, train_occ, mean, scale, args,
                args.seed + 100 * fold + arm_index,
            )
            predictions = _predict_occurrences(model, panels, valid_occ, mean, scale, args.device)
            for (name, query), values in predictions.items():
                left, right = int(panels[name].query_ptr[query]), int(panels[name].query_ptr[query + 1])
                if np.any(np.isfinite(oof[arm][name][left:right])):
                    raise RuntimeError("OOF candidate row was predicted more than once")
                oof[arm][name][left:right] = values
            arm_report[arm] = {"initial_loss": history[0], "final_loss": history[-1]}
        fold_reports.append({
            "fold": fold,
            "training_occurrences": len(train_occ),
            "validation_occurrences": len(valid_occ),
            "training_formulas": len(train_formulas),
            "validation_formulas": len(valid_formulas),
            "arms": arm_report,
        })
        print(f"[unified integrator] completed global formula fold {fold + 1}/{args.folds}", flush=True)
    for arm in ARMS:
        for name in PANELS:
            if not np.all(np.isfinite(oof[arm][name])):
                raise RuntimeError(f"OOF coverage is incomplete: {arm}/{name}")

    method_scores: dict[str, dict[str, np.ndarray]] = {}
    for name in PANELS:
        method_scores[name] = dict(panels[name].scores)
        method_scores[name].update({arm: oof[arm][name] for arm in ARMS})
    common_methods = set.intersection(*(set(method_scores[name]) for name in PANELS))
    ordered_methods = [
        method for method in (
            "official_dreams", WSE, "noise_v1", "p2b_noise_v1_frozen", MSG,
            "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
            CONTROL_ARM, PRIMARY_ARM,
        ) if method in common_methods
    ]
    ordered_methods.extend(sorted(common_methods - set(ordered_methods)))

    panel_reports: dict[str, dict] = {}
    for panel_index, name in enumerate(PANELS):
        panel = panels[name]
        metrics = {
            method: _metric_summary(method_scores[name][method], panel)[0]
            for method in ordered_methods
        }
        comparisons = {}
        for index, baseline in enumerate((WSE, CONTROL_ARM)):
            comparisons[f"{PRIMARY_ARM}_vs_{baseline}"] = _comparison(
                method_scores[name][PRIMARY_ARM], method_scores[name][baseline], panel,
                args.bootstrap_resamples, args.seed + 1000 * panel_index + index,
            )
        panel_reports[name] = {"methods": metrics, "primary_comparisons": comparisons}

    joint, collapsed = {}, {}
    for method in ordered_methods:
        joint[method], collapsed[method] = _joint_query_balanced(panels, method_scores, method)
    primary_delta = np.asarray([
        collapsed[PRIMARY_ARM][key][0] - collapsed[WSE][key][0]
        for key in sorted(collapsed[PRIMARY_ARM])
    ], dtype=np.float64)
    primary_formula = np.asarray([
        collapsed[PRIMARY_ARM][key][2] for key in sorted(collapsed[PRIMARY_ARM])
    ], dtype="U")
    joint_primary_comparison = _formula_ci(
        primary_delta, primary_formula, args.bootstrap_resamples, args.seed + 9000,
    )

    final_mean, final_scale = _fit_normalization(panels, occurrences)
    final_models: dict[str, CandidateEvidenceIntegrator] = {}
    final_histories: dict[str, list[float]] = {}
    for arm_index, arm in enumerate(ARMS):
        final_models[arm], final_histories[arm] = _fit_arm(
            arm, panels, occurrences, final_mean, final_scale, args,
            args.seed + 50_000 + arm_index,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        checkpoint_path = staging / "unified_candidate_integrator.pt"
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "selection_class": "candidate_evidence_integration",
            "primary_arm": PRIMARY_ARM,
            "control_arm": CONTROL_ARM,
            "arm_selection_from_oof": False,
            "encoder_names": ENCODERS,
            "required_families": (
                *ENCODERS, WSE, "p2b_noise_v1_frozen", MSG,
            ),
            "optional_families": ("chemaware_v2_continuous_null",),
            "optional_v2_noop_helper": (
                "enveda_unified_candidate_integrator_core.unavailable_chemaware"
            ),
            "auxiliary_family_names": (
                WSE, "p2b_noise_v1_frozen", MSG, "chemaware_v2_continuous_null",
            ),
            "scalar_feature_names": feature_names,
            "scalar_mean": final_mean,
            "scalar_scale": final_scale,
            "reliability_context": True,
            "anchor_feature": WSE,
            "anchor_index": anchor_index,
            "hidden": args.hidden,
            "primary_residual_bound": None,
            "primary_residual_semantics": (
                "unbounded learned residual regularized by training-only anchor penalty; "
                "no post-result margin-derived hard ceiling"
            ),
            "arms": {
                arm: {key: value.detach().cpu() for key, value in model.state_dict().items()}
                for arm, model in final_models.items()
            },
            "training": {
                "dataset": "Enveda-180 consumed primary panels",
                "role": "fusion_meta_development_consumed",
                "folds": args.folds,
                "formula_global_across_panels": True,
                "formula_input_contract": (
                    "upstream normalized molecular formula string; trainer strips surrounding "
                    "whitespace only and does not reparse or canonicalize"
                ),
                "repeated_query_total_weight": 1.0,
                "epochs": args.epochs,
                "batch_queries": args.batch_queries,
                "batch_unit": "unique query key; all cross-panel occurrences remain together",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "anchor_strength": args.anchor_strength,
                "seed": args.seed,
                "hyperparameter_scan": False,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        prediction_arrays: dict[str, np.ndarray] = {}
        for name in PANELS:
            panel = panels[name]
            prefix = f"{name}__"
            prediction_arrays.update({
                prefix + "query_ptr": panel.query_ptr,
                prefix + "labels": panel.labels,
                prefix + "query_keys": panel.query_keys,
                prefix + "formula": panel.formulas,
                prefix + "near": panel.near,
                prefix + "query_weight": panel.query_weights,
                prefix + "fold": np.asarray([
                    formula_fold(value, args.folds, args.seed) for value in panel.formulas
                ], dtype=np.int16),
            })
            for method in ordered_methods:
                prediction_arrays[prefix + method] = method_scores[name][method]
        np.savez_compressed(staging / "candidate_predictions_oof.npz", **prediction_arrays)

        unique_queries = len(set(np.concatenate([panels[name].query_keys for name in PANELS])))
        report = {
            "status": "ENVEDA_UNIFIED_CANDIDATE_INTEGRATOR_META_DEVELOPMENT_COMPLETE",
            "dataset": "Enveda-180 cross-condition benchmark primary panels",
            "evaluation_role": "fusion_meta_development_consumed",
            "truth_status": "consumed",
            "upstream_expert_training_relation": (
                "external to the upstream Noise/ChemAware/P2b/MSG training corpora under the "
                "benchmark exclusion ledger; labels are consumed here to fit the fusion head"
            ),
            "allow_final_external_claim": False,
            "claim_boundary": (
                "Enveda estimates OOF reliability of the fusion head only; it is no longer an "
                "untouched external test. A genuinely new external collection is required for SOTA."
            ),
            "primary_model": PRIMARY_ARM,
            "control_model": CONTROL_ARM,
            "architecture_selection": (
                "primary nonlinear arm predeclared before OOF; additive arm is a non-promotable "
                "capacity-lower comparator, not a capacity-matched mechanism proof; no OOF arm picking"
            ),
            "panels": panel_reports,
            "query_balanced_joint": joint,
            "query_balanced_joint_primary_vs_wse": joint_primary_comparison,
            "occurrence_accounting": {
                "panel_occurrences": int(sum(len(panels[name].query_keys) for name in PANELS)),
                "unique_queries": int(unique_queries),
                "repeated_occurrences_total_training_weight_one": True,
                "naive_union_ranking_metric_reported": False,
            },
            "folds": fold_reports,
            "final_refit": {
                arm: {"initial_loss": history[0], "final_loss": history[-1]}
                for arm, history in final_histories.items()
            },
            "checkpoint": {
                "path": "unified_candidate_integrator.pt",
                "sha256": _sha(checkpoint_path),
                "deployable_when_v2_unavailable": True,
                "v2_unavailable_semantics": "six V2 scalar fields are exact zero including availability",
            },
            "prediction_ledger": {
                "path": "candidate_predictions_oof.npz",
                "sha256": _sha(staging / "candidate_predictions_oof.npz"),
            },
            "features": {
                "encoder_names": list(ENCODERS),
                "auxiliary_families": [WSE, "p2b_noise_v1_frozen", MSG, "chemaware_v2_continuous_null"],
                "scalar_feature_names": list(feature_names),
                "reliability_context": True,
                "p2b_input_once": True,
                "msg_input_once_as_frozen_composite": True,
                "official_models_are_comparators_only": True,
                "required_families": [
                    *ENCODERS, WSE, "p2b_noise_v1_frozen", MSG,
                ],
                "optional_families": ["chemaware_v2_continuous_null"],
                "primary_residual_bound": None,
            },
            "optimization": {
                "folds": args.folds, "epochs": args.epochs,
                "batch_queries": args.batch_queries, "hidden": args.hidden,
                "batch_unit": "unique query key with indivisible cross-panel occurrences",
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "anchor_strength": args.anchor_strength, "seed": args.seed,
                "hyperparameter_scan": False,
            },
            "input_sha256": {
                "score_bundle": _sha(args.score_bundle),
                "msg_report": _sha(args.msg_cache / "report.json"),
                "stage1_embeddings": _sha(args.stage1_embeddings),
                "phasea_embeddings": _sha(args.phasea_embeddings),
                "fullmsg_embeddings": _sha(args.fullmsg_embeddings),
                "chem_report": _sha(args.chem_dir / "report.json"),
                **{f"panel_{name}": _sha(panel_paths[name]) for name in PANELS},
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "primary_model": PRIMARY_ARM,
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

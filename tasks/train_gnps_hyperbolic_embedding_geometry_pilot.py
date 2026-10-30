#!/usr/bin/env python
"""Strict Euclidean-vs-hyperbolic readout pilot on frozen GNPS embeddings.

The experiment asks one narrow question: after the same frozen random input
projection and the same trainable linear readout, does Lorentz distance add
identity-discrimination value beyond an equally trained Euclidean distance?

Only the formula-disjoint GNPS panel supplies training queries.  Formula-hash
outer folds ensure that every reported query is scored by a model that did not
train on its molecular formula.  The identity panel is prediction-only under
the same fold models.  GNPS is consumed development evidence; this script does
not create an external or SOTA claim.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from enveda_unified_candidate_integrator_core import formula_fold
from evaluate_gnps_gold_silver_10ppm_embeddings import (
    graph_from_panel,
    load_embeddings,
    paired_summary,
)
from noise_corrected_fullgraph_evaluation import (
    GraphScores,
    expanded_indices,
    full_metrics,
    score_embeddings,
)


PANELS = ("identity_disjoint", "formula_disjoint")
METHODS = (
    "original_cosine_1024",
    "fixed_random_projection_cosine",
    "learned_euclidean_projection",
    "learned_lorentz_projection",
)
SCHEMA = "gnps_hyperbolic_embedding_geometry_pilot_v1"
STATUS = "GNPS_HYPERBOLIC_EMBEDDING_GEOMETRY_PILOT_COMPLETE"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--identity-panel", type=Path)
    parser.add_argument("--formula-panel", type=Path)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-queries", type=int, default=128)
    parser.add_argument("--fixed-dimension", type=int, default=256)
    parser.add_argument("--learned-dimension", type=int, default=16)
    parser.add_argument("--maximum-radius", type=float, default=2.0)
    parser.add_argument("--anchor-cosine", action="store_true")
    parser.add_argument("--residual-bound", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _positions(rows: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    at = np.searchsorted(sorted_rows, requested)
    if np.any(at == len(sorted_rows)) or not np.array_equal(sorted_rows[at], requested):
        raise RuntimeError(f"embedding cache misses a {label} row")
    return order[at]


@dataclass(frozen=True)
class IndexedPanel:
    name: str
    graph: object
    query_positions: np.ndarray
    candidate_positions: np.ndarray
    pair_molecule: np.ndarray
    formula_folds: np.ndarray


def _indexed_panel(name: str, graph: object, rows: np.ndarray, folds: int, seed: int) -> IndexedPanel:
    pair_molecule = np.repeat(
        np.arange(len(graph.molecule_label), dtype=np.int64),
        np.diff(np.asarray(graph.molecule_ptr, dtype=np.int64)),
    )
    if len(pair_molecule) != len(graph.pair_candidate_row):
        raise RuntimeError("pair-to-molecule expansion drifted")
    formula_folds = np.asarray(
        [formula_fold(str(value), folds, seed) for value in graph.query_formula],
        dtype=np.int64,
    )
    return IndexedPanel(
        name=name,
        graph=graph,
        query_positions=_positions(rows, graph.query_row, "query"),
        candidate_positions=_positions(rows, graph.pair_candidate_row, "candidate"),
        pair_molecule=pair_molecule,
        formula_folds=formula_folds,
    )


@dataclass(frozen=True)
class PackedQueries:
    pair_indices: np.ndarray
    molecule_indices: np.ndarray
    query_positions_by_pair: np.ndarray
    candidate_positions_by_pair: np.ndarray
    local_molecule_by_pair: np.ndarray
    local_query_ptr: np.ndarray
    local_positive: np.ndarray


def _pack_queries(panel: IndexedPanel, queries: np.ndarray) -> PackedQueries:
    graph = panel.graph
    pair_parts: list[np.ndarray] = []
    molecule_parts: list[np.ndarray] = []
    query_position_parts: list[np.ndarray] = []
    local_molecule_parts: list[np.ndarray] = []
    local_query_ptr = [0]
    local_positive: list[int] = []
    molecule_cursor = 0
    for query_value in np.asarray(queries, dtype=np.int64):
        query = int(query_value)
        molecule_left, molecule_right = map(
            int, graph.query_ptr[query:query + 2],
        )
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        molecules = np.arange(molecule_left, molecule_right, dtype=np.int64)
        pairs = np.arange(pair_left, pair_right, dtype=np.int64)
        local_widths = np.diff(graph.molecule_ptr[molecule_left:molecule_right + 1])
        pair_parts.append(pairs)
        molecule_parts.append(molecules)
        query_position_parts.append(
            np.full(len(pairs), panel.query_positions[query], dtype=np.int64),
        )
        local_molecule_parts.append(
            np.repeat(
                np.arange(molecule_cursor, molecule_cursor + len(molecules), dtype=np.int64),
                local_widths,
            )
        )
        positive = np.flatnonzero(np.asarray(graph.molecule_label[molecule_left:molecule_right]) == 1)
        if len(positive) != 1:
            raise RuntimeError(f"query {query} does not have exactly one positive molecule")
        local_positive.append(molecule_cursor + int(positive[0]))
        molecule_cursor += len(molecules)
        local_query_ptr.append(molecule_cursor)
    pair_indices = np.concatenate(pair_parts)
    molecule_indices = np.concatenate(molecule_parts)
    return PackedQueries(
        pair_indices=pair_indices,
        molecule_indices=molecule_indices,
        query_positions_by_pair=np.concatenate(query_position_parts),
        candidate_positions_by_pair=panel.candidate_positions[pair_indices],
        local_molecule_by_pair=np.concatenate(local_molecule_parts),
        local_query_ptr=np.asarray(local_query_ptr, dtype=np.int64),
        local_positive=np.asarray(local_positive, dtype=np.int64),
    )


class MetricProjection(nn.Module):
    """Parameter-matched Euclidean or Lorentz distance readout."""

    def __init__(
        self, input_dimension: int, output_dimension: int,
        maximum_radius: float, geometry: str, *,
        anchor_cosine: bool = False, residual_bound: float = 0.25,
    ):
        super().__init__()
        if geometry not in {"euclidean", "lorentz"}:
            raise ValueError("geometry must be euclidean or lorentz")
        self.geometry = geometry
        self.maximum_radius = float(maximum_radius)
        self.anchor_cosine = bool(anchor_cosine)
        self.residual_bound = float(residual_bound)
        if self.residual_bound <= 0:
            raise ValueError("residual bound must be positive")
        self.projection = nn.Linear(input_dimension, output_dimension, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def tangent(self, values: torch.Tensor) -> torch.Tensor:
        raw = self.projection(values)
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
        radius = self.maximum_radius * torch.tanh(norm)
        return raw * (radius / norm)

    def pair_score(
        self, query: torch.Tensor, candidate: torch.Tensor,
        base_cosine: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.tangent(query)
        c = self.tangent(candidate)
        if self.geometry == "euclidean":
            distance_squared = torch.sum(torch.square(q - c), dim=-1)
        else:
            q_norm = torch.linalg.vector_norm(q, dim=-1).clamp_min(1e-8)
            c_norm = torch.linalg.vector_norm(c, dim=-1).clamp_min(1e-8)
            q_time = torch.cosh(q_norm)
            c_time = torch.cosh(c_norm)
            q_space = torch.sinh(q_norm).unsqueeze(-1) * q / q_norm.unsqueeze(-1)
            c_space = torch.sinh(c_norm).unsqueeze(-1) * c / c_norm.unsqueeze(-1)
            lorentz_inner = q_time * c_time - torch.sum(q_space * c_space, dim=-1)
            distance = torch.acosh(lorentz_inner.clamp_min(1.0 + 1e-6))
            distance_squared = torch.square(distance)
        temperature = torch.exp(self.log_temperature).clamp(0.05, 20.0)
        geometry_score = -distance_squared / temperature
        if not self.anchor_cosine:
            return geometry_score
        if base_cosine is None or base_cosine.shape != geometry_score.shape:
            raise ValueError("cosine-anchored geometry requires aligned base scores")
        residual = self.residual_bound * torch.tanh(
            self.residual_gate * geometry_score / self.residual_bound,
        )
        return base_cosine + residual


def _molecule_scores(pair_scores: torch.Tensor, packed: PackedQueries, device: str) -> torch.Tensor:
    indices = torch.as_tensor(packed.local_molecule_by_pair, dtype=torch.int64, device=device)
    output = torch.full(
        (len(packed.molecule_indices),), -torch.inf,
        dtype=pair_scores.dtype, device=device,
    )
    output.scatter_reduce_(0, indices, pair_scores, reduce="amax", include_self=True)
    if not torch.all(torch.isfinite(output)):
        raise RuntimeError("molecule score aggregation produced non-finite values")
    return output


def _batch_loss(
    model: MetricProjection, bank: torch.Tensor, panel: IndexedPanel,
    queries: np.ndarray, raw_pair_scores: np.ndarray, device: str,
) -> torch.Tensor:
    packed = _pack_queries(panel, queries)
    query = bank[torch.as_tensor(packed.query_positions_by_pair, dtype=torch.int64, device=device)]
    candidate = bank[torch.as_tensor(packed.candidate_positions_by_pair, dtype=torch.int64, device=device)]
    base = torch.as_tensor(
        raw_pair_scores[packed.pair_indices], dtype=torch.float32, device=device,
    )
    molecule = _molecule_scores(model.pair_score(query, candidate, base), packed, device)
    losses = []
    for index, positive in enumerate(packed.local_positive):
        left, right = map(int, packed.local_query_ptr[index:index + 2])
        losses.append(torch.logsumexp(molecule[left:right], dim=0) - molecule[int(positive)])
    return torch.stack(losses).mean()


def _fit(
    model: MetricProjection, bank: torch.Tensor, panel: IndexedPanel,
    train_queries: np.ndarray, raw_pair_scores: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> tuple[MetricProjection, list[float]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for _ in range(args.epochs):
        shuffled = rng.permutation(train_queries)
        losses = []
        model.train()
        for start in range(0, len(shuffled), args.batch_queries):
            chosen = shuffled[start:start + args.batch_queries]
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model, bank, panel, chosen, raw_pair_scores, args.device,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("metric geometry loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.inference_mode()
def _predict_queries(
    model: MetricProjection, bank: torch.Tensor, panel: IndexedPanel,
    raw_pair_scores: np.ndarray,
    queries: np.ndarray, pair_output: np.ndarray, molecule_output: np.ndarray,
    batch_queries: int, device: str,
) -> None:
    model.eval()
    for start in range(0, len(queries), batch_queries):
        chosen = queries[start:start + batch_queries]
        packed = _pack_queries(panel, chosen)
        query = bank[torch.as_tensor(packed.query_positions_by_pair, dtype=torch.int64, device=device)]
        candidate = bank[torch.as_tensor(packed.candidate_positions_by_pair, dtype=torch.int64, device=device)]
        base = torch.as_tensor(
            raw_pair_scores[packed.pair_indices], dtype=torch.float32, device=device,
        )
        pair = model.pair_score(query, candidate, base)
        molecule = _molecule_scores(pair, packed, device)
        pair_output[packed.pair_indices] = pair.float().cpu().numpy()
        molecule_output[packed.molecule_indices] = molecule.float().cpu().numpy()


@torch.inference_mode()
def _radius_summary(model: MetricProjection, bank: torch.Tensor, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(bank), size=min(4096, len(bank)), replace=False)
    values = model.tangent(bank[torch.as_tensor(indices, dtype=torch.int64, device=bank.device)])
    radius = torch.linalg.vector_norm(values, dim=-1).cpu().numpy()
    return {
        "mean": float(np.mean(radius)),
        "q05": float(np.quantile(radius, 0.05)),
        "median": float(np.median(radius)),
        "q95": float(np.quantile(radius, 0.95)),
        "maximum": float(np.max(radius)),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds < 2 or args.epochs < 1 or args.batch_queries < 1:
        raise ValueError("invalid training budget")
    if args.fixed_dimension < args.learned_dimension or args.learned_dimension < 2:
        raise ValueError("invalid fixed/learned dimensions")
    if (
        args.maximum_radius <= 0 or args.residual_bound <= 0
        or args.bootstrap_resamples < 100
    ):
        raise ValueError("invalid geometry/evaluation settings")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    panel_paths = {
        "identity_disjoint": (
            args.identity_panel
            if args.identity_panel is not None
            else args.benchmark / "panel_identity_disjoint.npz"
        ),
        "formula_disjoint": (
            args.formula_panel
            if args.formula_panel is not None
            else args.benchmark / "panel_formula_disjoint.npz"
        ),
    }
    missing_panels = [str(path) for path in panel_paths.values() if not path.is_file()]
    if missing_panels:
        raise FileNotFoundError(f"GNPS panel files are missing: {missing_panels}")

    manifest_rows = sum(1 for _ in open(args.benchmark / "manifest.csv.gz", "rb"))
    # The compressed CSV line count is not a row count.  Use the frozen maximum
    # source row instead; load_embeddings still validates every requested row.
    with np.load(panel_paths["identity_disjoint"], allow_pickle=False) as panel:
        manifest_rows = int(max(np.max(panel["query_row"]), np.max(panel["candidate_row"])) + 1)
    rows, embeddings = load_embeddings(args.embeddings, manifest_rows)
    if embeddings.shape[1] != 1024:
        raise RuntimeError("registered embedding dimension is not 1024")

    rng = np.random.default_rng(args.seed)
    if args.fixed_dimension == embeddings.shape[1]:
        random_projection = np.eye(embeddings.shape[1], dtype=np.float32)
        compressed = embeddings.copy()
        fixed_projection_kind = "identity_full_embedding"
    else:
        random_projection = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=(embeddings.shape[1], args.fixed_dimension),
        ) / math.sqrt(args.fixed_dimension)
        compressed = np.asarray(embeddings @ random_projection, dtype=np.float32)
        compressed /= np.linalg.norm(compressed, axis=1, keepdims=True).clip(1e-8)
        fixed_projection_kind = "seeded_rademacher"
    bank = torch.as_tensor(compressed, dtype=torch.float32, device=args.device)

    panels = {
        name: _indexed_panel(
            name,
            graph_from_panel(panel_paths[name]),
            rows, args.folds, args.seed,
        )
        for name in PANELS
    }
    training_panel = panels["formula_disjoint"]
    raw_scores = {
        name: score_embeddings(panel.graph, rows, embeddings)
        for name, panel in panels.items()
    }
    compressed_scores = {
        name: score_embeddings(panel.graph, rows, compressed)
        for name, panel in panels.items()
    }
    oof = {
        geometry: {
            name: GraphScores(
                pair=np.full(len(panel.graph.pair_candidate_row), np.nan, np.float32),
                molecule=np.full(len(panel.graph.molecule_label), np.nan, np.float32),
            )
            for name, panel in panels.items()
        }
        for geometry in ("euclidean", "lorentz")
    }
    checkpoints: dict[str, list[dict[str, torch.Tensor]]] = {"euclidean": [], "lorentz": []}
    learned_methods = (
        (
            "cosine_anchored_euclidean_residual",
            "cosine_anchored_lorentz_residual",
        )
        if args.anchor_cosine else METHODS[2:]
    )
    fold_reports = []
    for held_out in range(args.folds):
        train_queries = np.flatnonzero(training_panel.formula_folds != held_out)
        validation_formulas = {
            str(value) for panel in panels.values()
            for value in panel.graph.query_formula[panel.formula_folds == held_out]
        }
        train_formulas = {
            str(value) for value in training_panel.graph.query_formula[train_queries]
        }
        if train_formulas.intersection(validation_formulas):
            raise RuntimeError("formula outer-fold leakage detected")

        torch.manual_seed(args.seed + 100 * held_out)
        euclidean = MetricProjection(
            args.fixed_dimension, args.learned_dimension,
            args.maximum_radius, "euclidean",
            anchor_cosine=args.anchor_cosine,
            residual_bound=args.residual_bound,
        ).to(args.device)
        initial_state = copy.deepcopy(euclidean.state_dict())
        lorentz = MetricProjection(
            args.fixed_dimension, args.learned_dimension,
            args.maximum_radius, "lorentz",
            anchor_cosine=args.anchor_cosine,
            residual_bound=args.residual_bound,
        ).to(args.device)
        lorentz.load_state_dict(initial_state)
        euclidean, euclidean_history = _fit(
            euclidean, bank, training_panel, train_queries,
            raw_scores["formula_disjoint"].pair,
            args, args.seed + 1000 + held_out,
        )
        lorentz, lorentz_history = _fit(
            lorentz, bank, training_panel, train_queries,
            raw_scores["formula_disjoint"].pair,
            args, args.seed + 1000 + held_out,
        )
        for geometry, model in (("euclidean", euclidean), ("lorentz", lorentz)):
            for name, panel in panels.items():
                held_queries = np.flatnonzero(panel.formula_folds == held_out)
                _predict_queries(
                    model, bank, panel, raw_scores[name].pair, held_queries,
                    oof[geometry][name].pair, oof[geometry][name].molecule,
                    args.batch_queries, args.device,
                )
            checkpoints[geometry].append({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
        fold_reports.append({
            "fold": held_out,
            "training_queries": int(len(train_queries)),
            "held_out_identity_queries": int(np.sum(panels["identity_disjoint"].formula_folds == held_out)),
            "held_out_formula_queries": int(np.sum(panels["formula_disjoint"].formula_folds == held_out)),
            "formula_overlap": 0,
            "euclidean_loss": euclidean_history,
            "lorentz_loss": lorentz_history,
            "euclidean_radius": _radius_summary(euclidean, bank, args.seed + held_out),
            "lorentz_radius": _radius_summary(lorentz, bank, args.seed + held_out),
        })
        print(f"[hyperbolic pilot] completed formula fold {held_out + 1}/{args.folds}", flush=True)

    for geometry in oof:
        for name in PANELS:
            if np.any(~np.isfinite(oof[geometry][name].pair)) or np.any(~np.isfinite(oof[geometry][name].molecule)):
                raise RuntimeError(f"{geometry}/{name} OOF prediction coverage is incomplete")

    methods_by_panel = {}
    report_panels = {}
    outcomes = {}
    for panel_index, name in enumerate(PANELS):
        graph = panels[name].graph
        methods = {
            METHODS[0]: raw_scores[name],
            METHODS[1]: compressed_scores[name],
            learned_methods[0]: oof["euclidean"][name],
            learned_methods[1]: oof["lorentz"][name],
        }
        methods_by_panel[name] = methods
        metric_tables = {}
        metrics = {}
        adduct = np.full(graph.n_queries, "[M+H]+", dtype="U6")
        for method, scores in methods.items():
            metrics[method], metric_tables[method] = full_metrics(graph, scores, adduct)
        comparisons = {}
        for comparison_index, (candidate, baseline) in enumerate((
            (learned_methods[0], METHODS[0]),
            (learned_methods[1], METHODS[0]),
            (learned_methods[1], learned_methods[0]),
            (METHODS[1], METHODS[0]),
        )):
            paired, outcome = paired_summary(
                metric_tables[baseline], metric_tables[candidate],
                args.bootstrap_resamples,
                args.seed + panel_index * 1000 + comparison_index * 100,
                hypotheses=16,
            )
            key = f"{candidate}_vs_{baseline}"
            comparisons[key] = paired
            outcomes[(name, key)] = outcome
        report_panels[name] = {
            "queries": int(graph.n_queries),
            "molecule_candidates": int(len(graph.molecule_label)),
            "spectrum_edges": int(len(graph.pair_candidate_row)),
            "metrics": metrics,
            "comparisons": comparisons,
        }

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "embedding_name": args.embedding_name,
        "dataset": {
            "name": "GNPS Gold/Silver 10-ppm",
            "role": "consumed development only",
            "training_panel": "formula_disjoint only",
            "identity_panel_role": "formula-OOF prediction-only stress test",
            "allow_external_claim": False,
        },
        "design": {
            "original_embedding_dimension": 1024,
            "fixed_rademacher_dimension": args.fixed_dimension,
            "fixed_projection_kind": fixed_projection_kind,
            "learned_dimension": args.learned_dimension,
            "parameter_matched_geometry_heads": True,
            "shared_initialization_within_fold": True,
            "same_training_queries_and_budget": True,
            "maximum_tangent_radius": args.maximum_radius,
            "cosine_anchored_residual": args.anchor_cosine,
            "residual_bound_per_pair": args.residual_bound if args.anchor_cosine else None,
            "exact_cosine_initialization": args.anchor_cosine,
            "lorentz_curvature": -1.0,
            "candidate_reference_aggregation": "maximum across all frozen reference spectra",
            "no_source_or_panel_indicator": True,
        },
        "training": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "trainable_parameters_per_learned_arm": int(sum(p.numel() for p in euclidean.parameters())),
        },
        "folds": fold_reports,
        "panels": report_panels,
        "claim_boundary": (
            "This consumed-GNPS development pilot tests geometry after a fixed random "
            "compression. Hyperbolic benefit requires Lorentz to beat the parameter-matched "
            "Euclidean arm, not merely raw cosine. No external or SOTA claim is permitted."
        ),
    }
    prediction_arrays = {}
    for name, panel in panels.items():
        prefix = name + "__"
        prediction_arrays.update({
            prefix + "query_ptr": np.asarray(panel.graph.query_ptr, dtype=np.int64),
            prefix + "molecule_ptr": np.asarray(panel.graph.molecule_ptr, dtype=np.int64),
            prefix + "molecule_label": np.asarray(panel.graph.molecule_label, dtype=np.int8),
            prefix + "query_formula": np.asarray(panel.graph.query_formula).astype("U"),
            prefix + "near": np.asarray(panel.graph.query_has_near, dtype=bool),
        })
        for method, scores in methods_by_panel[name].items():
            prediction_arrays[prefix + method + "__pair"] = scores.pair.astype(np.float32)
            prediction_arrays[prefix + method + "__molecule"] = scores.molecule.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        torch.save({
            "schema": SCHEMA,
            "embedding_name": args.embedding_name,
            "fixed_random_projection": torch.as_tensor(random_projection),
            "fold_states": checkpoints,
            "settings": payload["design"],
            "training": payload["training"],
        }, staging / "geometry_fold_models.pt")
        np.savez_compressed(staging / "candidate_predictions_outer_oof.npz", **prediction_arrays)
        for (panel_name, key), outcome in outcomes.items():
            outcome.to_csv(
                staging / f"paired_{panel_name}__{key}.csv.gz",
                index=False, compression="gzip",
            )
        (staging / "report.json").write_text(
            json.dumps(_jsonable(payload), indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": STATUS, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Apply one MassSpecGym-selected unified ranker to fixed GNPS panels.

This program never fits or selects parameters.  The supplied model already
contains its arm, normalization and weights; GNPS is read only after that
artifact exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from build_grand_unified_evidence_bundle import load_named_arrays
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from massspecgym_embedding_readout_core import (
    FULL_INTEGRATION_ARMS, REQUIRED_ENCODERS, FrozenEmbeddingReadout,
    ReadoutSettings, build_independent_scalar_features, comparison,
    rank_metrics, strict_ranks,
)
from noise_final_core import sha256_file


PANELS = ("identity_disjoint", "formula_disjoint")
BASELINE_NAMES = (
    "weighted_spectral_entropy", "noise_v1", "p2b_noise_v1_frozen",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--embedding", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args()


def load_model(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    settings = ReadoutSettings(**payload["settings"])
    if tuple(payload["encoder_names"]) != settings.encoder_names:
        raise RuntimeError("selected model encoder registry drifted")
    if str(payload["arm"]) not in FULL_INTEGRATION_ARMS:
        raise RuntimeError("fixed GNPS evaluation requires a full-embedding integration arm")
    if payload.get("selection_class") != "full_embedding_integration":
        raise RuntimeError("selected model lacks the full-integration selection contract")
    if settings.encoder_names != REQUIRED_ENCODERS:
        raise RuntimeError("selected model does not contain the exact four registered encoders")
    model = FrozenEmbeddingReadout(str(payload["arm"]), settings)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload, model.to(device).eval(), settings


def parse_embeddings(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=NPZ, got {value!r}")
        name, path = value.split("=", 1)
        if name in output:
            raise ValueError(f"duplicate encoder {name}")
        output[name] = Path(path)
    return output


def embedding_molecule_evidence(graph, rows: np.ndarray, bank: np.ndarray):
    row_map = np.full(int(rows.max()) + 1, -1, dtype=np.int64)
    row_map[rows] = np.arange(len(rows), dtype=np.int64)
    required_max = int(max(graph.query_row.max(), graph.pair_candidate_row.max()))
    if required_max >= len(row_map):
        row_map = np.pad(row_map, (0, required_max + 1 - len(row_map)), constant_values=-1)
    query_edge_ptr = graph.molecule_ptr[graph.query_ptr]
    edge_query = np.repeat(np.arange(graph.n_queries), np.diff(query_edge_ptr))
    query_local = row_map[graph.query_row[edge_query]]
    candidate_local = row_map[graph.pair_candidate_row]
    if np.any(query_local < 0) or np.any(candidate_local < 0):
        raise RuntimeError("GNPS embedding cache does not cover the complete panel")
    edge = np.sum(bank[query_local] * bank[candidate_local], axis=1, dtype=np.float32)
    molecule = np.maximum.reduceat(edge, graph.molecule_ptr[:-1]).astype(np.float32)
    reference = np.empty(len(molecule), dtype=np.int64)
    for index, (left, right) in enumerate(zip(graph.molecule_ptr[:-1], graph.molecule_ptr[1:])):
        local = int(np.argmax(edge[int(left):int(right)]))
        reference[index] = candidate_local[int(left) + local]
    query_for_molecule = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    query_bank_row = row_map[graph.query_row[query_for_molecule]]
    return molecule, reference, query_bank_row


def load_chemical(path: Path, graph) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        chem = {name: np.asarray(body[name]) for name in body.files}
    if str(chem["schema"].item()) != "gnps_chemaware_continuous_null_evidence_v1":
        raise RuntimeError(f"unsupported GNPS chemistry cache: {path}")
    if not np.array_equal(chem["query_ptr"], graph.query_ptr):
        raise RuntimeError("GNPS chemistry query axis drifted")
    return {
        "chem_valid": np.asarray(chem["availability"], dtype=bool),
        "candidate_utility_correct": np.asarray(chem["utility_correct"], dtype=np.float32),
        "candidate_utility_zero": np.asarray(chem["utility_zero"], dtype=np.float32),
        "candidate_utility_reversed": np.asarray(chem["utility_reversed"], dtype=np.float32),
        "candidate_utility_rotated": np.asarray(chem["utility_candidate_mismatch"], dtype=np.float32),
    }


def metrics(scores: np.ndarray, graph) -> dict:
    ranks = strict_ranks(scores, graph.query_ptr)
    output = rank_metrics(ranks, graph.query_has_near)
    labels = np.asarray(graph.molecule_label, dtype=np.int8)
    output["micro_candidate_auroc"] = float(roc_auc_score(labels, scores))
    output["micro_candidate_auprc"] = float(average_precision_score(labels, scores))
    macro_auc, macro_ap = [], []
    for left, right in zip(graph.query_ptr[:-1], graph.query_ptr[1:]):
        y = labels[int(left):int(right)]
        s = scores[int(left):int(right)]
        macro_auc.append(roc_auc_score(y, s))
        macro_ap.append(average_precision_score(y, s))
    output["macro_query_auroc"] = float(np.mean(macro_auc))
    output["macro_query_auprc"] = float(np.mean(macro_ap))
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GNPS unified evaluation requires an allocated GPU")
    payload, model, settings = load_model(args.model, device)
    embedding_paths = parse_embeddings(args.embedding)
    if tuple(embedding_paths) != settings.encoder_names:
        raise RuntimeError("GNPS embedding arguments differ from selected model order")
    manifest_rows = sum(1 for _ in __import__("gzip").open(
        args.benchmark / "manifest.csv.gz", "rt", encoding="utf-8"
    )) - 1
    loaded = {
        name: load_embeddings(path, manifest_rows)
        for name, path in embedding_paths.items()
    }
    first_rows = next(iter(loaded.values()))[0]
    if any(not np.array_equal(rows, first_rows) for rows, _ in loaded.values()):
        raise RuntimeError("GNPS encoder caches use different row registries")
    banks = {
        name: torch.as_tensor(bank, dtype=torch.float32, device=device)
        for name, (_, bank) in loaded.items()
    }
    base_by_panel = {
        panel: load_named_arrays(args.base_scores, panel) for panel in PANELS
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    report = {
        "status": "GNPS_UNIFIED_CANDIDATE_RANKER_FIXED_EVALUATION_COMPLETE",
        "selection_or_training_on_gnps": False,
        "model": {"path": str(args.model), "sha256": sha256_file(args.model),
                  "arm": str(payload["arm"])},
        "panels": {},
    }
    try:
        for panel_index, panel in enumerate(PANELS):
            graph = graph_from_panel(args.benchmark / f"panel_{panel}.npz")
            raw = base_by_panel[panel]
            missing = [name for name in BASELINE_NAMES if name not in raw]
            if missing:
                raise RuntimeError(f"base score bundle misses {panel}: {missing}")
            encoder_scores, references, query_rows = {}, [], None
            for name in settings.encoder_names:
                score, reference, encoder_query = embedding_molecule_evidence(
                    graph, loaded[name][0], loaded[name][1],
                )
                encoder_scores[name] = score
                references.append(reference)
                if query_rows is None:
                    query_rows = encoder_query
                elif not np.array_equal(query_rows, encoder_query):
                    raise RuntimeError("GNPS query row mapping differs across encoders")
            evidence = {
                "v1_cosine": np.asarray(raw["noise_v1"], dtype=np.float32),
                "weighted_entropy": np.asarray(raw["weighted_spectral_entropy"], dtype=np.float32),
                "p2b_fused": np.asarray(raw["p2b_noise_v1_frozen"], dtype=np.float32),
            }
            chemical = load_chemical(
                args.chem_dir / f"continuous_chem_{panel}.npz", graph,
            )
            scalar, names = build_independent_scalar_features(
                evidence, graph, encoder_scores, "noise_v1", chemical,
            )
            if list(names) != list(payload["scalar_feature_names"]):
                raise RuntimeError("GNPS scalar feature contract differs from selected model")
            mean = np.asarray(payload["scalar_mean"], dtype=np.float32)
            scale = np.asarray(payload["scalar_scale"], dtype=np.float32)
            reference_rows = np.stack(references, axis=1)
            scores = np.empty(len(graph.molecule_label), dtype=np.float32)
            with torch.inference_mode():
                for left in range(0, len(scores), 16384):
                    right = min(left + 16384, len(scores))
                    scores[left:right] = model(
                        torch.as_tensor((scalar[left:right] - mean) / scale, device=device),
                        torch.as_tensor(query_rows[left:right], device=device),
                        torch.as_tensor(reference_rows[left:right], device=device),
                        banks,
                    ).float().cpu().numpy()
            baselines = {
                name: np.maximum.reduceat(
                    np.asarray(raw[name], dtype=np.float32), graph.molecule_ptr[:-1],
                )
                for name in BASELINE_NAMES
            }
            panel_metrics = {"unified_candidate_ranker": metrics(scores, graph)}
            panel_metrics.update({name: metrics(values, graph) for name, values in baselines.items()})
            comparisons = {
                f"unified_vs_{name}": comparison(
                    scores, values, graph.query_ptr, graph.query_formula,
                    args.bootstrap_resamples, 20261025 + panel_index * 100 + index,
                )
                for index, (name, values) in enumerate(baselines.items())
            }
            report["panels"][panel] = {
                "queries": int(graph.n_queries), "candidates": int(len(scores)),
                "metrics": panel_metrics, "comparisons": comparisons,
                "chemical_available_candidates": int(chemical["chem_valid"].sum()),
            }
            np.savez_compressed(
                staging / f"candidate_scores_{panel}.npz",
                unified_candidate_ranker=scores,
                **baselines,
                query_ptr=graph.query_ptr, molecule_label=graph.molecule_label,
                query_formula=graph.query_formula, molecule_ik14=graph.molecule_ik14,
            )
        report["provenance"] = {
            "base_scores_sha256": sha256_file(args.base_scores),
            "embedding_sha256": {name: sha256_file(path) for name, path in embedding_paths.items()},
            "chemistry_sha256": {
                panel: sha256_file(args.chem_dir / f"continuous_chem_{panel}.npz")
                for panel in PANELS
            },
        }
        report["claim_limit"] = (
            "Fixed GNPS evaluation after MassSpecGym-only arm selection and refit; "
            "the evaluator performs no fit, threshold search, routing or model selection."
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

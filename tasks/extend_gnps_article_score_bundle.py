#!/usr/bin/env python
"""Append frozen embedding methods and pair-score caches to a GNPS bundle.

This is the ChemAware-owned integration layer.  It never rebuilds classical
scores, selects checkpoints, tunes on GNPS, or edits the base Noise/P2b bundle.
Every appended method is scored on the exact frozen edge order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from gnps_pair_score_cache import PANELS, load_pair_score_cache
from noise_corrected_fullgraph_evaluation import score_embeddings


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument(
        "--embedding-method", action="append", default=[], metavar="NAME=PATH",
        help="Aligned embedding cache created by encode_gnps_gold_silver_10ppm_checkpoint.py",
    )
    parser.add_argument(
        "--pair-score-method", action="append", default=[], metavar="NAME=CACHE_DIR",
        help="Immutable cache following gnps_pair_score_cache_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_specs(values: list[str], kind: str) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must have NAME=PATH form: {value}")
        name, raw_path = value.split("=", 1)
        if not name or any(char.isspace() for char in name):
            raise ValueError(f"invalid method name: {name!r}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        output.append((name, path))
    names = [name for name, _ in output]
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate {kind} names")
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    embedding_specs = parse_specs(args.embedding_method, "embedding method")
    pair_specs = parse_specs(args.pair_score_method, "pair-score method")
    if not embedding_specs and not pair_specs:
        raise RuntimeError("at least one method must be appended")

    manifest_rows = len(pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"]))
    graphs = {
        name: graph_from_panel(args.benchmark / f"panel_{name}.npz") for name in PANELS
    }
    with np.load(args.base_bundle, allow_pickle=False) as base:
        base_methods = list(map(str, base["method_names"]))
        base_scores = {
            name: np.asarray(base[f"scores_{name}"], dtype=np.float32) for name in PANELS
        }
    for name in PANELS:
        expected = (len(base_methods), len(graphs[name].pair_candidate_row))
        if base_scores[name].shape != expected or not np.all(np.isfinite(base_scores[name])):
            raise RuntimeError(f"base bundle is malformed for {name}: {base_scores[name].shape}")

    appended_names = [name for name, _ in embedding_specs + pair_specs]
    if set(base_methods) & set(appended_names):
        raise RuntimeError("an appended method collides with a base method name")
    if len(set(appended_names)) != len(appended_names):
        raise RuntimeError("appended method names are duplicated across input types")

    appended: dict[str, dict[str, np.ndarray]] = {}
    sources: dict[str, dict[str, object]] = {}
    for method, path in embedding_specs:
        rows, embeddings = load_embeddings(path, manifest_rows)
        appended[method] = {
            name: score_embeddings(graphs[name], rows, embeddings).pair for name in PANELS
        }
        sources[method] = {
            "kind": "aligned_spectrum_embedding",
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": int(len(rows)),
            "dimension": int(embeddings.shape[1]),
        }
    for method, path in pair_specs:
        cache_report, scores = load_pair_score_cache(path, args.benchmark)
        appended[method] = scores
        sources[method] = {
            "kind": "frozen_pair_score_cache",
            "path": str(path),
            "report_sha256": sha256_file(path / "report.json"),
            "method": cache_report["method"],
        }

    all_methods = base_methods + appended_names
    arrays: dict[str, np.ndarray] = {"method_names": np.asarray(all_methods, dtype="U96")}
    for panel in PANELS:
        extra = np.stack([appended[name][panel] for name in appended_names]).astype(np.float32)
        matrix = np.concatenate([base_scores[panel], extra], axis=0)
        if matrix.shape != (len(all_methods), len(graphs[panel].pair_candidate_row)):
            raise RuntimeError(f"extended bundle shape drift for {panel}")
        arrays[f"scores_{panel}"] = matrix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    base_metadata_path = args.base_bundle.with_suffix(args.base_bundle.suffix + ".json")
    metadata = {
        "status": "chemaware_gnps_article_score_bundle_extension_complete",
        "base_bundle": {
            "path": str(args.base_bundle),
            "sha256": sha256_file(args.base_bundle),
            "metadata_sha256": (
                sha256_file(base_metadata_path) if base_metadata_path.is_file() else None
            ),
            "methods": base_methods,
        },
        "appended_sources": sources,
        "methods": all_methods,
        "gnps_used_for_checkpoint_selection_or_tuning": False,
        "aggregation_deferred_to_evaluator": "maximum pair score within candidate molecule",
        "claim_limit": (
            "Score integration only. Method provenance and training-overlap strata must be "
            "reported; this file alone does not establish an independent external claim."
        ),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

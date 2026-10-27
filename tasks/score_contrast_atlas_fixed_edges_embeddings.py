#!/usr/bin/env python3
"""Attach frozen embedding similarities to an exact Contrast-Atlas edge axis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from encode_crossstudy_mgfs_checkpoint import sha256


KEY_COLUMNS = [
    "left_study_id",
    "right_study_id",
    "evidence_world",
    "left_entity_id",
    "right_entity_id",
]


def parse_specs(specifications: list[str]) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = defaultdict(list)
    for item in specifications:
        if "=" not in item:
            raise ValueError(f"invalid --embedding value: {item}")
        method, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not method or not path.is_file():
            raise FileNotFoundError(path)
        output[method].append(path)
    if not output:
        raise ValueError("at least one embedding cache is required")
    return dict(output)


def load_method(paths: list[Path]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as body:
            ids = np.asarray(body["entity_id"]).astype("U")
            matrix = np.asarray(body["embeddings"], dtype=np.float32)
        if matrix.ndim != 2 or len(ids) != matrix.shape[0]:
            raise RuntimeError(f"invalid embedding cache: {path}")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.all(np.isfinite(matrix)) or not np.allclose(norms, 1.0, atol=2e-4):
            raise RuntimeError(f"embedding cache is not finite and unit-normalised: {path}")
        for entity, vector in zip(ids.tolist(), matrix, strict=True):
            if entity in output:
                if not np.allclose(output[entity], vector, atol=1e-6):
                    raise RuntimeError(f"conflicting duplicate embedding for {entity}")
                continue
            output[entity] = vector
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-handoff", required=True, type=Path)
    parser.add_argument(
        "--embedding",
        action="append",
        required=True,
        help="Repeated METHOD=NPZ; the same method may have several disjoint caches",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.edge_handoff.is_file():
        raise FileNotFoundError(args.edge_handoff)

    specs = parse_specs(args.embedding)
    edges = pd.read_csv(args.edge_handoff, sep="\t", dtype={column: str for column in KEY_COLUMNS})
    if edges.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("fixed edge handoff contains duplicate join keys")

    method_reports: dict[str, object] = {}
    for method, paths in specs.items():
        cache = load_method(paths)
        left = edges["left_entity_id"].astype(str).to_numpy()
        right = edges["right_entity_id"].astype(str).to_numpy()
        available = np.asarray(
            [(a in cache) and (b in cache) for a, b in zip(left, right, strict=True)],
            dtype=bool,
        )
        score = np.full(len(edges), np.nan, dtype=np.float32)
        indices = np.flatnonzero(available)
        for index in indices:
            score[index] = float(np.dot(cache[left[index]], cache[right[index]]))
        edges[f"{method}_score"] = score
        edges[f"{method}_evidence_available"] = available
        edges[f"{method}_base_score_on_available_axis"] = np.where(
            available,
            pd.to_numeric(edges["chemical_similarity"], errors="coerce").to_numpy(float),
            np.nan,
        )
        method_reports[method] = {
            "embedding_entities": len(cache),
            "edge_rows_scored": int(available.sum()),
            "edge_rows_unavailable": int((~available).sum()),
            "all_world_unique_edges_scored": int(
                (available & edges["evidence_world"].eq("all").to_numpy()).sum()
            ),
            "score_min": float(np.nanmin(score)),
            "score_median": float(np.nanmedian(score)),
            "score_max": float(np.nanmax(score)),
            "caches": [
                {"path": str(path), "sha256": sha256(path)} for path in paths
            ],
        }

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "fixed_edge_embedding_scores.tsv.gz"
    edges.to_csv(
        output,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report = {
        "status": "CONTRAST_ATLAS_FIXED_EDGE_EMBEDDING_SCORES_COMPLETE",
        "data_role": "development_consumed",
        "labels_responses_or_relations_accessed": False,
        "candidate_contract": "exact input edge rows and join keys; no edge added or removed",
        "denominators": {
            "edge_rows": len(edges),
            "all_world_unique_edges": int(edges["evidence_world"].eq("all").sum()),
        },
        "methods": method_reports,
        "inputs": {
            "edge_handoff": {
                "path": str(args.edge_handoff),
                "sha256": sha256(args.edge_handoff),
            }
        },
        "output": {"path": str(output), "sha256": sha256(output)},
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checksums.sha256").write_text(
        f"{sha256(output)}  {output.name}\n{sha256(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

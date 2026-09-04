#!/usr/bin/env python
"""Fail-closed validator for the BioAware v2 experimental feature graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-formal-nodes", type=int, default=250)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_bioaware_v2_feature_graph_complete":
        raise RuntimeError("unexpected feature-graph status")
    if report.get("phenotype_labels_used") or report.get("candidate_identities_used"):
        raise RuntimeError("forbidden identity/phenotype information entered feature graph")
    for panel in report["panels"]:
        name = panel["panel"]
        nodes = pd.read_csv(args.output_dir / f"{name}__nodes.csv.gz")
        edges = pd.read_csv(args.output_dir / f"{name}__edges.csv.gz")
        cache = np.load(args.output_dir / f"{name}__feature_embeddings.npz")
        feature_id = cache["feature_id"]
        embedding = cache["embedding"]
        if len(nodes) != len(feature_id) or embedding.shape[0] != len(nodes):
            raise RuntimeError(f"{name}: node/embedding mismatch")
        if nodes["feature_id"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate feature nodes")
        if len(edges) and (
            edges[["feature_id_a", "feature_id_b"]].duplicated().any()
            or (edges["feature_id_a"] == edges["feature_id_b"]).any()
        ):
            raise RuntimeError(f"{name}: duplicate or self edges")
        node_set = set(nodes["feature_id"].astype(int))
        edge_set = set(edges["feature_id_a"].astype(int)) | set(edges["feature_id_b"].astype(int))
        if not edge_set <= node_set:
            raise RuntimeError(f"{name}: edge references absent node")
        norms = np.linalg.norm(embedding, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5):
            raise RuntimeError(f"{name}: feature embeddings are not unit-normalized")
        if report["formal"] and len(nodes) < args.minimum_formal_nodes:
            raise RuntimeError(f"{name}: only {len(nodes)} formal nodes")
    print(
        f"[validate_bioaware_v2_feature_graph] PASS panels={len(report['panels'])} "
        f"formal={report['formal']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract the public MetDNA2 KEGG reaction-pair graph to an auditable table."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-rda", type=Path, default=Path(
        "third_party/MetDNA2/data/reaction_pair_network.rda"))
    parser.add_argument("--compound-rda", type=Path, default=Path(
        "third_party/MetDNA2/data/cpd_emrn.rda"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/reference/metdna2_kegg_network_20260828"))
    args = parser.parse_args()
    for path in (args.network_rda, args.compound_rda):
        if not path.exists():
            raise FileNotFoundError(path)
    try:
        import pyreadr
        import rdata
    except ImportError as exc:
        raise RuntimeError("extractor requires pyreadr and rdata") from exc

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    compound_objects = pyreadr.read_r(str(args.compound_rda))
    compounds = compound_objects.get("cpd_emrn")
    if compounds is None:
        raise RuntimeError("cpd_emrn object missing")
    mapping = compounds[["id", "inchikey1", "source", "formula", "name"]].copy()
    mapping["id"] = mapping["id"].fillna("").astype(str)
    mapping["inchikey1"] = mapping["inchikey1"].fillna("").astype(str)
    mapping = mapping[(mapping["id"] != "") & (mapping["inchikey1"].str.len() == 14)]
    mapping = mapping.sort_values(["id", "source", "inchikey1"]).drop_duplicates("id", keep="first")

    network = rdata.read_rda(str(args.network_rda)).get("reaction_pair_network")
    if not isinstance(network, dict) or "version1" not in network:
        raise RuntimeError("MetDNA2 reaction_pair_network$version1 missing")
    graph = network["version1"]
    if not isinstance(graph, list) or len(graph) < 9:
        raise RuntimeError("unexpected igraph serialization")
    vertex_names = np.asarray(graph[8][2][np.str_("name")]).astype(str)
    source_index = np.asarray(graph[2], dtype=int)
    target_index = np.asarray(graph[3], dtype=int)
    if source_index.min() < 0 or target_index.min() < 0:
        raise RuntimeError("negative igraph vertex index")
    if source_index.max() >= len(vertex_names) or target_index.max() >= len(vertex_names):
        raise RuntimeError("igraph vertex index out of bounds")
    edges = pd.DataFrame({
        "source_kegg": vertex_names[source_index],
        "target_kegg": vertex_names[target_index],
    })
    lookup = mapping.set_index("id")["inchikey1"]
    edges["source_ik14"] = edges["source_kegg"].map(lookup).fillna("")
    edges["target_ik14"] = edges["target_kegg"].map(lookup).fillna("")
    edges = edges[(edges["source_ik14"].str.len() == 14) &
                  (edges["target_ik14"].str.len() == 14) &
                  (edges["source_ik14"] != edges["target_ik14"])].copy()
    canonical = np.sort(edges[["source_ik14", "target_ik14"]].to_numpy(str), axis=1)
    edges["ik14_a"] = canonical[:, 0]
    edges["ik14_b"] = canonical[:, 1]
    edges = edges.sort_values(["ik14_a", "ik14_b", "source_kegg", "target_kegg"])
    edges = edges.drop_duplicates(["ik14_a", "ik14_b"]).reset_index(drop=True)
    edge_path = output / "metdna2_kegg_edges.csv.gz"
    mapping_path = output / "metdna2_kegg_compounds.csv.gz"
    edges.to_csv(edge_path, index=False, compression="gzip")
    mapping.to_csv(mapping_path, index=False, compression="gzip")
    report = {
        "status": "metdna2_kegg_network_extraction_complete",
        "formal": True,
        "source": "MetDNA2 reaction_pair_network$version1 (public KEGG MRN)",
        "vertices_in_serialized_graph": int(len(vertex_names)),
        "edges_in_serialized_graph": int(len(source_index)),
        "mapped_unique_edges": int(len(edges)),
        "mapped_compounds": int(len(set(edges["ik14_a"]) | set(edges["ik14_b"]))),
        "provenance": {
            "network_rda_sha256": sha256(args.network_rda),
            "compound_rda_sha256": sha256(args.compound_rda),
            "edge_table_sha256": sha256(edge_path),
            "compound_table_sha256": sha256(mapping_path),
        },
        "contracts": {
            "version2_emrn_or_predicted_edges_used": False,
            "edge_direction": "undirected KEGG reaction pair",
            "outcome_labels_used": False,
        },
        "claim_limit": "Source extraction only; no annotation or coverage gain is claimed.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

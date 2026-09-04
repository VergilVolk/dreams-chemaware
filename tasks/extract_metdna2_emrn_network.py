#!/usr/bin/env python
"""Extract MetDNA2 eMRN edges with their earliest expansion step."""
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


def attribute_array(attributes: dict, name: str, size: int, default: object) -> np.ndarray:
    value = attributes.get(np.str_(name))
    if value is None:
        return np.full(size, default, dtype=object)
    if np.ma.isMaskedArray(value):
        mask = np.ma.getmaskarray(value)
        result = np.asarray(value.data, dtype=object)
        result[mask] = default
    else:
        result = np.asarray(value, dtype=object)
    if len(result) != size:
        raise RuntimeError(f"edge attribute {name} has {len(result)} rows, expected {size}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-rda", type=Path, default=Path(
        "third_party/MetDNA2/data/reaction_pair_network.rda"))
    parser.add_argument("--compound-rda", type=Path, default=Path(
        "third_party/MetDNA2/data/cpd_emrn.rda"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/reference/metdna2_emrn_network_20260828"))
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
    compounds = pyreadr.read_r(str(args.compound_rda)).get("cpd_emrn")
    if compounds is None:
        raise RuntimeError("cpd_emrn object missing")
    mapping = compounds[["id", "inchikey1", "source", "formula", "name", "type",
                         "min_reaction_step"]].copy()
    mapping["id"] = mapping["id"].fillna("").astype(str)
    mapping["inchikey1"] = mapping["inchikey1"].fillna("").astype(str)
    mapping = mapping[(mapping["id"] != "") & (mapping["inchikey1"].str.len() == 14)]
    mapping = mapping.sort_values(["id", "source", "inchikey1"]).drop_duplicates("id", keep="first")
    lookup = mapping.set_index("id")["inchikey1"]

    network = rdata.read_rda(str(args.network_rda)).get("reaction_pair_network")
    if not isinstance(network, dict) or not isinstance(network.get("version2"), dict):
        raise RuntimeError("MetDNA2 reaction_pair_network$version2 missing")
    graphs = network["version2"]
    frames: list[pd.DataFrame] = []
    cumulative: list[dict[str, int]] = []
    for step in range(9):
        key = f"step{step}"
        graph = graphs.get(key)
        if not isinstance(graph, list) or len(graph) < 9:
            raise RuntimeError(f"unexpected igraph serialization for {key}")
        vertex_names = np.asarray(graph[8][2][np.str_("name")]).astype(str)
        source_index = np.asarray(graph[2], dtype=int)
        target_index = np.asarray(graph[3], dtype=int)
        if source_index.max(initial=-1) >= len(vertex_names) or target_index.max(initial=-1) >= len(vertex_names):
            raise RuntimeError(f"igraph vertex index out of bounds for {key}")
        attributes = graph[8][3]
        size = len(source_index)
        frame = pd.DataFrame({
            "source_id": vertex_names[source_index], "target_id": vertex_names[target_index],
            "minimum_step": step,
            "edge_source": attribute_array(attributes, "source", size, ""),
            "edge_label": attribute_array(attributes, "label", size, ""),
            "reaction_id": attribute_array(attributes, "reaction_id", size, ""),
            "reaction": attribute_array(attributes, "reaction", size, ""),
            "structural_similarity": pd.to_numeric(
                attribute_array(attributes, "str_sim", size, np.nan), errors="coerce"),
        })
        frame["source_ik14"] = frame["source_id"].map(lookup).fillna("")
        frame["target_ik14"] = frame["target_id"].map(lookup).fillna("")
        frame = frame[(frame["source_ik14"].str.len() == 14) &
                      (frame["target_ik14"].str.len() == 14) &
                      (frame["source_ik14"] != frame["target_ik14"])].copy()
        canonical = np.sort(frame[["source_ik14", "target_ik14"]].to_numpy(str), axis=1)
        frame["ik14_a"] = canonical[:, 0]
        frame["ik14_b"] = canonical[:, 1]
        frame = frame.sort_values(["ik14_a", "ik14_b", "source_id", "target_id"])
        frame = frame.drop_duplicates(["ik14_a", "ik14_b"])
        frames.append(frame)
        cumulative.append({
            "step": step, "serialized_vertices": int(len(vertex_names)),
            "serialized_edges": int(size), "mapped_unique_edges": int(len(frame)),
        })
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values([
        "minimum_step", "ik14_a", "ik14_b", "source_id", "target_id"
    ]).drop_duplicates(["ik14_a", "ik14_b"], keep="first").reset_index(drop=True)
    edge_path = output / "metdna2_emrn_edges.csv.gz"
    mapping_path = output / "metdna2_emrn_compounds.csv.gz"
    combined.to_csv(edge_path, index=False, compression="gzip")
    mapping.to_csv(mapping_path, index=False, compression="gzip")
    new_by_step = combined.groupby("minimum_step").size().reindex(range(9), fill_value=0)
    report = {
        "status": "metdna2_emrn_network_extraction_complete", "formal": True,
        "source": "MetDNA2 reaction_pair_network$version2",
        "cumulative_serialized_graphs": cumulative,
        "unique_mapped_edges": int(len(combined)),
        "mapped_compounds": int(len(set(combined["ik14_a"]) | set(combined["ik14_b"]))),
        "new_unique_edges_by_minimum_step": {
            str(step): int(new_by_step.loc[step]) for step in range(9)
        },
        "provenance": {
            "network_rda_sha256": sha256(args.network_rda),
            "compound_rda_sha256": sha256(args.compound_rda),
            "edge_table_sha256": sha256(edge_path),
            "compound_table_sha256": sha256(mapping_path),
        },
        "contracts": {
            "minimum_step_preserved": True,
            "outcome_labels_used": False,
            "edge_direction": "undirected as serialized by MetDNA2",
        },
        "claim_limit": "Source extraction only; predicted eMRN edges require step-stratified validation.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze source-disjoint panel pairs before any Pan-ReDU response score exists."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_polarity(value: object) -> str:
    text = str(value).strip().lower()
    if "positive" in text or text in {"pos", "+"}:
        return "positive"
    if "negative" in text or text in {"neg", "-"}:
        return "negative"
    raise ValueError(f"unsupported or missing polarity: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--panel-metadata", required=True, type=Path)
    parser.add_argument("--required-polarity", choices=("positive", "negative"), default="positive")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    usecols = [
        "panel_id", "dataset_accession", "preliminary_source_family",
        "contrast_id", "topology_status",
    ]
    topology = pd.read_csv(args.topology, sep="\t", dtype=str, usecols=usecols)
    topology = topology.loc[topology.topology_status.str.startswith("PASS")].copy()
    metadata = pd.read_csv(args.panel_metadata, sep="\t", dtype=str, usecols=["panel_id", "polarity"])
    metadata["canonical_polarity"] = metadata.polarity.map(canonical_polarity)
    panel_polarity = metadata[["panel_id", "canonical_polarity"]].drop_duplicates()
    if panel_polarity.groupby("panel_id").canonical_polarity.nunique().max() != 1:
        raise RuntimeError("a panel maps to multiple polarities")
    missing_metadata = sorted(set(topology.panel_id) - set(panel_polarity.panel_id))
    if missing_metadata:
        raise RuntimeError(f"topology-pass panels lack polarity metadata: {missing_metadata}")
    topology_before_polarity = topology.copy()
    topology = topology.merge(panel_polarity, on="panel_id", how="left", validate="many_to_one")
    topology = topology.loc[topology.canonical_polarity.eq(args.required_polarity)].copy()
    if topology.empty:
        raise RuntimeError(f"no topology-pass panels have {args.required_polarity} polarity")
    panels = topology[["panel_id", "dataset_accession", "preliminary_source_family"]].drop_duplicates()
    panels = panels.merge(panel_polarity, on="panel_id", how="left", validate="one_to_one")
    if panels.groupby("panel_id").preliminary_source_family.nunique().max() != 1:
        raise RuntimeError("a panel maps to multiple source families")
    panels = panels.sort_values(["preliminary_source_family", "panel_id"], kind="stable")
    rows = []
    for left, right in itertools.combinations(panels.to_dict("records"), 2):
        if left["preliminary_source_family"] == right["preliminary_source_family"]:
            continue
        rows.append({
            "pair_index": len(rows),
            "left_panel_id": left["panel_id"], "right_panel_id": right["panel_id"],
            "left_accession": left["dataset_accession"], "right_accession": right["dataset_accession"],
            "left_source_family": left["preliminary_source_family"],
            "right_source_family": right["preliminary_source_family"],
            "left_polarity": left["canonical_polarity"],
            "right_polarity": right["canonical_polarity"],
        })
    pairs = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "panel_pairs.tsv"
    pairs.to_csv(path, sep="\t", index=False)
    contrast_counts = topology.groupby("panel_id").contrast_id.nunique()
    report = {
        "schema": "pan_redu_source_disjoint_partial_transport_pairs_v2_polarity_safe",
        "status": "PRETRUTH_SOURCE_DISJOINT_SAME_POLARITY_PANEL_PAIRS_FROZEN",
        "denominators": {
            "topology_pass_panels_before_polarity": int(topology_before_polarity.panel_id.nunique()),
            "topology_pass_contrasts_before_polarity": int(topology_before_polarity.contrast_id.nunique()),
            "topology_pass_panels": int(panels.panel_id.nunique()),
            "source_families": int(panels.preliminary_source_family.nunique()),
            "contrasts": int(topology.contrast_id.nunique()),
            "excluded_wrong_polarity_panels": int(
                topology_before_polarity.panel_id.nunique() - panels.panel_id.nunique()
            ),
            "excluded_wrong_polarity_contrasts": int(
                topology_before_polarity.contrast_id.nunique() - topology.contrast_id.nunique()
            ),
            "source_disjoint_panel_pairs": len(pairs),
            "ordered_contrast_queries": int(sum(
                2 * int(contrast_counts[left]) * int(contrast_counts[right])
                for left, right in zip(pairs.left_panel_id, pairs.right_panel_id)
            )),
        },
        "selection_rule": (
            "all unordered panel pairs with different preliminary source families, PASS topology, "
            f"and frozen {args.required_polarity} polarity"
        ),
        "required_polarity": args.required_polarity,
        "leakage_status": "topology and source provenance only; no abundance, effect, embedding, edge or retrieval score read",
        "provenance": {
            "topology_sha256": sha256(args.topology),
            "panel_metadata_sha256": sha256(args.panel_metadata),
            "panel_pairs_sha256": sha256(path),
        },
        "claim_boundary": "Source-disjoint pairing does not make ontology contrasts biologically comparable or independent truth.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Audit whether ChemProp2 M1 MS1-response retrieval is parent-drug driven.

The primary analysis uses MS1 abundance trajectories on feature IDs that have
non-empty MS/MS.  This audit removes (i) direct parent-drug library matches,
(ii) broader library-labelled drug-related ions, and (iii) every FBMN component
containing a direct parent-drug match.  The component exclusion is deliberately
conservative: it can remove metabolites, adducts, fragments, and unrelated
spectrally similar features, so it is a sensitivity upper bound rather than a
validated chemical deconvolution.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from analyze_chemprop2_response_reproducibility_m1 import (
    CONTROL_DRUGS,
    cluster_bootstrap_delta,
    feature_qc_mask,
    median_center,
    nonempty_mgf_scans,
    parse_samples,
    response_profiles,
    retrieval,
    robust_log2,
    sha256,
    summarize,
)


DRUG_PATTERNS = {
    "cilnidipine": r"cilnidipine",
    "clomifen": r"clomif(?:ene|en)|clomiphene|enclomiphene",
    "erythromycin": r"erythromycin",
    "ketoconazole": r"ketoconazole|\bket\b",
    "lansoprazole": r"lansoprazole",
    "loratadine": r"loratadine",
    "metronidazole": r"metronidazole",
    "montelukast": r"montelukast",
    "omeprazole": r"\bomeprazole\b",
    "sertraline": r"sertraline",
    "simvastatin": r"simvastatin",
    "telmisartan": r"telmisartan",
}
RELATED_TOKENS = re.compile(
    r"analogue|analog|related|fragment|hydrate|metabolite|transformation|standard",
    flags=re.IGNORECASE,
)


def labelled_drug_nodes(libraries: pd.DataFrame, score: float) -> tuple[set[int], set[int], pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    names = libraries["Compound_Name"].fillna("").astype(str)
    for drug, pattern in DRUG_PATTERNS.items():
        selected = libraries.loc[
            names.str.contains(pattern, case=False, regex=True)
            & libraries["MQScore"].ge(score)
        ].copy()
        selected["study_drug"] = drug
        rows.append(selected)
    matched = pd.concat(rows, ignore_index=True).drop_duplicates("#Scan#")
    matched["broad_related_label"] = matched["Compound_Name"].str.contains(RELATED_TOKENS)
    broad = set(matched["#Scan#"].astype(int))
    direct = set(matched.loc[~matched["broad_related_label"], "#Scan#"].astype(int))
    return direct, broad, matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/reference/chemprop2_m1_20261011"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/chemprop2_parent_drug_dependence_v2_20261011"),
    )
    parser.add_argument("--library-score", type=float, default=0.80)
    parser.add_argument("--blank-ratio", type=float, default=3.0)
    parser.add_argument("--qc-cv-max", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")

    paths = {
        "quantification": args.source_dir / "quantification.csv",
        "mgf": args.source_dir / "consensus.mgf",
        "library": args.source_dir / "library_matches.tsv",
        "network": args.source_dir / "network_singletons.graphml",
    }
    quant = pd.read_csv(paths["quantification"])
    samples = parse_samples(list(quant.columns))
    scans = nonempty_mgf_scans(paths["mgf"])
    analysis_mask, audit = feature_qc_mask(
        quant, samples, scans, args.blank_ratio, args.qc_cv_max,
    )
    libraries = pd.read_csv(paths["library"], sep="\t")
    libraries["#Scan#"] = pd.to_numeric(libraries["#Scan#"], errors="raise").astype(int)
    libraries["MQScore"] = pd.to_numeric(libraries["MQScore"], errors="raise")
    direct_nodes, broad_nodes, matched = labelled_drug_nodes(libraries, args.library_score)

    graph = nx.read_graphml(paths["network"])
    direct_graph_nodes = {str(node) for node in direct_nodes if str(node) in graph}
    component_nodes: set[int] = set()
    component_ids: set[object] = set()
    for node in direct_graph_nodes:
        component = nx.node_connected_component(graph, node)
        component_nodes.update(int(item) for item in component)
        component_ids.add(graph.nodes[node].get("component"))

    supported_nodes = set(
        libraries.loc[libraries["MQScore"].ge(args.library_score), "#Scan#"].astype(int)
    )
    row_ids = quant["row ID"].astype(int)
    supported = row_ids.isin(supported_nodes).to_numpy() & analysis_mask
    unresolved = analysis_mask & ~supported
    feature_ids = row_ids.to_numpy()[analysis_mask]

    values = quant[samples["column"].tolist()].to_numpy(float).T
    profiles, meta = response_profiles(
        median_center(robust_log2(values)), samples, analysis_mask, "trajectory_auc",
    )
    exclusions = {
        "none": set(),
        "direct_parent_library_nodes": direct_nodes,
        "broad_drug_related_library_nodes": broad_nodes,
        "parent_containing_fbmn_components": component_nodes,
    }
    universes = {
        "all_ms2_qc": np.ones(int(analysis_mask.sum()), dtype=bool),
        "unresolved": unresolved[analysis_mask],
    }
    results: dict[str, dict[str, object]] = {}
    frames: list[pd.DataFrame] = []
    baseline_frames: dict[str, pd.DataFrame] = {}
    for exclusion_name, excluded_ids in exclusions.items():
        keep = ~np.isin(feature_ids, list(excluded_ids))
        for universe_name, universe in universes.items():
            selected = keep & universe
            frame = retrieval(profiles, meta, selected)
            key = f"{exclusion_name}|{universe_name}"
            tagged = frame.copy()
            tagged.insert(0, "features", int(selected.sum()))
            tagged.insert(0, "universe", universe_name)
            tagged.insert(0, "exclusion", exclusion_name)
            frames.append(tagged)
            results[key] = {
                "features": int(selected.sum()),
                **summarize(frame),
            }
            if exclusion_name == "none":
                baseline_frames[universe_name] = frame
            else:
                results[key]["delta_mrr_vs_no_exclusion"] = cluster_bootstrap_delta(
                    frame, baseline_frames[universe_name], "reciprocal_rank",
                    args.seed + len(results),
                )

    matched = matched.sort_values(["study_drug", "MQScore"], ascending=[True, False])
    matched["in_analysis_universe"] = matched["#Scan#"].isin(set(feature_ids))
    matched["in_parent_component_exclusion"] = matched["#Scan#"].isin(component_nodes)
    args.output_dir.mkdir(parents=True)
    matched.to_csv(args.output_dir / "drug_labelled_library_nodes.csv", index=False)
    audit.to_csv(args.output_dir / "feature_qc_audit.csv.gz", index=False, compression="gzip")
    pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "retrieval_results.csv", index=False)

    report = {
        "status": "chemprop2_parent_drug_dependence_audit_complete",
        "formal": False,
        "evidence_role": "development_consumed_sensitivity",
        "dataset": {
            "accession": "MSV000094899",
            "queries": int(len(meta)),
            "drugs": int(meta["drug"].nunique()),
            "analysis_features": int(analysis_mask.sum()),
            "source_hashes": {name: sha256(path) for name, path in paths.items()},
        },
        "information_used_for_retrieval": (
            "MS1 abundance trajectories on natural feature IDs; MS/MS is used only to require "
            "a non-empty spectrum and to define library/network sensitivity masks"
        ),
        "exclusion_audit": {
            "library_score_threshold": args.library_score,
            "direct_parent_nodes_total": len(direct_nodes),
            "direct_parent_nodes_in_analysis": len(direct_nodes & set(feature_ids)),
            "broad_drug_related_nodes_total": len(broad_nodes),
            "broad_drug_related_nodes_in_analysis": len(broad_nodes & set(feature_ids)),
            "parent_containing_component_nodes_total": len(component_nodes),
            "parent_containing_component_nodes_in_analysis": len(component_nodes & set(feature_ids)),
            "parent_containing_components": len(component_ids),
            "component_exclusion_interpretation": (
                "conservative sensitivity upper bound, not validated metabolite/adduct deconvolution"
            ),
        },
        "retrieval": results,
        "does_not_establish": [
            "incremental value of MS/MS peaks or a spectral embedding",
            "cross-study entity matching or response transport",
            "complete removal of every drug-derived transformation product",
            "general structure-agnostic biological discovery",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

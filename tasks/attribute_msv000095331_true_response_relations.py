#!/usr/bin/env python3
"""Attribute registered response-retrieval scores to evidence categories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_msv000095331_chemprop_open_retrieval import (  # noqa: E402
    CHEMPROP_MGF,
    load_candidate_effects,
    load_new_effects,
)


TRUTH = ((1, "sertraline"), (3, "omeprazole"), (4, "loratadine"))


def new_names(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    for column in ("#Scan#", "MQScore", "SharedPeaks"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.loc[
        table["#Scan#"].notna()
        & table["MQScore"].ge(0.9)
        & table["SharedPeaks"].ge(5)
    ].sort_values(["#Scan#", "MQScore"], ascending=[True, False]).drop_duplicates("#Scan#")
    return dict(zip(
        table["#Scan#"].astype(int).astype(str), table["Compound_Name"].fillna("").astype(str)
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--common-identity-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    edges = pd.read_csv(
        args.retrieval_dir / "phenotype_blind_mutual_top1_edges.tsv.gz",
        sep="\t",
        dtype={"left_raw_entity_id": str, "right_raw_entity_id": str},
    )
    edges = edges.loc[edges["evidence_world"].eq("all")].copy()
    new_effects = load_new_effects(args.response_dir / "effects.tsv")
    old_effects = load_candidate_effects(
        args.collection / "response_entities.tsv.gz", set(CHEMPROP_MGF)
    )
    query_names = new_names(args.library)
    query_identity = pd.read_csv(
        args.response_dir / "source_native_identity.tsv", sep="\t", dtype={"entity_id": str}
    ).set_index("entity_id")["identity_status"].to_dict()
    target_names, target_identity = {}, {}
    for study in CHEMPROP_MGF:
        table = pd.read_csv(
            args.common_identity_root / study / "identity_partition.tsv.gz",
            sep="\t", dtype={"raw_entity_id": str},
        )
        target_names[study] = dict(zip(
            table["raw_entity_id"], table["library_name"].fillna("").astype(str)
        ))
        target_identity[study] = dict(zip(table["raw_entity_id"], table["identity_status"]))

    rows = []
    for mixture, drug in TRUTH:
        query_id = f"drug_mix_{mixture}_pooled_microbiota"
        query = new_effects.loc[new_effects["contrast_id"].eq(query_id)].set_index("raw_entity_id")
        for study in ("MSV000094899", "MSV000093571"):
            candidate_id = f"drug_{drug}"
            candidate = old_effects.loc[
                old_effects["study_id"].eq(study)
                & old_effects["contrast_id"].eq(candidate_id)
            ].set_index("raw_entity_id")
            block = edges.loc[edges["right_study_id"].eq(study)]
            contributions = []
            for edge in block.itertuples(index=False):
                if edge.left_raw_entity_id not in query.index or edge.right_raw_entity_id not in candidate.index:
                    continue
                weight = (
                    float(query.loc[edge.left_raw_entity_id, "response_weight"])
                    * float(candidate.loc[edge.right_raw_entity_id, "response_weight"])
                    * float(edge.chemical_similarity)
                )
                if weight <= 0:
                    continue
                query_name = query_names.get(str(edge.left_raw_entity_id), "")
                candidate_name = target_names[study].get(str(edge.right_raw_entity_id), "")
                contributions.append({
                    "weight": weight,
                    "identity_track": edge.edge_track == "precursor_compatible_identity_hypothesis",
                    "both_unresolved": (
                        query_identity.get(str(edge.left_raw_entity_id)) == "unresolved_entity"
                        and target_identity[study].get(str(edge.right_raw_entity_id)) == "unresolved_entity"
                    ),
                    "drug_named": drug in query_name.lower() or drug in candidate_name.lower(),
                })
            contribution = pd.DataFrame(contributions)
            total = float(contribution["weight"].sum())
            rows.append({
                "query_contrast_id": query_id,
                "candidate_study_id": study,
                "candidate_contrast_id": candidate_id,
                "aligned_edge_count": int(len(contribution)),
                "aligned_weight_total": total,
                "precursor_compatible_weight_fraction": float(
                    contribution.loc[contribution["identity_track"], "weight"].sum() / total
                ),
                "both_unresolved_weight_fraction": float(
                    contribution.loc[contribution["both_unresolved"], "weight"].sum() / total
                ),
                "drug_named_weight_fraction": float(
                    contribution.loc[contribution["drug_named"], "weight"].sum() / total
                ),
            })
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "true_relation_attribution.tsv", sep="\t", index=False)
    report = {
        "status": "MSV000095331_TRUE_RELATION_ATTRIBUTION_COMPLETE",
        "data_role": "post-result interpretation; development only",
        "registered_relations": len(summary),
        "precursor_compatible_weight_fraction_range": [
            float(summary["precursor_compatible_weight_fraction"].min()),
            float(summary["precursor_compatible_weight_fraction"].max()),
        ],
        "both_unresolved_weight_fraction_range": [
            float(summary["both_unresolved_weight_fraction"].min()),
            float(summary["both_unresolved_weight_fraction"].max()),
        ],
        "drug_named_weight_fraction_range": [
            float(summary["drug_named_weight_fraction"].min()),
            float(summary["drug_named_weight_fraction"].max()),
        ],
        "claim_boundary": [
            "Drug-named classification uses post-result public-library labels only for interpretation.",
            "Low drug-named fraction argues against direct parent/propagated-drug peaks dominating the score; it does not establish an endogenous mechanism.",
            "Mass-compatible pairs remain identity hypotheses rather than confirmed identical molecules.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

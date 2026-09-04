"""BioAware context audit for LCNEC priority dark-metabolite hypotheses.

The graph is used for context and hub abstention only.  It does not override
the frozen spectral candidate.  High-degree currency metabolites are explicitly
barred from serving as pathway-specific evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure-ledger", type=Path,
        default=Path("data/validation/lcnec_hsst3n_priority_structure/priority_structure_ledger.csv"),
    )
    parser.add_argument(
        "--participants", type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"),
    )
    parser.add_argument(
        "--reactions", type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_reactions.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/lcnec_hsst3n_bioaware_context"),
    )
    parser.add_argument("--maximum-specific-degree", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure = pd.read_csv(args.structure_ledger)
    participants = pd.read_csv(args.participants)
    reactions = pd.read_csv(args.reactions)
    rows = []
    path_rows = []
    for hypothesis in structure.itertuples(index=False):
        body = participants[participants["compound_id"].eq(hypothesis.ik14)]
        reaction_ids = sorted(set(body["reaction_id"].astype(int)))
        linked = reactions[reactions["reaction_id"].isin(reaction_ids)]
        reactome = linked[linked["reactome_ids"].notna()]
        degree = int(body["reaction_degree"].max()) if len(body) else 0
        currency = bool(body["is_currency"].any()) if len(body) else False
        specific = bool(
            len(body) and not currency and degree <= args.maximum_specific_degree
            and reactome["reaction_id"].nunique() >= 1
        )
        rows.append({
            "priority_name": hypothesis.priority_name,
            "ik14": hypothesis.ik14,
            "rhea_reactions": len(reaction_ids),
            "reaction_degree": degree,
            "is_currency": currency,
            "reactome_linked_reactions": int(reactome["reaction_id"].nunique()),
            "bioaware_specific_anchor": specific,
            "hub_abstention": bool(currency or degree > args.maximum_specific_degree),
        })
        for reaction in reactome.itertuples(index=False):
            for reactome_id in str(reaction.reactome_ids).split(";"):
                path_rows.append({
                    "priority_name": hypothesis.priority_name,
                    "ik14": hypothesis.ik14,
                    "reaction_id": int(reaction.reaction_id),
                    "rhea_master_id": int(reaction.rhea_master_id),
                    "reactome_id": reactome_id,
                    "direction_semantics": str(reaction.direction_semantics),
                })
    ledger = pd.DataFrame(rows)
    paths = pd.DataFrame(path_rows)
    report = {
        "status": "lcnec_hsst3n_bioaware_context_complete",
        "formal": True,
        "hypotheses": len(ledger),
        "specific_nonhub_anchors": int(ledger["bioaware_specific_anchor"].sum()),
        "hub_abstentions": int(ledger["hub_abstention"].sum()),
        "rows": ledger.to_dict("records"),
        "decision": (
            "Quinolinate, ADP-ribose and ascorbate may anchor pathway context. ADP is retained as a spectral abundance "
            "hypothesis but is forbidden as pathway-specific BioAware evidence because it is a currency hub."
        ),
        "contracts": {
            "spectral_candidate_changed": False,
            "phenotype_used_for_identity": False,
            "currency_metabolites_can_activate_pathway_claim": False,
            "network_role": "context and abstention only",
        },
        "claim_limit": (
            "Reaction membership supports biochemical plausibility, not reaction direction, flux, enzyme activity, or causal disease mechanism."
        ),
    }
    report["pass"] = report["specific_nonhub_anchors"] >= 3 and report["hub_abstentions"] >= 1
    ledger.to_csv(args.output_dir / "bioaware_context_ledger.csv", index=False)
    paths.to_csv(args.output_dir / "reactome_paths.csv", index=False)
    (args.output_dir / "bioaware_context_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

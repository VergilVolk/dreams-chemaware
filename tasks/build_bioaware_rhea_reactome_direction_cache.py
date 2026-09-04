#!/usr/bin/env python
"""Add conservative Reactome-supported direction semantics to the frozen Rhea cache.

Rhea left/right serialization is not a biological direction.  The only rows
promoted here are Rhea master reactions whose Reactome cross-references agree
on exactly one directed representation (LR or RL).  Mixed, undefined, or
unmapped reactions remain direction-unknown.  The original participant sides
are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consensus_semantics(values: pd.Series) -> str:
    directions = {str(value).strip().upper() for value in values if str(value).strip()}
    # Undefined evidence contaminates the direction call.  This conservative
    # rule prevents a single directed mapping from overriding an incompatible
    # or explicitly undefined Reactome mapping.
    if directions == {"LR"}:
        return "reactome_consensus_lr"
    if directions == {"RL"}:
        return "reactome_consensus_rl"
    if directions == {"LR", "RL"}:
        return "reactome_consensus_bidirectional"
    return "reaction_direction_unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/reference/bioaware_rhea_reactome_direction_20260830"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    participants_path = source / "rhea_participants.csv.gz"
    reactions_path = source / "rhea_reactions.csv.gz"
    mapping_path = source / "rhea2reactome.tsv"
    for path in (participants_path, reactions_path, mapping_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(participants_path, dtype={"reaction_id": str})
    reactions = pd.read_csv(reactions_path, dtype={"reaction_id": str})
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str)
    required_mapping = {"MASTER_ID", "DIRECTION", "ID"}
    if not required_mapping <= set(mapping.columns):
        raise RuntimeError(f"Reactome mapping missing columns: {sorted(required_mapping - set(mapping.columns))}")
    if not {"reaction_id", "direction_semantics"} <= set(participants.columns):
        raise RuntimeError("participant cache lacks direction columns")

    mapping["MASTER_ID"] = mapping["MASTER_ID"].astype(str)
    semantic_by_master = mapping.groupby("MASTER_ID", sort=True)["DIRECTION"].apply(consensus_semantics)
    reactome_count = mapping.groupby("MASTER_ID", sort=True)["ID"].nunique().astype(int)
    direction_count = mapping.groupby("MASTER_ID", sort=True)["DIRECTION"].nunique().astype(int)

    participants["direction_semantics"] = participants["reaction_id"].map(semantic_by_master).fillna(
        "reaction_direction_unknown"
    )
    reactions["direction_semantics"] = reactions["reaction_id"].map(semantic_by_master).fillna(
        "reaction_direction_unknown"
    )
    reactions["reactome_mapping_count"] = reactions["reaction_id"].map(reactome_count).fillna(0).astype(int)
    reactions["reactome_direction_count"] = reactions["reaction_id"].map(direction_count).fillna(0).astype(int)

    participant_out = output / "rhea_participants.csv.gz"
    reaction_out = output / "rhea_reactions.csv.gz"
    participants.to_csv(participant_out, index=False)
    reactions.to_csv(reaction_out, index=False)

    reaction_semantics = reactions["direction_semantics"].value_counts().sort_index().to_dict()
    participant_semantics = participants["direction_semantics"].value_counts().sort_index().to_dict()
    directed = reactions["direction_semantics"].isin({"reactome_consensus_lr", "reactome_consensus_rl"})
    report = {
        "status": "bioaware_rhea_reactome_direction_cache_complete",
        "formal": True,
        "source_reactions": int(len(reactions)),
        "source_participant_rows": int(len(participants)),
        "reactome_mapped_master_reactions": int(mapping["MASTER_ID"].nunique()),
        "directed_consensus_reactions": int(directed.sum()),
        "directed_consensus_fraction": float(directed.mean()),
        "reaction_semantics": {str(key): int(value) for key, value in reaction_semantics.items()},
        "participant_semantics": {str(key): int(value) for key, value in participant_semantics.items()},
        "contract": {
            "rhea_left_right_is_not_biological_direction": True,
            "direction_source": "Reactome cross-reference DIRECTION consensus per Rhea master reaction",
            "mixed_or_undefined_mapping": "reaction_direction_unknown",
            "unmapped_reaction": "reaction_direction_unknown",
            "participant_sides_rewritten": False,
        },
        "provenance": {
            "source_participants_sha256": sha256(participants_path),
            "source_reactions_sha256": sha256(reactions_path),
            "rhea2reactome_sha256": sha256(mapping_path),
            "participants_sha256": sha256(participant_out),
            "reactions_sha256": sha256(reaction_out),
        },
        "claim_limit": (
            "Reactome-supported direction is available only for consensus-mapped reactions; "
            "this cache does not establish organism-, tissue-, compartment-, or condition-specific flux direction."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

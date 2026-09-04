#!/usr/bin/env python
"""Build a versioned Rhea reaction-hypergraph cache for BioAware v1.

Rhea's ``rhea-reaction-smiles.tsv`` contains two artificial directed encodings
(left-to-right and right-to-left).  BioAware v1 stores only the canonical LR
encoding and labels it explicitly as *not a physiological direction*.  The
default expert therefore propagates undirected one-hop evidence.  A future
directed model must use curated Reactome/Rhea direction and compartment data,
not the arbitrary LR serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors


RHEA_BASE = "https://ftp.expasy.org/databases/rhea/tsv"
FILES = {
    "reaction_smiles": "rhea-reaction-smiles.tsv",
    "directions": "rhea-directions.tsv",
    "reactome": "rhea2reactome.tsv",
}

# Rhea explicitly contains isolated protons.  RDKit emits one warning for every
# such participant even though the structures are parsed correctly; suppress
# only warnings, not errors, to keep formal logs auditable.
RDLogger.DisableLog("rdApp.warning")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    # ``requests`` uses certifi in the project environment.  Python's urllib
    # can fail on some Windows hosts while loading the system certificate store.
    # TLS verification remains enabled; we never fall back to verify=False.
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as out:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    out.write(chunk)
    temporary.replace(path)


def molecule_record(smiles: str) -> dict | None:
    if "*" in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not inchikey:
        return None
    return {
        "compound_id": inchikey[:14],
        "full_inchikey": inchikey,
        "participant_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": float(Descriptors.ExactMolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
    }


def split_side(side_smiles: str) -> list[str]:
    # Rhea reaction SMILES use dot-separated participants.  Components inside a
    # participant are already represented as one charged structure where needed.
    return [piece for piece in side_smiles.split(".") if piece]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/reference/bioaware_rhea"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-reactions", type=int, default=0, help="Smoke only; 0 means all")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    raw_dir = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw_paths = {}
    for label, filename in FILES.items():
        path = raw_dir / filename
        if args.force_download or not path.exists():
            print(f"[download] {filename}", flush=True)
            download(f"{RHEA_BASE}/{filename}", path)
        if path.stat().st_size == 0:
            raise RuntimeError(f"empty Rhea input: {path}")
        raw_paths[label] = path

    directions = pd.read_csv(raw_paths["directions"], sep="\t")
    required = {"RHEA_ID_MASTER", "RHEA_ID_LR", "RHEA_ID_RL", "RHEA_ID_BI"}
    if not required <= set(directions.columns):
        raise RuntimeError(f"unexpected rhea-directions schema: {directions.columns.tolist()}")
    directions = directions.astype(int)
    lr_to_master = dict(zip(directions["RHEA_ID_LR"], directions["RHEA_ID_MASTER"]))

    reaction_smiles = pd.read_csv(
        raw_paths["reaction_smiles"], sep="\t", names=["rhea_directional_id", "reaction_smiles"],
        header=None, dtype={"rhea_directional_id": int, "reaction_smiles": str},
    )
    reaction_smiles = reaction_smiles[reaction_smiles["rhea_directional_id"].isin(lr_to_master)].copy()
    reaction_smiles["rhea_master_id"] = reaction_smiles["rhea_directional_id"].map(lr_to_master)
    reaction_smiles = reaction_smiles.sort_values("rhea_master_id")
    if args.max_reactions:
        reaction_smiles = reaction_smiles.head(args.max_reactions)

    reactome = pd.read_csv(raw_paths["reactome"], sep="\t", dtype=str)
    reactome_by_master = (
        reactome.groupby("MASTER_ID")["ID"].agg(lambda values: ";".join(sorted(set(values.astype(str)))))
        if {"MASTER_ID", "ID"} <= set(reactome.columns)
        else pd.Series(dtype=str)
    )

    participants: list[dict] = []
    reactions: list[dict] = []
    invalid_reactions = 0
    skipped_wildcard_or_invalid = 0
    for position, row in enumerate(reaction_smiles.itertuples(index=False), start=1):
        text = str(row.reaction_smiles)
        if ">>" not in text:
            invalid_reactions += 1
            continue
        left_text, right_text = text.split(">>", 1)
        reaction_rows = []
        for side, side_text in [("left", left_text), ("right", right_text)]:
            for participant_smiles in split_side(side_text):
                record = molecule_record(participant_smiles)
                if record is None:
                    skipped_wildcard_or_invalid += 1
                    continue
                record.update(
                    {
                        "reaction_id": str(int(row.rhea_master_id)),
                        "rhea_lr_id": int(row.rhea_directional_id),
                        "side": side,
                        "stoichiometry": 1,
                        "reaction_weight": 1.0,
                        "direction_semantics": "canonical_lr_not_physiological",
                    }
                )
                reaction_rows.append(record)
        if len({record["side"] for record in reaction_rows}) < 2:
            invalid_reactions += 1
            continue
        participants.extend(reaction_rows)
        reactions.append(
            {
                "reaction_id": str(int(row.rhea_master_id)),
                "rhea_master_id": int(row.rhea_master_id),
                "rhea_lr_id": int(row.rhea_directional_id),
                "reactome_ids": str(reactome_by_master.get(str(int(row.rhea_master_id)), "")),
                "direction_semantics": "canonical_lr_not_physiological",
                "n_left": sum(record["side"] == "left" for record in reaction_rows),
                "n_right": sum(record["side"] == "right" for record in reaction_rows),
            }
        )
        if position % 2500 == 0:
            print(f"[parse] {position:,}/{len(reaction_smiles):,}", flush=True)

    p = pd.DataFrame(participants).drop_duplicates(["reaction_id", "side", "compound_id"])
    if p.empty:
        raise RuntimeError("Rhea parsing yielded no valid participants")
    degree = p.groupby("compound_id")["reaction_id"].nunique().astype(int)
    p["reaction_degree"] = p["compound_id"].map(degree).astype(int)
    # Only obvious graph-dominating hubs and single-heavy-atom participants are
    # automatically excluded.  The flag is auditable and can be replaced by a
    # curated currency list without changing the expert API.
    p["is_currency"] = (p["heavy_atoms"] <= 1) | (p["reaction_degree"] > 500)
    r = pd.DataFrame(reactions).drop_duplicates("reaction_id")

    participants_path = out / "rhea_participants.csv.gz"
    reactions_path = out / "rhea_reactions.csv.gz"
    p.to_csv(participants_path, index=False)
    r.to_csv(reactions_path, index=False)
    top_hubs = degree.sort_values(ascending=False).head(25)
    report = {
        "status": "bioaware_rhea_cache_complete",
        "formal": args.max_reactions == 0,
        "reactions": int(r["reaction_id"].nunique()),
        "participant_rows": int(len(p)),
        "unique_compounds_ik14": int(p["compound_id"].nunique()),
        "currency_rows": int(p["is_currency"].sum()),
        "invalid_reactions": int(invalid_reactions),
        "skipped_wildcard_or_invalid_participants": int(skipped_wildcard_or_invalid),
        "direction_contract": "canonical LR serialization retained as metadata; BioAware v1 must use undirected propagation unless curated physiological direction is supplied",
        "top_hubs": {str(k): int(v) for k, v in top_hubs.items()},
        "provenance": {
            label: {"url": f"{RHEA_BASE}/{FILES[label]}", "sha256": sha256(path), "bytes": path.stat().st_size}
            for label, path in raw_paths.items()
        },
        "artifacts": {
            "participants": str(participants_path),
            "participants_sha256": sha256(participants_path),
            "reactions": str(reactions_path),
            "reactions_sha256": sha256(reactions_path),
        },
        "parameters": {"max_reactions": args.max_reactions},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

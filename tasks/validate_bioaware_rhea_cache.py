#!/usr/bin/env python
"""Fail-closed validation for a BioAware Rhea cache."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--minimum-reactions", type=int, default=10000)
    parser.add_argument("--minimum-compounds", type=int, default=5000)
    args = parser.parse_args()

    root = args.cache_dir.resolve()
    report_path = root / "report.json"
    participants_path = root / "rhea_participants.csv.gz"
    reactions_path = root / "rhea_reactions.csv.gz"
    for path in [report_path, participants_path, reactions_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing/empty cache artifact: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    participants = pd.read_csv(participants_path)
    reactions = pd.read_csv(reactions_path)
    required = {
        "reaction_id", "side", "compound_id", "full_inchikey", "formula", "exact_mass",
        "reaction_degree", "is_currency", "direction_semantics",
    }
    missing = required - set(participants)
    if missing:
        raise RuntimeError(f"participant schema missing {sorted(missing)}")
    if not set(participants.side) <= {"left", "right"}:
        raise RuntimeError("invalid reaction side")
    side_counts = participants.groupby("reaction_id")["side"].nunique()
    if (side_counts != 2).any():
        raise RuntimeError(f"{int((side_counts != 2).sum())} reactions do not have both sides")
    if participants.duplicated(["reaction_id", "side", "compound_id"]).any():
        raise RuntimeError("duplicate reaction-side-compound rows")
    if not participants["compound_id"].astype(str).str.fullmatch(r"[A-Z]{14}").all():
        raise RuntimeError("compound_id must be IK14")
    if set(participants["direction_semantics"]) != {"canonical_lr_not_physiological"}:
        raise RuntimeError("direction semantics drifted")
    degree = participants.groupby("compound_id")["reaction_id"].nunique().astype(int)
    stored = participants.groupby("compound_id")["reaction_degree"].first().astype(int)
    if not degree.sort_index().equals(stored.sort_index()):
        raise RuntimeError("stored reaction_degree does not reproduce")
    if participants["reaction_id"].nunique() != reactions["reaction_id"].nunique():
        raise RuntimeError("reaction table/participant table mismatch")
    if participants["reaction_id"].nunique() < args.minimum_reactions:
        raise RuntimeError("insufficient reaction coverage")
    if participants["compound_id"].nunique() < args.minimum_compounds:
        raise RuntimeError("insufficient compound coverage")
    expected = report.get("artifacts", {})
    if expected.get("participants_sha256") != sha256(participants_path):
        raise RuntimeError("participant hash mismatch")
    if expected.get("reactions_sha256") != sha256(reactions_path):
        raise RuntimeError("reaction hash mismatch")
    print(
        json.dumps(
            {
                "status": "bioaware_rhea_cache_validation_passed",
                "reactions": int(participants["reaction_id"].nunique()),
                "compounds": int(participants["compound_id"].nunique()),
                "participant_rows": int(len(participants)),
                "currency_rows": int(participants["is_currency"].astype(bool).sum()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


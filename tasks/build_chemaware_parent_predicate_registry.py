"""Freeze RDKit functional-group queries as parent-predicate candidates.

These SMARTS are a hypothesis vocabulary, not fragmentation rules.  A pair of
functional-group predicate and spectral observation becomes an empirical rule
only after formula-disjoint association and matched-control confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Fragments


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rdkit_fragment_patterns() -> list[dict]:
    patterns = []
    for index, (name, function) in enumerate(Fragments.fns):
        defaults = function.__defaults__ or ()
        query = next((value for value in reversed(defaults) if isinstance(value, Chem.Mol)), None)
        if query is None:
            raise RuntimeError(f"cannot recover RDKit SMARTS for {name}")
        smarts = Chem.MolToSmarts(query)
        if Chem.MolFromSmarts(smarts) is None:
            raise RuntimeError(f"RDKit emitted invalid SMARTS for {name}")
        patterns.append({
            "predicate_id": f"RDKIT_FG:{index:03d}:{name}",
            "name": name,
            "smarts_any": [smarts],
            "scientific_status": "parent_structure_hypothesis_only",
            "may_define_fragmentation_rule": False,
        })
    if len({item["predicate_id"] for item in patterns}) != len(patterns):
        raise RuntimeError("duplicate parent predicate id")
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_parent_predicates_rdkit_v1.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "data/validation/chemaware_parent_predicates_v1/report.json",
    )
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite parent-predicate artifacts")
    patterns = rdkit_fragment_patterns()
    body = {
        "schema": "chemaware_parent_predicate_registry_v1",
        "scientific_type": "structure_predicate_hypothesis_vocabulary",
        "predicates": patterns,
        "contracts": {
            "fragmentation_claim": False,
            "association_claim": False,
            "formula_disjoint_empirical_confirmation_required": True,
            "matched_formula_controls_required": True,
        },
        "provenance": {
            "source": "RDKit Chem.Fragments built-in functional-group queries",
            "rdkit_version": rdBase.rdkitVersion,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2), encoding="utf-8")
    report = {
        "status": "CHEMAWARE_PARENT_PREDICATE_REGISTRY_PASS",
        "formal_rules_admitted": 0,
        "predicate_candidates": len(patterns),
        "registry_sha256": sha256_file(args.output),
        "next_gate": "formula-disjoint predicate-observation association with matched-formula controls",
    }
    args.report.parent.mkdir(parents=True, exist_ok=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

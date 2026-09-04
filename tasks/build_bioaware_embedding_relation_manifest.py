#!/usr/bin/env python
"""Build P3-disjoint typed supervision for BioAware embedding adaptation.

This manifest is deliberately independent of P2b and phenotype labels.  Rhea
neighbours are auxiliary relation labels, never same-molecule positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from annotation.bioaware_relations import typed_reaction_pairs  # noqa: E402


FORBIDDEN_COLUMNS = {"phenotype", "group", "disease", "case", "control", "qvalue", "pvalue"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ], dtype=str)


def load_allowed(path: Path | None, formal: bool) -> set[str] | None:
    if path is None or not path.exists():
        if formal:
            raise FileNotFoundError("formal manifest requires sealed P3 allow-list")
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("p3_query_overlap") != 0:
        raise RuntimeError("P3 allow-list reports nonzero overlap")
    allowed = set(map(str, body["real_train_primary"]["ik14"]))
    if not allowed:
        raise RuntimeError("P3 allow-list is empty")
    return allowed


def add_pair(
    rows: list[dict], seen: set[tuple[str, str, str]], a: str, b: str,
    relation: str, formula_by_identity: dict[str, str], **extra,
) -> None:
    if a == b and relation != "same_identity":
        return
    key = (a, b, relation)
    if key in seen:
        return
    seen.add(key)
    rows.append({
        "identity_a": a,
        "identity_b": b,
        "formula_a": formula_by_identity[a],
        "formula_b": formula_by_identity[b],
        "relation_type": relation,
        **extra,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--participants", type=Path, default=ROOT / "data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz")
    parser.add_argument("--pairs", type=Path, default=ROOT / "tasks/massspecgym_isomers/pairs.json")
    parser.add_argument("--p3-allow", type=Path, default=ROOT / "data/validation/g8r_p3_test/p3_p2_allowed_training_ik14.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/bioaware_embedding_relation_manifest")
    parser.add_argument("--formula-folds", type=int, default=5)
    parser.add_argument("--max-rows-per-identity-adduct", type=int, default=12)
    parser.add_argument("--controls-per-identity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--formal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    required = [args.data, args.participants, args.pairs]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"fail-closed: output exists: {args.output_dir}")
    allowed = load_allowed(args.p3_allow, args.formal)

    with h5py.File(args.data, "r") as handle:
        keys = set(handle.keys())
        suspicious = sorted(key for key in keys if any(token in key.lower() for token in FORBIDDEN_COLUMNS))
        if suspicious:
            raise RuntimeError(f"phenotype-like HDF5 columns forbidden: {suspicious}")
        fold = decode(handle["fold"][:])
        simulation = decode(handle["SIMULATION_CHALLENGE"][:])
        ik_full = decode(handle["INCHIKEY"][:])
        formula = decode(handle["FORMULA"][:])
        adduct = decode(handle["adduct"][:])
        instrument = decode(handle["INSTRUMENT_TYPE"][:])
        ce = np.asarray(handle["COLLISION_ENERGY"][:], dtype=np.float64)
        precursor_mz = np.asarray(handle["precursor_mz"][:], dtype=np.float64)
    ik14 = np.asarray([value[:14] for value in ik_full], dtype=str)
    valid = (fold == "train") & (simulation == "False") & (np.char.str_len(ik14) == 14)
    if allowed is not None:
        valid &= np.isin(ik14, sorted(allowed))
    indices = np.flatnonzero(valid)
    row_table = pd.DataFrame({
        "row": indices,
        "ik14": ik14[indices],
        "formula": formula[indices],
        "adduct": adduct[indices],
        "instrument": instrument[indices],
        "collision_energy": ce[indices],
        "precursor_mz": precursor_mz[indices],
    })
    row_table = row_table.sort_values(["ik14", "adduct", "instrument", "collision_energy", "row"])
    row_table = row_table.groupby(["ik14", "adduct"], sort=False).head(args.max_rows_per_identity_adduct)
    formula_counts = row_table.groupby("ik14")["formula"].nunique()
    ambiguous = set(formula_counts[formula_counts != 1].index.astype(str))
    if ambiguous:
        row_table = row_table[~row_table["ik14"].isin(ambiguous)]
    formula_by_identity = row_table.groupby("ik14")["formula"].first().astype(str).to_dict()
    identities = sorted(formula_by_identity)
    if args.formal and len(identities) < 2000:
        raise RuntimeError(f"too few P3-disjoint real identities: {len(identities)}")

    pair_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for identity in identities:
        add_pair(pair_rows, seen, identity, identity, "same_identity", formula_by_identity,
                 reaction_ids="", evidence_count=0, mces_grade=-1)

    pair_body = json.loads(args.pairs.read_text(encoding="utf-8"))
    structural_edges: set[tuple[str, str]] = set()
    grade_map = {"near": 1, "mid": 4, "far": 8, "uncomputed": 11}
    for family, entries in pair_body.items():
        grade = grade_map.get(str(family), 11)
        for item in entries:
            a, b = str(item["ik_a"])[:14], str(item["ik_b"])[:14]
            if a not in formula_by_identity or b not in formula_by_identity:
                continue
            canonical = tuple(sorted((a, b)))
            structural_edges.add(canonical)
            relation = "near_isomer" if family == "near" else f"structural_{family}"
            add_pair(pair_rows, seen, canonical[0], canonical[1], relation, formula_by_identity,
                     reaction_ids="", evidence_count=1, mces_grade=grade)

    participants = pd.read_csv(args.participants)
    reaction_pairs, reaction_report = typed_reaction_pairs(participants, identities)
    reaction_edges: set[tuple[str, str]] = set()
    for pair in reaction_pairs:
        reaction_edges.add(tuple(sorted((pair.identity_a, pair.identity_b))))
        add_pair(
            pair_rows, seen, pair.identity_a, pair.identity_b, pair.relation_type,
            formula_by_identity, reaction_ids=";".join(pair.reaction_ids),
            evidence_count=pair.evidence_count, mces_grade=-1,
        )

    # Label-free, deterministic controls.  Same-formula controls are preferred;
    # mass controls are sampled from a narrow precursor-mass neighbourhood.
    rng = np.random.default_rng(args.seed)
    identities_by_formula: dict[str, list[str]] = defaultdict(list)
    mass_by_identity = row_table.groupby("ik14")["precursor_mz"].median().to_dict()
    for identity in identities:
        identities_by_formula[formula_by_identity[identity]].append(identity)
    ordered_mass = sorted(identities, key=lambda value: (mass_by_identity[value], value))
    position = {value: index for index, value in enumerate(ordered_mass)}
    for identity in identities:
        forbidden = {identity}
        forbidden.update(other for edge in structural_edges | reaction_edges if identity in edge for other in edge)
        same_formula = [value for value in identities_by_formula[formula_by_identity[identity]] if value not in forbidden]
        same_formula = sorted(same_formula)
        rng.shuffle(same_formula)
        selected = same_formula[: args.controls_per_identity]
        for other in selected:
            a, b = sorted((identity, other))
            add_pair(pair_rows, seen, a, b, "formula_matched_control", formula_by_identity,
                     reaction_ids="", evidence_count=0, mces_grade=-1)
        needed = args.controls_per_identity - len(selected)
        if needed > 0:
            center = position[identity]
            neighbours = []
            for distance in range(1, 101):
                for candidate_index in (center - distance, center + distance):
                    if 0 <= candidate_index < len(ordered_mass):
                        candidate = ordered_mass[candidate_index]
                        if candidate not in forbidden and candidate not in selected:
                            neighbours.append(candidate)
                if len(neighbours) >= needed:
                    break
            for other in neighbours[:needed]:
                a, b = sorted((identity, other))
                add_pair(pair_rows, seen, a, b, "mass_matched_control", formula_by_identity,
                         reaction_ids="", evidence_count=0, mces_grade=-1)

    pair_table = pd.DataFrame(pair_rows)
    pair_table["formula_fold_a"] = pair_table["formula_a"].map(
        lambda value: stable_fold(str(value), args.formula_folds, args.seed)
    )
    pair_table["formula_fold_b"] = pair_table["formula_b"].map(
        lambda value: stable_fold(str(value), args.formula_folds, args.seed)
    )
    pair_table["same_formula"] = pair_table["formula_a"] == pair_table["formula_b"]
    relation_counts = pair_table["relation_type"].value_counts().sort_index().to_dict()
    report = {
        "status": "bioaware_embedding_relation_manifest_complete",
        "formal": args.formal,
        "real_train_rows": int(len(row_table)),
        "identities": int(len(identities)),
        "formulas": int(row_table["formula"].nunique()),
        "pairs": int(len(pair_table)),
        "relation_counts": {str(k): int(v) for k, v in relation_counts.items()},
        "reaction_audit": reaction_report,
        "ambiguous_identity_formulas_excluded": len(ambiguous),
        "p3_identity_overlap": 0 if allowed is not None else None,
        "contracts": {
            "reaction_neighbour_is_positive": False,
            "same_identity_is_only_retrieval_positive": True,
            "reaction_type_is_auxiliary_supervision": True,
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "inference": "one shared clean-spectrum encoder",
            "outer_fold": "evaluation formulas are removed from both pair endpoints",
        },
        "gates": {
            "identities_ge_2000": len(identities) >= 2000,
            "near_pairs_ge_500": relation_counts.get("near_isomer", 0) >= 500,
            "reaction_pairs_ge_100": sum(v for k, v in relation_counts.items() if str(k).startswith("reaction_")) >= 100,
            "controls_ge_1000": sum(v for k, v in relation_counts.items() if str(k).endswith("_control")) >= 1000,
            "p3_disjoint": allowed is not None,
        },
        "provenance": {
            "hdf5_sha256": sha256(args.data),
            "participants_sha256": sha256(args.participants),
            "pairs_sha256": sha256(args.pairs),
            "p3_allow_sha256": sha256(args.p3_allow) if args.p3_allow.exists() else None,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Training manifest only; no embedding or retrieval improvement is claimed.",
    }
    if args.formal and not all(report["gates"].values()):
        raise RuntimeError(f"formal BioAware relation manifest gates failed: {report['gates']}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bioaware_relation_", dir=args.output_dir.parent))
    try:
        row_table.to_csv(staging / "rows.csv.gz", index=False)
        pair_table.to_csv(staging / "identity_pairs.csv.gz", index=False)
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.output_dir.exists():
            if not args.overwrite:
                raise RuntimeError("output appeared during build")
            shutil.rmtree(args.output_dir)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


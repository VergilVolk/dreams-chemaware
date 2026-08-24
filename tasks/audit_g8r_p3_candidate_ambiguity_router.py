"""Post-hoc audit of a candidate-set structural-ambiguity safety firewall.

The proposed deployment-visible rule is deliberately simple: when any two
candidate molecules in the strict-10ppm candidate set form a cached MCES-near
pair, retain DreaMS; otherwise allow the frozen P2b ranking.  This script runs
on consumed P3 only to test the failure mechanism.  It must not be used to
claim a new test result or select a threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from audit_g8r_p3_transition_mechanisms import as_bool, sha256_file, transition_counts


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERY = ROOT / "data/validation/g8r_p2b_p3_final.per_query.csv"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p3_candidate_ambiguity_audit.json"

PANEL_FILES = {
    "P3-main-real-pristine": "p3_main_real_pristine_manifest.json",
    "P3-isomer-real-pristine": "p3_isomer_real_pristine_manifest.json",
    "P3-near-core-real-pristine": "p3_near_core_real_pristine_manifest.json",
    "P3-nearmid-real-pristine": "p3_nearmid_real_pristine_manifest.json",
    "P3-isomer-real-exposed-extension": "p3_isomer_real_exposed_extension_manifest.json",
    "P3-sim-to-real-secondary": "p3_sim_to_real_secondary_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def text_array(dataset) -> np.ndarray:
    values = dataset[:]
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def relation_sets(path: Path) -> dict[str, set[tuple[str, str]]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for grade in ("near", "mid", "far"):
        result[grade] = {
            tuple(sorted((str(row["ik_a"])[:14], str(row["ik_b"])[:14])))
            for row in body.get(grade, [])
        }
    return result


def candidate_ambiguity(
    identities: list[str],
    formula_by_identity: dict[str, str],
    relations: dict[str, set[tuple[str, str]]],
) -> dict[str, bool | int]:
    identities = sorted(set(identities))
    pairs = [tuple(sorted(pair)) for pair in itertools.combinations(identities, 2)]
    formulas = [formula_by_identity.get(identity, "") for identity in identities]
    return {
        "n_candidate_identities": len(identities),
        "has_any_near_pair": any(pair in relations["near"] for pair in pairs),
        "has_any_mid_pair": any(pair in relations["mid"] for pair in pairs),
        "has_any_nearmid_pair": any(
            pair in relations["near"] or pair in relations["mid"] for pair in pairs
        ),
        "has_any_same_formula_pair": any(
            left and left == right for left, right in itertools.combinations(formulas, 2)
        ),
    }


def panel_report(rows: list[dict]) -> dict:
    near = np.asarray([row["has_any_near_pair"] for row in rows], dtype=bool)
    nearmid = np.asarray([row["has_any_nearmid_pair"] for row in rows], dtype=bool)
    same_formula = np.asarray([row["has_any_same_formula_pair"] for row in rows], dtype=bool)
    policies = {
        "frozen_p2b": np.ones(len(rows), dtype=bool),
        "fallback_if_candidate_set_has_near": ~near,
        "fallback_if_candidate_set_has_nearmid": ~nearmid,
        "fallback_if_candidate_set_has_same_formula_pair": ~same_formula,
    }
    transition_by_near = defaultdict(Counter)
    for row in rows:
        base, fusion = row["dreams_top1"], row["p2b_frozen_top1"]
        transition = (
            "corrected" if not base and fusion else
            "introduced" if base and not fusion else
            "persistent_right" if base else "persistent_wrong"
        )
        transition_by_near[str(bool(row["has_any_near_pair"]))][transition] += 1
    return {
        "n_queries": len(rows),
        "candidate_set_near_fraction": float(near.mean()),
        "candidate_set_nearmid_fraction": float(nearmid.mean()),
        "candidate_set_same_formula_fraction": float(same_formula.mean()),
        "policies_diagnostic_only": {
            name: transition_counts(rows, use) for name, use in policies.items()
        },
        "transition_by_candidate_set_has_near": {
            key: dict(value) for key, value in transition_by_near.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    required = [args.per_query, args.pairs, args.data]
    required += [args.p3_dir / filename for filename in PANEL_FILES.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(args.data, "r") as handle:
        ik14_by_row = np.asarray([value[:14] for value in text_array(handle["INCHIKEY"])])
        formulas = text_array(handle["FORMULA"])
    formula_by_identity: dict[str, str] = {}
    for identity, formula in zip(ik14_by_row, formulas):
        if identity not in formula_by_identity and formula not in {"", "nan", "None"}:
            formula_by_identity[str(identity)] = str(formula)
    relations = relation_sets(args.pairs)

    per_query = {}
    with args.per_query.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["panel"], int(row["query_row"]))
            if key in per_query:
                raise RuntimeError(f"duplicate per-query result: {key}")
            row["dreams_top1"] = as_bool(row["dreams_top1"])
            row["p2b_frozen_top1"] = as_bool(row["p2b_frozen_top1"])
            per_query[key] = row

    panels: dict[str, list[dict]] = defaultdict(list)
    for panel, filename in PANEL_FILES.items():
        manifest = json.loads((args.p3_dir / filename).read_text(encoding="utf-8"))
        for query in manifest["queries"]:
            key = (panel, int(query["row"]))
            if key not in per_query:
                raise RuntimeError(f"missing P3 score row: {key}")
            candidate_identities = [str(ik14_by_row[int(row)]) for row in query["candidate_rows"]]
            ambiguity = candidate_ambiguity(candidate_identities, formula_by_identity, relations)
            panels[panel].append({**per_query[key], **ambiguity})
    expected = sum(len(rows) for rows in panels.values())
    if expected != len(per_query):
        raise RuntimeError(f"manifest/result row mismatch: {expected} != {len(per_query)}")

    report = {
        "status": "g8r_p3_candidate_ambiguity_audit_complete",
        "source_per_query_sha256": sha256_file(args.per_query),
        "pairs_sha256": sha256_file(args.pairs),
        "p3_is_consumed": True,
        "may_be_used_for_model_selection": False,
        "panels": {panel: panel_report(rows) for panel, rows in panels.items()},
        "claim_boundary": (
            "These are post-hoc mechanism diagnostics. The MCES threshold was not fitted "
            "here, but any resulting router still requires P2-only development and a new "
            "sealed external evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

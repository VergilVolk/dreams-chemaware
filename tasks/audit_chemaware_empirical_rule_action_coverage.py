"""Measure whether admitted empirical rules cover enough training actions.

This is a CPU-only eligibility screen.  It inspects structures and observed
peaks only in action discovery/confirmation formula folds; it never encodes a
spectrum, reads retrieval outcomes, or touches embedding-evaluation folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_retrieval_graph import RetrievalGraph  # noqa: E402


def stable_formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little") % folds
        for value in np.asarray(formulas).astype(str)
    ], dtype=np.int16)


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--rules", type=Path, default=ROOT / "dreams/models/chem_aware/chem_action_knowledge_v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_empirical_rule_action_coverage_v1/report.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--discovery-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--minimum-identities", type=int, default=20)
    parser.add_argument("--minimum-formulas", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.minimum_identities < 20 or args.minimum_formulas < 10:
        raise ValueError("rule-action coverage thresholds were weakened")
    graph = RetrievalGraph(args.graph)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    formula = graph.query_formula[selected].astype(str)
    fold = stable_formula_folds(formula, args.folds, args.fold_seed)
    inspected = np.flatnonzero(np.isin(fold, (*args.discovery_folds, args.confirmation_fold)))
    rules = json.loads(args.rules.read_text(encoding="utf-8"))["rules"]
    compiled = []
    for rule in rules:
        queries = [Chem.MolFromSmarts(value) for value in rule["parent_predicate"]["smarts_any"]]
        if any(value is None for value in queries):
            raise RuntimeError(f"invalid admitted SMARTS: {rule['rule_id']}")
        compiled.append((rule, queries))
    active = np.zeros(len(selected), dtype=bool)
    matched_rule_count = np.zeros(len(selected), dtype=np.int16)
    matched_peak_count = np.zeros(len(selected), dtype=np.int16)
    rows = graph.query_row[selected]
    with h5py.File(args.data, "r") as handle:
        for position in inspected:
            row = int(rows[position])
            molecule = Chem.MolFromSmiles(decode(handle["smiles"][row]))
            if molecule is None:
                continue
            spectrum = np.asarray(handle["spectrum"][row], dtype=np.float64)
            valid_mz = spectrum[0, (spectrum[0] > 0) & (spectrum[1] >= 0.01)]
            target_keys = set()
            for rule, queries in compiled:
                if not any(molecule.HasSubstructMatch(query) for query in queries):
                    continue
                matched_rule_count[position] += 1
                observation = rule["observation"]
                target_keys.add((float(observation["exact_mass_da"]), float(observation["tolerance_da"])))
            for target, tolerance in target_keys:
                matched_peak_count[position] += int(np.any(np.abs(valid_mz - target) <= tolerance))
            active[position] = matched_peak_count[position] > 0
    splits = {
        "discovery": np.flatnonzero(np.isin(fold, args.discovery_folds)),
        "confirmation": np.flatnonzero(fold == args.confirmation_fold),
    }
    counts = {}
    gates = {}
    for name, positions in splits.items():
        enabled = positions[active[positions]]
        counts[name] = {
            "eligible_queries": int(len(enabled)),
            "eligible_identities": int(len(np.unique(graph.query_ik14[selected[enabled]].astype(str)))),
            "eligible_formulas": int(len(np.unique(formula[enabled]))),
            "queries_with_parent_predicate": int(np.sum(matched_rule_count[positions] > 0)),
            "queries_with_observed_action_peak": int(np.sum(active[positions])),
        }
        gates[f"{name}_identity_coverage"] = counts[name]["eligible_identities"] >= args.minimum_identities
        gates[f"{name}_formula_coverage"] = counts[name]["eligible_formulas"] >= args.minimum_formulas
    passed = bool(all(gates.values()))
    report = {
        "status": "CHEMAWARE_RULE_ACTION_COVERAGE_PASS" if passed else "CHEMAWARE_RULE_ACTION_COVERAGE_FAIL",
        "pass_to_frozen_embedding_headroom": passed,
        "formal_training_authorized": False,
        "admitted_rules": len(rules), "counts": counts, "gates": gates,
        "scope": {
            "weights_updated": False, "retrieval_outcomes_read": False,
            "embedding_evaluation_formulas_inspected": False,
            "outer_formulas_inspected": False,
            "overlapping_rules_union_without_dose_accumulation": True,
            "new_mz_synthesis": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output.parent / "coverage.npz", selected_query=selected, formula=formula,
        formula_fold=fold, active=active, matched_rule_count=matched_rule_count,
        matched_peak_count=matched_peak_count,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

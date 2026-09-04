"""Audit candidate-specific, structure-derived peak evidence without touching the outer fold.

The audit asks a deliberately narrow question before any model change: do simple
single-bond structural fragments from the true molecule explain more query-peak
intensity than the official DreaMS hard-negative molecule?  A positive result is
only evidence that a training-only structural peak view may be useful; it is not
an annotation method or a fragmentation-mechanism claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

from noise_final_core import CandidateGraph, stable_fold


PROTON = 1.007276466621


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@lru_cache(maxsize=None)
def single_cut_fragments(smiles: str) -> tuple[tuple[float, str], ...]:
    """Return unique ``(exact mass, canonical SMILES)`` single-cut fragments."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumBonds() == 0:
        return ()
    values: dict[str, float] = {}
    for bond in molecule.GetBonds():
        editable = Chem.RWMol(molecule)
        editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for fragment in Chem.GetMolFrags(editable, asMols=True, sanitizeFrags=False):
            try:
                text = Chem.MolToSmiles(fragment, canonical=True)
                clean = Chem.MolFromSmiles(text)
                if clean is not None and clean.GetNumHeavyAtoms() > 0:
                    values[text] = round(float(ExactMolWt(clean)), 6)
            except (RuntimeError, ValueError):
                continue
    if not values:
        return ()
    return tuple(sorted((mass, text) for text, mass in values.items()))


@lru_cache(maxsize=None)
def single_cut_fragment_masses(smiles: str) -> np.ndarray:
    """Return unique exact masses of sanitized fragments from one bond cut."""

    return np.unique(np.asarray([mass for mass, _ in single_cut_fragments(smiles)], dtype=np.float64))


def matched_mask(observed: np.ndarray, theoretical: np.ndarray, ppm: float, floor_da: float) -> np.ndarray:
    if len(observed) == 0 or len(theoretical) == 0:
        return np.zeros(len(observed), dtype=bool)
    theoretical = np.sort(np.asarray(theoretical, dtype=np.float64))
    right = np.searchsorted(theoretical, observed)
    left = np.clip(right - 1, 0, len(theoretical) - 1)
    right = np.clip(right, 0, len(theoretical) - 1)
    distance = np.minimum(np.abs(observed - theoretical[left]), np.abs(observed - theoretical[right]))
    tolerance = np.maximum(floor_da, np.abs(observed) * ppm * 1e-6)
    return distance <= tolerance


def evidence(peaks: np.ndarray, precursor_mz: float, smiles: str, ppm: float, floor_da: float) -> dict:
    mz = np.asarray(peaks[0], dtype=np.float64)
    intensity = np.asarray(peaks[1], dtype=np.float64)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0) & (mz < precursor_mz)
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) == 0:
        return {"union": 0.0, "neutral_loss": 0.0, "fragment": 0.0, "matched_peaks": 0}
    intensity = intensity / np.sum(intensity)
    masses = single_cut_fragment_masses(smiles)
    neutral_loss = precursor_mz - mz
    nl_match = matched_mask(neutral_loss, masses, ppm, floor_da)
    # Fragment ions can be represented approximately by their neutral mass or
    # protonated mass.  The union is used only as an audit feature.
    fragment_theory = np.unique(np.concatenate((masses, masses + PROTON))) if len(masses) else masses
    fragment_match = matched_mask(mz, fragment_theory, ppm, floor_da)
    union = nl_match | fragment_match
    return {
        "union": float(np.sum(intensity[union])),
        "neutral_loss": float(np.sum(intensity[nl_match])),
        "fragment": float(np.sum(intensity[fragment_match])),
        "matched_peaks": int(np.sum(union)),
    }


def strict_rank(scores: np.ndarray) -> int:
    return 1 + int(np.sum(scores[1:] >= scores[0]))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_queries": 0}
    truth = np.asarray([row["truth_union"] for row in rows])
    hard = np.asarray([row["hard_negative_union"] for row in rows])
    best = np.asarray([row["best_negative_union"] for row in rows])
    rank = np.asarray([row["structural_rank"] for row in rows])
    official_rank = np.asarray([row["official_rank"] for row in rows])
    baseline_error = official_rank != 1
    return {
        "n_queries": len(rows),
        "mean_truth_union": float(np.mean(truth)),
        "mean_hard_negative_union": float(np.mean(hard)),
        "mean_truth_minus_hard": float(np.mean(truth - hard)),
        "median_truth_minus_hard": float(np.median(truth - hard)),
        "truth_gt_hard_fraction": float(np.mean(truth > hard)),
        "truth_gt_best_negative_fraction": float(np.mean(truth > best)),
        "structural_hit1": float(np.mean(rank == 1)),
        "structural_mrr": float(np.mean(1.0 / rank)),
        "baseline_errors": int(np.sum(baseline_error)),
        "baseline_error_truth_gt_hard_fraction": (
            float(np.mean(truth[baseline_error] > hard[baseline_error])) if np.any(baseline_error) else None
        ),
        "baseline_error_mean_truth_minus_hard": (
            float(np.mean((truth - hard)[baseline_error])) if np.any(baseline_error) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--floor-da", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    graph = CandidateGraph(args.graph)
    args.output.mkdir(parents=True, exist_ok=False)
    query_fold = np.asarray(
        [stable_fold(value, args.folds, args.fold_seed) for value in graph.query_formula.astype(str)],
        dtype=np.int8,
    )
    selected = np.flatnonzero(query_fold != args.outer_fold)
    rows: list[dict] = []
    with h5py.File(args.hdf5, "r") as handle:
        for query in selected:
            query_row = int(graph.query_row[query])
            peaks = np.asarray(handle["spectrum"][query_row], dtype=np.float64)
            precursor = float(handle["precursor_mz"][query_row])
            molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
            official = graph.official_molecule_scores(query)
            official_rank = strict_rank(official)
            scores = []
            details = []
            smiles_values = []
            for molecule in range(molecule_left, molecule_right):
                pair_left = int(graph.molecule_ptr[molecule])
                candidate_row = int(graph.pair_candidate_row[pair_left])
                smiles = decode(handle["smiles"][candidate_row])
                item = evidence(peaks, precursor, smiles, args.ppm, args.floor_da)
                scores.append(item["union"])
                details.append(item)
                smiles_values.append(smiles)
            scores = np.asarray(scores, dtype=np.float64)
            hard_index = 1 + int(np.argmax(official[1:]))
            best_negative = 1 + int(np.argmax(scores[1:]))
            rows.append({
                "query": int(query),
                "fold": int(query_fold[query]),
                "split": "inner" if query_fold[query] == args.inner_fold else "train",
                "query_row": query_row,
                "query_ik14": str(graph.query_ik14[query]),
                "query_formula": str(graph.query_formula[query]),
                "n_candidates": int(len(scores)),
                "official_rank": official_rank,
                "official_margin": float(official[0] - official[hard_index]),
                "structural_rank": strict_rank(scores),
                "truth_union": float(scores[0]),
                "truth_neutral_loss": details[0]["neutral_loss"],
                "truth_fragment": details[0]["fragment"],
                "truth_matched_peaks": details[0]["matched_peaks"],
                "hard_negative_ik14": str(graph.molecule_ik14[molecule_left + hard_index]),
                "hard_negative_union": float(scores[hard_index]),
                "truth_minus_hard": float(scores[0] - scores[hard_index]),
                "best_negative_ik14": str(graph.molecule_ik14[molecule_left + best_negative]),
                "best_negative_union": float(scores[best_negative]),
                "truth_minus_best_negative": float(scores[0] - scores[best_negative]),
                "truth_smiles": smiles_values[0],
                "hard_negative_smiles": smiles_values[hard_index],
            })

    fieldnames = list(rows[0]) if rows else []
    with (args.output / "per_query.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "diagnostic_only",
        "outer_fold_not_evaluated": int(args.outer_fold),
        "inner_fold": int(args.inner_fold),
        "matching": {"ppm": args.ppm, "floor_da": args.floor_da, "single_bond_cuts": True},
        "train": summarize([row for row in rows if row["split"] == "train"]),
        "inner": summarize([row for row in rows if row["split"] == "inner"]),
        "claim_limit": (
            "Single-cut mass matches are approximate training-view evidence only; they do not prove "
            "fragment identities, mechanisms, or deployable annotation performance."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

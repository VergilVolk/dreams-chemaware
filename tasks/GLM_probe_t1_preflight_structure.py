"""GLM-authored CPU probe: T1 outer-train preflight, structure, CE, margins.

Four cheap, independent digs that change the T1/T3 route decision:

1. T1 mining census on outer-train formulas (conservative lower bound):
   how many graph queries outside formula fold 0 carry same-formula
   near/mid/far rival molecules with measured spectra -- the raw material
   for relation-aware negative pools.  Stage-6 died for lack of exactly this
   preflight.
2. Tanimoto structure distribution of top rivals (Morgan r=2, 2048 bit):
   near-1 rivals on residual errors are label-noise / unresolvable-stereoisomer
   suspects and refine the realistic ceiling of every training route.
3. Collision-energy alignment test on residual errors: does a |dCE|<=10 eV
   pool filter raise naive cosine separability (an unused harvestable axis)?
4. Flip-distance of stage-1 rank-2/3 residuals in the frozen dreams geometry:
   how much margin separates the true molecule from the top rival.

Read-only; no training; held ledgers are consumed only for stratification.

Author: GLM-5.3 (DeepSeek Harness session).  Outputs are prefixed GLM.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from GLM_probe_noise_rival_separability import (
    cosine_similarity,
    sha256_file,
)

PROBE_VERSION = "GLM_t1_preflight_structure_v1"
FORMULA_FOLD_SEED = 20260825
OUTER_FOLDS = 5
CE_WINDOW_EV = 10.0


def stable_fold(value: str, folds: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--held-per-query", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    for path in (args.graph, args.held_per_query, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    with gzip.open(args.held_per_query, "rt", encoding="utf-8", newline="") as s:
        ledger = list(csv.DictReader(s))
    with np.load(args.graph, allow_pickle=False) as body:
        graph = {name: body[name] for name in body.files}
    feature_names = list(map(str, graph["feature_names"]))
    score_column = feature_names.index("dreams_similarity")
    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    molecule_label = np.asarray(graph["molecule_label"], dtype=np.int8)
    molecule_ik14 = np.asarray(graph["molecule_ik14"]).astype(str)
    molecule_formula = np.asarray(graph["molecule_formula"]).astype(str)
    molecule_grade = np.asarray(graph["molecule_mces_grade"], dtype=np.int8)
    query_row = np.asarray(graph["query_row"], dtype=np.int64)
    query_formula = np.asarray(graph["query_formula"]).astype(str)
    pair_candidate_row = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
    pair_scores = np.asarray(graph["features"][:, score_column], dtype=np.float32)
    molecule_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

    # ---- Section 1: T1 outer-train census (conservative lower bound). ----
    held_query_set = {int(row["query_index"]) for row in ledger}
    held_folds = set()
    train_queries = []
    for query in range(len(query_row)):
        fold = stable_fold(str(query_formula[query]), OUTER_FOLDS, FORMULA_FOLD_SEED)
        if query in held_query_set:
            held_folds.add(fold)
            continue
        if fold != 0:
            train_queries.append(query)
    census = {
        "train_queries_examined": len(train_queries),
        "held_fold_values_seen": sorted(int(f) for f in held_folds),
        "validation_formula_note": (
            "stage1 validation formula set is not local; excluding ALL fold-0 "
            "formulas is stricter than the real T1 filter, so every census "
            "number is a lower bound"
        ),
    }
    per_query_negatives = {"near": [], "mid": [], "far": [], "any_same_formula": []}
    queries_with = {"near": 0, "mid": 0, "far": 0, "any": 0}
    multi_spectrum_rivals = 0
    total_rival_blocks = 0
    events_cap3 = 0
    for query in train_queries:
        left, right = int(query_ptr[query]), int(query_ptr[query + 1])
        formula = str(query_formula[query])
        counts = {"near": 0, "mid": 0, "far": 0}
        any_same = 0
        for molecule in range(left + 1, right):
            if str(molecule_formula[molecule]) != formula:
                continue
            grade = int(molecule_grade[molecule])
            spectra_count = int(molecule_ptr[molecule + 1] - molecule_ptr[molecule])
            total_rival_blocks += 1
            if spectra_count >= 2:
                multi_spectrum_rivals += 1
            if grade == 0:
                counts["near"] += 1
            elif grade == 1:
                counts["mid"] += 1
            elif grade == 2:
                counts["far"] += 1
            any_same += 1
        for key in counts:
            per_query_negatives[key].append(counts[key])
            if counts[key] > 0:
                queries_with[key] += 1
        per_query_negatives["any_same_formula"].append(any_same)
        if any_same > 0:
            queries_with["any"] += 1
        events_cap3 += min(any_same, 3)

    def _hist(values: list[int]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.int64)
        return {
            "mean": float(array.mean()),
            "queries_with_at_least_1": int(np.sum(array >= 1)),
            "queries_with_at_least_2": int(np.sum(array >= 2)),
            "queries_with_at_least_3": int(np.sum(array >= 3)),
        }

    census["same_formula_rival_molecules_per_query"] = {
        key: _hist(values) for key, values in per_query_negatives.items()
    }
    census["queries_with_any_same_formula_rival"] = queries_with["any"]
    census["fraction_train_queries_with_relation"] = (
        queries_with["any"] / len(train_queries) if train_queries else 0.0
    )
    census["queries_with_near"] = queries_with["near"]
    census["queries_with_mid"] = queries_with["mid"]
    census["queries_with_far"] = queries_with["far"]
    census["rival_blocks_total"] = total_rival_blocks
    census["rival_blocks_with_at_least_2_spectra"] = multi_spectrum_rivals
    census["fraction_rival_blocks_multi_spectrum"] = (
        multi_spectrum_rivals / total_rival_blocks if total_rival_blocks else 0.0
    )
    census["events_if_capped_at_3_per_query"] = events_cap3

    # Held concentration for context.
    held_ik14s = [row["query_ik14"] for row in ledger]
    census["held_distinct_molecules"] = len(set(held_ik14s))
    census["held_distinct_formulas"] = len(set(
        str(query_formula[int(row["query_index"])]) for row in ledger
    ))

    # ---- Shared loading for sections 2-4 ----
    # Rival selection identical to the relation audit (verified 0 disagreements
    # in the separability probe).
    plan: list[dict[str, object]] = []
    needed_rows: set[int] = set()
    for source in ledger:
        query = int(source["query_index"])
        left, right = int(query_ptr[query]), int(query_ptr[query + 1])
        scores = molecule_scores[left:right]
        rival = left + 1 + int(np.argmax(scores[1:]))
        rival_rows = pair_candidate_row[
            molecule_ptr[rival]:molecule_ptr[rival + 1]
        ]
        pos_rows = pair_candidate_row[
            molecule_ptr[left]:molecule_ptr[left + 1]
        ]
        pos_rows = pos_rows[pos_rows != int(query_row[query])]
        plan.append({
            "source": source, "query": query, "rival": int(rival),
            "pos_rows": pos_rows.astype(np.int64),
            "rival_rows": rival_rows.astype(np.int64),
        })
        needed_rows.add(int(query_row[query]))
        needed_rows.update(int(r) for r in pos_rows)
        needed_rows.update(int(r) for r in rival_rows)
    sorted_rows = np.asarray(sorted(needed_rows), dtype=np.int64)
    row_position = {int(row): i for i, row in enumerate(sorted_rows)}
    print(f"[GLM preflight] loading {len(sorted_rows)} rows", flush=True)
    with h5py.File(args.data, "r") as handle:
        all_spectra = np.asarray(handle["spectrum"][()], dtype=np.float32)
        spectra = all_spectra[sorted_rows]
        precursors = np.asarray(handle["precursor_mz"][()])[sorted_rows]
        energies = np.asarray(handle["COLLISION_ENERGY"][()])[sorted_rows]
        smiles_all = np.asarray(handle["smiles"][()]).astype(str)
        smiles = smiles_all[sorted_rows]
        del all_spectra, smiles_all
    counts = np.sum(
        (spectra[:, 0, :] > 0) & (spectra[:, 1, :] > 0), axis=1,
    ).astype(np.int64)
    peaks = [
        (np.ascontiguousarray(spectra[i, 0, : int(counts[i])]),
         np.ascontiguousarray(spectra[i, 1, : int(counts[i])]))
        for i in range(len(sorted_rows))
    ]
    del spectra

    # ---- Section 2: Tanimoto structure distribution ----
    fingerprints: dict[int, object] = {}

    def fingerprint(position: int):
        if position not in fingerprints:
            molecule = Chem.MolFromSmiles(str(smiles[position]))
            if molecule is None:
                fingerprints[position] = False
            else:
                fingerprints[position] = AllChem.GetMorganFingerprintAsBitVect(
                    molecule, radius=2, nBits=2048,
                )
        return fingerprints[position]

    stage1_errors = [p for p in plan if int(p["source"]["stage1_rank"]) > 1]
    official_errors = [p for p in plan if int(p["source"]["official_rank"]) > 1]

    def tanimoto_set(plans: list[dict[str, object]]) -> dict[str, object]:
        values: list[float] = []
        failures = 0
        for entry in plans:
            qpos = row_position[int(query_row[int(entry["query"])])]
            rrow = int(entry["rival_rows"][0])
            fp_q = fingerprint(qpos)
            fp_r = fingerprint(row_position[rrow])
            if fp_q is False or fp_r is False:
                failures += 1
                continue
            values.append(float(DataStructs.TanimotoSimilarity(fp_q, fp_r)))
        array = np.asarray(values, dtype=np.float64)
        return {
            "compared": len(values),
            "smiles_failures": failures,
            "mean": float(array.mean()) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "fraction_ge_0.90": float(np.mean(array >= 0.90)) if len(array) else None,
            "fraction_ge_0.95": float(np.mean(array >= 0.95)) if len(array) else None,
            "fraction_ge_0.99": float(np.mean(array >= 0.99)) if len(array) else None,
            "fraction_le_0.60": float(np.mean(array <= 0.60)) if len(array) else None,
        }

    tanimoto_report = {
        "background_all_queries": tanimoto_set(plan),
        "stage1_residual_errors": tanimoto_set(stage1_errors),
        "official_errors": tanimoto_set(official_errors),
        "interpretation": (
            "Rivals with Tanimoto near 1 on residual errors are structural "
            "near-duplicates (stereoisomers or annotation-duplicate suspects); "
            "they are prime label-noise mass and cap every training route.  "
            "Compare the error distributions against the background "
            "distribution for the excess."
        ),
    }
    # Strata cross-tab for residual errors.
    strata_tanimoto: dict[str, dict[str, object]] = {}
    for stratum in sorted({
        str(p["source"]["selection_geometry_best_negative_relation"])
        for p in stage1_errors
    }):
        subset = [
            p for p in stage1_errors
            if str(p["source"]["selection_geometry_best_negative_relation"])
            == stratum
        ]
        strata_tanimoto[stratum] = tanimoto_set(subset)
    tanimoto_report["stage1_errors_by_stratum"] = strata_tanimoto

    # ---- Section 3: collision-energy alignment test ----
    def cosine_win(entry: dict[str, object], ce_filter: bool) -> float | None:
        qpos = row_position[int(query_row[int(entry["query"])])]
        qmz, qint = peaks[qpos]
        qce = float(energies[qpos])
        best_true, best_rival = -1.0, -1.0
        for rows, bucket in (
            (entry["pos_rows"], "true"), (entry["rival_rows"], "rival"),
        ):
            for row in rows:
                position = row_position[int(row)]
                if ce_filter:
                    cce = float(energies[position])
                    if np.isnan(qce) or np.isnan(cce):
                        continue
                    if abs(cce - qce) > CE_WINDOW_EV:
                        continue
                cmz, cint = peaks[position]
                value = float(cosine_similarity(qmz, qint, cmz, cint))
                if bucket == "true":
                    best_true = max(best_true, value)
                else:
                    best_rival = max(best_rival, value)
        if best_true < 0 or best_rival < 0:
            return None
        return 1.0 if best_true > best_rival else 0.0

    ce_results = {"unfiltered": [], "ce_filtered": [], "paired_delta": []}
    ce_unusable = 0
    for entry in stage1_errors:
        base = cosine_win(entry, ce_filter=False)
        filtered = cosine_win(entry, ce_filter=True)
        if base is None or filtered is None:
            ce_unusable += 1
            continue
        ce_results["unfiltered"].append(base)
        ce_results["ce_filtered"].append(filtered)
        ce_results["paired_delta"].append(filtered - base)
    ce_report = {
        "stage1_errors": len(stage1_errors),
        "paired_usable": len(ce_results["unfiltered"]),
        "unusable_after_filter": ce_unusable,
        "ce_window_ev": CE_WINDOW_EV,
        "win_rate_unfiltered": float(np.mean(ce_results["unfiltered"])),
        "win_rate_ce_filtered": float(np.mean(ce_results["ce_filtered"])),
        "mean_paired_gain": float(np.mean(ce_results["paired_delta"])),
        "queries_gained": int(np.sum(np.asarray(ce_results["paired_delta"]) > 0)),
        "queries_lost": int(np.sum(np.asarray(ce_results["paired_delta"]) < 0)),
    }

    # ---- Section 4: flip-distance of rank-2/3 residuals ----
    flip = {}
    for bucket, ranks in (("rank2", {2}), ("rank3", {3})):
        entries = [
            p for p in stage1_errors
            if int(p["source"]["stage1_rank"]) in ranks
        ]
        margins = []
        for entry in entries:
            source = entry["source"]
            try:
                margin = float(source["selection_geometry_positive_score"]) - float(
                    source["selection_geometry_best_negative_score"]
                )
            except (KeyError, TypeError, ValueError):
                continue
            margins.append(margin)
        array = np.asarray(margins, dtype=np.float64)
        flip[bucket] = {
            "queries": len(array),
            "median_margin": float(np.median(array)) if len(array) else None,
            "fraction_within_0.02": float(np.mean(np.abs(array) <= 0.02)) if len(array) else None,
            "fraction_within_0.05": float(np.mean(np.abs(array) <= 0.05)) if len(array) else None,
            "fraction_within_0.10": float(np.mean(np.abs(array) <= 0.10)) if len(array) else None,
        }

    report = {
        "status": "GLM_T1_PREFLIGHT_STRUCTURE_COMPLETE",
        "probe_version": PROBE_VERSION,
        "author": "GLM-5.3 via DeepSeek Harness",
        "inputs": {
            "graph": str(args.graph),
            "held_per_query": str(args.held_per_query),
            "data": str(args.data),
            "graph_sha256": sha256_file(args.graph),
            "held_per_query_sha256": sha256_file(args.held_per_query),
            "data_sha256": sha256_file(args.data),
        },
        "t1_outer_train_census": census,
        "tanimoto_structure": tanimoto_report,
        "collision_energy_test": ce_report,
        "flip_distance": flip,
        "claim_limit": (
            "The T1 census is a conservative lower bound: it excludes all "
            "fold-0 formulas because the stage-1 validation formula set is "
            "not local, and it counts graph structure, not qualified training "
            "events.  Tanimoto used the query's own SMILES versus the rival "
            "molecule's first row; stereoisomer pairs share near-identical "
            "fingerprints, so near-1 values are suspects, not adjudicated "
            "label errors.  CE test is paired within query on naive cosine "
            "against the proxy rival only.  Flip distances use the frozen "
            "selection geometry, not re-encoded Stage-1."
        ),
    }

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "t1_outer_train_census": {
            key: census[key] for key in (
                "train_queries_examined",
                "queries_with_any_same_formula_rival",
                "fraction_train_queries_with_relation",
                "events_if_capped_at_3_per_query",
                "fraction_rival_blocks_multi_spectrum",
            )
        },
        "tanimoto_stage1_errors": tanimoto_report["stage1_residual_errors"],
        "collision_energy_test": ce_report,
        "flip_distance": flip,
    }, indent=2))


if __name__ == "__main__":
    main()
